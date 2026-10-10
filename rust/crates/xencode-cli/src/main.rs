use std::io::{self, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use std::sync::Arc;
use xencode_analysis_rs::analyzer::CodeAnalyzer;
use xencode_analysis_rs::images::{analyze_image, is_image_path, ImageMeta};
use xencode_analysis_rs::issues::CodeIssue;
use xencode_analysis_rs::security::VulnerabilityScanner;
use xencode_analysis_rs::web::{fetch_url, FetchedPage};
use xencode_cache_rs::ResponseCache;
use xencode_config_rs::{SecretProvider, XencodeConfig};
use xencode_context_rs::doctor as doc;
use xencode_core_rs::{scan_workspace, ScanOptions};
use xencode_memory_rs::scoped::{MemoryScope, ScopedSharedMemoryStore, WorkerMemoryPolicy};
use xencode_memory_rs::ConversationMemory;
use xencode_models_rs::{find_llama_server, LlamaCppClient, LlamaCppOptions, OllamaClient};
use xencode_plugin_rs::{default_plugin_dir, PluginRegistry, PluginRuntime};
use xencode_providers_rs::{
    ChatMessage, ContentPart, EgressPolicy, ImageUrlPart, MessageContent, ProviderManager,
};
use xencode_server_rs::ws::AppState as ServerState;

/// What `xencode generate` emits.
#[derive(clap::ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum GenerateArtifact {
    /// Shell completion script for `--shell`.
    Completions,
    /// Roff man page for `xencode(1)`.
    Man,
}

/// Shells `xencode generate completions` supports.
#[derive(clap::ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum GenerateShell {
    Bash,
    Fish,
    Zsh,
    Powershell,
    Elvish,
}

/// Output format for analysis results
#[derive(clap::ValueEnum, Clone, Debug)]
enum OutputFormat {
    Text,
    Json,
}

/// How `xencode query` writes its answer. `text` prints the model's words as
/// they arrive, which is what someone reading a terminal wants. `ndjson` prints
/// one JSON object per line — see [`query_stream`] for the shapes — which is
/// what another program piping the output wants.
#[derive(clap::ValueEnum, Clone, Copy, PartialEq, Eq, Debug)]
enum QueryFormat {
    Text,
    Ndjson,
}

/// The line protocol behind `xencode query --format ndjson`.
///
/// Every line is one object carrying `"v": 1`. The version is written per line
/// rather than once at the top because a script that joins mid-stream (a pipe
/// opened on a partial run, a `tail -f`) must still be able to tell which
/// schema it is reading. A consumer that meets a higher `v` than 1 must stop
/// rather than guess; a consumer that meets an unknown `type` at a `v` it does
/// know must skip the line.
///
/// The one guarantee worth stating, because a script is built on it: the
/// `text` fields of every `token` line, concatenated in order, equal the
/// `answer` field of the terminating `done` line. That holds for a cached
/// answer too, which is why the cached path emits the answer as a `token` line
/// instead of only in `done`.
mod query_stream {
    use serde_json::Value;

    /// Bumped only when a line written by the previous version would be read
    /// wrongly, not when a field is added.
    pub const VERSION: u8 = 1;

    fn event(kind: &str) -> Value {
        serde_json::json!({ "v": VERSION, "type": kind })
    }

    /// Emitted before any model output, once the model and route are settled.
    /// `source` is the same word the metrics rows use: `local` or `cloud`.
    pub fn start(model: &str, provider: &str, source: &str, session: Option<&str>) -> Value {
        let mut line = event("start");
        line["model"] = model.into();
        line["provider"] = provider.into();
        line["source"] = source.into();
        line["session"] = match session {
            Some(id) => Value::String(id.to_string()),
            None => Value::Null,
        };
        line
    }

    /// A piece of the answer, as it arrived. The split is the network's, not
    /// the model's: pieces do not align with words.
    pub fn token(text: &str) -> Value {
        let mut line = event("token");
        line["text"] = text.into();
        line
    }

    /// The answer, plus how it was obtained. A route reports token counts only
    /// when its response carries them, and a llama.cpp server reports them only
    /// when its stream ends with a usage chunk; `tokens_generated` and
    /// `tokens_per_second` are null whenever nothing was reported — including
    /// on a cached answer, where no generation happened. `elapsed_ms` is always
    /// this command's own wall clock.
    pub fn done(
        answer: &str,
        cached: bool,
        elapsed_ms: u64,
        tokens_generated: Option<u64>,
        tokens_per_second: Option<f64>,
    ) -> Value {
        let mut line = event("done");
        line["answer"] = answer.into();
        line["cached"] = cached.into();
        line["elapsed_ms"] = elapsed_ms.into();
        line["tokens_generated"] = match tokens_generated {
            Some(n) => n.into(),
            None => Value::Null,
        };
        line["tokens_per_second"] = match tokens_per_second {
            Some(rate) => serde_json::json!((rate * 100.0).round() / 100.0),
            None => Value::Null,
        };
        line
    }

    /// The last line of a run that produced no answer. A failure before the
    /// model was dialed (a bad `--json-schema`, say) still ends this way, so a
    /// script can rely on exactly one terminating line per run.
    pub fn error(message: &str) -> Value {
        let mut line = event("error");
        line["message"] = message.into();
        line
    }
}

/// Xencode — AI development assistant (Rust core)
#[derive(Parser)]
#[command(name = "xencode", version, about, long_about = None)]
struct Cli {
    /// Dump the engine composition and resolved configuration as JSON (AF-3)
    #[arg(long)]
    dump_config: bool,

    /// Defaults to the TUI when omitted
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Scan a workspace and list all entries
    Scan {
        /// Path to scan (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Include hidden files and directories
        #[arg(long)]
        hidden: bool,

        /// Maximum directory depth to traverse
        #[arg(long)]
        max_depth: Option<usize>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Local model management (Ollama & llama.cpp)
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Response cache management
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// The session server's audit log
    Audit {
        #[command(subcommand)]
        action: AuditAction,
    },

    /// Known security advisories for the crates this project depends on
    Advisories {
        #[command(subcommand)]
        action: AdvisoryAction,
    },

    /// Supply-chain report: shell to the installed dependency checkers
    /// (cargo-shear, cargo-deny) and stream their findings. Report only — it
    /// never edits a manifest or auto-fixes a dependency.
    Deps {
        /// Project to check (default: the current directory)
        #[arg(long, default_value = ".")]
        path: PathBuf,

        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Send a query to a model
    Query {
        /// The prompt to send
        prompt: String,

        /// Model to use (overrides config default)
        #[arg(long, short)]
        model: Option<String>,

        /// Do not use cached responses
        #[arg(long)]
        no_cache: bool,

        /// Session ID to attach to
        #[arg(long)]
        session: Option<String>,

        /// llama.cpp sampling: temperature (e.g. 0.7)
        #[arg(long)]
        temperature: Option<f64>,

        /// llama.cpp sampling: top-k
        #[arg(long, name = "top-k")]
        top_k: Option<i32>,

        /// llama.cpp sampling: min-p (e.g. 0.05)
        #[arg(long, name = "min-p")]
        min_p: Option<f64>,

        /// llama.cpp sampling: mirostat mode (0, 1, or 2)
        #[arg(long)]
        mirostat: Option<i32>,

        /// llama.cpp sampling: seed, for output that can be produced again
        #[arg(long)]
        seed: Option<i64>,

        /// llama.cpp sampling: max generated tokens
        #[arg(long = "max-tokens")]
        max_tokens: Option<u32>,

        /// llama.cpp sampling: GBNF grammar file/string
        #[arg(long)]
        grammar: Option<String>,

        /// llama.cpp sampling: JSON schema for structured output
        #[arg(long = "json-schema")]
        json_schema: Option<String>,

        /// Image to send with the prompt (repeatable); needs a vision-capable model
        #[arg(long = "image", value_name = "PATH")]
        images: Vec<String>,

        /// How to write the answer: plain words, or one JSON event per line
        #[arg(long, default_value = "text")]
        format: QueryFormat,
    },

    /// Manage conversation memory
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },

    /// Manage background tasks (file-backed, survives this process)
    Tasks {
        #[command(subcommand)]
        action: TaskAction,
    },

    /// Manage git worktrees of the current repository
    Worktree {
        #[command(subcommand)]
        action: WorktreeAction,
    },

    /// Google Colab bridge: preflight, then up / status / down for a model
    /// server running on a Colab VM
    Colab {
        #[command(subcommand)]
        action: ColabAction,
    },

    /// Manage remote inference hosts reached over SSH: add, list, use, forget
    Remote {
        #[command(subcommand)]
        action: RemoteAction,
    },

    /// Manage and inspect registered computer backends (AF-4)
    Computers {
        #[command(subcommand)]
        action: Option<ComputersAction>,
        /// Output results as JSON
        #[arg(long, global = true)]
        json: bool,
    },

    /// Compete candidate implementations on isolated branches, verify each, and
    /// let a person pick one (AF-5)
    Compete {
        #[command(subcommand)]
        action: CompeteAction,
    },

    /// Evaluate merge conflicts with git merge-tree and land branches under a human approval gate (OR-5)
    Merge {
        #[command(subcommand)]
        action: MergeAction,
    },

    /// Read the team recipes this project keeps in `.xencode/teams`: the roles,
    /// who plays each one, and which checks gate it — and run one, but only under
    /// a name that approves it, and only where the Local-Only posture will allow
    /// the work to go (OR-9, OR-10, OR-13)
    Team {
        #[command(subcommand)]
        action: TeamAction,
    },

    /// The orchestrator's own control surface, run headless: what the fleet is,
    /// what it is doing, what it cost, and what a launch would be allowed to do —
    /// plus stopping a task, re-running one role, and handing this terminal to a
    /// vendor's own running session (OR-14)
    Orchestrator {
        #[command(subcommand)]
        action: OrchestratorAction,
    },

    /// Repository insights from the .xencode snapshot: broken imports,
    /// import cycles, hub files and orphans
    Advise {
        /// Only report findings whose file path contains this substring
        filter: Option<String>,

        /// Machine-readable output
        #[arg(long)]
        json: bool,

        /// Maximum findings to show (0 shows all)
        #[arg(long, default_value = "40")]
        limit: usize,
    },

    /// Start the collaboration server
    Server {
        /// Port to listen on
        #[arg(long, default_value = "8765")]
        port: u16,

        /// Address to bind (default: loopback only)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// TLS certificate in PEM form; requires --key
        #[arg(long)]
        cert: Option<PathBuf>,

        /// TLS private key in PEM form; requires --cert
        #[arg(long)]
        key: Option<PathBuf>,

        /// Audit log path; "none" disables (default: <state dir>/audit.jsonl)
        #[arg(long)]
        audit_path: Option<String>,

        /// Allow a non-loopback bind without TLS — tokens and activity then
        /// travel in clear text; read the warning before reaching for this
        #[arg(long)]
        allow_insecure_public: bool,
    },

    /// Analyze code for issues and vulnerabilities
    Analyze {
        /// Path to analyze (file or directory)
        path: std::path::PathBuf,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,

        /// Report async and concurrency hazards: blocking calls on the reactor
        /// thread, unbounded channels, and dropped task handles. Needs the
        /// `ast-grep` binary; without it the report says the check did not run
        /// rather than reporting nothing found.
        #[arg(long)]
        runtime: bool,
    },

    /// Fetch a web page and extract research-ready text
    Fetch {
        /// URL to fetch (http/https only)
        url: String,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Measure what the coding-agent CLIs on this machine actually do
    ///
    /// The `AR-1` interop probe. Every agent is launched headless on a
    /// read-only task and what came back is recorded, with each cell marked as
    /// observed or read from a help screen. A run that failed is recorded as a
    /// failure, never as an empty success. The task asks one question about
    /// one file and requests no changes, and an agent with no account stops at
    /// its auth check — which is itself an observation. But a signed-in agent
    /// makes real provider calls: measured 2026-10-02, the free-tier agents
    /// report zero or no cost while kiro-cli meters credits and kilo answers on
    /// a metered free model, so one probe run is a small spend, not zero.
    Interop {
        /// Probe only these agents (repeatable); default is every installed
        /// agent the operator has not stood down
        #[arg(long = "agent")]
        agents: Vec<String>,

        /// Per-agent wall-clock limit in seconds
        #[arg(long, default_value_t = 60)]
        timeout: u64,

        /// Where to write the JSON report; omit to print it
        #[arg(long)]
        out: Option<std::path::PathBuf>,

        /// Output format for the printed summary
        #[arg(long, default_value = "text")]
        format: OutputFormat,

        /// Run each agent this many times and compare. One run is a reading;
        /// two is a check, and a fact that differs between runs is reported
        /// rather than smoothed over.
        #[arg(long, default_value_t = 1)]
        repeat: u32,

        /// Run every selected agent at the same time instead of one after
        /// another, and report what the overlap saved. Cannot be combined
        /// with `--repeat`, which needs sequential runs to compare.
        #[arg(long, conflicts_with = "repeat")]
        fan_out: bool,

        /// Report, read-only, which agents look configured on this machine, and
        /// what to run if one is not. Starts no login and reads no credential.
        #[arg(long)]
        check_auth: bool,

        /// Keep each agent's whole run on disk, in `<dir>/<agent>/capture/`:
        /// `raw.jsonl` (every line the vendor printed, unredacted),
        /// `normalized.jsonl` (the common events, each naming its raw line) and
        /// `metadata.json` (what was run and what it said it cost). Written
        /// `0600` and only when you ask. Costs no extra run — it keeps what the
        /// probe already received.
        #[arg(long)]
        capture_dir: Option<std::path::PathBuf>,

        /// Print the trace view of a capture written earlier and probe nothing.
        /// Takes one capture (`../captures/opencode`) or the whole root
        /// (`../captures`) to render every vendor in the same view.
        #[arg(long, value_name = "CAPTURE", conflicts_with = "check_auth")]
        trace: Option<std::path::PathBuf>,
    },

    /// Find this repository's build and test commands, run them, and record
    /// only the ones that actually worked
    Anchor {
        /// Path to the repository (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Per-command wall-clock limit in seconds. A command that runs out of
        /// time is recorded as unverified, never as a pass.
        #[arg(long, default_value_t = 900)]
        timeout: u64,

        /// List what was found without running anything
        #[arg(long)]
        dry_run: bool,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Run the project's own toolchain checks, and report structured evidence
    Toolchain {
        /// What to run: lint, fix, fmt, or shear
        action: String,

        /// Allow `fix` on a tree with uncommitted edits
        #[arg(long)]
        allow_dirty: bool,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Write one bug report: configuration, secrets, disk, providers, models,
    /// MCP servers and the Colab bridge. Flags narrow it to one part.
    Doctor {
        /// Machine environment facts
        #[arg(long)]
        env: bool,

        /// Dependency health: outdated list plus advisory state
        #[arg(long)]
        deps: bool,

        /// Self-debug slice: index, git, providers, MCP servers, metrics, cache
        #[arg(long)]
        selfcheck: bool,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Name sessions, resolve them, and export redacted transcripts
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },

    /// Start the floating badge that shows what running xencode sessions are
    /// doing and when one needs you
    Badge,

    /// Run xencode as an agent for an editor that speaks the Agent Client
    /// Protocol, such as Zed. The editor starts it and talks to it over
    /// standard input and output; each session works through its folder's
    /// engine, as the terminal app does. In Zed's settings:
    /// "agent_servers": {"xencode": {"type": "custom", "command": "xencode", "args": ["acp"]}}
    Acp,

    /// Run the agent work for one project in its own process, for the
    /// terminal app and other windows to connect to. It exits on its own once
    /// no window is connected and nothing is running.
    Engine {
        /// The project folder (default: the current folder)
        #[arg(long)]
        project: Option<PathBuf>,

        /// Seconds a question, approval or review may wait while no window
        /// is connected; then the question is withdrawn and its task
        /// stopped, the approval denied, or the review completed with its
        /// changes kept
        #[arg(long, default_value_t = xencode_tui_rs::engine::server::WAIT_LIMIT.as_secs())]
        wait_limit: u64,
    },

    /// Where xencode keeps its own files: settings, session records, cache and
    /// downloaded models, and whether they are still in `~/.xencode`
    Paths {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Move the files in `~/.xencode` to the four directories they belong in.
    /// Nothing is overwritten and the old directory is only removed once empty.
    Migrate {
        /// Print what would move and change nothing
        #[arg(long)]
        dry_run: bool,
    },

    /// Run the machine-checkable checklist: test, lint, fmt — each verified, none graded
    Verify {
        /// Skip these checks (repeatable); skipped is reported, never passed
        #[arg(long)]
        skip: Vec<String>,

        /// Wall-clock ceiling in seconds for the test slot
        #[arg(long, default_value_t = 1800)]
        timeout: u64,

        /// Session ID to file checks under in the verification ledger (defaults to active session or 'cli')
        #[arg(long)]
        session: Option<String>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Report environment keys read in code against the templates that document them
    Envcheck {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// List installed agents with versions and install provenance, inspect worker health (AR-8), or manage continuation packages (AR-7)
    Agents {
        /// Verify each roster claim against the agent's live --help
        #[arg(long)]
        contract: bool,

        /// Report worker health (installed, version, authenticated, responsive, rate-limited) (AR-8)
        #[arg(long)]
        health: bool,

        /// Specific agent to inspect or resume with (e.g. claude, agy, cursor-agent)
        #[arg(long)]
        agent: Option<String>,

        /// Build a worker continuation package from workspace diff and test runs (AR-7)
        #[arg(long)]
        build_package: Option<String>,

        /// Path to inspect or load a worker continuation package (AR-7)
        #[arg(long)]
        package: Option<PathBuf>,

        /// Resume a task from a continuation package using the specified agent (AR-7)
        #[arg(long)]
        resume: bool,

        /// Test commands to execute for package verification (defaults to 'cargo test')
        #[arg(long = "test-cmd")]
        test_cmds: Vec<String>,

        /// Route a task on capabilities the probe confirmed here, printing how each
        /// number behind the choice was known
        #[arg(long = "route")]
        route_task: Option<String>,

        /// Capabilities required for the routed task (e.g. stream, acp, mcp, resume, daemon, approval) (OR-6)
        #[arg(long = "require-cap")]
        require_caps: Vec<String>,

        /// Maximum cost ceiling allowed for the routed task; applied only where a
        /// price was measured, and reported as unchecked where it was not
        #[arg(long = "max-cost")]
        max_cost: Option<f64>,

        /// Re-dispatch a killed or failed worker's task onto another agent (OR-7)
        #[arg(long = "redispatch")]
        redispatch_task: Option<String>,

        /// Replacement agent to resume the re-dispatched task (OR-7)
        #[arg(long = "replacement-agent")]
        replacement_agent: Option<String>,

        /// Process stop reason for the killed worker (e.g. signal:9, exit:137, timeout:300) (OR-7)
        #[arg(long = "stop-reason")]
        stop_reason: Option<String>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Rank files by churn times size with bus factor and owners
    Hotspots {
        /// How many files to list
        #[arg(long, default_value_t = 10)]
        limit: usize,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// What a change to one file affects: the crates that depend on its crate,
    /// the files that link it, and the files its history is coupled to
    Impact {
        /// The file to analyse, by path or by its tail
        file: String,

        /// Answer from rust-analyzer's semantic index (SCIP) instead of `use`
        /// paths: every file that refers to a symbol this file defines. Builds the
        /// index first when it is missing or stale, which takes minutes and runs
        /// the project's build scripts and procedural macros, as `cargo build` does
        #[arg(long)]
        semantic: bool,

        /// With --semantic: only files that refer to this symbol of the file
        #[arg(long, requires = "semantic")]
        symbol: Option<String>,

        /// How many entries to list in each section
        #[arg(long, default_value_t = 15)]
        limit: usize,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// What deleting one file would cost: the links it holds up, and the modules
    /// that become dead code the moment it is taken out
    Removal {
        /// The file to imagine removing, by path or by its tail
        file: String,

        /// How many entries to list in each section
        #[arg(long, default_value_t = 15)]
        limit: usize,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Print shell completions or the man page; both are generated from the
    /// clap definition, never written by hand
    Generate {
        /// What to emit: completions or man
        #[arg(value_enum)]
        artifact: GenerateArtifact,

        /// Shell for completions (ignored for man)
        #[arg(long, value_enum, default_value = "bash")]
        shell: GenerateShell,
    },

    /// Find code whose tests cannot tell right from wrong
    Mutants {
        /// Only mutants in the diff against this ref
        #[arg(long)]
        diff: Option<String>,

        /// Wall-clock limit per mutant, in seconds
        #[arg(long, default_value_t = 60)]
        timeout: u64,

        /// Judge a proposed repair instead of running anything
        #[arg(long)]
        check_repair: Option<std::path::PathBuf>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Report which lines this diff added were never executed
    Cov {
        /// Compare against this ref instead of the working tree
        #[arg(long)]
        base: Option<String>,

        /// Run this test command instead of the repository's own verified one
        #[arg(long)]
        test: Option<String>,

        /// List only the file and line numbers
        #[arg(long)]
        show_missing_lines: bool,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Measure the hot paths against a stored baseline, and refuse the verdict
    /// when the run is too noisy to support one
    Perf {
        #[command(subcommand)]
        action: PerfAction,
    },

    /// Where the prices a cost report uses come from, and reading them again
    Prices {
        #[command(subcommand)]
        action: Option<PriceAction>,
    },

    /// Run the tests, and never call a test that only passed on retry a pass
    Test {
        /// Only these packages (repeatable)
        #[arg(long = "package")]
        packages: Vec<String>,

        /// Retries allowed per failing test. Kept at 0 by default: a retry-pass
        /// is not a pass.
        #[arg(long, default_value_t = 0)]
        retries: u32,

        /// Run each test this many times, to surface flakes and order dependence
        #[arg(long, default_value_t = 0)]
        stress_count: u32,

        /// Wall-clock ceiling in seconds
        #[arg(long, default_value_t = 1800)]
        timeout: u64,

        /// Classify one failing test against the clean base tree instead of
        /// running the suite: PRE_EXISTING_FAILURE, INTRODUCED, or FLAKY
        #[arg(long)]
        isolate: Option<String>,

        /// The ref the base tree is taken at for --isolate
        #[arg(long, default_value = "HEAD")]
        base: String,

        /// Runs per side for --isolate; a pass on any run means flaky
        #[arg(long, default_value_t = 3)]
        repeat: u32,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,

        /// Session ID to file checks under in the verification ledger (defaults to active session or 'cli')
        #[arg(long)]
        session: Option<String>,
    },

    /// Draft the release notes from the commits since the last release and the
    /// changelog block this project keeps, and report where the two disagree
    ReleaseNotes {
        /// Start the range here instead of at the newest tag. An empty value
        /// means no lower bound: every commit reachable from --to.
        #[arg(long)]
        from: Option<String>,

        /// End the range here (default: HEAD)
        #[arg(long)]
        to: Option<String>,

        /// Label the draft heading with this version instead of `[Unreleased]`
        #[arg(long)]
        release: Option<String>,

        /// Write the draft here instead of printing it. A file that already
        /// exists is not replaced unless --force says so.
        #[arg(long)]
        out: Option<PathBuf>,

        /// Replace the file named by --out even if it is already there
        #[arg(long)]
        force: bool,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Review the diff between a base branch and HEAD, file by file
    Review {
        /// Base branch, tag, commit — or HEAD for uncommitted changes. Defaults
        /// to origin/HEAD, init.defaultBranch, or 'main'.
        #[arg(long)]
        base: Option<String>,

        /// Also print the result envelope for this session: the checks it ran,
        /// read from the run ledger, and the status they support
        #[arg(long)]
        session: Option<String>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Run a recorded session again from the bytes it was made of
    Replay {
        /// Which run: its full id, or enough of the start to be unique. Omit it
        /// with --list to see what has been recorded.
        run_id: Option<String>,

        /// List recorded runs, newest first
        #[arg(long)]
        list: bool,

        /// Let the replay's tool calls really run. Without this the permission
        /// gate stays in charge, so a call it would have asked a person about
        /// comes back denied.
        #[arg(long)]
        run_tools: bool,

        /// The tree the replay's tool calls work against (default: this one)
        #[arg(long)]
        tool_root: Option<PathBuf>,

        /// Where to write tool_calls.jsonl and the replay's own recording
        /// (default: <project>/.xencode/cache/replays/<run id>)
        #[arg(long)]
        out: Option<PathBuf>,
    },

    /// Which runs happened, what each asked a person, and the commit trailer
    /// naming it. Reads `.xencode/cache/runs.jsonl` only, so it works with
    /// every model server down.
    Runs {
        #[command(subcommand)]
        action: Option<RunsAction>,
    },

    /// Run an agent turn from the command line, in the foreground or detached
    /// so it survives the terminal. A detached run persists every completed
    /// round under `.xencode/cache/detached/<run-id>/`, so a kill is resumed
    /// with `--resume` instead of restarted, and stops on round, wall-clock
    /// and cost caps as well as the model finishing.
    Run {
        /// The task, in plain words. Foreground unless `--detach`.
        prompt: Option<String>,

        /// Start the run in the background and print its id. The terminal
        /// may go away; the run keeps going under its caps.
        #[arg(long)]
        detach: bool,

        /// Continue a crashed run from its last completed round. Refuses a
        /// run that finished, was stopped, or is still going.
        #[arg(long)]
        resume: Option<String>,

        /// List detached runs and what each is doing.
        #[arg(long)]
        list: bool,

        /// Show one run: its spec, its status, its rounds and its exit.
        #[arg(long)]
        show: Option<String>,

        /// Print the tail of one run's log.
        #[arg(long)]
        log: Option<String>,

        /// Ask one running run to stop. Reads as stopped, not crashed.
        #[arg(long)]
        stop: Option<String>,

        /// Run with this model instead of the configured default.
        #[arg(long)]
        model: Option<String>,

        /// The tree the run works in (default: this project).
        #[arg(long)]
        tool_root: Option<PathBuf>,

        /// Stop after this many completed rounds.
        #[arg(long)]
        max_rounds: Option<u32>,

        /// Stop after this many minutes of wall-clock time. System time, so
        /// a suspended laptop counts — a cap that slept through suspend
        /// would be a way past it.
        #[arg(long)]
        max_minutes: Option<f64>,

        /// Stop after spending this many dollars. Needs a model
        /// `pricing.json` (or the fetched listing) names and a route that
        /// reports token counts; without both the run is refused, because a
        /// cap that cannot count cannot stop.
        #[arg(long)]
        max_cost: Option<f64>,

        /// Pre-approve shell commands. Without this a detached run has
        /// nobody to ask, so shell calls are refused where they stand.
        #[arg(long)]
        allow_shell: bool,

        /// How many log lines `run --log` prints.
        #[arg(long, default_value_t = 40)]
        tail: usize,

        /// Where an Ollama server is, for a model id with no prefix.
        #[arg(long)]
        ollama_url: Option<String>,

        /// Where a llama.cpp server is, for a `llamacpp:` model id.
        #[arg(long)]
        llamacpp_url: Option<String>,

        /// The detached worker itself. Started by `run --detach`, never typed.
        #[arg(long, hide = true)]
        child: Option<String>,

        /// Where the detached worker's run lives. Passed by the starting
        /// parent, because the worker's own working directory is the run's
        /// tree rather than the project.
        #[arg(long, hide = true)]
        xencode_dir: Option<PathBuf>,
    },

    /// Score the agent on defects that were seeded on purpose
    Eval {
        #[command(subcommand)]
        action: EvalAction,
    },

    /// Manage plugins
    Plugin {
        #[command(subcommand)]
        action: PluginAction,
    },

    /// Let another program drive xencode's tools over Model Context Protocol
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },

    /// llama.cpp server management (status/start/stop/load/unload)
    Llamacpp {
        #[command(subcommand)]
        action: LlamacppAction,
    },

    /// What this machine can serve, read from the machine
    Hw {
        #[command(subcommand)]
        action: HwAction,
    },

    /// How fast this repository's history is to ask about, and how to speed it up
    History {
        #[command(subcommand)]
        action: HistoryAction,
    },

    /// Write the files a project xencode has never seen is missing
    Bootstrap {
        /// The project to write into (defaults to the current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Report what would be written and create nothing
        #[arg(long)]
        check: bool,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Launch the Terminal User Interface
    Tui {
        /// Run the agent work inside this terminal instead of in the
        /// project's engine process
        #[arg(long)]
        in_process: bool,
    },
}

#[derive(Subcommand)]
enum EvalAction {
    /// List the shapes that can be run, and every run recorded so far
    List,
    /// Seed the defects, let the agent work on them, and grade what it changed
    Run {
        /// Which defect to seed (repeatable; default: all eight)
        #[arg(short = 'c', long = "case")]
        cases: Vec<String>,

        /// Model to score, in the form the router understands: a plain name for
        /// Ollama, `llamacpp:<name>`, or `remote:<name>` (default: the config's)
        #[arg(short = 'm', long)]
        model: Option<String>,

        /// How many times to run each defect. Eight shapes three times is
        /// twenty-four cases, which is the smallest sample worth reading.
        #[arg(long, default_value_t = 1)]
        repeats: usize,

        /// Tool rounds a single case gets before the loop has to answer
        #[arg(long)]
        max_rounds: Option<usize>,

        /// Let the model really run shell commands. Without this a `run_command`
        /// is refused and the refusal is recorded like any other call.
        #[arg(long)]
        allow_shell: bool,

        /// Where the seeded repositories and their diffs go
        /// (default: a stamped directory under the system temporary directory)
        #[arg(long)]
        out: Option<PathBuf>,

        /// Where an Ollama server is, for a model id with no prefix
        #[arg(long)]
        ollama_url: Option<String>,

        /// Where a llama.cpp server is, for a `llamacpp:` model id
        #[arg(long)]
        llamacpp_url: Option<String>,

        /// Seconds one model request may take before the case gives up on it
        #[arg(long, default_value_t = 120)]
        timeout: u64,

        /// How long one answer may be. Default 1024; a small model that will not
        /// stop talking otherwise holds a case for minutes. `0` leaves the limit
        /// to the server.
        #[arg(long, default_value_t = 1024)]
        max_tokens: u32,

        /// After every verdict is in, ask a model to rank the attempts that came
        /// close. Two more requests per run, and no verdict changes.
        #[arg(long)]
        judge: bool,

        /// Rank with a different model than the one under test, which is the only
        /// thing here that does anything about a judge favouring its own style.
        /// Defaults to the model being scored, and says so in the report.
        #[arg(long)]
        judge_model: Option<String>,
    },
}

#[derive(Subcommand)]
enum PluginAction {
    /// List installed plugins, whether each one loads, and what it contributes
    List,
    /// Install a plugin from a git URL or a local path
    Install {
        /// A git URL to clone (`https://…`, `git@…:…`, `file:///…`) or a path to
        /// a plugin directory or manifest file
        source: String,
        /// Install this branch, tag or commit instead of the repository's
        /// default branch. What is installed is pinned to the one commit the
        /// name resolved to, and `update` follows this same name.
        #[arg(long)]
        rev: Option<String>,
    },
    /// Fetch a plugin's own repository again and show what changed before it is
    /// applied
    Update {
        /// Name of an installed plugin
        name: String,
        /// Move to this branch, tag or commit rather than the one the plugin was
        /// installed at
        #[arg(long)]
        rev: Option<String>,
        /// Apply the fetched version even though it changes what the plugin puts
        /// in front of the agent. Without this, such an update is only shown.
        #[arg(long)]
        yes: bool,
    },
    /// Remove a plugin by name
    Remove {
        /// Name of the plugin
        name: String,
    },
}

#[derive(Subcommand)]
enum McpAction {
    /// Serve xencode's tools as an MCP server on standard input and output
    ///
    /// For a program that cannot answer an approval prompt — an editor, a
    /// script, another agent. So this starts read-only: `read_file`, `list_dir`
    /// and `search_files` run, and `write_file`, `edit_file` and `run_command`
    /// are refused with the flag that would have permitted them. A path given
    /// to a tool stays inside `--workspace` whatever the flags; a command the
    /// caller is allowed to run is not checked that way, so `--allow
    /// run_command` hands over a shell.
    Serve {
        /// The directory the tools work on; a `path` or `cwd` argument that
        /// leaves it is refused
        #[arg(long, default_value = ".")]
        workspace: PathBuf,

        /// Permit this one tool to run despite the read-only default. Repeat
        /// it per tool; the name must be one of the six xencode publishes, so a
        /// typo is reported instead of doing nothing.
        #[arg(long = "allow")]
        allow: Vec<String>,

        /// Also publish the team tools a lead agent uses to direct worker
        /// agents on this project: team_agents, team_start, team_status,
        /// team_result, team_message, team_stop, team_merge. They act through
        /// the project's engine, starting it if none runs.
        #[arg(long)]
        team: bool,
    },
}
#[derive(Subcommand)]
enum ConfigAction {
    /// Display current configuration
    Show {
        /// Include full engine composition summary
        #[arg(long)]
        composition: bool,
    },
    /// Dump the engine composition and resolved configuration as JSON (AF-3)
    Dump,
    /// Set a configuration value
    Set {
        /// Configuration key (e.g., default_model, ollama_url). A key naming a
        /// provider credential — `openai_api_key`, `openrouter_api_key`,
        /// `google_gemini_api_key`, `qwen_api_key`, `remote_key`,
        /// `nvidia_api_key`, `brave_api_key`, `tavily_api_key` — is stored and
        /// never printed back.
        key: String,
        /// Value to set. A leading hyphen is allowed because the value people
        /// set most often is `llama_cpp_args`, which is a server command line,
        /// and the line `xencode hw probe` hands them to paste starts with a
        /// flag. For a credential key the value may instead be
        /// `command:<program> <args>`: only that reference is kept, and the
        /// program supplies the secret when it is needed.
        #[arg(allow_hyphen_values = true)]
        value: String,
        /// Validate and report the change without writing config.json
        #[arg(long)]
        dry_run: bool,
    },
    /// Reset configuration to defaults
    Reset,
}

#[derive(Subcommand)]
enum ColabAction {
    /// Verify the google-colab-cli bridge is usable before bringing a VM up
    Preflight {
        /// Generate the ed25519 key pair into the xencode config dir if absent
        #[arg(long)]
        generate_key: bool,
    },
    /// Bring a Colab VM up: create the session, install the runtime, and hold
    /// an SSH forward so the VM's OpenAI endpoint appears on the laptop
    Up {
        /// Session name (defaults to config colab.session)
        #[arg(long)]
        session: Option<String>,
        /// GPU accelerator (T4, L4, G4, H100, A100) for colab new
        #[arg(long)]
        gpu: Option<String>,
        /// Inference runtime on the VM: llama.cpp (pinned llama-server + GGUF,
        /// heavier install, one-shot) or ollama (tags flow into the model picker)
        #[arg(long)]
        runtime: Option<String>,
        /// Model repo/tag installed on the VM (defaults to config colab.model)
        #[arg(long)]
        model: Option<String>,
        /// Weights source: hf (llama.cpp only; drive/gcs are refused with a fix)
        #[arg(long)]
        weights: Option<String>,
        /// GGUF quantization fragment to serve (llama.cpp only, e.g. Q4_K_M;
        /// defaults to config colab.quant)
        #[arg(long)]
        quant: Option<String>,
        /// Local port the SSH forward exposes (defaults to config / 18000)
        #[arg(long)]
        local_port: Option<u16>,
        /// VM-side port the runtime binds (0 = runtime-native: llama.cpp 18080,
        /// ollama 11434; defaults to config)
        #[arg(long)]
        remote_port: Option<u16>,
        /// Rebuild a broken bridge from the recorded colab.json (re-spawn the
        /// forward, or re-create the VM if it was reaped) instead of a full up
        #[arg(long)]
        reconnect: bool,
        /// Report what a successful bring-up would start and would write to
        /// config.json, without touching the VM, the bridge or the config
        #[arg(long)]
        dry_run: bool,
    },
    /// Report the Colab bridge state: forward pid, `colab sessions`, and a
    /// /v1/models probe on the forward
    Status,
    /// Tear the Colab bridge down: kill the forward, `colab stop`, clear state
    Down,
}

#[derive(Subcommand)]
enum RemoteAction {
    /// Record a remote host profile
    Add {
        /// Profile name (up to 32 characters: a-z, 0-9, _, -, .)
        #[arg(allow_hyphen_values = true)]
        name: String,
        /// Host destination: [user@]host[:port] or ~/.ssh/config alias
        #[arg(allow_hyphen_values = true)]
        host: String,
        /// Inference runtime: llama.cpp or ollama
        #[arg(long, default_value = "llama.cpp")]
        runtime: String,
        /// Model repo/tag to serve on the remote machine
        #[arg(long)]
        model: Option<String>,
        /// SSH port (defaults to 22 or the port parsed from host)
        #[arg(long)]
        port: Option<u16>,
        /// Local port the SSH forward listens on (defaults to 18100)
        #[arg(long, default_value_t = xencode_config_rs::remotes::DEFAULT_LOCAL_PORT)]
        local_port: u16,
        /// Remote port the inference runtime binds inside the machine (0 = runtime default)
        #[arg(long, default_value_t = 0)]
        remote_port: u16,
        /// Overwrite an existing profile with this name
        #[arg(long)]
        force: bool,
    },
    /// List recorded remote host profiles
    List,
    /// Select the active remote host profile used by default
    Use {
        /// Profile name to set as active
        name: String,
    },
    /// Remove a recorded remote host profile
    Forget {
        /// Profile name to remove
        name: String,
    },
    /// Show details of a remote host profile (or the active profile if omitted)
    Show {
        /// Profile name to display (defaults to the active profile)
        name: Option<String>,
    },
}

#[derive(Subcommand, Debug, Clone)]
enum ComputersAction {
    /// List all registered computer backends (default)
    List,
    /// Show details of a specific computer backend
    Show {
        /// Computer backend identifier (colab, ssh, docker)
        name: String,
    },
    /// Set the active computer backend
    Use {
        /// Computer backend identifier (colab, ssh, docker)
        name: String,
    },
    /// Probe connectivity to a computer backend
    Probe {
        /// Computer backend identifier (colab, ssh, docker)
        name: String,
    },
}

#[derive(Subcommand)]
enum TeamAction {
    /// List the team recipes this project keeps, and say what each one is
    List {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Print one recipe's roles, workers, gates and capacity as it was written
    Show {
        /// The recipe's `name`, as written in the file
        name: String,
    },

    /// Compile one recipe into the task graph the scheduler would run, and show
    /// the order, the critical path, what limits it, and what a previous run of
    /// the same recipe took. Nothing is launched and no check is run.
    Plan {
        /// The recipe's `name`, as written in the file
        name: String,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Run one recipe's roles as real tasks through the scheduler. Without
    /// `--approved-by` this prints the plan and launches nothing, and writes
    /// nothing either. With it, the same plan prints first and then the roles
    /// run, and what the run took is recorded so the next plan can quote it.
    Run {
        /// The recipe's `name`, as written in the file
        name: String,

        /// Your name, for the record. Required to launch anything: a team that
        /// runs with no name on it has nobody who agreed to it.
        #[arg(long = "approved-by")]
        approved_by: Option<String>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Let a worker agent sign in with your own login instead of an API key.
    /// Prints what that vendor's terms say about it and asks for `yes`
    LoginOptin {
        /// The agent: claude-code, codex, gemini or antigravity
        agent: String,
    },

    /// Remove the worktrees of worker agents that are not running (stopped,
    /// failed, or left by an engine that ended). A worktree with changes
    /// nobody committed is kept and named
    Clean,
}

/// The control surface `OR-14` puts beside the TUI's `/orchestrator` mode.
///
/// Every verb here reads or acts on state that already exists — the roster, the
/// posture in the config, the task registry, the recipes, the recorded runs, the
/// detached runs, the metrics and the price table — because the mode is a way of
/// working with the same state, not a second copy of it. Two consequences follow
/// for the wording, and both are the item's own done-when:
///
/// * A reading that has no data says so and names the directory it looked in,
///   rather than printing an empty table that would read as "checked and clear".
/// * `attach` runs only a command the vendor's own help documents as taking over
///   a session that is already running. It never resumes a saved conversation
///   and calls that a handover, and it never chooses a session for you.
#[derive(Subcommand)]
enum OrchestratorAction {
    /// What way of working this is, where work is allowed to go, and how much of
    /// the fleet is on record right now
    Status {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// The agents xencode has a roster row for: whether each is installed,
    /// whether the posture refuses work handed to it, and whether it can be
    /// attached to
    Agents {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// The two lists of processes xencode started here and still knows about: the
    /// background task registry, and the detached runs of its own agent loop
    Tasks {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// One recipe's dependency graph as the scheduler sees it: the waves, what
    /// each role waits on, the critical path, the bottleneck, and where each role
    /// ended the last time this exact recipe was recorded. With no recipe named,
    /// one row per run this project has recorded. Nothing is launched.
    Graph {
        /// The recipe's `name`, as written in the file
        recipe: Option<String>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Ask the planner to split one task into units, then score that split against
    /// what this workspace's own build requires and against the flat baseline: one
    /// node per file, no order at all. A split that does not beat the baseline is
    /// refused here and never reaches the scheduler, because a graph that orders no
    /// more than a file list is two workers editing one file at once for no reason
    /// anybody agreed to. Nothing is launched.
    Split {
        /// The change to split, in your own words. The planner sees this and the
        /// list of crates, and nothing else — so it cannot copy the answer.
        #[arg(long, conflicts_with = "commit")]
        task: Option<String>,

        /// Split the change a commit here already made: its message is the task and
        /// the files it touched are what the split is scored against. This is the
        /// honest measurement, since neither the task nor the file set came from the
        /// planner.
        #[arg(long)]
        commit: Option<String>,

        /// The files the change touches, when no commit is named. Repeat once per
        /// file. Without this or `--commit`, the file set is the split's own, and
        /// only the order half of the score means anything.
        #[arg(long = "path")]
        paths: Vec<String>,

        /// Score the split in this file instead of asking a model: the JSON the
        /// planner is asked for, as it came back. Needs no server and no spend.
        #[arg(long)]
        answer: Option<std::path::PathBuf>,

        /// The command that decides a unit is done, handed to every unit so a
        /// planner cannot pick an easier test than the baseline is graded by
        #[arg(long = "verify", default_value = "cargo check --offline --workspace")]
        verification: String,

        /// Model to ask. Defaults to this project's configured model.
        #[arg(long)]
        model: Option<String>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// The newest lines of a run's own log, naming the file they were read from.
    /// A detached run keeps a log; a recorded team run keeps timings and exit
    /// statuses and never captured its children's output, and says which of the
    /// two it is rather than printing an empty block.
    Logs {
        /// A detached run id or prefix, or a recorded team run id. Without it,
        /// the runs that have anything to read are listed and nothing is chosen
        /// for you.
        run: Option<String>,

        /// How many lines to read from the end
        #[arg(long, default_value = "20")]
        lines: usize,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// What a launch of an agent would be allowed to do under this project's
    /// approval mode — built by the same function a launch uses — beside the
    /// approvals this project's records say a person actually answered
    Permissions {
        /// The agent to build the launch line for. Without it every agent on the
        /// roster is shown, which is the honest width of the answer.
        agent: Option<String>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// What the model calls recorded here cost, priced only where a price is
    /// known, beside what the recorded team runs drew from the wall
    Costs {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// One thing in full, with the file each figure came from: a background task
    /// by its registry id, a detached run, a recorded team run, or a recipe by
    /// its name
    Inspect {
        /// A task id, a detached run id or prefix, a team run id, or a recipe name
        target: String,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Re-run one role of a recipe as a real `sh -c` child of this process and
    /// report what it did. Your name is required, because a second run of
    /// somebody else's work is a decision and not a retry button. The roles it
    /// waits on are not run, and the output says which ones were skipped.
    Retry {
        /// The recipe's `name`, as written in the file
        recipe: String,

        /// The role inside that recipe to run again
        role: String,

        /// Your name, for the record
        #[arg(long = "approved-by")]
        approved_by: Option<String>,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Stop something xencode started: a background task by its registry id, or
    /// a detached run by its id. This signals one process to stop, so it is only
    /// ever pointed at a task this machine recorded launching.
    Stop {
        /// A task id, or a detached run id or prefix
        target: String,
    },

    /// Hand this terminal to one of an agent's own running sessions, and take it
    /// back when the vendor's command ends. xencode prints nothing while it runs:
    /// the vendor's program has the keyboard, and this command waits for it to let
    /// go. Needs a real terminal, because that is the thing being handed over.
    Attach {
        /// The agent to attach to, by its roster name
        agent: String,

        /// The session or server address its own listing command prints. Omit it
        /// to be told what to look up, and to get that command; xencode does not
        /// pick a session for you.
        target: Option<String>,
    },
}

#[derive(Subcommand)]
enum CompeteAction {
    /// Build each candidate arm in its own worktree and branch, run the
    /// verification checklist on every one of them, and print the table.
    Run {
        /// The question the candidate implementations compete on
        prompt: String,

        /// A candidate arm as `id` or `id=label`. Give it twice for two arms,
        /// or three times for three. Omit it to compete `arm-a` against `arm-b`.
        #[arg(long = "arm", value_name = "ID[=LABEL]")]
        arms: Vec<String>,

        /// Write content into a file inside one arm's worktree: the arm id, the
        /// path relative to that worktree, then the file's full text. Repeat it
        /// once per file per arm.
        #[arg(long = "edit", num_args = 3, value_names = ["ARM", "PATH", "CONTENT"])]
        edits: Vec<String>,

        /// Run a shell command inside one arm's worktree after its edits: the
        /// arm id, then the command.
        #[arg(long = "command", num_args = 2, value_names = ["ARM", "CMD"])]
        commands: Vec<String>,

        /// Skip a check by name (repeatable): test, lint, or fmt. A skipped
        /// check is reported as skipped, never as passed.
        #[arg(long)]
        skip: Vec<String>,

        /// Wall-clock ceiling in seconds for each check
        #[arg(long, default_value_t = 1800)]
        timeout: u64,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// List recorded competing runs, newest first
    List {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Re-print the verification table of a recorded run, from its saved report
    Show {
        /// Which run to print
        run_id: String,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Switch the repository onto one arm's branch, leaving every other
    /// candidate branch and all evidence files on disk untouched.
    Pick {
        /// Which run
        run_id: String,

        /// Which arm of it to take
        arm_id: String,
    },
}

#[derive(Debug, Subcommand)]
enum MergeAction {
    /// Speculatively precheck a candidate branch against a base branch using git merge-tree
    Precheck {
        /// Candidate branch to evaluate
        branch: String,
        /// Target base branch
        #[arg(long, default_value = "main")]
        base: String,
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Build a multi-branch merge plan with conflict prechecks and worker checks
    Plan {
        /// Candidate branches to evaluate
        #[arg(long = "branch", required = true)]
        branches: Vec<String>,
        /// Target base branch
        #[arg(long, default_value = "main")]
        base: String,
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Land branches into base branch guarded by a named human decision
    Land {
        /// Candidate branches to merge
        #[arg(long = "branch", required = true)]
        branches: Vec<String>,
        /// Target base branch
        #[arg(long, default_value = "main")]
        base: String,
        /// Full name of the human approving the merge (required gate)
        #[arg(long = "approved-by")]
        approved_by: String,
        /// Post-integration test commands to re-run on the integrated tree
        #[arg(long = "test-cmd")]
        test_cmds: Vec<String>,
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Block a branch from landing, over a review or a verification outcome
    Veto {
        /// Candidate branch to block
        branch: String,
        /// What kind of outcome the block comes from: review or verification
        #[arg(long, default_value = "review")]
        source: String,
        /// Who or what raised it
        #[arg(long, default_value = "human")]
        raised_by: String,
        /// The worker being blocked; defaults to the branch's commit author
        #[arg(long)]
        worker: Option<String>,
        /// Why the branch must not land
        #[arg(long)]
        reason: String,
    },
    /// Lift one open veto, by a name that is not the blocked worker
    ClearVeto {
        /// The veto id, as printed by `merge plan` or `merge veto`
        id: String,
        /// The reviewer or the human lifting the block
        #[arg(long = "by")]
        by: String,
        /// A policy statement that names the veto it clears
        #[arg(long)]
        policy: Option<String>,
    },
    /// List the vetoes on record for this repository
    Vetoes {
        /// Only the ones still blocking
        #[arg(long)]
        open: bool,
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    /// List all installed Ollama models
    List,
    /// Check health of a specific model
    Health {
        /// Model name to check
        model: String,
    },
    /// Show the smart-selected default model
    Default,
    /// Say which GGUF this machine can serve, from the dated advice table,
    /// with the pinned address and checksum to fetch it by
    Advice,
}

#[derive(Subcommand)]
enum LlamacppAction {
    /// Show llama.cpp server status, loaded model, and token throughput
    Status,
    /// Start a llama-server process hosting the configured GGUF model
    Start {
        /// GGUF model path (overrides config llama_cpp_model_path)
        #[arg(long)]
        model: Option<String>,
        /// Port to bind (defaults to 8080)
        #[arg(long, default_value = "8080")]
        port: u16,
        /// llama-server executable path (overrides config / PATH lookup)
        #[arg(long)]
        exec: Option<String>,
    },
    /// Stop a llama-server process started by xencode
    Stop,
    /// Load / switch a model on a running llama-server
    Load {
        /// GGUF model path to load
        model: String,
    },
    /// Unload the currently loaded model
    Unload,
    /// List models available on a running llama-server
    List,
    /// Set the configured GGUF model path used for auto-start/load
    SetPath {
        /// GGUF model path
        path: String,
    },
}

#[derive(Subcommand)]
enum HwAction {
    /// Read RAM, cores and compute devices, and recommend launch flags
    Probe {
        /// GGUF file to size the answer against (defaults to the configured
        /// model, then to any GGUF in the usual cache directory)
        #[arg(long)]
        model: Option<String>,

        /// llama-server binary to ask what it can offload to
        #[arg(long)]
        exec: Option<String>,
    },
}

#[derive(Subcommand)]
enum SessionAction {
    /// Name a run so it can be resumed without its id
    Name {
        /// The run id (or a prefix of one)
        run: String,
        /// The name to give it
        name: String,
    },
    /// Resolve a name, id prefix, or `latest` to a full run id
    Resolve {
        /// Name, id prefix, or `latest`
        target: String,
    },
    /// Print a session's transcript; `--redacted` scrubs secrets
    Export {
        /// Name, id prefix, or `latest`
        target: String,
        /// Scrub secrets with the trace module's patterns
        #[arg(long)]
        redacted: bool,
    },
}

#[derive(Subcommand)]
enum HistoryAction {
    /// Show which history indexes exist here and time the queries that use them
    Status {
        /// Repository to look at (default: the current directory)
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// File to run a blame probe on (default: README.md, else the first
        /// tracked file)
        #[arg(long)]
        file: Option<String>,

        /// Emit JSON instead of a table
        #[arg(long)]
        json: bool,
    },
    /// Print the ~250-token history digest for one file
    Digest {
        /// Repository to read (default: the current directory)
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// File to digest, repository-relative
        file: String,

        /// Emit JSON instead of text
        #[arg(long)]
        json: bool,
    },
    /// Write the commit-graph and the multi-pack-index, then time them
    Setup {
        /// Repository to write into (default: the current directory)
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// File to run a blame probe on (default: README.md, else the first
        /// tracked file)
        #[arg(long)]
        file: Option<String>,

        /// Emit JSON instead of a table
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// Show cache statistics
    Stats,
    /// Clear all cached responses
    Clear,
    /// Drop the oldest cached responses until the cache directory fits under a
    /// size. The downloaded advisory corpora are not counted and cannot be
    /// removed by this command.
    Gc {
        /// Largest the cached responses may be, in megabytes
        #[arg(long)]
        max_mb: u64,
    },
}

#[derive(Subcommand)]
enum AuditAction {
    /// Check an audit log for records that were changed after they were written
    Verify {
        /// Log to check (default: <state dir>/audit.jsonl)
        path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum AdvisoryAction {
    /// Download both corpora: 6.3 MB of RustSec text plus a 3.5 MB OSV archive, about 20 MB unpacked
    Sync {
        /// Where to keep them (default: <config dir>/advisories)
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// What the corpus says about one crate, judged against a version when given
    Show {
        /// Crate name, as it appears in Cargo.lock
        crate_name: String,

        /// Version to judge; without it the records are listed but not assessed
        #[arg(long)]
        version: Option<String>,

        /// Corpus location (default: <config dir>/advisories)
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Judge every package in this project's Cargo.lock against the corpus
    Check {
        /// Project to read Cargo.lock from (default: the current directory)
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// Corpus location (default: <config dir>/advisories)
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Whether a corpus exists here, how big it is, and when it was taken
    Status {
        /// Corpus location (default: <config dir>/advisories)
        #[arg(long)]
        dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum MemoryAction {
    /// List all conversation sessions
    List {
        /// Show all sessions, including empty ones (0 messages)
        #[arg(long)]
        all: bool,
    },
    /// Show transcript of a session
    Show {
        /// Session ID
        session: String,
        /// Print only the first message recorded in the session's event log
        #[arg(long)]
        first: bool,
    },
    /// Fork a conversation session into a child holding an exact event prefix
    Fork {
        /// Source session ID
        session: String,
        /// New session ID (defaults to <session>_fork_<timestamp>)
        #[arg(long)]
        as_id: Option<String>,
        /// Prefix length: number of parent events to inherit (defaults to all)
        #[arg(long)]
        prefix: Option<usize>,
    },
    /// Delete conversation sessions that have no messages
    Prune,
    /// List the durable facts this repository contradicts, with how long each
    /// has been contradicted for. `--apply` retires the ones past a year.
    Gc {
        /// Remove the facts contradicted for twelve months or longer from
        /// `.xencode/state.md`
        #[arg(long)]
        apply: bool,
    },
    /// How many revisions each durable fact has been re-checked against, and what
    /// that evidence supports saying. Prints an interval, never a confidence.
    Evidence {
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },
    /// Publish one finding into the shared memory that workers hand each other
    Publish {
        /// Worker id publishing under (needs a policy that grants the scope)
        #[arg(long)]
        worker: String,
        /// Scope to publish into, e.g. `architecture` or `release-gate`
        #[arg(long)]
        scope: String,
        /// The finding itself
        content: String,
    },
    /// Read shared memory as the worker you name, marked and attributed
    Read {
        /// Worker id reading (needs a policy that grants each scope shown)
        #[arg(long)]
        worker: String,
        /// Read only this scope; without it, every scope the policy grants
        #[arg(long)]
        scope: Option<String>,
    },
    /// Set and inspect which scopes each worker may read and publish
    Policy {
        #[command(subcommand)]
        action: MemoryPolicyAction,
    },
}

#[derive(Subcommand)]
enum MemoryPolicyAction {
    /// Replace one worker's memory policy. A worker with no policy can neither
    /// read nor publish anything, and a repeat of this command overwrites what
    /// was granted before.
    Set {
        /// Worker id the policy applies to
        #[arg(long)]
        worker: String,
        /// Scopes this worker may read (repeatable; none means it reads nothing)
        #[arg(long = "read", value_name = "SCOPE")]
        read: Vec<String>,
        /// Scopes this worker may publish into (repeatable; none means it publishes nothing)
        #[arg(long = "publish", value_name = "SCOPE")]
        publish: Vec<String>,
    },
    /// Show the configured policies, or one worker's if named
    Show {
        /// Show only this worker's policy
        #[arg(long)]
        worker: Option<String>,
    },
}

#[derive(Subcommand)]
enum PerfAction {
    /// Measure every hot path and store those samples as the baseline.
    ///
    /// There is no filter here on purpose: a baseline that covers three of seven
    /// paths is not a baseline, and the whole tree has to be quiet for it to mean
    /// anything.
    Record {
        /// Store it even if a path was measured too widely to support a verdict
        #[arg(long)]
        force: bool,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Measure the hot paths and compare them against the stored baseline
    Check {
        /// Measure only the paths whose name contains this
        #[arg(long)]
        filter: Option<String>,

        /// The slowdown that raises a flag, as a percentage of the baseline
        #[arg(long, default_value_t = 5.0)]
        alert_pct: f64,

        /// The significance level a verdict is judged at
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Show the recorded baseline without measuring anything
    Show {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
}

#[derive(Subcommand)]
enum PriceAction {
    /// Show every price a cost report would use, which document each came from,
    /// and which of the models this project has actually run are unpriced
    Show {
        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Read the public catalogue again and replace the cached listing with it.
    ///
    /// This is the only thing in xencode that dials out for a price, and it is
    /// asked for. Nothing is sent with the request: the listing is published for
    /// anybody to read, which is the same reason no key is needed to fetch it.
    Fetch {
        /// Read from somewhere other than OpenRouter's listing — a gateway that
        /// publishes the same document, or an address for testing
        #[arg(long)]
        url: Option<String>,
    },
}

/// What `xencode runs` can do. Bare `xencode runs` lists, the way bare
/// `xencode prices` shows: the common question needs no verb.
#[derive(Subcommand)]
enum RunsAction {
    /// List recent runs, oldest first. The window is not a cap: `run_by_id`
    /// reaches past it, and so does `show` with a full id.
    List {
        /// How many of the newest runs to print
        #[arg(long, default_value_t = 20)]
        limit: usize,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Show one run: its model, every question a person answered while it
    /// went, and the verification rows its session left behind.
    Show {
        /// Which run: its full id, or enough of the start to name one run
        /// and no other — the same rule `xencode replay` uses.
        run_id: String,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Print the commit trailer block naming one run, for pasting into a
    /// commit message. Every line is a `Token: value` trailer, so `git
    /// interpret-trailers` reads it as trailers.
    Trailer {
        /// Which run: its full id, or an unambiguous prefix of it.
        run_id: String,
    },
}

#[derive(Subcommand)]
enum WorktreeAction {
    /// List git worktrees of the current repository
    List,
    /// Create a worktree at <path>, checking out <branch> (or a new branch
    /// named after the directory when omitted)
    Add {
        /// Directory for the new worktree
        path: String,

        /// Existing branch or commit to check out
        branch: Option<String>,
    },
    /// Remove a worktree (git refuses dirty worktrees; main never removable)
    Remove {
        /// Path of the worktree to remove
        path: String,
    },
}

#[derive(Subcommand)]
enum TaskAction {
    /// List known tasks with their derived status
    List {
        /// Emit JSON instead of a table
        #[arg(long)]
        json: bool,
    },
    /// Start a background task (survives this CLI process)
    Start {
        /// Shell command to run
        command: String,

        /// Human-friendly label (defaults to the command)
        #[arg(long, short)]
        name: Option<String>,
    },
    /// Show a task's status and captured output
    Poll {
        /// Task ID
        id: u64,

        /// Trailing output lines to print
        #[arg(long, default_value = "50")]
        lines: usize,
    },
    /// Ask a running task to stop
    Stop {
        /// Task ID
        id: u64,
    },
    /// Forget a finished task and delete its files
    Rm {
        /// Task ID
        id: u64,
    },
}

/// Stack for the thread the whole program runs on.
///
/// The command dispatch is one async function over every subcommand, and its
/// future is large. Windows gives a program's main thread 1 MiB, which a debug
/// build compiled with Rust 1.99 overflowed before printing `--version`
/// (2026-10-08); Linux gives 8 MiB, which is why CI never saw it. So the
/// program runs on a thread whose stack is set here instead of by the platform.
const MAIN_STACK_BYTES: usize = 32 * 1024 * 1024;

fn main() {
    let worker = std::thread::Builder::new()
        .name("xencode".into())
        .stack_size(MAIN_STACK_BYTES)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("the async runtime starts")
                .block_on(async_main())
        })
        .expect("the main thread starts");
    if let Err(panic) = worker.join() {
        // Same behaviour as a panic on the main thread: message already
        // printed by the hook, process ends with the panic's own unwinding.
        std::panic::resume_unwind(panic);
    }
}

async fn async_main() {
    let cli = Cli::parse();

    if cli.dump_config {
        if let Err(e) = run_dump_config() {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }

    // Bare `xencode` launches the TUI, as documented in the README
    let result = match cli.command.unwrap_or(Commands::Tui { in_process: false }) {
        Commands::Scan {
            path,
            hidden,
            max_depth,
            format,
        } => run_scan(path, hidden, max_depth, format),
        Commands::Config { action } => run_config(action),
        Commands::Models { action } => run_models(action).await,
        Commands::Cache { action } => run_cache(action),
        Commands::Audit { action } => run_audit(action),
        Commands::Advisories { action } => run_advisories(action).await,
        Commands::Deps { path, format } => run_deps(path, format),
        Commands::Query {
            prompt,
            model,
            no_cache,
            session,
            temperature,
            top_k,
            min_p,
            mirostat,
            seed,
            max_tokens,
            grammar,
            json_schema,
            images,
            format,
        } => {
            run_query(
                prompt,
                model,
                no_cache,
                session,
                temperature,
                top_k,
                min_p,
                mirostat,
                seed,
                max_tokens,
                grammar,
                json_schema,
                images,
                format,
            )
            .await
        }
        Commands::Memory { action } => run_memory(action),
        Commands::Tasks { action } => run_tasks(action),
        Commands::Worktree { action } => run_worktree(action),
        Commands::Advise {
            filter,
            json,
            limit,
        } => run_advise(filter, json, limit),
        Commands::Server {
            port,
            host,
            cert,
            key,
            audit_path,
            allow_insecure_public,
        } => run_server(port, host, cert, key, audit_path, allow_insecure_public).await,
        Commands::Analyze {
            path,
            format,
            runtime,
        } => run_analyze(path, format, runtime),
        Commands::Fetch { url, format } => run_fetch(url, format).await,
        Commands::Interop {
            agents,
            timeout,
            out,
            format,
            repeat,
            fan_out,
            check_auth,
            capture_dir,
            trace,
        } => run_interop(
            agents,
            timeout,
            out,
            format,
            repeat,
            fan_out,
            check_auth,
            capture_dir,
            trace,
        ),
        Commands::Anchor {
            path,
            timeout,
            dry_run,
            format,
        } => run_anchor(path, timeout, dry_run, format),
        Commands::Toolchain {
            action,
            allow_dirty,
            format,
        } => run_toolchain(&action, allow_dirty, format),
        Commands::Doctor {
            env,
            deps,
            selfcheck,
            format,
        } => run_doctor(env, deps, selfcheck, format).await,
        Commands::Session { action } => run_session(action),
        Commands::Badge => run_badge(),
        Commands::Acp => xencode_acp_rs::serve_stdio().await,
        Commands::Engine {
            project,
            wait_limit,
        } => run_engine(project, wait_limit).await,
        Commands::Paths { format } => run_paths(format),
        Commands::Migrate { dry_run } => run_migrate(dry_run),
        Commands::Verify {
            skip,
            timeout,
            session,
            format,
        } => run_verify(skip, timeout, session, format),
        Commands::Envcheck { format } => run_envcheck(format),
        Commands::Agents {
            contract,
            health,
            agent,
            build_package,
            package,
            resume,
            test_cmds,
            route_task,
            require_caps,
            max_cost,
            redispatch_task,
            replacement_agent,
            stop_reason,
            format,
        } => run_agents(AgentsArgs {
            contract,
            health,
            target_agent: agent.as_deref(),
            build_package: build_package.as_deref(),
            package_path: package.as_deref(),
            resume,
            test_cmds: &test_cmds,
            route_task: route_task.as_deref(),
            require_caps: &require_caps,
            max_cost,
            redispatch_task: redispatch_task.as_deref(),
            replacement_agent: replacement_agent.as_deref(),
            stop_reason: stop_reason.as_deref(),
            format,
        }),
        Commands::Hotspots { limit, format } => run_hotspots(limit, format),
        Commands::Impact {
            file,
            semantic,
            symbol,
            limit,
            format,
        } => {
            if semantic {
                run_impact_semantic(&file, symbol.as_deref(), limit, format)
            } else {
                run_impact(&file, limit, format)
            }
        }
        Commands::Removal {
            file,
            limit,
            format,
        } => run_removal(&file, limit, format),
        Commands::Generate { artifact, shell } => run_generate(artifact, shell),
        Commands::Mutants {
            diff,
            timeout,
            check_repair,
            format,
        } => run_mutants(diff, timeout, check_repair, format),
        Commands::Cov {
            base,
            test,
            show_missing_lines,
            format,
        } => run_cov(base, test, show_missing_lines, format),
        Commands::Perf { action } => run_perf(action),
        Commands::Prices { action } => run_prices(action).await,
        Commands::Test {
            packages,
            retries,
            stress_count,
            timeout,
            isolate,
            base,
            repeat,
            format,
            session,
        } => run_test(
            packages,
            retries,
            stress_count,
            timeout,
            isolate,
            base,
            repeat,
            format,
            session,
        ),
        Commands::ReleaseNotes {
            from,
            to,
            release,
            out,
            force,
            format,
        } => run_release_notes(from, to, release, out, force, format),
        Commands::Review {
            base,
            session,
            format,
        } => run_review(base, session, format),
        Commands::Replay {
            run_id,
            list,
            run_tools,
            tool_root,
            out,
        } => run_replay(run_id, list, run_tools, tool_root, out).await,
        Commands::Runs { action } => run_runs(action),
        Commands::Run {
            prompt,
            detach,
            resume,
            list,
            show,
            log,
            stop,
            model,
            tool_root,
            max_rounds,
            max_minutes,
            max_cost,
            allow_shell,
            tail,
            ollama_url,
            llamacpp_url,
            child,
            xencode_dir: child_xencode_dir,
        } => {
            run_detached_command(DetachedCommandOptions {
                prompt,
                detach,
                resume,
                list,
                show,
                log,
                stop,
                model,
                tool_root,
                max_rounds,
                max_minutes,
                max_cost,
                allow_shell,
                tail,
                ollama_url,
                llamacpp_url,
                child,
                child_xencode_dir,
            })
            .await
        }
        Commands::Plugin { action } => run_plugin_action(action),
        Commands::Mcp { action } => run_mcp(action).await,
        Commands::Eval { action } => run_eval(action).await,
        Commands::Llamacpp { action } => run_llamacpp(action).await,
        Commands::Hw { action } => run_hw(action),
        Commands::History { action } => run_history(action),
        Commands::Colab { action } => run_colab(action).await,
        Commands::Remote { action } => run_remote(action),
        Commands::Computers { action, json } => run_computers(action, json).await,
        Commands::Compete { action } => run_compete(action),
        Commands::Merge { action } => run_merge(action),
        Commands::Team { action } => run_team(action).await,
        Commands::Orchestrator { action } => run_orchestrator(action).await,
        Commands::Bootstrap {
            path,
            check,
            format,
        } => run_bootstrap(&path, check, format),
        Commands::Tui { in_process } => run_tui(in_process).await,
    };

    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run_scan(
    root: PathBuf,
    include_hidden: bool,
    max_depth: Option<usize>,
    format: OutputFormat,
) -> Result<(), String> {
    let options = ScanOptions {
        max_depth,
        include_hidden,
        ..ScanOptions::default()
    };

    let entries = scan_workspace(root, &options).map_err(|e| e.to_string())?;
    match format {
        OutputFormat::Json => {
            let arr: Vec<serde_json::Value> = entries
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "kind": e.kind.to_string(),
                        "bytes": e.bytes,
                        "path": e.path.display().to_string(),
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&arr).map_err(|e| e.to_string())?
            );
        }
        OutputFormat::Text => {
            for entry in entries {
                let bytes = entry
                    .bytes
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".to_string());
                println!("{}\t{}\t{}", entry.kind, bytes, entry.path.display());
            }
        }
    }
    Ok(())
}

/// Split a `config set` list value (I4-01): comma-separated, trimmed, empties
/// dropped. Used by `agent_fallback_models`, whose order is the try order.
fn parse_comma_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Parse a `config set` boolean (`1`/`true` also accepted for shell ergonomics).
fn parse_bool(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean: {value}")),
    }
}

/// Parse a `config set` port (1..=65535).
fn parse_u16(value: &str, key: &str) -> Result<u16, String> {
    let port: u16 = value
        .parse()
        .map_err(|_| format!("invalid port for {key}: {value}"))?;
    if port == 0 {
        return Err(format!("{key} must be 1..=65535"));
    }
    Ok(port)
}

/// Parse a `config set` daily budget cap: a positive number no larger than
/// `max`, or an empty value meaning "no cap". Zero is refused rather than
/// stored — a cap of nothing is crossed before the first turn ends, which is a
/// way to disable the product that reads as a budget. The upper bound is where
/// the unit stops being an amount anyone spends in a day, so a mistyped number
/// is caught at the keyboard instead of deciding tomorrow's context window.
fn parse_daily_cap(value: &str, key: &str, max: u64, unit: &str) -> Result<Option<u64>, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let amount: u64 = trimmed
        .parse()
        .map_err(|_| format!("invalid number for {key}: {value}"))?;
    if amount == 0 {
        return Err(format!(
            "{key} cannot be 0 — a cap of nothing is crossed by the first turn, which is not a budget"
        ));
    }
    if amount > max {
        return Err(format!("{key} must be between 1 and {max} {unit}"));
    }
    Ok(Some(amount))
}

/// Settings whose empty value means "nothing set at all". Printing
/// `set x = ` after one of those describes a value that is no longer there, so
/// the removal gets its own wording.
const UNSET_WHEN_EMPTY: [&str; 6] = [
    "power_cents_per_kwh",
    "budget_tokens_per_day",
    "budget_energy_wh_per_day",
    "budget_usd_micros_per_day",
    "budget_minutes_per_day",
    "search_searxng_url",
];

/// Print where each kind of xencode's own files is kept, and say in as many
/// words when they are still in the single directory this layout replaced.
/// `xencode badge` (DK-2): start the floating badge, detached from this
/// terminal. A badge that is already running makes the new copy exit at once.
fn run_badge() -> Result<(), String> {
    if xencode_live_rs::live_dir().is_ok_and(|dir| xencode_live_rs::badge_running(&dir)) {
        println!("The badge is already running.");
        return Ok(());
    }
    let beside = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let exe = xencode_live_rs::find_badge(beside.as_deref()).ok_or_else(|| {
        "xencode-badge was not found next to xencode or on PATH. Build it with: cargo build --release --manifest-path rust/badge/Cargo.toml"
            .to_string()
    })?;
    xencode_live_rs::spawn_badge(&exe)
        .map_err(|e| format!("could not start {}: {e}", exe.display()))?;
    println!("Started {}", exe.display());
    Ok(())
}

fn run_paths(format: OutputFormat) -> Result<(), String> {
    use xencode_config_rs::paths;

    let places = paths::locations();
    let override_root = paths::override_root();
    if matches!(format, OutputFormat::Json) {
        let out = serde_json::json!({
            "override": override_root.as_ref().map(|root| root.display().to_string()),
            "legacy_in_use": paths::legacy_in_use(),
            "locations": places.iter().map(|place| serde_json::json!({
                "kind": place.kind.label(),
                "in_use": place.in_use.display().to_string(),
                "modern": place.modern.display().to_string(),
                "legacy": place.legacy,
            })).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("\n  Where xencode keeps its own files:");
    for place in &places {
        println!("  • {:>17}  {}", place.kind.label(), place.in_use.display());
    }
    if let Some(root) = &override_root {
        println!(
            "\n  $XCODE_CONFIG_DIR is set to {}, so all four are read from that one tree.",
            root.display()
        );
        println!("  The XDG locations above are where they would go without it.");
        return Ok(());
    }
    let still_old: Vec<_> = places
        .iter()
        .filter(|place| place.legacy)
        .map(|place| place.kind.label())
        .collect();
    if still_old.is_empty() {
        println!("  Nothing is left in the old single directory.");
    } else {
        println!("\n  Still in ~/.xencode: {}", still_old.join(", "));
        println!(
            "  Run `xencode migrate --dry-run` to see what would move, then `xencode migrate`."
        );
        println!("  Until you run it, nothing moves: xencode keeps reading the old directory.");
    }
    Ok(())
}

/// Move an existing installation's files out of `~/.xencode` into the four
/// directories they belong in, printing every entry the migration could not
/// complete rather than hiding it.
fn run_migrate(dry_run: bool) -> Result<(), String> {
    use xencode_config_rs::paths;

    let report = if dry_run {
        paths::plan_migration()
    } else {
        paths::migrate()
    }
    .map_err(|e| e.to_string())?;

    if report.is_empty() {
        println!(
            "  {} holds nothing to migrate.",
            report.old_directory.display()
        );
        println!("  Run `xencode paths` to see where each kind of file is read from.");
        return Ok(());
    }
    if report.moved.is_empty() {
        println!(
            "  {} from {}.",
            if dry_run {
                "Nothing would move"
            } else {
                "Nothing moved"
            },
            report.old_directory.display()
        );
    } else {
        println!(
            "  {} {} {} from {}:",
            if dry_run { "Would move" } else { "Moved" },
            report.moved.len(),
            if report.moved.len() == 1 {
                "entry"
            } else {
                "entries"
            },
            report.old_directory.display()
        );
    }
    for kind in paths::ALL {
        let moved: Vec<_> = report
            .moved
            .iter()
            .filter(|entry| entry.kind == kind)
            .collect();
        if moved.is_empty() {
            continue;
        }
        // The destination, not `paths::dir`: before the move that still answers
        // with the old directory, which is the thing being left.
        let where_to = paths::locations()
            .into_iter()
            .find(|place| place.kind == kind)
            .map(|place| place.modern)
            .unwrap_or_default();
        println!("  {} → {}", kind.label(), where_to.display());
        for entry in moved {
            let name = entry
                .from
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            println!("    {name}");
        }
    }
    for left in &report.left {
        println!("  Left where it is: {}", left.path.display());
        println!("      {}", left.reason);
    }
    if dry_run {
        println!(
            "  {} {}.",
            report.old_directory.display(),
            if report.left.is_empty() {
                "is empty afterwards and would be removed"
            } else {
                "would be kept, because the entries above cannot all move"
            }
        );
    } else if report.old_directory_removed {
        println!(
            "  {} is now empty and has been removed.",
            report.old_directory.display()
        );
    } else {
        println!(
            "  {} is kept, because not everything in it moved.",
            report.old_directory.display()
        );
    }
    if dry_run {
        println!("\n  Nothing above has happened yet — run `xencode migrate` to do it.");
    }
    Ok(())
}

fn run_dump_config() -> Result<(), String> {
    let config = XencodeConfig::load().map_err(|e| e.to_string())?;
    let summary = config.composition_summary();
    let json = serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}

fn run_config(action: ConfigAction) -> Result<(), String> {
    match action {
        ConfigAction::Show { composition } => {
            let config = XencodeConfig::load().map_err(|e| e.to_string())?;
            if composition {
                let summary = config.composition_summary();
                let json = serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?;
                println!("{json}");
            } else {
                let json = config.to_json().map_err(|e| e.to_string())?;
                println!("{json}");
            }
            Ok(())
        }
        ConfigAction::Dump => run_dump_config(),
        ConfigAction::Set {
            key,
            value,
            dry_run,
        } => {
            let mut config = XencodeConfig::load().map_err(|e| e.to_string())?;
            // Which provider a credential key belongs to, when this is a credential.
            let credential = SecretProvider::from_config_key(&key);
            let secret = credential.is_some();
            match key.as_str() {
                "default_model" => config.default_model = value.clone(),
                "ollama_url" => config.ollama_url = value.clone(),
                "llama_cpp_url" => config.llama_cpp_url = value.clone(),
                "remote_url" => {
                    let trimmed = value.trim();
                    if !trimmed.is_empty()
                        && !(trimmed.starts_with("http://") || trimmed.starts_with("https://"))
                    {
                        return Err(
                            "remote_url must be an http:// or https:// URL, or empty to unset"
                                .to_string(),
                        );
                    }
                    config.remote_base_url = trimmed.to_string();
                }
                // A credential, for any provider that takes one. Stored and never
                // echoed back. The value may also be `command:<program> <args>`,
                // which stores the reference and reads the secret from that
                // command when it is needed, so `config.json` holds no key at all;
                // each provider can equally be left unset here and supplied by the
                // environment variable named for it.
                _ if credential.is_some() => {
                    config
                        .api_keys
                        .set_secret(credential.expect("the guard just matched this key"), &value);
                }
                "llama_cpp_model_path" => config.llama_cpp_model_path = value.clone(),
                "llama_cpp_model_url" => config.llama_cpp_model_url = value.clone(),
                // A SHA256 is accepted in the forms it is usually copied in:
                // lower or upper case, optionally prefixed with `sha256-`. It is
                // stored lowercased so a later comparison with a digest has one
                // spelling to deal with. Blank unsets the pin.
                "llama_cpp_model_sha256" => {
                    let trimmed = value
                        .trim()
                        .trim_start_matches("sha256-")
                        .trim_start_matches("SHA256-")
                        .to_string();
                    if !trimmed.is_empty()
                        && (trimmed.len() != 64 || !trimmed.chars().all(|c| c.is_ascii_hexdigit()))
                    {
                        return Err(
                            "llama_cpp_model_sha256 must be 64 hexadecimal characters, or empty \
                             to stop checking the model file"
                                .to_string(),
                        );
                    }
                    config.llama_cpp_model_sha256 = trimmed.to_lowercase();
                }
                "llama_cpp_executable" => config.llama_cpp_executable = value.clone(),
                "llama_cpp_args" => {
                    config.llama_cpp_args =
                        value.split_whitespace().map(|s| s.to_string()).collect();
                }
                // How much a local model may think before answering. Checked
                // with the same code that turns the word into a launch flag, so
                // a value accepted here is one that will actually take effect
                // later. "auto" and blank both mean "leave it to the model", and
                // are stored as nothing rather than as a word.
                "llama_cpp_reasoning" => {
                    let trimmed = value.trim();
                    xencode_models_rs::llamacpp::reasoning_launch_args(Some(trimmed))?;
                    config.llama_cpp_reasoning =
                        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                            None
                        } else {
                            Some(trimmed.to_string())
                        };
                }
                // The same two questions the request body will answer, checked
                // with the code that turns the word into the field. Ollama has no
                // thinking budget, so a number here is refused with the route
                // that does have one named.
                "ollama_reasoning" => {
                    let trimmed = value.trim();
                    xencode_providers_rs::OllamaRequest::from_settings(Some(trimmed), None)?;
                    config.ollama_reasoning =
                        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                            None
                        } else {
                            Some(trimmed.to_string())
                        };
                }
                // The duration itself is Ollama's to read; all that is checked
                // here is that the value is shaped like a duration at all, since
                // a wrong one otherwise surfaces as a failed request in the
                // middle of a turn rather than as a message now.
                "ollama_keep_alive" => {
                    let trimmed = value.trim();
                    if !trimmed.is_empty() && !trimmed.chars().any(|c| c.is_ascii_digit()) {
                        return Err(format!(
                            "ollama_keep_alive must be a duration like \"10m\" or \"30s\", or \"0\" \
                             to unload the model after a request, not \"{value}\""
                        ));
                    }
                    config.ollama_keep_alive = if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed.to_string())
                    };
                }
                "max_cache_size" => {
                    config.max_cache_size = value
                        .parse()
                        .map_err(|_| format!("invalid number: {value}"))?;
                }
                "response_timeout" => {
                    config.response_timeout = value
                        .parse()
                        .map_err(|_| format!("invalid number: {value}"))?;
                }
                "cache_enabled" => {
                    config.cache_enabled = value
                        .parse()
                        .map_err(|_| format!("invalid boolean: {value}"))?;
                }
                "memory_enabled" => {
                    config.memory_enabled = value
                        .parse()
                        .map_err(|_| format!("invalid boolean: {value}"))?;
                }
                "max_memory_items" => {
                    config.max_memory_items = value
                        .parse()
                        .map_err(|_| format!("invalid number: {value}"))?;
                }
                "layout" => config.layout = value.clone(),
                // The context budget profile. "auto" hands the choice to the
                // memory probe; a name pins it. Reject a word that is neither,
                // rather than storing a value that silently means "auto".
                "hardware_profile" => {
                    let trimmed = value.trim().to_ascii_lowercase();
                    if trimmed != "auto"
                        && xencode_context_rs::HardwareProfile::from_key(&trimmed).is_none()
                    {
                        return Err(format!(
                            "hardware_profile must be \"auto\", \"low\", \"balanced\" or \"high\", not \"{value}\""
                        ));
                    }
                    config.hardware_profile = trimmed;
                }
                "rounded_borders" => {
                    config.rounded_borders = value
                        .parse()
                        .map_err(|_| format!("invalid boolean: {value}"))?;
                }
                "show_scrollbars" => {
                    config.show_scrollbars = value
                        .parse()
                        .map_err(|_| format!("invalid boolean: {value}"))?;
                }
                "show_line_numbers" => {
                    config.show_line_numbers = value
                        .parse()
                        .map_err(|_| format!("invalid boolean: {value}"))?;
                }
                // The mouse is a trade and the Settings row is the way out of it,
                // so the trade has to be settlable from the shell as well.
                "mouse_capture" => config.mouse_capture = parse_bool(&value)?,
                "agent_approval" => config.agent_approval = value.clone(),
                "agent_max_rounds" => {
                    let rounds: usize = value
                        .parse()
                        .map_err(|_| format!("invalid number: {value}"))?;
                    if !(1..=64).contains(&rounds) {
                        return Err("agent_max_rounds must be 1..=64".to_string());
                    }
                    config.agent_max_rounds = rounds;
                }
                "agent_command_timeout" => {
                    let seconds: u64 = value
                        .parse()
                        .map_err(|_| format!("invalid number: {value}"))?;
                    if !(1..=600).contains(&seconds) {
                        return Err("agent_command_timeout must be 1..=600 seconds".to_string());
                    }
                    config.agent_command_timeout = seconds;
                }
                // Comma-separated ordered fallbacks (I4-01); the primary model
                // is tried first regardless, so the list holds only alternates.
                "agent_fallback_models" => {
                    config.agent_fallback_models = parse_comma_list(&value);
                }
                // Consent to send a prompt off this machine at all (PR-2). Kept
                // separate from the `*_key` entries on purpose: a key proves who
                // you are to a provider, it does not authorise the trip.
                "allow_cloud_models" => config.allow_cloud_models = parse_bool(&value)?,
                // The other half of the same idea, for a text file rather than a
                // prompt: `read_docs` stays on cargo's local copies until this is
                // on, and crates.io/docs.rs are never dialed behind it.
                "allow_online_docs" => config.allow_online_docs = parse_bool(&value)?,
                // Consent to price a model this project has no hand-written rate
                // for off a catalogue somebody else published. It does not authorise
                // a trip — `xencode prices fetch` is the only thing that dials out,
                // and it is asked for. This only decides whether the copy already on
                // disk is read, which is why the documented path stays offline.
                "price_lookup" => config.price_lookup = parse_bool(&value)?,
                // Whether the agent is offered `web_fetch` at all. The switch is
                // about the address the model picks, which is why it is separate
                // from the two above: this trip has no named host in front of it,
                // so it stays unoffered until the user says it may be.
                "allow_web_fetch" => config.allow_web_fetch = parse_bool(&value)?,
                // The same kind of consent, about work instead of bytes: while this
                // is false, xencode hands nothing to another vendor's coding agent,
                // because that program signs into its own account and xencode never
                // sees the traffic it sends. It is not a network switch, so it is not
                // checked against the four above — see `xencode team plan` or
                // `xencode agents --route` for the surfaces that refuse on it.
                "allow_external_workers" => config.allow_external_workers = parse_bool(&value)?,
                // Which engine `web_search` may ask, if any. Checked against the
                // names the search code itself accepts, so a typo here is told at
                // the keyboard instead of at the first search of the next session.
                "search_provider" => {
                    let name = value.trim().to_ascii_lowercase();
                    if !xencode_analysis_rs::search::PROVIDER_NAMES.contains(&name.as_str()) {
                        return Err(format!(
                            "search_provider must be one of {}; `none` leaves the tool unoffered",
                            xencode_analysis_rs::search::PROVIDER_NAMES.join(", ")
                        ));
                    }
                    config.search_provider = name;
                }
                // The instance to ask when the provider above is `searxng`. Same
                // shape rule as `remote_url`, because this address is the one a
                // search request is dialed to and it is checked again, against
                // private ranges, before every connection.
                "search_searxng_url" => {
                    let trimmed = value.trim().trim_end_matches('/').to_string();
                    if !trimmed.is_empty()
                        && !(trimmed.starts_with("http://") || trimmed.starts_with("https://"))
                    {
                        return Err(
                            "search_searxng_url must be an http:// or https:// URL, or empty to \
                             unset"
                                .to_string(),
                        );
                    }
                    config.search_searxng_url = trimmed;
                }
                // Keep every model call of every run, in the clear, under
                // `.xencode/cache/sessions`. Off by default because it is the
                // most sensitive copy this program can make of a conversation.
                "session_recording" => config.session_recording = parse_bool(&value)?,
                // Fence each run_command, background_start and shell hook in a
                // bubblewrap namespace (SE-7). Off by default because it changes
                // what an approved command can reach, and on it needs bwrap — a
                // command is then refused rather than run unsandboxed when bwrap
                // is missing, so a machine without it must not turn this on blind.
                "run_command_sandbox" => config.run_command_sandbox = parse_bool(&value)?,
                // Let a saved profile take a turn on its own when the prompt reads
                // as the kind of work it is marked for. Off until it is asked for,
                // because a model the user did not choose answering a query is a
                // surprise a script cannot see in its own output.
                "model_routing" => config.model_routing = parse_bool(&value)?,
                // The electricity tariff, in cents per kilowatt-hour. Empty clears
                // it, which leaves a local turn showing its watt-hours with no
                // price next to them — the honest rendering, not a free one.
                "power_cents_per_kwh" => {
                    let trimmed = value.trim();
                    if trimmed.is_empty() {
                        config.power_cents_per_kwh = None;
                    } else {
                        let cents: f64 = trimmed
                            .parse()
                            .map_err(|_| format!("invalid number: {value}"))?;
                        if !cents.is_finite() || cents < 0.0 || cents > 1_000.0 {
                            return Err(
                                "power_cents_per_kwh must be a tariff between 0 and 1000 cents"
                                    .to_string(),
                            );
                        }
                        config.power_cents_per_kwh = Some(cents);
                    }
                }
                // The four daily caps. They are one mechanism seen from four
                // angles, so they are validated by one function; what each unit
                // means is in the config field's own documentation, and
                // `xencode help` points at it. Crossing one buys the next turn
                // down a smaller context — nothing is ever refused over it.
                "budget_tokens_per_day" => {
                    config.budget_tokens_per_day =
                        parse_daily_cap(&value, "budget_tokens_per_day", 1_000_000_000, "tokens")?;
                }
                "budget_energy_wh_per_day" => {
                    config.budget_energy_wh_per_day =
                        parse_daily_cap(&value, "budget_energy_wh_per_day", 24_000, "watt-hours")?;
                }
                "budget_usd_micros_per_day" => {
                    config.budget_usd_micros_per_day = parse_daily_cap(
                        &value,
                        "budget_usd_micros_per_day",
                        10_000_000_000,
                        "micro-dollars ($1.00 = 1000000)",
                    )?;
                }
                "budget_minutes_per_day" => {
                    config.budget_minutes_per_day =
                        parse_daily_cap(&value, "budget_minutes_per_day", 1_440, "minutes")?;
                }
                "mcp_timeout" => {
                    let seconds: u64 = value
                        .parse()
                        .map_err(|_| format!("invalid number: {value}"))?;
                    if !(1..=300).contains(&seconds) {
                        return Err("mcp_timeout must be 1..=300 seconds".to_string());
                    }
                    config.mcp_timeout = seconds;
                }
                // Colab bridge (Milestone K). `colab_enabled` gates the whole
                // feature; the rest tune the forward/session that `up` builds.
                "colab_enabled" => config.colab.enabled = parse_bool(&value)?,
                "colab_session" => config.colab.session = value.clone(),
                "colab_local_port" => {
                    config.colab.local_port = parse_u16(&value, "colab_local_port")?
                }
                "colab_remote_port" => {
                    config.colab.remote_port = parse_u16(&value, "colab_remote_port")?
                }
                "colab_runtime" => {
                    if !matches!(value.as_str(), "llama.cpp" | "ollama") {
                        return Err("colab_runtime must be \"llama.cpp\" or \"ollama\"".to_string());
                    }
                    config.colab.runtime = value.clone();
                }
                "colab_model" => config.colab.model = value.clone(),
                "colab_weights_source" => {
                    if !matches!(value.as_str(), "hf" | "drive" | "gcs") {
                        return Err(
                            "colab_weights_source must be \"hf\", \"drive\" or \"gcs\"".to_string()
                        );
                    }
                    config.colab.weights_source = value.clone();
                }
                "colab_quant" => config.colab.quant = value.clone(),
                "colab_auto_connect" => config.colab.auto_connect = parse_bool(&value)?,
                "composition_profile" => {
                    let trimmed = value.trim();
                    if xencode_config_rs::CompositionProfile::for_name(trimmed).is_none() {
                        let known =
                            xencode_config_rs::CompositionProfile::known_profiles().join(", ");
                        return Err(format!("composition_profile must be one of: {known}"));
                    }
                    config.composition_profile = trimmed.to_string();
                }
                "computer_backend" => {
                    let trimmed = value.trim();
                    if trimmed.is_empty() {
                        return Err("computer_backend cannot be empty".to_string());
                    }
                    config.computer_backend = trimmed.to_string();
                }
                "worker_adapter" => {
                    let trimmed = value.trim();
                    if trimmed.is_empty() {
                        return Err("worker_adapter cannot be empty".to_string());
                    }
                    config.worker_adapter = trimmed.to_string();
                }
                _ => return Err(format!("unknown config key: {key}")),
            }
            if dry_run {
                // Everything above ran — the key was recognised and the value
                // validated against the real config — and nothing was written.
                if secret {
                    if xencode_config_rs::is_secret_reference(&value) {
                        // A preview does not execute the person's helper: a
                        // reference could name a command that changes something.
                        println!("would set {key} = a command reference — nothing written, and the command was not run (--dry-run)");
                    } else if value.trim().is_empty() {
                        println!("would clear {key} — nothing written (--dry-run)");
                    } else {
                        println!("would set {key} (value not shown) — nothing written (--dry-run)");
                    }
                } else {
                    println!("would set {key} = {value} — nothing written (--dry-run)");
                }
                return Ok(());
            }
            config.save().map_err(|e| e.to_string())?;
            if secret {
                if xencode_config_rs::is_secret_reference(&value) {
                    // The reference is what was stored, and it is not printed: a
                    // person who wrote the key into the command line by mistake
                    // would otherwise see it come back.
                    println!("set {key} = a command reference — the secret is read from that command and stays out of config.json");
                    // Run it once now, so a reference that cannot be read is
                    // found at the moment it is written down instead of in the
                    // middle of a turn.
                    if let Some(provider) = credential {
                        match config.api_keys.secret(provider) {
                            Ok(Some(_)) => println!("note: the command answered with a key."),
                            Ok(None) => println!(
                                "note: the command names nothing, so it answered with no key."
                            ),
                            Err(problem) => println!("note: {problem}"),
                        }
                    }
                } else if value.trim().is_empty() {
                    // Blank is the way a credential is unset, so say that rather
                    // than claiming something was stored.
                    println!("cleared {key} — the environment variable named for the provider, if any, now supplies it");
                } else {
                    println!("set {key} = (stored, not shown)");
                }
            } else if value.trim().is_empty() && UNSET_WHEN_EMPTY.contains(&key.as_str()) {
                println!("cleared {key} — nothing is set for it, and the behaviour is what it was before it was ever named");
            } else {
                println!("set {key} = {value}");
            }
            // A layout name that is neither a shipped preset nor a template the
            // config declares renders classic. `config set` is where a user
            // types such a name, so it is where saying so is worth the line —
            // the value is stored anyway, because the template it names may be
            // added afterwards.
            if key == "layout" {
                if let Some(problem) =
                    xencode_tui_rs::templates::problem(&config.layout_templates, &config.layout)
                {
                    println!("note: {problem}");
                }
            }
            // An energy cap is measured from the machine's own counter. On a
            // machine that publishes none there is no reading that could ever
            // reach it, and the person who set it should hear that now rather
            // than three days from now, from a budget that never moved.
            if key == "budget_energy_wh_per_day"
                && config.budget_energy_wh_per_day.is_some()
                && xencode_context_rs::power::package_energy_uj(
                    xencode_context_rs::power::POWERCAP_ROOT,
                )
                .is_none()
            {
                println!(
                    "note: this machine publishes no CPU energy counter, so that cap cannot be \
                     reached here; the reading it waits for does not exist."
                );
            }
            Ok(())
        }
        ConfigAction::Reset => {
            let config = XencodeConfig::default();
            let path = XencodeConfig::config_path().map_err(|e| e.to_string())?;
            // The one save allowed to write over an unreadable file: discarding
            // what is there is what this command was asked to do.
            config.force_save_to(&path).map_err(|e| e.to_string())?;
            println!("configuration reset to defaults");
            Ok(())
        }
    }
}

async fn run_models(action: ModelAction) -> Result<(), String> {
    let config = XencodeConfig::load().unwrap_or_default();
    let mut client = OllamaClient::new(&config.ollama_url, config.response_timeout);
    let mut llama_client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);

    match action {
        ModelAction::List => {
            let ollama_models = client.list_models().await.unwrap_or_default();
            let llama_models = llama_client.list_models().await.unwrap_or_default();

            if ollama_models.is_empty() && llama_models.is_empty() {
                println!("No local models installed or running.");
                println!("For Ollama:   ollama pull qwen2.5:7b");
                println!("For llama.cpp: start llama-server with your GGUF model");
                return Ok(());
            }

            println!("{:<30} {:>12} PROVIDER", "MODEL", "SIZE");
            println!("{}", "-".repeat(60));
            for model in &ollama_models {
                let size_mb = model.size as f64 / 1_048_576.0;
                println!("{:<30} {:>9.1} MB [ollama]", model.name, size_mb);
            }
            for model in &llama_models {
                let display_name = format!("llamacpp:{}", model.id);
                println!("{:<30} {:>12} [llama.cpp]", display_name, "N/A");
            }
            let total = ollama_models.len() + llama_models.len();
            println!("\n{} model(s) available", total);
            Ok(())
        }
        ModelAction::Health { model } => {
            println!("Checking health of {model}...");
            if model.starts_with("llamacpp:")
                || model.starts_with("llama.cpp:")
                || model.starts_with("llama:")
            {
                let raw_model = model
                    .strip_prefix("llamacpp:")
                    .or_else(|| model.strip_prefix("llama.cpp:"))
                    .or_else(|| model.strip_prefix("llama:"))
                    .unwrap_or(&model);
                match llama_client.check_health(raw_model).await {
                    Ok(health) => {
                        println!("  Provider:      llama.cpp ({})", config.llama_cpp_url);
                        println!("  Status:        {}", health.status);
                        println!("  Response time: {:.3}s", health.response_time);
                        if let Some(ref err) = health.error_message {
                            println!("  Error:         {err}");
                        }
                    }
                    Err(e) => {
                        println!("  Provider:      llama.cpp ({})", config.llama_cpp_url);
                        println!("  Status:        unavailable");
                        println!("  Error:         {e}");
                    }
                }
            } else {
                let health = client
                    .check_health(&model)
                    .await
                    .map_err(|e| e.to_string())?;
                println!("  Provider:      Ollama ({})", config.ollama_url);
                println!("  Status:        {}", health.status);
                println!("  Response time: {:.3}s", health.response_time);
                if let Some(ref err) = health.error_message {
                    println!("  Error:         {err}");
                }
            }
            Ok(())
        }
        ModelAction::Default => {
            let default = client
                .get_smart_default()
                .await
                .map_err(|e| e.to_string())?;
            match default {
                Some(model) => println!("Smart default: {model}"),
                None => println!("No models available"),
            }
            Ok(())
        }
        ModelAction::Advice => run_models_advice(),
    }
}

/// Answer "which GGUF should this machine serve" from the advice table, and say
/// how old that answer is.
///
/// The capacity the table is matched against is the biggest single place a
/// model's bytes could go — one memory pool, not the sum of several, because
/// `llama-server` puts a model's weights in one place. That is the same reading
/// the launch preflight uses, so the two commands cannot disagree about whether
/// a model fits.
fn run_models_advice() -> Result<(), String> {
    use xencode_context_rs::hwprobe;
    use xencode_models_rs::advice;

    let advice = advice::Advice::load(&advice::default_path());
    if let advice::AdviceSource::EmbeddedAfterRefusingUserFile(reason) = &advice.source {
        println!("note:     {reason}");
    }

    let total_kib = xencode_context_rs::budget::total_memory_kib().unwrap_or(0);
    let available_kib = hwprobe::available_memory_kib().unwrap_or(0);
    let exe = xencode_models_rs::llamacpp::find_llama_server(None);
    let devices: Vec<hwprobe::ComputeDevice> = match &exe {
        Some(exe) => hwprobe::server_devices(exe)
            .as_deref()
            .map(hwprobe::parse_llama_devices)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    let pools = hwprobe::memory_pools(&devices, total_kib / 1024, available_kib / 1024);
    let largest = pools.iter().max_by_key(|pool| pool.usable_mib);
    let capacity_bytes = largest
        .map(|pool| pool.usable_mib * 1024 * 1024)
        .unwrap_or(0);
    match largest {
        Some(pool) => println!(
            "room:     {} ({})",
            xencode_models_rs::human_bytes(capacity_bytes),
            pool.label
        ),
        None => println!(
            "room:     nothing measured — this machine reported no memory a model could go in"
        ),
    }

    let age = advice.age_days(advice::AdviceFile::today_epoch_days());
    println!("advice:   checked {}", advice.file.as_of);
    if let Some(age) = age {
        if age > advice::ROT_HORIZON_DAYS {
            println!(
                "          {age} days old, past the {} days this table is meant to be trusted \
                 for — the entries may name models that no longer exist or quants that have \
                 been beaten",
                advice::ROT_HORIZON_DAYS
            );
        } else {
            println!("          {age} days old");
        }
    }

    let source = match &advice.source {
        advice::AdviceSource::Embedded => "the table shipped with xencode".to_string(),
        advice::AdviceSource::UserFile(path) => path.clone(),
        advice::AdviceSource::EmbeddedAfterRefusingUserFile(_) => {
            "the table shipped with xencode".to_string()
        }
    };
    println!("from:     {source}");

    match advice.tier_for(capacity_bytes) {
        None => println!(
            "answer:   nothing in the table fits in {r}",
            r = xencode_models_rs::human_bytes(capacity_bytes)
        ),
        Some(tier) => {
            println!("tier:     {}", tier.name);
            for entry in &tier.gguf {
                println!("          {}", entry.label);
                println!(
                    "            size     {}",
                    xencode_models_rs::human_bytes(entry.size_bytes)
                );
                println!("            url      {}", entry.url());
                println!("            sha256   {}", entry.sha256);
            }
            println!(
                "\nto serve one of these:\n  \
                 xencode config set llama_cpp_model_url \"{}\"\n  \
                 xencode config set llama_cpp_model_sha256 \"{}\"\n  \
                 xencode config set llama_cpp_model_path \"{}\"\n  \
                 xencode llamacpp start\n\n\
                 The URL is pinned to a repository revision and the checksum is the one the \
                 host published for these bytes on {}, so a launch refuses the file rather than \
                 serving bytes that are not the ones this table described.",
                tier.gguf[0].url(),
                tier.gguf[0].sha256,
                default_gguf_path(&tier.gguf[0].file),
                advice.file.as_of
            );
        }
    }
    Ok(())
}

/// Where a fetched model lands by default: the directory xencode keeps the
/// things it fetched, which is not the cache — a weight costs a long download to
/// replace, and a cache cleaner is meant to be able to throw a cache away.
fn default_gguf_path(file_name: &str) -> String {
    xencode_config_rs::paths::data_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join(file_name)
        .display()
        .to_string()
}

/// A duration git was measured taking, in milliseconds with one decimal,
/// because a warm repository answers in single-digit milliseconds and rounding
/// to whole numbers would print several probes as 0 ms.
fn history_ms(microseconds: u128) -> String {
    format!("{:.1} ms", microseconds as f64 / 1000.0)
}

/// How much a query handed back — what a model would have to pay to be shown
/// the history. Under a kibibyte the number is given in bytes, because "0 KiB"
/// for a line of text reads like "nothing".
fn history_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else {
        format!("{:.0} KiB", bytes as f64 / 1024.0)
    }
}

/// Print one status page: what exists, what it costs, what to do about it.
fn print_history_status(status: &xencode_context_rs::HistoryStatus) {
    println!("repo:    {}", status.git_dir);
    println!(
        "history: {} reachable commit(s){}{}",
        status.reachable_commits,
        if status.shallow { ", shallow" } else { "" },
        status
            .partial_clone
            .as_ref()
            .map(|f| format!(", partial clone (filter {f})"))
            .unwrap_or_default(),
    );
    match &status.commit_graph {
        Some(graph) => println!(
            "commit-graph: {} ({}){}",
            graph.path,
            history_bytes(graph.size as usize),
            match graph.verifies {
                Some(true) => ", verifies",
                Some(false) => ", DOES NOT VERIFY against the objects here",
                None => "",
            }
        ),
        None => println!("commit-graph: none"),
    }
    match &status.pack_index {
        Some(index) => println!(
            "multi-pack-index: {} ({}) over {} pack(s)",
            index.path,
            history_bytes(index.size as usize),
            index.packs
        ),
        None => println!("multi-pack-index: none"),
    }
    println!("timed now:");
    for query in &status.queries {
        match &query.failed {
            Some(reason) => println!("  {:<46} failed: {reason}", query.label),
            None => println!(
                "  {:<46} {:>9}  {} out",
                query.label,
                history_ms(query.microseconds),
                history_bytes(query.bytes),
            ),
        }
    }
    for action in status.actions() {
        println!("todo:    {action}");
    }
}

/// `xencode history` — read what makes history queries fast here, or write it.
fn run_history(action: HistoryAction) -> Result<(), String> {
    use xencode_context_rs::{default_blame_target, history_setup, history_status};

    if let HistoryAction::Digest { path, file, json } = action {
        let digest = xencode_context_rs::history_digest(&path, &file);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "file": file,
                    "digest": digest,
                    "chars": digest.len(),
                    "cap": xencode_context_rs::DIGEST_CHAR_CAP,
                }))
                .unwrap_or_default()
            );
        } else {
            println!("{digest}");
        }
        return Ok(());
    }
    let (command, path, file, json) = match action {
        HistoryAction::Status { path, file, json } => ("status", path, file, json),
        HistoryAction::Setup { path, file, json } => ("setup", path, file, json),
        HistoryAction::Digest { .. } => unreachable!("handled above"),
    };
    let blame = match file {
        Some(file) => Some(file),
        None => default_blame_target(&path),
    };

    if command == "setup" {
        let setup = history_setup(&path, blame.as_deref())?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&setup).unwrap_or_default()
            );
            return Ok(());
        }
        print_history_status(&setup.before);
        println!();
        for line in &setup.wrote {
            println!("wrote:   {line}");
        }
        for line in &setup.refused {
            println!("skipped: {line}");
        }
        println!();
        println!(
            "after:  (timed in the same run, so this table also has the operating system's page \
             cache warmed by the one above — read it as an upper bound, not as the index's own \
             effect)"
        );
        print_history_status(&setup.after);
        // The point of the command is the speed, so say plainly whether it
        // moved rather than leaving the reader to compare two tables. Two
        // runs of the same query on the same machine differ by a couple of
        // milliseconds, so a claim needs to clear both a share and a floor.
        const MIN_RATIO: f64 = 1.2;
        const MIN_FLOOR_US: u128 = 2_000;
        let mut moved = false;
        for (before, after) in setup.before.queries.iter().zip(&setup.after.queries) {
            if before.failed.is_some() || after.failed.is_some() || after.microseconds == 0 {
                continue;
            }
            let gap = before.microseconds.abs_diff(after.microseconds);
            if gap < MIN_FLOOR_US {
                continue;
            }
            let ratio = before.microseconds as f64 / after.microseconds as f64;
            let words = if ratio >= MIN_RATIO {
                moved = true;
                format!("{ratio:.1}× faster")
            } else if ratio <= 1.0 / MIN_RATIO {
                moved = true;
                format!("{:.1}× slower", 1.0 / ratio)
            } else {
                continue;
            };
            println!(
                "{}: {} → {} ({words})",
                after.label,
                history_ms(before.microseconds),
                history_ms(after.microseconds),
            );
        }
        if !moved {
            println!(
                "no query changed by more than both 20% and 2 ms — on {} commits the commit \
                 chain was not the cost, so these indexes are there for the queries built on \
                 history rather than for the ones timed above",
                setup.before.reachable_commits
            );
        }
        return Ok(());
    }

    let status = history_status(&path, blame.as_deref())?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&status).unwrap_or_default()
        );
    } else {
        print_history_status(&status);
    }
    Ok(())
}

/// Report this machine as a place a model could be served, and the flags to
/// start a server with. Everything printed here is read from a named source at
/// run time, including the server's own device list — the two sources disagree
/// routinely and the point of the command is to show both, not to average them.
fn run_hw(action: HwAction) -> Result<(), String> {
    use xencode_context_rs::hwprobe;

    let HwAction::Probe { model, exec } = action;
    let config = XencodeConfig::load().unwrap_or_default();

    let total_kib = xencode_context_rs::budget::total_memory_kib();
    let available_kib = hwprobe::available_memory_kib();
    let fmt = |kib: Option<u64>| match kib {
        Some(kib) => format!("{:.1} GiB", kib as f64 / (1024.0 * 1024.0)),
        None => "unknown".to_string(),
    };
    println!(
        "ram:     {} total, {} available",
        fmt(total_kib),
        fmt(available_kib)
    );
    println!(
        "cores:   {}",
        hwprobe::cpu_core_count()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );

    let asked_exec = exec
        .as_deref()
        .or(if config.llama_cpp_executable.is_empty() {
            None
        } else {
            Some(config.llama_cpp_executable.as_str())
        });
    let exe = xencode_models_rs::llamacpp::find_llama_server(asked_exec);
    match &exe {
        Some(exe) => match hwprobe::server_version(exe) {
            Some(version) => println!("server:  {exe}  version {version}"),
            None => println!("server:  {exe}"),
        },
        None => println!(
            "server:  no llama-server found — set config llama_cpp_executable or pass --exec, \
             because what the binary can offload to cannot be read without asking it"
        ),
    }
    let devices: Vec<hwprobe::ComputeDevice> = match &exe {
        Some(exe) => match hwprobe::server_devices(exe) {
            Some(text) => hwprobe::parse_llama_devices(&text),
            None => {
                println!(
                    "devices: this server does not answer --list-devices, so what it can offload \
                     to is unknown rather than absent"
                );
                Vec::new()
            }
        },
        None => Vec::new(),
    };

    let cards = hwprobe::drm_cards(std::path::Path::new("/sys/class/drm"));
    if !cards.is_empty() {
        println!("cards:   the kernel's view, which does not report memory at all");
        for card in &cards {
            println!(
                "         {} · {} {} · driver {}{}",
                card.card,
                hwprobe::vendor_name(&card.vendor),
                card.device,
                if card.driver.is_empty() {
                    "none"
                } else {
                    &card.driver
                },
                if card.has_connector {
                    ", driving a display"
                } else {
                    ", no display attached"
                }
            );
        }
    }
    if !cards.is_empty() {
        println!(
            "         no memory figure is printed for a card here: PCI config space is what the \
             kernel exposes without a vendor tool, and on this box it reports a 256 MiB window \
             for a 2048 MiB device. The sizes below come from the server."
        );
    }
    if !devices.is_empty() {
        println!("devices  what the server itself can use:");
        let ram_total_mib = total_kib.map(|k| k / 1024).unwrap_or(0);
        for device in &devices {
            let kind = if !device.is_offload_target() {
                "the CPU path, reported as a device and not one"
            } else if hwprobe::is_shared_memory_device(device.total_mib, ram_total_mib) {
                "shares system memory — measured slower than the CPU here"
            } else {
                "its own memory"
            };
            println!(
                "         {:<9} {:<40} {:>6} MiB total, {:>6} MiB free · {}",
                device.id, device.name, device.total_mib, device.free_mib, kind
            );
        }
    }

    let model_path = model
        .clone()
        .or_else(|| {
            if config.llama_cpp_model_path.trim().is_empty() {
                None
            } else {
                Some(config.llama_cpp_model_path.clone())
            }
        })
        .or_else(|| xencode_models_rs::llamacpp::resolve_gguf_model(None, None));
    let shape = model_path.as_deref().and_then(hwprobe::read_gguf_shape);
    match (&model_path, &shape) {
        (Some(path), Some(shape)) => println!(
            "model:   {path}\n         {} · {} blocks · {} KV heads · head width {} · {} MiB of \
             weights",
            if shape.name.is_empty() {
                shape.architecture.as_str()
            } else {
                shape.name.as_str()
            },
            shape.block_count,
            shape.kv_head_count,
            shape.head_dim,
            shape.file_bytes / 1024 / 1024
        ),
        (Some(path), None) => println!("model:   {path} · no GGUF geometry could be read from it"),
        (None, _) => println!(
            "model:   none given — `--model <path.gguf>`, or set llama_cpp_model_path, and the \
             cache arithmetic below has a number under it"
        ),
    }
    if let Some(shape) = &shape {
        println!(
            "cache:   {:.1} KiB per token at 16-bit, {:.1} at {} keys and {} values — this is \
             what sits on top of the weights and it grows with every token of context",
            shape.kv_kib_per_token("f16", "f16"),
            shape.kv_kib_per_token(hwprobe::CACHE_K, hwprobe::CACHE_V),
            hwprobe::CACHE_K,
            hwprobe::CACHE_V
        );
    }
    if let (Some(available_kib), Some(shape)) = (available_kib, &shape) {
        let weights_mib = shape.file_bytes / 1024 / 1024;
        println!(
            "mmap:    the weights are read off disk as they are used; {} MiB available now \
             against {} MiB of model, and a build that fills the page cache is what turns a \
             serving model into a stalling one",
            available_kib / 1024,
            weights_mib
        );
    }

    let decision = xencode_context_rs::ProfileDecision::resolve(&config.hardware_profile);
    let ram_total_mib = total_kib.map(|k| k / 1024).unwrap_or(0);
    let rec = hwprobe::recommend(
        &devices,
        shape.as_ref(),
        ram_total_mib,
        decision.profile.ctx_tokens(),
    );
    println!(
        "budget:  {} · the token budget layer works with a {} token window",
        decision.describe(),
        decision.profile.ctx_tokens()
    );
    println!("recommend");
    for note in &rec.notes {
        println!("         · {note}");
    }
    let flags: Vec<&str> = rec.args.iter().map(String::as_str).collect();
    println!("         flags: {}", flags.join(" "));
    println!(
        "         to keep them: xencode config set llama_cpp_args \"{}\"\n         \
         (config args are passed last, and llama-server takes the later of a repeated flag, so \
         this overrides the profile's own --ctx-size)",
        flags.join(" ")
    );
    Ok(())
}

/// Bring a GGUF file onto the machine, saying how far it has got as it goes.
///
/// A model file is the one part of running a local server that takes minutes
/// rather than seconds, which makes it both the part worth interrupting and the
/// part whose interruption should not cost the entire transfer again. The bytes
/// land in a sidecar `.part` file and the next attempt asks the server for the
/// tail of it, so re-running the command after a Ctrl-C or a closed laptop lid
/// continues where it stopped.
/// `expected_sha256` is the checksum configured for this model, if the person
/// set one. It comes from outside the transfer, which is the only way comparing
/// with it proves anything: a digest taken from the server's own answer would
/// match whatever the server chose to send.
async fn fetch_model(url: &str, path: &str, expected_sha256: Option<&str>) -> Result<(), String> {
    use std::io::IsTerminal;
    let free = xencode_context_rs::hwprobe::free_disk_bytes(path);
    let left_off = xencode_models_rs::partial_bytes(path);
    if left_off > 0 {
        println!(
            "  {path} is not there yet, but a stopped download is: {} of its bytes are on disk.",
            xencode_models_rs::human_bytes(left_off)
        );
        if expected_sha256.is_some() {
            println!(
                "  those bytes will be hashed along with the rest, so a prefix that is not the \
                 file this checksum describes fails the whole transfer."
            );
        }
    } else {
        println!("  {path} is not there yet; fetching it now.");
    }
    println!("  from {url}");

    let tty = io::stdout().is_terminal();
    // The callback is shared as a plain `Fn`, so the throttle keeps its own
    // state rather than borrowing one from here.
    let last_percent = std::sync::atomic::AtomicI64::new(-1);
    let on_progress = |progress: xencode_models_rs::Progress| {
        let line = progress.label();
        if tty {
            print!("\r  {line}");
            let _ = io::stdout().flush();
        } else if let Some(fraction) = progress.fraction() {
            // One line every five percent when the output is being piped or
            // saved, where a carriage return would only be clutter.
            let percent = (fraction * 100.0) as i64;
            if percent >= last_percent.load(std::sync::atomic::Ordering::Relaxed) + 5 {
                last_percent.store(percent, std::sync::atomic::Ordering::Relaxed);
                println!("  {line}");
            }
        } else {
            println!("  {line}");
        }
    };
    let result =
        xencode_models_rs::fetch_model_file(url, path, free, expected_sha256, &on_progress).await;
    if tty {
        println!();
    }
    let got = result.map_err(|e| format!("the model download did not finish: {e}"))?;
    println!(
        "  model ready: {}",
        xencode_models_rs::human_bytes(got.bytes)
    );
    if got.verified {
        println!(
            "  checksum verified: the bytes hash to what was expected ({short}…)",
            short = xencode_models_rs::short_rev(&got.sha256)
        );
    } else {
        println!(
            "  unsigned: the bytes hash to {short}…, but nothing was expected, so that number \
             describes this file and proves nothing about where it came from. Set \
             llama_cpp_model_sha256 (or run xencode models advice for a pinned checksum) to \
             make a future launch check it.",
            short = xencode_models_rs::short_rev(&got.sha256)
        );
    }
    if got.resumed_from > 0 {
        println!(
            "  {} of that came from the bytes the earlier attempt had already fetched.",
            xencode_models_rs::human_bytes(got.resumed_from)
        );
    }
    if got.resume_refused {
        println!(
            "  note: the server sent the whole file rather than the part that was still \
             missing, so an interruption here would start the transfer over."
        );
    }
    Ok(())
}

async fn run_llamacpp(action: LlamacppAction) -> Result<(), String> {
    fn pid_file() -> std::path::PathBuf {
        xencode_config_rs::paths::state_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("llamaserver.pid")
    }

    fn write_pid_file(path: &std::path::Path, pid: u32) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, pid.to_string()).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn read_pid_file(path: &std::path::Path) -> Option<u32> {
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u32>()
            .ok()
    }

    fn kill_pid(pid: u32) -> Result<(), String> {
        #[cfg(windows)]
        {
            let status = std::process::Command::new("taskkill")
                .args(["/F", "/PID", &pid.to_string()])
                .output()
                .map_err(|e| e.to_string())?;
            if !status.status.success() {
                return Err(String::from_utf8_lossy(&status.stderr).to_string());
            }
        }
        #[cfg(not(windows))]
        {
            let status = std::process::Command::new("kill")
                .args([pid.to_string()])
                .output()
                .map_err(|e| e.to_string())?;
            if !status.status.success() {
                return Err(String::from_utf8_lossy(&status.stderr).to_string());
            }
        }
        Ok(())
    }

    match action {
        LlamacppAction::Status => {
            let config = XencodeConfig::load().unwrap_or_default();
            let client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);
            match client.ping().await {
                Ok(resp_time) => {
                    println!("llama.cpp status: healthy  ({:.3}s)", resp_time);
                    println!("  URL: {}", config.llama_cpp_url);
                    let models = client.list_models().await.unwrap_or_default();
                    for m in &models {
                        println!("  Model: {}", m.id);
                    }
                    if let Some(t) = client.timings().await.ok().flatten() {
                        if t.tokens_generated > 0 {
                            println!(
                                "  Throughput: ~{:.0} tok/s · {} tokens (last gen)",
                                t.predicted_per_second, t.tokens_generated
                            );
                        }
                    }
                }
                Err(e) => {
                    println!("llama.cpp status: unavailable");
                    println!("  URL: {}", config.llama_cpp_url);
                    println!("  Error: {e}");
                }
            }
            Ok(())
        }
        LlamacppAction::Start { model, port, exec } => {
            // Propagated, not defaulted: this branch ends in a save, and a
            // fallback here would write a full default block over the file the
            // person's keys are in.
            let mut config = XencodeConfig::load().map_err(|e| e.to_string())?;
            let model_path = model
                .clone()
                .unwrap_or_else(|| config.llama_cpp_model_path.clone());
            if model_path.trim().is_empty() {
                return Err(
                    "no GGUF model path set (use --model or `xencode config set llama_cpp_model_path <path>`)"
                        .to_string(),
                );
            }
            // Nothing below this line means anything without a model on disk, so
            // the fetch comes first: it is the longest step of a bring-up and the
            // only one that can be interrupted and picked up again.
            // What the file is supposed to hash to, if anyone said. Empty config
            // means nothing is pinned, which is reported as such rather than
            // treated as a check that passed.
            let expected = config.llama_cpp_model_sha256.trim().to_string();
            let expected = if expected.is_empty() {
                None
            } else {
                Some(expected.as_str())
            };
            if !std::path::Path::new(&model_path).exists() {
                let url = config.llama_cpp_model_url.trim().to_string();
                if url.is_empty() {
                    return Err(format!(
                        "no model at {model_path}, and nowhere to get it from (set config llama_cpp_model_url to the HTTPS address of the GGUF)"
                    ));
                }
                fetch_model(&url, &model_path, expected).await?;
            } else {
                // A file that is already here has not been looked at since it
                // arrived. Checking it costs one read of the file — about what
                // the loader spends opening it — and turns "the host replaced
                // this file" into a refusal before a server is started rather
                // than into wrong answers afterwards.
                let check = xencode_models_rs::check_model_file(&model_path, expected);
                match &check {
                    xencode_models_rs::FileCheck::Mismatch { expected, actual } => {
                        return Err(format!(
                            "refusing to start: {model_path} hashes to {}, not the {} this \
                             configuration expects. The file is not the one that was pinned — \
                             delete it and start again to fetch it fresh, or set \
                             llama_cpp_model_sha256 to the checksum you now want.",
                            xencode_models_rs::short_rev(actual),
                            xencode_models_rs::short_rev(expected)
                        ));
                    }
                    xencode_models_rs::FileCheck::Unreadable { reason, .. } => {
                        return Err(format!("refusing to start: {reason}"));
                    }
                    other => {
                        let label = other.label();
                        println!("  model:   {model_path} ({label})");
                        // A file with no checksum configured has nothing to be
                        // checked against, but xencode may have written down
                        // where its bytes came from when it fetched them. That
                        // record is this program's own note, so it is printed as
                        // one and not as a passing test.
                        if matches!(other, xencode_models_rs::FileCheck::Unsigned { .. }) {
                            match xencode_models_rs::read_provenance(&model_path) {
                                Some(record) => {
                                    let now = std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .map(|since| since.as_secs())
                                        .unwrap_or(0);
                                    let age_days = now.saturating_sub(record.fetched_at) / 86_400;
                                    let revision = match &record.revision {
                                        Some(revision) => format!(
                                            "at revision {}",
                                            xencode_models_rs::short_rev(revision)
                                        ),
                                        None => "from a host that named no revision".to_string(),
                                    };
                                    println!(
                                        "  record:  xencode wrote this file down {} day(s) ago, \
                                         {revision}, {}",
                                        age_days,
                                        xencode_models_rs::human_bytes(record.bytes)
                                    );
                                }
                                None => println!(
                                    "  record:  no checksum is set for this file and xencode did \
                                     not download it, so nothing here says what its bytes are \
                                     supposed to be"
                                ),
                            }
                        }
                    }
                }
            }
            let exe = find_llama_server(exec.as_deref().or(
                if config.llama_cpp_executable.is_empty() {
                    None
                } else {
                    Some(config.llama_cpp_executable.as_str())
                },
            ));
            let exe = exe.ok_or_else(|| {
                "could not find llama-server on PATH (set config llama_cpp_executable)".to_string()
            })?;

            println!("Starting llama-server: {exe}");
            println!("  model: {model_path}");
            println!("  port:  {port}");

            // Same preset the TUI's auto-start uses: the profile's flags first,
            // then what the reasoning setting asks for, then the config's own
            // `llama_cpp_args` so they have the final say. Here a reasoning
            // setting that names nothing stops the command instead of being
            // shrugged off — this server was asked for by name, so saying "off"
            // and starting a server that thinks is the one thing not to do.
            let profile = xencode_context_rs::ProfileDecision::resolve(&config.hardware_profile);
            let asked = xencode_models_rs::llamacpp::ServerReport {
                context_tokens: Some(profile.profile.ctx_tokens() as u32),
                slots: Some(1),
            };
            let reasoning = xencode_models_rs::llamacpp::reasoning_launch_args(
                config.llama_cpp_reasoning.as_deref(),
            )?;
            let mut preset = profile.profile.llama_cpp_args();
            preset.extend_from_slice(&reasoning);
            let mut args = xencode_models_rs::llamacpp::server_launch_args(
                &preset,
                None,
                &config.llama_cpp_args,
            );

            // Ask the machine before the server, from the server's own device
            // list and the model's own header: a model no memory here can hold
            // is said so once, in the second before anything starts, instead of
            // being launched, waited on for two minutes, and reported as a
            // timeout that never happened.
            // The window the command line will actually run with is the last
            // `--ctx-size` in it, because the config's own flags come after the
            // preset and the later value is the one `llama-server` obeys.
            let window = xencode_models_rs::ctx_size_in(&args)
                .unwrap_or_else(|| profile.profile.ctx_tokens());
            let preflight =
                xencode_context_rs::hwprobe::launch_preflight(&exe, &model_path, window, &args);
            for line in &preflight.lines {
                println!("  {line}");
            }
            if let Some(reason) = preflight.refuse {
                return Err(reason);
            }
            if let Some(shorter) = preflight.window {
                // Appended last, which is the position that decides the value
                // `llama-server` runs with.
                args.push("--ctx-size".to_string());
                args.push(shorter.to_string());
            }
            println!("  flags: {}", args.join(" "));

            let mut server = match xencode_models_rs::launch_and_wait(
                &exe,
                &model_path,
                port,
                &args,
                xencode_models_rs::Patience {
                    tries: 120,
                    gap: std::time::Duration::from_millis(500),
                },
                &|| false,
                &xencode_context_rs::hwprobe::smaller_window,
            )
            .await
            {
                xencode_models_rs::LaunchOutcome::Started { server, notes } => {
                    for note in notes {
                        println!("  {note}");
                    }
                    server
                }
                xencode_models_rs::LaunchOutcome::Failed { lines } => {
                    return Err(lines.join("\n"));
                }
                xencode_models_rs::LaunchOutcome::Cancelled => {
                    return Err("stopped before the server answered".to_string());
                }
            };

            // Save PID once the server is known to be alive: a file naming a
            // process that died during load is a `xencode llamacpp stop` that
            // reports stopping something it never started.
            let _ = write_pid_file(&pid_file(), server.pid());
            let client = LlamaCppClient::new(&server.base_url, 5);

            config.llama_cpp_url = server.base_url.clone();
            config.llama_cpp_model_path = model_path.clone();
            config.save().map_err(|e| e.to_string())?;

            println!("llama-server ready at {}", server.base_url);
            println!("Model loaded: {}", model_path);
            // A server that answers is not yet a server that has loaded its
            // model, so ask until it says what it is running as.
            let report = client
                .report_when_ready(120, std::time::Duration::from_millis(500))
                .await;
            println!(
                "{}",
                xencode_models_rs::llamacpp::settings_check_line(
                    &format!("{} preset", profile.profile.name()),
                    asked,
                    report
                )
            );
            println!("Server will keep running; use `xencode llamacpp stop` to terminate.");
            // Keep the process alive under this CLI invocation.
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(604800)).await;
                }
            });
            let _ = tokio::signal::ctrl_c().await;
            let _ = server.stop();
            let _ = std::fs::remove_file(pid_file());
            Ok(())
        }
        LlamacppAction::Stop => {
            match read_pid_file(&pid_file()) {
                Some(pid) => {
                    println!("Stopping llama-server (PID {pid})...");
                    match kill_pid(pid) {
                        Ok(()) => {
                            let _ = std::fs::remove_file(pid_file());
                            println!("Stopped");
                        }
                        Err(e) => eprintln!("Failed to stop PID {pid}: {e}"),
                    }
                }
                None => {
                    println!("No xencode-managed llama-server PID file found.");
                }
            }
            Ok(())
        }
        LlamacppAction::Load { model } => {
            let config = XencodeConfig::load().unwrap_or_default();
            let client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);
            println!("Loading model: {model}");
            match client.load_model(&model).await {
                Ok(()) => {
                    println!("Model loaded: {model}");
                    Ok(())
                }
                Err(e) => Err(format!("failed to load model: {e}")),
            }
        }
        LlamacppAction::Unload => {
            let config = XencodeConfig::load().unwrap_or_default();
            let client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);
            match client.unload_models().await {
                Ok(()) => {
                    println!("Model unloaded");
                    Ok(())
                }
                Err(e) => Err(format!("failed to unload model: {e}")),
            }
        }
        LlamacppAction::List => {
            let config = XencodeConfig::load().unwrap_or_default();
            let client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);
            let models = client.list_models().await.map_err(|e| e.to_string())?;
            for m in &models {
                println!("{}", m.id);
            }
            Ok(())
        }
        LlamacppAction::SetPath { path } => {
            let mut config = XencodeConfig::load().map_err(|e| e.to_string())?;
            config.llama_cpp_model_path = path.clone();
            config.save().map_err(|e| e.to_string())?;
            println!("llama_cpp_model_path = {path}");
            Ok(())
        }
    }
}

async fn run_colab(action: ColabAction) -> Result<(), String> {
    match action {
        ColabAction::Preflight { generate_key } => run_colab_preflight(generate_key).await,
        ColabAction::Up {
            session,
            gpu,
            runtime,
            model,
            weights,
            quant,
            local_port,
            remote_port,
            reconnect,
            dry_run,
        } => {
            run_colab_up_cli(
                session,
                gpu,
                runtime,
                model,
                weights,
                quant,
                local_port,
                remote_port,
                reconnect,
                dry_run,
            )
            .await
        }
        ColabAction::Status => run_colab_status_cli().await,
        ColabAction::Down => run_colab_down_cli().await,
    }
}

/// What a successful `colab up` would persist to config.json, computed by the
/// same call the real bring-up makes — so the preview cannot disagree with the
/// thing it previews. Only the keys that would actually change are listed.
fn colab_up_config_preview(current: &XencodeConfig, runtime: &str, local_port: u16) -> Vec<String> {
    let mut preview = current.clone();
    xencode_colab_rs::point_config_at_forward(
        &mut preview,
        runtime,
        &format!("http://127.0.0.1:{local_port}"),
    );
    let mut lines = Vec::new();
    for (key, before, after) in [
        (
            "llama_cpp_url",
            current.llama_cpp_url.as_str(),
            preview.llama_cpp_url.as_str(),
        ),
        (
            "ollama_url",
            current.ollama_url.as_str(),
            preview.ollama_url.as_str(),
        ),
        (
            "remote_base_url",
            current.remote_base_url.as_str(),
            preview.remote_base_url.as_str(),
        ),
    ] {
        if before != after {
            lines.push(format!(
                "config.json would change: {key}: {before} → {after}"
            ));
        }
    }
    if lines.is_empty() {
        lines.push("config.json would not change".to_string());
    }
    lines
}

/// Write, or only report, the three files a project that has never run xencode
/// is usually missing. See `xencode_context_rs::bootstrap` for why every byte of
/// them is a fact or a blank, and never a guessed command.
fn run_bootstrap(path: &std::path::Path, check: bool, format: OutputFormat) -> Result<(), String> {
    if !path.is_dir() {
        return Err(format!("not a directory: {}", path.display()));
    }
    // Every key the loader reads, at the value it ships with, from the same
    // struct this binary saves and reads: a template written by hand would drift
    // from the config the day a field is added, and a field added by hand would
    // teach a person a key that does not exist. The nine credential fields are
    // `None`, which serialises to `null`, so nothing but the shape of a real
    // settings file is written into somebody's repository.
    let template = serde_json::to_value(XencodeConfig::default())
        .map(|mut value| {
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "_comment".to_string(),
                    serde_json::json!("Every key xencode reads, at the value this binary ships with. Copy the keys you want into `config.json` in the settings directory (`$XCODE_CONFIG_DIR`, or `~/.config/xencode`) — this file is an example, not a live setting, and the loader ignores a key it does not know. A credential is best left null here and supplied by `xencode config set <KEY> command:<program> <args>`, which stores the reference and not the secret, or by the provider's own environment variable. Nothing in this file was typed out by hand: it is generated from the struct the loader reads, so a key here exists and a key that is missing was added after this binary was built."),
                );
            }
            value
        })
        .and_then(|value| serde_json::to_string_pretty(&value))
        .map_err(|e| format!("could not render the settings template: {e}"))?;
    let report =
        xencode_context_rs::bootstrap(path, &template, check).map_err(|e| format!("{e}"))?;

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "root": report.root,
                "git": report.facts.is_git,
                "branch": report.facts.branch,
                "revision": report.facts.revision,
                "files": report.facts.files,
                "scrubbed": report.scrubbed,
                "check": report.check_only,
                "entries": report.entries,
            })
        );
        return Ok(());
    }

    println!("Project: {}", report.root);
    let where_from = if !report.facts.is_git {
        "not a git repository".to_string()
    } else {
        let revision = report.facts.revision.as_deref().unwrap_or("(no revision)");
        let files = report
            .facts
            .files
            .map(|n| n.to_string())
            .unwrap_or_else(|| "file count unknown".to_string());
        match report.facts.branch.as_deref() {
            Some(branch) => format!("git branch {branch} at {revision}, {files} files"),
            None => format!("git detached head at {revision}, {files} files"),
        }
    };
    println!("  {where_from}");
    for entry in &report.entries {
        println!(
            "  {:<12} {:<23} {}",
            if report.check_only && entry.action == "write" {
                "would write"
            } else {
                entry.action
            },
            entry.path,
            entry.reason
        );
    }
    let writing = report.writing().count();
    let keeping = report.keeping().count();
    if report.check_only {
        println!(
            "\nReport only: {writing} would be written, {keeping} already in place. Nothing created."
        );
    } else if writing == 0 {
        println!("\nNothing to write: all {keeping} files are already there.");
    } else {
        println!(
            "\n{writing} file{} written. Nothing that already existed was touched.",
            if writing == 1 { "" } else { "s" }
        );
    }
    if report.scrubbed {
        println!(
            "A name on this disk held something credential-shaped, and was written as [redacted]."
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // CLI flags map 1:1 to colab up flags; a struct would just rename them
async fn run_colab_up_cli(
    session: Option<String>,
    gpu: Option<String>,
    runtime: Option<String>,
    model: Option<String>,
    weights: Option<String>,
    quant: Option<String>,
    local_port: Option<u16>,
    remote_port: Option<u16>,
    reconnect: bool,
    dry_run: bool,
) -> Result<(), String> {
    let config = XencodeConfig::load().map_err(|e| e.to_string())?;
    if !config.colab.enabled {
        return Err(
            "colab is disabled — set it: `xencode config set colab_enabled true`".to_string(),
        );
    }

    // Flags override config; config provides the defaults.
    let session = session
        .or(if config.colab.session.is_empty() {
            None
        } else {
            Some(config.colab.session.clone())
        })
        .unwrap_or("xencode-vm".to_string());
    let runtime = runtime.unwrap_or(config.colab.runtime.clone());
    let model = model
        .or(if config.colab.model.is_empty() {
            None
        } else {
            Some(config.colab.model.clone())
        })
        .unwrap_or("Qwen/Qwen2.5-7B-Instruct-GGUF".to_string());
    let weights_source = weights.unwrap_or(config.colab.weights_source.clone());
    let quant = quant.unwrap_or(config.colab.quant.clone());
    let local_port = local_port.unwrap_or(config.colab.local_port);
    let remote_port = remote_port.unwrap_or(config.colab.remote_port);

    if dry_run {
        // Deliberately before the preflight: preflight(true) would create the
        // SSH keypair, and a preview writes nothing anywhere.
        println!("colab up --dry-run — nothing was started and nothing was written.");
        println!("  session:   {session}");
        println!(
            "  runtime:   {runtime}   gpu: {}   model: {model}",
            gpu.as_deref().unwrap_or("T4")
        );
        println!(
            "  weights:   {weights_source}   quant: {}",
            if quant.is_empty() {
                "Q4_K_M (the default the VM serves)"
            } else {
                quant.as_str()
            }
        );
        println!(
            "  forward:   http://127.0.0.1:{local_port} → the VM's port {}",
            xencode_colab_rs::effective_remote_port(&runtime, remote_port)
        );
        for line in colab_up_config_preview(&config, &runtime, local_port) {
            println!("  {line}");
        }
        if reconnect {
            println!("  reconnect: reuses the URL in the recorded state when the forward is already live");
        }
        return Ok(());
    }

    // Gate on the bridge being usable; preflight also ensures the SSH key.
    let report = xencode_colab_rs::preflight(true)
        .await
        .map_err(|e| format!("colab preflight: {e}"))?;
    if !report.ready() {
        let failed: Vec<String> = report
            .checks
            .iter()
            .filter(|c| !c.ok)
            .map(|c| {
                format!(
                    "{} — {}",
                    c.name,
                    c.fix.as_deref().unwrap_or(c.detail.as_str())
                )
            })
            .collect();
        return Err(format!(
            "colab preflight not ready — run `xencode colab preflight` to see fixes:\n  {}",
            failed.join("\n  ")
        ));
    }

    let bins = xencode_colab_rs::resolve_binaries()?;
    let config_dir = XencodeConfig::config_dir().map_err(|e| e.to_string())?;
    let key = config_dir.join(xencode_colab_rs::KEY_FILENAME);

    let opts = xencode_colab_rs::UpOptions {
        session: session.clone(),
        gpu: gpu.unwrap_or("T4".to_string()),
        runtime: runtime.clone(),
        model: model.clone(),
        weights_source,
        quant,
        local_port,
        remote_port,
    };

    let url = if reconnect {
        xencode_colab_rs::run_colab_reconnect(&bins, &key, &opts).await?
    } else {
        xencode_colab_rs::run_colab_up(&bins, &key, &opts).await?
    };

    // Point the provider URLs at the forward so the picker and the
    // remote:/llama/ollama routes see the VM, then persist the change.
    let mut config = XencodeConfig::load().map_err(|e| e.to_string())?;
    let base = url.trim_end_matches("/v1");
    xencode_colab_rs::point_config_at_forward(&mut config, &runtime, base);
    config
        .save()
        .map_err(|e| format!("colab up: could not save config: {e}"))?;

    if reconnect {
        println!("Colab reconnected — {url}");
    } else {
        println!("Colab up — {url}");
    }
    println!("  session:   {session}");
    println!("  runtime:   {runtime}");
    println!("  model:     {model}");
    println!(
        "  provider:  {} -> {}",
        if runtime == "ollama" {
            "ollama_url"
        } else {
            "llama_cpp_url"
        },
        base
    );
    println!(
        "  remote:    {} (OpenAI-compatible)",
        config.remote_base_url
    );
    // What the VM's bootstrap warned about, such as a GPU build that ended up
    // serving on the CPU (K-6).
    if let Ok(Some(state)) = xencode_colab_rs::ColabState::load() {
        for warning in &state.warnings {
            println!("  {warning}");
        }
    }
    println!("  The VM endpoint is live. `xencode colab status` shows the bridge.");
    Ok(())
}

async fn run_colab_status_cli() -> Result<(), String> {
    let bins = xencode_colab_rs::resolve_binaries()?;
    let config = XencodeConfig::load().map_err(|e| e.to_string())?;
    // An empty `colab.session` is not a session name; `status` prefers the
    // recorded colab.json and only falls back to what the caller asks for.
    let report = xencode_colab_rs::run_colab_status(&bins, &config.colab.session).await;
    for line in &report.lines {
        println!("  {line}");
    }
    Ok(())
}

async fn run_colab_down_cli() -> Result<(), String> {
    let bins = xencode_colab_rs::resolve_binaries()?;
    let summary = xencode_colab_rs::run_colab_down(&bins).await?;
    println!("{summary}");
    Ok(())
}

async fn run_colab_preflight(generate_key: bool) -> Result<(), String> {
    let report = xencode_colab_rs::preflight(generate_key)
        .await
        .map_err(|e| format!("colab preflight: {e}"))?;

    println!("Colab preflight — google-colab-cli bridge");
    for check in &report.checks {
        let mark = if check.ok { "[ ok ]" } else { "[FAIL]" };
        println!("  {mark} {:<18} {}", check.name, check.detail);
        if let Some(fix) = &check.fix {
            println!("         fix: {fix}");
        }
    }

    if report.ready() {
        println!("Ready — the Colab bridge can be brought up.");
        Ok(())
    } else {
        Err(format!(
            "colab preflight found {} failing check(s)",
            report.failed()
        ))
    }
}

fn run_remote(action: RemoteAction) -> Result<(), String> {
    use xencode_config_rs::remotes::{
        active, forget, list, load, parse_destination, save, set_active, Forgot, RemoteProfile,
    };
    match action {
        RemoteAction::Add {
            name,
            host,
            runtime,
            model,
            port,
            local_port,
            remote_port,
            force,
        } => {
            let dest = parse_destination(&host).map_err(|e| e.to_string())?;
            let ssh_port = port.or(dest.port).unwrap_or(22);
            let profile = RemoteProfile {
                name: name.clone(),
                host: dest.host,
                user: dest.user,
                port: ssh_port,
                runtime,
                model,
                local_port,
                remote_port,
            };
            let path = save(&profile, force).map_err(|e| e.to_string())?;
            println!(
                "Recorded remote profile `{}` in {}",
                profile.name,
                path.display()
            );
            Ok(())
        }
        RemoteAction::List => {
            let inventory = list().map_err(|e| e.to_string())?;
            let current = active().map_err(|e| e.to_string())?;
            let active_name = current.as_ref().map(|p| p.name.as_str());
            if inventory.is_empty() {
                println!(
                    "No remote profiles recorded. Add one with `xencode remote add <name> <user@host>`."
                );
                return Ok(());
            }
            println!("Remote profiles:");
            for profile in &inventory.profiles {
                let marker = if active_name == Some(profile.name.as_str()) {
                    "*"
                } else {
                    " "
                };
                println!("  {marker} {:<12} {}", profile.name, profile.summary());
            }
            for (bad, err) in &inventory.unreadable {
                println!("  ! {:<12} unreadable: {err}", bad);
            }
            Ok(())
        }
        RemoteAction::Use { name } => {
            set_active(&name).map_err(|e| e.to_string())?;
            println!("Active remote profile set to `{name}`.");
            Ok(())
        }
        RemoteAction::Forget { name } => match forget(&name).map_err(|e| e.to_string())? {
            Forgot::Removed => {
                println!("Removed remote profile `{name}`.");
                Ok(())
            }
            Forgot::RemovedWhileActive => {
                println!("Removed remote profile `{name}` (active profile cleared).");
                Ok(())
            }
            Forgot::NotRecorded => Err(format!("`{name}` is not a recorded remote profile")),
        },
        RemoteAction::Show { name } => {
            let target = match name {
                Some(n) => load(&n)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| format!("`{n}` is not a recorded remote profile"))?,
                None => active().map_err(|e| e.to_string())?.ok_or_else(|| {
                    "No active remote profile. Use `xencode remote use <name>` or specify a profile name."
                        .to_string()
                })?,
            };
            let is_active = active()
                .ok()
                .flatten()
                .map(|a| a.name == target.name)
                .unwrap_or(false);
            let active_marker = if is_active { " [active]" } else { "" };
            println!("Remote profile `{}`{active_marker}", target.name);
            println!("  destination: {}", target.destination());
            println!("  ssh port:    {}", target.port);
            println!("  runtime:     {}", target.runtime);
            if let Some(m) = &target.model {
                println!("  model:       {m}");
            }
            println!("  local port:  {}", target.local_port);
            let rport = if target.remote_port == 0 {
                "0 (runtime default)".to_string()
            } else {
                target.remote_port.to_string()
            };
            println!("  remote port: {rport}");
            Ok(())
        }
    }
}

async fn run_computers(action: Option<ComputersAction>, json: bool) -> Result<(), String> {
    let bins = match xencode_colab_rs::resolve_binaries() {
        Ok(b) => b,
        Err(_) => {
            let colab_path = xencode_colab_rs::which("colab")
                .unwrap_or_else(|| std::path::PathBuf::from("colab"));
            let ssh_path =
                xencode_colab_rs::which("ssh").unwrap_or_else(|| std::path::PathBuf::from("ssh"));
            xencode_colab_rs::Binaries {
                colab: colab_path,
                ssh: ssh_path,
            }
        }
    };
    let config = XencodeConfig::load().map_err(|e| e.to_string())?;
    let key_path = XencodeConfig::config_dir()
        .map_err(|e| e.to_string())?
        .join(xencode_colab_rs::KEY_FILENAME);
    let registry = xencode_colab_rs::BackendRegistry::default();

    match action.unwrap_or(ComputersAction::List) {
        ComputersAction::List => {
            let list = registry.list_computers(&bins, &key_path);
            if json {
                let mut json_items = Vec::new();
                for c in &list {
                    json_items.push(serde_json::json!({
                        "id": c.id,
                        "kind": c.kind,
                        "description": c.description,
                        "available": c.available,
                        "detail": c.detail,
                        "active": c.id == config.computer_backend,
                    }));
                }
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json_items).map_err(|e| e.to_string())?
                );
                return Ok(());
            }

            println!("Registered computer backends:");
            for c in &list {
                let marker = if c.id == config.computer_backend {
                    "* "
                } else {
                    "  "
                };
                let status = if c.available {
                    "available"
                } else {
                    "unavailable"
                };
                println!(
                    "{marker}{:<8} ({:<6}) — {} [{}: {}]",
                    c.id, c.kind, c.description, status, c.detail
                );
            }
            Ok(())
        }
        ComputersAction::Show { name } => {
            let list = registry.list_computers(&bins, &key_path);
            let item = list.into_iter().find(|c| c.id == name).ok_or_else(|| {
                format!(
                    "unknown computer backend `{name}`; available: {}",
                    registry.available_backends().join(", ")
                )
            })?;
            if json {
                let doc = serde_json::json!({
                    "id": item.id,
                    "kind": item.kind,
                    "description": item.description,
                    "available": item.available,
                    "detail": item.detail,
                    "active": item.id == config.computer_backend,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            let active_str = if item.id == config.computer_backend {
                " (active)"
            } else {
                ""
            };
            println!("Computer `{}`{active_str}:", item.id);
            println!("  kind:        {}", item.kind);
            println!("  description: {}", item.description);
            println!(
                "  status:      {}",
                if item.available {
                    "available"
                } else {
                    "unavailable"
                }
            );
            println!("  detail:      {}", item.detail);
            Ok(())
        }
        ComputersAction::Use { name } => {
            if !registry.available_backends().contains(&name.as_str()) {
                return Err(format!(
                    "unknown computer backend `{name}`; available backends: {}",
                    registry.available_backends().join(", ")
                ));
            }
            let mut cfg = config;
            cfg.computer_backend = name.clone();
            cfg.save().map_err(|e| e.to_string())?;
            println!("Active computer backend set to `{name}`.");
            Ok(())
        }
        ComputersAction::Probe { name } => {
            let backend = registry.resolve(&name, &bins, &key_path)?;
            println!(
                "Probing computer backend `{name}` (kind: {})...",
                backend.kind()
            );
            let (avail, detail) = backend.is_available();
            if !avail {
                return Err(format!("computer `{name}` cannot be reached: {detail}"));
            }
            match backend.provision("probe-test", "none").await {
                Ok(()) => {
                    println!("Successfully connected to computer backend `{name}`.");
                    Ok(())
                }
                Err(e) => Err(format!("computer `{name}` cannot be reached: {e}")),
            }
        }
    }
}

/// Turn `--arm`, `--edit` and `--command` into candidate arm specs, refusing an
/// edit that names an unknown arm or tries to write outside its worktree.
fn build_candidate_arms(
    arms: &[String],
    edits: &[String],
    commands: &[String],
) -> Result<Vec<xencode_analysis_rs::CandidateArmSpec>, String> {
    use std::path::Component;
    use xencode_analysis_rs::CandidateArmSpec;

    let mut specs: Vec<CandidateArmSpec> = if arms.is_empty() {
        [
            ("arm-a", "Candidate Approach A"),
            ("arm-b", "Candidate Approach B"),
        ]
        .iter()
        .map(|(id, label)| CandidateArmSpec {
            arm_id: id.to_string(),
            label: label.to_string(),
            branch: None,
            command: None,
            file_edits: Vec::new(),
        })
        .collect()
    } else {
        arms.iter()
            .map(|raw| {
                let (arm_id, label) = match raw.split_once('=') {
                    Some((id, label)) => (id.trim().to_string(), label.trim().to_string()),
                    None => (raw.trim().to_string(), String::new()),
                };
                if arm_id.is_empty() {
                    return Err(format!(
                        "arm `{raw}` has no id: use `--arm id` or `--arm id=label`"
                    ));
                }
                let label = if label.is_empty() {
                    arm_id.clone()
                } else {
                    label
                };
                Ok(CandidateArmSpec {
                    arm_id,
                    label,
                    branch: None,
                    command: None,
                    file_edits: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?
    };

    let ids: Vec<String> = specs.iter().map(|s| s.arm_id.clone()).collect();
    for id in &ids {
        if ids.iter().filter(|other| *other == id).count() > 1 {
            return Err(format!(
                "arm id `{id}` given twice: every arm needs its own id"
            ));
        }
    }
    let known = || format!("known arms: {}", ids.join(", "));

    for chunk in edits.chunks(3) {
        let [arm, path, content] = chunk else {
            return Err("`--edit` needs three values: the arm, the path, the content".to_string());
        };
        let spec = specs
            .iter_mut()
            .find(|s| s.arm_id == *arm)
            .ok_or_else(|| format!("`--edit {arm}` names no arm — {}", known()))?;
        let rel = std::path::Path::new(path);
        if rel.is_absolute()
            || rel
                .components()
                .any(|c| matches!(c, Component::ParentDir | Component::RootDir))
        {
            return Err(format!(
                "`--edit` path `{path}` escapes the worktree: give it relative to the arm's own tree"
            ));
        }
        spec.file_edits
            .push((std::path::PathBuf::from(path), content.clone()));
    }

    for chunk in commands.chunks(2) {
        let [arm, cmd] = chunk else {
            return Err("`--command` needs two values: the arm, the shell command".to_string());
        };
        let spec = specs
            .iter_mut()
            .find(|s| s.arm_id == *arm)
            .ok_or_else(|| format!("`--command {arm}` names no arm — {}", known()))?;
        if spec.command.is_some() {
            return Err(format!(
                "arm `{arm}` already has a command — give one per arm"
            ));
        }
        spec.command = Some(cmd.clone());
    }

    Ok(specs)
}

/// Print one run's table, or its JSON, and — for a fresh run — how to pick.
fn print_competing_report(
    report: &xencode_analysis_rs::CompetingReport,
    format: OutputFormat,
    fresh: bool,
) -> Result<(), String> {
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(report).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!("{}", xencode_analysis_rs::format_competing_table(report));
    if fresh {
        println!("\nWorktrees kept for inspection:");
        for arm in &report.arms {
            println!("  {:<10} {}", arm.arm_id, arm.worktree_path.display());
        }
        println!(
            "\nThe rows above are only what the machine observed: nothing is computed over them \
             and nothing is declared the better arm. Choose with\n  xencode compete pick {} <arm-id>",
            report.run_id
        );
    }
    Ok(())
}

/// Where this project keeps its team recipes: `<project>/.xencode/teams`.
fn team_recipes_dir() -> std::path::PathBuf {
    xencode_context_rs::default_root()
        .join(xencode_context_rs::XENCODE_DIR)
        .join(xencode_core_rs::RECIPES_DIR)
}

/// `OR-9` and `OR-10` — read the team recipes, and run one that has been
/// approved. Listing, showing and planning are reads; `run` launches roles only
/// under a name, and only after it has printed the same plan the read shows.
/// `xencode team login-optin <agent>` (TM-5): the vendor's terms line, then
/// a `yes` read from the terminal turns the login on.
fn team_login_optin(agent: &str) -> Result<(), String> {
    use std::io::{BufRead, Write};
    let spec = xencode_team_rs::agents::find(agent).ok_or_else(|| {
        let known: Vec<&str> = xencode_team_rs::agents::AGENTS
            .iter()
            .map(|a| a.name)
            .collect();
        format!(
            "no outside agent called `{agent}`; the ones that can use a login are: {}",
            known.join(", ")
        )
    })?;
    println!("What the vendor's terms say about using your own login through another program:");
    println!();
    println!("  {}", spec.terms_line);
    println!("  {}", spec.terms_url);
    println!();
    println!(
        "With a login, {}'s work is on your own plan and xencode does not price it.",
        spec.name
    );
    print!("Type yes to let {} use your login: ", spec.name);
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    let mut answer = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut answer)
        .map_err(|e| format!("could not read the answer: {e}"))?;
    if answer.trim() != "yes" {
        return Err("not turned on: the answer was not `yes`".to_string());
    }
    let dir = xencode_config_rs::XencodeConfig::config_dir().map_err(|e| e.to_string())?;
    xencode_team_rs::agents::opt_in(&dir, spec.name)?;
    println!(
        "{} may now use your login; this is kept in {}.",
        spec.name,
        dir.join(xencode_team_rs::agents::OPTINS_FILE).display()
    );
    Ok(())
}

/// `xencode team clean` (TM-5): asks the project's engine, which knows which
/// workers still run, to remove the rest's worktrees.
async fn team_clean() -> Result<(), String> {
    let root = xencode_context_rs::default_root();
    let client = xencode_tui_rs::mcp_team::TeamClient::new(&root);
    let body = client
        .request(xencode_tui_rs::engine::proto::TeamRequest::Clean)
        .await?;
    let removed: Vec<&str> = body["removed"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if removed.is_empty() {
        println!("removed: none");
    } else {
        println!("removed: {}", removed.join(", "));
    }
    for kept in body["kept"].as_array().into_iter().flatten() {
        println!(
            "kept {}: {}",
            kept["id"].as_str().unwrap_or("?"),
            kept["why"].as_str().unwrap_or("")
        );
    }
    Ok(())
}

async fn run_team(action: TeamAction) -> Result<(), String> {
    // These two read no recipe, so a recipe folder that cannot be read does
    // not stop them.
    let action = match action {
        TeamAction::LoginOptin { agent } => return team_login_optin(&agent),
        TeamAction::Clean => return team_clean().await,
        other => other,
    };
    let dir = team_recipes_dir();
    let files = xencode_core_rs::load_recipes(&dir)
        .map_err(|e| format!("cannot read the team recipes in {}: {e}", dir.display()))?;
    match action {
        TeamAction::LoginOptin { agent } => team_login_optin(&agent),
        TeamAction::Clean => team_clean().await,
        TeamAction::List { format } => {
            if matches!(format, OutputFormat::Json) {
                let out = serde_json::json!({
                    "dir": dir.display().to_string(),
                    "recipes": files.iter().map(recipe_file_json).collect::<Vec<_>>(),
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            if files.is_empty() {
                println!("No team recipes in {}.", dir.display());
                println!(
                    "A recipe is one TOML file in that directory naming the roles on a team, \
                     which worker plays each one, and which checks gate it."
                );
                return Ok(());
            }
            let runnable: Vec<&xencode_core_rs::RecipeFile> =
                files.iter().filter(|f| recipe_status(f).is_ok()).collect();
            let faulty: Vec<&xencode_core_rs::RecipeFile> =
                files.iter().filter(|f| recipe_status(f).is_err()).collect();
            if !runnable.is_empty() {
                println!("{:<16} {:>5}  {:>7}  FILE", "RECIPE", "ROLES", "AT MOST");
                for file in runnable.iter() {
                    let recipe = file.recipe.as_ref().unwrap();
                    println!(
                        "{:<16} {:>5}  {:>7}  {}",
                        recipe.name,
                        recipe.roles.len(),
                        recipe.scheduler().capacity(),
                        file.path.display()
                    );
                }
                println!(
                    "\n`AT MOST` is the smaller of the recipe's own worker and verification \
                     numbers — the depth the queue actually runs at."
                );
            }
            if !faulty.is_empty() {
                if !runnable.is_empty() {
                    println!();
                }
                println!(
                    "Files in {} that are not a team that could schedule:",
                    dir.display()
                );
                for file in &faulty {
                    println!("  {}", recipe_status(file).err().unwrap());
                }
            }
            if !runnable.is_empty() {
                println!("\nShow one as written:  xencode team show <name>");
                println!("Compile one to a schedule:  xencode team plan <name>");
            }
            Ok(())
        }
        TeamAction::Show { name } => {
            let file = find_recipe(&files, &dir, &name)?;
            let recipe = file.recipe.as_ref().unwrap();
            println!(
                "{}  —  {} {}",
                recipe.name,
                recipe.roles.len(),
                if recipe.roles.len() == 1 {
                    "role"
                } else {
                    "roles"
                }
            );
            println!("{}", file.path.display());
            for role in &recipe.roles {
                println!("\n  {}", role.name);
                println!("    worker:   {}", role.worker);
                println!("    gate:     {}", gate_words(role));
                println!("    needs:    {}", needs_words(role));
                println!("    command:  {}", role.command);
            }
            println!(
                "\n  capacity: {} workers, {} verifications at once — the queue runs {} at a \
                 time, limited by {}",
                recipe.capacity.workers,
                recipe.capacity.verification_throughput,
                recipe.scheduler().capacity(),
                recipe.scheduler().binding().label(),
            );
            // Showing a recipe is a read of the file, so a recipe that cannot
            // schedule is still shown as written — and said out loud at the end.
            if let Err(problem) = recipe.validate() {
                println!("\n  …but this recipe would not schedule: {problem}");
            }
            Ok(())
        }
        TeamAction::Plan { name, format } => {
            let planned = plan_recipe(&files, &dir, &name)?;
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&plan_json(&planned, false))
                        .map_err(|e| e.to_string())?
                );
            } else {
                print_plan(&planned, false);
            }
            Ok(())
        }
        TeamAction::Run {
            name,
            approved_by,
            format,
        } => {
            let planned = plan_recipe(&files, &dir, &name)?;
            let approved_by = match approved_by {
                // The plan view is the default. Asking to run a recipe with no
                // name on it shows what would happen and changes nothing — not a
                // role, not a record.
                None => {
                    if matches!(format, OutputFormat::Json) {
                        let mut out = plan_json(&planned, false);
                        out["approval_required"] = serde_json::Value::Bool(true);
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
                        );
                    } else {
                        print_plan(&planned, true);
                    }
                    return Ok(());
                }
                Some(who) if who.trim().is_empty() => {
                    return Err(
                        "`--approved-by` needs a name: the record of a run says who agreed to \
                         it, and a blank is nobody"
                            .to_string(),
                    );
                }
                Some(who) => who,
            };
            // The posture is checked after the name, because a run with nobody on
            // it was never asked for, and before anything else, because a team is
            // not launched with refused roles dropped or one role left behind
            // (`OR-13`).
            if !planned.refused.is_empty() {
                let lines: Vec<String> = planned
                    .refused
                    .iter()
                    .map(|refusal| format!("  - {}", refusal.line()))
                    .collect();
                return Err(format!(
                    "the {} posture refuses {} of the agents this recipe assigns roles to, so \
                     nothing was launched and nothing was recorded:\n{}\nOpen the rule with \
                     `xencode config set allow_external_workers true`, or give those roles a \
                     worker the posture does not refuse — a team is never launched with the \
                     refused roles quietly dropped.",
                    planned.profile.name(),
                    planned.refused.len(),
                    lines.join("\n"),
                ));
            }
            if !matches!(format, OutputFormat::Json) {
                // The JSON form prints one document once the run is over — the
                // plan it ran on and what it took, in the same object.
                print_plan(&planned, false);
                println!(
                    "\n  Launching now, approved by {approved_by}. Each role is a real `sh -c` \
                     child of this process, at most {} at once.",
                    planned.scheduler.capacity()
                );
            }
            run_recipe(&planned, &approved_by, format).await
        }
    }
}

/// Where the record of what a run took is kept: `<project>/.xencode/team-runs`.
/// Not inside `.xencode/teams`, which is the committed recipe data — these are
/// measurements of one machine, and a tariff or a laptop change makes them
/// somebody else's numbers.
fn team_runs_dir() -> std::path::PathBuf {
    xencode_context_rs::default_root()
        .join(xencode_context_rs::XENCODE_DIR)
        .join(xencode_core_rs::RUNS_DIR)
}

/// A recipe read, checked, compiled into the graph `OR-2` schedules, and put
/// beside the estimate a previous run of this exact recipe left behind. Building
/// one launches nothing and writes nothing: the runs directory is read, and a
/// directory that is not there is an empty answer.
struct PlannedRecipe {
    recipe: xencode_core_rs::TeamRecipe,
    file: std::path::PathBuf,
    graph: xencode_core_rs::TaskGraph,
    scheduler: xencode_core_rs::Scheduler,
    waves: Vec<Vec<xencode_core_rs::NodeId>>,
    critical: Vec<xencode_core_rs::NodeId>,
    bottleneck: Option<xencode_core_rs::NodeId>,
    fingerprint: String,
    estimate: Option<xencode_core_rs::Estimate>,
    runs_dir: std::path::PathBuf,
    tariff: Option<f64>,
    /// The posture this plan was read under, and the roles it refuses. A plan is
    /// still a read, so it shows the whole team and marks what would be declined;
    /// only `xencode team run` acts on the refusals.
    profile: xencode_core_rs::Profile,
    refused: Vec<xencode_core_rs::WorkerRefusal>,
}

impl PlannedRecipe {
    /// The refusal a role's worker drew, if it drew one.
    fn refusal_for(&self, worker: &str) -> Option<&xencode_core_rs::WorkerRefusal> {
        self.refused.iter().find(|r| r.worker == worker)
    }
}

fn plan_recipe(
    files: &[xencode_core_rs::RecipeFile],
    dir: &std::path::Path,
    name: &str,
) -> Result<PlannedRecipe, String> {
    let file = find_recipe(files, dir, name)?;
    let recipe = file.recipe.as_ref().unwrap();
    // Every structural fault — a role with no name, a need that names nothing, a
    // cycle — is caught here, before anything could be launched.
    let graph = recipe
        .to_task_graph()
        .map_err(|e| format!("{e}\nnothing was launched"))?;
    let runs_dir = team_runs_dir();
    let runs = xencode_core_rs::load_runs(&runs_dir).map_err(|e| {
        format!(
            "cannot read the recorded runs in {}: {e}",
            runs_dir.display()
        )
    })?;
    let fingerprint = xencode_core_rs::recipe_fingerprint(recipe);
    let estimate = xencode_core_rs::estimate_from_runs(&runs, &fingerprint);
    // The posture is read once, here, so the plan view, the JSON and the refusal
    // at run time all quote the same two settings rather than each loading a
    // config and possibly disagreeing (`OR-13`).
    let config = XencodeConfig::load().unwrap_or_default();
    let mut considered: Vec<(String, bool)> = Vec::new();
    for role in &recipe.roles {
        let answer = (
            role.worker.clone(),
            xencode_agents_rs::is_external_worker(&role.worker),
        );
        if !considered.contains(&answer) {
            considered.push(answer);
        }
    }
    let refused = config.profile().refused_workers(&considered);
    Ok(PlannedRecipe {
        waves: readiness_waves(&graph),
        critical: graph.longest_chain().unwrap_or_default(),
        bottleneck: graph.bottleneck().map(|node| node.id.clone()),
        scheduler: recipe.scheduler(),
        fingerprint,
        estimate,
        runs_dir,
        recipe: recipe.clone(),
        file: file.path.clone(),
        graph,
        tariff: config.power_cents_per_kwh,
        profile: config.profile(),
        refused,
    })
}

/// The plan view, in the words a person reads. `awaiting_approval` adds the line
/// that says what would have to be true for any of it to actually run, which is
/// the only difference between this and `xencode team plan`.
fn print_plan(planned: &PlannedRecipe, awaiting_approval: bool) {
    println!(
        "Recipe {} would run {} roles at most {} at once, limited by {}.",
        planned.recipe.name,
        planned.recipe.roles.len(),
        planned.scheduler.capacity(),
        planned.scheduler.binding().label(),
    );
    println!("Nothing was launched and no check was run — this is the plan only.");
    // Only the worker rule is enforced here, so only the worker rule is quoted:
    // a model route is decided by the egress policy where the prompt is sent, not
    // by a plan that launches nothing.
    println!(
        "Posture: {} — {}",
        planned.profile.name(),
        planned.profile.worker_rule()
    );
    for (n, wave) in planned.waves.iter().enumerate() {
        for (i, id) in wave.iter().enumerate() {
            let role = planned
                .recipe
                .roles
                .iter()
                .find(|r| &r.name == id)
                .expect("the graph was built from these roles");
            println!(
                "\n  {} wave {}  {}",
                if i == 0 { "▸" } else { " " },
                n + 1,
                role.name
            );
            println!("      worker:   {}", worker_status(role));
            if let Some(refusal) = planned.refusal_for(&role.worker) {
                println!("      refused:  {}", refusal.why);
            }
            println!("      gate:     {}", gate_words(role));
            println!("      needs:    {}", needs_words(role));
            println!("      command:  {}", role.command);
        }
    }
    println!();
    if planned.critical.is_empty() {
        println!("  critical path: none");
    } else {
        println!(
            "  critical path: {} ({} roles) — the shortest wall clock no number of workers can \
             beat",
            planned.critical.join(" → "),
            planned.critical.len()
        );
    }
    match &planned.bottleneck {
        Some(id) => println!(
            "  serial bottleneck: {id} — it waits on more than one role and sits on that path, \
             so it is where the branches are forced back into one line"
        ),
        None => println!(
            "  serial bottleneck: none — no role waits on more than one other role on the \
             longest path"
        ),
    }
    println!();
    match &planned.estimate {
        Some(estimate) => {
            println!(
                "  estimated wall clock: {}, at most {} at once — measured from run {} \
                 (approved by {}, {} roles)",
                wall_clock_label(estimate.wall_clock_ms),
                estimate.peak_concurrency,
                estimate.run_id,
                estimate.approved_by,
                estimate.roles,
            );
            println!(
                "  estimated cost:       {}",
                energy_label(
                    estimate.watt_hours,
                    estimate.cost_micros(planned.tariff),
                    planned.tariff,
                )
            );
            println!(
                "  Both are measurements of one past run on this machine, not a promise: what \
                 the roles do from now on is their own."
            );
        }
        None => {
            println!(
                "  estimated wall clock: unknown — this recipe has never run here, so there is \
                 no measurement to quote."
            );
            println!(
                "  estimated cost:       unknown for the same reason. A run records what it \
                 took; the next plan of this recipe can then quote it."
            );
        }
    }
    println!(
        "\n  The gates above are named, not run. xencode verify is what runs those three checks."
    );
    if !planned.refused.is_empty() {
        let roles = planned
            .recipe
            .roles
            .iter()
            .filter(|r| planned.refusal_for(&r.worker).is_some())
            .count();
        println!(
            "\n  {roles} of {} {} assigned to {} the posture above refuses, and a team is \
             not run with roles quietly dropped: `xencode team run {}` would launch nothing at \
             all while it stands. The posture line above names the setting that opens the rule; \
             the other way is to give those roles a worker the posture does not refuse.",
            plural_count(planned.recipe.roles.len(), "role", "roles"),
            if roles == 1 { "is" } else { "are" },
            if planned.refused.len() == 1 {
                "an agent"
            } else {
                "agents"
            },
            planned.recipe.name,
        );
    }
    if awaiting_approval {
        println!(
            "\n  Nothing above has been launched, and nothing will be: run\n     xencode team \
             run {} --approved-by <your name>\n  to approve it. That prints this plan again, \
             then runs the roles as written{}.",
            planned.recipe.name,
            if planned.refused.is_empty() {
                String::new()
            } else {
                " — or refuses the whole team, for the reason above".to_string()
            }
        );
    }
}

/// The plan as JSON, including the estimate and the recipe's fingerprint — the
/// identity an estimate is tied to. `launches` is false for a plan and for an
/// unapproved run, and true only for a run that really ran.
fn plan_json(planned: &PlannedRecipe, launches: bool) -> serde_json::Value {
    serde_json::json!({
        "recipe": planned.recipe.name,
        "file": planned.file.display().to_string(),
        "capacity": planned.scheduler.capacity(),
        "binding": planned.scheduler.binding().label(),
        "posture": planned.profile.name(),
        "posture_rules": planned.profile.rules(),
        "launches": launches,
        "waves": planned.waves,
        "critical_path": planned.critical,
        "bottleneck": planned.bottleneck,
        "fingerprint": planned.fingerprint,
        "estimate": planned.estimate.as_ref().map(|estimate| serde_json::json!({
            "run_id": estimate.run_id,
            "approved_by": estimate.approved_by,
            "started_at_unix_ms": estimate.started_at_unix_ms,
            "wall_clock_ms": estimate.wall_clock_ms,
            "peak_concurrency": estimate.peak_concurrency,
            "roles": estimate.roles,
            "watt_hours": estimate.watt_hours,
            "cents_per_kwh": planned.tariff,
            "cost_micros": estimate.cost_micros(planned.tariff),
        })),
        "roles": planned.recipe.roles.iter().map(|role| serde_json::json!({
            "name": role.name,
            "worker": role.worker,
            "worker_status": worker_status(role),
            "refused_by_posture": planned
                .refusal_for(&role.worker)
                .map(|refusal| refusal.why.clone()),
            "gate": role.gate,
            "needs": role.needs,
            "command": role.command,
        })).collect::<Vec<_>>(),
    })
}

/// Launch the roles for real, price what they took, and record it. Every child
/// here is an `sh -c` process this command waits on through `OR-2`'s queue, and
/// every number in the record is one that run produced.
async fn run_recipe(
    planned: &PlannedRecipe,
    approved_by: &str,
    format: OutputFormat,
) -> Result<(), String> {
    let started_at_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut manager = xencode_core_rs::TaskManager::new();
    let window = xencode_context_rs::power::PowerWindow::begin();
    let report = planned
        .scheduler
        .run(&planned.graph, &mut manager)
        .await
        .map_err(|e| format!("{e}\nnothing was launched"))?;
    let use_ = window.finish();
    let record = xencode_core_rs::TeamRun::from_report(
        &planned.recipe,
        &report,
        &xencode_core_rs::RunObservation {
            recipe_file: &planned.file,
            fingerprint: &planned.fingerprint,
            approved_by,
            started_at_unix_ms,
            watt_hours: use_.total_watt_hours(),
            cents_per_kwh: planned.tariff,
        },
    );
    let path = record.write(&planned.runs_dir)?;
    let failed: Vec<&xencode_core_rs::RoleRun> = record
        .roles
        .iter()
        .filter(|role| role.status != "exited(0)")
        .collect();

    if matches!(format, OutputFormat::Json) {
        let mut out = plan_json(planned, true);
        out["approved_by"] = serde_json::Value::String(approved_by.to_string());
        out["actual"] = serde_json::json!({
            "run_id": record.run_id,
            "wall_clock_ms": record.elapsed_ms,
            "peak_concurrency": record.peak_concurrency,
            "roles": record.roles,
            "watt_hours": record.watt_hours,
            "cents_per_kwh": record.cents_per_kwh,
            "cost_micros": record.cost_micros(planned.tariff),
            "failed_roles": failed.iter().map(|r| r.name.clone()).collect::<Vec<_>>(),
        });
        out["record"] = serde_json::Value::String(path.display().to_string());
        println!(
            "{}",
            serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
        );
    } else {
        println!();
        for role in &record.roles {
            println!(
                "  ▸ {:<14} {}  started {} ms, finished {} ms",
                role.name, role.status, role.started_ms, role.finished_ms
            );
        }
        println!();
        println!(
            "  wall clock:      {} ({} ms), peak concurrency {} of {} at once",
            wall_clock_label(record.elapsed_ms),
            record.elapsed_ms,
            record.peak_concurrency,
            planned.scheduler.capacity(),
        );
        println!(
            "  this run cost:   {}",
            energy_label(
                record.watt_hours,
                record.cost_micros(planned.tariff),
                planned.tariff,
            )
        );
        match &planned.estimate {
            Some(estimate) => {
                println!(
                    "  the estimate was {} · {}",
                    wall_clock_label(estimate.wall_clock_ms),
                    energy_label(
                        estimate.watt_hours,
                        estimate.cost_micros(planned.tariff),
                        planned.tariff,
                    ),
                );
                println!(
                    "  checked:         {}",
                    difference_label(record.elapsed_ms, estimate.wall_clock_ms)
                );
            }
            None => println!(
                "  checked:         nothing — this recipe had never run here before now. These \
                 numbers are what the next plan of it will quote."
            ),
        }
        println!("\n  Recorded in {}", path.display());
        if failed.is_empty() {
            println!("  Every role exited 0. The gates the recipe names were still not run.");
        } else {
            println!(
                "  {} of {} roles did not exit 0: {}. The gates were not run.",
                failed.len(),
                record.roles.len(),
                failed
                    .iter()
                    .map(|role| format!("{} ({})", role.name, role.status))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} of {} roles did not exit 0: {}",
            failed.len(),
            record.roles.len(),
            failed
                .iter()
                .map(|role| role.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

/// A wall clock as the machine's own power line says it, so a plan and a run
/// quote time in the same units a person already reads elsewhere in xencode.
fn wall_clock_label(ms: u64) -> String {
    xencode_context_rs::power::elapsed_label(std::time::Duration::from_millis(ms))
}

/// One read of everything the orchestrator's verbs report on, taken once per
/// command so two lines of the same screen cannot quote different counts of the
/// same directory. `OR-12` set this rule for the panel; the CLI is the same
/// reader with a keyboard taken out of it.
///
/// Every field is what was *found*, and a directory that could not be read is
/// kept as the problem rather than as an empty list: a fleet command that said
/// "no runs recorded" when it failed to open the folder would be reporting an
/// absence it did not observe.
struct Fleet {
    xencode_dir: std::path::PathBuf,
    profile: xencode_core_rs::Profile,
    teams_dir: std::path::PathBuf,
    recipes: Vec<xencode_core_rs::RecipeFile>,
    recipes_problem: Option<String>,
    runs_dir: std::path::PathBuf,
    runs: Vec<xencode_core_rs::RunFile>,
    runs_problem: Option<String>,
    tasks: Result<
        std::collections::BTreeMap<u64, (xencode_core_rs::FileTask, xencode_core_rs::TaskStatus)>,
        String,
    >,
    detached: Vec<String>,
    ledgers: Vec<xencode_context_rs::RunRecord>,
}

impl Fleet {
    fn read() -> Fleet {
        let xencode_dir = project_xencode_dir();
        let teams_dir = xencode_dir.join(xencode_core_rs::RECIPES_DIR);
        let runs_dir = xencode_dir.join(xencode_core_rs::RUNS_DIR);
        let recipes = match xencode_core_rs::load_recipes(&teams_dir) {
            Ok(files) => files,
            Err(e) => {
                return Fleet {
                    recipes_problem: Some(format!("{}: {e}", teams_dir.display())),
                    xencode_dir,
                    profile: XencodeConfig::load().unwrap_or_default().profile(),
                    teams_dir,
                    recipes: Vec::new(),
                    runs_dir,
                    runs: Vec::new(),
                    runs_problem: None,
                    tasks: Ok(std::collections::BTreeMap::new()),
                    detached: Vec::new(),
                    ledgers: Vec::new(),
                }
            }
        };
        let runs = match xencode_core_rs::load_runs(&runs_dir) {
            Ok(files) => (files, None),
            Err(e) => (
                Vec::new(),
                Some(format!(
                    "cannot read the recorded runs in {}: {e}",
                    runs_dir.display()
                )),
            ),
        };
        Fleet {
            runs_problem: runs.1,
            runs: runs.0,
            recipes_problem: None,
            runs_dir,
            profile: XencodeConfig::load().unwrap_or_default().profile(),
            teams_dir,
            recipes,
            xencode_dir,
            tasks: tasks_registry().poll().map_err(|e| e.to_string()),
            detached: detached_runs::list_run_ids(&project_xencode_dir()),
            ledgers: xencode_context_rs::read_runs(&project_xencode_dir()),
        }
    }

    /// The newest recorded run of this exact recipe, for a screen that can say
    /// where each node ended the last time this graph was actually walked.
    fn newest_run_matching(&self, fingerprint: &str) -> Option<&xencode_core_rs::TeamRun> {
        self.runs
            .iter()
            .filter_map(|file| file.run.as_ref().ok())
            .filter(|run| run.fingerprint == fingerprint)
            .max_by_key(|run| run.started_at_unix_ms)
    }

    /// The newest recorded run, if there is one whose file could be read.
    fn newest_run(&self) -> Option<&xencode_core_rs::TeamRun> {
        self.runs
            .iter()
            .filter_map(|file| file.run.as_ref().ok())
            .max_by_key(|run| run.started_at_unix_ms)
    }

    /// How many recorded runs this fingerprint has, and how many of them ended
    /// with something other than an exit 0. Both are counts of files, not
    /// judgements about the work.
    fn runs_for(&self, fingerprint: &str) -> (usize, Vec<String>) {
        let mut n = 0;
        let mut failed = Vec::new();
        for file in &self.runs {
            if let Ok(run) = file.run.as_ref() {
                if run.fingerprint == fingerprint {
                    n += 1;
                    for role in &run.roles {
                        if role.status != "exited(0)" {
                            failed
                                .push(format!("{} · {} · {}", run.run_id, role.name, role.status));
                        }
                    }
                }
            }
        }
        (n, failed)
    }
}

/// The `OR-14` control surface. Each verb reads the state above, or acts on one
/// process this machine started; none of them writes a config value, and the only
/// ones that write anything at all are `retry` — which launches a real child — and
/// `stop`, which kills one.
async fn run_orchestrator(action: OrchestratorAction) -> Result<(), String> {
    match action {
        OrchestratorAction::Status { format } => orchestrator_status(format),
        OrchestratorAction::Agents { format } => orchestrator_agents(format),
        OrchestratorAction::Tasks { format } => orchestrator_tasks(format),
        OrchestratorAction::Graph { recipe, format } => orchestrator_graph(recipe, format),
        OrchestratorAction::Split {
            task,
            commit,
            paths,
            answer,
            verification,
            model,
            format,
        } => orchestrator_split(task, commit, paths, answer, verification, model, format).await,
        OrchestratorAction::Logs { run, lines, format } => orchestrator_logs(run, lines, format),
        OrchestratorAction::Permissions { agent, format } => {
            orchestrator_permissions(agent, format)
        }
        OrchestratorAction::Costs { format } => orchestrator_costs(format),
        OrchestratorAction::Inspect { target, format } => orchestrator_inspect(&target, format),
        OrchestratorAction::Retry {
            recipe,
            role,
            approved_by,
            format,
        } => orchestrator_retry(&recipe, &role, approved_by.as_deref(), format).await,
        OrchestratorAction::Stop { target } => orchestrator_stop(&target),
        OrchestratorAction::Attach { agent, target } => {
            orchestrator_attach(&agent, target.as_deref())
        }
    }
}

/// The closing line every reading shares: what to type next, and the fact that
/// nothing here changed anything.
fn orchestrator_ran_nothing() {
    println!(
        "\n  Nothing was launched, stopped or changed by this reading. `xencode team run \
         <recipe> --approved-by <name>` runs a recipe; `/orchestrator on` turns the same state \
         into a mode inside xencode's own screen."
    );
}

fn orchestrator_status(format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    let running = match &fleet.tasks {
        Ok(entries) => Some(
            entries
                .values()
                .filter(|(_, status)| matches!(status, xencode_core_rs::TaskStatus::Running))
                .count(),
        ),
        Err(_) => None,
    };
    let refused = refused_names(&fleet.profile);
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "posture": fleet.profile.name(),
                "rules": fleet.profile.rules(),
                "refused_agents": refused,
                "teams_dir": fleet.teams_dir.display().to_string(),
                "recipes": fleet.recipes.len(),
                "recipes_unreadable": fleet.recipes.iter().filter(|f| f.recipe.is_err()).count(),
                "recipes_problem": fleet.recipes_problem,
                "runs_dir": fleet.runs_dir.display().to_string(),
                "recorded_runs": fleet.runs.len(),
                "runs_problem": fleet.runs_problem,
                "newest_run": fleet.newest_run().map(|run| serde_json::json!({
                    "run_id": run.run_id, "recipe": run.recipe, "approved_by": run.approved_by,
                    "elapsed_ms": run.elapsed_ms, "roles": run.roles.len(),
                })),
                "background_tasks": running,
                "tasks_problem": fleet.tasks.as_ref().err(),
                "detached_runs": fleet.detached.len(),
                "approval_records": fleet.ledgers.len(),
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!(
        "Orchestrator status · read from {}",
        fleet.xencode_dir.display()
    );
    // One label column for the whole screen, so the eight readings below line up
    // and a difference between two of them is a difference in the state.
    let field = |label: &str, value: String| println!("  {:<21}{}", format!("{label}:"), value);
    field(
        "work may go to",
        format!("{} — {}", fleet.profile.name(), fleet.profile.worker_rule()),
    );
    field("models may go to", fleet.profile.model_rule());
    if !refused.is_empty() {
        field(
            "roster refused",
            format!(
                "{} of the {} agents on the roster ({})",
                refused.len(),
                xencode_agents_rs::ROSTER.len(),
                refused.join(", ")
            ),
        );
    }
    match &fleet.recipes_problem {
        Some(problem) => field("recipes", format!("could not be read — {problem}")),
        None if fleet.recipes.is_empty() => field(
            "recipes",
            format!(
                "none in {} — `xencode team run` has nothing to plan until a `.toml` is put \
                 there",
                fleet.teams_dir.display()
            ),
        ),
        None => {
            let faulty = fleet.recipes.iter().filter(|f| f.recipe.is_err()).count();
            field(
                "recipes",
                format!(
                    "{} in {}{}",
                    fleet.recipes.len(),
                    fleet.teams_dir.display(),
                    if faulty == 0 {
                        String::new()
                    } else {
                        format!(", {faulty} of them unreadable as a recipe")
                    }
                ),
            );
        }
    }
    match &fleet.runs_problem {
        Some(problem) => field("recorded runs", format!("could not be read — {problem}")),
        None if fleet.runs.is_empty() => field(
            "recorded runs",
            format!(
                "none in {} — every figure below is therefore unmeasured",
                fleet.runs_dir.display()
            ),
        ),
        None => match fleet.newest_run() {
            Some(run) => field(
                "recorded runs",
                format!(
                    "{} · newest {} (recipe {}, {} roles, approved by {}, {})",
                    fleet.runs.len(),
                    run.run_id,
                    run.recipe,
                    run.roles.len(),
                    run.approved_by,
                    wall_clock_label(run.elapsed_ms),
                ),
            ),
            None => field(
                "recorded runs",
                format!(
                    "{} files in {}, none of them readable as a run",
                    fleet.runs.len(),
                    fleet.runs_dir.display()
                ),
            ),
        },
    }
    match running {
        Some(n) => field(
            "background tasks",
            format!(
                "{} running of {} in the registry",
                n,
                fleet
                    .tasks
                    .as_ref()
                    .map(|entries| entries.len())
                    .unwrap_or_default()
            ),
        ),
        None => field(
            "background tasks",
            "unknown — the registry could not be read; a task may be running that this command \
             cannot see"
                .to_string(),
        ),
    }
    field(
        "detached runs",
        format!(
            "{}{}",
            fleet.detached.len(),
            match fleet.detached.last() {
                Some(id) => format!(
                    " — newest is {}, `xencode orchestrator logs {id}` reads its log",
                    truncate_id(id)
                ),
                None => " — none in `cache/detached`".to_string(),
            }
        ),
    );
    field(
        "approvals on record",
        format!(
            "{} runs in {}, each carrying the tool calls a person answered",
            fleet.ledgers.len(),
            xencode_context_rs::runs_path(&fleet.xencode_dir).display()
        ),
    );
    println!(
        "\n  This is the same state xencode's `/orchestrator` mode shows; the mode lives in a \
         running session and is not a setting, so nothing here turns it on for the next one."
    );
    Ok(())
}

/// The roster names the posture would refuse work handed to.
fn refused_names(profile: &xencode_core_rs::Profile) -> Vec<String> {
    let considered: Vec<(String, bool)> = xencode_agents_rs::ROSTER
        .iter()
        .map(|spec| (spec.name.to_string(), true))
        .collect();
    profile
        .refused_workers(&considered)
        .into_iter()
        .map(|refusal| refusal.worker)
        .collect()
}

/// A run id is long and machine-made; a screen quoting it can show less of it.
fn truncate_id(id: &str) -> String {
    if id.chars().count() <= 12 {
        return id.to_string();
    }
    format!("{}…", id.chars().take(12).collect::<String>())
}

fn orchestrator_agents(format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    let refused = refused_names(&fleet.profile);
    let rows: Vec<serde_json::Value> = xencode_agents_rs::ROSTER
        .iter()
        .map(|spec| {
            let installed = spec
                .binaries
                .iter()
                .find_map(|b| xencode_agents_rs::roster::which(b));
            serde_json::json!({
                "agent": spec.name,
                "installed": installed.is_some(),
                "found_at": installed.map(|p| p.display().to_string()),
                "refused_by_posture": refused.iter().any(|n| n == spec.name),
                "handover": spec.attach,
                "session_listing": spec.session_list,
                "one_shot": spec.one_shot,
                "cells_read_on": spec.read_on,
                "parked": spec.parked,
            })
        })
        .collect();
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "posture": fleet.profile.name(),
                "worker_rule": fleet.profile.worker_rule(),
                "agents": rows,
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!(
        "{:<14} {:<10} {:<9} {:<34} WHAT XENCODE RUNS IT WITH",
        "AGENT", "INSTALLED", "POSTURE", "HANDOVER"
    );
    println!("{}", "-".repeat(102));
    for spec in xencode_agents_rs::ROSTER {
        let installed = spec
            .binaries
            .iter()
            .find_map(|b| xencode_agents_rs::roster::which(b));
        let handover = match spec.attach {
            Some(command) => command.to_string(),
            None => "no attach verb read from its help".to_string(),
        };
        println!(
            "{:<14} {:<10} {:<9} {:<34} {}",
            spec.name,
            if installed.is_some() { "yes" } else { "no" },
            if refused.iter().any(|n| n == spec.name) {
                "refused"
            } else {
                "allowed"
            },
            handover,
            spec.one_shot,
        );
    }
    println!(
        "\n  POSTURE is the {} profile's answer to work being handed to this agent, read from \
         its roster row — being on the row is the whole reason. It is not a measurement of what \
         the agent did.",
        fleet.profile.name()
    );
    println!(
        "  HANDOVER is a command the vendor's own help documents as taking over a session that \
         is already running, which is all `xencode orchestrator attach` will ever run. {} of the \
         {} rows here have none, and saying so is the point.",
        xencode_agents_rs::ROSTER
            .iter()
            .filter(|spec| spec.attach.is_none())
            .count(),
        xencode_agents_rs::ROSTER.len()
    );
    println!(
        "  Every cell was read from `--help` on the date the row records; nothing here is a \
         claim that a probe ran the agent. `xencode agents --health` is what asks a program \
         whether it answers."
    );
    orchestrator_ran_nothing();
    Ok(())
}

/// The two lists of processes xencode started here and still knows about. They
/// live in different directories, are reaped differently and outlive different
/// things, so they are printed apart: one merged count would hide which of the
/// two a stopped process belongs to.
///
/// A registry that could not be read is reported as unreadable. "No tasks" is a
/// claim about the machine, and this command is not owed one it did not observe.
fn orchestrator_tasks(format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    let tasks_dir = fleet.xencode_dir.join("tasks");
    let detached_dir = xencode_tui_rs::detached::detached_dir(&fleet.xencode_dir);
    let detached: Vec<(String, String, usize, String)> = fleet
        .detached
        .iter()
        .map(|id| {
            let dir = xencode_tui_rs::detached::run_dir(&fleet.xencode_dir, id);
            (
                id.clone(),
                xencode_tui_rs::detached::derive_status(&dir)
                    .label()
                    .to_string(),
                xencode_tui_rs::detached::read_rounds(&dir).len(),
                xencode_tui_rs::detached::read_spec(&dir)
                    .map(|spec| spec.model)
                    .unwrap_or_else(|| "no spec".to_string()),
            )
        })
        .collect();

    if matches!(format, OutputFormat::Json) {
        let registry = match &fleet.tasks {
            Ok(entries) => serde_json::Value::Array(
                entries
                    .values()
                    .map(|(task, status)| {
                        serde_json::json!({
                            "id": task.id,
                            "name": task.name,
                            "command": task.command,
                            "pid": task.pid,
                            "status": status.label(),
                            "killed": task.killed,
                            "source": task.source,
                            "started_at_unix_secs": task.started_at,
                        })
                    })
                    .collect(),
            ),
            Err(problem) => serde_json::json!({ "unreadable": problem }),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "tasks_dir": tasks_dir.display().to_string(),
                "tasks": registry,
                "detached_dir": detached_dir.display().to_string(),
                "detached_runs": detached.iter().map(|(id, status, rounds, model)|
                    serde_json::json!({
                        "run_id": id, "status": status, "rounds": rounds, "model": model,
                    })).collect::<Vec<_>>(),
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Background tasks · {}", tasks_dir.display());
    match &fleet.tasks {
        Err(problem) => println!(
            "  could not be read — {problem}. Nothing is claimed about running tasks: one may \
             be alive that this command cannot see."
        ),
        Ok(entries) if entries.is_empty() => {
            println!("  none recorded — the registry answered the read and held no rows.")
        }
        Ok(entries) => {
            println!(
                "{:>4}  {:<12} {:>8}  {:<12}  NAME",
                "ID", "STATUS", "PID", "SOURCE"
            );
            for (task, status) in entries.values() {
                println!(
                    "{:>4}  {:<12} {:>8}  {:<12}  {}",
                    task.id,
                    status.label(),
                    task.pid,
                    task.source.as_deref().unwrap_or("cli"),
                    task.name
                );
            }
            println!(
                "\n  STATUS is what the registry concluded by asking the operating system whether \
                 that pid is alive, so a task whose child was reaped elsewhere reads as ended."
            );
            println!("  `xencode tasks poll <id>` reads a row's output; `xencode orchestrator inspect <id>` shows the row with its last lines.");
        }
    }

    println!("\nDetached runs · {}", detached_dir.display());
    if detached.is_empty() {
        println!("  none — `xencode run \"the task\" --detach` starts one.");
    } else {
        for (id, status, rounds, model) in &detached {
            println!(
                "  {}  {:<10} {:>3} round(s)  {}",
                truncate_id(id),
                status,
                rounds,
                model
            );
        }
        println!(
            "\n  A detached run is xencode's own agent loop in a forked child; the rows above are \
             read from its spec, its round log and its exit file, never from the child's memory."
        );
    }
    orchestrator_ran_nothing();
    Ok(())
}

/// The Cargo workspace a split is judged inside: the nearest directory at or above
/// here whose manifest declares a `[workspace]`, because that is the tree whose
/// crates `cargo metadata` names. There is no `--root` to point it somewhere else,
/// on purpose — a reference built from the wrong tree is not a rough reading, it is
/// a wrong answer, and the only sign would be a score nobody questions.
fn split_workspace_root() -> Result<std::path::PathBuf, String> {
    let start = std::env::current_dir()
        .map_err(|e| format!("cannot read the working directory to find the workspace: {e}"))?;
    let mut dir = start.as_path();
    loop {
        let manifest = dir.join("Cargo.toml");
        let declares_workspace = std::fs::read_to_string(&manifest)
            .map(|text| text.contains("[workspace]"))
            .unwrap_or(false);
        if declares_workspace {
            return Ok(dir.to_path_buf());
        }
        dir = dir
            .parent()
            .ok_or_else(|| format!("no Cargo workspace at or above {}", start.display()))?;
    }
}

/// `OR-1` — ask for a split, score it against what the build itself requires, and
/// refuse it here rather than handing it to a scheduler.
///
/// The order of operations is the whole item. The reference is built *before*
/// anything is asked, and out of something other than the answer whenever a person
/// can name one: `--commit` gives the change's own message as the task and the
/// files git says it touched as the file set; `--path` gives a list somebody
/// wrote. With neither, the split can only be compared on ordering, and the report
/// says which half was actually measured instead of printing full coverage as if it
/// had been earned.
///
/// A refused split comes back as an error, so it exits non-zero and stops a script
/// where the warning would have scrolled past.
async fn orchestrator_split(
    task: Option<String>,
    commit: Option<String>,
    paths: Vec<String>,
    answer: Option<std::path::PathBuf>,
    verification: String,
    model: Option<String>,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_core_rs::{decide, Reference, Split};
    use xencode_tui_rs::decompose::{
        ask_planner, commit_change, reference_from_paths, reference_from_split, workspace_layout,
        PlannerOptions,
    };

    let root = split_workspace_root()?;
    let change = match &commit {
        Some(sha) => Some(commit_change(&root, sha)?),
        None => None,
    };
    let task_text = match &change {
        Some(read) => read.task(),
        None => task.filter(|text| !text.trim().is_empty()).ok_or_else(|| {
            "nothing to split: name the change with --task \"…\", or take a change that \
                 already happened with --commit <sha>"
                .to_string()
        })?,
    };

    // The split: read from a file when one is given, which needs no server and no
    // spend, and otherwise asked of the model just now.
    let (split, asked) = match &answer {
        Some(file) => {
            let text = std::fs::read_to_string(file)
                .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            let value = match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(value) => value,
                // A saved raw answer usually carries the model's prose around the
                // JSON, so the same reader the live path uses gets the first shot
                // at it before this calls the file unreadable.
                Err(e) => xencode_providers_rs::schema::read_answer(
                    &text,
                    &xencode_tui_rs::decompose::schema(),
                )
                .map_err(|why| format!("{} is not a split ({e}): {why}", file.display()))?,
            };
            let split = Split::from_value(&value)
                .map_err(|e| format!("{} does not hold a split: {e}", file.display()))?;
            (split, None)
        }
        None => {
            let config = XencodeConfig::load().unwrap_or_default();
            let options = PlannerOptions {
                ollama_url: config.ollama_url.clone(),
                llama_cpp_url: config.llama_cpp_url.clone(),
                temperature: config.llama_cpp_temperature.unwrap_or(0.0),
                seed: config.llama_cpp_seed.unwrap_or(42),
                ..PlannerOptions::default()
            };
            let chosen = model.unwrap_or(config.default_model.clone());
            let layout = workspace_layout(&root)?;
            let run = ask_planner(&task_text, &layout, &chosen, &options).await?;
            (run.split.clone(), Some(run))
        }
    };

    let reference: Reference = match (&change, paths.is_empty()) {
        (Some(read), _) => reference_from_paths(&root, &read.paths, &verification, &read.source())?,
        (None, false) => reference_from_paths(
            &root,
            &paths,
            &verification,
            &format!("{} files named with --path", paths.len()),
        )?,
        (None, true) => reference_from_split(&split, &root, &verification)?,
    };
    let flat = Split::flat_baseline(&reference);
    let decision = decide(&split, &reference, &flat);

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&split_json(&root, &reference, &split, &decision, &asked))
                .map_err(|e| e.to_string())?
        );
        return split_verdict(&decision);
    }

    println!("Split · {}", reference.source);
    println!("  workspace: {}", root.display());
    println!("  every unit graded by: {verification}");
    match &asked {
        Some(run) => println!(
            "  asked {} in {} ms, and got {} units back",
            run.model,
            run.elapsed_ms,
            run.split.subtasks.len()
        ),
        None => println!(
            "  read from a file, so no model was asked and nothing was spent: {}",
            answer
                .as_ref()
                .map(|f| f.display().to_string())
                .unwrap_or_default()
        ),
    }
    println!();
    println!("  the units a scheduler would run");
    for node in &split.subtasks {
        println!(
            "    {:<16} {} — {}",
            node.id,
            node.goal,
            if node.needs.is_empty() {
                "starts at once".to_string()
            } else {
                format!("waits on {}", node.needs.join(", "))
            }
        );
        println!("        writes {}", node.paths.join(", "));
    }
    println!();
    for line in decision.lines() {
        println!("{line}");
    }
    // The disagreements are not printed again here: every one that stops the split
    // is already named in the reasons above, and the pair counts below say how
    // many of each kind there were. The JSON keeps the lists themselves.
    for line in decision.score.counts() {
        println!("  {line}");
    }
    println!(
        "  the same figures for the baseline ({} units):",
        flat.subtasks.len()
    );
    for line in decision.baseline.counts() {
        println!("    {line}");
    }
    if reference.source.contains("the split's own file list") {
        println!(
            "\n  Only the ordering above is a measurement: the file list came from the split \
             itself, so nothing was compared on coverage. Name --commit <sha> or --path to get \
             that half too."
        );
    }
    println!();
    println!(
        "  Nothing was launched or scheduled by this reading. This screen decides how the work \
         would be cut up; `xencode team run <recipe>` is where a plan becomes processes."
    );
    split_verdict(&decision)
}

/// The refusal, in the words the exit code carries. A split that did not beat the
/// baseline is not a warning: the scheduler would run it, so this is the point
/// where it stops.
fn split_verdict(decision: &xencode_core_rs::SplitDecision) -> Result<(), String> {
    if decision.schedulable {
        return Ok(());
    }
    Err(format!(
        "this split is not scheduled: {}",
        decision.reasons.join("; ")
    ))
}

fn split_json(
    root: &std::path::Path,
    reference: &xencode_core_rs::Reference,
    split: &xencode_core_rs::Split,
    decision: &xencode_core_rs::SplitDecision,
    asked: &Option<xencode_tui_rs::decompose::PlannerRun>,
) -> serde_json::Value {
    let score = |s: &xencode_core_rs::SplitScore| {
        serde_json::json!({
            "files_expected": s.paths_expected,
            "files_owned": s.paths_owned,
            "coverage": s.coverage(),
            "orders_expected": s.pairs_expected,
            "orders_stated": s.agreed,
            "ordering": s.ordering(),
            "unclaimed": s.unclaimed,
            "contested": s.contested.iter().map(|(path, owners)| serde_json::json!({"path": path, "owners": owners})).collect::<Vec<_>>(),
            "backwards": s.contradicted,
            "left_silent": s.unstated,
            "inside_one_node": s.same_node,
            "not_in_the_change": s.invented,
        })
    };
    serde_json::json!({
        "workspace": root,
        "reference": {
            "source": reference.source,
            "verification": reference.verification,
            "paths": reference.paths,
            "before": reference.before,
        },
        "planner": asked.as_ref().map(|run| serde_json::json!({
            "model": run.model,
            "elapsed_ms": run.elapsed_ms,
            "prompt": run.prompt,
            "answer": run.raw,
        })),
        "units": split.subtasks.iter().map(|n| serde_json::json!({
            "id": n.id, "goal": n.goal, "paths": n.paths, "needs": n.needs, "verify": n.verify,
        })).collect::<Vec<_>>(),
        "schedulable": decision.schedulable,
        "reasons": decision.reasons,
        "score": score(&decision.score),
        "baseline": score(&decision.baseline),
    })
}

/// The dependency shape. With a recipe name it is the graph a run *would* walk,
/// beside where the last recorded run of that same recipe actually ended; with
/// none it is every run this project has recorded, in the same words the worker
/// panel uses, because the panel and this command read the same files.
fn orchestrator_graph(recipe: Option<String>, format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    match recipe {
        Some(name) => {
            let planned = plan_recipe(&fleet.recipes, &fleet.teams_dir, &name)?;
            let (runs, failed) = fleet.runs_for(&planned.fingerprint);
            let last = fleet.newest_run_matching(&planned.fingerprint);
            if matches!(format, OutputFormat::Json) {
                let mut out = plan_json(&planned, false);
                out["recorded_runs"] = serde_json::json!(runs);
                out["failed_roles_on_record"] = serde_json::json!(failed);
                out["last_run"] = last
                    .map(|run| {
                        serde_json::json!({
                            "run_id": run.run_id, "roles": run.roles,
                            "peak_concurrency": run.peak_concurrency, "elapsed_ms": run.elapsed_ms,
                        })
                    })
                    .unwrap_or(serde_json::Value::Null);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            println!(
                "Graph · {} ({})",
                planned.recipe.name,
                planned.file.display()
            );
            println!(
                "  capacity: {} at once, limited by {}",
                planned.scheduler.capacity(),
                planned.scheduler.binding().label()
            );
            for (n, wave) in planned.waves.iter().enumerate() {
                println!("  wave {}:  {}", n + 1, wave.join("  +  "));
                for id in wave {
                    let role = planned
                        .recipe
                        .roles
                        .iter()
                        .find(|r| &r.name == id)
                        .expect("the waves were built from these roles");
                    println!("      {} ← waits on {}", role.name, needs_words(role));
                }
            }
            println!(
                "  critical path: {}",
                if planned.critical.is_empty() {
                    "none".to_string()
                } else {
                    planned.critical.join(" → ")
                }
            );
            println!(
                "  bottleneck: {}",
                match &planned.bottleneck {
                    Some(id) => id.clone(),
                    None => "none — nothing forces the branches back into one line".to_string(),
                }
            );
            println!("\n  Recorded runs of this exact recipe: {runs}");
            match last {
                Some(run) => {
                    for role in &run.roles {
                        println!(
                            "      {:<14} {}  {} ms → {} ms",
                            role.name, role.status, role.started_ms, role.finished_ms
                        );
                    }
                    println!(
                        "    from run {} — peak {}, {}",
                        run.run_id,
                        run.peak_concurrency,
                        wall_clock_label(run.elapsed_ms)
                    );
                }
                None if runs == 0 => println!(
                    "    none, so nothing above has been measured on this machine. `xencode team \
                     plan {}` is the plan; the estimate line stays unknown until one runs.",
                    planned.recipe.name
                ),
                None => println!(
                    "    {runs} file(s) carry this fingerprint and none of them reads as a run."
                ),
            }
            if !failed.is_empty() {
                println!("\n  Roles that did not exit 0 on record:");
                for line in failed {
                    println!("    {line}");
                }
            }
            orchestrator_ran_nothing();
            Ok(())
        }
        None => {
            let rows = xencode_tui_rs::worker_panel::graph_rows(&fleet.runs, &fleet.runs_dir);
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "runs_dir": fleet.runs_dir.display().to_string(),
                        "runs": fleet.runs.iter().map(|file| serde_json::json!({
                            "path": file.path.display().to_string(),
                            "run": file.run.as_ref().ok().map(|run| serde_json::json!({
                                "run_id": run.run_id, "recipe": run.recipe,
                                "launch_order": run.launch_order, "roles": run.roles,
                                "peak_concurrency": run.peak_concurrency,
                                "capacity": run.capacity, "binding": run.binding,
                                "elapsed_ms": run.elapsed_ms,
                            })),
                            "problem": file.run.as_ref().err().map(|e| e.to_string()),
                        })).collect::<Vec<_>>(),
                    }))
                    .map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            println!("Recorded graphs · {}", fleet.runs_dir.display());
            for row in &rows {
                println!("  {}", row.line);
            }
            println!(
                "\n  One row per recorded run, oldest first. `xencode orchestrator inspect <run \
               id>` opens a row with the file each figure came from."
            );
            orchestrator_ran_nothing();
            Ok(())
        }
    }
}

/// What a run said while it went. A detached run keeps its own log file, so this
/// prints real bytes from disk; a team run keeps timings and exit statuses and
/// never captured its children's output, and says so rather than showing an
/// empty block as if the child had been quiet.
fn orchestrator_logs(
    run: Option<String>,
    lines: usize,
    format: OutputFormat,
) -> Result<(), String> {
    let fleet = Fleet::read();
    let Some(given) = run else {
        if matches!(format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "detached_runs": fleet.detached,
                    "team_runs": fleet.runs.iter().filter_map(|file| file.run.as_ref().ok())
                        .map(|run| serde_json::json!({"run_id": run.run_id, "recipe": run.recipe}))
                        .collect::<Vec<_>>(),
                }))
                .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        println!("Nothing was named, so here is what has a log to read:");
        if fleet.detached.is_empty() {
            println!(
                "  detached runs: none in {}",
                xencode_tui_rs::detached::detached_dir(&fleet.xencode_dir).display()
            );
        } else {
            println!("  detached runs (each keeps a log file):");
            for id in &fleet.detached {
                println!("    {}", id);
            }
        }
        let team: Vec<&xencode_core_rs::TeamRun> = fleet
            .runs
            .iter()
            .filter_map(|file| file.run.as_ref().ok())
            .collect();
        if team.is_empty() {
            println!("  team runs:     none in {}", fleet.runs_dir.display());
        } else {
            println!("  team runs (timings and exit status, no captured output):");
            for run in &team {
                println!("    {}  {}", run.run_id, run.recipe);
            }
        }
        println!(
            "\n  `xencode orchestrator logs <id>` reads one; the same id works for `inspect`."
        );
        return Ok(());
    };

    if let Some(id) = xencode_tui_rs::detached::resolve_run_id(&fleet.xencode_dir, &given) {
        let dir = xencode_tui_rs::detached::run_dir(&fleet.xencode_dir, &id);
        let tail = xencode_tui_rs::detached::read_log_tail(&dir, lines);
        if matches!(format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "kind": "detached", "run_id": id,
                    "log": xencode_tui_rs::detached::log_path(&dir).display().to_string(),
                    "lines": tail,
                }))
                .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        println!(
            "{} · {} line(s) from {}",
            id,
            tail.len(),
            xencode_tui_rs::detached::log_path(&dir).display()
        );
        for line in tail {
            println!("{line}");
        }
        return Ok(());
    }

    let file = fleet.runs.iter().find(|file| {
        file.run
            .as_ref()
            .ok()
            .is_some_and(|run| run.run_id == given || run.run_id.starts_with(given.as_str()))
    });
    match file {
        Some(file) => {
            let Some(run) = file.run.as_ref().ok() else {
                return Err(format!(
                    "{} is there and cannot be read as a run: {}",
                    file.path.display(),
                    file.run
                        .as_ref()
                        .err()
                        .map(|e| e.to_string())
                        .unwrap_or_default()
                ));
            };
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(run).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            println!("Team run {} · {}", run.run_id, file.path.display());
            println!("  recipe {} — approved by {}", run.recipe, run.approved_by);
            println!(
                "  launched: {}",
                if run.launch_order.is_empty() {
                    "nothing".to_string()
                } else {
                    run.launch_order.join(" → ")
                }
            );
            for role in &run.roles {
                println!(
                    "  {:<14} {:<12} {} ms → {} ms",
                    role.name, role.status, role.started_ms, role.finished_ms
                );
            }
            println!(
                "  {}",
                energy_label(
                    run.watt_hours,
                    run.cost_micros(run.cents_per_kwh),
                    run.cents_per_kwh
                )
            );
            println!(
                "\n  There is no output above because a team run never captured it: each role is a \
                 real `sh -c` child whose stdout went to the terminal that ran `xencode team \
                 run`. What the record keeps is when each role started, how it ended, and what \
                 the machine drew while it did so."
            );
            Ok(())
        }
        None => Err(format!(
            "no run {given} — not a detached run in {} and not a recorded team run in {}. \
             `xencode orchestrator logs` with no name lists both.",
            xencode_tui_rs::detached::detached_dir(&fleet.xencode_dir).display(),
            fleet.runs_dir.display()
        )),
    }
}

/// What a launch would be allowed to do, and what a person has already answered.
/// The first half is the broker's own decision on the config value this project
/// carries, computed by the same function a launch is built with; the second is
/// read off the run ledger, which is the only place an approval this project ever
/// gave is recorded.
fn orchestrator_permissions(agent: Option<String>, format: OutputFormat) -> Result<(), String> {
    use xencode_agents_rs::AgentSpec;
    use xencode_tui_rs::agent_tools::ApprovalMode;
    use xencode_tui_rs::permission_broker::{choose_grant, plan_launch, supports_prompt, Grant};
    let fleet = Fleet::read();
    let config = XencodeConfig::load().unwrap_or_default();
    let mode = ApprovalMode::parse(&config.agent_approval);
    let specs: Vec<&'static AgentSpec> = match &agent {
        Some(given) => vec![xencode_agents_rs::roster::find(given).ok_or_else(|| {
            format!(
                "no agent named {given} on the roster — `xencode orchestrator agents` lists the \
                 {} rows it holds",
                xencode_agents_rs::ROSTER.len()
            )
        })?],
        None => xencode_agents_rs::ROSTER.iter().collect(),
    };

    let mut rows = Vec::new();
    for spec in specs {
        // The roster's one-shot command, with the prompt slot dropped: what the
        // prompt says has nothing to do with what the launch may do.
        let base: Vec<String> = spec
            .one_shot
            .split_whitespace()
            .filter(|token| !token.contains("{prompt}"))
            .map(|token| token.to_string())
            .collect();
        let (argv, grant) = plan_launch(spec.name, mode, &base, &[], None);
        // The same launch with a worker asking for its own autonomy, run through
        // the same function, so the refusal to grant it is shown rather than
        // asserted.
        let self_granted: Vec<String> = vec![
            "--yolo".to_string(),
            "--permission-mode".to_string(),
            "bypassPermissions".to_string(),
        ];
        let (overruled, _) = plan_launch(spec.name, mode, &base, &self_granted, None);
        rows.push(PermissionRow {
            agent: spec.name,
            installed: spec
                .binaries
                .iter()
                .find_map(|b| xencode_agents_rs::roster::which(b))
                .is_some(),
            can_prompt_back: supports_prompt(spec.name),
            grant,
            argv,
            overruled,
        });
    }

    let asked: usize = fleet.ledgers.iter().map(|row| row.approvals.len()).sum();
    let granted: usize = fleet
        .ledgers
        .iter()
        .flat_map(|row| row.approvals.iter())
        .filter(|approval| approval.decision.granted())
        .count();
    let mut denied_by_tool: std::collections::BTreeMap<String, usize> = Default::default();
    for row in &fleet.ledgers {
        for approval in &row.approvals {
            if !approval.decision.granted() {
                *denied_by_tool.entry(approval.tool.clone()).or_insert(0) += 1;
            }
        }
    }
    let grant_words = Grant::words;

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "config_key": "agent_approval",
                "config_value": config.agent_approval,
                "mode": format!("{mode:?}"),
                "choose_grant_on_this_mode": format!("{:?}", choose_grant("claude", mode, Some("ask_user"))),
                "launches": rows.iter().map(|row| serde_json::json!({
                    "agent": row.agent,
                    "installed": row.installed,
                    "can_prompt_back": row.can_prompt_back,
                    "grant": format!("{:?}", row.grant),
                    "grant_words": grant_words(row.grant),
                    "argv": row.argv,
                    "argv_if_worker_asked_for_its_own": row.overruled,
                })).collect::<Vec<_>>(),
                "approval_history": {
                    "runs_on_record": fleet.ledgers.len(),
                    "calls_asked": asked,
                    "calls_allowed": granted,
                    "calls_denied": asked - granted,
                    "denied_by_tool": denied_by_tool,
                },
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    let config_file = XencodeConfig::config_path()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "the config file xencode could not locate".to_string());
    println!(
        "Permissions · agent_approval = \"{}\" → {:?}",
        config.agent_approval, mode
    );
    println!(
        "  That value is read from {config_file}, which is what a launch is built from. \
         Nothing here changes it."
    );
    println!(
        "\n{:<14} {:<11} {:<20} LAUNCH ARGUMENTS",
        "AGENT", "INSTALLED", "GRANT"
    );
    println!("{}", "-".repeat(96));
    for row in &rows {
        println!(
            "{:<14} {:<11} {:<20} {}",
            row.agent,
            if row.installed { "yes" } else { "no" },
            grant_words(row.grant),
            row.argv.join(" ")
        );
    }
    if rows.iter().any(|row| row.grant == Grant::Nothing) {
        println!(
            "\n  A `no autonomy flag` row is the strictest of the three grants: the worker keeps \
             its own prompting default and xencode adds nothing."
        );
    }
    if rows.iter().any(|row| row.can_prompt_back) {
        println!(
            "  The one route that would let a worker ask instead — a vendor whose approvals come \
             back to xencode through a prompt tool — needs a tool name from a running session, \
             which this command has none of, so it is not chosen here even for an agent that \
             could use it. `xencode agents` is where a session supplies one."
        );
    }
    let unchanged = rows.iter().filter(|row| row.overruled == row.argv).count();
    println!(
        "\n  The grant comes from xencode's mode and what the vendor can do, never from the \
         worker. Each row was then rebuilt the way a launch is, with the worker asking for its \
         own autonomy — `--yolo --permission-mode bypassPermissions` — and this is what that \
         launch would carry: {unchanged} of {} came back the line above, unchanged.",
        rows.len()
    );
    for row in &rows {
        if row.overruled == row.argv {
            println!(
                "    {:<14} unchanged — this line is what the mode grants; the worker's asking \
                 added nothing to it",
                row.agent
            );
        } else {
            println!("    {:<14} without: {}", row.agent, row.argv.join(" "));
            println!("    {:<14} with:    {}", "", row.overruled.join(" "));
        }
    }
    println!(
        "\n  Answered here, on record: {} run(s) in {}, {} tool call(s) asked, {} allowed, {} \
         denied.",
        fleet.ledgers.len(),
        xencode_context_rs::runs_path(&fleet.xencode_dir).display(),
        asked,
        granted,
        asked - granted
    );
    if !denied_by_tool.is_empty() {
        let top: Vec<String> = denied_by_tool
            .iter()
            .map(|(tool, n)| format!("{tool} ({n})"))
            .collect();
        println!("  Denied by tool: {}", top.join(", "));
    } else if asked > 0 {
        println!("  Nothing was denied in those runs.");
    }
    println!(
        "  Only calls xencode's own agent loop made are in that count; a vendor run through \
         `xencode agents` answers for itself and is not xencode's gate."
    );
    orchestrator_ran_nothing();
    Ok(())
}

struct PermissionRow {
    agent: &'static str,
    installed: bool,
    can_prompt_back: bool,
    grant: xencode_tui_rs::permission_broker::Grant,
    argv: Vec<String>,
    overruled: Vec<String>,
}

/// Money and energy, from the two places xencode measures them: the token
/// records of what a model answered, and the power counter of what a team run
/// drew from the wall.
fn orchestrator_costs(format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    let config = XencodeConfig::load().unwrap_or_default();
    let table =
        xencode_context_rs::PriceTable::load_with_lookup(&fleet.xencode_dir, config.price_lookup);
    let mut rollup = xencode_context_rs::MetricsRollup::empty();
    let records = xencode_context_rs::read_metrics(&fleet.xencode_dir);
    for row in &records {
        rollup.fold(row);
    }
    let report = xencode_context_rs::cost_of(&rollup.by_model, &table);
    let mut drawn = 0usize;
    let mut readable = 0usize;
    let mut watt_hours = 0f64;
    let mut energy_micros = 0u64;
    for file in &fleet.runs {
        if let Ok(run) = file.run.as_ref() {
            readable += 1;
            if let Some(wh) = run.watt_hours {
                drawn += 1;
                watt_hours += wh;
                energy_micros += run.cost_micros(run.cents_per_kwh).unwrap_or_default();
            }
        }
    }

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "metrics_dir": fleet.xencode_dir.display().to_string(),
                "metric_records": records.len(),
                "pricing_file": {
                    "path": table.path.display().to_string(),
                    "present": table.file_present,
                    "lookup_enabled": config.price_lookup,
                },
                "models": report.per_model.iter().map(|model| serde_json::json!({
                    "model": model.model,
                    "requests": model.tokens.requests,
                    "prompt_tokens": model.tokens.prompt_tokens,
                    "completion_tokens": model.tokens.completion_tokens,
                    "micros": model.micros,
                    "unknown_because": model.unknown_because,
                })).collect::<Vec<_>>(),
                "known_micros": report.known_micros,
                "unpriced_models": report.unpriced,
                "priced_from_listing": report.priced_from_listing,
                "team_runs": {
                    "recorded": fleet.runs.len(),
                    "with_power_counter": drawn,
                    "watt_hours": watt_hours,
                    "cents_per_kwh": config.power_cents_per_kwh,
                    "cost_micros": (drawn > 0).then_some(energy_micros),
                },
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Costs · read from {}", fleet.xencode_dir.display());
    println!(
        "  {} metric record(s); pricing from {}{}",
        records.len(),
        table.path.display(),
        if table.file_present {
            ""
        } else {
            ", which is not there"
        }
    );
    if report.per_model.is_empty() {
        println!("  models:  nothing recorded, so no token count exists to price.");
    } else {
        println!(
            "\n{:<40} {:>8} {:>10} {:>10}  {:>10}",
            "MODEL", "REQUESTS", "PROMPT", "COMPLETION", "COST"
        );
        for model in &report.per_model {
            let cost = match model.micros {
                Some(micros) => xencode_context_rs::format_usd(micros),
                None => "unpriced".to_string(),
            };
            println!(
                "{:<40} {:>8} {:>10} {:>10}  {:>10}",
                if model.model.is_empty() {
                    "(no model named)"
                } else {
                    model.model.as_str()
                },
                model.tokens.requests,
                model.tokens.prompt_tokens,
                model.tokens.completion_tokens,
                cost
            );
        }
        println!(
            "\n  priced total:    {} across {} priced model(s)",
            xencode_context_rs::format_usd(report.known_micros),
            report
                .per_model
                .iter()
                .filter(|m| m.micros.is_some())
                .count()
        );
        if !report.unpriced.is_empty() {
            println!(
                "  unpriced:        {} — their tokens are counted above and are not in the \
                 total, which is why an unknown is shown instead of a smaller number: {}",
                report.unpriced.len(),
                report.unpriced.join(", ")
            );
        }
        if !report.priced_from_listing.is_empty() {
            println!(
                "  from the fetched listing rather than pricing.json: {}",
                report.priced_from_listing.join(", ")
            );
        }
    }

    println!();
    match fleet.runs.iter().find(|file| file.run.is_ok()) {
        None if fleet.runs.is_empty() => println!(
            "  team energy:     nothing measured — no run is recorded in {}",
            fleet.runs_dir.display()
        ),
        None => println!(
            "  team energy:     {} file(s) in {}, none readable as a run",
            fleet.runs.len(),
            fleet.runs_dir.display()
        ),
        Some(_) => println!(
            "  team energy:     {}",
            energy_label(
                (drawn > 0).then_some(watt_hours),
                (drawn > 0).then_some(energy_micros),
                config.power_cents_per_kwh
            )
        ),
    }
    if drawn > 0 {
        println!(
            "                   from {} of {} recorded run(s){}",
            drawn,
            readable,
            if drawn == readable {
                " — every one of them left a counter reading."
            } else {
                "; the rest drew no counter, which is what happened when the machine reported \
                 nothing to read."
            }
        );
        println!(
            "                   Each run stores its own tariff, so a run priced last month keeps \
             last month's price rather than being re-priced at today's."
        );
    }
    orchestrator_ran_nothing();
    Ok(())
}

/// One row opened. The id may belong to a background task, a detached run, a
/// recorded team run or a recipe name; each is looked up in the place that holds
/// it, and a miss says which places were checked rather than guessing.
fn orchestrator_inspect(target: &str, format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    let json = matches!(format, OutputFormat::Json);

    if let Ok(id) = target.trim().parse::<u64>() {
        let registry = tasks_registry();
        let root = registry.root().to_path_buf();
        let entries = registry
            .poll()
            .map_err(|e| format!("cannot read the task registry: {e}"))?;
        let (task, status) = entries
            .get(&id)
            .ok_or_else(|| format!("no background task #{id} in {}", root.display()))?;
        let tail = registry.output(id, 20);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "kind": "task", "id": task.id, "name": task.name, "command": task.command,
                    "pid": task.pid, "status": status.label(), "killed": task.killed,
                    "source": task.source, "started_at_unix_secs": task.started_at,
                    "output_tail": tail,
                }))
                .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        println!("Background task #{id} · {}", root.display());
        println!("  name:     {}", task.name);
        println!("  command:  {}", task.command);
        println!("  pid:      {} ({})", task.pid, status.label());
        println!(
            "  started:  {} — the registry's own stamp, in whole seconds since the epoch",
            task.started_at
        );
        println!(
            "  killed:   {}",
            if task.killed {
                "yes — this xencode stopped it"
            } else {
                "no"
            }
        );
        println!(
            "  source:   {}",
            task.source
                .as_deref()
                .unwrap_or("started from the command line")
        );
        println!("  last {} output line(s):", tail.len());
        if tail.is_empty() {
            println!("    none held — the child has written nothing xencode captured");
        }
        for line in tail {
            println!("    {line}");
        }
        orchestrator_ran_nothing();
        return Ok(());
    }

    // Looked up in that order on purpose: a team run id is `<recipe>-<number>`, so
    // a prefix search would answer for the recipe's own name and open a run nobody
    // asked for. An exact id, then the recipe, then the prefixes.
    if let Some(file) = fleet.runs.iter().find(|file| {
        file.run
            .as_ref()
            .ok()
            .is_some_and(|run| run.run_id == target)
    }) {
        return orchestrator_show_run(file, &fleet.runs_dir, json);
    }

    let named = fleet
        .recipes
        .iter()
        .filter(|f| f.recipe.as_ref().is_ok_and(|r| r.name == target))
        .count();
    if named == 1 {
        let planned = plan_recipe(&fleet.recipes, &fleet.teams_dir, target)?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan_json(&planned, false))
                    .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        print_plan(&planned, false);
        return Ok(());
    }
    if named > 1 {
        return Err(format!(
            "{named} recipes in {} are all named {target}, so there is no one plan to open. \
             `xencode orchestrator graph <name>` says which file each is in.",
            fleet.teams_dir.display()
        ));
    }

    if let Some(id) = xencode_tui_rs::detached::resolve_run_id(&fleet.xencode_dir, target) {
        let dir = xencode_tui_rs::detached::run_dir(&fleet.xencode_dir, &id);
        let spec = xencode_tui_rs::detached::read_spec(&dir);
        let rounds = xencode_tui_rs::detached::read_rounds(&dir);
        let exit = xencode_tui_rs::detached::read_exit(&dir);
        let status = xencode_tui_rs::detached::derive_status(&dir);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "kind": "detached_run", "run_id": id, "status": status.label(),
                    "dir": dir.display().to_string(),
                    "log": xencode_tui_rs::detached::log_path(&dir).display().to_string(),
                    "spec": spec, "rounds": rounds, "exit": exit,
                }))
                .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        show_detached_run(&fleet.xencode_dir, &id)?;
        println!(
            "  log:      {}",
            xencode_tui_rs::detached::log_path(&dir).display()
        );
        println!(
            "  tokens:   {} prompt + {} completion over the whole run",
            rounds.iter().filter_map(|r| r.prompt_tokens).sum::<u64>(),
            rounds
                .iter()
                .filter_map(|r| r.completion_tokens)
                .sum::<u64>()
        );
        orchestrator_ran_nothing();
        return Ok(());
    }

    if let Some(file) = fleet.runs.iter().find(|file| {
        file.run
            .as_ref()
            .ok()
            .is_some_and(|run| run.run_id.starts_with(target))
    }) {
        return orchestrator_show_run(file, &fleet.runs_dir, json);
    }

    Err(format!(
        "nothing here is named {target}. Checked: the background tasks in {}, the recipes in \
         {}, the detached runs in {}, and the recorded team runs in {}.",
        fleet.xencode_dir.join("tasks").display(),
        fleet.teams_dir.display(),
        xencode_tui_rs::detached::detached_dir(&fleet.xencode_dir).display(),
        fleet.runs_dir.display(),
    ))
}

/// One recorded run, opened: the panel's own row with every figure's file named,
/// or the record itself when JSON was asked for. A file that is there and cannot
/// be read is shown as that, because the alternative is a not-found for a run
/// whose file is on disk.
fn orchestrator_show_run(
    file: &xencode_core_rs::RunFile,
    runs_dir: &std::path::Path,
    json: bool,
) -> Result<(), String> {
    if json {
        match &file.run {
            Ok(run) => println!(
                "{}",
                serde_json::to_string_pretty(run).map_err(|e| e.to_string())?
            ),
            Err(problem) => println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "path": file.path.display().to_string(), "unreadable": problem,
                }))
                .map_err(|e| e.to_string())?
            ),
        }
        return Ok(());
    }
    println!("Recorded team run · {}", file.path.display());
    for row in xencode_tui_rs::worker_panel::graph_rows(std::slice::from_ref(file), runs_dir) {
        println!("{}", row.detail());
    }
    orchestrator_ran_nothing();
    Ok(())
}

/// Re-run one role for real. This is the one verb on this surface that starts a
/// process: the role's own `sh -c` command, waited on by the same queue a team
/// run uses, at capacity one so nothing else is launched beside it.
///
/// It deliberately writes no run record. A record carries the peak concurrency
/// and the wall clock of a whole team, and a single role replayed on its own
/// would become the number the next plan quotes — which would be a measurement
/// of something that never happened.
async fn orchestrator_retry(
    recipe: &str,
    role: &str,
    approved_by: Option<&str>,
    format: OutputFormat,
) -> Result<(), String> {
    let fleet = Fleet::read();
    let planned = plan_recipe(&fleet.recipes, &fleet.teams_dir, recipe)?;
    let spec = planned
        .recipe
        .roles
        .iter()
        .find(|r| r.name == role)
        .ok_or_else(|| {
            format!(
                "{} has no role named `{role}`. It has: {}",
                planned.recipe.name,
                planned
                    .recipe
                    .roles
                    .iter()
                    .map(|r| r.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let who = match approved_by {
        None => {
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "approval_required": true,
                        "recipe": planned.recipe.name,
                        "role": spec.name,
                        "worker": spec.worker,
                        "command": spec.command,
                        "gate": spec.gate,
                        "needs": spec.needs,
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                println!(
                    "This would re-run one role of {} for real:\n  role:    {}\n  worker:  \
                     {}\n  command: {}\n  gate:    {}\n  needs:   {}",
                    planned.recipe.name,
                    spec.name,
                    spec.worker,
                    spec.command,
                    gate_words(spec),
                    needs_words(spec)
                );
                println!(
                    "\n  Nothing has run. One role replayed is not the team: the roles above it \
                     in the graph were not re-run, so this one reads whatever the last full run \
                     left on disk, and the roles below it will not be told anything changed."
                );
                println!(
                    "\n  To do it: xencode orchestrator retry {recipe} {role} --approved-by \
                     <your name>"
                );
            }
            return Ok(());
        }
        Some(blank) if blank.trim().is_empty() => {
            return Err(
                "`--approved-by` needs a name: a re-run launches a real process, and the record \
                 of who asked for it is the point of the flag"
                    .to_string(),
            );
        }
        Some(who) => who,
    };
    planned
        .profile
        .check_worker(
            &spec.worker,
            xencode_agents_rs::is_external_worker(&spec.worker),
        )
        .map_err(|refusal| {
            format!(
                "the {} posture refuses the worker this role names, so nothing was launched: {}",
                planned.profile.name(),
                refusal.why
            )
        })?;

    let mut graph = xencode_core_rs::TaskGraph::new();
    graph.add(xencode_core_rs::TaskNode {
        id: spec.name.clone(),
        command: spec.command.clone(),
        needs: Vec::new(),
    });
    let mut manager = xencode_core_rs::TaskManager::new();
    let window = xencode_context_rs::power::PowerWindow::begin();
    let report = xencode_core_rs::Scheduler::new(1, 1)
        .run(&graph, &mut manager)
        .await
        .map_err(|e| format!("{e}\nnothing was launched"))?;
    let energy = window.finish();
    let outcome = report.outcome(&spec.name).ok_or_else(|| {
        "the scheduler reported no outcome for the role it just launched".to_string()
    })?;
    let output: Vec<String> = manager
        .list()
        .iter()
        .flat_map(|record| record.output().iter().cloned())
        .rev()
        .take(20)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let downstream: Vec<&str> = planned
        .recipe
        .roles
        .iter()
        .filter(|r| r.needs.iter().any(|n| n == &spec.name))
        .map(|r| r.name.as_str())
        .collect();

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "recipe": planned.recipe.name,
                "role": spec.name,
                "worker": spec.worker,
                "command": spec.command,
                "approved_by": who,
                "status": outcome.status.label(),
                "started_ms": outcome.started_ms,
                "finished_ms": outcome.finished_ms,
                "elapsed_ms": report.elapsed_ms,
                "watt_hours": energy.total_watt_hours(),
                "cents_per_kwh": planned.tariff,
                "cost_micros": energy.cost_micros(planned.tariff),
                "output_tail": output,
                "run_record_written": false,
                "not_run": {
                    "needed_by_this_role": spec.needs,
                    "roles_that_were_waiting_on_this_one": downstream,
                },
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!(
        "Re-ran {} of {} · approved by {who}\n",
        spec.name, planned.recipe.name
    );
    println!("  command:  {}", spec.command);
    println!(
        "  status:   {}  ({} ms → {} ms, {} wall clock)",
        outcome.status.label(),
        outcome.started_ms,
        outcome.finished_ms,
        wall_clock_label(report.elapsed_ms)
    );
    println!(
        "  energy:   {}",
        energy_label(
            energy.total_watt_hours(),
            energy.cost_micros(planned.tariff),
            planned.tariff
        )
    );
    if !output.is_empty() {
        println!("  last {} output line(s):", output.len());
        for line in &output {
            println!("    {line}");
        }
    }
    println!();
    if spec.needs.is_empty() {
        println!("  It waits on nothing, so the graph above it is not in question.");
    } else {
        println!(
            "  It was launched without: {} — those roles did not run, so this one read whatever \
             they left on disk, which may be older than this command.",
            spec.needs.join(", ")
        );
    }
    if downstream.is_empty() {
        println!("  Nothing in the recipe waits on this role, so no other role is affected by what it just did.");
    } else {
        println!(
            "  Not re-run after it: {} — they would have been launched by the full team, and \
             this command stopped at the one role you named.",
            downstream.join(", ")
        );
    }
    println!(
        "\n  No run record was written: a record carries a whole team's wall clock and peak \
         concurrency, and quoting one role as if it were that would make the next plan of this \
         recipe estimate itself from a run that never happened. `xencode team run {}` is what \
         records.",
        planned.recipe.name
    );
    Ok(())
}

/// Stop one process this machine started and can still name. A background task by
/// its registry number, a detached run by its id — the two things xencode has a
/// pid for. Nothing else is in reach: it will not kill a process it cannot tie to
/// a record of launching it.
fn orchestrator_stop(target: &str) -> Result<(), String> {
    let fleet = Fleet::read();
    if let Ok(id) = target.trim().parse::<u64>() {
        let registry = tasks_registry();
        let root = registry.root().to_path_buf();
        let entries = registry
            .poll()
            .map_err(|e| format!("cannot read the task registry: {e}"))?;
        let (task, status) = entries
            .get(&id)
            .ok_or_else(|| format!("no background task #{id} in {}", root.display()))?;
        if !matches!(status, xencode_core_rs::TaskStatus::Running) {
            return Err(format!(
                "task #{id} is {}, not running — there is nothing to stop",
                status.label()
            ));
        }
        let name = task.name.clone();
        let pid = task.pid;
        registry
            .stop(id)
            .map_err(|e| format!("could not stop task #{id} (pid {pid}): {e}"))?;
        println!(
            "Stopped task #{id} ({name}, pid {pid}). The process was signalled by this command."
        );
        println!(
            "  The pid the registry holds is the `sh -c` wrapper the task was started with, so a \
             program that wrapper spawned keeps running unless it went down with it."
        );
        println!(
            "  Its record stays in the registry, so `xencode orchestrator inspect {id}` still \
             reads what it had said."
        );
        return Ok(());
    }
    stop_detached_run(&fleet.xencode_dir, target).map_err(|problem| {
        format!(
            "{problem}. This reaches a background task by its registry number and a detached run \
             by its id; a recorded team run is written after its children have all been waited on, \
             so there is no process of its own left to stop. `xencode orchestrator tasks` names \
             both kinds that can be stopped."
        )
    })
}

/// Hand this terminal to a vendor's own running session — and only that. The
/// command comes from the roster's `attach` cell, which is filled from the
/// vendor's help text for exactly the verb that takes over a session that already
/// exists. Most of the roster has none, and for those this refuses in words
/// rather than starting a second copy of the vendor and calling it a take-over.
fn orchestrator_attach(agent: &str, target: Option<&str>) -> Result<(), String> {
    use std::io::IsTerminal;
    use xencode_agents_rs::Handover;
    match xencode_agents_rs::handover(agent, target) {
        Handover::Ready { argv, template } => {
            if !io::stdout().is_terminal() {
                return Err(format!(
                    "`attach` hands this terminal to {}, and standard output here is not a \
                     terminal — a pipe, a redirect, a cron line or a CI step. There is no \
                     terminal to hand over, so nothing was started.",
                    argv.join(" ")
                ));
            }
            println!("Handing the terminal over · roster row `{template}`");
            println!("  running: {}", argv.join(" "));
            println!(
                "  xencode is not in the middle of this. When the vendor's own session lets go, \
                 the terminal comes back here."
            );
            let status = std::process::Command::new(&argv[0])
                .args(&argv[1..])
                .stdin(std::process::Stdio::inherit())
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .status()
                .map_err(|e| format!("{} could not be started: {e}", argv[0]))?;
            println!("\n{agent} returned {status}; the terminal is this command's again.");
            Ok(())
        }
        Handover::NoHandoverVerb {
            one_shot,
            read_on,
            session_list,
        } => Err(format!(
            "{agent} has no command that takes over a session it already has. Its help, read \
             on {read_on}, documents only a one-shot call — `{one_shot}` — which would start a \
             new vendor process rather than hand you one, so this command refuses instead of \
             pretending{}.",
            session_list
                .map(|listing| format!("; its own listing is `{listing}`, which you can run"))
                .unwrap_or_default()
        )),
        Handover::NeedsTarget {
            template,
            session_list,
        } => Err(format!(
            "attach {agent} needs the session to hand over, given as the vendor's own usage \
             line `{template}`. xencode does not choose it for you and does not list it for \
             you{} — a listing command is the vendor's own browser, and running it here would \
             take the terminal you have not agreed to hand over yet.",
            session_list
                .map(|listing| format!(": `{listing}` prints what it knows"))
                .unwrap_or_default()
        )),
        Handover::NotInstalled { template } => Err(format!(
            "{agent} is on the roster with the handover command `{template}`, but no binary for \
             it is on PATH here, so there is nothing to hand the terminal to."
        )),
        Handover::Unknown(name) => Err(format!(
            "{name} is not an agent xencode has a roster row for, so nothing here is known about \
             how to hand a terminal to one of its sessions — and xencode will not guess a \
             command and run it. `xencode orchestrator agents` lists the {} rows there are, and \
             `xencode agents` reports what a probe found on this machine.",
            xencode_agents_rs::ROSTER.len()
        )),
    }
}

/// Energy and its price on one line. `power_line` is what a generation window
/// prints; a record only holds the watt-hours it measured, so the same labels are
/// joined here.
fn energy_label(watt_hours: Option<f64>, cost_micros: Option<u64>, tariff: Option<f64>) -> String {
    match watt_hours {
        Some(wh) => format!(
            "{} · {} · this machine only",
            xencode_context_rs::power::watt_hours_label(Some(wh)),
            xencode_context_rs::power::cost_label(cost_micros, tariff)
        ),
        None => "energy unknown — this machine reports no power counter to read".to_string(),
    }
}

/// How one measured run compares to the estimate a previous one set. A
/// percentage is only offered when there is a number to divide by.
fn difference_label(actual_ms: u64, estimated_ms: u64) -> String {
    if estimated_ms == 0 {
        return format!("{actual_ms} ms against an estimate of 0 ms, which cannot be divided by");
    }
    let percent = (actual_ms as f64 - estimated_ms as f64) / estimated_ms as f64 * 100.0;
    let rounded = percent.round() as i64;
    if rounded == 0 {
        return format!(
            "{actual_ms} ms against an estimated {estimated_ms} ms — the same to the nearest \
             percent"
        );
    }
    let by = rounded.unsigned_abs();
    let word = if rounded > 0 { "longer" } else { "shorter" };
    format!("{actual_ms} ms against an estimated {estimated_ms} ms: {by}% {word} than the estimate")
}

/// A file's recipe, or the reason it is not a team that could schedule — the
/// read first, then what a recipe has to get right on its own. Planning checks
/// this through `TaskGraph`, which adds the dependency faults; this is the part
/// that is true before the graph is built.
fn recipe_status(
    file: &xencode_core_rs::RecipeFile,
) -> Result<&xencode_core_rs::TeamRecipe, xencode_core_rs::RecipeError> {
    let recipe = file.recipe.as_ref().map_err(Clone::clone)?;
    recipe.validate()?;
    Ok(recipe)
}

/// One recipe file in JSON form, including the files that are not schedulable —
/// a typing mistake in one recipe must not delete the others from the answer.
fn recipe_file_json(file: &xencode_core_rs::RecipeFile) -> serde_json::Value {
    let path = file.path.display().to_string();
    let recipe = match recipe_status(file) {
        Ok(recipe) => recipe,
        Err(problem) => {
            let name = file
                .recipe
                .as_ref()
                .ok()
                .map(|r| serde_json::Value::String(r.name.clone()))
                .unwrap_or(serde_json::Value::Null);
            return serde_json::json!({
                "file": path,
                "name": name,
                "schedulable": false,
                "problem": problem.to_string(),
            });
        }
    };
    serde_json::json!({
        "file": path,
        "name": recipe.name,
        "schedulable": true,
        "roles": recipe.roles.iter().map(|role| serde_json::json!({
            "name": role.name,
            "worker": role.worker,
            "gate": role.gate,
            "needs": role.needs,
            "command": role.command,
        })).collect::<Vec<_>>(),
        "capacity": {
            "workers": recipe.capacity.workers,
            "verification_throughput": recipe.capacity.verification_throughput,
            "at_most": recipe.scheduler().capacity(),
            "binding": recipe.scheduler().binding().label(),
        },
    })
}

/// Find one recipe by the name written inside it. Two files claiming the same
/// name is refused by path rather than resolved by taking the first, because the
/// one that gets picked would then be whichever the directory walk happened to
/// read first.
fn find_recipe<'a>(
    files: &'a [xencode_core_rs::RecipeFile],
    dir: &std::path::Path,
    name: &str,
) -> Result<&'a xencode_core_rs::RecipeFile, String> {
    let matches: Vec<&xencode_core_rs::RecipeFile> = files
        .iter()
        .filter(|f| f.recipe.as_ref().is_ok_and(|r| r.name == name))
        .collect();
    match matches.len() {
        1 => Ok(matches[0]),
        0 => {
            // A recipe that parses but would not schedule is named here too: the
            // person asking has a name in mind, and hiding the one that is
            // broken is the opposite of what a lookup should do.
            let named: Vec<&str> = files
                .iter()
                .filter_map(|f| f.recipe.as_ref().ok())
                .map(|r| r.name.as_str())
                .collect();
            if named.is_empty() {
                return Err(format!(
                    "no team recipe named `{name}`: {} holds no recipe that reads",
                    dir.display()
                ));
            }
            Err(format!(
                "no team recipe named `{name}`. {} has: {}",
                dir.display(),
                named.join(", ")
            ))
        }
        _ => Err(format!(
            "{} team recipes are all named `{name}`: {}; give them different recipe names, \
             because picking one by directory order would schedule a team you did not read",
            matches.len(),
            matches
                .iter()
                .map(|f| f.path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// What a role's gate means, said the way it actually is: an empty gate is not
/// a passing gate.
fn gate_words(role: &xencode_core_rs::RoleSpec) -> String {
    if role.gate.is_empty() {
        "none — no check gates this role's output".to_string()
    } else {
        role.gate.join(", ")
    }
}

fn needs_words(role: &xencode_core_rs::RoleSpec) -> String {
    if role.needs.is_empty() {
        "nothing (it can start at once)".to_string()
    } else {
        role.needs.join(", ")
    }
}

/// Whether the worker a role names is on this machine — a roster lookup and a
/// `PATH` walk. The agent is never started, which is what keeps planning a read.
fn worker_status(role: &xencode_core_rs::RoleSpec) -> String {
    match xencode_agents_rs::roster::find(&role.worker) {
        None => format!(
            "{} (not an agent xencode has a roster row for; nothing here checks whether it exists)",
            role.worker
        ),
        Some(spec) => {
            let installed = spec
                .binaries
                .iter()
                .find_map(|b| xencode_agents_rs::roster::which(b));
            match installed {
                Some(path) => format!("{} (installed at {})", role.worker, path.display()),
                None => format!(
                    "{} (a known agent, but not installed on this machine)",
                    role.worker
                ),
            }
        }
    }
}

/// Which roles could start together, in the order the scheduler would launch
/// them: every role ready now forms one wave, and the next wave is what becomes
/// ready once that whole wave is done. `TaskGraph::validate` already refused a
/// cycle, so a wave is never empty while roles remain.
fn readiness_waves(graph: &xencode_core_rs::TaskGraph) -> Vec<Vec<xencode_core_rs::NodeId>> {
    let mut done = std::collections::BTreeSet::new();
    let mut waves = Vec::new();
    while done.len() < graph.nodes().len() {
        let wave: Vec<xencode_core_rs::NodeId> =
            graph.ready(&done).iter().map(|n| n.id.clone()).collect();
        done.extend(wave.iter().cloned());
        waves.push(wave);
    }
    waves
}

fn run_compete(action: CompeteAction) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    match action {
        CompeteAction::Run {
            prompt,
            arms,
            edits,
            commands,
            skip,
            timeout,
            format,
        } => {
            for name in &skip {
                if !["test", "lint", "fmt"].contains(&name.as_str()) {
                    return Err(format!(
                        "cannot skip {name:?}: the checklist is test, lint, fmt"
                    ));
                }
            }
            let arm_specs = build_candidate_arms(&arms, &edits, &commands)?;
            if arm_specs
                .iter()
                .all(|s| s.file_edits.is_empty() && s.command.is_none())
            {
                eprintln!(
                    "warning: no arm has any code of its own, so both branches carry the same \
                     candidate note and will verify identically. Describe each arm with `--edit` \
                     or `--command` to actually compete two implementations."
                );
            }
            // A competing run is synchronous and pays for the toolchain once per
            // arm, so the cost is stated before it starts, not after.
            let planned: Vec<&str> = ["fmt", "lint", "test"]
                .iter()
                .copied()
                .filter(|check| !skip.iter().any(|s| s == *check))
                .collect();
            let planned = if planned.is_empty() {
                "nothing — every check is skipped".to_string()
            } else {
                planned.join(", ")
            };
            eprintln!(
                "compete: {} arms, each checked by [{}] in its own worktree beside {}; the \
                 toolchain runs once per arm with a {timeout}s ceiling per check.",
                arm_specs.len(),
                planned,
                root.display(),
            );
            let config = xencode_analysis_rs::CompetingConfig {
                prompt,
                arm_specs,
                skip_checks: skip,
                timeout_secs: timeout,
            };
            let report = xencode_analysis_rs::run_competing_arms(&root, &config)?;
            print_competing_report(&report, format, true)
        }
        CompeteAction::List { format } => {
            let runs = xencode_analysis_rs::list_competing_runs(&root)?;
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&runs).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            if runs.is_empty() {
                println!(
                    "No competing runs recorded under {}.",
                    root.join(".xencode").join("compete").display()
                );
                return Ok(());
            }
            println!("{:<24} {:<5} {:<8} PROMPT", "RUN", "ARMS", "PICKED");
            for run in &runs {
                println!(
                    "{:<24} {:<5} {:<8} {}",
                    run.run_id,
                    run.arms.len(),
                    run.picked_arm.clone().unwrap_or_else(|| "-".to_string()),
                    run.prompt
                );
            }
            println!("\nRe-print any of them: xencode compete show <run-id>");
            Ok(())
        }
        CompeteAction::Show { run_id, format } => {
            let report = xencode_analysis_rs::load_competing_run(&root, &run_id)?;
            print_competing_report(&report, format, false)
        }
        CompeteAction::Pick { run_id, arm_id } => {
            let outcome = xencode_analysis_rs::pick_arm(&root, &run_id, &arm_id)?;
            println!(
                "Checked out arm `{}` on branch `{}`.",
                outcome.picked_arm_id, outcome.picked_branch
            );
            if outcome.other_branches.is_empty() {
                println!("No other candidate branch survived this run to preserve.");
            } else {
                println!("Candidate branches left on disk:");
                for branch in &outcome.other_branches {
                    println!("  {branch}");
                }
            }
            if outcome.preserved_evidence.is_empty() {
                println!("No evidence directory was preserved for the other arms.");
            } else {
                println!("Evidence left on disk:");
                for evidence in &outcome.preserved_evidence {
                    println!("  {}", evidence.display());
                }
            }
            Ok(())
        }
    }
}

/// Who last committed on this branch, read from git rather than asserted. A
/// veto has to name the party it blocks before anyone asks who is clearing it,
/// and the branch's own author is the only identity here that was not typed in
/// by whoever is asking.
fn branch_author(repo: &std::path::Path, branch: &str) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["log", "-1", "--format=%an", branch])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let author = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!author.is_empty()).then_some(author)
}

/// The files `branch` changed since `base` that its lease did not give it, judged
/// by the task contract (OR-18). `None` when the branch never ran under a lease
/// or stayed inside it. A leases file that cannot be read is an error: landing
/// past a check that could not run would be landing unchecked.
fn lease_breaches(
    root: &std::path::Path,
    base: &str,
    branch: &str,
) -> Result<Option<Vec<String>>, String> {
    let file = root
        .join(xencode_context_rs::XENCODE_DIR)
        .join("leases.json");
    let registry = xencode_core_rs::LeaseRegistry::load(root.to_path_buf(), &file)?;
    let Some(lease) = registry.lease_for_branch(branch) else {
        return Ok(None);
    };
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["diff", "--name-only", &format!("{base}...{branch}")])
        .output()
        .map_err(|e| format!("cannot run git diff for `{branch}`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cannot list what `{branch}` changed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let changed: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect();
    let contract = xencode_core_rs::TaskContract {
        task: lease.task_id.clone(),
        lease: lease.lease_id.clone(),
        workspace: root.to_path_buf(),
        allowed_files: lease.declared_files.clone(),
        forbidden_paths: vec![".git".to_string()],
        deliverables: Vec::new(),
        verification_commands: Vec::new(),
    };
    Ok(contract.check_finish(&changed).err().map(|breaches| {
        breaches
            .iter()
            .map(|b| match b {
                xencode_core_rs::Breach::OutsideLease { path }
                | xencode_core_rs::Breach::Undeclared { path }
                | xencode_core_rs::Breach::ForbiddenPath { path, .. } => path.clone(),
            })
            .collect()
    }))
}

/// The trail a veto belongs to: the same chained `audit.jsonl` the server writes
/// session events to, so `xencode audit verify` walks one log rather than two.
fn veto_audit_sink() -> Result<xencode_server_rs::audit::AuditSink, String> {
    let path = xencode_config_rs::paths::state_dir()
        .map_err(|e| format!("cannot find where xencode keeps its records: {e}"))?
        .join("audit.jsonl");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    Ok(xencode_server_rs::audit::AuditSink::to_file(path))
}

/// One veto record as the trail wants it: who acted, what it was about, and the
/// reasoning rather than a pointer to it.
fn veto_audit_event(
    action: xencode_collaboration_rs::AuditAction,
    veto: &xencode_analysis_rs::Veto,
    actor: &str,
    detail: String,
) -> xencode_collaboration_rs::AuditEvent {
    xencode_collaboration_rs::AuditEvent {
        seq: 0,
        at: xencode_collaboration_rs::audit_stamp(),
        actor: actor.to_string(),
        action,
        target: veto.branch.clone(),
        detail,
    }
}

fn run_merge(action: MergeAction) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    match action {
        MergeAction::Precheck {
            branch,
            base,
            format,
        } => {
            let precheck = xencode_analysis_rs::precheck_branch(&root, &base, &branch)?;
            match format {
                OutputFormat::Json => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&precheck).map_err(|e| e.to_string())?
                    );
                }
                _ => {
                    if precheck.clean {
                        println!("Precheck for branch '{branch}' against base '{base}': CLEAN (no merge conflicts)");
                    } else {
                        println!("Precheck for branch '{branch}' against base '{base}': CONFLICT");
                        println!("Conflicting files ({}):", precheck.conflict_files.len());
                        for f in &precheck.conflict_files {
                            println!("  - {f}");
                        }
                        if let Some(diff) = &precheck.rendered_conflict {
                            println!("\nRendered conflict diff:\n{diff}");
                        }
                    }
                }
            }
            Ok(())
        }
        MergeAction::Plan {
            branches,
            base,
            format,
        } => {
            let mut specs = Vec::new();
            for b in &branches {
                let verify = std::process::Command::new("git")
                    .arg("-C")
                    .arg(&root)
                    .args(["rev-parse", "--verify", b])
                    .output();
                let (passed, sha) = match verify {
                    Ok(out) if out.status.success() => (
                        true,
                        String::from_utf8_lossy(&out.stdout).trim().to_string(),
                    ),
                    _ => (false, String::new()),
                };
                specs.push(xencode_analysis_rs::BranchSpec {
                    branch: b.clone(),
                    worker: branch_author(&root, b).unwrap_or_else(|| "unknown".to_string()),
                    task_id: format!("task-{}", b),
                    checks: vec![xencode_analysis_rs::BranchCheck {
                        name: "branch-commit-verified".to_string(),
                        exit_code: if passed { 0 } else { 1 },
                        passed,
                        evidence_ref: if passed {
                            format!("sha:{}", sha)
                        } else {
                            "git rev-parse failed".to_string()
                        },
                    }],
                });
            }
            let plan = xencode_analysis_rs::build_merge_plan(&root, &base, &specs)?;
            match format {
                OutputFormat::Json => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&plan).map_err(|e| e.to_string())?
                    );
                }
                _ => {
                    let open: usize = plan.branches.iter().map(|b| b.vetoes.len()).sum();
                    println!("Merge Plan for base branch '{}':", plan.base_branch);
                    println!("  Overall clean: {}", plan.all_clean);
                    println!(
                        "  All worker checks passed: {}",
                        plan.all_worker_checks_passed
                    );
                    if open > 0 {
                        println!("  Open vetoes: {open} — this plan cannot land");
                    }
                    println!("  Candidate branches ({}):", plan.branches.len());
                    for bv in &plan.branches {
                        let short_sha = if bv.commit_sha.len() >= 8 {
                            &bv.commit_sha[..8]
                        } else {
                            &bv.commit_sha
                        };
                        println!(
                            "    - branch '{}' (sha: {}, clean: {}, checks: {}, eligible: {})",
                            bv.branch,
                            short_sha,
                            bv.precheck.clean,
                            bv.worker_checks.iter().all(|c| c.passed),
                            bv.eligible
                        );
                        if !bv.precheck.clean {
                            println!("      Conflicts: {:?}", bv.precheck.conflict_files);
                        }
                        for veto in &bv.vetoes {
                            println!("      BLOCKED — {}", veto.summary_line());
                            println!("      clear it: {}", veto.who_may_clear());
                        }
                    }
                }
            }
            Ok(())
        }
        MergeAction::Land {
            branches,
            base,
            approved_by,
            test_cmds,
            format,
        } => {
            // OR-18: a branch that ran under a lease lands only if its changes
            // stayed inside the files it was given.
            for b in &branches {
                if let Some(outside) = lease_breaches(&root, &base, b)? {
                    return Err(format!(
                        "refusing to land `{b}`: it changed files outside the ones its task was given — {}",
                        outside.join(", ")
                    ));
                }
            }
            let mut specs = Vec::new();
            for b in &branches {
                let verify = std::process::Command::new("git")
                    .arg("-C")
                    .arg(&root)
                    .args(["rev-parse", "--verify", b])
                    .output();
                let (passed, sha) = match verify {
                    Ok(out) if out.status.success() => (
                        true,
                        String::from_utf8_lossy(&out.stdout).trim().to_string(),
                    ),
                    _ => (false, String::new()),
                };
                specs.push(xencode_analysis_rs::BranchSpec {
                    branch: b.clone(),
                    worker: branch_author(&root, b).unwrap_or_else(|| "unknown".to_string()),
                    task_id: format!("task-{}", b),
                    checks: vec![xencode_analysis_rs::BranchCheck {
                        name: "branch-commit-verified".to_string(),
                        exit_code: if passed { 0 } else { 1 },
                        passed,
                        evidence_ref: if passed {
                            format!("sha:{}", sha)
                        } else {
                            "git rev-parse failed".to_string()
                        },
                    }],
                });
            }
            let plan = xencode_analysis_rs::build_merge_plan(&root, &base, &specs)?;

            let approval = xencode_analysis_rs::HumanMergeApproval::new(
                &approved_by,
                !approved_by.trim().is_empty(),
                format!("Land requested via CLI by {}", approved_by),
            );

            let test_cmd_refs: Vec<&str> = test_cmds.iter().map(|s| s.as_str()).collect();
            let outcome =
                xencode_analysis_rs::execute_merge(&root, &plan, &approval, &test_cmd_refs)?;

            match format {
                OutputFormat::Json => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&outcome).map_err(|e| e.to_string())?
                    );
                }
                _ => {
                    println!("Merge Landed Successfully!");
                    println!("  Base branch: {}", outcome.base_branch);
                    println!("  Merged branches: {:?}", outcome.merged_branches);
                    println!("  Approved by: {}", outcome.approved_by);
                    if let Some(sha) = &outcome.integration_commit {
                        println!("  Integration commit: {}", sha);
                    }
                    println!(
                        "  Post-integration checks ({}):",
                        outcome.post_integration_checks.len()
                    );
                    for chk in &outcome.post_integration_checks {
                        println!(
                            "    - `{}`: {} (exit code {})",
                            chk.command,
                            if chk.passed { "PASSED" } else { "FAILED" },
                            chk.exit_code
                        );
                    }
                }
            }
            Ok(())
        }
        MergeAction::Veto {
            branch,
            source,
            raised_by,
            worker,
            reason,
        } => {
            let source = xencode_analysis_rs::VetoSource::parse(&source).ok_or_else(|| {
                format!("unknown veto source '{source}' — it is 'review' or 'verification'")
            })?;
            let exists = std::process::Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["rev-parse", "--verify", &branch])
                .output()
                .map(|out| out.status.success())
                .unwrap_or(false);
            if !exists {
                return Err(format!(
                    "'{branch}' is not a branch in this repository, and a veto names the change \
                     it blocks"
                ));
            }
            let author = match worker.as_deref() {
                Some(given) if !given.trim().is_empty() => given.trim().to_string(),
                _ => branch_author(&root, &branch).ok_or_else(|| {
                    format!("cannot read who authored '{branch}' from git; pass --worker")
                })?,
            };
            let veto = xencode_analysis_rs::record_veto(
                &root, &branch, &author, source, &raised_by, &reason,
            )?;
            println!("Veto {} blocks '{}'.", veto.id, veto.branch);
            println!("  blocked worker: {}", veto.worker);
            println!("  reason: {}", veto.reason);
            println!("  clear it: {}", veto.who_may_clear());
            println!("  xencode merge clear-veto {} --by \"<name>\"", veto.id);
            // The block stands whatever the logging does: refusing to record a
            // veto because the trail is unwritable would remove protection.
            let event = veto_audit_event(
                xencode_collaboration_rs::AuditAction::MergeVetoed,
                &veto,
                &veto.raised_by,
                format!("{}: {}", veto.id, veto.reason),
            );
            match veto_audit_sink()?.append_external(&event) {
                Ok(seq) => println!("  recorded as audit sequence {seq}"),
                Err(e) => eprintln!("  warning: the veto holds but is not in the audit trail: {e}"),
            }
            Ok(())
        }
        MergeAction::ClearVeto { id, by, policy } => {
            // Decide, then record, then write. The clearance never reaches the
            // veto file until the audit trail has accepted it, so a clear that
            // cannot be audited leaves the branch blocked without having to be
            // taken back afterwards.
            let on_record = xencode_analysis_rs::load_vetoes(&root)?;
            let veto = on_record
                .iter()
                .find(|v| v.id == id.trim())
                .ok_or_else(|| format!("no veto '{id}' is on record in this repository"))?;
            let cleared = xencode_analysis_rs::check_clearance(veto, &by, policy.as_deref())?;
            let detail = match &cleared.cleared {
                Some(c) => {
                    let policy = c
                        .policy
                        .as_deref()
                        .map(|text| format!(" under the policy: {text}"))
                        .unwrap_or_default();
                    format!(
                        "{} lifted by {}{policy} — it had blocked '{}' for: {}",
                        cleared.id, c.cleared_by, cleared.branch, cleared.reason
                    )
                }
                None => format!("{} lifted by {by}", cleared.id),
            };
            let event = veto_audit_event(
                xencode_collaboration_rs::AuditAction::MergeVetoCleared,
                &cleared,
                &by,
                detail,
            );
            if let Err(e) = veto_audit_sink()?.append_external(&event) {
                return Err(format!(
                    "clearing a veto is an audited event, and the audit trail would not take \
                     it: {e}. '{}' is still open and nothing was changed.",
                    cleared.id
                ));
            }
            xencode_analysis_rs::persist_clearance(&root, &cleared).map_err(|e| {
                format!(
                    "the audit trail now says '{}' was cleared by {}, but the veto file would \
                     not take it: {e}. The branch stays blocked, so it is the record that is \
                     wrong, not the merge.",
                    cleared.id,
                    by.trim()
                )
            })?;
            println!("Veto {} no longer blocks '{}'.", cleared.id, cleared.branch);
            println!("  cleared by: {}", by.trim());
            if let Some(p) = cleared.cleared.as_ref().and_then(|c| c.policy.as_ref()) {
                println!("  under policy: {p}");
            }
            Ok(())
        }
        MergeAction::Vetoes { open, format } => {
            let mut all = xencode_analysis_rs::load_vetoes(&root)?;
            if open {
                all.retain(|veto| veto.is_open());
            }
            match format {
                OutputFormat::Json => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&all).map_err(|e| e.to_string())?
                    );
                }
                _ => {
                    if all.is_empty() {
                        println!("No vetoes on record for this repository.");
                    }
                    for veto in &all {
                        let state = match &veto.cleared {
                            None => "open".to_string(),
                            Some(c) => match &c.policy {
                                Some(p) => format!("cleared by {} under policy: {p}", c.cleared_by),
                                None => format!("cleared by {}", c.cleared_by),
                            },
                        };
                        println!(
                            "{}  {:<12}  {}  [{state}]",
                            veto.id,
                            veto.branch,
                            veto.summary_line()
                        );
                    }
                }
            }
            Ok(())
        }
    }
}

fn run_cache(action: CacheAction) -> Result<(), String> {
    match action {
        CacheAction::Stats => {
            match ResponseCache::with_persistence(100, 3600.0) {
                Ok(cache) => {
                    let stats = cache.stats();
                    println!("Cache Statistics:");
                    println!("  {stats}");
                }
                Err(_) => {
                    // Fall back to reporting empty stats
                    println!("Cache Statistics:");
                    println!("  entries: 0, hits: 0, misses: 0, hit_rate: 0.0%, evictions: 0");
                }
            }
            Ok(())
        }
        CacheAction::Clear => {
            match ResponseCache::with_persistence(100, 3600.0) {
                Ok(mut cache) => {
                    cache.clear().map_err(|e| e.to_string())?;
                    println!("Cache cleared successfully");
                }
                Err(_) => {
                    println!("No cache to clear");
                }
            }
            Ok(())
        }
        CacheAction::Gc { max_mb } => {
            let dir = ResponseCache::persistence_dir().map_err(|e| e.to_string())?;
            let cap_bytes = max_mb.saturating_mul(1024 * 1024);
            let report =
                xencode_cache_rs::collect_to_size(&dir, cap_bytes).map_err(|e| e.to_string())?;
            println!("cache:   {}", dir.display());
            if report.was_within_cap() {
                println!(
                    "Cached responses take {}, which fits under the {} cap. Nothing removed.",
                    megabytes_label(report.bytes_before),
                    megabytes_label(cap_bytes)
                );
            } else {
                println!(
                    "Removed {} cached response{}, oldest first: {} down to {} ({} freed), to fit \
                     under {}.",
                    report.removed,
                    if report.removed == 1 { "" } else { "s" },
                    megabytes_label(report.bytes_before),
                    megabytes_label(report.bytes_after),
                    megabytes_label(report.freed_bytes()),
                    megabytes_label(cap_bytes)
                );
                println!(
                    "The advisory corpora under advisories/ are a separate download and were not \
                     counted or touched."
                );
            }
            Ok(())
        }
    }
}

/// A size in the unit a person asked in. Under a mebibyte the exact byte count
/// is printed, because "0.0 MiB" for 300 KiB reads as though there were nothing there.
fn megabytes_label(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    if bytes < MIB {
        format!("{bytes} B")
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    }
}

/// Score the agent against defects that were put there on purpose.
///
/// The command measures; it does not gate. A pass rate of one in eight is a
/// correct answer and exits successfully, because the number is the product.
/// Only a run that reached no verdict at all — every case failed before a
/// grader could be consulted — is reported as an error.
async fn run_eval(action: EvalAction) -> Result<(), String> {
    use xencode_tui_rs::task_eval::{
        parse_shape, read_task_eval_runs, run_task_eval, shape_names, TaskEvalOptions,
    };

    match action {
        EvalAction::List => {
            println!("defects that can be seeded:");
            for shape in xencode_context_rs::BugShape::all() {
                println!("  {:<20} {}", shape.slug(), shape.title());
            }
            let runs = read_task_eval_runs(&project_xencode_dir());
            if runs.is_empty() {
                println!("\nno eval run recorded yet");
                return Ok(());
            }
            let now = xencode_context_rs::conversation::now_millis();
            println!("\nrecorded runs, oldest first:");
            for run in runs {
                let age_minutes = now.saturating_sub(run.ts_unix_ms) / 60_000;
                // A ranking is not a verdict, so it is said after the numbers
                // rather than mixed into them.
                let ranked = if run.judge_ranking.is_empty() {
                    String::new()
                } else {
                    format!(" · closest to a fix: {}", run.judge_ranking.join(", "))
                };
                println!(
                    "  {:>6}  {:<26} {:>3}/{:<3} graded passed  {} · prompts {} · {}{ranked}",
                    format_age_minutes(age_minutes),
                    run.model,
                    run.passed,
                    run.graded,
                    run.server,
                    &run.prompt_version[..8.min(run.prompt_version.len())],
                    run.approval,
                );
            }
            Ok(())
        }
        EvalAction::Run {
            cases,
            model,
            repeats,
            max_rounds,
            allow_shell,
            out,
            ollama_url,
            llamacpp_url,
            timeout,
            max_tokens,
            judge,
            judge_model,
        } => {
            let config = XencodeConfig::load().unwrap_or_default();
            let shapes = if cases.is_empty() {
                xencode_context_rs::BugShape::all().to_vec()
            } else {
                let mut picked = Vec::with_capacity(cases.len());
                for word in &cases {
                    match parse_shape(word) {
                        Some(shape) => picked.push(shape),
                        None => {
                            return Err(format!(
                                "no defect is called `{word}`; the shapes are {}",
                                shape_names().join(", ")
                            ))
                        }
                    }
                }
                picked
            };
            let out_dir = out.unwrap_or_else(|| {
                std::env::temp_dir().join(format!(
                    "xencode-eval-{}",
                    xencode_context_rs::conversation::now_millis()
                ))
            });
            let options = TaskEvalOptions {
                out_dir,
                model: model.unwrap_or(config.default_model.clone()),
                shapes,
                repeats: repeats.max(1),
                ollama_url,
                llama_cpp_url: llamacpp_url,
                remote_base_url: None,
                max_rounds: max_rounds.unwrap_or(xencode_tui_rs::task_eval::DEFAULT_MAX_ROUNDS),
                allow_shell,
                temperature: Some(config.llama_cpp_temperature.unwrap_or(0.0)),
                seed: Some(config.llama_cpp_seed.unwrap_or(42)),
                timeout_secs: timeout,
                max_tokens: (max_tokens > 0).then_some(max_tokens),
                history_dir: Some(project_xencode_dir()),
                judge,
                judge_model,
            };
            let report = run_task_eval(&options).await?;
            for line in report.lines() {
                println!("{line}");
            }
            if report.graded() == 0 {
                return Err("no case reached a verdict: nothing was measured".to_string());
            }
            Ok(())
        }
    }
}

/// How long ago a recorded run happened, in the unit a person would say out loud.
fn format_age_minutes(minutes: u64) -> String {
    match minutes {
        0 => "now".to_string(),
        m if m < 60 => format!("{m}m ago"),
        m if m < 60 * 24 => format!("{}h ago", m / 60),
        m => format!("{}d ago", m / (60 * 24)),
    }
}

/// Check the session server's audit log for records that were edited after
/// the fact. Exits non-zero when something does not add up.
fn run_audit(action: AuditAction) -> Result<(), String> {
    use xencode_server_rs::audit::{describe, verdict, verify_chain};

    match action {
        AuditAction::Verify { path } => {
            let path = match path {
                Some(path) => path,
                None => resolve_audit_path(None)?.ok_or_else(|| {
                    "the audit log is turned off, so nothing was recorded".to_string()
                })?,
            };
            if !path.exists() {
                println!(
                    "{}: no audit log has been written here, so there is nothing to check",
                    path.display()
                );
                return Ok(());
            }
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("{} could not be read: {e}", path.display()))?;
            let report = verify_chain(&text);
            for problem in &report.problems {
                println!("{}", describe(problem));
            }
            println!("{}", verdict(&path, &report));
            if report.intact() {
                Ok(())
            } else {
                Err(format!(
                    "{} was changed after it was written",
                    path.display()
                ))
            }
        }
    }
}

/// Where the advisory corpora live: the cache directory, because they are a
/// download that `advisories sync` can repeat. `--dir` exists so a corpus can be
/// kept on a shared or offline path without touching it.
fn advisory_corpus(dir: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(dir) = dir {
        return Ok(dir);
    }
    let cache_dir = xencode_config_rs::paths::cache_dir().map_err(|e| e.to_string())?;
    Ok(xencode_analysis_rs::advisories::corpus_dir(&cache_dir))
}

/// SE-6 — supply-chain report.
///
/// Shells out to whichever dependency checkers are installed (`cargo-shear`
/// for unused dependencies, `cargo-deny` for advisories/bans/licenses),
/// parses their JSON and streams the findings, alongside the local facts that
/// need no external tool — duplicate majors and the delta of this `Cargo.lock`
/// against the one committed at HEAD. It is report only: auto-fixing a
/// dependency is how the supply chain becomes the attack, so nothing here
/// edits a manifest.
///
/// A checker that is not installed is reported as such, never as a clean tree.
fn run_deps(path: PathBuf, format: OutputFormat) -> Result<(), String> {
    use xencode_analysis_rs::deps;

    let manifest = xencode_context_rs::verify::manifest_dir(&path)?;

    enum Outcome {
        Ran(String),
        Missing,
        Failed(String),
    }
    // Invoke the hyphenated binary directly so a tool that is not installed is
    // an OS NotFound, not a cargo "no such command" message on stderr.
    let run_checker = |binary: &str, args: &[&str]| -> Outcome {
        match std::process::Command::new(binary)
            .current_dir(&manifest)
            .args(args)
            .output()
        {
            Ok(out) => Outcome::Ran(String::from_utf8_lossy(&out.stdout).into_owned()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Outcome::Missing,
            Err(e) => Outcome::Failed(e.to_string()),
        }
    };

    let mut findings: Vec<serde_json::Value> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    let shear_status: String;
    let deny_status: String;

    match run_checker("cargo-shear", &["--format", "json", "--offline"]) {
        Outcome::Ran(stdout) => match deps::parse_shear_json(&stdout) {
            Ok(report) => {
                shear_status =
                    format!("{} error(s), {} warning(s)", report.errors, report.warnings);
                for f in report.findings {
                    let sev = if f.severity == "error" {
                        "High"
                    } else {
                        "Medium"
                    };
                    lines.push(format!(
                        "  [{sev}] unused-dependency {} — {} ({})",
                        f.file,
                        f.message,
                        f.help.as_deref().unwrap_or("no suggestion")
                    ));
                    findings.push(serde_json::json!({
                        "checker": "cargo-shear", "severity": f.severity,
                        "kind": "unused-dependency", "file": f.file,
                        "message": f.message, "help": f.help,
                    }));
                }
            }
            Err(e) => shear_status = e,
        },
        Outcome::Missing => shear_status = "cargo-shear not installed".to_string(),
        Outcome::Failed(e) => shear_status = format!("cargo-shear failed: {e}"),
    }

    match run_checker(
        "cargo-deny",
        &["check", "--all-features", "--format", "json"],
    ) {
        Outcome::Ran(stdout) => match deps::parse_deny_json(&stdout) {
            Ok(found) => {
                deny_status = format!("{} finding(s)", found.len());
                for f in found {
                    let sev = if f.severity == "error" {
                        "Critical"
                    } else {
                        "Medium"
                    };
                    let scope = f.krate.as_deref().unwrap_or("(project)");
                    lines.push(format!(
                        "  [{sev}] {} {} {scope} — {}",
                        f.check, f.id, f.message
                    ));
                    findings.push(serde_json::json!({
                        "checker": "cargo-deny", "severity": f.severity,
                        "kind": f.check, "id": f.id, "crate": f.krate,
                        "message": f.message,
                    }));
                }
            }
            Err(e) => deny_status = e,
        },
        Outcome::Missing => {
            deny_status =
                "cargo-deny not installed — advisories and licenses are not checked here; \
                 use `xencode advisories check` for the offline RustSec/OSV corpus"
                    .to_string();
        }
        Outcome::Failed(e) => deny_status = format!("cargo-deny failed: {e}"),
    }

    // Local, tool-free facts.
    let lock_text = std::fs::read_to_string(manifest.join("Cargo.lock")).unwrap_or_default();
    for dup in deps::duplicate_versions(&lock_text) {
        lines.push(format!(
            "  [Medium] duplicate-major {} pinned at {} versions: {}",
            dup.krate,
            dup.versions.len(),
            dup.versions.join(", ")
        ));
        findings.push(serde_json::json!({
            "checker": "local", "severity": "medium", "kind": "duplicate-major",
            "crate": dup.krate, "versions": dup.versions,
        }));
    }
    let mut delta_lines: Vec<String> = Vec::new();
    if let Some((_, head)) = deps::head_lock_text(&manifest) {
        let delta = deps::lock_delta(&head, &lock_text);
        if !delta.added.is_empty() || !delta.removed.is_empty() || !delta.upgraded.is_empty() {
            for (name, version) in &delta.added {
                delta_lines.push(format!("  + {name} {version}"));
                findings.push(serde_json::json!({
                    "checker": "local", "kind": "lock-added", "crate": name, "version": version,
                }));
            }
            for (name, version) in &delta.removed {
                delta_lines.push(format!("  - {name} {version}"));
                findings.push(serde_json::json!({
                    "checker": "local", "kind": "lock-removed", "crate": name, "version": version,
                }));
            }
            for (name, from, to) in &delta.upgraded {
                delta_lines.push(format!("  ~ {name} {from} -> {to}"));
                findings.push(serde_json::json!({
                    "checker": "local", "kind": "lock-upgraded", "crate": name, "from": from, "to": to,
                }));
            }
        }
    }

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "checkers": { "cargo-shear": shear_status, "cargo-deny": deny_status },
                "findings": findings,
            })
        );
        return Ok(());
    }

    println!("dependency checkers");
    println!("  cargo-shear (unused dependencies): {shear_status}");
    println!("  cargo-deny (advisories, bans, licenses): {deny_status}");
    if !lines.is_empty() {
        println!("\nfindings");
        for line in &lines {
            println!("{line}");
        }
    } else {
        println!("\nno checker raised a finding against a tool it could run");
    }
    if !delta_lines.is_empty() {
        println!("\nlockfile vs HEAD (report only — review before merging):");
        for line in delta_lines {
            println!("{line}");
        }
    }
    println!("\nreport only: this command edits no manifest. Re-run a checker's own fix yourself if you accept a finding.");
    Ok(())
}

async fn run_advisories(action: AdvisoryAction) -> Result<(), String> {
    use xencode_analysis_rs::advisories as adv;

    match action {
        AdvisoryAction::Sync { dir } => {
            let corpus = advisory_corpus(dir)?;
            let outcome = adv::sync(&corpus).await.map_err(|e| e.to_string())?;
            println!(
                "corpus at {} — {} RustSec advisories, {} OSV records, {} index lines",
                corpus.display(),
                outcome.info.rustsec_advisories,
                outcome.info.osv_records,
                outcome.info.index_lines
            );
            println!(
                "RustSec revision {} ({}); OSV download {} bytes",
                outcome.info.rustsec_revision,
                if outcome.rustsec_pulled {
                    "pulled"
                } else {
                    "cloned fresh"
                },
                outcome.osv_bytes
            );
            println!(
                "lookups are offline from here; run `xencode advisories check` to read Cargo.lock"
            );
            Ok(())
        }
        AdvisoryAction::Show {
            crate_name,
            version,
            dir,
        } => {
            let corpus = advisory_corpus(dir)?;
            let lookup = match adv::advisories_for(&corpus, &crate_name) {
                Ok(lookup) => lookup,
                Err(e) => return Err(e.to_string()),
            };
            print!("{}", adv::render_lookup(&lookup, version.as_deref()));
            Ok(())
        }
        AdvisoryAction::Check { path, dir } => {
            let corpus = advisory_corpus(dir)?;
            let lock = xencode_tui_rs::crate_sources::lock_file(&path).ok_or_else(|| {
                format!(
                    "no Cargo.lock is readable for {} — the search stops at the project root, \
                     so pass --path to the directory that holds the lock file",
                    path.display()
                )
            })?;
            let text =
                std::fs::read_to_string(&lock).map_err(|e| format!("{}: {e}", lock.display()))?;
            let packages = adv::locked_packages(&text);
            let hits = adv::check_lockfile(&corpus, &text).map_err(|e| e.to_string())?;
            let affected_packages = hits
                .iter()
                .map(|(package, ..)| package)
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            println!(
                "{} — {} locked packages; {} of them are named by {} advisory record(s):",
                lock.display(),
                packages.len(),
                affected_packages,
                hits.len()
            );
            let mut last = String::new();
            for (package, version, advisory, outcome) in &hits {
                if package != &last {
                    println!("  {package} {version}");
                    last = package.clone();
                }
                let verdict = match outcome {
                    adv::Outcome::Vulnerable { fix: Some(fix) } => {
                        format!("affected, {} is offered as safe", fix)
                    }
                    adv::Outcome::Vulnerable { fix: None } => {
                        "affected, no safe version".to_string()
                    }
                    adv::Outcome::Notice { kind } => format!("informational ({kind})"),
                    other => format!("{other:?}"),
                };
                println!(
                    "    {} [{}] {} — {}",
                    advisory.id,
                    advisory.corpus.label(),
                    advisory.date,
                    verdict
                );
                if let Some(url) = &advisory.url {
                    println!("      {url}");
                }
            }
            if hits.is_empty() {
                println!(
                    "  nothing — but that means no advisory matches these versions, not that \
                     the dependencies are safe"
                );
            }
            Ok(())
        }
        AdvisoryAction::Status { dir } => {
            let corpus = advisory_corpus(dir)?;
            match adv::sync_info(&corpus) {
                Ok(info) => {
                    println!("corpus: {}", corpus.display());
                    println!(
                        "  {} RustSec advisories at revision {}, {} OSV records, {} index lines",
                        info.rustsec_advisories,
                        info.rustsec_revision,
                        info.osv_records,
                        info.index_lines
                    );
                    println!("  synced {}", adv::age_days(info.synced_at_unix));
                }
                Err(e) => println!("{e}"),
            }
            Ok(())
        }
    }
}

/// Parse `--json-schema` strictly: invalid JSON is a user error, not a
/// string to send. Pure — unit-tested.
fn parse_json_schema(schema: Option<String>) -> Result<Option<serde_json::Value>, String> {
    schema
        .map(|s| {
            serde_json::from_str(&s)
                .map_err(|e| format!("invalid --json-schema (must be JSON): {e}"))
        })
        .transpose()
}

/// Whether an answer may be given as one that satisfies `--json-schema`. With no
/// schema declared there is nothing to satisfy; with one, the text has to be that
/// JSON and nothing else.
fn structured_answer_fits(answer: &str, schema: Option<&serde_json::Value>) -> bool {
    match schema {
        None => true,
        Some(schema) => xencode_providers_rs::schema::read_answer(answer, schema).is_ok(),
    }
}

#[allow(clippy::too_many_arguments)] // CLI flags map 1:1 to sampling options; a struct would just rename them
async fn run_query(
    prompt: String,
    model_override: Option<String>,
    no_cache: bool,
    session_id: Option<String>,
    temperature: Option<f64>,
    top_k: Option<i32>,
    min_p: Option<f64>,
    mirostat: Option<i32>,
    seed: Option<i64>,
    max_tokens: Option<u32>,
    grammar: Option<String>,
    json_schema: Option<String>,
    images: Vec<String>,
    format: QueryFormat,
) -> Result<(), String> {
    let outcome = run_query_once(
        prompt,
        model_override,
        no_cache,
        session_id,
        temperature,
        top_k,
        min_p,
        mirostat,
        seed,
        max_tokens,
        grammar,
        json_schema,
        images,
        format,
    )
    .await;
    // One terminating line, whatever failed. Without this a script sees an
    // empty stream and cannot tell "no answer" from "still running".
    if let (Err(message), QueryFormat::Ndjson) = (&outcome, format) {
        println!("{}", query_stream::error(message));
        let _ = io::stdout().flush();
    }
    outcome
}

#[allow(clippy::too_many_arguments)] // CLI flags map 1:1 to sampling options; a struct would just rename them
async fn run_query_once(
    prompt: String,
    model_override: Option<String>,
    no_cache: bool,
    session_id: Option<String>,
    temperature: Option<f64>,
    top_k: Option<i32>,
    min_p: Option<f64>,
    mirostat: Option<i32>,
    seed: Option<i64>,
    max_tokens: Option<u32>,
    grammar: Option<String>,
    json_schema: Option<String>,
    images: Vec<String>,
    format: QueryFormat,
) -> Result<(), String> {
    let ndjson = format == QueryFormat::Ndjson;
    // Read before the cache is consulted or anything is printed: a schema that
    // is not JSON is the caller's mistake, and a schema that is one is a promise
    // about the answer. Same for images: an unreadable one is the caller's
    // mistake, and it is a mistake rather than a silent drop either way.
    let schema = parse_json_schema(json_schema)?;
    // Same intake the TUI's attach path uses: read, cap, verify it really is an
    // image, shrink to what a vision encoder can use, and report what changed —
    // so a re-encoded file is visible rather than quiet.
    let image_data_urls = images
        .iter()
        .map(|path| encode_query_image(path))
        .collect::<Result<Vec<_>, _>>()?;
    let started = std::time::Instant::now();
    let config = XencodeConfig::load().unwrap_or_default();
    let client = OllamaClient::new(&config.ollama_url, config.response_timeout);

    // If no model override is provided, verify default model against Ollama's installed models
    let named_a_model = model_override.is_some();
    let model = match model_override {
        Some(m) => m,
        None => {
            if !config.default_model.contains('/') {
                if let Ok(installed) = client.list_models().await {
                    if !installed.is_empty()
                        && !installed.iter().any(|m| m.name == config.default_model)
                    {
                        // Configured default is not installed; use smart default or first installed
                        if let Ok(Some(smart)) = client.get_smart_default().await {
                            smart
                        } else if let Some(first) = installed.first() {
                            first.name.clone()
                        } else {
                            config.default_model.clone()
                        }
                    } else {
                        config.default_model.clone()
                    }
                } else {
                    config.default_model.clone()
                }
            } else {
                config.default_model.clone()
            }
        }
    };

    // A saved profile marked for the kind of work this prompt reads as takes the
    // turn (MI-7) — unless `--model` named one, which is an answer rather than a
    // question, and a rule does not overrule it. The sampling rides with the
    // profile only where a flag left it unset, for the same reason. Said on stderr
    // before anything about the request, so `-f ndjson` output stays parsable and a
    // script's `start` line still names the model that will actually answer.
    let mut model = model;
    let mut temperature = temperature;
    let mut max_tokens = max_tokens;
    if !named_a_model {
        let choice = xencode_tui_rs::task_profiles::choose_profile(&config, &model, &prompt);
        if let Some(profile) = choice.profile() {
            model = profile.model.clone();
            if temperature.is_none() {
                temperature = profile.temperature;
            }
            if max_tokens.is_none() {
                max_tokens = profile.max_tokens;
            }
        }
        if let Some(note) = choice.note() {
            eprintln!("profile: {note}");
        }
    }

    let mut cache = if config.cache_enabled && !no_cache {
        ResponseCache::with_persistence(config.max_cache_size, config.response_timeout as f64).ok()
    } else {
        None
    };

    let mut memory = if config.memory_enabled {
        ConversationMemory::with_persistence(config.max_memory_items).ok()
    } else {
        None
    };

    if let Some(ref mut mem) = memory {
        if let Some(ref sid) = session_id {
            mem.switch_session(sid);
        } else {
            mem.start_session(None);
        }
    }

    // Where the prompt is going, from the same prefix rules the router walks.
    // Reported before anything else so a stream that never finishes still says
    // which model it was waiting on.
    let routing = xencode_providers_rs::RoutingFacts {
        openrouter_key: config.api_keys.has_secret(SecretProvider::OpenRouter),
        remote_host: (!config.remote_base_url.is_empty())
            .then(|| xencode_providers_rs::url_host(&config.remote_base_url))
            .flatten(),
    };
    if ndjson {
        let session = memory.as_ref().and_then(|mem| mem.current_session());
        println!(
            "{}",
            query_stream::start(
                &model,
                xencode_providers_rs::provider_for(&model, routing),
                source_word(xencode_providers_rs::classify(&model, routing)),
                session.map(String::as_str),
            )
        );
        let _ = io::stdout().flush();
    }

    if let Some(ref mut c) = cache {
        if let Some(cached_resp) = c.get(&prompt, &model) {
            // A cached reply was written against whatever was asked the first
            // time, so it is only this run's answer if it still fits the schema
            // being declared now. If it does not, the model is asked again.
            if structured_answer_fits(&cached_resp, schema.as_ref()) {
                if ndjson {
                    // The answer travels as a `token` line too, so a script that
                    // concatenates tokens always has the whole reply — cached or
                    // not. Nothing was generated, so there are no token counts.
                    println!("{}", query_stream::token(&cached_resp));
                    println!(
                        "{}",
                        query_stream::done(&cached_resp, true, elapsed_ms(started), None, None,)
                    );
                } else {
                    println!("{}", cached_resp);
                }

                if let Some(ref mut mem) = memory {
                    mem.add_message("user", &prompt, None);
                    mem.add_message("assistant", &cached_resp, Some(model.clone()));
                }
                return Ok(());
            }
        }
    }

    // Project context injection, same engine as the TUI live path: a
    // byte-stable system head (KV-cacheable) + budgeted history turns +
    // retrieval/state/git riding in the final user turn (§10 tiers).
    // Memory is persisted after the generation below, so this snapshot holds
    // prior turns only — the current prompt arrives separately and unsqueezable.
    let history: Vec<(String, String)> = memory
        .as_ref()
        .map(|mem| {
            mem.get_context(25)
                .into_iter()
                .map(|m| (m.role, m.content))
                .collect()
        })
        .unwrap_or_default();
    let root = xencode_context_rs::default_root();
    // How much of this window project context may fill is a profile decision,
    // and the profile is no longer a constant: the config can name one, and if
    // it does not, the memory this machine reports chooses. What was chosen and
    // why is printed, because a budget that trims someone's context silently is
    // worse than a budget that says so.
    let hardware = xencode_context_rs::ProfileDecision::resolve(&config.hardware_profile);
    eprintln!("hardware: {}", hardware.describe());
    // A command that runs once has no earlier request to measure this turn's
    // prompt against, so it has nothing to scale retrieval from and keeps the
    // profile's own numbers. The interactive session sizes them per turn instead.
    let caps = xencode_context_rs::ContextCaps::from_profile(hardware.profile);
    eprintln!(
        "retrieval: up to {} files, {} characters each",
        caps.top_k, caps.content_cap_chars
    );
    let live = xencode_context_rs::collect_live_context(&root, &prompt, caps);
    // Which weights retrieval used, and the words in the prompt that chose them:
    // a turn read as bugfix scores differently from one read as a rename, and
    // without this line the difference looks like a random result.
    eprintln!(
        "read as {} work — {}",
        live.shape.shape,
        live.shape.reasons.join("; ")
    );
    // The window the server is actually running with, when the model is served
    // by a llama.cpp process that will say. A family table cannot know `-c`;
    // every other route keeps the table's answer.
    let probe = xencode_providers_rs::routes_to_llamacpp(&model)
        .then(|| LlamaCppClient::new(&config.llama_cpp_url, 3));
    let server_window = match &probe {
        Some(client) => client.context_window().await.unwrap_or(None),
        None => None,
    };
    if let Some(tokens) = server_window {
        eprintln!(
            "context: {tokens}-token window reported by the server at {}",
            config.llama_cpp_url
        );
    }
    // What the model itself says about the window its weights hold and whether
    // it claims it can think, asked of Ollama when this model goes there (MI-2).
    // The answer settles two numbers that have to be one number: the window
    // written into the request and the window the context below is filled for. A
    // turn budgeted past what the server was told to open is refused part-way
    // through with `exceed_context_size_error` rather than quietly truncated.
    let on_ollama = xencode_providers_rs::routes_to_ollama(&model);
    let ollama_show = if on_ollama {
        OllamaClient::new(&config.ollama_url, 3)
            .show_model(&model)
            .await
            .ok()
    } else {
        None
    };
    let mut ollama_request = xencode_providers_rs::OllamaRequest::from_settings(
        config.ollama_reasoning.as_deref(),
        config.ollama_keep_alive.as_deref(),
    )
    .map_err(|e| e.to_string())?;
    let mut context_window = xencode_providers_rs::effective_context_window(&model, server_window);
    if on_ollama {
        // What to ask for before the model gets a say: what this machine's
        // profile can serve, never the family table's advertised window.
        ollama_request.num_ctx = Some(xencode_providers_rs::ollama_window_asked(
            context_window,
            hardware.profile.ctx_tokens() as u32,
        ));
        let (decided, notes) =
            xencode_providers_rs::ollama_request_for(ollama_show.as_ref(), ollama_request);
        ollama_request = decided;
        context_window = ollama_request.num_ctx;
        if let Some(tokens) = context_window {
            eprintln!("context: asking Ollama for a {tokens}-token window");
        }
        for note in notes {
            eprintln!("context: {note}");
        }
        // Two controls Ollama's `/api/chat` has nowhere to put. Asking for one and
        // receiving an ordinary answer would be a silence nobody could find except
        // in the source, so each is said as it is dropped.
        if grammar.is_some() {
            eprintln!(
                "sampling: --grammar was not sent — Ollama takes a JSON schema in `format`, not a GBNF grammar"
            );
        }
        if mirostat.is_some() {
            eprintln!(
                "sampling: --mirostat was not sent — a running Ollama answers it as `invalid option provided`"
            );
        }
    }
    let assembly = xencode_context_rs::assemble_chat(xencode_context_rs::ChatInput {
        profile: hardware.profile,
        context_window,
        system: xencode_context_rs::prompts::AGENT_SYSTEM,
        agents_md: live.agents_md.as_deref(),
        anchor_md: live.anchor_md.as_deref(),
        scoped_md: live.scoped_md.as_deref(),
        state_md: live.state_md.as_deref(),
        notes_md: live.notes_md.as_deref(),
        git_summary: &live.git_summary,
        repo_map: &live.repo_map,
        retrieved: live.blocks,
        attached_block: "",
        history: &history,
        prompt: &prompt,
    });
    // What the prompt costs in the model's own vocabulary, asked of the server
    // once before the request goes out — the only moment a count exists at all,
    // since a reply's usage figures arrive after it has been paid for. The
    // budgeter's figure is printed beside it, and neither is a better number than
    // the other: a plain `/tokenize` of the whole prompt leaves out the framing a
    // chat template adds around each message — 13 tokens for the two-message
    // request measured here, out of 5,753 — while the budgeter's figure stops at
    // what it was allowed to spend, because the question and any attached files
    // are the parts it may not trim. Printed together, a run that is about to
    // overflow shows it. Measured on a 512-token server here: the arithmetic said
    // 384, the server counted 566 of the same turn, and the request was refused at
    // 579 tokens.
    let counted = match &probe {
        Some(client) => Some(
            client
                .count_tokens(&assembly.prompt_text())
                .await
                .map_err(|e| e.to_string()),
        ),
        None => None,
    };
    match counted {
        Some(Ok(Some(tokens))) => {
            eprintln!(
                "context: {tokens} tokens counted by the server, {} by character arithmetic",
                assembly.total_tokens
            );
            if let Some(window) = server_window {
                if tokens > window as u64 {
                    eprintln!(
                        "warning: the prompt was counted at {tokens} tokens, which is more than the {window}-token window this server is running with"
                    );
                }
            }
        }
        Some(Ok(None)) => {
            // The server answered the count request and refused it — a build
            // without /tokenize, or one that rejected this text.
            eprintln!(
                "context: {} tokens by character arithmetic (this server gave no count for /tokenize)",
                assembly.total_tokens
            );
        }
        Some(Err(reason)) => {
            // Nothing was ever counted, so the arithmetic line is not "the server
            // declined" — it is the only number there is because the ask failed.
            eprintln!(
                "context: {} tokens by character arithmetic (no count could be asked of this server: {reason})",
                assembly.total_tokens
            );
        }
        None => {}
    }
    if !live.index_present {
        eprintln!(
            "hint: run `xencode` → /init once for project-aware answers (continuing with guidelines + history only)."
        );
    }
    let mut context_messages: Vec<ChatMessage> = assembly
        .turns
        .into_iter()
        .map(|t| ChatMessage {
            role: t.role,
            content: t.content.into(),
        })
        .collect();
    // Images ride as message parts on the final user turn, never inlined into
    // the prompt text — same shape the TUI's attach path builds.
    if !image_data_urls.is_empty()
        && !attach_images_to_final_user_message(&mut context_messages, image_data_urls)
    {
        return Err(
            "the attached images could not be sent with this turn (no final user message) \
             — they were dropped, not seen by the model"
                .to_string(),
        );
    }

    let client = OllamaClient::new(&config.ollama_url, config.response_timeout);
    let llama_client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);

    let llamacpp_opts = LlamaCppOptions {
        temperature,
        top_k,
        min_p,
        mirostat,
        max_tokens,
        seed,
        grammar,
        json_schema: schema.clone(),
    };

    let provider = ProviderManager::new(
        client,
        api_key(&config, SecretProvider::OpenRouter),
        api_key(&config, SecretProvider::Qwen),
        api_key(&config, SecretProvider::Gemini),
        None,
    )
    .with_llama_cpp(llama_client)
    .with_remote(
        &config.remote_base_url,
        api_key(&config, SecretProvider::Remote),
    )
    .with_nvidia(api_key(&config, SecretProvider::Nvidia))
    .with_request_timeout(config.response_timeout)
    .with_ollama_request(ollama_request)
    .with_egress_policy(EgressPolicy::new(config.allow_cloud_models));

    let mut response_content = String::new();
    let result = provider
        .generate_stream_with_options(&model, &context_messages, Some(&llamacpp_opts), |token| {
            if ndjson {
                println!("{}", query_stream::token(token));
            } else {
                print!("{}", token);
            }
            let _ = io::stdout().flush();
            response_content.push_str(token);
        })
        .await;

    match result {
        Ok(_) => {
            // The tokens have already streamed, so this cannot unsay them; what
            // it can do is end the stream with a failure instead of a completion,
            // and keep an answer that does not fit out of the cache and the
            // conversation.
            if let Some(ref schema) = schema {
                if let Err(reason) =
                    xencode_providers_rs::schema::read_answer(&response_content, schema)
                {
                    return Err(format!("the answer does not fit --json-schema: {reason}"));
                }
            }
            let timings = provider.last_llamacpp_timings();
            if ndjson {
                println!(
                    "{}",
                    query_stream::done(
                        &response_content,
                        false,
                        elapsed_ms(started),
                        timings.as_ref().map(|t| t.tokens_generated),
                        timings.as_ref().map(|t| t.predicted_per_second),
                    )
                );
                let _ = io::stdout().flush();
            } else {
                println!(); // Ensure final newline
                if let Some(timings) = timings {
                    println!(
                        "\n(llama.cpp ~{} tok/s · {} tokens generated)",
                        timings.predicted_per_second as u64, timings.tokens_generated
                    );
                }
            }
            if let Some(ref mut c) = cache {
                c.set(&prompt, &model, &response_content);
            }
            if let Some(ref mut mem) = memory {
                mem.add_message("user", &prompt, None);
                mem.add_message("assistant", &response_content, Some(model));
            }
            Ok(())
        }
        Err(e) => Err(format!("Query failed: {}", e)),
    }
}

/// Fold image data URLs into the final user turn as content parts, keeping the
/// assembled prompt text ahead of them. Returns false when there is no user turn
/// to attach to, which the caller reports rather than sending a request the
/// model would answer without ever seeing the images.
fn attach_images_to_final_user_message(messages: &mut [ChatMessage], urls: Vec<String>) -> bool {
    let Some(last) = messages.iter_mut().rev().find(|m| m.role == "user") else {
        return false;
    };
    let text = last.text_content();
    let mut parts = Vec::with_capacity(urls.len() + 1);
    if !text.is_empty() {
        parts.push(ContentPart::Text { text });
    }
    parts.extend(urls.into_iter().map(|url| ContentPart::ImageUrl {
        image_url: ImageUrlPart { url, detail: None },
    }));
    last.content = MessageContent::Parts(parts);
    true
}

/// Read one `--image` path into the base64 data URL a vision request carries,
/// applying the same size cap, format check, and downscale the TUI's attach
/// path does. Errors name the file so a mistyped path is obvious.
fn encode_query_image(path: &str) -> Result<String, String> {
    use xencode_analysis_rs::{
        inspect_bytes, prepare_for_send, to_data_url, ImageError, MAX_IMAGE_BYTES,
    };
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read image {path}: {e}"))?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(format!(
            "{path} exceeds the {} MiB image cap",
            MAX_IMAGE_BYTES / 1024 / 1024
        ));
    }
    inspect_bytes(path, &bytes).map_err(|e| match e {
        ImageError::UnknownFormat(_) => format!("{path} is not a recognized image"),
        other => format!("{path}: {other}"),
    })?;
    let prepared = prepare_for_send(&bytes);
    if let Some(note) = prepared.summary() {
        eprintln!("image: {path} ({note})");
    }
    Ok(to_data_url(prepared.format, &prepared.bytes))
}

/// Milliseconds since this command started, including config load and context
/// build.
fn elapsed_ms(started: std::time::Instant) -> u64 {
    (started.elapsed().as_secs_f64() * 1_000.0) as u64
}

/// The destination word the stream writes. Kept as one function so a test can
/// pin it against the spelling the metrics rows already use.
fn source_word(egress: xencode_providers_rs::Egress) -> &'static str {
    match egress {
        xencode_providers_rs::Egress::Local => "local",
        xencode_providers_rs::Egress::Cloud => "cloud",
    }
}

fn run_memory(action: MemoryAction) -> Result<(), String> {
    let mut mem = ConversationMemory::with_persistence(50).map_err(|e| e.to_string())?;

    match action {
        MemoryAction::List { all } => {
            let all_sessions = mem.list_all_sessions();
            let non_empty = mem.list_sessions();
            let empty_count = all_sessions.len().saturating_sub(non_empty.len());
            let sessions = if all { all_sessions } else { non_empty };
            if sessions.is_empty() {
                if empty_count > 0 {
                    println!(
                        "No non-empty conversation sessions found ({} empty session(s) omitted; use --all to show, or `xencode memory prune` to delete).",
                        empty_count
                    );
                } else {
                    println!("No conversation sessions found.");
                }
            } else {
                println!("Conversation Sessions:");
                for s in &sessions {
                    if all {
                        let count = mem
                            .get_session(s)
                            .map(|sess| sess.messages.len())
                            .unwrap_or(0);
                        println!(
                            "  {} ({} message{})",
                            s,
                            count,
                            if count == 1 { "" } else { "s" }
                        );
                    } else {
                        println!("  {}", s);
                    }
                }
                if !all && empty_count > 0 {
                    println!(
                        "\n  ({} empty session(s) omitted; use --all to show, or `xencode memory prune` to delete)",
                        empty_count
                    );
                }
            }
            Ok(())
        }
        MemoryAction::Show { session, first } => {
            if let Some(sess) = mem.get_session(&session) {
                if first {
                    if let Some(msg) = sess.first_message() {
                        let role = msg.role.to_uppercase();
                        println!("[{}] {}", role, msg.timestamp);
                        println!("{}\n", msg.content);
                    } else {
                        println!("Session {} has no messages.", session);
                    }
                } else if sess.messages.is_empty() && sess.events.is_empty() {
                    println!("Session {} is empty (0 messages).", session);
                } else if !sess.events.is_empty() {
                    for ev in &sess.events {
                        let role = ev.message.role.to_uppercase();
                        println!("[{}] {}", role, ev.timestamp);
                        println!("{}\n", ev.message.content);
                    }
                } else {
                    for msg in &sess.messages {
                        let role = msg.role.to_uppercase();
                        println!("[{}] {}", role, msg.timestamp);
                        println!("{}\n", msg.content);
                    }
                }
            } else {
                println!("Session not found: {}", session);
            }
            Ok(())
        }
        MemoryAction::Fork {
            session,
            as_id,
            prefix,
        } => {
            let child_id = mem
                .fork_session(&session, as_id, prefix)
                .map_err(|e| e.to_string())?;
            println!("Forked session {} into {}", session, child_id);
            Ok(())
        }
        MemoryAction::Prune => {
            let pruned = mem.prune_empty_sessions();
            if pruned == 0 {
                println!("No empty conversation sessions to prune.");
            } else {
                println!(
                    "Pruned {} empty conversation session{}.",
                    pruned,
                    if pruned == 1 { "" } else { "s" }
                );
            }
            Ok(())
        }
        MemoryAction::Gc { apply } => {
            let xencode = xencode_context_rs::default_root().join(xencode_context_rs::XENCODE_DIR);
            let now = xencode_context_rs::gc_now_ms();
            let report =
                xencode_context_rs::collect_gc(&xencode, now, apply).map_err(|e| e.to_string())?;
            println!(
                "Durable facts: {} contradicted, {} past {} months, {} removed",
                report.pending.len(),
                report.aged.len(),
                xencode_context_rs::RETIRE_AFTER_MONTHS,
                report.removed.len()
            );
            for entry in report.pending.iter().take(20) {
                println!(
                    "  {} — {}: {}",
                    xencode_context_rs::age_words(entry.months(now)),
                    entry.problem.reason(),
                    xencode_context_rs::fact_prose(&entry.fact)
                );
            }
            if report.pending.len() > 20 {
                println!("  … and {} more", report.pending.len() - 20);
            }
            // The queue is what makes retirement a decision rather than a hunch,
            // so a run that cannot offer any has to say why: twelve months of
            // unbroken contradiction start counting the first time this looked.
            if report.removed.is_empty() && !report.aged.is_empty() {
                println!(
                    "  `--apply` retires the {} above; a fact that stops being contradicted leaves the queue instead of ageing toward removal.",
                    report.aged.len()
                );
            }
            if !report.retired.is_empty() {
                println!("  {} retired by an earlier run.", report.retired.len());
            }
            if report.unverifiable > 0 {
                println!(
                    "  {} stayed in the prompt because this repository could not check them.",
                    report.unverifiable
                );
            }
            if report.disagreeing > 0 {
                println!(
                    "  {} that the code places in another file. Those are disagreement, not staleness: `xencode doctor` names them and no sweep removes them.",
                    report.disagreeing
                );
            }
            if report.pending.is_empty()
                && report.retired.is_empty()
                && report.unverifiable == 0
                && report.disagreeing == 0
            {
                println!("  Nothing to collect: this repository agrees with every durable fact it can check.");
            }
            Ok(())
        }
        MemoryAction::Evidence { format } => {
            let xencode = xencode_context_rs::default_root().join(xencode_context_rs::XENCODE_DIR);
            let rows = xencode_context_rs::evidence_rows(&xencode);
            match format {
                OutputFormat::Json => {
                    let body: Vec<serde_json::Value> = rows
                        .iter()
                        .map(|row| {
                            serde_json::json!({
                                "fact": row.fact,
                                "revisions_checked": row.trials,
                                "survived": row.survived,
                                "unchecked": row.unchecked,
                                "wilson_95_of_next_check_agreeing": row.interval.map(|(lower, upper)| {
                                    [
                                        (lower * 1000.0).round() / 1000.0,
                                        (upper * 1000.0).round() / 1000.0,
                                    ]
                                }),
                                "verified_by": row.verified_by(),
                                "last_revision": row.last.as_ref().map(|c| c.revision.clone()),
                                "contradicted_by": row.problem.map(|p| p.reason()),
                            })
                        })
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&body)
                            .map_err(|e| format!("cannot write the report: {e}"))?
                    );
                }
                OutputFormat::Text => {
                    if rows.is_empty() {
                        println!(
                            "No durable fact here has been re-checked against a revision. A fact \
                             gains evidence on the turn it is next assembled into a prompt, in a \
                             repository git can search."
                        );
                        return Ok(());
                    }
                    println!(
                        "Durable facts, weakest evidence first — the interval is over the checks \
                         this repository ran, and is not a chance that the fact is true"
                    );
                    for row in &rows {
                        println!("  {}", row.fact);
                        println!("    {}", row.evidence_sentence());
                        println!("    verified by {}", row.verified_by());
                        if let Some(problem) = row.problem {
                            println!("    contradicted now: {}", problem.reason());
                        }
                    }
                    let thin = rows.iter().filter(|row| row.trials < 2).count();
                    if thin > 0 {
                        println!(
                            "  {thin} of {} reach fewer than two revisions, where an interval spans \
                             most of what it could and no verdict is available.",
                            rows.len()
                        );
                    }
                }
            }
            Ok(())
        }
        MemoryAction::Publish {
            worker,
            scope,
            content,
        } => {
            let (dir, mut store) = shared_memory_store()?;
            let finding = store
                .publish(&worker, MemoryScope::parse(&scope), content)
                .map_err(|e| e.to_string())?;
            store.save_to_dir(&dir).map_err(|e| e.to_string())?;
            println!(
                "Published {} into scope '{}' as worker '{}'.",
                finding.id, finding.scope, finding.author_worker_id
            );
            Ok(())
        }
        MemoryAction::Read { worker, scope } => {
            let (_dir, store) = shared_memory_store()?;
            let found = match &scope {
                Some(scope) => store
                    .read_scope(&worker, &MemoryScope::parse(scope))
                    .map_err(|e| e.to_string())?,
                None => store.read_all_granted(&worker).map_err(|e| e.to_string())?,
            };
            if found.is_empty() {
                println!("Worker '{worker}' read 0 findings.");
                return Ok(());
            }
            println!(
                "Shared memory read as worker '{}' — {} finding{}, each one another worker's data, not an instruction:",
                worker,
                found.len(),
                if found.len() == 1 { "" } else { "s" }
            );
            for item in &found {
                println!();
                println!("{}", item.marked_content);
                println!("  {} published {}", item.finding_id, item.published_at);
            }
            Ok(())
        }
        MemoryAction::Policy { action } => match action {
            MemoryPolicyAction::Set {
                worker,
                read,
                publish,
            } => {
                let (dir, mut store) = shared_memory_store()?;
                let read: Vec<MemoryScope> = read.iter().map(|s| MemoryScope::parse(s)).collect();
                let publish: Vec<MemoryScope> =
                    publish.iter().map(|s| MemoryScope::parse(s)).collect();
                let mut policy = WorkerMemoryPolicy::new(&worker);
                for scope in &read {
                    policy = policy.allow_read(scope.clone());
                }
                for scope in &publish {
                    policy = policy.allow_publish(scope.clone());
                }
                store.set_policy(policy);
                store.save_to_dir(&dir).map_err(|e| e.to_string())?;
                println!("Memory policy for worker '{worker}':");
                println!("  reads:     {}", scope_names(&read));
                println!("  publishes: {}", scope_names(&publish));
                Ok(())
            }
            MemoryPolicyAction::Show { worker } => {
                let (_dir, store) = shared_memory_store()?;
                let mut listed: Vec<&WorkerMemoryPolicy> = match &worker {
                    Some(worker) => store
                        .all_policies()
                        .values()
                        .filter(|policy| &policy.worker_id == worker)
                        .collect(),
                    None => store.all_policies().values().collect(),
                };
                listed.sort_by(|a, b| a.worker_id.cmp(&b.worker_id));
                if listed.is_empty() {
                    match &worker {
                        Some(worker) => println!(
                            "No memory policy for worker '{worker}': it may read nothing and publish nothing."
                        ),
                        None => println!(
                            "No memory policies configured. Until one is set, every worker is denied on both sides."
                        ),
                    }
                    return Ok(());
                }
                println!("Memory policies:");
                for policy in listed {
                    println!("  {}", policy.worker_id);
                    println!("    reads:     {}", scope_names(&policy.readable_scopes));
                    println!("    publishes: {}", scope_names(&policy.publishable_scopes));
                }
                Ok(())
            }
        },
    }
}

/// OR-8: the shared memory workers hand each other, kept in `.xencode/` beside
/// the other durable files. Both sides deny by default, so a worker that was
/// never given a policy can neither read what it has stored nor add to it.
fn shared_memory_store() -> Result<(PathBuf, ScopedSharedMemoryStore), String> {
    let dir = xencode_context_rs::default_root().join(xencode_context_rs::XENCODE_DIR);
    let store = ScopedSharedMemoryStore::load_from_dir(&dir).map_err(|e| e.to_string())?;
    Ok((dir, store))
}

/// Scopes in a stable order. A policy grants a hash set, and a listing that
/// changed shape between two runs of the same command cannot be compared.
fn scope_names<'a>(scopes: impl IntoIterator<Item = &'a MemoryScope>) -> String {
    let mut names: Vec<String> = scopes.into_iter().map(|scope| scope.to_string()).collect();
    names.sort();
    names.dedup();
    if names.is_empty() {
        "nothing".to_string()
    } else {
        names.join(", ")
    }
}

/// Registry location: `<default_root>/.xencode/tasks`, the same dir layout
/// `xencode init` uses for project state.
fn tasks_registry() -> xencode_core_rs::FileTaskRegistry {
    let root = xencode_context_rs::default_root()
        .join(xencode_context_rs::XENCODE_DIR)
        .join("tasks");
    xencode_core_rs::FileTaskRegistry::new(root)
}

fn run_tasks(action: TaskAction) -> Result<(), String> {
    let reg = tasks_registry();
    match action {
        TaskAction::List { json } => {
            let entries = reg.poll().map_err(|e| e.to_string())?;
            if json {
                let rows: Vec<serde_json::Value> = entries
                    .values()
                    .map(|(t, status)| {
                        serde_json::json!({
                            "id": t.id,
                            "name": t.name,
                            "command": t.command,
                            "pid": t.pid,
                            "started_at": t.started_at,
                            "status": status.label(),
                        })
                    })
                    .collect();
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?
                );
            } else if entries.is_empty() {
                println!("No background tasks.");
            } else {
                println!("{:>4}  {:<12} {:>8}  NAME", "ID", "STATUS", "PID");
                for (task, status) in entries.values() {
                    println!(
                        "{:>4}  {:<12} {:>8}  {}",
                        task.id,
                        status.label(),
                        task.pid,
                        task.name
                    );
                }
            }
            Ok(())
        }
        TaskAction::Start { command, name } => {
            let task = reg
                .start(name.as_deref().unwrap_or(""), &command)
                .map_err(|e| e.to_string())?;
            println!(
                "started task {} (pid {}): {}",
                task.id, task.pid, task.command
            );
            Ok(())
        }
        TaskAction::Poll { id, lines } => {
            let task = reg
                .list()
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|t| t.id == id)
                .ok_or_else(|| format!("no such task: {id}"))?;
            println!(
                "task {} [{}] {} (pid {})",
                task.id,
                reg.status(&task).label(),
                task.command,
                task.pid
            );
            let output = reg.output(id, lines);
            if output.is_empty() {
                println!("  (no output)");
            } else {
                for line in output {
                    println!("  {line}");
                }
            }
            Ok(())
        }
        TaskAction::Stop { id } => {
            reg.stop(id).map_err(|e| e.to_string())?;
            println!("stopped task {id}");
            Ok(())
        }
        TaskAction::Rm { id } => {
            reg.remove(id).map_err(|e| e.to_string())?;
            println!("removed task {id}");
            Ok(())
        }
    }
}

fn run_worktree(action: WorktreeAction) -> Result<(), String> {
    use xencode_context_rs::{worktree_add, worktree_list, worktree_remove, WorktreeInfo};
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let print_list = |list: &[WorktreeInfo]| {
        println!("{:<10} {:<10} PATH", "BRANCH", "HEAD");
        for wt in list {
            let tag = if wt.is_main { " [main]" } else { "" };
            println!(
                "{:<10} {:<10} {}{tag}",
                wt.display_branch(),
                wt.short_head(),
                wt.path.display(),
            );
        }
    };
    match action {
        WorktreeAction::List => {
            let list = worktree_list(&root)?;
            print_list(&list);
            Ok(())
        }
        WorktreeAction::Add { path, branch } => {
            let list = worktree_add(&root, std::path::Path::new(&path), branch.as_deref(), false)?;
            println!("added worktree {path}");
            print_list(&list);
            Ok(())
        }
        WorktreeAction::Remove { path } => {
            if path == "." || std::path::Path::new(&path) == root {
                return Err("the main worktree is not removable".to_string());
            }
            let list = worktree_remove(&root, std::path::Path::new(&path), false)?;
            println!("removed worktree {path}");
            print_list(&list);
            Ok(())
        }
    }
}

/// Findings for `root`'s `.xencode` snapshot, narrowed by `filter`
/// (substring of the file path). The read+analysis half of `xencode advise`,
/// unit-tested without the print layer.
fn compute_advise(
    root: &std::path::Path,
    filter: Option<&str>,
) -> Result<Vec<xencode_context_rs::Advice>, String> {
    let mut items = xencode_context_rs::advise_from_snapshot(root).map_err(|e| e.to_string())?;
    if let Some(needle) = filter {
        items.retain(|a| a.file.contains(needle));
    }
    Ok(items)
}

fn run_advise(filter: Option<String>, json: bool, limit: usize) -> Result<(), String> {
    let root = xencode_context_rs::default_root();
    let filter = filter.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let mut items = compute_advise(&root, filter)?;
    let total = items.len();
    if limit > 0 {
        items.truncate(limit);
    }
    if json {
        let rendered: Vec<serde_json::Value> = items
            .iter()
            .map(|a| {
                serde_json::json!({
                    "file": a.file,
                    "kind": format!("{:?}", a.kind),
                    "message": a.message,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&rendered).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if items.is_empty() {
        println!("no findings — the symbol graph is clean.");
        return Ok(());
    }
    for (i, a) in items.iter().enumerate() {
        println!(
            "{:>2}. {:<19} {}",
            i + 1,
            format!("{:?}", a.kind),
            a.message
        );
    }
    if total > items.len() {
        println!(
            "… +{} more (raise --limit, or pass a path filter)",
            total - items.len()
        );
    }
    Ok(())
}

/// What `xencode server` decided about its bind, before any socket exists.
#[derive(Debug, PartialEq, Eq)]
struct ServerBind {
    addr: std::net::SocketAddr,
    /// (cert, key) when TLS was requested.
    tls: Option<(PathBuf, PathBuf)>,
    /// Non-fatal condition the operator must see on startup.
    warning: Option<String>,
}

/// Decide how the server binds. Pure — no sockets, no filesystem.
///
/// The loopback question is an `IpAddr` property, not a string prefix:
/// `::1` is loopback and `127.0.0.53` is too, while a `starts_with("127.")`
/// check would silently refuse the former and bless lookalikes.
fn resolve_bind(
    host: &str,
    port: u16,
    cert: Option<&std::path::Path>,
    key: Option<&std::path::Path>,
    allow_insecure_public: bool,
) -> Result<ServerBind, String> {
    let ip: std::net::IpAddr = match host {
        "localhost" => std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
        other => other.parse().map_err(|_| {
            format!("invalid --host '{host}': expected an IP address or 'localhost'")
        })?,
    };
    let tls = match (cert, key) {
        (Some(c), Some(k)) => Some((c.to_path_buf(), k.to_path_buf())),
        (Some(_), None) => return Err("--cert given without --key; TLS needs both".to_string()),
        (None, Some(_)) => return Err("--key given without --cert; TLS needs both".to_string()),
        (None, None) => None,
    };
    let public_plain = tls.is_none() && !ip.is_loopback();
    if public_plain && !allow_insecure_public {
        return Err(format!(
            "refusing to bind {ip} without TLS: pass --cert and --key, \
             or --allow-insecure-public to serve plain ws:// on a public interface"
        ));
    }
    let warning = public_plain.then(|| {
        format!(
            "serving WITHOUT TLS on {ip} because --allow-insecure-public was set: \
             tokens and activity travel in clear text"
        )
    });
    Ok(ServerBind {
        addr: std::net::SocketAddr::new(ip, port),
        tls,
        warning,
    })
}

/// Where the audit log goes: `--audit-path none` disables, an explicit path
/// is taken verbatim, and the default is the directory xencode keeps its
/// records in — `~/.local/state/xencode/audit.jsonl`, or `~/.xencode/audit.jsonl`
/// for a person still on the old layout. Sessions are not repo-scoped, so
/// neither is the trail.
fn resolve_audit_path(audit_path: Option<&str>) -> Result<Option<PathBuf>, String> {
    match audit_path {
        Some("none") => Ok(None),
        Some(p) => Ok(Some(PathBuf::from(p))),
        None => Ok(Some(
            xencode_config_rs::paths::state_dir()
                .map_err(|e| e.to_string())?
                .join("audit.jsonl"),
        )),
    }
}

async fn run_server(
    port: u16,
    host: String,
    cert: Option<PathBuf>,
    key: Option<PathBuf>,
    audit_path: Option<String>,
    allow_insecure_public: bool,
) -> Result<(), String> {
    use xencode_server_rs::audit::AuditSink;

    let bind = resolve_bind(
        &host,
        port,
        cert.as_deref(),
        key.as_deref(),
        allow_insecure_public,
    )?;
    if let Some(warning) = &bind.warning {
        println!("⚠ {warning}");
    }

    let audit = resolve_audit_path(audit_path.as_deref())?;
    if let Some(path) = &audit {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
    }
    let state = Arc::new(match &audit {
        Some(path) => ServerState::with_audit(Arc::new(AuditSink::to_file(path))),
        None => ServerState::new(),
    });
    let app = xencode_server_rs::build_app_with_state(state);

    let (http, ws) = if bind.tls.is_some() {
        ("https", "wss")
    } else {
        ("http", "ws")
    };
    println!("🚀 Xencode server starting on {http}://{}", bind.addr);
    println!("   WebSocket: {ws}://{}/ws/{{session_id}}", bind.addr);
    println!(
        "   audit: {}",
        audit
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "disabled".to_string())
    );

    match bind.tls {
        Some((cert_path, key_path)) => {
            let rustls =
                axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
                    .await
                    .map_err(|e| format!("TLS configuration failed: {e}"))?;
            axum_server::bind_rustls(bind.addr, rustls)
                .serve(app.into_make_service())
                .await
                .map_err(|e| e.to_string())
        }
        None => {
            let listener = tokio::net::TcpListener::bind(bind.addr)
                .await
                .map_err(|e| e.to_string())?;
            axum::serve(listener, app)
                .await
                .map_err(|e| e.to_string())?;
            Ok(())
        }
    }
}

/// One-line image inventory for text output. Pure — unit-tested.
fn format_image_text(meta: &ImageMeta) -> String {
    let dims = match (meta.width, meta.height) {
        (Some(w), Some(h)) => format!("{w}x{h}"),
        _ => "vector".to_string(),
    };
    format!(
        "{} — {} {} · {} bytes",
        meta.path,
        meta.format.mime(),
        dims,
        meta.bytes
    )
}

/// Cap for `--format text` page dumps: JSON keeps the whole body, but an
/// unbounded terminal dump helps nobody. The trailer says how much was cut.
/// The number is the one the fetch layer publishes, shared with the agent's
/// `web_fetch` tool so "the same page" is the same size to both readers.
pub const FETCH_TEXT_CAP_CHARS: usize = xencode_analysis_rs::DEFAULT_TEXT_CAP_CHARS;

/// One-screen research summary for a fetched page. Pure — unit-tested.
fn format_fetch_text(page: &FetchedPage) -> String {
    let title = page.title.as_deref().unwrap_or("(no title)");
    let (kept, dropped) = xencode_analysis_rs::cap_chars(&page.text, FETCH_TEXT_CAP_CHARS);
    let body = if dropped > 0 {
        format!("{kept}\n…[truncated to {FETCH_TEXT_CAP_CHARS} chars — use --format json for the full text]")
    } else {
        page.text.clone()
    };
    format!("{}\n{} ({} bytes)\n\n{}", page.url, title, page.bytes, body)
}

async fn run_fetch(url: String, format: OutputFormat) -> Result<(), String> {
    let page = fetch_url(&url).await.map_err(|e| e.to_string())?;
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&page).map_err(|e| e.to_string())?
            );
        }
        OutputFormat::Text => {
            println!("{}", format_fetch_text(&page));
        }
    }
    Ok(())
}

/// One file in a review: diff stats plus working-tree analysis. `note` is
/// set instead of `issues` when the file cannot be analyzed (deleted,
/// binary, unreadable, oversized image) — visible, never silent.
#[derive(Debug)]
struct ReviewedFile {
    path: String,
    added: Option<u64>,
    deleted: Option<u64>,
    issues: Vec<CodeIssue>,
    note: Option<String>,
}

/// Review one diff entry against the working tree.
fn review_file(root: &std::path::Path, diff: &xencode_context_rs::DiffFile) -> ReviewedFile {
    let full = root.join(&diff.path);
    let mut file = ReviewedFile {
        path: diff.path.clone(),
        added: diff.added,
        deleted: diff.deleted,
        issues: Vec::new(),
        note: None,
    };
    if is_image_path(&full) {
        match analyze_image(&full) {
            Ok(meta) => file.note = Some(format_image_text(&meta)),
            Err(e) => file.note = Some(format!("image not readable: {e}")),
        }
        return file;
    }
    match CodeAnalyzer::analyze_file(&full) {
        Ok(issues) => file.issues = issues,
        Err(e) => file.note = Some(format!("not analyzed: {e}")),
    }
    file
}

/// Where the base branch for `xencode review` came from, reported in the
/// triage header so a person knows what baseline was used.
#[derive(Debug, Clone, PartialEq, Eq)]
enum BaseSource {
    Explicit,
    OriginHead,
    InitDefaultBranch(String),
    DefaultFallback,
}

impl BaseSource {
    fn header_suffix(&self) -> String {
        match self {
            BaseSource::Explicit => String::new(),
            BaseSource::OriginHead => " [base resolved from origin/HEAD]".to_string(),
            BaseSource::InitDefaultBranch(name) => {
                format!(" [no remote; fell back to init.defaultBranch ({name})]")
            }
            BaseSource::DefaultFallback => " [no remote; fell back to default 'main']".to_string(),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            BaseSource::Explicit => "explicit",
            BaseSource::OriginHead => "origin/HEAD",
            BaseSource::InitDefaultBranch(_) => "init.defaultBranch",
            BaseSource::DefaultFallback => "fallback",
        }
    }
}

/// PR-level triage view: per-file stats plus issue counts. Pure — unit-tested.
fn format_review_text(base: &str, suffix: Option<&str>, files: &[ReviewedFile]) -> String {
    let s = suffix.unwrap_or("");
    let mut out = format!("Review of diff {base}...HEAD ({} files){s}\n", files.len());
    for file in files {
        let stats = match (file.added, file.deleted) {
            (Some(a), Some(d)) => format!("+{a} -{d}"),
            _ => "binary".to_string(),
        };
        out.push_str(&format!("\n  {} ({})", file.path, stats));
        if let Some(note) = &file.note {
            out.push_str(&format!("\n    {note}"));
        }
        if !file.issues.is_empty() {
            out.push_str(&format!("\n    {} issue(s)", file.issues.len()));
        }
    }
    out
}

/// Repo root for diff paths: the enclosing git toplevel when inside a
/// repo (diff paths are repo-relative), else the working directory.
/// Pure over `cwd` — unit-tested with a temp repo.
fn resolve_review_root(cwd: &std::path::Path) -> std::path::PathBuf {
    let toplevel = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(cwd)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    toplevel
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| cwd.to_path_buf())
}

/// Base branch for `xencode review`: explicit `--base`, else `origin/HEAD`,
/// else `init.defaultBranch` (or existing `master`), else fallback `'main'`.
fn resolve_review_base(
    root: &std::path::Path,
    explicit_base: Option<&str>,
) -> (String, BaseSource) {
    if let Some(b) = explicit_base {
        return (b.to_string(), BaseSource::Explicit);
    }

    let ref_exists = |r: &str| -> bool {
        std::process::Command::new("git")
            .args(["rev-parse", "--verify", "--quiet", r])
            .current_dir(root)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };

    // 1. Try git symbolic-ref refs/remotes/origin/HEAD
    let origin_head = std::process::Command::new("git")
        .args(["symbolic-ref", "refs/remotes/origin/HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(sym_ref) = origin_head {
        let short_ref = sym_ref
            .strip_prefix("refs/remotes/")
            .unwrap_or(&sym_ref)
            .to_string();
        if ref_exists(&short_ref) || ref_exists(&sym_ref) {
            return (short_ref, BaseSource::OriginHead);
        }
    }

    // 2. Try git config --get init.defaultBranch
    let default_branch = std::process::Command::new("git")
        .args(["config", "--get", "init.defaultBranch"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(branch) = default_branch {
        if ref_exists(&branch) {
            return (branch.clone(), BaseSource::InitDefaultBranch(branch));
        }
    }

    // If init.defaultBranch was unset, check if local master exists
    if ref_exists("master") {
        return (
            "master".to_string(),
            BaseSource::InitDefaultBranch("master".to_string()),
        );
    }

    // 3. Fall back to "main"
    ("main".to_string(), BaseSource::DefaultFallback)
}

/// The `AR-1` interop probe.
///
/// The scratch fixture is built in a temporary directory and removed afterwards,
/// so running the probe never leaves anything in the workspace and never touches
/// a file the operator cares about.
fn run_toolchain(action: &str, allow_dirty: bool, format: OutputFormat) -> Result<(), String> {
    use xencode_analysis_rs::toolchain as kit;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let manifest = kit::manifest_dir(&root)?;
    let as_json = matches!(format, OutputFormat::Json);

    match action {
        "lint" => {
            let report = kit::clippy_report(&manifest)?;
            if as_json {
                let out = serde_json::json!({
                    "command": report.command,
                    "count": report.count(),
                    "by_lint": report.by_lint().iter().map(|(l, n)| serde_json::json!({
                        "lint": l, "count": n,
                    })).collect::<Vec<_>>(),
                    "diagnostics": report.diagnostics.iter().map(|d| serde_json::json!({
                        "lint": d.lint, "level": d.level, "message": d.message,
                        "file": d.file, "line": d.line, "suggestion": d.suggestion,
                    })).collect::<Vec<_>>(),
                    "notes": report.notes,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
                );
            } else {
                println!("\n  {}", report.summary());
                for d in &report.diagnostics {
                    let at = match (&d.file, d.line) {
                        (Some(f), Some(l)) => format!("{f}:{l}"),
                        (Some(f), None) => f.clone(),
                        _ => "(no location)".to_string(),
                    };
                    let fixable = if d.suggestion {
                        " [machine-fixable]"
                    } else {
                        ""
                    };
                    println!("\n    {}{} — {} ({})", d.lint, fixable, d.message, at);
                }
                for note in &report.notes {
                    println!("\n  note: {note}");
                }
            }
            if report.count() == 0 {
                Ok(())
            } else {
                Err(format!("{} clippy diagnostic(s)", report.count()))
            }
        }
        "fix" => {
            let outcome = kit::cargo_fix(&manifest, &root, allow_dirty)?;
            if as_json {
                let out = serde_json::json!({
                    "command": outcome.command,
                    "changed": outcome.changed,
                    "diffstat": outcome.diffstat,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
                );
            } else if outcome.changed {
                println!("\n  fix changed files:\n\n{}", outcome.diffstat);
            } else {
                println!("\n  fix changed nothing");
            }
            Ok(())
        }
        "fmt" => {
            let (clean, report) = kit::fmt_check_output(&manifest)?;
            if as_json {
                println!("{}", serde_json::json!({ "clean": clean }));
            } else if clean {
                println!("\n  formatting is clean");
            } else {
                println!("\n  formatting differs — run `cargo fmt`");
                if !report.is_empty() {
                    println!();
                    for line in report.lines().take(40) {
                        println!("    {line}");
                    }
                }
            }
            if clean {
                Ok(())
            } else {
                Err("formatting differs".to_string())
            }
        }
        "shear" => {
            let report = kit::shear(&manifest);
            if as_json {
                println!(
                    "{}",
                    serde_json::json!({ "clean": report.clean, "lines": report.lines })
                );
            } else if report.clean {
                println!("\n  no unused dependencies reported");
            } else {
                println!("\n  shear says:");
                for line in &report.lines {
                    println!("    {line}");
                }
            }
            for line in &report.lines {
                if line.contains("not installed") {
                    return Err("cargo-shear is not installed".to_string());
                }
            }
            if report.clean {
                Ok(())
            } else {
                Err("unused dependencies reported".to_string())
            }
        }
        other => Err(format!(
            "unknown toolchain action {other:?} — want lint, fix, fmt, or shear"
        )),
    }
}

async fn run_doctor(
    env: bool,
    deps: bool,
    selfcheck: bool,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_context_rs::doctor;

    if !env && !deps && !selfcheck {
        // Bare `xencode doctor` is the bug report: every check the other flags
        // ask about, plus the state files, permissions and free space a person
        // pastes into an issue. It is the superset, not a fourth thing.
        return run_bug_report(format).await;
    }
    if selfcheck {
        return run_selfcheck(format).await;
    }
    if deps {
        return run_doctor_deps(format);
    }
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let facts = doctor::probe_env();

    // Colab route presence: a state file with a live forward pid. Absent
    // state is not an error — most machines have never run `colab up`.
    let (colab_state, colab_alive) = match xencode_colab_rs::ColabState::load() {
        Ok(Some(state)) => {
            let alive = state.forward_pid.is_some_and(xencode_colab_rs::pid_alive);
            (true, alive)
        }
        _ => (false, false),
    };

    // U-3's W11 row: the configuration drift result rides along as one row,
    // so `doctor --json` is the single shape DB-6 reads.
    let refs = xencode_analysis_rs::envdrift::extract_env_refs(&root);
    let templates = xencode_analysis_rs::envdrift::read_templates(&root);
    let drift = xencode_analysis_rs::envdrift::compare(&refs, &templates);

    // QK-4: a durable fact the code has contradicted leaves the prompt without a
    // word, which is the right thing for a model and nothing but confusion for the
    // person who wrote it. The same pass, reported. It rewrites nothing — a fact
    // stale on this checkout is often true again one branch over.
    let stale =
        xencode_context_rs::audit_durable_facts(&root.join(xencode_context_rs::XENCODE_DIR));
    let believed = xencode_context_rs::ContextState::from_markdown(&stale.text);
    let believed = believed.completed.len() + believed.decisions.len() + believed.unresolved.len();

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "machine": facts,
                "colab": {"state": colab_state, "forward_alive": colab_alive},
                "config_drift": {
                    "sources_searched": drift.sources_searched,
                    "undocumented": drift.undocumented.len(),
                    "unreferenced": drift.unreferenced,
                    "os_provided": drift.os_provided.len(),
                    "panicking": drift.panicking.len(),
                },
                "durable_facts": {
                    "believed": believed,
                    "unverifiable": stale.unverifiable,
                    "dropped": stale.dropped.iter().map(|fact| serde_json::json!({
                        "line": fact.line,
                        "reason": fact.problem.reason(),
                    })).collect::<Vec<_>>(),
                    "disagreeing": stale.disagreeing.iter().map(|odds| serde_json::json!({
                        "line": odds.line,
                        "name": odds.name,
                        "cited": odds.cited,
                        "declared_in": odds.declared_in,
                    })).collect::<Vec<_>>(),
                },
            })
        );
    } else {
        println!("\n  machine:");
        println!(
            "    cores: {}  mem_available: {}  psi: {}  cgroup_limit: {}",
            facts.nproc.map_or("?".to_string(), |n| n.to_string()),
            facts
                .mem_available_kib
                .map_or("?".to_string(), |k| format!("{k} KiB")),
            if facts.psi_readable {
                "readable"
            } else {
                "absent"
            },
            facts.cgroup_memory_limit.as_deref().unwrap_or("none"),
        );
        if facts.nvidia_gpus.is_empty() {
            println!("    gpus: none visible");
        } else {
            for gpu in &facts.nvidia_gpus {
                println!("    gpu: {gpu}");
            }
        }
        println!(
            "    journalctl: {}  dmesg_denied: {}",
            if facts.journalctl_readable {
                "readable"
            } else {
                "unreadable"
            },
            facts.dmesg_denied
        );
        println!(
            "\n  colab route: {}",
            match (colab_state, colab_alive) {
                (true, true) => "live forward".to_string(),
                (true, false) => "state file, no live forward".to_string(),
                _ => "none".to_string(),
            }
        );
        println!(
            "\n  config drift: {} undocumented, {} unreferenced, {} os-provided, {} panicking",
            drift.undocumented.len(),
            drift.unreferenced.len(),
            drift.os_provided.len(),
            drift.panicking.len(),
        );
        println!(
            "\n  durable facts: {} reaching the model, {} dropped, {} that could not be checked, {} that the code places in another file",
            believed,
            stale.dropped.len(),
            stale.unverifiable,
            stale.disagreeing.len()
        );
        // The reasons are the item: a count alone would say something was taken and
        // leave the person hunting for which line and why.
        for fact in stale.dropped.iter().take(10) {
            println!(
                "    dropped — {}: {}",
                fact.problem.reason(),
                xencode_context_rs::fact_prose(&fact.line)
            );
        }
        if stale.dropped.len() > 10 {
            println!("    … and {} more", stale.dropped.len() - 10);
        }
        // Not dropped, and not resolved either: the file the fact cites is untouched
        // and the name it was written against is still declared, in some other file.
        // Which one the fact meant is a question for whoever wrote it.
        for odds in stale.disagreeing.iter().take(10) {
            println!(
                "    disagrees — {} names {}, declared in {} rather than the cited {}",
                xencode_context_rs::fact_prose(&odds.line),
                odds.name,
                odds.declared_in.join(", "),
                odds.cited
            );
        }
        if stale.disagreeing.len() > 10 {
            println!("    … and {} more", stale.disagreeing.len() - 10);
        }
    }
    Ok(())
}

fn run_doctor_deps(format: OutputFormat) -> Result<(), String> {
    use xencode_analysis_rs::deps;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let manifest = xencode_context_rs::verify::manifest_dir(&root)?;
    let corpus = xencode_config_rs::paths::cache_dir()
        .map(|dir| xencode_analysis_rs::advisories::corpus_dir(&dir))
        .ok();
    let report = deps::doctor_deps(&manifest, corpus.as_deref())?;
    let rows = &report.rows;

    let state_word = |row: &deps::DepRow| match &row.state {
        deps::DepState::Clean => "clean",
        deps::DepState::Vulnerable(_) => "VULNERABLE",
        deps::DepState::Unknown(_) => "unknown",
    };
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "dependencies": rows.iter().map(|r| serde_json::json!({
                    "crate": r.krate,
                    "locked": r.locked,
                    "update_to": r.update_to,
                    "state": state_word(r),
                    "detail": match &r.state {
                        deps::DepState::Vulnerable(ids) => ids.join(","),
                        deps::DepState::Unknown(why) => why.clone(),
                        deps::DepState::Clean => String::new(),
                    },
                })).collect::<Vec<_>>(),
            })
        );
    } else {
        let (mut vuln, mut unknown) = (0, 0);
        for row in rows {
            match &row.state {
                deps::DepState::Vulnerable(ids) => {
                    vuln += 1;
                    println!(
                        "\n  VULNERABLE {} {}: {}",
                        row.krate,
                        row.locked.as_deref().unwrap_or("?"),
                        ids.join(", ")
                    );
                }
                deps::DepState::Unknown(why) => {
                    unknown += 1;
                    println!("\n  unknown {}: {}", row.krate, why);
                }
                deps::DepState::Clean => {}
            }
            if let Some(to) = &row.update_to {
                println!(
                    "    update available: {} -> {to}",
                    row.locked.as_deref().unwrap_or("?")
                );
            }
        }
        println!(
            "\n  {} checked, {vuln} vulnerable, {unknown} unknown",
            rows.len()
        );
        if !report.updates_checked {
            println!("  update check did not run — the update column is unknown, not clean");
        }
        let lock_text = std::fs::read_to_string(manifest.join("Cargo.lock")).unwrap_or_default();
        let dups = deps::duplicate_versions(&lock_text);
        if !dups.is_empty() {
            println!("\n  duplicate majors (ask, never block):");
            let rev = deps::reverse_deps(&lock_text);
            for dup in dups.iter().take(10) {
                println!("    {}: {}", dup.krate, dup.versions.join(", "));
                for version in &dup.versions {
                    if let Some(owners) = rev.get(&(dup.krate.clone(), version.clone())) {
                        let shown: Vec<&str> = owners.iter().take(3).map(String::as_str).collect();
                        let more = if owners.len() > 3 {
                            format!(" +{} more", owners.len() - 3)
                        } else {
                            String::new()
                        };
                        println!("      {version} via {}{more}", shown.join(", "));
                    }
                }
            }
            if dups.len() > 10 {
                println!("    +{} more", dups.len() - 10);
            }
        }
        match deps::head_lock_text(&manifest) {
            Some((_, old)) => {
                let delta = deps::lock_delta(&old, &lock_text);
                if !delta.added.is_empty() || !delta.upgraded.is_empty() {
                    println!("\n  since HEAD (new vs already present):");
                    for (name, from, to) in &delta.upgraded {
                        println!("    {name}: {from} -> {to}");
                    }
                    for (name, version) in &delta.added {
                        println!("    + {name} {version}");
                    }
                }
            }
            None => println!("\n  no HEAD lock to compare: new-vs-present unknown"),
        }
    }
    if rows
        .iter()
        .any(|r| matches!(r.state, deps::DepState::Vulnerable(_)))
    {
        Err("vulnerable dependencies found".to_string())
    } else {
        Ok(())
    }
}

/// One address the configuration says a model comes from, and what would start
/// it if nothing answers. A cloud provider is the same shape with no start
/// command: the server belongs to someone else.
struct Endpoint {
    name: &'static str,
    url: String,
    start_command: Option<&'static str>,
}

/// Every endpoint this configuration would dial, in a fixed order so two runs of
/// the same report line up.
fn endpoints(config: &xencode_config_rs::XencodeConfig) -> Vec<Endpoint> {
    let mut list = vec![
        Endpoint {
            name: "ollama",
            url: config.ollama_url.clone(),
            start_command: Some("ollama serve"),
        },
        Endpoint {
            name: "llamacpp",
            url: config.llama_cpp_url.clone(),
            start_command: Some("llama-server --model <path>"),
        },
    ];
    if !config.remote_base_url.trim().is_empty() {
        list.push(Endpoint {
            name: "remote",
            url: config.remote_base_url.clone(),
            start_command: None,
        });
    }

    // A cloud provider is only worth dialling when a key is configured —
    // reaching an endpoint nobody set up proves nothing about this machine.
    // Env-resolved: a key living in NVIDIA_NIM_API_KEY counts as configured
    // here too, since the route would use it. Presence is asked the cheap way: a
    // cloud endpoint is worth dialling when a credential exists, whether it is
    // stored, stored as a command reference, or exported.
    for (provider, host) in [
        (SecretProvider::OpenAi, "api.openai.com"),
        (SecretProvider::OpenRouter, "openrouter.ai"),
        (SecretProvider::Gemini, "generativelanguage.googleapis.com"),
        (SecretProvider::Qwen, "chat.qwen.ai"),
        (SecretProvider::Nvidia, "integrate.api.nvidia.com"),
    ] {
        if config.api_keys.has_secret(provider) {
            list.push(Endpoint {
                name: provider.slug(),
                url: format!("https://{host}"),
                start_command: None,
            });
        }
    }
    list
}

/// Providers are dialled at the address the config points them at. This machine's
/// llama.cpp server is not on the port a remembered default would guess, so a
/// check that hardcoded an address proved nothing about the route the model
/// would actually take.
fn check_endpoints(config: &xencode_config_rs::XencodeConfig) -> Vec<doc::SelfCheck> {
    use std::time::Duration;
    endpoints(config)
        .into_iter()
        .map(|endpoint| match xencode_mcp_rs::address_of(&endpoint.url) {
            Some((host, port)) => doc::check_provider(
                endpoint.name,
                &host,
                port,
                Duration::from_secs(2),
                endpoint.start_command,
            ),
            None => doc::SelfCheck {
                name: format!("provider:{}", endpoint.name),
                state: "fail".to_string(),
                detail: format!("{} is not an http or https address to dial", endpoint.url),
                fix: Some(format!(
                    "xencode config set {}_url <http-or-https-address>",
                    endpoint.name
                )),
                raw: None,
            },
        })
        .collect()
}

/// The command that would actually work for each way Ollama can refuse the
/// default model. A model id carrying the `ollama:` prefix is something this
/// configuration can hold and Ollama cannot serve — it names its models without
/// a provider prefix — so the row says which name to use rather than telling
/// the reader to pull a model that will never exist.
fn ollama_fix(error: &xencode_models_rs::OllamaError, model: &str) -> String {
    match error {
        xencode_models_rs::OllamaError::ModelNotFound(name) => match name.strip_prefix("ollama:") {
            Some(bare) => format!(
                "Ollama names its models without a provider prefix — set the default to \
                 `{bare}` or `ollama pull {bare}`"
            ),
            None => format!("ollama pull {name}"),
        },
        xencode_models_rs::OllamaError::NotRunning(_) => "ollama serve".to_string(),
        _ => format!("xencode model health {model}"),
    }
}

/// Does the server that would serve the default model actually know it by name.
/// This is the question `xencode model health` asks, asked of the model every
/// turn starts on — and only of a server on this machine, because the answer for
/// a model that lives with a service is that service's provider row.
async fn model_check(config: &xencode_config_rs::XencodeConfig) -> doc::SelfCheck {
    let model = config.default_model.trim().to_string();
    if model.is_empty() {
        return doc::check_model(
            false,
            "no default model is configured".to_string(),
            Some("xencode config set default_model <name>".to_string()),
        );
    }
    // The same prefix chain the router walks, so the report asks the server that
    // a real turn would contact rather than the one a name looks like.
    let routing = xencode_providers_rs::RoutingFacts {
        openrouter_key: config.api_keys.has_secret(SecretProvider::OpenRouter),
        remote_host: (!config.remote_base_url.is_empty())
            .then(|| xencode_providers_rs::url_host(&config.remote_base_url))
            .flatten(),
    };
    match xencode_providers_rs::provider_for(&model, routing) {
        "llamacpp" => {
            let client = LlamaCppClient::new(&config.llama_cpp_url, 4);
            if client.model_ready().await {
                doc::check_model(
                    true,
                    format!(
                        "{model}: llama-server at {} has a model loaded",
                        config.llama_cpp_url
                    ),
                    None,
                )
            } else {
                doc::check_model(
                    false,
                    format!(
                        "llama-server at {} answers but has no model loaded",
                        config.llama_cpp_url
                    ),
                    Some("xencode llamacpp load <gguf-path>".to_string()),
                )
            }
        }
        "ollama" => {
            let client = OllamaClient::new(&config.ollama_url, 4);
            match client.show_model(&model).await {
                Ok(show) => {
                    let context = show
                        .trained_context_tokens
                        .map(|n| format!("{n} tokens"))
                        .unwrap_or_else(|| "no context length reported".to_string());
                    let capabilities = if show.capabilities.is_empty() {
                        "nothing it declared".to_string()
                    } else {
                        show.capabilities.join(", ")
                    };
                    doc::check_model(
                        true,
                        format!("{model} is in Ollama's list: {context}, can do {capabilities}"),
                        None,
                    )
                }
                Err(error) => doc::check_model(
                    false,
                    format!("Ollama at {}: {error}", config.ollama_url),
                    Some(ollama_fix(&error, &model)),
                )
                .with_raw(error.raw_message()),
            }
        }
        other => doc::SelfCheck {
            name: "model".to_string(),
            state: "absent".to_string(),
            detail: format!(
                "{model} is served by {other}; provider:{other} is the row that dials it"
            ),
            fix: None,
            raw: None,
        },
    }
}

/// The rows every `doctor` surface shares: the project's own state, the
/// configured endpoints, the default model, and each declared MCP server.
async fn spine_checks(
    root: &std::path::Path,
    config: &xencode_config_rs::XencodeConfig,
) -> Vec<doc::SelfCheck> {
    let xencode_dir = root.join(xencode_context_rs::XENCODE_DIR);
    let mut checks = vec![
        doc::check_index(&xencode_dir),
        doc::check_git(root),
        doc::check_metrics(&xencode_dir),
        doc::check_cache_writable(&xencode_dir),
    ];
    checks.extend(check_endpoints(config));
    checks.push(model_check(config).await);

    // An MCP server is checked by the client the TUI uses: it is started, asked
    // to introduce itself, and killed. A pass is a completed handshake, and a
    // failure is the client's own sentence about why there was not one.
    for (name, server) in &config.mcp_servers {
        let spec = match xencode_tui_rs::mcp::spec_from_config(name, server) {
            Ok(spec) => spec,
            // A declaration that does not say how the server is reached, or says
            // both ways at once, is the named problem — there is nothing to try.
            Err(problem) => {
                checks.push(doc::check_mcp(
                    name,
                    false,
                    format!("the declaration {problem}"),
                    Some(format!(
                        "give the server `{name}` either a command or a url in the configuration"
                    )),
                ));
                continue;
            }
        };
        let fix = server.command.as_ref().map(|command| {
            format!("run `{command}` in a shell and check it answers, or remove `{name}`")
        });
        match xencode_mcp_rs::McpClient::start(&spec, std::time::Duration::from_secs(5)).await {
            Ok(client) => {
                let endpoint = client.endpoint();
                let declared = client.capabilities().declared();
                client.shutdown().await;
                checks.push(doc::check_mcp(
                    name,
                    true,
                    format!(
                        "{endpoint} started and answered the handshake; it offers {}",
                        if declared.is_empty() {
                            "nothing it declared".to_string()
                        } else {
                            declared.join(", ")
                        }
                    ),
                    None,
                ));
            }
            Err(error) => checks.push(doc::check_mcp(name, false, error.to_string(), fix)),
        }
    }
    checks
}

/// The preflight's rows, restated as report rows. The preflight names its own
/// checks "colab CLI" and "Colab API"; inside this report they are one group, so
/// the word is said once.
fn bridge_rows(report: xencode_colab_rs::PreflightReport) -> Vec<doc::SelfCheck> {
    report
        .checks
        .into_iter()
        .map(|check| {
            let short = check
                .name
                .strip_prefix("colab ")
                .or_else(|| check.name.strip_prefix("Colab "))
                .unwrap_or(check.name);
            doc::SelfCheck {
                name: format!("colab:{short}"),
                state: if check.ok { "pass" } else { "fail" }.to_string(),
                detail: check.detail,
                fix: check.fix,
                raw: None,
            }
        })
        .collect()
}

/// The Colab bridge, reported by the gate `xencode colab up` runs through. It is
/// delegated rather than re-implemented: that preflight owns the version floor
/// and the keypair, and a second copy of those judgements is a second truth. It
/// is only asked when the bridge is real on this machine, because its probes
/// reach the Google backend and a machine that never ran `colab up` has nothing
/// to report.
async fn colab_checks() -> Vec<doc::SelfCheck> {
    let installed = xencode_colab_rs::which("colab").is_some();
    let state_file = matches!(xencode_colab_rs::ColabState::load(), Ok(Some(_)));
    let keypair = xencode_colab_rs::preflight::key_paths()
        .map(|(private, _)| private.exists())
        .unwrap_or(false);
    if !(installed || state_file || keypair) {
        return vec![doc::SelfCheck {
            name: "colab".to_string(),
            state: "absent".to_string(),
            detail: "the Colab bridge is not installed here and has never been brought up"
                .to_string(),
            fix: None,
            raw: None,
        }];
    }
    // Never `generate_key`: a report reads the machine, it does not create
    // secret material on it.
    match xencode_colab_rs::preflight(false).await {
        Ok(report) => bridge_rows(report),
        Err(problem) => vec![doc::SelfCheck {
            name: "colab".to_string(),
            state: "fail".to_string(),
            detail: format!("the bridge could not be probed: {problem}"),
            fix: Some("xencode colab preflight".to_string()),
            raw: None,
        }],
    }
}

/// The configuration and the words for how it went, as one pair: the report and
/// the defaults it falls back to cannot then disagree about what was loaded.
fn load_config() -> (xencode_config_rs::XencodeConfig, doc::ConfigRead) {
    let Ok(path) = xencode_config_rs::XencodeConfig::config_path() else {
        return (
            xencode_config_rs::XencodeConfig::default(),
            doc::ConfigRead::Unparseable("no home directory to hold the config".to_string()),
        );
    };
    let outcome = xencode_config_rs::XencodeConfig::load();
    let read = match &outcome {
        Ok(_) if path.is_file() => doc::ConfigRead::Loaded,
        Ok(_) => doc::ConfigRead::Absent,
        Err(xencode_config_rs::ConfigError::NewerFile { found, known, .. }) => {
            doc::ConfigRead::TooNew {
                found: *found,
                known: *known,
            }
        }
        Err(xencode_config_rs::ConfigError::NotAConfig { found, .. }) => {
            doc::ConfigRead::Unparseable(format!(
                "it holds a JSON {found}, where an object of settings was expected"
            ))
        }
        // The loader's own sentence names the file, and the doctor row prefixes
        // it again, so this part takes the file out of what it says.
        Err(xencode_config_rs::ConfigError::Corrupt { problem, .. }) => {
            doc::ConfigRead::Unparseable(format!("it is not readable JSON — {problem}"))
        }
        Err(problem) => doc::ConfigRead::Unparseable(problem.to_string()),
    };
    (outcome.unwrap_or_default(), read)
}

/// The credential a provider is to use, read through the tiers: the value stored
/// in `config.json` (running it, when what is stored is a `command:` reference),
/// otherwise the variable named for that provider.
///
/// A reference that fails is said out loud. A keyring helper that is not on
/// `PATH` looks exactly like a provider nobody configured from the outside, and
/// the person would then go and paste the key in again.
fn api_key(config: &XencodeConfig, provider: SecretProvider) -> Option<String> {
    match config.api_keys.secret(provider) {
        Ok(value) => value,
        Err(problem) => {
            eprintln!(
                "xencode: the {} key is unusable — {problem}",
                provider.slug()
            );
            None
        }
    }
}

/// `xencode doctor --selfcheck`: the slice a person runs when xencode itself
/// looks broken.
async fn run_selfcheck(format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let (config, _) = load_config();
    let checks = spine_checks(&root, &config).await;
    render_checks("selfcheck", &checks, format)
}

/// `xencode doctor` with no flag: the whole bug report. Everything the slice
/// checks, plus the state files this install owns — does the configuration
/// parse, can its secrets be read by anyone else, is there room on the volume,
/// how much disk the response cache has taken, and whether the Colab bridge is
/// usable. The JSON is the report; the text is a rendering of the same rows.
async fn run_bug_report(format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let (config, read) = load_config();
    // Three directories, because three kinds of file live in them: the settings
    // this report is about, the records the free-space row cares about, and the
    // cache whose size is worth naming. Asking only for the settings directory
    // and guessing the rest with `join("cache")` is how a report ends up
    // measuring a directory that holds nothing.
    let settings_dir = xencode_config_rs::paths::settings_dir().ok();
    let state_dir = xencode_config_rs::paths::state_dir().ok();
    let cache_dir = xencode_config_rs::paths::cache_dir().ok();

    let mut checks = Vec::new();
    let config_path = settings_dir
        .as_ref()
        .map(|dir| dir.join("config.json"))
        .unwrap_or_default();
    checks.push(doc::check_config(&config_path, read));
    let still_old: Vec<String> = xencode_config_rs::paths::locations()
        .iter()
        .filter(|place| place.legacy)
        .map(|place| place.kind.label().to_string())
        .collect();
    checks.push(doc::check_layout(
        &still_old,
        xencode_config_rs::paths::override_root().as_deref(),
    ));
    if let Some(dir) = settings_dir.as_ref() {
        checks.push(doc::check_config_version(
            &config_path,
            xencode_config_rs::XencodeConfig::version_of(&config_path),
            xencode_config_rs::CURRENT_CONFIG_VERSION,
        ));
        checks.push(doc::check_permissions(
            "config",
            &dir.join("config.json"),
            doc::file_mode(&dir.join("config.json")),
        ));
        let (key_path, _) = xencode_colab_rs::preflight::key_paths().unwrap_or_else(|_| {
            (
                dir.join(xencode_colab_rs::KEY_FILENAME),
                dir.join("unused.pub"),
            )
        });
        checks.push(doc::check_permissions(
            "colab-key",
            &key_path,
            doc::file_mode(&key_path),
        ));
    }
    if let Some(dir) = state_dir.as_ref() {
        checks.push(doc::check_free_disk(
            "state",
            dir,
            xencode_context_rs::hwprobe::free_disk_bytes(&dir.display().to_string()),
        ));
    }
    if let Some(dir) = cache_dir.as_ref() {
        checks.push(doc::check_dir_size("cache", dir));
    }
    checks.extend(spine_checks(&root, &config).await);
    checks.extend(colab_checks().await);
    // QK-4: a durable fact the code has contradicted leaves the prompt without a
    // word. The row says which lines went, and why, in the report a person pastes.
    checks.push(doc::check_durable_facts(
        &root.join(xencode_context_rs::XENCODE_DIR),
    ));
    checks.push(doc::check_anchor(
        &root.join(xencode_context_rs::XENCODE_DIR),
    ));

    render_checks("report", &checks, format)
}

/// One list, two renderings. The rows are the report: `--format json` emits them
/// as they are, and the text listing is the same list with a mark in front.
fn render_checks(
    surface: &str,
    checks: &[doc::SelfCheck],
    format: OutputFormat,
) -> Result<(), String> {
    let failing: Vec<&str> = checks
        .iter()
        .filter(|check| check.state == "fail")
        .map(|check| check.name.as_str())
        .collect();
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "doctor": surface,
                "version": env!("CARGO_PKG_VERSION"),
                "ok": failing.is_empty(),
                "failing": failing,
                "checks": checks,
            })
        );
        if failing.is_empty() {
            return Ok(());
        } else {
            return Err(format!(
                "doctor found {} failing check{}",
                failing.len(),
                if failing.len() == 1 { "" } else { "s" }
            ));
        }
    }
    for check in checks {
        println!("  {:<6} {:<22} {}", check.mark(), check.name, check.detail);
        if let Some(fix) = &check.fix {
            println!("         {:<22} fix: {fix}", "");
        }
    }
    if !failing.is_empty() {
        println!("\n  failing: {}", failing.join(", "));
        return Err(format!(
            "doctor found {} failing check{}",
            failing.len(),
            if failing.len() == 1 { "" } else { "s" }
        ));
    }
    Ok(())
}

fn run_session(action: SessionAction) -> Result<(), String> {
    use xencode_context_rs::session as sess;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let xencode_dir = root.join(xencode_context_rs::XENCODE_DIR);
    match action {
        SessionAction::Name { run, name } => {
            let run_id = sess::resolve_session(&xencode_dir, &run)?;
            sess::name_session(&xencode_dir, &name, &run_id)?;
            println!("  {name} now names run {run_id}");
            Ok(())
        }
        SessionAction::Resolve { target } => {
            let run_id = sess::resolve_session(&xencode_dir, &target)?;
            println!("{run_id}");
            Ok(())
        }
        SessionAction::Export { target, redacted } => {
            print!(
                "{}",
                sess::export_transcript(&xencode_dir, &target, redacted)?
            );
            Ok(())
        }
    }
}

fn run_verify(
    skip: Vec<String>,
    timeout: u64,
    session: Option<String>,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_analysis_rs::toolchain as kit;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    for name in &skip {
        if !["test", "lint", "fmt"].contains(&name.as_str()) {
            return Err(format!(
                "cannot skip {name:?}: the checklist is test, lint, fmt"
            ));
        }
    }
    let target_session = session.or_else(|| {
        ConversationMemory::with_persistence(50)
            .ok()
            .and_then(|m| m.current_session().cloned())
    });
    let list = kit::run_checklist_for_session(&root, &skip, timeout, target_session.as_deref())?;
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "ok": list.ok(),
                "checks": list.checks.iter().map(|c| serde_json::json!({
                    "name": c.name, "ran": c.ran,
                    "exit": c.exit_code, "evidence": c.evidence_ref,
                })).collect::<Vec<_>>(),
                "failed": list.failed(),
                "skipped": list.skipped(),
            })
        );
    } else {
        for check in &list.checks {
            let state = if !check.ran {
                "SKIPPED".to_string()
            } else if check.passed() {
                "PASS".to_string()
            } else {
                format!("FAIL (exit {})", check.exit_code.unwrap_or(-1))
            };
            println!("  {:<7} {:<5} {}", check.name, state, check.evidence_ref);
        }
        if !list.skipped().is_empty() {
            println!("\n  skipped, not passed: {}", list.skipped().join(", "));
        }
    }
    if list.ok() {
        Ok(())
    } else {
        Err(format!("checklist failed: {}", list.failed().join(", ")))
    }
}

fn run_envcheck(format: OutputFormat) -> Result<(), String> {
    use xencode_analysis_rs::envdrift;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let refs = envdrift::extract_env_refs(&root);
    let templates = envdrift::read_templates(&root);
    let report = envdrift::compare(&refs, &templates);

    if matches!(format, OutputFormat::Json) {
        let out = serde_json::json!({
            "sources_searched": report.sources_searched,
            "undocumented": report.undocumented.iter().map(|r| serde_json::json!({
                "key": r.key, "file": r.file, "line": r.line,
                "panics_when_absent": r.panics_when_absent,
            })).collect::<Vec<_>>(),
            "unreferenced": report.unreferenced,
            "os_provided": report.os_provided.iter().map(|r| serde_json::json!({
                "key": r.key, "file": r.file, "line": r.line,
            })).collect::<Vec<_>>(),
            "panicking": report.panicking.iter().map(|r| serde_json::json!({
                "key": r.key, "file": r.file, "line": r.line,
            })).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
        );
    } else {
        if report.sources_searched.is_empty() {
            println!(
                "\n  no template found (.env.example, .env.template) — nothing documents the configuration"
            );
        } else {
            println!("\n  templates read: {}", report.sources_searched.join(", "));
        }
        if report.undocumented.is_empty() {
            println!("  no read-but-undocumented keys");
        } else {
            println!("\n  read-but-undocumented:");
            for (key, refs) in envdrift::by_key(&report.undocumented) {
                let at = refs
                    .iter()
                    .map(|r| format!("{}:{}", r.file, r.line))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("    {key}  (read at {at})");
            }
        }
        if !report.unreferenced.is_empty() {
            println!("\n  documented-but-unreferenced (not unnecessary):");
            for key in &report.unreferenced {
                println!("    {key}");
            }
        }
        if !report.panicking.is_empty() {
            println!("\n  panics-when-absent outside tests:");
            for r in &report.panicking {
                println!("    {}  ({}:{})", r.key, r.file, r.line);
            }
        }
        println!(
            "\n  {} os-provided key(s) listed separately, not reported",
            report.os_provided.len()
        );
    }
    Ok(())
}

struct AgentsArgs<'a> {
    contract: bool,
    health: bool,
    target_agent: Option<&'a str>,
    build_package: Option<&'a str>,
    package_path: Option<&'a std::path::Path>,
    resume: bool,
    test_cmds: &'a [String],
    route_task: Option<&'a str>,
    require_caps: &'a [String],
    max_cost: Option<f64>,
    redispatch_task: Option<&'a str>,
    replacement_agent: Option<&'a str>,
    stop_reason: Option<&'a str>,
    format: OutputFormat,
}

fn run_agents(args: AgentsArgs<'_>) -> Result<(), String> {
    if let Some(task_id) = args.redispatch_task {
        let repo_dir = std::env::current_dir().map_err(|e| e.to_string())?;
        let initial_worker = args.target_agent.unwrap_or("failed-worker");
        let replacement_worker = args.replacement_agent.unwrap_or("replacement-worker");

        // Parse process stop reason
        let stop_reason = if let Some(reason_str) = args.stop_reason {
            if let Some(sig) = reason_str.strip_prefix("signal:") {
                let code = sig
                    .parse::<i32>()
                    .map_err(|_| format!("invalid signal in '{reason_str}'"))?;
                xencode_agents_rs::WorkerStopReason::Signal { signal: code }
            } else if let Some(exit) = reason_str.strip_prefix("exit:") {
                let code = exit
                    .parse::<i32>()
                    .map_err(|_| format!("invalid exit code in '{reason_str}'"))?;
                xencode_agents_rs::WorkerStopReason::ExitCode { code }
            } else if let Some(secs) = reason_str.strip_prefix("timeout:") {
                let dur = secs
                    .parse::<u64>()
                    .map_err(|_| format!("invalid duration in '{reason_str}'"))?;
                xencode_agents_rs::WorkerStopReason::TimedOut { duration_secs: dur }
            } else {
                xencode_agents_rs::WorkerStopReason::Signal { signal: 9 }
            }
        } else {
            xencode_agents_rs::WorkerStopReason::Signal { signal: 9 }
        };

        let default_cmd = "cargo test".to_string();
        let test_cmd_refs: Vec<&str> = if args.test_cmds.is_empty() {
            vec![default_cmd.as_str()]
        } else {
            args.test_cmds.iter().map(String::as_str).collect()
        };

        let ledger_path = repo_dir.join(".xencode").join("task_ledger.jsonl");

        let outcome = xencode_agents_rs::redispatch_failed_worker(
            &repo_dir,
            task_id,
            initial_worker,
            replacement_worker,
            stop_reason,
            &test_cmd_refs,
            &ledger_path,
            |context, _dir| {
                if matches!(args.format, OutputFormat::Text) {
                    println!(
                        "Re-dispatching task '{task_id}' onto worker '{replacement_worker}'..."
                    );
                    println!("{context}");
                }
                Ok(0)
            },
        )?;

        if matches!(args.format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&outcome).map_err(|e| e.to_string())?
            );
        } else {
            println!("Task Re-dispatch Completed for '{}':", outcome.task_id);
            println!(
                "  Initial worker: {} ({})",
                outcome.initial_worker,
                outcome.initial_stop_reason.summary()
            );
            println!("  Replacement worker: {}", outcome.replacement_worker);
            println!("  Verification tests passed: {}", outcome.all_tests_passed);
            println!(
                "  Preserved diff files ({}):",
                outcome.preserved_diff.changed_files.len()
            );
            for f in &outcome.preserved_diff.changed_files {
                println!("    * {f}");
            }
            println!(
                "  Ledger entries logged ({}) to {}:",
                outcome.ledger_entries.len(),
                ledger_path.display()
            );
            for entry in &outcome.ledger_entries {
                println!(
                    "    - Attempt {}: worker '{}' -> passed: {}, stop reason: {}",
                    entry.attempt,
                    entry.worker,
                    entry.passed,
                    entry.stop_reason.summary()
                );
            }
        }
        return Ok(());
    }

    if let Some(task_id) = args.route_task {
        // Everything the router is allowed to use was read off this machine a few
        // seconds ago, and the line of evidence behind each capability travels with
        // it, so a choice can be checked rather than believed (`OR-11`).
        let claims = xencode_agents_rs::probe_contract();
        let confirmed_caps_by_agent = xencode_agents_rs::confirmed_capabilities(&claims);
        let evidence_by_agent = xencode_agents_rs::evidence_by_agent(&claims);
        let probed: std::collections::BTreeSet<String> =
            claims.iter().map(|claim| claim.agent.to_string()).collect();

        // The posture is read from the same config the TUI shows, and it refuses a
        // roster agent by that name before the capability, load and cost checks are
        // consulted, so the reason printed here is the reason the router actually
        // applied (`OR-13`).
        let profile = XencodeConfig::load().unwrap_or_default().profile();

        let installed = xencode_agents_rs::inventory();
        let names: Vec<String> = if installed.is_empty() {
            // Nothing on `PATH` matched the roster. Its names are still listed, so a
            // person asking who could take a task sees what was considered — each one
            // marked as never probed, which is not the same answer as probing it.
            xencode_agents_rs::ROSTER
                .iter()
                .map(|spec| spec.name.to_string())
                .collect()
        } else {
            installed
                .iter()
                .map(|agent| agent.name.to_string())
                .collect()
        };

        let candidates: Vec<xencode_core_rs::WorkerCandidate> = names
            .iter()
            .map(|name| xencode_core_rs::WorkerCandidate {
                id: name.clone(),
                probed_capabilities: confirmed_caps_by_agent
                    .get(name)
                    .cloned()
                    .unwrap_or_default(),
                probed: probed.contains(name),
                // Answered from the roster row, not from the shape of the name:
                // every candidate listed here came off that roster, so under the
                // shipped posture all of them are refused before a measurement is
                // read, and the refusal says which setting would change that.
                external: xencode_agents_rs::is_external_worker(name),
                capability_evidence: evidence_by_agent.get(name).cloned().unwrap_or_default(),
                load: xencode_core_rs::Fact::unknown(
                    "xencode cannot see what another process has given this worker to do; the \
                     only work it counts is what a team of its own has leased, and this question \
                     leased nothing",
                ),
                capacity: xencode_core_rs::Fact::unknown(
                    "nobody measured how many tasks this worker runs at once; the roster records \
                     what its help output says, and help does not say this",
                ),
                cost: xencode_core_rs::Fact::unknown(
                    "no price is known for this worker: xencode's price documents name models, \
                     and these agents bill their own accounts",
                ),
            })
            .collect();

        let mut required_caps = std::collections::BTreeSet::new();
        for cap in args.require_caps {
            required_caps.insert(cap.to_string());
        }

        let task = xencode_core_rs::TaskRequirement {
            task_id: task_id.to_string(),
            required_capabilities: required_caps,
            cost_ceiling: args.max_cost,
            profile,
        };

        let decision = xencode_core_rs::CapabilityRouter::route(&task, &candidates)?;

        // Whether anything was refused on the posture at all, so the closing note
        // only claims the rule is in force when it was the reason.
        let refused_by_profile = candidates
            .iter()
            .any(|c| profile.check_worker(&c.id, c.external).is_err());

        if matches!(args.format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&decision).map_err(|e| e.to_string())?
            );
        } else {
            println!("Routing decision for task '{}':", decision.task_id);
            println!();
            println!("  Posture: {}", profile.name());
            for rule in profile.rules() {
                println!("    {rule}");
            }
            println!();
            match &decision.selected_worker {
                Some(worker) => println!("  chosen: {worker}"),
                None => println!("  chosen: nothing — no candidate could take this task"),
            }
            println!("  why:    {}", decision.explanation);

            println!("\n  What the router asked, in the order it asked it:");
            for step in &decision.steps {
                let standing = if !step.ran {
                    "not asked"
                } else if step.decided {
                    "decided it"
                } else {
                    "asked, settled nothing"
                };
                println!("    {:14} {:20} {}", step.check, standing, step.words);
            }

            // The chosen worker gets its facts in full, because that is the list a
            // reader is being asked to check. The others are listed with what the
            // probe found on them and why any was refused. A check that did not run
            // is said once, with how many candidates it did not run for.
            let considered = decision.candidate_evaluations.len();
            let mut not_asked: std::collections::BTreeMap<&String, usize> =
                std::collections::BTreeMap::new();
            for evaluation in &decision.candidate_evaluations {
                for line in &evaluation.not_checked {
                    *not_asked.entry(line).or_insert(0) += 1;
                }
            }

            if let Some(chosen) = decision
                .candidate_evaluations
                .iter()
                .find(|e| decision.selected_worker.as_deref() == Some(e.worker_id.as_str()))
            {
                println!("\n  The facts behind the choice — {}:", chosen.worker_id);
                for fact in &chosen.facts {
                    println!("      {fact}");
                }
            }

            println!("\n  Every worker considered ({considered}):");
            for evaluation in &decision.candidate_evaluations {
                println!(
                    "    {} {:<13} probed here as: {}",
                    if evaluation.eligible { "▸" } else { "✗" },
                    evaluation.worker_id,
                    if evaluation.probed_capabilities.is_empty() {
                        "nothing this machine could see".to_string()
                    } else {
                        evaluation.probed_capabilities.join(", ")
                    }
                );
                if let Some(reason) = &evaluation.rejection {
                    println!("      why not: {}", reason.words());
                }
            }

            if !not_asked.is_empty() {
                println!(
                    "\n  Checks that did not run, and how many candidates they would have covered:"
                );
                for (line, count) in &not_asked {
                    println!("      for {count} of {considered}: {line}");
                }
            }
            println!(
                "\n  The capability lines above were not inferred from a name or a document: they \
                came from the help output of the binaries on this machine, read during this \
                command. Where a line says `not measured`, that is the whole of what xencode \
                knows, and no worker was accepted or refused because of it."
            );
            if refused_by_profile {
                println!(
                    "  The one thing decided before the capability, load and cost checks were \
                    consulted is the posture: it refuses a roster agent by its roster row, which \
                    is a policy and not a measurement. The `Posture:` block above names the \
                    setting that would open the rule (`xencode config set \
                    allow_external_workers true`)."
                );
            }
        }
        return Ok(());
    }

    if let Some(task_id) = args.build_package {
        let repo_dir = std::env::current_dir().map_err(|e| e.to_string())?;
        let default_cmd = "cargo test".to_string();
        let cmds: Vec<&str> = if args.test_cmds.is_empty() {
            vec![default_cmd.as_str()]
        } else {
            args.test_cmds.iter().map(String::as_str).collect()
        };
        let prev = args.target_agent.unwrap_or("worker");
        let pkg = xencode_agents_rs::WorkerPackage::build(
            &repo_dir,
            task_id,
            format!("Continuation package for task {task_id}"),
            prev,
            xencode_agents_rs::WorkerStopReason::ExitCode { code: 1 },
            &cmds,
            vec![],
        )?;
        let out_dir = repo_dir.join(".xencode").join("packages");
        std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        let out_file = out_dir.join(format!("{task_id}.json"));
        pkg.save_to_file(&out_file)?;

        if matches!(args.format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&pkg).map_err(|e| e.to_string())?
            );
        } else {
            println!(
                "Worker continuation package written to {}",
                out_file.display()
            );
            println!("\n{}", pkg.resumption_context());
        }
        return Ok(());
    }

    if let Some(path) = args.package_path {
        let pkg = xencode_agents_rs::WorkerPackage::load_from_file(path)?;
        if args.resume {
            let repo_dir = std::env::current_dir().map_err(|e| e.to_string())?;
            let next_agent = args.target_agent.unwrap_or("resuming-worker");
            let default_cmd = "cargo test".to_string();
            let cmds: Vec<&str> = if args.test_cmds.is_empty() {
                vec![default_cmd.as_str()]
            } else {
                args.test_cmds.iter().map(String::as_str).collect()
            };
            let outcome = xencode_agents_rs::resume_task_from_package(
                &pkg,
                &repo_dir,
                next_agent,
                &cmds,
                |context, _dir| {
                    if matches!(args.format, OutputFormat::Text) {
                        println!(
                            "Resuming task '{}' with worker '{}'...",
                            pkg.task_id, next_agent
                        );
                        println!("{context}");
                    }
                    Ok(0)
                },
            )?;
            if matches!(args.format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&outcome).map_err(|e| e.to_string())?
                );
            } else {
                println!("\nResumption outcome for task '{}':", outcome.task_id);
                println!("- Resuming worker: {}", outcome.resuming_agent);
                println!("- Verification tests passed: {}", outcome.all_tests_passed);
                for t in &outcome.post_test_runs {
                    let status = if t.passed { "PASSED" } else { "FAILED" };
                    println!("  * `{}` -> exit {} [{status}]", t.command, t.exit_code);
                }
            }
            return Ok(());
        } else {
            if matches!(args.format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&pkg).map_err(|e| e.to_string())?
                );
            } else {
                println!("{}", pkg.resumption_context());
            }
            return Ok(());
        }
    }

    if args.contract {
        return run_contract(args.format);
    }
    if args.health {
        let timeout = std::time::Duration::from_secs(3);
        let selected = if let Some(target) = args.target_agent {
            let spec = xencode_agents_rs::ROSTER
                .iter()
                .find(|s| s.name == target)
                .ok_or_else(|| format!("no roster agent named '{target}'"))?;
            vec![xencode_agents_rs::check_worker_health(spec, timeout)]
        } else {
            xencode_agents_rs::check_all_worker_health(timeout)
        };
        if matches!(args.format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&selected).map_err(|e| e.to_string())?
            );
        } else {
            println!(
                "{}",
                xencode_agents_rs::format_worker_health_table(&selected)
            );
        }
        return Ok(());
    }
    let found = xencode_agents_rs::inventory();
    if matches!(args.format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "agents": found.iter().map(|a| serde_json::json!({
                    "name": a.name,
                    "binary": a.binary.display().to_string(),
                    "version": a.version,
                    "source": a.source.label(),
                })).collect::<Vec<_>>(),
            })
        );
    } else if found.is_empty() {
        println!("\n  no roster agents found on PATH");
    } else {
        for agent in &found {
            println!(
                "\n  {:<8} {}",
                agent.name,
                agent.version.as_deref().unwrap_or("(no --version answer)")
            );
            println!(
                "           {} [{}]",
                agent.binary.display(),
                agent.source.label()
            );
        }
        println!("\n  discovery only: nothing was installed, upgraded, or written");
    }
    Ok(())
}

fn run_contract(format: OutputFormat) -> Result<(), String> {
    use xencode_agents_rs::Verdict;

    let results = xencode_agents_rs::probe_contract();
    let bad = results
        .iter()
        .filter(|r| matches!(r.verdict, Verdict::Contradicted(_)))
        .count();
    let untested = results
        .iter()
        .filter(|r| matches!(r.verdict, Verdict::Untestable(_)))
        .count();
    // A confirmed claim is not one kind: "the flag is there" and "the flag is
    // not there, and here is the word we looked for" are opposite facts that
    // happen to share a verdict. Counted apart, so a reader is not told that
    // every one of these numbers is a capability.
    let absences = results
        .iter()
        .filter(|r| !r.expected && matches!(r.verdict, Verdict::Confirmed))
        .count();
    let confirmed = results.len() - bad - untested;
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "summary": {
                    "claims": results.len(),
                    "confirmed": confirmed,
                    "confirmed_absences": absences,
                    "contradicted": bad,
                    "untested": untested,
                },
                "claims": results.iter().map(|r| serde_json::json!({
                    "agent": r.agent,
                    "claim": r.claim,
                    "expected": r.expected,
                    "verdict": format!("{:?}", r.verdict),
                    "found": r.found,
                    "missing": r.missing,
                    "sources": r.sources,
                    "evidence": r.evidence_line(),
                })).collect::<Vec<_>>(),
            })
        );
    } else {
        for result in &results {
            match &result.verdict {
                Verdict::Confirmed => {}
                Verdict::Contradicted(why) => {
                    println!("\n  CONTRADICTED {} {}: {why}", result.agent, result.claim)
                }
                Verdict::Untestable(why) => {
                    println!("\n  untestable {} {}: {why}", result.agent, result.claim)
                }
            }
        }
        println!(
            "\n  {confirmed} claims confirmed ({} of them absences the probe searched for), \
             {bad} contradicted, {untested} untested",
            absences
        );
    }
    if bad == 0 {
        Ok(())
    } else {
        Err("roster claims contradicted by live --help".to_string())
    }
}

fn run_hotspots(limit: usize, format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let rows = xencode_context_rs::hotspots(&root, limit);
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "hotspots": rows.iter().map(|r| serde_json::json!({
                    "file": r.file,
                    "kind": format!("{:?}", r.kind),
                    "message": r.message,
                })).collect::<Vec<_>>(),
            })
        );
    } else if rows.is_empty() {
        println!("\n  no history to rank — untracked files or outside a repository");
    } else {
        for row in &rows {
            println!("\n  {}", row.message);
        }
    }
    Ok(())
}

/// `QD-1`: what a change to one file affects, in three layers that carry three
/// different strengths of evidence and are never blended. The crate list is exact
/// (it comes from `cargo metadata`); the file list is a hop-capped prediction over
/// the source graph; the coupling list is history. The command runs with no
/// `.xencode` index, building the graph from the files on disk.
/// `xencode impact <file> --semantic`: the files that refer to symbols this file
/// defines, from rust-analyzer's SCIP index (`LSP-2`). A missing or stale index
/// is rebuilt first, and the reason is printed, because the rebuild takes
/// minutes and a person waiting on it should know why.
fn run_impact_semantic(
    file: &str,
    symbol: Option<&str>,
    limit: usize,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_context_rs::scip_index::{self, ScipError};
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let workspace = xencode_context_rs::verify::manifest_dir(&root)?;
    let mut rebuilt = None;
    if let Err(why) = scip_index::freshness(&workspace) {
        if !matches!(why, ScipError::Missing(_) | ScipError::Stale(_)) {
            return Err(why.to_string());
        }
        eprintln!(
            "  {why}\n  building it with rust-analyzer scip in {} — this runs the project's build scripts and takes minutes on a large workspace…",
            workspace.display()
        );
        let meta = scip_index::generate(&workspace, scip_index::SCIP_TIMEOUT)
            .map_err(|e| e.to_string())?;
        eprintln!(
            "  indexed {} files in {:.0} s with {}",
            meta.files.len(),
            meta.seconds,
            meta.rust_analyzer
        );
        rebuilt = Some(meta);
    }
    let report = scip_index::semantic_impact_for(
        &workspace,
        file,
        symbol,
        xencode_context_rs::IMPACT_MAX_HOPS,
    )
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    let meta = match rebuilt {
        Some(meta) => meta,
        None => scip_index::read_meta(&workspace).ok_or("the semantic index record is missing")?,
    };

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "target": report.target,
                "symbol": report.symbol,
                "tier": "semantic",
                "index": {
                    "rust_analyzer": meta.rust_analyzer,
                    "head": meta.head,
                    "seconds": meta.seconds,
                    "files": meta.files.len(),
                },
                "declared": report.declared,
                "basis": report.basis(),
                "affected": report.files.iter().map(|f| serde_json::json!({
                    "file": f.file, "hops": f.hops, "via": f.via,
                })).collect::<Vec<_>>(),
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    match &report.symbol {
        Some(symbol) => println!("\n  who uses `{symbol}` from {}", report.target),
        None => println!("\n  who uses {}", report.target),
    }
    println!(
        "  semantic index: {} files, built by {} in {:.0} s",
        meta.files.len(),
        meta.rust_analyzer,
        meta.seconds
    );
    if report.files.is_empty() {
        println!("\n    nothing outside this file refers to it");
    } else {
        for f in report.files.iter().take(limit) {
            let names = if f.via.len() > 6 {
                format!("{} and {} more", f.via[..6].join(", "), f.via.len() - 6)
            } else {
                f.via.join(", ")
            };
            println!("    {} — {} hop(s), refers to {}", f.file, f.hops, names);
        }
        if report.files.len() > limit {
            println!("    … {} more", report.files.len() - limit);
        }
    }
    println!("\n  {}", report.basis());
    Ok(())
}

fn run_impact(file: &str, limit: usize, format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let workspace = xencode_context_rs::verify::manifest_dir(&root)?;
    let impact = xencode_context_rs::change_impact(&workspace, file).map_err(|e| e.to_string())?;

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "target": impact.target,
                "crate": impact.crate_name,
                "crate_basis": "exact, from `cargo metadata --no-deps`",
                "reverse_crates": impact.reverse_crates.iter().map(|(n, hops)| serde_json::json!({
                    "crate": n, "hops": hops,
                })).collect::<Vec<_>>(),
                "direct_crates": impact.direct_crates.iter().map(|(d, kind)| serde_json::json!({
                    "crate": d, "kind": kind,
                })).collect::<Vec<_>>(),
                "files": {
                    "basis": impact.files.basis(),
                    "affected": impact.files.files.iter().map(|f| serde_json::json!({
                        "file": f.file, "hops": f.hops, "via": f.via,
                    })).collect::<Vec<_>>(),
                },
                "history": {
                    "known": impact.history_known,
                    "own_commits": impact.own_commits,
                    "cochange": impact.cochange.iter().map(|(f, n)| serde_json::json!({
                        "file": f, "commits_together": n,
                    })).collect::<Vec<_>>(),
                },
                "cli_commands": impact.cli_impact.as_ref().map(|cli| serde_json::json!({
                    "source_file": cli.source_file,
                    "variants_count": cli.variants_count(),
                    "commands": cli.commands.iter().map(|c| serde_json::json!({
                        "variant": c.variant,
                        "subcommand": c.subcommand,
                        "line": c.line,
                        "doc_refs": c.doc_refs.iter().map(|d| serde_json::json!({
                            "file": d.file,
                            "line": d.line,
                            "text": d.text,
                        })).collect::<Vec<_>>(),
                    })).collect::<Vec<_>>(),
                    "stale_docs": cli.stale_docs.iter().map(|s| serde_json::json!({
                        "file": s.file,
                        "line": s.line,
                        "subcommand": s.subcommand,
                        "text": s.text,
                    })).collect::<Vec<_>>(),
                })),
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("\n  impact of {}", impact.target);
    match &impact.crate_name {
        Some(name) => println!("  crate: {name}"),
        None => println!("  crate: (this file is not inside a workspace crate)"),
    }

    if let Some(cli) = &impact.cli_impact {
        println!(
            "\n  CLI commands defined ({} clap variant(s) in {}):",
            cli.commands.len(),
            cli.source_file
        );
        for cmd in cli.commands.iter().take(limit) {
            println!(
                "    {} — Commands::{} (line {})",
                cmd.subcommand, cmd.variant, cmd.line
            );
            if cmd.doc_refs.is_empty() {
                println!("      manuals: (none)");
            } else {
                println!("      manuals:");
                for doc in &cmd.doc_refs {
                    println!("        {}:{} — {}", doc.file, doc.line, doc.text);
                }
            }
        }
        if cli.commands.len() > limit {
            println!("    … {} more", cli.commands.len() - limit);
        }

        if !cli.stale_docs.is_empty() {
            println!("\n  stale manual references (documented subcommands not in Commands enum):");
            for stale in &cli.stale_docs {
                println!(
                    "    {}:{} — `{}` not in Commands ({})",
                    stale.file, stale.line, stale.subcommand, stale.text
                );
            }
        } else {
            println!("\n  stale manual references: none — all documented subcommands match active variants");
        }
    }

    println!("\n  crates that depend on it — exact, from cargo metadata:");
    if impact.reverse_crates.is_empty() {
        println!("    none — nothing first-party depends on this crate");
    } else {
        for (name, hops) in impact.reverse_crates.iter().take(limit) {
            let direct = impact
                .direct_crates
                .iter()
                .find(|(d, _)| d == name)
                .map(|(_, kind)| format!(" ({kind} edge)"))
                .unwrap_or_default();
            println!("    {name} — {hops} hop(s){direct}");
        }
    }

    println!("\n  files that link it — predicted, hop-capped:");
    if impact.files.files.is_empty() {
        println!("    none in the graph — this file's own edges point outward, not in");
    } else {
        for f in impact.files.files.iter().take(limit) {
            println!(
                "    {} — {} hop(s) via {}",
                f.file,
                f.hops,
                f.via.join(", ")
            );
        }
        if impact.files.files.len() > limit {
            println!("    … {} more", impact.files.files.len() - limit);
        }
    }
    println!("\n  {}", impact.files.basis());

    println!("\n  coupled by history — most co-changed with:");
    if !impact.history_known {
        println!("    unknown — no git history here to read");
    } else if impact.cochange.is_empty() {
        println!("    none — this file is not committed alongside others");
    } else {
        for (f, n) in impact.cochange.iter().take(limit) {
            println!("    {f} — {n} commit(s) together");
        }
    }
    if impact.history_known {
        println!(
            "\n  own churn: {} commit(s) touching {}",
            impact.own_commits, impact.target
        );
    }
    Ok(())
}

fn run_removal(file: &str, limit: usize, format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let workspace = xencode_context_rs::verify::manifest_dir(&root)?;
    let removal =
        xencode_context_rs::removal_impact(&workspace, file).map_err(|e| e.to_string())?;

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "target": removal.target,
                "basis": "the dependency graph with one file removed, off resolved `use`/`mod`/`impl` names — not a compile",
                "breaks": removal.breaks,
                "orphans": removal.orphans,
                "indexed_files": removal.indexed_files,
                "edges": removal.edges,
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("\n  if you delete {}", removal.target);

    println!("\n  files that lose a link to it:");
    if removal.breaks.is_empty() {
        println!("    none — no other file resolves a name into this one");
    } else {
        for f in removal.breaks.iter().take(limit) {
            println!("    {f}");
        }
        if removal.breaks.len() > limit {
            println!("    … {} more", removal.breaks.len() - limit);
        }
    }

    println!("\n  files that become dead code (nothing else reaches them):");
    if removal.orphans.is_empty() {
        println!("    none — every module this one pulled in is pulled in elsewhere");
    } else {
        for f in removal.orphans.iter().take(limit) {
            println!("    {f}");
        }
        if removal.orphans.len() > limit {
            println!("    … {} more", removal.orphans.len() - limit);
        }
    }

    println!(
        "\n  From the graph of {} Rust files and {} resolved edges. A link means a \
         file wrote a `use` path, a `mod` declaration or an `impl Trait for Type` that \
         resolves to another — name resolution through module paths, not a type-checked \
         compile. Deletion is stronger evidence than an edit here: a change a consumer \
         might absorb, a missing file it cannot.",
        removal.indexed_files, removal.edges
    );
    Ok(())
}

fn run_generate(artifact: GenerateArtifact, shell: GenerateShell) -> Result<(), String> {
    use clap::CommandFactory;
    let mut cmd = Cli::command();
    match artifact {
        GenerateArtifact::Completions => {
            let shell = match shell {
                GenerateShell::Bash => clap_complete::Shell::Bash,
                GenerateShell::Fish => clap_complete::Shell::Fish,
                GenerateShell::Zsh => clap_complete::Shell::Zsh,
                GenerateShell::Powershell => clap_complete::Shell::PowerShell,
                GenerateShell::Elvish => clap_complete::Shell::Elvish,
            };
            clap_complete::generate(shell, &mut cmd, "xencode", &mut std::io::stdout());
            Ok(())
        }
        GenerateArtifact::Man => {
            let man = clap_mangen::Man::new(cmd);
            man.render(&mut std::io::stdout())
                .map_err(|e| format!("could not render the man page: {e}"))?;
            Ok(())
        }
    }
}

fn run_mutants(
    diff: Option<String>,
    timeout: u64,
    check_repair: Option<std::path::PathBuf>,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_analysis_rs::mutation as mtest;

    if let Some(patch) = check_repair {
        return judge_repair(patch, format);
    }
    if !mtest::available() {
        return Err(
            "cargo mutants is not installed. Install it with `cargo install cargo-mutants`"
                .to_string(),
        );
    }
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let manifest = xencode_context_rs::verify::manifest_dir(&root)?;

    // The diff is written first, with pinned prefixes, because cargo-mutants
    // takes a file rather than a ref and silently filters nothing when the paths
    // carry git's mnemonic `i/`/`w/` prefixes.
    let patch = mtest::write_diff(&manifest, diff.as_deref())?;
    let argv = mtest::mutants_argv(Some(&patch), Some(timeout));
    let rendered = format!("cargo {}", argv.join(" "));
    let mut command = std::process::Command::new("cargo");
    command.current_dir(&manifest).args(&argv);
    eprintln!("  running: {rendered}");
    eprintln!("  this runs the whole suite once per surviving mutant, so it is not quick");
    let status = command
        .stdout(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("could not start cargo mutants: {e}"))?;
    // A non-zero exit means mutants were missed, which is a result rather than
    // a failure to run, so the report is read either way.
    let _ = status;

    let run = mtest::parse_report(&manifest.join(mtest::REPORT_DIR))?;
    let missed = run.missed();

    if matches!(format, OutputFormat::Json) {
        let report = serde_json::json!({
            "command": rendered,
            "total": run.mutants.len(),
            "counts": run.counts(),
            "usable": run.is_usable(),
            "symbols": run.per_symbol().iter().map(|s| serde_json::json!({
                "file": s.file,
                "symbol": s.symbol,
                "caught": s.caught,
                "missed": s.missed,
                "unviable": s.unviable,
                "timeout": s.timeout,
                "viable": s.score().map(|(_, v)| v),
            })).collect::<Vec<_>>(),
            "missed": missed.iter().map(|m| serde_json::json!({
                "file": m.file, "name": m.name, "key": m.key(),
            })).collect::<Vec<_>>(),
            "notes": run.notes,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else {
        println!("\n  {}\n", run.counts());
        for note in &run.notes {
            println!("  note:  {note}");
        }
        // QD-3: the rollup names the function the tests are weakest on, so a
        // score points at code to read rather than a vague "feature is untested".
        println!("  per function (worst first):");
        for line in run.symbol_report() {
            println!("    {line}");
        }
        if missed.is_empty() {
            println!("\n  no missed mutants — every mutation was caught by a test");
        } else {
            println!("\n  MISSED — code no test can tell from correct:");
            for m in &missed {
                println!("\n    {}", m.file);
                println!("      {}", m.name);
            }
            println!(
                "\n  Repair these by making the tests stronger. A repair is only real if \
                 the same mutants are re-run and come back caught."
            );
        }
    }
    if run.is_usable() && missed.is_empty() {
        Ok(())
    } else {
        Err("mutants survived".to_string())
    }
}

/// Judge a proposed repair against the four gate conditions.
///
/// Reads the patch, the missed-mutant list and both assertion counts from a JSON
/// file, so the judgement itself is a pure function that can be tested without a
/// mutation run. Re-running the same mutant set is the caller's job and is
/// reported as missing when its result is absent — never assumed.
fn judge_repair(path: std::path::PathBuf, format: OutputFormat) -> Result<(), String> {
    use xencode_analysis_rs::mutation as mtest;

    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;

    let counts_of = |key: &str| -> std::collections::BTreeMap<String, usize> {
        value
            .get(key)
            .and_then(|v| v.as_object())
            .map(|o| {
                o.iter()
                    .filter_map(|(f, n)| n.as_u64().map(|n| (f.clone(), n as usize)))
                    .collect()
            })
            .unwrap_or_default()
    };

    let targeted: Vec<mtest::Mutant> = value
        .get("targeted")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    Some(mtest::Mutant {
                        file: m.get("file")?.as_str()?.to_string(),
                        name: m.get("name")?.as_str()?.to_string(),
                        // A repair patch names mutants by file and name, not by
                        // the enclosing function, so the rollup bucket is unknown.
                        symbol: mtest::OUTSIDE_FUNCTION.to_string(),
                        verdict: mtest::Verdict::Missed,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // The re-run, when one is recorded. Its absence is the rejection.
    let rerun = value.get("rerun").and_then(|r| {
        let mutants: Vec<mtest::Mutant> = r
            .get("mutants")?
            .as_array()?
            .iter()
            .filter_map(|m| {
                Some(mtest::Mutant {
                    file: m.get("file")?.as_str()?.to_string(),
                    name: m.get("name")?.as_str()?.to_string(),
                    symbol: mtest::OUTSIDE_FUNCTION.to_string(),
                    verdict: match m.get("verdict").and_then(|v| v.as_str()).unwrap_or("") {
                        "caught" => mtest::Verdict::Caught,
                        "unviable" => mtest::Verdict::Unviable,
                        "timeout" => mtest::Verdict::Timeout,
                        _ => mtest::Verdict::Missed,
                    },
                })
            })
            .collect();
        Some(mtest::Run {
            mutants,
            ..mtest::Run::default()
        })
    });

    let patch = value
        .get("patch")
        .and_then(|p| p.as_str())
        .unwrap_or_default();
    let before = counts_of("assertions_before");
    let after = counts_of("assertions_after");
    let verdict = mtest::check_repair(&mtest::Repair {
        patch,
        targeted: &targeted,
        assertions_before: &before,
        assertions_after: &after,
        same_set_rerun: rerun.as_ref(),
    });

    if matches!(format, OutputFormat::Json) {
        let report = serde_json::json!({
            "accepted": verdict.accepted,
            "targeted": verdict.targeted,
            "edited": verdict.edited,
            "violations": verdict.violations.iter().map(|v| serde_json::json!({
                "rule": v.rule, "detail": v.detail,
            })).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else if verdict.accepted {
        println!("\n  repair accepted — it only strengthened tests, and the same mutants now fail");
    } else {
        println!("\n  repair REJECTED:");
        for v in &verdict.violations {
            println!("\n  [{}]", v.rule);
            for line in v.detail.split(". ") {
                println!("    {line}");
            }
        }
    }
    if verdict.accepted {
        Ok(())
    } else {
        Err("repair rejected".to_string())
    }
}

fn run_cov(
    base: Option<String>,
    test: Option<String>,
    show_missing_lines: bool,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_analysis_rs::covdiff;

    if !covdiff::available() {
        return Err(
            "cargo llvm-cov is not installed. Install it with `cargo install cargo-llvm-cov` \
             and rustup component add llvm-tools-preview"
                .to_string(),
        );
    }
    let root = std::env::current_dir().map_err(|e| e.to_string())?;

    // The repository's own verified test command, so the numbers describe the
    // suite that exists rather than one invented here.
    let test_command = test.or_else(|| {
        let discovery = xencode_context_rs::anchor::discover(&root);
        discovery
            .verified(xencode_context_rs::anchor::Kind::Test)
            .map(|r| r.command.clone())
    });

    let run = covdiff::run(&root, base.as_deref(), test_command.as_deref())?;
    let c = &run.coverage;

    if matches!(format, OutputFormat::Json) {
        let report = serde_json::json!({
            "command": run.command,
            "cold_build": run.build == covdiff::ColdReport::Cold,
            "seconds": run.took.as_secs_f64(),
            "covered": c.total_covered(),
            "uncovered": c.total_uncovered(),
            "unknown": c.total_unknown(),
            "ratio": c.ratio(),
            "complete": c.is_complete(),
            "unmeasured_files": c.unmeasured,
            "summary": c.summary(),
            "notes": run.notes,
            "files": c.files.iter().map(|f| serde_json::json!({
                "file": f.file,
                "covered": f.covered,
                "uncovered": f.uncovered,
                "unknown": f.unknown,
            })).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else if show_missing_lines {
        // The read-only shape `VF-2` asks for: file to line numbers, nothing else.
        for file in &c.files {
            if file.uncovered.is_empty() && file.unknown.is_empty() {
                continue;
            }
            println!("{}", file.file);
            if !file.uncovered.is_empty() {
                println!("  never run:  {}", join(&file.uncovered));
            }
            if !file.unknown.is_empty() {
                println!("  no data:    {}", join(&file.unknown));
            }
        }
    } else {
        println!("\n  {}", c.summary());
        println!(
            "  build: {} ({}s)",
            match run.build {
                covdiff::ColdReport::Cold => "cold — every crate was rebuilt instrumented",
                covdiff::ColdReport::Warm => "warm — reused the instrumented target directory",
            },
            run.took.as_secs()
        );
        for note in &run.notes {
            println!("  note:  {note}");
        }
        for file in &c.files {
            if file.uncovered.is_empty() && file.unknown.is_empty() {
                continue;
            }
            println!("\n  {}", file.file);
            if let Some(ratio) = file.ratio() {
                println!("    {:.0}% of measurable added lines ran", ratio * 100.0);
            } else {
                println!("    no measurable line in this file");
            }
            if !file.uncovered.is_empty() {
                println!("    never run: {}", join(&file.uncovered));
            }
            if !file.unknown.is_empty() {
                println!("    no data:   {}", join(&file.unknown));
            }
        }
    }
    if c.is_complete() {
        Ok(())
    } else {
        Err("some added lines were not exercised".to_string())
    }
}

/// QO-4 — measure the hot paths, and decide whether they moved.
///
/// The table is the product here: a number without a verdict is what every
/// benchmark already prints, and a verdict the run cannot support is worse than
/// none. So a path whose samples are more than 5% spread reports `NO VERDICT`
/// with the reason, and a baseline measured over a different tree refuses every
/// comparison in it rather than quietly reporting the difference as a change.
fn run_perf(action: PerfAction) -> Result<(), String> {
    use xencode_context_rs::perf;

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let repo_root = perf::state_root(&cwd);
    let manifest_root = xencode_context_rs::verify::manifest_dir(&cwd)?;

    match action {
        PerfAction::Show { format } => show_baseline(&repo_root, format),
        PerfAction::Record { force, format } => {
            let measured = perf::measure(&manifest_root, None)?;
            let baseline = perf::record_baseline(&repo_root, &measured, force)?;
            let path = perf::Baseline::path_for(&repo_root);
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "command": measured.command,
                        "seconds": measured.took.as_secs_f64(),
                        "corpus_files": baseline.corpus_files,
                        "baseline": path.display().to_string(),
                        "forced": force,
                        "paths": baseline.benches.values().map(perf_samples_json).collect::<Vec<_>>(),
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                println!("  {}", measured.command);
                println!(
                    "  measured {} path(s) over {} file(s) in {:.1} s",
                    baseline.benches.len(),
                    baseline.corpus_files,
                    measured.took.as_secs_f64()
                );
                for samples in baseline.benches.values() {
                    println!(
                        "  {:<34}  {:>12}  ({} samples, {:.1}% spread)",
                        samples.id,
                        human_duration(samples.median()),
                        samples.n(),
                        samples.cv().unwrap_or_default() * 100.0
                    );
                }
                println!("\n  baseline written to {}", path.display());
                if force {
                    let noisy = perf::noisy_paths(&measured);
                    if !noisy.is_empty() {
                        println!(
                            "  note: recorded despite being forced; {} measured wider than \
                             the {:.0}% a verdict rests on, and a comparison against it will \
                             refuse",
                            noisy.join(", "),
                            perf::MAX_CV * 100.0
                        );
                    }
                }
            }
            Ok(())
        }
        PerfAction::Check {
            filter,
            alert_pct,
            alpha,
            format,
        } => {
            let report = perf::check(
                &repo_root,
                &manifest_root,
                filter.as_deref(),
                alert_pct,
                alpha,
            )?;
            let regressions = report.regressions();
            let refusals = report.refusals();

            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "command": report.command,
                        "seconds": report.took.as_secs_f64(),
                        "corpus_files": report.corpus_files,
                        "baseline_corpus_files": report.baseline_corpus_files,
                        "corpus_mismatch": report.corpus_mismatch,
                        "alert_pct": alert_pct,
                        "alpha": alpha,
                        "max_cv": perf::MAX_CV,
                        "regressions": regressions.len(),
                        "refusals": refusals.len(),
                        "notes": report.notes,
                        "paths": report.comparisons.iter().map(perf_comparison_json).collect::<Vec<_>>(),
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                println!("  {}", report.command);
                println!(
                    "  {} file(s) measured, baseline recorded over {}, in {:.1} s",
                    report.corpus_files,
                    report
                        .baseline_corpus_files
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "nothing".to_string()),
                    report.took.as_secs_f64()
                );
                println!(
                    "  alert at {alert_pct}% of the baseline, judged at α = {alpha}, verdicts \
                     refused past a {:.0}% spread",
                    perf::MAX_CV * 100.0
                );
                for comparison in &report.comparisons {
                    let delta = comparison
                        .delta_pct
                        .map(|d| format!("{d:+.2}%"))
                        .unwrap_or_else(|| "     —".to_string());
                    let p = comparison
                        .p_value
                        .map(|p| format!("{p:.4}"))
                        .unwrap_or_else(|| "  —  ".to_string());
                    let spread = comparison
                        .cv
                        .map(|c| format!("{:.1}%", c * 100.0))
                        .unwrap_or_else(|| "  —  ".to_string());
                    println!(
                        "\n  {}\n    {:<11}  delta {delta}   p = {p} ({})   spread {spread}",
                        comparison.id,
                        comparison.outcome.label(),
                        comparison.p_value_method.unwrap_or("not tested"),
                    );
                    if let Some(reason) = &comparison.reason {
                        println!("    {reason}");
                    }
                }
                for note in &report.notes {
                    println!("\n  note: {note}");
                }
                println!(
                    "\n  {} path(s) compared: {} regression(s), {} refusal(s)",
                    report.comparisons.len(),
                    regressions.len(),
                    refusals.len()
                );
            }

            if regressions.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "{} hot path(s) measurably slower than the baseline",
                    regressions.len()
                ))
            }
        }
    }
}

fn perf_samples_json(samples: &xencode_context_rs::perf::Samples) -> serde_json::Value {
    serde_json::json!({
        "id": samples.id,
        "samples": samples.n(),
        "median_ns": samples.median(),
        "mean_ns": samples.mean(),
        "cv": samples.cv(),
    })
}

fn perf_comparison_json(comparison: &xencode_context_rs::perf::Comparison) -> serde_json::Value {
    serde_json::json!({
        "id": comparison.id,
        "outcome": comparison.outcome.label(),
        "baseline_median_ns": comparison.baseline_median_ns,
        "current_median_ns": comparison.current_median_ns,
        "delta_pct": comparison.delta_pct,
        "p_value": comparison.p_value,
        "p_value_method": comparison.p_value_method,
        "cv": comparison.cv,
        "reason": comparison.reason,
    })
}

/// The recorded baseline and nothing else — no measurement, no verdict.
fn show_baseline(repo_root: &std::path::Path, format: OutputFormat) -> Result<(), String> {
    use xencode_context_rs::perf;

    let path = perf::Baseline::path_for(repo_root);
    let Some(baseline) = perf::Baseline::load(repo_root)? else {
        return Err(format!(
            "no baseline recorded yet — run `xencode perf record` (it would write {})",
            path.display()
        ));
    };
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&baseline).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!("  baseline at {}", path.display());
    println!(
        "  recorded over {} file(s) in {}",
        baseline.corpus_files, baseline.recorded_in
    );
    let age_days = {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(baseline.recorded_unix) / 86_400
    };
    println!(
        "  {} day(s) old, {} path(s)",
        age_days,
        baseline.benches.len()
    );
    for samples in baseline.benches.values() {
        println!(
            "  {:<34}  {:>12}  ({} samples, {:.1}% spread)",
            samples.id,
            human_duration(samples.median()),
            samples.n(),
            samples.cv().unwrap_or_default() * 100.0
        );
    }
    Ok(())
}

/// A duration in nanoseconds, in the unit the number actually reads in.
///
/// The hot paths span five orders of magnitude — the token trimmer runs in
/// microseconds, a whole-tree symbol pass in seconds — so a fixed unit gives a
/// column of either `0.015` or `801449.095`.
fn human_duration(ns: f64) -> String {
    const UNITS: [(f64, &str); 4] = [
        (1.0, "ns"),
        (1_000.0, "µs"),
        (1_000_000.0, "ms"),
        (1_000_000_000.0, "s"),
    ];
    let mut chosen = UNITS[0];
    for unit in UNITS {
        if ns >= unit.0 {
            chosen = unit;
        }
    }
    format!("{:.3} {}", ns / chosen.0, chosen.1)
}

/// `1 entry`, `3 commits` — the coverage counts are read by a person, so they
/// are not printed as `1 entr(y|ies)`.
fn plural_count(count: usize, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}

/// `xencode prices`: the two documents a cost report reads its rates out of, what
/// each of them says, and which of the models this project has actually run have
/// no rate in either.
///
/// Nothing here is computed — no spend, no totals. The point of the command is
/// provenance: a number on a cost report is only as good as the paper it came
/// from, and one of those two papers is somebody else's catalogue read on a day
/// that has already passed.
async fn run_prices(action: Option<PriceAction>) -> Result<(), String> {
    let xencode_dir = project_xencode_dir();
    let config = XencodeConfig::load().unwrap_or_default();
    let action = action.unwrap_or(PriceAction::Show {
        format: OutputFormat::Text,
    });

    match action {
        PriceAction::Fetch { url } => {
            let url = url
                .as_deref()
                .unwrap_or(xencode_providers_rs::listing::OPENROUTER_MODELS_URL);
            let body = xencode_providers_rs::listing::fetch_listing(url)
                .await
                .map_err(|e| e.to_string())?;
            let now = xencode_context_rs::conversation::now_millis();
            let lookup = xencode_context_rs::PriceLookup::parse_openrouter(&body, now)?;
            let path = lookup.write(&xencode_dir).map_err(|e| e.to_string())?;
            println!(
                "{} prices read off {url} and written to {}",
                lookup.priced_models(),
                path.display()
            );
            if lookup.unreadable > 0 {
                println!(
                    "  {} entries the listing gave in a shape no price could be read out of, counted and left out",
                    lookup.unreadable
                );
            }
            if !config.price_lookup {
                println!(
                    "note: price_lookup is off, so nothing is read from that file yet — `xencode config set price_lookup true`"
                );
            }
            println!(
                "note: a report reads that file for {} days, then stops pricing from it until it is fetched again.",
                xencode_context_rs::PRICE_TTL_DAYS
            );
        }
        PriceAction::Show { format } => {
            let table =
                xencode_context_rs::PriceTable::load_with_lookup(&xencode_dir, config.price_lookup);
            // Folded here rather than read from the rollup file: refreshing that
            // sidecar means writing to somebody's disk to answer a question about
            // it, and a project driven only through this command has never written
            // one. The records are the same ones the rollup is built from.
            let mut rollup = xencode_context_rs::MetricsRollup::empty();
            for row in xencode_context_rs::read_metrics(&xencode_dir) {
                rollup.fold(&row);
            }
            let run: Vec<String> = rollup.by_model.keys().cloned().collect();
            let unpriced: Vec<&String> = run
                .iter()
                .filter(|model| table.price_for_model(model.as_str()).is_none())
                .collect();
            let from_listing: Vec<&String> = run
                .iter()
                .filter(|model| {
                    matches!(
                        table.price_for_model(model.as_str()),
                        Some(xencode_context_rs::PriceSource::Listing { .. })
                    )
                })
                .collect();

            // Read off the disk rather than through the table, so the answer does
            // not depend on whether this build is allowed to price anything from
            // it — the question here is what the file says.
            let listing = xencode_context_rs::PriceLookup::load(&xencode_dir);
            let now = xencode_context_rs::conversation::now_millis();

            if matches!(format, OutputFormat::Json) {
                let line = serde_json::json!({
                    "pricing_json": {
                        "path": table.path.display().to_string(),
                        "present": table.file_present,
                        "models": &table.models,
                        "rejected": &table.rejected,
                    },
                    "listing": listing.as_ref().map(|lookup| serde_json::json!({
                        "source": &lookup.source,
                        "fetched_at_unix_ms": lookup.fetched_at_unix_ms,
                        "fetched_on": xencode_context_rs::local_day_key(lookup.fetched_at_unix_ms),
                        "age_days": lookup.age_days(now),
                        "expired": lookup.stale(now),
                        "priced_models": lookup.priced_models(),
                        "unreadable": lookup.unreadable,
                    })),
                    "price_lookup": config.price_lookup,
                    "listing_is_read": table.lookup.is_some() && !table.listing_expired,
                    "models_run": &run,
                    "priced_from_listing": &from_listing,
                    "unpriced": &unpriced,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&line).unwrap_or_default()
                );
                return Ok(());
            }

            println!("pricing.json — {}", table.path.display());
            if !table.file_present {
                println!("  nothing there yet, so no model is priced by hand");
            } else {
                println!(
                    "  {}, priced by hand",
                    plural_count(table.models.len(), "model", "models")
                );
                for (name, price) in &table.models {
                    println!("  • {name}  {}", describe_rates(price));
                }
            }
            for problem in &table.rejected {
                println!("  could not be read: {problem}");
            }

            println!(
                "fetched listing — {}",
                xencode_context_rs::lookup_path(&xencode_dir).display()
            );
            match &listing {
                Some(lookup) => {
                    println!(
                        "  {} prices off {}, read on {}, {} days ago",
                        lookup.priced_models(),
                        lookup.source,
                        xencode_context_rs::local_day_key(lookup.fetched_at_unix_ms),
                        lookup.age_days(now),
                    );
                    if lookup.unreadable > 0 {
                        println!("  {} entries carried no readable price", lookup.unreadable);
                    }
                    if !config.price_lookup {
                        println!(
                            "  not consulted: price_lookup is off. `xencode config set price_lookup true` lets a model missing from pricing.json be priced from here; a rate written by hand always outranks one read off a listing."
                        );
                    } else if let Some(note) = table.listing_expired_note(now) {
                        println!("{note}");
                    }
                }
                None => println!(
                    "  nothing fetched. `xencode prices fetch` reads the public catalogue once and keeps it here."
                ),
            }

            if run.is_empty() {
                println!("models this project has run: none recorded yet");
            } else {
                println!(
                    "models this project has run: {}; priced from the listing: {}; with no price in either document: {}",
                    run.len(),
                    from_listing.len(),
                    unpriced.len()
                );
                // Which name each looked-up price was matched under, because a
                // rate that arrived by matching a different string has to be
                // checkable against the catalogue it came from.
                for model in &from_listing {
                    if let Some(xencode_context_rs::PriceSource::Listing { id, price }) =
                        table.price_for_model(model)
                    {
                        println!(
                            "  • {model} — the listing's {id}: {}",
                            describe_rates(price)
                        );
                    }
                }
                for model in &unpriced {
                    println!(
                        "  • {model} — no price. A cost is reported as unknown, never as nothing."
                    );
                }
            }
        }
    }
    Ok(())
}

/// A hand-written rate the way it is written into the file, so `prices show` can
/// be read against the editor rather than against a mental conversion.
fn describe_rates(price: &xencode_context_rs::ModelPrice) -> String {
    // `format_usd` already writes the dollar sign, so none is added here.
    let mut line = format!(
        "{} in / {} out per million tokens",
        xencode_context_rs::format_usd((price.input_usd_per_mtok * 1_000_000.0).round() as u64),
        xencode_context_rs::format_usd((price.output_usd_per_mtok * 1_000_000.0).round() as u64),
    );
    line.push_str(&match price.cached_input_usd_per_mtok {
        Some(cached) => format!(
            ", cache reads {}",
            xencode_context_rs::format_usd((cached * 1_000_000.0).round() as u64)
        ),
        None => ", no cache rate — reads billed as input, which is an upper bound".to_string(),
    });
    line
}

/// QO-6 — put the release notes together from the two places this project
/// already writes about what shipped: the commits since the last release, and
/// the changelog's own unreleased block.
///
/// The two disagree in both directions and both disagreements are the product:
/// work that shipped with no entry behind it never reaches a reader, and an entry
/// whose commit sits below the release line would be announced twice. The output
/// is a draft — to standard output, or to a path that must not already exist
/// unless `--force` says otherwise, because a person is meant to edit it after.
fn run_release_notes(
    from: Option<String>,
    to: Option<String>,
    release: Option<String>,
    out: Option<PathBuf>,
    force: bool,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_context_rs::releasenotes;

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let draft = releasenotes::draft(&cwd, from.as_deref(), to.as_deref())?;
    let markdown = releasenotes::to_markdown(&draft, release.as_deref());

    let written = match &out {
        Some(path) => {
            let path = if path.is_absolute() {
                path.clone()
            } else {
                cwd.join(path)
            };
            releasenotes::write_draft(&path, &markdown, force)?;
            Some(path)
        }
        None => None,
    };

    let named = draft.commits.len() - draft.uncovered.len();
    match format {
        OutputFormat::Json => println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "to": draft.to,
                "from": draft.from,
                "range_note": draft.range_note,
                "entries": draft.entries.iter().map(|entry| serde_json::json!({
                    "category": entry.category,
                    "title": entry.title,
                    "ids": entry.ids,
                })).collect::<Vec<_>>(),
                "commits": draft.commits.len(),
                "named_by_an_entry": named,
                "covered_ids": draft.covered_ids(),
                "uncovered": draft.uncovered.iter().map(|commit| serde_json::json!({
                    "hash": releasenotes::short_hash(&commit.hash),
                    "subject": commit.subject,
                    "ids": commit.ids,
                })).collect::<Vec<_>>(),
                "unmatched": draft.unmatched.iter().map(|entry| serde_json::json!({
                    "title": entry.title,
                    "ids": entry.ids,
                })).collect::<Vec<_>>(),
                "out": written.as_ref().map(|path| path.display().to_string()),
            }))
            .map_err(|e| e.to_string())?
        ),
        OutputFormat::Text => {
            match &written {
                Some(path) => {
                    println!("  draft written to {}", path.display());
                    println!("  {}", draft.range_note);
                    println!(
                        "  {}, {} of them named by the changelog's unreleased block ({})",
                        plural_count(draft.commits.len(), "commit", "commits"),
                        named,
                        plural_count(draft.entries.len(), "entry", "entries"),
                    );
                    if !draft.uncovered.is_empty() {
                        println!(
                            "  {}, listed in the draft",
                            plural_count(
                                draft.uncovered.len(),
                                "commit no entry accounts for",
                                "commits no entry accounts for"
                            )
                        );
                    }
                    if !draft.unmatched.is_empty() {
                        println!(
                            "  {}, listed in the draft",
                            plural_count(
                                draft.unmatched.len(),
                                "entry names no commit in the range",
                                "entries name no commit in the range"
                            )
                        );
                    }
                    if draft.uncovered.is_empty() && draft.unmatched.is_empty() {
                        println!("  nothing to reconcile: every commit is explained, every entry matches");
                    }
                }
                None => print!("{markdown}"),
            }
        }
    }
    Ok(())
}

/// Line numbers as compact ranges, so `1,2,3,7,9,10` reads as `1-3, 7, 9-10`.
fn join(lines: &[u32]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut start: Option<u32> = None;
    let mut prev: u32 = 0;
    for &line in lines {
        match start {
            None => start = Some(line),
            Some(_) if line == prev + 1 => {}
            Some(s) => {
                out.push(if s == prev {
                    s.to_string()
                } else {
                    format!("{s}-{prev}")
                });
                start = Some(line);
            }
        }
        prev = line;
    }
    if let Some(s) = start {
        out.push(if s == prev {
            s.to_string()
        } else {
            format!("{s}-{prev}")
        });
    }
    out.join(", ")
}

/// Eight args because the  subcommand grew a second mode ()
/// beside the suite mode, and splitting the function would separate the ledger
/// write the two modes share. The ban stays on everywhere else.
#[allow(clippy::too_many_arguments)]
fn run_test(
    packages: Vec<String>,
    retries: u32,
    stress_count: u32,
    timeout: u64,
    isolate: Option<String>,
    base: String,
    repeat: u32,
    format: OutputFormat,
    session: Option<String>,
) -> Result<(), String> {
    use xencode_context_rs::verify;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    if let Some(test) = isolate {
        let isolation = verify::isolate(
            &root,
            &test,
            &base,
            repeat,
            std::time::Duration::from_secs(timeout),
        );
        if matches!(format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::json!({
                    "test": isolation.test,
                    "base": isolation.base,
                    "work_fails": isolation.work_fails,
                    "work_runs": isolation.work_runs,
                    "base_fails": isolation.base_fails,
                    "base_runs": isolation.base_runs,
                    "class": isolation.class.label(),
                    "notes": isolation.notes,
                })
            );
        } else {
            println!("\n  {} — {}", isolation.class.label(), isolation.test);
            println!(
                "  base ({}): {}",
                isolation.base,
                match isolation.base_fails {
                    Some(f) => format!("{f}/{} failed", isolation.base_runs),
                    None => "could not run".to_string(),
                }
            );
            println!(
                "  worktree: {}/{} failed",
                isolation.work_fails, isolation.work_runs
            );
            for note in &isolation.notes {
                println!("  note: {note}");
            }
            if isolation.class == verify::FailureClass::PreExisting {
                println!("\n  not yours — do not fix it here");
            }
        }
        return if isolation.class == verify::FailureClass::Introduced {
            Err("the failure was introduced in the working tree".to_string())
        } else {
            Ok(())
        };
    }
    let opts = verify::Options {
        filter: None,
        retries,
        stress: stress_count,
        packages,
        budget: std::time::Duration::from_secs(timeout),
    };
    let outcome = verify::run(&root, &opts);

    // EVd-1's first producer: every run leaves a row — session, class, exit
    // code, what it ran against — whether or not anyone reads it back. The run
    // log goes to the session artifacts, tail-capped, and the ledger row points
    // at it; then old passing sessions are pruned, because a test loop that
    // never prunes fills a disk.
    {
        use xencode_context_rs::{artifacts, ledger};
        let xencode_dir = root.join(xencode_context_rs::XENCODE_DIR);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let mut log = format!(
            "command: {}\nexit: {}\nflaky: {}\nfailed: {}\n",
            outcome.command,
            outcome.exit.unwrap_or(-1),
            outcome.flaky.join(", "),
            outcome.failed.join(", "),
        );
        for note in &outcome.notes {
            log.push_str(&format!("note: {note}\n"));
        }
        let target_session = session.or_else(|| {
            ConversationMemory::with_persistence(50)
                .ok()
                .and_then(|m| m.current_session().cloned())
        });
        let session_tag = target_session.as_deref().unwrap_or("cli");
        let log_ref = artifacts::write_artifact(
            &xencode_dir,
            session_tag,
            &format!("test-{now_ms}.log"),
            &log,
        )
        .ok()
        .map(|p| {
            p.strip_prefix(&root)
                .map(|r| r.to_string_lossy().into_owned())
                .unwrap_or_else(|_| p.display().to_string())
        })
        .unwrap_or_default();
        let _ = artifacts::prune_artifacts(&xencode_dir, artifacts::KEEP_LAST_PASSING);
        let entry = ledger::LedgerEntry {
            ts_unix_ms: now_ms,
            session: Some(session_tag.to_string()),
            run_class: ledger::RunClass::Test,
            exit_code: outcome.exit.unwrap_or(-1),
            subjects: vec![ledger::digest_hex(&outcome.command)],
            log_ref,
            note: String::new(),
        };
        let _ = ledger::append_ledger(&xencode_dir, &entry);
    }

    if matches!(format, OutputFormat::Json) {
        let report = serde_json::json!({
            "ok": outcome.ok,
            "engine": outcome.engine.map(|e| e.label()),
            "command": outcome.command,
            "exit": outcome.exit,
            "flaky": outcome.flaky,
            "failed": outcome.failed,
            "notes": outcome.notes,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else {
        if let Some(engine) = outcome.engine {
            println!("\n  engine: {}", engine.label());
        }
        if !outcome.command.is_empty() {
            println!("  ran:    {}", outcome.command);
        }
        for note in &outcome.notes {
            println!("  note:   {note}");
        }
        if !outcome.flaky.is_empty() {
            println!("\n  FLAKY — passed only on a retry, so not a pass:\n");
            for name in &outcome.flaky {
                println!("    {name}");
            }
        }
        if !outcome.failed.is_empty() {
            println!("\n  FAILED:\n");
            for name in &outcome.failed {
                println!("    {name}");
            }
        }
        println!(
            "\n  {}",
            if outcome.ok {
                "pass"
            } else {
                "not a pass — see above"
            }
        );
    }
    if outcome.ok {
        Ok(())
    } else {
        Err("tests did not pass".to_string())
    }
}

fn run_anchor(
    path: PathBuf,
    timeout: u64,
    dry_run: bool,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_context_rs::anchor;

    let root = path.canonicalize().unwrap_or(path);
    let mut discovery = anchor::discover(&root);

    if !dry_run {
        let budget = std::time::Duration::from_secs(timeout);
        for recipe in &mut discovery.recipes {
            eprintln!("  running: {}", recipe.command);
            anchor::prove(&root, recipe, budget);
        }
    }

    let text = anchor::render(&discovery);
    let verified = discovery.recipes.iter().filter(|r| r.is_verified()).count();
    let found = discovery.recipes.len();

    let written = if dry_run {
        None
    } else if anchor::is_current(&root, &text) {
        eprintln!("  anchor.md already current — left untouched");
        Some(root.join(xencode_context_rs::XENCODE_DIR).join("anchor.md"))
    } else {
        match anchor::write_anchor(&root, &text) {
            Ok(p) => Some(p),
            Err(e) => return Err(format!("could not write the anchor: {e}")),
        }
    };

    if !dry_run {
        let now_s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let meta = anchor::AnchorMeta {
            proved_at_unix_s: now_s,
            candidates: found,
            verified,
        };
        if let Err(e) = anchor::write_anchor_meta(&root, &meta) {
            eprintln!("  warning: could not write anchor.meta sidecar: {e}");
        }
    }

    if matches!(format, OutputFormat::Json) {
        let report = serde_json::json!({
            "probed": discovery.probed,
            "candidates": found,
            "verified": verified,
            "dry_run": dry_run,
            "written": written.as_ref().map(|p| p.display().to_string()),
            "recipes": discovery.recipes.iter().map(|r| serde_json::json!({
                "kind": r.kind.label(),
                "command": r.command,
                "source": r.source,
                "provenance": format!("{:?}", r.provenance).to_lowercase(),
                "verdict": r.verdict.as_ref().map(|v| v.describe()),
                "verified": r.is_verified(),
            })).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
    } else {
        println!("\n  anchor — {}", root.display());
        if discovery.probed.is_empty() {
            println!("\n  read no CI file, task runner, manifest or README to learn from");
        } else {
            println!("  read: {}", discovery.probed.join(", "));
        }
        println!("\n  {} candidate(s), {} verified\n", found, verified);
        for recipe in &discovery.recipes {
            let mark = if recipe.is_verified() {
                "verified"
            } else {
                "unverified"
            };
            println!(
                "    {:<9} {:<10} {}  [{}]",
                recipe.kind.label(),
                mark,
                recipe.command,
                recipe.source
            );
        }
        if verified == 0 && !dry_run {
            println!("\n  nothing was proven — anchor.md says so rather than implying otherwise");
        }
        if let Some(p) = written {
            println!("\n  {}", p.display());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // CLI flags map 1:1 to interop options; a struct would just rename them
fn run_interop(
    agents: Vec<String>,
    timeout: u64,
    out: Option<std::path::PathBuf>,
    format: OutputFormat,
    repeat: u32,
    fan_out: bool,
    check_auth: bool,
    capture_dir: Option<std::path::PathBuf>,
    trace: Option<std::path::PathBuf>,
) -> Result<(), String> {
    // Also read-only and also free: a trace view is a file the operator already
    // paid for, rendered again, so it launches nothing.
    if let Some(dir) = &trace {
        let dirs = xencode_agents_rs::capture::find_captures(dir);
        if dirs.is_empty() {
            return Err(format!(
                "{} holds no capture; point --trace at a directory written by --capture-dir",
                dir.display()
            ));
        }
        for (i, capture_dir) in dirs.iter().enumerate() {
            let capture = xencode_agents_rs::capture::read_capture(capture_dir)
                .map_err(|e| format!("cannot read that capture: {e}"))?;
            if i > 0 {
                println!("\n{}\n", "─".repeat(60));
            }
            println!("{}", capture.trace());
        }
        return Ok(());
    }
    // Checked first and on its own: it launches nothing and spends nothing, so
    // it is the cheap way to find out what a real run would do.
    if check_auth {
        let selected: Vec<&xencode_agents_rs::AgentSpec> = if agents.is_empty() {
            xencode_agents_rs::ROSTER.iter().collect()
        } else {
            xencode_agents_rs::ROSTER
                .iter()
                .filter(|a| agents.iter().any(|n| n == a.name))
                .collect()
        };
        println!("credential status — read-only, nothing launched, nothing signed in\n");
        for spec in selected {
            println!(
                "  {}",
                xencode_agents_rs::probe::credential_status(spec).summary()
            );
        }
        return Ok(());
    }
    let scratch = tempfile::tempdir().map_err(|e| format!("cannot make a scratch dir: {e}"))?;
    let workdir = scratch.path().join("task");
    xencode_agents_rs::probe::seed_task_dir(&workdir)
        .map_err(|e| format!("cannot seed the task fixture: {e}"))?;

    let options = xencode_agents_rs::ProbeOptions {
        only: agents,
        timeout: std::time::Duration::from_secs(timeout.max(1)),
        workdir,
        task: xencode_agents_rs::probe::default_task().to_string(),
        repeat,
        fan_out,
    };
    let report = xencode_agents_rs::probe::run_probe(&options);

    // The operator may keep each whole run, not only the lines the report chose
    // to show. This writes from the bytes the probe already received, so keeping
    // a capture costs no extra run of anything.
    let kept: Vec<std::path::PathBuf> = match &capture_dir {
        None => Vec::new(),
        Some(root) => {
            std::fs::create_dir_all(root)
                .map_err(|e| format!("cannot create {}: {e}", root.display()))?;
            let runs: Vec<&xencode_agents_rs::RunCapture> = if report.all_runs.is_empty() {
                report.captures.iter().collect()
            } else {
                report.all_runs.iter().collect()
            };
            let mut dirs = Vec::new();
            for (index, run) in runs.iter().enumerate() {
                // Repeats are compared against each other, so a second run must
                // not overwrite the first one's raw stream.
                let base = if repeat > 1 {
                    root.join(format!("run-{}", index + 1))
                } else {
                    root.clone()
                };
                dirs.push(xencode_agents_rs::capture::write_capture(&base, run)?);
            }
            dirs
        }
    };

    if let Some(path) = &out {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| format!("cannot render the report: {e}"))?;
        // Atomic, like every other write in the product: a report half-written
        // is worse than no report.
        xencode_core_rs::write_atomic(path, json.as_bytes())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }

    match format {
        OutputFormat::Json => println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|e| format!("cannot render the report: {e}"))?
        ),
        _ => {
            println!("interop probe — {}", report.run_on);
            println!();
            for capture in &report.captures {
                println!("  {}", capture.summary());
            }
            if !report.absent.is_empty() {
                println!("\n  no binary on PATH for these names:");
                for absent in &report.absent {
                    println!("    {} — {}", absent.name, absent.why);
                }
            }
            if !report.unknown.is_empty() {
                println!("\n  installed here but this probe has no adapter for them:");
                for unknown in &report.unknown {
                    println!("    {} — {}", unknown.name, unknown.why);
                }
            }
            if let Some(fan) = &report.fan_out {
                println!("\n  fan-out: {}", fan.summary());
            }
            let with_usage: Vec<_> = report
                .captures
                .iter()
                .filter(|c| c.provenance == xencode_agents_rs::Provenance::Observed)
                .collect();
            if !with_usage.is_empty() {
                println!("\n  what the working agents said they cost:");
                for capture in &with_usage {
                    let cost = match &capture.usage {
                        Some(usage) => usage.summary(),
                        None => "reported no usage at all".to_string(),
                    };
                    let model = capture.model.as_deref().unwrap_or("did not name a model");
                    println!("    {}: {cost} [model: {model}]", capture.agent);
                }
            }
            if !report.parked.is_empty() {
                println!("\n  stood down by the operator, so not run (name one with --agent to probe it anyway):");
                for parked in &report.parked {
                    println!("    {} — {}", parked.name, parked.why);
                }
            }
            if let Some(first) = report.stability.first() {
                println!("\n  across {} run(s) each:", first.runs);
                for verdict in &report.stability {
                    println!("    {}", verdict.summary());
                }
            }
            println!("\n  still unanswered:");
            for item in &report.unanswered {
                println!("    - {item}");
            }
            if let Some(path) = &out {
                println!("\n  report written to {}", path.display());
            }
        }
    }
    if !kept.is_empty() {
        let note = format!(
            "\n  {} capture(s) kept; read one back with `xencode interop --trace <dir>`",
            kept.len()
        );
        match format {
            // stdout stays a parseable report.
            OutputFormat::Json => eprintln!("{note}"),
            _ => println!("{note}"),
        }
        for dir in &kept {
            match format {
                OutputFormat::Json => eprintln!("    {}", dir.display()),
                _ => println!("    {}", dir.display()),
            }
        }
    }
    Ok(())
}

fn run_review(
    explicit_base: Option<String>,
    session: Option<String>,
    format: OutputFormat,
) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let root = resolve_review_root(&cwd);
    let (base, source) = resolve_review_base(&root, explicit_base.as_deref());
    let diffs = xencode_context_rs::git_diff_numstat(&root, &base)?;
    let files: Vec<ReviewedFile> = diffs.iter().map(|d| review_file(&root, d)).collect();
    // AE-1: the evidence half, from the ledger rows this session wrote.
    let envelope = session.as_deref().map(|session| {
        xencode_context_rs::envelope_for_session(
            &root.join(xencode_context_rs::XENCODE_DIR),
            session,
            &format!("changes against {base}"),
            files.iter().map(|f| f.path.clone()).collect(),
        )
    });
    match format {
        OutputFormat::Json => {
            let arr: Vec<serde_json::Value> = files
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "path": f.path,
                        "added": f.added,
                        "deleted": f.deleted,
                        "issues": f.issues,
                        "note": f.note,
                    })
                })
                .collect();
            let mut output = serde_json::json!({
                "base": base,
                "files_changed": files.len(),
                "files": arr,
            });
            if let Some(obj) = output.as_object_mut() {
                obj.insert(
                    "base_source".to_string(),
                    serde_json::Value::String(source.label().to_string()),
                );
                if let Some(envelope) = &envelope {
                    obj.insert(
                        "envelope".to_string(),
                        serde_json::to_value(envelope).map_err(|e| e.to_string())?,
                    );
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?
            );
        }
        OutputFormat::Text => {
            println!(
                "{}",
                format_review_text(&base, Some(&source.header_suffix()), &files)
            );
            for file in &files {
                for issue in &file.issues {
                    let icon = match issue.severity.label() {
                        "critical" | "high" => "R",
                        "medium" => "Y",
                        _ => "G",
                    };
                    println!(
                        "    {} [{}] {} Ln{}: {} -- {}",
                        icon,
                        issue.severity.label(),
                        file.path,
                        issue.line_number,
                        issue.message,
                        issue.suggestion
                    );
                }
            }
        }
    }
    if let (OutputFormat::Text, Some(envelope)) = (&format, &envelope) {
        println!(
            "
Result envelope, from .xencode/ledger.jsonl:"
        );
        print!("{}", envelope.for_reviewer().render());
    }
    Ok(())
}

/// `.xencode` for the tree being worked on, in the same place the TUI writes its
/// recordings and its traces.
fn project_xencode_dir() -> std::path::PathBuf {
    xencode_context_rs::default_root().join(xencode_context_rs::XENCODE_DIR)
}

/// The first thing the person asked, for `--list`. A recording is named by a
/// number nobody remembers, so the listing shows the prompt instead.
fn recorded_prompt(session: &xencode_context_rs::Session) -> Option<String> {
    let opening = session.opening_messages()?;
    let last_user = opening
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(|r| r.as_str()) == Some("user"))?;
    let text = match last_user.get("content")? {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" "),
        _ => return None,
    };
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.is_empty() {
        return None;
    }
    Some(if one_line.chars().count() > 72 {
        format!("{}…", one_line.chars().take(72).collect::<String>())
    } else {
        one_line
    })
}

/// The word for a count, so a listing says "1 recording" rather than
/// "1 recording(s)".
fn count_word(count: usize, one: &'static str, many: &'static str) -> &'static str {
    if count == 1 {
        one
    } else {
        many
    }
}

fn list_recordings(xencode_dir: &std::path::Path) -> Result<(), String> {
    let dir = xencode_context_rs::sessions_dir(xencode_dir);
    let ids = xencode_context_rs::list_session_ids(xencode_dir);
    if ids.is_empty() {
        println!("nothing recorded in {}", dir.display());
        println!(
            "a run is recorded only with session_recording on \
             (`xencode config set session_recording true`), and only when the model is served by \
             Ollama, llama.cpp, an OpenAI-compatible endpoint or OpenRouter."
        );
        return Ok(());
    }
    println!(
        "{} {} in {}",
        ids.len(),
        count_word(ids.len(), "recording", "recordings"),
        dir.display()
    );
    for id in ids {
        let Ok(session) = xencode_context_rs::read_session(xencode_dir, &id) else {
            // A half-written file is still a run that happened; say what is
            // unreadable about it rather than leaving it out of the list.
            println!("{id}  unreadable");
            continue;
        };
        let tools: usize = session.calls.iter().map(|call| call.tools.len()).sum();
        println!(
            "{id}  {} model {}, {} tool {}, {}",
            session.calls.len(),
            count_word(session.calls.len(), "call", "calls"),
            tools,
            count_word(tools, "call", "calls"),
            session.run.model
        );
        if let Some(prompt) = recorded_prompt(&session) {
            println!("    \"{prompt}\"");
        }
    }
    Ok(())
}

/// Read the run ledger, nothing else: no model is asked, no server is
/// dialled, so this works with everything down.
fn run_runs(action: Option<RunsAction>) -> Result<(), String> {
    let xencode_dir = project_xencode_dir();
    let action = action.unwrap_or(RunsAction::List {
        limit: xencode_context_rs::RUNS_WINDOW,
        format: OutputFormat::Text,
    });
    match action {
        RunsAction::List { limit, format } => {
            let rows = xencode_context_rs::recent_runs(&xencode_dir, limit);
            if rows.is_empty() {
                println!(
                    "no runs in {}",
                    xencode_context_rs::runs_path(&xencode_dir).display()
                );
                println!("a row is written when an agent turn ends in the TUI.");
                return Ok(());
            }
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            for row in &rows {
                let granted = row
                    .approvals
                    .iter()
                    .filter(|a| a.decision.granted())
                    .count();
                let denied = row.approvals.len() - granted;
                let comp_info = match &row.computer {
                    Some(c) => format!("  [{c}]"),
                    None => String::new(),
                };
                println!(
                    "{}  {}  {} asked ({} allowed, {} denied)  {}{}",
                    row.run_id,
                    row.model.as_deref().unwrap_or("model ?"),
                    row.approvals.len(),
                    granted,
                    denied,
                    row.recording.as_deref().unwrap_or("no recording"),
                    comp_info,
                );
            }
            Ok(())
        }
        RunsAction::Show { run_id, format } => {
            let row = xencode_context_rs::run_by_id(&xencode_dir, &run_id).ok_or_else(|| {
                format!(
                    "no run {run_id} in {}",
                    xencode_context_rs::runs_path(&xencode_dir).display()
                )
            })?;
            let evidence = xencode_context_rs::run_evidence(&xencode_dir, &row);
            if matches!(format, OutputFormat::Json) {
                let doc = serde_json::json!({
                    "run": row,
                    "checks": evidence.checks,
                    "verified": evidence.verified(),
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            println!("run {}", row.run_id);
            println!("  model: {}", row.model.as_deref().unwrap_or("?"));
            if let Some(c) = &row.computer {
                println!("  computer: {c}");
            }
            println!(
                "  session: {}",
                row.session.as_deref().unwrap_or("(none open)")
            );
            println!(
                "  recording: {}",
                row.recording.as_deref().unwrap_or("none")
            );
            if row.approvals.is_empty() {
                println!("  approvals: none asked");
            } else {
                println!("  approvals:");
                for a in &row.approvals {
                    println!("    {} [{}]: {}", a.tool, a.class, a.decision.as_str());
                }
            }
            // The join the ledger exists to make: the run names a session, and
            // the session's verification rows are read from the EVd-1 ledger,
            // not copied here.
            if evidence.checks.is_empty() {
                println!("  checks: none — nothing verified this run");
            } else {
                println!(
                    "  checks: {} ({} passed)",
                    evidence.checks.len(),
                    evidence.checks.iter().filter(|c| c.passed()).count(),
                );
                for check in &evidence.checks {
                    println!("    exit {}  {}", check.exit_code, check.log_ref);
                }
            }
            Ok(())
        }
        RunsAction::Trailer { run_id } => {
            let row = xencode_context_rs::run_by_id(&xencode_dir, &run_id).ok_or_else(|| {
                format!(
                    "no run {run_id} in {}",
                    xencode_context_rs::runs_path(&xencode_dir).display()
                )
            })?;
            println!(
                "{}",
                xencode_context_rs::accountability_trailer(&row, env!("CARGO_PKG_VERSION"))
            );
            Ok(())
        }
    }
}

/// The flags `xencode run` takes, as the dispatcher hands them over.
struct DetachedCommandOptions {
    prompt: Option<String>,
    detach: bool,
    resume: Option<String>,
    list: bool,
    show: Option<String>,
    log: Option<String>,
    stop: Option<String>,
    model: Option<String>,
    tool_root: Option<PathBuf>,
    max_rounds: Option<u32>,
    max_minutes: Option<f64>,
    max_cost: Option<f64>,
    allow_shell: bool,
    tail: usize,
    ollama_url: Option<String>,
    llamacpp_url: Option<String>,
    child: Option<String>,
    child_xencode_dir: Option<PathBuf>,
}

use xencode_tui_rs::detached as detached_runs;

/// Run an agent turn from the command line: foreground, detached, resumed,
/// or reported on. Exactly one action per invocation — a prompt with `--log`
/// is two things being asked at once, and gets an error saying so.
async fn run_detached_command(options: DetachedCommandOptions) -> Result<(), String> {
    if let Some(run_id) = options.child {
        // The forked worker. Its log file carries the transcript; the
        // process exit only says whether the spec was even readable. The
        // state directory rides along because the worker's cwd is the
        // run's tree, not the project.
        let xencode_dir = options
            .child_xencode_dir
            .unwrap_or_else(project_xencode_dir);
        let exit = detached_runs::run_child(&xencode_dir, &run_id).await;
        if matches!(exit.reason, xencode_tui_rs::detached::ExitReason::Error) {
            return Err(exit.note.clone());
        }
        return Ok(());
    }
    let DetachedCommandOptions {
        prompt,
        detach,
        resume,
        list,
        show,
        log,
        stop,
        model,
        tool_root,
        max_rounds,
        max_minutes,
        max_cost,
        allow_shell,
        tail,
        ollama_url,
        llamacpp_url,
        child: _,
        child_xencode_dir: _,
    } = options;
    let actions = prompt.is_some() as u8
        + resume.is_some() as u8
        + list as u8
        + show.is_some() as u8
        + log.is_some() as u8
        + stop.is_some() as u8;
    if actions != 1 {
        return Err(
            "pick one: a prompt to run, --resume, --list, --show, --log or --stop".to_string(),
        );
    }
    let xencode_dir = project_xencode_dir();
    if list {
        return list_detached_runs(&xencode_dir);
    }
    if let Some(given) = show {
        return show_detached_run(&xencode_dir, &given);
    }
    if let Some(given) = log {
        return log_detached_run(&xencode_dir, &given, tail);
    }
    if let Some(given) = stop {
        return stop_detached_run(&xencode_dir, &given);
    }
    if let Some(given) = &resume {
        let run_id = detached_runs::resolve_run_id(&xencode_dir, given).ok_or_else(|| {
            format!(
                "no detached run {given} in {}",
                detached_runs::detached_dir(&xencode_dir).display()
            )
        })?;
        return resume_detached_run(&xencode_dir, &run_id, detach).await;
    }
    let prompt = prompt.expect("clap counted exactly one action, and it is the prompt");
    start_detached_run(
        &xencode_dir,
        &prompt,
        StartOptions {
            detach,
            model,
            tool_root,
            max_rounds,
            max_minutes,
            max_cost,
            allow_shell,
            ollama_url,
            llamacpp_url,
        },
    )
    .await
}

struct StartOptions {
    detach: bool,
    model: Option<String>,
    tool_root: Option<PathBuf>,
    max_rounds: Option<u32>,
    max_minutes: Option<f64>,
    max_cost: Option<f64>,
    allow_shell: bool,
    ollama_url: Option<String>,
    llamacpp_url: Option<String>,
}

/// Check the caps before anything runs: a cap that cannot count cannot stop,
/// so it refuses the run instead of starting one it cannot end.
fn build_caps(
    xencode_dir: &std::path::Path,
    model: &str,
    max_rounds: Option<u32>,
    max_minutes: Option<f64>,
    max_cost: Option<f64>,
) -> Result<xencode_tui_rs::detached::DetachedCaps, String> {
    let config = XencodeConfig::load().unwrap_or_default();
    let mut caps = xencode_tui_rs::detached::DetachedCaps {
        max_rounds: max_rounds.unwrap_or(config.agent_max_rounds.clamp(1, 64) as u32),
        ..Default::default()
    };
    if caps.max_rounds == 0 {
        return Err("--max-rounds stops nothing at 0; pass at least 1".to_string());
    }
    if let Some(minutes) = max_minutes {
        if !minutes.is_finite() || minutes <= 0.0 {
            return Err("--max-minutes stops nothing at 0; pass a positive number".to_string());
        }
        caps.max_wall_ms = (minutes * 60_000.0).round().max(1.0) as u64;
    }
    if let Some(dollars) = max_cost {
        if !dollars.is_finite() || dollars <= 0.0 {
            return Err("--max-cost stops nothing at 0; pass a positive amount".to_string());
        }
        let table =
            xencode_context_rs::PriceTable::load_with_lookup(xencode_dir, config.price_lookup);
        if table.price_for_model(model).is_none() {
            return Err(format!(
                "no price for {model} in pricing.json or the fetched listing, so --max-cost \
                 cannot count it — name a price first, or run without the cap"
            ));
        }
        caps.max_cost_micros = Some((dollars * 1_000_000.0).round().max(1.0) as u64);
    }
    Ok(caps)
}

/// Write the spec and either run it here or fork it. Both paths persist the
/// same state, so a foreground run killed from another terminal resumes the
/// same way a detached one does.
async fn start_detached_run(
    xencode_dir: &std::path::Path,
    prompt: &str,
    options: StartOptions,
) -> Result<(), String> {
    let config = XencodeConfig::load().unwrap_or_default();
    let model = options
        .model
        .unwrap_or_else(|| config.default_model.clone());
    if model.trim().is_empty() {
        return Err("no model: pass --model or set one in the config".to_string());
    }
    let tool_root = options
        .tool_root
        .unwrap_or_else(xencode_context_rs::default_root);
    if !tool_root.is_dir() {
        return Err(format!("{} is not a directory", tool_root.display()));
    }
    let caps = build_caps(
        xencode_dir,
        &model,
        options.max_rounds,
        options.max_minutes,
        options.max_cost,
    )?;
    let run_id = xencode_context_rs::new_run_id(prompt);
    let dir = detached_runs::run_dir(xencode_dir, &run_id);
    let spec = xencode_tui_rs::detached::DetachedSpec {
        kind: xencode_tui_rs::detached::DETACHED_KIND_RUN.to_string(),
        prompt: prompt.to_string(),
        model,
        tool_root: tool_root.to_string_lossy().into_owned(),
        caps,
        allow_shell: options.allow_shell,
        ollama_url: options.ollama_url,
        llamacpp_url: options.llamacpp_url,
        created_ms: xencode_context_rs::conversation::now_millis(),
    };
    detached_runs::write_spec(&dir, &spec)
        .map_err(|e| format!("could not write the run spec in {}: {e}", dir.display()))?;
    if !options.detach {
        let exit = detached_runs::run_child(xencode_dir, &run_id).await;
        print_detached_exit(&run_id, &exit);
        return Ok(());
    }
    let exe = std::env::current_exe().map_err(|e| format!("cannot re-run this binary: {e}"))?;
    let pid = detached_runs::spawn_child(
        &exe,
        &run_id,
        xencode_dir,
        &tool_root,
        &detached_runs::log_path(&dir),
    )
    .map_err(|e| format!("could not fork the run: {e}"))?;
    let _ = detached_runs::write_pid(&dir, pid);
    println!("detached run {run_id} (pid {pid})");
    println!(
        "watch it with `xencode run --log {run_id}`, stop it with `xencode run --stop {run_id}`"
    );
    Ok(())
}

/// Continue a run that died: same spec, same caps minus what is spent, prior
/// rounds as history. Anything but a crash is refused with the reason.
async fn resume_detached_run(
    xencode_dir: &std::path::Path,
    run_id: &str,
    detach: bool,
) -> Result<(), String> {
    let dir = detached_runs::run_dir(xencode_dir, run_id);
    match detached_runs::derive_status(&dir) {
        xencode_tui_rs::detached::DetachedStatus::Missing => {
            return Err(format!("no detached run {run_id}"));
        }
        xencode_tui_rs::detached::DetachedStatus::Finished(exit) => {
            return Err(format!(
                "run {run_id} already finished ({:?}); there is nothing to resume",
                exit.reason
            ));
        }
        xencode_tui_rs::detached::DetachedStatus::Stopped { .. } => {
            return Err(format!(
                "run {run_id} was stopped; resume a crash, not a decision"
            ));
        }
        xencode_tui_rs::detached::DetachedStatus::Running { pid } => {
            return Err(format!("run {run_id} is still going under pid {pid}"));
        }
        xencode_tui_rs::detached::DetachedStatus::NeverRan
        | xencode_tui_rs::detached::DetachedStatus::Crashed { .. } => {}
    }
    let spec = detached_runs::read_spec(&dir).ok_or_else(|| format!("no detached run {run_id}"))?;
    let (used_rounds, used_wall_ms, used_cost) = detached_runs::used_totals(&dir);
    // The caps are spent by the run, not the attempt: resuming into an
    // already-spent budget would start a loop its first round must end.
    if xencode_tui_rs::detached::check_caps(used_rounds, used_wall_ms, used_cost, &spec.caps)
        .is_some()
    {
        return Err(format!(
            "run {run_id} already spent its caps ({used_rounds} rounds); there is nothing to resume under"
        ));
    }
    // The cost rate is re-read, not trusted from the attempt that died: the
    // file may have changed underfoot, and spending unpriced is what the
    // start-time check exists to refuse.
    if spec.caps.max_cost_micros.is_some() {
        let config = XencodeConfig::load().unwrap_or_default();
        let table =
            xencode_context_rs::PriceTable::load_with_lookup(xencode_dir, config.price_lookup);
        if table.price_for_model(&spec.model).is_none() {
            return Err(format!(
                "no price for {} anymore, so the cost cap cannot count — name a price first",
                spec.model
            ));
        }
    }
    if !detach {
        let exit = detached_runs::run_child(xencode_dir, run_id).await;
        print_detached_exit(run_id, &exit);
        return Ok(());
    }
    let tool_root = PathBuf::from(&spec.tool_root);
    let exe = std::env::current_exe().map_err(|e| format!("cannot re-run this binary: {e}"))?;
    let pid = detached_runs::spawn_child(
        &exe,
        run_id,
        xencode_dir,
        &tool_root,
        &detached_runs::log_path(&dir),
    )
    .map_err(|e| format!("could not fork the run: {e}"))?;
    let _ = detached_runs::write_pid(&dir, pid);
    println!("resumed detached run {run_id} (pid {pid}) from {used_rounds} completed rounds");
    Ok(())
}

fn print_detached_exit(run_id: &str, exit: &xencode_tui_rs::detached::DetachedExit) {
    use xencode_tui_rs::detached::ExitReason;
    let why = match &exit.reason {
        ExitReason::Done => "done".to_string(),
        ExitReason::RoundCap => format!("stopped: spent its {} rounds", exit.rounds),
        ExitReason::WallCap => "stopped: spent its wall-clock budget".to_string(),
        ExitReason::CostCap => "stopped: spent its cost budget".to_string(),
        ExitReason::Stopped => "stopped on request".to_string(),
        ExitReason::Error => format!("could not run: {}", exit.note),
    };
    println!(
        "run {run_id}: {why} ({} {})",
        exit.rounds,
        count_word(exit.rounds as usize, "round", "rounds")
    );
}

fn list_detached_runs(xencode_dir: &std::path::Path) -> Result<(), String> {
    let ids = detached_runs::list_run_ids(xencode_dir);
    if ids.is_empty() {
        println!(
            "no detached runs in {}",
            detached_runs::detached_dir(xencode_dir).display()
        );
        println!("start one with `xencode run \"the task\" --detach`.");
        return Ok(());
    }
    for id in ids {
        let dir = detached_runs::run_dir(xencode_dir, &id);
        let status = detached_runs::derive_status(&dir);
        let rounds = detached_runs::read_rounds(&dir).len();
        let model = detached_runs::read_spec(&dir)
            .map(|spec| spec.model)
            .unwrap_or_else(|| "?".to_string());
        println!("{id}  {}  {rounds} rounds  {model}", status.label());
    }
    Ok(())
}

fn show_detached_run(xencode_dir: &std::path::Path, given: &str) -> Result<(), String> {
    let run_id = detached_runs::resolve_run_id(xencode_dir, given)
        .ok_or_else(|| format!("no detached run {given}"))?;
    let dir = detached_runs::run_dir(xencode_dir, &run_id);
    let spec = detached_runs::read_spec(&dir).ok_or_else(|| format!("no detached run {run_id}"))?;
    let status = detached_runs::derive_status(&dir);
    let rounds = detached_runs::read_rounds(&dir);
    println!("run {run_id}");
    println!("  status: {}", status.label());
    println!("  model: {}", spec.model);
    println!("  prompt: {}", spec.prompt);
    println!("  tree: {}", spec.tool_root);
    println!(
        "  caps: {} rounds, {} minutes wall-clock{}",
        spec.caps.max_rounds,
        spec.caps.max_wall_ms as f64 / 60_000.0,
        spec.caps
            .max_cost_micros
            .map(|micros| format!(", ${:.2} cost", micros as f64 / 1_000_000.0))
            .unwrap_or_default(),
    );
    println!(
        "  approvals: {}",
        if spec.allow_shell {
            "shell pre-approved (--allow-shell)"
        } else {
            "edits pre-approved; shell refused unasked"
        }
    );
    if rounds.is_empty() {
        println!("  rounds: none completed");
    } else {
        println!("  rounds:");
        for row in &rounds {
            println!(
                "    {:>3}  {} new turns  {} prompt + {} completion tokens",
                row.round,
                row.turns.len(),
                row.prompt_tokens
                    .map(|tokens| tokens.to_string())
                    .as_deref()
                    .unwrap_or("?"),
                row.completion_tokens
                    .map(|tokens| tokens.to_string())
                    .as_deref()
                    .unwrap_or("?"),
            );
        }
    }
    if let xencode_tui_rs::detached::DetachedStatus::Finished(exit) = &status {
        println!("  exit: {:?} after {} rounds", exit.reason, exit.rounds);
        if !exit.note.is_empty() {
            println!("  why: {}", exit.note);
        }
    }
    Ok(())
}

fn log_detached_run(xencode_dir: &std::path::Path, given: &str, tail: usize) -> Result<(), String> {
    let run_id = detached_runs::resolve_run_id(xencode_dir, given)
        .ok_or_else(|| format!("no detached run {given}"))?;
    let dir = detached_runs::run_dir(xencode_dir, &run_id);
    if detached_runs::read_spec(&dir).is_none() {
        return Err(format!("no detached run {run_id}"));
    }
    for line in detached_runs::read_log_tail(&dir, tail) {
        println!("{line}");
    }
    Ok(())
}

fn stop_detached_run(xencode_dir: &std::path::Path, given: &str) -> Result<(), String> {
    let run_id = detached_runs::resolve_run_id(xencode_dir, given)
        .ok_or_else(|| format!("no detached run {given}"))?;
    let dir = detached_runs::run_dir(xencode_dir, &run_id);
    match detached_runs::derive_status(&dir) {
        xencode_tui_rs::detached::DetachedStatus::Missing => {
            Err(format!("no detached run {run_id}"))
        }
        xencode_tui_rs::detached::DetachedStatus::Running { pid } => {
            println!("{}", detached_runs::stop_child(&dir, pid));
            Ok(())
        }
        status => Err(format!(
            "run {run_id} is {}, not running; only a running run can be stopped",
            status.label()
        )),
    }
}

async fn run_replay(
    run_id: Option<String>,
    list: bool,
    run_tools: bool,
    tool_root: Option<PathBuf>,
    out: Option<PathBuf>,
) -> Result<(), String> {
    let xencode_dir = project_xencode_dir();
    if list {
        return list_recordings(&xencode_dir);
    }
    let given = run_id.ok_or_else(|| {
        "name the run to replay, or pass --list to see what has been recorded".to_string()
    })?;
    let resolved = xencode_context_rs::resolve_run_id(&xencode_dir, &given)?;
    let options = xencode_tui_rs::replay::ReplayOptions {
        run_id: resolved.clone(),
        xencode_dir: xencode_dir.clone(),
        tool_root: tool_root.unwrap_or_else(xencode_context_rs::default_root),
        out_dir: out.unwrap_or_else(|| {
            xencode_dir
                .join("cache")
                .join("replays")
                .join(resolved.clone())
        }),
        run_tools,
    };
    let report = xencode_tui_rs::replay::replay(&options).await?;
    for line in report.lines() {
        println!("{line}");
    }
    if report.matches() {
        Ok(())
    } else {
        Err(format!(
            "the replay of {} did not reproduce the recording",
            report.run_id
        ))
    }
}

/// How long each structural query may take. Per rule, not per scan, so a slow
/// tree is bounded by rules × this rather than not at all.
const TIMED_OUT: std::time::Duration = std::time::Duration::from_secs(30);

/// `--runtime` short-circuits the rest of `analyze`: it is a different question
/// with a different engine, and folding it into the image/security inventory
/// would make one flag mean two reports.
fn run_runtime_analyze(path: &std::path::Path, format: OutputFormat) -> Result<(), String> {
    let scan = xencode_analysis_rs::runtime_hazards::analyze_runtime(path, TIMED_OUT);
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&scan)
                    .map_err(|e| format!("cannot render JSON: {e}"))?
            );
        }
        _ => {
            println!("{}", scan.summary());
            for finding in &scan.findings {
                println!("\n{}", finding.to_report());
            }
        }
    }
    match scan.engine {
        xencode_analysis_rs::runtime_hazards::EngineStatus::Ran { .. } => Ok(()),
        xencode_analysis_rs::runtime_hazards::EngineStatus::Unavailable { .. } => {
            Err("the runtime hazard check did not run, so this is not a clean result".to_string())
        }
    }
}

fn run_analyze(
    path: std::path::PathBuf,
    format: OutputFormat,
    runtime: bool,
) -> Result<(), String> {
    if runtime {
        return run_runtime_analyze(&path, format);
    }
    if path.is_dir() {
        // Full-tree walk: no depth cap (an explicit user action), junk dirs
        // (target/, node_modules/, .git/…) skipped like `scan`, and every
        // failure reported on stderr instead of silently dropped.
        let mut all_issue_lists = Vec::new();
        let mut image_metas: Vec<ImageMeta> = Vec::new();
        let mut skipped = 0usize;
        let excluded = ScanOptions::default().excluded_dirs;
        let walker = walkdir::WalkDir::new(&path).into_iter();

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("  Skipping unreadable entry: {e}");
                    skipped += 1;
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let fp = entry.path();
            let in_junk = fp.components().any(|c| match c {
                std::path::Component::Normal(name) => {
                    excluded.iter().any(|x| name == std::ffi::OsStr::new(x))
                }
                _ => false,
            });
            if in_junk {
                continue;
            }
            if is_image_path(fp) {
                match analyze_image(fp) {
                    Ok(meta) => image_metas.push(meta),
                    Err(e) => {
                        eprintln!("  Skipping {}: {}", fp.display(), e);
                        skipped += 1;
                    }
                }
                continue;
            }
            // Every other file goes through analysis: unknown extensions
            // fall back to generic checks; unreadable (binary) files count
            // as skipped, visibly.
            match CodeAnalyzer::analyze_file(fp) {
                Ok(issues) => {
                    all_issue_lists.push((fp.display().to_string(), issues));
                }
                Err(e) => {
                    eprintln!("  Skipping {}: {}", fp.display(), e);
                    skipped += 1;
                    continue;
                }
            }
            // Run security scan (stderr: stdout stays valid JSON).
            if let Ok(findings) = VulnerabilityScanner::scan_file(fp) {
                if !findings.is_empty() {
                    // A pattern scan, not a security verdict (EVd-3): these are
                    // lines that match a rule, for a person to read.
                    eprintln!(
                        "  Pattern scan: {} possible issue(s) in {}",
                        findings.len(),
                        fp.display()
                    );
                }
            }
        }

        match format {
            OutputFormat::Json => {
                // Documented dir schema: issues pairs, image inventory, and
                // the skip count. Single-file shapes below are unchanged.
                let output = serde_json::json!({
                    "issues": all_issue_lists,
                    "images": image_metas,
                    "skipped": skipped,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?
                );
            }
            OutputFormat::Text => {
                let total: usize = all_issue_lists.iter().map(|(_, issues)| issues.len()).sum();
                println!("Results for: {}", path.display());
                println!("   Files analyzed: {}", all_issue_lists.len());
                println!("   Total issues:   {}", total);
                println!("   Images:         {}", image_metas.len());
                println!("   Skipped:        {}", skipped);
                for meta in &image_metas {
                    println!("\n  {}", format_image_text(meta));
                }
                for (file_path, issues) in &all_issue_lists {
                    if !issues.is_empty() {
                        println!(
                            "
  {} ({} issues)",
                            file_path,
                            issues.len()
                        );
                        for issue in issues {
                            let icon = match issue.severity.label() {
                                "critical" | "high" => "R",
                                "medium" => "Y",
                                _ => "G",
                            };
                            println!(
                                "    {} [{}] Ln{}: {} -- {}",
                                icon,
                                issue.severity.label(),
                                issue.line_number,
                                issue.message,
                                issue.suggestion
                            );
                        }
                    }
                }
            }
        }
    } else {
        // Single file — images take the intake path (metadata, not issues).
        if is_image_path(&path) {
            let meta = analyze_image(&path).map_err(|e| e.to_string())?;
            match format {
                OutputFormat::Json => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?
                    );
                }
                OutputFormat::Text => {
                    println!("Image: {}", path.display());
                    println!("   {}", format_image_text(&meta));
                }
            }
            return Ok(());
        }
        let issues = CodeAnalyzer::analyze_file(&path).map_err(|e| e.to_string())?;

        match format {
            OutputFormat::Json => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&issues).map_err(|e| e.to_string())?
                );
            }
            OutputFormat::Text => {
                println!("Analysis of: {}", path.display());
                println!("   Issues: {}", issues.len());
                for issue in &issues {
                    println!(
                        "  [{}] Ln{}: {} -- {}",
                        issue.severity.label(),
                        issue.line_number,
                        issue.message,
                        issue.suggestion
                    );
                }
            }
        }
    }
    Ok(())
}

/// Show what a plugin puts in front of every agent turn, line by line, with the
/// prompt text itself visible rather than counted.
fn render_declaration(report: &xencode_plugin_rs::LoadReport) -> String {
    let mut out = format!(
        "  {} v{} — {}.\n",
        report.name,
        report.version,
        if report.loaded {
            "loaded"
        } else {
            "NOT LOADED"
        }
    );
    if let Some(origin) = report.source.as_ref().and_then(|s| s.summary()) {
        out.push_str(&format!("      from {origin}\n"));
    }
    if !report.loaded {
        out.push_str(&format!(
            "      it contributes nothing: {}\n",
            report
                .reason
                .clone()
                .unwrap_or_else(|| "unknown".to_string())
        ));
        return out;
    }
    if report.prompt_text.is_empty() {
        out.push_str("      it contributes no prompt text.\n");
    } else {
        out.push_str(&format!(
            "      it puts {} line(s) ahead of the agent's system prompt on every turn:\n",
            report.prompt_text.lines().count()
        ));
        for line in report.prompt_text.lines() {
            out.push_str(&format!("      | {line}\n"));
        }
    }
    if report.before_hooks + report.after_hooks > 0 {
        out.push_str(&format!(
            "      it declares {} before hook(s) and {} after hook(s), each of which runs a \
             shell command in the workspace.\n",
            report.before_hooks, report.after_hooks
        ));
    }
    out
}

/// "abc1234def5678… (7-character prefix abc1234)" for a full SHA, plain for a
/// message that is not one.
fn format_commit(commit: &str) -> String {
    if commit.len() > 7 && commit.chars().all(|c| c.is_ascii_hexdigit()) {
        format!("{}…", xencode_plugin_rs::short_commit(commit))
    } else {
        commit.to_string()
    }
}

/// `xencode mcp serve` (M-5): xencode as the server, on a pipe.
async fn run_mcp(action: McpAction) -> Result<(), String> {
    use xencode_tui_rs::agent_tools::{new_task_runtime, HeadlessPolicy};
    use xencode_tui_rs::mcp_serve;

    match action {
        McpAction::Serve {
            workspace,
            allow,
            team,
        } => {
            let root = std::fs::canonicalize(&workspace).map_err(|error| {
                format!(
                    "workspace {} cannot be opened: {error}",
                    workspace.display()
                )
            })?;
            // A flag naming nothing is a typo, and a typo accepted quietly leaves
            // the operator believing a tool is permitted when it is not.
            let published = mcp_serve::exposed_tool_names();
            for name in &allow {
                if !published.contains(name) {
                    return Err(format!(
                        "`--allow {name}` names no tool xencode publishes; the six are {}",
                        published.join(", ")
                    ));
                }
            }
            let policy = HeadlessPolicy::new(allow.clone());
            // stdout carries the protocol, so anything meant for the person who
            // started the server has to go to stderr.
            if policy.is_read_only() {
                eprintln!(
                    "xencode serving {} read-only over stdio; a call that would write or \
                     run a command is refused. Restart with --allow <tool> to permit one.",
                    root.display()
                );
            } else {
                eprintln!(
                    "xencode serving {} over stdio, permitting {} in addition to reads. \
                     A `path` or `cwd` that leaves that directory stays refused.",
                    root.display(),
                    allow.join(", ")
                );
                // The boundary is checked on the arguments, not inside a shell
                // command, so an allowed `run_command` reaches wherever this
                // user's shell can — and nobody is on the pipe to approve it.
                if allow.iter().any(|name| name == "run_command") {
                    eprintln!(
                        "warning: `run_command` is permitted. A permitted command runs \
                         exactly as the caller wrote it, so it can touch files outside \
                         {}; that is a shell, not a jailed one, and there is no approval \
                         prompt on a pipe.",
                        root.display()
                    );
                }
            }
            mcp_serve::serve(
                root,
                env!("CARGO_PKG_VERSION"),
                policy,
                new_task_runtime(),
                team,
            )
            .await
        }
    }
}

fn run_plugin_action(action: PluginAction) -> Result<(), String> {
    let plugin_dir = default_plugin_dir();
    let version = env!("CARGO_PKG_VERSION");

    match action {
        PluginAction::List => {
            // Not a listing of the plugin directory: this is the same load the
            // TUI performs at startup, so "installed" and "took hold" cannot
            // drift apart.
            let runtime = PluginRuntime::load(&plugin_dir, version);
            if runtime.reports().is_empty() {
                println!("No plugins installed in: {}", plugin_dir.display());
                println!("Use 'xencode plugin install <git-url|path>' to install a plugin.");
            } else {
                println!(
                    "📦 Plugins in {} (xencode {version}):",
                    plugin_dir.display()
                );
                for report in runtime.reports() {
                    print!("{}", render_declaration(report));
                }
                println!(
                    "  {} of {} loaded — a loaded plugin's prompt prefix and hooks apply to every agent turn.",
                    runtime.loaded_count(),
                    runtime.reports().len()
                );
            }
        }
        PluginAction::Install { source, rev } => {
            // Clone or read, verify the manifest and show what it declares —
            // before a single file lands where the loader will scan it.
            let pending = if xencode_plugin_rs::is_git_source(&source) {
                xencode_plugin_rs::prepare_git(&plugin_dir, &source, rev.as_deref(), version)
            } else {
                xencode_plugin_rs::prepare_path(&plugin_dir, std::path::Path::new(&source), version)
            }
            .map_err(|e| e.to_string())?;
            println!("{}", pending.declaration());
            let outcome = pending.apply().map_err(|e| e.to_string())?;
            println!(
                "\n✅ Installed {} v{} into {}.",
                outcome.name,
                outcome.version,
                outcome.destination.display()
            );
            match (
                outcome.source.url.as_deref(),
                outcome.source.commit.as_deref(),
            ) {
                // An install that named no commit still pinned one: say which,
                // so what is installed can be named exactly later.
                (Some(_), Some(commit)) => println!(
                    "   Pinned to commit {shown}{rest}. Nothing more is fetched until \
                     `xencode plugin update {name}` is run.",
                    shown = format_commit(commit),
                    rest = match outcome.source.git_ref.as_deref() {
                        Some(rev) if rev != commit => format!(" (installed from {rev})"),
                        _ => String::new(),
                    },
                    name = outcome.name
                ),
                _ => println!(
                    "   Copied from a local path, so no commit is pinned. A later install of the \
                     same name asks you to remove it first."
                ),
            }
        }
        PluginAction::Update { name, rev, yes } => {
            let plan =
                xencode_plugin_rs::prepare_update(&plugin_dir, &name, rev.as_deref(), version)
                    .map_err(|e| e.to_string())?;
            println!(
                "{}: {} from {} at {}",
                if plan.current {
                    "Already up to date"
                } else {
                    "Compared"
                },
                name,
                plan.url,
                format_commit(&plan.to_commit)
            );
            if plan.current {
                println!(
                    "   Installed copy is {} at {} and declares the same thing.",
                    plan.from_version,
                    format_commit(&plan.to_commit)
                );
                return Ok(());
            }
            for change in &plan.changes {
                println!("   · {change}");
            }
            if !plan.diff.is_empty() {
                println!("   Its manifest differs:");
                for line in plan.diff.lines() {
                    println!("   {line}");
                }
            }
            if plan.reaches_agent && !yes {
                println!(
                    "   NOT APPLIED: {} still v{} at {}. The version above changes what this \
                     plugin puts in front of the agent on every turn, so read the diff and \
                     re-run with --yes to install it.",
                    name,
                    plan.from_version,
                    format_commit(plan.from.commit.as_deref().unwrap_or("no recorded commit"))
                );
                return Ok(());
            }
            let (from, to) = (plan.from_version.clone(), plan.to_version.clone());
            let source = plan.apply().map_err(|e| e.to_string())?;
            println!(
                "   ✅ Applied: v{from} → v{to}, now pinned to {}. Run `/plugin reload` in a \
                 running xencode, or start a new session, for it to take hold.",
                format_commit(source.commit.as_deref().unwrap_or("no recorded commit")),
            );
        }
        PluginAction::Remove { name } => {
            // The name comes from the command line, so it passes the same guard
            // a manifest name does: one normal path component, nothing that can
            // walk out of the plugin directory.
            let path = PluginRegistry::new(plugin_dir.clone())
                .plugin_path(&name)
                .ok_or_else(|| format!("Invalid plugin name: {name}"))?;
            if !path.exists() {
                return Err(format!("Plugin '{name}' not found"));
            }
            // Say what is being taken away, so a removal can be checked against
            // the intention — including which commit the copy came from.
            let runtime = PluginRuntime::load(&plugin_dir, version);
            let contributions: String = runtime
                .reports()
                .iter()
                .filter(|report| report.name == name)
                .map(render_declaration)
                .collect();
            std::fs::remove_dir_all(&path).map_err(|e| format!("Failed to remove: {}", e))?;
            println!("✅ Plugin '{name}' removed.");
            let lines: Vec<&str> = contributions
                .lines()
                .map(str::trim_start)
                .filter(|line| !line.is_empty())
                .collect();
            if !lines.is_empty() {
                println!("   It was contributing:");
                for line in lines {
                    println!("     {line}");
                }
                println!(
                    "   A session already running still has the old prompt until /plugin reload."
                );
            }
        }
    }
    Ok(())
}

/// What to say when the interactive screen is asked for somewhere it cannot be
/// drawn. This used to be the terminal library's `ENXIO` — "No such device or
/// address" — which named neither the terminal nor a command that would have
/// worked in its place.
const NO_TERMINAL: &str = concat!(
    "the interactive screen needs a terminal to draw on, and standard output here is not one ",
    "(a pipe, a redirect, a cron line or a CI step). Without a terminal these work: ",
    "`xencode query <prompt>` for one answer, `xencode run <task>` for an agent turn, ",
    "`xencode scan`, `xencode analyze`, `xencode doctor`. `xencode --help` lists the rest."
);

async fn run_engine(project: Option<PathBuf>, wait_limit: u64) -> Result<(), String> {
    let project = match project {
        Some(project) => project,
        None => {
            std::env::current_dir().map_err(|e| format!("cannot read the current folder: {e}"))?
        }
    };
    xencode_tui_rs::engine::server::serve(project, std::time::Duration::from_secs(wait_limit))
        .await?;
    // The model list and health checks may still have work in flight that
    // nobody will read; the engine is finished, so the process ends now
    // rather than waiting for them.
    std::process::exit(0);
}

async fn run_tui(in_process: bool) -> Result<(), String> {
    use std::io::IsTerminal;
    // A panic from here on would otherwise leave raw mode and the alternate
    // screen switched on, hiding its own message: restore the terminal first
    // and record the crash where `xencode doctor` can find it.
    if let Some(record) = xencode_tui_rs::panic::default_record_path() {
        xencode_tui_rs::panic::install_panic_hook(record);
    }
    if !io::stdout().is_terminal() {
        return Err(NO_TERMINAL.to_string());
    }
    crossterm::terminal::enable_raw_mode().map_err(|e| {
        format!("the interactive screen could not take over the terminal ({e}). {NO_TERMINAL}")
    })?;
    let mut stdout = io::stdout();
    // Mouse capture is deliberately absent here: the app turns it on from the
    // `mouse_capture` setting on its first frame, so a user who handed the
    // mouse back to the terminal never gets it taken (`V-7`). The cleanup below
    // still disables it, which is correct whether or not it was ever enabled.
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)
        .map_err(|e| e.to_string())?;

    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend).map_err(|e| e.to_string())?;

    let res = xencode_tui_rs::run_app(&mut terminal, in_process).await;

    // Restore terminal
    crossterm::terminal::disable_raw_mode().map_err(|e| e.to_string())?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture
    )
    .map_err(|e| e.to_string())?;
    terminal.show_cursor().map_err(|e| e.to_string())?;

    res.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        attach_images_to_final_user_message, compute_advise, doc, format_image_text,
        human_duration, join, parse_comma_list, plural_count, resolve_audit_path, resolve_bind,
        Cli, Commands, GenerateShell, OutputFormat, PerfAction, PriceAction, RunsAction,
        SessionAction,
    };
    use clap::Parser;
    use xencode_analysis_rs::images::{ImageFormat, ImageMeta};
    use xencode_providers_rs::{ChatMessage, ContentPart, MessageContent};

    /// Minimal standard-alphabet base64 decoder, so a test can check what the
    /// encoder put on the wire without adding a dependency for one assertion.
    fn decode_base64(encoded: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(encoded.len() / 4 * 3);
        let mut acc: u32 = 0;
        let mut bits = 0;
        for c in encoded.bytes() {
            let value = match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => break,
                b'\r' | b'\n' => continue,
                other => panic!("unexpected byte {other:#x} in base64"),
            } as u32;
            acc = (acc << 6) | value;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((acc >> bits) as u8);
            }
        }
        out
    }

    /// A 1x1 opaque PNG — the smallest file that is genuinely an image, so the
    /// intake path is exercised against real bytes.
    const ONE_PIXEL_PNG: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
        0x77, 0x53, 0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60,
        0xf8, 0xcf, 0xc0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xdd, 0x8d, 0xb0, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];

    /// The three repair shapes the mutation run found survive, decided by the
    /// pure gate so each one is pinned by a test rather than by a re-run.
    fn repair_input(
        patch: &str,
        before: usize,
        after: usize,
        rerun_verdict: Option<&str>,
    ) -> String {
        let mutants = serde_json::json!([{
            "file": "src/lib.rs",
            "name": "src/lib.rs:2:5: replace is_even -> bool with true",
        }]);
        let mut value = serde_json::json!({
            "patch": patch,
            "assertions_before": { "tests/lib.rs": before },
            "assertions_after": { "tests/lib.rs": after },
            "targeted": mutants,
        });
        if let Some(verdict) = rerun_verdict {
            value["rerun"] = serde_json::json!({
                "mutants": [{
                    "file": "src/lib.rs",
                    "name": "src/lib.rs:2:5: replace is_even -> bool with true",
                    "verdict": verdict,
                }],
            });
        }
        serde_json::to_string_pretty(&value).unwrap()
    }

    fn judge(text: &str) -> Result<(), String> {
        let path = std::env::temp_dir().join(format!(
            "xe-repair-{}-{:?}.json",
            std::process::id(),
            text.len()
        ));
        std::fs::write(&path, text).unwrap();
        let out = super::judge_repair(path.clone(), super::OutputFormat::Text);
        let _ = std::fs::remove_file(&path);
        out
    }

    #[test]
    fn a_tautological_repair_is_rejected() {
        // The measured fake fix: the negative assertion is replaced by
        // `x || !x`, which is true for every value, and the count goes *up*.
        let text = repair_input(
            "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -4,1 +4,2 @@\n-    assert!(!is_even(3));\n+    assert!(x || !x);\n",
            2, 3,
            Some("missed"),
        );
        let err = judge(&text).unwrap_err();
        assert!(err.contains("rejected"), "{err}");
    }

    #[test]
    fn a_repair_that_edits_the_code_under_mutation_is_rejected() {
        let text = repair_input(
            "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -2,1 +2,1 @@\n-    n % 2 == 0\n+    true\n",
            1,
            4,
            Some("caught"),
        );
        assert!(judge(&text).is_err());
    }

    #[test]
    fn a_repair_with_no_rerun_is_rejected() {
        let text = repair_input(
            "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -4,0 +5 @@\n+    assert!(!is_even(3));\n",
            1,
            2,
            None,
        );
        assert!(judge(&text).is_err());
    }

    #[test]
    fn a_real_repair_is_accepted() {
        let text = repair_input(
            "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -4,0 +5 @@\n+    assert!(!is_even(3));\n",
            1,
            2,
            Some("caught"),
        );
        assert!(judge(&text).is_ok());
    }

    fn generated_artifact(kind: &str, shell: Option<GenerateShell>) -> String {
        use clap::CommandFactory;
        let mut cmd = super::Cli::command();
        let mut out = Vec::new();
        match kind {
            "man" => {
                clap_mangen::Man::new(cmd).render(&mut out).unwrap();
            }
            _ => {
                let shell = match shell {
                    Some(GenerateShell::Fish) => clap_complete::Shell::Fish,
                    Some(GenerateShell::Zsh) => clap_complete::Shell::Zsh,
                    _ => clap_complete::Shell::Bash,
                };
                clap_complete::generate(shell, &mut cmd, "xencode", &mut out);
            }
        }
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn the_committed_completions_are_generated_never_hand_written() {
        // The WF-6 trap is drift: these files must equal what clap emits now.
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../docs");
        for (file, kind, shell) in [
            (
                "completions/xencode.fish",
                "completions",
                Some(GenerateShell::Fish),
            ),
            (
                "completions/xencode.zsh",
                "completions",
                Some(GenerateShell::Zsh),
            ),
            (
                "completions/xencode.bash",
                "completions",
                Some(GenerateShell::Bash),
            ),
            ("man/xencode.1", "man", None),
        ] {
            let committed = std::fs::read_to_string(root.join(file))
                .unwrap_or_else(|_| panic!("missing {file}"));
            assert_eq!(
                generated_artifact(kind, shell),
                committed,
                "{file} drifted from what clap generates — regenerate it, do not edit it"
            );
        }
    }

    fn repo_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    #[test]
    fn the_release_pipeline_is_dist_generated_not_hand_written() {
        // WF-5's trap is keys in CI and drift in the release files. The workflow
        // must be dist's output, and must reference no secret but the automatic
        // token — anything else is a key waiting to leak.
        let workflow = std::fs::read_to_string(repo_root().join(".github/workflows/release.yml"))
            .expect("release workflow");
        assert!(
            workflow.contains("autogenerated by dist"),
            "release.yml must be dist's output, not a hand edit"
        );
        for line in workflow.lines() {
            if line.contains("secrets.") {
                assert!(
                    line.contains("secrets.GITHUB_TOKEN"),
                    "only the automatic token may appear: {line}"
                );
            }
        }
        assert!(!workflow.contains("GPG"), "no signing keys in CI");
    }

    #[test]
    fn dist_config_names_targets_installers_and_repository() {
        let config = std::fs::read_to_string(repo_root().join("rust/dist-workspace.toml"))
            .expect("dist config");
        for key in ["cargo-dist-version", "targets", "installers", "hosting"] {
            assert!(config.contains(key), "dist config lacks {key}");
        }
        assert!(config.contains("shell"), "no shell installer");
        let manifest =
            std::fs::read_to_string(repo_root().join("rust/crates/xencode-cli/Cargo.toml"))
                .expect("cli manifest");
        assert!(
            manifest.contains("repository = \"https://github.com/"),
            "dist needs a repository URL and will not guess one"
        );
    }

    #[test]
    fn completions_name_real_subcommands() {
        let fish = generated_artifact("completions", Some(GenerateShell::Fish));
        for subcommand in [
            "toolchain",
            "test",
            "cov",
            "mutants",
            "anchor",
            "generate",
            "session",
            "envcheck",
            "paths",
            "migrate",
        ] {
            assert!(fish.contains(subcommand), "completions omit {subcommand}");
        }
    }

    #[test]
    fn agents_contract_parses() {
        let cli = Cli::try_parse_from(["xencode", "agents", "--contract"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Agents { contract: true, .. })
        ));
    }

    #[test]
    fn agents_health_parses() {
        let cli =
            Cli::try_parse_from(["xencode", "agents", "--health", "--agent", "claude"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Agents {
                health: true,
                agent: Some(ref a),
                ..
            }) if a == "claude"
        ));
    }

    #[test]
    fn agents_parses() {
        let cli = Cli::try_parse_from(["xencode", "agents"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Agents { .. })));
    }

    #[test]
    fn agents_package_and_resume_parse() {
        let cli =
            Cli::try_parse_from(["xencode", "agents", "--build-package", "task-123"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Agents {
                build_package: Some(ref t),
                ..
            }) if t == "task-123"
        ));

        let cli = Cli::try_parse_from([
            "xencode",
            "agents",
            "--package",
            "pkg.json",
            "--resume",
            "--agent",
            "claude",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Agents {
                package: Some(ref p),
                resume: true,
                agent: Some(ref a),
                ..
            }) if p.to_str() == Some("pkg.json") && a == "claude"
        ));
    }

    #[test]
    fn agents_route_parses() {
        let cli = Cli::try_parse_from([
            "xencode",
            "agents",
            "--route",
            "task-456",
            "--require-cap",
            "acp",
            "--require-cap",
            "stream",
            "--max-cost",
            "1.50",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Agents {
                route_task: Some(ref t),
                ref require_caps,
                max_cost: Some(cost),
                ..
            }) => {
                assert_eq!(t, "task-456");
                assert_eq!(require_caps, &["acp", "stream"]);
                assert_eq!(cost, 1.50);
            }
            _ => panic!("expected agents --route"),
        }
    }

    #[test]
    fn agents_redispatch_parses() {
        let cli = Cli::try_parse_from([
            "xencode",
            "agents",
            "--redispatch",
            "task-789",
            "--agent",
            "initial-worker",
            "--replacement-agent",
            "second-worker",
            "--stop-reason",
            "signal:9",
            "--test-cmd",
            "cargo test",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Agents {
                redispatch_task: Some(ref t),
                agent: Some(ref a),
                replacement_agent: Some(ref rep),
                stop_reason: Some(ref s),
                ref test_cmds,
                ..
            }) => {
                assert_eq!(t, "task-789");
                assert_eq!(a, "initial-worker");
                assert_eq!(rep, "second-worker");
                assert_eq!(s, "signal:9");
                assert_eq!(test_cmds, &["cargo test"]);
            }
            _ => panic!("expected agents --redispatch"),
        }
    }

    #[test]
    fn paths_and_migrate_parse() {
        let cli = Cli::try_parse_from(["xencode", "paths"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Paths { .. })));
        let cli = Cli::try_parse_from(["xencode", "migrate"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Migrate { dry_run: false })
        ));
        let cli = Cli::try_parse_from(["xencode", "migrate", "--dry-run"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Migrate { dry_run: true })
        ));
    }

    #[test]
    fn hotspots_parses_with_limit() {
        let cli = Cli::try_parse_from(["xencode", "hotspots"]).unwrap();
        match cli.command {
            Some(Commands::Hotspots { limit, .. }) => assert_eq!(limit, 10),
            _ => panic!("expected hotspots"),
        }
        let cli = Cli::try_parse_from(["xencode", "hotspots", "--limit", "3"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Hotspots { limit: 3, .. })
        ));
    }

    #[test]
    fn impact_parses_a_file_and_optional_limit() {
        // The file is positional; without it the command must not parse.
        assert!(Cli::try_parse_from(["xencode", "impact"]).is_err());
        let cli = Cli::try_parse_from(["xencode", "impact", "src/lib.rs"]).unwrap();
        match cli.command {
            Some(Commands::Impact { file, limit, .. }) => {
                assert_eq!(file, "src/lib.rs");
                assert_eq!(limit, 15, "default section size");
            }
            _ => panic!("expected impact"),
        }
        let cli = Cli::try_parse_from([
            "xencode",
            "impact",
            "src/lib.rs",
            "--limit",
            "4",
            "--format",
            "json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Impact { file, limit: 4, format: super::OutputFormat::Json, .. })
                if file == "src/lib.rs"
        ));
    }

    #[test]
    fn removal_parses_a_file_and_optional_limit() {
        // The file is positional; without it the command must not parse.
        assert!(Cli::try_parse_from(["xencode", "removal"]).is_err());
        let cli = Cli::try_parse_from(["xencode", "removal", "src/leaf.rs"]).unwrap();
        match cli.command {
            Some(Commands::Removal { file, limit, .. }) => {
                assert_eq!(file, "src/leaf.rs");
                assert_eq!(limit, 15, "default section size");
            }
            _ => panic!("expected removal"),
        }
        let cli =
            Cli::try_parse_from(["xencode", "removal", "src/leaf.rs", "--format", "json"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Removal { file, limit: 15, format: super::OutputFormat::Json, .. })
                if file == "src/leaf.rs"
        ));
    }

    #[test]
    fn doctor_selfcheck_parses() {
        let cli = Cli::try_parse_from(["xencode", "doctor", "--selfcheck"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Doctor {
                selfcheck: true,
                ..
            })
        ));
    }

    #[test]
    fn doctor_deps_parses() {
        let cli = Cli::try_parse_from(["xencode", "doctor", "--deps"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Doctor { deps: true, .. })
        ));
    }

    #[test]
    fn doctor_env_parses() {
        let cli = Cli::try_parse_from(["xencode", "doctor", "--env"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Doctor { env: true, .. })
        ));
        let cli = Cli::try_parse_from(["xencode", "doctor"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Doctor { env: false, .. })
        ));
    }

    #[test]
    fn session_actions_parse() {
        let cli = Cli::try_parse_from(["xencode", "session", "resolve", "latest"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Session {
                action: SessionAction::Resolve { .. }
            })
        ));
        let cli =
            Cli::try_parse_from(["xencode", "session", "export", "demo", "--redacted"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Session {
                action: SessionAction::Export { .. }
            })
        ));
        let cli = Cli::try_parse_from(["xencode", "session", "name", "abc123", "demo"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Session {
                action: SessionAction::Name { .. }
            })
        ));
    }

    #[test]
    fn envcheck_parses_with_optional_format() {
        let cli = Cli::try_parse_from(["xencode", "envcheck"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Envcheck { .. })));
        let cli = Cli::try_parse_from(["xencode", "envcheck", "--format", "json"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Envcheck { .. })));
    }

    #[test]
    fn line_ranges_read_as_ranges() {
        assert_eq!(join(&[1, 2, 3, 7, 9, 10]), "1-3, 7, 9-10");
        assert_eq!(join(&[5]), "5");
        assert_eq!(join(&[]), "");
    }

    #[test]
    fn the_new_verification_subcommands_parse() {
        // These are the arms `xencode cov` coverage found unexercised, so
        // parsing them is the cheapest honest check that the wiring is right.
        let cli = Cli::try_parse_from(["xencode", "cov", "--base", "main"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Cov { base: Some(_), .. })
        ));

        let cli = Cli::try_parse_from(["xencode", "cov", "--show-missing-lines"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Cov {
                show_missing_lines: true,
                ..
            })
        ));

        // Retries default to 0, and a retry-pass is not a pass.
        let cli = Cli::try_parse_from(["xencode", "test"]).unwrap();
        match cli.command {
            Some(Commands::Test {
                retries,
                stress_count,
                packages,
                isolate,
                base,
                ..
            }) => {
                assert_eq!(retries, 0, "a retry must be asked for, not assumed");
                assert_eq!(stress_count, 0);
                assert!(packages.is_empty());
                assert!(isolate.is_none());
                assert_eq!(base, "HEAD");
            }
            _ => panic!("expected the test subcommand"),
        }

        let cli = Cli::try_parse_from([
            "xencode",
            "test",
            "--isolate",
            "old_broken",
            "--base",
            "main",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Test {
                isolate,
                base,
                repeat,
                ..
            }) => {
                assert_eq!(isolate.as_deref(), Some("old_broken"));
                assert_eq!(base, "main");
                assert_eq!(repeat, 3);
            }
            _ => panic!("expected the test subcommand"),
        }

        let cli = Cli::try_parse_from([
            "xencode",
            "test",
            "--retries",
            "3",
            "--package",
            "a",
            "--package",
            "b",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Test {
                retries, packages, ..
            }) => {
                assert_eq!(retries, 3);
                assert_eq!(packages, vec!["a".to_string(), "b".to_string()]);
            }
            _ => panic!("expected the test subcommand"),
        }
    }

    #[test]
    fn cov_takes_a_test_command_and_the_anchor_command_is_the_default() {
        let cli = Cli::try_parse_from(["xencode", "cov", "--test", "cargo test"]).unwrap();
        match cli.command {
            Some(Commands::Cov { test, .. }) => assert_eq!(test.as_deref(), Some("cargo test")),
            _ => panic!("expected cov"),
        }
        let cli = Cli::try_parse_from(["xencode", "cov"]).unwrap();
        match cli.command {
            // None means "use the repository's own verified test command".
            Some(Commands::Cov { test, base, .. }) => {
                assert!(test.is_none());
                assert!(base.is_none());
            }
            _ => panic!("expected cov"),
        }
    }

    /// QO-4's two thresholds are product decisions, so the defaults are pinned
    /// here as well as in the module: alert at 5% of the baseline, judge at
    /// α = 0.05.
    #[test]
    fn perf_check_defaults_to_a_five_percent_alert_judged_at_five_percent_significance() {
        let cli = Cli::try_parse_from(["xencode", "perf", "check"]).unwrap();
        match cli.command {
            Some(Commands::Perf {
                action:
                    PerfAction::Check {
                        filter,
                        alert_pct,
                        alpha,
                        ..
                    },
            }) => {
                assert!(filter.is_none());
                assert_eq!(alert_pct, 5.0);
                assert_eq!(alpha, 0.05);
            }
            _ => panic!("expected perf check"),
        }

        let cli = Cli::try_parse_from([
            "xencode",
            "perf",
            "check",
            "--filter",
            "retrieve",
            "--alert-pct",
            "2.5",
            "--alpha",
            "0.01",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Perf {
                action:
                    PerfAction::Check {
                        filter,
                        alert_pct,
                        alpha,
                        ..
                    },
            }) => {
                assert_eq!(filter.as_deref(), Some("retrieve"));
                assert_eq!(alert_pct, 2.5);
                assert_eq!(alpha, 0.01);
            }
            _ => panic!("expected perf check"),
        }
    }

    #[test]
    fn perf_record_takes_no_filter_because_a_partial_baseline_is_not_a_baseline() {
        let cli = Cli::try_parse_from(["xencode", "perf", "record", "--force"]).unwrap();
        match cli.command {
            Some(Commands::Perf {
                action: PerfAction::Record { force, .. },
            }) => assert!(force),
            _ => panic!("expected perf record"),
        }
        let rejected = Cli::try_parse_from(["xencode", "perf", "record", "--filter", "retrieve"]);
        assert!(
            rejected.is_err(),
            "recording three of seven paths would replace the baseline with three of seven \
             paths, so the flag is not offered"
        );

        let cli = Cli::try_parse_from(["xencode", "perf", "show"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Perf {
                action: PerfAction::Show { .. }
            })
        ));
    }

    #[test]
    fn a_measured_path_reads_in_the_unit_its_number_belongs_in() {
        // The seven paths span five orders of magnitude, so a fixed unit would
        // print either 0.016 or 801449.095.
        assert_eq!(human_duration(999.0), "999.000 ns");
        assert_eq!(human_duration(15_842.0), "15.842 µs");
        assert_eq!(human_duration(1_038_604.0), "1.039 ms");
        assert_eq!(human_duration(801_449_095.0), "801.449 ms");
        assert_eq!(human_duration(1_500_000_000.0), "1.500 s");
    }

    /// QO-6 — the whole flag set, parsed.
    #[test]
    fn release_notes_parses_every_flag_it_offers() {
        let cli = Cli::try_parse_from([
            "xencode",
            "release-notes",
            "--from",
            "v2.1.0",
            "--to",
            "main",
            "--release",
            "2.2.0",
            "--out",
            "draft/notes.md",
            "--force",
            "--format",
            "json",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::ReleaseNotes {
                from,
                to,
                release,
                out,
                force,
                format,
            }) => {
                assert_eq!(from.as_deref(), Some("v2.1.0"));
                assert_eq!(to.as_deref(), Some("main"));
                assert_eq!(release.as_deref(), Some("2.2.0"));
                assert_eq!(out, Some(path("draft/notes.md")));
                assert!(force);
                assert!(matches!(format, OutputFormat::Json));
            }
            _ => panic!("expected release-notes"),
        }
    }

    #[test]
    fn release_notes_without_flags_drafts_head_as_an_unreleased_block() {
        let cli = Cli::try_parse_from(["xencode", "release-notes"]).unwrap();
        match cli.command {
            Some(Commands::ReleaseNotes {
                from,
                to,
                release,
                out,
                force,
                format,
            }) => {
                assert!(from.is_none(), "the newest tag picks the range");
                assert!(to.is_none(), "the default upper end is HEAD");
                assert!(
                    release.is_none(),
                    "no version is invented for the heading: it stays [Unreleased]"
                );
                assert!(out.is_none(), "the draft goes to standard output");
                assert!(!force, "an existing file is not replaced by default");
                assert!(matches!(format, OutputFormat::Text));
            }
            _ => panic!("expected release-notes"),
        }
    }

    #[test]
    fn a_count_next_to_its_noun_reads_as_english() {
        assert_eq!(plural_count(1, "entry", "entries"), "1 entry");
        assert_eq!(plural_count(0, "entry", "entries"), "0 entries");
        assert_eq!(plural_count(131, "entry", "entries"), "131 entries");
    }

    /// `xencode prices` with nothing after it is the listing of what a cost report
    /// would read — the fetch is the part that dials out, so it is never implied.
    #[test]
    fn prices_alone_shows_the_documents_and_fetches_nothing() {
        let cli = Cli::try_parse_from(["xencode", "prices"]).unwrap();
        match cli.command {
            Some(Commands::Prices { action }) => {
                assert!(action.is_none(), "no action means the showing one");
            }
            _ => panic!("expected prices"),
        }
    }

    #[test]
    fn prices_show_takes_a_format_and_fetch_takes_a_url() {
        let cli = Cli::try_parse_from(["xencode", "prices", "show", "--format", "json"]).unwrap();
        match cli.command {
            Some(Commands::Prices {
                action: Some(PriceAction::Show { format }),
            }) => assert!(matches!(format, OutputFormat::Json)),
            _ => panic!("expected prices show"),
        }
        let cli = Cli::try_parse_from([
            "xencode",
            "prices",
            "fetch",
            "--url",
            "http://127.0.0.1:9/models",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Prices {
                action: Some(PriceAction::Fetch { url }),
            }) => assert_eq!(url.as_deref(), Some("http://127.0.0.1:9/models")),
            _ => panic!("expected prices fetch"),
        }
    }

    /// Bare `xencode runs` lists, the way bare `xencode prices` shows: the
    /// common question needs no verb, and nothing here dials out.
    #[test]
    fn runs_alone_lists_and_show_takes_an_id() {
        let cli = Cli::try_parse_from(["xencode", "runs"]).unwrap();
        match cli.command {
            Some(Commands::Runs { action }) => {
                assert!(action.is_none(), "no action means the listing one");
            }
            _ => panic!("expected runs"),
        }
        let cli = Cli::try_parse_from(["xencode", "runs", "show", "1700000000-aaaa1111"]).unwrap();
        match cli.command {
            Some(Commands::Runs {
                action: Some(RunsAction::Show { run_id, .. }),
            }) => assert_eq!(run_id, "1700000000-aaaa1111"),
            _ => panic!("expected runs show"),
        }
        let cli = Cli::try_parse_from(["xencode", "runs", "trailer", "1700000000-aaaa"]).unwrap();
        match cli.command {
            Some(Commands::Runs {
                action: Some(RunsAction::Trailer { run_id }),
            }) => assert_eq!(run_id, "1700000000-aaaa"),
            _ => panic!("expected runs trailer"),
        }
    }

    /// `xencode run` takes one action per invocation: a prompt with `--log`
    /// is refused by the dispatcher, not guessed at.
    #[test]
    fn run_takes_a_prompt_and_caps_or_one_reporting_flag() {
        let cli = Cli::try_parse_from([
            "xencode",
            "run",
            "fix the typo",
            "--detach",
            "--max-rounds",
            "4",
            "--max-minutes",
            "10",
            "--max-cost",
            "0.5",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Run {
                prompt,
                detach,
                max_rounds,
                max_minutes,
                max_cost,
                ..
            }) => {
                assert_eq!(prompt.as_deref(), Some("fix the typo"));
                assert!(detach);
                assert_eq!(max_rounds, Some(4));
                assert_eq!(max_minutes, Some(10.0));
                assert_eq!(max_cost, Some(0.5));
            }
            _ => panic!("expected run"),
        }
        let cli = Cli::try_parse_from(["xencode", "run", "--resume", "1700000000-aaaa"]).unwrap();
        match cli.command {
            Some(Commands::Run { resume, .. }) => {
                assert_eq!(resume.as_deref(), Some("1700000000-aaaa"))
            }
            _ => panic!("expected run --resume"),
        }
        // The worker is forked, never typed — but it parses, because the
        // forked child is this same binary.
        let cli = Cli::try_parse_from(["xencode", "run", "--child", "1700000000-aaaa"]).unwrap();
        match cli.command {
            Some(Commands::Run { child, .. }) => {
                assert_eq!(child.as_deref(), Some("1700000000-aaaa"))
            }
            _ => panic!("expected run --child"),
        }
    }

    fn path(p: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(p)
    }

    /// `config set agent_fallback_models "a, b,, c"` yields the ordered,
    /// trimmed, non-empty list — the order is the try order (I4-01).
    #[test]
    fn comma_list_parses_ordered_trimmed_and_drops_empties() {
        assert_eq!(parse_comma_list("a, b,, c"), vec!["a", "b", "c"]);
        assert_eq!(
            parse_comma_list(" qwen2.5:14b ,gemini:gemini-2.0-flash , "),
            vec!["qwen2.5:14b", "gemini:gemini-2.0-flash"]
        );
        assert!(parse_comma_list("").is_empty());
        assert!(parse_comma_list("  , ,").is_empty());
    }

    #[test]
    fn loopback_binds_are_accepted_without_tls_or_warning() {
        // "::1" is the case a `starts_with("127.")` check gets wrong.
        for host in ["127.0.0.1", "localhost", "::1", "127.0.0.53"] {
            let bind = resolve_bind(host, 8765, None, None, false).unwrap();
            assert!(bind.addr.ip().is_loopback(), "{host} misclassified");
            assert_eq!(bind.addr.port(), 8765);
            assert!(bind.tls.is_none());
            assert!(bind.warning.is_none());
        }
    }

    #[test]
    fn a_public_plain_bind_is_refused_and_the_error_names_both_ways_out() {
        let err = resolve_bind("0.0.0.0", 8765, None, None, false).unwrap_err();
        assert!(err.contains("--cert") && err.contains("--key"));
        assert!(err.contains("--allow-insecure-public"));
        // IPv6 public bind: same refusal.
        assert!(resolve_bind("::", 8765, None, None, false).is_err());
    }

    #[test]
    fn the_escape_hatch_binds_but_warns_loudly() {
        let bind = resolve_bind("0.0.0.0", 9000, None, None, true).unwrap();
        assert_eq!(bind.addr.port(), 9000);
        assert!(bind.tls.is_none());
        let warning = bind.warning.expect("plain public bind must warn");
        assert!(warning.contains("WITHOUT TLS"));
    }

    #[test]
    fn cert_and_key_must_come_as_a_pair() {
        assert!(resolve_bind(
            "127.0.0.1",
            8765,
            Some(path("c.pem").as_path()),
            None,
            false
        )
        .unwrap_err()
        .contains("--key"));
        assert!(resolve_bind(
            "127.0.0.1",
            8765,
            None,
            Some(path("k.pem").as_path()),
            false
        )
        .unwrap_err()
        .contains("--cert"));
    }

    #[test]
    fn certs_win_over_the_escape_hatch_and_speak() {
        let bind = resolve_bind(
            "0.0.0.0",
            8765,
            Some(path("c.pem").as_path()),
            Some(path("k.pem").as_path()),
            true,
        )
        .unwrap();
        assert_eq!(
            bind.tls,
            Some((path("c.pem"), path("k.pem"))),
            "TLS must be on, not the plain hatch"
        );
        assert!(bind.warning.is_none());
    }

    #[test]
    fn tls_on_loopback_needs_no_hatch() {
        let bind = resolve_bind(
            "::1",
            8765,
            Some(path("c.pem").as_path()),
            Some(path("k.pem").as_path()),
            false,
        )
        .unwrap();
        assert!(bind.tls.is_some());
    }

    #[test]
    fn invalid_hosts_are_rejected_not_resolved_lazily() {
        // Hostnames would need a DNS lookup — run_server must not pretend.
        assert!(resolve_bind("example.com", 8765, None, None, true).is_err());
        assert!(resolve_bind("0.0.0.256", 8765, None, None, true).is_err());
        assert!(resolve_bind("", 8765, None, None, true).is_err());
    }

    #[test]
    fn audit_path_none_disables_explicit_paths_pass_through() {
        assert_eq!(resolve_audit_path(Some("none")).unwrap(), None);
        assert_eq!(
            resolve_audit_path(Some("/tmp/x.jsonl")).unwrap(),
            Some(path("/tmp/x.jsonl"))
        );
        // The default lands in the directory xencode keeps its records in,
        // never the repo.
        let default = resolve_audit_path(None).unwrap().unwrap();
        assert_eq!(default.file_name().unwrap(), "audit.jsonl");
        assert_eq!(
            default.parent().unwrap(),
            xencode_config_rs::paths::state_dir().unwrap()
        );
    }

    #[test]
    fn compute_advise_reads_snapshot_filters_and_errors() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use xencode_context_rs::AdviceKind;
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "xencode-advise-cli-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "mod a;\nmod b;\nmod c;\n").unwrap();
        std::fs::write(root.join("src/a.rs"), "use crate::b::bee;\n").unwrap();
        std::fs::write(root.join("src/b.rs"), "use crate::a::ay;\n").unwrap();
        std::fs::write(root.join("src/c.rs"), "pub fn cc() {}\n").unwrap();
        // Nothing declares or imports this one, so it is the only real orphan in
        // the tree: `mod c;` in lib.rs connects c.rs, which the graph could not
        // see until module declarations counted as edges.
        std::fs::write(root.join("src/unreferenced.rs"), "pub fn u() {}\n").unwrap();
        std::fs::File::create(root.join("Cargo.toml")).unwrap();
        xencode_context_rs::init_project(
            &root,
            std::sync::Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .expect("init");

        let items = compute_advise(&root, None).unwrap();
        assert!(items
            .iter()
            .any(|i| i.kind == AdviceKind::Cycle && i.file == "src/a.rs"));
        assert!(items
            .iter()
            .any(|i| i.kind == AdviceKind::Orphan && i.file == "src/unreferenced.rs"));
        assert!(!items
            .iter()
            .any(|i| i.kind == AdviceKind::Orphan && i.file == "src/c.rs"));

        let filtered = compute_advise(&root, Some("unreferenced.rs")).unwrap();
        assert!(!filtered.is_empty());
        assert!(filtered.iter().all(|i| i.file.contains("unreferenced.rs")));
        assert!(compute_advise(&root, Some("nope")).unwrap().is_empty());

        // No snapshot at all → actionable error, not an empty report.
        let err = compute_advise(std::path::Path::new("/definitely-not-a-repo-xencode"), None)
            .expect_err("should fail");
        assert!(err.contains("no project index"), "{err}");

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn image_text_line_shows_mime_dimensions_and_bytes() {
        let meta = ImageMeta {
            path: "assets/logo.png".to_string(),
            format: ImageFormat::Png,
            width: Some(800),
            height: Some(600),
            bytes: 12345,
        };
        assert_eq!(
            format_image_text(&meta),
            "assets/logo.png — image/png 800x600 · 12345 bytes"
        );
    }

    #[test]
    fn image_text_line_marks_vector_without_dimensions() {
        let meta = ImageMeta {
            path: "assets/icon.svg".to_string(),
            format: ImageFormat::Svg,
            width: None,
            height: None,
            bytes: 512,
        };
        assert_eq!(
            format_image_text(&meta),
            "assets/icon.svg — image/svg+xml vector · 512 bytes"
        );
    }

    #[test]
    fn fetch_text_line_shows_url_title_and_body() {
        use xencode_analysis_rs::web::FetchedPage;
        let page = FetchedPage {
            url: "https://example.com/a".to_string(),
            title: Some("Example".to_string()),
            text: "hello web".to_string(),
            bytes: 128,
        };
        assert_eq!(
            super::format_fetch_text(&page),
            "https://example.com/a\nExample (128 bytes)\n\nhello web"
        );
        let untitled = FetchedPage {
            title: None,
            ..page
        };
        assert!(super::format_fetch_text(&untitled).contains("(no title)"));
    }

    #[test]
    fn fetch_text_truncates_huge_pages_with_notice() {
        use xencode_analysis_rs::web::FetchedPage;
        let page = FetchedPage {
            url: "https://example.com/big".to_string(),
            title: Some("Big".to_string()),
            text: "z".repeat(super::FETCH_TEXT_CAP_CHARS + 10),
            bytes: 99999,
        };
        let out = super::format_fetch_text(&page);
        assert!(out.contains("…[truncated to"), "{out}");
        assert!(out.contains("--format json"), "{out}");
        assert!(!out.contains(&"z".repeat(super::FETCH_TEXT_CAP_CHARS + 10)));
    }

    #[test]
    fn json_schema_rejects_invalid_json() {
        let ok = super::parse_json_schema(Some(r#"{"type":"object"}"#.to_string())).unwrap();
        assert_eq!(ok, Some(serde_json::json!({"type": "object"})));
        assert_eq!(super::parse_json_schema(None).unwrap(), None);
        let err = super::parse_json_schema(Some("{broken".to_string())).unwrap_err();
        assert!(err.contains("invalid --json-schema"), "{err}");
    }

    #[test]
    fn a_declared_schema_decides_what_counts_as_an_answer() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["files"],
            "properties": {"files": {"type": "array", "items": {"type": "string"}}}
        });
        assert!(super::structured_answer_fits(
            "{\"files\": [\"a.rs\"]}\n",
            Some(&schema)
        ));
        // Nothing declared, nothing to fit.
        assert!(super::structured_answer_fits("any prose at all", None));
        // Prose, the wrong shape of JSON, and an empty reply all fail.
        assert!(!super::structured_answer_fits(
            "here you go: {\"files\": \"a.rs\"}",
            Some(&schema)
        ));
        assert!(!super::structured_answer_fits(
            r#"{"files": "a.rs"}"#,
            Some(&schema)
        ));
        assert!(!super::structured_answer_fits("", Some(&schema)));
    }

    #[test]
    fn review_text_summarizes_files_and_notes() {
        let files = vec![
            super::ReviewedFile {
                path: "src/a.rs".to_string(),
                added: Some(10),
                deleted: Some(2),
                issues: vec![],
                note: None,
            },
            super::ReviewedFile {
                path: "assets/logo.png".to_string(),
                added: None,
                deleted: None,
                issues: vec![],
                note: Some("binary".to_string()),
            },
        ];
        let out = super::format_review_text("main", None, &files);
        assert!(
            out.contains("Review of diff main...HEAD (2 files)"),
            "{out}"
        );
        assert!(out.contains("src/a.rs (+10 -2)"), "{out}");
        assert!(out.contains("assets/logo.png (binary)"), "{out}");

        let suffixed = super::format_review_text(
            "origin/main",
            Some(" [base resolved from origin/HEAD]"),
            &files,
        );
        assert!(
            suffixed.contains(
                "Review of diff origin/main...HEAD (2 files) [base resolved from origin/HEAD]"
            ),
            "{suffixed}"
        );
    }

    #[test]
    fn resolve_review_base_explicit_and_fallbacks() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-cli-base-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        // 1. Explicit base always wins
        let (base, src) = super::resolve_review_base(&dir, Some("custom-branch"));
        assert_eq!(base, "custom-branch");
        assert_eq!(src, super::BaseSource::Explicit);
        assert_eq!(src.header_suffix(), "");

        // Outside a repo: falls back to "main"
        let (base, src) = super::resolve_review_base(&dir, None);
        assert_eq!(base, "main");
        assert_eq!(src, super::BaseSource::DefaultFallback);
        assert_eq!(
            src.header_suffix(),
            " [no remote; fell back to default 'main']"
        );

        // Inside a repo with only master branch:
        let git = |args: &[&str]| {
            assert!(std::process::Command::new("git")
                .args(args)
                .current_dir(&dir)
                .output()
                .unwrap()
                .status
                .success());
        };
        git(&["init", "-q", "-b", "master"]);
        git(&["config", "user.email", "test@xencode.local"]);
        git(&["config", "user.name", "Xencode Test"]);
        git(&["commit", "--allow-empty", "-q", "-m", "initial"]);

        let (base, src) = super::resolve_review_base(&dir, None);
        assert_eq!(base, "master");
        assert_eq!(
            src,
            super::BaseSource::InitDefaultBranch("master".to_string())
        );
        assert_eq!(
            src.header_suffix(),
            " [no remote; fell back to init.defaultBranch (master)]"
        );

        // With origin/HEAD pointing to origin/main:
        git(&["update-ref", "refs/remotes/origin/main", "HEAD"]);
        git(&[
            "symbolic-ref",
            "refs/remotes/origin/HEAD",
            "refs/remotes/origin/main",
        ]);
        let (base, src) = super::resolve_review_base(&dir, None);
        assert_eq!(base, "origin/main");
        assert_eq!(src, super::BaseSource::OriginHead);
        assert_eq!(src.header_suffix(), " [base resolved from origin/HEAD]");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn review_file_notes_missing_and_binary() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-cli-review-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Deleted from the working tree: note, not silence, not panic.
        let gone = super::review_file(
            &dir,
            &xencode_context_rs::DiffFile {
                path: "gone.rs".to_string(),
                added: Some(1),
                deleted: Some(1),
            },
        );
        assert!(gone.note.is_some(), "{gone:?}");
        // Binary content: analyzer cannot read it → note.
        std::fs::write(dir.join("blob.o"), [0xFF, 0xFE, 0x00]).unwrap();
        let bin = super::review_file(
            &dir,
            &xencode_context_rs::DiffFile {
                path: "blob.o".to_string(),
                added: None,
                deleted: None,
            },
        );
        assert!(bin.note.is_some(), "{bin:?}");
        // Plain code analyzes.
        std::fs::write(dir.join("ok.rs"), "pub fn f() {}\n").unwrap();
        let ok = super::review_file(
            &dir,
            &xencode_context_rs::DiffFile {
                path: "ok.rs".to_string(),
                added: Some(1),
                deleted: Some(0),
            },
        );
        assert!(ok.note.is_none(), "{ok:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn review_root_prefers_git_toplevel() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-cli-root-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        // Outside any repo: falls back to cwd itself.
        assert_eq!(
            super::resolve_review_root(&dir.join("sub")),
            dir.join("sub")
        );
        assert_eq!(super::resolve_review_root(&dir), dir);
        // Inside a repo: the toplevel, even from a subdirectory.
        let git = |args: &[&str]| {
            assert!(std::process::Command::new("git")
                .args(args)
                .current_dir(&dir)
                .output()
                .unwrap()
                .status
                .success());
        };
        git(&["init", "-q"]);
        let toplevel = super::resolve_review_root(&dir.join("sub"));
        let canon = |p: std::path::PathBuf| p.canonicalize().unwrap();
        assert_eq!(canon(toplevel), canon(dir.clone()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A stream line as a script receives it: the rendered text must hold
    /// exactly one JSON object, whatever the answer contained.
    fn stream_line(event: &serde_json::Value) -> serde_json::Value {
        let text = event.to_string();
        assert!(
            !text.contains('\n'),
            "a stream line held a newline: {text:?}"
        );
        serde_json::from_str(&text).expect("a stream line is not valid JSON")
    }

    #[test]
    fn every_stream_event_names_the_schema_version_and_its_kind() {
        let events = vec![
            super::query_stream::start("llama:", "llamacpp", "local", None),
            super::query_stream::token("hi"),
            super::query_stream::done("hi", false, 5, None, None),
            super::query_stream::error("Query failed: nobody is listening"),
        ];
        for event in &events {
            let parsed = stream_line(event);
            assert_eq!(parsed["v"], super::query_stream::VERSION, "{parsed}");
            assert!(parsed["type"].is_string(), "{parsed}");
        }
        let kinds: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(kinds, vec!["start", "token", "done", "error"]);
    }

    #[test]
    fn an_answer_full_of_newlines_and_quotes_still_fits_on_one_line() {
        // The reason a script can read this stream by splitting on newlines.
        let answer = "two\nlines \"quoted\"\ttab 日本語 \\ backslash";
        assert_eq!(
            stream_line(&super::query_stream::token(answer))["text"],
            answer
        );
        assert_eq!(
            stream_line(&super::query_stream::done(answer, false, 1, None, None))["answer"],
            answer
        );
    }

    #[test]
    fn token_lines_rebuilt_in_order_are_the_answer_the_done_line_reports() {
        // The invariant a piping script depends on, including for an answer
        // the cache served without generating anything.
        let pieces = ["Hel", "lo\n", "世界"];
        let answer: String = pieces.concat();
        let rebuilt: String = pieces
            .iter()
            .map(|piece| {
                stream_line(&super::query_stream::token(piece))["text"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(rebuilt, answer);
    }

    #[test]
    fn the_start_line_says_which_route_was_chosen_and_says_so_when_there_is_no_session() {
        let with_session = stream_line(&super::query_stream::start(
            "qwen2.5:7b",
            "ollama",
            "local",
            Some("s-1"),
        ));
        assert_eq!(with_session["model"], "qwen2.5:7b");
        assert_eq!(with_session["provider"], "ollama");
        assert_eq!(with_session["session"], "s-1");
        let without = stream_line(&super::query_stream::start(
            "anthropic:claude",
            "anthropic",
            "cloud",
            None,
        ));
        assert!(
            without["session"].is_null(),
            "a run with no conversation said otherwise: {without}"
        );
    }

    #[test]
    fn a_done_line_leaves_out_the_counts_no_route_reported() {
        let measured = stream_line(&super::query_stream::done(
            "hi",
            false,
            1_537,
            Some(42),
            Some(6.5432),
        ));
        assert_eq!(measured["tokens_generated"], 42);
        assert_eq!(
            measured["tokens_per_second"], 6.54,
            "rate should round plainly"
        );
        assert_eq!(measured["elapsed_ms"], 1_537);
        assert_eq!(measured["cached"], false);
        let unmeasured = stream_line(&super::query_stream::done("hi", true, 3, None, None));
        assert!(unmeasured["tokens_generated"].is_null(), "{unmeasured}");
        assert!(unmeasured["tokens_per_second"].is_null(), "{unmeasured}");
        assert_eq!(unmeasured["cached"], true);
    }

    /// Images go out as parts on the final user turn, with the prompt text
    /// ahead of them — the shape vision-capable endpoints accept.
    #[test]
    fn images_attach_as_parts_on_the_final_user_message() {
        let mut msgs = vec![
            ChatMessage::text("system", "be brief"),
            ChatMessage::text("user", "what is this?"),
        ];
        assert!(attach_images_to_final_user_message(
            &mut msgs,
            vec!["data:image/jpeg;base64,AAAA".to_string()]
        ));
        assert_eq!(
            msgs[1].content,
            MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "what is this?".to_string()
                },
                ContentPart::ImageUrl {
                    image_url: xencode_providers_rs::ImageUrlPart {
                        url: "data:image/jpeg;base64,AAAA".to_string(),
                        detail: None,
                    }
                },
            ])
        );
        assert_eq!(
            msgs[0].content,
            MessageContent::Text("be brief".to_string())
        );
    }

    /// Several images keep their order after the text.
    #[test]
    fn several_images_keep_their_order() {
        let mut msgs = vec![ChatMessage::text("user", "compare")];
        assert!(attach_images_to_final_user_message(
            &mut msgs,
            vec![
                "data:image/png;base64,ONE".to_string(),
                "data:image/png;base64,TWO".to_string()
            ]
        ));
        match &msgs[0].content {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 3);
                assert_eq!(
                    msgs[0].image_urls(),
                    vec!["data:image/png;base64,ONE", "data:image/png;base64,TWO"]
                );
                assert_eq!(msgs[0].text_content(), "compare");
            }
            other => panic!("expected parts, got {other:?}"),
        }
    }

    /// A turn with no user message cannot carry images. Reported, not silently
    /// dropped — a request that quietly loses the images would be answered from
    /// the prompt alone and look like the model ignored the picture.
    #[test]
    fn images_are_refused_rather_than_dropped_when_there_is_no_user_turn() {
        let mut msgs = vec![ChatMessage::text("system", "be brief")];
        assert!(!attach_images_to_final_user_message(
            &mut msgs,
            vec!["data:image/png;base64,AAAA".to_string()]
        ));
        let mut empty: Vec<ChatMessage> = Vec::new();
        assert!(!attach_images_to_final_user_message(
            &mut empty,
            vec!["data:image/png;base64,AAAA".to_string()]
        ));
    }

    /// An empty prompt alongside images still sends the images.
    #[test]
    fn images_go_out_even_when_the_text_part_is_empty() {
        let mut msgs = vec![ChatMessage::text("user", "")];
        assert!(attach_images_to_final_user_message(
            &mut msgs,
            vec!["data:image/png;base64,AAAA".to_string()]
        ));
        assert_eq!(msgs[0].image_urls(), vec!["data:image/png;base64,AAAA"]);
        assert_eq!(msgs[0].text_content(), "");
    }

    /// The flag parses, repeats, and is absent unless asked for.
    #[test]
    fn the_image_flag_parses_and_repeats() {
        let image_paths = |args: &[&str]| -> Vec<String> {
            match Cli::parse_from(args.iter().copied()).command {
                Some(Commands::Query { images, .. }) => images,
                _ => panic!("expected a query command"),
            }
        };
        assert_eq!(
            image_paths(&["xencode", "query", "--image", "a.png", "hello"]),
            vec!["a.png".to_string()]
        );
        assert_eq!(
            image_paths(&["xencode", "query", "--image", "a.png", "--image", "b.jpg", "hello"]),
            vec!["a.png".to_string(), "b.jpg".to_string()]
        );
        assert!(image_paths(&["xencode", "query", "hello"]).is_empty());
    }

    /// A file that is not an image, or cannot be read, is named and refused
    /// before any request goes out.
    #[test]
    fn an_unusable_image_is_named_and_refused() {
        let dir = std::env::temp_dir().join(format!("xe-img-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("fake.png");
        std::fs::write(&fake, b"not an image").unwrap();
        let reason = super::encode_query_image(fake.to_str().unwrap()).unwrap_err();
        assert_eq!(
            reason,
            format!("{} is not a recognized image", fake.display())
        );

        let gone = dir.join("gone.png");
        let reason = super::encode_query_image(gone.to_str().unwrap()).unwrap_err();
        assert!(
            reason.starts_with(&format!("cannot read image {}", gone.display())),
            "{reason}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A real image becomes a data URL the vision path can send.
    #[test]
    fn a_real_image_becomes_a_data_url() {
        let dir = std::env::temp_dir().join(format!("xe-img-ok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shot.png");
        std::fs::write(&path, ONE_PIXEL_PNG).unwrap();
        let url = super::encode_query_image(path.to_str().unwrap()).unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "{url}");
        // The payload decodes back to real image bytes, not a header alone: the
        // PNG magic is in there and the format is still recognisable. Checked on
        // the encoded text so the assertion is about what goes on the wire.
        let (_, encoded) = url.split_once(";base64,").unwrap();
        let raw = decode_base64(encoded);
        assert_eq!(&raw[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(
            xencode_analysis_rs::images::detect_format(&raw),
            Some(ImageFormat::Png)
        );
        assert!(raw.len() > 8, "only a signature was sent");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_source_word_is_the_same_spelling_the_metrics_rows_use() {
        use xencode_context_rs::MetricSource;
        use xencode_providers_rs::Egress;
        let spelled = |source: MetricSource| serde_json::to_value(source).unwrap();
        assert_eq!(
            spelled(MetricSource::Local),
            super::source_word(Egress::Local),
            "the stream and cache/metrics.jsonl disagree about what a local route is called"
        );
        assert_eq!(
            spelled(MetricSource::Cloud),
            super::source_word(Egress::Cloud),
            "the stream and cache/metrics.jsonl disagree about what an off-machine route is called"
        );
    }

    /// The `nvidia` endpoint is resolved from `NVIDIA_NIM_API_KEY` as well as
    /// from the configuration file, so a test cannot pin it without editing the
    /// environment every other test reads. The rest are asserted on in full.
    fn endpoint_names(config: &xencode_config_rs::XencodeConfig) -> Vec<&'static str> {
        super::endpoints(config)
            .iter()
            .filter(|endpoint| endpoint.name != "nvidia")
            .map(|endpoint| endpoint.name)
            .collect()
    }

    #[test]
    fn a_config_with_no_keys_dials_only_the_two_local_servers() {
        let config = xencode_config_rs::XencodeConfig::default();
        assert_eq!(endpoint_names(&config), ["ollama", "llamacpp"]);

        // The start command is the row's fix, so it has to be the real one.
        let list = super::endpoints(&config);
        assert_eq!(list[0].start_command, Some("ollama serve"));
        assert_eq!(
            list[1].start_command,
            Some("llama-server --model <path>"),
            "the report must tell the reader to run the command that loads a model"
        );
        assert_eq!(list[0].url, config.ollama_url);
        assert_eq!(list[1].url, config.llama_cpp_url);
    }

    #[test]
    fn a_configured_remote_and_a_keyed_cloud_join_the_dial_list() {
        let mut config = xencode_config_rs::XencodeConfig {
            remote_base_url: "http://127.0.0.1:18000/v1".to_string(),
            ..Default::default()
        };
        assert_eq!(endpoint_names(&config), ["ollama", "llamacpp", "remote"]);
        // A service the person may not have running is not something this
        // machine can be told to start, so there is no command to name.
        assert_eq!(super::endpoints(&config)[2].start_command, None);

        config.api_keys.openai_api_key = Some("sk-not-a-real-key".to_string());
        config.api_keys.google_gemini_api_key = Some("not-a-real-key".to_string());
        assert_eq!(
            endpoint_names(&config),
            ["ollama", "llamacpp", "remote", "openai", "gemini"],
            "clouds come after the local endpoints, in the order the code lists them"
        );
        let clouds: Vec<String> = super::endpoints(&config)
            .iter()
            .skip(3)
            .map(|endpoint| endpoint.url.clone())
            .collect();
        assert_eq!(
            clouds,
            [
                "https://api.openai.com".to_string(),
                "https://generativelanguage.googleapis.com".to_string()
            ],
            "the address dialled is the provider's own host, not one the report guessed"
        );
    }

    #[test]
    fn a_key_that_is_only_whitespace_does_not_open_a_cloud_row() {
        let config = xencode_config_rs::XencodeConfig {
            api_keys: xencode_config_rs::ApiKeys {
                openrouter_api_key: Some("   ".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            endpoint_names(&config),
            ["ollama", "llamacpp"],
            "a saved blank key is not a configured provider"
        );
    }

    #[test]
    fn the_bridge_rows_are_named_once_inside_the_report() {
        let report = xencode_colab_rs::PreflightReport {
            checks: vec![
                xencode_colab_rs::Check {
                    name: "colab CLI",
                    ok: true,
                    detail: "/home/someone/.local/bin/colab".to_string(),
                    fix: None,
                },
                xencode_colab_rs::Check {
                    name: "Colab API",
                    ok: false,
                    detail: "`colab sessions` failed — not signed in".to_string(),
                    fix: Some("gcloud auth application-default login".to_string()),
                },
                xencode_colab_rs::Check {
                    name: "OpenSSH",
                    ok: true,
                    detail: "ssh at /usr/bin/ssh".to_string(),
                    fix: None,
                },
            ],
        };
        let rows = super::bridge_rows(report);
        let names: Vec<&str> = rows.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(names, ["colab:CLI", "colab:API", "colab:OpenSSH"]);
        assert_eq!(rows[1].state, "fail");
        assert_eq!(
            rows[1].fix.as_deref(),
            Some("gcloud auth application-default login"),
            "the remediation the bridge owns rides through unchanged"
        );
    }

    #[test]
    fn a_refused_model_is_answered_with_the_command_that_would_work() {
        // The three ways Ollama can turn the default model away, each answered
        // with what the reader should actually run.
        use xencode_models_rs::OllamaError;
        assert_eq!(
            super::ollama_fix(
                &OllamaError::NotRunning("error sending request".to_string()),
                "ollama:qwen2.5:7b"
            ),
            "ollama serve"
        );
        // An id carrying the provider prefix is what the configuration can hold
        // and Ollama cannot serve: `ollama pull ollama:qwen2.5:7b` would fetch
        // nothing, so the row names the bare model instead.
        let prefixed = super::ollama_fix(
            &OllamaError::ModelNotFound("ollama:qwen2.5:7b".to_string()),
            "ollama:qwen2.5:7b",
        );
        assert!(
            prefixed.contains("set the default to `qwen2.5:7b`"),
            "{prefixed}"
        );
        assert!(prefixed.contains("`ollama pull qwen2.5:7b`"), "{prefixed}");
        assert!(
            !prefixed.contains("pull ollama:"),
            "the fix must not tell the reader to pull a prefixed name: {prefixed}"
        );
        assert_eq!(
            super::ollama_fix(
                &OllamaError::ModelNotFound("qwen3:4b".to_string()),
                "qwen3:4b"
            ),
            "ollama pull qwen3:4b"
        );
        // Anything else is the Ollama client's own problem, and the command that
        // explains it in full is the health check.
        assert_eq!(
            super::ollama_fix(&OllamaError::Timeout("20 s".to_string()), "qwen3:4b"),
            "xencode model health qwen3:4b"
        );
    }

    #[test]
    fn the_json_report_omits_the_fix_key_only_where_there_is_nothing_to_fix() {
        let passing = doc::check_model(true, "qwen3:4b is loaded".to_string(), None);
        let json = serde_json::to_value(&passing).unwrap();
        assert!(
            json.get("fix").is_none(),
            "a reader counting failures should not have to tell a null from an action: {json}"
        );

        let failing = doc::check_model(
            false,
            "nothing listening on 11434".to_string(),
            Some("ollama serve".to_string()),
        );
        let json = serde_json::to_value(&failing).unwrap();
        assert_eq!(json["fix"], serde_json::json!("ollama serve"));
        assert_eq!(json["state"], serde_json::json!("fail"));
    }

    #[test]
    fn the_colab_preview_names_the_keys_a_bring_up_would_rewrite() {
        let config = xencode_config_rs::XencodeConfig::default();
        let lines = super::colab_up_config_preview(&config, "llama.cpp", 18000);
        let text = lines.join("\n");
        assert!(
            text.contains("llama_cpp_url: http://localhost:8080 → http://127.0.0.1:18000"),
            "{text}"
        );
        assert!(
            text.contains("remote_base_url:  → http://127.0.0.1:18000/v1"),
            "{text}"
        );
        assert!(
            !text.contains("ollama_url"),
            "a llama.cpp forward does not move Ollama's endpoint: {text}"
        );

        let ollama = super::colab_up_config_preview(&config, "ollama", 19000);
        let text = ollama.join("\n");
        assert!(text.contains("ollama_url"), "{text}");
        assert!(text.contains("http://127.0.0.1:19000"), "{text}");
    }

    #[test]
    fn the_colab_preview_says_so_when_the_config_already_points_at_the_forward() {
        let mut config = xencode_config_rs::XencodeConfig::default();
        xencode_colab_rs::point_config_at_forward(
            &mut config,
            "llama.cpp",
            "http://127.0.0.1:18000",
        );
        let lines = super::colab_up_config_preview(&config, "llama.cpp", 18000);
        assert_eq!(
            lines,
            vec!["config.json would not change".to_string()],
            "an up that changes nothing says that, instead of printing a diff of nothing"
        );
    }

    #[test]
    fn compete_run_parses_arms_edits_and_commands() {
        let cli = Cli::try_parse_from([
            "xencode",
            "compete",
            "run",
            "Loop or recurse?",
            "--arm",
            "loop=Iterative",
            "--arm",
            "recur",
            "--edit",
            "loop",
            "src/lib.rs",
            "pub fn a() { while false {} }",
            "--command",
            "recur",
            "touch READY",
            "--skip",
            "test",
            "--timeout",
            "30",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Compete {
                action:
                    super::CompeteAction::Run {
                        prompt,
                        arms,
                        edits,
                        commands,
                        skip,
                        timeout,
                        format,
                    },
            }) => {
                assert_eq!(prompt, "Loop or recurse?");
                assert_eq!(arms, ["loop=Iterative", "recur"]);
                assert_eq!(
                    edits,
                    ["loop", "src/lib.rs", "pub fn a() { while false {} }"]
                );
                assert_eq!(commands, ["recur", "touch READY"]);
                assert_eq!(skip, ["test"]);
                assert_eq!(timeout, 30);
                assert!(matches!(format, OutputFormat::Text));
            }
            _ => panic!("expected compete run"),
        }

        let specs = super::build_candidate_arms(
            &["loop=Iterative".to_string(), "recur".to_string()],
            &[
                "loop".to_string(),
                "src/lib.rs".to_string(),
                "fn a() {}".to_string(),
            ],
            &["recur".to_string(), "touch READY".to_string()],
        )
        .unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].arm_id, "loop");
        assert_eq!(specs[0].label, "Iterative");
        assert_eq!(
            specs[0].file_edits[0].0,
            std::path::PathBuf::from("src/lib.rs")
        );
        assert_eq!(specs[1].arm_id, "recur");
        // An arm named without a label is called by its id, not left blank.
        assert_eq!(specs[1].label, "recur");
        assert_eq!(specs[1].command.as_deref(), Some("touch READY"));

        // Omitting --arm competes the two default arms.
        let defaults = super::build_candidate_arms(&[], &[], &[]).unwrap();
        assert_eq!(
            defaults
                .iter()
                .map(|s| s.arm_id.as_str())
                .collect::<Vec<_>>(),
            ["arm-a", "arm-b"]
        );
    }

    /// An edit is a write into a worktree, so the paths it names are checked
    /// before anything is built: no escaping the arm's own tree, no writing to
    /// an arm nobody declared.
    #[test]
    fn compete_refuses_edits_that_escape_or_name_no_arm() {
        let escape = super::build_candidate_arms(
            &["a".to_string(), "b".to_string()],
            &[
                "a".to_string(),
                "../escape.txt".to_string(),
                "x".to_string(),
            ],
            &[],
        )
        .unwrap_err();
        assert!(escape.contains("escapes the worktree"), "{escape}");

        let absolute = super::build_candidate_arms(
            &["a".to_string(), "b".to_string()],
            &["a".to_string(), "/etc/passwd".to_string(), "x".to_string()],
            &[],
        )
        .unwrap_err();
        assert!(absolute.contains("escapes the worktree"), "{absolute}");

        let unknown = super::build_candidate_arms(
            &["a".to_string(), "b".to_string()],
            &["c".to_string(), "src/lib.rs".to_string(), "x".to_string()],
            &[],
        )
        .unwrap_err();
        assert!(unknown.contains("known arms: a, b"), "{unknown}");

        let dupe = super::build_candidate_arms(
            &["a".to_string(), "a".to_string()],
            &[],
            &["a".to_string(), "true".to_string()],
        )
        .unwrap_err();
        assert!(dupe.contains("given twice"), "{dupe}");

        let two_cmds = super::build_candidate_arms(
            &["a".to_string(), "b".to_string()],
            &[],
            &[
                "a".to_string(),
                "touch one".to_string(),
                "a".to_string(),
                "touch two".to_string(),
            ],
        )
        .unwrap_err();
        assert!(two_cmds.contains("already has a command"), "{two_cmds}");

        let blank = super::build_candidate_arms(&["=no id".to_string(), "b".to_string()], &[], &[])
            .unwrap_err();
        assert!(blank.contains("has no id"), "{blank}");
    }

    #[test]
    fn compete_list_show_and_pick_parse_their_ids() {
        let cli = Cli::try_parse_from(["xencode", "compete", "list"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Compete {
                action: super::CompeteAction::List { .. }
            })
        ));

        let cli = Cli::try_parse_from([
            "xencode",
            "compete",
            "show",
            "compete-1",
            "--format",
            "json",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Compete {
                action: super::CompeteAction::Show { run_id, format },
            }) => {
                assert_eq!(run_id, "compete-1");
                assert!(matches!(format, OutputFormat::Json));
            }
            _ => panic!("expected compete show"),
        }

        let cli = Cli::try_parse_from(["xencode", "compete", "pick", "compete-1", "loop"]).unwrap();
        match cli.command {
            Some(Commands::Compete {
                action: super::CompeteAction::Pick { run_id, arm_id },
            }) => {
                assert_eq!(run_id, "compete-1");
                assert_eq!(arm_id, "loop");
            }
            _ => panic!("expected compete pick"),
        }

        // `run` cannot be invoked without the question it is answering.
        assert!(Cli::try_parse_from(["xencode", "compete", "run"]).is_err());
    }

    #[test]
    fn cli_merge_parsing() {
        use super::MergeAction;

        // Precheck
        let cli =
            Cli::try_parse_from(["xencode", "merge", "precheck", "feat-1", "--base", "master"])
                .unwrap();
        match cli.command {
            Some(Commands::Merge {
                action: MergeAction::Precheck { branch, base, .. },
            }) => {
                assert_eq!(branch, "feat-1");
                assert_eq!(base, "master");
            }
            _ => panic!("expected merge precheck"),
        }

        // Plan
        let cli = Cli::try_parse_from([
            "xencode", "merge", "plan", "--branch", "feat-1", "--branch", "feat-2", "--base",
            "master",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Merge {
                action: MergeAction::Plan { branches, base, .. },
            }) => {
                assert_eq!(branches, vec!["feat-1", "feat-2"]);
                assert_eq!(base, "master");
            }
            _ => panic!("expected merge plan"),
        }

        // Land
        let cli = Cli::try_parse_from([
            "xencode",
            "merge",
            "land",
            "--branch",
            "feat-1",
            "--branch",
            "feat-2",
            "--approved-by",
            "Alice",
            "--test-cmd",
            "cargo test",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Merge {
                action:
                    MergeAction::Land {
                        branches,
                        approved_by,
                        test_cmds,
                        ..
                    },
            }) => {
                assert_eq!(branches, vec!["feat-1", "feat-2"]);
                assert_eq!(approved_by, "Alice");
                assert_eq!(test_cmds, vec!["cargo test"]);
            }
            _ => panic!("expected merge land"),
        }

        // Land without approved-by fails (required human gate)
        assert!(Cli::try_parse_from(["xencode", "merge", "land", "--branch", "feat-1",]).is_err());
    }
}
