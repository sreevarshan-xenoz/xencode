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
use xencode_config_rs::XencodeConfig;
use xencode_core_rs::{scan_workspace, ScanOptions};
use xencode_memory_rs::ConversationMemory;
use xencode_models_rs::{find_llama_server, LlamaCppClient, LlamaCppOptions, OllamaClient};
use xencode_plugin_rs::{default_plugin_dir, PluginRegistry, PluginRuntime};
use xencode_providers_rs::{ChatMessage, EgressPolicy, ProviderManager};
use xencode_server_rs::ws::AppState as ServerState;

/// Output format for analysis results
#[derive(clap::ValueEnum, Clone)]
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

        /// Audit log path; "none" disables (default: ~/.xencode/audit.jsonl)
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
    /// failure, never as an empty success. Costs nothing: the task asks one
    /// question about one file and asks for no changes, so an agent with no
    /// account stops at its auth check — which is itself an observation.
    Interop {
        /// Probe only these agents (repeatable); default is the whole roster
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

        /// Report, read-only, which agents look configured on this machine, and
        /// what to run if one is not. Starts no login and reads no credential.
        #[arg(long)]
        check_auth: bool,
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

    /// Review the diff between a base branch and HEAD, file by file
    Review {
        /// Base branch, tag, commit — or HEAD for uncommitted changes
        #[arg(long, default_value = "main")]
        base: String,

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

    /// Launch the Terminal User Interface
    Tui,
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
    /// List installed plugins and whether each one actually loads
    List,
    /// Install a plugin from a path
    Install {
        /// Path to plugin directory or manifest
        path: std::path::PathBuf,
    },
    /// Remove a plugin by name
    Remove {
        /// Name of the plugin
        name: String,
    },
}
#[derive(Subcommand)]
enum ConfigAction {
    /// Display current configuration
    Show,
    /// Set a configuration value
    Set {
        /// Configuration key (e.g., default_model, ollama_url)
        key: String,
        /// Value to set. A leading hyphen is allowed because the value people
        /// set most often is `llama_cpp_args`, which is a server command line,
        /// and the line `xencode hw probe` hands them to paste starts with a
        /// flag.
        #[arg(allow_hyphen_values = true)]
        value: String,
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
    },
    /// Report the Colab bridge state: forward pid, `colab sessions`, and a
    /// /v1/models probe on the forward
    Status,
    /// Tear the Colab bridge down: kill the forward, `colab stop`, clear state
    Down,
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
}

#[derive(Subcommand)]
enum AuditAction {
    /// Check an audit log for records that were changed after they were written
    Verify {
        /// Log to check (default: ~/.xencode/audit.jsonl)
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
    List,
    /// Show transcript of a session
    Show {
        /// Session ID
        session: String,
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

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Bare `xencode` launches the TUI, as documented in the README
    let result = match cli.command.unwrap_or(Commands::Tui) {
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
            check_auth,
        } => run_interop(agents, timeout, out, format, repeat, check_auth),
        Commands::Anchor {
            path,
            timeout,
            dry_run,
            format,
        } => run_anchor(path, timeout, dry_run, format),
        Commands::Review { base, format } => run_review(base, format),
        Commands::Replay {
            run_id,
            list,
            run_tools,
            tool_root,
            out,
        } => run_replay(run_id, list, run_tools, tool_root, out).await,
        Commands::Plugin { action } => run_plugin_action(action),
        Commands::Eval { action } => run_eval(action).await,
        Commands::Llamacpp { action } => run_llamacpp(action).await,
        Commands::Hw { action } => run_hw(action),
        Commands::History { action } => run_history(action),
        Commands::Colab { action } => run_colab(action).await,
        Commands::Tui => run_tui().await,
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

fn run_config(action: ConfigAction) -> Result<(), String> {
    match action {
        ConfigAction::Show => {
            let config = XencodeConfig::load().map_err(|e| e.to_string())?;
            let json = config.to_json().map_err(|e| e.to_string())?;
            println!("{json}");
            Ok(())
        }
        ConfigAction::Set { key, value } => {
            let mut config = XencodeConfig::load().map_err(|e| e.to_string())?;
            let mut secret = false;
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
                // The value is a token: report that it was stored, never echo it back.
                "remote_key" => {
                    config.api_keys.remote_api_key = if value.trim().is_empty() {
                        None
                    } else {
                        Some(value.clone())
                    };
                    secret = true;
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
                // Keep every model call of every run, in the clear, under
                // `.xencode/cache/sessions`. Off by default because it is the
                // most sensitive copy this program can make of a conversation.
                "session_recording" => config.session_recording = parse_bool(&value)?,
                // Let a saved profile take a turn on its own when the prompt reads
                // as the kind of work it is marked for. Off until it is asked for,
                // because a model the user did not choose answering a query is a
                // surprise a script cannot see in its own output.
                "model_routing" => config.model_routing = parse_bool(&value)?,
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
                _ => return Err(format!("unknown config key: {key}")),
            }
            config.save().map_err(|e| e.to_string())?;
            if secret {
                println!("set {key} = (stored, not shown)");
            } else {
                println!("set {key} = {value}");
            }
            Ok(())
        }
        ConfigAction::Reset => {
            let config = XencodeConfig::default();
            config.save().map_err(|e| e.to_string())?;
            println!("configuration reset to defaults");
            Ok(())
        }
    }
}

async fn run_models(action: ModelAction) -> Result<(), String> {
    let config = XencodeConfig::load().unwrap_or_default();
    let mut client = OllamaClient::new(&config.ollama_url, config.response_timeout);
    let llama_client = LlamaCppClient::new(&config.llama_cpp_url, config.response_timeout);

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
                match llama_client.ping().await {
                    Ok(resp_time) => {
                        println!("  Provider:      llama.cpp ({})", config.llama_cpp_url);
                        println!("  Status:        healthy");
                        println!("  Response time: {:.3}s", resp_time);
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

/// Where a fetched model lands by default, under the cache directory xencode
/// already uses for its own state.
fn default_gguf_path(file_name: &str) -> String {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".xencode")
        .join("models")
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

    let (command, path, file, json) = match action {
        HistoryAction::Status { path, file, json } => ("status", path, file, json),
        HistoryAction::Setup { path, file, json } => ("setup", path, file, json),
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
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".xencode")
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
            let mut config = XencodeConfig::load().unwrap_or_default();
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
            let mut config = XencodeConfig::load().unwrap_or_default();
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
            )
            .await
        }
        ColabAction::Status => run_colab_status_cli().await,
        ColabAction::Down => run_colab_down_cli().await,
    }
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

/// Where the advisory corpora live. `--dir` exists so a corpus can be kept on a
/// shared or offline path without touching the config directory.
fn advisory_corpus(dir: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(dir) = dir {
        return Ok(dir);
    }
    let config_dir = XencodeConfig::config_dir().map_err(|e| e.to_string())?;
    Ok(xencode_analysis_rs::advisories::corpus_dir(&config_dir))
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
    format: QueryFormat,
) -> Result<(), String> {
    let ndjson = format == QueryFormat::Ndjson;
    // Read before the cache is consulted or anything is printed: a schema that
    // is not JSON is the caller's mistake, and a schema that is one is a promise
    // about the answer.
    let schema = parse_json_schema(json_schema)?;
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
        openrouter_key: config.api_keys.openrouter_api_key.is_some(),
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
        "retrieval: up to {} files, {} characters each (character arithmetic, not measured)",
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
        state_md: live.state_md.as_deref(),
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
    let context_messages: Vec<ChatMessage> = assembly
        .turns
        .into_iter()
        .map(|t| ChatMessage {
            role: t.role,
            content: t.content.into(),
        })
        .collect();

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
        config.api_keys.openrouter_api_key.clone(),
        config.api_keys.qwen_api_key.clone(),
        config.api_keys.google_gemini_api_key.clone(),
        None,
    )
    .with_llama_cpp(llama_client)
    .with_remote(
        &config.remote_base_url,
        config.api_keys.remote_api_key.clone(),
    )
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
    let mem = ConversationMemory::with_persistence(50).map_err(|e| e.to_string())?;

    match action {
        MemoryAction::List => {
            let sessions = mem.list_sessions();
            if sessions.is_empty() {
                println!("No conversation sessions found.");
            } else {
                println!("Conversation Sessions:");
                for s in sessions {
                    println!("  {}", s);
                }
            }
            Ok(())
        }
        MemoryAction::Show { session } => {
            if let Some(sess) = mem.get_session(&session) {
                for msg in &sess.messages {
                    let role = msg.role.to_uppercase();
                    println!("[{}] {}", role, msg.timestamp);
                    println!("{}\n", msg.content);
                }
            } else {
                println!("Session not found: {}", session);
            }
            Ok(())
        }
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
/// is taken verbatim, and the default is the user-level config directory
/// (`~/.xencode/audit.jsonl`) — sessions are not repo-scoped, so neither is
/// the trail.
fn resolve_audit_path(audit_path: Option<&str>) -> Result<Option<PathBuf>, String> {
    match audit_path {
        Some("none") => Ok(None),
        Some(p) => Ok(Some(PathBuf::from(p))),
        None => Ok(Some(
            XencodeConfig::config_dir()
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
pub const FETCH_TEXT_CAP_CHARS: usize = 30_000;

/// One-screen research summary for a fetched page. Pure — unit-tested.
fn format_fetch_text(page: &FetchedPage) -> String {
    let title = page.title.as_deref().unwrap_or("(no title)");
    let body = if page.text.chars().count() > FETCH_TEXT_CAP_CHARS {
        let kept: String = page.text.chars().take(FETCH_TEXT_CAP_CHARS).collect();
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

/// PR-level triage view: per-file stats plus issue counts. Pure — unit-tested.
fn format_review_text(base: &str, files: &[ReviewedFile]) -> String {
    let mut out = format!("Review of diff {base}...HEAD ({} files)\n", files.len());
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

/// The `AR-1` interop probe.
///
/// The scratch fixture is built in a temporary directory and removed afterwards,
/// so running the probe never leaves anything in the workspace and never touches
/// a file the operator cares about.
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

fn run_interop(
    agents: Vec<String>,
    timeout: u64,
    out: Option<std::path::PathBuf>,
    format: OutputFormat,
    repeat: u32,
    check_auth: bool,
) -> Result<(), String> {
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
    };
    let report = xencode_agents_rs::probe::run_probe(&options);

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
                println!("\n  not installed here:");
                for absent in &report.absent {
                    println!("    {} — {}", absent.name, absent.why);
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
    Ok(())
}

fn run_review(base: String, format: OutputFormat) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let root = resolve_review_root(&cwd);
    let diffs = xencode_context_rs::git_diff_numstat(&root, &base)?;
    let files: Vec<ReviewedFile> = diffs.iter().map(|d| review_file(&root, d)).collect();
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
            let output = serde_json::json!({
                "base": base,
                "files_changed": files.len(),
                "files": arr,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?
            );
        }
        OutputFormat::Text => {
            println!("{}", format_review_text(&base, &files));
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
                    eprintln!("  Security: {} issues in {}", findings.len(), fp.display());
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
                println!("Use 'xencode plugin install <path>' to install a plugin.");
            } else {
                println!(
                    "📦 Plugins in {} (xencode {version}):",
                    plugin_dir.display()
                );
                for report in runtime.reports() {
                    println!("  {}", report.summary());
                }
                println!(
                    "  {} of {} loaded — a loaded plugin's prompt prefix and hooks apply to every agent turn.",
                    runtime.loaded_count(),
                    runtime.reports().len()
                );
            }
        }
        PluginAction::Install { path } => {
            if !path.exists() {
                return Err(format!("Path does not exist: {}", path.display()));
            }
            // Copy plugin directory to plugin folder
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("plugin");
            let dest = plugin_dir.join(name);
            if dest.exists() {
                return Err(format!("Plugin '{}' is already installed", name));
            }
            std::fs::create_dir_all(&plugin_dir).map_err(|e| e.to_string())?;

            if path.is_dir() {
                copy_dir_recursive(&path, &dest)
                    .map_err(|e| format!("Failed to install: {}", e))?;
            } else {
                std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
                std::fs::copy(&path, dest.join(path.file_name().unwrap()))
                    .map_err(|e| e.to_string())?;
            }
            println!("✅ Plugin '{name}' installed to {}.", dest.display());
            // Say right away whether the copied files are a plugin this build
            // can load, rather than letting the user find out by silence.
            let runtime = PluginRuntime::load(&plugin_dir, version);
            match runtime.reports().iter().find(|r| r.name == name) {
                Some(report) => println!("   {}", report.summary()),
                None => println!(
                    "   NOT LOADED: no plugin.json or manifest.json declaring this name — the files were copied but nothing loads from them."
                ),
            }
        }
        PluginAction::Remove { name } => {
            // The name comes from the command line, so it passes the same guard
            // a manifest name does: one normal path component, nothing that can
            // walk out of the plugin directory.
            let path = PluginRegistry::new(plugin_dir.clone())
                .plugin_path(&name)
                .ok_or_else(|| format!("Invalid plugin name: {name}"))?;
            if path.exists() {
                std::fs::remove_dir_all(&path).map_err(|e| format!("Failed to remove: {}", e))?;
                println!("✅ Plugin '{name}' removed.");
            } else {
                return Err(format!("Plugin '{name}' not found"));
            }
        }
    }
    Ok(())
}

fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest_path = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else {
            std::fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

async fn run_tui() -> Result<(), String> {
    // A panic from here on would otherwise leave raw mode and the alternate
    // screen switched on, hiding its own message: restore the terminal first
    // and record the crash where `xencode doctor` can find it.
    if let Some(record) = xencode_tui_rs::panic::default_record_path() {
        xencode_tui_rs::panic::install_panic_hook(record);
    }
    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;
    let mut stdout = io::stdout();
    crossterm::execute!(
        stdout,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )
    .map_err(|e| e.to_string())?;

    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = ratatui::Terminal::new(backend).map_err(|e| e.to_string())?;

    let res = xencode_tui_rs::run_app(&mut terminal).await;

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
        compute_advise, format_image_text, parse_comma_list, resolve_audit_path, resolve_bind,
    };
    use xencode_analysis_rs::images::{ImageFormat, ImageMeta};

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
        // The default lands in the user config dir, never the repo.
        let default = resolve_audit_path(None).unwrap().unwrap();
        assert_eq!(default.file_name().unwrap(), "audit.jsonl");
        assert_eq!(
            default.parent().unwrap(),
            xencode_config_rs::XencodeConfig::config_dir().unwrap()
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
        let out = super::format_review_text("main", &files);
        assert!(
            out.contains("Review of diff main...HEAD (2 files)"),
            "{out}"
        );
        assert!(out.contains("src/a.rs (+10 -2)"), "{out}");
        assert!(out.contains("assets/logo.png (binary)"), "{out}");
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
}
