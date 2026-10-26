//! The command line's definition: the top-level `Cli`, every subcommand in
//! `Commands`, and the action enums of the subcommands that have their own.

use super::*;

/// Xencode — AI development assistant (Rust core)
#[derive(Parser)]
#[command(name = "xencode", version, about, long_about = None)]
pub(crate) struct Cli {
    /// Dump the engine composition and resolved configuration as JSON (AF-3)
    #[arg(long)]
    pub(crate) dump_config: bool,

    /// Defaults to the TUI when omitted
    #[command(subcommand)]
    pub(crate) command: Option<Commands>,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
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
pub(crate) enum EvalAction {
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
pub(crate) enum PluginAction {
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
pub(crate) enum McpAction {
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
pub(crate) enum ConfigAction {
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
pub(crate) enum ColabAction {
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
pub(crate) enum RemoteAction {
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
pub(crate) enum ComputersAction {
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
pub(crate) enum TeamAction {
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

    /// Run an outside worker agent's program from `--cwd` with the named
    /// environment variables removed. The engine starts outside agents this
    /// way, so an adapter never reads the worker's tree and never sees
    /// another vendor's key
    #[command(hide = true)]
    Exec {
        /// The folder to run the program in
        #[arg(long)]
        cwd: PathBuf,
        /// An environment variable to remove; repeat for each
        #[arg(long)]
        unset: Vec<String>,
        /// The program and its arguments, after `--`
        #[arg(last = true, required = true)]
        command: Vec<String>,
    },
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
pub(crate) enum OrchestratorAction {
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
pub(crate) enum CompeteAction {
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
pub(crate) enum MergeAction {
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
pub(crate) enum ModelAction {
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
pub(crate) enum LlamacppAction {
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
pub(crate) enum HwAction {
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
pub(crate) enum SessionAction {
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
pub(crate) enum HistoryAction {
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
pub(crate) enum CacheAction {
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
pub(crate) enum AuditAction {
    /// Check an audit log for records that were changed after they were written
    Verify {
        /// Log to check (default: <state dir>/audit.jsonl)
        path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
pub(crate) enum AdvisoryAction {
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
pub(crate) enum MemoryAction {
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
pub(crate) enum MemoryPolicyAction {
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
pub(crate) enum PerfAction {
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
pub(crate) enum PriceAction {
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
pub(crate) enum RunsAction {
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
pub(crate) enum WorktreeAction {
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
pub(crate) enum TaskAction {
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
