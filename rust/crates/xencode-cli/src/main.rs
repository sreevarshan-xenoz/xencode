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
use xencode_models_rs::{
    find_llama_server, start_llama_server, LlamaCppClient, LlamaCppOptions, OllamaClient,
};
use xencode_plugin_rs::PluginRegistry;
use xencode_providers_rs::{ChatMessage, ProviderManager};
use xencode_server_rs::ws::AppState as ServerState;

/// Output format for analysis results
#[derive(clap::ValueEnum, Clone)]
enum OutputFormat {
    Text,
    Json,
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

        /// llama.cpp sampling: max generated tokens
        #[arg(long = "max-tokens")]
        max_tokens: Option<u32>,

        /// llama.cpp sampling: GBNF grammar file/string
        #[arg(long)]
        grammar: Option<String>,

        /// llama.cpp sampling: JSON schema for structured output
        #[arg(long = "json-schema")]
        json_schema: Option<String>,
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
    },

    /// Fetch a web page and extract research-ready text
    Fetch {
        /// URL to fetch (http/https only)
        url: String,

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

    /// Launch the Terminal User Interface
    Tui,
}

#[derive(Subcommand)]
enum PluginAction {
    /// List installed plugins
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
        /// Value to set
        value: String,
    },
    /// Reset configuration to defaults
    Reset,
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
enum CacheAction {
    /// Show cache statistics
    Stats,
    /// Clear all cached responses
    Clear,
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
        Commands::Query {
            prompt,
            model,
            no_cache,
            session,
            temperature,
            top_k,
            min_p,
            mirostat,
            max_tokens,
            grammar,
            json_schema,
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
                max_tokens,
                grammar,
                json_schema,
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
        Commands::Analyze { path, format } => run_analyze(path, format),
        Commands::Fetch { url, format } => run_fetch(url, format).await,
        Commands::Review { base, format } => run_review(base, format),
        Commands::Plugin { action } => run_plugin_action(action),
        Commands::Llamacpp { action } => run_llamacpp(action).await,
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
            match key.as_str() {
                "default_model" => config.default_model = value.clone(),
                "ollama_url" => config.ollama_url = value.clone(),
                "llama_cpp_url" => config.llama_cpp_url = value.clone(),
                "llama_cpp_model_path" => config.llama_cpp_model_path = value.clone(),
                "llama_cpp_executable" => config.llama_cpp_executable = value.clone(),
                "llama_cpp_args" => {
                    config.llama_cpp_args =
                        value.split_whitespace().map(|s| s.to_string()).collect();
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
                _ => return Err(format!("unknown config key: {key}")),
            }
            config.save().map_err(|e| e.to_string())?;
            println!("set {key} = {value}");
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
    }
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

            let extra: Vec<&str> = config.llama_cpp_args.iter().map(|s| s.as_str()).collect();
            let mut server =
                start_llama_server(&exe, &model_path, port, &extra).map_err(|e| e.to_string())?;

            // Save PID so `xencode llamacpp stop` can terminate it later.
            let _ = write_pid_file(&pid_file(), server.pid());

            // Wait for the server to become healthy (ping until success).
            let client = LlamaCppClient::new(&server.base_url, 5);
            let mut ready = false;
            for _ in 0..120 {
                if client.ping().await.is_ok() {
                    ready = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            if !ready {
                let _ = server.stop();
                let _ = std::fs::remove_file(pid_file());
                return Err("llama-server did not become ready in time".to_string());
            }

            config.llama_cpp_url = server.base_url.clone();
            config.llama_cpp_model_path = model_path.clone();
            config.save().map_err(|e| e.to_string())?;

            println!("llama-server ready at {}", server.base_url);
            println!("Model loaded: {}", model_path);
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
    max_tokens: Option<u32>,
    grammar: Option<String>,
    json_schema: Option<String>,
) -> Result<(), String> {
    let config = XencodeConfig::load().unwrap_or_default();
    let client = OllamaClient::new(&config.ollama_url, config.response_timeout);

    // If no model override is provided, verify default model against Ollama's installed models
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

    if let Some(ref mut c) = cache {
        if let Some(cached_resp) = c.get(&prompt, &model) {
            println!("{}", cached_resp);

            if let Some(ref mut mem) = memory {
                mem.add_message("user", &prompt, None);
                mem.add_message("assistant", &cached_resp, Some(model.clone()));
            }
            return Ok(());
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
    let live = xencode_context_rs::collect_live_context(
        &root,
        &prompt,
        xencode_context_rs::HardwareProfile::Balanced,
    );
    // The model's real window when known (Step 3 capabilities); unknown
    // routes defer to the profile default.
    let context_window = xencode_providers_rs::capabilities_for(&model).context_window;
    let assembly = xencode_context_rs::assemble_chat(xencode_context_rs::ChatInput {
        profile: xencode_context_rs::HardwareProfile::Balanced,
        context_window,
        system: xencode_context_rs::AGENT_SYSTEM_PROMPT,
        agents_md: live.agents_md.as_deref(),
        anchor_md: live.anchor_md.as_deref(),
        state_md: live.state_md.as_deref(),
        git_summary: &live.git_summary,
        retrieved: live.blocks,
        attached_block: "",
        history: &history,
        prompt: &prompt,
    });
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
        grammar,
        json_schema: parse_json_schema(json_schema)?,
    };

    let provider = ProviderManager::new(
        client,
        config.api_keys.openrouter_api_key.clone(),
        config.api_keys.qwen_api_key.clone(),
        config.api_keys.google_gemini_api_key.clone(),
        None,
    )
    .with_llama_cpp(llama_client)
    .with_request_timeout(config.response_timeout);

    let mut response_content = String::new();
    let result = provider
        .generate_stream_with_options(&model, &context_messages, Some(&llamacpp_opts), |token| {
            print!("{}", token);
            let _ = io::stdout().flush();
            response_content.push_str(token);
        })
        .await;

    println!(); // Ensure final newline

    match result {
        Ok(_) => {
            if let Some(timings) = provider.last_llamacpp_timings() {
                println!(
                    "\n(llama.cpp ~{} tok/s · {} tokens generated)",
                    timings.predicted_per_second as u64, timings.tokens_generated
                );
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

fn run_analyze(path: std::path::PathBuf, format: OutputFormat) -> Result<(), String> {
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
    let plugin_dir = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("xencode")
        .join("plugins");

    match action {
        PluginAction::List => {
            let registry = PluginRegistry::new(std::path::PathBuf::from(&plugin_dir));
            let manifests = registry.discover();
            if manifests.is_empty() {
                println!("No plugins installed in: {}", plugin_dir.display());
                println!("Use 'xencode plugin install <path>' to install a plugin.");
            } else {
                println!("📦 Installed Plugins (from {}):", plugin_dir.display());
                for m in &manifests {
                    println!(
                        "  {} v{} — {} (by {})",
                        m.name, m.version, m.description, m.author
                    );
                }
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
            println!("✅ Plugin '{}' installed successfully.", name);
        }
        PluginAction::Remove { name } => {
            let path = plugin_dir.join(&name);
            if path.exists() {
                std::fs::remove_dir_all(&path).map_err(|e| format!("Failed to remove: {}", e))?;
                println!("✅ Plugin '{}' removed.", name);
            } else {
                return Err(format!("Plugin '{}' not found", name));
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
    use super::{compute_advise, format_image_text, resolve_audit_path, resolve_bind};
    use xencode_analysis_rs::images::{ImageFormat, ImageMeta};

    fn path(p: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(p)
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
            .any(|i| i.kind == AdviceKind::Orphan && i.file == "src/c.rs"));

        let filtered = compute_advise(&root, Some("c.rs")).unwrap();
        assert!(!filtered.is_empty());
        assert!(filtered.iter().all(|i| i.file.contains("c.rs")));
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
}
