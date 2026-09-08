use std::io::{self, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use std::sync::Arc;
use xencode_analysis_rs::analyzer::CodeAnalyzer;
use xencode_analysis_rs::security::VulnerabilityScanner;
use xencode_cache_rs::ResponseCache;
use xencode_config_rs::XencodeConfig;
use xencode_core_rs::{scan_workspace, ScanOptions};
use xencode_memory_rs::ConversationMemory;
use xencode_models_rs::OllamaClient;
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
    #[command(subcommand)]
    command: Commands,
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
    },

    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Ollama model management
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
    },

    /// Manage conversation memory
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },

    /// Start the collaboration server
    Server {
        /// Port to listen on
        #[arg(long, default_value = "8765")]
        port: u16,
    },

    /// Analyze code for issues and vulnerabilities
    Analyze {
        /// Path to analyze (file or directory)
        path: std::path::PathBuf,

        /// Output format
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },

    /// Manage plugins
    Plugin {
        #[command(subcommand)]
        action: PluginAction,
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

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Scan {
            path,
            hidden,
            max_depth,
        } => run_scan(path, hidden, max_depth),
        Commands::Config { action } => run_config(action),
        Commands::Models { action } => run_models(action).await,
        Commands::Cache { action } => run_cache(action),
        Commands::Query {
            prompt,
            model,
            no_cache,
            session,
        } => run_query(prompt, model, no_cache, session).await,
        Commands::Memory { action } => run_memory(action),
        Commands::Server { port } => run_server(port).await,
        Commands::Analyze { path, format } => run_analyze(path, format),
        Commands::Plugin { action } => run_plugin_action(action),
        Commands::Tui => run_tui().await,
    };

    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run_scan(root: PathBuf, include_hidden: bool, max_depth: Option<usize>) -> Result<(), String> {
    let options = ScanOptions {
        max_depth,
        include_hidden,
        ..ScanOptions::default()
    };

    let entries = scan_workspace(root, &options).map_err(|e| e.to_string())?;
    for entry in entries {
        let bytes = entry
            .bytes
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string());
        println!("{}\t{}\t{}", entry.kind, bytes, entry.path.display());
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
    let mut client = OllamaClient::default_client();

    match action {
        ModelAction::List => {
            let models = client.list_models().await.map_err(|e| e.to_string())?;
            if models.is_empty() {
                println!("No models installed.");
                println!("Install one with: ollama pull qwen2.5:7b");
                return Ok(());
            }
            println!("{:<30} {:>12} MODIFIED", "MODEL", "SIZE");
            println!("{}", "-".repeat(60));
            for model in &models {
                let size_mb = model.size as f64 / 1_048_576.0;
                let modified = if model.modified_at.len() > 19 {
                    &model.modified_at[..19]
                } else {
                    &model.modified_at
                };
                println!("{:<30} {:>9.1} MB {}", model.name, size_mb, modified);
            }
            println!("\n{} model(s) installed", models.len());
            Ok(())
        }
        ModelAction::Health { model } => {
            println!("Checking health of {model}...");
            let health = client
                .check_health(&model)
                .await
                .map_err(|e| e.to_string())?;
            println!("  Status:        {}", health.status);
            println!("  Response time: {:.3}s", health.response_time);
            if let Some(ref err) = health.error_message {
                println!("  Error:         {err}");
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

async fn run_query(
    prompt: String,
    model_override: Option<String>,
    no_cache: bool,
    session_id: Option<String>,
) -> Result<(), String> {
    let config = XencodeConfig::load().unwrap_or_default();
    let model = model_override.unwrap_or(config.default_model);

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

    let mut context_messages = Vec::new();
    if let Some(ref mem) = memory {
        for msg in mem.get_context(10) {
            context_messages.push(ChatMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            });
        }
    }

    context_messages.push(ChatMessage {
        role: "user".to_string(),
        content: prompt.clone(),
    });

    let client = OllamaClient::new(&config.ollama_url, config.response_timeout);
    let provider = ProviderManager::new(
        client,
        config.api_keys.openrouter_api_key.clone(),
        config.api_keys.qwen_api_key.clone(),
        config.api_keys.google_gemini_api_key.clone(),
        None,
    );

    let mut response_content = String::new();
    let result = provider
        .generate_stream(&model, &context_messages, |token| {
            print!("{}", token);
            let _ = io::stdout().flush();
            response_content.push_str(token);
        })
        .await;

    println!(); // Ensure final newline

    match result {
        Ok(_) => {
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

async fn run_server(port: u16) -> Result<(), String> {
    use std::net::SocketAddr;
    let state = Arc::new(ServerState::new());
    let app = xencode_server_rs::build_app_with_state(state.clone());
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("🚀 Xencode server starting on http://0.0.0.0:{}", port);
    println!(
        "   WebSocket: ws://0.0.0.0:{}/ws/{{session_id}}/{{username}}",
        port
    );
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| e.to_string())?;
    axum::serve(listener, app)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn run_analyze(path: std::path::PathBuf, format: OutputFormat) -> Result<(), String> {
    if path.is_dir() {
        // Scan directory
        let mut all_issue_lists = Vec::new();
        let walker = walkdir::WalkDir::new(&path)
            .max_depth(3)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file());

        for entry in walker {
            let fp = entry.path();
            let ext = fp.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext {
                "py" | "rs" | "ts" | "js" | "tsx" | "jsx" | "go" | "rb" | "java" => {
                    match CodeAnalyzer::analyze_file(fp) {
                        Ok(issues) => {
                            all_issue_lists.push((fp.display().to_string(), issues));
                        }
                        Err(e) => eprintln!("  Skipping {}: {}", fp.display(), e),
                    }
                    // Run security scan
                    if let Ok(findings) = VulnerabilityScanner::scan_file(fp) {
                        if !findings.is_empty() {
                            println!("  Security: {} issues in {}", findings.len(), fp.display());
                        }
                    }
                }
                _ => {}
            }
        }

        match format {
            OutputFormat::Json => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&all_issue_lists).map_err(|e| e.to_string())?
                );
            }
            OutputFormat::Text => {
                let total: usize = all_issue_lists.iter().map(|(_, issues)| issues.len()).sum();
                println!("Results for: {}", path.display());
                println!("   Files analyzed: {}", all_issue_lists.len());
                println!("   Total issues:   {}", total);
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
        // Single file
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
            std::fs::copy(&entry.path(), &dest_path)?;
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
