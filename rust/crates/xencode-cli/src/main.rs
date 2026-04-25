use std::path::PathBuf;
use std::io::{self, Write};

use clap::{Parser, Subcommand};

use xencode_cache_rs::ResponseCache;
use xencode_config_rs::XencodeConfig;
use xencode_core_rs::{scan_workspace, ScanOptions};
use xencode_memory_rs::ConversationMemory;
use xencode_models_rs::OllamaClient;
use xencode_providers_rs::{ChatMessage, ProviderManager};

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

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Scan {
            path,
            hidden,
            max_depth,
        } => run_scan(path, hidden, max_depth),
        Commands::Config { action } => run_config(action),
        Commands::Models { action } => run_models(action),
        Commands::Cache { action } => run_cache(action),
        Commands::Query { prompt, model, no_cache, session } => run_query(prompt, model, no_cache, session),
        Commands::Memory { action } => run_memory(action),
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
) -> Result<(), String> {
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

fn run_models(action: ModelAction) -> Result<(), String> {
    let mut client = OllamaClient::default_client();

    match action {
        ModelAction::List => {
            let models = client.list_models().map_err(|e| e.to_string())?;
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
            let health = client.check_health(&model).map_err(|e| e.to_string())?;
            println!("  Status:        {}", health.status);
            println!("  Response time: {:.3}s", health.response_time);
            if let Some(ref err) = health.error_message {
                println!("  Error:         {err}");
            }
            Ok(())
        }
        ModelAction::Default => {
            let default = client.get_smart_default().map_err(|e| e.to_string())?;
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

fn run_query(
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
    let manager = ProviderManager::new(client);

    let mut response_content = String::new();
    let result = manager.generate_stream(&model, &context_messages, |token| {
        print!("{}", token);
        let _ = io::stdout().flush();
        response_content.push_str(token);
    });

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
