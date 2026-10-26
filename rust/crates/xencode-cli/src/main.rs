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

mod orchestrator;
mod query_stream;
use orchestrator::*;
mod cmd_config;
use cmd_config::*;
mod cmd_models;
use cmd_models::*;
mod cmd_remote;
use cmd_remote::*;
mod cmd_team;
use cmd_team::*;
mod cmd_merge;
use cmd_merge::*;
mod cmd_maintenance;
use cmd_maintenance::*;
mod cmd_query;
use cmd_query::*;
mod cmd_memory;
use cmd_memory::*;
mod cmd_serve;
use cmd_serve::*;
mod cmd_doctor;
use cmd_doctor::*;
mod cmd_agents;
use cmd_agents::*;
mod cmd_quality;
use cmd_quality::*;
mod cmd_repo;
use cmd_repo::*;
mod cmd_runs;
use cmd_runs::*;
mod cmd_extensions;
use cmd_extensions::*;
mod cli;
use cli::*;

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
        crossterm::event::DisableMouseCapture,
        crossterm::event::DisableFocusChange,
    )
    .map_err(|e| e.to_string())?;
    terminal.show_cursor().map_err(|e| e.to_string())?;

    res.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests;
