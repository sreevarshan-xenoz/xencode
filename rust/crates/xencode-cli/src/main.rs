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
                // The conversation screen (TD-1, TD-2). Each takes a short
                // list of words, so a typo is refused rather than stored.
                "tool_output" => {
                    let word = value.trim().to_ascii_lowercase();
                    if !matches!(word.as_str(), "fold" | "expand") {
                        return Err(format!(
                            "tool_output must be \"fold\" or \"expand\", not \"{value}\""
                        ));
                    }
                    config.tool_output = word;
                }
                "side_panel" => {
                    let word = value.trim().to_ascii_lowercase();
                    if !matches!(word.as_str(), "auto" | "on" | "off") {
                        return Err(format!(
                            "side_panel must be \"auto\", \"on\" or \"off\", not \"{value}\""
                        ));
                    }
                    config.side_panel = word;
                }
                "status_line" => {
                    const ITEMS: [&str; 6] =
                        ["model", "provider", "approval", "context", "cost", "branch"];
                    let items: Vec<String> = value
                        .split(',')
                        .map(|item| item.trim().to_ascii_lowercase())
                        .filter(|item| !item.is_empty())
                        .collect();
                    if let Some(bad) = items.iter().find(|item| !ITEMS.contains(&item.as_str())) {
                        return Err(format!(
                            "status_line items are {}; \"{bad}\" is not one",
                            ITEMS.join(", ")
                        ));
                    }
                    config.status_line = items;
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
