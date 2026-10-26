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
