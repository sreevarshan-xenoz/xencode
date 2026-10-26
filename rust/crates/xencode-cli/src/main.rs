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

/// `xencode team exec` (TM-5, hidden): run an outside agent's program in
/// `cwd` without the `unset` variables, its standard streams the agent's
/// protocol, and end with its exit code.
fn team_exec(cwd: &std::path::Path, unset: &[String], command: &[String]) -> Result<(), String> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| "no program to run".to_string())?;
    let mut cmd = std::process::Command::new(program);
    cmd.args(args).current_dir(cwd);
    for var in unset {
        cmd.env_remove(var);
    }
    let status = cmd
        .status()
        .map_err(|e| format!("could not start {program}: {e}"))?;
    std::process::exit(status.code().unwrap_or(1));
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
        TeamAction::Exec {
            cwd,
            unset,
            command,
        } => return team_exec(&cwd, &unset, &command),
        other => other,
    };
    let dir = team_recipes_dir();
    let files = xencode_core_rs::load_recipes(&dir)
        .map_err(|e| format!("cannot read the team recipes in {}: {e}", dir.display()))?;
    match action {
        TeamAction::LoginOptin { agent } => team_login_optin(&agent),
        TeamAction::Clean => team_clean().await,
        TeamAction::Exec {
            cwd,
            unset,
            command,
        } => team_exec(&cwd, &unset, &command),
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
