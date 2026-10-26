//! the cache, eval, audit, deps and advisories subcommands
//! `xencode cache`, `eval`, `audit`, `deps` and `advisories`.

use super::*;

pub(crate) fn run_cache(action: CacheAction) -> Result<(), String> {
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
pub(crate) fn megabytes_label(bytes: u64) -> String {
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
pub(crate) async fn run_eval(action: EvalAction) -> Result<(), String> {
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
pub(crate) fn format_age_minutes(minutes: u64) -> String {
    match minutes {
        0 => "now".to_string(),
        m if m < 60 => format!("{m}m ago"),
        m if m < 60 * 24 => format!("{}h ago", m / 60),
        m => format!("{}d ago", m / (60 * 24)),
    }
}

/// Check the session server's audit log for records that were edited after
/// the fact. Exits non-zero when something does not add up.
pub(crate) fn run_audit(action: AuditAction) -> Result<(), String> {
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
pub(crate) fn advisory_corpus(dir: Option<PathBuf>) -> Result<PathBuf, String> {
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
pub(crate) fn run_deps(path: PathBuf, format: OutputFormat) -> Result<(), String> {
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

pub(crate) async fn run_advisories(action: AdvisoryAction) -> Result<(), String> {
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
