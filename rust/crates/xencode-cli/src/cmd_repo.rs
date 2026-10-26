//! the prices, release-notes, test, anchor, interop and review subcommands
//! `xencode prices`, `release-notes`, `test`, `anchor`, `interop` and `review`.

use super::*;

/// `xencode prices`: the two documents a cost report reads its rates out of, what
/// each of them says, and which of the models this project has actually run have
/// no rate in either.
///
/// Nothing here is computed — no spend, no totals. The point of the command is
/// provenance: a number on a cost report is only as good as the paper it came
/// from, and one of those two papers is somebody else's catalogue read on a day
/// that has already passed.
pub(crate) async fn run_prices(action: Option<PriceAction>) -> Result<(), String> {
    let xencode_dir = project_xencode_dir();
    let config = XencodeConfig::load().unwrap_or_default();
    let action = action.unwrap_or(PriceAction::Show {
        format: OutputFormat::Text,
    });

    match action {
        PriceAction::Fetch { url } => {
            let url = url
                .as_deref()
                .unwrap_or(xencode_providers_rs::listing::OPENROUTER_MODELS_URL);
            let body = xencode_providers_rs::listing::fetch_listing(url)
                .await
                .map_err(|e| e.to_string())?;
            let now = xencode_context_rs::conversation::now_millis();
            let lookup = xencode_context_rs::PriceLookup::parse_openrouter(&body, now)?;
            let path = lookup.write(&xencode_dir).map_err(|e| e.to_string())?;
            println!(
                "{} prices read off {url} and written to {}",
                lookup.priced_models(),
                path.display()
            );
            if lookup.unreadable > 0 {
                println!(
                    "  {} entries the listing gave in a shape no price could be read out of, counted and left out",
                    lookup.unreadable
                );
            }
            if !config.price_lookup {
                println!(
                    "note: price_lookup is off, so nothing is read from that file yet — `xencode config set price_lookup true`"
                );
            }
            println!(
                "note: a report reads that file for {} days, then stops pricing from it until it is fetched again.",
                xencode_context_rs::PRICE_TTL_DAYS
            );
        }
        PriceAction::Show { format } => {
            let table =
                xencode_context_rs::PriceTable::load_with_lookup(&xencode_dir, config.price_lookup);
            // Folded here rather than read from the rollup file: refreshing that
            // sidecar means writing to somebody's disk to answer a question about
            // it, and a project driven only through this command has never written
            // one. The records are the same ones the rollup is built from.
            let mut rollup = xencode_context_rs::MetricsRollup::empty();
            for row in xencode_context_rs::read_metrics(&xencode_dir) {
                rollup.fold(&row);
            }
            let run: Vec<String> = rollup.by_model.keys().cloned().collect();
            let unpriced: Vec<&String> = run
                .iter()
                .filter(|model| table.price_for_model(model.as_str()).is_none())
                .collect();
            let from_listing: Vec<&String> = run
                .iter()
                .filter(|model| {
                    matches!(
                        table.price_for_model(model.as_str()),
                        Some(xencode_context_rs::PriceSource::Listing { .. })
                    )
                })
                .collect();

            // Read off the disk rather than through the table, so the answer does
            // not depend on whether this build is allowed to price anything from
            // it — the question here is what the file says.
            let listing = xencode_context_rs::PriceLookup::load(&xencode_dir);
            let now = xencode_context_rs::conversation::now_millis();

            if matches!(format, OutputFormat::Json) {
                let line = serde_json::json!({
                    "pricing_json": {
                        "path": table.path.display().to_string(),
                        "present": table.file_present,
                        "models": &table.models,
                        "rejected": &table.rejected,
                    },
                    "listing": listing.as_ref().map(|lookup| serde_json::json!({
                        "source": &lookup.source,
                        "fetched_at_unix_ms": lookup.fetched_at_unix_ms,
                        "fetched_on": xencode_context_rs::local_day_key(lookup.fetched_at_unix_ms),
                        "age_days": lookup.age_days(now),
                        "expired": lookup.stale(now),
                        "priced_models": lookup.priced_models(),
                        "unreadable": lookup.unreadable,
                    })),
                    "price_lookup": config.price_lookup,
                    "listing_is_read": table.lookup.is_some() && !table.listing_expired,
                    "models_run": &run,
                    "priced_from_listing": &from_listing,
                    "unpriced": &unpriced,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&line).unwrap_or_default()
                );
                return Ok(());
            }

            println!("pricing.json — {}", table.path.display());
            if !table.file_present {
                println!("  nothing there yet, so no model is priced by hand");
            } else {
                println!(
                    "  {}, priced by hand",
                    plural_count(table.models.len(), "model", "models")
                );
                for (name, price) in &table.models {
                    println!("  • {name}  {}", describe_rates(price));
                }
            }
            for problem in &table.rejected {
                println!("  could not be read: {problem}");
            }

            println!(
                "fetched listing — {}",
                xencode_context_rs::lookup_path(&xencode_dir).display()
            );
            match &listing {
                Some(lookup) => {
                    println!(
                        "  {} prices off {}, read on {}, {} days ago",
                        lookup.priced_models(),
                        lookup.source,
                        xencode_context_rs::local_day_key(lookup.fetched_at_unix_ms),
                        lookup.age_days(now),
                    );
                    if lookup.unreadable > 0 {
                        println!("  {} entries carried no readable price", lookup.unreadable);
                    }
                    if !config.price_lookup {
                        println!(
                            "  not consulted: price_lookup is off. `xencode config set price_lookup true` lets a model missing from pricing.json be priced from here; a rate written by hand always outranks one read off a listing."
                        );
                    } else if let Some(note) = table.listing_expired_note(now) {
                        println!("{note}");
                    }
                }
                None => println!(
                    "  nothing fetched. `xencode prices fetch` reads the public catalogue once and keeps it here."
                ),
            }

            if run.is_empty() {
                println!("models this project has run: none recorded yet");
            } else {
                println!(
                    "models this project has run: {}; priced from the listing: {}; with no price in either document: {}",
                    run.len(),
                    from_listing.len(),
                    unpriced.len()
                );
                // Which name each looked-up price was matched under, because a
                // rate that arrived by matching a different string has to be
                // checkable against the catalogue it came from.
                for model in &from_listing {
                    if let Some(xencode_context_rs::PriceSource::Listing { id, price }) =
                        table.price_for_model(model)
                    {
                        println!(
                            "  • {model} — the listing's {id}: {}",
                            describe_rates(price)
                        );
                    }
                }
                for model in &unpriced {
                    println!(
                        "  • {model} — no price. A cost is reported as unknown, never as nothing."
                    );
                }
            }
        }
    }
    Ok(())
}

/// A hand-written rate the way it is written into the file, so `prices show` can
/// be read against the editor rather than against a mental conversion.
pub(crate) fn describe_rates(price: &xencode_context_rs::ModelPrice) -> String {
    // `format_usd` already writes the dollar sign, so none is added here.
    let mut line = format!(
        "{} in / {} out per million tokens",
        xencode_context_rs::format_usd((price.input_usd_per_mtok * 1_000_000.0).round() as u64),
        xencode_context_rs::format_usd((price.output_usd_per_mtok * 1_000_000.0).round() as u64),
    );
    line.push_str(&match price.cached_input_usd_per_mtok {
        Some(cached) => format!(
            ", cache reads {}",
            xencode_context_rs::format_usd((cached * 1_000_000.0).round() as u64)
        ),
        None => ", no cache rate — reads billed as input, which is an upper bound".to_string(),
    });
    line
}

/// QO-6 — put the release notes together from the two places this project
/// already writes about what shipped: the commits since the last release, and
/// the changelog's own unreleased block.
///
/// The two disagree in both directions and both disagreements are the product:
/// work that shipped with no entry behind it never reaches a reader, and an entry
/// whose commit sits below the release line would be announced twice. The output
/// is a draft — to standard output, or to a path that must not already exist
/// unless `--force` says otherwise, because a person is meant to edit it after.
pub(crate) fn run_release_notes(
    from: Option<String>,
    to: Option<String>,
    release: Option<String>,
    out: Option<PathBuf>,
    force: bool,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_context_rs::releasenotes;

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let draft = releasenotes::draft(&cwd, from.as_deref(), to.as_deref())?;
    let markdown = releasenotes::to_markdown(&draft, release.as_deref());

    let written = match &out {
        Some(path) => {
            let path = if path.is_absolute() {
                path.clone()
            } else {
                cwd.join(path)
            };
            releasenotes::write_draft(&path, &markdown, force)?;
            Some(path)
        }
        None => None,
    };

    let named = draft.commits.len() - draft.uncovered.len();
    match format {
        OutputFormat::Json => println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "to": draft.to,
                "from": draft.from,
                "range_note": draft.range_note,
                "entries": draft.entries.iter().map(|entry| serde_json::json!({
                    "category": entry.category,
                    "title": entry.title,
                    "ids": entry.ids,
                })).collect::<Vec<_>>(),
                "commits": draft.commits.len(),
                "named_by_an_entry": named,
                "covered_ids": draft.covered_ids(),
                "uncovered": draft.uncovered.iter().map(|commit| serde_json::json!({
                    "hash": releasenotes::short_hash(&commit.hash),
                    "subject": commit.subject,
                    "ids": commit.ids,
                })).collect::<Vec<_>>(),
                "unmatched": draft.unmatched.iter().map(|entry| serde_json::json!({
                    "title": entry.title,
                    "ids": entry.ids,
                })).collect::<Vec<_>>(),
                "out": written.as_ref().map(|path| path.display().to_string()),
            }))
            .map_err(|e| e.to_string())?
        ),
        OutputFormat::Text => {
            match &written {
                Some(path) => {
                    println!("  draft written to {}", path.display());
                    println!("  {}", draft.range_note);
                    println!(
                        "  {}, {} of them named by the changelog's unreleased block ({})",
                        plural_count(draft.commits.len(), "commit", "commits"),
                        named,
                        plural_count(draft.entries.len(), "entry", "entries"),
                    );
                    if !draft.uncovered.is_empty() {
                        println!(
                            "  {}, listed in the draft",
                            plural_count(
                                draft.uncovered.len(),
                                "commit no entry accounts for",
                                "commits no entry accounts for"
                            )
                        );
                    }
                    if !draft.unmatched.is_empty() {
                        println!(
                            "  {}, listed in the draft",
                            plural_count(
                                draft.unmatched.len(),
                                "entry names no commit in the range",
                                "entries name no commit in the range"
                            )
                        );
                    }
                    if draft.uncovered.is_empty() && draft.unmatched.is_empty() {
                        println!("  nothing to reconcile: every commit is explained, every entry matches");
                    }
                }
                None => print!("{markdown}"),
            }
        }
    }
    Ok(())
}

/// Line numbers as compact ranges, so `1,2,3,7,9,10` reads as `1-3, 7, 9-10`.
pub(crate) fn join(lines: &[u32]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut start: Option<u32> = None;
    let mut prev: u32 = 0;
    for &line in lines {
        match start {
            None => start = Some(line),
            Some(_) if line == prev + 1 => {}
            Some(s) => {
                out.push(if s == prev {
                    s.to_string()
                } else {
                    format!("{s}-{prev}")
                });
                start = Some(line);
            }
        }
        prev = line;
    }
    if let Some(s) = start {
        out.push(if s == prev {
            s.to_string()
        } else {
            format!("{s}-{prev}")
        });
    }
    out.join(", ")
}

/// Eight args because the  subcommand grew a second mode ()
/// beside the suite mode, and splitting the function would separate the ledger
/// write the two modes share. The ban stays on everywhere else.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_test(
    packages: Vec<String>,
    retries: u32,
    stress_count: u32,
    timeout: u64,
    isolate: Option<String>,
    base: String,
    repeat: u32,
    format: OutputFormat,
    session: Option<String>,
) -> Result<(), String> {
    use xencode_context_rs::verify;

    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    if let Some(test) = isolate {
        let isolation = verify::isolate(
            &root,
            &test,
            &base,
            repeat,
            std::time::Duration::from_secs(timeout),
        );
        if matches!(format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::json!({
                    "test": isolation.test,
                    "base": isolation.base,
                    "work_fails": isolation.work_fails,
                    "work_runs": isolation.work_runs,
                    "base_fails": isolation.base_fails,
                    "base_runs": isolation.base_runs,
                    "class": isolation.class.label(),
                    "notes": isolation.notes,
                })
            );
        } else {
            println!("\n  {} — {}", isolation.class.label(), isolation.test);
            println!(
                "  base ({}): {}",
                isolation.base,
                match isolation.base_fails {
                    Some(f) => format!("{f}/{} failed", isolation.base_runs),
                    None => "could not run".to_string(),
                }
            );
            println!(
                "  worktree: {}/{} failed",
                isolation.work_fails, isolation.work_runs
            );
            for note in &isolation.notes {
                println!("  note: {note}");
            }
            if isolation.class == verify::FailureClass::PreExisting {
                println!("\n  not yours — do not fix it here");
            }
        }
        return if isolation.class == verify::FailureClass::Introduced {
            Err("the failure was introduced in the working tree".to_string())
        } else {
            Ok(())
        };
    }
    let opts = verify::Options {
        filter: None,
        retries,
        stress: stress_count,
        packages,
        budget: std::time::Duration::from_secs(timeout),
    };
    let outcome = verify::run(&root, &opts);

    // EVd-1's first producer: every run leaves a row — session, class, exit
    // code, what it ran against — whether or not anyone reads it back. The run
    // log goes to the session artifacts, tail-capped, and the ledger row points
    // at it; then old passing sessions are pruned, because a test loop that
    // never prunes fills a disk.
    {
        use xencode_context_rs::{artifacts, ledger};
        let xencode_dir = root.join(xencode_context_rs::XENCODE_DIR);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let mut log = format!(
            "command: {}\nexit: {}\nflaky: {}\nfailed: {}\n",
            outcome.command,
            outcome.exit.unwrap_or(-1),
            outcome.flaky.join(", "),
            outcome.failed.join(", "),
        );
        for note in &outcome.notes {
            log.push_str(&format!("note: {note}\n"));
        }
        let target_session = session.or_else(|| {
            ConversationMemory::with_persistence(50)
                .ok()
                .and_then(|m| m.current_session().cloned())
        });
        let session_tag = target_session.as_deref().unwrap_or("cli");
        let log_ref = artifacts::write_artifact(
            &xencode_dir,
            session_tag,
            &format!("test-{now_ms}.log"),
            &log,
        )
        .ok()
        .map(|p| {
            p.strip_prefix(&root)
                .map(|r| r.to_string_lossy().into_owned())
                .unwrap_or_else(|_| p.display().to_string())
        })
        .unwrap_or_default();
        let _ = artifacts::prune_artifacts(&xencode_dir, artifacts::KEEP_LAST_PASSING);
        let entry = ledger::LedgerEntry {
            ts_unix_ms: now_ms,
            session: Some(session_tag.to_string()),
            run_class: ledger::RunClass::Test,
            exit_code: outcome.exit.unwrap_or(-1),
            subjects: vec![ledger::digest_hex(&outcome.command)],
            log_ref,
            note: String::new(),
        };
        let _ = ledger::append_ledger(&xencode_dir, &entry);
    }

    if matches!(format, OutputFormat::Json) {
        let report = serde_json::json!({
            "ok": outcome.ok,
            "engine": outcome.engine.map(|e| e.label()),
            "command": outcome.command,
            "exit": outcome.exit,
            "flaky": outcome.flaky,
            "failed": outcome.failed,
            "notes": outcome.notes,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else {
        if let Some(engine) = outcome.engine {
            println!("\n  engine: {}", engine.label());
        }
        if !outcome.command.is_empty() {
            println!("  ran:    {}", outcome.command);
        }
        for note in &outcome.notes {
            println!("  note:   {note}");
        }
        if !outcome.flaky.is_empty() {
            println!("\n  FLAKY — passed only on a retry, so not a pass:\n");
            for name in &outcome.flaky {
                println!("    {name}");
            }
        }
        if !outcome.failed.is_empty() {
            println!("\n  FAILED:\n");
            for name in &outcome.failed {
                println!("    {name}");
            }
        }
        println!(
            "\n  {}",
            if outcome.ok {
                "pass"
            } else {
                "not a pass — see above"
            }
        );
    }
    if outcome.ok {
        Ok(())
    } else {
        Err("tests did not pass".to_string())
    }
}

pub(crate) fn run_anchor(
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

    if !dry_run {
        let now_s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let meta = anchor::AnchorMeta {
            proved_at_unix_s: now_s,
            candidates: found,
            verified,
        };
        if let Err(e) = anchor::write_anchor_meta(&root, &meta) {
            eprintln!("  warning: could not write anchor.meta sidecar: {e}");
        }
    }

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

#[allow(clippy::too_many_arguments)] // CLI flags map 1:1 to interop options; a struct would just rename them
pub(crate) fn run_interop(
    agents: Vec<String>,
    timeout: u64,
    out: Option<std::path::PathBuf>,
    format: OutputFormat,
    repeat: u32,
    fan_out: bool,
    check_auth: bool,
    capture_dir: Option<std::path::PathBuf>,
    trace: Option<std::path::PathBuf>,
) -> Result<(), String> {
    // Also read-only and also free: a trace view is a file the operator already
    // paid for, rendered again, so it launches nothing.
    if let Some(dir) = &trace {
        let dirs = xencode_agents_rs::capture::find_captures(dir);
        if dirs.is_empty() {
            return Err(format!(
                "{} holds no capture; point --trace at a directory written by --capture-dir",
                dir.display()
            ));
        }
        for (i, capture_dir) in dirs.iter().enumerate() {
            let capture = xencode_agents_rs::capture::read_capture(capture_dir)
                .map_err(|e| format!("cannot read that capture: {e}"))?;
            if i > 0 {
                println!("\n{}\n", "─".repeat(60));
            }
            println!("{}", capture.trace());
        }
        return Ok(());
    }
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
        fan_out,
    };
    let report = xencode_agents_rs::probe::run_probe(&options);

    // The operator may keep each whole run, not only the lines the report chose
    // to show. This writes from the bytes the probe already received, so keeping
    // a capture costs no extra run of anything.
    let kept: Vec<std::path::PathBuf> = match &capture_dir {
        None => Vec::new(),
        Some(root) => {
            std::fs::create_dir_all(root)
                .map_err(|e| format!("cannot create {}: {e}", root.display()))?;
            let runs: Vec<&xencode_agents_rs::RunCapture> = if report.all_runs.is_empty() {
                report.captures.iter().collect()
            } else {
                report.all_runs.iter().collect()
            };
            let mut dirs = Vec::new();
            for (index, run) in runs.iter().enumerate() {
                // Repeats are compared against each other, so a second run must
                // not overwrite the first one's raw stream.
                let base = if repeat > 1 {
                    root.join(format!("run-{}", index + 1))
                } else {
                    root.clone()
                };
                dirs.push(xencode_agents_rs::capture::write_capture(&base, run)?);
            }
            dirs
        }
    };

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
                println!("\n  no binary on PATH for these names:");
                for absent in &report.absent {
                    println!("    {} — {}", absent.name, absent.why);
                }
            }
            if !report.unknown.is_empty() {
                println!("\n  installed here but this probe has no adapter for them:");
                for unknown in &report.unknown {
                    println!("    {} — {}", unknown.name, unknown.why);
                }
            }
            if let Some(fan) = &report.fan_out {
                println!("\n  fan-out: {}", fan.summary());
            }
            let with_usage: Vec<_> = report
                .captures
                .iter()
                .filter(|c| c.provenance == xencode_agents_rs::Provenance::Observed)
                .collect();
            if !with_usage.is_empty() {
                println!("\n  what the working agents said they cost:");
                for capture in &with_usage {
                    let cost = match &capture.usage {
                        Some(usage) => usage.summary(),
                        None => "reported no usage at all".to_string(),
                    };
                    let model = capture.model.as_deref().unwrap_or("did not name a model");
                    println!("    {}: {cost} [model: {model}]", capture.agent);
                }
            }
            if !report.parked.is_empty() {
                println!("\n  stood down by the operator, so not run (name one with --agent to probe it anyway):");
                for parked in &report.parked {
                    println!("    {} — {}", parked.name, parked.why);
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
    if !kept.is_empty() {
        let note = format!(
            "\n  {} capture(s) kept; read one back with `xencode interop --trace <dir>`",
            kept.len()
        );
        match format {
            // stdout stays a parseable report.
            OutputFormat::Json => eprintln!("{note}"),
            _ => println!("{note}"),
        }
        for dir in &kept {
            match format {
                OutputFormat::Json => eprintln!("    {}", dir.display()),
                _ => println!("    {}", dir.display()),
            }
        }
    }
    Ok(())
}

pub(crate) fn run_review(
    explicit_base: Option<String>,
    session: Option<String>,
    format: OutputFormat,
) -> Result<(), String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let root = resolve_review_root(&cwd);
    let (base, source) = resolve_review_base(&root, explicit_base.as_deref());
    let diffs = xencode_context_rs::git_diff_numstat(&root, &base)?;
    let files: Vec<ReviewedFile> = diffs.iter().map(|d| review_file(&root, d)).collect();
    // AE-1: the evidence half, from the ledger rows this session wrote.
    let envelope = session.as_deref().map(|session| {
        xencode_context_rs::envelope_for_session(
            &root.join(xencode_context_rs::XENCODE_DIR),
            session,
            &format!("changes against {base}"),
            files.iter().map(|f| f.path.clone()).collect(),
        )
    });
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
            let mut output = serde_json::json!({
                "base": base,
                "files_changed": files.len(),
                "files": arr,
            });
            if let Some(obj) = output.as_object_mut() {
                obj.insert(
                    "base_source".to_string(),
                    serde_json::Value::String(source.label().to_string()),
                );
                if let Some(envelope) = &envelope {
                    obj.insert(
                        "envelope".to_string(),
                        serde_json::to_value(envelope).map_err(|e| e.to_string())?,
                    );
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?
            );
        }
        OutputFormat::Text => {
            println!(
                "{}",
                format_review_text(&base, Some(&source.header_suffix()), &files)
            );
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
    if let (OutputFormat::Text, Some(envelope)) = (&format, &envelope) {
        println!(
            "
Result envelope, from .xencode/ledger.jsonl:"
        );
        print!("{}", envelope.for_reviewer().render());
    }
    Ok(())
}
