//! the hotspots, impact, removal, generate, mutants, coverage and perf subcommands
//! `xencode hotspots`, `impact`, `removal`, `generate`, `mutants`, `cov` and `perf`.

use super::*;

pub(crate) fn run_hotspots(limit: usize, format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let rows = xencode_context_rs::hotspots(&root, limit);
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "hotspots": rows.iter().map(|r| serde_json::json!({
                    "file": r.file,
                    "kind": format!("{:?}", r.kind),
                    "message": r.message,
                })).collect::<Vec<_>>(),
            })
        );
    } else if rows.is_empty() {
        println!("\n  no history to rank — untracked files or outside a repository");
    } else {
        for row in &rows {
            println!("\n  {}", row.message);
        }
    }
    Ok(())
}

/// `QD-1`: what a change to one file affects, in three layers that carry three
/// different strengths of evidence and are never blended. The crate list is exact
/// (it comes from `cargo metadata`); the file list is a hop-capped prediction over
/// the source graph; the coupling list is history. The command runs with no
/// `.xencode` index, building the graph from the files on disk.
/// `xencode impact <file> --semantic`: the files that refer to symbols this file
/// defines, from rust-analyzer's SCIP index (`LSP-2`). A missing or stale index
/// is rebuilt first, and the reason is printed, because the rebuild takes
/// minutes and a person waiting on it should know why.
pub(crate) fn run_impact_semantic(
    file: &str,
    symbol: Option<&str>,
    limit: usize,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_context_rs::scip_index::{self, ScipError};
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let workspace = xencode_context_rs::verify::manifest_dir(&root)?;
    let mut rebuilt = None;
    if let Err(why) = scip_index::freshness(&workspace) {
        if !matches!(why, ScipError::Missing(_) | ScipError::Stale(_)) {
            return Err(why.to_string());
        }
        eprintln!(
            "  {why}\n  building it with rust-analyzer scip in {} — this runs the project's build scripts and takes minutes on a large workspace…",
            workspace.display()
        );
        let meta = scip_index::generate(&workspace, scip_index::SCIP_TIMEOUT)
            .map_err(|e| e.to_string())?;
        eprintln!(
            "  indexed {} files in {:.0} s with {}",
            meta.files.len(),
            meta.seconds,
            meta.rust_analyzer
        );
        rebuilt = Some(meta);
    }
    let report = scip_index::semantic_impact_for(
        &workspace,
        file,
        symbol,
        xencode_context_rs::IMPACT_MAX_HOPS,
    )
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    let meta = match rebuilt {
        Some(meta) => meta,
        None => scip_index::read_meta(&workspace).ok_or("the semantic index record is missing")?,
    };

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "target": report.target,
                "symbol": report.symbol,
                "tier": "semantic",
                "index": {
                    "rust_analyzer": meta.rust_analyzer,
                    "head": meta.head,
                    "seconds": meta.seconds,
                    "files": meta.files.len(),
                },
                "declared": report.declared,
                "basis": report.basis(),
                "affected": report.files.iter().map(|f| serde_json::json!({
                    "file": f.file, "hops": f.hops, "via": f.via,
                })).collect::<Vec<_>>(),
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    match &report.symbol {
        Some(symbol) => println!("\n  who uses `{symbol}` from {}", report.target),
        None => println!("\n  who uses {}", report.target),
    }
    println!(
        "  semantic index: {} files, built by {} in {:.0} s",
        meta.files.len(),
        meta.rust_analyzer,
        meta.seconds
    );
    if report.files.is_empty() {
        println!("\n    nothing outside this file refers to it");
    } else {
        for f in report.files.iter().take(limit) {
            let names = if f.via.len() > 6 {
                format!("{} and {} more", f.via[..6].join(", "), f.via.len() - 6)
            } else {
                f.via.join(", ")
            };
            println!("    {} — {} hop(s), refers to {}", f.file, f.hops, names);
        }
        if report.files.len() > limit {
            println!("    … {} more", report.files.len() - limit);
        }
    }
    println!("\n  {}", report.basis());
    Ok(())
}

pub(crate) fn run_impact(file: &str, limit: usize, format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let workspace = xencode_context_rs::verify::manifest_dir(&root)?;
    let impact = xencode_context_rs::change_impact(&workspace, file).map_err(|e| e.to_string())?;

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "target": impact.target,
                "crate": impact.crate_name,
                "crate_basis": "exact, from `cargo metadata --no-deps`",
                "reverse_crates": impact.reverse_crates.iter().map(|(n, hops)| serde_json::json!({
                    "crate": n, "hops": hops,
                })).collect::<Vec<_>>(),
                "direct_crates": impact.direct_crates.iter().map(|(d, kind)| serde_json::json!({
                    "crate": d, "kind": kind,
                })).collect::<Vec<_>>(),
                "files": {
                    "basis": impact.files.basis(),
                    "affected": impact.files.files.iter().map(|f| serde_json::json!({
                        "file": f.file, "hops": f.hops, "via": f.via,
                    })).collect::<Vec<_>>(),
                },
                "history": {
                    "known": impact.history_known,
                    "own_commits": impact.own_commits,
                    "cochange": impact.cochange.iter().map(|(f, n)| serde_json::json!({
                        "file": f, "commits_together": n,
                    })).collect::<Vec<_>>(),
                },
                "cli_commands": impact.cli_impact.as_ref().map(|cli| serde_json::json!({
                    "source_file": cli.source_file,
                    "variants_count": cli.variants_count(),
                    "commands": cli.commands.iter().map(|c| serde_json::json!({
                        "variant": c.variant,
                        "subcommand": c.subcommand,
                        "line": c.line,
                        "doc_refs": c.doc_refs.iter().map(|d| serde_json::json!({
                            "file": d.file,
                            "line": d.line,
                            "text": d.text,
                        })).collect::<Vec<_>>(),
                    })).collect::<Vec<_>>(),
                    "stale_docs": cli.stale_docs.iter().map(|s| serde_json::json!({
                        "file": s.file,
                        "line": s.line,
                        "subcommand": s.subcommand,
                        "text": s.text,
                    })).collect::<Vec<_>>(),
                })),
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("\n  impact of {}", impact.target);
    match &impact.crate_name {
        Some(name) => println!("  crate: {name}"),
        None => println!("  crate: (this file is not inside a workspace crate)"),
    }

    if let Some(cli) = &impact.cli_impact {
        println!(
            "\n  CLI commands defined ({} clap variant(s) in {}):",
            cli.commands.len(),
            cli.source_file
        );
        for cmd in cli.commands.iter().take(limit) {
            println!(
                "    {} — Commands::{} (line {})",
                cmd.subcommand, cmd.variant, cmd.line
            );
            if cmd.doc_refs.is_empty() {
                println!("      manuals: (none)");
            } else {
                println!("      manuals:");
                for doc in &cmd.doc_refs {
                    println!("        {}:{} — {}", doc.file, doc.line, doc.text);
                }
            }
        }
        if cli.commands.len() > limit {
            println!("    … {} more", cli.commands.len() - limit);
        }

        if !cli.stale_docs.is_empty() {
            println!("\n  stale manual references (documented subcommands not in Commands enum):");
            for stale in &cli.stale_docs {
                println!(
                    "    {}:{} — `{}` not in Commands ({})",
                    stale.file, stale.line, stale.subcommand, stale.text
                );
            }
        } else {
            println!("\n  stale manual references: none — all documented subcommands match active variants");
        }
    }

    println!("\n  crates that depend on it — exact, from cargo metadata:");
    if impact.reverse_crates.is_empty() {
        println!("    none — nothing first-party depends on this crate");
    } else {
        for (name, hops) in impact.reverse_crates.iter().take(limit) {
            let direct = impact
                .direct_crates
                .iter()
                .find(|(d, _)| d == name)
                .map(|(_, kind)| format!(" ({kind} edge)"))
                .unwrap_or_default();
            println!("    {name} — {hops} hop(s){direct}");
        }
    }

    println!("\n  files that link it — predicted, hop-capped:");
    if impact.files.files.is_empty() {
        println!("    none in the graph — this file's own edges point outward, not in");
    } else {
        for f in impact.files.files.iter().take(limit) {
            println!(
                "    {} — {} hop(s) via {}",
                f.file,
                f.hops,
                f.via.join(", ")
            );
        }
        if impact.files.files.len() > limit {
            println!("    … {} more", impact.files.files.len() - limit);
        }
    }
    println!("\n  {}", impact.files.basis());

    println!("\n  coupled by history — most co-changed with:");
    if !impact.history_known {
        println!("    unknown — no git history here to read");
    } else if impact.cochange.is_empty() {
        println!("    none — this file is not committed alongside others");
    } else {
        for (f, n) in impact.cochange.iter().take(limit) {
            println!("    {f} — {n} commit(s) together");
        }
    }
    if impact.history_known {
        println!(
            "\n  own churn: {} commit(s) touching {}",
            impact.own_commits, impact.target
        );
    }
    Ok(())
}

pub(crate) fn run_removal(file: &str, limit: usize, format: OutputFormat) -> Result<(), String> {
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let workspace = xencode_context_rs::verify::manifest_dir(&root)?;
    let removal =
        xencode_context_rs::removal_impact(&workspace, file).map_err(|e| e.to_string())?;

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "target": removal.target,
                "basis": "the dependency graph with one file removed, off resolved `use`/`mod`/`impl` names — not a compile",
                "breaks": removal.breaks,
                "orphans": removal.orphans,
                "indexed_files": removal.indexed_files,
                "edges": removal.edges,
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("\n  if you delete {}", removal.target);

    println!("\n  files that lose a link to it:");
    if removal.breaks.is_empty() {
        println!("    none — no other file resolves a name into this one");
    } else {
        for f in removal.breaks.iter().take(limit) {
            println!("    {f}");
        }
        if removal.breaks.len() > limit {
            println!("    … {} more", removal.breaks.len() - limit);
        }
    }

    println!("\n  files that become dead code (nothing else reaches them):");
    if removal.orphans.is_empty() {
        println!("    none — every module this one pulled in is pulled in elsewhere");
    } else {
        for f in removal.orphans.iter().take(limit) {
            println!("    {f}");
        }
        if removal.orphans.len() > limit {
            println!("    … {} more", removal.orphans.len() - limit);
        }
    }

    println!(
        "\n  From the graph of {} Rust files and {} resolved edges. A link means a \
         file wrote a `use` path, a `mod` declaration or an `impl Trait for Type` that \
         resolves to another — name resolution through module paths, not a type-checked \
         compile. Deletion is stronger evidence than an edit here: a change a consumer \
         might absorb, a missing file it cannot.",
        removal.indexed_files, removal.edges
    );
    Ok(())
}

pub(crate) fn run_generate(artifact: GenerateArtifact, shell: GenerateShell) -> Result<(), String> {
    use clap::CommandFactory;
    let mut cmd = Cli::command();
    match artifact {
        GenerateArtifact::Completions => {
            let shell = match shell {
                GenerateShell::Bash => clap_complete::Shell::Bash,
                GenerateShell::Fish => clap_complete::Shell::Fish,
                GenerateShell::Zsh => clap_complete::Shell::Zsh,
                GenerateShell::Powershell => clap_complete::Shell::PowerShell,
                GenerateShell::Elvish => clap_complete::Shell::Elvish,
            };
            clap_complete::generate(shell, &mut cmd, "xencode", &mut std::io::stdout());
            Ok(())
        }
        GenerateArtifact::Man => {
            let man = clap_mangen::Man::new(cmd);
            man.render(&mut std::io::stdout())
                .map_err(|e| format!("could not render the man page: {e}"))?;
            Ok(())
        }
    }
}

pub(crate) fn run_mutants(
    diff: Option<String>,
    timeout: u64,
    check_repair: Option<std::path::PathBuf>,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_analysis_rs::mutation as mtest;

    if let Some(patch) = check_repair {
        return judge_repair(patch, format);
    }
    if !mtest::available() {
        return Err(
            "cargo mutants is not installed. Install it with `cargo install cargo-mutants`"
                .to_string(),
        );
    }
    let root = std::env::current_dir().map_err(|e| e.to_string())?;
    let manifest = xencode_context_rs::verify::manifest_dir(&root)?;

    // The diff is written first, with pinned prefixes, because cargo-mutants
    // takes a file rather than a ref and silently filters nothing when the paths
    // carry git's mnemonic `i/`/`w/` prefixes.
    let patch = mtest::write_diff(&manifest, diff.as_deref())?;
    let argv = mtest::mutants_argv(Some(&patch), Some(timeout));
    let rendered = format!("cargo {}", argv.join(" "));
    let mut command = std::process::Command::new("cargo");
    command.current_dir(&manifest).args(&argv);
    eprintln!("  running: {rendered}");
    eprintln!("  this runs the whole suite once per surviving mutant, so it is not quick");
    let status = command
        .stdout(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("could not start cargo mutants: {e}"))?;
    // A non-zero exit means mutants were missed, which is a result rather than
    // a failure to run, so the report is read either way.
    let _ = status;

    let run = mtest::parse_report(&manifest.join(mtest::REPORT_DIR))?;
    let missed = run.missed();

    if matches!(format, OutputFormat::Json) {
        let report = serde_json::json!({
            "command": rendered,
            "total": run.mutants.len(),
            "counts": run.counts(),
            "usable": run.is_usable(),
            "symbols": run.per_symbol().iter().map(|s| serde_json::json!({
                "file": s.file,
                "symbol": s.symbol,
                "caught": s.caught,
                "missed": s.missed,
                "unviable": s.unviable,
                "timeout": s.timeout,
                "viable": s.score().map(|(_, v)| v),
            })).collect::<Vec<_>>(),
            "missed": missed.iter().map(|m| serde_json::json!({
                "file": m.file, "name": m.name, "key": m.key(),
            })).collect::<Vec<_>>(),
            "notes": run.notes,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else {
        println!("\n  {}\n", run.counts());
        for note in &run.notes {
            println!("  note:  {note}");
        }
        // QD-3: the rollup names the function the tests are weakest on, so a
        // score points at code to read rather than a vague "feature is untested".
        println!("  per function (worst first):");
        for line in run.symbol_report() {
            println!("    {line}");
        }
        if missed.is_empty() {
            println!("\n  no missed mutants — every mutation was caught by a test");
        } else {
            println!("\n  MISSED — code no test can tell from correct:");
            for m in &missed {
                println!("\n    {}", m.file);
                println!("      {}", m.name);
            }
            println!(
                "\n  Repair these by making the tests stronger. A repair is only real if \
                 the same mutants are re-run and come back caught."
            );
        }
    }
    if run.is_usable() && missed.is_empty() {
        Ok(())
    } else {
        Err("mutants survived".to_string())
    }
}

/// Judge a proposed repair against the four gate conditions.
///
/// Reads the patch, the missed-mutant list and both assertion counts from a JSON
/// file, so the judgement itself is a pure function that can be tested without a
/// mutation run. Re-running the same mutant set is the caller's job and is
/// reported as missing when its result is absent — never assumed.
pub(crate) fn judge_repair(path: std::path::PathBuf, format: OutputFormat) -> Result<(), String> {
    use xencode_analysis_rs::mutation as mtest;

    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;

    let counts_of = |key: &str| -> std::collections::BTreeMap<String, usize> {
        value
            .get(key)
            .and_then(|v| v.as_object())
            .map(|o| {
                o.iter()
                    .filter_map(|(f, n)| n.as_u64().map(|n| (f.clone(), n as usize)))
                    .collect()
            })
            .unwrap_or_default()
    };

    let targeted: Vec<mtest::Mutant> = value
        .get("targeted")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    Some(mtest::Mutant {
                        file: m.get("file")?.as_str()?.to_string(),
                        name: m.get("name")?.as_str()?.to_string(),
                        // A repair patch names mutants by file and name, not by
                        // the enclosing function, so the rollup bucket is unknown.
                        symbol: mtest::OUTSIDE_FUNCTION.to_string(),
                        verdict: mtest::Verdict::Missed,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // The re-run, when one is recorded. Its absence is the rejection.
    let rerun = value.get("rerun").and_then(|r| {
        let mutants: Vec<mtest::Mutant> = r
            .get("mutants")?
            .as_array()?
            .iter()
            .filter_map(|m| {
                Some(mtest::Mutant {
                    file: m.get("file")?.as_str()?.to_string(),
                    name: m.get("name")?.as_str()?.to_string(),
                    symbol: mtest::OUTSIDE_FUNCTION.to_string(),
                    verdict: match m.get("verdict").and_then(|v| v.as_str()).unwrap_or("") {
                        "caught" => mtest::Verdict::Caught,
                        "unviable" => mtest::Verdict::Unviable,
                        "timeout" => mtest::Verdict::Timeout,
                        _ => mtest::Verdict::Missed,
                    },
                })
            })
            .collect();
        Some(mtest::Run {
            mutants,
            ..mtest::Run::default()
        })
    });

    let patch = value
        .get("patch")
        .and_then(|p| p.as_str())
        .unwrap_or_default();
    let before = counts_of("assertions_before");
    let after = counts_of("assertions_after");
    let verdict = mtest::check_repair(&mtest::Repair {
        patch,
        targeted: &targeted,
        assertions_before: &before,
        assertions_after: &after,
        same_set_rerun: rerun.as_ref(),
    });

    if matches!(format, OutputFormat::Json) {
        let report = serde_json::json!({
            "accepted": verdict.accepted,
            "targeted": verdict.targeted,
            "edited": verdict.edited,
            "violations": verdict.violations.iter().map(|v| serde_json::json!({
                "rule": v.rule, "detail": v.detail,
            })).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else if verdict.accepted {
        println!("\n  repair accepted — it only strengthened tests, and the same mutants now fail");
    } else {
        println!("\n  repair REJECTED:");
        for v in &verdict.violations {
            println!("\n  [{}]", v.rule);
            for line in v.detail.split(". ") {
                println!("    {line}");
            }
        }
    }
    if verdict.accepted {
        Ok(())
    } else {
        Err("repair rejected".to_string())
    }
}

pub(crate) fn run_cov(
    base: Option<String>,
    test: Option<String>,
    show_missing_lines: bool,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_analysis_rs::covdiff;

    if !covdiff::available() {
        return Err(
            "cargo llvm-cov is not installed. Install it with `cargo install cargo-llvm-cov` \
             and rustup component add llvm-tools-preview"
                .to_string(),
        );
    }
    let root = std::env::current_dir().map_err(|e| e.to_string())?;

    // The repository's own verified test command, so the numbers describe the
    // suite that exists rather than one invented here.
    let test_command = test.or_else(|| {
        let discovery = xencode_context_rs::anchor::discover(&root);
        discovery
            .verified(xencode_context_rs::anchor::Kind::Test)
            .map(|r| r.command.clone())
    });

    let run = covdiff::run(&root, base.as_deref(), test_command.as_deref())?;
    let c = &run.coverage;

    if matches!(format, OutputFormat::Json) {
        let report = serde_json::json!({
            "command": run.command,
            "cold_build": run.build == covdiff::ColdReport::Cold,
            "seconds": run.took.as_secs_f64(),
            "covered": c.total_covered(),
            "uncovered": c.total_uncovered(),
            "unknown": c.total_unknown(),
            "ratio": c.ratio(),
            "complete": c.is_complete(),
            "unmeasured_files": c.unmeasured,
            "summary": c.summary(),
            "notes": run.notes,
            "files": c.files.iter().map(|f| serde_json::json!({
                "file": f.file,
                "covered": f.covered,
                "uncovered": f.uncovered,
                "unknown": f.unknown,
            })).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
        );
    } else if show_missing_lines {
        // The read-only shape `VF-2` asks for: file to line numbers, nothing else.
        for file in &c.files {
            if file.uncovered.is_empty() && file.unknown.is_empty() {
                continue;
            }
            println!("{}", file.file);
            if !file.uncovered.is_empty() {
                println!("  never run:  {}", join(&file.uncovered));
            }
            if !file.unknown.is_empty() {
                println!("  no data:    {}", join(&file.unknown));
            }
        }
    } else {
        println!("\n  {}", c.summary());
        println!(
            "  build: {} ({}s)",
            match run.build {
                covdiff::ColdReport::Cold => "cold — every crate was rebuilt instrumented",
                covdiff::ColdReport::Warm => "warm — reused the instrumented target directory",
            },
            run.took.as_secs()
        );
        for note in &run.notes {
            println!("  note:  {note}");
        }
        for file in &c.files {
            if file.uncovered.is_empty() && file.unknown.is_empty() {
                continue;
            }
            println!("\n  {}", file.file);
            if let Some(ratio) = file.ratio() {
                println!("    {:.0}% of measurable added lines ran", ratio * 100.0);
            } else {
                println!("    no measurable line in this file");
            }
            if !file.uncovered.is_empty() {
                println!("    never run: {}", join(&file.uncovered));
            }
            if !file.unknown.is_empty() {
                println!("    no data:   {}", join(&file.unknown));
            }
        }
    }
    if c.is_complete() {
        Ok(())
    } else {
        Err("some added lines were not exercised".to_string())
    }
}

/// QO-4 — measure the hot paths, and decide whether they moved.
///
/// The table is the product here: a number without a verdict is what every
/// benchmark already prints, and a verdict the run cannot support is worse than
/// none. So a path whose samples are more than 5% spread reports `NO VERDICT`
/// with the reason, and a baseline measured over a different tree refuses every
/// comparison in it rather than quietly reporting the difference as a change.
pub(crate) fn run_perf(action: PerfAction) -> Result<(), String> {
    use xencode_context_rs::perf;

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let repo_root = perf::state_root(&cwd);
    let manifest_root = xencode_context_rs::verify::manifest_dir(&cwd)?;

    match action {
        PerfAction::Show { format } => show_baseline(&repo_root, format),
        PerfAction::Record { force, format } => {
            let measured = perf::measure(&manifest_root, None)?;
            let baseline = perf::record_baseline(&repo_root, &measured, force)?;
            let path = perf::Baseline::path_for(&repo_root);
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "command": measured.command,
                        "seconds": measured.took.as_secs_f64(),
                        "corpus_files": baseline.corpus_files,
                        "baseline": path.display().to_string(),
                        "forced": force,
                        "paths": baseline.benches.values().map(perf_samples_json).collect::<Vec<_>>(),
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                println!("  {}", measured.command);
                println!(
                    "  measured {} path(s) over {} file(s) in {:.1} s",
                    baseline.benches.len(),
                    baseline.corpus_files,
                    measured.took.as_secs_f64()
                );
                for samples in baseline.benches.values() {
                    println!(
                        "  {:<34}  {:>12}  ({} samples, {:.1}% spread)",
                        samples.id,
                        human_duration(samples.median()),
                        samples.n(),
                        samples.cv().unwrap_or_default() * 100.0
                    );
                }
                println!("\n  baseline written to {}", path.display());
                if force {
                    let noisy = perf::noisy_paths(&measured);
                    if !noisy.is_empty() {
                        println!(
                            "  note: recorded despite being forced; {} measured wider than \
                             the {:.0}% a verdict rests on, and a comparison against it will \
                             refuse",
                            noisy.join(", "),
                            perf::MAX_CV * 100.0
                        );
                    }
                }
            }
            Ok(())
        }
        PerfAction::Check {
            filter,
            alert_pct,
            alpha,
            format,
        } => {
            let report = perf::check(
                &repo_root,
                &manifest_root,
                filter.as_deref(),
                alert_pct,
                alpha,
            )?;
            let regressions = report.regressions();
            let refusals = report.refusals();

            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "command": report.command,
                        "seconds": report.took.as_secs_f64(),
                        "corpus_files": report.corpus_files,
                        "baseline_corpus_files": report.baseline_corpus_files,
                        "corpus_mismatch": report.corpus_mismatch,
                        "alert_pct": alert_pct,
                        "alpha": alpha,
                        "max_cv": perf::MAX_CV,
                        "regressions": regressions.len(),
                        "refusals": refusals.len(),
                        "notes": report.notes,
                        "paths": report.comparisons.iter().map(perf_comparison_json).collect::<Vec<_>>(),
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                println!("  {}", report.command);
                println!(
                    "  {} file(s) measured, baseline recorded over {}, in {:.1} s",
                    report.corpus_files,
                    report
                        .baseline_corpus_files
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "nothing".to_string()),
                    report.took.as_secs_f64()
                );
                println!(
                    "  alert at {alert_pct}% of the baseline, judged at α = {alpha}, verdicts \
                     refused past a {:.0}% spread",
                    perf::MAX_CV * 100.0
                );
                for comparison in &report.comparisons {
                    let delta = comparison
                        .delta_pct
                        .map(|d| format!("{d:+.2}%"))
                        .unwrap_or_else(|| "     —".to_string());
                    let p = comparison
                        .p_value
                        .map(|p| format!("{p:.4}"))
                        .unwrap_or_else(|| "  —  ".to_string());
                    let spread = comparison
                        .cv
                        .map(|c| format!("{:.1}%", c * 100.0))
                        .unwrap_or_else(|| "  —  ".to_string());
                    println!(
                        "\n  {}\n    {:<11}  delta {delta}   p = {p} ({})   spread {spread}",
                        comparison.id,
                        comparison.outcome.label(),
                        comparison.p_value_method.unwrap_or("not tested"),
                    );
                    if let Some(reason) = &comparison.reason {
                        println!("    {reason}");
                    }
                }
                for note in &report.notes {
                    println!("\n  note: {note}");
                }
                println!(
                    "\n  {} path(s) compared: {} regression(s), {} refusal(s)",
                    report.comparisons.len(),
                    regressions.len(),
                    refusals.len()
                );
            }

            if regressions.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "{} hot path(s) measurably slower than the baseline",
                    regressions.len()
                ))
            }
        }
    }
}

pub(crate) fn perf_samples_json(samples: &xencode_context_rs::perf::Samples) -> serde_json::Value {
    serde_json::json!({
        "id": samples.id,
        "samples": samples.n(),
        "median_ns": samples.median(),
        "mean_ns": samples.mean(),
        "cv": samples.cv(),
    })
}

pub(crate) fn perf_comparison_json(
    comparison: &xencode_context_rs::perf::Comparison,
) -> serde_json::Value {
    serde_json::json!({
        "id": comparison.id,
        "outcome": comparison.outcome.label(),
        "baseline_median_ns": comparison.baseline_median_ns,
        "current_median_ns": comparison.current_median_ns,
        "delta_pct": comparison.delta_pct,
        "p_value": comparison.p_value,
        "p_value_method": comparison.p_value_method,
        "cv": comparison.cv,
        "reason": comparison.reason,
    })
}

/// The recorded baseline and nothing else — no measurement, no verdict.
pub(crate) fn show_baseline(
    repo_root: &std::path::Path,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_context_rs::perf;

    let path = perf::Baseline::path_for(repo_root);
    let Some(baseline) = perf::Baseline::load(repo_root)? else {
        return Err(format!(
            "no baseline recorded yet — run `xencode perf record` (it would write {})",
            path.display()
        ));
    };
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&baseline).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!("  baseline at {}", path.display());
    println!(
        "  recorded over {} file(s) in {}",
        baseline.corpus_files, baseline.recorded_in
    );
    let age_days = {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(baseline.recorded_unix) / 86_400
    };
    println!(
        "  {} day(s) old, {} path(s)",
        age_days,
        baseline.benches.len()
    );
    for samples in baseline.benches.values() {
        println!(
            "  {:<34}  {:>12}  ({} samples, {:.1}% spread)",
            samples.id,
            human_duration(samples.median()),
            samples.n(),
            samples.cv().unwrap_or_default() * 100.0
        );
    }
    Ok(())
}

/// A duration in nanoseconds, in the unit the number actually reads in.
///
/// The hot paths span five orders of magnitude — the token trimmer runs in
/// microseconds, a whole-tree symbol pass in seconds — so a fixed unit gives a
/// column of either `0.015` or `801449.095`.
pub(crate) fn human_duration(ns: f64) -> String {
    const UNITS: [(f64, &str); 4] = [
        (1.0, "ns"),
        (1_000.0, "µs"),
        (1_000_000.0, "ms"),
        (1_000_000_000.0, "s"),
    ];
    let mut chosen = UNITS[0];
    for unit in UNITS {
        if ns >= unit.0 {
            chosen = unit;
        }
    }
    format!("{:.3} {}", ns / chosen.0, chosen.1)
}

/// `1 entry`, `3 commits` — the coverage counts are read by a person, so they
/// are not printed as `1 entr(y|ies)`.
pub(crate) fn plural_count(count: usize, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}
