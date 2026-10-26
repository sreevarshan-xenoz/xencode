//! the compete and merge subcommands
//! `xencode compete` and `xencode merge`, with lease and veto checks.

use super::*;

pub(crate) fn run_compete(action: CompeteAction) -> Result<(), String> {
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
pub(crate) fn branch_author(repo: &std::path::Path, branch: &str) -> Option<String> {
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
pub(crate) fn lease_breaches(
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
pub(crate) fn veto_audit_sink() -> Result<xencode_server_rs::audit::AuditSink, String> {
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
pub(crate) fn veto_audit_event(
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

pub(crate) fn run_merge(action: MergeAction) -> Result<(), String> {
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
