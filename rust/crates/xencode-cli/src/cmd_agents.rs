//! the agents and contract subcommands
//! `xencode agents` and `xencode contract`.

use super::*;

pub(crate) struct AgentsArgs<'a> {
    pub(crate) contract: bool,
    pub(crate) health: bool,
    pub(crate) target_agent: Option<&'a str>,
    pub(crate) build_package: Option<&'a str>,
    pub(crate) package_path: Option<&'a std::path::Path>,
    pub(crate) resume: bool,
    pub(crate) test_cmds: &'a [String],
    pub(crate) route_task: Option<&'a str>,
    pub(crate) require_caps: &'a [String],
    pub(crate) max_cost: Option<f64>,
    pub(crate) redispatch_task: Option<&'a str>,
    pub(crate) replacement_agent: Option<&'a str>,
    pub(crate) stop_reason: Option<&'a str>,
    pub(crate) format: OutputFormat,
}

pub(crate) fn run_agents(args: AgentsArgs<'_>) -> Result<(), String> {
    if let Some(task_id) = args.redispatch_task {
        let repo_dir = std::env::current_dir().map_err(|e| e.to_string())?;
        let initial_worker = args.target_agent.unwrap_or("failed-worker");
        let replacement_worker = args.replacement_agent.unwrap_or("replacement-worker");

        // Parse process stop reason
        let stop_reason = if let Some(reason_str) = args.stop_reason {
            if let Some(sig) = reason_str.strip_prefix("signal:") {
                let code = sig
                    .parse::<i32>()
                    .map_err(|_| format!("invalid signal in '{reason_str}'"))?;
                xencode_agents_rs::WorkerStopReason::Signal { signal: code }
            } else if let Some(exit) = reason_str.strip_prefix("exit:") {
                let code = exit
                    .parse::<i32>()
                    .map_err(|_| format!("invalid exit code in '{reason_str}'"))?;
                xencode_agents_rs::WorkerStopReason::ExitCode { code }
            } else if let Some(secs) = reason_str.strip_prefix("timeout:") {
                let dur = secs
                    .parse::<u64>()
                    .map_err(|_| format!("invalid duration in '{reason_str}'"))?;
                xencode_agents_rs::WorkerStopReason::TimedOut { duration_secs: dur }
            } else {
                xencode_agents_rs::WorkerStopReason::Signal { signal: 9 }
            }
        } else {
            xencode_agents_rs::WorkerStopReason::Signal { signal: 9 }
        };

        let default_cmd = "cargo test".to_string();
        let test_cmd_refs: Vec<&str> = if args.test_cmds.is_empty() {
            vec![default_cmd.as_str()]
        } else {
            args.test_cmds.iter().map(String::as_str).collect()
        };

        let ledger_path = repo_dir.join(".xencode").join("task_ledger.jsonl");

        let outcome = xencode_agents_rs::redispatch_failed_worker(
            &repo_dir,
            task_id,
            initial_worker,
            replacement_worker,
            stop_reason,
            &test_cmd_refs,
            &ledger_path,
            |context, _dir| {
                if matches!(args.format, OutputFormat::Text) {
                    println!(
                        "Re-dispatching task '{task_id}' onto worker '{replacement_worker}'..."
                    );
                    println!("{context}");
                }
                Ok(0)
            },
        )?;

        if matches!(args.format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&outcome).map_err(|e| e.to_string())?
            );
        } else {
            println!("Task Re-dispatch Completed for '{}':", outcome.task_id);
            println!(
                "  Initial worker: {} ({})",
                outcome.initial_worker,
                outcome.initial_stop_reason.summary()
            );
            println!("  Replacement worker: {}", outcome.replacement_worker);
            println!("  Verification tests passed: {}", outcome.all_tests_passed);
            println!(
                "  Preserved diff files ({}):",
                outcome.preserved_diff.changed_files.len()
            );
            for f in &outcome.preserved_diff.changed_files {
                println!("    * {f}");
            }
            println!(
                "  Ledger entries logged ({}) to {}:",
                outcome.ledger_entries.len(),
                ledger_path.display()
            );
            for entry in &outcome.ledger_entries {
                println!(
                    "    - Attempt {}: worker '{}' -> passed: {}, stop reason: {}",
                    entry.attempt,
                    entry.worker,
                    entry.passed,
                    entry.stop_reason.summary()
                );
            }
        }
        return Ok(());
    }

    if let Some(task_id) = args.route_task {
        // Everything the router is allowed to use was read off this machine a few
        // seconds ago, and the line of evidence behind each capability travels with
        // it, so a choice can be checked rather than believed (`OR-11`).
        let claims = xencode_agents_rs::probe_contract();
        let confirmed_caps_by_agent = xencode_agents_rs::confirmed_capabilities(&claims);
        let evidence_by_agent = xencode_agents_rs::evidence_by_agent(&claims);
        let probed: std::collections::BTreeSet<String> =
            claims.iter().map(|claim| claim.agent.to_string()).collect();

        // The posture is read from the same config the TUI shows, and it refuses a
        // roster agent by that name before the capability, load and cost checks are
        // consulted, so the reason printed here is the reason the router actually
        // applied (`OR-13`).
        let profile = XencodeConfig::load().unwrap_or_default().profile();

        let installed = xencode_agents_rs::inventory();
        let names: Vec<String> = if installed.is_empty() {
            // Nothing on `PATH` matched the roster. Its names are still listed, so a
            // person asking who could take a task sees what was considered — each one
            // marked as never probed, which is not the same answer as probing it.
            xencode_agents_rs::ROSTER
                .iter()
                .map(|spec| spec.name.to_string())
                .collect()
        } else {
            installed
                .iter()
                .map(|agent| agent.name.to_string())
                .collect()
        };

        let candidates: Vec<xencode_core_rs::WorkerCandidate> = names
            .iter()
            .map(|name| xencode_core_rs::WorkerCandidate {
                id: name.clone(),
                probed_capabilities: confirmed_caps_by_agent
                    .get(name)
                    .cloned()
                    .unwrap_or_default(),
                probed: probed.contains(name),
                // Answered from the roster row, not from the shape of the name:
                // every candidate listed here came off that roster, so under the
                // shipped posture all of them are refused before a measurement is
                // read, and the refusal says which setting would change that.
                external: xencode_agents_rs::is_external_worker(name),
                capability_evidence: evidence_by_agent.get(name).cloned().unwrap_or_default(),
                load: xencode_core_rs::Fact::unknown(
                    "xencode cannot see what another process has given this worker to do; the \
                     only work it counts is what a team of its own has leased, and this question \
                     leased nothing",
                ),
                capacity: xencode_core_rs::Fact::unknown(
                    "nobody measured how many tasks this worker runs at once; the roster records \
                     what its help output says, and help does not say this",
                ),
                cost: xencode_core_rs::Fact::unknown(
                    "no price is known for this worker: xencode's price documents name models, \
                     and these agents bill their own accounts",
                ),
            })
            .collect();

        let mut required_caps = std::collections::BTreeSet::new();
        for cap in args.require_caps {
            required_caps.insert(cap.to_string());
        }

        let task = xencode_core_rs::TaskRequirement {
            task_id: task_id.to_string(),
            required_capabilities: required_caps,
            cost_ceiling: args.max_cost,
            profile,
        };

        let decision = xencode_core_rs::CapabilityRouter::route(&task, &candidates)?;

        // Whether anything was refused on the posture at all, so the closing note
        // only claims the rule is in force when it was the reason.
        let refused_by_profile = candidates
            .iter()
            .any(|c| profile.check_worker(&c.id, c.external).is_err());

        if matches!(args.format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&decision).map_err(|e| e.to_string())?
            );
        } else {
            println!("Routing decision for task '{}':", decision.task_id);
            println!();
            println!("  Posture: {}", profile.name());
            for rule in profile.rules() {
                println!("    {rule}");
            }
            println!();
            match &decision.selected_worker {
                Some(worker) => println!("  chosen: {worker}"),
                None => println!("  chosen: nothing — no candidate could take this task"),
            }
            println!("  why:    {}", decision.explanation);

            println!("\n  What the router asked, in the order it asked it:");
            for step in &decision.steps {
                let standing = if !step.ran {
                    "not asked"
                } else if step.decided {
                    "decided it"
                } else {
                    "asked, settled nothing"
                };
                println!("    {:14} {:20} {}", step.check, standing, step.words);
            }

            // The chosen worker gets its facts in full, because that is the list a
            // reader is being asked to check. The others are listed with what the
            // probe found on them and why any was refused. A check that did not run
            // is said once, with how many candidates it did not run for.
            let considered = decision.candidate_evaluations.len();
            let mut not_asked: std::collections::BTreeMap<&String, usize> =
                std::collections::BTreeMap::new();
            for evaluation in &decision.candidate_evaluations {
                for line in &evaluation.not_checked {
                    *not_asked.entry(line).or_insert(0) += 1;
                }
            }

            if let Some(chosen) = decision
                .candidate_evaluations
                .iter()
                .find(|e| decision.selected_worker.as_deref() == Some(e.worker_id.as_str()))
            {
                println!("\n  The facts behind the choice — {}:", chosen.worker_id);
                for fact in &chosen.facts {
                    println!("      {fact}");
                }
            }

            println!("\n  Every worker considered ({considered}):");
            for evaluation in &decision.candidate_evaluations {
                println!(
                    "    {} {:<13} probed here as: {}",
                    if evaluation.eligible { "▸" } else { "✗" },
                    evaluation.worker_id,
                    if evaluation.probed_capabilities.is_empty() {
                        "nothing this machine could see".to_string()
                    } else {
                        evaluation.probed_capabilities.join(", ")
                    }
                );
                if let Some(reason) = &evaluation.rejection {
                    println!("      why not: {}", reason.words());
                }
            }

            if !not_asked.is_empty() {
                println!(
                    "\n  Checks that did not run, and how many candidates they would have covered:"
                );
                for (line, count) in &not_asked {
                    println!("      for {count} of {considered}: {line}");
                }
            }
            println!(
                "\n  The capability lines above were not inferred from a name or a document: they \
                came from the help output of the binaries on this machine, read during this \
                command. Where a line says `not measured`, that is the whole of what xencode \
                knows, and no worker was accepted or refused because of it."
            );
            if refused_by_profile {
                println!(
                    "  The one thing decided before the capability, load and cost checks were \
                    consulted is the posture: it refuses a roster agent by its roster row, which \
                    is a policy and not a measurement. The `Posture:` block above names the \
                    setting that would open the rule (`xencode config set \
                    allow_external_workers true`)."
                );
            }
        }
        return Ok(());
    }

    if let Some(task_id) = args.build_package {
        let repo_dir = std::env::current_dir().map_err(|e| e.to_string())?;
        let default_cmd = "cargo test".to_string();
        let cmds: Vec<&str> = if args.test_cmds.is_empty() {
            vec![default_cmd.as_str()]
        } else {
            args.test_cmds.iter().map(String::as_str).collect()
        };
        let prev = args.target_agent.unwrap_or("worker");
        let pkg = xencode_agents_rs::WorkerPackage::build(
            &repo_dir,
            task_id,
            format!("Continuation package for task {task_id}"),
            prev,
            xencode_agents_rs::WorkerStopReason::ExitCode { code: 1 },
            &cmds,
            vec![],
        )?;
        let out_dir = repo_dir.join(".xencode").join("packages");
        std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
        let out_file = out_dir.join(format!("{task_id}.json"));
        pkg.save_to_file(&out_file)?;

        if matches!(args.format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&pkg).map_err(|e| e.to_string())?
            );
        } else {
            println!(
                "Worker continuation package written to {}",
                out_file.display()
            );
            println!("\n{}", pkg.resumption_context());
        }
        return Ok(());
    }

    if let Some(path) = args.package_path {
        let pkg = xencode_agents_rs::WorkerPackage::load_from_file(path)?;
        if args.resume {
            let repo_dir = std::env::current_dir().map_err(|e| e.to_string())?;
            let next_agent = args.target_agent.unwrap_or("resuming-worker");
            let default_cmd = "cargo test".to_string();
            let cmds: Vec<&str> = if args.test_cmds.is_empty() {
                vec![default_cmd.as_str()]
            } else {
                args.test_cmds.iter().map(String::as_str).collect()
            };
            let outcome = xencode_agents_rs::resume_task_from_package(
                &pkg,
                &repo_dir,
                next_agent,
                &cmds,
                |context, _dir| {
                    if matches!(args.format, OutputFormat::Text) {
                        println!(
                            "Resuming task '{}' with worker '{}'...",
                            pkg.task_id, next_agent
                        );
                        println!("{context}");
                    }
                    Ok(0)
                },
            )?;
            if matches!(args.format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&outcome).map_err(|e| e.to_string())?
                );
            } else {
                println!("\nResumption outcome for task '{}':", outcome.task_id);
                println!("- Resuming worker: {}", outcome.resuming_agent);
                println!("- Verification tests passed: {}", outcome.all_tests_passed);
                for t in &outcome.post_test_runs {
                    let status = if t.passed { "PASSED" } else { "FAILED" };
                    println!("  * `{}` -> exit {} [{status}]", t.command, t.exit_code);
                }
            }
            return Ok(());
        } else {
            if matches!(args.format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&pkg).map_err(|e| e.to_string())?
                );
            } else {
                println!("{}", pkg.resumption_context());
            }
            return Ok(());
        }
    }

    if args.contract {
        return run_contract(args.format);
    }
    if args.health {
        let timeout = std::time::Duration::from_secs(3);
        let selected = if let Some(target) = args.target_agent {
            let spec = xencode_agents_rs::ROSTER
                .iter()
                .find(|s| s.name == target)
                .ok_or_else(|| format!("no roster agent named '{target}'"))?;
            vec![xencode_agents_rs::check_worker_health(spec, timeout)]
        } else {
            xencode_agents_rs::check_all_worker_health(timeout)
        };
        if matches!(args.format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&selected).map_err(|e| e.to_string())?
            );
        } else {
            println!(
                "{}",
                xencode_agents_rs::format_worker_health_table(&selected)
            );
        }
        return Ok(());
    }
    let found = xencode_agents_rs::inventory();
    if matches!(args.format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "agents": found.iter().map(|a| serde_json::json!({
                    "name": a.name,
                    "binary": a.binary.display().to_string(),
                    "version": a.version,
                    "source": a.source.label(),
                })).collect::<Vec<_>>(),
            })
        );
    } else if found.is_empty() {
        println!("\n  no roster agents found on PATH");
    } else {
        for agent in &found {
            println!(
                "\n  {:<8} {}",
                agent.name,
                agent.version.as_deref().unwrap_or("(no --version answer)")
            );
            println!(
                "           {} [{}]",
                agent.binary.display(),
                agent.source.label()
            );
        }
        println!("\n  discovery only: nothing was installed, upgraded, or written");
    }
    Ok(())
}

pub(crate) fn run_contract(format: OutputFormat) -> Result<(), String> {
    use xencode_agents_rs::Verdict;

    let results = xencode_agents_rs::probe_contract();
    let bad = results
        .iter()
        .filter(|r| matches!(r.verdict, Verdict::Contradicted(_)))
        .count();
    let untested = results
        .iter()
        .filter(|r| matches!(r.verdict, Verdict::Untestable(_)))
        .count();
    // A confirmed claim is not one kind: "the flag is there" and "the flag is
    // not there, and here is the word we looked for" are opposite facts that
    // happen to share a verdict. Counted apart, so a reader is not told that
    // every one of these numbers is a capability.
    let absences = results
        .iter()
        .filter(|r| !r.expected && matches!(r.verdict, Verdict::Confirmed))
        .count();
    let confirmed = results.len() - bad - untested;
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::json!({
                "summary": {
                    "claims": results.len(),
                    "confirmed": confirmed,
                    "confirmed_absences": absences,
                    "contradicted": bad,
                    "untested": untested,
                },
                "claims": results.iter().map(|r| serde_json::json!({
                    "agent": r.agent,
                    "claim": r.claim,
                    "expected": r.expected,
                    "verdict": format!("{:?}", r.verdict),
                    "found": r.found,
                    "missing": r.missing,
                    "sources": r.sources,
                    "evidence": r.evidence_line(),
                })).collect::<Vec<_>>(),
            })
        );
    } else {
        for result in &results {
            match &result.verdict {
                Verdict::Confirmed => {}
                Verdict::Contradicted(why) => {
                    println!("\n  CONTRADICTED {} {}: {why}", result.agent, result.claim)
                }
                Verdict::Untestable(why) => {
                    println!("\n  untestable {} {}: {why}", result.agent, result.claim)
                }
            }
        }
        println!(
            "\n  {confirmed} claims confirmed ({} of them absences the probe searched for), \
             {bad} contradicted, {untested} untested",
            absences
        );
    }
    if bad == 0 {
        Ok(())
    } else {
        Err("roster claims contradicted by live --help".to_string())
    }
}
