//! `xencode orchestrator`: the commands that report on and act on the agents and
//! runs this machine started.

use super::*;

/// The `OR-14` control surface. Each verb reads the state above, or acts on one
/// process this machine started; none of them writes a config value, and the only
/// ones that write anything at all are `retry` — which launches a real child — and
/// `stop`, which kills one.
pub(crate) async fn run_orchestrator(action: OrchestratorAction) -> Result<(), String> {
    match action {
        OrchestratorAction::Status { format } => orchestrator_status(format),
        OrchestratorAction::Agents { format } => orchestrator_agents(format),
        OrchestratorAction::Tasks { format } => orchestrator_tasks(format),
        OrchestratorAction::Graph { recipe, format } => orchestrator_graph(recipe, format),
        OrchestratorAction::Split {
            task,
            commit,
            paths,
            answer,
            verification,
            model,
            format,
        } => orchestrator_split(task, commit, paths, answer, verification, model, format).await,
        OrchestratorAction::Logs { run, lines, format } => orchestrator_logs(run, lines, format),
        OrchestratorAction::Permissions { agent, format } => {
            orchestrator_permissions(agent, format)
        }
        OrchestratorAction::Costs { format } => orchestrator_costs(format).await,
        OrchestratorAction::Inspect { target, format } => orchestrator_inspect(&target, format),
        OrchestratorAction::Retry {
            recipe,
            role,
            approved_by,
            format,
        } => orchestrator_retry(&recipe, &role, approved_by.as_deref(), format).await,
        OrchestratorAction::Stop { target } => orchestrator_stop(&target),
        OrchestratorAction::Attach { agent, target } => {
            orchestrator_attach(&agent, target.as_deref())
        }
    }
}

/// The closing line every reading shares: what to type next, and the fact that
/// nothing here changed anything.
pub(crate) fn orchestrator_ran_nothing() {
    println!(
        "\n  Nothing was launched, stopped or changed by this reading. `xencode team run \
         <recipe> --approved-by <name>` runs a recipe; `/orchestrator on` turns the same state \
         into a mode inside xencode's own screen."
    );
}

pub(crate) fn orchestrator_status(format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    let running = match &fleet.tasks {
        Ok(entries) => Some(
            entries
                .values()
                .filter(|(_, status)| matches!(status, xencode_core_rs::TaskStatus::Running))
                .count(),
        ),
        Err(_) => None,
    };
    let refused = refused_names(&fleet.profile);
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "posture": fleet.profile.name(),
                "rules": fleet.profile.rules(),
                "refused_agents": refused,
                "teams_dir": fleet.teams_dir.display().to_string(),
                "recipes": fleet.recipes.len(),
                "recipes_unreadable": fleet.recipes.iter().filter(|f| f.recipe.is_err()).count(),
                "recipes_problem": fleet.recipes_problem,
                "runs_dir": fleet.runs_dir.display().to_string(),
                "recorded_runs": fleet.runs.len(),
                "runs_problem": fleet.runs_problem,
                "newest_run": fleet.newest_run().map(|run| serde_json::json!({
                    "run_id": run.run_id, "recipe": run.recipe, "approved_by": run.approved_by,
                    "elapsed_ms": run.elapsed_ms, "roles": run.roles.len(),
                })),
                "background_tasks": running,
                "tasks_problem": fleet.tasks.as_ref().err(),
                "detached_runs": fleet.detached.len(),
                "approval_records": fleet.ledgers.len(),
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!(
        "Orchestrator status · read from {}",
        fleet.xencode_dir.display()
    );
    // One label column for the whole screen, so the eight readings below line up
    // and a difference between two of them is a difference in the state.
    let field = |label: &str, value: String| println!("  {:<21}{}", format!("{label}:"), value);
    field(
        "work may go to",
        format!("{} — {}", fleet.profile.name(), fleet.profile.worker_rule()),
    );
    field("models may go to", fleet.profile.model_rule());
    if !refused.is_empty() {
        field(
            "roster refused",
            format!(
                "{} of the {} agents on the roster ({})",
                refused.len(),
                xencode_agents_rs::ROSTER.len(),
                refused.join(", ")
            ),
        );
    }
    match &fleet.recipes_problem {
        Some(problem) => field("recipes", format!("could not be read — {problem}")),
        None if fleet.recipes.is_empty() => field(
            "recipes",
            format!(
                "none in {} — `xencode team run` has nothing to plan until a `.toml` is put \
                 there",
                fleet.teams_dir.display()
            ),
        ),
        None => {
            let faulty = fleet.recipes.iter().filter(|f| f.recipe.is_err()).count();
            field(
                "recipes",
                format!(
                    "{} in {}{}",
                    fleet.recipes.len(),
                    fleet.teams_dir.display(),
                    if faulty == 0 {
                        String::new()
                    } else {
                        format!(", {faulty} of them unreadable as a recipe")
                    }
                ),
            );
        }
    }
    match &fleet.runs_problem {
        Some(problem) => field("recorded runs", format!("could not be read — {problem}")),
        None if fleet.runs.is_empty() => field(
            "recorded runs",
            format!(
                "none in {} — every figure below is therefore unmeasured",
                fleet.runs_dir.display()
            ),
        ),
        None => match fleet.newest_run() {
            Some(run) => field(
                "recorded runs",
                format!(
                    "{} · newest {} (recipe {}, {} roles, approved by {}, {})",
                    fleet.runs.len(),
                    run.run_id,
                    run.recipe,
                    run.roles.len(),
                    run.approved_by,
                    wall_clock_label(run.elapsed_ms),
                ),
            ),
            None => field(
                "recorded runs",
                format!(
                    "{} files in {}, none of them readable as a run",
                    fleet.runs.len(),
                    fleet.runs_dir.display()
                ),
            ),
        },
    }
    match running {
        Some(n) => field(
            "background tasks",
            format!(
                "{} running of {} in the registry",
                n,
                fleet
                    .tasks
                    .as_ref()
                    .map(|entries| entries.len())
                    .unwrap_or_default()
            ),
        ),
        None => field(
            "background tasks",
            "unknown — the registry could not be read; a task may be running that this command \
             cannot see"
                .to_string(),
        ),
    }
    field(
        "detached runs",
        format!(
            "{}{}",
            fleet.detached.len(),
            match fleet.detached.last() {
                Some(id) => format!(
                    " — newest is {}, `xencode orchestrator logs {id}` reads its log",
                    truncate_id(id)
                ),
                None => " — none in `cache/detached`".to_string(),
            }
        ),
    );
    field(
        "approvals on record",
        format!(
            "{} runs in {}, each carrying the tool calls a person answered",
            fleet.ledgers.len(),
            xencode_context_rs::runs_path(&fleet.xencode_dir).display()
        ),
    );
    println!(
        "\n  This is the same state xencode's `/orchestrator` mode shows; the mode lives in a \
         running session and is not a setting, so nothing here turns it on for the next one."
    );
    Ok(())
}

/// The roster names the posture would refuse work handed to.
pub(crate) fn refused_names(profile: &xencode_core_rs::Profile) -> Vec<String> {
    let considered: Vec<(String, bool)> = xencode_agents_rs::ROSTER
        .iter()
        .map(|spec| (spec.name.to_string(), true))
        .collect();
    profile
        .refused_workers(&considered)
        .into_iter()
        .map(|refusal| refusal.worker)
        .collect()
}

/// A run id is long and machine-made; a screen quoting it can show less of it.
pub(crate) fn truncate_id(id: &str) -> String {
    if id.chars().count() <= 12 {
        return id.to_string();
    }
    format!("{}…", id.chars().take(12).collect::<String>())
}

pub(crate) fn orchestrator_agents(format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    let refused = refused_names(&fleet.profile);
    let rows: Vec<serde_json::Value> = xencode_agents_rs::ROSTER
        .iter()
        .map(|spec| {
            let installed = spec
                .binaries
                .iter()
                .find_map(|b| xencode_agents_rs::roster::which(b));
            serde_json::json!({
                "agent": spec.name,
                "installed": installed.is_some(),
                "found_at": installed.map(|p| p.display().to_string()),
                "refused_by_posture": refused.iter().any(|n| n == spec.name),
                "handover": spec.attach,
                "session_listing": spec.session_list,
                "one_shot": spec.one_shot,
                "cells_read_on": spec.read_on,
                "parked": spec.parked,
            })
        })
        .collect();
    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "posture": fleet.profile.name(),
                "worker_rule": fleet.profile.worker_rule(),
                "agents": rows,
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!(
        "{:<14} {:<10} {:<9} {:<34} WHAT XENCODE RUNS IT WITH",
        "AGENT", "INSTALLED", "POSTURE", "HANDOVER"
    );
    println!("{}", "-".repeat(102));
    for spec in xencode_agents_rs::ROSTER {
        let installed = spec
            .binaries
            .iter()
            .find_map(|b| xencode_agents_rs::roster::which(b));
        let handover = match spec.attach {
            Some(command) => command.to_string(),
            None => "no attach verb read from its help".to_string(),
        };
        println!(
            "{:<14} {:<10} {:<9} {:<34} {}",
            spec.name,
            if installed.is_some() { "yes" } else { "no" },
            if refused.iter().any(|n| n == spec.name) {
                "refused"
            } else {
                "allowed"
            },
            handover,
            spec.one_shot,
        );
    }
    println!(
        "\n  POSTURE is the {} profile's answer to work being handed to this agent, read from \
         its roster row — being on the row is the whole reason. It is not a measurement of what \
         the agent did.",
        fleet.profile.name()
    );
    println!(
        "  HANDOVER is a command the vendor's own help documents as taking over a session that \
         is already running, which is all `xencode orchestrator attach` will ever run. {} of the \
         {} rows here have none, and saying so is the point.",
        xencode_agents_rs::ROSTER
            .iter()
            .filter(|spec| spec.attach.is_none())
            .count(),
        xencode_agents_rs::ROSTER.len()
    );
    println!(
        "  Every cell was read from `--help` on the date the row records; nothing here is a \
         claim that a probe ran the agent. `xencode agents --health` is what asks a program \
         whether it answers."
    );
    orchestrator_ran_nothing();
    Ok(())
}

/// The two lists of processes xencode started here and still knows about. They
/// live in different directories, are reaped differently and outlive different
/// things, so they are printed apart: one merged count would hide which of the
/// two a stopped process belongs to.
///
/// A registry that could not be read is reported as unreadable. "No tasks" is a
/// claim about the machine, and this command is not owed one it did not observe.
pub(crate) fn orchestrator_tasks(format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    let tasks_dir = fleet.xencode_dir.join("tasks");
    let detached_dir = xencode_tui_rs::detached::detached_dir(&fleet.xencode_dir);
    let detached: Vec<(String, String, usize, String)> = fleet
        .detached
        .iter()
        .map(|id| {
            let dir = xencode_tui_rs::detached::run_dir(&fleet.xencode_dir, id);
            (
                id.clone(),
                xencode_tui_rs::detached::derive_status(&dir)
                    .label()
                    .to_string(),
                xencode_tui_rs::detached::read_rounds(&dir).len(),
                xencode_tui_rs::detached::read_spec(&dir)
                    .map(|spec| spec.model)
                    .unwrap_or_else(|| "no spec".to_string()),
            )
        })
        .collect();

    if matches!(format, OutputFormat::Json) {
        let registry = match &fleet.tasks {
            Ok(entries) => serde_json::Value::Array(
                entries
                    .values()
                    .map(|(task, status)| {
                        serde_json::json!({
                            "id": task.id,
                            "name": task.name,
                            "command": task.command,
                            "pid": task.pid,
                            "status": status.label(),
                            "killed": task.killed,
                            "source": task.source,
                            "started_at_unix_secs": task.started_at,
                        })
                    })
                    .collect(),
            ),
            Err(problem) => serde_json::json!({ "unreadable": problem }),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "tasks_dir": tasks_dir.display().to_string(),
                "tasks": registry,
                "detached_dir": detached_dir.display().to_string(),
                "detached_runs": detached.iter().map(|(id, status, rounds, model)|
                    serde_json::json!({
                        "run_id": id, "status": status, "rounds": rounds, "model": model,
                    })).collect::<Vec<_>>(),
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Background tasks · {}", tasks_dir.display());
    match &fleet.tasks {
        Err(problem) => println!(
            "  could not be read — {problem}. Nothing is claimed about running tasks: one may \
             be alive that this command cannot see."
        ),
        Ok(entries) if entries.is_empty() => {
            println!("  none recorded — the registry answered the read and held no rows.")
        }
        Ok(entries) => {
            println!(
                "{:>4}  {:<12} {:>8}  {:<12}  NAME",
                "ID", "STATUS", "PID", "SOURCE"
            );
            for (task, status) in entries.values() {
                println!(
                    "{:>4}  {:<12} {:>8}  {:<12}  {}",
                    task.id,
                    status.label(),
                    task.pid,
                    task.source.as_deref().unwrap_or("cli"),
                    task.name
                );
            }
            println!(
                "\n  STATUS is what the registry concluded by asking the operating system whether \
                 that pid is alive, so a task whose child was reaped elsewhere reads as ended."
            );
            println!("  `xencode tasks poll <id>` reads a row's output; `xencode orchestrator inspect <id>` shows the row with its last lines.");
        }
    }

    println!("\nDetached runs · {}", detached_dir.display());
    if detached.is_empty() {
        println!("  none — `xencode run \"the task\" --detach` starts one.");
    } else {
        for (id, status, rounds, model) in &detached {
            println!(
                "  {}  {:<10} {:>3} round(s)  {}",
                truncate_id(id),
                status,
                rounds,
                model
            );
        }
        println!(
            "\n  A detached run is xencode's own agent loop in a forked child; the rows above are \
             read from its spec, its round log and its exit file, never from the child's memory."
        );
    }
    orchestrator_ran_nothing();
    Ok(())
}

/// The Cargo workspace a split is judged inside: the nearest directory at or above
/// here whose manifest declares a `[workspace]`, because that is the tree whose
/// crates `cargo metadata` names. There is no `--root` to point it somewhere else,
/// on purpose — a reference built from the wrong tree is not a rough reading, it is
/// a wrong answer, and the only sign would be a score nobody questions.
pub(crate) fn split_workspace_root() -> Result<std::path::PathBuf, String> {
    let start = std::env::current_dir()
        .map_err(|e| format!("cannot read the working directory to find the workspace: {e}"))?;
    let mut dir = start.as_path();
    loop {
        let manifest = dir.join("Cargo.toml");
        let declares_workspace = std::fs::read_to_string(&manifest)
            .map(|text| text.contains("[workspace]"))
            .unwrap_or(false);
        if declares_workspace {
            return Ok(dir.to_path_buf());
        }
        dir = dir
            .parent()
            .ok_or_else(|| format!("no Cargo workspace at or above {}", start.display()))?;
    }
}

/// `OR-1` — ask for a split, score it against what the build itself requires, and
/// refuse it here rather than handing it to a scheduler.
///
/// The order of operations is the whole item. The reference is built *before*
/// anything is asked, and out of something other than the answer whenever a person
/// can name one: `--commit` gives the change's own message as the task and the
/// files git says it touched as the file set; `--path` gives a list somebody
/// wrote. With neither, the split can only be compared on ordering, and the report
/// says which half was actually measured instead of printing full coverage as if it
/// had been earned.
///
/// A refused split comes back as an error, so it exits non-zero and stops a script
/// where the warning would have scrolled past.
pub(crate) async fn orchestrator_split(
    task: Option<String>,
    commit: Option<String>,
    paths: Vec<String>,
    answer: Option<std::path::PathBuf>,
    verification: String,
    model: Option<String>,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_core_rs::{decide, Reference, Split};
    use xencode_tui_rs::decompose::{
        ask_planner, commit_change, reference_from_paths, reference_from_split, workspace_layout,
        PlannerOptions,
    };

    let root = split_workspace_root()?;
    let change = match &commit {
        Some(sha) => Some(commit_change(&root, sha)?),
        None => None,
    };
    let task_text = match &change {
        Some(read) => read.task(),
        None => task.filter(|text| !text.trim().is_empty()).ok_or_else(|| {
            "nothing to split: name the change with --task \"…\", or take a change that \
                 already happened with --commit <sha>"
                .to_string()
        })?,
    };

    // The split: read from a file when one is given, which needs no server and no
    // spend, and otherwise asked of the model just now.
    let (split, asked) = match &answer {
        Some(file) => {
            let text = std::fs::read_to_string(file)
                .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            let value = match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(value) => value,
                // A saved raw answer usually carries the model's prose around the
                // JSON, so the same reader the live path uses gets the first shot
                // at it before this calls the file unreadable.
                Err(e) => xencode_providers_rs::schema::read_answer(
                    &text,
                    &xencode_tui_rs::decompose::schema(),
                )
                .map_err(|why| format!("{} is not a split ({e}): {why}", file.display()))?,
            };
            let split = Split::from_value(&value)
                .map_err(|e| format!("{} does not hold a split: {e}", file.display()))?;
            (split, None)
        }
        None => {
            let config = XencodeConfig::load().unwrap_or_default();
            let options = PlannerOptions {
                ollama_url: config.ollama_url.clone(),
                llama_cpp_url: config.llama_cpp_url.clone(),
                temperature: config.llama_cpp_temperature.unwrap_or(0.0),
                seed: config.llama_cpp_seed.unwrap_or(42),
                ..PlannerOptions::default()
            };
            let chosen = model.unwrap_or(config.default_model.clone());
            let layout = workspace_layout(&root)?;
            let run = ask_planner(&task_text, &layout, &chosen, &options).await?;
            (run.split.clone(), Some(run))
        }
    };

    let reference: Reference = match (&change, paths.is_empty()) {
        (Some(read), _) => reference_from_paths(&root, &read.paths, &verification, &read.source())?,
        (None, false) => reference_from_paths(
            &root,
            &paths,
            &verification,
            &format!("{} files named with --path", paths.len()),
        )?,
        (None, true) => reference_from_split(&split, &root, &verification)?,
    };
    let flat = Split::flat_baseline(&reference);
    let decision = decide(&split, &reference, &flat);

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&split_json(&root, &reference, &split, &decision, &asked))
                .map_err(|e| e.to_string())?
        );
        return split_verdict(&decision);
    }

    println!("Split · {}", reference.source);
    println!("  workspace: {}", root.display());
    println!("  every unit graded by: {verification}");
    match &asked {
        Some(run) => println!(
            "  asked {} in {} ms, and got {} units back",
            run.model,
            run.elapsed_ms,
            run.split.subtasks.len()
        ),
        None => println!(
            "  read from a file, so no model was asked and nothing was spent: {}",
            answer
                .as_ref()
                .map(|f| f.display().to_string())
                .unwrap_or_default()
        ),
    }
    println!();
    println!("  the units a scheduler would run");
    for node in &split.subtasks {
        println!(
            "    {:<16} {} — {}",
            node.id,
            node.goal,
            if node.needs.is_empty() {
                "starts at once".to_string()
            } else {
                format!("waits on {}", node.needs.join(", "))
            }
        );
        println!("        writes {}", node.paths.join(", "));
    }
    println!();
    for line in decision.lines() {
        println!("{line}");
    }
    // The disagreements are not printed again here: every one that stops the split
    // is already named in the reasons above, and the pair counts below say how
    // many of each kind there were. The JSON keeps the lists themselves.
    for line in decision.score.counts() {
        println!("  {line}");
    }
    println!(
        "  the same figures for the baseline ({} units):",
        flat.subtasks.len()
    );
    for line in decision.baseline.counts() {
        println!("    {line}");
    }
    if reference.source.contains("the split's own file list") {
        println!(
            "\n  Only the ordering above is a measurement: the file list came from the split \
             itself, so nothing was compared on coverage. Name --commit <sha> or --path to get \
             that half too."
        );
    }
    println!();
    println!(
        "  Nothing was launched or scheduled by this reading. This screen decides how the work \
         would be cut up; `xencode team run <recipe>` is where a plan becomes processes."
    );
    split_verdict(&decision)
}

/// The refusal, in the words the exit code carries. A split that did not beat the
/// baseline is not a warning: the scheduler would run it, so this is the point
/// where it stops.
pub(crate) fn split_verdict(decision: &xencode_core_rs::SplitDecision) -> Result<(), String> {
    if decision.schedulable {
        return Ok(());
    }
    Err(format!(
        "this split is not scheduled: {}",
        decision.reasons.join("; ")
    ))
}

pub(crate) fn split_json(
    root: &std::path::Path,
    reference: &xencode_core_rs::Reference,
    split: &xencode_core_rs::Split,
    decision: &xencode_core_rs::SplitDecision,
    asked: &Option<xencode_tui_rs::decompose::PlannerRun>,
) -> serde_json::Value {
    let score = |s: &xencode_core_rs::SplitScore| {
        serde_json::json!({
            "files_expected": s.paths_expected,
            "files_owned": s.paths_owned,
            "coverage": s.coverage(),
            "orders_expected": s.pairs_expected,
            "orders_stated": s.agreed,
            "ordering": s.ordering(),
            "unclaimed": s.unclaimed,
            "contested": s.contested.iter().map(|(path, owners)| serde_json::json!({"path": path, "owners": owners})).collect::<Vec<_>>(),
            "backwards": s.contradicted,
            "left_silent": s.unstated,
            "inside_one_node": s.same_node,
            "not_in_the_change": s.invented,
        })
    };
    serde_json::json!({
        "workspace": root,
        "reference": {
            "source": reference.source,
            "verification": reference.verification,
            "paths": reference.paths,
            "before": reference.before,
        },
        "planner": asked.as_ref().map(|run| serde_json::json!({
            "model": run.model,
            "elapsed_ms": run.elapsed_ms,
            "prompt": run.prompt,
            "answer": run.raw,
        })),
        "units": split.subtasks.iter().map(|n| serde_json::json!({
            "id": n.id, "goal": n.goal, "paths": n.paths, "needs": n.needs, "verify": n.verify,
        })).collect::<Vec<_>>(),
        "schedulable": decision.schedulable,
        "reasons": decision.reasons,
        "score": score(&decision.score),
        "baseline": score(&decision.baseline),
    })
}

/// The dependency shape. With a recipe name it is the graph a run *would* walk,
/// beside where the last recorded run of that same recipe actually ended; with
/// none it is every run this project has recorded, in the same words the worker
/// panel uses, because the panel and this command read the same files.
pub(crate) fn orchestrator_graph(
    recipe: Option<String>,
    format: OutputFormat,
) -> Result<(), String> {
    let fleet = Fleet::read();
    match recipe {
        Some(name) => {
            let planned = plan_recipe(&fleet.recipes, &fleet.teams_dir, &name)?;
            let (runs, failed) = fleet.runs_for(&planned.fingerprint);
            let last = fleet.newest_run_matching(&planned.fingerprint);
            if matches!(format, OutputFormat::Json) {
                let mut out = plan_json(&planned, false);
                out["recorded_runs"] = serde_json::json!(runs);
                out["failed_roles_on_record"] = serde_json::json!(failed);
                out["last_run"] = last
                    .map(|run| {
                        serde_json::json!({
                            "run_id": run.run_id, "roles": run.roles,
                            "peak_concurrency": run.peak_concurrency, "elapsed_ms": run.elapsed_ms,
                        })
                    })
                    .unwrap_or(serde_json::Value::Null);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            println!(
                "Graph · {} ({})",
                planned.recipe.name,
                planned.file.display()
            );
            println!(
                "  capacity: {} at once, limited by {}",
                planned.scheduler.capacity(),
                planned.scheduler.binding().label()
            );
            for (n, wave) in planned.waves.iter().enumerate() {
                println!("  wave {}:  {}", n + 1, wave.join("  +  "));
                for id in wave {
                    let role = planned
                        .recipe
                        .roles
                        .iter()
                        .find(|r| &r.name == id)
                        .expect("the waves were built from these roles");
                    println!("      {} ← waits on {}", role.name, needs_words(role));
                }
            }
            println!(
                "  critical path: {}",
                if planned.critical.is_empty() {
                    "none".to_string()
                } else {
                    planned.critical.join(" → ")
                }
            );
            println!(
                "  bottleneck: {}",
                match &planned.bottleneck {
                    Some(id) => id.clone(),
                    None => "none — nothing forces the branches back into one line".to_string(),
                }
            );
            println!("\n  Recorded runs of this exact recipe: {runs}");
            match last {
                Some(run) => {
                    for role in &run.roles {
                        println!(
                            "      {:<14} {}  {} ms → {} ms",
                            role.name, role.status, role.started_ms, role.finished_ms
                        );
                    }
                    println!(
                        "    from run {} — peak {}, {}",
                        run.run_id,
                        run.peak_concurrency,
                        wall_clock_label(run.elapsed_ms)
                    );
                }
                None if runs == 0 => println!(
                    "    none, so nothing above has been measured on this machine. `xencode team \
                     plan {}` is the plan; the estimate line stays unknown until one runs.",
                    planned.recipe.name
                ),
                None => println!(
                    "    {runs} file(s) carry this fingerprint and none of them reads as a run."
                ),
            }
            if !failed.is_empty() {
                println!("\n  Roles that did not exit 0 on record:");
                for line in failed {
                    println!("    {line}");
                }
            }
            orchestrator_ran_nothing();
            Ok(())
        }
        None => {
            let rows = xencode_tui_rs::worker_panel::graph_rows(&fleet.runs, &fleet.runs_dir);
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "runs_dir": fleet.runs_dir.display().to_string(),
                        "runs": fleet.runs.iter().map(|file| serde_json::json!({
                            "path": file.path.display().to_string(),
                            "run": file.run.as_ref().ok().map(|run| serde_json::json!({
                                "run_id": run.run_id, "recipe": run.recipe,
                                "launch_order": run.launch_order, "roles": run.roles,
                                "peak_concurrency": run.peak_concurrency,
                                "capacity": run.capacity, "binding": run.binding,
                                "elapsed_ms": run.elapsed_ms,
                            })),
                            "problem": file.run.as_ref().err().map(|e| e.to_string()),
                        })).collect::<Vec<_>>(),
                    }))
                    .map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            println!("Recorded graphs · {}", fleet.runs_dir.display());
            for row in &rows {
                println!("  {}", row.line);
            }
            println!(
                "\n  One row per recorded run, oldest first. `xencode orchestrator inspect <run \
               id>` opens a row with the file each figure came from."
            );
            orchestrator_ran_nothing();
            Ok(())
        }
    }
}

/// What a run said while it went. A detached run keeps its own log file, so this
/// prints real bytes from disk; a team run keeps timings and exit statuses and
/// never captured its children's output, and says so rather than showing an
/// empty block as if the child had been quiet.
pub(crate) fn orchestrator_logs(
    run: Option<String>,
    lines: usize,
    format: OutputFormat,
) -> Result<(), String> {
    let fleet = Fleet::read();
    let Some(given) = run else {
        if matches!(format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "detached_runs": fleet.detached,
                    "team_runs": fleet.runs.iter().filter_map(|file| file.run.as_ref().ok())
                        .map(|run| serde_json::json!({"run_id": run.run_id, "recipe": run.recipe}))
                        .collect::<Vec<_>>(),
                }))
                .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        println!("Nothing was named, so here is what has a log to read:");
        if fleet.detached.is_empty() {
            println!(
                "  detached runs: none in {}",
                xencode_tui_rs::detached::detached_dir(&fleet.xencode_dir).display()
            );
        } else {
            println!("  detached runs (each keeps a log file):");
            for id in &fleet.detached {
                println!("    {}", id);
            }
        }
        let team: Vec<&xencode_core_rs::TeamRun> = fleet
            .runs
            .iter()
            .filter_map(|file| file.run.as_ref().ok())
            .collect();
        if team.is_empty() {
            println!("  team runs:     none in {}", fleet.runs_dir.display());
        } else {
            println!("  team runs (timings and exit status, no captured output):");
            for run in &team {
                println!("    {}  {}", run.run_id, run.recipe);
            }
        }
        println!(
            "\n  `xencode orchestrator logs <id>` reads one; the same id works for `inspect`."
        );
        return Ok(());
    };

    if let Some(id) = xencode_tui_rs::detached::resolve_run_id(&fleet.xencode_dir, &given) {
        let dir = xencode_tui_rs::detached::run_dir(&fleet.xencode_dir, &id);
        let tail = xencode_tui_rs::detached::read_log_tail(&dir, lines);
        if matches!(format, OutputFormat::Json) {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "kind": "detached", "run_id": id,
                    "log": xencode_tui_rs::detached::log_path(&dir).display().to_string(),
                    "lines": tail,
                }))
                .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        println!(
            "{} · {} line(s) from {}",
            id,
            tail.len(),
            xencode_tui_rs::detached::log_path(&dir).display()
        );
        for line in tail {
            println!("{line}");
        }
        return Ok(());
    }

    let file = fleet.runs.iter().find(|file| {
        file.run
            .as_ref()
            .ok()
            .is_some_and(|run| run.run_id == given || run.run_id.starts_with(given.as_str()))
    });
    match file {
        Some(file) => {
            let Some(run) = file.run.as_ref().ok() else {
                return Err(format!(
                    "{} is there and cannot be read as a run: {}",
                    file.path.display(),
                    file.run
                        .as_ref()
                        .err()
                        .map(|e| e.to_string())
                        .unwrap_or_default()
                ));
            };
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(run).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            println!("Team run {} · {}", run.run_id, file.path.display());
            println!("  recipe {} — approved by {}", run.recipe, run.approved_by);
            println!(
                "  launched: {}",
                if run.launch_order.is_empty() {
                    "nothing".to_string()
                } else {
                    run.launch_order.join(" → ")
                }
            );
            for role in &run.roles {
                println!(
                    "  {:<14} {:<12} {} ms → {} ms",
                    role.name, role.status, role.started_ms, role.finished_ms
                );
            }
            println!(
                "  {}",
                energy_label(
                    run.watt_hours,
                    run.cost_micros(run.cents_per_kwh),
                    run.cents_per_kwh
                )
            );
            println!(
                "\n  There is no output above because a team run never captured it: each role is a \
                 real `sh -c` child whose stdout went to the terminal that ran `xencode team \
                 run`. What the record keeps is when each role started, how it ended, and what \
                 the machine drew while it did so."
            );
            Ok(())
        }
        None => Err(format!(
            "no run {given} — not a detached run in {} and not a recorded team run in {}. \
             `xencode orchestrator logs` with no name lists both.",
            xencode_tui_rs::detached::detached_dir(&fleet.xencode_dir).display(),
            fleet.runs_dir.display()
        )),
    }
}

/// What a launch would be allowed to do, and what a person has already answered.
/// The first half is the broker's own decision on the config value this project
/// carries, computed by the same function a launch is built with; the second is
/// read off the run ledger, which is the only place an approval this project ever
/// gave is recorded.
pub(crate) fn orchestrator_permissions(
    agent: Option<String>,
    format: OutputFormat,
) -> Result<(), String> {
    use xencode_agents_rs::AgentSpec;
    use xencode_tui_rs::agent_tools::ApprovalMode;
    use xencode_tui_rs::permission_broker::{choose_grant, plan_launch, supports_prompt, Grant};
    let fleet = Fleet::read();
    let config = XencodeConfig::load().unwrap_or_default();
    let mode = ApprovalMode::parse(&config.agent_approval);
    let specs: Vec<&'static AgentSpec> = match &agent {
        Some(given) => vec![xencode_agents_rs::roster::find(given).ok_or_else(|| {
            format!(
                "no agent named {given} on the roster — `xencode orchestrator agents` lists the \
                 {} rows it holds",
                xencode_agents_rs::ROSTER.len()
            )
        })?],
        None => xencode_agents_rs::ROSTER.iter().collect(),
    };

    let mut rows = Vec::new();
    for spec in specs {
        // The roster's one-shot command, with the prompt slot dropped: what the
        // prompt says has nothing to do with what the launch may do.
        let base: Vec<String> = spec
            .one_shot
            .split_whitespace()
            .filter(|token| !token.contains("{prompt}"))
            .map(|token| token.to_string())
            .collect();
        let (argv, grant) = plan_launch(spec.name, mode, &base, &[], None);
        // The same launch with a worker asking for its own autonomy, run through
        // the same function, so the refusal to grant it is shown rather than
        // asserted.
        let self_granted: Vec<String> = vec![
            "--yolo".to_string(),
            "--permission-mode".to_string(),
            "bypassPermissions".to_string(),
        ];
        let (overruled, _) = plan_launch(spec.name, mode, &base, &self_granted, None);
        rows.push(PermissionRow {
            agent: spec.name,
            installed: spec
                .binaries
                .iter()
                .find_map(|b| xencode_agents_rs::roster::which(b))
                .is_some(),
            can_prompt_back: supports_prompt(spec.name),
            grant,
            argv,
            overruled,
        });
    }

    let asked: usize = fleet.ledgers.iter().map(|row| row.approvals.len()).sum();
    let granted: usize = fleet
        .ledgers
        .iter()
        .flat_map(|row| row.approvals.iter())
        .filter(|approval| approval.decision.granted())
        .count();
    let mut denied_by_tool: std::collections::BTreeMap<String, usize> = Default::default();
    for row in &fleet.ledgers {
        for approval in &row.approvals {
            if !approval.decision.granted() {
                *denied_by_tool.entry(approval.tool.clone()).or_insert(0) += 1;
            }
        }
    }
    let grant_words = Grant::words;

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "config_key": "agent_approval",
                "config_value": config.agent_approval,
                "mode": format!("{mode:?}"),
                "choose_grant_on_this_mode": format!("{:?}", choose_grant("claude", mode, Some("ask_user"))),
                "launches": rows.iter().map(|row| serde_json::json!({
                    "agent": row.agent,
                    "installed": row.installed,
                    "can_prompt_back": row.can_prompt_back,
                    "grant": format!("{:?}", row.grant),
                    "grant_words": grant_words(row.grant),
                    "argv": row.argv,
                    "argv_if_worker_asked_for_its_own": row.overruled,
                })).collect::<Vec<_>>(),
                "approval_history": {
                    "runs_on_record": fleet.ledgers.len(),
                    "calls_asked": asked,
                    "calls_allowed": granted,
                    "calls_denied": asked - granted,
                    "denied_by_tool": denied_by_tool,
                },
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    let config_file = XencodeConfig::config_path()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "the config file xencode could not locate".to_string());
    println!(
        "Permissions · agent_approval = \"{}\" → {:?}",
        config.agent_approval, mode
    );
    println!(
        "  That value is read from {config_file}, which is what a launch is built from. \
         Nothing here changes it."
    );
    println!(
        "\n{:<14} {:<11} {:<20} LAUNCH ARGUMENTS",
        "AGENT", "INSTALLED", "GRANT"
    );
    println!("{}", "-".repeat(96));
    for row in &rows {
        println!(
            "{:<14} {:<11} {:<20} {}",
            row.agent,
            if row.installed { "yes" } else { "no" },
            grant_words(row.grant),
            row.argv.join(" ")
        );
    }
    if rows.iter().any(|row| row.grant == Grant::Nothing) {
        println!(
            "\n  A `no autonomy flag` row is the strictest of the three grants: the worker keeps \
             its own prompting default and xencode adds nothing."
        );
    }
    if rows.iter().any(|row| row.can_prompt_back) {
        println!(
            "  The one route that would let a worker ask instead — a vendor whose approvals come \
             back to xencode through a prompt tool — needs a tool name from a running session, \
             which this command has none of, so it is not chosen here even for an agent that \
             could use it. `xencode agents` is where a session supplies one."
        );
    }
    let unchanged = rows.iter().filter(|row| row.overruled == row.argv).count();
    println!(
        "\n  The grant comes from xencode's mode and what the vendor can do, never from the \
         worker. Each row was then rebuilt the way a launch is, with the worker asking for its \
         own autonomy — `--yolo --permission-mode bypassPermissions` — and this is what that \
         launch would carry: {unchanged} of {} came back the line above, unchanged.",
        rows.len()
    );
    for row in &rows {
        if row.overruled == row.argv {
            println!(
                "    {:<14} unchanged — this line is what the mode grants; the worker's asking \
                 added nothing to it",
                row.agent
            );
        } else {
            println!("    {:<14} without: {}", row.agent, row.argv.join(" "));
            println!("    {:<14} with:    {}", "", row.overruled.join(" "));
        }
    }
    println!(
        "\n  Answered here, on record: {} run(s) in {}, {} tool call(s) asked, {} allowed, {} \
         denied.",
        fleet.ledgers.len(),
        xencode_context_rs::runs_path(&fleet.xencode_dir).display(),
        asked,
        granted,
        asked - granted
    );
    if !denied_by_tool.is_empty() {
        let top: Vec<String> = denied_by_tool
            .iter()
            .map(|(tool, n)| format!("{tool} ({n})"))
            .collect();
        println!("  Denied by tool: {}", top.join(", "));
    } else if asked > 0 {
        println!("  Nothing was denied in those runs.");
    }
    println!(
        "  Only calls xencode's own agent loop made are in that count; a vendor run through \
         `xencode agents` answers for itself and is not xencode's gate."
    );
    orchestrator_ran_nothing();
    Ok(())
}

pub(crate) struct PermissionRow {
    agent: &'static str,
    installed: bool,
    can_prompt_back: bool,
    grant: xencode_tui_rs::permission_broker::Grant,
    argv: Vec<String>,
    overruled: Vec<String>,
}

/// Money and energy, from the two places xencode measures them: the token
/// records of what a model answered, and the power counter of what a team run
/// drew from the wall.
/// One worker's cost as `orchestrator costs` says it: what its agent
/// reported, "on your plan" for a login, never a $0 nobody measured.
pub(crate) fn worker_cost(w: &xencode_team_rs::WorkerSnapshot) -> String {
    if w.on_plan {
        "on your plan, not priced".to_string()
    } else {
        match w.cost_micros {
            Some(micros) => format!("${:.4}", micros as f64 / 1_000_000.0),
            None => "no cost reported".to_string(),
        }
    }
}

pub(crate) async fn orchestrator_costs(format: OutputFormat) -> Result<(), String> {
    let workers = xencode_tui_rs::mcp_team::running_team(&xencode_context_rs::default_root()).await;
    let fleet = Fleet::read();
    let config = XencodeConfig::load().unwrap_or_default();
    let table =
        xencode_context_rs::PriceTable::load_with_lookup(&fleet.xencode_dir, config.price_lookup);
    let mut rollup = xencode_context_rs::MetricsRollup::empty();
    let records = xencode_context_rs::read_metrics(&fleet.xencode_dir);
    for row in &records {
        rollup.fold(row);
    }
    let report = xencode_context_rs::cost_of(&rollup.by_model, &table);
    let mut drawn = 0usize;
    let mut readable = 0usize;
    let mut watt_hours = 0f64;
    let mut energy_micros = 0u64;
    for file in &fleet.runs {
        if let Ok(run) = file.run.as_ref() {
            readable += 1;
            if let Some(wh) = run.watt_hours {
                drawn += 1;
                watt_hours += wh;
                energy_micros += run.cost_micros(run.cents_per_kwh).unwrap_or_default();
            }
        }
    }

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "metrics_dir": fleet.xencode_dir.display().to_string(),
                "metric_records": records.len(),
                "pricing_file": {
                    "path": table.path.display().to_string(),
                    "present": table.file_present,
                    "lookup_enabled": config.price_lookup,
                },
                "models": report.per_model.iter().map(|model| serde_json::json!({
                    "model": model.model,
                    "requests": model.tokens.requests,
                    "prompt_tokens": model.tokens.prompt_tokens,
                    "completion_tokens": model.tokens.completion_tokens,
                    "micros": model.micros,
                    "unknown_because": model.unknown_because,
                })).collect::<Vec<_>>(),
                "known_micros": report.known_micros,
                "unpriced_models": report.unpriced,
                "priced_from_listing": report.priced_from_listing,
                "team_runs": {
                    "recorded": fleet.runs.len(),
                    "with_power_counter": drawn,
                    "watt_hours": watt_hours,
                    "cents_per_kwh": config.power_cents_per_kwh,
                    "cost_micros": (drawn > 0).then_some(energy_micros),
                },
                // `null` when no engine runs for this project.
                "workers": workers.as_ref().map(|ws| ws.iter().map(|w| serde_json::json!({
                    "id": w.id,
                    "agent": w.agent,
                    "tokens": w.tokens,
                    "cost_micros": w.cost_micros,
                    "on_plan": w.on_plan,
                })).collect::<Vec<_>>()),
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }

    println!("Costs · read from {}", fleet.xencode_dir.display());
    println!(
        "  {} metric record(s); pricing from {}{}",
        records.len(),
        table.path.display(),
        if table.file_present {
            ""
        } else {
            ", which is not there"
        }
    );
    if report.per_model.is_empty() {
        println!("  models:  nothing recorded, so no token count exists to price.");
    } else {
        println!(
            "\n{:<40} {:>8} {:>10} {:>10}  {:>10}",
            "MODEL", "REQUESTS", "PROMPT", "COMPLETION", "COST"
        );
        for model in &report.per_model {
            let cost = match model.micros {
                Some(micros) => xencode_context_rs::format_usd(micros),
                None => "unpriced".to_string(),
            };
            println!(
                "{:<40} {:>8} {:>10} {:>10}  {:>10}",
                if model.model.is_empty() {
                    "(no model named)"
                } else {
                    model.model.as_str()
                },
                model.tokens.requests,
                model.tokens.prompt_tokens,
                model.tokens.completion_tokens,
                cost
            );
        }
        println!(
            "\n  priced total:    {} across {} priced model(s)",
            xencode_context_rs::format_usd(report.known_micros),
            report
                .per_model
                .iter()
                .filter(|m| m.micros.is_some())
                .count()
        );
        if !report.unpriced.is_empty() {
            println!(
                "  unpriced:        {} — their tokens are counted above and are not in the \
                 total, which is why an unknown is shown instead of a smaller number: {}",
                report.unpriced.len(),
                report.unpriced.join(", ")
            );
        }
        if !report.priced_from_listing.is_empty() {
            println!(
                "  from the fetched listing rather than pricing.json: {}",
                report.priced_from_listing.join(", ")
            );
        }
    }

    println!();
    match &workers {
        None => println!(
            "  workers:         no engine runs for this project, so no worker agent is running"
        ),
        Some(ws) if ws.is_empty() => {
            println!("  workers:         the project's engine runs none")
        }
        Some(ws) => {
            println!(
                "  workers:         as each worker's agent reported, from the project's engine"
            );
            for w in ws {
                println!("    {} ({}): {}", w.id, w.agent, worker_cost(w));
            }
        }
    }
    match fleet.runs.iter().find(|file| file.run.is_ok()) {
        None if fleet.runs.is_empty() => println!(
            "  team energy:     nothing measured — no run is recorded in {}",
            fleet.runs_dir.display()
        ),
        None => println!(
            "  team energy:     {} file(s) in {}, none readable as a run",
            fleet.runs.len(),
            fleet.runs_dir.display()
        ),
        Some(_) => println!(
            "  team energy:     {}",
            energy_label(
                (drawn > 0).then_some(watt_hours),
                (drawn > 0).then_some(energy_micros),
                config.power_cents_per_kwh
            )
        ),
    }
    if drawn > 0 {
        println!(
            "                   from {} of {} recorded run(s){}",
            drawn,
            readable,
            if drawn == readable {
                " — every one of them left a counter reading."
            } else {
                "; the rest drew no counter, which is what happened when the machine reported \
                 nothing to read."
            }
        );
        println!(
            "                   Each run stores its own tariff, so a run priced last month keeps \
             last month's price rather than being re-priced at today's."
        );
    }
    orchestrator_ran_nothing();
    Ok(())
}

/// One row opened. The id may belong to a background task, a detached run, a
/// recorded team run or a recipe name; each is looked up in the place that holds
/// it, and a miss says which places were checked rather than guessing.
pub(crate) fn orchestrator_inspect(target: &str, format: OutputFormat) -> Result<(), String> {
    let fleet = Fleet::read();
    let json = matches!(format, OutputFormat::Json);

    if let Ok(id) = target.trim().parse::<u64>() {
        let registry = tasks_registry();
        let root = registry.root().to_path_buf();
        let entries = registry
            .poll()
            .map_err(|e| format!("cannot read the task registry: {e}"))?;
        let (task, status) = entries
            .get(&id)
            .ok_or_else(|| format!("no background task #{id} in {}", root.display()))?;
        let tail = registry.output(id, 20);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "kind": "task", "id": task.id, "name": task.name, "command": task.command,
                    "pid": task.pid, "status": status.label(), "killed": task.killed,
                    "source": task.source, "started_at_unix_secs": task.started_at,
                    "output_tail": tail,
                }))
                .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        println!("Background task #{id} · {}", root.display());
        println!("  name:     {}", task.name);
        println!("  command:  {}", task.command);
        println!("  pid:      {} ({})", task.pid, status.label());
        println!(
            "  started:  {} — the registry's own stamp, in whole seconds since the epoch",
            task.started_at
        );
        println!(
            "  killed:   {}",
            if task.killed {
                "yes — this xencode stopped it"
            } else {
                "no"
            }
        );
        println!(
            "  source:   {}",
            task.source
                .as_deref()
                .unwrap_or("started from the command line")
        );
        println!("  last {} output line(s):", tail.len());
        if tail.is_empty() {
            println!("    none held — the child has written nothing xencode captured");
        }
        for line in tail {
            println!("    {line}");
        }
        orchestrator_ran_nothing();
        return Ok(());
    }

    // Looked up in that order on purpose: a team run id is `<recipe>-<number>`, so
    // a prefix search would answer for the recipe's own name and open a run nobody
    // asked for. An exact id, then the recipe, then the prefixes.
    if let Some(file) = fleet.runs.iter().find(|file| {
        file.run
            .as_ref()
            .ok()
            .is_some_and(|run| run.run_id == target)
    }) {
        return orchestrator_show_run(file, &fleet.runs_dir, json);
    }

    let named = fleet
        .recipes
        .iter()
        .filter(|f| f.recipe.as_ref().is_ok_and(|r| r.name == target))
        .count();
    if named == 1 {
        let planned = plan_recipe(&fleet.recipes, &fleet.teams_dir, target)?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan_json(&planned, false))
                    .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        print_plan(&planned, false);
        return Ok(());
    }
    if named > 1 {
        return Err(format!(
            "{named} recipes in {} are all named {target}, so there is no one plan to open. \
             `xencode orchestrator graph <name>` says which file each is in.",
            fleet.teams_dir.display()
        ));
    }

    if let Some(id) = xencode_tui_rs::detached::resolve_run_id(&fleet.xencode_dir, target) {
        let dir = xencode_tui_rs::detached::run_dir(&fleet.xencode_dir, &id);
        let spec = xencode_tui_rs::detached::read_spec(&dir);
        let rounds = xencode_tui_rs::detached::read_rounds(&dir);
        let exit = xencode_tui_rs::detached::read_exit(&dir);
        let status = xencode_tui_rs::detached::derive_status(&dir);
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "kind": "detached_run", "run_id": id, "status": status.label(),
                    "dir": dir.display().to_string(),
                    "log": xencode_tui_rs::detached::log_path(&dir).display().to_string(),
                    "spec": spec, "rounds": rounds, "exit": exit,
                }))
                .map_err(|e| e.to_string())?
            );
            return Ok(());
        }
        show_detached_run(&fleet.xencode_dir, &id)?;
        println!(
            "  log:      {}",
            xencode_tui_rs::detached::log_path(&dir).display()
        );
        println!(
            "  tokens:   {} prompt + {} completion over the whole run",
            rounds.iter().filter_map(|r| r.prompt_tokens).sum::<u64>(),
            rounds
                .iter()
                .filter_map(|r| r.completion_tokens)
                .sum::<u64>()
        );
        orchestrator_ran_nothing();
        return Ok(());
    }

    if let Some(file) = fleet.runs.iter().find(|file| {
        file.run
            .as_ref()
            .ok()
            .is_some_and(|run| run.run_id.starts_with(target))
    }) {
        return orchestrator_show_run(file, &fleet.runs_dir, json);
    }

    Err(format!(
        "nothing here is named {target}. Checked: the background tasks in {}, the recipes in \
         {}, the detached runs in {}, and the recorded team runs in {}.",
        fleet.xencode_dir.join("tasks").display(),
        fleet.teams_dir.display(),
        xencode_tui_rs::detached::detached_dir(&fleet.xencode_dir).display(),
        fleet.runs_dir.display(),
    ))
}

/// One recorded run, opened: the panel's own row with every figure's file named,
/// or the record itself when JSON was asked for. A file that is there and cannot
/// be read is shown as that, because the alternative is a not-found for a run
/// whose file is on disk.
pub(crate) fn orchestrator_show_run(
    file: &xencode_core_rs::RunFile,
    runs_dir: &std::path::Path,
    json: bool,
) -> Result<(), String> {
    if json {
        match &file.run {
            Ok(run) => println!(
                "{}",
                serde_json::to_string_pretty(run).map_err(|e| e.to_string())?
            ),
            Err(problem) => println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "path": file.path.display().to_string(), "unreadable": problem,
                }))
                .map_err(|e| e.to_string())?
            ),
        }
        return Ok(());
    }
    println!("Recorded team run · {}", file.path.display());
    for row in xencode_tui_rs::worker_panel::graph_rows(std::slice::from_ref(file), runs_dir) {
        println!("{}", row.detail());
    }
    orchestrator_ran_nothing();
    Ok(())
}

/// Re-run one role for real. This is the one verb on this surface that starts a
/// process: the role's own `sh -c` command, waited on by the same queue a team
/// run uses, at capacity one so nothing else is launched beside it.
///
/// It deliberately writes no run record. A record carries the peak concurrency
/// and the wall clock of a whole team, and a single role replayed on its own
/// would become the number the next plan quotes — which would be a measurement
/// of something that never happened.
pub(crate) async fn orchestrator_retry(
    recipe: &str,
    role: &str,
    approved_by: Option<&str>,
    format: OutputFormat,
) -> Result<(), String> {
    let fleet = Fleet::read();
    let planned = plan_recipe(&fleet.recipes, &fleet.teams_dir, recipe)?;
    let spec = planned
        .recipe
        .roles
        .iter()
        .find(|r| r.name == role)
        .ok_or_else(|| {
            format!(
                "{} has no role named `{role}`. It has: {}",
                planned.recipe.name,
                planned
                    .recipe
                    .roles
                    .iter()
                    .map(|r| r.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let who = match approved_by {
        None => {
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "approval_required": true,
                        "recipe": planned.recipe.name,
                        "role": spec.name,
                        "worker": spec.worker,
                        "command": spec.command,
                        "gate": spec.gate,
                        "needs": spec.needs,
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else {
                println!(
                    "This would re-run one role of {} for real:\n  role:    {}\n  worker:  \
                     {}\n  command: {}\n  gate:    {}\n  needs:   {}",
                    planned.recipe.name,
                    spec.name,
                    spec.worker,
                    spec.command,
                    gate_words(spec),
                    needs_words(spec)
                );
                println!(
                    "\n  Nothing has run. One role replayed is not the team: the roles above it \
                     in the graph were not re-run, so this one reads whatever the last full run \
                     left on disk, and the roles below it will not be told anything changed."
                );
                println!(
                    "\n  To do it: xencode orchestrator retry {recipe} {role} --approved-by \
                     <your name>"
                );
            }
            return Ok(());
        }
        Some(blank) if blank.trim().is_empty() => {
            return Err(
                "`--approved-by` needs a name: a re-run launches a real process, and the record \
                 of who asked for it is the point of the flag"
                    .to_string(),
            );
        }
        Some(who) => who,
    };
    planned
        .profile
        .check_worker(
            &spec.worker,
            xencode_agents_rs::is_external_worker(&spec.worker),
        )
        .map_err(|refusal| {
            format!(
                "the {} posture refuses the worker this role names, so nothing was launched: {}",
                planned.profile.name(),
                refusal.why
            )
        })?;

    let mut graph = xencode_core_rs::TaskGraph::new();
    graph.add(xencode_core_rs::TaskNode {
        id: spec.name.clone(),
        command: spec.command.clone(),
        needs: Vec::new(),
    });
    let mut manager = xencode_core_rs::TaskManager::new();
    let window = xencode_context_rs::power::PowerWindow::begin();
    let report = xencode_core_rs::Scheduler::new(1, 1)
        .run(&graph, &mut manager)
        .await
        .map_err(|e| format!("{e}\nnothing was launched"))?;
    let energy = window.finish();
    let outcome = report.outcome(&spec.name).ok_or_else(|| {
        "the scheduler reported no outcome for the role it just launched".to_string()
    })?;
    let output: Vec<String> = manager
        .list()
        .iter()
        .flat_map(|record| record.output().iter().cloned())
        .rev()
        .take(20)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let downstream: Vec<&str> = planned
        .recipe
        .roles
        .iter()
        .filter(|r| r.needs.iter().any(|n| n == &spec.name))
        .map(|r| r.name.as_str())
        .collect();

    if matches!(format, OutputFormat::Json) {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "recipe": planned.recipe.name,
                "role": spec.name,
                "worker": spec.worker,
                "command": spec.command,
                "approved_by": who,
                "status": outcome.status.label(),
                "started_ms": outcome.started_ms,
                "finished_ms": outcome.finished_ms,
                "elapsed_ms": report.elapsed_ms,
                "watt_hours": energy.total_watt_hours(),
                "cents_per_kwh": planned.tariff,
                "cost_micros": energy.cost_micros(planned.tariff),
                "output_tail": output,
                "run_record_written": false,
                "not_run": {
                    "needed_by_this_role": spec.needs,
                    "roles_that_were_waiting_on_this_one": downstream,
                },
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    println!(
        "Re-ran {} of {} · approved by {who}\n",
        spec.name, planned.recipe.name
    );
    println!("  command:  {}", spec.command);
    println!(
        "  status:   {}  ({} ms → {} ms, {} wall clock)",
        outcome.status.label(),
        outcome.started_ms,
        outcome.finished_ms,
        wall_clock_label(report.elapsed_ms)
    );
    println!(
        "  energy:   {}",
        energy_label(
            energy.total_watt_hours(),
            energy.cost_micros(planned.tariff),
            planned.tariff
        )
    );
    if !output.is_empty() {
        println!("  last {} output line(s):", output.len());
        for line in &output {
            println!("    {line}");
        }
    }
    println!();
    if spec.needs.is_empty() {
        println!("  It waits on nothing, so the graph above it is not in question.");
    } else {
        println!(
            "  It was launched without: {} — those roles did not run, so this one read whatever \
             they left on disk, which may be older than this command.",
            spec.needs.join(", ")
        );
    }
    if downstream.is_empty() {
        println!("  Nothing in the recipe waits on this role, so no other role is affected by what it just did.");
    } else {
        println!(
            "  Not re-run after it: {} — they would have been launched by the full team, and \
             this command stopped at the one role you named.",
            downstream.join(", ")
        );
    }
    println!(
        "\n  No run record was written: a record carries a whole team's wall clock and peak \
         concurrency, and quoting one role as if it were that would make the next plan of this \
         recipe estimate itself from a run that never happened. `xencode team run {}` is what \
         records.",
        planned.recipe.name
    );
    Ok(())
}

/// Stop one process this machine started and can still name. A background task by
/// its registry number, a detached run by its id — the two things xencode has a
/// pid for. Nothing else is in reach: it will not kill a process it cannot tie to
/// a record of launching it.
pub(crate) fn orchestrator_stop(target: &str) -> Result<(), String> {
    let fleet = Fleet::read();
    if let Ok(id) = target.trim().parse::<u64>() {
        let registry = tasks_registry();
        let root = registry.root().to_path_buf();
        let entries = registry
            .poll()
            .map_err(|e| format!("cannot read the task registry: {e}"))?;
        let (task, status) = entries
            .get(&id)
            .ok_or_else(|| format!("no background task #{id} in {}", root.display()))?;
        if !matches!(status, xencode_core_rs::TaskStatus::Running) {
            return Err(format!(
                "task #{id} is {}, not running — there is nothing to stop",
                status.label()
            ));
        }
        let name = task.name.clone();
        let pid = task.pid;
        registry
            .stop(id)
            .map_err(|e| format!("could not stop task #{id} (pid {pid}): {e}"))?;
        println!(
            "Stopped task #{id} ({name}, pid {pid}). The process was signalled by this command."
        );
        println!(
            "  The pid the registry holds is the `sh -c` wrapper the task was started with, so a \
             program that wrapper spawned keeps running unless it went down with it."
        );
        println!(
            "  Its record stays in the registry, so `xencode orchestrator inspect {id}` still \
             reads what it had said."
        );
        return Ok(());
    }
    stop_detached_run(&fleet.xencode_dir, target).map_err(|problem| {
        format!(
            "{problem}. This reaches a background task by its registry number and a detached run \
             by its id; a recorded team run is written after its children have all been waited on, \
             so there is no process of its own left to stop. `xencode orchestrator tasks` names \
             both kinds that can be stopped."
        )
    })
}

/// Hand this terminal to a vendor's own running session — and only that. The
/// command comes from the roster's `attach` cell, which is filled from the
/// vendor's help text for exactly the verb that takes over a session that already
/// exists. Most of the roster has none, and for those this refuses in words
/// rather than starting a second copy of the vendor and calling it a take-over.
pub(crate) fn orchestrator_attach(agent: &str, target: Option<&str>) -> Result<(), String> {
    use std::io::IsTerminal;
    use xencode_agents_rs::Handover;
    match xencode_agents_rs::handover(agent, target) {
        Handover::Ready { argv, template } => {
            if !io::stdout().is_terminal() {
                return Err(format!(
                    "`attach` hands this terminal to {}, and standard output here is not a \
                     terminal — a pipe, a redirect, a cron line or a CI step. There is no \
                     terminal to hand over, so nothing was started.",
                    argv.join(" ")
                ));
            }
            println!("Handing the terminal over · roster row `{template}`");
            println!("  running: {}", argv.join(" "));
            println!(
                "  xencode is not in the middle of this. When the vendor's own session lets go, \
                 the terminal comes back here."
            );
            let status = std::process::Command::new(&argv[0])
                .args(&argv[1..])
                .stdin(std::process::Stdio::inherit())
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .status()
                .map_err(|e| format!("{} could not be started: {e}", argv[0]))?;
            println!("\n{agent} returned {status}; the terminal is this command's again.");
            Ok(())
        }
        Handover::NoHandoverVerb {
            one_shot,
            read_on,
            session_list,
        } => Err(format!(
            "{agent} has no command that takes over a session it already has. Its help, read \
             on {read_on}, documents only a one-shot call — `{one_shot}` — which would start a \
             new vendor process rather than hand you one, so this command refuses instead of \
             pretending{}.",
            session_list
                .map(|listing| format!("; its own listing is `{listing}`, which you can run"))
                .unwrap_or_default()
        )),
        Handover::NeedsTarget {
            template,
            session_list,
        } => Err(format!(
            "attach {agent} needs the session to hand over, given as the vendor's own usage \
             line `{template}`. xencode does not choose it for you and does not list it for \
             you{} — a listing command is the vendor's own browser, and running it here would \
             take the terminal you have not agreed to hand over yet.",
            session_list
                .map(|listing| format!(": `{listing}` prints what it knows"))
                .unwrap_or_default()
        )),
        Handover::NotInstalled { template } => Err(format!(
            "{agent} is on the roster with the handover command `{template}`, but no binary for \
             it is on PATH here, so there is nothing to hand the terminal to."
        )),
        Handover::Unknown(name) => Err(format!(
            "{name} is not an agent xencode has a roster row for, so nothing here is known about \
             how to hand a terminal to one of its sessions — and xencode will not guess a \
             command and run it. `xencode orchestrator agents` lists the {} rows there are, and \
             `xencode agents` reports what a probe found on this machine.",
            xencode_agents_rs::ROSTER.len()
        )),
    }
}
