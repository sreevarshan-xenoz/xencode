//! the team and recipe subcommands
//! `xencode team` and the recipe planning and fleet code.

use super::*;

/// Turn `--arm`, `--edit` and `--command` into candidate arm specs, refusing an
/// edit that names an unknown arm or tries to write outside its worktree.
pub(crate) fn build_candidate_arms(
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
pub(crate) fn print_competing_report(
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
pub(crate) fn team_recipes_dir() -> std::path::PathBuf {
    xencode_context_rs::default_root()
        .join(xencode_context_rs::XENCODE_DIR)
        .join(xencode_core_rs::RECIPES_DIR)
}

/// `OR-9` and `OR-10` — read the team recipes, and run one that has been
/// approved. Listing, showing and planning are reads; `run` launches roles only
/// under a name, and only after it has printed the same plan the read shows.
/// `xencode team login-optin <agent>` (TM-5): the vendor's terms line, then
/// a `yes` read from the terminal turns the login on.
pub(crate) fn team_login_optin(agent: &str) -> Result<(), String> {
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
pub(crate) fn team_exec(
    cwd: &std::path::Path,
    unset: &[String],
    command: &[String],
) -> Result<(), String> {
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
pub(crate) async fn team_clean() -> Result<(), String> {
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

pub(crate) async fn run_team(action: TeamAction) -> Result<(), String> {
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
pub(crate) fn team_runs_dir() -> std::path::PathBuf {
    xencode_context_rs::default_root()
        .join(xencode_context_rs::XENCODE_DIR)
        .join(xencode_core_rs::RUNS_DIR)
}

/// A recipe read, checked, compiled into the graph `OR-2` schedules, and put
/// beside the estimate a previous run of this exact recipe left behind. Building
/// one launches nothing and writes nothing: the runs directory is read, and a
/// directory that is not there is an empty answer.
pub(crate) struct PlannedRecipe {
    pub(crate) recipe: xencode_core_rs::TeamRecipe,
    pub(crate) file: std::path::PathBuf,
    pub(crate) graph: xencode_core_rs::TaskGraph,
    pub(crate) scheduler: xencode_core_rs::Scheduler,
    pub(crate) waves: Vec<Vec<xencode_core_rs::NodeId>>,
    pub(crate) critical: Vec<xencode_core_rs::NodeId>,
    pub(crate) bottleneck: Option<xencode_core_rs::NodeId>,
    pub(crate) fingerprint: String,
    pub(crate) estimate: Option<xencode_core_rs::Estimate>,
    pub(crate) runs_dir: std::path::PathBuf,
    pub(crate) tariff: Option<f64>,
    /// The posture this plan was read under, and the roles it refuses. A plan is
    /// still a read, so it shows the whole team and marks what would be declined;
    /// only `xencode team run` acts on the refusals.
    pub(crate) profile: xencode_core_rs::Profile,
    pub(crate) refused: Vec<xencode_core_rs::WorkerRefusal>,
}

impl PlannedRecipe {
    /// The refusal a role's worker drew, if it drew one.
    pub(crate) fn refusal_for(&self, worker: &str) -> Option<&xencode_core_rs::WorkerRefusal> {
        self.refused.iter().find(|r| r.worker == worker)
    }
}

pub(crate) fn plan_recipe(
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
pub(crate) fn print_plan(planned: &PlannedRecipe, awaiting_approval: bool) {
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
pub(crate) fn plan_json(planned: &PlannedRecipe, launches: bool) -> serde_json::Value {
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
pub(crate) async fn run_recipe(
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
pub(crate) fn wall_clock_label(ms: u64) -> String {
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
pub(crate) struct Fleet {
    pub(crate) xencode_dir: std::path::PathBuf,
    pub(crate) profile: xencode_core_rs::Profile,
    pub(crate) teams_dir: std::path::PathBuf,
    pub(crate) recipes: Vec<xencode_core_rs::RecipeFile>,
    pub(crate) recipes_problem: Option<String>,
    pub(crate) runs_dir: std::path::PathBuf,
    pub(crate) runs: Vec<xencode_core_rs::RunFile>,
    pub(crate) runs_problem: Option<String>,
    pub(crate) tasks: Result<
        std::collections::BTreeMap<u64, (xencode_core_rs::FileTask, xencode_core_rs::TaskStatus)>,
        String,
    >,
    pub(crate) detached: Vec<String>,
    pub(crate) ledgers: Vec<xencode_context_rs::RunRecord>,
}

impl Fleet {
    pub(crate) fn read() -> Fleet {
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
    pub(crate) fn newest_run_matching(
        &self,
        fingerprint: &str,
    ) -> Option<&xencode_core_rs::TeamRun> {
        self.runs
            .iter()
            .filter_map(|file| file.run.as_ref().ok())
            .filter(|run| run.fingerprint == fingerprint)
            .max_by_key(|run| run.started_at_unix_ms)
    }

    /// The newest recorded run, if there is one whose file could be read.
    pub(crate) fn newest_run(&self) -> Option<&xencode_core_rs::TeamRun> {
        self.runs
            .iter()
            .filter_map(|file| file.run.as_ref().ok())
            .max_by_key(|run| run.started_at_unix_ms)
    }

    /// How many recorded runs this fingerprint has, and how many of them ended
    /// with something other than an exit 0. Both are counts of files, not
    /// judgements about the work.
    pub(crate) fn runs_for(&self, fingerprint: &str) -> (usize, Vec<String>) {
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
pub(crate) fn energy_label(
    watt_hours: Option<f64>,
    cost_micros: Option<u64>,
    tariff: Option<f64>,
) -> String {
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
pub(crate) fn difference_label(actual_ms: u64, estimated_ms: u64) -> String {
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
pub(crate) fn recipe_status(
    file: &xencode_core_rs::RecipeFile,
) -> Result<&xencode_core_rs::TeamRecipe, xencode_core_rs::RecipeError> {
    let recipe = file.recipe.as_ref().map_err(Clone::clone)?;
    recipe.validate()?;
    Ok(recipe)
}

/// One recipe file in JSON form, including the files that are not schedulable —
/// a typing mistake in one recipe must not delete the others from the answer.
pub(crate) fn recipe_file_json(file: &xencode_core_rs::RecipeFile) -> serde_json::Value {
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
pub(crate) fn find_recipe<'a>(
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
pub(crate) fn gate_words(role: &xencode_core_rs::RoleSpec) -> String {
    if role.gate.is_empty() {
        "none — no check gates this role's output".to_string()
    } else {
        role.gate.join(", ")
    }
}

pub(crate) fn needs_words(role: &xencode_core_rs::RoleSpec) -> String {
    if role.needs.is_empty() {
        "nothing (it can start at once)".to_string()
    } else {
        role.needs.join(", ")
    }
}

/// Whether the worker a role names is on this machine — a roster lookup and a
/// `PATH` walk. The agent is never started, which is what keeps planning a read.
pub(crate) fn worker_status(role: &xencode_core_rs::RoleSpec) -> String {
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
pub(crate) fn readiness_waves(
    graph: &xencode_core_rs::TaskGraph,
) -> Vec<Vec<xencode_core_rs::NodeId>> {
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
