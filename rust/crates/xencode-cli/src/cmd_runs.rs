//! the recorded-run, detached-run, replay and analyze subcommands
//! `xencode runs`, detached runs, `xencode replay` and `xencode analyze`.

use super::*;

/// `.xencode` for the tree being worked on, in the same place the TUI writes its
/// recordings and its traces.
pub(crate) fn project_xencode_dir() -> std::path::PathBuf {
    xencode_context_rs::default_root().join(xencode_context_rs::XENCODE_DIR)
}

/// The first thing the person asked, for `--list`. A recording is named by a
/// number nobody remembers, so the listing shows the prompt instead.
pub(crate) fn recorded_prompt(session: &xencode_context_rs::Session) -> Option<String> {
    let opening = session.opening_messages()?;
    let last_user = opening
        .iter()
        .rev()
        .find(|message| message.get("role").and_then(|r| r.as_str()) == Some("user"))?;
    let text = match last_user.get("content")? {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" "),
        _ => return None,
    };
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.is_empty() {
        return None;
    }
    Some(if one_line.chars().count() > 72 {
        format!("{}…", one_line.chars().take(72).collect::<String>())
    } else {
        one_line
    })
}

/// The word for a count, so a listing says "1 recording" rather than
/// "1 recording(s)".
pub(crate) fn count_word(count: usize, one: &'static str, many: &'static str) -> &'static str {
    if count == 1 {
        one
    } else {
        many
    }
}

pub(crate) fn list_recordings(xencode_dir: &std::path::Path) -> Result<(), String> {
    let dir = xencode_context_rs::sessions_dir(xencode_dir);
    let ids = xencode_context_rs::list_session_ids(xencode_dir);
    if ids.is_empty() {
        println!("nothing recorded in {}", dir.display());
        println!(
            "a run is recorded only with session_recording on \
             (`xencode config set session_recording true`), and only when the model is served by \
             Ollama, llama.cpp, an OpenAI-compatible endpoint or OpenRouter."
        );
        return Ok(());
    }
    println!(
        "{} {} in {}",
        ids.len(),
        count_word(ids.len(), "recording", "recordings"),
        dir.display()
    );
    for id in ids {
        let Ok(session) = xencode_context_rs::read_session(xencode_dir, &id) else {
            // A half-written file is still a run that happened; say what is
            // unreadable about it rather than leaving it out of the list.
            println!("{id}  unreadable");
            continue;
        };
        let tools: usize = session.calls.iter().map(|call| call.tools.len()).sum();
        println!(
            "{id}  {} model {}, {} tool {}, {}",
            session.calls.len(),
            count_word(session.calls.len(), "call", "calls"),
            tools,
            count_word(tools, "call", "calls"),
            session.run.model
        );
        if let Some(prompt) = recorded_prompt(&session) {
            println!("    \"{prompt}\"");
        }
    }
    Ok(())
}

/// Read the run ledger, nothing else: no model is asked, no server is
/// dialled, so this works with everything down.
pub(crate) fn run_runs(action: Option<RunsAction>) -> Result<(), String> {
    let xencode_dir = project_xencode_dir();
    let action = action.unwrap_or(RunsAction::List {
        limit: xencode_context_rs::RUNS_WINDOW,
        format: OutputFormat::Text,
    });
    match action {
        RunsAction::List { limit, format } => {
            let rows = xencode_context_rs::recent_runs(&xencode_dir, limit);
            if rows.is_empty() {
                println!(
                    "no runs in {}",
                    xencode_context_rs::runs_path(&xencode_dir).display()
                );
                println!("a row is written when an agent turn ends in the TUI.");
                return Ok(());
            }
            if matches!(format, OutputFormat::Json) {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            for row in &rows {
                let granted = row
                    .approvals
                    .iter()
                    .filter(|a| a.decision.granted())
                    .count();
                let denied = row.approvals.len() - granted;
                let comp_info = match &row.computer {
                    Some(c) => format!("  [{c}]"),
                    None => String::new(),
                };
                println!(
                    "{}  {}  {} asked ({} allowed, {} denied)  {}{}",
                    row.run_id,
                    row.model.as_deref().unwrap_or("model ?"),
                    row.approvals.len(),
                    granted,
                    denied,
                    row.recording.as_deref().unwrap_or("no recording"),
                    comp_info,
                );
            }
            Ok(())
        }
        RunsAction::Show { run_id, format } => {
            let row = xencode_context_rs::run_by_id(&xencode_dir, &run_id).ok_or_else(|| {
                format!(
                    "no run {run_id} in {}",
                    xencode_context_rs::runs_path(&xencode_dir).display()
                )
            })?;
            let evidence = xencode_context_rs::run_evidence(&xencode_dir, &row);
            if matches!(format, OutputFormat::Json) {
                let doc = serde_json::json!({
                    "run": row,
                    "checks": evidence.checks,
                    "verified": evidence.verified(),
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?
                );
                return Ok(());
            }
            println!("run {}", row.run_id);
            println!("  model: {}", row.model.as_deref().unwrap_or("?"));
            if let Some(c) = &row.computer {
                println!("  computer: {c}");
            }
            println!(
                "  session: {}",
                row.session.as_deref().unwrap_or("(none open)")
            );
            println!(
                "  recording: {}",
                row.recording.as_deref().unwrap_or("none")
            );
            if row.approvals.is_empty() {
                println!("  approvals: none asked");
            } else {
                println!("  approvals:");
                for a in &row.approvals {
                    println!("    {} [{}]: {}", a.tool, a.class, a.decision.as_str());
                }
            }
            // The join the ledger exists to make: the run names a session, and
            // the session's verification rows are read from the EVd-1 ledger,
            // not copied here.
            if evidence.checks.is_empty() {
                println!("  checks: none — nothing verified this run");
            } else {
                println!(
                    "  checks: {} ({} passed)",
                    evidence.checks.len(),
                    evidence.checks.iter().filter(|c| c.passed()).count(),
                );
                for check in &evidence.checks {
                    println!("    exit {}  {}", check.exit_code, check.log_ref);
                }
            }
            Ok(())
        }
        RunsAction::Trailer { run_id } => {
            let row = xencode_context_rs::run_by_id(&xencode_dir, &run_id).ok_or_else(|| {
                format!(
                    "no run {run_id} in {}",
                    xencode_context_rs::runs_path(&xencode_dir).display()
                )
            })?;
            println!(
                "{}",
                xencode_context_rs::accountability_trailer(&row, env!("CARGO_PKG_VERSION"))
            );
            Ok(())
        }
    }
}

/// The flags `xencode run` takes, as the dispatcher hands them over.
pub(crate) struct DetachedCommandOptions {
    pub(crate) prompt: Option<String>,
    pub(crate) detach: bool,
    pub(crate) resume: Option<String>,
    pub(crate) list: bool,
    pub(crate) show: Option<String>,
    pub(crate) log: Option<String>,
    pub(crate) stop: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) tool_root: Option<PathBuf>,
    pub(crate) max_rounds: Option<u32>,
    pub(crate) max_minutes: Option<f64>,
    pub(crate) max_cost: Option<f64>,
    pub(crate) allow_shell: bool,
    pub(crate) tail: usize,
    pub(crate) ollama_url: Option<String>,
    pub(crate) llamacpp_url: Option<String>,
    pub(crate) child: Option<String>,
    pub(crate) child_xencode_dir: Option<PathBuf>,
}

pub(crate) use xencode_tui_rs::detached as detached_runs;

/// Run an agent turn from the command line: foreground, detached, resumed,
/// or reported on. Exactly one action per invocation — a prompt with `--log`
/// is two things being asked at once, and gets an error saying so.
pub(crate) async fn run_detached_command(options: DetachedCommandOptions) -> Result<(), String> {
    if let Some(run_id) = options.child {
        // The forked worker. Its log file carries the transcript; the
        // process exit only says whether the spec was even readable. The
        // state directory rides along because the worker's cwd is the
        // run's tree, not the project.
        let xencode_dir = options
            .child_xencode_dir
            .unwrap_or_else(project_xencode_dir);
        let exit = detached_runs::run_child(&xencode_dir, &run_id).await;
        if matches!(exit.reason, xencode_tui_rs::detached::ExitReason::Error) {
            return Err(exit.note.clone());
        }
        return Ok(());
    }
    let DetachedCommandOptions {
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
        child: _,
        child_xencode_dir: _,
    } = options;
    let actions = prompt.is_some() as u8
        + resume.is_some() as u8
        + list as u8
        + show.is_some() as u8
        + log.is_some() as u8
        + stop.is_some() as u8;
    if actions != 1 {
        return Err(
            "pick one: a prompt to run, --resume, --list, --show, --log or --stop".to_string(),
        );
    }
    let xencode_dir = project_xencode_dir();
    if list {
        return list_detached_runs(&xencode_dir);
    }
    if let Some(given) = show {
        return show_detached_run(&xencode_dir, &given);
    }
    if let Some(given) = log {
        return log_detached_run(&xencode_dir, &given, tail);
    }
    if let Some(given) = stop {
        return stop_detached_run(&xencode_dir, &given);
    }
    if let Some(given) = &resume {
        let run_id = detached_runs::resolve_run_id(&xencode_dir, given).ok_or_else(|| {
            format!(
                "no detached run {given} in {}",
                detached_runs::detached_dir(&xencode_dir).display()
            )
        })?;
        return resume_detached_run(&xencode_dir, &run_id, detach).await;
    }
    let prompt = prompt.expect("clap counted exactly one action, and it is the prompt");
    start_detached_run(
        &xencode_dir,
        &prompt,
        StartOptions {
            detach,
            model,
            tool_root,
            max_rounds,
            max_minutes,
            max_cost,
            allow_shell,
            ollama_url,
            llamacpp_url,
        },
    )
    .await
}

pub(crate) struct StartOptions {
    pub(crate) detach: bool,
    pub(crate) model: Option<String>,
    pub(crate) tool_root: Option<PathBuf>,
    pub(crate) max_rounds: Option<u32>,
    pub(crate) max_minutes: Option<f64>,
    pub(crate) max_cost: Option<f64>,
    pub(crate) allow_shell: bool,
    pub(crate) ollama_url: Option<String>,
    pub(crate) llamacpp_url: Option<String>,
}

/// Check the caps before anything runs: a cap that cannot count cannot stop,
/// so it refuses the run instead of starting one it cannot end.
pub(crate) fn build_caps(
    xencode_dir: &std::path::Path,
    model: &str,
    max_rounds: Option<u32>,
    max_minutes: Option<f64>,
    max_cost: Option<f64>,
) -> Result<xencode_tui_rs::detached::DetachedCaps, String> {
    let config = XencodeConfig::load().unwrap_or_default();
    let mut caps = xencode_tui_rs::detached::DetachedCaps {
        max_rounds: max_rounds.unwrap_or(config.agent_max_rounds.clamp(1, 64) as u32),
        ..Default::default()
    };
    if caps.max_rounds == 0 {
        return Err("--max-rounds stops nothing at 0; pass at least 1".to_string());
    }
    if let Some(minutes) = max_minutes {
        if !minutes.is_finite() || minutes <= 0.0 {
            return Err("--max-minutes stops nothing at 0; pass a positive number".to_string());
        }
        caps.max_wall_ms = (minutes * 60_000.0).round().max(1.0) as u64;
    }
    if let Some(dollars) = max_cost {
        if !dollars.is_finite() || dollars <= 0.0 {
            return Err("--max-cost stops nothing at 0; pass a positive amount".to_string());
        }
        let table =
            xencode_context_rs::PriceTable::load_with_lookup(xencode_dir, config.price_lookup);
        if table.price_for_model(model).is_none() {
            return Err(format!(
                "no price for {model} in pricing.json or the fetched listing, so --max-cost \
                 cannot count it — name a price first, or run without the cap"
            ));
        }
        caps.max_cost_micros = Some((dollars * 1_000_000.0).round().max(1.0) as u64);
    }
    Ok(caps)
}

/// Write the spec and either run it here or fork it. Both paths persist the
/// same state, so a foreground run killed from another terminal resumes the
/// same way a detached one does.
pub(crate) async fn start_detached_run(
    xencode_dir: &std::path::Path,
    prompt: &str,
    options: StartOptions,
) -> Result<(), String> {
    let config = XencodeConfig::load().unwrap_or_default();
    let model = options
        .model
        .unwrap_or_else(|| config.default_model.clone());
    if model.trim().is_empty() {
        return Err("no model: pass --model or set one in the config".to_string());
    }
    let tool_root = options
        .tool_root
        .unwrap_or_else(xencode_context_rs::default_root);
    if !tool_root.is_dir() {
        return Err(format!("{} is not a directory", tool_root.display()));
    }
    let caps = build_caps(
        xencode_dir,
        &model,
        options.max_rounds,
        options.max_minutes,
        options.max_cost,
    )?;
    let run_id = xencode_context_rs::new_run_id(prompt);
    let dir = detached_runs::run_dir(xencode_dir, &run_id);
    let spec = xencode_tui_rs::detached::DetachedSpec {
        kind: xencode_tui_rs::detached::DETACHED_KIND_RUN.to_string(),
        prompt: prompt.to_string(),
        model,
        tool_root: tool_root.to_string_lossy().into_owned(),
        caps,
        allow_shell: options.allow_shell,
        ollama_url: options.ollama_url,
        llamacpp_url: options.llamacpp_url,
        created_ms: xencode_context_rs::conversation::now_millis(),
    };
    detached_runs::write_spec(&dir, &spec)
        .map_err(|e| format!("could not write the run spec in {}: {e}", dir.display()))?;
    if !options.detach {
        let exit = detached_runs::run_child(xencode_dir, &run_id).await;
        print_detached_exit(&run_id, &exit);
        return Ok(());
    }
    let exe = std::env::current_exe().map_err(|e| format!("cannot re-run this binary: {e}"))?;
    let pid = detached_runs::spawn_child(
        &exe,
        &run_id,
        xencode_dir,
        &tool_root,
        &detached_runs::log_path(&dir),
    )
    .map_err(|e| format!("could not fork the run: {e}"))?;
    let _ = detached_runs::write_pid(&dir, pid);
    println!("detached run {run_id} (pid {pid})");
    println!(
        "watch it with `xencode run --log {run_id}`, stop it with `xencode run --stop {run_id}`"
    );
    Ok(())
}

/// Continue a run that died: same spec, same caps minus what is spent, prior
/// rounds as history. Anything but a crash is refused with the reason.
pub(crate) async fn resume_detached_run(
    xencode_dir: &std::path::Path,
    run_id: &str,
    detach: bool,
) -> Result<(), String> {
    let dir = detached_runs::run_dir(xencode_dir, run_id);
    match detached_runs::derive_status(&dir) {
        xencode_tui_rs::detached::DetachedStatus::Missing => {
            return Err(format!("no detached run {run_id}"));
        }
        xencode_tui_rs::detached::DetachedStatus::Finished(exit) => {
            return Err(format!(
                "run {run_id} already finished ({:?}); there is nothing to resume",
                exit.reason
            ));
        }
        xencode_tui_rs::detached::DetachedStatus::Stopped { .. } => {
            return Err(format!(
                "run {run_id} was stopped; resume a crash, not a decision"
            ));
        }
        xencode_tui_rs::detached::DetachedStatus::Running { pid } => {
            return Err(format!("run {run_id} is still going under pid {pid}"));
        }
        xencode_tui_rs::detached::DetachedStatus::NeverRan
        | xencode_tui_rs::detached::DetachedStatus::Crashed { .. } => {}
    }
    let spec = detached_runs::read_spec(&dir).ok_or_else(|| format!("no detached run {run_id}"))?;
    let (used_rounds, used_wall_ms, used_cost) = detached_runs::used_totals(&dir);
    // The caps are spent by the run, not the attempt: resuming into an
    // already-spent budget would start a loop its first round must end.
    if xencode_tui_rs::detached::check_caps(used_rounds, used_wall_ms, used_cost, &spec.caps)
        .is_some()
    {
        return Err(format!(
            "run {run_id} already spent its caps ({used_rounds} rounds); there is nothing to resume under"
        ));
    }
    // The cost rate is re-read, not trusted from the attempt that died: the
    // file may have changed underfoot, and spending unpriced is what the
    // start-time check exists to refuse.
    if spec.caps.max_cost_micros.is_some() {
        let config = XencodeConfig::load().unwrap_or_default();
        let table =
            xencode_context_rs::PriceTable::load_with_lookup(xencode_dir, config.price_lookup);
        if table.price_for_model(&spec.model).is_none() {
            return Err(format!(
                "no price for {} anymore, so the cost cap cannot count — name a price first",
                spec.model
            ));
        }
    }
    if !detach {
        let exit = detached_runs::run_child(xencode_dir, run_id).await;
        print_detached_exit(run_id, &exit);
        return Ok(());
    }
    let tool_root = PathBuf::from(&spec.tool_root);
    let exe = std::env::current_exe().map_err(|e| format!("cannot re-run this binary: {e}"))?;
    let pid = detached_runs::spawn_child(
        &exe,
        run_id,
        xencode_dir,
        &tool_root,
        &detached_runs::log_path(&dir),
    )
    .map_err(|e| format!("could not fork the run: {e}"))?;
    let _ = detached_runs::write_pid(&dir, pid);
    println!("resumed detached run {run_id} (pid {pid}) from {used_rounds} completed rounds");
    Ok(())
}

pub(crate) fn print_detached_exit(run_id: &str, exit: &xencode_tui_rs::detached::DetachedExit) {
    use xencode_tui_rs::detached::ExitReason;
    let why = match &exit.reason {
        ExitReason::Done => "done".to_string(),
        ExitReason::RoundCap => format!("stopped: spent its {} rounds", exit.rounds),
        ExitReason::WallCap => "stopped: spent its wall-clock budget".to_string(),
        ExitReason::CostCap => "stopped: spent its cost budget".to_string(),
        ExitReason::Stopped => "stopped on request".to_string(),
        ExitReason::Error => format!("could not run: {}", exit.note),
    };
    println!(
        "run {run_id}: {why} ({} {})",
        exit.rounds,
        count_word(exit.rounds as usize, "round", "rounds")
    );
}

pub(crate) fn list_detached_runs(xencode_dir: &std::path::Path) -> Result<(), String> {
    let ids = detached_runs::list_run_ids(xencode_dir);
    if ids.is_empty() {
        println!(
            "no detached runs in {}",
            detached_runs::detached_dir(xencode_dir).display()
        );
        println!("start one with `xencode run \"the task\" --detach`.");
        return Ok(());
    }
    for id in ids {
        let dir = detached_runs::run_dir(xencode_dir, &id);
        let status = detached_runs::derive_status(&dir);
        let rounds = detached_runs::read_rounds(&dir).len();
        let model = detached_runs::read_spec(&dir)
            .map(|spec| spec.model)
            .unwrap_or_else(|| "?".to_string());
        println!("{id}  {}  {rounds} rounds  {model}", status.label());
    }
    Ok(())
}

pub(crate) fn show_detached_run(xencode_dir: &std::path::Path, given: &str) -> Result<(), String> {
    let run_id = detached_runs::resolve_run_id(xencode_dir, given)
        .ok_or_else(|| format!("no detached run {given}"))?;
    let dir = detached_runs::run_dir(xencode_dir, &run_id);
    let spec = detached_runs::read_spec(&dir).ok_or_else(|| format!("no detached run {run_id}"))?;
    let status = detached_runs::derive_status(&dir);
    let rounds = detached_runs::read_rounds(&dir);
    println!("run {run_id}");
    println!("  status: {}", status.label());
    println!("  model: {}", spec.model);
    println!("  prompt: {}", spec.prompt);
    println!("  tree: {}", spec.tool_root);
    println!(
        "  caps: {} rounds, {} minutes wall-clock{}",
        spec.caps.max_rounds,
        spec.caps.max_wall_ms as f64 / 60_000.0,
        spec.caps
            .max_cost_micros
            .map(|micros| format!(", ${:.2} cost", micros as f64 / 1_000_000.0))
            .unwrap_or_default(),
    );
    println!(
        "  approvals: {}",
        if spec.allow_shell {
            "shell pre-approved (--allow-shell)"
        } else {
            "edits pre-approved; shell refused unasked"
        }
    );
    if rounds.is_empty() {
        println!("  rounds: none completed");
    } else {
        println!("  rounds:");
        for row in &rounds {
            println!(
                "    {:>3}  {} new turns  {} prompt + {} completion tokens",
                row.round,
                row.turns.len(),
                row.prompt_tokens
                    .map(|tokens| tokens.to_string())
                    .as_deref()
                    .unwrap_or("?"),
                row.completion_tokens
                    .map(|tokens| tokens.to_string())
                    .as_deref()
                    .unwrap_or("?"),
            );
        }
    }
    if let xencode_tui_rs::detached::DetachedStatus::Finished(exit) = &status {
        println!("  exit: {:?} after {} rounds", exit.reason, exit.rounds);
        if !exit.note.is_empty() {
            println!("  why: {}", exit.note);
        }
    }
    Ok(())
}

pub(crate) fn log_detached_run(
    xencode_dir: &std::path::Path,
    given: &str,
    tail: usize,
) -> Result<(), String> {
    let run_id = detached_runs::resolve_run_id(xencode_dir, given)
        .ok_or_else(|| format!("no detached run {given}"))?;
    let dir = detached_runs::run_dir(xencode_dir, &run_id);
    if detached_runs::read_spec(&dir).is_none() {
        return Err(format!("no detached run {run_id}"));
    }
    for line in detached_runs::read_log_tail(&dir, tail) {
        println!("{line}");
    }
    Ok(())
}

pub(crate) fn stop_detached_run(xencode_dir: &std::path::Path, given: &str) -> Result<(), String> {
    let run_id = detached_runs::resolve_run_id(xencode_dir, given)
        .ok_or_else(|| format!("no detached run {given}"))?;
    let dir = detached_runs::run_dir(xencode_dir, &run_id);
    match detached_runs::derive_status(&dir) {
        xencode_tui_rs::detached::DetachedStatus::Missing => {
            Err(format!("no detached run {run_id}"))
        }
        xencode_tui_rs::detached::DetachedStatus::Running { pid } => {
            println!("{}", detached_runs::stop_child(&dir, pid));
            Ok(())
        }
        status => Err(format!(
            "run {run_id} is {}, not running; only a running run can be stopped",
            status.label()
        )),
    }
}

pub(crate) async fn run_replay(
    run_id: Option<String>,
    list: bool,
    run_tools: bool,
    tool_root: Option<PathBuf>,
    out: Option<PathBuf>,
) -> Result<(), String> {
    let xencode_dir = project_xencode_dir();
    if list {
        return list_recordings(&xencode_dir);
    }
    let given = run_id.ok_or_else(|| {
        "name the run to replay, or pass --list to see what has been recorded".to_string()
    })?;
    let resolved = xencode_context_rs::resolve_run_id(&xencode_dir, &given)?;
    let options = xencode_tui_rs::replay::ReplayOptions {
        run_id: resolved.clone(),
        xencode_dir: xencode_dir.clone(),
        tool_root: tool_root.unwrap_or_else(xencode_context_rs::default_root),
        out_dir: out.unwrap_or_else(|| {
            xencode_dir
                .join("cache")
                .join("replays")
                .join(resolved.clone())
        }),
        run_tools,
    };
    let report = xencode_tui_rs::replay::replay(&options).await?;
    for line in report.lines() {
        println!("{line}");
    }
    if report.matches() {
        Ok(())
    } else {
        Err(format!(
            "the replay of {} did not reproduce the recording",
            report.run_id
        ))
    }
}

/// How long each structural query may take. Per rule, not per scan, so a slow
/// tree is bounded by rules × this rather than not at all.
pub(crate) const TIMED_OUT: std::time::Duration = std::time::Duration::from_secs(30);

/// `--runtime` short-circuits the rest of `analyze`: it is a different question
/// with a different engine, and folding it into the image/security inventory
/// would make one flag mean two reports.
pub(crate) fn run_runtime_analyze(
    path: &std::path::Path,
    format: OutputFormat,
) -> Result<(), String> {
    let scan = xencode_analysis_rs::runtime_hazards::analyze_runtime(path, TIMED_OUT);
    match format {
        OutputFormat::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&scan)
                    .map_err(|e| format!("cannot render JSON: {e}"))?
            );
        }
        _ => {
            println!("{}", scan.summary());
            for finding in &scan.findings {
                println!("\n{}", finding.to_report());
            }
        }
    }
    match scan.engine {
        xencode_analysis_rs::runtime_hazards::EngineStatus::Ran { .. } => Ok(()),
        xencode_analysis_rs::runtime_hazards::EngineStatus::Unavailable { .. } => {
            Err("the runtime hazard check did not run, so this is not a clean result".to_string())
        }
    }
}

pub(crate) fn run_analyze(
    path: std::path::PathBuf,
    format: OutputFormat,
    runtime: bool,
) -> Result<(), String> {
    if runtime {
        return run_runtime_analyze(&path, format);
    }
    if path.is_dir() {
        // Full-tree walk: no depth cap (an explicit user action), junk dirs
        // (target/, node_modules/, .git/…) skipped like `scan`, and every
        // failure reported on stderr instead of silently dropped.
        let mut all_issue_lists = Vec::new();
        let mut image_metas: Vec<ImageMeta> = Vec::new();
        let mut skipped = 0usize;
        let excluded = ScanOptions::default().excluded_dirs;
        let walker = walkdir::WalkDir::new(&path).into_iter();

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("  Skipping unreadable entry: {e}");
                    skipped += 1;
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let fp = entry.path();
            let in_junk = fp.components().any(|c| match c {
                std::path::Component::Normal(name) => {
                    excluded.iter().any(|x| name == std::ffi::OsStr::new(x))
                }
                _ => false,
            });
            if in_junk {
                continue;
            }
            if is_image_path(fp) {
                match analyze_image(fp) {
                    Ok(meta) => image_metas.push(meta),
                    Err(e) => {
                        eprintln!("  Skipping {}: {}", fp.display(), e);
                        skipped += 1;
                    }
                }
                continue;
            }
            // Every other file goes through analysis: unknown extensions
            // fall back to generic checks; unreadable (binary) files count
            // as skipped, visibly.
            match CodeAnalyzer::analyze_file(fp) {
                Ok(issues) => {
                    all_issue_lists.push((fp.display().to_string(), issues));
                }
                Err(e) => {
                    eprintln!("  Skipping {}: {}", fp.display(), e);
                    skipped += 1;
                    continue;
                }
            }
            // Run security scan (stderr: stdout stays valid JSON).
            if let Ok(findings) = VulnerabilityScanner::scan_file(fp) {
                if !findings.is_empty() {
                    // A pattern scan, not a security verdict (EVd-3): these are
                    // lines that match a rule, for a person to read.
                    eprintln!(
                        "  Pattern scan: {} possible issue(s) in {}",
                        findings.len(),
                        fp.display()
                    );
                }
            }
        }

        match format {
            OutputFormat::Json => {
                // Documented dir schema: issues pairs, image inventory, and
                // the skip count. Single-file shapes below are unchanged.
                let output = serde_json::json!({
                    "issues": all_issue_lists,
                    "images": image_metas,
                    "skipped": skipped,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?
                );
            }
            OutputFormat::Text => {
                let total: usize = all_issue_lists.iter().map(|(_, issues)| issues.len()).sum();
                println!("Results for: {}", path.display());
                println!("   Files analyzed: {}", all_issue_lists.len());
                println!("   Total issues:   {}", total);
                println!("   Images:         {}", image_metas.len());
                println!("   Skipped:        {}", skipped);
                for meta in &image_metas {
                    println!("\n  {}", format_image_text(meta));
                }
                for (file_path, issues) in &all_issue_lists {
                    if !issues.is_empty() {
                        println!(
                            "
  {} ({} issues)",
                            file_path,
                            issues.len()
                        );
                        for issue in issues {
                            let icon = match issue.severity.label() {
                                "critical" | "high" => "R",
                                "medium" => "Y",
                                _ => "G",
                            };
                            println!(
                                "    {} [{}] Ln{}: {} -- {}",
                                icon,
                                issue.severity.label(),
                                issue.line_number,
                                issue.message,
                                issue.suggestion
                            );
                        }
                    }
                }
            }
        }
    } else {
        // Single file — images take the intake path (metadata, not issues).
        if is_image_path(&path) {
            let meta = analyze_image(&path).map_err(|e| e.to_string())?;
            match format {
                OutputFormat::Json => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?
                    );
                }
                OutputFormat::Text => {
                    println!("Image: {}", path.display());
                    println!("   {}", format_image_text(&meta));
                }
            }
            return Ok(());
        }
        let issues = CodeAnalyzer::analyze_file(&path).map_err(|e| e.to_string())?;

        match format {
            OutputFormat::Json => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&issues).map_err(|e| e.to_string())?
                );
            }
            OutputFormat::Text => {
                println!("Analysis of: {}", path.display());
                println!("   Issues: {}", issues.len());
                for issue in &issues {
                    println!(
                        "  [{}] Ln{}: {} -- {}",
                        issue.severity.label(),
                        issue.line_number,
                        issue.message,
                        issue.suggestion
                    );
                }
            }
        }
    }
    Ok(())
}

/// Show what a plugin puts in front of every agent turn, line by line, with the
/// prompt text itself visible rather than counted.
pub(crate) fn render_declaration(report: &xencode_plugin_rs::LoadReport) -> String {
    let mut out = format!(
        "  {} v{} — {}.\n",
        report.name,
        report.version,
        if report.loaded {
            "loaded"
        } else {
            "NOT LOADED"
        }
    );
    if let Some(origin) = report.source.as_ref().and_then(|s| s.summary()) {
        out.push_str(&format!("      from {origin}\n"));
    }
    if !report.loaded {
        out.push_str(&format!(
            "      it contributes nothing: {}\n",
            report
                .reason
                .clone()
                .unwrap_or_else(|| "unknown".to_string())
        ));
        return out;
    }
    if report.prompt_text.is_empty() {
        out.push_str("      it contributes no prompt text.\n");
    } else {
        out.push_str(&format!(
            "      it puts {} line(s) ahead of the agent's system prompt on every turn:\n",
            report.prompt_text.lines().count()
        ));
        for line in report.prompt_text.lines() {
            out.push_str(&format!("      | {line}\n"));
        }
    }
    if report.before_hooks + report.after_hooks > 0 {
        out.push_str(&format!(
            "      it declares {} before hook(s) and {} after hook(s), each of which runs a \
             shell command in the workspace.\n",
            report.before_hooks, report.after_hooks
        ));
    }
    out
}

/// "abc1234def5678… (7-character prefix abc1234)" for a full SHA, plain for a
/// message that is not one.
pub(crate) fn format_commit(commit: &str) -> String {
    if commit.len() > 7 && commit.chars().all(|c| c.is_ascii_hexdigit()) {
        format!("{}…", xencode_plugin_rs::short_commit(commit))
    } else {
        commit.to_string()
    }
}
