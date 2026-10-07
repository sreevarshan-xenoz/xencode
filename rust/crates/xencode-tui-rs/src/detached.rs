//! Detached agent runs (`LF-4`) — `.xencode/cache/detached/<run-id>/`.
//!
//! `xencode run --detach` starts an agent turn that survives the terminal: the
//! work happens in a forked child, and everything another process needs to
//! report on it later is on disk. A second queue is not built for this —
//! `GL-4` will add `xencode goal` as a row type on this queue rather than a
//! queue of its own — so the spec carries a `kind` field that says `run` and
//! nothing else yet.
//!
//! # Status is derived, never stored
//!
//! The trap the plan names is a status field that lies after a crash, so there
//! is none. What is on disk per run is a spec, one JSONL line per completed
//! round, an exit file the child writes only when it finishes, a pid hint, a
//! stop-request flag, and the child's log. [`derive_status`] reads those in
//! the order that cannot misreport: an exit file wins over everything, then a
//! live pid, then a stop request, then the rounds left behind by a crash.
//!
//! # The resume unit is a completed round
//!
//! After every trip through the agent loop the child appends the round's new
//! history turns to `rounds.jsonl`, in provider-neutral [`AgentTurn`] form.
//! A resume reads those turns back and hands them to a fresh loop as prior
//! history, so a run killed mid-round redoes at most the round that never
//! completed. Tool results are replayed from the file, never re-executed: a
//! resumed run must not run a command twice because the first attempt died
//! after it.
//!
//! # Caps are checked between rounds
//!
//! Rounds, wall-clock milliseconds and microdollars each end a run, and each
//! is checked after a round completes — a single round that never returns is
//! not stopped from inside. Wall-clock is system time, not monotonic time: a
//! laptop that slept for an hour ran an hour, which is the trap the plan
//! calls lid-close suspend. [`check_caps`] is pure so the rule can be tested
//! without a model.
//!
//! # Nobody is listening
//!
//! A detached run has no approval overlay. Like the eval harness, it runs
//! `edit-allow`: file edits are pre-approved, and anything the policy would
//! have asked a person about is refused the moment it is asked, because the
//! prompt channel has no reader. `--allow-shell` opts into `all-allow`
//! instead. Either way the refusal is recorded like any other call outcome.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use xencode_core_rs::{read_jsonl_tolerant, write_atomic};
use xencode_providers_rs::AgentTurn;

/// Which kind of row this is. `run` is the only one yet; `GL-4` adds `goal`
/// beside it, on this queue and not a second one.
pub const DETACHED_KIND_RUN: &str = "run";

/// Stop conditions for one detached run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetachedCaps {
    /// Trips through the agent loop, including a last one that fails.
    pub max_rounds: u32,
    /// Wall-clock budget in milliseconds, system time — suspend counts.
    pub max_wall_ms: u64,
    /// Microdollar budget. `None` means no cost cap, which is the only honest
    /// default while most routes report no token counts.
    #[serde(default)]
    pub max_cost_micros: Option<u64>,
}

impl Default for DetachedCaps {
    fn default() -> Self {
        Self {
            max_rounds: 16,
            max_wall_ms: 30 * 60_000,
            max_cost_micros: None,
        }
    }
}

/// What a detached run was asked to do. Written once, before the fork, and
/// never rewritten: a resume reads the same spec, so the second attempt cannot
/// quietly become a different run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetachedSpec {
    /// [`DETACHED_KIND_RUN`], until `GL-4` adds its row type.
    pub kind: String,
    /// The task in the user's own words.
    pub prompt: String,
    /// The model id as selected, prefix and all.
    pub model: String,
    /// The tree the run works in, as an absolute path.
    pub tool_root: String,
    pub caps: DetachedCaps,
    /// `true` runs `all-allow`; `false` runs `edit-allow` and refuses shell
    /// calls unasked.
    pub allow_shell: bool,
    /// Server overrides from the command line, as given. A resume replays
    /// the same addresses rather than whatever the config says by then.
    #[serde(default)]
    pub ollama_url: Option<String>,
    #[serde(default)]
    pub llamacpp_url: Option<String>,
    /// UTC epoch millis the spec was written.
    pub created_ms: u64,
}

/// What [`crate::app::agent_rounds`] hands the child after each completed
/// round: the round's number, the history turns it added, and the token
/// counts the route reported for it — `None` where the route reported none,
/// which is most routes.
#[derive(Debug, Clone)]
pub struct RoundReport {
    pub round: u32,
    pub new_turns: Vec<AgentTurn>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

/// The hook [`crate::app::agent_rounds`] calls after each completed round.
pub type RoundHook = Arc<dyn Fn(&RoundReport) + Send + Sync>;

/// One completed round, as persisted. Totals are cumulative across attempts,
/// so the last line is what a resume subtracts the caps from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetachedRound {
    /// Which trip through the loop this completed, counting from 1 across
    /// attempts — a resumed run does not renumber what already happened.
    pub round: u32,
    /// The history turns this round added, in order.
    pub turns: Vec<AgentTurn>,
    /// This round's token counts as the route reported them, if it did.
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    /// What this round cost at the run's rate, in microdollars.
    #[serde(default)]
    pub round_cost_micros: u64,
    /// Wall-clock used by the whole run up to and including this round.
    pub elapsed_ms: u64,
    /// Microdollars spent by the whole run up to and including this round.
    pub total_cost_micros: u64,
}

/// Why a detached run is no longer going.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitReason {
    /// The model stopped asking for tools.
    Done,
    /// The round budget was spent.
    RoundCap,
    /// The wall-clock budget was spent.
    WallCap,
    /// The cost budget was spent.
    CostCap,
    /// `xencode run --stop` asked and the child is dead.
    Stopped,
    /// The child never reached the loop. Free text, scrubbed on the way in.
    Error,
}

/// The file a child writes exactly once, when it finishes for any reason but
/// a kill. Its presence is what separates finished from crashed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetachedExit {
    pub reason: ExitReason,
    pub rounds: u32,
    pub elapsed_ms: u64,
    pub cost_micros: u64,
    pub ts_unix_ms: u64,
    #[serde(default)]
    pub note: String,
}

/// Which cap ended a run, for [`check_caps`] and the child's stop flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapStop {
    RoundCap,
    WallCap,
    CostCap,
}

impl CapStop {
    pub fn reason(self) -> ExitReason {
        match self {
            Self::RoundCap => ExitReason::RoundCap,
            Self::WallCap => ExitReason::WallCap,
            Self::CostCap => ExitReason::CostCap,
        }
    }

    pub fn word(self) -> &'static str {
        match self {
            Self::RoundCap => "round cap",
            Self::WallCap => "wall-clock cap",
            Self::CostCap => "cost cap",
        }
    }
}

/// The cheapest question a run can be asked: with this much spent against
/// these caps, is it over, and if so on which cap. Pure, so it is tested
/// without a model. Rounds first, then wall-clock, then cost — the order the
/// child reports them in too.
pub fn check_caps(
    used_rounds: u32,
    used_wall_ms: u64,
    used_cost_micros: u64,
    caps: &DetachedCaps,
) -> Option<CapStop> {
    if used_rounds >= caps.max_rounds.max(1) {
        return Some(CapStop::RoundCap);
    }
    if used_wall_ms >= caps.max_wall_ms.max(1) {
        return Some(CapStop::WallCap);
    }
    if let Some(max) = caps.max_cost_micros {
        if used_cost_micros >= max {
            return Some(CapStop::CostCap);
        }
    }
    None
}

/// What a round cost at a written rate: prompt tokens at the input rate plus
/// completion tokens at the output rate. The same multiplication
/// [`xencode_context_rs::pricing::cost_of`] does, without the cache split —
/// llama.cpp's cached prefix still occupied the window, and billing it at the
/// input rate can only overcount, which is the safe direction for a cap.
pub fn round_cost_micros(
    prompt_tokens: u64,
    completion_tokens: u64,
    price: &xencode_context_rs::ModelPrice,
) -> u64 {
    (prompt_tokens as f64 * price.input_usd_per_mtok
        + completion_tokens as f64 * price.output_usd_per_mtok)
        .max(0.0)
        .round() as u64
}

/// `.xencode/cache/detached`, where detached runs live. Cache, because a run
/// is working state a cleaner may take once it is finished — and because the
/// `Files` split in `xencode-config-rs` puts regenerable state here, not in
/// the state directory.
pub fn detached_dir(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join("cache").join("detached")
}

/// The directory holding one run's files.
pub fn run_dir(xencode_dir: &Path, run_id: &str) -> PathBuf {
    detached_dir(xencode_dir).join(run_id)
}

fn spec_path(dir: &Path) -> PathBuf {
    dir.join("spec.json")
}

fn rounds_path(dir: &Path) -> PathBuf {
    dir.join("rounds.jsonl")
}

fn exit_path(dir: &Path) -> PathBuf {
    dir.join("exit.json")
}

fn pid_path(dir: &Path) -> PathBuf {
    dir.join("pid")
}

fn stop_path(dir: &Path) -> PathBuf {
    dir.join("stop")
}

/// Where the child's stdout and stderr go. The only place a detached run can
/// be watched while it goes.
pub fn log_path(dir: &Path) -> PathBuf {
    dir.join("log")
}

/// Write the spec atomically, creating the run directory. The fork happens
/// after this returns, so a run with no spec is a run that was never started.
pub fn write_spec(dir: &Path, spec: &DetachedSpec) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(spec)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    write_atomic(&spec_path(dir), text.as_bytes())
}

/// Append one completed round. Each line is one round, so a kill between
/// lines loses at most the round in flight — which by definition never
/// completed.
pub fn append_round(dir: &Path, round: &DetachedRound) -> std::io::Result<()> {
    use std::io::Write;
    let path = rounds_path(dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let line = serde_json::to_string(round)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    writeln!(file, "{line}")
}

/// The exit file, written once when the child finishes. Atomic, like the
/// spec: a reader either sees the whole reason or no file at all.
pub fn write_exit(dir: &Path, exit: &DetachedExit) -> std::io::Result<()> {
    let mut exit = exit.clone();
    exit.note = xencode_context_rs::trace::redact_secrets(&exit.note);
    let text = serde_json::to_string_pretty(&exit)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    write_atomic(&exit_path(dir), text.as_bytes())
}

/// The child's pid, as a hint — liveness is always rechecked against `/proc`,
/// because a number in a file says nothing about what runs now.
pub fn write_pid(dir: &Path, pid: u32) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(pid_path(dir), format!("{pid}\n"))
}

/// A stop request. Read by status, not by the child: the child is ended with
/// a signal, and this file is what keeps that from reading as a crash.
pub fn write_stop(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(stop_path(dir), "stop\n")
}

fn read_pid(dir: &Path) -> Option<u32> {
    std::fs::read_to_string(pid_path(dir))
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// The spec, or `None` when this id names nothing.
pub fn read_spec(dir: &Path) -> Option<DetachedSpec> {
    let raw = std::fs::read_to_string(spec_path(dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Every completed round, oldest first. A line that does not parse is
/// dropped, as in every other JSONL file here: one broken round must not
/// hide the ones that completed before it.
pub fn read_rounds(dir: &Path) -> Vec<DetachedRound> {
    read_jsonl_tolerant::<DetachedRound>(&rounds_path(dir)).rows
}

/// The conversation so far, as prior history for a fresh loop: every round's
/// turns concatenated in order.
pub fn read_history(dir: &Path) -> Vec<AgentTurn> {
    read_rounds(dir)
        .into_iter()
        .flat_map(|round| round.turns)
        .collect()
}

/// The exit file, when the child finished instead of dying.
pub fn read_exit(dir: &Path) -> Option<DetachedExit> {
    let raw = std::fs::read_to_string(exit_path(dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// What a run is doing, derived on read from the exit file and `/proc` —
/// never from a stored string, so it stays honest after a crash. An exit
/// file wins over everything: a pid that was reused after the child died
/// must not resurrect it.
#[derive(Debug, Clone)]
pub enum DetachedStatus {
    /// No spec here: this id names nothing.
    Missing,
    /// The spec is written but no round completed and no child lives. Either
    /// the fork never happened or it died before the first round.
    NeverRan,
    /// No exit file and the hinted pid answers in `/proc`.
    Running { pid: u32 },
    /// No exit file, nobody home, and a stop was asked for.
    Stopped { completed: u32 },
    /// No exit file and nobody home: killed, or the machine went down.
    Crashed { completed: u32 },
    /// The child finished and said why.
    Finished(DetachedExit),
}

impl DetachedStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::NeverRan => "never ran",
            Self::Running { .. } => "running",
            Self::Stopped { .. } => "stopped",
            Self::Crashed { .. } => "crashed",
            Self::Finished(_) => "finished",
        }
    }

    /// `true` once no further round will run under this id: finished, or
    /// stopped. A crash is resumable, so it is not final.
    pub fn final_state(&self) -> bool {
        matches!(self, Self::Finished(_) | Self::Stopped { .. })
    }
}

pub fn derive_status(dir: &Path) -> DetachedStatus {
    if read_spec(dir).is_none() {
        return DetachedStatus::Missing;
    }
    if let Some(exit) = read_exit(dir) {
        return DetachedStatus::Finished(exit);
    }
    if let Some(pid) = read_pid(dir) {
        if xencode_core_rs::tasks_file::pid_alive(pid) {
            return DetachedStatus::Running { pid };
        }
    }
    let completed = read_rounds(dir).len() as u32;
    if stop_path(dir).exists() {
        return DetachedStatus::Stopped { completed };
    }
    if completed > 0 {
        return DetachedStatus::Crashed { completed };
    }
    DetachedStatus::NeverRan
}

/// One run id by full id or unambiguous prefix — the same rule `xencode
/// replay` and `xencode runs` use, so an id that works there works here.
pub fn resolve_run_id(xencode_dir: &Path, given: &str) -> Option<String> {
    let ids = list_run_ids(xencode_dir);
    if ids.iter().any(|id| id == given) {
        return Some(given.to_string());
    }
    let matches: Vec<&String> = ids.iter().filter(|id| id.starts_with(given)).collect();
    if matches.len() == 1 {
        return Some(matches[0].clone());
    }
    None
}

/// All run ids with a spec, oldest first.
pub fn list_run_ids(xencode_dir: &Path) -> Vec<String> {
    let dir = detached_dir(xencode_dir);
    let mut ids: Vec<(u64, String)> = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(spec) = read_spec(&path) else {
            continue;
        };
        ids.push((spec.created_ms, name.to_string()));
    }
    ids.sort();
    ids.into_iter().map(|(_, id)| id).collect()
}

/// The totals a resume subtracts the caps from: completed rounds, wall-clock
/// milliseconds and microdollars, each from the last persisted line.
pub fn used_totals(dir: &Path) -> (u32, u64, u64) {
    let rounds = read_rounds(dir);
    let completed = rounds.len() as u32;
    let last = rounds.last();
    (
        completed,
        last.map(|row| row.elapsed_ms).unwrap_or(0),
        last.map(|row| row.total_cost_micros).unwrap_or(0),
    )
}

fn run_brief(task: &str) -> String {
    format!(
        "You are running detached: no person will answer an approval prompt, so a tool the policy does not pre-approve comes back refused — plan around that rather than asking. Work the task to done and then stop asking for tools.\n\nTask: {task}"
    )
}

/// The child: run the spec's prompt to completion under its caps, persisting
/// every round, and leave an exit file saying why it stopped.
///
/// `xencode_dir` is the project's `.xencode` directory; the tools work in the
/// spec's `tool_root`, which is where the parent forked from.
pub async fn run_child(xencode_dir: &Path, run_id: &str) -> DetachedExit {
    use crate::app::{agent_rounds, App, LoopSink};
    use xencode_memory_rs::ConversationMemory;

    let dir = run_dir(xencode_dir, run_id);
    let now_ms = || xencode_context_rs::conversation::now_millis();
    let finish =
        |reason: ExitReason, rounds: u32, elapsed_ms: u64, cost_micros: u64, note: String| {
            DetachedExit {
                reason,
                rounds,
                elapsed_ms,
                cost_micros,
                ts_unix_ms: now_ms(),
                note,
            }
        };

    let Some(spec) = read_spec(&dir) else {
        return finish(
            ExitReason::Error,
            0,
            0,
            0,
            format!("no spec in {}", dir.display()),
        );
    };
    // My own pid, so a later status reads running while I live. A stale
    // number from a dead attempt is overwritten, not trusted.
    let _ = write_pid(&dir, std::process::id());

    let (completed, used_wall_ms, used_cost_micros) = used_totals(&dir);
    let prior_history = read_history(&dir);
    let remaining_rounds = spec.caps.max_rounds.saturating_sub(completed).max(1);

    // The rate the cost cap multiplies by, read once. Unpriced is refused
    // before the fork, so this is a lookup that must succeed — and if the
    // file changed underfoot, the run stops rather than spends unpriced.
    let table = xencode_context_rs::PriceTable::load_with_lookup(
        xencode_dir,
        xencode_config_rs::XencodeConfig::load()
            .map(|config| config.price_lookup)
            .unwrap_or(false),
    );
    let price = spec
        .caps
        .max_cost_micros
        .and_then(|_| table.price_for_model(&spec.model))
        .map(|source| source.price().clone());

    let mut config = xencode_config_rs::XencodeConfig::load().unwrap_or_default();
    config.default_model = spec.model.clone();
    if let Some(url) = &spec.ollama_url {
        config.ollama_url = url.clone();
    }
    if let Some(url) = &spec.llamacpp_url {
        config.llama_cpp_url = url.clone();
    }
    // Nobody is listening, so anything the mode does not pre-approve is
    // refused the moment it is asked for — the eval harness runs headless
    // the same way.
    config.agent_approval = if spec.allow_shell {
        "all-allow".to_string()
    } else {
        "edit-allow".to_string()
    };
    let mut app = App::with_config_and_memory(
        config,
        ConversationMemory::new(50),
        std::path::PathBuf::new(),
    );
    app.persist_config = false;

    let tool_root = PathBuf::from(&spec.tool_root);
    let assembly = app.delegated_context(&tool_root, &spec.prompt, run_brief);
    let messages = App::chat_messages(assembly.turns);
    // The spawn sink, not chat: it reports each call, each round's answer as
    // one log line, and — crucially for a run nobody watches — the error a
    // failed round died on, which the chat sink leaves for the transcript.
    // The id is 0 because the run's real name is the detached id, already in
    // every log line's file.
    let mut run = app.agent_run_with_id(
        LoopSink::Spawn(0),
        messages,
        &spec.prompt,
        run_id.to_string(),
    );
    run.tool_root = tool_root;
    run.trace_dir = xencode_dir.to_path_buf();
    run.max_rounds = remaining_rounds.clamp(1, 64) as usize;
    run.resume_history = prior_history;
    // A dropped receiver: with no TUI attached a prompt would hang the run,
    // so anything not pre-approved is refused instead.
    if !spec.allow_shell {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        run.approval.prompts = tx;
    }
    // L-7's repair gate stays on: a detached run that edits files is
    // verified by the project's own checks like any other turn.

    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_cause: Arc<Mutex<Option<CapStop>>> = Arc::new(Mutex::new(None));
    let totals = Arc::new(Mutex::new((used_cost_micros, completed)));
    let rate = price.clone();
    let wall_start_ms = now_ms();
    let wall_already_ms = used_wall_ms;
    let caps = spec.caps.clone();
    let hook_dir = dir.clone();
    let hook_flag = stop_flag.clone();
    let hook_cause = stop_cause.clone();
    let hook_totals = totals.clone();
    run.stop_flag = Some(stop_flag.clone());
    run.round_hook = Some(Arc::new(move |report: &RoundReport| {
        let mut totals = hook_totals.lock().unwrap();
        let round_no = completed + report.round;
        let round_cost = match (report.prompt_tokens, report.completion_tokens) {
            (Some(prompt), Some(completion)) => rate
                .as_ref()
                .map(|price| round_cost_micros(prompt, completion, price))
                .unwrap_or(0),
            _ => 0,
        };
        totals.0 += round_cost;
        totals.1 += 1;
        let elapsed = wall_already_ms + now_ms().saturating_sub(wall_start_ms);
        let row = DetachedRound {
            round: round_no,
            turns: report.new_turns.clone(),
            prompt_tokens: report.prompt_tokens,
            completion_tokens: report.completion_tokens,
            round_cost_micros: round_cost,
            elapsed_ms: elapsed,
            total_cost_micros: totals.0,
        };
        // A round that cannot be written down did not happen as far as a
        // resume is concerned — but the tools already ran, so stopping the
        // loop would strand them. The failure goes in the log; the loop
        // continues and the next round carries the turns again.
        if let Err(e) = append_round(&hook_dir, &row) {
            eprintln!("detached run: could not persist round {round_no}: {e}");
        }
        if let Some(stop) = check_caps(totals.1, elapsed, totals.0, &caps) {
            *hook_cause.lock().unwrap() = Some(stop);
            hook_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }));

    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    // The log is append-only across attempts: a resume's rounds follow the
    // crash in the same file, in the order they happened.
    let log_path = log_path(&dir);
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path);
    if let Ok(log) = log.as_mut() {
        use std::io::Write;
        let _ = writeln!(
            log,
            "=== attempt at {run_id} ({} completed rounds already) ===",
            completed
        );
    }
    agent_rounds(run, tx).await;
    while let Ok(line) = rx.try_recv() {
        if let Ok(log) = log.as_mut() {
            use std::io::Write;
            let _ = writeln!(log, "{line}");
        }
    }

    let (total_cost, total_rounds) = *totals.lock().unwrap();
    let elapsed = wall_already_ms + now_ms().saturating_sub(wall_start_ms);
    let reason = stop_cause
        .lock()
        .unwrap()
        .map(CapStop::reason)
        .or_else(|| check_caps(total_rounds, elapsed, total_cost, &spec.caps).map(CapStop::reason))
        .unwrap_or(ExitReason::Done);
    let exit = finish(reason, total_rounds, elapsed, total_cost, String::new());
    let _ = write_exit(&dir, &exit);
    exit
}

/// How long `run log --tail` keeps by default. The log holds every attempt,
/// so the tail is what a person asks for; the whole file is one read away.
pub const LOG_TAIL_LINES: usize = 40;

/// The last `lines` lines of the child's log, oldest first of the tail.
pub fn read_log_tail(dir: &Path, lines: usize) -> Vec<String> {
    let Ok(raw) = std::fs::read_to_string(log_path(dir)) else {
        return Vec::new();
    };
    let all: Vec<String> = raw.lines().map(str::to_string).collect();
    let skip = all.len().saturating_sub(lines.max(1));
    all.into_iter().skip(skip).collect()
}

/// Detach the current process's child: a fork of this binary that outlives
/// the terminal, with its output in the run's log. Returns the child's pid.
///
/// `xencode_dir` is passed through rather than re-derived, because the child
/// runs with its working directory in the run's tree — re-deriving it there
/// would point at the tree's own `.xencode`, not the project's.
pub fn spawn_child(
    _exe: &Path,
    run_id: &str,
    xencode_dir: &Path,
    cwd: &Path,
    log: &Path,
) -> std::io::Result<u32> {
    use std::os::fd::AsRawFd;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)?;
    let log_fd = file.as_raw_fd();
    let xencode_dir_buf = xencode_dir.to_path_buf();
    let run_id_str = run_id.to_string();
    let cwd_buf = cwd.to_path_buf();

    // SAFETY: fork creates a detached worker child without re-executing this binary (AE-7).
    // The child sets sid, redirects stdout/stderr, and calls run_child directly in tokio.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if pid == 0 {
        unsafe {
            libc::setsid();
            libc::dup2(log_fd, libc::STDOUT_FILENO);
            libc::dup2(log_fd, libc::STDERR_FILENO);
        }
        let _ = std::env::set_current_dir(&cwd_buf);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("detached worker tokio runtime");
        rt.block_on(async move {
            let _ = run_child(&xencode_dir_buf, &run_id_str).await;
        });
        std::process::exit(0);
    }
    Ok(pid as u32)
}

/// Send the run's child a `SIGTERM` and wait briefly for it to die. The
/// `stop` file is written first, so whatever the signal achieves already,
/// the run reads as stopped rather than crashed.
pub fn stop_child(dir: &Path, pid: u32) -> String {
    let _ = write_stop(dir);
    // SAFETY: `kill` with `SIGTERM` sends a signal and nothing else.
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }
    for _ in 0..20 {
        if !xencode_core_rs::tasks_file::pid_alive(pid) {
            return format!("stopped (pid {pid} is dead)");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    format!("asked pid {pid} to stop and it is still alive; it may be ignoring SIGTERM")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> DetachedCaps {
        DetachedCaps {
            max_rounds: 4,
            max_wall_ms: 60_000,
            max_cost_micros: Some(1_000_000),
        }
    }

    #[test]
    fn no_cap_spent_is_no_stop() {
        assert_eq!(check_caps(0, 0, 0, &caps()), None);
        assert_eq!(check_caps(3, 59_999, 999_999, &caps()), None);
    }

    #[test]
    fn each_cap_fires_on_its_own_measure() {
        assert_eq!(
            check_caps(4, 0, 0, &caps()),
            Some(CapStop::RoundCap),
            "the fourth completed round spends four rounds"
        );
        assert_eq!(check_caps(0, 60_000, 0, &caps()), Some(CapStop::WallCap));
        assert_eq!(check_caps(0, 0, 1_000_000, &caps()), Some(CapStop::CostCap));
    }

    #[test]
    fn rounds_win_when_everything_is_spent() {
        assert_eq!(
            check_caps(9, 99_000, 9_000_000, &caps()),
            Some(CapStop::RoundCap),
            "the report names one cap, in rounds-wall-cost order"
        );
    }

    #[test]
    fn no_cost_cap_is_no_cost_stop() {
        let mut free = caps();
        free.max_cost_micros = None;
        assert_eq!(check_caps(1, 1_000, u64::MAX, &free), None);
    }

    #[test]
    fn a_round_costs_prompt_at_input_plus_completion_at_output() {
        let price = xencode_context_rs::ModelPrice {
            input_usd_per_mtok: 2.0,
            output_usd_per_mtok: 8.0,
            cached_input_usd_per_mtok: None,
        };
        assert_eq!(round_cost_micros(1_000_000, 1_000_000, &price), 10_000_000);
        assert_eq!(round_cost_micros(0, 0, &price), 0);
    }

    fn temp_run(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xe-detached-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn spec() -> DetachedSpec {
        DetachedSpec {
            kind: DETACHED_KIND_RUN.to_string(),
            prompt: "fix the typo".to_string(),
            model: "llamacpp:qwen/qwen3-8b".to_string(),
            tool_root: "/tmp".to_string(),
            caps: caps(),
            allow_shell: false,
            ollama_url: None,
            llamacpp_url: None,
            created_ms: 1_700_000_000_000,
        }
    }

    #[test]
    fn a_spec_with_no_child_has_never_run() {
        let dir = temp_run("never");
        write_spec(&dir, &spec()).unwrap();
        assert!(matches!(derive_status(&dir), DetachedStatus::NeverRan));
    }

    #[test]
    fn an_unknown_id_is_missing_not_crashed() {
        let dir = temp_run("missing").join("nope");
        assert!(matches!(derive_status(&dir), DetachedStatus::Missing));
    }

    #[test]
    fn rounds_without_an_exit_or_a_life_are_a_crash() {
        let dir = temp_run("crashed");
        write_spec(&dir, &spec()).unwrap();
        append_round(
            &dir,
            &DetachedRound {
                round: 1,
                turns: Vec::new(),
                prompt_tokens: None,
                completion_tokens: None,
                round_cost_micros: 0,
                elapsed_ms: 1_000,
                total_cost_micros: 0,
            },
        )
        .unwrap();
        match derive_status(&dir) {
            DetachedStatus::Crashed { completed } => assert_eq!(completed, 1),
            other => panic!("a dead run with rounds is a crash, not {}", other.label()),
        }
    }

    #[test]
    fn an_exit_file_wins_over_a_live_pid() {
        let dir = temp_run("exitwins");
        write_spec(&dir, &spec()).unwrap();
        write_pid(&dir, std::process::id()).unwrap();
        write_exit(
            &dir,
            &DetachedExit {
                reason: ExitReason::Done,
                rounds: 2,
                elapsed_ms: 3_000,
                cost_micros: 0,
                ts_unix_ms: 1_700_000_000_000,
                note: String::new(),
            },
        )
        .unwrap();
        match derive_status(&dir) {
            DetachedStatus::Finished(exit) => assert_eq!(exit.reason, ExitReason::Done),
            other => panic!("an exit file is finished, not {}", other.label()),
        }
    }

    #[test]
    fn a_stop_request_reads_as_stopped_not_crashed() {
        let dir = temp_run("stopped");
        write_spec(&dir, &spec()).unwrap();
        write_pid(&dir, 4000000000).unwrap();
        write_stop(&dir).unwrap();
        match derive_status(&dir) {
            DetachedStatus::Stopped { completed } => assert_eq!(completed, 0),
            other => panic!("an asked stop is stopped, not {}", other.label()),
        }
    }

    #[test]
    fn history_is_every_rounds_turns_in_order() {
        use xencode_providers_rs::{AgentTurn, ToolCall};
        let dir = temp_run("history");
        let call = ToolCall {
            id: "call_0".to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": "a"}),
        };
        for round in 1..=2 {
            append_round(
                &dir,
                &DetachedRound {
                    round,
                    turns: vec![
                        AgentTurn::Assistant {
                            text: format!("t{round}"),
                            calls: vec![call.clone()],
                        },
                        AgentTurn::ToolResult {
                            id: "call_0".to_string(),
                            content: "ok".to_string(),
                        },
                    ],
                    prompt_tokens: None,
                    completion_tokens: None,
                    round_cost_micros: 0,
                    elapsed_ms: u64::from(round) * 1_000,
                    total_cost_micros: 0,
                },
            )
            .unwrap();
        }
        let history = read_history(&dir);
        assert_eq!(history.len(), 4);
        let (used_rounds, used_ms, used_cost) = used_totals(&dir);
        assert_eq!((used_rounds, used_ms, used_cost), (2, 2_000, 0));
    }

    #[test]
    fn a_broken_round_line_hides_nothing_but_itself() {
        let dir = temp_run("broken");
        append_round(
            &dir,
            &DetachedRound {
                round: 1,
                turns: Vec::new(),
                prompt_tokens: None,
                completion_tokens: None,
                round_cost_micros: 0,
                elapsed_ms: 1_000,
                total_cost_micros: 0,
            },
        )
        .unwrap();
        use std::io::Write;
        writeln!(
            std::fs::OpenOptions::new()
                .append(true)
                .open(rounds_path(&dir))
                .unwrap(),
            "not json at all"
        )
        .unwrap();
        assert_eq!(read_rounds(&dir).len(), 1);
    }
}
