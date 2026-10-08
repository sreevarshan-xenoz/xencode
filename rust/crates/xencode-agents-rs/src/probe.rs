//! Running one agent headless and recording what actually came back.
//!
//! Every field in a [`RunCapture`] is something a process did. Nothing here reads
//! documentation, and nothing here fills a cell it did not earn — see the crate
//! documentation for why that is the whole design.

use crate::roster::{which, AgentSpec, Provenance, ROSTER};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// How much of an agent's output to keep.
///
/// A transcript of a model answering is small; a transcript of a model failing
/// can be enormous. 64 KiB keeps the useful part and bounds the rest, and
/// [`RunCapture::stdout_truncated`] says when the cap was reached, so a truncated
/// capture is never mistaken for a complete one.
const OUTPUT_CAP: usize = 64 * 1024;

/// Values that look like credentials, replaced before anything is written down.
///
/// Agent CLIs echo their environment into diagnostics, and a probe that saves
/// those to disk would be writing secrets to a file the operator did not ask for.
const SECRET_PATTERNS: &[(&str, &str)] = &[
    // `sk-` + 20 or more characters, the shape every vendor uses.
    (r"sk-[A-Za-z0-9_\-]{20,}", "[redacted:key]"),
    // `ghp_`/`gho_` GitHub tokens.
    (r"gh[pousr]_[A-Za-z0-9]{20,}", "[redacted:github-token]"),
    // Google API keys.
    (r"AIza[A-Za-z0-9_\-]{30,}", "[redacted:google-key]"),
    // `Bearer <something long>`.
    (r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}", "Bearer [redacted]"),
    // JWTs, which vendor session files are full of.
    (
        r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",
        "[redacted:jwt]",
    ),
];

/// Replace anything that looks like a credential.
pub fn redact(text: &str) -> String {
    let mut out = text.to_string();
    for (pattern, replacement) in SECRET_PATTERNS {
        if let Ok(re) = regex::Regex::new(pattern) {
            out = re.replace_all(&out, *replacement).into_owned();
        }
    }
    out
}

/// What the operator allowed this run to do.
#[derive(Debug, Clone)]
pub struct ProbeOptions {
    /// Only probe these agents. Empty means the whole roster.
    pub only: Vec<String>,
    /// Per-agent wall-clock limit.
    pub timeout: Duration,
    /// Scratch directory for the task fixture.
    pub workdir: std::path::PathBuf,
    /// The read-only task to hand each agent.
    pub task: String,
    /// Run every selected agent at once instead of one after another.
    ///
    /// `AR-1` asks for the cost of a five-worker fan-out, and that question
    /// cannot be answered by a sequential run: five agents in series measures
    /// arithmetic, not concurrency. A fan-out is only ever run on a read-only
    /// task, and it is opt-in, because the point of the flag is to spend real
    /// concurrent requests against whatever accounts are configured.
    pub fan_out: bool,
    /// How many times to run each agent.
    ///
    /// One is a reading; two is a check. `AR-1`'s done-when asks for cells that
    /// came from a help screen to be marked as such, and the gap this closes is
    /// the one next to it: nothing has been compared against a *second* run, so
    /// a single observation has been standing in for a fact.
    pub repeat: u32,
}

impl ProbeOptions {
    /// A safe default: no agent is allowed to spend, and the timeout is short
    /// enough that a hung CLI cannot hold a working session.
    pub fn conservative(workdir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            only: Vec::new(),
            timeout: Duration::from_secs(60),
            workdir: workdir.into(),
            task: default_task().to_string(),
            repeat: 1,
            fan_out: false,
        }
    }
}

/// The task every agent gets.
///
/// Read-only by construction and answerable without a model: it names one file
/// and asks what is in it. An agent that runs it touches nothing, and an agent
/// that cannot answer says so, which is the observation.
pub fn default_task() -> &'static str {
    "Read the file notes.txt in this directory and reply with the single word it contains. \
     Do not create, edit or delete any file."
}

/// One event seen on an agent's machine-readable stream.
#[derive(Debug, Clone, Serialize)]
pub struct ObservedEvent {
    /// The event's own name, however the vendor spelled it.
    pub kind: String,
    /// The line it came from, redacted and capped.
    pub excerpt: String,
}

/// What one run of one agent produced.
#[derive(Debug, Clone, Serialize)]
pub struct RunCapture {
    pub agent: String,
    /// `None` when no binary was found.
    pub binary: Option<String>,
    /// The version string, when the agent printed one.
    pub version: Option<String>,
    /// The argv actually executed, with the prompt replaced by a marker so the
    /// report does not repeat the task into every row.
    pub argv: Vec<String>,
    /// The directory the agent was launched in. Recorded because an agent that
    /// resolves paths against somewhere else — its own project root rather than
    /// this — will say so here instead of quietly reading the wrong file.
    pub workdir: String,
    pub exit_code: Option<i32>,
    pub duration_ms: u128,
    pub stdout: String,
    pub stderr: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    /// Events read from a machine-readable stream, when the output was one.
    pub events: Vec<ObservedEvent>,
    /// Whether the stream looked machine-readable at all. `crush` is expected to
    /// be `false`, and that is a finding rather than a failure.
    pub stream_recognised: bool,
    /// A session id, if one appeared anywhere in the output.
    pub session_id: Option<String>,
    /// Whether the run stopped on an authentication check. A real observation,
    /// and the common outcome with no credentials.
    pub stopped_on_auth: bool,
    /// Whether the tool asked for approval, and how it said so.
    pub permission_signal: Option<String>,
    /// Tokens and money, read out of the run's own stream.
    ///
    /// Absent when the agent said nothing about usage, which is a finding and
    /// not a gap in the reader: on 2026-10-02 `cursor-agent` reported none at
    /// all. Every field is `None` rather than `0` where the agent did not report
    /// it, so "this agent charges nothing" is never printed as "this run cost
    /// nothing".
    pub usage: Option<Usage>,
    /// Which model answered, when the stream said so.
    ///
    /// Most do not, and that is worth knowing before anything routes by
    /// capability: on 2026-10-02 four of the seven working agents named a model
    /// or provider, and `cline` named only its own brand. `None` here means the
    /// question is unanswered, not that the answer was empty.
    pub model: Option<String>,
    /// How much of the matrix this run can speak to.
    pub provenance: Provenance,
    /// Why the run did not produce an answer, when it did not.
    pub failure: Option<String>,
}

/// What a run cost, in whatever units that particular agent reports.
///
/// Six working agents report usage in six shapes on 2026-10-02: `cline` sends a
/// `usage` event with `totalCost` beside the token counts, `opencode` and its
/// `kilo` fork put a `tokens` object on each `step_finish`, `agy` sends
/// snake_case `input_tokens`, and `kiro-cli` meters *credits* with no token count
/// at all. This is the first cut at one shape for them, and it is deliberately
/// not lossy: a unit the reader does not recognise is carried as a string rather
/// than dropped or guessed at.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    /// What the agent charged, when it said. `cline` said `0`.
    pub cost: Option<f64>,
    /// The unit `cost` is in, when it is not a currency.
    pub cost_unit: Option<String>,
    /// Tokens summed over every event that reported any.
    pub summed_over_events: u64,
    /// Whether a run-level summary restated the totals. When true the totals
    /// come from that summary rather than from the sum of per-step events.
    pub saw_summary: bool,
    /// How many lines carried a metered charge.
    pub metered_events: u64,
}

impl Usage {
    /// One line for a report, naming what is missing rather than rounding it to
    /// zero.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(n) = self.input_tokens {
            parts.push(format!("in {n}"));
        }
        if let Some(n) = self.output_tokens {
            parts.push(format!("out {n}"));
        }
        if let Some(n) = self.reasoning_tokens {
            parts.push(format!("reasoning {n}"));
        }
        if let Some(n) = self.cache_read_tokens {
            parts.push(format!("cache read {n}"));
        }
        match (self.cost, &self.cost_unit) {
            (Some(c), Some(u)) => parts.push(format!("{c:.6} {u}")),
            (Some(c), None) => parts.push(format!("{c}")),
            _ => parts.push("no cost reported".to_string()),
        }
        format!(
            "{} (summed over {} event(s))",
            parts.join(", "),
            self.summed_over_events
        )
    }
}

impl RunCapture {
    /// One line for the report.
    pub fn summary(&self) -> String {
        match (&self.binary, &self.failure) {
            (None, _) => format!("{}: not installed", self.agent),
            (Some(_), Some(why)) => format!(
                "{}: exit {:?} after {} ms — {why}",
                self.agent, self.exit_code, self.duration_ms
            ),
            (Some(_), None) => format!(
                "{}: exit {:?} after {} ms, {} event(s){}{}",
                self.agent,
                self.exit_code,
                self.duration_ms,
                self.events.len(),
                if self.stopped_on_auth {
                    ", stopped on auth"
                } else {
                    ""
                },
                if self.stream_recognised {
                    ""
                } else {
                    ", no machine-readable stream"
                },
            ),
        }
    }
}

/// One fact about an agent, compared across its runs.
///
/// The comparison is deliberately blunt: a fact is `stable` when every run agreed
/// and `varied` when they did not, and a varying fact is *reported* with the
/// values seen rather than averaged into something comfortable. A vocabulary that
/// comes back different twice is a finding about the agent, not noise to smooth
/// over.
#[derive(Debug, Clone, Serialize)]
pub struct FactCheck {
    pub fact: &'static str,
    pub stable: bool,
    /// The distinct values seen, in first-seen order.
    pub values: Vec<String>,
}

impl FactCheck {
    fn new(fact: &'static str, values: Vec<String>) -> Self {
        // First-seen order, so two runs of the same thing print the same line.
        let mut distinct: Vec<String> = Vec::new();
        for value in values {
            if !distinct.contains(&value) {
                distinct.push(value);
            }
        }
        FactCheck {
            fact,
            stable: distinct.len() <= 1,
            values: distinct,
        }
    }
}

/// What repeated runs of one agent showed.
#[derive(Debug, Clone, Serialize)]
pub struct Stability {
    pub agent: String,
    pub runs: usize,
    /// The fact only runs can answer, and the one that costs the most to be
    /// wrong about: the event vocabulary.
    pub checks: Vec<FactCheck>,
}

impl Stability {
    /// The vocabulary line, phrased for a report.
    pub fn summary(&self) -> String {
        let vocabulary = self.checks.iter().find(|c| c.fact == "event vocabulary");
        let Some(vocabulary) = vocabulary else {
            return format!("{}: {} run(s), no comparison made", self.agent, self.runs);
        };
        let verdict = if vocabulary.stable {
            "stable"
        } else {
            "VARIED between runs"
        };
        format!(
            "{}: {} run(s), vocabulary {verdict}: {}",
            self.agent,
            self.runs,
            vocabulary.values.join(" | ")
        )
    }
}

/// Compare repeated runs of one agent.
pub fn check_stability(agent: &str, runs: &[RunCapture]) -> Stability {
    let vocabulary = runs
        .iter()
        .map(|c| {
            let mut kinds: Vec<&str> = c.events.iter().map(|e| e.kind.as_str()).collect();
            kinds.sort_unstable();
            kinds.dedup();
            if kinds.is_empty() {
                "(none)".to_string()
            } else {
                kinds.join(", ")
            }
        })
        .collect();
    let outcome = runs
        .iter()
        .map(|c| match c.provenance {
            Provenance::Observed => "ran".to_string(),
            Provenance::RunFailed => c
                .failure
                .clone()
                .unwrap_or_else(|| "failed".into())
                .to_lowercase(),
            other => other.as_str().to_string(),
        })
        .collect();
    // Presence, not value. A session id is *supposed* to differ between two runs
    // — comparing the values reported "VARIED" for claude, which said nothing
    // except that it issues a fresh id each time. Whether one is issued at all is
    // the fact worth checking; the specific id is not.
    let session = runs
        .iter()
        .map(|c| {
            if c.session_id.is_some() {
                "present".to_string()
            } else {
                "absent".to_string()
            }
        })
        .collect();
    let stream = runs
        .iter()
        .map(|c| if c.stream_recognised { "yes" } else { "no" }.to_string())
        .collect();
    let count = runs.iter().map(|c| c.events.len().to_string()).collect();
    Stability {
        agent: agent.to_string(),
        runs: runs.len(),
        checks: vec![
            FactCheck::new("event vocabulary", vocabulary),
            FactCheck::new("outcome", outcome),
            FactCheck::new("session id", session),
            FactCheck::new("machine-readable stream", stream),
            FactCheck::new("event count", count),
        ],
    }
}

/// A whole probe: one [`RunCapture`] per agent, plus the absent ones.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeReport {
    /// ISO-ish date the probe ran, so a stale report is obvious.
    pub run_on: String,
    /// One entry per agent: the first run in full, then any repeats behind it.
    pub captures: Vec<RunCapture>,
    /// Every run, when more than one was made. Empty at `repeat = 1`.
    pub all_runs: Vec<RunCapture>,
    /// What the repeats showed. Empty at `repeat = 1`, because one run is a
    /// reading and not a check.
    pub stability: Vec<Stability>,
    /// Agents named by the proposal whose binary is not on this machine, with
    /// the reason that matters. Measured by `PATH` lookup at report time, so it
    /// cannot claim an installed agent is missing.
    pub absent: Vec<AbsentAgent>,
    /// Agents that *are* installed but that this crate has no adapter for. Kept
    /// separate from [`ProbeReport::absent`] because "nothing to measure here"
    /// and "we did not look at this" are different answers.
    pub unknown: Vec<AbsentAgent>,
    /// Agents the operator has parked, and so were not run. Reported rather than
    /// dropped: a skipped agent is a decision on the record, and a run that
    /// silently covered fewer agents than it did last week reads as a
    /// regression. Naming one with `--agent` still probes it.
    pub parked: Vec<ParkedAgent>,
    /// Anything the run could not answer, stated rather than left blank.
    pub unanswered: Vec<String>,
    /// Set when the run was a fan-out: what concurrency bought, measured.
    pub fan_out: Option<FanOut>,
}

/// What a concurrent run actually cost, next to what it would have cost alone.
///
/// The honest part is the units. Six agents do not agree on what a run costs, so
/// this does not invent a single dollar figure: it reports the tokens each agent
/// reported, the one agent that metered its own units, and the wall-clock saving
/// from running them at once — which is the only figure every agent agrees on
/// and the one an orchestrator actually controls.
#[derive(Debug, Clone, Serialize)]
pub struct FanOut {
    /// How many workers ran at once.
    pub workers: usize,
    /// Wall-clock for the whole run.
    pub wall_clock_ms: u128,
    /// The sum of what each worker took on its own. Always at least
    /// `wall_clock_ms`, because they overlapped.
    pub summed_worker_ms: u128,
    /// `summed_worker_ms / wall_clock_ms`. Below 1 would mean the clock lied.
    pub overlap_factor: f64,
    /// The slowest single worker, which is the floor on any future schedule.
    pub slowest_worker_ms: u128,
    /// Slowest worker and how long it took.
    pub slowest: String,
}

impl FanOut {
    /// What a set of finished workers cost, given how long the whole run took.
    pub fn from_captures(workers: &[RunCapture], wall_clock_ms: u128) -> Self {
        let summed_worker_ms: u128 = workers.iter().map(|c| c.duration_ms).sum();
        let slowest = workers
            .iter()
            .max_by_key(|c| c.duration_ms)
            .map(|c| (c.agent.clone(), c.duration_ms))
            .unwrap_or_else(|| ("none".to_string(), 0));
        Self {
            workers: workers.len(),
            wall_clock_ms,
            summed_worker_ms,
            overlap_factor: if wall_clock_ms == 0 {
                0.0
            } else {
                summed_worker_ms as f64 / wall_clock_ms as f64
            },
            slowest_worker_ms: slowest.1,
            slowest: slowest.0,
        }
    }

    /// One line for a report.
    pub fn summary(&self) -> String {
        format!(
            "{} worker(s) in {} ms; run alone they would take {} ms ({}x), slowest {} at {} ms",
            self.workers,
            self.wall_clock_ms,
            self.summed_worker_ms,
            self.overlap_factor,
            self.slowest,
            self.slowest_worker_ms
        )
    }
}

/// An agent this probe cannot say anything about, and why.
#[derive(Debug, Clone, Serialize)]
pub struct AbsentAgent {
    pub name: String,
    pub why: String,
}

/// An agent this probe deliberately did not run, and on whose instruction.
///
/// Separate from [`AbsentAgent`] because the two claims are opposites: absent
/// means "not here to measure", parked means "here, measured or not by choice".
/// Collapsing them would let a standing decision read as a missing install.
#[derive(Debug, Clone, Serialize)]
pub struct ParkedAgent {
    pub name: String,
    pub why: String,
}

/// Cap and redact one stream.
fn finish(raw: &[u8]) -> (String, bool) {
    let text = String::from_utf8_lossy(raw);
    let truncated = text.len() > OUTPUT_CAP;
    let slice = if truncated {
        let mut end = OUTPUT_CAP;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        &text[..end]
    } else {
        &text[..]
    };
    (redact(slice), truncated)
}

/// Does this look like an event line, and if so what is it called?
///
/// Deliberately shape-based rather than schema-aware: the point of the probe is
/// to discover the shapes, and a parser that expected a schema would only
/// confirm what it already assumed.
fn event_from_line(line: &str) -> Option<ObservedEvent> {
    let trimmed = line.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let kind = ["type", "event", "kind", "event_type", "name"]
        .iter()
        .find_map(|key| value.get(*key).and_then(|v| v.as_str()))
        .unwrap_or("(untyped json object)")
        .to_string();
    let excerpt = if trimmed.len() > 240 {
        let mut end = 240;
        while end > 0 && !trimmed.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &trimmed[..end])
    } else {
        trimmed.to_string()
    };
    Some(ObservedEvent { kind, excerpt })
}

/// Find something that looks like a session id.
fn session_id_in(text: &str) -> Option<String> {
    // Key names measured across the vendors probed on 2026-10-02. Two shapes
    // that were missing here cost a real reading: kilo emits `"sessionID"` and
    // was reported as issuing no session id at all, which matters because a
    // resume is impossible without one.
    const KEYS: &[&str] = &[
        "\"sessionId\"",
        "\"session_id\"",
        "\"sessionID\"",
        "\"conversationId\"",
        "\"conversation_id\"",
        "\"threadId\"",
        "\"thread_id\"",
    ];
    for key in KEYS {
        let Some(at) = text.find(key) else { continue };
        let rest = &text[at + key.len()..];
        let rest = rest.trim_start().trim_start_matches(':').trim_start();
        let Some(body) = rest.strip_prefix('"') else {
            continue;
        };
        let end = body.find('"')?;
        let id = &body[..end];
        if !id.is_empty() {
            return Some(id.to_string());
        }
    }
    None
}

/// Does the output read as an authentication failure?
fn stopped_on_auth(text: &str) -> bool {
    // Wording measured on 2026-09-28, one per vendor, because a probe that
    // cannot recognise an auth failure reports it as a plain non-zero exit — and
    // "the agent refused to spend money" and "the agent is broken" are very
    // different facts.
    const MARKERS: &[&str] = &[
        "not logged in",
        "please run /login",
        "please log in",
        "please login",
        "login required",
        "authentication",
        "auth method",
        "unauthorized",
        "no api key",
        "api key not found",
        "no providers configured",
        "credentials",
        "sign in",
        "authenticate",
        "token expired",
        // Measured 2026-10-02 from cursor-agent: "Error: Authentication
        // required. Please run 'agent login' first, or set CURSOR_API_KEY
        // environment variable." It says "Authentication", so it matched
        // already — but only because of the lowercase `authentication`
        // marker above. Kept explicit here so the case that carried this
        // finding is not lost if that marker is ever narrowed.
        "authentication required",
    ];
    let lower = text.to_lowercase();
    MARKERS.iter().any(|m| lower.contains(m))
}

/// Find a signal that the agent wanted permission for something.
fn permission_signal_in(text: &str) -> Option<String> {
    const MARKERS: &[&str] = &[
        "approve",
        "approval",
        "permission",
        "allow",
        "confirm",
        "authorize",
    ];
    let lower = text.to_lowercase();
    MARKERS
        .iter()
        .find(|m| lower.contains(**m))
        .map(|m| (*m).to_string())
}

/// Run one command to completion, killing it at the deadline.
fn run_bounded(
    command: &mut std::process::Command,
    timeout: Duration,
) -> std::io::Result<std::process::Output> {
    use std::process::Stdio;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait()? {
            Some(_) => return child.wait_with_output(),
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "the agent did not finish in time and was killed",
                ));
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

/// The argv for a spec, with the prompt in place and the report showing a marker.
fn argv_for(spec: &AgentSpec, task: &str) -> Vec<String> {
    let mut argv: Vec<String> = spec
        .one_shot
        .split_whitespace()
        .map(|token| {
            if token == "{prompt}" {
                task.to_string()
            } else {
                token.to_string()
            }
        })
        .collect();
    // The stream flag the roster recorded is *used*, which it was not at first:
    // without it every agent printed prose and `stream_recognised` was false for
    // all six, which would have made the event half of `AR-1` vacuous.
    //
    // Split on whitespace rather than pushed whole. A flag with a value is two
    // argv entries, and passing `--format json` as one entry made opencode,
    // claude and gemini all print their usage text and exit 1 — which read as
    // three agents with no machine-readable output, and was purely this bug. The
    // roster's values were right all along; the way they were handed over was not.
    if let Some(flag) = spec.stream_flag {
        let tokens: Vec<&str> = flag.split_whitespace().collect();
        if !tokens.is_empty() && !argv.iter().any(|a| a == tokens[0]) {
            argv.extend(tokens.into_iter().map(|t| t.to_string()));
        }
    }
    argv
}

/// Run one agent, capturing everything, and never inventing a cell.
pub fn probe_one(spec: &AgentSpec, options: &ProbeOptions) -> RunCapture {
    let Some(binary) = spec.binaries.iter().find_map(|b| which(b)) else {
        return RunCapture {
            agent: spec.name.to_string(),
            binary: None,
            version: None,
            argv: Vec::new(),
            workdir: options.workdir.display().to_string(),
            exit_code: None,
            duration_ms: 0,
            stdout: String::new(),
            stderr: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            events: Vec::new(),
            stream_recognised: false,
            session_id: None,
            stopped_on_auth: false,
            permission_signal: None,
            usage: None,
            model: None,
            provenance: Provenance::NotInstalled,
            failure: Some(format!("no `{}` on PATH", spec.binaries.join("` or `"))),
        };
    };
    let binary = binary.to_string_lossy().into_owned();

    // The version is a separate, harmless invocation: it is the one cell that is
    // always observable, even with no account.
    let version = std::process::Command::new(&binary)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| redact(String::from_utf8_lossy(&o.stdout).trim()))
        .filter(|s| !s.is_empty());

    let argv = argv_for(spec, &options.task);
    let (program, rest) = argv.split_first().expect("one_shot is never empty");
    let mut command = std::process::Command::new(program);
    command.args(rest).current_dir(&options.workdir);
    // `PWD` is set as well as the real working directory, and both are needed.
    // Measured on 2026-09-28: opencode launched into a scratch fixture reported
    // the *xencode* repository as its project and globbed there, finding no
    // fixture file, while the same command with `PWD` naming the fixture found it
    // and answered. `Command::current_dir` calls chdir and leaves `PWD` alone, so
    // an agent that trusts the variable over `getcwd()` is pointed at the
    // operator's tree instead of the fixture — and the fixture is read-only, so
    // the first version of this was a probe that could look at the wrong
    // repository. Every agent gets both set, so no agent has to guess which of
    // the two the operator meant.
    command.env("PWD", &options.workdir);

    let started = std::time::Instant::now();
    let outcome = run_bounded(&mut command, options.timeout);
    let duration_ms = started.elapsed().as_millis();

    match outcome {
        Err(e) => RunCapture {
            agent: spec.name.to_string(),
            binary: Some(binary),
            version,
            argv,
            workdir: options.workdir.display().to_string(),
            exit_code: None,
            duration_ms,
            stdout: String::new(),
            stderr: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            events: Vec::new(),
            stream_recognised: false,
            session_id: None,
            stopped_on_auth: false,
            permission_signal: None,
            usage: None,
            model: None,
            provenance: Provenance::RunFailed,
            failure: Some(format!("could not run: {e}")),
        },
        Ok(output) => {
            let (stdout, stdout_truncated) = finish(&output.stdout);
            let (stderr, stderr_truncated) = finish(&output.stderr);
            let events: Vec<ObservedEvent> = stdout
                .lines()
                .chain(stderr.lines())
                .filter_map(event_from_line)
                .take(200)
                .collect();
            let combined = format!("{stdout}\n{stderr}");
            let auth = stopped_on_auth(&combined);
            let success = output.status.success();
            RunCapture {
                agent: spec.name.to_string(),
                binary: Some(binary),
                version,
                argv,
                workdir: options.workdir.display().to_string(),
                exit_code: output.status.code(),
                duration_ms,
                stdout,
                stderr,
                stdout_truncated,
                stderr_truncated,
                stream_recognised: !events.is_empty(),
                events,
                session_id: session_id_in(&combined),
                stopped_on_auth: auth,
                permission_signal: permission_signal_in(&combined),
                usage: usage_in(&combined),
                model: model_in(&combined),
                provenance: if success && !auth {
                    Provenance::Observed
                } else {
                    Provenance::RunFailed
                },
                failure: if auth {
                    Some("stopped on an authentication check".to_string())
                } else if !success {
                    Some(format!(
                        "exited {}",
                        output
                            .status
                            .code()
                            .map(|c| c.to_string())
                            .unwrap_or_else(|| "by signal".to_string())
                    ))
                } else {
                    None
                },
            }
        }
    }
}

/// Read tokens and money out of a stream, whatever shape the agent used.
///
/// Every branch below is a shape that was actually observed on 2026-10-02, with
/// the agent and event that produced it named. Nothing here infers a number the
/// agent did not print: a stream that mentions neither tokens nor cost yields
/// `None`, because a usage of zero would be indistinguishable from an agent that
/// is genuinely free — `cline` really does report `totalCost: 0`, and that is not
/// the same claim.
pub fn usage_in(text: &str) -> Option<Usage> {
    let mut usage = Usage::default();
    let mut saw_any = false;
    for line in text.lines() {
        let Some(parsed) = json_value(line) else {
            continue;
        };
        let outer = &parsed;
        // A vendor wraps the numbers at whatever depth it likes, and three
        // different depths were measured on 2026-10-02: cline's first run put
        // `usage` at the top of the line, its second run nested it under
        // `event`, and agy puts it under `result`. Each nesting level is
        // searched rather than the first one that happens to match.
        for object in usage_objects(outer) {
            let is_summary = is_run_summary(outer);
            if let Some(found) = read_token_shape(object) {
                saw_any = true;
                if is_summary {
                    // A summary restates what the per-step events already added
                    // up to, so it replaces rather than adds. Adding both is how a
                    // 13,911-token run gets reported as 41,733.
                    usage.input_tokens = found.input_tokens.or(usage.input_tokens);
                    usage.output_tokens = found.output_tokens.or(usage.output_tokens);
                    usage.reasoning_tokens = found.reasoning_tokens.or(usage.reasoning_tokens);
                    usage.cache_read_tokens = found.cache_read_tokens.or(usage.cache_read_tokens);
                    usage.cache_write_tokens =
                        found.cache_write_tokens.or(usage.cache_write_tokens);
                    usage.saw_summary = true;
                    usage.cost = read_cost(object).or(usage.cost);
                    usage.cost_unit = usage
                        .cost_unit
                        .or_else(|| Some("currency unstated".to_string()));
                } else {
                    usage.summed_over_events += 1;
                    usage.input_tokens = add(usage.input_tokens, found.input_tokens);
                    usage.output_tokens = add(usage.output_tokens, found.output_tokens);
                    usage.reasoning_tokens = add(usage.reasoning_tokens, found.reasoning_tokens);
                    usage.cache_read_tokens = add(usage.cache_read_tokens, found.cache_read_tokens);
                    usage.cache_write_tokens =
                        add(usage.cache_write_tokens, found.cache_write_tokens);
                    if let Some(cost) = read_cost(object) {
                        usage.cost = Some(usage.cost.unwrap_or(0.0) + cost);
                        usage.cost_unit = Some("currency unstated".to_string());
                    }
                }
            }
            if let Some((credits, unit)) = read_metered(object) {
                saw_any = true;
                // Taken as the largest figure seen, not their sum: whether an
                // agent's metering line restates the total or adds to it is not
                // documented by any of them, and summing a running total
                // double-counts. The larger reading is the safer of the two
                // errors, so it is the one taken.
                if usage.cost.is_none_or(|seen_credits| credits > seen_credits) {
                    usage.cost = Some(credits);
                    usage.cost_unit = unit;
                }
                usage.metered_events += 1;
            }
        }
        // cline states a price in the same units it counts tokens, on the model
        // description rather than on a usage line, so a cost of zero with a
        // token count is a real reading and not a missing one.
        if let Some(pricing) = read_pricing(outer) {
            saw_any = true;
            if usage.cost.is_none() {
                usage.cost = Some(pricing);
                usage.cost_unit = Some("listed price".to_string());
            }
        }
    }
    saw_any.then_some(usage)
}

/// The usage-shaped objects in one line, at every nesting level that was
/// measured: the line itself, and under `event`, `result`, `data` and `part`.
///
/// Descent stops as soon as an object yields counts, because an object and its
/// own `tokens` child describe the same step. Counting both is how one
/// 40,780-token step is reported as 81,560.
fn usage_objects(outer: &serde_json::Value) -> Vec<&serde_json::Value> {
    let mut found = vec![outer];
    // `usage`, `tokens` and `aggregateUsage` are included because that is where
    // the three summary shapes put their totals: cline's `run_result` holds them
    // under `usage` and repeats them in `aggregateUsage`.
    for key in [
        "event",
        "result",
        "data",
        "part",
        "usage",
        "tokens",
        "aggregateUsage",
    ] {
        let Some(nested) = outer.get(key).filter(|v| v.is_object()) else {
            continue;
        };
        found.push(nested);
        let nested_yields = read_token_shape(nested).is_some() || read_metered(nested).is_some();
        if !nested_yields {
            for key in ["usage", "tokens"] {
                if let Some(deep) = nested.get(key).filter(|v| v.is_object()) {
                    found.push(deep);
                }
            }
        }
    }
    found
}

/// Whether this line restates the whole run rather than describing one step.
///
/// cline's `run_result` and agy's `result` event both carry a total that
/// repeats what their per-step events already said.
fn is_run_summary(outer: &serde_json::Value) -> bool {
    outer.get("finishReason").is_some()
        || outer.get("aggregateUsage").is_some()
        || outer.get("event").and_then(|e| e.get("status")).is_some()
}

/// Read a token count out of whichever spelling this object uses.
///
/// Two shapes, both measured: `opencode` and its `kilo` fork put a `tokens`
/// object with a nested `cache` on each step, while `cline` and `agy` use
/// `inputTokens` / `input_tokens` beside an optional `cacheReadTokens`.
fn read_token_shape(object: &serde_json::Value) -> Option<TokenShape> {
    let source = object.get("tokens").unwrap_or(object);
    let (input_key, output_key, read_key, write_key) = if source.get("input").is_some() {
        ("input", "output", "read", "write")
    } else {
        (
            "inputTokens",
            "outputTokens",
            "cacheReadTokens",
            "cacheWriteTokens",
        )
    };
    let input = num(source, input_key).or_else(|| num(object, input_key));
    let output = num(source, output_key).or_else(|| num(object, output_key));
    let input = input
        .or_else(|| num(source, "input_tokens"))
        .or_else(|| num(object, "input_tokens"));
    let output = output
        .or_else(|| num(source, "output_tokens"))
        .or_else(|| num(object, "output_tokens"));
    if input.is_none() && output.is_none() {
        return None;
    }
    let cache = source.get("cache");
    let thinking = ["reasoning", "reasoningTokenCount", "thinking_tokens"];
    Some(TokenShape {
        input_tokens: input,
        output_tokens: output,
        reasoning_tokens: thinking
            .iter()
            .find_map(|k| num(source, k).or_else(|| num(object, k))),
        cache_read_tokens: num(cache.unwrap_or(object), read_key)
            .or_else(|| num(object, "cache_read_tokens"))
            .or_else(|| num(object, "cacheRead")),
        cache_write_tokens: num(cache.unwrap_or(object), write_key)
            .or_else(|| num(object, "cache_write_tokens"))
            .or_else(|| num(object, "cacheWrite")),
    })
}

struct TokenShape {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
    cache_read_tokens: Option<u64>,
    cache_write_tokens: Option<u64>,
}

/// A stated cost on a usage object: cline's `cost` per step and `totalCost` on
/// its summary.
fn read_cost(object: &serde_json::Value) -> Option<f64> {
    object
        .get("totalCost")
        .or_else(|| object.get("cost"))
        .and_then(|v| v.as_f64())
}

/// kiro-cli's metered credits: an array of `{value, unit}` with no tokens.
fn read_metered(object: &serde_json::Value) -> Option<(f64, Option<String>)> {
    let metering = object.get("meteringUsage")?.as_array()?;
    let mut credits = 0.0;
    let mut unit = None;
    for entry in metering {
        if let Some(value) = entry.get("value").and_then(|v| v.as_f64()) {
            credits += value;
        }
        if let Some(name) = entry.get("unit").and_then(|v| v.as_str()) {
            unit = Some(name.to_string());
        }
    }
    (credits > 0.0).then_some((credits, unit))
}

/// A price list, read as a price of zero.
///
/// cline prints the model's `pricing` block and it was all zeros for the model
/// it chose on 2026-10-02. That is how a free run becomes measurable rather than
/// merely uncounted.
fn read_pricing(outer: &serde_json::Value) -> Option<f64> {
    let pricing = outer.get("pricing")?.as_object()?;
    let input = pricing.get("input")?.as_f64()?;
    let output = pricing
        .get("output")
        .and_then(|v| v.as_f64())
        .unwrap_or(input);
    Some(input + output)
}

/// Which model answered, when the stream said so.
pub fn model_in(text: &str) -> Option<String> {
    for line in text.lines() {
        let Some(parsed) = json_value(line) else {
            continue;
        };
        let value = &parsed;
        // `kilo` puts provider and model on the same object, nested under the
        // step's `part`; other agents put it at the top of the event.
        if let Some(model) = value
            .get("model")
            .or_else(|| value.get("part").and_then(|p| p.get("model")))
        {
            if let (Some(p), Some(m)) = (
                model.get("providerID").and_then(|v| v.as_str()),
                model.get("modelID").and_then(|v| v.as_str()),
            ) {
                return Some(format!("{p}/{m}"));
            }
            // `cursor-agent` reports `"model":"Auto"`, which names a routing
            // decision rather than a model. Recording it would let a capability
            // router think it knew what answered.
            if let Some(m) = model.as_str().filter(|m| !is_unnamed_model(m)) {
                return Some(m.to_string());
            }
        }
        for key in ["model", "modelId", "model_id"] {
            if let Some(m) = value
                .get(key)
                .or_else(|| value.get("data").and_then(|d| d.get(key)))
                .and_then(|m| m.as_str())
            {
                if !is_unnamed_model(m) {
                    return Some(m.to_string());
                }
            }
        }
    }
    None
}

fn is_unnamed_model(model: &str) -> bool {
    model.trim().is_empty()
        || model.eq_ignore_ascii_case("auto")
        || model.eq_ignore_ascii_case("default")
}

fn add(current: Option<u64>, next: Option<u64>) -> Option<u64> {
    match (current, next) {
        (Some(a), Some(b)) => Some(a + b),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

fn num(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|v| v.as_u64())
}

fn json_value(line: &str) -> Option<serde_json::Value> {
    let line = line.trim();
    if !line.starts_with('{') {
        return None;
    }
    serde_json::from_str(line).ok()
}

/// Build the read-only fixture the agents are pointed at.
pub fn seed_task_dir(root: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(root)?;
    std::fs::write(root.join("notes.txt"), "xencode\n")?;
    std::fs::write(
        root.join("README.md"),
        "# Probe fixture\n\n\
         A scratch directory for the interop probe. Nothing here is part of the product, and \
         the task handed to each agent is read-only, so an agent that follows it changes \
         nothing.\n",
    )?;
    // A git repository, because a real project is one — and because an agent
    // found on 2026-09-28 refuses to run at all without it: `codex exec` answered
    // "Not inside a trusted directory and --skip-git-repo-check was not specified"
    // and exited in 100 ms. Making the fixture a repository is more faithful than
    // passing a per-vendor escape hatch, and it exercises the same directory
    // shape the agents will meet in real use.
    for args in [
        vec!["init", "-q"],
        vec!["config", "user.email", "probe@xencode.invalid"],
        vec!["config", "user.name", "xencode probe"],
    ] {
        let _ = std::process::Command::new("git")
            .current_dir(root)
            .args(&args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    let _ = std::process::Command::new("git")
        .current_dir(root)
        .args(["add", "."])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let _ = std::process::Command::new("git")
        .current_dir(root)
        .args(["commit", "-qm", "probe fixture"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    Ok(())
}

/// Which agents a run will cover.
///
/// A bare run covers every agent the operator has not parked. Naming an agent
/// with `options.only` overrides that: asking for a parked agent by name is the
/// operator changing their mind, so it is probed.
pub fn selected_agents(options: &ProbeOptions) -> Vec<&'static AgentSpec> {
    if options.only.is_empty() {
        ROSTER.iter().filter(|a| !a.parked).collect()
    } else {
        ROSTER
            .iter()
            .filter(|a| options.only.iter().any(|n| n == a.name))
            .collect()
    }
}

/// Run the probe over [`selected_agents`].
pub fn run_probe(options: &ProbeOptions) -> ProbeReport {
    let selected = selected_agents(options);
    let repeat = options.repeat.max(1);
    let mut all_runs: Vec<RunCapture> = Vec::new();
    let mut captures: Vec<RunCapture> = Vec::new();
    let fan_out = if options.fan_out && selected.len() > 1 {
        // One worker per thread, all launched together, because the question
        // this answers is what concurrency *buys* — so measuring it by running
        // them one after another would answer a different question.
        //
        // The fixture is seeded once and shared, and it is read-only: five agents
        // asked to read one file cannot conflict over it. Each worker still gets
        // its own `PWD` and its own process.
        let started = std::time::Instant::now();
        let mut handles = Vec::with_capacity(selected.len());
        for spec in &selected {
            let spec: &'static AgentSpec = spec;
            let options = options.clone();
            handles.push(std::thread::spawn(move || probe_one(spec, &options)));
        }
        let mut fan_captures = Vec::with_capacity(selected.len());
        for handle in handles {
            match handle.join() {
                Ok(capture) => fan_captures.push(capture),
                Err(_) => {
                    // A worker thread that panicked has already said so on the
                    // terminal. Losing its capture silently would turn a crash
                    // into a missing row, which is the failure this crate exists
                    // to avoid.
                    unreachable!("a probe worker panicked; see the panic above")
                }
            }
        }
        let fan = FanOut::from_captures(&fan_captures, started.elapsed().as_millis());
        all_runs.extend(fan_captures.iter().cloned());
        captures.extend(fan_captures);
        Some(fan)
    } else {
        for spec in &selected {
            let mut runs: Vec<RunCapture> = Vec::with_capacity(repeat as usize);
            for _ in 0..repeat {
                runs.push(probe_one(spec, options));
            }
            // The first run is the one the table shows; the rest exist to be
            // compared against it, and are all kept so a report can be re-read.
            all_runs.extend(runs.iter().cloned());
            captures.push(runs[0].clone());
        }
        None
    };
    let stability: Vec<Stability> = if repeat > 1 {
        selected
            .iter()
            .map(|spec| {
                let runs: Vec<RunCapture> = all_runs
                    .iter()
                    .filter(|c| c.agent == spec.name)
                    .cloned()
                    .collect();
                check_stability(spec.name, &runs)
            })
            .collect()
    } else {
        Vec::new()
    };

    // What the run could not answer, said out loud rather than left blank.
    let mut unanswered = Vec::new();
    for capture in &captures {
        match capture.provenance {
            Provenance::NotInstalled => unanswered.push(format!(
                "{}: not installed, so every cell stays unknown",
                capture.agent
            )),
            Provenance::RunFailed => unanswered.push(format!(
                "{}: {}",
                capture.agent,
                capture
                    .failure
                    .clone()
                    .unwrap_or_else(|| "run failed".into())
            )),
            Provenance::Observed => {
                if !capture.stream_recognised {
                    unanswered.push(format!(
                        "{}: ran, but emitted no machine-readable stream, so no event kinds \
                         are known",
                        capture.agent
                    ));
                }
            }
            Provenance::ReadFromHelp => {}
        }
    }
    if options.repeat.max(1) <= 1 {
        unanswered.push(
            "Only one run per agent: no cell has been compared against a second run for \
             stability."
                .to_string(),
        );
    }

    for verdict in &stability {
        for check in &verdict.checks {
            if !check.stable {
                unanswered.push(format!(
                    "{}: {} VARIED across {} run(s) — {}",
                    verdict.agent,
                    check.fact,
                    verdict.runs,
                    check.values.join(" vs ")
                ));
            }
        }
    }

    ProbeReport {
        run_on: chrono_today(),
        captures,
        all_runs,
        stability,
        absent: crate::roster::absent_agents()
            .into_iter()
            .map(|(name, why)| AbsentAgent { name, why })
            .collect(),
        unknown: crate::roster::installed_but_unknown()
            .into_iter()
            .map(|(name, why)| AbsentAgent { name, why })
            .collect(),
        parked: crate::roster::parked_agents()
            .into_iter()
            .map(|(name, why)| ParkedAgent { name, why })
            .collect(),
        unanswered,
        fan_out,
    }
}

/// Today's date, without pulling in a date library for one field.
fn chrono_today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days since the epoch, converted with the civil-from-days algorithm. Only
    // ever used to stamp a report, so a leap-year edge here is cosmetic.
    let days = (secs / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Read-only check for whether an agent looks usable on this machine.
///
/// Answers "would a run here spend anything?", never "log in". No agent's
/// credentials are read, printed or guessed at: the check is for the *config
/// files and directories an agent is known to keep*, and it says plainly when it
/// finds nothing, because "no config directory" and "configured but not logged in"
/// are different answers and only the first is visible from out here.
pub fn credential_status(spec: &AgentSpec) -> CredentialStatus {
    // A missing HOME is not a reason to fail: it just means nothing is found, and
    // that is reported as "not configured" rather than as an error.
    let home = std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| home.join(".local/share"));
    let (paths, hint): (Vec<std::path::PathBuf>, &str) = match spec.name {
        "opencode" => (
            vec![data.join("opencode"), home.join(".config/opencode")],
            "no sign-in step needed if `opencode providers` lists one",
        ),
        "cline" => (vec![home.join(".cline")], "run `cline auth` in a terminal"),
        "codex" => (vec![home.join(".codex")], "run `codex login` in a terminal"),
        "claude" => (
            vec![
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                    .join(".claude"),
            ],
            "run `claude` once and use `/login`, or `claude setup-token`",
        ),
        "gemini" => (
            vec![
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                    .join(".gemini"),
            ],
            "run `gemini` once and choose an auth method, or set GEMINI_API_KEY",
        ),
        "crush" => (
            vec![home.join(".local/share/crush")],
            "run `crush login` in a terminal",
        ),
        _ => (Vec::new(), "no known check for this agent"),
    };
    let present: Vec<String> = paths
        .iter()
        .filter(|p| p.exists())
        .map(|p| p.display().to_string())
        .collect();
    CredentialStatus {
        agent: spec.name.to_string(),
        looks_configured: !present.is_empty(),
        found: present,
        note: hint.to_string(),
    }
}

/// What a read-only look could and could not see.
#[derive(Debug, Clone, Serialize)]
pub struct CredentialStatus {
    pub agent: String,
    /// Whether a config directory exists. **Not** a claim that a session works.
    pub looks_configured: bool,
    pub found: Vec<String>,
    /// What the operator would run to fix it, if the answer was no.
    pub note: String,
}

impl CredentialStatus {
    pub fn summary(&self) -> String {
        if self.looks_configured {
            format!(
                "{:<9} config present ({}) — whether a run spends is not knowable from out here",
                self.agent,
                self.found
                    .iter()
                    .map(|p| {
                        let home = std::env::var("HOME").unwrap_or_default();
                        p.strip_prefix(&format!("{home}/"))
                            .map(|r| r.to_string())
                            .unwrap_or_else(|| p.clone())
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            format!("{:<9} no config found — {}", self.agent, self.note)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_usage_shapes_measured_on_2026_10_02_all_read_as_one() {
        // Every string below is a line copied out of that agent's own stream, at
        // the nesting depth it actually used. The nesting is the point: cline
        // reported `usage` at the top of the line in one run and nested under
        // `event` in the next, and agy nests under `result`. A reader that only
        // looked at the top level called both of them free.
        let opencode = r#"{"type":"step_finish","part":{"type":"step-finish","tokens":{"total":42761,"input":40780,"output":39,"reasoning":0,"cache":{"write":0,"read":42722}}}}"#;
        let cline_step = r#"{"type":"agent_event","event":{"type":"usage","inputTokens":6901,"outputTokens":83,"cacheReadTokens":241,"cacheWriteTokens":0,"cost":0,"reasoningTokenCount":136,"totalInputTokens":6901}}"#;
        let cline_nested_step = r#"{"type":"agent_event","event":{"type":"usage","inputTokens":7010,"outputTokens":32,"cacheReadTokens":241,"cacheWriteTokens":0,"cost":0,"reasoningTokenCount":327,"totalInputTokens":13911}}"#;
        let cline_summary = r#"{"type":"run_result","finishReason":"completed","iterations":2,"usage":{"inputTokens":13911,"outputTokens":115,"cacheReadTokens":482,"cacheWriteTokens":0,"totalCost":0},"aggregateUsage":{"inputTokens":13911}}"#;
        let agy = r#"{"event":"result","result":{"conversation_id":"8719","status":"SUCCESS","usage":{"input_tokens":28174,"output_tokens":160,"thinking_tokens":109,"cache_read_tokens":0,"total_tokens":28334}}}"#;
        let kiro = r#"{"type":"metadata","data":{"meteringUsage":[{"value":0.04186594859038143,"unit":"credit","unitPlural":"credits"}]}}"#;
        let cline_pricing = r#"{"maxTokens":943718,"pricing":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0},"family":"muse"}"#;

        let o = usage_in(opencode).expect("opencode reported tokens");
        assert_eq!(o.input_tokens, Some(40780));
        assert_eq!(o.cache_read_tokens, Some(42722));
        assert_eq!(o.cost, None, "opencode never said what it cost");

        // Two per-step events plus a summary that restates them. The summary must
        // win, not add: 6,901 + 7,010 + 13,911 is the bug this guards.
        let c = usage_in(&format!(
            "{cline_step}\n{cline_nested_step}\n{cline_summary}"
        ))
        .expect("cline reported usage");
        assert_eq!(
            c.input_tokens,
            Some(13911),
            "cline's own total, counted once"
        );
        assert_eq!(c.output_tokens, Some(115));
        assert_eq!(c.cache_read_tokens, Some(482));
        // 136 + 327: cline's per-step figures are increments, not running
        // totals, so these add. Its input figures add for the same reason, and
        // the run-level summary then replaces the sum rather than joining it.
        assert_eq!(c.reasoning_tokens, Some(463));
        assert_eq!(c.cost, Some(0.0), "cline really does charge zero here");
        assert!(c.saw_summary, "cline's run_result restated the totals");
        assert!(
            !c.summary().contains("no cost"),
            "a reported zero is not a missing cost"
        );

        let a = usage_in(agy).expect("agy reported usage under result");
        assert_eq!(a.input_tokens, Some(28174));
        assert_eq!(
            a.reasoning_tokens,
            Some(109),
            "agy calls reasoning thinking_tokens"
        );

        let k = usage_in(kiro).expect("kiro metered credits");
        assert_eq!(
            k.input_tokens, None,
            "kiro-cli reports no token count at all"
        );
        assert_eq!(
            k.cost,
            Some(0.041_865_948_590_381_43),
            "kiro's credits must not be rounded away"
        );
        assert_eq!(k.cost_unit.as_deref(), Some("credit"));

        // A free model with a printed price of zero is a measured cost, not a
        // missing one, and it must not read as "no cost reported".
        let free = usage_in(cline_pricing).expect("cline printed a price list");
        assert_eq!(free.cost, Some(0.0));
        assert!(!free.summary().contains("no cost"));
    }

    #[test]
    fn a_metered_charge_is_taken_as_the_largest_figure_not_their_sum() {
        // Two metering lines, as kiro-cli emits. Whether the second restates the
        // total or adds to it is documented by nobody, so the larger figure is
        // used: over-reporting a charge is recoverable, under-reporting one is
        // not.
        let stream = concat!(
            r#"{"type":"metadata","data":{"meteringUsage":[{"value":0.0418,"unit":"credit"}]}}"#,
            "\n",
            r#"{"type":"metadata","data":{"meteringUsage":[{"value":0.0232,"unit":"credit"}]}}"#,
        );
        let u = usage_in(stream).expect("kiro metered twice");
        assert_eq!(u.cost, Some(0.0418), "the larger reading, not the sum");
        assert_eq!(u.metered_events, 2);
    }

    #[test]
    fn a_fan_out_reports_the_overlap_it_bought_and_names_the_slowest_worker() {
        // Five workers measured on 2026-10-02 took 13.6, 11.6, 25.2, 25.9 and
        // 9.0 seconds. Run one after another that is 85.3 seconds; run together
        // it was 29.4. The number that decides a schedule is the slowest worker,
        // because no amount of concurrency beats it.
        let worker = |agent: &str, ms: u128| RunCapture {
            agent: agent.to_string(),
            binary: Some("/usr/bin/true".into()),
            version: None,
            argv: vec![],
            workdir: "/tmp".into(),
            exit_code: Some(0),
            duration_ms: ms,
            stdout: String::new(),
            stderr: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            events: Vec::new(),
            stream_recognised: true,
            session_id: None,
            stopped_on_auth: false,
            permission_signal: None,
            usage: None,
            model: None,
            provenance: Provenance::Observed,
            failure: None,
        };
        let workers = vec![
            worker("opencode", 13_553),
            worker("cline", 11_596),
            worker("agy", 25_214),
            worker("kilo", 25_919),
            worker("kiro-cli", 9_014),
        ];
        let fan = FanOut::from_captures(&workers, 29_415);
        assert_eq!(fan.workers, 5);
        assert_eq!(fan.summed_worker_ms, 85_296);
        assert_eq!(fan.slowest, "kilo");
        assert_eq!(fan.slowest_worker_ms, 25_919);
        assert!(
            fan.overlap_factor > 2.8 && fan.overlap_factor < 3.0,
            "overlap should be about 2.9x, got {}",
            fan.overlap_factor
        );
        // The saving is real only if it stays under the sequential total.
        assert!(fan.wall_clock_ms < fan.summed_worker_ms);
        assert!(fan.summary().contains("kilo"), "{}", fan.summary());
    }

    #[test]
    fn a_fan_out_of_one_worker_reports_no_speedup_rather_than_a_lie() {
        let worker = RunCapture {
            agent: "solo".into(),
            binary: Some("/usr/bin/true".into()),
            version: None,
            argv: vec![],
            workdir: "/tmp".into(),
            exit_code: Some(0),
            duration_ms: 5_000,
            stdout: String::new(),
            stderr: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            events: Vec::new(),
            stream_recognised: true,
            session_id: None,
            stopped_on_auth: false,
            permission_signal: None,
            usage: None,
            model: None,
            provenance: Provenance::Observed,
            failure: None,
        };
        let fan = FanOut::from_captures(&[worker], 5_000);
        assert_eq!(fan.overlap_factor, 1.0, "one worker cannot overlap itself");
    }

    #[test]
    fn a_stream_that_never_mentions_usage_reports_none_rather_than_zero() {
        // The distinction matters: `cline` really does report a cost of zero,
        // and "this agent charges nothing" is a different claim from "this
        // reader found no cost line". Collapsing them would let an orchestrator
        // treat an unreported cost as a free run.
        let quiet = "{\"type\":\"step_start\",\"sessionID\":\"ses_1\"}\nplain text, no usage";
        assert_eq!(usage_in(quiet), None);
        assert!(!usage_in(quiet).is_some_and(|u| u.cost == Some(0.0)));
    }

    #[test]
    fn a_model_is_read_when_the_stream_names_one_and_left_unanswered_when_not() {
        assert_eq!(
            model_in(r#"{"type":"step_finish","part":{"model":{"providerID":"kilo","modelID":"stealth/space-bunny-alpha"}}}"#)
                .as_deref(),
            Some("kilo/stealth/space-bunny-alpha")
        );
        assert_eq!(
            model_in(r#"{"type":"system","subtype":"init","model":"Auto"}"#),
            None,
            "`Auto` is not a model name, so it must not be recorded as one"
        );
        assert_eq!(model_in(r#"{"type":"text","text":"hello"}"#), None);
    }

    #[test]
    fn a_parked_agent_is_reported_as_skipped_rather_than_as_a_failure() {
        // The point of parking: a bare run must not carry an agent nobody is
        // going to authenticate. It must also not silently vanish — a report that
        // quietly covered fewer agents reads as a regression.
        let bare = selected_agents(&ProbeOptions::conservative("."));
        let parked = crate::roster::parked_agents();
        assert!(
            !parked.is_empty(),
            "nothing is parked, so nothing is tested"
        );
        for (name, _) in &parked {
            assert!(
                !bare.iter().any(|s| s.name == *name),
                "{name} is parked but a bare run still selected it"
            );
            // Naming it is the operator changing their mind, so it runs.
            let named = selected_agents(&ProbeOptions {
                only: vec![name.clone()],
                ..ProbeOptions::conservative(".")
            });
            assert!(
                named.iter().any(|s| s.name == *name),
                "{name} was named explicitly and must still be probed"
            );
            assert_eq!(named.len(), 1, "naming {name} selected something else too");
        }
    }

    #[test]
    fn a_parked_agent_keeps_its_roster_row_and_says_why_it_was_parked() {
        for (name, why) in crate::roster::parked_agents() {
            let spec = crate::roster::find(&name).expect("a parked agent is still known");
            assert!(
                spec.parked,
                "{name} is listed as parked but its row disagrees"
            );
            assert!(!why.is_empty(), "{name} is parked with no reason recorded");
            assert!(
                spec.one_shot.contains("{prompt}"),
                "{name} is parked but its row stopped describing how to run it"
            );
        }
    }

    #[test]
    fn a_redaction_pass_removes_the_shapes_agents_actually_echo() {
        let text = "\
key sk-FAKE-NOT-A-REAL-TEST-KEY\n\
ghp_FAKEFAKEFAKEFAKE12345\n\
AIzaNOTREALKEYNOTREALKEYNOTREALKEY\n\
Authorization: Bearer FAKEJwtNotARealToken.payload.sig\n";
        let out = redact(text);
        assert!(!out.contains("sk-FAKE-NOT-A-REAL"), "{out}");
        assert!(!out.contains("ghp_FAKEFAKE"), "{out}");
        assert!(!out.contains("AIzaNOTREALKEY"), "{out}");
        assert!(!out.contains("FAKEJwtNotARealToken"), "{out}");
        assert_eq!(out.matches("[redacted").count(), 4, "{out}");
    }

    #[test]
    fn redaction_leaves_ordinary_output_alone() {
        let text = "reading notes.txt\nthe answer is xencode\n";
        assert_eq!(redact(text), text);
    }

    #[test]
    fn an_event_is_named_from_whichever_key_the_vendor_used() {
        // Four different spellings, all of which appear in the wild.
        for (line, expected) in [
            (r#"{"type":"assistant","text":"hi"}"#, "assistant"),
            (r#"{"event":"tool_use","name":"read"}"#, "tool_use"),
            (r#"{"kind":"item.completed"}"#, "item.completed"),
            (r#"{"event_type":"message"}"#, "message"),
        ] {
            let found = event_from_line(line).expect("should recognise the shape");
            assert_eq!(found.kind, expected);
        }
        // And something that is not an event at all is not invented into one.
        assert!(event_from_line("just some text").is_none());
        assert!(event_from_line("{not json").is_none());
    }

    #[test]
    fn an_untyped_json_line_is_still_an_event_and_says_so() {
        let found = event_from_line(r#"{"foo":1}"#).expect("a json object is an event");
        assert_eq!(found.kind, "(untyped json object)");
    }

    #[test]
    fn a_session_id_is_found_under_any_of_the_names_vendors_use() {
        for (line, expected) in [
            (r#"{"sessionId":"abc-123"}"#, Some("abc-123")),
            (r#"{"session_id":"def-456"}"#, Some("def-456")),
            (r#"{"threadId":"ghi"}"#, Some("ghi")),
            (r#"{"conversation_id":"jkl"}"#, Some("jkl")),
            // kilo's own casing, measured 2026-10-02. Missing from this list,
            // kilo was reported as issuing no session id and therefore not
            // resumable, which was false.
            (
                r#"{"type":"step_start","sessionID":"ses_f0508be90f"}"#,
                Some("ses_f0508be90f"),
            ),
            (r#"{"type":"assistant"}"#, None),
        ] {
            assert_eq!(session_id_in(line).as_deref(), expected, "{line}");
        }
    }

    #[test]
    fn an_auth_failure_is_distinguished_from_a_real_answer() {
        // The three exact sentences this probe actually saw on 2026-09-28. Each
        // was missed by an earlier version of this matcher, which reported all
        // three as plain non-zero exits.
        assert!(stopped_on_auth("Not logged in \u{b7} Please run /login"));
        assert!(stopped_on_auth(
            "Please set an Auth method in your /home/you/.gemini/settings.json or specify \
             one of the following environment variables"
        ));
        assert!(stopped_on_auth(
            "ERROR  No providers configured - please run 'crush' to set up a provider"
        ));
        assert!(stopped_on_auth("Error: not logged in. Run `claude login`."));
        assert!(stopped_on_auth("401 Unauthorized"));
        assert!(stopped_on_auth("No API key found"));
        // cursor-agent's own words, measured 2026-10-02.
        assert!(stopped_on_auth(
            "Error: Authentication required. Please run 'agent login' first, or set CURSOR_API_KEY environment variable."
        ));
        // A normal answer that merely mentions the word must not be misread.
        assert!(!stopped_on_auth("The file contains the word xencode."));
    }

    #[test]
    fn the_prompt_is_replaced_by_a_marker_in_the_reported_argv() {
        let spec = crate::roster::find("crush").unwrap(); // no stream flag
        let argv = argv_for(spec, "the task text");
        assert_eq!(argv, ["crush", "run", "the task text"]);
    }

    #[test]
    fn a_flag_with_a_value_becomes_two_argv_entries() {
        // Passing `--format json` as one entry made three agents print usage and
        // exit 1. The roster stores the flag as one readable string; the process
        // needs it split.
        for (agent, expected) in [
            ("opencode", vec!["--format", "json"]),
            ("claude", vec!["--output-format", "stream-json"]),
        ] {
            let spec = crate::roster::find(agent).unwrap();
            let argv = argv_for(spec, "TASK");
            // Appended after the prompt, which every one of these CLIs accepts.
            let tail: Vec<&str> = argv[argv.len() - expected.len()..]
                .iter()
                .map(String::as_str)
                .collect();
            assert_eq!(tail, expected, "{agent}: {argv:?}");
            assert_eq!(argv.first().map(String::as_str), Some(spec.binaries[0]));
            assert!(argv.contains(&"TASK".to_string()), "{agent}: {argv:?}");
        }
        // A single-token flag stays one token.
        let codex = crate::roster::find("codex").unwrap();
        assert!(argv_for(codex, "TASK").contains(&"--json".to_string()));
    }

    #[test]
    fn a_missing_binary_is_a_recorded_fact_not_a_crash() {
        let fake = AgentSpec {
            name: "not-a-real-agent",
            binaries: &["definitely-not-installed-xyz"],
            one_shot: "definitely-not-installed-xyz run {prompt}",
            stream_flag: None,
            advertises_daemon: false,
            advertises_acp: false,
            advertises_mcp: false,
            advertises_resume: false,
            advertises_approval: false,
            attach: None,
            session_list: None,
            parked: false,
            park_reason: None,
            read_on: "2026-09-28",
        };
        let options = ProbeOptions::conservative(std::env::temp_dir());
        let capture = probe_one(&fake, &options);
        assert_eq!(capture.provenance, Provenance::NotInstalled);
        assert!(capture.failure.is_some());
        assert!(capture.summary().contains("not installed"));
    }

    #[test]
    fn output_is_capped_and_the_capping_is_reported() {
        let big = "x".repeat(OUTPUT_CAP * 2);
        let (text, truncated) = finish(big.as_bytes());
        assert!(truncated, "a 128 KiB capture must say it was capped");
        assert!(text.len() <= OUTPUT_CAP);
        let small = "short".repeat(10);
        let (_, truncated) = finish(small.as_bytes());
        assert!(!truncated);
    }

    #[test]
    fn a_session_ids_value_is_not_compared_because_it_is_meant_to_differ() {
        // A fresh id each run is correct behaviour, so checking the value would
        // report "VARIED" for an agent doing exactly the right thing.
        let capture = |id: Option<&str>| RunCapture {
            agent: "x".to_string(),
            binary: Some("b".to_string()),
            version: None,
            argv: vec![],
            workdir: "/tmp".to_string(),
            exit_code: Some(0),
            duration_ms: 1,
            stdout: String::new(),
            stderr: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            events: vec![],
            stream_recognised: false,
            session_id: id.map(|s| s.to_string()),
            stopped_on_auth: false,
            permission_signal: None,
            usage: None,
            model: None,
            provenance: Provenance::Observed,
            failure: None,
        };
        let differing_ids = check_stability("x", &[capture(Some("aaa")), capture(Some("bbb"))]);
        let id_check = differing_ids
            .checks
            .iter()
            .find(|c| c.fact == "session id")
            .expect("the session-id check exists");
        assert!(id_check.stable, "two fresh ids is not a difference in fact");
        assert_eq!(id_check.values, ["present"]);

        // Presence *is* a fact, and its absence is a real difference.
        let appearing = check_stability("x", &[capture(None), capture(Some("aaa"))]);
        let id_check = appearing
            .checks
            .iter()
            .find(|c| c.fact == "session id")
            .unwrap();
        assert!(!id_check.stable, "absent then present is a real difference");
    }

    #[test]
    fn a_varying_fact_is_reported_with_both_values_not_averaged() {
        let capture = |n: usize| RunCapture {
            agent: "x".to_string(),
            binary: Some("b".to_string()),
            version: None,
            argv: vec![],
            workdir: "/tmp".to_string(),
            exit_code: Some(0),
            duration_ms: 1,
            stdout: String::new(),
            stderr: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            events: (0..n)
                .map(|i| ObservedEvent {
                    kind: format!("e{i}"),
                    excerpt: String::new(),
                })
                .collect(),
            stream_recognised: n > 0,
            session_id: None,
            stopped_on_auth: false,
            permission_signal: None,
            usage: None,
            model: None,
            provenance: Provenance::Observed,
            failure: None,
        };
        let verdict = check_stability("x", &[capture(18), capture(20)]);
        let vocabulary = verdict
            .checks
            .iter()
            .find(|c| c.fact == "event vocabulary")
            .unwrap();
        assert!(!vocabulary.stable, "18 kinds then 20 kinds is a difference");
        // Both values kept, so the report shows the two and not a mean of 19.
        assert_eq!(vocabulary.values.len(), 2);
        assert!(
            verdict.summary().contains("VARIED"),
            "{}",
            verdict.summary()
        );
        let count = verdict
            .checks
            .iter()
            .find(|c| c.fact == "event count")
            .unwrap();
        assert!(!count.stable);
        assert!(
            verdict.summary().contains("|"),
            "both values shown: {}",
            verdict.summary()
        );
    }

    #[test]
    fn the_task_fixture_asks_a_read_only_question_and_contains_its_answer() {
        let dir = std::env::temp_dir().join(format!("xencode-probe-{}", std::process::id()));
        seed_task_dir(&dir).unwrap();
        let notes = std::fs::read_to_string(dir.join("notes.txt")).unwrap();
        assert!(default_task().contains("Do not create, edit or delete"));
        assert!(default_task().contains("notes.txt"));
        assert!(
            notes.contains("xencode"),
            "the fixture must hold the answer"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
