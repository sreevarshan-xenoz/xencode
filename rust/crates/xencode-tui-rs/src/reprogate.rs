//! The red-to-green reproduction gate (U-6).
//!
//! A fix that was never preceded by a failing test is a coincidence with a
//! diff attached to it. "The test passes now" is consistent with three
//! different worlds — the fix worked, the test always passed, the test passes
//! for an unrelated reason — and only a measurement taken *before* the change
//! tells them apart. So this module turns one bug fix into a two-point
//! measurement on the same command:
//!
//! ```text
//! same command · unmodified tree · FAILED
//! same command · after the fix   · PASSED
//! ```
//!
//! Enforcement is by capability, not by prompt, because a tool set that cannot
//! express the violation beats an instruction asking for restraint. While the
//! gate is waiting for its failure, the agent may read anything, run commands,
//! and write exactly one file: the reproduction test. Every other write is
//! refused here, at the same choke point that decides whether a call runs at
//! all, and the tools that only ever edit production source are not even
//! offered for the model to waste a round on.
//!
//! The trap this has to survive is a reproduction that cannot fail: a trivially
//! true assertion, a swallowed panic, or a test of code the bug never touched.
//! A run whose exit says failure but whose text says nothing about *which*
//! assertion failed is refused, and the failure's file and line are recorded as
//! the evidence — so a fix can be checked against the assertion it was supposed
//! to satisfy, and a failure landing outside the reported bug's neighbourhood is
//! flagged rather than accepted.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The tool that runs a reproduction and judges it.
pub const REPRO_TOOL: &str = "reproduce_bug";

/// A reproduction is `cargo test` on a cold workspace, which the general
/// 30-second command budget routinely loses to. The gate holds its own floor so
/// turning a fast command's timeout down cannot strangle a test run; a larger
/// configured value still wins.
pub const REPRO_MIN_TIMEOUT_SECS: u64 = 180;

/// Cap on the captured output used for judging. A compile error can bury the
/// assertion in noise; the panic sites are kept from whatever we do capture.
const REPRO_OUTPUT_CAP: usize = 512 * 1024;

/// Where a bug fix has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phase {
    /// No fix is in progress: the gate is not engaged and forbids nothing.
    #[default]
    Off,
    /// A fix is in progress and its failure has not been seen yet. Production
    /// writes are refused. This is the state that makes "I fixed it" mean
    /// something later.
    AwaitingRed,
    /// The failure has been witnessed. Production edits are unlocked, and the
    /// reproduction file itself is frozen — the assertion is what it asserted
    /// when it failed, and rewriting it is how a manufactured pass is made.
    RedWitnessed,
    /// Both points of the measurement exist. The fix is evidenced.
    Verified,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::AwaitingRed => "awaiting a failing reproduction",
            Self::RedWitnessed => "red witnessed, awaiting the green",
            Self::Verified => "red then green recorded",
        }
    }
}

/// One observed test failure, as the evidence records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    /// The file the assertion lives in, workspace-relative when we can say so.
    pub location: String,
    pub line: Option<u32>,
    /// The panic or assertion text, first lines only.
    pub message: String,
    /// The test's own name, when the runner printed one.
    pub test: Option<String>,
}

impl Failure {
    /// How the evidence quotes it: `src/foo.rs:12 — assertion failed: x`.
    pub fn short(&self) -> String {
        let line = match self.line {
            Some(n) => format!("{n}"),
            None => "?".to_string(),
        };
        let first = self.message.lines().next().unwrap_or("").trim().to_string();
        format!("{}:{line} — {first}", self.location)
    }
}

/// What the gate holds about the bug fix it is supervising.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    /// The single file this fix was allowed to write while awaiting its red.
    pub repro: String,
    /// The command witnessed failing, and re-run to make the green.
    pub command: String,
    /// The reported bug's neighbourhood, as path prefixes.
    pub scope: Vec<String>,
    pub red: Option<Failure>,
    pub red_exit: Option<i32>,
    pub red_artifact: Option<String>,
    pub green_exit: Option<i32>,
    pub green_artifact: Option<String>,
    /// Exit code of the surrounding suite, if one was run and recorded.
    pub suite_exit: Option<i32>,
    pub suite_command: Option<String>,
}

/// A persistent record of a completed red-to-green reproduction measurement (AE-3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproRecord {
    pub session: Option<String>,
    pub repro: String,
    pub command: String,
    pub scope: Vec<String>,
    pub red_exit: Option<i32>,
    pub red_artifact: Option<String>,
    pub red_failure: Option<Failure>,
    pub green_exit: Option<i32>,
    pub green_artifact: Option<String>,
    pub suite_exit: Option<i32>,
    pub suite_command: Option<String>,
    pub ts_unix_ms: u64,
}

pub fn repro_history_path(xencode_dir: &Path) -> std::path::PathBuf {
    xencode_dir.join("cache").join("repro.jsonl")
}

pub fn append_repro_record(xencode_dir: &Path, record: &ReproRecord) -> std::io::Result<()> {
    let path = repro_history_path(xencode_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut rows = xencode_core_rs::read_jsonl_tolerant::<ReproRecord>(&path).rows;
    rows.push(record.clone());
    let mut text = String::new();
    for row in &rows {
        let mut json = serde_json::to_string(row)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        json = xencode_context_rs::trace::redact_secrets(&json);
        text.push_str(&json);
        text.push('\n');
    }
    xencode_core_rs::write_atomic(&path, text.as_bytes())
}

pub fn read_repro_history(root: &Path, xencode_dir: &Path) -> Vec<ReproRecord> {
    let path = repro_history_path(xencode_dir);
    let rows = xencode_core_rs::read_jsonl_tolerant::<ReproRecord>(&path).rows;
    rows.into_iter()
        .filter(|row| {
            let red_exists = match &row.red_artifact {
                Some(p) => {
                    root.join(p).is_file()
                        || xencode_dir.join(p).is_file()
                        || Path::new(p).is_file()
                }
                None => false,
            };
            let green_exists = match &row.green_artifact {
                Some(p) => {
                    root.join(p).is_file()
                        || xencode_dir.join(p).is_file()
                        || Path::new(p).is_file()
                }
                None => false,
            };
            red_exists && green_exists
        })
        .collect()
}

pub fn display_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(|r| r.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.display().to_string())
}

/// The outcome of judging one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// A failure with a recorded assertion, inside the reported neighbourhood.
    Red { failure: Failure },
    /// A failure whose recorded location sits outside the reported
    /// neighbourhood. Not accepted: this is where a test of unrelated code, or a
    /// deliberately broken one, shows up.
    Suspect {
        failure: Failure,
        scope: Vec<String>,
    },
    /// The run passed, so it reproduces nothing.
    Passed,
    /// The run failed but no failing assertion can be pointed at.
    NoFailure,
    /// The run hit its wall clock.
    TimedOut,
    /// The run never started, with the reason.
    DidNotRun(String),
}

impl Verdict {
    pub fn describe(&self) -> String {
        match self {
            Self::Red { failure } => format!("red witnessed: {}", failure.short()),
            Self::Suspect { failure, scope } => format!(
                "suspect: {} is outside the reported bug's neighbourhood [{}]",
                failure.short(),
                scope.join(", ")
            ),
            Self::Passed => "the reproduction passed, so it does not reproduce the bug".to_string(),
            Self::NoFailure => {
                "the run failed without a failing assertion this gate can record".to_string()
            }
            Self::TimedOut => "the reproduction run timed out".to_string(),
            Self::DidNotRun(why) => format!("the reproduction never ran: {why}"),
        }
    }
}

pub fn regex() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r#"panicked at ([^\r\n'"]+?):(\d+):(\d+):"#).expect("panic-site regex")
    })
}

pub fn failed_test_regex() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(
            r"(?m)^\s*(?:test\s+)?([^\s]+)\s+\.\.\.\s+FAILED|^\s*FAIL\s*\[[^\]]*\]:\s*([^\s]+)",
        )
        .expect("failed-test regex")
    })
}

/// Every failing assertion the runner printed, in the order it printed them.
/// Rust writes `panicked at <file>:<line>:<col>:` and puts the message on the
/// following lines; `cargo test`, `nextest` and a bare binary all use that shape.
pub fn panic_sites(output: &str) -> Vec<(String, u32, u32, String)> {
    let mut sites = Vec::new();
    for capture in regex().captures_iter(output) {
        let (Some(file), Some(line), Some(col)) = (capture.get(1), capture.get(2), capture.get(3))
        else {
            continue;
        };
        // Windows prints `tests\repro.rs`; every comparison here is made in
        // forward slashes, so the file is stored that way.
        let file = file.as_str().trim().replace('\\', "/");
        if file.is_empty() {
            continue;
        }
        let line = line.as_str().parse().unwrap_or(0);
        let col = col.as_str().parse().unwrap_or(0);
        // The message is whatever follows the header line, up to the next blank
        // line or the next panic.
        let after = &output[capture.get(0).unwrap().end()..];
        let message: String = after
            .lines()
            .skip(1)
            .take_while(|l| !l.trim().is_empty() && !l.contains("panicked at"))
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
        let message = if message.is_empty() {
            after.lines().next().unwrap_or("").trim().to_string()
        } else {
            message
        };
        sites.push((file, line, col, message));
    }
    sites
}

/// The test names the runner marked as failed, if it printed any.
pub fn failed_tests(output: &str) -> Vec<String> {
    failed_test_regex()
        .captures_iter(output)
        .filter_map(|capture| {
            capture
                .get(1)
                .or_else(|| capture.get(2))
                .map(|m| m.as_str().to_string())
        })
        .collect()
}

/// Strip `:line:col` off a panic location so it can be compared to a scope
/// prefix, and normalise `./src/foo.rs` and `src\foo.rs` to `src/foo.rs`.
fn bare_location(location: &str) -> String {
    let location = location.replace('\\', "/");
    let location = location.as_str();
    let cut = location.rfind(':').map_or(location, |i| &location[..i]);
    let cut = cut.rfind(':').map_or(cut, |i| &cut[..i]);
    cut.trim_start_matches("./")
        .trim_start_matches('/')
        .to_string()
}

/// Whether a failure's file sits in the reported neighbourhood. A scope entry
/// matches as a path prefix either way, so `src/auth` covers `src/auth/mod.rs`
/// and `src/auth/login.rs` covers the directory it was reported against — and a
/// runner that printed the file under a longer root still matches.
pub fn in_neighbourhood(location: &str, scope: &[String]) -> bool {
    let file = bare_location(location);
    if file.is_empty() {
        return false;
    }
    scope.iter().any(|entry| {
        let entry = bare_location(entry);
        !entry.is_empty()
            && (file == entry
                || file.starts_with(&format!("{entry}/"))
                || entry.starts_with(&format!("{file}/"))
                || file.ends_with(&format!("/{entry}"))
                || entry.ends_with(&format!("/{file}")))
    })
}

/// Whether two ways of naming the same file agree. A runner prints a path
/// relative to whatever root it was started from, which for a workspace member
/// carries the member's directory and for a single crate does not, so a suffix
/// in either direction counts as the same file.
pub fn same_file(a: &str, b: &str) -> bool {
    let (a, b) = (bare_location(a), bare_location(b));
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a == b || a.ends_with(&format!("/{b}")) || b.ends_with(&format!("/{a}"))
}

/// Judge one completed run of a reproduction command.
///
/// `repro` is the reproduction file, and `scope` the reported bug's
/// neighbourhood. An assertion written in the test fails *in the test*, so a
/// location there is the ordinary case and proves nothing about the bug's
/// neighbourhood. What this can and must check is everything else: a failure
/// recorded in some third file happened neither in the reproduction nor where
/// the bug was reported, which is the shape of a test of unrelated code, a crash
/// in shared setup, or a deliberately broken file.
pub fn judge(
    exit: Option<i32>,
    timed_out: bool,
    spawn_error: Option<&str>,
    output: &str,
    repro: Option<&str>,
    scope: &[String],
) -> Verdict {
    if timed_out {
        return Verdict::TimedOut;
    }
    if let Some(why) = spawn_error {
        return Verdict::DidNotRun(why.to_string());
    }
    match exit {
        Some(0) => return Verdict::Passed,
        Some(_) => {}
        None => return Verdict::DidNotRun("the process was killed by a signal".to_string()),
    }
    // A failing signal or non-zero code is not by itself a reproduction: the
    // evidence is the assertion that failed. Take the first site the runner
    // printed, which is where execution stopped.
    let sites = panic_sites(output);
    let names = failed_tests(output);
    let Some((file, line, col, message)) = sites.first() else {
        return Verdict::NoFailure;
    };
    // Kept as `file:line:col`, the shape a reader of the evidence can open a
    // buffer at, and what the neighbourhood is matched against.
    let location = format!("{file}:{line}:{col}");
    let failure = Failure {
        location,
        line: Some(*line),
        message: message.clone(),
        test: names.first().cloned(),
    };
    let inside_repro = repro.is_some_and(|repro| same_file(&failure.location, repro));
    if inside_repro || scope.is_empty() || in_neighbourhood(&failure.location, scope) {
        Verdict::Red { failure }
    } else {
        Verdict::Suspect {
            failure,
            scope: scope.to_vec(),
        }
    }
}

/// What the gate says about one write the agent is attempting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteVerdict {
    /// Free to write.
    Allowed,
    /// Free, and now recorded as this fix's reproduction file.
    AcceptedReproduction(String),
    /// Refused, with the reason the model is told.
    Refused(String),
}

/// Path shapes that mean "a test", across the languages this project meets.
const TEST_DIRECTORY_SEGMENTS: &[&str] = &["tests", "test", "__tests__", "spec"];

/// Whether a workspace-relative path is a test file rather than production
/// code. The gate allows writes here while it waits for a red; anything else is
/// a production edit and is refused.
pub fn is_test_path(rel: &str) -> bool {
    let normalised = rel.replace('\\', "/");
    let mut parts = normalised.trim_matches('/').split('/').peekable();
    let file = parts.next_back().unwrap_or_default();
    let stem = file.split('.').next().unwrap_or_default();
    let in_test_dir = parts
        .clone()
        .filter(|p| !p.is_empty())
        .any(|p| TEST_DIRECTORY_SEGMENTS.contains(&p));
    if in_test_dir {
        return true;
    }
    // Outside a test directory, the file has to say what it is.
    file.ends_with("_test.rs")
        || file.ends_with("_test.go")
        || file.ends_with(".test.ts")
        || file.ends_with(".test.tsx")
        || file.ends_with(".test.js")
        || file.ends_with(".spec.ts")
        || file.ends_with(".spec.js")
        || file.ends_with("_spec.rb")
        || stem.starts_with("test_")
        || (stem.ends_with("Test") && stem.chars().next().is_some_and(|c| c.is_uppercase()))
}

#[derive(Debug, Default)]
struct GateState {
    phase: Phase,
    /// Whether this gate may refuse a write. Only a user-opened gate may: an
    /// agent that calls the tool on its own gets the measurement recorded and
    /// forbids nothing, because a refusal nobody asked for is a lock the agent
    /// cannot pick — and the escape from one is `release`, which is the user's
    /// command, not the model's.
    enforcing: bool,
    scope: Vec<String>,
    repro: Option<String>,
    command: Option<String>,
    evidence: Option<Evidence>,
    /// Production writes this gate refused, for the transcript's count.
    refusals: usize,
}

/// The session's reproduction gate. One per session, shared across its runs the
/// same way the secret taint bit is: the state is about the bug being fixed, not
/// about the turn that noticed it, and a gate that reset between turns would let
/// the second turn edit freely on the first turn's evidence.
#[derive(Debug, Default)]
pub struct ReproGate {
    state: Arc<Mutex<GateState>>,
}

impl ReproGate {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GateState> {
        // A poisoned lock holds plain fields and no invariant this gate can
        // violate by reading them; keep the decision rather than panic a tool.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn phase(&self) -> Phase {
        self.lock().phase
    }

    pub fn is_off(&self) -> bool {
        self.lock().phase == Phase::Off
    }

    /// The recorded reproduction file, if one has been accepted.
    pub fn repro_path(&self) -> Option<String> {
        self.lock().repro.clone()
    }

    pub fn scope(&self) -> Vec<String> {
        self.lock().scope.clone()
    }

    pub fn command(&self) -> Option<String> {
        self.lock().command.clone()
    }

    pub fn evidence(&self) -> Option<Evidence> {
        self.lock().evidence.clone()
    }

    pub fn refusals(&self) -> usize {
        self.lock().refusals
    }

    /// Start supervising a bug fix. `scope` is the reported neighbourhood: the
    /// paths the bug is said to live in, which every recorded failure is checked
    /// against. `enforcing` decides whether this gate may refuse a write — a user
    /// opening it does, an agent measuring a fix on its own does not. Re-engaging
    /// resets the measurement, because it starts a new bug.
    pub fn engage<S: AsRef<str>>(&self, scope: &[S], enforcing: bool) {
        let mut state = self.lock();
        state.phase = Phase::AwaitingRed;
        state.enforcing = enforcing;
        state.scope = scope
            .iter()
            .map(|s| s.as_ref().trim_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .collect();
        state.repro = None;
        state.command = None;
        state.evidence = None;
        state.refusals = 0;
    }

    /// Stop supervising. Only the user asks for this: releasing the gate is how a
    /// fix that never reproduced would get written, so it is never automatic.
    /// The measurement goes with it — a half-recorded red is not evidence, and
    /// leaving it behind would let a new bug inherit the last one's failure.
    pub fn release(&self) {
        let mut state = self.lock();
        state.phase = Phase::Off;
        state.enforcing = false;
        state.scope = Vec::new();
        state.repro = None;
        state.command = None;
        state.evidence = None;
    }

    /// Put the gate back to waiting for a failure, keeping what was written.
    /// Used when a reproduction is re-declared after a suspect failure.
    pub fn await_red(&self) {
        let mut state = self.lock();
        state.phase = Phase::AwaitingRed;
    }

    /// Whether this gate is the kind that refuses writes.
    pub fn is_enforcing(&self) -> bool {
        self.lock().enforcing
    }

    /// The gate's answer to one write of a workspace-relative path. Called for
    /// every edit-class tool before it runs.
    pub fn check_write(&self, rel: &str) -> WriteVerdict {
        let mut state = self.lock();
        if !state.enforcing {
            return WriteVerdict::Allowed;
        }
        match state.phase {
            Phase::Off => WriteVerdict::Allowed,
            Phase::AwaitingRed => {
                if !is_test_path(rel) {
                    state.refusals += 1;
                    return WriteVerdict::Refused(format!(
                        "the reproduction gate is waiting for a failing test before any \
                         production file changes. {rel} is not a test file. Write the \
                         reproduction test under a test path, then run it with \
                         {REPRO_TOOL}."
                    ));
                }
                let recorded = state.repro.clone();
                match recorded {
                    None => {
                        state.repro = Some(rel.to_string());
                        if let Some(evidence) = state.evidence.as_mut() {
                            evidence.repro = rel.to_string();
                        }
                        WriteVerdict::AcceptedReproduction(rel.to_string())
                    }
                    Some(ref path) if path == rel => WriteVerdict::Allowed,
                    Some(path) => {
                        state.refusals += 1;
                        WriteVerdict::Refused(format!(
                            "this fix already has one reproduction file, {path}. {rel} \
                             would be a second; finish the measurement on the first."
                        ))
                    }
                }
            }
            // The reproduction is frozen once its failure has been seen. Editing
            // it now is how a pass is manufactured out of the assertion that
            // failed, which is exactly the failure mode this gate exists to stop.
            Phase::RedWitnessed => {
                if state.repro.as_deref() == Some(rel) {
                    state.refusals += 1;
                    return WriteVerdict::Refused(format!(
                        "{rel} is this fix's reproduction and its failure is already on \
                         record. Change the production code and re-run {REPRO_TOOL}; to \
                         replace the reproduction itself, have the user re-engage the gate."
                    ));
                }
                WriteVerdict::Allowed
            }
            Phase::Verified => WriteVerdict::Allowed,
        }
    }

    /// Fill in the reported neighbourhood after the gate is already open, for a
    /// reproduction whose scope the agent declares when it runs the test rather
    /// than when the user engaged the gate.
    pub fn set_scope<S: AsRef<str>>(&self, scope: &[S]) {
        let mut state = self.lock();
        if state.scope.is_empty() {
            let declared: Vec<String> = scope
                .iter()
                .map(|s| s.as_ref().trim_matches('/').to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if declared.is_empty() {
                return;
            }
            let for_evidence = declared.clone();
            state.scope = declared;
            if let Some(evidence) = state.evidence.as_mut() {
                evidence.scope = for_evidence;
            }
        }
    }

    /// Declare the reproduction file and its neighbourhood without writing
    /// anything, for a test that already exists in the tree.
    pub fn declare_repro(&self, rel: &str) -> Option<String> {
        let mut state = self.lock();
        if state.phase != Phase::AwaitingRed {
            return None;
        }
        match &state.repro {
            None => {
                state.repro = Some(rel.to_string());
                None
            }
            Some(recorded) if recorded == rel => None,
            Some(recorded) => Some(format!(
                "this fix already has one reproduction file, {recorded}. {rel} would be a second."
            )),
        }
    }

    /// Record the command the measurement is made with. The green must re-run the
    /// same command, or the two points are not comparable.
    pub fn set_command(&self, command: &str) -> Result<(), String> {
        let mut state = self.lock();
        match &state.command {
            None => {
                state.command = Some(command.to_string());
                if let Some(evidence) = state.evidence.as_mut() {
                    evidence.command = command.to_string();
                }
                Ok(())
            }
            Some(recorded) if recorded == command => Ok(()),
            Some(recorded) => Err(format!(
                "the reproduction command is already recorded as `{recorded}`; a green \
                 measured with a different command is not the same measurement"
            )),
        }
    }

    /// The failure has been seen. Unlock production edits and record the
    /// assertion and optional artifact path as the evidence.
    pub fn record_red_with_artifact(
        &self,
        failure: Failure,
        exit: Option<i32>,
        artifact: Option<String>,
    ) {
        let mut state = self.lock();
        let repro = state.repro.clone().unwrap_or_default();
        let command = state.command.clone().unwrap_or_default();
        let scope = state.scope.clone();
        state.phase = Phase::RedWitnessed;
        state.evidence = Some(Evidence {
            repro,
            command,
            scope,
            red: Some(failure),
            red_exit: exit,
            red_artifact: artifact,
            green_exit: None,
            green_artifact: None,
            suite_exit: None,
            suite_command: None,
        });
    }

    pub fn record_red(&self, failure: Failure, exit: Option<i32>) {
        self.record_red_with_artifact(failure, exit, None);
    }

    /// The same command now passes. The measurement is complete.
    pub fn record_green_with_artifact(&self, exit: Option<i32>, artifact: Option<String>) {
        let mut state = self.lock();
        state.phase = Phase::Verified;
        if let Some(evidence) = state.evidence.as_mut() {
            evidence.green_exit = exit;
            evidence.green_artifact = artifact;
        }
    }

    pub fn record_green(&self, exit: Option<i32>) {
        self.record_green_with_artifact(exit, None);
    }

    /// Record the surrounding suite's exit code, run against the fixed tree.
    pub fn record_suite(&self, command: &str, exit: Option<i32>) {
        let mut state = self.lock();
        if let Some(evidence) = state.evidence.as_mut() {
            evidence.suite_command = Some(command.to_string());
            evidence.suite_exit = exit;
        }
    }

    /// One line for the status bar and the transcript.
    pub fn status_line(&self) -> String {
        let state = self.lock();
        let mut line = format!("reproduction gate: {}", state.phase.label());
        if state.phase != Phase::Off && !state.enforcing {
            line.push_str(" (recording only)");
        }
        if let Some(repro) = &state.repro {
            line.push_str(&format!(" · {repro}"));
        }
        if let Some(evidence) = &state.evidence {
            if let Some(red) = &evidence.red {
                line.push_str(&format!(" · red {}", red.short()));
            }
        }
        if state.refusals > 0 {
            line.push_str(&format!(" · {} writes refused", state.refusals));
        }
        line
    }
}

/// A reproduction run, kept apart from the command output the agent is shown so
/// the judgement reads the whole capture.
#[derive(Debug, Clone)]
pub struct Run {
    pub exit: Option<i32>,
    pub output: String,
    pub timed_out: bool,
    pub spawn_error: Option<String>,
}

impl Run {
    pub fn verdict(&self, repro: Option<&str>, scope: &[String]) -> Verdict {
        judge(
            self.exit,
            self.timed_out,
            self.spawn_error.as_deref(),
            &self.output,
            repro,
            scope,
        )
    }

    /// Enough of the output to act on: the last lines, where a compile error or
    /// a runner's summary lives.
    pub fn excerpt(&self) -> String {
        let tail = self.output.lines().rev().take(20).collect::<Vec<_>>();
        tail.into_iter().rev().collect::<Vec<_>>().join("\n")
    }
}

/// What one `{REPRO_TOOL}` call decided, in the words the model is answered
/// with. Every branch states the reason, because a bare refusal teaches an agent
/// to retry rather than to write a better reproduction.
pub async fn reproduce_bug(
    gate: &ReproGate,
    root: &Path,
    args: &serde_json::Map<String, serde_json::Value>,
    timeout_secs: u64,
    sandbox: &crate::sandbox::Sandbox,
    session: Option<&str>,
) -> String {
    let Some(raw_path) = args.get("path").and_then(|v| v.as_str()) else {
        return "error: reproduce_bug needs a string \"path\" — the reproduction test's file"
            .to_string();
    };
    let Some(command) = args.get("command").and_then(|v| v.as_str()) else {
        return "error: reproduce_bug needs a string \"command\" — the command that runs the test"
            .to_string();
    };
    let Ok((full, display)) = crate::agent_tools::workspace_path(root, raw_path) else {
        return format!("error: {raw_path} is outside the workspace, so it is not a reproduction of a bug in this project.");
    };
    if !is_test_path(&display) {
        return format!(
            "error: {display} is not a test file. A reproduction is a test — under a \
             `tests/` directory or named as one — so that the gate can tell the bug's \
             evidence apart from the code being fixed."
        );
    }
    if !full.is_file() {
        return format!(
            "error: {display} does not exist. Write the reproduction first, then run it."
        );
    }
    let scope: Vec<String> = args
        .get("scope")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.trim_matches('/').to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let suite = args
        .get("suite")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    // A fix measured without the user opening the gate gets its evidence
    // recorded and forbids nothing: a lock the model chose for itself has no way
    // out, since releasing the gate is the user's command.
    if gate.is_off() {
        gate.engage::<&str>(&[], false);
        gate.set_scope(&scope);
    } else {
        gate.set_scope(&scope);
    }
    // Once a failure is on record, the measurement belongs to the file that
    // produced it. A second test run green proves only that the second test
    // passes.
    if gate.phase() != Phase::AwaitingRed {
        if let Some(recorded) = gate.repro_path() {
            if recorded != display {
                return format!(
                    "error: this fix's reproduction is recorded as {recorded}, not {display}. \
                      Re-run the one that failed."
                );
            }
        }
    }
    if let Some(conflict) = gate.declare_repro(&display) {
        return format!("error: {conflict}");
    }
    if let Err(reason) = gate.set_command(command) {
        return format!("error: {reason}");
    }
    let declared_scope = gate.scope();
    let timeout = timeout_secs.max(REPRO_MIN_TIMEOUT_SECS);
    let xencode_dir = root.join(".xencode");
    let session_tag = args
        .get("session")
        .and_then(|v| v.as_str())
        .or(session)
        .unwrap_or("chat");

    if gate.phase() == Phase::AwaitingRed {
        let run = witness(root, command, timeout, sandbox).await;
        return match run.verdict(Some(&display), &declared_scope) {
            Verdict::Red { failure } => {
                let exit = run.exit;
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let redacted = xencode_context_rs::trace::redact_secrets(&run.output);
                let art_path = xencode_context_rs::artifacts::write_artifact(
                    &xencode_dir,
                    session_tag,
                    &format!("repro-red-{now_ms}.log"),
                    &redacted,
                )
                .ok();
                let art_rel = art_path.as_ref().map(|p| display_relative(root, p));
                if let Some(ref rel) = art_rel {
                    let _ = xencode_context_rs::ledger::append_ledger(
                        &xencode_dir,
                        &xencode_context_rs::ledger::LedgerEntry {
                            ts_unix_ms: now_ms,
                            session: Some(session_tag.to_string()),
                            run_class: xencode_context_rs::ledger::RunClass::Test,
                            exit_code: exit.unwrap_or(1),
                            subjects: vec![xencode_context_rs::ledger::digest_hex(command)],
                            log_ref: rel.clone(),
                            note: format!("repro red: {}", failure.short()),
                        },
                    );
                }
                gate.record_red_with_artifact(failure.clone(), exit, art_rel);
                let scope_note = if declared_scope.is_empty() {
                    "\nNo neighbourhood was declared for this bug, so the failure's location \nwent unchecked: it is recorded, and that is the weaker form of this measurement."
                } else {
                    ""
                };
                format!(
                    "red witnessed.\n{}\n\nThe failure is on record, so production edits are \
                     unlocked now. Change the code, then call {REPRO_TOOL} again with the same \
                     path and command: the fix counts when that same run passes. Do not edit \
                     {display} — its assertion is the measurement.{scope_note}",
                    failure.short()
                )
            }
            Verdict::Suspect { failure, scope } => format!(
                "error: suspect reproduction — {} is outside the reported bug's neighbourhood \
                 [{}]. The run did fail, but not where the bug was reported, so this is not \
                 accepted as the reproduction: it may be testing code the bug never touched, or \
                 failing for an unrelated reason. Either name the code the bug is in via \
                 \"scope\", or reproduce the reported behaviour itself.\n{}",
                failure.short(),
                scope.join(", "),
                run.excerpt()
            ),
            Verdict::Passed => format!(
                "error: rejected — {display} passed on the tree as it stands, so it does not \
                 reproduce the bug. A test that cannot fail proves nothing about a fix. Make the \
                 reproduction assert the reported behaviour, against code you have not changed."
            ),
            Verdict::NoFailure => format!(
                "error: rejected — the run exited {} but no failing assertion was recorded, so \
                 there is nothing to hold a fix against. A crash in unrelated setup, a compile \
                 error or a killed process is not a reproduction. If the test catches a panic \
                 instead of letting it fail the run, let it fail.\n{}",
                run.exit
                    .map_or("on a signal".to_string(), |c| format!("non-zero ({c})")),
                run.excerpt()
            ),
            Verdict::TimedOut => format!(
                "error: the reproduction ran out of its {timeout}s budget. Narrow the command to \
                 the one test, or raise `agent_command_timeout`."
            ),
            Verdict::DidNotRun(why) => format!("error: the reproduction never ran — {why}"),
        };
    }

    // Phase 2: the failure is already on record, so this run is the second point
    // of the measurement and must be the same command (checked above).
    let run = witness(root, command, timeout, sandbox).await;
    let evidence_note = match gate.evidence() {
        Some(evidence) => match &evidence.red {
            Some(red) => format!("recorded red: {}", red.short()),
            None => "recorded red: (none)".to_string(),
        },
        None => String::new(),
    };
    match run.verdict(Some(&display), &declared_scope) {
        Verdict::Passed => {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let redacted = xencode_context_rs::trace::redact_secrets(&run.output);
            let art_path = xencode_context_rs::artifacts::write_artifact(
                &xencode_dir,
                session_tag,
                &format!("repro-green-{now_ms}.log"),
                &redacted,
            )
            .ok();
            let art_rel = art_path.as_ref().map(|p| display_relative(root, p));
            if let Some(ref rel) = art_rel {
                let _ = xencode_context_rs::ledger::append_ledger(
                    &xencode_dir,
                    &xencode_context_rs::ledger::LedgerEntry {
                        ts_unix_ms: now_ms,
                        session: Some(session_tag.to_string()),
                        run_class: xencode_context_rs::ledger::RunClass::Test,
                        exit_code: run.exit.unwrap_or(0),
                        subjects: vec![xencode_context_rs::ledger::digest_hex(command)],
                        log_ref: rel.clone(),
                        note: "repro green passed".to_string(),
                    },
                );
            }
            gate.record_green_with_artifact(run.exit, art_rel);
            let mut reply = format!(
                "red to green. {display} failed on the unmodified tree and passes on this one. \
                 {evidence_note}\nThe fix is evidenced for this reproduction."
            );
            let mut suite_exit = None;
            if let Some(suite) = suite {
                let suite_run = witness(root, &suite, timeout, sandbox).await;
                suite_exit = suite_run.exit;
                gate.record_suite(&suite, suite_run.exit);
                reply.push_str(&match suite_run.exit {
                    Some(0) => format!("\nSuite `{suite}` passed."),
                    Some(code) => format!(
                        "\nSuite `{suite}` FAILED (exit {code}) — this change broke something:\n{}",
                        suite_run.excerpt()
                    ),
                    None => format!(
                        "\nSuite `{suite}` did not finish: {}",
                        suite_run
                            .spawn_error
                            .clone()
                            .unwrap_or_else(|| "timed out".to_string())
                    ),
                });
            } else {
                reply.push_str(
                    "\nThe surrounding suite has not been run, so nothing here says the rest of \
                     the tree still works. Run it before calling this fixed.",
                );
            }
            if let Some(ev) = gate.evidence() {
                let record = ReproRecord {
                    session: Some(session_tag.to_string()),
                    repro: ev.repro.clone(),
                    command: ev.command.clone(),
                    scope: ev.scope.clone(),
                    red_exit: ev.red_exit,
                    red_artifact: ev.red_artifact.clone(),
                    red_failure: ev.red.clone(),
                    green_exit: ev.green_exit,
                    green_artifact: ev.green_artifact.clone(),
                    suite_exit: ev.suite_exit.or(suite_exit),
                    suite_command: ev.suite_command.clone(),
                    ts_unix_ms: now_ms,
                };
                let _ = append_repro_record(&xencode_dir, &record);
            }
            reply
        }
        other => {
            let still = if gate.phase() == Phase::Verified {
                "and a measurement that was complete has gone red again"
            } else {
                "the fix has not landed"
            };
            let why = match &other {
                Verdict::Red { failure } => format!("it fails at {}", failure.short()),
                Verdict::Suspect { failure, .. } => {
                    format!(
                        "it fails at {}, outside the reported neighbourhood",
                        failure.short()
                    )
                }
                Verdict::NoFailure => "it failed with no assertion on record".to_string(),
                Verdict::TimedOut => format!("it ran out of its {timeout}s budget"),
                Verdict::DidNotRun(why) => format!("it never ran: {why}"),
                Verdict::Passed => "it passed".to_string(),
            };
            format!(
                "not green: {display} {still} — {why}. The measurement needs this same command \
                 to pass.\n{}",
                run.excerpt()
            )
        }
    }
}

/// Run the reproduction for real, in the workspace root, under the same
/// isolation `run_command` gets (SE-7). Truncated to the last `REPRO_OUTPUT_CAP`
/// bytes so a long build log cannot crowd the assertion out of memory, though
/// the panic sites we need are usually near the end.
pub async fn witness(
    root: &Path,
    command: &str,
    timeout_secs: u64,
    sandbox: &crate::sandbox::Sandbox,
) -> Run {
    let (program, arguments) = match sandbox.wrap(command, false) {
        Ok(Some((program, wrap_args))) => (program, wrap_args),
        Ok(None) => (
            "sh".to_string(),
            vec!["-c".to_string(), command.to_string()],
        ),
        Err(reason) => {
            return Run {
                exit: None,
                output: String::new(),
                timed_out: false,
                spawn_error: Some(reason),
            }
        }
    };
    let mut built = tokio::process::Command::new(&program);
    built
        .args(&arguments)
        .current_dir(root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let spawned = built.spawn();
    let Ok(child) = spawned else {
        return Run {
            exit: None,
            output: String::new(),
            timed_out: false,
            spawn_error: Some(format!(
                "could not start `{program}`: {}",
                spawned.unwrap_err()
            )),
        };
    };
    let wait = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs.max(1)),
        child.wait_with_output(),
    );
    match wait.await {
        Err(_) => Run {
            exit: None,
            output: String::new(),
            timed_out: true,
            spawn_error: None,
        },
        Ok(Err(e)) => Run {
            exit: None,
            output: String::new(),
            timed_out: false,
            spawn_error: Some(format!("the run did not finish: {e}")),
        },
        Ok(Ok(output)) => {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&output.stdout);
            bytes.extend_from_slice(&output.stderr);
            let text = String::from_utf8_lossy(&bytes).to_string();
            let text = if text.len() > REPRO_OUTPUT_CAP {
                let start = text.len() - REPRO_OUTPUT_CAP;
                format!("(output truncated) …{}", &text[start..])
            } else {
                text
            };
            Run {
                exit: output.status.code(),
                output: text,
                timed_out: false,
                spawn_error: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_root(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "xencode-reprogate-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join("tests")).unwrap();
        dir
    }

    /// A dependency-free crate with a real bug in it, so that every judgement
    /// below is made from output a real `cargo test` run actually printed.
    fn write_crate(root: &Path, library: &str, test: &str) {
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"thing\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
             [workspace]\n",
        )
        .unwrap();
        std::fs::write(root.join("src/lib.rs"), library).unwrap();
        std::fs::write(root.join("tests/repro.rs"), test).unwrap();
    }

    /// `--offline` because the crate has no dependencies and must not reach a
    /// registry; `CARGO_TARGET_DIR` is cleared because this test process inherits
    /// one from the workspace it runs inside, and the throwaway crate belongs in
    /// its own directory.
    const REPRO_COMMAND: &str = "env -u CARGO_TARGET_DIR cargo test --offline";
    const SUITE_COMMAND: &str = "env -u CARGO_TARGET_DIR cargo test --offline --test repro";

    /// The bug: `doubled` adds instead of multiplying. The test asserts the
    /// documented behaviour, so it fails where it is written.
    const BUGGY_LIB: &str = "pub fn doubled(value: i32) -> i32 {\n    value + 1\n}\n";
    const FIXED_LIB: &str = "pub fn doubled(value: i32) -> i32 {\n    value * 2\n}\n";
    const ASSERTING_TEST: &str = "use thing::doubled;\n\n#[test]\nfn doubled_two_is_four() {\n    \
                                  assert_eq!(doubled(2), 4, \"doubled must return twice its input\");\n}\n";
    /// The bug as a crash inside the library instead: the failure lands in
    /// `src/lib.rs`, which is where the neighbourhood check has something to say.
    const PANICKING_LIB: &str =
        "pub fn doubled(_value: i32) -> i32 {\n    panic!(\"doubled is not implemented\")\n}\n";
    const CALLING_TEST: &str = "use thing::doubled;\n\n#[test]\nfn doubled_two_is_four() {\n    \
                                assert_eq!(doubled(2), 4);\n}\n";

    async fn run_command(root: &Path, command: &str) -> Run {
        witness(root, command, 240, &crate::sandbox::Sandbox::disabled()).await
    }

    fn tool_args(
        path: &str,
        command: &str,
        scope: &[&str],
        suite: Option<&str>,
    ) -> serde_json::Map<String, serde_json::Value> {
        let mut args = serde_json::Map::new();
        args.insert("path".to_string(), serde_json::json!(path));
        args.insert("command".to_string(), serde_json::json!(command));
        args.insert("scope".to_string(), serde_json::json!(scope));
        if let Some(suite) = suite {
            args.insert("suite".to_string(), serde_json::json!(suite));
        }
        args
    }

    #[test]
    fn a_test_file_is_told_apart_from_the_code_it_tests() {
        for is_test in [
            "tests/repro.rs",
            "crates/x/tests/cli.rs",
            "src/auth/tests/login.rs",
            "app/javascript/__tests__/cart.test.js",
            "pkg/handler/api_test.go",
            "spec/models/user_spec.rb",
            "tests/test_parsing.py",
            "src/test/java/com/acme/LoginTest.java",
            "src/main/kotlin/com/acme/LoginTest.kt",
        ] {
            assert!(is_test_path(is_test), "{is_test} is a test file");
        }
        for production in [
            "src/lib.rs",
            "src/auth/login.rs",
            "src/contest.rs",
            "src/testing/mod.rs",
            "tests_helpers/cache.rs",
            "crates/x/src/tests_support.rs",
            "",
        ] {
            assert!(!is_test_path(production), "{production} is not a test file");
        }
    }

    #[test]
    fn a_windows_panic_location_is_read_with_forward_slashes() {
        let output =
            "thread 'doubled_two_is_four' panicked at tests\\repro.rs:5:5:\nassertion failed\n";
        let sites = panic_sites(output);
        assert_eq!(sites[0].0, "tests/repro.rs");
        assert!(same_file("tests\\repro.rs:5:5", "tests/repro.rs"));
        assert!(in_neighbourhood(
            "src\\auth\\login.rs:5:9",
            &["src/auth".to_string()]
        ));
        assert!(in_neighbourhood(
            "src/auth/login.rs:5:9",
            &["src\\auth".to_string()]
        ));
    }

    #[test]
    fn a_neighbourhood_covers_a_file_a_directory_and_a_deeper_path() {
        let one = |s: &str| vec![s.to_string()];
        assert!(in_neighbourhood("src/auth/login.rs:5:9", &one("src/auth")));
        assert!(in_neighbourhood("src/auth.rs:5:9", &one("src/auth.rs")));
        assert!(in_neighbourhood("src/auth.rs:5:9", &one("src")));
        assert!(in_neighbourhood(
            "src/auth/login.rs:1:1",
            &one("src/auth/login.rs")
        ));
        assert!(!in_neighbourhood("src/billing.rs:5:9", &one("src/auth")));
        // A workspace member's runner prints the member's own directory in front.
        assert!(in_neighbourhood(
            "rust/crates/x/src/auth.rs:1:1",
            &one("src/auth.rs")
        ));
        assert!(!in_neighbourhood("src/billing.rs:1:1", &one("")));
    }

    #[test]
    fn a_bare_panic_and_a_runner_summary_are_both_read() {
        // Output taken from a real run of the throwaway crate below, to pin the
        // reader's shape without paying for a compile: file, line, column, then
        // the assertion's own words.
        let output = "running 1 test\ntest doubled_two_is_four ... FAILED\n\nfailures:\n\n---- doubled_two_is_four stdout ----\nthread 'doubled_two_is_four' panicked at tests/repro.rs:5:5:\nassertion `left == right` failed: doubled must return twice its input\n  left: 3\n right: 4\nnote: run with `RUST_BACKTRACE=1` environment variable to display a backtrace\n";
        let sites = panic_sites(output);
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].0, "tests/repro.rs");
        assert_eq!(sites[0].1, 5);
        assert_eq!(sites[0].2, 5);
        assert!(sites[0].3.contains("left == right"), "{}", sites[0].3);
        assert!(
            sites[0].3.contains("doubled must return twice"),
            "the assertion's message was cut: {}",
            sites[0].3
        );
        assert_eq!(failed_tests(output), vec!["doubled_two_is_four"]);
        // A runner that prints no `test … FAILED` line still yields its panic.
        assert!(failed_tests("thread 'main' panicked at src/x.rs:1:1:\nboom\n").is_empty());
    }

    #[tokio::test]
    async fn a_real_failing_assertion_is_recorded_with_its_file_and_line() {
        let root = temp_root("recorded-red");
        write_crate(&root, BUGGY_LIB, ASSERTING_TEST);
        let run = run_command(&root, REPRO_COMMAND).await;
        assert_eq!(run.exit, Some(101), "{}", run.excerpt());
        let verdict = run.verdict(Some("tests/repro.rs"), &["src".to_string()]);
        let Verdict::Red { failure } = &verdict else {
            panic!(
                "a real failing test was not read as a red: {verdict:?}\n{}",
                run.excerpt()
            );
        };
        // The assertion lives in the reproduction, which is where a Rust assert
        // fails, so the neighbourhood of the bug cannot rule it out.
        assert_eq!(
            failure.location,
            "tests/repro.rs:5:5",
            "{}",
            failure.short()
        );
        assert_eq!(failure.line, Some(5));
        assert_eq!(failure.test.as_deref(), Some("doubled_two_is_four"));
        assert!(failure.message.contains("left: 3"), "{}", failure.message);
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_failure_in_code_nobody_reported_is_suspect_not_accepted() {
        let root = temp_root("suspect-red");
        write_crate(&root, PANICKING_LIB, CALLING_TEST);
        let run = run_command(&root, REPRO_COMMAND).await;
        assert_eq!(run.exit, Some(101), "{}", run.excerpt());
        // The panic is in `src/lib.rs`. Named as the bug's home, that is a red.
        let inside = run.verdict(Some("tests/repro.rs"), &["src/lib.rs".to_string()]);
        assert!(
            matches!(&inside, Verdict::Red { failure } if failure.location == "src/lib.rs:2:5"),
            "{inside:?}"
        );
        // The same run, reported against a different file, is not that red.
        let outside = run.verdict(Some("tests/repro.rs"), &["src/billing.rs".to_string()]);
        let Verdict::Suspect { failure, scope } = &outside else {
            panic!("a failure outside the reported bug was accepted: {outside:?}");
        };
        assert!(
            failure.location.starts_with("src/lib.rs:"),
            "{}",
            failure.location
        );
        assert_eq!(scope, &vec!["src/billing.rs".to_string()]);
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_reproduction_that_passes_reproduces_nothing() {
        let root = temp_root("passed-tree");
        write_crate(&root, FIXED_LIB, ASSERTING_TEST);
        let run = run_command(&root, REPRO_COMMAND).await;
        assert_eq!(run.exit, Some(0), "{}", run.excerpt());
        assert_eq!(
            run.verdict(Some("tests/repro.rs"), &["src".to_string()]),
            Verdict::Passed
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_non_zero_exit_with_no_assertion_on_record_is_refused() {
        let root = temp_root("no-assertion");
        // A build break, not a failing test: non-zero, no panic site anywhere.
        let run = run_command(&root, "sh -c 'echo \"error: something broke\"; exit 3'").await;
        assert_eq!(run.exit, Some(3), "{}", run.excerpt());
        assert_eq!(
            run.verdict(Some("tests/repro.rs"), &["src".to_string()]),
            Verdict::NoFailure
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_command_that_cannot_be_started_is_reported_as_never_having_run() {
        let root = temp_root("did-not-run");
        let run = run_command(&root, "__xencode_no_such_program__ --version").await;
        assert!(run.exit.is_none() || run.exit != Some(0), "{run:?}");
        let verdict = run.verdict(Some("tests/repro.rs"), &[]);
        assert!(
            matches!(verdict, Verdict::NoFailure | Verdict::DidNotRun(_)),
            "{verdict:?}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_gate_waiting_for_its_failure_refuses_the_code_it_would_fix() {
        let gate = ReproGate::new();
        gate.engage(&["src/auth"], true);
        let refused = gate.check_write("src/auth/login.rs");
        let WriteVerdict::Refused(reason) = &refused else {
            panic!("a production write was allowed while awaiting a red: {refused:?}");
        };
        assert!(reason.contains("not a test file"), "{reason}");
        assert!(reason.contains(REPRO_TOOL), "{reason}");
        // The reproduction is the one write this phase allows, and it is recorded.
        assert_eq!(
            gate.check_write("tests/repro.rs"),
            WriteVerdict::AcceptedReproduction("tests/repro.rs".to_string())
        );
        assert_eq!(gate.repro_path().as_deref(), Some("tests/repro.rs"));
        // Re-writing it while it takes shape is allowed; a second file is not.
        assert_eq!(gate.check_write("tests/repro.rs"), WriteVerdict::Allowed);
        let second = gate.check_write("tests/other.rs");
        assert!(
            matches!(&second, WriteVerdict::Refused(reason) if reason.contains("one reproduction file")),
            "{second:?}"
        );
        // A write outside the workspace names nothing the gate can allow.
        let unnamed = gate.check_write("");
        assert!(matches!(unnamed, WriteVerdict::Refused(_)), "{unnamed:?}");
        assert_eq!(gate.refusals(), 3);
        assert!(gate.is_enforcing());
    }

    #[test]
    fn a_gate_the_agent_opened_itself_refuses_nothing() {
        // Only a user-opened gate may lock the session: the model calling the tool
        // asks for evidence, not for permission it cannot withdraw.
        let gate = ReproGate::new();
        gate.engage(&["src/auth"], false);
        assert_eq!(gate.check_write("src/auth/login.rs"), WriteVerdict::Allowed);
        assert_eq!(gate.check_write("tests/repro.rs"), WriteVerdict::Allowed);
        assert_eq!(gate.refusals(), 0);
        assert!(!gate.is_enforcing());
        assert!(
            gate.status_line().contains("recording only"),
            "{}",
            gate.status_line()
        );
    }

    #[test]
    fn the_reproduction_is_frozen_once_its_failure_is_on_record() {
        let gate = ReproGate::new();
        gate.engage(&["src"], true);
        assert!(matches!(
            gate.check_write("tests/repro.rs"),
            WriteVerdict::AcceptedReproduction(_)
        ));
        gate.record_red(
            Failure {
                location: "tests/repro.rs:5:5".to_string(),
                line: Some(5),
                message: "assertion `left == right` failed".to_string(),
                test: Some("doubled_two_is_four".to_string()),
            },
            Some(101),
        );
        // The code under test is unlocked now…
        assert_eq!(gate.check_write("src/lib.rs"), WriteVerdict::Allowed);
        // …and the assertion that failed is not: rewriting it is how a pass is
        // manufactured out of the failure that was witnessed.
        let frozen = gate.check_write("tests/repro.rs");
        let WriteVerdict::Refused(reason) = &frozen else {
            panic!("the reproduction could be edited after its red: {frozen:?}");
        };
        assert!(reason.contains("already on record"), "{reason}");
        assert_eq!(gate.phase(), Phase::RedWitnessed);
        let line = gate.status_line();
        assert!(line.contains("tests/repro.rs:5"), "{line}");
    }

    #[test]
    fn the_green_has_to_come_from_the_command_that_failed() {
        let gate = ReproGate::new();
        gate.engage::<&str>(&[], true);
        gate.set_command("cargo test --offline").unwrap();
        // The same command again is the second point of one measurement.
        gate.set_command("cargo test --offline").unwrap();
        // Anything else is a different measurement, and says so.
        let err = gate.set_command("echo all good").unwrap_err();
        assert!(err.contains("already recorded as"), "{err}");
        assert!(err.contains("not the same measurement"), "{err}");
    }

    #[tokio::test]
    async fn a_fix_is_measured_red_then_green_on_one_real_command() {
        let root = temp_root("red-to-green");
        write_crate(&root, BUGGY_LIB, ASSERTING_TEST);
        let gate = ReproGate::new();
        gate.engage(&["src"], true);

        // 1. Nothing has failed yet, so the library cannot be touched.
        assert!(matches!(
            gate.check_write("src/lib.rs"),
            WriteVerdict::Refused(_)
        ));
        assert!(matches!(
            gate.check_write("tests/repro.rs"),
            WriteVerdict::AcceptedReproduction(_)
        ));

        // 2. Running the reproduction against the unchanged tree witnesses it.
        let red = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/repro.rs", REPRO_COMMAND, &["src"], None),
            1,
            &crate::sandbox::Sandbox::disabled(),
            Some("session-ae3"),
        )
        .await;
        assert!(red.starts_with("red witnessed."), "{red}");
        assert!(red.contains("tests/repro.rs:5:5"), "{red}");
        assert_eq!(gate.phase(), Phase::RedWitnessed);

        // 3. Now the fix is reachable, and the reproduction is not.
        assert_eq!(gate.check_write("src/lib.rs"), WriteVerdict::Allowed);
        assert!(matches!(
            gate.check_write("tests/repro.rs"),
            WriteVerdict::Refused(_)
        ));
        std::fs::write(root.join("src/lib.rs"), FIXED_LIB).unwrap();

        // 4. The same command against the fixed tree is the second point, and the
        //    suite named alongside it is genuinely run.
        let green = reproduce_bug(
            &gate,
            &root,
            &tool_args(
                "tests/repro.rs",
                REPRO_COMMAND,
                &["src"],
                Some(SUITE_COMMAND),
            ),
            1,
            &crate::sandbox::Sandbox::disabled(),
            Some("session-ae3"),
        )
        .await;
        assert!(green.starts_with("red to green."), "{green}");
        assert!(green.contains("recorded red: tests/repro.rs:5"), "{green}");
        assert!(
            green.contains("Suite `") && green.contains("passed."),
            "{green}"
        );
        assert_eq!(gate.phase(), Phase::Verified);
        let evidence = gate.evidence().expect("the measurement was not kept");
        assert_eq!(evidence.red_exit, Some(101), "{evidence:?}");
        assert_eq!(evidence.green_exit, Some(0));
        assert_eq!(evidence.suite_exit, Some(0));
        assert_eq!(evidence.repro, "tests/repro.rs");
        assert_eq!(evidence.command, REPRO_COMMAND);
        assert_eq!(evidence.scope, vec!["src".to_string()]);
        assert!(evidence.red_artifact.is_some());
        assert!(evidence.green_artifact.is_some());

        // AE-3: Evidence outlives the process. A second session sees that history.
        let xencode_dir = root.join(".xencode");
        let history = read_repro_history(&root, &xencode_dir);
        assert_eq!(history.len(), 1);
        let rec = &history[0];
        assert_eq!(rec.command, REPRO_COMMAND);
        assert_eq!(rec.red_exit, Some(101));
        assert_eq!(rec.green_exit, Some(0));
        assert_eq!(rec.session.as_deref(), Some("session-ae3"));
        assert!(rec.red_artifact.is_some());
        assert!(rec.green_artifact.is_some());

        // 5. A second test cannot claim this fix's green, and a different command
        //    cannot measure it.
        std::fs::write(root.join("tests/other.rs"), ASSERTING_TEST).unwrap();
        let wrong_file = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/other.rs", REPRO_COMMAND, &["src"], None),
            1,
            &crate::sandbox::Sandbox::disabled(),
            Some("session-ae3"),
        )
        .await;
        assert!(
            wrong_file.starts_with("error:") && wrong_file.contains("recorded as tests/repro.rs"),
            "{wrong_file}"
        );
        let wrong_command = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/repro.rs", "echo all good", &["src"], None),
            1,
            &crate::sandbox::Sandbox::disabled(),
            Some("session-ae3"),
        )
        .await;
        assert!(
            wrong_command.contains("already recorded as"),
            "{wrong_command}"
        );

        // Deleting either artifact removes the item from reproduction history.
        let red_art = rec.red_artifact.as_ref().unwrap();
        std::fs::remove_file(root.join(red_art)).unwrap();
        let history_after_delete = read_repro_history(&root, &xencode_dir);
        assert!(
            history_after_delete.is_empty(),
            "history must vanish if artifact is deleted"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_reproduction_that_passes_unmodified_never_unlocks_the_fix() {
        let root = temp_root("always-green");
        write_crate(&root, FIXED_LIB, ASSERTING_TEST);
        let gate = ReproGate::new();
        gate.engage(&["src"], true);
        assert!(matches!(
            gate.check_write("tests/repro.rs"),
            WriteVerdict::AcceptedReproduction(_)
        ));
        let refused = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/repro.rs", REPRO_COMMAND, &["src"], None),
            1,
            &crate::sandbox::Sandbox::disabled(),
            None,
        )
        .await;
        assert!(refused.starts_with("error: rejected"), "{refused}");
        assert!(refused.contains("does not reproduce the bug"), "{refused}");
        // Still no red, so still no production write.
        assert!(matches!(
            gate.check_write("src/lib.rs"),
            WriteVerdict::Refused(_)
        ));
        assert_eq!(gate.phase(), Phase::AwaitingRed);
        assert!(gate.evidence().is_none());
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_suspect_reproduction_is_flagged_and_changes_nothing() {
        let root = temp_root("outside-scope");
        write_crate(&root, PANICKING_LIB, CALLING_TEST);
        let gate = ReproGate::new();
        // The bug is reported in a file the actual failure never touches.
        gate.engage(&["src/billing.rs"], true);
        assert!(matches!(
            gate.check_write("tests/repro.rs"),
            WriteVerdict::AcceptedReproduction(_)
        ));
        let suspect = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/repro.rs", REPRO_COMMAND, &["src/billing.rs"], None),
            1,
            &crate::sandbox::Sandbox::disabled(),
            None,
        )
        .await;
        assert!(suspect.contains("suspect reproduction"), "{suspect}");
        assert!(suspect.contains("src/lib.rs:"), "{suspect}");
        // Suspect is not accepted: the gate has not moved and the code stays shut.
        assert_eq!(gate.phase(), Phase::AwaitingRed);
        assert!(gate.evidence().is_none());
        assert!(matches!(
            gate.check_write("src/lib.rs"),
            WriteVerdict::Refused(_)
        ));
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_failing_build_is_refused_because_nothing_was_asserted() {
        let root = temp_root("no-assertion-tool");
        write_crate(&root, BUGGY_LIB, ASSERTING_TEST);
        // Break the build: a non-zero run with no panic site in it.
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn doubled(value: i32) -> i32 {\n    oops\n}\n",
        )
        .unwrap();
        let gate = ReproGate::new();
        gate.engage(&["src"], true);
        assert!(matches!(
            gate.check_write("tests/repro.rs"),
            WriteVerdict::AcceptedReproduction(_)
        ));
        let refused = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/repro.rs", REPRO_COMMAND, &["src"], None),
            1,
            &crate::sandbox::Sandbox::disabled(),
            None,
        )
        .await;
        assert!(refused.starts_with("error: rejected"), "{refused}");
        assert!(refused.contains("no failing assertion"), "{refused}");
        // The compile error is quoted back, so the agent can act on it.
        assert!(refused.contains("error"), "{refused}");
        assert_eq!(gate.phase(), Phase::AwaitingRed);
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn the_reproduction_must_be_a_test_that_exists_in_the_workspace() {
        let root = temp_root("bad-arguments");
        write_crate(&root, BUGGY_LIB, ASSERTING_TEST);
        let gate = ReproGate::new();
        let sandbox = crate::sandbox::Sandbox::disabled();
        // Production code is not a reproduction.
        let not_a_test = reproduce_bug(
            &gate,
            &root,
            &tool_args("src/lib.rs", REPRO_COMMAND, &["src"], None),
            1,
            &sandbox,
            None,
        )
        .await;
        assert!(not_a_test.starts_with("error:"), "{not_a_test}");
        assert!(not_a_test.contains("is not a test file"), "{not_a_test}");
        assert!(gate.is_off(), "an invalid call must not open a gate");
        // A test file that has not been written cannot be run.
        let missing = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/not_written.rs", REPRO_COMMAND, &["src"], None),
            1,
            &sandbox,
            None,
        )
        .await;
        assert!(missing.contains("does not exist"), "{missing}");
        assert!(gate.is_off(), "{missing}");
        // A path outside the workspace is refused before anything runs.
        let outside = reproduce_bug(
            &gate,
            &root,
            &tool_args("../../etc/passwd", "cat", &[], None),
            1,
            &sandbox,
            None,
        )
        .await;
        assert!(outside.starts_with("error:"), "{outside}");
        assert!(gate.is_off(), "{outside}");
        // The arguments the tool cannot do without.
        let no_path = reproduce_bug(
            &gate,
            &root,
            &tool_args("", REPRO_COMMAND, &[], None),
            1,
            &sandbox,
            None,
        )
        .await;
        assert!(no_path.starts_with("error:"), "{no_path}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_gate_the_agent_opens_records_the_failure_and_locks_nothing() {
        let root = temp_root("recording-only");
        write_crate(&root, BUGGY_LIB, ASSERTING_TEST);
        let gate = ReproGate::new();
        assert!(gate.is_off());
        // No `/gate bugfix` was ever typed, so the library is writable throughout.
        assert_eq!(gate.check_write("src/lib.rs"), WriteVerdict::Allowed);
        let red = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/repro.rs", REPRO_COMMAND, &["src"], None),
            1,
            &crate::sandbox::Sandbox::disabled(),
            None,
        )
        .await;
        assert!(red.starts_with("red witnessed."), "{red}");
        assert!(
            red.contains("No neighbourhood was declared") || red.contains("tests/repro.rs:5"),
            "{red}"
        );
        assert!(!gate.is_enforcing());
        assert_eq!(gate.check_write("src/lib.rs"), WriteVerdict::Allowed);
        // The measurement it recorded is still the one the green has to match.
        assert_eq!(gate.command().as_deref(), Some(REPRO_COMMAND));
        assert_eq!(gate.refusals(), 0);
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_second_run_that_still_fails_says_the_fix_has_not_landed() {
        let root = temp_root("still-red");
        write_crate(&root, BUGGY_LIB, ASSERTING_TEST);
        let gate = ReproGate::new();
        gate.engage(&["src"], true);
        let sandbox = crate::sandbox::Sandbox::disabled();
        let red = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/repro.rs", REPRO_COMMAND, &["src"], None),
            1,
            &sandbox,
            None,
        )
        .await;
        assert!(red.starts_with("red witnessed."), "{red}");
        // Nothing was fixed, so the same command fails again and the gate says so
        // in the same terms both times.
        let again = reproduce_bug(
            &gate,
            &root,
            &tool_args("tests/repro.rs", REPRO_COMMAND, &["src"], None),
            1,
            &sandbox,
            None,
        )
        .await;
        assert!(again.starts_with("not green:"), "{again}");
        assert!(again.contains("has not landed"), "{again}");
        assert_eq!(gate.phase(), Phase::RedWitnessed);
        assert!(gate.evidence().unwrap().green_exit.is_none());
        std::fs::remove_dir_all(&root).ok();
    }
}
