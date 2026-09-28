//! Running one agent headless and recording what actually came back.
//!
//! Every field in a [`RunCapture`] is something a process did. Nothing here reads
//! documentation, and nothing here fills a cell it did not earn — see the crate
//! documentation for why that is the whole design.

use crate::roster::{which, AgentSpec, Provenance, ROSTER};
use serde::Serialize;
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
    /// How much of the matrix this run can speak to.
    pub provenance: Provenance,
    /// Why the run did not produce an answer, when it did not.
    pub failure: Option<String>,
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
    /// Agents named by the proposal that are not installed here, with the reason.
    pub absent: Vec<AbsentAgent>,
    /// Anything the run could not answer, stated rather than left blank.
    pub unanswered: Vec<String>,
}

/// An agent this probe cannot say anything about, and why.
#[derive(Debug, Clone, Serialize)]
pub struct AbsentAgent {
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
    const KEYS: &[&str] = &[
        "\"sessionId\"",
        "\"session_id\"",
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

/// Run the probe over the roster, or over `options.only`.
pub fn run_probe(options: &ProbeOptions) -> ProbeReport {
    let selected: Vec<&AgentSpec> = if options.only.is_empty() {
        ROSTER.iter().collect()
    } else {
        ROSTER
            .iter()
            .filter(|a| options.only.iter().any(|n| n == a.name))
            .collect()
    };
    let repeat = options.repeat.max(1);
    let mut all_runs: Vec<RunCapture> = Vec::new();
    let mut captures: Vec<RunCapture> = Vec::new();
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
        absent: crate::roster::NOT_INSTALLED
            .iter()
            .map(|(name, why)| AbsentAgent {
                name: (*name).to_string(),
                why: (*why).to_string(),
            })
            .collect(),
        unanswered,
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
    fn a_redaction_pass_removes_the_shapes_agents_actually_echo() {
        let text = "\
key sk-abcdefghijklmnopqrstuvwxyz012345\n\
ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345\n\
AIzaSyA1234567890abcdefghijklmnopqrstuv\n\
Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVP\n";
        let out = redact(text);
        assert!(!out.contains("sk-abcdefghij"), "{out}");
        assert!(!out.contains("ghp_ABCDEF"), "{out}");
        assert!(!out.contains("AIzaSyA1"), "{out}");
        assert!(!out.contains("eyJhbGciOiJIUzI1NiJ9"), "{out}");
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
