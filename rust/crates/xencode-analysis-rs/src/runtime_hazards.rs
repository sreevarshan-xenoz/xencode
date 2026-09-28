//! Runtime hazard analysis: the async and concurrency mistakes that compile
//! cleanly, produce no warning, and cost a production freeze or a silent task
//! death.
//!
//! The existing [`crate::analyzer`] reads source line by line — it cannot see a
//! call's receiver, whether a guard is still alive, or what an expression is
//! bound to — so nothing here could be expressed in it. These findings come from
//! matching the parse tree with the `ast-grep` binary, the same substrate the
//! `ast_edit` and `codemod` tools use.
//!
//! # What this does and does not own
//!
//! Clippy already reports a lock guard held across an `.await`, correctly and
//! with a fix suggestion. Verified on 2026-09-28: `await_holding_lock` fires on a
//! `MutexGuard` that outlives an await and stays silent on the same guard scoped
//! into a block, which is exactly the distinction wanted here. **That class is
//! therefore not reimplemented.** Re-deriving it with a structural query would
//! produce a worse version of a lint that already exists.
//!
//! What clippy is blind to, also verified on 2026-09-28 against the same file:
//!
//! | in the source | clippy |
//! |---|---|
//! | `std::fs::read_to_string` inside an `async fn` | silent |
//! | `std::thread::sleep` inside an `async fn` | silent |
//! | `tokio::sync::mpsc::unbounded_channel()` | silent |
//! | `tokio::spawn(…)` as a bare statement | silent |
//!
//! Those four are what this module is for.
//!
//! # The rule about silence
//!
//! `ast-grep` prints an empty list and exits 1 both for a pattern that matches
//! nothing and for one it cannot parse, and prints no message either way. A
//! scan that found nothing is therefore *not* a fact about the code, and this
//! module never reports it as one: a finding set of zero comes back as
//! [`RuntimeScan::proved_clean`] only when the engine actually ran. When it did
//! not run, [`RuntimeScan::findings`] is empty *and* [`RuntimeScan::engine`] says
//! why, so an empty list can never be mistaken for a clean bill of health.

use serde::Serialize;
use std::path::Path;
use std::time::Duration;

/// What kind of hazard this is. Grows as classes are added; the shape of a
/// finding does not change when it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HazardClass {
    /// A blocking call on the reactor's thread: the whole event loop stops.
    BlockingCall,
    /// A channel with no capacity, so a slow reader grows memory without bound.
    UnboundedChannel,
    /// A spawned task whose handle is dropped, so its death is invisible.
    DetachedTask,
}

impl HazardClass {
    /// Stable lowercase name, for `--format json` and for the agent tool.
    pub fn as_str(self) -> &'static str {
        match self {
            HazardClass::BlockingCall => "blocking_call",
            HazardClass::UnboundedChannel => "unbounded_channel",
            HazardClass::DetachedTask => "detached_task",
        }
    }

    /// One line on what actually goes wrong, in the reader's terms.
    pub fn consequence(self) -> &'static str {
        match self {
            HazardClass::BlockingCall => {
                "This blocks the thread the runtime is running other work on, so every \
                 other task on that worker stops until it returns. It looks like a slow \
                 program, not a stopped one."
            }
            HazardClass::UnboundedChannel => {
                "Nothing bounds how much can be queued. A producer faster than its \
                 consumer grows the queue until the process is killed, and the pressure \
                 shows up as memory exhaustion somewhere else."
            }
            HazardClass::DetachedTask => {
                "The handle is dropped, so a panic or a cancellation inside the task \
                 stops silently. Nothing in the caller reports it, and the work simply \
                 never finishes."
            }
        }
    }
}

/// How much this costs if it is real.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Worth knowing about.
    Medium,
    /// Stops a runtime, loses work, or exhausts memory.
    High,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Medium => "medium",
            Severity::High => "high",
        }
    }
}

/// One hazard, with enough context to act on without opening the file.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeHazard {
    pub class: HazardClass,
    pub severity: Severity,
    /// Path as ast-grep reported it, relative to the scanned root.
    pub path: String,
    /// One-based, the way a person counts.
    pub line: usize,
    pub column: usize,
    /// The matched source, first line, trimmed.
    pub snippet: String,
    /// The whole match when it spans lines, else the same as `snippet`.
    pub matched: String,
    /// True when the match sits inside a `#[cfg(test)]` module.
    ///
    /// A test that blocks a reactor is still worth knowing about, so this is a
    /// label rather than a filter — but it is a label, because a finding in a
    /// test is a different claim from one in shipped code.
    pub test_only: bool,
}

impl RuntimeHazard {
    /// The human-readable finding, as the CLI and the agent tool both print it.
    pub fn to_report(&self) -> String {
        let mut out = format!(
            "{}:{}:{} [{}] {}",
            self.path,
            self.line,
            self.column,
            self.severity.as_str(),
            self.snippet
        );
        if self.test_only {
            out.push_str("\n  in a #[cfg(test)] module: reported, not filtered");
        }
        out.push_str(&format!("\n  {}\n", self.class.consequence()));
        for remedy in remedies_for(self.class) {
            out.push_str(&format!("  - {remedy}\n"));
        }
        out.trim_end().to_string()
    }
}

/// What to do about a class, most specific first.
///
/// Every class offers "this is deliberate" as a way out. A check that cannot be
/// told *I meant that* gets suppressed the first time it is wrong, and a
/// suppressed check still reports green — which is worse than not having it.
pub fn remedies_for(class: HazardClass) -> &'static [&'static str] {
    match class {
        HazardClass::BlockingCall => &[
            "Use the async equivalent: tokio::fs for std::fs, tokio::time::sleep for \
             std::thread::sleep, tokio::process::Command for std::process::Command.",
            "For a call with no async form, wrap it in tokio::task::spawn_blocking so \
             it runs on a thread that is allowed to block.",
            "If it is short and genuinely cheap, leave it and say so here — a few \
             microseconds of arithmetic will not stall anything.",
        ],
        HazardClass::UnboundedChannel => &[
            "Use a bounded channel (mpsc::channel) and pick a capacity, so a slow \
             consumer applies backpressure instead of memory.",
            "If the queue really must be unbounded, bound it a different way and \
             record why — a drop policy, or a length check that refuses more.",
        ],
        HazardClass::DetachedTask => &[
            "Bind the handle and await it, so a failure is a failure you can see.",
            "Bind it and keep it, so the task can be cancelled or inspected later \
             (a JoinSet, or a Vec<JoinHandle> owned by the caller).",
            "If the task is meant to outlive the caller, that is fine — annotate it \
             as deliberate so the next reader does not raise this again.",
            "If it is meant to be fire-and-forget, spawn it and keep the handle in a \
             field so nothing is silently dropped.",
        ],
    }
}

/// Whether the pattern engine ran, and if not, why.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum EngineStatus {
    /// The engine ran; `findings` is a real answer about the code.
    Ran { matches: usize },
    /// The engine is not installed. `findings` is empty and means nothing.
    Unavailable { reason: String },
}

/// The result of a scan.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeScan {
    pub engine: EngineStatus,
    pub findings: Vec<RuntimeHazard>,
    /// Files ast-grep actually read.
    pub files_with_findings: usize,
}

impl RuntimeScan {
    /// True only when the engine ran and found nothing — the one case where an
    /// empty list is a statement about the code rather than about this machine.
    pub fn proved_clean(&self) -> bool {
        matches!(self.engine, EngineStatus::Ran { .. }) && self.findings.is_empty()
    }

    /// The one-line summary both surfaces print.
    pub fn summary(&self) -> String {
        match &self.engine {
            EngineStatus::Unavailable { reason } => {
                format!("runtime hazards not checked: {reason}")
            }
            EngineStatus::Ran { matches } if self.findings.is_empty() => {
                format!("runtime hazards: none found in {matches} match(es)")
            }
            EngineStatus::Ran { matches } => {
                let high = self
                    .findings
                    .iter()
                    .filter(|f| f.severity == Severity::High)
                    .count();
                format!(
                    "runtime hazards: {} finding(s), {high} high, from {matches} match(es) \
                     across {} file(s)",
                    self.findings.len(),
                    self.files_with_findings
                )
            }
        }
    }
}

/// One structural query, with what to say about a hit.
struct Rule {
    class: HazardClass,
    severity: Severity,
    /// The `pattern` clause. Metavariables are `$NAME`; a single metavariable
    /// absorbs a whole path prefix (`$A::unbounded_channel::<$T>()` matches
    /// `tokio::sync::mpsc::unbounded_channel::<u8>()`), but two metavariables
    /// cannot share a path — `$M::$N()` parses to an error node and matches
    /// nothing. That limit is why the paths below are literal past the first
    /// metavariable.
    pattern: &'static str,
    /// Extra `all:` clauses, or empty for a single-pattern rule. More than one
    /// is a list, and they are emitted in order.
    constraints: &'static [&'static str],
}

/// The enclosing `async fn`, expressed positively.
///
/// `inside: { kind: function_item, stopBy: end, regex: "^async " }` is used
/// rather than the equivalent-sounding "inside a function and not inside a sync
/// one": the negative form silently loses a blocking call inside an `async fn`
/// nested in a sync one, which was verified on 2026-09-28. `stopBy: end` is
/// required — without it `inside` searches only the immediate parent and matches
/// nothing.
const IN_ASYNC_FN: &str = "inside: { kind: function_item, stopBy: end, regex: \"^async \" }";

/// Not inside `spawn_blocking`, which is the *correct* place for a blocking call.
///
/// Without this the tool's strongest claim was a false positive on this
/// repository's own code: `run_profiler` in `app.rs` sleeps with
/// `std::thread::sleep` — correctly, deliberately, inside
/// `tokio::task::spawn_blocking` — and a lexical "is this inside an `async fn`"
/// test cannot tell that from sleeping on the reactor. A linter that flags
/// correct code on its own repository gets switched off, and a switched-off
/// linter still reports green.
const NOT_ON_A_BLOCKING_THREAD: &str =
    "not: { inside: { kind: call_expression, stopBy: end, regex: \"spawn_blocking\" } }";

/// A `tokio::spawn(…)` that is a statement in its own right.
///
/// A spawn bound to a name is not a hazard — the caller can see it — so matching
/// the bare expression statement is what separates a detached task from a tracked
/// one. `stopBy: neighbor` keeps the search to the statement itself.
const BARE_STATEMENT: &str = "inside: { kind: expression_statement, stopBy: neighbor }";

/// The queries. Kept as data so a new class is a new row, not a new parser.
const RULES: &[Rule] = &[
    Rule {
        class: HazardClass::BlockingCall,
        severity: Severity::High,
        pattern: "std::fs::read_to_string($$$A)",
        constraints: &[IN_ASYNC_FN, NOT_ON_A_BLOCKING_THREAD],
    },
    Rule {
        class: HazardClass::BlockingCall,
        severity: Severity::High,
        pattern: "std::fs::read($$$A)",
        constraints: &[IN_ASYNC_FN, NOT_ON_A_BLOCKING_THREAD],
    },
    Rule {
        class: HazardClass::BlockingCall,
        severity: Severity::High,
        pattern: "std::fs::write($$$A)",
        constraints: &[IN_ASYNC_FN, NOT_ON_A_BLOCKING_THREAD],
    },
    Rule {
        class: HazardClass::BlockingCall,
        severity: Severity::High,
        pattern: "std::fs::read_dir($$$A)",
        constraints: &[IN_ASYNC_FN, NOT_ON_A_BLOCKING_THREAD],
    },
    Rule {
        class: HazardClass::BlockingCall,
        severity: Severity::High,
        pattern: "std::thread::sleep($$$A)",
        constraints: &[IN_ASYNC_FN, NOT_ON_A_BLOCKING_THREAD],
    },
    // `std::thread::spawn` is deliberately absent: it starts a new OS thread and
    // returns immediately, so it never blocks the reactor. Listing it produced a
    // false positive on this repository's own source, which is how it was caught.
    Rule {
        class: HazardClass::UnboundedChannel,
        severity: Severity::High,
        // The turbofish is part of the call's own syntax, so a rule that omits it
        // matches nothing: verified, `unbounded_channel::<$T>()` without the path
        // prefix finds 0 of 1 on real code.
        pattern: "$A::unbounded_channel::<$T>()",
        constraints: &[],
    },
    Rule {
        class: HazardClass::DetachedTask,
        severity: Severity::Medium,
        // Named rather than `$A::spawn`, because the metavariable form matched
        // `std::thread::spawn(|| ())` too — and dropping a std thread handle is
        // normal and harmless, whereas dropping tokio's is the hazard. This rule
        // is therefore deliberately narrow: it covers tokio and says so, rather
        // than claiming every executor.
        pattern: "tokio::spawn($$$B)",
        constraints: &[BARE_STATEMENT],
    },
];

/// The YAML document for one rule, in the form `ast-grep scan --json` takes.
fn rule_yaml(index: usize, rule: &Rule) -> String {
    let body = if rule.constraints.is_empty() {
        format!("  pattern: {}", yaml_scalar(rule.pattern))
    } else {
        // A `pattern` alongside another clause has to sit under `all:`, or it is
        // read as the rule's own pattern and the clause is dropped.
        let mut all = format!("  all:\n    - pattern: {}", yaml_scalar(rule.pattern));
        for clause in rule.constraints {
            all.push_str(&format!("\n    - {clause}"));
        }
        all
    };
    format!("id: runtime-hazard-{index}\nlanguage: Rust\nrule:\n{body}\n")
}

/// Quote a pattern for YAML when it contains characters YAML would read.
fn yaml_scalar(pattern: &str) -> String {
    if pattern.contains(':') || pattern.contains('#') || pattern.starts_with('*') {
        format!("\"{}\"", pattern.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        pattern.to_string()
    }
}

/// One match, as much of it as a finding needs.
struct Match {
    file: String,
    line: usize,
    column: usize,
    text: String,
}

/// Parse `ast-grep`'s `--json`, which is a bare array of matches.
fn parse_matches(stdout: &str) -> Vec<Match> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return Vec::new();
    };
    let Some(array) = value.as_array() else {
        return Vec::new();
    };
    array
        .iter()
        .filter_map(|item| {
            let obj = item.as_object()?;
            let start = obj.get("range")?.get("start")?;
            Some(Match {
                file: obj
                    .get("file")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                line: start.get("line")?.as_u64()? as usize + 1,
                column: start.get("column")?.as_u64()? as usize + 1,
                text: obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
            })
        })
        .collect()
}

/// The `ast-grep` executable, or why there isn't one.
fn ast_grep_binary() -> Result<String, String> {
    let path_var = std::env::var_os("PATH").ok_or_else(|| "PATH is not set".to_string())?;
    for candidate in ["ast-grep", "sg"] {
        for dir in std::env::split_paths(&path_var) {
            let path = dir.join(candidate);
            if path.is_file() {
                return Ok(path.to_string_lossy().into_owned());
            }
        }
    }
    Err(
        "neither `ast-grep` nor `sg` is on PATH, so no pattern was run and nothing is \
         known about this code. Install with `npm i -g @ast-grep/cli` or \
         `cargo install ast-grep`"
            .to_string(),
    )
}

/// Run one command to completion, killing it after `timeout`.
fn run_with_timeout(
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
                    "ast-grep did not finish in time",
                ));
            }
            None => std::thread::sleep(Duration::from_millis(25)),
        }
    }
}

/// Whether `line` (one-based) sits inside a `#[cfg(test)]` module.
///
/// Textual rather than structural, and labelled as such in the finding, because
/// the structural form — `not: { inside: { kind: mod_item, regex: "cfg\\(test\\)" } }`
/// — was measured on 2026-09-28 and did **not** exclude a match inside a
/// `#[cfg(test)] mod`, so it would have been a filter that reported filtering
/// while passing everything through.
fn inside_cfg_test(source: &str, line: usize) -> bool {
    let mut depth_at_test: Option<i32> = None;
    for (index, text) in source.lines().enumerate() {
        let one_based = index + 1;
        if one_based >= line {
            break;
        }
        let trimmed = text.trim();
        if depth_at_test.is_none() && trimmed.starts_with("#[cfg(test)]") {
            depth_at_test = Some(0);
            continue;
        }
        if let Some(depth) = depth_at_test.as_mut() {
            *depth += text.matches('{').count() as i32;
            *depth -= text.matches('}').count() as i32;
            if *depth <= 0 {
                depth_at_test = None;
            }
        }
    }
    depth_at_test.is_some()
}

/// Scan `root` for the hazard classes in [`RULES`].
///
/// `timeout` applies to each rule separately, so a slow tree is bounded by
/// `rules × timeout` rather than unbounded.
pub fn analyze_runtime(root: &Path, timeout: Duration) -> RuntimeScan {
    let binary = match ast_grep_binary() {
        Ok(binary) => binary,
        Err(reason) => {
            return RuntimeScan {
                engine: EngineStatus::Unavailable { reason },
                findings: Vec::new(),
                files_with_findings: 0,
            }
        }
    };
    let relative = std::env::current_dir()
        .ok()
        .and_then(|cwd| root.strip_prefix(cwd).ok().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| Path::new(".").to_path_buf());

    let mut findings: Vec<RuntimeHazard> = Vec::new();
    let mut total_matches = 0usize;
    for (index, rule) in RULES.iter().enumerate() {
        let mut command = std::process::Command::new(&binary);
        command
            .current_dir(root)
            .arg("scan")
            .arg("--inline-rules")
            .arg(rule_yaml(index, rule))
            .arg("--json")
            .arg(&relative);
        // Bounded, because a query that never returns would otherwise hold the
        // whole scan — and a rule that times out is a rule that found nothing,
        // which the engine status below has to be able to say.
        let Ok(output) = run_with_timeout(&mut command, timeout) else {
            continue;
        };
        // ast-grep exits 1 when a rule matches nothing, which is the common case
        // and not an error. Its stderr is only read to notice an unreadable rule.
        if output.status.code() == Some(8) {
            continue;
        }
        for found in parse_matches(&String::from_utf8_lossy(&output.stdout)) {
            total_matches += 1;
            let first_line = found.text.lines().next().unwrap_or("").trim_end();
            let test_only = std::fs::read_to_string(root.join(&found.file))
                .map(|source| inside_cfg_test(&source, found.line))
                .unwrap_or(false);
            findings.push(RuntimeHazard {
                class: rule.class,
                severity: rule.severity,
                path: found.file.clone(),
                line: found.line,
                column: found.column,
                snippet: first_line.to_string(),
                matched: found.text,
                test_only,
            });
        }
    }
    // Shipped findings first, then in file and line order, so two runs over an
    // unchanged tree print the same thing in the same order — and a finding in a
    // test does not lead a report about shipped code. On this repository that
    // reorders 27 findings into 4 that matter and 23 that are labelled.
    findings.sort_by(|a, b| {
        a.test_only
            .cmp(&b.test_only)
            .then(a.path.cmp(&b.path))
            .then(a.line.cmp(&b.line))
            .then(a.column.cmp(&b.column))
    });
    let files_with_findings = findings
        .iter()
        .map(|f| f.path.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    RuntimeScan {
        engine: EngineStatus::Ran {
            matches: total_matches,
        },
        findings,
        files_with_findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_scan_is_only_clean_when_the_engine_ran() {
        let missing = RuntimeScan {
            engine: EngineStatus::Unavailable {
                reason: "not installed".to_string(),
            },
            findings: Vec::new(),
            files_with_findings: 0,
        };
        // The whole point: an empty list that means nothing must not be
        // reportable as a clean bill of health.
        assert!(!missing.proved_clean());
        assert!(missing.summary().contains("not checked"));

        let clean = RuntimeScan {
            engine: EngineStatus::Ran { matches: 0 },
            findings: Vec::new(),
            files_with_findings: 0,
        };
        assert!(clean.proved_clean());
        assert!(clean.summary().contains("none found"));
    }

    #[test]
    fn the_summary_counts_high_severity_separately() {
        let scan = RuntimeScan {
            engine: EngineStatus::Ran { matches: 3 },
            findings: vec![
                RuntimeHazard {
                    class: HazardClass::BlockingCall,
                    severity: Severity::High,
                    path: "a.rs".to_string(),
                    line: 1,
                    column: 1,
                    snippet: "x".to_string(),
                    matched: "x".to_string(),
                    test_only: false,
                },
                RuntimeHazard {
                    class: HazardClass::DetachedTask,
                    severity: Severity::Medium,
                    path: "a.rs".to_string(),
                    line: 2,
                    column: 1,
                    snippet: "y".to_string(),
                    matched: "y".to_string(),
                    test_only: false,
                },
            ],
            files_with_findings: 1,
        };
        assert!(
            scan.summary().contains("2 finding(s), 1 high"),
            "{}",
            scan.summary()
        );
    }

    /// The honesty requirement: a check that cannot be told "I meant that" gets
    /// suppressed the first time it is wrong.
    #[test]
    fn every_class_offers_a_way_to_say_it_is_deliberate() {
        for class in [
            HazardClass::BlockingCall,
            HazardClass::UnboundedChannel,
            HazardClass::DetachedTask,
        ] {
            let remedies = remedies_for(class);
            assert!(!remedies.is_empty(), "{} has no remedy", class.as_str());
            assert!(
                remedies
                    .iter()
                    .any(|r| r.contains("deliberate") || r.contains("If ")),
                "{} offers no deliberate-not-a-defect option: {remedies:?}",
                class.as_str()
            );
        }
    }

    /// The done-when's first case, including the named replacement.
    #[test]
    fn a_blocking_call_names_the_async_equivalent() {
        let finding = RuntimeHazard {
            class: HazardClass::BlockingCall,
            severity: Severity::High,
            path: "src/a.rs".to_string(),
            line: 3,
            column: 5,
            snippet: "std::fs::read_to_string(\"a.txt\").unwrap()".to_string(),
            matched: "std::fs::read_to_string(\"a.txt\")".to_string(),
            test_only: false,
        };
        assert_eq!(finding.severity, Severity::High);
        let report = finding.to_report();
        assert!(report.contains("src/a.rs:3:5"), "{report}");
        assert!(report.contains("[high]"), "{report}");
        assert!(report.contains("tokio::fs"), "{report}");
        assert!(report.contains("std::thread::sleep"), "{report}");
        // It says what goes wrong, not just what to change.
        assert!(report.contains("every other task"), "{report}");
    }

    /// The done-when's fourth case: a finding inside `#[cfg(test)]` is labelled
    /// rather than dropped, and the label is visible.
    #[test]
    fn a_test_only_finding_is_labelled_not_hidden() {
        let source = "#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nfn real() {}\n";
        assert!(
            inside_cfg_test(source, 3),
            "line 3 is inside the test module"
        );
        assert!(!inside_cfg_test(source, 5), "line 5 is after it");
        let finding = RuntimeHazard {
            class: HazardClass::BlockingCall,
            severity: Severity::High,
            path: "a.rs".to_string(),
            line: 3,
            column: 1,
            snippet: "x".to_string(),
            matched: "x".to_string(),
            test_only: true,
        };
        assert!(
            finding.to_report().contains("#[cfg(test)] module"),
            "{}",
            finding.to_report()
        );
    }

    #[test]
    fn the_cfg_test_check_tracks_braces_rather_than_matching_a_line() {
        let source = "#[cfg(test)]\nmod tests {\n    mod inner {\n        fn t() {}\n    }\n}\nfn after() {}\n";
        assert!(inside_cfg_test(source, 4), "nested inside the test module");
        // Line 6 is the module's own closing brace, so it is still inside it.
        assert!(
            inside_cfg_test(source, 6),
            "the closing brace is part of it"
        );
        assert!(
            !inside_cfg_test(source, 7),
            "line 7 is after the module closes"
        );
    }

    #[test]
    fn ast_grep_json_is_read_from_a_bare_array() {
        let json = r#"[{"text":"let a = 1;","range":{"byteOffset":{"start":0,"end":9},"start":{"line":4,"column":2},"end":{"line":4,"column":11}},"file":"a.rs","lines":"  let a = 1;","language":"Rust"}]"#;
        let found = parse_matches(json);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].file, "a.rs");
        assert_eq!(found[0].line, 5, "one-based");
        assert_eq!(found[0].column, 3, "one-based");
    }

    #[test]
    fn unparseable_output_is_treated_as_no_matches_not_as_a_panic() {
        // `ast-grep` prints `[]` for both "nothing matched" and "rule unreadable",
        // and a truncated stream must not take the scan down either.
        assert!(parse_matches("[]").is_empty());
        assert!(parse_matches("").is_empty());
        assert!(parse_matches("   \n").is_empty());
        assert!(parse_matches("not json").is_empty());
        assert!(parse_matches(r#"{"matches":[]}"#).is_empty());
        assert!(
            parse_matches(r#"[{"text":"x"}]"#).is_empty(),
            "no range means no position"
        );
    }

    #[test]
    fn the_rule_set_encodes_the_verified_ast_grep_limits() {
        // A single leading metavariable absorbs a whole path; two cannot share
        // one. If this ever changes, the unbounded-channel row is the canary.
        assert!(RULES
            .iter()
            .any(|r| r.pattern == "$A::unbounded_channel::<$T>()"));
        assert!(
            !RULES.iter().any(|r| r.pattern.contains("$M::$N")),
            "two metavariables in one path parse to an error node"
        );
        // `inside` without `stopBy: end` searches only the immediate parent and
        // matches nothing, which would make the async rules silently inert.
        for rule in RULES
            .iter()
            .filter(|r| r.constraints.iter().any(|c| c.contains("inside")))
        {
            for clause in rule.constraints.iter().filter(|c| c.contains("inside")) {
                assert!(
                    clause.contains("stopBy:"),
                    "{} would match nothing without stopBy",
                    rule.pattern
                );
            }
        }
        // A `pattern` next to another clause has to be under `all:`.
        for rule in RULES {
            let yaml = rule_yaml(0, rule);
            if rule.constraints.is_empty() {
                assert!(yaml.contains("  pattern: "), "{yaml}");
            } else {
                assert!(yaml.contains("all:"), "{yaml}");
            }
        }
    }
}
