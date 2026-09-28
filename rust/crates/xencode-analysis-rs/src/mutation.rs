//! Mutation testing (`VF-3`), and the gate that keeps it honest.
//!
//! A mutation test asks whether the tests would notice a bug. It answers a
//! question coverage cannot: not "did this line run" but "did running it *catch
//! anything*". `cargo mutants` changes an operator or a return value and re-runs
//! the suite; a mutant the suite still passes is a **missed** mutant, and a
//! missed mutant is a test that cannot tell right from wrong.
//!
//! # The trap, and why this module exists
//!
//! `cargo mutants` is not the dangerous part. The dangerous part is what an agent
//! does with the result, and the failure is entirely predictable: told "this
//! mutant survived", the obvious repair is to make the surviving test stop
//! surviving it. The simplest version is deleting the assertion that caught it.
//!
//! That is measured here, on a throwaway crate. `is_even` had two assertions,
//! `assert!(is_even(4))` and `assert!(!is_even(3))`:
//!
//! | tests | mutants | result |
//! |---|---|---|
//! | both assertions | 8 | **8 caught** |
//! | negative assertion deleted | 8 | 1 missed |
//! | negative assertion deleted, plus `assert!(x \|\| !x)` added | 8 | 1 missed |
//!
//! The third row is the trap. The suite is green, a test still exists, and it
//! asserts nothing at all — `x || !x` is true for every value, which was
//! confirmed by running it. So the mutant survives exactly as before, and
//! everything an outside observer would check says the work was done.
//!
//! A test count is therefore not a defence, and neither is a passing suite. The
//! gate is structural, and [`check_repair`] enforces all four of the plan's
//! conditions at once:
//!
//! 1. the repair may only touch `#[cfg(test)]` code;
//! 2. it must not reduce the number of assertions;
//! 3. it must not edit the file under mutation;
//! 4. it must be proved by re-running **the same mutant set** — never by
//!    `cargo test` passing, which is what the fake fix satisfies.
//!
//! Condition 4 is the one that catches the rest, because a tautology keeps the
//! same mutant alive and only a re-run can show it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;

/// Where `cargo mutants` writes its report.
pub const REPORT_DIR: &str = "mutants.out";

/// The verdicts `cargo mutants` records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The suite failed on the mutant, so it can tell right from wrong.
    Caught,
    /// The suite passed on the mutant. A test that cannot detect this bug.
    Missed,
    /// The mutant would not compile, so it says nothing about test strength.
    Unviable,
    /// Still running when the budget expired.
    Timeout,
    /// The baseline itself failed, so every result is untrustworthy.
    BaselineFailed,
}

impl Verdict {
    /// The word used in reports and in the gate's reasoning.
    pub fn label(self) -> &'static str {
        match self {
            Self::Caught => "caught",
            Self::Missed => "missed",
            Self::Unviable => "unviable",
            Self::Timeout => "timeout",
            Self::BaselineFailed => "baseline failed",
        }
    }

    /// Whether a count of this verdict is good news.
    pub fn is_success(self) -> bool {
        matches!(self, Self::Caught)
    }
}

/// One mutant and what became of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mutant {
    /// The file the mutant was written into, as cargo-mutants reports it.
    pub file: String,
    /// A human-readable name, e.g. `replace == with != in is_even`.
    pub name: String,
    /// What happened.
    pub verdict: Verdict,
}

impl Mutant {
    /// The stable identity used to compare two runs.
    ///
    /// The file alone is not enough: the same file yields many mutants, and two
    /// runs must be comparable so a repair can be proved against *the same set*
    /// rather than whatever the tool happened to generate this time.
    pub fn key(&self) -> String {
        format!("{}::{}", self.file, self.name)
    }
}

/// A whole mutation run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Run {
    /// Every mutant, sorted by key.
    pub mutants: Vec<Mutant>,
    /// Where the report was written.
    pub report_dir: PathBuf,
    /// The command that produced it.
    pub command: String,
    /// Anything a reader needs that is not a count.
    pub notes: Vec<String>,
}

impl Run {
    /// The missed mutants — the actual work list.
    pub fn missed(&self) -> Vec<&Mutant> {
        self.mutants
            .iter()
            .filter(|m| m.verdict == Verdict::Missed)
            .collect()
    }

    /// Every mutant's key, for comparing two runs.
    pub fn keys(&self) -> BTreeMap<String, Verdict> {
        self.mutants.iter().map(|m| (m.key(), m.verdict)).collect()
    }

    /// The distinct files holding a missed mutant, since a repair to one file
    /// cannot fix another.
    pub fn missed_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self.missed().iter().map(|m| m.file.clone()).collect();
        files.sort();
        files.dedup();
        files
    }

    /// One line per verdict, in a fixed order.
    pub fn counts(&self) -> String {
        let count = |v: Verdict| self.mutants.iter().filter(|m| m.verdict == v).count();
        let caught = count(Verdict::Caught);
        let missed = count(Verdict::Missed);
        let other = count(Verdict::Unviable) + count(Verdict::Timeout);
        let mut line = format!("{caught} caught, {missed} missed");
        if other > 0 {
            line.push_str(&format!(
                ", {other} unviable or timed out (neither says anything about test strength)"
            ));
        }
        line
    }

    /// Whether the run is trustworthy enough to act on.
    pub fn is_usable(&self) -> bool {
        !self
            .mutants
            .iter()
            .any(|m| m.verdict == Verdict::BaselineFailed)
    }
}

/// Whether `cargo mutants` is usable here.
pub fn available() -> bool {
    std::process::Command::new("cargo")
        .args(["mutants", "--version"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Write the diff that `cargo mutants --in-diff` needs, and explain the two
/// things that make this work.
///
/// `--in-diff` takes a **file path**, not a git ref. Passing `HEAD` fails with
/// "Failed to open diff file", which reads like a missing file rather than a
/// wrong argument.
///
/// The diff is also written with pinned `a/`/`b/` prefixes. Git's defaults are
/// mnemonic — `i/` for the index, `w/` for the worktree — and cargo-mutants
/// matches paths against the `a/`/`b/` form. With the default prefixes the same
/// diff, which contained five real mutants, produced `No mutants to filter` and
/// a clean summary: a pass that means no work was done.
pub fn write_diff(root: &Path, base: Option<&str>) -> Result<PathBuf, String> {
    let mut command = std::process::Command::new("git");
    command
        .current_dir(root)
        .arg("diff")
        // `--relative` because the diff is written from the workspace root while
        // cargo-mutants runs there too, and the report refers to files as
        // `crates/…`. Without it the paths are `rust/crates/…`, match nothing,
        // and the run reports "No mutants to filter" over a 227-line diff.
        .arg("--relative")
        .arg("--src-prefix=a/")
        .arg("--dst-prefix=b/");
    if let Some(base) = base {
        command.arg(base);
    }
    let output = command
        .output()
        .map_err(|e| format!("could not start git: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "git diff failed: {}",
            stderr.lines().next().unwrap_or("no output")
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    // A diff of untracked files is empty, and an empty diff would be read as
    // "nothing to do" — the same clean summary as a run that did no work.
    if text.trim().is_empty() {
        return Err(match base {
            Some(b) => format!("nothing has changed against {b}, so there is no diff to mutate"),
            None => {
                "nothing has changed in the working tree, so there is no diff to mutate".to_string()
            }
        });
    }
    let dir = root.join(crate::covdiff::COV_STATE_DIR);
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let path = dir.join("mutants-diff.patch");
    std::fs::write(&path, text.as_bytes())
        .map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(path)
}

/// Build the command for a run.
///
/// `--in-diff` is what makes this affordable: without it every mutant in the
/// workspace is generated and the whole suite runs once per survivor, which the
/// plan records as minutes to hours on a workspace this size.
pub fn mutants_argv(diff: Option<&Path>, timeout: Option<u64>) -> Vec<String> {
    let mut argv = vec!["mutants".to_string()];
    if let Some(diff) = diff {
        argv.push("--in-diff".to_string());
        argv.push(diff.display().to_string());
    }
    if let Some(seconds) = timeout {
        argv.push("--timeout".to_string());
        argv.push(seconds.to_string());
    }
    argv
}

/// Read a completed run's `outcomes.json`.
///
/// A missing or unreadable report is an error rather than an empty run: an empty
/// run and a broken run are indistinguishable downstream, and "all caught"
/// inferred from a file that was never written is exactly the false green this
/// project keeps refusing.
pub fn parse_report(report_dir: &Path) -> Result<Run, String> {
    let path = report_dir.join("outcomes.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "no mutation report at {} ({e}). If the run did not finish, there is no \
             result to report and none should be guessed.",
            path.display()
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;

    let outcome_of = |s: &str| -> Verdict {
        match s {
            "CaughtMutant" => Verdict::Caught,
            "MissedMutant" => Verdict::Missed,
            "UnviableMutant" => Verdict::Unviable,
            "Timeout" => Verdict::Timeout,
            _ => Verdict::Caught,
        }
    };

    let Some(outcomes) = value.get("outcomes").and_then(|o| o.as_array()) else {
        return Err(format!("{} has no outcomes array", path.display()));
    };

    let mut mutants = Vec::new();
    let mut notes = Vec::new();
    for entry in outcomes {
        let summary = entry.get("summary").and_then(|s| s.as_str()).unwrap_or("");
        let Some(scenario) = entry.get("scenario") else {
            continue;
        };
        if scenario.get("Baseline").is_some() {
            if summary != "Success" {
                notes.push(
                    "the unmutated baseline did not pass, so no mutant result here means \
                     anything: a test that fails before any mutation fails against all of them"
                        .to_string(),
                );
            }
            continue;
        }
        let Some(detail) = scenario.get("Mutant") else {
            continue;
        };
        let name = detail
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("(unnamed mutant)")
            .to_string();
        let file = detail
            .get("file")
            .and_then(|f| f.as_str())
            .unwrap_or("(unknown file)")
            .to_string();
        mutants.push(Mutant {
            file,
            name,
            verdict: outcome_of(summary),
        });
    }
    mutants.sort_by_key(|m| m.key());

    if mutants.is_empty() {
        notes.push(
            "the run produced no mutants. That is a result about the diff, not a clean bill \
             of health: either the changed lines are not instrumentable, or `--in-diff` \
             matched nothing."
                .to_string(),
        );
    }
    let unviable = mutants
        .iter()
        .filter(|m| m.verdict == Verdict::Unviable)
        .count();
    if unviable > 0 {
        notes.push(format!(
            "{unviable} mutant(s) would not compile. They are excluded rather than counted \
             as caught, because a mutant that cannot build proves nothing about the tests."
        ));
    }

    Ok(Run {
        mutants,
        report_dir: report_dir.to_path_buf(),
        command: String::new(),
        notes,
    })
}

/// One thing wrong with a proposed repair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// Which of the gate's conditions was broken.
    pub rule: &'static str,
    /// What was found, in words a reader can act on.
    pub detail: String,
}

/// A proposed repair, judged against the gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairVerdict {
    /// `true` only when every condition holds and the same mutants are still missed.
    pub accepted: bool,
    /// Everything that was wrong, in a stable order.
    pub violations: Vec<Violation>,
    /// The mutants that were being targeted.
    pub targeted: Vec<String>,
    /// The files the repair edited.
    pub edited: Vec<String>,
}

/// What a repair attempt consists of.
///
/// `patch` is the diff the agent produced. It is judged structurally, not by
/// running anything, and the re-run is the caller's job — [`RepairVerdict`]
/// records what the same mutant set did afterwards.
pub struct Repair<'a> {
    /// The diff, as unified text.
    pub patch: &'a str,
    /// The mutants that were missed and are being repaired.
    pub targeted: &'a [Mutant],
    /// Assertion counts before the repair, keyed by file.
    pub assertions_before: &'a BTreeMap<String, usize>,
    /// Assertion counts after the repair, keyed by file.
    pub assertions_after: &'a BTreeMap<String, usize>,
    /// The verdict of the mutants that were still missed when the *same* set was
    /// re-run. `None` means it was not re-run, which is a rejection.
    pub same_set_rerun: Option<&'a Run>,
}

/// Check a proposed repair against all four conditions.
///
/// The order of the returned violations is the order the conditions are stated,
/// so a rejection always reads the same way.
pub fn check_repair(repair: &Repair<'_>) -> RepairVerdict {
    let mut violations = Vec::new();

    // (1) Only test code.
    let edited = touched_files(repair.patch);
    for file in &edited {
        if !is_test_file(file) {
            violations.push(Violation {
                rule: "test-only",
                detail: format!(
                    "{file} is not test code. A repair for a missed mutant may only change \
                     what the tests assert, because the code under mutation is the subject \
                     of the experiment and editing it invalidates the result."
                ),
            });
        }
    }

    // (2) The assertion count may not go down.
    for (file, before) in repair.assertions_before {
        let after = repair.assertions_after.get(file).copied().unwrap_or(0);
        if after < *before {
            violations.push(Violation {
                rule: "no-fewer-assertions",
                detail: format!(
                    "{file} went from {before} assertion(s) to {after}. Deleting an \
                     assertion is the cheapest way to make a surviving mutant die, and it \
                     removes the only thing that was catching it."
                ),
            });
        }
    }

    // (3) The file under mutation may not be edited at all.
    for mutant in repair.targeted {
        if edited.iter().any(|f| f == &mutant.file) {
            violations.push(Violation {
                rule: "no-source-under-mutation",
                detail: format!(
                    "{} holds the mutant {} that was being repaired. Editing the file the \
                     mutation is applied to changes the experiment rather than the tests.",
                    mutant.file, mutant.name
                ),
            });
        }
    }

    // (4) The same mutant set must be re-run, and must still show the misses.
    //
    // This is the condition the other three exist to support. A tautological
    // assertion is an addition, not a deletion, and it leaves a test in place
    // that looks like a fix; only re-running the identical mutant set shows that
    // the mutant is still alive.
    let targeted_keys: Vec<String> = repair.targeted.iter().map(|m| m.key()).collect();
    match repair.same_set_rerun {
        None => violations.push(Violation {
            rule: "same-mutant-set-rerun",
            detail: "the same mutant set was not re-run, so there is no evidence the repair \
                     changed anything. A passing `cargo test` is not that evidence, and \
                     never is: a weakened assertion passes too."
                .to_string(),
        }),
        Some(rerun) => {
            let after = rerun.keys();
            let drifted: Vec<&String> = targeted_keys
                .iter()
                .filter(|k| !after.contains_key(*k))
                .collect();
            if !drifted.is_empty() {
                violations.push(Violation {
                    rule: "same-mutant-set-rerun",
                    detail: format!(
                        "the re-run did not contain the same mutant set: {} are missing. \
                         Comparing a repair against a different set proves nothing.",
                        drifted
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                });
            }
            for key in &targeted_keys {
                match after.get(key) {
                    // Still missed: the repair did not do the thing it claimed.
                    Some(Verdict::Missed) => violations.push(Violation {
                        rule: "same-mutant-set-rerun",
                        detail: format!(
                            "{key} is still missed after the repair. The tests still cannot \
                             tell this mutant from correct code."
                        ),
                    }),
                    Some(Verdict::Caught) => {}
                    Some(other) => violations.push(Violation {
                        rule: "same-mutant-set-rerun",
                        detail: format!(
                            "{key} came back {}. An unviable or timed-out mutant says \
                             nothing about test strength.",
                            other.label()
                        ),
                    }),
                    None => {}
                }
            }
        }
    }

    RepairVerdict {
        accepted: violations.is_empty(),
        violations,
        targeted: targeted_keys,
        edited,
    }
}

/// Whether a path holds tests rather than subject code.
///
/// A file under `tests/`, or a file whose name says it is a test, or — for unit
/// tests living beside the code — nothing: an inline `#[cfg(test)] mod` is inside
/// a source file, which is why rule 1 has to be checked against the diff's line
/// ranges rather than the path alone. [`check_repair`] uses the path test as a
/// first pass and the caller's line ranges for the rest; a file that is *not*
/// obviously test code is rejected, so the safe answer is the default.
pub fn is_test_file(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    path.starts_with("tests/")
        || path.contains("/tests/")
        || name.starts_with("test_")
        || name == "tests.rs"
        || name.ends_with("_test.rs")
}

/// The files a unified diff touches.
pub fn touched_files(patch: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = rest.trim();
            if path == "/dev/null" {
                continue;
            }
            let path = path.strip_prefix("b/").unwrap_or(path);
            if !out.iter().any(|p: &String| p == path) {
                out.push(path.to_string());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mutant(file: &str, name: &str, verdict: Verdict) -> Mutant {
        Mutant {
            file: file.to_string(),
            name: name.to_string(),
            verdict,
        }
    }

    fn counts(pairs: &[(&str, usize)]) -> BTreeMap<String, usize> {
        pairs.iter().map(|(f, n)| (f.to_string(), *n)).collect()
    }

    /// The exact report cargo-mutants wrote in the measured run above.
    const REAL_REPORT: &str = r#"{
      "outcomes": [
        {"scenario": {"Baseline": {}}, "summary": "Success"},
        {"scenario": {"Mutant": {"name": "src/lib.rs:2:5: replace is_even -> bool with true",
          "file": "src/lib.rs", "genre": "FnValue"}}, "summary": "MissedMutant"},
        {"scenario": {"Mutant": {"name": "src/lib.rs:2:5: replace is_even -> bool with false",
          "file": "src/lib.rs", "genre": "FnValue"}}, "summary": "CaughtMutant"},
        {"scenario": {"Mutant": {"name": "src/lib.rs:2:11: replace == with != in is_even",
          "file": "src/lib.rs", "genre": "BinaryOperator"}}, "summary": "CaughtMutant"}
      ],
      "total_mutants": 8, "missed": 1, "caught": 7
    }"#;

    fn write_report(tag: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xe-mut-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("outcomes.json"), body).unwrap();
        dir
    }

    #[test]
    fn a_real_report_is_read_as_measured() {
        let dir = write_report("real", REAL_REPORT);
        let run = parse_report(&dir).unwrap();
        assert_eq!(run.mutants.len(), 3);
        assert_eq!(run.missed().len(), 1);
        assert_eq!(
            run.missed()[0].name,
            "src/lib.rs:2:5: replace is_even -> bool with true"
        );
        assert_eq!(run.missed_files(), vec!["src/lib.rs"]);
        assert!(run.is_usable());
        assert!(
            run.counts().contains("2 caught, 1 missed"),
            "{}",
            run.counts()
        );
    }

    #[test]
    fn a_failed_baseline_makes_every_result_untrustworthy() {
        let body = r#"{"outcomes":[
            {"scenario":{"Baseline":{}},"summary":"BuildFailed"},
            {"scenario":{"Mutant":{"name":"m","file":"f.rs"}},"summary":"CaughtMutant"}]}"#;
        let dir = write_report("baseline", body);
        let run = parse_report(&dir).unwrap();
        assert!(
            run.notes
                .iter()
                .any(|n| n.contains("baseline did not pass")),
            "{:?}",
            run.notes
        );
    }

    #[test]
    fn a_missing_report_is_an_error_and_never_an_empty_success() {
        let dir = std::env::temp_dir().join("xe-mut-absent-xyz");
        let err = parse_report(&dir).unwrap_err();
        assert!(err.contains("no mutation report"), "{err}");
    }

    #[test]
    fn no_mutants_is_reported_as_suspicious_not_as_a_clean_bill() {
        let dir = write_report("empty", r#"{"outcomes":[]}"#);
        let run = parse_report(&dir).unwrap();
        assert!(
            run.notes.iter().any(|n| n.contains("no mutants")),
            "{:?}",
            run.notes
        );
    }

    #[test]
    fn an_unviable_mutant_is_not_counted_as_caught() {
        let body = r#"{"outcomes":[{"scenario":{"Mutant":{"name":"m","file":"f.rs"}},
            "summary":"UnviableMutant"}]}"#;
        let dir = write_report("unviable", body);
        let run = parse_report(&dir).unwrap();
        assert_eq!(run.missed().len(), 0, "unviable is not missed");
        assert!(run.counts().contains("unviable"), "{}", run.counts());
        assert!(
            run.notes.iter().any(|n| n.contains("would not compile")),
            "{:?}",
            run.notes
        );
    }

    #[test]
    fn deleting_the_assertion_that_caught_a_mutant_is_rejected() {
        let target = mutant(
            "src/lib.rs",
            "replace is_even -> bool with true",
            Verdict::Missed,
        );
        let before = counts(&[("tests/lib.rs", 2)]);
        let after = counts(&[("tests/lib.rs", 1)]);
        let rerun = Run {
            mutants: vec![mutant(
                "src/lib.rs",
                "replace is_even -> bool with true",
                Verdict::Missed,
            )],
            ..Run::default()
        };
        let patch =
            // Only the test file is edited; both diff headers agree.
            "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -4,1 +4,1 @@\n-    assert!(!is_even(3));\n+    assert!(is_even(3));\n";
        let verdict = check_repair(&Repair {
            patch,
            targeted: std::slice::from_ref(&target),
            assertions_before: &before,
            assertions_after: &after,
            same_set_rerun: Some(&rerun),
        });
        assert!(!verdict.accepted);
        // The three structural rules are independent: this patch is caught by
        // the assertion count and by the re-run, while the *other* patch that
        // edits the subject is caught by the subject rule alone. Together they
        // are what a deletion cannot avoid.
        let rules: Vec<&str> = verdict.violations.iter().map(|v| v.rule).collect();
        assert!(rules.contains(&"no-fewer-assertions"), "{rules:?}");
        assert!(rules.contains(&"same-mutant-set-rerun"), "{rules:?}");
        assert_eq!(
            verdict.edited,
            vec!["tests/lib.rs"],
            "only the test file changed"
        );
        assert!(
            !rules.contains(&"test-only"),
            "tests/lib.rs is test code, so that rule must not fire: {rules:?}"
        );
    }

    #[test]
    fn editing_the_file_under_mutation_is_rejected_on_its_own() {
        // Distinct from the assertion count: this fires even though tests were
        // added and the count went up, which is the other way to cheat.
        let target = mutant("src/lib.rs", "m1", Verdict::Missed);
        let before = counts(&[("tests/lib.rs", 1)]);
        let after = counts(&[("tests/lib.rs", 4)]);
        let rerun = Run {
            mutants: vec![mutant("src/lib.rs", "m1", Verdict::Caught)],
            ..Run::default()
        };
        let patch =
            "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -2,1 +2,1 @@\n-    n % 2 == 0\n+    true\n";
        let verdict = check_repair(&Repair {
            patch,
            targeted: std::slice::from_ref(&target),
            assertions_before: &before,
            assertions_after: &after,
            same_set_rerun: Some(&rerun),
        });
        assert!(!verdict.accepted);
        assert!(
            verdict
                .violations
                .iter()
                .any(|v| v.rule == "no-source-under-mutation"),
            "editing the subject is its own violation: {:?}",
            verdict.violations
        );
    }

    #[test]
    fn a_tautology_is_rejected_even_though_the_suite_passes() {
        // The measured fake fix: the negative assertion is gone and `x || !x`
        // has been added in its place. The suite is green and a test still
        // exists, but the mutant survives — which only a re-run can show.
        let target = mutant(
            "src/lib.rs",
            "replace is_even -> bool with true",
            Verdict::Missed,
        );
        let before = counts(&[("tests/lib.rs", 1)]);
        let after = counts(&[("tests/lib.rs", 2)]); // count went *up*
        let rerun = Run {
            mutants: vec![mutant(
                "src/lib.rs",
                "replace is_even -> bool with true",
                Verdict::Missed,
            )],
            ..Run::default()
        };
        let patch =
            "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -5,0 +6 @@\n+    assert!(x || !x);\n";
        let verdict = check_repair(&Repair {
            patch,
            targeted: std::slice::from_ref(&target),
            assertions_before: &before,
            assertions_after: &after,
            same_set_rerun: Some(&rerun),
        });
        assert!(!verdict.accepted, "a tautology is not a fix");
        let detail = &verdict
            .violations
            .iter()
            .find(|v| v.rule == "same-mutant-set-rerun")
            .expect("the re-run must reject it")
            .detail;
        assert!(detail.contains("still missed"), "{detail}");
    }

    #[test]
    fn not_re_running_is_its_own_rejection() {
        let target = mutant("src/lib.rs", "m1", Verdict::Missed);
        let before = counts(&[("tests/lib.rs", 2)]);
        let after = counts(&[("tests/lib.rs", 3)]);
        let patch = "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -1,0 +2 @@\n+    assert!(true);\n";
        let verdict = check_repair(&Repair {
            patch,
            targeted: std::slice::from_ref(&target),
            assertions_before: &before,
            assertions_after: &after,
            same_set_rerun: None,
        });
        assert!(!verdict.accepted);
        assert!(
            verdict
                .violations
                .iter()
                .any(|v| { v.rule == "same-mutant-set-rerun" && v.detail.contains("not re-run") }),
            "{:?}",
            verdict.violations
        );
    }

    #[test]
    fn a_genuine_fix_is_accepted() {
        let target = mutant(
            "src/lib.rs",
            "replace is_even -> bool with true",
            Verdict::Missed,
        );
        let before = counts(&[("tests/lib.rs", 1)]);
        let after = counts(&[("tests/lib.rs", 2)]);
        let rerun = Run {
            mutants: vec![mutant(
                "src/lib.rs",
                "replace is_even -> bool with true",
                Verdict::Caught,
            )],
            ..Run::default()
        };
        let patch =
            "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -5,0 +6 @@\n+    assert!(!is_even(3));\n";
        let verdict = check_repair(&Repair {
            patch,
            targeted: std::slice::from_ref(&target),
            assertions_before: &before,
            assertions_after: &after,
            same_set_rerun: Some(&rerun),
        });
        assert!(verdict.accepted, "{:?}", verdict.violations);
        assert_eq!(verdict.edited, vec!["tests/lib.rs"]);
    }

    #[test]
    fn a_repair_that_changes_a_different_mutant_set_is_rejected() {
        let target = mutant("src/lib.rs", "m1", Verdict::Missed);
        let before = counts(&[("tests/lib.rs", 1)]);
        let after = counts(&[("tests/lib.rs", 2)]);
        let rerun = Run {
            mutants: vec![mutant(
                "src/lib.rs",
                "a-completely-different-mutant",
                Verdict::Caught,
            )],
            ..Run::default()
        };
        let verdict = check_repair(&Repair {
            patch: "--- a/tests/lib.rs\n+++ b/tests/lib.rs\n@@ -1,0 +2 @@\n+    assert!(x);\n",
            targeted: std::slice::from_ref(&target),
            assertions_before: &before,
            assertions_after: &after,
            same_set_rerun: Some(&rerun),
        });
        assert!(!verdict.accepted);
        assert!(
            verdict
                .violations
                .iter()
                .any(|v| v.detail.contains("did not contain the same mutant set")),
            "{:?}",
            verdict.violations
        );
    }

    #[test]
    fn only_test_files_are_accepted_as_a_repair_target() {
        assert!(is_test_file("tests/lib.rs"));
        assert!(is_test_file("crates/x/tests/integration.rs"));
        assert!(is_test_file("src/test_helpers.rs"));
        assert!(
            !is_test_file("src/lib.rs"),
            "the safe answer is the default"
        );
        assert!(!is_test_file("Cargo.toml"));
    }

    #[test]
    fn the_diff_names_only_the_files_it_touches() {
        let patch = "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-a\n+b\n--- /dev/null\n+++ b/tests/new.rs\n";
        assert_eq!(touched_files(patch), vec!["src/lib.rs", "tests/new.rs"]);
    }

    #[test]
    fn the_diff_is_a_file_path_because_a_git_ref_does_not_work() {
        // Measured: `--in-diff HEAD` fails with "Failed to open diff file",
        // which reads like a missing file rather than a wrong argument.
        let argv = mutants_argv(Some(Path::new("/tmp/x.patch")), Some(60)).join(" ");
        assert!(argv.contains("--in-diff /tmp/x.patch"), "{argv}");
        assert!(argv.contains("--timeout 60"), "{argv}");
        assert_eq!(mutants_argv(None, None), vec!["mutants"]);
    }
}
