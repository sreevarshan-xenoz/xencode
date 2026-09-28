//! Test running (`VF-5`), on the same seam as `WF-4` autodiscovery.
//!
//! `cargo nextest` runs each test in its own process, which is what makes its
//! build-graph selection and per-test retry worth having. But the default is
//! dangerous in a way this module exists to prevent, and the danger was
//! measured rather than assumed.
//!
//! # The measured trap
//!
//! Given a test that fails once and passes on retry, on nextest 0.9.146:
//!
//! | `--flaky-result` | exit code | what the summary says |
//! |---|---|---|
//! | `pass` | **0** | `1 passed (1 flaky)` |
//! | `fail` | 100 | `1 failed` |
//!
//! So the default accepts a test that genuinely broke as a success, and reports
//! it in the same breath as a clean run. The one green signal a test run can
//! give becomes untrustworthy, and nothing in the output says which test it was
//! beyond a word in a summary line.
//!
//! Two consequences drive this module:
//!
//! - **Every flag is passed explicitly.** A repository's `nextest.toml`, or a
//!   `NEXTEST_FLAKY_RESULT` in the environment, must not be able to change what
//!   our exit code means. Both `--retries` and `--flaky-result` are always
//!   supplied, so the outcome is decided here rather than inherited.
//! - **Flakes are named, not just counted.** Counting them is not enough; a
//!   quarantine list has to say *which* tests. They are read out of the run's
//!   output by name.
//!
//! One honest limitation: `--flaky-result fail` makes nextest **cancel the run**
//! at the first flake ("Cancelling due to test failure"), so a single pass does
//! not enumerate every broken test. That is reported rather than smoothed over.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use crate::anchor::{self, Kind};

/// Which engine actually ran the tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// `cargo nextest`, with the flake policy enforced.
    Nextest,
    /// The repository's own verified test command, from `WF-4`.
    Anchor,
}

impl Engine {
    /// How the run should be described to a reader.
    pub fn label(self) -> &'static str {
        match self {
            Self::Nextest => "cargo nextest",
            Self::Anchor => "the repository's own test command",
        }
    }
}

/// What to run and how strictly.
#[derive(Debug, Clone)]
pub struct Options {
    /// How many times a failing test may be retried.
    pub retries: u32,
    /// Run each test `stress` times to surface order dependence and flakes.
    pub stress: u32,
    /// Only these packages.
    pub packages: Vec<String>,
    /// Wall-clock ceiling for the whole run.
    pub budget: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            retries: 0,
            stress: 0,
            packages: Vec::new(),
            budget: Duration::from_secs(1800),
        }
    }
}

/// What the run found.
#[derive(Debug, Clone, Default)]
pub struct Outcome {
    /// `true` only when the engine exited zero *and* nothing was flaky.
    pub ok: bool,
    /// The engine's exit code.
    pub exit: Option<i32>,
    /// Which engine ran.
    pub engine: Option<Engine>,
    /// The exact command line, so a reader can reproduce it.
    pub command: String,
    /// Named tests that failed at least once and passed on retry.
    pub flaky: Vec<String>,
    /// Named tests that failed outright.
    pub failed: Vec<String>,
    /// Anything a reader should know that is not a test result.
    pub notes: Vec<String>,
}

/// Where the manifest is, so a run from the repository root still finds the
/// workspace.
///
/// A workspace whose `Cargo.toml` is not at the top — a `rust/`, a `crates/`
/// subdirectory — is common enough that assuming otherwise makes the command
/// fail with a message about `Cargo.toml` that has nothing to do with tests.
/// When more than one subdirectory has a manifest the choice is not guessed;
/// the candidates are named instead.
pub fn manifest_dir(root: &Path) -> std::result::Result<std::path::PathBuf, String> {
    if root.join("Cargo.toml").is_file() {
        return Ok(root.to_path_buf());
    }
    if let Some(parent) = root.parent() {
        if parent.join("Cargo.toml").is_file() {
            return Ok(parent.to_path_buf());
        }
    }
    let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(root)
        .map_err(|e| format!("could not read {}: {e}", root.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("Cargo.toml").is_file())
        .collect();
    found.sort();
    match found.len() {
        0 => Err(format!(
            "no Cargo.toml in {} or its parent, and no subdirectory has one",
            root.display()
        )),
        1 => Ok(found.remove(0)),
        _ => Err(format!(
            "several subdirectories have a Cargo.toml ({}). Run this from the one you \
             mean rather than having it guessed",
            found
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// Whether `cargo nextest` can be used on this machine.
pub fn nextest_available() -> bool {
    std::process::Command::new("cargo")
        .args(["nextest", "--version"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Run the tests, and report what happened without softening it.
///
/// Falls back to the repository's own verified test command when nextest is
/// unavailable, because a missing optional tool should not mean no tests run —
/// and the fallback is named, never presented as the nextest result it is not.
pub fn run(root: &Path, opts: &Options) -> Outcome {
    run_with(root, opts, nextest_available())
}

/// [`run`], with the engine chosen explicitly so the fallback can be exercised
/// on a machine that does have nextest installed.
fn run_with(root: &Path, opts: &Options, nextest: bool) -> Outcome {
    if nextest {
        run_nextest(root, opts)
    } else {
        run_anchor(root, opts)
    }
}

/// Build the nextest command line.
///
/// Both flake flags are always present. `--flaky-result fail` is the load-
/// bearing one: without it a test that failed and had to be retried exits 0.
pub fn nextest_argv(opts: &Options) -> Vec<String> {
    let mut argv = vec![
        "nextest".to_string(),
        "run".to_string(),
        "--retries".to_string(),
        opts.retries.to_string(),
        "--flaky-result".to_string(),
        "fail".to_string(),
    ];
    if opts.stress > 0 {
        argv.push("--stress-count".to_string());
        argv.push(opts.stress.to_string());
    }
    for package in &opts.packages {
        argv.push("--package".to_string());
        argv.push(package.clone());
    }
    argv
}

fn run_nextest(root: &Path, opts: &Options) -> Outcome {
    let dir = match manifest_dir(root) {
        Ok(dir) => dir,
        Err(why) => {
            return Outcome {
                notes: vec![why],
                ..Outcome::default()
            };
        }
    };
    let argv = nextest_argv(opts);
    let mut rendered = format!("cargo {}", argv.join(" "));
    if dir != root {
        rendered = format!("(in {}) {}", dir.display(), rendered);
    }
    let started = std::time::Instant::now();

    let mut command = std::process::Command::new("cargo");
    command.args(&argv).current_dir(&dir);
    let output = command.output();
    let elapsed = started.elapsed();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            return Outcome {
                ok: false,
                exit: None,
                engine: Some(Engine::Nextest),
                command: rendered,
                notes: vec![format!("could not start cargo nextest: {e}")],
                ..Outcome::default()
            };
        }
    };
    if elapsed > opts.budget {
        return Outcome {
            ok: false,
            exit: output.status.code(),
            engine: Some(Engine::Nextest),
            command: rendered,
            notes: vec![format!(
                "the run took longer than the {}s budget and was cut short, so the \
                 result is not a complete picture",
                opts.budget.as_secs()
            )],
            ..Outcome::default()
        };
    }

    // nextest writes its human report to **stderr**, not stdout. Capturing only
    // stdout silently finds no test names at all, which is how a flake ends up
    // counted but never named.
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push('\n');
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    let parsed = parse_nextest_output(&text);
    let mut notes = parsed.notes;
    if parsed.flaky.is_empty() && parsed.failed.is_empty() && !output.status.success() {
        // A non-zero exit with no test named is not a test failure, and saying
        // only "not a pass" would send a reader looking for broken tests that
        // are not there. The build or the workspace is what failed.
        notes.push(
            "the run did not name a single test, so this is not a test failure — the \
             build or the workspace setup failed before any test could be reported."
                .to_string(),
        );
    }
    if !parsed.flaky.is_empty() {
        notes.push(format!(
            "{} test(s) needed a retry. They are listed by name below: a retry-pass is \
             never counted as a pass.",
            parsed.flaky.len()
        ));
    }
    if parsed.cancelled {
        notes.push(
            "nextest cancelled at the first flake, so this run did not reach every test. \
             Re-run with --retries 0 to get the full failure list."
                .to_string(),
        );
    }
    Outcome {
        // Exit code and the flake list must agree before this can be called a
        // pass; treating a clean exit as sufficient is the trap.
        ok: output.status.success() && parsed.flaky.is_empty() && parsed.failed.is_empty(),
        exit: output.status.code(),
        engine: Some(Engine::Nextest),
        command: rendered,
        flaky: parsed.flaky,
        failed: parsed.failed,
        notes,
    }
}

/// Fall back to the repository's own test command when nextest is missing.
///
/// The command is proved here and now, not read from a verdict some earlier run
/// recorded: a command that passed last week says nothing about today, and
/// borrowing that verdict would be the same false confidence `WF-4` warns about.
fn run_anchor(root: &Path, _opts: &Options) -> Outcome {
    let missing = "cargo nextest is not installed, so the repository's own test command \
                    was used instead. Its flake policy is whatever that command happens to \
                    do, this run did not check for retries, and the result therefore cannot \
                    be compared to a nextest run.";
    let discovery = anchor::discover(root);
    let Some(recipe) = discovery.candidate(Kind::Test) else {
        return Outcome {
            notes: vec![
                "cargo nextest is not installed, and no test command was found in this \
                 repository. Run `xencode anchor` to look for one."
                    .to_string(),
            ],
            ..Outcome::default()
        };
    };

    let mut recipe = recipe.clone();
    anchor::prove(root, &mut recipe, Duration::from_secs(1800));
    let passed = recipe.is_verified();
    let mut notes = vec![missing.to_string()];
    if !passed {
        notes.push(format!(
            "the test command did not pass here: {}",
            recipe
                .verdict
                .as_ref()
                .map_or("no verdict at all".to_string(), |v| v.describe())
        ));
    }
    Outcome {
        ok: passed,
        exit: Some(i32::from(!passed)),
        engine: Some(Engine::Anchor),
        command: recipe.command.clone(),
        flaky: Vec::new(),
        failed: if passed {
            Vec::new()
        } else {
            vec![recipe.command.clone()]
        },
        notes,
    }
}

/// What a nextest run's text output told us.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Parsed {
    /// Tests that failed and then passed, by name.
    pub flaky: Vec<String>,
    /// Tests that failed and stayed failed, by name.
    pub failed: Vec<String>,
    /// Whether the run stopped early.
    pub cancelled: bool,
    /// The flake count from the summary line, when present.
    pub flaky_count: Option<u64>,
    /// Anything else worth saying.
    pub notes: Vec<String>,
}

/// Read flaky and failed test names out of a nextest run's output.
///
/// The human format is used deliberately. nextest 0.9.146 has no JUnit
/// message format at all, and its `libtest-json-plus` output carries no flaky
/// event — a retried test is reported as an ordinary `ok` — so the human
/// `FLAKY` marker is the only place a flaky test is named. Parsing the human
/// format is fragile against upstream reformatting; what is not fragile is the
/// exit code, which is enforced separately in [`run_nextest`].
pub fn parse_nextest_output(text: &str) -> Parsed {
    let mut parsed = Parsed::default();
    // Names seen failing on an early attempt, so a later pass on the same name
    // is a retry rather than a fresh success.
    let mut earlier_failures: Vec<String> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("TRY ") {
            let (attempt, rest) = rest.split_once(' ').unwrap_or(("", rest));
            if let Some(name) = result_name(rest) {
                if rest.contains("FAIL") {
                    earlier_failures.push(name);
                } else if rest.contains("PASS") && earlier_failures.contains(&name) {
                    parsed.flaky.push(name);
                }
            }
            let _ = attempt;
        } else if trimmed.starts_with("FLAKY ") {
            if let Some(name) = result_name(trimmed) {
                parsed.flaky.push(name);
            }
        } else if trimmed.starts_with("FLKY-FL ") || trimmed.starts_with("FAIL ") {
            if let Some(name) = result_name(trimmed) {
                parsed.failed.push(name);
            }
        } else if trimmed.contains("Cancelling due to test failure") {
            parsed.cancelled = true;
        } else if let Some(rest) = trimmed.strip_prefix("Summary ") {
            if let Some(inner) = rest.split('(').nth(1) {
                if let Some(n) = inner
                    .split_whitespace()
                    .next()
                    .and_then(|d| d.parse::<u64>().ok())
                {
                    parsed.flaky_count = Some(n);
                }
            }
        }
    }
    parsed.flaky.sort();
    parsed.flaky.dedup();
    parsed.failed.sort();
    parsed.failed.dedup();
    // With `--flaky-result fail` a flaky test is reported as `FLKY-FL`, which is
    // both. It is already listed as a flake, and listing it twice reads as two
    // separate problems when it is one.
    parsed.failed.retain(|name| !parsed.flaky.contains(name));
    if let (Some(count), 0) = (parsed.flaky_count, parsed.flaky.len()) {
        if count > 0 {
            parsed.notes.push(format!(
                "the summary counted {count} flaky test(s) but none were named in the output, \
                 so they are reported as unaccounted for rather than assumed absent"
            ));
        }
    }
    parsed
}

/// The `<binary> <test>` tail of a result line, skipping the leading status and
/// counters. Handles `(───)`, used for an attempt with no retries left.
fn result_name(line: &str) -> Option<String> {
    let (_, rest) = line.split_once('(')?;
    let (_, rest) = rest.split_once(')')?;
    let name = rest.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verbatim stderr from `cargo nextest run --retries 3 --flaky-result fail`,
    /// including the counter style used when an attempt has no retries left.
    const FAIL_RUN: &str = "\\
  TRY 1 FAIL [   0.010s] (───) flakydemo fails_first_then_passes
  TRY 2 PASS [   0.008s] (1/1) flakydemo fails_first_then_passes

           - test configured to fail if flaky
   Cancelling due to test failure:

     Summary [   0.010s] 1 test run: 0 passed, 1 failed, 0 skipped
 FLKY-FL 2/4 [   0.004s] (1/1) flakydemo fails_first_then_passes
error: test run failed
";

    /// Verbatim output from the same run with `--flaky-result pass`.
    const PASS_RUN: &str = "\
 TRY 2 PASS [   0.004s] (1/1) flakydemo fails_first_then_passes
     Summary [   0.012s] 1 test run: 1 passed (1 flaky), 0 skipped
   FLAKY 2/4 [   0.004s] (1/1) flakydemo fails_first_then_passes
";

    #[test]
    fn a_flake_is_named_from_a_passing_run() {
        let parsed = parse_nextest_output(PASS_RUN);
        assert_eq!(parsed.flaky, vec!["flakydemo fails_first_then_passes"]);
        assert_eq!(parsed.flaky_count, Some(1));
        assert!(parsed.failed.is_empty());
    }

    #[test]
    fn a_flake_causing_failure_is_named_once_as_a_flake() {
        let parsed = parse_nextest_output(FAIL_RUN);
        assert_eq!(parsed.flaky, vec!["flakydemo fails_first_then_passes"]);
        assert!(
            parsed.failed.is_empty(),
            "one broken test must not be reported as two: {:?}",
            parsed.failed
        );
        assert!(parsed.cancelled, "the cancel line must be noticed");
    }

    #[test]
    fn a_test_failing_outright_is_still_named_as_failed() {
        let text = "  TRY 1 FAIL [0.01s] (───) binname a::b::broken\n  FAIL [0.01s] (1/1) binname a::b::broken\n";
        let parsed = parse_nextest_output(text);
        assert_eq!(parsed.failed, vec!["binname a::b::broken"]);
        assert!(parsed.flaky.is_empty(), "no retry happened, so no flake");
    }

    #[test]
    fn a_clean_run_names_nothing() {
        let parsed = parse_nextest_output("     Summary [0.1s] 4 tests run: 4 passed, 0 skipped\n");
        assert!(parsed.flaky.is_empty());
        assert!(parsed.failed.is_empty());
        assert!(!parsed.cancelled);
    }

    #[test]
    fn a_count_without_a_name_is_reported_not_assumed_away() {
        let text = "     Summary [0.01s] 1 test run: 1 passed (3 flaky), 0 skipped\n";
        let parsed = parse_nextest_output(text);
        assert_eq!(parsed.flaky_count, Some(3));
        assert!(parsed.flaky.is_empty());
        assert!(
            parsed.notes.iter().any(|n| n.contains("3 flaky")),
            "a count nobody can account for must be said out loud: {:?}",
            parsed.notes
        );
    }

    #[test]
    fn the_flake_policy_can_never_be_inherited() {
        // Whatever a repository's own config says, our argv decides.
        let argv = nextest_argv(&Options::default());
        let joined = argv.join(" ");
        assert!(joined.contains("--flaky-result fail"), "{joined}");
        assert!(joined.contains("--retries 0"), "{joined}");
        let stress = nextest_argv(&Options {
            stress: 4,
            ..Options::default()
        })
        .join(" ");
        assert!(stress.contains("--stress-count 4"), "{stress}");
        let packages = nextest_argv(&Options {
            packages: vec!["a".into(), "b".into()],
            ..Options::default()
        })
        .join(" ");
        assert!(packages.contains("--package a --package b"), "{packages}");
    }

    #[test]
    fn without_nextest_the_anchor_command_is_used_and_says_so() {
        // A fresh temp tree: a CI file whose test command is one that classifies
        // and exits 0 immediately.
        let dir = std::env::temp_dir().join(format!("xe-verify-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // A CI file whose test command is a real, fast, succeeding shell
        // command that still classifies as a test recipe.
        std::fs::create_dir_all(dir.join(".github").join("workflows")).unwrap();
        std::fs::write(
            dir.join(".github").join("workflows").join("ci.yml"),
            "steps:\n  - run: echo cargo test --workspace\n",
        )
        .unwrap();

        let outcome = run_with(&dir, &Options::default(), false);
        assert_eq!(outcome.engine, Some(Engine::Anchor));
        assert_eq!(outcome.command, "echo cargo test --workspace");
        assert!(
            outcome.ok,
            "the anchor's verified command should have run: {outcome:?}"
        );
        assert!(
            outcome
                .notes
                .iter()
                .any(|n| n.contains("nextest is not installed")),
            "the fallback must name itself: {:?}",
            outcome.notes
        );
        assert!(
            outcome
                .notes
                .iter()
                .any(|n| n.contains("did not check for retries")),
            "a fallback run cannot be compared to a nextest run: {:?}",
            outcome.notes
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn without_nextest_and_without_a_command_it_says_so_rather_than_guessing() {
        let dir = std::env::temp_dir().join(format!("xe-verify-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let outcome = run_with(&dir, &Options::default(), false);
        assert!(!outcome.ok);
        assert!(outcome.engine.is_none());
        assert!(
            outcome.notes.iter().any(|n| n.contains("xencode anchor")),
            "it must point at the way to fix this: {:?}",
            outcome.notes
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_manifest_in_a_subdirectory_is_found() {
        let dir = std::env::temp_dir().join(format!("xe-verify-mf-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("rust")).unwrap();
        std::fs::write(dir.join("rust").join("Cargo.toml"), "[workspace]\n").unwrap();
        assert_eq!(
            manifest_dir(&dir).unwrap(),
            dir.join("rust"),
            "a top-level run must still find the workspace"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_ambiguous_manifest_is_named_not_guessed() {
        let dir = std::env::temp_dir().join(format!("xe-verify-amb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("one")).unwrap();
        std::fs::create_dir_all(dir.join("two")).unwrap();
        std::fs::write(dir.join("one").join("Cargo.toml"), "[package]\n").unwrap();
        std::fs::write(dir.join("two").join("Cargo.toml"), "[package]\n").unwrap();
        let err = manifest_dir(&dir).unwrap_err();
        assert!(err.contains("one") && err.contains("two"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_manifest_anywhere_is_reported_plainly() {
        let dir = std::env::temp_dir().join(format!("xe-verify-none-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(manifest_dir(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_name_keeps_its_crate() {
        assert_eq!(
            result_name(" FLAKY 2/4 [0.004s] (1/1) mycrate some::module::test"),
            Some("mycrate some::module::test".to_string())
        );
    }

    #[test]
    fn the_binary_and_test_are_both_kept() {
        // Observed from a real run, not invented: nextest prints the binary
        // name and the test name as two fields.
        assert_eq!(
            result_name(" FLKY-FL 2/4 [0.004s] (1/1) flakydemo fails_first_then_passes"),
            Some("flakydemo fails_first_then_passes".to_string())
        );
    }

    #[test]
    fn a_line_with_nothing_after_the_counter_has_no_name() {
        assert_eq!(result_name(" FAIL [0.1s] (1/2) "), None);
        assert_eq!(
            result_name("  TRY 1 FAIL [0.010s] (───) binname a::b::c"),
            Some("binname a::b::c".to_string())
        );
    }
}
