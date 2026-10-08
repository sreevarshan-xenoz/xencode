//! `OR-17` — the veto, end to end through the real binary.
//!
//! Every claim here is about a `xencode` process that actually ran: a branch
//! that reached master or did not, a refusal a worker got back, a record sitting
//! in the audit trail that `xencode audit verify` then walks. The veto state
//! lives in the repository, so each test builds a real git repo. The audit trail
//! lives in `state_dir()`, so `$XCODE_CONFIG_DIR` points at the test's own
//! directory and nothing is written into the person's real records.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

fn xencode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xencode")
}

fn git(repo: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .status()
        .expect("git is on PATH");
    assert!(status.success(), "git {args:?} failed");
}

/// A repo on `main` plus one branch `arm-1` whose last commit was made by
/// `Tester`, and a private place for the audit trail.
struct Fixture {
    #[allow(dead_code)]
    dir: TempDir,
    repo: PathBuf,
    state: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let repo = dir.path().join("repo");
        fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-q"]);
        git(&repo, &["config", "user.email", "tester@example.com"]);
        git(&repo, &["config", "user.name", "Tester"]);
        fs::write(repo.join("README.md"), "# Project\n").unwrap();
        git(&repo, &["add", "README.md"]);
        git(&repo, &["commit", "-qm", "initial"]);
        git(&repo, &["branch", "-M", "main"]);
        git(&repo, &["checkout", "-qb", "arm-1"]);
        fs::write(repo.join("feature.rs"), "fn feature() {}\n").unwrap();
        git(&repo, &["add", "feature.rs"]);
        git(&repo, &["commit", "-qm", "the worker's change"]);
        git(&repo, &["checkout", "-q", "main"]);
        let state = dir.path().join("state");
        fs::create_dir_all(&state).unwrap();
        Self { dir, repo, state }
    }

    fn run(&self, args: &[&str]) -> (bool, String) {
        run_in(&self.repo, &self.state, args)
    }

    /// The land, with the approval name a merge always has to be given.
    fn land(&self) -> (bool, String) {
        self.run(&[
            "merge",
            "land",
            "--branch",
            "arm-1",
            "--base",
            "main",
            "--approved-by",
            "Grace",
        ])
    }

    fn audit_log(&self) -> PathBuf {
        self.state.join("audit.jsonl")
    }
}

fn run_in(repo: &Path, state: &Path, args: &[&str]) -> (bool, String) {
    let output = Command::new(xencode_bin())
        .args(args)
        .current_dir(repo)
        .env("XCODE_CONFIG_DIR", state)
        .output()
        .expect("the xencode binary ran");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (output.status.success(), text)
}

/// Record one veto and hand back what it was called.
fn veto(fixture: &Fixture, reason: &str) -> String {
    let (ok, text) = fixture.run(&[
        "merge",
        "veto",
        "arm-1",
        "--reason",
        reason,
        "--raised-by",
        "Alice",
        "--source",
        "review",
    ]);
    assert!(ok, "`merge veto` failed: {text}");
    assert!(text.contains("veto-0001"), "{text}");
    "veto-0001".to_string()
}

/// The plan is what a reviewer reads, so the block has to be legible there: the
/// reason, and who is allowed to lift it.
#[test]
fn the_plan_reads_as_blocked_and_says_who_may_clear_it() {
    let fixture = Fixture::new();
    veto(&fixture, "the retry swallows the error");

    let (ok, text) = fixture.run(&["merge", "plan", "--branch", "arm-1", "--base", "main"]);
    assert!(ok, "`merge plan` failed: {text}");
    assert!(text.contains("BLOCKED"), "{text}");
    assert!(text.contains("the retry swallows the error"), "{text}");
    assert!(
        text.contains("not Tester, which is the worker this veto blocks"),
        "the plan has to name the party that may not clear it: {text}"
    );
    assert!(text.contains("Open vetoes: 1"), "{text}");

    let (json_ok, json) = fixture.run(&[
        "merge", "plan", "--branch", "arm-1", "--base", "main", "--format", "json",
    ]);
    assert!(json_ok, "{json}");
    let value: serde_json::Value = serde_json::from_str(&json).expect("plan json parses");
    assert_eq!(value["branches"][0]["vetoes"][0]["id"], "veto-0001");
    assert_eq!(value["branches"][0]["eligible"], false);
    assert_eq!(value["all_worker_checks_passed"], false);
}

/// The gate itself: a named human approval is no longer enough on its own.
#[test]
fn a_vetoed_branch_does_not_reach_master() {
    let fixture = Fixture::new();
    veto(&fixture, "the retry swallows the error");

    let (ok, text) = fixture.land();
    assert!(!ok, "the land went through with a veto open: {text}");
    assert!(text.contains("merge refused: 1 open veto"), "{text}");
    assert!(
        !fixture.repo.join("feature.rs").exists(),
        "the worker's file reached main anyway"
    );
    let head = String::from_utf8_lossy(
        &Command::new("git")
            .arg("-C")
            .arg(&fixture.repo)
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
            .unwrap()
            .stdout,
    )
    .trim()
    .to_string();
    assert_eq!(head, "main", "land left the base branch checked out");
}

/// The other half of the done-when: the blocked party gets nothing back that
/// unblocks it, and the record still says open afterwards.
#[test]
fn the_blocked_worker_cannot_lift_its_own_veto() {
    let fixture = Fixture::new();
    veto(&fixture, "the retry swallows the error");

    let (ok, text) = fixture.run(&["merge", "clear-veto", "veto-0001", "--by", "Tester"]);
    assert!(!ok, "the worker cleared its own veto: {text}");
    assert!(text.contains("cannot lift its own block"), "{text}");

    // A policy that does not say what it clears is not a way around it either.
    let (vague_ok, vague) = fixture.run(&[
        "merge",
        "clear-veto",
        "veto-0001",
        "--by",
        "Grace",
        "--policy",
        "small fixes land on their own",
    ]);
    assert!(!vague_ok, "an unnamed policy cleared a veto: {vague}");
    assert!(vague.contains("does not name"), "{vague}");

    let (open_ok, listing) = fixture.run(&["merge", "vetoes", "--open"]);
    assert!(open_ok, "{listing}");
    assert!(listing.contains("veto-0001"), "{listing}");
    assert!(listing.contains("[open]"), "{listing}");

    let (land_ok, land_text) = fixture.land();
    assert!(!land_ok, "it landed after two refused clears: {land_text}");
}

/// A clearance that does happen is a line in the same chained log the server
/// writes to — verifiable by the real `audit verify`, not by a field someone
/// could have edited.
#[test]
fn clearing_by_another_name_lands_it_and_leaves_an_audited_trail() {
    let fixture = Fixture::new();
    veto(&fixture, "the retry swallows the error");

    let (ok, text) = fixture.run(&[
        "merge",
        "clear-veto",
        "veto-0001",
        "--by",
        "Grace",
        "--policy",
        "veto-0001 is cleared: the retry was reworked and the tests re-ran on main",
    ]);
    assert!(ok, "a named reviewer could not clear the veto: {text}");
    assert!(text.contains("under policy"), "{text}");

    let (land_ok, land_text) = fixture.land();
    assert!(land_ok, "the land still failed after a clear: {land_text}");
    assert!(fixture.repo.join("feature.rs").exists());

    let (verify_ok, verified) =
        fixture.run(&["audit", "verify", &fixture.audit_log().to_string_lossy()]);
    assert!(verify_ok, "{verified}");
    assert!(verified.contains("chain intact"), "{verified}");

    let log = fs::read_to_string(fixture.audit_log()).unwrap();
    assert_eq!(
        log.lines().count(),
        2,
        "one for the veto, one for the clear"
    );
    assert!(log.contains("\"action\":\"merge_vetoed\""), "{log}");
    assert!(log.contains("\"action\":\"merge_veto_cleared\""), "{log}");
    let cleared: serde_json::Value =
        serde_json::from_str(log.lines().nth(1).unwrap()).expect("second line parses");
    assert_eq!(cleared["actor"], "Grace");
    assert_eq!(cleared["target"], "arm-1");
    assert_eq!(cleared["seq"], 2);
    assert!(
        cleared["detail"]
            .as_str()
            .unwrap()
            .contains("the retry swallows the error"),
        "the trail says what was blocked, not just that something changed: {cleared}"
    );
}

/// The trail is the point of a clearance, so an audit log that will not take the
/// line has to leave the veto open rather than unblock it quietly.
#[test]
fn a_clear_with_nowhere_to_log_leaves_the_veto_open() {
    let fixture = Fixture::new();
    veto(&fixture, "the retry swallows the error");

    // Where `audit.jsonl` should be, put a directory: every open for appending
    // fails from here, and no part of the veto code gets to pretend otherwise.
    fs::remove_file(fixture.audit_log()).ok();
    fs::create_dir_all(fixture.audit_log()).unwrap();

    let (ok, text) = fixture.run(&["merge", "clear-veto", "veto-0001", "--by", "Grace"]);
    assert!(
        !ok,
        "a clear succeeded with an unwritable audit trail: {text}"
    );
    assert!(
        text.contains("clearing a veto is an audited event"),
        "{text}"
    );

    let (open_ok, listing) = fixture.run(&["merge", "vetoes", "--open"]);
    assert!(open_ok, "{listing}");
    assert!(
        listing.contains("[open]"),
        "the veto was rolled back as claimed, the listing says: {listing}"
    );
    let (land_ok, land_text) = fixture.land();
    assert!(
        !land_ok,
        "the branch landed on an unlogged clear: {land_text}"
    );
}

/// A veto is raised against a branch that exists, and the record of it survives
/// being read back by a different process.
#[test]
fn a_veto_against_a_branch_that_is_not_there_is_refused() {
    let fixture = Fixture::new();
    let (ok, text) = fixture.run(&[
        "merge",
        "veto",
        "arm-does-not-exist",
        "--reason",
        "nothing to look at",
        "--raised-by",
        "Alice",
    ]);
    assert!(!ok, "a veto was recorded against a missing branch: {text}");
    assert!(text.contains("arm-does-not-exist"), "{text}");

    let (listed, listing) = fixture.run(&["merge", "vetoes"]);
    assert!(listed, "{listing}");
    assert!(
        listing.contains("No vetoes on record"),
        "the refusal left something behind: {listing}"
    );
}

#[test]
fn an_unparsable_source_is_refused_before_it_is_recorded() {
    let fixture = Fixture::new();
    let (ok, text) = fixture.run(&[
        "merge", "veto", "arm-1", "--reason", "x", "--source", "vibes",
    ]);
    assert!(!ok, "{text}");
    assert!(text.contains("it is 'review' or 'verification'"), "{text}");
    let (_, listing) = fixture.run(&["merge", "vetoes"]);
    assert!(listing.contains("No vetoes on record"), "{listing}");
}
