//! `QK-1` end to end: `xencode memory evidence`, driven by the binary a person
//! would type, reading the ledger a real turn wrote.
//!
//! The split is the whole point of putting these checks here rather than in
//! `xencode-context-rs`. That crate's tests prove what the turn path *records*;
//! only a second process can prove what a person *reads*. So the ledger on disk
//! here is written by `collect_live_context` — the function the TUI calls before
//! it sends a prompt — and never by the test, and the report comes out of
//! `env!("CARGO_BIN_EXE_xencode")`. Every number asserted below was computed from
//! Wilson's formula by hand and is quoted against the published value for its
//! case.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use xencode_context_rs::{collect_live_context, ContextCaps, ContextState, HardwareProfile};

fn scratch(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-evidence-cli-{label}-{unique}"));
    let _ = std::fs::remove_dir_all(&root);
    root
}

fn git(root: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .expect("git should be on PATH");
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn commit(root: &Path, message: &str) {
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", message]);
}

fn head8(root: &Path) -> String {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim()[..8].to_string()
}

/// A committed repository, and the `.xencode` directory the turn path writes its
/// ledger into.
fn repo(label: &str) -> PathBuf {
    let root = scratch(label);
    std::fs::create_dir_all(root.join(".xencode")).unwrap();
    git(&root, &["init", "-q", "-b", "main"]);
    git(&root, &["config", "user.email", "test@xencode.local"]);
    git(&root, &["config", "user.name", "Xencode Test"]);
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/auth.rs"), "fn validate_token() {}\n").unwrap();
    commit(&root, "initial");
    root
}

/// Promote a fact through the writer that stamps its source revision, so the
/// marker in `state.md` is the one a real session would have.
fn promote(root: &Path, fact: &str) {
    let xencode = root.join(".xencode");
    let state = ContextState {
        working_on: String::new(),
        completed: vec![],
        decisions: vec![fact.to_string()],
        unresolved: vec![],
    };
    xencode_context_rs::write_state_candidate(&state, &xencode).unwrap();
    xencode_context_rs::promote_state_candidate(&xencode).unwrap();
}

/// One turn through the production path: this is what writes
/// `facts.evidence.jsonl`, not the test.
fn turn(root: &Path) {
    collect_live_context(
        root,
        "where is login handled?",
        ContextCaps::from_profile(HardwareProfile::Balanced),
    );
}

/// Run the built binary in the project. `XCODE_CONFIG_DIR` and `HOME` are pointed
/// at a directory that belongs to nobody else, because `run_memory` builds a
/// `ConversationMemory` before it dispatches, and that creates its state
/// directory wherever the environment says it lives. Without the override every
/// run of this test would write into a real person's configuration tree.
fn xencode(root: &Path, home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .current_dir(root)
        .env("HOME", home)
        .env("XCODE_CONFIG_DIR", home.join("portable"))
        .env_remove("XDG_STATE_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .output()
        .expect("the binary should run")
}

/// The report's text, on both streams: a refusal goes where a person will see it.
fn report(root: &Path, args: &[&str]) -> String {
    let home = scratch("home");
    let out = xencode(root, &home, args);
    assert!(out.status.success(), "the command failed to run");
    assert!(
        home.join("portable").is_dir(),
        "the run wrote its state under the override, not into a real home directory"
    );
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::remove_dir_all(&home).ok();
    text
}

#[test]
fn the_built_binary_prints_the_evidence_a_real_turn_recorded() {
    let root = repo("text");
    promote(&root, "the login entry point is src/auth.rs");
    turn(&root);
    // A second revision that leaves the cited file alone. The fact has now been
    // re-checked against two different states of the code, which is the evidence
    // the report is made of.
    std::fs::write(root.join("src/other.rs"), "fn unrelated() {}\n").unwrap();
    commit(&root, "second");
    turn(&root);

    let text = report(&root, &["memory", "evidence"]);
    assert!(
        text.contains("the login entry point is src/auth.rs"),
        "the fact is printed as prose, markers and all: {text}"
    );
    assert!(
        !text.contains("[src:"),
        "a person reading this has to re-type a commit hash to get at the sentence: {text}"
    );
    assert!(
        text.contains("checked against 2 revisions; 2 of them found nothing to contradict it"),
        "the count is of revisions, not turns: {text}"
    );
    // Wilson's 95% score interval for 2 of 2 is 0.3424 to 1, the published value.
    assert!(
        text.contains("34.2% to 100.0%"),
        "the interval a two-revision run earns: {text}"
    );
    assert!(
        text.contains("is not a chance that the fact is true"),
        "the header has to say what the number is not: {text}"
    );
    let head = head8(&root);
    assert!(
        text.contains(&format!(
            "verified by the file-and-name re-check, at revision {head}"
        )),
        "the line names the search that answered and the revision it ran at: {text}"
    );
    assert!(
        !text.contains("model"),
        "no model judged this fact, so no model's name belongs in its verdict: {text}"
    );
}

#[test]
fn the_json_report_agrees_with_the_text_report_line_for_line() {
    let root = repo("json");
    promote(&root, "the login entry point is src/auth.rs");
    turn(&root);
    std::fs::write(root.join("src/other.rs"), "fn unrelated() {}\n").unwrap();
    commit(&root, "second");
    turn(&root);

    let home = scratch("home");
    let out = xencode(&root, &home, &["memory", "evidence", "--format", "json"]);
    assert!(out.status.success(), "the command failed to run");
    let body = String::from_utf8_lossy(&out.stdout).to_string();
    std::fs::remove_dir_all(&home).ok();

    let rows: Vec<serde_json::Value> =
        serde_json::from_str(&body).expect("`--format json` must print one JSON array");
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row["fact"], "the login entry point is src/auth.rs");
    assert_eq!(row["revisions_checked"], 2, "{row}");
    assert_eq!(row["survived"], 2);
    assert_eq!(row["unchecked"], 0);
    assert_eq!(
        row["wilson_95_of_next_check_agreeing"][0].as_f64(),
        Some(0.342),
        "the same interval the text report prints, at its own precision: {row}"
    );
    assert_eq!(row["wilson_95_of_next_check_agreeing"][1], 1.0);
    assert_eq!(row["last_revision"].as_str(), Some(head8(&root).as_str()));
    assert!(row["contradicted_by"].is_null(), "{row}");
    // The two formats must not be two different claims about the same ledger.
    let text = report(&root, &["memory", "evidence"]);
    let sentence = row["verified_by"].as_str().expect("verified_by");
    assert!(
        text.contains(sentence),
        "json says {sentence}, text says something else:\n{text}"
    );
}

#[test]
fn a_contradicted_fact_is_named_with_the_reason_the_code_gave() {
    let root = repo("contradicted");
    promote(&root, "the login entry point is src/auth.rs");
    turn(&root);
    // The cited file moves: the pass that kept the fact for one revision now
    // removes it from the prompt, and the report still has to show the revision
    // that agreed before it.
    std::fs::write(root.join("src/auth.rs"), "fn validate_token_v2() {}\n").unwrap();
    commit(&root, "auth moved");
    turn(&root);

    let text = report(&root, &["memory", "evidence"]);
    assert!(
        text.contains("checked against 2 revisions; 1 of them found nothing to contradict it"),
        "a revision that said no counts against the fact instead of disappearing: {text}"
    );
    assert!(
        text.contains("contradicted now: the file it cites has changed since"),
        "the reason is the one the staleness pass returned, not a paraphrase: {text}"
    );
    let (lower, _upper) = xencode_context_rs::evidence_rows(&root.join(".xencode"))
        .remove(0)
        .interval
        .expect("two answers");
    assert!(
        text.contains(&format!("{:.1}%", lower * 100.0)),
        "the range the ledger computes is the range printed: {text}"
    );
}

#[test]
fn a_project_with_no_rechecked_fact_says_so_rather_than_printing_nothing() {
    let root = repo("empty");
    std::fs::write(
        root.join(".xencode/state.md"),
        "# State\n\n## decisions\n- we chose the smaller rewrite\n",
    )
    .unwrap();
    let text = report(&root, &["memory", "evidence"]);
    assert!(
        text.contains("No durable fact here has been re-checked against a revision."),
        "an empty report has to explain why it is empty: {text}"
    );
    assert!(
        !text.contains("weakest evidence first"),
        "a run that printed no rows must not print the row header either: {text}"
    );
}

#[test]
fn a_single_revision_run_names_how_thin_its_own_evidence_is() {
    let root = repo("thin");
    promote(&root, "the login entry point is src/auth.rs");
    turn(&root);

    let text = report(&root, &["memory", "evidence"]);
    assert!(
        text.contains("checked against 1 revision; 1 of them found nothing to contradict it"),
        "singular, because the count is revisions and there is one: {text}"
    );
    // Wilson's 95% for 1 of 1 is 0.2065 to 1 — one observation, and the report
    // says so by spanning most of the range rather than rounding to a number.
    assert!(
        text.contains("20.7% to 100.0%"),
        "the floor a single agreeing revision earns: {text}"
    );
    assert!(
        text.contains("1 of 1 reach fewer than two revisions"),
        "the footer counts the rows a person should not lean on: {text}"
    );
}
