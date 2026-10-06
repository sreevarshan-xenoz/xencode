//! `QK-1` — what a project records about how much re-checking each durable fact
//! has survived, and what that evidence is allowed to be printed as.
//!
//! The fixture is a real repository built here, and the turns are the real turn
//! path: [`collect_live_context`] is what the TUI calls before it sends anything,
//! so the ledger these tests read is written by the code that writes it in a
//! session, not by the test. `git` is the binary on this machine. Nothing is
//! recorded, nothing is faked, and every number quoted below was computed from
//! Wilson's formula by hand.

use std::path::{Path, PathBuf};

use xencode_context_rs::{
    collect_live_context, evidence_rows, read_evidence, CheckVerdict, ContextCaps, ContextState,
    HardwareProfile, StaleFacts,
};

fn scratch(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-evidence-{label}-{unique}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join(".xencode")).unwrap();
    root
}

fn git(root: &Path, args: &[&str]) {
    let out = std::process::Command::new("git")
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

fn commit(root: &Path, message: &str) -> String {
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", message]);
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A committed repository with one Rust file declaring one name.
fn repo(label: &str) -> PathBuf {
    let root = scratch(label);
    git(&root, &["init", "-q", "-b", "main"]);
    git(&root, &["config", "user.email", "test@xencode.local"]);
    git(&root, &["config", "user.name", "Xencode Test"]);
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/auth.rs"), "fn validate_token() {}\n").unwrap();
    commit(&root, "initial");
    root
}

/// One turn through the path a session uses.
fn turn(root: &Path) {
    collect_live_context(
        root,
        "where is login handled?",
        ContextCaps::from_profile(HardwareProfile::Balanced),
    );
}

/// Promote a fact about the committed file through the writer that stamps it.
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

fn the_only_row(root: &Path) -> xencode_context_rs::EvidenceRow {
    let rows = evidence_rows(&root.join(".xencode"));
    assert_eq!(rows.len(), 1, "one fact in, one row out");
    rows.into_iter().next().unwrap()
}

#[test]
fn a_fact_earns_one_entry_per_revision_rather_than_per_turn() {
    let root = repo("revisions");
    promote(&root, "the login entry point is src/auth.rs");

    // Three turns at one revision. The tree did not move, so the check's answer
    // was settled by the first of them, and counting the other two would let a
    // busy afternoon read like a fact that survived three changes.
    turn(&root);
    turn(&root);
    turn(&root);
    let row = the_only_row(&root);
    assert_eq!(row.trials, 1, "three turns at one revision are one answer");
    assert_eq!(row.survived, 1);

    // A second revision, with the cited file untouched: this is the movement the
    // ledger is supposed to be counting.
    std::fs::write(root.join("src/other.rs"), "fn unrelated() {}\n").unwrap();
    commit(&root, "second");
    turn(&root);
    let row = the_only_row(&root);
    assert_eq!(row.trials, 2, "a second revision is a second answer");
    assert_eq!(row.survived, 2);
}

#[test]
fn the_interval_is_wilson_s_and_its_width_is_what_a_thin_run_prints() {
    let root = repo("interval");
    promote(&root, "the login entry point is src/auth.rs");
    turn(&root);

    let row = the_only_row(&root);
    let (lower, upper) = row.interval.expect("one answered revision has an interval");
    // Wilson's 95% score interval for 1 of 1 is 0.2065 to 1 — computed by hand
    // from the formula, and the published value for that case.
    assert!(
        (lower - 0.2065).abs() < 1e-3 && (upper - 1.0).abs() < 1e-9,
        "one revision of unbroken agreement must print how little it proves, got {lower}..{upper}"
    );
    let sentence = row.evidence_sentence();
    assert!(
        sentence.contains("checked against 1 revision;") && sentence.contains("20.7% to 100.0%"),
        "the sentence says what was counted and what the range covers: {sentence}"
    );

    // Two of two reaches 0.342, so the floor rises with evidence rather than
    // arriving as a fixed label.
    std::fs::write(root.join("src/other.rs"), "fn unrelated() {}\n").unwrap();
    commit(&root, "second");
    turn(&root);
    let grown = the_only_row(&root).interval.expect("two revisions").0;
    assert!(
        grown > lower + 0.10 && (grown - 0.3424).abs() < 1e-3,
        "a second agreeing revision should lift the floor to about 0.342, got {grown}"
    );
}

#[test]
fn a_contradicted_revision_counts_against_the_fact_instead_of_vanishing() {
    let root = repo("contradicted");
    promote(&root, "the login entry point is src/auth.rs");
    turn(&root);
    assert_eq!(the_only_row(&root).survived, 1);

    // The cited file moves: the same pass that kept the fact now drops it, and
    // the ledger has to record that a revision said no rather than lose the row.
    std::fs::write(root.join("src/auth.rs"), "fn validate_token_v2() {}\n").unwrap();
    commit(&root, "auth moved");
    turn(&root);

    let row = the_only_row(&root);
    assert_eq!(row.trials, 2, "both revisions answered");
    assert_eq!(row.survived, 1, "one of them contradicted the fact");
    assert_eq!(
        row.problem,
        Some(xencode_context_rs::FactProblem::SourceChanged),
        "the report names why the line is out of the prompt"
    );
    assert!(
        row.interval.expect("two answers").0 < 0.35,
        "a fact contradicted once cannot read as stronger than one that never was"
    );
    assert!(
        !the_only_row(&root).verified_by().contains("nothing"),
        "the row still names the check it did run"
    );
}

#[test]
fn an_answer_the_check_could_not_get_is_not_counted_as_evidence() {
    let root = repo("unchecked");
    promote(&root, "the login entry point is src/auth.rs");
    let xencode = root.join(".xencode");
    let stored = std::fs::read_to_string(xencode.join("state.md")).unwrap();

    // The pass reaching no conclusion at all — git would not answer — must not be
    // filed as a revision that agreed.
    let no_answer = StaleFacts::default();
    xencode_context_rs::record_evidence(&xencode, &stored, &no_answer, 1_700_000_000_000).unwrap();
    let rows = evidence_rows(&xencode);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].trials, 0, "not-an-answer is not an answer");
    assert_eq!(rows[0].unchecked, 1);
    assert!(
        rows[0].interval.is_none(),
        "nothing to build an interval from"
    );
    assert!(
        rows[0]
            .evidence_sentence()
            .contains("1 check could not be run here"),
        "{}",
        rows[0].evidence_sentence()
    );

    // The same turn one revision later, answering: the entry for the revision that
    // could not be answered stays, and still contributes no trial.
    let xencode_rows = read_evidence(&xencode);
    assert_eq!(xencode_rows[0].checks[0].verdict, CheckVerdict::Unchecked);
}

#[test]
fn the_ledger_rewrites_only_when_a_revision_or_a_verdict_is_new() {
    let root = repo("quiet");
    promote(&root, "the login entry point is src/auth.rs");
    let xencode = root.join(".xencode");
    let stored = std::fs::read_to_string(xencode.join("state.md")).unwrap();

    let survived = StaleFacts {
        passed: vec![stored
            .lines()
            .find(|line| line.contains("[src:"))
            .unwrap()
            .trim_start_matches('-')
            .trim()
            .to_string()],
        ..Default::default()
    };
    xencode_context_rs::record_evidence(&xencode, &stored, &survived, 1_700_000_000_000).unwrap();
    let first = std::fs::read_to_string(xencode.join("facts.evidence.jsonl")).unwrap();

    // A later turn at the same revision, with the same answer and a much later
    // clock: the file must still carry the moment the revision first appeared,
    // because otherwise the ledger records how often the project was used.
    xencode_context_rs::record_evidence(&xencode, &stored, &survived, 1_900_000_000_000).unwrap();
    let second = std::fs::read_to_string(xencode.join("facts.evidence.jsonl")).unwrap();
    assert_eq!(first, second, "an unchanged turn rewrote the ledger anyway");
    assert!(
        second.contains("1700000000000"),
        "the clock moved on a row that had nothing new to say: {second}"
    );
}

#[test]
fn a_fact_taken_out_of_the_file_takes_its_tally_with_it() {
    let root = repo("leaves");
    promote(&root, "the login entry point is src/auth.rs");
    turn(&root);
    assert_eq!(read_evidence(&root.join(".xencode")).len(), 1);

    let xencode = root.join(".xencode");
    std::fs::write(
        xencode.join("state.md"),
        "# State\n\n## decisions\n- a different line with no marker at all\n",
    )
    .unwrap();
    turn(&root);
    assert!(
        read_evidence(&xencode).is_empty(),
        "the evidence belongs to the line, and the line is gone"
    );
    assert!(
        evidence_rows(&xencode).is_empty(),
        "a report of a fact nobody holds is a report of nothing"
    );
}

#[test]
fn dropping_one_of_two_facts_leaves_the_tally_of_the_one_that_stayed() {
    let root = repo("prune");
    std::fs::write(root.join("src/sessions.rs"), "fn fold_state() {}\n").unwrap();
    commit(&root, "second file");
    let head = git_head(&root);
    let xencode = root.join(".xencode");
    std::fs::write(
        xencode.join("state.md"),
        format!(
            "# State\n\n## decisions\n- the login entry point is src/auth.rs [src:src/auth.rs@{head}]\n- session folding lives in src/sessions.rs [src:src/sessions.rs@{head}]\n"
        ),
    )
    .unwrap();
    turn(&root);
    let rows = evidence_rows(&xencode);
    assert_eq!(rows.len(), 2, "both facts were checked at this revision");
    assert!(rows.iter().all(|row| row.trials == 1));

    // One line goes; the ledger must go with it. The case is deliberately not
    // "every fact left", which the empty-file path also handles: this is the turn
    // where a stale row would otherwise sit beside a live one and be reported as
    // though the project still believed it.
    std::fs::write(
        xencode.join("state.md"),
        format!(
            "# State\n\n## decisions\n- the login entry point is src/auth.rs [src:src/auth.rs@{head}]\n"
        ),
    )
    .unwrap();
    turn(&root);
    let kept = read_evidence(&xencode);
    assert_eq!(
        kept.len(),
        1,
        "the deleted line's tally is not evidence about anything"
    );
    assert!(
        kept[0].fact.starts_with("the login entry point"),
        "{}",
        kept[0].fact
    );
    assert_eq!(evidence_rows(&xencode).len(), 1);
}

#[test]
fn a_project_with_nothing_marked_never_creates_the_ledger() {
    let root = repo("silent");
    let xencode = root.join(".xencode");
    std::fs::write(
        xencode.join("state.md"),
        "# State\n\n## decisions\n- we chose the smaller rewrite\n",
    )
    .unwrap();
    turn(&root);
    turn(&root);
    assert!(
        !xencode.join("facts.evidence.jsonl").is_file(),
        "nothing was asked of the code, so there is nothing to record and no reason to ask git"
    );
}

#[test]
fn rows_are_ordered_with_the_thinnest_evidence_first() {
    let root = repo("order");
    std::fs::write(root.join("src/sessions.rs"), "fn fold_state() {}\n").unwrap();
    commit(&root, "second file");
    let xencode = root.join(".xencode");
    // Two facts, one of which cites a revision this repository cannot resolve: it
    // stays in the prompt, and it has no evidence to be believed on.
    std::fs::write(
        xencode.join("state.md"),
        "# State\n\n## decisions\n- the login entry point is src/auth.rs [src:src/auth.rs@00000000]\n- session folding lives in src/sessions.rs [src:src/sessions.rs@aaaaaaaa]\n",
    )
    .unwrap();
    turn(&root);

    let rows = evidence_rows(&xencode);
    assert_eq!(rows.len(), 2, "both facts are reported");
    assert_eq!(rows[0].trials, 0, "the unresolvable one has no answer");
    assert_eq!(rows[1].trials, 0, "nor does the other, at this revision");
    assert!(
        rows[0].fact.starts_with("session folding"),
        "equal evidence falls back to the fact's own text, which is the only other thing \
         the row has: {:?}",
        rows[0].fact
    );
    assert!(rows[1].fact.starts_with("the login"));

    // Now give the second fact a revision it can be checked at, and watch the
    // ordering follow the evidence rather than the file's order.
    let head = git_head(&root);
    std::fs::write(
        xencode.join("state.md"),
        format!(
            "# State\n\n## decisions\n- the login entry point is src/auth.rs [src:src/auth.rs@{head}]\n- session folding lives in src/sessions.rs [src:src/sessions.rs@aaaaaaaa]\n"
        ),
    )
    .unwrap();
    turn(&root);
    let rows = evidence_rows(&xencode);
    assert_eq!(
        rows[0].trials, 0,
        "the fact with nothing behind it is still the first thing a person reads"
    );
    assert_eq!(rows[1].trials, 1);
}

fn git_head(root: &Path) -> String {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim()[..8].to_string()
}

#[test]
fn the_revision_stored_is_the_one_the_provenance_marker_uses() {
    let root = repo("key");
    promote(&root, "the login entry point is src/auth.rs");
    turn(&root);

    let stored = read_evidence(&root.join(".xencode"));
    let record = &stored[0].checks[0];
    let head = git_head(&root);
    assert_eq!(
        record.revision, head,
        "a ledger keyed on a different truncation than the marker cannot be joined to it"
    );
    assert_eq!(record.verdict, CheckVerdict::Survived);
    assert!(
        the_only_row(&root)
            .verified_by()
            .contains(&format!("at revision {head}")),
        "the report names the revision the check ran against"
    );
}
