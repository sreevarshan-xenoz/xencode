//! `QK-6` — the queue of durable facts a repository contradicts, and the twelve
//! months of unbroken contradiction that make one of them retirable.
//!
//! The fixture is a real repository built by this test (`git init`, two committed
//! files, a promoted `state.md`), because a staleness record that no check
//! produced is not a record this feature can produce. The twelve-month clock is
//! moved by passing a different instant to the same call the command makes: the
//! age arithmetic is real, only the moment it is measured from is chosen.

use std::path::{Path, PathBuf};

const MONTH_MS: u64 = 30 * 24 * 60 * 60 * 1000;
/// An arbitrary Tuesday in 2026, far enough from every other test's clock that a
/// record written by one cannot be read as another's.
const BASE: u64 = 1_760_000_000_000;

fn scratch(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-factgc-{label}-{unique}"));
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

/// A committed repository holding one contradicted fact and one that holds: the
/// auth file is modified in the working tree, the readme is not. Promotion is
/// what puts the `[src:…]` markers on the lines, so this is the file a person
/// would actually be looking at when they ran `xencode memory gc`.
fn fixture(label: &str) -> PathBuf {
    let root = scratch(label);
    git(&root, &["init", "-q"]);
    git(&root, &["config", "user.email", "test@xencode.local"]);
    git(&root, &["config", "user.name", "Xencode Test"]);
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/auth.rs"), "fn login() {}\n").unwrap();
    std::fs::write(
        root.join("README.md"),
        "Install with cargo install xencode.\n",
    )
    .unwrap();
    git(&root, &["add", "src/auth.rs", "README.md"]);
    git(&root, &["commit", "-q", "-m", "initial"]);
    std::fs::write(
        root.join("AGENTS.md"),
        "# Rules\n\n- always run the tests before committing\n",
    )
    .unwrap();

    let xencode = root.join(".xencode");
    xencode_context_rs::write_state_candidate(
        &xencode_context_rs::ContextState {
            working_on: "answer where authentication lives".to_string(),
            completed: vec![],
            decisions: vec![
                "the login entry point is src/auth.rs".to_string(),
                "what to install is written in README.md".to_string(),
            ],
            unresolved: vec![],
        },
        &xencode,
    )
    .unwrap();
    xencode_context_rs::promote_state_candidate(&xencode).unwrap();
    // The contradiction: the file one fact describes now says something else.
    std::fs::write(
        root.join("src/auth.rs"),
        "fn authenticate() {}\nfn login() {}\n",
    )
    .unwrap();
    root
}

fn queue_file(root: &Path) -> String {
    std::fs::read_to_string(xencode_context_rs::tombstone_path(&root.join(".xencode")))
        .unwrap_or_default()
}

fn gc(root: &Path, now: u64, apply: bool) -> xencode_context_rs::GcReport {
    xencode_context_rs::collect_gc(&root.join(".xencode"), now, apply).unwrap()
}

#[test]
fn a_contradicted_fact_is_queued_the_moment_it_is_noticed() {
    let root = fixture("queue");
    let report = gc(&root, BASE, false);

    assert_eq!(
        report.pending.len(),
        1,
        "one fact is contradicted: {:?}",
        report.pending.iter().map(|e| &e.fact).collect::<Vec<_>>()
    );
    let entry = &report.pending[0];
    assert_eq!(
        entry.problem,
        xencode_context_rs::FactProblem::SourceChanged
    );
    assert_eq!(entry.first_seen_ms, BASE);
    assert_eq!(entry.retired_ms, None);
    // The record is the whole point, so it has to be on disk, not just returned.
    let on_disk = xencode_context_rs::read_tombstones(&root.join(".xencode"));
    assert_eq!(on_disk, report.pending);
    // Nothing is retirable the day it is contradicted.
    assert!(report.aged.is_empty());
    assert!(report.removed.is_empty());
    // …and the file still holds the line, because contradicting it is not editing it.
    let state = std::fs::read_to_string(root.join(".xencode/state.md")).unwrap();
    assert!(state.contains("the login entry point is src/auth.rs"));
}

#[test]
fn assembling_a_turn_writes_the_record_nobody_asked_for() {
    let root = fixture("turn-path");
    // The per-turn pass is the only thing that sees a fact go stale in a normal
    // session. If it did not record, the queue would be a list of everything
    // anyone thought to run a command about, which is not a clock at all.
    xencode_context_rs::collect_live_context(
        &root,
        "where is login handled?",
        xencode_context_rs::ContextCaps::from_profile(
            xencode_context_rs::HardwareProfile::Balanced,
        ),
    );
    let queued = xencode_context_rs::read_tombstones(&root.join(".xencode"));
    assert_eq!(
        queued.len(),
        1,
        "the turn left no record: {:?}",
        queue_file(&root)
    );
    let now = xencode_context_rs::gc_now_ms();
    assert!(
        now.saturating_sub(queued[0].first_seen_ms) < 60_000,
        "the record is not from this turn: {:?}",
        queued[0]
    );
}

#[test]
fn the_clock_holds_at_eleven_months_and_retires_at_twelve() {
    let root = fixture("gate");
    gc(&root, BASE, false);
    // Re-running the pass must not restart the clock it is measuring.
    let untouched = std::fs::read_to_string(root.join(".xencode/state.md")).unwrap();
    let mid = gc(&root, BASE + 11 * MONTH_MS, true);
    assert_eq!(mid.pending[0].first_seen_ms, BASE, "the clock drifted");
    assert!(
        mid.aged.is_empty() && mid.removed.is_empty(),
        "eleven months retired a fact: {:?}",
        mid.aged
    );
    assert_eq!(
        std::fs::read_to_string(root.join(".xencode/state.md")).unwrap(),
        untouched,
        "--apply at eleven months edited state.md anyway"
    );

    let year = gc(&root, BASE + 12 * MONTH_MS, true);
    assert_eq!(
        year.removed.len(),
        1,
        "a year of contradiction retired nothing"
    );
    assert!(year.removed[0].starts_with("the login entry point is src/auth.rs"));
}

#[test]
fn a_retirement_takes_one_line_and_leaves_the_rest_of_the_file_alone() {
    let root = fixture("surgical");
    gc(&root, BASE, false);
    let agents = std::fs::read_to_string(root.join("AGENTS.md")).unwrap();
    let report = gc(&root, BASE + 13 * MONTH_MS, true);

    let state = std::fs::read_to_string(root.join(".xencode/state.md")).unwrap();
    assert!(
        !state.contains("the login entry point is src/auth.rs"),
        "the aged fact survived the sweep:\n{state}"
    );
    // Everything the sweep was not asked to touch, in the bytes it went in with.
    assert!(state.contains("what to install is written in README.md"));
    assert!(state.contains("## working-on"));
    assert!(state.contains("answer where authentication lives"));
    assert!(state.contains("# State"));
    // The instruction file is not this command's to edit, aged or not.
    assert_eq!(
        std::fs::read_to_string(root.join("AGENTS.md")).unwrap(),
        agents
    );
    assert!(
        report.retired.is_empty(),
        "a fresh retirement is not history yet"
    );
    let after = xencode_context_rs::read_tombstones(&root.join(".xencode"));
    let retired: Vec<_> = after
        .iter()
        .filter(|e| e.retired_ms == Some(BASE + 13 * MONTH_MS))
        .collect();
    assert_eq!(
        retired.len(),
        1,
        "the sweep left no record of what it took: {after:?}"
    );
    // Asking again retires nothing a second time.
    let repeat = gc(&root, BASE + 14 * MONTH_MS, true);
    assert!(repeat.removed.is_empty());
    assert_eq!(repeat.retired.len(), 1);
}

#[test]
fn a_fact_brought_back_after_a_retirement_starts_its_clock_again() {
    let root = fixture("second-clock");
    gc(&root, BASE, false);
    let first = gc(&root, BASE + 13 * MONTH_MS, true);
    let line = first.removed[0].clone();

    // The person promotes the same claim again, a year and a month later.
    std::fs::write(
        root.join(".xencode/state.md"),
        format!("# State\n\n## decisions\n- {line}\n"),
    )
    .unwrap();
    let later = BASE + 25 * MONTH_MS;
    let report = gc(&root, later, true);
    assert_eq!(report.pending.len(), 1);
    assert_eq!(
        report.pending[0].first_seen_ms, later,
        "a re-promoted fact inherited a retired one's clock and went straight to the sweep"
    );
    assert!(
        report.removed.is_empty(),
        "the new fact was retired at once"
    );
    assert!(std::fs::read_to_string(root.join(".xencode/state.md"))
        .unwrap()
        .contains("the login entry point"));
}

#[test]
fn a_fact_that_stops_being_contradicted_leaves_the_queue() {
    let root = fixture("repaired");
    gc(&root, BASE, false);
    // The change is reverted, so the fact is true of the tree again.
    git(&root, &["checkout", "--", "src/auth.rs"]);
    let year_on = BASE + 13 * MONTH_MS;
    let report = gc(&root, year_on, true);
    assert!(
        report.pending.is_empty(),
        "a fixed fact is still queued: {:?}",
        report.pending
    );
    assert!(report.removed.is_empty());
    assert_eq!(
        queue_file(&root).lines().count(),
        0,
        "the queue never forgets"
    );
    // And the line is still the person's to keep.
    assert!(std::fs::read_to_string(root.join(".xencode/state.md"))
        .unwrap()
        .contains("the login entry point is src/auth.rs"));
}

#[test]
fn a_fact_that_cannot_be_checked_is_never_retired() {
    let root = fixture("unchecked");
    gc(&root, BASE, false);
    // The same `.xencode` and the same files, in a directory git knows nothing
    // about: the provenance marker cannot be resolved, so nothing is contradicted
    // and nothing can be. A year-old record must not retire a line on the strength
    // of a judgement this run is not able to re-make.
    let elsewhere = scratch("unchecked-orphan");
    std::fs::create_dir_all(elsewhere.join("src")).unwrap();
    std::fs::copy(root.join("src/auth.rs"), elsewhere.join("src/auth.rs")).unwrap();
    std::fs::copy(root.join("README.md"), elsewhere.join("README.md")).unwrap();
    std::fs::remove_dir_all(elsewhere.join(".xencode")).unwrap();
    std::fs::rename(root.join(".xencode"), elsewhere.join(".xencode")).unwrap();
    let before = std::fs::read_to_string(elsewhere.join(".xencode/state.md")).unwrap();

    let report = gc(&elsewhere, BASE + 13 * MONTH_MS, true);
    assert!(
        report.unverifiable > 0,
        "a tree with no git still contradicted something: {report:?}"
    );
    assert!(
        report.pending.is_empty(),
        "an uncheckable fact aged: {:?}",
        report.pending
    );
    assert!(report.removed.is_empty());
    assert_eq!(
        std::fs::read_to_string(elsewhere.join(".xencode/state.md")).unwrap(),
        before
    );
}

#[test]
fn a_credential_in_a_fact_never_reaches_the_queue() {
    let root = fixture("secret");
    let xencode = root.join(".xencode");
    // A fact is model-written text about a file that may quote whatever it was
    // shown, and the queue is read and indexed like any other file in the project.
    let state = std::fs::read_to_string(xencode.join("state.md")).unwrap();
    let commit = state
        .split("[src:src/auth.rs@")
        .nth(1)
        .and_then(|tail| tail.split(']').next())
        .expect("promotion cites the auth file and the revision it was written at")
        .to_string();
    std::fs::write(
        xencode.join("state.md"),
        format!(
            "# State\n\n## decisions\n- the auth module is configured with OPENAI sk-FAKE-NOT-A-REAL-TEST-KEY [src:src/auth.rs@{commit}]\n"
        ),
    )
    .unwrap();

    let report = gc(&root, BASE, false);
    assert_eq!(report.pending.len(), 1, "{:?}", report.pending);
    let on_disk = queue_file(&root);
    assert!(
        !on_disk.contains("FAKE-NOT-A-REAL") && !on_disk.contains("sk-FAKE"),
        "the queue stored the credential: {on_disk}"
    );
    assert!(
        on_disk.contains("[redacted]"),
        "scrubbed, but not marked: {on_disk}"
    );
    // The queue is scrubbed; the person's own file is not rewritten over their
    // shoulder, stale or not.
    assert!(std::fs::read_to_string(xencode.join("state.md"))
        .unwrap()
        .contains("sk-FAKE"));
}
