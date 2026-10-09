//! EVd-5: the fold that replaces a conversation is handed what actually ran —
//! the run ledger's newest rows and the checks recent turns recorded — beside
//! the notes pad, so a summary's "completed" can rest on exit codes rather than
//! on what the conversation claimed.

use std::path::PathBuf;

use xencode_context_rs::{
    append_ledger, append_trace, compaction_notes, hard_compact_prompt, ChecksVerdict,
    ContextState, LedgerEntry, RunClass, Transcript, TurnTrace,
};

fn temp_xencode(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-ledger-fold-{label}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let xencode = dir.join(".xencode");
    std::fs::create_dir_all(&xencode).unwrap();
    xencode
}

fn row(exit_code: i32, note: &str) -> LedgerEntry {
    LedgerEntry {
        ts_unix_ms: 1,
        session: Some("s1".to_string()),
        run_class: RunClass::Test,
        exit_code,
        subjects: Vec::new(),
        log_ref: String::new(),
        note: note.to_string(),
    }
}

#[test]
fn the_fold_is_handed_the_runs_that_were_recorded() {
    let xencode = temp_xencode("records");
    xencode_context_rs::append_note(&xencode, "remember: the parser owns line numbers").unwrap();
    append_ledger(&xencode, &row(1, "2 failed")).unwrap();
    append_ledger(&xencode, &row(0, "")).unwrap();
    let mut turn = TurnTrace::new(10, 2);
    turn.checks = Some(ChecksVerdict {
        ran: vec!["cargo test".into()],
        failed: vec!["cargo test".into()],
        ..Default::default()
    });
    append_trace(&xencode, &turn).unwrap();

    let notes = compaction_notes(&xencode).expect("there is something to hand on");
    assert!(notes.contains("the parser owns line numbers"), "{notes}");
    assert!(notes.contains("test run exited 1 — 2 failed"), "{notes}");
    assert!(notes.contains("test run exited 0"), "{notes}");
    assert!(
        notes.contains("after a turn's edits: checks: 1 ran, 1 failed"),
        "{notes}"
    );
    assert!(notes.contains("is not a fact"), "{notes}");
    // Oldest first: the failing run is listed before the passing one.
    assert!(notes.find("exited 1").unwrap() < notes.find("exited 0").unwrap());

    let prompt = hard_compact_prompt(
        &ContextState::default(),
        &Transcript::default(),
        Some(&notes),
    );
    assert!(prompt.contains("test run exited 1"), "{prompt}");

    let _ = std::fs::remove_dir_all(xencode.parent().unwrap());
}

#[test]
fn with_nothing_recorded_the_fold_gets_the_notes_unchanged() {
    let xencode = temp_xencode("plain");
    assert_eq!(compaction_notes(&xencode), None);
    xencode_context_rs::append_note(&xencode, "only a note").unwrap();
    let notes = compaction_notes(&xencode).unwrap();
    assert!(notes.contains("only a note"));
    assert!(!notes.contains("Recorded runs"), "{notes}");
    let _ = std::fs::remove_dir_all(xencode.parent().unwrap());
}
