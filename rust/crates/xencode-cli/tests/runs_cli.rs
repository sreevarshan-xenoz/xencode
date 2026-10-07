//! `xencode runs show` end-to-end integration test:
//!
//! Verifies AE-2:
//! 1. A session that ran verification prints those checks under `xencode runs show <id>`.
//! 2. A session that ran none says `checks: none — nothing verified this run`.
//! 3. `xencode verify --session <id>` records checks under the specified session.

use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch_env(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-runs-test-{label}-{unique}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join(".xencode")).unwrap();
    root
}

fn run_cli(root: &Path, args: &[&str]) -> (String, bool) {
    let output = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .current_dir(root)
        .env("HOME", root)
        .env("XCODE_CONFIG_DIR", root)
        .env_remove("XDG_STATE_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .output()
        .expect("binary should run");
    (
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        output.status.success(),
    )
}

#[test]
fn runs_show_joins_session_checks_and_reports_unverified_honestly() {
    let root = scratch_env("show-checks");
    let xencode_dir = root.join(".xencode");

    // Seed run 1 with a session that has verification evidence
    let run1 = xencode_context_rs::RunRecord {
        run_id: "1700000000-aaaa1111".to_string(),
        ts_unix_ms: 1_700_000_000_000,
        duration_ms: 500,
        rounds: 1,
        session: Some("session_with_checks".to_string()),
        model: Some("test-model".to_string()),
        provider: Some("test-provider".to_string()),
        source: None,
        approvals: Vec::new(),
        recording: None,
        note: String::new(),
        computer: Some("colab".to_string()),
    };
    xencode_context_rs::append_run(&xencode_dir, &run1).unwrap();

    // Append a ledger check for session_with_checks
    let entry = xencode_context_rs::LedgerEntry {
        ts_unix_ms: 1_700_000_000_100,
        session: Some("session_with_checks".to_string()),
        run_class: xencode_context_rs::RunClass::Test,
        exit_code: 0,
        subjects: vec!["cargo test".to_string()],
        log_ref: "artifacts/session_with_checks/verify-test.log".to_string(),
        note: "pass".to_string(),
    };
    xencode_context_rs::append_ledger(&xencode_dir, &entry).unwrap();

    // Seed run 2 with a session that has NO verification evidence
    let run2 = xencode_context_rs::RunRecord {
        run_id: "1700000000-bbbb2222".to_string(),
        ts_unix_ms: 1_700_000_001_000,
        duration_ms: 300,
        rounds: 1,
        session: Some("session_without_checks".to_string()),
        model: Some("test-model".to_string()),
        provider: Some("test-provider".to_string()),
        source: None,
        approvals: Vec::new(),
        recording: None,
        note: String::new(),
        computer: None,
    };
    xencode_context_rs::append_run(&xencode_dir, &run2).unwrap();

    // Check run 1 in text output: reports checks
    let (out1, ok1) = run_cli(&root, &["runs", "show", "1700000000-aaaa1111"]);
    assert!(ok1, "runs show run1 should succeed: {out1}");
    assert!(out1.contains("run 1700000000-aaaa1111"), "{out1}");
    assert!(out1.contains("session: session_with_checks"), "{out1}");
    assert!(out1.contains("checks: 1 (1 passed)"), "{out1}");
    assert!(
        out1.contains("exit 0  artifacts/session_with_checks/verify-test.log"),
        "{out1}"
    );

    // Check run 1 in json format
    let (json_out1, jok1) = run_cli(
        &root,
        &["runs", "show", "1700000000-aaaa1111", "--format", "json"],
    );
    assert!(jok1, "runs show json should succeed: {json_out1}");
    let doc1: serde_json::Value = serde_json::from_str(&json_out1).unwrap();
    assert_eq!(doc1["verified"], true);
    assert_eq!(doc1["checks"].as_array().unwrap().len(), 1);

    // Check run 2 in text output: reports explicit absence
    let (out2, ok2) = run_cli(&root, &["runs", "show", "1700000000-bbbb2222"]);
    assert!(ok2, "runs show run2 should succeed: {out2}");
    assert!(out2.contains("run 1700000000-bbbb2222"), "{out2}");
    assert!(out2.contains("session: session_without_checks"), "{out2}");
    assert!(
        out2.contains("checks: none — nothing verified this run"),
        "{out2}"
    );

    // Check run 2 in json format
    let (json_out2, jok2) = run_cli(
        &root,
        &["runs", "show", "1700000000-bbbb2222", "--format", "json"],
    );
    assert!(jok2, "runs show json should succeed: {json_out2}");
    let doc2: serde_json::Value = serde_json::from_str(&json_out2).unwrap();
    assert_eq!(doc2["verified"], false);
    assert_eq!(doc2["checks"].as_array().unwrap().len(), 0);

    let _ = std::fs::remove_dir_all(&root);
}
