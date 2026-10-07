//! `xencode memory` sessions behavior end to end:
//!
//! Verifies AA-4:
//! 1. A failed query leaves conversation_memory.json byte-identical (does not persist empty sessions).
//! 2. `xencode memory list` filters empty sessions (with 0 messages) by default.
//! 3. `xencode memory list --all` displays all sessions including empty ones.
//! 4. `xencode memory prune` deletes empty sessions from disk.

use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch_env(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-mem-test-{label}-{unique}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn run_cli(config_dir: &Path, args: &[&str]) -> (String, bool) {
    let output = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .current_dir(config_dir)
        .env("HOME", config_dir)
        .env("XCODE_CONFIG_DIR", config_dir)
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
fn failed_query_leaves_memory_file_byte_identical() {
    let env_dir = scratch_env("query-fail");
    let mem_file = env_dir.join("conversation_memory.json");

    // Write a starting memory file with an existing conversation
    let initial_data = serde_json::json!({
        "conversations": {
            "existing_session": {
                "messages": [
                    {
                        "role": "user",
                        "content": "existing turn",
                        "timestamp": "2026-10-06T12:00:00Z",
                        "model": null
                    }
                ],
                "created": "2026-10-06T12:00:00Z",
                "last_updated": "2026-10-06T12:00:00Z",
                "model": null
            }
        },
        "current_session": "existing_session",
        "last_updated": "2026-10-06T12:00:00Z"
    });
    let initial_bytes = serde_json::to_string_pretty(&initial_data).unwrap();
    std::fs::write(&mem_file, &initial_bytes).unwrap();

    // Run a query against an invalid model name so it fails at provider resolution
    let (out, ok) = run_cli(
        &env_dir,
        &["query", "hello", "--model", "ollama/no-such-model"],
    );
    assert!(!ok, "query should fail: {out}");

    // The file must be byte-identical!
    let after_bytes = std::fs::read_to_string(&mem_file).unwrap();
    assert_eq!(
        initial_bytes, after_bytes,
        "memory file must remain byte-identical after failed query"
    );

    let _ = std::fs::remove_dir_all(&env_dir);
}

#[test]
fn memory_list_filters_empty_and_prune_removes_them() {
    let env_dir = scratch_env("mem-list-prune");
    let mem_file = env_dir.join("conversation_memory.json");

    let data = serde_json::json!({
        "conversations": {
            "full_session": {
                "messages": [
                    {
                        "role": "user",
                        "content": "something real",
                        "timestamp": "2026-10-06T12:00:00Z",
                        "model": null
                    }
                ],
                "created": "2026-10-06T12:00:00Z",
                "last_updated": "2026-10-06T12:00:00Z",
                "model": null
            },
            "empty_session_1": {
                "messages": [],
                "created": "2026-10-06T12:00:00Z",
                "last_updated": "2026-10-06T12:00:00Z",
                "model": null
            }
        },
        "current_session": "full_session",
        "last_updated": "2026-10-06T12:00:00Z"
    });
    std::fs::write(&mem_file, serde_json::to_string_pretty(&data).unwrap()).unwrap();

    // 1. memory list filters out empty session
    let (out, ok) = run_cli(&env_dir, &["memory", "list"]);
    assert!(ok, "memory list should succeed: {out}");
    assert!(out.contains("full_session"), "{out}");
    assert!(
        !out.contains("empty_session_1"),
        "empty session should be filtered: {out}"
    );
    assert!(
        out.contains(
            "1 empty session(s) omitted; use --all to show, or `xencode memory prune` to delete"
        ),
        "{out}"
    );

    // 2. memory list --all shows all sessions
    let (all_out, ok) = run_cli(&env_dir, &["memory", "list", "--all"]);
    assert!(ok, "memory list --all should succeed: {all_out}");
    assert!(all_out.contains("full_session (1 message)"), "{all_out}");
    assert!(
        all_out.contains("empty_session_1 (0 messages)"),
        "{all_out}"
    );

    // 3. memory prune deletes the empty session
    let (prune_out, ok) = run_cli(&env_dir, &["memory", "prune"]);
    assert!(ok, "memory prune should succeed: {prune_out}");
    assert!(
        prune_out.contains("Pruned 1 empty conversation session."),
        "{prune_out}"
    );

    // 4. memory list --all now shows only full_session
    let (after_prune, ok) = run_cli(&env_dir, &["memory", "list", "--all"]);
    assert!(ok, "{after_prune}");
    assert!(
        after_prune.contains("full_session (1 message)"),
        "{after_prune}"
    );
    assert!(!after_prune.contains("empty_session_1"), "{after_prune}");

    let _ = std::fs::remove_dir_all(&env_dir);
}
