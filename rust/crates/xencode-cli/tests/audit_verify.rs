//! `xencode audit verify` from end to end.
//!
//! The log is produced by the server's own sink, edited the way someone with
//! an editor would edit it — a string changed in the file — and then checked by
//! a separate process running the real binary. A passing run here means the
//! writer and the command agree about the format, which a unit test inside the
//! writing crate cannot show.

use std::path::{Path, PathBuf};
use std::process::Command;

use xencode_collaboration_rs::{Role, WorkspaceManager};
use xencode_server_rs::audit::AuditSink;

/// Two real audit records, chained: a workspace created and a member added.
fn recorded_log(dir: &Path) -> PathBuf {
    let path = dir.join("audit.jsonl");
    let mut workspaces = WorkspaceManager::new();
    workspaces.create_workspace_with_id("s1", "session", "alice");
    workspaces
        .add_member("s1", "alice", "bob", Role::Editor)
        .expect("alice is admin and may add a member");
    AuditSink::to_file(&path).sync_from(&workspaces);
    path
}

/// What the command printed, and whether it considers the log sound.
fn verify(path: &Path) -> (String, bool) {
    let output = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .arg("audit")
        .arg("verify")
        .arg(path)
        .output()
        .expect("the xencode binary ran");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    (text, output.status.success())
}

fn rewrite(path: &Path, line: usize, needle: &str, replacement: &str) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    assert!(
        lines[line - 1].contains(needle),
        "line {line} of a freshly written log should contain {needle}, it held: {}",
        lines[line - 1]
    );
    lines[line - 1] = lines[line - 1].replace(needle, replacement);
    std::fs::write(path, format!("{}\n", lines.join("\n"))).unwrap();
}

#[test]
fn a_log_nobody_touched_verifies_and_the_command_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let path = recorded_log(dir.path());

    let (text, ok) = verify(&path);
    assert!(ok, "{text}");
    assert!(text.contains("2 records"), "{text}");
    assert!(text.contains("chain intact"), "{text}");
}

#[test]
fn changing_who_was_added_is_reported_on_the_line_that_changed() {
    let dir = tempfile::tempdir().unwrap();
    let path = recorded_log(dir.path());
    rewrite(&path, 2, "bob", "mallory");

    let (text, ok) = verify(&path);
    assert!(!ok, "an edited log must not exit 0: {text}");
    assert!(text.contains("line 2"), "{text}");
    assert!(text.contains("digest"), "{text}");
}

#[test]
fn removing_a_record_is_reported_at_the_record_that_followed_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = recorded_log(dir.path());
    let text = std::fs::read_to_string(&path).unwrap();
    let last = text.lines().last().unwrap();
    std::fs::write(&path, format!("{last}\n")).unwrap();

    let (output, ok) = verify(&path);
    assert!(!ok, "{output}");
    assert!(output.contains("line 1"), "{output}");
    assert!(output.contains("predecessor"), "{output}");
}

#[test]
fn a_log_that_has_not_been_written_yet_is_not_called_broken() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("audit.jsonl");

    let (text, ok) = verify(&path);
    assert!(ok, "{text}");
    assert!(text.contains("nothing to check"), "{text}");
}
