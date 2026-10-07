use std::fs;
use std::process::Command;
use tempfile::tempdir;

fn xencode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xencode")
}

fn init_git_repo(path: &std::path::Path) {
    let run = |args: &[&str]| {
        let status = Command::new("git")
            .arg("-C")
            .arg(path)
            .args(args)
            .status()
            .expect("git setup failed");
        assert!(status.success());
    };

    run(&["init"]);
    run(&["config", "user.email", "tester@example.com"]);
    run(&["config", "user.name", "Tester"]);
    fs::write(path.join("README.md"), "# Initial Project\n").unwrap();
    run(&["add", "README.md"]);
    run(&["commit", "-m", "initial commit"]);
}

#[test]
fn cli_agents_redispatch_preserves_diff_and_logs_second_attempt_in_ledger() {
    let dir = tempdir().unwrap();
    let repo = dir.path();
    init_git_repo(repo);

    // Initial worker modified src/lib.rs before being killed
    fs::create_dir_all(repo.join("src")).unwrap();
    let initial_code = "pub fn add_numbers(a: i32, b: i32) -> i32 { a + b }\n";
    fs::write(repo.join("src/lib.rs"), initial_code).unwrap();

    // Re-dispatch the task onto a replacement worker with failure reason signal:9
    let output = Command::new(xencode_bin())
        .arg("agents")
        .arg("--redispatch")
        .arg("task-failover-99")
        .arg("--agent")
        .arg("killed-worker")
        .arg("--replacement-agent")
        .arg("survivor-worker")
        .arg("--stop-reason")
        .arg("signal:9")
        .arg("--test-cmd")
        .arg("test -f src/lib.rs && grep -q 'add_numbers' src/lib.rs")
        .arg("--format")
        .arg("json")
        .current_dir(repo)
        .output()
        .expect("cli execution failed");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let outcome: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON output");

    // 1. Task completed elsewhere on replacement worker
    assert_eq!(outcome["task_id"], "task-failover-99");
    assert_eq!(outcome["initial_worker"], "killed-worker");
    assert_eq!(outcome["replacement_worker"], "survivor-worker");
    assert_eq!(outcome["attempts"], 2);
    assert_eq!(outcome["all_tests_passed"], true);

    // 2. Diff was preserved without loss
    let diff_files = outcome["preserved_diff"]["changed_files"]
        .as_array()
        .expect("changed_files array");
    assert!(diff_files.iter().any(|f| f.as_str() == Some("src/lib.rs")));
    let on_disk_code = fs::read_to_string(repo.join("src/lib.rs")).unwrap();
    assert_eq!(on_disk_code, initial_code);

    // 3. Retry is visible in the ledger as a second attempt on one task
    let ledger_path = repo.join(".xencode").join("task_ledger.jsonl");
    assert!(ledger_path.exists());
    let ledger_content = fs::read_to_string(&ledger_path).unwrap();
    let lines: Vec<&str> = ledger_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(lines.len(), 2, "ledger must record exactly 2 attempts");

    let attempt1: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(attempt1["task_id"], "task-failover-99");
    assert_eq!(attempt1["attempt"], 1);
    assert_eq!(attempt1["worker"], "killed-worker");
    assert_eq!(attempt1["passed"], false);
    assert_eq!(attempt1["stop_reason"]["kind"], "signal");
    assert_eq!(attempt1["stop_reason"]["signal"], 9);

    let attempt2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(attempt2["task_id"], "task-failover-99");
    assert_eq!(attempt2["attempt"], 2);
    assert_eq!(attempt2["worker"], "survivor-worker");
    assert_eq!(attempt2["passed"], true);
    assert_eq!(attempt2["stop_reason"]["kind"], "exit_code");
    assert_eq!(attempt2["stop_reason"]["code"], 0);
}
