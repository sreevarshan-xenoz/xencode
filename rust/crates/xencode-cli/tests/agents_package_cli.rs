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
fn cli_build_package_records_observed_facts_without_progress_claims() {
    let dir = tempdir().unwrap();
    let repo = dir.path();
    init_git_repo(repo);

    // Modify a file
    fs::write(repo.join("README.md"), "# Initial Project\n## Feature\n").unwrap();

    let output = Command::new(xencode_bin())
        .arg("agents")
        .arg("--build-package")
        .arg("task-cli-1")
        .arg("--agent")
        .arg("first-agent")
        .arg("--test-cmd")
        .arg("test -f README.md")
        .current_dir(repo)
        .output()
        .expect("cli execution failed");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Worker continuation package written to"));
    assert!(stdout.contains("task-cli-1"));
    assert!(stdout.contains("first-agent"));
    assert!(stdout.contains("README.md"));
    assert!(stdout.contains("No self-reported progress percentages"));

    let pkg_path = repo
        .join(".xencode")
        .join("packages")
        .join("task-cli-1.json");
    assert!(pkg_path.exists());
    let pkg_json = fs::read_to_string(&pkg_path).unwrap();

    // Verify invariant: zero progress claims in serialized JSON
    assert!(!pkg_json.contains("progress_percent"));
    assert!(!pkg_json.contains("completion_claim"));
}

#[test]
fn cli_view_package_displays_resumption_context() {
    let dir = tempdir().unwrap();
    let repo = dir.path();
    init_git_repo(repo);

    fs::write(repo.join("README.md"), "# Updated\n").unwrap();

    let build_out = Command::new(xencode_bin())
        .arg("agents")
        .arg("--build-package")
        .arg("task-view-1")
        .arg("--test-cmd")
        .arg("test -f README.md")
        .current_dir(repo)
        .output()
        .unwrap();
    assert!(build_out.status.success());

    let pkg_path = repo
        .join(".xencode")
        .join("packages")
        .join("task-view-1.json");

    let view_out = Command::new(xencode_bin())
        .arg("agents")
        .arg("--package")
        .arg(&pkg_path)
        .output()
        .unwrap();

    assert!(view_out.status.success());
    let stdout = String::from_utf8_lossy(&view_out.stdout);
    assert!(stdout.contains("# Task Continuation: task-view-1"));
    assert!(stdout.contains("Observed Repository State"));
    assert!(stdout.contains("README.md"));
}

#[test]
fn cli_resume_package_executes_second_worker_and_checks_outcome() {
    let dir = tempdir().unwrap();
    let repo = dir.path();
    init_git_repo(repo);

    // Initial edit with incomplete work
    fs::write(repo.join("test.txt"), "incomplete\n").unwrap();

    // Check command that verifies "complete"
    let check_cmd = "grep -q 'complete' test.txt";

    // Build continuation package
    let build_out = Command::new(xencode_bin())
        .arg("agents")
        .arg("--build-package")
        .arg("task-resume-1")
        .arg("--test-cmd")
        .arg(check_cmd)
        .current_dir(repo)
        .output()
        .unwrap();
    assert!(build_out.status.success());

    let pkg_path = repo
        .join(".xencode")
        .join("packages")
        .join("task-resume-1.json");

    // Worker 2 completes work in the repository
    fs::write(repo.join("test.txt"), "complete\n").unwrap();

    // Resume task with worker 2
    let resume_out = Command::new(xencode_bin())
        .arg("agents")
        .arg("--package")
        .arg(&pkg_path)
        .arg("--resume")
        .arg("--agent")
        .arg("second-worker")
        .arg("--test-cmd")
        .arg(check_cmd)
        .current_dir(repo)
        .output()
        .unwrap();

    assert!(resume_out.status.success());
    let stdout = String::from_utf8_lossy(&resume_out.stdout);
    assert!(stdout.contains("Resumption outcome for task 'task-resume-1':"));
    assert!(stdout.contains("second-worker"));
    assert!(stdout.contains("Verification tests passed: true"));
    assert!(stdout.contains("PASSED"));
}

#[test]
fn cli_refuses_corrupted_package_with_progress_claims() {
    let dir = tempdir().unwrap();
    let corrupt_file = dir.path().join("fraudulent.json");

    fs::write(
        &corrupt_file,
        r#"{
        "task_id": "fake",
        "task_description": "fake",
        "previous_agent": "worker",
        "stop_reason": { "kind": "unknown" },
        "observed_diff": { "changed_files": [], "patch": "", "insertions": 0, "deletions": 0 },
        "verified_tests": [],
        "recent_events": [],
        "created_at_unix_ms": 123,
        "progress_percent": 80
    }"#,
    )
    .unwrap();

    let view_out = Command::new(xencode_bin())
        .arg("agents")
        .arg("--package")
        .arg(&corrupt_file)
        .output()
        .unwrap();

    assert!(!view_out.status.success());
    let stderr = String::from_utf8_lossy(&view_out.stderr);
    assert!(
        stderr.contains("unknown field `progress_percent`")
            || stderr.contains("invalid worker package")
    );
}
