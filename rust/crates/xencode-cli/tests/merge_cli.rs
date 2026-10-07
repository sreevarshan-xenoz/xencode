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
    run(&["branch", "-M", "main"]);
}

fn create_branch_with_commit(path: &std::path::Path, branch: &str, file: &str, content: &str) {
    let run = |args: &[&str]| {
        let status = Command::new("git")
            .arg("-C")
            .arg(path)
            .args(args)
            .status()
            .expect("git branch setup failed");
        assert!(status.success());
    };

    run(&["checkout", "main"]);
    run(&["checkout", "-b", branch]);
    fs::write(path.join(file), content).unwrap();
    run(&["add", file]);
    run(&["commit", "-m", &format!("commit on {branch}")]);
    run(&["checkout", "main"]);
}

#[test]
fn cli_merge_precheck_clean_and_conflict() {
    let dir = tempdir().unwrap();
    let repo = dir.path();
    init_git_repo(repo);

    // Branch 1: clean disjoint edit
    create_branch_with_commit(repo, "clean-feat", "feature1.rs", "fn f1() {}\n");

    // Branch 2: divergent modification to README.md created before main commits its edit
    create_branch_with_commit(
        repo,
        "conflict-feat",
        "README.md",
        "# Conflict branch changes\n",
    );

    // Main edits README.md creating a divergent conflict with conflict-feat
    fs::write(repo.join("README.md"), "# Main line changes\n").unwrap();
    let _ = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["commit", "-am", "main edits readme"])
        .status();

    // Precheck clean branch
    let output1 = Command::new(xencode_bin())
        .arg("merge")
        .arg("precheck")
        .arg("clean-feat")
        .arg("--base")
        .arg("main")
        .current_dir(repo)
        .output()
        .expect("cli execution failed");

    assert!(output1.status.success());
    let stdout1 = String::from_utf8_lossy(&output1.stdout);
    assert!(stdout1.contains("CLEAN (no merge conflicts)"));

    // Precheck conflicting branch
    let output2 = Command::new(xencode_bin())
        .arg("merge")
        .arg("precheck")
        .arg("conflict-feat")
        .arg("--base")
        .arg("main")
        .current_dir(repo)
        .output()
        .expect("cli execution failed");

    assert!(output2.status.success());
    let stdout2 = String::from_utf8_lossy(&output2.stdout);
    assert!(stdout2.contains("CONFLICT"));
    assert!(stdout2.contains("README.md"));
}

#[test]
fn cli_merge_plan_and_land_with_post_integration_tests() {
    let dir = tempdir().unwrap();
    let repo = dir.path();
    init_git_repo(repo);

    // Create 4 distinct branches
    create_branch_with_commit(repo, "arm-1", "file1.txt", "Arm 1 contents\n");
    create_branch_with_commit(repo, "arm-2", "file2.txt", "Arm 2 contents\n");
    create_branch_with_commit(repo, "arm-3", "file3.txt", "Arm 3 contents\n");
    create_branch_with_commit(repo, "arm-4", "file4.txt", "Arm 4 contents\n");

    // 1. Plan evaluation
    let plan_out = Command::new(xencode_bin())
        .arg("merge")
        .arg("plan")
        .arg("--branch")
        .arg("arm-1")
        .arg("--branch")
        .arg("arm-2")
        .arg("--branch")
        .arg("arm-3")
        .arg("--branch")
        .arg("arm-4")
        .arg("--base")
        .arg("main")
        .current_dir(repo)
        .output()
        .expect("cli merge plan failed");

    assert!(plan_out.status.success());
    let stdout = String::from_utf8_lossy(&plan_out.stdout);
    assert!(stdout.contains("Overall clean: true"));
    assert!(stdout.contains("Candidate branches (4)"));
    assert!(stdout.contains("arm-1"));
    assert!(stdout.contains("arm-4"));

    // 2. Land requires a named human approval gate
    let missing_gate = Command::new(xencode_bin())
        .arg("merge")
        .arg("land")
        .arg("--branch")
        .arg("arm-1")
        .current_dir(repo)
        .output()
        .expect("cli merge land failed");
    assert!(
        !missing_gate.status.success(),
        "must fail without --approved-by"
    );

    // 3. Land with human approval and post-integration tests
    let land_out = Command::new(xencode_bin())
        .arg("merge")
        .arg("land")
        .arg("--branch")
        .arg("arm-1")
        .arg("--branch")
        .arg("arm-2")
        .arg("--branch")
        .arg("arm-3")
        .arg("--branch")
        .arg("arm-4")
        .arg("--base")
        .arg("main")
        .arg("--approved-by")
        .arg("Lead Architect Alice")
        .arg("--test-cmd")
        .arg("test -f file1.txt && test -f file2.txt")
        .arg("--test-cmd")
        .arg("test -f file3.txt && test -f file4.txt")
        .current_dir(repo)
        .output()
        .expect("cli merge land failed");

    assert!(land_out.status.success());
    let land_stdout = String::from_utf8_lossy(&land_out.stdout);
    assert!(land_stdout.contains("Merge Landed Successfully!"));
    assert!(land_stdout.contains("Lead Architect Alice"));
    assert!(land_stdout.contains("Post-integration checks (2):"));
    assert!(land_stdout.contains("PASSED"));

    // Verify all 4 files exist in main
    assert!(repo.join("file1.txt").exists());
    assert!(repo.join("file2.txt").exists());
    assert!(repo.join("file3.txt").exists());
    assert!(repo.join("file4.txt").exists());
}
