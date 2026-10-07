//! `xencode review` end to end, driven by the real CLI binary.
//!
//! Verifies that running `xencode review` in a repository whose only branch
//! is `master` reviews cleanly with no `--base`, reporting fallback to
//! `init.defaultBranch (master)`, that `--base` overrides discovery, and that
//! an origin/HEAD remote ref is detected and reported.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_master_repo(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-review-cli-{label}-{unique}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn hello() {}\n").unwrap();
    for args in [
        vec!["init", "-q", "-b", "master"],
        vec!["config", "user.email", "test@xencode.local"],
        vec!["config", "user.name", "Xencode Test"],
        vec!["add", "."],
        vec!["commit", "-q", "-m", "initial"],
    ] {
        let out = Command::new("git")
            .args(&args)
            .current_dir(&root)
            .output()
            .expect("git should be on PATH");
        assert!(out.status.success(), "git {args:?} failed");
    }
    root
}

fn run(root: &Path, args: &[&str]) -> (String, bool) {
    let output = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .current_dir(root)
        .output()
        .expect("the binary should run");
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
fn master_only_repository_reviews_cleanly_without_base() {
    let root = fixture_master_repo("master-only");

    // Default text output: reports 0 files and fallback to init.defaultBranch (master)
    let (out, ok) = run(&root, &["review"]);
    assert!(ok, "xencode review should succeed on master repo: {out}");
    assert!(
        out.contains("Review of diff master...HEAD (0 files) [no remote; fell back to init.defaultBranch (master)]"),
        "unexpected output: {out}"
    );

    // JSON output: includes base and base_source
    let (json_out, ok) = run(&root, &["review", "--format", "json"]);
    assert!(
        ok,
        "xencode review --format json should succeed: {json_out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&json_out).expect("review json output should be valid json");
    assert_eq!(v["base"], "master");
    assert_eq!(v["base_source"], "init.defaultBranch");
    assert_eq!(v["files_changed"], 0);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn explicit_base_overrides_discovery_and_omits_suffix() {
    let root = fixture_master_repo("explicit-base");

    let (out, ok) = run(&root, &["review", "--base", "master"]);
    assert!(ok, "xencode review --base master should succeed: {out}");
    assert!(
        out.contains("Review of diff master...HEAD (0 files)"),
        "unexpected output: {out}"
    );
    assert!(
        !out.contains("["),
        "explicit base should not have source bracket suffix: {out}"
    );

    let (json_out, ok) = run(&root, &["review", "--base", "master", "--format", "json"]);
    assert!(ok, "json run should succeed");
    let v: serde_json::Value = serde_json::from_str(&json_out).unwrap();
    assert_eq!(v["base"], "master");
    assert_eq!(v["base_source"], "explicit");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn origin_head_is_reported_when_present() {
    let root = fixture_master_repo("origin-head");

    // Add a remote ref and origin/HEAD
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()
            .unwrap();
        assert!(out.status.success());
    };
    git(&["update-ref", "refs/remotes/origin/main", "HEAD"]);
    git(&[
        "symbolic-ref",
        "refs/remotes/origin/HEAD",
        "refs/remotes/origin/main",
    ]);

    let (out, ok) = run(&root, &["review"]);
    assert!(ok, "xencode review should succeed with origin/HEAD: {out}");
    assert!(
        out.contains(
            "Review of diff origin/main...HEAD (0 files) [base resolved from origin/HEAD]"
        ),
        "unexpected output: {out}"
    );

    let (json_out, ok) = run(&root, &["review", "--format", "json"]);
    assert!(ok, "json run should succeed");
    let v: serde_json::Value = serde_json::from_str(&json_out).unwrap();
    assert_eq!(v["base"], "origin/main");
    assert_eq!(v["base_source"], "origin/HEAD");

    let _ = std::fs::remove_dir_all(&root);
}
