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

/// A real repository with one detectable pattern already on an untouched line
/// and another added in a later commit. The first commit is tagged `base` so
/// the review diff is deterministic and does not depend on branch discovery.
fn fixture_introduced_repo(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-review-diff-{label}-{unique}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()
            .expect("git should be on PATH");
        assert!(out.status.success(), "git {args:?} failed");
    };
    git(&["init", "-q", "-b", "main"]);
    git(&["config", "user.email", "test@xencode.local"]);
    git(&["config", "user.name", "Xencode Test"]);

    // The generic analyzer flags every TODO line. The first one is present at
    // `base` and left untouched; the second is added in the commit under review.
    std::fs::write(
        root.join("notes.txt"),
        "TODO: existing marker\nstable line\n",
    )
    .unwrap();
    git(&["add", "."]);
    git(&["commit", "-q", "-m", "base"]);
    git(&["tag", "base"]);

    std::fs::write(
        root.join("notes.txt"),
        "TODO: existing marker\nstable line\nTODO: added marker\n",
    )
    .unwrap();
    git(&["add", "."]);
    git(&["commit", "-q", "-m", "add a finding"]);
    root
}

#[test]
fn introduced_and_preexisting_findings_are_labeled() {
    let root = fixture_introduced_repo("labels");

    // Text output: both findings are shown with a label, and the line number
    // ties the label to the right finding.
    let (out, ok) = run(&root, &["review", "--base", "base"]);
    assert!(ok, "xencode review should succeed: {out}");
    assert!(
        out.contains("[preexisting]"),
        "missing pre-existing label: {out}"
    );
    assert!(
        out.contains("[introduced]"),
        "missing introduced label: {out}"
    );
    let pre_line = out.lines().find(|l| l.contains("Ln1:")).unwrap_or_else(|| {
        panic!("no finding on untouched line 1 in: {out}");
    });
    assert!(
        pre_line.contains("[preexisting]"),
        "untouched line should be pre-existing: {pre_line}"
    );
    let added_line = out
        .lines()
        .find(|l| l.contains("Ln3:"))
        .unwrap_or_else(|| panic!("no finding on added line 3 in: {out}"));
    assert!(
        added_line.contains("[introduced]"),
        "added line should be introduced: {added_line}"
    );

    // JSON output: the label is a machine-readable `origin` field, and every
    // original CodeIssue field (such as code_snippet) is preserved.
    let (json_out, ok) = run(&root, &["review", "--base", "base", "--format", "json"]);
    assert!(
        ok,
        "xencode review --format json should succeed: {json_out}"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&json_out).expect("review json should be valid");
    let files = parsed["files"].as_array().expect("files array");
    let file = files
        .iter()
        .find(|f| f["path"] == "notes.txt")
        .expect("notes.txt in review");
    let issues = file["issues"].as_array().expect("issues array");
    assert_eq!(
        issues.len(),
        2,
        "both findings shown by default: {json_out}"
    );
    let existing = issues
        .iter()
        .find(|i| i["code_snippet"] == "TODO: existing marker")
        .expect("pre-existing finding");
    assert_eq!(existing["origin"], "preexisting");
    assert_eq!(existing["line_number"], 1);
    let added = issues
        .iter()
        .find(|i| i["code_snippet"] == "TODO: added marker")
        .expect("introduced finding");
    assert_eq!(added["origin"], "introduced");
    assert_eq!(added["line_number"], 3);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn introduced_only_hides_preexisting_findings() {
    let root = fixture_introduced_repo("only");

    // Text: the pre-existing label and finding disappear.
    let (out, ok) = run(&root, &["review", "--base", "base", "--introduced-only"]);
    assert!(ok, "xencode review --introduced-only should succeed: {out}");
    assert!(
        out.contains("[introduced]"),
        "introduced finding kept: {out}"
    );
    assert!(
        !out.contains("[preexisting]"),
        "pre-existing finding should be hidden: {out}"
    );

    // JSON: exactly one issue remains, the one on the added line.
    let (json_out, ok) = run(
        &root,
        &[
            "review",
            "--base",
            "base",
            "--introduced-only",
            "--format",
            "json",
        ],
    );
    assert!(ok, "json run should succeed: {json_out}");
    let parsed: serde_json::Value = serde_json::from_str(&json_out).unwrap();
    let files = parsed["files"].as_array().unwrap();
    let file = files.iter().find(|f| f["path"] == "notes.txt").unwrap();
    let issues = file["issues"].as_array().unwrap();
    assert_eq!(
        issues.len(),
        1,
        "only the introduced finding remains: {json_out}"
    );
    assert_eq!(issues[0]["origin"], "introduced");
    assert_eq!(issues[0]["code_snippet"], "TODO: added marker");

    let _ = std::fs::remove_dir_all(&root);
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
