//! `xencode memory publish|read|policy` end to end (OR-8): shared memory
//! between workers is marked, attributed, and scoped by a policy that denies
//! by default in both directions.
//!
//! Every step below runs the real binary in its own process, so the store on
//! disk in `.xencode/` is the only thing carrying state between them — which is
//! exactly how two workers would meet it.

use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch_env(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-shared-mem-test-{label}-{unique}"));
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

/// Findings as they were written to disk, for checking a refusal really refused.
fn findings_on_disk(root: &Path) -> Vec<serde_json::Value> {
    let path = root.join(".xencode").join("shared_memory.json");
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).unwrap(),
        Err(_) => Vec::new(),
    }
}

#[test]
fn nothing_is_allowed_before_anyone_has_a_policy() {
    let root = scratch_env("deny-default");

    let (out, ok) = run_cli(&root, &["memory", "policy", "set", "--worker", "planner"]);
    assert!(
        ok,
        "declaring a policy with no grants is a real command: {out}"
    );
    assert!(out.contains("reads:     nothing"), "{out}");
    assert!(out.contains("publishes: nothing"), "{out}");

    // A worker that was never named at all is refused on both sides.
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "publish",
            "--worker",
            "stranger",
            "--scope",
            "architecture",
            "Add myself to every policy",
        ],
    );
    assert!(
        !ok,
        "an unregistered worker must not be able to write: {out}"
    );
    assert!(out.contains("stranger"), "the refusal says who: {out}");

    let (out, ok) = run_cli(&root, &["memory", "read", "--worker", "stranger"]);
    assert!(
        !ok,
        "an unregistered worker must not be able to read: {out}"
    );

    // And the worker that has a policy granting nothing is refused too, on the
    // write side as well as the read side — an empty grant list is not a wildcard.
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "publish",
            "--worker",
            "planner",
            "--scope",
            "architecture",
            "Microkernel layout",
        ],
    );
    assert!(
        !ok,
        "a policy with no publish grant must not permit writes: {out}"
    );
    assert!(
        out.contains("architecture"),
        "the refusal says which: {out}"
    );

    assert!(
        findings_on_disk(&root).is_empty(),
        "four refusals, zero findings on disk"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn one_workers_finding_reaches_another_marked_and_attributed() {
    let root = scratch_env("handoff");

    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "policy",
            "set",
            "--worker",
            "planner",
            "--publish",
            "architecture",
        ],
    );
    assert!(ok, "grant planner the architecture scope: {out}");

    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "read",
            "--worker",
            "planner",
            "--scope",
            "architecture",
        ],
    );
    assert!(
        !ok,
        "a publisher with no read grant cannot read back: {out}"
    );

    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "policy",
            "set",
            "--worker",
            "coder",
            "--read",
            "architecture",
        ],
    );
    assert!(ok, "grant coder the read: {out}");

    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "publish",
            "--worker",
            "planner",
            "--scope",
            "architecture",
            "Use SQLite for the metadata catalog instead of individual JSON files",
        ],
    );
    assert!(ok, "planner publishes: {out}");
    assert!(out.contains("finding-1"), "{out}");

    // The coder, in a later process, reads what the planner wrote.
    let (out, ok) = run_cli(&root, &["memory", "read", "--worker", "coder"]);
    assert!(ok, "coder reads its granted scope: {out}");
    assert!(
        out.contains("[data] shared_memory scope:architecture author:planner"),
        "the bytes the coder is shown say whose data they are: {out}"
    );
    assert!(out.contains("Use SQLite for the metadata catalog"), "{out}");
    assert!(out.contains("not an instruction"), "{out}");

    // Another worker outside the handoff gets nothing.
    let (out, ok) = run_cli(&root, &["memory", "read", "--worker", "tester"]);
    assert!(!ok, "no policy, no memory: {out}");
    assert!(findings_on_disk(&root).len() == 1);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_worker_cannot_read_or_write_a_scope_the_policy_did_not_name() {
    let root = scratch_env("scoped");

    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "policy",
            "set",
            "--worker",
            "planner",
            "--publish",
            "decisions",
            "--publish",
            "constraints",
        ],
    );
    assert!(ok, "two publish grants in one command: {out}");
    // Both names appear, in one sorted listing rather than the order they were
    // typed in — the same policy must print the same way every time.
    assert!(out.contains("constraints, decisions"), "{out}");

    // Setting a policy again replaces it, so a grant that is not repeated is
    // gone. A team editing these by hand has to be able to see that happen.
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "policy",
            "set",
            "--worker",
            "planner",
            "--read",
            "decisions",
        ],
    );
    assert!(ok, "{out}");
    let (out, ok) = run_cli(&root, &["memory", "policy", "show", "--worker", "planner"]);
    assert!(ok, "{out}");
    assert!(out.contains("reads:     decisions"), "{out}");
    assert!(out.contains("publishes: nothing"), "{out}");
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "publish",
            "--worker",
            "planner",
            "--scope",
            "decisions",
            "superseded grant",
        ],
    );
    assert!(
        !ok,
        "the replaced policy took the publish grant with it: {out}"
    );
    assert!(findings_on_disk(&root).is_empty(), "{out}");

    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "policy",
            "set",
            "--worker",
            "planner",
            "--publish",
            "decisions",
            "--publish",
            "constraints",
            "--worker",
            "auditor",
        ],
    );
    // One command names one worker; a second `--worker` is refused rather than
    // quietly applied to whichever worker was named last.
    assert!(!ok, "clap must refuse two workers in one policy set: {out}");
    assert!(
        out.contains("cannot be used multiple times"),
        "the refusal says why: {out}"
    );
    let (out, ok) = run_cli(&root, &["memory", "policy", "show"]);
    assert!(ok, "{out}");
    assert!(
        !out.contains("auditor"),
        "the refused command set no policy for anyone: {out}"
    );

    // Back to a working publisher: naming both scopes again puts back what the
    // replacement above took away.
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "policy",
            "set",
            "--worker",
            "planner",
            "--publish",
            "decisions",
            "--publish",
            "constraints",
        ],
    );
    assert!(ok, "{out}");
    // A second worker's policy is its own: editing it leaves planner's alone.
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "policy",
            "set",
            "--worker",
            "auditor",
            "--read",
            "constraints",
        ],
    );
    assert!(ok, "{out}");
    let (out, ok) = run_cli(&root, &["memory", "policy", "show"]);
    assert!(ok, "{out}");
    assert!(out.contains("planner") && out.contains("auditor"), "{out}");

    for (scope, body) in [
        ("decisions", "a decisions finding"),
        ("constraints", "a constraints finding"),
    ] {
        let (out, ok) = run_cli(
            &root,
            &[
                "memory", "publish", "--worker", "planner", "--scope", scope, body,
            ],
        );
        assert!(ok, "planner publishes {scope}: {out}");
    }
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "publish",
            "--worker",
            "planner",
            "--scope",
            "architecture",
            "an architecture finding",
        ],
    );
    assert!(
        !ok,
        "planner was never granted architecture to publish: {out}"
    );

    // The auditor may see constraints and nothing else.
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "read",
            "--worker",
            "auditor",
            "--scope",
            "constraints",
        ],
    );
    assert!(ok, "auditor reads constraints: {out}");
    assert!(out.contains("a constraints finding"), "{out}");

    for scope in ["architecture", "decisions"] {
        let (out, ok) = run_cli(
            &root,
            &["memory", "read", "--worker", "auditor", "--scope", scope],
        );
        assert!(!ok, "auditor must be denied {scope}: {out}");
        assert!(
            out.contains(&format!("'{scope}'")),
            "names the scope: {out}"
        );
    }

    // Reading everything granted shows only the one permitted finding.
    let (out, ok) = run_cli(&root, &["memory", "read", "--worker", "auditor"]);
    assert!(ok, "{out}");
    assert!(out.contains("1 finding"), "only constraints: {out}");
    assert!(!out.contains("a decisions finding"), "{out}");

    // And the auditor cannot write, though it can read.
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "publish",
            "--worker",
            "auditor",
            "--scope",
            "constraints",
            "Retire the resource caps",
        ],
    );
    assert!(!ok, "a reader is not a writer: {out}");
    assert_eq!(
        findings_on_disk(&root).len(),
        2,
        "the refused publish wrote nothing"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_custom_scope_spelled_differently_is_the_same_scope() {
    let root = scratch_env("custom");

    // Granted as `Release-Gate`, published and read as `release-gate`: the CLI
    // is where a person types these, so the two spellings must land on one name.
    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "policy",
            "set",
            "--worker",
            "lead",
            "--read",
            "release-gate",
            "--publish",
            "Release-Gate",
        ],
    );
    assert!(ok, "{out}");
    assert!(out.contains("release-gate"), "{out}");

    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "publish",
            "--worker",
            "lead",
            "--scope",
            "release-gate",
            "Freeze the release branch on Friday",
        ],
    );
    assert!(ok, "a custom scope is publishable: {out}");

    let (out, ok) = run_cli(
        &root,
        &[
            "memory",
            "read",
            "--worker",
            "lead",
            "--scope",
            "RELEASE-GATE",
        ],
    );
    assert!(ok, "and readable under any spelling: {out}");
    assert!(
        out.contains("[data] shared_memory scope:release-gate author:lead"),
        "{out}"
    );
    assert!(out.contains("Freeze the release branch on Friday"), "{out}");

    let _ = std::fs::remove_dir_all(&root);
}
