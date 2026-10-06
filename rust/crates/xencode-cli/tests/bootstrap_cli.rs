//! `xencode bootstrap` end to end, driven by the binary a person would type.
//!
//! The unit tests in `xencode-context-rs/tests/bootstrap.rs` cover what the
//! writer decides. What they cannot show is the other half of the claim: that a
//! settings file this command wrote into somebody's project is a settings file
//! this same binary can read back, and that the anchor it wrote is the anchor the
//! prompt assembles. Both are checked here against the real process, and the
//! second through the product's own loaders rather than a test's expectations.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A committed repository, because the anchor's content comes from git.
fn fixture(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-boot-cli-{label}-{unique}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("src")).unwrap();
    for (name, text) in [
        ("src/lib.rs", "pub fn a() {}\n"),
        ("Cargo.toml", "[package]\nname = \"demo\"\n"),
        ("README.md", "# demo\n"),
    ] {
        std::fs::write(root.join(name), text).unwrap();
    }
    for args in [
        vec!["init", "-q", "-b", "main"],
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
    // Both streams, because a refusal is printed where a person will see it and
    // a test that reads only stdout cannot tell the two apart.
    (
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        output.status.success(),
    )
}

fn read(root: &Path, rel: &str) -> String {
    std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

#[test]
fn what_the_command_writes_is_read_by_the_things_that_consume_it() {
    let root = fixture("round-trip");
    let (printed, ok) = run(&root, &["bootstrap", ".", "--format", "json"]);
    assert!(ok, "{printed}");
    let report: serde_json::Value = serde_json::from_str(&printed).expect("json from the command");
    assert_eq!(report["check"], serde_json::json!(false));
    assert_eq!(report["entries"].as_array().unwrap().len(), 3, "{report}");
    for entry in report["entries"].as_array().unwrap() {
        assert_eq!(entry["action"], "write", "{entry}");
    }

    // The anchor is the tail of the byte-stable prompt head, so the product's own
    // question about it has to answer yes.
    let anchor = read(&root, ".xencode/anchor.md");
    assert!(
        xencode_context_rs::is_current(&root, &anchor),
        "the prompt reader does not recognise the file the command wrote"
    );
    assert!(anchor.contains("git branch `main`"), "{anchor}");

    // The settings file has to load through the loader that reads config.json,
    // which is the only way to know a generated template is a real config shape.
    let config = xencode_config_rs::XencodeConfig::load_from(root.join(".xencode.example.json"))
        .expect("the written settings template does not load");
    assert_eq!(
        config.api_keys,
        xencode_config_rs::ApiKeys::default(),
        "a credential was not absent in the file the command wrote"
    );
    assert!(
        config.agent_hooks.before.is_empty() && config.agent_hooks.after.is_empty(),
        "the command seeded a hook, which is a command run on this machine"
    );

    let agents = read(&root, "AGENTS.md");
    assert!(agents.contains("`xencode bootstrap` put these headings here"));
    for forbidden in ["cargo test", "cargo build", "npm run", "pytest", "make "] {
        assert!(
            !agents.contains(forbidden),
            "AGENTS.md names {forbidden:?}, which nothing ran"
        );
    }
}

#[test]
fn a_report_written_by_the_command_creates_no_directory_and_says_so() {
    let root = fixture("check");
    let (printed, ok) = run(&root, &["bootstrap", ".", "--check", "--format", "json"]);
    assert!(ok, "{printed}");
    let report: serde_json::Value = serde_json::from_str(&printed).expect("json");
    assert_eq!(report["check"], serde_json::json!(true));
    assert_eq!(report["entries"].as_array().unwrap().len(), 3);
    assert!(
        !root.join(".xencode").exists(),
        "a report left state behind"
    );
    assert!(!root.join("AGENTS.md").exists());

    // The text form is what a person reads, and it must not say "written".
    let (text, ok) = run(&root, &["bootstrap", ".", "--check"]);
    assert!(ok, "{text}");
    assert!(text.contains("would write"), "{text}");
    assert!(!text.contains("3 files written"), "{text}");
    assert!(!root.join("AGENTS.md").exists());
}

/// The one guarantee that matters most here, checked by making a person's file
/// first and then running the command twice over the directory.
#[test]
fn the_command_run_again_over_a_project_somebody_already_answered_changes_nothing() {
    let root = fixture("twice");
    std::fs::write(
        root.join("AGENTS.md"),
        "# demo\n\nCheck with `just verify`. Never touch src/generated/.\n",
    )
    .unwrap();
    let (_, ok) = run(&root, &["bootstrap", "."]);
    assert!(ok);
    let agents = read(&root, "AGENTS.md");
    let anchor = read(&root, ".xencode/anchor.md");
    let settings = read(&root, ".xencode.example.json");

    let (second, ok) = run(&root, &["bootstrap", "."]);
    assert!(ok, "{second}");
    assert!(second.contains("Nothing to write"), "{second}");
    assert_eq!(read(&root, "AGENTS.md"), agents);
    assert_eq!(read(&root, ".xencode/anchor.md"), anchor);
    assert_eq!(read(&root, ".xencode.example.json"), settings);
}

#[test]
fn a_path_that_is_not_a_project_directory_is_named_and_refused() {
    let root = fixture("refusal");
    let missing = root.join("nope");
    let (printed, ok) = run(&root, &["bootstrap", "nope", "--format", "json"]);
    assert!(!ok, "a nonexistent path was accepted: {printed}");
    assert!(
        printed.contains("not a directory"),
        "the refusal did not name the path: {printed}"
    );
    assert!(!missing.exists());

    let file = root.join("Cargo.toml");
    let (printed, ok) = run(&root, &["bootstrap", "Cargo.toml"]);
    assert!(!ok, "a file was accepted as a project: {printed}");
    assert!(file.exists(), "refusing a file must not have touched it");
}
