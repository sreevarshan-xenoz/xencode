//! `QK-9` — the files a project xencode has never seen is missing, written by a
//! program that ran nothing in it.
//!
//! Every fixture here is a real repository built by this test (`git init`, real
//! commits, real file names), because the whole claim of the feature is that
//! what lands in those files was read off a disk rather than written by
//! imagination. The settings template is passed in as bytes the CLI generates
//! from `XencodeConfig::default()`; what these tests hold is where those bytes
//! go and what may not be in them.

use std::path::{Path, PathBuf};

/// A directory with nothing in it yet — no `.xencode/`, because a report-only run
/// must not be the reason that directory appears.
fn bare(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-bootstrap-{label}-{unique}"));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn git(root: &Path, args: &[&str]) {
    let out = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .expect("git should be on PATH");
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn git_out(root: &Path, args: &[&str]) -> String {
    let out = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .expect("git should be on PATH");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A committed repository shaped like something a person cloned: a readme, a
/// manifest, two source files, and nothing under `.xencode/`.
fn fixture(label: &str) -> PathBuf {
    let root = bare(label);
    git(&root, &["init", "-q", "-b", "main"]);
    git(&root, &["config", "user.email", "test@xencode.local"]);
    git(&root, &["config", "user.name", "Xencode Test"]);
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"demo\"\n").unwrap();
    std::fs::write(root.join("README.md"), "# demo\n").unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn a() {}\n").unwrap();
    std::fs::write(root.join("src/auth.rs"), "pub fn login() {}\n").unwrap();
    std::fs::write(root.join("NOTES.md"), "scratch\n").unwrap();
    git(&root, &["add", "."]);
    git(&root, &["commit", "-q", "-m", "initial"]);
    root
}

/// The bytes the CLI hands over: every key at its default, nine credentials
/// absent. Written out here rather than read from the config crate because this
/// crate does not depend on it — the CLI's own test loads this same shape
/// through the real config loader.
fn template() -> String {
    r#"{
  "api_keys": {
    "openai_api_key": null,
    "openrouter_api_key": null,
    "google_gemini_api_key": null,
    "qwen_client_id": null,
    "qwen_api_key": null,
    "remote_api_key": null,
    "nvidia_api_key": null,
    "brave_api_key": null,
    "tavily_api_key": null
  },
  "agent_hooks": { "before": {}, "after": {} }
}"#
    .to_string()
}

fn read(root: &Path, rel: &str) -> String {
    std::fs::read_to_string(root.join(rel))
        .unwrap_or_else(|e| panic!("{rel} was not readable: {e}"))
}

#[test]
fn a_clone_with_nothing_writes_three_files_saying_what_it_ran_on() {
    let root = fixture("write");
    let report = xencode_context_rs::bootstrap(&root, &template(), false).unwrap();
    assert_eq!(report.writing().count(), 3, "{:?}", report.entries);
    assert_eq!(report.keeping().count(), 0);

    // The revision and branch are the ones git itself reports, not a string the
    // test made up.
    let head = git_out(&root, &["rev-parse", "HEAD"]);
    let anchor = read(&root, ".xencode/anchor.md");
    assert!(anchor.contains("git branch `main`"), "{anchor}");
    assert!(
        anchor.contains(&head.chars().take(8).collect::<String>()),
        "the anchor names a revision that is not HEAD: {anchor}"
    );
    assert!(anchor.contains("- files git reports here: 5"), "{anchor}");
    assert!(
        anchor.contains("`Cargo.toml`, `NOTES.md`, `README.md`"),
        "{anchor}"
    );
    // Equal counts resolve by name, which is the only reason two machines agree
    // on what this file says.
    assert!(anchor.contains(".md (2), .rs (2), .toml (1)"), "{anchor}");

    // The byte-stable tier may not hold a path that only exists on this machine.
    assert!(
        !anchor.contains(root.display().to_string().as_str()),
        "an absolute path in the anchor costs every later request its cache"
    );

    let agents = read(&root, "AGENTS.md");
    for forbidden in ["cargo test", "cargo build", "npm run", "pytest", "make "] {
        assert!(
            !agents.contains(forbidden),
            "AGENTS.md tells an agent to run {forbidden:?}, which nothing checked"
        );
    }
    assert!(agents.contains("`xencode bootstrap` put these headings here"));
}

#[test]
fn a_second_run_replaces_no_byte_of_what_is_already_there() {
    let root = fixture("twice");
    xencode_context_rs::bootstrap(&root, &template(), false).unwrap();
    let agents = read(&root, "AGENTS.md");
    let anchor = read(&root, ".xencode/anchor.md");
    let settings = read(&root, ".xencode.example.json");

    let again = xencode_context_rs::bootstrap(&root, &template(), false).unwrap();
    assert_eq!(again.writing().count(), 0, "{:?}", again.entries);
    assert_eq!(again.keeping().count(), 3);
    assert_eq!(
        read(&root, "AGENTS.md"),
        agents,
        "the first file was replaced"
    );
    assert_eq!(read(&root, ".xencode/anchor.md"), anchor);
    assert_eq!(read(&root, ".xencode.example.json"), settings);
}

#[test]
fn a_file_a_person_answered_is_never_the_one_written() {
    let root = fixture("edited");
    std::fs::write(
        root.join("AGENTS.md"),
        "# demo\n\nRun `just check` before committing anything.\n",
    )
    .unwrap();
    let report = xencode_context_rs::bootstrap(&root, &template(), false).unwrap();
    assert_eq!(report.keeping().count(), 1, "{:?}", report.entries);
    assert_eq!(report.writing().count(), 2);
    assert_eq!(
        read(&root, "AGENTS.md"),
        "# demo\n\nRun `just check` before committing anything.\n",
        "a person's instruction file was overwritten by a skeleton"
    );
    // The command they wrote survives into the anchor's neighbour, not the other
    // way round: only the missing files are offered.
    assert!(read(&root, ".xencode/anchor.md").contains("git branch `main`"));
}

#[test]
fn a_report_opens_the_directory_and_creates_nothing() {
    let root = fixture("check");
    let report = xencode_context_rs::bootstrap(&root, &template(), true).unwrap();
    assert!(report.check_only);
    assert_eq!(report.writing().count(), 3, "{:?}", report.entries);
    assert!(!root.join("AGENTS.md").exists());
    assert!(!root.join(".xencode.example.json").exists());
    assert!(
        !root.join(".xencode").exists(),
        "a report left a state directory behind"
    );
    // Reading the plan twice must stay cheap and stay identical.
    let again = xencode_context_rs::bootstrap(&root, &template(), true).unwrap();
    assert_eq!(again, report);
}

/// The anchor is only worth writing if the prompt reader is the thing that reads
/// it, and `is_current` is that reader's own question.
#[test]
fn the_anchor_lands_on_the_path_the_prompt_reads_and_nowhere_else() {
    let root = fixture("anchor-path");
    xencode_context_rs::bootstrap(&root, &template(), false).unwrap();
    assert!(
        !root.join("anchor.md").exists(),
        "a root anchor is never read"
    );
    let text = read(&root, ".xencode/anchor.md");
    assert!(
        xencode_context_rs::is_current(&root, &text),
        "the product's own check does not recognise the file it would read"
    );
}

#[test]
fn a_repository_before_its_first_commit_is_described_as_it_is() {
    let root = bare("unborn");
    git(&root, &["init", "-q", "-b", "main"]);
    git(&root, &["config", "user.email", "test@xencode.local"]);
    git(&root, &["config", "user.name", "Xencode Test"]);
    std::fs::write(root.join("README.md"), "# nothing committed yet\n").unwrap();

    let report = xencode_context_rs::bootstrap(&root, &template(), false).unwrap();
    assert!(report.facts.is_git);
    assert_eq!(
        report.facts.revision.as_deref(),
        Some("(unborn HEAD)"),
        "{:?}",
        report.facts
    );
    let anchor = read(&root, ".xencode/anchor.md");
    assert!(
        anchor.contains("a git repository with no commit yet"),
        "{anchor}"
    );
    assert!(
        !anchor.contains("(unborn HEAD)"),
        "the branch line reads as though a revision exists: {anchor}"
    );
    // One file is reported even with no commit, because `git ls-files -co` counts
    // the untracked ones. The honest reading of an unborn repository is therefore
    // "no revision", not "empty directory", and the anchor says each separately.
    assert!(anchor.contains("- files git reports here: 1"), "{anchor}");
    assert!(
        anchor.contains("- files at the repository root: `README.md`"),
        "{anchor}"
    );
}

/// A file name is somebody else's string, and this file is meant to be committed.
#[test]
fn a_credential_shaped_file_name_does_not_reach_the_written_anchor() {
    let root = fixture("secret-name");
    std::fs::write(
        root.join("sk-FAKE-NOT-A-REAL-TEST-KEY.txt"),
        "a name, not a key\n",
    )
    .unwrap();
    git(&root, &["add", "sk-FAKE-NOT-A-REAL-TEST-KEY.txt"]);
    git(
        &root,
        &["commit", "-q", "-m", "a file with an unlucky name"],
    );

    let report = xencode_context_rs::bootstrap(&root, &template(), false).unwrap();
    assert!(report.scrubbed, "the run did not notice it had to scrub");
    let anchor = read(&root, ".xencode/anchor.md");
    assert!(!anchor.contains("sk-FAKE"), "{anchor}");
    assert!(anchor.contains("[redacted]"), "{anchor}");
    // The person's own file is still there under the name they gave it.
    assert!(root.join("sk-FAKE-NOT-A-REAL-TEST-KEY.txt").exists());
}

/// The bootstrap is a first-minute command. It must not become a second writer
/// for anything the durable-knowledge features already own.
#[test]
fn state_and_the_retirement_queue_are_not_mine_to_touch() {
    let root = fixture("neighbours");
    std::fs::create_dir_all(root.join(".xencode")).unwrap();
    std::fs::write(
        root.join(".xencode/state.md"),
        "# State\n\n## decisions\n- the login entry point is src/auth.rs [src:src/auth.rs@deadbeef]\n",
    )
    .unwrap();
    std::fs::write(
        root.join(".xencode/facts.tombstones.jsonl"),
        "{\"fact\":\"x\",\"problem\":\"source_missing\",\"first_seen_ms\":1,\"retired_ms\":null}\n",
    )
    .unwrap();
    let state = read(&root, ".xencode/state.md");
    let queue = read(&root, ".xencode/facts.tombstones.jsonl");

    let report = xencode_context_rs::bootstrap(&root, &template(), false).unwrap();
    assert_eq!(report.writing().count(), 3, "{:?}", report.entries);
    assert_eq!(read(&root, ".xencode/state.md"), state);
    assert_eq!(read(&root, ".xencode/facts.tombstones.jsonl"), queue);
}

#[test]
fn the_settings_file_it_writes_keeps_every_credential_absent() {
    let root = fixture("settings");
    xencode_context_rs::bootstrap(&root, &template(), false).unwrap();
    let written = read(&root, ".xencode.example.json");
    let value: serde_json::Value = serde_json::from_str(&written).expect("valid JSON");
    let keys = value["api_keys"]
        .as_object()
        .expect("the template names the api_keys block");
    assert_eq!(keys.len(), 9, "{keys:?}");
    for (name, value) in keys {
        assert!(
            value.is_null(),
            "the template put a value in {name}, which is a credential field"
        );
    }
    let hooks = value["agent_hooks"]
        .as_object()
        .expect("the template names the two hook blocks");
    assert!(
        hooks["before"].as_object().unwrap().is_empty(),
        "a seeded hook is a command run on this machine that nobody chose"
    );
    assert!(hooks["after"].as_object().unwrap().is_empty());
    // A file that ends mid-token would be read by the next person as their config.
    assert!(written.ends_with("}\n"), "{written}");
    // What the scrubber would have done to this is the reason the seed does not
    // run it here: `redact_secrets` replaces the value beside a key that looks
    // like a credential, so it rewrites `"openai_api_key": null` into a string.
    // The keys being null is the property; a text scrubber cannot attest to it.
    let mangled = xencode_context_rs::redact_secrets(&written);
    assert!(
        mangled.contains("[redacted]"),
        "the scrubber no longer rewrites a null credential, so the reason for \
         leaving the template alone needs re-checking"
    );
}
