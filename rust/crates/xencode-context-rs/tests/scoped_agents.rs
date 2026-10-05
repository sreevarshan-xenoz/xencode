//! EV-5, at the level a person would notice: a rule written in the `AGENTS.md`
//! of the directory being edited reaches the model on that turn, and loading it
//! does not cost a local server its cached prompt head.
//!
//! Which directories a turn works in is read from `git status` in a repository
//! this test builds with the `git` binary on this machine — `git init`, committed
//! sources, then a real edit. Nothing is faked into the prompt: the assertions
//! below are made against the text [`xencode_context_rs::assemble_prompt`] hands
//! to a model, and the head comparison is the shipped
//! [`xencode_context_rs::ContextDoc::stable_prefix`].

use std::path::{Path, PathBuf};

fn scratch(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-scoped-live-{label}-{unique}"));
    std::fs::create_dir_all(root.join(".xencode")).unwrap();
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

/// Commit with an identity set on the command line: the fixture has nothing to
/// do with whoever built this repository on this machine.
fn commit(root: &Path, message: &str) {
    git(
        root,
        &[
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "user.name=Fixture",
            "commit",
            "-q",
            "-m",
            message,
        ],
    );
}

/// A committed two-package tree with one instruction file per package.
///
/// Both files are untrusted when the repository is built, which is what a fresh
/// clone actually is; the test that wants them trusted calls [`trust_file`] itself.
fn repo(label: &str) -> PathBuf {
    let root = scratch(label);
    for package in ["auth", "api"] {
        let dir = root.join("src").join(package);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("mod.rs"), "pub fn handle() {}\n").unwrap();
    }
    std::fs::write(root.join("src/auth/AGENTS.md"), format!("{AUTH_RULE}\n")).unwrap();
    std::fs::write(root.join("src/api/AGENTS.md"), format!("{API_RULE}\n")).unwrap();
    std::fs::write(root.join("AGENTS.md"), "# root rule\n").unwrap();
    std::fs::write(root.join("README.md"), "# project\n").unwrap();
    git(&root, &["init", "-q"]);
    git(&root, &["add", "."]);
    commit(&root, "baseline");
    root
}

/// Grant one instruction file the way the `/trust` command does, by naming its
/// path. Using the real function rather than writing the store by hand is the
/// point: this file is where the reader and the grant are tested together, and
/// a rule that differs from its file by one newline is a rule nobody trusted.
fn trust_file(root: &Path, rel: &str) {
    xencode_context_rs::trust_agents_at(root, rel)
        .unwrap_or_else(|e| panic!("could not trust {rel}: {e}"));
}

const AUTH_RULE: &str = "every handler here calls check_auth before reading a request body";
const API_RULE: &str = "this package opens sockets; never reach the network from a unit test";

fn turn(root: &Path) -> xencode_context_rs::context::ContextDoc {
    let live = xencode_context_rs::collect_live_context(
        root,
        "where is login handled?",
        xencode_context_rs::ContextCaps::from_profile(
            xencode_context_rs::HardwareProfile::Balanced,
        ),
    );
    xencode_context_rs::assemble_prompt(
        xencode_context_rs::HardwareProfile::Balanced,
        "You are a coding agent.",
        live.agents_md.as_deref(),
        live.anchor_md.as_deref(),
        live.scoped_md.as_deref(),
        live.state_md.as_deref(),
        live.notes_md.as_deref(),
        &live.git_summary,
        &live.repo_map,
        vec![],
        "user: where is login handled?\nassistant: in the auth module",
    )
}

#[test]
fn an_edit_in_a_nested_directory_loads_that_directorys_own_directives() {
    let root = repo("load");
    trust_file(&root, "src/auth/AGENTS.md");
    trust_file(&root, "src/api/AGENTS.md");

    // Nothing dirty yet: a turn that works on no file loads no directory rules.
    let clean = turn(&root);
    assert!(
        !clean.text.contains("## Instructions For These Directories"),
        "a clean tree loaded nested instructions anyway"
    );

    // Edit one package: only that package's file is read.
    std::fs::write(
        root.join("src/auth/mod.rs"),
        "pub fn handle() { check_auth(); }\n",
    )
    .unwrap();
    let doc = turn(&root);
    assert!(
        doc.text.contains("## Instructions For These Directories"),
        "the section a nested rule arrives under is missing:\n{}",
        doc.text
    );
    assert!(
        doc.text.contains(AUTH_RULE),
        "the edited directory's own rule did not reach the prompt:\n{}",
        doc.text
    );
    assert!(
        !doc.text.contains(API_RULE),
        "a directory this turn is not working in was read anyway"
    );
    let tier = doc
        .tiers
        .iter()
        .find(|t| t.name == "AGENTS.md (scoped)")
        .expect("the nested instructions were assembled but never budgeted");
    assert!(tier.tokens > 0);
    assert_eq!(
        tier.class,
        xencode_context_rs::SourceClass::AgentFile { trusted: true },
        "a file the user trusted is reported as if it still were not"
    );

    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn an_untrusted_nested_rule_reaches_the_model_marked_as_data() {
    // The done-when above trusts its bytes first. This one is the case a fresh
    // clone is actually in: the file is a stranger's, so it must not sit in the
    // instruction position unlabelled — and it must still be *there*, because
    // marking something as data is not the same as deleting it.
    let root = repo("untrusted");
    std::fs::write(
        root.join("src/auth/mod.rs"),
        "pub fn handle() { check_auth(); }\n",
    )
    .unwrap();
    let doc = turn(&root);
    let text = &doc.text;
    assert!(
        text.contains(xencode_context_rs::UNTRUSTED_BANNER),
        "an untrusted file was sent as if it were instructions"
    );
    assert!(
        text.contains(AUTH_RULE),
        "the banner swallowed the file's own bytes"
    );
    let banner = text.find(xencode_context_rs::UNTRUSTED_BANNER).unwrap();
    let body = text.find(AUTH_RULE).unwrap();
    assert!(banner < body, "the marker must lead the content it marks");
    let tier = doc
        .tiers
        .iter()
        .find(|t| t.name == "AGENTS.md (scoped)")
        .expect("no scoped tier");
    assert_eq!(
        tier.class,
        xencode_context_rs::SourceClass::AgentFile { trusted: false },
        "the ledger is told the file was trusted"
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_different_directory_changes_the_tier_but_not_the_cached_head() {
    // The trap this item is built around. A local server keeps the key/value
    // cache of the head it was prefilled with; anything that varies per turn
    // inside that head is a full re-prefill on the next request. Nested rules
    // vary per turn by definition, so the two turns below differ in which
    // directories they work in — and must still agree byte for byte up to the
    // marker.
    let root = repo("kv");
    trust_file(&root, "src/auth/AGENTS.md");
    trust_file(&root, "src/api/AGENTS.md");

    std::fs::write(
        root.join("src/auth/mod.rs"),
        "pub fn handle() {\n    check_auth();\n}\n",
    )
    .unwrap();
    let first = turn(&root);
    assert!(
        first.text.contains(AUTH_RULE),
        "the auth directory's rule did not load:\n{}",
        first.text
    );

    // The other package is now the dirty one, and the first is committed clean.
    git(&root, &["add", "src/auth/mod.rs"]);
    commit(&root, "auth edit");
    std::fs::write(root.join("src/api/mod.rs"), "pub fn call() {}\n").unwrap();
    let second = turn(&root);
    assert!(
        second.text.contains(API_RULE),
        "the new directory's rule did not load:\n{}",
        second.text
    );
    assert!(
        !second.text.contains(AUTH_RULE),
        "the previous directory's rule is still being carried"
    );

    assert_eq!(
        first.stable_prefix, second.stable_prefix,
        "the stable head moved between two turns"
    );
    assert_eq!(
        first.stable_prefix_sha256(),
        second.stable_prefix_sha256(),
        "the head hash the interface compares moved"
    );
    assert_ne!(
        first.text, second.text,
        "the two turns were identical, so this proves nothing about the tier"
    );
    assert!(
        second
            .text
            .contains(xencode_context_rs::context::STABLE_END_MARKER),
        "the dynamic section must sit after the marker"
    );
    let after = second
        .text
        .find(xencode_context_rs::context::STABLE_END_MARKER)
        .unwrap();
    let rule = second.text.find(API_RULE).unwrap();
    assert!(
        after < rule,
        "nested instructions landed inside the cached head"
    );

    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_full_size_root_file_does_not_starve_the_nested_ones() {
    // The nested set has its own cap rather than whatever the root file left,
    // because the root `AGENTS.md` is the one a real project fills up. Here it
    // is over its 1200 tokens on purpose: the tier must still appear, still be
    // cut to its own ceiling, and the turn must still fit the window.
    let root = repo("budget");
    let big_root = format!("{}\n{}", "# root rule", "root detail\n".repeat(2000));
    std::fs::write(root.join("AGENTS.md"), &big_root).unwrap();
    trust_file(&root, "AGENTS.md");
    // Big enough to pass the reader's 8 KiB ceiling and to blow the nested cap:
    // the budgeter, not the reader, is what cuts it.
    let long = format!("{}\n{}", AUTH_RULE, "detail line\n".repeat(560));
    std::fs::write(root.join("src/auth/AGENTS.md"), &long).unwrap();
    trust_file(&root, "src/auth/AGENTS.md");
    std::fs::write(
        root.join("src/auth/mod.rs"),
        "pub fn handle() { check_auth(); }\n",
    )
    .unwrap();
    let doc = turn(&root);
    let agents = doc
        .tiers
        .iter()
        .find(|t| t.name == "agents.md")
        .expect("the root file is tier 2")
        .tokens;
    assert_eq!(
        agents,
        xencode_context_rs::context::AGENTS_CAP_TOKENS,
        "the root file was supposed to spend its whole cap"
    );
    let scoped = doc
        .tiers
        .iter()
        .find(|t| t.name == "AGENTS.md (scoped)")
        .expect("a full root file must not silence the directory being edited")
        .tokens;
    assert!(
        scoped <= xencode_context_rs::context::SCOPED_AGENTS_CAP_TOKENS,
        "the nested set cost {scoped} against its own {}-token cap",
        xencode_context_rs::context::SCOPED_AGENTS_CAP_TOKENS
    );
    assert!(
        scoped < xencode_context_rs::budget::est_tokens(long.len(), false),
        "the file was sent whole, so nothing was cut: {scoped} tokens"
    );
    // A tier that hit its cap has to say so, or the preview reads as a prompt
    // that fit.
    assert!(
        doc.truncated,
        "a trimmed instruction file reported no trimming"
    );
    assert!(
        doc.text.contains(AUTH_RULE),
        "cutting from the head dropped the rule that leads the file"
    );
    assert!(
        doc.total_tokens <= doc.target_tokens,
        "the turn went over its window: {} > {}",
        doc.total_tokens,
        doc.target_tokens
    );
    std::fs::remove_dir_all(&root).unwrap();
}
