//! QM-2, checked at the level a person would notice: a durable fact stops
//! reaching the prompt the moment the file it describes changes.
//!
//! The fixture is a real repository built by this test — `git init`, a committed
//! source file, a promoted `state.md` — because a provenance marker that has no
//! revision behind it tests nothing. No recorded bytes and no fake git: every
//! claim below comes out of the `git` binary on this machine.

use std::path::{Path, PathBuf};

fn scratch(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-stale-live-{label}-{unique}"));
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

fn login_fact() -> xencode_context_rs::ContextState {
    xencode_context_rs::ContextState {
        working_on: "answer where authentication lives".to_string(),
        completed: vec![],
        decisions: vec!["the login entry point is src/auth.rs".to_string()],
        unresolved: vec![],
    }
}

fn turn_mentions_login(root: &Path) -> bool {
    let live = collect(root);
    let doc = xencode_context_rs::assemble_prompt(
        xencode_context_rs::HardwareProfile::Balanced,
        "You are a coding agent.",
        live.agents_md.as_deref(),
        live.anchor_md.as_deref(),
        live.state_md.as_deref(),
        &live.git_summary,
        &live.repo_map,
        vec![],
        "user: where is login\nassistant: in the auth module",
    );
    doc.text.contains("the login entry point is src/auth.rs")
}

fn collect(root: &Path) -> xencode_context_rs::LiveContext {
    xencode_context_rs::collect_live_context(
        root,
        "where is login handled?",
        xencode_context_rs::ContextCaps::from_profile(
            xencode_context_rs::HardwareProfile::Balanced,
        ),
    )
}

#[test]
fn a_cited_fact_reaches_the_prompt_until_the_file_it_describes_changes() {
    let root = scratch("assembly");
    git(&root, &["init", "-q"]);
    git(&root, &["config", "user.email", "test@xencode.local"]);
    git(&root, &["config", "user.name", "Xencode Test"]);
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/auth.rs"), "fn login() {}\n").unwrap();
    git(&root, &["add", "src/auth.rs"]);
    git(&root, &["commit", "-q", "-m", "initial"]);

    let xencode = root.join(".xencode");
    xencode_context_rs::write_state_candidate(&login_fact(), &xencode).unwrap();
    let (stamped, _) = xencode_context_rs::promote_state_candidate(&xencode).unwrap();
    let durable = std::fs::read_to_string(xencode.join("state.md")).unwrap();
    assert!(
        durable.contains("[src:src/auth.rs@"),
        "promotion wrote the tier without the provenance the reader needs:\n{durable}"
    );
    assert_eq!(stamped.decisions.len(), 1);

    // Before anything changes, the fact is in the prompt the turn sends.
    let live = collect(&root);
    assert!(
        live.state_md
            .as_deref()
            .unwrap()
            .contains("login entry point"),
        "a believed fact was already missing from the tier:\n{durable}"
    );
    assert!(live.stale_state_facts.is_empty());
    assert!(
        turn_mentions_login(&root),
        "the assembled prompt dropped a fact whose file is untouched"
    );

    // The claim is about a file. The file changes. The claim stops being sent,
    // and the turn is told which one went missing.
    std::fs::write(root.join("src/auth.rs"), "fn login_as(user: &str) {}\n").unwrap();
    let live = collect(&root);
    assert!(
        !live
            .state_md
            .as_deref()
            .unwrap_or_default()
            .contains("login entry point"),
        "a fact about a changed file was sent to the model as if it were current"
    );
    assert_eq!(
        live.stale_state_facts.len(),
        1,
        "the fold did not report which durable fact it refused to send"
    );
    assert!(
        live.stale_state_facts[0].contains("the login entry point is src/auth.rs"),
        "the report named the wrong thing: {:?}",
        live.stale_state_facts
    );
    assert!(
        !turn_mentions_login(&root),
        "the assembled prompt still carried the stale fact"
    );

    // And putting the file back brings the fact with it: nothing was deleted,
    // only withheld while it could not be believed.
    std::fs::write(root.join("src/auth.rs"), "fn login() {}\n").unwrap();
    let live = collect(&root);
    assert!(
        live.state_md
            .as_deref()
            .unwrap()
            .contains("login entry point"),
        "reverting the file did not restore a fact that was never destroyed"
    );
    assert!(turn_mentions_login(&root));

    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_tier_written_by_hand_is_sent_exactly_as_its_author_wrote_it() {
    // Nothing in this item may disturb a `state.md` that carries no markers —
    // every file written before QM-2, and every note a person types themselves.
    let root = scratch("unmarked");
    std::fs::write(
        root.join(".xencode/state.md"),
        "# State\n\n## decisions\n- Rust-first for new code\n",
    )
    .unwrap();
    let live = collect(&root);
    assert_eq!(
        live.state_md.as_deref().unwrap(),
        "# State\n\n## decisions\n- Rust-first for new code\n"
    );
    assert!(live.stale_state_facts.is_empty());
    std::fs::remove_dir_all(&root).unwrap();
}
