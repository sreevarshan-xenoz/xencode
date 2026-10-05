//! QM-2 and MEM-3, checked at the level a person would notice: a durable fact
//! stops reaching the prompt the moment the file it describes changes, or the
//! code it names stops existing.
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
        live.scoped_md.as_deref(),
        live.state_md.as_deref(),
        None,
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
        live.stale_state_facts[0]
            .line
            .contains("the login entry point is src/auth.rs"),
        "the report named the wrong thing: {:?}",
        live.stale_state_facts
    );
    assert_eq!(
        live.stale_state_facts[0].problem,
        xencode_context_rs::FactProblem::SourceChanged,
        "an edited file is reported as some other kind of staleness"
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

/// The prompt text one turn would actually send, so a claim about "dropped at
/// inject time" is checked where the model receives it.
fn prompt(root: &Path, ask: &str) -> String {
    let live = collect(root);
    xencode_context_rs::assemble_prompt(
        xencode_context_rs::HardwareProfile::Balanced,
        "You are a coding agent.",
        live.agents_md.as_deref(),
        live.anchor_md.as_deref(),
        live.scoped_md.as_deref(),
        live.state_md.as_deref(),
        None,
        &live.git_summary,
        &live.repo_map,
        vec![],
        ask,
    )
    .text
}

#[test]
fn a_fact_naming_code_leaves_the_prompt_when_that_code_stops_being_true() {
    // MEM-3's done-when at the level a person would notice it. Two facts about two
    // different kinds of claim, plus a sentence of prose that must survive both
    // stages — collateral damage here is the failure nobody sees coming.
    let root = scratch("checks");
    git(&root, &["init", "-q"]);
    git(&root, &["config", "user.email", "test@xencode.local"]);
    git(&root, &["config", "user.name", "Xencode Test"]);
    std::fs::create_dir_all(root.join("src")).unwrap();
    // The call and its target are kept in separate files on purpose: a declaration
    // mentioning a name is not a call, and one file would let the symbol check
    // vouch for a call that had already been deleted next to it.
    std::fs::write(root.join("src/auth.rs"), "pub fn validate_token() {}\n").unwrap();
    std::fs::write(
        root.join("src/handlers.rs"),
        "fn reject_request() { crate::auth::validate_token(); }\n",
    )
    .unwrap();
    git(&root, &["add", "-A"]);
    git(
        &root,
        &["commit", "-q", "-m", "token check and the request path"],
    );

    let xencode = root.join(".xencode");
    xencode_context_rs::write_state_candidate(
        &xencode_context_rs::ContextState {
            working_on: "answer what the request path does".to_string(),
            completed: vec![],
            decisions: vec![
                "validate_token rejects an empty token".to_string(),
                "reject_request calls validate_token".to_string(),
                "Rust-first for new code, decided in review".to_string(),
            ],
            unresolved: vec![],
        },
        &xencode,
    )
    .unwrap();
    let (_, report) = xencode_context_rs::promote_state_candidate(&xencode).unwrap();
    let durable = std::fs::read_to_string(xencode.join("state.md")).unwrap();
    assert_eq!(
        report.checks_recorded, 2,
        "the fold under-reported how many lines it had given a re-checkable \
         claim:\n{durable}"
    );
    assert!(
        durable.contains("reject_request>validate_token"),
        "the call was never recorded, so nothing could ever find it missing:\n{durable}"
    );
    assert!(
        !durable
            .lines()
            .find(|line| line.contains("Rust-first"))
            .unwrap()
            .contains("[chk:"),
        "a sentence of prose was marked as a claim about the code:\n{durable}"
    );

    let before = prompt(&root, "what happens to a bad request?");
    assert!(
        before.contains("validate_token rejects an empty token")
            && before.contains("reject_request calls validate_token"),
        "unmoved code was refused its own facts:\n{before}"
    );

    // ── stage 1: the call stops happening, both names still declared ───────
    std::fs::write(root.join("src/handlers.rs"), "fn reject_request() {}\n").unwrap();
    git(&root, &["add", "-A"]);
    git(
        &root,
        &[
            "commit",
            "-q",
            "-m",
            "the request path no longer checks the token",
        ],
    );
    let live = collect(&root);
    assert_eq!(
        live.stale_state_facts.len(),
        1,
        "stage 1 should drop the call and nothing else: {:?}",
        live.stale_state_facts
    );
    assert_eq!(
        live.stale_state_facts[0].problem,
        xencode_context_rs::FactProblem::CallGone,
        "reported as the wrong problem: {:?}",
        live.stale_state_facts[0]
    );
    let after = prompt(&root, "what happens to a bad request?");
    assert!(
        !after.contains("reject_request calls validate_token"),
        "the falsified call is still in the prompt the turn sends"
    );
    assert!(
        after.contains("validate_token rejects an empty token"),
        "a fact that is still true was dropped as collateral damage"
    );
    assert!(
        after.contains("Rust-first for new code"),
        "the prose decision left the prompt with the line next to it"
    );

    // ── stage 2: the name itself goes away ─────────────────────────────────
    std::fs::write(root.join("src/auth.rs"), "pub fn check_token() {}\n").unwrap();
    git(&root, &["add", "-A"]);
    git(&root, &["commit", "-q", "-m", "rename the token check"]);
    let live = collect(&root);
    assert_eq!(
        live.stale_state_facts.len(),
        2,
        "renaming the symbol should take both facts with it: {:?}",
        live.stale_state_facts
    );
    assert_eq!(
        live.stale_state_facts[0].problem,
        xencode_context_rs::FactProblem::SymbolGone,
        "a renamed symbol is reported as something other than gone: {:?}",
        live.stale_state_facts[0]
    );
    let after = prompt(&root, "what happens to a bad request?");
    assert!(
        !after.contains("validate_token"),
        "a fact naming a symbol this code no longer declares reached the model:\n{after}"
    );
    assert!(
        after.contains("Rust-first for new code"),
        "nothing but the code-shaped facts may leave"
    );
    assert!(
        std::fs::read_to_string(xencode.join("state.md"))
            .unwrap()
            .contains("validate_token rejects an empty token"),
        "the durable file was edited instead of the turn being filtered"
    );

    std::fs::remove_dir_all(&root).unwrap();
}
