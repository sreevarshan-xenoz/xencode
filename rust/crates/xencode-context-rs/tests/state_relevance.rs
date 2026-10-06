//! EV-4 at the level a person would notice it: a project that has been promoting
//! durable facts for months has a `state.md` bigger than one turn, and the fact
//! that answers *this* question still reaches the model even when it sits at the
//! bottom of the file.
//!
//! The fixture is a real repository — `git init`, committed source, a promoted
//! `state.md` — because a fact that has no file behind it is not a durable fact,
//! and the provenance stamp it would carry is what the staleness pass reads.

use std::path::{Path, PathBuf};

fn scratch(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let unique = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let root = std::env::temp_dir().join(format!("xencode-relevance-{label}-{unique}"));
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

/// Six committed files, so every filler fact can cite something real and none of
/// them is the file the question below asks about.
fn repo(label: &str) -> PathBuf {
    let root = scratch(label);
    git(&root, &["init", "-q"]);
    git(&root, &["config", "user.email", "test@xencode.local"]);
    git(&root, &["config", "user.name", "Xencode Test"]);
    std::fs::create_dir_all(root.join("src")).unwrap();
    for name in [
        "auth.rs",
        "session.rs",
        "config.rs",
        "ladder.rs",
        "ledger.rs",
        "keys.rs",
    ] {
        std::fs::write(
            root.join("src").join(name),
            format!("pub fn {}() {{}}\n", name.trim_end_matches(".rs")),
        )
        .unwrap();
    }
    git(&root, &["add", "-A"]);
    git(
        &root,
        &[
            "commit",
            "-q",
            "-m",
            "the modules a long project accumulates",
        ],
    );
    root
}

/// `count` promoted facts about ordinary files, then the one the question asks
/// about. Written in that order on purpose: the fact worth injecting is the one a
/// cut from the front of the file never reaches.
fn promote(root: &Path, task: &str, count: usize, target: &str) {
    let mut decisions = Vec::new();
    for n in 0..count {
        let file = ["auth", "session", "config", "ladder", "ledger", "keys"][n % 6];
        decisions.push(format!(
            "filler {n}: the retry ladder in src/{file}.rs waits a second between attempts and \
             gives up after four, and the fourth failure is recorded"
        ));
    }
    decisions.push(target.to_string());
    let xencode = root.join(".xencode");
    xencode_context_rs::write_state_candidate(
        &xencode_context_rs::ContextState {
            working_on: task.to_string(),
            completed: vec![],
            decisions,
            unresolved: vec![],
        },
        &xencode,
    )
    .unwrap();
    xencode_context_rs::promote_state_candidate(&xencode).unwrap();
}

fn turn(root: &Path, ask: &str) -> xencode_context_rs::LiveContext {
    xencode_context_rs::collect_live_context(
        root,
        ask,
        xencode_context_rs::ContextCaps::from_profile(
            xencode_context_rs::HardwareProfile::Balanced,
        ),
    )
}

/// The whole prompt text one turn would send, so a claim about what is injected is
/// checked where the model reads it.
fn prompt(root: &Path, ask: &str) -> String {
    let live = turn(root, ask);
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

const TARGET: &str = "the header row in src/csv_export.rs is written before the first data row \
                      is asked for";

#[test]
fn a_question_reaches_the_fact_at_the_bottom_of_a_store_no_turn_can_hold() {
    let root = repo("bottom");
    std::fs::write(root.join("src/csv_export.rs"), "pub fn csv_export() {}\n").unwrap();
    git(&root, &["add", "-A"]);
    git(
        &root,
        &[
            "commit",
            "-q",
            "-m",
            "the export the next question will be about",
        ],
    );
    promote(&root, "answer why an export has no header", 40, TARGET);

    let ask = "why does the csv export omit a header row?";
    let on_disk = std::fs::read_to_string(root.join(".xencode/state.md")).unwrap();
    assert!(
        on_disk.len() > 3_200,
        "this is not a store bigger than a turn, so it cannot test the choice: {} bytes",
        on_disk.len()
    );
    assert!(
        !live_nothing(&root, ask),
        "the promoted facts were all contradicted, so the tier is empty for a reason \
         unrelated to the budget"
    );

    let live = turn(&root, ask);
    let tier = live.state_md.as_deref().unwrap_or_default().to_string();
    assert!(
        tier.contains("src/csv_export.rs"),
        "the fact this question is about never reached the turn:\n{tier}"
    );
    assert!(
        !tier.contains("filler 39:"),
        "the fact next to the one that matters was sent while it was not, so the tier is \
         still a cut from somewhere:\n{tier}"
    );
    assert!(
        live.state_facts_left_out > 15,
        "a turn that cannot hold the store reported holding nearly all of it: {} sent, \
         {} left out",
        live.state_facts_sent,
        live.state_facts_left_out
    );
    assert!(
        xencode_context_rs::budget::est_tokens(tier.len(), false)
            <= xencode_context_rs::context::STATE_CAP_TOKENS,
        "the chosen facts still came in over the turn's budget: {} bytes",
        tier.len()
    );
    assert!(
        tier.contains("answer why an export has no header"),
        "the tier lost the line saying what the turn is working on:\n{tier}"
    );
    // The model's copy, not the tier in isolation.
    let text = prompt(&root, ask);
    assert!(
        text.contains("src/csv_export.rs"),
        "the fact reached the tier but not the prompt:\n{text}"
    );

    // And what the rule before this module sent: the front of the file, at the same
    // budget. If that had contained the fact, nothing here would be proving a
    // choice was made rather than a lucky size.
    let (head, _) = xencode_context_rs::budget::truncate_to_tokens(
        &on_disk,
        xencode_context_rs::context::STATE_CAP_TOKENS,
        false,
    );
    assert!(
        !head.contains("src/csv_export.rs"),
        "a cut from the front would have found the fact too, so this fixture proves \
         nothing about ranking"
    );
    std::fs::remove_dir_all(&root).unwrap();
}

/// Whether the turn sent any durable fact at all — a store whose every line the
/// code contradicted would pass a `contains` check for the wrong reason.
fn live_nothing(root: &Path, ask: &str) -> bool {
    turn(root, ask).state_facts_sent == 0
}

#[test]
fn a_store_a_turn_can_hold_arrives_untouched() {
    let root = repo("fits");
    std::fs::write(root.join("src/csv_export.rs"), "pub fn csv_export() {}\n").unwrap();
    git(&root, &["add", "-A"]);
    git(
        &root,
        &["commit", "-q", "-m", "a small project exports too"],
    );
    promote(&root, "answer why an export has no header", 2, TARGET);

    let on_disk = std::fs::read_to_string(root.join(".xencode/state.md")).unwrap();
    let live = turn(&root, "why does the csv export omit a header row?");
    let tier = live.state_md.as_deref().unwrap_or_default().to_string();
    assert_eq!(
        tier, on_disk,
        "a store with nothing to choose between was re-laid out by the chooser"
    );
    assert_eq!(live.state_facts_left_out, 0);
    assert_eq!(live.state_facts_sent, 3);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn two_questions_over_one_store_get_two_different_tiers() {
    // The claim the whole module exists for: the file does not change, the turn
    // does. Asking about the ledger gets the ledger fact, and the export fact that
    // won the previous turn is what this one leaves behind.
    let root = repo("twice");
    std::fs::write(root.join("src/csv_export.rs"), "pub fn csv_export() {}\n").unwrap();
    std::fs::write(root.join("src/audit_trail.rs"), "pub fn audit_trail() {}\n").unwrap();
    git(&root, &["add", "-A"]);
    git(
        &root,
        &["commit", "-q", "-m", "two files two questions will name"],
    );
    promote(&root, "answer why an export has no header", 40, TARGET);
    // The fact about the other question: added second, so neither turn can hold it
    // by being first in the file.
    let xencode = root.join(".xencode");
    let mut state = xencode_context_rs::ContextState::from_disk(&xencode).unwrap();
    state.decisions.insert(
        1,
        "the audit trail in src/audit_trail.rs records every retry".to_string(),
    );
    state.write(&xencode).unwrap();

    let first = turn(&root, "why does the csv export omit a header row?");
    let first_tier = first.state_md.clone().unwrap_or_default();
    assert!(
        first_tier.contains("src/csv_export.rs"),
        "the export question did not get the export fact:\n{first_tier}"
    );

    let second = turn(&root, "where is a retry recorded in the audit trail?");
    let second_tier = second.state_md.clone().unwrap_or_default();
    assert!(
        second_tier.contains("src/audit_trail.rs"),
        "the audit question did not get the audit fact:\n{second_tier}"
    );
    assert_ne!(
        first_tier, second_tier,
        "two different questions over one store were sent the same bytes"
    );
    // The file behind both turns is the file they started with: choosing is not
    // editing.
    let after = std::fs::read_to_string(xencode.join("state.md")).unwrap();
    assert!(
        after.contains("src/csv_export.rs") && after.contains("src/audit_trail.rs"),
        "a turn's choice wrote the store down for the next one"
    );
    std::fs::remove_dir_all(&root).unwrap();
}
