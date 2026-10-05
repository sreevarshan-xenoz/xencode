//! EV-6, at the level a person would notice: a note the agent wrote for itself is
//! in the prompt the next turn sends, and it is still there after the conversation
//! around it has been compacted.
//!
//! Everything here is the shipped path. The note is written by
//! [`xencode_context_rs::append_note`] into a real `.xencode/notes.md`, the context
//! is gathered by [`xencode_context_rs::collect_live_context`] reading that real
//! file, and the compaction is [`xencode_context_rs::soft_compact`] — the same
//! deterministic pass that drops old transcript entries and keeps `[d]` lines —
//! followed by the real hard-compaction prompt. No transcript is faked into a
//! prompt: the assertion that matters is that the note arrives from disk.

use std::path::{Path, PathBuf};

fn scratch(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!(
        "xencode-notes-tier-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(root.join(".xencode")).unwrap();
    root
}

fn collect(root: &Path) -> xencode_context_rs::LiveContext {
    xencode_context_rs::collect_live_context(
        root,
        "why did the worker exit?",
        xencode_context_rs::ContextCaps::from_profile(
            xencode_context_rs::HardwareProfile::Balanced,
        ),
    )
}

fn turn(root: &Path, recent: &str) -> String {
    let live = collect(root);
    xencode_context_rs::assemble_prompt(
        xencode_context_rs::HardwareProfile::Balanced,
        "You are a coding agent.",
        live.agents_md.as_deref(),
        live.anchor_md.as_deref(),
        None,
        live.state_md.as_deref(),
        live.notes_md.as_deref(),
        &live.git_summary,
        &live.repo_map,
        vec![],
        recent,
    )
    .text
}

const NOTE: &str = "the worker holds the migration lock, so a second one exits at once";

#[test]
fn a_note_the_agent_wrote_reaches_the_next_turns_prompt() {
    let root = scratch("assembly");
    let xencode = root.join(".xencode");

    // Nothing written yet: no tier, no header, nothing invented.
    let empty = turn(&root, "user: why did the worker exit?");
    assert!(
        !empty.contains("## Notes To Self"),
        "an empty scratchpad still bought a tier:\n{empty}"
    );

    xencode_context_rs::append_note(&xencode, NOTE).unwrap();
    let live = collect(&root);
    assert_eq!(
        live.notes_md.as_deref().map(|t| t.contains(NOTE)),
        Some(true),
        "collect_live_context did not read the file write_note wrote"
    );

    let text = turn(&root, "user: why did the worker exit?");
    assert!(
        text.contains("## Notes To Self"),
        "the note tier is missing from the prompt:\n{text}"
    );
    assert!(
        text.contains(NOTE),
        "the note itself did not reach the prompt:\n{text}"
    );
    // It sits beside the durable tier and above git, not buried in history:
    // the order is what tells the model whose words these are.
    let live_doc = {
        let live = collect(&root);
        xencode_context_rs::assemble_prompt(
            xencode_context_rs::HardwareProfile::Balanced,
            "You are a coding agent.",
            live.agents_md.as_deref(),
            live.anchor_md.as_deref(),
            None,
            live.state_md.as_deref(),
            live.notes_md.as_deref(),
            &live.git_summary,
            &live.repo_map,
            vec![],
            "user: why did the worker exit?",
        )
    };
    let names: Vec<&str> = live_doc.tiers.iter().map(|t| t.name).collect();
    let notes_at = names
        .iter()
        .position(|n| *n == "notes.md")
        .unwrap_or_else(|| panic!("no notes.md tier was budgeted: {names:?}"));
    assert!(
        names.iter().take(notes_at).any(|n| *n == "system"),
        "the notes tier was budgeted ahead of the stable head: {names:?}"
    );
    assert_eq!(
        live_doc
            .tiers
            .iter()
            .find(|t| t.name == "notes.md")
            .unwrap()
            .class,
        xencode_context_rs::SourceClass::Scratchpad,
        "the tier reports someone else's provenance for the agent's own words"
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_note_survives_the_compaction_that_eats_the_turn_that_wrote_it() {
    use xencode_context_rs::Transcript;

    let root = scratch("compact");
    let xencode = root.join(".xencode");
    xencode_context_rs::append_note(&xencode, NOTE).unwrap();

    // A conversation whose middle — including the turn that wrote the note — is
    // old enough for compaction to remove.
    let mut transcript = Transcript::new("s1");
    transcript.add("user", "start the worker");
    transcript.add("assistant", &format!("noted it: {NOTE}"));
    for i in 0..12 {
        transcript.add("user", &format!("question {i}"));
        transcript.add("assistant", &format!("answer {i}"));
    }

    let before = transcript.entries.len();
    let report = xencode_context_rs::soft_compact(&mut transcript, 0.5);
    assert!(report.dropped > 0, "compaction had nothing to drop");
    assert!(
        transcript.entries.len() < before,
        "the transcript kept every entry"
    );
    assert!(
        !transcript.entries.iter().any(|e| e.content.contains(NOTE)),
        "the test is not proving anything: the turn that said the note survived"
    );

    // The note is still sent, because it never lived in the transcript.
    let recent = transcript
        .entries
        .iter()
        .map(|e| format!("{}: {}", e.role, e.content))
        .collect::<Vec<_>>()
        .join("\n");
    let text = turn(&root, &recent);
    assert!(
        text.contains(NOTE),
        "a note died with the conversation that produced it:\n{text}"
    );

    // And the hard fold is handed the pad rather than letting the cap lose it:
    // the fold's output is a candidate a person promotes, so this is the note's
    // only route into the durable tier.
    let state = xencode_context_rs::believed_state(&xencode);
    let prompt = xencode_context_rs::hard_compact_prompt(&state, &transcript, Some(NOTE));
    assert!(
        prompt.contains("# Notes the agent kept for itself"),
        "the folding prompt does not ask about the scratchpad:\n{prompt}"
    );
    assert!(
        prompt.contains(NOTE),
        "the note was left out of the fold that replaces the conversation:\n{prompt}"
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn the_tier_carries_the_newest_notes_and_the_fold_still_gets_the_whole_pad() {
    let root = scratch("cap");
    let xencode = root.join(".xencode");
    // Wide enough that the tier's own budget cannot carry all of it.
    for i in 0..xencode_context_rs::NOTES_MAX_LINES {
        xencode_context_rs::append_note(&xencode, &format!("note {i}: {}", "detail ".repeat(20)))
            .unwrap();
    }
    let live = collect(&root);
    let notes = live.notes_md.as_deref().expect("the pad was written");
    let all = xencode_context_rs::note_lines(notes);
    assert_eq!(all.len(), xencode_context_rs::NOTES_MAX_LINES);

    let doc = xencode_context_rs::assemble_prompt(
        xencode_context_rs::HardwareProfile::Balanced,
        "You are a coding agent.",
        None,
        None,
        None,
        None,
        Some(notes),
        "",
        "",
        vec![],
        "user: keep going",
    );
    let tier = doc
        .tiers
        .iter()
        .find(|t| t.name == "notes.md")
        .expect("a full pad is budgeted");
    assert!(
        tier.tokens <= xencode_context_rs::context::NOTES_CAP_TOKENS,
        "the notes tier went over its own cap: {}",
        tier.tokens
    );
    assert!(
        doc.text.contains("note 39:"),
        "the newest note is not the one the pad kept"
    );
    assert!(
        !doc.text.contains("note 0:"),
        "the pad carried every note anyway, so the cap is not a cap"
    );
    // The fold sees the file, not the truncated tier — otherwise the cap would be
    // silent loss instead of a bound on what one turn reads.
    let transcript = xencode_context_rs::Transcript::new("s1");
    let prompt = xencode_context_rs::hard_compact_prompt(
        &xencode_context_rs::ContextState::default(),
        &transcript,
        Some(notes),
    );
    assert!(
        prompt.contains("note 0:"),
        "the fold was handed the truncated tier and the oldest notes vanished"
    );
    std::fs::remove_dir_all(&root).unwrap();
}
