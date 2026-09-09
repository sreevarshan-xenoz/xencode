//! Layered soft vs hard compaction (M3 — context lifecycle, spec §11).
//!
//! ```text
//! trigger usage ≥ 70% → SOFT  (deterministic, no LLM call)
//! trigger usage ≥ 90% → HARD  (one model call → layered summary)
//! ```
//!
//! The canonical transcript is never destroyed — both paths snapshot it to
//! `.xencode/cache/transcript/<ts>.json` before rewriting `current.json`
//! (§11 "Raw transcript is dumped … before rewrite") and only the *working
//! projection* (the prompt built per request) is reduced.

use crate::conversation::Transcript;
use crate::state::ContextState;
use std::path::PathBuf;

pub const SOFT_THRESHOLD: f32 = 0.70;
pub const HARD_THRESHOLD: f32 = 0.90;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionKind {
    Soft,
    Hard,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactReport {
    pub kind: Option<CompactionKind>,
    pub before: usize,
    pub after: usize,
    /// Entries dropped from the working projection.
    pub dropped: usize,
    /// Decision-marked entries retained because they must survive.
    pub retained_decisions: usize,
    /// Snapshot written before the rewrite (None when nothing changed).
    pub snapshot: Option<PathBuf>,
    pub state_rebuilt: bool,
}

/// Map context usage onto a compaction action (§11).
pub fn should_compact(usage_pct: f32) -> Option<CompactionKind> {
    if usage_pct >= HARD_THRESHOLD {
        Some(CompactionKind::Hard)
    } else if usage_pct >= SOFT_THRESHOLD {
        Some(CompactionKind::Soft)
    } else {
        None
    }
}

/// Deterministic soft compaction: drop the oldest 30% of entries unless they
/// carry a `[d]` decision marker, which always survive regardless of age.
/// No LLM call. Returns a report so callers can surface it in `state.md` or UI.
pub fn soft_compact(transcript: &mut Transcript, keep_fraction: f32) -> CompactReport {
    let before = transcript.entries.len();
    let keep_fraction = keep_fraction.clamp(0.0, 1.0);
    let keep_target = (before as f32 * keep_fraction).ceil() as usize;
    let keep_cut = keep_target.clamp(1, before);

    // Single chronological pass: keep the most recent `keep_cut` entries by
    // position, plus any older entry carrying a `[d]` decision marker. No sort
    // is needed, so identical-millisecond timestamps can't reorder history.
    let recent_start = before.saturating_sub(keep_cut);
    let retained: Vec<crate::conversation::TranscriptEntry> = transcript
        .entries
        .iter()
        .enumerate()
        .filter(|(i, e)| *i >= recent_start || e.is_decision)
        .map(|(_, e)| e.clone())
        .collect();

    let after = retained.len();
    transcript.entries = retained;
    transcript.updated_at_unix_ms = crate::conversation::now_millis();

    CompactReport {
        kind: Some(CompactionKind::Soft),
        before,
        after,
        dropped: before.saturating_sub(after),
        retained_decisions: transcript
            .entries
            .iter()
            .filter(|e| e.is_decision)
            .count(),
        snapshot: None,
        state_rebuilt: false,
    }
}

/// Build the single model call that folds the transcript into a layered
/// summary (§11 layer separation). Output shape mirrors `state.md` sections so
/// `parse_hard_compact_reply` can rebuild the state deterministically.
pub fn hard_compact_prompt(state: &ContextState, transcript: &Transcript) -> String {
    let recent = transcript
        .recent(6)
        .iter()
        .map(|e| format!("{}: {}", e.role, e.content))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You are compressing a working coding conversation into its durable layers. \
Keep immutable facts, decisions, the current task, completed work, and unresolved issues. \
Never invent facts. Keep the last 6 messages verbatim.\n\n\
# Current state\n{}\n\n\
# Transcript tail (this is the working window being folded)\n{TRANSCRIPT_TAIL}\n\n\
Return ONLY this exact markdown shape:\n\n\
## working-on\n<one parallel sentence>\n\n\
## completed\n- <item>\n\n\
## decisions\n- <decision [d]>\n\n\
## unresolved\n- <item>\n\n\
## recent (last 6 messages, verbatim)\n{recent}",
        if state.present() { state.to_markdown() } else { "(empty)".to_string() },
        TRANSCRIPT_TAIL = transcript_tail(transcript),
    )
}

fn transcript_tail(transcript: &Transcript) -> String {
    transcript
        .recent(6)
        .iter()
        .map(|e| format!("{}: {}", e.role, e.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse a hard-compaction reply into `ContextState`. The `## recent` section
/// is *not* folded into state (it becomes the new tier-7 window instead).
pub fn parse_hard_compact_reply(reply: &str) -> ContextState {
    // Strip the `## recent` section before reusing the state.md parser so it
    // can't land in an unknown bucket.
    let mut without_recent = String::new();
    let mut in_recent = false;
    for line in reply.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("## recent") {
            in_recent = true;
            continue;
        }
        if in_recent {
            if trimmed.starts_with("##") {
                in_recent = false;
            } else {
                continue;
            }
        }
        without_recent.push_str(line);
        without_recent.push('\n');
    }
    ContextState::from_markdown(&without_recent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::Transcript;

    fn seeded() -> Transcript {
        let mut t = Transcript::new("s1");
        for i in 0..10 {
            let marker = if i == 1 { " [d]" } else { "" };
            t.add("user", &format!("message {i}{marker}"));
            t.add("assistant", &format!("reply {i}"));
        }
        t
    }

    #[test]
    fn thresholds_map_to_actions() {
        assert_eq!(should_compact(0.50), None);
        assert_eq!(should_compact(0.70), Some(CompactionKind::Soft));
        assert_eq!(should_compact(0.89), Some(CompactionKind::Soft));
        assert_eq!(should_compact(0.90), Some(CompactionKind::Hard));
        assert_eq!(should_compact(0.97), Some(CompactionKind::Hard));
    }

    #[test]
    fn soft_compact_drops_oldest_30pct_but_keeps_decisions() {
        let mut t = seeded();
        assert_eq!(t.entries.len(), 20);
        let report = soft_compact(&mut t, 0.70);
        // 20 * 0.70 = 14 kept from the tail; the old decision (message 1)
        // sits inside the dropped range but is preserved → 15 kept.
        assert_eq!(report.dropped, 5);
        assert_eq!(t.entries.len(), 15);
        // The old decision (message 1, is_decision) survived despite age.
        assert!(t
            .entries
            .iter()
            .any(|e| e.is_decision && e.content.contains("message 1")));
        // Recent entries survived.
        assert!(t.entries.last().unwrap().content.contains("reply 9"));
    }

    #[test]
    fn soft_compact_keeps_at_least_one_entry() {
        let mut t = Transcript::new("s1");
        t.add("user", "only");
        let report = soft_compact(&mut t, 0.0);
        assert_eq!(t.entries.len(), 1);
        assert_eq!(report.dropped, 0);
    }

    #[test]
    fn hard_compact_prompt_contains_core_sections() {
        let mut t = Transcript::new("s1");
        t.add("user", "P1 [d]");
        t.add("assistant", "A1");
        let state = ContextState {
            working_on: "Fix auth".to_string(),
            completed: vec![],
            decisions: vec![],
            unresolved: vec![],
        };
        let p = hard_compact_prompt(&state, &t);
        assert!(p.contains("## working-on"));
        assert!(p.contains("## completed"));
        assert!(p.contains("## decisions"));
        assert!(p.contains("## unresolved"));
        assert!(p.contains("A1"));
    }

    #[test]
    fn parse_hard_compact_reply_round_trips_state() {
        let reply = "\
## working-on
Wire /ctx compact into the TUI

## completed
- M2 context assembly
- M3 starts

## decisions
- Rust-first [d]

## unresolved
- Windows CI

## recent (last 6 messages, verbatim)
user: hi
assistant: hello";
        let state = parse_hard_compact_reply(reply);
        assert_eq!(state.working_on, "Wire /ctx compact into the TUI");
        assert_eq!(state.completed, vec!["M2 context assembly", "M3 starts"]);
        assert_eq!(state.decisions, vec!["Rust-first [d]"]);
        assert_eq!(state.unresolved, vec!["Windows CI"]);
        assert!(state.present());
    }
}