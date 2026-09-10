//! Structured `state.md` (M3 — context lifecycle).
//!
//! `state.md` is tier 4 of the assembled prompt and the only file that soft
//! and hard compaction may rewrite. It carries the model's *current* picture
//! of the task, distinct from the immutable `anchor.md` (tier 3) and the
//! canonical transcript (`.xencode/cache/transcript/`).

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

// Section names reused by `to_markdown`, `from_markdown` and the hard-
// compaction reply parser (`compact::parse_hard_compact_reply`).
pub const HDR_WORKING: &str = "## working-on";
pub const HDR_COMPLETED: &str = "## completed";
pub const HDR_DECISIONS: &str = "## decisions";
pub const HDR_UNRESOLVED: &str = "## unresolved";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextState {
    #[serde(default)]
    pub working_on: String,
    #[serde(default)]
    pub completed: Vec<String>,
    #[serde(default)]
    pub decisions: Vec<String>,
    #[serde(default)]
    pub unresolved: Vec<String>,
}

impl ContextState {
    /// Render as `state.md` — what tier 4 of the prompt injects.
    pub fn to_markdown(&self) -> String {
        let mut out = String::from("# State\n\n");
        if !self.working_on.is_empty() {
            out.push_str(HDR_WORKING);
            out.push('\n');
            out.push_str(&self.working_on);
            out.push_str("\n\n");
        }
        push_list(&mut out, HDR_COMPLETED, &self.completed);
        push_list(&mut out, HDR_DECISIONS, &self.decisions);
        push_list(&mut out, HDR_UNRESOLVED, &self.unresolved);
        out.trim_end().to_string()
    }

    pub fn present(&self) -> bool {
        !self.working_on.is_empty()
            || !self.completed.is_empty()
            || !self.decisions.is_empty()
            || !self.unresolved.is_empty()
    }

    /// Parse an existing `state.md`. Missing/unknown sections are tolerated and
    /// dropped, so a hand-written state file survives a round-trip intact.
    pub fn from_markdown(text: &str) -> Self {
        let mut state = ContextState::default();
        let mut section: Option<&str> = None;
        let mut collected: Vec<String> = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("##") {
                commit_section(&mut state, section, &collected);
                collected = Vec::new();
                section = Some(trimmed);
                continue;
            }
            if section.is_some() && !trimmed.is_empty() {
                collected.push(trimmed.to_string());
            }
        }
        commit_section(&mut state, section, &collected);
        state
    }

    /// Load `state.md` from `.xencode` (missing → default empty state).
    pub fn from_disk(xencode_dir: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(xencode_dir.join("state.md")).ok()?;
        Some(Self::from_markdown(&text))
    }

    /// Atomically write `state.md`.
    pub fn write(&self, xencode_dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(xencode_dir)?;
        crate::index::write_str_atomic(&xencode_dir.join("state.md"), &self.to_markdown())
    }
}

fn push_list(out: &mut String, header: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    out.push_str(header);
    out.push('\n');
    for item in items {
        out.push_str("- ");
        out.push_str(item);
        out.push('\n');
    }
    out.push('\n');
}

fn commit_section(state: &mut ContextState, section: Option<&str>, items: &[String]) {
    let Some(section) = section else { return };
    let rest = section.trim_start_matches("##").trim();
    let list: Vec<String> = items
        .iter()
        .map(|s| s.trim_start_matches('-').trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if rest.starts_with("working-on") {
        state.working_on = list.into_iter().next().unwrap_or_default();
    } else if rest.starts_with("completed") {
        state.completed = list;
    } else if rest.starts_with("decisions") {
        state.decisions = list;
    } else if rest.starts_with("unresolved") {
        state.unresolved = list;
    }
}

impl std::fmt::Display for ContextState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_markdown())
    }
}

/// Does `content` carry a `[d]` decision marker (§11)?
pub fn has_decision_marker(content: &str) -> bool {
    content.contains("[d]")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn temp_dir() -> PathBuf {
        // A process-wide counter, not a timestamp. Tests in one binary run in
        // parallel threads and can read the same nanosecond, which had them share
        // a directory and clobber each other's assertions.
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-state-test-{unique}"))
    }

    fn sample() -> ContextState {
        ContextState {
            working_on: "Wire /ctx into the TUI".to_string(),
            completed: vec!["M2 context assembly".to_string()],
            decisions: vec!["Rust-first for new code [d]".to_string()],
            unresolved: vec!["Test on Windows + Linux CI".to_string()],
        }
    }

    #[test]
    fn markdown_round_trips() {
        let state = sample();
        let md = state.to_markdown();
        assert!(md.contains(HDR_WORKING));
        assert!(md.contains(HDR_DECISIONS));
        assert_eq!(ContextState::from_markdown(&md), state);
    }

    #[test]
    fn parses_handwritten_state_with_unknown_sections() {
        let md = concat!(
            "# State\n\n",
            "## working-on\nRewrite the auth flow\n\n",
            "## arbitrary\nnot a real section\n\n",
            "## decisions\n- Use actix-web [d]\n",
            "## unresolved\n- Pin llama.cpp version\n",
        );
        let state = ContextState::from_markdown(md);
        assert_eq!(state.working_on, "Rewrite the auth flow");
        assert_eq!(state.decisions, vec!["Use actix-web [d]"]);
        assert_eq!(state.unresolved, vec!["Pin llama.cpp version"]);
        assert!(state.completed.is_empty());
    }

    #[test]
    fn empty_state_stays_empty_through_round_trip() {
        let state = ContextState::default();
        let md = state.to_markdown();
        assert!(!state.present());
        assert_eq!(ContextState::from_markdown(&md), ContextState::default());
    }

    #[test]
    fn write_then_read_from_disk() {
        let root = temp_dir();
        let xencode = root.join(".xencode");
        sample().write(&xencode).unwrap();
        let loaded = ContextState::from_disk(&xencode).unwrap();
        assert_eq!(loaded, sample());
        std::fs::remove_dir_all(root).unwrap();
    }
}