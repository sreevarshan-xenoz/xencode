//! The instructions this program sends to a model, as named, versioned text.
//!
//! Every prompt that reaches a provider lives in `prompts/` as a plain markdown
//! file, compiled in with [`include_str!`]. Two reasons for that shape. As files,
//! a prompt is a legible diff and a readable thing to point at; compiled in rather
//! than read at runtime, the tier-1 head cannot change underneath a running
//! session, which is what llama.cpp's KV-prefix reuse depends on (§13) — a prompt
//! an editor saved mid-session would quietly cost every later request its cache.
//!
//! A version is a hash of the text, not a number someone remembers to bump. Edit
//! one word and the version moves, so a metric row or an eval score carries the
//! prompt set that actually produced it rather than a claim about it.
//!
//! The joiners a request needs — the blank lines that separate the tool
//! vocabulary from the system prompt, the `Task:` line that follows a brief — stay
//! in code, because they are not prompt text. Each file's last byte is therefore
//! part of what a model reads.

use sha2::{Digest, Sha256};
use std::sync::LazyLock;

pub const AGENT_SYSTEM: &str = include_str!("../prompts/agent-system.md");
pub const TOOLS: &str = include_str!("../prompts/tools.md");
pub const COMPACT_TRANSCRIPT: &str = include_str!("../prompts/compact-transcript.md");
pub const SUBAGENT_BRIEF: &str = include_str!("../prompts/subagent-brief.md");
pub const SUBAGENT_WORKTREE_BRIEF: &str = include_str!("../prompts/subagent-worktree-brief.md");

/// Drop a final newline if an editor added one. The files are written without
/// one, so their last byte is prompt text; this keeps a stray newline from
/// changing what a model is asked.
fn body(text: &str) -> &str {
    text.strip_suffix('\n').unwrap_or(text)
}

/// One prompt: the name it is recorded under, where its text lives, and the exact
/// bytes sent to a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prompt {
    pub name: &'static str,
    pub path: &'static str,
    pub text: &'static str,
}

impl Prompt {
    /// First 8 hexadecimal characters of the SHA-256 of this prompt's name and
    /// text. The name is inside the hash so two prompts with identical wording
    /// still tell each other apart.
    pub fn version(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        hasher.update([0u8]);
        hasher.update(self.text.as_bytes());
        let digest = hasher.finalize();
        let mut out = String::with_capacity(8);
        for byte in &digest[..4] {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }
}

/// Every prompt a model can be asked to obey, in the order a request meets them.
///
/// This list is the registry: a prompt that is not in it has no version, so a
/// change to its wording would show up in an eval comparison as "nothing changed".
/// Adding a prompt means adding it here and giving it a file.
pub fn registry() -> Vec<Prompt> {
    vec![
        Prompt {
            name: "agent-system",
            path: "prompts/agent-system.md",
            text: body(AGENT_SYSTEM),
        },
        Prompt {
            name: "tools",
            path: "prompts/tools.md",
            text: body(TOOLS),
        },
        Prompt {
            name: "compact-transcript",
            path: "prompts/compact-transcript.md",
            text: body(COMPACT_TRANSCRIPT),
        },
        Prompt {
            name: "subagent-brief",
            path: "prompts/subagent-brief.md",
            text: body(SUBAGENT_BRIEF),
        },
        Prompt {
            name: "subagent-worktree-brief",
            path: "prompts/subagent-worktree-brief.md",
            text: body(SUBAGENT_WORKTREE_BRIEF),
        },
    ]
}

/// The version of the whole set: moves when any prompt's text moves, and only
/// then. Twelve hex characters stay readable in a metrics row while leaving room
/// to treat a collision as impossible in practice.
pub fn set_version() -> &'static str {
    &SET_VERSION
}

static SET_VERSION: LazyLock<String> = LazyLock::new(|| digest_of(&registry()));

/// The digest for a set of prompts. Split out so a test can ask what the version
/// would be after an edit, rather than copying this arithmetic.
fn digest_of(prompts: &[Prompt]) -> String {
    let mut hasher = Sha256::new();
    for prompt in prompts {
        hasher.update(prompt.name.as_bytes());
        hasher.update([0u8]);
        hasher.update(prompt.version().as_bytes());
        hasher.update([0u8]);
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(12);
    for byte in &digest[..6] {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// The system turn with the tool vocabulary on the end of it, byte for byte as it
/// goes over the wire. Riding on the assembled system message rather than being
/// assembled into it is what keeps the stable prefix intact (§13).
pub fn append_tool_hint(text: &mut String) {
    text.push_str("\n\n");
    text.push_str(body(TOOLS));
}

/// The instructions for a delegated run and the task it was given. Deliberately
/// short: a delegated run gets the same system turn a chat turn does, so the tool
/// vocabulary is already in front of it.
pub fn subagent_brief(task: &str) -> String {
    format!("{}\n\nTask: {}", body(SUBAGENT_BRIEF), task)
}

/// The same for a run that works in its own git worktree.
pub fn worktree_brief(task: &str) -> String {
    format!("{}\n\nTask: {}", body(SUBAGENT_WORKTREE_BRIEF), task)
}

/// Fill the three holes in the transcript-folding prompt. Replacement rather
/// than a formatting macro, so the file's own braces and angle brackets in the
/// markdown shape it asks for are just text.
pub fn compaction_prompt(state: &str, transcript_tail: &str, recent: &str) -> String {
    body(COMPACT_TRANSCRIPT)
        .replace("{state}", state)
        .replace("{transcript_tail}", transcript_tail)
        .replace("{recent}", recent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_prompt_is_named_non_empty_and_unique() {
        let prompts = registry();
        assert_eq!(prompts.len(), 5, "the registry covers five prompts");
        let mut names: Vec<&str> = prompts.iter().map(|p| p.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), prompts.len(), "a name is used twice");
        for prompt in &prompts {
            assert!(!prompt.text.is_empty(), "{} has no text", prompt.name);
            // A prompt that ends in whitespace means a file whose convention
            // leaked into the bytes sent.
            assert_eq!(
                prompt.text.trim_end(),
                prompt.text,
                "{} carries trailing whitespace",
                prompt.name
            );
        }
    }

    #[test]
    fn a_registered_prompt_matches_the_file_it_claims() {
        // The paths are written by hand next to text pulled in by `include_str!`,
        // so a renamed or edited-out file would otherwise leave the registry
        // quoting a source that no longer says what was sent.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for prompt in registry() {
            let on_disk = std::fs::read_to_string(root.join(prompt.path))
                .unwrap_or_else(|e| panic!("{} names an unreadable file: {e}", prompt.name));
            assert_eq!(
                body(&on_disk),
                prompt.text,
                "{} does not match {}",
                prompt.name,
                prompt.path
            );
        }
    }

    #[test]
    fn a_prompt_version_moves_only_when_its_text_or_name_moves() {
        let prompts = registry();
        let agent = prompts.iter().find(|p| p.name == "agent-system").unwrap();
        let original = agent.version();
        assert_eq!(original, agent.version(), "the same bytes twice");

        let edited = Prompt {
            text: "You are Xencode, a coding agent.",
            ..*agent
        };
        assert_ne!(
            original,
            edited.version(),
            "a reworded prompt still says so"
        );

        let renamed = Prompt {
            name: "agent-system-v2",
            ..*agent
        };
        assert_ne!(original, renamed.version(), "a rename is a new prompt");
    }

    #[test]
    fn the_set_version_follows_the_prompt_it_covers() {
        let before = set_version().to_string();
        assert_eq!(before, set_version(), "the set version is computed once");
        assert_eq!(before.len(), 12, "short enough to read in a metrics row");
        // Recompute what the digest would be with one word changed, the way a
        // commit to prompts/agent-system.md would.
        let mut prompts = registry();
        prompts
            .iter_mut()
            .find(|p| p.name == "agent-system")
            .unwrap()
            .text = "You are Xencode.";
        assert_ne!(before, digest_of(&prompts));
    }

    #[test]
    fn the_tool_hint_lands_on_the_system_turn_the_way_it_always_did() {
        let mut text = "You are Xencode, a coding agent.".to_string();
        append_tool_hint(&mut text);
        assert!(
            text.starts_with("You are Xencode, a coding agent.\n\n## Tools\n"),
            "the joiner between the system prompt and the tool list changed"
        );
        assert!(text.contains("update_plan(items=[{text,status}])"));
        assert!(
            !text.ends_with('\n'),
            "the file's final newline stays in the file"
        );
    }

    #[test]
    fn a_brief_is_the_instructions_with_the_task_appended() {
        for (built, file) in [
            (subagent_brief("Fix the failing test"), SUBAGENT_BRIEF),
            (
                worktree_brief("Fix the failing test"),
                SUBAGENT_WORKTREE_BRIEF,
            ),
        ] {
            assert!(
                built.starts_with(body(file)),
                "the brief no longer opens with its own file"
            );
            assert!(
                built.ends_with("\n\nTask: Fix the failing test"),
                "the task stopped riding on the `Task:` joiner"
            );
            assert_eq!(
                built,
                format!("{}\n\nTask: Fix the failing test", body(file)),
                "the joiner between a brief and its task changed"
            );
        }
        // The two briefs differ only in whether the run is isolated, so a job that
        // asks for a worktree cannot be handed the shared-checkout wording.
        assert_ne!(
            subagent_brief("x"),
            worktree_brief("x"),
            "both briefs collapsed onto the same text"
        );
    }

    #[test]
    fn the_compaction_prompt_has_no_holes_left_after_filling() {
        let filled = compaction_prompt("# Current state\n- working-on: x", "user: hi", "user: hi");
        for hole in ["{state}", "{transcript_tail}", "{recent}"] {
            assert!(!filled.contains(hole), "{hole} was never filled in");
        }
        assert!(filled.contains("# Current state\n- working-on: x"));
        assert!(filled.contains("## recent (last 6 messages, verbatim)\nuser: hi"));
    }
}
