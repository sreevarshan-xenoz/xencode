//! Context assembly (M2) — §10 of the architecture spec.
//!
//! Turns profile + stable layer files + retrieval candidates + recent
//! conversation into the actual prompt for the model, applying the token
//! budget algorithm. Tiers 1–3 form the byte-stable prefix below which the
//! KV cache can be reused across requests; nothing dynamic may precede it.
//!
//! Pure/disk-free (`assemble_prompt` takes strings and pre-built fenced
//! blocks), which keeps the budget logic unit-testable without a repository.

use crate::budget::{est_tokens, truncate_tail_to_tokens, truncate_to_tokens, HardwareProfile};
use crate::gitinfo::{current_git_info, dirty_paths};
use crate::index::FileEntry;
use crate::retrieve::{retrieve, RetrievalIndex, RetrieveOptions, RetrievedFile};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::Path;

/// Per-tier token caps (§10).
pub const AGENTS_CAP_TOKENS: u64 = 1200;
pub const ANCHOR_CAP_TOKENS: u64 = 2000;
pub const STATE_CAP_TOKENS: u64 = 800;
pub const GIT_CAP_TOKENS: u64 = 300;
/// Tiers 4–6 stop consuming once less than this remains for recent messages.
pub const MARGIN_TOKENS: u64 = 200;
/// Below this much room for recent messages → soft compaction should trigger.
pub const RECENT_MIN_TOKENS: u64 = 40;

/// The marker that closes the stable prefix (byte-identical every request).
pub const STABLE_END_MARKER: &str = "<!-- xencode:stable-prefix-end -->";

/// Frozen identity line every model request starts with (tier 1 head).
pub const AGENT_SYSTEM_PROMPT: &str =
    "You are Xencode, a coding agent. Follow the project guidelines below exactly.";

/// A retrieved file's fenced body, ready to inject.
#[derive(Debug, Clone)]
pub struct RetrievedBlock {
    pub path: String,
    pub score: u64,
    /// Already rendered: `File: <path>\n```<lang>\n<content>\n````.
    pub body: String,
}

/// One consumed budget tier (for metrics/meter display).
#[derive(Debug, Clone)]
pub struct TierDoc {
    pub name: &'static str,
    pub tokens: u64,
}

/// Result of [`assemble_prompt`].
#[derive(Debug, Clone)]
pub struct ContextDoc {
    /// The full assembled prompt (stable prefix first).
    pub text: String,
    /// Byte-stable head: SYSTEM + AGENTS.md + anchor.md + end marker.
    pub stable_prefix: String,
    pub target_tokens: u64,
    pub total_tokens: u64,
    pub tiers: Vec<TierDoc>,
    /// Any tier hit its cap (retrieved trimmed / recent dropped / stable overflowed).
    pub truncated: bool,
    /// Tier 7 got less than roughly one message → caller should soft compact.
    pub soft_compaction_needed: bool,
    pub retrieved_included: usize,
    pub retrieved_total: usize,
}

impl ContextDoc {
    /// The KV-reuse contract (§13): `SYSTEM + AGENTS.md + anchor.md` (the
    /// stable prefix) must be byte-identical across requests and precede every
    /// dynamic tier. Returns the SHA-256 of that head so a drift turns into a
    /// loud comparison failure instead of a silent KV-cache miss.
    pub fn stable_prefix_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.stable_prefix.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Tiers 1–3 (§10): SYSTEM + AGENTS.md + anchor.md + end marker.
///
/// Shared by the `/ctx` text preview ([`assemble_prompt`]) and the live chat
/// assembly ([`assemble_chat`]) so both emit a byte-identical head — the
/// precondition for llama.cpp KV-prefix reuse (§13).
struct StableHead {
    prefix: String,
    tokens: u64,
    system_tokens: u64,
    agents_tokens: u64,
    anchor_tokens: u64,
    truncated: bool,
}

fn stable_head(system: &str, agents_md: Option<&str>, anchor_md: Option<&str>) -> StableHead {
    let system_tokens = est_tokens(system.len(), false);
    let (agents_head, agents_tokens) =
        truncate_to_tokens(agents_md.unwrap_or(""), AGENTS_CAP_TOKENS, false);
    let truncated_agents = agents_md.is_some_and(|a| a.len() > agents_head.len());
    let (anchor_head, anchor_tokens) =
        truncate_to_tokens(anchor_md.unwrap_or(""), ANCHOR_CAP_TOKENS, false);
    let truncated_anchor = anchor_md.is_some_and(|a| a.len() > anchor_head.len());

    let mut parts: Vec<&str> = Vec::new();
    if !system.is_empty() {
        parts.push(system);
    }
    if !agents_head.is_empty() {
        parts.push(&agents_head);
    }
    if !anchor_head.is_empty() {
        parts.push(&anchor_head);
    }
    parts.push(STABLE_END_MARKER);
    StableHead {
        prefix: parts.join("\n\n"),
        tokens: system_tokens + agents_tokens + anchor_tokens,
        system_tokens,
        agents_tokens,
        anchor_tokens,
        truncated: truncated_agents || truncated_anchor,
    }
}

/// The frozen identity block every model request starts with: system prompt +
/// project guidelines + anchor, closed by [`STABLE_END_MARKER`].
///
/// Byte-identical to the head [`assemble_chat`] sends as its `system` turn,
/// so single-purpose calls (e.g. code review) reuse the same cached prefix.
pub fn stable_system_text(
    system: &str,
    agents_md: Option<&str>,
    anchor_md: Option<&str>,
) -> String {
    stable_head(system, agents_md, anchor_md).prefix
}

/// Build the promoted-tier prompt per §10.
///
/// `retrieved` must already be sorted best-first; the budgeter trims from the
/// bottom. `recent_text` should be the rolling message window, oldest first.
#[allow(clippy::too_many_arguments)] // eight positional args are the documented contract (§10 tiers)
pub fn assemble_prompt(
    profile: HardwareProfile,
    system: &str,
    agents_md: Option<&str>,
    anchor_md: Option<&str>,
    state_md: Option<&str>,
    git_summary: &str,
    retrieved: Vec<RetrievedBlock>,
    recent_text: &str,
) -> ContextDoc {
    let target = (profile.ctx_tokens() as f64 * profile.utilization()).floor() as u64;
    let mut tiers: Vec<TierDoc> = Vec::new();
    let mut truncated = false;

    // ── Tiers 1–3: stable prefix ─────────────────────────────────────────
    let stable = stable_head(system, agents_md, anchor_md);
    tiers.push(TierDoc {
        name: "system",
        tokens: stable.system_tokens,
    });
    tiers.push(TierDoc {
        name: "agents.md",
        tokens: stable.agents_tokens,
    });
    tiers.push(TierDoc {
        name: "anchor.md",
        tokens: stable.anchor_tokens,
    });

    let stable_prefix = stable.prefix;
    let mut remaining = target.saturating_sub(stable.tokens);
    truncated |= stable.truncated || stable.tokens > target;

    // ── Tier 4: state.md ─────────────────────────────────────────────────
    // Admitted only with margin to spare; the emit below follows the same
    // flag so text and budget can never diverge (unemitted text would bust
    // the margin the gate protects).
    let (state_head, state_tok) =
        truncate_to_tokens(state_md.unwrap_or(""), STATE_CAP_TOKENS, false);
    let state_included = !state_head.is_empty() && remaining >= MARGIN_TOKENS;
    if state_included {
        tiers.push(TierDoc {
            name: "state.md",
            tokens: state_tok,
        });
        remaining = remaining.saturating_sub(state_tok);
    }

    // ── Tier 5: git summary ──────────────────────────────────────────────
    let (git_head, git_tok) = truncate_to_tokens(git_summary, GIT_CAP_TOKENS, false);
    let git_included = !git_head.is_empty() && remaining >= MARGIN_TOKENS;
    if git_included {
        tiers.push(TierDoc {
            name: "git",
            tokens: git_tok,
        });
        remaining = remaining.saturating_sub(git_tok);
    }

    // ── Tier 6: retrieved files ──────────────────────────────────────────
    let retrieved_total = retrieved.len();
    let mut retrieved_included = 0usize;
    let mut retrieved_head: Vec<RetrievedBlock> = Vec::new();
    for block in retrieved {
        let block_tok = est_tokens(block.body.len(), true);
        if remaining < MARGIN_TOKENS + block_tok {
            truncated = true;
            break;
        }
        remaining = remaining.saturating_sub(block_tok);
        retrieved_included += 1;
        tiers.push(TierDoc {
            name: "retrieved",
            tokens: block_tok,
        });
        retrieved_head.push(block);
    }

    // ── Tier 7: recent messages (most recent wins) ───────────────────────
    let (recent_head, recent_tok) = if remaining >= RECENT_MIN_TOKENS {
        truncate_tail_to_tokens(recent_text, remaining, false)
    } else {
        truncated = true;
        (String::new(), 0)
    };
    if !recent_head.is_empty() {
        tiers.push(TierDoc {
            name: "recent",
            tokens: recent_tok,
        });
    }

    // ── Assemble ─────────────────────────────────────────────────────────
    let mut text = stable_prefix.clone();
    if state_included {
        text.push_str("\n\n## Current Task State\n\n");
        text.push_str(&state_head);
    }
    if git_included {
        text.push_str("\n\n## Git\n\n");
        text.push_str(&git_head);
    }
    if !retrieved_head.is_empty() {
        text.push_str("\n\n## Retrieval\n\n");
        let blocks: Vec<String> = retrieved_head.iter().map(|b| b.body.clone()).collect();
        text.push_str(&blocks.join("\n\n"));
    }
    if !recent_head.is_empty() {
        text.push_str("\n\n## Conversation\n\n");
        text.push_str(&recent_head);
    }

    let total_tokens = target.saturating_sub(remaining);
    let soft_compaction_needed = recent_tok < RECENT_MIN_TOKENS;

    ContextDoc {
        text,
        stable_prefix,
        target_tokens: target,
        total_tokens,
        tiers,
        truncated: truncated || soft_compaction_needed,
        soft_compaction_needed,
        retrieved_included,
        retrieved_total,
    }
}

/// One chat turn produced by [`assemble_chat`]. Roles are the provider-native
/// `"system"` / `"user"` / `"assistant"` strings, so turns map 1:1 onto the
/// provider `ChatMessage` without a translation layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// Framing overhead counted per history turn (role tags the chat template
/// adds around each message). Small, deterministic, and deliberately
/// pessimistic so history can never silently overflow the budget.
pub const HISTORY_TURN_OVERHEAD_TOKENS: u64 = 4;

/// All inputs for [`assemble_chat`] in one struct — the chat counterpart of
/// [`assemble_prompt`]'s eight positional arguments.
#[derive(Debug, Clone)]
pub struct ChatInput<'a> {
    pub profile: HardwareProfile,
    /// Real model context window in tokens (from `ModelCapabilities`).
    /// `None` = unknown → the hardware profile default governs. Utilization
    /// and all other policy knobs always come from `profile`.
    pub context_window: Option<u32>,
    pub system: &'a str,
    pub agents_md: Option<&'a str>,
    pub anchor_md: Option<&'a str>,
    pub state_md: Option<&'a str>,
    pub git_summary: &'a str,
    /// Best-first retrieved blocks; the budgeter trims from the bottom.
    pub retrieved: Vec<RetrievedBlock>,
    /// Pre-rendered explicitly-attached files (e.g. `<file path=…>` blocks).
    /// Sacred like the prompt: always included whole, never trimmed.
    pub attached_block: &'a str,
    /// Prior conversation, oldest-first `(role, content)` pairs. The current
    /// user prompt is NOT part of history — it arrives as `prompt` so the
    /// budgeter can never squeeze the actual question out.
    pub history: &'a [(String, String)],
    pub prompt: &'a str,
}

/// Result of [`assemble_chat`]: ready-to-send turns plus the same budget
/// telemetry [`ContextDoc`] carries, so `/ctx` previews and real generations
/// report comparable numbers.
#[derive(Debug, Clone)]
pub struct ChatAssembly {
    pub turns: Vec<ChatTurn>,
    pub target_tokens: u64,
    pub total_tokens: u64,
    pub tiers: Vec<TierDoc>,
    pub truncated: bool,
    pub soft_compaction_needed: bool,
    pub retrieved_included: usize,
    pub retrieved_total: usize,
    pub history_kept: usize,
    pub history_total: usize,
}

/// Build the per-turn chat messages the model actually receives.
///
/// Same §10 tier discipline as [`assemble_prompt`], but conversation history
/// stays structured (real `user`/`assistant` turns, newest wins) instead of
/// being flattened into text, and the retrieved/state/git context rides along
/// inside the final user turn — the layout hosted chat APIs and llama.cpp's
/// `/v1/chat/completions` both consume natively. Turn 0 is always the
/// byte-stable system head, so KV-prefix reuse (§13) applies to every
/// generation, not just `/ctx` previews.
pub fn assemble_chat(input: ChatInput) -> ChatAssembly {
    // The model's real window when known (Step 3 capabilities); the profile
    // only supplies the default window plus the fill/utilization policy.
    let window = input
        .context_window
        .unwrap_or(input.profile.ctx_tokens() as u32);
    let target = (window as f64 * input.profile.utilization()).floor() as u64;
    let mut tiers: Vec<TierDoc> = Vec::new();
    let mut truncated = false;

    // ── Tiers 1–3: stable system head ────────────────────────────────────
    let stable = stable_head(input.system, input.agents_md, input.anchor_md);
    tiers.push(TierDoc {
        name: "system",
        tokens: stable.system_tokens,
    });
    tiers.push(TierDoc {
        name: "agents.md",
        tokens: stable.agents_tokens,
    });
    tiers.push(TierDoc {
        name: "anchor.md",
        tokens: stable.anchor_tokens,
    });
    let mut remaining = target.saturating_sub(stable.tokens);
    truncated |= stable.truncated || stable.tokens > target;

    // ── Sacred content: current prompt + explicitly attached files ───────
    // Reserved up front and always included whole; if they alone overflow
    // the budget everything else is dropped and the overflow is flagged.
    let sacred_tok =
        est_tokens(input.prompt.len(), false) + est_tokens(input.attached_block.len(), false);
    if sacred_tok >= remaining {
        truncated = true;
    }
    remaining = remaining.saturating_sub(sacred_tok);

    // ── Tier 4: state.md ─────────────────────────────────────────────────
    let (state_head, state_tok) =
        truncate_to_tokens(input.state_md.unwrap_or(""), STATE_CAP_TOKENS, false);
    let state_included = !state_head.is_empty() && remaining >= MARGIN_TOKENS;
    if state_included {
        tiers.push(TierDoc {
            name: "state.md",
            tokens: state_tok,
        });
        remaining = remaining.saturating_sub(state_tok);
    }

    // ── Tier 5: git summary ──────────────────────────────────────────────
    let (git_head, git_tok) = truncate_to_tokens(input.git_summary, GIT_CAP_TOKENS, false);
    let git_included = !git_head.is_empty() && remaining >= MARGIN_TOKENS;
    if git_included {
        tiers.push(TierDoc {
            name: "git",
            tokens: git_tok,
        });
        remaining = remaining.saturating_sub(git_tok);
    }

    // ── Tier 6: retrieved files ──────────────────────────────────────────
    let retrieved_total = input.retrieved.len();
    let mut retrieved_included = 0usize;
    let mut retrieved_head: Vec<RetrievedBlock> = Vec::new();
    for block in input.retrieved {
        let block_tok = est_tokens(block.body.len(), true);
        if remaining < MARGIN_TOKENS + block_tok {
            truncated = true;
            break;
        }
        remaining = remaining.saturating_sub(block_tok);
        retrieved_included += 1;
        tiers.push(TierDoc {
            name: "retrieved",
            tokens: block_tok,
        });
        retrieved_head.push(block);
    }

    // ── Tier 7: structured history (most recent wins) ────────────────────
    let history_total = input.history.len();
    let mut kept: Vec<(String, String)> = Vec::new();
    let mut history_tok = 0u64;
    if remaining >= RECENT_MIN_TOKENS {
        for (role, content) in input.history.iter().rev() {
            let turn_tok =
                est_tokens(role.len() + content.len(), false) + HISTORY_TURN_OVERHEAD_TOKENS;
            if remaining < turn_tok {
                truncated = true;
                break;
            }
            remaining = remaining.saturating_sub(turn_tok);
            history_tok += turn_tok;
            kept.push((role.clone(), content.clone()));
        }
        kept.reverse();
    } else if history_total > 0 {
        truncated = true;
    }
    if !kept.is_empty() {
        tiers.push(TierDoc {
            name: "history",
            tokens: history_tok,
        });
    }

    // ── Assemble turns ───────────────────────────────────────────────────
    let history_kept = kept.len();
    let mut turns = Vec::with_capacity(history_kept + 2);
    turns.push(ChatTurn {
        role: "system".to_string(),
        content: stable.prefix,
    });
    for (role, content) in kept {
        turns.push(ChatTurn { role, content });
    }
    let mut user_turn = String::new();
    if state_included {
        user_turn.push_str("## Current Task State\n\n");
        user_turn.push_str(&state_head);
        user_turn.push_str("\n\n");
    }
    if git_included {
        user_turn.push_str("## Git\n\n");
        user_turn.push_str(&git_head);
        user_turn.push_str("\n\n");
    }
    if !retrieved_head.is_empty() {
        user_turn.push_str("## Retrieval\n\n");
        let blocks: Vec<String> = retrieved_head.iter().map(|b| b.body.clone()).collect();
        user_turn.push_str(&blocks.join("\n\n"));
        user_turn.push_str("\n\n");
    }
    if !input.attached_block.trim().is_empty() {
        user_turn.push_str("## Attached Files\n\n");
        user_turn.push_str(input.attached_block.trim());
        user_turn.push_str("\n\n");
    }
    user_turn.push_str(input.prompt);
    turns.push(ChatTurn {
        role: "user".to_string(),
        content: user_turn,
    });

    let total_tokens = target.saturating_sub(remaining);
    // Compaction is actionable only when real history was lost: unlike the
    // text preview (where a short `recent_text` trips the 40-token floor on
    // every fresh conversation), dropping zero turns must never suggest it.
    let soft_compaction_needed = history_total > history_kept;

    ChatAssembly {
        turns,
        target_tokens: target,
        total_tokens,
        tiers,
        truncated: truncated || soft_compaction_needed,
        soft_compaction_needed,
        retrieved_included,
        retrieved_total,
        history_kept,
        history_total,
    }
}

/// Everything the live chat path needs from disk for one user turn.
#[derive(Debug, Clone)]
pub struct LiveContext {
    pub agents_md: Option<String>,
    pub anchor_md: Option<String>,
    pub state_md: Option<String>,
    pub git_summary: String,
    pub blocks: Vec<RetrievedBlock>,
    /// Whether `.xencode/` holds a usable retrieval index. `false` means the
    /// model still gets identity + guidelines + history, just no file bodies —
    /// callers should nudge toward `/init` once per session.
    pub index_present: bool,
    /// Retrieval candidates before unreadable-file filtering.
    pub retrieved_total: usize,
}

/// Gather project context for one user query: stable-layer files, git summary,
/// and deterministic retrieval against the `/init` index when present.
///
/// Never fails — every source degrades to empty independently, so a missing
/// index or unreadable file can never break a generation.
pub fn collect_live_context(root: &Path, query: &str, profile: HardwareProfile) -> LiveContext {
    let xencode = root.join(crate::XENCODE_DIR);
    let agents_md = std::fs::read_to_string(root.join("AGENTS.md")).ok();
    let anchor_md = std::fs::read_to_string(xencode.join("anchor.md")).ok();
    let state_md = std::fs::read_to_string(xencode.join("state.md")).ok();
    let git_summary = git_summary_text(root).unwrap_or_default();
    let mut blocks = Vec::new();
    let mut index_present = false;
    let mut retrieved_total = 0;
    if let Some(index) = RetrievalIndex::load(&xencode) {
        index_present = true;
        let changed: HashSet<String> = dirty_paths(root).into_iter().collect();
        let opts = RetrieveOptions {
            top_k: profile.top_k(),
            ..Default::default()
        };
        let results = retrieve(query, &index, &changed, &opts);
        retrieved_total = results.len();
        blocks = read_retrieved_bodies(root, &index.files, &results, profile.content_cap_chars());
    }
    LiveContext {
        agents_md,
        anchor_md,
        state_md,
        git_summary,
        blocks,
        index_present,
        retrieved_total,
    }
}

/// Build the tier-5 git summary text: `🎋 branch @ head — N dirty file(s)`
/// plus the changed-file list (capped to `GIT_CAP_TOKENS` downstream).
pub fn git_summary_text(root: &Path) -> Option<String> {
    let info = current_git_info(root)?;
    let changed = dirty_paths(root);
    let head = if info.head.len() > 8 {
        &info.head[..8]
    } else {
        &info.head
    };
    let mut s = format!("🎋 {} @ {head} — {} dirty file(s)", info.branch, info.dirty);
    if !changed.is_empty() {
        s.push_str("\nChanged files:\n");
        for p in changed.iter().take(12) {
            s.push_str(&format!("  • {p}\n"));
        }
        if changed.len() > 12 {
            s.push_str(&format!("  … +{} more\n", changed.len() - 12));
        }
    }
    Some(s)
}

/// Read and fence the bodies of the retrieved files, capped per profile.
/// Secret/binary entries from the stale index are skipped; unreadable files
/// are skipped silently.
pub fn read_retrieved_bodies(
    root: &Path,
    files: &[FileEntry],
    retrieved: &[RetrievedFile],
    cap_chars: usize,
) -> Vec<RetrievedBlock> {
    let mut blocks = Vec::new();
    for r in retrieved {
        let Some(entry) = files.iter().find(|f| f.path == r.path) else {
            continue;
        };
        if entry.secret || entry.binary {
            continue;
        }
        let full = root.join(&r.path);
        let Ok(text) = std::fs::read_to_string(&full) else {
            continue;
        };
        let mut content = text;
        if content.len() > cap_chars {
            let mut end = cap_chars;
            while end > 0 && !content.is_char_boundary(end) {
                end -= 1;
            }
            content = content[..end].to_string();
        }
        blocks.push(RetrievedBlock {
            path: r.path.clone(),
            score: r.score,
            body: format!("File: {}\n```{}\n{}\n```", r.path, entry.language, content),
        });
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::init_project;

    const SYSTEM: &str = "You are a coding agent. Be concise.";
    const AGENTS: &str = "# Rules\n- Rust first\n- commit each change\n";
    const ANCHOR: &str = "# Project Anchor\n\nArchitecture: CLI → core → providers.\n";

    fn sample_retrieved() -> Vec<RetrievedBlock> {
        vec![
            RetrievedBlock {
                path: "src/auth.rs".to_string(),
                score: 18,
                body: "File: src/auth.rs\n```rust\npub fn authenticate() {}\n```".to_string(),
            },
            RetrievedBlock {
                path: "src/database.rs".to_string(),
                score: 7,
                body: "File: src/database.rs\n```rust\npub fn query() {}\n```".to_string(),
            },
        ]
    }

    #[test]
    fn margin_gate_excludes_state_from_text_and_budget() {
        // Stable prefix overflows the whole Low budget, so remaining is 0 at
        // tier 4: the margin gate fails and the state tier must be absent
        // from BOTH the text and the tier list (never emitted-but-uncounted).
        let sys = "s\n".repeat(9000);
        let state = "st\n".repeat(2000);
        let doc = assemble_prompt(
            HardwareProfile::Low,
            &sys,
            None,
            None,
            Some(&state),
            "",
            vec![],
            "",
        );
        assert!(!doc.text.contains("## Current Task State"));
        assert!(!doc.tiers.iter().any(|t| t.name == "state.md"));
    }

    #[test]
    fn stable_prefix_honors_fixed_order_and_marker() {        let doc = assemble_prompt(
            HardwareProfile::Balanced,
            SYSTEM,
            Some(AGENTS),
            Some(ANCHOR),
            None,
            "",
            Vec::new(),
            "",
        );
        let sp = &doc.stable_prefix;
        assert!(sp.starts_with(SYSTEM));
        assert!(sp.contains(AGENTS));
        assert!(sp.contains(ANCHOR));
        assert!(sp.contains(STABLE_END_MARKER));
        // Marker is the last line — nothing dynamic precedes it.
        assert!(sp.ends_with(STABLE_END_MARKER));
        assert!(doc.text.starts_with(&doc.stable_prefix));
    }

    #[test]
    fn budget_fills_all_tiers_within_target() {
        // A long enough recent window so tier 7 clears the soft-compaction
        // floor (~40 tokens) on the Balanced profile.
        let recent: String = (0..40)
            .map(|i| format!("[m{i}] user: fix the auth flow now please\n"))
            .collect();
        let doc = assemble_prompt(
            HardwareProfile::Balanced,
            SYSTEM,
            Some(AGENTS),
            Some(ANCHOR),
            Some("# state: fixing auth"),
            "🎋 main @ abc1234 — 1 dirty file(s)",
            sample_retrieved(),
            &recent,
        );
        assert!(doc.total_tokens <= doc.target_tokens);
        assert_eq!(doc.retrieved_included, 2);
        assert_eq!(doc.retrieved_total, 2);
        assert!(!doc.soft_compaction_needed);
        assert!(doc.text.contains("## Retrieval"));
        assert!(doc.text.contains("## Conversation"));
        assert!(doc.tiers.iter().any(|t| t.name == "git"));
    }

    #[test]
    fn tight_low_profile_trims_retrieved_and_flags_compaction() {
        let long_retrieved: Vec<RetrievedBlock> = (0..20)
            .map(|i| RetrievedBlock {
                path: format!("src/f{i}.rs"),
                score: 10 + i as u64,
                body: format!(
                    "File: src/f{i}.rs\n```rust\n{}\n```",
                    "pub fn x(i: u64) -> u64 { i * 2 }\n".repeat(8)
                ),
            })
            .collect();
        let doc = assemble_prompt(
            HardwareProfile::Low,
            SYSTEM,
            Some(&"# Rules\n".repeat(200)),
            Some(&"# Anchor\n\n".repeat(200)),
            None,
            "",
            long_retrieved,
            "",
        );
        assert!(doc.total_tokens <= doc.target_tokens);
        // Not all blocks fit the LOW budget, but the best ones do.
        assert!(doc.retrieved_included > 0);
        assert!(doc.retrieved_included < doc.retrieved_total);
        assert!(doc.truncated);
    }

    #[test]
    fn empty_retrieval_still_produces_a_prompt() {
        let doc = assemble_prompt(
            HardwareProfile::High,
            SYSTEM,
            None,
            None,
            None,
            "",
            Vec::new(),
            "",
        );
        assert!(!doc.text.is_empty());
        assert!(doc.text.contains(STABLE_END_MARKER));
        assert_eq!(doc.retrieved_total, 0);
    }

    fn sample_history() -> Vec<(String, String)> {
        vec![
            ("user".to_string(), "how does auth work?".to_string()),
            (
                "assistant".to_string(),
                "it uses the auth module".to_string(),
            ),
        ]
    }

    fn sample_chat_input<'a>(
        retrieved: Vec<RetrievedBlock>,
        history: &'a [(String, String)],
    ) -> ChatInput<'a> {
        ChatInput {
            profile: HardwareProfile::Balanced,
            context_window: None,
            system: SYSTEM,
            agents_md: Some(AGENTS),
            anchor_md: Some(ANCHOR),
            state_md: Some("# state: fixing auth"),
            git_summary: "main @ abc1234",
            retrieved,
            attached_block: "",
            history,
            prompt: "where is the login handler?",
        }
    }

    #[test]
    fn chat_system_turn_matches_preview_stable_prefix() {
        let history = sample_history();
        let chat = assemble_chat(sample_chat_input(sample_retrieved(), &history));
        let preview = assemble_prompt(
            HardwareProfile::Balanced,
            SYSTEM,
            Some(AGENTS),
            Some(ANCHOR),
            Some("# state: fixing auth"),
            "main @ abc1234",
            sample_retrieved(),
            "",
        );
        assert_eq!(chat.turns[0].role, "system");
        // Byte-identical to the /ctx preview head: same bytes hit the model,
        // so llama.cpp KV-prefix reuse applies to real generations too.
        assert_eq!(chat.turns[0].content, preview.stable_prefix);
        assert_eq!(
            chat.turns[0].content,
            stable_system_text(SYSTEM, Some(AGENTS), Some(ANCHOR))
        );
    }

    #[test]
    fn chat_final_turn_carries_retrieval_and_prompt() {
        let history = sample_history();
        let chat = assemble_chat(sample_chat_input(sample_retrieved(), &history));
        // system + 2 history turns + 1 final user turn.
        assert_eq!(chat.turns.len(), 4);
        let last = chat.turns.last().unwrap();
        assert_eq!(last.role, "user");
        assert!(last.content.contains("pub fn authenticate() {}"));
        assert!(last.content.contains("where is the login handler?"));
        // History keeps its roles (not flattened into one blob).
        assert_eq!(chat.turns[1].role, "user");
        assert_eq!(chat.turns[2].role, "assistant");
        assert_eq!(chat.retrieved_included, 2);
        assert_eq!(chat.history_kept, 2);
        assert_eq!(chat.history_total, 2);
        assert!(chat.total_tokens <= chat.target_tokens);
        assert!(!chat.truncated);
    }

    #[test]
    fn chat_history_prefers_newest_under_tight_budget() {
        // ~70 tokens/turn × 40 ≈ 2800 > LOW target (2457), so the oldest
        // turns must give way while the newest survive.
        let history: Vec<(String, String)> = (0..40)
            .map(|i| {
                (
                    "user".to_string(),
                    format!(
                        "question number {i} about the codebase routines and helpers. {}",
                        "please explain the surrounding module in detail. ".repeat(4)
                    ),
                )
            })
            .collect();
        let input = ChatInput {
            profile: HardwareProfile::Low,
            ..sample_chat_input(Vec::new(), &history)
        };
        let chat = assemble_chat(input);
        assert_eq!(chat.history_total, 40);
        assert!(chat.history_kept > 0);
        assert!(chat.history_kept < chat.history_total);
        assert!(chat.truncated);
        // Newest-first: the last history turn survives, the oldest is dropped.
        let kept: Vec<&str> = chat.turns[1..chat.turns.len() - 1]
            .iter()
            .map(|t| t.content.as_str())
            .collect();
        assert!(kept.iter().any(|c| c.contains("question number 39")));
        assert!(!kept.iter().any(|c| c.contains("question number 0")));
    }

    #[test]
    fn chat_builds_without_any_optional_inputs() {
        let chat = assemble_chat(sample_chat_input(Vec::new(), &[]));
        assert_eq!(chat.turns.len(), 2);
        assert_eq!(chat.turns[0].role, "system");
        assert_eq!(chat.turns[1].role, "user");
        assert!(chat.turns[1]
            .content
            .contains("where is the login handler?"));
        assert_eq!(chat.retrieved_included, 0);
        assert_eq!(chat.history_kept, 0);
    }

    fn fixture_project(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "xencode-chat-{}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::SeqCst),
            tag
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("src").join("authenticate.rs"),
            "pub fn authenticate_user(name: &str) -> bool {\n    !name.is_empty()\n}\n",
        )
        .unwrap();
        std::fs::write(dir.join("AGENTS.md"), "# Rules\n- Rust first\n").unwrap();
        dir
    }

    #[test]
    fn live_context_collects_index_and_bodies() {
        let dir = fixture_project("indexed");
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        init_project(&dir, cancel, |_| {}).expect("fixture /init must succeed");
        let live = collect_live_context(&dir, "authenticate", HardwareProfile::Balanced);
        assert!(live.index_present);
        assert!(live.retrieved_total > 0);
        assert!(!live.blocks.is_empty());
        assert!(live.blocks[0].body.contains("authenticate_user"));
        assert!(live.agents_md.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn chat_uses_real_model_window_when_known() {
        let history = sample_history();
        let base = sample_chat_input(sample_retrieved(), &history);
        let profiled = assemble_chat(base.clone());
        assert_eq!(
            profiled.target_tokens,
            (HardwareProfile::Balanced.ctx_tokens() as f64
                * HardwareProfile::Balanced.utilization()) as u64
        );
        // A 128k model on the same profile: same utilization policy, but the
        // budgeter now knows the real room — nothing gets trimmed.
        let wide = assemble_chat(ChatInput {
            context_window: Some(128_000),
            ..base
        });
        assert_eq!(
            wide.target_tokens,
            (128_000f64 * HardwareProfile::Balanced.utilization()) as u64
        );
        assert!(wide.target_tokens > profiled.target_tokens);
        assert_eq!(wide.history_kept, wide.history_total);
        assert!(!wide.truncated);
    }

    #[test]
    fn live_context_without_index_reports_missing() {
        let dir = fixture_project("plain");
        let live = collect_live_context(&dir, "authenticate", HardwareProfile::Balanced);
        assert!(!live.index_present);
        assert!(live.blocks.is_empty());
        // The model still gets identity + guidelines + the question.
        let chat = assemble_chat(ChatInput {
            profile: HardwareProfile::Balanced,
            context_window: None,
            system: SYSTEM,
            agents_md: live.agents_md.as_deref(),
            anchor_md: live.anchor_md.as_deref(),
            state_md: live.state_md.as_deref(),
            git_summary: &live.git_summary,
            retrieved: live.blocks,
            attached_block: "",
            history: &[],
            prompt: "authenticate?",
        });
        assert_eq!(chat.turns.len(), 2);
        assert!(chat.turns[1].content.contains("authenticate?"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_bodies_caps_and_skips_unreadable() {
        let root = std::path::PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let files = vec![FileEntry {
            path: "Cargo.toml".to_string(),
            language: "toml".to_string(),
            size: 0,
            loc: 0,
            ext: "toml".to_string(),
            important: true,
            secret: false,
            binary: false,
        }];
        let retrieved = vec![RetrievedFile {
            path: "Cargo.toml".to_string(),
            score: 10,
            reasons: vec!["filename exact".to_string()],
        }];
        let blocks = read_retrieved_bodies(&root, &files, &retrieved, 40);
        assert_eq!(blocks.len(), 1);
        let body = &blocks[0].body;
        assert!(body.starts_with("File: Cargo.toml\n```toml\n"));
        assert!(body.len() <= 40 + 64, "body must be capped");
        assert!(!blocks[0].body.contains("\n```\n```"));
    }
}
