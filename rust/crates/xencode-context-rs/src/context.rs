//! Context assembly (M2) — §10 of the architecture spec.
//!
//! Turns profile + stable layer files + retrieval candidates + recent
//! conversation into the actual prompt for the model, applying the token
//! budget algorithm. Tiers 1–3 form the byte-stable prefix below which the
//! KV cache can be reused across requests; nothing dynamic may precede it.
//!
//! Pure/disk-free (`assemble_prompt` takes strings and pre-built fenced
//! blocks), which keeps the budget logic unit-testable without a repository.

use crate::budget::{
    est_tokens, fill_target, truncate_tail_to_tokens, truncate_to_tokens, ContextCaps,
    HardwareProfile,
};
use crate::gitinfo::{current_git_info, dirty_paths};
use crate::index::FileEntry;
use crate::redact::{Redactor, Vault};
use crate::retrieve::{retrieve, RetrievalIndex, RetrieveOptions, RetrievedFile};
use crate::source::SourceClass;
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

/// A prompt budgeted at or below what the `Low` profile fills gets a symbol-only
/// repo map instead of more file bodies: at that size the retrieval tier holds
/// one to three files and the model has no view of the rest of the repository.
/// The rule is about the budget, not the profile name, because a `Balanced`
/// machine behind a small server is in the same position as a `Low` one — the
/// lesson AC-4 learned about caps.
pub fn budget_wants_repo_map(target_tokens: u64) -> bool {
    let low = (HardwareProfile::Low.ctx_tokens() as f64 * HardwareProfile::Low.utilization())
        .floor() as u64;
    target_tokens <= low
}

/// The marker that closes the stable prefix (byte-identical every request).
pub const STABLE_END_MARKER: &str = "<!-- xencode:stable-prefix-end -->";

/// A retrieved file's fenced body, ready to inject.
#[derive(Debug, Clone)]
pub struct RetrievedBlock {
    pub path: String,
    pub score: u64,
    /// Already rendered: `File: <path>\n```<lang>\n<content>\n````.
    pub body: String,
}

/// One consumed budget tier (for metrics/meter display).
///
/// `class` is QK-3's contribution to the ledger: a tier cannot be added without
/// saying whose bytes it holds, so a report can tell the human's own words from
/// fetched content instead of only counting them together.
#[derive(Debug, Clone)]
pub struct TierDoc {
    pub name: &'static str,
    pub tokens: u64,
    pub class: crate::SourceClass,
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
    /// What the turn is made of, by whose bytes it is: tokens summed per
    /// [`crate::SourceClass`], biggest first. QK-3's second consumer — a tier
    /// that names no class cannot be added, so this cannot silently leave a
    /// fetched body out of the count. `/egress` shows it; the stable head is
    /// included because the human is entitled to know what a turn carries.
    pub fn source_totals(&self) -> Vec<(crate::SourceClass, u64)> {
        crate::source::totals_by_class(&self.tiers)
    }

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

/// The repo map tier, admitted only when the prompt is budgeted like `Low` and
/// the map fits in the room still left after the tiers above it. Returns the
/// text to emit (`None` = no tier) and what it cost.
fn repo_map_tier(repo_map: &str, target: u64, remaining: u64) -> (Option<String>, u64) {
    if repo_map.trim().is_empty() || !budget_wants_repo_map(target) {
        return (None, 0);
    }
    let tokens = crate::budget::est_tokens(repo_map.len(), false);
    if remaining < tokens + MARGIN_TOKENS {
        return (None, 0);
    }
    (Some(repo_map.to_string()), tokens)
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
    repo_map: &str,
    retrieved: Vec<RetrievedBlock>,
    recent_text: &str,
) -> ContextDoc {
    let target = (profile.ctx_tokens() as f64 * profile.utilization()).floor() as u64;
    let mut tiers: Vec<TierDoc> = Vec::new();
    let mut truncated = false;

    // ── Tiers 1–3: stable prefix ─────────────────────────────────────────
    let stable = stable_head(system, agents_md, anchor_md);
    let agents_class = agents_md
        .map(SourceClass::of_agents_md)
        .unwrap_or(SourceClass::AgentFile { trusted: true });
    tiers.push(TierDoc {
        name: "system",
        tokens: stable.system_tokens,
        class: SourceClass::Instructions,
    });
    tiers.push(TierDoc {
        name: "agents.md",
        tokens: stable.agents_tokens,
        class: agents_class,
    });
    tiers.push(TierDoc {
        name: "anchor.md",
        tokens: stable.anchor_tokens,
        class: SourceClass::AnchorFile,
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
            class: SourceClass::ProjectState,
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
            class: SourceClass::Repository,
        });
        remaining = remaining.saturating_sub(git_tok);
    }

    // ── Tier 5b: repo map, on a small-budget prompt only ─────────────────
    let (map_head, map_tok) = repo_map_tier(repo_map, target, remaining);
    if map_head.is_some() {
        tiers.push(TierDoc {
            name: "repo map",
            tokens: map_tok,
            class: SourceClass::Repository,
        });
        remaining = remaining.saturating_sub(map_tok);
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
            class: SourceClass::Repository,
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
            class: SourceClass::History,
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
        text.push_str(REPO_DATA_NOTE);
        text.push_str(&git_head);
    }
    if let Some(map) = &map_head {
        text.push_str("\n\n## Repo Map\n\n");
        text.push_str(map);
    }
    if !retrieved_head.is_empty() {
        text.push_str("\n\n## Retrieval\n\n");
        text.push_str(REPO_DATA_NOTE);
        let blocks: Vec<String> = retrieved_head.iter().map(|b| b.body.clone()).collect();
        text.push_str(&blocks.join("\n\n"));
    }
    if !recent_head.is_empty() {
        text.push_str("\n\n## Conversation\n\n");
        text.push_str(&recent_head);
    }

    let total_tokens = target.saturating_sub(remaining);
    // Actionable only when recent content was actually dropped: like the
    // chat assembler (`history_total > history_kept`), a short-or-empty
    // history on a fresh conversation must never suggest compaction.
    let soft_compaction_needed = !recent_text.is_empty() && recent_head.len() < recent_text.len();

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
    /// The symbol-only repo map ([`crate::repo_map_text`]); admitted only on a
    /// prompt budgeted like `Low`, and empty when there is no index.
    pub repo_map: &'a str,
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
    /// Characters across the [`retrieved_included`] bodies, with their `File:`
    /// header and code fence. Kept so a server's token count for a prompt can be
    /// split between retrieval and the fixed cost of a turn.
    pub retrieved_chars_included: usize,
    /// Which files those [`retrieved_included`] bodies came from, best-match
    /// first — the ones the budget kept, not the candidates it trimmed. The
    /// turn trace records this list so a turn can be asked what it was looking
    /// at; the bodies themselves are not kept anywhere.
    pub retrieved_files: Vec<String>,
    pub history_kept: usize,
    pub history_total: usize,
    /// The secret values this assembly took out of the dynamic tiers, keyed by
    /// the placeholder the model sees. Empty when nothing credential-shaped was
    /// found. The stable head is never in here because it is never redacted. The
    /// chat path hands this to the executor so a placeholder in a tool call's
    /// arguments becomes the real value again at the point of running.
    pub vault: Vault,
}

impl ChatAssembly {
    /// Everything the model is about to be shown, as one plain string, for a
    /// server that counts tokens for text rather than for a message list
    /// (llama.cpp's `/tokenize`).
    ///
    /// Roles are left out and the turns are joined with a blank line, so the
    /// count that comes back is what the pieces contain — not the rendered
    /// request, which additionally carries the chat template's per-message
    /// markers. Those are the framing `HISTORY_TURN_OVERHEAD_TOKENS` stands for
    /// in the estimate, and they are missing here: the counted number is a floor.
    pub fn prompt_text(&self) -> String {
        self.turns
            .iter()
            .map(|turn| turn.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Characters of prompt text, which is what a server's token count for it is
    /// a proportion of — see [`crate::PromptOverhead::observe`].
    pub fn prompt_chars(&self) -> usize {
        self.turns.iter().map(|turn| turn.content.len()).sum()
    }

    /// Characters of the retrieved file bodies that made it into this prompt,
    /// which is the part of it that is project code rather than the fixed cost of
    /// a turn. Zero when the budget kept no files.
    pub fn retrieved_chars(&self) -> usize {
        self.retrieved_chars_included
    }

    /// Tokens per [`crate::SourceClass`] in this turn — see
    /// [`ContextDoc::source_totals`], the same accounting for the same tiers.
    pub fn source_totals(&self) -> Vec<(crate::SourceClass, u64)> {
        crate::source::totals_by_class(&self.tiers)
    }
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
    let target = fill_target(input.profile, input.context_window);
    let mut tiers: Vec<TierDoc> = Vec::new();
    let mut truncated = false;

    // ── Tiers 1–3: stable system head ────────────────────────────────────
    let stable = stable_head(input.system, input.agents_md, input.anchor_md);
    tiers.push(TierDoc {
        name: "system",
        tokens: stable.system_tokens,
        class: SourceClass::Instructions,
    });
    tiers.push(TierDoc {
        name: "agents.md",
        tokens: stable.agents_tokens,
        class: input
            .agents_md
            .map(SourceClass::of_agents_md)
            .unwrap_or(SourceClass::AgentFile { trusted: true }),
    });
    tiers.push(TierDoc {
        name: "anchor.md",
        tokens: stable.anchor_tokens,
        class: SourceClass::AnchorFile,
    });
    let mut remaining = target.saturating_sub(stable.tokens);
    truncated |= stable.truncated || stable.tokens > target;

    // ── Sacred content: current prompt + explicitly attached files ───────
    // Reserved up front and always included whole; if they alone overflow
    // the budget everything else is dropped and the overflow is flagged.
    let prompt_tok = est_tokens(input.prompt.len(), false);
    let attached_tok = est_tokens(input.attached_block.len(), false);
    let sacred_tok = prompt_tok + attached_tok;
    if sacred_tok >= remaining {
        truncated = true;
    }
    remaining = remaining.saturating_sub(sacred_tok);
    tiers.push(TierDoc {
        name: "prompt",
        tokens: prompt_tok,
        class: SourceClass::UserTurn,
    });
    if !input.attached_block.is_empty() {
        tiers.push(TierDoc {
            name: "attached",
            tokens: attached_tok,
            class: SourceClass::AttachedFile,
        });
    }

    // ── Tier 4: state.md ─────────────────────────────────────────────────
    let (state_head, state_tok) =
        truncate_to_tokens(input.state_md.unwrap_or(""), STATE_CAP_TOKENS, false);
    let state_included = !state_head.is_empty() && remaining >= MARGIN_TOKENS;
    if state_included {
        tiers.push(TierDoc {
            name: "state.md",
            tokens: state_tok,
            class: SourceClass::ProjectState,
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
            class: SourceClass::Repository,
        });
        remaining = remaining.saturating_sub(git_tok);
    }

    // ── Tier 5b: repo map, on a small-budget prompt only ─────────────────
    let (map_head, map_tok) = repo_map_tier(input.repo_map, target, remaining);
    if map_head.is_some() {
        tiers.push(TierDoc {
            name: "repo map",
            tokens: map_tok,
            class: SourceClass::Repository,
        });
        remaining = remaining.saturating_sub(map_tok);
    }

    // ── Tier 6: retrieved files ──────────────────────────────────────────
    let retrieved_total = input.retrieved.len();
    let mut retrieved_included = 0usize;
    let mut retrieved_chars_included = 0usize;
    let mut retrieved_head: Vec<RetrievedBlock> = Vec::new();
    for block in input.retrieved {
        let block_tok = est_tokens(block.body.len(), true);
        if remaining < MARGIN_TOKENS + block_tok {
            truncated = true;
            break;
        }
        remaining = remaining.saturating_sub(block_tok);
        retrieved_included += 1;
        retrieved_chars_included += block.body.len();
        tiers.push(TierDoc {
            name: "retrieved",
            tokens: block_tok,
            class: SourceClass::Repository,
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
            class: SourceClass::History,
        });
    }

    // ── Assemble turns ───────────────────────────────────────────────────
    // The stable head (`turns[0]`, `stable.prefix`) is sent byte-for-byte every
    // turn so a local server can reuse its key/value cache — redacting it would
    // break that reuse and trip `/ctx`'s drift check, so it is left untouched.
    // Everything after it is dynamic and has no cache to lose: those tiers are
    // what gets redacted on the way out. One redactor spans the whole turn so a
    // secret repeated across tiers collapses to one placeholder.
    let history_kept = kept.len();
    let mut redactor = Redactor::new();
    let mut turns = Vec::with_capacity(history_kept + 2);
    turns.push(ChatTurn {
        role: "system".to_string(),
        content: stable.prefix,
    });
    for (role, content) in kept {
        turns.push(ChatTurn {
            role,
            content: redactor.redact(&content),
        });
    }
    let mut user_turn = String::new();
    if state_included {
        user_turn.push_str("## Current Task State\n\n");
        user_turn.push_str(&state_head);
        user_turn.push_str("\n\n");
    }
    if git_included {
        user_turn.push_str("## Git\n\n");
        user_turn.push_str(REPO_DATA_NOTE);
        user_turn.push_str(&git_head);
        user_turn.push_str("\n\n");
    }
    if let Some(map) = &map_head {
        user_turn.push_str("## Repo Map\n\n");
        user_turn.push_str(map);
        user_turn.push_str("\n\n");
    }
    if !retrieved_head.is_empty() {
        user_turn.push_str("## Retrieval\n\n");
        user_turn.push_str(REPO_DATA_NOTE);
        let blocks: Vec<String> = retrieved_head.iter().map(|b| b.body.clone()).collect();
        user_turn.push_str(&blocks.join("\n\n"));
        user_turn.push_str("\n\n");
    }
    if !input.attached_block.trim().is_empty() {
        user_turn.push_str("## Attached Files\n\n");
        user_turn.push_str(crate::source::ATTACHED_DATA_NOTE);
        user_turn.push_str(input.attached_block.trim());
        user_turn.push_str("\n\n");
    }
    user_turn.push_str(input.prompt);
    let user_turn = redactor.redact(&user_turn);
    turns.push(ChatTurn {
        role: "user".to_string(),
        content: user_turn,
    });

    let vault = redactor.into_vault();

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
        retrieved_files: retrieved_head.iter().map(|b| b.path.clone()).collect(),
        retrieved_chars_included,
        history_kept,
        history_total,
        vault,
    }
}

/// Everything the live chat path needs from disk for one user turn.
#[derive(Debug, Clone)]
pub struct LiveContext {
    pub agents_md: Option<String>,
    pub anchor_md: Option<String>,
    pub state_md: Option<String>,
    pub git_summary: String,
    /// The symbol-only repo map for this turn, built from the index and seeded
    /// by what retrieval and the working tree already picked. Whether it is
    /// worth tokens at all is the budgeter's decision, not this one's — see
    /// [`budget_wants_repo_map`].
    pub repo_map: String,
    pub blocks: Vec<RetrievedBlock>,
    /// Whether `.xencode/` holds a usable retrieval index. `false` means the
    /// model still gets identity + guidelines + history, just no file bodies —
    /// callers should nudge toward `/init` once per session.
    pub index_present: bool,
    /// Retrieval candidates before unreadable-file filtering.
    pub retrieved_total: usize,
    /// Which kind of work the prompt was read as, and the words that said so.
    /// Retrieval was already weighted by it; this is here so the interface can
    /// show the reading instead of leaving a changed file list unexplained.
    pub shape: crate::ShapeRead,
}

/// SE-2: what a repository-derived section says about itself. The system
/// prompt tells the model that fetched content carries no instructions; this
/// line is where a section that lands inside the *user* turn says so too,
/// instead of letting git output and file bodies ride in unlabelled. The
/// wording is owned by [`crate::SourceClass::Repository`] (QK-3).
const REPO_DATA_NOTE: &str = crate::source::REPO_DATA_NOTE;

/// Gather project context for one user query: stable-layer files, git summary,
/// and deterministic retrieval against the `/init` index when present.
///
/// `caps` is how much retrieval this turn may ask for — see
/// [`ContextCaps::for_turn`] for where the number comes from, since retrieval
/// happens here and therefore cannot see the prompt it is being retrieved for.
///
/// Never fails — every source degrades to empty independently, so a missing
/// index or unreadable file can never break a generation.
pub fn collect_live_context(root: &Path, query: &str, caps: ContextCaps) -> LiveContext {
    let xencode = root.join(crate::XENCODE_DIR);
    // SE-3: through the trust seam, like every other reader — an untrusted
    // `AGENTS.md` reaches the live turn marked as data, never as instructions.
    let agents_md = crate::trust::read_agents_md(root);
    let anchor_md = std::fs::read_to_string(xencode.join("anchor.md")).ok();
    let state_md = std::fs::read_to_string(xencode.join("state.md")).ok();
    let git_summary = git_summary_text(root).unwrap_or_default();
    // Read once, and used for the weights and for what is reported about them,
    // so the two cannot disagree about which shape the turn was retrieved as.
    let shape = crate::shape_of(query);
    let mut blocks = Vec::new();
    let mut repo_map = String::new();
    let mut index_present = false;
    let mut retrieved_total = 0;
    if let Some(index) = RetrievalIndex::load(&xencode) {
        index_present = true;
        let changed: HashSet<String> = dirty_paths(root).into_iter().collect();
        let opts = RetrieveOptions::for_live_chat(caps.top_k, shape.shape);
        let results = retrieve(query, &index, &changed, &opts);
        retrieved_total = results.len();
        blocks = read_retrieved_bodies(root, &index.files, &results, caps.content_cap_chars);
        // Seeded by what this turn is already about, so the map points away from
        // the bodies the model is being handed rather than repeating them.
        let mut seeds: Vec<String> = blocks.iter().map(|b| b.path.clone()).collect();
        seeds.extend(changed.iter().cloned());
        repo_map = crate::repo_map::repo_map_text(&index, &seeds);
    }
    LiveContext {
        agents_md,
        anchor_md,
        state_md,
        git_summary,
        repo_map,
        blocks,
        index_present,
        retrieved_total,
        shape,
    }
}

/// Build the tier-5 git summary text: `🎋 branch @ head — N dirty file(s)`
/// plus the changed-file list (capped to `GIT_CAP_TOKENS` downstream).
pub fn git_summary_text(root: &Path) -> Option<String> {
    let info = current_git_info(root)?;
    let changed = dirty_paths(root);
    let mut s = format!(
        "🎋 {} @ {} — {} dirty file(s)",
        info.branch,
        info.revision_label(),
        info.dirty
    );
    if !changed.is_empty() {
        s.push_str("\nChanged files:\n");
        for p in changed.iter().take(12) {
            s.push_str(&format!("  • {p}\n"));
        }
        if changed.len() > 12 {
            s.push_str(&format!("  … +{} more\n", changed.len() - 12));
        }
    }
    // Extra worktrees only: a lone checkout is the status quo and noise.
    if let Ok(wts) = crate::worktree::worktree_list(root) {
        if wts.len() > 1 {
            s.push_str("\nWorktrees:\n");
            for wt in wts.iter().take(8) {
                let dirty = if crate::gitinfo::dirty_paths(&wt.path).is_empty() {
                    ""
                } else {
                    " [dirty]"
                };
                s.push_str(&format!(
                    "  • {} ({}){dirty}{}\n",
                    wt.path.display(),
                    wt.display_branch(),
                    if wt.is_main { " [main]" } else { "" }
                ));
            }
            if wts.len() > 8 {
                s.push_str(&format!("  … +{} more\n", wts.len() - 8));
            }
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
    fn preview_compaction_flag_only_when_content_dropped() {
        // Fresh conversation: empty recent text must not suggest compaction.
        let fresh = assemble_prompt(
            HardwareProfile::Balanced,
            SYSTEM,
            Some(AGENTS),
            Some(ANCHOR),
            None,
            "",
            "",
            vec![],
            "",
        );
        assert!(!fresh.soft_compaction_needed);
        assert!(!fresh.truncated);

        // Recent content that fits is kept whole: no flag.
        let kept = assemble_prompt(
            HardwareProfile::Balanced,
            SYSTEM,
            Some(AGENTS),
            Some(ANCHOR),
            None,
            "",
            "",
            vec![],
            "user: hello",
        );
        assert!(!kept.soft_compaction_needed);

        // Recent content cut by a blown stable prefix: flag on.
        let sys = "s\n".repeat(20000);
        let recent = "user: hello\n".repeat(500);
        let cut = assemble_prompt(
            HardwareProfile::Low,
            &sys,
            None,
            None,
            None,
            "",
            "",
            vec![],
            &recent,
        );
        assert!(cut.soft_compaction_needed);
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
            "",
            vec![],
            "",
        );
        assert!(!doc.text.contains("## Current Task State"));
        assert!(!doc.tiers.iter().any(|t| t.name == "state.md"));
    }

    #[test]
    fn stable_prefix_honors_fixed_order_and_marker() {
        let doc = assemble_prompt(
            HardwareProfile::Balanced,
            SYSTEM,
            Some(AGENTS),
            Some(ANCHOR),
            None,
            "",
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
            "",
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
            "",
            Vec::new(),
            "",
        );
        assert!(!doc.text.is_empty());
        assert!(doc.text.contains(STABLE_END_MARKER));
        assert_eq!(doc.retrieved_total, 0);
    }

    #[test]
    fn the_repository_sections_say_they_are_data_not_instructions() {
        // SE-2: a repository-derived section that lands inside the user turn
        // must not ride in unlabelled next to the user's own words.
        let assembly = assemble_chat(sample_chat_input(sample_retrieved(), &[]));
        let last = assembly.turns.last().unwrap().content.clone();
        for section in ["## Git", "## Retrieval"] {
            let at = last
                .find(section)
                .unwrap_or_else(|| panic!("{section} missing from the turn:\n{last}"));
            assert!(
                last[at..].starts_with(&format!("{section}\n\n{REPO_DATA_NOTE}")),
                "{section} no longer opens with its data attribution:\n{}",
                &last[at..last.len().min(at + 140)]
            );
        }
        // The preview assembler the `/ctx` panel shows says the same thing the
        // live turn says, because a preview that disagrees is a false preview.
        let doc = assemble_prompt(
            HardwareProfile::Balanced,
            SYSTEM,
            None,
            None,
            None,
            "main @ abc1234",
            "",
            sample_retrieved(),
            "",
        );
        assert!(doc.text.contains(&format!("## Git\n\n{REPO_DATA_NOTE}")));
        assert!(
            doc.text
                .contains(&format!("## Retrieval\n\n{REPO_DATA_NOTE}")),
            "{}",
            doc.text
        );
    }

    /// PR-3: a credential in a *dynamic* tier is held back from what leaves the
    /// machine, while the stable head is left exactly as a key/value cache
    /// expects it. The same turn, asked twice with and without the secret, must
    /// produce a byte-identical `turns[0]` or llama.cpp's prefix reuse breaks.
    #[test]
    fn a_secret_in_a_dynamic_tier_is_held_back_and_the_stable_head_is_untouched() {
        let clean = assemble_chat({
            let mut input = sample_chat_input(Vec::new(), &[]);
            input.prompt = "where is the login handler?";
            input
        });
        let secret = assemble_chat({
            let mut input = sample_chat_input(Vec::new(), &[]);
            input.prompt = "run it with AWS_SECRET_ACCESS_KEY=\"FAKE_NOT_A_REAL_SECRET_KEY\"";
            input
        });

        assert_eq!(
            clean.turns[0].content, secret.turns[0].content,
            "the stable head changed once a secret appeared in a lower tier"
        );
        assert!(
            !secret.turns[0]
                .content
                .contains("FAKE_NOT_A_REAL_SECRET_KEY"),
            "the stable head must never carry the secret either way"
        );

        let last = secret.turns.last().unwrap().content.clone();
        assert!(
            !last.contains("FAKE_NOT_A_REAL_SECRET_KEY"),
            "the raw secret must not be in the offered turn:\n{last}"
        );
        assert!(
            last.contains("«xencode-secret-1»"),
            "the turn shows the placeholder instead:\n{last}"
        );
        assert_eq!(secret.vault.len(), 1, "exactly one secret held back");
        // The value never named in the placeholder form is recoverable for the
        // run that must actually use it.
        assert!(secret
            .vault
            .restore(&last)
            .contains("FAKE_NOT_A_REAL_SECRET_KEY"));
    }

    #[test]
    fn a_secret_free_turn_holds_back_nothing() {
        let assembly = assemble_chat(sample_chat_input(sample_retrieved(), &[]));
        assert!(assembly.vault.is_empty());
        // A turn with nothing to redact must come back exactly as assembled —
        // the placeholder machinery is invisible when there is no credential.
        assert!(assembly
            .turns
            .last()
            .unwrap()
            .content
            .contains("where is the login handler?"));
    }

    #[test]
    fn a_source_line_survives_the_history_budget_trim() {
        // The budget trims history by dropping whole turns, newest first.
        // Whatever survives must survive byte for byte: a marker shaved off
        // by sizing is a marker lost.
        let marked: Vec<(String, String)> = (0..24)
            .map(|i| {
                (
                    "tool".to_string(),
                    format!(
                        "[data] run_command git log -n {i}\n{}",
                        "commit abcd ".repeat(40)
                    ),
                )
            })
            .collect();
        let assembly = assemble_chat(ChatInput {
            profile: HardwareProfile::Low,
            history: &marked,
            ..sample_chat_input(Vec::new(), &[])
        });
        let kept: Vec<&String> = assembly
            .turns
            .iter()
            .filter(|t| t.content.starts_with("[data] "))
            .map(|t| &t.content)
            .collect();
        assert!(!kept.is_empty(), "every marked turn was dropped");
        for turn in &kept {
            assert!(
                marked.iter().any(|(_, c)| c == *turn),
                "a kept turn is not the bytes it was written as:\n{}",
                &turn[..turn.len().min(120)]
            );
        }
        assert!(
            assembly
                .turns
                .iter()
                .any(|t| t.content == *marked.last().unwrap().1),
            "the most recent marked turn must be the one that survives"
        );
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
            repo_map: "",
            retrieved,
            attached_block: "",
            history,
            prompt: "where is the login handler?",
        }
    }

    #[test]
    fn a_turn_reports_which_class_owns_its_tokens() {
        // QK-3's second consumer: the ledger `/egress` reads. A tier that named
        // no class would silently vanish from this report.
        let history = sample_history();
        let retrieved = vec![RetrievedBlock {
            path: "src/auth.rs".to_string(),
            score: 1,
            body: "File: src/auth.rs\n```rust\nfn login() {}\n```".to_string(),
        }];
        let assembly = assemble_chat(ChatInput {
            attached_block: "<file path=\"src/auth.rs\">\n```rust\nfn login() {}\n```\n</file>",
            ..sample_chat_input(retrieved, &history)
        });
        let totals = assembly.source_totals();
        let named: Vec<&str> = totals.iter().map(|(class, _)| class.name()).collect();
        assert!(named.contains(&"system prompt"), "{named:?}");
        assert!(named.contains(&"your words"), "{named:?}");
        assert!(named.contains(&"repository"), "{named:?}");
        assert!(named.contains(&"attached file"), "{named:?}");
        assert!(named.contains(&"conversation"), "{named:?}");
        // Every token in a tier is counted once, and only data classes are the
        // ones the report has to flag as not-instructions.
        let sum: u64 = totals.iter().map(|(_, tokens)| tokens).sum();
        assert_eq!(sum, assembly.tiers.iter().map(|t| t.tokens).sum::<u64>());
        assert!(
            totals
                .iter()
                .any(|(class, _)| class.is_data() && class.name() == "attached file"),
            "an attachment is data: {named:?}"
        );
    }

    #[test]
    fn an_attached_file_reaches_the_turn_labelled_as_data() {
        let history = sample_history();
        let body = "<file path=\"src/auth.rs\">\n```rust\nfn login() {}\n```\n</file>";
        let assembly = assemble_chat(ChatInput {
            attached_block: body,
            ..sample_chat_input(Vec::new(), &history)
        });
        let user = assembly.turns.last().expect("a user turn");
        let note_at = user
            .content
            .find(crate::source::ATTACHED_DATA_NOTE)
            .expect("the attachment section carries no marker");
        let body_at = user
            .content
            .find(body)
            .expect("the attachment did not reach the turn");
        assert!(note_at < body_at, "the marker must lead its bytes");
    }

    #[test]
    fn an_untrusted_agent_file_is_counted_as_data_and_a_trusted_one_is_not() {
        let history = sample_history();
        let trusted = assemble_chat(sample_chat_input(Vec::new(), &history));
        let class_of = |a: &ChatAssembly| {
            a.tiers
                .iter()
                .find(|t| t.name == "agents.md")
                .map(|t| t.class)
                .expect("the agents tier is always reported")
        };
        assert_eq!(
            class_of(&trusted),
            crate::SourceClass::AgentFile { trusted: true }
        );

        let bannered = format!("{}\n{AGENTS}", crate::trust::UNTRUSTED_BANNER);
        let untrusted = assemble_chat(ChatInput {
            agents_md: Some(&bannered),
            ..sample_chat_input(Vec::new(), &history)
        });
        assert_eq!(
            class_of(&untrusted),
            crate::SourceClass::AgentFile { trusted: false }
        );
        assert!(
            !class_of(&untrusted).may_persist_durable(),
            "bytes nobody trusted may not become durable knowledge"
        );
    }

    /// The map tier, in the words the assemblers will show the model.
    fn sample_map() -> String {
        "Repo map — files nearest the current work, most depended-on first, names only:\n\
          • src/auth.rs [the current work]: Session, login\n"
            .to_string()
    }

    #[test]
    fn a_low_budget_prompt_is_offered_the_repo_map_and_a_wide_one_is_not() {
        let low = assemble_prompt(
            HardwareProfile::Low,
            SYSTEM,
            None,
            None,
            None,
            "",
            &sample_map(),
            Vec::new(),
            "user: hello",
        );
        assert!(low.text.contains("## Repo Map"), "{}", low.text);
        assert!(low.tiers.iter().any(|t| t.name == "repo map"));
        assert!(
            low.total_tokens <= low.target_tokens,
            "{} over {}",
            low.total_tokens,
            low.target_tokens
        );

        let high = assemble_prompt(
            HardwareProfile::High,
            SYSTEM,
            None,
            None,
            None,
            "",
            &sample_map(),
            Vec::new(),
            "user: hello",
        );
        assert!(!high.text.contains("## Repo Map"));
        assert!(!high.tiers.iter().any(|t| t.name == "repo map"));
    }

    #[test]
    fn the_map_is_paid_for_before_the_files_it_orients() {
        // Admitting the tier is charged to the budget, not slipped in, and it
        // sits above the bodies so it orients the reader before them.
        let map = sample_map();
        let low = assemble_chat(ChatInput {
            profile: HardwareProfile::Low,
            context_window: Some(HardwareProfile::Low.ctx_tokens() as u32),
            repo_map: &map,
            ..sample_chat_input(sample_retrieved(), &[])
        });
        assert!(
            budget_wants_repo_map(low.target_tokens),
            "a Low window is a Low-budget prompt"
        );
        let last = low.turns.last().unwrap().content.clone();
        assert!(last.contains("## Repo Map"), "{last}");
        assert!(
            last.find("## Repo Map") < last.find("## Retrieval"),
            "the map comes before the bodies: {last}"
        );
        assert!(
            low.total_tokens <= low.target_tokens,
            "{} over {}",
            low.total_tokens,
            low.target_tokens
        );
        assert!(low.tiers.iter().any(|t| t.name == "repo map"));

        let balanced = assemble_chat(ChatInput {
            repo_map: &map,
            ..sample_chat_input(sample_retrieved(), &[])
        });
        assert!(!balanced
            .turns
            .last()
            .unwrap()
            .content
            .contains("## Repo Map"));
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
            "",
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
        // The trace asks which files those were, not just how many.
        assert_eq!(
            chat.retrieved_files,
            vec!["src/auth.rs".to_string(), "src/database.rs".to_string()]
        );
        assert_eq!(chat.history_kept, 2);
        assert_eq!(chat.history_total, 2);
        assert!(chat.total_tokens <= chat.target_tokens);
        assert!(!chat.truncated);
    }

    /// What a server gets asked to count when it is asked to count the prompt:
    /// every turn's text, in the order the model is shown them, with nothing
    /// from the message list (roles) mixed in.
    #[test]
    fn prompt_text_holds_every_turn_in_order() {
        let history = sample_history();
        let chat = assemble_chat(sample_chat_input(sample_retrieved(), &history));
        let text = chat.prompt_text();
        let mut past = 0usize;
        for turn in &chat.turns {
            let found = text[past..]
                .find(&turn.content)
                .unwrap_or_else(|| panic!("the {} turn is not in the counted text", turn.role));
            past += found + turn.content.len();
        }
        // The budgeter's own figure covers this text plus the framing and the
        // harder divisor for code, so it can never be below a plain-prose count
        // of the same bytes.
        // Nothing is added to or taken from a turn on the way here: the only
        // extra bytes are the blank line between pieces, so a count of this text
        // is a count of exactly what the budget kept.
        let turns: usize = chat.turns.iter().map(|t| t.content.len()).sum();
        assert_eq!(text.len(), turns + 2 * (chat.turns.len() - 1));
    }

    /// The trace names the files a turn was shown, and only the ones the budget
    /// actually kept.
    #[test]
    fn chat_lists_the_retrieved_files_it_kept_and_not_the_ones_it_trimmed() {
        let long_retrieved: Vec<RetrievedBlock> = (0..20)
            .map(|i| RetrievedBlock {
                path: format!("src/f{i}.rs"),
                score: 10 + i as u64,
                body: format!(
                    "File: src/f{i}.rs\n```rust\n{}\n```",
                    "pub fn x(i: u64) -> u64 { i * 2 }\n".repeat(40)
                ),
            })
            .collect();
        let input = ChatInput {
            profile: HardwareProfile::Low,
            ..sample_chat_input(long_retrieved, &[])
        };
        let chat = assemble_chat(input);
        assert!(chat.retrieved_included > 0);
        assert!(chat.retrieved_included < chat.retrieved_total);
        assert_eq!(chat.retrieved_files.len(), chat.retrieved_included);
        // In the order they were offered, which is the order the bodies went in.
        assert_eq!(chat.retrieved_files[0], "src/f0.rs");
        for (index, path) in chat.retrieved_files.iter().enumerate() {
            assert_eq!(*path, format!("src/f{index}.rs"));
            assert!(chat.turns.last().unwrap().content.contains(path));
        }
        let trimmed = format!("src/f{}.rs", chat.retrieved_total - 1);
        assert!(!chat
            .turns
            .last()
            .unwrap()
            .content
            .contains(&format!("File: {trimmed}")));
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
        let live = collect_live_context(
            &dir,
            "authenticate",
            ContextCaps::from_profile(HardwareProfile::Balanced),
        );
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
        let live = collect_live_context(
            &dir,
            "authenticate",
            ContextCaps::from_profile(HardwareProfile::Balanced),
        );
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
            repo_map: "",
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
