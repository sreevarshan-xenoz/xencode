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
use crate::retrieve::RetrievedFile;
use sha2::{Digest, Sha256};
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

/// Build the promoted-tier prompt per §10.
///
/// `retrieved` must already be sorted best-first; the budgeter trims from the
/// bottom. `recent_text` should be the rolling message window, oldest first.
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
    let system_tok = est_tokens(system.len(), false);
    tiers.push(TierDoc {
        name: "system",
        tokens: system_tok,
    });

    let (agents_head, agents_tok) =
        truncate_to_tokens(agents_md.unwrap_or(""), AGENTS_CAP_TOKENS, false);
    let truncated_agents = agents_md.is_some_and(|a| a.len() > agents_head.len());
    tiers.push(TierDoc {
        name: "agents.md",
        tokens: agents_tok,
    });

    let (anchor_head, anchor_tok) =
        truncate_to_tokens(anchor_md.unwrap_or(""), ANCHOR_CAP_TOKENS, false);
    let truncated_anchor = anchor_md.is_some_and(|a| a.len() > anchor_head.len());
    tiers.push(TierDoc {
        name: "anchor.md",
        tokens: anchor_tok,
    });

    let mut stable_parts: Vec<&str> = Vec::new();
    if !system.is_empty() {
        stable_parts.push(system);
    }
    if !agents_head.is_empty() {
        stable_parts.push(&agents_head);
    }
    if !anchor_head.is_empty() {
        stable_parts.push(&anchor_head);
    }
    stable_parts.push(STABLE_END_MARKER);
    let stable_prefix = stable_parts.join("\n\n");
    let stable_tokens = system_tok + agents_tok + anchor_tok;
    let mut remaining = target.saturating_sub(stable_tokens);
    truncated |= truncated_agents || truncated_anchor || stable_tokens > target;

    // ── Tier 4: state.md ─────────────────────────────────────────────────
    let (state_head, state_tok) =
        truncate_to_tokens(state_md.unwrap_or(""), STATE_CAP_TOKENS, false);
    if !state_head.is_empty() && remaining >= MARGIN_TOKENS {
        tiers.push(TierDoc {
            name: "state.md",
            tokens: state_tok,
        });
        remaining = remaining.saturating_sub(state_tok);
    }

    // ── Tier 5: git summary ──────────────────────────────────────────────
    let (git_head, git_tok) = truncate_to_tokens(git_summary, GIT_CAP_TOKENS, false);
    if !git_head.is_empty() && remaining >= MARGIN_TOKENS {
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
    if !state_head.is_empty() {
        text.push_str("\n\n## Current Task State\n\n");
        text.push_str(&state_head);
    }
    if !git_head.is_empty() {
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
    fn stable_prefix_honors_fixed_order_and_marker() {
        let doc = assemble_prompt(
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
