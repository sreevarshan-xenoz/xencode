//! Hardware profiles + deterministic token estimator (§10, §14).
//!
//! Profiles are data, never hardcoded logic — logic reads the fields here.
//! The estimator is deliberately cheap and only used for budgeting; real
//! token counts come from llama.cpp `usage` and are what the metrics layer
//! displays.

/// VRAM-based inference profiles (§14 defaults).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareProfile {
    /// <4 GB — smallest context, aggressive compaction.
    Low,
    /// 4–8 GB — the Xencode default target.
    Balanced,
    /// 8 GB+ — room for larger contexts.
    High,
}

impl HardwareProfile {
    pub fn name(self) -> &'static str {
        match self {
            HardwareProfile::Low => "LOW",
            HardwareProfile::Balanced => "BALANCED",
            HardwareProfile::High => "HIGH",
        }
    }

    /// Model context window in tokens.
    pub fn ctx_tokens(self) -> u64 {
        match self {
            HardwareProfile::Low => 4096,
            HardwareProfile::Balanced => 8192,
            HardwareProfile::High => 16384,
        }
    }

    /// Fraction of the context window the context builder may fill.
    pub fn utilization(self) -> f64 {
        match self {
            HardwareProfile::Low => 0.60,
            HardwareProfile::Balanced => 0.75,
            HardwareProfile::High => 0.85,
        }
    }

    /// Retrieval top-K cap.
    pub fn top_k(self) -> usize {
        match self {
            HardwareProfile::Low => 3,
            HardwareProfile::Balanced => 5,
            HardwareProfile::High => 8,
        }
    }

    /// Per-file content cap in characters (injected file bodies).
    pub fn content_cap_chars(self) -> usize {
        match self {
            HardwareProfile::Low => 8_000,
            HardwareProfile::Balanced => 16_000,
            HardwareProfile::High => 24_000,
        }
    }

    /// (soft, hard) compaction trigger fractions — used by M3.
    pub fn compaction_pct(self) -> (f64, f64) {
        match self {
            HardwareProfile::Low => (0.60, 0.80),
            HardwareProfile::Balanced => (0.70, 0.90),
            HardwareProfile::High => (0.80, 0.90),
        }
    }
}

/// Deterministic token estimate: `ceil(chars / 4)` prose, `ceil(chars / 3)`
/// for code bodies (§10).
pub fn est_tokens(chars: usize, is_code: bool) -> u64 {
    let divisor = if is_code { 3 } else { 4 };
    ((chars + divisor - 1) / divisor) as u64
}

/// Take the head of `text` up to `max_tokens`, cutting at a line boundary, and
/// return `(kept, kept_tokens)`.
pub fn truncate_to_tokens(text: &str, max_tokens: u64, is_code: bool) -> (String, u64) {
    if max_tokens == 0 {
        return (String::new(), 0);
    }
    let divisor = if is_code { 3 } else { 4 } as u64;
    let max_chars = (max_tokens * divisor) as usize;
    if text.len() <= max_chars {
        return (text.to_string(), est_tokens(text.len(), is_code));
    }
    let mut cut = max_chars.min(text.len());
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    if let Some(rel) = text[..cut].rfind('\n') {
        cut = rel + 1;
    }
    let head = &text[..cut];
    (head.to_string(), est_tokens(head.len(), is_code))
}

/// Take the *tail* of `text` up to `max_tokens` so the most recent content
/// wins (§10 tier 7: "most-recent messages win (drop oldest first)").
pub fn truncate_tail_to_tokens(text: &str, max_tokens: u64, is_code: bool) -> (String, u64) {
    if max_tokens == 0 {
        return (String::new(), 0);
    }
    let divisor = if is_code { 3 } else { 4 } as u64;
    let max_chars = (max_tokens * divisor) as usize;
    if text.len() <= max_chars {
        return (text.to_string(), est_tokens(text.len(), is_code));
    }
    let mut start = text.len() - max_chars;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    let tail = &text[start..];
    // Back up to a line boundary for a cleaner cut when possible.
    let cut = tail.find('\n').map(|i| i + 1).unwrap_or(tail.len());
    let tail = &tail[cut.min(tail.len())..];
    (tail.to_string(), est_tokens(tail.len(), is_code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimators_are_ceil_based() {
        assert_eq!(est_tokens(0, false), 0);
        assert_eq!(est_tokens(4, false), 1);
        assert_eq!(est_tokens(5, false), 2);
        assert_eq!(est_tokens(4, true), 2);
        assert_eq!(est_tokens(3, true), 1);
    }

#[test]
    fn truncate_cuts_at_line_boundary_and_under_cap() {
        let text = "aaaaaa\nbbbbbb\ncccccc\nddddd\n";
        let (kept, tokens) = truncate_to_tokens(text, 3, false);
        // 3 tokens prose = 12 chars max → only the first line fits.
        assert_eq!(kept, "aaaaaa\n");
        assert_eq!(kept.len(), 7);
        assert_eq!(tokens, est_tokens(kept.len(), false));
        assert!(tokens <= 3);
    }

    #[test]
    fn truncate_keeps_short_text_untouched() {
        let (kept, tokens) = truncate_to_tokens("short", 10, false);
        assert_eq!(kept, "short");
        assert_eq!(tokens, est_tokens(5, false));
    }

    #[test]
    fn truncate_tail_keeps_most_recent_content() {
        let text = "line1-old\nline2-mid\nline3-new\n";
        let (kept, _) = truncate_tail_to_tokens(text, 3, false);
        assert!(kept.contains("line3-new"));
        assert!(!kept.contains("line1-old"));
        // Short text passes through whole.
        let (kept, _) = truncate_tail_to_tokens(text, 100, false);
        assert_eq!(kept, text);
    }

    #[test]
    fn profiles_have_monotonic_resources() {
        assert!(HardwareProfile::Low.ctx_tokens() < HardwareProfile::Balanced.ctx_tokens());
        assert!(
            HardwareProfile::Balanced.ctx_tokens() < HardwareProfile::High.ctx_tokens()
        );
        assert!(HardwareProfile::Low.top_k() < HardwareProfile::Balanced.top_k());
        assert_ne!(HardwareProfile::Balanced.top_k(), HardwareProfile::High.top_k());
    }
}