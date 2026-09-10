//! Per-request context metrics (§16) — appended to `cache/metrics.jsonl`.
//!
//! Every LLM request that goes through the context builder records a row so
//! we can measure the two things M2 exists to validate: is deterministic
//! retrieval finding the right files, and how much of the context window is
//! actually being used (incl. the KV `cached_tokens` reuse once providers
//! expose it).

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompactAction {
    None,
    Soft,
    #[default]
    Hard,
}

/// One row of `.xencode/cache/metrics.jsonl` (§16 schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// UTC epoch millis when the request completed.
    pub ts_unix_ms: u64,
    /// Hardware profile name (`LOW` / `BALANCED` / `HIGH`).
    pub profile: String,
    /// Model context window size in tokens.
    pub context_limit: u32,
    /// Total prompt tokens (from llama.cpp `usage`).
    pub prompt_tokens: u32,
    /// `prompt_tokens - prompt_tokens_evaluated` → KV-cache reuse.
    pub cached_tokens: u32,
    pub completion_tokens: u32,
    /// 0.0..=1.0 — prompt_tokens / context_limit.
    pub context_usage: f32,
    pub generation_tok_s: f32,
    pub prompt_tok_s: f32,
    pub retrieved_files: u8,
    pub compaction: CompactAction,
}

impl RequestMetrics {
    /// A row with zeroed counters, ready for the caller to fill in.
    pub fn new(profile: &str, context_limit: u32) -> Self {
        Self {
            ts_unix_ms: 0,
            profile: profile.to_string(),
            context_limit,
            prompt_tokens: 0,
            cached_tokens: 0,
            completion_tokens: 0,
            context_usage: 0.0,
            generation_tok_s: 0.0,
            prompt_tok_s: 0.0,
            retrieved_files: 0,
            compaction: CompactAction::None,
        }
    }

    /// Fill in the llama.cpp-driven counters in one call (§13). `cached_tokens`
    /// is the KV-cache win: total prompt tokens minus what was actually
    /// re-evaluated this request.
    #[allow(clippy::too_many_arguments)]
    pub fn from_timings(
        profile: &str,
        context_limit: u32,
        prompt_tokens: u32,
        tokens_evaluated: u32,
        completion_tokens: u32,
        generation_tok_s: f32,
        prompt_tok_s: f32,
        retrieved_files: u8,
    ) -> Self {
        let mut m = Self::new(profile, context_limit);
        m.ts_unix_ms = crate::conversation::now_millis();
        m.prompt_tokens = prompt_tokens;
        m.cached_tokens = prompt_tokens.saturating_sub(tokens_evaluated);
        m.completion_tokens = completion_tokens;
        m.context_usage = if context_limit > 0 {
            (prompt_tokens as f32 / context_limit as f32).min(1.0)
        } else {
            0.0
        };
        m.generation_tok_s = generation_tok_s;
        m.prompt_tok_s = prompt_tok_s;
        m.retrieved_files = retrieved_files;
        m
    }

    /// 0.0..=1.0 — fraction of the prompt served from the KV cache.
    pub fn kv_reuse_ratio(&self) -> f32 {
        if self.prompt_tokens == 0 {
            0.0
        } else {
            (self.cached_tokens as f32 / self.prompt_tokens as f32).clamp(0.0, 1.0)
        }
    }

    /// Last row recorded for each profile, most recent win (metrics already
    /// arrive in append order).
    pub fn latest_per_profile(rows: &[RequestMetrics]) -> Vec<&RequestMetrics> {
        let mut out: Vec<&RequestMetrics> = Vec::new();
        for row in rows.iter().rev() {
            if !out.iter().any(|r| r.profile == row.profile) {
                out.push(row);
            }
        }
        out
    }
}

pub fn metrics_path(xencode_dir: &Path) -> std::path::PathBuf {
    xencode_dir.join("cache").join("metrics.jsonl")
}

/// Append one JSON line to `metrics.jsonl` (append-only; safe to call
/// concurrently as lines are O_APPEND writes).
pub fn append_metrics(xencode_dir: &Path, m: &RequestMetrics) -> std::io::Result<()> {
    let path = metrics_path(xencode_dir);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let line = serde_json::to_string(m)?;
    use std::io::Write;
    writeln!(file, "{line}")
}

/// Read every row recorded so far; corrupt lines are skipped.
pub fn read_metrics(xencode_dir: &Path) -> Vec<RequestMetrics> {
    let Ok(text) = fs::read_to_string(metrics_path(xencode_dir)) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
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
        std::env::temp_dir().join(format!("xencode-metrics-test-{unique}"))
    }

    #[test]
    fn appends_and_reads_rows_round_trip() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");

        let mut m1 = RequestMetrics::new("BALANCED", 8192);
        m1.prompt_tokens = 5760;
        m1.cached_tokens = 4912;
        m1.completion_tokens = 130;
        m1.context_usage = 0.72;
        m1.retrieved_files = 5;
        append_metrics(&xencode, &m1).unwrap();

        let mut m2 = RequestMetrics::new("LOW", 4096);
        m2.prompt_tokens = 2048;
        append_metrics(&xencode, &m2).unwrap();

        let rows = read_metrics(&xencode);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].cached_tokens, 4912);
        assert_eq!(rows[1].profile, "LOW");

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_file_reads_empty() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        assert!(read_metrics(&dir.join(".xencode")).is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_timings_derives_cached_tokens_and_kv_reuse() {
        let m = RequestMetrics::from_timings("BALANCED", 8192, 5760, 848, 130, 12.3, 400.0, 5);
        assert_eq!(m.cached_tokens, 4912);
        assert!((m.kv_reuse_ratio() - 4912.0 / 5760.0).abs() < 1e-5);
        assert!((m.context_usage - 5760.0 / 8192.0).abs() < 1e-5);
        assert_eq!(m.generation_tok_s, 12.3);
        assert_eq!(m.completion_tokens, 130);

        let cold = RequestMetrics::from_timings("LOW", 4096, 2048, 2048, 0, 0.0, 0.0, 0);
        assert_eq!(cold.kv_reuse_ratio(), 0.0);
        assert_eq!(cold.cached_tokens, 0);
    }

    #[test]
    fn latest_per_profile_prefers_most_recent() {
        let mut a = RequestMetrics::new("BALANCED", 8192);
        a.cached_tokens = 100;
        let mut b = RequestMetrics::new("BALANCED", 8192);
        b.cached_tokens = 300;
        let mut c = RequestMetrics::new("LOW", 4096);
        c.cached_tokens = 50;
        let rows = vec![a, b, c];
        let latest = RequestMetrics::latest_per_profile(&rows);
        assert_eq!(latest.len(), 2);
        let bal = latest.iter().find(|r| r.profile == "BALANCED").unwrap();
        assert_eq!(bal.cached_tokens, 300);
    }
}
