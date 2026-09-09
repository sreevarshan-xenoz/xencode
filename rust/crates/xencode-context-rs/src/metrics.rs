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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("xencode-metrics-test-{stamp}"))
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
}