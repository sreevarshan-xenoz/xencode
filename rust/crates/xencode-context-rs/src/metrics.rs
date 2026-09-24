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

/// Where a recorded request was served from. Written as `local` or `cloud`,
/// and `null` on rows from before this field existed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MetricSource {
    Local,
    Cloud,
}

/// Which conversation and which model a row belongs to, so a writer can stamp
/// a row without knowing how each field is derived.
#[derive(Debug, Clone, Default)]
pub struct MetricsIdentity {
    pub session_id: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub source: Option<MetricSource>,
}

impl MetricsIdentity {
    pub fn apply(self, row: &mut RequestMetrics) {
        row.session_id = self.session_id;
        row.model = self.model;
        row.provider = self.provider;
        row.source = self.source;
    }
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
    /// The conversation this request belonged to. Without it every figure can
    /// only be an average across all sessions at once.
    #[serde(default)]
    pub session_id: Option<String>,
    /// The model id as it was asked for, prefix included.
    #[serde(default)]
    pub model: Option<String>,
    /// Which client the id resolved to (`ollama`, `llamacpp`, `remote`,
    /// `openrouter`, `qwen`, …).
    #[serde(default)]
    pub provider: Option<String>,
    /// Whether the prompt left this machine.
    #[serde(default)]
    pub source: Option<MetricSource>,
    /// Estimated cost in micro-dollars. Nothing computes a price yet, so this
    /// is `None` on every row written today; it exists so the cost work has
    /// somewhere to land without changing the schema again.
    #[serde(default)]
    pub est_cost_micros: Option<u64>,
    /// Power draw sampled while the request ran, in watts. Also unmeasured
    /// until the hardware sampling exists.
    #[serde(default)]
    pub power_w: Option<f32>,
    /// The temperature this turn was asked to sample at, as it was sent. `None`
    /// means nothing went over the wire, which is different from `0.0` — the
    /// server picked its own, so the answer cannot be repeated.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// The seed the sampler was told to use. See `temperature`: absent means the
    /// server chose one per request and the run is not reproducible. A negative
    /// seed asks the server to keep choosing, so it means the same as absent.
    #[serde(default)]
    pub seed: Option<i64>,
    /// Which set of instructions the turn was asked to obey, as a short digest of
    /// the prompt registry ([`crate::prompts::set_version`]). A row written before
    /// prompts were versioned reads back as `None`, which is the honest answer:
    /// that build did not know what its prompts were worth.
    #[serde(default)]
    pub prompt_version: Option<String>,
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
            session_id: None,
            model: None,
            provider: None,
            source: None,
            est_cost_micros: None,
            power_w: None,
            temperature: None,
            seed: None,
            prompt_version: Some(crate::prompts::set_version().to_string()),
        }
    }

    /// Whether the turn this row describes could be run again for the same
    /// answer: a seed names the sampler's draws, and a temperature of zero takes
    /// the best token every time, which makes the seed irrelevant. Anything else
    /// — including a row written before either field existed — ran on sampling
    /// the server picked for itself.
    ///
    /// A negative seed is not a pin: llama.cpp documents `-1` as "use a random
    /// seed", so a row that says it was sent `-1` asked for a fresh draw as
    /// surely as one that sent nothing.
    pub fn repeatable(&self) -> bool {
        matches!(self.seed, Some(seed) if seed >= 0)
            || matches!(self.temperature, Some(temp) if temp == 0.0)
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

/// How many bytes from the end of `metrics.jsonl` a tail read looks at. One
/// record is a couple of hundred bytes, so this window holds a thousand or more.
const TAIL_READ_BYTES: u64 = 256 * 1024;

/// The last `limit` records, oldest first like [`read_metrics`], without reading
/// the whole file. Callers that show "the recent turns" used to parse everything
/// to pick out a handful of lines from the end.
///
/// Falls back to reading the file when the window did not hold `limit` records,
/// so asking for more than fits is answered rather than quietly truncated.
pub fn read_metrics_tail(xencode_dir: &Path, limit: usize) -> Vec<RequestMetrics> {
    use std::io::{Read, Seek, SeekFrom};

    if limit == 0 {
        return Vec::new();
    }
    let path = metrics_path(xencode_dir);
    let len = match fs::metadata(&path) {
        Ok(meta) => meta.len(),
        Err(_) => return Vec::new(),
    };
    let start = len.saturating_sub(TAIL_READ_BYTES);
    let mut bytes = Vec::new();
    if let Ok(mut file) = fs::File::open(&path) {
        if file.seek(SeekFrom::Start(start)).is_ok() {
            let _ = file.read_to_end(&mut bytes);
        }
    }
    // The window most likely starts inside a record: that fragment is dropped,
    // unless the read started at the top of the file.
    let body = if start == 0 {
        &bytes[..]
    } else {
        match bytes.iter().position(|byte| *byte == b'\n') {
            Some(newline) => &bytes[newline + 1..],
            // Not even one whole record in the window.
            None => return last_rows(&read_metrics(xencode_dir), limit),
        }
    };
    let mut rows = Vec::new();
    for line in body.split(|byte| *byte == b'\n').rev() {
        if line.is_empty() {
            continue;
        }
        if let Ok(row) = serde_json::from_slice::<RequestMetrics>(line) {
            rows.push(row);
            if rows.len() == limit {
                break;
            }
        }
    }
    if rows.len() < limit && start > 0 {
        return last_rows(&read_metrics(xencode_dir), limit);
    }
    rows.reverse();
    rows
}

fn last_rows(rows: &[RequestMetrics], limit: usize) -> Vec<RequestMetrics> {
    let skip = rows.len().saturating_sub(limit);
    rows.iter().skip(skip).cloned().collect()
}

/// Read every row recorded so far. A run that was killed while appending
/// leaves a partial last line, and `metrics.jsonl` is still readable
/// otherwise, so that line is dropped rather than failing the read.
pub fn read_metrics(xencode_dir: &Path) -> Vec<RequestMetrics> {
    xencode_core_rs::read_jsonl_tolerant(&metrics_path(xencode_dir)).rows
}

/// Read the rows appended since `offset`, and report where to carry on.
///
/// The file only grows, and a caller that wants the newest rows was re-reading
/// and re-parsing all of them to get them. This starts at a byte position
/// instead, and advances only over lines that are complete: a process killed
/// mid-append leaves a partial final line, and treating that as consumed would
/// strand the caller at a position where every later line fails to parse.
///
/// A file shorter than `offset` was replaced rather than appended to, so the
/// read restarts from the beginning. A line that does not parse at all is
/// skipped — rows this version of the reader does not recognise must not stop
/// a rollup from catching up on the ones it does.
pub fn read_metrics_since(xencode_dir: &Path, offset: u64) -> (Vec<RequestMetrics>, u64) {
    use std::io::{Read, Seek, SeekFrom};

    let path = metrics_path(xencode_dir);
    let start = match fs::metadata(&path) {
        Ok(meta) if meta.len() >= offset => offset,
        // Missing or truncated: nothing to catch up on, and the next append
        // should be read from the top of the new file.
        _ => 0,
    };
    let Ok(opened) = fs::File::open(&path) else {
        return (Vec::new(), start);
    };
    let mut file = opened;
    if file.seek(SeekFrom::Start(start)).is_err() {
        return (Vec::new(), start);
    }
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return (Vec::new(), start);
    }
    let Some(last_newline) = bytes.iter().rposition(|byte| *byte == b'\n') else {
        return (Vec::new(), start);
    };
    let mut rows = Vec::new();
    for line in bytes[..=last_newline].split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        if let Ok(row) = serde_json::from_slice::<RequestMetrics>(line) {
            rows.push(row);
        }
    }
    (rows, start + last_newline as u64 + 1)
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
    fn a_row_cut_off_by_a_crash_is_dropped_and_the_earlier_ones_survive() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let mut m = RequestMetrics::new("BALANCED", 8192);
        m.cached_tokens = 4912;
        append_metrics(&xencode, &m).unwrap();
        // Simulate the process being killed mid-append: the bytes written so
        // far stop inside the next record.
        use std::io::Write;
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(metrics_path(&xencode))
            .unwrap();
        f.write_all(b"{\"profile\":\"LOW\",\"max_context").unwrap();
        f.flush().unwrap();

        let rows = read_metrics(&xencode);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].cached_tokens, 4912);
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

    #[test]
    fn a_tail_read_gives_the_newest_rows_in_the_same_order_as_a_full_read() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        for i in 0..25u8 {
            let mut m = RequestMetrics::new("LOW", 4096);
            m.retrieved_files = i;
            append_metrics(&xencode, &m).unwrap();
        }

        let tail = read_metrics_tail(&xencode, 5);
        let all = read_metrics(&xencode);
        let ids = |rows: &[RequestMetrics]| -> Vec<u8> {
            rows.iter().map(|r| r.retrieved_files).collect()
        };
        assert_eq!(tail.len(), 5);
        assert_eq!(
            ids(&tail),
            vec![20, 21, 22, 23, 24],
            "the tail must be oldest-first, like a full read"
        );
        assert_eq!(ids(&tail), ids(&all[all.len() - 5..]));

        // Asking for more than exist is not an error and is not padded.
        assert_eq!(read_metrics_tail(&xencode, 500).len(), 25);
        assert!(read_metrics_tail(&xencode, 0).is_empty());
        assert!(read_metrics_tail(&dir.join("nowhere"), 5).is_empty());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_tail_read_of_a_half_written_last_line_returns_the_whole_rows() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let mut m = RequestMetrics::new("LOW", 4096);
        m.retrieved_files = 1;
        append_metrics(&xencode, &m).unwrap();
        m.retrieved_files = 2;
        append_metrics(&xencode, &m).unwrap();
        use std::io::Write;
        fs::OpenOptions::new()
            .append(true)
            .open(metrics_path(&xencode))
            .unwrap()
            .write_all(b"{\"profile\":\"LOW\",\"context")
            .unwrap();

        let tail = read_metrics_tail(&xencode, 10);
        assert_eq!(
            tail.iter().map(|r| r.retrieved_files).collect::<Vec<_>>(),
            vec![1, 2]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_row_written_before_the_identity_fields_existed_still_reads() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        fs::create_dir_all(xencode.join("cache")).unwrap();
        // Exactly the shape `metrics.jsonl` held before the schema grew: ten
        // fields, none of them new.
        fs::write(
            metrics_path(&xencode),
            br#"{"ts_unix_ms":1,"profile":"LOW","context_limit":4096,"prompt_tokens":10,"cached_tokens":4,"completion_tokens":2,"context_usage":0.002,"generation_tok_s":11.5,"prompt_tok_s":300.0,"retrieved_files":1,"compaction":"none"}
"#,
        )
        .unwrap();

        let rows = read_metrics(&xencode);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].generation_tok_s, 11.5);
        assert_eq!(rows[0].model, None);
        assert_eq!(rows[0].provider, None);
        assert_eq!(rows[0].session_id, None);
        assert_eq!(rows[0].source, None);
        assert_eq!(rows[0].est_cost_micros, None);
        assert_eq!(rows[0].power_w, None);
        assert_eq!(rows[0].temperature, None);
        assert_eq!(rows[0].seed, None);
        // A row from before prompts were versioned does not claim a prompt set.
        assert_eq!(rows[0].prompt_version, None);
        // Reading an old row is not the same as claiming it repeats: with no
        // sampling recorded, the only honest answer is that the server chose.
        assert!(!rows[0].repeatable());
        fs::remove_dir_all(dir).unwrap();
    }

    /// What makes a run repeatable is what went over the wire, so the row holds
    /// the settings rather than a verdict — a seed of `0` and a temperature of
    /// `0.0` are both real values, and both have to survive the round trip as
    /// themselves instead of collapsing into "not set".
    #[test]
    fn a_row_records_the_sampling_it_was_asked_for_and_says_whether_it_repeats() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let mut free = RequestMetrics::from_timings("LOW", 4096, 100, 100, 10, 11.0, 300.0, 1);
        free.temperature = Some(0.7);
        append_metrics(&xencode, &free).unwrap();

        let mut pinned = RequestMetrics::from_timings("LOW", 4096, 100, 100, 10, 11.0, 300.0, 1);
        pinned.temperature = Some(0.0);
        pinned.seed = Some(0);
        append_metrics(&xencode, &pinned).unwrap();

        let rows = read_metrics(&xencode);
        assert_eq!(rows.len(), 2);
        assert!(!rows[0].repeatable(), "0.7 with no seed is still a draw");
        assert!(rows[1].repeatable());
        assert_eq!(rows[1].temperature, Some(0.0));
        assert_eq!(rows[1].seed, Some(0));
        fs::remove_dir_all(dir).unwrap();
    }

    /// A turn has to say which instructions it was asked to obey, or the score it
    /// produced cannot be lined up with the one from before a prompt was reworded.
    #[test]
    fn every_row_records_the_prompt_set_this_build_carries() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let row = RequestMetrics::from_timings("LOW", 4096, 100, 100, 10, 11.0, 300.0, 1);
        assert_eq!(
            row.prompt_version.as_deref(),
            Some(crate::prompts::set_version()),
            "a fresh row names the prompts it was built with"
        );
        append_metrics(&xencode, &row).unwrap();
        let raw = fs::read_to_string(metrics_path(&xencode)).unwrap();
        assert!(
            raw.contains(r#""prompt_version":""#),
            "the digest never reached the file: {raw}"
        );
        assert_eq!(read_metrics(&xencode)[0].prompt_version, row.prompt_version);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_negative_seed_asks_for_a_fresh_draw_so_it_repeats_nothing() {
        let mut row = RequestMetrics::new("LOW", 4096);
        row.seed = Some(-1);
        assert!(!row.repeatable());
        // Greedy decoding repeats whatever the seed says, and a seed of its own
        // repeats whatever the temperature says.
        row.temperature = Some(0.0);
        assert!(row.repeatable());
        row.temperature = Some(0.001);
        row.seed = Some(1);
        assert!(row.repeatable());
    }

    #[test]
    fn a_stamped_row_carries_its_model_provider_session_and_destination() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let mut m = RequestMetrics::from_timings("BALANCED", 8192, 5760, 848, 130, 12.3, 400.0, 5);
        MetricsIdentity {
            session_id: Some("session_1700000000".to_string()),
            model: Some("qwen2.5:7b".to_string()),
            provider: Some("ollama".to_string()),
            source: Some(MetricSource::Local),
        }
        .apply(&mut m);
        append_metrics(&xencode, &m).unwrap();

        let raw = fs::read_to_string(metrics_path(&xencode)).unwrap();
        // Lowercase, as the schema says, so a reader can grep it.
        assert!(raw.contains(r#""source":"local""#), "{raw}");
        assert!(raw.contains(r#""model":"qwen2.5:7b""#), "{raw}");
        // Unmeasured fields are written as null rather than left out, so the
        // row says it was not measured instead of looking like an old row.
        assert!(raw.contains(r#""est_cost_micros":null"#), "{raw}");

        let rows = read_metrics(&xencode);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].session_id.as_deref(), Some("session_1700000000"));
        assert_eq!(rows[0].provider.as_deref(), Some("ollama"));
        assert_eq!(rows[0].source, Some(MetricSource::Local));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reading_since_an_offset_returns_only_the_new_rows_and_says_where_to_resume() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let first = RequestMetrics::new("LOW", 4096);
        append_metrics(&xencode, &first).unwrap();
        let after_first = metrics_path(&xencode).metadata().unwrap().len();

        let (rows, offset) = read_metrics_since(&xencode, 0);
        assert_eq!(rows.len(), 1);
        assert_eq!(offset, after_first);

        // Nothing new since: the same position comes back, and no rows.
        let (rows, offset_again) = read_metrics_since(&xencode, offset);
        assert!(rows.is_empty());
        assert_eq!(offset_again, offset);

        let mut second = RequestMetrics::new("HIGH", 32768);
        second.retrieved_files = 7;
        append_metrics(&xencode, &second).unwrap();
        let (rows, _) = read_metrics_since(&xencode, offset);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].profile, "HIGH");
        assert_eq!(rows[0].retrieved_files, 7);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_line_still_being_written_is_left_for_the_next_read() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let mut m = RequestMetrics::new("LOW", 4096);
        m.retrieved_files = 3;
        append_metrics(&xencode, &m).unwrap();
        let settled = metrics_path(&xencode).metadata().unwrap().len();
        // A writer that was killed partway through its next record.
        use std::io::Write;
        fs::OpenOptions::new()
            .append(true)
            .open(metrics_path(&xencode))
            .unwrap()
            .write_all(b"{\"profile\":\"HIGH\",\"context")
            .unwrap();

        let (rows, offset) = read_metrics_since(&xencode, settled);
        assert!(rows.is_empty(), "the torn line was consumed: {rows:?}");
        assert_eq!(offset, settled, "the reader moved past a half-written line");

        // Once that record is finished, asking from the same position picks it
        // up — the reader never advanced over the fragment.
        let mut finished = RequestMetrics::new("HIGH", 4096);
        finished.retrieved_files = 9;
        let bytes = fs::read(metrics_path(&xencode)).unwrap();
        let mut completed = String::from_utf8(bytes[..settled as usize].to_vec()).unwrap();
        completed.push_str(&serde_json::to_string(&finished).unwrap());
        completed.push('\n');
        fs::write(metrics_path(&xencode), completed).unwrap();
        let (rows, _) = read_metrics_since(&xencode, offset);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].retrieved_files, 9);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_replaced_file_is_read_from_the_start_again() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append_metrics(&xencode, &RequestMetrics::new("LOW", 4096)).unwrap();
        append_metrics(&xencode, &RequestMetrics::new("LOW", 4096)).unwrap();
        let long = metrics_path(&xencode).metadata().unwrap().len();

        // The file disappearing entirely is not an error, and the caller is
        // told to start from the top of whatever comes next.
        fs::remove_file(metrics_path(&xencode)).unwrap();
        let (rows, offset) = read_metrics_since(&xencode, long);
        assert!(rows.is_empty());
        assert_eq!(offset, 0);

        let mut fresh = RequestMetrics::new("HIGH", 8192);
        fresh.retrieved_files = 1;
        append_metrics(&xencode, &fresh).unwrap();
        let (rows, _) = read_metrics_since(&xencode, offset);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].profile, "HIGH");
        fs::remove_dir_all(dir).unwrap();
    }
}
