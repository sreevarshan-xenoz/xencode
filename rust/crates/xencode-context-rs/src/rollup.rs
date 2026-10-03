//! An incremental rollup over `metrics.jsonl` (§16, CX-1).
//!
//! Every figure the profiler, `/cost` and the status row show used to come from
//! re-reading and re-summing the whole metrics file, which only grows. This
//! folds each newly appended record into a small JSON sidecar
//! (`cache/metrics-rollup.json`) and resumes from a byte position, so the cost
//! of a refresh is proportional to what was written since the last time — one
//! turn — rather than to everything ever recorded.
//!
//! A refresh also bounds the file: once it has grown to twice [`METRICS_KEEP_BYTES`]
//! the rows that are already folded are cut off the front, so the record of every
//! request the project has made stops being a record that never ends.
//!
//! What is kept is a sum plus a fixed-size window of the recent rate samples:
//! the totals answer "how many tokens", the window answers "how fast, usually".
//! Percentiles are exact over that window and say how long the window is; no
//! estimate is presented as if it were measured over all rows.

use crate::metrics::{read_metrics_since, RequestMetrics};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Bumped when the sidecar holds something the previous shape did not. A
/// rollup written by another version is rebuilt from the records rather than
/// trusted. Version 2 added the repeatability counts, which a version 1 file
/// would read back as zeros — a wrong answer rather than a missing one.
pub const ROLLUP_VERSION: u8 = 2;

/// How many of the most recent rate samples are kept for the percentiles. The
/// window is the whole of what a percentile here can claim to cover.
pub const RATE_SAMPLE_WINDOW: usize = 512;

/// Token counts added up over some scope: everything, one session, one model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TokenTotals {
    /// How many records contributed.
    pub requests: u64,
    pub prompt_tokens: u64,
    /// The part of `prompt_tokens` served from the KV cache.
    pub cached_tokens: u64,
    pub completion_tokens: u64,
}

impl TokenTotals {
    fn add_record(&mut self, row: &RequestMetrics) {
        self.requests += 1;
        self.prompt_tokens += row.prompt_tokens as u64;
        self.cached_tokens += row.cached_tokens as u64;
        self.completion_tokens += row.completion_tokens as u64;
    }

    /// 0.0..=1.0 — what share of the prompt tokens never had to be evaluated.
    /// `None` when nothing was prompted, which is not the same as 0%.
    pub fn kv_reuse_ratio(&self) -> Option<f64> {
        if self.prompt_tokens == 0 {
            None
        } else {
            Some(self.cached_tokens.min(self.prompt_tokens) as f64 / self.prompt_tokens as f64)
        }
    }
}

/// The newest record seen for one hardware profile, kept so the per-profile
/// view does not have to read the file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProfileSample {
    pub ts_unix_ms: u64,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub generation_tok_s: f32,
    pub retrieved_files: u8,
}

/// One conversation session's share of the records, split by model because a
/// price is per model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionTotals {
    pub tokens: TokenTotals,
    pub by_model: BTreeMap<String, TokenTotals>,
}

/// The sidecar itself. Every field has a default so a partially written or
/// older file degrades to a rebuild rather than an error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MetricsRollup {
    pub v: u8,
    /// Byte position in `metrics.jsonl` that has been folded in.
    pub byte_offset: u64,
    /// Records folded. Grows by one per turn.
    pub rows: u64,
    pub first_ts_unix_ms: u64,
    pub last_ts_unix_ms: u64,
    pub totals: TokenTotals,
    /// Grouped by the session id the record was written with. The empty string
    /// is the group for records written before sessions were recorded.
    pub by_session: BTreeMap<String, SessionTotals>,
    /// Grouped by the model id as it was asked for. The empty string again means
    /// the record did not name one.
    pub by_model: BTreeMap<String, TokenTotals>,
    pub last_by_profile: BTreeMap<String, ProfileSample>,
    /// Oldest first, truncated to [`RATE_SAMPLE_WINDOW`] from the front. A
    /// record that reported no rate contributes nothing here — a zero measured
    /// by a server that stayed silent would drag the percentile down.
    pub generation_tok_s: Vec<f32>,
    pub prompt_tok_s: Vec<f32>,
    /// Of everything folded, how many records were asked for an answer that could
    /// be produced again — see [`RequestMetrics::repeatable`]. This is kept
    /// because a figure over runs that cannot be repeated describes one afternoon
    /// rather than a result, and the reader deserves to know which it is.
    /// Counted over [`Self::rows_generated`] only: a record that generated no
    /// tokens sampled nothing, so it can neither be repeatable nor not.
    pub rows_repeatable: u64,
    /// Records that describe a generation, i.e. the denominator for
    /// [`Self::rows_repeatable`]. Context-assembly records write no completion
    /// tokens and are excluded; they say how big a prompt was, not what came back.
    pub rows_generated: u64,
    /// The sampling of the newest repeatable record, worded for a report.
    pub last_sampling: Option<String>,
}

impl Default for MetricsRollup {
    fn default() -> Self {
        Self::empty()
    }
}

impl MetricsRollup {
    pub fn empty() -> Self {
        Self {
            v: ROLLUP_VERSION,
            byte_offset: 0,
            rows: 0,
            first_ts_unix_ms: 0,
            last_ts_unix_ms: 0,
            totals: TokenTotals::default(),
            by_session: BTreeMap::new(),
            by_model: BTreeMap::new(),
            last_by_profile: BTreeMap::new(),
            generation_tok_s: Vec::new(),
            prompt_tok_s: Vec::new(),
            rows_repeatable: 0,
            rows_generated: 0,
            last_sampling: None,
        }
    }

    /// Fold one record in. Public because a caller holding records from somewhere
    /// else can sum them exactly the way the refresh does.
    pub fn fold(&mut self, row: &RequestMetrics) {
        if row.ts_unix_ms != 0 {
            if self.rows == 0 || row.ts_unix_ms < self.first_ts_unix_ms {
                self.first_ts_unix_ms = row.ts_unix_ms;
            }
            if row.ts_unix_ms >= self.last_ts_unix_ms {
                self.last_ts_unix_ms = row.ts_unix_ms;
            }
        }
        self.totals.add_record(row);
        let model = row.model.clone().unwrap_or_default();
        let session = row.session_id.clone().unwrap_or_default();
        let group = self.by_session.entry(session).or_default();
        group.tokens.add_record(row);
        group
            .by_model
            .entry(model.clone())
            .or_default()
            .add_record(row);
        self.by_model.entry(model).or_default().add_record(row);
        self.last_by_profile.insert(
            row.profile.clone(),
            ProfileSample {
                ts_unix_ms: row.ts_unix_ms,
                prompt_tokens: row.prompt_tokens,
                cached_tokens: row.cached_tokens,
                generation_tok_s: row.generation_tok_s,
                retrieved_files: row.retrieved_files,
            },
        );
        push_sample(&mut self.generation_tok_s, row.generation_tok_s);
        push_sample(&mut self.prompt_tok_s, row.prompt_tok_s);
        // The subset holds by construction: only a record that produced tokens
        // can have produced them repeatably.
        if row.completion_tokens > 0 {
            self.rows_generated += 1;
            if row.repeatable() {
                self.rows_repeatable += 1;
                self.last_sampling = Some(sampling_words(row));
            }
        }
        self.rows += 1;
    }

    /// Share of prompt tokens served from the KV cache over everything folded.
    pub fn kv_reuse_ratio(&self) -> Option<f64> {
        self.totals.kv_reuse_ratio()
    }

    /// p50 and p95 over the samples kept, with how many samples that is. The
    /// window, not the file, is what these describe.
    pub fn generation_percentiles(&self) -> Option<Percentiles> {
        Percentiles::new(&self.generation_tok_s)
    }

    pub fn prompt_percentiles(&self) -> Option<Percentiles> {
        Percentiles::new(&self.prompt_tok_s)
    }

    /// Records in the rollup that were grouped by session, ignoring the
    /// catch-all group of records that predate the session field.
    pub fn session_count(&self) -> usize {
        self.by_session.keys().filter(|key| !key.is_empty()).count()
    }
}

fn sampling_words(row: &RequestMetrics) -> String {
    let mut parts = Vec::new();
    if let Some(temp) = row.temperature {
        parts.push(format!("temperature {temp}"));
    }
    if let Some(seed) = row.seed {
        parts.push(format!("seed {seed}"));
    }
    parts.join(" · ")
}

fn push_sample(window: &mut Vec<f32>, value: f32) {
    if !(value.is_finite() && value > 0.0) {
        return;
    }
    if window.len() == RATE_SAMPLE_WINDOW {
        window.remove(0);
    }
    window.push(value);
}

/// The 50th and 95th values of a sample set, by rank — the sample at that
/// position once sorted, with nothing interpolated between two measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Percentiles {
    pub p50: f32,
    pub p95: f32,
    pub samples: usize,
}

impl Percentiles {
    fn new(samples: &[f32]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(Self {
            p50: rank_at(&sorted, 0.50),
            p95: rank_at(&sorted, 0.95),
            samples: sorted.len(),
        })
    }
}

fn rank_at(sorted: &[f32], fraction: f64) -> f32 {
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

pub fn rollup_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join("cache").join("metrics-rollup.json")
}

/// The rollup as last written, without touching `metrics.jsonl`. `None` when it
/// is missing, unreadable, cut off, or written by another version.
pub fn read_rollup(xencode_dir: &Path) -> Option<MetricsRollup> {
    let text = std::fs::read_to_string(rollup_path(xencode_dir)).ok()?;
    let rollup: MetricsRollup = serde_json::from_str(&text).ok()?;
    (rollup.v == ROLLUP_VERSION).then_some(rollup)
}

/// How much of `metrics.jsonl` to keep once it has outgrown itself, in bytes.
///
/// The file is a record of every request the project has made, and it only ever
/// grew: a year of daily use is thousands of lines that nothing reads, because
/// the totals a person is shown come from the sidecar. This is the point where
/// the newest rows stop being worth their weight in disk.
pub const METRICS_KEEP_BYTES: u64 = 1024 * 1024;

/// How far past the kept size the file has to grow before it is trimmed back,
/// as a multiple of [`METRICS_KEEP_BYTES`]. Without the slack a file sitting at
/// the limit would be rewritten on every single turn; with it, the rewrite
/// happens once per limit's worth of requests rather than once per request.
const METRICS_TRIM_AT: u64 = 2;

/// Cut whole rows off the front of `metrics.jsonl` until it fits within
/// `keep_bytes`, and report how many bytes went.
///
/// The sidecar already holds the totals for every row ever folded in, so
/// discarding the oldest rows loses the lines rather than the figures — which is
/// why this runs from [`refresh_rollup`] and nowhere else. A trim performed
/// before the rows had been folded would delete records no one had summed.
///
/// Cutting starts at the first row boundary at or after the point `keep_bytes`
/// from the end, so what remains is complete rows. A file whose last line was
/// torn off by a process killed mid-append is left alone: there is no boundary
/// to cut at, and a row that cannot be read is not worth rewriting the file for.
pub fn trim_metrics(xencode_dir: &Path, keep_bytes: u64) -> std::io::Result<u64> {
    use std::io::{Read, Seek, SeekFrom, Write};

    let path = crate::metrics::metrics_path(xencode_dir);
    let mut file = match std::fs::File::open(&path) {
        Ok(file) => file,
        Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(failure) => return Err(failure),
    };
    let length = file.metadata()?.len();
    if length <= keep_bytes.saturating_mul(METRICS_TRIM_AT) {
        return Ok(0);
    }

    // One read of the window that is worth keeping, so the bytes written back
    // are the bytes that were measured rather than a second look at a file that
    // may have grown in between.
    let window_start = length - keep_bytes;
    file.seek(SeekFrom::Start(window_start))?;
    let mut tail = Vec::with_capacity(keep_bytes as usize);
    file.read_to_end(&mut tail)?;
    let Some(newline) = tail.iter().position(|byte| *byte == b'\n') else {
        // No row boundary in the whole window: the file is one enormous row, or
        // shorter than a line. Leave it as it is.
        return Ok(0);
    };
    // Everything up to and including that newline is older than the window.
    let cut = window_start + newline as u64 + 1;
    let mut kept = tail[newline + 1..].to_vec();
    // A row torn by a kill stays at the end of the file, where it already was.
    if !kept.last().is_some_and(|byte| *byte == b'\n') {
        kept.push(b'\n');
    }

    let temp = path.with_extension("jsonl.trimming");
    {
        let mut opened = std::fs::File::create(&temp)?;
        opened.write_all(&kept)?;
        opened.sync_all()?;
    }
    // Bail rather than replace if a second writer appended in the meantime: the
    // rows it added are not in `kept`, and skipping one trim costs nothing.
    if std::fs::metadata(&path)?.len() != length {
        let _ = std::fs::remove_file(&temp);
        return Ok(0);
    }
    std::fs::rename(&temp, &path)?;
    Ok(cut)
}

/// Fold whatever was appended since the last refresh, save the sidecar, and
/// return it. This is the only way to get a current rollup.
///
/// A metrics file that was replaced rather than appended to restarts the
/// rollup from nothing, because the records it holds now are not the ones the
/// totals were built from. A file that has gone missing leaves the last rollup
/// as it was: nothing was seen to change, and the totals still describe the
/// records that were there.
pub fn refresh_rollup(xencode_dir: &Path) -> std::io::Result<MetricsRollup> {
    let mut rollup = read_rollup(xencode_dir).unwrap_or_default();
    let (rows, offset) = read_metrics_since(xencode_dir, rollup.byte_offset);
    // Nothing complete to fold: the position is still current, or the file has
    // gone away. Either way the rollup stands, and the sidecar is not rewritten.
    if rows.is_empty() {
        return Ok(rollup);
    }
    if offset < rollup.byte_offset {
        rollup = MetricsRollup::empty();
    }
    for row in &rows {
        rollup.fold(row);
    }
    rollup.byte_offset = offset;
    rollup.v = ROLLUP_VERSION;
    // Rows older than the kept window can go now that every one of them has been
    // folded: the cut removes lines, not figures, because their totals are in the
    // sidecar. Carrying the position back by the same amount is what keeps the
    // offset pointing inside the file that is left, so this happens before the
    // single write below rather than after it.
    let cut = trim_metrics(xencode_dir, METRICS_KEEP_BYTES)?;
    rollup.byte_offset = rollup.byte_offset.saturating_sub(cut);
    write_rollup(xencode_dir, &rollup)?;
    Ok(rollup)
}

/// Write the sidecar so a reader either sees the whole previous file or the
/// whole new one, never a mixture.
pub fn write_rollup(xencode_dir: &Path, rollup: &MetricsRollup) -> std::io::Result<()> {
    let path = rollup_path(xencode_dir);
    let json = serde_json::to_string_pretty(rollup)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    xencode_core_rs::write_atomic(&path, json.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{MetricSource, MetricsIdentity};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn temp_dir() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-rollup-test-{unique}"))
    }

    /// Records built the way the writers build them, then appended, so every
    /// assertion below runs over the bytes in `metrics.jsonl` rather than over
    /// a struct handed straight to the folder.
    fn append(dir: &Path, rows: &[RequestMetrics]) {
        for row in rows {
            crate::metrics::append_metrics(dir, row).unwrap();
        }
    }

    fn record(profile: &str, prompt: u32, evaluated: u32, completion: u32) -> RequestMetrics {
        RequestMetrics::from_timings(profile, 8192, prompt, evaluated, completion, 0.0, 0.0, 3)
    }

    #[test]
    fn the_rollup_sums_the_records_that_are_actually_on_disk() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append(
            &xencode,
            &[
                record("LOW", 1000, 400, 50),
                record("BALANCED", 2000, 2000, 80),
            ],
        );

        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.rows, 2);
        // Hand-summed: 1000 + 2000 prompted, of which 600 + 0 were cached.
        assert_eq!(rollup.totals.prompt_tokens, 3000);
        assert_eq!(rollup.totals.cached_tokens, 600);
        assert_eq!(rollup.totals.completion_tokens, 130);
        assert!((rollup.kv_reuse_ratio().unwrap() - 0.2).abs() < 1e-9);
        fs_reset(&dir);
    }

    #[test]
    fn a_refresh_after_more_records_adds_only_the_new_ones() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append(&xencode, &[record("LOW", 1000, 400, 50)]);
        let first = refresh_rollup(&xencode).unwrap();
        assert_eq!(first.rows, 1);

        append(&xencode, &[record("LOW", 500, 500, 10)]);
        let second = refresh_rollup(&xencode).unwrap();
        assert_eq!(second.rows, 2);
        assert_eq!(second.totals.prompt_tokens, 1500);
        // 600 from the first record, none from the second: it re-evaluated
        // every prompt token.
        assert_eq!(second.totals.cached_tokens, 600);
        assert!(second.byte_offset > first.byte_offset);
        fs_reset(&dir);
    }

    #[test]
    fn refreshing_with_nothing_new_leaves_the_sidecar_alone() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append(&xencode, &[record("LOW", 1000, 400, 50)]);
        refresh_rollup(&xencode).unwrap();
        let written = std::fs::metadata(rollup_path(&xencode))
            .unwrap()
            .modified()
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let again = refresh_rollup(&xencode).unwrap();
        assert_eq!(again.rows, 1);
        assert_eq!(
            written,
            std::fs::metadata(rollup_path(&xencode))
                .unwrap()
                .modified()
                .unwrap(),
            "a refresh with nothing to fold rewrote the sidecar"
        );
        fs_reset(&dir);
    }

    #[test]
    fn tokens_group_by_the_session_and_model_the_record_names() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let mut a = record("LOW", 1000, 400, 50);
        MetricsIdentity {
            session_id: Some("session_one".to_string()),
            model: Some("qwen2.5:7b".to_string()),
            provider: Some("ollama".to_string()),
            source: Some(MetricSource::Local),
        }
        .apply(&mut a);
        let mut b = record("LOW", 300, 300, 20);
        MetricsIdentity {
            session_id: Some("session_two".to_string()),
            model: Some("qwen2.5:7b".to_string()),
            provider: Some("ollama".to_string()),
            source: Some(MetricSource::Local),
        }
        .apply(&mut b);
        append(&xencode, &[a, b]);

        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.session_count(), 2);
        assert_eq!(rollup.by_session["session_one"].tokens.prompt_tokens, 1000);
        assert_eq!(rollup.by_session["session_two"].tokens.prompt_tokens, 300);
        assert_eq!(rollup.by_model["qwen2.5:7b"].requests, 2);
        assert_eq!(rollup.by_model["qwen2.5:7b"].completion_tokens, 70);
        // A session's tokens are split by model too, which is what lets one
        // session's spend be priced.
        assert_eq!(
            rollup.by_session["session_one"].by_model["qwen2.5:7b"].requests,
            1
        );
        fs_reset(&dir);
    }

    #[test]
    fn a_replaced_metrics_file_rebuilds_the_totals_instead_of_adding_to_them() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append(
            &xencode,
            &[record("LOW", 1000, 0, 0), record("LOW", 1000, 0, 0)],
        );
        refresh_rollup(&xencode).unwrap();

        // Log rotation: a shorter file under the same name.
        std::fs::remove_file(crate::metrics::metrics_path(&xencode)).unwrap();
        append(&xencode, &[record("LOW", 250, 0, 0)]);
        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.rows, 1);
        assert_eq!(rollup.totals.prompt_tokens, 250);
        fs_reset(&dir);
    }

    #[test]
    fn a_missing_metrics_file_leaves_the_last_rollup_in_place() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append(&xencode, &[record("LOW", 1000, 400, 50)]);
        refresh_rollup(&xencode).unwrap();
        std::fs::remove_file(crate::metrics::metrics_path(&xencode)).unwrap();

        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.rows, 1);
        assert_eq!(rollup.totals.prompt_tokens, 1000);
        fs_reset(&dir);
    }

    #[test]
    fn percentiles_cover_the_samples_kept_and_say_so() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        // Ten records at ten different generation rates, one token/s apart.
        let rows: Vec<RequestMetrics> = (1..=10u32)
            .map(|i| RequestMetrics::from_timings("LOW", 4096, 100, 100, 10, i as f32, 0.0, 1))
            .collect();
        append(&xencode, &rows);
        let rollup = refresh_rollup(&xencode).unwrap();
        let speed = rollup.generation_percentiles().unwrap();
        assert_eq!(speed.samples, 10);
        // Sorted 1.0 ..= 10.0: p50 is position round(0.5 × 9) = 5, p95 position
        // round(0.95 × 9) = 9.
        assert_eq!(speed.p50, 6.0);
        assert_eq!(speed.p95, 10.0);
        // A server that reported no rate is not counted as a zero.
        assert!(rollup.prompt_tok_s.is_empty());
        fs_reset(&dir);
    }

    #[test]
    fn the_rate_window_is_bounded_however_many_records_accumulate() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let extra = 200u32;
        let rows: Vec<RequestMetrics> = (1..=(RATE_SAMPLE_WINDOW as u32 + extra))
            .map(|i| RequestMetrics::from_timings("LOW", 4096, 10, 10, 1, i as f32, 0.0, 1))
            .collect();
        append(&xencode, &rows);
        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.generation_tok_s.len(), RATE_SAMPLE_WINDOW);
        assert_eq!(rollup.rows, 712);
        let speed = rollup.generation_percentiles().unwrap();
        assert_eq!(speed.samples, RATE_SAMPLE_WINDOW);
        // The 200 oldest rates were dropped, so the window covers 201.0 ..=
        // 712.0 and its median is the 257th of those.
        assert_eq!(speed.p50, 457.0);
        fs_reset(&dir);
    }

    #[test]
    fn the_newest_record_per_profile_survives_without_reading_the_file() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let mut old = record("BALANCED", 1000, 900, 10);
        old.ts_unix_ms = 100;
        let mut newer = record("BALANCED", 2000, 0, 20);
        newer.ts_unix_ms = 200;
        let low = record("LOW", 500, 500, 5);
        append(&xencode, &[old, newer, low]);

        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.last_by_profile["BALANCED"].prompt_tokens, 2000);
        assert_eq!(rollup.last_by_profile["LOW"].prompt_tokens, 500);
        fs_reset(&dir);
    }

    #[test]
    fn a_sidecar_written_by_another_version_is_rebuilt_from_the_records() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append(&xencode, &[record("LOW", 1000, 400, 50)]);
        std::fs::create_dir_all(xencode.join("cache")).unwrap();
        std::fs::write(
            rollup_path(&xencode),
            r#"{"v":99,"byte_offset":99999,"rows":4242}"#,
        )
        .unwrap();

        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.rows, 1);
        assert_eq!(rollup.v, ROLLUP_VERSION);
        fs_reset(&dir);
    }

    #[test]
    fn a_half_written_sidecar_reads_as_no_rollup_and_is_rewritten() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append(&xencode, &[record("LOW", 1000, 400, 50)]);
        std::fs::create_dir_all(xencode.join("cache")).unwrap();
        std::fs::write(rollup_path(&xencode), b"{\"v\":1,\"rows\":12").unwrap();

        assert!(read_rollup(&xencode).is_none());
        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.rows, 1);
        assert_eq!(read_rollup(&xencode).unwrap().rows, 1);
        fs_reset(&dir);
    }

    #[test]
    fn a_sessions_recorded_tokens_price_out_to_what_a_hand_sum_gives() {
        use crate::pricing::{cost_of, pricing_path, PriceTable};
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        std::fs::create_dir_all(&xencode).unwrap();
        std::fs::write(
            pricing_path(&xencode),
            r#"{"models":{"qwen2.5:7b":{"input_usd_per_mtok":0.3,"output_usd_per_mtok":0.6,"cached_input_usd_per_mtok":0.06}}}"#,
        )
        .unwrap();

        // Three turns in two sessions, written the way the writers write them.
        let named = |session: &str, prompt: u32, evaluated: u32, completion: u32| {
            let mut m = RequestMetrics::from_timings(
                "BALANCED", 8192, prompt, evaluated, completion, 12.5, 300.0, 4,
            );
            m.session_id = Some(session.to_string());
            m.model = Some("qwen2.5:7b".to_string());
            m
        };
        append(
            &xencode,
            &[
                named("session_a", 1000, 400, 200),
                named("session_a", 500, 500, 100),
                named("session_b", 200, 200, 50),
            ],
        );

        let rollup = refresh_rollup(&xencode).unwrap();
        // Hand-summed off the three records above.
        assert_eq!(rollup.totals.prompt_tokens, 1700);
        assert_eq!(rollup.totals.cached_tokens, 600);
        assert_eq!(rollup.totals.completion_tokens, 350);

        let table = PriceTable::load(&xencode);
        let a = cost_of(&rollup.by_session["session_a"].by_model, &table);
        // 900 fresh input × $0.30 + 600 cached × $0.06 + 300 generated × $0.60,
        // all per million tokens, so the millionths cancel: 270 + 36 + 180.
        assert_eq!(a.known_micros, 486);
        assert!(a.complete());
        let b = cost_of(&rollup.by_session["session_b"].by_model, &table);
        assert_eq!(b.known_micros, 90);
        // Both sessions together, priced through the same table.
        let all = cost_of(&rollup.by_model, &table);
        assert_eq!(all.known_micros, 576);
        fs_reset(&dir);
    }

    /// The counts say how much of the record can be produced again, and only the
    /// rows that asked for it contribute. A row written with no seed and no
    /// temperature of zero ran on sampling the server chose, which is the same
    /// answer a row from before these fields existed has to give — an assumed
    /// "repeatable" there would overstate what the run can prove.
    #[test]
    fn repeatability_counts_only_the_records_that_asked_for_it() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let mut rows = Vec::new();
        // Nothing sent at all: the server drew its own seed.
        rows.push({
            let mut m = record("LOW", 100, 100, 10);
            m.ts_unix_ms = 1;
            m
        });
        // A temperature on its own still leaves the draws to chance.
        rows.push({
            let mut m = record("LOW", 200, 200, 20);
            m.ts_unix_ms = 2;
            m.temperature = Some(0.7);
            m
        });
        // A seed pins the sampler whatever the temperature is.
        rows.push({
            let mut m = record("LOW", 300, 300, 30);
            m.ts_unix_ms = 3;
            m.temperature = Some(1.0);
            m.seed = Some(5);
            m
        });
        // Greedy decoding needs no seed to give the same answer twice.
        rows.push({
            let mut m = record("LOW", 400, 400, 40);
            m.ts_unix_ms = 4;
            m.temperature = Some(0.0);
            m
        });
        // And a later unrepeatably-sampled turn must not overwrite the wording
        // for the last one that was pinned.
        rows.push({
            let mut m = record("LOW", 500, 500, 50);
            m.ts_unix_ms = 5;
            m
        });
        // A context-assembly record: no tokens came back, so there was no
        // sampling to judge and it belongs in neither side of the count.
        rows.push({
            let mut m = record("LOW", 600, 600, 0);
            m.ts_unix_ms = 6;
            m
        });
        append(&xencode, &rows);

        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.rows, 6);
        assert_eq!(rollup.rows_generated, 5);
        assert_eq!(rollup.rows_repeatable, 2);
        assert_eq!(rollup.last_sampling.as_deref(), Some("temperature 0"));
        fs_reset(&dir);
    }

    /// A version 1 sidecar is a real file from before the repeatability counts
    /// existed: the same shape, minus those two fields. Because every field has
    /// a default, it would read back happily and report that nothing was ever
    /// repeatable — a wrong answer presented as a measurement. The version check
    /// is what turns that file into a rebuild.
    #[test]
    fn a_version_1_sidecar_is_rebuilt_because_it_never_answered_the_question() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append(
            &xencode,
            &[{
                let mut m = record("LOW", 1000, 400, 50);
                m.temperature = Some(0.2);
                m.seed = Some(7);
                m
            }],
        );
        let current = refresh_rollup(&xencode).unwrap();
        assert_eq!(current.rows_repeatable, 1);
        std::fs::write(
            rollup_path(&xencode),
            format!(
                r#"{{"v":1,"byte_offset":{},"rows":{}}}"#,
                current.byte_offset, current.rows
            ),
        )
        .unwrap();

        assert!(read_rollup(&xencode).is_none());
        let rollup = refresh_rollup(&xencode).unwrap();
        assert_eq!(rollup.rows, 1);
        assert_eq!(rollup.rows_generated, 1);
        assert_eq!(rollup.rows_repeatable, 1);
        assert_eq!(
            rollup.last_sampling.as_deref(),
            Some("temperature 0.2 · seed 7")
        );
        fs_reset(&dir);
    }

    /// The same bytes `append_metrics` would have written, in one write, so a
    /// test can build a file the size of a year of turns without a write per row.
    fn write_rows(xencode_dir: &Path, count: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        let line = serde_json::to_string(&record("LOW", 1000, 400, 50)).unwrap();
        for _ in 0..count {
            bytes.extend_from_slice(line.as_bytes());
            bytes.push(b'\n');
        }
        std::fs::create_dir_all(xencode_dir.join("cache")).unwrap();
        std::fs::write(crate::metrics::metrics_path(xencode_dir), &bytes).unwrap();
        bytes
    }

    /// A file that has outgrown the window is cut back on the refresh that
    /// catches up with it, and the figures still describe every turn — including
    /// the rows that are no longer on disk.
    #[test]
    fn a_refresh_cuts_an_oversized_metrics_file_without_losing_its_totals() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        // Comfortably past the point where a trim is owed, at a row size the
        // writer actually produces.
        let rows = 12_000;
        let written = write_rows(&xencode, rows);
        let before = crate::metrics::metrics_path(&xencode)
            .metadata()
            .unwrap()
            .len();
        assert!(
            before > METRICS_KEEP_BYTES * 2,
            "the test file is only {before} bytes, which never triggers a trim"
        );

        let rollup = refresh_rollup(&xencode).unwrap();
        let after = crate::metrics::metrics_path(&xencode)
            .metadata()
            .unwrap()
            .len();

        assert!(
            after <= METRICS_KEEP_BYTES,
            "{before} bytes came down to {after}, still over the window"
        );
        // Every record was folded before any was cut, so nothing was lost by the
        // trim: 12 000 rows at 1 000 prompted and 50 completed tokens each.
        assert_eq!(rollup.rows as usize, rows);
        assert_eq!(rollup.totals.prompt_tokens, 1000 * rows as u64);
        assert_eq!(rollup.totals.completion_tokens, 50 * rows as u64);
        assert_eq!(rollup.byte_offset, after, "the position is past the file");
        // What survives is the newest part of the file, byte for byte.
        assert_eq!(
            std::fs::read(crate::metrics::metrics_path(&xencode)).unwrap(),
            written[written.len() - after as usize..]
        );
        fs_reset(&dir);
    }

    /// The trim is not the end of the story: the next turn appends to a smaller
    /// file and the rollup keeps counting from where it was.
    #[test]
    fn the_rollup_keeps_counting_after_the_file_was_cut() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        write_rows(&xencode, 12_000);
        let trimmed = refresh_rollup(&xencode).unwrap();
        let prompted_before = trimmed.totals.prompt_tokens;

        append(&xencode, &[record("LOW", 1000, 400, 50)]);
        let after = refresh_rollup(&xencode).unwrap();

        assert_eq!(after.rows, trimmed.rows + 1);
        assert_eq!(
            after.totals.prompt_tokens,
            prompted_before + 1000,
            "a row appended after the trim was not counted"
        );
        assert!(after.byte_offset > trimmed.byte_offset);
        assert!(after.byte_offset < METRICS_KEEP_BYTES + 512);
        fs_reset(&dir);
    }

    #[test]
    fn a_metrics_file_that_still_fits_is_left_untouched() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        append(&xencode, &[record("LOW", 1000, 400, 50)]);
        let before = std::fs::read(crate::metrics::metrics_path(&xencode)).unwrap();

        let cut = trim_metrics(&xencode, 1024).unwrap();

        assert_eq!(cut, 0);
        assert_eq!(
            std::fs::read(crate::metrics::metrics_path(&xencode)).unwrap(),
            before,
            "a file under the window was rewritten"
        );
        fs_reset(&dir);
    }

    /// The cut lands on a row boundary, and a row torn off by a process killed
    /// mid-append stays where it was — the reader already knows to drop it.
    #[test]
    fn a_trim_cuts_at_a_row_boundary_and_keeps_a_torn_row() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        std::fs::create_dir_all(xencode.join("cache")).unwrap();
        let mut bytes = Vec::new();
        for index in 0..200 {
            bytes.extend(
                format!(
                    "{{\"profile\":\"LOW\",\"prompt_tokens\":{},\"completion_tokens\":7}}\n",
                    index
                )
                .into_bytes(),
            );
        }
        // The last row is the one a kill left half-written.
        bytes.extend(b"{\"profile\":\"LO");
        std::fs::write(crate::metrics::metrics_path(&xencode), &bytes).unwrap();

        let cut = trim_metrics(&xencode, 1000).unwrap();
        let kept = std::fs::read(crate::metrics::metrics_path(&xencode)).unwrap();

        assert!(cut > 0);
        assert!(
            kept.len() <= 1000,
            "{} bytes kept of a 1000 cap",
            kept.len()
        );
        // Whole rows only, ending with the torn one and a newline.
        assert!(kept.starts_with(b"{"), "cut into the middle of a row");
        assert!(
            String::from_utf8_lossy(&kept)
                .lines()
                .all(|line| line.starts_with("{\"profile\"")),
            "a fragment of a row survived the cut"
        );
        assert!(kept.last().is_some_and(|byte| *byte == b'\n'));
        fs_reset(&dir);
    }

    /// A file with no row boundary inside the window is one row too big to cut
    /// anywhere, so it is left alone rather than halved at an arbitrary byte.
    #[test]
    fn a_metrics_file_that_is_one_row_is_not_cut() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        std::fs::create_dir_all(xencode.join("cache")).unwrap();
        let one_row = vec![b'x'; 4096];
        std::fs::write(crate::metrics::metrics_path(&xencode), &one_row).unwrap();

        assert_eq!(trim_metrics(&xencode, 1000).unwrap(), 0);
        assert_eq!(
            std::fs::read(crate::metrics::metrics_path(&xencode)).unwrap(),
            one_row
        );
        fs_reset(&dir);
    }

    #[test]
    fn trimming_a_file_that_is_not_there_is_not_a_failure() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        assert_eq!(trim_metrics(&xencode, 1024).unwrap(), 0);
        fs_reset(&dir);
    }

    fn fs_reset(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }
}
