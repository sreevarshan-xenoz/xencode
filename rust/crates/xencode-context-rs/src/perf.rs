//! QO-4 — deciding whether the hot paths got slower, and refusing to decide
//! when the numbers cannot support it.
//!
//! A benchmark that always prints a number always prints a verdict too, and most
//! of those verdicts are noise. This machine's measurement floor is about 1.6%
//! coefficient of variation near idle and 5–6% under load (fact Q-1.18), so a
//! 2% "regression" reported from ten samples means nothing at all. The harness
//! therefore has three jobs, in order: measure the real hot paths with
//! `cargo bench`, compare each one against a stored baseline with a test that
//! says whether the two sets of samples could come from the same distribution,
//! and *decline* when the spread in the data is wider than the claim being made.
//!
//! What it compares against is a snapshot of the raw per-iteration samples under
//! `.xencode/perf/baseline.json`, not the summary figures criterion prints.
//! Criterion's numbers are derived values; keeping the samples means a verdict
//! can be re-derived later with a different threshold, and it means the baseline
//! records how many files the corpus had. A bench run over a different tree is a
//! different measurement, and comparing one against this one is the single
//! mistake that would make every later number meaningless while still looking
//! like a pass.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Where the recorded baseline lives, relative to the repository root.
pub const PERF_STATE_DIR: &str = ".xencode/perf";

/// The baseline's file name inside [`PERF_STATE_DIR`].
pub const BASELINE_FILE: &str = "baseline.json";

/// Above this coefficient of variation the spread in the samples is comparable
/// to the smallest change worth reporting, so no verdict is given. Fact Q-1.18
/// measured about 1.6% here near idle; anything at or over 5% is a loaded
/// machine, and a verdict from it would be a description of the load.
pub const MAX_CV: f64 = 0.05;

/// The smallest slowdown that raises a flag, as a percentage of the baseline.
///
/// Below the harness's own acceptance number: the done-when is an injected 10%
/// slowdown firing, and a threshold sitting at 10% would only just catch the one
/// case it is tested against.
pub const DEFAULT_ALERT_PCT: f64 = 5.0;

/// The significance level the Mann-Whitney test is judged at.
pub const DEFAULT_ALPHA: f64 = 0.05;

/// Large enough to enumerate exactly and small enough to finish in well under a
/// second — ten against ten is 184,756 splits. Past this the tie-corrected
/// normal approximation is used, and the result says which one produced it.
const EXACT_PERMUTATION_LIMIT: u128 = 1_000_000;

/// The crate and bench target this harness measures.
pub const BENCH_PACKAGE: &str = "xencode-context-rs";
pub const BENCH_TARGET: &str = "hot_paths";

/// The line the bench prints to declare what it measured against. A baseline
/// that disagrees with it is not a baseline for this run.
pub const CORPUS_MARKER: &str = "xencode-perf-corpus:";

/// One measured path: the nanoseconds each iteration took, kept raw.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Samples {
    /// Criterion's full bench id, such as `retrieve/hybrid_top10`.
    pub id: String,
    /// Per-iteration time in nanoseconds, in measurement order.
    pub ns_per_iteration: Vec<f64>,
}

impl Samples {
    pub fn n(&self) -> usize {
        self.ns_per_iteration.len()
    }

    pub fn mean(&self) -> f64 {
        if self.n() == 0 {
            return 0.0;
        }
        self.ns_per_iteration.iter().sum::<f64>() / self.n() as f64
    }

    /// The middle of the sorted samples — the level the delta is taken at,
    /// because two outliers in ten measurements move a mean but not a median.
    pub fn median(&self) -> f64 {
        let mut sorted = self.ns_per_iteration.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        if n == 0 {
            return 0.0;
        }
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    }

    /// Sample standard deviation (divided by n − 1). `None` below two samples,
    /// where spread is not a measurable quantity rather than a small one.
    pub fn stdev(&self) -> Option<f64> {
        if self.n() < 2 {
            return None;
        }
        let mean = self.mean();
        let var = self
            .ns_per_iteration
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / (self.n() - 1) as f64;
        Some(var.sqrt())
    }

    /// Standard deviation over the mean: the fraction of the level that is
    /// spread. That is the quantity [`MAX_CV`] is judged on.
    pub fn cv(&self) -> Option<f64> {
        let mean = self.mean();
        if mean <= 0.0 {
            return None;
        }
        self.stdev().map(|s| s / mean)
    }
}

/// Bump when the stored shape changes meaning.
pub const BASELINE_VERSION: u32 = 1;

/// A recorded set of samples, with the tree it was measured against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    /// Format version, so a stale file is named rather than misread.
    pub version: u32,
    /// Rust files in the corpus when the bench ran: the identity of the tree.
    pub corpus_files: usize,
    /// Where it was recorded. Not compared — two checkouts of the same tree on
    /// one machine measure the same thing, and the file count catches the rest.
    pub recorded_in: String,
    /// Seconds since the epoch.
    pub recorded_unix: u64,
    /// id → samples.
    pub benches: BTreeMap<String, Samples>,
}

impl Baseline {
    pub fn path_for(root: &Path) -> PathBuf {
        root.join(PERF_STATE_DIR).join(BASELINE_FILE)
    }

    /// The stored baseline, or `None` when nothing has been recorded yet.
    ///
    /// A file that cannot be read is an error rather than an absence: silently
    /// treating a corrupt baseline as "no baseline" would let a regression pass
    /// as a first run.
    pub fn load(root: &Path) -> Result<Option<Baseline>, String> {
        let path = Self::path_for(root);
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("could not read {}: {e}", path.display()))?;
        let baseline: Baseline = serde_json::from_str(&text)
            .map_err(|e| format!("{} is not a readable baseline: {e}", path.display()))?;
        if baseline.version != BASELINE_VERSION {
            return Err(format!(
                "{} was written as format version {}, this build reads version {}",
                path.display(),
                baseline.version,
                BASELINE_VERSION
            ));
        }
        Ok(Some(baseline))
    }

    pub fn write(&self, root: &Path) -> Result<PathBuf, String> {
        let path = Self::path_for(root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| format!("could not encode the baseline: {e}"))?;
        // Written under a temporary name in the same directory and renamed, so an
        // interrupted write cannot leave a half-file that replaces the baseline
        // that was already there.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text.as_bytes())
            .map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("could not move {} into place: {e}", tmp.display()))?;
        Ok(path)
    }
}

/// The result of one measurement run: the samples plus the tree they came from.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub benches: BTreeMap<String, Samples>,
    pub corpus_files: usize,
    pub command: String,
    pub took: Duration,
}

/// What a comparison concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Slower by at least the alert threshold, and significantly so.
    Regressed,
    /// Faster by at least the alert threshold, and significantly so.
    Improved,
    /// Nothing beyond the threshold, or not significant.
    NoChange,
    /// Measured, but the numbers do not support a claim.
    Refused,
    /// In the baseline, not measured this time.
    NotMeasured,
    /// Measured this time, absent from the baseline.
    NoBaseline,
}

impl Outcome {
    pub fn label(self) -> &'static str {
        match self {
            Outcome::Regressed => "REGRESSION",
            Outcome::Improved => "faster",
            Outcome::NoChange => "no change",
            Outcome::Refused => "NO VERDICT",
            Outcome::NotMeasured => "not measured",
            Outcome::NoBaseline => "no baseline",
        }
    }
}

/// One bench's verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub id: String,
    pub baseline_median_ns: Option<f64>,
    pub current_median_ns: Option<f64>,
    /// `(current − baseline) / baseline × 100`, positive meaning slower.
    pub delta_pct: Option<f64>,
    /// Two-sided Mann-Whitney p-value.
    pub p_value: Option<f64>,
    /// Whether that p-value came from the exact enumeration or the approximation.
    pub p_value_method: Option<&'static str>,
    /// Spread of the current samples as a fraction of their level.
    pub cv: Option<f64>,
    pub outcome: Outcome,
    pub reason: Option<String>,
}

/// The whole run's report.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub corpus_files: usize,
    pub baseline_corpus_files: Option<usize>,
    /// The run-wide refusal: samples taken over a different corpus than the
    /// baseline, which invalidates every comparison at once.
    pub corpus_mismatch: bool,
    pub comparisons: Vec<Comparison>,
    pub took: Duration,
    pub command: String,
    pub notes: Vec<String>,
}

impl Report {
    pub fn has_regression(&self) -> bool {
        self.comparisons
            .iter()
            .any(|c| c.outcome == Outcome::Regressed)
    }

    pub fn regressions(&self) -> Vec<&Comparison> {
        self.comparisons
            .iter()
            .filter(|c| c.outcome == Outcome::Regressed)
            .collect()
    }

    pub fn refusals(&self) -> Vec<&Comparison> {
        self.comparisons
            .iter()
            .filter(|c| c.outcome == Outcome::Refused)
            .collect()
    }
}

/// Whether the bench target exists in this workspace. The harness measures a
/// real `[[bench]]` target or says it is not there; it never invents a number.
pub fn bench_target_present(manifest_root: &Path) -> bool {
    manifest_root
        .join("crates")
        .join(BENCH_PACKAGE)
        .join("benches")
        .join(format!("{BENCH_TARGET}.rs"))
        .is_file()
}

/// Run the benches and read criterion's per-iteration samples back.
///
/// `filter` is passed to criterion as a substring filter, so one path can be
/// re-measured without paying for all seven.
pub fn measure(manifest_root: &Path, filter: Option<&str>) -> Result<Measured, String> {
    let started = Instant::now();
    if !bench_target_present(manifest_root) {
        return Err(format!(
            "no bench target at crates/{BENCH_PACKAGE}/benches/{BENCH_TARGET}.rs under {}",
            manifest_root.display()
        ));
    }

    let mut command = std::process::Command::new("cargo");
    command
        .arg("bench")
        .arg("-p")
        .arg(BENCH_PACKAGE)
        .arg("--bench")
        .arg(BENCH_TARGET)
        .current_dir(manifest_root);
    if let Some(filter) = filter {
        command.arg("--").arg(filter);
    }
    let rendered = format!(
        "cargo bench -p {BENCH_PACKAGE} --bench {BENCH_TARGET}{}",
        filter.map(|f| format!(" -- {f}")).unwrap_or_default()
    );

    // Drop the previous run's samples first. Criterion leaves a filtered-out
    // group's files exactly where they were, so without this a path that was not
    // measured by this run would be compared — and agree to the last nanosecond —
    // with a sample file from some earlier one.
    clear_previous_samples(manifest_root)?;

    let output = command
        .output()
        .map_err(|e| format!("could not start cargo bench: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "benchmark run failed ({}): {}",
            output.status.code().unwrap_or(-1),
            stderr.lines().rev().take(6).collect::<Vec<_>>().join(" / ")
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let corpus_files = parse_corpus_files(&stdout).ok_or_else(|| {
        format!(
            "the bench did not declare its corpus: expected a line starting with \
             \"{CORPUS_MARKER}\" in its output"
        )
    })?;

    let benches = read_samples(manifest_root)?;
    if benches.is_empty() {
        return Err(format!(
            "cargo bench succeeded but measured nothing{}, so there are no samples under {}",
            filter
                .map(|f| format!(" for the filter \"{f}\""))
                .unwrap_or_default(),
            manifest_root.join("target/criterion").display()
        ));
    }

    Ok(Measured {
        benches,
        corpus_files,
        command: rendered,
        took: started.elapsed(),
    })
}

/// The file count read off the bench's own declaration line.
pub fn parse_corpus_files(bench_stdout: &str) -> Option<usize> {
    for line in bench_stdout.lines() {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix(CORPUS_MARKER) else {
            continue;
        };
        for field in rest.split_whitespace() {
            if let Some(value) = field.strip_prefix("files=") {
                if let Ok(n) = value.parse::<usize>() {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Delete the `new/sample.json` of every criterion group under this workspace.
///
/// Only the sample file, and only under `target/`, which is generated state:
/// criterion rebuilds what it needs for the groups it runs, and a group a filter
/// excludes is left with nothing to read. Returns how many were cleared.
pub fn clear_previous_samples(manifest_root: &Path) -> Result<usize, String> {
    let criterion = manifest_root.join("target").join("criterion");
    if !criterion.is_dir() {
        return Ok(0);
    }
    let mut cleared = 0;
    let entries = std::fs::read_dir(&criterion)
        .map_err(|e| format!("could not read {}: {e}", criterion.display()))?;
    for entry in entries.filter_map(|e| e.ok()) {
        let sample = entry.path().join("new").join("sample.json");
        if sample.is_file() {
            std::fs::remove_file(&sample)
                .map_err(|e| format!("could not clear {}: {e}", sample.display()))?;
            cleared += 1;
        }
    }
    Ok(cleared)
}

/// Collect `id → per-iteration samples` from every criterion group that has a
/// finished run under `new/`.
pub fn read_samples(manifest_root: &Path) -> Result<BTreeMap<String, Samples>, String> {
    let criterion = manifest_root.join("target").join("criterion");
    let mut out = BTreeMap::new();
    if !criterion.is_dir() {
        return Ok(out);
    }
    let entries = std::fs::read_dir(&criterion)
        .map_err(|e| format!("could not read {}: {e}", criterion.display()))?;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let new = path.join("new");
        let (Some(id), Some(ns)) = (read_bench_id(&new), read_sample_times(&new)) else {
            continue;
        };
        out.insert(
            id.clone(),
            Samples {
                id,
                ns_per_iteration: ns,
            },
        );
    }
    Ok(out)
}

fn read_bench_id(new_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(new_dir.join("benchmark.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value
        .get("full_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// `sample.json` holds total elapsed nanoseconds per measurement point, each run
/// for a number of iterations. Divided, that is the per-iteration time the test
/// needs; undivided, a bench that happened to iterate more looks slower.
fn read_sample_times(new_dir: &Path) -> Option<Vec<f64>> {
    let text = std::fs::read_to_string(new_dir.join("sample.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let iters = value.get("iters")?.as_array()?;
    let times = value.get("times")?.as_array()?;
    if iters.len() != times.len() || iters.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(iters.len());
    for (i, t) in iters.iter().zip(times.iter()) {
        let iterations = i.as_f64()?;
        let total_ns = t.as_f64()?;
        if iterations <= 0.0 {
            return None;
        }
        out.push(total_ns / iterations);
    }
    Some(out)
}

/// Mid-ranks over the combined sample, ties averaged — so a tie cannot quietly
/// favour whichever side happened to measure it first.
fn mid_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && values[order[j + 1]] == values[order[i]] {
            j += 1;
        }
        let average = (i + j + 2) as f64 / 2.0;
        for k in i..=j {
            ranks[order[k]] = average;
        }
        i = j + 1;
    }
    ranks
}

/// `n` choose `k`, saturating at the widest the caller compares against.
///
/// The running quotient is exact at every step — it is `C(n, i + 1)` — so no
/// rounding is involved; the only overflow path returns a value that is by
/// construction past the limit, which is the answer the caller needs.
fn binom(n: usize, k: usize) -> u128 {
    let k = k.min(n.saturating_sub(k));
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = match acc.checked_mul((n - i) as u128) {
            Some(product) => product / (i as u128 + 1),
            None => return EXACT_PERMUTATION_LIMIT + 1,
        };
    }
    acc
}

/// Mann-Whitney U of `current` against `baseline`, with a two-sided p-value.
///
/// The p-value is exact while the group sizes keep the split count under
/// [`EXACT_PERMUTATION_LIMIT`]: every way the pooled ranks could have been dealt
/// into two groups of the observed sizes is enumerated, which needs no assumption
/// about the shape of the distribution — the one thing a ten-sample timing run
/// cannot afford to assume. Past that limit the tie-corrected normal approximation
/// stands in, and the method travels with the number so a reader knows which they
/// are looking at.
pub fn mann_whitney(current: &[f64], baseline: &[f64]) -> Option<(&'static str, f64)> {
    let (n_a, n_b) = (current.len(), baseline.len());
    if n_a == 0 || n_b == 0 {
        return None;
    }
    let mut pooled = current.to_vec();
    pooled.extend_from_slice(baseline);
    let n = n_a + n_b;
    let ranks = mid_ranks(&pooled);

    let rank_sum_a: f64 = ranks[..n_a].iter().sum();
    let expected_sum_a = n_a as f64 * (n + 1) as f64 / 2.0;
    let deviation = (rank_sum_a - expected_sum_a).abs();

    if binom(n, n_a) <= EXACT_PERMUTATION_LIMIT {
        let mut tally = RankSumTally {
            ranks: &ranks,
            need: n_a,
            expected: expected_sum_a,
            target: deviation,
            tol: 1e-9,
            total: 0,
            extreme: 0,
        };
        tally.walk(0, 0, 0.0);
        let (total, extreme) = (tally.total, tally.extreme);
        if total == 0 {
            return None;
        }
        return Some((
            "exact permutation",
            (extreme as f64 / total as f64).min(1.0),
        ));
    }

    let mean_u = n_a as f64 * n_b as f64 / 2.0;
    let tie_groups = tie_group_sizes(&pooled);
    let tie_correction: f64 = tie_groups.iter().map(|t| (t.pow(3) - *t) as f64).sum();
    let denominator = (n * (n - 1)).max(1) as f64;
    let sigma_sq =
        (n_a as f64 * n_b as f64 / 12.0) * ((n + 1) as f64 - tie_correction / denominator);
    if sigma_sq <= 0.0 {
        return None;
    }
    let u_a = rank_sum_a - n_a as f64 * (n_a + 1) as f64 / 2.0;
    // Continuity correction: a discrete statistic compared against a continuous
    // curve, half a step in the conservative direction.
    let z = (u_a - mean_u).abs() - 0.5;
    if z <= 0.0 {
        return Some(("normal approximation", 1.0));
    }
    let p = two_sided_p_from_z(z / sigma_sq.sqrt());
    Some(("normal approximation", p))
}

fn tie_group_sizes(values: &[f64]) -> Vec<usize> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut groups = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j + 1 < sorted.len() && sorted[j + 1] == sorted[i] {
            j += 1;
        }
        if j > i {
            groups.push(j - i + 1);
        }
        i = j + 1;
    }
    groups
}

/// Two-sided p-value from an absolute z score: `2 · Φ̄(z)`, by Abramowitz &
/// Stegun 26.2.17.
///
/// Accurate to 7.5 parts in 10⁸ absolute. That decides the only thing the
/// approximation branch is ever used for — whether p is under the significance
/// level — and a p below about 10⁻⁷ is past the point where the digits mean
/// anything; it reports as 0.0, which reads as "far below any threshold in use".
fn two_sided_p_from_z(z: f64) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let pdf = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    (2.0 * pdf * poly).clamp(0.0, 1.0)
}

/// Depth-first walk over every size-`need` subset of the pooled ranks, counting
/// the ones at least `target` away from the expected rank sum. Both tails are
/// counted, which is what makes the resulting p-value two-sided.
struct RankSumTally<'a> {
    ranks: &'a [f64],
    need: usize,
    expected: f64,
    target: f64,
    tol: f64,
    total: u128,
    extreme: u128,
}

impl<'a> RankSumTally<'a> {
    /// `index` is how far into `ranks` the walk has reached, `picked` how many
    /// of those it has taken, and `sum` their rank sum so far.
    fn walk(&mut self, index: usize, picked: usize, sum: f64) {
        if picked == self.need {
            self.total += 1;
            if (sum - self.expected).abs() >= self.target - self.tol {
                self.extreme += 1;
            }
            return;
        }
        if self.ranks.len() - index < self.need - picked {
            return;
        }
        self.walk(index + 1, picked + 1, sum + self.ranks[index]);
        self.walk(index + 1, picked, sum);
    }
}

/// Compare a fresh run against a baseline.
///
/// A refusal is a distinct outcome from "no change" on purpose: a run that
/// declines to answer has to be visible, otherwise a loaded machine reads as a
/// clean bill of health.
pub fn compare(
    baseline: &Baseline,
    measured: &Measured,
    alert_pct: f64,
    alpha: f64,
) -> Vec<Comparison> {
    let corpus_mismatch = baseline.corpus_files != measured.corpus_files;
    let mut ids: Vec<String> = baseline.benches.keys().cloned().collect();
    for id in measured.benches.keys() {
        if !baseline.benches.contains_key(id) {
            ids.push(id.clone());
        }
    }
    ids.sort();
    ids.dedup();

    ids.into_iter()
        .map(|id| compare_one(baseline, measured, &id, alert_pct, alpha, corpus_mismatch))
        .collect()
}

fn compare_one(
    baseline: &Baseline,
    measured: &Measured,
    id: &str,
    alert_pct: f64,
    alpha: f64,
    corpus_mismatch: bool,
) -> Comparison {
    let base = baseline.benches.get(id);
    let current = measured.benches.get(id);

    let mut comparison = Comparison {
        id: id.to_string(),
        baseline_median_ns: base.map(|s| s.median()),
        current_median_ns: current.map(|s| s.median()),
        delta_pct: None,
        p_value: None,
        p_value_method: None,
        cv: current.and_then(|s| s.cv()),
        outcome: Outcome::NoChange,
        reason: None,
    };

    let (Some(base), Some(current)) = (base, current) else {
        if current.is_none() {
            comparison.outcome = Outcome::NotMeasured;
            comparison.reason = Some("in the baseline but not measured by this run".to_string());
        } else {
            comparison.outcome = Outcome::NoBaseline;
            comparison.reason = Some(
                "measured now and absent from the baseline, so there is nothing to compare it \
                 against"
                    .to_string(),
            );
        }
        return comparison;
    };

    if base.n() < 2 || current.n() < 2 {
        comparison.outcome = Outcome::Refused;
        comparison.reason = Some(format!(
            "one side has {} sample(s), and spread cannot be estimated from fewer than two",
            base.n().min(current.n())
        ));
        return comparison;
    }

    let base_median = base.median();
    if base_median > 0.0 {
        comparison.delta_pct = Some((current.median() - base_median) / base_median * 100.0);
    }
    let Some((method, p)) = mann_whitney(&current.ns_per_iteration, &base.ns_per_iteration) else {
        comparison.outcome = Outcome::Refused;
        comparison.reason = Some("the test could not be computed from these samples".to_string());
        return comparison;
    };
    comparison.p_value = Some(p);
    comparison.p_value_method = Some(method);

    if corpus_mismatch {
        comparison.outcome = Outcome::Refused;
        comparison.reason = Some(format!(
            "the baseline was measured over {} file(s) and this run over {}, which are \
             different trees; re-record the baseline before reading a verdict",
            baseline.corpus_files, measured.corpus_files
        ));
        return comparison;
    }

    if let Some(refusal) = spread_refusal(comparison.cv, base.cv()) {
        comparison.outcome = Outcome::Refused;
        comparison.reason = Some(refusal);
        return comparison;
    }

    let delta = comparison.delta_pct.unwrap_or(0.0);
    if p > alpha {
        comparison.outcome = Outcome::NoChange;
        // No reason here: the line above already carries the p-value, and a
        // table where every clean row apologises for being clean says nothing.
    } else if delta >= alert_pct {
        comparison.outcome = Outcome::Regressed;
        comparison.reason = Some(format!(
            "the samples separate at the {}% level (p = {p:.4}) and the path runs \
             {}{delta:.2}% slower than the baseline",
            alpha * 100.0,
            if delta >= 0.0 { "+" } else { "" },
        ));
    } else if delta <= -alert_pct {
        comparison.outcome = Outcome::Improved;
    } else {
        comparison.outcome = Outcome::NoChange;
        comparison.reason = Some(format!(
            "separated at p = {p:.4} but only {}{:.2}% away, under the {:.0}% this harness \
             reports as a change",
            if delta >= 0.0 { "+" } else { "" },
            delta.abs(),
            alert_pct
        ));
    }
    comparison
}

/// The spread reason, if either side is too noisy to support a verdict.
fn spread_refusal(current_cv: Option<f64>, baseline_cv: Option<f64>) -> Option<String> {
    if let Some(cv) = current_cv {
        if cv > MAX_CV {
            return Some(format!(
                "this run's spread is {:.1}% of its own level, past the {:.0}% a verdict is \
                 allowed to rest on",
                cv * 100.0,
                MAX_CV * 100.0
            ));
        }
    }
    if let Some(cv) = baseline_cv {
        if cv > MAX_CV {
            return Some(format!(
                "the baseline was recorded at {:.1}% spread, past the {:.0}% this build \
                 refuses at; re-record it on a quieter machine",
                cv * 100.0,
                MAX_CV * 100.0
            ));
        }
    }
    None
}

/// Where the baseline belongs: the nearest ancestor holding `.git`, so running
/// from `rust/` does not create a second `.xencode/` beside the workspace.
/// Outside a repository that is the starting directory itself.
pub fn state_root(start: &Path) -> PathBuf {
    let mut dir = start.to_path_buf();
    loop {
        if dir.join(".git").exists() {
            return dir;
        }
        if !dir.pop() {
            return start.to_path_buf();
        }
    }
}

/// The paths in a run whose own spread is past [`MAX_CV`].
///
/// A baseline recorded from those poisons every comparison made against it
/// afterwards — the refusal then lands on the *later* run, which had nothing to
/// do with it — so `record` turns this into a refusal at the point of writing.
pub fn noisy_paths(measured: &Measured) -> Vec<String> {
    measured
        .benches
        .values()
        .filter(|s| s.cv().is_some_and(|c| c > MAX_CV))
        .map(|s| format!("{} at {:.1}%", s.id, s.cv().unwrap_or_default() * 100.0))
        .collect()
}

/// Record a baseline from a fresh run, keeping the corpus it was measured over.
///
/// `force` writes it anyway. Without that, a run disturbed by anything else on
/// the machine is refused here rather than haunting the next ten checks.
pub fn record_baseline(
    repo_root: &Path,
    measured: &Measured,
    force: bool,
) -> Result<Baseline, String> {
    let noisy = noisy_paths(measured);
    if !noisy.is_empty() && !force {
        return Err(format!(
            "refused to record a baseline from a run wider than {:.0}% spread on: {}; \
             wait for the machine to go quiet, or pass --force to record it anyway",
            MAX_CV * 100.0,
            noisy.join(", ")
        ));
    }
    let baseline = Baseline {
        version: BASELINE_VERSION,
        corpus_files: measured.corpus_files,
        recorded_in: repo_root.display().to_string(),
        recorded_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        benches: measured.benches.clone(),
    };
    baseline.write(repo_root)?;
    Ok(baseline)
}

/// Measure and compare in one step, loading the stored baseline first.
///
/// With no baseline recorded the report is empty by design and says so: a table
/// of fresh numbers next to no verdict invites the reader to invent one.
pub fn check(
    repo_root: &Path,
    manifest_root: &Path,
    filter: Option<&str>,
    alert_pct: f64,
    alpha: f64,
) -> Result<Report, String> {
    let stored = Baseline::load(repo_root)?;
    let measured = measure(manifest_root, filter)?;
    let Some(baseline) = stored else {
        return Ok(Report {
            corpus_files: measured.corpus_files,
            baseline_corpus_files: None,
            corpus_mismatch: false,
            comparisons: Vec::new(),
            took: measured.took,
            command: measured.command,
            notes: vec![format!(
                "no baseline recorded yet, so nothing can be compared — run `xencode perf \
                 record` first (it writes {})",
                Baseline::path_for(repo_root).display()
            )],
        });
    };

    let comparisons = compare(&baseline, &measured, alert_pct, alpha);
    let corpus_mismatch = baseline.corpus_files != measured.corpus_files;
    let mut notes = Vec::new();
    if corpus_mismatch {
        notes.push(format!(
            "every comparison was refused: the baseline describes a {}-file tree and this \
             run measured a {}-file one",
            baseline.corpus_files, measured.corpus_files
        ));
    }
    if measured
        .benches
        .values()
        .any(|s| s.cv().is_some_and(|c| c > MAX_CV))
    {
        notes.push(format!(
            "at least one path was measured above the {:.0}% spread a verdict is allowed to \
             rest on — this machine was doing something else",
            MAX_CV * 100.0
        ));
    }
    Ok(Report {
        corpus_files: measured.corpus_files,
        baseline_corpus_files: Some(baseline.corpus_files),
        corpus_mismatch,
        comparisons,
        took: measured.took,
        command: measured.command,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(id: &str, values: &[f64]) -> Samples {
        Samples {
            id: id.to_string(),
            ns_per_iteration: values.to_vec(),
        }
    }

    /// A level with a fixed three-point wobble, so its spread is known: jitter of
    /// 0.5% of the level stays well under the refusal threshold.
    fn wobbling(id: &str, value: f64, n: usize, jitter: f64) -> Samples {
        samples(
            id,
            &(0..n)
                .map(|i| value + jitter * ((i % 3) as f64 - 1.0))
                .collect::<Vec<f64>>(),
        )
    }

    fn baseline_of(benches: Vec<Samples>) -> Baseline {
        Baseline {
            version: BASELINE_VERSION,
            corpus_files: 400,
            recorded_in: "test".to_string(),
            recorded_unix: 0,
            benches: benches.into_iter().map(|s| (s.id.clone(), s)).collect(),
        }
    }

    fn measured_of(benches: Vec<Samples>, corpus_files: usize) -> Measured {
        Measured {
            benches: benches.into_iter().map(|s| (s.id.clone(), s)).collect(),
            corpus_files,
            command: "test".to_string(),
            took: Duration::from_secs(1),
        }
    }

    #[test]
    fn cv_is_spread_over_level_and_needs_two_samples() {
        assert_eq!(samples("x", &[100.0]).stdev(), None);
        let tight = samples("x", &[100.0, 101.0, 99.0, 100.0, 100.0]);
        assert!(tight.cv().unwrap() < 0.02);
        let loose = samples("x", &[100.0, 60.0, 140.0, 90.0, 110.0]);
        assert!(loose.cv().unwrap() > 0.2);
    }

    #[test]
    fn the_median_ignores_an_outlier_that_would_move_the_mean() {
        let s = samples("x", &[10.0, 10.0, 10.0, 10.0, 1000.0]);
        assert_eq!(s.median(), 10.0);
        assert!(s.mean() > 200.0);
    }

    #[test]
    fn mid_ranks_average_a_tied_block() {
        let ranks = mid_ranks(&[5.0, 1.0, 5.0, 3.0]);
        // 1.0 is rank 1, 3.0 is rank 2, and the two 5.0s share ranks 3 and 4.
        assert_eq!(ranks, vec![3.5, 1.0, 3.5, 2.0]);
    }

    #[test]
    fn binom_counts_the_splits_the_enumerator_walks() {
        assert_eq!(binom(20, 10), 184_756);
        assert_eq!(binom(8, 4), 70);
        assert_eq!(binom(5, 0), 1);
        assert!(binom(60, 30) > EXACT_PERMUTATION_LIMIT);
    }

    #[test]
    fn the_normal_approximation_matches_the_textbook_values() {
        // z = 0 is the whole probability mass: two sides of 0.5.
        assert!((two_sided_p_from_z(0.0) - 1.0).abs() < 1e-7);
        assert!((two_sided_p_from_z(1.0) - 0.317_310_5).abs() < 1e-6);
        // The textbook two-sided p at z = 1.959964 is exactly 0.05.
        assert!((two_sided_p_from_z(1.959_963_985) - 0.05).abs() < 1e-6);
        assert!((two_sided_p_from_z(3.0) - 0.002_699_8).abs() < 1e-6);
        assert!((two_sided_p_from_z(6.0) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn mann_whitney_separates_a_shift_and_not_an_identical_pair() {
        let a: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        let b: Vec<f64> = (0..10).map(|i| 112.0 + i as f64).collect();
        let (method, p) = mann_whitney(&b, &a).unwrap();
        assert_eq!(method, "exact permutation");
        assert!(p < 0.05, "a 12% shift should separate, got p = {p}");

        let same = mann_whitney(&a, &a).unwrap();
        assert!(
            same.1 > 0.9,
            "identical samples must not separate, got p = {}",
            same.1
        );
    }

    #[test]
    fn an_interleaved_pair_reads_as_no_separation() {
        let a = vec![1.0, 4.0, 6.0, 9.0];
        let b = vec![2.0, 3.0, 7.0, 8.0];
        let (_, p) = mann_whitney(&a, &b).unwrap();
        assert!(p > 0.5, "interleaving should not separate, got p = {p}");
    }

    #[test]
    fn large_groups_fall_back_to_the_approximation_and_stay_probabilities() {
        let a: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let b: Vec<f64> = (0..20).map(|i| 130.0 + i as f64).collect();
        let (method, p) = mann_whitney(&b, &a).unwrap();
        assert_eq!(method, "normal approximation");
        assert!((0.0..=1.0).contains(&p));
        assert!(p < 0.001);
    }

    #[test]
    fn an_injected_ten_percent_slowdown_fires_and_a_clean_rerun_does_not() {
        let base = baseline_of(vec![wobbling(
            "compaction/soft_compact",
            1_000_000.0,
            10,
            5_000.0,
        )]);

        let quiet = measured_of(
            vec![wobbling(
                "compaction/soft_compact",
                1_000_600.0,
                10,
                5_000.0,
            )],
            400,
        );
        let report = Report {
            corpus_files: 400,
            baseline_corpus_files: Some(400),
            corpus_mismatch: false,
            comparisons: compare(&base, &quiet, DEFAULT_ALERT_PCT, DEFAULT_ALPHA),
            took: Duration::from_secs(1),
            command: "test".to_string(),
            notes: vec![],
        };
        assert!(!report.has_regression(), "a no-change rerun must not fire");

        let injected = measured_of(
            vec![wobbling(
                "compaction/soft_compact",
                1_100_000.0,
                10,
                5_500.0,
            )],
            400,
        );
        let comparisons = compare(&base, &injected, DEFAULT_ALERT_PCT, DEFAULT_ALPHA);
        let fired = comparisons
            .iter()
            .find(|c| c.outcome == Outcome::Regressed)
            .expect("an injected 10% slowdown must fire");
        assert!(fired.delta_pct.unwrap() > 9.0);
        assert!(fired.p_value.unwrap() <= DEFAULT_ALPHA);
    }

    #[test]
    fn a_run_over_a_different_tree_is_refused_rather_than_compared() {
        let base = baseline_of(vec![wobbling(
            "retrieve/hybrid_top10",
            30_000_000.0,
            10,
            30_000.0,
        )]);
        // A 50% "slowdown", which would fire on any other tree.
        let run = measured_of(
            vec![wobbling(
                "retrieve/hybrid_top10",
                45_000_000.0,
                10,
                45_000.0,
            )],
            411,
        );
        let comparisons = compare(&base, &run, DEFAULT_ALERT_PCT, DEFAULT_ALPHA);
        assert_eq!(comparisons[0].outcome, Outcome::Refused);
        assert!(comparisons[0]
            .reason
            .as_ref()
            .unwrap()
            .contains("different trees"));
        assert!(!comparisons.iter().any(|c| c.outcome == Outcome::Regressed));
    }

    #[test]
    fn a_loaded_run_above_five_percent_spread_refuses_even_a_real_slowdown() {
        let base = baseline_of(vec![wobbling(
            "index_build/scan_tree",
            1_000_000.0,
            10,
            5_000.0,
        )]);
        let noisy = samples(
            "index_build/scan_tree",
            &[
                900_000.0,
                1_600_000.0,
                950_000.0,
                1_500_000.0,
                1_000_000.0,
                1_400_000.0,
                980_000.0,
                1_550_000.0,
                1_020_000.0,
                1_480_000.0,
            ],
        );
        assert!(noisy.cv().unwrap() > MAX_CV);
        let run = measured_of(vec![noisy], 400);
        let comparisons = compare(&base, &run, DEFAULT_ALERT_PCT, DEFAULT_ALPHA);
        assert_eq!(comparisons[0].outcome, Outcome::Refused);
        assert!(comparisons[0]
            .reason
            .as_ref()
            .unwrap()
            .contains("spread is"));
    }

    #[test]
    fn a_baseline_recorded_on_a_noisy_machine_is_refused_too() {
        let mut noisy_base = wobbling("compaction/soft_compact", 1_000_000.0, 10, 5_000.0);
        noisy_base.ns_per_iteration = vec![
            800_000.0,
            1_200_000.0,
            850_000.0,
            1_300_000.0,
            900_000.0,
            1_100_000.0,
            820_000.0,
            1_400_000.0,
            950_000.0,
            1_250_000.0,
        ];
        let base = baseline_of(vec![noisy_base]);
        let run = measured_of(
            vec![wobbling(
                "compaction/soft_compact",
                1_100_000.0,
                10,
                5_500.0,
            )],
            400,
        );
        let comparisons = compare(&base, &run, DEFAULT_ALERT_PCT, DEFAULT_ALPHA);
        assert_eq!(comparisons[0].outcome, Outcome::Refused);
        assert!(comparisons[0]
            .reason
            .as_ref()
            .unwrap()
            .contains("baseline was recorded"));
    }

    #[test]
    fn a_significant_but_small_difference_is_reported_as_no_change() {
        let base = baseline_of(vec![wobbling(
            "truncation/truncate_to_tokens",
            1_000_000.0,
            10,
            500.0,
        )]);
        // 1% slower: the samples separate cleanly, the threshold does not.
        let run = measured_of(
            vec![wobbling(
                "truncation/truncate_to_tokens",
                1_010_000.0,
                10,
                505.0,
            )],
            400,
        );
        let comparisons = compare(&base, &run, DEFAULT_ALERT_PCT, DEFAULT_ALPHA);
        assert_eq!(comparisons[0].outcome, Outcome::NoChange);
        assert!(comparisons[0].p_value.unwrap() < DEFAULT_ALPHA);
        assert!((comparisons[0].delta_pct.unwrap() - 1.0).abs() < 0.1);
    }

    #[test]
    fn missing_and_extra_paths_are_named_instead_of_dropped() {
        let base = baseline_of(vec![wobbling("compaction/soft_compact", 1.0, 10, 0.0)]);
        let run = measured_of(
            vec![wobbling("truncation/truncate_to_tokens", 1.0, 10, 0.0)],
            400,
        );
        let comparisons = compare(&base, &run, DEFAULT_ALERT_PCT, DEFAULT_ALPHA);
        assert_eq!(comparisons.len(), 2);
        assert_eq!(comparisons[0].outcome, Outcome::NotMeasured);
        assert_eq!(comparisons[1].outcome, Outcome::NoBaseline);
    }

    #[test]
    fn the_corpus_count_is_read_from_the_line_the_bench_prints() {
        let stdout = "\
running 1 test
xencode-perf-corpus: files=143 root=/home/sree/Projects/xencode/rust
Benchmarking compaction/soft_compact
";
        assert_eq!(parse_corpus_files(stdout), Some(143));
        assert_eq!(parse_corpus_files("nothing here"), None);
        // A marker line whose count does not parse is not a count.
        assert_eq!(
            parse_corpus_files("xencode-perf-corpus: files=many root=/x"),
            None
        );
    }

    #[test]
    fn samples_are_divided_by_their_iteration_count() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-perf-samples-{}-{}",
            std::process::id(),
            line!()
        ));
        let new = dir.join("target/criterion/some_bench/new");
        std::fs::create_dir_all(&new).unwrap();
        std::fs::write(
            new.join("benchmark.json"),
            r#"{"full_id":"retrieve/hybrid_top10","directory_name":"some_bench"}"#,
        )
        .unwrap();
        std::fs::write(
            new.join("sample.json"),
            r#"{"sampling_mode":"Linear","iters":[2.0,4.0],"times":[60.0,240.0]}"#,
        )
        .unwrap();
        let read = read_samples(&dir).unwrap();
        assert_eq!(
            read["retrieve/hybrid_top10"].ns_per_iteration,
            vec![30.0, 60.0]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_group_without_its_sample_file_is_skipped_not_guessed_at() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-perf-partial-{}-{}",
            std::process::id(),
            line!()
        ));
        let new = dir.join("target/criterion/half_written/new");
        std::fs::create_dir_all(&new).unwrap();
        std::fs::write(new.join("benchmark.json"), r#"{"full_id":"a/b"}"#).unwrap();
        assert!(read_samples(&dir).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clearing_leaves_nothing_for_a_filtered_run_to_mistake_for_a_measurement() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-perf-clear-{}-{}",
            std::process::id(),
            line!()
        ));
        let new = dir.join("target/criterion/some_bench/new");
        std::fs::create_dir_all(&new).unwrap();
        std::fs::write(new.join("sample.json"), r#"{"iters":[1.0],"times":[5.0]}"#).unwrap();
        assert_eq!(clear_previous_samples(&dir).unwrap(), 1);
        assert!(!new.join("sample.json").exists());
        assert!(read_samples(&dir).unwrap().is_empty());
        // A second clear has nothing to do, and a workspace without a criterion
        // directory is not an error.
        assert_eq!(clear_previous_samples(&dir).unwrap(), 0);
        assert_eq!(clear_previous_samples(&dir.join("nowhere")).unwrap(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_wide_run_is_refused_as_a_baseline_rather_than_warned_about() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-perf-noisy-record-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let wide = measured_of(
            vec![
                wobbling("compaction/soft_compact", 1_000_000.0, 10, 5_000.0),
                samples(
                    "index_build/extract_symbols",
                    &[
                        700_000_000.0,
                        900_000_000.0,
                        720_000_000.0,
                        880_000_000.0,
                        750_000_000.0,
                        850_000_000.0,
                        710_000_000.0,
                        890_000_000.0,
                        760_000_000.0,
                        840_000_000.0,
                    ],
                ),
            ],
            200,
        );
        assert_eq!(noisy_paths(&wide).len(), 1);
        let err = record_baseline(&dir, &wide, false).unwrap_err();
        assert!(err.contains("refused to record a baseline"), "{err}");
        assert!(err.contains("index_build/extract_symbols"), "{err}");
        assert!(
            !Baseline::path_for(&dir).exists(),
            "a refusal writes nothing"
        );

        let baseline = record_baseline(&dir, &wide, true).unwrap();
        assert_eq!(baseline.corpus_files, 200);
        assert!(Baseline::path_for(&dir).is_file());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_baseline_round_trips_and_a_stale_version_is_named() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-perf-baseline-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(Baseline::load(&dir).unwrap(), None);

        let base = baseline_of(vec![samples("compaction/soft_compact", &[1.0, 2.0, 3.0])]);
        let path = base.write(&dir).unwrap();
        assert!(path.is_file());
        let loaded = Baseline::load(&dir).unwrap().unwrap();
        assert_eq!(loaded.corpus_files, 400);
        assert_eq!(
            loaded.benches["compaction/soft_compact"].ns_per_iteration,
            vec![1.0, 2.0, 3.0]
        );

        let text = std::fs::read_to_string(&path).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["version"] = serde_json::json!(99);
        std::fs::write(&path, value.to_string()).unwrap();
        let err = Baseline::load(&dir).unwrap_err();
        assert!(err.contains("version 99"), "{err}");

        std::fs::write(&path, "{ not json").unwrap();
        assert!(Baseline::load(&dir)
            .unwrap_err()
            .contains("not a readable baseline"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
