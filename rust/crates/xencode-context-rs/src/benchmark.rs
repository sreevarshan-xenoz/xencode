//! Benchmark foundation for the Intelligence Core (XI-9).
//!
//! This module is the measurement layer the rest of the intelligence work has to
//! earn its way past. It does not measure intelligence itself — no model is
//! called from here — it records *what was measured, under which instructions
//! and configuration, how many times, and with what spread*, then refuses to
//! call a difference an improvement when the two records are not comparable.
//!
//! Everything in the record is reused from the existing measurement types:
//! - the retrieval half embeds [`crate::EvalReport`] (recall@k, precision@k, MRR
//!   and per-probe ranks) rather than copying its fields into new ones;
//! - the probes are [`crate::EvalItem`]s from the checked-in gold set, and the
//!   task half names the ten [`crate::BugShape`]s from the seeded-case harness;
//! - rates are derived from real [`crate::TurnTrace`], [`crate::seeds::GraderRun`]
//!   and [`crate::RequestMetrics`] rows, and a field is `None` when there was no
//!   evidence for it, never a zero invented to complete a table;
//! - the instruction set is the digest from [`crate::prompts::set_version`].
//!
//! An [`EvidenceKind::HarnessValidation`] record proves the harness works: a
//! checked-in reference change makes a seeded case go from red to green. It is
//! emphatically not a model pass rate. Only a [`EvidenceKind::Model`] record,
//! with enough repeats under the same prompts, config and corpus, may carry an
//! improvement claim — and even then the claim is about those recorded runs,
//! not about a feature being finished. Real-model pass rates need GPU/model
//! access and are blocked in environments that do not have it.

use crate::eval::EvalReport;
use crate::metrics::RequestMetrics;
use crate::seeds::{BugShape, GraderRun};
use crate::trace::{ChecksVerdict, TurnTrace};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Bump only when the on-disk shape of [`BenchmarkRecord`] changes in a way that
/// makes an older record impossible to compare honestly.
pub const BENCHMARK_SCHEMA_VERSION: u32 = 1;

/// How many finished runs a record needs before a comparison will look at its
/// numbers at all. A single run is an observation, not a measurement.
pub const MIN_REPEATS: usize = 3;

/// Whether a record describes the harness itself or a real model run. This is
/// part of the schema because a reference-fix run and a model run can produce
/// the same pass rate for completely different reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    /// A checked-in reference change graded against the seeded cases. Validates
    /// the measurement, never the intelligence being measured.
    HarnessValidation,
    /// A run whose outcomes came from a model. The only kind that may support
    /// an improvement claim.
    Model,
}

/// The conditions a run was measured under. Stored in full, not just as a
/// digest, so a reader can see what changed; the digest is what the comparison
/// checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkConfig {
    pub model: Option<String>,
    pub provider: Option<String>,
    pub hardware_profile: Option<String>,
    pub top_k: usize,
    /// `None` when the sampler chose its own draws: such a run cannot be
    /// repeated even in principle.
    pub seed: Option<i64>,
    pub temperature: Option<f64>,
    /// The grader command, joined for the record, when a task half ran.
    pub grader: Option<String>,
}

impl BenchmarkConfig {
    /// Stable digest of the whole configuration, in declaration order.
    pub fn digest(&self) -> String {
        digest_json(self)
    }
}

/// The fixed input set: retrieval probes plus the shapes the task half uses.
/// Both are reused verbatim from the checked-in corpus; neither is generated
/// by the code under test.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkCorpus {
    /// Gold probes, reused from [`crate::eval::default_gold`] or
    /// `.xencode/eval/gold.json` — never invented inside this module.
    pub gold: Vec<crate::EvalItem>,
    /// Slugs of the seeded bug shapes, in [`crate::BugShape::all`] order.
    pub seed_shapes: Vec<String>,
}

impl BenchmarkCorpus {
    /// The checked-in gold set plus every seeded shape, as one corpus.
    pub fn from_checked_in(gold: Vec<crate::EvalItem>) -> Self {
        Self {
            gold,
            seed_shapes: BugShape::all()
                .iter()
                .map(|shape| shape.slug().to_string())
                .collect(),
        }
    }

    /// Stable digest of the corpus. A comparison across two different corpora
    /// is a comparison of the questions, not of the system being asked.
    pub fn digest(&self) -> String {
        digest_json(self)
    }
}

/// One measured task result. These are the booleans a harness fills from a
/// real grader, a real edit and real checks — deliberately not a copy of the
/// TUI's `CaseResult`, which also carries diffs, retry counts and reports this
/// layer has no use for.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskOutcome {
    /// Slug of the seeded shape this case came from.
    pub shape: String,
    /// A real edit landed on one of the files the case expects to change.
    pub edit_applied: bool,
    /// The case's own grader exited successfully.
    pub grader_passed: bool,
    /// The answer claimed it was verified. Recorded separately so a claim that
    /// was never checked can be counted as a false verification, not as a pass.
    pub claimed_verified: bool,
    /// What the checks after the edit actually found, when any ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checks: Option<ChecksVerdict>,
    /// A check that had passed before this run failed on it.
    pub regressed: bool,
    /// This case hit a failure and then recovered within the same run.
    pub recovered: bool,
    /// Whether this case hit any failure at all, the denominator for recovery.
    pub had_failure: bool,
}

impl TaskOutcome {
    /// Build an outcome from a real grader run and the checks that ran with it.
    /// `claim` is the answer's own statement, kept apart from what the checks
    /// found so the two cannot quietly become the same thing.
    #[allow(clippy::too_many_arguments)]
    pub fn from_grader(
        shape: &str,
        grader: &GraderRun,
        checks: Option<ChecksVerdict>,
        claim: Option<bool>,
        edit_applied: bool,
        regressed: bool,
        had_failure: bool,
        recovered: bool,
    ) -> Self {
        Self {
            shape: shape.to_string(),
            edit_applied,
            grader_passed: grader.passed,
            claimed_verified: claim.unwrap_or(false),
            checks,
            regressed,
            recovered,
            had_failure,
        }
    }

    /// A claimed verification with no passing check behind it. An exit code
    /// from a grader is not a check that the change was right; only a real
    /// [`ChecksVerdict`] counts, and an absent one is not a pass.
    pub fn falsely_verified(&self) -> bool {
        self.claimed_verified && !self.checks.as_ref().is_some_and(ChecksVerdict::all_passed)
    }
}

/// The named metrics of the benchmark. Retrieval metrics come from the embedded
/// report; the rest come from [`TaskMetrics`]. Keeping one enum for both means
/// a comparison and a variance summary cannot drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    TaskCompletion,
    EditSuccess,
    RetrievalMrr,
    RetrievalRecallAt1,
    RetrievalPrecisionAt1,
    ToolCallValidity,
    RegressionRate,
    FalseVerifiedRate,
    RecoverySuccess,
    LatencyMs,
    ResourceCostMicros,
    ResourceEnergyUj,
    Tokens,
}

impl Metric {
    /// Every metric, in the order a comparison reports them.
    pub const ALL: &'static [Metric] = &[
        Metric::TaskCompletion,
        Metric::EditSuccess,
        Metric::RetrievalMrr,
        Metric::RetrievalRecallAt1,
        Metric::RetrievalPrecisionAt1,
        Metric::ToolCallValidity,
        Metric::RegressionRate,
        Metric::FalseVerifiedRate,
        Metric::RecoverySuccess,
        Metric::LatencyMs,
        Metric::ResourceCostMicros,
        Metric::ResourceEnergyUj,
        Metric::Tokens,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Metric::TaskCompletion => "task_completion",
            Metric::EditSuccess => "edit_success",
            Metric::RetrievalMrr => "retrieval_mrr",
            Metric::RetrievalRecallAt1 => "retrieval_recall_at_1",
            Metric::RetrievalPrecisionAt1 => "retrieval_precision_at_1",
            Metric::ToolCallValidity => "tool_call_validity",
            Metric::RegressionRate => "regression_rate",
            Metric::FalseVerifiedRate => "false_verified_rate",
            Metric::RecoverySuccess => "recovery_success",
            Metric::LatencyMs => "latency_ms",
            Metric::ResourceCostMicros => "resource_cost_micros",
            Metric::ResourceEnergyUj => "resource_energy_uj",
            Metric::Tokens => "tokens",
        }
    }

    /// This metric's value in one run. `None` is the honest answer when nothing
    /// measured it — a run with no model has no task completion, and a run with
    /// no power counter has no energy reading.
    pub fn value(self, run: &BenchmarkRun) -> Option<f64> {
        match self {
            Metric::TaskCompletion => run.task.task_completion,
            Metric::EditSuccess => run.task.edit_success,
            Metric::RetrievalMrr => Some(run.retrieval.mrr),
            Metric::RetrievalRecallAt1 => run.retrieval.recall_at.first().copied(),
            Metric::RetrievalPrecisionAt1 => run.retrieval.precision_at.first().copied(),
            Metric::ToolCallValidity => run.task.tool_call_validity,
            Metric::RegressionRate => run.task.regression_rate,
            Metric::FalseVerifiedRate => run.task.false_verified_rate,
            Metric::RecoverySuccess => run.task.recovery_success,
            Metric::LatencyMs => run.task.latency_ms,
            Metric::ResourceCostMicros => run.task.est_cost_micros.map(|v| v as f64),
            Metric::ResourceEnergyUj => run.task.energy_uj.map(|v| v as f64),
            Metric::Tokens => run.task.tokens.map(|v| v as f64),
        }
    }
}

/// Task-side rates for one run. Every field is optional on purpose: a partial
/// run must be visible as partial, not rounded up to a reassuring zero.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TaskMetrics {
    /// Fraction of cases whose own grader passed.
    pub task_completion: Option<f64>,
    /// Fraction of cases where a real edit touched an expected file.
    pub edit_success: Option<f64>,
    /// Fraction of tool calls that completed (`done`) rather than being denied,
    /// refused or failing, from real [`TurnTrace`] rows.
    pub tool_call_validity: Option<f64>,
    /// Fraction of cases where a previously passing check regressed.
    pub regression_rate: Option<f64>,
    /// Fraction of verification claims with no passing check behind them.
    pub false_verified_rate: Option<f64>,
    /// Recovered cases over the cases that hit a failure.
    pub recovery_success: Option<f64>,
    /// Mean wall-clock milliseconds of the turns in this run.
    pub latency_ms: Option<f64>,
    /// Sum of the estimated costs on the request rows, in micro-dollars.
    pub est_cost_micros: Option<u64>,
    /// Sum of the energy readings on the request rows, in microjoules.
    pub energy_uj: Option<u64>,
    /// Sum of prompt plus completion tokens on the request rows.
    pub tokens: Option<u64>,
}

impl TaskMetrics {
    /// Derive every rate from real outcomes, traces and request rows. Nothing
    /// is filled in from a default: no outcomes means no task rates, no traces
    /// means no tool or latency rates, no request rows means no resource rates.
    pub fn from_evidence(
        outcomes: &[TaskOutcome],
        traces: &[TurnTrace],
        requests: &[RequestMetrics],
    ) -> Self {
        let mean = |values: Vec<f64>| -> Option<f64> {
            (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
        };
        let ratio =
            |num: usize, den: usize| -> Option<f64> { (den > 0).then(|| num as f64 / den as f64) };

        let tool_calls: Vec<&crate::trace::ToolTrace> =
            traces.iter().flat_map(|trace| trace.tools.iter()).collect();
        let failed_cases = outcomes.iter().filter(|o| o.had_failure).count();
        let recovered = outcomes.iter().filter(|o| o.recovered).count();

        let est_cost_micros: Vec<u64> = requests
            .iter()
            .filter_map(|row| row.est_cost_micros)
            .collect();
        let energy_uj: Vec<u64> = requests.iter().filter_map(|row| row.energy_uj).collect();
        let tokens: u64 = requests
            .iter()
            .map(|row| row.prompt_tokens as u64 + row.completion_tokens as u64)
            .sum();
        let latencies: Vec<f64> = traces
            .iter()
            .map(|trace| trace.duration_ms as f64)
            .chain(
                requests
                    .iter()
                    .filter_map(|row| row.elapsed_ms.map(|ms| ms as f64)),
            )
            .collect();

        Self {
            task_completion: mean(
                outcomes
                    .iter()
                    .map(|o| u8::from(o.grader_passed) as f64)
                    .collect(),
            ),
            edit_success: mean(
                outcomes
                    .iter()
                    .map(|o| u8::from(o.edit_applied) as f64)
                    .collect(),
            ),
            tool_call_validity: ratio(
                tool_calls.iter().filter(|t| t.outcome == "done").count(),
                tool_calls.len(),
            ),
            regression_rate: mean(
                outcomes
                    .iter()
                    .map(|o| u8::from(o.regressed) as f64)
                    .collect(),
            ),
            false_verified_rate: mean(
                outcomes
                    .iter()
                    .map(|o| u8::from(o.falsely_verified()) as f64)
                    .collect(),
            ),
            recovery_success: ratio(recovered, failed_cases),
            latency_ms: mean(latencies),
            est_cost_micros: (!est_cost_micros.is_empty()).then(|| est_cost_micros.iter().sum()),
            energy_uj: (!energy_uj.is_empty()).then(|| energy_uj.iter().sum()),
            tokens: (!requests.is_empty()).then_some(tokens),
        }
    }
}

/// Why a case failed, so a regression can be read as a kind of problem rather
/// than an undifferentiated drop in a single number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    RetrievalMiss,
    EditFailure,
    GraderFailure,
    CheckFailure,
    FalseVerified,
    ToolError,
    ProviderError,
    NoChangeClaim,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailureClass {
    pub kind: FailureKind,
    /// `None` for failures that are not attached to a seeded shape.
    pub shape: Option<String>,
    pub count: usize,
}

/// Classify the failures that are actually in the evidence. A probe with no
/// hit in the top-K is a retrieval miss; a failed turn is a provider error; a
/// failed edit tool or a no-change note is an edit failure; a case whose grader
/// is red is a grader failure. No failure class is emitted without a row that
/// produced it.
pub fn classify_failures(
    report: &EvalReport,
    outcomes: &[TaskOutcome],
    traces: &[TurnTrace],
) -> Vec<FailureClass> {
    let mut counts: Vec<(FailureKind, Option<String>, usize)> = Vec::new();
    let mut add = |kind: FailureKind, shape: Option<String>| {
        if let Some(row) = counts
            .iter_mut()
            .find(|(k, s, _)| *k == kind && *s == shape)
        {
            row.2 += 1;
        } else {
            counts.push((kind, shape, 1));
        }
    };

    for (_, _, rank, _) in &report.hits {
        if *rank == usize::MAX {
            add(FailureKind::RetrievalMiss, None);
        }
    }
    for outcome in outcomes {
        let shape = Some(outcome.shape.clone());
        if !outcome.grader_passed {
            add(FailureKind::GraderFailure, shape.clone());
        }
        if outcome.falsely_verified() {
            add(FailureKind::FalseVerified, shape.clone());
        }
        if outcome
            .checks
            .as_ref()
            .is_some_and(|checks| !checks.failed.is_empty() || !checks.skipped.is_empty())
        {
            add(FailureKind::CheckFailure, shape);
        }
    }
    for trace in traces {
        if trace.failed {
            add(FailureKind::ProviderError, None);
        }
        if trace.failed_edit_count > 0 {
            add(FailureKind::EditFailure, None);
        }
        if !trace.no_change_notes.is_empty() {
            add(FailureKind::NoChangeClaim, None);
        }
        if trace
            .tools
            .iter()
            .any(|tool| tool.outcome == "failed" || tool.outcome == "refused")
        {
            add(FailureKind::ToolError, None);
        }
    }

    counts
        .into_iter()
        .map(|(kind, shape, count)| FailureClass { kind, shape, count })
        .collect()
}

/// One finished repetition of the benchmark.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkRun {
    pub run_index: usize,
    pub ts_unix_ms: u64,
    /// Retrieval results kept in the shape [`crate::eval::evaluate`] already
    /// produces; the recall/precision vectors and MRR are not re-declared here.
    pub retrieval: EvalReport,
    pub task: TaskMetrics,
    pub outcomes: Vec<TaskOutcome>,
    pub failures: Vec<FailureClass>,
}

impl BenchmarkRun {
    /// Wrap one real retrieval report and its task evidence. `run_index` is the
    /// caller's position in the repeat sequence, not a count of anything that
    /// might have been skipped.
    pub fn new(
        run_index: usize,
        retrieval: EvalReport,
        outcomes: Vec<TaskOutcome>,
        traces: &[TurnTrace],
        requests: &[RequestMetrics],
    ) -> Self {
        let task = TaskMetrics::from_evidence(&outcomes, traces, requests);
        let failures = classify_failures(&retrieval, &outcomes, traces);
        Self {
            run_index,
            ts_unix_ms: crate::conversation::now_millis(),
            retrieval,
            task,
            outcomes,
            failures,
        }
    }
}

/// A versioned, self-describing benchmark result. A score on its own is not a
/// result: the prompts, config, corpus, number of repeats and evidence kind
/// travel with it, and a comparison refuses to speak when they do not line up.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkRecord {
    pub schema_version: u32,
    pub benchmark_id: String,
    /// Which arm of the system was measured, in the runner's own words.
    pub arm: String,
    pub evidence_kind: EvidenceKind,
    pub ts_unix_ms: u64,
    /// The prompt-set digest this build carried, from the prompt registry.
    pub prompt_version: String,
    pub config: BenchmarkConfig,
    pub config_digest: String,
    pub corpus: BenchmarkCorpus,
    pub corpus_digest: String,
    pub repeats: Vec<BenchmarkRun>,
}

impl BenchmarkRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        benchmark_id: &str,
        arm: &str,
        evidence_kind: EvidenceKind,
        config: BenchmarkConfig,
        corpus: BenchmarkCorpus,
        repeats: Vec<BenchmarkRun>,
    ) -> Self {
        let config_digest = config.digest();
        let corpus_digest = corpus.digest();
        Self {
            schema_version: BENCHMARK_SCHEMA_VERSION,
            benchmark_id: benchmark_id.to_string(),
            arm: arm.to_string(),
            evidence_kind,
            ts_unix_ms: crate::conversation::now_millis(),
            prompt_version: crate::prompts::set_version().to_string(),
            config,
            config_digest,
            corpus,
            corpus_digest,
            repeats,
        }
    }

    /// Mean and sample variance of one metric across the repeats that actually
    /// reported it. `None` variance with one sample is the honest answer: a
    /// single observation has no spread.
    pub fn stats(&self, metric: Metric) -> MetricStats {
        let values: Vec<f64> = self
            .repeats
            .iter()
            .filter_map(|run| metric.value(run))
            .collect();
        let samples = values.len();
        let mean = (!values.is_empty()).then(|| values.iter().sum::<f64>() / samples as f64);
        let variance = if samples >= 2 {
            let mean = mean.expect("two samples have a mean");
            Some(
                values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / (samples - 1) as f64,
            )
        } else {
            None
        };
        MetricStats {
            samples,
            mean,
            variance,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct MetricStats {
    /// Repeats that reported this metric, which can be fewer than the record's
    /// repeat count when a partial run did not measure it.
    pub samples: usize,
    pub mean: Option<f64>,
    /// Sample variance (n-1), `None` with fewer than two samples.
    pub variance: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricDelta {
    pub metric: Metric,
    pub baseline: Option<f64>,
    pub candidate: Option<f64>,
    /// Candidate minus baseline, or `None` when either side did not measure it.
    pub delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Comparison {
    pub baseline_id: String,
    pub candidate_id: String,
    pub prompt_version_baseline: String,
    pub prompt_version_candidate: String,
    pub repeats_baseline: usize,
    pub repeats_candidate: usize,
    /// `true` whenever the two records are not a fair pair. The caller must
    /// show this before showing any delta.
    pub insufficient_evidence: bool,
    /// Plain-language reasons evidence was refused, so a user is told *why* a
    /// delta was withheld rather than being handed a number with a caveat.
    pub reasons: Vec<String>,
    pub deltas: Vec<MetricDelta>,
    /// Only set when evidence was sufficient, both sides are real model runs,
    /// and task completion rose. A unit test or a reference-fix run never sets
    /// this, and even a `true` here is a claim about these runs, not a statement
    /// that a feature is finished.
    pub improvement_claimed: bool,
}

/// Compare two records without ever inventing permission to compare them.
/// Deltas are always calculated and shown; `insufficient_evidence` and `reasons`
/// say whether they may be believed. That separation matters: hiding the
/// numbers would conceal what changed, while showing them without the refusal
/// would dress a prompt edit or a single lucky run up as progress.
pub fn compare(baseline: &BenchmarkRecord, candidate: &BenchmarkRecord) -> Comparison {
    let mut reasons: Vec<String> = Vec::new();
    if baseline.schema_version != candidate.schema_version {
        reasons.push(format!(
            "schema version {} vs {}",
            baseline.schema_version, candidate.schema_version
        ));
    }
    if baseline.benchmark_id != candidate.benchmark_id {
        reasons.push(format!(
            "different suites: {:?} vs {:?}",
            baseline.benchmark_id, candidate.benchmark_id
        ));
    }
    if baseline.arm != candidate.arm {
        reasons.push(format!(
            "different arms: {:?} vs {:?}",
            baseline.arm, candidate.arm
        ));
    }
    if baseline.evidence_kind != candidate.evidence_kind {
        reasons.push("one side is not the same kind of evidence".to_string());
    }
    if baseline.prompt_version != candidate.prompt_version {
        reasons.push(format!(
            "prompt version {} vs {}",
            baseline.prompt_version, candidate.prompt_version
        ));
    }
    if baseline.config_digest != candidate.config_digest {
        reasons.push(format!(
            "configuration {} vs {}",
            baseline.config_digest, candidate.config_digest
        ));
    }
    if baseline.corpus_digest != candidate.corpus_digest {
        reasons.push(format!(
            "corpus {} vs {}",
            baseline.corpus_digest, candidate.corpus_digest
        ));
    }
    if baseline.repeats.len() < MIN_REPEATS || candidate.repeats.len() < MIN_REPEATS {
        reasons.push(format!(
            "too few repeats: {} vs {} (need {MIN_REPEATS})",
            baseline.repeats.len(),
            candidate.repeats.len()
        ));
    }

    let deltas: Vec<MetricDelta> = Metric::ALL
        .iter()
        .map(|&metric| {
            let before = baseline.stats(metric).mean;
            let after = candidate.stats(metric).mean;
            MetricDelta {
                metric,
                baseline: before,
                candidate: after,
                delta: before.zip(after).map(|(before, after)| after - before),
            }
        })
        .collect();

    let insufficient_evidence = !reasons.is_empty();
    let completion_delta = deltas
        .iter()
        .find(|d| d.metric == Metric::TaskCompletion)
        .and_then(|d| d.delta)
        .is_some_and(|delta| delta > 0.0);
    let improvement_claimed = !insufficient_evidence
        && baseline.evidence_kind == EvidenceKind::Model
        && candidate.evidence_kind == EvidenceKind::Model
        && completion_delta;

    Comparison {
        baseline_id: baseline.benchmark_id.clone(),
        candidate_id: candidate.benchmark_id.clone(),
        prompt_version_baseline: baseline.prompt_version.clone(),
        prompt_version_candidate: candidate.prompt_version.clone(),
        repeats_baseline: baseline.repeats.len(),
        repeats_candidate: candidate.repeats.len(),
        insufficient_evidence,
        reasons,
        deltas,
        improvement_claimed,
    }
}

/// Where a CLI-facing report lives inside a project's `.xencode/` directory.
pub fn benchmark_report_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join("reports").join("benchmark.json")
}

/// Write a record with the stable schema to disk, creating the reports directory
/// on a real filesystem. Returns the path written, for the caller to display.
pub fn write_benchmark_report(
    xencode_dir: &Path,
    record: &BenchmarkRecord,
) -> std::io::Result<PathBuf> {
    let path = benchmark_report_path(xencode_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(record)?;
    crate::index::write_str_atomic(&path, &text)?;
    Ok(path)
}

/// Read the report back. A missing file is `None`; a present file that does not
/// parse is an error, because a silently ignored corrupt report would be worse
/// than one that fails to open.
pub fn read_benchmark_report(xencode_dir: &Path) -> std::io::Result<Option<BenchmarkRecord>> {
    let text = match std::fs::read_to_string(benchmark_report_path(xencode_dir)) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    Ok(Some(serde_json::from_str(&text)?))
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let text = serde_json::to_string(value).expect("benchmark values serialize to JSON");
    let digest = Sha256::digest(text.as_bytes());
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::FileEntry;
    use crate::seeds::{self, write_seed};
    use std::collections::HashSet;

    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("crate sits at <root>/rust/crates/<name>")
            .to_path_buf()
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("xencode-benchmark-{name}-{unique}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build an index out of the real files the checked-in gold answers name.
    /// The bodies and symbols are read from this workspace; nothing is faked.
    fn index_of_gold_answers(gold: &[crate::EvalItem]) -> crate::RetrievalIndex {
        let root = workspace_root();
        let mut index = crate::RetrievalIndex::default();
        for item in gold {
            for path in &item.expected {
                if index.files.iter().any(|file| file.path == *path) {
                    continue;
                }
                let body = std::fs::read_to_string(root.join(path))
                    .unwrap_or_else(|e| panic!("gold answer {path} is unreadable: {e}"));
                let ext = std::path::Path::new(path)
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("");
                index.files.push(FileEntry {
                    path: path.clone(),
                    language: if ext == "rs" { "rust" } else { ext }.to_string(),
                    size: body.len() as u64,
                    loc: body.lines().count() as u64,
                    ext: ext.to_string(),
                    important: false,
                    secret: false,
                    binary: false,
                });
                if ext == "rs" {
                    index
                        .symbols
                        .insert(path.clone(), crate::extract_rust_symbols(&body));
                }
            }
        }
        index
    }

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            model: None,
            provider: None,
            hardware_profile: Some("BALANCED".to_string()),
            top_k: 5,
            seed: Some(0),
            temperature: Some(0.0),
            grader: None,
        }
    }

    #[test]
    fn a_record_carries_prompt_version_shapes_repeats_and_a_stable_schema() {
        let gold = crate::default_gold();
        let index = index_of_gold_answers(&gold);
        let corpus = BenchmarkCorpus::from_checked_in(gold.clone());

        let repeats = (0..MIN_REPEATS)
            .map(|run_index| {
                // Three real retrieval passes over the checked-in gold set.
                let report = crate::evaluate(&index, &gold, 5, &HashSet::new(), false);
                BenchmarkRun::new(run_index, report, vec![], &[], &[])
            })
            .collect();
        let record = BenchmarkRecord::new(
            "xi9-foundation",
            "structural",
            EvidenceKind::HarnessValidation,
            config(),
            corpus,
            repeats,
        );

        assert_eq!(record.schema_version, BENCHMARK_SCHEMA_VERSION);
        assert_eq!(record.prompt_version, crate::prompts::set_version());
        assert_eq!(record.repeats.len(), MIN_REPEATS);
        assert_eq!(record.corpus.seed_shapes.len(), BugShape::all().len());
        assert!(
            record
                .corpus
                .seed_shapes
                .contains(&"off-by-one".to_string()),
            "the seeded shape is part of the fixed corpus"
        );
        assert_eq!(record.corpus.gold.len(), gold.len());
        assert!(
            record
                .repeats
                .iter()
                .all(|run| run.retrieval.queries == gold.len()),
            "every repeat ran the whole checked-in corpus"
        );

        let mrr = record.stats(Metric::RetrievalMrr);
        let recall = record.stats(Metric::RetrievalRecallAt1);
        let precision = record.stats(Metric::RetrievalPrecisionAt1);
        println!(
            "recorded {} probes over {} seeded shapes, {} repeats: MRR {:.3} (variance {:.6}), \
             recall@1 {:.3}, precision@1 {:.3}, prompt {}",
            gold.len(),
            record.corpus.seed_shapes.len(),
            record.repeats.len(),
            mrr.mean.unwrap_or(f64::NAN),
            mrr.variance.unwrap_or(f64::NAN),
            recall.mean.unwrap_or(f64::NAN),
            precision.mean.unwrap_or(f64::NAN),
            record.prompt_version
        );

        let dir = scratch("round-trip");
        let written = write_benchmark_report(&dir, &record).expect("write a real report");
        assert_eq!(written, benchmark_report_path(&dir));
        let read = read_benchmark_report(&dir)
            .expect("read a real report")
            .expect("the report we just wrote");
        assert_eq!(read, record, "the schema survives a disk round-trip");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_real_seeded_case_supplies_task_outcomes_and_failure_classes() {
        let dir = scratch("seeded");
        let task = write_seed(&dir, BugShape::OffByOne).expect("a real seeded repository");
        let before = task.run_grader().expect("the seeded grader runs");
        assert!(
            !before.passed,
            "the seeded case starts red:\n{}",
            before.tail
        );

        let touched = task
            .apply_reference_fix()
            .expect("the reference change applies");
        assert_eq!(touched, task.case.expected_files);
        let after = task.run_grader().expect("the fixed grader runs");
        assert!(
            after.passed,
            "the reference change makes it green:\n{}",
            after.tail
        );

        let checks = ChecksVerdict {
            ran: vec![after.command.clone()],
            failed: vec![],
            skipped: vec![],
            evidence_ref: None,
        };
        assert!(checks.all_passed());
        let outcome = TaskOutcome::from_grader(
            task.case.shape.slug(),
            &after,
            Some(checks),
            Some(true),
            true,
            false,
            !before.passed,
            true,
        );
        assert!(
            !outcome.falsely_verified(),
            "a real passing check backs the claim"
        );

        let failures = classify_failures(
            &crate::EvalReport::default(),
            std::slice::from_ref(&outcome),
            &[],
        );
        assert!(
            !failures
                .iter()
                .any(|class| class.kind == FailureKind::GraderFailure),
            "the fixed case is green; no grader failure should be recorded"
        );

        let metrics = TaskMetrics::from_evidence(std::slice::from_ref(&outcome), &[], &[]);
        println!(
            "seeded {}: grader red -> green in {} ms -> {} ms, task completion {:.3}, \
             edit success {:.3}, false-verified {:.3}",
            outcome.shape,
            before.elapsed_ms,
            after.elapsed_ms,
            metrics.task_completion.unwrap_or(f64::NAN),
            metrics.edit_success.unwrap_or(f64::NAN),
            metrics.false_verified_rate.unwrap_or(f64::NAN)
        );
        assert_eq!(metrics.task_completion, Some(1.0));
        assert_eq!(metrics.edit_success, Some(1.0));
        assert_eq!(metrics.false_verified_rate, Some(0.0));
        assert_eq!(metrics.recovery_success, Some(1.0));
        // No model ran, so resource and tool columns stay empty instead of zero.
        assert_eq!(metrics.est_cost_micros, None);
        assert_eq!(metrics.tool_call_validity, None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn pair(repeats: usize) -> (BenchmarkRecord, BenchmarkRecord) {
        let gold = crate::default_gold();
        let index = index_of_gold_answers(&gold);
        // Baseline and candidate are two runs of the same suite; what may
        // differ between them is measured under a different prompt set, not a
        // different benchmark id.
        let make = || {
            let repeats = (0..repeats)
                .map(|run_index| {
                    let report = crate::evaluate(&index, &gold, 5, &HashSet::new(), false);
                    BenchmarkRun::new(run_index, report, vec![], &[], &[])
                })
                .collect();
            BenchmarkRecord::new(
                "xi9-foundation",
                "structural",
                EvidenceKind::HarnessValidation,
                config(),
                BenchmarkCorpus::from_checked_in(gold.clone()),
                repeats,
            )
        };
        (make(), make())
    }

    #[test]
    fn comparison_refuses_mismatched_versions_and_too_few_repeats() {
        let (baseline, candidate) = pair(MIN_REPEATS);
        let fair = compare(&baseline, &candidate);
        assert!(
            !fair.insufficient_evidence,
            "matching records are comparable: {:?}",
            fair.reasons
        );
        assert!(
            !fair.improvement_claimed,
            "a reference-fix run must never claim improvement"
        );
        let mrr = fair
            .deltas
            .iter()
            .find(|delta| delta.metric == Metric::RetrievalMrr)
            .expect("MRR is always reported");
        assert_eq!(
            mrr.delta,
            Some(0.0),
            "the same arm measures the same number"
        );
        println!(
            "fair comparison: {} metrics, MRR {:.3} -> {:.3} ({:+.3}), \
             insufficient={}, improvement_claimed={}",
            fair.deltas.len(),
            mrr.baseline.unwrap(),
            mrr.candidate.unwrap(),
            mrr.delta.unwrap(),
            fair.insufficient_evidence,
            fair.improvement_claimed
        );

        let mut other_prompts = candidate.clone();
        other_prompts.prompt_version = "a-different-prompt-set".to_string();
        let refused = compare(&baseline, &other_prompts);
        assert!(refused.insufficient_evidence);
        assert!(
            refused
                .reasons
                .iter()
                .any(|reason| reason.contains("prompt version")),
            "the refusal must name the prompt change: {:?}",
            refused.reasons
        );
        assert!(!refused.improvement_claimed);

        let mut short = candidate.clone();
        short.repeats.truncate(1);
        let refused = compare(&baseline, &short);
        assert!(refused.insufficient_evidence);
        assert!(
            refused
                .reasons
                .iter()
                .any(|reason| reason.contains("too few repeats")),
            "the refusal must name the missing repeats: {:?}",
            refused.reasons
        );
        assert!(!refused.improvement_claimed);

        let mut other_corpus = candidate.clone();
        other_corpus.corpus.gold.clear();
        other_corpus.corpus_digest = other_corpus.corpus.digest();
        let refused = compare(&baseline, &other_corpus);
        assert!(refused.insufficient_evidence);
        assert!(
            refused
                .reasons
                .iter()
                .any(|reason| reason.contains("corpus")),
            "a different question set is not a progress claim: {:?}",
            refused.reasons
        );
        println!(
            "refusals: prompt mismatch {:?}, too few repeats {:?}, corpus mismatch {:?}",
            compare(&baseline, &other_prompts).reasons,
            compare(&baseline, &short).reasons,
            refused.reasons
        );
    }

    #[test]
    fn improvement_flag_is_gated_by_evidence_kind_and_sufficiency() {
        // A pure test of the comparison's contract, not a model run. It proves
        // the claim flag follows evidence kind and evidence sufficiency; it
        // does not pretend a model was measured here, and no product claim is
        // made by this test.
        let (baseline, candidate) = pair(MIN_REPEATS);
        let mut model_baseline = baseline.clone();
        model_baseline.evidence_kind = EvidenceKind::Model;
        let mut model_candidate = candidate.clone();
        model_candidate.evidence_kind = EvidenceKind::Model;
        for run in &mut model_candidate.repeats {
            run.task.task_completion = Some(0.9);
        }
        for run in &mut model_baseline.repeats {
            run.task.task_completion = Some(0.7);
        }
        let comparison = compare(&model_baseline, &model_candidate);
        assert!(!comparison.insufficient_evidence);
        assert!(comparison.improvement_claimed);
        println!(
            "pure flag-contract check (no model ran): completion {:.3} -> {:.3}, \
             improvement_flag={}",
            comparison.deltas[0].baseline.unwrap(),
            comparison.deltas[0].candidate.unwrap(),
            comparison.improvement_claimed
        );

        // The same numbers under harness validation are still not a claim.
        let mut harness = model_candidate;
        harness.evidence_kind = EvidenceKind::HarnessValidation;
        assert!(!compare(&baseline, &harness).improvement_claimed);
    }

    #[test]
    fn unused_seeded_shapes_are_the_real_ten_from_the_harness() {
        // Guards against the corpus quietly losing the task half of the suite.
        let corpus = BenchmarkCorpus::from_checked_in(crate::default_gold());
        for shape in seeds::BugShape::all() {
            assert!(corpus.seed_shapes.contains(&shape.slug().to_string()));
        }
        println!(
            "benchmark corpus names {} seeded shapes",
            corpus.seed_shapes.len()
        );
    }
}
