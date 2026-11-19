//! Intelligence observability (XI-8): inspect the decisions the coordinator
//! really made, without building a second dashboard.
//!
//! This module is a **pure loader plus a string renderer**. It reads the JSON
//! the [`crate::Coordinator`] already writes under an artifact root and turns
//! it into lines that fit the presentation pattern of the TUI's `/trace`
//! renderer: a header of counts, then one block per decision. It never runs a
//! model, never measures anything itself and never invents a value. A field
//! that was not measured reads as `unknown`; a store this slice does not read
//! reads as `not recorded in this session`.
//!
//! # What is authoritative
//!
//! Every figure comes from a recorded [`IntelligenceDecision`] or
//! [`IntelligenceOutcome`]:
//!
//! - selected model and the capability-view evidence that explains it — from
//!   the decision's `selected_model` and `rationale`;
//! - context budget and quoted retrieval evidence — from `context_budget` and
//!   the rationale's retrieval rows;
//! - reasoning stage, round cap, recovery rule and escalation — from the
//!   opt-in `reasoning_plan` (absent on the ordinary decide path);
//! - execution and verification — from the outcome, including the checks the
//!   caller actually named. A `VerificationStatus::NotRun` outcome is rendered
//!   as **unknown**, never as a pass, even when its `checks` vector is not
//!   empty. A decision with no readable outcome is rendered with unknown
//!   verification too.
//!
//! There is deliberately **no confidence field, no percentage and no
//! self-report treated as truth** anywhere in the output. Latency, resource
//! use, adaptation observations and policy changes have no measured field in
//! these two records, so they render as not recorded; adjacent stores
//! (`policies/`, `repairs/`, `diagnoses/`) are separate opt-in files and this
//! loader does not pretend to have read them.
//!
//! # Integration point (documented, not wired in this slice)
//!
//! The dependency graph is unchanged: neither `xencode-tui-rs` nor
//! `xencode-cli` depends on this crate, and the dependency-boundary tests
//! enforce that. This slice therefore only supplies the lines; it does not add
//! a view, a command or a navigation entry.
//!
//! The existing `/trace` renderer is
//! `rust/crates/xencode-tui-rs/src/app.rs::trace_report` (around line 11160).
//! It already builds a `Vec<String>` named `out` and returns it. A future slice
//! that adds the dependency can append, immediately before that return and
//! only when a coordinator artifact root is available:
//!
//! ```text
//! if let Ok(records) = xencode_intelligence_rs::load_decisions(&artifact_root) {
//!     if !records.is_empty() {
//!         out.extend(xencode_intelligence_rs::intelligence_report(&records));
//!     }
//! }
//! ```
//!
//! The CLI's trace-style text output can append the same vector through its
//! existing string presentation path. Until such a slice exists, a normal
//! session shows nothing new — which is honest, because the coordinator is
//! opt-in and inert in production paths and writes no decisions by default.

use crate::coordinator::Coordinator;
use crate::types::{CapabilityView, IntelligenceDecision, IntelligenceOutcome, Measurement};
use crate::VerificationStatus;
use std::fmt;
use std::path::{Path, PathBuf};

/// Plain-English marker for a value that was never measured.
pub const UNKNOWN: &str = "unknown";
/// Marker for an observable that has no field in the recorded decision/outcome.
pub const NOT_RECORDED: &str = "not recorded here";
/// Marker for an observable held in a separate, unread store.
pub const NOT_RECORDED_SESSION: &str = "not recorded in this session";

/// Records loaded from disk together with the honest notes from the load.
///
/// A note never changes a recorded value. It says what was skipped (an
/// unreadable decision file) or what was missing (an outcome file), so the
/// reader can tell a real "not run" from a file that could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionLoad {
    /// Decisions paired with their recorded outcome, or with an explicit
    /// not-run outcome when no outcome could be read for that decision.
    pub records: Vec<(IntelligenceDecision, IntelligenceOutcome)>,
    /// One line per skipped, orphaned or inconsistent record.
    pub notes: Vec<String>,
}

/// Why a load could not enumerate the artifact root at all.
///
/// Individual unreadable files are *not* errors: they are noted in
/// [`DecisionLoad::notes`] and skipped. Only a failure to list the decisions
/// directory itself reaches here.
#[derive(Debug)]
pub enum ObservabilityError {
    ReadDir {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl fmt::Display for ObservabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadDir { path, source } => {
                write!(f, "could not list {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for ObservabilityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReadDir { source, .. } => Some(source),
        }
    }
}

/// Load every decision under `root` together with the outcome recorded for it.
///
/// `root` is the coordinator's artifact root: decisions live in `root/decisions`
/// and outcomes in `root/outcomes`, exactly as [`Coordinator::decisions_dir`]
/// and [`Coordinator::outcomes_dir`] define them. A reader needs no capability
/// view — the read methods use only the root — so an empty one is used here.
///
/// Files are enumerated as their stable hex ids, sorted, so the returned order
/// is deterministic on any filesystem. An unreadable decision is skipped and
/// noted: no decision is invented to fill its place. A decision whose outcome
/// is missing or unreadable is kept with an explicit `NotRun` outcome whose
/// summary says no outcome was recorded, so it renders with unknown
/// verification instead of a fabricated pass.
///
/// A root with no decisions directory is an empty load, not an error: that is
/// the normal, opt-in-inert state of a production session.
pub fn load_decisions(
    root: &Path,
) -> Result<Vec<(IntelligenceDecision, IntelligenceOutcome)>, ObservabilityError> {
    Ok(load_decisions_with_notes(root)?.records)
}

/// The same load, with the notes about skipped and orphaned files.
pub fn load_decisions_with_notes(root: &Path) -> Result<DecisionLoad, ObservabilityError> {
    let coordinator = Coordinator::new(CapabilityView::default(), root);
    let decisions_dir = coordinator.decisions_dir();
    let entries = match std::fs::read_dir(&decisions_dir) {
        Ok(entries) => entries,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Ok(DecisionLoad {
                records: Vec::new(),
                notes: Vec::new(),
            });
        }
        Err(source) => {
            return Err(ObservabilityError::ReadDir {
                path: decisions_dir,
                source,
            });
        }
    };

    let mut ids = Vec::new();
    let mut notes = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(source) => {
                notes.push(format!(
                    "skipped a directory entry in {}: {source}",
                    decisions_dir.display()
                ));
                continue;
            }
        };
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(id) = path.file_stem().and_then(|stem| stem.to_str()) else {
            notes.push(format!(
                "skipped {} because its file name is not UTF-8",
                path.display()
            ));
            continue;
        };
        ids.push(id.to_string());
    }
    ids.sort();
    ids.dedup();

    let mut records = Vec::new();
    for id in ids {
        let decision = match coordinator.read_decision(&id) {
            Ok(decision) => decision,
            Err(error) => {
                notes.push(format!(
                    "skipped decision {id}: the recorded decision could not be read ({error})"
                ));
                continue;
            }
        };
        match coordinator.read_outcome(&id) {
            Ok(outcome) => {
                if outcome.decision_id != id {
                    notes.push(format!(
                        "decision {id} has an outcome that names {}; both are shown as recorded",
                        outcome.decision_id
                    ));
                }
                records.push((decision, outcome));
            }
            Err(error) => {
                notes.push(format!(
                    "decision {id} has no readable outcome ({error}); verification renders as {UNKNOWN}"
                ));
                records.push((decision, no_outcome(&id)));
            }
        }
    }

    Ok(DecisionLoad { records, notes })
}

/// An explicit "nothing was recorded" outcome for an orphaned decision.
///
/// This is not a claim about what happened: both statuses are `NotRun` and the
/// text says no outcome was recorded. It exists only so the decision can still
/// be rendered with unknown verification rather than being silently dropped
/// or given an invented pass.
fn no_outcome(id: &str) -> IntelligenceOutcome {
    use crate::types::{ExecutionRecord, ExecutionStatus, VerificationRecord};

    IntelligenceOutcome {
        decision_id: id.to_string(),
        execution: ExecutionRecord {
            status: ExecutionStatus::NotRun,
            summary: "no outcome was recorded for this decision".to_string(),
        },
        verification: VerificationRecord {
            status: VerificationStatus::NotRun,
            checks: Vec::new(),
            detail: "no outcome was recorded for this decision".to_string(),
        },
    }
}

/// Render the intelligence inspector block for already-loaded records.
///
/// Pure: it reads only the records it is given and returns lines in the same
/// order, so the same records always produce the same block. The caller places
/// the block after the existing `/trace` lines; see the module docs for the
/// exact integration point. This function never reads a file, calls a model or
/// produces a confidence figure.
pub fn intelligence_report(records: &[(IntelligenceDecision, IntelligenceOutcome)]) -> Vec<String> {
    let mut out = Vec::new();
    let selected = records
        .iter()
        .filter(|(decision, _)| decision.selected_model.is_some())
        .count();
    let passed = records
        .iter()
        .filter(|(_, outcome)| outcome.verification.status == VerificationStatus::Passed)
        .count();
    let unknown = records
        .iter()
        .filter(|(_, outcome)| outcome.verification.status == VerificationStatus::NotRun)
        .count();
    let failed = records.len() - passed - unknown;

    out.push(format!(
        "INTELLIGENCE INSPECTOR — {} · {} · {passed} verification passed · {unknown} verification unknown · {failed} verification failed",
        count_words(records.len(), "decision"),
        count_words(selected, "selected model"),
    ));
    if records.is_empty() {
        out.push(
            "No intelligence decisions recorded. The coordinator is opt-in and inert in production paths, so an ordinary session writes none; this is expected, not a failure."
                .to_string(),
        );
        return out;
    }
    out.push(
        "Every line below is read from recorded decisions and outcomes. An unmeasured value reads as unknown; nothing is estimated."
            .to_string(),
    );

    for (decision, outcome) in records {
        out.push(format!(
            "#{} · {} work · {} · {} · verification {}",
            outcome.decision_id,
            decision.task_shape.as_str(),
            match &decision.selected_model {
                Some(model) => format!("model {model}"),
                None => "model none selected (no candidate met the constraints)".to_string(),
            },
            execution_line(&outcome.execution),
            verification_word(&outcome.verification.status),
        ));

        let model_evidence: Vec<&str> = decision
            .rationale
            .iter()
            .filter(|item| item.source == crate::types::sources::CAPABILITY_VIEW)
            .map(|item| item.fact.as_str())
            .collect();
        if model_evidence.is_empty() {
            out.push(format!("   why this model: {NOT_RECORDED}"));
        } else {
            for fact in model_evidence {
                out.push(format!("   why this model: {fact}"));
            }
        }

        out.push(format!(
            "   context budget: {}",
            match &decision.context_budget {
                Measurement::Known(budget) => budget.to_string(),
                Measurement::Unknown =>
                    format!("{UNKNOWN} (no context budget measured for this decision)"),
            }
        ));
        let retrieval: Vec<&str> = decision
            .rationale
            .iter()
            .filter(|item| item.source == crate::types::sources::RETRIEVAL_EVALUATION)
            .map(|item| item.fact.as_str())
            .collect();
        if retrieval.is_empty() {
            out.push(format!("   retrieval evidence: {NOT_RECORDED}"));
        } else {
            for fact in retrieval {
                out.push(format!("   retrieval evidence: {fact}"));
            }
        }

        match &decision.reasoning_plan {
            Some(plan) => {
                out.push(format!(
                    "   reasoning stage: {} · max tool rounds: {} · recovery rule: {} (planned response; attempts {NOT_RECORDED})",
                    plan.stage.as_str(),
                    plan.max_tool_rounds,
                    plan.recovery.as_str(),
                ));
                out.push(format!(
                    "   escalation: eligible {} · routes off-machine {}",
                    yes_no(plan.escalation_eligible),
                    yes_no(plan.escalation_is_cloud)
                ));
                for evidence in &plan.rationale {
                    out.push(format!("   reasoning evidence: {}", evidence.fact));
                }
            }
            None => out.push(format!(
                "   reasoning stage: {NOT_RECORDED} · recovery attempts: {NOT_RECORDED}"
            )),
        }

        out.push(format!(
            "   adaptation observations: {NOT_RECORDED_SESSION}"
        ));
        out.push(format!("   policy changes: {NOT_RECORDED_SESSION}"));

        let checks = match &outcome.verification.status {
            VerificationStatus::NotRun => "none recorded (verification was not run)".to_string(),
            _ if outcome.verification.checks.is_empty() => "none named by the caller".to_string(),
            _ => outcome.verification.checks.join(", "),
        };
        out.push(format!("   checks that really ran: {checks}"));
        if !outcome.verification.detail.is_empty() {
            out.push(format!(
                "   verification detail: {}",
                outcome.verification.detail
            ));
        }
        out.push(format!("   latency / resource: {NOT_RECORDED}"));
    }
    out
}

/// Render the inspector block and append the loader's honest notes.
///
/// Notes go at the end so the per-decision blocks stay identical to
/// [`intelligence_report`].
pub fn intelligence_report_with_notes(load: &DecisionLoad) -> Vec<String> {
    let mut out = intelligence_report(&load.records);
    if !load.notes.is_empty() {
        out.push("record-read notes:".to_string());
        for note in &load.notes {
            out.push(format!("   - {note}"));
        }
    }
    out
}

fn verification_word(status: &VerificationStatus) -> &'static str {
    match status {
        VerificationStatus::Passed => "passed",
        VerificationStatus::Failed { .. } => "failed",
        VerificationStatus::NotRun => "unknown (not run)",
    }
}

fn execution_line(execution: &crate::types::ExecutionRecord) -> String {
    match &execution.status {
        crate::types::ExecutionStatus::NotRun => {
            format!("execution not run: {}", execution.summary)
        }
        crate::types::ExecutionStatus::Completed => {
            format!("execution completed: {}", execution.summary)
        }
        crate::types::ExecutionStatus::Failed { reason } => {
            format!("execution failed ({reason}): {}", execution.summary)
        }
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn count_words(count: usize, word: &str) -> String {
    if count == 1 {
        format!("1 {word}")
    } else {
        format!("{count} {word}s")
    }
}
