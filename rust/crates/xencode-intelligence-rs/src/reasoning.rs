//! Adaptive reasoning controller (XI-4), computed without a model call.
//!
//! Given one already-finished turn — the facts the agent loop wrote to
//! `.xencode/cache/turns.jsonl` as a [`TurnTrace`] — plus the task shape and the
//! caller's constraints, this module decides *how hard the next turn should
//! try*: a stage, a hard cap on tool rounds, a verification intensity, and a
//! recovery strategy. It is a pure function of those inputs: no provider is
//! dialled, no file is written and no clock is read. The same trace always
//! yields the same [`ReasoningPlan`].
//!
//! Two rules are absolute here:
//!
//! - **Observed events only.** The only signals are real recorded facts —
//!   failed tool outcomes, no-change notes, failed checks, the absence of
//!   retrieved files and the DT-1 failed-edit count. Anything a model *said*
//!   about its own confidence or reasoning is narrative, not evidence, and is
//!   never read. There is no confidence field in [`TurnTrace`] by design.
//! - **Budgets are hard caps.** `max_tool_rounds` can never rise above the
//!   caller's measured reasoning budget, nor above [`HARD_MAX_TOOL_ROUNDS`],
//!   whatever the stage asks for. An unmeasured budget defaults conservatively
//!   instead of being invented.
//!
//! Escalation is a *decision for a caller to act on*, not a new routing path.
//! The one-turn deep-think override already exists in the CLI; this plan only
//! says whether that existing path is warranted and permitted. Cloud
//! escalation needs the same explicit permission as any other off-machine
//! model: `allow_cloud_models` mapped into [`Constraints`], with local-only and
//! no-network each able to veto it on their own.
//!
//! Nothing here changes production agent-loop behaviour by itself. A caller
//! opts in through [`crate::Coordinator::decide_with_reasoning`]; the ordinary
//! [`crate::Coordinator::decide`] path leaves the plan unset.

use crate::types::{sources, Constraints, DecisionEvidence, Measurement, ReasoningBudget};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use xencode_context_rs::{
    count_failed_edits, escalation_recommended, TaskShape, ToolTrace, TurnTrace,
    ESCALATION_FAILED_EDIT_THRESHOLD,
};

/// Conservative round allowance when the caller supplied no reasoning budget.
/// It is a default for an unmeasured value, never a guess presented as one.
pub const DEFAULT_MAX_TOOL_ROUNDS: u32 = 8;

/// Absolute ceiling a plan may never exceed, even for escalation.
pub const HARD_MAX_TOOL_ROUNDS: u32 = 32;

const FAST_STAGE_ROUNDS: u32 = 4;
const NORMAL_STAGE_ROUNDS: u32 = 8;
const ESCALATE_STAGE_ROUNDS: u32 = 16;

/// How much computation the next turn should spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningStage {
    /// Nothing went wrong: keep the cheap default path.
    Fast,
    /// A specific, recoverable problem was observed.
    Normal,
    /// Repeated failed edits: hand the next turn to the existing deeper model.
    Escalate,
}

impl ReasoningStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Normal => "normal",
            Self::Escalate => "escalate",
        }
    }
}

/// How thoroughly the outcome of the next turn must be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationIntensity {
    Light,
    Standard,
    Thorough,
}

impl VerificationIntensity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Standard => "standard",
            Self::Thorough => "thorough",
        }
    }
}

/// The first thing the next turn should do about the observed problem.
///
/// Each variant names a deterministic rule; none of them calls a model or
/// edits anything. They are instructions a caller can inspect and choose to
/// follow, ranked by specificity below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryStrategy {
    /// No failure was observed; proceed as the agent loop normally would.
    Proceed,
    /// A failed tool call reported missing or invalid arguments: call it again
    /// with a correctly formed invocation.
    CorrectInvocation,
    /// The same call failed more than once: do not repeat it unchanged.
    StopRepeating,
    /// No repository file was retrieved for a bugfix: ask for more context,
    /// using the XI-3 context-depth and evidence-order plan concept.
    RequestMoreContext,
    /// A build or lint check failed: read its structured diagnostics before
    /// editing again.
    StructuredDiagnostics,
    /// A regression check failed: look at the affected dependencies (XI-2).
    AffectedDependencies,
    /// Repeated failed edits with no progress: use the existing one-turn
    /// deep-think escalation path.
    DeepThink,
}

impl RecoveryStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proceed => "proceed",
            Self::CorrectInvocation => "correct_invocation",
            Self::StopRepeating => "stop_repeating",
            Self::RequestMoreContext => "request_more_context",
            Self::StructuredDiagnostics => "structured_diagnostics",
            Self::AffectedDependencies => "affected_dependencies",
            Self::DeepThink => "deep_think",
        }
    }
}

/// The whole adaptive reasoning decision for one turn. Pure data: no function
/// pointers, no I/O handles and nothing that could make two equal inputs
/// disagree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningPlan {
    pub stage: ReasoningStage,
    /// Hard cap on tool rounds for the next turn.
    pub max_tool_rounds: u32,
    pub verification: VerificationIntensity,
    pub recovery: RecoveryStrategy,
    /// Whether the existing deep-think escalation path is warranted *and*
    /// permitted. `false` can mean "no observed event" or "a veto"; the
    /// rationale says which.
    pub escalation_eligible: bool,
    /// Whether that escalation, if used, would route off-machine. Kept so a
    /// caller cannot mistake a permitted local escalation for a cloud route.
    pub escalation_is_cloud: bool,
    /// Plain-English reason for each choice, with the observed event behind it.
    /// Never empty.
    pub rationale: Vec<DecisionEvidence>,
}

/// Inputs to the controller. Everything is supplied by the caller: the trace
/// is read from disk elsewhere, the constraints come from the request, and
/// the deeper-model facts are classified from config and the provider table.
#[derive(Debug, Clone)]
pub struct ReasoningInput<'a> {
    /// The finished turn this decision reacts to. Borrowed: the controller
    /// never reads a file or writes one.
    pub trace: &'a TurnTrace,
    pub shape: TaskShape,
    pub constraints: &'a Constraints,
    /// The caller's measured reasoning budget, if any.
    pub budget: Measurement<ReasoningBudget>,
    /// Whether a deeper model is configured at all (`deep_think_model` set).
    pub deep_think_available: bool,
    /// Whether that configured deeper model routes off-machine, as classified
    /// by the caller from config and the provider's egress table.
    pub deep_think_is_cloud: bool,
}

impl<'a> ReasoningInput<'a> {
    /// The most conservative input: no budget was measured, no deeper model is
    /// available, and the request's own constraints apply unchanged.
    pub fn new(trace: &'a TurnTrace, shape: TaskShape, constraints: &'a Constraints) -> Self {
        Self {
            trace,
            shape,
            constraints,
            budget: Measurement::Unknown,
            deep_think_available: false,
            deep_think_is_cloud: false,
        }
    }
}

/// Decide the reasoning plan for one turn from a borrowed trace.
pub fn decide(input: &ReasoningInput<'_>) -> ReasoningPlan {
    let mut rationale = Vec::new();
    let signals = observe(input);

    // The observed facts are written down first, so every later choice can be
    // checked against what the loop actually recorded.
    let failed_edits = failed_edit_detail(&signals);
    rationale.push(DecisionEvidence::new(
        sources::OBSERVED_TURN_TRACE,
        format!(
            "the finished turn recorded {} rounds, {} tool calls, {} failed edit attempts (threshold {}){}; the provider {}",
            input.trace.rounds,
            input.trace.tools.len(),
            signals.failed_edit_count,
            ESCALATION_FAILED_EDIT_THRESHOLD,
            failed_edits,
            if signals.provider_failed {
                "ended in error"
            } else {
                "finished normally"
            }
        ),
    ));
    rationale.push(DecisionEvidence::new(
        sources::REASONING_RULES,
        "model-verbalized confidence and narrative text are never inputs: only recorded tool outcomes, check results and trace fields decide this plan",
    ));

    if let Some(detail) = &signals.repeated_failure {
        rationale.push(DecisionEvidence::new(
            sources::OBSERVED_TURN_TRACE,
            format!("the same failed call was repeated: {detail}; repeating it unchanged cannot make progress"),
        ));
    }
    if signals.missing_args {
        rationale.push(DecisionEvidence::new(
            sources::OBSERVED_TURN_TRACE,
            "a failed tool call reported missing or invalid arguments in its redacted output",
        ));
    }
    if !signals.compile_checks.is_empty() {
        rationale.push(DecisionEvidence::new(
            sources::OBSERVED_TURN_TRACE,
            format!(
                "build/lint checks failed: {}; their structured diagnostics come before another edit",
                signals.compile_checks.join(", ")
            ),
        ));
    }
    if !signals.regression_checks.is_empty() {
        rationale.push(DecisionEvidence::new(
            sources::OBSERVED_TURN_TRACE,
            format!(
                "regression checks failed: {}; look at the affected dependencies (XI-2 impact) before editing again",
                signals.regression_checks.join(", ")
            ),
        ));
    }
    if signals.no_repo_evidence {
        rationale.push(DecisionEvidence::new(
            sources::OBSERVED_TURN_TRACE,
            "no repository file was retrieved this turn; request more context using the XI-3 retrieval-depth and evidence-order concept",
        ));
    }

    let (escalation_eligible, refusal) = escalation_permission(input, &signals);
    rationale.push(DecisionEvidence::new(
        sources::OBSERVED_TURN_TRACE,
        format!(
            "escalation event from the DT-1 failed-edit signal: {} (recomputed from real tool outcomes and no-change notes, threshold {})",
            yes_no(signals.escalation_event),
            ESCALATION_FAILED_EDIT_THRESHOLD
        ),
    ));
    if let Some(reason) = refusal {
        rationale.push(DecisionEvidence::new(sources::CONSTRAINTS, reason));
    }
    if escalation_eligible {
        rationale.push(DecisionEvidence::new(
            sources::CONSTRAINTS,
            "escalation is warranted and permitted; the caller's existing one-turn deep-think override is the way to act on it, and this crate implements no new routing",
        ));
    }

    // Pick the most specific recovery that the observed facts support.
    // Escalation wins because a second round of failed edits means the local
    // model is not making progress; otherwise root causes beat generic retries.
    let (stage, recovery) = if escalation_eligible {
        (ReasoningStage::Escalate, RecoveryStrategy::DeepThink)
    } else if !signals.regression_checks.is_empty() {
        (
            ReasoningStage::Normal,
            RecoveryStrategy::AffectedDependencies,
        )
    } else if !signals.compile_checks.is_empty() {
        (
            ReasoningStage::Normal,
            RecoveryStrategy::StructuredDiagnostics,
        )
    } else if signals.missing_args {
        (ReasoningStage::Normal, RecoveryStrategy::CorrectInvocation)
    } else if signals.repeated_failure.is_some() || signals.escalation_event {
        // An observed escalation we could not act on still must not repeat the
        // same failing call under a cheaper model.
        (ReasoningStage::Normal, RecoveryStrategy::StopRepeating)
    } else if input.shape == TaskShape::Bugfix && signals.no_repo_evidence {
        (ReasoningStage::Normal, RecoveryStrategy::RequestMoreContext)
    } else {
        (ReasoningStage::Fast, RecoveryStrategy::Proceed)
    };

    // Budgets are hard caps. The stage asks for an allowance; the measured
    // budget and the absolute ceiling decide what it actually gets.
    let stage_cap = match stage {
        ReasoningStage::Fast => FAST_STAGE_ROUNDS,
        ReasoningStage::Normal => NORMAL_STAGE_ROUNDS,
        ReasoningStage::Escalate => ESCALATE_STAGE_ROUNDS,
    };
    let budget_cap = match input.budget {
        Measurement::Known(budget) => budget.max_rounds.min(HARD_MAX_TOOL_ROUNDS),
        Measurement::Unknown => DEFAULT_MAX_TOOL_ROUNDS.min(HARD_MAX_TOOL_ROUNDS),
    };
    let max_tool_rounds = stage_cap.min(budget_cap);
    rationale.push(DecisionEvidence::new(
        sources::MEASUREMENTS,
        format!(
            "max_tool_rounds is {max_tool_rounds}: stage {} asks for {stage_cap}, bounded by {} of {}",
            stage.as_str(),
            match input.budget {
                Measurement::Known(_) => "the measured reasoning budget of".to_string(),
                Measurement::Unknown => format!("the conservative default of {DEFAULT_MAX_TOOL_ROUNDS} for an unmeasured budget, itself bounded by"),
            },
            budget_cap
        ),
    ));

    let verification = if !signals.regression_checks.is_empty() || escalation_eligible {
        VerificationIntensity::Thorough
    } else if input.constraints.require_verification {
        VerificationIntensity::Standard
    } else {
        VerificationIntensity::Light
    };
    rationale.push(DecisionEvidence::new(
        sources::CONSTRAINTS,
        format!(
            "verification intensity is {}: {}",
            verification.as_str(),
            if verification == VerificationIntensity::Light {
                "the caller waived verification for this turn; no pass is claimed on that basis"
            } else {
                "the observed failures and the turn's verification requirement set it"
            }
        ),
    ));

    ReasoningPlan {
        stage,
        max_tool_rounds,
        verification,
        recovery,
        escalation_eligible,
        escalation_is_cloud: input.deep_think_is_cloud,
        rationale,
    }
}

/// Convenience for callers that only have the default inputs.
pub fn decide_from_trace(
    trace: &TurnTrace,
    shape: TaskShape,
    constraints: &Constraints,
) -> ReasoningPlan {
    decide(&ReasoningInput::new(trace, shape, constraints))
}

/// The observed signals, all derived from the recorded trace. Kept as one
/// struct so the decision and its evidence cannot disagree about a fact.
struct Signals {
    failed_edit_count: usize,
    escalation_event: bool,
    repeated_failure: Option<String>,
    missing_args: bool,
    compile_checks: Vec<String>,
    regression_checks: Vec<String>,
    no_repo_evidence: bool,
    provider_failed: bool,
    failed_edit_tools: Vec<String>,
    no_change_notes: usize,
}

fn observe(input: &ReasoningInput<'_>) -> Signals {
    let trace = input.trace;
    // Recompute from the real tool rows and no-change notes, then take the
    // larger of that and the writer's summary. An old row with a stale summary
    // cannot hide the failures still visible in its own tool list.
    let counted = count_failed_edits(&trace.tools, &trace.no_change_notes);
    let failed_edit_count = counted.max(trace.failed_edit_count);
    let escalation_event = trace.escalation_recommended || escalation_recommended(counted);
    let failed_edit_tools = trace
        .tools
        .iter()
        .filter(|tool| {
            (tool.name == "edit_file" || tool.name == "write_file") && tool.outcome == "failed"
        })
        .map(|tool| tool.name.clone())
        .collect();

    let (compile_checks, regression_checks) = match &trace.checks {
        Some(checks) => (
            checks
                .failed
                .iter()
                .filter(|name| is_compile_check(name))
                .cloned()
                .collect(),
            checks
                .failed
                .iter()
                .filter(|name| is_regression_check(name))
                .cloned()
                .collect(),
        ),
        None => (Vec::new(), Vec::new()),
    };

    Signals {
        failed_edit_count,
        escalation_event,
        repeated_failure: repeated_failed_signature(&trace.tools),
        missing_args: trace.tools.iter().any(|tool| {
            tool.outcome == "failed" && mentions_missing_argument(tool.tail.as_deref())
        }),
        compile_checks,
        regression_checks,
        no_repo_evidence: trace.retrieved_files.is_empty(),
        provider_failed: trace.failed,
        failed_edit_tools,
        no_change_notes: trace.no_change_notes.len(),
    }
}

/// A short, factual account of what made up the failed-edit count.
fn failed_edit_detail(signals: &Signals) -> String {
    if signals.failed_edit_count == 0 {
        return String::new();
    }
    format!(
        " ({} failed edit tool calls [{}], {} no-change notes)",
        signals.failed_edit_tools.len(),
        if signals.failed_edit_tools.is_empty() {
            "none".to_string()
        } else {
            signals.failed_edit_tools.join(", ")
        },
        signals.no_change_notes
    )
}

/// Whether the configured escalation may be used. The refusal reason, when
/// there is one, is returned so the rationale can name the exact veto.
fn escalation_permission(input: &ReasoningInput<'_>, signals: &Signals) -> (bool, Option<String>) {
    if !signals.escalation_event {
        return (false, None);
    }
    if !input.deep_think_available {
        return (
            false,
            Some(
                "escalation refused: no deeper model is configured (deep_think_model is unset); nothing was invented"
                    .to_string(),
            ),
        );
    }
    if let Measurement::Known(budget) = input.budget {
        if !budget.allow_deep_think {
            return (
                false,
                Some(
                    "escalation refused: the caller's measured reasoning budget does not allow deep think"
                        .to_string(),
                ),
            );
        }
    }
    if input.deep_think_is_cloud && !input.constraints.cloud_permitted() {
        return (
            false,
            Some(
                "escalation refused: the deeper model routes off-machine and cloud was not explicitly permitted (allow_cloud_models is off, or local-only/no-network vetoes it)"
                    .to_string(),
            ),
        );
    }
    (true, None)
}

/// Find a failed call signature seen at least twice. The signature is the
/// tool name plus the stored argument preview, so only genuinely identical
/// calls count; a changed argument is a different attempt, not a repeat.
fn repeated_failed_signature(tools: &[ToolTrace]) -> Option<String> {
    let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
    for tool in tools.iter().filter(|tool| tool.outcome == "failed") {
        let key = (
            tool.name.clone(),
            tool.arguments.clone().unwrap_or_default(),
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .find(|(_, count)| *count >= 2)
        .map(|((name, arguments), count)| {
            if arguments.is_empty() {
                format!("{name} failed {count} times")
            } else {
                format!("{name} failed {count} times with the same arguments")
            }
        })
}

/// Whether a failed call's redacted output says its arguments were missing or
/// invalid. Only the trace's own tail is read; no model text is consulted.
fn mentions_missing_argument(tail: Option<&str>) -> bool {
    let Some(tail) = tail else {
        return false;
    };
    let text = tail.to_ascii_lowercase();
    (text.contains("missing") && text.contains("arg"))
        || text.contains("required argument")
        || text.contains("expected argument")
        || text.contains("invalid argument")
        || text.contains("usage:")
}

fn is_compile_check(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    ["build", "check", "compile", "clippy", "rustc", "fmt"]
        .iter()
        .any(|part| name.contains(part))
}

fn is_regression_check(name: &str) -> bool {
    name.to_ascii_lowercase().contains("test")
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}
