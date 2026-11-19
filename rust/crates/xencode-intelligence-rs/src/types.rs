//! The request, decision and outcome values the coordinator passes around.
//!
//! Every value here is inspectable: a caller can print a decision and see which
//! measurement or constraint produced each part of it. Nothing in this module
//! measures hardware, classifies a model id, retrieves files or calls a model.
//! Those jobs stay in the crates that already own them.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use xencode_config_rs::XencodeConfig;
use xencode_context_rs::{ContextCaps, EvalItem, EvalReport, TaskShape};

/// A fact that may or may not have been measured.
///
/// `Unknown` is a first-class answer. A coordinator that guessed a value here
/// would be presenting a preference as a measurement, which is the one thing
/// this slice exists to refuse.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Measurement<T> {
    Known(T),
    #[default]
    Unknown,
}

impl<T> Measurement<T> {
    /// Carry a measured value into another type without turning it into a
    /// default when it is missing.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Measurement<U> {
        match self {
            Self::Known(value) => Measurement::Known(f(value)),
            Self::Unknown => Measurement::Unknown,
        }
    }

    /// Borrow the measured value, if there is one.
    pub fn as_ref(&self) -> Measurement<&T> {
        match self {
            Self::Known(value) => Measurement::Known(value),
            Self::Unknown => Measurement::Unknown,
        }
    }
}

/// A serializable picture of the context budget derived elsewhere.
///
/// `ContextCaps` is the value the budgeter in `xencode-context-rs` computes. It
/// is not serializable, so decisions carry this plain copy of its two numbers.
/// The conversion never recomputes a budget; it only records what was measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextBudget {
    /// How many file bodies the budget allows.
    pub top_k: usize,
    /// Characters kept from each body.
    pub content_cap_chars: usize,
}

impl From<ContextCaps> for ContextBudget {
    fn from(caps: ContextCaps) -> Self {
        Self {
            top_k: caps.top_k,
            content_cap_chars: caps.content_cap_chars,
        }
    }
}

impl fmt::Display for ContextBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} files at {} characters each",
            self.top_k, self.content_cap_chars
        )
    }
}

/// A serializable picture of the reasoning budget supplied by the caller.
///
/// The coordinator does not decide how many rounds a turn gets or whether a
/// deeper model may be used. Those are product policy, already owned elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningBudget {
    pub max_rounds: u32,
    pub allow_deep_think: bool,
}

/// The measurements a caller hands to the coordinator.
///
/// Both start unknown. Supplying one is a claim by the caller that it was
/// really measured; leaving it out is not a request for the coordinator to
/// invent a number.
#[derive(Debug, Clone, Default)]
pub struct Measurements {
    pub context: Measurement<ContextCaps>,
    pub reasoning: Measurement<ReasoningBudget>,
}

/// An opaque identifier for a model, as the rest of xencode spells it.
///
/// There is no shared model-id type in the workspace yet, so coordination
/// keeps this one as a thin newtype. It is never parsed for egress here: that
/// classification belongs to the provider crate and arrives in [`ModelEgress`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ModelId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for ModelId {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

impl From<String> for ModelId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

/// Where a model would send a prompt, as reported by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelEgress {
    Local,
    Cloud,
}

impl ModelEgress {
    /// Plain English for a decision's rationale.
    pub fn label(self) -> &'static str {
        match self {
            Self::Local => "on this machine",
            Self::Cloud => "off-machine",
        }
    }
}

/// One candidate model and what is actually known about it.
///
/// `available` is the caller's own judgement that the model can be used now;
/// the coordinator never probes a provider to find out. An unavailable or
/// unmeasured candidate is refused, not selected on the chance that it works.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCandidate {
    pub id: ModelId,
    pub available: bool,
    pub egress: Measurement<ModelEgress>,
    pub supports_vision: Measurement<bool>,
    pub context_window: Measurement<u64>,
}

/// The models a caller is willing to let the coordinator consider.
///
/// Integration point for the CLI: `xencode-cli/src/model_decision.rs` already
/// measures the machine, reads the advice table and knows which model ids route
/// where. It can build this view (one [`ModelCandidate`] per model it knows,
/// with only the facts it actually measured) and pass it to
/// [`crate::Coordinator::new`]. This crate deliberately does not import the CLI,
/// the provider router, the TUI or the hardware probe, so no hardware or
/// egress logic is duplicated here.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityView {
    pub candidates: Vec<ModelCandidate>,
}

impl CapabilityView {
    pub fn new(candidates: Vec<ModelCandidate>) -> Self {
        Self { candidates }
    }

    pub fn candidates(&self) -> &[ModelCandidate] {
        &self.candidates
    }
}

/// A small, serializable reference to a real retrieval evaluation.
///
/// Built from `xencode_context_rs::EvalReport`. The coordinator quotes these
/// numbers as evidence; it never runs retrieval or evaluates anything itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalEvidence {
    pub label: String,
    pub queries: usize,
    pub mrr: f64,
}

impl RetrievalEvidence {
    pub fn from_report(label: impl Into<String>, report: &EvalReport) -> Self {
        Self {
            label: label.into(),
            queries: report.queries,
            mrr: report.mrr,
        }
    }
}

/// The task a caller wants coordinated.
#[derive(Debug, Clone)]
pub struct Task {
    /// The prompt itself. It is read with the same whole-word rule retrieval
    /// uses, so the shape in a decision can be traced back to these words.
    pub prompt: String,
    /// An optional shape word from a caller that already knows one. It is read
    /// with `TaskShape::parse`; an unknown word is ignored, never defaulted.
    pub declared_shape: Option<String>,
    /// Whether this turn needs a model that reports vision support.
    pub requires_vision: bool,
    /// An evaluation report the caller wants recorded as evidence, if any.
    pub retrieval_evidence: Option<RetrievalEvidence>,
    /// A single evaluation probe the caller wants named in the rationale.
    pub probe: Option<EvalItem>,
}

impl Task {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            declared_shape: None,
            requires_vision: false,
            retrieval_evidence: None,
            probe: None,
        }
    }
}

/// The limits a caller places on a decision.
///
/// These mirror xencode's egress posture, but no network call is made here to
/// enforce them: refusing a route at decision time is the whole job of this
/// layer, and the provider that actually dials a model keeps its own gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraints {
    /// Confine every choice to this machine. This is an absolute rule: it can
    /// narrow what the caller permits but can never be widened by `allow_cloud`.
    pub local_only: bool,
    /// Whether the caller permits an off-machine route at all.
    pub allow_cloud: bool,
    /// Whether the turn may use anything that needs a network.
    pub no_network: bool,
    /// Whether an outcome needs a verification result before it is recorded.
    pub require_verification: bool,
}

impl Default for Constraints {
    fn default() -> Self {
        Self::local_only()
    }
}

impl Constraints {
    /// The posture a new configuration ships with: nothing leaves the machine.
    pub fn local_only() -> Self {
        Self {
            local_only: true,
            allow_cloud: false,
            no_network: true,
            require_verification: true,
        }
    }

    /// A caller that has explicitly opened egress and still wants verification.
    pub fn cloud_allowed() -> Self {
        Self {
            local_only: false,
            allow_cloud: true,
            no_network: false,
            require_verification: true,
        }
    }

    /// Derive the egress half from a real configuration.
    ///
    /// The config's `allow_cloud_models` is the permission for an off-machine
    /// route. There is no separate local-only switch in config, so this does
    /// not set `local_only`: a caller that wants a hard local lock sets it
    /// explicitly, and the coordinator never clears it on its own.
    pub fn from_config(config: &XencodeConfig) -> Self {
        Self {
            local_only: false,
            allow_cloud: config.allow_cloud_models,
            no_network: !config.allow_cloud_models,
            require_verification: true,
        }
    }

    /// Whether all three gates permit an off-machine model. Local-only and the
    /// no-network rule each veto on their own, even if cloud was allowed.
    pub fn cloud_permitted(self) -> bool {
        self.allow_cloud && !self.local_only && !self.no_network
    }
}

/// One request to the coordinator.
#[derive(Debug, Clone)]
pub struct IntelligenceRequest {
    pub task: Task,
    pub session_id: String,
    pub project_root: PathBuf,
    pub constraints: Constraints,
    pub measurements: Measurements,
}

impl IntelligenceRequest {
    pub fn new(
        task: Task,
        session_id: impl Into<String>,
        project_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            task,
            session_id: session_id.into(),
            project_root: project_root.into(),
            constraints: Constraints::default(),
            measurements: Measurements::default(),
        }
    }
}

/// Plain-English names for where a piece of evidence came from.
pub mod sources {
    pub const TASK_PROMPT: &str = "the task prompt";
    pub const CAPABILITY_VIEW: &str = "the model capability view supplied by the caller";
    pub const CONSTRAINTS: &str = "the request's local-only and network constraints";
    pub const MEASUREMENTS: &str = "the measurements supplied by the caller";
    pub const RETRIEVAL_EVALUATION: &str = "the retrieval evaluation supplied by the caller";
}

/// One supporting fact attached to a decision.
///
/// `source` is a plain-English label so a printed decision can be argued with
/// without knowing the crate's internal type names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionEvidence {
    pub source: String,
    pub fact: String,
}

impl DecisionEvidence {
    pub fn new(source: impl Into<String>, fact: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            fact: fact.into(),
        }
    }

    /// Quote one existing evaluation probe as evidence without rerunning it.
    pub fn from_eval_item(label: &str, item: &EvalItem) -> Self {
        let expected = if item.expected.is_empty() {
            "no named file".to_string()
        } else {
            item.expected.join(", ")
        };
        Self::new(
            sources::RETRIEVAL_EVALUATION,
            format!(
                "{} probed {:?} expecting {}{}",
                label,
                item.query,
                expected,
                item.task_shape()
                    .map(|shape| format!(", read as {}", shape.as_str()))
                    .unwrap_or_default()
            ),
        )
    }

    /// Quote an existing evaluation report as evidence without rerunning it.
    pub fn from_eval_report(label: &str, report: &EvalReport) -> Self {
        let snapshot = RetrievalEvidence::from_report(label, report);
        Self::new(
            sources::RETRIEVAL_EVALUATION,
            format!(
                "{} measured {} queries at mean reciprocal rank {:.3}",
                snapshot.label, snapshot.queries, snapshot.mrr
            ),
        )
    }
}

/// What the caller expects to happen before an outcome is recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationPolicy {
    Required,
    Waived,
}

/// The decision a coordinator reached.
///
/// Every field is either a measured fact, a rule from the request, or the
/// outcome of applying those rules. An empty `selected_model` is a valid,
/// honest answer when nothing qualifies; a made-up model never is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntelligenceDecision {
    pub task_shape: TaskShape,
    pub selected_model: Option<ModelId>,
    pub context_budget: Measurement<ContextBudget>,
    pub verification_policy: VerificationPolicy,
    pub reasoning_budget: Measurement<ReasoningBudget>,
    pub rationale: Vec<DecisionEvidence>,
}

/// What actually happened when a decision was carried out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub status: ExecutionStatus,
    /// A short factual note from the caller, not a judgement by this crate.
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    NotRun,
    Completed,
    Failed { reason: String },
}

/// The verification result supplied by the caller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationRecord {
    pub status: VerificationStatus,
    /// The checks that really ran, named by the caller.
    pub checks: Vec<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus {
    NotRun,
    Passed,
    Failed { reason: String },
}

/// A decision paired with the real execution and verification that followed it.
///
/// Outcomes are recorded, never consulted by `decide`. A failed turn does not
/// change the next decision: this slice has no learning loop at all, and in
/// particular no unverified or repeated one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntelligenceOutcome {
    pub decision_id: String,
    pub execution: ExecutionRecord,
    pub verification: VerificationRecord,
}
