//! Coordination for context intelligence — a foundation slice, not a claim.
//!
//! This crate owns the *coordination* of a turn: given a task, a set of models a
//! caller is willing to offer, and the caller's constraints, it reaches one
//! inspectable decision and records what actually happened afterwards. It does
//! not implement retrieval, model selection from hardware, provider routing,
//! verification or learning. Those stay in the crates that already own them,
//! and this crate deliberately depends on neither the CLI nor the TUI.
//!
//! Three rules are enforced here, and they are the whole point of the slice:
//!
//! - **No model is chosen without measurements.** A candidate whose
//!   destination is unknown, that is unavailable, or that lacks a capability
//!   the task needs is refused with a reason. Missing budgets stay `Unknown`.
//! - **No cloud route without permission.** `local_only` and `no_network` veto
//!   an off-machine model on their own, even when `allow_cloud` is set; a
//!   cloud route needs the caller's explicit permission.
//! - **No repeated or unverified self-learning.** Decisions are a pure
//!   function of the request and capability view. Recorded outcomes are never
//!   read back by [`Coordinator::decide`], so a failed or unverified turn
//!   cannot quietly train the next one.
//!
//! Nothing here claims that xencode is "intelligent" yet: there is no model
//! quality score, no learned preference and no model-driven adaptation. The
//! [`reasoning`] module adds a deterministic, observed-event-only reasoning
//! plan (XI-4) that a caller may opt into, but it calls no model, edits nothing
//! and changes no existing agent-loop path on its own. This is the seam where
//! learned behaviour could later be attached without duplicating the model,
//! context or verification logic that already exists.
//!
//! The [`adaptation`] module (XI-5) adds bounded policy adaptation from
//! observed outcomes. It is not training and not self-modification: it changes
//! no weight, edits no file and widens no permission. Level 1 is an in-memory
//! session policy; level 2 is a repo-scoped, provenance-bearing policy that
//! can be rolled back exactly; level 3 (cross-session) is off until a caller
//! opts in and carries no private repository content. Model claims, denials,
//! unverified edits and incomplete checks are all refused as evidence.
//!
//! The [`diagnosis`] module (XI-6) adds evidence-driven failure diagnosis: a
//! candidate cause names a real command that would kill it, and no hypothesis
//! can leave `Proposed` until that command actually ran with its output
//! captured. There is no `Confirmed` status. The plan said to reuse an AI-2
//! hypothesis architecture; none exists in this tree (no `Hypothesis` type, no
//! implemented AI-2 row), so this module is the substrate a future AI-2 may
//! build on, not a claim that AI-2 is present.
//!
//! The [`repair`] module (XI-7) adds the pure coordination of a red-to-green
//! repair. It mirrors the TUI's production U-6 gate without importing it: no
//! production edit is authorized before a real failing check was witnessed,
//! the reproduction is frozen at that failure, only a granted human approval
//! unlocks an edit, and verification needs a clean checks verdict plus a real
//! check independent of the implementation step. A failed or skipped check
//! becomes `NotVerified` and can never be promoted to success. The state
//! machine writes nothing and runs nothing; the file edits and commands stay
//! in the TUI and CLI. Its guarantees are structural, not a measured drop in
//! false-verified repairs (that needs a GPU-backed model run).
//!
//! The [`observability`] module (XI-8) loads the decisions and outcomes the
//! coordinator already persisted and renders them as lines for the existing
//! `/trace` presentation. It is a loader plus a pure string renderer, not a
//! new dashboard or view: every figure comes from a real recorded value,
//! an unmeasured value reads as `unknown`, and no confidence or percentage is
//! invented. The dependency graph is unchanged — the TUI and CLI integration
//! point is documented in that module, not wired in here.
//!
//! # Integration
//!
//! The CLI already measures hardware and reads the advice table in
//! `xencode-cli/src/model_decision.rs`. When a caller wants a coordinated
//! turn, it should build a [`CapabilityView`] from those facts — one
//! [`ModelCandidate`] per model, carrying only measurements that really happened
//! — and pass it to [`Coordinator::new`]. This crate never imports the CLI, a
//! provider, the TUI or a hardware probe, so the dependency arrow points one
//! way and no domain crate needs to depend on this one.

pub mod adaptation;
pub mod coordinator;
pub mod diagnosis;
pub mod observability;
pub mod reasoning;
pub mod repair;
pub mod types;

pub use adaptation::{
    classify, evaluate_holdout, update_policy, AdaptationDecision, AdaptationError,
    AdaptationObservation, AdaptationParameters, AdaptationPolicy, Classification, Eligibility,
    EvidenceRecord, EvidenceRef, FixtureEvent, HoldoutReport, IneligibleObservation,
    LearningOptions, OutcomeKind, PolicyEnvelope, PolicyStatus, PolicyStore, Provenance,
    RecoveryHint, Scope, SessionAdaptation, SignalOrigin, ToolOutcomeStatus, DEFAULT_TOOL_ROUNDS,
    MAX_TOOL_ROUNDS, MIN_TOOL_ROUNDS, ROUNDS_STEP,
};
pub use coordinator::{Coordinator, IntelligenceError};
pub use diagnosis::{
    CheckExecution, DiagnosisEngine, DiagnosisError, DiagnosisReport, DiagnosisStore, Falsifier,
    FalsifierKind, Hypothesis, HypothesisId, HypothesisStatus, RecoveryAction, CARGO_CHECK,
    CARGO_TEST, MAX_HYPOTHESES,
};
pub use observability::{
    intelligence_report, intelligence_report_with_notes, load_decisions, load_decisions_with_notes,
    DecisionLoad, ObservabilityError, NOT_RECORDED, NOT_RECORDED_SESSION, UNKNOWN,
};
pub use reasoning::{
    decide as decide_reasoning, decide_from_trace, ReasoningInput, ReasoningPlan, ReasoningStage,
    RecoveryStrategy, VerificationIntensity, DEFAULT_MAX_TOOL_ROUNDS, HARD_MAX_TOOL_ROUNDS,
};
pub use repair::{
    ImplementAuthorization, PipelineStage, RepairAction, RepairApproval, RepairCheck, RepairError,
    RepairEvidence, RepairLoop, RepairPhase, RepairResult, RepairStore, ReproIdentity, Transition,
};
pub use types::{
    sources, CapabilityView, Constraints, ContextBudget, DecisionEvidence, ExecutionRecord,
    ExecutionStatus, IntelligenceDecision, IntelligenceOutcome, IntelligenceRequest, Measurement,
    Measurements, ModelCandidate, ModelEgress, ModelId, ReasoningBudget, RetrievalEvidence, Task,
    VerificationPolicy, VerificationRecord, VerificationStatus,
};
