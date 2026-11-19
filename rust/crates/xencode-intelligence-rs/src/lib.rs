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
//! quality score, no learned preference and no adaptive behaviour. This is the
//! seam where those things could later be attached without duplicating the
//! model, context or verification logic that already exists.
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

pub mod coordinator;
pub mod types;

pub use coordinator::{Coordinator, IntelligenceError};
pub use types::{
    sources, CapabilityView, Constraints, ContextBudget, DecisionEvidence, ExecutionRecord,
    ExecutionStatus, IntelligenceDecision, IntelligenceOutcome, IntelligenceRequest, Measurement,
    Measurements, ModelCandidate, ModelEgress, ModelId, ReasoningBudget, RetrievalEvidence, Task,
    VerificationPolicy, VerificationRecord, VerificationStatus,
};
