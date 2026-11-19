//! The coordinator: decide once, record what actually happened.
//!
//! [`Coordinator::decide`] is a pure function of the request and the capability
//! view it was built with. It reads no files, starts no processes and dials no
//! provider. [`Coordinator::record`] is the only method that touches disk, and
//! it writes what the caller tells it happened; it never upgrades a failure to
//! a success. Recorded outcomes are never fed back into `decide`, so there is
//! no repeated or unverified self-learning in this slice.

use crate::reasoning::{self, ReasoningInput};
use crate::types::{
    sources, CapabilityView, ContextBudget, DecisionEvidence, ExecutionRecord,
    IntelligenceDecision, IntelligenceOutcome, IntelligenceRequest, Measurement, ModelEgress,
    ModelId, VerificationPolicy, VerificationRecord,
};
use std::fmt;
use std::path::PathBuf;
use xencode_context_rs::{shape_of, TaskShape};
use xencode_core_rs::write_atomic;

/// Coordinates one turn's decision and its durable record.
pub struct Coordinator {
    capability: CapabilityView,
    artifact_root: PathBuf,
}

impl Coordinator {
    /// Build a coordinator from the models a caller is willing to offer and
    /// the directory where decisions and outcomes should be kept.
    ///
    /// The caller owns hardware measurement and egress classification. Feed it
    /// a [`CapabilityView`] built from those facts; see the integration note on
    /// that type. No other crate is required to depend on this one.
    pub fn new(capability: CapabilityView, artifact_root: impl Into<PathBuf>) -> Self {
        Self {
            capability,
            artifact_root: artifact_root.into(),
        }
    }

    pub fn capability(&self) -> &CapabilityView {
        &self.capability
    }

    pub fn decisions_dir(&self) -> PathBuf {
        self.artifact_root.join("decisions")
    }

    pub fn outcomes_dir(&self) -> PathBuf {
        self.artifact_root.join("outcomes")
    }

    /// Decide what to do with a turn, with the evidence for every choice.
    ///
    /// Identical inputs always produce an identical decision. A model is only
    /// selected when its destination was measured, it is available, and every
    /// capability the task asked for was measured as supported. Missing values
    /// stay `Unknown`; no model, budget or capability is guessed.
    pub fn decide(
        &self,
        request: &IntelligenceRequest,
    ) -> Result<IntelligenceDecision, IntelligenceError> {
        if request.task.prompt.trim().is_empty() {
            return Err(IntelligenceError::EmptyTask);
        }
        if request.session_id.trim().is_empty() {
            return Err(IntelligenceError::EmptySessionId);
        }

        let read = shape_of(&request.task.prompt);
        let mut rationale = Vec::new();

        let task_shape = match request.task.declared_shape.as_deref() {
            Some(word) => match TaskShape::parse(word) {
                Some(shape) => {
                    rationale.push(DecisionEvidence::new(
                        sources::TASK_PROMPT,
                        format!(
                            "the caller declared shape {}, and the prompt read as {}: {}",
                            shape.as_str(),
                            read.shape.as_str(),
                            read.reasons.join("; ")
                        ),
                    ));
                    shape
                }
                None => {
                    rationale.push(DecisionEvidence::new(
                        sources::TASK_PROMPT,
                        format!(
                            "the caller's shape word {word:?} is not a known shape, so the prompt's own words read as {}: {}",
                            read.shape.as_str(),
                            read.reasons.join("; ")
                        ),
                    ));
                    read.shape
                }
            },
            None => {
                rationale.push(DecisionEvidence::new(
                    sources::TASK_PROMPT,
                    format!(
                        "read as {} work: {}",
                        read.shape.as_str(),
                        read.reasons.join("; ")
                    ),
                ));
                read.shape
            }
        };

        let cloud_permitted = request.constraints.cloud_permitted();
        rationale.push(DecisionEvidence::new(
            sources::CONSTRAINTS,
            format!(
                "local-only is {}, no-network is {}, cloud is allowed by the caller is {}, so an off-machine model {} be selected",
                yes_no(request.constraints.local_only),
                yes_no(request.constraints.no_network),
                yes_no(request.constraints.allow_cloud),
                if cloud_permitted { "may" } else { "cannot" }
            ),
        ));

        let context_budget: Measurement<ContextBudget> =
            request.measurements.context.map(ContextBudget::from);
        match &context_budget {
            Measurement::Known(budget) => rationale.push(DecisionEvidence::new(
                sources::MEASUREMENTS,
                format!("the caller measured a context budget of {budget}"),
            )),
            Measurement::Unknown => rationale.push(DecisionEvidence::new(
                sources::MEASUREMENTS,
                "no context budget was measured, so this decision claims no context size",
            )),
        }

        let reasoning_budget = request.measurements.reasoning;
        match &reasoning_budget {
            Measurement::Known(budget) => rationale.push(DecisionEvidence::new(
                sources::MEASUREMENTS,
                format!(
                    "the caller supplied a reasoning budget of {} rounds with deep-think {}",
                    budget.max_rounds,
                    if budget.allow_deep_think {
                        "allowed"
                    } else {
                        "not allowed"
                    }
                ),
            )),
            Measurement::Unknown => rationale.push(DecisionEvidence::new(
                sources::MEASUREMENTS,
                "no reasoning budget was measured, so this decision claims no round limit",
            )),
        }

        let verification_policy = if request.constraints.require_verification {
            VerificationPolicy::Required
        } else {
            VerificationPolicy::Waived
        };
        rationale.push(DecisionEvidence::new(
            sources::CONSTRAINTS,
            match verification_policy {
                VerificationPolicy::Required => {
                    "verification is required before an outcome is recorded"
                }
                VerificationPolicy::Waived => {
                    "the caller waived verification for this turn; no pass is claimed on that basis"
                }
            },
        ));

        if let Some(evaluation) = &request.task.retrieval_evidence {
            rationale.push(DecisionEvidence::new(
                sources::RETRIEVAL_EVALUATION,
                format!(
                    "{} measured {} queries at mean reciprocal rank {:.3}",
                    evaluation.label, evaluation.queries, evaluation.mrr
                ),
            ));
        }
        if let Some(probe) = &request.task.probe {
            rationale.push(DecisionEvidence::from_eval_item("a retrieval probe", probe));
        }

        let selected_model = self.choose_model(request, cloud_permitted, &mut rationale);

        Ok(IntelligenceDecision {
            task_shape,
            selected_model,
            context_budget,
            verification_policy,
            reasoning_budget,
            rationale,
            reasoning_plan: None,
        })
    }

    /// Decide a turn and attach an adaptive reasoning plan (XI-4).
    ///
    /// This is the opt-in consumption point. The plan is a pure function of a
    /// finished [`xencode_context_rs::TurnTrace`] the caller supplies in
    /// [`ReasoningInput`], plus this request's shape and constraints. The
    /// ordinary [`Self::decide`] path is unchanged and leaves the plan unset,
    /// so no existing agent-loop behaviour changes; a future CLI turn may call
    /// this and act on `decision.reasoning_plan` through the existing deep-think
    /// and tool-loop seams. Nothing here routes a model or edits a file.
    pub fn decide_with_reasoning(
        &self,
        request: &IntelligenceRequest,
        reasoning: &ReasoningInput<'_>,
    ) -> Result<IntelligenceDecision, IntelligenceError> {
        let mut decision = self.decide(request)?;
        decision.reasoning_plan = Some(reasoning::decide(reasoning));
        Ok(decision)
    }

    /// Walk candidates in a stable order and take the first one that truly
    /// qualifies. Every refusal is written to the rationale so a reader can
    /// see why a model was skipped.
    fn choose_model(
        &self,
        request: &IntelligenceRequest,
        cloud_permitted: bool,
        rationale: &mut Vec<DecisionEvidence>,
    ) -> Option<ModelId> {
        let mut candidates: Vec<&crate::types::ModelCandidate> =
            self.capability.candidates.iter().collect();
        candidates.sort_by(|left, right| left.id.as_str().cmp(right.id.as_str()));

        for candidate in candidates {
            let id = candidate.id.as_str();
            if !candidate.available {
                rationale.push(DecisionEvidence::new(
                    sources::CAPABILITY_VIEW,
                    format!("model {id} was refused: the caller reported it unavailable"),
                ));
                continue;
            }

            let egress = match candidate.egress {
                Measurement::Known(egress) => egress,
                Measurement::Unknown => {
                    rationale.push(DecisionEvidence::new(
                        sources::CAPABILITY_VIEW,
                        format!("model {id} was refused: where it sends a prompt was not measured"),
                    ));
                    continue;
                }
            };
            if egress == ModelEgress::Cloud && !cloud_permitted {
                let why = if request.constraints.local_only {
                    "local-only is set and cannot be overridden here"
                } else if request.constraints.no_network {
                    "network use is disabled for this turn"
                } else {
                    "the caller did not allow cloud models"
                };
                rationale.push(DecisionEvidence::new(
                    sources::CAPABILITY_VIEW,
                    format!(
                        "model {id} was refused: it routes {} and {why}",
                        egress.label()
                    ),
                ));
                continue;
            }

            if request.task.requires_vision {
                match candidate.supports_vision {
                    Measurement::Known(true) => {}
                    Measurement::Known(false) => {
                        rationale.push(DecisionEvidence::new(
                            sources::CAPABILITY_VIEW,
                            format!("model {id} was refused: it reports no vision support"),
                        ));
                        continue;
                    }
                    Measurement::Unknown => {
                        rationale.push(DecisionEvidence::new(
                            sources::CAPABILITY_VIEW,
                            format!("model {id} was refused: its vision support was not measured"),
                        ));
                        continue;
                    }
                }
            }

            rationale.push(DecisionEvidence::new(
                sources::CAPABILITY_VIEW,
                format!(
                    "model {id} was selected: it is available, it routes {}, and every capability this task asked for was measured as supported",
                    egress.label()
                ),
            ));
            return Some(candidate.id.clone());
        }

        rationale.push(DecisionEvidence::new(
            sources::CAPABILITY_VIEW,
            "no model was selected: no candidate met the constraints and measured capabilities, and none was invented",
        ));
        None
    }

    /// A stable identifier for one decision of one request.
    ///
    /// Derived only from the inputs and the decision itself, so it is the same
    /// on every machine and never depends on a clock or a directory listing.
    pub fn decision_id(
        &self,
        request: &IntelligenceRequest,
        decision: &IntelligenceDecision,
    ) -> String {
        let canonical = serde_json::to_vec(&(
            &request.task.prompt,
            &request.task.declared_shape,
            request.task.requires_vision,
            &request.session_id,
            &request.project_root,
            request.constraints,
            decision,
        ))
        .expect("decision inputs must serialize");
        stable_id(&canonical)
    }

    /// Record a decision together with what really followed it.
    ///
    /// The caller supplies the execution and verification results; this method
    /// does not run anything and does not infer a pass. It recomputes the
    /// decision from the request first, so an outcome cannot be attached to a
    /// decision this coordinator would not have made. Both JSON files are
    /// written atomically under the artifact root.
    pub fn record(
        &self,
        request: &IntelligenceRequest,
        decision: &IntelligenceDecision,
        execution: ExecutionRecord,
        verification: VerificationRecord,
    ) -> Result<IntelligenceOutcome, IntelligenceError> {
        // The reasoning plan is opt-in and derived from a separate trace, not
        // from the request, so it is not part of what `decide` recomputes here.
        // Compare the base decision without it: a caller may attach a plan and
        // still record a real outcome, and a forged plan cannot turn a decision
        // this coordinator would not make into a valid one.
        let mut recomputed = self.decide(request)?;
        recomputed.reasoning_plan = None;
        let mut submitted = decision.clone();
        submitted.reasoning_plan = None;
        if recomputed != submitted {
            return Err(IntelligenceError::DecisionMismatch);
        }
        let id = self.decision_id(request, decision);
        let outcome = IntelligenceOutcome {
            decision_id: id.clone(),
            execution,
            verification,
        };

        let decision_path = self.decision_path(&id);
        let outcome_path = self.outcome_path(&id);
        let decision_bytes =
            serde_json::to_vec_pretty(decision).expect("a decision must serialize");
        let outcome_bytes = serde_json::to_vec_pretty(&outcome).expect("an outcome must serialize");
        write_atomic(&decision_path, &decision_bytes).map_err(|source| {
            IntelligenceError::Write {
                path: decision_path.clone(),
                source,
            }
        })?;
        write_atomic(&outcome_path, &outcome_bytes).map_err(|source| IntelligenceError::Write {
            path: outcome_path.clone(),
            source,
        })?;
        Ok(outcome)
    }

    /// Read back a decision that was recorded earlier.
    pub fn read_decision(
        &self,
        decision_id: &str,
    ) -> Result<IntelligenceDecision, IntelligenceError> {
        let path = self.decision_path(decision_id);
        let bytes = std::fs::read(&path).map_err(|source| IntelligenceError::Read {
            path: path.clone(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|source| IntelligenceError::Decode { path, source })
    }

    /// Read back an outcome that was recorded earlier.
    pub fn read_outcome(
        &self,
        decision_id: &str,
    ) -> Result<IntelligenceOutcome, IntelligenceError> {
        let path = self.outcome_path(decision_id);
        let bytes = std::fs::read(&path).map_err(|source| IntelligenceError::Read {
            path: path.clone(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|source| IntelligenceError::Decode { path, source })
    }

    fn decision_path(&self, id: &str) -> PathBuf {
        self.decisions_dir().join(format!("{id}.json"))
    }

    fn outcome_path(&self, id: &str) -> PathBuf {
        self.outcomes_dir().join(format!("{id}.json"))
    }
}

/// FNV-1a over the canonical input bytes. A small, stable hash is enough for
/// an identifier: it is not a signature, and it must not change between
/// builds the way `DefaultHasher` may.
fn stable_id(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

/// Why a coordinator call could not produce or store a decision.
///
/// A model being unavailable or unmeasured is *not* an error: it produces a
/// decision with no selected model and the reasons in its rationale. Errors
/// are reserved for invalid requests and failed storage, so a caller can never
/// mistake one for a successful decision.
#[derive(Debug)]
pub enum IntelligenceError {
    EmptyTask,
    EmptySessionId,
    DecisionMismatch,
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Decode {
        path: PathBuf,
        source: serde_json::Error,
    },
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl fmt::Display for IntelligenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTask => f.write_str("a decision needs a non-empty task prompt"),
            Self::EmptySessionId => f.write_str("a decision needs a non-empty session id"),
            Self::DecisionMismatch => f.write_str(
                "the decision does not match what this coordinator would decide for the request",
            ),
            Self::Read { path, source } => {
                write!(f, "could not read {}: {source}", path.display())
            }
            Self::Decode { path, source } => {
                write!(f, "could not decode {}: {source}", path.display())
            }
            Self::Write { path, source } => {
                write!(f, "could not write {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for IntelligenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } | Self::Write { source, .. } => Some(source),
            Self::Decode { source, .. } => Some(source),
            _ => None,
        }
    }
}
