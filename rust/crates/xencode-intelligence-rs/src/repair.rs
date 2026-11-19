//! Evidence-gated repair coordination (XI-7): a pure red-to-green state machine.
//!
//! This module is the *coordination* part of a repair, not the part that edits
//! files or runs commands. It decides when a production edit may happen and
//! what evidence is needed before a repair may be called verified. The actual
//! file writes and process spawns stay in the TUI and the CLI, which consult
//! this state machine and supply the typed evidence it asks for. This crate
//! depends on neither of them, and the dependency-boundary test pins that.
//!
//! # The gate, mirrored from the TUI
//!
//! [`RepairPhase`] mirrors `xencode-tui-rs::reprogate::Phase` (the production
//! U-6 gate): `Off → AwaitingRed → RedWitnessed → Verified`. That type cannot
//! be imported here without breaking the dependency boundary, so this is an
//! equivalent model of the *same* rule, not a second enforcement point. The TUI
//! gate remains the place where a write tool call is accepted or refused on
//! disk; this module only records whether the evidence for that decision
//! exists. A caller cannot smuggle a "verified" result past it with a claim:
//!
//! - no production edit is authorized until a real failing check has been
//!   witnessed (a `CheckExecution` produced by actually running a process);
//! - the reproduction file's identity is frozen at the red, and any later
//!   change to it invalidates the repair (the manufactured-pass guard);
//! - only a human approval row (the same tool/class/decision shape the run
//!   ledger stores) unlocks an edit; a denial never unlocks one, and no new
//!   permission is invented here;
//! - verification needs both a clean [`ChecksVerdict`] and a *separate* real
//!   check whose command differs from the implementation step. The component
//!   that proposed or applied the fix cannot be the sole authority for it;
//! - a failed or skipped check, or a missing piece of evidence, renders as
//!   [`RepairResult::NotVerified`] and loops back to diagnosis. A failed
//!   verification can never be promoted to success later.
//!
//! The [`ChecksVerdict`] is reused from `xencode-context-rs`: its `all_passed`
//! means "the due checks really ran and none failed", **not** "the change is
//! correct". This module pairs that verdict with a fresh real check so a stale
//! passing ledger row cannot stand in for a verification.
//!
//! # What is deliberately not here
//!
//! There is no calibration improvement claim. The guarantees above are
//! structural — they come from the shape of the transitions and the evidence
//! types, not from a measured drop in false-verified rate. Measuring that rate
//! on real repairs needs a GPU-backed model run, which this environment does
//! not have. Nothing in this module selects a model, edits a file on its own,
//! or widens a permission.

use crate::diagnosis::CheckExecution;
use crate::reasoning::{ReasoningPlan, RecoveryStrategy};
use crate::types::DecisionEvidence;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use xencode_context_rs::{ApprovalDecision, ChecksVerdict};
use xencode_core_rs::{read_jsonl_tolerant, write_atomic};

/// Where a repair has got to. This mirrors the TUI's red-to-green `Phase`; the
/// TUI gate is the real write choke point, this enum is the coordination view
/// of the same measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairPhase {
    /// No repair is in progress; nothing is forbidden here.
    #[default]
    Off,
    /// A repair is open and its failure has not been seen. No production edit.
    AwaitingRed,
    /// The failure was witnessed by a real check. Edits may now be authorized.
    RedWitnessed,
    /// Both points of the measurement exist and passed review.
    Verified,
}

impl RepairPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::AwaitingRed => "awaiting a failing reproduction",
            Self::RedWitnessed => "red witnessed, awaiting the green",
            Self::Verified => "red then green recorded",
        }
    }
}

/// The ordered pipeline a repair walks through. Only evidence-backed
/// transitions may move it forward; a failed review moves it back to
/// `Diagnose`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    #[default]
    Understand,
    Diagnose,
    Plan,
    Reproduce,
    Implement,
    Verify,
    EvidenceReview,
    Report,
}

impl PipelineStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Understand => "understand",
            Self::Diagnose => "diagnose",
            Self::Plan => "plan",
            Self::Reproduce => "reproduce",
            Self::Implement => "implement",
            Self::Verify => "verify",
            Self::EvidenceReview => "evidence_review",
            Self::Report => "report",
        }
    }
}

/// The honest outcome of a repair. `NotVerified` is the default for anything
/// unfinished, interrupted, incomplete or ambiguous; it is never a guess that
/// the change worked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairResult {
    /// A witnessed failure, a clean verdict and a separate passing check exist.
    Verified,
    /// No such evidence exists, or a check failed or was skipped.
    #[default]
    NotVerified,
    /// The repair was stopped before a result was reached; it can be resumed.
    Interrupted,
}

impl RepairResult {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::NotVerified => "not_verified",
            Self::Interrupted => "interrupted",
        }
    }

    /// Only a real, complete measurement counts as success.
    pub fn is_success(self) -> bool {
        self == Self::Verified
    }
}

/// A real check's captured output, copied from a [`CheckExecution`] that was
/// produced by running a process. There is deliberately no constructor that
/// takes a model's prose: the only way to make one is to execute a real check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairCheck {
    pub command: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl RepairCheck {
    /// Snapshot a check that really ran. This is the only way to build a
    /// `RepairCheck`, which is why a claim cannot stand in for one.
    pub fn from_execution(execution: &CheckExecution) -> Self {
        Self {
            command: execution.command.clone(),
            exit_code: execution.exit_code,
            stdout: execution.stdout.clone(),
            stderr: execution.stderr.clone(),
        }
    }

    /// A check passed only when the real process exited zero.
    pub fn passed(&self) -> bool {
        self.exit_code == 0
    }

    /// The real stdout and stderr kept together, as captured.
    pub fn output(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }
}

/// The identity of the reproduction file at the moment red was witnessed:
/// its path and a digest of the exact bytes on disk. Changing either the path
/// or the bytes later invalidates the repair. The caller reads the file; this
/// crate never touches the filesystem to decide a transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproIdentity {
    pub path: String,
    pub digest: String,
}

impl ReproIdentity {
    /// Fingerprint the real bytes of a reproduction file at a given path.
    pub fn of(path: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            path: path.into(),
            digest: fnv1a(bytes),
        }
    }
}

/// Mirror of `xencode_context_rs::ApprovalRow` (tool / class / decision).
/// Nothing here grants a new permission: it only records the decision a human
/// already made in the existing approval overlay and run ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairApproval {
    pub tool: String,
    pub class: String,
    pub decision: ApprovalDecision,
}

impl RepairApproval {
    pub fn new(
        tool: impl Into<String>,
        class: impl Into<String>,
        decision: ApprovalDecision,
    ) -> Self {
        Self {
            tool: tool.into(),
            class: class.into(),
            decision,
        }
    }

    /// Both kinds of "yes" count, exactly as they do in the run ledger. A
    /// denial never unlocks an edit.
    pub fn granted(&self) -> bool {
        self.decision.granted()
    }
}

/// The action a caller is currently allowed to take. This is what the TUI or
/// CLI consults before it edits a file or runs a check; there is no hidden
/// route to `Implement`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairAction {
    /// Engage (or re-engage) the gate and reproduce the failure.
    Reproduce,
    /// A granted human approval is needed before an edit can be authorized.
    AwaitHumanApproval,
    /// A production edit may now be authorized by the caller.
    Implement,
    /// Run the independent verification check and review the verdict.
    Verify,
    /// A failed verification: go back to diagnosis with a fresh measurement.
    Rediagnose,
    /// The measurement is complete; report the evidence.
    Report,
}

/// One recorded step of a repair, kept in the evidence record so a reader can
/// argue with the path that led to the result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transition {
    pub from_stage: PipelineStage,
    pub to_stage: PipelineStage,
    pub from_phase: RepairPhase,
    pub to_phase: RepairPhase,
    pub detail: String,
}

/// Why a transition or an authorization was refused. Each variant names the
/// missing piece of evidence; none of them can be satisfied by a model claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairError {
    /// The gate was not engaged yet.
    NotEngaged,
    /// A production edit or verification was attempted before a real failure.
    RedNotWitnessed,
    /// A red was recorded while the gate was in the wrong phase.
    NotAwaitingRed,
    /// The supposed reproduction check actually exited zero.
    RedCheckPassed,
    /// No granted human approval is on record, so no edit may be authorized.
    ApprovalRequired,
    /// The green used the implementation's command; verification is not
    /// independent of the step it is checking.
    NotIndependent {
        impl_command: String,
        verify_command: String,
    },
    /// The green re-ran a different command than the witnessed red, so the two
    /// points are not the same measurement.
    NotSameMeasurement {
        red_command: String,
        verify_command: String,
    },
    /// The frozen reproduction changed after red.
    ReproChanged { expected: String, actual: String },
    /// A failed verification is terminal for this repair; promote it to a new
    /// repair with fresh evidence instead.
    VerificationAlreadyFailed,
    /// A model claim was offered as verification evidence and refused.
    ModelClaimNotEvidence,
    /// Verification was attempted before the gate reached red.
    NotReadyToVerify { phase: String },
    /// Verification was attempted before an edit was authorized.
    ImplementNotAuthorized,
    /// The repair finished or was interrupted, so this transition is stale.
    NotInProgress,
    /// The reasoning plan's hard round cap is exhausted.
    HardCapReached { cap: u32 },
    /// A store read or write failed.
    Store { path: PathBuf, message: String },
}

impl fmt::Display for RepairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEngaged => f.write_str("the repair gate has not been engaged"),
            Self::RedNotWitnessed => f.write_str(
                "no production edit is allowed before a real failing reproduction was witnessed",
            ),
            Self::NotAwaitingRed => {
                f.write_str("a failure can only be witnessed while awaiting red")
            }
            Self::RedCheckPassed => {
                f.write_str("the reproduction check exited zero, so no failure was witnessed")
            }
            Self::ApprovalRequired => {
                f.write_str("a granted human approval is required before a production edit")
            }
            Self::NotIndependent {
                impl_command,
                verify_command,
            } => write!(
                f,
                "verification command {verify_command:?} is the implementation command {impl_command:?}; verification must be independent"
            ),
            Self::NotSameMeasurement {
                red_command,
                verify_command,
            } => write!(
                f,
                "the red used {red_command:?} but the green used {verify_command:?}; that is not the same measurement"
            ),
            Self::ReproChanged { expected, actual } => write!(
                f,
                "the reproduction was frozen as {expected} but is now {actual}; the repair is invalid"
            ),
            Self::VerificationAlreadyFailed => {
                f.write_str("this repair already failed verification and cannot become verified")
            }
            Self::ModelClaimNotEvidence => {
                f.write_str("a model claim is not a check and cannot verify a repair")
            }
            Self::NotReadyToVerify { phase } => {
                write!(f, "verification is not available in phase {phase}")
            }
            Self::ImplementNotAuthorized => {
                f.write_str("no production edit has been authorized for this repair")
            }
            Self::NotInProgress => f.write_str("the repair is not in progress"),
            Self::HardCapReached { cap } => {
                write!(f, "the planned hard cap of {cap} rounds is exhausted")
            }
            Self::Store { path, message } => {
                write!(f, "could not use {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for RepairError {}

/// The authorization returned by [`RepairLoop::implement`]. It is a record of
/// what the human permitted, not a permission the crate invents: the caller
/// still performs the edit in its own layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImplementAuthorization {
    pub repair_id: String,
    pub diagnosis_id: String,
    pub approval: RepairApproval,
    pub impl_command: String,
}

/// The persisted evidence for one repair. A store writes this under a caller
/// directory with the atomic writer; a reader can resume an interrupted repair
/// from it. `result` is optional on the wire on purpose: an absent, ambiguous
/// or incomplete result reads as `NotVerified`, never as success.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairEvidence {
    pub repair_id: String,
    pub diagnosis_id: String,
    pub phase: RepairPhase,
    pub stage: PipelineStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<RepairResult>,
    /// The real failing reproduction, with its captured output and exit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub red: Option<RepairCheck>,
    /// The independent real green check, with its captured output and exit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub green: Option<RepairCheck>,
    /// The due-check verdict that accompanied the green.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<ChecksVerdict>,
    /// The human approval that unlocked the edit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval: Option<RepairApproval>,
    /// The frozen reproduction identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repro: Option<ReproIdentity>,
    /// The command/evidence ref of the implementation step, kept so
    /// verification can be checked for independence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub impl_command: Option<String>,
    /// True only once a real green check passed. An authorized-but-unverified
    /// edit is never recorded as a write.
    pub write_recorded: bool,
    /// Model claims that were offered as evidence and refused.
    #[serde(default)]
    pub refused_claims: Vec<DecisionEvidence>,
    /// The recovery rule from the XI-4 reasoning plan used on a loop-back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery: Option<RecoveryStrategy>,
    /// The hard round cap from that same plan, when one was supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub round_cap: Option<u32>,
    pub attempts: u32,
    /// Why verification did not succeed, when it did not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
    pub transitions: Vec<Transition>,
}

impl RepairEvidence {
    /// The honest result: missing or ambiguous evidence is `NotVerified`.
    pub fn result(&self) -> RepairResult {
        self.result.unwrap_or(RepairResult::NotVerified)
    }

    /// `true` only for a complete, verified measurement.
    pub fn is_success(&self) -> bool {
        self.result().is_success()
    }
}

/// The pure repair state machine. It never spawns a process, edits a file or
/// reads a clock; every input is typed evidence supplied by a caller.
#[derive(Debug, Clone)]
pub struct RepairLoop {
    evidence: RepairEvidence,
}

impl RepairLoop {
    /// Start a repair. It is `Understand` / `Off` with no evidence at all.
    pub fn new(repair_id: impl Into<String>, diagnosis_id: impl Into<String>) -> Self {
        Self {
            evidence: RepairEvidence {
                repair_id: repair_id.into(),
                diagnosis_id: diagnosis_id.into(),
                phase: RepairPhase::Off,
                stage: PipelineStage::Understand,
                result: None,
                red: None,
                green: None,
                verdict: None,
                approval: None,
                repro: None,
                impl_command: None,
                write_recorded: false,
                refused_claims: Vec::new(),
                recovery: None,
                round_cap: None,
                attempts: 0,
                failure_reason: None,
                transitions: Vec::new(),
            },
        }
    }

    /// Record that the failure is understood, as the first pipeline step.
    pub fn diagnose(&mut self) -> Result<(), RepairError> {
        self.advance(PipelineStage::Diagnose, self.evidence.phase, "understood")
    }

    /// Record that a recovery plan was chosen.
    pub fn plan(&mut self) -> Result<(), RepairError> {
        self.advance(PipelineStage::Plan, self.evidence.phase, "planned")
    }

    /// Attach the XI-4 reasoning plan. Its recovery rule and hard round cap are
    /// reused here instead of inventing a second retry policy.
    pub fn set_reasoning_plan(&mut self, plan: &ReasoningPlan) {
        self.evidence.recovery = Some(plan.recovery);
        self.evidence.round_cap = Some(plan.max_tool_rounds);
    }

    /// Engage the gate: move to `Reproduce` / `AwaitingRed`. Nothing else may
    /// be edited until a failure is witnessed.
    pub fn begin(&mut self) -> Result<(), RepairError> {
        if self.evidence.phase != RepairPhase::Off {
            return Err(RepairError::NotInProgress);
        }
        self.advance(
            PipelineStage::Reproduce,
            RepairPhase::AwaitingRed,
            "gate engaged; awaiting a real failing reproduction",
        )
    }

    /// Witness the red: a real check that actually exited nonzero, plus the
    /// identity of the reproduction file as it stood at that moment.
    pub fn witness_red(
        &mut self,
        check: &CheckExecution,
        repro: ReproIdentity,
    ) -> Result<(), RepairError> {
        if self.evidence.phase != RepairPhase::AwaitingRed {
            return Err(RepairError::NotAwaitingRed);
        }
        if check.passed() {
            return Err(RepairError::RedCheckPassed);
        }
        let red = RepairCheck::from_execution(check);
        let detail = format!(
            "real {} exited {} and reproduced the failure; reproduction {} frozen as {}",
            red.command, red.exit_code, repro.path, repro.digest
        );
        self.evidence.red = Some(red);
        self.evidence.repro = Some(repro);
        self.advance(PipelineStage::Implement, RepairPhase::RedWitnessed, detail)
    }

    /// Authorize a production edit. This is the only route to an `Implement`
    /// action and it needs both a witnessed red and a granted human approval.
    pub fn implement(
        &mut self,
        approval: &RepairApproval,
        impl_command: &str,
    ) -> Result<ImplementAuthorization, RepairError> {
        if self.evidence.phase != RepairPhase::RedWitnessed {
            return Err(RepairError::RedNotWitnessed);
        }
        if !approval.granted() {
            return Err(RepairError::ApprovalRequired);
        }
        if impl_command.trim().is_empty() {
            return Err(RepairError::ApprovalRequired);
        }
        self.evidence.approval = Some(approval.clone());
        self.evidence.impl_command = Some(impl_command.to_string());
        // An authorization is not a write: no file change is recorded until
        // its independent check passes.
        self.advance(
            PipelineStage::Implement,
            self.evidence.phase,
            format!(
                "human approval for {} ({}) recorded; edit step {impl_command} authorized, no write recorded",
                approval.tool, approval.class
            ),
        )?;
        Ok(ImplementAuthorization {
            repair_id: self.evidence.repair_id.clone(),
            diagnosis_id: self.evidence.diagnosis_id.clone(),
            approval: approval.clone(),
            impl_command: impl_command.to_string(),
        })
    }

    /// Record a model claim as refused. It never advances the repair and never
    /// becomes a check: the returned error is also stored in the evidence.
    pub fn refuse_model_claim(&mut self, claim: DecisionEvidence) -> RepairError {
        self.evidence.refused_claims.push(claim.clone());
        let _ = self.advance(
            self.evidence.stage,
            self.evidence.phase,
            format!(
                "refused a claim from {:?} as verification evidence",
                claim.source
            ),
        );
        RepairError::ModelClaimNotEvidence
    }

    /// The action the caller may currently take. Deliberately never returns
    /// `Implement` while the phase is `AwaitingRed` or the approval is absent.
    pub fn next_action(&self) -> RepairAction {
        if self.evidence.result == Some(RepairResult::NotVerified) {
            return RepairAction::Rediagnose;
        }
        if self.evidence.result == Some(RepairResult::Interrupted) {
            return RepairAction::Reproduce;
        }
        if self.evidence.result == Some(RepairResult::Verified) {
            return RepairAction::Report;
        }
        match self.evidence.phase {
            RepairPhase::Off | RepairPhase::AwaitingRed => RepairAction::Reproduce,
            RepairPhase::Verified => RepairAction::Report,
            RepairPhase::RedWitnessed => {
                if self.evidence.approval.is_none() || self.evidence.impl_command.is_none() {
                    RepairAction::AwaitHumanApproval
                } else {
                    RepairAction::Verify
                }
            }
        }
    }

    /// Verify with a clean verdict and a real green check. The green must be a
    /// different command from the implementation step (independence) and the
    /// same command as the witnessed red (one measurement). A failed or skipped
    /// check, or a changed reproduction, is terminal `NotVerified`.
    pub fn verify(
        &mut self,
        verdict: &ChecksVerdict,
        green: &CheckExecution,
        current_repro: &ReproIdentity,
    ) -> Result<RepairResult, RepairError> {
        if self.evidence.result == Some(RepairResult::NotVerified) {
            return Err(RepairError::VerificationAlreadyFailed);
        }
        if self.evidence.phase != RepairPhase::RedWitnessed {
            return Err(RepairError::NotReadyToVerify {
                phase: self.evidence.phase.label().to_string(),
            });
        }
        let impl_command = self
            .evidence
            .impl_command
            .clone()
            .ok_or(RepairError::ImplementNotAuthorized)?;
        let red_command = self
            .evidence
            .red
            .as_ref()
            .map(|r| r.command.clone())
            .ok_or(RepairError::RedNotWitnessed)?;
        if green.command == impl_command {
            return Err(RepairError::NotIndependent {
                impl_command,
                verify_command: green.command.clone(),
            });
        }
        if green.command != red_command {
            return Err(RepairError::NotSameMeasurement {
                red_command,
                verify_command: green.command.clone(),
            });
        }
        let frozen = self
            .evidence
            .repro
            .as_ref()
            .ok_or(RepairError::RedNotWitnessed)?;
        if frozen != current_repro {
            let expected = format!("{}@{}", frozen.path, frozen.digest);
            let actual = format!("{}@{}", current_repro.path, current_repro.digest);
            let reason = format!("reproduction changed after red: {expected} became {actual}");
            self.advance(PipelineStage::Diagnose, self.evidence.phase, &reason)?;
            self.evidence.result = Some(RepairResult::NotVerified);
            self.evidence.failure_reason = Some(reason);
            self.evidence.write_recorded = false;
            return Err(RepairError::ReproChanged { expected, actual });
        }

        self.advance(
            PipelineStage::Verify,
            self.evidence.phase,
            format!(
                "independent real {} exited {}; verdict {}",
                green.command,
                green.exit_code,
                verdict.summary()
            ),
        )?;
        self.evidence.green = Some(RepairCheck::from_execution(green));
        self.evidence.verdict = Some(verdict.clone());
        self.evidence.attempts += 1;

        if !green.passed() {
            return self.not_verified(format!(
                "the verification check {} exited {}, not zero",
                green.command, green.exit_code
            ));
        }
        if !verdict.all_passed() {
            return self.not_verified(format!("checks did not all pass: {}", verdict.summary()));
        }

        self.advance(
            PipelineStage::EvidenceReview,
            self.evidence.phase,
            "red and green witnessed on the same command, verdict clean, approval on record",
        )?;
        self.advance(
            PipelineStage::Report,
            RepairPhase::Verified,
            "repair verified on real evidence",
        )?;
        self.evidence.result = Some(RepairResult::Verified);
        // Only now is the edit recorded as a verified write.
        self.evidence.write_recorded = true;
        Ok(RepairResult::Verified)
    }

    /// Stop before a result is reached. Persisting at this point records no
    /// verified write, so a later run can resume without replaying one.
    pub fn interrupt(&mut self) -> Result<(), RepairError> {
        if self.evidence.result == Some(RepairResult::Verified) {
            return Err(RepairError::NotInProgress);
        }
        self.evidence.result = Some(RepairResult::Interrupted);
        self.advance(
            self.evidence.stage,
            self.evidence.phase,
            "repair interrupted before verification",
        )
    }

    /// Resume an interrupted repair from its persisted evidence. The red,
    /// approval, frozen reproduction and authorization are kept; the green and
    /// the verified-write flag are not invented.
    pub fn resume(&mut self) -> Result<(), RepairError> {
        if self.evidence.result != Some(RepairResult::Interrupted) {
            return Err(RepairError::NotInProgress);
        }
        self.evidence.result = None;
        self.advance(
            self.evidence.stage,
            self.evidence.phase,
            "interrupted repair resumed; verification still requires a real green check",
        )
    }

    /// Start a fresh cycle after a failed verification, bounded by the XI-4
    /// plan's hard round cap. A new red is required; the failed result cannot
    /// carry forward as a success.
    pub fn rediagnose(&mut self) -> Result<(), RepairError> {
        if self.evidence.result != Some(RepairResult::NotVerified) {
            return Err(RepairError::NotInProgress);
        }
        if let Some(cap) = self.evidence.round_cap {
            if self.evidence.attempts >= cap {
                return Err(RepairError::HardCapReached { cap });
            }
        }
        self.evidence.result = None;
        self.evidence.failure_reason = None;
        self.evidence.red = None;
        self.evidence.green = None;
        self.evidence.verdict = None;
        self.evidence.approval = None;
        self.evidence.repro = None;
        self.evidence.impl_command = None;
        self.evidence.write_recorded = false;
        let detail = self
            .evidence
            .recovery
            .map(|r| format!("loop back to diagnosis using {}", r.as_str()))
            .unwrap_or_else(|| "loop back to diagnosis".to_string());
        self.advance(PipelineStage::Diagnose, RepairPhase::AwaitingRed, detail)
    }

    /// The evidence record for this repair, suitable for atomic persistence.
    pub fn evidence(&self) -> RepairEvidence {
        self.evidence.clone()
    }

    /// Rebuild a repair from persisted evidence. A record that claims
    /// `Verified` but is missing any of the required real evidence is demoted
    /// to `NotVerified`: an incomplete record never reads as success.
    pub fn from_evidence(mut evidence: RepairEvidence) -> Result<Self, RepairError> {
        if evidence.result == Some(RepairResult::Verified) {
            let red_ok = evidence.red.as_ref().is_some_and(|r| !r.passed());
            let green_ok = evidence.green.as_ref().is_some_and(|r| r.passed());
            let verdict_ok = evidence.verdict.as_ref().is_some_and(|v| v.all_passed());
            let approval_ok = evidence.approval.as_ref().is_some_and(|a| a.granted());
            let independent = match (&evidence.impl_command, &evidence.green) {
                (Some(impl_command), Some(green)) => impl_command != &green.command,
                _ => false,
            };
            if !(red_ok
                && green_ok
                && verdict_ok
                && approval_ok
                && independent
                && evidence.repro.is_some()
                && evidence.write_recorded)
            {
                evidence.result = Some(RepairResult::NotVerified);
                evidence.failure_reason = Some(
                    "a stored Verified result lacked a required piece of real evidence".to_string(),
                );
            }
        }
        Ok(Self { evidence })
    }

    fn advance(
        &mut self,
        to_stage: PipelineStage,
        to_phase: RepairPhase,
        detail: impl Into<String>,
    ) -> Result<(), RepairError> {
        let from_stage = self.evidence.stage;
        let from_phase = self.evidence.phase;
        self.evidence.transitions.push(Transition {
            from_stage,
            to_stage,
            from_phase,
            to_phase,
            detail: detail.into(),
        });
        self.evidence.stage = to_stage;
        self.evidence.phase = to_phase;
        Ok(())
    }

    fn not_verified(&mut self, reason: String) -> Result<RepairResult, RepairError> {
        self.advance(
            PipelineStage::Diagnose,
            self.evidence.phase,
            format!("verification failed: {reason}"),
        )?;
        self.evidence.result = Some(RepairResult::NotVerified);
        self.evidence.failure_reason = Some(reason);
        // A failed verification never records a write, even when the check
        // itself was real.
        self.evidence.write_recorded = false;
        Ok(RepairResult::NotVerified)
    }
}

/// Atomic store for repair evidence and its transition history. Evidence is
/// written under `repairs/<id>.json`; transitions are kept as a tolerant JSONL
/// file so a torn tail does not hide the steps before it.
pub struct RepairStore {
    root: PathBuf,
}

impl RepairStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path(&self, id: &str) -> PathBuf {
        self.root.join("repairs").join(format!("{id}.json"))
    }

    fn history_path(&self, id: &str) -> PathBuf {
        self.root
            .join("repairs")
            .join(format!("{id}.transitions.jsonl"))
    }

    /// Save evidence atomically. A resumed repair overwrites its own record;
    /// the transition history only grows by the steps that are new.
    pub fn save(&self, evidence: &RepairEvidence) -> Result<(), RepairError> {
        let path = self.path(&evidence.repair_id);
        let bytes = serde_json::to_vec_pretty(evidence).map_err(|error| RepairError::Store {
            path: path.clone(),
            message: error.to_string(),
        })?;
        write_atomic(&path, &bytes).map_err(|error| RepairError::Store {
            path: path.clone(),
            message: error.to_string(),
        })?;

        let history_path = self.history_path(&evidence.repair_id);
        let existing = read_jsonl_tolerant::<Transition>(&history_path).rows;
        let added: Vec<&Transition> = evidence
            .transitions
            .iter()
            .filter(|t| !existing.contains(t))
            .collect();
        if !added.is_empty() {
            let mut all = String::new();
            for prior in &existing {
                let line = serde_json::to_string(prior).map_err(|error| RepairError::Store {
                    path: history_path.clone(),
                    message: error.to_string(),
                })?;
                all.push_str(&line);
                all.push('\n');
            }
            for transition in added {
                let mut json =
                    serde_json::to_string(transition).map_err(|error| RepairError::Store {
                        path: history_path.clone(),
                        message: error.to_string(),
                    })?;
                json = xencode_context_rs::trace::redact_secrets(&json);
                all.push_str(&json);
                all.push('\n');
            }
            write_atomic(&history_path, all.as_bytes()).map_err(|error| RepairError::Store {
                path: history_path.clone(),
                message: error.to_string(),
            })?;
        }
        Ok(())
    }

    /// Load evidence. A missing or unreadable file is an error, never an empty
    /// record that could be mistaken for a verified repair.
    pub fn load(&self, id: &str) -> Result<RepairEvidence, RepairError> {
        let path = self.path(id);
        let bytes = std::fs::read(&path).map_err(|error| RepairError::Store {
            path: path.clone(),
            message: error.to_string(),
        })?;
        serde_json::from_slice(&bytes).map_err(|error| RepairError::Store {
            path,
            message: error.to_string(),
        })
    }

    /// The transition history read with the same tolerant reader the other
    /// ledgers use.
    pub fn history(&self, id: &str) -> Vec<Transition> {
        read_jsonl_tolerant::<Transition>(&self.history_path(id)).rows
    }
}

/// Stable FNV-1a over the reproduction bytes, so a digest does not depend on
/// a randomized hash seed.
fn fnv1a(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}
