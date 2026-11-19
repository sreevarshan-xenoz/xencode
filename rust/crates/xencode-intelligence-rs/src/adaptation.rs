//! Bounded online adaptation from observed outcomes (XI-5).
//!
//! This is **policy adaptation**, not model training and not self-modification.
//! Nothing here changes a weight, edits source, or rewrites an existing agent
//! loop. A caller feeds the pure classifier typed records that refer to facts
//! another part of xencode already recorded (tool outcomes in `turns.jsonl`,
//! check results, run verification, an explicit human correction). The
//! classifier turns an eligible fact into an [`AdaptationObservation`], and the
//! updater folds observations into a small, bounded [`AdaptationPolicy`].
//!
//! The safety rules are enforced by types, not by good intentions:
//!
//! - **Observed facts only.** A model's own claim about its work
//!   ([`EvidenceRecord::ModelClaim`]) is always ineligible. An edit that was
//!   not checked on disk and a recovery that was not confirmed are refused as
//!   unverified. A check that never ran, or has no exit code, is refused as
//!   incomplete. An unrecognised tool outcome is refused as ambiguous.
//! - **No security learning at all.** The policy surface contains only a round
//!   allowance and a recovery hint. There is no field for local-only, cloud
//!   permission, network access, egress or verification, so nothing here can
//!   weaken a permission. A tool call that was denied or refused by policy is
//!   itself ineligible evidence.
//! - **Three explicit levels.** Level 1 ([`SessionAdaptation`]) is in-memory
//!   only and writes nothing. Level 2 ([`PolicyStore`] with [`Scope::Project`])
//!   is repo-scoped, opt-in, and every write carries provenance and can be
//!   rolled back to the exact earlier policy. Level 3 (`Scope::CrossSession`)
//!   is off by default and refuses without an explicit opt-in; a global policy
//!   carries no evidence references, so no private repository content leaves
//!   the repo.
//! - **Human corrections are a distinct signal** ([`SignalOrigin::HumanCorrection`])
//!   and are never confused with a machine-observed outcome.
//! - **Verification is independent of the implementing model.** Every decision
//!   here is a deterministic function of typed evidence and a closed set of
//!   rules; no model is asked to judge its own work.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use xencode_context_rs::{ChecksVerdict, RunEvidence, ToolTrace};
use xencode_core_rs::{read_jsonl_tolerant, write_atomic};

/// Smallest round allowance a policy may hand out. It is a floor, not a guess.
pub const MIN_TOOL_ROUNDS: u32 = 4;
/// Absolute ceiling a learned policy may never exceed.
pub const MAX_TOOL_ROUNDS: u32 = 16;
/// Bounded step added per distinct observed failure.
pub const ROUNDS_STEP: u32 = 2;
/// The fixed baseline used by the chronological-holdout comparison.
pub const DEFAULT_TOOL_ROUNDS: u32 = MIN_TOOL_ROUNDS;
/// File holding the current policy, written atomically.
pub const CURRENT_FILE: &str = "current.json";
/// Append-only provenance log, read with the tolerant JSONL reader.
pub const HISTORY_FILE: &str = "history.jsonl";

/// How far a learned policy may reach. Escalating scope is always an explicit
/// choice by the caller, never something an observation implies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Level 1: this conversation only, in memory, nothing written.
    Session,
    /// Level 2: one repository only, persisted with provenance and rollback.
    Project,
    /// Level 3: across sessions, off by default, requires an explicit opt-in.
    CrossSession,
}

impl Scope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Project => "project",
            Self::CrossSession => "cross_session",
        }
    }
}

/// Where a signal came from. A human correction is never folded into the
/// observed bucket: a reader must be able to tell the two apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalOrigin {
    /// Derived from a real record another part of xencode wrote.
    Observed,
    /// Typed by a person as an explicit correction.
    HumanCorrection,
}

/// A reference to a real record, not its content. Kept deliberately small so
/// a global policy can never carry a repository path or a transcript body.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// The store that owns the record, e.g. `turns.jsonl` or `runs.jsonl`.
    pub store: String,
    /// The id of the record inside that store (a run id, a tool index, …).
    pub record_id: String,
    /// Whether a person typed it or it was observed.
    pub origin: SignalOrigin,
}

impl EvidenceRef {
    pub fn new(
        store: impl Into<String>,
        record_id: impl Into<String>,
        origin: SignalOrigin,
    ) -> Self {
        Self {
            store: store.into(),
            record_id: record_id.into(),
            origin,
        }
    }
}

/// The status a tool row reported. Anything outside these four words is
/// unknown, and is refused rather than guessed at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutcomeStatus {
    Done,
    Failed,
    /// A permission decision. Never learnable: no policy can be learned from
    /// something a human or a rule refused.
    Denied,
    /// A capability refusal. Also not learnable.
    Refused,
    /// A word this controller does not understand.
    Unrecognized(String),
}

/// One typed record pointing at an already-recorded fact.
///
/// The caller builds these from the existing stores; this crate reads no
/// ledger on its own and owns no memory store. The `from_*` constructors show
/// the intended mapping from the real context types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvidenceRecord {
    /// A tool call, normally derived from a `TurnTrace` row.
    ToolOutcome {
        tool: String,
        outcome: ToolOutcomeStatus,
        evidence: EvidenceRef,
        session_id: String,
        project_root: PathBuf,
    },
    /// Whether an edit actually changed the working tree. `checked_on_disk`
    /// must be true or this is only a claim and is refused.
    EditOutcome {
        verified_changed: bool,
        checked_on_disk: bool,
        evidence: EvidenceRef,
        session_id: String,
        project_root: PathBuf,
    },
    /// A check that really ran, with its exit code. A skipped check has no
    /// exit code and never counts as a pass.
    CheckOutcome {
        name: String,
        ran: bool,
        exit_code: Option<i32>,
        evidence: EvidenceRef,
        session_id: String,
        project_root: PathBuf,
    },
    /// Whether a recovery worked. `confirmed` must be true or the success is
    /// unverified narrative and is refused.
    RecoveryOutcome {
        succeeded: bool,
        confirmed: bool,
        evidence: EvidenceRef,
        session_id: String,
        project_root: PathBuf,
    },
    /// An explicit human correction, tagged distinctly from observed signals.
    HumanCorrection {
        text: String,
        evidence: EvidenceRef,
        session_id: String,
        project_root: PathBuf,
    },
    /// What a model *said* about its own work. Never evidence.
    ModelClaim {
        text: String,
        evidence: EvidenceRef,
        session_id: String,
        project_root: PathBuf,
    },
}

impl EvidenceRecord {
    /// The reference shared by every variant, used in refusal messages.
    pub fn evidence(&self) -> &EvidenceRef {
        match self {
            Self::ToolOutcome { evidence, .. }
            | Self::EditOutcome { evidence, .. }
            | Self::CheckOutcome { evidence, .. }
            | Self::RecoveryOutcome { evidence, .. }
            | Self::HumanCorrection { evidence, .. }
            | Self::ModelClaim { evidence, .. } => evidence,
        }
    }

    /// Build a tool record from a real trace row. An outcome word this
    /// controller does not know stays [`ToolOutcomeStatus::Unrecognized`] and
    /// is refused by the classifier.
    pub fn from_tool_trace(
        tool: &ToolTrace,
        evidence: EvidenceRef,
        session_id: impl Into<String>,
        project_root: impl Into<PathBuf>,
    ) -> Self {
        let outcome = match tool.outcome.as_str() {
            "done" => ToolOutcomeStatus::Done,
            "failed" => ToolOutcomeStatus::Failed,
            "denied" => ToolOutcomeStatus::Denied,
            "refused" => ToolOutcomeStatus::Refused,
            other => ToolOutcomeStatus::Unrecognized(other.to_string()),
        };
        Self::ToolOutcome {
            tool: tool.name.clone(),
            outcome,
            evidence,
            session_id: session_id.into(),
            project_root: project_root.into(),
        }
    }

    /// Build check records from a real checks verdict. A check in `skipped`
    /// has no exit code here and is refused as incomplete; `failed` and `ran`
    /// are the only sources of an exit code.
    pub fn from_checks(
        verdict: &ChecksVerdict,
        evidence: EvidenceRef,
        session_id: impl Into<String>,
        project_root: impl Into<PathBuf>,
    ) -> Vec<Self> {
        let session_id = session_id.into();
        let project_root = project_root.into();
        let mut names: BTreeSet<&str> = BTreeSet::new();
        names.extend(verdict.ran.iter().map(String::as_str));
        names.extend(verdict.failed.iter().map(String::as_str));
        names
            .into_iter()
            .map(|name| {
                let failed = verdict.failed.iter().any(|n| n == name);
                let ran = verdict.ran.iter().any(|n| n == name) || failed;
                Self::CheckOutcome {
                    name: name.to_string(),
                    ran,
                    exit_code: if failed {
                        Some(1)
                    } else if ran {
                        Some(0)
                    } else {
                        None
                    },
                    evidence: EvidenceRef {
                        record_id: format!("{}:{name}", evidence.record_id),
                        ..evidence.clone()
                    },
                    session_id: session_id.clone(),
                    project_root: project_root.clone(),
                }
            })
            .collect()
    }

    /// Build a recovery record from a real run joined to its verification
    /// rows. Success without a check that actually ran is not confirmed.
    pub fn from_run_evidence(
        row: &RunEvidence,
        evidence: EvidenceRef,
        session_id: impl Into<String>,
        project_root: impl Into<PathBuf>,
    ) -> Self {
        Self::RecoveryOutcome {
            succeeded: row.verified(),
            confirmed: !row.checks.is_empty(),
            evidence,
            session_id: session_id.into(),
            project_root: project_root.into(),
        }
    }
}

/// The outcome a classifier derived from one eligible fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeKind {
    ToolSuccess,
    ToolFailure,
    EditSuccess,
    EditFailure,
    TestPassed,
    TestFailed,
    RecoverySuccess,
    RecoveryFailure,
    HumanCorrection,
}

impl OutcomeKind {
    pub fn is_failure(self) -> bool {
        matches!(
            self,
            Self::ToolFailure
                | Self::EditFailure
                | Self::TestFailed
                | Self::RecoveryFailure
                | Self::HumanCorrection
        )
    }
}

/// Why a record was refused. Each is a distinct, human-readable verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Eligibility {
    /// A model's own narration is never evidence.
    ModelNarrative,
    /// The work was claimed but not checked against reality.
    UnverifiedClaim,
    /// Conflicting or uninterpretable outcome.
    AmbiguousOutcome,
    /// A check that did not run or produced no exit code.
    IncompleteOutcome,
    /// A denial or refusal is a permission decision, not learnable evidence.
    PolicyDecision,
}

impl Eligibility {
    pub fn reason(self) -> &'static str {
        match self {
            Self::ModelNarrative => "a model's own claim about its work is not evidence",
            Self::UnverifiedClaim => "the claimed result was never confirmed against reality",
            Self::AmbiguousOutcome => "the outcome could not be interpreted unambiguously",
            Self::IncompleteOutcome => "the check did not run or produced no exit code",
            Self::PolicyDecision => {
                "a denial or refusal is a permission decision and cannot be learned from"
            }
        }
    }
}

/// One eligible observation, ready to be folded into a policy.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AdaptationObservation {
    /// Stable id; identical facts collapse to one observation, so repeating a
    /// failure cannot inflate a policy.
    pub id: String,
    pub scope: Scope,
    pub outcome: OutcomeKind,
    pub origin: SignalOrigin,
    pub evidence: Vec<EvidenceRef>,
    pub session_id: String,
    pub project_root: PathBuf,
}

/// A refused record and the exact reason. Kept so a caller can show *why*
/// nothing was learned, rather than silently dropping it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IneligibleObservation {
    pub reference: EvidenceRef,
    pub reason: Eligibility,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Classification {
    Eligible(AdaptationObservation),
    Ineligible(IneligibleObservation),
}

/// Classify one typed record. Pure: no file, clock or model is read.
///
/// `scope` is supplied by the caller: a record does not promote itself to a
/// wider scope. Permission to *persist* that scope is checked later by
/// [`PolicyStore`] and [`update_policy`].
pub fn classify(record: &EvidenceRecord, scope: Scope) -> Classification {
    let (outcome, origin) = match record {
        EvidenceRecord::ModelClaim { .. } => {
            return refused(record, Eligibility::ModelNarrative);
        }
        EvidenceRecord::ToolOutcome { outcome, .. } => match outcome {
            ToolOutcomeStatus::Done => (OutcomeKind::ToolSuccess, SignalOrigin::Observed),
            ToolOutcomeStatus::Failed => (OutcomeKind::ToolFailure, SignalOrigin::Observed),
            ToolOutcomeStatus::Denied | ToolOutcomeStatus::Refused => {
                return refused(record, Eligibility::PolicyDecision);
            }
            ToolOutcomeStatus::Unrecognized(_) => {
                return refused(record, Eligibility::AmbiguousOutcome);
            }
        },
        EvidenceRecord::EditOutcome {
            verified_changed,
            checked_on_disk,
            ..
        } => {
            if !checked_on_disk {
                return refused(record, Eligibility::UnverifiedClaim);
            }
            (
                if *verified_changed {
                    OutcomeKind::EditSuccess
                } else {
                    OutcomeKind::EditFailure
                },
                SignalOrigin::Observed,
            )
        }
        EvidenceRecord::CheckOutcome { ran, exit_code, .. } => {
            if !ran || exit_code.is_none() {
                return refused(record, Eligibility::IncompleteOutcome);
            }
            let code = exit_code.unwrap_or(0);
            (
                if code == 0 {
                    OutcomeKind::TestPassed
                } else {
                    OutcomeKind::TestFailed
                },
                SignalOrigin::Observed,
            )
        }
        EvidenceRecord::RecoveryOutcome {
            succeeded,
            confirmed,
            ..
        } => {
            if !confirmed {
                return refused(record, Eligibility::UnverifiedClaim);
            }
            (
                if *succeeded {
                    OutcomeKind::RecoverySuccess
                } else {
                    OutcomeKind::RecoveryFailure
                },
                SignalOrigin::Observed,
            )
        }
        EvidenceRecord::HumanCorrection { .. } => {
            (OutcomeKind::HumanCorrection, SignalOrigin::HumanCorrection)
        }
    };

    // The origin in the caller's reference is not trusted: it is rebuilt from
    // the record's variant, so a model claim cannot masquerade as a correction.
    let evidence = vec![EvidenceRef {
        origin,
        ..record.evidence().clone()
    }];
    let id = stable_id(&(
        scope.as_str(),
        outcome as u8 as char,
        origin.as_str(),
        &evidence,
    ));

    Classification::Eligible(AdaptationObservation {
        id,
        scope,
        outcome,
        origin,
        evidence,
        session_id: record.session_id().to_string(),
        project_root: record.project_root().to_path_buf(),
    })
}

fn refused(record: &EvidenceRecord, reason: Eligibility) -> Classification {
    Classification::Ineligible(IneligibleObservation {
        reference: record.evidence().clone(),
        reason,
    })
}

impl EvidenceRecord {
    fn session_id(&self) -> &str {
        match self {
            Self::ToolOutcome { session_id, .. }
            | Self::EditOutcome { session_id, .. }
            | Self::CheckOutcome { session_id, .. }
            | Self::RecoveryOutcome { session_id, .. }
            | Self::HumanCorrection { session_id, .. }
            | Self::ModelClaim { session_id, .. } => session_id,
        }
    }

    fn project_root(&self) -> &Path {
        match self {
            Self::ToolOutcome { project_root, .. }
            | Self::EditOutcome { project_root, .. }
            | Self::CheckOutcome { project_root, .. }
            | Self::RecoveryOutcome { project_root, .. }
            | Self::HumanCorrection { project_root, .. }
            | Self::ModelClaim { project_root, .. } => project_root,
        }
    }
}

impl SignalOrigin {
    fn as_str(self) -> &'static str {
        match self {
            Self::Observed => "observed",
            Self::HumanCorrection => "human_correction",
        }
    }
}

/// A deterministic, stable 64-bit id over canonical bytes. Not a signature;
/// it must only stay the same across builds, which `DefaultHasher` need not.
fn stable_id<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("observation inputs must serialize");
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The only recoveries a policy may choose. All are deterministic rules; none
/// routes a model, changes a permission or edits a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryHint {
    Proceed,
    CorrectInvocation,
    StopRepeating,
    StructuredDiagnostics,
    AffectedDependencies,
    RequestMoreContext,
}

impl RecoveryHint {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Proceed => "proceed",
            Self::CorrectInvocation => "correct_invocation",
            Self::StopRepeating => "stop_repeating",
            Self::StructuredDiagnostics => "structured_diagnostics",
            Self::AffectedDependencies => "affected_dependencies",
            Self::RequestMoreContext => "request_more_context",
        }
    }
}

/// The complete learnable surface. There is deliberately no field here for
/// permissions, egress, cloud access, network use or verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptationParameters {
    /// Hard-capped round allowance, never above [`MAX_TOOL_ROUNDS`].
    pub tool_rounds: u32,
    /// The deterministic recovery rule to try next.
    pub recovery: RecoveryHint,
}

/// Lifecycle of a stored policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyStatus {
    Active,
    Superseded,
}

/// A bounded, inspectable policy. `version`, scope, parameters, evidence
/// provenance and status are all part of the value, so two equal policies are
/// equal in every respect and a rollback can restore one exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptationPolicy {
    pub version: u64,
    pub scope: Scope,
    /// `Some` only for a project policy, and always bound to one repository.
    pub project_root: Option<PathBuf>,
    pub parameters: AdaptationParameters,
    /// Evidence references behind this policy. Empty for a global policy so
    /// no private repository content can leave the repo.
    pub based_on: Vec<EvidenceRef>,
    pub status: PolicyStatus,
}

/// What the caller permits. Session learning is always in memory and needs no
/// opt-in; both persistent scopes default to off.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningOptions {
    pub allow_project: bool,
    pub allow_cross_session: bool,
}

impl LearningOptions {
    pub fn session_only() -> Self {
        Self::default()
    }

    /// Enable repo-scoped persistence (level 2).
    pub fn with_project(mut self) -> Self {
        self.allow_project = true;
        self
    }

    /// Enable global persistence (level 3). Still writes no private content.
    pub fn with_cross_session(mut self) -> Self {
        self.allow_cross_session = true;
        self
    }
}

#[derive(Debug)]
pub enum AdaptationError {
    /// The caller asked for a scope they have not opted into.
    ScopeNotPermitted { scope: Scope },
    /// A project policy was used with no repository, or against a different one.
    RepoBoundary {
        expected: PathBuf,
        found: Option<PathBuf>,
    },
    /// No eligible observation was supplied.
    EmptyObservations,
    /// An observation came from a different repository than the policy claims.
    ObservationOutsideRepo { expected: PathBuf, found: PathBuf },
    /// A persistent write needs a named actor and a reason.
    MissingProvenance,
    /// A session policy must never be persisted.
    SessionNotPersistent,
    /// A global policy carried evidence references, which is not allowed.
    GlobalCarriedEvidence,
    /// A rollback was asked for a version that is not in the history.
    UnknownVersion { version: u64 },
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

impl fmt::Display for AdaptationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScopeNotPermitted { scope } => write!(
                f,
                "{} adaptation is not permitted without an explicit opt-in",
                scope.as_str()
            ),
            Self::RepoBoundary { expected, found } => write!(
                f,
                "this policy is bound to {} and cannot be used with {}",
                expected.display(),
                found
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "no repository".to_string())
            ),
            Self::EmptyObservations => f.write_str("no eligible observation was supplied"),
            Self::ObservationOutsideRepo { expected, found } => write!(
                f,
                "an observation from {} was used for a policy bound to {}",
                found.display(),
                expected.display()
            ),
            Self::MissingProvenance => {
                f.write_str("a persistent policy change needs a named actor and a reason")
            }
            Self::SessionNotPersistent => {
                f.write_str("a session policy is in-memory only and cannot be persisted")
            }
            Self::GlobalCarriedEvidence => f.write_str(
                "a global policy carried evidence references, which could contain private repository content",
            ),
            Self::UnknownVersion { version } => {
                write!(f, "policy version {version} is not in the history")
            }
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

impl std::error::Error for AdaptationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } | Self::Write { source, .. } => Some(source),
            Self::Decode { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The recovery a given outcome calls for. A rule, not a learned weight: it
/// never widens a permission and never lowers verification.
fn recovery_for(outcome: OutcomeKind) -> RecoveryHint {
    match outcome {
        OutcomeKind::ToolFailure => RecoveryHint::CorrectInvocation,
        OutcomeKind::EditFailure => RecoveryHint::StructuredDiagnostics,
        OutcomeKind::TestFailed => RecoveryHint::AffectedDependencies,
        OutcomeKind::RecoveryFailure => RecoveryHint::StopRepeating,
        OutcomeKind::HumanCorrection => RecoveryHint::RequestMoreContext,
        OutcomeKind::ToolSuccess
        | OutcomeKind::EditSuccess
        | OutcomeKind::TestPassed
        | OutcomeKind::RecoverySuccess => RecoveryHint::Proceed,
    }
}

/// Fold eligible observations into a bounded policy. Pure and deterministic:
/// the same observations, scope, previous version and options always give the
/// same policy. The version advances by one from `previous`.
pub fn update_policy(
    previous: Option<&AdaptationPolicy>,
    observations: &[AdaptationObservation],
    scope: Scope,
    project_root: Option<PathBuf>,
    options: &LearningOptions,
) -> Result<AdaptationPolicy, AdaptationError> {
    if scope == Scope::CrossSession && !options.allow_cross_session {
        return Err(AdaptationError::ScopeNotPermitted { scope });
    }
    if scope == Scope::Project && !options.allow_project {
        return Err(AdaptationError::ScopeNotPermitted { scope });
    }
    if scope == Scope::Project && project_root.is_none() {
        return Err(AdaptationError::RepoBoundary {
            expected: PathBuf::from("<this repository>"),
            found: None,
        });
    }
    if scope == Scope::Project {
        let bound = project_root.as_deref().expect("checked just above");
        for observation in observations {
            if !same_path(&observation.project_root, bound) {
                return Err(AdaptationError::ObservationOutsideRepo {
                    expected: bound.to_path_buf(),
                    found: observation.project_root.clone(),
                });
            }
        }
    }
    if observations.is_empty() {
        return Err(AdaptationError::EmptyObservations);
    }

    // Identical observations (same stable id) count once, so repeating a
    // failure cannot make the policy grow.
    let mut distinct: BTreeSet<&AdaptationObservation> = BTreeSet::new();
    let mut order: Vec<&AdaptationObservation> = Vec::new();
    for observation in observations {
        if distinct.insert(observation) {
            order.push(observation);
        }
    }

    let mut counts: std::collections::BTreeMap<OutcomeKind, usize> =
        std::collections::BTreeMap::new();
    for observation in &order {
        if observation.outcome.is_failure() {
            *counts.entry(observation.outcome).or_insert(0) += 1;
        }
    }
    let dominant = counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)))
        .map(|(kind, _)| kind);
    let recovery = dominant.map(recovery_for).unwrap_or(RecoveryHint::Proceed);
    let failures: u32 = order
        .iter()
        .filter(|o| o.outcome.is_failure())
        .count()
        .try_into()
        .unwrap_or(u32::MAX);
    let tool_rounds = DEFAULT_TOOL_ROUNDS
        .saturating_add(ROUNDS_STEP.saturating_mul(failures))
        .min(MAX_TOOL_ROUNDS);

    let mut based_on: Vec<EvidenceRef> = order
        .iter()
        .flat_map(|observation| observation.evidence.iter().cloned())
        .collect();
    based_on.sort();
    based_on.dedup();
    if scope == Scope::CrossSession {
        // A global record carries no evidence references: no run id, tool
        // row, file or repository path leaves the project.
        based_on.clear();
    }

    let version = previous.map_or(1, |p| p.version.saturating_add(1));
    let project_root = match scope {
        Scope::Project => project_root,
        Scope::Session | Scope::CrossSession => None,
    };
    Ok(AdaptationPolicy {
        version,
        scope,
        project_root,
        parameters: AdaptationParameters {
            tool_rounds,
            recovery,
        },
        based_on,
        status: PolicyStatus::Active,
    })
}

/// Level 1: session adaptation. Immediate, in-memory, and persists nothing.
/// Repeating the same fact is a no-op, so a failure loop cannot grow a policy.
pub struct SessionAdaptation {
    session_id: String,
    project_root: PathBuf,
    seen: BTreeSet<String>,
    observations: Vec<AdaptationObservation>,
    policy: AdaptationPolicy,
}

impl SessionAdaptation {
    pub fn new(session_id: impl Into<String>, project_root: impl Into<PathBuf>) -> Self {
        let session_id = session_id.into();
        let project_root = project_root.into();
        let policy = AdaptationPolicy {
            version: 0,
            scope: Scope::Session,
            project_root: None,
            parameters: AdaptationParameters {
                tool_rounds: DEFAULT_TOOL_ROUNDS,
                recovery: RecoveryHint::Proceed,
            },
            based_on: Vec::new(),
            status: PolicyStatus::Active,
        };
        Self {
            session_id,
            project_root,
            seen: BTreeSet::new(),
            observations: Vec::new(),
            policy,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    /// Offer one record. Ineligible records are returned as such and change
    /// nothing; an identical eligible record is counted once.
    pub fn observe(&mut self, record: &EvidenceRecord) -> Classification {
        let classification = classify(record, Scope::Session);
        if let Classification::Eligible(observation) = &classification {
            if self.seen.insert(observation.id.clone()) {
                self.observations.push(observation.clone());
                // Session scope needs no opt-in and cannot fail here: it is a
                // pure function of eligible observations already collected.
                self.policy = update_policy(
                    None,
                    &self.observations,
                    Scope::Session,
                    None,
                    &LearningOptions::session_only(),
                )
                .expect("a session policy is always computable from eligible observations");
            }
        }
        classification
    }

    /// The current in-memory policy. Version 0 means nothing eligible seen.
    pub fn policy(&self) -> &AdaptationPolicy {
        &self.policy
    }

    /// A deterministic decision from the current policy. No model is called
    /// and no file is touched.
    pub fn decide(&self) -> AdaptationDecision {
        AdaptationDecision {
            recovery: self.policy.parameters.recovery,
            tool_rounds: self.policy.parameters.tool_rounds,
            scope: Scope::Session,
            policy_version: self.policy.version,
        }
    }
}

/// The decision a caller may act on. Its fields are all bounded values from a
/// policy; there is no permission or egress field here either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptationDecision {
    pub recovery: RecoveryHint,
    pub tool_rounds: u32,
    pub scope: Scope,
    pub policy_version: u64,
}

/// Who wrote a persistent policy and when. Combined with the policy's
/// `based_on` references, this is the provenance of a stored change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub actor: String,
    pub written_at_ms: u64,
    pub note: String,
}

/// One row of the append-only policy history. `sequence` is assigned by the
/// store and is unique even across a rollback; `policy.version` is the
/// policy's own version and may legitimately repeat after a rollback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyEnvelope {
    pub sequence: u64,
    pub policy: AdaptationPolicy,
    pub provenance: Provenance,
}

/// Level 2 and 3 persistence, under a caller-supplied directory.
///
/// Every write is provenance-bearing, append-only in history, and reversible
/// via [`Self::rollback`]. A project store is bound to one repository and
/// refuses a policy for any other. A global store refuses outright unless
/// [`LearningOptions::allow_cross_session`] was set by the caller.
pub struct PolicyStore {
    root: PathBuf,
    project_root: PathBuf,
    options: LearningOptions,
}

impl PolicyStore {
    pub fn new(
        root: impl Into<PathBuf>,
        project_root: impl Into<PathBuf>,
        options: LearningOptions,
    ) -> Self {
        Self {
            root: root.into(),
            project_root: project_root.into(),
            options,
        }
    }

    pub fn current_path(&self) -> PathBuf {
        self.root.join("policies").join(CURRENT_FILE)
    }

    pub fn history_path(&self) -> PathBuf {
        self.root.join("policies").join(HISTORY_FILE)
    }

    fn check_provenance(provenance: &Provenance) -> Result<(), AdaptationError> {
        if provenance.actor.trim().is_empty() || provenance.note.trim().is_empty() {
            return Err(AdaptationError::MissingProvenance);
        }
        Ok(())
    }

    fn check_policy(&self, policy: &AdaptationPolicy) -> Result<(), AdaptationError> {
        match policy.scope {
            Scope::Session => Err(AdaptationError::SessionNotPersistent),
            Scope::Project => {
                if !self.options.allow_project {
                    return Err(AdaptationError::ScopeNotPermitted {
                        scope: Scope::Project,
                    });
                }
                match &policy.project_root {
                    Some(root) if same_path(root, &self.project_root) => Ok(()),
                    found => Err(AdaptationError::RepoBoundary {
                        expected: self.project_root.clone(),
                        found: found.clone(),
                    }),
                }
            }
            Scope::CrossSession => {
                if !self.options.allow_cross_session {
                    return Err(AdaptationError::ScopeNotPermitted {
                        scope: Scope::CrossSession,
                    });
                }
                if policy.project_root.is_some() || !policy.based_on.is_empty() {
                    return Err(AdaptationError::GlobalCarriedEvidence);
                }
                Ok(())
            }
        }
    }

    /// Read the current policy, if a write has happened yet.
    pub fn read_current(&self) -> Option<AdaptationPolicy> {
        let bytes = std::fs::read(self.current_path()).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Read the append-only history. A torn trailing line is dropped by the
    /// tolerant reader; complete rows before it survive.
    pub fn read_history(&self) -> Vec<PolicyEnvelope> {
        read_jsonl_tolerant::<PolicyEnvelope>(&self.history_path()).rows
    }

    /// Append a new policy to history and make it current atomically.
    pub fn write(
        &self,
        policy: &AdaptationPolicy,
        provenance: &Provenance,
    ) -> Result<(), AdaptationError> {
        Self::check_provenance(provenance)?;
        self.check_policy(policy)?;
        if let Some(current) = self.read_current() {
            if policy.version <= current.version && policy.scope == current.scope {
                // A new write must advance the policy version; rollback is the
                // only operation that intentionally restores an older one.
                return Err(AdaptationError::UnknownVersion {
                    version: policy.version,
                });
            }
        }
        let sequence = self.read_history().len() as u64 + 1;
        let envelope = PolicyEnvelope {
            sequence,
            policy: policy.clone(),
            provenance: provenance.clone(),
        };
        let mut line = serde_json::to_vec(&envelope).map_err(|source| AdaptationError::Decode {
            path: self.history_path(),
            source,
        })?;
        line.push(b'\n');
        append_line(&self.history_path(), &line)?;
        let current =
            serde_json::to_vec_pretty(policy).map_err(|source| AdaptationError::Decode {
                path: self.current_path(),
                source,
            })?;
        write_atomic(&self.current_path(), &current).map_err(|source| AdaptationError::Write {
            path: self.current_path(),
            source,
        })
    }

    /// Restore a previous policy exactly: the current file becomes a
    /// byte-for-byte serialization of that historical policy, including its
    /// version, parameters, references and status. The rollback itself is
    /// appended to history as a new, auditable row.
    pub fn rollback(
        &self,
        version: u64,
        provenance: &Provenance,
    ) -> Result<AdaptationPolicy, AdaptationError> {
        Self::check_provenance(provenance)?;
        let row = self
            .read_history()
            .into_iter()
            .find(|row| row.policy.version == version)
            .ok_or(AdaptationError::UnknownVersion { version })?;
        self.check_policy(&row.policy)?;
        let sequence = self.read_history().len() as u64 + 1;
        let envelope = PolicyEnvelope {
            sequence,
            policy: row.policy.clone(),
            provenance: provenance.clone(),
        };
        let mut line = serde_json::to_vec(&envelope).map_err(|source| AdaptationError::Decode {
            path: self.history_path(),
            source,
        })?;
        line.push(b'\n');
        append_line(&self.history_path(), &line)?;
        let restored =
            serde_json::to_vec_pretty(&row.policy).map_err(|source| AdaptationError::Decode {
                path: self.current_path(),
                source,
            })?;
        write_atomic(&self.current_path(), &restored).map_err(|source| AdaptationError::Write {
            path: self.current_path(),
            source,
        })?;
        Ok(row.policy)
    }
}

fn append_line(path: &Path, line: &[u8]) -> Result<(), AdaptationError> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| AdaptationError::Write {
            path: path.to_path_buf(),
            source,
        })?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|source| AdaptationError::Write {
            path: path.to_path_buf(),
            source,
        })?;
    file.write_all(line)
        .map_err(|source| AdaptationError::Write {
            path: path.to_path_buf(),
            source,
        })
}

/// Compare two repo paths without requiring them to exist yet. Canonicalising
/// an existing path removes `..` and `./` differences; a path that does not
/// exist yet is compared as written.
fn same_path(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => left == right,
    }
}

/// One event in a chronological fixture: an observed fact plus the recovery
/// and round allowance that actually resolve it, recorded independently of
/// the learning rule so the evaluation is not circular.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureEvent {
    pub record: EvidenceRecord,
    pub expected_recovery: RecoveryHint,
    pub needs_rounds: u32,
}

/// Measured result of a chronological holdout comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoldoutReport {
    pub holdout_total: usize,
    pub baseline_hits: usize,
    pub adaptive_hits: usize,
    pub baseline_recovery_hits: usize,
    pub adaptive_recovery_hits: usize,
    pub baseline_round_hits: usize,
    pub adaptive_round_hits: usize,
    pub adaptive_parameters: AdaptationParameters,
}

impl HoldoutReport {
    /// The release-gate question, answered from measured counts only.
    pub fn adaptation_outperforms_baseline(&self) -> bool {
        self.adaptive_hits > self.baseline_hits
    }
}

/// Compare a fixed baseline against session adaptation, splitting strictly by
/// time: every training event precedes every held-out event. There is no
/// random split of one session, so a later outcome cannot leak into training.
pub fn evaluate_holdout(train: &[FixtureEvent], holdout: &[FixtureEvent]) -> HoldoutReport {
    let mut session = SessionAdaptation::new("holdout-train", PathBuf::from("fixture"));
    for event in train {
        session.observe(&event.record);
    }
    let adaptive = session.decide();
    let baseline = AdaptationParameters {
        tool_rounds: DEFAULT_TOOL_ROUNDS,
        recovery: RecoveryHint::Proceed,
    };

    let mut report = HoldoutReport {
        holdout_total: holdout.len(),
        baseline_hits: 0,
        adaptive_hits: 0,
        baseline_recovery_hits: 0,
        adaptive_recovery_hits: 0,
        baseline_round_hits: 0,
        adaptive_round_hits: 0,
        adaptive_parameters: session.policy().parameters,
    };
    for event in holdout {
        let base_recovery = baseline.recovery == event.expected_recovery;
        let adaptive_recovery = adaptive.recovery == event.expected_recovery;
        let base_rounds = baseline.tool_rounds >= event.needs_rounds;
        let adaptive_rounds = adaptive.tool_rounds >= event.needs_rounds;
        report.baseline_recovery_hits += usize::from(base_recovery);
        report.adaptive_recovery_hits += usize::from(adaptive_recovery);
        report.baseline_round_hits += usize::from(base_rounds);
        report.adaptive_round_hits += usize::from(adaptive_rounds);
        report.baseline_hits += usize::from(base_recovery && base_rounds);
        report.adaptive_hits += usize::from(adaptive_recovery && adaptive_rounds);
    }
    report
}
