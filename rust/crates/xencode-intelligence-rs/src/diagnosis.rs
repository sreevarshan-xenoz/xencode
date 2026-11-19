//! Evidence-driven failure diagnosis (XI-6): hypotheses with real falsifiers.
//!
//! A diagnosis here is never a verdict by a model. It is a small set of
//! candidate explanations built from failures that
//! [`xencode_analysis_rs::classify_command_output`] already classified out of a
//! command's real output, each paired with the *one command that would kill it*.
//! A hypothesis starts [`HypothesisStatus::Proposed`] and cannot move at all
//! until a real process has been run and its actual stdout, stderr and exit code
//! have been captured into a [`CheckExecution`]. There is deliberately no
//! `Confirmed` status: surviving one check means "not killed yet", never
//! "proved true"; a model's explanation of its own failure is never a cause.
//!
//! The command a falsifier names is one the repository already runs —
//! [`CARGO_CHECK`] and [`CARGO_TEST`] — so a caller can execute it with
//! [`CheckExecution::run`] instead of inventing a result. The engine itself is
//! pure: it proposes and interprets, it never spawns a process and never reads
//! a check's outcome from the caller's prose.
//!
//! **On the plan's "reuse AI-2" wording.** The plan row for XI-6 said to reuse
//! an AI-2 hypothesis architecture. No such architecture exists in the tree:
//! there is no `Hypothesis` type anywhere in `rust/crates`, and `AI-2` appears
//! only as a future research-pool idea in `NEXT_PLAN_TASKS.md`, never as an
//! implemented row or type. This module is therefore the genuine, minimal
//! substrate a future AI-2 could build on. It does not claim AI-2 exists, and
//! it does not duplicate a second hypothesis engine.

use crate::reasoning::RecoveryStrategy;
use crate::types::{sources, DecisionEvidence};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use xencode_analysis_rs::{classify_command_output, Category, FailureDiagnostic};
use xencode_core_rs::write_atomic;

/// The real build check this repository already runs (see `xencode-analysis-rs`
/// and `xencode-core-rs::rustc_json`). A falsifier names this, not a made-up
/// command.
pub const CARGO_CHECK: &str = "cargo check";
/// The real test check this repository already runs.
pub const CARGO_TEST: &str = "cargo test";
/// At most three candidate causes are proposed for one failure set.
pub const MAX_HYPOTHESES: usize = 3;

/// Identifier for one hypothesis inside a diagnosis.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HypothesisId(pub String);

impl HypothesisId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HypothesisId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for HypothesisId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl From<&str> for HypothesisId {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

/// A command that really ran, with everything it actually wrote and its real
/// exit code.
///
/// Instances are produced by [`CheckExecution::run`], which executes a process.
/// A status of `Survived` or `Falsified` is reachable only from one of these, so
/// a check that never ran cannot move a hypothesis and a failed check keeps its
/// real output. The type is serializable so a report can be stored; a stored
/// record is only as trustworthy as the run that produced it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckExecution {
    /// The command label, which must equal the falsifier's command.
    pub command: String,
    /// The process's real exit code. `-1` only when a signal gave no code.
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl CheckExecution {
    /// Execute a real command and capture its actual output and exit code.
    ///
    /// `label` records the command being checked; the caller passes a prepared
    /// [`std::process::Command`] (for example `cargo check` in a fixture
    /// directory) so no process is started by the pure proposal engine.
    pub fn run(label: &str, command: &mut std::process::Command) -> std::io::Result<Self> {
        let output = command.output()?;
        Ok(Self {
            command: label.to_string(),
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }

    /// Passed means the process really exited zero. Nothing else counts.
    pub fn passed(&self) -> bool {
        self.exit_code == 0
    }

    /// stdout and stderr kept together, in the order a caller can store them.
    pub fn combined_output(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }
}

/// The observation that kills a hypothesis, once the named command has really
/// been run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FalsifierKind {
    /// The hypothesis dies when this fragment appears in the real output.
    Appears { fragment: String },
    /// The hypothesis dies when this fragment is no longer in the real output.
    Disappears { fragment: String },
    /// The hypothesis dies when the real command exits zero.
    ExitZero,
}

impl FalsifierKind {
    fn expected_observation(&self) -> String {
        match self {
            Self::Appears { fragment } => format!("the output contains {fragment:?}"),
            Self::Disappears { fragment } => format!("the output no longer contains {fragment:?}"),
            Self::ExitZero => "the command exits zero".to_string(),
        }
    }

    fn seen(&self, execution: &CheckExecution) -> bool {
        let output = execution.combined_output();
        match self {
            Self::Appears { fragment } => output.contains(fragment),
            Self::Disappears { fragment } => !output.contains(fragment),
            Self::ExitZero => execution.passed(),
        }
    }
}

/// A real command plus the concrete observation that would disprove the claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Falsifier {
    /// A command the repository already runs, e.g. `cargo check`.
    pub command: String,
    pub kind: FalsifierKind,
    /// Why seeing that observation kills the claim, in plain English.
    pub rationale: String,
}

impl Falsifier {
    /// The expected observation as a readable sentence.
    pub fn expected_observation(&self) -> String {
        format!("{}: {}", self.command, self.kind.expected_observation())
    }

    /// Whether the falsifying observation happened in a real execution. A check
    /// of a different command is never accepted as this falsifier's check.
    pub fn observation_seen(&self, execution: &CheckExecution) -> bool {
        execution.command == self.command && self.kind.seen(execution)
    }
}

/// Where a hypothesis stands. There is no `Confirmed` variant on purpose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum HypothesisStatus {
    /// Proposed from classified failures; no command has been run for it yet.
    Proposed,
    /// A real command ran and its output is stored. `passed` means the
    /// falsifying observation was seen, so the claim is about to be killed.
    Checked {
        passed: bool,
        execution: CheckExecution,
    },
    /// Checked, and the falsifying observation was *not* seen. This is not a
    /// confirmation: the claim merely survived one real check.
    Survived { execution: CheckExecution },
    /// Checked, and the falsifying observation was seen. The claim is dead.
    Falsified { execution: CheckExecution },
}

impl HypothesisStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Proposed => "proposed",
            Self::Checked { passed, .. } => {
                if *passed {
                    "checked (falsifying observation seen)"
                } else {
                    "checked (falsifying observation absent)"
                }
            }
            Self::Survived { .. } => "survived",
            Self::Falsified { .. } => "falsified",
        }
    }

    /// The captured real check, present in every state after `Proposed`.
    pub fn execution(&self) -> Option<&CheckExecution> {
        match self {
            Self::Proposed => None,
            Self::Checked { execution, .. }
            | Self::Survived { execution }
            | Self::Falsified { execution } => Some(execution),
        }
    }
}

/// Why a status transition was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosisError {
    /// A check was attempted on a hypothesis that had already been checked.
    NotProposed,
    /// A conclusion was attempted without a real execution behind it.
    NotChecked,
    /// The execution was of a different command than the falsifier names.
    CommandMismatch { expected: String, actual: String },
    /// A report with this diagnosis id was already stored.
    DuplicateDiagnosis(String),
    /// No stored report carries this id.
    NotFound(String),
    /// A store read/write failed.
    Write { path: PathBuf, message: String },
}

impl fmt::Display for DiagnosisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotProposed => f.write_str("the hypothesis was already checked"),
            Self::NotChecked => {
                f.write_str("no real check was recorded, so the status cannot change")
            }
            Self::CommandMismatch { expected, actual } => write!(
                f,
                "the falsifier names {expected:?} but the recorded check ran {actual:?}"
            ),
            Self::DuplicateDiagnosis(id) => write!(f, "a diagnosis with id {id} is already stored"),
            Self::NotFound(id) => write!(f, "no stored diagnosis has id {id}"),
            Self::Write { path, message } => {
                write!(f, "could not use {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for DiagnosisError {}

/// One candidate cause, tied to a classified failure and to the check that
/// would kill it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hypothesis {
    pub id: HypothesisId,
    /// The explanation, in words a reader can argue with.
    pub claim: String,
    pub falsifier: Falsifier,
    pub status: HypothesisStatus,
    /// The real classified failure this candidate tries to explain.
    pub diagnostic: FailureDiagnostic,
    /// Evidence behind the proposal, including the classified failure itself.
    pub evidence: Vec<DecisionEvidence>,
}

impl Hypothesis {
    /// Propose a hypothesis from one classified failure. No process is run and
    /// the status is always `Proposed`.
    pub fn propose(
        id: HypothesisId,
        diagnostic: FailureDiagnostic,
        evidence: &[DecisionEvidence],
    ) -> Self {
        let (command, kind, why) = falsifier_for(&diagnostic);
        let claim = format!(
            "{} at {} — {}",
            diagnostic.category.label(),
            diagnostic.location(),
            diagnostic.message
        );
        let mut own_evidence = vec![DecisionEvidence::new(
            sources::CLASSIFIED_FAILURE,
            format!(
                "{} at {}: {}; suggested next step: {}",
                diagnostic.category.label(),
                diagnostic.location(),
                diagnostic.message,
                diagnostic.suggested_action
            ),
        )];
        own_evidence.extend_from_slice(evidence);
        Self {
            id,
            claim,
            falsifier: Falsifier {
                command: command.to_string(),
                kind,
                rationale: why,
            },
            status: HypothesisStatus::Proposed,
            diagnostic,
            evidence: own_evidence,
        }
    }

    /// Record a real check: `Proposed` becomes `Checked`. The execution's
    /// captured output and exit code are stored on the status itself.
    pub fn check(self, execution: CheckExecution) -> Result<Self, DiagnosisError> {
        let Self {
            id,
            claim,
            falsifier,
            status,
            diagnostic,
            evidence,
        } = self;
        if status != HypothesisStatus::Proposed {
            return Err(DiagnosisError::NotProposed);
        }
        if execution.command != falsifier.command {
            return Err(DiagnosisError::CommandMismatch {
                expected: falsifier.command.clone(),
                actual: execution.command.clone(),
            });
        }
        let passed = falsifier.observation_seen(&execution);
        Ok(Self {
            id,
            claim,
            falsifier,
            status: HypothesisStatus::Checked { passed, execution },
            diagnostic,
            evidence,
        })
    }

    /// Conclude a recorded check: `Checked` becomes `Survived` or `Falsified`.
    /// This is the only route to either terminal status, and it cannot be
    /// reached from `Proposed`.
    pub fn conclude(self) -> Result<Self, DiagnosisError> {
        let Self {
            id,
            claim,
            falsifier,
            status,
            diagnostic,
            evidence,
        } = self;
        let status = match status {
            HypothesisStatus::Checked { passed, execution } => {
                if passed {
                    HypothesisStatus::Falsified { execution }
                } else {
                    HypothesisStatus::Survived { execution }
                }
            }
            _ => return Err(DiagnosisError::NotChecked),
        };
        Ok(Self {
            id,
            claim,
            falsifier,
            status,
            diagnostic,
            evidence,
        })
    }

    /// Convenience for the full real-check transition in one call.
    pub fn check_and_conclude(self, execution: CheckExecution) -> Result<Self, DiagnosisError> {
        self.check(execution)?.conclude()
    }

    /// The recovery rule a caller should consider if this hypothesis survived.
    pub fn recovery_hint(&self) -> RecoveryStrategy {
        recovery_strategy(self.diagnostic.category)
    }
}

/// The deterministic proposal engine. It classifies real output (through the
/// EC-1 classifier) and names candidates; it never runs or invents a check.
#[derive(Debug, Default)]
pub struct DiagnosisEngine {
    next: usize,
}

impl DiagnosisEngine {
    pub fn new() -> Self {
        Self { next: 1 }
    }

    /// Propose up to [`MAX_HYPOTHESES`] causes from already-classified failures.
    pub fn propose(
        &mut self,
        failures: &[FailureDiagnostic],
        evidence: &[DecisionEvidence],
    ) -> DiagnosisReport {
        let mut hypotheses = Vec::new();
        for diagnostic in failures.iter().take(MAX_HYPOTHESES) {
            hypotheses.push(Hypothesis::propose(
                self.next_id(),
                diagnostic.clone(),
                evidence,
            ));
        }
        let mut seed = Vec::new();
        for hypothesis in &hypotheses {
            seed.extend_from_slice(hypothesis.claim.as_bytes());
            seed.push(b'\n');
        }
        for item in evidence {
            seed.extend_from_slice(item.source.as_bytes());
            seed.extend_from_slice(item.fact.as_bytes());
        }
        DiagnosisReport {
            id: format!("diag-{}", digest(&seed)),
            hypotheses,
            evidence: evidence.to_vec(),
        }
    }

    /// Classify a command's real captured output, then propose causes. The
    /// output is supplied by the caller; this method runs nothing.
    pub fn from_output(&mut self, output: &str, evidence: &[DecisionEvidence]) -> DiagnosisReport {
        self.propose(&classify_command_output(output), evidence)
    }

    fn next_id(&mut self) -> HypothesisId {
        let id = HypothesisId::new(format!("H{}", self.next));
        self.next += 1;
        id
    }
}

/// One diagnosis: its candidate causes, every piece of caller evidence (kept
/// verbatim, including contradictions) and the actions that follow real checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosisReport {
    /// Stable id derived from the proposals and evidence, not from a clock.
    pub id: String,
    pub hypotheses: Vec<Hypothesis>,
    /// All caller evidence, in the order supplied. Nothing is dropped here.
    pub evidence: Vec<DecisionEvidence>,
}

impl DiagnosisReport {
    pub fn hypothesis(&self, id: &HypothesisId) -> Option<&Hypothesis> {
        self.hypotheses.iter().find(|h| &h.id == id)
    }

    /// Mutable lookup so a caller can replace a proposal with its checked form
    /// without touching the other candidates.
    pub fn hypothesis_mut(&mut self, id: &HypothesisId) -> Option<&mut Hypothesis> {
        self.hypotheses.iter_mut().find(|h| &h.id == id)
    }

    /// Pairs of facts from the same source that say different things. Both
    /// members of every pair are still present in `evidence`; a contradiction
    /// is recorded, never silently resolved.
    pub fn contradictory_evidence(&self) -> Vec<(&DecisionEvidence, &DecisionEvidence)> {
        let mut pairs = Vec::new();
        for (i, left) in self.evidence.iter().enumerate() {
            for right in self.evidence.iter().skip(i + 1) {
                if left.source == right.source && left.fact != right.fact {
                    pairs.push((left, right));
                }
            }
        }
        pairs
    }

    /// Actions for hypotheses that survived a real check. Each names this
    /// diagnosis's id, so a recovery step can be traced back to it.
    pub fn recovery_actions(&self) -> Vec<RecoveryAction> {
        self.hypotheses
            .iter()
            .filter_map(|hypothesis| match &hypothesis.status {
                HypothesisStatus::Survived { execution } => Some(RecoveryAction {
                    diagnosis_id: self.id.clone(),
                    hypothesis_id: hypothesis.id.clone(),
                    strategy: hypothesis.recovery_hint(),
                    detail: format!(
                        "the real {} check exited {} and still reported the failure; {}",
                        execution.command,
                        execution.exit_code,
                        hypothesis.diagnostic.suggested_action
                    ),
                }),
                HypothesisStatus::Proposed
                | HypothesisStatus::Checked { .. }
                | HypothesisStatus::Falsified { .. } => None,
            })
            .collect()
    }
}

/// A next step a caller may take, traceable to the diagnosis and hypothesis
/// that earned it. Produced only for a hypothesis that survived a real check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryAction {
    pub diagnosis_id: String,
    pub hypothesis_id: HypothesisId,
    pub strategy: RecoveryStrategy,
    pub detail: String,
}

/// Storage for diagnosis reports. Reports are keyed by id and are never
/// silently overwritten: a second report with an existing id is an error.
pub struct DiagnosisStore {
    root: PathBuf,
}

impl DiagnosisStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn path(&self, id: &str) -> PathBuf {
        self.root.join("diagnoses").join(format!("{id}.json"))
    }

    /// Save a report atomically, refusing to replace one with the same id.
    pub fn save(&self, report: &DiagnosisReport) -> Result<(), DiagnosisError> {
        let path = self.path(&report.id);
        if path.exists() {
            return Err(DiagnosisError::DuplicateDiagnosis(report.id.clone()));
        }
        let bytes = serde_json::to_vec_pretty(report).map_err(|error| DiagnosisError::Write {
            path: path.clone(),
            message: error.to_string(),
        })?;
        write_atomic(&path, &bytes).map_err(|error| DiagnosisError::Write {
            path,
            message: error.to_string(),
        })
    }

    /// Read a report back. Its stored checks are whatever the run captured.
    pub fn load(&self, id: &str) -> Result<DiagnosisReport, DiagnosisError> {
        let path = self.path(id);
        let bytes = std::fs::read(&path).map_err(|error| DiagnosisError::Write {
            path: path.clone(),
            message: error.to_string(),
        })?;
        serde_json::from_slice(&bytes).map_err(|error| DiagnosisError::Write {
            path,
            message: error.to_string(),
        })
    }

    /// Every stored report, sorted by id so the order is stable on any host.
    pub fn load_all(&self) -> Result<Vec<DiagnosisReport>, DiagnosisError> {
        let dir = self.root.join("diagnoses");
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(DiagnosisError::Write {
                    path: dir,
                    message: error.to_string(),
                })
            }
        };
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| DiagnosisError::Write {
                path: dir.clone(),
                message: error.to_string(),
            })?;
            if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                paths.push(entry.path());
            }
        }
        paths.sort();
        paths
            .into_iter()
            .map(|path| {
                let bytes = std::fs::read(&path).map_err(|error| DiagnosisError::Write {
                    path: path.clone(),
                    message: error.to_string(),
                })?;
                serde_json::from_slice(&bytes).map_err(|error| DiagnosisError::Write {
                    path,
                    message: error.to_string(),
                })
            })
            .collect()
    }
}

/// Map a classified failure to a command the repo really runs and the
/// observation that would disprove the proposed cause.
fn falsifier_for(diagnostic: &FailureDiagnostic) -> (&'static str, FalsifierKind, String) {
    match diagnostic.category {
        Category::CompileError => {
            let code = diagnostic
                .message
                .split_once(": ")
                .map(|(code, _)| code.trim().to_string())
                .unwrap_or_else(|| clip(&diagnostic.message, 48));
            (
                CARGO_CHECK,
                FalsifierKind::Disappears {
                    fragment: code.clone(),
                },
                format!(
                    "if a real `cargo check` no longer reports {code}, the claim that this error explains the build failure is dead"
                ),
            )
        }
        Category::TestFailure => (
            CARGO_TEST,
            FalsifierKind::ExitZero,
            "if the real `cargo test` run exits zero, the failure no longer reproduces and this cause is dead".to_string(),
        ),
        Category::Panic => (
            CARGO_TEST,
            FalsifierKind::Disappears {
                fragment: "panicked at".to_string(),
            },
            "if a real `cargo test` run no longer reports a panic, this cause is dead".to_string(),
        ),
        Category::Io | Category::Permission => {
            let fragment = clip(&diagnostic.message, 48);
            (
                CARGO_TEST,
                FalsifierKind::Disappears {
                    fragment: fragment.clone(),
                },
                format!(
                    "if a real `cargo test` run no longer reports {fragment:?}, the I/O or permission cause is dead; callers with a more specific real command may narrow the falsifier"
                ),
            )
        }
    }
}

fn recovery_strategy(category: Category) -> RecoveryStrategy {
    match category {
        Category::CompileError | Category::Panic => RecoveryStrategy::StructuredDiagnostics,
        Category::TestFailure => RecoveryStrategy::AffectedDependencies,
        Category::Io => RecoveryStrategy::RequestMoreContext,
        Category::Permission => RecoveryStrategy::StopRepeating,
    }
}

fn clip(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect()
}

/// FNV-1a over the seed bytes; stable across builds, unlike `DefaultHasher`.
fn digest(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    // Captured verbatim from a real `cargo check` in a scratch crate, the same
    // output the EC-1 classifier's own tests use. Used only where no process is
    // needed; every status transition below is backed by a real executed check.
    const CAPTURED_COMPILE_OUTPUT: &str = r#"   Compiling compile_probe v0.1.0 (/tmp/opencode/ec1-fixtures/compile_probe)
error[E0308]: mismatched types
 --> src/lib.rs:7:15
  |
7 |     needs_u64(n)
  |     --------- ^ expected `u64`, found `u32`
  |     |
  |     arguments to this function are incorrect
  |
note: function defined here
 --> src/lib.rs:1:8
  |
1 | pub fn needs_u64(x: u64) -> u64 {
  |        ^^^^^^^^^ ------
help: you can convert a `u32` to a `u64`
  |
7 |     needs_u64(n.into())
  |                +++++++

For more information about this error, try `rustc --explain E0308`.
error: could not compile `compile_probe` (lib) due to 1 previous error
"#;

    /// Write a tiny real crate whose trait is genuinely not implemented, then
    /// run `cargo check` in it. The inner command gets its own target dir so it
    /// cannot wait on the target dir lock held by the outer `cargo test`.
    fn broken_crate(dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname=\"xi6_probe\"\nversion=\"0.1.0\"\nedition=\"2021\"\n\n[workspace]\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("src/lib.rs"),
            "pub trait Greeter { fn greet(&self) -> String; }\npub struct Robot;\npub fn say(r: &Robot) -> String { r.greet() }\n",
        )
        .unwrap();
    }

    fn real_cargo_check(dir: &std::path::Path) -> CheckExecution {
        CheckExecution::run(
            CARGO_CHECK,
            Command::new("cargo")
                .args(["check", "--quiet"])
                .current_dir(dir)
                // Keep the captured output plain and classifiable. CI sets
                // CARGO_TERM_COLOR=always; ANSI codes around `error[E0599]`
                // would stop the failure classifier recognising the cause.
                // The same setting is used by the real seed grader.
                .env("CARGO_TERM_COLOR", "never")
                .env("CARGO_TARGET_DIR", dir.join("target"))
                .env_remove("RUSTC_WRAPPER")
                .env_remove("RUSTC_WORKSPACE_WRAPPER"),
        )
        .expect("cargo check really runs")
    }

    #[test]
    fn a_real_cargo_check_moves_a_hypothesis_through_checked_to_survived_and_falsified() {
        let dir = tempfile::tempdir().unwrap();
        broken_crate(dir.path());

        // First real check: the crate genuinely does not compile.
        let first = real_cargo_check(dir.path());
        assert!(!first.passed(), "the fixture is meant to fail: {first:?}");
        assert!(
            first.combined_output().contains("E0599"),
            "rustc reports the missing impl: {}",
            first.combined_output()
        );

        let mut engine = DiagnosisEngine::new();
        let report = engine.from_output(
            &first.combined_output(),
            &[DecisionEvidence::new(
                "the real check",
                "cargo check exited nonzero on the fixture",
            )],
        );
        assert_eq!(report.hypotheses.len(), 1, "one classified cause here");
        let proposed = report.hypotheses[0].clone();
        assert_eq!(proposed.status, HypothesisStatus::Proposed);
        assert_eq!(proposed.falsifier.command, CARGO_CHECK);

        // Proposed -> Checked is explicit, and the output is stored.
        let checked = proposed.clone().check(first.clone()).unwrap();
        assert!(matches!(
            checked.status,
            HypothesisStatus::Checked { passed: false, .. }
        ));
        assert_eq!(checked.status.execution().unwrap(), &first);

        // Checked -> Survived: the falsifying observation was absent.
        let survived = checked.conclude().unwrap();
        assert!(matches!(survived.status, HypothesisStatus::Survived { .. }));
        assert!(survived
            .status
            .execution()
            .unwrap()
            .combined_output()
            .contains("E0599"));

        // Now fix the missing impl and run a second, independent real check on
        // a freshly proposed hypothesis. A survived hypothesis is a terminal
        // record; it is not silently re-marked as falsified.
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "pub trait Greeter { fn greet(&self) -> String; }\npub struct Robot;\npub fn say(r: &Robot) -> String { r.greet() }\nimpl Greeter for Robot { fn greet(&self) -> String { \"hello\".to_string() } }\n",
        )
        .unwrap();
        let second = real_cargo_check(dir.path());
        assert!(second.passed(), "the fix must really compile: {second:?}");

        let second_report = engine.from_output(&first.combined_output(), &[]);
        let fresh = second_report.hypotheses[0].clone();
        let falsified = fresh.check_and_conclude(second.clone()).unwrap();
        assert!(matches!(
            falsified.status,
            HypothesisStatus::Falsified { .. }
        ));
        assert_eq!(falsified.status.execution().unwrap(), &second);
        assert!(
            !falsified
                .status
                .execution()
                .unwrap()
                .combined_output()
                .contains("E0599"),
            "the stored output is the real second run"
        );
    }

    #[test]
    fn no_real_check_means_no_status_change() {
        let mut engine = DiagnosisEngine::new();
        let report = engine.from_output(CAPTURED_COMPILE_OUTPUT, &[]);
        let hypothesis = &report.hypotheses[0];
        assert_eq!(hypothesis.status, HypothesisStatus::Proposed);
        assert_eq!(hypothesis.status.label(), "proposed");
        assert!(hypothesis.status.execution().is_none());
        assert_eq!(
            hypothesis.clone().conclude().unwrap_err(),
            DiagnosisError::NotChecked
        );
    }

    #[test]
    fn a_mismatched_real_command_cannot_check_a_hypothesis() {
        let mut engine = DiagnosisEngine::new();
        let report = engine.from_output(CAPTURED_COMPILE_OUTPUT, &[]);
        let hypothesis = report.hypotheses[0].clone();
        let other = CheckExecution::run("cargo --version", Command::new("cargo").arg("--version"))
            .expect("cargo --version really runs");
        assert_eq!(
            hypothesis.clone().check(other).unwrap_err(),
            DiagnosisError::CommandMismatch {
                expected: CARGO_CHECK.to_string(),
                actual: "cargo --version".to_string(),
            }
        );
        assert_eq!(hypothesis.status, HypothesisStatus::Proposed);
    }

    #[test]
    fn contradictory_evidence_is_preserved_not_resolved() {
        let evidence = vec![
            DecisionEvidence::new("check result", "cargo check passed"),
            DecisionEvidence::new("check result", "cargo check failed"),
        ];
        let mut engine = DiagnosisEngine::new();
        let report = engine.from_output(CAPTURED_COMPILE_OUTPUT, &evidence);
        assert_eq!(report.evidence.len(), 2);
        assert!(report.evidence.contains(&evidence[0]));
        assert!(report.evidence.contains(&evidence[1]));
        let contradictions = report.contradictory_evidence();
        assert_eq!(contradictions.len(), 1);
        assert_eq!(*contradictions[0].0, evidence[0]);
        assert_eq!(*contradictions[0].1, evidence[1]);
        // The proposal still carries both facts and stays honestly untested.
        assert_eq!(report.hypotheses[0].evidence.len(), 3);
        assert_eq!(report.hypotheses[0].status, HypothesisStatus::Proposed);
    }

    #[test]
    fn hypotheses_get_distinct_ids_and_reports_are_not_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let store = DiagnosisStore::new(dir.path());

        let mut first_engine = DiagnosisEngine::new();
        let mut second_engine = DiagnosisEngine::new();
        let first = first_engine.from_output(
            &format!("{CAPTURED_COMPILE_OUTPUT}{CAPTURED_COMPILE_OUTPUT}"),
            &[],
        );
        assert_eq!(first.hypotheses.len(), 2, "two classified failures");
        assert_ne!(first.hypotheses[0].id, first.hypotheses[1].id);
        assert!(first.hypothesis(&first.hypotheses[0].id).is_some());
        assert!(first.hypothesis(&first.hypotheses[1].id).is_some());

        // Same inputs from a fresh engine give the same stable diagnosis id.
        let second = second_engine.from_output(
            &format!("{CAPTURED_COMPILE_OUTPUT}{CAPTURED_COMPILE_OUTPUT}"),
            &[],
        );
        assert_eq!(second.id, first.id);
        store.save(&first).unwrap();
        assert_eq!(
            store.save(&second).unwrap_err(),
            DiagnosisError::DuplicateDiagnosis(first.id.clone())
        );
        let stored = store.load(&first.id).unwrap();
        assert_eq!(stored, first);
        assert_eq!(store.load_all().unwrap().len(), 1);
        assert_eq!(
            store.load_all().unwrap()[0].hypotheses[0].id,
            first.hypotheses[0].id
        );
    }

    #[test]
    fn recovery_actions_reference_the_diagnosis_and_only_follow_real_checks() {
        let dir = tempfile::tempdir().unwrap();
        broken_crate(dir.path());
        let execution = real_cargo_check(dir.path());
        assert!(!execution.passed());

        let mut engine = DiagnosisEngine::new();
        let mut report = engine.from_output(&execution.combined_output(), &[]);
        assert!(
            report.recovery_actions().is_empty(),
            "proposals earn nothing"
        );

        let id = report.hypotheses[0].id.clone();
        let survived = report.hypotheses[0]
            .clone()
            .check_and_conclude(execution)
            .unwrap();
        *report.hypothesis_mut(&id).unwrap() = survived;

        let actions = report.recovery_actions();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].diagnosis_id, report.id);
        assert_eq!(actions[0].hypothesis_id, id);
        assert_eq!(actions[0].strategy, RecoveryStrategy::StructuredDiagnostics);
        assert!(actions[0].detail.contains(CARGO_CHECK));

        // A persisted report keeps the real output and the recovery reference.
        let store = DiagnosisStore::new(dir.path());
        store.save(&report).unwrap();
        let loaded = store.load(&report.id).unwrap();
        assert_eq!(loaded.recovery_actions()[0].diagnosis_id, report.id);
        assert!(loaded.hypotheses[0]
            .status
            .execution()
            .unwrap()
            .combined_output()
            .contains("E0599"));
    }
}
