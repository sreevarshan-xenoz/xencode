//! Real behaviour tests for evidence-gated repair coordination (XI-7).
//!
//! No mocks: every check in these tests is a real `cargo check` process with
//! its real exit code and captured output, every reproduction identity is a
//! digest of real bytes on disk, and evidence is written to and read from real
//! temporary directories with the atomic writer.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use xencode_context_rs::{
    ApprovalDecision, ChecksVerdict, LedgerEntry, RunClass, RunEvidence, RunRecord,
};
use xencode_intelligence_rs::diagnosis::CheckExecution;
use xencode_intelligence_rs::*;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A scratch directory that is always removed, including on a failed test.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "xencode-repair-{}-{name}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("scratch directory must be created");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A tiny real crate whose trait is genuinely not implemented, so a real
/// `cargo check` fails with E0599 until a test writes the implementation.
fn broken_crate(dir: &Path) {
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname=\"xi7_repair_probe\"\nversion=\"0.1.0\"\nedition=\"2021\"\n\n[workspace]\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("src/lib.rs"),
        "pub trait Greeter { fn greet(&self) -> String; }\npub struct Robot;\npub fn say(r: &Robot) -> String { r.greet() }\n",
    )
    .unwrap();
}

fn fix_the_crate(dir: &Path) {
    std::fs::write(
        dir.join("src/lib.rs"),
        "pub trait Greeter { fn greet(&self) -> String; }\npub struct Robot;\npub fn say(r: &Robot) -> String { r.greet() }\nimpl Greeter for Robot { fn greet(&self) -> String { \"hello\".to_string() } }\n",
    )
    .unwrap();
}

/// Run a real `cargo check` in the fixture. Colour is forced off so a CI
/// setting of `CARGO_TERM_COLOR=always` cannot wrap E0599 in ANSI codes, and
/// the target dir is isolated so the outer `cargo test` cannot hold its lock.
fn real_check(dir: &Path, label: &str) -> CheckExecution {
    CheckExecution::run(
        label,
        Command::new("cargo")
            .args(["check", "--quiet"])
            .current_dir(dir)
            .env("CARGO_TERM_COLOR", "never")
            .env("CARGO_TARGET_DIR", dir.join("target"))
            .env_remove("RUSTC_WRAPPER")
            .env_remove("RUSTC_WORKSPACE_WRAPPER"),
    )
    .expect("cargo check really runs")
}

/// A real reproduction file, fingerprinted from the bytes actually on disk.
fn reproduction(dir: &Path, content: &str) -> ReproIdentity {
    let file = dir.join("repro").join("assertion.txt");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, content).unwrap();
    let bytes = std::fs::read(&file).unwrap();
    ReproIdentity::of(
        file.strip_prefix(dir)
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        &bytes,
    )
}

fn clean_verdict(label: &str) -> ChecksVerdict {
    ChecksVerdict {
        ran: vec![label.to_string()],
        failed: Vec::new(),
        skipped: Vec::new(),
        evidence_ref: Some("tools[0]".to_string()),
    }
}

fn approval(decision: ApprovalDecision) -> RepairApproval {
    RepairApproval::new("edit_file", "file change", decision)
}

fn started(id: &str, diagnosis: &str) -> RepairLoop {
    let mut repair = RepairLoop::new(id, diagnosis);
    repair.diagnose().unwrap();
    repair.plan().unwrap();
    repair.begin().unwrap();
    repair
}

/// Witness a real red and freeze a real reproduction file.
fn witness_red(repair: &mut RepairLoop, dir: &Path) -> ReproIdentity {
    let red = real_check(dir, "cargo check");
    assert!(!red.passed(), "the fixture must genuinely fail: {red:?}");
    assert!(
        red.combined_output().contains("E0599"),
        "rustc reports the missing impl: {}",
        red.combined_output()
    );
    let repro = reproduction(dir, "assert!(!Robot.say().is_empty());\n");
    repair.witness_red(&red, repro.clone()).unwrap();
    repro
}

#[test]
fn no_production_edit_before_a_witnessed_failure() {
    let scratch = Scratch::new("no-red");
    broken_crate(scratch.path());
    let mut repair = started("repair-no-red", "diag-1");

    assert_eq!(repair.next_action(), RepairAction::Reproduce);
    assert_eq!(
        repair.implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs"),
        Err(RepairError::RedNotWitnessed)
    );
    let evidence = repair.evidence();
    assert!(evidence.approval.is_none());
    assert!(evidence.impl_command.is_none());
    assert!(!evidence.write_recorded);
    assert_eq!(evidence.result(), RepairResult::NotVerified);
}

#[test]
fn a_skipped_failed_or_empty_check_never_becomes_success() {
    // A check that was skipped cannot pass even though the one that ran did.
    let skipped = Scratch::new("skipped");
    broken_crate(skipped.path());
    let mut repair = started("repair-skipped", "diag-skipped");
    let repro = witness_red(&mut repair, skipped.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();
    fix_the_crate(skipped.path());
    let green = real_check(skipped.path(), "cargo check");
    assert!(green.passed(), "the fix really compiles: {green:?}");

    let mut verdict = clean_verdict("cargo check");
    verdict.skipped.push("cargo test".to_string());
    assert!(!verdict.all_passed());
    assert_eq!(
        repair.verify(&verdict, &green, &repro),
        Ok(RepairResult::NotVerified)
    );
    assert_eq!(repair.next_action(), RepairAction::Rediagnose);
    assert_eq!(repair.evidence().result(), RepairResult::NotVerified);
    assert!(!repair.evidence().write_recorded);
    // A failed verification is terminal for this repair.
    assert_eq!(
        repair.verify(&clean_verdict("cargo check"), &green, &repro),
        Err(RepairError::VerificationAlreadyFailed)
    );

    // A failed check is a failure even with an otherwise clean verdict.
    let failed = Scratch::new("failed");
    broken_crate(failed.path());
    let mut repair = started("repair-failed", "diag-failed");
    let repro = witness_red(&mut repair, failed.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();
    let still_red = real_check(failed.path(), "cargo check");
    assert!(!still_red.passed());
    assert_eq!(
        repair.verify(&clean_verdict("cargo check"), &still_red, &repro),
        Ok(RepairResult::NotVerified)
    );
    assert_eq!(repair.evidence().result(), RepairResult::NotVerified);

    // An empty verdict has no evidence that any check ran at all.
    let empty = Scratch::new("empty");
    broken_crate(empty.path());
    let mut repair = started("repair-empty", "diag-empty");
    let repro = witness_red(&mut repair, empty.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();
    fix_the_crate(empty.path());
    let green = real_check(empty.path(), "cargo check");
    let no_runs = ChecksVerdict::default();
    assert!(!no_runs.all_passed());
    assert_eq!(
        repair.verify(&no_runs, &green, &repro),
        Ok(RepairResult::NotVerified)
    );
    assert_eq!(repair.evidence().result(), RepairResult::NotVerified);
}

#[test]
fn a_model_claim_without_a_real_check_is_refused() {
    let scratch = Scratch::new("claim");
    broken_crate(scratch.path());
    let mut repair = started("repair-claim", "diag-claim");
    let repro = witness_red(&mut repair, scratch.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();

    let claim = DecisionEvidence::new(
        "model",
        "the fix is obviously correct; the checks would certainly pass",
    );
    assert_eq!(
        repair.refuse_model_claim(claim.clone()),
        RepairError::ModelClaimNotEvidence
    );
    let evidence = repair.evidence();
    assert_eq!(evidence.refused_claims, vec![claim]);
    assert_eq!(evidence.phase, RepairPhase::RedWitnessed);
    assert_eq!(evidence.result(), RepairResult::NotVerified);
    assert!(!evidence.write_recorded);
    // The only constructor for a RepairCheck is from a real CheckExecution, so
    // this claim cannot stand in for the green check.
    assert_eq!(repair.next_action(), RepairAction::Verify);
    assert_eq!(evidence.green, None);
    let _ = repro;
}

#[test]
fn changing_the_reproduction_after_red_invalidates_the_repair() {
    let scratch = Scratch::new("frozen");
    broken_crate(scratch.path());
    let mut repair = started("repair-frozen", "diag-frozen");
    let frozen = witness_red(&mut repair, scratch.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();

    // Rewrite the real reproduction with an always-true assertion.
    let changed = reproduction(scratch.path(), "assert!(true);\n");
    assert_ne!(frozen, changed);

    fix_the_crate(scratch.path());
    let green = real_check(scratch.path(), "cargo check");
    assert!(green.passed());
    assert_eq!(
        repair.verify(&clean_verdict("cargo check"), &green, &changed),
        Err(RepairError::ReproChanged {
            expected: format!("{}@{}", frozen.path, frozen.digest),
            actual: format!("{}@{}", changed.path, changed.digest),
        })
    );
    assert_eq!(repair.evidence().result(), RepairResult::NotVerified);
    assert!(!repair.evidence().write_recorded);
    assert!(repair.evidence().failure_reason.is_some());
}

#[test]
fn a_granted_human_approval_is_required_for_a_production_edit() {
    let scratch = Scratch::new("approval");
    broken_crate(scratch.path());
    let mut repair = started("repair-approval", "diag-approval");
    witness_red(&mut repair, scratch.path());

    assert_eq!(repair.next_action(), RepairAction::AwaitHumanApproval);
    assert_eq!(
        repair.implement(&approval(ApprovalDecision::Denied), "edit_file:src/lib.rs"),
        Err(RepairError::ApprovalRequired)
    );
    assert!(repair.evidence().approval.is_none());

    let granted = approval(ApprovalDecision::Allowed);
    let authorization = repair.implement(&granted, "edit_file:src/lib.rs").unwrap();
    assert_eq!(authorization.repair_id, "repair-approval");
    assert_eq!(authorization.diagnosis_id, "diag-approval");
    assert_eq!(authorization.approval, granted);
    assert_eq!(authorization.impl_command, "edit_file:src/lib.rs");
    assert_eq!(repair.evidence().approval, Some(granted));
    assert_eq!(repair.next_action(), RepairAction::Verify);
    assert!(!repair.evidence().write_recorded);
}

#[test]
fn verification_command_must_differ_from_the_implementation_command() {
    let scratch = Scratch::new("independent");
    broken_crate(scratch.path());
    let mut repair = started("repair-independent", "diag-independent");
    let repro = witness_red(&mut repair, scratch.path());
    // A caller that claims the edit itself was the same command used to verify.
    repair
        .implement(&approval(ApprovalDecision::Allowed), "cargo check")
        .unwrap();
    fix_the_crate(scratch.path());
    let green = real_check(scratch.path(), "cargo check");
    assert!(green.passed());
    assert_eq!(
        repair.verify(&clean_verdict("cargo check"), &green, &repro),
        Err(RepairError::NotIndependent {
            impl_command: "cargo check".to_string(),
            verify_command: "cargo check".to_string(),
        })
    );
    assert_eq!(repair.evidence().result(), RepairResult::NotVerified);

    // The green must also be the same measurement as the red, not some other
    // command that happens to pass.
    let other = Scratch::new("other-command");
    broken_crate(other.path());
    let mut repair = started("repair-other-command", "diag-other");
    let repro = witness_red(&mut repair, other.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();
    fix_the_crate(other.path());
    let version = CheckExecution::run("cargo --version", Command::new("cargo").arg("--version"))
        .expect("cargo --version really runs");
    assert!(version.passed());
    assert_eq!(
        repair.verify(&clean_verdict("cargo --version"), &version, &repro),
        Err(RepairError::NotSameMeasurement {
            red_command: "cargo check".to_string(),
            verify_command: "cargo --version".to_string(),
        })
    );
}

#[test]
fn a_real_red_to_green_loop_ends_verified_with_a_persisted_record() {
    let scratch = Scratch::new("green");
    broken_crate(scratch.path());
    let mut repair = started("repair-green", "diag-green");

    let plan = ReasoningPlan {
        stage: ReasoningStage::Normal,
        max_tool_rounds: 8,
        verification: VerificationIntensity::Thorough,
        recovery: RecoveryStrategy::StructuredDiagnostics,
        escalation_eligible: false,
        escalation_is_cloud: false,
        rationale: vec![DecisionEvidence::new(
            "observed",
            "a real cargo check failed",
        )],
    };
    repair.set_reasoning_plan(&plan);

    let red = real_check(scratch.path(), "cargo check");
    assert!(!red.passed());
    let repro = reproduction(scratch.path(), "assert!(!Robot.say().is_empty());\n");
    repair.witness_red(&red, repro.clone()).unwrap();
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();
    fix_the_crate(scratch.path());
    let green = real_check(scratch.path(), "cargo check");
    assert!(green.passed(), "the fix really compiles: {green:?}");

    let verdict = clean_verdict("cargo check");
    assert_eq!(
        repair.verify(&verdict, &green, &repro),
        Ok(RepairResult::Verified)
    );
    let evidence = repair.evidence();
    assert_eq!(evidence.phase, RepairPhase::Verified);
    assert_eq!(evidence.stage, PipelineStage::Report);
    assert_eq!(evidence.result(), RepairResult::Verified);
    assert!(evidence.is_success());
    assert!(evidence.write_recorded);
    assert_eq!(evidence.attempts, 1);
    assert_eq!(
        evidence.recovery,
        Some(RecoveryStrategy::StructuredDiagnostics)
    );
    assert_eq!(evidence.round_cap, Some(8));
    let red = evidence.red.as_ref().unwrap();
    let green = evidence.green.as_ref().unwrap();
    assert!(!red.passed());
    assert!(green.passed());
    assert!(!red.output().is_empty());
    assert!(red.output().contains("E0599"));
    assert_eq!(evidence.verdict, Some(verdict));
    assert_eq!(repair.next_action(), RepairAction::Report);

    let store = RepairStore::new(scratch.path());
    store.save(&evidence).unwrap();
    let loaded = store.load("repair-green").unwrap();
    assert_eq!(loaded, evidence);
    let resumed = RepairLoop::from_evidence(loaded).unwrap();
    assert_eq!(resumed.evidence().result(), RepairResult::Verified);
    assert_eq!(
        store.history("repair-green").len(),
        evidence.transitions.len()
    );
}

#[test]
fn an_interrupted_repair_resumes_without_replaying_an_unverified_write() {
    let scratch = Scratch::new("resume");
    broken_crate(scratch.path());
    let mut repair = started("repair-resume", "diag-resume");
    let repro = witness_red(&mut repair, scratch.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();

    // The edit was authorized but its check never ran, so no write is recorded.
    assert!(!repair.evidence().write_recorded);
    assert!(repair.evidence().green.is_none());
    repair.interrupt().unwrap();
    assert_eq!(repair.evidence().result(), RepairResult::Interrupted);

    let store = RepairStore::new(scratch.path());
    store.save(&repair.evidence()).unwrap();
    let persisted = store.load("repair-resume").unwrap();
    let mut resumed = RepairLoop::from_evidence(persisted).unwrap();
    assert_eq!(resumed.evidence().result(), RepairResult::Interrupted);
    assert!(!resumed.evidence().write_recorded);
    resumed.resume().unwrap();
    assert_eq!(resumed.next_action(), RepairAction::Verify);
    assert!(resumed.evidence().green.is_none());
    assert!(!resumed.evidence().write_recorded);

    // Resuming did not invent a green: a fresh real check is still required.
    fix_the_crate(scratch.path());
    let green = real_check(scratch.path(), "cargo check");
    assert!(green.passed());
    assert_eq!(
        resumed.verify(&clean_verdict("cargo check"), &green, &repro),
        Ok(RepairResult::Verified)
    );
    store.save(&resumed.evidence()).unwrap();
    let final_record = store.load("repair-resume").unwrap();
    assert_eq!(final_record.result(), RepairResult::Verified);
    assert!(final_record.write_recorded);
}

#[test]
fn a_skipped_check_that_looks_passing_is_a_structural_guard_not_a_calibration_claim() {
    // This is a structural test, not a measured improvement in a false-verified
    // rate. Measuring that rate on real repairs needs a GPU-backed model run,
    // which this environment does not have; nothing here claims one.
    let scratch = Scratch::new("calibration");
    broken_crate(scratch.path());
    let mut repair = started("repair-calibration", "diag-calibration");
    let repro = witness_red(&mut repair, scratch.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();
    fix_the_crate(scratch.path());
    let green = real_check(scratch.path(), "cargo check");
    assert!(green.passed());

    let mut verdict = clean_verdict("cargo check");
    verdict.skipped.push("cargo test".to_string());
    // A passing ledger row can make RunEvidence::verified() say yes, but that
    // only means "a check exited zero at some point", not "the due checks ran".
    let run = RunRecord {
        run_id: "run-calibration".to_string(),
        ts_unix_ms: 0,
        duration_ms: 0,
        rounds: 1,
        session: Some("session-calibration".to_string()),
        model: None,
        provider: None,
        source: None,
        approvals: Vec::new(),
        recording: None,
        note: String::new(),
        computer: None,
    };
    let passing_row = LedgerEntry {
        ts_unix_ms: 0,
        session: Some("session-calibration".to_string()),
        run_class: RunClass::Test,
        exit_code: 0,
        subjects: Vec::new(),
        log_ref: "tools[0]".to_string(),
        note: String::new(),
    };
    assert!(RunEvidence {
        run,
        checks: vec![passing_row],
    }
    .verified());
    assert!(!verdict.all_passed());

    assert_eq!(
        repair.verify(&verdict, &green, &repro),
        Ok(RepairResult::NotVerified)
    );
    assert_eq!(repair.evidence().result(), RepairResult::NotVerified);
}

#[test]
fn an_incomplete_verified_record_never_loads_as_success() {
    let scratch = Scratch::new("incomplete");
    broken_crate(scratch.path());
    let mut repair = started("repair-incomplete", "diag-incomplete");
    let repro = witness_red(&mut repair, scratch.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();
    fix_the_crate(scratch.path());
    let green = real_check(scratch.path(), "cargo check");
    assert!(green.passed());
    assert_eq!(
        repair.verify(&clean_verdict("cargo check"), &green, &repro),
        Ok(RepairResult::Verified)
    );

    // Strip the real green and the verdict from an otherwise verified record.
    let mut evidence = repair.evidence();
    evidence.green = None;
    evidence.verdict = None;
    let demoted = RepairLoop::from_evidence(evidence).unwrap();
    assert_eq!(demoted.evidence().result(), RepairResult::NotVerified);
    assert!(!demoted.evidence().is_success());
    assert!(demoted
        .evidence()
        .failure_reason
        .as_deref()
        .unwrap()
        .contains("lacked a required piece of real evidence"));
}

#[test]
fn a_hard_round_cap_stops_endless_failed_repair_cycles() {
    let scratch = Scratch::new("cap");
    broken_crate(scratch.path());
    let mut repair = started("repair-cap", "diag-cap");
    let plan = ReasoningPlan {
        stage: ReasoningStage::Normal,
        max_tool_rounds: 1,
        verification: VerificationIntensity::Standard,
        recovery: RecoveryStrategy::AffectedDependencies,
        escalation_eligible: false,
        escalation_is_cloud: false,
        rationale: vec![DecisionEvidence::new("observed", "one failed check")],
    };
    repair.set_reasoning_plan(&plan);
    let repro = witness_red(&mut repair, scratch.path());
    repair
        .implement(&approval(ApprovalDecision::Allowed), "edit_file:src/lib.rs")
        .unwrap();
    let still_red = real_check(scratch.path(), "cargo check");
    assert_eq!(
        repair.verify(&clean_verdict("cargo check"), &still_red, &repro),
        Ok(RepairResult::NotVerified)
    );
    assert_eq!(
        repair.rediagnose(),
        Err(RepairError::HardCapReached { cap: 1 })
    );
    assert_eq!(repair.evidence().result(), RepairResult::NotVerified);
}
