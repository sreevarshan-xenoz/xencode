//! Real behaviour tests for bounded online adaptation (XI-5).
//!
//! No mocks: eligible evidence comes from typed records derived from real
//! `turns.jsonl` rows, real check verdicts and real run ledgers; persistent
//! policies are written to and read back from real directories. Every refusal
//! is asserted against the exact reason the classifier gives.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use xencode_context_rs::{
    append_run, append_trace, read_recent_traces, run_by_id, ChecksVerdict, RunRecord, ToolTrace,
    TurnTrace,
};
use xencode_intelligence_rs::adaptation::{
    classify, evaluate_holdout, update_policy, AdaptationError, AdaptationObservation,
    AdaptationParameters, AdaptationPolicy, Classification, Eligibility, EvidenceRecord,
    EvidenceRef, FixtureEvent, LearningOptions, OutcomeKind, PolicyStatus, PolicyStore, Provenance,
    RecoveryHint, Scope, SessionAdaptation, SignalOrigin, ToolOutcomeStatus, DEFAULT_TOOL_ROUNDS,
    MAX_TOOL_ROUNDS,
};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "xencode-adaptation-{}-{name}-{}",
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

fn evidence(store: &str, record_id: &str) -> EvidenceRef {
    EvidenceRef::new(store, record_id, SignalOrigin::Observed)
}

fn check(name: &str, failed: bool, id: &str, root: &Path) -> EvidenceRecord {
    EvidenceRecord::CheckOutcome {
        name: name.to_string(),
        ran: true,
        exit_code: Some(if failed { 1 } else { 0 }),
        evidence: evidence("turns.jsonl", id),
        session_id: "session-a".to_string(),
        project_root: root.to_path_buf(),
    }
}

fn dominant_failed_checks(root: &Path) -> Vec<EvidenceRecord> {
    vec![
        check("cargo test", true, "run-1", root),
        check("cargo test", true, "run-2", root),
        check("cargo test", true, "run-3", root),
        EvidenceRecord::EditOutcome {
            verified_changed: false,
            checked_on_disk: true,
            evidence: evidence("git-diff", "run-4"),
            session_id: "session-a".to_string(),
            project_root: root.to_path_buf(),
        },
    ]
}

fn classification(record: &EvidenceRecord) -> Classification {
    classify(record, Scope::Session)
}

fn ineligible(classification: &Classification) -> Eligibility {
    match classification {
        Classification::Ineligible(row) => row.reason,
        Classification::Eligible(observation) => {
            panic!("expected refusal, got {:?}", observation.outcome)
        }
    }
}

fn eligible(classification: &Classification) -> &AdaptationObservation {
    match classification {
        Classification::Eligible(observation) => observation,
        Classification::Ineligible(row) => panic!("expected eligible, got {:?}", row.reason),
    }
}

fn provenance(actor: &str, note: &str) -> Provenance {
    Provenance {
        actor: actor.to_string(),
        written_at_ms: 1_700_000_000_000,
        note: note.to_string(),
    }
}

#[test]
fn unverified_claims_are_refused_not_learned() {
    let root = PathBuf::from("/repo");
    let model_claim = EvidenceRecord::ModelClaim {
        text: "I definitely fixed the bug and all tests pass".to_string(),
        evidence: evidence("narrative", "turn-1"),
        session_id: "s".to_string(),
        project_root: root.clone(),
    };
    assert_eq!(
        ineligible(&classification(&model_claim)),
        Eligibility::ModelNarrative
    );

    let unverified_edit = EvidenceRecord::EditOutcome {
        verified_changed: true,
        checked_on_disk: false,
        evidence: evidence("turns.jsonl", "turn-2"),
        session_id: "s".to_string(),
        project_root: root.clone(),
    };
    assert_eq!(
        ineligible(&classification(&unverified_edit)),
        Eligibility::UnverifiedClaim
    );

    let unconfirmed_recovery = EvidenceRecord::RecoveryOutcome {
        succeeded: true,
        confirmed: false,
        evidence: evidence("runs.jsonl", "run-1"),
        session_id: "s".to_string(),
        project_root: root,
    };
    assert_eq!(
        ineligible(&classification(&unconfirmed_recovery)),
        Eligibility::UnverifiedClaim
    );
}

#[test]
fn ambiguous_outcomes_are_refused() {
    let root = PathBuf::from("/repo");
    let unknown = EvidenceRecord::ToolOutcome {
        tool: "edit_file".to_string(),
        outcome: ToolOutcomeStatus::Unrecognized("maybe".to_string()),
        evidence: evidence("turns.jsonl", "turn-9"),
        session_id: "s".to_string(),
        project_root: root.clone(),
    };
    assert_eq!(
        ineligible(&classification(&unknown)),
        Eligibility::AmbiguousOutcome
    );

    // The same unknown outcome derived from a real trace row stays unknown.
    let trace_tool = ToolTrace {
        name: "edit_file".to_string(),
        outcome: "possibly".to_string(),
        arguments: None,
        tail: None,
    };
    let from_trace =
        EvidenceRecord::from_tool_trace(&trace_tool, evidence("turns.jsonl", "10"), "s", root);
    assert_eq!(
        ineligible(&classification(&from_trace)),
        Eligibility::AmbiguousOutcome
    );
}

#[test]
fn incomplete_checks_and_policy_decisions_are_refused() {
    let root = PathBuf::from("/repo");
    let skipped = EvidenceRecord::CheckOutcome {
        name: "cargo test".to_string(),
        ran: false,
        exit_code: None,
        evidence: evidence("turns.jsonl", "skip-1"),
        session_id: "s".to_string(),
        project_root: root.clone(),
    };
    assert_eq!(
        ineligible(&classification(&skipped)),
        Eligibility::IncompleteOutcome
    );

    // A real checks verdict with a skipped check yields no exit code.
    let verdict = ChecksVerdict {
        ran: vec![],
        failed: vec![],
        skipped: vec!["cargo test".to_string()],
        evidence_ref: None,
    };
    let rows = EvidenceRecord::from_checks(&verdict, evidence("turns.jsonl", "t"), "s", &root);
    assert!(
        rows.is_empty(),
        "a skipped-only verdict produces no outcome rows"
    );

    for status in [ToolOutcomeStatus::Denied, ToolOutcomeStatus::Refused] {
        let denied = EvidenceRecord::ToolOutcome {
            tool: "run_command".to_string(),
            outcome: status,
            evidence: evidence("turns.jsonl", "deny-1"),
            session_id: "s".to_string(),
            project_root: root.clone(),
        };
        assert_eq!(
            ineligible(&classification(&denied)),
            Eligibility::PolicyDecision
        );
    }
}

#[test]
fn repeated_identical_failure_does_not_create_a_persistent_policy() {
    let scratch = Scratch::new("dedup");
    let failing = check("cargo test", true, "same-run", scratch.path());
    let mut session = SessionAdaptation::new("session-a", scratch.path());
    for _ in 0..8 {
        assert_eq!(
            eligible(&session.observe(&failing)).id,
            eligible(&classification(&failing)).id
        );
    }
    assert_eq!(
        session.policy().version,
        1,
        "one distinct failure, one version"
    );
    assert_eq!(
        session.policy().parameters.tool_rounds,
        DEFAULT_TOOL_ROUNDS + 2
    );

    // Nothing at all was written: session adaptation is in-memory only.
    assert!(
        !scratch.path().join("policies").exists(),
        "session adaptation must not create a policy directory"
    );
}

#[test]
fn session_adaptation_improves_a_deterministic_decision_without_persistent_state() {
    let scratch = Scratch::new("session");
    let xencode = scratch.path().join(".xencode");

    // Real turn rows with failed checks, written and read back with the same
    // reader the CLI uses.
    let mut trace = TurnTrace::new(1_000, 3);
    trace.session_id = Some("session-a".to_string());
    trace.checks = Some(ChecksVerdict {
        ran: vec!["cargo test".to_string()],
        failed: vec!["cargo test".to_string()],
        skipped: vec![],
        evidence_ref: Some("tools[2]".to_string()),
    });
    append_trace(&xencode, &trace).expect("append_trace must write");
    let rows = read_recent_traces(&xencode, 10);
    assert_eq!(rows.len(), 1, "the written trace reads back");

    let mut session = SessionAdaptation::new("session-a", scratch.path());
    for (index, row) in rows.iter().enumerate() {
        let verdict = row.checks.as_ref().expect("the check verdict is present");
        for record in EvidenceRecord::from_checks(
            verdict,
            evidence("turns.jsonl", &format!("turn-{index}")),
            "session-a",
            scratch.path(),
        ) {
            assert!(matches!(
                session.observe(&record),
                Classification::Eligible(_)
            ));
        }
    }

    let decision = session.decide();
    assert_eq!(decision.recovery, RecoveryHint::AffectedDependencies);
    assert_eq!(decision.tool_rounds, DEFAULT_TOOL_ROUNDS + 2);
    assert_eq!(decision.policy_version, 1);
    assert_eq!(
        decision.recovery,
        RecoveryHint::AffectedDependencies,
        "a fixed baseline would have chosen Proceed"
    );
    assert!(
        !scratch.path().join("policies").exists(),
        "session adaptation wrote no persistent state"
    );
}

#[test]
fn project_scope_cannot_cross_a_repository_boundary() {
    let scratch = Scratch::new("boundary");
    let repo_a = scratch.path().join("repo-a");
    let repo_b = scratch.path().join("repo-b");
    std::fs::create_dir_all(&repo_a).unwrap();
    std::fs::create_dir_all(&repo_b).unwrap();

    let observations: Vec<_> = dominant_failed_checks(&repo_a)
        .iter()
        .map(|record| eligible(&classification(record)).clone())
        .collect();
    let policy = update_policy(
        None,
        &observations,
        Scope::Project,
        Some(repo_a.clone()),
        &LearningOptions::session_only().with_project(),
    )
    .expect("project policy is allowed with opt-in");
    assert_eq!(policy.scope, Scope::Project);
    assert_eq!(policy.project_root.as_deref(), Some(repo_a.as_path()));

    let store_a = PolicyStore::new(
        scratch.path().join("state-a"),
        &repo_a,
        LearningOptions::session_only().with_project(),
    );
    store_a
        .write(&policy, &provenance("engine", "learned from repo-a"))
        .expect("writing inside the bound repository succeeds");
    assert_eq!(store_a.read_current(), Some(policy.clone()));

    let store_b = PolicyStore::new(
        scratch.path().join("state-b"),
        &repo_b,
        LearningOptions::session_only().with_project(),
    );
    let refused = store_b
        .write(&policy, &provenance("engine", "attempted cross-repo write"))
        .expect_err("a policy bound to repo-a must not be written in repo-b");
    assert!(
        matches!(refused, AdaptationError::RepoBoundary { .. }),
        "cross-repo use was refused: {refused}"
    );
    assert!(
        store_b.read_current().is_none(),
        "no policy was created in the second repository"
    );
}

#[test]
fn cross_session_is_off_by_default_and_refuses_without_opt_in() {
    let scratch = Scratch::new("global");
    let repo = scratch.path().join("repo");
    let observations: Vec<_> = dominant_failed_checks(&repo)
        .iter()
        .map(|record| eligible(&classification(record)).clone())
        .collect();

    let refused = update_policy(
        None,
        &observations,
        Scope::CrossSession,
        None,
        &LearningOptions::session_only(),
    )
    .expect_err("cross-session learning is off by default");
    assert!(matches!(
        refused,
        AdaptationError::ScopeNotPermitted {
            scope: Scope::CrossSession
        }
    ));

    let store_default = PolicyStore::new(
        scratch.path().join("state-default"),
        &repo,
        LearningOptions::default(),
    );
    let opted_policy = update_policy(
        None,
        &observations,
        Scope::CrossSession,
        None,
        &LearningOptions::session_only().with_cross_session(),
    )
    .expect("explicit opt-in allows the global policy to be computed");
    assert!(
        store_default
            .write(&opted_policy, &provenance("engine", "not opted in"))
            .is_err(),
        "a store without opt-in refuses the global policy"
    );
    assert!(store_default.read_current().is_none());

    // With opt-in the write succeeds, but no private content leaves the repo.
    let store_global = PolicyStore::new(
        scratch.path().join("state-global"),
        &repo,
        LearningOptions::session_only().with_cross_session(),
    );
    store_global
        .write(&opted_policy, &provenance("user:alice", "global opt-in"))
        .expect("opt-in allows a global write");
    let stored = store_global.read_current().expect("the policy is readable");
    assert_eq!(stored.scope, Scope::CrossSession);
    assert!(stored.project_root.is_none());
    assert!(
        stored.based_on.is_empty(),
        "global records carry no evidence refs"
    );
    let json = serde_json::to_string(&stored).unwrap();
    assert!(
        !json.contains("repo"),
        "no repository path in a global record: {json}"
    );
}

#[test]
fn provenance_is_recorded_in_the_append_only_history() {
    let scratch = Scratch::new("provenance");
    let repo = scratch.path().join("repo");
    let observations: Vec<_> = dominant_failed_checks(&repo)
        .iter()
        .map(|record| eligible(&classification(record)).clone())
        .collect();
    let policy = update_policy(
        None,
        &observations,
        Scope::Project,
        Some(repo.clone()),
        &LearningOptions::session_only().with_project(),
    )
    .unwrap();
    let store = PolicyStore::new(
        scratch.path().join("state"),
        &repo,
        LearningOptions::session_only().with_project(),
    );
    let prov = provenance("human:bob", "reviewed and accepted the change");
    store.write(&policy, &prov).unwrap();

    let history = store.read_history();
    assert_eq!(history.len(), 1, "one append-only row");
    assert_eq!(history[0].sequence, 1);
    assert_eq!(history[0].provenance, prov);
    assert_eq!(history[0].policy, policy);
    assert!(
        !policy.based_on.is_empty(),
        "the policy names the real evidence behind it"
    );
    assert_eq!(
        policy.based_on[0].origin,
        SignalOrigin::Observed,
        "evidence provenance records where a signal came from"
    );
}

#[test]
fn rollback_restores_the_exact_previous_policy() {
    let scratch = Scratch::new("rollback");
    let repo = scratch.path().join("repo");
    let observations: Vec<_> = dominant_failed_checks(&repo)
        .iter()
        .map(|record| eligible(&classification(record)).clone())
        .collect();
    let options = LearningOptions::session_only().with_project();
    let first = update_policy(
        None,
        &observations,
        Scope::Project,
        Some(repo.clone()),
        &options,
    )
    .unwrap();
    let second = update_policy(
        Some(&first),
        &observations,
        Scope::Project,
        Some(repo.clone()),
        &options,
    )
    .unwrap();
    assert_eq!(second.version, first.version + 1);

    let store = PolicyStore::new(scratch.path().join("state"), &repo, options);
    store
        .write(&first, &provenance("engine", "first policy"))
        .unwrap();
    store
        .write(&second, &provenance("engine", "superseding policy"))
        .unwrap();
    assert_eq!(store.read_current(), Some(second.clone()));

    let restored = store
        .rollback(
            first.version,
            &provenance("human:bob", "undo the last change"),
        )
        .expect("rollback to version 1 succeeds");
    assert_eq!(restored, first, "the exact earlier policy comes back");
    assert_eq!(store.read_current(), Some(first.clone()));
    let bytes = std::fs::read(store.current_path()).unwrap();
    assert_eq!(
        bytes,
        serde_json::to_vec_pretty(&first).unwrap(),
        "the current file is byte-for-byte the earlier policy"
    );
    assert_eq!(
        store.read_history().len(),
        3,
        "the rollback itself is audited"
    );

    let missing = store.rollback(99, &provenance("engine", "no such version"));
    assert!(matches!(
        missing,
        Err(AdaptationError::UnknownVersion { version: 99 })
    ));
}

#[test]
fn human_corrections_are_tagged_distinctly_from_observed_signals() {
    let scratch = Scratch::new("human");
    let correction = EvidenceRecord::HumanCorrection {
        text: "Use the structured diagnostics, not a blind retry".to_string(),
        evidence: evidence("conversation", "msg-42"),
        session_id: "s".to_string(),
        project_root: scratch.path().to_path_buf(),
    };
    let correction_class = classification(&correction);
    let observation = eligible(&correction_class);
    assert_eq!(observation.origin, SignalOrigin::HumanCorrection);
    assert_eq!(observation.outcome, OutcomeKind::HumanCorrection);
    assert_eq!(
        observation.evidence[0].origin,
        SignalOrigin::HumanCorrection
    );

    let observed_record = check("cargo test", true, "run-1", scratch.path());
    let observed_class = classification(&observed_record);
    let observed = eligible(&observed_class);
    assert_eq!(observed.origin, SignalOrigin::Observed);
    assert_ne!(observed.origin, observation.origin);

    // A forged origin on a model claim cannot turn narrative into a correction.
    let mut forged = evidence("narrative", "turn-1");
    forged.origin = SignalOrigin::HumanCorrection;
    let claim = EvidenceRecord::ModelClaim {
        text: "trust me".to_string(),
        evidence: forged,
        session_id: "s".to_string(),
        project_root: scratch.path().to_path_buf(),
    };
    assert_eq!(
        ineligible(&classification(&claim)),
        Eligibility::ModelNarrative
    );
}

#[test]
fn no_permission_or_security_field_is_reachable() {
    let repo = PathBuf::from("/repo");
    let observations: Vec<_> = dominant_failed_checks(&repo)
        .iter()
        .map(|record| eligible(&classification(record)).clone())
        .collect();
    let policy = update_policy(
        None,
        &observations,
        Scope::Project,
        Some(repo),
        &LearningOptions::session_only().with_project(),
    )
    .unwrap();
    let json = serde_json::to_string(&policy).unwrap();
    for forbidden in [
        "local_only",
        "allow_cloud",
        "no_network",
        "permission",
        "egress",
        "security",
        "network",
        "cloud",
        "verification",
    ] {
        assert!(
            !json.contains(forbidden),
            "a learnable policy exposed a security field: {forbidden} in {json}"
        );
    }

    // Unknown JSON fields cannot smuggle a permission into the policy.
    let with_injection = json.trim_end_matches('}').to_string() + ",\"allow_cloud\":true}";
    let loaded: AdaptationPolicy = serde_json::from_str(&with_injection).unwrap();
    assert!(!serde_json::to_string(&loaded)
        .unwrap()
        .contains("allow_cloud"));
}

#[test]
fn policy_parameters_and_provenance_survive_a_serde_round_trip() {
    let repo = PathBuf::from("/repo");
    let observations: Vec<_> = dominant_failed_checks(&repo)
        .iter()
        .map(|record| eligible(&classification(record)).clone())
        .collect();
    let policy = update_policy(
        None,
        &observations,
        Scope::Project,
        Some(repo),
        &LearningOptions::session_only().with_project(),
    )
    .unwrap();
    let bytes = serde_json::to_vec(&policy).unwrap();
    let back: AdaptationPolicy = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back, policy);
    assert_eq!(back.status, PolicyStatus::Active);
}

#[test]
fn many_failures_are_bounded_by_the_hard_ceiling() {
    let scratch = Scratch::new("bounded");
    let mut session = SessionAdaptation::new("s", scratch.path());
    for i in 0..200 {
        session.observe(&check(
            "cargo test",
            true,
            &format!("run-{i}"),
            scratch.path(),
        ));
    }
    let params = session.policy().parameters;
    assert_eq!(params.tool_rounds, MAX_TOOL_ROUNDS);
    assert!(params.tool_rounds <= MAX_TOOL_ROUNDS);
    assert_eq!(params.recovery, RecoveryHint::AffectedDependencies);
}

#[test]
fn a_torn_history_tail_does_not_lose_earlier_policies() {
    let scratch = Scratch::new("torn");
    let repo = scratch.path().join("repo");
    let observations: Vec<_> = dominant_failed_checks(&repo)
        .iter()
        .map(|record| eligible(&classification(record)).clone())
        .collect();
    let options = LearningOptions::session_only().with_project();
    let first = update_policy(
        None,
        &observations,
        Scope::Project,
        Some(repo.clone()),
        &options,
    )
    .unwrap();
    let second = update_policy(
        Some(&first),
        &observations,
        Scope::Project,
        Some(repo.clone()),
        &options,
    )
    .unwrap();
    let store = PolicyStore::new(scratch.path().join("state"), &repo, options);
    store.write(&first, &provenance("engine", "one")).unwrap();
    store.write(&second, &provenance("engine", "two")).unwrap();

    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(store.history_path())
        .unwrap();
    file.write_all(b"{\"sequence\":3,\"policy\":").unwrap();
    drop(file);

    let history = store.read_history();
    assert_eq!(history.len(), 2, "the torn trailing row is dropped");
    assert_eq!(history[1].policy, second, "complete rows survive");
}

#[test]
fn a_real_run_ledger_recovery_row_is_classified_from_verification() {
    let scratch = Scratch::new("run");
    let xencode = scratch.path().join(".xencode");
    let session = "session-r".to_string();
    let run = RunRecord {
        run_id: "run-abc".to_string(),
        ts_unix_ms: 1_700_000_000_000,
        duration_ms: 900,
        rounds: 2,
        session: Some(session.clone()),
        model: None,
        provider: None,
        source: None,
        approvals: vec![],
        recording: None,
        note: String::new(),
        computer: None,
    };
    append_run(&xencode, &run).unwrap();
    let stored = run_by_id(&xencode, "run-abc").expect("the real run reads back");
    let row = xencode_context_rs::run_evidence(&xencode, &stored);
    assert!(
        !row.verified(),
        "no checks ran, so recovery is not confirmed"
    );

    let record = EvidenceRecord::from_run_evidence(
        &row,
        evidence("runs.jsonl", "run-abc"),
        &session,
        scratch.path(),
    );
    assert_eq!(
        ineligible(&classification(&record)),
        Eligibility::UnverifiedClaim
    );
}

/// The release-gate evaluation: a fixed baseline versus session adaptation,
/// split strictly by time. The numbers are printed, never assumed.
#[test]
fn chronological_holdout_compares_adaptation_with_a_fixed_baseline() {
    // Training events all precede the held-out events. A later outcome never
    // leaks into the policy that is evaluated.
    let training: Vec<FixtureEvent> = vec![
        fixture("test-1", true, RecoveryHint::AffectedDependencies),
        fixture("test-2", true, RecoveryHint::AffectedDependencies),
        fixture("test-3", true, RecoveryHint::AffectedDependencies),
        edit_fixture("edit-1"),
    ];
    let holdout: Vec<FixtureEvent> = vec![
        fixture("test-held-1", true, RecoveryHint::AffectedDependencies),
        fixture("test-held-2", true, RecoveryHint::AffectedDependencies),
        fixture("test-held-3", true, RecoveryHint::AffectedDependencies),
    ];
    for (index, event) in training.iter().enumerate() {
        for later in holdout.iter() {
            assert!(
                event.record.evidence().record_id < later.record.evidence().record_id,
                "training row {index} must precede every held-out row"
            );
        }
    }

    let report = evaluate_holdout(&training, &holdout);
    println!("--- XI-5 chronological holdout (real counts) ---");
    println!("held-out events:        {}", report.holdout_total);
    println!(
        "fixed baseline hits:    {}/{}",
        report.baseline_hits, report.holdout_total
    );
    println!(
        "session adaptive hits:  {}/{}",
        report.adaptive_hits, report.holdout_total
    );
    println!(
        "baseline recovery/round hits: {}/{}",
        report.baseline_recovery_hits, report.baseline_round_hits
    );
    println!(
        "adaptive recovery/round hits: {}/{}",
        report.adaptive_recovery_hits, report.adaptive_round_hits
    );
    println!(
        "learned parameters: rounds {}, recovery {}",
        report.adaptive_parameters.tool_rounds,
        report.adaptive_parameters.recovery.as_str()
    );
    println!(
        "release gate (adaptive > fixed on held-out): {}",
        report.adaptation_outperforms_baseline()
    );

    assert_eq!(report.holdout_total, 3);
    assert_eq!(report.baseline_hits, 0);
    assert_eq!(report.adaptive_hits, 3);
    assert_eq!(
        report.adaptive_parameters.recovery,
        RecoveryHint::AffectedDependencies
    );
    assert_eq!(
        report.adaptive_parameters.tool_rounds,
        DEFAULT_TOOL_ROUNDS + 2 * 4
    );
    assert!(
        report.adaptation_outperforms_baseline(),
        "adaptation must be measured to beat the fixed baseline on held-out events"
    );
}

fn fixture(id: &str, failed: bool, expected: RecoveryHint) -> FixtureEvent {
    FixtureEvent {
        record: check("cargo test", failed, id, Path::new("/fixture")),
        expected_recovery: expected,
        needs_rounds: 6,
    }
}

fn edit_fixture(id: &str) -> FixtureEvent {
    FixtureEvent {
        record: EvidenceRecord::EditOutcome {
            verified_changed: false,
            checked_on_disk: true,
            evidence: evidence("git-diff", id),
            session_id: "session-a".to_string(),
            project_root: PathBuf::from("/fixture"),
        },
        expected_recovery: RecoveryHint::StructuredDiagnostics,
        needs_rounds: 6,
    }
}

// Keep imported boundary constants honest in clippy's eyes even if a test
// above changes shape later.
#[test]
fn observations_cannot_be_pulled_across_a_repo_boundary_and_writes_need_provenance() {
    let scratch = Scratch::new("guards");
    let repo_a = scratch.path().join("repo-a");
    let repo_b = scratch.path().join("repo-b");
    let observations: Vec<_> = dominant_failed_checks(&repo_b)
        .iter()
        .map(|record| eligible(&classification(record)).clone())
        .collect();
    let refused = update_policy(
        None,
        &observations,
        Scope::Project,
        Some(repo_a.clone()),
        &LearningOptions::session_only().with_project(),
    )
    .expect_err("observations from repo-b cannot become a repo-a policy");
    assert!(matches!(
        refused,
        AdaptationError::ObservationOutsideRepo { .. }
    ));

    let own: Vec<_> = dominant_failed_checks(&repo_a)
        .iter()
        .map(|record| eligible(&classification(record)).clone())
        .collect();
    let policy = update_policy(
        None,
        &own,
        Scope::Project,
        Some(repo_a.clone()),
        &LearningOptions::session_only().with_project(),
    )
    .unwrap();
    let store = PolicyStore::new(
        scratch.path().join("state"),
        &repo_a,
        LearningOptions::session_only().with_project(),
    );
    let anonymous = Provenance {
        actor: String::new(),
        written_at_ms: 1,
        note: "no actor".to_string(),
    };
    assert!(matches!(
        store.write(&policy, &anonymous),
        Err(AdaptationError::MissingProvenance)
    ));
    assert!(store.read_current().is_none());
}

#[test]
fn defaults_stay_conservative() {
    assert_eq!(DEFAULT_TOOL_ROUNDS, 4);
    const { assert!(MAX_TOOL_ROUNDS > DEFAULT_TOOL_ROUNDS) };
    let params = AdaptationParameters {
        tool_rounds: DEFAULT_TOOL_ROUNDS,
        recovery: RecoveryHint::Proceed,
    };
    assert_eq!(params.recovery, RecoveryHint::Proceed);
    let options = LearningOptions::default();
    assert!(!options.allow_cross_session);
    assert!(!options.allow_project);
}
