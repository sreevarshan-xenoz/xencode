//! Real behaviour tests for the coordination slice. No mocks: every decision
//! comes from a real coordinator over real values, and the end-to-end test
//! writes and reads actual JSON files in a real temporary directory.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use xencode_context_rs::{ContextCaps, EvalItem, EvalReport, TaskShape};
use xencode_intelligence_rs::*;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A scratch directory that is always removed, including on a failed test.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "xencode-intelligence-{}-{name}-{}",
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

fn local_model(id: &str) -> ModelCandidate {
    ModelCandidate {
        id: ModelId::new(id),
        available: true,
        egress: Measurement::Known(ModelEgress::Local),
        supports_vision: Measurement::Unknown,
        context_window: Measurement::Known(8192),
    }
}

fn cloud_model(id: &str) -> ModelCandidate {
    ModelCandidate {
        id: ModelId::new(id),
        available: true,
        egress: Measurement::Known(ModelEgress::Cloud),
        supports_vision: Measurement::Unknown,
        context_window: Measurement::Known(32_768),
    }
}

fn request(prompt: &str) -> IntelligenceRequest {
    IntelligenceRequest::new(
        Task::new(prompt),
        "session-1",
        PathBuf::from("/tmp/project"),
    )
}

fn has_evidence(decision: &IntelligenceDecision, needle: &str) -> bool {
    decision
        .rationale
        .iter()
        .any(|item| item.fact.contains(needle))
}

#[test]
fn identical_inputs_produce_identical_decisions() {
    let scratch = Scratch::new("deterministic");
    let capability = CapabilityView::new(vec![
        local_model("local:zeta"),
        cloud_model("cloud:alpha"),
        local_model("local:alpha"),
    ]);
    let coordinator = Coordinator::new(capability, scratch.path());
    let mut input = request("why does the cache break on a cold start");
    input.measurements.context = Measurement::Known(ContextCaps::for_free_space(12_000));

    let first = coordinator.decide(&input).expect("first decision");
    let second = coordinator.decide(&input).expect("second decision");

    assert_eq!(first, second, "the same inputs must give the same decision");
    assert_eq!(first.task_shape, TaskShape::Bugfix);
    assert_eq!(
        first.selected_model.as_ref().map(ModelId::as_str),
        Some("local:alpha")
    );
    assert_eq!(
        first.context_budget,
        Measurement::Known(ContextBudget::from(ContextCaps::for_free_space(12_000)))
    );
    assert!(first
        .rationale
        .iter()
        .any(|item| item.source == sources::TASK_PROMPT));
    assert!(has_evidence(&first, "the prompt says break"));
}

#[test]
fn unknown_measurements_stay_unknown() {
    let scratch = Scratch::new("unknown");
    let mut unmeasured = local_model("local:unknown");
    unmeasured.egress = Measurement::Unknown;
    let coordinator = Coordinator::new(CapabilityView::new(vec![unmeasured]), scratch.path());

    let decision = coordinator
        .decide(&request("what does this function do"))
        .expect("an unmeasured model yields a decision, not an error");

    assert_eq!(decision.task_shape, TaskShape::General);
    assert_eq!(
        decision.selected_model, None,
        "an unmeasured route cannot be chosen"
    );
    assert_eq!(decision.context_budget, Measurement::Unknown);
    assert_eq!(decision.reasoning_budget, Measurement::Unknown);
    assert!(has_evidence(
        &decision,
        "where it sends a prompt was not measured"
    ));
    assert!(has_evidence(&decision, "no context budget was measured"));
    assert!(has_evidence(&decision, "no reasoning budget was measured"));
    assert!(has_evidence(&decision, "none was invented"));
}

#[test]
fn unsupported_capabilities_are_refused() {
    let scratch = Scratch::new("unsupported");
    let mut unavailable = local_model("local:offline");
    unavailable.available = false;
    let mut text_only = local_model("local:text-only");
    text_only.supports_vision = Measurement::Known(false);
    let mut vision = local_model("local:vision");
    vision.supports_vision = Measurement::Known(true);
    let unmeasured_vision = local_model("local:unmeasured-vision");

    let coordinator = Coordinator::new(
        CapabilityView::new(vec![unavailable, text_only, vision, unmeasured_vision]),
        scratch.path(),
    );
    let mut input = request("describe the screenshot attached to this turn");
    input.task.requires_vision = true;

    let decision = coordinator.decide(&input).expect("vision decision");
    assert_eq!(
        decision.selected_model.as_ref().map(ModelId::as_str),
        Some("local:vision")
    );
    assert!(has_evidence(
        &decision,
        "local:offline was refused: the caller reported it unavailable"
    ));
    assert!(has_evidence(
        &decision,
        "local:text-only was refused: it reports no vision support"
    ));
    assert!(has_evidence(
        &decision,
        "local:unmeasured-vision was refused: its vision support was not measured"
    ));

    // Without a vision requirement the unavailable model is still skipped.
    let mut plain = request("explain this module");
    plain.task.requires_vision = false;
    let coordinator = Coordinator::new(
        CapabilityView::new(vec![{
            let mut offline = local_model("local:offline");
            offline.available = false;
            offline
        }]),
        scratch.path(),
    );
    let decision = coordinator.decide(&plain).expect("plain decision");
    assert_eq!(decision.selected_model, None);
    assert!(has_evidence(&decision, "reported it unavailable"));
    assert!(!has_evidence(&decision, "was selected: it is available"));
}

#[test]
fn local_only_cannot_be_overridden_and_cloud_needs_permission() {
    let scratch = Scratch::new("constraints");
    let coordinator = Coordinator::new(
        CapabilityView::new(vec![cloud_model("cloud:far"), local_model("local:near")]),
        scratch.path(),
    );

    let mut locked = request("summarise the changelog");
    locked.constraints.local_only = true;
    locked.constraints.allow_cloud = true;
    let decision = coordinator.decide(&locked).expect("locked decision");
    assert_eq!(
        decision.selected_model.as_ref().map(ModelId::as_str),
        Some("local:near")
    );
    assert!(has_evidence(
        &decision,
        "local-only is set and cannot be overridden here"
    ));

    let mut no_network = request("summarise the changelog");
    no_network.constraints.local_only = false;
    no_network.constraints.allow_cloud = true;
    no_network.constraints.no_network = true;
    let decision = coordinator
        .decide(&no_network)
        .expect("no-network decision");
    assert_eq!(
        decision.selected_model.as_ref().map(ModelId::as_str),
        Some("local:near")
    );
    assert!(has_evidence(
        &decision,
        "network use is disabled for this turn"
    ));

    let mut no_permission = request("summarise the changelog");
    no_permission.constraints.local_only = false;
    no_permission.constraints.no_network = false;
    no_permission.constraints.allow_cloud = false;
    let decision = coordinator
        .decide(&no_permission)
        .expect("no-permission decision");
    assert_eq!(
        decision.selected_model.as_ref().map(ModelId::as_str),
        Some("local:near")
    );
    assert!(has_evidence(
        &decision,
        "the caller did not allow cloud models"
    ));

    let mut allowed = request("summarise the changelog");
    allowed.constraints = Constraints::cloud_allowed();
    let decision = coordinator.decide(&allowed).expect("allowed decision");
    assert_eq!(
        decision.selected_model.as_ref().map(ModelId::as_str),
        Some("cloud:far")
    );
    assert!(has_evidence(&decision, "it routes off-machine"));
}

#[test]
fn errors_are_errors_and_never_become_decisions() {
    let scratch = Scratch::new("errors");
    let coordinator = Coordinator::new(
        CapabilityView::new(vec![local_model("local:near")]),
        scratch.path(),
    );

    assert!(matches!(
        coordinator.decide(&request("   ")),
        Err(IntelligenceError::EmptyTask)
    ));
    let mut no_session = request("a real prompt");
    no_session.session_id = String::new();
    assert!(matches!(
        coordinator.decide(&no_session),
        Err(IntelligenceError::EmptySessionId)
    ));

    let original = request("a real prompt");
    let decision = coordinator.decide(&original).expect("real decision");
    let mut other = request("a different prompt");
    other.session_id = "session-2".to_string();
    // The two decisions happen to have the same contents, but they belong to
    // different turns: their identifiers must still differ.
    assert_ne!(
        coordinator.decision_id(&original, &decision),
        coordinator.decision_id(&other, &decision)
    );
    let mut forged = decision.clone();
    forged.selected_model = None;
    let execution = ExecutionRecord {
        status: ExecutionStatus::Failed {
            reason: "the command exited 3".to_string(),
        },
        summary: "nothing was changed".to_string(),
    };
    let verification = VerificationRecord {
        status: VerificationStatus::NotRun,
        checks: Vec::new(),
        detail: "no checks ran".to_string(),
    };
    assert!(matches!(
        coordinator.record(&original, &forged, execution.clone(), verification.clone()),
        Err(IntelligenceError::DecisionMismatch)
    ));
    assert!(
        !coordinator.decisions_dir().exists(),
        "a refused record must not write a decision"
    );

    assert!(matches!(
        coordinator.decide(&request("   ")).map(|_| ()),
        Err(IntelligenceError::EmptyTask)
    ));
    assert!(
        coordinator.read_decision("does-not-exist").is_err(),
        "an absent decision is an error, not a made-up decision"
    );
}

#[test]
fn an_end_to_end_turn_persists_a_decision_and_its_provenance() {
    let scratch = Scratch::new("end-to-end");
    let project = scratch.path().join("project");
    std::fs::create_dir_all(&project).expect("project directory must be created");
    let task_file = project.join("task.md");
    let prompt = "fix the buffer that breaks on a cold start";
    std::fs::write(&task_file, prompt).expect("task file must be written");
    let read_back = std::fs::read_to_string(&task_file).expect("task file must be read");
    assert_eq!(read_back, prompt, "the real filesystem supplied the prompt");

    let report = EvalReport {
        queries: 3,
        top_k: 5,
        reranked: false,
        recall_at: vec![0.5, 0.7, 0.9, 1.0, 1.0],
        precision_at: vec![0.5, 0.4, 0.3, 0.25, 0.2],
        mrr: 0.667,
        hits: Vec::new(),
    };
    let capability = CapabilityView::new(vec![local_model("llamacpp:local-gguf")]);
    let coordinator = Coordinator::new(capability, scratch.path().join("records"));

    let mut input = IntelligenceRequest::new(Task::new(&read_back), "session-e2e", project.clone());
    input.measurements.context = Measurement::Known(ContextCaps::for_free_space(18_000));
    input.measurements.reasoning = Measurement::Known(ReasoningBudget {
        max_rounds: 8,
        allow_deep_think: false,
    });
    input.task.retrieval_evidence = Some(RetrievalEvidence::from_report("local gold set", &report));
    input.task.probe = Some(EvalItem {
        query: prompt.to_string(),
        expected: vec!["src/buffer.rs".to_string()],
        shape: Some("bugfix".to_string()),
    });

    let decision = coordinator.decide(&input).expect("end-to-end decision");
    assert_eq!(decision.task_shape, TaskShape::Bugfix);
    assert_eq!(
        decision.selected_model.as_ref().map(ModelId::as_str),
        Some("llamacpp:local-gguf")
    );
    assert_eq!(decision.verification_policy, VerificationPolicy::Required);
    assert!(has_evidence(&decision, "local gold set measured 3 queries"));
    assert!(has_evidence(&decision, "src/buffer.rs"));
    assert!(has_evidence(&decision, "read as bugfix"));

    let outcome = coordinator
        .record(
            &input,
            &decision,
            ExecutionRecord {
                status: ExecutionStatus::Completed,
                summary: "edited the buffer and ran the crate tests".to_string(),
            },
            VerificationRecord {
                status: VerificationStatus::Passed,
                checks: vec!["cargo test -p xencode-intelligence-rs".to_string()],
                detail: "the real test run exited 0".to_string(),
            },
        )
        .expect("outcome must be recorded");
    assert_eq!(outcome.execution.status, ExecutionStatus::Completed);
    assert_eq!(outcome.verification.status, VerificationStatus::Passed);

    let decisions_dir = coordinator.decisions_dir();
    let outcomes_dir = coordinator.outcomes_dir();
    let decision_file = decisions_dir.join(format!("{}.json", outcome.decision_id));
    let outcome_file = outcomes_dir.join(format!("{}.json", outcome.decision_id));
    assert!(decision_file.is_file(), "the decision must be on disk");
    assert!(outcome_file.is_file(), "the outcome must be on disk");

    let saved_decision = coordinator
        .read_decision(&outcome.decision_id)
        .expect("saved decision must parse");
    let saved_outcome = coordinator
        .read_outcome(&outcome.decision_id)
        .expect("saved outcome must parse");
    assert_eq!(saved_decision, decision);
    assert_eq!(saved_outcome, outcome);

    let decision_json =
        std::fs::read_to_string(&decision_file).expect("decision JSON must be readable");
    let outcome_json =
        std::fs::read_to_string(&outcome_file).expect("outcome JSON must be readable");
    assert!(decision_json.contains("llamacpp:local-gguf"));
    assert!(decision_json.contains("the task prompt"));
    assert!(outcome_json.contains(&outcome.decision_id));
    assert!(outcome_json.contains("cargo test -p xencode-intelligence-rs"));
    assert_eq!(
        outcome.decision_id.len(),
        16,
        "a stable id is 16 hex digits"
    );
    assert!(outcome.decision_id.chars().all(|c| c.is_ascii_hexdigit()));

    // A recorded outcome never teaches the coordinator anything. An identical
    // request still produces the identical decision after a failed outcome too.
    let repeated = coordinator.decide(&input).expect("repeated decision");
    assert_eq!(
        repeated, decision,
        "recorded outcomes do not change decisions"
    );
}

#[test]
fn a_declared_shape_uses_the_shared_shape_parser() {
    let scratch = Scratch::new("shape");
    let coordinator = Coordinator::new(
        CapabilityView::new(vec![local_model("local:near")]),
        scratch.path(),
    );
    let mut input = request("rename the reader and its callers");
    input.task.declared_shape = Some("bugfix".to_string());
    let decision = coordinator.decide(&input).expect("declared shape decision");
    assert_eq!(decision.task_shape, TaskShape::Bugfix);
    assert!(has_evidence(&decision, "the caller declared shape bugfix"));

    input.task.declared_shape = Some("refactor".to_string());
    let decision = coordinator.decide(&input).expect("unknown shape decision");
    assert_eq!(decision.task_shape, TaskShape::General);
    assert!(has_evidence(&decision, "is not a known shape"));
}
