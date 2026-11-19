//! Real behaviour tests for the adaptive reasoning controller (XI-4).
//!
//! No mocks: every plan comes from a real `TurnTrace` that is written to
//! `turns.jsonl` in a temporary directory and read back with the same reader
//! the CLI uses, then handed to the pure controller. Timings come from real
//! `Instant` readings over the checked-in seeded bug shapes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use xencode_config_rs::XencodeConfig;
use xencode_context_rs::{
    append_trace, read_recent_traces, BugShape, ChecksVerdict, ContinuationNote, TaskShape,
    ToolTrace, TurnTrace, ESCALATION_FAILED_EDIT_THRESHOLD,
};
use xencode_intelligence_rs::reasoning::decide;
use xencode_intelligence_rs::*;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "xencode-reasoning-{}-{name}-{}",
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

fn tool(name: &str, outcome: &str, arguments: Option<&str>, tail: Option<&str>) -> ToolTrace {
    ToolTrace {
        name: name.to_string(),
        outcome: outcome.to_string(),
        arguments: arguments.map(str::to_string),
        tail: tail.map(str::to_string),
    }
}

/// A normal turn with a successful tool and repository evidence present, so
/// the only signals are the ones a test adds explicitly.
fn healthy_trace() -> TurnTrace {
    let mut trace = TurnTrace::new(1_250, 3);
    trace.retrieved_files = vec!["src/lib.rs".to_string()];
    trace.tools = vec![tool(
        "read_file",
        "done",
        Some("{\"path\":\"src/lib.rs\"}"),
        None,
    )];
    trace
}

/// Write a trace to a real `turns.jsonl` and read it back with the CLI reader.
fn roundtrip(dir: &Path, trace: &TurnTrace) -> TurnTrace {
    std::fs::create_dir_all(dir).expect("trace directory must be created");
    let xencode = dir.join(".xencode");
    append_trace(&xencode, trace).expect("append_trace must write");
    let rows = read_recent_traces(&xencode, 1);
    assert_eq!(rows.len(), 1, "the written row must read back");
    rows.into_iter().next().unwrap()
}

fn failed_edits_trace() -> TurnTrace {
    let mut trace = healthy_trace();
    trace.tools = vec![
        tool("edit_file", "failed", Some("{\"path\":\"src/a.rs\"}"), None),
        tool(
            "write_file",
            "failed",
            Some("{\"path\":\"src/b.rs\"}"),
            None,
        ),
    ];
    trace
}

fn escalation_input<'a>(
    trace: &'a TurnTrace,
    shape: TaskShape,
    constraints: &'a Constraints,
    budget: Measurement<ReasoningBudget>,
    deep_think_is_cloud: bool,
) -> ReasoningInput<'a> {
    ReasoningInput {
        trace,
        shape,
        constraints,
        budget,
        deep_think_available: true,
        deep_think_is_cloud,
    }
}

fn has_evidence(plan: &ReasoningPlan, needle: &str) -> bool {
    plan.rationale.iter().any(|item| item.fact.contains(needle))
}

#[test]
fn a_simple_turn_stays_fast_with_no_recovery() {
    let scratch = Scratch::new("fast");
    let trace = roundtrip(scratch.path(), &healthy_trace());
    let constraints = Constraints::local_only();

    let plan = decide_from_trace(&trace, TaskShape::General, &constraints);

    assert_eq!(plan.stage, ReasoningStage::Fast);
    assert_eq!(plan.recovery, RecoveryStrategy::Proceed);
    assert_eq!(plan.verification, VerificationIntensity::Standard);
    assert!(!plan.escalation_eligible);
    assert_eq!(
        plan.max_tool_rounds, 4,
        "the fast stage uses its own small cap"
    );
    assert!(has_evidence(&plan, "0 failed edit attempts"));
    assert!(has_evidence(&plan, "model-verbalized confidence"));
}

#[test]
fn repeated_failed_edits_escalate_from_the_observed_signal() {
    let scratch = Scratch::new("escalate");
    // Leave the writer's summary at zero: the controller must recompute the
    // signal from the real tool rows it just read back from disk.
    let trace = roundtrip(scratch.path(), &failed_edits_trace());
    assert_eq!(trace.failed_edit_count, 0, "an older row has no summary");
    let constraints = Constraints::local_only();
    let input = escalation_input(
        &trace,
        TaskShape::Bugfix,
        &constraints,
        Measurement::Known(ReasoningBudget {
            max_rounds: 16,
            allow_deep_think: true,
        }),
        false,
    );

    let plan = decide(&input);

    assert_eq!(plan.stage, ReasoningStage::Escalate);
    assert_eq!(plan.recovery, RecoveryStrategy::DeepThink);
    assert!(plan.escalation_eligible);
    assert!(!plan.escalation_is_cloud);
    assert_eq!(plan.verification, VerificationIntensity::Thorough);
    assert!(has_evidence(&plan, "2 failed edit attempts"));
    assert!(
        has_evidence(&plan, "edit_file"),
        "the failing tool is named"
    );
    assert!(has_evidence(&plan, "escalation event from the DT-1"));
    assert_eq!(plan.max_tool_rounds, 16);
}

#[test]
fn a_measured_budget_is_a_hard_cap_even_when_escalating() {
    let scratch = Scratch::new("budget");
    let trace = roundtrip(scratch.path(), &failed_edits_trace());
    let constraints = Constraints::local_only();

    for cap in [0, 3, 8] {
        let input = escalation_input(
            &trace,
            TaskShape::Bugfix,
            &constraints,
            Measurement::Known(ReasoningBudget {
                max_rounds: cap,
                allow_deep_think: true,
            }),
            false,
        );
        let plan = decide(&input);
        assert_eq!(
            plan.max_tool_rounds, cap,
            "a measured budget of {cap} is never exceeded"
        );
        assert!(
            plan.max_tool_rounds <= HARD_MAX_TOOL_ROUNDS,
            "the absolute ceiling also holds"
        );
    }

    // Even the escalation stage cannot exceed the absolute ceiling.
    let no_budget = escalation_input(
        &trace,
        TaskShape::Bugfix,
        &constraints,
        Measurement::Unknown,
        false,
    );
    let plan = decide(&no_budget);
    assert_eq!(plan.stage, ReasoningStage::Escalate);
    assert!(
        plan.max_tool_rounds <= DEFAULT_MAX_TOOL_ROUNDS,
        "an unmeasured budget defaults conservatively"
    );
}

#[test]
fn the_same_failing_call_is_not_repeated_without_edits() {
    let scratch = Scratch::new("repeat");
    let mut trace = healthy_trace();
    // Two identical failed search calls: a repeated failure, but not a failed
    // edit, so the DT-1 escalation signal is absent.
    let arguments = "{\"pattern\":\"TODO\"}";
    trace.tools = vec![
        tool("search_files", "failed", Some(arguments), None),
        tool("search_files", "failed", Some(arguments), None),
    ];
    let trace = roundtrip(scratch.path(), &trace);
    let constraints = Constraints::local_only();
    let input = escalation_input(
        &trace,
        TaskShape::General,
        &constraints,
        Measurement::Unknown,
        false,
    );

    let plan = decide(&input);

    assert_eq!(plan.stage, ReasoningStage::Normal);
    assert_eq!(plan.recovery, RecoveryStrategy::StopRepeating);
    assert!(!plan.escalation_eligible);
    assert!(has_evidence(&plan, "search_files failed 2 times"));
    assert!(has_evidence(
        &plan,
        "repeating it unchanged cannot make progress"
    ));
}

#[test]
fn user_constraints_override_verification_and_deep_think() {
    let scratch = Scratch::new("constraints");
    let trace = roundtrip(scratch.path(), &failed_edits_trace());

    let mut waived = Constraints::local_only();
    waived.require_verification = false;
    let plan = decide_from_trace(&trace, TaskShape::Bugfix, &waived);
    assert_eq!(plan.verification, VerificationIntensity::Light);
    assert!(has_evidence(&plan, "the caller waived verification"));

    // A measured budget that forbids deep think vetoes escalation even though
    // the failed-edit signal and a configured deeper model are both present.
    let input = escalation_input(
        &trace,
        TaskShape::Bugfix,
        &waived,
        Measurement::Known(ReasoningBudget {
            max_rounds: 12,
            allow_deep_think: false,
        }),
        false,
    );
    let plan = decide(&input);
    assert!(!plan.escalation_eligible);
    assert_ne!(plan.recovery, RecoveryStrategy::DeepThink);
    assert!(has_evidence(&plan, "does not allow deep think"));
}

#[test]
fn cloud_escalation_requires_allow_cloud_models() {
    let scratch = Scratch::new("cloud");
    let trace = roundtrip(scratch.path(), &failed_edits_trace());
    let budget = Measurement::Known(ReasoningBudget {
        max_rounds: 16,
        allow_deep_think: true,
    });

    let denied = XencodeConfig::default();
    assert!(
        !denied.allow_cloud_models,
        "the shipped default denies cloud"
    );
    let constraints = Constraints::from_config(&denied);
    let plan = decide(&escalation_input(
        &trace,
        TaskShape::Bugfix,
        &constraints,
        budget,
        true,
    ));
    assert!(!plan.escalation_eligible);
    assert!(plan.escalation_is_cloud);
    assert!(has_evidence(&plan, "routes off-machine"));
    assert!(has_evidence(&plan, "allow_cloud_models is off"));

    let allowed = XencodeConfig {
        allow_cloud_models: true,
        ..Default::default()
    };
    let constraints = Constraints::from_config(&allowed);
    let plan = decide(&escalation_input(
        &trace,
        TaskShape::Bugfix,
        &constraints,
        budget,
        true,
    ));
    assert!(plan.escalation_eligible, "explicit permission allows it");
    assert!(plan.escalation_is_cloud);
}

#[test]
fn a_local_only_request_vetoes_a_cloud_deeper_model_even_if_cloud_is_allowed() {
    let scratch = Scratch::new("local-veto");
    let trace = roundtrip(scratch.path(), &failed_edits_trace());
    let mut constraints = Constraints::cloud_allowed();
    constraints.local_only = true;
    let plan = decide(&escalation_input(
        &trace,
        TaskShape::Bugfix,
        &constraints,
        Measurement::Known(ReasoningBudget {
            max_rounds: 16,
            allow_deep_think: true,
        }),
        true,
    ));
    assert!(!plan.escalation_eligible);
    assert!(has_evidence(&plan, "local-only/no-network vetoes it"));
}

#[test]
fn every_escalation_traces_to_a_real_recorded_event() {
    let scratch = Scratch::new("traceable");
    let constraints = Constraints::local_only();

    // The same shape and constraints, but one turn failed twice and one did
    // not. Only the observed event may change the plan.
    let clean = roundtrip(&scratch.path().join("clean"), &healthy_trace());
    let mut broken = healthy_trace();
    broken.tools = vec![
        tool("edit_file", "failed", Some("{\"path\":\"src/a.rs\"}"), None),
        tool("edit_file", "failed", Some("{\"path\":\"src/a.rs\"}"), None),
    ];
    broken.no_change_notes.push(ContinuationNote {
        kind: "task_no_edit".to_string(),
        round: 2,
        edit_followed: false,
    });
    let broken = roundtrip(&scratch.path().join("broken"), &broken);

    let input = escalation_input(
        &broken,
        TaskShape::Bugfix,
        &constraints,
        Measurement::Unknown,
        false,
    );
    let escalated = decide(&input);
    let plain = decide_from_trace(&clean, TaskShape::Bugfix, &constraints);

    assert_eq!(plain.stage, ReasoningStage::Fast);
    assert_eq!(escalated.stage, ReasoningStage::Escalate);
    assert_eq!(
        broken.failed_edit_count, 0,
        "summary absent on the disk row"
    );
    assert!(
        has_evidence(&escalated, "3 failed edit attempts"),
        "the count comes from the two failed calls and the one no-change note"
    );
    assert!(has_evidence(&escalated, "1 no-change notes"));
    assert!(
        has_evidence(
            &escalated,
            &format!("threshold {ESCALATION_FAILED_EDIT_THRESHOLD}")
        ),
        "the DT-1 signal names its threshold"
    );
}

#[test]
fn specific_observations_pick_specific_recoveries() {
    let scratch = Scratch::new("recoveries");
    let constraints = Constraints::local_only();

    // Missing tool arguments in a real (redacted) output tail.
    let mut trace = healthy_trace();
    trace.tools = vec![tool(
        "run_command",
        "failed",
        Some("{\"command\":\"apply_patch\"}"),
        Some("error: missing required argument --path"),
    )];
    let trace = roundtrip(scratch.path(), &trace);
    let plan = decide_from_trace(&trace, TaskShape::General, &constraints);
    assert_eq!(plan.recovery, RecoveryStrategy::CorrectInvocation);
    assert_eq!(plan.stage, ReasoningStage::Normal);

    // A failed build check: structured diagnostics, standard verification.
    let mut trace = healthy_trace();
    trace.checks = Some(ChecksVerdict {
        ran: vec!["cargo clippy".to_string()],
        failed: vec!["cargo clippy".to_string()],
        skipped: vec![],
        evidence_ref: Some("tools[2]".to_string()),
    });
    let trace = roundtrip(scratch.path(), &trace);
    let plan = decide_from_trace(&trace, TaskShape::Bugfix, &constraints);
    assert_eq!(plan.recovery, RecoveryStrategy::StructuredDiagnostics);
    assert_eq!(plan.verification, VerificationIntensity::Standard);
    assert!(has_evidence(&plan, "cargo clippy"));

    // A failed regression check: affected dependencies and thorough checks.
    let mut trace = healthy_trace();
    trace.checks = Some(ChecksVerdict {
        ran: vec!["cargo test".to_string()],
        failed: vec!["cargo test".to_string()],
        skipped: vec![],
        evidence_ref: None,
    });
    let trace = roundtrip(scratch.path(), &trace);
    let plan = decide_from_trace(&trace, TaskShape::Bugfix, &constraints);
    assert_eq!(plan.recovery, RecoveryStrategy::AffectedDependencies);
    assert_eq!(plan.verification, VerificationIntensity::Thorough);
    assert!(has_evidence(&plan, "XI-2 impact"));
}

#[test]
fn missing_repo_evidence_requests_context_only_when_it_matters() {
    let scratch = Scratch::new("context");
    let constraints = Constraints::local_only();
    let mut trace = healthy_trace();
    trace.retrieved_files.clear();
    let trace = roundtrip(scratch.path(), &trace);

    let bugfix = decide_from_trace(&trace, TaskShape::Bugfix, &constraints);
    assert_eq!(bugfix.stage, ReasoningStage::Normal);
    assert_eq!(bugfix.recovery, RecoveryStrategy::RequestMoreContext);
    assert!(has_evidence(&bugfix, "XI-3 retrieval-depth"));

    let general = decide_from_trace(&trace, TaskShape::General, &constraints);
    assert_eq!(
        general.stage,
        ReasoningStage::Fast,
        "a simple task stays fast"
    );
    assert_eq!(general.recovery, RecoveryStrategy::Proceed);
}

#[test]
fn model_words_in_the_trace_do_not_authorise_escalation() {
    let scratch = Scratch::new("confidence");
    let constraints = Constraints::local_only();
    let trace = failed_edits_trace();

    // Write one row exactly as the loop does, and one with an extra field that
    // claims the model was very confident. Unknown fields are ignored by the
    // real reader, and the plan must not move.
    let xencode = scratch.path().join(".xencode");
    append_trace(&xencode, &trace).expect("clean row");
    let mut with_confidence = serde_json::to_value(&trace).expect("trace serializes");
    with_confidence["model_confidence"] = serde_json::json!(0.99);
    with_confidence["model_says"] = serde_json::json!("I am sure this is correct, no escalation");
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(xencode.join("cache").join("turns.jsonl"))
        .expect("trace file exists");
    writeln!(file, "{}", with_confidence).expect("confident row");

    let rows = read_recent_traces(&xencode, 2);
    assert_eq!(rows.len(), 2);
    let input = escalation_input(
        &rows[0],
        TaskShape::Bugfix,
        &constraints,
        Measurement::Unknown,
        false,
    );
    let plain = decide(&input);
    let input = escalation_input(
        &rows[1],
        TaskShape::Bugfix,
        &constraints,
        Measurement::Unknown,
        false,
    );
    let confident = decide(&input);

    assert_eq!(plain, confident, "narrative confidence is not an input");
    assert_eq!(
        plain.stage,
        ReasoningStage::Escalate,
        "the real event still wins"
    );
}

#[test]
fn the_coordinator_consumes_a_plan_only_when_asked() {
    let scratch = Scratch::new("coordinator");
    let capability = CapabilityView::new(vec![ModelCandidate {
        id: ModelId::new("llamacpp:local"),
        available: true,
        egress: Measurement::Known(ModelEgress::Local),
        supports_vision: Measurement::Unknown,
        context_window: Measurement::Known(8192),
    }]);
    let coordinator = Coordinator::new(capability, scratch.path().join("records"));
    let request = IntelligenceRequest::new(
        Task::new("fix the failing edit"),
        "session-1",
        scratch.path().to_path_buf(),
    );
    let trace = roundtrip(scratch.path(), &failed_edits_trace());
    let constraints = request.constraints;
    let input = escalation_input(
        &trace,
        TaskShape::Bugfix,
        &constraints,
        Measurement::Unknown,
        false,
    );

    let plain = coordinator.decide(&request).expect("plain decision");
    assert!(plain.reasoning_plan.is_none(), "default path stays inert");

    let reasoned = coordinator
        .decide_with_reasoning(&request, &input)
        .expect("reasoned decision");
    let plan = reasoned.reasoning_plan.as_ref().expect("plan is attached");
    assert_eq!(plan.stage, ReasoningStage::Escalate);
    assert_eq!(reasoned.task_shape, plain.task_shape);
    assert_eq!(reasoned.selected_model, plain.selected_model);

    // A reasoned decision still records a real outcome, and the plan survives
    // the JSON round trip through the coordinator's own files.
    let outcome = coordinator
        .record(
            &request,
            &reasoned,
            ExecutionRecord {
                status: ExecutionStatus::NotRun,
                summary: "deferred to the caller".to_string(),
            },
            VerificationRecord {
                status: VerificationStatus::NotRun,
                checks: Vec::new(),
                detail: "no checks were run by this test".to_string(),
            },
        )
        .expect("a reasoned decision records an outcome");
    let saved = coordinator
        .read_decision(&outcome.decision_id)
        .expect("saved decision parses");
    assert_eq!(saved.reasoning_plan, reasoned.reasoning_plan);

    // An old decision with no plan field still decodes to `None`.
    let mut json = serde_json::to_value(&plain).expect("decision serializes");
    json.as_object_mut().unwrap().remove("reasoning_plan");
    let old: IntelligenceDecision = serde_json::from_value(json).expect("old decision parses");
    assert!(old.reasoning_plan.is_none());
}

#[test]
fn benchmark_over_seeded_shapes_measures_decision_cost_not_task_gain() {
    let scratch = Scratch::new("benchmark");
    let constraints = Constraints::local_only();
    let budget = Measurement::Known(ReasoningBudget {
        max_rounds: 16,
        allow_deep_think: true,
    });
    let mut total_ns: u128 = 0;
    let mut decisions = 0usize;

    // Real seeded bug shapes (EV-1 / XI-9), each with a real failed-edit trace
    // written to disk and read back. Three repeats per shape, as the benchmark
    // foundation requires before it will even compare two records.
    for shape in BugShape::all() {
        let trace = roundtrip(scratch.path(), &failed_edits_trace());
        let input = escalation_input(&trace, TaskShape::Bugfix, &constraints, budget, false);
        for _ in 0..3 {
            let started = Instant::now();
            let plan = decide(&input);
            let elapsed = started.elapsed();
            total_ns += elapsed.as_nanos();
            decisions += 1;
            assert_eq!(plan.stage, ReasoningStage::Escalate, "{}", shape.slug());
            assert!(plan.max_tool_rounds <= 16);
            // Determinism: the same inputs cannot change the plan mid-run.
            assert_eq!(decide(&input), plan);
        }
    }

    assert_eq!(decisions, 30, "ten shapes times three repeats");
    let mean_micros = (total_ns / decisions as u128) / 1_000;
    eprintln!(
        "XI-4 decision benchmark: {decisions} decisions over {} seeded shapes, mean {mean_micros} microseconds each; task-completion gain is UNMEASURED here (needs a GPU-backed model run)",
        BugShape::all().len()
    );
    assert!(
        mean_micros < 100_000,
        "a pure decision should stay far below a hundred milliseconds, measured {mean_micros} us"
    );
}
