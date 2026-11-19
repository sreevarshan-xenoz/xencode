//! Real observability tests (XI-8): every record here is written by a real
//! coordinator into a real temporary directory and read back from disk. No
//! mocks, no fabricated decisions and no invented measurements.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use xencode_context_rs::{ContextCaps, TaskShape, ToolTrace, TurnTrace};
use xencode_intelligence_rs::*;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A scratch directory that is always removed, including on a failed test.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "xencode-intelligence-obs-{}-{name}-{}",
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

fn request(prompt: &str) -> IntelligenceRequest {
    IntelligenceRequest::new(
        Task::new(prompt),
        "session-obs",
        PathBuf::from("/tmp/project"),
    )
}

fn passed_verification() -> VerificationRecord {
    VerificationRecord {
        status: VerificationStatus::Passed,
        checks: vec!["cargo test -p xencode-intelligence-rs".to_string()],
        detail: "the real test run exited 0".to_string(),
    }
}

fn not_run_verification() -> VerificationRecord {
    // The checks vector is deliberately not empty: a populated list of names
    // must still never turn a NotRun verification into a pass.
    VerificationRecord {
        status: VerificationStatus::NotRun,
        checks: vec!["cargo test was never run".to_string()],
        detail: "no checks ran".to_string(),
    }
}

#[test]
fn a_recorded_decision_loads_and_renders_real_values() {
    let scratch = Scratch::new("real-values");
    let root = scratch.path().join("artifacts");
    let coordinator = Coordinator::new(
        CapabilityView::new(vec![local_model("llamacpp:local")]),
        &root,
    );

    let mut input = request("fix the cache that breaks on a cold start");
    input.measurements.context = Measurement::Known(ContextCaps::for_free_space(18_000));
    input.measurements.reasoning = Measurement::Known(ReasoningBudget {
        max_rounds: 8,
        allow_deep_think: false,
    });
    let decision = coordinator.decide(&input).expect("a real decision");
    let expected_budget = match decision.context_budget {
        Measurement::Known(budget) => budget.to_string(),
        Measurement::Unknown => panic!("this decision has a measured budget"),
    };

    // A real trace drives the opt-in reasoning plan, so the inspector's stage
    // and escalation observables come from recorded events, not a guess.
    let mut trace = TurnTrace::new(1_200, 2);
    trace.tools = vec![ToolTrace {
        name: "cargo check".to_string(),
        outcome: "failed".to_string(),
        arguments: None,
        tail: Some("error[E0599]: no method named missing".to_string()),
    }];
    let reasoned = coordinator
        .decide_with_reasoning(
            &input,
            &ReasoningInput::new(&trace, TaskShape::Bugfix, &input.constraints),
        )
        .expect("a reasoned decision");
    assert!(reasoned.reasoning_plan.is_some());

    let outcome = coordinator
        .record(
            &input,
            &reasoned,
            ExecutionRecord {
                status: ExecutionStatus::Completed,
                summary: "edited the cache and ran the crate tests".to_string(),
            },
            passed_verification(),
        )
        .expect("a real recorded outcome");
    let id = outcome.decision_id.clone();

    let loaded = load_decisions_with_notes(&root).expect("decisions must load");
    assert!(loaded.notes.is_empty(), "a complete pair needs no notes");
    assert_eq!(loaded.records.len(), 1);
    let text = intelligence_report(&loaded.records).join("\n");

    assert!(
        text.contains("INTELLIGENCE INSPECTOR"),
        "header present: {text}"
    );
    assert!(
        text.contains("1 decision"),
        "header counts real records: {text}"
    );
    assert!(
        text.contains(&format!("#{id}")),
        "the decision id is on its group: {text}"
    );
    assert!(
        text.contains("model llamacpp:local"),
        "the selected model is real: {text}"
    );
    assert!(
        text.contains(&expected_budget),
        "the measured budget is rendered: {text}"
    );
    assert!(
        text.contains("why this model: model llamacpp:local was selected"),
        "the rationale that explains the model is shown: {text}"
    );
    assert!(
        text.contains("checks that really ran: cargo test -p xencode-intelligence-rs"),
        "the check the caller really ran is shown: {text}"
    );
    assert!(
        text.contains("verification passed"),
        "a real pass reads as passed: {text}"
    );
    assert!(
        text.contains("reasoning stage:"),
        "the real plan stage is shown: {text}"
    );
    assert!(
        text.contains("escalation: eligible"),
        "escalation is inspectable: {text}"
    );
    assert!(
        text.contains(NOT_RECORDED_SESSION),
        "unread stores stay honest: {text}"
    );

    let confidence_lines: Vec<&str> = text
        .lines()
        .filter(|line| line.contains("confidence"))
        .collect();
    assert!(
        confidence_lines
            .iter()
            .all(|line| line.contains("never inputs")),
        "the only mention of confidence is the recorded rule refusing it: {confidence_lines:?}"
    );
    assert!(
        !text.contains("confidence:"),
        "no confidence figure is invented: {text}"
    );
    assert!(!text.contains('%'), "no percentage is invented: {text}");
}

#[test]
fn unmeasured_values_render_as_unknown_without_a_pass() {
    let scratch = Scratch::new("unknown");
    let root = scratch.path().join("artifacts");
    let coordinate = Coordinator::new(
        CapabilityView::new(vec![{
            let mut unmeasured = local_model("model:unknown-route");
            unmeasured.egress = Measurement::Unknown;
            unmeasured
        }]),
        &root,
    );
    let input = request("what does this function do");
    let decision = coordinate.decide(&input).expect("a decision with unknowns");
    assert_eq!(decision.selected_model, None);
    assert_eq!(decision.context_budget, Measurement::Unknown);

    coordinate
        .record(
            &input,
            &decision,
            ExecutionRecord {
                status: ExecutionStatus::NotRun,
                summary: "nothing was executed".to_string(),
            },
            not_run_verification(),
        )
        .expect("the not-run outcome must be recorded");

    let records = load_decisions(&root).expect("decisions must load");
    let text = intelligence_report(&records).join("\n");
    assert!(
        text.contains(&format!("context budget: {UNKNOWN}")),
        "{text}"
    );
    assert!(
        text.contains("reasoning stage: not recorded here"),
        "an unset plan is not a stage: {text}"
    );
    assert!(
        text.contains("model none selected"),
        "no selected model is stated, not guessed: {text}"
    );
    assert!(
        text.contains("latency / resource: not recorded here"),
        "latency is not invented: {text}"
    );
    assert!(!text.contains("confidence"), "{text}");
    assert!(!text.contains('%'), "{text}");
}

#[test]
fn not_run_verification_never_reads_as_verified() {
    let scratch = Scratch::new("not-run");
    let root = scratch.path().join("artifacts");
    let coordinator = Coordinator::new(CapabilityView::new(vec![local_model("local:near")]), &root);
    let input = request("explain the reader");
    let decision = coordinator.decide(&input).expect("decision");
    coordinator
        .record(
            &input,
            &decision,
            ExecutionRecord {
                status: ExecutionStatus::NotRun,
                summary: "no execution".to_string(),
            },
            not_run_verification(),
        )
        .expect("record");

    let records = load_decisions(&root).expect("load");
    let text = intelligence_report(&records).join("\n");
    let block = text
        .lines()
        .find(|line| line.starts_with('#'))
        .expect("there is one group line");
    assert!(
        block.contains("verification unknown (not run)"),
        "the group line must not claim a pass: {block}"
    );
    assert!(!block.contains("verification passed"), "{block}");
    assert!(!block.contains("verified"), "{block}");
    assert!(
        !text.contains("checks that really ran: cargo test was never run"),
        "a NotRun outcome's check names are not presented as having run: {text}"
    );
    assert!(
        text.contains("checks that really ran: none recorded (verification was not run)"),
        "{text}"
    );
}

#[test]
fn an_orphaned_decision_renders_with_unknown_verification() {
    let scratch = Scratch::new("orphan");
    let root = scratch.path().join("artifacts");
    let coordinator = Coordinator::new(CapabilityView::new(vec![local_model("local:near")]), &root);
    let input = request("rename the reader and its callers");
    let decision = coordinator.decide(&input).expect("decision");
    let outcome = coordinator
        .record(
            &input,
            &decision,
            ExecutionRecord {
                status: ExecutionStatus::Completed,
                summary: "renamed".to_string(),
            },
            passed_verification(),
        )
        .expect("record");
    let id = outcome.decision_id.clone();

    // Remove only the outcome file. The decision is real and stays readable.
    let outcome_file = coordinator.outcomes_dir().join(format!("{id}.json"));
    std::fs::remove_file(&outcome_file).expect("the outcome file was written and is removed");

    let loaded = load_decisions_with_notes(&root).expect("load");
    assert_eq!(loaded.records.len(), 1, "the decision is still rendered");
    assert!(
        loaded
            .notes
            .iter()
            .any(|note| note.contains(&id) && note.contains("no readable outcome")),
        "the missing outcome is noted: {:?}",
        loaded.notes
    );
    let text = intelligence_report_with_notes(&loaded).join("\n");
    let group = text
        .lines()
        .find(|line| line.starts_with('#'))
        .expect("the orphan still has a group line");
    assert!(group.contains(&format!("#{id}")), "{group}");
    assert!(
        group.contains("verification unknown (not run)"),
        "an orphan has unknown verification: {group}"
    );
    assert!(!group.contains("verification passed"), "{group}");
    assert!(
        text.contains("record-read notes:"),
        "the note is surfaced: {text}"
    );
}

#[test]
fn an_unreadable_decision_is_skipped_and_noted() {
    let scratch = Scratch::new("unreadable");
    let root = scratch.path().join("artifacts");
    let coordinator = Coordinator::new(CapabilityView::new(vec![local_model("local:near")]), &root);
    let input = request("a real prompt that will be recorded");
    let decision = coordinator.decide(&input).expect("decision");
    coordinator
        .record(
            &input,
            &decision,
            ExecutionRecord {
                status: ExecutionStatus::NotRun,
                summary: "nothing ran".to_string(),
            },
            not_run_verification(),
        )
        .expect("record");

    // A corrupt file on the real filesystem is not a decision we can render.
    let corrupt_id = "deadbeefdeadbeef";
    std::fs::create_dir_all(coordinator.decisions_dir()).expect("decisions dir exists");
    std::fs::write(
        coordinator
            .decisions_dir()
            .join(format!("{corrupt_id}.json")),
        "{ this is not json",
    )
    .expect("a corrupt file is written to disk");

    let loaded = load_decisions_with_notes(&root).expect("load");
    assert_eq!(
        loaded.records.len(),
        1,
        "only the readable decision is returned"
    );
    assert!(
        loaded
            .notes
            .iter()
            .any(|note| note.contains(corrupt_id) && note.contains("could not be read")),
        "the unreadable file is noted: {:?}",
        loaded.notes
    );
    let text = intelligence_report_with_notes(&loaded).join("\n");
    let groups: Vec<&str> = text.lines().filter(|line| line.starts_with('#')).collect();
    assert_eq!(groups.len(), 1, "only one real decision is rendered");
    assert!(
        !groups[0].contains(corrupt_id),
        "a skipped decision is not shown as a group: {groups:?}"
    );
    assert!(
        text.lines()
            .any(|line| line.contains(corrupt_id) && line.trim_start().starts_with('-')),
        "the skipped id is named in a note: {text}"
    );
    assert!(text.contains("record-read notes:"), "{text}");
}

#[test]
fn a_session_with_no_decisions_reports_that_honestly() {
    let scratch = Scratch::new("empty");
    let root = scratch.path().join("no-artifacts-yet");
    assert!(
        !root.exists(),
        "there is no artifact root in a normal session"
    );

    let records = load_decisions(&root).expect("an absent decisions dir is an empty load");
    assert!(records.is_empty());
    let text = intelligence_report(&records).join("\n");
    assert!(
        text.contains("INTELLIGENCE INSPECTOR — 0 decisions"),
        "{text}"
    );
    assert!(
        text.contains("No intelligence decisions recorded"),
        "the honest empty message is present: {text}"
    );
    assert!(
        text.contains("opt-in and inert in production paths"),
        "the empty state is explained, not reported as a failure: {text}"
    );
    assert!(!text.contains("confidence"), "{text}");
    assert!(!text.contains('%'), "{text}");
}

#[test]
fn loading_and_rendering_are_deterministic() {
    let scratch = Scratch::new("deterministic");
    let root = scratch.path().join("artifacts");
    let coordinator = Coordinator::new(
        CapabilityView::new(vec![local_model("local:beta"), local_model("local:alpha")]),
        &root,
    );
    for prompt in [
        "second prompt about the reader",
        "first prompt about the writer",
    ] {
        let input = request(prompt);
        let decision = coordinator.decide(&input).expect("decision");
        coordinator
            .record(
                &input,
                &decision,
                ExecutionRecord {
                    status: ExecutionStatus::NotRun,
                    summary: "no execution".to_string(),
                },
                not_run_verification(),
            )
            .expect("record");
    }

    let first = load_decisions_with_notes(&root).expect("first load");
    let second = load_decisions_with_notes(&root).expect("second load");
    assert_eq!(
        first.records, second.records,
        "the filesystem gives the same records"
    );
    assert_eq!(first.notes, second.notes);
    assert_eq!(
        intelligence_report(&first.records),
        intelligence_report(&second.records),
        "the renderer is a pure function of its records"
    );
    let ids: Vec<&str> = first
        .records
        .iter()
        .map(|(_, outcome)| outcome.decision_id.as_str())
        .collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "decisions are enumerated in stable id order");
}
