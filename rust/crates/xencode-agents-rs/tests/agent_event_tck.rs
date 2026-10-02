//! `AR-10` — the Agent Event TCK.
//!
//! A compatibility kit for `AR-9`'s protocol. `protocol_streams.rs` asks whether
//! each captured stream still reaches the model and behaves (it finished, it said
//! the answer, its tool call carried an id). This file asks a different question:
//! *which shapes* each real stream produces. That is the regression firewall for
//! the adapter layer — if a normalisation edit makes `codex` stop emitting a
//! `tool_started`, or teaches `opencode` to invent a shape it never had, a table
//! here fails on the next run rather than a user discovering it live.
//!
//! The lines replayed are the vendors' own output, committed under `fixtures/`.
//! Nothing here is fed a line that no agent actually printed: a TCK greenlit on
//! fabricated input certifies nothing. The two variants no captured stream on
//! this box produced — `permission_requested` and `error` — are declared as
//! evidence gaps below and asserted *absent*, never tested against invented
//! input. `file_changed` is absent for a different reason: the protocol never
//! derives it from a worker's stream (see `protocol.rs`), so it is not a gap in
//! the evidence but a rule of the model, and it has its own unit test there.

use std::collections::BTreeMap;

use xencode_agents_rs::protocol::{normalise_line, run_completed, AgentEvent};

/// The eight streams captured on 2026-10-02, the ones this kit replays.
const CAPTURED: &[&str] = &[
    "opencode",
    "kilo",
    "cline",
    "codex",
    "agy",
    "cursor-agent",
    "kiro-cli",
    "claude",
];

/// Every variant the model defines, by its `name()`.
const ALL_VARIANTS: &[&str] = &[
    "session_started",
    "message",
    "tool_requested",
    "tool_started",
    "tool_output",
    "file_changed",
    "permission_requested",
    "error",
    "completed",
    "session_ended",
];

fn events_for(agent: &str) -> Vec<AgentEvent> {
    let raw = match agent {
        "opencode" => include_str!("fixtures/opencode.ndjson"),
        "kilo" => include_str!("fixtures/kilo.ndjson"),
        "cline" => include_str!("fixtures/cline.ndjson"),
        "codex" => include_str!("fixtures/codex.ndjson"),
        "agy" => include_str!("fixtures/agy.ndjson"),
        "cursor-agent" => include_str!("fixtures/cursor-agent.ndjson"),
        "kiro-cli" => include_str!("fixtures/kiro-cli.ndjson"),
        "claude" => include_str!("fixtures/claude.ndjson"),
        other => panic!("no captured stream for {other}"),
    };
    raw.lines()
        .flat_map(|line| normalise_line(agent, line))
        .collect()
}

/// The distinct variant names one stream produced, sorted.
fn shapes(agent: &str) -> Vec<&'static str> {
    let mut seen: BTreeMap<&str, ()> = BTreeMap::new();
    for event in events_for(agent) {
        seen.insert(event.name(), ());
    }
    seen.keys().copied().collect()
}

/// What each captured stream actually reaches, committed as the expected shape
/// inventory. A real agent's output, recorded verbatim: the `name()`s that came
/// out of `normalise_line` on 2026-10-02, and no others. This is the whole kit in
/// one table — edit the normaliser so a stream gains or loses a shape, and the
/// matching row fails.
const COVERAGE: &[(&str, &[&str])] = &[
    (
        "opencode",
        &["completed", "message", "session_started", "tool_output"],
    ),
    (
        "kilo",
        &["completed", "message", "session_started", "tool_output"],
    ),
    (
        "cline",
        &[
            "completed",
            "message",
            "session_ended",
            "session_started",
            "tool_output",
            "tool_requested",
        ],
    ),
    (
        "codex",
        &[
            "completed",
            "message",
            "session_started",
            "tool_output",
            "tool_started",
        ],
    ),
    (
        "agy",
        &[
            "completed",
            "message",
            "session_started",
            "tool_output",
            "tool_started",
        ],
    ),
    (
        "cursor-agent",
        &[
            "completed",
            "message",
            "session_ended",
            "session_started",
            "tool_output",
            "tool_requested",
        ],
    ),
    (
        "kiro-cli",
        &[
            "completed",
            "message",
            "session_ended",
            "session_started",
            "tool_output",
            "tool_requested",
        ],
    ),
    ("claude", &["completed", "message", "session_started"]),
];

#[test]
fn each_captured_stream_reaches_exactly_the_committed_shapes() {
    for (agent, expected) in COVERAGE {
        assert_eq!(
            shapes(agent),
            *expected,
            "{agent}'s shape inventory moved, so either the normaliser changed \
             or this capture no longer represents it"
        );
    }
}

#[test]
fn every_captured_stream_reports_a_session_and_terminates() {
    // The finish semantics, replayed per vendor: each real stream opens a session
    // and reaches a variant `run_completed` accepts. An agent that never said it
    // finished would leave an orchestrator waiting forever, so this is checked for
    // all eight, not one.
    for agent in CAPTURED {
        let events = events_for(agent);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::SessionStarted { .. })),
            "{agent} opened no session"
        );
        assert!(run_completed(&events), "{agent} never reported finishing");
    }
}

#[test]
fn the_real_streams_together_cover_seven_of_the_ten_variants() {
    // The honest coverage number. Seven of the model's ten shapes came out of real
    // vendor lines; the table below names exactly which. Recording the count (not
    // a vague "most") is what makes the three uncovered variants a stated fact
    // rather than an oversight.
    let mut reached: BTreeMap<&str, ()> = BTreeMap::new();
    for agent in CAPTURED {
        for name in shapes(agent) {
            reached.insert(name, ());
        }
    }
    let covered: Vec<&str> = reached.keys().copied().collect();
    assert_eq!(
        covered,
        vec![
            "completed",
            "message",
            "session_ended",
            "session_started",
            "tool_output",
            "tool_requested",
            "tool_started",
        ],
        "the set of shapes the real streams produce changed"
    );
    assert_eq!(covered.len(), 7);
}

#[test]
fn the_two_unobserved_variants_are_gaps_and_not_fabricated_input() {
    // `permission_requested` and `error` are the model's two shapes no captured
    // stream on this box produced. This asserts that ABSENCE, so the gap is a
    // fact someone re-checks every run, and so nobody quietly adds a made-up line
    // to a fixture and calls the variant covered — a TCK greenlit on fabricated
    // lines certifies nothing. When a claude re-run or a ninth vendor emits one,
    // this test fails, which is the signal to replace the gap with a real case.
    //
    // `file_changed` is the third unobserved variant but is NOT an evidence gap:
    // `normalise_line` never derives it from a worker's stream by design (it comes
    // from xencode's own diff of the lease), so it can only ever be absent here.
    for agent in CAPTURED {
        for event in events_for(agent) {
            assert!(
                !matches!(
                    event,
                    AgentEvent::PermissionRequested { .. } | AgentEvent::Error { .. }
                ),
                "{agent} produced {:?}, so a variant the kit records as an \
                 evidence gap is now covered by a real line — replace this \
                 assertion with a positive case built on that capture",
                event
            );
        }
    }
}

#[test]
fn no_real_line_is_ever_read_as_an_invention() {
    // Origin discipline across the whole kit, not one chosen line: every event any
    // captured stream yields is `Observed`. If a normalisation edit starts
    // synthesising anything from a bare vendor line, this fails for the vendor that
    // regressed, naming it.
    for agent in CAPTURED {
        let synthesised: Vec<&str> = events_for(agent)
            .iter()
            .filter(|e| e.origin() != xencode_agents_rs::protocol::Origin::Observed)
            .map(|e| e.name())
            .collect();
        assert!(
            synthesised.is_empty(),
            "{agent} yielded events xencode invented rather than read: {synthesised:?}"
        );
    }
}

#[test]
fn the_variant_names_are_the_ten_the_model_defines() {
    // The kit keys on `AgentEvent::name()`. If a variant is added or renamed in
    // `protocol.rs`, this table no longer matches it and the whole kit is stale —
    // so this check ties the kit's vocabulary to the model's own, rather than to a
    // hand-typed list that silently drifts.
    let from_model: BTreeMap<&str, ()> = [
        AgentEvent::SessionStarted {
            session_id: None,
            origin: Default::default(),
        },
        AgentEvent::Message {
            text: String::new(),
            origin: Default::default(),
        },
        AgentEvent::ToolRequested {
            tool: String::new(),
            call_id: None,
            origin: Default::default(),
        },
        AgentEvent::ToolStarted {
            tool: String::new(),
            call_id: None,
            origin: Default::default(),
        },
        AgentEvent::ToolOutput {
            tool: String::new(),
            call_id: None,
            output: None,
            origin: Default::default(),
        },
        AgentEvent::FileChanged {
            path: String::new(),
            origin: Default::default(),
        },
        AgentEvent::PermissionRequested {
            tool: String::new(),
            call_id: None,
            origin: Default::default(),
        },
        AgentEvent::Error {
            message: String::new(),
            origin: Default::default(),
        },
        AgentEvent::Completed {
            outcome: None,
            origin: Default::default(),
        },
        AgentEvent::SessionEnded {
            reason: None,
            origin: Default::default(),
        },
    ]
    .iter()
    .map(|e| (e.name(), ()))
    .collect();

    let declared: BTreeMap<&str, ()> = ALL_VARIANTS.iter().map(|n| (*n, ())).collect();
    assert_eq!(
        from_model.keys().collect::<Vec<_>>(),
        declared.keys().collect::<Vec<_>>(),
        "the kit's variant list and the model's variants disagree"
    );
}
