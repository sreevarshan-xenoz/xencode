//! Every real stream the probe captured on 2026-10-02, run through the common
//! event model.
//!
//! The fixtures under `fixtures/` are the agents' own output, verbatim, with
//! long lines shortened. They exist because the unit tests in `protocol.rs`
//! pick one line per agent, and picking one line is exactly how a reader gets
//! built that works on the line you chose. These run whole streams: every event
//! name, every field, every odd spelling the vendor happened to use that day.

use std::collections::BTreeMap;

use xencode_agents_rs::protocol::{normalise_line, run_completed, AgentEvent, Origin};

fn stream(agent: &str) -> Vec<AgentEvent> {
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

fn counts(events: &[AgentEvent]) -> BTreeMap<&'static str, usize> {
    let mut out = BTreeMap::new();
    for event in events {
        *out.entry(event.name()).or_insert(0) += 1;
    }
    out
}

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

#[test]
fn every_captured_stream_reaches_the_model_and_says_it_finished() {
    for agent in CAPTURED {
        let events = stream(agent);
        assert!(
            !events.is_empty(),
            "{agent}'s captured stream produced nothing"
        );
        assert!(
            run_completed(&events),
            "{agent}'s captured stream never reported finishing, so an orchestrator \
             would wait on it forever"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::SessionStarted { .. })),
            "{agent} never announced a session"
        );
    }
}

#[test]
fn no_captured_stream_contains_an_uncorroborated_invention() {
    // Every event here came out of a real vendor line. If this test ever fails,
    // something is being added that no agent said.
    for agent in CAPTURED {
        let invented: Vec<&str> = stream(agent)
            .iter()
            .filter(|e| e.origin() != Origin::Observed)
            .map(|e| e.name())
            .collect();
        assert!(
            invented.is_empty(),
            "{agent} produced events xencode invented: {invented:?}"
        );
    }
}

#[test]
fn every_agent_that_ran_a_tool_reported_one() {
    // The probe task asks an agent to read one file, so each stream that
    // completed should show the tool it used. An agent that answered without a
    // tool call is a finding, not a pass — so this is asserted per agent that
    // the run was real, and the counts are printed for whoever reads a failure.
    for agent in [
        "opencode",
        "kilo",
        "cline",
        "codex",
        "agy",
        "cursor-agent",
        "kiro-cli",
    ] {
        let events = stream(agent);
        let tools = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    AgentEvent::ToolRequested { .. }
                        | AgentEvent::ToolStarted { .. }
                        | AgentEvent::ToolOutput { .. }
                )
            })
            .count();
        assert!(
            tools > 0,
            "{agent} reported finishing without ever reporting a tool call: {:?}",
            counts(&events)
        );
    }
}

#[test]
fn the_answer_reached_the_model_from_every_agent_that_said_it() {
    // The fixture task's answer is the word `xencode`. An agent that solved the
    // task and said nothing is one xencode cannot show a user.
    for agent in [
        "opencode",
        "kilo",
        "cline",
        "codex",
        "agy",
        "cursor-agent",
        "kiro-cli",
    ] {
        // Joined, not matched per event: kiro-cli streams its answer as `xen`
        // then `code`, so the word exists in the run and not in any one line.
        // Reading one event at a time would call a correct agent silent.
        let said: String = stream(agent)
            .iter()
            .filter_map(|e| match e {
                AgentEvent::Message { text, .. } => Some(text.as_str()),
                AgentEvent::ToolOutput {
                    output: Some(o), ..
                } => Some(o.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            said.contains("xencode"),
            "{agent} finished the task without reporting the word in it: {said:?}"
        );
    }
}

#[test]
fn the_tool_call_carries_its_own_identifier() {
    // `codex` was the only agent whose stream carried a correlation id, and
    // matching an output back to its request needs one. Every agent that
    // announced a tool call also carried an id, which the model records rather
    // than inventing.
    for agent in ["opencode", "cline", "codex", "cursor-agent", "kiro-cli"] {
        let events = stream(agent);
        let ids: Vec<&Option<String>> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolRequested { call_id, .. }
                | AgentEvent::ToolStarted { call_id, .. }
                | AgentEvent::ToolOutput { call_id, .. } => Some(call_id),
                _ => None,
            })
            .collect();
        assert!(!ids.is_empty(), "{agent} made no tool call to check");
        assert!(
            ids.iter().all(|id| id.is_some()),
            "{agent} announced a tool call with no id to match its output against: {ids:?}"
        );
    }
}

#[test]
fn agy_is_the_one_agent_whose_tool_calls_cannot_be_paired_up() {
    // Measured 2026-10-02: agy's tool steps carry `tool_name` and nothing else,
    // so a consumer cannot tell which of several concurrent tool calls an output
    // belongs to. Recorded rather than papered over, because it is a real limit
    // on routing concurrent work to agy.
    let events = stream("agy");
    let ids: Vec<&Option<String>> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolStarted { call_id, .. } | AgentEvent::ToolOutput { call_id, .. } => {
                Some(call_id)
            }
            _ => None,
        })
        .collect();
    assert!(!ids.is_empty(), "agy made no tool call to check");
    assert!(
        ids.iter().all(|id| id.is_none()),
        "agy now sends correlation ids, so this limit is gone and the test should be rewritten: {ids:?}"
    );
}
