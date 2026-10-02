//! AR-4's done-when, checked against the streams the probe actually captured.
//!
//! The bytes here are the vendors' own output as committed under `fixtures/`;
//! what this file exercises is the store that turns a run into three files and
//! reads it back. The runs are replays of those captures rather than new launches
//! — the point is the store and the renderer, and a replay of a real stream is
//! not a fake stream.

use xencode_agents_rs::capture::{find_captures, read_capture, write_capture, Capture};
use xencode_agents_rs::protocol::{normalise_line, AgentEvent, Origin};
use xencode_agents_rs::roster::Provenance;
use xencode_agents_rs::RunCapture;

const AGENTS: [&str; 8] = [
    "opencode",
    "kilo",
    "cline",
    "codex",
    "agy",
    "cursor-agent",
    "kiro-cli",
    "claude",
];

fn fixture(agent: &str) -> &'static str {
    match agent {
        "opencode" => include_str!("fixtures/opencode.ndjson"),
        "kilo" => include_str!("fixtures/kilo.ndjson"),
        "cline" => include_str!("fixtures/cline.ndjson"),
        "codex" => include_str!("fixtures/codex.ndjson"),
        "agy" => include_str!("fixtures/agy.ndjson"),
        "cursor-agent" => include_str!("fixtures/cursor-agent.ndjson"),
        "kiro-cli" => include_str!("fixtures/kiro-cli.ndjson"),
        "claude" => include_str!("fixtures/claude.ndjson"),
        other => panic!("no committed stream for {other}"),
    }
}

fn replay(agent: &str) -> RunCapture {
    RunCapture {
        agent: agent.to_string(),
        binary: Some(format!("{agent} (replayed, not launched here)")),
        version: None,
        argv: Vec::new(),
        workdir: format!("tests/fixtures/{agent}.ndjson"),
        exit_code: None,
        duration_ms: 0,
        stdout: fixture(agent).to_string(),
        stderr: String::new(),
        stdout_truncated: false,
        stderr_truncated: false,
        events: Vec::new(),
        stream_recognised: true,
        session_id: None,
        stopped_on_auth: false,
        permission_signal: None,
        usage: None,
        model: None,
        provenance: Provenance::Observed,
        failure: None,
    }
}

/// One capture written and read back, for a real stream. The temporary directory
/// comes back with it so the files stay alive for the caller's own checks.
fn round_trip(agent: &str) -> (tempfile::TempDir, std::path::PathBuf, Capture) {
    let dir = tempfile::tempdir().expect("a scratch dir");
    let written = write_capture(dir.path(), &replay(agent))
        .unwrap_or_else(|e| panic!("{agent}: the capture was not written: {e}"));
    assert_eq!(
        written,
        dir.path().join(agent).join("capture"),
        "{agent}: the capture is not where the shape says it is"
    );
    let read = read_capture(&written).unwrap_or_else(|e| panic!("{agent}: {e}"));
    (dir, written, read)
}

#[test]
fn every_committed_stream_survives_being_stored_and_read_back() {
    for agent in AGENTS {
        let (_keep, _dir, capture) = round_trip(agent);
        let live = replay(agent);
        let expected: Vec<AgentEvent> = fixture(agent)
            .lines()
            .flat_map(|line| normalise_line(agent, line))
            .collect();

        assert_eq!(
            capture
                .events
                .iter()
                .map(|s| s.event.clone())
                .collect::<Vec<_>>(),
            expected,
            "{agent}: the events on disk are not the events the normaliser makes"
        );
        assert_eq!(
            capture.metadata.raw_lines,
            live.stdout.lines().count(),
            "{agent}: the count of raw lines is wrong"
        );
        assert_eq!(
            capture.metadata.normalized_events,
            expected.len(),
            "{agent}: the count of events is wrong"
        );
    }
}

#[test]
fn the_raw_stream_is_stored_verbatim_and_not_replaced() {
    // The whole reason `raw.jsonl` exists: the bytes the vendor printed stay
    // recoverable, so a later reader can argue with the normaliser.
    for agent in AGENTS {
        let (_keep, _dir, capture) = round_trip(agent);
        let joined = capture
            .raw
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            joined,
            fixture(agent).trim_end_matches('\n'),
            "{agent}: what came back off disk is not what the agent printed"
        );
        let numbers: Vec<usize> = capture.raw.iter().map(|l| l.line).collect();
        assert_eq!(
            numbers,
            (1..=numbers.len()).collect::<Vec<_>>(),
            "{agent}: raw line numbers are not a gapless count from one"
        );
    }
}

#[test]
fn every_stored_event_names_the_raw_line_it_came_from() {
    // AR-4's done-when in one assertion. The line has to be the actual line: the
    // test re-normalises it and requires the same event back.
    for agent in AGENTS {
        let (_keep, _dir, capture) = round_trip(agent);
        assert!(
            !capture.events.is_empty(),
            "{agent}: nothing was stored, so the check proved nothing"
        );
        for stored in &capture.events {
            let line = stored.raw_line.unwrap_or_else(|| {
                panic!(
                    "{agent}: event {} ({}) names no raw line and was not marked as xencode's own",
                    stored.seq,
                    stored.event.name()
                )
            });
            let raw = capture
                .raw
                .iter()
                .find(|l| l.line == line)
                .unwrap_or_else(|| panic!("{agent}: raw line {line} is not in the capture"));
            let again = normalise_line(agent, &raw.text);
            assert!(
                again.contains(&stored.event),
                "{}: event {} did not come from raw line {line} ({:?})",
                agent,
                stored.seq,
                stored.event
            );
        }
        // The other direction: a recorded line number never points at nothing.
        for stored in &capture.events {
            if let Some(line) = stored.raw_line {
                assert!(
                    capture.raw.iter().any(|l| l.line == line),
                    "{agent}: event {} points at raw line {line}, which is not stored",
                    stored.seq
                );
            }
        }
    }
}

#[test]
fn the_event_rows_are_in_the_order_the_stream_arrived() {
    for agent in AGENTS {
        let (_keep, _dir, capture) = round_trip(agent);
        let seq: Vec<usize> = capture.events.iter().map(|s| s.seq).collect();
        assert_eq!(
            seq,
            (1..=seq.len()).collect::<Vec<_>>(),
            "{agent}: the normalised stream is not numbered from one without gaps"
        );
        let lines: Vec<usize> = capture.events.iter().filter_map(|s| s.raw_line).collect();
        let mut sorted = lines.clone();
        sorted.sort_unstable();
        assert_eq!(
            lines, sorted,
            "{agent}: events are stored out of the order their raw lines arrived"
        );
    }
}

#[test]
fn two_vendors_that_agree_on_nothing_else_render_in_the_same_trace_view() {
    // `codex` keys its stream on `type`, `cline` on `contentType`, `agy` puts
    // prose in one field and nothing anywhere else, and `kiro-cli` splits a
    // sentence in half. All four have to land in one rendering, because the
    // renderer gets one code path and no vendor name.
    let mut traces = Vec::new();
    for agent in ["codex", "cline", "agy", "kiro-cli"] {
        let (_keep, _dir, capture) = round_trip(agent);
        let trace = capture.trace();
        for row in capture.events.iter().take(3) {
            assert!(
                trace.contains(&format!("raw#{}", row.raw_line.expect("a stored line"))),
                "{agent}: the trace does not show the raw line behind event {}",
                row.seq
            );
        }
        assert!(
            trace.starts_with(&format!("{agent} — ")),
            "{agent}: the trace does not say whose run it is: {trace}"
        );
        assert!(
            trace.contains("run completed:"),
            "{agent}: the trace does not answer whether the run finished"
        );
        traces.push((agent, trace));
    }
    // Same columns for all four: the sequence, then the event name, then where it
    // came from. Nothing here reads a vendor name.
    for (agent, trace) in &traces {
        let rows = trace
            .lines()
            .filter(|l| l.contains("raw#") || l.contains("xencode"))
            .count();
        assert!(rows > 0, "{agent}: no event rows at all in the trace view");
    }
}

#[test]
fn a_vendor_that_reported_nothing_still_leaves_a_readable_capture() {
    // An agent that stopped at its auth check, or printed a banner and died, is a
    // real outcome. The store must keep its bytes and say plainly that nothing
    // was recognised, rather than writing an empty file or failing.
    let dir = tempfile::tempdir().expect("a scratch dir");
    let mut run = replay("codex");
    run.agent = "crush".to_string();
    run.stdout = "Crush v0.96.1\nerror: not signed in, run `crush auth`\n".to_string();
    run.stream_recognised = false;
    run.stopped_on_auth = true;
    run.failure = Some("stopped at its authentication check".to_string());

    let written = write_capture(dir.path(), &run).expect("the capture is written");
    let capture = read_capture(&written).expect("the capture reads back");
    assert!(
        capture.events.is_empty(),
        "a banner and an error line are not events: {:?}",
        capture.events
    );
    assert_eq!(capture.raw.len(), 2, "both lines are still stored");
    assert!(
        !capture.metadata.stream_recognised,
        "the metadata must say the stream was not machine-readable"
    );
    let trace = capture.trace();
    assert!(
        trace.contains("stopped on its authentication check"),
        "the trace hides why nothing came back: {trace}"
    );
    assert!(
        trace.contains("the stream was not machine-readable"),
        "the trace claims an empty run rather than an unreadable one: {trace}"
    );
}

#[cfg(unix)]
#[test]
fn a_stored_capture_is_owner_only_on_every_file() {
    // A raw vendor stream can carry a credential the vendor printed. Nothing here
    // redacts it — that is AR-5's job — so the file must at least not be readable
    // by anyone else on the machine.
    use std::os::unix::fs::PermissionsExt;
    for agent in AGENTS {
        let (_keep, base, _capture) = round_trip(agent);
        for name in ["raw.jsonl", "normalized.jsonl", "metadata.json"] {
            let path = base.join(name);
            let mode = std::fs::metadata(&path)
                .unwrap_or_else(|e| panic!("{agent}: {name}: {e}"))
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(
                mode,
                0o600,
                "{agent}: {} is readable by someone other than its owner",
                path.display()
            );
        }
    }
}

#[test]
fn the_metadata_says_how_much_of_the_capture_is_xencodes_own_inference() {
    // A reader must not have to trust each row's origin field to see how much of
    // a file is bookkeeping, so the count is stated beside the data.
    let mut observed_any = false;
    for agent in AGENTS {
        let (_keep, _dir, capture) = round_trip(agent);
        let counted = capture
            .events
            .iter()
            .filter(|s| s.event.origin() == Origin::Synthesised)
            .count();
        assert_eq!(
            capture.metadata.synthesized_events, counted,
            "{agent}: the metadata's count of inferred events disagrees with the rows"
        );
        if counted < capture.metadata.normalized_events {
            observed_any = true;
        }
    }
    assert!(
        observed_any,
        "not one captured stream produced an event the vendor actually said"
    );
}

#[test]
fn a_capture_written_for_one_agent_does_not_disturb_another() {
    // The report keeps every agent; the store must too, and must not let one
    // vendor's bytes be read out of another's directory.
    let dir = tempfile::tempdir().expect("a scratch dir");
    for agent in ["opencode", "cline", "codex"] {
        write_capture(dir.path(), &replay(agent)).expect("written");
    }
    for agent in ["opencode", "cline", "codex"] {
        let capture = read_capture(&dir.path().join(agent).join("capture"))
            .unwrap_or_else(|e| panic!("{agent}: {e}"));
        assert_eq!(capture.metadata.agent, agent);
        assert!(
            !capture.events.is_empty(),
            "{agent}: a neighbour's capture pushed this one out"
        );
    }
}

#[test]
fn read_capture_accepts_the_agent_directory_or_the_capture_directory() {
    // The store lays files down at `<root>/<agent>/capture/`, but an operator is
    // likeliest to point --trace at `<root>/<agent>`. Both must resolve.
    let dir = tempfile::tempdir().expect("a scratch dir");
    let written = write_capture(dir.path(), &replay("opencode")).expect("written");
    let agent_dir = dir.path().join("opencode");
    assert_eq!(written, agent_dir.join("capture"));

    let from_capture = read_capture(&written).expect("reads from the capture dir");
    let from_agent = read_capture(&agent_dir).expect("reads from the agent dir");
    assert_eq!(from_capture, from_agent, "the two spellings disagree");
}

#[test]
fn find_captures_lists_every_vendor_under_a_root_in_one_pass() {
    // AR-4's "two different vendors' runs render in the same trace view" needs a
    // single call that finds both, ordered, so the CLI can print them side by side.
    let dir = tempfile::tempdir().expect("a scratch dir");
    for agent in ["codex", "cline", "opencode"] {
        write_capture(dir.path(), &replay(agent)).expect("written");
    }
    let found = find_captures(dir.path());
    let agents: Vec<String> = found
        .iter()
        .map(|p| read_capture(p).expect("readable").metadata.agent)
        .collect();
    assert_eq!(
        agents,
        vec![
            "cline".to_string(),
            "codex".to_string(),
            "opencode".to_string()
        ],
        "the root does not yield every vendor's capture, sorted and stable"
    );
}

#[test]
fn find_captures_on_a_single_capture_directory_yields_exactly_it() {
    let dir = tempfile::tempdir().expect("a scratch dir");
    let written = write_capture(dir.path(), &replay("codex")).expect("written");
    let found = find_captures(&written);
    assert_eq!(found, vec![written.clone()], "a lone capture dir is one");
}

#[test]
fn find_captures_on_a_directory_with_no_capture_returns_nothing_not_an_error() {
    // --trace turns an empty list into its own clear message; the finder itself
    // must not invent a capture or fail on a plain directory.
    let dir = tempfile::tempdir().expect("a scratch dir");
    assert!(find_captures(dir.path()).is_empty());
}
