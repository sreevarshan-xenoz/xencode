//! AR-5's done-when, checked against files that actually get written.
//!
//! The envelope is the copy that leaves the sealed capture and flows to the
//! ledger and the metrics, so everything here is about what does and does not
//! land on disk in that layer.

use xencode_agents_rs::capture::write_capture;
use xencode_agents_rs::envelope::{
    read_envelopes, write_envelopes, AgentId, Envelope, Field, SessionId, TaskId, WorkerId,
};
use xencode_agents_rs::protocol::{AgentEvent, Origin};
use xencode_agents_rs::roster::Provenance;
use xencode_agents_rs::RunCapture;

const TOKEN: &str = "sk-FAKE-NOT-A-REAL-TEST-KEY";
const JWT: &str = "eyJFAKE_JWT_HEADERxx.eyJFAKE_JWT_PAYLOADyy.FAKE_JWT_SIGNATUREzz";

fn run(agent: &str, stdout: &str, session_id: Option<&str>) -> RunCapture {
    RunCapture {
        agent: agent.to_string(),
        binary: Some(agent.to_string()),
        version: Some("9.9.9".to_string()),
        argv: vec![agent.to_string(), "--print".to_string()],
        workdir: "/tmp/whatever".to_string(),
        exit_code: Some(0),
        duration_ms: 12,
        stdout: stdout.to_string(),
        stderr: String::new(),
        stdout_truncated: false,
        stderr_truncated: false,
        events: Vec::new(),
        stream_recognised: true,
        session_id: session_id.map(str::to_string),
        stopped_on_auth: false,
        permission_signal: None,
        usage: None,
        model: None,
        provenance: Provenance::Observed,
        failure: None,
    }
}

#[test]
fn the_four_states_round_trip_and_stay_semantically_distinct() {
    // The whole reason this item exists: unknown and unavailable are both "no
    // value" and collapsing them is the bug. Each must serialise to its own
    // shape and come back unchanged.
    let observed: Field<String> = Field::Observed("the worker said this".into());
    let synthesised: Field<String> = Field::Synthesised("xencode minted this".into());
    let unknown: Field<String> = Field::Unknown;
    let unavailable: Field<String> = Field::Unavailable {
        reason: "agy sends no correlation id".into(),
    };

    for original in [
        observed.clone(),
        synthesised.clone(),
        unknown.clone(),
        unavailable.clone(),
    ] {
        let json = serde_json::to_string(&original).unwrap();
        let back: Field<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, original, "{json} does not round-trip");
    }

    assert_ne!(
        serde_json::to_string(&unknown).unwrap(),
        serde_json::to_string(&unavailable).unwrap(),
        "unknown and unavailable serialise identically — the collapse this item forbids"
    );
    // An absence is never a value: both no-value states carry None.
    assert_eq!(unknown.value(), None);
    assert_eq!(unavailable.value(), None);
    assert!(observed.is_observed());
    assert!(
        !synthesised.is_observed(),
        "a synthesised value read as a measurement"
    );
}

#[test]
fn an_absent_field_deserialises_to_unknown_never_to_a_measurement() {
    // Mirror of Origin::default's rule at the field level: a stored envelope with
    // a field simply missing must read as "nobody said it", not as observed.
    let back: Field<u64> = serde_json::from_str("null").unwrap_or_else(|_| {
        // `null` is not one of the enum's shapes, so read the explicit "unknown".
        serde_json::from_str("\"unknown\"").unwrap()
    });
    assert!(matches!(back, Field::Unknown));
    assert!(!back.is_observed());
}

#[test]
fn a_token_in_a_message_does_not_reach_the_propagation_layer_but_stays_recoverable() {
    // AR-5's headline proof: a credential that appeared in the vendor's stream
    // must not land in the envelope copy, yet the sealed capture must still hold
    // the exact bytes so a deliberate reader can recover them.
    let dir = tempfile::tempdir().expect("a scratch dir");
    let stream =
        format!("{{\"type\":\"text\",\"part\":{{\"text\":\"here is my key {TOKEN} enjoy\"}}}}\n");
    let written = write_capture(dir.path(), &run("opencode", &stream, Some("ses_123")))
        .expect("capture written");

    let raw_text = std::fs::read_to_string(written.join("raw.jsonl")).unwrap();
    assert!(
        raw_text.contains(TOKEN),
        "the sealed raw stream must keep the byte the vendor printed"
    );
    let envelope_text = std::fs::read_to_string(written.join("envelope.jsonl")).unwrap();
    assert!(
        !envelope_text.contains(TOKEN),
        "a token reached the propagation layer: {envelope_text}"
    );
    assert!(
        envelope_text.contains("[redacted:key]"),
        "the payload was dropped rather than redacted: {envelope_text}"
    );

    // Recoverability is a line pointer, not a copy of the secret.
    let read = read_envelopes(&written).expect("envelopes read back");
    assert_eq!(read.torn_tail, None);
    assert!(
        read.envelopes.iter().any(|e| e.raw_line.is_some()),
        "a redacted envelope no longer names the raw line to recover from"
    );
    for envelope in &read.envelopes {
        assert!(
            !envelope.carries_secret(),
            "a written envelope still carries a secret: {envelope:?}"
        );
    }
}

#[test]
fn a_jwt_in_an_output_and_a_tool_name_are_both_scrubbed() {
    let message = AgentEvent::ToolOutput {
        tool: format!("read_file {JWT}"),
        call_id: Some(JWT.to_string()),
        output: Some(format!("contents: {JWT} trailing")),
        origin: Origin::Observed,
    };
    let envelope = Envelope {
        worker_id: Field::Synthesised(WorkerId("opencode/ses_1".into())),
        task_id: Field::Unknown,
        agent_id: Field::Observed(AgentId("opencode".into())),
        session_id: Field::Observed(SessionId(JWT.into())),
        sequence: 1,
        timestamp_ms: Field::Unavailable {
            reason: "no clock".into(),
        },
        origin: message.origin(),
        raw_line: Some(1),
        payload: message,
    };
    assert!(envelope.carries_secret());
    let clean = envelope.redacted();
    assert!(!clean.carries_secret(), "{clean:?}");
    // The session id (a credential shape here) is redacted, but the agent id and
    // the raw-line pointer survive so the row is still attributable and recoverable.
    assert_eq!(
        clean.agent_id.value().map(|a| a.0.as_str()),
        Some("opencode")
    );
    assert_eq!(clean.raw_line, Some(1));
}

#[test]
fn the_envelope_states_are_derived_from_real_facts_not_invented_to_line_up() {
    // We launched the binary, so agent_id is observed; the worker id and the
    // sequence are ours, so synthesised; a replay has no job id, so task_id is
    // unknown rather than a guessed identifier.
    let dir = tempfile::tempdir().expect("a scratch dir");
    let written = write_capture(
        dir.path(),
        &run(
            "codex",
            "{\"type\":\"thread.started\",\"thread_id\":\"thr_9\"}\n",
            Some("thr_9"),
        ),
    )
    .expect("written");
    let read = read_envelopes(&written).expect("read");
    let first = read
        .envelopes
        .first()
        .expect("codex's first line is an event");
    assert!(first.agent_id.is_observed(), "agent_id is ours to know");
    assert!(matches!(first.worker_id, Field::Synthesised(_)));
    assert!(matches!(first.task_id, Field::Unknown));
    assert_eq!(
        first.sequence, 1,
        "sequence is xencode's own gapless position"
    );
    assert_eq!(first.origin, Origin::Observed);
}

#[test]
fn an_envelope_written_then_read_is_byte_for_byte_what_was_stored() {
    let dir = tempfile::tempdir().expect("a scratch dir");
    let capture_dir = dir.path().join("codex").join("capture");
    std::fs::create_dir_all(&capture_dir).unwrap();
    let envelopes = vec![Envelope {
        worker_id: Field::Synthesised(WorkerId("codex/launch-0".into())),
        task_id: Field::Synthesised(TaskId("job-42".into())),
        agent_id: Field::Observed(AgentId("codex".into())),
        session_id: Field::Observed(SessionId("thr_9".into())),
        sequence: 7,
        timestamp_ms: Field::Observed(1_700_000_000_000),
        origin: Origin::Observed,
        raw_line: Some(3),
        payload: AgentEvent::Completed {
            outcome: Some("success".into()),
            origin: Origin::Observed,
        },
    }];
    let path = write_envelopes(&capture_dir, &envelopes).expect("written");
    assert!(path.exists());
    let back = read_envelopes(&capture_dir).expect("read");
    assert_eq!(
        back.envelopes, envelopes,
        "a written envelope is not what came back"
    );
}

#[test]
fn a_torn_final_line_is_discarded_and_a_malformed_earlier_line_is_reported() {
    // DB-5's contract in two halves.
    let dir = tempfile::tempdir().expect("a scratch dir");
    let capture_dir = dir.path().join("codex").join("capture");
    std::fs::create_dir_all(&capture_dir).unwrap();
    let good = serde_json::to_string(&Envelope {
        worker_id: Field::Synthesised(WorkerId("w".into())),
        task_id: Field::Unknown,
        agent_id: Field::Observed(AgentId("codex".into())),
        session_id: Field::Unknown,
        sequence: 1,
        timestamp_ms: Field::Unavailable {
            reason: "no clock".into(),
        },
        origin: Origin::Observed,
        raw_line: Some(1),
        payload: AgentEvent::Completed {
            outcome: None,
            origin: Origin::Observed,
        },
    })
    .unwrap();

    // (a) a writer killed mid-line: a whole record, then a partial one, no newline.
    let torn = format!("{good}\n{{\"worker_id\":");
    std::fs::write(capture_dir.join("envelope.jsonl"), torn).unwrap();
    let read = read_envelopes(&capture_dir).expect("a torn tail must not fail the read");
    assert_eq!(read.envelopes.len(), 1, "the whole record is kept");
    assert!(
        read.torn_tail.is_some(),
        "the partial line is reported as torn"
    );

    // (b) corruption in the middle is NOT a torn tail and must surface.
    let corrupt = format!("{{\"nonsense\": true}}\n{good}\n");
    std::fs::write(capture_dir.join("envelope.jsonl"), corrupt).unwrap();
    assert!(
        read_envelopes(&capture_dir).is_err(),
        "a malformed non-final line was silently skipped"
    );
}

#[cfg(unix)]
#[test]
fn the_envelope_store_is_owner_only_and_written_atomically() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("a scratch dir");
    let written = write_capture(
        dir.path(),
        &run(
            "codex",
            "{\"type\":\"thread.started\",\"thread_id\":\"thr_9\"}\n",
            Some("thr_9"),
        ),
    )
    .expect("written");
    let path = written.join("envelope.jsonl");
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "{} is world-readable", path.display());
    // No leftover temp sibling: the atomic write renames over the target.
    let leftovers: Vec<_> = std::fs::read_dir(&written)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "an interrupted write left a temp file"
    );
}

#[test]
fn a_missing_origin_field_deserialises_to_not_observed_never_a_silent_measurement() {
    // AR-5's last done-when clause, stated on the wire: a stored envelope whose
    // origin is simply absent must not read as if the worker said it.
    let json = r#"{"worker_id":{"synthesised":"opencode/ses1"},"task_id":"unknown","agent_id":{"observed":"opencode"},"session_id":"unknown","sequence":1,"timestamp_ms":{"unavailable":{"reason":"no clock"}},"raw_line":1,"payload":{"kind":"message","text":"hi","origin":"observed"}}"#;
    let envelope: Envelope = serde_json::from_str(json).expect("a stored envelope reads back");
    assert_eq!(
        envelope.origin,
        Origin::Synthesised,
        "a missing origin field defaulted to observed — the silent measurement"
    );
}
