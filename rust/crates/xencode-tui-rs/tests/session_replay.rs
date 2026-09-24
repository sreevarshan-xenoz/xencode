//! Replaying a recorded session twice, over a loopback socket, and checking
//! that the two replays cannot be told apart.
//!
//! The recording under `tests/fixtures/sessions/` was made by a real agent run
//! against a local model server (see its README). Nothing here starts a model:
//! [`xencode_tui_rs::replay::replay`] serves the recorded bytes again on its own
//! port, and the whole agent loop runs against it — the HTTP client, the stream
//! reader that has to reassemble a tool call arriving in pieces, the permission
//! gate, and the tool itself, which really executes and really has to produce
//! the output the next recorded request expects.
//!
//! Two replays of one recording are compared as bytes, which is the point QA-1
//! has to prove: a ledger that stamped when it ran could never match, so every
//! time field in it comes out of the recording instead.

use std::path::{Path, PathBuf};

use xencode_context_rs::{RecordedToolCall, Session};
use xencode_tui_rs::replay::{replay, ReplayOptions};

const RECORDED_RUN: &str = "1790240197-eee44c61";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sessions")
        .join(format!("{RECORDED_RUN}.jsonl"))
}

/// A scratch tree holding the recording where `replay` expects to find it:
/// `<xencode dir>/cache/sessions/<run id>.jsonl`.
fn staged_recording(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-session-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(dir.join("cache/sessions")).unwrap();
    std::fs::copy(
        fixture(),
        dir.join("cache/sessions")
            .join(format!("{RECORDED_RUN}.jsonl")),
    )
    .unwrap();
    dir
}

fn recorded(xencode_dir: &Path) -> Session {
    xencode_context_rs::read_session(xencode_dir, RECORDED_RUN).expect("the committed recording")
}

#[tokio::test]
async fn two_replays_of_one_recording_write_the_same_ledger_byte_for_byte() {
    let xencode = staged_recording("identity");
    let tool_root = xencode.join("tree");
    std::fs::create_dir_all(&tool_root).unwrap();

    let first = replay(&ReplayOptions {
        run_id: RECORDED_RUN.to_string(),
        xencode_dir: xencode.clone(),
        tool_root: tool_root.clone(),
        out_dir: xencode.join("replay-a"),
        run_tools: true,
    })
    .await
    .expect("the first replay failed");
    let second = replay(&ReplayOptions {
        run_id: RECORDED_RUN.to_string(),
        xencode_dir: xencode.clone(),
        tool_root: tool_root.clone(),
        out_dir: xencode.join("replay-b"),
        run_tools: true,
    })
    .await
    .expect("the second replay failed");

    assert!(first.matches(), "{:?}", first);
    assert!(second.matches(), "{:?}", second);
    let a = std::fs::read_to_string(&first.ledger).unwrap();
    let b = std::fs::read_to_string(&second.ledger).unwrap();
    assert_eq!(a, b, "two replays of one recording differ:\n{a}\n---\n{b}");
    // Different output directories, so the only way the two files can be equal
    // is for nothing about the replay's own moment or place to be in them.
    assert_ne!(first.ledger, second.ledger);

    // The command-line default is one output directory per recording, so a second
    // `xencode replay` lands on top of the first replay's own recording. That has
    // to replace it, not continue it: a file holding two runs reads back as one
    // run that said everything twice, and the report would be about a replay that
    // never happened.
    let again = replay(&ReplayOptions {
        run_id: RECORDED_RUN.to_string(),
        xencode_dir: xencode.clone(),
        tool_root: tool_root.clone(),
        out_dir: xencode.join("replay-a"),
        run_tools: true,
    })
    .await
    .expect("the replay into a used output directory failed");
    assert!(again.matches(), "{:?}", again);
    assert_eq!(
        again.calls_replayed, 2,
        "only this replay's calls, not both"
    );
    assert_eq!(
        std::fs::read_to_string(&again.ledger).unwrap(),
        a,
        "replaying into a directory an earlier replay used changed the answer"
    );

    let _ = std::fs::remove_dir_all(&xencode);
}

#[tokio::test]
async fn the_replay_ran_the_tool_it_recorded_and_the_model_saw_its_real_output() {
    let xencode = staged_recording("tool");
    let tool_root = xencode.join("tree");
    std::fs::create_dir_all(&tool_root).unwrap();
    let report = replay(&ReplayOptions {
        run_id: RECORDED_RUN.to_string(),
        xencode_dir: xencode.clone(),
        tool_root: tool_root.clone(),
        out_dir: xencode.join("replay"),
        run_tools: true,
    })
    .await
    .expect("replay");

    let recording = recorded(&xencode);
    assert_eq!(report.calls_recorded, 2, "the fixture is one tool round");
    assert_eq!(report.calls_replayed, 2);
    assert_eq!(report.unanswered, 0);
    assert_eq!(report.tool_calls_replayed, 1);
    assert_eq!(report.tool_calls_recorded, 1);

    let ledger = std::fs::read_to_string(&report.ledger).unwrap();
    let lines: Vec<serde_json::Value> = ledger
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["name"], "run_command");
    assert_eq!(
        lines[0]["arguments"]["command"],
        recording.calls[0].tools[0].arguments["command"]
    );
    assert_eq!(lines[0]["outcome"], "done");
    // The digest is of what the shell printed during THIS replay, taken from a
    // string written here rather than copied from the recording, so a replay
    // that ran the command and got something else could not pass.
    let expected = RecordedToolCall {
        result: "$ echo $((27 * 43))\nexit 0\n1161".to_string(),
        ..recording.calls[0].tools[0].clone()
    };
    assert_eq!(lines[0]["result_sha256"], expected.result_digest());
    assert_eq!(
        lines[0]["result_chars"].as_u64().unwrap() as usize,
        expected.result.chars().count()
    );
    // Times come from the recording: a replay stamps the moment it was asked
    // for, and then no two replays could ever be compared.
    assert_eq!(
        lines[0]["recorded_ts_unix_ms"],
        serde_json::json!(recording.calls[0].ts_unix_ms)
    );
    assert_eq!(
        lines[0]["recorded_duration_ms"],
        serde_json::json!(recording.calls[0].duration_ms)
    );
    let _ = std::fs::remove_dir_all(&xencode);
}

#[tokio::test]
async fn a_replay_without_permission_to_run_tools_stops_where_a_headless_run_would() {
    let xencode = staged_recording("refused");
    let tool_root = xencode.join("tree");
    std::fs::create_dir_all(&tool_root).unwrap();
    let report = replay(&ReplayOptions {
        run_id: RECORDED_RUN.to_string(),
        xencode_dir: xencode.clone(),
        tool_root: tool_root.clone(),
        out_dir: xencode.join("replay"),
        run_tools: false,
    })
    .await
    .expect("replay");

    // The first request is answered from the recording; the tool behind it is
    // refused, so there is no second request to answer and the replay says so
    // instead of quietly ending short.
    assert_eq!(report.calls_replayed, 1);
    assert!(!report.matches());
    let ledger = std::fs::read_to_string(&report.ledger).unwrap();
    let line: serde_json::Value = serde_json::from_str(ledger.lines().next().unwrap()).unwrap();
    assert_eq!(line["outcome"], "denied");
    assert_eq!(line["replayed"], true);
    let _ = std::fs::remove_dir_all(&xencode);
}

#[tokio::test]
async fn a_replay_cannot_be_told_to_overwrite_the_recording_it_is_reading() {
    let xencode = staged_recording("selfwrite");
    let error = replay(&ReplayOptions {
        run_id: RECORDED_RUN.to_string(),
        xencode_dir: xencode.clone(),
        tool_root: xencode.clone(),
        out_dir: xencode.clone(),
        run_tools: false,
    })
    .await
    .expect_err("writing a replay over its own recording must be refused");
    assert!(error.contains("where the recording lives"), "{error}");
    assert!(fixture().exists(), "the fixture itself must be untouched");
    let _ = std::fs::remove_dir_all(&xencode);
}
