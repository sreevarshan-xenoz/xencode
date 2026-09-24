//! Replaying recorded provider traffic through the production streaming reader.
//!
//! Everything in `src/` is exercised in-process somewhere, and that is exactly
//! the problem: an in-process fake hands the reader a buffer it chose, so it
//! can never catch a stream that arrives in pieces. These tests start a real
//! loopback server, send the recorded request over a real socket, and let the
//! same `post_sse_stream` that talks to a real llama.cpp parse the answer. The
//! only thing simulated is the far end.
//!
//! The four cassettes in `tests/fixtures/cassettes` are captures of traffic
//! taken from a `llama-server` process running on the machine these tests were
//! written on. They are not expectations about what a server would say; when a
//! recording disagrees with an assumption, the recording wins and the
//! assumption is what gets fixed.

use std::path::{Path, PathBuf};

use xencode_providers_rs::playback::{self, Cassette, CASSETTE_FORMAT};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("cassettes")
}

fn cassette(name: &str) -> Cassette {
    let path = fixtures().join(name);
    Cassette::load(&path).unwrap_or_else(|e| panic!("{name} refused to load: {e}"))
}

fn all_cassettes() -> Vec<(&'static str, Cassette)> {
    [
        "answer-one-eighty.json",
        "answer-japanese.json",
        "calculator-two-turns.json",
        "reasoning-only.json",
    ]
    .into_iter()
    .map(|name| (name, cassette(name)))
    .collect()
}

/// The house rule for this directory: a fixture is a recording, and a
/// recording that does not say where it came from is just a guess with a file
/// extension. A test that lets someone replace a capture with something they
/// typed by hand is worse than no fixture at all.
#[test]
fn every_committed_cassette_says_where_its_bytes_came_from() {
    let recorded = all_cassettes();
    assert_eq!(recorded.len(), 4);
    for (name, cassette) in recorded {
        assert_eq!(cassette.format, CASSETTE_FORMAT, "{name}");
        assert!(
            cassette.captured.server.contains("build 10809"),
            "{name} does not name the server build that produced it"
        );
        assert!(
            cassette.captured.model.contains("GGUF") || cassette.captured.model.contains("gguf"),
            "{name} does not name the model file that was loaded"
        );
        assert!(
            !cassette.captured.tool.is_empty(),
            "{name} says nothing about how the bytes were taken"
        );
        assert!(
            cassette.captured.note.len() > 80,
            "{name} has no note explaining what the recording shows"
        );
        assert_ne!(cassette.captured.at_unix_ms, 0, "{name} is undated");
        for interaction in &cassette.interactions {
            assert_eq!(interaction.request.method, "POST");
            assert!(interaction.response.body.contains("data: "));
        }
    }
}

#[tokio::test]
async fn a_recorded_answer_arrives_whole_even_when_every_line_is_cut_in_two() {
    for (name, expected) in [
        ("answer-one-eighty.json", "The number is 108."),
        ("answer-japanese.json", "サーバは準備が整いました。"),
    ] {
        let recorded = cassette(name);
        let interaction = &recorded.interactions[0];
        let server = recorded
            .play_with(playback::Delivery::SplitMidLine)
            .expect("loopback server");
        let request =
            playback::recorded_request(interaction).expect("the cassette kept its request");

        let mut streamed = String::new();
        let step = playback::replay(
            &server.base_url(),
            &interaction.request.path,
            &request,
            |piece| streamed.push_str(piece),
        )
        .await
        .unwrap_or_else(|e| panic!("{name} could not be replayed: {e}"));

        assert_eq!(step.text, expected, "{name}: the answer came back changed");
        assert_eq!(
            streamed, expected,
            "{name}: what the caller was shown differs from what was assembled"
        );
        assert!(
            step.tool_calls.is_empty(),
            "{name}: nobody asked for a tool call"
        );
        assert_eq!(
            server.misses(),
            0,
            "{name} refused its own recorded request"
        );
    }
}

#[tokio::test]
async fn a_tool_call_split_across_a_dozen_frames_reassembles_into_one_call() {
    let recorded = cassette("calculator-two-turns.json");
    let first = &recorded.interactions[0];
    let server = recorded
        .play_with(playback::Delivery::SplitMidLine)
        .expect("loopback server");
    let request = playback::recorded_request(first).expect("the cassette kept its request");

    let step = playback::replay(&server.base_url(), &first.request.path, &request, |_| {})
        .await
        .expect("turn 1 replays");

    assert_eq!(step.tool_calls.len(), 1);
    let call = &step.tool_calls[0];
    assert_eq!(call.name, "calculator");
    let arguments: serde_json::Value = serde_json::from_str(
        call.arguments
            .as_str()
            .unwrap_or_else(|| panic!("arguments arrived as {:?}", call.arguments)),
    )
    .expect("the assembled arguments are the JSON object the model meant");
    assert_eq!(arguments["expr"], "27 * 43");
    assert_eq!(server.misses(), 0);
}

/// The point of the two-turn recording: the answer a caller gets depends on a
/// number no one wrote into a test file. The expression is taken from the
/// replayed tool call and evaluated by a real shell; the recording of the
/// second turn is only a valid answer if it carries that same result.
#[tokio::test]
async fn the_second_turn_of_a_recorded_agent_run_answers_from_a_real_tool_result() {
    let recorded = cassette("calculator-two-turns.json");
    let server = recorded
        .play_with(playback::Delivery::SplitMidLine)
        .expect("loopback server");

    let first = &recorded.interactions[0];
    let step = playback::replay(
        &server.base_url(),
        &first.request.path,
        &playback::recorded_request(first).expect("recorded request"),
        |_| {},
    )
    .await
    .expect("turn 1 replays");
    let arguments: serde_json::Value = serde_json::from_str(
        step.tool_calls[0]
            .arguments
            .as_str()
            .expect("string arguments"),
    )
    .expect("arguments are JSON");
    let expression = arguments["expr"]
        .as_str()
        .expect("expr is text")
        .to_string();
    assert!(
        expression
            .chars()
            .all(|c| c.is_ascii_digit() || " +-*/%()".contains(c)),
        "refusing to hand {expression:?} to a shell: only arithmetic is expected here"
    );

    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!("echo $(({expression}))"))
        .output()
        .expect("the shell ran");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result = String::from_utf8(output.stdout)
        .expect("arithmetic is text")
        .trim()
        .to_string();
    assert_eq!(result, "1161");

    let second = &recorded.interactions[1];
    let recorded_request = second
        .request
        .body
        .as_deref()
        .expect("turn 2 kept its request");
    assert!(
        recorded_request.contains(&result),
        "the recording's second turn carries a different tool result than the one \
         this machine just computed, so replaying it would be pretending"
    );

    let step = playback::replay(
        &server.base_url(),
        &second.request.path,
        &playback::recorded_request(second).expect("recorded request"),
        |_| {},
    )
    .await
    .expect("turn 2 replays");
    assert_eq!(step.text, "The result of 27 times 43 is 1161.");
    assert_eq!(server.misses(), 0);
}

/// What a thinking model does to this product today: it answers at length, the
/// transport carries every word, and none of it reaches the user, because the
/// text arrives as `reasoning_content` and no reader in the crate looks at that
/// field. This test asserts the empty answer on purpose. When reasoning is
/// surfaced, this is the test that has to change, and it should change to
/// expect the thinking text rather than have it deleted.
#[tokio::test]
async fn a_model_that_thinks_out_loud_answers_with_nothing_a_reader_can_see() {
    let recorded = cassette("reasoning-only.json");
    let interaction = &recorded.interactions[0];
    let body = &interaction.response.body;
    let reasoning_frames = body.matches("\"reasoning_content\":").count();
    let content_frames = body.matches("\"content\":\"").count();
    assert!(
        reasoning_frames > 50,
        "the recording should still show a model writing thinking, found {reasoning_frames}"
    );
    assert_eq!(
        content_frames, 0,
        "this recording answers through reasoning only; if it was replaced with \
         an ordinary answer it no longer pins anything"
    );

    let server = recorded
        .play_with(playback::Delivery::SplitMidLine)
        .expect("loopback server");
    let step = playback::replay(
        &server.base_url(),
        &interaction.request.path,
        &serde_json::json!({"model": "recorded-model", "messages": [], "stream": true}),
        |_| {},
    )
    .await
    .expect("the cassette answers any POST to that path");
    assert!(
        step.text.is_empty(),
        "reasoning text is now reaching the answer, which is good news and means \
         this test's name and expectation need updating together: {step:?}"
    );
    assert_eq!(server.misses(), 0);
}

/// A cassette that answers a question nobody asked is a silent lie, so the
/// matcher has to be able to say no.
#[tokio::test]
async fn replaying_a_question_that_was_not_recorded_is_refused_loudly() {
    let recorded = cassette("answer-one-eighty.json");
    let server = recorded.play().expect("loopback server");
    let error = playback::replay(
        &server.base_url(),
        "/v1/chat/completions",
        &serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "what is the capital of France?"}], "stream": true}),
        |_| {},
    )
    .await
    .expect_err("nobody recorded an answer about France");
    assert!(
        error.to_string().contains("recorded") || error.to_string().contains("400"),
        "the refusal should say the request was not recorded, got: {error}"
    );
    assert_eq!(server.misses(), 1);
}
