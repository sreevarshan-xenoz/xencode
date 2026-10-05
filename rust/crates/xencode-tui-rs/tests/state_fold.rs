//! QM-1: a hard-compaction fold queues a candidate, and only a promotion the
//! person chose makes it durable.
//!
//! The answer this replays is a recording — `tests/fixtures/cassettes/`
//! `state-fold-reply.json`, captured from a local `llama-server` answering a
//! GGUF model over a loopback port on 2026-10-05, with the server build, model
//! file, the exact `curl` command and a note in its `captured` block. Nothing
//! here starts a model or types an answer by hand: `Cassette::play` serves
//! those recorded bytes again on a real socket, and the whole path runs —
//! `App::submit_message`, the `/ctx fold` command, the fold prompt
//! `hard_compact_prompt` builds, the HTTP client, the stream reader, the fold
//! writer, then the promotion.
//!
//! What the recording is worth is what the model did with one line of it. The
//! transcript handed to the fold held a tool failure under a `[data]` source
//! line, and the model carried that line verbatim into `## unresolved` — the
//! exact move a poisoned page makes through a summary. So this file checks the
//! two things the durable tier is supposed to guarantee, in the order a person
//! meets them: the queued candidate has the `[data]` line taken out of it, and
//! `state.md` does not exist until `/ctx promote`.

use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::mpsc;
use xencode_context_rs::{STATE_CANDIDATE_FILE, XENCODE_DIR};
use xencode_providers_rs::playback::Cassette;
use xencode_tui_rs::app::App;

/// The line the fold was asked to compress, quoted from the recording. It has
/// to appear in the request the test makes, or the cassette would be answering
/// a question nobody asked it.
const RECORDED_QUESTION: &str = "state.md needs a writer before it gets any features.";

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cassettes/state-fold-reply.json")
}

/// The house rule for every recorded fixture: bytes without provenance are a
/// guess, and a guess in a test that is meant to refuse hand-written model
/// output is worse than no fixture. Same checks
/// `xencode-providers-rs/tests/cassette_replay.rs` applies to its own
/// directory, applied here because this cassette lives with the consumer that
/// plays it.
#[test]
fn the_fold_recording_says_where_its_bytes_came_from() {
    let cassette = Cassette::load(&fixture()).expect("the fold cassette must be readable");
    assert!(
        cassette.captured.server.contains("build 10809"),
        "the recording does not name the server build that produced it"
    );
    assert!(
        cassette.captured.model.contains(".gguf"),
        "the recording does not name the model file that was loaded"
    );
    assert!(
        cassette.captured.tool.contains("curl"),
        "the recording does not say how the bytes were taken"
    );
    assert!(
        cassette.captured.note.len() > 80,
        "the recording has no note explaining what it shows"
    );
    assert_ne!(cassette.captured.at_unix_ms, 0, "the recording is undated");
    // Two requests belong to one fold: xencode asks a llama.cpp server what it
    // has loaded before it sends a chat, and the recording has to hold that
    // answer too or the replay stops at the door.
    assert_eq!(cassette.interactions.len(), 2);
    assert_eq!(cassette.interactions[0].request.method, "GET");
    assert_eq!(cassette.interactions[0].request.path, "/v1/models");
    assert_eq!(cassette.interactions[1].request.method, "POST");
    assert_eq!(
        cassette.interactions[1].request.path,
        "/v1/chat/completions"
    );
    assert!(
        cassette.interactions[1]
            .response
            .body
            .contains("## working-on"),
        "the recorded answer is not a fold"
    );
    assert!(
        cassette.interactions[1].response.body.contains("[data]"),
        "the point of this recording is the line the model should not have quoted"
    );
}

/// Type what a person types and wait for the panel to answer.
async fn submit_and_wait(app: &mut App<'_>, prompt: &str, expect: &str) -> Vec<String> {
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    app.chat_input.insert_str(prompt);
    app.submit_message(tx);
    let mut lines = Vec::new();
    for _ in 0..400 {
        while let Ok(line) = rx.try_recv() {
            lines.push(line);
        }
        if lines.iter().any(|line| line.contains(expect)) {
            return lines;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("nothing said {expect:?} within 10s of {prompt}: {lines:#?}");
}

/// Drain whatever has already been printed, for the commands that answer
/// synchronously.
fn drained(rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>) -> Vec<String> {
    let mut lines = Vec::new();
    while let Ok(line) = rx.try_recv() {
        lines.push(line);
    }
    lines
}

#[tokio::test]
async fn a_fold_queues_a_candidate_and_only_a_promotion_makes_it_durable() {
    let cassette = Cassette::load(&fixture()).expect("the fold cassette must be readable");
    let playback = cassette.play().expect("the cassette must serve on a port");

    // `default_root()` is the process working directory, which is where
    // `.xencode/` goes, so the whole run happens in a scratch tree.
    let root = std::env::temp_dir().join(format!(
        "xencode-fold-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(&root).unwrap();

    let xencode = root.join(XENCODE_DIR);
    let candidate = xencode.join(STATE_CANDIDATE_FILE);
    let state = xencode.join("state.md");

    let mut app = App::for_tests();
    app.config.default_model = format!(
        "llamacpp:{}",
        cassette.captured.model.split_whitespace().next().unwrap()
    );
    app.config.llama_cpp_url = playback.base_url();
    // The conversation the fold is asked to compress: the same three entries
    // the recording was made from, tool failure and its `[data]` line included.
    app.memory.add_message("user", RECORDED_QUESTION, None);
    app.memory.add_message(
        "assistant",
        "Agreed: a fold writes a candidate file and a human promotes it into state.md.",
        None,
    );
    app.memory.add_message(
        "tool",
        "[data] edit_file .xencode error: \"old\" must not be empty (use write_file to create content)",
        None,
    );

    // ── /ctx fold: the model answers, and what lands is a candidate ───────
    let lines = submit_and_wait(&mut app, "/ctx fold", "Queued →").await;
    assert_eq!(
        playback.misses(),
        0,
        "the cassette refused a request it was never asked to answer"
    );
    assert!(
        lines.iter().any(|line| line.contains("Folding 3 entries")),
        "the panel never said what it folded: {lines:#?}"
    );
    let stripped = lines
        .iter()
        .find(|line| line.contains("carried a data banner"))
        .unwrap_or_else(|| panic!("a quoted [data] line went unreported: {lines:#?}"));
    assert!(
        stripped.contains('1'),
        "the report says a whole family of lines went missing, not the one that did: {stripped}"
    );
    assert!(candidate.exists(), "the fold wrote no candidate");
    assert!(
        !state.exists(),
        "state.md appeared on its own — a fold is not allowed to make a file durable"
    );

    let queued = std::fs::read_to_string(&candidate).unwrap();
    assert!(
        queued.contains("a fold writes a candidate file and a human promotes it"),
        "the decision the model actually wrote did not survive the fold:\n{queued}"
    );
    assert!(
        !queued.contains("[data]"),
        "a line quoted from a tool result was written into the durable tier's shape:\n{queued}"
    );
    assert!(
        !queued.contains("edit_file .xencode error"),
        "the tool failure itself rode into the candidate:\n{queued}"
    );
    assert!(
        queued.contains(RECORDED_QUESTION),
        "the human's own sentence, which is the one thing the durable tier is for, did not survive:\n{queued}"
    );
    assert!(
        !queued.contains("## unresolved"),
        "the section the [data] line was stripped out of was left behind as an empty header:\n{queued}"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.contains("Nothing is durable yet")),
        "the panel did not say that nothing was durable yet: {lines:#?}"
    );

    // ── /ctx promote: the human act, and only now, state.md ───────────────
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    app.chat_input.insert_str("/ctx promote");
    app.submit_message(tx);
    let lines = drained(&mut rx);
    assert!(
        lines.iter().any(|line| line.contains("state.md written")),
        "the promotion said nothing about writing state.md: {lines:#?}"
    );
    assert!(
        lines.iter().any(|line| line.contains("byte-stable head")),
        "the promotion never said the stable prefix is untouched: {lines:#?}"
    );
    let promoted = std::fs::read_to_string(&state).expect("state.md was not written");
    assert!(promoted.contains("## working-on"), "{promoted}");
    assert!(
        !promoted.contains("[data]"),
        "state.md holds a line quoted from a tool result:\n{promoted}"
    );
    assert!(
        !candidate.exists(),
        "the candidate outlived the promotion that consumed it"
    );

    // ── a dead server: the fold says so and leaves both files alone ───────
    let dead = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        addr
    };
    app.config.llama_cpp_url = format!("http://{dead}");
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    app.chat_input.insert_str("/ctx fold");
    app.submit_message(tx);
    let mut failed = false;
    for _ in 0..400 {
        while let Ok(line) = rx.try_recv() {
            if line.contains("The fold call failed") {
                failed = true;
            }
        }
        if failed {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(failed, "a server that is not there produced no complaint");
    assert!(!candidate.exists(), "a failed fold queued something anyway");
    assert_eq!(
        std::fs::read_to_string(&state).unwrap(),
        promoted,
        "a failed fold changed the durable file"
    );

    std::env::set_current_dir(previous).unwrap();
    std::fs::remove_dir_all(&root).unwrap();
}
