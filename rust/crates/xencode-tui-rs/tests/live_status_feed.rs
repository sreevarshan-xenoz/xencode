//! DK-1: the terminal app's status file follows the turn lifecycle.
use xencode_live_rs::{read_all, LiveState, Read};
use xencode_tui_rs::app::App;
use xencode_tui_rs::live_status::LiveFeed;

fn state_of(dir: &std::path::Path) -> (LiveState, String) {
    match read_all(dir).into_iter().next().expect("a status file") {
        Read::Ok(s) => (s.state, s.headline),
        Read::Skipped { reason, .. } => panic!("{reason}"),
    }
}

fn app_with_feed(dir: &std::path::Path) -> App<'static> {
    let mut app = App::for_tests();
    app.live = Some(LiveFeed::new(
        dir.to_path_buf(),
        "s1".into(),
        std::path::Path::new("."),
    ));
    app
}

#[test]
fn the_file_follows_working_needs_you_finished_and_failed() {
    let dir = tempfile::tempdir().unwrap();
    let mut app = app_with_feed(dir.path());
    app.live_turn_started();
    assert_eq!(state_of(dir.path()).0, LiveState::Working);
    app.live_tool_started("write_file src/auth.rs");
    assert_eq!(state_of(dir.path()).1, "write_file src/auth.rs");
    app.live_approval_waiting("write_file src/auth.rs");
    let (state, headline) = state_of(dir.path());
    assert_eq!(state, LiveState::NeedsYou);
    assert_eq!(headline, "waiting for you to allow: write_file src/auth.rs");
    app.live_turn_ended(None);
    assert_eq!(state_of(dir.path()).0, LiveState::Finished);
    app.live_turn_started();
    app.live_turn_ended(Some("connection refused\nmore detail"));
    let (state, headline) = state_of(dir.path());
    assert_eq!(state, LiveState::Failed);
    assert_eq!(headline, "connection refused");
    app.live_stopped();
    assert_eq!(
        state_of(dir.path()),
        (LiveState::Idle, "stopped".to_string())
    );
}

#[test]
fn a_secret_in_a_tool_summary_never_reaches_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut app = app_with_feed(dir.path());
    app.live_tool_started(
        "run_command curl -H \"Authorization: Bearer sk-FAKE-NOT-A-REAL-TEST-KEY\"",
    );
    let text = std::fs::read_to_string(dir.path().join("s1.json")).unwrap();
    assert!(!text.contains("sk-FAKE-NOT-A-REAL-TEST-KEY"), "{text}");
}

#[test]
fn dropping_the_feed_removes_the_file() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut app = app_with_feed(dir.path());
        app.live_turn_started();
        assert!(dir.path().join("s1.json").exists());
    }
    assert!(!dir.path().join("s1.json").exists());
}

#[tokio::test]
async fn the_real_lifecycle_points_update_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut app = app_with_feed(dir.path());
    // Nothing listens on port 9, so the turn's request fails on its own;
    // only the state written at each point is checked here.
    app.config.default_model = "llamacpp:none".into();
    app.config.llama_cpp_url = "http://127.0.0.1:9".into();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    app.chat_input.insert_str("hello");
    app.submit_message(tx);
    assert_eq!(state_of(dir.path()).0, LiveState::Working);
    app.append_generation("[DONE]");
    assert_eq!(state_of(dir.path()).0, LiveState::Finished);

    app.live_turn_started();
    app.live_turn_error = Some("llamacpp:none failed: connection refused".into());
    app.append_generation("[DONE]");
    assert_eq!(
        state_of(dir.path()),
        (
            LiveState::Failed,
            "llamacpp:none failed: connection refused".to_string()
        )
    );

    app.bytebot_running = true;
    app.bytebot_event("call:write_file notes.md");
    assert_eq!(
        state_of(dir.path()),
        (LiveState::Working, "write_file notes.md".to_string())
    );
}
