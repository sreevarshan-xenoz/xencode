//! EN-2: `xencode engine` runs a project's agent work in its own process,
//! reached over the local socket. Every test starts the real binary against
//! a settings folder of its own, whose model server address has nothing
//! listening on it, so any model call fails at once and in words.

use std::io::BufRead;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc as std_mpsc;
use std::time::{Duration, Instant};

use xencode_tui_rs::engine::address::Address;
use xencode_tui_rs::engine::proto::{self, ClientMsg, EngineMsg, PROTOCOL_VERSION};
use xencode_tui_rs::engine::transport::{connect, Conn};
use xencode_tui_rs::engine::view::View;

/// Settings whose model calls go to a port nobody listens on, and whose
/// llama.cpp server program does not exist, so nothing is started.
fn settings() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("no-llama-server-here");
    let config = serde_json::json!({
        "default_model": "llamacpp:none",
        "llama_cpp_url": "http://127.0.0.1:9",
        "llama_cpp_executable": missing.to_string_lossy(),
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    dir
}

/// A running `xencode engine`, killed when the test ends however it ends.
struct Engine {
    child: Child,
    lines: std_mpsc::Receiver<String>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Engine {
    fn start(project: &Path, config: &Path) -> Engine {
        Engine::start_with(project, config, &[])
    }

    fn start_with(project: &Path, config: &Path, extra: &[&str]) -> Engine {
        let mut child = Command::new(env!("CARGO_BIN_EXE_xencode"))
            .args(["engine", "--project"])
            .arg(project)
            .args(extra)
            .env("XCODE_CONFIG_DIR", config)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, lines) = std_mpsc::channel();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout)
                .lines()
                .map_while(Result::ok)
            {
                if tx.send(line).is_err() {
                    return;
                }
            }
        });
        Engine { child, lines }
    }

    /// The next line the engine prints, within `wait`.
    fn line(&self, wait: Duration) -> Option<String> {
        self.lines.recv_timeout(wait).ok()
    }

    /// Where the engine says it listens, once it does.
    fn address(&self) -> Address {
        let line = self
            .line(Duration::from_secs(30))
            .expect("the engine says where it listens");
        let at = line
            .split_once(" listening on ")
            .unwrap_or_else(|| panic!("not a listening line: {line}"))
            .1
            .trim()
            .to_string();
        if at.starts_with(r"\\.\pipe\") {
            Address::Pipe(at)
        } else {
            Address::Socket(at.into())
        }
    }

    /// Whether the process ends within `wait`.
    fn exits_within(&mut self, wait: Duration) -> Option<std::process::ExitStatus> {
        let start = Instant::now();
        while start.elapsed() < wait {
            if let Some(status) = self.child.try_wait().unwrap() {
                return Some(status);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        None
    }
}

async fn send(conn: &mut Conn, msg: ClientMsg) {
    conn.send(&proto::encode(&msg)).await.unwrap();
}

/// The next message, within ten seconds.
async fn next(conn: &mut Conn) -> EngineMsg {
    let line = tokio::time::timeout(Duration::from_secs(10), conn.recv())
        .await
        .expect("the engine answered in time")
        .unwrap()
        .expect("the engine kept the connection open");
    proto::decode_engine(&line).unwrap()
}

/// Connect and say hello; returns the connection and the first full view.
async fn join(addr: &Address, client: &str) -> (Conn, View) {
    let mut conn = connect(addr).await.unwrap();
    send(
        &mut conn,
        ClientMsg::Hello {
            version: PROTOCOL_VERSION,
            client: client.into(),
        },
    )
    .await;
    match next(&mut conn).await {
        EngineMsg::Hello { version, .. } => assert_eq!(version, PROTOCOL_VERSION),
        other => panic!("expected hello, got {other:?}"),
    }
    match next(&mut conn).await {
        EngineMsg::View { view } => (conn, *view),
        other => panic!("expected the full view, got {other:?}"),
    }
}

/// Read views until `done` holds for the window's copy of the transcript
/// and tasks, or `wait` passes.
async fn watch_until(
    conn: &mut Conn,
    mut seen: View,
    wait: Duration,
    done: impl Fn(&View) -> bool,
) -> View {
    let start = Instant::now();
    while !done(&seen) {
        assert!(
            start.elapsed() < wait,
            "gave up waiting; last view: {seen:?}"
        );
        let Ok(Ok(Some(line))) =
            tokio::time::timeout(Duration::from_millis(500), conn.recv()).await
        else {
            continue;
        };
        if let Ok(EngineMsg::View { view }) = proto::decode_engine(&line) {
            let from = view.messages_from.min(seen.messages.len());
            seen.messages.truncate(from);
            seen.messages.extend(view.messages);
            if view.tasks.is_some() {
                seen.tasks = view.tasks;
            }
        }
    }
    seen
}

#[tokio::test]
async fn hello_gets_a_snapshot() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let (mut conn, view) = join(&addr, "test").await;
    assert_eq!(view.messages_from, 0);
    assert_eq!(view.model.as_deref(), Some("llamacpp:none"));
    assert_eq!(view.generating, Some(false));
    assert_eq!(view.tasks, Some(Vec::new()));
    send(&mut conn, ClientMsg::Goodbye).await;
}

#[tokio::test]
async fn two_clients_both_reach_the_engine() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let (mut a, seen_a) = join(&addr, "a").await;
    let (mut b, seen_b) = join(&addr, "b").await;
    let note = |text: &str| ClientMsg::Note {
        role: "system".into(),
        content: text.into(),
    };
    tokio::join!(send(&mut a, note("from a")), send(&mut b, note("from b")));
    let both = |view: &View| {
        let has = |text: &str| view.messages.iter().any(|m| m.content == text);
        has("from a") && has("from b")
    };
    let wait = Duration::from_secs(10);
    let seen_a = watch_until(&mut a, seen_a, wait, both).await;
    let seen_b = watch_until(&mut b, seen_b, wait, both).await;
    assert_eq!(seen_a.messages, seen_b.messages);
}

#[tokio::test]
async fn a_task_outlives_its_window() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    {
        let (mut first, _) = join(&addr, "first").await;
        send(
            &mut first,
            ClientMsg::EnqueueTask {
                text: "write a note".into(),
            },
        )
        .await;
        // The window goes away without saying goodbye.
    }
    let (mut second, seen) = join(&addr, "second").await;
    let ended = watch_until(&mut second, seen, Duration::from_secs(60), |view| {
        view.tasks.as_ref().is_some_and(|tasks| {
            tasks
                .iter()
                .any(|t| t.task.text == "write a note" && t.task.state.words() == "failed")
        })
    })
    .await;
    let task = &ended.tasks.unwrap()[0];
    assert!(
        task.this_session,
        "the task was made in the engine's session"
    );
    assert!(
        task.task
            .note
            .as_deref()
            .is_some_and(|note| !note.is_empty()),
        "the failure is said in words: {task:?}"
    );
}

#[test]
fn an_idle_engine_exits() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let mut engine = Engine::start(project.path(), config.path());
    let _ = engine.address();
    let status = engine
        .exits_within(Duration::from_secs(20))
        .expect("an engine with no window and nothing to do exits");
    assert!(status.success(), "{status:?}");
}

#[tokio::test]
async fn a_second_engine_for_the_same_project_steps_aside() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let mut second = Engine::start(project.path(), config.path());
    let said = second.line(Duration::from_secs(30)).unwrap_or_default();
    assert!(said.contains("already running"), "{said}");
    let status = second
        .exits_within(Duration::from_secs(10))
        .expect("the second engine steps aside");
    assert!(status.success(), "{status:?}");
    let _ = join(&addr, "still there").await;
}

#[tokio::test]
async fn an_engine_killed_mid_way_is_replaced_by_a_new_one() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let mut first = Engine::start(project.path(), config.path());
    let addr = first.address();
    first.child.kill().unwrap();
    first.child.wait().unwrap();
    // On Unix the socket file and lock file of the killed engine are still
    // there; a new engine replaces them.
    let second = Engine::start(project.path(), config.path());
    assert_eq!(second.address(), addr);
    let _ = join(&addr, "after").await;
}

// The terminal app as a window onto the engine: a real `App`, linked to a
// real engine process, driven frame by frame the way the frame loop does.

use xencode_tui_rs::app::App;
use xencode_tui_rs::engine::link;

/// Run window frames until `done` holds or `wait` passes.
async fn frames_until(app: &mut App<'_>, wait: Duration, done: impl Fn(&App<'_>) -> bool) {
    // A frame always runs first: lines the window just added only reach the
    // engine on a frame.
    let start = Instant::now();
    loop {
        let frame = link::frame(app);
        assert!(
            frame.lost.is_none(),
            "the engine went away: {:?}",
            frame.lost
        );
        if done(app) {
            return;
        }
        assert!(start.elapsed() < wait, "gave up waiting on the window");
        tokio::time::sleep(Duration::from_millis(33)).await;
    }
}

async fn window_onto(addr: &Address) -> App<'static> {
    let (linked, view) = link::open(addr, "terminal").await.unwrap();
    let mut app = App::for_tests();
    link::become_window(&mut app, linked, view);
    app
}

#[tokio::test]
async fn a_window_sends_bytebot_work_to_the_engine_and_sees_it_end() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let mut app = window_onto(&addr).await;
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    xencode_tui_rs::engine::act(
        &mut app,
        ClientMsg::SubmitChat {
            prompt: "/bytebot write a note".into(),
        },
        &tx,
    );
    frames_until(&mut app, Duration::from_secs(60), |app| {
        app.bytebot_tasks
            .iter()
            .any(|t| t.text == "write a note" && t.state.words() == "failed")
    })
    .await;
    assert!(app.bytebot_tasks[0].this_session);
    assert!(!app.bytebot_running, "the window runs nothing itself");
    assert!(
        app.messages
            .iter()
            .any(|m| m.content == "/bytebot write a note"),
        "the engine's transcript, shown in the window, has the command"
    );
}

#[tokio::test]
async fn a_command_run_in_the_window_reaches_every_window() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let mut app = window_onto(&addr).await;
    let (mut other, seen) = join(&addr, "other").await;
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    xencode_tui_rs::engine::act(
        &mut app,
        ClientMsg::SubmitChat {
            prompt: "/help".into(),
        },
        &tx,
    );
    let lines = app.messages.len();
    assert!(lines > 0, "/help wrote its answer in the window");
    frames_until(&mut app, Duration::from_secs(10), |app| {
        app.messages.len() == lines && app.messages.iter().any(|m| m.content == "/help")
    })
    .await;
    let seen = watch_until(&mut other, seen, Duration::from_secs(10), |view| {
        view.messages.len() == lines
    })
    .await;
    let mine: Vec<_> = app.messages.iter().map(|m| m.content.clone()).collect();
    let theirs: Vec<_> = seen.messages.iter().map(|m| m.content.clone()).collect();
    assert_eq!(mine, theirs);
}

#[tokio::test]
async fn a_lost_engine_is_said_in_words() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let mut engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let mut app = window_onto(&addr).await;
    engine.child.kill().unwrap();
    engine.child.wait().unwrap();

    let start = Instant::now();
    let why = loop {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the loss was never noticed"
        );
        if let Some(why) = link::frame(&mut app).lost {
            break why;
        }
        tokio::time::sleep(Duration::from_millis(33)).await;
    };
    assert!(app.engine_link.is_none());

    let started = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Engine>::new()));
    let begin: link::Starter = {
        let started = started.clone();
        let (project, config) = (project.path().to_path_buf(), config.path().to_path_buf());
        std::sync::Arc::new(move || {
            started
                .lock()
                .unwrap()
                .push(Engine::start(&project, &config));
            Ok(())
        })
    };
    let asked = Instant::now();
    link::recover(&mut app, &addr, begin, &why);
    assert!(
        asked.elapsed() < Duration::from_millis(500),
        "the window froze for {:?} while a new engine started",
        asked.elapsed()
    );
    frames_until(&mut app, Duration::from_secs(20), |app| {
        app.engine_link.is_some()
    })
    .await;
    let said: Vec<_> = app.toasts.iter().map(|t| t.message.clone()).collect();
    assert!(
        said.iter()
            .any(|m| m.starts_with("the engine stopped (") && m.ends_with("starting a new one")),
        "{said:?}"
    );
    assert!(
        app.engine_link.is_some(),
        "a new engine was reached: {said:?}"
    );
}

#[tokio::test]
async fn a_window_never_changes_the_engines_model_by_itself() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let mut app = window_onto(&addr).await;
    assert_eq!(app.config.default_model, "llamacpp:none");
    // The window's own model list does not hold the engine's model; it used
    // to switch to the first model it found and save that to the settings.
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    app.apply_token(r#"[MODELS]["ollama:something-else"]"#, &tx);
    assert_eq!(app.config.default_model, "llamacpp:none");
    assert_eq!(
        app.available_models,
        vec!["ollama:something-else".to_string()]
    );
}

#[tokio::test]
async fn a_window_that_stops_reading_is_let_go() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let (mut stalled, _) = join(&addr, "stalled").await;
    let (busy, _) = join(&addr, "busy").await;
    let (mut busy_in, mut busy_out) = busy.split();
    let drain = tokio::spawn(async move { while let Ok(Some(_)) = busy_in.recv().await {} });
    // The busy window keeps the transcript growing; the stalled one reads
    // nothing, as a suspended terminal would.
    let text = "x".repeat(4000);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(20) {
        let note = ClientMsg::Note {
            role: "system".into(),
            content: text.clone(),
        };
        busy_out.send(&proto::encode(&note)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let ended = tokio::time::timeout(Duration::from_secs(30), async {
        while let Ok(Some(_)) = stalled.recv().await {}
    })
    .await;
    drain.abort();
    assert!(
        ended.is_ok(),
        "the engine kept queueing for a window that stopped reading"
    );
}

#[tokio::test]
async fn a_window_whose_engine_cannot_come_back_stays_responsive() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let mut engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let mut app = window_onto(&addr).await;
    engine.child.kill().unwrap();
    engine.child.wait().unwrap();
    let start = Instant::now();
    let why = loop {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the loss was never noticed"
        );
        if let Some(why) = link::frame(&mut app).lost {
            break why;
        }
        tokio::time::sleep(Duration::from_millis(33)).await;
    };
    // Starting succeeds but no engine ever answers.
    let nothing: link::Starter = std::sync::Arc::new(|| Ok(()));
    let asked = Instant::now();
    link::recover(&mut app, &addr, nothing, &why);
    assert!(
        asked.elapsed() < Duration::from_millis(500),
        "the window froze for {:?} waiting for an engine",
        asked.elapsed()
    );
    let given_up = Instant::now();
    while !app
        .toasts
        .iter()
        .any(|t| t.message.starts_with("no engine could be started"))
    {
        assert!(
            given_up.elapsed() < Duration::from_secs(20),
            "never gave up"
        );
        let _ = link::frame(&mut app);
        tokio::time::sleep(Duration::from_millis(33)).await;
    }
}

// Prompts that only a real model raises. These run against a llama.cpp
// server named by XENCODE_LIVE_LLAMACPP_URL (and XENCODE_LIVE_MODEL, default
// `llamacpp:qwen3-4b`), so they are ignored unless asked for:
//   XENCODE_LIVE_LLAMACPP_URL=http://127.0.0.1:18080 cargo test -p xencode-cli --test engine_cli -- --ignored

fn live_settings() -> tempfile::TempDir {
    let url = std::env::var("XENCODE_LIVE_LLAMACPP_URL")
        .expect("set XENCODE_LIVE_LLAMACPP_URL to a running llama.cpp server");
    let model =
        std::env::var("XENCODE_LIVE_MODEL").unwrap_or_else(|_| "llamacpp:qwen3-4b".to_string());
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({ "default_model": model, "llama_cpp_url": url });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    dir
}

/// Read until an approval prompt arrives; returns its id.
async fn approval_shown(conn: &mut Conn, wait: Duration) -> u64 {
    let start = Instant::now();
    loop {
        assert!(start.elapsed() < wait, "no approval was asked for");
        let Ok(Ok(Some(line))) =
            tokio::time::timeout(Duration::from_millis(500), conn.recv()).await
        else {
            continue;
        };
        if let Ok(EngineMsg::ApprovalRequested { approval }) = proto::decode_engine(&line) {
            return approval.id;
        }
    }
}

/// Everything a window receives over `wait`.
async fn received(conn: &mut Conn, wait: Duration) -> Vec<EngineMsg> {
    let mut got = Vec::new();
    let start = Instant::now();
    while start.elapsed() < wait {
        if let Ok(Ok(Some(line))) =
            tokio::time::timeout(Duration::from_millis(200), conn.recv()).await
        {
            if let Ok(msg) = proto::decode_engine(&line) {
                got.push(msg);
            }
        }
    }
    got
}

#[tokio::test]
#[ignore = "needs a running llama.cpp server: XENCODE_LIVE_LLAMACPP_URL"]
async fn two_windows_racing_for_one_approval() {
    let project = tempfile::tempdir().unwrap();
    let config = live_settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let (mut a, _) = join(&addr, "window a").await;
    let (mut b, _) = join(&addr, "window b").await;
    send(
        &mut a,
        ClientMsg::EnqueueTask {
            text: "Create a file named race.txt containing exactly the word: race".into(),
        },
    )
    .await;
    let wait = Duration::from_secs(180);
    let (id_a, id_b) = tokio::join!(approval_shown(&mut a, wait), approval_shown(&mut b, wait));
    assert_eq!(id_a, id_b, "both windows show the same prompt");
    let answer = |id| ClientMsg::AnswerApproval {
        id,
        answer: proto::WireAnswer::Allow,
    };
    tokio::join!(send(&mut a, answer(id_a)), send(&mut b, answer(id_b)));
    let (got_a, got_b) = tokio::join!(
        received(&mut a, Duration::from_secs(5)),
        received(&mut b, Duration::from_secs(5))
    );
    let winners = |got: &[EngineMsg]| {
        got.iter()
            .filter_map(|m| match m {
                EngineMsg::ApprovalResolved { id, by, .. } if *id == id_a => Some(by.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let (won_a, won_b) = (winners(&got_a), winners(&got_b));
    assert_eq!(won_a.len(), 1, "window a was told once: {got_a:?}");
    assert_eq!(won_a, won_b, "both windows name the same winner");
    let refused = |got: &[EngineMsg]| {
        got.iter().any(|m| {
            matches!(m, EngineMsg::Error { message } if message == "that approval is no longer waiting")
        })
    };
    let loser = if won_a[0] == "window a" {
        &got_b
    } else {
        &got_a
    };
    let winner = if won_a[0] == "window a" {
        &got_a
    } else {
        &got_b
    };
    assert!(refused(loser), "the slower answer is refused: {loser:?}");
    assert!(!refused(winner), "{winner:?}");
    assert_eq!(
        link::answered_elsewhere(
            &EngineMsg::ApprovalResolved {
                id: id_a,
                answer: proto::WireAnswer::Allow,
                by: won_a[0].clone(),
            },
            if won_a[0] == "window a" {
                "window b"
            } else {
                "window a"
            }
        ),
        Some(format!("answered in {}: allow", won_a[0]))
    );
}

#[tokio::test]
async fn a_window_speaking_another_version_is_told_and_let_go() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let engine = Engine::start(project.path(), config.path());
    let addr = engine.address();
    let mut conn = connect(&addr).await.unwrap();
    send(
        &mut conn,
        ClientMsg::Hello {
            version: 999,
            client: "from the future".into(),
        },
    )
    .await;
    match next(&mut conn).await {
        EngineMsg::Error { message } => {
            assert!(
                message.contains('1') && message.contains("999"),
                "{message}"
            )
        }
        other => panic!("expected an error in words, got {other:?}"),
    }
    let end = tokio::time::timeout(Duration::from_secs(5), conn.recv())
        .await
        .expect("the engine ends the connection after saying why");
    assert!(matches!(end, Ok(None) | Err(_)), "{end:?}");
}

#[tokio::test]
#[ignore = "needs a running llama.cpp server: XENCODE_LIVE_LLAMACPP_URL"]
async fn a_wait_with_no_window_ends_by_the_limit() {
    let project = tempfile::tempdir().unwrap();
    let config = live_settings();
    let mut engine = Engine::start_with(project.path(), config.path(), &["--wait-limit", "5"]);
    let addr = engine.address();
    {
        let (mut window, _) = join(&addr, "leaves").await;
        send(
            &mut window,
            ClientMsg::EnqueueTask {
                text: "Create a file named left.txt containing exactly the word: left".into(),
            },
        )
        .await;
        approval_shown(&mut window, Duration::from_secs(180)).await;
        // The window goes away with the approval still waiting.
    }
    let left = Instant::now();
    let status = tokio::task::spawn_blocking(move || {
        let status = engine.exits_within(Duration::from_secs(300));
        (status, engine)
    })
    .await
    .unwrap();
    let (status, _engine) = status;
    let status = status.expect("the engine ended the waits and exited");
    assert!(status.success(), "{status:?}");
    assert!(
        left.elapsed() >= Duration::from_secs(5),
        "nothing ends before the limit: {:?}",
        left.elapsed()
    );
    assert!(
        !project.path().join("left.txt").exists(),
        "a denied approval writes nothing"
    );
}
