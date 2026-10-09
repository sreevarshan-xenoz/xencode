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
        let mut child = Command::new(env!("CARGO_BIN_EXE_xencode"))
            .args(["engine", "--project"])
            .arg(project)
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
