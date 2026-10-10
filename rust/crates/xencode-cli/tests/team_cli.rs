//! TM: worker agents, each in its own worktree. The workers here are the real
//! `xencode acp` (TM-1), with a model address nobody listens on or one that
//! never answers, so no model is needed.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use tokio::sync::mpsc;
use xencode_team_rs::worker::{LaunchSpec, WorkerEvent, WorkerHandle};
use xencode_team_rs::{worktree, WorkerSnapshot, WorkerState};

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A real repository with one commit, in a temp folder of its own so the
/// sibling `-team` folder goes with it.
fn repo() -> (tempfile::TempDir, PathBuf) {
    let outer = tempfile::tempdir().unwrap();
    let root = outer.path().join("proj");
    std::fs::create_dir(&root).unwrap();
    git(&root, &["init", "-q", "-b", "main"]);
    git(&root, &["config", "user.email", "t@example.invalid"]);
    git(&root, &["config", "user.name", "t"]);
    git(&root, &["config", "core.autocrlf", "false"]);
    std::fs::write(root.join("a.txt"), "one\n").unwrap();
    git(&root, &["add", "."]);
    git(&root, &["commit", "-q", "-m", "first"]);
    (outer, root)
}

/// Settings whose model server address has nothing listening on it.
fn unreachable() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "default_model": "llamacpp:none",
            "llama_cpp_url": "http://127.0.0.1:9",
            "llama_cpp_executable": dir.path().join("none").to_string_lossy(),
        })
        .to_string(),
    )
    .unwrap();
    dir
}

/// Settings whose model server accepts and never answers; the listener is
/// returned so it lives as long as the test.
fn stalled() -> (tempfile::TempDir, std::net::TcpListener) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "default_model": "llamacpp:none",
            "llama_cpp_url": format!("http://{}", listener.local_addr().unwrap()),
            "llama_cpp_executable": dir.path().join("none").to_string_lossy(),
        })
        .to_string(),
    )
    .unwrap();
    (dir, listener)
}

/// `xencode acp` as the worker agent, with `config` as its settings.
fn xencode_worker(config: &Path) -> LaunchSpec {
    LaunchSpec {
        program: PathBuf::from(env!("CARGO_BIN_EXE_xencode")),
        args: vec!["acp".to_string()],
        env: vec![(
            "XCODE_CONFIG_DIR".to_string(),
            config.to_string_lossy().to_string(),
        )],
    }
}

fn start(
    root: &Path,
    spec: LaunchSpec,
    task: &str,
) -> (WorkerHandle, mpsc::UnboundedReceiver<WorkerEvent>, PathBuf) {
    let (path, branch) = worktree::create(root, "w1", "main").unwrap();
    let (tx, rx) = mpsc::unbounded_channel();
    let handle = WorkerHandle::start("w1".into(), "xencode", task, spec, path.clone(), branch, tx);
    (handle, rx, path)
}

/// The first snapshot in `state`, within `wait`.
async fn until(
    events: &mut mpsc::UnboundedReceiver<WorkerEvent>,
    state: WorkerState,
    wait: Duration,
) -> WorkerSnapshot {
    tokio::time::timeout(wait, async {
        loop {
            match events.recv().await {
                Some(WorkerEvent::Changed(s)) if s.state == state => return s,
                Some(_) => {}
                None => panic!("the worker's events ended before {state:?}"),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("no {state:?} within {wait:?}"))
}

#[tokio::test]
async fn a_worker_runs_its_task_in_its_own_worktree() {
    let (_outer, root) = repo();
    let config = unreachable();
    let (_handle, mut events, path) = start(&root, xencode_worker(config.path()), "say hi");
    let done = until(&mut events, WorkerState::Done, Duration::from_secs(60)).await;
    assert!(done.answer.contains("127.0.0.1:9"), "{done:?}");
    assert_eq!(done.branch, "xencode/team/w1");
    assert_eq!(done.worktree, path);
    assert!(path.join("a.txt").exists());
}

#[tokio::test]
async fn a_message_continues_the_same_session() {
    let (_outer, root) = repo();
    let config = unreachable();
    let (handle, mut events, _path) = start(&root, xencode_worker(config.path()), "say hi");
    until(&mut events, WorkerState::Done, Duration::from_secs(60)).await;
    handle.message("again").unwrap();
    until(&mut events, WorkerState::Working, Duration::from_secs(10)).await;
    let done = until(&mut events, WorkerState::Done, Duration::from_secs(60)).await;
    assert!(done.answer.contains("127.0.0.1:9"), "{done:?}");
}

#[tokio::test]
async fn a_stopped_worker_says_stopped() {
    let (_outer, root) = repo();
    let (config, _stall) = stalled();
    let (handle, mut events, path) = start(&root, xencode_worker(config.path()), "say hi");
    until(&mut events, WorkerState::Working, Duration::from_secs(30)).await;
    // A message while it works is refused: one turn at a time.
    assert!(handle.message("too soon").is_err());
    handle.stop();
    until(&mut events, WorkerState::Stopped, Duration::from_secs(30)).await;
    assert!(path.exists(), "a stop keeps the worktree");
}

/// Review Focus 1: the agent's process ends without speaking the protocol.
#[tokio::test]
async fn a_worker_whose_agent_ends_is_failed_and_keeps_its_worktree() {
    let (_outer, root) = repo();
    let spec = LaunchSpec {
        program: PathBuf::from(env!("CARGO_BIN_EXE_xencode")),
        args: vec!["--version".to_string()],
        env: Vec::new(),
    };
    let (_handle, mut events, path) = start(&root, spec, "say hi");
    let failed = until(&mut events, WorkerState::Failed, Duration::from_secs(30)).await;
    assert!(failed.error.is_some(), "{failed:?}");
    assert!(path.exists());
}

// ---- TM-1, task 2: the engine hosts the team ----

use xencode_tui_rs::engine::address::Address;
use xencode_tui_rs::engine::link::{self, EngineLink, LinkEvent};
use xencode_tui_rs::engine::proto::{ClientMsg, EngineMsg, TeamRequest};

/// A real `xencode engine` for `project`, killed when dropped, and where it
/// listens.
struct Engine(std::process::Child);

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn engine(project: &Path, config: &Path) -> (Engine, Address) {
    use std::io::BufRead;
    let mut child = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(["engine", "--project"])
        .arg(project)
        .env("XCODE_CONFIG_DIR", config)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let mut line = String::new();
    std::io::BufReader::new(child.stdout.as_mut().unwrap())
        .read_line(&mut line)
        .unwrap();
    let at = line
        .split_once(" listening on ")
        .unwrap_or_else(|| panic!("not a listening line: {line}"))
        .1
        .trim()
        .to_string();
    let addr = if at.starts_with(r"\\.\pipe\") {
        Address::Pipe(at)
    } else {
        Address::Socket(at.into())
    };
    (Engine(child), addr)
}

async fn window(addr: &Address) -> EngineLink {
    let start: link::Starter = std::sync::Arc::new(|| Ok(()));
    link::connect_or_start_as(addr, &*start, "team-test")
        .await
        .unwrap()
        .0
}

/// Send one team request and wait for its reply.
async fn ask(link: &mut EngineLink, req: u64, request: TeamRequest) -> (bool, serde_json::Value) {
    assert!(link.send(&ClientMsg::Team { req, request }));
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match link.next().await {
                LinkEvent::Msg(EngineMsg::TeamReply { req: r, ok, body }) if r == req => {
                    return (ok, body)
                }
                LinkEvent::Msg(_) => {}
                LinkEvent::Lost(why) => panic!("engine lost: {why}"),
            }
        }
    })
    .await
    .expect("a reply")
}

/// The first view in which worker `id` is in `state`.
async fn team_until(link: &mut EngineLink, id: &str, state: WorkerState) -> WorkerSnapshot {
    tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            match link.next().await {
                LinkEvent::Msg(EngineMsg::View { view }) => {
                    if let Some(w) = view
                        .team
                        .iter()
                        .flatten()
                        .find(|w| w.id == id && w.state == state)
                    {
                        return w.clone();
                    }
                }
                LinkEvent::Msg(_) => {}
                LinkEvent::Lost(why) => panic!("engine lost: {why}"),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{id} never reached {state:?}"))
}

#[tokio::test]
async fn the_engine_starts_a_worker_and_every_window_sees_it() {
    let (_outer, root) = repo();
    let config = unreachable();
    let (_engine, addr) = engine(&root, config.path());
    let mut link = window(&addr).await;
    let (ok, body) = ask(
        &mut link,
        1,
        TeamRequest::Start {
            agent: "xencode".into(),
            task: "say hi".into(),
            base: None,
        },
    )
    .await;
    assert!(ok, "{body}");
    assert_eq!(body["id"], "w1");
    let done = team_until(&mut link, "w1", WorkerState::Done).await;
    assert!(done.answer.contains("127.0.0.1:9"), "{done:?}");
    assert_eq!(done.branch, "xencode/team/w1");

    let (ok, body) = ask(
        &mut link,
        2,
        TeamRequest::Status {
            id: Some("w9".into()),
        },
    )
    .await;
    assert!(!ok);
    assert!(body.to_string().contains("no worker w9"), "{body}");

    let (ok, body) = ask(&mut link, 3, TeamRequest::Status { id: None }).await;
    assert!(ok, "{body}");
    assert_eq!(body[0]["id"], "w1");
}

#[tokio::test]
async fn an_unknown_agent_is_refused_in_words() {
    let (_outer, root) = repo();
    let config = unreachable();
    let (_engine, addr) = engine(&root, config.path());
    let mut link = window(&addr).await;
    let (ok, body) = ask(
        &mut link,
        1,
        TeamRequest::Start {
            agent: "nobody".into(),
            task: "x".into(),
            base: None,
        },
    )
    .await;
    assert!(!ok);
    assert!(body.to_string().contains("nobody"), "{body}");
}

// ---- TM-2: the lead's tools on `xencode mcp serve --team` ----

/// The real `xencode mcp serve --team`, spoken to as an MCP client would:
/// JSON-RPC lines over its standard input and output.
struct Mcp {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    lines: std::sync::mpsc::Receiver<String>,
    next: u64,
}

impl Drop for Mcp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Mcp {
    fn start(workspace: &Path, config: &Path) -> Mcp {
        use std::io::BufRead;
        let mut child = Command::new(env!("CARGO_BIN_EXE_xencode"))
            .args(["mcp", "serve", "--team", "--workspace"])
            .arg(workspace)
            .env("XCODE_CONFIG_DIR", config)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, lines) = std::sync::mpsc::channel();
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
        let mut mcp = Mcp {
            child,
            stdin,
            lines,
            next: 0,
        };
        mcp.request(
            "initialize",
            serde_json::json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "team-test", "version": "1"}
            }),
        );
        mcp.send(serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}));
        mcp
    }

    fn send(&mut self, message: serde_json::Value) {
        use std::io::Write;
        let mut line = message.to_string();
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).unwrap();
        self.stdin.flush().unwrap();
    }

    fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        self.next += 1;
        let id = self.next;
        self.send(
            serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}),
        );
        loop {
            let line = self
                .lines
                .recv_timeout(Duration::from_secs(60))
                .expect("the server answered");
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            if value["id"].as_u64() == Some(id) {
                assert!(value.get("error").is_none(), "{value}");
                return value["result"].clone();
            }
        }
    }

    /// Call a tool; its text and whether it was an error result.
    fn call(&mut self, name: &str, arguments: serde_json::Value) -> (String, bool) {
        let result = self.request(
            "tools/call",
            serde_json::json!({"name": name, "arguments": arguments}),
        );
        let text = result["content"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        (text, result["isError"].as_bool().unwrap_or(false))
    }
}

#[test]
fn the_lead_lists_the_team_tools_and_runs_a_worker_through_them() {
    let (_outer, root) = repo();
    let config = unreachable();
    let mut mcp = Mcp::start(&root, config.path());

    let tools = mcp.request("tools/list", serde_json::json!({}));
    let names: Vec<String> = tools["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();
    for want in [
        "team_agents",
        "team_start",
        "team_status",
        "team_result",
        "team_message",
        "team_stop",
        "team_merge",
    ] {
        assert!(names.iter().any(|n| n == want), "{want} missing: {names:?}");
    }

    let (text, error) = mcp.call(
        "team_start",
        serde_json::json!({"agent": "xencode", "task": "say hi"}),
    );
    assert!(!error, "{text}");
    assert!(text.contains("w1"), "{text}");

    let start = std::time::Instant::now();
    loop {
        let (text, error) = mcp.call("team_status", serde_json::json!({"id": "w1"}));
        assert!(!error, "{text}");
        if text.contains("\"done\"") {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(90),
            "never done: {text}"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    let (text, error) = mcp.call("team_result", serde_json::json!({"id": "w1"}));
    assert!(!error, "{text}");
    assert!(text.contains("127.0.0.1:9"), "the answer came back: {text}");
}

/// Review Focus 4.
#[test]
fn an_unknown_worker_is_refused_in_words() {
    let (_outer, root) = repo();
    let config = unreachable();
    let mut mcp = Mcp::start(&root, config.path());
    let (text, error) = mcp.call("team_status", serde_json::json!({"id": "nope"}));
    assert!(error);
    assert!(text.contains("no worker nope"), "{text}");
    let (text, error) = mcp.call("team_start", serde_json::json!({"task": "no agent named"}));
    assert!(error);
    assert!(text.contains("agent"), "{text}");
}
