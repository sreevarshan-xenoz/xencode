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
        on_plan: false,
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
                Some(WorkerEvent::Changed(s)) if s.state == state => return *s,
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
        on_plan: false,
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
    engine_with(project, config, true)
}

/// A real engine for `project`; `without_keys` keeps the person's vendor
/// keys from reaching it.
fn engine_with(project: &Path, config: &Path, without_keys: bool) -> (Engine, Address) {
    use std::io::BufRead;
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_xencode"));
    cmd.args(["engine", "--project"])
        .arg(project)
        .env("XCODE_CONFIG_DIR", config)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());
    if without_keys {
        for var in [
            "ANTHROPIC_API_KEY",
            "CODEX_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
        ] {
            cmd.env_remove(var);
        }
    }
    let mut child = cmd.spawn().unwrap();
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

// ---- TM-3: the checked merge through the lead's tool ----

fn wait_done(mcp: &mut Mcp, id: &str) {
    let start = std::time::Instant::now();
    loop {
        let (text, error) = mcp.call("team_status", serde_json::json!({ "id": id }));
        assert!(!error, "{text}");
        if text.contains("\"done\"") {
            return;
        }
        assert!(
            start.elapsed() < Duration::from_secs(90),
            "never done: {text}"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
}

#[test]
fn the_lead_merges_a_workers_change_when_the_checks_pass() {
    let (_outer, root) = repo();
    std::fs::create_dir_all(root.join(".xencode")).unwrap();
    std::fs::write(
        root.join(".xencode").join("team.toml"),
        "checks = [\"git --version\"]\n",
    )
    .unwrap();
    // The settings file is the person's own; commit it so the working copy
    // is clean for the merge.
    std::fs::write(
        root.join(".gitignore"),
        ".xencode/cache/\n.xencode/*.json*\n",
    )
    .unwrap();
    git(&root, &["add", "."]);
    git(&root, &["commit", "-q", "-m", "team settings"]);
    let config = unreachable();
    let mut mcp = Mcp::start(&root, config.path());
    let (text, error) = mcp.call(
        "team_start",
        serde_json::json!({"agent": "xencode", "task": "say hi"}),
    );
    assert!(!error, "{text}");
    wait_done(&mut mcp, "w1");
    // What a worker would have done in its worktree.
    let worktree = root.parent().unwrap().join("proj-team").join("w1");
    std::fs::write(worktree.join("from-worker.txt"), "made by w1\n").unwrap();

    let (text, error) = mcp.call("team_merge", serde_json::json!({"id": "w1"}));
    assert!(!error, "{text}");
    assert!(text.contains("landed"), "{text}");
    assert_eq!(
        std::fs::read_to_string(root.join("from-worker.txt")).unwrap(),
        "made by w1\n"
    );
    assert!(!worktree.exists(), "a landed worker's worktree is removed");
}

#[test]
fn red_checks_keep_the_workers_change_off_the_base_branch() {
    let (_outer, root) = repo();
    std::fs::create_dir_all(root.join(".xencode")).unwrap();
    std::fs::write(
        root.join(".xencode").join("team.toml"),
        "checks = [\"git no-such-command\"]\n",
    )
    .unwrap();
    std::fs::write(
        root.join(".gitignore"),
        ".xencode/cache/\n.xencode/*.json*\n",
    )
    .unwrap();
    git(&root, &["add", "."]);
    git(&root, &["commit", "-q", "-m", "team settings"]);
    let config = unreachable();
    let mut mcp = Mcp::start(&root, config.path());
    mcp.call(
        "team_start",
        serde_json::json!({"agent": "xencode", "task": "say hi"}),
    );
    wait_done(&mut mcp, "w1");
    let worktree = root.parent().unwrap().join("proj-team").join("w1");
    std::fs::write(worktree.join("from-worker.txt"), "made by w1\n").unwrap();
    let (text, error) = mcp.call("team_merge", serde_json::json!({"id": "w1"}));
    assert!(error, "{text}");
    assert!(text.contains("checks_failed"), "{text}");
    assert!(!root.join("from-worker.txt").exists());
    assert!(worktree.exists(), "kept for another try");
}

// ---- TM-4: the Team section of the worker panel ----

async fn frames_until(
    app: &mut xencode_tui_rs::app::App<'_>,
    what: &str,
    done: impl Fn(&xencode_tui_rs::app::App<'_>) -> bool,
) {
    let start = std::time::Instant::now();
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
        assert!(start.elapsed() < Duration::from_secs(60), "never: {what}");
        tokio::time::sleep(Duration::from_millis(33)).await;
    }
}

fn worker_is(app: &xencode_tui_rs::app::App<'_>, id: &str, state: WorkerState) -> bool {
    app.team_view.iter().any(|w| w.id == id && w.state == state)
}

#[tokio::test]
async fn a_window_shows_the_team_and_s_stops_the_selected_worker() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    let (_outer, root) = repo();
    let config = unreachable();
    let (_engine, addr) = engine(&root, config.path());
    let (linked, view) = link::open(&addr, "terminal").await.unwrap();
    let mut app = xencode_tui_rs::app::App::for_tests();
    link::become_window(&mut app, linked, view);
    let (tx, _rx) = mpsc::unbounded_channel();

    app.team_command(
        TeamRequest::Start {
            agent: "xencode".into(),
            task: "say hi".into(),
            base: None,
        },
        &tx,
    );
    frames_until(&mut app, "w1 done", |a| {
        worker_is(a, "w1", WorkerState::Done)
    })
    .await;

    app.refresh_worker_panel();
    let at = app
        .workers_rows
        .iter()
        .position(|r| xencode_tui_rs::worker_panel::team_worker(r) == Some("w1"))
        .unwrap_or_else(|| panic!("no row for w1: {:#?}", app.workers_rows));
    assert!(
        app.workers_rows[at].line.contains("xencode · done"),
        "{}",
        app.workers_rows[at].line
    );
    app.focus = xencode_tui_rs::app::FocusArea::WorkerPanel;
    app.workers_selected = at;
    xencode_tui_rs::keymap::handle_key(
        &mut app,
        KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE),
        &tx,
    );
    frames_until(&mut app, "w1 stopped", |a| {
        worker_is(a, "w1", WorkerState::Stopped)
    })
    .await;
}

// ---- TM-5: the outside agents ----

/// `unreachable()` settings with another vendor's agents allowed.
fn outside_allowed() -> tempfile::TempDir {
    let dir = unreachable();
    let path = dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    config["allow_external_workers"] = serde_json::json!(true);
    std::fs::write(&path, config.to_string()).unwrap();
    dir
}

#[tokio::test]
async fn local_only_lists_the_outside_agents_and_refuses_to_start_one() {
    let (_outer, root) = repo();
    let config = unreachable();
    let (_engine, addr) = engine(&root, config.path());
    let mut link = window(&addr).await;
    let (ok, body) = ask(&mut link, 1, TeamRequest::Agents).await;
    assert!(ok, "{body}");
    let names: Vec<&str> = body
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["xencode", "claude-code", "codex", "gemini", "antigravity"]
    );
    assert_eq!(body[0]["available"], true);
    assert_eq!(body[1]["available"], false);
    assert!(
        body[1]["refused"].as_str().unwrap().contains("Local Only"),
        "{body}"
    );

    let (ok, body) = ask(
        &mut link,
        2,
        TeamRequest::Start {
            agent: "claude-code".into(),
            task: "say hi".into(),
            base: None,
        },
    )
    .await;
    assert!(!ok);
    assert!(
        body.to_string().contains("allow_external_workers"),
        "{body}"
    );
}

#[tokio::test]
async fn an_outside_agent_without_a_key_or_login_is_refused_with_the_fix() {
    let (_outer, root) = repo();
    let config = outside_allowed();
    let (_engine, addr) = engine(&root, config.path());
    let mut link = window(&addr).await;
    let (ok, body) = ask(
        &mut link,
        1,
        TeamRequest::Start {
            agent: "codex".into(),
            task: "say hi".into(),
            base: None,
        },
    )
    .await;
    assert!(!ok, "{body}");
    let said = body.as_str().unwrap();
    // Which of the two this machine gets depends on whether Node.js is
    // installed here; both say what to do.
    assert!(
        said.contains("cannot sign in: set CODEX_API_KEY or OPENAI_API_KEY")
            || said.contains("cannot start: `npx` is not on PATH"),
        "{said}"
    );
    assert!(
        !root.parent().unwrap().join("proj-team").join("w1").exists(),
        "a refused start makes no worktree"
    );
}

#[test]
fn a_login_is_turned_on_only_by_a_typed_yes() {
    use std::io::Write;
    let config = tempfile::tempdir().unwrap();
    let optin = |answer: &str| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_xencode"))
            .args(["team", "login-optin", "gemini"])
            .env("XCODE_CONFIG_DIR", config.path())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(answer.as_bytes())
            .unwrap();
        child.wait_with_output().unwrap()
    };
    let no = optin("no\n");
    assert!(!no.status.success());
    let shown = String::from_utf8_lossy(&no.stdout);
    assert!(
        shown.contains("geminicli.com/docs/resources/tos-privacy"),
        "{shown}"
    );
    assert!(!config.path().join("team-optins.json").exists());

    let yes = optin("yes\n");
    assert!(
        yes.status.success(),
        "{}",
        String::from_utf8_lossy(&yes.stderr)
    );
    let kept = std::fs::read_to_string(config.path().join("team-optins.json")).unwrap();
    assert!(kept.contains("gemini"), "{kept}");

    let unknown = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(["team", "login-optin", "xencode"])
        .env("XCODE_CONFIG_DIR", config.path())
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(!unknown.status.success());
}

#[tokio::test]
async fn team_clean_removes_a_stopped_workers_worktree() {
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
    team_until(&mut link, "w1", WorkerState::Done).await;
    let (ok, _) = ask(&mut link, 2, TeamRequest::Stop { id: "w1".into() }).await;
    assert!(ok);
    team_until(&mut link, "w1", WorkerState::Stopped).await;
    let worktree = root.parent().unwrap().join("proj-team").join("w1");
    assert!(worktree.exists());

    // The person's command, from the project folder, through the same engine.
    let start = std::time::Instant::now();
    loop {
        let out = Command::new(env!("CARGO_BIN_EXE_xencode"))
            .args(["team", "clean"])
            .current_dir(&root)
            .env("XCODE_CONFIG_DIR", config.path())
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap();
        let said = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(
            out.status.success(),
            "{said}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        if said.contains("removed: w1") {
            break;
        }
        // On Windows the stopped worker's own engine can hold the folder for
        // a few seconds after it ends.
        assert!(start.elapsed() < Duration::from_secs(40), "{said}");
        std::thread::sleep(Duration::from_secs(2));
    }
    assert!(!worktree.exists());
}

/// Each outside agent, live: it needs that vendor's key and costs money, so
/// it runs only when asked (`cargo test -- --ignored <name>`).
async fn an_outside_agent_answers(agent: &str) {
    let (_outer, root) = repo();
    let config = outside_allowed();
    let (_engine, addr) = engine_keeping_keys(&root, config.path());
    let mut link = window(&addr).await;
    let (ok, body) = ask(
        &mut link,
        1,
        TeamRequest::Start {
            agent: agent.into(),
            task: "Reply with the single word: ready".into(),
            base: None,
        },
    )
    .await;
    assert!(ok, "{body}");
    let id = body["id"].as_str().unwrap().to_string();
    let done = team_until(&mut link, &id, WorkerState::Done).await;
    assert!(!done.answer.trim().is_empty(), "{done:?}");
}

/// As `engine`, but the person's vendor keys reach it.
fn engine_keeping_keys(project: &Path, config: &Path) -> (Engine, Address) {
    engine_with(project, config, false)
}

#[tokio::test]
#[ignore = "needs ANTHROPIC_API_KEY and Node.js, and costs money"]
async fn live_claude_code_answers_as_a_worker() {
    an_outside_agent_answers("claude-code").await;
}

#[tokio::test]
#[ignore = "needs CODEX_API_KEY or OPENAI_API_KEY and Node.js, and costs money"]
async fn live_codex_answers_as_a_worker() {
    an_outside_agent_answers("codex").await;
}

#[tokio::test]
#[ignore = "needs GEMINI_API_KEY and the gemini CLI, and costs money"]
async fn live_gemini_answers_as_a_worker() {
    an_outside_agent_answers("gemini").await;
}

#[tokio::test]
#[ignore = "needs GEMINI_API_KEY, Antigravity set to Gemini, and costs money"]
async fn live_antigravity_answers_as_a_worker() {
    an_outside_agent_answers("antigravity").await;
}

/// Security review: the wrapper an outside agent starts through runs the
/// program in the folder it is given, with the named variables gone.
#[test]
fn team_exec_runs_in_its_folder_without_the_hidden_keys() {
    let dir = tempfile::tempdir().unwrap();
    let printer: Vec<&str> = if cfg!(windows) {
        vec![
            "cmd",
            "/C",
            "cd & echo [%XENCODE_TEST_HIDDEN%] [%XENCODE_TEST_KEPT%]",
        ]
    } else {
        vec![
            "sh",
            "-c",
            "pwd; echo \"[${XENCODE_TEST_HIDDEN-gone}] [${XENCODE_TEST_KEPT}]\"",
        ]
    };
    let out = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(["team", "exec", "--cwd"])
        .arg(dir.path())
        .args(["--unset", "XENCODE_TEST_HIDDEN", "--"])
        .args(&printer)
        .env("XENCODE_TEST_HIDDEN", "FAKE-NOT-A-REAL-KEY")
        .env("XENCODE_TEST_KEPT", "kept")
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "{said}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!said.contains("FAKE-NOT-A-REAL-KEY"), "{said}");
    assert!(said.contains("[kept]"), "{said}");
    let name = dir
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    assert!(said.contains(&name), "ran in its folder: {said}");
}
