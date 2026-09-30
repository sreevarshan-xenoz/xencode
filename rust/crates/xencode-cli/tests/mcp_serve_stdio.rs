//! `xencode mcp serve` over a real pipe (M-5).
//!
//! These tests start the actual binary and speak JSON-RPC to its standard
//! input the way any MCP client does, then read its standard output. Nothing is
//! stubbed: the workspace is a directory on disk, the read returns its bytes,
//! and a refused write leaves no file behind — which is checked, not assumed.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// The binary under test, built by cargo for this test target.
fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_xencode"))
}

/// One launched server, with the JSON-RPC plumbing around it.
struct Server {
    child: Child,
    stdin: std::process::ChildStdin,
    /// Lines as the server's own reader thread delivers them, so a call never
    /// has to borrow the pipe it is also writing to.
    incoming: mpsc::Receiver<String>,
    /// What the server printed for the person who started it. It has to be read
    /// continuously or a long notice could fill the pipe and stop the server.
    notices: mpsc::Receiver<String>,
    next_id: u64,
}

impl Server {
    /// Start `xencode mcp serve` on a workspace, then complete the handshake.
    fn start(workspace: &Path, allow: &[&str]) -> Server {
        let mut command = Command::new(binary());
        command
            .arg("mcp")
            .arg("serve")
            .arg("--workspace")
            .arg(workspace);
        for name in allow {
            command.arg("--allow").arg(name);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // The server's own notices go to stderr; keeping them off stdout is
            // what makes stdout parseable.
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to start xencode mcp serve");
        let stdin = child.stdin.take().expect("stdin pipe");
        let stdout = child.stdout.take().expect("stdout pipe");
        let (sender, incoming) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break, // EOF: the server closed the pipe.
                    Ok(_) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let stderr = child.stderr.take().expect("stderr pipe");
        let (notice_sender, notices) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if notice_sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        let mut server = Server {
            child,
            stdin,
            incoming,
            notices,
            next_id: 0,
        };
        // Every client does this before it asks for anything.
        let initialized = server.request(
            "initialize",
            serde_json::json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": { "name": "integration-test", "version": "1" }
            }),
        );
        assert_eq!(initialized["serverInfo"]["name"], "xencode");
        assert!(
            initialized["capabilities"]["tools"].is_object(),
            "a server that lists tools must say so: {initialized}"
        );
        // The handshake tells the caller which directory it is bound to, so a
        // client cannot be surprised later about where a write would land.
        assert!(
            initialized["instructions"]
                .as_str()
                .is_some_and(|text| text.contains(&workspace.display().to_string())),
            "the handshake said nothing about its workspace: {initialized}"
        );
        server.notify("notifications/initialized", serde_json::json!({}));
        server
    }

    fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        self.next_id += 1;
        let id = self.next_id;
        self.write(serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params
        }));
        loop {
            let line = self.read_line();
            let value: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|error| panic!("server sent {line:?}: {error}"));
            // A notification or a progress message can arrive before the
            // answer; the reply is the one carrying this request's id.
            if value.get("id").and_then(|v| v.as_u64()) == Some(id) {
                if let Some(error) = value.get("error") {
                    panic!("method {method} failed: {error}");
                }
                return value["result"].clone();
            }
        }
    }

    fn notify(&mut self, method: &str, params: serde_json::Value) {
        self.write(serde_json::json!({ "jsonrpc": "2.0", "method": method, "params": params }));
    }

    fn write(&mut self, message: serde_json::Value) {
        // One JSON value per line: that is the stdio framing, and the server
        // would sit waiting forever on a message without the newline.
        let mut line = message.to_string();
        line.push('\n');
        self.stdin
            .write_all(line.as_bytes())
            .expect("write to server");
        self.stdin.flush().expect("flush to server");
    }

    fn read_line(&mut self) -> String {
        // Answers come from a separate process, so bound the wait and say so
        // rather than hanging the run or failing with an empty response.
        self.incoming
            .recv_timeout(Duration::from_secs(20))
            .unwrap_or_else(|_| panic!("the server sent nothing within 20s"))
    }

    /// Everything the server has printed for the operator so far. The notice is
    /// written before the first reply, so this normally returns at once; the
    /// deadline is only what stops the run hanging if it never arrives.
    fn notices(&mut self) -> String {
        let mut text = String::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            match self.notices.recv_timeout(Duration::from_millis(100)) {
                Ok(line) => text.push_str(&line),
                Err(mpsc::RecvTimeoutError::Timeout) if !text.is_empty() => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(_) => {}
            }
        }
        text
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A workspace with real files in it, unique per test so runs cannot collide.
fn workspace(label: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("xencode-mcp-serve-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("hello.txt"), "one\ntwo\nthree\n").unwrap();
    std::fs::write(
        dir.join("src/main.rs"),
        "fn main() {\n    println!(\"hi\");\n}\n",
    )
    .unwrap();
    dir
}

/// The text a tool call produced, and whether the server marked it failed.
fn call(server: &mut Server, name: &str, arguments: serde_json::Value) -> (String, bool) {
    let result = server.request(
        "tools/call",
        serde_json::json!({"name": name, "arguments": arguments}),
    );
    let text = result["content"]
        .as_array()
        .map(|blocks| {
            blocks
                .iter()
                .filter(|b| b["type"].as_str() == Some("text"))
                .filter_map(|b| b["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    (text, result["isError"].as_bool().unwrap_or(false))
}

#[test]
fn a_client_can_list_the_tools_xencode_publishes() {
    let dir = workspace("list");
    let mut server = Server::start(&dir, &[]);
    let listed = server.request("tools/list", serde_json::json!({}));
    let names: Vec<String> = listed["tools"]
        .as_array()
        .expect("tools must be an array")
        .iter()
        .map(|tool| tool["name"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        names,
        [
            "read_file",
            "list_dir",
            "search_files",
            "write_file",
            "edit_file",
            "run_command"
        ]
    );
    // A schema is what makes the tool callable by a client that has never seen
    // xencode, so each one has to arrive as an object schema. `inputSchema` is
    // the spelling on the wire; only the SDK's Rust type uses snake_case.
    for tool in listed["tools"].as_array().unwrap() {
        assert_eq!(tool["inputSchema"]["type"], "object", "{tool}");
        assert!(
            tool["description"]
                .as_str()
                .is_some_and(|text| !text.trim().is_empty()),
            "{tool}"
        );
    }
    // The read-only hint is what a client uses to decide whether a call is safe
    // to make on its own initiative, so it has to arrive per tool.
    let hint = |name: &str| {
        listed["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == name)
            .and_then(|tool| tool["annotations"]["readOnlyHint"].as_bool())
    };
    assert_eq!(hint("read_file"), Some(true));
    assert_eq!(hint("write_file"), Some(false));
    assert_eq!(hint("run_command"), Some(false));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn reads_work_and_writes_are_refused_with_the_flag_that_would_allow_them() {
    let dir = workspace("default");
    let mut server = Server::start(&dir, &[]);

    let (text, failed) = call(
        &mut server,
        "read_file",
        serde_json::json!({"path": "hello.txt"}),
    );
    assert!(!failed, "a read must not be reported as an error: {text}");
    assert_eq!(text, "1\tone\n2\ttwo\n3\tthree");

    let (text, failed) = call(
        &mut server,
        "search_files",
        serde_json::json!({"pattern": "println"}),
    );
    assert!(!failed, "{text}");
    assert!(text.contains("src/main.rs"), "{text}");

    let (text, failed) = call(
        &mut server,
        "write_file",
        serde_json::json!({"path": "written.txt", "content": "hi\n"}),
    );
    assert!(failed, "a write must not be reported as a success: {text}");
    assert!(text.contains("--allow write_file"), "{text}");
    assert!(
        !dir.join("written.txt").exists(),
        "a refused write left a file behind"
    );

    let (text, failed) = call(
        &mut server,
        "run_command",
        serde_json::json!({"command": "touch ran-by-client"}),
    );
    assert!(failed, "{text}");
    assert!(text.contains("--allow run_command"), "{text}");
    assert!(
        !dir.join("ran-by-client").exists(),
        "a refused command still ran"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_allowed_tool_writes_the_file_and_a_path_outside_stays_refused() {
    let dir = workspace("allow");
    let mut server = Server::start(&dir, &["write_file"]);

    let (text, failed) = call(
        &mut server,
        "write_file",
        serde_json::json!({"path": "written.txt", "content": "hi there\n"}),
    );
    assert!(!failed, "{text}");
    assert_eq!(
        std::fs::read_to_string(dir.join("written.txt")).unwrap(),
        "hi there\n"
    );

    // The permission does not move the boundary.
    let (text, failed) = call(
        &mut server,
        "write_file",
        serde_json::json!({"path": "../escape.txt", "content": "x"}),
    );
    assert!(failed, "{text}");
    assert!(text.contains("outside this workspace"), "{text}");
    assert!(
        !dir.join("../escape.txt").exists(),
        "an allowed tool wrote outside the workspace"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The honest limit of `--workspace`: the boundary is checked on the `path` and
/// `cwd` a call carries, and a command the caller was *allowed* to run is
/// executed exactly as written — so it can name files anywhere this user's shell
/// can reach. An operator who grants `run_command` is handing over a shell with
/// nobody on the pipe to approve what it does next, and the launch says that.
#[test]
fn a_permitted_command_reaches_where_it_says_and_the_launch_warned_of_it() {
    let dir = workspace("shell");
    let outside =
        std::env::temp_dir().join(format!("xencode-mcp-serve-outside-{}", std::process::id()));
    let _ = std::fs::remove_file(&outside);
    let mut server = Server::start(&dir, &["run_command"]);
    let notice = server.notices();

    let (text, failed) = call(
        &mut server,
        "run_command",
        serde_json::json!({"command": format!("echo proof > {}", outside.display())}),
    );
    assert!(!failed, "{text}");
    assert!(
        outside.exists(),
        "the permitted command did not reach outside the workspace"
    );
    assert!(notice.contains("warning"), "{notice}");
    assert!(notice.contains("run_command"), "{notice}");

    let _ = std::fs::remove_file(&outside);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_launch_naming_no_published_tool_is_refused_rather_than_ignored() {
    let dir = workspace("typo");
    // `--allow write-files` is a typo for a tool that exists. Silently starting
    // read-only would leave the operator believing the tool was permitted.
    let output = Command::new(binary())
        .arg("mcp")
        .arg("serve")
        .arg("--workspace")
        .arg(&dir)
        .arg("--allow")
        .arg("write-files")
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stdin(std::process::Stdio::piped())
        .output()
        .expect("failed to run xencode mcp serve");
    let text = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(!output.status.success(), "{text}");
    assert!(text.contains("no tool xencode publishes"), "{text}");
    assert!(text.contains("read_file"), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}
