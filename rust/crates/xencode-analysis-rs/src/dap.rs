//! Stop a test at a line and read its locals, through a real debugger (`CI-7`).
//!
//! The debugger is spoken to over the Debug Adapter Protocol: `lldb-dap` when
//! it is installed, otherwise GDB's own adapter (`gdb -i=dap`, GDB 14 and
//! later). Both are programs somebody else maintains; this module is only the
//! client — the framing, the order of requests the protocol requires, and a
//! deadline on every wait.
//!
//! A debugger is a stateful process and an agent's tool calls are not, which is
//! the trap the plan item names. So one call is one whole session: build the
//! test binary, start the adapter, set the breakpoint, run to it, read the
//! frame's variables, and end the session — the debuggee and the adapter are
//! both gone before the answer is returned. Nothing is left running between
//! calls for a later call to find in a state it did not expect.

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

/// What to debug.
#[derive(Debug, Clone)]
pub struct DebugRequest {
    /// The Cargo workspace (or package) directory.
    pub workspace: PathBuf,
    /// `-p` for `cargo test`, when the workspace has several packages.
    pub package: Option<String>,
    /// The test's full name as `cargo test -- --list` prints it.
    pub test: String,
    /// The source file, relative to `workspace` or absolute.
    pub file: String,
    /// One-based line to stop at.
    pub line: u32,
    /// Longest the build plus the session may take.
    pub timeout: Duration,
}

/// One variable as the debugger rendered it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable {
    pub name: String,
    pub value: String,
    pub ty: Option<String>,
    /// `Locals` or `Arguments`, as the adapter named the scope.
    pub scope: String,
}

/// What one session saw.
#[derive(Debug, Clone)]
pub struct DebugOutcome {
    /// The adapter used, with its version line.
    pub adapter: String,
    /// The test binary that was run.
    pub binary: PathBuf,
    /// Where it stopped — `function at file:line` — if it stopped at all.
    pub stopped_at: Option<String>,
    /// Why it stopped (`breakpoint`), as the adapter said.
    pub reason: Option<String>,
    pub variables: Vec<Variable>,
    /// The exit code when the program ran to the end without stopping.
    pub exit_code: Option<i64>,
}

#[derive(Debug, thiserror::Error)]
pub enum DapError {
    #[error("no debug adapter here: install lldb-dap (LLVM) or GDB 14 or later; {0}")]
    NoAdapter(String),
    #[error("the test binary did not build: {0}")]
    Build(String),
    #[error("no test binary of this workspace lists a test named `{0}`")]
    NoSuchTest(String),
    #[error("the debug adapter refused `{command}`: {message}")]
    Refused { command: String, message: String },
    #[error("the debugger did not answer within {0} seconds")]
    TimedOut(u64),
    #[error("the debug adapter stopped talking: {0}")]
    Lost(String),
}

/// The adapter to start, and how it introduced itself.
fn find_adapter() -> Result<(Vec<String>, String), DapError> {
    if let Ok(out) = Command::new("lldb-dap").arg("--version").output() {
        if out.status.success() {
            let version = first_line(&out.stdout);
            return Ok((vec!["lldb-dap".into()], version));
        }
    }
    let out = Command::new("gdb")
        .arg("--version")
        .output()
        .map_err(|e| DapError::NoAdapter(format!("gdb did not start: {e}")))?;
    let version = first_line(&out.stdout);
    let major = version
        .split_whitespace()
        .rev()
        .find_map(|word| word.split('.').next()?.parse::<u32>().ok())
        .unwrap_or(0);
    if major < 14 {
        return Err(DapError::NoAdapter(format!(
            "`{version}` predates GDB's debug adapter, which arrived in GDB 14"
        )));
    }
    Ok((vec!["gdb".into(), "-q".into(), "-i=dap".into()], version))
}

fn first_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Build the test binaries and return the one that lists `test`.
fn test_binary(req: &DebugRequest) -> Result<PathBuf, DapError> {
    let mut cmd = Command::new("cargo");
    cmd.args(["test", "--no-run", "--message-format=json"])
        .current_dir(&req.workspace);
    if let Some(package) = &req.package {
        cmd.args(["-p", package]);
    }
    let out = cmd
        .output()
        .map_err(|e| DapError::Build(format!("cargo did not start: {e}")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(8).collect();
        return Err(DapError::Build(
            tail.into_iter().rev().collect::<Vec<_>>().join("\n"),
        ));
    }
    let mut binaries = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Ok(msg) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if msg["reason"] == "compiler-artifact" && msg["profile"]["test"] == true {
            if let Some(exe) = msg["executable"].as_str() {
                binaries.push(PathBuf::from(exe));
            }
        }
    }
    for binary in binaries {
        let listed = Command::new(&binary)
            .args(["--list", "--format=terse"])
            .current_dir(&req.workspace)
            .output();
        let Ok(listed) = listed else { continue };
        let wanted = format!("{}: test", req.test);
        if String::from_utf8_lossy(&listed.stdout)
            .lines()
            .any(|l| l.trim() == wanted)
        {
            return Ok(binary);
        }
    }
    Err(DapError::NoSuchTest(req.test.clone()))
}

/// One debug adapter, spoken to over its stdin and stdout.
struct Session {
    child: Child,
    stdin: ChildStdin,
    incoming: Receiver<Value>,
    /// Messages read while waiting for something else, kept in arrival order.
    pending: VecDeque<Value>,
    seq: i64,
    deadline: Instant,
    timeout: Duration,
}

impl Session {
    fn start(argv: &[String], cwd: &Path, timeout: Duration) -> Result<Session, DapError> {
        let mut child = Command::new(&argv[0])
            .args(&argv[1..])
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| DapError::NoAdapter(format!("{} did not start: {e}", argv[0])))?;
        let stdin = child.stdin.take().expect("piped");
        let stdout = child.stdout.take().expect("piped");
        let (tx, incoming) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            while let Some(message) = read_message(&mut reader) {
                if tx.send(message).is_err() {
                    return;
                }
            }
        });
        Ok(Session {
            child,
            stdin,
            incoming,
            pending: VecDeque::new(),
            seq: 0,
            deadline: Instant::now() + timeout,
            timeout,
        })
    }

    fn send(&mut self, command: &str, arguments: Value) -> Result<i64, DapError> {
        self.seq += 1;
        let body = json!({
            "seq": self.seq,
            "type": "request",
            "command": command,
            "arguments": arguments,
        })
        .to_string();
        write!(self.stdin, "Content-Length: {}\r\n\r\n{body}", body.len())
            .and_then(|_| self.stdin.flush())
            .map_err(|e| DapError::Lost(format!("writing `{command}`: {e}")))?;
        Ok(self.seq)
    }

    /// The next message matching `wanted`, keeping the others for later waits.
    fn wait_for(&mut self, wanted: impl Fn(&Value) -> bool) -> Result<Value, DapError> {
        if let Some(i) = self.pending.iter().position(&wanted) {
            return Ok(self.pending.remove(i).expect("found"));
        }
        loop {
            let left = self.deadline.saturating_duration_since(Instant::now());
            match self.incoming.recv_timeout(left) {
                Ok(message) if wanted(&message) => return Ok(message),
                Ok(message) => self.pending.push_back(message),
                Err(RecvTimeoutError::Timeout) => {
                    return Err(DapError::TimedOut(self.timeout.as_secs()))
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(DapError::Lost("the adapter closed its output".into()))
                }
            }
        }
    }

    /// Send a request and wait for its response, refusing a failed one.
    fn request(&mut self, command: &str, arguments: Value) -> Result<Value, DapError> {
        let seq = self.send(command, arguments)?;
        let response = self.wait_for(|m| m["type"] == "response" && m["request_seq"] == seq)?;
        if response["success"] != true {
            return Err(DapError::Refused {
                command: command.to_string(),
                message: response["message"]
                    .as_str()
                    .or_else(|| response["body"]["error"]["format"].as_str())
                    .unwrap_or("no reason given")
                    .to_string(),
            });
        }
        Ok(response)
    }

    fn event(&mut self, names: &[&str]) -> Result<Value, DapError> {
        self.wait_for(|m| m["type"] == "event" && names.iter().any(|name| m["event"] == *name))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // The debuggee is the adapter's child; ending the adapter ends it.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Read one `Content-Length`-framed message, or `None` at end of stream.
fn read_message(reader: &mut impl BufRead) -> Option<Value> {
    let mut length = None;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).ok()? == 0 {
            return None;
        }
        let header = header.trim_end();
        if header.is_empty() {
            if length.is_some() {
                break;
            }
            continue;
        }
        if let Some(n) = header.strip_prefix("Content-Length:") {
            length = n.trim().parse::<usize>().ok();
        }
    }
    let mut body = vec![0u8; length?];
    reader.read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

/// Run `req.test` under a debugger until it reaches `req.file:req.line`, and
/// return what the frame held there. A test that ends without reaching the
/// line is reported with its exit code and no variables, not as a failure of
/// this function: that is a finding about the code.
pub fn debug_test(req: &DebugRequest) -> Result<DebugOutcome, DapError> {
    let started = Instant::now();
    let (argv, adapter) = find_adapter()?;
    let binary = test_binary(req)?;
    let source = {
        let path = Path::new(&req.file);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            req.workspace.join(path)
        }
    };
    let left = req.timeout.saturating_sub(started.elapsed());
    let mut session = Session::start(&argv, &req.workspace, left)?;

    session.request(
        "initialize",
        json!({
            "clientID": "xencode",
            "adapterID": "xencode",
            "linesStartAt1": true,
            "columnsStartAt1": true,
            "pathFormat": "path",
        }),
    )?;
    // Launch first, then wait for `initialized` before configuring: both
    // adapters send it only once a program is loaded, and lldb-dap answers the
    // launch itself only after `configurationDone`, so its response is
    // collected at the end rather than waited for here.
    let launch = session.send(
        "launch",
        json!({
            "program": binary,
            "args": [req.test, "--exact", "--nocapture", "--test-threads=1"],
            "cwd": req.workspace,
            "stopOnEntry": false,
        }),
    )?;
    session.event(&["initialized"])?;
    let set = session.request(
        "setBreakpoints",
        json!({
            "source": { "path": source },
            "breakpoints": [{ "line": req.line }],
        }),
    )?;
    let verified = set["body"]["breakpoints"][0]["verified"] == true;
    session.request("configurationDone", json!({}))?;
    let launched = session.wait_for(|m| m["type"] == "response" && m["request_seq"] == launch)?;
    if launched["success"] != true {
        return Err(DapError::Refused {
            command: "launch".into(),
            message: launched["message"]
                .as_str()
                .unwrap_or("no reason given")
                .into(),
        });
    }
    if !verified {
        // Not fatal: some adapters verify a pending breakpoint only once the
        // code is loaded. Whether it was hit is what the events below say.
    }

    let event = session.event(&["stopped", "exited", "terminated"])?;
    let mut outcome = DebugOutcome {
        adapter,
        binary,
        stopped_at: None,
        reason: None,
        variables: Vec::new(),
        exit_code: None,
    };
    if event["event"] != "stopped" {
        if event["event"] == "exited" {
            outcome.exit_code = event["body"]["exitCode"].as_i64();
        } else if let Ok(exited) = session.event(&["exited"]) {
            outcome.exit_code = exited["body"]["exitCode"].as_i64();
        }
        return Ok(outcome);
    }
    outcome.reason = event["body"]["reason"].as_str().map(str::to_string);
    let thread = event["body"]["threadId"].as_i64().unwrap_or(1);
    let trace = session.request(
        "stackTrace",
        json!({ "threadId": thread, "startFrame": 0, "levels": 1 }),
    )?;
    let frame = &trace["body"]["stackFrames"][0];
    outcome.stopped_at = Some(format!(
        "{} at {}:{}",
        frame["name"].as_str().unwrap_or("?"),
        frame["source"]["path"].as_str().unwrap_or("?"),
        frame["line"].as_i64().unwrap_or(0)
    ));
    let frame_id = frame["id"].as_i64().unwrap_or(0);
    let scopes = session.request("scopes", json!({ "frameId": frame_id }))?;
    for scope in scopes["body"]["scopes"]
        .as_array()
        .cloned()
        .unwrap_or_default()
    {
        let name = scope["name"].as_str().unwrap_or("").to_string();
        // Registers and globals are not what "the locals" means.
        if !(name.contains("Local") || name.contains("Argument")) {
            continue;
        }
        let reference = scope["variablesReference"].as_i64().unwrap_or(0);
        if reference == 0 {
            continue;
        }
        let vars = session.request("variables", json!({ "variablesReference": reference }))?;
        for var in vars["body"]["variables"]
            .as_array()
            .cloned()
            .unwrap_or_default()
        {
            outcome.variables.push(Variable {
                name: var["name"].as_str().unwrap_or("").to_string(),
                value: var["value"].as_str().unwrap_or("").to_string(),
                ty: var["type"].as_str().map(str::to_string),
                scope: name.clone(),
            });
        }
    }
    let _ = session.send("disconnect", json!({ "terminateDebuggee": true }));
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_framed_message_is_read_whole_and_a_closed_stream_is_none() {
        let body = r#"{"type":"event","event":"initialized"}"#;
        let raw = format!("Content-Length: {}\r\n\r\n{body}", body.len());
        let mut reader = std::io::Cursor::new(raw.into_bytes());
        let message = read_message(&mut reader).expect("one message");
        assert_eq!(message["event"], "initialized");
        assert!(read_message(&mut reader).is_none());
    }

    /// The plan item's own done-when: stop at a breakpoint in a failing test and
    /// read a local. Runs a real debugger on a real test binary; skipped, saying
    /// so, where no adapter is installed.
    #[test]
    fn a_failing_test_stops_at_the_line_and_its_locals_are_read() {
        if let Err(e) = find_adapter() {
            eprintln!("skipping: {e}");
            return;
        }
        let dir = std::env::temp_dir().join(format!("xencode-dap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"dapsample\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("src/lib.rs"),
            "pub fn add(a: u32, b: u32) -> u32 {\n    a + b\n}\n\n\
             #[cfg(test)]\nmod tests {\n    #[test]\n    fn sums() {\n        \
             let total = super::add(2, 3);\n        let expected = 6u32;\n        \
             assert_eq!(total, expected);\n    }\n}\n",
        )
        .unwrap();
        let outcome = debug_test(&DebugRequest {
            workspace: dir.clone(),
            package: None,
            test: "tests::sums".into(),
            file: "src/lib.rs".into(),
            line: 11,
            timeout: Duration::from_secs(240),
        })
        .expect("a debugger session runs");
        eprintln!("debug session: {outcome:#?}");
        assert_eq!(outcome.reason.as_deref(), Some("breakpoint"), "{outcome:?}");
        let value = |name: &str| {
            outcome
                .variables
                .iter()
                .find(|v| v.name == name)
                .map(|v| v.value.clone())
        };
        assert_eq!(value("total").as_deref(), Some("5"), "{outcome:?}");
        assert_eq!(value("expected").as_deref(), Some("6"), "{outcome:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
