//! A Model Context Protocol client over the stdio transport.
//!
//! One [`McpClient`] owns one child process: we write newline-delimited JSON to
//! its stdin and read the same back from its stdout. Requests are matched to
//! responses by JSON-RPC id, every request has a deadline, and the child is
//! killed when the client goes away — a server that hangs must not outlive the
//! session that asked it a question.
//!
//! Scope is deliberate: tools only. Resources, prompts, sampling and roots are
//! not part of what xencode's agent loop can act on, so they are never
//! negotiated. Message shapes live in [`crate::protocol`], where they are
//! testable without a process on the other end of a pipe.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout};
use tokio::sync::{oneshot, Mutex as AsyncMutex};

use crate::{error::McpError, protocol};

/// How to launch one server, as declared in the user's config. Plain data:
/// this crate does not depend on xencode's config crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSpec {
    /// Name as written in config — becomes the `mcp__<name>__<tool>` prefix.
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    /// Extra variables for the child. Inherited variables stay in place: a
    /// server needs `PATH` to find its own runtime.
    pub env: Vec<(String, String)>,
}

impl ServerSpec {
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        ServerSpec {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            env: Vec::new(),
        }
    }

    pub fn args(mut self, args: &[String]) -> Self {
        self.args = args.to_vec();
        self
    }
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, McpError>>>>>;

/// What the reader tasks notice but the caller only asks about on demand.
#[derive(Default)]
struct Watchdog {
    /// Lines the server printed that were not JSON-RPC. The spec forbids this;
    /// servers that log to stdout anyway do not get to wedge a session, so
    /// they are counted and reported instead of treated as fatal.
    stray_lines: AtomicUsize,
    last_stray: Mutex<Option<String>>,
    /// Tail of the server's stderr — usually the only evidence of why it died.
    stderr: Mutex<String>,
}

const STDERR_KEEP: usize = 8 * 1024;

pub struct McpClient {
    name: String,
    request_timeout: Duration,
    stdin: Arc<AsyncMutex<ChildStdin>>,
    child: AsyncMutex<Child>,
    pending: Pending,
    next_id: AtomicU64,
    watchdog: Arc<Watchdog>,
    reader: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl McpClient {
    /// Spawn the server, complete the `initialize` handshake, and send the
    /// `notifications/initialized` follow-up. A failure names the server; if
    /// it wrote anything to stderr, that is appended.
    pub async fn start(spec: &ServerSpec, request_timeout: Duration) -> Result<Self, McpError> {
        let mut child = tokio::process::Command::new(&spec.command)
            .args(&spec.args)
            .envs(spec.env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| McpError::Spawn {
                server: spec.name.clone(),
                reason: e.to_string(),
            })?;

        // All three pipes were requested above, so `take()` cannot yield None.
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");

        let watchdog = Arc::new(Watchdog::default());
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let reader = tokio::spawn(read_loop(
            stdout,
            spec.name.clone(),
            Arc::clone(&pending),
            Arc::clone(&watchdog),
        ));
        tokio::spawn(collect_stderr(stderr, Arc::clone(&watchdog)));

        let client = McpClient {
            name: spec.name.clone(),
            request_timeout,
            stdin: Arc::new(AsyncMutex::new(stdin)),
            child: AsyncMutex::new(child),
            pending,
            next_id: AtomicU64::new(1),
            watchdog,
            reader: Mutex::new(Some(reader)),
        };

        if let Err(e) = client
            .request(
                "initialize",
                protocol::initialize_params(env!("CARGO_PKG_VERSION")),
            )
            .await
        {
            let failure = client.with_stderr(e);
            client.shutdown().await;
            return Err(failure);
        }
        if let Err(e) = client
            .write_line(&protocol::notification(
                "notifications/initialized",
                json!({}),
            ))
            .await
        {
            let failure = client.with_stderr(e);
            client.shutdown().await;
            return Err(failure);
        }
        Ok(client)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// `tools/list`. An empty list is an honest answer, not an error.
    pub async fn list_tools(&self) -> Result<Vec<protocol::McpTool>, McpError> {
        let result = self.request("tools/list", Value::Null).await?;
        Ok(protocol::parse_tools(&result))
    }

    /// `tools/call`. A result the server flagged as an error keeps that fact
    /// in the text the model reads, so a failed call cannot be mistaken for
    /// a successful one.
    pub async fn call_tool(&self, tool: &str, arguments: Value) -> Result<String, McpError> {
        let arguments = match arguments {
            Value::Null => json!({}),
            object @ Value::Object(_) => object,
            other => json!({ "value": other }),
        };
        let result = self
            .request("tools/call", protocol::call_tool_params(tool, arguments))
            .await?;
        let outcome = protocol::parse_tool_result(&result);
        if outcome.is_error {
            Ok(format!(
                "MCP tool `{tool}` reported an error:\n{}",
                outcome.text
            ))
        } else if outcome.text.is_empty() {
            Ok(format!("MCP tool `{tool}` returned no content."))
        } else {
            Ok(outcome.text)
        }
    }

    /// How many lines of non-JSON the server has printed.
    pub fn stray_lines(&self) -> usize {
        self.watchdog.stray_lines.load(Ordering::Relaxed)
    }

    /// The last lines the server wrote to stderr, for `/mcp status` and for
    /// handshake failures.
    pub fn stderr_tail(&self) -> String {
        self.watchdog
            .stderr
            .lock()
            .map(|buf| {
                buf.lines()
                    .rev()
                    .take(4)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    }

    /// The most recent line that was neither JSON-RPC nor ours, if any.
    pub fn last_stray(&self) -> Option<String> {
        self.watchdog
            .last_stray
            .lock()
            .ok()
            .and_then(|last| last.clone())
    }

    /// Kill the server and stop reading. Safe to call twice.
    pub async fn shutdown(&self) {
        let mut child = self.child.lock().await;
        let _ = child.start_kill();
        let _ = child.wait().await;
        drop(child);
        if let Ok(mut reader) = self.reader.lock() {
            if let Some(handle) = reader.take() {
                handle.abort();
            }
        }
    }

    /// Append the server's own words to an error, when it had any. A handshake
    /// failure that says only "closed the connection" is half the story: the
    /// server almost always explained itself on stderr first.
    fn with_stderr(&self, error: McpError) -> McpError {
        let tail = self.stderr_tail();
        if tail.trim().is_empty() {
            return error;
        }
        McpError::Protocol {
            server: self.name.clone(),
            message: format!("{} — server said: {}", error, tail.trim()),
        }
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (responder, answer) = oneshot::channel();
        if let Ok(mut pending) = self.pending.lock() {
            pending.insert(id, responder);
        }
        if let Err(e) = self
            .write_line(&protocol::request(id, method, params))
            .await
        {
            self.forget(id);
            return Err(e);
        }
        match tokio::time::timeout(self.request_timeout, answer).await {
            Err(_elapsed) => {
                self.forget(id);
                Err(McpError::Timeout {
                    server: self.name.clone(),
                    method: method.to_string(),
                    secs: self.request_timeout.as_secs().max(1),
                })
            }
            Ok(Err(_dropped)) => {
                self.forget(id);
                Err(McpError::Closed {
                    server: self.name.clone(),
                })
            }
            Ok(Ok(result)) => result,
        }
    }

    fn forget(&self, id: u64) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.remove(&id);
        }
    }

    async fn write_line(&self, message: &Value) -> Result<(), McpError> {
        let mut line = serde_json::to_string(message).map_err(|e| McpError::Protocol {
            server: self.name.clone(),
            message: format!("cannot encode request: {e}"),
        })?;
        line.push('\n');
        let mut stdin = self.stdin.lock().await;
        match stdin.write_all(line.as_bytes()).await {
            Ok(()) => stdin.flush().await.map_err(|_| McpError::Closed {
                server: self.name.clone(),
            }),
            // A broken pipe means the server is already gone.
            Err(_) => Err(McpError::Closed {
                server: self.name.clone(),
            }),
        }
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        if let Ok(mut reader) = self.reader.lock() {
            if let Some(handle) = reader.take() {
                handle.abort();
            }
        }
        // `kill_on_drop(true)` reaps the process; aborting the reader above
        // just stops us from spinning on a pipe nobody will answer on.
    }
}

fn deliver(pending: &Pending, id: u64, payload: Result<Value, McpError>) {
    let responder = pending.lock().ok().and_then(|mut map| map.remove(&id));
    // A receiver that already gave up (timed out) simply is not there.
    if let Some(responder) = responder {
        let _ = responder.send(payload);
    }
}

async fn read_loop(stdout: ChildStdout, server: String, pending: Pending, watchdog: Arc<Watchdog>) {
    let mut lines = BufReader::new(stdout).lines();
    loop {
        match lines.next_line().await {
            Ok(None) => break,
            Ok(Some(line)) => {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(&line) {
                    Ok(value) => dispatch(&server, &pending, &watchdog, value),
                    Err(_) => {
                        if let Ok(mut last) = watchdog.last_stray.lock() {
                            *last = Some(truncate(&line, 200));
                        }
                        watchdog.stray_lines.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            // The pipe broke: the server died or wrote something undecodable.
            Err(_) => break,
        }
    }
    // After this nobody can answer; wake every waiter with the truth.
    if let Ok(mut map) = pending.lock() {
        for (_, responder) in map.drain() {
            let _ = responder.send(Err(McpError::Closed {
                server: server.clone(),
            }));
        }
    }
}

fn dispatch(server: &str, pending: &Pending, _watchdog: &Watchdog, value: Value) {
    // Our ids are numbers. Anything else — a string id, or a server-initiated
    // notification with no id — is not ours to answer.
    let Some(id) = value.get("id").and_then(Value::as_u64) else {
        return;
    };
    if let Some(error) = value.get("error") {
        let (code, message) = protocol::parse_error(error);
        deliver(
            pending,
            id,
            Err(McpError::Server {
                server: server.to_string(),
                code,
                message,
            }),
        );
        return;
    }
    match value.get("result") {
        Some(result) => deliver(pending, id, Ok(result.clone())),
        None => deliver(
            pending,
            id,
            Err(McpError::Protocol {
                server: server.to_string(),
                message: format!("response {id} has neither result nor error"),
            }),
        ),
    }
}

async fn collect_stderr(stderr: ChildStderr, watchdog: Arc<Watchdog>) {
    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if let Ok(mut buf) = watchdog.stderr.lock() {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(&line);
            if buf.len() > STDERR_KEEP {
                let mut start = buf.len() - STDERR_KEEP;
                while start > 0 && !buf.is_char_boundary(start) {
                    start -= 1;
                }
                buf.drain(..start);
            }
        }
    }
}

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}
