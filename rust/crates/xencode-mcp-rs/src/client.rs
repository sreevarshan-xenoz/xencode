//! A Model Context Protocol client, over one of two transports.
//!
//! [`Transport::Stdio`] owns a child process: we write newline-delimited JSON to
//! its stdin and read the same back from its stdout. [`Transport::Http`] posts
//! each message to a hosted server — one POST per request, the answer coming
//! back as a JSON body or as a `text/event-stream` of them — which is how a
//! server that is not on this machine is reached.
//!
//! Both share everything above the pipes: requests are matched to responses by
//! JSON-RPC id, every request has a deadline, and a stdio server is killed when
//! the client goes away — a server that hangs must not outlive the session that
//! asked it a question. An HTTP session is told we are finished instead.
//!
//! Scope is deliberate: tools, resources and prompts. Sampling and roots are
//! requests a server would make of us, and xencode answers neither, so they are
//! never declared. What a server offers is read from its handshake and a method
//! it did not declare is not sent. Message shapes live in [`crate::protocol`],
//! where they are testable without a process or a port on the other end.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout};
use tokio::sync::{oneshot, Mutex as AsyncMutex};

use crate::{error::McpError, protocol};

/// How the server is reached. A server is one or the other: a hosted endpoint
/// has no process to spawn, and a process on a pipe has no address to post to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transport {
    /// Spawn `command` with `args`, with `env` added to the inherited
    /// environment — a server needs `PATH` to find its own runtime.
    Stdio {
        command: String,
        args: Vec<String>,
        env: Vec<(String, String)>,
    },
    /// Post every message to `url`, an `http://` or `https://` MCP endpoint.
    Http {
        url: String,
        /// Sent on every request: this is where a hosted server's API token
        /// goes. A value is never printed — a failure names the request, not
        /// what it was authenticated with.
        headers: Vec<(String, String)>,
    },
}

/// How to launch or contact one server, as declared in the user's config. Plain
/// data: this crate does not depend on xencode's config crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSpec {
    /// Name as written in config — becomes the `mcp__<name>__<tool>` prefix.
    pub name: String,
    pub transport: Transport,
}

impl ServerSpec {
    /// A server to spawn on a pipe.
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        ServerSpec {
            name: name.into(),
            transport: Transport::Stdio {
                command: command.into(),
                args: Vec::new(),
                env: Vec::new(),
            },
        }
    }

    /// A server to reach at an address — an `https://` url, with a token in a
    /// header, is how a hosted MCP server becomes reachable at all.
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        ServerSpec {
            name: name.into(),
            transport: Transport::Http {
                url: url.into(),
                headers: Vec::new(),
            },
        }
    }

    /// The command line for a stdio server. An HTTP spec has no process to hand
    /// arguments to, so this leaves one as it is.
    pub fn args(mut self, args: &[String]) -> Self {
        if let Transport::Stdio { args: slot, .. } = &mut self.transport {
            *slot = args.to_vec();
        }
        self
    }

    /// Extra variables for a stdio server's child process.
    pub fn env(mut self, env: Vec<(String, String)>) -> Self {
        if let Transport::Stdio { env: slot, .. } = &mut self.transport {
            *slot = env;
        }
        self
    }

    /// One more header for an HTTP server, e.g. `("authorization", "Bearer …")`.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        if let Transport::Http { headers, .. } = &mut self.transport {
            headers.push((name.into(), value.into()));
        }
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

/// The header a Streamable HTTP server issues to identify our session, and the
/// one we send back with every later request.
const SESSION_HEADER: &str = "mcp-session-id";
/// Which revision of the protocol the requests after the handshake speak.
const PROTOCOL_VERSION_HEADER: &str = "mcp-protocol-version";

/// Where a session's messages go.
enum Connection {
    Stdio {
        stdin: Arc<AsyncMutex<ChildStdin>>,
        child: AsyncMutex<Child>,
    },
    Http {
        link: HttpLink,
    },
}

/// One HTTP endpoint, with what it takes to talk to it in order.
struct HttpLink {
    client: reqwest::Client,
    url: String,
    headers: Vec<(String, String)>,
    /// The session id the server issued, once it has issued one.
    session: AsyncMutex<Option<String>>,
    /// The revision the handshake settled on, once the handshake has run.
    negotiated: AsyncMutex<Option<String>>,
}

impl HttpLink {
    /// A request carrying the headers this endpoint needs, including the two
    /// that only exist once the handshake has produced them.
    async fn request(&self, message: &Value) -> reqwest::RequestBuilder {
        let mut request = self
            .client
            .post(&self.url)
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .json(message);
        for (name, value) in &self.headers {
            request = request.header(name.as_str(), value.as_str());
        }
        if let Some(session) = self.session.lock().await.as_ref() {
            request = request.header(SESSION_HEADER, session.as_str());
        }
        if let Some(version) = self.negotiated.lock().await.as_ref() {
            request = request.header(PROTOCOL_VERSION_HEADER, version.as_str());
        }
        request
    }

    /// Tell the endpoint we are finished. Nothing waits on the answer: a server
    /// that will not accept the news is not our server, and a session we never
    /// opened has no id to revoke.
    async fn end_session(&self) {
        let mut request = self.client.delete(&self.url);
        for (name, value) in &self.headers {
            request = request.header(name.as_str(), value.as_str());
        }
        if let Some(session) = self.session.lock().await.as_ref() {
            request = request.header(SESSION_HEADER, session.as_str());
        }
        let _ = tokio::time::timeout(Duration::from_secs(5), request.send()).await;
    }
}

pub struct McpClient {
    name: String,
    /// What the user declared, kept so `/mcp` can say where the server is.
    launch: Transport,
    request_timeout: Duration,
    connection: Connection,
    pending: Pending,
    next_id: AtomicU64,
    watchdog: Arc<Watchdog>,
    /// The method sets this server said it has, from the handshake.
    capabilities: protocol::ServerCapabilities,
    reader: Mutex<Option<tokio::task::JoinHandle<()>>>,
    stderr_collector: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl McpClient {
    /// Reach the server by its transport, complete the `initialize` handshake,
    /// and send the `notifications/initialized` follow-up. A failure names the
    /// server; for a stdio server that wrote anything to stderr, that is
    /// appended, because it is usually the only explanation there is.
    pub async fn start(spec: &ServerSpec, request_timeout: Duration) -> Result<Self, McpError> {
        let mut client = Self::open(spec, request_timeout)?;
        match client
            .request(
                "initialize",
                protocol::initialize_params(env!("CARGO_PKG_VERSION")),
            )
            .await
        {
            // What the server said it has, kept for every later call: a method
            // it did not declare must not be sent.
            Ok(offered) => {
                client.capabilities = protocol::parse_server_capabilities(&offered);
                client.record_revision(&offered);
            }
            Err(e) => {
                let failure = client.with_stderr(e).await;
                client.shutdown().await;
                return Err(failure);
            }
        }
        if let Err(e) = client
            .write_line(&protocol::notification(
                "notifications/initialized",
                json!({}),
            ))
            .await
        {
            let failure = client.with_stderr(e).await;
            client.shutdown().await;
            return Err(failure);
        }
        Ok(client)
    }

    /// Establish the transport without speaking yet. A stdio server's pipes get
    /// their reader and stderr collector; an HTTP endpoint needs no task, since
    /// every answer arrives on the request that asked for it.
    fn open(spec: &ServerSpec, request_timeout: Duration) -> Result<Self, McpError> {
        let watchdog = Arc::new(Watchdog::default());
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let (connection, reader, stderr_collector) = match &spec.transport {
            Transport::Stdio { command, args, env } => {
                let (stdin, stdout, stderr, child) = spawn(command, args, env, &spec.name)?;
                let reader = tokio::spawn(read_loop(
                    stdout,
                    spec.name.clone(),
                    Arc::clone(&pending),
                    Arc::clone(&watchdog),
                ));
                let stderr_collector = tokio::spawn(collect_stderr(stderr, Arc::clone(&watchdog)));
                (
                    Connection::Stdio {
                        stdin: Arc::new(AsyncMutex::new(stdin)),
                        child: AsyncMutex::new(child),
                    },
                    Some(reader),
                    Some(stderr_collector),
                )
            }
            Transport::Http { url, headers } => (
                Connection::Http {
                    link: HttpLink {
                        client: reqwest::Client::new(),
                        url: url.clone(),
                        headers: headers.clone(),
                        session: AsyncMutex::new(None),
                        negotiated: AsyncMutex::new(None),
                    },
                },
                None,
                None,
            ),
        };
        Ok(McpClient {
            name: spec.name.clone(),
            launch: spec.transport.clone(),
            request_timeout,
            connection,
            pending,
            next_id: AtomicU64::new(1),
            watchdog,
            capabilities: protocol::ServerCapabilities::default(),
            reader: Mutex::new(reader),
            stderr_collector: Mutex::new(stderr_collector),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// What the handshake said this server has, so a caller can tell "this
    /// server has no resources" apart from "this server is not connected".
    pub fn capabilities(&self) -> &protocol::ServerCapabilities {
        &self.capabilities
    }

    /// Refuse before sending when the server declared what it has and this was
    /// not among it. A server that declared nothing is let through — see
    /// [`protocol::ServerCapabilities::offers`].
    fn require(&self, feature: protocol::Feature) -> Result<(), McpError> {
        if self.capabilities.offers(feature) {
            return Ok(());
        }
        Err(McpError::NotOffered {
            server: self.name.clone(),
            feature: feature.key(),
        })
    }

    /// `tools/list`. An empty list is an honest answer, not an error.
    pub async fn list_tools(&self) -> Result<Vec<protocol::McpTool>, McpError> {
        self.require(protocol::Feature::Tools)?;
        // An empty object, not null: a strict server (Playwright MCP) drops a
        // `tools/list` whose params are null without answering, which reads as
        // a timeout rather than as the rejection it is.
        let result = self.request("tools/list", json!({})).await?;
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

    /// `resources/list`. An empty list is an honest answer, not an error.
    pub async fn list_resources(&self) -> Result<Vec<protocol::McpResource>, McpError> {
        self.require(protocol::Feature::Resources)?;
        let result = self.request("resources/list", json!({})).await?;
        Ok(protocol::parse_resources(&result))
    }

    /// `resources/read`. One uri can come back as several pieces, so all of
    /// them are returned; a server that cannot read it answers with a JSON-RPC
    /// error, which reaches the caller as [`McpError::Server`].
    pub async fn read_resource(
        &self,
        uri: &str,
    ) -> Result<Vec<protocol::ResourceContent>, McpError> {
        self.require(protocol::Feature::Resources)?;
        let result = self
            .request("resources/read", protocol::read_resource_params(uri))
            .await?;
        Ok(protocol::parse_resource_contents(&result))
    }

    /// `prompts/list`.
    pub async fn list_prompts(&self) -> Result<Vec<protocol::McpPrompt>, McpError> {
        self.require(protocol::Feature::Prompts)?;
        let result = self.request("prompts/list", json!({})).await?;
        Ok(protocol::parse_prompts(&result))
    }

    /// `prompts/get` — the messages a prepared prompt renders to, with the role
    /// the server gave each one.
    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: &[(String, String)],
    ) -> Result<Vec<protocol::PromptMessage>, McpError> {
        self.require(protocol::Feature::Prompts)?;
        let result = self
            .request("prompts/get", protocol::get_prompt_params(name, arguments))
            .await?;
        Ok(protocol::parse_prompt(&result))
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

    /// Where this session's messages go, in a form safe to show: a stdio server
    /// by the command line it was spawned with, an HTTP endpoint by its url with
    /// any credentials masked the way a provider key is. Header values — where an
    /// API token lives — are never part of it.
    pub fn endpoint(&self) -> String {
        match &self.launch {
            Transport::Stdio { command, args, .. } => {
                if args.is_empty() {
                    command.clone()
                } else {
                    format!("{command} {}", args.join(" "))
                }
            }
            Transport::Http { url, .. } => masked_url(url),
        }
    }

    /// Remember the revision the server answered with, so the requests after the
    /// handshake can name it. A server that said nothing keeps our own version.
    fn record_revision(&mut self, offered: &Value) {
        let Connection::Http { link } = &self.connection else {
            return;
        };
        let version = offered
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or(protocol::PROTOCOL_VERSION)
            .to_string();
        if let Ok(mut slot) = link.negotiated.try_lock() {
            *slot = Some(version);
        }
    }

    /// End the session: kill the process, or tell the endpoint we are finished.
    /// Safe to call twice.
    pub async fn shutdown(&self) {
        match &self.connection {
            Connection::Stdio { child, .. } => {
                let mut child = child.lock().await;
                let _ = child.start_kill();
                let _ = child.wait().await;
            }
            Connection::Http { link } => link.end_session().await,
        }
        if let Ok(mut reader) = self.reader.lock() {
            if let Some(handle) = reader.take() {
                handle.abort();
            }
        }
        if let Ok(mut slot) = self.stderr_collector.lock() {
            if let Some(handle) = slot.take() {
                handle.abort();
            }
        }
    }

    /// Append the server's own words to an error, when it had any. A handshake
    /// failure that says only "closed the connection" is half the story: the
    /// server almost always explained itself on stderr first. When a stdio
    /// server exited, we boundedly await the collector task so buffered lines
    /// are drained into the watchdog before reading the tail.
    async fn with_stderr(&self, error: McpError) -> McpError {
        if let Connection::Stdio { child, .. } = &self.connection {
            let mut child = child.lock().await;
            let exited = match child.try_wait() {
                Ok(Some(_status)) => true,
                Ok(None) => {
                    if matches!(error, McpError::Closed { .. }) {
                        tokio::time::timeout(Duration::from_millis(100), child.wait())
                            .await
                            .ok()
                            .is_some()
                    } else {
                        false
                    }
                }
                Err(_) => false,
            };
            if exited {
                drop(child);
                let handle = self
                    .stderr_collector
                    .lock()
                    .ok()
                    .and_then(|mut slot| slot.take());
                if let Some(handle) = handle {
                    let _ = tokio::time::timeout(Duration::from_millis(500), handle).await;
                }
            }
        }
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
        match &self.connection {
            Connection::Stdio { stdin, .. } => {
                let mut line = serde_json::to_string(message).map_err(|e| McpError::Protocol {
                    server: self.name.clone(),
                    message: format!("cannot encode request: {e}"),
                })?;
                line.push('\n');
                let mut stdin = stdin.lock().await;
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
            Connection::Http { link } => self.post_http(link, message).await,
        }
    }

    /// One POST per message, under the same deadline a pipe answer gets: a
    /// hosted server that accepts the request and then says nothing must not
    /// hold the turn open past its budget.
    async fn post_http(&self, link: &HttpLink, message: &Value) -> Result<(), McpError> {
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("request")
            .to_string();
        let expecting = message.get("id").and_then(Value::as_u64);
        let sent = link.request(message).await;
        match tokio::time::timeout(
            self.request_timeout,
            self.read_answer(link, sent, expecting),
        )
        .await
        {
            Err(_elapsed) => Err(McpError::Timeout {
                server: self.name.clone(),
                method,
                secs: self.request_timeout.as_secs().max(1),
            }),
            Ok(result) => result,
        }
    }

    /// Take the answer(s) off one HTTP response and hand each message to the
    /// waiting request, which is the same delivery a pipe reader does.
    ///
    /// The body is either one JSON document or a `text/event-stream` of them. A
    /// server is allowed to keep that stream open after answering — it may still
    /// have notifications to send — so the stream is read until the id we posted
    /// has been answered, then dropped.
    async fn read_answer(
        &self,
        link: &HttpLink,
        request: reqwest::RequestBuilder,
        expecting: Option<u64>,
    ) -> Result<(), McpError> {
        let response = request
            .send()
            .await
            .map_err(|e| self.broke(&link.url, &e.to_string()))?;
        let status = response.status();
        if let Some(session) = response
            .headers()
            .get(SESSION_HEADER)
            .and_then(|value| value.to_str().ok())
        {
            *link.session.lock().await = Some(session.to_string());
        }
        // 202 is the endpoint's way of saying "noted" to a notification: there
        // is no body and nothing to wait for.
        if status == reqwest::StatusCode::ACCEPTED || status == reqwest::StatusCode::NO_CONTENT {
            return Ok(());
        }
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| status.canonical_reason().unwrap_or_default().to_string());
            // A server-side failure is a refusal with a status number rather
            // than a JSON-RPC code; either way the caller reads it as the
            // server's own error.
            return Err(McpError::Server {
                server: self.name.clone(),
                code: i64::from(status.as_u16()),
                message: truncate(self.kept_clean(&link.url, &body).trim(), 400),
            });
        }
        let streamed = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|typed| typed.contains("text/event-stream"));
        if !streamed {
            let body = response
                .text()
                .await
                .map_err(|e| self.broke(&link.url, &e.to_string()))?;
            self.absorb(&body);
            return Ok(());
        }

        let mut stream = response.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| self.broke(&link.url, &e.to_string()))?;
            buffer.extend_from_slice(&chunk);
            while let Some(frame) = take_frame(&mut buffer) {
                if let Some(payload) = sse_payload(&frame) {
                    self.absorb(&payload);
                }
                if self.answered(expecting) {
                    // Dropping the stream is how we leave a stream the server
                    // intends to keep open.
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Whether the request we posted has been answered — by this response or by
    /// anything else that arrived first.
    fn answered(&self, expecting: Option<u64>) -> bool {
        let Some(id) = expecting else {
            return false;
        };
        match self.pending.lock() {
            Ok(pending) => !pending.contains_key(&id),
            // A poisoned map says nothing; treat it as unanswered and keep reading.
            Err(_) => false,
        }
    }

    /// Parse one body as protocol and deliver it, or count it as the server
    /// printing something that is not JSON-RPC.
    fn absorb(&self, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        absorb_message(
            &self.name,
            &self.pending,
            &self.watchdog,
            &text.replace("\n", " "),
        );
    }

    /// A failure of the transport, in words that name the endpoint without
    /// repeating any credential the url or the error may carry.
    fn broke(&self, url: &str, reason: &str) -> McpError {
        McpError::Protocol {
            server: self.name.clone(),
            message: format!("{}: {}", masked_url(url), self.kept_clean(url, reason)),
        }
    }

    /// Cut any copy of the request url — the only place a token written into a
    /// url could reappear — back out of text an error carries.
    fn kept_clean(&self, url: &str, text: &str) -> String {
        let masked = masked_url(url);
        if masked == url {
            return text.to_string();
        }
        text.replace(url, &masked)
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

/// The one place a message that arrived becomes an answer to whoever asked,
/// shared by the pipe reader and an HTTP response body: neither transport gets
/// to decide on its own what counts as protocol.
fn absorb_message(server: &str, pending: &Pending, watchdog: &Watchdog, text: &str) {
    match serde_json::from_str::<Value>(text) {
        Ok(value) => dispatch(server, pending, watchdog, value),
        Err(_) => {
            if let Ok(mut last) = watchdog.last_stray.lock() {
                *last = Some(truncate(text, 200));
            }
            watchdog.stray_lines.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Spawn a stdio server with its three pipes. `kill_on_drop` means a client that
/// is dropped without `shutdown` still cannot leave a process behind.
fn spawn(
    command: &str,
    args: &[String],
    env: &[(String, String)],
    server: &str,
) -> Result<(ChildStdin, ChildStdout, ChildStderr, Child), McpError> {
    let mut child = tokio::process::Command::new(command)
        .args(args)
        .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| McpError::Spawn {
            server: server.to_string(),
            reason: e.to_string(),
        })?;

    // All three pipes were requested above, so `take()` cannot yield None.
    let stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");
    Ok((stdin, stdout, stderr, child))
}

/// Cut the first complete `text/event-stream` frame off the buffer. Both the
/// `\n\n` and the `\r\n\r\n` spelling end a frame, and the longer one must not
/// be read as `\n` plus a frame that starts with a newline.
fn take_frame(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let (index, width) = match (find_sub(buffer, b"\r\n\r\n"), find_sub(buffer, b"\n\n")) {
        (Some(crlf), Some(linefeed)) => {
            if crlf <= linefeed {
                (crlf, 4)
            } else {
                (linefeed, 2)
            }
        }
        (Some(crlf), None) => (crlf, 4),
        (None, Some(linefeed)) => (linefeed, 2),
        (None, None) => return None,
    };
    // The delimiter is framing, not content: it is dropped along with the bytes
    // that carried it.
    let frame = buffer[..index].to_vec();
    buffer.drain(..index + width);
    Some(frame)
}

fn find_sub(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// The `data:` payload of one frame. Consecutive `data:` lines belong to a
/// single event and are joined with a newline; one leading space is part of the
/// framing rather than the payload. `event:`, `id:`, `retry:` and `:` comments
/// carry nothing xencode acts on.
fn sse_payload(frame: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(frame);
    let mut parts: Vec<&str> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            parts.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("\n"))
}

/// The host and port an address points at, the way a reachability check wants
/// them: 443 for `https://`, 80 for `http://`, and whatever the url itself says.
/// `None` for anything that is not one of those two schemes, or has no host —
/// which is also an address this client cannot post a message to.
pub fn address_of(url: &str) -> Option<(String, u16)> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let default = match parsed.scheme() {
        "https" => 443,
        "http" => 80,
        _ => return None,
    };
    let host = parsed.host_str()?.to_string();
    Some((host, parsed.port().unwrap_or(default)))
}

/// Render a secret for a screen or an error message: bullets, with the last
/// four characters kept so two tokens can be told apart. A value short enough
/// to be revealed by that tail is hidden entirely. This is the one masking rule
/// in xencode — the settings panel draws a provider key with it, and an MCP
/// endpoint that carries a token in its url is drawn with the same one.
pub fn mask_secret(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= 8 {
        return "••••".to_string();
    }
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{}{tail}", "•".repeat((chars.len() - 4).min(12)))
}

/// A url with its credentials masked the way a provider key is, for anything a
/// person or a model reads. `https://user:token@host/mcp` is a legal way to carry
/// a secret and xencode must not repeat it back.
///
/// The text is cut rather than re-serialised: `Url` percent-encodes the bullets
/// we put in the credentials part, which would show a person a row of `%E2%80%A2`
/// instead of saying that something is hidden.
pub fn masked_url(raw: &str) -> String {
    // Only a url that parses can carry credentials in the position this looks
    // for, so an unparseable value is shown as written — it fails at the
    // request, and hiding it would say less, not more.
    if reqwest::Url::parse(raw).is_err() {
        return truncate(raw, 200);
    }
    let Some(authority_start) = raw.find("://").map(|at| at + 3) else {
        return truncate(raw, 200);
    };
    let authority_end = raw[authority_start..]
        .find(['/', '?', '#'])
        .map(|offset| authority_start + offset)
        .unwrap_or(raw.len());
    // The last `@` in the authority ends the credentials: a host name may
    // legitimately contain none, and a percent-encoded one contains no literal.
    let Some(at) = raw[authority_start..authority_end]
        .rfind('@')
        .map(|offset| authority_start + offset)
    else {
        return truncate(raw, 200);
    };
    let credentials = &raw[authority_start..at];
    let masked = match credentials.split_once(':') {
        Some((user, password)) => format!("{}:{}", mask_secret(user), mask_secret(password)),
        // A token carried as the user name, which is how a bearer-only endpoint
        // is written down when nobody bothers with a dummy password.
        None => mask_secret(credentials),
    };
    let mut shown = String::with_capacity(raw.len());
    shown.push_str(&raw[..authority_start]);
    shown.push_str(&masked);
    shown.push_str(&raw[at..]);
    truncate(&shown, 200)
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
                absorb_message(&server, &pending, &watchdog, &line);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(bytes: &[u8]) -> Vec<String> {
        let mut buffer = bytes.to_vec();
        let mut out = Vec::new();
        while let Some(frame) = take_frame(&mut buffer) {
            out.push(String::from_utf8_lossy(&frame).to_string());
        }
        out
    }

    #[test]
    fn a_frame_is_cut_at_either_spelling_of_its_ending() {
        assert_eq!(
            frames(b"data: one\n\ndata: two"),
            ["data: one"],
            "one newline pair ends a frame; the rest waits for its own"
        );
        assert_eq!(
            frames(b"data: one\r\n\r\ndata: two\r\n\r\n"),
            ["data: one", "data: two"],
            "the spelling with carriage returns ends a frame too, and is not \
             read as a lone newline"
        );
        // An incomplete frame stays in the buffer until the rest arrives.
        let mut half = b"data: part".to_vec();
        assert!(take_frame(&mut half).is_none());
        half.extend_from_slice(b"\n\n");
        assert_eq!(take_frame(&mut half), Some(b"data: part".to_vec()));
        assert!(half.is_empty(), "the delimiter was framing, not content");
    }

    #[test]
    fn only_the_data_line_carries_an_answer() {
        let frame = b": keep-alive\nevent: message\nid: 7\nretry: 5000\ndata: {\"id\":1}";
        assert_eq!(sse_payload(frame).as_deref(), Some("{\"id\":1}"));
        // Several `data:` lines are one event, joined with a newline, and the
        // one space after the colon belongs to the framing.
        assert_eq!(
            sse_payload(b"data: first\ndata:second").as_deref(),
            Some("first\nsecond")
        );
        assert_eq!(sse_payload(b"event: ping\n\n"), None);
    }

    #[test]
    fn a_masked_key_keeps_only_its_last_four_characters() {
        assert_eq!(
            mask_secret("sk-test-77-not-a-real-token"),
            "••••••••••••oken"
        );
        // Short enough that the tail would be the value: nothing is shown.
        assert_eq!(mask_secret("abc12345"), "••••");
        assert_eq!(mask_secret("123456789"), "•••••6789");
    }

    #[test]
    fn a_token_is_masked_whichever_half_of_the_url_it_was_written_in() {
        // A password, a bare username carrying the token, and both at once.
        assert_eq!(
            masked_url("https://user:hunter2@example.com:8443/mcp?x=1"),
            "https://••••:••••@example.com:8443/mcp?x=1"
        );
        assert_eq!(
            masked_url("http://sk-test-77-not-a-real-token@127.0.0.1:9/mcp"),
            "http://••••••••••••oken@127.0.0.1:9/mcp"
        );
        assert_eq!(
            masked_url("https://alice:sekrit-token@host/mcp"),
            "https://••••:••••••••oken@host/mcp"
        );
    }

    #[test]
    fn a_url_with_nothing_to_hide_is_shown_as_it_was_written() {
        let plain = "https://mcp.example.com/v1/mcp?retry=3";
        assert_eq!(masked_url(plain), plain);
        // The `@` below is in the path, not the credentials: a host that is not
        // there is not a secret.
        let mentioned = "https://example.com/invite/@team";
        assert_eq!(masked_url(mentioned), mentioned);
        // Not a url at all: shown rather than emptied, because it is the
        // declared value and the request is what will fail.
        assert_eq!(masked_url("not a url"), "not a url");
    }

    #[test]
    fn an_address_is_read_as_the_host_and_port_to_reach() {
        assert_eq!(
            address_of("https://mcp.example.com/v1/mcp"),
            Some(("mcp.example.com".to_string(), 443))
        );
        assert_eq!(
            address_of("http://127.0.0.1:34567/mcp"),
            Some(("127.0.0.1".to_string(), 34567))
        );
        // A scheme this client cannot post a message to, and a url with no host,
        // are both refused rather than reported as reachable somewhere.
        assert_eq!(address_of("ftp://example.com/file"), None);
        assert_eq!(address_of("data:text/plain,hi"), None);
    }

    #[test]
    fn truncation_never_cuts_a_multibyte_character() {
        assert_eq!(truncate("abcdef", 4), "abcd...");
        assert_eq!(truncate("héllo", 2), "h...");
        assert_eq!(truncate("héllo", 100), "héllo");
    }
}
