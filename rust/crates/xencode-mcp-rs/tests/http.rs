//! End-to-end tests over the Streamable HTTP transport.
//!
//! The client is real, reqwest is real, and the bytes cross a real TCP socket on
//! 127.0.0.1. What is scripted is the peer: a small HTTP/1.1 server in the test
//! process that answers the JSON-RPC it is posted, because a hosted third-party
//! MCP server cannot be reached from a test suite that has to run offline. Each
//! test also reads back what the peer saw, so the headers the client sent —
//! session id, protocol revision, authorization — are checked rather than
//! assumed.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use xencode_mcp_rs::{McpClient, McpError, ServerSpec};

/// Which spelling of answer the peer gives: one JSON body, or a stream the
/// server keeps open after it has said what it came to say.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Json,
    /// Answers as `text/event-stream` and then holds the connection open.
    EventStream,
    /// Refuses the request the way a server behind a bad deployment does.
    Failing,
}

/// Everything the peer saw, so a test can assert on the request as well as the
/// response.
#[derive(Clone, Default)]
struct Seen {
    /// `(method, headers)` per request received, in arrival order.
    requests: Vec<(String, Vec<(String, String)>)>,
}

const SESSION: &str = "fixture-session-1";
const TOKEN: &str = "sk-test-77-not-a-real-token";
/// The revision the peer answers the handshake with. Every request after it
/// should name it.
const NEGOTIATED: &str = "2025-06-18";

/// Serve one request per connection, the way a short-lived HTTP client call
/// arrives. The peer stops when `stop`'s sender is dropped, which also closes
/// the listener: a test that wants its server to go away part-way through just
/// drops it.
async fn serve(
    listener: tokio::net::TcpListener,
    mode: Mode,
    seen: Arc<Mutex<Seen>>,
    mut stop: tokio::sync::watch::Receiver<bool>,
) {
    loop {
        let accepted = tokio::select! {
            stopped = stop.changed() => match stopped {
                Ok(_) => continue,
                Err(_sender_gone) => return,
            },
            accepted = listener.accept() => accepted,
        };
        let Ok((socket, _)) = accepted else {
            return;
        };
        let seen = Arc::clone(&seen);
        tokio::spawn(async move {
            let (read, write) = socket.into_split();
            let mut reader = BufReader::new(read);
            let mut writer = write;
            while let Some((headers, message)) = read_request(&mut reader).await {
                let method = message
                    .get("method")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                seen.lock()
                    .expect("one peer")
                    .requests
                    .push((method.clone(), headers.clone()));
                let id = message.get("id").cloned();
                let extra = session_header(&method);
                let Some(id) = id else {
                    // A notification: the endpoint acknowledges and says nothing.
                    let _ = write_empty(&mut writer, 202, &extra).await;
                    continue;
                };
                let body =
                    json!({"jsonrpc":"2.0","id":id,"result":result_for(&method)}).to_string();
                match mode {
                    Mode::Json => {
                        let mut headers = vec![
                            "content-type: application/json".to_string(),
                            format!("content-length: {}", body.len()),
                            "connection: close".to_string(),
                        ];
                        headers.extend(extra);
                        if write_response(&mut writer, 200, &headers, body.as_bytes())
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    Mode::EventStream => {
                        let frame = format!(": keep-alive\n\nevent: message\ndata: {body}\n\n");
                        let mut headers = vec![
                            "content-type: text/event-stream".to_string(),
                            // No content-length: the body is whatever arrives
                            // before the connection closes, which is how a
                            // server that intends to keep talking answers.
                            "connection: close".to_string(),
                        ];
                        headers.extend(extra);
                        if write_response(&mut writer, 200, &headers, frame.as_bytes())
                            .await
                            .is_err()
                        {
                            return;
                        }
                        // Held open on purpose. A client that waits for the body
                        // to end rather than for its own answer would fail the
                        // test that follows.
                        tokio::time::sleep(Duration::from_secs(3)).await;
                    }
                    Mode::Failing => {
                        let body = "{\"error\":\"upstream deployment is starting\"}";
                        let headers = vec![
                            "content-type: application/json".to_string(),
                            format!("content-length: {}", body.len()),
                            "connection: close".to_string(),
                        ];
                        let _ = write_response(&mut writer, 503, &headers, body.as_bytes()).await;
                    }
                }
                // One request per connection, then the peer hangs up.
                return;
            }
        });
    }
}

fn result_for(method: &str) -> Value {
    match method {
        "initialize" => json!({
            "protocolVersion": NEGOTIATED,
            // Resources offered, prompts deliberately not.
            "capabilities": {"tools": {}, "resources": {"listChanged": false}},
            "serverInfo": {"name": "hosted", "version": "1.0"}
        }),
        "tools/list" => json!({"tools": [
            {"name": "now", "description": "The server's own time",
             "inputSchema": {"type": "object"}}
        ]}),
        "resources/list" => json!({"resources": [
            {"uri": "hosted://queue", "name": "queue", "mimeType": "text/plain"}
        ]}),
        "resources/read" => json!({"contents": [
            {"uri": "hosted://queue", "mimeType": "text/plain", "text": "three builds queued"}
        ]}),
        _ => json!({}),
    }
}

/// The session id is only issued once, by the handshake response.
fn session_header(method: &str) -> Vec<String> {
    match method {
        "initialize" => vec![format!("mcp-session-id: {SESSION}")],
        _ => Vec::new(),
    }
}

async fn read_request<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
) -> Option<(Vec<(String, String)>, Value)> {
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await.ok()? == 0 {
            return None;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            let name = name.trim().to_lowercase();
            if name == "content-length" {
                length = value.trim().parse().unwrap_or(0);
            }
            headers.push((name, value.trim().to_string()));
        }
    }
    if length == 0 {
        return Some((headers, Value::Null));
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).await.ok()?;
    let text = String::from_utf8_lossy(&body).to_string();
    Some((
        headers,
        serde_json::from_str::<Value>(&text).unwrap_or(Value::Null),
    ))
}

/// Status line, the given header lines, the blank line that ends them, then the
/// body. Every header the peer writes goes through here, because a response
/// whose headers are not terminated is not a response an HTTP client can read.
async fn write_response<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    status: u16,
    headers: &[String],
    body: &[u8],
) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {status} {}\r\n", reason(status));
    for line in headers {
        head.push_str(line);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    writer.write_all(head.as_bytes()).await?;
    writer.write_all(body).await
}

/// An answer with no body: 202 to a notification, which is the endpoint's way of
/// saying it noted the message and has nothing to say back.
async fn write_empty<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    status: u16,
    headers: &[String],
) -> std::io::Result<()> {
    let mut all = headers.to_vec();
    all.push("content-length: 0".to_string());
    all.push("connection: close".to_string());
    write_response(writer, status, &all, b"").await
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

/// Bind a real listener and hand back what the client needs to reach it, plus
/// the sender that keeps the peer alive: dropping it stops the server.
async fn endpoint(mode: Mode) -> (String, Arc<Mutex<Seen>>, tokio::sync::watch::Sender<bool>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a local port");
    let addr = listener.local_addr().expect("the port we got");
    let seen = Arc::new(Mutex::new(Seen::default()));
    let (stop, running) = tokio::sync::watch::channel(false);
    tokio::spawn(serve(listener, mode, Arc::clone(&seen), running));
    (format!("http://{addr}/mcp"), seen, stop)
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

/// A hosted server, reached over HTTP with a token in a header.
#[tokio::test]
async fn a_server_is_connected_over_http_and_the_token_reaches_it() {
    let (url, seen, _stop) = endpoint(Mode::Json).await;
    let client = McpClient::start(
        &ServerSpec::http("hosted", &url).header("authorization", format!("Bearer {TOKEN}")),
        Duration::from_secs(5),
    )
    .await
    .expect("the handshake should complete over HTTP");

    // The declared capabilities are read from the response exactly as from a
    // pipe, and the ones the server left out are refused without being sent.
    assert!(client.capabilities().tools);
    assert!(client.capabilities().resources);
    assert!(!client.capabilities().prompts);
    assert_eq!(
        client
            .list_tools()
            .await
            .expect("tools/list over http")
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["now"]
    );
    let body = client
        .read_resource("hosted://queue")
        .await
        .expect("resources/read over http");
    assert_eq!(body[0].rendered(), "three builds queued");
    match client
        .list_prompts()
        .await
        .expect_err("the server did not declare prompts")
    {
        McpError::NotOffered { feature, .. } => assert_eq!(feature, "prompts"),
        other => panic!("expected a not-offered refusal, got {other:?}"),
    }

    // What the peer saw, taken out of its log before anything else is awaited:
    // the guard is a plain mutex and the peer wants the same lock to answer the
    // shutdown request.
    let observed = seen.lock().expect("the peer's log").requests.clone();
    client.shutdown().await;

    let methods: Vec<&str> = observed.iter().map(|(method, _)| method.as_str()).collect();
    assert_eq!(
        methods,
        [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "resources/read"
        ],
        "the notification is answered with 202 and no body, which the client \
         must read as success rather than as a missing response, and prompts/list \
         is never sent to a server that did not declare it"
    );

    // Every request carries the token; the requests after the handshake also
    // carry the session id the peer issued and the revision it answered with.
    let bearer = format!("Bearer {TOKEN}");
    for (method, headers) in &observed {
        assert_eq!(
            header(headers, "authorization"),
            Some(bearer.as_str()),
            "{method}"
        );
        assert_eq!(header(headers, "content-type"), Some("application/json"));
        if method == "initialize" {
            assert_eq!(
                header(headers, "mcp-session-id"),
                None,
                "there is no session to name yet"
            );
        } else {
            assert_eq!(header(headers, "mcp-session-id"), Some(SESSION), "{method}");
            assert_eq!(
                header(headers, "mcp-protocol-version"),
                Some(NEGOTIATED),
                "{method}"
            );
        }
    }
}

/// A server may answer in a `text/event-stream` and keep the connection open
/// afterwards. The answer is what we came for.
#[tokio::test]
async fn an_answer_arriving_on_an_open_stream_is_taken_and_the_stream_left() {
    let (url, _seen, _stop) = endpoint(Mode::EventStream).await;
    let started = std::time::Instant::now();
    let client = McpClient::start(&ServerSpec::http("streamed", &url), Duration::from_secs(20))
        .await
        .expect("the handshake should complete over an event stream");
    let tools = client
        .list_tools()
        .await
        .expect("tools/list on an event stream");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "now");
    // The peer holds its connection open for three seconds on purpose. Returning
    // in far less than that is the evidence the client stopped at its answer.
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "waited {elapsed:?} for a stream that stays open"
    );
    // The session is what let the next request work on a fresh connection.
    let resources = client
        .list_resources()
        .await
        .expect("a second request after the stream was dropped");
    assert_eq!(resources[0].uri, "hosted://queue");
    client.shutdown().await;
}

/// A deployment that refuses the request is reported as the server's own error,
/// with its status and its words.
#[tokio::test]
async fn a_server_that_refuses_over_http_is_reported_with_its_status() {
    let (url, _seen, _stop) = endpoint(Mode::Failing).await;
    let error = McpClient::start(&ServerSpec::http("refusing", &url), Duration::from_secs(5))
        .await
        .err()
        .expect("503 is not a handshake");
    match error {
        McpError::Server {
            server,
            code,
            message,
        } => {
            assert_eq!(server, "refusing");
            assert_eq!(code, 503);
            assert!(message.contains("upstream deployment"), "{message}");
        }
        other => panic!("expected a server error, got {other:?}"),
    }
}

/// Nothing about a request may repeat the secret back: not the endpoint line a
/// status screen shows, not an error about a failed one.
#[tokio::test]
async fn a_token_never_comes_back_out_of_an_http_session() {
    let (url, _seen, stop) = endpoint(Mode::Json).await;
    // The same secret in both places a caller might put it: a header, and the
    // credentials part of the url.
    let credentialed = url.replace("127.0.0.1", &format!("{TOKEN}@127.0.0.1"));
    let spec = ServerSpec::http("private", &credentialed).header("authorization", TOKEN);
    let client = McpClient::start(&spec, Duration::from_secs(5))
        .await
        .expect("credentials in the url still reach the same server");

    // The endpoint a person is shown names where the server is, not what it was
    // authenticated with.
    let shown = client.endpoint();
    assert!(!shown.contains(TOKEN), "{shown}");
    assert!(shown.contains("••••"), "{shown}");
    assert!(shown.contains("/mcp"), "{shown}");

    // Stop the peer, so the next request fails and the failure is the text we
    // are about to show someone.
    drop(stop);
    let error = client
        .list_tools()
        .await
        .expect_err("nothing is listening any more");
    let rendered = error.to_string();
    assert!(!rendered.contains(TOKEN), "{rendered}");
    // The address is still there, because it is the useful half.
    assert!(rendered.contains("127.0.0.1"), "{rendered}");
}

/// A port that answers nothing is a failure that says so, not a hang.
#[tokio::test]
async fn an_endpoint_that_is_not_there_fails_by_name_without_waiting_forever() {
    // Bind and drop to learn a port nothing holds.
    let spare = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a free port");
    let addr = spare.local_addr().expect("its address");
    drop(spare);
    let started = std::time::Instant::now();
    let error = McpClient::start(
        &ServerSpec::http("absent", format!("http://{addr}/mcp")),
        Duration::from_secs(5),
    )
    .await
    .err()
    .expect("nothing is listening there");
    assert!(error.to_string().contains("absent"), "{error}");
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "a refused connection should fail at once, not run out the deadline"
    );
}

/// A stream that carries a heartbeat comment and an `event:` field before the
/// answer is still one answer: the framing is not content, and nothing is
/// counted as a line that is not protocol.
#[tokio::test]
async fn stream_frames_are_cut_on_their_own_boundaries() {
    let (url, _seen, _stop) = endpoint(Mode::EventStream).await;
    let client = McpClient::start(&ServerSpec::http("framing", &url), Duration::from_secs(10))
        .await
        .expect("handshake");
    let body = client
        .read_resource("hosted://queue")
        .await
        .expect("resources/read over a stream");
    assert_eq!(body[0].rendered(), "three builds queued");
    // The heartbeat comment and the `event:` field were framing, not content:
    // nothing was counted as a line that is not protocol.
    assert_eq!(client.stray_lines(), 0);
    client.shutdown().await;
}
