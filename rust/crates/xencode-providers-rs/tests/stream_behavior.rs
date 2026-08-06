//! Behavioral tests for `generate_stream`'s retry semantics.
//!
//! Unlike the unit tests (which drive `retry_async_with_guard` directly),
//! these tests exercise the real HTTP path: `generate_stream` →
//! `generate_stream_inner` → `generate_stream_ollama` against a live local
//! TCP server that speaks chunked NDJSON like Ollama's `/api/chat`.
//!
//! They verify the full lifecycle:
//! - tokens are delivered exactly once (no duplicate output on retry)
//! - a mid-stream disconnect surfaces the error without re-running
//! - pre-emission 5xx failures are retried, then succeed

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use xencode_models_rs::OllamaClient;
use xencode_providers_rs::retry::RetryConfig;
use xencode_providers_rs::{ChatMessage, ProviderManager};

fn fast_config() -> RetryConfig {
    RetryConfig {
        max_retries: 2,
        base_delay_ms: 5,
        max_delay_ms: 20,
        backoff_factor: 2.0,
    }
}

fn messages() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: "hi".to_string(),
    }]
}

/// One Ollama-style NDJSON line: `{"message":{"role":"assistant","content":"..."},"done":bool}`
fn ndjson_line(content: &str, done: bool) -> String {
    format!(
        "{{\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"done\":{}}}\n",
        content, done
    )
}

/// Scripted local server: each connection reads the request, then responds
/// according to `plan` (one entry per connection). Returns the base URL and a
/// connection counter.
async fn spawn_server(
    plan: Vec<PlanStep>,
) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counter = Arc::new(AtomicUsize::new(0));

    let plan = Arc::new(plan);
    let counter_task = counter.clone();
    tokio::spawn(async move {
        let mut idx = 0usize;
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { break };
            counter_task.fetch_add(1, Ordering::SeqCst);

            // Drain the request (headers + small body), with a timeout so a
            // stalled client can't hang the test binary forever.
            let mut buf = vec![0u8; 4096];
            let mut read = 0usize;
            let read_loop = async {
                loop {
                    let n = sock.read(&mut buf[read..]).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    read += n;
                    // Headers end at the first "\r\n\r\n" anywhere in the buffer
                    // (headers+body may arrive in a single read).
                    if buf[..read].windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                    if read == buf.len() {
                        break;
                    }
                }
            };
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), read_loop).await;
            let step = plan.get(idx.min(plan.len() - 1)).cloned().unwrap_or(PlanStep::ChunkedClose);
            idx += 1;
            let bytes = step.to_http();
            let _ = sock.write_all(&bytes).await;
            let _ = sock.flush().await;
            if matches!(step, PlanStep::ChunkedClose) {
                // Drop without the terminating chunk → mid-stream EOF error.
                drop(sock);
            } else {
                let _ = sock.shutdown().await;
            }
        }
    });

    (format!("http://{}", addr), counter)
}

#[derive(Clone)]
enum PlanStep {
    /// HTTP 503 with a JSON error body.
    Http503,
    /// Chunked NDJSON: stream one token, then close without the final chunk.
    ChunkedClose,
    /// Chunked NDJSON: stream the given lines, then terminate cleanly.
    ChunkedOk(Vec<(String, bool)>),
}

impl PlanStep {
    fn to_http(&self) -> Vec<u8> {
        match self {
            PlanStep::Http503 => {
                let body = "{\"error\":\"temporarily unavailable\"}";
                format!(
                    "HTTP/1.1 503 Service Unavailable\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .into_bytes()
            }
            PlanStep::ChunkedClose => {
                let body = ndjson_line("Hello", false);
                format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: application/x-ndjson\r\n\
                     Transfer-Encoding: chunked\r\n\
                     Connection: close\r\n\r\n\
                     {:x}\r\n{}\r\n",
                    body.len(),
                    body
                )
                .into_bytes()
            }
            PlanStep::ChunkedOk(lines) => {
                let body: String = lines
                    .iter()
                    .map(|(c, d)| ndjson_line(c, *d))
                    .collect();
                format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: application/x-ndjson\r\n\
                     Transfer-Encoding: chunked\r\n\
                     Connection: close\r\n\r\n\
                     {:x}\r\n{}\r\n0\r\n\r\n",
                    body.len(),
                    body
                )
                .into_bytes()
            }
        }
    }
}

/// The core scenario this fix targets: a stream that fails *mid-way* (after
/// tokens were delivered) must NOT be retried — retrying would re-deliver the
/// same tokens (duplicated output). The connection counter proves the
/// operation ran exactly once, and the tokens prove no duplicates.
#[tokio::test]
async fn mid_stream_disconnect_is_not_retried_and_tokens_delivered_once() {
    let (base_url, counter) = spawn_server(vec![PlanStep::ChunkedClose]).await;
    let manager = ProviderManager::new(
        OllamaClient::new(&base_url, 5),
        None,
        None,
        None,
        None,
    );
    let manager = manager.with_retry_config(fast_config());

    let mut tokens: Vec<String> = Vec::new();
    let result = manager
        .generate_stream("test-model", &messages(), |t| tokens.push(t.to_string()))
        .await;

    // Tokens that were delivered before the failure must be delivered exactly
    // once — a retry would have delivered "Hello" twice.
    assert_eq!(tokens, vec!["Hello".to_string()]);
    // The failure is surfaced as-is (not masked by a retried success).
    assert!(result.is_err(), "expected error, got {result:?}");
    assert_eq!(counter.load(Ordering::SeqCst), 1, "operation must not re-run after emission");
}

/// Pre-emission 5xx failures must be retried (with backoff), then succeed.
/// This also verifies HTTP error statuses are actually surfaced as errors —
/// a swallowed 503 would return Ok("") with only one connection.
#[tokio::test]
async fn pre_emission_503_is_retried_then_succeeds() {
    let (base_url, counter) = spawn_server(vec![
        PlanStep::Http503,
        PlanStep::Http503,            PlanStep::ChunkedOk(vec![("Hello".to_string(), false), (" world".to_string(), true)]),
    ])
    .await;
    let manager = ProviderManager::new(
        OllamaClient::new(&base_url, 5),
        None,
        None,
        None,
        None,
    );
    let manager = manager.with_retry_config(fast_config());

    let mut tokens: Vec<String> = Vec::new();
    let result = manager
        .generate_stream("test-model", &messages(), |t| tokens.push(t.to_string()))
        .await;

    let output = result.expect("stream should succeed after retries");
    assert_eq!(output, "Hello world");
    assert_eq!(tokens, vec!["Hello".to_string(), " world".to_string()]);
    // initial + 2 retries = 3 connections
    assert_eq!(counter.load(Ordering::SeqCst), 3);
}

/// Clean stream: tokens delivered in order, full response assembled, exactly
/// one connection.
#[tokio::test]
async fn clean_stream_delivers_tokens_in_order() {
    let (base_url, counter) = spawn_server(vec![PlanStep::ChunkedOk(vec![
        ("Hel".to_string(), false),
        ("lo".to_string(), false),
        ("!".to_string(), true),
    ])])
    .await;
    let manager = ProviderManager::new(
        OllamaClient::new(&base_url, 5),
        None,
        None,
        None,
        None,
    );
    let manager = manager.with_retry_config(fast_config());

    let mut tokens: Vec<String> = Vec::new();
    let result = manager
        .generate_stream("test-model", &messages(), |t| tokens.push(t.to_string()))
        .await;

    let output = result.expect("clean stream should succeed");
    assert_eq!(output, "Hello!");
    assert_eq!(tokens, vec!["Hel".to_string(), "lo".to_string(), "!".to_string()]);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}
