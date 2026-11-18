//! `--deep-think <model>` against a real OpenAI-compatible loopback server.
//!
//! The server records the bytes of the request body and answers in the SSE
//! shape `/chat/completions` uses, so the model id asserted below is the one
//! that actually crossed the socket — not a value read back from the config.

use std::path::PathBuf;
use std::process::Output;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

struct Fixture {
    _root: tempfile::TempDir,
    cwd: PathBuf,
    config: PathBuf,
    captured: Arc<Mutex<String>>,
    server: tokio::task::JoinHandle<()>,
}

impl Fixture {
    async fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let cwd = root.path().join("project");
        let config = root.path().join("xencode");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&config).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(String::new()));
        let server = tokio::spawn(serve_openai_once(listener, captured.clone()));

        // The endpoint is on this machine, so a `remote:` model stays local and
        // needs no egress opt-in. The configured default is a different model:
        // the assertion below proves --deep-think, not the default, was sent.
        let body = serde_json::json!({
            "default_model": "small-local-model",
            "ollama_url": "http://127.0.0.1:1",
            "llama_cpp_url": "http://127.0.0.1:1",
            "remote_base_url": format!("http://{addr}/v1"),
            "cache_enabled": false,
            "memory_enabled": false,
            "response_timeout": 10,
            "allow_cloud_models": false,
        });
        std::fs::write(
            config.join("config.json"),
            serde_json::to_vec_pretty(&body).unwrap(),
        )
        .unwrap();

        Self {
            _root: root,
            cwd,
            config,
            captured,
            server,
        }
    }

    fn run(&self, extra: &[&str]) -> Output {
        std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
            .arg("query")
            .args(extra)
            .arg("answer me")
            .current_dir(&self.cwd)
            .env("XCODE_CONFIG_DIR", &self.config)
            .output()
            .expect("xencode is built by the time these tests run")
    }

    fn request_body(&self) -> serde_json::Value {
        let text = self.captured.lock().unwrap().clone();
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("not a JSON request: {text}: {e}"))
    }
}

/// Accept one `/chat/completions` request, keep its whole body, and answer with
/// a two-token stream terminated by the `[DONE]` marker.
async fn serve_openai_once(listener: TcpListener, captured: Arc<Mutex<String>>) {
    let (mut sock, _) = listener.accept().await.unwrap();
    let mut head: Vec<u8> = Vec::new();
    let mut buf = [0u8; 4096];
    let mut header_end = None;
    loop {
        let n = sock.read(&mut buf).await.unwrap();
        if n == 0 {
            break;
        }
        head.extend_from_slice(&buf[..n]);
        if let Some(pos) = head.windows(4).position(|w| w == b"\r\n\r\n") {
            header_end = Some(pos + 4);
            break;
        }
    }
    let header_end = header_end.expect("the request has headers");
    let header = String::from_utf8_lossy(&head[..header_end]).into_owned();
    let content_len: usize = header
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse().ok())
        .unwrap_or(0);
    while head.len() < header_end + content_len {
        let n = sock.read(&mut buf).await.unwrap();
        if n == 0 {
            break;
        }
        head.extend_from_slice(&buf[..n]);
    }
    *captured.lock().unwrap() = String::from_utf8_lossy(&head[header_end..]).into_owned();

    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"deep\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" answer\"}}]}\n\n",
        "data: [DONE]\n\n",
    );
    let reply = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        sse.len(),
        sse
    );
    sock.write_all(reply.as_bytes()).await.unwrap();
    let _ = sock.shutdown().await;
}

/// The explicit override reaches the provider as the requested model id and is
/// announced on stderr. A different configured default and an explicit
/// `--model` are both present, so this can only pass if --deep-think wins.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn query_deep_think_sends_the_requested_model_id() {
    let fixture = Fixture::new().await;
    let out = fixture.run(&[
        "--model",
        "another-small-model",
        "--deep-think",
        "remote:strong-reasoner",
    ]);

    assert!(
        out.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("deep answer"), "stdout: {stdout}");

    let body = fixture.request_body();
    assert_eq!(
        body["model"].as_str(),
        Some("strong-reasoner"),
        "the requested id reached the endpoint: {body}"
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("deep-think: using remote:strong-reasoner"),
        "a clear line said deep-think was used: {stderr}"
    );
    let _ = fixture.server.await;
}
