//! `xencode git commit-message` against a real git repository and a real local
//! model server. The server is a loopback socket that records the request body
//! and answers in Ollama's NDJSON shape, so the prompt and the printed message
//! are checked against bytes that actually crossed the network.

use std::path::PathBuf;
use std::process::Output;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const SUGGESTION: &str = "feat: add an updated line to file.txt";

/// A real git repo with one staged change, an isolated config, and a real
/// loopback model server that captures the request body.
struct Fixture {
    _root: tempfile::TempDir,
    repo: PathBuf,
    config: PathBuf,
    captured: Arc<Mutex<String>>,
    server: tokio::task::JoinHandle<()>,
}

impl Fixture {
    async fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let repo = root.path().join("repo");
        let config = root.path().join("xencode");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::create_dir_all(&config).unwrap();

        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .current_dir(&repo)
                .args(args)
                .output()
                .expect("git runs");
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        git(&["init", "-q", "-b", "main"]);
        git(&["config", "user.name", "tester"]);
        git(&["config", "user.email", "tester@example.invalid"]);
        std::fs::write(repo.join("file.txt"), "hello world\n").unwrap();
        git(&["add", "file.txt"]);
        git(&["commit", "-qm", "initial"]);
        // A real change, really staged: this is what the model must see.
        std::fs::write(repo.join("file.txt"), "hello world updated\n").unwrap();
        git(&["add", "file.txt"]);

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(String::new()));
        let server = tokio::spawn(serve_once(listener, captured.clone(), SUGGESTION));

        let body = serde_json::json!({
            "default_model": "commit-model:latest",
            "ollama_url": format!("http://{addr}"),
            "llama_cpp_url": format!("http://{addr}"),
            "remote_base_url": "",
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
            repo,
            config,
            captured,
            server,
        }
    }

    fn run(&self, extra: &[&str]) -> Output {
        std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
            .args(["git", "commit-message"])
            .args(extra)
            .current_dir(&self.repo)
            .env("XCODE_CONFIG_DIR", &self.config)
            .output()
            .expect("xencode is built by the time these tests run")
    }

    fn request_body(&self) -> String {
        self.captured.lock().unwrap().clone()
    }

    fn log(&self) -> String {
        let out = std::process::Command::new("git")
            .current_dir(&self.repo)
            .args(["log", "--oneline"])
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    fn staged(&self) -> String {
        let out = std::process::Command::new("git")
            .current_dir(&self.repo)
            .args(["diff", "--cached", "--no-color"])
            .output()
            .unwrap();
        String::from_utf8_lossy(&out.stdout).into_owned()
    }
}

/// Answer one Ollama `/api/chat` request with a scripted message, recording the
/// body verbatim. A chunked, close-delimited NDJSON reply is what the provider
/// stack reads, so this exercises the real response parser.
async fn serve_once(listener: TcpListener, captured: Arc<Mutex<String>>, message: &str) {
    let (mut sock, _) = listener.accept().await.unwrap();
    let mut head = Vec::new();
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
    let body = String::from_utf8_lossy(&head[header_end..]).into_owned();
    *captured.lock().unwrap() = body;

    let line = format!(
        "{}\n",
        serde_json::json!({
            "message": {"role": "assistant", "content": message},
            "done": true
        })
    );
    let reply = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        line.len(),
        line
    );
    sock.write_all(reply.as_bytes()).await.unwrap();
    let _ = sock.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_suggestion_prints_the_model_message_and_never_commits_by_itself() {
    let fix = Fixture::new().await;
    let output = fix.run(&[]);
    fix.server.abort();

    assert!(
        output.status.success(),
        "exit={} stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(SUGGESTION),
        "the printed message must be the model's: {stdout}"
    );

    // The prompt really carried the staged diff over the wire.
    let body = fix.request_body();
    assert!(
        body.contains("Staged diff"),
        "the instructions were not in the request: {body}"
    );
    assert!(
        body.contains("diff --git a/file.txt b/file.txt"),
        "the diff header was not in the request: {body}"
    );
    assert!(
        body.contains("+hello world updated"),
        "the staged line was not in the request: {body}"
    );

    // Reporting is not committing: the staged change and the commit count are
    // exactly what they were before.
    assert_eq!(
        fix.log().lines().count(),
        1,
        "a commit happened: {}",
        fix.log()
    );
    assert!(
        fix.staged().contains("+hello world updated"),
        "the staged change was lost"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_commit_flag_commits_the_staged_changes_with_the_suggested_message() {
    let fix = Fixture::new().await;
    let output = fix.run(&["--commit"]);
    fix.server.abort();

    assert!(
        output.status.success(),
        "exit={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let log = fix.log();
    assert!(
        log.contains(SUGGESTION),
        "the new commit does not carry the message: {log}"
    );
    assert_eq!(log.lines().count(), 2, "expected one new commit: {log}");
    assert!(
        fix.staged().is_empty(),
        "the staged changes were not committed"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_model_flag_overrides_the_configured_default() {
    let fix = Fixture::new().await;
    let output = fix.run(&["--model", "other:latest"]);
    fix.server.abort();

    assert!(
        output.status.success(),
        "exit={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let body = fix.request_body();
    assert!(
        body.contains("other:latest"),
        "the override did not reach the model: {body}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nothing_staged_is_reported_honestly_and_exits_nonzero() {
    let fix = Fixture::new().await;
    // Drop the staged change by committing it first: the index is now clean.
    let status = std::process::Command::new("git")
        .current_dir(&fix.repo)
        .args(["commit", "-qm", "take the staged change"])
        .status()
        .unwrap();
    assert!(status.success());

    let output = fix.run(&[]);
    fix.server.abort();
    assert!(
        !output.status.success(),
        "an empty index produced a message"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("nothing is staged"),
        "the failure was not explained: {stderr}"
    );
    assert!(
        fix.request_body().is_empty(),
        "the model was called with nothing staged"
    );
}
