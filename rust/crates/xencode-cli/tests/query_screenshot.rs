//! `xencode query --screenshot` against a real capture tool and a real local
//! model server. The image travels to the loopback server as bytes that were
//! read off the screen; only the model's reading of it is out of scope here.

use std::path::PathBuf;
use std::process::Output;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const MODEL: &str = "llava-vision-test";
const ANSWER: &str = "the screenshot was attached";

struct Fixture {
    _root: tempfile::TempDir,
    config: PathBuf,
    captured: Arc<Mutex<String>>,
    server: tokio::task::JoinHandle<()>,
}

impl Fixture {
    async fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let config = root.path().join("xencode");
        std::fs::create_dir_all(&config).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(String::new()));
        let server = tokio::spawn(serve_once(listener, captured.clone()));
        let settings = serde_json::json!({
            "default_model": MODEL,
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
            serde_json::to_vec_pretty(&settings).unwrap(),
        )
        .unwrap();
        Self {
            _root: root,
            config,
            captured,
            server,
        }
    }

    fn run(&self) -> Output {
        std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
            .args([
                "query",
                "--screenshot",
                "--model",
                MODEL,
                "What is on the screen?",
            ])
            .current_dir(self._root.path())
            .env("XCODE_CONFIG_DIR", &self.config)
            .env_remove("XENCODE_SCREENSHOT_TOOL")
            .output()
            .expect("xencode is built by the time these tests run")
    }
}

/// Answer one Ollama `/api/chat` request in NDJSON, recording the body verbatim
/// so the attached image can be checked as bytes that crossed the network.
async fn serve_once(listener: TcpListener, captured: Arc<Mutex<String>>) {
    // The query path first asks /api/show whether the model reports vision;
    // answer that from the same loopback server, then answer /api/chat.
    loop {
        let (mut sock, _) = match listener.accept().await {
            Ok(pair) => pair,
            Err(_) => break,
        };
        serve_one(&mut sock, captured.clone()).await;
    }
}

async fn serve_one(sock: &mut tokio::net::TcpStream, captured: Arc<Mutex<String>>) {
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
    *captured.lock().unwrap() = String::from_utf8_lossy(&head[header_end..]).into_owned();

    let request_line = header.lines().next().unwrap_or("");
    let (body, content_type) = if request_line.contains("/api/show") {
        (
            serde_json::json!({"capabilities": ["vision", "completion"]}).to_string(),
            "application/json",
        )
    } else {
        let line = serde_json::json!({
            "message": {"role": "assistant", "content": ANSWER},
            "done": true
        })
        .to_string();
        (line, "application/x-ndjson")
    };
    let reply = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        content_type,
        body.len(),
        body
    );
    sock.write_all(reply.as_bytes()).await.unwrap();
    let _ = sock.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_flag_captures_a_real_image_and_attaches_it_to_the_request() {
    if !capture_is_available() {
        eprintln!("skipped: no capture program and no display on this host");
        return;
    }
    let fix = Fixture::new().await;
    let output = fix.run();
    fix.server.abort();

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "exit={} stderr={}",
        output.status,
        stderr
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains(ANSWER),
        "the answer was not printed"
    );

    // The capture really ran and named a file; clean up that file afterwards.
    let line = stderr
        .lines()
        .find(|l| l.starts_with("screenshot: "))
        .unwrap_or_else(|| panic!("no capture line in stderr: {stderr}"));
    let shot = PathBuf::from(
        line.trim_start_matches("screenshot: ")
            .split(" (attached")
            .next()
            .unwrap(),
    );
    assert!(
        shot.is_file(),
        "capture did not leave a file: {}",
        shot.display()
    );
    assert!(std::fs::metadata(&shot).unwrap().len() > 0, "empty capture");
    let captured_bytes = std::fs::read(&shot).unwrap();

    // The loopback server saw the picture as a message part, in the shape the
    // Ollama provider sends: raw base64 under `images` on the final user turn.
    let body = fix.captured.lock().unwrap().clone();
    let request: serde_json::Value = serde_json::from_str(&body).unwrap_or_else(|e| {
        panic!("the request was not JSON: {e}: {body}");
    });
    let user = request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|m| m["role"] == "user")
        .expect("a user turn");
    let images = user["images"].as_array().expect("the images array");
    assert_eq!(images.len(), 1, "exactly one screenshot was attached");
    let data = images[0].as_str().unwrap();
    assert!(
        data.starts_with("/9j/") || data.starts_with("iVBORw0KGgo"),
        "the attached bytes are not a real JPEG or PNG: {}...",
        &data[..data.len().min(24)]
    );
    eprintln!(
        "captured {} bytes at {}; request carried {} base64 characters",
        captured_bytes.len(),
        shot.display(),
        data.len()
    );
    std::fs::remove_file(&shot).unwrap();
}

#[test]
fn the_flag_refuses_capture_when_no_tool_is_available() {
    let root = tempfile::tempdir().unwrap();
    let config = root.path().join("xencode");
    std::fs::create_dir_all(&config).unwrap();
    let settings = serde_json::json!({"default_model": MODEL, "allow_cloud_models": false});
    std::fs::write(
        config.join("config.json"),
        serde_json::to_vec_pretty(&settings).unwrap(),
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(["query", "--screenshot", "what is this?"])
        .current_dir(root.path())
        .env("XCODE_CONFIG_DIR", &config)
        .env("XENCODE_SCREENSHOT_TOOL", "none")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "capture with no tool must fail");
    assert!(
        stderr.contains("no screen capture tool is available"),
        "{stderr}"
    );
    assert!(stderr.contains("XENCODE_SCREENSHOT_TOOL=none"), "{stderr}");
}

fn capture_is_available() -> bool {
    let on_path = |name| {
        std::env::var_os("PATH")
            .map(|paths| {
                std::env::split_paths(&paths).any(|dir| {
                    let candidate = dir.join(name);
                    std::fs::metadata(candidate).is_ok()
                })
            })
            .unwrap_or(false)
    };
    on_path("grim")
        && std::env::var("WAYLAND_DISPLAY")
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
}
