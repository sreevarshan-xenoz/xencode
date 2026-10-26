//! `xencode server` prints what its libraries log.
//!
//! The server's crates report through `tracing`: sign-ins, dropped peers,
//! plugin events, and an audit log that could not be written. These tests start
//! the real binary, make that last one happen for real (the audit path is a
//! directory, which cannot be opened as a file), and read the server's
//! standard error for the warning.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_xencode"))
}

/// A port nothing is listening on right now.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// One running server, stopped when dropped, with its standard error read on
/// a thread so a full pipe can never stall it.
struct Server {
    child: Child,
    port: u16,
    stderr: mpsc::Receiver<String>,
    _home: tempfile::TempDir,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    fn start(audit_path: &std::path::Path, rust_log: Option<&str>) -> Server {
        let home = tempfile::tempdir().unwrap();
        let port = free_port();
        let mut command = Command::new(binary());
        command
            .args(["server", "--host", "127.0.0.1", "--port", &port.to_string()])
            .arg("--audit-path")
            .arg(audit_path)
            .env("XCODE_CONFIG_DIR", home.path().join("config"))
            .env("HOME", home.path())
            .env("USERPROFILE", home.path())
            .env_remove("RUST_LOG")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if let Some(level) = rust_log {
            command.env("RUST_LOG", level);
        }
        let mut child = command.spawn().expect("the xencode binary starts");
        let (tx, rx) = mpsc::channel();
        let stderr = child.stderr.take().unwrap();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let server = Server {
            child,
            port,
            stderr: rx,
            _home: home,
        };
        let deadline = Instant::now() + Duration::from_secs(60);
        while TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(
                Instant::now() < deadline,
                "the server never started listening"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
        server
    }

    /// One HTTP/1.1 request; returns the response body.
    fn post(&self, path: &str, token: Option<&str>, body: &str) -> String {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        let auth = token
            .map(|t| format!("Authorization: Bearer {t}\r\n"))
            .unwrap_or_default();
        write!(
            stream,
            "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
             {auth}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(
            response.starts_with("HTTP/1.1 200"),
            "{path} answered: {}",
            response.lines().next().unwrap_or("")
        );
        response
            .split_once("\r\n\r\n")
            .map(|(_, b)| b.to_string())
            .unwrap_or_default()
    }

    /// Sign in and create a session, which writes the audit log.
    fn create_session(&self) {
        let login = self.post("/auth/login", None, r#"{"username":"tester"}"#);
        let login: serde_json::Value = serde_json::from_str(&login).expect("login answers JSON");
        let token = login["token"].as_str().expect("login returns a token");
        self.post("/sessions/create", Some(token), "{}");
    }

    /// Standard error lines that arrive within `wait`.
    fn stderr_lines(&self, wait: Duration) -> Vec<String> {
        let deadline = Instant::now() + wait;
        let mut lines = Vec::new();
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            match self.stderr.recv_timeout(left) {
                Ok(line) => lines.push(line),
                Err(_) => break,
            }
        }
        lines
    }
}

#[test]
fn an_audit_log_that_cannot_be_opened_is_reported_on_standard_error() {
    let audit_dir = tempfile::tempdir().unwrap();
    let server = Server::start(audit_dir.path(), None);
    server.create_session();

    let lines = server.stderr_lines(Duration::from_secs(5));
    let warning = lines
        .iter()
        .find(|l| l.contains("audit sink disabled: cannot open"))
        .unwrap_or_else(|| panic!("no audit warning on standard error; got {lines:?}"));
    assert!(warning.contains("WARN"), "logged as a warning: {warning}");
}

#[test]
fn rust_log_sets_how_much_the_server_reports() {
    let audit_dir = tempfile::tempdir().unwrap();
    let server = Server::start(audit_dir.path(), Some("error"));
    server.create_session();

    let lines = server.stderr_lines(Duration::from_secs(5));
    assert!(
        !lines.iter().any(|l| l.contains("audit sink disabled")),
        "RUST_LOG=error leaves the warning out; got {lines:?}"
    );
}
