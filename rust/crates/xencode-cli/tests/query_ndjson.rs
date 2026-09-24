//! `xencode query --format ndjson` written by the real binary, read back the
//! way a script reads it: split stdout on newlines, parse each line.
//!
//! The model server is a port that was bound and released, so the connection
//! is refused the moment it is tried. That is deliberate — a run that reaches
//! an answer would depend on someone having a model loaded, and the shape of a
//! stream that ends in failure is the part worth pinning.

use std::path::Path;
use std::process::Output;

/// A port nothing is listening on: bound, read, dropped. On loopback the
/// kernel answers with a refusal straight away, so the run fails fast and
/// offline.
fn dead_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("loopback bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

/// A configuration of our own, in a directory only this test sees, so the
/// developer's `~/.xencode` is neither read nor written.
fn isolated_config(dir: &Path) -> std::path::PathBuf {
    let config_dir = dir.join("xencode");
    std::fs::create_dir_all(&config_dir).unwrap();
    let port = dead_port();
    let body = serde_json::json!({
        "default_model": "test-model:latest",
        "ollama_url": format!("http://127.0.0.1:{port}"),
        "llama_cpp_url": format!("http://127.0.0.1:{port}"),
        "remote_base_url": "",
        "cache_enabled": false,
        "memory_enabled": false,
        "response_timeout": 5,
        "allow_cloud_models": false,
    });
    std::fs::write(
        config_dir.join("config.json"),
        serde_json::to_vec_pretty(&body).unwrap(),
    )
    .unwrap();
    config_dir
}

fn query(config_dir: &Path, extra: &[&str]) -> Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(["query", "what is two plus two?", "--no-cache"])
        .args(extra)
        .env("XCODE_CONFIG_DIR", config_dir)
        .output()
        .expect("xencode is built by the time these tests run")
}

fn events(stdout: &[u8]) -> Vec<serde_json::Value> {
    let text = std::str::from_utf8(stdout).expect("stdout is UTF-8");
    assert!(
        !text.contains("\n\n"),
        "the stream held an empty line, which a line reader would skip in silence"
    );
    text.lines()
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("line {line:?} is not one JSON object: {e}"))
        })
        .collect()
}

#[test]
fn a_run_that_cannot_reach_a_server_still_reports_what_it_was_about_to_do() {
    let dir = tempfile::TempDir::new().unwrap();
    let config = isolated_config(dir.path());
    let output = query(&config, &["--format", "ndjson"]);
    assert!(
        !output.status.success(),
        "a refused server produced an answer"
    );
    let lines = events(&output.stdout);
    assert_eq!(lines.len(), 2, "expected a start line and a final line");
    assert_eq!(lines[0]["type"], "start");
    assert_eq!(lines[0]["model"], "test-model:latest");
    assert_eq!(lines[0]["provider"], "ollama");
    assert_eq!(lines[0]["source"], "local");
    assert!(
        lines[0]["session"].is_null(),
        "memory was off, so no session"
    );
    assert_eq!(lines[1]["type"], "error");
    assert!(
        lines[1]["message"]
            .as_str()
            .unwrap()
            .contains("Query failed"),
        "{}",
        lines[1]
    );
    // The human-readable failure stays on stderr, out of the parsed stream.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("error:"), "{stderr}");
}

#[test]
fn plain_text_output_is_unchanged_by_the_flag_existing_at_all() {
    let dir = tempfile::TempDir::new().unwrap();
    let config = isolated_config(dir.path());
    let output = query(&config, &[]);
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains('"'),
        "the default format started emitting JSON on its own: {stdout}"
    );
}

#[test]
fn an_unknown_format_is_refused_before_anything_runs() {
    let dir = tempfile::TempDir::new().unwrap();
    let config = isolated_config(dir.path());
    let output = query(&config, &["--format", "yaml"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("possible values"), "{stderr}");
    assert!(stderr.contains("ndjson"), "{stderr}");
}
