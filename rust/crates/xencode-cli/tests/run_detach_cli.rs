//! EN-4: `xencode run --detach` starts a run that outlives the command and
//! finishes on its own, on Windows as on Unix. The model server address has
//! nothing listening on it, so the run's one model call fails at once and
//! the run ends with that failure in words.

use std::path::Path;
use std::process::{Command, Output};
use std::time::{Duration, Instant};

fn settings() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "default_model": "llamacpp:none",
        "llama_cpp_url": "http://127.0.0.1:9",
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    dir
}

fn xencode(project: &Path, config: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .current_dir(project)
        .env("XCODE_CONFIG_DIR", config)
        .output()
        .unwrap()
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

#[test]
fn a_detached_run_returns_at_once_and_finishes_on_its_own() {
    let project = tempfile::tempdir().unwrap();
    let config = settings();
    let started = Instant::now();
    let out = xencode(
        project.path(),
        config.path(),
        &[
            "run",
            "write a note",
            "--detach",
            "--model",
            "llamacpp:none",
            "--llamacpp-url",
            "http://127.0.0.1:9",
        ],
    );
    let said = text(&out);
    assert!(out.status.success(), "{said}");
    assert!(started.elapsed() < Duration::from_secs(30), "it waited for the run");
    let line = said
        .lines()
        .find(|l| l.starts_with("detached run "))
        .unwrap_or_else(|| panic!("no run id in: {said}"));
    let id = line.split_whitespace().nth(2).unwrap().to_string();
    assert!(line.contains("(pid "), "{line}");

    let start = Instant::now();
    let shown = loop {
        let shown = text(&xencode(project.path(), config.path(), &["run", "--show", &id]));
        if shown.contains("status: finished") {
            break shown;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the run never finished: {shown}"
        );
        std::thread::sleep(Duration::from_millis(250));
    };
    assert!(shown.contains("prompt: write a note"), "{shown}");
    let log = text(&xencode(project.path(), config.path(), &["run", "--log", &id]));
    assert!(
        log.contains("127.0.0.1:9"),
        "the log says the model server could not be reached: {log}"
    );
}
