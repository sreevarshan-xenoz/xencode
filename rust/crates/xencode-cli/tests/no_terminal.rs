//! A bare `xencode` in a pipe — a cron line, a CI step, `.output()` in a test —
//! asked for the interactive screen and got a raw errno:
//! `error: No such device or address (os error 6)`. That says neither that the
//! terminal is the problem nor that other commands work without one, so the
//! refusal is now said in those words. Driven through the real binary, because
//! the defect was the message a person actually reads.

use std::path::Path;
use std::process::{Command, Output};

fn xencode(config_dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        // A private config tree, so a refusal that happens before any file is
        // read still cannot reach the developer's own settings.
        .env("XCODE_CONFIG_DIR", config_dir)
        // Piped, not inherited: the point of the case is that stdout is not a
        // terminal.
        .output()
        .expect("xencode is built by the time these tests run")
}

fn temp_config_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-no-terminal-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn refused(config_dir: &Path, args: &[&str]) -> String {
    let out = xencode(config_dir, args);
    assert!(
        !out.status.success(),
        "asking for the screen without a terminal must not look like a success: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stderr.contains("interactive screen needs a terminal"),
        "{stderr}"
    );
    assert!(
        !stderr.contains("os error 6"),
        "the bare errno is the thing being replaced: {stderr}"
    );
    // It has to name something that works here, and `query` is the one a
    // one-off question needs.
    assert!(stderr.contains("xencode query"), "{stderr}");
    assert!(stderr.contains("xencode doctor"), "{stderr}");
    stderr
}

#[test]
fn a_bare_xencode_with_no_terminal_says_what_it_needed() {
    let dir = temp_config_dir("bare");
    refused(&dir, &[]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_explicit_tui_command_refuses_the_same_way() {
    let dir = temp_config_dir("tui");
    refused(&dir, &["tui"]);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The other half: a command that never needed a terminal is unaffected by the
/// check, so the fix cannot have swallowed the non-interactive path it names.
#[test]
fn a_command_that_works_without_a_terminal_still_works() {
    let dir = temp_config_dir("paths");
    let out = xencode(&dir, &["paths"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.stdout.is_empty(), "the listing came back empty");
    std::fs::remove_dir_all(&dir).unwrap();
}
