//! Integration tests for `xencode remote add|list|use|forget|show` (`L-2`).
//!
//! Driven through the compiled binary with an isolated `$XCODE_CONFIG_DIR`,
//! verifying that profiles are stored, retrieved, chosen, displayed, and
//! removed with the exact permissions, formats, and safety checks required.

use std::path::{Path, PathBuf};
use std::process::Output;

fn xencode(config_dir: &Path, args: &[&str]) -> Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .env("XCODE_CONFIG_DIR", config_dir)
        .output()
        .expect("xencode binary is built for tests")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-remotes-cli-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn remote_list_when_empty_prints_helpful_guidance() {
    let dir = temp_dir("empty");
    let out = xencode(&dir, &["remote", "list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("No remote profiles recorded"),
        "expected empty message, got: {stdout}"
    );
    assert!(stdout.contains("xencode remote add <name> <user@host>"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn remote_add_list_show_roundtrip() {
    let dir = temp_dir("roundtrip");

    // 1. Add a remote profile
    let out = xencode(
        &dir,
        &[
            "remote",
            "add",
            "lab",
            "dev@lab-server:2222",
            "--runtime",
            "llama.cpp",
            "--model",
            "qwen2.5:7b",
        ],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Recorded remote profile `lab` in"));

    // 2. Profile file is on disk and has 0600 mode on Unix
    let profile_file = dir.join("remotes/lab.json");
    assert!(
        profile_file.exists(),
        "profile file was not written to remotes/lab.json"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&profile_file)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "profile file must be mode 0600, was: {mode:o}");
    }

    // 3. List contains the profile
    let out = xencode(&dir, &["remote", "list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("lab"));
    assert!(stdout.contains("dev@lab-server · port 2222 · llama.cpp · qwen2.5:7b · local 18100"));

    // 4. Show the profile by name
    let out = xencode(&dir, &["remote", "show", "lab"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Remote profile `lab`"));
    assert!(stdout.contains("destination: dev@lab-server"));
    assert!(stdout.contains("ssh port:    2222"));
    assert!(stdout.contains("runtime:     llama.cpp"));
    assert!(stdout.contains("model:       qwen2.5:7b"));
    assert!(stdout.contains("local port:  18100"));

    // 5. Add duplicate without force is refused
    let out = xencode(&dir, &["remote", "add", "lab", "other@server:22"]);
    assert!(
        !out.status.success(),
        "duplicate add without force must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--force"),
        "refusal did not mention --force: {stderr}"
    );

    // 6. Add duplicate with force succeeds and updates destination
    let out = xencode(
        &dir,
        &["remote", "add", "lab", "other@server:22", "--force"],
    );
    assert!(
        out.status.success(),
        "duplicate add with --force failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = xencode(&dir, &["remote", "show", "lab"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("destination: other@server"));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn remote_use_marks_active_and_forget_clears_it() {
    let dir = temp_dir("use-forget");

    // Add two profiles: primary and secondary
    xencode(&dir, &["remote", "add", "primary", "work@box1:22"]);
    xencode(
        &dir,
        &[
            "remote",
            "add",
            "secondary",
            "work@box2:22",
            "--runtime",
            "ollama",
        ],
    );

    // Initially, show without name fails because no profile is active
    let out = xencode(&dir, &["remote", "show"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("No active remote profile"));

    // Set primary as active
    let out = xencode(&dir, &["remote", "use", "primary"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Active remote profile set to `primary`"));

    // Now show with no args shows primary
    let out = xencode(&dir, &["remote", "show"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Remote profile `primary` [active]"));

    // List shows active marker '*' beside primary
    let out = xencode(&dir, &["remote", "list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("* primary"));
    assert!(stdout.contains("  secondary"));

    // Setting a nonexistent profile as active fails
    let out = xencode(&dir, &["remote", "use", "nonexistent"]);
    assert!(!out.status.success());

    // Forget primary: removes file AND clears active pointer
    let out = xencode(&dir, &["remote", "forget", "primary"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Removed remote profile `primary` (active profile cleared)"));

    // Active pointer is gone
    let out = xencode(&dir, &["remote", "show"]);
    assert!(!out.status.success());

    // Secondary still exists
    let out = xencode(&dir, &["remote", "show", "secondary"]);
    assert!(out.status.success());

    // Forget secondary
    let out = xencode(&dir, &["remote", "forget", "secondary"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Removed remote profile `secondary`"));

    // Forgetting already forgotten profile errors
    let out = xencode(&dir, &["remote", "forget", "secondary"]);
    assert!(!out.status.success());

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn remote_add_refuses_unsafe_inputs() {
    let dir = temp_dir("validation");

    // Host with proxy command option
    let out = xencode(&dir, &["remote", "add", "pwn", "-oProxyCommand=touch"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("is not a host xencode will hand to `ssh`"));

    // Path traversal name
    let out = xencode(&dir, &["remote", "add", "../bad", "host"]);
    assert!(!out.status.success());

    // Port 0
    let out = xencode(&dir, &["remote", "add", "zero", "host:0"]);
    assert!(!out.status.success());

    // Invalid runtime
    let out = xencode(
        &dir,
        &["remote", "add", "badrt", "host", "--runtime", "unknown"],
    );
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("is not a runtime xencode can start"));

    // Name with leading dash
    let out = xencode(&dir, &["remote", "add", "-bad", "host"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("is not a name xencode can store a host under"));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn remote_list_reports_unreadable_profile_without_crashing() {
    let dir = temp_dir("unreadable");
    xencode(&dir, &["remote", "add", "valid", "work@box:22"]);

    // Create an unreadable/broken json file in remotes/
    let broken_file = dir.join("remotes/damaged.json");
    std::fs::write(&broken_file, "{ invalid json").unwrap();

    let out = xencode(&dir, &["remote", "list"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("valid"), "valid profile was missing");
    assert!(
        stdout.contains("! damaged"),
        "unreadable profile was not reported"
    );
    assert!(
        stdout.contains("unreadable:"),
        "unreadable detail was missing"
    );

    let _ = std::fs::remove_dir_all(dir);
}
