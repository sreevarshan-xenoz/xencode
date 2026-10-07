//! Integration tests for `xencode agents --health` (AR-8).

use std::process::Command;

#[test]
fn agents_health_text_runs_and_reports_real_workers() {
    let bin = env!("CARGO_BIN_EXE_xencode");
    let out = Command::new(bin)
        .args(["agents", "--health"])
        .output()
        .expect("must run xencode agents --health");

    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(stdout.contains("Agent Worker Health:"));
    assert!(stdout.contains("read-only: worker health checks never silently fix credentials"));
    assert!(stdout.contains("Installed:"));
    assert!(stdout.contains("Responsive:"));
}

#[test]
fn agents_health_json_format_produces_structured_entries() {
    let bin = env!("CARGO_BIN_EXE_xencode");
    let out = Command::new(bin)
        .args(["agents", "--health", "--format", "json"])
        .output()
        .expect("must run xencode agents --health --format json");

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("must parse JSON");

    let array = json.as_array().expect("must be an array of worker health");
    assert!(!array.is_empty());

    for item in array {
        assert!(item.get("name").is_some());
        assert!(item.get("installed").is_some());
        assert!(item.get("authenticated").is_some());
        assert!(item.get("responsive").is_some());
        assert!(item.get("rate_limited").is_some());
        assert!(item.get("login_command").is_some());
    }
}

#[test]
fn agents_health_filter_by_agent() {
    let bin = env!("CARGO_BIN_EXE_xencode");
    let out = Command::new(bin)
        .args([
            "agents", "--health", "--agent", "claude", "--format", "json",
        ])
        .output()
        .expect("must run xencode agents --health --agent claude");

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("must parse JSON");
    let array = json.as_array().expect("must be an array");
    assert_eq!(array.len(), 1);
    assert_eq!(array[0]["name"], "claude");
}

#[test]
fn agents_health_unknown_agent_refuses_actionably() {
    let bin = env!("CARGO_BIN_EXE_xencode");
    let out = Command::new(bin)
        .args(["agents", "--health", "--agent", "nonexistent-agent-xyz"])
        .output()
        .expect("must run command");

    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("no roster agent named 'nonexistent-agent-xyz'"));
}

#[test]
fn expired_auth_display_only_offers_human_terminal_command() {
    use std::path::PathBuf;
    use xencode_agents_rs::{format_worker_health_table, AuthStatus, Responsiveness, WorkerHealth};

    let worker = WorkerHealth {
        name: "test-worker".to_string(),
        installed: true,
        binary: Some(PathBuf::from("/usr/bin/test-worker")),
        version: Some("1.0.0".to_string()),
        authenticated: AuthStatus::Expired {
            message: "OAuth token expired at 2026-10-01".to_string(),
            login_command: "test-worker auth login".to_string(),
        },
        responsive: Responsiveness::Responsive { latency_ms: 12 },
        rate_limited: false,
        login_command: Some("test-worker auth login".to_string()),
    };

    let table = format_worker_health_table(&[worker]);
    assert!(table.contains("Auth:         EXPIRED (OAuth token expired at 2026-10-01)"));
    assert!(table.contains("Next step:    Run `test-worker auth login` in your terminal"));
    assert!(table.contains("read-only: worker health checks never silently fix credentials"));

    // Ensure no automatic fix is advertised or performed
    assert!(!table.to_lowercase().contains("auto-fix"));
    assert!(!table.to_lowercase().contains("automated"));
}
