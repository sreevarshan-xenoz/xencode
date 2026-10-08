//! Tests for computer backend registry CLI (`xencode computers`, AF-4).
//!
//! Verifies:
//! 1. `xencode computers` lists registered backends with what each is (colab / ssh / docker).
//! 2. `xencode computers --json` emits machine-readable backend array.
//! 3. `xencode computers show <name>` displays backend details.
//! 4. `xencode computers use <name>` binds the active backend in config.
//! 5. Probing an unreachable machine answers honestly that it cannot reach it.

use std::path::Path;
use std::process::Output;

fn xencode(config_dir: &Path, args: &[&str]) -> Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .env("XCODE_CONFIG_DIR", config_dir)
        .output()
        .expect("xencode binary should execute")
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-computers-{tag}-{}-{}",
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
fn computers_list_shows_all_three_arms_with_what_each_is() {
    let tmp = temp_dir("list");
    let out = xencode(&tmp, &["computers"]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(stdout.contains("Registered computer backends:"), "{stdout}");
    assert!(stdout.contains("colab"), "{stdout}");
    assert!(stdout.contains("(colab"), "{stdout}");
    assert!(stdout.contains("ssh"), "{stdout}");
    assert!(stdout.contains("(ssh"), "{stdout}");
    assert!(stdout.contains("docker"), "{stdout}");
    assert!(stdout.contains("(docker"), "{stdout}");
}

#[test]
fn computers_json_emits_valid_array_with_kinds() {
    let tmp = temp_dir("json");
    let out = xencode(&tmp, &["computers", "--json"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let items: serde_json::Value = serde_json::from_str(&stdout).expect("must be valid JSON");
    let array = items.as_array().expect("must be JSON array");

    let ids: Vec<&str> = array.iter().filter_map(|v| v["id"].as_str()).collect();
    assert!(ids.contains(&"colab"));
    assert!(ids.contains(&"ssh"));
    assert!(ids.contains(&"docker"));

    for item in array {
        assert!(item["kind"].is_string());
        assert!(item["description"].is_string());
        assert!(item["available"].is_boolean());
        assert!(item["active"].is_boolean());
    }
}

#[test]
fn computers_show_displays_backend_details() {
    let tmp = temp_dir("show");
    let out = xencode(&tmp, &["computers", "show", "docker"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Computer `docker`:"), "{stdout}");
    assert!(stdout.contains("kind:        docker"), "{stdout}");
}

#[test]
fn computers_use_updates_active_computer_backend() {
    let tmp = temp_dir("use");
    let use_out = xencode(&tmp, &["computers", "use", "ssh"]);
    assert!(use_out.status.success());
    let stdout = String::from_utf8_lossy(&use_out.stdout);
    assert!(
        stdout.contains("Active computer backend set to `ssh`."),
        "{stdout}"
    );

    // Verify it is active in list
    let list_out = xencode(&tmp, &["computers"]);
    assert!(list_out.status.success());
    let list_stdout = String::from_utf8_lossy(&list_out.stdout);
    assert!(list_stdout.contains("* ssh"), "{list_stdout}");
}

#[test]
fn computers_probe_answers_honestly_when_machine_unreachable() {
    // This is the unreachable case. Where a docker daemon answers — GitHub's
    // runners run one — the probe rightly succeeds, which is a different test.
    let reachable = std::process::Command::new("docker")
        .arg("info")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false);
    if reachable {
        eprintln!("skipping: a docker daemon answers here, so it is not unreachable");
        return;
    }
    let tmp = temp_dir("probe_docker");
    // The probe must answer honestly that it cannot reach the machine, not succeed
    let out = xencode(&tmp, &["computers", "probe", "docker"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("cannot be reached"), "{stderr}");
}
