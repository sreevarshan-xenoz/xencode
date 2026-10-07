//! Tests for engine composition discipline and configuration dump (AF-3).
//!
//! Verifies:
//! - `xencode --dump-config` prints valid composition summary JSON
//! - `xencode config dump` behaves identically
//! - `xencode config show --composition` outputs composition summary
//! - `composition_profile`, `computer_backend`, `worker_adapter` can be inspected and configured
//! - Invalid profile selection is rejected

use std::path::Path;
use std::process::Output;

fn xencode(config_dir: &Path, args: &[&str]) -> Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .env("XCODE_CONFIG_DIR", config_dir)
        .output()
        .expect("xencode is built by the time these tests run")
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-composition-{tag}-{}-{}",
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
fn dump_config_flag_emits_valid_composition_json() {
    let tmp = temp_dir("flag");
    let out = xencode(&tmp, &["--dump-config"]);
    assert!(
        out.status.success(),
        "stdout: {}, stderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let val: serde_json::Value = serde_json::from_str(&stdout).expect("must be valid JSON");

    assert_eq!(val["profile"], "coding");
    assert_eq!(val["computer_backend"], "colab");
    assert_eq!(val["worker_adapter"], "mcp");
    assert!(val["capabilities"]["filesystem.read"] == "allow");
    assert!(val["capabilities"]["filesystem.write"] == "allow");
    assert!(val["available_computer_backends"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!("colab")));
    assert!(val["available_worker_adapters"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!("mcp")));
    assert!(val["known_plugin_permissions"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!("prompt")));
    assert!(val["known_plugin_permissions"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!("hooks")));
}

#[test]
fn config_dump_subcommand_matches_dump_flag() {
    let tmp = temp_dir("subcmd");
    let out = xencode(&tmp, &["config", "dump"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let val: serde_json::Value = serde_json::from_str(&stdout).expect("must be valid JSON");

    assert_eq!(val["profile"], "coding");
    assert_eq!(val["computer_backend"], "colab");
    assert_eq!(val["worker_adapter"], "mcp");
}

#[test]
fn config_show_composition_flag_emits_composition_json() {
    let tmp = temp_dir("show_comp");
    let out = xencode(&tmp, &["config", "show", "--composition"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let val: serde_json::Value = serde_json::from_str(&stdout).expect("must be valid JSON");

    assert_eq!(val["profile"], "coding");
}

#[test]
fn setting_composition_profile_updates_dump() {
    let tmp = temp_dir("set_profile");
    // Initial profile is coding
    let set_out = xencode(&tmp, &["config", "set", "composition_profile", "minimal"]);
    assert!(
        set_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&set_out.stderr)
    );

    let dump_out = xencode(&tmp, &["--dump-config"]);
    assert!(dump_out.status.success());
    let val: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&dump_out.stdout)).unwrap();
    assert_eq!(val["profile"], "minimal");
    assert_eq!(val["capabilities"]["filesystem.write"], "deny");
    assert_eq!(val["capabilities"]["shell.execute"], "deny");
}

#[test]
fn setting_invalid_profile_is_refused() {
    let tmp = temp_dir("invalid_profile");
    let set_out = xencode(
        &tmp,
        &["config", "set", "composition_profile", "super-unrestricted"],
    );
    assert!(!set_out.status.success());
    let stderr = String::from_utf8_lossy(&set_out.stderr);
    assert!(
        stderr.contains("composition_profile must be one of"),
        "{stderr}"
    );
}
