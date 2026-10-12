//! `xencode config set` for the conversation screen's settings (TD-1, TD-2):
//! `tool_output`, `side_panel` and `status_line`, driven through the real
//! binary against a config directory of its own.

use std::path::{Path, PathBuf};
use std::process::Output;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-screen-settings-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn xencode(config_dir: &Path, args: &[&str]) -> Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .env("XCODE_CONFIG_DIR", config_dir)
        .output()
        .expect("xencode is built by the time these tests run")
}

fn saved(config_dir: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(config_dir.join("config.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn the_screen_settings_are_written_when_they_are_words_it_knows() {
    let dir = temp_dir("good");
    for (key, value) in [
        ("tool_output", "expand"),
        ("side_panel", "off"),
        ("status_line", "model, cost,branch"),
    ] {
        let out = xencode(&dir, &["config", "set", key, value]);
        assert!(
            out.status.success(),
            "{key}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let config = saved(&dir);
    assert_eq!(config["tool_output"], "expand");
    assert_eq!(config["side_panel"], "off");
    assert_eq!(
        config["status_line"],
        serde_json::json!(["model", "cost", "branch"])
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_word_it_does_not_know_is_refused_and_nothing_is_written() {
    let dir = temp_dir("bad");
    for (key, value, named) in [
        ("tool_output", "collapse", "fold"),
        ("side_panel", "maybe", "auto"),
        ("status_line", "model,weather", "weather"),
    ] {
        let out = xencode(&dir, &["config", "set", key, value]);
        assert!(!out.status.success(), "{key}={value} was accepted");
        let said = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(said.contains(named), "{key}: {said}");
    }
    if dir.join("config.json").exists() {
        let config = saved(&dir);
        assert_ne!(config["tool_output"], "collapse");
        assert_ne!(config["side_panel"], "maybe");
    }
    std::fs::remove_dir_all(&dir).ok();
}
