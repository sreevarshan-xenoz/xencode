//! `xencode config set bitnet_url` driven through the real binary with an
//! isolated configuration directory. The CLI is the surface a person uses to
//! point the `bitnet:` route at a local bitnet.cpp server, so the key has to be
//! written, read back, validated, clearable, and still reject unknown keys.

use std::path::{Path, PathBuf};
use std::process::Output;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-bitnet-config-{tag}-{}-{}",
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
fn bitnet_url_is_written_shown_validated_and_cleared() {
    let dir = temp_dir("roundtrip");

    let out = xencode(
        &dir,
        &["config", "set", "bitnet_url", "http://127.0.0.1:8080/v1"],
    );
    assert!(
        out.status.success(),
        "set failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(saved(&dir)["bitnet_url"], "http://127.0.0.1:8080/v1");

    let shown = xencode(&dir, &["config", "show"]);
    assert!(shown.status.success());
    let text = String::from_utf8_lossy(&shown.stdout);
    assert!(
        text.contains("\"bitnet_url\": \"http://127.0.0.1:8080/v1\""),
        "config show did not print the new key: {text}"
    );

    let bad = xencode(&dir, &["config", "set", "bitnet_url", "ftp://127.0.0.1/v1"]);
    assert!(
        !bad.status.success(),
        "a non-http URL must be refused: {}",
        String::from_utf8_lossy(&bad.stdout)
    );
    let error = format!(
        "{}{}",
        String::from_utf8_lossy(&bad.stdout),
        String::from_utf8_lossy(&bad.stderr)
    );
    assert!(
        error.contains("bitnet_url") && error.contains("http://"),
        "the refusal should name the key and the allowed scheme: {error}"
    );
    assert_eq!(
        saved(&dir)["bitnet_url"],
        "http://127.0.0.1:8080/v1",
        "a refused value must not be written"
    );

    let cleared = xencode(&dir, &["config", "set", "bitnet_url", ""]);
    assert!(
        cleared.status.success(),
        "empty clears the key: {}",
        String::from_utf8_lossy(&cleared.stderr)
    );
    assert_eq!(saved(&dir)["bitnet_url"], "");

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_unknown_config_key_is_still_rejected() {
    let dir = temp_dir("unknown");
    let out = xencode(
        &dir,
        &["config", "set", "bitnet_urll", "http://127.0.0.1/v1"],
    );
    assert!(!out.status.success(), "a typo must not be accepted");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("unknown config key: bitnet_urll"),
        "unhelpful refusal: {text}"
    );
    assert!(
        !dir.join("config.json").exists(),
        "nothing should be written for an unknown key"
    );
    let _ = std::fs::remove_dir_all(dir);
}
