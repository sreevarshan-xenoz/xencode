//! DK-2: `xencode badge` says plainly when the badge program is missing.
use std::process::Command;

#[test]
fn a_missing_badge_is_named_with_how_to_build_it() {
    let empty = tempfile::tempdir().unwrap();
    let config = tempfile::tempdir().unwrap();
    // No badge next to the test binary (it builds in rust/badge/target), and
    // PATH holds only an empty folder.
    let output = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .arg("badge")
        .env("PATH", empty.path())
        .env("XCODE_CONFIG_DIR", config.path())
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("xencode-badge was not found next to xencode or on PATH"),
        "{stderr}"
    );
    assert!(stderr.contains("rust/badge/Cargo.toml"), "{stderr}");
    // The manual quotes this sentence; it must read the same.
    assert!(
        stderr.contains("or on PATH. Build it with: cargo build --release"),
        "{stderr}"
    );
}

#[test]
fn a_badge_already_running_is_reported_instead_of_started_again() {
    let empty = tempfile::tempdir().unwrap();
    let config = tempfile::tempdir().unwrap();
    // A running badge holds this lock; the test holds it in its place.
    let _lock = xencode_live_rs::take_badge_lock(&config.path().join("live")).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .arg("badge")
        .env("PATH", empty.path())
        .env("XCODE_CONFIG_DIR", config.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("The badge is already running."), "{stdout}");
    assert!(!stdout.contains("Started"), "{stdout}");
}
