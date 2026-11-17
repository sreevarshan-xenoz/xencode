//! The cache commands must not claim success when the cache cannot be opened.

use std::process::Command;

fn xencode() -> Command {
    Command::new(env!("CARGO_BIN_EXE_xencode"))
}

#[test]
fn cache_stats_fails_when_the_cache_cannot_be_read() {
    let out = xencode()
        .arg("cache")
        .arg("stats")
        .env("XDG_CACHE_HOME", "/dev/null/x")
        .output()
        .expect("run xencode cache stats");
    assert!(
        !out.status.success(),
        "an unreadable cache was reported as an empty one: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn cache_clear_fails_when_the_cache_cannot_be_opened() {
    let out = xencode()
        .arg("cache")
        .arg("clear")
        .env("XDG_CACHE_HOME", "/dev/null/x")
        .output()
        .expect("run xencode cache clear");
    assert!(
        !out.status.success(),
        "an unreadable cache was reported as cleared: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}
