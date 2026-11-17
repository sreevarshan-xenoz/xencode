//! Narrowing flags on `doctor` select different reports and must not be
//! silently combined: a request for two slices used to return only one.

use std::process::Command;

#[test]
fn doctor_rejects_combined_narrowing_flags() {
    let out = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(["doctor", "--env", "--selfcheck", "--format", "json"])
        .output()
        .expect("run xencode doctor");
    assert!(
        !out.status.success(),
        "combined scopes were accepted silently"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("use one at a time"),
        "the error did not explain the conflict: {stderr}"
    );
}
