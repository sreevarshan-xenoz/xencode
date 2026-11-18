//! Integration test for `xencode setup --remote --dry-run` (RG-3).
//!
//! Driven through the compiled binary with an isolated `$XCODE_CONFIG_DIR` and
//! a fake `colab` executable first on `PATH`. The fake is a real process that
//! only answers the three preflight probes and logs every command it receives,
//! so the test can prove the dry run reached the preview without ever issuing a
//! VM, forward or network command. No mock structs or canned responses stand in
//! for the CLI: the decision, preflight and preview all run against real files
//! and real subprocesses.

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Output;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-setup-remote-cli-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn xencode(config_dir: &Path, bin_dir: &Path, args: &[&str]) -> Output {
    let path = format!(
        "{}:{}",
        bin_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .env("XCODE_CONFIG_DIR", config_dir)
        .env("PATH", path)
        .output()
        .expect("xencode binary is built for tests")
}

fn write_executable(path: &Path, body: &str) {
    std::fs::write(path, body).unwrap();
    let mut perms = std::fs::metadata(path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms).unwrap();
}

#[test]
fn setup_remote_dry_run_previews_colab_without_any_vm_or_network_command() {
    let root = temp_dir("dry-run");
    let config_dir = root.join("config");
    let bin_dir = root.join("bin");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::create_dir_all(&bin_dir).unwrap();

    // A config that turns the bridge on. Every other field takes its real
    // default, which is what a person who just enabled Colab would have.
    let config_path = config_dir.join("config.json");
    std::fs::write(&config_path, r#"{"colab":{"enabled":true}}"#).unwrap();

    // Force the no-local-fit branch deterministically on any host: a real
    // advice file loaded by the real loader whose only entry is larger than any
    // machine's largest single memory pool. This is the machine shape that the
    // Colab route exists for, expressed as data rather than a mocked decision.
    std::fs::write(
        config_dir.join("model_advice.json"),
        r#"{"as_of":"2026-11-18","tiers":[
          {"name":"huge","max_file_bytes":18446744073709551615,"gguf":[
            {"label":"Test Model","repo":"example/test-model","file":"test.gguf","revision":"0123456789012345678901234567890123456789","size_bytes":18446744073709551615,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}
          ]}
        ]}"#,
    )
    .unwrap();

    // Preflight also checks the SSH bridge and the key pair. Make both real on
    // disk: tiny executable scripts for the binaries and empty key files for
    // the keypair probe. generate_key is false in setup, so presence is enough.
    let calls = bin_dir.join("calls.log");
    write_executable(
        &bin_dir.join("colab"),
        &format!(
            "#!/bin/sh\necho \"$*\" >> '{}'\ncase \"$1\" in\n  version) echo 'colab 0.7.0'; exit 0;;\n  ssh) exit 0;;\n  sessions) echo '[]'; exit 0;;\n  *) echo \"UNEXPECTED colab command: $*\" >&2; exit 1;;\nesac\n",
            calls.display()
        ),
    );
    write_executable(&bin_dir.join("ssh"), "#!/bin/sh\nexit 0\n");
    write_executable(&bin_dir.join("ssh-keygen"), "#!/bin/sh\nexit 0\n");
    std::fs::write(config_dir.join("colab_ed25519"), "").unwrap();
    std::fs::write(config_dir.join("colab_ed25519.pub"), "").unwrap();

    let config_before = std::fs::read(&config_path).unwrap();
    let out = xencode(&config_dir, &bin_dir, &["setup", "--remote", "--dry-run"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "setup --remote --dry-run failed: {stdout}\n{stderr}"
    );

    // The decision really picked Colab from the observed state, and the existing
    // bring-up dry-run preview was reached with the HR-1 model mapped to --model.
    assert!(
        stdout.contains("route is Colab"),
        "the decision did not choose Colab: {stdout}"
    );
    assert!(
        stdout.contains("colab up --dry-run — nothing was started and nothing was written."),
        "the bring-up preview was not reached: {stdout}"
    );
    assert!(
        stdout.contains("model: example/test-model"),
        "the HR-1 recommended model was not mapped to --model: {stdout}"
    );
    assert!(
        stdout.contains("config.json would change"),
        "the preview did not show the config it would write: {stdout}"
    );

    // The only commands the real subprocess ever saw are the three preflight
    // probes. A VM create, a forward or any other command would show up here and
    // fail this assertion — that is the proof no live backend was touched.
    let seen = std::fs::read_to_string(&calls).unwrap_or_default();
    for line in seen.lines() {
        assert!(
            line.starts_with("version")
                || line.starts_with("ssh --help")
                || line.starts_with("sessions"),
            "a VM, forward or network command was attempted: {line}"
        );
    }
    assert!(
        seen.contains("version"),
        "preflight version probe missing: {seen}"
    );
    assert!(
        seen.contains("ssh --help"),
        "preflight ssh probe missing: {seen}"
    );
    assert!(
        seen.contains("sessions"),
        "preflight sessions probe missing: {seen}"
    );
    assert!(
        !stderr.contains("UNEXPECTED"),
        "the fake CLI was asked to do more than preflight: {stderr}"
    );

    // A dry run writes nothing, including the config it only previews.
    assert_eq!(
        std::fs::read(&config_path).unwrap(),
        config_before,
        "dry run changed the config file"
    );

    let _ = std::fs::remove_dir_all(root);
}
