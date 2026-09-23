//! The `xencode colab preflight` gate: verify the google-colab-cli bridge is
//! actually usable before anything tries to bring a VM up.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use xencode_config_rs::XencodeConfig;

/// Minimum `google-colab-cli` that ships the `ssh` subcommand. The 0.6.0
/// release shipped without it (upstream issue #102); it landed in 0.7.0.
pub const MIN_COLAB_VERSION: &str = "0.7.0";

/// Filename of the ed25519 private key inside the xencode config dir.
pub const KEY_FILENAME: &str = "colab_ed25519";

/// Subprocesses that touch the Colab backend may be slow (token fetch +
/// network); everything else returns in milliseconds. One ceiling for all.
const CMD_TIMEOUT: Duration = Duration::from_secs(20);

/// One preflight check result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    /// What the check gates ("colab CLI", "Colab API sign-in", …).
    pub name: &'static str,
    pub ok: bool,
    /// What was found, or why the check failed.
    pub detail: String,
    /// One-line remediation when `ok` is false.
    pub fix: Option<String>,
}

/// The full set of preflight checks, in run order.
#[derive(Debug, Default)]
pub struct PreflightReport {
    pub checks: Vec<Check>,
}

impl PreflightReport {
    /// Every check passing means `xencode colab up` can proceed.
    pub fn ready(&self) -> bool {
        self.checks.iter().all(|c| c.ok)
    }

    /// Number of failing checks.
    pub fn failed(&self) -> usize {
        self.checks.iter().filter(|c| !c.ok).count()
    }
}

/// First executable found on `PATH`, if any.
pub fn which(bin: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Paths of the colab ed25519 key pair inside the xencode config dir.
pub fn key_paths() -> Result<(PathBuf, PathBuf), String> {
    let dir = XencodeConfig::config_dir().map_err(|e| e.to_string())?;
    Ok((
        dir.join(KEY_FILENAME),
        dir.join(format!("{KEY_FILENAME}.pub")),
    ))
}

/// Run `exe args` (no shell), capture stdout/stderr, with [`CMD_TIMEOUT`].
async fn run(exe: &Path, args: &[&str]) -> std::io::Result<(bool, Vec<u8>, Vec<u8>)> {
    let output = tokio::time::timeout(
        CMD_TIMEOUT,
        tokio::process::Command::new(exe).args(args).output(),
    )
    .await
    .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "command timed out"))??;
    Ok((output.status.success(), output.stdout, output.stderr))
}

/// Run every preflight check. `generate_key` additionally creates the
/// ed25519 key pair with `ssh-keygen` when it is missing.
pub async fn preflight(generate_key: bool) -> Result<PreflightReport, String> {
    let mut checks = Vec::new();

    let colab = which("colab");
    match &colab {
        Some(path) => checks.push(Check {
            name: "colab CLI",
            ok: true,
            detail: path.display().to_string(),
            fix: None,
        }),
        None => checks.push(Check {
            name: "colab CLI",
            ok: false,
            detail: "google-colab-cli not on PATH".to_string(),
            fix: Some(
                "uv tool install google-colab-cli   (or: pip install google-colab-cli)".to_string(),
            ),
        }),
    }

    if let Some(colab_path) = colab.as_deref() {
        // Version gate: 0.6.0 exists but lacks the `ssh` bridge (issue #102).
        match run(colab_path, &["version"]).await {
            Ok((_, stdout, _)) => {
                let shown = first_version(&String::from_utf8_lossy(&stdout));
                match shown {
                    Some(version) if version_at_least(&version, MIN_COLAB_VERSION) => {
                        checks.push(Check {
                            name: "colab version",
                            ok: true,
                            detail: format!("{version} >= {MIN_COLAB_VERSION}"),
                            fix: None,
                        })
                    }
                    Some(version) => checks.push(Check {
                        name: "colab version",
                        ok: false,
                        detail: format!(
                            "{version} — needs >= {MIN_COLAB_VERSION} (0.6.0 has no `ssh` bridge)"
                        ),
                        fix: Some(
                            "colab update   (or: uv tool upgrade google-colab-cli / pip install -U google-colab-cli)".to_string(),
                        ),
                    }),
                    None => checks.push(Check {
                        name: "colab version",
                        ok: false,
                        detail: "could not parse `colab version` output".to_string(),
                        fix: Some(
                            "colab update   (or: uv tool upgrade google-colab-cli / pip install -U google-colab-cli)".to_string(),
                        ),
                    }),
                }
            }
            Err(err) => checks.push(Check {
                name: "colab version",
                ok: false,
                detail: format!("`colab version` failed: {err}"),
                fix: Some(
                    "colab update   (or: uv tool upgrade google-colab-cli / pip install -U google-colab-cli)".to_string(),
                ),
            }),
        }

        // Functional guard against a build whose `ssh` subcommand is missing
        // (the 0.6.0 trap), independent of the version string.
        match run(colab_path, &["ssh", "--help"]).await {
            Ok((ok, _, _)) if ok => checks.push(Check {
                name: "colab ssh bridge",
                ok: true,
                detail: "`colab ssh` accepted (proxy-mode bridge available)".to_string(),
                fix: None,
            }),
            _ => checks.push(Check {
                name: "colab ssh bridge",
                ok: false,
                detail: "`colab ssh` rejected by this build".to_string(),
                fix: Some("colab update — the ssh bridge landed in 0.7.0".to_string()),
            }),
        }

        // The sign-in probe: `colab sessions` hits the backend with the
        // current Application Default Credentials (or `--auth oauth2` flow).
        match run(colab_path, &["sessions"]).await {
            Ok((ok, _, _)) if ok => checks.push(Check {
                name: "Colab API",
                ok: true,
                detail: "`colab sessions` succeeded (backend reachable)".to_string(),
                fix: None,
            }),
            Ok((_, _, stderr)) => checks.push(Check {
                name: "Colab API",
                ok: false,
                detail: format!(
                    "`colab sessions` failed{}",
                    first_line(&String::from_utf8_lossy(&stderr))
                        .map(|l| format!(" — {l}"))
                        .unwrap_or_default()
                ),
                fix: Some(
                    "make Google Application Default Credentials available: gcloud auth application-default login   (or run colab with --auth oauth2 once)".to_string(),
                ),
            }),
            Err(err) => checks.push(Check {
                name: "Colab API",
                ok: false,
                detail: format!("`colab sessions` failed: {err}"),
                fix: Some(
                    "make Google Application Default Credentials available: gcloud auth application-default login   (or run colab with --auth oauth2 once)".to_string(),
                ),
            }),
        }
    }

    // OpenSSH is the bridge's other half: local `ssh -L` forwarding needs the
    // ssh client, and key generation needs ssh-keygen.
    for bin in ["ssh", "ssh-keygen"] {
        match which(bin) {
            Some(path) => checks.push(Check {
                name: "OpenSSH",
                ok: true,
                detail: format!("{bin} at {}", path.display()),
                fix: None,
            }),
            None => checks.push(Check {
                name: "OpenSSH",
                ok: false,
                detail: format!("`{bin}` not on PATH"),
                fix: Some(
                    "install an OpenSSH client (apt install openssh-client / brew install openssh)"
                        .to_string(),
                ),
            }),
        }
    }

    // The SSH key pair the bridge authenticates with (`colab ssh -i`).
    let (key_path, pub_path) = key_paths()?;
    if key_path.exists() && pub_path.exists() {
        checks.push(Check {
            name: "SSH ed25519 key",
            ok: true,
            detail: format!("{} (+ .pub)", key_path.display()),
            fix: None,
        });
    } else if generate_key {
        match generate_keypair(&key_path).await {
            Ok(()) => checks.push(Check {
                name: "SSH ed25519 key",
                ok: true,
                detail: format!("generated {}", key_path.display()),
                fix: None,
            }),
            Err(err) => checks.push(Check {
                name: "SSH ed25519 key",
                ok: false,
                detail: format!("generation failed: {err}"),
                fix: Some(
                    "install ssh-keygen (OpenSSH client), then `xencode colab preflight --generate-key` again".to_string(),
                ),
            }),
        }
    } else {
        checks.push(Check {
            name: "SSH ed25519 key",
            ok: false,
            detail: format!("{} missing", key_path.display()),
            fix: Some("xencode colab preflight --generate-key".to_string()),
        });
    }

    Ok(PreflightReport { checks })
}

/// Generate an ed25519 key pair at `key_path` (and `key_path.pub`) with
/// `ssh-keygen`, no passphrase.
async fn generate_keypair(key_path: &Path) -> std::io::Result<()> {
    if let Some(parent) = key_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let ssh_keygen = which("ssh-keygen").ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "ssh-keygen not on PATH")
    })?;
    let status = tokio::process::Command::new(ssh_keygen)
        .args(["-t", "ed25519", "-f"])
        .arg(key_path)
        .args(["-N", "", "-C", "xencode-colab@localhost"])
        .status()
        .await?;
    if !status.success() {
        return Err(std::io::Error::other("ssh-keygen exited non-zero"));
    }
    Ok(())
}

/// First `x.y[.z...]` version-looking token in `output`.
fn first_version(output: &str) -> Option<String> {
    let bytes = output.as_bytes();
    let mut start: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        if b.is_ascii_digit() {
            start.get_or_insert(i);
            continue;
        }
        if b == b'.' && start.is_some() {
            continue; // mid-version dot: keep accumulating
        }
        if let Some(s) = start.take() {
            let token = &output[s..i];
            if token.contains('.') {
                return Some(token.to_string());
            }
        }
    }
    start.and_then(|s| {
        let token = &output[s..];
        token.contains('.').then(|| token.to_string())
    })
}

/// Parse `s` into leading-integer version segments.
fn parse_version(s: &str) -> Vec<u64> {
    s.split('.')
        .filter_map(|part| {
            let digits: String = part.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse().ok()
        })
        .collect()
}

/// True when `actual` (e.g. "0.7.2") is >= `minimum` (e.g. "0.7.0").
pub fn version_at_least(actual: &str, minimum: &str) -> bool {
    let actual = parse_version(actual);
    let minimum = parse_version(minimum);
    let len = actual.len().max(minimum.len());
    for i in 0..len {
        let a = actual.get(i).copied().unwrap_or(0);
        let b = minimum.get(i).copied().unwrap_or(0);
        if a != b {
            return a > b;
        }
    }
    true
}

/// First non-blank line of `s`, trimmed and capped — for error tails.
fn first_line(s: &str) -> Option<String> {
    let line = s.lines().find(|l| !l.trim().is_empty())?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.chars().take(160).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{temp_dir, with_env, write_script};
    use std::path::Path;

    /// Fake google-colab-cli: a modern 0.7.2 that accepts `ssh`.
    fn fake_colab_modern(dir: &Path) {
        write_script(
            dir,
            "colab",
            r#"#!/bin/sh
case "$1" in
  version) echo "colab 0.7.2" ;;
  ssh) echo "Usage: colab ssh [OPTIONS] -s session" >&2; exit 0 ;;
  sessions) echo "no active sessions" ;;
  *) exit 1 ;;
esac
"#,
        );
    }

    /// Fake google-colab-cli 0.6.0: version parses, but `ssh` is unknown.
    fn fake_colab_060(dir: &Path) {
        write_script(
            dir,
            "colab",
            r#"#!/bin/sh
case "$1" in
  version) echo "colab 0.6.0" ;;
  ssh) echo "Error: No such command 'ssh'" >&2; exit 2 ;;
  sessions) echo "no active sessions" ;;
  *) exit 1 ;;
esac
"#,
        );
    }

    /// Fake ssh-keygen that creates `<key>` and `<key>.pub` on disk, using
    /// only shell builtins (the hermetic test PATH hides mkdir/dirname).
    fn fake_ssh_keygen(dir: &Path) {
        write_script(
            dir,
            "ssh-keygen",
            r#"#!/bin/sh
out=""
prev=""
for a in "$@"; do
  [ "$prev" = "-f" ] && out="$a"
  prev="$a"
done
[ -n "$out" ] || exit 1
: > "$out"
: > "${out}.pub"
exit 0
"#,
        );
    }

    fn fake_ssh(dir: &Path) {
        write_script(dir, "ssh", "#!/bin/sh\nexit 0\n");
    }

    #[test]
    fn version_gate_compares_numerically() {
        assert!(version_at_least("0.7.2", "0.7.0"));
        assert!(version_at_least("0.7.0", "0.7.0"));
        assert!(version_at_least("0.8.0", "0.7.2"));
        assert!(version_at_least("0.7.2", "0.6"));
        assert!(!version_at_least("0.6.0", "0.7.0"));
        assert!(!version_at_least("0.7.1", "0.7.2"));
        // Suffixed or partial strings still compare by leading numbers.
        assert!(version_at_least("colab 0.7.2", "0.7.0"));
    }

    #[test]
    fn first_version_extracts_x_y_from_mixed_text() {
        assert_eq!(first_version("colab 0.7.2\n"), Some("0.7.2".to_string()));
        assert_eq!(
            first_version("version: 0.7 (colab)\n"),
            Some("0.7".to_string())
        );
        assert_eq!(first_version("0\nno version here"), None);
        assert_eq!(first_version(""), None);
    }

    #[test]
    fn which_finds_a_binary_on_path() {
        let dir = temp_dir("which");
        write_script(&dir, "colab", "#!/bin/sh\nexit 0\n");
        let _lock = with_env(&dir, &dir);
        assert_eq!(
            which("colab"),
            Some(dir.join("colab")),
            "fake colab must be found on PATH"
        );
        assert_eq!(which("definitely-not-a-real-bin-xyz"), None);
    }

    #[tokio::test]
    async fn modern_cli_with_generated_key_passes_everything() {
        let bin_dir = temp_dir("modern");
        fake_colab_modern(&bin_dir);
        fake_ssh(&bin_dir);
        fake_ssh_keygen(&bin_dir);
        let xcode_dir = temp_dir("modern-cfg");
        let _guard = with_env(&bin_dir, &xcode_dir);

        let report = preflight(true).await.expect("preflight runs");

        assert!(report.ready(), "all checks green: {report:?}");
        let key = xcode_dir.join(KEY_FILENAME);
        assert!(key.exists(), "private key generated");
        assert!(key.with_extension("pub").exists(), "public key generated");
    }

    #[tokio::test]
    async fn missing_key_asks_for_generate_flag() {
        let bin_dir = temp_dir("nokey");
        fake_colab_modern(&bin_dir);
        fake_ssh(&bin_dir);
        fake_ssh_keygen(&bin_dir);
        let xcode_dir = temp_dir("nokey-cfg");
        let _guard = with_env(&bin_dir, &xcode_dir);

        let report = preflight(false).await.expect("preflight runs");

        assert!(!report.ready());
        let key_check = report
            .checks
            .iter()
            .find(|c| c.name == "SSH ed25519 key")
            .expect("key check present");
        assert!(!key_check.ok);
        assert_eq!(
            key_check.fix.as_deref(),
            Some("xencode colab preflight --generate-key")
        );
        assert!(!xcode_dir.join(KEY_FILENAME).exists());
    }

    #[tokio::test]
    async fn the_060_build_is_rejected_on_version_and_ssh() {
        let bin_dir = temp_dir("old");
        fake_colab_060(&bin_dir);
        fake_ssh(&bin_dir);
        fake_ssh_keygen(&bin_dir);
        let xcode_dir = temp_dir("old-cfg");
        let _guard = with_env(&bin_dir, &xcode_dir);

        let report = preflight(true).await.expect("preflight runs");
        assert!(!report.ready());

        let version_check = report
            .checks
            .iter()
            .find(|c| c.name == "colab version")
            .expect("version check present");
        assert!(!version_check.ok);
        assert!(version_check.detail.contains("0.6.0"));

        let ssh_check = report
            .checks
            .iter()
            .find(|c| c.name == "colab ssh bridge")
            .expect("ssh bridge check present");
        assert!(!ssh_check.ok);
    }

    #[tokio::test]
    async fn missing_cli_reports_unpowered_and_skips_backend_probes() {
        let bin_dir = temp_dir("no-cli");
        fake_ssh(&bin_dir);
        let xcode_dir = temp_dir("no-cli-cfg");
        let _guard = with_env(&bin_dir, &xcode_dir);

        let report = preflight(false).await.expect("preflight runs");
        assert!(!report.ready());

        let cli_check = report
            .checks
            .iter()
            .find(|c| c.name == "colab CLI")
            .expect("cli check present");
        assert!(!cli_check.ok);
        assert!(cli_check.detail.contains("not on PATH"));

        // No colab binary -> no version / ssh / sessions probes run.
        let ran_probes = ["colab version", "colab ssh bridge", "Colab API"]
            .iter()
            .any(|name| report.checks.iter().any(|c| c.name == *name));
        assert!(!ran_probes, "backend probes skipped when CLI missing");

        // OpenSSH + key checks still run (they do not need the CLI).
        assert_eq!(
            report.checks.iter().filter(|c| c.name == "OpenSSH").count(),
            2,
            "both ssh and ssh-keygen are reported"
        );
    }

    #[tokio::test]
    async fn generation_without_ssh_keygen_is_reported_not_panicked() {
        let bin_dir = temp_dir("no-kg");
        fake_colab_modern(&bin_dir);
        fake_ssh(&bin_dir);
        let xcode_dir = temp_dir("no-kg-cfg");
        let _guard = with_env(&bin_dir, &xcode_dir);

        let report = preflight(true).await.expect("preflight runs");
        assert!(!report.ready());

        let key_check = report
            .checks
            .iter()
            .find(|c| c.name == "SSH ed25519 key")
            .expect("key check present");
        assert!(!key_check.ok);
        assert!(key_check.detail.contains("generation failed"));
    }
}
