//! Bridge orchestration core: builds and spawns the local `ssh -L` forward
//! that carries the Colab VM's OpenAI-compatible endpoint to the laptop.
//!
//! The forward is one OpenSSH process with the colab bridge as its
//! `ProxyCommand`:
//!
//! ```text
//! ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 -i <key> \
//!     -o ProxyCommand="colab ssh --proxy-mode -s <session> -i <key>" \
//!     root@colab
//! ```
//!
//! OpenSSH runs `ProxyCommand` through the user's shell, so every value we
//! interpolate into it ([`shell_quote`]) is single-quoted — a session name is
//! user-typed and must never be interpreted as shell syntax.

use std::path::{Path, PathBuf};

use crate::preflight::which;

/// Paths to the optional tools the bridge needs, resolved from `PATH`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binaries {
    /// The `google-colab-cli` executable (`colab`).
    pub colab: PathBuf,
    /// The OpenSSH client (`ssh`).
    pub ssh: PathBuf,
}

/// Resolve `colab` and `ssh` from `PATH`. Error strings are end-user fixes —
/// the same "report itself unpowered" pattern as the preflight checks.
pub fn resolve_binaries() -> Result<Binaries, String> {
    let colab = which("colab").ok_or_else(|| {
        "google-colab-cli not on PATH — install it (uv tool install google-colab-cli, or pip install google-colab-cli)".to_string()
    })?;
    let ssh = which("ssh").ok_or_else(|| {
        "an OpenSSH client was not found on PATH — the forward needs `ssh`".to_string()
    })?;
    Ok(Binaries { colab, ssh })
}

/// Single-quote `s` for interpolation into a shell string: wrap in `'` and
/// escape any inner `'` as `'\''`. Safe for arbitrary user-typed values.
pub fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

/// The argv of the `colab ssh --proxy-mode -s <session> -i <key>` bridge that
/// OpenSSH runs as a `ProxyCommand`. Passed as a vector, never through a
/// shell, so no quoting or injection applies on the colab side.
pub fn proxy_argv(colab: &Path, session: &str, key: &Path) -> Vec<String> {
    vec![
        colab.display().to_string(),
        "ssh".to_string(),
        "--proxy-mode".to_string(),
        "-s".to_string(),
        session.to_string(),
        "-i".to_string(),
        key.display().to_string(),
    ]
}

/// The `-o ProxyCommand=...` value for the forward: the colab argv joined,
/// shell-quoted so the session name cannot escape into OpenSSH's `sh -c`.
pub fn proxy_command_opt(colab: &Path, session: &str, key: &Path) -> String {
    let argv = proxy_argv(colab, session, key);
    let joined = argv
        .iter()
        .map(|a| shell_quote(a))
        .collect::<Vec<_>>()
        .join(" ");
    format!("ProxyCommand={joined}")
}

/// The argv of the local forward: `ssh -N -L 127.0.0.1:local:127.0.0.1:remote`,
/// with the colab bridge as its `ProxyCommand`, the shared key, and sane
/// no-prompt/keepalive options. Runs `-N` (no remote command) so it just
/// forwards until killed.
pub fn forward_argv(
    bins: &Binaries,
    session: &str,
    key: &Path,
    local_port: u16,
    remote_port: u16,
) -> Vec<String> {
    vec![
        bins.ssh.display().to_string(),
        "-N".to_string(),
        "-L".to_string(),
        format!("127.0.0.1:{local_port}:127.0.0.1:{remote_port}"),
        "-i".to_string(),
        key.display().to_string(),
        "-o".to_string(),
        proxy_command_opt(&bins.colab, session, key),
        // Never prompt interactively: the bridge does auth, the forward is a
        // background process with no terminal.
        "-o".to_string(),
        "BatchMode=yes".to_string(),
        "-o".to_string(),
        "StrictHostKeyChecking=no".to_string(),
        "-o".to_string(),
        "UserKnownHostsFile=/dev/null".to_string(),
        // Keep the tunnel from silently going stale.
        "-o".to_string(),
        "ServerAliveInterval=15".to_string(),
        "-o".to_string(),
        "ServerAliveCountMax=2".to_string(),
        "-o".to_string(),
        "ExitOnForwardFailure=yes".to_string(),
        // The ProxyCommand handles the real host; OpenSSH just needs a name.
        "colab-vm".to_string(),
    ]
}

/// The URL the forward exposes — what `xencode config set remote_url` points
/// at. `/v1` is the OpenAI-compatible API root served on the VM.
pub fn forward_url(local_port: u16) -> String {
    format!("http://127.0.0.1:{local_port}/v1")
}

/// Spawn the forward as a background process. Returns the URL it exposes and
/// the child handle. `ssh` inherits stderr (bridge/connect errors surface)
/// with stdin closed and stdout discarded.
pub async fn spawn_forward(
    bins: &Binaries,
    session: &str,
    key: &Path,
    local_port: u16,
    remote_port: u16,
) -> Result<(String, tokio::process::Child), String> {
    let argv = forward_argv(bins, session, key, local_port, remote_port);
    let child = tokio::process::Command::new(&bins.ssh)
        .args(&argv[1..])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("could not spawn ssh forward: {e}"))?;
    Ok((forward_url(local_port), child))
}

/// True when the process with `pid` exists (a `kill(pid, 0)` probe). `0` is
/// never a real pid we track.
pub fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    // Let the kernel decide; the probe sends no signal.
    // SAFETY: kill(2) is async-signal-safe and raw; a bogus pid only errno.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{temp_dir, with_env, write_script};

    #[test]
    fn shell_quote_wraps_and_escapes_inner_quotes() {
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("a'b"), "'a'\\''b'");
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("$(rm -rf /)"), "'$(rm -rf /)'");
    }

    #[test]
    fn proxy_argv_orders_colab_flags() {
        let argv = proxy_argv(
            Path::new("/bin/colab"),
            "xencode-t4",
            Path::new("/k/colab_ed25519"),
        );
        assert_eq!(
            argv,
            vec![
                "/bin/colab",
                "ssh",
                "--proxy-mode",
                "-s",
                "xencode-t4",
                "-i",
                "/k/colab_ed25519"
            ]
        );
    }

    #[test]
    fn proxy_command_opt_quotes_so_session_names_cannot_escape() {
        let opt = proxy_command_opt(Path::new("/bin/colab"), "ev; touch /pwn", Path::new("/k/k"));
        assert_eq!(
            opt,
            "ProxyCommand='/bin/colab' 'ssh' '--proxy-mode' '-s' 'ev; touch /pwn' '-i' '/k/k'"
        );
        assert!(!opt.contains('\n'));
    }

    #[test]
    fn forward_argv_is_a_background_non_interactive_tunnel() {
        let bins = Binaries {
            colab: PathBuf::from("/bin/colab"),
            ssh: PathBuf::from("/usr/bin/ssh"),
        };
        let argv = forward_argv(&bins, "xencode", Path::new("/k/k"), 18000, 8000);
        assert_eq!(argv[0], "/usr/bin/ssh");
        assert!(argv.contains(&"-N".to_string()));
        assert!(argv.contains(&"127.0.0.1:18000:127.0.0.1:8000".to_string()));
        assert!(argv.contains(&"-o".to_string()));
        let opts: Vec<&String> = argv.iter().collect();
        assert!(opts.contains(&&"BatchMode=yes".to_string()));
        assert!(opts.contains(&&"ExitOnForwardFailure=yes".to_string()));
        assert_eq!(argv.last().map(String::as_str), Some("colab-vm"));
    }

    #[test]
    fn forward_url_is_openai_compatible_v1() {
        assert_eq!(forward_url(18000), "http://127.0.0.1:18000/v1");
    }

    #[test]
    fn resolve_binaries_reports_missing_colab_with_a_fix() {
        let bin_dir = temp_dir("resolve-colab");
        write_script(&bin_dir, "ssh", "#!/bin/sh\nexit 0\n");
        let xcode_dir = temp_dir("resolve-colab-cfg");
        let _g = with_env(&bin_dir, &xcode_dir);

        let err = resolve_binaries().expect_err("no colab -> error");
        assert!(err.contains("google-colab-cli not on PATH"));
    }

    #[test]
    fn resolve_binaries_reports_missing_ssh_with_a_fix() {
        let bin_dir = temp_dir("resolve-ssh");
        write_script(&bin_dir, "colab", "#!/bin/sh\nexit 0\n");
        let xcode_dir = temp_dir("resolve-ssh-cfg");
        let _g = with_env(&bin_dir, &xcode_dir);

        let err = resolve_binaries().expect_err("no ssh -> error");
        assert!(err.contains("OpenSSH client"));
    }

    #[test]
    fn resolve_binaries_finds_both_when_present() {
        let bin_dir = temp_dir("resolve-both");
        write_script(&bin_dir, "colab", "#!/bin/sh\nexit 0\n");
        write_script(&bin_dir, "ssh", "#!/bin/sh\nexit 0\n");
        let xcode_dir = temp_dir("resolve-both-cfg");
        let _g = with_env(&bin_dir, &xcode_dir);

        let bins = resolve_binaries().expect("both present");
        assert_eq!(bins.colab, bin_dir.join("colab"));
        assert_eq!(bins.ssh, bin_dir.join("ssh"));
    }

    #[test]
    fn pid_alive_0_is_never_alive() {
        assert!(!pid_alive(0));
    }

    #[test]
    fn pid_alive_tracks_a_spawned_process() {
        // /bin/sleep exists regardless of PATH; the pid probe uses the kernel.
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let pid = child.id();
        assert!(pid_alive(pid), "live process reported alive");
        child.kill().expect("kill sleep");
        child.wait().expect("reap sleep");
        // Give the kernel a beat to mark the process gone.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!pid_alive(pid), "reaped process reported dead");
    }
}
