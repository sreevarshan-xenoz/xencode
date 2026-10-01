//! Backend impl #1: Google Colab over the official `colab ssh` bridge.
//!
//! What is Colab-specific (and stays here) versus generic (in `orchestrate`
//! / `lifecycle` / `backend`):
//!
//! * the transport: every ssh invocation rides `colab ssh --proxy-mode` as
//!   its `ProxyCommand`, and logs in as `root` — the bridge injects the
//!   pubkey for `root` only (verified live on a free-tier T4);
//! * provisioning: `colab sessions` to list, `colab new --gpu` to create
//!   (which also spawns the official keep-alive daemon), `colab stop` to
//!   release;
//! * the single-bridge slot: a runtime serves one SSH bridge and a dead
//!   bridge's slot takes ~45 s to free, so `is_transient` names the two
//!   refusal shapes worth waiting out;
//! * the 12-hour reap hint: free-tier VMs are dropped after ~12 h, so a stale
//!   bridge of that age with a dead endpoint is "reaped", not "broken".

use std::path::{Path, PathBuf};

use crate::backend::{first_line, run_capture, Backend, TransportCmd};
use crate::orchestrate::{shell_quote, started_age_hours, Binaries};

/// Timeout for one `colab sessions` / `colab new` / `colab stop` run.
pub(crate) const CMD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Colab reaps idle free-tier VMs; a bridge this old is likely pointing at a
/// dead VM even when a stale pid survives.
pub(crate) const VM_MAX_AGE_HOURS: f64 = 12.0;

/// The user Colab's sshd accepts. Verified live on a free-tier T4 runtime
/// (2026-09-23): the bridge injects the pubkey for `root` only — logging in as
/// the local username, `colab`, `sree` or `user` all end in
/// `Permission denied (publickey)`.
pub const SSH_USER: &str = "root";

/// The Colab backend: its binaries, and the key the bridge authenticates with.
#[derive(Debug, Clone)]
pub struct ColabBackend {
    pub bins: Binaries,
    pub key: PathBuf,
}

impl ColabBackend {
    pub fn new(bins: Binaries, key: PathBuf) -> Self {
        ColabBackend { bins, key }
    }
}

impl Backend for ColabBackend {
    fn id(&self) -> &'static str {
        "colab"
    }

    /// Create the session only when it does not exist server-side (idempotent
    /// `up`). `colab new` also launches the official keep-alive daemon.
    async fn provision(&self, session: &str, gpu: &str) -> Result<(), String> {
        let existing = self
            .list_sessions()
            .await
            .map_err(|e| format!("could not list sessions: {e}"))?;
        if existing.iter().any(|n| n == session) {
            return Ok(());
        }
        let out = run_capture(
            &self.bins.colab,
            &colab_new_argv(&self.bins.colab, session, gpu),
            CMD_TIMEOUT,
        )
        .await
        .map_err(|e| format!("colab new failed: {e}"))?;
        if !out.status {
            return Err(format!(
                "colab new failed{}",
                first_line(&String::from_utf8_lossy(&out.stderr))
                    .map(|l| format!(" — {l}"))
                    .unwrap_or_default()
            ));
        }
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>, String> {
        let output = run_capture(
            &self.bins.colab,
            &colab_sessions_argv(&self.bins.colab),
            CMD_TIMEOUT,
        )
        .await?;
        Ok(parse_sessions(&String::from_utf8_lossy(&output.stdout)))
    }

    /// `colab stop -s <session>` — best-effort: absence is fine, hard backend
    /// errors surface as a message from `down`, never a crash.
    async fn deprovision(&self, session: &str) -> Result<(), String> {
        let argv = colab_stop_argv(&self.bins.colab, session);
        let out = run_capture(&self.bins.colab, &argv, CMD_TIMEOUT).await?;
        if out.status {
            Ok(())
        } else {
            Err(first_line(&String::from_utf8_lossy(&out.stderr))
                .unwrap_or_else(|| "colab stop failed".to_string()))
        }
    }

    fn forward_command(&self, session: &str, local_port: u16, remote_port: u16) -> TransportCmd {
        TransportCmd {
            exe: self.bins.ssh.clone(),
            argv: forward_argv(&self.bins, session, &self.key, local_port, remote_port),
        }
    }

    fn exec_command(&self, session: &str, command: &str) -> TransportCmd {
        TransportCmd {
            exe: self.bins.ssh.clone(),
            argv: exec_ssh_argv(&self.bins, session, &self.key, command),
        }
    }

    fn reap_hint(&self, started_at: Option<&str>, endpoint_ok: bool) -> Option<String> {
        let age = started_at.and_then(started_age_hours)?;
        if age > VM_MAX_AGE_HOURS && !endpoint_ok {
            Some(format!(
                "vm: reaped (started {age:.0}h ago — Colab drops idle VMs after \
                 ~12h). Run `xencode colab up --reconnect` for a fresh VM."
            ))
        } else {
            None
        }
    }

    /// The two bridge-slot refusals seen on a real free-tier runtime are worth
    /// waiting out; a genuine failure (bad key, dead session) is not:
    /// the proxy refusing a second bridge (`HTTP 429 … Already-active SSH
    /// session`), and ssh connecting to the bridge but never getting an SSH
    /// banner because the dying bridge still owns the runtime's sshd slot
    /// (`Connection timed out during banner exchange`).
    fn is_transient(&self, err: &str) -> bool {
        err.contains("Already-active SSH session")
            || err.contains("HTTP 429")
            || err.contains("banner exchange")
    }
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
        "-l".to_string(),
        SSH_USER.to_string(),
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

/// The argv of a *remote command* over the same colab bridge: ssh without
/// `-N`, ending in the target `colab-vm` + a single command string. `bash -s`
/// (reading the bootstrap from stdin) is the shell form we run.
pub fn exec_ssh_argv(bins: &Binaries, session: &str, key: &Path, command: &str) -> Vec<String> {
    vec![
        bins.ssh.display().to_string(),
        "-l".to_string(),
        SSH_USER.to_string(),
        "-i".to_string(),
        key.display().to_string(),
        "-o".to_string(),
        proxy_command_opt(&bins.colab, session, key),
        "-o".to_string(),
        "BatchMode=yes".to_string(),
        "-o".to_string(),
        "StrictHostKeyChecking=no".to_string(),
        "-o".to_string(),
        "UserKnownHostsFile=/dev/null".to_string(),
        "-o".to_string(),
        "ServerAliveInterval=15".to_string(),
        "-o".to_string(),
        "ServerAliveCountMax=2".to_string(),
        "colab-vm".to_string(),
        command.to_string(),
    ]
}

/// argv of `colab sessions` (backend reachability + the session list).
pub fn colab_sessions_argv(colab: &Path) -> Vec<String> {
    vec![colab.display().to_string(), "sessions".to_string()]
}

/// argv of `colab new --gpu <gpu> -s <session>` — verified against the real
/// CLI (`--session/-s`, `--gpu T4|L4|G4|H100|A100`). `new` also spawns the
/// official keep-alive daemon, which is what keeps the VM alive.
pub fn colab_new_argv(colab: &Path, session: &str, gpu: &str) -> Vec<String> {
    vec![
        colab.display().to_string(),
        "new".to_string(),
        "--gpu".to_string(),
        gpu.to_string(),
        "-s".to_string(),
        session.to_string(),
    ]
}

/// argv of `colab stop -s <session>` (tears the VM down and kills its
/// keep-alive daemon).
pub fn colab_stop_argv(colab: &Path, session: &str) -> Vec<String> {
    vec![
        colab.display().to_string(),
        "stop".to_string(),
        "-s".to_string(),
        session.to_string(),
    ]
}

/// Parse `colab sessions` output into the session names it lists. Lines look
/// like `[name] endpoint | Hardware: X | Shape: Y | Variant: Z`; the
/// "no active sessions" notice has the reserved `[colab]` name and is dropped.
pub fn parse_sessions(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let name = line
                .strip_prefix('[')
                .and_then(|rest| rest.split(']').next())
                .map(|n| n.trim())
                .unwrap_or("");
            if name.is_empty() || name == "colab" {
                None
            } else {
                Some(name.to_string())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
        // Colab's sshd only accepts the bridge-injected key for `root`; every
        // other login ends in `Permission denied (publickey)`.
        let l = argv.iter().position(|a| a == "-l").expect("-l root");
        assert_eq!(argv[l + 1], "root");
        let opts: Vec<&String> = argv.iter().collect();
        assert!(opts.contains(&&"BatchMode=yes".to_string()));
        assert!(opts.contains(&&"ExitOnForwardFailure=yes".to_string()));
        assert_eq!(argv.last().map(String::as_str), Some("colab-vm"));
    }

    #[test]
    fn parse_sessions_keeps_names_and_drops_the_notice() {
        let out = "\
[xencode-t4] http://colab-xyz.web.app | Hardware: T4 | Shape: STANDARD | Variant: GPU
[mine2] http://colab-abc.web.app | Hardware: None | Shape: STANDARD | Variant: DEFAULT
[colab] No active sessions found on server.
";
        assert_eq!(parse_sessions(out), vec!["xencode-t4", "mine2"]);
        assert!(parse_sessions("").is_empty());
        assert!(parse_sessions("[colab] No active sessions found on server.").is_empty());
    }

    #[test]
    fn colab_subcommand_argv_matches_the_real_cli() {
        assert_eq!(
            colab_sessions_argv(Path::new("/bin/colab")),
            vec!["/bin/colab", "sessions"]
        );
        assert_eq!(
            colab_new_argv(Path::new("/bin/colab"), "xencode-t4", "T4"),
            vec!["/bin/colab", "new", "--gpu", "T4", "-s", "xencode-t4"]
        );
        assert_eq!(
            colab_stop_argv(Path::new("/bin/colab"), "xencode-t4"),
            vec!["/bin/colab", "stop", "-s", "xencode-t4"]
        );
    }

    #[test]
    fn exec_ssh_argv_runs_a_remote_command_over_the_bridge() {
        let bins = Binaries {
            colab: PathBuf::from("/bin/colab"),
            ssh: PathBuf::from("/usr/bin/ssh"),
        };
        let argv = exec_ssh_argv(&bins, "mine", Path::new("/k/k"), "bash -s");
        assert_eq!(argv[0], "/usr/bin/ssh");
        assert!(!argv.contains(&"-N".to_string()), "no tunnel flag");
        assert_eq!(argv.last().map(String::as_str), Some("bash -s"));
        assert!(
            argv.iter().any(|a| a.starts_with("ProxyCommand=")),
            "bridge present"
        );
        assert!(argv.iter().any(|a| a == "BatchMode=yes"));
    }

    /// The two bridge refusals seen on a real free-tier runtime are retried;
    /// a genuine failure (bad key, dead session) is not.
    #[test]
    fn only_bridge_slot_errors_are_worth_waiting_out() {
        let backend = ColabBackend::new(
            Binaries {
                colab: PathBuf::from("/bin/colab"),
                ssh: PathBuf::from("/usr/bin/ssh"),
            },
            PathBuf::from("/k/k"),
        );
        assert!(backend.is_transient(
            "colab up: VM bootstrap failed (exit 1) — HTTP 429 Already-active SSH session"
        ));
        assert!(backend.is_transient(
            "colab up: VM bootstrap failed (exit 255) — Connection timed out during banner exchange"
        ));
        assert!(
            !backend.is_transient("sree@colab-vm: Permission denied (publickey)"),
            "an auth failure must surface, not retry"
        );
        assert!(
            !backend.is_transient("colab up: VM bootstrap did not report READY"),
            "a broken runtime must surface, not retry"
        );
    }

    #[test]
    fn reap_hint_names_a_stale_dead_bridge_and_nothing_else() {
        let backend = ColabBackend::new(
            Binaries {
                colab: PathBuf::from("/bin/colab"),
                ssh: PathBuf::from("/usr/bin/ssh"),
            },
            PathBuf::from("/k/k"),
        );
        let hint = backend
            .reap_hint(Some("2020-01-02T03:04:05Z"), false)
            .expect("old dead bridge gets a hint");
        assert!(hint.contains("reaped"), "{hint}");
        assert!(hint.contains("--reconnect"), "{hint}");
        assert!(
            backend
                .reap_hint(Some("2020-01-02T03:04:05Z"), true)
                .is_none(),
            "a serving endpoint is not reaped whatever its age"
        );
        assert!(
            backend.reap_hint(None, false).is_none(),
            "no stamp, no hint"
        );
    }
}
