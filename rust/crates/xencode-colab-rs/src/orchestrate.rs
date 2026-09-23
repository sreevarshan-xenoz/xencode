//! Bridge orchestration core: builds and spawns the local `ssh -L` forward
//! that carries the Colab VM's OpenAI-compatible endpoint to the laptop.
//!
//! The forward is one OpenSSH process with the colab bridge as its
//! `ProxyCommand`:
//!
//! ```text
//! ssh -N -L 127.0.0.1:18000:127.0.0.1:8080 -i <key> \
//!     -o ProxyCommand="colab ssh --proxy-mode -s <session> -i <key>" \
//!     root@colab
//! ```
//!
//! OpenSSH runs `ProxyCommand` through the user's shell, so every value we
//! interpolate into it ([`shell_quote`]) is single-quoted — a session name is
//! user-typed and must never be interpreted as shell syntax. The `-L` source
//! port is the laptop's `local_port`; the destination is `127.0.0.1` inside
//! the VM, where the runtime is pinned (llama.cpp 8080, ollama 11434) unless
//! the config overrides it.
//!
//! Beyond the forward itself, the same ssh is used to *bootstrap* the VM
//! (`-N` omitted, `bash -s` fed on stdin) — the bridge is ssh's transport, so
//! a remote command is just ssh without the tunnel flag.

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

/// Terminate a process we spawned: SIGTERM first, escalate to SIGKILL after a
/// short grace period. Idempotent against an already-dead pid.
pub fn terminate(pid: u32) {
    if pid == 0 || !pid_alive(pid) {
        return;
    }
    // SAFETY: kill(2) on a pid we spawned; a zombie only errno ESRCH.
    unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    for _ in 0..5 {
        if !pid_alive(pid) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if pid_alive(pid) {
        // SAFETY: as above.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
    }
}

/// Session names flow through OpenSSH's `ProxyCommand` (a shell string, even
/// though we single-quote it) and into the VM's process list, so they are
/// restricted to a safe charset before anything interpolates them. Colab
/// itself does not impose one; we do.
pub fn validate_session_name(name: &str) -> Result<(), String> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if valid {
        Ok(())
    } else {
        Err(format!(
            "invalid session name {name:?}: use 1-64 chars of [A-Za-z0-9_-]"
        ))
    }
}

/// The port the inference server listens on *inside* the VM. `configured` is
/// the config's `remote_port`; `0` means "use the runtime's native port"
/// (llama.cpp pins 8080, ollama 11434) so a config written before the runtime
/// was chosen still boots the right endpoint.
pub fn effective_remote_port(runtime: &str, configured: u16) -> u16 {
    if configured != 0 {
        return configured;
    }
    match runtime {
        "ollama" => 11434,
        _ => 8080,
    }
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

/// The argv of a *remote command* over the same colab bridge: ssh without
/// `-N`, ending in the target `colab-vm` + a single command string. `bash -s`
/// (reading the bootstrap from stdin) is the shell form we run.
pub fn exec_ssh_argv(bins: &Binaries, session: &str, key: &Path, command: &str) -> Vec<String> {
    vec![
        bins.ssh.display().to_string(),
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

/// A minimal HTTP GET of `/v1/models` on the forward's local port, retried
/// `attempts` times with a short gap — the liveness probe `status` reuses and
/// `up` waits on after spawning the tunnel. `Ok` once the endpoint answers
/// HTTP 200 any way, `Err` with the last failure after exhausting retries.
pub async fn probe_models(
    local_port: u16,
    attempts: u32,
    roundtrip: Option<std::time::Duration>,
) -> Result<(), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut last_err = "no attempt".to_string();
    for _ in 0..attempts {
        match tokio::time::timeout(
            roundtrip.unwrap_or(std::time::Duration::from_secs(3)),
            async {
                let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{local_port}"))
                    .await
                    .map_err(|e| format!("connect to forward: {e}"))?;
                stream
                    .write_all(b"GET /v1/models HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
                    .await
                    .map_err(|e| format!("write probe: {e}"))?;
                let mut buf = Vec::new();
                stream
                    .read_to_end(&mut buf)
                    .await
                    .map_err(|e| format!("read probe: {e}"))?;
                let head = String::from_utf8_lossy(&buf[..buf.len().min(16)]);
                if head.starts_with("HTTP/1.0 200") || head.starts_with("HTTP/1.1 200") {
                    Ok(())
                } else {
                    Err(format!("forward answered {head:?}"))
                }
            },
        )
        .await
        {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(e)) => last_err = e,
            Err(_) => last_err = "probe timed out".to_string(),
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
    Err(format!(
        "endpoint did not answer /v1/models after {attempts} attempt(s): {last_err}"
    ))
}

/// Current local time formatted as RFC3339 (`YYYY-MM-DDTHH:MM:SS+ZZ:ZZ`) — the
/// `started_at` stamp. Deterministic enough for state, no datetime dep.
pub fn now_rfc3339() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as libc::time_t)
        .unwrap_or(0);
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: localtime_r fills `tm` from an epoch seconds value; the struct
    // outlives the call.
    unsafe { libc::localtime_r(&now, &mut tm) };
    let off = tm.tm_gmtoff;
    let sign = if off < 0 { '-' } else { '+' };
    let abs = off.unsigned_abs();
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{sign}{:02}:{:02}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        abs / 3600,
        (abs % 3600) / 60,
    )
}

/// Age in hours of a `started_at` stamp (`YYYY-MM-DDTHH:MM:SS({+,-}HH:MM|Z)`)
/// relative to the wall clock, or `None` when the stamp is unparseable.
/// Colab reaps idle VMs (free tier keeps sessions ~12h), so `status` uses this
/// to tell "VM reaped" apart from a transient blip.
pub fn started_age_hours(started_at: &str) -> Option<f64> {
    let b = started_at.as_bytes();
    if b.len() < 20 {
        return None;
    }
    let year: i64 = started_at.get(0..4)?.parse().ok()?;
    let mon: i64 = started_at.get(5..7)?.parse().ok()?;
    let day: i64 = started_at.get(8..10)?.parse().ok()?;
    let hour: i64 = started_at.get(11..13)?.parse().ok()?;
    let min: i64 = started_at.get(14..16)?.parse().ok()?;
    let sec: i64 = started_at.get(17..19)?.parse().ok()?;
    if !(1..=12).contains(&mon) || !(1..=31).contains(&day) || hour > 23 || min > 59 || sec > 61 {
        return None;
    }

    // Offset seconds past `+HH:MM`/`-HH:MM`; `Z`/`z` (or a bare timestamp
    // without one) is UTC.
    let offset = match b.get(19) {
        Some(b'Z') | Some(b'z') => 0,
        Some(b'+') | Some(b'-') => {
            if b.len() < 25 {
                return None;
            }
            let oh: i64 = started_at.get(20..22)?.parse().ok()?;
            let om: i64 = started_at.get(23..25)?.parse().ok()?;
            if oh > 14 || om > 59 {
                return None;
            }
            if b[19] == b'-' {
                -(oh * 3600 + om * 60)
            } else {
                oh * 3600 + om * 60
            }
        }
        _ => 0,
    };

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    // Days-from-civil (Howard Hinnant's algorithm): epoch day for the date's
    // UTC instant, then combine with clock and offset.
    let y = year - if mon <= 2 { 1 } else { 0 };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (if mon > 2 { mon - 3 } else { mon + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let epoch_day = era * 146097 + doe - 719468;
    let stamp = epoch_day * 86400 + hour * 3600 + min * 60 + sec - offset;
    Some((now - stamp) as f64 / 3600.0)
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

    #[test]
    fn terminate_kills_an_uncooperative_process() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let pid = child.id();
        // Our own SIGTERM target: a plain sleep ignores SIGTERM until we SIGKILL.
        terminate(pid);
        child.wait().expect("terminate reaps");
        assert!(!pid_alive(pid), "terminate left the process alive");
        // Idempotent on a dead pid.
        terminate(pid);
    }

    #[test]
    fn session_names_are_restricted_to_a_safe_charset() {
        assert!(validate_session_name("xencode-t4").is_ok());
        assert!(validate_session_name("a").is_ok());
        assert!(validate_session_name("ab_c-D9").is_ok());
        assert!(validate_session_name("").is_err());
        assert!(validate_session_name("has space").is_err());
        assert!(validate_session_name("$(rm)").is_err());
        assert!(validate_session_name(&"x".repeat(65)).is_err());
        assert!(validate_session_name("unïcode").is_err());
    }

    #[test]
    fn effective_remote_port_uses_runtime_native_when_unconfigured() {
        assert_eq!(effective_remote_port("llama.cpp", 0), 8080);
        assert_eq!(effective_remote_port("ollama", 0), 11434);
        assert_eq!(effective_remote_port("llama.cpp", 9000), 9000);
        assert_eq!(effective_remote_port("ollama", 7000), 7000);
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

    #[tokio::test]
    async fn probe_models_fails_when_nothing_is_listening() {
        // Pick a port that is overwhelmingly free then never bind it.
        let err = probe_models(1, 2, None)
            .await
            .expect_err("nothing listening");
        assert!(err.contains("/v1/models"), "error names the probe: {err}");
    }

    #[test]
    fn now_rfc3339_looks_like_a_timestamp() {
        let ts = now_rfc3339();
        let bytes = ts.as_bytes();
        assert!(bytes.len() >= 19, "at least YYYY-MM-DDTHH:MM:SS: {ts}");
        assert_eq!(bytes[4], b'-', "dash after year: {ts}");
        assert_eq!(bytes[7], b'-', "dash after month: {ts}");
        assert_eq!(bytes[10], b'T', "T separator: {ts}");

        // Round-trip: `now_rfc3339` must parse back to ~0h of age.
        let age = started_age_hours(&ts).expect("our own stamp parses");
        assert!(
            (0.0..=2.0).contains(&age),
            "recent stamp ages as nearly zero, got {age}h"
        );
    }

    #[test]
    fn started_age_hours_pins_offsets_and_parses() {
        // A stamp for 2026-09-23T06:00:00+00:00. When the local offset is
        // +1000-ish it is ~16h back; the point is the parse is reproducible
        // against a fixed UTC instant, so check the offset math directly:
        // UTC-noon is stored as -0500 afternoon on the same day.
        let utc_noon = "2026-09-23T12:00:00Z";
        let neg = "2026-09-23T07:00:00-05:00";
        let plus = "2026-09-23T17:00:00+05:00";
        let a = started_age_hours(utc_noon).expect("Z parses");
        let b = started_age_hours(neg).expect("-05:00 parses");
        let c = started_age_hours(plus).expect("+05:00 parses");
        // All three denote the same instant, so ages agree to ~1e-6 h.
        for (lbl, v) in [("Z", a), ("-05:00", b), ("+05:00", c)] {
            assert!(
                (v - a).abs() < 1e-6,
                "same instant in {lbl} ages like Z: got {v}, Z {a}"
            );
        }

        let junk = started_age_hours("not a date");
        assert!(junk.is_none(), "garbage -> None");

        let future = started_age_hours("2099-01-02T03:04:05Z").expect("future parses");
        assert!(future < 0.0, "future stamp ages negative: {future}");
    }
}
