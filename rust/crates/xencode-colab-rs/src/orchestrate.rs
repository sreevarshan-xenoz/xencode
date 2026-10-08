//! Bridge orchestration core: the generic half of the `ssh -L` forward that
//! carries a VM's OpenAI-compatible endpoint to the laptop.
//!
//! Colab specifics (the `colab ssh --proxy-mode` `ProxyCommand`, `root`
//! login, `colab new/sessions/stop` argv) live in [`crate::colab`]; this
//! module owns what every backend reuses: resolving the tools, quoting,
//! spawning and probing the forward, pid liveness, session-name validation,
//! runtime port defaults, and the clock helpers behind state stamps.
//!
//! Beyond the forward itself, the same ssh is used to *bootstrap* the VM
//! (`-N` omitted, `bash -s` fed on stdin) — the bridge is ssh's transport, so
//! a remote command is just ssh without the tunnel flag.

use std::path::PathBuf;

use crate::backend::TransportCmd;
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

/// The URL the forward exposes — what `xencode config set remote_url` points
/// at. `/v1` is the OpenAI-compatible API root served on the VM.
pub fn forward_url(local_port: u16) -> String {
    format!("http://127.0.0.1:{local_port}/v1")
}

/// Spawn the forward as a background process from a backend-built command.
/// Returns the URL it exposes and the child handle. The forward outlives this
/// function, so it holds no handle the caller owns: stderr is discarded (an
/// inherited stderr keeps a `xencode colab up | grep` pipeline open until
/// the tunnel dies, seen live), stdin closed, stdout discarded.
/// Bridge/connect refusals surface through the exit status in the caller's
/// settle check.
pub async fn spawn_forward_cmd(
    cmd: &TransportCmd,
    local_port: u16,
) -> Result<(String, tokio::process::Child), String> {
    let child = tokio::process::Command::new(&cmd.exe)
        .args(cmd.argv.iter().skip(1))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("could not spawn ssh forward: {e}"))?;
    Ok((forward_url(local_port), child))
}

/// True when the process with `pid` exists and has not exited. `0` is never a
/// real pid we track.
pub fn pid_alive(pid: u32) -> bool {
    xencode_core_rs::sys::pid_alive(pid)
}

/// Terminate a process we spawned: ask first, force after a short grace
/// period. Idempotent against an already-dead pid.
pub fn terminate(pid: u32) {
    xencode_core_rs::sys::terminate(pid)
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
/// (llama.cpp 18080, ollama 11434) so a config written before the runtime was
/// chosen still boots the right endpoint.
///
/// llama.cpp's own default is 8080, but a Colab runtime cannot use it: the
/// notebook container runs its own node proxy on `*:8080` (measured live on a
/// free-tier T4, `ss -tlnp` pid 6). Binding there fails outright, so the
/// bridge picks a port the VM has free.
pub fn effective_remote_port(runtime: &str, configured: u16) -> u16 {
    if configured != 0 {
        return configured;
    }
    match runtime {
        "ollama" => 11434,
        _ => 18080,
    }
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
/// `started_at` stamp, to the whole second.
pub fn now_rfc3339() -> String {
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%:z")
        .to_string()
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
        assert_eq!(effective_remote_port("llama.cpp", 0), 18080);
        assert_eq!(effective_remote_port("ollama", 0), 11434);
        assert_eq!(effective_remote_port("llama.cpp", 9000), 9000);
        assert_eq!(effective_remote_port("ollama", 7000), 7000);
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
