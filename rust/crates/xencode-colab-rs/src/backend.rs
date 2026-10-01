//! The bridge backend seam (L-1): what "a machine you can reach over ssh"
//! means, independent of who provides the machine.
//!
//! Everything Colab-specific used to live inline in `lifecycle.rs` next to
//! the generic sequencing (bootstrap, poll-until-serving, forward spawn/hold,
//! state file, reconnect). That sequencing is identical for any second
//! backend, so it now runs against this trait and Colab is impl #1 in
//! [`crate::colab`]. The three seams are the ones L-1 names:
//!
//! * `provision` — create/attach the compute (plus `list_sessions` to ask
//!   what is attached and `deprovision` to give it back; one seam, three
//!   methods, because status and down need to ask and release without
//!   creating),
//! * the transport — `forward_command` (the argv that carries `ssh -N -L`
//!   style forwarding) and `exec_command` (the argv that runs one remote
//!   command, which is what the bootstrap rides on),
//! * `reap_hint` — the provider-specific "why is it gone" line.
//!
//! Two more methods fell out of the split and are documented where they sit:
//! `id` names the backend in state and errors, and `is_transient` tells the
//! retry loops which failures are worth waiting out (Colab's single-bridge
//! slot; a backend without one keeps the default `false`).

use std::path::PathBuf;
use std::time::Duration;

/// One subprocess to run: the executable plus its full argv (`argv[0]` is the
/// exe itself, mirroring the other builders, and is skipped when spawning).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportCmd {
    pub exe: PathBuf,
    pub argv: Vec<String>,
}

/// A machine provider behind the `up`/`status`/`down` lifecycle. All methods
/// are fallible with end-user fix strings; nothing here panics on a backend
/// being unreachable. Callers take `impl Backend` / `B: Backend` (native
/// `async fn`, no extra dependency) rather than `dyn`, which native async
/// methods are not object-safe for.
#[allow(async_fn_in_trait)] // no `dyn Backend` exists anywhere: every driver
                            // is generic over `B: Backend`, so the auto-trait bounds the lint worries
                            // about are never needed.
pub trait Backend: Send + Sync {
    /// Short name recorded in state (`"colab"`) and used in error prefixes.
    fn id(&self) -> &'static str;

    /// Create the compute when absent, attach when present. Idempotent: an
    /// already-listed session is not created twice.
    async fn provision(&self, session: &str, gpu: &str) -> Result<(), String>;

    /// Sessions the provider currently lists (for `status` and for
    /// `provision`'s idempotence check).
    async fn list_sessions(&self) -> Result<Vec<String>, String>;

    /// Release the compute. Best-effort: absence is fine, hard backend errors
    /// surface as a message, never a crash.
    async fn deprovision(&self, session: &str) -> Result<(), String>;

    /// The `ssh -N -L`-style forward: local `local_port` reaches the inference
    /// server on `remote_port` inside the compute.
    fn forward_command(&self, session: &str, local_port: u16, remote_port: u16) -> TransportCmd;

    /// One remote command over the same transport the forward uses (`bash -s`
    /// reading the bootstrap from stdin, in Colab's case).
    fn exec_command(&self, session: &str, command: &str) -> TransportCmd;

    /// Why a bridge this old is probably gone, in words for `status` — or
    /// `None` when the backend has nothing provider-specific to say.
    /// `started_at` is the state's RFC3339 stamp, `endpoint_ok` whether
    /// `/v1/models` just answered.
    fn reap_hint(&self, started_at: Option<&str>, endpoint_ok: bool) -> Option<String>;

    /// True when `err` is a transient transport-slot refusal worth retrying
    /// rather than a user error to surface. Colab's runtime serves one SSH
    /// bridge and the slot of a dead one takes ~45 s to free; a backend with
    /// no such slot keeps the default.
    fn is_transient(&self, _err: &str) -> bool {
        false
    }
}

/// A captured subprocess run, with a timeout. `argv` mirrors the other
/// builders — `argv[0]` is the exe itself (it must equal `exe`) and is
/// skipped here so `Command` gets its args only.
pub(crate) async fn run_capture(
    exe: &std::path::Path,
    argv: &[String],
    timeout: Duration,
) -> Result<Output, String> {
    tokio::time::timeout(
        timeout,
        tokio::process::Command::new(exe).args(&argv[1..]).output(),
    )
    .await
    .map_err(|_| "command timed out".to_string())?
    .map(|out| Output {
        status: out.status.success(),
        stdout: out.stdout,
        stderr: out.stderr,
    })
    .map_err(|e| format!("could not run {}: {e}", exe.display()))
}

pub(crate) struct Output {
    pub(crate) status: bool,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
}

/// First non-blank line of `s`, trimmed and capped — for error tails.
pub(crate) fn first_line(s: &str) -> Option<String> {
    let line = s.lines().find(|l| !l.trim().is_empty())?;
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.chars().take(160).collect())
}

/// The tail of a subprocess' output — what actually went wrong is at the end;
/// the first line is usually ssh's host-key warning.
pub(crate) fn error_tail(s: &str) -> Option<String> {
    let lines: Vec<&str> = s
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("Warning: Permanently added"))
        .collect();
    if lines.is_empty() {
        return None;
    }
    let skip = lines.len().saturating_sub(3);
    Some(lines[skip..].join(" / ").chars().take(400).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_line_trims_and_caps() {
        assert_eq!(first_line("  hi there\n"), Some("hi there".to_string()));
        assert_eq!(first_line("\n\n  "), None);
        assert_eq!(first_line(&"x".repeat(300)).map(|l| l.len()), Some(160));
    }

    #[test]
    fn error_tail_skips_host_key_warnings_and_keeps_the_end() {
        let out = "Warning: Permanently added 'x' (ED25519) to the list of known hosts.\ninstalling deps\nFAILED badly wrong";
        let tail = error_tail(out).expect("non-empty tail");
        assert!(!tail.contains("Permanently added"), "{tail}");
        assert!(tail.contains("FAILED badly wrong"), "{tail}");
    }

    #[test]
    fn error_tail_of_nothing_is_nothing() {
        assert_eq!(error_tail(""), None);
        assert_eq!(
            error_tail("Warning: Permanently added 'x' to the list.\n"),
            None
        );
    }
}
