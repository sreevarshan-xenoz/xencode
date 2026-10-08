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

/// Pinned boxed future for dynamic dispatch on backends.
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Object-safe companion of [`Backend`] for engine composition (AF-3, AF-4).
///
/// Enables polymorphic selection and invocation of computer backends
/// without hardcoded `match` branches in the execution loops.
pub trait ComputerBackend: std::fmt::Debug + Send + Sync {
    fn id(&self) -> &'static str;
    fn kind(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn is_available(&self) -> (bool, String);
    fn provision<'a>(&'a self, session: &'a str, gpu: &'a str)
        -> BoxFuture<'a, Result<(), String>>;
    fn list_sessions<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, String>>;
    fn deprovision<'a>(&'a self, session: &'a str) -> BoxFuture<'a, Result<(), String>>;
    fn forward_command(&self, session: &str, local_port: u16, remote_port: u16) -> TransportCmd;
    fn exec_command(&self, session: &str, command: &str) -> TransportCmd;
    fn reap_hint(&self, started_at: Option<&str>, endpoint_ok: bool) -> Option<String>;
    fn is_transient(&self, _err: &str) -> bool {
        false
    }
}

impl ComputerBackend for crate::colab::ColabBackend {
    fn id(&self) -> &'static str {
        <Self as Backend>::id(self)
    }

    fn kind(&self) -> &'static str {
        "colab"
    }

    fn description(&self) -> &'static str {
        "Google Colab cloud GPU compute backend"
    }

    fn is_available(&self) -> (bool, String) {
        if !self.bins.colab.is_file() {
            (false, "google-colab-cli not found on PATH".to_string())
        } else if !self.bins.ssh.is_file() {
            (false, "OpenSSH client not found on PATH".to_string())
        } else {
            (true, "google-colab-cli and OpenSSH available".to_string())
        }
    }

    fn provision<'a>(
        &'a self,
        session: &'a str,
        gpu: &'a str,
    ) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move { <Self as Backend>::provision(self, session, gpu).await })
    }

    fn list_sessions<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, String>> {
        Box::pin(async move { <Self as Backend>::list_sessions(self).await })
    }

    fn deprovision<'a>(&'a self, session: &'a str) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move { <Self as Backend>::deprovision(self, session).await })
    }

    fn forward_command(&self, session: &str, local_port: u16, remote_port: u16) -> TransportCmd {
        <Self as Backend>::forward_command(self, session, local_port, remote_port)
    }

    fn exec_command(&self, session: &str, command: &str) -> TransportCmd {
        <Self as Backend>::exec_command(self, session, command)
    }

    fn reap_hint(&self, started_at: Option<&str>, endpoint_ok: bool) -> Option<String> {
        <Self as Backend>::reap_hint(self, started_at, endpoint_ok)
    }

    fn is_transient(&self, err: &str) -> bool {
        <Self as Backend>::is_transient(self, err)
    }
}

/// Metadata and status report of a registered computer backend (AF-4).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ComputerInfo {
    pub id: String,
    pub kind: String,
    pub description: String,
    pub available: bool,
    pub detail: String,
}

/// Constructor for dynamically mounted computer backends.
pub type BackendConstructor = Box<
    dyn Fn(&crate::orchestrate::Binaries, &std::path::Path) -> Box<dyn ComputerBackend>
        + Send
        + Sync,
>;

/// Statically linked mount point for computer backends (AF-3, AF-4).
pub struct BackendRegistry {
    entries: std::collections::BTreeMap<&'static str, BackendConstructor>,
}

impl Default for BackendRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register(
            "colab",
            Box::new(|bins, key| {
                Box::new(crate::colab::ColabBackend::new(
                    bins.clone(),
                    key.to_path_buf(),
                ))
            }),
        );
        reg.register(
            "ssh",
            Box::new(|bins, key| {
                Box::new(crate::ssh::SshBackend::new(bins.clone(), key.to_path_buf()))
            }),
        );
        reg.register(
            "docker",
            Box::new(|_bins, _key| Box::new(crate::docker::DockerBackend::default())),
        );
        reg
    }
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            entries: std::collections::BTreeMap::new(),
        }
    }

    pub fn register(&mut self, id: &'static str, ctor: BackendConstructor) {
        self.entries.insert(id, ctor);
    }

    pub fn available_backends(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }

    pub fn list_computers(
        &self,
        bins: &crate::orchestrate::Binaries,
        key: &std::path::Path,
    ) -> Vec<ComputerInfo> {
        let mut infos = Vec::new();
        for (id, ctor) in &self.entries {
            let backend = ctor(bins, key);
            let (available, detail) = backend.is_available();
            infos.push(ComputerInfo {
                id: (*id).to_string(),
                kind: backend.kind().to_string(),
                description: backend.description().to_string(),
                available,
                detail,
            });
        }
        infos
    }

    pub fn resolve(
        &self,
        id: &str,
        bins: &crate::orchestrate::Binaries,
        key: &std::path::Path,
    ) -> Result<Box<dyn ComputerBackend>, String> {
        match self.entries.get(id) {
            Some(ctor) => Ok(ctor(bins, key)),
            None => Err(format!(
                "unknown computer backend `{id}`; available backends: {}",
                self.available_backends().join(", ")
            )),
        }
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

    #[test]
    fn backend_registry_resolves_statically_registered_backends() {
        let reg = BackendRegistry::default();
        assert_eq!(reg.available_backends(), vec!["colab", "docker", "ssh"]);

        let bins = crate::orchestrate::Binaries {
            colab: std::path::PathBuf::from("/bin/colab"),
            ssh: std::path::PathBuf::from("/bin/ssh"),
        };
        let key = std::path::Path::new("/tmp/id_test");
        let colab_backend = reg
            .resolve("colab", &bins, key)
            .expect("must resolve colab");
        assert_eq!(colab_backend.id(), "colab");
        assert_eq!(colab_backend.kind(), "colab");

        let ssh_backend = reg.resolve("ssh", &bins, key).expect("must resolve ssh");
        assert_eq!(ssh_backend.id(), "ssh");
        assert_eq!(ssh_backend.kind(), "ssh");

        let docker_backend = reg
            .resolve("docker", &bins, key)
            .expect("must resolve docker");
        assert_eq!(docker_backend.id(), "docker");
        assert_eq!(docker_backend.kind(), "docker");

        let err = match reg.resolve("virsh-vm", &bins, key) {
            Err(e) => e,
            Ok(_) => panic!("virsh-vm should not be registered"),
        };
        assert!(err.contains("unknown computer backend `virsh-vm`"), "{err}");

        let computers = reg.list_computers(&bins, key);
        assert_eq!(computers.len(), 3);
        assert_eq!(computers[0].id, "colab");
        assert_eq!(computers[0].kind, "colab");
        assert_eq!(computers[1].id, "docker");
        assert_eq!(computers[1].kind, "docker");
        assert_eq!(computers[2].id, "ssh");
        assert_eq!(computers[2].kind, "ssh");
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn unreachable_ssh_backend_answers_honestly_that_it_cannot_connect() {
        let bins = crate::orchestrate::Binaries {
            colab: std::path::PathBuf::from("/bin/colab"),
            ssh: std::path::PathBuf::from("/usr/bin/ssh"),
        };
        let key = std::path::PathBuf::from("/tmp/nonexistent_test_key");
        // Target an unreachable non-routable address / unused port
        let backend = crate::ssh::SshBackend::new(bins, key)
            .with_destination("192.0.2.1") // RFC 5737 TEST-NET-1 (unreachable)
            .with_port(9999);

        let res = Backend::provision(&backend, "test-session", "none").await;
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("cannot reach ssh host `192.0.2.1`"), "{err}");
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn docker_backend_answers_honestly_when_daemon_unreachable() {
        let backend = crate::docker::DockerBackend::new(
            std::path::PathBuf::from("/usr/bin/docker"),
            "alpine:latest".to_string(),
        );
        let res = Backend::provision(&backend, "test-session", "none").await;
        // On this host, docker daemon is not accessible to non-root, or daemon is not running
        // It must answer honestly with an error, not panic or mock green
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.contains("docker engine unreachable"), "{err}");
    }
}
