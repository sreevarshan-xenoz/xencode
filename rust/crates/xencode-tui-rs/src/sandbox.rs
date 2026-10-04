//! SE-7 / QTR-3 — a `bubblewrap` sandbox for `run_command`, `background_start`
//! and shell hooks.
//!
//! An approved shell command runs with the full privileges of whoever started
//! xencode, so a credential the model was told to read, or a hook the config
//! names, can reach `~/.ssh`, cloud tokens and the rest of the home the same way
//! you can. This wraps a command in a `bwrap` mount namespace: the workspace and
//! `~/.cargo` stay writable so a build still works, the rest of the home (the
//! `~/.ssh` directory among them) is mounted over with an empty tmpfs so it is
//! not merely denied but *absent*, and the network namespace is dropped unless
//! the call explicitly asks for it.
//!
//! Two rules this module is built around:
//!
//! - **No silent fallback.** When the sandbox is asked for and `bwrap` is not
//!   there, [`Sandbox::wrap`] returns an error rather than handing back an
//!   unsandboxed command. A guard that quietly steps aside when its tool is
//!   missing is not a guard, and the whole point is that the caller cannot tell
//!   the difference from the outcome.
//! - **It does not claim to contain the compiler.** `build.rs` scripts run free
//!   inside the writable workspace, and anything the workspace can reach is
//!   reachable. This bounds what an arbitrary command can read outside the
//!   project; it is not a full jail and is not sold as one.

use std::path::{Path, PathBuf};

/// The resolved sandbox decision for a run: whether it is wanted, whether this
/// machine can do it, and the paths the namespace is built around.
#[derive(Debug, Clone)]
pub struct Sandbox {
    /// `run_command_sandbox` — whether the person asked for isolation.
    enabled: bool,
    /// Whether `bwrap` answered `--version` when this was built.
    available: bool,
    /// The workspace, bind-mounted read-write.
    workspace: PathBuf,
    /// `~/.cargo`, bind-mounted read-write so cargo can still fetch and build.
    cargo: PathBuf,
    /// The home directory, mounted over with an empty tmpfs.
    home: PathBuf,
}

impl Sandbox {
    /// Build the policy for a run at `workspace`, honouring the `enabled`
    /// switch. Probing `bwrap` is the only work that touches the machine, and it
    /// is one quick `--version`; an environment without `bwrap` resolves to
    /// `available: false`, which turns an enabled sandbox into a refusal rather
    /// than a silent pass-through.
    pub fn resolve(enabled: bool, workspace: &Path) -> Sandbox {
        Sandbox {
            enabled,
            available: bwrap_present(),
            workspace: workspace.to_path_buf(),
            cargo: cargo_home(),
            home: home_dir(),
        }
    }

    /// A sandbox that never wraps: the path `run_command` takes when isolation is
    /// off, and what a test that does not care about the namespace passes in.
    pub fn disabled() -> Sandbox {
        Sandbox {
            enabled: false,
            available: true,
            workspace: PathBuf::new(),
            cargo: PathBuf::new(),
            home: PathBuf::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn available(&self) -> bool {
        self.available
    }

    /// What `run_command` should spawn: either the plain `sh -c` (isolation off),
    /// a `bwrap … sh -c` (on and able), or an error (on but `bwrap` missing).
    ///
    /// `net` is the per-call `--net` grant: with isolation on the network is off
    /// unless the approved command asked for it.
    pub fn wrap(&self, command: &str, net: bool) -> Result<Option<(String, Vec<String>)>, String> {
        if !self.enabled {
            return Ok(None);
        }
        if !self.available {
            return Err(
                "run_command_sandbox is on but `bwrap` is not installed — refusing to run the \
                 command unsandboxed rather than silently skipping the isolation. Install \
                 bubblewrap, or set run_command_sandbox = false."
                    .to_string(),
            );
        }
        let mut args: Vec<String> = vec![
            "--die-with-parent".to_string(),
            "--unshare-all".to_string(),
            // The workspace and ~/.cargo are re-bound after the home tmpfs, so
            // they survive being mounted over; ordering is the whole mechanism.
            "--ro-bind".into(),
            "/usr".into(),
            "/usr".into(),
            "--ro-bind".into(),
            "/bin".into(),
            "/bin".into(),
            "--ro-bind-try".into(),
            "/lib".into(),
            "/lib".into(),
            "--ro-bind-try".into(),
            "/lib64".into(),
            "/lib64".into(),
            "--ro-bind-try".into(),
            "/etc".into(),
            "/etc".into(),
            "--ro-bind-try".into(),
            "/opt".into(),
            "/opt".into(),
            "--dev".into(),
            "/dev".into(),
            "--proc".into(),
            "/proc".into(),
            "--tmpfs".into(),
            self.home.to_string_lossy().into_owned(),
            "--bind".into(),
            "/tmp".into(),
            "/tmp".into(),
            "--bind".into(),
            self.workspace.to_string_lossy().into_owned(),
            self.workspace.to_string_lossy().into_owned(),
        ];
        if self.cargo.exists() {
            args.extend([
                "--bind".to_string(),
                self.cargo.to_string_lossy().into_owned(),
                self.cargo.to_string_lossy().into_owned(),
            ]);
        }
        args.extend([
            "--setenv".to_string(),
            "HOME".to_string(),
            self.home.to_string_lossy().into_owned(),
        ]);
        // --unshare-all already drops the network; a granted --net puts it back.
        if net {
            args.push("--share-net".to_string());
        }
        args.push("sh".to_string());
        args.push("-c".to_string());
        args.push(command.to_string());
        Ok(Some(("bwrap".to_string(), args)))
    }
}

/// Whether the `bwrap` binary answers to `--version` on this `PATH`.
fn bwrap_present() -> bool {
    std::process::Command::new("bwrap")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// The cargo home: `$CARGO_HOME`, else `~/.cargo`.
fn cargo_home() -> PathBuf {
    match std::env::var_os("CARGO_HOME") {
        Some(dir) => PathBuf::from(dir),
        None => home_dir().join(".cargo"),
    }
}

/// The home directory: `$HOME`, falling back to the passwd entry via `dirs`
/// is not needed here — `HOME` is what `run_command` inherits anyway.
fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole reason this exists: with the sandbox on, a command cannot see a
    /// secret that sits in the home. `wrap` builds the namespace, and a real run
    /// of it is what the done-when test exercises; here we assert the shape that
    /// makes it true — the home is tmpfs'd and the network is dropped — and that
    /// the opt-ins reverse it.
    #[test]
    fn wrap_builds_a_namespace_that_hides_home_and_drops_net() {
        let sb = Sandbox {
            enabled: true,
            available: true,
            workspace: PathBuf::from("/work"),
            cargo: PathBuf::from("/nonexistent-cargo-home"),
            home: PathBuf::from("/home/person"),
        };
        let (program, args) = sb.wrap("cat ~/.ssh/id_rsa", false).unwrap().unwrap();
        assert_eq!(program, "bwrap");
        let joined = args.join(" ");
        assert!(joined.contains("--unshare-all"), "{joined}");
        assert!(joined.contains("--tmpfs /home/person"), "{joined}");
        // The workspace bind comes AFTER the home tmpfs, which is what keeps it
        // reachable while the rest of the home is gone.
        let tmpfs_at = joined.find("--tmpfs /home/person").unwrap();
        let bind_at = joined.find("--bind /work /work").unwrap();
        assert!(
            bind_at > tmpfs_at,
            "workspace must be re-bound after the tmpfs"
        );
        assert!(
            !joined.contains("--share-net"),
            "net must stay off by default"
        );
        assert!(joined.ends_with("sh -c cat ~/.ssh/id_rsa"), "{joined}");
    }

    #[test]
    fn a_net_grant_puts_the_network_namespace_back() {
        let sb = Sandbox {
            enabled: true,
            available: true,
            workspace: PathBuf::from("/work"),
            cargo: PathBuf::from("/nonexistent-cargo-home"),
            home: PathBuf::from("/home/person"),
        };
        let (_, args) = sb.wrap("curl example.invalid", true).unwrap().unwrap();
        assert!(args.iter().any(|a| a == "--share-net"));
    }

    #[test]
    fn disabled_sandbox_runs_the_plain_shell() {
        let sb = Sandbox::disabled();
        assert_eq!(sb.wrap("echo hi", false).unwrap(), None);
    }

    /// The trap: isolation asked for, `bwrap` absent — an error, never a
    /// pass-through the caller would read as "it ran, so it was allowed".
    #[test]
    fn enabled_without_bwrap_is_a_refusal_not_a_fallback() {
        let sb = Sandbox {
            enabled: true,
            available: false,
            workspace: PathBuf::from("/work"),
            cargo: PathBuf::from("/c"),
            home: PathBuf::from("/h"),
        };
        let err = sb.wrap("echo hi", false).unwrap_err();
        assert!(
            err.contains("bwrap") && err.contains("unsandboxed"),
            "{err}"
        );
    }
}
