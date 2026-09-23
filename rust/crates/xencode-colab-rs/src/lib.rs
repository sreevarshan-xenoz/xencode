//! Google Colab orchestration for xencode.
//!
//! Xencode can serve its OpenAI-compatible inference endpoint from a Colab VM
//! (T4 GPU and friends) and keep a local `ssh -L` port forward to it. This
//! crate owns that bridge lifecycle: the preflight gate (is the `colab` CLI
//! installed, signed in, and is an SSH key ready?), the orchestration core
//! that builds and spawns the forward, the `colab.json` state the lifecycle
//! commands persist, then `up` / `status` / `down` in the same module family.
//!
//! The only supported transport is the official `google-colab-cli`
//! (`colab ssh --proxy-mode`), an authenticated WebSocket SSH route that the
//! Colab CLI itself implements — never a public tunnel. Colab's free tier
//! prohibits ngrok/cloudflared style public tunnels (account suspension), so
//! nothing here ever starts one.
//!
//! `unsafe` is forbidden everywhere except three narrowly-scoped `libc`
//! touchpoints in [`orchestrate`]: `pid_alive` / `terminate` (the `kill`
//! probe + signal) and `now_rfc3339` (`localtime_r`). `preflight`, `state`,
//! `bootstrap`, and `lifecycle` forbid it themselves.

pub mod bootstrap;
pub mod lifecycle;
pub mod orchestrate;
pub mod preflight;
pub mod state;

pub use bootstrap::bootstrap_script;
pub use lifecycle::{
    point_config_at_forward, run_colab_down, run_colab_reconnect, run_colab_status, run_colab_up,
    UpOptions,
};
pub use orchestrate::{
    forward_argv, forward_url, pid_alive, proxy_argv, resolve_binaries, shell_quote, spawn_forward,
    terminate, Binaries,
};
pub use preflight::{preflight, which, Check, PreflightReport, KEY_FILENAME};
pub use state::{remove_state, save_state, ColabState, STATE_FILENAME};

#[cfg(test)]
pub(crate) mod testutil {
    use std::ffi::OsString;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard};

    /// Tests that touch `$PATH` / `$XCODE_CONFIG_DIR` must not run
    /// concurrently inside this test binary.
    pub static ENV_LOCK: Mutex<()> = Mutex::new(());

    pub struct EnvGuard {
        _lock: MutexGuard<'static, ()>,
        old_path: Option<OsString>,
        old_xcode: Option<OsString>,
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.old_path {
                Some(p) => std::env::set_var("PATH", p),
                None => std::env::remove_var("PATH"),
            }
            match &self.old_xcode {
                Some(x) => std::env::set_var("XCODE_CONFIG_DIR", x),
                None => std::env::remove_var("XCODE_CONFIG_DIR"),
            }
        }
    }

    /// Lock + swap `$PATH` and `$XCODE_CONFIG_DIR` to a hermetic fake world
    /// (PATH becomes *only* `bin_dir`, so real-host binaries cannot leak into
    /// "missing tool" scenarios). Restored on drop. Recovering from a poisoned
    /// lock keeps one panicking test from cascading failures into the rest.
    pub fn with_env(bin_dir: &Path, xcode_dir: &Path) -> EnvGuard {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old_path = std::env::var_os("PATH");
        let old_xcode = std::env::var_os("XCODE_CONFIG_DIR");
        std::env::set_var("PATH", bin_dir);
        std::env::set_var("XCODE_CONFIG_DIR", xcode_dir);
        EnvGuard {
            _lock: lock,
            old_path,
            old_xcode,
        }
    }

    pub fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("xencode-colab-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    pub fn write_script(dir: &Path, name: &str, body: &str) {
        let path = dir.join(name);
        fs::write(&path, body).expect("write fake bin");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod fake bin");
    }
}
