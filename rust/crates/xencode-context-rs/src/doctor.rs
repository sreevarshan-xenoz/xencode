//! Machine environment probe (`QO-5`).
//!
//! Probe and display, nothing more: core count, available memory, pressure
//! stall information, cgroup limits, GPUs, log readability, and whether the
//! colab route looks live. No "adaptive execution strategy" — adapting to a
//! machine on the basis of one probe is unverifiable, and this module does not
//! try.
//!
//! Every fact is best-effort. An unreadable file, a missing binary, or a
//! denied syscall yields `None` (or `false` for readability checks), never an
//! error: a doctor that cannot run on a locked-down machine is a doctor that
//! fails exactly where it is needed most.

use serde::{Deserialize, Serialize};

/// What the machine looks like, as far as can be told without privileges.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvFacts {
    /// Logical core count, when countable.
    pub nproc: Option<usize>,
    /// Available memory in KiB, when readable.
    pub mem_available_kib: Option<u64>,
    /// Whether `/proc/pressure` Stall information is readable.
    pub psi_readable: bool,
    /// The cgroup memory limit in effect, when one is set.
    pub cgroup_memory_limit: Option<String>,
    /// `nvidia-smi -L` lines, when the tool runs.
    pub nvidia_gpus: Vec<String>,
    /// Whether `journalctl --user` produces output.
    pub journalctl_readable: bool,
    /// Whether `dmesg` refuses (EPERM on locked-down machines).
    pub dmesg_denied: bool,
}

/// Read the machine. Never fails; unknown facts stay empty.
pub fn probe_env() -> EnvFacts {
    EnvFacts {
        nproc: crate::hwprobe::cpu_core_count(),
        mem_available_kib: crate::hwprobe::available_memory_kib(),
        psi_readable: std::fs::read_dir("/proc/pressure").is_ok(),
        cgroup_memory_limit: cgroup_limit(),
        nvidia_gpus: nvidia_list(),
        journalctl_readable: command_readable("journalctl", &["--user", "--no-pager", "-n", "1"]),
        dmesg_denied: command_denied("dmesg", &["--ctime"]),
    }
}

/// The cgroup memory ceiling, v1 or v2, or `max` when unlimited.
fn cgroup_limit() -> Option<String> {
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ] {
        if let Ok(text) = std::fs::read_to_string(path) {
            let value = text.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

/// One line per GPU from `nvidia-smi -L`. Absent tool, absent GPUs, or any
/// failure all mean the same thing here: no list.
fn nvidia_list() -> Vec<String> {
    let output = std::process::Command::new("nvidia-smi").arg("-L").output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("GPU "))
        .map(str::to_string)
        .collect()
}

/// Whether a command produces any output at all.
fn command_readable(program: &str, args: &[&str]) -> bool {
    std::process::Command::new(program)
        .args(args)
        .output()
        .is_ok_and(|o| o.status.success() && !o.stdout.is_empty())
}

/// Whether running a command is refused. Used for `dmesg`, which answers EPERM
/// where unprivileged reads are locked down.
fn command_denied(program: &str, args: &[&str]) -> bool {
    match std::process::Command::new(program).args(args).output() {
        Ok(output) => !output.status.success(),
        Err(_) => false,
    }
}

/// Parse `nvidia-smi -L` output without running anything.
pub fn parse_nvidia_list(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| l.starts_with("GPU "))
        .map(str::to_string)
        .collect()
}

/// One self-debug check and what it found.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfCheck {
    /// `index`, `git`, `provider:ollama`, `mcp:server-name`, `metrics`, `cache`.
    pub name: String,
    /// `pass`, `fail`, or `absent` (not applicable here, not broken).
    pub state: String,
    /// The named failure string, or what was found.
    pub detail: String,
}

impl SelfCheck {
    /// `true` only for an explicit pass. Absent is not failed — a machine that
    /// never recorded metrics is not a broken machine.
    pub fn passed(&self) -> bool {
        self.state == "pass"
    }
}

/// Whether TCP connects within the budget. A real connection attempt: refused
/// and timeout both read as unreachable, and the caller says which address.
pub fn tcp_reachable(host: &str, port: u16, timeout: std::time::Duration) -> bool {
    use std::net::ToSocketAddrs;
    let addrs: Vec<_> = match (host, port).to_socket_addrs() {
        Ok(addrs) => addrs.collect(),
        Err(_) => return false,
    };
    addrs
        .into_iter()
        .any(|addr| std::net::TcpStream::connect_timeout(&addr, timeout).is_ok())
}

/// Whether a command resolves on PATH, like a shell spawn would find it.
pub fn resolve_on_path(command: &str) -> Option<std::path::PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(command);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            for ext in ["exe", "bat", "cmd"] {
                let candidate = dir.join(format!("{command}.{ext}"));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// Does the context index open: manifest present and parseable as JSON.
pub fn check_index(xencode_dir: &std::path::Path) -> SelfCheck {
    let path = crate::index::manifest_path(xencode_dir);
    match std::fs::read_to_string(&path) {
        Err(_) => SelfCheck {
            name: "index".to_string(),
            state: "absent".to_string(),
            detail: "no index manifest; run /init for project-aware answers".to_string(),
        },
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(_) => SelfCheck {
                name: "index".to_string(),
                state: "pass".to_string(),
                detail: path.display().to_string(),
            },
            Err(e) => SelfCheck {
                name: "index".to_string(),
                state: "fail".to_string(),
                detail: format!("{} does not parse: {e}", path.display()),
            },
        },
    }
}

/// Is this directory inside a git repository.
pub fn check_git(dir: &std::path::Path) -> SelfCheck {
    match std::process::Command::new("git")
        .current_dir(dir)
        .args(["rev-parse", "--show-toplevel"])
        .output()
    {
        Ok(output) if output.status.success() => SelfCheck {
            name: "git".to_string(),
            state: "pass".to_string(),
            detail: String::from_utf8_lossy(&output.stdout).trim().to_string(),
        },
        _ => SelfCheck {
            name: "git".to_string(),
            state: "fail".to_string(),
            detail: "not inside a git repository".to_string(),
        },
    }
}

/// Do the recorded metrics parse. A missing file is absent, not failed.
pub fn check_metrics(xencode_dir: &std::path::Path) -> SelfCheck {
    let path = crate::metrics_path(xencode_dir);
    if !path.is_file() {
        return SelfCheck {
            name: "metrics".to_string(),
            state: "absent".to_string(),
            detail: "no metrics recorded yet".to_string(),
        };
    }
    let rows = crate::read_metrics(xencode_dir);
    SelfCheck {
        name: "metrics".to_string(),
        state: "pass".to_string(),
        detail: format!("{} row(s) parse", rows.len()),
    }
}

/// Is the cache directory writable, proved by writing.
pub fn check_cache_writable(xencode_dir: &std::path::Path) -> SelfCheck {
    let dir = xencode_dir.join("cache");
    if std::fs::create_dir_all(&dir).is_err() {
        return SelfCheck {
            name: "cache".to_string(),
            state: "fail".to_string(),
            detail: format!("cannot create {}", dir.display()),
        };
    }
    let probe = dir.join(".doctor-write-probe");
    match std::fs::write(&probe, b"ok") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            SelfCheck {
                name: "cache".to_string(),
                state: "pass".to_string(),
                detail: dir.display().to_string(),
            }
        }
        Err(e) => SelfCheck {
            name: "cache".to_string(),
            state: "fail".to_string(),
            detail: format!("{} is not writable: {e}", dir.display()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_gpu_lines_count() {
        let text = "GPU 0: Tesla T4 (UUID: GPU-123)\nFailed to init\nGPU 1: L4 (UUID: GPU-456)\n";
        assert_eq!(
            parse_nvidia_list(text),
            vec![
                "GPU 0: Tesla T4 (UUID: GPU-123)".to_string(),
                "GPU 1: L4 (UUID: GPU-456)".to_string()
            ]
        );
        assert!(parse_nvidia_list("no gpus here\n").is_empty());
    }

    #[test]
    fn probing_never_fails_even_where_everything_is_missing() {
        // Whatever this machine lacks, the probe reports rather than errors.
        let facts = probe_env();
        let text = serde_json::to_string(&facts).unwrap();
        assert!(text.contains("nproc"), "{text}");
        // nproc is countable on any machine that runs tests at all.
        assert!(facts.nproc.unwrap_or(0) >= 1);
    }

    #[test]
    fn a_closed_port_refuses_and_a_listener_accepts() {
        // Both halves are real syscalls: nothing is faked, and both outcomes
        // are deterministic — a listener on localhost, and port 1, which no
        // user process may bind.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(tcp_reachable(
            "127.0.0.1",
            port,
            std::time::Duration::from_secs(2)
        ));
        drop(listener);
        assert!(!tcp_reachable(
            "127.0.0.1",
            1,
            std::time::Duration::from_secs(2)
        ));
        assert!(!tcp_reachable(
            "no-such-host.invalid",
            443,
            std::time::Duration::from_secs(2)
        ));
    }

    #[test]
    fn path_resolution_finds_what_a_shell_would() {
        // `sh` is on PATH wherever tests run; a missing binary resolves to nothing.
        assert!(resolve_on_path("sh").is_some());
        assert!(resolve_on_path("xencode-no-such-binary-xyz").is_none());
    }

    #[test]
    fn absent_index_metrics_and_cache_states_are_honest() {
        let dir = std::env::temp_dir().join(format!("xe-doc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let xencode = dir.join(".xencode");
        assert_eq!(check_index(&xencode).state, "absent");
        assert_eq!(check_metrics(&xencode).state, "absent");
        assert!(check_cache_writable(&xencode).passed());
        assert_eq!(check_git(&dir).state, "fail");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_denied_command_is_not_a_missing_one() {
        // `true` defies testing on an open machine, but the shape is pinned:
        // missing binary reads as absent, never as denied.
        assert!(!command_denied("xencode-no-such-binary-xyz", &[]));
    }
}
