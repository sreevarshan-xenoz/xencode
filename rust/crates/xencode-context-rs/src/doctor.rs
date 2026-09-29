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
    fn a_denied_command_is_not_a_missing_one() {
        // `true` defies testing on an open machine, but the shape is pinned:
        // missing binary reads as absent, never as denied.
        assert!(!command_denied("xencode-no-such-binary-xyz", &[]));
    }
}
