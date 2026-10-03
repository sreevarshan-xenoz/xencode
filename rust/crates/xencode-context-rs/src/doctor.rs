//! Machine environment probe (`QO-5`), self-debug checks (`QO-7`) and the
//! aggregated bug report (`DB-6`).
//!
//! Probe and display, nothing more: core count, available memory, pressure
//! stall information, cgroup limits, GPUs, log readability, and whether the
//! colab route looks live. No "adaptive execution strategy" — adapting to a
//! machine on the basis of one probe is unverifiable, and this module does not
//! try.
//!
//! The `SelfCheck` half is the slice a person runs when xencode itself is the
//! thing that looks broken: does the config parse and keep its keys private, is
//! there room on the volume that holds its state, does the context index open,
//! is a git repo found, does each configured provider answer, does the default
//! model exist where the config says it lives, does each configured MCP server
//! start and get through its handshake, does `metrics.jsonl` parse, is the
//! cache directory writable. Every check reuses the code path the feature uses —
//! the same manifest path, the same `git rev-parse`, the same TCP connect, the
//! same write — and each one returns a sentence naming what to do about the
//! failure, plus the command or edit that would fix it.
//!
//! The rows are the bug report. `--format json` serialises this exact struct,
//! and the text listing is a rendering of the same list, so there is one shape
//! to attach to an issue rather than two that can disagree.
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
    /// `config`, `permissions:config`, `disk:state`, `size:cache`, `index`,
    /// `git`, `provider:ollama`, `model`, `mcp:server-name`, `metrics`, `cache`,
    /// `colab:<check>`.
    pub name: String,
    /// `pass`, `fail`, or `absent` (not applicable here, not broken).
    pub state: String,
    /// The named failure string, or what was found.
    pub detail: String,
    /// The command or edit that would fix it, when there is one. `None` means
    /// the row is evidence rather than an action — a size, or a finding whose
    /// remedy belongs to the provider rather than to this machine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

impl SelfCheck {
    /// `true` only for an explicit pass. Absent is not failed — a machine that
    /// never recorded metrics is not a broken machine.
    pub fn passed(&self) -> bool {
        self.state == "pass"
    }

    /// The row as it prints: the mark a reader scans for.
    pub fn mark(&self) -> &'static str {
        match self.state.as_str() {
            "pass" => "PASS",
            "fail" => "FAIL",
            _ => "ABSENT",
        }
    }
}

/// A byte count in the unit a person reads, so a report says `412 MiB` and not
/// `432012288`. Binary units: this is a filesystem, not a network adapter.
pub fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    if bytes < KIB {
        return format!("{bytes} B");
    }
    const UNITS: [(&str, u64); 4] = [
        ("KiB", KIB),
        ("MiB", KIB * KIB),
        ("GiB", KIB * KIB * KIB),
        ("TiB", KIB * KIB * KIB * KIB),
    ];
    let unit = UNITS
        .iter()
        .rev()
        .find(|(_, size)| bytes >= *size)
        .unwrap_or(&UNITS[0]);
    let value = bytes as f64 / unit.1 as f64;
    // A count that is already whole is printed as a whole number: `64 GiB free`
    // reads as a measurement, `64.0 GiB free` reads as a float.
    if value >= 100.0 || value.fract() == 0.0 {
        format!("{:.0} {}", value, unit.0)
    } else {
        format!("{:.1} {}", value, unit.0)
    }
}

/// The permission bits of a file, or `None` when it cannot be read (or the
/// platform has no such bits). `None` is never reported as a violation.
pub fn file_mode(path: &std::path::Path) -> Option<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path)
            .ok()
            .map(|m| m.permissions().mode() & 0o777)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// Total bytes and file count under `dir`, without following symlinks. `None`
/// means the directory could not be listed — which is a finding, not a size of
/// zero.
pub fn dir_usage(dir: &std::path::Path) -> Option<(u64, usize)> {
    let mut stack = vec![dir.to_path_buf()];
    let mut bytes = 0u64;
    let mut files = 0usize;
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            // A directory that vanished mid-walk says nothing about the rest.
            if current != dir {
                continue;
            }
            return None;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if file_type.is_symlink() {
                continue;
            }
            if let Ok(meta) = std::fs::metadata(&path) {
                bytes += meta.len();
                files += 1;
            }
        }
    }
    Some((bytes, files))
}

/// Whether the config file was found and what it did when parsed. The CLI reads
/// it with the product's own loader; this is the wording of that outcome.
#[derive(Debug, Clone)]
pub enum ConfigRead {
    /// `~/.xencode/config.json` loaded as the product would load it.
    Loaded,
    /// No file at all: every setting is a default.
    Absent,
    /// A file that the loader refused, with the loader's sentence.
    Unparseable(String),
}

/// Does the configuration load. A file that does not parse means every setting
/// a person made is unread, so this is the first row of a bug report.
pub fn check_config(path: &std::path::Path, read: ConfigRead) -> SelfCheck {
    let (state, detail, fix) = match read {
        ConfigRead::Loaded => ("pass", path.display().to_string(), None),
        ConfigRead::Absent => (
            "absent",
            format!("no {} — running on defaults", path.display()),
            Some("run `xencode config set <key> <value>` to create one".to_string()),
        ),
        ConfigRead::Unparseable(problem) => (
            "fail",
            format!("{} does not parse: {problem}", path.display()),
            Some(format!(
                "repair the JSON at {}, or move the file aside to start from defaults",
                path.display()
            )),
        ),
    };
    SelfCheck {
        name: "config".to_string(),
        state: state.to_string(),
        detail,
        fix,
    }
}

/// Is a file that holds secrets readable only by its owner. The check is the
/// product's own promise: `write_atomic` creates every state file at `0600`, so
/// a wider mode means something else wrote it, or an old version did.
pub fn check_permissions(label: &str, path: &std::path::Path, mode: Option<u32>) -> SelfCheck {
    let name = format!("permissions:{label}");
    let Some(mode) = mode else {
        return SelfCheck {
            name,
            state: "absent".to_string(),
            detail: format!("{} is not there to check", path.display()),
            fix: None,
        };
    };
    let private = mode & 0o077 == 0;
    SelfCheck {
        name,
        state: if private { "pass" } else { "fail" }.to_string(),
        detail: if private {
            format!("{:o} — owner only", mode)
        } else {
            format!(
                "{:o} — readable beyond the owner, and it holds secrets",
                mode
            )
        },
        fix: (!private).then(|| format!("chmod 600 {}", path.display())),
    }
}

/// Room left on the volume that holds the state directory. `free` comes from
/// the same `statvfs` call the model download uses; `None` means the filesystem
/// refused to answer, which is reported rather than guessed at.
pub fn check_free_disk(label: &str, dir: &std::path::Path, free: Option<u64>) -> SelfCheck {
    const FLOOR: u64 = 256 * 1024 * 1024;
    let name = format!("disk:{label}");
    match free {
        None => SelfCheck {
            name,
            state: "absent".to_string(),
            detail: format!("{} would not report its free space", dir.display()),
            fix: None,
        },
        Some(bytes) if bytes >= FLOOR => SelfCheck {
            name,
            state: "pass".to_string(),
            detail: format!("{} has {} free", dir.display(), format_bytes(bytes)),
            fix: None,
        },
        Some(bytes) => SelfCheck {
            name,
            state: "fail".to_string(),
            detail: format!(
                "{} has {} free, under the {} this needs to write state",
                dir.display(),
                format_bytes(bytes),
                format_bytes(FLOOR)
            ),
            fix: Some(format!(
                "free space on the volume holding {}, or set XCODE_CONFIG_DIR to a larger one",
                dir.display()
            )),
        },
    }
}

/// How much disk the state a feature accumulates is taking. A size is evidence,
/// not a verdict, so this passes whenever it could be measured and fails only
/// when the directory exists but could not be read.
pub fn check_dir_size(label: &str, dir: &std::path::Path) -> SelfCheck {
    let name = format!("size:{label}");
    if !dir.exists() {
        return SelfCheck {
            name,
            state: "absent".to_string(),
            detail: format!("{} does not exist", dir.display()),
            fix: None,
        };
    }
    match dir_usage(dir) {
        Some((bytes, files)) => SelfCheck {
            name,
            state: "pass".to_string(),
            detail: format!(
                "{} holds {} file(s), {}",
                dir.display(),
                files,
                format_bytes(bytes)
            ),
            fix: None,
        },
        None => SelfCheck {
            name,
            state: "fail".to_string(),
            detail: format!("{} cannot be listed", dir.display()),
            fix: Some(format!(
                "check the permissions on {}, or remove it and let xencode create it again",
                dir.display()
            )),
        },
    }
}

/// The model a turn would actually use, asked of the server the config points
/// at. The caller asks with the client the product generates text with, so a
/// pass means that server knows the model by name.
pub fn check_model(known: bool, detail: impl Into<String>, fix: Option<String>) -> SelfCheck {
    SelfCheck {
        name: "model".to_string(),
        state: if known { "pass" } else { "fail" }.to_string(),
        detail: detail.into(),
        fix: if known { None } else { fix },
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
            fix: Some("run /init in the TUI to build the project index".to_string()),
        },
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(_) => SelfCheck {
                name: "index".to_string(),
                state: "pass".to_string(),
                detail: path.display().to_string(),
                fix: None,
            },
            Err(e) => SelfCheck {
                name: "index".to_string(),
                state: "fail".to_string(),
                detail: format!("{} does not parse: {e}", path.display()),
                fix: Some(format!(
                    "run /init again, or delete {} and rebuild it",
                    path.display()
                )),
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
            fix: None,
        },
        _ => SelfCheck {
            name: "git".to_string(),
            state: "fail".to_string(),
            detail: "not inside a git repository".to_string(),
            fix: Some(format!(
                "run `git init` in {}, or start xencode inside a repository",
                dir.display()
            )),
        },
    }
}

/// Do the recorded metrics parse, and how large is the file. A missing file is
/// absent, not failed.
pub fn check_metrics(xencode_dir: &std::path::Path) -> SelfCheck {
    let path = crate::metrics_path(xencode_dir);
    if !path.is_file() {
        return SelfCheck {
            name: "metrics".to_string(),
            state: "absent".to_string(),
            detail: "no metrics recorded yet".to_string(),
            fix: None,
        };
    }
    let rows = crate::read_metrics(xencode_dir);
    let size = std::fs::metadata(&path).ok().map(|m| format_bytes(m.len()));
    SelfCheck {
        name: "metrics".to_string(),
        state: "pass".to_string(),
        detail: format!(
            "{} row(s) parse in {}",
            rows.len(),
            size.as_deref()
                .unwrap_or("a file that would not say its size")
        ),
        fix: None,
    }
}

/// A provider endpoint, dialled at the address the config points it at.
/// `start_command` is what to run when the endpoint is a server that belongs on
/// this machine; a cloud address has none, and its failure says only that the
/// address did not answer, because there is nothing here to start.
pub fn check_provider(
    name: &str,
    host: &str,
    port: u16,
    timeout: std::time::Duration,
    start_command: Option<&str>,
) -> SelfCheck {
    let reachable = tcp_reachable(host, port, timeout);
    let (state, detail, fix) = if reachable {
        ("pass", format!("{host}:{port} accepts TCP"), None)
    } else {
        match start_command {
            Some(command) => (
                "fail",
                format!(
                    "{host}:{port} refused: nothing is listening; start it with `{command}` or \
                     point the config elsewhere"
                ),
                Some(command.to_string()),
            ),
            None => (
                "fail",
                format!("{host}:{port} refused: the {name} endpoint does not answer from here"),
                Some(
                    "check the network; the address dialed is the one the config names".to_string(),
                ),
            ),
        }
    };
    SelfCheck {
        name: format!("provider:{name}"),
        state: state.to_string(),
        detail,
        fix,
    }
}

/// One configured MCP server, in the words the attempt to reach it came back
/// with. The caller runs the real client, so a pass means the handshake was
/// answered and a failure carries the client's own sentence for why it was not.
pub fn check_mcp(
    name: &str,
    reached: bool,
    detail: impl Into<String>,
    fix: Option<String>,
) -> SelfCheck {
    SelfCheck {
        name: format!("mcp:{name}"),
        state: if reached { "pass" } else { "fail" }.to_string(),
        detail: detail.into(),
        fix: if reached { None } else { fix },
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
            fix: Some(format!(
                "make {} writable, or remove it and let xencode create it again",
                dir.display()
            )),
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
                fix: None,
            }
        }
        Err(e) => SelfCheck {
            name: "cache".to_string(),
            state: "fail".to_string(),
            detail: format!("{} is not writable: {e}", dir.display()),
            fix: Some(format!("chmod u+w {}", dir.display())),
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
    fn a_provider_failure_names_the_thing_to_run() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let up = check_provider(
            "ollama",
            "127.0.0.1",
            port,
            std::time::Duration::from_secs(2),
            Some("ollama serve"),
        );
        assert_eq!(up.name, "provider:ollama");
        assert!(up.passed(), "{}: {}", up.state, up.detail);
        drop(listener);

        // The same port with nothing behind it: the check says what to do, not
        // just that something refused.
        let down = check_provider(
            "ollama",
            "127.0.0.1",
            port,
            std::time::Duration::from_secs(2),
            Some("ollama serve"),
        );
        assert_eq!(down.state, "fail");
        assert!(down.detail.contains("ollama serve"), "{}", down.detail);

        // An address with no local server to start says so without inventing a
        // command. Port 1 is privileged, so nothing here can be listening.
        let cloud = check_provider(
            "cloud:openai",
            "127.0.0.1",
            1,
            std::time::Duration::from_secs(2),
            None,
        );
        assert_eq!(cloud.state, "fail");
        assert!(!cloud.detail.contains('`'), "{}", cloud.detail);
        assert!(cloud.detail.contains("does not answer"), "{}", cloud.detail);
    }

    #[test]
    fn an_mcp_server_is_named_by_its_attempt() {
        let up = check_mcp(
            "docs",
            true,
            "starts and answers the handshake",
            Some("unused".to_string()),
        );
        assert_eq!(up.name, "mcp:docs");
        assert!(up.passed());
        // A pass has nothing to fix, so the fix cannot leak into the row.
        assert_eq!(up.fix, None);
        let down = check_mcp(
            "docs",
            false,
            "cannot start MCP server `docs`: not executable",
            Some("install or rebuild the server, then remove it from config".to_string()),
        );
        assert_eq!(down.state, "fail");
        assert!(down.detail.contains("not executable"), "{}", down.detail);
        assert!(down.fix.unwrap().contains("remove it from config"));
    }

    #[test]
    fn a_byte_count_is_stated_in_the_unit_a_person_reads() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(999), "999 B");
        assert_eq!(format_bytes(1024), "1 KiB");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(256 * 1024 * 1024), "256 MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3 GiB");
        assert_eq!(format_bytes(64 * 1024 * 1024 * 1024), "64 GiB");
        // A part of a unit is kept to one decimal, not a wall of digits.
        assert_eq!(
            format_bytes(1024 * 1024 * 1024 + 512 * 1024 * 1024),
            "1.5 GiB"
        );
    }

    #[test]
    fn a_config_that_does_not_parse_says_so_and_says_what_to_do() {
        let path = std::path::Path::new("/home/sree/.xencode/config.json");
        let loaded = check_config(path, ConfigRead::Loaded);
        assert_eq!(loaded.state, "pass");
        assert!(loaded.fix.is_none());

        let missing = check_config(path, ConfigRead::Absent);
        assert_eq!(missing.state, "absent");
        assert!(missing.detail.contains("defaults"), "{}", missing.detail);

        let broken = check_config(
            path,
            ConfigRead::Unparseable("trailing comma at line 4".to_string()),
        );
        assert_eq!(broken.state, "fail");
        assert!(
            broken.detail.contains("trailing comma"),
            "{}",
            broken.detail
        );
        assert!(broken.fix.unwrap().contains("config.json"));
    }

    #[cfg(unix)]
    #[test]
    fn a_secret_file_is_checked_against_its_own_group() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("xe-perm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, b"{}").unwrap();

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let private = check_permissions("config", &path, file_mode(&path));
        assert!(private.passed(), "{}: {}", private.state, private.detail);
        assert!(private.detail.contains("600"), "{}", private.detail);

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let open = check_permissions("config", &path, file_mode(&path));
        assert_eq!(open.state, "fail");
        assert!(
            open.fix.as_deref().unwrap().starts_with("chmod 600"),
            "{:?}",
            open.fix
        );

        // A file that is not there is not a violation.
        let gone = check_permissions("config", &dir.join("nope.json"), None);
        assert_eq!(gone.state, "absent");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn free_space_is_measured_and_the_floor_is_named() {
        let dir = std::path::Path::new("/");
        // Whatever this volume reports, the row states a number rather than a
        // feeling, and only a failure carries an instruction.
        let free = crate::hwprobe::free_disk_bytes("/");
        let row = check_free_disk("state", dir, free);
        assert!(
            row.detail.chars().any(|c| c.is_ascii_digit()),
            "{}",
            row.detail
        );
        assert_eq!(row.fix.is_some(), row.state == "fail");
        assert_eq!(row.mark(), if row.passed() { "PASS" } else { "FAIL" });

        let tight = check_free_disk("state", dir, Some(1024 * 1024));
        assert_eq!(tight.state, "fail");
        assert!(
            tight.detail.contains("256 MiB"),
            "the floor is stated: {}",
            tight.detail
        );
        assert!(tight.fix.unwrap().contains("XCODE_CONFIG_DIR"));

        let mute = check_free_disk("state", dir, None);
        assert_eq!(mute.state, "absent");
        assert_eq!(mute.mark(), "ABSENT");
    }

    #[test]
    fn a_directory_size_is_evidence_not_a_verdict() {
        let dir = std::env::temp_dir().join(format!("xe-size-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let absent = check_dir_size("cache", &dir);
        assert_eq!(absent.state, "absent");

        std::fs::create_dir_all(dir.join("nested")).unwrap();
        std::fs::write(dir.join("a.json"), b"0123456789").unwrap();
        std::fs::write(dir.join("nested").join("b.json"), b"01234").unwrap();
        let present = check_dir_size("cache", &dir);
        assert!(present.passed(), "{}: {}", present.state, present.detail);
        // Both files count, in the nested directory as well as the top one.
        assert!(present.detail.contains("2 file(s)"), "{}", present.detail);
        assert!(present.detail.contains("15 B"), "{}", present.detail);
        assert_eq!(dir_usage(&dir), Some((15, 2)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_model_the_server_does_not_know_names_the_command_to_fix_it() {
        let known = check_model(
            true,
            "qwen3:4b is in the server's list",
            Some("ollama pull qwen3:4b".to_string()),
        );
        assert_eq!(known.name, "model");
        assert!(known.passed());
        assert_eq!(known.fix, None, "a pass has nothing to fix");

        let missing = check_model(
            false,
            "the server answers but has no qwen3:4b",
            Some("ollama pull qwen3:4b".to_string()),
        );
        assert_eq!(missing.state, "fail");
        assert_eq!(missing.fix.as_deref(), Some("ollama pull qwen3:4b"));
    }

    #[test]
    fn a_denied_command_is_not_a_missing_one() {
        // `true` defies testing on an open machine, but the shape is pinned:
        // missing binary reads as absent, never as denied.
        assert!(!command_denied("xencode-no-such-binary-xyz", &[]));
    }
}
