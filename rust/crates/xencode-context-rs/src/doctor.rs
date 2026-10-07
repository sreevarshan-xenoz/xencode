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
    /// `config`, `config:version`, `layout`, `permissions:config`, `disk:state`,
    /// `size:cache`, `index`, `git`, `provider:ollama`, `model`,
    /// `mcp:server-name`, `metrics`, `cache`, `colab:<check>`.
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
    /// Raw unredacted transport/system error for automated issue filing,
    /// kept out of human prose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

impl SelfCheck {
    /// Attach the original unredacted error to this check result.
    pub fn with_raw(mut self, raw: impl Into<String>) -> Self {
        self.raw = Some(raw.into());
        self
    }

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

/// Whether this project's durable facts still agree with its code, as one row.
///
/// The pass that keeps a contradicted fact out of the prompt is silent by design:
/// `state.md` goes on showing the line while the model stops obeying it, which is
/// right for a turn and undiagnosable for the person who wrote it. This says what
/// was taken out and why, and rewrites nothing — a fact stale against this
/// checkout is frequently true again one branch over, and a doctor that cleaned up
/// the human's own file would have the tool decide what they are allowed to keep.
pub fn check_durable_facts(xencode_dir: &std::path::Path) -> SelfCheck {
    let name = "knowledge:stale".to_string();
    if !xencode_dir.join("state.md").is_file() {
        return SelfCheck {
            name,
            state: "absent".to_string(),
            detail: "no state.md — nothing has been promoted to durable memory".to_string(),
            fix: None,
            raw: None,
        };
    }
    let check = crate::compact::audit_durable_facts(xencode_dir);
    let kept = crate::state::ContextState::from_markdown(&check.text);
    let believed = kept.completed.len() + kept.decisions.len() + kept.unresolved.len();
    if check.dropped.is_empty() && check.unverifiable == 0 && check.disagreeing.is_empty() {
        return SelfCheck {
            name,
            state: "pass".to_string(),
            detail: format!(
                "{believed} durable fact{}, every one agreed with by the code",
                if believed == 1 { "" } else { "s" }
            ),
            fix: None,
            raw: None,
        };
    }
    let mut reasons: Vec<String> = check
        .dropped
        .iter()
        .take(3)
        .map(|fact| format!("{}: {}", short_fact(&fact.line), fact.problem.reason()))
        .collect();
    if check.dropped.len() > 3 {
        reasons.push(format!("{} more", check.dropped.len() - 3));
    }
    let mut detail = format!(
        "{believed} believed, {} dropped, {} could not be checked",
        check.dropped.len(),
        check.unverifiable
    );
    if !reasons.is_empty() {
        detail.push_str(&format!(" — {}", reasons.join("; ")));
    }
    // A row that says every stored fact agreed with the code while the prompt
    // carries a notice saying otherwise is worse than no row, so the two surfaces
    // report the same set. Nothing here drops a fact for it.
    if !check.disagreeing.is_empty() {
        let odds = check
            .disagreeing
            .iter()
            .take(2)
            .map(|odds| {
                format!(
                    "{} names {} but cites {}",
                    short_fact(&odds.line),
                    odds.name,
                    odds.cited
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        detail.push_str(&format!(
            ", {} place{} a name in a file that does not declare it ({})",
            check.disagreeing.len(),
            if check.disagreeing.len() == 1 {
                "s"
            } else {
                ""
            },
            odds
        ));
    }
    let fix = if check.dropped.is_empty() {
        "re-read the cited file and the files that do declare the name, then correct the \
         citation in state.md; the fact itself was not dropped for this"
            .to_string()
    } else {
        "re-read the file each dropped line cites and promote a corrected fact; \
         the lines stay in state.md until you say otherwise"
            .to_string()
    };
    SelfCheck {
        name,
        state: "fail".to_string(),
        detail,
        raw: None,
        fix: Some(fix),
    }
}

/// Check build and test recipe anchor freshness (`AB-1`).
///
/// An anchor file in `.xencode/anchor.md` provides build and test commands for the prompt.
/// The recipes were proved to work at the time they were run, but repositories change.
/// This check reads `.xencode/anchor.meta` to determine how long ago the anchor recipes
/// were verified, reporting when they have aged past [`crate::anchor::ANCHOR_STALE_AGE_DAYS`].
pub fn check_anchor(xencode_dir: &std::path::Path) -> SelfCheck {
    let name = "knowledge:anchor".to_string();
    let has_anchor = xencode_dir.join("anchor.md").is_file()
        || xencode_dir
            .join(crate::init::XENCODE_DIR)
            .join("anchor.md")
            .is_file();
    if !has_anchor {
        return SelfCheck {
            name,
            state: "absent".to_string(),
            detail: "no anchor.md — run `xencode anchor` to discover and prove build recipes"
                .to_string(),
            raw: None,
            fix: Some("run `xencode anchor` to generate .xencode/anchor.md".to_string()),
        };
    }

    let meta = crate::anchor::read_anchor_meta_from_dir(xencode_dir);
    match meta {
        Some(meta) => {
            let now_s = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let days = crate::anchor::anchor_age_days(meta.proved_at_unix_s, now_s);
            if days >= crate::anchor::ANCHOR_STALE_AGE_DAYS {
                SelfCheck {
                    name,
                    state: "fail".to_string(),
                    detail: format!(
                        "anchor proved {days} days ago; run `xencode anchor` to re-check"
                    ),
                    raw: None,
                    fix: Some(
                        "run `xencode anchor` to re-verify build and test recipes".to_string(),
                    ),
                }
            } else {
                SelfCheck {
                    name,
                    state: "pass".to_string(),
                    detail: format!(
                        "anchor proved {days} day{} ago ({} candidate{}, {} verified)",
                        if days == 1 { "" } else { "s" },
                        meta.candidates,
                        if meta.candidates == 1 { "" } else { "s" },
                        meta.verified
                    ),
                    fix: None,
                    raw: None,
                }
            }
        }
        None => SelfCheck {
            name,
            state: "fail".to_string(),
            detail: "anchor has no proof record; run `xencode anchor` to re-check".to_string(),
            raw: None,
            fix: Some(
                "run `xencode anchor` to prove build recipes and record provenance".to_string(),
            ),
        },
    }
}

/// A fact line as the row can show it: the sentence, cut short. A durable fact is a
/// sentence and a doctor row is not, and the markers are the tool's own file format
/// rather than anything about the project.
fn short_fact(line: &str) -> String {
    let one_line = crate::compact::fact_prose(line).trim();
    match one_line.char_indices().nth(60) {
        Some((at, _)) => format!("{}…", one_line[..at].trim_end()),
        None => one_line.to_string(),
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
    /// The settings file, loaded as the product would load it.
    Loaded,
    /// No file at all: every setting is a default.
    Absent,
    /// A file written by a newer xencode than this one. It is neither read nor
    /// overwritten, so every setting a person made is unread until they run the
    /// binary that can see it.
    TooNew { found: u32, known: u32 },
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
        ConfigRead::TooNew { .. } => (
            "fail",
            format!(
                "{} holds settings this xencode cannot read, so this run is on defaults (see the \
                 version row)",
                path.display()
            ),
            Some(
                "run the xencode that wrote this file, or point XCODE_CONFIG_DIR at a config this \
                 one can read"
                    .to_string(),
            ),
        ),
        ConfigRead::Unparseable(problem) => (
            "fail",
            format!("{} cannot be loaded: {problem}", path.display()),
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
        raw: None,
    }
}

/// What the version the config file declares means next to the one this binary
/// writes.
///
/// `declared` is what the file itself says, or `None` when there is no readable
/// file to ask. `Some(0)` is the answer for a file written before the key
/// existed. Below `current` the file will be brought up to this shape on read
/// and stamped the next time it is saved; above it, this binary refuses to read
/// or write it, which is the only thing standing between a newer config and an
/// older xencode dropping the fields it cannot see.
pub fn check_config_version(
    path: &std::path::Path,
    declared: Option<u32>,
    current: u32,
) -> SelfCheck {
    let name = "config:version".to_string();
    match declared {
        None => SelfCheck {
            name,
            state: "absent".to_string(),
            detail: format!("no configuration at {} to ask", path.display()),
            fix: None,
            raw: None,
        },
        Some(found) if found > current => SelfCheck {
            name,
            state: "fail".to_string(),
            detail: format!(
                "{path} declares version {found}; this xencode only knows up to {current}, so it \
                 neither reads nor overwrites the file",
                path = path.display()
            ),
            raw: None,
            fix: Some(
                "run the xencode that wrote this file, or point XCODE_CONFIG_DIR at a config this \
                 one can read"
                    .to_string(),
            ),
        },
        Some(found) if found == current => SelfCheck {
            name,
            state: "pass".to_string(),
            detail: format!("version {found} — the shape this xencode writes"),
            fix: None,
            raw: None,
        },
        Some(0) => SelfCheck {
            name,
            state: "pass".to_string(),
            detail: format!(
                "no version key — written before versions existed, so it is migrated on read and \
                 stamped {current} on the next save"
            ),
            fix: None,
            raw: None,
        },
        Some(found) => SelfCheck {
            name,
            state: "pass".to_string(),
            detail: format!(
                "version {found} — older than the {current} this xencode writes, so it is migrated \
                 on read"
            ),
            fix: None,
            raw: None,
        },
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
            raw: None,
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
        raw: None,
    }
}

/// Whether xencode is still reading its own files out of the single directory it
/// used before settings, records, cache and downloaded models were split into
/// four.
///
/// This is not a broken installation — everything works, and nothing has been
/// lost, which is exactly why it needs its own row rather than silence: a report
/// that says the configuration loads and the cache is a reasonable size would
/// otherwise leave the person with no reason to expect that their history is
/// still in `~/.xencode`, and that a machine-wide cleaner aimed at `~/.cache`
/// cannot see it. `still_old` names the kinds that are; `override_root` is
/// `$XCODE_CONFIG_DIR`, under which every kind is deliberately in one tree and
/// no migration is offered.
pub fn check_layout(still_old: &[String], override_root: Option<&std::path::Path>) -> SelfCheck {
    let name = "layout".to_string();
    if let Some(root) = override_root {
        return SelfCheck {
            name,
            state: "pass".to_string(),
            detail: format!(
                "every kind is read from {0}, as $XCODE_CONFIG_DIR asks",
                root.display()
            ),
            fix: None,
            raw: None,
        };
    }
    if still_old.is_empty() {
        return SelfCheck {
            name,
            state: "pass".to_string(),
            detail: "settings, records, cache and downloaded models are in their own directories"
                .to_string(),
            fix: None,
            raw: None,
        };
    }
    SelfCheck {
        name,
        state: "fail".to_string(),
        detail: format!(
            "{0} still read from ~/.xencode, where a cache cleaner cannot be aimed at one kind \
             without risking the others",
            still_old.join(", ")
        ),
        raw: None,
        fix: Some(
            "run `xencode paths` to see the directories, then `xencode migrate --dry-run`"
                .to_string(),
        ),
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
            raw: None,
        },
        Some(bytes) if bytes >= FLOOR => SelfCheck {
            name,
            state: "pass".to_string(),
            detail: format!("{} has {} free", dir.display(), format_bytes(bytes)),
            fix: None,
            raw: None,
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
            raw: None,
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
            raw: None,
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
            raw: None,
        },
        None => SelfCheck {
            name,
            state: "fail".to_string(),
            detail: format!("{} cannot be listed", dir.display()),
            raw: None,
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
        raw: None,
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
            raw: None,
            fix: Some("run /init in the TUI to build the project index".to_string()),
        },
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(_) => SelfCheck {
                name: "index".to_string(),
                state: "pass".to_string(),
                detail: path.display().to_string(),
                fix: None,
                raw: None,
            },
            Err(e) => SelfCheck {
                name: "index".to_string(),
                state: "fail".to_string(),
                detail: format!("{} does not parse: {e}", path.display()),
                raw: None,
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
            raw: None,
        },
        _ => SelfCheck {
            name: "git".to_string(),
            state: "fail".to_string(),
            detail: "not inside a git repository".to_string(),
            raw: None,
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
            raw: None,
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
        raw: None,
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
        raw: None,
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
        raw: None,
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
            raw: None,
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
                raw: None,
            }
        }
        Err(e) => SelfCheck {
            name: "cache".to_string(),
            state: "fail".to_string(),
            detail: format!("{} is not writable: {e}", dir.display()),
            raw: None,
            fix: Some(format!("chmod u+w {}", dir.display())),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layout that still reads `~/.xencode` is not broken, and the row has to
    /// say both things: that it works, and that the fix is named.
    #[test]
    fn a_layout_still_in_the_old_directory_names_the_command_that_moves_it() {
        let row = check_layout(&["settings".to_string(), "state".to_string()], None);
        assert_eq!(row.name, "layout");
        assert_eq!(row.state, "fail");
        assert!(row.detail.contains("settings, state"), "{}", row.detail);
        let fix = row.fix.expect("a layout row always carries the command");
        assert!(fix.contains("xencode paths"), "{fix}");
        assert!(fix.contains("xencode migrate"), "{fix}");

        let moved = check_layout(&[], None);
        assert!(moved.passed(), "{}", moved.detail);
        assert!(moved.fix.is_none(), "nothing is wrong to fix");

        // Under the override one tree is the point, so it is not a finding.
        let portable = check_layout(&[], Some(std::path::Path::new("/mnt/usb/xencode-config")));
        assert!(portable.passed());
        assert!(
            portable.detail.contains("/mnt/usb/xencode-config"),
            "{}",
            portable.detail
        );
    }

    /// QK-4: the row that answers "why did the model stop obeying the note I
    /// promoted" — the pass itself is silent, so the report is the only place a
    /// person can find out, and it has to name the line and the reason rather than
    /// a count.
    #[test]
    fn the_durable_row_names_the_facts_the_code_no_longer_agrees_with() {
        let unique = std::process::id();
        let dir = std::env::temp_dir().join(format!("xencode-doctor-durable-{unique}"));
        let xencode = dir.join(".xencode");
        std::fs::create_dir_all(&xencode).unwrap();

        let absent = check_durable_facts(&xencode);
        assert_eq!(absent.state, "absent", "{}", absent.detail);
        assert!(
            absent.detail.contains("no state.md"),
            "an absent row must say what is absent: {}",
            absent.detail
        );

        std::fs::write(
            xencode.join("state.md"),
            "# State\n\n## completed\n- the crate is Rust-first and cites no file\n",
        )
        .unwrap();
        let clean = check_durable_facts(&xencode);
        assert!(clean.passed(), "{}", clean.detail);
        assert!(
            clean.detail.contains("1 durable fact,"),
            "one fact must not read as several: {}",
            clean.detail
        );

        std::fs::write(
            xencode.join("state.md"),
            "# State\n\n## completed\n- login lives in src/auth.rs [src:src/auth.rs@11111111]\n",
        )
        .unwrap();
        let stale = check_durable_facts(&xencode);
        assert_eq!(stale.state, "fail", "{}", stale.detail);
        assert!(
            stale.detail.contains("login lives in src/auth.rs"),
            "the row dropped the line it is complaining about: {}",
            stale.detail
        );
        assert!(
            stale.detail.contains("the file it cites is gone"),
            "the row gives no reason: {}",
            stale.detail
        );
        assert!(
            stale.fix.is_some(),
            "a failing row has to say what to do about it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_disagreement_keeps_the_row_from_claiming_everything_agreed() {
        // The two surfaces must say the same thing. A prompt carrying a notice and a
        // doctor row reading "every fact agreed with the code" is worse than the row
        // not existing.
        let unique = std::process::id();
        let root = std::env::temp_dir().join(format!("xencode-doctor-odds-{unique}"));
        let xencode = root.join(".xencode");
        std::fs::create_dir_all(&xencode).unwrap();
        let run = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(&root)
                .output()
                .expect("git should be on PATH");
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            out
        };
        for args in [
            &["init", "-q"][..],
            &["config", "user.email", "test@xencode.local"][..],
            &["config", "user.name", "Xencode Test"][..],
        ] {
            run(args);
        }
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/auth.rs"), "pub fn validate_token() {}\n").unwrap();
        std::fs::write(root.join("src/session.rs"), "pub fn refresh_session() {}\n").unwrap();
        for args in [&["add", "-A"][..], &["commit", "-q", "-m", "seed"][..]] {
            run(args);
        }
        let head = String::from_utf8(run(&["rev-parse", "--short=8", "HEAD"]).stdout)
            .unwrap()
            .trim()
            .to_string();

        std::fs::write(
            xencode.join("state.md"),
            format!(
                "# State\n\n## decisions\n- validate_token rejects an empty token before src/session.rs runs [src:src/session.rs@{head}] [chk:validate_token]\n"
            ),
        )
        .unwrap();
        let row = check_durable_facts(&xencode);
        assert!(
            !row.passed(),
            "the row certified a fact the prompt says is misplaced: {}",
            row.detail
        );
        assert!(
            row.detail
                .contains("1 places a name in a file that does not declare it"),
            "the row did not say what kind of trouble this is: {}",
            row.detail
        );
        assert!(
            row.detail.contains("0 dropped"),
            "a disagreement was counted as a dropped fact: {}",
            row.detail
        );
        assert!(
            row.fix
                .as_deref()
                .is_some_and(|fix| fix.contains("was not dropped")),
            "the row told the reader to re-promote a fact nothing removed: {:?}",
            row.fix
        );

        // The same tier with the citation pointing at the file that really declares
        // the name: nothing left to say, and no row that reads as an alarm.
        std::fs::write(
            xencode.join("state.md"),
            format!(
                "# State\n\n## decisions\n- validate_token rejects an empty token [src:src/auth.rs@{head}] [chk:validate_token]\n"
            ),
        )
        .unwrap();
        let agrees = check_durable_facts(&xencode);
        assert!(
            agrees.passed(),
            "a correct citation was still reported: {}",
            agrees.detail
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_anchor_row_reports_age_and_freshness() {
        let unique = std::process::id();
        let dir = std::env::temp_dir().join(format!("xencode-doctor-anchor-{unique}"));
        let xencode = dir.join(".xencode");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&xencode).unwrap();

        // 1. Without anchor.md -> absent
        let absent = check_anchor(&xencode);
        assert_eq!(absent.state, "absent");
        assert!(absent.detail.contains("no anchor.md"));

        // 2. With anchor.md but without anchor.meta -> fail (no proof record)
        std::fs::write(
            xencode.join("anchor.md"),
            "# Anchor\n## Build\n- cargo build\n",
        )
        .unwrap();
        let unproven = check_anchor(&xencode);
        assert_eq!(unproven.state, "fail");
        assert!(unproven.detail.contains("no proof record"));

        // 3. With fresh anchor.meta (proved today) -> pass
        let now_s = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let fresh_meta = crate::anchor::AnchorMeta {
            proved_at_unix_s: now_s,
            candidates: 2,
            verified: 2,
        };
        crate::anchor::write_anchor_meta(&dir, &fresh_meta).unwrap();
        let fresh = check_anchor(&xencode);
        assert!(fresh.passed(), "{}", fresh.detail);
        assert!(fresh.detail.contains("0 days ago"));

        // 4. With aged anchor.meta (proved 20 days ago) -> fail with exact wording
        let aged_meta = crate::anchor::AnchorMeta {
            proved_at_unix_s: now_s - (20 * 86400),
            candidates: 2,
            verified: 2,
        };
        crate::anchor::write_anchor_meta(&dir, &aged_meta).unwrap();
        let aged = check_anchor(&xencode);
        assert_eq!(aged.state, "fail");
        assert_eq!(
            aged.detail,
            "anchor proved 20 days ago; run `xencode anchor` to re-check"
        );
        assert!(aged.fix.unwrap().contains("xencode anchor"));

        let _ = std::fs::remove_dir_all(&dir);
    }

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

        // A port nothing on this machine can answer on. The port this test just
        // released cannot be reused for the assertion below: another test in this
        // same binary binds ephemeral ports, and whichever one takes it first makes
        // the check honestly report a provider that is up. Measured failing that way
        // in three workspace runs while passing alone. Port 1 is privileged, so
        // nothing here can be listening.
        drop(listener);
        let down = check_provider(
            "ollama",
            "127.0.0.1",
            1,
            std::time::Duration::from_secs(2),
            Some("ollama serve"),
        );
        assert_eq!(down.state, "fail");
        assert!(down.detail.contains("ollama serve"), "{}", down.detail);

        // The same refused port with no local server to start says so without
        // inventing a command.
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
        let path = std::path::Path::new("/home/user/.config/xencode/config.json");
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

        let too_new = check_config(path, ConfigRead::TooNew { found: 9, known: 1 });
        assert_eq!(too_new.state, "fail");
        assert!(too_new.detail.contains("cannot read"), "{}", too_new.detail);
        assert!(too_new.detail.contains("defaults"), "{}", too_new.detail);
        assert!(
            too_new.fix.unwrap().contains("XCODE_CONFIG_DIR"),
            "the fix has to name the only knob that helps here"
        );
    }

    #[test]
    fn a_config_version_is_reported_as_migrating_current_or_refused() {
        let path = std::path::Path::new("/home/user/.config/xencode/config.json");

        let absent = check_config_version(path, None, 1);
        assert_eq!(absent.state, "absent");
        assert!(absent.detail.contains("to ask"), "{}", absent.detail);
        assert!(absent.fix.is_none());

        // A file from a newer xencode is the one case the report must fail:
        // nothing was read and nothing was written.
        let ahead = check_config_version(path, Some(4), 1);
        assert_eq!(ahead.state, "fail");
        assert!(ahead.detail.contains("4"), "{}", ahead.detail);
        assert!(
            ahead.fix.unwrap().contains("XCODE_CONFIG_DIR"),
            "the person needs the escape hatch, not just the number"
        );

        let current = check_config_version(path, Some(1), 1);
        assert_eq!(current.state, "pass");
        assert!(current.detail.contains("1"), "{}", current.detail);

        // Keyless, and one rung behind: both are work this binary does on its
        // own, so they pass and say what will happen.
        let legacy = check_config_version(path, Some(0), 1);
        assert_eq!(legacy.state, "pass");
        assert!(
            legacy.detail.contains("migrated on read"),
            "{}",
            legacy.detail
        );

        let behind = check_config_version(path, Some(1), 2);
        assert_eq!(behind.state, "pass");
        assert!(behind.detail.contains("older"), "{}", behind.detail);
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

    #[test]
    fn check_model_raw_field_serialization() {
        let check_without_raw = check_model(false, "detail", Some("fix".to_string()));
        let json1 = serde_json::to_value(&check_without_raw).unwrap();
        assert!(json1.get("raw").is_none());

        let check_with_raw = check_without_raw.with_raw("original transport error");
        let json2 = serde_json::to_value(&check_with_raw).unwrap();
        assert_eq!(json2["raw"], "original transport error");
    }
}
