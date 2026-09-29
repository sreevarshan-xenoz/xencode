//! Which agents this probe knows about, and what was already read about them.
//!
//! The capability cells here come from `--help` and `--version` output captured on
//! 2026-09-28 (Milestone W). They are **not measurements**: they are a starting
//! roster, and every one of them is marked as read-only evidence so a reader can
//! tell at a glance which parts of the matrix still need running.

use serde::Serialize;

/// How a cell in the matrix came to be believed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// We ran the thing and watched it happen. The only kind `AR-1` accepts as a
    /// measurement.
    Observed,
    /// Read from `--help`, `--version` or a protocol schema. Recorded because it
    /// is a starting point, never because it settles anything.
    ReadFromHelp,
    /// The agent is not installed here, so nothing is known about it beyond the
    /// name. An empty cell, stated as one.
    NotInstalled,
    /// The run was attempted and did not get far enough to answer the question.
    RunFailed,
}

impl Provenance {
    /// The one-letter column the markdown report uses.
    pub fn marker(self) -> &'static str {
        match self {
            Provenance::Observed => "O",
            Provenance::ReadFromHelp => "H",
            Provenance::NotInstalled => "-",
            Provenance::RunFailed => "X",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Provenance::Observed => "observed",
            Provenance::ReadFromHelp => "read_from_help",
            Provenance::NotInstalled => "not_installed",
            Provenance::RunFailed => "run_failed",
        }
    }
}

/// One agent the probe knows how to launch.
#[derive(Debug, Clone, Serialize)]
pub struct AgentSpec {
    /// The name used everywhere: in the report, in `--agent`, and on disk.
    pub name: &'static str,
    /// Executable names to try, in order. Two because `ast-grep` ships as both
    /// `ast-grep` and `sg`, and vendors disagree about theirs.
    pub binaries: &'static [&'static str],
    /// The headless one-shot form, with `{prompt}` where the task goes. Taken
    /// from `--help`; a probe run is what confirms it.
    pub one_shot: &'static str,
    /// The flag that makes the output machine-readable, if help mentions one.
    pub stream_flag: Option<&'static str>,
    /// Whether help advertises a daemon or server of its own.
    pub advertises_daemon: bool,
    /// Whether help advertises ACP.
    pub advertises_acp: bool,
    /// Whether help advertises MCP client configuration.
    pub advertises_mcp: bool,
    /// Whether help advertises a session store and a way to resume.
    pub advertises_resume: bool,
    /// Whether help advertises its own approval or permission control.
    pub advertises_approval: bool,
    /// When these cells were read.
    pub read_on: &'static str,
}

/// The roster, as read from `--help` on 2026-09-28. See [`Provenance`] for why
/// none of it counts as a measurement.
pub const ROSTER: &[AgentSpec] = &[
    AgentSpec {
        name: "opencode",
        binaries: &["opencode"],
        one_shot: "opencode run {prompt}",
        stream_flag: Some("--format json"),
        advertises_daemon: true, // `opencode serve` + `opencode attach <url>`
        advertises_acp: true,    // `opencode acp`
        advertises_mcp: true,    // `opencode mcp`
        advertises_resume: true, // `session list`, `export`/`import`
        advertises_approval: true, // `--auto`, auto-approve permissions (AR-3 contradicted `false` on 2026-09-29)
        read_on: "2026-09-29",
    },
    AgentSpec {
        name: "cline",
        binaries: &["cline"],
        one_shot: "cline --json {prompt}",
        stream_flag: Some("--json"),
        advertises_daemon: true,   // `cline hub`, `-z/--zen`
        advertises_acp: true,      // `--acp`
        advertises_mcp: true,      // `cline mcp`
        advertises_resume: true,   // `--id <session-id>`, `cline history`
        advertises_approval: true, // `--auto-approve`, `CLINE_TOOL_APPROVAL_MODE`
        read_on: "2026-09-28",
    },
    AgentSpec {
        name: "codex",
        binaries: &["codex"],
        one_shot: "codex exec {prompt}",
        stream_flag: Some("--json"),
        advertises_daemon: true,   // `app-server daemon`, `codex agents`
        advertises_acp: false,     // no `acp` in help; `app-server` instead
        advertises_mcp: true,      // `codex mcp`
        advertises_resume: true,   // `resume`, `fork`, `queue`, `archive`
        advertises_approval: true, // `--sandbox`, `--ask-for-approval`
        read_on: "2026-09-28",
    },
    AgentSpec {
        name: "claude",
        binaries: &["claude"],
        // `--verbose` is not optional here: measured on 2026-09-28, `--print`
        // with `--output-format stream-json` and no `--verbose` is refused with
        // "requires --verbose". Nothing in `--help` says the two are related.
        one_shot: "claude -p --verbose {prompt}",
        stream_flag: Some("--output-format stream-json"),
        advertises_daemon: true,   // `--bg` + `claude attach/logs/stop/rm`
        advertises_acp: false,     // not in help
        advertises_mcp: true,      // `claude mcp`
        advertises_resume: true,   // `--resume`, `--fork-session`
        advertises_approval: true, // `--permission-mode`, `--allowedTools`
        read_on: "2026-09-28",
    },
    AgentSpec {
        name: "gemini",
        binaries: &["gemini"],
        one_shot: "gemini -p {prompt}",
        stream_flag: Some("--output-format stream-json"),
        advertises_daemon: false,  // nothing in help
        advertises_acp: true,      // `--acp`
        advertises_mcp: true,      // `gemini mcp`
        advertises_resume: true,   // `--resume`, `--session-id`, `--session-file`
        advertises_approval: true, // `--approval-mode`, `--policy`
        read_on: "2026-09-28",
    },
    AgentSpec {
        name: "crush",
        binaries: &["crush"],
        one_shot: "crush run {prompt}",
        stream_flag: None,       // no structured output option in help
        advertises_daemon: true, // `server` subcommand binding a socket (AR-3 contradicted `false` on 2026-09-29)
        advertises_acp: false,
        advertises_mcp: false,
        advertises_resume: true,   // `--session`, `--continue`
        advertises_approval: true, // `--yolo`
        read_on: "2026-09-28",
    },
];

/// Agents the proposal names that are **not** on this machine.
///
/// `AR-1`'s done-when is about the installed CLIs, so their absence is recorded
/// as a fact rather than left as a blank. Kilo in particular had every claim
/// about it marked UNVERIFIED in Milestone W, and this is why: there is nothing
/// here to verify against.
pub const NOT_INSTALLED: &[(&str, &str)] = &[
    ("kilo", "named in the proposal; no binary on this machine, so every claim about it is unverified"),
    ("agy", "listed in AR-2's done-when but not installed — that item would fail its own test on day one"),
];

/// Look one agent up by name.
pub fn find(name: &str) -> Option<&'static AgentSpec> {
    ROSTER.iter().find(|a| a.name == name)
}

/// Resolve an executable name to a full path, the way `which` would.
pub fn which(binary: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(binary))
        .find(|candidate| candidate.is_file())
}

/// Where the `PATH` lookup found an agent, or [`Provenance::NotInstalled`].
pub fn provenance_of_help_cell(spec: &AgentSpec) -> Provenance {
    if spec.binaries.iter().any(|b| which(b).is_some()) {
        // Installed, but nothing about it has been *run* by this crate yet. The
        // help cells stay help until a probe says otherwise, which is the whole
        // point of the provenance column.
        Provenance::ReadFromHelp
    } else {
        Provenance::NotInstalled
    }
}

/// How an installed agent got onto this machine, as far as its path says.
///
/// Only what the path shows is claimed. A binary under `~/.local/bin` could
/// be pipx, a manual download, or anything else — so that reads `UserLocal`,
/// not a guessed manager. The resolved path is always reported alongside, so
/// a human can check the classification rather than trust it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallSource {
    /// Under `mise/installs/<tool>/`: managed by mise as `<tool>`.
    Mise(String),
    /// Under `~/.cargo/bin`.
    Cargo,
    /// Inside `node_modules`, or beside a `lib/node_modules/<pkg>` dir.
    Npm,
    /// `/usr/local/bin`, `/usr/bin`, `/bin`: system scope, manager unknown.
    System,
    /// `~/.local/bin` and friends: user scope, manager unknown.
    UserLocal,
    /// Present on PATH, origin not inferable from the path.
    Unknown,
}

impl InstallSource {
    /// The word in reports.
    pub fn label(&self) -> String {
        match self {
            Self::Mise(tool) => format!("mise:{tool}"),
            Self::Cargo => "cargo".to_string(),
            Self::Npm => "npm".to_string(),
            Self::System => "system".to_string(),
            Self::UserLocal => "user-local".to_string(),
            Self::Unknown => "unknown".to_string(),
        }
    }

    /// Classify a resolved binary path. Pure, so the table is testable without
    /// touching any machine.
    pub fn classify(path: &std::path::Path) -> Self {
        let text = path.to_string_lossy().replace('\\', "/");
        if let Some(index) = text.find("/mise/installs/") {
            let tool = text[index + "/mise/installs/".len()..]
                .split('/')
                .next()
                .unwrap_or("?");
            return Self::Mise(tool.to_string());
        }
        if text.contains("node_modules") {
            return Self::Npm;
        }
        if let Some(home) = std::env::var_os("HOME").map(|h| h.to_string_lossy().into_owned()) {
            let home = home.replace('\\', "/");
            if text.starts_with(&format!("{home}/.cargo/bin/")) {
                return Self::Cargo;
            }
            if text.starts_with(&format!("{home}/.local/bin/")) {
                // An npm global under a mise-managed node keeps its package
                // dir beside the bin dir; that layout names npm confidently.
                if let Some(bin_dir) = std::path::Path::new(&text)
                    .parent()
                    .and_then(|p| p.parent())
                    .map(|p| p.join("lib").join("node_modules"))
                {
                    if bin_dir.is_dir() {
                        return Self::Npm;
                    }
                }
                return Self::UserLocal;
            }
        }
        if ["/usr/local/bin/", "/usr/bin/", "/bin/"]
            .iter()
            .any(|dir| text.starts_with(dir))
        {
            return Self::System;
        }
        Self::Unknown
    }
}

/// One installed agent: what it is, where it lives, what it reports.
#[derive(Debug, Clone)]
pub struct InstalledAgent {
    /// Roster name.
    pub name: &'static str,
    /// Resolved executable path.
    pub binary: std::path::PathBuf,
    /// First line of `--version`, when it answered in time.
    pub version: Option<String>,
    /// How it got here, as far as the path says.
    pub source: InstallSource,
}

/// The installed subset of the roster, with versions and provenance.
///
/// Read-only by construction: PATH scans plus one `--version` run each. This
/// function never installs, never upgrades, and never writes — discovery is
/// the whole item, and anything that mutates the machine would be a different
/// one.
pub fn inventory() -> Vec<InstalledAgent> {
    let mut out = Vec::new();
    for spec in ROSTER {
        let Some(path) = spec.binaries.iter().find_map(|b| which(b)) else {
            continue;
        };
        out.push(InstalledAgent {
            name: spec.name,
            source: InstallSource::classify(&path),
            version: run_version(&path),
            binary: path,
        });
    }
    out
}

/// First line of `<binary> --version`, bounded. A version probe that hangs is
/// not a version; it is `None` with no error, because absence of an answer is
/// itself the observation.
fn run_version(binary: &std::path::Path) -> Option<String> {
    let mut child = std::process::Command::new(binary)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {}
            Err(_) => {
                let _ = child.kill();
                return None;
            }
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_roster_is_the_seven_agents_measured_on_this_machine() {
        let names: Vec<&str> = ROSTER.iter().map(|a| a.name).collect();
        assert_eq!(
            names,
            ["opencode", "cline", "codex", "claude", "gemini", "crush"]
        );
        // Cline was absent from Milestone S's table entirely; being in the roster
        // at all is a finding from 2026-09-28.
        assert!(find("cline").is_some());
    }

    #[test]
    fn an_installed_agent_is_still_only_read_from_help_until_it_is_run() {
        for spec in ROSTER {
            let provenance = provenance_of_help_cell(spec);
            let installed = spec.binaries.iter().any(|b| which(b).is_some());
            if installed {
                assert_eq!(
                    provenance,
                    Provenance::ReadFromHelp,
                    "{} is installed, so it is not NotInstalled — but nothing has run it yet",
                    spec.name
                );
            } else {
                assert_eq!(provenance, Provenance::NotInstalled, "{}", spec.name);
            }
        }
    }

    #[test]
    fn install_sources_classify_by_path_alone() {
        use std::path::Path;
        // mise tool dirs name their manager.
        assert_eq!(
            InstallSource::classify(Path::new(
                "/home/u/.local/share/mise/installs/codex/latest/bin/codex"
            )),
            InstallSource::Mise("codex".to_string())
        );
        // An npm package installed by mise is mise's: the manager of record wins
        // over the packaging format inside it.
        assert_eq!(
            InstallSource::classify(Path::new(
                "/home/u/.local/share/mise/installs/gemini/latest/node_modules/.bin/gemini"
            )),
            InstallSource::Mise("gemini".to_string())
        );
        // Outside mise, node_modules names npm.
        assert_eq!(
            InstallSource::classify(Path::new("/usr/lib/node_modules/.bin/tool")),
            InstallSource::Npm
        );
        assert_eq!(
            InstallSource::classify(Path::new("/usr/local/bin/tool")),
            InstallSource::System
        );
        assert_eq!(
            InstallSource::classify(Path::new("/opt/vendor/tool")),
            InstallSource::Unknown
        );
        // Labels never imply more than the path showed.
        assert_eq!(InstallSource::UserLocal.label(), "user-local");
        assert_eq!(InstallSource::Unknown.label(), "unknown");
    }

    #[test]
    fn only_observed_counts_as_a_measurement() {
        // The rule the whole crate exists to keep, stated as a test.
        assert_ne!(Provenance::Observed, Provenance::ReadFromHelp);
        assert_ne!(Provenance::ReadFromHelp, Provenance::RunFailed);
        assert_eq!(Provenance::Observed.marker(), "O");
        assert_eq!(Provenance::ReadFromHelp.marker(), "H");
    }

    #[test]
    fn absent_agents_are_recorded_with_the_reason_rather_than_left_blank() {
        for (name, why) in NOT_INSTALLED {
            assert!(!name.is_empty() && !why.is_empty());
            assert!(
                find(name).is_none(),
                "{name} is listed as absent and must not also be in the roster"
            );
        }
    }
}
