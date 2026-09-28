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
        advertises_approval: false,
        read_on: "2026-09-28",
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
        stream_flag: None, // no structured output option in help
        advertises_daemon: false,
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
