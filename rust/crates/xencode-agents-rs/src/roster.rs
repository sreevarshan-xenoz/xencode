//! Which agents this probe knows about, and what was already read about them.
//!
//! The capability cells here come from `--help` and `--version` output captured on
//! 2026-09-28 (Milestone W). They are **not measurements**: they are a starting
//! roster, and every one of them is marked as read-only evidence so a reader can
//! tell at a glance which parts of the matrix still need running.

use serde::{Deserialize, Serialize};

/// How a cell in the matrix came to be believed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// The vendor's own command that opens one of its **already-running**
    /// sessions in this terminal. The placeholder is the vendor's own argument
    /// name — `{id}` for a session, `{url}` for a server address — so the row
    /// says what has to go there instead of xencode inventing a word for it.
    ///
    /// `None` is the common case and the honest one. An agent whose help
    /// advertises resuming a saved conversation as a flag on a *new* run has no
    /// running process to hand a terminal to, and `OR-14`'s done-when is exactly
    /// that difference: "attach" may only ever mean handing the real terminal to
    /// a process that has one, so xencode starting a second copy of the vendor
    /// and calling it a take-over would be the lie this field exists to prevent.
    /// Read from `--help` on 2026-10-07.
    pub attach: Option<&'static str>,
    /// The vendor's own command that prints the sessions it knows about, which
    /// is where a `{id}` for [`AgentSpec::attach`] comes from. xencode **prints
    /// this command for the person to run** and never runs it: a listing that
    /// browses instead of printing would dump escape codes into the terminal,
    /// and whether a vendor's own tool is worth starting is `AR-8`'s rule to
    /// leave with the human. `None` when the help read here shows no such
    /// command — recorded as not known rather than guessed at, because a wrong
    /// listing command prints nothing and a person would read that as an empty
    /// session list.
    /// Read from `--help` on 2026-10-07.
    pub session_list: Option<&'static str>,
    /// The operator has stood this agent down, so the probe skips it by default.
    ///
    /// This is a decision, not an observation. A parked agent is still
    /// installed and still fully described — its roster row and its capability
    /// cells stay, because what is true about it has not stopped being true.
    /// What changes is that a run stops reporting it as a failure it cannot fix:
    /// an agent with no account left every report carrying three login walls
    /// that no amount of re-probing would clear, which trains a reader to ignore
    /// the failures that *are* actionable.
    ///
    /// Naming the agent explicitly (`--agent claude`) still probes it, because
    /// the operator asking for it is a fresh decision.
    pub parked: bool,
    /// Why this agent was parked, in words fit for a report.
    ///
    /// `None` while `parked` is false. Kept beside the flag rather than folded
    /// into it because the flag says what happened and this says whether it was
    /// a choice about the vendor or a missing account — and only the second is
    /// fixed by logging in.
    pub park_reason: Option<&'static str>,
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
        // `opencode attach <url>` — "attach to a running opencode server", read
        // from `opencode --help` on 2026-10-07. The placeholder is named `{url}`
        // because that is what the vendor's own usage line asks for: the address
        // a server it already started printed. `opencode session list` names the
        // sessions on disk, which is a different thing, and is recorded as
        // `session_list` for that reason rather than as the source of a `{url}`.
        attach: Some("opencode attach {url}"),
        // `opencode session list` was run on 2026-10-07 and printed real session
        // ids from this machine's own state, with no network involved.
        session_list: Some("opencode session list"),
        parked: false,
        park_reason: None,
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
        // `--id <session-id>` resumes a session, but as a flag on a run cline
        // starts here and now — there is no cline process already holding that
        // terminal, so there is nothing to hand over (2026-10-07 help read).
        attach: None,
        session_list: None,
        parked: false,
        park_reason: None,
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
        // `codex agents` "browse[s] all agent sessions on the shared local
        // app-server daemon" and `codex resume` opens a picker; neither takes a
        // session argument to hand this terminal to, so xencode reads no
        // handover verb here (2026-10-07 help read). That is a statement about
        // what was read, not a claim codex cannot be attached to.
        attach: None,
        session_list: None,
        parked: false,
        park_reason: None,
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
        // `claude attach <id>` — "Open the background session in this terminal",
        // read from `claude attach --help` on 2026-10-07. A session started with
        // `--bg` keeps running with no terminal, and this is the vendor's own way
        // to give it one.
        attach: Some("claude attach {id}"),
        // Its `--bg` help says "`claude agents` lists them", so the vendor names
        // this as the way to see the background sessions. xencode prints the
        // command rather than running it: it may browse rather than print, and
        // deciding that is the person's, not ours.
        session_list: Some("claude agents"),
        parked: true,
        park_reason: Some("no Claude account on this box; the operator has not asked for one"),
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
        // `-r/--resume` takes "latest" or an index on a new run, which is a
        // conversation picked up, not a running process handed a terminal
        // (2026-10-07 help read).
        attach: None,
        session_list: None,
        parked: true,
        park_reason: Some("no Gemini auth configured; the operator has not asked for one"),
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
        // `crush server` binds a socket and `crush session list` lists sessions,
        // but help shows no `attach`: nothing in what was read takes a session
        // and puts it in this terminal (2026-10-07 help read).
        attach: None,
        session_list: Some("crush session list"),
        parked: true,
        park_reason: Some("no provider configured in crush; the operator has not asked for one"),
        read_on: "2026-09-28",
    },
    AgentSpec {
        name: "agy",
        binaries: &["agy"],
        // Read from `agy --help` on 2026-10-02, which AR-2's done-when names
        // but the roster had never covered. `-p/--print` is the headless one
        // shot and `--output-format stream-json` the machine-readable stream,
        // both confirmed in the help text above; a probe run is what confirms
        // they behave as advertised.
        one_shot: "agy --print {prompt}",
        stream_flag: Some("--output-format stream-json"),
        advertises_daemon: true,   // `--remote-control`, `agy remote-control`
        advertises_acp: false,     // not in help
        advertises_mcp: true,      // `agy mcp`
        advertises_resume: true,   // `--conversation`, `--continue`
        advertises_approval: true, // `--mode`, `--dangerously-skip-permissions`
        // `agy agents` lists the agent *definitions* it knows, not running
        // sessions, and `--continue`/`--conversation` resume on a new run — so
        // there is no handover verb and no session listing to offer here
        // (2026-10-07 help read).
        attach: None,
        session_list: None,
        parked: false,
        park_reason: None,
        read_on: "2026-10-02",
    },
    AgentSpec {
        name: "cursor-agent",
        binaries: &["cursor-agent"],
        // Read from `cursor-agent --help` on 2026-10-02. It is installed here
        // and answers `--version` (2026.09.28-64d2043), but no probe had ever
        // run it: it was in no list at all, so the gap in knowledge was
        // invisible rather than recorded.
        // `--trust` is not optional in a non-interactive run, measured
        // 2026-10-02: without it cursor-agent refuses with "⚠ Workspace Trust
        // Required" and exits 1, which reads as a broken agent rather than a
        // deliberate prompt it cannot show. It trusts *the scratch fixture the
        // probe itself created*, and is not an account login.
        one_shot: "cursor-agent --print --trust {prompt}",
        stream_flag: Some("--output-format stream-json"),
        advertises_daemon: true,   // `persist`, `worker` subcommands
        advertises_acp: false,     // not in help
        advertises_mcp: true,      // `cursor-agent mcp`, `--approve-mcps`
        advertises_resume: true,   // `--resume`, `--continue`
        advertises_approval: true, // `--mode`, `--force`/`--yolo`, `--sandbox`
        // `resume` "resume[s] the latest chat session" and `--resume [chatId]`
        // picks one up on a new run; neither is a process already holding a
        // terminal, so there is nothing for xencode to hand over (2026-10-07
        // help read).
        attach: None,
        session_list: None,
        parked: false,
        park_reason: None,
        read_on: "2026-10-02",
    },
    AgentSpec {
        name: "kilo",
        // Installed 2026-10-02 under `~/.kilo/bin/`, which is **not** on
        // `PATH`, so a bare `kilo` does not resolve. `~/.kilo/bin` is where
        // kilo's own installer puts it; `which` walks `PATH` only, so this
        // entry keeps the documented location as a fallback rather than
        // reporting an installed agent as absent.
        binaries: &["kilo", ".kilo/bin/kilo"],
        one_shot: "kilo run {prompt}",
        stream_flag: Some("--format json"),
        advertises_daemon: true,   // `kilo serve`, `kilo attach <url>`
        advertises_acp: true,      // `kilo acp`
        advertises_mcp: true,      // `kilo mcp`
        advertises_resume: true,   // `kilo session`, `kilo run --continue`
        advertises_approval: true, // `kilo run --auto`
        // `kilo attach <url>` — "attach to a running kilo server", read from
        // `kilo --help` on 2026-10-07. Same shape as opencode: the target is the
        // address a server this machine started is listening on.
        attach: Some("kilo attach {url}"),
        session_list: Some("kilo session list"),
        parked: false,
        park_reason: None,
        read_on: "2026-10-02",
    },
    AgentSpec {
        name: "kiro-cli",
        // AWS Kiro. The binary is `kiro-cli`, not `kiro`, on `PATH`; read from
        // `kiro-cli --help-all` and `kiro-cli chat --help` on 2026-10-02.
        binaries: &["kiro-cli"],
        one_shot: "kiro-cli chat --no-interactive --output-format stream-json {prompt}",
        stream_flag: Some("--output-format stream-json"),
        advertises_daemon: false, // `crew`, `--cloud` are hosted, not a local daemon
        advertises_acp: true,     // `kiro-cli acp`
        advertises_mcp: true,     // `kiro-cli mcp`
        advertises_resume: true,  // `-r/--resume`, `--resume-id`
        advertises_approval: true, // `-a/--trust-all-tools`, `--trust-tools`
        // `kiro-cli --help` on 2026-10-07 lists chat, agent, doctor, settings and
        // quit: no attach verb and no session listing in what it showed.
        attach: None,
        session_list: None,
        parked: false,
        park_reason: None,
        read_on: "2026-10-02",
    },
];

/// Agents the proposal names that this crate does not know how to launch.
///
/// These are **candidates worth probing**, not a claim that they are missing.
/// The list used to be a hand-typed statement of absence, which is how `agy`
/// came to be reported as "not installed" while sitting on `PATH` answering
/// `1.2.13` — a list nobody re-checked against the machine. Whether one of
/// these is actually here is now decided by [`which`], and a name found on
/// `PATH` is reported as installed-but-unknown rather than absent.
///
/// Kilo had every claim about it marked UNVERIFIED in Milestone W, and the
/// reason is now settled rather than guessed: kilo was never installed on this
/// machine, so there was nothing to verify against. Both names moved into
/// [`ROSTER`] on 2026-10-02 once they were installed, which empties this list.
///
/// A candidate is added by name on purpose. `AR-2`'s done-when requires
/// discovery to say nothing about the rest of the filesystem, so scanning `PATH`
/// for anything that looks like an agent would breach it. Naming an agent we
/// were asked about keeps the search deliberate while still checking the answer.
pub const CANDIDATES: &[(&str, &str)] = &[];

/// Named agents that are genuinely not on this machine, and why that matters.
///
/// Each entry is `(name, reason)` where the reason is only shown when the lookup
/// actually found nothing, so a stale entry cannot outlive the machine it was
/// written for.
pub fn absent_agents() -> Vec<(String, String)> {
    CANDIDATES
        .iter()
        .filter(|(name, _)| which(name).is_none())
        .map(|(name, why)| ((*name).to_string(), (*why).to_string()))
        .collect()
}

/// Named agents that are installed but that this crate cannot launch yet.
///
/// Reported instead of dropped: an installed agent missing from [`ROSTER`] is a
/// gap in our knowledge, and saying so is the difference between "nothing to
/// measure" and "we did not look".
pub fn installed_but_unknown() -> Vec<(String, String)> {
    CANDIDATES
        .iter()
        .filter_map(|(name, why)| {
            which(name).map(|path| {
                (
                    (*name).to_string(),
                    format!("{why} (found at {})", path.display()),
                )
            })
        })
        .collect()
}

/// Agents the operator has stood down, with the reason they were stood down.
///
/// The reason is kept with the decision because a park is not permanent: whoever
/// unparkes an agent later needs to know whether it was parked because it is
/// unwanted or because its account was missing, and those call for opposite
/// amounts of work.
pub fn parked_agents() -> Vec<(String, String)> {
    ROSTER
        .iter()
        .filter(|a| a.parked)
        .map(|a| {
            (
                a.name.to_string(),
                a.park_reason
                    .unwrap_or("parked by the operator")
                    .to_string(),
            )
        })
        .collect()
}

/// Look one agent up by name.
pub fn find(name: &str) -> Option<&'static AgentSpec> {
    ROSTER.iter().find(|a| a.name == name)
}

/// Whether this worker name is on the roster — that is, whether xencode has a
/// row saying it belongs to another vendor's coding-agent CLI.
///
/// `true` is a fact about the list, not a guess from a name: every row here was
/// written from a binary somebody else installs, signs into and bills, so work
/// handed to one travels through a program whose traffic xencode does not
/// control. `false` is deliberately weaker than "local". It means there is no
/// roster row to read, which is the same thing [`find`] could not answer — a
/// caller must say it as an unanswered question rather than as permission.
pub fn is_external_worker(name: &str) -> bool {
    find(name).is_some()
}

/// What a terminal handover to `{name}`'s own session would take, decided from
/// its roster row and what is on this machine.
///
/// This is a sum type rather than an `Option` because the three refusals say
/// different things and each has to be quotable: an agent with no handover verb
/// in its help is a different fact from an agent whose verb is known but whose
/// program is not here, and both differ from the case where the person has not
/// yet said which session. Guessing at the last one — passing an empty id, or
/// picking the newest session — is how a control plane starts lying about what
/// it attached to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Handover {
    /// The row's own command, as argv, ready to be handed this terminal. The
    /// first element is the resolved path when this machine has the program.
    Ready {
        argv: Vec<String>,
        /// The roster template it came from, so the screen can quote the
        /// vendor's own usage line back rather than xencode's reconstruction.
        template: &'static str,
    },
    /// Nothing in what was read from this agent's help hands a terminal to a
    /// running session. `one_shot` is what the row *does* advertise, so the
    /// refusal can name the difference instead of just saying no.
    NoHandoverVerb {
        one_shot: &'static str,
        read_on: &'static str,
        /// The vendor's own way to list its sessions, when help shows one —
        /// printed for the person to run, never run here.
        session_list: Option<&'static str>,
    },
    /// The row advertises the verb but the program is not on this machine.
    NotInstalled { template: &'static str },
    /// A session was not named, and xencode does not choose one for you.
    NeedsTarget {
        template: &'static str,
        session_list: Option<&'static str>,
    },
    /// No roster row for that name, so nothing is known about how to hand a
    /// terminal to it.
    Unknown(String),
}

/// What `orchestrator attach` may do with `name`, given the session it was
/// pointed at. See [`Handover`].
pub fn handover(name: &str, target: Option<&str>) -> Handover {
    let Some(spec) = find(name) else {
        return Handover::Unknown(name.to_string());
    };
    let Some(template) = spec.attach else {
        return Handover::NoHandoverVerb {
            one_shot: spec.one_shot,
            read_on: spec.read_on,
            session_list: spec.session_list,
        };
    };
    let Some(target) = target.filter(|t| !t.trim().is_empty()) else {
        return Handover::NeedsTarget {
            template,
            session_list: spec.session_list,
        };
    };
    let Some(binary) = spec.binaries.iter().find_map(|b| which(b)) else {
        return Handover::NotInstalled { template };
    };
    Handover::Ready {
        argv: attach_argv(template, target, &binary.to_string_lossy()),
        template,
    }
}

/// Split a roster handover template into argv, putting `target` where the row's
/// placeholder is and `binary` — the path this machine resolved — first.
///
/// The placeholder is matched as a whole word (`{id}`, `{url}`) because the
/// vendor's own argument name is documentation the person reading the line
/// should keep seeing: a row that says `attach {url}` and a row that says
/// `attach {id}` want different things after them.
pub fn attach_argv(template: &str, target: &str, binary: &str) -> Vec<String> {
    let mut argv: Vec<String> = template
        .split_whitespace()
        .map(|token| {
            if token.starts_with('{') && token.ends_with('}') {
                target.to_string()
            } else {
                token.to_string()
            }
        })
        .collect();
    if !argv.is_empty() {
        argv[0] = binary.to_string();
    }
    argv
}

/// Resolve an executable name to a full path, the way `which` would.
///
/// A name containing a path separator is resolved directly rather than walked
/// on `PATH`, which is how an agent installed outside `PATH` is still found:
/// `kilo` installs itself to `~/.kilo/bin/kilo` and adds nothing to the shell
/// profile, so a bare `kilo` does not resolve while the binary is present and
/// working. A `~` prefix is expanded against `$HOME`, or `%USERPROFILE%` where
/// no `HOME` is set. A bare name is looked up the way a shell would run it,
/// including `claude.exe` and `claude.cmd` on Windows.
pub fn which(binary: &str) -> Option<std::path::PathBuf> {
    if binary.contains('/') {
        let relative = binary.strip_prefix("~/").unwrap_or(binary);
        let full = match binary.strip_prefix("~/") {
            Some(_) => std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))?
                .into(),
            None => std::path::PathBuf::new(),
        }
        .join(relative);
        return full.is_file().then_some(full);
    }
    xencode_core_rs::sys::which(binary)
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
    /// `~/.kilo/bin/` and similar: the vendor's own installer put it there.
    VendorDotDir(String),
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
            Self::VendorDotDir(vendor) => format!("{vendor}-self-installed"),
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
            // A vendor that installs itself into its own dot-directory. Kilo
            // lands in `~/.kilo/bin/` and is not on PATH at all, which the path
            // makes obvious and is worth naming rather than reporting as
            // unclassifiable.
            if let Some(vendor) = ["kilo", "kiro", "agy", "cursor"]
                .iter()
                .find(|d| text.starts_with(&format!("{home}/.{d}/")))
            {
                return Self::VendorDotDir((*vendor).to_string());
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
    fn the_roster_is_the_ten_agents_measured_on_this_machine() {
        let names: Vec<&str> = ROSTER.iter().map(|a| a.name).collect();
        assert_eq!(
            names,
            [
                "opencode",
                "cline",
                "codex",
                "claude",
                "gemini",
                "crush",
                "agy",
                "cursor-agent",
                "kilo",
                "kiro-cli"
            ]
        );
        // Two agents reached the roster because a probe met them, not because a
        // proposal named them. Cline was absent from Milestone S's table
        // entirely (2026-09-28). agy was *in* that table and in AR-2's
        // done-when, but had no AgentSpec, so the probe reported it absent while
        // its binary sat on PATH answering `1.2.13` — the absence list was a
        // hand-typed claim rather than a lookup (2026-10-02).
        // And a third found the other way: `cursor-agent` was in no list at
        // all, so its gap in our knowledge was invisible instead of recorded
        // (2026-10-02).
        assert!(find("cline").is_some());
        assert!(find("agy").is_some());
        assert!(find("cursor-agent").is_some());
        // And two more, installed later the same day and each found by a
        // different mistake: `kilo` because it is not on PATH at all, and
        // `kiro-cli` because its binary is not called `kiro`.
        assert!(find("kilo").is_some());
        assert!(find("kiro-cli").is_some());
    }

    /// The three agents whose help was read as carrying a command that hands
    /// this terminal to a session already running, and the seven whose does not.
    ///
    /// Pinned as a list on purpose: `OR-14`'s done-when is that "attach" only
    /// ever means the handover, so a row moving from `None` to `Some` is a claim
    /// about a vendor's binary that has to be read off that binary again, not
    /// typed in because it would be nice for the command to work.
    #[test]
    fn only_the_rows_that_advertise_an_attach_verb_can_be_attached_to() {
        let with_verb: Vec<&str> = ROSTER
            .iter()
            .filter(|spec| spec.attach.is_some())
            .map(|spec| spec.name)
            .collect();
        assert_eq!(with_verb, ["opencode", "claude", "kilo"]);
    }

    /// A placeholder is the vendor's own argument name, so the line a person
    /// reads says what belongs after it — and xencode never invents a session to
    /// fill it with.
    #[test]
    fn the_handover_placeholder_carries_the_named_session_and_nothing_else() {
        let argv = attach_argv("claude attach {id}", "ses_42", "/usr/local/bin/claude");
        assert_eq!(argv, ["/usr/local/bin/claude", "attach", "ses_42"]);
        // `attach` with no argument is a different program's error, not a
        // question xencode answers by guessing.
        let no_placeholder = attach_argv("codex agents", "ignored", "/bin/codex");
        assert_eq!(no_placeholder, ["/bin/codex", "agents"]);
    }

    #[test]
    fn attaching_is_refused_in_words_that_distinguish_why() {
        // Nothing was read that hands a terminal over: the refusal says what the
        // row *does* advertise, so the person learns the difference rather than
        // being told a command failed.
        let cline = handover("cline", Some("anything"));
        assert!(
            matches!(
                &cline,
                Handover::NoHandoverVerb { one_shot, .. } if *one_shot == "cline --json {prompt}"
            ),
            "cline resumes with `--id` on a run it starts, which is not a handover: {cline:?}"
        );

        // The verb exists but no session was named, and xencode does not pick one.
        let claude = handover("claude", None);
        assert_eq!(
            claude,
            Handover::NeedsTarget {
                template: "claude attach {id}",
                session_list: Some("claude agents"),
            }
        );
        // A blank is not a session either.
        assert_eq!(
            handover("claude", Some("  ")),
            Handover::NeedsTarget {
                template: "claude attach {id}",
                session_list: Some("claude agents"),
            }
        );

        assert!(
            matches!(handover("not-on-the-roster", Some("x")),
                     Handover::Unknown(name) if name == "not-on-the-roster"),
            "a name with no row is answered as an unanswered question"
        );
    }

    /// Whether this resolves to a command or to "that program is not here" is a
    /// fact about the machine, not about the row — so the test asks which of the
    /// two it was, and that the one it got is well-formed.
    #[test]
    fn a_named_session_on_an_agent_with_a_handover_verb_resolves_or_says_it_is_absent() {
        let opencode = handover("opencode", Some("http://127.0.0.1:54321"));
        match &opencode {
            Handover::Ready { argv, template } => {
                assert_eq!(*template, "opencode attach {url}");
                assert_eq!(argv.len(), 3, "opencode, attach, and the url: {argv:?}");
                assert!(
                    std::path::Path::new(&argv[0]).is_file(),
                    "the resolved binary must be a file that exists: {argv:?}"
                );
                assert_eq!(&argv[1], "attach");
                assert_eq!(&argv[2], "http://127.0.0.1:54321");
            }
            Handover::NotInstalled { template } => {
                assert_eq!(*template, "opencode attach {url}");
            }
            other => {
                panic!("opencode has a handover verb and a named url, so it is neither {other:?}")
            }
        }
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

    /// A vendor that installs itself into its own dot-directory is not on
    /// `PATH` at all, so a `PATH`-only lookup reports it missing. Kilo landed
    /// here on 2026-10-02 and produced exactly that wrong answer until the
    /// lookup learned to check the documented location too.
    #[test]
    fn an_agent_installed_outside_path_is_still_found() {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .expect("HOME");
        // The lookup finds a path-shaped name, so a binary the vendor parked
        // outside PATH is not invisible. Asserted only where agy is installed
        // there, like kilo below.
        if std::path::Path::new(&format!("{home}/.local/bin/agy")).is_file() {
            assert_eq!(
                which("~/.local/bin/agy"),
                Some(std::path::PathBuf::from(home.clone()).join(".local/bin/agy"))
            );
        }
        // A real file outside PATH that is on this machine today. Asserted
        // against the machine only when it exists there, so the test still
        // means something on one without kilo.
        if std::path::Path::new(&format!("{home}/.kilo/bin/kilo")).is_file() {
            assert_eq!(
                which("~/.kilo/bin/kilo"),
                Some(std::path::PathBuf::from(format!("{home}/.kilo/bin/kilo"))),
                "a binary outside PATH is still found by its documented location"
            );
        }
        // A path-shaped name that is not there is still nothing, rather than
        // being reported as installed.
        assert_eq!(which("~/.definitely-not-here/nothing"), None);
        // And its own dot-dir is now named rather than reported as unknown.
        if std::path::Path::new(&format!("{home}/.kilo/bin/kilo")).is_file() {
            assert_eq!(
                InstallSource::classify(std::path::Path::new(&format!("{home}/.kilo/bin/kilo"))),
                InstallSource::VendorDotDir("kilo".to_string())
            );
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
    fn every_candidate_is_either_present_or_gives_a_reason_for_its_absence() {
        for (name, why) in CANDIDATES {
            assert!(!name.is_empty() && !why.is_empty());
            assert!(
                find(name).is_none(),
                "{name} is listed as a candidate and must not also be in the roster"
            );
        }
    }

    /// A name can be checked against the machine under more than one spelling.
    /// `kiro` was installed as `kiro-cli`, so looking only for the product name
    /// reported it missing while the binary sat in `~/.local/bin`; and `kilo`
    /// was installed outside `PATH` entirely, where even its real name does not
    /// resolve. Both are one entry with two executable names.
    #[test]
    fn an_agent_is_found_by_any_of_its_binary_names() {
        for spec in ROSTER {
            let found = spec.binaries.iter().find_map(|b| which(b));
            assert!(
                found.is_some(),
                "{} lists {:?} and none of them resolve",
                spec.name,
                spec.binaries
            );
        }
    }

    /// The bug this replaced: a name was called "not installed" from a
    /// hand-typed list, so it stayed wrong while the binary sat on `PATH`.
    /// Absence and presence are now both decided by the same `PATH` lookup,
    /// and the two answers can never overlap.
    #[test]
    fn presence_is_measured_and_the_two_lists_never_disagree() {
        let absent: Vec<String> = absent_agents().into_iter().map(|(n, _)| n).collect();
        let unknown: Vec<String> = installed_but_unknown()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for name in &absent {
            assert!(
                !unknown.contains(name),
                "{name} is reported both absent and installed"
            );
            assert!(
                which(name).is_none(),
                "{name} claimed absent but is on PATH"
            );
        }
        for name in &unknown {
            assert!(
                which(name).is_some(),
                "{name} claimed present but is not on PATH"
            );
            assert!(!absent.contains(name));
            // Anything installed but unknown must be a real gap: the roster is
            // the set we know how to launch.
            assert!(find(name).is_none(), "{name} is on PATH and in the roster");
        }
    }

    /// An agent on `PATH` that the roster does not cover is reported, not lost.
    /// `agy` is installed here and was being called absent; the whole point is
    /// that a reader can tell those two situations apart.
    #[test]
    fn an_installed_agent_outside_the_roster_is_named_as_unknown() {
        // `sh` is guaranteed on PATH and is deliberately not an agent.
        assert!(find("sh").is_none());
        assert!(which("sh").is_some(), "this test needs a PATH to look at");
        // The two functions together partition the candidates by lookup, so a
        // name can be in neither list only if it is genuinely off PATH.
        let covered: Vec<String> = absent_agents()
            .into_iter()
            .chain(installed_but_unknown())
            .map(|(n, _)| n)
            .collect();
        for (name, _) in CANDIDATES {
            assert!(covered.contains(&(*name).to_string()));
        }
    }

    /// The roster row is the whole of what `is_external_worker` claims, so the
    /// answer cannot drift from the list and never guesses from a name.
    #[test]
    fn external_is_answered_from_the_roster_row_and_not_from_the_shape_of_a_name() {
        for spec in ROSTER {
            assert!(
                is_external_worker(spec.name),
                "{} has a roster row and so is somebody else's agent",
                spec.name
            );
        }
        // A name that merely looks like it could belong to a vendor is not on
        // the roster, and the answer xencode can give for it is "no row", not
        // "local" — that distinction is the caller's to print.
        for name in ["xencode", "sh", "open-code", "codex-2", ""] {
            assert!(
                !is_external_worker(name),
                "{name} is not a roster name, so it cannot be claimed as another vendor's agent"
            );
        }
        assert_eq!(
            ROSTER.iter().filter(|s| is_external_worker(s.name)).count(),
            ROSTER.len()
        );
    }
}
