//! Contract probe (`AR-3`): the flags each agent actually advertises.
//!
//! The roster's capability cells were read from `--help` by a human, with the
//! evidence cited in comments. Comments rot; this probe re-runs the reading on
//! every invocation, against the live `--help` of every installed agent, and
//! reports confirmed or contradicted per claim. A contradiction is a stale
//! roster cell with the excerpt attached — the firewall working, not a failure.
//!
//! Two deliberate limits, both documented where they bite:
//!
//! - Only `--help` output is consulted — top-level plus the one-shot
//!   subcommand's help, because that is exactly the invocation path the probe
//!   uses. Anything a vendor documents elsewhere is invisible here, and the
//!   report says which helps were read.
//! - An absence is a measurement only when a token was searched for it. Where
//!   the roster denies a capability and an evidence token is registered, the
//!   probe says *not advertised* and names the words it looked for; where no
//!   token is registered it says *could not be tested*, because reporting an
//!   unexamined claim as a confirmed absence would hand the reader a fact
//!   nobody measured. That is also why the opposite direction — a token found
//!   under a denied claim — contradicts the roster only for `acp` and `mcp`:
//!   there is no single word that means "daemon" (`serve` also matches "MCP
//!   server names"), so an ambiguous word seen in help is reported as
//!   untestable rather than as a contradiction manufactured out of a substring.

use crate::roster::{which, AgentSpec, ROSTER};

/// What the probe concluded about one claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Expected and found, or expected-absent and absent.
    Confirmed,
    /// Expected but missing, or unexpected but present, with evidence.
    Contradicted(String),
    /// The agent is not installed, or its help could not be read.
    Untestable(String),
}

/// One claim and how it fared.
#[derive(Debug, Clone)]
pub struct ClaimResult {
    /// Roster name.
    pub agent: &'static str,
    /// `stream`, `acp`, `mcp`, `resume`, `daemon`, or `approval`.
    pub claim: &'static str,
    /// Whether the roster asserts it.
    pub expected: bool,
    /// Markers found / missing, for the report.
    pub found: Vec<String>,
    pub missing: Vec<String>,
    /// Which help screens were read.
    pub sources: Vec<String>,
    /// The verdict.
    pub verdict: Verdict,
}

impl ClaimResult {
    /// What this one claim is worth to a reader, in one line: the verdict, the
    /// screens it was read from, and the tokens that carried it. It says nothing
    /// about *which* claim that is — the name travels with the line wherever it is
    /// quoted, so this does not repeat it. Routing prints it beside a capability so
    /// a reason can be checked rather than trusted.
    pub fn evidence_line(&self) -> String {
        let read = if self.sources.is_empty() {
            "no help screen could be read".to_string()
        } else {
            format!(
                "read from {}",
                self.sources
                    .iter()
                    .map(|s| format!("`{} {}`", self.agent, s))
                    .collect::<Vec<_>>()
                    .join(" and ")
            )
        };
        match (&self.verdict, self.expected) {
            (Verdict::Confirmed, true) if !self.found.is_empty() => format!(
                "confirmed by {} ({read})",
                self.found
                    .iter()
                    .map(|t| format!("`{t}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            (Verdict::Confirmed, true) => {
                format!("the roster asserts it and nothing contradicted it ({read})")
            }
            (Verdict::Confirmed, false) if !self.missing.is_empty() => format!(
                "not advertised — {} searched for and absent ({read})",
                self.missing
                    .iter()
                    .map(|t| format!("`{t}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            (Verdict::Confirmed, false) => {
                format!("not advertised — no token for it appeared ({read})")
            }
            (Verdict::Contradicted(detail), _) => {
                format!("contradicted, {detail} ({read})")
            }
            (Verdict::Untestable(detail), _) => {
                format!("could not be tested here, {detail}")
            }
        }
    }
}

/// Evidence tokens per (agent, claim), taken from the roster comments and
/// verified against live `--help` on 2026-09-28. A token here is a promise: the
/// probe fails a claimed capability if the token stops appearing, and fails a
/// denied one (for `acp` and `mcp`, the two words with no competing meaning) if
/// it starts appearing. A (agent, claim) pair with no token listed has never
/// been searched, and [`probe_contract`] reports it as untestable rather than
/// as either a presence or an absence.
fn markers(agent: &str, claim: &str) -> &'static [&'static str] {
    match (agent, claim) {
        ("opencode", "stream") => &["--format"],
        ("opencode", "acp") => &["acp"],
        ("opencode", "mcp") => &["mcp"],
        ("opencode", "resume") => &["session", "export"],
        ("opencode", "daemon") => &["serve", "attach"],
        ("opencode", "approval") => &["--auto", "auto-approve"],
        ("cline", "stream") => &["--json"],
        ("cline", "acp") => &["acp"],
        ("cline", "mcp") => &["mcp"],
        ("cline", "resume") => &["--id", "history"],
        ("cline", "daemon") => &["hub", "zen"],
        ("cline", "approval") => &["auto-approve"],
        ("codex", "stream") => &["--json"],
        // The roster denies codex an `acp` mode, and `acp` is one of the two
        // words whose appearance in help means what it says. Measured absent
        // from `codex --help` and `codex exec --help` on 2026-10-08, so the
        // denial is now a searched absence rather than a sentence.
        ("codex", "acp") => &["acp"],
        ("codex", "mcp") => &["mcp"],
        ("codex", "resume") => &["resume"],
        ("codex", "daemon") => &["app-server", "daemon"],
        ("codex", "approval") => &["--sandbox", "--ask-for-approval"],
        ("claude", "stream") => &["--output-format", "stream-json"],
        // Denied, and searched: `acp` appears nowhere in `claude --help`
        // (2026-10-08).
        ("claude", "acp") => &["acp"],
        ("claude", "mcp") => &["mcp"],
        ("claude", "resume") => &["--resume", "--fork-session"],
        ("claude", "daemon") => &["attach", "--bg"],
        ("claude", "approval") => &["--permission-mode", "--allowedTools"],
        ("gemini", "stream") => &["--output-format", "stream-json"],
        ("gemini", "acp") => &["--acp"],
        ("gemini", "mcp") => &["mcp"],
        ("gemini", "resume") => &["--resume", "--session"],
        ("gemini", "approval") => &["--approval-mode"],
        // Denied, and searched: the roster's own reading of `gemini --help` is
        // that it has no daemon, and the word does not appear there (read again
        // 2026-10-08). Unlike `serve`, `daemon` has no competing meaning in
        // this screen.
        ("gemini", "daemon") => &["daemon"],
        ("crush", "resume") => &["--session", "--continue", "session"],
        ("crush", "daemon") => &["server"],
        ("crush", "approval") => &["--yolo"],
        // Both denied by the roster, and searched: neither `acp` nor `mcp`
        // appears in `crush --help` or `crush run --help` (2026-10-08).
        ("crush", "acp") => &["acp"],
        ("crush", "mcp") => &["mcp"],
        // agy, read from `agy --help` on 2026-10-02. It prints the literal
        // `Usage of agy:` with no subcommand section under `--help`, so every
        // claim is checked against that one screen.
        ("agy", "stream") => &["--output-format", "stream-json"],
        ("agy", "acp") => &["acp"],
        ("agy", "mcp") => &["mcp"],
        ("agy", "resume") => &["--conversation", "--continue"],
        ("agy", "daemon") => &["--remote-control"],
        ("agy", "approval") => &["--mode", "--dangerously-skip-permissions"],
        // cursor-agent, read from `cursor-agent --help` on 2026-10-02. Its
        // flags are at top level, and `--print` is the one-shot form.
        ("cursor-agent", "stream") => &["--output-format", "stream-json"],
        ("cursor-agent", "acp") => &["acp"],
        ("cursor-agent", "mcp") => &["mcp"],
        ("cursor-agent", "resume") => &["--resume", "--continue"],
        ("cursor-agent", "daemon") => &["persist", "worker"],
        ("cursor-agent", "approval") => &["--mode", "--force", "--sandbox"],
        // kilo, read from `kilo --help` and `kilo run --help` on 2026-10-02.
        // Its top-level help lists the subcommands, so every claim checks
        // against that one screen. Note `--format` here takes `json`, not
        // `stream-json` — a seventh different spelling for the same idea.
        ("kilo", "stream") => &["--format", "json"],
        ("kilo", "acp") => &["acp"],
        ("kilo", "mcp") => &["mcp"],
        ("kilo", "resume") => &["session", "--continue"],
        ("kilo", "daemon") => &["serve", "attach"],
        ("kilo", "approval") => &["--auto"],
        // kiro-cli, read from `kiro-cli --help-all` and `kiro-cli chat --help`
        // on 2026-10-02. Its flags live on the `chat` subcommand, so the probe
        // reads `kiro-cli chat --help` for the claims that are chat flags.
        ("kiro-cli", "stream") => &["--output-format", "stream-json"],
        ("kiro-cli", "acp") => &["acp"],
        ("kiro-cli", "mcp") => &["mcp"],
        ("kiro-cli", "resume") => &["--resume"],
        ("kiro-cli", "daemon") => &["crew"],
        ("kiro-cli", "approval") => &["--trust-all-tools", "--trust-tools"],
        _ => &[],
    }
}

/// Whether a token found under a *denied* claim contradicts it. Only the two
/// unambiguous words qualify: `acp` and `mcp` mean one thing each in a help
/// screen, while `serve`, `daemon`, `session` and the rest are words a vendor
/// uses for several things, so seeing one proves nothing about the capability it
/// was registered for. An absence is judged differently — by whether a token was
/// searched at all — and does not need this.
fn absence_checkable(claim: &str) -> bool {
    matches!(claim, "acp" | "mcp")
}

/// The subcommand of the one-shot form, when it has one (`codex exec` →
/// `exec`; `claude -p` → none, the flags live at top level).
fn oneshot_subcommand(spec: &AgentSpec) -> Option<&str> {
    let mut parts = spec.one_shot.split_whitespace();
    parts.next()?;
    let second = parts.next()?;
    if second.starts_with('-') || second.contains('{') || second.contains('}') {
        return None;
    }
    Some(second)
}

/// Run `<binary> [--help | <sub> --help]`, bounded. `None` when it cannot run.
///
/// Both stdout and stderr are read: several CLIs print help to stderr when
/// stdout is not a terminal (opencode does), and reading one stream finds
/// nothing while the other holds 56 lines.
fn help_text(binary: &std::path::Path, subcommand: Option<&str>) -> Option<String> {
    let mut command = std::process::Command::new(binary);
    if let Some(sub) = subcommand {
        command.arg(sub);
    }
    command
        .arg("--help")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = command.spawn().ok()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
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
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Some(text)
}

fn contains_token(help: &str, token: &str) -> bool {
    // Word boundaries for short tokens: `serve` must not match "servers", and
    // `acp` must not match inside a longer word. Longer tokens match plainly.
    if token.len() <= 6 && !token.starts_with("--") {
        let lower = help.to_lowercase();
        let token = token.to_lowercase();
        let mut start = 0;
        while let Some(index) = lower[start..].find(&token) {
            let absolute = start + index;
            let before = absolute == 0 || !lower.as_bytes()[absolute - 1].is_ascii_alphanumeric();
            let after = absolute + token.len() >= lower.len()
                || !lower.as_bytes()[absolute + token.len()].is_ascii_alphanumeric();
            if before && after {
                return true;
            }
            start = absolute + 1;
        }
        return false;
    }
    help.to_lowercase().contains(&token.to_lowercase())
}

/// Probe every claim of every installed agent.
pub fn probe_contract() -> Vec<ClaimResult> {
    let mut out = Vec::new();
    for spec in ROSTER {
        let Some(path) = spec.binaries.iter().find_map(|b| which(b)) else {
            continue;
        };
        let sub = oneshot_subcommand(spec);
        let mut helps = Vec::new();
        let mut sources = Vec::new();
        if let Some(top) = help_text(&path, None) {
            helps.push(top);
            sources.push("--help".to_string());
        }
        if let Some(sub) = sub {
            if let Some(text) = help_text(&path, Some(sub)) {
                helps.push(text);
                sources.push(format!("{sub} --help"));
            }
        }
        if helps.is_empty() {
            out.push(ClaimResult {
                agent: spec.name,
                claim: "(help unreadable)",
                expected: true,
                found: Vec::new(),
                missing: Vec::new(),
                sources,
                verdict: Verdict::Untestable("`--help` produced nothing usable".to_string()),
            });
            continue;
        }
        let combined: String = helps.join("\n");
        let mut claims: Vec<(&str, bool)> = vec![
            ("acp", spec.advertises_acp),
            ("mcp", spec.advertises_mcp),
            ("resume", spec.advertises_resume),
            ("daemon", spec.advertises_daemon),
            ("approval", spec.advertises_approval),
        ];
        if spec.stream_flag.is_some() {
            claims.push(("stream", true));
        }
        for (claim, expected) in claims {
            let tokens = markers(spec.name, claim);
            let mut found = Vec::new();
            let mut missing = Vec::new();
            for token in tokens {
                if contains_token(&combined, token) {
                    found.push(token.to_string());
                } else {
                    missing.push(token.to_string());
                }
            }
            out.push(ClaimResult {
                agent: spec.name,
                claim,
                expected,
                sources: sources.clone(),
                verdict: verdict_for(expected, claim, &found, &missing),
                found,
                missing,
            });
        }
    }
    out
}

/// What a token search proved, given what the roster asserts.
///
/// The four cases are the four ways this probe can be wrong, so they are worth
/// naming rather than collapsing into "confirmed":
///
/// | searched | asserted | conclusion |
/// |---|---|---|
/// | nothing | yes | contradicted — an asserted capability with no evidence behind it |
/// | nothing | no | untestable — an unexamined claim is not a confirmed absence |
/// | something | yes | confirmed, or contradicted by the tokens that were missing |
/// | something | no | confirmed absence, or untestable/contradicted by what turned up |
fn verdict_for(expected: bool, claim: &str, found: &[String], missing: &[String]) -> Verdict {
    if found.is_empty() && missing.is_empty() {
        return if expected {
            Verdict::Contradicted("no evidence tokens defined for this claim".to_string())
        } else {
            Verdict::Untestable(format!(
                "no help token is registered for {claim} on this agent, so nothing was searched \
                 for it"
            ))
        };
    }
    if expected {
        return if missing.is_empty() {
            Verdict::Confirmed
        } else {
            Verdict::Contradicted(format!("missing from help: {}", missing.join(", ")))
        };
    }
    if !found.is_empty() {
        return if absence_checkable(claim) {
            Verdict::Contradicted(format!(
                "present in help but roster says no: {}",
                found.join(", ")
            ))
        } else {
            Verdict::Untestable(format!(
                "{claim} is denied by the roster and {} appeared in help, but that word carries \
                 this claim neither way",
                found.join(", ")
            ))
        };
    }
    // Every registered token was searched and none of it appeared: that is an
    // absence with evidence behind it, and it names the words that were looked
    // for so a reader can tell it from a claim nobody examined.
    Verdict::Confirmed
}

use std::collections::{BTreeMap, BTreeSet};

/// Extract confirmed probed capabilities per agent from probe results.
///
/// Under AR-3's rule, "the router cannot see a capability that no probe recorded".
/// Only a claim the roster asserts *and* the probe confirms counts: a Confirmed
/// verdict on a claim the roster denies is a confirmed absence, which is the
/// opposite fact, and a router handed absences as abilities would route an `acp`
/// task to an agent whose help output never mentioned `acp`.
pub fn confirmed_capabilities(claims: &[ClaimResult]) -> BTreeMap<String, BTreeSet<String>> {
    let mut map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for claim in claims {
        if claim.expected && claim.verdict == Verdict::Confirmed {
            map.entry(claim.agent.to_string())
                .or_default()
                .insert(claim.claim.to_string());
        }
    }
    map
}

/// Per agent, per claim: the one line saying what the probe read. Built from the
/// same results [`confirmed_capabilities`] uses, so a routing reason and
/// `xencode agents --contract` can never disagree about what was seen.
pub fn evidence_by_agent(claims: &[ClaimResult]) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut out: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for claim in claims {
        out.entry(claim.agent.to_string())
            .or_default()
            .insert(claim.claim.to_string(), claim.evidence_line());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_evidence_line_names_the_screen_the_claim_was_read_from() {
        let confirmed = ClaimResult {
            agent: "opencode",
            claim: "stream",
            expected: true,
            found: vec!["--format".to_string()],
            missing: vec![],
            sources: vec!["--help".to_string(), "run --help".to_string()],
            verdict: Verdict::Confirmed,
        };
        let line = confirmed.evidence_line();
        assert!(line.contains("confirmed by `--format`"), "{line}");
        assert!(
            line.contains("read from `opencode --help` and `opencode run --help`"),
            "{line}"
        );

        let absent = ClaimResult {
            agent: "claude",
            claim: "acp",
            expected: false,
            found: vec![],
            missing: vec!["acp".to_string()],
            sources: vec!["--help".to_string()],
            verdict: Verdict::Confirmed,
        };
        let line = absent.evidence_line();
        assert!(line.contains("not advertised"), "{line}");

        let contradicted = ClaimResult {
            agent: "codex",
            claim: "daemon",
            expected: true,
            found: vec![],
            missing: vec!["app-server".to_string()],
            sources: vec!["--help".to_string()],
            verdict: Verdict::Contradicted("missing from help: app-server".to_string()),
        };
        let line = contradicted.evidence_line();
        assert!(line.contains("contradicted, missing from help"), "{line}");

        let untestable = ClaimResult {
            agent: "kilo",
            claim: "(help unreadable)",
            expected: true,
            found: vec![],
            missing: vec![],
            sources: vec![],
            verdict: Verdict::Untestable("`--help` produced nothing usable".to_string()),
        };
        let line = untestable.evidence_line();
        assert!(line.contains("could not be tested here"), "{line}");

        // And the grouping keeps every claim of an agent reachable by name.
        let grouped = evidence_by_agent(&[confirmed, absent, contradicted, untestable]);
        assert_eq!(
            grouped["opencode"]["stream"],
            "confirmed by `--format` (read from `opencode --help` and `opencode run --help`)"
        );
        assert!(grouped["kilo"]["(help unreadable)"].contains("could not be tested"));
        // Every caller quotes a line beside the claim it describes, so the line must
        // not start with the claim's own name or the reader gets `acp: acp:`.
        for (claim, line) in [
            ("stream", grouped["opencode"]["stream"].clone()),
            ("acp", grouped["claude"]["acp"].clone()),
            ("daemon", grouped["codex"]["daemon"].clone()),
        ] {
            assert!(!line.starts_with(&format!("{claim}: ")), "{line}");
        }
    }

    #[test]
    fn a_confirmed_absence_is_not_a_capability() {
        let claims = vec![
            ClaimResult {
                agent: "agy",
                claim: "stream",
                expected: true,
                found: vec!["--output-format".to_string()],
                missing: vec![],
                sources: vec!["--help".to_string()],
                verdict: Verdict::Confirmed,
            },
            ClaimResult {
                agent: "agy",
                claim: "acp",
                expected: false,
                found: vec![],
                missing: vec!["acp".to_string()],
                sources: vec!["--help".to_string()],
                verdict: Verdict::Confirmed,
            },
        ];
        let caps = confirmed_capabilities(&claims);
        assert!(caps["agy"].contains("stream"));
        assert!(
            !caps["agy"].contains("acp"),
            "the probe confirmed that acp is *not* advertised; handing that to a router as an \
             ability would send an acp task to an agent that cannot take it"
        );
        // The evidence map keeps both, because a refusal needs the line saying why.
        let evidence = evidence_by_agent(&claims);
        assert!(evidence["agy"]["acp"].contains("not advertised"));
    }

    #[test]
    fn confirmed_capabilities_only_includes_confirmed_claims() {
        let claims = vec![
            ClaimResult {
                agent: "agent-a",
                claim: "acp",
                expected: true,
                found: vec!["acp".to_string()],
                missing: vec![],
                sources: vec!["--help".to_string()],
                verdict: Verdict::Confirmed,
            },
            ClaimResult {
                agent: "agent-a",
                claim: "daemon",
                expected: true,
                found: vec![],
                missing: vec!["serve".to_string()],
                sources: vec!["--help".to_string()],
                verdict: Verdict::Contradicted("missing from help".to_string()),
            },
            ClaimResult {
                agent: "agent-b",
                claim: "stream",
                expected: true,
                found: vec!["--json".to_string()],
                missing: vec![],
                sources: vec!["--help".to_string()],
                verdict: Verdict::Confirmed,
            },
        ];

        let caps = confirmed_capabilities(&claims);
        assert_eq!(caps.get("agent-a").unwrap().len(), 1);
        assert!(caps.get("agent-a").unwrap().contains("acp"));
        assert!(!caps.get("agent-a").unwrap().contains("daemon"));
        assert!(caps.get("agent-b").unwrap().contains("stream"));
    }

    #[test]
    fn short_tokens_need_word_boundaries() {
        assert!(contains_token("opencode serve starts", "serve"));
        assert!(!contains_token("Manage MCP servers", "serve"));
        assert!(contains_token(
            "start ACP (Agent Client Protocol) server",
            "acp"
        ));
        assert!(!contains_token("escape sequences", "acp"));
        assert!(contains_token("--auto-approve permissions", "--auto"));
        assert!(contains_token("anything at all", "anything at all"));
    }

    #[test]
    fn subcommand_comes_from_the_oneshot_form() {
        let codex = crate::roster::find("codex").unwrap();
        assert_eq!(oneshot_subcommand(codex), Some("exec"));
        let claude = crate::roster::find("claude").unwrap();
        assert_eq!(oneshot_subcommand(claude), None);
    }

    #[test]
    fn a_claim_nobody_searched_is_reported_as_unexamined_not_as_an_absence() {
        // The hole this closes: a denied capability with no registered token used
        // to come back `Confirmed`, and its evidence line said "not advertised —
        // no token for it appeared" about a claim no one had looked at. Five rows
        // of the live report were written that way on 2026-10-08.
        let verdict = verdict_for(false, "acp", &[], &[]);
        assert!(
            matches!(&verdict, Verdict::Untestable(why) if why.contains("nothing was searched")),
            "{verdict:?}"
        );
        let line = ClaimResult {
            agent: "crush",
            claim: "acp",
            expected: false,
            found: vec![],
            missing: vec![],
            sources: vec!["--help".to_string()],
            verdict,
        }
        .evidence_line();
        assert!(
            !line.contains("not advertised"),
            "an unexamined claim was printed as an absence: {line}"
        );
    }

    #[test]
    fn a_searched_absence_names_the_words_it_looked_for() {
        // The opposite case, which *is* a measurement: the token was registered,
        // searched for, and did not appear. The line has to carry the word, so a
        // reader can tell this from the row above without opening the source.
        let searched = vec!["acp".to_string()];
        assert_eq!(
            verdict_for(false, "acp", &[], &searched),
            Verdict::Confirmed
        );
        let line = ClaimResult {
            agent: "claude",
            claim: "acp",
            expected: false,
            found: vec![],
            missing: searched,
            sources: vec!["--help".to_string()],
            verdict: Verdict::Confirmed,
        }
        .evidence_line();
        assert_eq!(
            line,
            "not advertised — `acp` searched for and absent (read from `claude --help`)"
        );
    }

    #[test]
    fn a_word_that_proves_nothing_proves_nothing_in_both_directions() {
        // Registered tokens that turn up under a denied claim: `acp` means one
        // thing, so it contradicts the roster; `crew` is a word a vendor may use
        // for anything, so it settles nothing and is said not to.
        let found = vec!["acp".to_string()];
        assert!(matches!(
            verdict_for(false, "acp", &found, &[]),
            Verdict::Contradicted(why) if why.contains("roster says no")
        ));
        let found = vec!["crew".to_string()];
        assert!(
            matches!(&verdict_for(false, "daemon", &found, &[]), Verdict::Untestable(why)
                if why.contains("carries this claim neither way")),
            "an ambiguous word was read as a contradiction or as an absence"
        );
    }

    #[test]
    fn an_asserted_capability_with_no_evidence_behind_it_is_still_a_contradiction() {
        // The half that has fired before: on 2026-10-02 the probe refused `agy`'s
        // claims because its tokens were not registered. A capability the roster
        // asserts and the probe cannot search for is not a capability.
        assert!(matches!(
            verdict_for(true, "stream", &[], &[]),
            Verdict::Contradicted(why) if why.contains("no evidence tokens")));
    }

    #[test]
    fn every_absence_the_live_probe_confirms_was_searched_for() {
        // The same rule, run against the machines this is checked out on: no row
        // may report a denied capability as an absence without naming a token it
        // looked for. On 2026-10-08 this failed for `codex acp`, `claude acp`,
        // `gemini daemon`, `crush acp` and `crush mcp`.
        for result in probe_contract() {
            if result.expected || result.verdict != Verdict::Confirmed {
                continue;
            }
            assert!(
                !result.missing.is_empty(),
                "{} {} is reported as a confirmed absence with no token searched: {:?}",
                result.agent,
                result.claim,
                result.evidence_line()
            );
        }
    }

    #[test]
    fn the_roster_matches_live_help() {
        // The firewall: every installed agent's claims are re-read from its
        // own `--help` output, and a stale cell fails the build. On a machine
        // with no roster agents installed this passes vacuously — there is
        // nothing to contradict.
        for result in probe_contract() {
            assert!(
                !matches!(result.verdict, Verdict::Contradicted(_)),
                "{} {}: {:?}",
                result.agent,
                result.claim,
                result.verdict
            );
        }
    }

    #[test]
    fn a_token_found_under_a_denied_claim_contradicts_only_for_unambiguous_words() {
        assert!(absence_checkable("acp"));
        assert!(absence_checkable("mcp"));
        assert!(!absence_checkable("daemon"));
        assert!(!absence_checkable("approval"));
        assert!(!absence_checkable("resume"));
        assert!(!absence_checkable("stream"));
    }
}
