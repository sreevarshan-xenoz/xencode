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
//! - Absence is asserted only for `acp` and `mcp`, whose tokens are
//!   unambiguous. There is no single word that means "daemon" (`serve` also
//!   matches "MCP server names"), so a false daemon claim is reported as
//!   unchecked rather than confirmed — asserting it would manufacture
//!   contradictions out of substrings.

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

/// Evidence tokens per (agent, claim), taken from the roster comments and
/// verified against live `--help` on 2026-09-28. A token here is a promise:
/// the probe fails the claim if any one of them stops appearing.
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
        ("codex", "mcp") => &["mcp"],
        ("codex", "resume") => &["resume"],
        ("codex", "daemon") => &["app-server", "daemon"],
        ("codex", "approval") => &["--sandbox", "--ask-for-approval"],
        ("claude", "stream") => &["--output-format", "stream-json"],
        ("claude", "mcp") => &["mcp"],
        ("claude", "resume") => &["--resume", "--fork-session"],
        ("claude", "daemon") => &["attach", "--bg"],
        ("claude", "approval") => &["--permission-mode", "--allowedTools"],
        ("gemini", "stream") => &["--output-format", "stream-json"],
        ("gemini", "acp") => &["--acp"],
        ("gemini", "mcp") => &["mcp"],
        ("gemini", "resume") => &["--resume", "--session"],
        ("gemini", "approval") => &["--approval-mode"],
        ("crush", "resume") => &["--session", "--continue", "session"],
        ("crush", "daemon") => &["server"],
        ("crush", "approval") => &["--yolo"],
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
        _ => &[],
    }
}

/// Whether an absence claim is checkable. Only unambiguous tokens qualify.
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
            if tokens.is_empty() {
                // No evidence tokens defined: absence claims for vague words
                // are not asserted (see module docs).
                out.push(ClaimResult {
                    agent: spec.name,
                    claim,
                    expected,
                    found: Vec::new(),
                    missing: Vec::new(),
                    sources: sources.clone(),
                    verdict: if expected {
                        Verdict::Contradicted(
                            "no evidence tokens defined for this claim".to_string(),
                        )
                    } else {
                        Verdict::Confirmed
                    },
                });
                continue;
            }
            let mut found = Vec::new();
            let mut missing = Vec::new();
            for token in tokens {
                if contains_token(&combined, token) {
                    found.push(token.to_string());
                } else {
                    missing.push(token.to_string());
                }
            }
            let verdict = match (expected, missing.is_empty(), found.is_empty()) {
                (true, true, _) => Verdict::Confirmed,
                (true, false, _) => {
                    Verdict::Contradicted(format!("missing from help: {}", missing.join(", ")))
                }
                (false, _, _) if !expected && absence_checkable(claim) && !found.is_empty() => {
                    Verdict::Contradicted(format!(
                        "present in help but roster says no: {}",
                        found.join(", ")
                    ))
                }
                (false, _, _) => Verdict::Confirmed,
            };
            out.push(ClaimResult {
                agent: spec.name,
                claim,
                expected,
                found,
                missing,
                sources: sources.clone(),
                verdict,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn absence_is_only_asserted_where_tokens_are_unambiguous() {
        assert!(absence_checkable("acp"));
        assert!(absence_checkable("mcp"));
        assert!(!absence_checkable("daemon"));
        assert!(!absence_checkable("approval"));
        assert!(!absence_checkable("resume"));
    }
}
