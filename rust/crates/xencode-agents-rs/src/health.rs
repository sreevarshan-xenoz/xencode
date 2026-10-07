//! Worker health monitoring for external agent runtimes (AR-8).
//!
//! Reports installed, version, authenticated, responsive, and rate-limited state
//! per agent runtime. Read-only by construction: never mutates configuration and
//! never silently attempts automated fixes.
//!
//! When authentication is expired, it is explicitly shown as expired and the
//! only offered next step is the exact command the human runs in their own terminal.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::roster::{which, AgentSpec, ROSTER};

/// Authentication status for an external agent runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthStatus {
    /// Agent is authenticated and valid.
    Authenticated,
    /// Agent has expired authentication credentials or session tokens.
    Expired {
        message: String,
        login_command: String,
    },
    /// No credentials or account found for this agent.
    MissingCredentials {
        message: String,
        login_command: String,
    },
    /// Agent is local or does not require external authentication.
    NotRequired,
    /// Authentication state could not be determined without running a paid task.
    Unknown { reason: String },
}

/// Responsiveness check outcome for an external agent process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Responsiveness {
    /// Agent process responded within timeout.
    Responsive { latency_ms: u64 },
    /// Agent process timed out.
    TimedOut { timeout_ms: u64 },
    /// Agent process exited with an unexpected failure.
    Failed { error: String },
    /// Agent executable is not installed.
    NotInstalled,
}

/// Overall health status for a single external agent worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerHealth {
    /// Canonical agent name.
    pub name: String,
    /// Whether the agent executable is present on the system.
    pub installed: bool,
    /// Resolved executable path on disk.
    pub binary: Option<PathBuf>,
    /// Reported version string.
    pub version: Option<String>,
    /// Authentication state.
    pub authenticated: AuthStatus,
    /// Responsiveness measurement.
    pub responsive: Responsiveness,
    /// Whether the agent indicates active rate limiting or quota exhaustion.
    pub rate_limited: bool,
    /// Human-actionable terminal command to authenticate this agent.
    pub login_command: Option<String>,
}

/// Recommended human terminal command to authenticate an agent.
pub fn default_login_command(agent_name: &str) -> &'static str {
    match agent_name {
        "claude" => "claude login",
        "cursor-agent" => "cursor-agent login",
        "agy" => "agy login",
        "cline" => "cline auth",
        "opencode" => "opencode auth login",
        "kilo" => "kilo auth login",
        "kiro-cli" => "kiro-cli login",
        "gemini" => "gemini login",
        "codex" => "codex login",
        "crush" => "crush",
        _ => "login",
    }
}

/// Check if text signals an expired authentication token or session.
pub fn is_expired_auth(text: &str) -> bool {
    const EXPIRED_MARKERS: &[&str] = &[
        "token expired",
        "token has expired",
        "session expired",
        "session has expired",
        "auth expired",
        "authentication expired",
        "credentials expired",
        "jwt expired",
        "refresh token expired",
        "oauth token expired",
    ];
    let lower = text.to_lowercase();
    EXPIRED_MARKERS.iter().any(|m| lower.contains(m))
}

/// Check if text signals missing authentication credentials.
pub fn is_missing_auth(text: &str) -> bool {
    const MISSING_MARKERS: &[&str] = &[
        "not logged in",
        "please log in",
        "please login",
        "login required",
        "authentication required",
        "no api key",
        "api key not found",
        "unauthorized",
        "no credentials found",
        "no account configured",
    ];
    let lower = text.to_lowercase();
    MISSING_MARKERS.iter().any(|m| lower.contains(m))
}

/// Check if text signals rate limiting or 429 quota exhaustion.
pub fn is_rate_limited(text: &str) -> bool {
    const RATE_MARKERS: &[&str] = &[
        "rate limit",
        "rate limited",
        "rate_limited",
        "429 too many requests",
        "too many requests",
        "quota exceeded",
        "quota limit reached",
        "capacity exceeded",
    ];
    let lower = text.to_lowercase();
    RATE_MARKERS.iter().any(|m| lower.contains(m))
}

/// Inspect authentication files or tokens locally without executing changes.
pub fn inspect_local_auth_files(agent_name: &str) -> Option<AuthStatus> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;

    match agent_name {
        "claude" => {
            let claude_json = home.join(".claude.json");
            let anthropic_dir = home.join(".anthropic");
            let key_env = std::env::var_os("ANTHROPIC_API_KEY")
                .or_else(|| std::env::var_os("CLAUDE_API_KEY"));
            if key_env.is_some() || claude_json.exists() || anthropic_dir.exists() {
                // If file exists, check if contents say expired
                if let Ok(content) = std::fs::read_to_string(&claude_json) {
                    if is_expired_auth(&content) {
                        return Some(AuthStatus::Expired {
                            message: "stored session in ~/.claude.json is expired".to_string(),
                            login_command: default_login_command(agent_name).to_string(),
                        });
                    }
                }
                Some(AuthStatus::Authenticated)
            } else {
                Some(AuthStatus::MissingCredentials {
                    message: "no credentials in ~/.claude.json or ANTHROPIC_API_KEY".to_string(),
                    login_command: default_login_command(agent_name).to_string(),
                })
            }
        }
        "cursor-agent" => {
            let cursor_key = std::env::var_os("CURSOR_API_KEY");
            let cursor_config = home.join(".cursor");
            if cursor_key.is_some() || cursor_config.exists() {
                Some(AuthStatus::Authenticated)
            } else {
                Some(AuthStatus::MissingCredentials {
                    message: "no credentials in ~/.cursor or CURSOR_API_KEY".to_string(),
                    login_command: default_login_command(agent_name).to_string(),
                })
            }
        }
        "agy" => {
            let agy_key = std::env::var_os("AGY_API_KEY");
            let agy_config = home.join(".gemini");
            if agy_key.is_some() || agy_config.exists() {
                Some(AuthStatus::Authenticated)
            } else {
                Some(AuthStatus::MissingCredentials {
                    message: "no credentials in ~/.gemini or AGY_API_KEY".to_string(),
                    login_command: default_login_command(agent_name).to_string(),
                })
            }
        }
        "opencode" => {
            let opencode_dir = home.join(".config/opencode");
            if opencode_dir.exists() {
                Some(AuthStatus::Authenticated)
            } else {
                Some(AuthStatus::MissingCredentials {
                    message: "no configuration in ~/.config/opencode".to_string(),
                    login_command: default_login_command(agent_name).to_string(),
                })
            }
        }
        "kilo" => {
            let kilo_dir = home.join(".kilo");
            if kilo_dir.exists() {
                Some(AuthStatus::Authenticated)
            } else {
                Some(AuthStatus::MissingCredentials {
                    message: "no configuration in ~/.kilo".to_string(),
                    login_command: default_login_command(agent_name).to_string(),
                })
            }
        }
        _ => None,
    }
}

/// Inspect the health of an agent runtime.
pub fn check_worker_health(spec: &AgentSpec, timeout: Duration) -> WorkerHealth {
    let resolved_binary = spec.binaries.iter().find_map(|b| which(b));
    let login_cmd = default_login_command(spec.name).to_string();

    let Some(binary_path) = resolved_binary else {
        return WorkerHealth {
            name: spec.name.to_string(),
            installed: false,
            binary: None,
            version: None,
            authenticated: AuthStatus::Unknown {
                reason: format!("no `{}` binary on PATH", spec.binaries.join("` or `")),
            },
            responsive: Responsiveness::NotInstalled,
            rate_limited: false,
            login_command: Some(login_cmd),
        };
    };

    // Run --version to test responsiveness and grab version string
    let started = Instant::now();
    let mut cmd = std::process::Command::new(&binary_path);
    cmd.arg("--version");
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut output_rate_limited = false;
    let mut output_auth = None;

    let (version, responsive) = match run_bounded_process(&mut cmd, timeout) {
        Ok(output) => {
            let latency_ms = started.elapsed().as_millis() as u64;
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let combined = format!("{stdout} {stderr}");

            if is_rate_limited(&combined) {
                output_rate_limited = true;
            }
            if is_expired_auth(&combined) {
                output_auth = Some(AuthStatus::Expired {
                    message: "session or token expired".to_string(),
                    login_command: login_cmd.clone(),
                });
            }

            let ver = if !stdout.is_empty() {
                stdout.lines().next().map(str::to_string)
            } else if !stderr.is_empty() {
                stderr.lines().next().map(str::to_string)
            } else {
                None
            };

            (ver, Responsiveness::Responsive { latency_ms })
        }
        Err(e) => (
            None,
            if e.kind() == std::io::ErrorKind::TimedOut {
                Responsiveness::TimedOut {
                    timeout_ms: timeout.as_millis() as u64,
                }
            } else {
                Responsiveness::Failed {
                    error: e.to_string(),
                }
            },
        ),
    };

    // Determine auth status from output signals or local state inspection
    let authenticated = if let Some(auth) = output_auth {
        auth
    } else if let Some(local_status) = inspect_local_auth_files(spec.name) {
        local_status
    } else {
        AuthStatus::Unknown {
            reason: "requires active task probe to verify upstream credentials".to_string(),
        }
    };

    WorkerHealth {
        name: spec.name.to_string(),
        installed: true,
        binary: Some(binary_path),
        version,
        authenticated,
        responsive,
        rate_limited: output_rate_limited,
        login_command: Some(login_cmd),
    }
}

/// Check health of all known roster agents.
pub fn check_all_worker_health(timeout: Duration) -> Vec<WorkerHealth> {
    ROSTER
        .iter()
        .map(|spec| check_worker_health(spec, timeout))
        .collect()
}

/// Format worker health entries into an actionable plain-text table.
pub fn format_worker_health_table(healths: &[WorkerHealth]) -> String {
    let mut out = String::new();
    out.push_str("Agent Worker Health:\n\n");

    for w in healths {
        out.push_str(&format!("  Agent: {}\n", w.name));
        out.push_str(&format!(
            "    Installed:    {}\n",
            if w.installed { "yes" } else { "no" }
        ));

        if let Some(ref v) = w.version {
            out.push_str(&format!("    Version:      {}\n", v));
        }
        if let Some(ref p) = w.binary {
            out.push_str(&format!("    Binary:       {}\n", p.display()));
        }

        match &w.responsive {
            Responsiveness::Responsive { latency_ms } => {
                out.push_str(&format!("    Responsive:   yes ({}ms)\n", latency_ms));
            }
            Responsiveness::TimedOut { timeout_ms } => {
                out.push_str(&format!(
                    "    Responsive:   timed out (>{}ms)\n",
                    timeout_ms
                ));
            }
            Responsiveness::Failed { error } => {
                out.push_str(&format!("    Responsive:   no ({})\n", error));
            }
            Responsiveness::NotInstalled => {
                out.push_str("    Responsive:   not installed\n");
            }
        }

        match &w.authenticated {
            AuthStatus::Authenticated => {
                out.push_str("    Auth:         authenticated\n");
            }
            AuthStatus::Expired {
                message,
                login_command,
            } => {
                out.push_str(&format!("    Auth:         EXPIRED ({})\n", message));
                out.push_str(&format!(
                    "    Next step:    Run `{}` in your terminal\n",
                    login_command
                ));
            }
            AuthStatus::MissingCredentials {
                message,
                login_command,
            } => {
                out.push_str(&format!("    Auth:         MISSING ({})\n", message));
                out.push_str(&format!(
                    "    Next step:    Run `{}` in your terminal\n",
                    login_command
                ));
            }
            AuthStatus::NotRequired => {
                out.push_str("    Auth:         not required\n");
            }
            AuthStatus::Unknown { reason } => {
                out.push_str(&format!("    Auth:         unknown ({})\n", reason));
            }
        }

        if w.rate_limited {
            out.push_str("    Rate Limited: YES\n");
        }
        out.push('\n');
    }

    out.push_str("  read-only: worker health checks never silently fix credentials or modify configuration\n");
    out.trim_end().to_string()
}

/// Run a child process with a hard wall-clock timeout.
fn run_bounded_process(
    command: &mut std::process::Command,
    timeout: Duration,
) -> std::io::Result<std::process::Output> {
    let mut child = command.spawn()?;
    let deadline = Instant::now() + timeout;

    loop {
        match child.try_wait()? {
            Some(_) => return child.wait_with_output(),
            None => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "process execution exceeded deadline",
                    ));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_auth_detection_matches_expired_markers() {
        assert!(is_expired_auth("Error: token expired at 2026-10-01"));
        assert!(is_expired_auth(
            "OAuth session expired, please re-authenticate"
        ));
        assert!(is_expired_auth("JWT expired: signature validation failed"));
        assert!(!is_expired_auth("Successfully authenticated"));
    }

    #[test]
    fn missing_auth_detection_matches_missing_markers() {
        assert!(is_missing_auth("Not logged in. Please run login"));
        assert!(is_missing_auth("Error: no API key found"));
        assert!(is_missing_auth("Unauthorized access: login required"));
        assert!(!is_missing_auth("Ready to serve"));
    }

    #[test]
    fn rate_limited_detection_matches_quota_and_429() {
        assert!(is_rate_limited("HTTP 429 Too Many Requests"));
        assert!(is_rate_limited("Rate limit exceeded for organization"));
        assert!(is_rate_limited("Quota limit reached"));
        assert!(!is_rate_limited("Request completed in 200ms"));
    }

    #[test]
    fn expired_auth_is_rendered_with_only_human_terminal_command() {
        let health = WorkerHealth {
            name: "claude".to_string(),
            installed: true,
            binary: Some(PathBuf::from("/usr/bin/claude")),
            version: Some("2.1.289".to_string()),
            authenticated: AuthStatus::Expired {
                message: "session expired on 2026-10-01".to_string(),
                login_command: "claude login".to_string(),
            },
            responsive: Responsiveness::Responsive { latency_ms: 15 },
            rate_limited: false,
            login_command: Some("claude login".to_string()),
        };

        let table = format_worker_health_table(&[health]);
        assert!(table.contains("Auth:         EXPIRED (session expired on 2026-10-01)"));
        assert!(table.contains("Next step:    Run `claude login` in your terminal"));
        assert!(table.contains("read-only: worker health checks never silently fix credentials"));

        // Must not contain any automated fix instructions
        let lower = table.to_lowercase();
        assert!(!lower.contains("auto-fix"));
        assert!(!lower.contains("fixing"));
    }

    #[test]
    fn default_login_commands_exist_for_known_agents() {
        assert_eq!(default_login_command("claude"), "claude login");
        assert_eq!(default_login_command("cursor-agent"), "cursor-agent login");
        assert_eq!(default_login_command("agy"), "agy login");
        assert_eq!(default_login_command("cline"), "cline auth");
        assert_eq!(default_login_command("opencode"), "opencode auth login");
        assert_eq!(default_login_command("kiro-cli"), "kiro-cli login");
    }

    #[test]
    fn check_all_worker_health_runs_over_roster_and_formats_table() {
        let results = check_all_worker_health(Duration::from_millis(500));
        assert!(!results.is_empty(), "must evaluate roster agents");
        for r in &results {
            assert!(!r.name.is_empty());
            assert!(r.login_command.is_some());
        }
        let table = format_worker_health_table(&results);
        assert!(table.contains("Agent Worker Health:"));
        assert!(table.contains("read-only: worker health checks never silently fix credentials"));
    }
}
