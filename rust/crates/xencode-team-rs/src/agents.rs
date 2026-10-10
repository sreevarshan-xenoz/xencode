//! The outside worker agents (TM-5): how each one starts over the Agent
//! Client Protocol, how it signs in, and whether this machine can start it
//! now. The commands and versions are the ACP registry's, pinned; changing
//! one is a code change.

use std::path::{Path, PathBuf};

/// One outside agent xencode can start as a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentSpec {
    /// The name the lead uses in `team_start`.
    pub name: &'static str,
    /// The program, looked up on PATH.
    pub program: &'static str,
    pub args: &'static [&'static str],
    /// Environment variables that hold an API key for it, in the order they
    /// are looked at.
    pub key_vars: &'static [&'static str],
    /// The xencode setting that can hold the same key (`xencode config set`).
    pub stored_key: Option<StoredKey>,
    /// Environment variables its own login is kept in, which only it sees.
    pub login_vars: &'static [&'static str],
    /// How to install the program when it is missing.
    pub install: &'static str,
    /// What the vendor's terms say about using a person's own login through
    /// another program, shown before a login is turned on.
    pub terms_line: &'static str,
    pub terms_url: &'static str,
}

/// A key xencode's own settings can hold for an outside agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredKey {
    OpenAi,
    Gemini,
}

pub const CLAUDE_CODE: AgentSpec = AgentSpec {
    name: "claude-code",
    program: "npx",
    args: &["-y", "@agentclientprotocol/claude-agent-acp@0.89.1"],
    key_vars: &["ANTHROPIC_API_KEY"],
    stored_key: None,
    login_vars: &["CLAUDE_CODE_OAUTH_TOKEN"],
    install: "install Node.js (it brings `npx`): https://nodejs.org",
    terms_line: "Anthropic does not permit third-party developers to route requests through \
                 Free, Pro, or Max plan credentials on behalf of their users.",
    terms_url: "https://code.claude.com/docs/en/legal-and-compliance",
};

pub const CODEX: AgentSpec = AgentSpec {
    name: "codex",
    program: "npx",
    args: &["-y", "@agentclientprotocol/codex-acp@2.2.2"],
    key_vars: &["CODEX_API_KEY", "OPENAI_API_KEY"],
    stored_key: Some(StoredKey::OpenAi),
    login_vars: &[],
    install: "install Node.js (it brings `npx`): https://nodejs.org",
    terms_line: "No term found restricting a ChatGPT login used through this adapter; \
                 OpenAI's terms could not be read in full when this was written.",
    terms_url: "https://openai.com/policies/terms-of-use",
};

pub const GEMINI: AgentSpec = AgentSpec {
    name: "gemini",
    program: "gemini",
    args: &["--acp"],
    key_vars: &["GEMINI_API_KEY"],
    stored_key: Some(StoredKey::Gemini),
    login_vars: &[],
    install: "npm install -g @google/gemini-cli",
    terms_line: "Directly accessing the services powering Gemini CLI using third-party \
                 software is a violation of Google's terms.",
    terms_url: "https://geminicli.com/docs/resources/tos-privacy",
};

pub const ANTIGRAVITY: AgentSpec = AgentSpec {
    name: "antigravity",
    program: "agy_acp_server",
    args: &[],
    key_vars: &["GEMINI_API_KEY"],
    stored_key: Some(StoredKey::Gemini),
    login_vars: &[],
    install: "install Google Antigravity, which puts `agy_acp_server` on PATH",
    terms_line: "Using third party software to access the Service (e.g. using OpenClaw \
                 with Antigravity OAuth) can get the account suspended.",
    terms_url: "https://antigravity.google/terms",
};

/// Every environment variable a vendor's key or login is kept in. An outside
/// worker starts with all of them removed but its own.
pub const VENDOR_KEY_VARS: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CODEX_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
];

/// The variables to remove for a worker of `spec` signing in with
/// `sign_in`: every vendor's and every one in `also`, but the one it uses.
pub fn hidden_vars(spec: &AgentSpec, sign_in: &SignIn, also: &[&str]) -> Vec<String> {
    let keep: Vec<&str> = match sign_in {
        SignIn::Key { var } => vec![var.as_str()],
        SignIn::Login => spec.login_vars.to_vec(),
    };
    let mut out: Vec<String> = VENDOR_KEY_VARS
        .iter()
        .chain(also)
        .filter(|v| !keep.contains(v))
        .map(|v| v.to_string())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Every outside agent, in the order `team_agents` lists them.
pub const AGENTS: &[AgentSpec] = &[CLAUDE_CODE, CODEX, GEMINI, ANTIGRAVITY];

/// The outside agent called `name`.
pub fn find(name: &str) -> Option<&'static AgentSpec> {
    AGENTS.iter().find(|a| a.name == name)
}

/// How a worker signs in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignIn {
    /// With the API key in this variable.
    Key { var: String },
    /// With the person's own login, which they turned on for this agent.
    Login,
}

/// Whether an outside agent can start here, now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    Ready { program: PathBuf, sign_in: SignIn },
    Missing { install: String },
    NoSignIn { fix: String },
}

/// Whether `spec` can start: its program on PATH, and a key (`key_var`, the
/// variable a key was found under) or a login the person turned on.
/// `antigravity_settings` is that agent's settings file, which must choose
/// Gemini for a key to be used.
pub fn availability(
    spec: &AgentSpec,
    program: Option<PathBuf>,
    key_var: Option<&str>,
    opted_in: bool,
    antigravity_settings: Option<&Path>,
) -> Availability {
    let Some(program) = program else {
        return Availability::Missing {
            install: format!("`{}` is not on PATH; {}", spec.program, spec.install),
        };
    };
    let login = || Availability::Ready {
        program: program.clone(),
        sign_in: SignIn::Login,
    };
    match key_var {
        Some(var) if spec.name == ANTIGRAVITY.name && !uses_gemini(antigravity_settings) => {
            if opted_in {
                login()
            } else {
                Availability::NoSignIn {
                    fix: format!(
                        "{var} is set, but Antigravity uses a key only when its settings \
                         ({}) say \"modelProvider\": \"gemini\"",
                        antigravity_settings
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "~/.gemini/antigravity-cli/settings.json".into())
                    ),
                }
            }
        }
        Some(var) => Availability::Ready {
            program,
            sign_in: SignIn::Key {
                var: var.to_string(),
            },
        },
        None if opted_in => login(),
        None => Availability::NoSignIn {
            fix: format!(
                "set {}{}, or run `xencode team login-optin {}` to use your own login",
                spec.key_vars.join(" or "),
                match spec.stored_key {
                    Some(StoredKey::OpenAi) => " (or `xencode config set openai_api_key …`)",
                    Some(StoredKey::Gemini) => " (or `xencode config set google_gemini_api_key …`)",
                    None => "",
                },
                spec.name
            ),
        },
    }
}

/// Whether Antigravity's settings choose Gemini as the model provider.
fn uses_gemini(settings: Option<&Path>) -> bool {
    settings
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|v| {
            v.get("modelProvider")
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .is_some_and(|m| m == "gemini")
}

/// Antigravity's settings file in `home`.
pub fn antigravity_settings(home: &Path) -> PathBuf {
    home.join(".gemini")
        .join("antigravity-cli")
        .join("settings.json")
}

/// Find `program` on `path` (a PATH value), the way a shell would: on
/// Windows also as `.exe`, `.cmd` and `.bat`, which is how npm installs
/// `npx` and `gemini` there.
pub fn find_program(program: &str, path: &std::ffi::OsStr) -> Option<PathBuf> {
    let names: Vec<String> = if cfg!(windows) {
        ["exe", "cmd", "bat"]
            .iter()
            .map(|ext| format!("{program}.{ext}"))
            .collect()
    } else {
        vec![program.to_string()]
    };
    std::env::split_paths(path)
        .filter(|dir| dir.is_absolute())
        .flat_map(|dir| names.iter().map(move |n| dir.join(n)).collect::<Vec<_>>())
        .find(|candidate| candidate.is_file())
}

/// The agents the person turned a login on for, kept in `team-optins.json`
/// in xencode's settings folder.
pub fn optins(settings_dir: &Path) -> Vec<String> {
    std::fs::read_to_string(settings_dir.join(OPTINS_FILE))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// Turn a login on for `agent`.
pub fn opt_in(settings_dir: &Path, agent: &str) -> Result<(), String> {
    let mut names = optins(settings_dir);
    if !names.iter().any(|n| n == agent) {
        names.push(agent.to_string());
    }
    std::fs::create_dir_all(settings_dir).map_err(|e| e.to_string())?;
    let text = serde_json::to_string_pretty(&names).map_err(|e| e.to_string())?;
    std::fs::write(settings_dir.join(OPTINS_FILE), text)
        .map_err(|e| format!("could not write {}: {e}", OPTINS_FILE))
}

pub const OPTINS_FILE: &str = "team-optins.json";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_program_that_is_not_on_path_is_missing_with_the_install_line() {
        let empty = tempfile::tempdir().unwrap();
        let found = find_program("gemini", empty.path().as_os_str());
        assert_eq!(found, None);
        match availability(&GEMINI, found, Some("GEMINI_API_KEY"), false, None) {
            Availability::Missing { install } => {
                assert!(
                    install.contains("npm install -g @google/gemini-cli"),
                    "{install}"
                )
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_program_on_path_with_a_key_is_ready_to_sign_in_with_the_key() {
        let dir = tempfile::tempdir().unwrap();
        let name = if cfg!(windows) {
            "gemini.cmd"
        } else {
            "gemini"
        };
        std::fs::write(dir.path().join(name), "").unwrap();
        let found = find_program("gemini", dir.path().as_os_str());
        assert_eq!(found.as_deref(), Some(dir.path().join(name).as_path()));
        assert_eq!(
            availability(&GEMINI, found.clone(), Some("GEMINI_API_KEY"), false, None),
            Availability::Ready {
                program: found.unwrap(),
                sign_in: SignIn::Key {
                    var: "GEMINI_API_KEY".into()
                }
            }
        );
    }

    #[test]
    fn without_a_key_a_login_is_used_only_once_turned_on() {
        let program = Some(PathBuf::from("/x/codex"));
        match availability(&CODEX, program.clone(), None, false, None) {
            Availability::NoSignIn { fix } => {
                assert!(fix.contains("CODEX_API_KEY or OPENAI_API_KEY"), "{fix}");
                assert!(fix.contains("xencode team login-optin codex"), "{fix}");
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            availability(&CODEX, program, None, true, None),
            Availability::Ready {
                sign_in: SignIn::Login,
                ..
            }
        ));
    }

    #[test]
    fn antigravity_uses_a_key_only_when_its_settings_choose_gemini() {
        let home = tempfile::tempdir().unwrap();
        let settings = antigravity_settings(home.path());
        let program = Some(PathBuf::from("/x/agy_acp_server"));
        let key = Some("GEMINI_API_KEY");
        assert!(matches!(
            availability(&ANTIGRAVITY, program.clone(), key, false, Some(&settings)),
            Availability::NoSignIn { .. }
        ));
        std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
        std::fs::write(&settings, r#"{"modelProvider": "gemini"}"#).unwrap();
        assert!(matches!(
            availability(&ANTIGRAVITY, program, key, false, Some(&settings)),
            Availability::Ready {
                sign_in: SignIn::Key { .. },
                ..
            }
        ));
    }

    #[test]
    fn a_login_turned_on_is_remembered_once() {
        let dir = tempfile::tempdir().unwrap();
        assert!(optins(dir.path()).is_empty());
        opt_in(dir.path(), "gemini").unwrap();
        opt_in(dir.path(), "gemini").unwrap();
        assert_eq!(optins(dir.path()), vec!["gemini".to_string()]);
    }

    /// Security review: a worker sees its own key and no other vendor's.
    #[test]
    fn a_worker_keeps_only_its_own_key() {
        let hidden = hidden_vars(
            &GEMINI,
            &SignIn::Key {
                var: "GEMINI_API_KEY".into(),
            },
            &["API_KEY_OPENAI"],
        );
        assert!(
            !hidden.contains(&"GEMINI_API_KEY".to_string()),
            "{hidden:?}"
        );
        for var in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "API_KEY_OPENAI",
        ] {
            assert!(hidden.contains(&var.to_string()), "{var}: {hidden:?}");
        }
        let login = hidden_vars(&CLAUDE_CODE, &SignIn::Login, &[]);
        assert!(
            !login.contains(&"CLAUDE_CODE_OAUTH_TOKEN".to_string()),
            "{login:?}"
        );
        assert!(
            login.contains(&"ANTHROPIC_API_KEY".to_string()),
            "{login:?}"
        );
    }

    #[test]
    fn the_adapters_are_the_pinned_registry_versions() {
        assert!(CLAUDE_CODE
            .args
            .contains(&"@agentclientprotocol/claude-agent-acp@0.89.1"));
        assert!(CODEX.args.contains(&"@agentclientprotocol/codex-acp@2.2.2"));
        assert_eq!(GEMINI.args, &["--acp"]);
        assert_eq!(find("antigravity"), Some(&ANTIGRAVITY));
        assert_eq!(find("xencode"), None, "xencode is not an outside agent");
    }
}
