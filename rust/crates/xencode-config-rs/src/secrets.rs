//! Where a provider credential is allowed to live, and how it is read.
//!
//! `config.json` holds API keys as plain strings, which is the documented
//! baseline: owner-only permissions, no vault. Two tiers sit on top of it for
//! people who want the secret out of the file entirely.
//!
//! 1. An environment variable named for the provider ([`SecretProvider::env_vars`]).
//!    Nothing is written to disk; the variable is read when the credential is used.
//! 2. A command reference — the stored value is `command:<program> <args>`, and the
//!    program prints the secret on standard output. Only the reference is in the
//!    file, so a `~/.xencode` that is backed up, synced or shared still holds no
//!    secret. This is the shape `pass`, `op`, `secret-tool` and `pinentry` are all
//!    driven with.
//!
//! A Linux desktop keyring (`org.freedesktop.secrets`) is reachable through tier 2
//! — `command:secret-tool lookup …` — and it is worth being exact about what that
//! buys. It protects against the config file being backed up, synced or read by
//! another user. It does **not** protect against a process running as you: the
//! Secret Service answers anything in your own desktop session, and it is not
//! available at all over SSH or headless. A keyring is a different place to keep
//! the same secret, not a stronger lock on it.

use std::fmt;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// What a stored value must start with to name a command rather than hold a secret.
pub const SECRET_COMMAND_PREFIX: &str = "command:";

/// How long a command reference may take before it is given up on. A keyring
/// helper that is waiting for a passphrase nobody can see must not hang the
/// interface that asked for a key.
pub const SECRET_HELPER_TIMEOUT: Duration = Duration::from_secs(10);

/// Whether a stored value names a command to run rather than holding a secret.
/// Anything that displays a credential needs this: the reference is safe to show
/// in full, the key is not.
pub fn is_secret_reference(value: &str) -> bool {
    value.trim().starts_with(SECRET_COMMAND_PREFIX)
}

/// A credential the configuration can point at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretProvider {
    OpenAi,
    OpenRouter,
    Gemini,
    Qwen,
    /// The bring-your-own OpenAI-compatible endpoint (`remote:` models).
    Remote,
    Nvidia,
    /// Key for the Brave Search API. Used by `web_search` and by nothing else:
    /// it is not a model provider, so it never joins a route.
    Brave,
    /// Key for Tavily, the other paid `web_search` backend.
    Tavily,
}

impl SecretProvider {
    /// Every provider that takes a credential: the model services the settings
    /// panels list, plus the two search engines `web_search` asks. `config show`
    /// and `config set` walk this list, which is why a search key appears in them
    /// without needing its own panel.
    pub const ALL: [SecretProvider; 8] = [
        SecretProvider::OpenAi,
        SecretProvider::OpenRouter,
        SecretProvider::Gemini,
        SecretProvider::Qwen,
        SecretProvider::Remote,
        SecretProvider::Nvidia,
        SecretProvider::Brave,
        SecretProvider::Tavily,
    ];

    /// The name used in `config show` and the `doctor` rows.
    pub fn slug(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::OpenRouter => "openrouter",
            Self::Gemini => "gemini",
            Self::Qwen => "qwen",
            Self::Remote => "remote",
            Self::Nvidia => "nvidia",
            Self::Brave => "brave",
            Self::Tavily => "tavily",
        }
    }

    /// Environment variables that can carry this credential, checked in order.
    ///
    /// `XENCODE_API_KEY` fills only [`SecretProvider::Remote`] — the endpoint the
    /// person runs themselves. There is deliberately no variable that applies to
    /// every provider: one name cannot say which account the key belongs to, and
    /// guessing would hand one provider's credential to another provider's
    /// endpoint.
    pub fn env_vars(self) -> &'static [&'static str] {
        match self {
            Self::OpenAi => &["API_KEY_OPENAI"],
            Self::OpenRouter => &["API_KEY_OPENROUTER"],
            Self::Gemini => &["API_KEY_GEMINI"],
            Self::Qwen => &["API_KEY_QWEN"],
            Self::Remote => &["XENCODE_API_KEY", "API_KEY_REMOTE"],
            // `NVIDIA_NIM_API_KEY` is the name this project shipped first, so it
            // keeps working; the pattern-matching alias joins it rather than
            // replacing it.
            Self::Nvidia => &["NVIDIA_NIM_API_KEY", "API_KEY_NVIDIA"],
            Self::Brave => &["API_KEY_BRAVE"],
            Self::Tavily => &["API_KEY_TAVILY"],
        }
    }

    /// The `config set` key under which this credential is stored.
    pub fn config_key(self) -> &'static str {
        match self {
            Self::OpenAi => "openai_api_key",
            Self::OpenRouter => "openrouter_api_key",
            Self::Gemini => "google_gemini_api_key",
            Self::Qwen => "qwen_api_key",
            // Named for the endpoint it dials, not for a vendor; `remote_api_key`
            // is accepted as its alias, since that is the field name in the file.
            Self::Remote => "remote_key",
            Self::Nvidia => "nvidia_api_key",
            Self::Brave => "brave_api_key",
            Self::Tavily => "tavily_api_key",
        }
    }

    /// The field this credential occupies inside the `api_keys` object of
    /// `config.json`.
    pub fn json_field(self) -> &'static str {
        match self {
            Self::OpenAi => "openai_api_key",
            Self::OpenRouter => "openrouter_api_key",
            Self::Gemini => "google_gemini_api_key",
            Self::Qwen => "qwen_api_key",
            Self::Remote => "remote_api_key",
            Self::Nvidia => "nvidia_api_key",
            Self::Brave => "brave_api_key",
            Self::Tavily => "tavily_api_key",
        }
    }

    /// The provider a `config set` key names, or `None` when the key is not a
    /// credential. `remote_key` and `remote_api_key` are the same slot — the
    /// shorter name is what this CLI has always taken, the longer one is what the
    /// file calls it.
    pub fn from_config_key(key: &str) -> Option<SecretProvider> {
        SecretProvider::ALL
            .into_iter()
            .find(|provider| provider.config_key() == key)
            .or_else(|| (key == "remote_api_key").then_some(SecretProvider::Remote))
    }

    /// The value this provider's field holds in `config.json`, if any.
    pub fn stored(self, keys: &crate::config::ApiKeys) -> Option<&str> {
        match self {
            Self::OpenAi => keys.openai_api_key.as_deref(),
            Self::OpenRouter => keys.openrouter_api_key.as_deref(),
            Self::Gemini => keys.google_gemini_api_key.as_deref(),
            Self::Qwen => keys.qwen_api_key.as_deref(),
            Self::Remote => keys.remote_api_key.as_deref(),
            Self::Nvidia => keys.nvidia_api_key.as_deref(),
            Self::Brave => keys.brave_api_key.as_deref(),
            Self::Tavily => keys.tavily_api_key.as_deref(),
        }
    }
}

/// Why a credential could not be read from where the configuration points at it.
///
/// The command is named in every variant because that is the thing the person can
/// act on. The secret is never part of these messages: a helper that echoed the
/// key and then failed would print it, so the captured output is dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretProblem {
    /// The value was `command:` with nothing after it.
    NoCommand,
    /// The program could not be started. It is looked up in `PATH` and run
    /// directly — no shell is involved, so nothing in the reference is expanded.
    NotStarted { command: String, problem: String },
    /// It ran and exited with a failure.
    Failed {
        command: String,
        /// What it exited with, in words the person can act on.
        status: String,
    },
    /// It was still running after [`SECRET_HELPER_TIMEOUT`].
    TimedOut { command: String },
    /// It succeeded and printed nothing usable.
    Empty { command: String },
}

impl fmt::Display for SecretProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCommand => write!(
                f,
                "a key is stored as a command reference but names no command (`command:` with nothing after it)"
            ),
            Self::NotStarted { command, problem } => write!(
                f,
                "the key command `{command}` could not be started: {problem}. It is run directly, not through a shell, so the program has to be on PATH and `*` or `$HOME` in the reference will not be expanded"
            ),
            Self::Failed { command, status } => {
                write!(f, "the key command `{command}` exited {status} and gave no key")
            }
            Self::TimedOut { command } => write!(
                f,
                "the key command `{command}` did not answer within {}s — it is started with no terminal attached, so a passphrase prompt it is waiting on can never be answered",
                SECRET_HELPER_TIMEOUT.as_secs()
            ),
            Self::Empty { command } => write!(
                f,
                "the key command `{command}` ran but printed nothing on its first line"
            ),
        }
    }
}

/// The credential for one provider: what the file holds, plus the variables that
/// can fill it.
///
/// Order: a value in the file wins, because putting it there was an explicit act.
/// A stored value beginning with [`SECRET_COMMAND_PREFIX`] is a command to run,
/// not a secret. An environment variable is used only when the file says nothing.
/// Blank counts as unset on both sides, so an empty export cannot shadow a real
/// configured key with nothing.
pub fn resolve(
    configured: Option<&str>,
    env_vars: &[&'static str],
) -> Result<Option<String>, SecretProblem> {
    if let Some(stored) = non_blank(configured) {
        if let Some(command) = stored.strip_prefix(SECRET_COMMAND_PREFIX) {
            return run_command_reference(command);
        }
        return Ok(Some(stored.to_string()));
    }
    Ok(first_env_value(env_vars))
}

/// Whether a credential exists for this provider, without reading it. Cheap: no
/// command reference is run. A panel that shows "configured" uses this, so it
/// never pays for a keyring lookup per redraw.
pub fn is_present(configured: Option<&str>, env_vars: &[&'static str]) -> bool {
    non_blank(configured).is_some() || first_env_value(env_vars).is_some()
}

/// Where a credential comes from, for display. Never the value.
///
/// A stored command reference is shown as stored, because the reference is not the
/// secret; the file it points at is the thing to edit.
pub fn describe(configured: Option<&str>, env_vars: &[&'static str]) -> Option<String> {
    match non_blank(configured) {
        Some(stored) if stored.starts_with(SECRET_COMMAND_PREFIX) => {
            Some(format!("command reference — {stored}"))
        }
        Some(_) => Some("set in config.json (value not shown)".to_string()),
        None => {
            environment_value_name(env_vars).map(|name| format!("set in the environment as {name}"))
        }
    }
}

/// The trimmed value, or `None` when the field is absent or blank.
fn non_blank(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|found| !found.is_empty())
}

/// The first environment variable of `env_vars` that holds a non-blank value.
fn first_env_value(env_vars: &[&'static str]) -> Option<String> {
    env_vars.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

/// The name of that variable, for [`describe`].
fn environment_value_name(env_vars: &[&'static str]) -> Option<&'static str> {
    env_vars.iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(|_| *name)
    })
}

/// Run the helper directly: no shell, `PATH` lookup by the operating system, and
/// both output pipes captured.
///
/// A script that was written moments ago — a helper being replaced by a package
/// upgrade, a password-store syncing — can still be open for writing when the key
/// is asked for, and the kernel refuses to exec it with `Text file busy` rather
/// than run a half-written file. Trying again shortly is the correct answer to
/// that; anything else is a real failure and is reported at once.
fn spawn_directly(program: &str, args: &[String]) -> Result<std::process::Child, std::io::Error> {
    let attempts = 5;
    let mut last = None;
    for attempt in 0..attempts {
        let mut command = Command::new(program);
        command
            .args(args)
            // Nothing for the helper to read: a program that expects to prompt on
            // a terminal fails at once instead of holding the request open.
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        match command.spawn() {
            Ok(child) => return Ok(child),
            Err(problem)
                if problem.kind() == std::io::ErrorKind::ExecutableFileBusy
                    && attempt + 1 < attempts =>
            {
                std::thread::sleep(Duration::from_millis(20));
                last = Some(problem);
            }
            Err(problem) => return Err(problem),
        }
    }
    Err(last.expect("the last busy error is kept for the final attempt"))
}

/// Run `command` and take its first non-blank line as the credential.
fn run_command_reference(command: &str) -> Result<Option<String>, SecretProblem> {
    let command = command.trim();
    if command.is_empty() {
        return Err(SecretProblem::NoCommand);
    }
    let argv = split_command_line(command);
    let Some(program) = argv.first() else {
        return Err(SecretProblem::NoCommand);
    };
    let mut child =
        spawn_directly(program, &argv[1..]).map_err(|problem| SecretProblem::NotStarted {
            command: command.to_string(),
            problem: problem.to_string(),
        })?;

    let mut stdout = child.stdout.take().expect("stdout is piped");
    let mut stderr = child.stderr.take().expect("stderr is piped");
    let failed = |problem: std::io::Error| SecretProblem::NotStarted {
        command: command.to_string(),
        problem: problem.to_string(),
    };
    // Both pipes are drained on their own threads, whether the helper finishes or
    // is stopped, so a program that writes more than a pipe buffer cannot block on
    // the write while this side waits for it to exit. The scope ends by joining
    // them, including on the paths that give up.
    let outcome = std::thread::scope(
        |scope| -> Result<(Option<std::process::ExitStatus>, Vec<u8>), SecretProblem> {
            let out = scope.spawn(|| {
                let mut buffer = Vec::new();
                let _ = stdout.read_to_end(&mut buffer);
                buffer
            });
            let errors = scope.spawn(|| {
                let mut buffer = Vec::new();
                let _ = stderr.read_to_end(&mut buffer);
                buffer
            });
            let deadline = Instant::now() + SECRET_HELPER_TIMEOUT;
            let mut stopped_by_deadline = false;
            let status = loop {
                match child.try_wait() {
                    Ok(Some(found)) => break Some(found),
                    Ok(None) if Instant::now() >= deadline => {
                        let _ = child.kill();
                        let _ = child.wait();
                        stopped_by_deadline = true;
                        break None;
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                    Err(problem) => return Err(failed(problem)),
                }
            };
            let output = out.join().unwrap_or_default();
            // Read and then dropped: what the helper complained about is not shown,
            // because a program that prints the credential on its way out would
            // otherwise put it on the terminal.
            drop(errors.join().unwrap_or_default());
            if stopped_by_deadline {
                return Err(SecretProblem::TimedOut {
                    command: command.to_string(),
                });
            }
            Ok((status, output))
        },
    )?;

    let (status, output) = match outcome {
        (Some(status), output) => (status, output),
        (None, _) => {
            return Err(SecretProblem::TimedOut {
                command: command.to_string(),
            })
        }
    };
    if !status.success() {
        return Err(SecretProblem::Failed {
            command: command.to_string(),
            status: exit_status_of(&status),
        });
    }
    let first_line = String::from_utf8_lossy(&output)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string);
    match first_line {
        Some(secret) => Ok(Some(secret)),
        // The helper's own complaint is dropped rather than surfaced: some tools
        // print the value they were asked for on their way out, and a credential
        // has no business travelling into a terminal scrollback.
        None => Err(SecretProblem::Empty {
            command: command.to_string(),
        }),
    }
}

/// How a helper ended, in the words a person acts on.
#[cfg(unix)]
fn exit_status_of(status: &std::process::ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(code), _) => code.to_string(),
        (None, Some(signal)) => format!("by signal {signal}"),
        (None, None) => "unusually".to_string(),
    }
}

#[cfg(not(unix))]
fn exit_status_of(status: &std::process::ExitStatus) -> String {
    status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "unusually".to_string())
}

/// Split a command line into program and arguments.
///
/// Quotes group, and there is no shell behind this: `*`, `$HOME` and `;` reach the
/// helper as the characters they are. That is the point — a reference somebody can
/// put into the file becomes a command that takes fixed arguments, not a line of
/// shell.
fn split_command_line(line: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut quoted = false;
    for ch in line.chars() {
        match quote {
            Some(mark) if ch == mark => quote = None,
            Some(_) => current.push(ch),
            None if ch == '\'' || ch == '"' => {
                quote = Some(ch);
                quoted = true;
            }
            None if ch.is_whitespace() => {
                if quoted || !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                    quoted = false;
                }
            }
            None => current.push(ch),
        }
    }
    if quoted || !current.is_empty() {
        words.push(current);
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Mutex, MutexGuard};

    /// Tests that set an environment variable must not overlap: the variable is
    /// process-global, and two tests swapping it read each other's values.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Each test running a helper gets its own directory, so a script written by
    /// one cannot be picked up by another.
    fn temp_dir() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-secret-test-{unique}"))
    }

    /// Write a real executable and return the `command:` reference naming it.
    /// Nothing here is simulated: the helper is a script on disk that the resolver
    /// forks and runs.
    fn helper(dir: &PathBuf, name: &str, body: &str) -> String {
        let path = write_helper(dir, name, body);
        format!("{SECRET_COMMAND_PREFIX}{}", path.display())
    }

    fn write_helper(dir: &PathBuf, name: &str, body: &str) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        fs::write(&path, body).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        }
        path
    }

    #[test]
    // The helper is a `#!/bin/sh` script, which only Unix can execute.
    #[cfg(unix)]
    fn a_stored_key_wins_over_the_environment() {
        let _guard = lock_env();
        let dir = temp_dir();
        let reference = helper(&dir, "prints-a-key", "#!/bin/sh\necho helper-key\n");
        let previous = std::env::var_os("XCODE_SECRET_TEST_KEY");
        std::env::set_var("XCODE_SECRET_TEST_KEY", "env-key");
        assert_eq!(
            resolve(Some(&reference), &["XCODE_SECRET_TEST_KEY"])
                .unwrap()
                .as_deref(),
            Some("helper-key"),
            "the file named a command, so the command answers"
        );
        assert_eq!(
            resolve(Some("  stored-key  "), &["XCODE_SECRET_TEST_KEY"])
                .unwrap()
                .as_deref(),
            Some("stored-key"),
            "a plain stored value is trimmed and used as it stands"
        );
        restore_env("XCODE_SECRET_TEST_KEY", previous);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_environment_variable_fills_a_provider_that_stored_nothing() {
        let _guard = lock_env();
        let previous = std::env::var_os("XCODE_SECRET_TEST_KEY");
        std::env::set_var("XCODE_SECRET_TEST_KEY", "  env-key  ");
        assert_eq!(
            resolve(None, &["XCODE_SECRET_TEST_KEY"])
                .unwrap()
                .as_deref(),
            Some("env-key"),
            "blank around the value is not part of the key"
        );
        // Blank counts as unset, so an empty export cannot hide a real key.
        std::env::set_var("XCODE_SECRET_TEST_KEY", "   ");
        assert_eq!(
            resolve(None, &["XCODE_SECRET_TEST_KEY"]).unwrap(),
            None,
            "an empty variable is not a credential"
        );
        restore_env("XCODE_SECRET_TEST_KEY", previous);
    }

    #[test]
    fn the_first_variable_that_holds_a_value_wins() {
        let _guard = lock_env();
        let previous_a = std::env::var_os("XCODE_SECRET_TEST_A");
        let previous_b = std::env::var_os("XCODE_SECRET_TEST_B");
        std::env::remove_var("XCODE_SECRET_TEST_A");
        std::env::set_var("XCODE_SECRET_TEST_B", "second");
        assert_eq!(
            resolve(None, &["XCODE_SECRET_TEST_A", "XCODE_SECRET_TEST_B"])
                .unwrap()
                .as_deref(),
            Some("second")
        );
        std::env::set_var("XCODE_SECRET_TEST_A", "first");
        assert_eq!(
            resolve(None, &["XCODE_SECRET_TEST_A", "XCODE_SECRET_TEST_B"])
                .unwrap()
                .as_deref(),
            Some("first"),
            "the order the provider declares is the order checked"
        );
        restore_env("XCODE_SECRET_TEST_A", previous_a);
        restore_env("XCODE_SECRET_TEST_B", previous_b);
    }

    #[test]
    // The helper is a `#!/bin/sh` script, which only Unix can execute.
    #[cfg(unix)]
    fn a_command_reference_is_run_and_its_first_line_is_the_key() {
        let dir = temp_dir();
        let reference = helper(
            &dir,
            "multiline",
            "#!/bin/sh\nprintf 'key-from-helper\\ntrailing noise on stdout\\n'\n",
        );
        assert_eq!(
            resolve(Some(&reference), &[]).unwrap().as_deref(),
            Some("key-from-helper"),
            "only the first non-blank line is taken"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    // The helper is a `#!/bin/sh` script, which only Unix can execute.
    #[cfg(unix)]
    fn a_failing_helper_says_which_command_and_keeps_its_own_output_back() {
        let dir = temp_dir();
        let reference = helper(
            &dir,
            "fails",
            "#!/bin/sh\necho 'sk-leaked-value' >&2\nexit 3\n",
        );
        let problem = resolve(Some(&reference), &[]).unwrap_err();
        assert!(
            matches!(problem, SecretProblem::Failed { .. }),
            "a nonzero exit is a failure, not an empty key: {problem}"
        );
        let words = problem.to_string();
        assert!(words.contains("fails"), "names the command: {words}");
        assert!(words.contains("3"), "and says how it ended: {words}");
        assert!(
            !words.contains("sk-leaked-value"),
            "the helper's stderr is never carried into the message: {words}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    // The helper is a `#!/bin/sh` script, which only Unix can execute.
    #[cfg(unix)]
    fn a_helper_that_prints_nothing_is_reported_not_treated_as_no_key() {
        let dir = temp_dir();
        let reference = helper(&dir, "silent", "#!/bin/sh\nexit 0\n");
        let problem = resolve(Some(&reference), &[]).unwrap_err();
        assert!(
            matches!(problem, SecretProblem::Empty { .. }),
            "silence is a broken helper: {problem}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    // The helper is a `#!/bin/sh` script, which only Unix can execute.
    #[cfg(unix)]
    fn a_helper_that_never_answers_is_stopped_and_said_so() {
        let dir = temp_dir();
        // `sleep 30` is longer than the shipped ten-second deadline: the test
        // proves the deadline is what ends the wait, not the helper.
        let reference = helper(&dir, "hangs", "#!/bin/sh\nexec sleep 30\n");
        let started = Instant::now();
        let problem = resolve(Some(&reference), &[])
            .expect_err("the helper never answers, so this must not be Ok");
        let elapsed = started.elapsed();
        assert!(
            matches!(problem, SecretProblem::TimedOut { .. }),
            "a helper that does not answer must not be waited on forever: {problem}"
        );
        assert!(
            elapsed >= SECRET_HELPER_TIMEOUT,
            "the deadline is {}s, and it gave up after {}s",
            SECRET_HELPER_TIMEOUT.as_secs(),
            elapsed.as_secs()
        );
        assert!(
            elapsed < SECRET_HELPER_TIMEOUT + Duration::from_secs(5),
            "gave up at the deadline rather than waiting out the helper's own 30s: {}s",
            elapsed.as_secs()
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_program_is_named_in_the_message_and_no_shell_is_used() {
        let problem = resolve(Some("command:/definitely/not/here print"), &[]).unwrap_err();
        assert!(
            matches!(problem, SecretProblem::NotStarted { .. }),
            "{problem}"
        );
        assert!(
            problem.to_string().contains("not through a shell"),
            "the message has to say why `*` and `$HOME` will not expand: {problem}"
        );
        assert_eq!(
            resolve(Some("command:"), &[]).unwrap_err(),
            SecretProblem::NoCommand,
            "`command:` alone names nothing"
        );
    }

    #[test]
    fn quotes_group_and_shell_metacharacters_do_not() {
        assert_eq!(
            split_command_line("secret-tool lookup service xencode"),
            vec!["secret-tool", "lookup", "service", "xencode"]
        );
        assert_eq!(
            split_command_line("pass show \"api keys/openrouter\""),
            vec!["pass", "show", "api keys/openrouter"]
        );
        assert_eq!(
            split_command_line("op read 'op://vault/item/field'"),
            vec!["op", "read", "op://vault/item/field"]
        );
        // A `$HOME` or `*` reaches the helper as written, because nothing expands
        // it. That is the security property, so it gets a test.
        assert_eq!(
            split_command_line("helper --dir $HOME --glob '*'"),
            vec!["helper", "--dir", "$HOME", "--glob", "*"]
        );
        assert!(split_command_line("   ").is_empty());
    }

    #[test]
    // The helper is a `#!/bin/sh` script, which only Unix can execute.
    #[cfg(unix)]
    fn a_quoted_argument_reaches_the_helper_as_one_word() {
        let dir = temp_dir();
        // Prints its first two arguments, so an argument that was split wrongly is
        // visible in the value that comes back.
        let path = write_helper(&dir, "shows-args", "#!/bin/sh\necho \"one=$1 two=$2\"\n");
        let reference = format!("{SECRET_COMMAND_PREFIX}{} 'a b' c", path.display());
        assert_eq!(
            resolve(Some(&reference), &[]).unwrap().as_deref(),
            Some("one=a b two=c"),
            "the quoted space stayed inside one argument"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn presence_is_answered_without_running_the_helper() {
        let dir = temp_dir();
        let marker = dir.join("was-run");
        let reference = helper(
            &dir,
            "marks",
            &format!("#!/bin/sh\ntouch {}\necho key\n", marker.display()),
        );
        assert!(
            is_present(Some(&reference), &[]),
            "a stored reference counts as a credential"
        );
        assert!(
            !marker.exists(),
            "asking whether a key exists must not look it up"
        );
        assert!(!is_present(Some("   "), &[]), "blank is not a credential");
        assert!(is_present(Some("literal"), &[]));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_displayed_form_names_the_tier_and_never_the_value() {
        let _guard = lock_env();
        let dir = temp_dir();
        assert_eq!(
            describe(Some("sk-secret-looking-value"), &["API_KEY_OPENAI"]).as_deref(),
            Some("set in config.json (value not shown)")
        );
        assert_eq!(
            describe(None, &["XCODE_SECRET_TEST_ABSENT"]),
            None,
            "nothing stored and nothing exported says nothing"
        );
        let previous = std::env::var_os("XCODE_SECRET_TEST_KEY");
        std::env::set_var("XCODE_SECRET_TEST_KEY", "env-value-nobody-sees");
        assert_eq!(
            describe(None, &["XCODE_SECRET_TEST_KEY"]).as_deref(),
            Some("set in the environment as XCODE_SECRET_TEST_KEY"),
            "the variable is named, its value is not"
        );
        restore_env("XCODE_SECRET_TEST_KEY", previous);
        let reference = helper(&dir, "shows", "#!/bin/sh\necho key\n");
        let shown = describe(Some(&reference), &[]).unwrap();
        assert!(
            shown.starts_with("command reference — command:"),
            "the reference is shown as stored: {shown}"
        );
        assert!(
            !shown.contains("sk-secret-looking-value") && !shown.contains("env-value-nobody-sees")
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_provider_has_a_variable_named_for_it() {
        for provider in SecretProvider::ALL {
            assert!(
                !provider.env_vars().is_empty(),
                "{} carries no environment tier",
                provider.slug()
            );
            for name in provider.env_vars() {
                assert!(
                    name.starts_with("API_KEY_")
                        || name.starts_with("XENCODE_")
                        || name.starts_with("NVIDIA_"),
                    "{name} does not follow the naming this project documents"
                );
            }
            assert!(
                provider.config_key().ends_with("_key"),
                "{} stores its credential under {}",
                provider.slug(),
                provider.config_key()
            );
        }
        // The endpoint the person runs themselves is the only one a generic
        // variable may fill.
        assert_eq!(
            SecretProvider::Remote.env_vars()[0],
            "XENCODE_API_KEY",
            "XENCODE_API_KEY must fill the bring-your-own endpoint"
        );
        for provider in SecretProvider::ALL {
            if provider != SecretProvider::Remote {
                assert!(
                    !provider.env_vars().contains(&"XENCODE_API_KEY"),
                    "{} would read a variable that names no provider",
                    provider.slug()
                );
            }
        }
    }

    #[test]
    fn the_command_prefix_is_the_documented_spelling() {
        assert_eq!(SECRET_COMMAND_PREFIX, "command:");
    }

    fn restore_env(name: &str, previous: Option<std::ffi::OsString>) {
        match previous {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}
