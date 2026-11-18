//! Local text-to-speech output: speak text through a real program installed on
//! this machine. This is output only — the product has no voice input.
//!
//! The program is resolved from [`TTS_CMD_ENV`] when it is set, otherwise from
//! a per-platform default (`espeak-ng` on Linux, `say` on macOS, and a fixed
//! PowerShell argument list using `System.Speech` on Windows). The override is
//! split on whitespace into a fixed argument vector; it is never handed to a
//! shell. The text to speak is always appended as its own argument, so text
//! containing spaces, quotes or shell syntax reaches the program literally.

use std::env;
use std::fmt;

/// Environment variable naming the TTS program and any fixed leading
/// arguments, for example `XENCODE_TTS_CMD="espeak-ng -s 150"`.
pub const TTS_CMD_ENV: &str = "XENCODE_TTS_CMD";

/// Why speaking failed. Each variant names the program that was tried.
#[derive(Debug)]
pub enum SpeakError {
    /// The resolved program does not exist on this machine.
    MissingBinary { program: String },
    /// The program exists but could not be started.
    SpawnFailed { program: String, detail: String },
    /// The program ran but reported failure through its exit status.
    ExitFailed {
        program: String,
        status: String,
        detail: String,
    },
}

impl fmt::Display for SpeakError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBinary { program } => write!(
                f,
                "text-to-speech command '{program}' was not found; install espeak-ng on Linux, \
                 say on macOS, or enable System.Speech on Windows, or set {TTS_CMD_ENV} to a \
                 local TTS binary"
            ),
            Self::SpawnFailed { program, detail } => {
                write!(
                    f,
                    "could not start text-to-speech command '{program}': {detail}"
                )
            }
            Self::ExitFailed {
                program,
                status,
                detail,
            } => {
                if detail.is_empty() {
                    write!(f, "text-to-speech command '{program}' exited with {status}")
                } else {
                    write!(
                        f,
                        "text-to-speech command '{program}' exited with {status}: {detail}"
                    )
                }
            }
        }
    }
}

impl std::error::Error for SpeakError {}

/// A program plus the fixed arguments that go before the spoken text.
struct ResolvedCommand {
    program: String,
    args: Vec<String>,
}

/// Resolve the program from the environment override, falling back to the
/// platform default when the override is absent or empty.
fn resolve_command(override_value: Option<&str>) -> ResolvedCommand {
    if let Some(value) = override_value.map(str::trim).filter(|s| !s.is_empty()) {
        let mut tokens = value.split_whitespace().map(str::to_string);
        if let Some(program) = tokens.next() {
            return ResolvedCommand {
                program,
                args: tokens.collect(),
            };
        }
    }
    default_command()
}

fn default_command() -> ResolvedCommand {
    #[cfg(target_os = "windows")]
    {
        ResolvedCommand {
            program: "powershell".to_string(),
            args: vec![
                "-NoProfile".to_string(),
                "-Command".to_string(),
                "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($args[0])".to_string(),
            ],
        }
    }
    #[cfg(target_os = "macos")]
    {
        ResolvedCommand {
            program: "say".to_string(),
            args: Vec::new(),
        }
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        ResolvedCommand {
            program: "espeak-ng".to_string(),
            args: Vec::new(),
        }
    }
}

/// Speak `text` through the configured local TTS program. A missing program is
/// reported with its name; a nonzero exit is reported with its status and
/// output. No shell is involved at any point.
pub fn speak(text: &str) -> Result<(), SpeakError> {
    let override_value = env::var(TTS_CMD_ENV).ok();
    speak_with(text, override_value.as_deref())
}

fn speak_with(text: &str, override_value: Option<&str>) -> Result<(), SpeakError> {
    let command = resolve_command(override_value);
    let output = std::process::Command::new(&command.program)
        .args(&command.args)
        .arg(text)
        .output()
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => SpeakError::MissingBinary {
                program: command.program.clone(),
            },
            _ => SpeakError::SpawnFailed {
                program: command.program.clone(),
                detail: error.to_string(),
            },
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Err(SpeakError::ExitFailed {
        program: command.program,
        status: output.status.to_string(),
        detail: [stderr, stdout]
            .into_iter()
            .find(|part| !part.is_empty())
            .unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The override is process-wide, so the tests that change it must not run
    // at the same time. A guard restores the previous value even on panic.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard(Option<String>);

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.0 {
                Some(previous) => env::set_var(TTS_CMD_ENV, previous),
                None => env::remove_var(TTS_CMD_ENV),
            }
        }
    }

    fn set_tts_cmd(value: &str) -> EnvGuard {
        let previous = env::var(TTS_CMD_ENV).ok();
        env::set_var(TTS_CMD_ENV, value);
        EnvGuard(previous)
    }

    #[cfg(unix)]
    #[test]
    fn a_real_script_receives_the_exact_text_and_exits_zero() {
        let _lock = ENV_LOCK.lock().expect("environment lock");
        let dir = tempfile::tempdir().expect("temporary directory");
        let script = dir.path().join("tts.sh");
        let captured = dir.path().join("argv.txt");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" >> '{}'\n",
                captured.display()
            ),
        )
        .expect("write the real executable script");
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                .expect("make the script executable");
        }

        let text = "hello from xencode; $(echo SHOULD_NOT_RUN) 'quoted' & more";
        let _guard = set_tts_cmd(script.to_str().expect("utf8 path"));
        speak(text).expect("the real script exits zero");

        let written = std::fs::read_to_string(&captured).expect("read the captured argv");
        assert_eq!(written.lines().collect::<Vec<_>>(), vec![text]);
    }

    #[test]
    fn a_missing_binary_is_named_in_the_error() {
        let _lock = ENV_LOCK.lock().expect("environment lock");
        let dir = tempfile::tempdir().expect("temporary directory");
        let missing = dir.path().join("no-such-tts-binary");
        let missing_name = missing.to_str().expect("utf8 path");
        let _guard = set_tts_cmd(missing_name);

        let error = speak("hello").expect_err("a missing binary must fail");
        match &error {
            SpeakError::MissingBinary { program } => assert_eq!(program, missing_name),
            other => panic!("expected MissingBinary, got {other:?}"),
        }
        assert!(
            error.to_string().contains(missing_name),
            "the error must name the missing binary: {error}"
        );
    }
}
