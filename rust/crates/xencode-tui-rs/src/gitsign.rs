//! Signed-commit passthrough (`WF-9`).
//!
//! When the user has commit signing configured, an agent-made commit must be
//! signed exactly like a hand-made one — and when signing cannot work, the
//! failure must be reported, never hung on.
//!
//! # The trap this is built around
//!
//! A TUI is often launched where no terminal setup ran: no `GPG_TTY`, no agent
//! forwarding, no pinentry path. GnuPG then cannot ask for a passphrase and the
//! commit either fails cryptically or waits forever on a prompt nobody can see.
//! So this module does two things the bare `git commit` call does not:
//!
//! - it hands the commit subprocess the signing environment explicitly
//!   (`GPG_TTY` resolved from stdin's tty when the shell never set it,
//!   `SSH_AUTH_SOCK` / `GNUPGHOME` passed through when present);
//! - it never changes *whether* signing happens. "Passthrough" means the
//!   user's `commit.gpgsign` decides; this module only makes the yes-case work
//!   and the no-key case fail fast with words.

use std::path::Path;

/// Variables that must reach the commit subprocess unchanged when set.
const PASSTHROUGH_VARS: &[&str] = &["SSH_AUTH_SOCK", "GNUPGHOME", "GPG_AGENT_INFO"];

/// The tty pinentry should ask on, if one can be determined.
///
/// The shell normally exports `GPG_TTY`; a TUI launched from a desktop entry
/// has no such setup, while its stdin *is* the terminal. Reading it back from
/// `/proc/self/fd/0` is Linux-specific and best-effort: anything unresolvable
/// yields `None`, and the commit then fails with words rather than hanging.
pub fn gpg_tty() -> Option<String> {
    if let Ok(tty) = std::env::var("GPG_TTY") {
        if !tty.trim().is_empty() {
            return Some(tty);
        }
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_link("/proc/self/fd/0")
            .ok()
            .map(|p| p.to_string_lossy().into_owned())
            .filter(|p| p.starts_with("/dev/"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Extra environment for a commit subprocess, as `(name, value)` pairs.
pub fn signing_env() -> Vec<(String, String)> {
    let mut env = Vec::new();
    if let Some(tty) = gpg_tty() {
        env.push(("GPG_TTY".to_string(), tty));
    }
    for var in PASSTHROUGH_VARS {
        if let Ok(value) = std::env::var(var) {
            if !value.trim().is_empty() {
                env.push((var.to_string(), value));
            }
        }
    }
    env
}

/// Whether this repository asks for signed commits.
pub fn signing_configured(repo: &Path) -> bool {
    let Ok(output) = std::process::Command::new("git")
        .current_dir(repo)
        .args(["config", "--get", "commit.gpgsign"])
        .output()
    else {
        return false;
    };
    output.status.success()
        && String::from_utf8_lossy(&output.stdout)
            .trim()
            .eq_ignore_ascii_case("true")
}

/// Commit `msg` with `-a`, through the signing environment.
///
/// Returns the first line of stdout on success. On failure returns the first
/// line of stderr, which for a missing key names the problem instead of
/// hanging on a prompt nobody can answer.
pub fn commit_signed(repo: &Path, msg: &str, timeout_secs: u64) -> Result<String, String> {
    let mut command = std::process::Command::new("git");
    command.current_dir(repo).args(["commit", "-am", msg]);
    for (name, value) in signing_env() {
        command.env(name, value);
    }
    // A commit that cannot sign must fail, not wait: gpg without a reachable
    // agent is an error within seconds everywhere this has been tried, and the
    // timeout below is the backstop, not the plan.
    let mut child = command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start git: {e}"))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs.max(10));
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = child.wait_with_output().map_err(|e| e.to_string())?;
                if status.success() {
                    return Ok(first_line(&output.stdout));
                }
                return Err(first_line(&output.stderr));
            }
            Ok(None) => {}
            Err(e) => return Err(format!("could not wait for git: {e}")),
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(
                "git commit did not finish in time — likely waiting on a passphrase prompt \
                 no one can see. Check GPG_TTY and the agent forwarding into this session."
                    .to_string(),
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
}

fn first_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_repo(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xe-sign-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn git(repo: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .current_dir(repo)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    /// A throwaway keyring with one unprotected test key. `%no-protection`
    /// because a passphrase would need the very pinentry path under test.
    fn make_key(home: &Path) -> String {
        let batch = home.join("batch");
        std::fs::write(
            &batch,
            "Key-Type: RSA\nKey-Length: 2048\nName-Real: Test\nName-Email: test@test\nExpire-Date: 0\n%no-protection\n%commit\n",
        )
        .unwrap();
        let output = std::process::Command::new("gpg")
            .arg("--homedir")
            .arg(home)
            .arg("--batch")
            .arg("--gen-key")
            .arg(&batch)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let list = std::process::Command::new("gpg")
            .arg("--homedir")
            .arg(home)
            .args(["--list-keys", "--with-colons"])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&list.stdout).into_owned();
        stdout
            .lines()
            .find_map(|l| {
                if l.starts_with("fpr:") {
                    l.split(':').nth(9).map(str::to_string)
                } else {
                    None
                }
            })
            .expect("a fingerprint")
    }

    #[test]
    fn an_agent_made_commit_verifies() {
        if which_gpg().is_err() {
            eprintln!("skipping: gpg is not installed");
            return;
        }
        let gpghome = temp_repo("gpghome");
        let fingerprint = make_key(&gpghome);
        let repo = temp_repo("repo");
        git(&repo, &["init", "-q", "-b", "main"]);
        git(&repo, &["config", "user.email", "test@test"]);
        git(&repo, &["config", "user.name", "Test"]);
        git(&repo, &["config", "commit.gpgsign", "true"]);
        git(&repo, &["config", "user.signingkey", &fingerprint]);
        std::fs::write(repo.join("f.txt"), "hi\n").unwrap();
        git(&repo, &["add", "f.txt"]);

        // The panel's exact path: GNUPGHOME the way a user session provides it.
        let prior = std::env::var("GNUPGHOME").ok();
        std::env::set_var("GNUPGHOME", &gpghome);
        let committed = commit_signed(&repo, "agent commit", 60);
        match prior {
            Some(v) => std::env::set_var("GNUPGHOME", v),
            None => std::env::remove_var("GNUPGHOME"),
        }
        assert!(committed.is_ok(), "{committed:?}");
        assert!(signing_configured(&repo));

        let verify = std::process::Command::new("git")
            .current_dir(&repo)
            .args(["verify-commit", "HEAD"])
            .env("GNUPGHOME", &gpghome)
            .output()
            .unwrap();
        assert!(
            verify.status.success(),
            "the agent-made commit does not verify: {}",
            String::from_utf8_lossy(&verify.stderr)
        );

        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&gpghome);
    }

    #[test]
    fn a_missing_key_fails_with_words_not_a_hang() {
        if which_gpg().is_err() {
            eprintln!("skipping: gpg is not installed");
            return;
        }
        let gpghome = temp_repo("emptyhome");
        let repo = temp_repo("nokey");
        git(&repo, &["init", "-q", "-b", "main"]);
        git(&repo, &["config", "user.email", "test@test"]);
        git(&repo, &["config", "user.name", "Test"]);
        git(&repo, &["config", "commit.gpgsign", "true"]);
        git(
            &repo,
            &[
                "config",
                "user.signingkey",
                "DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF",
            ],
        );
        std::fs::write(repo.join("f.txt"), "hi\n").unwrap();
        git(&repo, &["add", "f.txt"]);

        let prior = std::env::var("GNUPGHOME").ok();
        std::env::set_var("GNUPGHOME", &gpghome);
        let started = std::time::Instant::now();
        let result = commit_signed(&repo, "agent commit", 60);
        match prior {
            Some(v) => std::env::set_var("GNUPGHOME", v),
            None => std::env::remove_var("GNUPGHOME"),
        }
        let err = result.unwrap_err();
        assert!(
            started.elapsed() < std::time::Duration::from_secs(60),
            "it hung instead of failing"
        );
        assert!(!err.is_empty(), "an empty error explains nothing");
        let _ = std::fs::remove_dir_all(&repo);
        let _ = std::fs::remove_dir_all(&gpghome);
    }

    #[test]
    fn an_unset_tty_resolves_or_declines_without_panicking() {
        // Whatever stdin is here, this must return, not crash.
        let _ = gpg_tty();
    }

    fn which_gpg() -> Result<(), ()> {
        std::process::Command::new("gpg")
            .arg("--version")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
            .then_some(())
            .ok_or(())
    }

    use std::path::PathBuf;
}
