//! The checked merge (TM-3): a worker's branch lands on the base branch only
//! when the project's checks pass on the merged result, built in a scratch
//! worktree, so the person's working copy is never touched before it is green.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::worktree::{commit_of, git};

/// What decides that a merged result may land.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Checks {
    /// The `checks` listed in `.xencode/team.toml`.
    Commands(Vec<String>),
    /// A Cargo project: its tests.
    Cargo,
    /// Nothing to run: the person is asked instead.
    None,
}

/// How a merge ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum MergeOutcome {
    Landed { commit: String },
    ChecksFailed { output: String },
    Conflict { files: Vec<String> },
    NeedsPerson { why: String },
    Refused { why: String },
}

/// `.xencode/team.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TeamSettings {
    /// Commands whose success lets a merge land, run in order.
    #[serde(default)]
    pub checks: Vec<String>,
    /// Workers running at once.
    pub limit: Option<usize>,
    /// Seconds one check may take.
    pub check_timeout_secs: Option<u64>,
}

/// The project's team settings; defaults when the file is absent or cannot
/// be read.
pub fn team_settings(root: &Path) -> TeamSettings {
    std::fs::read_to_string(root.join(".xencode").join("team.toml"))
        .ok()
        .and_then(|text| toml::from_str(&text).ok())
        .unwrap_or_default()
}

/// The checks for the project at `root`: `team.toml` first, else a Cargo
/// project's tests, else none.
pub fn checks_for(root: &Path) -> Checks {
    let settings = team_settings(root);
    if !settings.checks.is_empty() {
        Checks::Commands(settings.checks)
    } else if root.join("Cargo.toml").is_file() {
        Checks::Cargo
    } else {
        Checks::None
    }
}

/// Settings that give a commit an author when the repository names none.
fn identity(root: &Path) -> Vec<String> {
    let has = |key: &str| {
        git(root, &["config", key])
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
    };
    let mut args = Vec::new();
    if !has("user.name") {
        args.extend(["-c".to_string(), "user.name=xencode team".to_string()]);
    }
    if !has("user.email") {
        args.extend([
            "-c".to_string(),
            "user.email=team@xencode.invalid".to_string(),
        ]);
    }
    args
}

fn git_with(dir: &Path, extra: &[String], args: &[&str]) -> Result<String, String> {
    let mut all: Vec<&str> = extra.iter().map(String::as_str).collect();
    all.extend_from_slice(args);
    git(dir, &all)
}

/// Commit everything the worker changed in its worktree. `false` when there
/// was nothing to commit.
pub fn commit_work(worktree: &Path, message: &str) -> Result<bool, String> {
    git(worktree, &["add", "-A"])?;
    if git(worktree, &["status", "--porcelain"])?.trim().is_empty() {
        return Ok(false);
    }
    let who = identity(worktree);
    git_with(worktree, &who, &["commit", "-q", "-m", message])?;
    Ok(true)
}

/// Merge `branch` into `base` if the checks pass on the merged result.
pub fn checked_merge(
    root: &Path,
    base: &str,
    branch: &str,
    checks: &Checks,
    timeout: Duration,
) -> MergeOutcome {
    merge_with(root, base, branch, checks, timeout, false)
}

fn merge_with(
    root: &Path,
    base: &str,
    branch: &str,
    checks: &Checks,
    timeout: Duration,
    allowed: bool,
) -> MergeOutcome {
    let root = &crate::worktree::plain(root);
    let commands = match checks {
        Checks::None => return MergeOutcome::NeedsPerson {
            why:
                "the project has no checks (no `checks` in .xencode/team.toml and no Cargo.toml), \
                      so nothing can say the merged result works"
                    .to_string(),
        },
        Checks::Cargo => vec!["cargo test --quiet".to_string()],
        Checks::Commands(c) => c.clone(),
    };
    match git(root, &["status", "--porcelain", "--untracked-files=no"]) {
        Ok(dirty) if !dirty.trim().is_empty() => {
            return MergeOutcome::Refused {
                why: "the working copy has uncommitted changes; commit or stash them first, \
                      so the merge cannot land on top of them"
                    .to_string(),
            }
        }
        Err(why) => return MergeOutcome::Refused { why },
        Ok(_) => {}
    }
    for attempt in 0..2 {
        let outcome = attempt_merge(root, base, branch, &commands, timeout, allowed);
        match outcome {
            Attempt::Done(outcome) => return outcome,
            Attempt::BaseMoved if attempt == 0 => continue,
            Attempt::BaseMoved => {
                return MergeOutcome::Refused {
                    why: format!("`{base}` kept moving while the checks ran; merge again"),
                }
            }
        }
    }
    unreachable!("the loop returns")
}

/// As [`checked_merge`], once the person said yes to landing a change that
/// edits how checks run: the checks still run.
pub fn checked_merge_person_allowed(
    root: &Path,
    base: &str,
    branch: &str,
    checks: &Checks,
    timeout: Duration,
) -> MergeOutcome {
    merge_with(root, base, branch, checks, timeout, true)
}

/// Files that decide what the checks run or which programs they find. A
/// worker's change to one could choose the checks for this merge or every
/// later one, so landing it is the person's call.
fn decides_the_checks(file: &str) -> bool {
    // Compared as the file system would see each name: Windows (and macOS by
    // default) fold letter case, and Windows drops a name's trailing dots and
    // spaces, so `.XENCODE./team.toml` is `.xencode/team.toml` there.
    let parts: Vec<String> = file
        .split(['/', '\\'])
        .map(|p| p.trim_end_matches(['.', ' ']).to_lowercase())
        .collect();
    let first = parts.first().map(String::as_str).unwrap_or("");
    let whole = parts.len() == 1;
    // A top name a file system may read as another one cannot be judged by
    // its spelling: a Windows short name (`XENCOD~1`), a character macOS
    // ignores, a look-alike letter. Anything but plain ASCII without `~`
    // counts, so the person is asked.
    let plain = first
        .chars()
        .all(|c| c.is_ascii_graphic() && c != '~' || c == ' ');
    !plain
        || first == ".xencode"
        || first == ".cargo"
        || (whole && (first == "rust-toolchain" || first == "rust-toolchain.toml"))
}

/// The files named in the person's question: escaped, so a character that
/// reorders or hides text shows as its code, and at most five of them.
fn named(files: &[String]) -> String {
    const SHOWN: usize = 5;
    let mut out: Vec<String> = files
        .iter()
        .take(SHOWN)
        .map(|f| f.escape_debug().to_string())
        .collect();
    if files.len() > SHOWN {
        out.push(format!("and {} more", files.len() - SHOWN));
    }
    out.join(", ")
}

/// PATH for the checks: only absolute folders, so nothing is found in the
/// merged tree (the worker's files) by being in the current folder.
fn check_path(path: &std::ffi::OsStr) -> std::ffi::OsString {
    let kept: Vec<PathBuf> = std::env::split_paths(path)
        .filter(|p| p.is_absolute())
        .collect();
    std::env::join_paths(kept).unwrap_or_default()
}

/// The process one check runs as, in `dir`.
fn check_command(dir: &Path, command: &str) -> Command {
    #[cfg(windows)]
    let mut cmd = {
        // `cmd` reads its command line itself; Rust's usual quoting of an
        // argument with quotes in it is not what `cmd` expects, so the check
        // is passed as written.
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("cmd");
        c.arg("/C").raw_arg(command);
        // Without this `cmd` looks for a program in the current folder before
        // PATH, and the current folder holds the worker's files.
        c.env("NoDefaultCurrentDirectoryInExePath", "1");
        c
    };
    #[cfg(not(windows))]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    };
    if let Some(path) = std::env::var_os("PATH") {
        cmd.env("PATH", check_path(&path));
    }
    cmd.current_dir(dir);
    cmd
}

enum Attempt {
    Done(MergeOutcome),
    BaseMoved,
}

/// A scratch worktree, removed when dropped.
struct Scratch<'a> {
    root: &'a Path,
    path: PathBuf,
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        let target = self.path.to_string_lossy().to_string();
        let _ = git(self.root, &["worktree", "remove", "--force", &target]);
    }
}

fn scratch_path(root: &Path) -> PathBuf {
    let name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("project");
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    root.parent()
        .unwrap_or(root)
        .join(format!("{name}-team"))
        .join(format!("merge-{}-{stamp}", std::process::id()))
}

fn attempt_merge(
    root: &Path,
    base: &str,
    branch: &str,
    commands: &[String],
    timeout: Duration,
    allowed: bool,
) -> Attempt {
    let done = Attempt::Done;
    let base_commit = match commit_of(root, base) {
        Ok(c) => c,
        Err(why) => return done(MergeOutcome::Refused { why }),
    };
    let branch_commit = match commit_of(root, branch) {
        Ok(c) => c,
        Err(why) => return done(MergeOutcome::Refused { why }),
    };
    let path = scratch_path(root);
    let target = path.to_string_lossy().to_string();
    if let Err(why) = git(
        root,
        &["worktree", "add", "-q", "--detach", &target, &base_commit],
    ) {
        return done(MergeOutcome::Refused { why });
    }
    let scratch = Scratch { root, path };
    let who = identity(root);
    let message = format!("Merge {branch} (checked by xencode)");
    if let Err(why) = git_with(
        &scratch.path,
        &who,
        &["merge", "--no-ff", "-q", "-m", &message, &branch_commit],
    ) {
        let files: Vec<String> = git(&scratch.path, &["diff", "--name-only", "--diff-filter=U"])
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(str::to_string)
            .collect();
        let _ = git(&scratch.path, &["merge", "--abort"]);
        return done(if files.is_empty() {
            MergeOutcome::Refused { why }
        } else {
            MergeOutcome::Conflict { files }
        });
    }
    if !allowed {
        // Names as stored, one per NUL, never quoted; if git cannot say which
        // files changed, nothing lands.
        let changed = match git(
            &scratch.path,
            &[
                "-c",
                "core.quotePath=false",
                "diff",
                "--name-only",
                "-z",
                "--no-renames",
                &base_commit,
                "HEAD",
            ],
        ) {
            Ok(names) => names,
            Err(why) => return done(MergeOutcome::Refused { why }),
        };
        let touched: Vec<String> = changed
            .split('\0')
            .filter(|f| !f.is_empty() && decides_the_checks(f))
            .map(str::to_string)
            .collect();
        if !touched.is_empty() {
            return done(MergeOutcome::NeedsPerson {
                why: format!(
                    "the change edits what decides how checks run ({}); land it only if you \
                     have read it",
                    named(&touched)
                ),
            });
        }
    }
    for command in commands {
        if let Err(output) = run_check(&scratch.path, command, timeout) {
            return done(MergeOutcome::ChecksFailed { output });
        }
    }
    let merged = match git(&scratch.path, &["rev-parse", "HEAD"]) {
        Ok(c) => c.trim().to_string(),
        Err(why) => return done(MergeOutcome::Refused { why }),
    };
    drop(scratch);
    let checked_out = git(root, &["rev-parse", "--abbrev-ref", "HEAD"])
        .map(|b| b.trim().to_string())
        .unwrap_or_default();
    let landed = if checked_out == base {
        git(root, &["merge", "--ff-only", "-q", &merged])
    } else {
        let refname = format!("refs/heads/{base}");
        git(root, &["update-ref", &refname, &merged, &base_commit])
    };
    match landed {
        Ok(_) => done(MergeOutcome::Landed { commit: merged }),
        Err(why) => {
            if commit_of(root, base).ok().as_deref() != Some(base_commit.as_str()) {
                Attempt::BaseMoved
            } else {
                done(MergeOutcome::Refused { why })
            }
        }
    }
}

/// Run one check in `dir`; `Err` with the end of its output when it fails
/// or runs past `timeout`.
fn run_check(dir: &Path, command: &str, timeout: Duration) -> Result<(), String> {
    let log = std::env::temp_dir().join(format!(
        "xencode-team-check-{}-{}.log",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let file =
        std::fs::File::create(&log).map_err(|e| format!("cannot keep the check's output: {e}"))?;
    let errors = file.try_clone().map_err(|e| e.to_string())?;
    let spawned = check_command(dir, command)
        .stdin(Stdio::null())
        .stdout(Stdio::from(file))
        .stderr(Stdio::from(errors))
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(e) => {
            let _ = std::fs::remove_file(&log);
            return Err(format!("`{command}` could not start: {e}"));
        }
    };
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(_) => break None,
        }
    };
    let output = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    let tail: String = {
        let lines: Vec<&str> = output.lines().collect();
        lines[lines.len().saturating_sub(40)..].join("\n")
    };
    match status {
        Some(s) if s.success() => Ok(()),
        Some(s) => Err(format!("`{command}` failed ({s}):\n{tail}")),
        None => Err(format!(
            "`{command}` ran past {} s and was stopped:\n{tail}",
            timeout.as_secs()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// A real repository with one commit, and a worker branch that changed
    /// `a.txt` in its own worktree (committed by `commit_work`).
    fn project() -> (tempfile::TempDir, PathBuf, PathBuf, String) {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("proj");
        std::fs::create_dir(&root).unwrap();
        git(&root, &["init", "-q", "-b", "main"]);
        git(&root, &["config", "user.email", "t@example.invalid"]);
        git(&root, &["config", "user.name", "t"]);
        git(&root, &["config", "core.autocrlf", "false"]);
        std::fs::write(root.join("a.txt"), "one\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "first"]);
        let (wt, branch) = crate::worktree::create(&root, "w1", "main").unwrap();
        std::fs::write(wt.join("a.txt"), "one\ntwo\n").unwrap();
        commit_work(&wt, "add two").unwrap();
        (outer, root, wt, branch)
    }

    fn checks(commands: &[&str]) -> Checks {
        Checks::Commands(commands.iter().map(|c| c.to_string()).collect())
    }

    const SECS: std::time::Duration = std::time::Duration::from_secs(60);

    #[test]
    fn green_checks_land_the_work_on_the_base_branch() {
        let (_o, root, _wt, branch) = project();
        let out = checked_merge(&root, "main", &branch, &checks(&["git --version"]), SECS);
        assert!(matches!(out, MergeOutcome::Landed { .. }), "{out:?}");
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "one\ntwo\n"
        );
        assert_eq!(git(&root, &["rev-parse", "--abbrev-ref", "HEAD"]), "main");
    }

    /// Security review: a worker's change to the files that decide how checks
    /// run (here the team's own settings) would let it choose the checks for
    /// every later merge; the person is asked first, and with their yes the
    /// checks still run.
    #[test]
    fn a_change_to_how_checks_run_needs_the_person() {
        let (_o, root, wt, branch) = project();
        std::fs::create_dir_all(wt.join(".xencode")).unwrap();
        std::fs::write(wt.join(".xencode").join("team.toml"), "checks = []\n").unwrap();
        commit_work(&wt, "no more checks").unwrap();
        let before = git(&root, &["rev-parse", "main"]);
        let out = checked_merge(&root, "main", &branch, &checks(&["git --version"]), SECS);
        match &out {
            MergeOutcome::NeedsPerson { why } => {
                assert!(why.contains(".xencode/team.toml"), "{why}");
                assert!(
                    !why.contains("  "),
                    "the question reads as one sentence: {why}"
                );
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(git(&root, &["rev-parse", "main"]), before);

        let out = checked_merge_person_allowed(
            &root,
            "main",
            &branch,
            &checks(&["git no-such-command"]),
            SECS,
        );
        assert!(
            matches!(out, MergeOutcome::ChecksFailed { .. }),
            "the person's yes does not skip the checks: {out:?}"
        );
        let out =
            checked_merge_person_allowed(&root, "main", &branch, &checks(&["git --version"]), SECS);
        assert!(matches!(out, MergeOutcome::Landed { .. }), "{out:?}");
    }

    /// Security review: a top folder named with characters a file system may
    /// read as other ones (a Windows short name such as `XENCOD~1`, a
    /// character macOS ignores, a look-alike letter) cannot be told apart from
    /// `.xencode` by its spelling, so it counts, and the person is asked.
    #[test]
    fn a_top_folder_that_might_be_another_name_counts() {
        for file in [
            "XENCOD~1/team.toml",
            ".xen\u{200c}code/team.toml",
            ".\u{ff58}encode/team.toml",
            "\u{fffd}/team.toml",
            "RUST-T~1.TOM",
        ] {
            assert!(decides_the_checks(file), "{file:?}");
        }
        for file in [
            "src/a~1.rs",
            "docs/caf\u{e9}.md",
            "README.md",
            "my notes.txt",
        ] {
            assert!(!decides_the_checks(file), "{file:?}");
        }
    }

    /// Security review: the names a worker chose go into the person's
    /// question, so a character that reorders or hides text is shown escaped,
    /// and only the first few names are listed.
    #[test]
    fn the_names_in_the_question_cannot_restyle_it() {
        let names: Vec<String> = (0..9)
            .map(|n| format!(".xencode/\u{202e}lmot{n}.toml"))
            .collect();
        let said = named(&names);
        assert!(!said.contains('\u{202e}'), "{said}");
        assert!(said.contains("\\u{202e}"), "{said}");
        assert!(said.contains("and 4 more"), "{said}");
    }

    /// Security review: Windows folds letter case and drops a trailing dot or
    /// space from a folder name, so each of these is the same file as one the
    /// checks read; they are told apart the way the file system would.
    #[test]
    fn a_settings_file_spelled_another_way_still_counts() {
        for file in [
            ".XENCODE/team.toml",
            ".xencode./team.toml",
            ".xencode /team.toml",
            ".Cargo/config.toml",
            "Rust-Toolchain.toml",
            "rust-toolchain.",
        ] {
            assert!(decides_the_checks(file), "{file}");
        }
        for file in [
            "src/.xencode.rs",
            "xencode/team.toml",
            "docs/rust-toolchain.md",
        ] {
            assert!(!decides_the_checks(file), "{file}");
        }
    }

    /// Security review: git quotes a name with characters outside ASCII
    /// (`".xencode/\303\251.toml"`) unless told not to; a quoted name must not
    /// slip past the guard.
    #[test]
    fn a_settings_file_with_a_quoted_name_still_needs_the_person() {
        let (_o, root, wt, branch) = project();
        std::fs::create_dir_all(wt.join(".xencode")).unwrap();
        std::fs::write(wt.join(".xencode").join("é.toml"), "x\n").unwrap();
        commit_work(&wt, "settings").unwrap();
        let out = checked_merge(&root, "main", &branch, &checks(&["git --version"]), SECS);
        assert!(matches!(out, MergeOutcome::NeedsPerson { .. }), "{out:?}");
    }

    /// Security review: on Windows `cmd` looks in the current folder before
    /// PATH, and the checks run in a tree holding the worker's files; a
    /// `git.bat` the worker wrote must not be what `git --version` runs.
    #[cfg(windows)]
    #[test]
    fn a_program_the_worker_wrote_is_not_what_a_check_runs() {
        let (o, root, wt, branch) = project();
        let marker = o.path().join("hijacked");
        std::fs::write(
            wt.join("git.bat"),
            format!("@echo hijacked> \"{}\"\r\n", marker.display()),
        )
        .unwrap();
        commit_work(&wt, "a helper").unwrap();
        let out = checked_merge(&root, "main", &branch, &checks(&["git --version"]), SECS);
        assert!(matches!(out, MergeOutcome::Landed { .. }), "{out:?}");
        assert!(!marker.exists(), "the worker's git.bat ran");
        // This holds even where the person's own environment does not set
        // the variable: the check's process sets it.
        let cmd = check_command(&root, "git --version");
        assert!(cmd
            .get_envs()
            .any(|(k, v)| k == "NoDefaultCurrentDirectoryInExePath"
                && v == Some(std::ffi::OsStr::new("1"))));
    }

    /// On Unix the same holds through PATH: an entry that names the current
    /// folder, or any folder relative to it, is left out for the checks.
    #[test]
    fn the_checks_path_names_no_folder_relative_to_the_tree() {
        let abs = std::env::temp_dir();
        let given = std::env::join_paths([
            abs.clone(),
            PathBuf::from("."),
            PathBuf::from("bin"),
            PathBuf::from("./tools"),
            abs.join("x"),
        ])
        .unwrap();
        let kept = check_path(&given);
        let kept: Vec<_> = std::env::split_paths(&kept).collect();
        let want = vec![abs.clone(), abs.join("x")];
        assert_eq!(kept, want);
    }

    /// The engine passes its canonical project path (the long Windows form).
    #[test]
    fn a_canonical_project_path_still_merges() {
        let (_o, root, _wt, branch) = project();
        let canonical = std::fs::canonicalize(&root).unwrap();
        let out = checked_merge(
            &canonical,
            "main",
            &branch,
            &checks(&["git --version"]),
            SECS,
        );
        assert!(matches!(out, MergeOutcome::Landed { .. }), "{out:?}");
    }

    #[test]
    fn red_checks_leave_the_base_branch_alone() {
        let (_o, root, _wt, branch) = project();
        let before = git(&root, &["rev-parse", "main"]);
        let out = checked_merge(
            &root,
            "main",
            &branch,
            &checks(&["git no-such-command"]),
            SECS,
        );
        assert!(matches!(out, MergeOutcome::ChecksFailed { .. }), "{out:?}");
        assert_eq!(git(&root, &["rev-parse", "main"]), before);
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "one\n"
        );
    }

    #[test]
    fn a_check_that_runs_too_long_is_stopped_and_lands_nothing() {
        let (_o, root, _wt, branch) = project();
        let before = git(&root, &["rev-parse", "main"]);
        let slow = if cfg!(windows) {
            "ping -n 30 127.0.0.1"
        } else {
            "sleep 30"
        };
        let start = std::time::Instant::now();
        let out = checked_merge(
            &root,
            "main",
            &branch,
            &checks(&[slow]),
            Duration::from_secs(2),
        );
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "{:?}",
            start.elapsed()
        );
        match &out {
            MergeOutcome::ChecksFailed { output } => {
                assert!(output.contains("was stopped"), "{output}")
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(git(&root, &["rev-parse", "main"]), before);
    }

    #[test]
    fn a_conflict_is_reported_with_its_files() {
        let (_o, root, _wt, branch) = project();
        std::fs::write(root.join("a.txt"), "one\nother\n").unwrap();
        git(&root, &["commit", "-q", "-am", "meanwhile"]);
        let out = checked_merge(&root, "main", &branch, &checks(&["git --version"]), SECS);
        match out {
            MergeOutcome::Conflict { files } => assert_eq!(files, vec!["a.txt".to_string()]),
            other => panic!("{other:?}"),
        }
    }

    /// Review Focus 3.
    #[test]
    fn uncommitted_changes_in_the_working_copy_refuse_the_merge() {
        let (_o, root, _wt, branch) = project();
        std::fs::write(root.join("a.txt"), "half done\n").unwrap();
        let out = checked_merge(&root, "main", &branch, &checks(&["git --version"]), SECS);
        assert!(matches!(out, MergeOutcome::Refused { .. }), "{out:?}");
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "half done\n"
        );
    }

    #[test]
    fn no_checks_means_the_person_is_asked() {
        let (_o, root, _wt, branch) = project();
        let out = checked_merge(&root, "main", &branch, &Checks::None, SECS);
        assert!(matches!(out, MergeOutcome::NeedsPerson { .. }), "{out:?}");
    }

    /// Review Focus 2: the base moves while the checks run (here the check
    /// itself commits to it); the merge starts over from the new base and
    /// lands with both changes.
    #[test]
    fn a_base_that_moves_during_the_checks_is_merged_again() {
        let (_o, root, _wt, branch) = project();
        let marker = root.join("moved-once");
        let mover = format!(
            "git -C \"{root}\" rev-parse --verify --quiet refs/tags/moved || (git -C \"{root}\" commit -q --allow-empty -m moved && git -C \"{root}\" tag moved)",
            root = root.to_string_lossy().replace('\\', "/")
        );
        let _ = marker;
        let out = checked_merge(&root, "main", &branch, &checks(&[&mover]), SECS);
        assert!(matches!(out, MergeOutcome::Landed { .. }), "{out:?}");
        let log = git(&root, &["log", "--format=%s", "main"]);
        assert!(log.contains("moved"), "{log}");
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "one\ntwo\n"
        );
    }

    #[test]
    fn a_worker_with_nothing_to_commit_is_said() {
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("proj");
        std::fs::create_dir(&root).unwrap();
        git(&root, &["init", "-q", "-b", "main"]);
        git(&root, &["config", "user.email", "t@example.invalid"]);
        git(&root, &["config", "user.name", "t"]);
        std::fs::write(root.join("a.txt"), "one\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "first"]);
        let (wt, _branch) = crate::worktree::create(&root, "w1", "main").unwrap();
        assert!(!commit_work(&wt, "nothing").unwrap(), "nothing to commit");
    }

    #[test]
    fn the_checks_come_from_team_toml_then_cargo() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(checks_for(dir.path()), Checks::None));
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        assert!(matches!(checks_for(dir.path()), Checks::Cargo));
        std::fs::create_dir(dir.path().join(".xencode")).unwrap();
        std::fs::write(
            dir.path().join(".xencode").join("team.toml"),
            "checks = [\"make test\"]\n",
        )
        .unwrap();
        match checks_for(dir.path()) {
            Checks::Commands(c) => assert_eq!(c, vec!["make test".to_string()]),
            other => panic!("{other:?}"),
        }
    }
}
