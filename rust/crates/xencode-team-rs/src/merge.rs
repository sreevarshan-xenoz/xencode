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
        let outcome = attempt_merge(root, base, branch, &commands, timeout);
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
    #[cfg(windows)]
    let mut cmd = {
        // `cmd` reads its command line itself; Rust's usual quoting of an
        // argument with quotes in it is not what `cmd` expects, so the check
        // is passed as written.
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("cmd");
        c.arg("/C").raw_arg(command);
        c
    };
    #[cfg(not(windows))]
    let mut cmd = {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    };
    let spawned = cmd
        .current_dir(dir)
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
