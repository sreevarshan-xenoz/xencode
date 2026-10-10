//! A worker's own git worktree (TM-1): `<parent>/<repo>-team/<id>` on the
//! branch `xencode/team/<id>`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Whether a git setting names a program for git to run: the ways besides
/// hooks that a repository's own config makes git execute something.
pub fn runs_a_program(key: &str, value: &str) -> bool {
    let key = key.to_ascii_lowercase();
    let off = matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "false" | "0" | "no" | "off"
    );
    let ends = |suffix: &str| key.ends_with(suffix);
    match key.as_str() {
        "core.fsmonitor" => !off,
        "core.sshcommand" | "core.askpass" | "diff.external" | "gpg.program" | "core.pager"
        | "core.editor" | "sequence.editor" | "credential.helper" => true,
        _ => {
            (key.starts_with("diff.") && (ends(".command") || ends(".textconv")))
                || (key.starts_with("filter.")
                    && (ends(".clean") || ends(".smudge") || ends(".process")))
                || (key.starts_with("merge.") && ends(".driver"))
                || (key.starts_with("gpg.") && ends(".program"))
                || (key.starts_with("credential.") && ends(".helper"))
        }
    }
}

/// A hooks folder that does not exist, so git finds no hooks to run.
fn no_hooks() -> String {
    let path = std::env::temp_dir().join("xencode-team-git-runs-no-hooks");
    format!("core.hooksPath={}", path.to_string_lossy())
}

/// Run git in `dir`; its standard output, or why it failed in words.
///
/// Repository hooks never run: they are files in `.git`, which a worker
/// agent working in the repository could have written, and the engine runs
/// these commands on its own, without anyone approving them.
pub(crate) fn git(dir: &Path, args: &[&str]) -> Result<String, String> {
    refuse_program_settings(dir)?;
    run_git(dir, args)
}

/// The project's effective settings (git follows any `include`) are read
/// before each command, and git is not run while one of them names a
/// program: a worker agent can write the shared `.git/config`.
fn refuse_program_settings(dir: &Path) -> Result<(), String> {
    // Only what the repository itself holds: the person's own user-wide
    // settings (a credential helper, an editor) are theirs, not something a
    // worker could have written. `--includes` follows any file the
    // repository's settings pull in.
    let mut listed = run_git(
        dir,
        &["config", "--local", "--includes", "--null", "--list"],
    )
    .unwrap_or_default();
    if let Ok(per_worktree) = run_git(
        dir,
        &["config", "--worktree", "--includes", "--null", "--list"],
    ) {
        listed.push_str(&per_worktree);
    }
    for entry in listed.split('\0').filter(|e| !e.is_empty()) {
        let (key, value) = entry.split_once('\n').unwrap_or((entry, ""));
        if runs_a_program(key, value) {
            return Err(format!(
                "the repository's git settings name a program (`{key}`), which git would run; \
                 xencode does not run git here until that setting is removed"
            ));
        }
    }
    Ok(())
}

/// Run git with hooks and the file monitor switched off, whatever the
/// settings say; diffs are asked for with `--no-ext-diff --no-textconv`.
fn run_git(dir: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-c")
        .arg(no_hooks())
        .args(["-c", "core.fsmonitor=false"])
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    } else {
        Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

/// The branch a worker with this id works on.
pub fn branch_for(id: &str) -> String {
    format!("xencode/team/{id}")
}

/// `path` without Windows' long-path prefix (`\\?\C:\…`), which a canonical
/// path carries and git cannot make folders under.
fn plain(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => path.to_path_buf(),
    }
}

/// Make worker `id`'s worktree off `base`. Returns its path and branch.
pub fn create(root: &Path, id: &str, base: &str) -> Result<(PathBuf, String), String> {
    let root = &plain(root);
    let name = root
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("{} has no folder name", root.display()))?;
    let parent = root
        .parent()
        .ok_or_else(|| format!("{} has no parent folder", root.display()))?;
    let path = parent.join(format!("{name}-team")).join(id);
    if path.exists() {
        return Err(format!(
            "a worktree for {id} already exists at {}",
            path.display()
        ));
    }
    let commit = commit_of(root, base)?;
    let branch = branch_for(id);
    let target = path.to_string_lossy().to_string();
    git(
        root,
        &["worktree", "add", "-q", "-b", &branch, &target, &commit],
    )?;
    Ok((path, branch))
}

/// The commit `base` names. `base` comes from the lead agent, a model, so
/// it is resolved before it reaches any other git command: what is not a
/// commit — an option such as `--help`, a branch that does not exist — is
/// refused here, and only the commit's hash travels on.
pub fn commit_of(root: &Path, base: &str) -> Result<String, String> {
    let wanted = format!("{base}^{{commit}}");
    match git(
        root,
        &[
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            &wanted,
        ],
    ) {
        Ok(hash) if !hash.trim().is_empty() => Ok(hash.trim().to_string()),
        Err(why) if why.contains("name a program") => Err(why),
        _ => Err(format!("`{base}` is not a commit in this repository")),
    }
}

/// Remove a worker's worktree and its branch.
pub fn remove(root: &Path, path: &Path, branch: &str) -> Result<(), String> {
    let target = path.to_string_lossy().to_string();
    git(root, &["worktree", "remove", "--force", &target])?;
    git(root, &["branch", "-D", branch])?;
    Ok(())
}

/// Make files the worker created but never added show in `git diff`. Only
/// the worktree's index changes; nothing is committed.
fn include_new_files(worktree: &Path) -> Result<(), String> {
    git(worktree, &["add", "-A", "-N"]).map(|_| ())
}

/// The files the worker changed against `base`, committed or not.
pub fn changed_files(worktree: &Path, base: &str) -> Result<Vec<String>, String> {
    include_new_files(worktree)?;
    Ok(git(worktree, &["diff", "--name-only", base])?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_string)
        .collect())
}

/// The worker's whole change against `base`, as a diff.
pub fn diff(worktree: &Path, base: &str) -> Result<String, String> {
    include_new_files(worktree)?;
    git(worktree, &["diff", "--no-ext-diff", "--no-textconv", base])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn git(dir: &std::path::Path, args: &[&str]) {
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
    }

    /// A real repository with one commit, inside its own temp folder so the
    /// sibling `-team` folder is cleaned up with it.
    fn repo() -> (tempfile::TempDir, std::path::PathBuf) {
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
        (outer, root)
    }

    #[test]
    fn a_worker_gets_its_own_worktree_and_branch_and_they_go_away_together() {
        let (_outer, root) = repo();
        let (path, branch) = create(&root, "w1", "main").unwrap();
        assert_eq!(branch, "xencode/team/w1");
        assert!(path.ends_with("proj-team/w1"), "{path:?}");
        assert_eq!(
            std::fs::read_to_string(path.join("a.txt")).unwrap(),
            "one\n"
        );

        std::fs::write(path.join("a.txt"), "two\n").unwrap();
        std::fs::write(path.join("b.txt"), "new\n").unwrap();
        let mut files = changed_files(&path, "main").unwrap();
        files.sort();
        assert_eq!(files, vec!["a.txt".to_string(), "b.txt".to_string()]);
        let d = diff(&path, "main").unwrap();
        assert!(d.contains("+two") && d.contains("+new"), "{d}");

        remove(&root, &path, &branch).unwrap();
        assert!(!path.exists());
        let branches = Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["branch", "--list", branch.as_str()])
            .output()
            .unwrap();
        assert!(String::from_utf8_lossy(&branches.stdout).trim().is_empty());
    }

    /// The engine works from a canonical project path, which on Windows has
    /// the long `\\?\` form git cannot make folders under.
    #[test]
    fn a_canonical_project_path_still_gets_a_worktree() {
        let (_outer, root) = repo();
        let canonical = std::fs::canonicalize(&root).unwrap();
        let (path, _branch) = create(&canonical, "w1", "main").unwrap();
        assert!(path.join("a.txt").exists(), "{path:?}");
    }

    /// Security review: `base` comes from the lead agent, a model, and goes to
    /// git; something that is not a commit, such as an option, is refused.
    #[test]
    fn a_base_that_is_not_a_commit_is_refused() {
        let (_outer, root) = repo();
        for bad in ["--help", "-c", "no-such-branch", "main;echo"] {
            let err = create(&root, "w1", bad).unwrap_err();
            assert!(err.contains("not a commit"), "{bad}: {err}");
        }
    }

    /// Security review: the engine runs git on the project; a hook in the
    /// repository is code nobody approved, so the engine's git runs none.
    #[test]
    fn the_engines_git_runs_no_repository_hooks() {
        let (_outer, root) = repo();
        let marker = root.join("hook-ran");
        let hook = root.join(".git").join("hooks").join("post-checkout");
        std::fs::write(
            &hook,
            format!(
                "#!/bin/sh\necho ran > '{}'\n",
                marker.to_string_lossy().replace('\\', "/")
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        // Sanity: plain git does run it.
        git(&root, &["checkout", "-q", "-b", "probe"]);
        assert!(marker.exists(), "the hook is a real one");
        std::fs::remove_file(&marker).unwrap();
        git(&root, &["checkout", "-q", "main"]);
        std::fs::remove_file(&marker).ok();

        let (path, branch) = create(&root, "w1", "main").unwrap();
        std::fs::write(path.join("b.txt"), "x").unwrap();
        changed_files(&path, "main").unwrap();
        remove(&root, &path, &branch).unwrap();
        assert!(
            !marker.exists(),
            "a repository hook ran under the engine's git"
        );
    }

    /// Security review, second round: hooks are not git's only way to run a
    /// program. A repository setting that names one (here `core.fsmonitor`)
    /// makes the engine refuse to run git there, naming the setting, and the
    /// program never runs.
    #[test]
    fn a_repository_setting_that_runs_a_program_stops_the_engines_git() {
        let (_outer, root) = repo();
        let (path, _branch) = create(&root, "w1", "main").unwrap();
        let marker = root.join("monitor-ran");
        let script = root.join("monitor.sh");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\necho ran > '{}'\n",
                marker.to_string_lossy().replace('\\', "/")
            ),
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        git(
            &root,
            &[
                "config",
                "core.fsmonitor",
                &script.to_string_lossy().replace('\\', "/"),
            ],
        );
        std::fs::write(path.join("b.txt"), "x").unwrap();
        let err = changed_files(&path, "main").unwrap_err();
        assert!(err.contains("core.fsmonitor"), "{err}");
        assert!(!marker.exists(), "the configured program ran");
    }

    #[test]
    fn the_settings_that_run_programs_are_recognised() {
        for key in [
            "core.fsmonitor",
            "core.sshcommand",
            "diff.external",
            "diff.foo.command",
            "diff.foo.textconv",
            "filter.lfs.smudge",
            "filter.x.clean",
            "filter.x.process",
            "merge.ours.driver",
            "gpg.program",
            "gpg.ssh.program",
        ] {
            assert!(runs_a_program(key, "something"), "{key}");
        }
        assert!(!runs_a_program("core.fsmonitor", "false"), "switched off");
        for key in [
            "user.name",
            "core.autocrlf",
            "remote.origin.url",
            "branch.main.merge",
        ] {
            assert!(!runs_a_program(key, "x"), "{key}");
        }
    }

    #[test]
    fn a_second_worker_with_the_same_id_is_refused_in_words() {
        let (_outer, root) = repo();
        create(&root, "w1", "main").unwrap();
        let err = create(&root, "w1", "main").unwrap_err();
        assert!(err.contains("w1"), "{err}");
    }
}
