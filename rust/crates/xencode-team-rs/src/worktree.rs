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
        "core.sshcommand"
        | "core.askpass"
        | "diff.external"
        | "gpg.program"
        | "core.pager"
        | "core.editor"
        | "sequence.editor"
        | "credential.helper"
        | "core.gitproxy"
        | "core.alternaterefscommand"
        | "uploadpack.packobjectshook"
        | "interactive.difffilter"
        | "core.hookspath"
        | "diff.tool"
        | "merge.tool"
        | "diff.guitool"
        | "merge.guitool" => true,
        _ => {
            (key.starts_with("diff.") && (ends(".command") || ends(".textconv")))
                || (key.starts_with("filter.")
                    && (ends(".clean") || ends(".smudge") || ends(".process")))
                || (key.starts_with("merge.") && ends(".driver"))
                || (key.starts_with("gpg.") && ends(".program"))
                || (key.starts_with("credential.") && ends(".helper"))
                || (key.starts_with("remote.") && (ends(".uploadpack") || ends(".receivepack")))
                // `checkout`, `rebase` and `merge` are git's own; only a `!`
                // value names a command.
                || (key.starts_with("submodule.")
                    && ends(".update")
                    && value.trim_start().starts_with('!'))
                || (key.starts_with("difftool.") && ends(".cmd"))
                || (key.starts_with("mergetool.") && ends(".cmd"))
                || key.starts_with("pager.")
        }
    }
}

/// A hooks folder that cannot exist, so git finds no hooks to run: a name
/// under this program's own file, where no folder can ever be made. A shared
/// temp folder would not do, since anything running as the person could make
/// it and put hooks in it.
fn no_hooks() -> String {
    let path = std::env::current_exe()
        .unwrap_or_else(|_| PathBuf::from("/dev/null"))
        .join("no-hooks");
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
    // Fails closed: settings that cannot be read are not assumed safe.
    let mut listed = run_git(
        dir,
        &["config", "--local", "--includes", "--null", "--list"],
    )
    .map_err(|why| {
        format!("cannot read the repository's git settings, so git is not run: {why}")
    })?;
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
    format!("{BRANCH_PREFIX}{id}")
}

/// What every worker branch's name starts with.
pub const BRANCH_PREFIX: &str = "xencode/team/";

/// `path` without Windows' long-path prefix (`\\?\C:\…`), which a canonical
/// path carries and git cannot make folders under.
pub(crate) fn plain(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => path.to_path_buf(),
    }
}

/// The folder beside the project that holds its worker worktrees:
/// `<parent>/<repo>-team`.
pub fn team_dir(root: &Path) -> Result<PathBuf, String> {
    let root = &plain(root);
    let name = root
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("{} has no folder name", root.display()))?;
    let parent = root
        .parent()
        .ok_or_else(|| format!("{} has no parent folder", root.display()))?;
    Ok(parent.join(format!("{name}-team")))
}

/// Folders in the team folder named like a worker (`w1`, `w2`, …) that git
/// no longer lists as worktrees: what a removal left behind while a process
/// still had the folder open.
pub fn leftover_folders(root: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let registered: Vec<PathBuf> = team_worktrees(root)?
        .into_iter()
        .map(|(_, path, _)| path)
        .collect();
    let dir = team_dir(root)?;
    let mut found = Vec::new();
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let is_worker = is_worker_id(&name);
        let path = entry.path();
        let listed = registered.iter().any(|r| same_folder(r, &path));
        if is_worker && path.is_dir() && !listed {
            found.push((name, path));
        }
    }
    Ok(found)
}

/// The branch the project has checked out, read with the engine's own git.
pub fn current_branch(root: &Path) -> Option<String> {
    let root = &plain(root);
    git(root, &["rev-parse", "--abbrev-ref", "HEAD"])
        .ok()
        .map(|b| b.trim().to_string())
        .filter(|b| !b.is_empty())
}

/// Whether `name` is one of the project's local branches. A worker's work
/// lands on a branch, so its base must be one.
pub fn is_branch(root: &Path, name: &str) -> bool {
    let root = &plain(root);
    !name.starts_with('-')
        && git(
            root,
            &[
                "show-ref",
                "--verify",
                "--quiet",
                &format!("refs/heads/{name}"),
            ],
        )
        .is_ok()
}

/// Remove the scratch worktrees (`merge-…`) merges left in the team folder,
/// which a removal can leave behind on Windows. Call only while no merge
/// runs. Returns the folders removed.
pub fn sweep_scratch(root: &Path) -> Vec<String> {
    let root = &plain(root);
    let _ = git(root, &["worktree", "prune"]);
    let Ok(dir) = team_dir(root) else {
        return Vec::new();
    };
    let mut removed = Vec::new();
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let scratch = name.strip_prefix("merge-").is_some_and(|rest| {
            !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit() || c == '-')
        });
        let path = entry.path();
        if !scratch || !path.is_dir() {
            continue;
        }
        let target = path.to_string_lossy().to_string();
        let _ = git(root, &["worktree", "remove", "--force", &target]);
        let _ = git(root, &["worktree", "prune"]);
        if path.exists() {
            let _ = std::fs::remove_dir_all(&path);
        }
        if !path.exists() {
            removed.push(name);
        }
    }
    removed
}

/// Whether `name` is a worker id: `w` and a number.
pub fn is_worker_id(name: &str) -> bool {
    name.len() > 1 && name.starts_with('w') && name[1..].chars().all(|c| c.is_ascii_digit())
}

/// Whether `path` is a worker's folder: directly inside the project's team
/// folder and named like a worker.
fn is_workers_folder(root: &Path, path: &Path) -> bool {
    let (Ok(dir), Some(parent), Some(name)) = (
        team_dir(root),
        path.parent(),
        path.file_name().and_then(|n| n.to_str()),
    ) else {
        return false;
    };
    is_worker_id(name) && same_folder(&dir, parent)
}

fn same_folder(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| plain(p).to_string_lossy().replace('\\', "/").to_lowercase();
    norm(a) == norm(b)
}

/// Make worker `id`'s worktree off `base`. Returns its path and branch.
pub fn create(root: &Path, id: &str, base: &str) -> Result<(PathBuf, String), String> {
    let root = &plain(root);
    let path = team_dir(root)?.join(id);
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
    let root = &plain(root);
    let path = &plain(path);
    let target = path.to_string_lossy().to_string();
    // Only ever a worker's own folder: a worktree made elsewhere on a
    // worker-looking branch (by a worker running git itself) is not deleted.
    if !is_workers_folder(root, path) {
        return Err(format!(
            "{target} is not a worker's folder in {}; it is left alone",
            team_dir(root)
                .map(|d| d.display().to_string())
                .unwrap_or_default()
        ));
    }
    // git unregisters the worktree and deletes its files even when the folder
    // itself cannot go yet (on Windows, while a process works in it); a later
    // try then finds no worktree, so what git left is finished here.
    if git(root, &["worktree", "remove", "--force", &target]).is_err() {
        git(root, &["worktree", "prune"])?;
        if registered(root, path)? {
            return Err(format!("git could not remove the worktree {target}"));
        }
    }
    if path.exists() {
        std::fs::remove_dir_all(path).map_err(|e| format!("could not remove {target}: {e}"))?;
    }
    let known = git(root, &["branch", "--list", branch])?;
    if !known.trim().is_empty() {
        git(root, &["branch", "-D", branch])?;
    }
    Ok(())
}

/// Whether git still lists `path` as one of the project's worktrees.
fn registered(root: &Path, path: &Path) -> Result<bool, String> {
    let wanted = path.to_string_lossy().replace('\\', "/").to_lowercase();
    Ok(git(root, &["worktree", "list", "--porcelain"])?
        .lines()
        .filter_map(|l| l.strip_prefix("worktree "))
        .any(|l| l.replace('\\', "/").to_lowercase() == wanted))
}

/// The project's worker worktrees git knows about: each worker's id, its
/// folder and its branch.
pub fn team_worktrees(root: &Path) -> Result<Vec<(String, PathBuf, String)>, String> {
    let root = &plain(root);
    let listing = git(root, &["worktree", "list", "--porcelain"])?;
    let mut found = Vec::new();
    for entry in listing.split("\n\n") {
        let path = entry.lines().find_map(|l| l.strip_prefix("worktree "));
        let branch = entry
            .lines()
            .find_map(|l| l.strip_prefix("branch refs/heads/"));
        if let (Some(path), Some(branch)) = (path, branch) {
            if let Some(id) = branch.strip_prefix(BRANCH_PREFIX) {
                let path = PathBuf::from(path);
                // Only worktrees where xencode makes them, named for their
                // worker; one made anywhere else is not the team's.
                let named = path.file_name().and_then(|n| n.to_str()) == Some(id);
                if named && is_workers_folder(root, &path) {
                    found.push((id.to_string(), path, branch.to_string()));
                }
            }
        }
    }
    Ok(found)
}

/// Whether the worktree holds changes nobody committed.
pub fn has_changes(worktree: &Path) -> Result<bool, String> {
    let mut args = vec!["status", "--porcelain"];
    args.extend_from_slice(&WORK);
    Ok(!git(worktree, &args)?.trim().is_empty())
}

/// What counts as the worker's work: every path but `.xencode/`, where a
/// xencode worker's own engine keeps its state and the person keeps the
/// team's settings. A worker's change there is never committed for it.
pub const WORK: [&str; 3] = ["--", ".", ":(exclude).xencode"];

/// Make files the worker created but never added show in `git diff`. Only
/// the worktree's index changes; nothing is committed.
fn include_new_files(worktree: &Path) -> Result<(), String> {
    let mut args = vec!["add", "-A", "-N"];
    args.extend_from_slice(&WORK);
    git(worktree, &args).map(|_| ())
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

    /// Review: scratch worktrees a merge left are swept; nothing else in the
    /// team folder is touched.
    #[test]
    fn leftover_merge_scratch_is_swept_and_nothing_else() {
        let (_outer, root) = repo();
        let dir = team_dir(&root).unwrap();
        let registered = dir.join("merge-1-2");
        let target = registered.to_string_lossy().to_string();
        super::git(
            &root,
            &["worktree", "add", "-q", "--detach", &target, "main"],
        )
        .unwrap();
        std::fs::create_dir_all(dir.join("merge-3-4")).unwrap();
        std::fs::create_dir_all(dir.join("notes")).unwrap();
        let mut removed = sweep_scratch(&root);
        removed.sort();
        assert_eq!(
            removed,
            vec!["merge-1-2".to_string(), "merge-3-4".to_string()]
        );
        assert!(dir.join("notes").exists());
        assert!(is_branch(&root, "main"));
        assert!(!is_branch(&root, "--help"));
        let head = super::git(&root, &["rev-parse", "HEAD"]).unwrap();
        assert!(!is_branch(&root, head.trim()), "a commit is not a branch");
        assert_eq!(current_branch(&root).as_deref(), Some("main"));
    }

    /// Security review: a worker could make a worktree of its own anywhere
    /// on a worker-looking branch; removing one is refused unless it is a
    /// worker's folder inside the project's team folder, so nothing else is
    /// ever deleted.
    #[test]
    fn only_a_workers_own_folder_is_ever_removed() {
        let (outer, root) = repo();
        let elsewhere = outer.path().join("precious");
        let target = elsewhere.to_string_lossy().to_string();
        super::git(
            &root,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "xencode/team/w9",
                &target,
                "main",
            ],
        )
        .unwrap();
        std::fs::write(elsewhere.join("keep.txt"), "mine\n").unwrap();
        let err = remove(&root, &elsewhere, "xencode/team/w9").unwrap_err();
        assert!(err.contains("not a worker's folder"), "{err}");
        assert!(elsewhere.join("keep.txt").exists());
        let named = team_dir(&root).unwrap().join("notes");
        std::fs::create_dir_all(&named).unwrap();
        assert!(remove(&root, &named, "xencode/team/notes").is_err());
        assert!(named.exists());
        let listed: Vec<String> = team_worktrees(&root)
            .unwrap()
            .into_iter()
            .map(|(id, _, _)| id)
            .collect();
        assert!(!listed.contains(&"w9".to_string()), "{listed:?}");
    }

    /// git can unregister a worktree and delete its files but leave the
    /// folder (on Windows, while a process works in it); removing it again
    /// finishes the job instead of failing on "not a working tree".
    #[test]
    fn a_half_removed_worktree_is_finished_off() {
        let (_outer, root) = repo();
        let canonical = std::fs::canonicalize(&root).unwrap();
        let (path, branch) = create(&canonical, "w1", "main").unwrap();
        let target = path.to_string_lossy().to_string();
        super::git(&root, &["worktree", "remove", "--force", &target]).unwrap();
        std::fs::create_dir_all(&path).unwrap();

        remove(&canonical, &std::fs::canonicalize(&path).unwrap(), &branch).unwrap();
        assert!(!path.exists());
        let branches = super::git(&root, &["branch", "--list", &branch]).unwrap();
        assert!(branches.trim().is_empty(), "{branches}");
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
    fn the_hooks_folder_can_never_be_made() {
        // Review: a shared temp folder could be made, and filled with hooks,
        // by anything running as the person; a folder under a file cannot.
        let setting = no_hooks();
        let path = PathBuf::from(setting.strip_prefix("core.hooksPath=").unwrap());
        assert!(path.parent().unwrap().is_file(), "{setting}");
        assert!(std::fs::create_dir_all(&path).is_err(), "{setting}");
    }

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
        // Security review, third round: more ways to name a program.
        for key in [
            "core.gitproxy",
            "core.alternaterefscommand",
            "uploadpack.packobjectshook",
            "remote.origin.uploadpack",
            "remote.origin.receivepack",
            "pager.diff",
            "interactive.difffilter",
            "core.hookspath",
            "diff.tool",
            "difftool.x.cmd",
            "mergetool.x.cmd",
        ] {
            assert!(runs_a_program(key, "something"), "{key}");
        }
        assert!(!runs_a_program("core.fsmonitor", "false"), "switched off");
        assert!(runs_a_program("submodule.lib.update", "!make"));
        assert!(!runs_a_program("submodule.lib.update", "rebase"));
        for key in [
            "user.name",
            "core.autocrlf",
            "remote.origin.url",
            "branch.main.merge",
        ] {
            assert!(!runs_a_program(key, "x"), "{key}");
        }
    }

    /// Security review, third round: when the settings cannot be read, git is
    /// not run (the check fails closed).
    #[test]
    fn settings_that_cannot_be_read_stop_the_engines_git() {
        let not_a_repo = tempfile::tempdir().unwrap();
        let err = changed_files(not_a_repo.path(), "main").unwrap_err();
        assert!(err.contains("cannot read"), "{err}");
    }

    #[test]
    fn a_second_worker_with_the_same_id_is_refused_in_words() {
        let (_outer, root) = repo();
        create(&root, "w1", "main").unwrap();
        let err = create(&root, "w1", "main").unwrap_err();
        assert!(err.contains("w1"), "{err}");
    }
}
