//! Git worktree helpers (Milestone D, D3-01).
//!
//! Parsing `git worktree list --porcelain` is a pure function so it can be
//! tested against fixture text; the shells around it use explicit args
//! (no string interpolation into a shell) and reuse `gitinfo`'s shared
//! error-reporting helper.

use std::path::{Path, PathBuf};

use crate::gitinfo::git_stdout;

/// One entry of `git worktree list --porcelain`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorktreeInfo {
    pub path: PathBuf,
    pub head: String,
    /// Short branch name (`main`), or `None` for detached/bare entries.
    pub branch: Option<String>,
    pub detached: bool,
    pub bare: bool,
    pub locked: Option<String>,
    pub prunable: Option<String>,
    /// The first entry git reports is the main worktree of the repo.
    pub is_main: bool,
}

impl WorktreeInfo {
    pub fn short_head(&self) -> String {
        self.head.chars().take(8).collect()
    }

    /// Branch name for display, falling back to the git convention.
    pub fn display_branch(&self) -> String {
        match &self.branch {
            Some(b) => b.clone(),
            None if self.detached => "* detached".to_string(),
            None => format!("* detached ({})", self.short_head()),
        }
    }
}

/// Pure parser for `git worktree list --porcelain` output. Paths are taken
/// verbatim after the `worktree ` prefix, so spaces are safe.
pub fn parse_worktree_list(output: &str) -> Vec<WorktreeInfo> {
    let mut worktrees: Vec<WorktreeInfo> = Vec::new();
    let mut current: Option<WorktreeInfo> = None;

    let flush = |current: &mut Option<WorktreeInfo>, worktrees: &mut Vec<WorktreeInfo>| {
        if let Some(wt) = current.take() {
            worktrees.push(wt);
        }
    };

    for line in output.lines() {
        if let Some(path) = line.strip_prefix("worktree ") {
            flush(&mut current, &mut worktrees);
            current = Some(WorktreeInfo {
                path: PathBuf::from(path),
                head: String::new(),
                branch: None,
                detached: false,
                bare: false,
                locked: None,
                prunable: None,
                is_main: worktrees.is_empty(),
            });
        } else if let Some(wt) = current.as_mut() {
            if let Some(sha) = line.strip_prefix("HEAD ") {
                wt.head = sha.to_string();
            } else if let Some(r#ref) = line.strip_prefix("branch ") {
                wt.branch = Some(
                    r#ref
                        .strip_prefix("refs/heads/")
                        .unwrap_or(r#ref)
                        .to_string(),
                );
            } else if line == "detached" {
                wt.detached = true;
            } else if line == "bare" {
                wt.bare = true;
            } else if let Some(reason) = line.strip_prefix("locked ") {
                wt.locked = Some(reason.to_string());
            } else if line == "locked" {
                wt.locked = Some(String::new());
            } else if let Some(reason) = line.strip_prefix("prunable ") {
                wt.prunable = Some(reason.to_string());
            } else if line == "prunable" {
                wt.prunable = Some(String::new());
            }
        }
    }
    flush(&mut current, &mut worktrees);
    worktrees
}

/// Live worktree list for the repo containing `root`.
pub fn worktree_list(root: &Path) -> Result<Vec<WorktreeInfo>, String> {
    git_stdout(root, &["worktree", "list", "--porcelain"]).map(|out| parse_worktree_list(&out))
}

/// `git worktree add <path> -b <new-branch>` when `create_branch` is set,
/// otherwise `git worktree add <path> [<branch>]`.
pub fn worktree_add(
    root: &Path,
    path: &Path,
    branch: Option<&str>,
    create_branch: bool,
) -> Result<Vec<WorktreeInfo>, String> {
    let path = path.display().to_string();
    let mut args: Vec<&str> = vec!["worktree", "add"];
    if create_branch {
        let branch = branch.ok_or("creating a branch needs a branch name")?;
        args.extend(["-b", branch, &path]);
    } else {
        args.push(&path);
        if let Some(branch) = branch {
            args.push(branch);
        }
    }
    git_stdout(root, &args)?;
    worktree_list(root)
}

/// `git worktree remove <path>` (fails on a dirty worktree unless forced —
/// that is git's own safety net, we do not paper over it).
pub fn worktree_remove(root: &Path, path: &Path, force: bool) -> Result<Vec<WorktreeInfo>, String> {
    let path = path.display().to_string();
    let mut args: Vec<&str> = vec!["worktree", "remove"];
    if force {
        args.push("--force");
    }
    args.push(&path);
    git_stdout(root, &args)?;
    worktree_list(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_main_plus_feature_worktrees() {
        let out = "worktree /home/u/proj\nHEAD 1a2b3c4d5e6f7890abcdef1234567890abcdef12\nbranch refs/heads/main\n\nworktree /home/u/proj-wt/feature\nHEAD 9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c\nbranch refs/heads/feature/x\n\n";
        let wts = parse_worktree_list(out);
        assert_eq!(wts.len(), 2);
        assert_eq!(wts[0].path, PathBuf::from("/home/u/proj"));
        assert_eq!(wts[0].branch.as_deref(), Some("main"));
        assert!(wts[0].is_main);
        assert!(!wts[1].is_main);
        assert_eq!(wts[1].branch.as_deref(), Some("feature/x"));
        assert_eq!(wts[1].short_head(), "9f8e7d6c");
        assert_eq!(wts[1].display_branch(), "feature/x");
    }

    #[test]
    fn parses_detached_locked_and_spaces_in_paths() {
        let out = "worktree /a dir with spaces/repo\nHEAD deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\ndetached\nlocked on a whim\n\nworktree /b\nHEAD cafe0001cafe0001cafe0001cafe0001cafe0001\nbranch refs/heads/main\nprunable\n";
        let wts = parse_worktree_list(out);
        assert_eq!(wts.len(), 2);
        assert_eq!(wts[0].path.display().to_string(), "/a dir with spaces/repo");
        assert!(wts[0].detached);
        assert_eq!(wts[0].branch, None);
        assert_eq!(wts[0].locked.as_deref(), Some("on a whim"));
        assert_eq!(wts[0].display_branch(), "* detached");
        assert_eq!(wts[1].prunable.as_deref(), Some(""));
        assert!(!wts[1].detached);
    }

    #[test]
    fn bare_repo_and_empty_input() {
        assert!(parse_worktree_list("").is_empty());
        let wts = parse_worktree_list(
            "worktree /srv/repo.git\nHEAD 1111222233334444111122223333444411112222\nbare\n",
        );
        assert_eq!(wts.len(), 1);
        assert!(wts[0].bare);
        assert!(wts[0].is_main);
        // No branch, not flagged detached: falls back to the sha form.
        assert_eq!(wts[0].display_branch(), "* detached (11112222)");
    }

    #[test]
    fn live_list_against_a_real_repo_matches_a_manual_add() {
        let tmp = std::env::temp_dir().join(format!("xencode-worktree-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let repo = tmp.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-b", "main"]);
        std::fs::write(repo.join("f.txt"), "hi").unwrap();
        git(&repo, &["add", "f.txt"]);
        git(
            &repo,
            &[
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-m",
                "init",
            ],
        );

        let mains = worktree_list(&repo).unwrap();
        assert_eq!(mains.len(), 1);
        assert!(mains[0].is_main);

        worktree_add(&repo, &tmp.join("wt1"), Some("feat"), true).unwrap();
        let after = worktree_list(&repo).unwrap();
        assert_eq!(after.len(), 2);
        assert_eq!(after[1].branch.as_deref(), Some("feat"));

        worktree_remove(&repo, &tmp.join("wt1"), false).unwrap();
        assert_eq!(worktree_list(&repo).unwrap().len(), 1);

        // Removing the linked worktree twice fails cleanly, not panics.
        assert!(worktree_remove(&repo, &tmp.join("wt1"), false).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn git(root: &Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
