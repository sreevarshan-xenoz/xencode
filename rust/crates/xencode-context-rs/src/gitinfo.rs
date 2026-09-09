//! Git snapshot + invalidation helpers (Pass "Git snapshot").
//!
//! Everything here shells out to the `git` binary, exactly like the rest of
//! the workspace. All paths are returned with `/` separators.

use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

/// branch + HEAD + dirty file count for a workspace.
#[derive(Debug, Clone, Default)]
pub struct GitInfo {
    pub branch: String,
    pub head: String,
    pub dirty: u64,
}

pub fn is_git_repo(root: &Path) -> bool {
    Command::new("git")
        .args(["-C", root.to_string_lossy().as_ref(), "rev-parse", "--git-dir"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Set of repo-relative paths (`/` separators) git considers part of the
/// project: tracked files plus untracked non-ignored files.
///
/// This is the single source of truth for `.gitignore` compliance during the
/// scan; a fallback exclusion list applies when not in a git repo.
pub fn git_file_set(root: &Path) -> Option<HashSet<String>> {
    if !is_git_repo(root) {
        return None;
    }
    let output = Command::new("git")
        .args([
            "-C",
            root.to_string_lossy().as_ref(),
            "ls-files",
            "-co",
            "--exclude-standard",
            "-z",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let mut set = HashSet::new();
    for path in String::from_utf8(output.stdout)
        .unwrap_or_default()
        .split('\0')
    {
        if !path.is_empty() {
            set.insert(path.to_string());
        }
    }
    Some(set)
}

pub fn current_git_info(root: &Path) -> Option<GitInfo> {
    if !is_git_repo(root) {
        return None;
    }
    let branch = Command::new("git")
        .args(["-C", root.to_string_lossy().as_ref(), "branch", "--show-current"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let head = Command::new("git")
        .args(["-C", root.to_string_lossy().as_ref(), "rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(root)
        .output()
        .ok()
        .map(|o| {
            String::from_utf8(o.stdout)
                .map(|s| s.lines().count() as u64)
                .unwrap_or(0)
        })
        .unwrap_or(0);

    Some(GitInfo {
        branch,
        head,
        dirty,
    })
}

/// Repo-relative changed paths (`/` separators) between two HEAD revisions.
pub fn changed_paths_between(root: &Path, old_head: &str, new_head: &str) -> Vec<String> {
    if old_head.is_empty() || new_head.is_empty() || old_head == new_head {
        return Vec::new();
    }
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["diff", "--name-only", old_head, new_head])
        .output()
        .ok()
        .map(|o| {
            String::from_utf8(o.stdout)
                .unwrap_or_default()
                .lines()
                .map(|l| l.replace('\\', "/"))
                .filter(|l| !l.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_repo() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("xencode-git-test-{stamp}"))
    }

    fn git(root: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .expect("git binary should be available");
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }

    #[test]
    fn reports_branch_head_and_dirty_count() {
        let root = temp_repo();
        fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "test@xencode.local"]);
        git(&root, &["config", "user.name", "Xencode Test"]);
        fs::write(root.join("a.rs"), "// a\nfn main() {}\n").unwrap();
        git(&root, &["add", "a.rs"]);
        git(&root, &["commit", "-q", "-m", "initial"]);

        let info = current_git_info(&root).expect("git repo");
        assert!(!info.head.is_empty());
        assert!(!info.branch.is_empty());
        assert_eq!(info.dirty, 0);

        fs::write(root.join("a.rs"), "changed\n").unwrap();
        let info = current_git_info(&root).unwrap();
        assert_eq!(info.dirty, 1);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn diffs_changed_paths_between_heads() {
        let root = temp_repo();
        fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "test@xencode.local"]);
        git(&root, &["config", "user.name", "Xencode Test"]);
        fs::write(root.join("a.rs"), "// a\n").unwrap();
        fs::write(root.join("b.rs"), "// b\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "c1"]);
        let h1 = git(&root, &["rev-parse", "HEAD"]);
        fs::write(root.join("c.rs"), "// c\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "c2"]);
        let h2 = git(&root, &["rev-parse", "HEAD"]);

        let changed = changed_paths_between(&root, &h1, &h2);
        assert_eq!(changed, vec!["c.rs".to_string()]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn git_file_set_respects_gitignore() {
        let root = temp_repo();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("ignored")).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "test@xencode.local"]);
        git(&root, &["config", "user.name", "Xencode Test"]);
        fs::write(root.join(".gitignore"), "ignored/\n").unwrap();
        fs::write(root.join("src/main.rs"), "// main\n").unwrap();
        fs::write(root.join("ignored/out.bin"), "x").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "c1"]);

        let set = git_file_set(&root).expect("tracked set");
        assert!(set.contains("src/main.rs"));
        assert!(set.contains(".gitignore"));
        assert!(!set.contains("ignored/out.bin"));

        fs::remove_dir_all(root).unwrap();
    }
}