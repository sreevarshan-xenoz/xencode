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

/// Longest diff text kept per file; the rest is cut with a marked trailer.
pub const MAX_DIFF_CHARS: usize = 100_000;

/// One file in a diff: line counts, or `None`/`None` for binary files
/// (numstat prints `- -` there).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DiffFile {
    pub path: String,
    pub added: Option<u64>,
    pub deleted: Option<u64>,
}

/// Parse `git diff --numstat` output. Pure — unit-tested. Handles renames
/// (`old => new` and `{a/old => a/new}` forms take the new path) and binary
/// (`- - path`) entries.
pub fn parse_numstat(output: &str) -> Vec<DiffFile> {
    let mut files = Vec::new();
    for line in output.lines() {
        let mut cols = line.split('\t');
        let (Some(added), Some(deleted), Some(raw_path)) = (cols.next(), cols.next(), cols.next())
        else {
            continue;
        };
        let path = if let Some((_, new)) = raw_path.rsplit_once(" => ") {
            new.trim_end_matches('}').to_string()
        } else {
            raw_path.to_string()
        };
        if path.is_empty() {
            continue;
        }
        let counts = |s: &str| {
            if s == "-" {
                None
            } else {
                s.parse::<u64>().ok()
            }
        };
        let is_binary = added == "-" && deleted == "-";
        match (counts(added), counts(deleted)) {
            // Binary (`- -`), or two real counts. Anything else is not a
            // numstat line — skipped, never guessed.
            (a, d) if is_binary || (a.is_some() && d.is_some()) => files.push(DiffFile {
                path,
                added: a,
                deleted: d,
            }),
            _ => {}
        }
    }
    files
}

/// Run git and return stdout, or a short error. Smallest shared helper so
/// every diff entry point reports failures the same way.
fn git_stdout(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("git failed to start: {e}"))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if detail.is_empty() {
            format!("git {} failed", args.join(" "))
        } else {
            detail.lines().next().unwrap_or(&detail).to_string()
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Files changed between `base` and HEAD with line counts — the PR-level
/// triage view. `base` is a branch/tag/commit; the special value `HEAD`
/// diffs the working tree (uncommitted changes) instead.
pub fn git_diff_numstat(root: &Path, base: &str) -> Result<Vec<DiffFile>, String> {
    let range;
    let args: &[&str] = if base == "HEAD" {
        &["diff", "--numstat", "HEAD"]
    } else {
        range = format!("{base}...HEAD");
        &["diff", "--numstat", &range]
    };
    git_stdout(root, args).map(|out| parse_numstat(&out))
}

/// Unified diff of one file between `base` and HEAD, capped at
/// [`MAX_DIFF_CHARS`] with a marked trailer. Same `HEAD` convention as
/// [`git_diff_numstat`].
pub fn git_diff_file(root: &Path, base: &str, path: &str) -> Result<String, String> {
    let range;
    let args: &[&str] = if base == "HEAD" {
        &["diff", "HEAD", "--", path]
    } else {
        range = format!("{base}...HEAD");
        &["diff", &range, "--", path]
    };
    let diff = git_stdout(root, args)?;
    if diff.chars().count() <= MAX_DIFF_CHARS {
        return Ok(diff);
    }
    let kept: String = diff.chars().take(MAX_DIFF_CHARS).collect();
    Ok(format!(
        "{kept}\n…[diff truncated to {MAX_DIFF_CHARS} chars]"
    ))
}

pub fn is_git_repo(root: &Path) -> bool {
    Command::new("git")
        .args([
            "-C",
            root.to_string_lossy().as_ref(),
            "rev-parse",
            "--git-dir",
        ])
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
        .args([
            "-C",
            root.to_string_lossy().as_ref(),
            "branch",
            "--show-current",
        ])
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

/// Repo-relative paths with uncommitted changes (`/` separators): modified,
/// staged or untracked. Parsed from `git status --porcelain`.
pub fn dirty_paths(root: &Path) -> Vec<String> {
    if !is_git_repo(root) {
        return Vec::new();
    }
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(root)
        .output()
        .ok();
    let Some(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let mut paths = Vec::new();
    for line in String::from_utf8(output.stdout).unwrap_or_default().lines() {
        // Porcelain layout: <XY> <path> for normal entries, "XY  old -> new"
        // for renames/copies.
        let entry = if line.len() > 3 { &line[3..] } else { continue };
        let path = match entry.split_once(" -> ") {
            Some((_, new)) => new,
            None => entry,
        };
        let path = path.trim().trim_matches('"').replace('\\', "/");
        if !path.is_empty() {
            paths.push(path);
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn temp_repo() -> PathBuf {
        // A process-wide counter, not a timestamp. Tests in one binary run in
        // parallel threads and can read the same nanosecond, which had them share
        // a directory and clobber each other's assertions.
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-git-test-{unique}"))
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

    #[test]
    fn numstat_parses_counts_renames_and_binaries() {
        let files = parse_numstat(
            "10\t5\tsrc/main.rs\n-\t-\tassets/logo.png\n3\t0\told.rs => new.rs\n1\t1\t{a/old.rs => a/new.rs}\nnot a numstat line\n",
        );
        assert_eq!(
            files,
            vec![
                DiffFile {
                    path: "src/main.rs".to_string(),
                    added: Some(10),
                    deleted: Some(5),
                },
                DiffFile {
                    path: "assets/logo.png".to_string(),
                    added: None,
                    deleted: None,
                },
                DiffFile {
                    path: "new.rs".to_string(),
                    added: Some(3),
                    deleted: Some(0),
                },
                DiffFile {
                    path: "a/new.rs".to_string(),
                    added: Some(1),
                    deleted: Some(1),
                },
            ]
        );
        assert!(parse_numstat("").is_empty());
    }

    #[test]
    fn diff_helpers_read_a_live_repo() {
        let root = temp_repo();
        fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "test@xencode.local"]);
        git(&root, &["config", "user.name", "Xencode Test"]);
        git(&root, &["checkout", "-q", "-b", "main"]);
        fs::write(root.join("a.rs"), "fn a() {}\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "base"]);
        git(&root, &["checkout", "-q", "-b", "feature"]);
        fs::write(root.join("a.rs"), "fn a() {}\nfn b() {}\n").unwrap();
        fs::write(root.join("new.rs"), "fn c() {}\n").unwrap();

        // Working-tree convention first (uncommitted). Untracked new.rs is
        // invisible to git diff by design — only the modified a.rs shows.
        let dirty = git_diff_numstat(&root, "HEAD").unwrap();
        assert_eq!(dirty.len(), 1);
        let a = dirty.iter().find(|f| f.path == "a.rs").unwrap();
        assert_eq!((a.added, a.deleted), (Some(1), Some(0)));

        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "feature"]);
        let files = git_diff_numstat(&root, "main").unwrap();
        assert_eq!(files.len(), 2);
        let diff = git_diff_file(&root, "main", "a.rs").unwrap();
        assert!(diff.contains("fn b() {}"), "{diff}");

        // Unknown base surfaces git's error, never panics.
        assert!(git_diff_numstat(&root, "no-such-branch").is_err());
        // Clean tree diffs to nothing.
        git(&root, &["checkout", "-q", "main"]);
        assert!(git_diff_numstat(&root, "HEAD").unwrap().is_empty());

        fs::remove_dir_all(root).unwrap();
    }
}
