//! Git snapshot + invalidation helpers (Pass "Git snapshot").
//!
//! Everything here shells out to the `git` binary, exactly like the rest of
//! the workspace. All paths are returned with `/` separators.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// branch + HEAD + dirty file count for a workspace.
#[derive(Debug, Clone, Default)]
pub struct GitInfo {
    pub branch: String,
    pub head: String,
    pub dirty: u64,
}

impl GitInfo {
    /// The commit this repository is on, or nothing when it has none yet:
    /// `git rev-parse HEAD` fails on a repository before its first commit. The
    /// index compares this on both sides of one check, so the empty case has to
    /// become `None` here rather than at each caller.
    pub fn revision(&self) -> Option<&str> {
        (!self.head.is_empty()).then_some(self.head.as_str())
    }

    /// The same thing for a person: eight characters, or the reason there are
    /// none.
    pub fn revision_label(&self) -> String {
        match self.revision() {
            Some(head) => head.chars().take(8).collect(),
            None => "(unborn HEAD)".to_string(),
        }
    }
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

/// Run git and return stdout, or a short error. The only place in this module
/// that starts a git process, so every history and diff entry point reports
/// failures the same way — and so no query can be added that misses the two
/// settings below.
///
/// A cloned repository can name its own `core.fsmonitor` hook, which git then
/// runs as part of an ordinary read, and a machine-wide `/etc/gitconfig` can
/// redirect that read's output or hang a filter off it. A repository the user
/// did not write should not be able to run code just because the agent asked
/// it a question about history, so the hook is switched off per call and the
/// system file is left unread. Repository-local config still applies: that is
/// where the user's own `core.commitGraph` setting lives.
pub(crate) fn git_stdout(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-c")
        .arg("core.fsmonitor=false")
        .args(args)
        .current_dir(root)
        .env("GIT_CONFIG_NOSYSTEM", "1")
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
    git_stdout(root, &["rev-parse", "--git-dir"]).is_ok()
}

/// The directory at the top of the repository that contains `dir`, as git
/// reports it, or `None` when `dir` is not inside one. Git resolves a
/// subdirectory to its own top level, which is what lets a command started
/// anywhere in the tree still find the files that live at the root —
/// `CHANGELOG.md`, for instance.
pub fn repo_toplevel(dir: &Path) -> Option<PathBuf> {
    git_stdout(dir, &["rev-parse", "--show-toplevel"])
        .ok()
        .map(|out| out.trim().to_string())
        .filter(|out| !out.is_empty())
        .map(PathBuf::from)
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
    let output = git_stdout(root, &["ls-files", "-co", "--exclude-standard", "-z"]).ok()?;
    let mut set = HashSet::new();
    for path in output.split('\0') {
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
    let read = |args: &[&str]| {
        git_stdout(root, args)
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    };
    Some(GitInfo {
        branch: read(&["branch", "--show-current"]),
        head: read(&["rev-parse", "HEAD"]),
        dirty: read(&["status", "--porcelain"]).lines().count() as u64,
    })
}

/// Repo-relative changed paths (`/` separators) between two HEAD revisions.
pub fn changed_paths_between(root: &Path, old_head: &str, new_head: &str) -> Vec<String> {
    if old_head.is_empty() || new_head.is_empty() || old_head == new_head {
        return Vec::new();
    }
    git_stdout(root, &["diff", "--name-only", old_head, new_head])
        .map(|out| paths_from(&out))
        .unwrap_or_default()
}

/// Turn git's newline-separated path output into `/`-separated paths.
fn paths_from(output: &str) -> Vec<String> {
    output
        .lines()
        .map(|l| l.replace('\\', "/"))
        .filter(|l| !l.is_empty())
        .collect()
}

/// Repo-relative paths with uncommitted changes (`/` separators): modified,
/// staged or untracked. Parsed from `git status --porcelain`.
pub fn dirty_paths(root: &Path) -> Vec<String> {
    if !is_git_repo(root) {
        return Vec::new();
    }
    let Ok(output) = git_stdout(root, &["status", "--porcelain"]) else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for line in output.lines() {
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
    fn a_repository_before_its_first_commit_has_no_revision() {
        let root = temp_repo();
        fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "-q"]);
        fs::write(root.join("a.rs"), "// a\n").unwrap();
        git(&root, &["add", "a.rs"]);

        // `git rev-parse HEAD` exits 128 there and prints the argument back as
        // if it had resolved it, so a reader that ignores the status ends up
        // calling the string "HEAD" this repository's commit.
        let info = current_git_info(&root).expect("git repo");
        assert!(info.head.is_empty(), "head was {:?}", info.head);
        assert_eq!(info.revision(), None);
        assert_eq!(info.revision_label(), "(unborn HEAD)");

        git(&root, &["config", "user.email", "test@xencode.local"]);
        git(&root, &["config", "user.name", "Xencode Test"]);
        git(&root, &["commit", "-q", "-m", "first"]);
        let info = current_git_info(&root).unwrap();
        assert_eq!(info.revision().map(str::len), Some(40));
        assert_eq!(info.revision_label().len(), 8);

        fs::remove_dir_all(root).unwrap();
    }

    /// Every git read in this file has to carry the settings that keep a cloned
    /// repository from running code during a question about its history. They
    /// are set in one place, so a call added beside that place is the failure
    /// this test catches.
    #[test]
    fn git_is_only_started_from_one_place_in_this_file() {
        let source =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gitinfo.rs"))
                .expect("read own source");
        let code = source.split("#[cfg(test)]").next().unwrap_or(&source);

        assert_eq!(
            code.matches("Command::new(\"git\")").count(),
            1,
            "git should be started from the one hardened helper, not from each caller"
        );
        let helper = code.split("Command::new(\"git\")").nth(1).unwrap_or("");
        assert!(
            helper.contains("core.fsmonitor=false"),
            "the place that starts git must switch the repository's fsmonitor hook off"
        );
        assert!(
            helper.contains("GIT_CONFIG_NOSYSTEM"),
            "the place that starts git must leave the machine-wide config unread"
        );
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
