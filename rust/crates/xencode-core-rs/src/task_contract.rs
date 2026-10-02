//! The task contract (`OR-15`).
//!
//! Before a worker is launched, xencode tells it what "done" means — and the
//! worker does not get to redefine any of it. This is the ordinary failure the
//! whole orchestrator is built to avoid: an agent that reports itself finished
//! because "finished" was never actually stated. The contract states it, in
//! machine-checkable slots (`QI-3`'s idea, living in a real launch path): the
//! lease and its workspace, the allowed file set, the forbidden paths, the
//! expected deliverables, the verification commands, and the completion
//! condition.
//!
//! The contract is written by xencode and has no method a worker's output could
//! call to widen it. Enforcement is two-fold, and this module is the second
//! half: at run time the forbidden paths are blocked by the worktree boundary
//! and `SE-4`'s gate rather than by asking the worker to behave, and after the
//! worker claims to finish, [`TaskContract::check_finish`] reads the files it
//! *actually* changed — taken from a real diff — and refuses the contract if any
//! of them lands outside the lease. A worker that finishes outside its lease
//! fails here instead of earning a merge.

use std::path::{Component, Path, PathBuf};

/// Why a finished task does not satisfy its contract. Each names the offending
/// path so the refusal can point at it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Breach {
    /// A changed file sits outside the lease workspace — it touched something it
    /// was never given. This is the failure that blocks a merge.
    OutsideLease { path: String },
    /// A changed file sits under a path the contract forbids outright.
    ForbiddenPath { path: String, forbidden: String },
    /// The lease declared an explicit file set and this change is not in it.
    Undeclared { path: String },
}

/// What one launch permits and requires. Built by xencode; a worker is handed a
/// value and no way to edit it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskContract {
    /// The task identifier this contract belongs to.
    pub task: String,
    /// The lease this task runs under (a worktree, one per worker — `OR-4`).
    pub lease: String,
    /// The workspace root the worker may act inside. Anything outside is a
    /// breach, whatever the file set says.
    pub workspace: PathBuf,
    /// The declared file set, relative to the workspace. Empty means "the whole
    /// workspace is allowed"; non-empty narrows it to these files and dirs.
    pub allowed_files: Vec<String>,
    /// Paths that are never writable, even inside the allowed set — secrets,
    /// `.git`, generated locks. Matched as a file or a directory prefix.
    pub forbidden_paths: Vec<String>,
    /// Deliverables the task must leave behind for it to count as done.
    pub deliverables: Vec<String>,
    /// The verification commands that define done (`QI-3`'s machine-checkable
    /// slots). Done means these ran and exited zero — not that the worker said
    /// so.
    pub verification_commands: Vec<String>,
}

impl TaskContract {
    /// Judge a worker's claimed finish against what it actually touched.
    /// `changed` are paths from a real diff, taken relative to the workspace (an
    /// absolute path outside it is caught as `OutsideLease`). Returns `Ok(())`
    /// only when every changed file sits inside the lease, off the forbidden
    /// list, and within the declared set — the point being that a change outside
    /// the lease fails the contract, not merely a change the worker forgot to
    /// mention.
    pub fn check_finish(&self, changed: &[String]) -> Result<(), Vec<Breach>> {
        let root = normalize(&absolutize(&self.workspace, Path::new("")));
        let mut breaches = Vec::new();
        for raw in changed {
            let abs = normalize(&absolutize(&self.workspace, Path::new(raw)));
            if !is_descendant(&root, &abs) {
                breaches.push(Breach::OutsideLease { path: raw.clone() });
                continue;
            }
            let rel = match abs.strip_prefix(&root) {
                Ok(rest) => rest.to_path_buf(),
                Err(_) => {
                    breaches.push(Breach::OutsideLease { path: raw.clone() });
                    continue;
                }
            };
            if let Some(forbidden) = self.forbidden_paths.iter().find(|f| covers(f, &rel)) {
                breaches.push(Breach::ForbiddenPath {
                    path: raw.clone(),
                    forbidden: forbidden.clone(),
                });
                continue;
            }
            if !self.allowed_files.is_empty() && !self.allowed_files.iter().any(|a| covers(a, &rel))
            {
                breaches.push(Breach::Undeclared { path: raw.clone() });
            }
        }
        if breaches.is_empty() {
            Ok(())
        } else {
            Err(breaches)
        }
    }

    /// Whether the task is done, decided by xencode rather than asserted by the
    /// worker. Every deliverable must exist (checked with `exists`, which the
    /// real launch path points at the filesystem), and every verification
    /// command must have exited zero. An empty deliverable list needs no files;
    /// an empty verification list means nothing was required to be proven. A
    /// worker reporting "done" changes nothing here — only files and exit codes
    /// do.
    pub fn completion_met(
        &self,
        exists: &dyn Fn(&Path) -> bool,
        verification_exit: &[i32],
    ) -> bool {
        self.deliverables.iter().all(|d| exists(Path::new(d)))
            && verification_exit.len() == self.verification_commands.len()
            && verification_exit.iter().all(|code| *code == 0)
    }
}

/// Lexical path tidy: resolves `.` and `..` without touching the filesystem, so
/// a not-yet-existing path a write would create can still be judged. A `..`
/// that pops above the root yields a path no longer starting with it, which the
/// descendant check then rejects.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn absolutize(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn is_descendant(root: &Path, candidate: &Path) -> bool {
    candidate.starts_with(root)
}

/// Whether a contract entry (a file or a directory, relative to the workspace)
/// covers a changed path — equal to it, or an ancestor of it. A trailing slash
/// in the entry marks a directory explicitly but a bare name matches both the
/// file and everything under it, which is what a forbidden `.git` or an allowed
/// `src/auth/` needs to mean.
fn covers(entry: &str, rel: &Path) -> bool {
    let trimmed = entry.trim_end_matches('/');
    let entry_path = Path::new(trimmed);
    rel == entry_path || rel.starts_with(entry_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(allowed: &[&str], forbidden: &[&str]) -> TaskContract {
        TaskContract {
            task: "split-retrieval".into(),
            lease: "wt-1".into(),
            workspace: PathBuf::from("/work/wt-1"),
            allowed_files: allowed.iter().map(|s| s.to_string()).collect(),
            forbidden_paths: forbidden.iter().map(|s| s.to_string()).collect(),
            deliverables: Vec::new(),
            verification_commands: Vec::new(),
        }
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// The heart of the item: a worker that finishes OUTSIDE its lease fails the
    /// contract rather than passing quietly.
    #[test]
    fn a_finish_outside_the_lease_is_a_breach_not_a_pass() {
        let c = contract(&["src/"], &[".git"]);
        // One change inside the lease, one that climbed out of it.
        let result = c.check_finish(&s(&["src/retrieval.rs", "../outside/config.toml"]));
        let breaches = result.expect_err("the outside write must fail the contract");
        assert!(
            matches!(&breaches[..], [Breach::OutsideLease { path }] if path == "../outside/config.toml"),
            "expected exactly the outside-lease breach, got {breaches:?}"
        );
    }

    #[test]
    fn an_absolute_path_outside_the_workspace_is_a_breach() {
        let c = contract(&[], &[]);
        let breaches = c
            .check_finish(&s(&["/etc/passwd"]))
            .expect_err("an absolute path outside the workspace is a breach");
        assert!(matches!(&breaches[..], [Breach::OutsideLease { .. }]));
    }

    #[test]
    fn a_forbidden_path_inside_the_workspace_is_still_a_breach() {
        // The workspace is fully allowed, but `.git` is on the forbidden list.
        let c = contract(&[], &[".git"]);
        let breaches = c
            .check_finish(&s(&[".git/HEAD"]))
            .expect_err(".git is forbidden even with no file-set narrowing");
        assert!(
            matches!(&breaches[..], [Breach::ForbiddenPath { forbidden, .. }] if forbidden == ".git"),
            "got {breaches:?}"
        );
    }

    #[test]
    fn a_change_outside_the_declared_file_set_is_undeclared() {
        let c = contract(&["src/auth/"], &[]);
        let breaches = c
            .check_finish(&s(&["docs/readme.md"]))
            .expect_err("a file outside the declared set is not permitted");
        assert!(matches!(&breaches[..], [Breach::Undeclared { .. }]));
    }

    /// A fully-compliant finish passes — the contract is not simply refusing
    /// everything, which would make the refusal meaningless.
    #[test]
    fn a_compliant_finish_passes_the_contract() {
        let c = contract(&["src/auth/"], &[".git"]);
        assert_eq!(
            c.check_finish(&s(&["src/auth/login.rs", "src/auth/session.rs"])),
            Ok(()),
            "changes within the lease and the set must satisfy the contract"
        );
    }

    #[test]
    fn an_empty_file_set_lets_the_whole_workspace_through() {
        let c = contract(&[], &[]);
        assert_eq!(c.check_finish(&s(&["anything/at/all.rs"])), Ok(()));
    }

    /// Done is decided from real files and real exit codes, never from a claim.
    /// A worker saying "done" with a missing deliverable or a failed check is
    /// simply not done.
    #[test]
    fn completion_needs_every_deliverable_present_and_every_check_passing() {
        let mut c = contract(&[], &[]);
        c.deliverables = s(&["out/report.md"]);
        c.verification_commands = s(&["cargo test", "cargo clippy"]);

        // Both checks passed and the deliverable is on disk: done.
        let present = |_: &Path| true;
        assert!(c.completion_met(&present, &[0, 0]));
        // One verification command exited nonzero: not done, exit code decides.
        assert!(!c.completion_met(&present, &[0, 101]));
        // Deliverable missing on disk: not done, whatever the checks said.
        let absent = |_: &Path| false;
        assert!(!c.completion_met(&absent, &[0, 0]));
        // A short result vector is not a pass — every command needs its code.
        assert!(!c.completion_met(&present, &[0]));
    }

    /// The completion check runs against the real filesystem: a deliverable that
    /// exists is present and one that does not is not — no faked predicate.
    #[test]
    fn completion_reads_real_file_existence() {
        let dir = std::env::temp_dir().join(format!("xencode-contract-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let here = dir.join("real.txt");
        std::fs::write(&here, b"x").unwrap();
        let there = dir.join("not-written.txt");

        let mut c = contract(&[], &[]);
        c.workspace = dir.clone();
        c.deliverables = vec![here.to_str().unwrap().to_string()];
        c.verification_commands = Vec::new();
        let real = |p: &Path| p.exists();
        assert!(
            c.completion_met(&real, &[]),
            "the deliverable that exists satisfies done"
        );

        c.deliverables = vec![there.to_str().unwrap().to_string()];
        assert!(
            !c.completion_met(&real, &[]),
            "a deliverable that was never written does not"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn covers_matches_a_file_and_everything_under_a_directory_entry() {
        assert!(covers("src/auth/", Path::new("src/auth/login.rs")));
        assert!(covers("src/auth", Path::new("src/auth/login.rs")));
        assert!(!covers("src/auth", Path::new("src/billing.rs")));
    }

    /// The item asks for QI-3's slot idea *in a real launch path*, so this drives
    /// the contract against a genuine `git diff` from a throwaway repository: the
    /// changed-file list is not hand-written, it is read from git after a real
    /// edit lands partly inside the declared set and partly outside it.
    #[test]
    fn a_real_git_diff_across_the_lease_boundary_fails_the_contract() {
        let dir = std::env::temp_dir().join(format!("xencode-contract-git-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src/auth")).unwrap();
        std::fs::create_dir_all(dir.join("src/billing")).unwrap();
        git(&dir, &["init", "-q"]);
        git(&dir, &["config", "user.email", "t@example.com"]);
        git(&dir, &["config", "user.name", "test"]);
        std::fs::write(dir.join("src/auth/login.rs"), b"v1").unwrap();
        std::fs::write(dir.join("src/billing/charge.rs"), b"v1").unwrap();
        std::fs::write(dir.join("tracked.txt"), b"v1").unwrap();
        git(&dir, &["add", "-A"]);
        git(&dir, &["commit", "-qm", "seed"]);

        // A real worker edits its own file (inside the set) AND another one
        // (outside the declared set) — then `git diff` reports both, honestly.
        std::fs::write(dir.join("src/auth/login.rs"), b"v2").unwrap();
        std::fs::write(dir.join("src/billing/charge.rs"), b"v2").unwrap();

        let changed = git_out(&dir, &["diff", "--name-only"]);
        let changed: Vec<String> = changed.lines().map(|l| l.to_string()).collect();
        assert!(
            changed.contains(&"src/auth/login.rs".to_string())
                && changed.contains(&"src/billing/charge.rs".to_string()),
            "git should report both edits, got {changed:?}"
        );

        // xencode's contract allowed only src/auth/. The real diff shows a change
        // outside it, so the finish fails — earned from git, not asserted here.
        let c = contract(&["src/auth/"], &[]);
        let breaches = c
            .check_finish(&changed)
            .expect_err("a real out-of-set change must fail the contract");
        assert!(
            breaches.iter().any(
                |b| matches!(b, Breach::Undeclared { path } if path == "src/billing/charge.rs")
            ),
            "the real diff's out-of-set file should be named: {breaches:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn git(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .expect("git is on PATH");
        assert!(status.success(), "git {args:?} failed");
    }

    fn git_out(dir: &Path, args: &[&str]) -> String {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git is on PATH");
        assert!(output.status.success(), "git {args:?} failed");
        String::from_utf8_lossy(&output.stdout).into_owned()
    }
}
