// Git-backed checkpoints (QTR-4).
//
// The in-memory snapshots in `agent_tools::CheckpointStore` hold the bytes a
// file had before the agent overwrote it, which is enough to put a file back —
// but never enough to notice that a person edited it in between. This module
// records what the agent wrote as a commit on a ref of its own, so `/rewind`
// can ask git whether the file on disk still matches the state the agent left
// it in, and refuse to overwrite a hand edit it knows nothing about.
//
// # What this must never do
//
// It runs on the user's repository while they are working in it, so the whole
// design is about not disturbing that work:
//
// - `HEAD`, the current branch, the working tree and the user's own index are
//   never touched. Every staging step happens in a throwaway index file passed
//   through `GIT_INDEX_FILE`, and the only ref written is
//   `refs/heads/xencode/ckpt`.
// - Objects are never staged wholesale. There is no `git add -A` and no
//   directory-wide spec here: exactly the files the agent wrote this turn go
//   into the commit, so the 46 GiB `target/` tree is absent by construction
//   rather than by luck, and anything a `.gitignore` covers is skipped
//   explicitly (see `write_turn`).
// - Nothing here asks for the build cache to move. A checkpoint commit holds
//   source-sized blobs of files the agent edited; the build directory is not
//   part of that set, so sharing or relocating it would buy nothing.
//
// Every failure is a reason string the caller shows as-is. A missing git, an
// unborn `HEAD` or a repository that refuses the ref update leaves the
// in-memory rewind working exactly as before — the human-edit guard is the
// thing that goes away, and it says so instead of pretending to have checked.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The one ref this module writes. Deliberately a `refs/heads/` name so that
/// `git log xencode/ckpt` and `git diff xencode/ckpt` work the way a person
/// would type them, and so it cannot collide with a real branch: a branch
/// called `xencode` would make the `xencode/` prefix unusable, and git reports
/// that rather than us silently writing somewhere else.
pub const CKPT_REF: &str = "refs/heads/xencode/ckpt";

/// Identity and signing forced onto `commit-tree`. These are scratch commits,
/// not the user's work, so they are attributed to xencode rather than to the
/// person editing the repository — and never signed. The signing half matters:
/// a TUI started from a desktop entry has no pinentry path, and a checkpoint
/// commit that stopped to ask for a passphrase would hang the turn the user is
/// waiting on. Signing the user's own commits is untouched, because that is
/// `gitsign`'s job and this never runs `git commit`.
const COMMIT_CONFIG: &[&str] = &[
    "-c",
    "user.name=xencode",
    "-c",
    "user.email=xencode@localhost",
    "-c",
    "commit.gpgsign=false",
];

struct GitOutcome {
    ok: bool,
    out: String,
    err: String,
}

/// The single place this module starts a git process.
///
/// Mirrors `xencode_context_rs::gitinfo`'s choke point: a repository can name
/// its own `core.fsmonitor` hook, which git runs during an ordinary call, and a
/// machine-wide `/etc/gitconfig` can redirect output. We are writing to this
/// repository rather than only reading it, so the hook is off and the system
/// file is unread for every call, and `ok` is returned rather than an error so
/// the two calls that use a non-zero exit as their answer (`rev-parse -q`,
/// `check-ignore`) can read it.
fn git(root: &Path, args: &[&str], index: Option<&Path>) -> GitOutcome {
    let mut command = Command::new("git");
    command
        .arg("-c")
        .arg("core.fsmonitor=false")
        .args(args)
        .current_dir(root)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .stdin(Stdio::null());
    if let Some(index) = index {
        command.env("GIT_INDEX_FILE", index);
    }
    match command.output() {
        Ok(output) => GitOutcome {
            ok: output.status.success(),
            out: String::from_utf8_lossy(&output.stdout).into_owned(),
            err: String::from_utf8_lossy(&output.stderr).into_owned(),
        },
        Err(e) => GitOutcome {
            ok: false,
            out: String::new(),
            err: format!("could not start git: {e}"),
        },
    }
}

/// First line of `err`, or a stand-in when the message is empty.
fn reason(err: &str, fallback: &str) -> String {
    err.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or(fallback)
        .to_string()
}

/// Workspace-relative form of a path, which is what every git pathspec here is
/// given. A path outside the root is refused rather than passed to git as
/// something like `../secrets/id_rsa`, which would read outside the project.
fn relative<'a>(root: &Path, full: &'a Path) -> Option<&'a str> {
    full.strip_prefix(root)
        .ok()
        .and_then(|rel| rel.to_str())
        .filter(|rel| !rel.is_empty())
}

/// The git pathspecs for a set of absolute paths.
///
/// With `drop_ignored`, a path the repository ignores is left out: build
/// output is not this repository's history to keep, and skipping the whole
/// turn because one file of it was ignored would be worse than skipping just
/// that file. `check-ignore` exits 0 when some path matched and 1 when none
/// did — both are answers rather than failures, so its exit status is not
/// consulted. A path outside the root is an error in both modes, because the
/// alternative is handing git something like `../secrets/id_rsa`.
fn workspace_paths<'a>(
    root: &Path,
    paths: &'a [PathBuf],
    drop_ignored: bool,
) -> Result<Vec<&'a str>, String> {
    let mut relatives: Vec<&str> = Vec::new();
    for full in paths {
        match relative(root, full) {
            Some(rel) => relatives.push(rel),
            None => return Err("a changed file is outside the workspace root".to_string()),
        }
    }
    if !drop_ignored || relatives.is_empty() {
        return Ok(relatives);
    }
    let mut probe = vec!["check-ignore", "--"];
    probe.extend(relatives.iter().copied());
    let listing = git(root, &probe, None);
    let dropped: Vec<&str> = listing
        .out
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    relatives.retain(|rel| !dropped.contains(rel));
    Ok(relatives)
}

/// `true` when this directory is a git working tree we could commit into.
pub fn available(root: &Path) -> bool {
    git(root, &["rev-parse", "--git-dir"], None).ok
}

/// Why the guard cannot run here, or `None` when it can. `/rewind` calls this
/// when a check fails, so the line it prints names the real reason — a plain
/// directory, a repository nobody has committed to yet, a branch that does not
/// exist because the agent has written nothing this session — instead of
/// implying it looked and found nothing.
pub fn unavailable_reason(root: &Path) -> Option<String> {
    if !available(root) {
        return Some("this is not a git repository".to_string());
    }
    let tip = git(root, &["rev-parse", "--verify", "-q", CKPT_REF], None);
    if !tip.ok {
        return Some(
            "no checkpoint branch yet, because the agent has not written a file this session"
                .to_string(),
        );
    }
    None
}

/// Commit what the agent wrote during one turn, on `xencode/ckpt`.
///
/// `Ok(Some(sha))` — a commit was made. `Ok(None)` — nothing to record, either
/// no path survived filtering or the tree is byte-for-byte what the previous
/// checkpoint already held. `Err(reason)` — the branch was not written, and the
/// caller keeps the in-memory rewind as the only undo.
///
/// Never called with a caller-supplied tree: the parent is always the current
/// checkpoint tip, so turns chain in the order they ran.
pub fn write_turn(root: &Path, turn: usize, paths: &[PathBuf]) -> Result<Option<String>, String> {
    let relatives = workspace_paths(root, paths, true)?;
    if relatives.is_empty() {
        return Ok(None);
    }

    // Parent for the new commit: the existing tip, or `HEAD` the first time.
    let tip = git(root, &["rev-parse", "--verify", "-q", CKPT_REF], None);
    let base = if tip.ok {
        tip.out.trim().to_string()
    } else {
        let head = git(root, &["rev-parse", "--verify", "-q", "HEAD"], None);
        if !head.ok {
            return Err(
                "this repository has no commits yet, so there is nothing to base a checkpoint on"
                    .to_string(),
            );
        }
        head.out.trim().to_string()
    };

    // A private index: `read-tree` fills it, `update-index` stages into it, and
    // it is deleted whatever happens. The user's index is never opened.
    let index = temp_index()?;
    let outcome = stage_turn(root, &index, &base, &relatives);
    let _ = std::fs::remove_file(&index);
    let tree = outcome?;
    if Some(tree.as_str()) == base_tree(root, &base).as_deref() {
        // Nothing the agent wrote differs from the last checkpoint — the file
        // was saved back as it was. No commit, so the branch stays a history of
        // turns that actually changed something.
        return Ok(None);
    }
    let message = format!("xencode checkpoint: turn {}", turn + 1);
    let mut args: Vec<&str> = COMMIT_CONFIG.to_vec();
    args.extend([
        "commit-tree",
        tree.as_str(),
        "-p",
        base.as_str(),
        "-m",
        message.as_str(),
    ]);
    let commit = git(root, &args, None);
    if !commit.ok {
        return Err(reason(
            &commit.err,
            "git could not write the checkpoint commit",
        ));
    }
    let sha = commit.out.trim().to_string();
    let update = git(root, &["update-ref", CKPT_REF, sha.as_str()], None);
    if !update.ok {
        return Err(reason(&update.err, "git refused the checkpoint ref update"));
    }
    Ok(Some(sha))
}

/// `read-tree` the base, stage the agent's paths into the private index, and
/// `write-tree`. Returns the new tree's object id.
fn stage_turn(root: &Path, index: &Path, base: &str, paths: &[&str]) -> Result<String, String> {
    let read = git(root, &["read-tree", base], Some(index));
    if !read.ok {
        return Err(reason(&read.err, "git could not read the checkpoint tree"));
    }
    // Files the agent wrote are added; files the agent deleted (or that have
    // since vanished) are removed from the index so the commit records the
    // deletion rather than keeping a stale blob.
    let (present, gone): (Vec<&str>, Vec<&str>) = paths
        .iter()
        .copied()
        .partition(|rel| root.join(rel).is_file());
    if !present.is_empty() {
        let mut args = vec!["update-index", "--add", "--"];
        args.extend(present.iter().copied());
        let staged = git(root, &args, Some(index));
        if !staged.ok {
            return Err(reason(&staged.err, "git could not stage the checkpoint"));
        }
    }
    if !gone.is_empty() {
        let mut args = vec!["update-index", "--force-remove", "--"];
        args.extend(gone.iter().copied());
        let removed = git(root, &args, Some(index));
        if !removed.ok {
            return Err(reason(&removed.err, "git could not stage the checkpoint"));
        }
    }
    let written = git(root, &["write-tree"], Some(index));
    if !written.ok {
        return Err(reason(
            &written.err,
            "git could not write the checkpoint tree",
        ));
    }
    Ok(written.out.trim().to_string())
}

/// The tree a commit holds, so an unchanged turn can be skipped. An unborn or
/// unreadable base has no tree to compare against, which `None` expresses; the
/// caller then just makes the commit.
fn base_tree(root: &Path, base: &str) -> Option<String> {
    let out = git(root, &["rev-parse", &format!("{base}^{{tree}}")], None);
    out.ok.then(|| out.out.trim().to_string())
}

/// A unique path for the throwaway index. `mkstemp`-style: the first name that
/// can be created empty wins, so two xencode runs in one process, or two
/// processes sharing `/tmp`, cannot write the same index.
fn temp_index() -> Result<PathBuf, String> {
    let dir = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for bump in 0..64u128 {
        let candidate = dir.join(format!(
            "xencode-ckpt-{}-{}.idx",
            std::process::id(),
            stamp + bump
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(_) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("could not create a scratch git index: {e}")),
        }
    }
    Err("could not create a scratch git index".to_string())
}

/// Which of `paths` a person has changed since the checkpoint branch recorded
/// them — the check that makes `/rewind` safe in a repository the user is
/// actively editing.
///
/// It compares two trees rather than asking `git diff` about the working tree,
/// and that choice is the whole point. `git diff <commit> -- <path>` only
/// consults the working tree for paths the *user's* index already knows, so a
/// file the agent created (and nobody has committed yet) comes back as deleted
/// even when it is sitting there unchanged — which would refuse every rewind
/// that touches a new file. Here the checkpoint tip is read into a scratch
/// index, the current bytes of exactly these paths are staged beside it, and
/// the two resulting trees are compared. That reports a file edited after the
/// agent wrote it and a file the agent created that has since been deleted, and
/// says nothing about a file nobody has touched since.
///
/// Paths the repository ignores are not in the checkpoint and cannot be
/// compared, so they are simply not guarded.
///
/// `Err` when the branch is not there to compare against.
pub fn human_edits(root: &Path, paths: &[PathBuf]) -> Result<Vec<String>, String> {
    let relatives = workspace_paths(root, paths, true)?;
    if relatives.is_empty() {
        return Ok(Vec::new());
    }
    let tip = git(root, &["rev-parse", "--verify", "-q", CKPT_REF], None);
    if !tip.ok {
        return Err(reason(
            &tip.err,
            "no checkpoint branch exists yet, so nothing can be compared against it",
        ));
    }
    let tip = tip.out.trim().to_string();
    let recorded = base_tree(root, &tip)
        .ok_or_else(|| "the checkpoint branch does not point at a commit".to_string())?;

    let index = temp_index()?;
    let outcome = stage_turn(root, &index, &tip, &relatives);
    let _ = std::fs::remove_file(&index);
    let current = outcome?;
    if current == recorded {
        return Ok(Vec::new());
    }
    let diff = git(
        root,
        &[
            "diff-tree",
            "-r",
            "--name-only",
            recorded.as_str(),
            current.as_str(),
        ],
        None,
    );
    if !diff.ok {
        return Err(reason(
            &diff.err,
            "git could not compare the checkpoint trees",
        ));
    }
    Ok(diff
        .out
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway repository, created here and deleted at the end of the test.
    /// Nothing in this module is exercised against the xencode working tree:
    /// these tests write commits, and doing that in the user's repository would
    /// be the exact side effect the design forbids.
    struct Repo {
        root: PathBuf,
    }

    impl Repo {
        fn new(initial_commit: bool) -> Repo {
            static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let root = std::env::temp_dir()
                .join(format!("xencode-ckptgit-test-{}-{n}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            let repo = Repo { root };
            repo.git(&["init", "-q", "."]);
            repo.git(&["config", "user.name", "tester"]);
            repo.git(&["config", "user.email", "tester@example.invalid"]);
            if initial_commit {
                std::fs::write(repo.at("keep.txt"), b"kept\n").unwrap();
                std::fs::write(repo.at("a.txt"), b"original\n").unwrap();
                repo.git(&["add", "-A"]);
                repo.git(&["commit", "-qm", "base"]);
            }
            repo
        }

        fn at(&self, rel: &str) -> PathBuf {
            self.root.join(rel)
        }

        fn git(&self, args: &[&str]) -> String {
            let out = Command::new("git")
                .args(args)
                .current_dir(&self.root)
                .output()
                .unwrap_or_else(|e| panic!("git {args:?} failed to start: {e}"));
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).into_owned()
        }

        /// Read one file out of the checkpoint tip, the only way a test can see
        /// what the branch holds without disturbing the working tree.
        fn ckpt_file(&self, rel: &str) -> Option<String> {
            let out = Command::new("git")
                .args(["show", &format!("{CKPT_REF}:{rel}")])
                .current_dir(&self.root)
                .output()
                .unwrap();
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
        }

        fn has_ckpt(&self) -> bool {
            Command::new("git")
                .args(["rev-parse", "--verify", "-q", CKPT_REF])
                .current_dir(&self.root)
                .output()
                .unwrap()
                .status
                .success()
        }
    }

    impl Drop for Repo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn a_turn_is_committed_on_the_checkpoint_branch_without_touching_the_users_history() {
        let repo = Repo::new(true);
        let head_before = repo.git(&["rev-parse", "HEAD"]).trim().to_string();
        let branch_before = repo
            .git(&["rev-parse", "--abbrev-ref", "HEAD"])
            .trim()
            .to_string();
        std::fs::write(repo.at("a.txt"), b"agent wrote this\n").unwrap();

        let sha = write_turn(&repo.root, 0, &[repo.at("a.txt")])
            .unwrap()
            .expect("a turn that changed a file makes a commit");
        assert!(!sha.is_empty());
        assert!(repo.has_ckpt());
        assert_eq!(
            repo.ckpt_file("a.txt").as_deref(),
            Some("agent wrote this\n"),
            "the checkpoint records what the agent wrote"
        );
        assert_eq!(
            repo.git(&["rev-parse", "HEAD"]).trim(),
            head_before,
            "the user's HEAD did not move"
        );
        assert_eq!(
            repo.git(&["rev-parse", "--abbrev-ref", "HEAD"]).trim(),
            branch_before,
            "the user is still on their own branch"
        );
        assert_eq!(
            repo.git(&["diff", "--cached", "--name-only"]),
            "",
            "the user's index was not staged into"
        );
        assert_eq!(
            std::fs::read_to_string(repo.at("a.txt")).unwrap(),
            "agent wrote this\n",
            "the working tree is exactly as the agent left it"
        );
        // Attributed to xencode and unsigned, so scratch history never looks
        // like the user's work.
        let author = repo.git(&["log", "-1", "--format=%an|%ae", CKPT_REF]);
        assert_eq!(author.trim(), "xencode|xencode@localhost");
    }

    #[test]
    fn turns_chain_so_the_branch_is_a_history_of_writes() {
        let repo = Repo::new(true);
        std::fs::write(repo.at("a.txt"), b"first\n").unwrap();
        write_turn(&repo.root, 0, &[repo.at("a.txt")]).unwrap();
        std::fs::write(repo.at("keep.txt"), b"second\n").unwrap();
        write_turn(&repo.root, 1, &[repo.at("keep.txt")]).unwrap();

        let subjects = repo
            .git(&["log", "--format=%s", CKPT_REF])
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert_eq!(
            subjects,
            vec![
                "xencode checkpoint: turn 2",
                "xencode checkpoint: turn 1",
                "base"
            ],
            "each turn parents the last, and the user's base commit is underneath"
        );
    }

    #[test]
    fn a_git_ignored_path_is_never_recorded() {
        let repo = Repo::new(true);
        std::fs::write(repo.at(".gitignore"), b"target/\n").unwrap();
        std::fs::create_dir_all(repo.at("target")).unwrap();
        std::fs::write(repo.at("target/blob.bin"), b"build output\n").unwrap();

        assert_eq!(
            write_turn(&repo.root, 0, &[repo.at("target/blob.bin")]).unwrap(),
            None,
            "an ignored file is not this repository's history to keep"
        );
        assert!(
            !repo.has_ckpt(),
            "a turn whose only write was ignored creates no branch"
        );

        // The same turn with one real file alongside it still records the real
        // file, rather than losing the whole checkpoint to the ignored one.
        std::fs::write(repo.at("a.txt"), b"real edit\n").unwrap();
        write_turn(
            &repo.root,
            0,
            &[repo.at("target/blob.bin"), repo.at("a.txt")],
        )
        .unwrap();
        assert_eq!(repo.ckpt_file("a.txt").as_deref(), Some("real edit\n"));
        assert_eq!(repo.ckpt_file("target/blob.bin"), None);
    }

    #[test]
    fn saving_a_file_back_as_it_was_makes_no_commit() {
        let repo = Repo::new(true);
        std::fs::write(repo.at("a.txt"), b"agent\n").unwrap();
        write_turn(&repo.root, 0, &[repo.at("a.txt")]).unwrap();
        let commits = repo
            .git(&["rev-list", "--count", CKPT_REF])
            .trim()
            .to_string();

        std::fs::write(repo.at("a.txt"), b"agent\n").unwrap();
        assert_eq!(
            write_turn(&repo.root, 1, &[repo.at("a.txt")]).unwrap(),
            None,
            "a turn that changed nothing against the last checkpoint is not a step"
        );
        assert_eq!(
            repo.git(&["rev-list", "--count", CKPT_REF]).trim(),
            commits,
            "and it added no commit"
        );
    }

    #[test]
    fn an_edit_made_by_hand_after_the_checkpoint_is_found_before_it_is_overwritten() {
        let repo = Repo::new(true);
        std::fs::write(repo.at("a.txt"), b"agent\n").unwrap();
        write_turn(&repo.root, 0, &[repo.at("a.txt")]).unwrap();

        assert_eq!(
            human_edits(&repo.root, &[repo.at("a.txt")]).unwrap(),
            Vec::<String>::new(),
            "the file still matches what the agent wrote, so rewinding is safe"
        );

        std::fs::write(repo.at("a.txt"), b"a person edited this\n").unwrap();
        assert_eq!(
            human_edits(&repo.root, &[repo.at("a.txt")]).unwrap(),
            vec!["a.txt".to_string()],
            "a hand edit after the agent's write is reported, not clobbered"
        );

        // A file the agent created that a person has since deleted is also an
        // interleaved edit — rewinding would delete something already gone.
        std::fs::write(repo.at("new.txt"), b"agent made me\n").unwrap();
        write_turn(&repo.root, 1, &[repo.at("new.txt")]).unwrap();
        assert_eq!(
            human_edits(&repo.root, &[repo.at("new.txt")]).unwrap(),
            Vec::<String>::new()
        );
        std::fs::remove_file(repo.at("new.txt")).unwrap();
        assert_eq!(
            human_edits(&repo.root, &[repo.at("new.txt")]).unwrap(),
            vec!["new.txt".to_string()]
        );
    }

    #[test]
    fn paths_outside_the_workspace_are_refused_rather_than_passed_to_git() {
        let repo = Repo::new(true);
        let outside = repo
            .root
            .parent()
            .unwrap()
            .join("xencode-ckptgit-outside.txt");
        let err = write_turn(&repo.root, 0, &[outside]).unwrap_err();
        assert!(
            err.contains("outside the workspace"),
            "unexpected error: {err}"
        );
        assert!(!repo.has_ckpt());
    }

    #[test]
    fn a_repository_with_no_commits_yet_cannot_be_given_a_checkpoint() {
        let repo = Repo::new(false);
        std::fs::write(repo.at("a.txt"), b"agent\n").unwrap();
        let err = write_turn(&repo.root, 0, &[repo.at("a.txt")]).unwrap_err();
        assert!(
            err.contains("no commits yet"),
            "an unborn HEAD must be explained, not guessed at: {err}"
        );
        assert!(!repo.has_ckpt());
    }

    #[test]
    fn the_guard_explains_itself_when_there_is_nothing_to_compare_against() {
        let repo = Repo::new(true);
        assert!(
            unavailable_reason(&repo.root).is_some(),
            "before the first write there is no branch, and the guard says so"
        );
        std::fs::write(repo.at("a.txt"), b"agent\n").unwrap();
        write_turn(&repo.root, 0, &[repo.at("a.txt")]).unwrap();
        assert!(unavailable_reason(&repo.root).is_none());

        // Not a repository at all: a plain directory gets an honest reason.
        let plain =
            std::env::temp_dir().join(format!("xencode-ckptgit-plain-{}", std::process::id()));
        std::fs::create_dir_all(&plain).unwrap();
        assert!(unavailable_reason(&plain).is_some());
        assert!(write_turn(&plain, 0, &[plain.join("a.txt")]).is_err());
        let _ = std::fs::remove_dir_all(&plain);
    }

    #[test]
    fn a_deletion_the_agent_made_is_recorded_as_a_deletion() {
        let repo = Repo::new(true);
        std::fs::remove_file(repo.at("a.txt")).unwrap();
        write_turn(&repo.root, 0, &[repo.at("a.txt")]).unwrap();
        assert_eq!(
            repo.ckpt_file("a.txt"),
            None,
            "the checkpoint holds the file gone, not the file as it was"
        );
        assert_eq!(
            human_edits(&repo.root, &[repo.at("a.txt")]).unwrap(),
            Vec::<String>::new(),
            "and a missing file that the checkpoint also lacks is not a hand edit"
        );
    }
}
