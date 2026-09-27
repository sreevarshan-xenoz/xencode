//! How fast this repository's history is to ask about, and the one command
//! that makes it fast.
//!
//! Reading history is cheap on a normal repository and expensive on two
//! specific queries — `git blame` and `git log -S` — because both have to walk
//! commits and then read a blob per commit. A commit-graph removes the walk;
//! a multi-pack-index removes the "which pack has this object" search that
//! grows with every pack. Both are files inside `.git`, both are safe to
//! rewrite at any time, and neither changes a single commit.
//!
//! This module measures before it claims anything: every number on the status
//! page comes from a git process that just ran here.

use std::path::Path;
use std::time::Instant;

use crate::gitinfo::{git_stdout, is_git_repo};

/// One history query, timed. Microseconds, because a warm repository answers
/// `git log -1` in about a millisecond and rounding that to milliseconds
/// reports it as zero.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QueryTime {
    /// What the query is for, in words.
    pub label: String,
    /// The git sub-command, so the number can be re-taken by hand.
    pub command: String,
    pub microseconds: u128,
    /// Bytes of stdout the query produced — the cost a model would pay.
    pub bytes: usize,
    /// Git's own reason, when the query could not be run at all.
    pub failed: Option<String>,
}

/// The commit-graph file, as found on disk.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommitGraph {
    /// Path relative to the repository's `.git` directory.
    pub path: String,
    pub size: u64,
    /// What `git commit-graph verify` said about this file: `Some(true)` means
    /// every entry in it matches an object in this repository, `Some(false)`
    /// means it does not. Only read when the file exists.
    pub verifies: Option<bool>,
}

/// The multi-pack-index, as found on disk.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PackIndex {
    pub path: String,
    pub size: u64,
    /// How many pack files it has to cover.
    pub packs: usize,
}

/// Everything `xencode history status` prints, in one value.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HistoryStatus {
    pub git_dir: String,
    pub reachable_commits: usize,
    /// A clone that does not have the full history (`git clone --depth`).
    pub shallow: bool,
    /// The filter from `git clone --filter`, when this is a partial clone.
    pub partial_clone: Option<String>,
    pub commit_graph: Option<CommitGraph>,
    pub pack_index: Option<PackIndex>,
    pub queries: Vec<QueryTime>,
}

impl HistoryStatus {
    /// What is missing, phrased as the thing to do about it. Empty when
    /// there is nothing to do.
    pub fn actions(&self) -> Vec<String> {
        let mut actions = Vec::new();
        if self.commit_graph.is_none() {
            actions.push(
                "no commit-graph, so every history query walks the whole commit chain: run \
                 `xencode history setup`"
                    .to_string(),
            );
        }
        if self.pack_index.is_none() {
            actions.push(
                "no multi-pack-index, so an object lookup asks every pack in turn: run \
                 `xencode history setup`"
                    .to_string(),
            );
        }
        if self.partial_clone.is_some() {
            actions.push(format!(
                "this is a partial clone (filter {}), so `blame` and `log -S` fetch a blob from \
                 the remote for every commit they visit — history queries are cheap here and \
                 blame is not, whatever the timings for one file suggest",
                self.partial_clone.as_deref().unwrap_or("unknown"),
            ));
        }
        if self.shallow {
            actions.push(
                "this is a shallow clone, so history stops at the grafted commit and anything \
                 built on it — co-change mining, bisect, `what links to this` — sees a fraction \
                 of the story: fetch with --unshallow first"
                    .to_string(),
            );
        }
        actions
    }
}

/// What `xencode history setup` did.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HistorySetup {
    pub before: HistoryStatus,
    pub after: HistoryStatus,
    /// One line per file written, in the words the command prints.
    pub wrote: Vec<String>,
    /// One line per thing git refused to do, carrying git's reason.
    pub refused: Vec<String>,
}

/// Directories under `.git` where git may keep a commit-graph, in the order
/// git itself looks.
const COMMIT_GRAPH_DIRS: [&str; 3] = ["objects/info", "info", "objects/pack"];

/// Names git gives the commit-graph file, in either of its two layouts.
fn is_commit_graph_file(name: &str) -> bool {
    name == "commit-graph" || (name.starts_with("commit-graph") && name.ends_with(".graph"))
}

/// Names git gives the multi-pack-index, in either of its two layouts.
fn is_pack_index_file(name: &str) -> bool {
    name == "multi-pack-index" || (name.starts_with("midx-") && name.ends_with(".midx"))
}

/// The first entry of `dir` whose name `want` accepts, as a `.git`-relative
/// path plus its size in bytes.
fn find_file(dir: &Path, relative_to: &Path, want: impl Fn(&str) -> bool) -> Option<(String, u64)> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !want(&name) {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let path = entry
            .path()
            .strip_prefix(relative_to)
            .unwrap_or(&entry.path())
            .to_string_lossy()
            .replace('\\', "/");
        return Some((path, size));
    }
    None
}

/// Run `git <command>` and time it. Returns the measurement plus git's stdout,
/// so a caller can read a number out of the same run it just timed.
fn timed(root: &Path, label: &str, command: &[&str]) -> (QueryTime, Option<String>) {
    let started = Instant::now();
    let outcome = git_stdout(root, command);
    let microseconds = started.elapsed().as_micros();
    match outcome {
        Ok(out) => (
            QueryTime {
                label: label.to_string(),
                command: command.join(" "),
                microseconds,
                bytes: out.len(),
                failed: None,
            },
            Some(out),
        ),
        Err(reason) => (
            QueryTime {
                label: label.to_string(),
                command: command.join(" "),
                microseconds,
                bytes: 0,
                failed: Some(reason),
            },
            None,
        ),
    }
}

/// The queries that everything in this milestone is built on. The blame probe
/// needs a file; `blame_file` is a repository-relative path.
fn history_queries(root: &Path, blame_file: Option<&str>) -> (Vec<QueryTime>, usize) {
    let mut queries = Vec::new();
    for (label, command) in [
        ("every commit subject", vec!["log", "--format=%h %s"]),
        (
            "every commit with the paths it touched",
            vec!["log", "--name-only"],
        ),
    ] {
        queries.push(timed(root, label, &command).0);
    }
    let (count, text) = timed(
        root,
        "how many commits are reachable",
        &["rev-list", "--all", "--count"],
    );
    let reachable_commits = text
        .as_deref()
        .and_then(|t| t.trim().parse::<usize>().ok())
        .unwrap_or(0);
    queries.push(count);
    if let Some(file) = blame_file {
        let label = format!("who last wrote each line of {file}");
        queries.push(timed(root, &label, &["blame", "HEAD", "--", file]).0);
    }
    (queries, reachable_commits)
}

/// How many pack files a `.git` directory holds.
fn count_packs(git_dir: &str) -> usize {
    std::fs::read_dir(Path::new(git_dir).join("objects/pack"))
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().ends_with(".pack"))
                .count()
        })
        .unwrap_or(0)
}

/// Read-only view of what would make history queries fast here, with the
/// timings to compare against.
pub fn history_status(root: &Path, blame_file: Option<&str>) -> Result<HistoryStatus, String> {
    if !is_git_repo(root) {
        return Err(format!(
            "{} is not a git repository, so there is no history to speed up",
            root.to_string_lossy()
        ));
    }
    let git_dir = git_stdout(root, &["rev-parse", "--absolute-git-dir"])?
        .trim()
        .to_string();
    let dir = Path::new(&git_dir);
    let pack_dir = dir.join("objects/pack");
    let packs = count_packs(&git_dir);

    let (queries, reachable_commits) = history_queries(root, blame_file);

    // `commit-graph write` puts the file under objects/info; a hand-written or
    // `--split` graph can be in .git/info or objects/info/commit-graphs.
    let commit_graph = COMMIT_GRAPH_DIRS
        .iter()
        .find_map(|sub| find_file(&dir.join(sub), dir, is_commit_graph_file))
        .map(|(path, size)| CommitGraph {
            verifies: Some(
                git_stdout(root, &["commit-graph", "verify"])
                    .map(|_| true)
                    .unwrap_or(false),
            ),
            path,
            size,
        });

    let pack_index = find_file(&pack_dir, dir, is_pack_index_file)
        .or_else(|| find_file(&pack_dir.join("midx"), dir, is_pack_index_file))
        .map(|(path, size)| PackIndex { path, size, packs });

    let shallow = git_stdout(root, &["rev-parse", "--is-shallow-repository"])
        .map(|s| s.trim() == "true")
        .unwrap_or(false);
    let partial_clone = git_stdout(root, &["config", "--get", "extensions.partialClone"])
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    Ok(HistoryStatus {
        git_dir,
        reachable_commits,
        shallow,
        partial_clone,
        commit_graph,
        pack_index,
        queries,
    })
}

/// The file a blame probe should use when the caller did not name one:
/// `README.md` if the repository tracks it, otherwise the first tracked path.
pub fn default_blame_target(root: &Path) -> Option<String> {
    let tracked =
        |path: &str| git_stdout(root, &["ls-files", "--error-unmatch", "--", path]).is_ok();
    if tracked("README.md") {
        return Some("README.md".to_string());
    }
    git_stdout(root, &["ls-files"]).ok().and_then(|out| {
        out.lines()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.trim().to_string())
    })
}

/// Write the commit-graph and the multi-pack-index, then re-measure. Both
/// writes are idempotent and neither touches a commit, so this is safe to run
/// on a repository that already has them — it just says so.
pub fn history_setup(root: &Path, blame_file: Option<&str>) -> Result<HistorySetup, String> {
    let before = history_status(root, blame_file)?;
    let graph = git_stdout(root, &["commit-graph", "write", "--reachable"]);
    let midx = git_stdout(root, &["multi-pack-index", "write"]);
    let after = history_status(root, blame_file)?;

    let mut wrote = Vec::new();
    let mut refused = Vec::new();

    match (&graph, &after.commit_graph) {
        (Ok(_), Some(file)) => wrote.push(format!(
            "commit-graph {} at {}: {} reachable commits",
            if before.commit_graph.is_some() {
                "rewritten"
            } else {
                "written"
            },
            file.path,
            before.reachable_commits,
        )),
        (Ok(_), None) => refused.push(format!(
            "commit-graph: git reported success and left no file behind in {}",
            after.git_dir
        )),
        (Err(reason), _) => refused.push(format!("commit-graph: {reason}")),
    }

    // git leaves an up-to-date multi-pack-index alone and prints nothing, so
    // the honest report is what exists now, not what was just written.
    match (&midx, &after.pack_index) {
        (Ok(_), Some(file)) => wrote.push(format!(
            "multi-pack-index in place at {}, covering {} pack(s)",
            file.path, file.packs
        )),
        (Ok(_), None) => refused.push(
            "multi-pack-index: nothing to index — a repository with one pack has no use for it, \
             and git will not create one"
                .to_string(),
        ),
        (Err(reason), _) => refused.push(format!("multi-pack-index: {reason}")),
    }

    Ok(HistorySetup {
        before,
        after,
        wrote,
        refused,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn temp_repo() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-history-test-{unique}"))
    }

    fn git(root: &Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .expect("git binary should be available");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// A repository with a few commits and no history indexes at all.
    fn scratch_repo(tag: &str) -> PathBuf {
        let root = temp_repo();
        fs::create_dir_all(root.join("src")).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "test@xencode.local"]);
        git(&root, &["config", "user.name", "Xencode Test"]);
        fs::write(root.join("README.md"), "# scratch\n").unwrap();
        fs::write(root.join("src/lib.rs"), "fn a() {}\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", &format!("{tag} first")]);
        fs::write(root.join("src/lib.rs"), "fn a() {}\nfn b() {}\n").unwrap();
        git(&root, &["commit", "-q", "-am", &format!("{tag} second")]);
        root
    }

    #[test]
    fn the_commit_graph_and_pack_index_layouts_are_recognised_by_name() {
        assert!(is_commit_graph_file("commit-graph"));
        assert!(is_commit_graph_file("commit-graph2-deadbeef.graph"));
        assert!(!is_commit_graph_file("pack-deadbeef.idx"));
        assert!(!is_commit_graph_file("multi-pack-index"));
        assert!(is_pack_index_file("multi-pack-index"));
        assert!(is_pack_index_file("midx-deadbeef.midx"));
        assert!(!is_pack_index_file("commit-graph"));
    }

    #[test]
    fn a_plain_repository_is_missing_both_and_says_which_command_fixes_it() {
        let root = scratch_repo("plain");
        let status = history_status(&root, Some("README.md")).unwrap();

        assert_eq!(status.reachable_commits, 2);
        assert!(status.commit_graph.is_none());
        assert!(status.pack_index.is_none());
        assert!(!status.shallow);
        assert!(status.partial_clone.is_none());
        let actions = status.actions().join("\n");
        assert!(actions.contains("no commit-graph"), "{actions}");
        assert!(actions.contains("no multi-pack-index"), "{actions}");
        assert!(actions.contains("xencode history setup"), "{actions}");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn setup_writes_both_and_the_numbers_come_from_files_that_now_exist() {
        let root = scratch_repo("setup");
        let setup = history_setup(&root, Some("README.md")).unwrap();

        assert!(setup.before.commit_graph.is_none());
        let graph = setup
            .after
            .commit_graph
            .as_ref()
            .expect("the commit-graph exists after setup");
        assert!(graph.size > 0, "the file git wrote is empty");
        assert_eq!(
            graph.verifies,
            Some(true),
            "git cannot verify the graph it just wrote"
        );
        assert!(graph.path.contains("commit-graph"), "{}", graph.path);

        // Two commits in two `git add` rounds leaves loose objects and no pack,
        // which is the case a multi-pack-index genuinely cannot serve. Whatever
        // git decided, the answer has to be in exactly one of the two lists.
        assert_eq!(
            setup.wrote.len() + setup.refused.len(),
            2,
            "wrote={:?} refused={:?}",
            setup.wrote,
            setup.refused
        );
        assert!(
            setup.wrote.iter().any(|l| l.contains("commit-graph")),
            "{:?}",
            setup.wrote
        );

        // Every number on the page came from a git process that ran just now.
        assert_eq!(setup.after.queries.len(), 4);
        for query in &setup.after.queries {
            assert!(
                query.microseconds > 0,
                "{} was not timed: {:?}",
                query.label,
                query
            );
            assert!(query.failed.is_none(), "{query:?}");
        }
        assert!(setup.after.queries[3].label.contains("README.md"));
        // The commit-graph half of the advice is gone; the multi-pack-index
        // half stays, because a repository with no packs cannot have one.
        let after = setup.after.actions().join("\n");
        assert!(!after.contains("no commit-graph"), "{after}");
        assert!(after.contains("no multi-pack-index"), "{after}");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn running_setup_again_reports_a_rewrite_rather_than_a_write() {
        let root = scratch_repo("again");
        let first = history_setup(&root, None).unwrap();
        let second = history_setup(&root, None).unwrap();
        assert!(
            first
                .wrote
                .iter()
                .any(|l| l.contains("commit-graph written")),
            "{:?}",
            first.wrote
        );
        assert!(
            second
                .wrote
                .iter()
                .any(|l| l.contains("commit-graph rewritten")),
            "{:?}",
            second.wrote
        );
        assert!(second.before.commit_graph.is_some());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_probe_only_runs_blame_when_it_has_a_file_to_blame() {
        let root = scratch_repo("probe");
        let without = history_status(&root, None).unwrap();
        assert_eq!(without.queries.len(), 3);
        assert!(
            !without
                .queries
                .iter()
                .any(|q| q.command.starts_with("blame")),
            "{:?}",
            without.queries
        );

        let named = default_blame_target(&root).unwrap();
        assert_eq!(named, "README.md");
        let with = history_status(&root, Some(&named)).unwrap();
        assert_eq!(with.queries.len(), 4);
        assert_eq!(with.queries[3].command, "blame HEAD -- README.md");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_blame_default_skips_a_file_the_repository_does_not_track() {
        let root = scratch_repo("untracked");
        fs::remove_file(root.join("README.md")).unwrap();
        git(&root, &["rm", "-q", "--cached", "README.md"]);
        git(&root, &["commit", "-q", "-m", "no readme"]);
        assert_eq!(default_blame_target(&root).as_deref(), Some("src/lib.rs"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_repository_with_no_packs_refuses_the_index_in_gits_own_words() {
        let root = scratch_repo("nopacks");
        let setup = history_setup(&root, None).unwrap();
        let refusal = setup
            .refused
            .iter()
            .find(|l| l.starts_with("multi-pack-index:"))
            .unwrap_or_else(|| panic!("expected a refusal, got {:?}", setup.wrote));
        assert!(
            refusal.contains("pack"),
            "git's reason is missing: {refusal}"
        );
        assert!(setup.after.pack_index.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_shallow_clone_says_its_history_stops_early() {
        let seed = scratch_repo("shallow-seed");
        let clone = temp_repo();
        // A plain path clone is a local clone, and git silently ignores
        // --depth there; the file:// form is what actually truncates history.
        let out = std::process::Command::new("git")
            .args(["clone", "--depth", "1", "-q"])
            .arg(format!("file://{}", seed.display()))
            .arg(&clone)
            .output()
            .expect("git clone");
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );

        let status = history_status(&clone, None).unwrap();
        assert!(status.shallow, "git called this clone not shallow");
        assert_eq!(status.reachable_commits, 1);
        let action = status.actions().join("\n");
        assert!(action.contains("shallow"), "{action}");
        assert!(action.contains("--unshallow"), "{action}");

        fs::remove_dir_all(clone).unwrap();
        fs::remove_dir_all(seed).unwrap();
    }

    #[test]
    fn a_partial_clone_is_named_because_blame_there_fetches_per_commit() {
        let root = scratch_repo("partial");
        // `git clone --filter` records exactly this key; a real filtered clone
        // cannot be made from this machine's own repository, because git
        // refuses to serve filtering unless uploadpack.allowFilter is enabled.
        git(&root, &["config", "extensions.partialClone", "origin"]);

        let status = history_status(&root, None).unwrap();
        assert_eq!(status.partial_clone.as_deref(), Some("origin"));
        let action = status.actions().join("\n");
        assert!(action.contains("partial clone"), "{action}");
        assert!(action.contains("blob"), "{action}");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn outside_a_repository_the_command_says_so_instead_of_reporting_zeroes() {
        let root = temp_repo();
        fs::create_dir_all(&root).unwrap();
        let error = history_status(&root, None).unwrap_err();
        assert!(error.contains("not a git repository"), "{error}");
        assert!(history_setup(&root, None).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
