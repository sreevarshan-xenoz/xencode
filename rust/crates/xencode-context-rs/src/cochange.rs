//! What this repository's own commit history says about which files belong
//! together — the co-change and recency terms retrieval can spend (GH-3).
//!
//! Dependency edges say what a file *imports now*; history says what a person
//! changes *alongside* it, which catches the pair that talks over a boundary
//! (a handler and the error type it returns, a migration and the schema it
//! matches) and never appears in either file's import list.
//!
//! The mining is one `git log` pass over the whole history, so it runs when the
//! index is built and not on every turn; the cost is measured and reported by
//! `xencode history`, which times the same command.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// A commit that touches at least this many files is treated as a reformat, a
/// squash or a vendor drop, and contributes nothing. This is the noise the
/// mining literature warns about: one such commit otherwise teaches every file
/// in the repository to be related to every other.
pub const MASS_COMMIT_FILES: usize = 25;
/// Partners kept per file. Beyond this the list is a popularity contest, and
/// the pairs that carry information are the strong ones at the head.
pub const TOP_PARTNERS: usize = 16;
/// What a file that has historically changed with one of this turn's seeds is
/// worth. Between a path segment (+5) and a symbol hit (+8): a real signal, but
/// weaker than the query naming the file or something inside it.
pub const COCHANGE_BONUS: u64 = 5;
/// A file that appears in at least one commit in this many is a hub: it is
/// edited alongside everything, so its pairs carry no information about what
/// belongs with what. On this repository — 782 commits that counted, out of 817
/// reachable — the rule names exactly four files: `README.md` (163 of them),
/// `NEXT_PLAN_TASKS.md` (146), `rust/crates/xencode-tui-rs/src/app.rs` (127) and
/// `CHANGELOG.md` (125). Those are the files a complete change has to touch; the
/// fifth most-edited file, `NEXT_PLAN.md` at 79, stays in. The median elsewhere
/// in this history is one commit per file.
///
/// The rule is what the first measurement of this term without it looked like:
/// `README.md` scored 575 against the right file's 157, and the gold set's
/// recall@1 fell from 0.680 to 0.240.
pub const HUB_COMMIT_DIVISOR: u32 = 8;
/// What a file committed alongside the newest work adds on its own. Deliberately
/// small: "this file was committed recently" is true of a handful of files at a
/// time and says nothing about the question that was asked.
pub const RECENCY_BONUS: u64 = 2;
/// How far back the recency term reaches, measured against the newest commit in
/// the mined range rather than the clock, so a repository that has been quiet
/// for a year still favours its most recently worked-on files.
pub const RECENCY_WINDOW_DAYS: u64 = 14;
/// Commits mined, newest first. The cap is what keeps a decades-old monorepo's
/// first index build from reading every commit it has ever made.
pub const COMMIT_LOG_LIMIT: u32 = 5000;
/// The marker that opens a commit's header line in the log this module parses.
/// git escapes control characters in paths by default, so a raw byte here cannot
/// collide with a file name.
pub const LOG_MARKER: char = '\u{1}';

/// One file's history, as the index stores it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileCommits {
    /// Files changed in the same commit, strongest first, then by path so the
    /// order is reproducible from the same history. Empty for a hub (see
    /// [`HUB_COMMIT_DIVISOR`]), and never naming a hub.
    #[serde(default)]
    pub partners: Vec<(String, u32)>,
    /// Commits this file appeared in, excluding the ones too large to count.
    #[serde(default)]
    pub commits: u32,
    /// Epoch seconds of the newest commit that touched it.
    #[serde(default)]
    pub last_commit: u64,
    /// Whether this file is edited alongside so much of the repository that its
    /// pairs say nothing. Set by the miner, not by the caller.
    #[serde(default)]
    pub hub: bool,
}

/// Path → its history, for the files that appear in the log at all. A wrapper
/// over the map rather than the map itself, so the questions retrieval asks of a
/// history — is this file recent? — live next to the data they are about.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CommitHistory(BTreeMap<String, FileCommits>);

impl std::ops::Deref for CommitHistory {
    type Target = BTreeMap<String, FileCommits>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<BTreeMap<String, FileCommits>> for CommitHistory {
    fn from(entries: BTreeMap<String, FileCommits>) -> Self {
        Self(entries)
    }
}

impl CommitHistory {
    /// The newest commit in the mined range, which is what recency is measured
    /// against. Zero when no commit carried a date.
    pub fn newest_commit(&self) -> u64 {
        self.0.values().map(|h| h.last_commit).max().unwrap_or(0)
    }

    /// Whether this file was committed within [`RECENCY_WINDOW_DAYS`] of the
    /// newest commit. A file with no date, or a history with no dates at all, is
    /// not recent — it is unknown.
    pub fn is_recent(&self, path: &str) -> bool {
        let newest = self.newest_commit();
        match self.0.get(path) {
            Some(entry) if newest > 0 => {
                newest.saturating_sub(entry.last_commit) <= RECENCY_WINDOW_DAYS * 86_400
            }
            _ => false,
        }
    }
}

/// Parse `git log --pretty=format:<marker>%ct --name-only` output into
/// [`CommitHistory`]. Pure, so the shaping decisions — which commits are noise,
/// which files are hubs, how many partners survive — are testable without a
/// repository.
pub fn parse_commit_log(
    output: &str,
    marker: char,
    mass_commit_files: usize,
    top_partners: usize,
) -> CommitHistory {
    let mut pairs: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut commits: BTreeMap<String, u32> = BTreeMap::new();
    let mut last: BTreeMap<String, u64> = BTreeMap::new();
    let mut files: Vec<String> = Vec::new();
    let mut stamp: u64 = 0;
    let mut counted: u32 = 0;

    for line in output.lines() {
        if let Some(rest) = line.strip_prefix(marker) {
            if record_commit(
                &files,
                stamp,
                mass_commit_files,
                &mut pairs,
                &mut commits,
                &mut last,
            ) {
                counted += 1;
            }
            files.clear();
            stamp = rest.trim().parse().unwrap_or(0);
            continue;
        }
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            files.push(unquote_path(trimmed));
        }
    }
    if record_commit(
        &files,
        stamp,
        mass_commit_files,
        &mut pairs,
        &mut commits,
        &mut last,
    ) {
        counted += 1;
    }

    // A file is a hub if it was edited alongside at least one commit in
    // `HUB_COMMIT_DIVISOR`. The floor of four is what keeps a young repository
    // from having every file called a hub — with a dozen commits the share alone
    // would silence the term exactly where there is too little history to judge.
    let hub_floor = (counted / HUB_COMMIT_DIVISOR).max(4);
    let hubs: BTreeSet<String> = commits
        .iter()
        .filter(|(_, n)| **n >= hub_floor)
        .map(|(path, _)| path.clone())
        .collect();

    let mut partner_of: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
    for ((a, b), count) in pairs {
        partner_of
            .entry(a.clone())
            .or_default()
            .push((b.clone(), count));
        partner_of.entry(b).or_default().push((a, count));
    }
    for path in &hubs {
        partner_of.remove(path);
    }
    for partners in partner_of.values_mut() {
        partners.retain(|(partner, _)| !hubs.contains(partner.as_str()));
        partners.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
        partners.truncate(top_partners);
    }

    let mut paths: BTreeSet<String> = BTreeSet::new();
    paths.extend(commits.keys().cloned());
    paths.extend(partner_of.keys().cloned());
    CommitHistory(
        paths
            .into_iter()
            .map(|path| {
                let history = FileCommits {
                    partners: partner_of.remove(&path).unwrap_or_default(),
                    commits: commits.remove(&path).unwrap_or(0),
                    last_commit: last.remove(&path).unwrap_or(0),
                    hub: hubs.contains(&path),
                };
                (path, history)
            })
            .collect(),
    )
}

/// Fold one commit's file list into the running totals, and report whether it
/// was folded in at all. An empty list is the leading header of the log, and a
/// commit at or above `mass_commit_files` is a reformat or a vendor drop:
/// neither teaches anything, and neither counts as a commit for the hub rule.
fn record_commit(
    files: &[String],
    stamp: u64,
    mass_commit_files: usize,
    pairs: &mut BTreeMap<(String, String), u32>,
    commits: &mut BTreeMap<String, u32>,
    last: &mut BTreeMap<String, u64>,
) -> bool {
    if files.is_empty() || files.len() >= mass_commit_files {
        return false;
    }
    for path in files {
        *commits.entry(path.clone()).or_insert(0) += 1;
        let entry = last.entry(path.clone()).or_insert(0);
        if stamp > *entry {
            *entry = stamp;
        }
    }
    for (i, a) in files.iter().enumerate() {
        for b in files.iter().skip(i + 1) {
            if a == b {
                continue;
            }
            let key = if a <= b {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            };
            *pairs.entry(key).or_insert(0) += 1;
        }
    }
    true
}

/// Undo git's octal quoting (`\303\251`) for the paths it chooses to escape.
fn unquote_path(raw: &str) -> String {
    let bytes = raw.as_bytes();
    if !bytes.contains(&b'\\') {
        return raw.to_string();
    }
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && i + 3 < bytes.len()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
        {
            let octal = &bytes[i + 1..i + 4];
            let value = u8::from_str_radix(&String::from_utf8_lossy(octal), 8).unwrap_or(0);
            out.push(value);
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// One `git log` pass over the repository's history. `None` when `root` is not a
/// git repository or git refuses to answer — the same "nothing to spend" case an
/// empty map means.
pub fn mine_commit_history(root: &Path) -> Option<CommitHistory> {
    if !crate::gitinfo::is_git_repo(root) {
        return None;
    }
    let limit = format!("--max-count={COMMIT_LOG_LIMIT}");
    let format = format!("--pretty=format:{LOG_MARKER}%ct");
    let output = crate::gitinfo::git_stdout(
        root,
        &["log", &limit, "--no-merges", &format, "--name-only"],
    )
    .ok()?;
    Some(parse_commit_log(
        &output,
        LOG_MARKER,
        MASS_COMMIT_FILES,
        TOP_PARTNERS,
    ))
}

/// Which revision of the mining rules wrote `history.json`. The version travels
/// with the data because a history mined by an older build applied rules this
/// one depends on — it is why an unchanged HEAD cannot reuse `history.json` from
/// a build that had no hub rule.
pub const HISTORY_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoryFile {
    version: u32,
    files: CommitHistory,
}

/// Read the mined history for a project: `Default` when there is no file, the
/// file is unreadable, or it was written by a build with different rules.
pub fn load_history(xencode_dir: &Path) -> CommitHistory {
    crate::index::read_json::<HistoryFile>(&crate::index::history_json_path(xencode_dir))
        .filter(|file| file.version == HISTORY_VERSION)
        .map(|file| file.files)
        .unwrap_or_default()
}

/// Write the mined history, stamped with the rules that produced it.
pub fn save_history(xencode_dir: &Path, files: &CommitHistory) -> std::io::Result<()> {
    crate::index::write_atomic(
        &crate::index::history_json_path(xencode_dir),
        &HistoryFile {
            version: HISTORY_VERSION,
            files: files.clone(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three commits: `a` with `b`, `a` with `b` and `c`, and `d` alone.
    fn log() -> String {
        format!(
            "{m}1000\n\na.rs\nb.rs\n{m}900\n\na.rs\nb.rs\nc.rs\n{m}800\n\nd.rs\n",
            m = LOG_MARKER
        )
    }

    fn parse(text: &str) -> CommitHistory {
        parse_commit_log(text, LOG_MARKER, MASS_COMMIT_FILES, TOP_PARTNERS)
    }

    #[test]
    fn files_changed_together_learn_about_each_other() {
        let h = parse(&log());
        let a = &h["a.rs"];
        assert_eq!(a.commits, 2, "a is in the first two, not the last");
        assert_eq!(a.partners[0], ("b.rs".to_string(), 2), "{a:?}");
        assert!(a.partners.contains(&("c.rs".to_string(), 1)), "{a:?}");
        // The pair is symmetric: b knows about a as strongly as a knows about b.
        assert_eq!(h["b.rs"].partners[0], ("a.rs".to_string(), 2));
        // A commit of one file makes no pair, but the file is still known.
        assert_eq!(h["d.rs"].commits, 1);
        assert!(h["d.rs"].partners.is_empty());
    }

    #[test]
    fn a_commit_that_touches_everything_teaches_nothing() {
        let mut big = format!("{m}1000\n\n", m = LOG_MARKER);
        for i in 0..MASS_COMMIT_FILES {
            big.push_str(&format!("f{i}.rs\n"));
        }
        big.push_str(&format!("{m}900\n\nx.rs\ny.rs\n", m = LOG_MARKER));
        let h = parse(&big);
        assert!(
            !h.contains_key("f0.rs"),
            "the noise commit is not even counted as a file that has history"
        );
        assert_eq!(h["x.rs"].partners, vec![("y.rs".to_string(), 1)]);
    }

    #[test]
    fn only_the_strongest_partners_survive() {
        let mut big = format!("{m}1000\n\nseed.rs\n", m = LOG_MARKER);
        for i in 0..TOP_PARTNERS + 5 {
            big.push_str(&format!("p{i}.rs\n"));
        }
        let h = parse_commit_log(&big, LOG_MARKER, 100, TOP_PARTNERS);
        assert_eq!(h["seed.rs"].partners.len(), TOP_PARTNERS);
        // The counts are all 1 here, so the order is by path: reproducible.
        assert_eq!(h["seed.rs"].partners[0].0, "p0.rs");
    }

    #[test]
    fn recency_is_measured_against_the_newest_commit() {
        let newest = 1_700_000_000u64;
        let just_inside = newest - (RECENCY_WINDOW_DAYS - 1) * 86_400;
        let just_outside = newest - (RECENCY_WINDOW_DAYS + 1) * 86_400;
        let text = format!(
            "{m}{newest}\n\nhottest.rs\npartner.rs\n\
             {m}{just_inside}\n\ninside.rs\n\
             {m}{just_outside}\n\noutside.rs\nhottest.rs\n",
            m = LOG_MARKER
        );
        let h = parse(&text);
        assert_eq!(h.newest_commit(), newest);
        assert!(h.is_recent("inside.rs"), "one day inside the window");
        assert!(!h.is_recent("outside.rs"), "one day past it");
        assert!(!h.is_recent("never.committed.rs"));
        // Being touched by an old commit too does not pull a file out of the
        // window: what counts is the newest commit that reached it.
        assert_eq!(h["hottest.rs"].commits, 2);
        assert!(h.is_recent("hottest.rs"), "{:?}", h["hottest.rs"]);
    }

    #[test]
    fn a_history_with_no_dates_gives_no_recency_signal() {
        let h = parse(&format!("{m}\n\na.rs\nb.rs\n", m = LOG_MARKER));
        assert_eq!(h.newest_commit(), 0, "nothing was dated");
        assert!(!h.is_recent("a.rs"), "unknown is not recent");
        assert_eq!(h["a.rs"].commits, 1, "the pair still counts");
    }

    /// Thirty-two commits, each of `hub.rs` plus one file of its own: the hub is
    /// in every commit, so it is edited alongside everything and pairs with
    /// nothing. 32/8 = the floor of four it clears many times over.
    #[test]
    fn a_file_committed_alongside_everything_pairs_with_nothing() {
        let mut text = String::new();
        for i in 0..32 {
            text.push_str(&format!("{m}1000\n\nhub.rs\nf{i}.rs\n", m = LOG_MARKER));
        }
        let h = parse(&text);
        assert!(h["hub.rs"].hub, "{:?}", h["hub.rs"]);
        assert!(
            h["hub.rs"].partners.is_empty(),
            "and cannot pull anything in"
        );
        assert!(
            h["f0.rs"].partners.is_empty(),
            "nor be pulled in by it: {:?}",
            h["f0.rs"]
        );
        assert_eq!(h["hub.rs"].commits, 32, "what it did is still known");
        assert!(!h["f0.rs"].hub, "one commit is not a hub");
    }

    /// The floor: in a repository with barely any history, being in every commit
    /// is not yet evidence of being always edited.
    #[test]
    fn three_commits_are_too_little_history_to_call_anything_a_hub() {
        let h = parse(&format!(
            "{m}3\n\nhub.rs\na.rs\n{m}2\n\nhub.rs\nb.rs\n{m}1\n\nhub.rs\nc.rs\n",
            m = LOG_MARKER
        ));
        assert_eq!(h["hub.rs"].commits, 3);
        assert!(!h["hub.rs"].hub, "3 of 3 commits, and still not a hub");
        assert_eq!(h["a.rs"].partners, vec![("hub.rs".to_string(), 1)]);
    }

    #[test]
    fn git_quoted_paths_are_decoded_back() {
        assert_eq!(unquote_path("caf\\303\\251.rs"), "café.rs");
        assert_eq!(unquote_path("plain.rs"), "plain.rs");
    }

    /// The parser against a real `git log`, in a scratch repository: the output
    /// shape this module assumes is git's, not a remembered one.
    #[test]
    fn the_log_git_actually_prints_parses() {
        let dir = scratch_dir("cochange");
        let git = |args: &[&str]| {
            let ran = std::process::Command::new("git")
                .current_dir(&dir)
                .args(args)
                .output()
                .unwrap();
            assert!(
                ran.status.success(),
                "{args:?}: {}",
                String::from_utf8_lossy(&ran.stderr)
            );
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "test@example.invalid"]);
        git(&["config", "user.name", "Test"]);
        std::fs::write(dir.join("a.rs"), "1").unwrap();
        std::fs::write(dir.join("b.rs"), "1").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "first"]);
        std::fs::write(dir.join("a.rs"), "2").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "second"]);

        let history = mine_commit_history(&dir).expect("a scratch repository has history");
        assert_eq!(history["a.rs"].commits, 2, "{:?}", history["a.rs"]);
        assert_eq!(history["a.rs"].partners[0].0, "b.rs");
        assert!(history["b.rs"].last_commit > 0, "just committed");
        assert!(
            history.is_recent("a.rs"),
            "the newest commit is seconds old"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_directory_that_is_not_a_repository_mines_nothing() {
        let dir = scratch_dir("not-a-repo");
        assert!(mine_commit_history(&dir).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    fn scratch_dir(kind: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("xencode-{kind}-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
