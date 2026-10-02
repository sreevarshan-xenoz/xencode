//! Release notes as a draft, built from what the repository actually says.
//!
//! Two sources, both already in the tree, and they disagree often enough to be
//! worth putting side by side: the commit messages between the last release and
//! here, and the `## [Unreleased]` block of `CHANGELOG.md` that the project
//! maintains as work lands. The draft is the changelog block with a coverage
//! report attached, because the failure mode this guards against is a release
//! going out with work in it that no reader was ever told about.
//!
//! Conventional-commit parsing is deliberately absent. The 900-odd messages here
//! are already prose — "Add `xencode perf` (QO-4): measure the hot paths, and
//! decline to judge a noisy run" — and a `feat:`/`fix:` prefix would add a
//! category to a subject that already carries one, in a dialect the project does
//! not write. Categorisation comes from the changelog's own `### Added` /
//! `### Fixed` headings instead, which are the categories the release notes use.
//!
//! Nothing here rewrites `CHANGELOG.md`. The output is a draft a person edits: it
//! goes to standard output, or to a path that must not already exist unless the
//! caller says otherwise.

use crate::gitinfo::git_stdout;
use regex::Regex;
use std::path::Path;

/// The heading whose block is the draft's body. Everything below the next `## `
/// is a previous release and is not up for release again.
pub const UNRELEASED_HEADING: &str = "## [Unreleased]";

/// The file the block is read from, relative to the repository root.
pub const CHANGELOG_FILE: &str = "CHANGELOG.md";

/// How many unexplained commits the draft names before saying how many more
/// there are. A repository with no tags puts its whole history in the range, and
/// eight hundred lines of subjects is not something an editor can work through —
/// the count, and the flag that narrows the range, are what they need from it.
pub const LIST_LIMIT: usize = 50;

/// `1 commit`, `870 commits`. The counts are read as prose, so they are not
/// printed as `1 commit(s)`.
fn plural<'a>(count: usize, one: &'a str, many: &'a str) -> &'a str {
    if count == 1 {
        one
    } else {
        many
    }
}

/// One entry of the unreleased block: its heading, its body, and the plan ids
/// the heading names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The changelog's own category, taken from the heading: `Added`, `Changed`,
    /// `Fixed`, or whatever word the file uses.
    pub category: String,
    /// The heading with the `### <category> — ` prefix removed.
    pub title: String,
    /// Every line under the heading, up to the next heading, as written.
    pub body: String,
    /// The ids this heading names, in the order they appear.
    pub ids: Vec<String>,
}

/// One commit in the range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commit {
    /// The full commit hash.
    pub hash: String,
    /// The subject line, exactly as git stores it.
    pub subject: String,
    /// The ids this subject names.
    pub ids: Vec<String>,
}

/// The draft, plus the two coverage lists that are the point of generating it.
#[derive(Debug, Clone)]
pub struct Draft {
    /// The upper end of the range, as given.
    pub to: String,
    /// The lower end, or `None` when the repository has nothing to take as a
    /// previous release.
    pub from: Option<String>,
    /// How the range was chosen, in the reader's terms rather than the shell's.
    pub range_note: String,
    /// The entries, in the order the changelog lists them.
    pub entries: Vec<Entry>,
    /// The commits, newest first, as git returns them.
    pub commits: Vec<Commit>,
    /// Commits nothing in the changelog accounts for: a release reader never
    /// heard about this work.
    pub uncovered: Vec<Commit>,
    /// Entries naming an id no commit in this range carries. Usually the work
    /// landed before the range; sometimes the id is simply wrong.
    pub unmatched: Vec<Entry>,
}

impl Draft {
    /// Ids that appear in a changelog entry and in a commit subject in the range.
    pub fn covered_ids(&self) -> Vec<String> {
        let entry_ids: Vec<&str> = self
            .entries
            .iter()
            .flat_map(|e| e.ids.iter().map(String::as_str))
            .collect();
        let mut shared: Vec<String> = self
            .commits
            .iter()
            .flat_map(|c| c.ids.iter())
            .filter(|id| entry_ids.contains(&id.as_str()))
            .cloned()
            .collect();
        shared.sort();
        shared.dedup();
        shared
    }
}

/// The id shape this project writes: `QO-4`, `M-6`, `EVd-8`, `L-12`.
///
/// Anchored so a hyphenated word carrying a digit at either end — `xencode-2`,
/// `--flag-1`, `sha1-256` — is not read as a plan id.
fn id_pattern() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?x) (?:^|[^A-Za-z0-9_-]) ([A-Z][A-Za-z]{0,4}-[0-9]{1,4}) (?:[^A-Za-z0-9_-]|$)",
        )
        .expect("the id pattern compiles")
    })
}

/// Every plan id named in a line of text, in order, without repeats.
pub fn ids_in(text: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for cap in id_pattern().captures_iter(text) {
        let id = &cap[1];
        if !found.iter().any(|seen| seen == id) {
            found.push(id.to_string());
        }
    }
    found
}

/// The entries of the `## [Unreleased]` block.
///
/// A file with no such heading is an empty answer, not an error: the block is
/// absent between releases, and the draft then says so and rests on the commits.
pub fn parse_unreleased(text: &str) -> Vec<Entry> {
    let mut entries = Vec::new();
    let mut inside = false;
    let mut current: Option<(String, String, Vec<String>)> = None;

    let flush = |current: &mut Option<(String, String, Vec<String>)>, entries: &mut Vec<Entry>| {
        if let Some((category, title, body)) = current.take() {
            let heading = format!("{category} — {title}");
            entries.push(Entry {
                ids: ids_in(&heading),
                category,
                title,
                body: body.join("\n").trim().to_string(),
            });
        }
    };

    for line in text.lines() {
        if line.trim_end() == UNRELEASED_HEADING {
            inside = true;
            continue;
        }
        if !inside {
            continue;
        }
        // The next `## ` is a previous release; the block ends there.
        if line.starts_with("## ") {
            break;
        }
        if let Some(rest) = line.strip_prefix("### ") {
            flush(&mut current, &mut entries);
            // `### Added — `QO-4`: title`, or `### Fixed — a sentence`.
            let (category, title) = match rest.split_once(" — ") {
                Some((category, title)) => (category.trim().to_string(), title.trim().to_string()),
                None => ("Other".to_string(), rest.trim().to_string()),
            };
            current = Some((category, title, Vec::new()));
            continue;
        }
        if let Some((_, _, body)) = current.as_mut() {
            body.push(line.to_string());
        }
    }
    flush(&mut current, &mut entries);
    entries
}

/// The most recent tag, or `None`. A repository with no tags has no previous
/// release to take as the start of a range, and saying that is the honest answer.
pub fn newest_tag(root: &Path) -> Option<String> {
    git_stdout(root, &["describe", "--tags", "--abbrev=0"])
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// The commits from `from` (exclusive) to `to` (inclusive). With no `from`, the
/// whole history reachable from `to`.
///
/// Merge commits are left out: their subject is `Merge branch …`, which describes
/// no work a reader would put in notes, and the commits being merged are in the
/// range anyway.
pub fn commits_between(root: &Path, from: Option<&str>, to: &str) -> Result<Vec<Commit>, String> {
    let range = match from {
        Some(from) if !from.is_empty() => format!("{from}..{to}"),
        _ => to.to_string(),
    };
    let out = git_stdout(root, &["log", "--no-merges", "--format=%H%x09%s", &range])?;
    let commits = out
        .lines()
        .filter_map(|line| {
            let (hash, subject) = line.split_once('\t')?;
            Some(Commit {
                hash: hash.to_string(),
                subject: subject.to_string(),
                ids: ids_in(subject),
            })
        })
        .collect();
    Ok(commits)
}

/// Read the repository's own two sources and put them together.
///
/// `dir` may be anywhere inside the repository: git reports its own top level,
/// and the changelog is read from there rather than from the starting directory.
pub fn draft(dir: &Path, from: Option<&str>, to: Option<&str>) -> Result<Draft, String> {
    let root = crate::gitinfo::repo_toplevel(dir).ok_or_else(|| {
        format!(
            "{} is not inside a git repository, so there is no history to draw release notes from",
            dir.display()
        )
    })?;
    let to = to.unwrap_or("HEAD").to_string();

    // An explicit `--from` is taken as given; an empty one means the caller has
    // no lower bound in mind, which is the same as not passing one. Otherwise the
    // newest tag is the previous release; with no tags there is no line to start
    // the range at, and the draft covers the whole history and says that is what
    // it is doing.
    let from = from.map(str::trim).filter(|from| !from.is_empty());
    let (from_resolved, range_note) = match from {
        Some(explicit) => (
            Some(explicit.to_string()),
            format!("commits after {explicit}, up to {to}"),
        ),
        None => match newest_tag(&root) {
            Some(tag) => (
                Some(tag.clone()),
                format!("commits since the newest tag {tag}, up to {to}"),
            ),
            None => (
                None,
                format!(
                    "this repository has no tags, so the draft covers every commit reachable \
                     from {to}"
                ),
            ),
        },
    };

    let changelog_path = root.join(CHANGELOG_FILE);
    let changelog = std::fs::read_to_string(&changelog_path).unwrap_or_default();
    let entries = parse_unreleased(&changelog);

    let commits = commits_between(&root, from_resolved.as_deref(), &to)?;

    let entry_ids: Vec<&str> = entries
        .iter()
        .flat_map(|e| e.ids.iter().map(String::as_str))
        .collect();
    let commit_ids: Vec<&str> = commits
        .iter()
        .flat_map(|c| c.ids.iter().map(String::as_str))
        .collect();

    let uncovered = commits
        .iter()
        .filter(|c| !c.ids.iter().any(|id| entry_ids.contains(&id.as_str())))
        .cloned()
        .collect();
    let unmatched = entries
        .iter()
        .filter(|e| !e.ids.is_empty() && !e.ids.iter().any(|id| commit_ids.contains(&id.as_str())))
        .cloned()
        .collect();

    Ok(Draft {
        to,
        from: from_resolved,
        range_note,
        entries,
        commits,
        uncovered,
        unmatched,
    })
}

/// The draft as markdown: the previous release's heading form, the entries in
/// the order the changelog already keeps them in, then the two coverage lists.
///
/// A `release` labels the heading; without one the heading stays `[Unreleased]`,
/// because inventing a version number is the editor's job, not the generator's.
pub fn to_markdown(draft: &Draft, release: Option<&str>) -> String {
    let heading = match release {
        Some(version) => format!("## [{version}]"),
        None => UNRELEASED_HEADING.to_string(),
    };
    let mut out = String::new();
    out.push_str(&heading);
    out.push_str("\n\n");
    out.push_str(&format!(
        "<!-- draft by `xencode release-notes`; {} of {} {} named by {} in \
         `CHANGELOG.md`. Edit this text before releasing it. -->\n\n",
        draft.commits.len() - draft.uncovered.len(),
        draft.commits.len(),
        plural(draft.commits.len(), "commit", "commits"),
        plural(
            draft.entries.len(),
            "unreleased entry",
            "unreleased entries"
        ),
    ));

    if draft.entries.is_empty() {
        out.push_str("_(no `## [Unreleased]` entries to draw from)_\n\n");
    }
    let mut category: Option<String> = None;
    for entry in &draft.entries {
        if category.as_deref() != Some(entry.category.as_str()) {
            if category.is_some() {
                out.push('\n');
            }
            out.push_str(&format!("### {}\n\n", entry.category));
            category = Some(entry.category.clone());
        }
        let ids = if entry.ids.is_empty() {
            String::new()
        } else {
            format!(" ({})", entry.ids.join(", "))
        };
        out.push_str(&format!("- **{}**{ids}\n", entry.title));
        for line in entry.body.trim().lines() {
            // An empty line stays empty: indenting it would put trailing spaces
            // in a file the editor is about to work on.
            if line.trim().is_empty() {
                out.push('\n');
            } else {
                out.push_str(&format!("  {line}\n"));
            }
        }
    }

    out.push_str(&format!("\n---\n*{}*\n", draft.range_note));
    if !draft.uncovered.is_empty() {
        out.push_str(&format!(
            "\n### {} {} with no changelog entry\n\n",
            draft.uncovered.len(),
            plural(draft.uncovered.len(), "commit", "commits"),
        ));
        for commit in draft.uncovered.iter().take(LIST_LIMIT) {
            out.push_str(&format!(
                "- `{}` — {}\n",
                short_hash(&commit.hash),
                commit.subject
            ));
        }
        let omitted = draft.uncovered.len().saturating_sub(LIST_LIMIT);
        if omitted > 0 {
            out.push_str(&format!(
                "- … and {omitted} more, oldest not printed; pass `--from <ref>` to put \
                 the range around this release alone\n"
            ));
        }
    }
    if !draft.unmatched.is_empty() {
        out.push_str(&format!(
            "\n### {} {} naming no commit in this range\n\n",
            draft.unmatched.len(),
            plural(draft.unmatched.len(), "entry", "entries"),
        ));
        for entry in &draft.unmatched {
            out.push_str(&format!(
                "- **{}** — named {}, not in the range\n",
                entry.title,
                entry.ids.join(", ")
            ));
        }
    }
    if draft.uncovered.is_empty() && draft.unmatched.is_empty() {
        out.push_str(
            "\nEvery commit in the range is accounted for by a changelog entry, and \
                      every entry matches a commit in the range.\n",
        );
    }
    out
}

/// The seven characters git itself prints for a hash.
pub fn short_hash(hash: &str) -> String {
    hash.chars().take(7).collect()
}

/// Write the draft, refusing to replace a file that is already there.
///
/// The refusal is the point: a draft is edited by a person, and regenerating it
/// over their edits is how an afternoon of wording is lost.
pub fn write_draft(path: &Path, markdown: &str, force: bool) -> Result<(), String> {
    if path.exists() && !force {
        return Err(format!(
            "{} already exists and may hold edits. Pass --force to replace it.",
            path.display()
        ));
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
        }
    }
    std::fs::write(path, markdown).map_err(|e| format!("could not write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    /// A fresh temporary directory per call. These tests run in parallel threads
    /// of one process, so the process id alone is not enough to keep them apart.
    fn unique_dir(tag: &str) -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-releasenotes-{tag}-{unique}"))
    }

    fn git(root: &Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .expect("the git binary should be available");
        assert!(
            out.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// The same, keeping what git printed.
    fn git_out(root: &Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .expect("the git binary should be available");
        assert!(
            out.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }

    /// A repository with two commits, one of them naming an id, and a changelog
    /// whose unreleased block accounts for only that one.
    fn sample_repo(tag: &str) -> PathBuf {
        let root = unique_dir(tag);
        std::fs::create_dir_all(root.join("crates").join("one")).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.email", "test@xencode.local"]);
        git(&root, &["config", "user.name", "Xencode Test"]);
        std::fs::write(
            root.join(CHANGELOG_FILE),
            "# Changelog\n\n## [Unreleased]\n\n### Added — `xencode thing` (QC-1)\n\n\
             Real prose describing the thing.\n\n## [1.0.0] - 2026-01-01\n\n\
             ### Added — shipped before this range\n",
        )
        .unwrap();
        std::fs::write(root.join("a.rs"), "// a\n").unwrap();
        git(&root, &["add", "-A"]);
        git(
            &root,
            &["commit", "-q", "-m", "Add the thing (QC-1): it works"],
        );
        std::fs::write(root.join("crates/one/lib.rs"), "// tidy\n").unwrap();
        git(&root, &["add", "-A"]);
        git(
            &root,
            &[
                "commit",
                "-q",
                "-m",
                "Rewrite one assertion for the current lint",
            ],
        );
        root
    }

    const SAMPLE: &str = "\
# Changelog

## [Unreleased]

### Added — `QO-4`: `xencode perf` — measure the hot paths

Seven benchmarks run over this repository read from disk.

### Added — `QD-2`: `/impact <file>` — the blast-radius panel

The panel is a projection, not a re-derivation.

### Fixed — a logged-in agent that still cannot run is not called broken

It says what it checked now.

## [2.1.0] - 2026-03-30

### Added — shipped last release

## [3.0.0] - 2024-10-15
";

    #[test]
    fn only_the_unreleased_block_is_read() {
        let entries = parse_unreleased(SAMPLE);
        assert_eq!(
            entries.len(),
            3,
            "the two released sections are not up for release"
        );
        assert_eq!(entries[0].category, "Added");
        assert_eq!(entries[2].category, "Fixed");
        assert!(
            !entries
                .iter()
                .any(|e| e.title.contains("shipped last release")),
            "a heading under an older version must never reach the draft"
        );
    }

    #[test]
    fn an_entry_carries_its_body_as_written() {
        let entries = parse_unreleased(SAMPLE);
        assert_eq!(
            entries[0].body,
            "Seven benchmarks run over this repository read from disk."
        );
    }

    #[test]
    fn the_category_word_is_the_changelogs_own() {
        // A heading with no ` — ` separator still lands somewhere: `Other`, not
        // a category invented here.
        let entries = parse_unreleased("## [Unreleased]\n\n### Something odd\n\nbody\n");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].category, "Other");
        assert_eq!(entries[0].title, "Something odd");
    }

    #[test]
    fn ids_are_read_from_headings_and_subjects_alike() {
        assert_eq!(
            ids_in("Add `xencode perf` (QO-4): measure the hot paths"),
            vec!["QO-4".to_string()]
        );
        assert_eq!(
            ids_in("Add the task evidence graph (EVd-8) after checking ten projects"),
            vec!["EVd-8".to_string()]
        );
        assert_eq!(
            ids_in("### Added — MCP client: hosted servers and prompts (M-6)"),
            vec!["M-6".to_string()]
        );
        assert_eq!(
            ids_in("Covered by `QD-1` and `QD-2` together"),
            vec!["QD-1".to_string(), "QD-2".to_string()]
        );
    }

    #[test]
    fn a_hyphenated_thing_with_a_digit_is_not_an_id() {
        // The shapes a changelog actually contains, none of which is a plan id.
        for text in [
            "xencode-context-rs",
            "tree-sitter-rust 0.24",
            "sha1-256",
            "--flag-1",
            "ab-12",
            "W0-1 progress",
            "2.1.0 - 2026-03-30",
        ] {
            assert!(ids_in(text).is_empty(), "{text} read as {:?}", ids_in(text));
        }
    }

    #[test]
    fn an_entry_with_no_id_claims_no_coverage() {
        let entries = parse_unreleased(SAMPLE);
        assert!(
            entries[2].ids.is_empty(),
            "a prose heading names no plan id, so it cannot mark a commit as explained"
        );
    }

    #[test]
    fn a_file_without_the_heading_yields_no_entries_rather_than_an_error() {
        assert!(parse_unreleased("# Changelog\n\n## [1.0.0] - 2024-08-15\n\n- done\n").is_empty());
        assert!(parse_unreleased("").is_empty());
    }

    #[test]
    fn the_markdown_groups_by_category_and_keeps_the_order() {
        let entries = parse_unreleased(SAMPLE);
        let draft = Draft {
            to: "HEAD".to_string(),
            from: None,
            range_note: "this repository has no tags".to_string(),
            commits: vec![Commit {
                hash: "abcdef1234567890".to_string(),
                subject: "Add `xencode perf` (QO-4): measure".to_string(),
                ids: vec!["QO-4".to_string()],
            }],
            uncovered: vec![],
            unmatched: vec![],
            entries,
        };
        let md = to_markdown(&draft, Some("2.2.0"));
        assert!(md.starts_with("## [2.2.0]"), "{md}");
        let added = md.find("### Added").unwrap();
        let fixed = md.find("### Fixed").unwrap();
        assert!(added < fixed, "the changelog's own order survives");
        assert!(
            md.contains("**`QO-4`: `xencode perf` — measure the hot paths** (QO-4)"),
            "the id the heading names travels with the title: {md}"
        );
        assert!(
            md.contains("Every commit in the range is accounted for"),
            "{md}"
        );
    }

    #[test]
    fn a_commit_is_listed_by_the_seven_characters_git_itself_prints() {
        assert_eq!(short_hash("abcdef1234567890"), "abcdef1");
        assert_eq!(short_hash("abc"), "abc", "a short hash is never padded");
    }

    #[test]
    fn a_heading_labels_the_draft_as_a_draft() {
        let draft = Draft {
            to: "HEAD".to_string(),
            from: Some("v1".to_string()),
            range_note: "commits since v1".to_string(),
            entries: vec![],
            commits: vec![],
            uncovered: vec![],
            unmatched: vec![],
        };
        let md = to_markdown(&draft, None);
        assert!(md.contains("## [Unreleased]"), "{md}");
        assert!(md.contains("Edit this text before releasing it"), "{md}");
        assert!(
            md.contains("(no `## [Unreleased]` entries to draw from)"),
            "an empty draft says so instead of printing nothing"
        );
    }

    #[test]
    fn the_two_coverage_lists_name_the_gaps_in_both_directions() {
        let entries = vec![
            Entry {
                category: "Added".to_string(),
                title: "`xencode perf`".to_string(),
                body: String::new(),
                ids: vec!["QO-4".to_string()],
            },
            Entry {
                category: "Fixed".to_string(),
                title: "an id that never shipped".to_string(),
                body: String::new(),
                ids: vec!["ZZ-9".to_string()],
            },
        ];
        let commits = vec![
            Commit {
                hash: "1111111111".to_string(),
                subject: "Add `xencode perf` (QO-4)".to_string(),
                ids: vec!["QO-4".to_string()],
            },
            Commit {
                hash: "2222222222".to_string(),
                subject: "Rewrite one assertion for the current lint".to_string(),
                ids: vec![],
            },
        ];
        let draft = Draft {
            to: "HEAD".to_string(),
            from: None,
            range_note: "every commit".to_string(),
            uncovered: commits
                .iter()
                .filter(|c| {
                    !c.ids
                        .iter()
                        .any(|id| entries.iter().any(|e| e.ids.contains(id)))
                })
                .cloned()
                .collect(),
            unmatched: entries
                .iter()
                .filter(|e| {
                    !e.ids
                        .iter()
                        .any(|id| commits.iter().any(|c| c.ids.contains(id)))
                })
                .cloned()
                .collect(),
            entries,
            commits,
        };
        assert_eq!(draft.covered_ids(), vec!["QO-4".to_string()]);
        assert_eq!(draft.uncovered.len(), 1);
        assert_eq!(
            draft.uncovered[0].subject,
            "Rewrite one assertion for the current lint"
        );
        assert_eq!(draft.unmatched.len(), 1);
        assert_eq!(draft.unmatched[0].ids, vec!["ZZ-9".to_string()]);

        let md = to_markdown(&draft, None);
        assert!(md.contains("### 1 commit with no changelog entry"), "{md}");
        assert!(
            md.contains("### 1 entry naming no commit in this range"),
            "{md}"
        );
    }

    #[test]
    fn a_long_unexplained_list_shows_its_head_and_counts_the_rest() {
        // The no-tags case in this repository puts every commit in the range, so
        // the draft has to stay readable: it names the first LIST_LIMIT and says
        // how many it left out, rather than pretending the list is short.
        let many = (0..LIST_LIMIT + 10)
            .map(|i| Commit {
                hash: format!("{i:040x}"),
                subject: format!("Unexplained work number {i}"),
                ids: vec![],
            })
            .collect::<Vec<_>>();
        let draft = Draft {
            to: "HEAD".to_string(),
            from: None,
            range_note: "every commit".to_string(),
            uncovered: many.clone(),
            unmatched: vec![],
            entries: vec![],
            commits: many,
        };
        let md = to_markdown(&draft, None);
        assert!(
            md.contains(&format!(
                "### {} commits with no changelog entry",
                LIST_LIMIT + 10
            )),
            "{md}"
        );
        assert!(md.contains("- … and 10 more, oldest not printed"), "{md}");
        assert_eq!(
            md.matches("- `").count(),
            LIST_LIMIT,
            "the printed list stops at the limit"
        );
    }

    #[test]
    fn a_draft_file_is_never_replaced_without_being_told_to() {
        let dir = unique_dir("overwrite");
        std::fs::create_dir_all(&dir).unwrap();
        let path: PathBuf = dir.join("notes.md");

        write_draft(&path, "first draft", false).unwrap();
        let err = write_draft(&path, "second draft", false).unwrap_err();
        assert!(err.contains("--force"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first draft");

        write_draft(&path, "second draft", true).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second draft");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn writing_creates_the_directories_a_named_path_needs() {
        let dir = unique_dir("write");
        let path: PathBuf = dir.join("drafts").join("notes.md");
        write_draft(&path, "draft", false).unwrap();
        assert!(path.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_non_repository_is_named_rather_than_measured_against_anything() {
        let dir = unique_dir("nogit");
        std::fs::create_dir_all(&dir).unwrap();
        let err = draft(&dir, None, None).unwrap_err();
        assert!(err.contains("not inside a git repository"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_draft_started_in_a_subdirectory_still_reads_the_changelog_at_the_top() {
        // The case that matters on this machine: cargo runs from `rust/` while
        // `CHANGELOG.md` sits one level up. Treating the starting directory as the
        // repository root would find no changelog and blame every commit for it.
        let root = sample_repo("sub");
        let built = draft(&root.join("crates").join("one"), None, None)
            .expect("a draft from a subdirectory");
        assert_eq!(
            built.entries.len(),
            1,
            "the changelog at the top is the one read"
        );
        assert_eq!(built.commits.len(), 2);
        assert_eq!(
            built.uncovered.len(),
            1,
            "the lint commit is explained by nothing"
        );
        // No tag to start a range from, so the draft says what it covered.
        assert!(built.range_note.contains("no tags"), "{}", built.range_note);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_tag_starts_the_range_so_work_before_it_is_not_released_twice() {
        let root = sample_repo("tagged");
        git(&root, &["tag", "v1.0.0"]);
        std::fs::write(root.join("c.rs"), "// c\n").unwrap();
        git(&root, &["add", "-A"]);
        git(
            &root,
            &["commit", "-q", "-m", "Add the other thing (QC-2): it works"],
        );

        let built = draft(&root, None, None).unwrap();
        assert_eq!(built.from.as_deref(), Some("v1.0.0"));
        assert!(
            built.range_note.contains("newest tag v1.0.0"),
            "{}",
            built.range_note
        );
        assert_eq!(
            built.commits.len(),
            1,
            "the two commits under the tag already shipped"
        );
        assert_eq!(built.uncovered.len(), 1, "QC-2 is in no changelog entry");
        assert_eq!(built.unmatched.len(), 1, "QC-1's commit is below the tag");

        let md = to_markdown(&built, Some("1.1.0"));
        assert!(md.starts_with("## [1.1.0]"), "{md}");
        assert!(
            md.contains("### 1 commit with no changelog entry"),
            "the commit after the tag is not explained: {md}"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn an_empty_from_is_no_lower_bound_rather_than_a_range_from_nothing() {
        let root = sample_repo("empty-from");
        let built = draft(&root, Some("  "), None).unwrap();
        assert!(built.from.is_none(), "{:?}", built.from);
        assert_eq!(
            built.commits.len(),
            2,
            "the whole history, as with no lower bound"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn an_explicit_from_is_taken_as_given_even_with_tags_present() {
        let root = sample_repo("explicit");
        git(&root, &["tag", "v0.9.0"]);
        // The tag sits on the tip, so a range started from it would be empty. The
        // explicit lower bound has to win, and it names the earlier commit.
        let earlier = git_out(&root, &["rev-parse", "HEAD~1"]);
        let built = draft(&root, Some(&earlier), None).unwrap();
        assert_eq!(built.from.as_deref(), Some(earlier.as_str()));
        assert_eq!(built.commits.len(), 1, "one commit after the named one");
        assert!(
            built.range_note.contains("commits after"),
            "{}",
            built.range_note
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
