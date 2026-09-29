//! Hotspots: churn × size and bus factor (`GH-4`).
//!
//! Files that change often *and* are large are where bugs live and where
//! rewrites hurt most; files touched by exactly one author are where knowledge
//! lives nowhere else. Both come from one `git log --pretty --name-only` mine
//! plus working-tree file sizes — never `--numstat`, which costs 15.2 s on a
//! repository this size for information the score does not need.
//!
//! Results are [`crate::advise::Advice`] rows so every existing surface can
//! render them, and every row carries an action: a finding without one is
//! decorative, and decorative panels get ignored until they get deleted.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::advise::{Advice, AdviceKind};

/// One file's history as the miner saw it.
#[derive(Debug, Clone, Default)]
pub struct FileHistory {
    /// How many commits touched the file.
    pub commits: usize,
    /// Author emails, one per human.
    pub authors: BTreeSet<String>,
}

/// Parse `git log --pretty=format:%ae --name-only --no-merges` output.
///
/// Commits are separated from their file lists by the `%ae` line; blank lines
/// are separators, not content. Anything unparseable is skipped rather than
/// guessed at, because a guessed author is a guessed bus factor.
pub fn parse_churn(text: &str) -> BTreeMap<String, FileHistory> {
    let mut out: BTreeMap<String, FileHistory> = BTreeMap::new();
    let mut author: Option<&str> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            author = None;
            continue;
        }
        if author.is_none() {
            // The first non-blank line after a separator is the email, unless
            // it looks like a path — a merge-less log always alternates.
            if line.contains('@') && !line.contains('/') && !line.contains(' ') {
                author = Some(line);
                continue;
            }
        }
        if let Some(email) = author {
            let entry = out.entry(line.to_string()).or_default();
            entry.commits += 1;
            entry.authors.insert(email.to_string());
        }
    }
    out
}

/// Mine the whole history. `--name-only`, never `--numstat`: per-file line
/// counts cost an order of magnitude more for a score that multiplies by
/// working-tree size instead.
pub fn mine_churn(root: &Path) -> BTreeMap<String, FileHistory> {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args(["log", "--pretty=format:%ae", "--no-merges", "--name-only"])
        .output();
    let Ok(output) = output else {
        return BTreeMap::new();
    };
    if !output.status.success() {
        return BTreeMap::new();
    }
    parse_churn(&String::from_utf8_lossy(&output.stdout))
}

/// One CODEOWNERS rule: glob patterns and the owners they assign.
#[derive(Debug, Clone)]
pub struct OwnerRule {
    /// As written, e.g. `*.rs` or `/docs/`.
    pub pattern: String,
    /// `@user` / `@team` tokens.
    pub owners: Vec<String>,
}

/// Parse a CODEOWNERS file. Last matching rule wins, per GitHub semantics.
/// Absent file means no rules, not an error: most repositories have none, and
/// "no owners declared" is itself reported per file.
pub fn parse_codeowners(text: &str) -> Vec<OwnerRule> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(pattern) = parts.next() else {
            continue;
        };
        let owners: Vec<String> = parts.map(str::to_string).collect();
        if owners.is_empty() {
            continue;
        }
        out.push(OwnerRule {
            pattern: pattern.to_string(),
            owners,
        });
    }
    out
}

/// Whether a CODEOWNERS pattern matches a repository-relative path.
///
/// GitHub subset: a leading `/` anchors to the root, a trailing `/` matches
/// everything beneath, `*` spans within a segment, `**` spans segments.
/// Anything fancier is declined (returns false) rather than approximated,
/// because an approximated owner is a wrong owner with confidence.
pub fn pattern_matches(pattern: &str, path: &str) -> bool {
    let (anchored, pattern) = match pattern.strip_prefix('/') {
        Some(rest) => (true, rest),
        None => (false, pattern),
    };
    if let Some(dir) = pattern.strip_suffix('/') {
        let dir = dir.trim_end_matches('/');
        if anchored {
            return path == dir || path.starts_with(&format!("{dir}/"));
        }
        let dir = dir.trim_end_matches('/');
        return path == dir
            || path.starts_with(&format!("{dir}/"))
            || path.contains(&format!("/{dir}/"));
    }
    if pattern.contains("**") {
        return match_double_star(anchored, pattern, path);
    }
    let pattern_segments: Vec<&str> = pattern.split('/').collect();
    let path_segments: Vec<&str> = path.split('/').collect();
    if pattern_segments.len() == 1 {
        // `*.rs` matches the file name at any depth.
        return segment_matches(pattern, path_segments.last().unwrap_or(&""));
    }
    if anchored {
        return pattern_segments.len() == path_segments.len()
            && pattern_segments
                .iter()
                .zip(path_segments.iter())
                .all(|(pat, seg)| segment_matches(pat, seg));
    }
    path_segments.windows(pattern_segments.len()).any(|window| {
        window
            .iter()
            .zip(pattern_segments.iter())
            .all(|(seg, pat)| segment_matches(pat, seg))
    })
}

/// Ordered-literal matching for `**` patterns: every literal chunk appears in
/// order, and an anchored pattern starts with its first chunk.
fn match_double_star(anchored: bool, pattern: &str, path: &str) -> bool {
    let chunks: Vec<&str> = pattern
        .split("**")
        .flat_map(|part| part.split('*'))
        .filter(|c| !c.is_empty())
        .collect();
    if chunks.is_empty() {
        return true;
    }
    let mut remaining = path;
    for (index, chunk) in chunks.iter().enumerate() {
        if index == 0 && anchored {
            let Some(rest) = remaining.strip_prefix(chunk) else {
                return false;
            };
            remaining = rest;
            continue;
        }
        match remaining.find(chunk) {
            Some(at) => remaining = &remaining[at + chunk.len()..],
            None => return false,
        }
    }
    true
}

fn segment_matches(pattern: &str, text: &str) -> bool {
    // `*` spans any run inside one segment; literal parts must appear in order
    // and consume the whole segment unless a `*` covers the tail.
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or("");
    if !text.starts_with(first) {
        return false;
    }
    let mut rest = &text[first.len()..];
    let mut trailing_star = false;
    for part in parts {
        if part.is_empty() {
            trailing_star = true;
            continue;
        }
        trailing_star = false;
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    trailing_star || rest.is_empty()
}

/// The owners CODEOWNERS assigns a path, last match wins. Empty when no rule
/// matches or no file exists.
pub fn owners_for(rules: &[OwnerRule], path: &str) -> Vec<String> {
    let mut owners = Vec::new();
    for rule in rules {
        if pattern_matches(&rule.pattern, path) {
            owners = rule.owners.clone();
        }
    }
    owners
}

/// Paths whose churn × size means nothing: build outputs change on every
/// build and are enormous, so they would outrank every real file. Skipping
/// them is not hiding information — a hotspot panel led by a 137 MB `.rlib`
/// is decorative, and decorative panels get ignored until they get deleted.
fn is_generated(path: &str) -> bool {
    const GENERATED_DIRS: &[&str] = &[
        "target",
        "node_modules",
        ".git",
        "dist",
        "build",
        ".xencode",
        ".venv",
        "__pycache__",
    ];
    // Any segment, not just the first: a workspace whose manifest lives under
    // `rust/` reports `rust/target/…`, and a prefix-only check misses it while
    // the 137 MB artifact keeps leading the panel.
    path.split('/').any(|seg| GENERATED_DIRS.contains(&seg))
}

/// Ranked hotspots: the top `limit` files by commits × working-tree bytes.
pub fn hotspots(root: &Path, limit: usize) -> Vec<Advice> {
    let churn = mine_churn(root);
    let rules = std::fs::read_to_string(root.join("CODEOWNERS"))
        .map(|text| parse_codeowners(&text))
        .unwrap_or_default();

    let mut scored: Vec<(String, usize, usize)> = Vec::new();
    for (file, history) in &churn {
        if is_generated(file) {
            continue;
        }
        let size = std::fs::metadata(root.join(file))
            .map(|m| m.len())
            .unwrap_or(0);
        if size == 0 {
            continue;
        }
        scored.push((file.clone(), history.commits, size as usize));
    }
    scored.sort_by(|a, b| (b.1 * b.2).cmp(&(a.1 * a.2)).then(a.0.cmp(&b.0)));

    let mut out = Vec::new();
    for (file, commits, size) in scored.iter().take(limit.max(1)) {
        let history = &churn[file];
        let owners = owners_for(&rules, file);
        let owner_note = if owners.is_empty() {
            "no CODEOWNERS entry — add an owner".to_string()
        } else {
            format!("owner: {}", owners.join(", "))
        };
        out.push(Advice {
            file: file.clone(),
            kind: AdviceKind::Hotspot,
            message: format!(
                "🔥 {file} changed in {commits} commit(s) at {size} bytes (score {}) across {} author(s); {owner_note} — split it or cover it with tests before the next change.",
                commits * size,
                history.authors.len(),
            ),
        });
        if history.authors.len() == 1 {
            let author = history
                .authors
                .iter()
                .next()
                .map(String::as_str)
                .unwrap_or("?");
            out.push(Advice {
                file: file.clone(),
                kind: AdviceKind::SingleOwner,
                message: format!(
                    "👤 {file} is touched only by {author} — bus factor 1. Get a second reviewer on its next change.",
                ),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "alice@x\nsrc/a.rs\nsrc/b.rs\n\nbob@x\nsrc/a.rs\n\n";

    #[test]
    fn churn_counts_commits_and_authors() {
        let history = parse_churn(LOG);
        assert_eq!(history["src/a.rs"].commits, 2);
        assert_eq!(history["src/a.rs"].authors.len(), 2);
        assert_eq!(history["src/b.rs"].commits, 1);
    }

    #[test]
    fn blank_lines_reset_the_author() {
        // Without the reset, a following file would inherit the last author.
        // Here no author is in effect at all, so the file stays unattributed.
        let history = parse_churn("alice@x\n\nsrc/a.rs\n");
        assert!(!history.contains_key("src/a.rs"), "{history:?}");
    }

    #[test]
    fn codeowners_last_match_wins() {
        let rules = parse_codeowners("*.rs @a\n/src/a.rs @b\n");
        assert_eq!(owners_for(&rules, "src/a.rs"), vec!["@b"]);
        assert_eq!(owners_for(&rules, "src/c.rs"), vec!["@a"]);
        assert!(owners_for(&rules, "notes.md").is_empty());
        assert!(parse_codeowners("# only a comment\n\n").is_empty());
    }

    #[test]
    fn directory_patterns_match_beneath() {
        assert!(pattern_matches("/docs/", "docs/guide.md"));
        assert!(!pattern_matches("/docs/", "src/docs.md"));
        assert!(pattern_matches("*.rs", "src/a.rs"));
        assert!(!pattern_matches("*.rs", "notes.md"));
        assert!(pattern_matches("src/*.rs", "src/a.rs"));
        assert!(!pattern_matches("src/*.rs", "src/deep/a.rs"));
    }

    #[test]
    fn generated_dirs_never_outrank_real_files() {
        // The failure this guards: a 137 MB build artifact changed twice
        // outscores every source file and leads the panel.
        assert!(is_generated("target/debug/libx.rlib"));
        assert!(is_generated(".xencode/coverage.lcov"));
        assert!(is_generated("node_modules/dep/index.js"));
        assert!(!is_generated("src/main.rs"));
        assert!(!is_generated("NEXT_PLAN_TASKS.md"));
    }

    #[test]
    fn every_row_carries_an_action() {
        let dir = std::env::temp_dir().join(format!("xe-hot-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let repo = dir.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let git = |args: &[&str]| {
            std::process::Command::new("git")
                .current_dir(&repo)
                .args(args)
                .output()
                .unwrap()
        };
        git(&["init", "-q", "-b", "main"]);
        for (email, name) in [("alice@x", "a"), ("bob@x", "b")] {
            git(&["config", "user.email", email]);
            git(&["config", "user.name", name]);
            std::fs::write(repo.join("hot.rs"), format!("// {email}\nfn f() {{}}\n")).unwrap();
            std::fs::write(repo.join("cold.rs"), "// once\n").unwrap();
            git(&["add", "-A"]);
            git(&["commit", "-qm", "change"]);
        }
        std::fs::write(repo.join("CODEOWNERS"), "*.rs @owners\n").unwrap();
        let rows = hotspots(&repo, 5);
        assert!(
            rows.iter()
                .any(|r| r.kind == AdviceKind::Hotspot && r.file == "hot.rs"),
            "{rows:?}"
        );
        for row in &rows {
            assert!(
                row.message.contains("— "),
                "a row without an action is decorative: {row:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
