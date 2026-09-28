//! Diff coverage (`VF-1`) and the read-only missing-line surface (`VF-2`).
//!
//! A green test suite answers "did nothing break". It does not answer "did the
//! lines I just wrote run at all" — a new error-handling branch can be entirely
//! unexercised and the suite still reports every test passing. The two together
//! are the only useful form of the question, so that is what this module
//! computes: which *added* lines in the diff were never executed.
//!
//! Two decisions shape everything here.
//!
//! - **Lines, not percentages.** A coverage percentage is a number nobody can act
//!   on, and comparing it against last week's number says nothing about whether
//!   *this* change is tested. The output is `file → [line numbers]`, which an
//!   agent can go and read. `VF-2`'s done-when asks for exactly this shape.
//! - **A cold run is stated, not hidden.** Coverage needs a second, instrumented
//!   build in its own target directory, and that build is not cheap — the
//!   project's own plan records a measured case at 377 s, almost all of it
//!   recompilation. The same target directory is reused across runs so only the
//!   first one pays, and [`ColdReport`] says which kind of run happened, because
//!   a reader who waits eight minutes deserves to know that the second run is the
//!   one that will be fast.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Where generated reports live, relative to the repository root.
pub const COV_STATE_DIR: &str = ".xencode";

/// The target directory coverage uses, kept apart from the normal build so an
/// instrumented build never invalidates a developer build or vice versa.
pub const COV_TARGET_DIR: &str = "target/llvm-cov-target";

/// A file and the added lines in it that never ran.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FileCoverage {
    /// Repository-relative path.
    pub file: String,
    /// Added lines that were executed, ascending.
    pub covered: Vec<u32>,
    /// Added lines that were never executed, ascending.
    pub uncovered: Vec<u32>,
    /// Added lines that no coverage data mentions.
    ///
    /// Distinct from uncovered: a line the tool never heard of is a line it
    /// cannot make a claim about, and reporting it as "untested" would be the
    /// same as saying the tool looked and found nothing. Derive and macro output
    /// are the usual cause, which is why `--ignore-filename-regex` exists.
    pub unknown: Vec<u32>,
}

impl FileCoverage {
    /// Added lines, all of them.
    pub fn added(&self) -> usize {
        self.covered.len() + self.uncovered.len() + self.unknown.len()
    }

    /// `covered / (covered + uncovered)`, ignoring unknown lines.
    ///
    /// `None` when nothing was measurable, rather than a misleading `1.0` for a
    /// file with no instrumentable added lines.
    pub fn ratio(&self) -> Option<f64> {
        let measurable = self.covered.len() + self.uncovered.len();
        if measurable == 0 {
            None
        } else {
            Some(self.covered.len() as f64 / measurable as f64)
        }
    }
}

/// The whole diff's coverage.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DiffCoverage {
    /// Per-file results, sorted by path.
    pub files: Vec<FileCoverage>,
    /// Files in the diff that coverage data had nothing for at all.
    pub unmeasured: Vec<String>,
    /// Lines added in files that have no coverage data.
    pub unmeasured_lines: u64,
}

impl DiffCoverage {
    /// Added lines never executed across every file.
    pub fn total_uncovered(&self) -> usize {
        self.files.iter().map(|f| f.uncovered.len()).sum()
    }

    /// Added lines executed across every file.
    pub fn total_covered(&self) -> usize {
        self.files.iter().map(|f| f.covered.len()).sum()
    }

    /// Added lines the tool cannot speak for.
    pub fn total_unknown(&self) -> usize {
        self.files.iter().map(|f| f.unknown.len()).sum()
    }

    /// The fraction of measurable added lines that ran.
    pub fn ratio(&self) -> Option<f64> {
        let measurable = self.total_covered() + self.total_uncovered();
        if measurable == 0 {
            None
        } else {
            Some(self.total_covered() as f64 / measurable as f64)
        }
    }

    /// Whether every measurable added line ran.
    pub fn is_complete(&self) -> bool {
        self.total_uncovered() == 0 && self.total_unknown() == 0
    }

    /// The report line, which never claims completeness it cannot support.
    pub fn summary(&self) -> String {
        let mut out = format!(
            "{} of {} measurable added line(s) executed",
            self.total_covered(),
            self.total_covered() + self.total_uncovered()
        );
        let unknown = self.total_unknown();
        if unknown > 0 {
            out.push_str(&format!(
                "; {unknown} added line(s) had no coverage data and are not counted as covered"
            ));
        }
        if !self.unmeasured.is_empty() {
            out.push_str(&format!(
                "; {} file(s) in the diff have no coverage data at all",
                self.unmeasured.len()
            ));
        }
        out
    }
}

/// Which kind of run this was, so the wait is explained rather than mysterious.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdReport {
    /// The instrumented build did not exist and was made.
    Cold,
    /// The instrumented build was reused.
    Warm,
}

/// What a coverage run produced.
#[derive(Debug, Clone)]
pub struct Run {
    /// Per-file coverage, empty when no diff was given.
    pub coverage: DiffCoverage,
    /// The exact command that ran.
    pub command: String,
    /// Whether the instrumented build was made or reused.
    pub build: ColdReport,
    /// How long the run took.
    pub took: Duration,
    /// Anything the reader needs that is not a number.
    pub notes: Vec<String>,
}

/// Whether `cargo llvm-cov` is usable here.
pub fn available() -> bool {
    std::process::Command::new("cargo")
        .args(["llvm-cov", "--version"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Run coverage over the added lines of `diff`.
///
/// `base` is the ref the diff is taken against; `None` means the working tree
/// against `HEAD`. The lcov output is produced first, then intersected with the
/// added lines, so a file with no added lines costs nothing to report.
pub fn run(root: &Path, base: Option<&str>, test_command: Option<&str>) -> Result<Run, String> {
    let started = Instant::now();
    // Coverage is a cargo operation, so it needs the directory holding the
    // manifest. This repository's is under `rust/`, and running from the root
    // fails with a message about a missing Cargo.toml that has nothing to do
    // with coverage.
    let manifest_root = xencode_context_rs::verify::manifest_dir(root)?;
    let cov_dir = manifest_root.join(COV_TARGET_DIR);
    let build = if cov_dir.exists() {
        ColdReport::Warm
    } else {
        ColdReport::Cold
    };

    let diff = added_lines(&manifest_root, base)?;
    if diff.is_empty() {
        return Ok(Run {
            coverage: DiffCoverage::default(),
            command: String::new(),
            build,
            took: started.elapsed(),
            notes: vec!["no added lines in the diff, so there is nothing to measure".to_string()],
        });
    }

    let mut notes = Vec::new();
    if build == ColdReport::Cold {
        notes.push(
            "the first run builds every crate a second time with instrumentation, which is \
             slow; this target directory is reused, so later runs are much faster"
                .to_string(),
        );
    }

    // The report lands beside the rest of the project's generated state, in the
    // repository root rather than the workspace, because that is where `/init`
    // and the anchor write. It has to be created: on a fresh clone nothing has
    // made it, and llvm-cov fails at the very end of a long run because a
    // directory was missing.
    let state = root.join(COV_STATE_DIR);
    std::fs::create_dir_all(&state)
        .map_err(|e| format!("could not create {}: {e}", state.display()))?;
    let lcov_path = state.join("coverage.lcov");

    let mut command = std::process::Command::new("cargo");
    command
        .arg("llvm-cov")
        .arg("--workspace")
        .arg("--lcov")
        .arg("--output-path")
        .arg(&lcov_path)
        .current_dir(&manifest_root)
        // Keep the instrumented build out of the developer's target directory.
        .env("CARGO_LLVM_COV_TARGET_DIR", &cov_dir)
        // The instrumented build is not something the developer should inherit.
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS");
    if let Some(test) = test_command {
        // Reuse the repository's own verified test command rather than
        // llvm-cov's default, so the numbers describe the suite that exists.
        command.arg("--").args(test.split_whitespace());
    }
    let rendered = "cargo llvm-cov --workspace --lcov".to_string();

    let output = command
        .output()
        .map_err(|e| format!("could not start cargo llvm-cov: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "coverage run failed ({}): {}",
            output.status.code().unwrap_or(-1),
            stderr.lines().rev().take(6).collect::<Vec<_>>().join(" / ")
        ));
    }

    if !lcov_path.is_file() {
        return Err(format!(
            "cargo llvm-cov exited zero but wrote no lcov file at {}",
            lcov_path.display()
        ));
    }
    // The manifest root is the reporting root: the diff is taken from the
    // repository root, so lcov's paths are compared against the tree cargo saw.
    let hits = parse_lcov(&lcov_path, &manifest_root);
    let coverage = intersect(&diff, &hits, &mut notes);

    Ok(Run {
        coverage,
        command: rendered,
        build,
        took: started.elapsed(),
        notes,
    })
}

/// The `added lines` map for a diff, from `git diff`.
///
/// Returns paths to ascending line numbers. `base` is the ref to compare
/// against; without one the working tree is compared to `HEAD`, which is what
/// "what have I just changed" means.
///
/// `--relative` is load-bearing: it makes git report paths from the directory
/// it is run in rather than from the repository root, so the diff paths and the
/// lcov paths share one base. Without it a workspace under `rust/` yields
/// `rust/crates/x/src/lib.rs` on one side and `crates/x/src/lib.rs` on the other,
/// every file misses, and the honest-looking answer is that no changed line was
/// ever exercised. `--no-prefix-remap` would be the belt to that braces — it
/// stops git rewriting the prefix when `diff.noprefix` is set in the user's
/// config — but it needs git 2.41, and it is not worth failing a run over.
pub fn added_lines(root: &Path, base: Option<&str>) -> Result<BTreeMap<String, Vec<u32>>, String> {
    let mut command = std::process::Command::new("git");
    command
        .arg("diff")
        .arg("--unified=0")
        .arg("--relative")
        // Pin the prefixes so the paths never depend on the reader's git
        // configuration. Git's defaults are mnemonic (`i/` for the index, `w/`
        // for the worktree), and one of those read as a directory name.
        .arg("--src-prefix=a/")
        .arg("--dst-prefix=b/");
    if let Some(base) = base {
        command.arg(base);
    }
    let output = command
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not start git: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "git diff failed: {}",
            stderr.lines().next().unwrap_or("no output")
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    Ok(parse_unified_diff_added(&text))
}

/// Added line numbers per file, from `git diff --unified=0` output.
///
/// A hunk header is `@@ -a,b +c,d @@`; the new-file start is `c` and the count
/// is `d`. Every following `+` line is one added line, numbered from `c` upward.
/// `+++` is the file header, not content.
pub fn parse_unified_diff_added(text: &str) -> BTreeMap<String, Vec<u32>> {
    let mut out: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let mut file: Option<String> = None;
    let mut next_line: u32 = 0;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = rest.trim();
            // Only an explicit `b/` is a path. Git's default prefixes are
            // mnemonic — `w/` for the worktree, `i/` for the index — and taking
            // one of those as a directory name makes every file miss to match,
            // which then reads as a confident and entirely wrong "none of your
            // added lines ran". A `+++` line with no known prefix is not a file
            // this parser can speak about, so it declines it rather than
            // inventing a path.
            file = match path {
                "/dev/null" => None,
                _ => path.strip_prefix("b/").map(|p| p.to_string()),
            };
            if let Some(path) = &file {
                out.entry(path.clone()).or_default();
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("@@") {
            // @@ -old,len +new,len @@ optional heading
            if let Some(plus) = rest.split('+').nth(1) {
                let start: u32 = plus
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.split(',').next())
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                next_line = start;
            }
            continue;
        }
        if let Some(path) = &file {
            if let Some(_added) = line.strip_prefix('+') {
                if next_line > 0 {
                    out.entry(path.clone()).or_default().push(next_line);
                }
                next_line += 1;
            } else if line.starts_with(' ') || line.starts_with('-') {
                // A context or removed line still advances the new-side counter
                // only for context; a removed line does not exist in the new file.
                if line.starts_with(' ') && next_line > 0 {
                    next_line += 1;
                }
            }
        }
    }
    out.retain(|_, lines| !lines.is_empty());
    out
}

/// Which lines were executed, per absolute file path, from an lcov file.
///
/// The `DA:<line>,<count>` records are the only ones that matter; a count of
/// zero means the line was compiled in and never ran.
pub fn parse_lcov(path: &Path, root: &Path) -> BTreeMap<String, Vec<u32>> {
    let mut out: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    let mut current: Option<PathBuf> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("SF:") {
            let raw = rest.trim();
            let p = PathBuf::from(raw);
            current = Some(if p.is_absolute() { p } else { root.join(p) });
            if let Some(p) = &current {
                out.insert(relative(p, root), Vec::new());
            }
        } else if let Some(rest) = line.strip_prefix("DA:") {
            let Some(p) = &current else { continue };
            let mut parts = rest.split(',');
            let Some(line_no) = parts.next().and_then(|n| n.trim().parse::<u32>().ok()) else {
                continue;
            };
            let count = parts
                .next()
                .and_then(|n| n.trim().parse::<u64>().ok())
                .unwrap_or(0);
            if count > 0 {
                out.entry(relative(p, root)).or_default().push(line_no);
            }
        }
    }
    out
}

/// Make a path repository-relative for reporting, tolerating a `..` that
/// lcov's absolute paths may contain.
fn relative(path: &Path, root: &Path) -> String {
    let cleaned: PathBuf = path
        .components()
        .filter(|c| !matches!(c, std::path::Component::CurDir))
        .collect();
    let root_text = root.to_string_lossy().into_owned();
    let text = cleaned.to_string_lossy().into_owned();
    if let Some(rest) = text.strip_prefix(&root_text) {
        return rest.trim_start_matches('/').to_string();
    }
    cleaned
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or(text)
}

/// Cross the added lines with what actually ran.
///
/// Anything the lcov file never mentions is `unknown`, not `uncovered`. That
/// distinction is the difference between "the tool looked and this did not run"
/// and "the tool has no idea about this line", and collapsing them would blame
/// the code for the tool's blind spot.
pub fn intersect(
    added: &BTreeMap<String, Vec<u32>>,
    hits: &BTreeMap<String, Vec<u32>>,
    notes: &mut Vec<String>,
) -> DiffCoverage {
    let mut out = DiffCoverage::default();
    let mut unknown_files = Vec::new();

    for (file, lines) in added {
        let covered_set = hits.get(file);
        let mut result = FileCoverage {
            file: file.clone(),
            ..FileCoverage::default()
        };
        match covered_set {
            None => {
                result.unknown = lines.clone();
                unknown_files.push(file.clone());
            }
            Some(covered) => {
                for line in lines {
                    if covered.contains(line) {
                        result.covered.push(*line);
                    } else {
                        result.uncovered.push(*line);
                    }
                }
            }
        }
        out.files.push(result);
    }

    if !unknown_files.is_empty() {
        notes.push(format!(
            "{} changed file(s) have no coverage data: {}. These are not counted as \
             uncovered — the tool never looked at them.",
            unknown_files.len(),
            unknown_files
                .iter()
                .map(|f| f.rsplit('/').next().unwrap_or(f).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    out.unmeasured = unknown_files;
    out.unmeasured_lines = out
        .files
        .iter()
        .filter(|f| out.unmeasured.contains(&f.file))
        .map(|f| f.unknown.len() as u64)
        .sum();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Tree(PathBuf);

    impl Tree {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("xe-cov-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(self, rel: &str, body: &str) -> Self {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let mut f = std::fs::File::create(&p).unwrap();
            f.write_all(body.as_bytes()).unwrap();
            drop(f);
            self
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const DIFF: &str = "\
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,0 +11,3 @@
+fn one() {}
+fn two() {}
+fn three() {}
@@ -30,1 +34,1 @@
-    old();
+    new();
";

    #[test]
    fn added_lines_are_numbered_from_the_new_side() {
        let added = parse_unified_diff_added(DIFF);
        assert_eq!(added.get("src/lib.rs").unwrap(), &vec![11, 12, 13, 34]);
    }

    #[test]
    fn a_mnemonic_worktree_prefix_is_not_taken_as_part_of_the_path() {
        // git reports `w/src/lib.rs` for the worktree side by default, which is
        // not a path. Taking it literally makes every file miss and produces a
        // confident, entirely wrong "no added line ran".
        let diff = "+++ w/src/lib.rs\n@@ -1,0 +2,1 @@\n+fn f() {}\n";
        let added = parse_unified_diff_added(diff);
        assert_eq!(
            added.keys().next().map(String::as_str),
            None,
            "a bare `+++` with no a/ or b/ prefix is not a file path"
        );
    }

    #[test]
    fn a_path_is_reported_exactly_as_git_gave_it() {
        // Making the paths agree is git's job, via `--relative`; the parser is
        // not allowed to quietly rewrite them. What it must not do is strip the
        // `b/` prefix and leave the rest alone, since that would put the two
        // sides of the comparison on different bases — the bug this had, where
        // the diff said `rust/crates/…` and lcov said `crates/…`, every file
        // missed, and the answer looked like "no added line was ever exercised".
        let diff = "--- a/rust/crates/x/src/lib.rs\n+++ b/rust/crates/x/src/lib.rs\n@@ -1,0 +2,1 @@\n+fn f() {}\n";
        let added = parse_unified_diff_added(diff);
        assert_eq!(
            added.keys().next().map(String::as_str),
            Some("rust/crates/x/src/lib.rs"),
            "the parser passes the path through; `--relative` is what shortens it"
        );
    }

    #[test]
    fn a_deleted_file_adds_nothing() {
        let diff = "--- a/src/gone.rs\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-fn a() {}\n-fn b() {}\n";
        assert!(parse_unified_diff_added(diff).is_empty());
    }

    #[test]
    fn a_context_line_shifts_the_counter_but_a_removed_one_does_not() {
        let diff = "--- a/f.rs\n+++ b/f.rs\n@@ -1,3 +1,3 @@\n ctx\n+added\n-removed\n";
        assert_eq!(
            parse_unified_diff_added(diff).get("f.rs").unwrap(),
            &vec![2]
        );
    }

    #[test]
    fn lcov_gives_executed_lines_per_file() {
        let tree = Tree::new("lcov").file(
            "cov.lcov",
            "SF:/x/src/lib.rs\nDA:11,3\nDA:12,0\nDA:13,0\nend_of_record\n",
        );
        let hits = parse_lcov(&tree.path().join("cov.lcov"), Path::new("/x"));
        assert_eq!(hits.get("src/lib.rs").unwrap(), &vec![11]);
    }

    #[test]
    fn an_added_line_that_ran_is_covered() {
        let mut added = BTreeMap::new();
        added.insert("f.rs".to_string(), vec![10, 11]);
        let mut hits = BTreeMap::new();
        hits.insert("f.rs".to_string(), vec![10]);
        let mut notes = Vec::new();
        let out = intersect(&added, &hits, &mut notes);
        assert_eq!(out.total_covered(), 1);
        assert_eq!(out.total_uncovered(), 1);
        assert!(!out.is_complete());
    }

    #[test]
    fn a_line_with_no_data_is_unknown_not_uncovered() {
        // The distinction that stops the tool blaming code for its own blind spot.
        let mut added = BTreeMap::new();
        added.insert("f.rs".to_string(), vec![1, 2]);
        let mut notes = Vec::new();
        let out = intersect(&added, &BTreeMap::new(), &mut notes);
        assert_eq!(out.total_uncovered(), 0, "nothing was looked for");
        assert_eq!(out.total_unknown(), 2, "but it is not covered either");
        assert!(out.ratio().is_none(), "no measurable line means no ratio");
        assert!(
            notes.iter().any(|n| n.contains("not counted as uncovered")),
            "{notes:?}"
        );
        assert!(
            out.summary().contains("no coverage data"),
            "{}",
            out.summary()
        );
    }

    #[test]
    fn a_file_the_tool_never_saw_is_named() {
        let mut added = BTreeMap::new();
        added.insert("src/a.rs".to_string(), vec![1]);
        added.insert("notes.md".to_string(), vec![1]);
        let mut hits = BTreeMap::new();
        hits.insert("src/a.rs".to_string(), Vec::new());
        let mut notes = Vec::new();
        let out = intersect(&added, &hits, &mut notes);
        assert_eq!(out.unmeasured, vec!["notes.md".to_string()]);
        assert_eq!(out.unmeasured_lines, 1);
        assert!(notes.iter().any(|n| n.contains("notes.md")), "{notes:?}");
    }

    #[test]
    fn a_ratio_is_refused_rather_than_reported_as_perfect() {
        let mut f = FileCoverage {
            file: "x.rs".to_string(),
            ..FileCoverage::default()
        };
        f.unknown = vec![1, 2, 3];
        assert!(f.ratio().is_none(), "no measurable line is not 100%");
        assert_eq!(f.added(), 3);
    }

    #[test]
    fn a_fully_covered_diff_is_complete() {
        let mut added = BTreeMap::new();
        added.insert("f.rs".to_string(), vec![1, 2]);
        let mut hits = BTreeMap::new();
        hits.insert("f.rs".to_string(), vec![1, 2]);
        let out = intersect(&added, &hits, &mut Vec::new());
        assert!(out.is_complete());
        assert_eq!(out.ratio(), Some(1.0));
    }

    #[test]
    fn a_report_never_claims_a_percentage_it_did_not_measure() {
        // Line 1 ran; line 2 is known and did not. Only these two are countable.
        let mut added = BTreeMap::new();
        added.insert("f.rs".to_string(), vec![1, 2]);
        let mut hits = BTreeMap::new();
        hits.insert("f.rs".to_string(), vec![1]);
        let out = intersect(&added, &hits, &mut Vec::new());
        assert_eq!(out.summary(), "1 of 2 measurable added line(s) executed");
        assert_eq!(out.ratio(), Some(0.5));

        // The same file with no data at all: nothing countable, so no ratio
        // rather than a flattering 1.0 or a damning 0.0.
        let mut notes = Vec::new();
        let unmeasured = intersect(&added, &BTreeMap::new(), &mut notes);
        assert!(unmeasured.ratio().is_none());
        assert!(
            unmeasured.summary().contains("no coverage data"),
            "{}",
            unmeasured.summary()
        );
    }

    #[test]
    fn lcov_paths_outside_the_root_still_report_a_filename() {
        assert_eq!(
            relative(
                Path::new("/elsewhere/deep/src/lib.rs"),
                Path::new("/home/me/proj")
            ),
            "lib.rs"
        );
        assert_eq!(
            relative(
                Path::new("/home/me/proj/rust/crates/x/src/lib.rs"),
                Path::new("/home/me/proj")
            ),
            "rust/crates/x/src/lib.rs"
        );
    }
}
