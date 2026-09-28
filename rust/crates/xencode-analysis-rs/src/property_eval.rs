//! VF-4 evaluation: do properties constrain behaviour, or only run?
//!
//! Target: `mutation::touched_files`. Four seeded defects break one rule each.
//! Each property is scored by how many defects it catches on fixed fixtures.

use crate::mutation::touched_files;
use proptest::prelude::*;
use std::collections::BTreeSet;

type Impl = fn(&str) -> Vec<String>;

fn correct(patch: &str) -> Vec<String> {
    touched_files(patch)
}

fn keep_devnull(patch: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = rest.trim().strip_prefix("b/").unwrap_or(rest.trim());
            if !out.iter().any(|p: &String| p == path) {
                out.push(path.to_string());
            }
        }
    }
    out
}

fn allow_duplicates(patch: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = rest.trim();
            if path == "/dev/null" {
                continue;
            }
            out.push(path.strip_prefix("b/").unwrap_or(path).to_string());
        }
    }
    out
}

fn keep_prefix(patch: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = rest.trim();
            if path == "/dev/null" {
                continue;
            }
            if !out.iter().any(|p: &String| p == path) {
                out.push(path.to_string());
            }
        }
    }
    out
}

fn drop_new_files(patch: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines = patch.lines().peekable();
    while let Some(line) = lines.next() {
        if line.starts_with("--- /dev/null") {
            let _ = lines.next();
            continue;
        }
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = rest.trim();
            if path == "/dev/null" {
                continue;
            }
            let path = path.strip_prefix("b/").unwrap_or(path);
            if !out.iter().any(|p: &String| p == path) {
                out.push(path.to_string());
            }
        }
    }
    out
}

const NEW_ONLY: &str = "--- /dev/null\n+++ b/tests/new.rs\n";
const DELETED_ONLY: &str = "--- a/src/gone.rs\n+++ /dev/null\n";
const DUPLICATE: &str = "--- a/a.rs\n+++ b/a.rs\n--- a/a.rs\n+++ b/a.rs\n";
const MIXED: &str = "--- a/src/lib.rs\n+++ b/src/lib.rs\n--- /dev/null\n+++ b/tests/new.rs\n";
const FIXTURES: [&str; 4] = [NEW_ONLY, DELETED_ONLY, DUPLICATE, MIXED];
const VARIANTS: [(&str, Impl); 4] = [
    ("keep /dev/null", keep_devnull),
    ("allow duplicates", allow_duplicates),
    ("keep b/ prefix", keep_prefix),
    ("drop new files", drop_new_files),
];

fn p0_vacuous(output: &[String], patch: &str) -> bool {
    output.len() <= patch.len()
}

fn p1_weak(output: &[String], patch: &str) -> bool {
    !patch.lines().any(|l| l.starts_with("+++ b/")) || !output.is_empty()
}

fn p2_candidate(output: &[String], _patch: &str) -> bool {
    !output
        .iter()
        .any(|p| p == "/dev/null" || p.starts_with("b/"))
}

fn expected_exact(patch: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("+++ ") {
            let path = rest.trim();
            if path == "/dev/null" {
                continue;
            }
            let path = path.strip_prefix("b/").unwrap_or(path);
            if !out.iter().any(|p: &String| p == path) {
                out.push(path.to_string());
            }
        }
    }
    out
}

fn p3_exact(output: &[String], patch: &str) -> bool {
    output == expected_exact(patch).as_slice()
}

fn catches(predicate: fn(&[String], &str) -> bool, variant: Impl) -> bool {
    FIXTURES
        .iter()
        .any(|patch| !predicate(&variant(patch), patch))
}

fn score(predicate: fn(&[String], &str) -> bool) -> usize {
    VARIANTS
        .iter()
        .filter(|(_, v)| catches(predicate, *v))
        .count()
}

#[derive(Clone, Debug)]
struct FileEdit {
    name: u8,
    kind: u8,
}

fn edit_name(n: u8) -> &'static str {
    match n % 3 {
        0 => "src/lib.rs",
        1 => "tests/new.rs",
        _ => "docs/note.md",
    }
}

fn patch_text(edits: &[FileEdit]) -> String {
    let mut text = String::new();
    for edit in edits {
        let name = edit_name(edit.name);
        match edit.kind % 3 {
            0 => {
                text.push_str(&format!("--- a/{name}\n+++ b/{name}\n"));
            }
            1 => {
                text.push_str(&format!("--- /dev/null\n+++ b/{name}\n"));
            }
            _ => {
                text.push_str(&format!("--- a/{name}\n+++ /dev/null\n"));
            }
        }
    }
    text
}

fn plus_paths(patch: &str) -> Vec<String> {
    patch
        .lines()
        .filter_map(|l| l.strip_prefix("+++ b/"))
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn generated_patches_never_report_devnull_or_prefixes(
        edits in prop::collection::vec(
            (0u8..3, 0u8..3).prop_map(|(name, kind)| FileEdit { name, kind }),
            1..6,
        )
    ) {
        let patch = patch_text(&edits);
        let output = correct(&patch);
        prop_assert!(!output.iter().any(|p| p == "/dev/null" || p.starts_with("b/")));
        let unique: BTreeSet<&String> = output.iter().collect();
        prop_assert_eq!(unique.len(), output.len());
        for wanted in plus_paths(&patch) {
            prop_assert!(output.contains(&wanted), "missing {wanted} in {output:?}");
        }
    }
}

#[test]
fn effectiveness_is_measured_not_claimed() {
    assert_eq!(
        score(p0_vacuous),
        0,
        "a property asserting nothing catches nothing"
    );
    assert_eq!(
        score(p1_weak),
        1,
        "the weak property misses almost everything"
    );
    assert_eq!(
        score(p2_candidate),
        2,
        "the candidate LLM property is partial"
    );
    assert_eq!(
        score(p3_exact),
        4,
        "the exact property catches every seeded defect"
    );
}

#[test]
fn each_seeded_defect_is_real() {
    assert!(catches(p3_exact, keep_devnull));
    assert!(catches(p3_exact, allow_duplicates));
    assert!(catches(p3_exact, keep_prefix));
    assert!(catches(p3_exact, drop_new_files));
}

#[test]
fn the_existing_example_test_catches_half_the_seeded_defects() {
    let fixture = "--- a/src/lib.rs\n+++ b/src/lib.rs\n--- /dev/null\n+++ b/tests/new.rs\n";
    assert_eq!(correct(fixture), vec!["src/lib.rs", "tests/new.rs"]);
    let caught = VARIANTS
        .iter()
        .filter(|(_, v)| v(fixture) != correct(fixture))
        .count();
    assert_eq!(caught, 2, "no deletion or duplicate in that fixture");
}
