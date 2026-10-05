//! The line-length rule cut its snippet at a byte index, so a line whose
//! character happened to straddle the cut did not get a shorter snippet — it
//! aborted the process. Both reproduced cases were written here from the shapes
//! that crashed: a `.py` line with an `é` across byte 100, a `.txt` line with one
//! across byte 120. Any file with an accented identifier, an em-dash in prose or
//! a `# -*- coding: utf-8 -*-` header has that shape.
//!
//! These go through `CodeAnalyzer::analyze_file`, the same entry the
//! `xencode analyze` command uses, rather than the private per-language
//! functions, so what is proven is the run and not just the helper.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use xencode_analysis_rs::analyzer::CodeAnalyzer;
use xencode_analysis_rs::issues::CodeIssue;

fn scratch(label: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "xencode-long-lines-{label}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A line with `ascii_before` ASCII characters, then a two-byte `é`, then `tail`
/// more characters: the character that lands across the cap, which is the whole
/// difficulty.
fn straddling(ascii_before: usize, tail: usize) -> String {
    format!("{}é{}", "a".repeat(ascii_before), "b".repeat(tail))
}

fn too_long(issues: &[CodeIssue]) -> Vec<&CodeIssue> {
    issues
        .iter()
        .filter(|issue| issue.message == "Line too long")
        .collect()
}

#[test]
fn a_python_line_bent_across_the_cut_is_reported_instead_of_crashing() {
    // 99 ASCII characters, then `é` in bytes 99 and 100. The old code tested the
    // byte length, saw more than 100, and sliced at byte 100 — inside the `é`.
    let dir = scratch("python");
    let file = dir.join("straddle.py");
    std::fs::write(&file, format!("{}\n", straddling(99, 3))).unwrap();

    let issues = read(&file);
    let flagged = too_long(&issues);
    assert_eq!(
        flagged.len(),
        1,
        "expected one long-line finding, got {issues:?}"
    );
    // The snippet is 100 *characters*: it keeps the é whole and stops where a
    // byte count would have stopped mid-way through it.
    let snippet = &flagged[0].code_snippet;
    assert_eq!(snippet.chars().count(), 100, "{snippet}");
    assert!(snippet.ends_with('é'), "{snippet}");
    assert_eq!(snippet.len(), 101, "the é made it one byte longer");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_plain_ascii_line_is_still_cut_at_one_hundred_characters() {
    let dir = scratch("ascii");
    let file = dir.join("plain.py");
    std::fs::write(&file, format!("{} = {}\n", "y", "z".repeat(150))).unwrap();

    let issues = read(&file);
    let flagged = too_long(&issues);
    assert_eq!(flagged.len(), 1);
    assert_eq!(flagged[0].code_snippet.chars().count(), 100);
    assert_eq!(flagged[0].column, 100);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_generic_file_cuts_on_a_character_boundary_too() {
    // The second crash: the text analyzer caps at 120, and `é` here occupies
    // bytes 119 and 120.
    let dir = scratch("generic");
    let file = dir.join("notes.txt");
    std::fs::write(&file, format!("{}\n", straddling(119, 2))).unwrap();

    let issues = read(&file);
    let flagged = too_long(&issues);
    assert_eq!(flagged.len(), 1, "the 120-character cap must still bite");
    assert_eq!(flagged[0].code_snippet.chars().count(), 120);
    // The character that used to be split is inside what is reported.
    assert_eq!(
        flagged[0].code_snippet.matches('é').count(),
        1,
        "{}",
        flagged[0].code_snippet
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_line_that_fits_the_screen_is_not_too_long_however_many_bytes_it_uses() {
    // 90 characters of text that is three bytes a character: 270 bytes, and the
    // byte length used to call that a style problem. The rule is about the line
    // as read, so a line of 90 characters is a line of 90 characters.
    let dir = scratch("wide");
    let file = dir.join("prose.txt");
    std::fs::write(&file, format!("{}\n", "漢字".repeat(45))).unwrap();

    let issues = read(&file);
    assert_eq!(
        too_long(&issues).len(),
        0,
        "90 characters is under the 120 cap: {issues:?}"
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap().len(), 271);

    std::fs::remove_dir_all(&dir).unwrap();
}

fn read(path: &Path) -> Vec<CodeIssue> {
    CodeAnalyzer::analyze_file(path).unwrap_or_else(|error| panic!("analysis failed: {error:?}"))
}
