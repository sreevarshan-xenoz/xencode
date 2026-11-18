//! Turning a unified diff into added-line ranges, and deciding whether a
//! finding sits on a line the diff introduced or one it left untouched.
//!
//! `xencode review` analyses the whole working-tree file, so a finding on an
//! old line is real but says nothing about the change being reviewed. The
//! helper here reads the `@@ -a,b +c,d @@` hunks of the real diff the CLI
//! already fetches and marks which new-file line numbers were added. The CLI
//! then labels each finding introduced or pre-existing. Both parts are pure:
//! no git process, no filesystem, no analysis run happens in this module.

use serde::{Deserialize, Serialize};

/// Where a finding came from relative to the diff under review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FindingOrigin {
    /// The finding is on a `+` line: the change brought it in.
    Introduced,
    /// The finding is on a context line, or a line we cannot prove was added.
    Preexisting,
}

impl FindingOrigin {
    /// Stable lowercase label used in text and JSON output.
    pub fn label(self) -> &'static str {
        match self {
            FindingOrigin::Introduced => "introduced",
            FindingOrigin::Preexisting => "preexisting",
        }
    }
}

/// An inclusive range of new-file line numbers that the diff added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedRange {
    pub start: u32,
    pub end: u32,
}

impl AddedRange {
    pub fn contains(&self, line: u32) -> bool {
        self.start <= line && line <= self.end
    }
}

/// Read the new-file start line out of a `@@ -a,b +c,d @@` header.
///
/// Both the `,b` and `,d` counts are optional in a unified diff (`@@ -1 +1 @@`
/// means a one-line range), so only the `+c` start is read. `None` for any
/// line that is not a well-formed hunk header.
fn parse_hunk_header(line: &str) -> Option<u32> {
    let rest = line.strip_prefix("@@ ")?;
    let mut parts = rest.split_whitespace();
    // The two ranges come before the closing "@@"; anything after that is
    // optional function context and is deliberately ignored.
    let old = parts.next()?;
    let new = parts.next()?;
    if !old.starts_with('-') || parts.next() != Some("@@") {
        return None;
    }
    let new = new.strip_prefix('+')?;
    new.split(',').next()?.parse::<u32>().ok()
}

/// Parse the `@@` hunks of a unified diff into inclusive new-file ranges of
/// added (`+`) lines.
///
/// Removed (`-`) lines consume no new-file line; context (` `) lines consume
/// one and break a run of additions. Lines before the first header (`diff`,
/// `index`, `---`, `+++`) and the `\ No newline at end of file` marker are not
/// part of any hunk body and are ignored. Multiple hunks and multiple files in
/// one diff are handled in order.
pub fn added_line_ranges(diff: &str) -> Vec<AddedRange> {
    let mut ranges = Vec::new();
    let mut new_cursor: Option<u32> = None;
    // A contiguous run of `+` lines in one hunk, as (first, last).
    let mut run: Option<(u32, u32)> = None;

    let flush = |run: &mut Option<(u32, u32)>, ranges: &mut Vec<AddedRange>| {
        if let Some((start, end)) = run.take() {
            ranges.push(AddedRange { start, end });
        }
    };

    for line in diff.lines() {
        if let Some(start) = parse_hunk_header(line) {
            flush(&mut run, &mut ranges);
            new_cursor = Some(start);
            continue;
        }
        // A new `diff --git` section ends the current hunk. Its header lines
        // (including `+++ b/...`) are skipped until the next `@@`.
        if line.starts_with("diff --git ") {
            flush(&mut run, &mut ranges);
            new_cursor = None;
            continue;
        }
        // The no-newline marker annotates the preceding line; it is not itself
        // a line in either file.
        if line.starts_with('\\') {
            continue;
        }
        let Some(cursor) = new_cursor.as_mut() else {
            continue;
        };
        match line.as_bytes().first() {
            Some(b'+') => {
                let n = *cursor;
                match run.as_mut() {
                    Some((_, end)) => *end = n,
                    None => run = Some((n, n)),
                }
                *cursor += 1;
            }
            Some(b' ') => {
                // A context line consumes a new-file line and splits a run.
                flush(&mut run, &mut ranges);
                *cursor += 1;
            }
            Some(b'-') => {
                // A removed line consumes no new-file line, but splits a run.
                flush(&mut run, &mut ranges);
            }
            _ => flush(&mut run, &mut ranges),
        }
    }
    flush(&mut run, &mut ranges);
    ranges
}

/// Classify a one-based line number against already-parsed added ranges.
pub fn classify_issue(line_number: u32, ranges: &[AddedRange]) -> FindingOrigin {
    if ranges.iter().any(|range| range.contains(line_number)) {
        FindingOrigin::Introduced
    } else {
        FindingOrigin::Preexisting
    }
}

/// Convenience for classifying against a single file's diff text.
pub fn classify_issue_in_diff(line_number: u32, diff: &str) -> FindingOrigin {
    classify_issue(line_number, &added_line_ranges(diff))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real `git diff` output captured from a scratch repository. Two hunks,
    /// with context lines, removed lines, and one added line in each hunk.
    const TWO_HUNKS: &str = "\
diff --git a/notes.txt b/notes.txt
index b1a510e..546f889 100644
--- a/notes.txt
+++ b/notes.txt
@@ -1,11 +1,11 @@
 line one
-line two
+CHANGED two
 line three
 line four
 line five
 line six
 line seven
-line eight
+INSERTED eight
 line nine
 line ten
 line eleven
@@ -13,3 +13,4 @@ line twelve
 line thirteen
 line fourteen
 line fifteen
+NEW sixteen
";

    /// Real `git diff` output where the hunk opens with two consecutive added
    /// lines, which must merge into one range.
    const LEADING_ADDITIONS: &str = "\
diff --git a/more.txt b/more.txt
index 85c3040..9e1dc49 100644
--- a/more.txt
+++ b/more.txt
@@ -1,3 +1,5 @@
+NEW first
+NEW second
 alpha
 beta
 gamma
";

    #[test]
    fn parses_two_hunks_with_context_and_removals() {
        assert_eq!(
            added_line_ranges(TWO_HUNKS),
            vec![
                AddedRange { start: 2, end: 2 },
                AddedRange { start: 8, end: 8 },
                AddedRange { start: 16, end: 16 },
            ]
        );
    }

    #[test]
    fn leading_consecutive_additions_form_one_range() {
        assert_eq!(
            added_line_ranges(LEADING_ADDITIONS),
            vec![AddedRange { start: 1, end: 2 }]
        );
    }

    #[test]
    fn multiple_files_and_no_newline_marker_are_handled() {
        // Two file sections joined, with a "\ No newline" marker after an
        // added line. The marker must not consume a line or split the range.
        let diff = format!("{TWO_HUNKS}{LEADING_ADDITIONS}");
        let ranges = added_line_ranges(&diff);
        assert_eq!(
            ranges,
            vec![
                AddedRange { start: 2, end: 2 },
                AddedRange { start: 8, end: 8 },
                AddedRange { start: 16, end: 16 },
                AddedRange { start: 1, end: 2 },
            ]
        );
    }

    #[test]
    fn a_no_newline_marker_does_not_consume_a_line() {
        let diff = "\
diff --git a/x.txt b/x.txt
--- a/x.txt
+++ b/x.txt
@@ -1 +1 @@
-old
\\ No newline at end of file
+new
\\ No newline at end of file
";
        assert_eq!(
            added_line_ranges(diff),
            vec![AddedRange { start: 1, end: 1 }]
        );
    }

    #[test]
    fn single_line_hunk_headers_without_counts_are_parsed() {
        let diff = "\
@@ -1 +1 @@
-old
+new
";
        assert_eq!(
            added_line_ranges(diff),
            vec![AddedRange { start: 1, end: 1 }]
        );
    }

    #[test]
    fn classifies_issues_against_added_ranges() {
        let ranges = added_line_ranges(TWO_HUNKS);
        assert_eq!(classify_issue(2, &ranges), FindingOrigin::Introduced);
        assert_eq!(classify_issue(8, &ranges), FindingOrigin::Introduced);
        assert_eq!(classify_issue(16, &ranges), FindingOrigin::Introduced);
        // Context lines: the pre-existing TODO on line 1 and a later line.
        assert_eq!(classify_issue(1, &ranges), FindingOrigin::Preexisting);
        assert_eq!(classify_issue(9, &ranges), FindingOrigin::Preexisting);
        assert_eq!(classify_issue(100, &ranges), FindingOrigin::Preexisting);
    }

    #[test]
    fn convenience_classifier_uses_the_same_ranges() {
        assert_eq!(
            classify_issue_in_diff(2, TWO_HUNKS),
            FindingOrigin::Introduced
        );
        assert_eq!(
            classify_issue_in_diff(1, TWO_HUNKS),
            FindingOrigin::Preexisting
        );
    }

    #[test]
    fn empty_and_headerless_diff_have_no_additions() {
        assert!(added_line_ranges("").is_empty());
        let headers_only = "diff --git a/x b/x\n--- a/x\n+++ b/x\n";
        assert!(added_line_ranges(headers_only).is_empty());
        assert_eq!(
            classify_issue(1, &added_line_ranges(headers_only)),
            FindingOrigin::Preexisting
        );
    }

    #[test]
    fn origin_labels_are_stable() {
        assert_eq!(FindingOrigin::Introduced.label(), "introduced");
        assert_eq!(FindingOrigin::Preexisting.label(), "preexisting");
    }
}
