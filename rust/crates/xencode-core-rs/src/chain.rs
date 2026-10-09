//! A hash chain over JSONL records: the tamper evidence the workspace audit
//! log introduced (EV-11), shared so every append-only log that wants it uses
//! the same primitive (EVd-7 put the run ledger on it).
//!
//! A record carries `prev` (the digest of the record before it) and `digest`
//! (a hash over this record's own contents, `prev` included). Changing any
//! earlier line therefore breaks the link the next line names, and changing
//! the last line breaks its own digest. [`verify_chain`] walks a file and says
//! what does not add up.
//!
//! What a hash chain without a key cannot do, stated plainly: anyone who can
//! rewrite the whole file can recompute the chain as they rewrite it, and
//! nothing here would notice. Nor can it prove a writer *omitted* a record, or
//! that the tail was not cut off and the remainder re-chained from an earlier
//! point. This makes casual editing detectable; it does not make a log
//! trustworthy against someone with write access and the source.

use std::path::Path;

/// What the `prev` field of the first record of a chain holds. Hashing a name
/// rather than using a row of zeros keeps it a real digest of something a
/// reader can recompute. Every chained log starts from this one value, so one
/// verifier (`xencode audit verify <file>`) checks any of them.
pub fn genesis() -> String {
    digest_of(b"xencode-audit/1")
}

/// SHA-256 as lowercase hex.
pub fn digest_of(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Build one chained record from `record` (a JSON object): the record's fields,
/// plus the digest it links to and its own digest. Returns the line to write
/// and the digest the next record must name.
///
/// `serde_json` writes an object's keys in sorted order, which is what makes
/// this reproducible: the same record built again hashes to the same value,
/// and a verifier that parses and re-serializes is comparing the same bytes
/// the writer hashed.
pub fn chained_line(
    mut record: serde_json::Value,
    prev: &str,
) -> serde_json::Result<(String, String)> {
    record["prev"] = serde_json::Value::String(prev.to_string());
    let body = serde_json::to_string(&record)?;
    let digest = digest_of(body.as_bytes());
    record["digest"] = serde_json::Value::String(digest.clone());
    Ok((serde_json::to_string(&record)?, digest))
}

/// What a record's link value is, for the purposes of the record after it.
///
/// A line written before its log was chained has no `digest` to hand on, and a
/// file that already exists is not something new code can rewrite. So such a
/// line links by the hash of its own text as written, which still ties the two
/// together: dropping or editing that line breaks the link the next record
/// carries. A line that *does* have a digest uses it, even though that is not
/// the hash of its text — the digest field is part of the text, so stripping
/// it changes the hash and is caught as the broken link it is.
pub fn link_of(raw: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(raw).ok()?;
    match parsed.get("digest").and_then(|v| v.as_str()) {
        Some(digest) => Some(digest.to_string()),
        None => Some(digest_of(raw.as_bytes())),
    }
}

/// Where a writer appending to an existing file picks the chain back up.
pub fn continue_from(path: &Path) -> String {
    let Ok(text) = std::fs::read_to_string(path) else {
        return genesis();
    };
    // A file that ends mid-line left the tail unpaired, the way a crash leaves
    // it; the digest of the last whole record is still the right thing to link
    // to, because the next verify reports the tail as damaged and no further.
    let last = text.lines().rfind(|line| !line.trim().is_empty());
    match last.and_then(link_of) {
        Some(link) => link,
        None => genesis(),
    }
}

/// What checking one record against the chain found wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainProblem {
    /// The line is not a JSON object, so nothing about it can be checked.
    NotARecord,
    /// The contents do not hash to the digest the line claims.
    ContentChanged,
    /// The line names a predecessor that is not the record before it: this
    /// record was moved, an earlier one was edited, or one was removed.
    LinkMismatch,
    /// The line carries no chain fields although the records before it do —
    /// which is what an appended forgery looks like when whoever added it did
    /// not know the chain existed.
    MissingChain,
}

/// One entry in a [`ChainReport`], with the 1-based line it refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoundProblem {
    pub line: usize,
    pub problem: ChainProblem,
}

/// What checking a whole chained log found.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ChainReport {
    /// Records that parsed as JSON objects.
    pub records: usize,
    /// How many of them were written before the log was chained, and so carry
    /// nothing that proves they are what the writer wrote.
    pub unchained: usize,
    /// Every problem, in line order.
    pub problems: Vec<FoundProblem>,
    /// The file stopped in the middle of a line. That is what a crash or a full
    /// disk leaves behind, not evidence of anyone editing the log, so it is
    /// reported separately rather than counted as tampering.
    pub torn_tail: bool,
}

impl ChainReport {
    /// True when every record that can be checked adds up.
    pub fn intact(&self) -> bool {
        self.problems.is_empty()
    }

    /// Every record is chained and the chain is unbroken.
    pub fn fully_chained(&self) -> bool {
        self.intact() && self.unchained == 0
    }
}

/// Check a chained log, given its whole text.
///
/// Each record has to hash to its own `digest` and name the digest of the
/// record before it, so any edit short of rewriting every later record shows
/// up as a problem on a specific line.
pub fn verify_chain(text: &str) -> ChainReport {
    let mut report = ChainReport::default();
    let mut expected_link = genesis();
    let mut chain_started = false;
    let lines: Vec<&str> = text.lines().collect();
    let last_line = last_content_line(&lines);
    for (index, raw) in lines.iter().enumerate() {
        let line = index + 1;
        if raw.trim().is_empty() {
            continue;
        }
        let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw) else {
            if index == last_line {
                // Only the final line can be torn by an interrupted write: a
                // writer holds its lock across a whole line.
                report.torn_tail = true;
            } else {
                report.problems.push(FoundProblem {
                    line,
                    problem: ChainProblem::NotARecord,
                });
            }
            continue;
        };
        let Some(object) = parsed.as_object() else {
            report.problems.push(FoundProblem {
                line,
                problem: ChainProblem::NotARecord,
            });
            continue;
        };
        report.records += 1;
        let this_link = link_of(raw).unwrap_or_default();
        match object.get("digest").and_then(|v| v.as_str()) {
            None => {
                if chain_started {
                    // The records before this one were chained, so a line with
                    // no chain fields here was not written by the program.
                    report.problems.push(FoundProblem {
                        line,
                        problem: ChainProblem::MissingChain,
                    });
                }
                report.unchained += 1;
            }
            Some(claimed) => {
                chain_started = true;
                let mut body = parsed.clone();
                if let Some(map) = body.as_object_mut() {
                    map.remove("digest");
                }
                let body = serde_json::to_string(&body).unwrap_or_default();
                if digest_of(body.as_bytes()) != claimed {
                    report.problems.push(FoundProblem {
                        line,
                        problem: ChainProblem::ContentChanged,
                    });
                }
            }
        }
        // The link check applies to unchained records too: what a line is
        // supposed to name depends only on the line before it.
        if let Some(prev) = object.get("prev").and_then(|v| v.as_str()) {
            if prev != expected_link {
                report.problems.push(FoundProblem {
                    line,
                    problem: ChainProblem::LinkMismatch,
                });
            }
        }
        expected_link = this_link;
    }
    report
}

/// The index of the last line with any content in it, ignoring a trailing
/// newline, which is what an append leaves after the final record.
fn last_content_line(lines: &[&str]) -> usize {
    lines
        .iter()
        .rposition(|line| !line.trim().is_empty())
        .unwrap_or(0)
}

/// A problem as one sentence. `writer` names who appends to this log ("the
/// server", "xencode"), for the forgery case.
pub fn describe(problem: &FoundProblem, writer: &str) -> String {
    match problem.problem {
        ChainProblem::NotARecord => format!(
            "line {}: not a JSON object, so nothing about this record can be checked",
            problem.line
        ),
        ChainProblem::ContentChanged => format!(
            "line {}: the contents do not match the digest recorded on this line",
            problem.line
        ),
        ChainProblem::LinkMismatch => format!(
            "line {}: it names a predecessor that is not the record before it — this line \
             was moved, an earlier one was edited, or one was removed",
            problem.line
        ),
        ChainProblem::MissingChain => format!(
            "line {}: it carries no digest at all, although the records before it do — this \
             line was not appended by {writer}",
            problem.line
        ),
    }
}
