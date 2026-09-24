//! Append-only JSONL mirror of the workspace audit trail.
//!
//! The trail itself lives in `WorkspaceManager` (in memory, ordered by a
//! monotonic `seq`); this sink persists every event it has not seen yet to
//! disk, one JSON object per line. Losing audit writes must never take down
//! the session plane, so the first IO error warns once and disables the
//! sink — the in-memory trail stays complete either way.
//!
//! Each line is also linked to the one before it, so the file can be checked
//! for edits made after the fact. A record carries `prev` (the digest of the
//! record before it) and `digest` (a hash over this record's own contents
//! including `prev`). Changing any earlier line therefore invalidates the
//! digest recorded on the *next* one, and changing the last line invalidates
//! its own. [`verify_chain`] walks a file and says what does not add up;
//! `xencode audit verify` prints that, and [`describe`] turns a problem into
//! a sentence.
//!
//! What a hash chain without a key cannot do, stated plainly: anyone who can
//! rewrite the whole file can recompute the chain as they rewrite it, and
//! nothing here would notice. Nor can it prove a server *omitted* an event it
//! decided not to record, or that the tail was not cut off and the remainder
//! re-chained from an earlier point. This makes casual editing detectable; it
//! does not make the log trustworthy against someone with write access and the
//! source.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use tracing::warn;
use xencode_collaboration_rs::{AuditEvent, WorkspaceManager};

/// What the `prev` field of the first record of a chain holds. Hashing a name
/// rather than using a row of zeros keeps it a real digest of something a
/// reader can recompute.
fn genesis() -> String {
    digest_of(b"xencode-audit/1")
}

/// SHA-256 as lowercase hex.
fn digest_of(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

struct State {
    /// `Some(None)` means the sink gave up after an IO error; `None` means
    /// no file was ever configured (or it is not open yet).
    file: Option<Option<File>>,
    path: Option<PathBuf>,
    /// Highest `seq` already written, so wiring is one `sync_from` call
    /// per mutation site and restarts append without duplicates.
    last_seq: u64,
    /// Digest the next record has to name as its `prev`. Read from the end of
    /// the file the first time this sink writes, which is what lets a server
    /// restart continue the chain instead of silently starting a new one.
    prev_link: String,
    /// Whether [`State::prev_link`] came from the file or is still a guess.
    linked: bool,
}

impl State {
    fn new(path: Option<PathBuf>) -> Self {
        Self {
            file: None,
            path,
            last_seq: 0,
            prev_link: genesis(),
            linked: false,
        }
    }
}

/// Persists `WorkspaceManager` audit events to a JSONL file.
pub struct AuditSink {
    state: Mutex<State>,
}

impl AuditSink {
    /// A sink that does no IO at all — the default for tests and for
    /// servers started with `--audit-path none`.
    pub fn disabled() -> Self {
        Self {
            state: Mutex::new(State::new(None)),
        }
    }

    /// A sink appending to `path`, created on first write.
    pub fn to_file(path: impl Into<PathBuf>) -> Self {
        Self {
            state: Mutex::new(State::new(Some(path.into()))),
        }
    }

    pub fn is_enabled(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.path.is_some() && !matches!(&state.file, Some(None))
    }

    /// Number of events this sink has written.
    pub fn written(&self) -> u64 {
        self.state.lock().unwrap().last_seq
    }

    /// Mirror every event in the manager's trail newer than the last one
    /// this sink wrote. Safe to call after any mutation; cheap no-op
    /// otherwise.
    pub fn sync_from(&self, workspaces: &WorkspaceManager) {
        let events: Vec<AuditEvent> = workspaces
            .audit_log()
            .iter()
            .filter(|e| e.seq > self.state.lock().unwrap().last_seq)
            .cloned()
            .collect();
        if events.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        let wrote_all = events
            .iter()
            .all(|event| Self::write_line(&mut state, event));
        if wrote_all {
            state.last_seq = events[events.len() - 1].seq;
        }
    }

    /// Returns false once the sink is dead. The lock is held across the
    /// write on purpose: audit lines must not interleave mid-record.
    fn write_line(state: &mut State, event: &AuditEvent) -> bool {
        let Some(path) = state.path.clone() else {
            return false;
        };
        if matches!(&state.file, Some(None)) {
            return false;
        }
        if !state.linked {
            state.prev_link = continue_from(&path);
            state.linked = true;
        }
        let line = match chained_line(event, &state.prev_link) {
            Ok((line, digest)) => {
                state.prev_link = digest;
                line
            }
            Err(e) => {
                warn!(
                    "audit sink disabled: cannot serialize event {}: {e}",
                    event.seq
                );
                state.file = Some(None);
                return false;
            }
        };
        Self::append(state, &path, &line)
    }

    /// Open the file if needed and put one complete line at the end of it.
    fn append(state: &mut State, path: &std::path::Path, line: &str) -> bool {
        if matches!(&state.file, Some(None)) {
            return false;
        }
        if state.file.is_none() {
            state.file = Some(
                match OpenOptions::new().append(true).create(true).open(path) {
                    Ok(file) => Some(file),
                    Err(e) => {
                        warn!("audit sink disabled: cannot open {}: {e}", path.display());
                        None
                    }
                },
            );
        }
        let Some(Some(file)) = state.file.as_mut() else {
            return false;
        };
        match file
            .write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
        {
            Ok(()) => true,
            Err(e) => {
                warn!(
                    "audit sink disabled: write to {} failed: {e}",
                    path.display()
                );
                state.file = Some(None);
                false
            }
        }
    }
}

/// Build one record: the event's fields, plus the digest it links to and its
/// own digest. Returns the line to write and the digest the next record must
/// name.
///
/// `serde_json` writes an object's keys in sorted order, which is what makes
/// this reproducible: the same record built again hashes to the same value,
/// and a verifier that parses and re-serializes is comparing the same bytes
/// the writer hashed.
fn chained_line(event: &AuditEvent, prev: &str) -> serde_json::Result<(String, String)> {
    let mut record = serde_json::to_value(event)?;
    record["prev"] = serde_json::Value::String(prev.to_string());
    let body = serde_json::to_string(&record)?;
    let digest = digest_of(body.as_bytes());
    record["digest"] = serde_json::Value::String(digest.clone());
    Ok((serde_json::to_string(&record)?, digest))
}

/// What a record's link value is, for the purposes of the record after it.
///
/// A line written before this change has no `digest` to hand on, and a log
/// that already exists on disk is not something the new code can rewrite. So
/// such a line links by the hash of its own text as written, which still ties
/// the two files together: dropping or editing that line breaks the link the
/// next record carries. A line that *does* have a digest uses it, even though
/// that is not the hash of its text — the digest field is part of the text, so
/// stripping it changes the hash and is caught as the broken link it is.
fn link_of(raw: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(raw).ok()?;
    match parsed.get("digest").and_then(|v| v.as_str()) {
        Some(digest) => Some(digest.to_string()),
        None => Some(digest_of(raw.as_bytes())),
    }
}

/// Where a sink appending to an existing file picks the chain back up.
fn continue_from(path: &std::path::Path) -> String {
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

/// What checking a whole audit log found.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ChainReport {
    /// Records that parsed as JSON objects.
    pub records: usize,
    /// How many of them were written before the log was chained, and so carry
    /// nothing that proves they are what the server wrote.
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

/// Check an audit log's chain, given its whole text.
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
                // Only the final line can be torn by an interrupted write: the
                // sink holds its lock across a whole line.
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
                    // no chain fields here was not written by this program.
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

/// A problem as one sentence, for `xencode audit verify` and for logs.
pub fn describe(problem: &FoundProblem) -> String {
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
             line was not appended by the server",
            problem.line
        ),
    }
}

/// A whole-file verdict as one sentence.
pub fn verdict(path: &std::path::Path, report: &ChainReport) -> String {
    if report.records == 0 {
        return format!("{}: no records to check", path.display());
    }
    let mut text = match report.problems.first() {
        None => format!(
            "{}: {} records, chain intact",
            path.display(),
            report.records
        ),
        Some(first) => format!(
            "{}: {} records, chain broken — {}",
            path.display(),
            report.records,
            describe(first)
        ),
    };
    if report.problems.len() > 1 {
        text += &format!(", and {} further problems", report.problems.len() - 1);
    }
    if report.unchained > 0 {
        text += &format!(
            "; {} of them were written before this log was chained and prove nothing about themselves",
            report.unchained
        );
    }
    if report.torn_tail {
        text +=
            "; the file ends in the middle of a line, which is what an interrupted write leaves";
    }
    text
}

impl Default for AuditSink {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager_with_trail() -> WorkspaceManager {
        let mut wm = WorkspaceManager::new();
        wm.create_workspace_with_id("s1", "session", "alice");
        wm.add_member("s1", "alice", "bob", xencode_collaboration_rs::Role::Editor)
            .unwrap();
        wm
    }

    fn read_lines(path: &std::path::Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).expect("each audit line must be valid JSON"))
            .collect()
    }

    #[test]
    fn a_file_sink_writes_one_json_line_per_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let sink = AuditSink::to_file(&path);

        sink.sync_from(&manager_with_trail());

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 2, "created + member_added, one line each");
        assert_eq!(lines[0]["action"], "workspace_created");
        assert_eq!(lines[1]["action"], "member_added");
        assert_eq!(lines[1]["actor"], "alice");
    }

    /// The restart story: a fresh sink on the same file appends rather
    /// than truncates, so history survives the server going away.
    #[test]
    fn a_second_sink_appends_to_the_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");

        AuditSink::to_file(&path).sync_from(&manager_with_trail());
        let mut wm2 = WorkspaceManager::new();
        wm2.create_workspace_with_id("s2", "later", "carol");
        AuditSink::to_file(&path).sync_from(&wm2);

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2]["target"], "s2");
    }

    #[test]
    fn a_sink_does_not_rewrite_events_it_already_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let sink = AuditSink::to_file(&path);
        let wm = manager_with_trail();

        sink.sync_from(&wm);
        sink.sync_from(&wm);
        sink.sync_from(&wm);

        assert_eq!(read_lines(&path).len(), 2);
        assert_eq!(sink.written(), 2);
    }

    /// An impossible path must degrade to a warning, not a panic — and
    /// after that the sink stays quiet.
    #[test]
    fn an_unwritable_path_disables_the_sink_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        // A directory where a file should be: every open fails.
        let blocked = dir.path().join("blocked.jsonl");
        std::fs::create_dir(&blocked).unwrap();
        let sink = AuditSink::to_file(&blocked);
        assert!(sink.is_enabled());

        sink.sync_from(&manager_with_trail());

        assert!(
            !sink.is_enabled(),
            "the failed open should disable the sink"
        );
        assert_eq!(sink.written(), 0);
        // Still no panic on further events.
        sink.sync_from(&manager_with_trail());
    }

    #[test]
    fn a_disabled_sink_does_no_io() {
        let sink = AuditSink::disabled();
        assert!(!sink.is_enabled());

        sink.sync_from(&manager_with_trail());

        assert_eq!(sink.written(), 0);
        assert!(!sink.is_enabled());
    }

    /// The event the sink persists must survive the sink's own serde
    /// round trip: `seq` and `at` included, since auditors sort by one and
    /// cross-reference the other.
    #[test]
    fn persisted_lines_carry_the_full_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let sink = AuditSink::to_file(&path);
        let wm = manager_with_trail();

        sink.sync_from(&wm);

        let lines = read_lines(&path);
        let in_memory = wm.audit_log();
        for (line, event) in lines.iter().zip(in_memory) {
            assert_eq!(line["seq"], event.seq);
            assert_eq!(line["at"], event.at);
            assert_eq!(line["actor"], event.actor.to_string());
            assert_eq!(line["target"], event.target);
            assert!(line["detail"].is_string());
        }
        assert!(
            lines
                .windows(2)
                .all(|w| w[0]["seq"].as_u64() < w[1]["seq"].as_u64()),
            "seq must be strictly increasing on disk"
        );
    }

    fn logged(dir: &std::path::Path) -> std::path::PathBuf {
        let path = dir.join("audit.jsonl");
        AuditSink::to_file(&path).sync_from(&manager_with_trail());
        path
    }

    fn read(path: &std::path::Path) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    /// Edit one field of one record, the way someone with an editor would.
    fn tamper_line(text: &str, line: usize, needle: &str, replacement: &str) -> String {
        let mut lines: Vec<String> = text.lines().map(|line| line.to_string()).collect();
        let target = &mut lines[line - 1];
        assert!(
            target.contains(needle),
            "line {line} of the log does not contain {needle}"
        );
        *target = target.replace(needle, replacement);
        format!("{}\n", lines.join("\n"))
    }

    fn kinds(report: &ChainReport) -> Vec<(usize, ChainProblem)> {
        report
            .problems
            .iter()
            .map(|problem| (problem.line, problem.problem))
            .collect()
    }

    #[test]
    fn every_record_a_sink_writes_is_linked_and_verifies() {
        let dir = tempfile::tempdir().unwrap();
        let report = verify_chain(&read(&logged(dir.path())));

        assert_eq!(report.records, 2);
        assert_eq!(report.unchained, 0, "nothing here predates chaining");
        assert!(report.intact(), "{:?}", report.problems);
    }

    #[test]
    fn editing_who_did_something_is_found_by_that_records_own_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = logged(dir.path());
        let tampered = tamper_line(
            &read(&path),
            1,
            r#""actor":"alice""#,
            r#""actor":"mallory""#,
        );

        let report = verify_chain(&tampered);
        assert_eq!(
            kinds(&report),
            [(1, ChainProblem::ContentChanged)],
            "changing the actor on the first line has to be reported on that line"
        );
        assert!(!report.intact(), "{}", verdict(&path, &report));
    }

    #[test]
    fn removing_a_record_is_found_at_the_one_after_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = logged(dir.path());
        let text = read(&path);
        let mut kept: Vec<&str> = text.lines().collect();
        kept.remove(0);

        let report = verify_chain(&format!("{}\n", kept[0]));
        assert_eq!(
            kinds(&report),
            [(1, ChainProblem::LinkMismatch)],
            "the surviving record still names the one that was taken out"
        );
    }

    #[test]
    fn adding_a_record_that_never_came_from_the_server_is_found() {
        let dir = tempfile::tempdir().unwrap();
        let path = logged(dir.path());
        let forged = format!(
            "{}{{\"seq\":9,\"at\":\"later\",\"actor\":\"mallory\",\"action\":\"member_added\",\"target\":\"s1\",\"detail\":\"self-promoted to admin\"}}\n",
            read(&path)
        );

        let report = verify_chain(&forged);
        assert_eq!(
            kinds(&report),
            [(3, ChainProblem::MissingChain)],
            "a line with no digest, after lines that have them, was not appended here"
        );
    }

    #[test]
    fn rewriting_the_last_record_is_found_by_its_own_digest() {
        // The case a link alone cannot cover: nothing follows the last line, so
        // only the digest inside it can say whether its contents are what the
        // server wrote.
        let dir = tempfile::tempdir().unwrap();
        let path = logged(dir.path());
        let tampered = tamper_line(&read(&path), 2, "bob", "mallory");

        assert_eq!(
            kinds(&verify_chain(&tampered)),
            [(2, ChainProblem::ContentChanged)]
        );
    }

    #[test]
    fn cutting_the_tail_off_a_log_leaves_a_shorter_chain_that_still_verifies() {
        // Not a flaw to fix here but the limit of the technique, so it is
        // written down where someone will trip over it: whoever can edit the
        // file can delete the end of it and re-link nothing. Proving the log is
        // complete needs an anchor outside the file, which is a different piece
        // of work.
        let dir = tempfile::tempdir().unwrap();
        let path = logged(dir.path());
        let first = read(&path).lines().next().unwrap().to_string();

        let report = verify_chain(&format!("{first}\n"));
        assert_eq!(report.records, 1);
        assert!(report.intact(), "{:?}", report.problems);
    }

    #[test]
    fn a_record_left_half_written_is_reported_as_an_interrupted_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = logged(dir.path());
        let text = read(&path);
        let whole = text.lines().last().unwrap();
        let torn = &whole[..whole.len() / 2];

        let report = verify_chain(&format!("{}\n{torn}", text.lines().next().unwrap()));
        assert!(report.torn_tail, "the file ends inside a record");
        assert!(report.intact(), "{:?}", report.problems);
        assert_eq!(report.records, 1);
    }

    #[test]
    fn a_log_written_before_chaining_links_into_the_records_after_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        // What the sink used to write: the event and nothing else.
        let legacy = serde_json::to_string(&manager_with_trail().audit_log()[0]).unwrap();
        std::fs::write(&path, format!("{legacy}\n")).unwrap();

        let mut workspaces = WorkspaceManager::new();
        workspaces.create_workspace_with_id("s9", "after the change", "carol");
        AuditSink::to_file(&path).sync_from(&workspaces);

        let text = read(&path);
        let report = verify_chain(&text);
        assert_eq!(report.records, 2);
        assert_eq!(report.unchained, 1, "the older line predates the chain");
        assert!(report.intact(), "{:?}", kinds(&report));
        // The new line is tied to the old one by its text, so deleting the
        // legacy record breaks the chain rather than going unnoticed.
        let without_legacy = text.lines().last().unwrap();
        assert_eq!(
            kinds(&verify_chain(&format!("{without_legacy}\n"))),
            [(1, ChainProblem::LinkMismatch)]
        );
    }

    #[test]
    fn a_server_restarting_mid_log_continues_the_same_chain() {
        let dir = tempfile::tempdir().unwrap();
        let path = logged(dir.path());

        let mut later = WorkspaceManager::new();
        later.create_workspace_with_id("s2", "next boot", "carol");
        AuditSink::to_file(&path).sync_from(&later);

        let report = verify_chain(&read(&path));
        assert_eq!(report.records, 3);
        assert!(report.fully_chained(), "{:?}", kinds(&report));
    }
}
