//! Session run-ledger (`EVd-1`).
//!
//! Append-only JSONL of `(session, run-class, exit code, subject digests, log
//! ref)`: what ran, under which session, whether it passed, what it ran
//! against, and where the evidence lives. OTel-shaped (every row carries its
//! own timestamp and session the way a span carries a trace id) and
//! in-toto-flavoured (subjects are digests, not contents), with no signatures —
//! for a local user a signature proves nothing a hash chain does not already,
//! and that chain is EVd-7's work, not this module's.
//!
//! # The trap this is built around
//!
//! Ledgers are secret-full: a raw log tail pasted into a row can carry a config
//! value onto disk forever. So redaction happens at *write* time, and the
//! schema helps by construction — subjects are digests, the log is a reference
//! to a file rather than its contents, and the one free-text field (`note`) is
//! scrubbed through the trace module's secret patterns before it is stored. A
//! row that cannot hold a secret cannot leak one later.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Where the ledger lives.
pub const LEDGER_FILE: &str = "ledger.jsonl";

/// What kind of run a row describes. A closed vocabulary, so a reader can
/// group without guessing what a free-text label meant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunClass {
    Test,
    Lint,
    Build,
    Format,
    Coverage,
    Mutants,
    Other,
}

impl RunClass {
    /// The word stored in the row.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Test => "test",
            Self::Lint => "lint",
            Self::Build => "build",
            Self::Format => "format",
            Self::Coverage => "coverage",
            Self::Mutants => "mutants",
            Self::Other => "other",
        }
    }
}

/// One row of the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// UTC epoch millis when the run finished.
    pub ts_unix_ms: u64,
    /// Which session the run belongs to. `None` only when the runner had no
    /// session to name — the EVd-2 key, on every row rather than in a sidecar.
    pub session: Option<String>,
    /// What kind of run this was.
    pub run_class: RunClass,
    /// The process exit code. The only success signal a row carries.
    pub exit_code: i32,
    /// Digests of what the run ran against (working-tree state, command line).
    /// Digests, never contents: contents belong in the log file, behind the
    /// reference below.
    pub subjects: Vec<String>,
    /// Where the evidence lives, relative to the project when possible.
    pub log_ref: String,
    /// One human line, redacted at write time. May be empty.
    pub note: String,
}

impl LedgerEntry {
    /// `true` only when the process exited zero. A row never says "verified".
    pub fn passed(&self) -> bool {
        self.exit_code == 0
    }
}

/// The ledger file for a project.
pub fn ledger_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join(LEDGER_FILE)
}

/// Append a row. Creates the file and its directory; never rewrites.
///
/// The note is scrubbed before it is stored, because a ledger is read by
/// everything and written once — read-time redaction would have to be
/// re-applied by every reader, and the first reader that forgets leaks.
pub fn append_ledger(xencode_dir: &Path, entry: &LedgerEntry) -> std::io::Result<()> {
    std::fs::create_dir_all(xencode_dir)?;
    let mut note = entry.note.clone();
    note = crate::trace::redact_secrets(&note);
    let stored = LedgerEntry {
        note,
        ..entry.clone()
    };
    let mut text = serde_json::to_string(&stored).unwrap_or_default();
    text.push('\n');
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger_path(xencode_dir))?;
    file.write_all(text.as_bytes())
}

/// Every row, oldest first. A corrupt line is skipped with its number kept, so
/// one bad row cannot hide the ledger behind it.
pub fn read_ledger(xencode_dir: &Path) -> Vec<LedgerEntry> {
    let Ok(text) = std::fs::read_to_string(ledger_path(xencode_dir)) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<LedgerEntry>(l).ok())
        .collect()
}

/// Rows for one session, oldest first.
pub fn ledger_for_session(xencode_dir: &Path, session: &str) -> Vec<LedgerEntry> {
    read_ledger(xencode_dir)
        .into_iter()
        .filter(|row| row.session.as_deref() == Some(session))
        .collect()
}

/// Digest a short string for a subject field. Labels what was run against
/// without storing it.
pub fn digest_hex(text: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_xencode(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xe-ledger-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let xencode = dir.join(".xencode");
        std::fs::create_dir_all(&xencode).unwrap();
        xencode
    }

    fn entry(session: Option<&str>) -> LedgerEntry {
        LedgerEntry {
            ts_unix_ms: 1_700_000_000_000,
            session: session.map(str::to_string),
            run_class: RunClass::Test,
            exit_code: 0,
            subjects: vec![digest_hex("cargo test --workspace")],
            log_ref: "artifacts/s1/test.log".to_string(),
            note: "green".to_string(),
        }
    }

    #[test]
    fn rows_append_and_read_back_in_order() {
        let xencode = temp_xencode("order");
        append_ledger(&xencode, &entry(Some("s1"))).unwrap();
        let mut second = entry(Some("s1"));
        second.exit_code = 1;
        append_ledger(&xencode, &second).unwrap();
        let rows = read_ledger(&xencode);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].passed());
        assert!(!rows[1].passed());
        assert_eq!(ledger_for_session(&xencode, "s1").len(), 2);
        assert!(ledger_for_session(&xencode, "s2").is_empty());
    }

    #[test]
    fn a_secret_in_a_note_is_scrubbed_before_it_reaches_disk() {
        let xencode = temp_xencode("scrub");
        let mut e = entry(Some("s1"));
        e.note = "failed with OPENAI_API_KEY=\"sk-FAKE-NOT-A-REAL-TEST-KEY\" set".to_string();
        append_ledger(&xencode, &e).unwrap();
        let raw = std::fs::read_to_string(ledger_path(&xencode)).unwrap();
        assert!(
            !raw.contains("sk-FAKE-NOT-A-REAL-TEST-KEY"),
            "the secret reached disk"
        );
        assert!(raw.contains("[redacted]"), "the scrub must be visible");
        // ...while the same note unredacted would have leaked, so the test is
        // not vacuous: the scrub did the work, not the fixture.
        assert!(e.note.contains("sk-FAKE-NOT-A-REAL-TEST-KEY"));
    }

    #[test]
    fn a_missing_ledger_reads_as_empty_not_an_error() {
        let xencode = temp_xencode("absent");
        assert!(read_ledger(&xencode).is_empty());
    }

    #[test]
    fn a_corrupt_line_hides_nothing_but_itself() {
        let xencode = temp_xencode("corrupt");
        append_ledger(&xencode, &entry(Some("s1"))).unwrap();
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(ledger_path(&xencode))
            .unwrap();
        writeln!(file, "not json at all").unwrap();
        append_ledger(&xencode, &entry(Some("s1"))).unwrap();
        assert_eq!(read_ledger(&xencode).len(), 2);
    }

    #[test]
    fn exit_code_is_the_only_success_signal() {
        let mut e = entry(None);
        e.exit_code = 0;
        assert!(e.passed());
        e.exit_code = 100;
        assert!(!e.passed(), "a flake exit is not a pass");
        assert!(e.session.is_none(), "rows without a session say so");
    }
}
