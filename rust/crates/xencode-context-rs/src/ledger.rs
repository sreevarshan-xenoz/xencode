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
    // Chained onto the row before it (EVd-7), so an edit made to the file
    // afterwards shows up when it is checked: `verify_ledger`, or
    // `xencode audit verify .xencode/ledger.jsonl`, which reads every chained
    // log the same way.
    let path = ledger_path(xencode_dir);
    let prev = xencode_core_rs::chain::continue_from(&path);
    let value = serde_json::to_value(&stored).map_err(std::io::Error::other)?;
    let (mut text, _digest) =
        xencode_core_rs::chain::chained_line(value, &prev).map_err(std::io::Error::other)?;
    text.push('\n');
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(text.as_bytes())
}

/// Check the ledger's chain: every row hashes to its own digest and names the
/// row before it. Rows written before the ledger was chained are counted, not
/// failed. This proves the file is self-consistent, nothing more: there is no
/// key, so whoever rewrites the whole file can rewrite the chain with it.
pub fn verify_ledger(xencode_dir: &Path) -> Result<xencode_core_rs::chain::ChainReport, String> {
    // A ledger that exists but cannot be read is an error, not an empty and
    // therefore "intact" one.
    let path = ledger_path(xencode_dir);
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(xencode_core_rs::chain::verify_chain(&text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Ok(xencode_core_rs::chain::ChainReport::default())
        }
        Err(e) => Err(format!("cannot read {}: {e}", path.display())),
    }
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

/// The result envelope for one session (AE-1, the producer `OR-16` never had):
/// one command per ledger row the session wrote, each pointing at the row it
/// came from — its line in `ledger.jsonl` and, for a chained row, its digest —
/// so every exit code a reviewer reads resolves to a run that happened. A
/// session with no rows proved nothing and is `Blocked`, never `Completed`; any
/// non-zero exit is `Failed`. `changed_files` comes from the caller's diff.
pub fn envelope_for_session(
    xencode_dir: &Path,
    session: &str,
    task: &str,
    changed_files: Vec<String>,
) -> xencode_core_rs::ResultEnvelope {
    use xencode_core_rs::{Evidence, FinishStatus, Handoff, RanCommand, ResultEnvelope};
    let path = ledger_path(xencode_dir);
    // Evidence is only as good as the ledger it comes from: a ledger that
    // cannot be read, or whose chain shows an edit, proves nothing, and the
    // envelope says so instead of reporting the rows as runs.
    let unusable = |reason: String| ResultEnvelope {
        status: FinishStatus::Blocked,
        agent: format!("session {session}"),
        task: task.to_string(),
        evidence: Evidence {
            changed_files: changed_files.clone(),
            commands: Vec::new(),
            artifact_refs: Vec::new(),
        },
        claims: Vec::new(),
        handoff: Handoff::Blocked { reason },
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return unusable(format!("cannot read {}: {e}", path.display())),
    };
    let chain = xencode_core_rs::chain::verify_chain(&text);
    if let Some(problem) = chain.problems.first() {
        return unusable(format!(
            "{LEDGER_FILE} was changed after it was written — {}",
            xencode_core_rs::chain::describe(problem, "xencode")
        ));
    }
    let mut commands = Vec::new();
    let mut artifact_refs = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let Ok(row) = serde_json::from_str::<LedgerEntry>(raw) else {
            continue;
        };
        if row.session.as_deref() != Some(session) {
            continue;
        }
        let digest = serde_json::from_str::<serde_json::Value>(raw)
            .ok()
            .and_then(|v| v.get("digest").and_then(|d| d.as_str()).map(str::to_string));
        let evidence_ref = match digest {
            Some(d) => format!(
                "{LEDGER_FILE} line {}, digest {}",
                index + 1,
                d.chars().take(16).collect::<String>()
            ),
            // A row from before the ledger was chained proves nothing about
            // itself, and the reference says so.
            None => format!(
                "{LEDGER_FILE} line {}, written before the ledger was chained, not tamper-evident",
                index + 1
            ),
        };
        let command = if row.note.trim().is_empty() {
            format!("{} run", row.run_class.as_str())
        } else {
            format!("{} run ({})", row.run_class.as_str(), row.note.trim())
        };
        if !row.log_ref.is_empty() {
            artifact_refs.push(row.log_ref.clone());
        }
        commands.push(RanCommand {
            command,
            exit_code: row.exit_code,
            evidence_ref,
        });
    }
    let (status, handoff) = if commands.is_empty() {
        (
            FinishStatus::Blocked,
            Handoff::Blocked {
                reason: "no check ran in this session, so nothing was proven".to_string(),
            },
        )
    } else if commands.iter().any(|c| !c.passed()) {
        (
            FinishStatus::Failed,
            Handoff::Blocked {
                reason: "a check this session ran exited non-zero".to_string(),
            },
        )
    } else {
        (
            FinishStatus::Completed,
            Handoff::NeedsReview {
                reason: "every check passed; a person decides whether it lands".to_string(),
            },
        )
    };
    ResultEnvelope {
        status,
        agent: format!("session {session}"),
        task: task.to_string(),
        evidence: Evidence {
            changed_files,
            commands,
            artifact_refs,
        },
        claims: Vec::new(),
        handoff,
    }
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

    /// AE-1: the envelope for a session lists exactly that session's rows, each
    /// pointing at the line and digest it came from; no rows is Blocked.
    #[test]
    fn a_sessions_envelope_points_every_exit_code_at_its_ledger_row() {
        let xencode = temp_xencode("envelope");
        let mut other = entry(Some("s2"));
        other.exit_code = 3;
        append_ledger(&xencode, &other).unwrap();
        append_ledger(&xencode, &entry(Some("s1"))).unwrap();
        let envelope = envelope_for_session(&xencode, "s1", "the task", vec!["a.rs".into()]);
        assert_eq!(envelope.status, xencode_core_rs::FinishStatus::Completed);
        assert_eq!(envelope.evidence.commands.len(), 1, "only s1's rows");
        let reference = &envelope.evidence.commands[0].evidence_ref;
        assert!(
            reference.starts_with("ledger.jsonl line 2, digest "),
            "{reference}"
        );
        let line2 = std::fs::read_to_string(ledger_path(&xencode))
            .unwrap()
            .lines()
            .nth(1)
            .unwrap()
            .to_string();
        let digest = reference.rsplit(' ').next().unwrap();
        assert!(line2.contains(digest), "the reference resolves to the row");

        let failed = envelope_for_session(&xencode, "s2", "t", Vec::new());
        assert_eq!(failed.status, xencode_core_rs::FinishStatus::Failed);
        let nothing = envelope_for_session(&xencode, "never-ran", "t", Vec::new());
        assert_eq!(nothing.status, xencode_core_rs::FinishStatus::Blocked);
        assert!(nothing.for_reviewer().render().contains("status: blocked"));

        // An edited ledger proves nothing: turning s2's failure into a pass
        // breaks the chain, and the envelope reports blocked, not completed.
        let path = ledger_path(&xencode);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(
            &path,
            text.replacen("\"exit_code\":3", "\"exit_code\":0", 1),
        )
        .unwrap();
        let forged = envelope_for_session(&xencode, "s2", "t", Vec::new());
        assert_eq!(forged.status, xencode_core_rs::FinishStatus::Blocked);
        assert!(forged.evidence.commands.is_empty());
        match &forged.handoff {
            xencode_core_rs::Handoff::Blocked { reason } => {
                assert!(reason.contains("changed after it was written"), "{reason}")
            }
            other => panic!("{other:?}"),
        }

        // A row whose digest is not the hash it claims (here not even hex) is
        // refused by the chain check before it is cited, and nothing panics.
        std::fs::write(
            &path,
            "{\"ts_unix_ms\":1,\"session\":\"s3\",\"run_class\":\"test\",\"exit_code\":0,\"subjects\":[],\"log_ref\":\"\",\"note\":\"\",\"digest\":\"ééééééééééééééééé\"}
",
        )
        .unwrap();
        let _ = envelope_for_session(&xencode, "s3", "t", Vec::new());
        let _ = std::fs::remove_dir_all(xencode.parent().unwrap());
    }

    /// EVd-7: each row is chained to the one before it, so an edit to a row in
    /// the middle of the file is caught on that line, and the rows still read
    /// back as rows.
    #[test]
    fn the_ledger_is_chained_and_an_edit_to_one_row_is_caught() {
        let xencode = temp_xencode("chain");
        for code in [0, 1, 0] {
            let mut row = entry(Some("s1"));
            row.exit_code = code;
            append_ledger(&xencode, &row).unwrap();
        }
        let report = verify_ledger(&xencode).unwrap();
        assert_eq!(report.records, 3);
        assert!(report.fully_chained(), "{report:?}");
        assert_eq!(
            read_ledger(&xencode).len(),
            3,
            "chain fields do not hide rows"
        );

        // Turn the failing run in the middle into a passing one.
        let path = ledger_path(&xencode);
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<String> = text
            .lines()
            .enumerate()
            .map(|(i, l)| {
                if i == 1 {
                    l.replace("\"exit_code\":1", "\"exit_code\":0")
                } else {
                    l.to_string()
                }
            })
            .collect();
        assert_ne!(
            lines.join(
                "
"
            ) + "
",
            text,
            "the edit happened"
        );
        std::fs::write(
            &path,
            lines.join(
                "
",
            ) + "
",
        )
        .unwrap();
        let report = verify_ledger(&xencode).unwrap();
        assert!(!report.intact());
        assert_eq!(report.problems[0].line, 2, "{report:?}");
        assert_eq!(
            report.problems[0].problem,
            xencode_core_rs::chain::ChainProblem::ContentChanged
        );
        let _ = std::fs::remove_dir_all(xencode.parent().unwrap());
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
