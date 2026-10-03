//! The run ledger (`QTR-5`) — `.xencode/cache/runs.jsonl`.
//!
//! One row per agent run: which run it was, which model answered it, which
//! decisions a person made while it went, and where its evidence lives. The
//! other records already answer adjacent questions and none of them answers
//! this one. `cache/turns.jsonl` says what a *turn* did — its rounds, its tool
//! calls, the tail of each result — but it has no identity of its own, so two
//! turns of the same run cannot be told apart afterwards, and nothing in it
//! says whether a human was asked anything. `ledger.jsonl` (`EVd-1`) says that a
//! check ran and exited 0 or 1, keyed on a session. `cache/sessions/<run id>.jsonl`
//! (`QA-1`) is a full recording, and it exists only when recording was asked
//! for. What none of them keep is the one thing an accountability question needs:
//! *this run*, with *these* decisions in it, reachable by one id.
//!
//! # It joins, it does not copy
//!
//! A row stores what only the run knows — its id, its model, and the approvals a
//! person gave or withheld — and points at everything else. Its verification
//! evidence is read out of [`crate::ledger`] by session when someone asks
//! ([`run_evidence`]), not duplicated into a second exit-code store; its
//! recording is named by path, not embedded. That is deliberate: two copies of
//! the same fact disagree the first time one of them is corrected, and a ledger
//! nobody can check is decoration.
//!
//! # The trap this is built around
//!
//! The same one [`crate::ledger`] documents: a ledger is read by everything and
//! written once, so anything secret has to be handled on the way in. The free
//! text field is scrubbed with [`crate::trace::redact_secrets`] before it is
//! stored, and there is nothing else free-form to scrub — a decision row is a
//! tool name, a class and a three-way answer. The prompt itself is not here at
//! all; [`crate::trace::prompt_digest`] already reduced it to a digest, and a
//! digest cannot be read back.

use crate::ledger::LedgerEntry;
use crate::metrics::MetricSource;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Where the run ledger lives, beside the turn trace and the metrics rows.
pub const RUNS_FILE: &str = "runs.jsonl";

/// How many rows [`recent_runs`] keeps. The ledger only grows, and the newest
/// twenty is what a person asks for; `run_by_id` reads past the window.
pub const RUNS_WINDOW: usize = 20;

/// What a person answered when the agent asked to run something.
///
/// Three answers because that is exactly what the approval overlay offers: allow
/// this once, allow this kind for the rest of the session, or refuse. A tool that
/// never got asked — allowed by policy, or refused outright as out of bounds —
/// is not an approval and is not recorded here, because the honest answer to
/// "what did the human decide" cannot include a decision nobody made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    /// Allowed this call, once.
    Allowed,
    /// Allowed this class of call for the rest of the session.
    AlwaysAllowed,
    /// Refused the call. Includes a call nobody answered because nothing was
    /// listening: a run with no interface attached is refused, and that refusal
    /// is somebody's decision as much as a typed `n` is.
    Denied,
}

impl ApprovalDecision {
    /// The word stored in the row and shown to a reader.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::AlwaysAllowed => "always allowed",
            Self::Denied => "denied",
        }
    }

    /// `true` for either kind of yes.
    pub fn granted(self) -> bool {
        !matches!(self, Self::Denied)
    }
}

/// One question a person answered, and how they answered it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalRow {
    /// The tool the model asked to use.
    pub tool: String,
    /// What kind of call it was — `read-only`, `file change`, `shell command`,
    /// `external tool`. Stored as the words the overlay itself shows, so the
    /// ledger and the screen cannot describe the same call differently.
    pub class: String,
    pub decision: ApprovalDecision,
}

/// One row of `.xencode/cache/runs.jsonl`: one run of the agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// The run's own id, in the same shape
    /// [`crate::session::new_run_id`] gives a recording, so one id names both
    /// when a recording exists and can be typed into `xencode replay`.
    pub run_id: String,
    /// UTC epoch millis when the run ended.
    pub ts_unix_ms: u64,
    /// Wall-clock milliseconds the loop ran for.
    pub duration_ms: u64,
    /// Trips through the loop, including a last one that failed.
    pub rounds: u32,
    /// The conversation this run belonged to — the `EVd-2` key, and the join
    /// column into the verification ledger. `None` when no session was open.
    #[serde(default)]
    pub session: Option<String>,
    /// The model as it was selected, route prefix and all.
    #[serde(default)]
    pub model: Option<String>,
    /// Which client the id resolved to (`ollama`, `llamacpp`, `remote`, …).
    #[serde(default)]
    pub provider: Option<String>,
    /// Whether this run's prompt left the machine.
    #[serde(default)]
    pub source: Option<MetricSource>,
    /// Every question a person answered while this run went, in the order they
    /// were answered. Empty means the run never asked — which is a fact worth
    /// reading, not a gap.
    #[serde(default)]
    pub approvals: Vec<ApprovalRow>,
    /// Where this run's recording is, when one was made. A path, not its bytes.
    #[serde(default)]
    pub recording: Option<String>,
    /// One human line, redacted on the way in. May be empty.
    #[serde(default)]
    pub note: String,
}

/// The ledger file for a project.
pub fn runs_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join("cache").join(RUNS_FILE)
}

/// Append a row, creating the file and its directory; never rewrites.
///
/// The note is scrubbed here rather than at read time, for the reason
/// [`crate::ledger`] gives: every reader would have to remember to do it, and
/// the first one that forgets leaks.
pub fn append_run(xencode_dir: &Path, record: &RunRecord) -> std::io::Result<()> {
    let path = runs_path(xencode_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let stored = RunRecord {
        note: crate::trace::redact_secrets(&record.note),
        ..record.clone()
    };
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let line = serde_json::to_string(&stored)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    writeln!(file, "{line}")
}

/// Every row, oldest first. A line that does not parse is dropped, as it is in
/// the ledger and the metrics file: one broken row must not hide the ones after
/// it, and must not hide the ones before it.
pub fn read_runs(xencode_dir: &Path) -> Vec<RunRecord> {
    xencode_core_rs::read_jsonl_tolerant::<RunRecord>(&runs_path(xencode_dir)).rows
}

/// The newest `limit` runs, oldest first — the window `xencode runs list`
/// prints.
pub fn recent_runs(xencode_dir: &Path, limit: usize) -> Vec<RunRecord> {
    if limit == 0 {
        return Vec::new();
    }
    let rows = read_runs(xencode_dir);
    let skip = rows.len().saturating_sub(limit);
    rows.into_iter().skip(skip).collect()
}

/// One run by id: the whole id, or enough of its start to name one run and no
/// other. Same rule `xencode replay` uses, so an id that works there works here.
pub fn run_by_id(xencode_dir: &Path, given: &str) -> Option<RunRecord> {
    let rows = read_runs(xencode_dir);
    if let Some(exact) = rows.iter().rev().find(|row| row.run_id == given) {
        return Some(exact.clone());
    }
    let matches: Vec<&RunRecord> = rows
        .iter()
        .filter(|row| row.run_id.starts_with(given))
        .collect();
    if matches.len() == 1 {
        return Some(matches[0].clone());
    }
    None
}

/// A run and the verification rows its session left behind — the join this
/// ledger exists to make and deliberately does not store.
#[derive(Debug, Clone)]
pub struct RunEvidence {
    pub run: RunRecord,
    /// The session's `EVd-1` rows, oldest first. Empty when the run's session
    /// left no checks, which is the honest reading of "nothing verified this".
    pub checks: Vec<LedgerEntry>,
}

impl RunEvidence {
    /// `true` when a check ran and exited zero. A row with a non-zero exit is
    /// evidence of a failure, not of nothing.
    pub fn verified(&self) -> bool {
        self.checks.iter().any(|row| row.passed())
    }
}

/// Read a run together with its session's ledger rows. A run with no session
/// cannot be joined to anything and reports no checks.
pub fn run_evidence(xencode_dir: &Path, run: &RunRecord) -> RunEvidence {
    let checks = match run.session.as_deref() {
        Some(session) => crate::ledger::ledger_for_session(xencode_dir, session),
        None => Vec::new(),
    };
    RunEvidence {
        run: run.clone(),
        checks,
    }
}

/// How many `Assisted-by` lines a run produces, and what they say.
///
/// The trailer names the run rather than describing it, because a trailer that
/// tried to summarise a run would be a second, lazier copy of the ledger row —
/// and the two would disagree as soon as anybody corrected one. `git
/// interpret-trailers` reads the line as a trailer because a trailer is `Token:
/// value`; the value here is a fixed phrase, built by [`accountability_trailer`].
pub fn accountability_trailer(run: &RunRecord, version: &str) -> String {
    let (granted, denied) = tally_of(run);
    let mut parts = vec![format!(
        "Assisted-by: xencode/{version} (run {}, {} asked, {} allowed, {} denied)",
        run.run_id,
        run.approvals.len(),
        granted,
        denied
    )];
    if let Some(model) = run.model.as_deref() {
        parts.push(format!("Xencode-Model: {model}"));
    }
    if run.recording.is_some() {
        parts.push(format!("Xencode-Replay: xencode replay {}", run.run_id));
    }
    parts.join("\n")
}

fn tally_of(run: &RunRecord) -> (usize, usize) {
    let granted = run
        .approvals
        .iter()
        .filter(|row| row.decision.granted())
        .count();
    (granted, run.approvals.len() - granted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_xencode(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xe-runs-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("cache")).unwrap();
        dir
    }

    fn approval(tool: &str, decision: ApprovalDecision) -> ApprovalRow {
        ApprovalRow {
            tool: tool.to_string(),
            class: "shell command".to_string(),
            decision,
        }
    }

    fn record(run_id: &str) -> RunRecord {
        RunRecord {
            run_id: run_id.to_string(),
            ts_unix_ms: 1_700_000_000_000,
            duration_ms: 4_100,
            rounds: 3,
            session: Some("s1".to_string()),
            model: Some("llamacpp:qwen/qwen3-8b".to_string()),
            provider: Some("llamacpp".to_string()),
            source: Some(MetricSource::Local),
            approvals: vec![
                approval("run_command", ApprovalDecision::Allowed),
                approval("write_file", ApprovalDecision::Denied),
            ],
            recording: None,
            note: String::new(),
        }
    }

    #[test]
    fn a_run_appends_and_reads_back_with_its_decisions() {
        let xencode = temp_xencode("round");
        append_run(&xencode, &record("1700000000-aaaa1111")).unwrap();
        let rows = read_runs(&xencode);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].run_id, "1700000000-aaaa1111");
        assert_eq!(rows[0].approvals.len(), 2);
        assert_eq!(rows[0].approvals[1].decision, ApprovalDecision::Denied);
        assert_eq!(rows[0].model.as_deref(), Some("llamacpp:qwen/qwen3-8b"));
    }

    #[test]
    fn a_run_with_nothing_asked_says_so_rather_than_nothing() {
        let xencode = temp_xencode("silent");
        let mut quiet = record("1700000000-bbbb2222");
        quiet.approvals.clear();
        append_run(&xencode, &quiet).unwrap();
        let rows = read_runs(&xencode);
        assert!(rows[0].approvals.is_empty());
        let trailer = accountability_trailer(&rows[0], "0.1.0");
        assert!(
            trailer.contains("0 asked, 0 allowed, 0 denied"),
            "{trailer}"
        );
    }

    #[test]
    fn a_newest_window_keeps_the_order_and_drops_the_older_ones() {
        let xencode = temp_xencode("window");
        for i in 0..25 {
            append_run(&xencode, &record(&format!("1700000000-{i:08x}"))).unwrap();
        }
        let recent = recent_runs(&xencode, 5);
        assert_eq!(recent.len(), 5);
        assert_eq!(recent[0].run_id, "1700000000-00000014");
        assert_eq!(recent[4].run_id, "1700000000-00000018");
        assert_eq!(read_runs(&xencode).len(), 25, "the window is not a cap");
    }

    #[test]
    fn an_id_is_matched_exactly_or_by_one_unambiguous_prefix() {
        let xencode = temp_xencode("ids");
        append_run(&xencode, &record("1700000000-aaaa1111")).unwrap();
        append_run(&xencode, &record("1700000000-bbbb2222")).unwrap();
        assert!(run_by_id(&xencode, "1700000000-aaaa1111").is_some());
        assert!(run_by_id(&xencode, "1700000000-aaaa").is_some());
        assert!(
            run_by_id(&xencode, "1700000000").is_none(),
            "a prefix naming two runs must not pick one"
        );
        assert!(run_by_id(&xencode, "nope").is_none());
    }

    #[test]
    fn a_broken_row_hides_nothing_but_itself() {
        let xencode = temp_xencode("corrupt");
        append_run(&xencode, &record("1700000000-cccc3333")).unwrap();
        use std::io::Write;
        writeln!(
            std::fs::OpenOptions::new()
                .append(true)
                .open(runs_path(&xencode))
                .unwrap(),
            "not json at all"
        )
        .unwrap();
        append_run(&xencode, &record("1700000000-dddd4444")).unwrap();
        let rows = read_runs(&xencode);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].run_id, "1700000000-dddd4444");
    }

    #[test]
    fn a_secret_in_a_note_is_scrubbed_before_it_reaches_disk() {
        let xencode = temp_xencode("scrub");
        let mut row = record("1700000000-eeee5555");
        row.note = "stopped after OPENAI_API_KEY=\"sk-live-abcdef1234567890\" leaked".to_string();
        append_run(&xencode, &row).unwrap();
        let raw = std::fs::read_to_string(runs_path(&xencode)).unwrap();
        assert!(
            !raw.contains("sk-live-abcdef1234567890"),
            "the secret reached disk"
        );
        assert!(raw.contains("[redacted]"), "the scrub must be visible");
        // The scrub did the work, not the fixture.
        assert!(row.note.contains("sk-live-abcdef1234567890"));
    }

    #[test]
    fn a_run_is_joined_to_its_sessions_checks_rather_than_copying_them() {
        let xencode = temp_xencode("join");
        append_run(&xencode, &record("1700000000-ffff6666")).unwrap();
        // The session's evidence lives in the verification ledger, one row per
        // check, and this run only has to name the session to reach it.
        let entry = LedgerEntry {
            ts_unix_ms: 1_700_000_000_500,
            session: Some("s1".to_string()),
            run_class: crate::ledger::RunClass::Test,
            exit_code: 0,
            subjects: vec![crate::ledger::digest_hex("cargo test --workspace")],
            log_ref: "artifacts/s1/test.log".to_string(),
            note: String::new(),
        };
        crate::ledger::append_ledger(&xencode, &entry).unwrap();

        let run = run_by_id(&xencode, "1700000000-ffff6666").unwrap();
        let evidence = run_evidence(&xencode, &run);
        assert_eq!(evidence.checks.len(), 1);
        assert!(evidence.verified());
        assert_eq!(evidence.checks[0].log_ref, "artifacts/s1/test.log");

        // A run from another session sees none of it.
        let other = RunRecord {
            session: Some("s2".to_string()),
            ..record("1700000000-ffff7777")
        };
        assert!(run_evidence(&xencode, &other).checks.is_empty());
    }

    #[test]
    fn no_session_means_no_checks_and_no_claim_about_evidence() {
        let xencode = temp_xencode("nosess");
        let orphan = RunRecord {
            session: None,
            ..record("1700000000-ffff8888")
        };
        append_run(&xencode, &orphan).unwrap();
        let evidence = run_evidence(&xencode, &orphan);
        assert!(evidence.checks.is_empty());
        assert!(!evidence.verified());
    }

    #[test]
    fn the_trailer_names_the_run_and_its_decisions_and_nothing_else() {
        let run = record("1700000000-aaaa1111");
        let trailer = accountability_trailer(&run, "0.1.0");
        let lines: Vec<&str> = trailer.lines().collect();
        assert_eq!(
            lines[0],
            "Assisted-by: xencode/0.1.0 (run 1700000000-aaaa1111, 2 asked, 1 allowed, 1 denied)"
        );
        assert_eq!(lines[1], "Xencode-Model: llamacpp:qwen/qwen3-8b");
        assert_eq!(lines.len(), 2, "no recording, so no replay line");
        // Every line is a trailer: a token, a colon, a space.
        for line in &lines {
            let (key, value) = line.split_once(": ").expect("a trailer is `Token: value`");
            assert!(
                key.chars()
                    .all(|c| c.is_alphanumeric() || c == '-' || c == '_'),
                "{key} is not a trailer token"
            );
            assert!(!value.is_empty());
        }
    }

    #[test]
    fn a_recorded_run_offers_the_replay_command_it_was_named_by() {
        let run = RunRecord {
            recording: Some(".xencode/cache/sessions/1700000000-aaaa1111.jsonl".to_string()),
            ..record("1700000000-aaaa1111")
        };
        let trailer = accountability_trailer(&run, "0.1.0");
        assert!(
            trailer.contains("Xencode-Replay: xencode replay 1700000000-aaaa1111"),
            "{trailer}"
        );
    }

    /// The whole point of the id is that a reader can go and look, so the run
    /// the trailer names has to be the run the ledger holds — model, decisions
    /// and all.
    #[test]
    fn a_trailer_points_at_a_run_that_can_be_read_back() {
        let xencode = temp_xencode("traceable");
        append_run(&xencode, &record("1700000000-9999aaaa")).unwrap();
        let stored = run_by_id(&xencode, "1700000000-9999aaaa").unwrap();
        let trailer = accountability_trailer(&stored, "0.1.0");
        assert!(trailer.contains("run 1700000000-9999aaaa"), "{trailer}");
        assert!(trailer.contains("1 denied"), "{trailer}");
        let read_back = run_by_id(&xencode, &stored.run_id).unwrap();
        assert_eq!(read_back.approvals, stored.approvals);
    }
}
