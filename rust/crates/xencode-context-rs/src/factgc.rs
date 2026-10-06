//! The queue of durable facts this repository has contradicted (`QK-6`).
//!
//! [`drop_stale_facts`] is silent and writes nothing back: a fact the code has
//! contradicted stops reaching every future turn, and `state.md` keeps the line
//! forever. That is right for a prompt and wrong for the person who promoted it,
//! who can still open the file and read what the model was told to stop believing.
//! [`audit_durable_facts`] answers the question `QK-4` shipped — what is being
//! excluded right now. This module answers the one a cleanup needs instead: for
//! how long, and what would it take to make that line go away.
//!
//! # Invalidate, do not delete
//!
//! Contradicting a fact never edits `state.md`. What gets written here is a
//! record that the code disagrees with it, stamped by the first turn that
//! noticed, and the line stays where a person can read it. Only twelve months of
//! unbroken contradiction make a line *retirable*, and even then removal is a
//! deliberate `--apply`, never a side effect of someone's prompt being assembled.
//!
//! `AGENTS.md` is not in scope and will not be: those are a human's instructions,
//! and the product writes there at one step only — `/lesson approve` (`EV-7`).
//!
//! # The trap this is built around
//!
//! The clock is about *continuous* contradiction, so an entry leaves the queue
//! the moment the check stops contradicting it — including on a turn where the
//! check could not run at all (no git, an unresolvable revision). That is the
//! conservative direction. A judgement this program cannot re-make today is not
//! a judgement it gets to act on twelve months from now, and a fact that has to
//! start the clock again after a gap costs nothing but patience.

use std::path::{Path, PathBuf};

use crate::compact::{drop_stale_facts, state_root, DroppedFact, FactProblem};

/// Where the queue lives, next to the file it is a record about.
pub const TOMBSTONE_FILE: &str = "facts.tombstones.jsonl";

/// How long a fact must be contradicted without a gap before a sweep may retire
/// it. A month is counted as 30 days: nobody schedules against this queue, and a
/// line that becomes retirable five days early is not a decision anyone loses.
pub const RETIRE_AFTER_MONTHS: u64 = 12;

const MS_PER_DAY: u64 = 24 * 60 * 60 * 1000;
const MS_PER_MONTH: u64 = 30 * MS_PER_DAY;

/// This instant, in the unit the queue stores its clocks in.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_millis() as u64)
        .unwrap_or_default()
}

/// One fact the repository contradicts, and since when.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FactTombstone {
    /// The fact as `state.md` holds it, provenance markers and all — the markers
    /// are what the next check is judged against, so they are what a year-old
    /// record has to quote. Scrubbed of anything that looks like a credential on
    /// the way in, the way every durable free-text field in this crate is.
    pub fact: String,
    /// Why the code contradicts it. Updated in place when the reason changes: the
    /// fact has been wrong the whole time either way, so the clock does not move.
    pub problem: FactProblem,
    /// The first turn that noticed. What the twelve-month clock runs on.
    pub first_seen_ms: u64,
    /// Set when a sweep removed the line from `state.md`. The entry stays behind:
    /// someone asking what a past cleanup took deserves a list, and a fact
    /// re-promoted later starts a new clock rather than inheriting this one.
    pub retired_ms: Option<u64>,
}

impl FactTombstone {
    /// Whole months this fact has been contradicted for.
    pub fn months(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.first_seen_ms) / MS_PER_MONTH
    }

    /// Whether the queue has held it long enough to be retirable. Age alone is
    /// not enough — see [`collect_gc`] for why the queue being populated at all
    /// is what carries the second half of that condition.
    pub fn is_aged(&self, now_ms: u64) -> bool {
        self.retired_ms.is_none()
            && now_ms.saturating_sub(self.first_seen_ms) >= RETIRE_AFTER_MONTHS * MS_PER_MONTH
    }
}

/// The queue's path for a project.
pub fn tombstone_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join(TOMBSTONE_FILE)
}

/// Whether this project has anything recorded. The turn path asks this before it
/// does any work, so a repository whose facts all hold costs one metadata call
/// per turn instead of a read and a parse.
pub fn queue_exists(xencode_dir: &Path) -> bool {
    tombstone_path(xencode_dir).is_file()
}

/// Every entry, oldest first. A queue that has never been written reads as
/// empty, and a line an interrupted write left half-written is dropped rather
/// than failing the read — everything before it is still a real clock.
pub fn read_tombstones(xencode_dir: &Path) -> Vec<FactTombstone> {
    let mut rows = xencode_core_rs::read_jsonl_tolerant(&tombstone_path(xencode_dir)).rows;
    sort_entries(&mut rows);
    rows
}

fn sort_entries(entries: &mut [FactTombstone]) {
    entries.sort_by(|a, b| {
        a.first_seen_ms
            .cmp(&b.first_seen_ms)
            .then_with(|| a.fact.cmp(&b.fact))
    });
}

/// The text a queue entry holds for a fact line. Redaction runs *before* the
/// comparison as well as before the write, so a line whose marker was scrubbed
/// on disk still matches the record made from it, and the unscrubbed bytes never
/// land in the queue or in a report.
pub(crate) fn stored_form(line: &str) -> String {
    crate::trace::redact_secrets(line)
}

/// The fact a line of `state.md` contributes, in the shape the staleness pass
/// works on it in: the parser takes a bullet's body with any leading `-` gone,
/// and that body is what a queue entry quotes.
pub(crate) fn body_of(line: &str) -> String {
    let trimmed = line.trim();
    trimmed.trim_start_matches('-').trim().to_string()
}

/// What reconciling the queue with one turn's check changed.
#[derive(Debug, Default)]
pub struct Recording {
    /// Entries added, dropped, or re-reasoned.
    pub changed: usize,
    /// The queue as it now stands, oldest first.
    pub queue: Vec<FactTombstone>,
}

/// Reconcile the queue with what the code contradicts right now.
///
/// Writes only when something moved. This runs on the way into a turn, and a
/// file that rewrites itself to say exactly what it already said is disk wear and
/// nothing else.
pub fn record_stale_facts(
    xencode_dir: &Path,
    dropped: &[DroppedFact],
    now_ms: u64,
) -> std::io::Result<Recording> {
    let contradicted: Vec<(String, FactProblem)> = dropped
        .iter()
        .map(|fact| (stored_form(&fact.line), fact.problem))
        .collect();
    let mut queue = read_tombstones(xencode_dir);
    let mut changed = 0usize;

    // A fact the code no longer contradicts loses its clock. So does one this
    // turn could not check, which is why a gap restarts the twelve months rather
    // than counting through it.
    let before = queue.len();
    queue.retain(|entry| {
        entry.retired_ms.is_some() || contradicted.iter().any(|(line, _)| *line == entry.fact)
    });
    changed += before - queue.len();

    for (line, problem) in contradicted {
        match queue.iter_mut().find(|entry| entry.fact == line) {
            Some(entry) => {
                let mut moved = false;
                if entry.problem != problem {
                    entry.problem = problem;
                    moved = true;
                }
                // Contradicted again after a retirement is a different fact going
                // wrong — the line was removed, then re-promoted — so it gets a
                // new clock and loses the stamp saying it was already swept.
                if entry.retired_ms.take().is_some() {
                    entry.first_seen_ms = now_ms;
                    moved = true;
                }
                changed += usize::from(moved);
            }
            None => {
                queue.push(FactTombstone {
                    fact: line,
                    problem,
                    first_seen_ms: now_ms,
                    retired_ms: None,
                });
                changed += 1;
            }
        }
    }

    if changed > 0 {
        sort_entries(&mut queue);
        write_tombstones(xencode_dir, &queue)?;
    }
    Ok(Recording { changed, queue })
}

fn write_tombstones(xencode_dir: &Path, queue: &[FactTombstone]) -> std::io::Result<()> {
    let mut text = String::new();
    for entry in queue {
        let row = serde_json::to_string(entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        text.push_str(&row);
        text.push('\n');
    }
    xencode_core_rs::write_atomic(&tombstone_path(xencode_dir), text.as_bytes())
}

/// What one call to `xencode memory gc` found.
#[derive(Debug, Default)]
pub struct GcReport {
    /// Contradicted and not yet retired, oldest first — including whatever this
    /// call just stamped.
    pub pending: Vec<FactTombstone>,
    /// The pending entries twelve months old: what `--apply` would remove.
    pub aged: Vec<FactTombstone>,
    /// The lines this call took out of `state.md`. Empty unless `--apply`.
    pub removed: Vec<String>,
    /// What past sweeps retired, oldest retirement first.
    pub retired: Vec<FactTombstone>,
    /// Facts that stayed in the prompt because they could not be checked. Not
    /// staleness, so no sweep counts them toward anything.
    pub unverifiable: usize,
    /// Facts the code places in a different file from the one they cite. These
    /// stay, they are reported by `QM-4`, and a sweep never touches them.
    pub disagreeing: usize,
}

/// The queue, reconciled against this repository as it stands, and the retirement
/// that reconciliation makes possible.
///
/// Every entry in the returned queue was contradicted by the check this same
/// call ran moments ago — that is what [`record_stale_facts`] just did — so age
/// is the only condition `is_aged` has to test. A twelve-month-old record cannot
/// retire a line the current code is happy with, because that line is no longer
/// in the queue.
pub fn collect_gc(xencode_dir: &Path, now_ms: u64, apply: bool) -> std::io::Result<GcReport> {
    let root = state_root(xencode_dir);
    let stored = std::fs::read_to_string(xencode_dir.join("state.md")).unwrap_or_default();
    let check = drop_stale_facts(&stored, &root);
    let recorded = record_stale_facts(xencode_dir, &check.dropped, now_ms)?;

    let mut report = GcReport {
        unverifiable: check.unverifiable,
        disagreeing: check.disagreeing.len(),
        ..Default::default()
    };
    for entry in recorded.queue {
        if entry.retired_ms.is_some() {
            report.retired.push(entry);
        } else {
            if entry.is_aged(now_ms) {
                report.aged.push(entry.clone());
            }
            report.pending.push(entry);
        }
    }
    if apply && !report.aged.is_empty() {
        report.removed = retire_lines(xencode_dir, &stored, &report.aged, now_ms)?;
    }
    Ok(report)
}

/// Take the aged lines out of `state.md` and stamp their queue entries retired.
///
/// The file is filtered line by line rather than parsed and re-rendered, so
/// everything this sweep was not asked to touch — the `## working-on` text, the
/// order of the facts that stay, a section a person typed in by hand — comes out
/// with the same bytes it went in with.
fn retire_lines(
    xencode_dir: &Path,
    stored: &str,
    aged: &[FactTombstone],
    now_ms: u64,
) -> std::io::Result<Vec<String>> {
    let retiring: Vec<String> = aged.iter().map(|entry| entry.fact.clone()).collect();
    let kept: String = stored
        .split_inclusive('\n')
        .filter(|line| !retiring.contains(&stored_form(&body_of(line))))
        .collect();
    xencode_core_rs::write_atomic(&xencode_dir.join("state.md"), kept.as_bytes())?;

    let mut queue = read_tombstones(xencode_dir);
    for entry in &mut queue {
        if entry.retired_ms.is_none() && retiring.contains(&entry.fact) {
            entry.retired_ms = Some(now_ms);
        }
    }
    write_tombstones(xencode_dir, &queue)?;
    Ok(retiring)
}

/// The words an interface puts on a fact's age. Kept beside the constant that
/// decides it, because a month here means thirty days and a sentence written at
/// the call site would have no way to know that.
pub fn age_words(months: u64) -> String {
    match months {
        0 => "contradicted for under a month".to_string(),
        1 => "contradicted for 1 month".to_string(),
        months => format!("contradicted for {months} months"),
    }
}
