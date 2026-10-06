//! How much re-checking a durable fact has actually survived (`QK-1`).
//!
//! [`drop_stale_facts`] answers a yes-or-no question every turn a project runs:
//! does this repository still agree with the line `state.md` holds? A fact that
//! has agreed for a year and one promoted five minutes ago arrive in the same
//! prompt looking exactly alike, and nothing in the project said which was which.
//! This module keeps the answers. One row per fact, one verdict per revision the
//! check was run at, and nothing else.
//!
//! # What the number is, and what it is not
//!
//! The evidence is *revisions*, not turns. A fact re-checked at the same commit
//! forty times was looked at once: the tree did not move and the check is
//! deterministic, so the forty-first answer was known by the first. Counting turns
//! would let a busy afternoon at one revision read like a fact that survived a
//! hundred changes. So the count is the number of distinct revisions this line was
//! checked against, and the interval is a Wilson score interval over how many of
//! them contradicted it.
//!
//! It is not, and never reads as, a probability that the fact is true. What the
//! check can answer is narrow — a cited file is still there, still matches the
//! revision it was written at, and still declares the name the line talks about.
//! A fact about *why* a decision was made passes that check forever and proves
//! nothing about its own reasoning. And the report prints a range and never a
//! single number, because a scalar on a line like this is exactly what `QK-1`
//! refuses to invent: `n = 1` reaching 0.21 is the honest sentence about one
//! observation, and a number formatted as `0.62` is a lie with two decimals.
//!
//! # Why no model's name appears here
//!
//! The verifier is `git grep` over the working tree, run by this binary. Naming a
//! model as the thing that verified a fact would misattribute a mechanical answer
//! to the model that happened to be driving the turn, and the plan's no-cross-model
//! transfer rule would then have to be defended against a judgement no model made.
//! The fingerprint a check leaves behind is therefore the revision it ran at, the
//! moment it ran, and the name of the search that answered — which is all any of
//! them can be, since what did the work is this program's own file-and-name re-check
//! rather than anything that could be identified by version.

use std::path::{Path, PathBuf};

use crate::compact::{state_root, FactProblem, StaleFacts};
use crate::perf::{wilson_interval, Z_AT_DEFAULT_ALPHA};
use crate::state::ContextState;

/// Where the ledger lives, beside `state.md` and the queue `QK-6` keeps.
pub const EVIDENCE_FILE: &str = "facts.evidence.jsonl";

/// How many revisions one fact keeps a verdict for. The oldest go first: the
/// interval is what the recent history supports, and a row that grew without a
/// bound is a file nobody reads.
pub const REVISIONS_KEPT: usize = 64;

/// What the check said about one fact at one revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckVerdict {
    /// The cited file was there and matched, and every name the line checked was
    /// still declared where it claimed.
    Survived,
    /// The code contradicted it. Which of the four ways is not stored twice: the
    /// reason rides in `QK-6`'s queue, and this row says only that a revision said
    /// no.
    Contradicted,
    /// The check could not be run here — git would not answer, or the revision the
    /// line names cannot be resolved. Recorded so a gap is visible, and never
    /// counted: not-an-answer is not an answer.
    Unchecked,
}

impl CheckVerdict {
    /// Whether this row is evidence at all.
    pub fn counts(self) -> bool {
        !matches!(self, CheckVerdict::Unchecked)
    }
}

/// One check of one fact, at one revision.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CheckRecord {
    /// Eight characters, the same truncation `QM-2` stamps into the marker.
    pub revision: String,
    pub verdict: CheckVerdict,
    pub at_ms: u64,
}

/// Everything the project has to say about one durable fact.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FactEvidence {
    /// The fact line, redacted, markers included — the same identity `QK-6` keys
    /// its queue on, so the two files cannot mean different things by the same row.
    pub fact: String,
    /// Oldest revision first, one entry per distinct revision the check ran at.
    pub checks: Vec<CheckRecord>,
}

impl FactEvidence {
    /// Revisions the check could answer at. The denominator.
    pub fn trials(&self) -> u64 {
        self.checks.iter().filter(|c| c.verdict.counts()).count() as u64
    }

    pub fn survived(&self) -> u64 {
        self.checks
            .iter()
            .filter(|c| c.verdict == CheckVerdict::Survived)
            .count() as u64
    }

    /// The Wilson score interval over `survived` of `trials`, or nothing when
    /// there is no answer to summarise.
    pub fn interval(&self) -> Option<(f64, f64)> {
        wilson_interval(self.survived(), self.trials(), Z_AT_DEFAULT_ALPHA)
    }

    /// The most recent check, whichever way it went.
    pub fn last(&self) -> Option<&CheckRecord> {
        self.checks.last()
    }
}

/// The ledger's path for a project.
pub fn evidence_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join(EVIDENCE_FILE)
}

/// Every row, in the order they were written. A file that has never been written
/// reads as empty, and a line an interrupted write left half-written is dropped
/// rather than failing the read.
pub fn read_evidence(xencode_dir: &Path) -> Vec<FactEvidence> {
    xencode_core_rs::read_jsonl_tolerant(&evidence_path(xencode_dir)).rows
}

/// Add this turn's answers to the ledger, and return what the ledger now says
/// about the facts in `state.md`.
///
/// Writes only when something moved, and only for a fact the check was able to
/// ask something about: an unmarked line was never tested, so a row for it would
/// be a row of nothing. A fact taken out of `state.md` loses its row with it —
/// the tally belongs to the line, and a person who deletes the line has already
/// made the decision the tally would have been arguing about.
pub fn record_evidence(
    xencode_dir: &Path,
    stored_state: &str,
    check: &StaleFacts,
    now_ms: u64,
) -> std::io::Result<Vec<FactEvidence>> {
    let marked: Vec<String> = marked_facts(stored_state);
    if marked.is_empty() {
        // No fact asked anything of the code, so nothing is recorded and git is
        // not asked at all — one metadata call is the whole cost. The one
        // exception is the ledger whose last fact just left the file: it is
        // emptied once, and then never touched again.
        if !evidence_path(xencode_dir).is_file() {
            return Ok(Vec::new());
        }
        if read_evidence(xencode_dir).is_empty() {
            return Ok(Vec::new());
        }
        write_evidence(xencode_dir, &[])?;
        return Ok(Vec::new());
    }
    let Some(revision) = checked_at(&state_root(xencode_dir)) else {
        // A repository with no commit yet has no revisions to have been checked
        // against. Saying so is truer than filing every answer under a blank key.
        return Ok(Vec::new());
    };
    let mut ledger = read_evidence(xencode_dir);
    let mut changed = false;

    for fact in &marked {
        let stored = crate::factgc::stored_form(fact);
        let verdict = verdict_for(&stored, check);
        let row = ledger.iter_mut().find(|row| row.fact == stored);
        match row {
            Some(row) => {
                if let Some(record) = row.checks.iter_mut().find(|c| c.revision == revision) {
                    // The moment stays the first one, and a turn at a revision the
                    // ledger already has writes nothing. Otherwise every turn in a
                    // busy afternoon rewrites the file to say what it said, and the
                    // clock it stamps describes how often the project was used rather
                    // than how much of the code the fact has been checked against.
                    if record.verdict != verdict {
                        record.verdict = verdict;
                        changed = true;
                    }
                } else {
                    row.checks.push(CheckRecord {
                        revision: revision.clone(),
                        verdict,
                        at_ms: now_ms,
                    });
                    if row.checks.len() > REVISIONS_KEPT {
                        row.checks.remove(0);
                    }
                    changed = true;
                }
            }
            None => {
                ledger.push(FactEvidence {
                    fact: stored,
                    checks: vec![CheckRecord {
                        revision: revision.clone(),
                        verdict,
                        at_ms: now_ms,
                    }],
                });
                changed = true;
            }
        }
    }

    let present: Vec<String> = marked
        .iter()
        .map(|fact| crate::factgc::stored_form(fact))
        .collect();
    let before = ledger.len();
    ledger.retain(|row| present.contains(&row.fact));
    changed |= ledger.len() != before;

    if changed {
        write_evidence(xencode_dir, &ledger)?;
    }
    Ok(ledger)
}

/// The fact lines `state.md` holds that carry something the check can answer on.
fn marked_facts(stored_state: &str) -> Vec<String> {
    let state = ContextState::from_markdown(stored_state);
    [&state.completed, &state.decisions, &state.unresolved]
        .into_iter()
        .flatten()
        .map(|line| crate::factgc::body_of(line))
        .filter(|line| {
            line.contains(crate::compact::SRC_OPEN) || line.contains(crate::compact::CHK_OPEN)
        })
        .collect()
}

/// How this turn's pass answered about one fact.
fn verdict_for(stored_fact: &str, check: &StaleFacts) -> CheckVerdict {
    if check
        .dropped
        .iter()
        .any(|fact| crate::factgc::stored_form(&fact.line) == stored_fact)
    {
        return CheckVerdict::Contradicted;
    }
    if check
        .passed
        .iter()
        .any(|line| crate::factgc::stored_form(line) == stored_fact)
    {
        return CheckVerdict::Survived;
    }
    CheckVerdict::Unchecked
}

/// The revision the check ran against, in the eight characters the markers use.
fn checked_at(root: &Path) -> Option<String> {
    let head = crate::gitinfo::git_stdout(root, &["rev-parse", "HEAD"])
        .ok()?
        .lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.chars().take(8).collect::<String>())?;
    (!head.is_empty()).then_some(head)
}

fn write_evidence(xencode_dir: &Path, ledger: &[FactEvidence]) -> std::io::Result<()> {
    let mut text = String::new();
    for row in ledger {
        let line = serde_json::to_string(row)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        text.push_str(&line);
        text.push('\n');
    }
    xencode_core_rs::write_atomic(&evidence_path(xencode_dir), text.as_bytes())
}

/// What `xencode memory evidence` prints, one row per fact the file holds.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceRow {
    /// The fact with its markers gone — what a person promoted.
    pub fact: String,
    pub survived: u64,
    pub trials: u64,
    pub unchecked: u64,
    /// The Wilson interval, when there is an answer to build one from.
    pub interval: Option<(f64, f64)>,
    /// The revision, moment and searching binary of the most recent check.
    pub last: Option<CheckRecord>,
    /// The reason a contradicted fact left the prompt, if `QK-6`'s queue has one.
    pub problem: Option<FactProblem>,
}

/// The ledger joined to the facts as `state.md` holds them now, weakest evidence
/// first: the row a person needs to read is the one that does not have enough
/// behind it to be trusted.
pub fn evidence_rows(xencode_dir: &Path) -> Vec<EvidenceRow> {
    let stored = std::fs::read_to_string(xencode_dir.join("state.md")).unwrap_or_default();
    let check = crate::compact::drop_stale_facts(&stored, &state_root(xencode_dir));
    let ledger = read_evidence(xencode_dir);
    let mut rows: Vec<EvidenceRow> = marked_facts(&stored)
        .into_iter()
        .map(|fact| {
            let stored_fact = crate::factgc::stored_form(&fact);
            let entry = ledger.iter().find(|row| row.fact == stored_fact);
            EvidenceRow {
                fact: crate::compact::fact_prose(&fact).to_string(),
                survived: entry.map(FactEvidence::survived).unwrap_or_default(),
                trials: entry.map(FactEvidence::trials).unwrap_or_default(),
                unchecked: entry
                    .map(|row| row.checks.len() as u64 - row.trials())
                    .unwrap_or_default(),
                interval: entry.and_then(FactEvidence::interval),
                last: entry.and_then(|row| row.last().cloned()),
                problem: check
                    .dropped
                    .iter()
                    .find(|dropped| crate::factgc::stored_form(&dropped.line) == stored_fact)
                    .map(|dropped| dropped.problem),
            }
        })
        .collect();
    rows.sort_by(|a, b| a.trials.cmp(&b.trials).then_with(|| a.fact.cmp(&b.fact)));
    rows
}

impl EvidenceRow {
    /// The sentence a report puts under a fact. Written here rather than at the
    /// call site because the wording has to keep the two things this number is not
    /// out of it: it is not a probability the fact is true, and a lone figure is
    /// not what a range is for.
    pub fn evidence_sentence(&self) -> String {
        match self.interval {
            Some((lower, upper)) => format!(
                "checked against {trials} revision{s}; {survived} of them found nothing to \
                 contradict it — the 95% interval over a future check agreeing runs {lower:.1}% \
                 to {upper:.1}%",
                trials = self.trials,
                s = if self.trials == 1 { "" } else { "s" },
                survived = self.survived,
                lower = lower * 100.0,
                upper = upper * 100.0,
            ),
            None if self.unchecked > 0 => format!(
                "{unchecked} check{s} could not be run here, so there is no interval to print",
                unchecked = self.unchecked,
                s = if self.unchecked == 1 { "" } else { "s" },
            ),
            None => "not yet checked against a revision — the project has not committed since \
                     this fact was promoted, or git would not answer here"
                .to_string(),
        }
    }

    /// Who verified it, in the only sense that is true. This is the `verified_by`
    /// the plan row asks for, and it names a search rather than a model: what
    /// answered about a durable fact was this binary looking for the cited file and
    /// the cited name in the working tree at one revision, at one moment. Which
    /// model was driving the turn has nothing to do with the answer, so naming one
    /// here would put a mechanical verdict in a model's mouth — and the no-cross-
    /// model-transfer rule `QK-1` carries would then be guarding a judgement no
    /// model made.
    pub fn verified_by(&self) -> String {
        let Some(record) = &self.last else {
            return "nothing has checked this fact yet".to_string();
        };
        let day = crate::rollup::local_day_key(record.at_ms);
        if day.is_empty() {
            return format!(
                "the file-and-name re-check, at revision {}, at a moment this ledger did not \
                 record",
                record.revision
            );
        }
        format!(
            "the file-and-name re-check, at revision {} since {}",
            record.revision, day
        )
    }
}
