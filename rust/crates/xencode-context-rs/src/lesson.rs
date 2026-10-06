//! EV-7: a lesson from a failure, drafted by the program and written by a person.
//!
//! The agent's work is undone in two places that leave a record: `/rewind` puts
//! files back because a person did not want what was done, and `/verify` reports
//! a checklist that failed with an exit code behind it. Both are facts about this
//! repository, and neither is a reason the program is allowed to invent.
//!
//! So this module drafts, and a person decides. An event leaves one line in
//! `<xencode dir>/lesson.candidate.md` under the evidence it came with, and the
//! lesson itself is an empty line the program never fills in. `/lesson approve` —
//! a keystroke, like `/ctx promote` — appends the person's own sentence to
//! `AGENTS.md`. Nothing here writes that file on any other path, and a draft whose
//! lesson line is still blank cannot be approved at all: the words that become
//! instructions are the human's, or nothing is.
//!
//! The blank is not laziness. Most rejected work is rejected without a stated
//! reason, and a reason written by the thing that was rejected is a guess about
//! someone else's motive dressed as a record. Recording the guess would make the
//! drift worse, not better: a later turn would read an invented lesson in
//! `AGENTS.md` and follow it.

use std::path::{Path, PathBuf};

/// The file a lesson waits in.
pub const LESSON_CANDIDATE_FILE: &str = "lesson.candidate.md";

/// How many events one draft may hold. A queue nobody reads is not a queue, and
/// the newest evidence is the one still in front of the person writing the lesson.
pub const LESSON_EVIDENCE_CAP: usize = 8;

/// How many failing checks in a row are worth a nudge on their own. One is a
/// typo; a rewind is already a decision, which is why it needs no streak.
pub const LESSON_CHECK_STREAK: usize = 3;

/// The line a person writes their lesson into, blank until they do.
const LESSON_PREFIX: &str = "- lesson:";

/// One thing that went wrong, as it is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evidence {
    /// The command that reported it — `/rewind`, `/verify`. Never a sentence
    /// this module makes up, because the reader needs to know who said this.
    pub source: String,
    /// What it said: which files went back, which check failed with what exit.
    pub detail: String,
    /// How many times exactly this has now happened. The same failing check
    /// reached three times is one idea, recorded three times: it holds one line
    /// with a count on it rather than three lines, and the count is what a streak
    /// is measured in.
    pub count: usize,
}

impl Evidence {
    pub fn new(source: &str, detail: String) -> Self {
        Self {
            source: source.to_string(),
            detail,
            count: 1,
        }
    }
}

/// What the draft holds: the events, and whether a person has written the lesson.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LessonDraft {
    pub evidence: Vec<Evidence>,
    /// The person's words, or `None` while the line is blank. Whitespace alone is
    /// blank: a lesson of three spaces is a person who has not written one yet.
    pub lesson: Option<String>,
}

impl LessonDraft {
    /// The file as text. Written atomically by [`draft_lesson`], and editable by
    /// hand for the same reason `state.candidate.md` is: the point is for a person
    /// to read it and change it.
    fn to_markdown(&self) -> String {
        let mut out = String::from(
            "# A lesson, waiting for your words\n\nNothing below was decided by anyone but you. \
             This draft was written because the events in it happened; the lesson line is empty \
             because the reason they happened is not something this program can know.\n\n\
             ## What happened\n\n",
        );
        for event in &self.evidence {
            out.push_str(&format!("- {}: {}", event.source, event.detail));
            if event.count > 1 {
                out.push_str(&format!(" ({} times)", event.count));
            }
            out.push('\n');
        }
        out.push_str("\n## The lesson\n\n");
        out.push_str(&match &self.lesson {
            Some(words) => format!("{LESSON_PREFIX} {words}\n"),
            None => format!("{LESSON_PREFIX}\n"),
        });
        out.push_str(
            "\nWrite it in this file, or with `/lesson set <words>`. `/lesson approve` appends \
             that one line to `AGENTS.md` and clears this draft; `/lesson drop` clears it without \
             writing anything.\n",
        );
        out
    }
}

/// Where a lesson waits for the person's decision.
pub fn lesson_candidate_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join(LESSON_CANDIDATE_FILE)
}

/// The draft, if one is waiting. A file with no `## What happened` and no lesson
/// line is not a draft: it is somebody's file, and this returns `None` rather
/// than reading it as one.
pub fn read_lesson(xencode_dir: &Path) -> Option<LessonDraft> {
    let text = std::fs::read_to_string(lesson_candidate_path(xencode_dir)).ok()?;
    parse_lesson(&text)
}

fn parse_lesson(text: &str) -> Option<LessonDraft> {
    if !text.contains("## What happened") && !text.contains(LESSON_PREFIX) {
        return None;
    }
    let mut draft = LessonDraft::default();
    let mut in_evidence = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "## What happened" {
            in_evidence = true;
            continue;
        }
        if trimmed.starts_with("## ") {
            in_evidence = false;
        }
        if in_evidence {
            if let Some(bullet) = trimmed.strip_prefix("- ") {
                // A stored event is `source: detail`, and a detail containing its
                // own colons is normal — an exit code, a path, a quoted line from
                // a compiler. Split once.
                let (source, detail) = bullet
                    .split_once(": ")
                    .map_or((bullet.to_string(), String::new()), |(s, d)| {
                        (s.to_string(), d.to_string())
                    });
                // A count is this module's own mark, read back off the end so a
                // hand-edited line without one still parses, as one event. Split
                // at the last bracket, because the detail is full of them: a
                // compiler says `clippy FAIL (exit 1)`.
                let (detail, count) = match detail
                    .strip_suffix(")")
                    .and_then(|head| head.rsplit_once(" ("))
                {
                    Some((body, tail)) if tail.ends_with(" times") => (
                        body.to_string(),
                        tail[..tail.len() - " times".len()]
                            .trim()
                            .parse()
                            .unwrap_or(1),
                    ),
                    _ => (detail, 1),
                };
                draft.evidence.push(Evidence {
                    source,
                    detail,
                    count: count.max(1),
                });
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix(LESSON_PREFIX) {
            let words = rest.trim().trim_start_matches(':').trim();
            if !words.is_empty() {
                draft.lesson = Some(words.to_string());
            }
        }
    }
    Some(draft)
}

/// Why an event could not be recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LessonWrite {
    /// The draft exists, holds something this parser does not recognise, and is
    /// not going to be rewritten over.
    NotALessonDraft,
    /// The draft could not be read or written.
    CouldNotWrite(String),
}

impl std::fmt::Display for LessonWrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LessonWrite::NotALessonDraft => f.write_str(
                "lesson.candidate.md is not a lesson draft, and it will not be rewritten over — \
                 move it aside or delete it by hand",
            ),
            LessonWrite::CouldNotWrite(problem) => {
                write!(f, "the lesson draft could not be written: {problem}")
            }
        }
    }
}

/// Record one event. Each call adds a line and keeps any lesson a person has
/// already written, so the newest failure cannot silently eat the sentence that
/// was being written about the last one.
///
/// Returns the draft as it now stands, and whether the whole event was a repeat
/// of the line already at the bottom — a person running `/verify` four times in a
/// row against the same broken build has had one idea, not four.
pub fn draft_lesson(
    event: &Evidence,
    xencode_dir: &Path,
) -> Result<(LessonDraft, bool), LessonWrite> {
    std::fs::create_dir_all(xencode_dir)
        .map_err(|problem| LessonWrite::CouldNotWrite(problem.to_string()))?;
    let path = lesson_candidate_path(xencode_dir);
    let mut draft = match std::fs::read_to_string(&path) {
        Ok(text) => parse_lesson(&text).ok_or(LessonWrite::NotALessonDraft)?,
        Err(problem) if problem.kind() == std::io::ErrorKind::NotFound => LessonDraft::default(),
        Err(problem) => return Err(LessonWrite::CouldNotWrite(problem.to_string())),
    };
    let repeat = draft
        .evidence
        .last()
        .is_some_and(|last| last.source == event.source && last.detail == event.detail);
    if repeat {
        // One line, a bigger count. Four identical failures are one symptom seen
        // four times, and the streak that asks for a lesson counts the times.
        if let Some(last) = draft.evidence.last_mut() {
            last.count += 1;
        }
        write_lesson_file(&draft, xencode_dir)?;
        return Ok((draft, true));
    }
    draft.evidence.push(event.clone());
    // Keep the newest `LESSON_EVIDENCE_CAP` and drop the oldest off the front.
    let overflow = draft.evidence.len().saturating_sub(LESSON_EVIDENCE_CAP);
    draft.evidence.drain(..overflow);
    write_lesson_file(&draft, xencode_dir)?;
    Ok((draft, false))
}

fn write_lesson_file(draft: &LessonDraft, xencode_dir: &Path) -> Result<(), LessonWrite> {
    let path = lesson_candidate_path(xencode_dir);
    crate::index::write_str_atomic(&path, &draft.to_markdown())
        .map_err(|problem| LessonWrite::CouldNotWrite(problem.to_string()))
}

/// Why the lesson line could not be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LessonRefusal {
    /// Nothing is waiting: no event has been recorded, or it was already dealt with.
    NoDraft,
    /// The draft is not recognisable as one.
    NotADraft,
    /// `/lesson set` was given nothing. A refusal here is the whole point of the
    /// command: an empty lesson is not a lesson.
    EmptyLesson,
    /// The draft holds no lesson yet.
    LessonBlank,
    /// `AGENTS.md` could not be read or written.
    CouldNotWrite(String),
}

impl std::fmt::Display for LessonRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LessonRefusal::NoDraft => f.write_str(
                "no lesson is waiting — a rewind or a failing check drafts one, and /lesson drop clears it",
            ),
            LessonRefusal::NotADraft => f.write_str(
                "lesson.candidate.md is not a lesson draft, so nothing here will be read out of it",
            ),
            LessonRefusal::EmptyLesson => {
                f.write_str("a lesson needs words: /lesson set <what to do differently next time>")
            }
            LessonRefusal::LessonBlank => f.write_str(
                "the lesson line is still blank, and a line this program wrote would be an \
                 invention, not a lesson — write it with /lesson set, or edit the draft",
            ),
            LessonRefusal::CouldNotWrite(problem) => {
                write!(f, "AGENTS.md could not be written: {problem}")
            }
        }
    }
}

/// Put the person's sentence into the draft. The evidence stays where it was.
pub fn set_lesson(words: &str, xencode_dir: &Path) -> Result<LessonDraft, LessonRefusal> {
    let mut draft = read_lesson(xencode_dir).ok_or(LessonRefusal::NoDraft)?;
    let trimmed = words.trim();
    if trimmed.is_empty() {
        return Err(LessonRefusal::EmptyLesson);
    }
    draft.lesson = Some(trimmed.to_string());
    write_lesson_file(&draft, xencode_dir)
        .map_err(|problem| LessonRefusal::CouldNotWrite(problem.to_string()))?;
    Ok(draft)
}

/// What one approval did, so the screen can say exactly that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovedLesson {
    /// The line appended to `AGENTS.md`, as the person wrote it.
    pub lesson: String,
    /// The file did not exist and this write created it.
    pub agents_created: bool,
    /// After this write, the file's bytes are not the bytes `/trust` was given.
    ///
    /// Trust is keyed on content, so appending a line takes the whole file back
    /// to being data until a person trusts it again. This is reported rather than
    /// fixed: re-trusting bytes that a machine just moved is not this module's
    /// decision to make, and doing it silently is how a lesson becomes an
    /// instruction nobody read.
    pub awaiting_trust: bool,
    /// How many events the draft was holding, so the message can name what the
    /// lesson was drawn from.
    pub evidence_count: usize,
}

/// Append the person's lesson to the workspace's `AGENTS.md` and clear the draft.
///
/// The only writer of `AGENTS.md` in the product, and it is reachable only from a
/// command a person typed. One line is added, at the end, under the heading this
/// module owns; everything already in the file is left byte for byte.
pub fn approve_lesson(xencode_dir: &Path) -> Result<ApprovedLesson, LessonRefusal> {
    let draft = read_lesson(xencode_dir).ok_or(LessonRefusal::NoDraft)?;
    let lesson = draft.lesson.clone().ok_or(LessonRefusal::LessonBlank)?;
    let root = xencode_dir
        .parent()
        .map(|parent| parent.to_path_buf())
        .unwrap_or_else(|| xencode_dir.to_path_buf());
    let path = root.join("AGENTS.md");
    let before = std::fs::read_to_string(&path).ok();
    let mut body = before
        .clone()
        .unwrap_or_else(|| "# Project instructions\n".to_string());
    if !body.ends_with('\n') {
        body.push('\n');
    }
    if !body.contains("## Lessons") {
        body.push_str("\n## Lessons\n");
    }
    body.push_str(&format!("- {lesson}\n"));
    crate::index::write_str_atomic(&path, &body)
        .map_err(|problem| LessonRefusal::CouldNotWrite(problem.to_string()))?;
    let awaiting_trust = !crate::trust::trusted_agents_hashes(xencode_dir)
        .contains(&crate::trust::agents_sha256(&body));
    std::fs::remove_file(lesson_candidate_path(xencode_dir))
        .map_err(|problem| LessonRefusal::CouldNotWrite(problem.to_string()))?;
    Ok(ApprovedLesson {
        lesson,
        agents_created: before.is_none(),
        awaiting_trust,
        evidence_count: draft.evidence.len(),
    })
}

/// Clear the draft without writing anything anywhere. An event that turned out to
/// be nothing is allowed to go away — that is the person's call, and it is the
/// same call `/ctx drop` gives a fold.
pub fn drop_lesson(xencode_dir: &Path) -> Result<LessonDraft, LessonRefusal> {
    let draft = read_lesson(xencode_dir).ok_or(LessonRefusal::NoDraft)?;
    std::fs::remove_file(lesson_candidate_path(xencode_dir))
        .map_err(|problem| LessonRefusal::CouldNotWrite(problem.to_string()))?;
    Ok(draft)
}

/// How many failing checks the recorded events add up to, counting back from the
/// newest. A rewind is a decision rather than a symptom, so the streak that asks
/// for a lesson is only ever about checks.
pub fn check_streak(draft: &LessonDraft) -> usize {
    draft
        .evidence
        .iter()
        .rev()
        .take_while(|event| event.source == "/verify")
        .map(|event| event.count)
        .sum()
}

/// Whether this draft is now worth interrupting a person for.
///
/// A rewind asks straight away. A failing check asks once it has failed
/// [`LESSON_CHECK_STREAK`] times on the same run of attempts, because one red
/// check is usually the next command's business rather than a lesson.
pub fn asks_for_words(draft: &LessonDraft, last: &Evidence) -> bool {
    last.source == "/rewind" || check_streak(draft) >= LESSON_CHECK_STREAK
}

/// The draft as it should be printed: the events, and the lesson line whether or
/// not it has been filled in.
pub fn render_lesson(draft: &LessonDraft) -> String {
    draft.to_markdown()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(label: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "xencode-lesson-{}-{}-{label}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(base.join(".xencode")).unwrap();
        base
    }

    fn xencode_of(root: &Path) -> PathBuf {
        root.join(".xencode")
    }

    #[test]
    fn a_rewind_drafts_a_lesson_and_writes_nothing_else() {
        let root = dir("rewind");
        let xencode = xencode_of(&root);
        let (draft, repeat) = draft_lesson(
            &Evidence::new(
                "/rewind",
                "1 turn back: src/auth.rs, src/session.rs".to_string(),
            ),
            &xencode,
        )
        .unwrap();
        assert!(!repeat);
        assert_eq!(draft.evidence.len(), 1);
        assert_eq!(draft.evidence[0].source, "/rewind");
        assert!(draft.lesson.is_none());
        // The draft is on disk and is the whole of what happened: AGENTS.md is
        // not touched by an event, only by a person approving one.
        assert!(!root.join("AGENTS.md").exists());
        let read = read_lesson(&xencode).unwrap();
        assert_eq!(read, draft);
        // And the reason it is worth a lesson is the rewind itself, not a streak.
        assert!(asks_for_words(&draft, &draft.evidence[0]));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_empty_lesson_line_cannot_be_approved_and_agents_md_is_untouched() {
        // The guard this item exists for. Remove it and a sentence invented by the
        // program lands in the file that tells every later turn what to do.
        let root = dir("blank");
        let xencode = xencode_of(&root);
        std::fs::write(
            root.join("AGENTS.md"),
            "# Project instructions\n- Rust first\n",
        )
        .unwrap();
        let before = std::fs::read_to_string(root.join("AGENTS.md")).unwrap();
        draft_lesson(
            &Evidence::new("/rewind", "1 turn back: src/a.rs".to_string()),
            &xencode,
        )
        .unwrap();
        let refused = approve_lesson(&xencode).unwrap_err();
        assert_eq!(refused, LessonRefusal::LessonBlank);
        assert!(refused.to_string().contains("an invention, not a lesson"));
        assert_eq!(
            std::fs::read_to_string(root.join("AGENTS.md")).unwrap(),
            before,
            "a refused approval still wrote the file"
        );
        assert!(lesson_candidate_path(&xencode).exists());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn approving_writes_the_persons_own_line_and_clears_the_draft() {
        let root = dir("approve");
        let xencode = xencode_of(&root);
        draft_lesson(
            &Evidence::new("/verify", "build FAIL (exit 101)".to_string()),
            &xencode,
        )
        .unwrap();
        set_lesson("  run cargo build before /verify, not after  ", &xencode).unwrap();
        let done = approve_lesson(&xencode).unwrap();
        assert_eq!(done.lesson, "run cargo build before /verify, not after");
        assert!(done.agents_created);
        assert_eq!(done.evidence_count, 1);
        let written = std::fs::read_to_string(root.join("AGENTS.md")).unwrap();
        assert!(written.contains("- run cargo build before /verify, not after"));
        assert!(written.contains("## Lessons"));
        assert!(!lesson_candidate_path(&xencode).exists());
        assert_eq!(read_lesson(&xencode), None);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_existing_agents_file_keeps_every_byte_and_asks_for_trust_again() {
        let root = dir("keep");
        let xencode = xencode_of(&root);
        std::fs::write(
            root.join("AGENTS.md"),
            "# Project instructions\n- Rust first\n- No mocks\n",
        )
        .unwrap();
        // Trust these exact bytes first, the way /trust does, so the claim after
        // the write is about a file that really was trusted and really moved.
        let hashes: std::collections::BTreeSet<String> = std::iter::once(
            crate::trust::agents_sha256("# Project instructions\n- Rust first\n- No mocks\n"),
        )
        .collect();
        std::fs::create_dir_all(xencode.join("cache")).unwrap();
        std::fs::write(
            xencode.join("cache").join("agents_trust.json"),
            serde_json::to_string(&Vec::from_iter(hashes)).unwrap(),
        )
        .unwrap();
        draft_lesson(
            &Evidence::new("/rewind", "2 turns back: src/a.rs".to_string()),
            &xencode,
        )
        .unwrap();
        set_lesson("do not edit a file that is already dirty", &xencode).unwrap();
        let done = approve_lesson(&xencode).unwrap();
        assert!(!done.agents_created);
        assert!(done.awaiting_trust);
        let written = std::fs::read_to_string(root.join("AGENTS.md")).unwrap();
        assert!(written.starts_with("# Project instructions\n- Rust first\n- No mocks\n"));
        assert_eq!(
            written
                .lines()
                .filter(|line| line.starts_with("- do not edit"))
                .count(),
            1
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_new_failure_keeps_the_lesson_a_person_was_writing() {
        let root = dir("keep-words");
        let xencode = xencode_of(&root);
        draft_lesson(
            &Evidence::new("/verify", "tests FAIL (exit 1)".to_string()),
            &xencode,
        )
        .unwrap();
        set_lesson("read the failing test before changing the code", &xencode).unwrap();
        let (draft, _) = draft_lesson(
            &Evidence::new("/rewind", "1 turn back: src/b.rs".to_string()),
            &xencode,
        )
        .unwrap();
        assert_eq!(draft.evidence.len(), 2);
        assert_eq!(
            draft.lesson.as_deref(),
            Some("read the failing test before changing the code")
        );
        assert!(!root.join("AGENTS.md").exists());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_same_failure_twice_is_one_event_and_evidence_is_capped() {
        let root = dir("cap");
        let xencode = xencode_of(&root);
        let first = Evidence::new("/verify", "clippy FAIL (exit 1)".to_string());
        draft_lesson(&first, &xencode).unwrap();
        let (twice, repeat) = draft_lesson(&first, &xencode).unwrap();
        assert!(repeat, "an identical event counted itself twice");
        assert_eq!(twice.evidence.len(), 1);
        assert_eq!(twice.evidence[0].count, 2);
        assert_eq!(read_lesson(&xencode).unwrap(), twice);
        for n in 0..(LESSON_EVIDENCE_CAP + 4) {
            draft_lesson(
                &Evidence::new("/verify", format!("check {n} FAIL (exit 1)")),
                &xencode,
            )
            .unwrap();
        }
        let capped = read_lesson(&xencode).unwrap();
        assert_eq!(capped.evidence.len(), LESSON_EVIDENCE_CAP);
        // The newest survives and the oldest is what fell off the front.
        assert!(capped.evidence.last().unwrap().detail.contains("check 11"));
        assert!(!capped.evidence[0].detail.contains("check 0"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_same_check_failing_three_times_asks_for_words() {
        // The live run found this: a person re-running `/verify` against one
        // broken build produces the same event every time, and an event held as
        // one line with a count on it used to leave the streak stuck at one.
        let root = dir("same-three");
        let xencode = xencode_of(&root);
        let same = Evidence::new("/verify", "fmt exit 1, test exit 1".to_string());
        for n in 1..LESSON_CHECK_STREAK {
            let (draft, _) = draft_lesson(&same, &xencode).unwrap();
            assert_eq!(draft.evidence.len(), 1);
            assert_eq!(check_streak(&draft), n);
            assert!(!asks_for_words(&draft, &same), "asked after {n}");
        }
        let (draft, _) = draft_lesson(&same, &xencode).unwrap();
        assert_eq!(check_streak(&draft), LESSON_CHECK_STREAK);
        assert!(asks_for_words(&draft, &same));
        assert_eq!(draft.evidence.len(), 1, "a count should not become lines");
        // And the count survives the file: it is read back, not remembered.
        let on_disk = read_lesson(&xencode).unwrap();
        assert_eq!(on_disk.evidence[0].count, LESSON_CHECK_STREAK);
        assert_eq!(on_disk, draft);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn two_failing_checks_do_not_ask_and_three_do() {
        let root = dir("streak");
        let xencode = xencode_of(&root);
        for n in 0..2 {
            let last = Evidence::new("/verify", format!("check {n} FAIL (exit 1)"));
            let (draft, _) = draft_lesson(&last, &xencode).unwrap();
            assert_eq!(check_streak(&draft), n + 1);
            assert!(
                !asks_for_words(&draft, &last),
                "asked for a lesson after {} failing check(s)",
                n + 1
            );
        }
        let last = Evidence::new("/verify", "check 2 FAIL (exit 1)".to_string());
        let (draft, _) = draft_lesson(&last, &xencode).unwrap();
        assert_eq!(check_streak(&draft), LESSON_CHECK_STREAK);
        assert!(asks_for_words(&draft, &last));
        // A rewind in the middle breaks the streak: it is not the same symptom.
        let (mixed, _) = draft_lesson(
            &Evidence::new("/rewind", "1 turn back: src/a.rs".to_string()),
            &xencode,
        )
        .unwrap();
        assert_eq!(check_streak(&mixed), 0);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_file_that_is_not_a_draft_is_never_rewritten_over() {
        let root = dir("notadraft");
        let xencode = xencode_of(&root);
        std::fs::write(
            lesson_candidate_path(&xencode),
            "somebody's own notes about something else entirely\n",
        )
        .unwrap();
        assert_eq!(read_lesson(&xencode), None);
        let refused = draft_lesson(
            &Evidence::new("/rewind", "1 turn back: src/a.rs".to_string()),
            &xencode,
        )
        .unwrap_err();
        assert_eq!(refused, LessonWrite::NotALessonDraft);
        assert_eq!(
            std::fs::read_to_string(lesson_candidate_path(&xencode)).unwrap(),
            "somebody's own notes about something else entirely\n"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_lesson_written_by_hand_in_the_draft_is_read_back() {
        // The draft is a file a person may open and type into, so the parser has
        // to see their line and not only the one this module wrote.
        let root = dir("handedit");
        let xencode = xencode_of(&root);
        draft_lesson(
            &Evidence::new("/rewind", "1 turn back: src/a.rs".to_string()),
            &xencode,
        )
        .unwrap();
        let path = lesson_candidate_path(&xencode);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(
            &path,
            text.replace(
                "- lesson:",
                "- lesson: never run the destructive migration twice",
            ),
        )
        .unwrap();
        let draft = read_lesson(&xencode).unwrap();
        assert_eq!(
            draft.lesson.as_deref(),
            Some("never run the destructive migration twice")
        );
        assert_eq!(draft.evidence.len(), 1);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_empty_lesson_is_not_a_lesson() {
        let root = dir("empty-words");
        let xencode = xencode_of(&root);
        draft_lesson(
            &Evidence::new("/rewind", "1 turn back: src/a.rs".to_string()),
            &xencode,
        )
        .unwrap();
        assert_eq!(
            set_lesson("   ", &xencode).unwrap_err(),
            LessonRefusal::EmptyLesson
        );
        assert_eq!(read_lesson(&xencode).unwrap().lesson, None);
        assert_eq!(
            drop_lesson(&xencode).unwrap().evidence.len(),
            1,
            "dropping reported the wrong draft"
        );
        assert_eq!(
            drop_lesson(&xencode).unwrap_err(),
            LessonRefusal::NoDraft,
            "a second drop found a draft that was already gone"
        );
        assert!(!root.join("AGENTS.md").exists());
        std::fs::remove_dir_all(&root).ok();
    }
}
