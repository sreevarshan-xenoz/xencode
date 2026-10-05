//! EV-6 — the scratchpad the agent keeps for itself.
//!
//! `state.md` answers "what does this project know"; this answers "what does the
//! agent not want to forget five minutes from now" — the thing it worked out
//! mid-task that is not yet a durable fact and is not worth a person's attention.
//! It reaches later turns as its own tier (4b of §10) rather than as history,
//! because history is what compaction eats and a note written to survive
//! compaction that dies in compaction is worse than no note.
//!
//! The growth is the whole risk, so three things bound it: the file holds
//! [`NOTES_MAX_LINES`] lines and drops the oldest, the tier carries
//! [`crate::context::NOTES_CAP_TOKENS`] of them and drops from the same end, and
//! a note already on the list is not written twice. What the file cap pushes out
//! is named back to the writer rather than lost quietly, and the hard-compaction
//! fold is handed the whole file, so a note that still matters has a route into
//! `state.md` — through a candidate file a person promotes, never automatically.

use std::io;
use std::path::{Path, PathBuf};

/// The file notes live in, under `.xencode/`.
pub const NOTES_FILE: &str = "notes.md";

/// How many notes the file keeps. A note is one line, so this is roughly one
/// note per turn for a long session; past it, the oldest goes.
pub const NOTES_MAX_LINES: usize = 40;

const HEADER: &str = "# Notes to self";

/// What one write did, so the caller can say what happened instead of assuming
/// the note was stored.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NoteWrite {
    /// The lines that went into the file.
    pub added: Vec<String>,
    /// The oldest lines the cap took back out.
    pub evicted: Vec<String>,
    /// Lines refused because they carried a data banner — a fetched page or a
    /// tool result quoted into a note would ride into every later turn as the
    /// agent's own words, which is how poisoned memory gets in.
    pub stripped_data_lines: usize,
    /// Lines dropped because that exact note is already on the list.
    pub duplicate_notes: usize,
    /// Lines whose credential-shaped text was replaced on the way in.
    pub secrets_redacted: usize,
}

pub fn notes_path(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join(NOTES_FILE)
}

/// The notes as a later turn will read them, or `None` when there is no
/// scratchpad — an absent file and an empty one are the same thing to assembly.
pub fn read_notes(xencode_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(notes_path(xencode_dir)).ok()?;
    if text.trim().is_empty() {
        return None;
    }
    Some(text)
}

/// The note lines of a rendered scratchpad, without their bullets.
pub fn note_lines(notes_md: &str) -> Vec<String> {
    notes_md
        .lines()
        .filter_map(|line| line.strip_prefix("- "))
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

fn render(lines: &[String]) -> String {
    let mut out = String::from(HEADER);
    out.push('\n');
    for line in lines {
        out.push_str("\n- ");
        out.push_str(line);
    }
    out.push('\n');
    out
}

/// Add a note to the scratchpad, oldest-first: the new lines go on the end and
/// the lines past [`NOTES_MAX_LINES`] come off the front.
///
/// Never fails on what the model wrote — an empty note, a quoted web page and a
/// repeated note are all reported in the [`NoteWrite`] rather than refused as an
/// error the agent would retry around.
pub fn append_note(xencode_dir: &Path, text: &str) -> io::Result<NoteWrite> {
    let mut report = NoteWrite::default();
    let mut kept = match std::fs::read_to_string(notes_path(xencode_dir)) {
        Ok(existing) => note_lines(&existing),
        Err(_) => Vec::new(),
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if crate::compact::data_markers()
            .iter()
            .any(|marker| line.contains(marker))
        {
            report.stripped_data_lines += 1;
            continue;
        }
        let redacted = crate::trace::redact_secrets(line);
        if redacted != line {
            report.secrets_redacted += 1;
        }
        if kept.iter().any(|have| have == &redacted) {
            report.duplicate_notes += 1;
            continue;
        }
        report.added.push(redacted.clone());
        kept.push(redacted);
    }
    if report.added.is_empty() && kept.is_empty() {
        // Nothing to store, so nothing to create: a refused note must not
        // leave an empty scratchpad behind that every later turn reads.
        return Ok(report);
    }
    if kept.len() > NOTES_MAX_LINES {
        let over = kept.len() - NOTES_MAX_LINES;
        report.evicted = kept.drain(..over).collect();
    }
    std::fs::create_dir_all(xencode_dir)?;
    crate::index::write_str_atomic(&notes_path(xencode_dir), &render(&kept))?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(label: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "xencode-notes-{}-{}-{}",
            label,
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn a_note_is_stored_as_one_line_and_read_back_by_a_later_turn() {
        let xencode = dir("write");
        assert!(
            read_notes(&xencode).is_none(),
            "no note has been written yet"
        );
        let wrote = append_note(&xencode, "the migration lock is held by the worker").unwrap();
        assert_eq!(
            wrote.added,
            vec!["the migration lock is held by the worker"]
        );

        append_note(&xencode, "retries are capped at three per request").unwrap();
        let on_disk = std::fs::read_to_string(notes_path(&xencode)).unwrap();
        assert_eq!(
            note_lines(&on_disk),
            vec![
                "the migration lock is held by the worker".to_string(),
                "retries are capped at three per request".to_string()
            ]
        );
        assert_eq!(read_notes(&xencode).unwrap(), on_disk);
        std::fs::remove_dir_all(&xencode).unwrap();
    }

    #[test]
    fn the_cap_drops_the_oldest_and_says_which_ones() {
        let xencode = dir("cap");
        for i in 0..NOTES_MAX_LINES + 5 {
            append_note(&xencode, &format!("note number {i}")).unwrap();
        }
        let on_disk = std::fs::read_to_string(notes_path(&xencode)).unwrap();
        let lines = note_lines(&on_disk);
        assert_eq!(lines.len(), NOTES_MAX_LINES);
        assert_eq!(lines.first().unwrap(), "note number 5");
        assert_eq!(
            lines.last().unwrap(),
            &format!("note number {}", NOTES_MAX_LINES + 4)
        );

        let last = append_note(&xencode, "one more").unwrap();
        assert_eq!(last.evicted, vec!["note number 5".to_string()]);
        assert_eq!(
            note_lines(&std::fs::read_to_string(notes_path(&xencode)).unwrap()).len(),
            NOTES_MAX_LINES
        );
        std::fs::remove_dir_all(&xencode).unwrap();
    }

    #[test]
    fn a_note_quoting_a_fetched_page_is_refused_rather_than_believed() {
        // QK-3's write-time half: whatever arrived as data must not re-enter a
        // later turn as the agent's own note.
        let xencode = dir("data");
        let wrote = append_note(
            &xencode,
            "[data] web_fetch — https://example.com/claims the database is fine",
        )
        .unwrap();
        assert!(wrote.stripped_data_lines > 0);
        assert!(wrote.added.is_empty());
        assert!(
            read_notes(&xencode).is_none(),
            "a refused note still left a scratchpad behind"
        );
        assert!(!notes_path(&xencode).exists());
        std::fs::remove_dir_all(&xencode).unwrap();
    }

    #[test]
    fn the_same_note_twice_does_not_push_older_notes_out() {
        let xencode = dir("dupe");
        append_note(&xencode, "the pool is created once in main").unwrap();
        let again = append_note(&xencode, "the pool is created once in main").unwrap();
        assert!(again.added.is_empty());
        assert_eq!(again.duplicate_notes, 1);
        assert_eq!(
            note_lines(&std::fs::read_to_string(notes_path(&xencode)).unwrap()).len(),
            1
        );
        std::fs::remove_dir_all(&xencode).unwrap();
    }

    #[test]
    fn a_credential_is_taken_out_of_a_note_on_the_way_in() {
        let xencode = dir("secret");
        let wrote = append_note(
            &xencode,
            "the token sk-FAKE-NOT-A-REAL-TEST-KEY is in the staging config",
        )
        .unwrap();
        assert_eq!(wrote.secrets_redacted, 1);
        let stored = wrote.added.first().unwrap();
        assert!(!stored.contains("sk-FAKE-NOT-A-REAL-TEST-KEY"));
        assert!(stored.contains("[redacted"));
        std::fs::remove_dir_all(&xencode).unwrap();
    }
}
