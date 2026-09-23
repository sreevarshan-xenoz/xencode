//! Reading append-only JSONL that a crash may have interrupted.
//!
//! Logs like `metrics.jsonl` and `audit.jsonl` are appended one JSON object
//! per line, which is the right store for a single binary: an append cannot
//! tear an earlier record. A crash or a full disk can, however, leave the
//! *last* line only partly written. Such a file is still useful — everything
//! up to the interrupted append is real evidence — so a partial trailing line
//! is dropped rather than failing the read.

use serde::de::DeserializeOwned;
use std::path::Path;

/// What a tolerant JSONL read found.
#[derive(Debug)]
pub struct JsonlRead<T> {
    /// Every record that parsed.
    pub rows: Vec<T>,
    /// Set when the file ended in the middle of a line, which is what an
    /// interrupted append leaves behind. That last line is dropped.
    pub torn_tail: bool,
    /// Lines that should have been records and were not. Unlike a torn tail
    /// these say something is wrong with the writer, so they are counted
    /// rather than passed over.
    pub bad_lines: usize,
}

impl<T> JsonlRead<T> {
    fn empty() -> Self {
        Self {
            rows: Vec::new(),
            torn_tail: false,
            bad_lines: 0,
        }
    }
}

/// Parse every line of `path` as a `T`.
///
/// A missing or unreadable file reads as empty: a log that has not been
/// written yet is normal, not an error. See [`JsonlRead`] for how a partial
/// trailing line and a corrupt line in the middle are told apart.
pub fn read_jsonl_tolerant<T: DeserializeOwned>(path: &Path) -> JsonlRead<T> {
    let Ok(bytes) = std::fs::read(path) else {
        return JsonlRead::empty();
    };
    if bytes.is_empty() {
        return JsonlRead::empty();
    }

    let ends_with_newline = bytes[bytes.len() - 1] == b'\n';
    let Ok(text) = String::from_utf8(bytes) else {
        // Not text at all: treat the whole file as unusable rather than
        // inventing records out of binary.
        return JsonlRead {
            rows: Vec::new(),
            torn_tail: false,
            bad_lines: 1,
        };
    };

    let mut read = JsonlRead::empty();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<T>(trimmed) {
            Ok(row) => read.rows.push(row),
            Err(_) => {
                let is_last = lines.peek().is_none();
                if is_last && !ends_with_newline {
                    read.torn_tail = true;
                } else {
                    read.bad_lines += 1;
                }
            }
        }
    }
    read
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Deserialize, PartialEq)]
    struct Row {
        seq: u64,
    }

    fn temp_dir() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "xencode-jsonl-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_raw(path: &Path, bytes: &[u8]) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    #[test]
    fn reads_every_complete_record() {
        let dir = temp_dir();
        let path = dir.join("log.jsonl");
        write_raw(&path, b"{\"seq\":1}\n{\"seq\":2}\n");
        let read = read_jsonl_tolerant::<Row>(&path);
        assert_eq!(read.rows, vec![Row { seq: 1 }, Row { seq: 2 }]);
        assert!(!read.torn_tail);
        assert_eq!(read.bad_lines, 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_partial_trailing_line_is_dropped_not_fatal() {
        let dir = temp_dir();
        let path = dir.join("log.jsonl");
        // Exactly what a kill or a full disk leaves: two whole records and a
        // third one cut off mid-object.
        write_raw(&path, b"{\"seq\":1}\n{\"seq\":2}\n{\"seq\":3");
        let read = read_jsonl_tolerant::<Row>(&path);
        assert_eq!(read.rows, vec![Row { seq: 1 }, Row { seq: 2 }]);
        assert!(read.torn_tail);
        assert_eq!(read.bad_lines, 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_broken_line_in_the_middle_is_counted_and_the_rest_survives() {
        let dir = temp_dir();
        let path = dir.join("log.jsonl");
        write_raw(&path, b"{\"seq\":1}\nnot json\n{\"seq\":3}\n");
        let read = read_jsonl_tolerant::<Row>(&path);
        assert_eq!(read.rows, vec![Row { seq: 1 }, Row { seq: 3 }]);
        assert!(!read.torn_tail);
        assert_eq!(read.bad_lines, 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_broken_final_line_that_was_terminated_is_counted_not_blamed_on_a_crash() {
        let dir = temp_dir();
        let path = dir.join("log.jsonl");
        write_raw(&path, b"{\"seq\":1}\ngarbage\n");
        let read = read_jsonl_tolerant::<Row>(&path);
        assert_eq!(read.rows, vec![Row { seq: 1 }]);
        assert!(
            !read.torn_tail,
            "a newline ended the line, so nothing was interrupted"
        );
        assert_eq!(read.bad_lines, 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_and_empty_files_read_as_nothing_rather_than_an_error() {
        let dir = temp_dir();
        let missing = dir.join("absent.jsonl");
        let read = read_jsonl_tolerant::<Row>(&missing);
        assert!(read.rows.is_empty() && !read.torn_tail && read.bad_lines == 0);

        let empty = dir.join("empty.jsonl");
        std::fs::File::create(&empty).unwrap();
        let read = read_jsonl_tolerant::<Row>(&empty);
        assert!(read.rows.is_empty() && !read.torn_tail && read.bad_lines == 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
