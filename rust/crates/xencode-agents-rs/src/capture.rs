//! AR-4: keep what a worker said, and what xencode inferred, in separate files.
//!
//! A normalised event stream is the convenient thing to store and it is the
//! wrong thing to store alone: every judgement the adapter made — that this
//! chunk was prose, that this line was the end of the run, that this tool call
//! and that output are the same call — is baked into it and cannot be argued
//! with afterwards. So one capture is three files:
//!
//! ```text
//! <root>/<agent>/capture/
//!     raw.jsonl          every line the vendor printed, verbatim
//!     normalized.jsonl   the common events, each naming its raw line
//!     metadata.json      what was run, what it cost, what was truncated
//! ```
//!
//! The raw stream is never replaced. `Origin::Synthesised` draws the same line
//! in memory between "the worker said this" and "xencode filled this in"; these
//! files keep that line on disk, so any later claim about a run can be proved
//! against the bytes rather than against the reader's judgement.
//!
//! # What is deliberately not here
//!
//! `raw.jsonl` is unredacted, because a redacted raw stream is not a raw stream.
//! A vendor that prints a credential into its own event stream therefore puts it
//! in this file. Two things contain that: a capture only exists when the operator
//! asks for one by directory, and every file is written `0600` through the same
//! owner-only atomic write the rest of xencode's private state uses. Redaction
//! that keeps recoverability is [`AR-5`]'s job, not this module's.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::probe::{RunCapture, Usage};
use crate::protocol::{self, AgentEvent, Origin};
use crate::roster::Provenance;

/// One line of the vendor's own output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawLine {
    /// 1-based, so a record never has to claim it came from line zero.
    pub line: usize,
    /// The line exactly as it arrived, without its newline.
    pub text: String,
}

/// One normalised event, with the raw line it was read out of.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredEvent {
    /// Position in the normalised stream, 1-based and gapless.
    pub seq: usize,
    /// The [`RawLine`] this came from. `None` only for an event xencode
    /// constructed itself, which is the same fact [`Origin`] records.
    pub raw_line: Option<usize>,
    pub event: AgentEvent,
}

/// What was run, and what it said about itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    pub agent: String,
    pub binary: Option<String>,
    pub version: Option<String>,
    /// The argv actually executed, with the prompt already replaced by a marker.
    pub argv: Vec<String>,
    pub workdir: String,
    pub exit_code: Option<i32>,
    pub duration_ms: u128,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub stream_recognised: bool,
    pub session_id: Option<String>,
    pub stopped_on_auth: bool,
    pub permission_signal: Option<String>,
    pub usage: Option<Usage>,
    pub model: Option<String>,
    pub provenance: Provenance,
    pub failure: Option<String>,
    /// How many raw lines were stored, and how many events came out of them.
    pub raw_lines: usize,
    pub normalized_events: usize,
    /// How many of those events are xencode's own inference rather than a
    /// worker's statement. Counted here so a reader does not have to trust the
    /// origin field on each row to see how much of a capture is bookkeeping.
    pub synthesized_events: usize,
    /// The code that turned raw into normalized. A capture is a measurement, so
    /// it records the instrument.
    pub normalizer: String,
}

/// A whole capture, read back off disk.
#[derive(Debug, Clone, PartialEq)]
pub struct Capture {
    pub metadata: Metadata,
    pub raw: Vec<RawLine>,
    pub events: Vec<StoredEvent>,
    /// The [`AR-5`] propagation envelopes, each derived from one stored event and
    /// written redacted. This is the copy that flows to the ledger and the
    /// metrics, so it never carries a credential the sealed `raw.jsonl` holds.
    pub envelopes: Vec<crate::envelope::Envelope>,
}

impl Capture {
    /// The trace view: one capture's events in stream order, with the raw line
    /// behind each one.
    ///
    /// This is the whole point of normalising into [`AgentEvent`] — two vendors
    /// that agree on nothing else end up in this one rendering, and where they
    /// disagree is visible as a difference in the rows rather than in the code
    /// that printed them.
    pub fn trace(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{} — {}",
            self.metadata.agent,
            self.metadata
                .version
                .clone()
                .unwrap_or_else(|| "version not reported".to_string())
        );
        let _ = writeln!(
            out,
            "argv: {}",
            if self.metadata.argv.is_empty() {
                "(nothing was launched)".to_string()
            } else {
                self.metadata.argv.join(" ")
            }
        );
        let _ = writeln!(
            out,
            "exit: {}, {} ms, {} raw line(s), {} event(s)",
            self.metadata
                .exit_code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.metadata.duration_ms,
            self.metadata.raw_lines,
            self.metadata.normalized_events,
        );
        if self.metadata.stopped_on_auth {
            let _ = writeln!(out, "stopped on its authentication check");
        }
        if let Some(failure) = &self.metadata.failure {
            let _ = writeln!(out, "failure: {failure}");
        }
        if self.metadata.stdout_truncated {
            let _ = writeln!(
                out,
                "the raw stream was cut off by the reader's limit; what follows is a prefix"
            );
        }
        let _ = writeln!(out);
        if self.events.is_empty() {
            let _ = writeln!(
                out,
                "(no event in {} raw line(s); the stream was not machine-readable)",
                self.metadata.raw_lines
            );
        }
        for stored in &self.events {
            let _ = writeln!(
                out,
                "{:>4}  {:<20}  {:>8}  {body}",
                stored.seq,
                stored.event.name(),
                match stored.raw_line {
                    Some(n) => format!("raw#{n}"),
                    None => "xencode".to_string(),
                },
                body = describe(&stored.event),
            );
        }
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "run completed: {}",
            if protocol::run_completed(
                &self
                    .events
                    .iter()
                    .map(|s| s.event.clone())
                    .collect::<Vec<_>>()
            ) {
                "yes"
            } else {
                "no — no completion event in this stream"
            }
        );
        let _ = write!(
            out,
            "{} of {} event(s) are xencode's inference, the rest the worker's own words",
            self.metadata.synthesized_events, self.metadata.normalized_events
        );
        out
    }
}

/// Write one run's capture and return the directory holding it.
///
/// Every file is written through the owner-only atomic write, so a reader never
/// sees a half-written capture and a capture never lands world-readable.
pub fn write_capture(root: &Path, run: &RunCapture) -> Result<PathBuf, String> {
    let dir = root.join(&run.agent).join("capture");
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;

    let raw = raw_lines(&run.stdout);
    let events = normalise_all(&run.agent, &raw);
    let synthesized = events
        .iter()
        .filter(|s| s.event.origin() == Origin::Synthesised)
        .count();

    let metadata = Metadata {
        agent: run.agent.clone(),
        binary: run.binary.clone(),
        version: run.version.clone(),
        argv: run.argv.clone(),
        workdir: run.workdir.clone(),
        exit_code: run.exit_code,
        duration_ms: run.duration_ms,
        stdout_truncated: run.stdout_truncated,
        stderr_truncated: run.stderr_truncated,
        stream_recognised: run.stream_recognised,
        session_id: run.session_id.clone(),
        stopped_on_auth: run.stopped_on_auth,
        permission_signal: run.permission_signal.clone(),
        usage: run.usage.clone(),
        model: run.model.clone(),
        provenance: run.provenance,
        failure: run.failure.clone(),
        raw_lines: raw.len(),
        normalized_events: events.len(),
        synthesized_events: synthesized,
        normalizer: "xencode-agents-rs::protocol::normalise_line".to_string(),
    };

    write_jsonl(&dir.join("raw.jsonl"), &raw)?;
    write_jsonl(&dir.join("normalized.jsonl"), &events)?;

    // AR-5: the propagation copy, derived from the stored events and written
    // redacted. `metadata.session_id` is the session the vendor opened; a replay
    // has no job id and no per-line clock, so those fields state that plainly.
    let envelopes = crate::envelope::envelopes_for_capture(
        &run.agent,
        run.session_id.as_deref(),
        None,
        0,
        &events,
        &raw,
    );
    crate::envelope::write_envelopes(&dir, &envelopes)?;

    write_json(
        &dir.join("metadata.json"),
        &serde_json::to_string_pretty(&metadata)
            .map_err(|e| format!("cannot serialise the metadata: {e}"))?,
    )?;
    Ok(dir)
}

/// Read a capture back. An unreadable capture is an error, never a partial one:
/// a trace view built from a file that failed to parse would present a truncated
/// run as if it were the whole run.
///
/// The argument may be the directory holding the three files, or the `<agent>`
/// directory above it, since that is what `--capture-dir` produces and what an
/// operator is likeliest to point at.
pub fn read_capture(dir: &Path) -> Result<Capture, String> {
    let dir = locate_capture(dir)?;
    let raw = read_jsonl::<RawLine>(&dir.join("raw.jsonl"))?;
    let events = read_jsonl::<StoredEvent>(&dir.join("normalized.jsonl"))?;
    let metadata: Metadata = serde_json::from_str(&read_text(&dir.join("metadata.json"))?)
        .map_err(|e| {
            format!(
                "{}: the metadata does not parse: {e}",
                dir.join("metadata.json").display()
            )
        })?;
    // DB-5: a torn trailing envelope line is a write interrupted mid-line, not a
    // stored envelope. It is dropped rather than failing the whole read; the
    // sealed raw stream still holds every byte, so nothing is truly lost.
    let read = crate::envelope::read_envelopes(&dir)?;
    Ok(Capture {
        metadata,
        raw,
        events,
        envelopes: read.envelopes,
    })
}

/// Resolve the directory that actually holds a capture's three files, accepting
/// either that directory or the `<agent>` directory one level above it.
fn locate_capture(dir: &Path) -> Result<PathBuf, String> {
    if dir.join("raw.jsonl").exists() {
        return Ok(dir.to_path_buf());
    }
    let nested = dir.join("capture");
    if nested.join("raw.jsonl").exists() {
        return Ok(nested);
    }
    Err(format!(
        "{} holds no capture (looked for raw.jsonl here and in ./capture)",
        dir.display()
    ))
}

/// Every capture under `root`, so one trace view can show several vendors side by
/// side. A single capture directory yields one; a probe's `--capture-dir` root
/// yields one per agent, ordered by agent name so the rendering is stable.
pub fn find_captures(root: &Path) -> Vec<PathBuf> {
    if let Ok(dir) = locate_capture(root) {
        return vec![dir];
    }
    let mut found = Vec::new();
    collect_captures(root, &mut found, 0);
    found.sort();
    found
}

fn collect_captures(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    // Bounded: a capture is always `root/<agent>/capture`, so three levels is
    // generous and stops this from walking an arbitrary tree.
    if depth > 3 {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if path.join("raw.jsonl").is_file() {
                out.push(path);
            } else {
                collect_captures(&path, out, depth + 1);
            }
        }
    }
}

/// Split a run's stdout into numbered lines. Blank lines are kept, because a
/// blank line in the raw stream is evidence about the stream.
fn raw_lines(stdout: &str) -> Vec<RawLine> {
    let mut lines = stdout
        .split('\n')
        .enumerate()
        .map(|(i, text)| RawLine {
            line: i + 1,
            text: text.to_string(),
        })
        .collect::<Vec<_>>();
    // A trailing newline is a line ending, not an empty final line.
    if lines.last().is_some_and(|l| l.text.is_empty()) {
        lines.pop();
    }
    lines
}

/// Run the normaliser over the raw stream, remembering which line each event
/// came from.
fn normalise_all(agent: &str, raw: &[RawLine]) -> Vec<StoredEvent> {
    let mut out = Vec::new();
    for line in raw {
        for event in protocol::normalise_line(agent, &line.text) {
            out.push(StoredEvent {
                seq: out.len() + 1,
                raw_line: Some(line.line),
                event,
            });
        }
    }
    out
}

/// One line of what an event actually says, so the trace is readable rather
/// than a list of variant names.
fn describe(event: &AgentEvent) -> String {
    match event {
        AgentEvent::SessionStarted { session_id, .. } => {
            format!(
                "session {}",
                session_id.as_deref().unwrap_or("(not reported)")
            )
        }
        AgentEvent::Message { text, .. } => clip(text),
        AgentEvent::ToolRequested { tool, call_id, .. } => {
            format!("{tool} {}", id_suffix(call_id))
        }
        AgentEvent::ToolStarted { tool, call_id, .. } => {
            format!("{tool} {}", id_suffix(call_id))
        }
        AgentEvent::ToolOutput {
            tool,
            call_id,
            output,
            ..
        } => match output {
            Some(o) => format!("{} {} → {}", tool, id_suffix(call_id), clip(o)),
            None => format!(
                "{} {} → (no output in the stream)",
                tool,
                id_suffix(call_id)
            ),
        },
        AgentEvent::FileChanged { path, .. } => path.clone(),
        AgentEvent::PermissionRequested { tool, .. } => {
            format!("approval wanted for {tool}")
        }
        AgentEvent::Error { message, .. } => clip(message),
        AgentEvent::Completed { outcome, .. } => match outcome {
            Some(o) => format!("completed ({o})"),
            None => "completed".to_string(),
        },
        AgentEvent::SessionEnded { reason, .. } => match reason {
            Some(o) => format!("session ended ({o})"),
            None => "session ended".to_string(),
        },
    }
}

fn id_suffix(call_id: &Option<String>) -> String {
    match call_id {
        Some(id) => format!("[{id}]"),
        None => "[no id from this vendor]".to_string(),
    }
}

/// Shorten for display. The stored event keeps the whole text; this is only the
/// trace view, which has to stay readable on a terminal.
fn clip(text: &str) -> String {
    const LIMIT: usize = 120;
    let one_line = text.replace('\n', " ⏎ ");
    if one_line.chars().count() <= LIMIT {
        return one_line;
    }
    let kept: String = one_line.chars().take(LIMIT).collect();
    format!("{kept}…")
}

fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<(), String> {
    let mut buf = Vec::new();
    for record in records {
        let line = serde_json::to_string(record)
            .map_err(|e| format!("{}: cannot serialise a record: {e}", path.display()))?;
        buf.extend_from_slice(line.as_bytes());
        buf.push(b'\n');
    }
    write_atomic(path, &buf)
}

fn write_json(path: &Path, text: &str) -> Result<(), String> {
    write_atomic(path, text.as_bytes())
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, String> {
    let text = read_text(path)?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<T>(line).map_err(|e| {
            format!(
                "{} line {}: not a record I can read ({e}); the raw stream is kept so this \
                 can be checked by hand",
                path.display(),
                i + 1
            )
        })?;
        out.push(record);
    }
    Ok(out)
}

fn read_text(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The same atomic, owner-only write xencode's other private state uses.
///
/// This crate cannot depend on the crate holding that helper without inverting
/// the boundary the orchestrator is built on — agents knows vendors, core knows
/// tasks, and a worker capture must not drag the scheduler in with it — so the
/// procedure is repeated here rather than shared. It is the same procedure:
/// write to a sibling, flush it, rename it over the target, sync the directory.
///
/// `pub(crate)` so the envelope store ([`crate::envelope`]) writes through the
/// identical owner-only path instead of keeping a third copy of this procedure.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;

    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "capture".to_string()),
        std::process::id()
    ));

    {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| format!("{}: {e}", tmp.display()))?;
        }
        file.write_all(bytes)
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
        file.flush()
            .and_then(|_| file.sync_all())
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
    }

    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("{}: {e}", path.display())
    })?;
    #[cfg(unix)]
    if let Ok(dir) = std::fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_line_is_kept_because_it_is_evidence_about_the_stream() {
        // "foo\n\nbar\n" has a real blank line in the middle: the vendor printed
        // it, so the raw store must number it, not swallow it.
        let lines = raw_lines("foo\n\nbar\n");
        assert_eq!(
            lines,
            vec![
                RawLine {
                    line: 1,
                    text: "foo".into()
                },
                RawLine {
                    line: 2,
                    text: String::new()
                },
                RawLine {
                    line: 3,
                    text: "bar".into()
                },
            ]
        );
    }

    #[test]
    fn a_final_newline_is_a_line_ending_and_not_an_empty_line() {
        // "a\nb\n" is two lines, not three with a trailing blank. split('\\n')
        // yields the phantom empty tail; raw_lines must drop exactly that.
        let lines = raw_lines("a\nb\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1].text, "b");
    }

    #[test]
    fn clip_shortens_for_display_and_marks_that_it_did() {
        let long = "x".repeat(200);
        let clipped = clip(&long);
        assert!(clipped.chars().count() <= 121, "{clipped}");
        assert!(clipped.ends_with('…'), "{clipped}");
        // Short text passes through untouched.
        assert_eq!(clip("hello"), "hello");
    }

    #[test]
    fn a_synthesised_row_serialises_raw_line_as_absent() {
        // An event xencode invented has no raw line behind it; that must show up
        // in the stored JSON as a missing field, not a fabricated line number.
        let row = StoredEvent {
            seq: 1,
            raw_line: None,
            event: AgentEvent::Completed {
                origin: Origin::Synthesised,
                outcome: None,
            },
        };
        let json = serde_json::to_string(&row).unwrap();
        assert!(json.contains("\"raw_line\":null"), "{json}");
        let back: StoredEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back, row);
    }
}
