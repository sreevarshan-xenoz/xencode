//! A whole agent turn, written down while it happened, so it can be run again.
//!
//! [`crate::trace`] records that a turn took 4.1 s and called two tools; that is
//! a summary, and a summary cannot be replayed. This file keeps the other half:
//! for every model call, the request as it was serialised, the response exactly
//! as the server sent it, what each tool the model asked for actually returned,
//! and the clock at the time. With those four things a run can be handed back to
//! the same code and made to happen again — see `xencode_tui_rs::replay` for the
//! reader end, which answers a replayed session from a real socket carrying the
//! recorded bytes rather than from a model that might answer differently now.
//!
//! A recording holds the entire prompt and every tool's full output, so it goes
//! under `.xencode/cache/sessions/` — the same ignored tree the turn trace and
//! the metrics log live in — and never into a source directory. Nothing here
//! writes one unless a caller asks for it: recording is opt-in per session, and
//! the `run_id` in the file name is what a later `xencode replay` is given.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The only recording format this code reads. A file that names another one is
/// refused rather than guessed at, because a misread field would replay as a
/// different conversation.
pub const SESSION_FORMAT: &str = "xencode-session/1";

/// One tool the model asked for, and what this machine answered with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedToolCall {
    /// Which call of the round this was, in the order the model asked.
    pub index: usize,
    /// The id the provider gave the call, which is how its result is handed
    /// back on the next request.
    #[serde(default)]
    pub id: String,
    pub name: String,
    /// The arguments verbatim, because a replay has to make the same call and
    /// not one that merely looks similar.
    pub arguments: serde_json::Value,
    /// `done`, `failed` or `refused` — the same three outcomes a turn's trace
    /// reports, so a recording and a summary of it cannot disagree.
    pub outcome: String,
    /// Everything the model was given as the tool's result.
    pub result: String,
}

impl RecordedToolCall {
    /// A digest of what the tool returned. The ledger a replay writes carries
    /// this instead of the output, so two recordings of the same session can be
    /// compared without printing a shell's whole stdout.
    pub fn result_digest(&self) -> String {
        digest_hex(&self.result)
    }
}

/// The SHA-256 of some text, as 64 lower-case hex characters.
fn digest_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let sum = hasher.finalize();
    sum.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One model call: what was sent, what came back, what the answer asked for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedCall {
    /// Counts up through the whole run, across rounds, in the order the
    /// requests were made.
    pub seq: u64,
    /// When the request went out, in milliseconds since the Unix epoch. A
    /// replay reads this rather than its own clock, which is the only way two
    /// replays of one run can be expected to produce the same bytes.
    pub ts_unix_ms: u64,
    /// How long the server took. Recorded, never re-measured.
    pub duration_ms: u64,
    pub method: String,
    pub path: String,
    /// The request body as it was serialised.
    pub request_body: String,
    pub status: u16,
    pub content_type: String,
    /// The response body exactly as it was received, `data:` lines and all.
    pub response_body: String,
    /// The calls this answer asked for and the results it was given. Empty for
    /// the answer that ended the run.
    #[serde(default)]
    pub tools: Vec<RecordedToolCall>,
}

/// The head of a recording: what run this was, and what produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedRun {
    /// Which recording format the file speaks, checked on the way back in.
    pub format: String,
    pub run_id: String,
    pub recorded_at_unix_ms: u64,
    /// The model id as it was selected, prefix and all.
    pub model: String,
    /// Where the recorded answers came from — the server's address, or a
    /// description of a capture. A recording that cannot say this is a
    /// fabrication with extra steps, so the format insists on it.
    pub server: String,
    /// The directory the tools ran in. Kept to tell a reader where the run
    /// happened; a replay is pointed at a tree of its own.
    pub tool_root: String,
    #[serde(default)]
    pub prompt_digest: Option<String>,
    /// Which version of the instruction files this run was had.
    pub prompt_version: String,
}

/// One line of a recording file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum SessionLine {
    Run(RecordedRun),
    Call(RecordedCall),
}

/// A recording read back whole.
#[derive(Debug, Clone)]
pub struct Session {
    pub run: RecordedRun,
    pub calls: Vec<RecordedCall>,
}

impl Session {
    /// The messages of the first request, which is everything the loop was
    /// told before the model answered: the conversation a replay has to start
    /// from. `None` when the recording's first request is not a body this code
    /// can read, which is a broken recording rather than an empty conversation.
    pub fn opening_messages(&self) -> Option<Vec<serde_json::Value>> {
        let body: serde_json::Value =
            serde_json::from_str(self.calls.first()?.request_body.as_str()).ok()?;
        body.get("messages")?.as_array().cloned()
    }

    /// The model id the first request carried — what a replay asks the
    /// recording for, without the route prefix that chose the server.
    pub fn requested_model(&self) -> Option<String> {
        let body: serde_json::Value =
            serde_json::from_str(self.calls.first()?.request_body.as_str()).ok()?;
        body.get("model")?.as_str().map(str::to_string)
    }
}

/// `.xencode/cache/sessions`, where recordings live.
pub fn sessions_dir(xencode_dir: &Path) -> PathBuf {
    xencode_dir.join("cache").join("sessions")
}

/// The file one run's recording is kept in.
pub fn session_path(xencode_dir: &Path, run_id: &str) -> PathBuf {
    sessions_dir(xencode_dir).join(format!("{run_id}.jsonl"))
}

/// An id for a run being recorded now: the second it started, then eight
/// characters that separate two runs of the same project that began together.
///
/// Deliberately not a random UUID — the id is typed into `xencode replay`, so
/// it should say when the run happened and be short enough to finish by hand.
pub fn new_run_id(seed: &str) -> String {
    let millis = crate::conversation::now_millis();
    let short = digest_hex(&format!("{millis}:{seed}"));
    format!("{}-{}", millis / 1000, &short[..8])
}

/// Append one line of a recording, creating the file and its directory.
///
/// Written as it goes rather than collected and written at the end, because the
/// run this describes can be killed: half a recording says which calls happened,
/// while nothing says that.
pub fn append_session_line(
    xencode_dir: &Path,
    run_id: &str,
    line: &SessionLine,
) -> std::io::Result<()> {
    let path = session_path(xencode_dir, run_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    let text = serde_json::to_string(line)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    writeln!(file, "{text}")
}

/// A run being recorded: the head line is already on disk, and each model call
/// is appended as it finishes, numbered from the first.
pub struct SessionWriter {
    xencode_dir: PathBuf,
    run_id: String,
    next_seq: u64,
}

impl SessionWriter {
    /// Start the file for a run. The caller keeps one of these for the length of
    /// the turn, so the numbering cannot restart mid-run.
    ///
    /// Any file already under this id is replaced rather than added to. A replay
    /// records itself under the id it is replaying, in a directory it may be told
    /// to reuse, and two runs spliced into one file would read back as one run
    /// that said everything twice.
    pub fn begin(xencode_dir: &Path, run: &RecordedRun) -> std::io::Result<SessionWriter> {
        let path = session_path(xencode_dir, &run.run_id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        let head = SessionLine::Run(run.clone());
        let text = serde_json::to_string(&head)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        writeln!(file, "{text}")?;
        Ok(SessionWriter {
            xencode_dir: xencode_dir.to_path_buf(),
            run_id: run.run_id.clone(),
            next_seq: 0,
        })
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Append one model call, giving it the next number in the run.
    pub fn record(&mut self, mut call: RecordedCall) -> std::io::Result<()> {
        call.seq = self.next_seq;
        self.next_seq += 1;
        append_session_line(&self.xencode_dir, &self.run_id, &SessionLine::Call(call))
    }
}

/// Read one recording back. A missing file, a line that is not JSON, a format
/// this code does not speak, or a recording with no calls of its own is an
/// error: each of them would otherwise replay as a run that did nothing.
pub fn read_session(xencode_dir: &Path, run_id: &str) -> Result<Session, String> {
    let path = session_path(xencode_dir, run_id);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read the recording at {}: {e}", path.display()))?;
    read_session_text(&text, &path.display().to_string())
}

fn read_session_text(text: &str, origin: &str) -> Result<Session, String> {
    let mut calls: Vec<RecordedCall> = Vec::new();
    let mut seen_run = None;
    for (number, raw) in text.lines().enumerate() {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let line: SessionLine = serde_json::from_str(raw).map_err(|e| {
            format!(
                "{origin} line {}: not a readable recording ({e})",
                number + 1
            )
        })?;
        match line {
            SessionLine::Run(run) => {
                if seen_run.is_some() {
                    return Err(format!("{origin} describes more than one run"));
                }
                seen_run = Some(run);
            }
            SessionLine::Call(call) => calls.push(call),
        }
    }
    let run = seen_run.ok_or_else(|| format!("{origin} has no line saying which run it is"))?;
    if run.format != SESSION_FORMAT {
        return Err(format!(
            "{} says format {:?}, this build reads {SESSION_FORMAT:?}",
            run.run_id, run.format
        ));
    }
    if calls.is_empty() {
        return Err(format!(
            "{} records no model call, so there is nothing to replay",
            run.run_id
        ));
    }
    Ok(Session { run, calls })
}

/// Every run id this project has a recording for, newest first.
pub fn list_session_ids(xencode_dir: &Path) -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(sessions_dir(xencode_dir))
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
                .filter_map(|path| {
                    path.file_stem()
                        .and_then(|stem| stem.to_str())
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default();
    // The id begins with the second the run started, so sorting as text is
    // sorting by when it happened.
    ids.sort_by(|a, b| b.cmp(a));
    ids
}

/// Turn a run id, or the part of one a person typed, into the whole id.
/// An exact match wins; otherwise one unambiguous prefix is enough, and two is
/// a question that gets asked back rather than a guess.
pub fn resolve_run_id(xencode_dir: &Path, given: &str) -> Result<String, String> {
    let ids = list_session_ids(xencode_dir);
    if ids.iter().any(|id| id == given) {
        return Ok(given.to_string());
    }
    let matches: Vec<&String> = ids.iter().filter(|id| id.starts_with(given)).collect();
    match matches.len() {
        1 => Ok(matches[0].clone()),
        0 => Err(format!(
            "no recording named {given} in {}; `xencode replay --list` shows what is here",
            sessions_dir(xencode_dir).display()
        )),
        count => {
            let mut names: Vec<&str> = matches.iter().map(|id| id.as_str()).collect();
            names.sort_unstable();
            Err(format!(
                "{count} recordings start with {given}: {}; say which one",
                names.join(", ")
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(id: &str) -> RecordedRun {
        RecordedRun {
            format: SESSION_FORMAT.to_string(),
            run_id: id.to_string(),
            recorded_at_unix_ms: 1_700_000_000_000,
            model: "remote:test-model".to_string(),
            server: "http://127.0.0.1:1/v1".to_string(),
            tool_root: "/tmp/seeded".to_string(),
            prompt_digest: Some("abcd".to_string()),
            prompt_version: "0123456789ab".to_string(),
        }
    }

    fn call(seq: u64, content: &str) -> RecordedCall {
        RecordedCall {
            seq,
            ts_unix_ms: 1_700_000_000_000 + seq,
            duration_ms: 12,
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            request_body: serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": content}]
            })
            .to_string(),
            status: 200,
            content_type: "text/event-stream".to_string(),
            response_body: format!("data: {seq}\n\ndata: [DONE]\n"),
            tools: vec![RecordedToolCall {
                index: 0,
                id: format!("call-{seq}"),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "notes.txt"}),
                outcome: "done".to_string(),
                result: content.to_string(),
            }],
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xencode-session-{}-{label}-{}",
            std::process::id(),
            crate::conversation::now_millis()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, calls: &[RecordedCall]) {
        let mut writer = SessionWriter::begin(dir, &run("r1")).unwrap();
        for call in calls {
            writer.record(call.clone()).unwrap();
        }
        assert_eq!(writer.run_id(), "r1");
    }

    #[test]
    fn starting_a_run_replaces_an_older_file_with_the_same_id() {
        let dir = temp_dir("restart");
        write(&dir, &[call(0, "first"), call(1, "second")]);
        write(&dir, &[call(0, "only")]);
        // Read back what is on disk, not what the writer remembered: the point is
        // that the earlier calls are gone rather than sitting under a second head.
        let session = read_session(&dir, "r1").expect("readable");
        assert_eq!(session.calls.len(), 1);
        assert_eq!(session.calls[0].tools[0].result, "only");
    }

    #[test]
    fn a_run_written_as_it_happens_comes_back_in_the_same_order() {
        let dir = temp_dir("round-trip");
        write(&dir, &[call(0, "first"), call(1, "second")]);
        let session = read_session(&dir, "r1").expect("readable");
        assert_eq!(session.run.model, "remote:test-model");
        assert_eq!(session.calls.len(), 2);
        assert_eq!(session.calls[1].seq, 1);
        assert_eq!(session.calls[0].tools[0].name, "read_file");
        // The tool's output is kept whole, because a replay has to hand the
        // model the same bytes it was given the first time.
        assert_eq!(session.calls[0].tools[0].result, "first");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_replay_starts_from_the_conversation_the_recording_was_made_in() {
        let dir = temp_dir("opening");
        write(&dir, &[call(0, "what is in the note")]);
        let session = read_session(&dir, "r1").unwrap();
        let messages = session.opening_messages().expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"], "what is in the note");
        assert_eq!(session.requested_model().as_deref(), Some("test-model"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_recording_of_nothing_replays_as_nothing_and_says_so() {
        let dir = temp_dir("empty");
        SessionWriter::begin(&dir, &run("r1")).unwrap();
        let error = read_session(&dir, "r1").unwrap_err();
        assert!(error.contains("records no model call"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_recording_from_another_format_is_refused_rather_than_guessed_at() {
        let dir = temp_dir("format");
        let mut head = run("r1");
        head.format = "xencode-session/2".to_string();
        SessionWriter::begin(&dir, &head).unwrap();
        let error = read_session(&dir, "r1").unwrap_err();
        assert!(error.contains("xencode-session/2"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_that_is_not_a_recording_is_not_replayed() {
        assert!(read_session_text("not json at all\n", "memory.bin").is_err());
        // A call with no run line above it cannot be attributed to a session.
        let line = serde_json::to_string(&SessionLine::Call(call(0, "x"))).unwrap();
        assert!(read_session_text(&line, "orphan.jsonl")
            .unwrap_err()
            .contains("no line saying which run"));
    }

    #[test]
    fn a_headless_run_is_numbered_and_can_be_found_by_the_part_typed() {
        let dir = temp_dir("listing");
        write(&dir, &[call(0, "x")]);
        std::fs::rename(session_path(&dir, "r1"), session_path(&dir, "1700-abc")).unwrap();
        let mut second = run("1700-def");
        second.recorded_at_unix_ms += 10;
        let mut other = SessionWriter::begin(&dir, &second).unwrap();
        other.record(call(1, "y")).unwrap();

        let ids = list_session_ids(&dir);
        assert_eq!(ids, vec!["1700-def".to_string(), "1700-abc".to_string()]);
        assert_eq!(resolve_run_id(&dir, "1700-abc").unwrap(), "1700-abc");
        assert!(resolve_run_id(&dir, "1700")
            .unwrap_err()
            .contains("2 recordings"));
        assert!(resolve_run_id(&dir, "nope")
            .unwrap_err()
            .contains("no recording named nope"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_id_says_when_the_run_started_and_is_not_a_uuid() {
        let id = new_run_id("what is in the note");
        let (seconds, tail) = id.split_once('-').expect("two parts");
        assert_eq!(seconds.len(), 10, "{id}");
        assert_eq!(tail.len(), 8, "{id}");
        assert_ne!(
            tail, "abcdefgh",
            "the tail has to separate two runs at once"
        );
    }

    #[test]
    fn a_tools_output_is_compared_by_digest_not_by_printing_it() {
        let tool = RecordedToolCall {
            index: 0,
            id: String::new(),
            name: "run_command".to_string(),
            arguments: serde_json::json!({}),
            outcome: "done".to_string(),
            result: "1161\n".to_string(),
        };
        assert_eq!(tool.result_digest().len(), 64);
        assert_ne!(tool.result_digest(), {
            let mut other = tool.clone();
            other.result = "1162\n".to_string();
            other.result_digest()
        });
    }
}
