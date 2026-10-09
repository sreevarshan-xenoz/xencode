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

use std::collections::BTreeMap;
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
    let last = text.lines().count();
    for (number, raw) in text.lines().enumerate() {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let line: SessionLine = serde_json::from_str(raw).map_err(|e| {
            // A last line with no newline after it was being written when the
            // run stopped (QA-6). It is still refused — replaying part of a run
            // as the whole of it would mislead — but it is named for what it is.
            if number + 1 == last && !text.ends_with('\n') {
                format!(
                    "{origin} ends in a half-written line {}: the run stopped while it \
                     was being recorded, so the recording is incomplete and is not replayed",
                    number + 1
                )
            } else {
                format!(
                    "{origin} line {}: not a readable recording ({e})",
                    number + 1
                )
            }
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

/// Names a human gave to runs, so `--resume <name>` finds a run across
/// processes without anyone memorising a run id.
fn names_path(xencode_dir: &Path) -> PathBuf {
    sessions_dir(xencode_dir).join("names.json")
}

fn read_names(xencode_dir: &Path) -> BTreeMap<String, String> {
    std::fs::read_to_string(names_path(xencode_dir))
        .ok()
        .and_then(|text| serde_json::from_str::<BTreeMap<String, String>>(&text).ok())
        .unwrap_or_default()
}

/// Whether a session name is usable. `latest` is reserved for resolution, and
/// anything that is not a short token is refused rather than sanitised into a
/// different name than the caller typed.
pub fn valid_session_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name != "latest"
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Name a run. Overwriting an existing name is refused: silently repointing a
/// name the caller may have resumed yesterday is how context gets swapped.
pub fn name_session(xencode_dir: &Path, name: &str, run_id: &str) -> Result<(), String> {
    if !valid_session_name(name) {
        return Err(format!(
            "{name:?} is not a usable session name: short tokens of letters, digits,              `-` and `_`, anything but `latest`"
        ));
    }
    if read_session(xencode_dir, run_id).is_err() {
        return Err(format!(
            "no recording for run {run_id}, so there is nothing to name"
        ));
    }
    let mut names = read_names(xencode_dir);
    if let Some(current) = names.get(name) {
        if current != run_id {
            return Err(format!(
                "{name:?} already names run {current}; pick another name rather than                  repointing it"
            ));
        }
        return Ok(());
    }
    names.insert(name.to_string(), run_id.to_string());
    let path = names_path(xencode_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    let temp = path.with_extension("json.tmp");
    std::fs::write(
        &temp,
        serde_json::to_string_pretty(&names).unwrap_or_default(),
    )
    .map_err(|e| format!("could not write {}: {e}", temp.display()))?;
    std::fs::rename(&temp, &path)
        .map_err(|e| format!("could not replace {}: {e}", path.display()))?;
    Ok(())
}

/// Resolve a name, an id prefix, or `latest` to a full run id.
///
/// An id prefix matching two recordings is refused with both candidates,
/// because resuming the wrong run restores the wrong context and nothing
/// downstream can tell.
pub fn resolve_session(xencode_dir: &Path, name_or_id: &str) -> Result<String, String> {
    if name_or_id == "latest" {
        return list_session_ids(xencode_dir)
            .into_iter()
            .next()
            .ok_or_else(|| "no recordings yet, so there is nothing to resume".to_string());
    }
    if let Some(run_id) = read_names(xencode_dir).get(name_or_id) {
        if session_path(xencode_dir, run_id).is_file() {
            return Ok(run_id.clone());
        }
        return Err(format!(
            "{name_or_id:?} names run {run_id}, whose recording is gone"
        ));
    }
    let mut hits: Vec<String> = list_session_ids(xencode_dir)
        .into_iter()
        .filter(|id| *id == name_or_id || id.starts_with(name_or_id))
        .collect();
    hits.sort();
    hits.dedup();
    match hits.as_slice() {
        [] => Err(format!("no session named or starting with {name_or_id:?}")),
        [one] => Ok(one.clone()),
        several => Err(format!(
            "{name_or_id:?} matches more than one recording — {} — so it is refused              rather than resumed into the wrong context",
            several.join(", ")
        )),
    }
}

/// Everything a new process needs to pick up where a run left off.
#[derive(Debug, Clone)]
pub struct ResumeContext {
    /// The full run id that was resolved.
    pub run_id: String,
    /// Model, server, and tool root the run recorded.
    pub model: String,
    pub server: String,
    pub tool_root: String,
    /// How many model calls the recording holds.
    pub calls: usize,
    /// The messages of the first request: the conversation a replay starts
    /// from. `None` means the recording is broken, not empty.
    pub opening_messages: Option<Vec<serde_json::Value>>,
}

/// Load a session by name, id prefix, or `latest`, across processes.
pub fn resume_context(xencode_dir: &Path, name_or_id: &str) -> Result<ResumeContext, String> {
    let run_id = resolve_session(xencode_dir, name_or_id)?;
    let session = read_session(xencode_dir, &run_id)?;
    Ok(ResumeContext {
        run_id,
        model: session.run.model.clone(),
        server: session.run.server.clone(),
        tool_root: session.run.tool_root.clone(),
        calls: session.calls.len(),
        opening_messages: session.opening_messages(),
    })
}

/// A session's transcript as markdown, with secrets redacted when asked.
///
/// Redaction reuses the trace module's secret patterns — PEM blocks, keyed
/// values with secret names, bearer tokens, prefixed tokens — because a second
/// secret detector is a second opinion nobody asked for. What is *not*
/// redacted is everything else, deliberately: a transcript that hides tool
/// names or file paths along with the secrets is useless for sharing.
pub fn export_transcript(
    xencode_dir: &Path,
    name_or_id: &str,
    redacted: bool,
) -> Result<String, String> {
    let run_id = resolve_session(xencode_dir, name_or_id)?;
    let session = read_session(xencode_dir, &run_id)?;
    let mut out = format!(
        "# Session {}\n\n- model: {}\n- server: {}\n- tool root: {}\n- calls: {}\n",
        run_id,
        session.run.model,
        session.run.server,
        session.run.tool_root,
        session.calls.len()
    );
    for call in &session.calls {
        out.push_str(&format!(
            "\n## call {} — {} {} → {}\n\n",
            call.seq, call.method, call.path, call.status
        ));
        out.push_str("request:\n```json\n");
        out.push_str(&call.request_body);
        out.push_str("\n```\nresponse:\n```\n");
        out.push_str(
            call.response_body
                .lines()
                .take(40)
                .collect::<Vec<_>>()
                .join("\n")
                .as_str(),
        );
        out.push_str("\n```\n");
        if !call.tools.is_empty() {
            out.push_str("\ntools:\n");
            for tool in &call.tools {
                out.push_str(&format!("- {}: {}\n", tool.index, tool.name));
            }
        }
    }
    if redacted {
        out = crate::trace::redact_secrets(&out);
    }
    Ok(out)
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

    fn write_as(dir: &Path, id: &str, calls: &[RecordedCall]) {
        let mut run = run(id);
        run.run_id = id.to_string();
        let mut writer = SessionWriter::begin(dir, &run).unwrap();
        for c in calls {
            writer.record(c.clone()).unwrap();
        }
    }

    /// QA-6: a run killed while its last call was being written leaves a
    /// half line at the end of the recording. It is refused, and the refusal
    /// says the run was cut off rather than calling the file unreadable.
    #[test]
    fn a_recording_cut_off_mid_line_is_refused_as_incomplete() {
        let dir = temp_dir("torn");
        write(&dir, &[call(0, "first"), call(1, "second")]);
        let path = session_path(&dir, "r1");
        let whole = std::fs::read_to_string(&path).unwrap();
        let cut = &whole[..whole.len() - 12];
        std::fs::write(&path, cut).unwrap();

        let err = read_session(&dir, "r1").expect_err("a torn recording is not replayed");
        assert!(err.contains("half-written line"), "{err}");
        assert!(err.contains("incomplete"), "{err}");

        // A bad line that is not the last one keeps the old wording.
        let lines: Vec<&str> = whole.lines().collect();
        let broken = format!(
            "{}
{{not json
{}
",
            lines[0],
            lines[1..].join(
                "
"
            )
        );
        std::fs::write(&path, broken).unwrap();
        let err = read_session(&dir, "r1").unwrap_err();
        assert!(err.contains("not a readable recording"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_name_survives_across_processes() {
        // Written here, resolved cold from disk: the point is that no
        // in-memory handle is involved.
        let dir = temp_dir("name");
        write_as(&dir, "r1", &[call(0, "hello")]);
        name_session(&dir, "demo", "r1").unwrap();
        assert_eq!(resolve_session(&dir, "demo").unwrap(), "r1");
        let ctx = resume_context(&dir, "demo").unwrap();
        assert_eq!(ctx.run_id, "r1");
        assert_eq!(ctx.calls, 1);
        let messages = ctx.opening_messages.expect("a readable first request");
        assert!(serde_json::to_string(&messages).unwrap().contains("hello"));
    }

    #[test]
    fn a_name_is_not_silently_repointed() {
        let dir = temp_dir("repoint");
        write_as(&dir, "r1", &[call(0, "one")]);
        write_as(&dir, "r2", &[call(0, "two")]);
        name_session(&dir, "demo", "r1").unwrap();
        let err = name_session(&dir, "demo", "r2").unwrap_err();
        assert!(err.contains("already names"), "{err}");
        assert_eq!(resolve_session(&dir, "demo").unwrap(), "r1");
    }

    #[test]
    fn reserved_and_sloppy_names_are_refused_not_sanitised() {
        let dir = temp_dir("names");
        write_as(&dir, "r1", &[call(0, "one")]);
        for bad in [
            "",
            "latest",
            "has space",
            "semi;colon",
            "a/very/long/name/that/keeps/going/past/sixty/four/characters/yes",
        ] {
            assert!(
                name_session(&dir, bad, "r1").is_err(),
                "{bad:?} should have been refused"
            );
        }
        assert!(name_session(&dir, "good-name_2", "r1").is_ok());
        assert!(name_session(&dir, "ghost", "no-such-run").is_err());
    }

    #[test]
    fn an_ambiguous_prefix_is_a_question_not_a_guess() {
        let dir = temp_dir("ambig");
        write_as(&dir, "1700000001-aaaa1111", &[call(0, "one")]);
        write_as(&dir, "1700000001-aaaa2222", &[call(0, "two")]);
        let err = resolve_session(&dir, "1700000001-aaaa").unwrap_err();
        assert!(err.contains("more than one"), "{err}");
        assert!(
            err.contains("aaaa1111") && err.contains("aaaa2222"),
            "{err}"
        );
        // ...while an unambiguous prefix still resolves.
        assert_eq!(
            resolve_session(&dir, "1700000001-aaaa1111").unwrap(),
            "1700000001-aaaa1111"
        );
    }

    #[test]
    fn latest_means_the_newest_recording() {
        let dir = temp_dir("latest");
        if list_session_ids(&dir).is_empty() {
            assert!(resolve_session(&dir, "latest").is_err());
        }
        write_as(&dir, "1700000001-first000", &[call(0, "one")]);
        write_as(&dir, "1700000002-second00", &[call(0, "two")]);
        assert_eq!(
            resolve_session(&dir, "latest").unwrap(),
            "1700000002-second00"
        );
    }

    const SEEDED_SECRET: &str = "sk-FAKE-NOT-A-REAL-TEST-KEY";
    const SEEDED_GITHUB: &str = "ghp_FAKE_NOT_A_REAL_TEST_KEY";
    const SEEDED_PEM: &str =
        "-----BEGIN RSA PRIVATE KEY-----\nFAKEKEYBODYnotAReal\n-----END RSA PRIVATE KEY-----";

    fn leaky_call() -> RecordedCall {
        let mut c = call(0, "do the thing");
        c.request_body = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "do the thing"},
                {"role": "system", "content": format!("key is {SEEDED_SECRET}")}
            ]
        })
        .to_string();
        c.response_body = format!(
            "the token is {SEEDED_GITHUB} and the key block is\n{SEEDED_PEM}\nAuthorization: Bearer FAKEJwtNotARealToken.payload.sig\n"
        );
        c
    }

    #[test]
    fn export_redacts_secrets_but_keeps_the_shape() {
        let dir = temp_dir("export");
        write_as(&dir, "r1", &[leaky_call()]);
        let redacted = export_transcript(&dir, "r1", true).unwrap();
        for secret in [
            SEEDED_SECRET,
            SEEDED_GITHUB,
            "FAKEKEYBODYnotAReal",
            "FAKEJwtNotARealToken",
        ] {
            assert!(
                !redacted.contains(secret),
                "redacted export leaks {secret:?}:\n{redacted}"
            );
        }
        assert!(redacted.contains("[redacted"), "redaction must be visible");
        // The shape survives: seq, model, tool name, call count.
        assert!(redacted.contains("call 0"));
        assert!(redacted.contains("remote:test-model"));
        assert!(redacted.contains("read_file"));
    }

    #[test]
    fn the_redaction_test_is_not_vacuous() {
        // If the unredacted export did not contain the secrets either, the test
        // above would pass while proving nothing.
        let dir = temp_dir("export-raw");
        write_as(&dir, "r1", &[leaky_call()]);
        let raw = export_transcript(&dir, "r1", false).unwrap();
        assert!(
            raw.contains(SEEDED_SECRET),
            "the fixture must actually leak"
        );
        assert!(raw.contains(SEEDED_GITHUB));
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
