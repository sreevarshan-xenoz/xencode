//! Running a recorded session again, from the bytes it was made of.
//!
//! [`crate::app`]'s agent loop normally asks a model what to do next. A replay
//! asks a recording instead: the responses a real server once sent are served
//! again over a loopback socket by [`xencode_providers_rs::playback::Playback`],
//! so the HTTP client, the frame reader, the tool-call accumulator, the
//! permission gate and the tools themselves all run for real, and only the far
//! end is remembered rather than generated. That is the difference between "this
//! is what the model said then" and "this is what a model says now", and it is
//! the only way a run can be checked for having been repeatable at all.
//!
//! Two things make the output of two replays comparable, and both are deliberate:
//!
//! - Every time in the ledger comes out of the recording. Nothing here asks the
//!   clock, because a replay that stamped its own `12:04:31` against an earlier
//!   one's `11:58:02` would differ by the fact of having been run twice, which
//!   says nothing about the run.
//! - The answers arrive over a socket in pieces, cut wherever the player likes,
//!   because whether the tool calls survive that is exactly what a replay of a
//!   recording should be able to show.
//!
//! What a replay writes is `tool_calls.jsonl` next to the recording's own copy of
//! what happened — the recording made during a replay is kept, so a disagreement
//! between then and now can be read rather than trusted.

use std::path::{Path, PathBuf};

use xencode_context_rs::{
    read_session, RecordedCall, RecordedRun, RecordedToolCall, Session, SessionWriter,
    SESSION_FORMAT,
};
use xencode_providers_rs::playback::{
    Capture, Cassette, Delivery, Interaction, RecordedResponse, RequestMatcher, CASSETTE_FORMAT,
};
use xencode_providers_rs::traffic::request_needle;
use xencode_providers_rs::{ChatMessage, MessageContent};

use crate::app::{agent_rounds, App, LoopSink};
use tokio::sync::mpsc;

/// What a caller asks for: which recording, where its tools may run, and where
/// the answer of the replay should go.
pub struct ReplayOptions {
    /// The run to replay, already resolved to a whole id.
    pub run_id: String,
    /// The project's `.xencode` directory, where the recording lives.
    pub xencode_dir: PathBuf,
    /// The tree the replay's tool calls run against. Nothing here seeds it; a
    /// replay of a session that edited files expects those files to be there.
    pub tool_root: PathBuf,
    /// Where `tool_calls.jsonl` and the replay's own recording are written.
    pub out_dir: PathBuf,
    /// Whether the replay may do what the recorded session did. Off by default,
    /// and off means the permission gate stays in charge: a call it would have
    /// asked a person about comes back denied, exactly as it would in a
    /// headless run, and the replay then says where it stopped matching.
    pub run_tools: bool,
}

/// What a replay is worth reporting.
#[derive(Debug, Default)]
pub struct ReplayReport {
    pub run_id: String,
    /// Model calls the recording holds, and how many the replay got answered.
    pub calls_recorded: usize,
    pub calls_replayed: usize,
    /// Requests the recording could not answer — a request that was not the one
    /// recorded, in other words.
    pub unanswered: usize,
    pub tool_calls_recorded: usize,
    pub tool_calls_replayed: usize,
    /// How each recorded outcome fared in the replay, counted: `done=1` says one
    /// call the recording ended with `done` ended the same way again, and a
    /// `denied` row shows the replay stopped at the permission gate.
    pub outcomes: std::collections::BTreeMap<String, usize>,
    pub ledger: PathBuf,
    /// How the replay's own copy of the answers compares with the recording.
    pub answers_identical: bool,
}

impl ReplayReport {
    /// Whether the replay reproduced the recorded session end to end.
    pub fn matches(&self) -> bool {
        self.unanswered == 0
            && self.calls_replayed == self.calls_recorded
            && self.tool_calls_replayed == self.tool_calls_recorded
            && self.answers_identical
    }

    /// The report in the words a person should read, one line each.
    pub fn lines(&self) -> Vec<String> {
        let mut out = vec![format!(
            "recording {}: {} model {}, {} tool {}",
            self.run_id,
            self.calls_recorded,
            if self.calls_recorded == 1 {
                "call"
            } else {
                "calls"
            },
            self.tool_calls_recorded,
            if self.tool_calls_recorded == 1 {
                "call"
            } else {
                "calls"
            }
        )];
        let unanswered = if self.unanswered == 1 {
            "1 request".to_string()
        } else {
            format!("{} requests", self.unanswered)
        };
        out.push(format!(
            "replay: {} of {} model calls answered, {} the recording could not answer",
            self.calls_replayed, self.calls_recorded, unanswered
        ));
        out.push(format!(
            "tools: {} of {} recorded calls asked again{}",
            self.tool_calls_replayed,
            self.tool_calls_recorded,
            if self.outcomes.is_empty() {
                String::new()
            } else {
                let outcomes: Vec<&String> = self.outcomes.keys().collect();
                format!(
                    " (outcomes: {})",
                    outcomes
                        .iter()
                        .map(|outcome| format!("{outcome}={}", self.outcomes[*outcome]))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        ));
        out.push(if self.answers_identical {
            "answers: the same bytes the recording holds, reassembled from a socket".to_string()
        } else {
            "answers: at least one call did not come back as it was recorded".to_string()
        });
        out.push(
            "clock: every time in the ledger is the recorded one; nothing here reads the clock, \
             which is why two replays of one run can be compared at all"
                .to_string(),
        );
        out.push(format!("ledger: {}", self.ledger.display()));
        out
    }
}

/// Replay one recorded session.
///
/// The model is never asked anything: the request goes out over a loopback
/// socket to a player holding the bytes the real server sent. A session that
/// ended without any tool calls still replays — what is being checked is that
/// the same conversation produces the same calls in the same order.
pub async fn replay(options: &ReplayOptions) -> Result<ReplayReport, String> {
    // A replay keeps its own copy of what it saw, written the same way as the
    // recording it is reading. Sending that to the same directory would replace
    // the only copy of what actually happened.
    if same_path(
        &xencode_context_rs::sessions_dir(&options.out_dir),
        &xencode_context_rs::sessions_dir(&options.xencode_dir),
    ) {
        return Err(format!(
            "the replay's output directory is where the recording lives ({}); choose another",
            xencode_context_rs::sessions_dir(&options.xencode_dir).display()
        ));
    }
    let session = read_session(&options.xencode_dir, &options.run_id)?;
    let cassette = cassette_of(&session)?;
    let playback = cassette
        .play_with(Delivery::SplitMidLine)
        .map_err(|e| format!("the replay server could not start: {e}"))?;

    std::fs::create_dir_all(&options.out_dir)
        .map_err(|e| format!("cannot write to {}: {e}", options.out_dir.display()))?;
    let recorder = SessionWriter::begin(&options.out_dir, &replay_head(&session.run, options))
        .map_err(|e| format!("the replay could not keep its own recording: {e}"))?;

    let mut app = App::for_tests();
    if options.run_tools {
        // Said out loud, because it is the one place this program runs a tool
        // without a person approving it, and it only happens because a caller
        // asked for a recorded session to be lived through again.
        app.config.agent_approval = "all-allow".to_string();
    }
    let opening = session
        .opening_messages()
        .ok_or_else(|| format!("{} records a request this code cannot read", options.run_id))?;
    let model = format!("remote:{}", session.requested_model().unwrap_or_default());
    let mut run = app.agent_run(LoopSink::Chat, opening_messages(&opening), &model);
    run.model = endpoint_model(&session, &mut run, playback.base_url());
    run.tool_root = options.tool_root.clone();
    run.trace_dir = options.out_dir.clone();
    run.prompt_digest = session.run.prompt_digest.clone();
    run.session = Some(recorder);
    if !options.run_tools {
        // Nobody is there to answer a prompt, so a prompt is a refusal. Dropping
        // the receiver is how the loop already treats a closed app.
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        run.approval.prompts = tx;
    }

    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    agent_rounds(run, tx).await;
    while rx.try_recv().is_ok() {}

    let replayed = read_session(&options.out_dir, &options.run_id).unwrap_or_else(|_| Session {
        run: session.run.clone(),
        calls: Vec::new(),
    });
    let ledger = write_ledger(&options.out_dir, &session, &replayed)?;

    let unanswered = playback.misses();
    Ok(ReplayReport {
        run_id: options.run_id.clone(),
        calls_recorded: session.calls.len(),
        calls_replayed: replayed.calls.len(),
        unanswered,
        tool_calls_recorded: count_tools(&session),
        tool_calls_replayed: count_tools(&replayed),
        outcomes: outcome_tally(&replayed),
        answers_identical: answers_match(&session, &replayed),
        ledger,
    })
}

/// The head line for the recording a replay makes of itself. It carries the
/// original run's id and clock, because everything in it is a copy: a replay
/// that stamped its own moment could never be compared with another.
fn replay_head(original: &RecordedRun, options: &ReplayOptions) -> RecordedRun {
    RecordedRun {
        format: SESSION_FORMAT.to_string(),
        run_id: original.run_id.clone(),
        recorded_at_unix_ms: original.recorded_at_unix_ms,
        model: original.model.clone(),
        server: format!(
            "replay of {} over a loopback port; tools ran in {}",
            original.run_id,
            options.tool_root.display()
        ),
        tool_root: options.tool_root.to_string_lossy().into_owned(),
        prompt_digest: original.prompt_digest.clone(),
        prompt_version: xencode_context_rs::prompts::set_version().to_string(),
    }
}

/// Whether two paths name the same place. A directory that does not exist yet
/// cannot be resolved, so those are compared as written.
fn same_path(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(first), Ok(second)) => first == second,
        _ => a == b,
    }
}

/// The recording, in the format the player serves: one interaction per model
/// call, each keeping the request it was made with so a replay that asks
/// something else is refused instead of being answered with an unrelated answer.
fn cassette_of(session: &Session) -> Result<Cassette, String> {
    let interactions = session
        .calls
        .iter()
        .map(|call| Interaction {
            request: RequestMatcher {
                method: call.method.clone(),
                path: call.path.clone(),
                body_contains: request_needle(&call.request_body).into_iter().collect(),
                body: Some(call.request_body.clone()),
            },
            response: RecordedResponse {
                status: call.status,
                content_type: call.content_type.clone(),
                body: call.response_body.clone(),
            },
        })
        .collect();
    let cassette = Cassette {
        format: CASSETTE_FORMAT.to_string(),
        captured: Capture {
            at_unix_ms: session.run.recorded_at_unix_ms,
            server: session.run.server.clone(),
            model: session.run.model.clone(),
            tool: "xencode session recording".to_string(),
            note: format!(
                "run {}, prompt set {}, tools ran in {}",
                session.run.run_id, session.run.prompt_version, session.run.tool_root
            ),
        },
        interactions,
    };
    cassette
        .validate()
        .map_err(|e| format!("the recording cannot be served again: {e}"))?;
    Ok(cassette)
}

/// Point the loop at the player instead of at a model, in whichever shape the
/// recording was made in: an OpenAI-compatible endpoint keeps its `/v1` prefix,
/// an Ollama one its host root. Returns the model id to select the route with.
fn endpoint_model(session: &Session, run: &mut crate::app::AgentRun, base_url: String) -> String {
    let recorded_model = session.requested_model().unwrap_or_default();
    let path = &session.calls[0].path;
    if let Some(prefix) = path.strip_suffix("/chat/completions") {
        run.remote_base_url = format!("{base_url}{prefix}");
        run.remote_api_key = None;
        return format!("remote:{recorded_model}");
    }
    // The Ollama route: its own reader, its own path, no `remote:` prefix.
    run.ollama_url = base_url;
    recorded_model
}

/// The conversation as the loop wants it: the messages of the first recorded
/// request, back into the shape the assembler had put them in.
fn opening_messages(values: &[serde_json::Value]) -> Vec<ChatMessage> {
    values
        .iter()
        .filter_map(|value| {
            let role = value.get("role")?.as_str()?.to_string();
            let content = match value.get("content")? {
                serde_json::Value::String(text) => MessageContent::Text(text.clone()),
                serde_json::Value::Array(parts) => MessageContent::Parts(
                    parts
                        .iter()
                        .filter_map(|part| {
                            if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                                return part.get("text")?.as_str().map(|text| {
                                    xencode_providers_rs::ContentPart::Text {
                                        text: text.to_string(),
                                    }
                                });
                            }
                            let image = part.get("image_url")?;
                            Some(xencode_providers_rs::ContentPart::ImageUrl {
                                image_url: xencode_providers_rs::ImageUrlPart {
                                    url: image.get("url")?.as_str()?.to_string(),
                                    detail: image
                                        .get("detail")
                                        .and_then(|d| d.as_str())
                                        .map(str::to_string),
                                },
                            })
                        })
                        .collect(),
                ),
                other => MessageContent::Text(other.to_string()),
            };
            Some(ChatMessage { role, content })
        })
        .collect()
}

fn count_tools(session: &Session) -> usize {
    session.calls.iter().map(|call| call.tools.len()).sum()
}

/// How the recorded outcomes fared, counted in a stable order.
fn outcome_tally(session: &Session) -> std::collections::BTreeMap<String, usize> {
    let mut tally = std::collections::BTreeMap::new();
    for call in &session.calls {
        for tool in &call.tools {
            *tally.entry(tool.outcome.clone()).or_insert(0) += 1;
        }
    }
    tally
}

/// Whether every answer the replay received is the answer that was recorded —
/// compared as bytes, because the point of a replay is that the same request
/// brings back the same response.
fn answers_match(recorded: &Session, replayed: &Session) -> bool {
    recorded.calls.len() == replayed.calls.len()
        && recorded
            .calls
            .iter()
            .zip(replayed.calls.iter())
            .all(|(was, now)| was.response_body == now.response_body)
}

/// Write what the replay's tools did, one JSON line each, in the order the model
/// asked for them.
fn write_ledger(out_dir: &Path, recorded: &Session, replayed: &Session) -> Result<PathBuf, String> {
    let path = out_dir.join("tool_calls.jsonl");
    let mut text = String::new();
    for call in &replayed.calls {
        for tool in &call.tools {
            text.push_str(&ledger_line(
                &replayed.run,
                call,
                timed_by(recorded, call),
                tool,
            ));
            text.push('\n');
        }
    }
    // A replay that stopped partway still lists the calls it did not reach, so
    // the file cannot be read as a shorter session than the one recorded.
    if replayed.calls.len() < recorded.calls.len() {
        for call in recorded.calls.iter().skip(replayed.calls.len()) {
            for tool in &call.tools {
                text.push_str(
                    &ledger_line(
                        &recorded.run,
                        call,
                        (call.ts_unix_ms, call.duration_ms),
                        tool,
                    )
                    .replace("\"replayed\":true", "\"replayed\":false"),
                );
                text.push('\n');
            }
        }
    }
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

/// The call whose moment a replayed line reports: the recorded call it stands
/// for, matched by its number in the run. A replay happens later than the run it
/// replays, so its own clock would put a different time in every copy of the
/// same file — which is the difference between a replay that can be compared and
/// one that cannot. Where the recording holds no such call, the replay went
/// somewhere the recording did not, and its own moment is the only true one.
fn timed_by(recorded: &Session, replayed: &RecordedCall) -> (u64, u64) {
    match recorded.calls.iter().find(|call| call.seq == replayed.seq) {
        Some(was) => (was.ts_unix_ms, was.duration_ms),
        None => (replayed.ts_unix_ms, replayed.duration_ms),
    }
}

/// One line of the ledger. Field order is fixed here because the file is
/// compared byte for byte: a map that changed its own ordering would make two
/// identical replays look different.
fn ledger_line(
    run: &RecordedRun,
    call: &RecordedCall,
    timed: (u64, u64),
    tool: &RecordedToolCall,
) -> String {
    let line = serde_json::json!({
        "replayed": true,
        "run_id": run.run_id,
        "call": call.seq,
        "index": tool.index,
        "name": tool.name,
        "arguments": tool.arguments,
        "outcome": tool.outcome,
        "result_sha256": tool.result_digest(),
        "result_chars": tool.result.chars().count(),
        "model": run.model,
        "prompt_version": run.prompt_version,
        "recorded_ts_unix_ms": timed.0,
        "recorded_duration_ms": timed.1,
    });
    line.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes a model server puts on the wire for one tool call: one SSE
    /// frame whose message carries the call, with the arguments as a JSON string.
    fn sse_tool_call(id: &str, name: &str, arguments: serde_json::Value) -> String {
        let call = serde_json::json!({
            "id": id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments.to_string(),
            },
        });
        let body = serde_json::json!({
            "choices": [{"message": {"content": "", "tool_calls": [call]}}],
        });
        format!("data: {body}\n\n")
    }

    fn recorded() -> Session {
        let run = RecordedRun {
            format: SESSION_FORMAT.to_string(),
            run_id: "1700000000-abcd1234".to_string(),
            recorded_at_unix_ms: 1_700_000_000_000,
            model: "remote:dolphin".to_string(),
            server: "http://127.0.0.1:8080/v1".to_string(),
            tool_root: "/tmp/seeded".to_string(),
            prompt_digest: Some("9d4a".to_string()),
            prompt_version: "8abca0eb4098".to_string(),
        };
        let mut calls = vec![RecordedCall {
            seq: 0,
            ts_unix_ms: 1_700_000_000_000,
            duration_ms: 940,
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            request_body: serde_json::json!({
                "model": "dolphin",
                "messages": [{"role": "user", "content": "what is 27 * 43?"}]
            })
            .to_string(),
            status: 200,
            content_type: "text/event-stream".to_string(),
            response_body: sse_tool_call(
                "call-0",
                "run_command",
                serde_json::json!({"command": "echo $((27 * 43))"}),
            ),
            tools: vec![RecordedToolCall {
                index: 0,
                id: "call-0".to_string(),
                name: "run_command".to_string(),
                arguments: serde_json::json!({"command": "echo $((27 * 43))"}),
                outcome: "done".to_string(),
                result: "1161\n".to_string(),
            }],
        }];
        calls.push(RecordedCall {
            seq: 1,
            duration_ms: 210,
            request_body: serde_json::json!({
                "model": "dolphin",
                "messages": [{"role": "tool", "content": "1161\n"}]
            })
            .to_string(),
            response_body: sse_tool_call(
                "call-1",
                "write_note",
                serde_json::json!({"text": "1161"}),
            ),
            tools: vec![RecordedToolCall {
                index: 0,
                id: "call-1".to_string(),
                name: "write_note".to_string(),
                arguments: serde_json::json!({"text": "1161"}),
                outcome: "done".to_string(),
                result: "written\n".to_string(),
            }],
            ..calls[0].clone()
        });
        Session { run, calls }
    }

    #[test]
    fn a_cassette_is_built_from_a_recording_with_its_own_request_as_the_match() {
        let cassette = cassette_of(&recorded()).expect("cassette");
        assert_eq!(cassette.interactions.len(), 2);
        assert_eq!(
            cassette.interactions[0].request.path,
            "/v1/chat/completions"
        );
        assert_eq!(
            cassette.interactions[1].request.body_contains,
            vec!["1161\\n".to_string()],
            "the matcher is the tool's own output, which a replay can only get \
             right by running the tool"
        );
        assert!(cassette.captured.note.contains("1700000000-abcd1234"));
    }

    #[test]
    fn an_ollama_recording_points_the_replay_at_the_same_host_root() {
        let mut session = recorded();
        session.calls[0].path = "/api/chat".to_string();
        session.calls[0].request_body = serde_json::json!({
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "hi"}]
        })
        .to_string();
        let app = App::for_tests();
        let mut run = app.agent_run(LoopSink::Chat, Vec::new(), "hi");
        let model = endpoint_model(&session, &mut run, "http://127.0.0.1:99".to_string());
        assert_eq!(model, "qwen2.5:7b");
        assert_eq!(run.ollama_url, "http://127.0.0.1:99");

        let mut run = app.agent_run(LoopSink::Chat, Vec::new(), "hi");
        let model = endpoint_model(&recorded(), &mut run, "http://127.0.0.1:99".to_string());
        assert_eq!(model, "remote:dolphin");
        assert_eq!(run.remote_base_url, "http://127.0.0.1:99/v1");
    }

    #[test]
    fn the_replay_of_a_session_keeps_the_clock_it_was_recorded_with() {
        let head = replay_head(
            &recorded().run,
            &ReplayOptions {
                run_id: "x".to_string(),
                xencode_dir: PathBuf::new(),
                tool_root: PathBuf::from("/tmp/seeded"),
                out_dir: PathBuf::new(),
                run_tools: false,
            },
        );
        assert_eq!(head.run_id, "1700000000-abcd1234");
        assert_eq!(
            head.recorded_at_unix_ms, 1_700_000_000_000,
            "a replay stamps the original moment, not its own"
        );
        assert!(head.server.starts_with("replay of "), "{}", head.server);
    }

    #[test]
    fn the_ledger_names_the_call_the_arguments_and_the_output_by_digest() {
        let session = recorded();
        let line = serde_json::from_str::<serde_json::Value>(&ledger_line(
            &session.run,
            &session.calls[0],
            (session.calls[0].ts_unix_ms, session.calls[0].duration_ms),
            &session.calls[0].tools[0],
        ))
        .unwrap();
        assert_eq!(line["name"], "run_command");
        assert_eq!(line["arguments"]["command"], "echo $((27 * 43))");
        assert_eq!(line["outcome"], "done");
        assert_eq!(line["call"], 0);
        assert_eq!(line["recorded_duration_ms"], 940);
        assert_eq!(line["result_chars"], 5);
        let digest = line["result_sha256"]
            .as_str()
            .expect("a digest")
            .to_string();
        assert_eq!(digest.len(), 64, "{digest}");
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(
            digest,
            serde_json::from_str::<serde_json::Value>(&ledger_line(
                &session.run,
                &session.calls[0],
                (session.calls[0].ts_unix_ms, session.calls[0].duration_ms),
                &RecordedToolCall {
                    result: "1162\n".to_string(),
                    ..session.calls[0].tools[0].clone()
                }
            ))
            .unwrap()["result_sha256"]
                .as_str()
                .unwrap()
                .to_string(),
            "a different answer from the shell has to show up"
        );
    }

    #[test]
    fn the_report_counts_outcomes_and_says_what_the_recording_could_not_answer() {
        let session = recorded();
        let mut tally = outcome_tally(&session);
        assert_eq!(tally.insert("denied".to_string(), 1), None);
        let report = ReplayReport {
            run_id: session.run.run_id.clone(),
            calls_recorded: session.calls.len(),
            calls_replayed: 1,
            unanswered: 1,
            tool_calls_recorded: count_tools(&session),
            tool_calls_replayed: 1,
            outcomes: tally,
            ledger: PathBuf::from("/tmp/tool_calls.jsonl"),
            answers_identical: false,
        };
        let lines = report.lines().join("\n");
        assert!(lines.contains("2 model calls, 2 tool calls"), "{lines}");
        assert!(lines.contains("outcomes: denied=1, done=2"), "{lines}");
        assert!(
            lines.contains("1 request the recording could not answer"),
            "{lines}"
        );
        assert!(
            lines.contains("did not come back as it was recorded"),
            "{lines}"
        );
        assert!(!report.matches());
        let report = ReplayReport {
            unanswered: 2,
            ..report
        };
        assert!(
            report
                .lines()
                .join("\n")
                .contains("2 requests the recording could not answer"),
            "a count of two still reads as one request: {}",
            report.lines().join(" / ")
        );
    }

    #[test]
    fn a_replayed_line_keeps_the_recorded_moment_rather_than_the_replays_own() {
        let was = recorded();
        let mut later = was.clone();
        for call in &mut later.calls {
            call.ts_unix_ms += 60_000;
            call.duration_ms = 12;
        }
        assert_eq!(
            timed_by(&was, &later.calls[0]),
            (was.calls[0].ts_unix_ms, was.calls[0].duration_ms),
            "a replay that stamped its own clock could never be compared with another"
        );
        // A call the recording has no answer for keeps the time it really took.
        let mut unrecorded = later.calls[0].clone();
        unrecorded.seq = 7;
        assert_eq!(
            timed_by(&was, &unrecorded),
            (unrecorded.ts_unix_ms, unrecorded.duration_ms)
        );
    }

    #[test]
    fn identical_replays_are_reported_as_such_and_a_missing_call_is_not() {
        let session = recorded();
        assert!(answers_match(&session, &session));
        let mut short = session.clone();
        short.calls.pop();
        assert!(!answers_match(&session, &short));
        let mut changed = session.clone();
        changed.calls[0].response_body.push('!');
        assert!(!answers_match(&session, &changed));
    }

    #[tokio::test]
    async fn a_ledger_written_from_a_replay_that_stopped_says_which_lines_it_did_not_reach() {
        let dir = std::env::temp_dir().join(format!(
            "xencode-replay-ledger-{}-{}",
            std::process::id(),
            xencode_context_rs::conversation::now_millis()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let session = recorded();
        let mut half = session.clone();
        half.calls.truncate(1);
        let path = write_ledger(&dir, &session, &half).expect("ledger");
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "one replayed, one not");
        assert!(lines[0].contains("\"replayed\":true"), "{}", lines[0]);
        assert!(lines[1].contains("\"replayed\":false"), "{}", lines[1]);
        assert!(text.ends_with('\n'));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A scratch directory that cannot collide with another test process.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xencode-{name}-{}-{}",
            std::process::id(),
            xencode_context_rs::conversation::now_millis()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Record a session from a model that is really answering, and print the
    /// recording so it can be committed under `tests/fixtures/sessions/`.
    ///
    /// Ignored because it needs a server on this machine: start one with
    /// `llama-server -m <gguf> -c 8192 --port 8099 --host 127.0.0.1 --no-webui --jinja`,
    /// then run this test with `--ignored`. Override `XENCODE_CAPTURE_URL` and
    /// `XENCODE_CAPTURE_MODEL` for a different port or route. The point of doing
    /// it this way is that nothing here hand-writes a response: the bytes come
    /// out of the same recorder a user's run fills in, so a replay of the
    /// committed recording is testing the same path that produced it.
    #[tokio::test]
    #[ignore = "needs a local model server answering on 127.0.0.1"]
    async fn capture_a_real_session_for_the_fixture() {
        let url = std::env::var("XENCODE_CAPTURE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8099/v1".into());
        let model =
            std::env::var("XENCODE_CAPTURE_MODEL").unwrap_or_else(|_| "remote:dolphin".into());
        let prompt = std::env::var("XENCODE_CAPTURE_PROMPT").unwrap_or_else(|_| {
            "Use the run_command tool to work out what 27 multiplied by 43 is, \
             and then tell me the number."
                .to_string()
        });
        let dir = scratch("capture");
        let mut app = App::for_tests();
        app.config.default_model = model.clone();
        app.config.remote_base_url = url.clone();
        app.config.agent_approval = "all-allow".to_string();
        let head = RecordedRun {
            format: SESSION_FORMAT.to_string(),
            run_id: xencode_context_rs::new_run_id(&prompt),
            recorded_at_unix_ms: xencode_context_rs::conversation::now_millis(),
            model,
            server: url,
            tool_root: dir.to_string_lossy().into_owned(),
            prompt_digest: Some(xencode_context_rs::prompt_digest(&prompt)),
            prompt_version: xencode_context_rs::prompts::set_version().to_string(),
        };
        let writer = SessionWriter::begin(&dir, &head).expect("begin the recording");
        let opening = vec![ChatMessage {
            role: "user".to_string(),
            content: MessageContent::Text(prompt.to_string()),
        }];
        let mut run = app.agent_run(LoopSink::Chat, opening, &prompt);
        run.session = Some(writer);
        run.tool_root = dir.clone();
        run.trace_dir = dir.join("traces");
        let (tx, mut rx) = mpsc::unbounded_channel();
        agent_rounds(run, tx).await;
        while rx.try_recv().is_ok() {}

        let text = std::fs::read_to_string(xencode_context_rs::session_path(&dir, &head.run_id))
            .expect("the run left no recording behind");
        println!("{text}");
        let session = xencode_context_rs::read_session(&dir, &head.run_id).expect("readable");
        assert!(
            session.calls.len() >= 2,
            "one answer with no tool round does not exercise the replay: got {} call(s)",
            session.calls.len()
        );
        assert!(
            count_tools(&session) >= 1,
            "the recording holds no tool call, so the replay would check nothing"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
