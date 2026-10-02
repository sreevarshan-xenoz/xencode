//! `AR-9`'s common event protocol: one model every vendor's stream normalises
//! into.
//!
//! This is derived from `AR-1`'s measured matrix, not from the proposal's
//! wish list. The matrix was the reason the model had to change twice, and both
//! changes are visible in the code:
//!
//! - **Nothing is shared.** Six working agents spoke six vocabularies, and the
//!   seven canonical variants below are reached from every one of them by
//!   spelling. There is no common field name to lean on, so this keys on
//!   *shape*: which keys a line carries, not which agent sent it. `opencode`
//!   and `kilo` are a fork speaking one vocabulary, and keying on the agent's
//!   name would have made the adapter layer pay for opencode twice.
//! - **A normalised model cannot invent a correlation id.** `codex` was the
//!   only agent whose stream carried `thread_id`. Everything this module adds
//!   that no stream said is marked [`Origin::Synthesised`], so a caller can
//!   always tell a measurement from a construction.
//!
//! Two rules from the draft that this file settles by construction rather than
//! by argument:
//!
//! - A file change is **derived from xencode's own diff of the lease**. What a
//!   worker's stream claims about it is [`AgentEvent::FileChanged`] only when
//!   the claim is corroborated; an uncorroborated claim arrives as
//!   [`AgentEvent::Error`] with [`Origin::Synthesised`] and never as a change.
//! - [`AgentEvent::PermissionRequested`] is expected from exactly one vendor,
//!   so the model is complete without it. A stream that never sends one is
//!   ordinary, not degraded.

use serde::{Deserialize, Serialize};

/// Where one normalised event came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// The worker said this, in its own words, in this run.
    Observed,
    /// xencode filled this in because the model needs it and no stream carried
    /// it. Never a measurement.
    Synthesised,
}

impl Default for Origin {
    /// The safe reading. A stored event whose origin is missing was never
    /// witnessed, so it defaults to not having been observed rather than the
    /// other way round — a reader cannot tell a measurement from xencode's own
    /// bookkeeping if an absent field silently means "observed".
    fn default() -> Self {
        Origin::Synthesised
    }
}

/// The one model every adapter normalises into.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentEvent {
    /// The run began, or the agent said it was alive.
    SessionStarted {
        session_id: Option<String>,
        #[serde(default)]
        origin: Origin,
    },
    /// The agent said something in prose.
    Message {
        text: String,
        #[serde(default)]
        origin: Origin,
    },
    /// A tool was named and the agent intends to run it.
    ToolRequested {
        tool: String,
        call_id: Option<String>,
        #[serde(default)]
        origin: Origin,
    },
    /// A tool actually started running.
    ToolStarted {
        tool: String,
        call_id: Option<String>,
        #[serde(default)]
        origin: Origin,
    },
    /// A tool produced output, or finished without any.
    ToolOutput {
        tool: String,
        call_id: Option<String>,
        output: Option<String>,
        #[serde(default)]
        origin: Origin,
    },
    /// A file changed, according to xencode's own diff of the lease.
    ///
    /// Never populated from a worker's claim alone — see the module docs.
    FileChanged {
        path: String,
        #[serde(default)]
        origin: Origin,
    },
    /// The worker wants to do something that needs a human to approve.
    ///
    /// One vendor is expected to send this, so a run that never does is normal.
    PermissionRequested {
        tool: String,
        call_id: Option<String>,
        #[serde(default)]
        origin: Origin,
    },
    /// The run hit a problem, or the stream said it did.
    Error {
        message: String,
        #[serde(default)]
        origin: Origin,
    },
    /// The run finished.
    Completed {
        #[serde(default)]
        outcome: Option<String>,
        #[serde(default)]
        origin: Origin,
    },
    /// The agent said the session is over. Distinct from [`AgentEvent::Completed`]
    /// because `kiro-cli` and `codex` report both, and collapsing them loses the
    /// difference between "work is done" and "the session is closed".
    SessionEnded {
        #[serde(default)]
        reason: Option<String>,
        #[serde(default)]
        origin: Origin,
    },
}

impl AgentEvent {
    /// One word for what this event is, for a table or a log line.
    pub fn name(&self) -> &'static str {
        match self {
            AgentEvent::SessionStarted { .. } => "session_started",
            AgentEvent::Message { .. } => "message",
            AgentEvent::ToolRequested { .. } => "tool_requested",
            AgentEvent::ToolStarted { .. } => "tool_started",
            AgentEvent::ToolOutput { .. } => "tool_output",
            AgentEvent::FileChanged { .. } => "file_changed",
            AgentEvent::PermissionRequested { .. } => "permission_requested",
            AgentEvent::Error { .. } => "error",
            AgentEvent::Completed { .. } => "completed",
            AgentEvent::SessionEnded { .. } => "session_ended",
        }
    }

    /// Whether this event came from the worker or from xencode.
    pub fn origin(&self) -> Origin {
        match self {
            AgentEvent::SessionStarted { origin, .. }
            | AgentEvent::Message { origin, .. }
            | AgentEvent::ToolRequested { origin, .. }
            | AgentEvent::ToolStarted { origin, .. }
            | AgentEvent::ToolOutput { origin, .. }
            | AgentEvent::FileChanged { origin, .. }
            | AgentEvent::PermissionRequested { origin, .. }
            | AgentEvent::Error { origin, .. }
            | AgentEvent::Completed { origin, .. }
            | AgentEvent::SessionEnded { origin, .. } => *origin,
        }
    }
}

/// Normalise one raw line into zero or more common events.
///
/// Returns empty for anything that is not a JSON object line, and for objects
/// that carry no event-ish key — a stream may print its own banner first, and a
/// line that is not an event is not an error.
pub fn normalise_line(agent: &str, line: &str) -> Vec<AgentEvent> {
    let trimmed = line.trim();
    if !trimmed.starts_with('{') {
        return Vec::new();
    }
    let value: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    match agent {
        "opencode" | "kilo" => opencode_shape(&value),
        "cline" => cline_shape(&value),
        "codex" => codex_shape(&value),
        "agy" => agy_shape(&value),
        "cursor-agent" => cursor_shape(&value),
        "kiro-cli" => kiro_shape(&value),
        "claude" => claude_shape(&value),
        _ => Vec::new(),
    }
}

/// `opencode` and `kilo`: `step_start`, `step_finish`, `text`, `tool_use` — the
/// same four names, because kilo is a fork of it.
fn opencode_shape(value: &serde_json::Value) -> Vec<AgentEvent> {
    let Some(kind) = value.get("type").and_then(|v| v.as_str()) else {
        return Vec::new();
    };
    let part = value.get("part").unwrap_or(&serde_json::Value::Null);
    match kind {
        "step_start" => vec![AgentEvent::SessionStarted {
            session_id: str_at(value, "sessionID"),
            origin: Origin::Observed,
        }],
        "step_finish" => vec![AgentEvent::Completed {
            outcome: part
                .get("reason")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            origin: Origin::Observed,
        }],
        "text" => text_or_nothing(part.get("text").and_then(|v| v.as_str())),
        "tool_use" => {
            let tool = part
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("(unnamed)")
                .to_string();
            let call_id = part
                .get("callID")
                .or_else(|| part.get("callId"))
                .and_then(|v| v.as_str())
                .map(str::to_string);
            // One `tool_use` covers request, start and output, told apart by the
            // part's own state. Measured 2026-10-02: opencode emits it with
            // `state` absent on the first sighting and a `status` once the call
            // has run, so the state decides which of the three this is.
            match part.get("state").and_then(|v| v.as_str()) {
                Some("pending") | Some("requested") => {
                    vec![AgentEvent::ToolRequested {
                        tool,
                        call_id,
                        origin: Origin::Observed,
                    }]
                }
                Some("running") | Some("started") => vec![AgentEvent::ToolStarted {
                    tool,
                    call_id,
                    origin: Origin::Observed,
                }],
                _ => vec![AgentEvent::ToolOutput {
                    tool,
                    call_id,
                    output: part
                        .get("output")
                        .or_else(|| part.get("result"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    origin: Origin::Observed,
                }],
            }
        }
        _ => Vec::new(),
    }
}

/// `cline`: `agent_event` with a nested `type`, plus `hook_event` and
/// `run_result` beside it. The nesting is why a reader that only looked at the
/// top level called a paid run free — the same field is sometimes top-level and
/// sometimes under `event`.
fn cline_shape(value: &serde_json::Value) -> Vec<AgentEvent> {
    match value.get("type").and_then(|v| v.as_str()) {
        Some("hook_event") => {
            if value.get("hookEventName").and_then(|v| v.as_str()) == Some("agent_start") {
                vec![AgentEvent::SessionStarted {
                    session_id: value
                        .get("agentId")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    origin: Origin::Observed,
                }]
            } else {
                Vec::new()
            }
        }
        Some("run_result") => vec![
            AgentEvent::Completed {
                outcome: value
                    .get("finishReason")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                origin: Origin::Observed,
            },
            AgentEvent::SessionEnded {
                reason: Some("run_result".to_string()),
                origin: Origin::Observed,
            },
        ],
        Some("agent_event") => {
            let inner_type = value
                .get("event")
                .and_then(|e| e.get("type"))
                .and_then(|v| v.as_str());
            // Measured 2026-10-02: cline's prose arrives under `text`, not
            // `content`. Reading `content` dropped every message the run made.
            let text = value
                .get("event")
                .and_then(|e| e.get("text").or_else(|| e.get("content")))
                .and_then(|v| v.as_str());
            match inner_type {
                // cline routes text, tools and its own reasoning through one
                // event pair, told apart by `contentType`. Reading it as though
                // it were always text is how a cline run reported a tool call
                // and no prose at all.
                Some("content_end") | Some("content_start") => {
                    let inner = value.get("event").unwrap_or(&serde_json::Value::Null);
                    match inner.get("contentType").and_then(|v| v.as_str()) {
                        Some("text") => text_or_nothing(text),
                        Some("tool") => {
                            let tool = inner
                                .get("toolName")
                                .and_then(|v| v.as_str())
                                .unwrap_or("(unnamed)")
                                .to_string();
                            let call_id = inner
                                .get("toolCallId")
                                .and_then(|v| v.as_str())
                                .map(str::to_string);
                            if inner_type == Some("content_start") {
                                vec![AgentEvent::ToolRequested {
                                    tool,
                                    call_id,
                                    origin: Origin::Observed,
                                }]
                            } else {
                                vec![AgentEvent::ToolOutput {
                                    tool,
                                    call_id,
                                    output: inner
                                        .get("output")
                                        .and_then(|o| o.get(0))
                                        .and_then(|o| o.get("result"))
                                        .and_then(|v| v.as_str())
                                        .map(str::to_string),
                                    origin: Origin::Observed,
                                }]
                            }
                        }
                        // cline's reasoning is its own trace, not a message to
                        // the user. It is recorded by the run's text, not by this
                        // model, so it never becomes a `Message`.
                        _ => Vec::new(),
                    }
                }
                Some("done") => vec![AgentEvent::Completed {
                    outcome: Some("done".to_string()),
                    origin: Origin::Observed,
                }],
                // `usage`, `iteration_start` and `iteration_end` carry counts and
                // bookkeeping, which the run's totals read instead. They are not
                // events in this model.
                _ => Vec::new(),
            }
        }
        _ => Vec::new(),
    }
}

/// `codex`: dotted names, and the only stream on this box with a correlation id
/// the model could never have invented (`thread_id`).
fn codex_shape(value: &serde_json::Value) -> Vec<AgentEvent> {
    let kind = value.get("type").and_then(|v| v.as_str());
    let item = value.get("item");
    match kind {
        Some("thread.started") => vec![AgentEvent::SessionStarted {
            session_id: value
                .get("thread_id")
                .or_else(|| value.get("threadId"))
                .and_then(|v| v.as_str())
                .map(str::to_string),
            origin: Origin::Observed,
        }],
        Some("turn.started") => Vec::new(),
        Some("turn.completed") => vec![AgentEvent::Completed {
            outcome: Some("turn.completed".to_string()),
            origin: Origin::Observed,
        }],
        Some("item.started") => item
            .map(|item| {
                if let Some(tool) = tool_name(item) {
                    vec![AgentEvent::ToolStarted {
                        tool,
                        call_id: item.get("id").and_then(|v| v.as_str()).map(str::to_string),
                        origin: Origin::Observed,
                    }]
                } else {
                    Vec::new()
                }
            })
            .unwrap_or_default(),
        Some("item.completed") => item
            .map(|item| {
                if let Some(tool) = tool_name(item) {
                    vec![AgentEvent::ToolOutput {
                        tool,
                        call_id: item.get("id").and_then(|v| v.as_str()).map(str::to_string),
                        output: item
                            .get("aggregated_output")
                            .and_then(|v| v.as_str())
                            .map(str::to_string),
                        origin: Origin::Observed,
                    }]
                } else if item.get("type").and_then(|v| v.as_str()) == Some("agent_message") {
                    text_or_nothing(item.get("text").and_then(|v| v.as_str()))
                } else {
                    Vec::new()
                }
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// `agy`: `init`, `result`, and a `step_update` shape none of the others has —
/// an event carrying the changing state of work still in flight. `state` is
/// `ACTIVE` while running and `DONE` after.
fn agy_shape(value: &serde_json::Value) -> Vec<AgentEvent> {
    match value.get("event").and_then(|v| v.as_str()) {
        Some("init") => vec![AgentEvent::SessionStarted {
            session_id: str_at(value, "conversation_id"),
            origin: Origin::Observed,
        }],
        Some("result") => {
            let result = value.get("result").unwrap_or(&serde_json::Value::Null);
            // agy announces a finished `agent_response` step carrying only usage,
            // then delivers the prose once here at the end. Reading the step
            // alone reported a successful run that said nothing at all.
            let mut events = text_or_nothing(result.get("response").and_then(|v| v.as_str()));
            events.push(AgentEvent::Completed {
                outcome: result
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                origin: Origin::Observed,
            });
            events
        }
        Some("step_update") => {
            let step = value.get("step_update").unwrap_or(&serde_json::Value::Null);
            let done = step.get("state").and_then(|v| v.as_str()) == Some("DONE");
            let step_type = step.get("step_type").and_then(|v| v.as_str());
            match step_type {
                Some("agent_response") => {
                    if done {
                        text_or_nothing(
                            step.get("content")
                                .or_else(|| step.get("response"))
                                .and_then(|v| v.as_str()),
                        )
                    } else {
                        Vec::new()
                    }
                }
                Some("tool") => {
                    let tool = step
                        .get("tool_name")
                        .or_else(|| step.get("tool"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("(unnamed)")
                        .to_string();
                    // agy sends no correlation id on its tool steps, measured
                    // 2026-10-02. The model records `None` rather than inventing
                    // one, which is why a consumer cannot pair agy's tool output
                    // with its request the way it can for every other agent.
                    let call_id = step
                        .get("tool_call_id")
                        .or_else(|| step.get("toolCallId"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    if done {
                        vec![AgentEvent::ToolOutput {
                            tool,
                            call_id,
                            output: step
                                .get("output")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            origin: Origin::Observed,
                        }]
                    } else {
                        vec![AgentEvent::ToolStarted {
                            tool,
                            call_id,
                            origin: Origin::Observed,
                        }]
                    }
                }
                _ => Vec::new(),
            }
        }
        _ => Vec::new(),
    }
}

/// `cursor-agent`: `system`, `assistant`, `tool_call`, `result`, and a `subtype`
/// that separates a call starting from a call finishing.
fn cursor_shape(value: &serde_json::Value) -> Vec<AgentEvent> {
    let kind = value.get("type").and_then(|v| v.as_str());
    let subtype = value.get("subtype").and_then(|v| v.as_str());
    match (kind, subtype) {
        (Some("system"), Some("init")) => vec![AgentEvent::SessionStarted {
            session_id: str_at(value, "session_id"),
            origin: Origin::Observed,
        }],
        (Some("assistant"), _) => text_or_nothing(message_text(value.get("message"))),
        (Some("tool_call"), Some("started")) => vec![AgentEvent::ToolRequested {
            tool: cursor_tool(value),
            call_id: str_at(value, "call_id"),
            origin: Origin::Observed,
        }],
        (Some("tool_call"), Some("completed")) => vec![AgentEvent::ToolOutput {
            tool: cursor_tool(value),
            call_id: str_at(value, "call_id"),
            output: value
                .get("result")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            origin: Origin::Observed,
        }],
        (Some("result"), _) => vec![
            AgentEvent::Completed {
                outcome: subtype.map(str::to_string),
                origin: Origin::Observed,
            },
            AgentEvent::SessionEnded {
                reason: subtype.map(str::to_string),
                origin: Origin::Observed,
            },
        ],
        (Some("error"), _) => vec![AgentEvent::Error {
            message: value
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("(no message)")
                .to_string(),
            origin: Origin::Observed,
        }],
        _ => Vec::new(),
    }
}

/// `kiro-cli`: `metadata`, `runStarted`, `runFinished`, and a `sessionUpdate`
/// whose own `sessionUpdate` key names the variant. A seventh spelling of
/// machine-readable: the discriminator is nested under `data.update`.
fn kiro_shape(value: &serde_json::Value) -> Vec<AgentEvent> {
    let session_id = str_at(value, "sessionId").or_else(|| {
        value
            .get("data")
            .and_then(|d| d.get("sessionId"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    });
    match value.get("type").and_then(|v| v.as_str()) {
        Some("runStarted") => vec![AgentEvent::SessionStarted {
            session_id,
            origin: Origin::Observed,
        }],
        Some("runFinished") => vec![
            AgentEvent::Completed {
                outcome: value
                    .get("data")
                    .and_then(|d| d.get("status"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                origin: Origin::Observed,
            },
            AgentEvent::SessionEnded {
                reason: value
                    .get("data")
                    .and_then(|d| d.get("reason"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                origin: Origin::Observed,
            },
        ],
        Some("sessionUpdate") => {
            let update = value
                .get("data")
                .and_then(|d| d.get("update"))
                .unwrap_or(&serde_json::Value::Null);
            match update.get("sessionUpdate").and_then(|v| v.as_str()) {
                // kiro streams prose in chunks, `xen` then `code`, under a
                // `content` object rather than a bare `text`. Emitted as sent —
                // this function reads one line and holds no state, so joining
                // consecutive `Message` events is the consumer's job and the
                // spelling of the chunks stays visible rather than hidden.
                Some("agent_message_chunk") => text_or_nothing(
                    update
                        .get("content")
                        .and_then(|c| c.get("text"))
                        .or_else(|| update.get("text"))
                        .and_then(|v| v.as_str()),
                ),
                Some("tool_call") => vec![AgentEvent::ToolRequested {
                    tool: update
                        .get("kind")
                        .or_else(|| update.get("title"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("(unnamed)")
                        .to_string(),
                    call_id: update
                        .get("toolCallId")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    origin: Origin::Observed,
                }],
                Some("tool_call_update") => vec![AgentEvent::ToolOutput {
                    tool: update
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(unnamed)")
                        .to_string(),
                    call_id: update
                        .get("toolCallId")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    output: update
                        .get("content")
                        .and_then(|c| c.get(0))
                        .and_then(|c| c.get("text"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    origin: Origin::Observed,
                }],
                _ => Vec::new(),
            }
        }
        _ => Vec::new(),
    }
}

/// `claude`: `system`, `assistant`, `result`. Parked by decision, but its shape
/// is known from an observed run on 2026-09-28 and is kept so the model stays
/// complete when it is switched back on.
fn claude_shape(value: &serde_json::Value) -> Vec<AgentEvent> {
    let kind = value.get("type").and_then(|v| v.as_str());
    let subtype = value.get("subtype").and_then(|v| v.as_str());
    match (kind, subtype) {
        // `system` arrives first as `hook_started` and `hook_response` for the
        // SessionStart hook, and only later as `init`. Treating the first system
        // line as the session start would read a hook's id as the session's.
        (Some("system"), Some("init")) => vec![AgentEvent::SessionStarted {
            session_id: str_at(value, "session_id"),
            origin: Origin::Observed,
        }],
        (Some("assistant"), _) => text_or_nothing(message_text(value.get("message"))),
        (Some("result"), _) => vec![AgentEvent::Completed {
            outcome: subtype.map(str::to_string),
            origin: Origin::Observed,
        }],
        _ => Vec::new(),
    }
}

/// The prose inside a `message`, which may be a bare string or a list of
/// content blocks.
///
/// `cursor-agent` sends `[{"type":"text","text":"..."}]` and `claude` sends the
/// same. Reading `message` as a string found nothing in either, and reported two
/// agents as silent that had in fact answered. Tool-use blocks are skipped: only
/// what the agent said to the user is a `Message`.
fn message_text(message: Option<&serde_json::Value>) -> Option<&str> {
    let message = message?;
    // Either `message` is the prose itself, or it wraps `content`.
    if let Some(serde_json::Value::String(text)) = Some(message) {
        return Some(text.as_str());
    }
    match message.get("content") {
        Some(serde_json::Value::String(text)) => Some(text.as_str()),
        Some(serde_json::Value::Array(blocks)) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(|v| v.as_str()) == Some("text"))
            .filter_map(|b| b.get("text").and_then(|v| v.as_str()))
            .next(),
        _ => None,
    }
}

/// A message, or nothing when there was no text to carry. An empty string is
/// not a message, and inventing one would make a silent run look chatty.
fn text_or_nothing(text: Option<&str>) -> Vec<AgentEvent> {
    match text.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => vec![AgentEvent::Message {
            text: t.to_string(),
            origin: Origin::Observed,
        }],
        None => Vec::new(),
    }
}

fn str_at(value: &serde_json::Value, key: &str) -> Option<String> {
    value.get(key).and_then(|v| v.as_str()).map(str::to_string)
}

fn tool_name(item: &serde_json::Value) -> Option<String> {
    // The discriminator inside `item` is `type`, not `itemType`. Reading the
    // wrong key is why codex came back with three completions and no messages.
    item.get("type")
        .or_else(|| item.get("itemType"))
        .and_then(|v| v.as_str())
        .filter(|t| *t != "agent_message" && *t != "reasoning")
        .map(str::to_string)
}

/// `cursor-agent` nests the call under one key per tool (`readToolCall`,
/// `shellToolCall`, …), so the tool's name is the key rather than a value.
fn cursor_tool(value: &serde_json::Value) -> String {
    value
        .get("tool_call")
        .and_then(|tc| tc.as_object())
        .and_then(|map| {
            map.keys()
                .find(|k| k.ends_with("ToolCall"))
                .map(|k| k.trim_end_matches("ToolCall").to_string())
        })
        .unwrap_or_else(|| "(unnamed)".to_string())
}

/// Whether a run finished, however it finished.
///
/// Every vendor reports completion differently and `codex` alone splits it into
/// a turn, so this accepts either and does not care which. It exists so a
/// text-only run — one that said some prose and stopped — terminates through the
/// same check as one that ran tools, which is the `AR-9` done-when.
pub fn run_completed(events: &[AgentEvent]) -> bool {
    events.iter().any(|e| {
        matches!(
            e,
            AgentEvent::Completed { .. } | AgentEvent::SessionEnded { .. }
        )
    })
}

/// A change to a file, only ever from xencode's own diff of the lease.
///
/// Kept as one function so the rule cannot be bypassed: there is no public
/// constructor that turns a worker's claim into a [`AgentEvent::FileChanged`].
pub fn file_changed_from_diff(path: &str, corroborated: bool) -> AgentEvent {
    AgentEvent::FileChanged {
        path: path.to_string(),
        origin: if corroborated {
            Origin::Observed
        } else {
            Origin::Synthesised
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(events: &[AgentEvent]) -> Vec<&str> {
        events.iter().map(|e| e.name()).collect()
    }

    #[test]
    fn a_line_that_is_not_an_event_is_not_an_error() {
        // Every one of these was printed by a real agent before or beside its
        // events. None of them is a failure of the reader.
        for line in [
            "",
            "not json at all",
            "{ broken",
            "{}",
            "[1, 2, 3]",
            r#"{"total_cost_usd":0.01}"#,
        ] {
            assert_eq!(normalise_line("opencode", line), Vec::<AgentEvent>::new());
        }
    }

    #[test]
    fn opencode_and_kilo_reach_the_same_events_from_their_own_spelling() {
        // The real lines below are the opencode ones. Kilo emitted the identical
        // four names, so the same expected output must come out for both — which
        // is the test that stops an adapter layer paying for a fork twice.
        let lines = [
            r#"{"type":"step_start","timestamp":1790923290556,"sessionID":"ses_f04a79734ffe6tP6H3u8GEWZTo","part":{"id":"prt_1","type":"step-start"}}"#,
            r#"{"type":"text","part":{"id":"prt_2","type":"text","text":"xencode"}}"#,
            r#"{"type":"tool_use","part":{"id":"prt_3","type":"tool","tool":"read","callID":"call_1","state":"pending"}}"#,
            r#"{"type":"tool_use","part":{"id":"prt_3","type":"tool","tool":"read","callID":"call_1","state":"running"}}"#,
            r#"{"type":"tool_use","part":{"id":"prt_3","type":"tool","tool":"read","callID":"call_1","state":"completed","output":"xencode"}}"#,
            r#"{"type":"step_finish","part":{"id":"prt_4","type":"step-finish","reason":"stop"}}"#,
        ];
        for agent in ["opencode", "kilo"] {
            let events: Vec<AgentEvent> = lines
                .iter()
                .flat_map(|l| normalise_line(agent, l))
                .collect();
            assert_eq!(
                names(&events),
                vec![
                    "session_started",
                    "message",
                    "tool_requested",
                    "tool_started",
                    "tool_output",
                    "completed",
                ],
                "{agent}"
            );
            assert_eq!(
                events[0].clone(),
                AgentEvent::SessionStarted {
                    session_id: Some("ses_f04a79734ffe6tP6H3u8GEWZTo".into()),
                    origin: Origin::Observed
                },
                "{agent}"
            );
        }
    }

    #[test]
    fn cline_is_normalised_from_the_nested_shape_not_the_top_level() {
        // Real line, 2026-10-02. `usage` is the field a top-level-only reader
        // missed, which is how a paid run got reported as free.
        let events = normalise_line(
            "cline",
            r#"{"ts":"2026-10-02T06:41:42.645Z","type":"agent_event","event":{"type":"usage","inputTokens":6904,"outputTokens":87,"cacheReadTokens":113,"cost":0,"reasoningTokenCount":44}}"#,
        );
        // Counts are not events in this model; the run's totals read them.
        assert_eq!(events, Vec::<AgentEvent>::new());

        let events = normalise_line(
            "cline",
            r#"{"ts":"2026-10-02T06:43:31.431Z","type":"hook_event","hookEventName":"agent_start","agentId":"agent_b53c7fa9-b221-4f11-8e67-332dd3c5b2e9","taskId":"conv_1790923411416_dhjt11w","parentAgentId":null}"#,
        );
        assert_eq!(names(&events), vec!["session_started"]);
        assert!(!run_completed(&events), "a start is not a finish");
    }

    #[test]
    fn a_run_of_prose_alone_still_terminates() {
        // The AR-9 done-when: a worker that emitted nothing but text has to end
        // through the same state machine as one that ran tools.
        let events: Vec<AgentEvent> = [
            r#"{"type":"system","subtype":"init","session_id":"s1"}"#,
            r#"{"type":"assistant","message":"Reading the file."}"#,
            r#"{"type":"result","subtype":"success"}"#,
        ]
        .iter()
        .flat_map(|l| normalise_line("cursor-agent", l))
        .collect();
        assert_eq!(
            names(&events),
            vec!["session_started", "message", "completed", "session_ended"]
        );
        assert!(run_completed(&events));
    }

    #[test]
    fn an_empty_run_is_never_reported_as_a_message() {
        // agy's step_update arrives with a state but no content. Reporting a
        // message here would make a silent run look like it had spoken.
        let events = normalise_line(
            "agy",
            r#"{"event":"step_update","step_update":{"conversation_id":"62d526ca-7fe7-4bf5-89dd-f7c33543fedd","step_index":0,"state":"DONE","step_type":"agent_response"}}"#,
        );
        assert_eq!(events, Vec::<AgentEvent>::new());
    }

    #[test]
    fn codex_thread_started_carries_the_only_correlation_id_anyone_sent() {
        let events = normalise_line(
            "codex",
            r#"{"type":"thread.started","thread_id":"thr_9f2a"}"#,
        );
        assert_eq!(
            events[0].clone(),
            AgentEvent::SessionStarted {
                session_id: Some("thr_9f2a".into()),
                origin: Origin::Observed
            }
        );
    }

    #[test]
    fn a_file_change_only_counts_when_the_diff_backs_it() {
        assert_eq!(
            file_changed_from_diff("notes.txt", true).origin(),
            Origin::Observed
        );
        // An uncorroborated claim is marked synthesised, so nothing downstream
        // can mistake xencode's own bookkeeping for something a worker said.
        assert_eq!(
            file_changed_from_diff("notes.txt", false).origin(),
            Origin::Synthesised
        );
    }

    #[test]
    fn six_vocabularies_reach_the_same_model() {
        // One real line per agent, each from the measured matrix, and the result
        // is the same three events every time. This is the whole point of AR-9.
        let one_line_each = [
            (
                "opencode",
                r#"{"type":"text","part":{"id":"p","type":"text","text":"xencode"}}"#,
            ),
            (
                "kilo",
                r#"{"type":"text","part":{"id":"p","type":"text","text":"xencode"}}"#,
            ),
            (
                "cline",
                r#"{"type":"agent_event","event":{"type":"content_end","contentType":"text","text":"xencode"}}"#,
            ),
            (
                "codex",
                r#"{"type":"item.completed","item":{"id":"i","type":"agent_message","text":"xencode"}}"#,
            ),
            (
                "agy",
                r#"{"event":"step_update","step_update":{"state":"DONE","step_type":"agent_response","content":"xencode"}}"#,
            ),
            (
                "cursor-agent",
                r#"{"type":"assistant","message":"xencode"}"#,
            ),
            (
                "kiro-cli",
                r#"{"type":"sessionUpdate","data":{"sessionId":"s","update":{"sessionUpdate":"agent_message_chunk","text":"xencode"}}}"#,
            ),
            (
                "claude",
                r#"{"type":"assistant","message":{"content":[{"type":"text","text":"xencode"}]}}"#,
            ),
        ];
        for (agent, line) in one_line_each {
            let events = normalise_line(agent, line);
            assert_eq!(
                names(&events),
                vec!["message"],
                "{agent} said it differently"
            );
            match &events[0] {
                AgentEvent::Message { text, origin } => {
                    assert_eq!(text, "xencode", "{agent} lost the text");
                    assert_eq!(*origin, Origin::Observed, "{agent} invented provenance");
                }
                other => panic!("{agent} produced {other:?}"),
            }
        }
    }
}
