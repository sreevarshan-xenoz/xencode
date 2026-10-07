//! AR-5: the envelope a normalised event travels in, and who said what.
//!
//! [`AR-4`]'s capture keeps a run sealed and verbatim. This module is the other
//! half: the shape an [`AgentEvent`] takes when it leaves the capture and goes
//! somewhere it can be queried, joined, or synced — `EVd-1`'s ledger, `CX-2`'s
//! metrics, the orchestrator's live state. That copy cannot be a byte-for-byte
//! mirror of the raw stream (it would carry the vendor's secrets onto every
//! surface the ledger touches) and cannot silently drop the fields the model
//! needs (a missing value that reads as a zero is the bug `AR-1` kept hitting).
//!
//! So the envelope fixes seven things and keeps them honest:
//!
//! - **session id, task id, agent id, worker id** — who this event is from and
//!   which job it belongs to, each in one of [`Field`]'s four states,
//! - **sequence** — xencode's own gapless position, never the vendor's,
//! - **timestamp** — the vendor's clock if it stamped the line, else unstated,
//!   never a fabricated `now`,
//! - **origin** and **payload** — the [`AgentEvent`] itself, carrying whether the
//!   worker said it or xencode did.
//!
//! # The four states are not interchangeable
//!
//! `unknown` and `unavailable` are both "no value", and collapsing them is the
//! exact mistake this item exists to stop:
//!
//! - [`Field::Observed`] — the worker said this, in this run.
//! - [`Field::Synthesised`] — xencode minted it because the model needs the slot
//!   and no stream carried it. The worker id and the sequence are always this.
//! - [`Field::Unknown`] — nobody said anything *on this run*. It is itself a
//!   recorded fact, and a later run from the same vendor may well say it.
//! - [`Field::Unavailable { reason }]` — this vendor **provably cannot** say it.
//!   `agy` emits no correlation id on a tool call (measured 2026-10-02), so its
//!   tool output can never be paired to its request from the stream: that is not
//!   a gap in this run, it is a property of the adapter.
//!
//! A verifier downstream must never read an absence as a zero or an inference as
//! a measurement, so nothing here has a numeric or string default that could be
//! mistaken for real data: an absent value is `None` *and* its state says why.
//!
//! # Redaction that keeps recoverability
//!
//! [`Envelope::redacted`] replaces anything shaped like a credential in the
//! payload and in every string field, using the same patterns the probe applies
//! to its own output. It does not touch [`Envelope::raw_line`], the session id,
//! or the sequence, so a redacted envelope still names the exact raw line in
//! [`AR-4`]'s `raw.jsonl` that holds the true bytes. Redaction therefore removes
//! the secret from the layer that propagates, while recovery stays a deliberate
//! act on the sealed capture — which is what `AR-4` deferred to this module.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::capture::{write_atomic, RawLine, StoredEvent};
use crate::probe::redact;
use crate::protocol::{AgentEvent, Origin};

/// An opaque identity that survives into the ledger and the metrics as a join
/// key. Held as a string because the vendors themselves hand out string ids;
/// wrapping it stops a `session_id` being passed where a `task_id` belongs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(pub String);
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl WorkerId {
    /// A stable id for one worker's run. It is xencode's own (no vendor issues
    /// it) so it is minted, not observed: `agent` plus the session the vendor
    /// opened, falling back to a launch counter when the vendor opened none, so
    /// two runs of the same agent in the same directory stay distinct.
    pub fn for_run(agent: &str, session_id: Option<&str>, launch_ordinal: u64) -> WorkerId {
        match session_id {
            Some(session) => WorkerId(format!("{agent}/{session}")),
            None => WorkerId(format!("{agent}/launch-{launch_ordinal}")),
        }
    }
}

/// One value inside an envelope, tagged with how it came to be what it is.
///
/// `Default` is [`Field::Unknown`], never `Observed`: an absent field in a stored
/// envelope is a gap that must read as "nobody said this", not as a measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Field<T> {
    /// The worker said it, this run.
    Observed(T),
    /// xencode filled the slot in because the model needs it and no stream
    /// carried it. Never a measurement.
    Synthesised(T),
    /// Nobody said anything on this run. Itself a recorded fact.
    Unknown,
    /// This vendor provably cannot supply it, with the reason it cannot.
    Unavailable { reason: String },
}

impl<T> Default for Field<T> {
    // Not `#[derive(Default)]`: a derived impl would demand `T: Default`, but the
    // no-value states carry no `T` at all, so an envelope field must default to
    // `Unknown` for a payload type that has no default (a whole capture, say).
    #[allow(clippy::derivable_impls)]
    fn default() -> Self {
        Field::Unknown
    }
}

impl<T> Field<T> {
    /// The value, if there is one. `Unknown` and `Unavailable` are always `None`;
    /// this is what stops an absence being read as a default.
    pub fn value(&self) -> Option<&T> {
        match self {
            Field::Observed(v) | Field::Synthesised(v) => Some(v),
            Field::Unknown | Field::Unavailable { .. } => None,
        }
    }

    /// Only an observed value counts as a measurement. A synthesised value is
    /// present but must never be reported as if the worker said it.
    pub fn is_observed(&self) -> bool {
        matches!(self, Field::Observed(_))
    }

    pub fn state(&self) -> &'static str {
        match self {
            Field::Observed(_) => "observed",
            Field::Synthesised(_) => "synthesised",
            Field::Unknown => "unknown",
            Field::Unavailable { .. } => "unavailable",
        }
    }

    /// Apply a function to whatever value is present, leaving the state intact.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Field<U> {
        match self {
            Field::Observed(v) => Field::Observed(f(v)),
            Field::Synthesised(v) => Field::Synthesised(f(v)),
            Field::Unknown => Field::Unknown,
            Field::Unavailable { reason } => Field::Unavailable { reason },
        }
    }
}

/// Redact the string inside a newtype id field, leaving its state untouched.
fn redact_id<T>(
    field: &Field<T>,
    get: impl Fn(&T) -> String,
    build: impl Fn(String) -> T,
) -> Field<T> {
    match field {
        Field::Observed(v) => Field::Observed(build(redact(&get(v)))),
        Field::Synthesised(v) => Field::Synthesised(build(redact(&get(v)))),
        Field::Unknown => Field::Unknown,
        Field::Unavailable { reason } => Field::Unavailable {
            reason: redact(reason),
        },
    }
}

/// The envelope one event travels in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    /// Which run this event belongs to. Always xencode's own, always present.
    pub worker_id: Field<WorkerId>,
    /// The job the worker was handed. `Unknown` when nothing carried it.
    pub task_id: Field<TaskId>,
    /// Which vendor ran. Observed: we launched the binary, so we know its name.
    pub agent_id: Field<AgentId>,
    /// The session the vendor opened. `Unavailable` when the vendor never issues
    /// one, `Unknown` when this particular line did not carry it.
    pub session_id: Field<SessionId>,
    /// xencode's gapless position in this worker's stream, 1-based.
    pub sequence: u64,
    /// Wall-clock if the vendor stamped the line, else unstated. Never a
    /// fabricated `now`.
    pub timestamp_ms: Field<u64>,
    /// Whether the payload was witnessed or inferred. A stored envelope whose
    /// `origin` is missing deserialises to *not observed*.
    #[serde(default)]
    pub origin: Origin,
    /// The [`AR-4`] raw line the payload came from, so a redacted field stays
    /// recoverable from the sealed capture. `None` only for an event xencode
    /// constructed itself.
    pub raw_line: Option<usize>,
    /// The common event itself.
    pub payload: AgentEvent,
}

impl Envelope {
    /// Build the envelope for one stored event of one run.
    ///
    /// The states are derived from the facts we actually have, not guessed to
    /// line the fields up: `agent_id` is observed (we ran that binary),
    /// `worker_id` and `sequence` are synthesised (xencode's bookkeeping),
    /// `session_id` is observed when a session was opened and known to this
    /// event, `task_id` and `timestamp_ms` are unknown here because neither a
    /// job identifier nor a per-line clock survives into a replay of a stream.
    pub fn from_stored(
        agent: &str,
        worker_id: &WorkerId,
        session_opened: Option<&str>,
        task_id: Option<&str>,
        stored: &StoredEvent,
    ) -> Envelope {
        let session_id = match session_opened {
            Some(session) => Field::Observed(SessionId(session.to_string())),
            None => Field::Unknown,
        };
        Envelope {
            worker_id: Field::Synthesised(worker_id.clone()),
            task_id: match task_id {
                Some(task) => Field::Synthesised(TaskId(task.to_string())),
                None => Field::Unknown,
            },
            agent_id: Field::Observed(AgentId(agent.to_string())),
            session_id,
            sequence: stored.seq as u64,
            // A replay of a stream has no per-line arrival time, and inventing
            // one would present a construction as a measurement.
            timestamp_ms: Field::Unavailable {
                reason: "the normaliser does not carry a per-line clock".to_string(),
            },
            origin: stored.event.origin(),
            raw_line: stored.raw_line,
            payload: stored.event.clone(),
        }
    }

    /// A copy with every credential-shaped string removed from the payload and
    /// the identity fields. `raw_line` and `sequence` are never touched: they
    /// are how a redacted value is recovered from `raw.jsonl`, not the thing
    /// being hidden. A redacted value keeps its [`Field`] state, so a scrubbed
    /// session id is still recorded as one the worker observed — it is simply
    /// not shown in the layer that propagates.
    pub fn redacted(&self) -> Envelope {
        Envelope {
            worker_id: redact_id(&self.worker_id, |w| w.0.clone(), WorkerId),
            task_id: redact_id(&self.task_id, |t| t.0.clone(), TaskId),
            agent_id: self.agent_id.clone(),
            session_id: redact_id(&self.session_id, |s| s.0.clone(), SessionId),
            sequence: self.sequence,
            timestamp_ms: self.timestamp_ms.clone(),
            origin: self.origin,
            raw_line: self.raw_line,
            payload: redact_event(&self.payload),
        }
    }

    /// Whether writing this envelope as-is would leak a credential. Redaction is
    /// idempotent, so a value is secret-bearing exactly when scrubbing it changes
    /// it — no second copy of the pattern list, and no guesswork.
    pub fn carries_secret(&self) -> bool {
        let rendered = serde_json::to_string(self).unwrap_or_default();
        redact(&rendered) != rendered
    }
}

/// Redact every string a payload can carry, leaving variant, ids and origin.
fn redact_event(event: &AgentEvent) -> AgentEvent {
    match event {
        AgentEvent::SessionStarted { session_id, origin } => AgentEvent::SessionStarted {
            session_id: session_id.as_deref().map(redact),
            origin: *origin,
        },
        AgentEvent::Message { text, origin } => AgentEvent::Message {
            text: redact(text),
            origin: *origin,
        },
        AgentEvent::ToolRequested {
            tool,
            call_id,
            origin,
        } => AgentEvent::ToolRequested {
            tool: redact(tool),
            call_id: call_id.as_deref().map(redact),
            origin: *origin,
        },
        AgentEvent::ToolStarted {
            tool,
            call_id,
            origin,
        } => AgentEvent::ToolStarted {
            tool: redact(tool),
            call_id: call_id.as_deref().map(redact),
            origin: *origin,
        },
        AgentEvent::ToolOutput {
            tool,
            call_id,
            output,
            origin,
        } => AgentEvent::ToolOutput {
            tool: redact(tool),
            call_id: call_id.as_deref().map(redact),
            output: output.as_deref().map(redact),
            origin: *origin,
        },
        AgentEvent::FileChanged { path, origin } => AgentEvent::FileChanged {
            path: redact(path),
            origin: *origin,
        },
        AgentEvent::PermissionRequested {
            tool,
            call_id,
            origin,
        } => AgentEvent::PermissionRequested {
            tool: redact(tool),
            call_id: call_id.as_deref().map(redact),
            origin: *origin,
        },
        AgentEvent::PermissionDenied {
            tool,
            call_id,
            reason,
            origin,
        } => AgentEvent::PermissionDenied {
            tool: redact(tool),
            call_id: call_id.as_deref().map(redact),
            reason: reason.as_deref().map(redact),
            origin: *origin,
        },
        AgentEvent::Error { message, origin } => AgentEvent::Error {
            message: redact(message),
            origin: *origin,
        },
        AgentEvent::Completed { outcome, origin } => AgentEvent::Completed {
            outcome: outcome.as_deref().map(redact),
            origin: *origin,
        },
        AgentEvent::SessionEnded { reason, origin } => AgentEvent::SessionEnded {
            reason: reason.as_deref().map(redact),
            origin: *origin,
        },
    }
}

/// Write the redacted envelopes for one capture and return the file path.
///
/// Written through the same owner-only atomic write as the rest of the capture,
/// so a half-written propagation store is not a thing that can happen.
pub fn write_envelopes(
    capture_dir: &Path,
    envelopes: &[Envelope],
) -> Result<std::path::PathBuf, String> {
    let path = capture_dir.join("envelope.jsonl");
    let mut buf = Vec::new();
    for envelope in envelopes {
        // Every envelope is written through `redacted` here regardless of what
        // the caller held, so there is no way to write the unscrubbed copy by
        // forgetting to call it first.
        let line = serde_json::to_string(&envelope.redacted())
            .map_err(|e| format!("{}: cannot serialise an envelope: {e}", path.display()))?;
        buf.extend_from_slice(line.as_bytes());
        buf.push(b'\n');
    }
    write_atomic(&path, &buf)?;
    Ok(path)
}

/// What a tolerant read of the envelope store found, and how much of it was torn.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvelopeRead {
    pub envelopes: Vec<Envelope>,
    /// A final line that failed to parse and had no newline after it is a write
    /// that was interrupted mid-line — `DB-5`'s torn tail — and is discarded, not
    /// reported as a stored envelope. `Some` holds the discarded text.
    pub torn_tail: Option<String>,
}

/// Read the envelope store, discarding a torn trailing line rather than failing.
///
/// `DB-5`'s contract: a propagation store the reader walks must survive the
/// writer being killed between the last whole line and its newline. A torn *tail*
/// is dropped (it was never complete), but a malformed line in the middle is a
/// real corruption and still reported, because silently skipping it would hide
/// that envelopes went missing.
pub fn read_envelopes(capture_dir: &Path) -> Result<EnvelopeRead, String> {
    let path = capture_dir.join("envelope.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    let mut torn_tail = None;
    // `str::lines` splits on '\n' and drops a trailing empty; we need to know
    // whether the file ended in a newline to tell a torn tail from a blank line.
    let ends_with_newline = text.ends_with('\n');
    let total = if text.is_empty() {
        0
    } else {
        text.lines().count()
    };
    for (i, line) in text.lines().enumerate() {
        let is_last = i + 1 == total;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Envelope>(line) {
            Ok(envelope) => out.push(envelope),
            Err(e) => {
                if is_last && !ends_with_newline {
                    torn_tail = Some(line.to_string());
                } else {
                    return Err(format!(
                        "{} line {}: not an envelope I can read ({e}); this is corruption, not a \
                         torn tail, because it is not the final partial line",
                        path.display(),
                        i + 1
                    ));
                }
            }
        }
    }
    Ok(EnvelopeRead {
        envelopes: out,
        torn_tail,
    })
}

/// Build the redacted envelopes for a whole capture, in stream order.
///
/// Takes the agent name, an optional session the run opened, and the stored
/// events so the caller does not have to reach into [`super::capture`]'s types
/// to assemble the identity of each row.
pub fn envelopes_for_capture(
    agent: &str,
    session_opened: Option<&str>,
    task_id: Option<&str>,
    launch_ordinal: u64,
    events: &[StoredEvent],
    _raw: &[RawLine],
) -> Vec<Envelope> {
    let worker = WorkerId::for_run(agent, session_opened, launch_ordinal);
    events
        .iter()
        .map(|stored| Envelope::from_stored(agent, &worker, session_opened, task_id, stored))
        .collect()
}
