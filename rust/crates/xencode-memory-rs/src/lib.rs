//! Conversation and durable-fact memory, and the scoped shared memory that
//! workers hand to each other (OR-8).

pub mod scoped;

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Maximum number of messages stored per session by default.
pub const DEFAULT_MAX_MEMORY_ITEMS: usize = 50;

/// Errors from memory operations.
#[derive(Debug)]
pub enum MemoryError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NoHomeDir,
    SessionNotFound(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::Io(source) => write!(f, "memory I/O error: {source}"),
            MemoryError::Json(source) => write!(f, "memory parse error: {source}"),
            MemoryError::NoHomeDir => write!(f, "could not determine home directory"),
            MemoryError::SessionNotFound(id) => write!(f, "session not found: {id}"),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Where one conversation event originated.
///
/// Mirrors the distinction settled for vendor envelopes: entries produced during
/// direct interaction are [`Origin::Observed`], while entries synthesised during
/// fork derivation or state reconstruction are [`Origin::Synthesised`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// The event was observed directly during this session's run.
    #[default]
    Observed,
    /// The event was inherited from a parent session via fork or reconstructed
    /// from older state. Never a direct observation of this run.
    Synthesised,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: String,
    #[serde(default)]
    pub model: Option<String>,
}

/// An append-only event recorded in a conversation session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationEvent {
    pub id: u64,
    pub session_id: String,
    pub timestamp: String,
    #[serde(default)]
    pub origin: Origin,
    pub message: Message,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSession {
    /// Append-only event log. Superseded entries are hidden from the active
    /// projection `messages` without being deleted.
    #[serde(default)]
    pub events: Vec<ConversationEvent>,
    /// Compacted projection derived from `events`, bounded by `max_items`.
    #[serde(default)]
    pub messages: Vec<Message>,
    pub created: String,
    pub last_updated: String,
    #[serde(default)]
    pub model: Option<String>,
}

impl ConversationSession {
    /// Create a new session.
    pub fn new(model: Option<String>) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            events: Vec::new(),
            messages: Vec::new(),
            created: now.clone(),
            last_updated: now,
            model,
        }
    }

    /// Rebuild the derived projection `messages` from the append-only event log.
    pub fn rebuild_projection(&mut self, max_items: usize) {
        let total = self.events.len();
        let start = total.saturating_sub(max_items);
        self.messages = self.events[start..]
            .iter()
            .map(|e| e.message.clone())
            .collect();
    }

    /// Retrieve the very first message recorded in this session.
    ///
    /// Preserved even after the active projection exceeds its cap and compacts.
    pub fn first_message(&self) -> Option<&Message> {
        self.events
            .first()
            .map(|e| &e.message)
            .or_else(|| self.messages.first())
    }

    /// Retrieve the first event recorded in this session.
    pub fn first_event(&self) -> Option<&ConversationEvent> {
        self.events.first()
    }

    /// Fork this session up to an optional prefix boundary.
    ///
    /// The child receives an exact prefix of the parent's events, and nothing after it.
    /// Inherited events are marked with [`Origin::Synthesised`] so they are distinguishable
    /// from future events observed in the child.
    pub fn fork(&self, new_session_id: &str, prefix_len: Option<usize>, max_items: usize) -> Self {
        let count = prefix_len
            .map(|n| n.min(self.events.len()))
            .unwrap_or(self.events.len());
        let inherited_events: Vec<ConversationEvent> = self.events[..count]
            .iter()
            .enumerate()
            .map(|(i, ev)| ConversationEvent {
                id: (i + 1) as u64,
                session_id: new_session_id.to_string(),
                timestamp: ev.timestamp.clone(),
                origin: Origin::Synthesised,
                message: ev.message.clone(),
            })
            .collect();

        let mut child = Self {
            events: inherited_events,
            messages: Vec::new(),
            created: Utc::now().to_rfc3339(),
            last_updated: Utc::now().to_rfc3339(),
            model: self.model.clone(),
        };
        child.rebuild_projection(max_items);
        child
    }
}

/// Read an append-only JSONL event file, tolerating a torn trailing line.
///
/// Follows `DB-5`'s contract: an interrupted write that didn't finish the last line
/// has its torn tail dropped gracefully, while a malformed line in the middle is reported.
pub fn read_events_tolerant<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<(Vec<ConversationEvent>, Option<String>), MemoryError> {
    let p = path.as_ref();
    if !p.exists() {
        return Ok((Vec::new(), None));
    }
    let text = std::fs::read_to_string(p).map_err(MemoryError::Io)?;
    let mut events = Vec::new();
    let mut torn_tail = None;
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
        match serde_json::from_str::<ConversationEvent>(line) {
            Ok(ev) => events.push(ev),
            Err(e) => {
                if is_last && !ends_with_newline {
                    torn_tail = Some(line.to_string());
                } else {
                    return Err(MemoryError::Json(e));
                }
            }
        }
    }
    Ok((events, torn_tail))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryData {
    #[serde(default)]
    conversations: HashMap<String, ConversationSession>,
    #[serde(default)]
    current_session: Option<String>,
    #[serde(default)]
    last_updated: String,
}

/// Advanced conversation memory with context management.
///
/// An append-only event log backs each session; active history for model
/// context is derived as a compacted projection.
pub struct ConversationMemory {
    max_items: usize,
    conversations: HashMap<String, ConversationSession>,
    current_session: Option<String>,
    memory_file: Option<PathBuf>,
    events_file: Option<PathBuf>,
}

impl ConversationMemory {
    /// Create a new in-memory instance.
    pub fn new(max_items: usize) -> Self {
        Self {
            max_items,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: None,
            events_file: None,
        }
    }

    /// The directory conversation memory persists to: `~/.local/state/xencode`
    /// (or `$XCODE_CONFIG_DIR`).
    pub fn memory_dir() -> Result<PathBuf, MemoryError> {
        xencode_config_rs::paths::state_dir().map_err(|_| MemoryError::NoHomeDir)
    }

    /// Create an instance that persists to `<state dir>/conversation_memory.json`
    /// and `<state dir>/conversation_events.jsonl`.
    pub fn with_persistence(max_items: usize) -> Result<Self, MemoryError> {
        let xencode_dir = Self::memory_dir()?;

        std::fs::create_dir_all(&xencode_dir).map_err(MemoryError::Io)?;
        let memory_file = xencode_dir.join("conversation_memory.json");
        let events_file = xencode_dir.join("conversation_events.jsonl");

        let mut mem = Self {
            max_items,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: Some(memory_file),
            events_file: Some(events_file),
        };
        mem.load_memory()?;
        Ok(mem)
    }

    /// Load conversation memory from disk.
    fn load_memory(&mut self) -> Result<(), MemoryError> {
        // DB-5: Read the append-only event log first if it exists, tolerating a torn tail.
        if let Some(ref ef) = self.events_file {
            if ef.exists() {
                if let Ok((events, _torn)) = read_events_tolerant(ef) {
                    for ev in events {
                        let sess = self
                            .conversations
                            .entry(ev.session_id.clone())
                            .or_insert_with(|| ConversationSession::new(None));
                        sess.last_updated = ev.timestamp.clone();
                        sess.events.push(ev);
                    }
                }
            }
        }

        if let Some(ref memory_file) = self.memory_file {
            if memory_file.exists() {
                let content = std::fs::read_to_string(memory_file).map_err(MemoryError::Io)?;
                if let Ok(data) = serde_json::from_str::<MemoryData>(&content) {
                    for (id, session) in data.conversations {
                        let entry = self
                            .conversations
                            .entry(id.clone())
                            .or_insert_with(|| ConversationSession::new(session.model.clone()));
                        if entry.events.is_empty() {
                            if !session.events.is_empty() {
                                entry.events = session.events;
                            } else {
                                // Legacy snapshot: reconstruct events from messages
                                for (idx, msg) in session.messages.iter().enumerate() {
                                    entry.events.push(ConversationEvent {
                                        id: (idx + 1) as u64,
                                        session_id: id.clone(),
                                        timestamp: msg.timestamp.clone(),
                                        origin: Origin::Synthesised,
                                        message: msg.clone(),
                                    });
                                }
                            }
                        }
                        entry.created = session.created;
                        entry.last_updated = session.last_updated;
                        if entry.model.is_none() {
                            entry.model = session.model;
                        }
                    }
                    self.current_session = data.current_session;
                }
            }
        }

        // Rebuild all projections from the event logs rather than trusting snapshots
        for session in self.conversations.values_mut() {
            session.rebuild_projection(self.max_items);
        }
        Ok(())
    }

    /// Save conversation memory to disk. Empty sessions with no messages
    /// or events are not written to disk, leaving the file untouched until a message is added.
    fn save_memory(&self) -> Result<(), MemoryError> {
        if let Some(ref memory_file) = self.memory_file {
            let active_conversations: HashMap<String, ConversationSession> = self
                .conversations
                .iter()
                .filter(|(_, s)| !s.events.is_empty() || !s.messages.is_empty())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if active_conversations.is_empty() && !memory_file.exists() {
                return Ok(());
            }
            let data = MemoryData {
                conversations: active_conversations,
                current_session: self.current_session.clone(),
                last_updated: Utc::now().to_rfc3339(),
            };
            let json = serde_json::to_string_pretty(&data).map_err(MemoryError::Json)?;
            xencode_core_rs::write_atomic(memory_file, json.as_bytes()).map_err(MemoryError::Io)?;
        }
        Ok(())
    }

    /// Start a new conversation session. Does not persist to disk until
    /// a message is added.
    pub fn start_session(&mut self, session_id: Option<String>) -> String {
        let id = session_id.unwrap_or_else(|| format!("session_{}", Utc::now().timestamp()));

        if !self.conversations.contains_key(&id) {
            self.conversations
                .insert(id.clone(), ConversationSession::new(None));
        }
        self.current_session = Some(id.clone());
        id
    }

    /// Add a message to the current session.
    ///
    /// The message is appended to the event log as [`Origin::Observed`] with
    /// `SE-5` secret scrubbing. The active projection is rebuilt to cap at `max_items`.
    pub fn add_message(&mut self, role: &str, content: &str, model: Option<String>) {
        if self.current_session.is_none() {
            self.start_session(None);
        }

        let session_id = self.current_session.as_ref().unwrap().clone();
        // SE-5: redact secrets before recording to event log
        let scrubbed = xencode_context_rs::redact_secrets(content);
        let now = Utc::now().to_rfc3339();

        let session = self.conversations.get_mut(&session_id).unwrap();
        let event_id = (session.events.len() + 1) as u64;
        let msg = Message {
            role: role.to_string(),
            content: scrubbed,
            timestamp: now.clone(),
            model,
        };

        let event = ConversationEvent {
            id: event_id,
            session_id: session_id.clone(),
            timestamp: now.clone(),
            origin: Origin::Observed,
            message: msg,
        };

        session.events.push(event.clone());
        session.last_updated = now;
        session.rebuild_projection(self.max_items);

        // Append to event log file if persistence is active
        if let Some(ref ef) = self.events_file {
            use std::io::Write;
            if let Ok(line) = serde_json::to_string(&event) {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(ef)
                {
                    let _ = writeln!(f, "{line}");
                }
            }
        }

        let _ = self.save_memory();
    }

    /// Retrieve the very first message for a session from its event log.
    ///
    /// Preserved and printed on demand even when the session has grown past
    /// `max_items` and its active projection has compacted away earlier turns.
    pub fn first_message(&self, session_id: &str) -> Option<&Message> {
        self.conversations
            .get(session_id)
            .and_then(|s| s.first_message())
    }

    /// Retrieve the very first event for a session from its event log.
    pub fn first_event(&self, session_id: &str) -> Option<&ConversationEvent> {
        self.conversations
            .get(session_id)
            .and_then(|s| s.first_event())
    }

    /// Retrieve all recorded events for a session in append order.
    pub fn session_events(&self, session_id: &str) -> Option<&[ConversationEvent]> {
        self.conversations
            .get(session_id)
            .map(|s| s.events.as_slice())
    }

    /// Fork a session into a child holding an exact prefix of the parent's events.
    ///
    /// The child receives the parent's events up to `prefix_len` (or all if `None`).
    /// Each inherited event is tagged [`Origin::Synthesised`] so it is distinguishable
    /// from subsequent entries observed directly in the child session.
    pub fn fork_session(
        &mut self,
        parent_session_id: &str,
        new_session_id: Option<String>,
        prefix_len: Option<usize>,
    ) -> Result<String, MemoryError> {
        let parent = self
            .conversations
            .get(parent_session_id)
            .ok_or_else(|| MemoryError::SessionNotFound(parent_session_id.to_string()))?;

        let child_id = new_session_id
            .unwrap_or_else(|| format!("{parent_session_id}_fork_{}", Utc::now().timestamp()));

        let child = parent.fork(&child_id, prefix_len, self.max_items);

        // Append inherited events to events_file if persistence is configured
        if let Some(ref ef) = self.events_file {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(ef)
            {
                for ev in &child.events {
                    if let Ok(line) = serde_json::to_string(ev) {
                        let _ = writeln!(f, "{line}");
                    }
                }
            }
        }

        self.conversations.insert(child_id.clone(), child);
        self.current_session = Some(child_id.clone());
        let _ = self.save_memory();
        Ok(child_id)
    }

    /// Get recent conversation context for model input.
    pub fn get_context(&self, max_messages: usize) -> Vec<Message> {
        if let Some(ref session_id) = self.current_session {
            if let Some(session) = self.conversations.get(session_id) {
                let start = if session.messages.len() > max_messages {
                    session.messages.len() - max_messages
                } else {
                    0
                };
                return session.messages[start..].to_vec();
            }
        }
        Vec::new()
    }

    /// List conversation session IDs that contain at least one message or event.
    pub fn list_sessions(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .conversations
            .iter()
            .filter(|(_, s)| !s.events.is_empty() || !s.messages.is_empty())
            .map(|(k, _)| k.clone())
            .collect();
        ids.sort();
        ids
    }

    /// List all conversation session IDs, including empty ones.
    pub fn list_all_sessions(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.conversations.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Prune empty conversation sessions that have no messages or events, saving to disk.
    /// Returns the number of pruned sessions.
    pub fn prune_empty_sessions(&mut self) -> usize {
        let before = self.conversations.len();
        self.conversations
            .retain(|_, s| !s.events.is_empty() || !s.messages.is_empty());
        if let Some(ref cur) = self.current_session {
            if !self.conversations.contains_key(cur) {
                self.current_session = None;
            }
        }
        let pruned = before - self.conversations.len();
        if pruned > 0 {
            let _ = self.save_memory();
        }
        pruned
    }

    /// Get a specific session.
    pub fn get_session(&self, session_id: &str) -> Option<&ConversationSession> {
        self.conversations.get(session_id)
    }

    /// Switch to a different conversation session.
    pub fn switch_session(&mut self, session_id: &str) -> bool {
        if self.conversations.contains_key(session_id) {
            self.current_session = Some(session_id.to_string());
            if let Some(sess) = self.conversations.get(session_id) {
                if !sess.messages.is_empty() || !sess.events.is_empty() {
                    let _ = self.save_memory();
                }
            }
            true
        } else {
            false
        }
    }

    /// Get the current session ID
    pub fn current_session(&self) -> Option<&String> {
        self.current_session.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn temp_dir() -> PathBuf {
        // A process-wide counter, not a timestamp. Tests in one binary run in
        // parallel threads and can read the same nanosecond, which had them share
        // a directory and clobber each other's assertions.
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-memory-test-{unique}"))
    }

    #[test]
    fn start_and_switch_session() {
        let mut mem = ConversationMemory::new(10);
        mem.start_session(Some("sess1".to_string()));
        let s2 = mem.start_session(Some("sess2".to_string()));

        assert_eq!(s2, "sess2");
        assert_eq!(mem.current_session(), Some(&"sess2".to_string()));

        let switched = mem.switch_session("sess1");
        assert!(switched);
        assert_eq!(mem.current_session(), Some(&"sess1".to_string()));
    }

    #[test]
    fn add_and_get_messages() {
        let mut mem = ConversationMemory::new(10);
        mem.start_session(Some("sess".to_string()));

        mem.add_message("user", "hello", None);
        mem.add_message("assistant", "hi there", Some("model-a".to_string()));

        let ctx = mem.get_context(5);
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].role, "user");
        assert_eq!(ctx[0].content, "hello");
        assert_eq!(ctx[1].role, "assistant");
        assert_eq!(ctx[1].model, Some("model-a".to_string()));
    }

    #[test]
    fn max_items_trimming() {
        let mut mem = ConversationMemory::new(2);
        mem.start_session(Some("sess".to_string()));

        mem.add_message("user", "msg1", None);
        mem.add_message("user", "msg2", None);
        mem.add_message("user", "msg3", None);

        let ctx = mem.get_context(10);
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].content, "msg2");
        assert_eq!(ctx[1].content, "msg3");
    }

    #[test]
    fn persistence_roundtrip() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("conversation_memory.json");

        let mut mem1 = ConversationMemory {
            max_items: 10,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: Some(file.clone()),
            events_file: None,
        };

        mem1.start_session(Some("persisted_sess".to_string()));
        mem1.add_message("user", "save me", None);

        let mut mem2 = ConversationMemory {
            max_items: 10,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: Some(file.clone()),
            events_file: None,
        };
        mem2.load_memory().unwrap();

        assert_eq!(mem2.current_session(), Some(&"persisted_sess".to_string()));
        let ctx = mem2.get_context(5);
        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx[0].content, "save me");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_session_leaves_file_untouched_until_message_added() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("conversation_memory.json");

        let mut mem = ConversationMemory {
            max_items: 10,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: Some(file.clone()),
            events_file: None,
        };

        // Starting a session must not write to disk
        let sid = mem.start_session(Some("unfilled".to_string()));
        assert_eq!(sid, "unfilled");
        assert!(
            !file.exists(),
            "starting an empty session must not create memory file"
        );
        assert!(
            mem.list_sessions().is_empty(),
            "list_sessions must filter out empty session"
        );
        assert_eq!(mem.list_all_sessions(), vec!["unfilled"]);

        // Adding a message must now write to disk
        mem.add_message("user", "first turn", None);
        assert!(file.exists(), "adding a message must persist memory file");
        assert_eq!(mem.list_sessions(), vec!["unfilled"]);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_sessions_are_filtered_from_listing_and_can_be_pruned() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("conversation_memory.json");

        let mut mem = ConversationMemory {
            max_items: 10,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: Some(file.clone()),
            events_file: None,
        };

        mem.start_session(Some("empty_1".to_string()));
        mem.start_session(Some("full_1".to_string()));
        mem.add_message("user", "content", None);
        mem.start_session(Some("empty_2".to_string()));

        assert_eq!(mem.list_sessions(), vec!["full_1"]);
        let mut all = mem.list_all_sessions();
        all.sort();
        assert_eq!(all, vec!["empty_1", "empty_2", "full_1"]);

        // Pruning removes the two empty sessions and keeps full_1
        let pruned = mem.prune_empty_sessions();
        assert_eq!(pruned, 2);
        assert_eq!(mem.list_all_sessions(), vec!["full_1"]);
        assert_eq!(mem.list_sessions(), vec!["full_1"]);

        // Reload from disk to verify disk state was updated
        let mut mem2 = ConversationMemory {
            max_items: 10,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: Some(file.clone()),
            events_file: None,
        };
        mem2.load_memory().unwrap();
        assert_eq!(mem2.list_all_sessions(), vec!["full_1"]);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn session_over_cap_preserves_and_prints_first_message_on_demand() {
        // Cap is 2 messages
        let mut mem = ConversationMemory::new(2);
        mem.start_session(Some("capped_sess".to_string()));

        mem.add_message("user", "first message", None);
        mem.add_message("assistant", "second message", None);
        mem.add_message("user", "third message", None);
        mem.add_message("assistant", "fourth message", None);

        // Context projection is compacted to the most recent 2 messages
        let ctx = mem.get_context(10);
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].content, "third message");
        assert_eq!(ctx[1].content, "fourth message");

        // First message is preserved in append-only event log and printed on demand
        let first = mem
            .first_message("capped_sess")
            .expect("first message exists");
        assert_eq!(first.role, "user");
        assert_eq!(first.content, "first message");

        // Event log holds all 4 recorded turns
        let events = mem.session_events("capped_sess").expect("events exist");
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].message.content, "first message");
        assert_eq!(events[0].origin, Origin::Observed);
    }

    #[test]
    fn fork_produces_child_with_exact_prefix_and_distinguishable_origins() {
        let mut mem = ConversationMemory::new(10);
        mem.start_session(Some("parent".to_string()));

        mem.add_message("user", "turn 1", None);
        mem.add_message("assistant", "turn 2", None);
        mem.add_message("user", "turn 3", None);

        // Fork prefix: inherit first 2 events
        let child_id = mem
            .fork_session("parent", Some("child".to_string()), Some(2))
            .expect("fork should succeed");
        assert_eq!(child_id, "child");

        // Add 4th message to parent after fork point
        mem.switch_session("parent");
        mem.add_message("assistant", "turn 4 after fork", None);

        // Parent has 4 events
        let parent_events = mem.session_events("parent").expect("parent events");
        assert_eq!(parent_events.len(), 4);

        // Child session has exact prefix of 2 events, and nothing after the fork point
        let child_events = mem.session_events("child").expect("child events");
        assert_eq!(child_events.len(), 2);
        assert_eq!(child_events[0].message.content, "turn 1");
        assert_eq!(child_events[1].message.content, "turn 2");

        // Inherited entries carry Origin::Synthesised
        assert_eq!(child_events[0].origin, Origin::Synthesised);
        assert_eq!(child_events[1].origin, Origin::Synthesised);

        // Now add a message directly into child session
        mem.switch_session("child");
        mem.add_message("user", "child unique turn", None);

        // Child's own new entry is distinguishable from inherited ones: Origin::Observed
        let child_events_after = mem.session_events("child").expect("child events");
        assert_eq!(child_events_after.len(), 3);
        assert_eq!(child_events_after[0].origin, Origin::Synthesised);
        assert_eq!(child_events_after[1].origin, Origin::Synthesised);
        assert_eq!(child_events_after[2].origin, Origin::Observed);
        assert_eq!(child_events_after[2].message.content, "child unique turn");

        // Parent still has 4 events, unpolluted by child turn
        assert_eq!(mem.session_events("parent").unwrap().len(), 4);
    }

    #[test]
    fn secret_scan_redacts_credentials_before_recording_event() {
        let mut mem = ConversationMemory::new(10);
        mem.start_session(Some("secure_sess".to_string()));

        let secret = "sk-FAKE-NOT-A-REAL-TEST-KEY";
        mem.add_message("user", &format!("My API token is {secret}"), None);

        let first = mem.first_message("secure_sess").expect("first message");
        assert!(
            !first.content.contains(secret),
            "secret was not redacted from message"
        );
        assert!(first.content.contains("[redacted]"));

        let event = mem.first_event("secure_sess").expect("first event");
        assert!(
            !event.message.content.contains(secret),
            "secret was not redacted from event log"
        );
    }

    #[test]
    fn db5_tolerant_event_reading_discards_torn_tail() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let events_path = dir.join("events.jsonl");

        let ev1 = ConversationEvent {
            id: 1,
            session_id: "s1".to_string(),
            timestamp: "2026-10-07T00:00:00Z".to_string(),
            origin: Origin::Observed,
            message: Message {
                role: "user".to_string(),
                content: "complete turn 1".to_string(),
                timestamp: "2026-10-07T00:00:00Z".to_string(),
                model: None,
            },
        };
        let ev2 = ConversationEvent {
            id: 2,
            session_id: "s1".to_string(),
            timestamp: "2026-10-07T00:00:01Z".to_string(),
            origin: Origin::Observed,
            message: Message {
                role: "assistant".to_string(),
                content: "complete turn 2".to_string(),
                timestamp: "2026-10-07T00:00:01Z".to_string(),
                model: None,
            },
        };

        // Two complete lines followed by a torn trailing line without a trailing newline
        let mut file_content = format!(
            "{}\n{}\n{{\"id\":3,\"session_id\":\"s1\",\"timestamp\":\"torn",
            serde_json::to_string(&ev1).unwrap(),
            serde_json::to_string(&ev2).unwrap()
        );
        fs::write(&events_path, &file_content).unwrap();

        let (events, torn) =
            read_events_tolerant(&events_path).expect("read should tolerate torn tail");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, 1);
        assert_eq!(events[1].id, 2);
        assert!(torn.is_some(), "torn tail should be captured and discarded");
        assert!(torn.unwrap().contains("torn"));

        // A malformed line in the middle of whole lines must return an error
        file_content = format!(
            "{}\n{{\"malformed\":json}}\n{}\n",
            serde_json::to_string(&ev1).unwrap(),
            serde_json::to_string(&ev2).unwrap()
        );
        fs::write(&events_path, &file_content).unwrap();
        let err = read_events_tolerant(&events_path);
        assert!(
            err.is_err(),
            "malformed mid-line corruption must be reported"
        );

        fs::remove_dir_all(&dir).unwrap();
    }
}
