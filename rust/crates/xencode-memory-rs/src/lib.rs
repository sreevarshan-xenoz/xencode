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
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::Io(source) => write!(f, "memory I/O error: {source}"),
            MemoryError::Json(source) => write!(f, "memory parse error: {source}"),
            MemoryError::NoHomeDir => write!(f, "could not determine home directory"),
        }
    }
}

impl std::error::Error for MemoryError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: String,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSession {
    pub messages: Vec<Message>,
    pub created: String,
    pub last_updated: String,
    #[serde(default)]
    pub model: Option<String>,
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
/// Mirrors `xencode/core/memory.py`.
pub struct ConversationMemory {
    max_items: usize,
    conversations: HashMap<String, ConversationSession>,
    current_session: Option<String>,
    memory_file: Option<PathBuf>,
}

impl ConversationMemory {
    /// Create a new in-memory instance.
    pub fn new(max_items: usize) -> Self {
        Self {
            max_items,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: None,
        }
    }

    /// Create an instance that persists to `~/.xencode/conversation_memory.json`.
    pub fn with_persistence(max_items: usize) -> Result<Self, MemoryError> {
        let xencode_dir = dirs::home_dir()
            .ok_or(MemoryError::NoHomeDir)?
            .join(".xencode");

        std::fs::create_dir_all(&xencode_dir).map_err(MemoryError::Io)?;
        let memory_file = xencode_dir.join("conversation_memory.json");

        let mut mem = Self {
            max_items,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: Some(memory_file),
        };
        mem.load_memory()?;
        Ok(mem)
    }

    /// Load conversation memory from disk.
    fn load_memory(&mut self) -> Result<(), MemoryError> {
        if let Some(ref memory_file) = self.memory_file {
            if memory_file.exists() {
                let content = std::fs::read_to_string(memory_file).map_err(MemoryError::Io)?;
                if let Ok(data) = serde_json::from_str::<MemoryData>(&content) {
                    self.conversations = data.conversations;
                    self.current_session = data.current_session;
                }
            }
        }
        Ok(())
    }

    /// Save conversation memory to disk.
    fn save_memory(&self) -> Result<(), MemoryError> {
        if let Some(ref memory_file) = self.memory_file {
            let data = MemoryData {
                conversations: self.conversations.clone(),
                current_session: self.current_session.clone(),
                last_updated: Utc::now().to_rfc3339(),
            };
            let json = serde_json::to_string_pretty(&data).map_err(MemoryError::Json)?;
            std::fs::write(memory_file, json).map_err(MemoryError::Io)?;
        }
        Ok(())
    }

    /// Start a new conversation session.
    pub fn start_session(&mut self, session_id: Option<String>) -> String {
        let id = session_id.unwrap_or_else(|| format!("session_{}", Utc::now().timestamp()));

        if !self.conversations.contains_key(&id) {
            self.conversations.insert(
                id.clone(),
                ConversationSession {
                    messages: Vec::new(),
                    created: Utc::now().to_rfc3339(),
                    last_updated: Utc::now().to_rfc3339(),
                    model: None,
                },
            );
        }
        self.current_session = Some(id.clone());
        let _ = self.save_memory();
        id
    }

    /// Add a message to the current session.
    pub fn add_message(&mut self, role: &str, content: &str, model: Option<String>) {
        if self.current_session.is_none() {
            self.start_session(None);
        }

        let session_id = self.current_session.as_ref().unwrap().clone();
        let session = self.conversations.get_mut(&session_id).unwrap();

        session.messages.push(Message {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Utc::now().to_rfc3339(),
            model,
        });
        session.last_updated = Utc::now().to_rfc3339();

        if session.messages.len() > self.max_items {
            let overflow = session.messages.len() - self.max_items;
            session.messages.drain(0..overflow);
        }

        let _ = self.save_memory();
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

    /// List all conversation session IDs.
    pub fn list_sessions(&self) -> Vec<String> {
        self.conversations.keys().cloned().collect()
    }

    /// Get a specific session.
    pub fn get_session(&self, session_id: &str) -> Option<&ConversationSession> {
        self.conversations.get(session_id)
    }

    /// Switch to a different conversation session.
    pub fn switch_session(&mut self, session_id: &str) -> bool {
        if self.conversations.contains_key(session_id) {
            self.current_session = Some(session_id.to_string());
            let _ = self.save_memory();
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
        };

        mem1.start_session(Some("persisted_sess".to_string()));
        mem1.add_message("user", "save me", None);

        let mut mem2 = ConversationMemory {
            max_items: 10,
            conversations: HashMap::new(),
            current_session: None,
            memory_file: Some(file.clone()),
        };
        mem2.load_memory().unwrap();

        assert_eq!(mem2.current_session(), Some(&"persisted_sess".to_string()));
        let ctx = mem2.get_context(5);
        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx[0].content, "save me");

        fs::remove_dir_all(&dir).unwrap();
    }
}
