//! Conversation memory system with session persistence and context recall.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use xencode_core_rs::ChatMessage;

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Session not found: {0}")]
    SessionNotFound(String),
}

/// A conversation session with messages and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub messages: Vec<ChatMessage>,
    pub message_count: usize,
    pub model: String,
    pub metadata: HashMap<String, String>,
}

impl Session {
    pub fn new(name: &str, model: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            messages: Vec::new(),
            message_count: 0,
            model: model.to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Add a message to the session.
    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
        self.message_count = self.messages.len();
        self.updated_at = Utc::now();
    }
}

/// Manages conversation sessions.
pub struct ConversationMemory {
    sessions: Mutex<HashMap<String, Session>>,
    active_session_id: Mutex<String>,
    memory_dir: PathBuf,
    max_sessions: usize,
}

impl ConversationMemory {
    /// Create a new conversation memory store.
    pub fn new(memory_dir: PathBuf, max_sessions: usize) -> Self {
        std::fs::create_dir_all(&memory_dir).ok();
        let memory = Self {
            sessions: Mutex::new(HashMap::new()),
            active_session_id: Mutex::new(String::new()),
            memory_dir,
            max_sessions,
        };
        // Load existing sessions
        let _ = memory.load_sessions();
        memory
    }

    /// Create a new session.
    pub fn create_session(&self, name: &str, model: &str) -> String {
        let session = Session::new(name, model);
        let id = session.id.clone();
        self.sessions.lock().unwrap().insert(id.clone(), session);
        *self.active_session_id.lock().unwrap() = id.clone();
        let _ = self.persist(&id);
        id
    }

    /// Get the active session.
    pub fn active_session(&self) -> Option<Session> {
        let id = self.active_session_id.lock().unwrap().clone();
        if id.is_empty() {
            return None;
        }
        self.sessions.lock().unwrap().get(&id).cloned()
    }

    /// Get a session by ID.
    pub fn get_session(&self, id: &str) -> Option<Session> {
        self.sessions.lock().unwrap().get(id).cloned()
    }

    /// Set the active session.
    pub fn set_active(&self, id: &str) -> Result<(), MemoryError> {
        if self.sessions.lock().unwrap().contains_key(id) {
            *self.active_session_id.lock().unwrap() = id.to_string();
            Ok(())
        } else {
            Err(MemoryError::SessionNotFound(id.to_string()))
        }
    }

    /// Add a message to the active session.
    pub fn add_message(&self, message: ChatMessage) -> Result<(), MemoryError> {
        let id = self.active_session_id.lock().unwrap().clone();
        if id.is_empty() {
            let new_id = self.create_session("default", "default");
            *self.active_session_id.lock().unwrap() = new_id;
        }
        let id = self.active_session_id.lock().unwrap().clone();
        if let Some(session) = self.sessions.lock().unwrap().get_mut(&id) {
            session.add_message(message);
            let _ = self.persist(&id);
            Ok(())
        } else {
            Err(MemoryError::SessionNotFound(id))
        }
    }

    /// List all sessions.
    pub fn list_sessions(&self) -> Vec<Session> {
        let mut sessions: Vec<Session> = self.sessions.lock().unwrap().values().cloned().collect();
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        sessions
    }

    /// Delete a session.
    pub fn delete_session(&self, id: &str) -> Result<(), MemoryError> {
        self.sessions.lock().unwrap().remove(id);
        let file_path = self.memory_dir.join(format!("session_{}.json", id));
        if file_path.exists() {
            std::fs::remove_file(&file_path)?;
        }
        Ok(())
    }

    /// Persist a session to disk.
    fn persist(&self, id: &str) -> Result<(), MemoryError> {
        std::fs::create_dir_all(&self.memory_dir)?;
        if let Some(session) = self.sessions.lock().unwrap().get(id) {
            let data = serde_json::to_string_pretty(session)?;
            let file_path = self.memory_dir.join(format!("session_{}.json", id));
            std::fs::write(&file_path, data)?;
        }
        Ok(())
    }

    /// Load all sessions from disk.
    fn load_sessions(&self) -> Result<(), MemoryError> {
        if !self.memory_dir.exists() {
            return Ok(());
        }
        for entry in std::fs::read_dir(&self.memory_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(data) = std::fs::read_to_string(&path) {
                    if let Ok(session) = serde_json::from_str::<Session>(&data) {
                        self.sessions.lock().unwrap().insert(session.id.clone(), session);
                    }
                }
            }
        }
        // Set first session as active if none set
        let active = self.active_session_id.lock().unwrap().clone();
        if active.is_empty() {
            let sessions = self.sessions.lock().unwrap();
            if let Some(first) = sessions.keys().next() {
                drop(sessions);
                *self.active_session_id.lock().unwrap() = first.clone();
            }
        }
        Ok(())
    }
}
