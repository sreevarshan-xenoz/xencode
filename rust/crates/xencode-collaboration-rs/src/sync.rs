use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: String,
    pub username: String,
    pub joined_at: String,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMessage {
    pub message_type: String,
    pub sender: String,
    pub payload: serde_json::Value,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: String,
    pub peers: Vec<PeerInfo>,
    pub created_at: String,
}

/// Coordinates sync between peers in collaboration sessions.
#[derive(Debug, Default)]
pub struct SyncCoordinator {
    sessions: HashMap<String, SessionState>,
}

impl SyncCoordinator {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Join a session as a new peer.
    pub fn join_session(&mut self, session_id: &str, username: &str) -> PeerInfo {
        let peer = PeerInfo {
            peer_id: uuid::Uuid::new_v4().to_string(),
            username: username.to_string(),
            joined_at: chrono::Utc::now().to_rfc3339(),
            is_active: true,
        };

        let entry = self
            .sessions
            .entry(session_id.to_string())
            .or_insert_with(|| {
                let now = chrono::Utc::now().to_rfc3339();
                SessionState {
                    session_id: session_id.to_string(),
                    peers: Vec::new(),
                    created_at: now,
                }
            });

        let info = peer.clone();
        entry.peers.push(peer);
        info
    }

    /// Leave a session.
    pub fn leave_session(&mut self, session_id: &str, peer_id: &str) {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.peers.retain(|p| p.peer_id != peer_id);
            if session.peers.is_empty() {
                self.sessions.remove(session_id);
            }
        }
    }

    /// Get all active peers in a session.
    pub fn get_peers(&self, session_id: &str) -> Vec<&PeerInfo> {
        self.sessions
            .get(session_id)
            .map(|s| s.peers.iter().filter(|p| p.is_active).collect())
            .unwrap_or_default()
    }

    /// Get a list of all active sessions.
    pub fn get_active_sessions(&self) -> Vec<&SessionState> {
        self.sessions.values().collect()
    }

    /// Check if a session exists and has peers.
    pub fn session_has_peers(&self, session_id: &str) -> bool {
        self.sessions
            .get(session_id)
            .map(|s| !s.peers.is_empty())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_and_leave_session() {
        let mut coord = SyncCoordinator::new();
        let peer = coord.join_session("session-1", "alice");
        assert!(coord.session_has_peers("session-1"));
        assert_eq!(coord.get_peers("session-1").len(), 1);

        coord.leave_session("session-1", &peer.peer_id);
        assert!(!coord.session_has_peers("session-1"));
    }

    #[test]
    fn test_multiple_peers() {
        let mut coord = SyncCoordinator::new();
        coord.join_session("session-1", "alice");
        coord.join_session("session-1", "bob");
        coord.join_session("session-1", "carol");

        assert_eq!(coord.get_peers("session-1").len(), 3);
    }

    #[test]
    fn test_get_active_sessions() {
        let mut coord = SyncCoordinator::new();
        coord.join_session("session-1", "alice");
        coord.join_session("session-2", "bob");

        assert_eq!(coord.get_active_sessions().len(), 2);
    }
}
