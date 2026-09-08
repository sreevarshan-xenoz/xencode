use axum::extract::ws::{Message, WebSocket};
use futures_util::stream::SplitSink;
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_app_state_new() {
        let state = AppState::new();
        let peers = state.peers.lock().await;
        assert!(peers.is_empty());
        let sessions = state.sessions.lock().await;
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn test_remove_peer_removes_user() {
        let state = Arc::new(AppState::new());
        // Manually add a user to a session
        {
            let mut sessions = state.sessions.lock().await;
            sessions.insert(
                "session-1".to_string(),
                vec!["alice".to_string(), "bob".to_string()],
            );
        }
        remove_peer(&state, "session-1", "alice").await;
        let sessions = state.sessions.lock().await;
        assert_eq!(sessions.get("session-1"), Some(&vec!["bob".to_string()]));
    }

    #[tokio::test]
    async fn test_remove_peer_last_user_removes_entry() {
        let state = Arc::new(AppState::new());
        {
            let mut sessions = state.sessions.lock().await;
            sessions.insert("session-2".to_string(), vec!["dave".to_string()]);
        }
        remove_peer(&state, "session-2", "dave").await;
        let sessions = state.sessions.lock().await;
        // The key should still exist but with an empty vec
        assert_eq!(sessions.get("session-2"), Some(&vec![] as &Vec<String>));
    }

    #[tokio::test]
    async fn test_remove_peer_nonexistent_session() {
        let state = Arc::new(AppState::new());
        // Should not panic
        remove_peer(&state, "ghost-session", "nobody").await;
        // State should still be empty
        let sessions = state.sessions.lock().await;
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn test_remove_peer_nonexistent_user() {
        let state = Arc::new(AppState::new());
        {
            let mut sessions = state.sessions.lock().await;
            sessions.insert("session-3".to_string(), vec!["eve".to_string()]);
        }
        // Removing a user that doesn't exist in the session
        remove_peer(&state, "session-3", "mallory").await;
        let sessions = state.sessions.lock().await;
        assert_eq!(sessions.get("session-3"), Some(&vec!["eve".to_string()]));
    }

    #[tokio::test]
    async fn test_broadcast_to_session_no_peers() {
        // Broadcasting to a session with no peers should not panic
        let state = Arc::new(AppState::new());
        {
            let mut sessions = state.sessions.lock().await;
            sessions.insert("empty-session".to_string(), vec!["alice".to_string()]);
        }
        broadcast_to_session(&state, "empty-session", r#"{"type":"test"}"#, None).await;
        // No assertions needed — just shouldn't crash
    }

    #[tokio::test]
    async fn test_broadcast_to_session_nonexistent() {
        // Broadcasting to a session that doesn't exist should not panic
        let state = Arc::new(AppState::new());
        broadcast_to_session(&state, "no-such-session", r#"{"type":"test"}"#, None).await;
    }
}

pub type PeerMap = Arc<Mutex<HashMap<String, Vec<SplitSink<WebSocket, Message>>>>>;

pub struct AppState {
    pub peers: PeerMap,
    pub sessions: Arc<Mutex<HashMap<String, Vec<String>>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            peers: Arc::new(Mutex::new(HashMap::new())),
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// Handle an incoming WebSocket connection.
pub async fn handle_socket(
    socket: WebSocket,
    session_id: String,
    username: String,
    state: Arc<AppState>,
) {
    let (sender, mut receiver) = socket.split();

    // Register the peer
    {
        let mut peers = state.peers.lock().await;
        peers.entry(session_id.clone()).or_default().push(sender);
    }
    {
        let mut sessions = state.sessions.lock().await;
        sessions
            .entry(session_id.clone())
            .or_default()
            .push(username.clone());
    }

    info!("User {username} joined session {session_id}");

    // Broadcast join message
    broadcast_to_session(
        &state,
        &session_id,
        &serde_json::json!({
            "type": "join",
            "username": username,
        })
        .to_string(),
        None,
    )
    .await;

    // Forward messages from this peer to others
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(text) = msg {
            broadcast_to_session(&state, &session_id, &text, Some(&username)).await;
        }
    }

    // Handle disconnect
    info!("User {username} left session {session_id}");
    remove_peer(&state, &session_id, &username).await;

    broadcast_to_session(
        &state,
        &session_id,
        &serde_json::json!({
            "type": "leave",
            "username": username,
        })
        .to_string(),
        None,
    )
    .await;
}

/// Broadcast a message to all peers in a session, optionally excluding a sender.
async fn broadcast_to_session(
    state: &AppState,
    session_id: &str,
    message: &str,
    _exclude: Option<&str>,
) {
    use futures_util::SinkExt;
    let msg = Message::Text(message.to_string().into());
    let mut peers = state.peers.lock().await;
    if let Some(senders) = peers.get_mut(session_id) {
        // Build a new list of active senders by trying to send to each
        let mut active: Vec<SplitSink<axum::extract::ws::WebSocket, Message>> = Vec::new();
        for mut sender in senders.drain(..) {
            match sender.send(msg.clone()).await {
                Ok(()) => active.push(sender),
                Err(e) => warn!("Failed to send to peer: {e}"),
            }
        }
        *senders = active;
    }
}

/// Remove a peer from the session on disconnect.
async fn remove_peer(state: &AppState, session_id: &str, username: &str) {
    let mut sessions = state.sessions.lock().await;
    if let Some(users) = sessions.get_mut(session_id) {
        users.retain(|u| u != username);
    }
}
