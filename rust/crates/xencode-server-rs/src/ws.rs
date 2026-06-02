use std::collections::HashMap;
use std::sync::Arc;
use axum::extract::ws::{Message, WebSocket};
use futures_util::stream::SplitSink;
use futures_util::StreamExt;
use tokio::sync::Mutex;
use tracing::{info, warn};

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
        sessions.entry(session_id.clone()).or_default().push(username.clone());
    }

    info!("User {username} joined session {session_id}");

    // Broadcast join message
    broadcast_to_session(
        &state,
        &session_id,
        &serde_json::json!({
            "type": "join",
            "username": username,
        }).to_string(),
        None,
    ).await;

    // Forward messages from this peer to others
    while let Some(Ok(msg)) = receiver.next().await {
        if let Message::Text(text) = msg {
            broadcast_to_session(
                &state,
                &session_id,
                &text,
                Some(&username),
            ).await;
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
        }).to_string(),
        None,
    ).await;
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
