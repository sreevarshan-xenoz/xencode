use axum::extract::ws::{Message, WebSocket};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

/// Messages buffered per peer before it is considered unable to keep up.
///
/// A peer that stops reading fills its buffer and is then dropped, rather than
/// growing a queue without limit. Broadcasts are small JSON frames, so this is
/// a generous allowance for a client that is merely slow.
const PEER_SEND_BUFFER: usize = 256;

/// Each peer is addressed through its own channel, drained by a dedicated
/// writer task that owns the socket's sink. Broadcasting therefore never waits
/// on a socket, only on the channel, which is what keeps one stalled client
/// from holding up every other session.
pub type PeerMap = Arc<Mutex<HashMap<String, Vec<mpsc::Sender<Message>>>>>;

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

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle an incoming WebSocket connection.
pub async fn handle_socket(
    socket: WebSocket,
    session_id: String,
    username: String,
    state: Arc<AppState>,
) {
    let (mut sink, mut receiver) = socket.split();

    // One writer task per peer owns the sink; everyone else reaches this peer
    // through `tx`. The task ends when every sender is dropped, i.e. once the
    // peer is gone from the map and this function has returned.
    let (tx, mut rx) = mpsc::channel::<Message>(PEER_SEND_BUFFER);
    tokio::spawn(async move {
        use futures_util::SinkExt;
        while let Some(msg) = rx.recv().await {
            if let Err(e) = sink.send(msg).await {
                warn!("Failed to send to peer: {e}");
                break;
            }
        }
        let _ = sink.close().await;
    });

    // Register the peer
    {
        let mut peers = state.peers.lock().await;
        peers.entry(session_id.clone()).or_default().push(tx);
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
    let msg = Message::Text(message.to_string().into());
    let mut peers = state.peers.lock().await;
    if let Some(senders) = peers.get_mut(session_id) {
        // `try_send` never waits, so the lock is not held across a socket write.
        // A peer is dropped when its receiver is gone (disconnected) or its
        // buffer is full (not keeping up) — in both cases there is nothing
        // useful left to do with it.
        senders.retain(|tx| match tx.try_send(msg.clone()) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                warn!("Dropping peer in session {session_id}: send buffer full");
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => false,
        });
    }
}

/// Remove a peer from the session on disconnect.
async fn remove_peer(state: &AppState, session_id: &str, username: &str) {
    let mut sessions = state.sessions.lock().await;
    if let Some(users) = sessions.get_mut(session_id) {
        users.retain(|u| u != username);
    }
}

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

    fn text_of(msg: &Message) -> String {
        match msg {
            Message::Text(t) => t.to_string(),
            other => panic!("expected a text frame, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn broadcast_reaches_every_peer_in_the_session() {
        let state = Arc::new(AppState::new());
        let (tx_a, mut rx_a) = mpsc::channel(8);
        let (tx_b, mut rx_b) = mpsc::channel(8);
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![tx_a, tx_b]);

        broadcast_to_session(&state, "s", "hello", None).await;

        assert_eq!(text_of(&rx_a.recv().await.unwrap()), "hello");
        assert_eq!(text_of(&rx_b.recv().await.unwrap()), "hello");
    }

    /// The regression this guards: a peer that has stopped reading must not
    /// hold up delivery to anyone else, and must not stall the broadcast.
    #[tokio::test]
    async fn a_stalled_peer_does_not_block_the_broadcast_or_other_peers() {
        let state = Arc::new(AppState::new());

        // A peer whose receiver exists but is never drained, with its buffer
        // already full — the state a client in this position reaches once it
        // stops reading from its socket.
        let (stalled_tx, _stalled_rx) = mpsc::channel::<Message>(1);
        stalled_tx
            .send(Message::Text("filler".into()))
            .await
            .unwrap();

        let (healthy_tx, mut healthy_rx) = mpsc::channel(8);
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![stalled_tx, healthy_tx]);

        // Completes without waiting on the stalled peer.
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            broadcast_to_session(&state, "s", "hello", None),
        )
        .await
        .expect("broadcast blocked on a stalled peer");

        // The healthy peer still got it...
        assert_eq!(text_of(&healthy_rx.recv().await.unwrap()), "hello");
        // ...and the stalled one was dropped rather than kept around.
        assert_eq!(state.peers.lock().await["s"].len(), 1);
    }

    #[tokio::test]
    async fn broadcast_drops_a_disconnected_peer() {
        let state = Arc::new(AppState::new());
        let (gone_tx, gone_rx) = mpsc::channel::<Message>(8);
        drop(gone_rx); // peer disconnected, its writer task is finished
        let (live_tx, _live_rx) = mpsc::channel::<Message>(8);
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![gone_tx, live_tx]);

        broadcast_to_session(&state, "s", "hello", None).await;

        assert_eq!(state.peers.lock().await["s"].len(), 1);
    }

    /// The lock must not be held across a socket write: a broadcast to a
    /// session with a stalled peer must not delay a broadcast to a different
    /// session.
    #[tokio::test]
    async fn a_stalled_peer_does_not_block_a_different_session() {
        let state = Arc::new(AppState::new());

        let (stalled_tx, _stalled_rx) = mpsc::channel::<Message>(1);
        stalled_tx
            .send(Message::Text("filler".into()))
            .await
            .unwrap();
        let (other_tx, mut other_rx) = mpsc::channel(8);

        {
            let mut peers = state.peers.lock().await;
            peers.insert("stalled-session".to_string(), vec![stalled_tx]);
            peers.insert("other-session".to_string(), vec![other_tx]);
        }

        broadcast_to_session(&state, "stalled-session", "a", None).await;

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            broadcast_to_session(&state, "other-session", "b", None),
        )
        .await
        .expect("a stalled peer in one session blocked another session");

        assert_eq!(text_of(&other_rx.recv().await.unwrap()), "b");
    }
}
