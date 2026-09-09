use axum::extract::ws::{Message, WebSocket};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};

/// Messages buffered per peer before it is considered unable to keep up.
///
/// A peer that stops reading fills its buffer and is then dropped, rather than
/// growing a queue without limit. Broadcasts are small JSON frames, so this is
/// a generous allowance for a client that is merely slow.
const PEER_SEND_BUFFER: usize = 256;

/// Identifies one WebSocket connection, so a broadcast can skip the connection
/// it came from.
///
/// Deliberately per-connection rather than per-user: a user with the editor
/// open in two tabs must still see their own edits arrive in the other tab, so
/// the username is too coarse to exclude on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerId(u64);

static NEXT_PEER_ID: AtomicU64 = AtomicU64::new(0);

impl PeerId {
    fn next() -> Self {
        Self(NEXT_PEER_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// One connected peer: the channel its writer task drains, tagged with the
/// connection's id.
pub struct Peer {
    pub id: PeerId,
    tx: mpsc::Sender<Message>,
}

/// Each peer is addressed through its own channel, drained by a dedicated
/// writer task that owns the socket's sink. Broadcasting therefore never waits
/// on a socket, only on the channel, which is what keeps one stalled client
/// from holding up every other session.
pub type PeerMap = Arc<Mutex<HashMap<String, Vec<Peer>>>>;

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
    let peer_id = PeerId::next();
    {
        let mut peers = state.peers.lock().await;
        peers
            .entry(session_id.clone())
            .or_default()
            .push(Peer { id: peer_id, tx });
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
            broadcast_to_session(&state, &session_id, &text, Some(peer_id)).await;
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

/// Broadcast a message to every peer in a session, skipping `exclude` if given.
///
/// `exclude` is the connection a message arrived on, so the sender is not sent
/// its own message back.
async fn broadcast_to_session(
    state: &AppState,
    session_id: &str,
    message: &str,
    exclude: Option<PeerId>,
) {
    let msg = Message::Text(message.to_string().into());
    let mut peers = state.peers.lock().await;
    if let Some(session_peers) = peers.get_mut(session_id) {
        // `try_send` never waits, so the lock is not held across a socket write.
        // A peer is dropped when its receiver is gone (disconnected) or its
        // buffer is full (not keeping up) — in both cases there is nothing
        // useful left to do with it. An excluded peer is skipped, not dropped.
        session_peers.retain(|peer| {
            if Some(peer.id) == exclude {
                return true;
            }
            match peer.tx.try_send(msg.clone()) {
                Ok(()) => true,
                Err(mpsc::error::TrySendError::Full(_)) => {
                    warn!("Dropping peer in session {session_id}: send buffer full");
                    false
                }
                Err(mpsc::error::TrySendError::Closed(_)) => false,
            }
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

    /// A peer with its own id, plus the receiver its writer task would drain.
    fn test_peer(capacity: usize) -> (Peer, mpsc::Receiver<Message>) {
        let (tx, rx) = mpsc::channel(capacity);
        let peer = Peer {
            id: PeerId::next(),
            tx,
        };
        (peer, rx)
    }

    #[tokio::test]
    async fn broadcast_reaches_every_peer_in_the_session() {
        let state = Arc::new(AppState::new());
        let (peer_a, mut rx_a) = test_peer(8);
        let (peer_b, mut rx_b) = test_peer(8);
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![peer_a, peer_b]);

        broadcast_to_session(&state, "s", "hello", None).await;

        assert_eq!(text_of(&rx_a.recv().await.unwrap()), "hello");
        assert_eq!(text_of(&rx_b.recv().await.unwrap()), "hello");
    }

    /// The regression this guards: the sending connection must not be sent its
    /// own message back.
    #[tokio::test]
    async fn broadcast_skips_the_excluded_connection() {
        let state = Arc::new(AppState::new());
        let (sender, mut sender_rx) = test_peer(8);
        let (other, mut other_rx) = test_peer(8);
        let sender_id = sender.id;
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![sender, other]);

        broadcast_to_session(&state, "s", "hello", Some(sender_id)).await;

        assert_eq!(text_of(&other_rx.recv().await.unwrap()), "hello");
        assert!(
            sender_rx.try_recv().is_err(),
            "the sending connection was echoed its own message"
        );
    }

    /// Exclusion is per-connection, not per-user: a second tab belonging to the
    /// same person is a different connection and must still receive the message.
    #[tokio::test]
    async fn broadcast_still_reaches_the_senders_other_connections() {
        let state = Arc::new(AppState::new());
        let (first_tab, mut first_rx) = test_peer(8);
        let (second_tab, mut second_rx) = test_peer(8);
        let first_id = first_tab.id;
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![first_tab, second_tab]);

        broadcast_to_session(&state, "s", "hello", Some(first_id)).await;

        assert_eq!(text_of(&second_rx.recv().await.unwrap()), "hello");
        assert!(first_rx.try_recv().is_err());
    }

    /// An excluded peer is skipped, not sent to — so a full buffer must not
    /// get it pruned. (A test that merely checks an excluded peer survives a
    /// broadcast would pass without the fix too, since a peer with room is
    /// never pruned either way; the full buffer is what makes this discriminate.)
    #[tokio::test]
    async fn an_excluded_peer_with_a_full_buffer_is_kept() {
        let state = Arc::new(AppState::new());
        let (stalled, _stalled_rx) = test_peer(1);
        stalled
            .tx
            .send(Message::Text("filler".into()))
            .await
            .unwrap();
        let stalled_id = stalled.id;
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![stalled]);

        broadcast_to_session(&state, "s", "hello", Some(stalled_id)).await;

        let peers = state.peers.lock().await;
        assert_eq!(peers["s"].len(), 1);
        assert_eq!(peers["s"][0].id, stalled_id);
    }

    #[tokio::test]
    async fn peer_ids_are_unique() {
        let ids: std::collections::HashSet<u64> = (0..1000).map(|_| PeerId::next().0).collect();
        assert_eq!(ids.len(), 1000);
    }

    /// The regression this guards: a peer that has stopped reading must not
    /// hold up delivery to anyone else, and must not stall the broadcast.
    #[tokio::test]
    async fn a_stalled_peer_does_not_block_the_broadcast_or_other_peers() {
        let state = Arc::new(AppState::new());

        // A peer whose receiver exists but is never drained, with its buffer
        // already full — the state a client in this position reaches once it
        // stops reading from its socket.
        let (stalled, _stalled_rx) = test_peer(1);
        stalled
            .tx
            .send(Message::Text("filler".into()))
            .await
            .unwrap();

        let (healthy, mut healthy_rx) = test_peer(8);
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![stalled, healthy]);

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
        let (gone, gone_rx) = test_peer(8);
        drop(gone_rx); // peer disconnected, its writer task is finished
        let (live, _live_rx) = test_peer(8);
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![gone, live]);

        broadcast_to_session(&state, "s", "hello", None).await;

        assert_eq!(state.peers.lock().await["s"].len(), 1);
    }

    /// The lock must not be held across a socket write: a broadcast to a
    /// session with a stalled peer must not delay a broadcast to a different
    /// session.
    #[tokio::test]
    async fn a_stalled_peer_does_not_block_a_different_session() {
        let state = Arc::new(AppState::new());

        let (stalled, _stalled_rx) = test_peer(1);
        stalled
            .tx
            .send(Message::Text("filler".into()))
            .await
            .unwrap();
        let (other, mut other_rx) = test_peer(8);

        {
            let mut peers = state.peers.lock().await;
            peers.insert("stalled-session".to_string(), vec![stalled]);
            peers.insert("other-session".to_string(), vec![other]);
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
