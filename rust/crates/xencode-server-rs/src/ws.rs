use crate::tokens::TokenStore;
use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{info, warn};
use xencode_collaboration_rs::wire::{
    ClientFrame, MemberInfo, ServerFrame, CLOSE_BAD_TOKEN, CLOSE_NO_SESSION, CLOSE_RBAC_DENIED,
    CLOSE_SESSION_FULL, MAX_SESSION_MEMBERS,
};
use xencode_collaboration_rs::{Role, SyncCoordinator, WorkspaceManager};

/// Messages buffered per peer before it is considered unable to keep up.
///
/// A peer that stops reading fills its buffer and is then dropped, rather than
/// growing a queue without limit. Broadcasts are small JSON frames, so this is
/// a generous allowance for a client that is merely slow.
const PEER_SEND_BUFFER: usize = 256;

/// How long a freshly-upgraded connection gets to send its `auth` frame.
/// Anything else — silence, garbage, a non-auth frame — is closed 4401.
const AUTH_TIMEOUT: Duration = Duration::from_secs(5);

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
/// connection's id and the identity it authenticated as.
pub struct Peer {
    pub id: PeerId,
    pub username: String,
    /// The `SyncCoordinator` presence record for this connection.
    pub sync_id: String,
    tx: mpsc::Sender<Message>,
}

/// Each peer is addressed through its own channel, drained by a dedicated
/// writer task that owns the socket's sink. Broadcasting therefore never waits
/// on a socket, only on the channel, which is what keeps one stalled client
/// from holding up every other session.
pub type PeerMap = Arc<Mutex<HashMap<String, Vec<Peer>>>>;

/// Server state. Two distinct membership notions live here on purpose:
/// `workspaces` says who *belongs* to a session (roles survive disconnects),
/// `sync` says who is *connected right now* (presence). The old single
/// `sessions` map conflated them and was writable by anyone who merely
/// opened a socket with a chosen username.
pub struct AppState {
    pub peers: PeerMap,
    pub workspaces: Arc<Mutex<WorkspaceManager>>,
    pub sync: Arc<Mutex<SyncCoordinator>>,
    pub tokens: Arc<Mutex<TokenStore>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            peers: Arc::new(Mutex::new(HashMap::new())),
            workspaces: Arc::new(Mutex::new(WorkspaceManager::new())),
            sync: Arc::new(Mutex::new(SyncCoordinator::new())),
            tokens: Arc::new(Mutex::new(TokenStore::new())),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Refuse a connection that never became a peer: one explanatory `error`
/// frame, then a close with a protocol-defined code.
async fn reject(
    sink: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    code: u16,
    reason: &str,
) {
    use futures_util::SinkExt;
    let frame = match code {
        CLOSE_BAD_TOKEN => "bad_token",
        CLOSE_NO_SESSION => "no_session",
        CLOSE_SESSION_FULL => "session_full",
        CLOSE_RBAC_DENIED => "rbac_denied",
        _ => "error",
    };
    let _ = sink
        .send(Message::Text(
            ServerFrame::error(frame, reason).to_json().into(),
        ))
        .await;
    let _ = sink
        .send(Message::Close(Some(CloseFrame {
            code,
            reason: reason.to_string().into(),
        })))
        .await;
    let _ = sink.close().await;
}

/// Handle an incoming WebSocket connection.
///
/// Identity comes from the first frame's token, never from the URL: a
/// username a caller types into a path argument proves nothing.
pub async fn handle_socket(socket: WebSocket, session_id: String, state: Arc<AppState>) {
    let (mut sink, mut receiver) = socket.split();

    // ---- authentication phase (before any peer exists) ----
    let outcome: Result<String, (u16, &'static str)> =
        match tokio::time::timeout(AUTH_TIMEOUT, receiver.next()).await {
            Ok(Some(Ok(Message::Text(text)))) => match ClientFrame::parse(&text) {
                Ok(ClientFrame::Auth { token }) => Ok(token),
                Ok(_) => Err((CLOSE_BAD_TOKEN, "first frame must be auth")),
                Err(_) => Err((CLOSE_BAD_TOKEN, "missing or malformed auth frame")),
            },
            // The client vanished before authenticating: nothing left to send.
            Ok(None) | Ok(Some(Err(_))) | Ok(Some(Ok(Message::Close(_)))) => return,
            Ok(Some(Ok(_))) => Err((CLOSE_BAD_TOKEN, "binary frame before auth")),
            Err(_) => Err((CLOSE_BAD_TOKEN, "auth timeout")),
        };
    let token = match outcome {
        Ok(token) => token,
        Err((code, reason)) => {
            reject(&mut sink, code, reason).await;
            return;
        }
    };
    let principal = {
        let mut tokens = state.tokens.lock().await;
        tokens.lookup(&token)
    };
    let Some(principal) = principal else {
        reject(&mut sink, CLOSE_BAD_TOKEN, "invalid or expired token").await;
        return;
    };
    let username = principal.username;

    // RBAC gate: the session must exist, and admitting a new identity must
    // not exceed the shared size cap. Joining is idempotent — a reconnect
    // keeps its role.
    let role = {
        let mut workspaces = state.workspaces.lock().await;
        if workspaces.get_workspace(&session_id).is_none() {
            drop(workspaces);
            reject(&mut sink, CLOSE_NO_SESSION, "no such session").await;
            return;
        }
        let known = workspaces.role_of(&session_id, &username).is_some();
        if !known && workspaces.member_count(&session_id) >= MAX_SESSION_MEMBERS {
            drop(workspaces);
            reject(&mut sink, CLOSE_SESSION_FULL, "session is full").await;
            return;
        }
        match workspaces.join(&session_id, &username) {
            Ok(role) => role,
            Err(_) => {
                drop(workspaces);
                reject(&mut sink, CLOSE_NO_SESSION, "no such session").await;
                return;
            }
        }
    };

    info!("{username} ({role}) authenticated for session {session_id}");

    // ---- peer phase ----
    let (tx, mut rx) = mpsc::channel::<Message>(PEER_SEND_BUFFER);
    let sync_id = state
        .sync
        .lock()
        .await
        .join_session(&session_id, &username)
        .peer_id;
    let peer_id = PeerId::next();
    {
        let mut peers = state.peers.lock().await;
        peers.entry(session_id.clone()).or_default().push(Peer {
            id: peer_id,
            username: username.clone(),
            sync_id: sync_id.clone(),
            tx: tx.clone(),
        });
    }

    // The writer task owns the sink from here on; the auth phase is over.
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

    let members = presence(&state, &session_id).await;
    let _ = tx
        .send(Message::Text(
            ServerFrame::AuthOk {
                username: username.clone(),
                role: role.to_string(),
                session_id: session_id.clone(),
                members: members.clone(),
            }
            .to_json()
            .into(),
        ))
        .await;
    broadcast_frame(
        &state,
        &session_id,
        &ServerFrame::Members { members },
        Some(peer_id),
    )
    .await;

    // ---- message loop ----
    while let Some(Ok(msg)) = receiver.next().await {
        let Message::Text(text) = msg else {
            continue;
        };
        match ClientFrame::parse(&text) {
            Ok(ClientFrame::Activity { text: body }) => {
                let allowed = {
                    let mut workspaces = state.workspaces.lock().await;
                    match workspaces.role_of(&session_id, &username) {
                        Some(role) if role.can(&Role::Editor) => true,
                        other => {
                            // The denial belongs in the audit trail whoever
                            // the actor is; roles can change mid-session.
                            workspaces.log_denied(
                                &username,
                                &session_id,
                                "relay activity",
                                other.as_ref(),
                            );
                            false
                        }
                    }
                };
                if !allowed {
                    let _ = tx
                        .send(Message::Text(
                            ServerFrame::error(
                                "rbac_denied",
                                "viewers cannot send activity to the session",
                            )
                            .to_json()
                            .into(),
                        ))
                        .await;
                    continue;
                }
                // `user` is set from server state, never from the frame.
                let relay = ServerFrame::Activity {
                    user: username.clone(),
                    text: body,
                };
                broadcast_frame(&state, &session_id, &relay, Some(peer_id)).await;
            }
            Ok(ClientFrame::Ping) => {
                let _ = tx
                    .send(Message::Text(ServerFrame::Pong.to_json().into()))
                    .await;
            }
            // Auth mid-session, or anything unparseable: answer, keep the
            // connection, make no state changes.
            Ok(ClientFrame::Auth { .. }) => {
                let _ = tx
                    .send(Message::Text(
                        ServerFrame::error("already_authenticated", "send activity or ping")
                            .to_json()
                            .into(),
                    ))
                    .await;
            }
            Err(_) => {
                let _ = tx
                    .send(Message::Text(
                        ServerFrame::error("bad_frame", "expected a tagged JSON frame")
                            .to_json()
                            .into(),
                    ))
                    .await;
            }
        }
    }

    // ---- disconnect ----
    info!("{username} left session {session_id}");
    remove_connection(&state, &session_id, peer_id).await;
    let members = presence(&state, &session_id).await;
    broadcast_frame(&state, &session_id, &ServerFrame::Members { members }, None).await;
}

/// Remove one connection (not the whole identity) from the peer and presence
/// maps. Dropping its sender ends the writer task.
async fn remove_connection(state: &AppState, session_id: &str, peer_id: PeerId) {
    let sync_id = {
        let mut peers = state.peers.lock().await;
        let Some(session_peers) = peers.get_mut(session_id) else {
            return;
        };
        let mut removed = None;
        session_peers.retain(|p| {
            if p.id == peer_id {
                removed = Some(p.sync_id.clone());
                false
            } else {
                true
            }
        });
        if session_peers.is_empty() {
            peers.remove(session_id);
        }
        removed
    };
    if let Some(sync_id) = sync_id {
        state.sync.lock().await.leave_session(session_id, &sync_id);
    }
}

/// Who is connected to a session right now, deduplicated by username, with
/// each identity's current workspace role.
///
/// Lock order is `peers` then `workspaces`; every path that takes both
/// follows it.
async fn presence(state: &AppState, session_id: &str) -> Vec<MemberInfo> {
    let usernames: Vec<String> = {
        let peers = state.peers.lock().await;
        let mut seen: Vec<String> = Vec::new();
        if let Some(session_peers) = peers.get(session_id) {
            for peer in session_peers {
                if !seen.contains(&peer.username) {
                    seen.push(peer.username.clone());
                }
            }
        }
        seen
    };
    let workspaces = state.workspaces.lock().await;
    usernames
        .into_iter()
        .map(|username| {
            let role = workspaces
                .role_of(session_id, &username)
                .unwrap_or(Role::Viewer)
                .to_string();
            MemberInfo { username, role }
        })
        .collect()
}

/// Broadcast a server frame to every peer in a session, skipping `exclude`.
async fn broadcast_frame(
    state: &AppState,
    session_id: &str,
    frame: &ServerFrame,
    exclude: Option<PeerId>,
) {
    broadcast_to_session(state, session_id, &frame.to_json(), exclude).await;
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_app_state_new() {
        let state = AppState::new();
        assert!(state.peers.lock().await.is_empty());
        assert_eq!(state.workspaces.lock().await.workspace_count(), 0);
        assert!(state.sync.lock().await.get_active_sessions().is_empty());
        let mut tokens = state.tokens.lock().await;
        assert!(tokens.lookup("xencode_nothing").is_none());
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
            username: "tester".to_string(),
            sync_id: uuid::Uuid::new_v4().to_string(),
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
            Duration::from_secs(5),
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
            Duration::from_secs(5),
            broadcast_to_session(&state, "other-session", "b", None),
        )
        .await
        .expect("a stalled peer in one session blocked another session");

        assert_eq!(text_of(&other_rx.recv().await.unwrap()), "b");
    }

    /// Register a peer's presence the way `handle_socket` does, and hand back
    /// the sync id so the test's `Peer` carries the same one.
    async fn join_presence(state: &AppState, session: &str, username: &str) -> String {
        state
            .sync
            .lock()
            .await
            .join_session(session, username)
            .peer_id
    }

    #[tokio::test]
    async fn remove_connection_drops_only_that_one_peer() {
        let state = Arc::new(AppState::new());
        let (mut first, _first_rx) = test_peer(8);
        let (mut second, _second_rx) = test_peer(8);
        first.sync_id = join_presence(&state, "s", "tester").await;
        second.sync_id = join_presence(&state, "s", "other").await;
        let (first_id, second_id) = (first.id, second.id);
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![first, second]);

        remove_connection(&state, "s", first_id).await;

        let peers = state.peers.lock().await;
        assert_eq!(peers["s"].len(), 1);
        assert_eq!(peers["s"][0].id, second_id);
        drop(peers);
        // Presence saw exactly this connection leave.
        let sync = state.sync.lock().await;
        let peers = sync.get_peers("s");
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].username, "other");
    }

    #[tokio::test]
    async fn removing_the_last_connection_clears_the_session_entry() {
        let state = Arc::new(AppState::new());
        let (mut only, _rx) = test_peer(8);
        only.sync_id = join_presence(&state, "s", "tester").await;
        let only_id = only.id;
        state.peers.lock().await.insert("s".to_string(), vec![only]);

        remove_connection(&state, "s", only_id).await;

        assert!(!state.peers.lock().await.contains_key("s"));
        assert!(!state.sync.lock().await.session_has_peers("s"));
    }

    #[tokio::test]
    async fn presence_deduplicates_by_username_and_reports_roles() {
        let state = Arc::new(AppState::new());
        let (first, _rx_a) = test_peer(8);
        let (second, _rx_b) = test_peer(8);
        state
            .peers
            .lock()
            .await
            .insert("s".to_string(), vec![first, second]);
        state
            .workspaces
            .lock()
            .await
            .create_workspace_with_id("s", "s", "tester");

        let members = presence(&state, "s").await;
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].username, "tester");
        // "tester" created the workspace, so the two connections share one
        // admin identity rather than appearing twice.
        assert_eq!(members[0].role, "admin");
    }
}
