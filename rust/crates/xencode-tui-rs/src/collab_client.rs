//! The Collaboration Hub's real client: log in, open the WebSocket, and
//! translate server frames into the `[COLLAB]` token grammar the app event
//! loop already ingests. No simulated teammates, no fabricated telemetry.
//!
//! There is deliberately no auto-reconnect: a hub session is a decision,
//! not a daemon. The user retries explicitly (G3-02 wires the keys).

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;
use xencode_collaboration_rs::wire::{
    ClientFrame, ServerFrame, CLOSE_BAD_TOKEN, CLOSE_NO_SESSION, CLOSE_RBAC_DENIED,
    CLOSE_SESSION_FULL,
};

/// How often the client pings the server to prove the link is alive.
const KEEPALIVE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);
/// No inbound frame within this window (including ping replies) means the
/// connection is gone; the task exits rather than hanging the hub.
const LIVENESS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// Translate one server frame into `[COLLAB]` tokens. Pure, so the whole
/// grammar — including hostile input — is testable without a socket.
pub fn frame_to_tokens(raw: &str) -> Vec<String> {
    let frame: Result<ServerFrame, serde_json::Error> = serde_json::from_str(raw);
    match frame {
        Ok(ServerFrame::AuthOk {
            username,
            role,
            session_id,
            members,
        }) => {
            let mut tokens = vec![
                "[COLLAB]status:connected".to_string(),
                format!("[COLLAB]session:{session_id}"),
                members_token(&members),
                format!("[COLLAB]log:🔗 Connected as {username} ({role}) — session {session_id}"),
                "[COLLAB]ready".to_string(),
            ];
            if members.len() > 1 {
                tokens.push(format!(
                    "[COLLAB]log:👥 {} members present",
                    members.len() - 1
                ));
            }
            tokens
        }
        Ok(ServerFrame::Members { members }) => vec![members_token(&members)],
        Ok(ServerFrame::Activity { user, text }) => vec![format!("[COLLAB]log:{user}: {text}")],
        Ok(ServerFrame::Error { code, message }) => vec![
            format!("[COLLAB]error:{code}: {message}"),
            format!("[COLLAB]log:⚠ {code}: {message}"),
        ],
        // The reply to our own keepalive pings; liveness is refreshed in the
        // read loop, so nothing reaches the UI.
        Ok(ServerFrame::Pong) => Vec::new(),
        Err(_) => {
            // Name what actually arrived instead of a bare "malformed": with
            // an unknown `type` the honest complaint is different from broken
            // JSON.
            let summary = serde_json::from_str::<serde_json::Value>(raw)
                .ok()
                .and_then(|v| {
                    v.get("type")
                        .and_then(|t| t.as_str())
                        .map(|t| format!("unknown frame type '{t}'"))
                })
                .unwrap_or_else(|| "malformed frame from server".to_string());
            vec![
                format!("[COLLAB]error:{summary}"),
                format!("[COLLAB]log:⚠ {summary}"),
            ]
        }
    }
}

fn members_token(members: &[xencode_collaboration_rs::wire::MemberInfo]) -> String {
    // One tag replaces the old per-member grammar: the server's `members`
    // frames are complete snapshots, so the app swaps the list wholesale.
    format!(
        "[COLLAB]members:{}",
        serde_json::to_string(members).unwrap_or_else(|_| "[]".to_string())
    )
}

/// Human text for the application close codes the server defines.
pub fn close_reason_text(code: u16) -> &'static str {
    match code {
        CLOSE_BAD_TOKEN => "rejected: invalid or expired token",
        CLOSE_RBAC_DENIED => "rejected: not permitted in this session",
        CLOSE_NO_SESSION => "rejected: no such session",
        CLOSE_SESSION_FULL => "rejected: session is full",
        _ => "connection closed by server",
    }
}

/// Map a server URL (`http(s)://…`, what the user types) to the WebSocket
/// URL. No DNS or connection happens here — just scheme honesty.
pub fn ws_url(server_url: &str) -> Result<String, String> {
    let trimmed = server_url.trim_end_matches('/');
    if let Some(rest) = trimmed.strip_prefix("https://") {
        Ok(format!("wss://{rest}"))
    } else if let Some(rest) = trimmed.strip_prefix("http://") {
        Ok(format!("ws://{rest}"))
    } else {
        Err(format!(
            "server URL must start with http:// or https://, got '{server_url}'"
        ))
    }
}

fn send_all(tx: &mpsc::UnboundedSender<String>, tokens: Vec<String>) {
    for token in tokens {
        let _ = tx.send(token);
    }
}

/// Run one authenticated session over an already-connected WebSocket:
/// send the `auth` frame first, then translate every server frame until the
/// link ends. Kept separate from `spawn_collab_worker` so tests can drive a
/// bare duplex stream.
pub async fn run_session<S>(
    mut ws: WebSocketStream<S>,
    token: &str,
    tx: &mpsc::UnboundedSender<String>,
) where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let auth = ClientFrame::Auth {
        token: token.to_string(),
    };
    let auth_json = serde_json::to_string(&auth).unwrap_or_else(|_| "{}".to_string());
    if ws.send(Message::Text(auth_json.into())).await.is_err() {
        send_all(
            tx,
            vec![
                "[COLLAB]error:could not send the auth frame".to_string(),
                "[COLLAB]status:disconnected".to_string(),
            ],
        );
        return;
    }

    let mut keepalive = tokio::time::interval(KEEPALIVE_INTERVAL);
    // The first tick fires immediately; the auth frame already proved the
    // link, so skip it.
    keepalive.tick().await;
    let mut last_inbound = tokio::time::Instant::now();

    loop {
        tokio::select! {
            inbound = ws.next() => match inbound {
                Some(Ok(Message::Text(text))) => {
                    last_inbound = tokio::time::Instant::now();
                    send_all(tx, frame_to_tokens(text.as_ref()));
                }
                Some(Ok(Message::Ping(payload))) => {
                    last_inbound = tokio::time::Instant::now();
                    let _ = ws.send(Message::Pong(payload)).await;
                }
                Some(Ok(Message::Pong(_))) => {
                    last_inbound = tokio::time::Instant::now();
                }
                Some(Ok(Message::Close(frame))) => {
                    let code = frame.map(|f| u16::from(f.code)).unwrap_or(1000);
                    send_all(
                        tx,
                        vec![
                            format!("[COLLAB]error:{}", close_reason_text(code)),
                            format!("[COLLAB]log:⚠ {}", close_reason_text(code)),
                            "[COLLAB]status:disconnected".to_string(),
                        ],
                    );
                    return;
                }
                Some(Ok(Message::Binary(_) | Message::Frame(_))) => {
                    last_inbound = tokio::time::Instant::now();
                }
                Some(Err(e)) => {
                    send_all(
                        tx,
                        vec![
                            format!("[COLLAB]error:connection failed: {e}"),
                            "[COLLAB]status:disconnected".to_string(),
                        ],
                    );
                    return;
                }
                None => {
                    send_all(tx, vec!["[COLLAB]status:disconnected".to_string()]);
                    return;
                }
            },
            _ = keepalive.tick() => {
                if last_inbound.elapsed() > LIVENESS_TIMEOUT {
                    send_all(
                        tx,
                        vec![
                            "[COLLAB]error:server stopped responding".to_string(),
                            "[COLLAB]log:⚠ server stopped responding".to_string(),
                            "[COLLAB]status:disconnected".to_string(),
                        ],
                    );
                    return;
                }
                let _ = ws.send(Message::Ping(Vec::new().into())).await;
            }
        }
    }
}

/// Log in and open the session, reporting everything through `[COLLAB]`
/// tokens. Returns the task handle so the hub can abort it on `Esc`.
///
/// An empty `session` asks the server to create one — the id then arrives
/// in the `auth_ok` frame, so the hub learns what it is joined to rather
/// than inventing it.
pub fn spawn_collab_worker(
    server_url: String,
    session: String,
    username: String,
    tx: mpsc::UnboundedSender<String>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let token = match login(&server_url, &username).await {
            Ok(token) => token,
            Err(e) => {
                send_all(
                    &tx,
                    vec![
                        format!("[COLLAB]error:login failed: {e}"),
                        format!("[COLLAB]log:⚠ Login failed: {e}"),
                        "[COLLAB]status:disconnected".to_string(),
                    ],
                );
                return;
            }
        };
        let session = if session.is_empty() {
            match create_session(&server_url, &token).await {
                Ok(id) => id,
                Err(e) => {
                    send_all(
                        &tx,
                        vec![
                            format!("[COLLAB]error:session creation failed: {e}"),
                            format!("[COLLAB]log:⚠ Session creation failed: {e}"),
                            "[COLLAB]status:disconnected".to_string(),
                        ],
                    );
                    return;
                }
            }
        } else {
            session
        };
        let url = match ws_url(&server_url).map(|base| format!("{base}/ws/{session}")) {
            Ok(url) => url,
            Err(e) => {
                send_all(
                    &tx,
                    vec![
                        format!("[COLLAB]error:{e}"),
                        "[COLLAB]status:disconnected".to_string(),
                    ],
                );
                return;
            }
        };
        let (ws, _response) = match tokio_tungstenite::connect_async(&url).await {
            Ok(pair) => pair,
            Err(e) => {
                send_all(
                    &tx,
                    vec![
                        format!("[COLLAB]error:could not reach {server_url}: {e}"),
                        format!("[COLLAB]log:⚠ Could not reach {server_url}: {e}"),
                        "[COLLAB]status:disconnected".to_string(),
                    ],
                );
                return;
            }
        };
        run_session(ws, &token, &tx).await;
    })
}

/// `POST /auth/login` — an identity claim, not a password check (the trust
/// model is documented in the CLI guide). Returns the session token.
async fn login(server_url: &str, username: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("{}/auth/login", server_url.trim_end_matches('/'));
    let response = client
        .post(&url)
        .json(&serde_json::json!({ "username": username }))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("server answered {status}"));
    }
    let body: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    body["token"]
        .as_str()
        .map(|t| t.to_string())
        .ok_or_else(|| "response carried no token".to_string())
}

/// `POST /sessions/create` — the server picks the id and makes this
/// identity its admin.
async fn create_session(server_url: &str, token: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;
    let url = format!("{}/sessions/create", server_url.trim_end_matches('/'));
    let response = client
        .post(url)
        .header("authorization", format!("Bearer {token}"))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("server answered {status}"));
    }
    let body: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    body["id"]
        .as_str()
        .map(|id| id.to_string())
        .ok_or_else(|| "response carried no session id".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
    use tokio_tungstenite::tungstenite::protocol::CloseFrame;
    use xencode_collaboration_rs::wire::MemberInfo;

    fn tokens_of(frames: &[ServerFrame]) -> Vec<String> {
        frames
            .iter()
            .flat_map(|f| frame_to_tokens(&f.to_json()))
            .collect()
    }

    fn member(name: &str, role: &str) -> MemberInfo {
        MemberInfo {
            username: name.to_string(),
            role: role.to_string(),
        }
    }

    fn auth_ok() -> ServerFrame {
        ServerFrame::AuthOk {
            username: "alice".to_string(),
            role: "editor".to_string(),
            session_id: "xencode-1234".to_string(),
            members: vec![member("alice", "editor"), member("bob", "viewer")],
        }
    }

    #[test]
    fn auth_ok_yields_the_full_connected_sequence() {
        let tokens = frame_to_tokens(&auth_ok().to_json());
        assert_eq!(tokens[0], "[COLLAB]status:connected");
        assert_eq!(tokens[1], "[COLLAB]session:xencode-1234");
        assert!(tokens[2].starts_with("[COLLAB]members:["));
        assert!(tokens[2].contains("\"username\":\"alice\""));
        assert!(tokens[2].contains("\"role\":\"editor\""));
        assert_eq!(
            tokens[3],
            "[COLLAB]log:🔗 Connected as alice (editor) — session xencode-1234"
        );
        assert_eq!(tokens[4], "[COLLAB]ready");
        // With one other member present, the count names them, not the crowd.
        assert_eq!(
            tokens[5], "[COLLAB]log:👥 1 members present",
            "got {tokens:?}"
        );
        assert_eq!(tokens.len(), 6);
    }

    #[test]
    fn solo_auth_ok_omits_the_presence_line() {
        let frame = ServerFrame::AuthOk {
            username: "alice".to_string(),
            role: "admin".to_string(),
            session_id: "s".to_string(),
            members: vec![member("alice", "admin")],
        };
        let tokens = frame_to_tokens(&frame.to_json());
        assert_eq!(tokens.len(), 5);
        assert!(
            !tokens
                .iter()
                .any(|t| t.contains("members present") || t.contains("present")),
            "got {tokens:?}"
        );
    }

    #[test]
    fn members_frame_is_one_snapshot_tag() {
        let tokens = frame_to_tokens(
            &ServerFrame::Members {
                members: vec![member("alice", "editor"), member("bob", "admin")],
            }
            .to_json(),
        );
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].starts_with("[COLLAB]members:["));
        let json = tokens[0].trim_start_matches("[COLLAB]members:");
        let parsed: Vec<MemberInfo> = serde_json::from_str(json).unwrap();
        assert_eq!(parsed[1].username, "bob");
    }

    #[test]
    fn activity_and_error_become_log_lines() {
        let tokens = tokens_of(&[
            ServerFrame::Activity {
                user: "bob".to_string(),
                text: "hello".to_string(),
            },
            ServerFrame::error("rbac_denied", "viewers cannot relay"),
        ]);
        assert_eq!(tokens[0], "[COLLAB]log:bob: hello");
        assert_eq!(tokens[1], "[COLLAB]error:rbac_denied: viewers cannot relay");
        assert_eq!(tokens[2], "[COLLAB]log:⚠ rbac_denied: viewers cannot relay");
    }

    #[test]
    fn pong_reaches_the_ui_not_at_all() {
        assert!(frame_to_tokens(&ServerFrame::Pong.to_json()).is_empty());
    }

    #[test]
    fn malformed_frames_say_what_arrived() {
        let tokens = frame_to_tokens("not json");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], "[COLLAB]error:malformed frame from server");

        let tokens = frame_to_tokens(r#"{"type":"quantum_sync","payload":1}"#);
        assert_eq!(tokens[0], "[COLLAB]error:unknown frame type 'quantum_sync'");
        assert_eq!(tokens[1], "[COLLAB]log:⚠ unknown frame type 'quantum_sync'");
    }

    #[test]
    fn ws_url_maps_schemes_honestly() {
        assert_eq!(
            ws_url("http://127.0.0.1:8765").unwrap(),
            "ws://127.0.0.1:8765"
        );
        assert_eq!(
            ws_url("https://team.example.com/").unwrap(),
            "wss://team.example.com"
        );
        assert!(ws_url("ftp://nope").is_err());
        assert!(ws_url("").is_err());
        // Only the scheme may be a WebSocket one already: rewrite once, not
        // twice, and reject nonsense rather than guess.
        assert!(ws_url("ws://already").is_err());
    }

    #[test]
    fn close_codes_read_as_sentences() {
        assert_eq!(
            close_reason_text(CLOSE_BAD_TOKEN),
            "rejected: invalid or expired token"
        );
        assert_eq!(
            close_reason_text(CLOSE_NO_SESSION),
            "rejected: no such session"
        );
        assert_eq!(
            close_reason_text(CLOSE_SESSION_FULL),
            "rejected: session is full"
        );
        assert_eq!(
            close_reason_text(CLOSE_RBAC_DENIED),
            "rejected: not permitted in this session"
        );
        assert_eq!(close_reason_text(1006), "connection closed by server");
    }

    #[test]
    fn auth_frame_is_the_tagged_wire_shape_the_server_expects() {
        let frame = ClientFrame::Auth {
            token: "xencode_abc".to_string(),
        };
        let json = serde_json::to_string(&frame).unwrap();
        assert_eq!(json, r#"{"type":"auth","token":"xencode_abc"}"#);
    }

    /// The whole client task, driven over an in-process duplex — no sockets.
    /// The fake server checks the auth frame the way the real one does, then
    /// answers with `auth_ok` and closes with a defined application code.
    #[tokio::test]
    async fn run_session_authenticates_then_translates_until_close() {
        let (client_io, server_io) = tokio::io::duplex(4096);
        let client_ws = WebSocketStream::from_raw_socket(
            client_io,
            tokio_tungstenite::tungstenite::protocol::Role::Client,
            None,
        )
        .await;
        let mut server_ws = WebSocketStream::from_raw_socket(
            server_io,
            tokio_tungstenite::tungstenite::protocol::Role::Server,
            None,
        )
        .await;

        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let session =
            tokio::spawn(async move { run_session(client_ws, "xencode_test", &tx).await });

        let first = server_ws.next().await.expect("a frame").expect("ok");
        let parsed: ClientFrame = ClientFrame::parse(first.to_text().unwrap()).unwrap();
        assert_eq!(
            parsed,
            ClientFrame::Auth {
                token: "xencode_test".to_string()
            }
        );

        server_ws
            .send(Message::Text(auth_ok().to_json().into()))
            .await
            .unwrap();
        server_ws
            .send(Message::Close(Some(CloseFrame {
                code: CloseCode::Library(CLOSE_SESSION_FULL),
                reason: "full".into(),
            })))
            .await
            .unwrap();

        let mut tokens = Vec::new();
        while let Ok(token) = rx.try_recv() {
            tokens.push(token);
        }
        // The close is only delivered once the client reads it; drain again
        // after awaiting the task so every token is observable.
        session.await.unwrap();
        while let Ok(token) = rx.try_recv() {
            tokens.push(token);
        }

        assert_eq!(tokens[0], "[COLLAB]status:connected");
        assert_eq!(tokens[1], "[COLLAB]session:xencode-1234");
        assert!(tokens[2].starts_with("[COLLAB]members:["));
        assert_eq!(tokens[4], "[COLLAB]ready");
        let tail = &tokens[tokens.len() - 3..];
        assert_eq!(
            tail[0], "[COLLAB]error:rejected: session is full",
            "got {tokens:?}"
        );
        assert_eq!(tail[2], "[COLLAB]status:disconnected");
    }
}
