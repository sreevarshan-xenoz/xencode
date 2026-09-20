//! End-to-end WebSocket handshake tests: real axum router, real
//! tokio-tungstenite client, connected through an in-process duplex pipe —
//! no ports, no dependence on the machine's network stack.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::DuplexStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;
use tower::ServiceExt;
use xencode_collaboration_rs::wire::{
    ClientFrame, MemberInfo, ServerFrame, CLOSE_BAD_TOKEN, CLOSE_NO_SESSION, CLOSE_SESSION_FULL,
    MAX_SESSION_MEMBERS,
};
use xencode_collaboration_rs::Role;
use xencode_server_rs::build_app_with_state;
use xencode_server_rs::ws::AppState;

type Ws = WebSocketStream<DuplexStream>;

/// Serve `app` over one half of a duplex pipe and hand back a connected,
/// upgraded WebSocket client on the other half. The Incoming→axum Body
/// mapping mirrors what `axum::serve` does internally, so upgrades survive.
async fn connect(app: Router, session: &str) -> Ws {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let service = app.map_request(|req: Request<Incoming>| req.map(Body::new));
        let _ = hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
            .serve_connection_with_upgrades(
                TokioIo::new(server_io),
                TowerToHyperService::new(service),
            )
            .await;
    });
    let (ws, _response) =
        tokio_tungstenite::client_async(format!("ws://xencode.test/ws/{session}"), client_io)
            .await
            .expect("WebSocket upgrade should succeed at the HTTP layer");
    ws
}

/// A state with a session `s1` owned by `owner`, plus issued tokens.
async fn state_with_session() -> (Arc<AppState>, String, String) {
    let state = Arc::new(AppState::new());
    let owner_token = state.tokens.lock().await.issue("owner").token;
    {
        let mut workspaces = state.workspaces.lock().await;
        workspaces.create_workspace_with_id("s1", "session one", "owner");
    }
    let guest_token = state.tokens.lock().await.issue("guest").token;
    (state, owner_token, guest_token)
}

async fn send(ws: &mut Ws, frame: &ClientFrame) {
    let json = serde_json::to_string(frame).unwrap();
    ws.send(Message::Text(json.into())).await.unwrap();
}

async fn send_raw(ws: &mut Ws, raw: &str) {
    ws.send(Message::Text(raw.to_string().into()))
        .await
        .unwrap();
}

#[derive(Debug)]
enum Event {
    Frame(ServerFrame),
    Closed(u16),
}

async fn next_event(ws: &mut Ws) -> Event {
    match tokio::time::timeout(Duration::from_secs(5), ws.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => {
            let parsed = serde_json::from_str(&text);
            match parsed {
                Ok(frame) => Event::Frame(frame),
                Err(e) => panic!("server sent an unparseable frame: {e}: {text}"),
            }
        }
        Ok(Some(Ok(Message::Close(Some(frame))))) => Event::Closed(u16::from(frame.code)),
        other => panic!("connection ended without a close frame: {other:?}"),
    }
}

/// Expect the next frame to be an `AuthOk`, returning it.
async fn expect_auth_ok(ws: &mut Ws) -> (String, String, Vec<MemberInfo>) {
    match next_event(ws).await {
        Event::Frame(ServerFrame::AuthOk {
            username,
            role,
            members,
            ..
        }) => (username, role, members),
        other => panic!("expected auth_ok, got {other:?}"),
    }
}

/// Authenticate an already-upgraded connection.
async fn authenticate(ws: &mut Ws, token: &str) -> (String, String, Vec<MemberInfo>) {
    send(
        ws,
        &ClientFrame::Auth {
            token: token.to_string(),
        },
    )
    .await;
    expect_auth_ok(ws).await
}

/// A rejection arrives as an error frame naming the reason, then a close
/// frame carrying the matching 44xx code.
async fn expect_rejection(ws: &mut Ws, code: &str, close_code: u16) {
    match next_event(ws).await {
        Event::Frame(ServerFrame::Error { code: c, .. }) => assert_eq!(c, code),
        other => panic!("expected error frame, got {other:?}"),
    }
    match next_event(ws).await {
        Event::Closed(c) => assert_eq!(c, close_code),
        other => panic!("expected close frame, got {other:?}"),
    }
}

#[tokio::test]
async fn valid_token_gets_auth_ok_with_self_in_members() {
    let (state, owner_token, _guest) = state_with_session().await;
    let app = build_app_with_state(state);
    let mut ws = connect(app, "s1").await;

    let (username, role, members) = authenticate(&mut ws, &owner_token).await;
    assert_eq!(username, "owner");
    assert_eq!(role, "admin");
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].username, "owner");
    assert_eq!(members[0].role, "admin");
}

/// The regression this guards: identity used to be whatever username the
/// caller put in the URL — no credential of any kind was required.
#[tokio::test]
async fn a_non_auth_first_frame_closes_4401() {
    let (state, _owner, _guest) = state_with_session().await;
    let app = build_app_with_state(state);
    let mut ws = connect(app, "s1").await;

    send(&mut ws, &ClientFrame::Activity { text: "hi".into() }).await;
    expect_rejection(&mut ws, "bad_token", CLOSE_BAD_TOKEN).await;
}

#[tokio::test]
async fn garbage_before_auth_closes_4401() {
    let (state, _owner, _guest) = state_with_session().await;
    let app = build_app_with_state(state);
    let mut ws = connect(app, "s1").await;

    send_raw(&mut ws, "not even json").await;
    expect_rejection(&mut ws, "bad_token", CLOSE_BAD_TOKEN).await;
}

#[tokio::test]
async fn an_unissued_token_closes_4401() {
    let (state, _owner, _guest) = state_with_session().await;
    let app = build_app_with_state(state);
    let mut ws = connect(app, "s1").await;

    send(
        &mut ws,
        &ClientFrame::Auth {
            token: "xencode_forged".into(),
        },
    )
    .await;
    expect_rejection(&mut ws, "bad_token", CLOSE_BAD_TOKEN).await;
}

#[tokio::test]
async fn an_unknown_session_closes_4404() {
    let (state, owner_token, _guest) = state_with_session().await;
    let app = build_app_with_state(state);
    let mut ws = connect(app, "no-such-session").await;

    send(&mut ws, &ClientFrame::Auth { token: owner_token }).await;
    expect_rejection(&mut ws, "no_session", CLOSE_NO_SESSION).await;
}

#[tokio::test]
async fn a_full_session_closes_4409_for_new_identities() {
    let (state, owner_token, guest_token) = state_with_session().await;
    // Fill the session to the cap with distinct identities.
    {
        let mut workspaces = state.workspaces.lock().await;
        for i in 0..MAX_SESSION_MEMBERS - 1 {
            workspaces
                .add_member("s1", "owner", &format!("member{i}"), Role::Editor)
                .unwrap();
        }
        assert_eq!(workspaces.member_count("s1"), MAX_SESSION_MEMBERS);
    }
    let app = build_app_with_state(state.clone());
    let mut ws = connect(app, "s1").await;

    send(&mut ws, &ClientFrame::Auth { token: guest_token }).await;
    expect_rejection(&mut ws, "session_full", CLOSE_SESSION_FULL).await;

    // An identity that already belongs may still reconnect to a full session.
    let mut ws = connect(build_app_with_state(state), "s1").await;
    let (_user, role, _) = authenticate(&mut ws, &owner_token).await;
    assert_eq!(role, "admin");
}

#[tokio::test]
async fn a_viewer_cannot_relay_activity_and_it_is_audited() {
    let (state, _owner_token, guest_token) = state_with_session().await;
    state
        .workspaces
        .lock()
        .await
        .add_member("s1", "owner", "guest", Role::Viewer)
        .unwrap();

    let app = build_app_with_state(state.clone());
    let mut ws = connect(app, "s1").await;
    let (_, role, _) = authenticate(&mut ws, &guest_token).await;
    assert_eq!(role, "viewer");

    send(
        &mut ws,
        &ClientFrame::Activity {
            text: "rm -rf /".into(),
        },
    )
    .await;
    match next_event(&mut ws).await {
        Event::Frame(ServerFrame::Error { code, .. }) => assert_eq!(code, "rbac_denied"),
        other => panic!("expected error frame, got {other:?}"),
    }
    let denied = state
        .workspaces
        .lock()
        .await
        .events_for("s1")
        .into_iter()
        .any(|e| e.action == xencode_collaboration_rs::AuditAction::Denied && e.actor == "guest");
    assert!(
        denied,
        "the viewer's denial was not written to the audit log"
    );
}

#[tokio::test]
async fn editor_activity_reaches_the_peer_with_the_servers_username() {
    let (state, owner_token, guest_token) = state_with_session().await;
    state
        .workspaces
        .lock()
        .await
        .add_member("s1", "owner", "guest", Role::Editor)
        .unwrap();

    let app = build_app_with_state(state);
    let mut owner_ws = connect(app.clone(), "s1").await;
    authenticate(&mut owner_ws, &owner_token).await;
    let mut guest_ws = connect(app, "s1").await;
    authenticate(&mut guest_ws, &guest_token).await;

    // Hand-rolled frame claiming a different sender: the relayed `user` is
    // the server's authenticated identity, whatever the client asserts.
    send_raw(
        &mut guest_ws,
        r#"{"type":"activity","user":"mallory","text":"fn main() {}"}"#,
    )
    .await;
    // The owner saw the guest's join as a members frame before the relay.
    let mut saw_members = false;
    for _ in 0..5 {
        match next_event(&mut owner_ws).await {
            Event::Frame(ServerFrame::Members { .. }) => {
                saw_members = true;
                continue;
            }
            Event::Frame(ServerFrame::Activity { user, text }) => {
                assert!(saw_members, "the join should have been announced first");
                assert_eq!(user, "guest");
                assert_eq!(text, "fn main() {}");
                return;
            }
            other => panic!("expected relayed activity, got {other:?}"),
        }
    }
    panic!("no relayed activity arrived");
}

#[tokio::test]
async fn ping_is_answered_with_pong() {
    let (state, owner_token, _guest) = state_with_session().await;
    let app = build_app_with_state(state);
    let mut ws = connect(app, "s1").await;
    authenticate(&mut ws, &owner_token).await;

    send(&mut ws, &ClientFrame::Ping).await;
    match next_event(&mut ws).await {
        Event::Frame(ServerFrame::Pong) => {}
        other => panic!("expected pong, got {other:?}"),
    }
}

#[tokio::test]
async fn a_second_connection_by_the_same_identity_is_idempotent() {
    let (state, owner_token, _guest) = state_with_session().await;
    let app = build_app_with_state(state.clone());
    let mut first = connect(app.clone(), "s1").await;
    authenticate(&mut first, &owner_token).await;

    let mut second = connect(app, "s1").await;
    let (_, role, members) = authenticate(&mut second, &owner_token).await;
    assert_eq!(role, "admin", "re-join must not change the role");
    assert_eq!(members.len(), 1, "presence must not double-count a user");
    assert_eq!(
        state.workspaces.lock().await.member_count("s1"),
        1,
        "membership must not double-count a user"
    );
}

#[tokio::test]
async fn disconnect_prunes_presence_and_broadcasts_members() {
    let (state, owner_token, guest_token) = state_with_session().await;
    let app = build_app_with_state(state);
    let mut owner_ws = connect(app.clone(), "s1").await;
    authenticate(&mut owner_ws, &owner_token).await;
    let mut guest_ws = connect(app, "s1").await;
    authenticate(&mut guest_ws, &guest_token).await;

    drop(owner_ws);

    // The guest must eventually see a members frame without the owner.
    for _ in 0..10 {
        if let Event::Frame(ServerFrame::Members { members }) = next_event(&mut guest_ws).await {
            if !members.iter().any(|m| m.username == "owner") {
                assert!(members.iter().any(|m| m.username == "guest"));
                return;
            }
        }
    }
    panic!("no members frame removing the owner arrived");
}

/// The route with the username baked into the path is gone; it must not keep
/// answering, or callers would stay on the unauthenticated surface.
#[tokio::test]
async fn the_old_username_in_path_route_is_gone() {
    let (state, _owner, _guest) = state_with_session().await;
    let app = build_app_with_state(state);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/ws/s1/owner")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
