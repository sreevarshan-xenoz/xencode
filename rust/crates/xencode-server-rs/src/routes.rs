use std::sync::Arc;
use axum::{
    extract::{Path, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use tower_http::cors::CorsLayer;

use crate::ws::AppState;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    service: String,
    version: String,
}

#[derive(Serialize)]
struct StatusResponse {
    online: bool,
    sessions: usize,
    uptime_secs: u64,
}

#[derive(Serialize)]
struct SessionInfo {
    id: String,
    members: Vec<String>,
    created_at: String,
}

/// Health check endpoint.
async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "online".to_string(),
        service: "Xencode Server".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Server status with session count and uptime.
async fn server_status(State(state): State<Arc<AppState>>) -> Json<StatusResponse> {
    let sessions = state.sessions.lock().await.len();
    Json(StatusResponse {
        online: true,
        sessions,
        uptime_secs: 0,
    })
}

/// Create a new collaboration session.
async fn create_session(State(_state): State<Arc<AppState>>) -> Json<SessionInfo> {
    let id = uuid::Uuid::new_v4().to_string();
    Json(SessionInfo {
        id: format!("xencode-{}", &id[..8]),
        members: Vec::new(),
        created_at: chrono::Utc::now().to_rfc3339(),
    })
}

/// Get session info by invite code (id).
async fn get_session(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Json<SessionInfo> {
    let members = state.sessions.lock().await
        .get(&id)
        .cloned()
        .unwrap_or_default();
    Json(SessionInfo {
        id,
        members,
        created_at: chrono::Utc::now().to_rfc3339(),
    })
}

/// WebSocket upgrade handler for collaboration sessions.
async fn ws_handler(
    ws: WebSocketUpgrade,
    Path((session_id, username)): Path<(String, String)>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| crate::ws::handle_socket(socket, session_id, username, state))
}

/// API config endpoint — returns current server configuration.
async fn get_config() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "max_session_size": 10,
        "supported_models": ["qwen2.5:7b", "llama3.1:8b", "gpt-4o", "claude-3.5-sonnet"],
        "features": ["collaboration", "code_analysis", "rag", "plugins"],
    }))
}

/// List available models.
async fn list_models() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "models": [
            {"name": "qwen2.5:7b", "provider": "ollama", "type": "local"},
            {"name": "llama3.1:8b", "provider": "ollama", "type": "local"},
            {"name": "gpt-4o", "provider": "openai", "type": "remote"},
            {"name": "claude-3.5-sonnet", "provider": "anthropic", "type": "remote"},
        ]
    }))
}

/// Build the complete axum Router with all routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(health_check))
        .route("/sessions/create", post(create_session))
        .route("/sessions/{id}", get(get_session))
        .route("/ws/{session_id}/{username}", get(ws_handler))
        .route("/auth/login", post(crate::auth::login))
        .route("/auth/verify", post(crate::auth::verify_token))
        .route("/api/config", get(get_config))
        .route("/api/models", get(list_models))
        .route("/api/status", get(server_status))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
