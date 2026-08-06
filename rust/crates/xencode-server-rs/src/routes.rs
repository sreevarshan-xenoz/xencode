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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check_body() {
        let resp = health_check().await;
        assert_eq!(resp.status, "online");
        assert_eq!(resp.service, "Xencode Server");
        assert_eq!(resp.version, env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_get_config_fields() {
        let config = get_config().await;
        assert_eq!(config.0["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(config.0["max_session_size"], 10);
        let features = config.0["features"].as_array().unwrap();
        assert!(features.contains(&serde_json::json!("collaboration")));
        assert!(features.contains(&serde_json::json!("code_analysis")));
        assert!(features.contains(&serde_json::json!("rag")));
        assert!(features.contains(&serde_json::json!("plugins")));
    }

    #[tokio::test]
    async fn test_list_models_count() {
        let models = list_models().await;
        let model_list = models.0["models"].as_array().unwrap();
        assert_eq!(model_list.len(), 4);
        let names: Vec<&str> = model_list
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"qwen2.5:7b"));
        assert!(names.contains(&"llama3.1:8b"));
        assert!(names.contains(&"gpt-4o"));
        assert!(names.contains(&"claude-3.5-sonnet"));
    }

    #[tokio::test]
    async fn test_server_status_initial() {
        let state = Arc::new(AppState::new());
        let resp = server_status(State(state)).await;
        assert!(resp.online);
        assert_eq!(resp.sessions, 0);
    }

    #[tokio::test]
    async fn test_create_session_returns_id() {
        let state = Arc::new(AppState::new());
        let session = create_session(State(state)).await;
        assert!(session.id.starts_with("xencode-"));
        assert!(session.members.is_empty());
        assert!(!session.created_at.is_empty());
    }

    #[tokio::test]
    async fn test_get_session_not_found() {
        let state = Arc::new(AppState::new());
        let session = get_session(Path("nonexistent".to_string()), State(state)).await;
        assert_eq!(session.id, "nonexistent");
        assert!(session.members.is_empty());
    }

    #[tokio::test]
    async fn test_get_session_with_members() {
        let state = Arc::new(AppState::new());
        // Manually add a session with members
        {
            let mut sessions = state.sessions.lock().await;
            sessions.insert("active-session".to_string(), vec!["alice".to_string(), "bob".to_string()]);
        }
        let session = get_session(Path("active-session".to_string()), State(state)).await;
        assert_eq!(session.members, vec!["alice", "bob"]);
    }

    #[tokio::test]
    async fn test_health_check_response_json() {
        let resp = health_check().await;
        let json = serde_json::to_value(&resp.0).unwrap();
        assert_eq!(json["status"], "online");
        assert_eq!(json["service"], "Xencode Server");
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_health_response_struct_serialize() {
        let resp = HealthResponse {
            status: "online".to_string(),
            service: "Test".to_string(),
            version: "0.1.0".to_string(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["status"], "online");
        assert_eq!(json["service"], "Test");
        assert_eq!(json["version"], "0.1.0");
    }

    #[test]
    fn test_status_response_struct_serialize() {
        let resp = StatusResponse {
            online: true,
            sessions: 5,
            uptime_secs: 3600,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["online"], true);
        assert_eq!(json["sessions"], 5);
        assert_eq!(json["uptime_secs"], 3600);
    }

    #[test]
    fn test_session_info_struct_serialize() {
        let resp = SessionInfo {
            id: "xencode-abc123".to_string(),
            members: vec!["alice".to_string(), "bob".to_string()],
            created_at: "2026-06-06T12:00:00+00:00".to_string(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["id"], "xencode-abc123");
        assert_eq!(json["members"], serde_json::json!(["alice", "bob"]));
        assert_eq!(json["created_at"], "2026-06-06T12:00:00+00:00");
    }
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
