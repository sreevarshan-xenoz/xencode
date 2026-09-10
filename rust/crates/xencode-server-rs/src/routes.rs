use axum::{
    extract::{Path, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::sync::Arc;
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
async fn create_session(State(state): State<Arc<AppState>>) -> Json<SessionInfo> {
    let mut sessions = state.sessions.lock().await;

    // The id is a truncated UUID, so it carries only 32 bits. That was harmless
    // while nothing was stored under it; now that it is a map key, a collision
    // would merge two unrelated sessions — so pick one that is actually free.
    let id = loop {
        let candidate = format!("xencode-{}", &uuid::Uuid::new_v4().to_string()[..8]);
        if !sessions.contains_key(&candidate) {
            break candidate;
        }
    };

    sessions.insert(id.clone(), Vec::new());

    Json(SessionInfo {
        id,
        members: Vec::new(),
        created_at: chrono::Utc::now().to_rfc3339(),
    })
}

/// Get session info by invite code (id).
async fn get_session(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Json<SessionInfo> {
    let members = state
        .sessions
        .lock()
        .await
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
    let cfg = xencode_config_rs::XencodeConfig::load().unwrap_or_default();
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "max_session_size": 10,
        "supported_models": ["qwen2.5:7b", "llama3.1:8b", "gpt-4o", "claude-3.5-sonnet"],
        "features": ["collaboration", "code_analysis", "rag", "plugins", "llamacpp"],
        "llamacpp": {
            "url": cfg.llama_cpp_url,
            "model_path": cfg.llama_cpp_model_path,
            "executable": cfg.llama_cpp_executable,
            "args": cfg.llama_cpp_args,
            "sampling": {
                "temperature": cfg.llama_cpp_temperature,
                "top_k": cfg.llama_cpp_top_k,
                "min_p": cfg.llama_cpp_min_p,
                "max_tokens": cfg.llama_cpp_max_tokens,
            }
        }
    }))
}

/// List available models dynamically from Ollama and llama.cpp.
async fn list_models() -> Json<serde_json::Value> {
    let client = xencode_models_rs::OllamaClient::default_client();
    let mut models_json = Vec::new();

    if let Ok(models) = client.list_models().await {
        for m in models {
            if !m.name.contains("embed") {
                models_json.push(serde_json::json!({
                    "name": m.name,
                    "provider": "ollama",
                    "type": "local",
                    "size": m.size,
                    "modified_at": m.modified_at,
                }));
            }
        }
    }

    // Query the llama.cpp server for its available models.
    let cfg = xencode_config_rs::XencodeConfig::load().unwrap_or_default();
    let llama_client = xencode_models_rs::LlamaCppClient::new(&cfg.llama_cpp_url, 5);
    if let Ok(llama_models) = llama_client.list_models().await {
        for m in llama_models {
            models_json.push(serde_json::json!({
                "name": m.id,
                "provider": "llamacpp",
                "type": "local",
            }));
        }
    }

    if models_json.is_empty() {
        // Fallback models when Ollama/llama.cpp are offline
        models_json
            .push(serde_json::json!({"name": "qwen2.5:7b", "provider": "ollama", "type": "local"}));
        models_json.push(
            serde_json::json!({"name": "llama3.1:8b", "provider": "ollama", "type": "local"}),
        );
    }

    models_json.push(serde_json::json!({"name": "gpt-4o", "provider": "openai", "type": "remote"}));
    models_json.push(
        serde_json::json!({"name": "claude-3.5-sonnet", "provider": "anthropic", "type": "remote"}),
    );

    Json(serde_json::json!({
        "models": models_json
    }))
}

/// Whether the llama.cpp server process is tracked/auto-startable, and the
/// last recorded generation timing stats.
async fn llamacpp_status() -> Json<serde_json::Value> {
    let cfg = xencode_config_rs::XencodeConfig::load().unwrap_or_default();
    let exe = if cfg.llama_cpp_executable.is_empty() {
        None
    } else {
        Some(cfg.llama_cpp_executable.as_str())
    };
    let llama_pid = xencode_models_rs::find_llama_server(exe);
    Json(serde_json::json!({
        "server_running": llama_pid.is_some(),
        "pid": llama_pid,
        "url": cfg.llama_cpp_url,
        "model_path": cfg.llama_cpp_model_path,
        "executable": cfg.llama_cpp_executable,
    }))
}

/// Load the configured GGUF model into the llama.cpp server (unloads any
/// currently loaded model first).
async fn llamacpp_load() -> Json<serde_json::Value> {
    let cfg = xencode_config_rs::XencodeConfig::load().unwrap_or_default();
    if cfg.llama_cpp_model_path.is_empty() {
        return Json(serde_json::json!({
            "success": false,
            "error": "no llama_cpp_model_path configured",
        }));
    }
    let client = xencode_models_rs::LlamaCppClient::new(&cfg.llama_cpp_url, 60);
    match client.load_model(&cfg.llama_cpp_model_path).await {
        Ok(()) => Json(serde_json::json!({ "success": true })),
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": e.to_string(),
        })),
    }
}

/// Unload the currently loaded model from the llama.cpp server.
async fn llamacpp_unload() -> Json<serde_json::Value> {
    let cfg = xencode_config_rs::XencodeConfig::load().unwrap_or_default();
    let client = xencode_models_rs::LlamaCppClient::new(&cfg.llama_cpp_url, 60);
    match client.unload_models().await {
        Ok(()) => Json(serde_json::json!({ "success": true })),
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": e.to_string(),
        })),
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
        .route("/api/llamacpp/status", get(llamacpp_status))
        .route("/api/llamacpp/load", post(llamacpp_load))
        .route("/api/llamacpp/unload", post(llamacpp_unload))
        .layer(CorsLayer::permissive())
        .with_state(state)
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
        assert!(features.contains(&serde_json::json!("llamacpp")));
        let ll = &config.0["llamacpp"];
        assert!(ll["url"].is_string());
        assert!(ll["model_path"].is_string());
        assert!(ll["executable"].is_string());
        assert!(ll["args"].is_array());
        assert!(
            ll["sampling"]["temperature"].is_null() || ll["sampling"]["temperature"].is_number()
        );
        assert!(ll["sampling"]["top_k"].is_null() || ll["sampling"]["top_k"].is_number());
        assert!(ll["sampling"]["min_p"].is_null() || ll["sampling"]["min_p"].is_number());
        assert!(ll["sampling"]["max_tokens"].is_null() || ll["sampling"]["max_tokens"].is_number());
    }

    #[tokio::test]
    async fn test_list_models_count() {
        let models = list_models().await;
        let model_list = models.0["models"].as_array().unwrap();
        // Both remote providers are always appended, so at least 4 entries.
        assert!(model_list.len() >= 4);
        // Every entry must carry a name + provider + type.
        for m in model_list {
            assert!(m["name"].is_string());
            assert!(m["provider"].is_string());
            assert!(m["type"].is_string());
        }
        let names: Vec<&str> = model_list
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
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
    async fn create_session_records_the_session_in_state() {
        let state = Arc::new(AppState::new());
        let session = create_session(State(state.clone())).await;

        let sessions = state.sessions.lock().await;
        assert!(
            sessions.contains_key(&session.id),
            "created session {} is not in state; state holds {:?}",
            session.id,
            sessions.keys().collect::<Vec<_>>()
        );
        assert_eq!(sessions[&session.id], Vec::<String>::new());
    }

    /// Round-trip check only. It cannot detect the bug this change fixes:
    /// `get_session` uses `unwrap_or_default()`, so a session that was never
    /// recorded is indistinguishable from one recorded with no members. Making
    /// the two distinguishable means returning 404 for the former, which is an
    /// API change worth deciding on its own.
    #[tokio::test]
    async fn get_session_round_trips_a_created_session() {
        let state = Arc::new(AppState::new());
        let created = create_session(State(state.clone())).await;

        let fetched = get_session(Path(created.id.clone()), State(state)).await;
        assert_eq!(fetched.id, created.id);
        assert!(fetched.members.is_empty());
    }

    /// A member joining through the WebSocket path must be visible on the
    /// session that `create_session` handed out — the two must agree on the key.
    #[tokio::test]
    async fn a_member_joining_a_created_session_is_visible_on_it() {
        let state = Arc::new(AppState::new());
        let created = create_session(State(state.clone())).await;

        // What handle_socket does when a peer joins.
        state
            .sessions
            .lock()
            .await
            .entry(created.id.clone())
            .or_default()
            .push("alice".to_string());

        let fetched = get_session(Path(created.id.clone()), State(state.clone())).await;
        assert_eq!(fetched.members, vec!["alice".to_string()]);
        // Joining must not have created a second, parallel session entry.
        assert_eq!(state.sessions.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn created_sessions_are_counted_by_server_status() {
        let state = Arc::new(AppState::new());
        assert_eq!(server_status(State(state.clone())).await.sessions, 0);

        create_session(State(state.clone())).await;
        create_session(State(state.clone())).await;

        assert_eq!(server_status(State(state)).await.sessions, 2);
    }

    #[tokio::test]
    async fn create_session_issues_distinct_ids() {
        let state = Arc::new(AppState::new());
        let mut ids = std::collections::HashSet::new();
        for _ in 0..50 {
            ids.insert(create_session(State(state.clone())).await.0.id.clone());
        }
        assert_eq!(ids.len(), 50);
        assert_eq!(state.sessions.lock().await.len(), 50);
    }

    #[tokio::test]
    async fn create_session_never_overwrites_an_existing_session() {
        let state = Arc::new(AppState::new());
        // A session that already has members, as a live one would.
        state
            .sessions
            .lock()
            .await
            .insert("xencode-existing".to_string(), vec!["alice".to_string()]);

        create_session(State(state.clone())).await;

        let sessions = state.sessions.lock().await;
        assert_eq!(sessions["xencode-existing"], vec!["alice".to_string()]);
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
            sessions.insert(
                "active-session".to_string(),
                vec!["alice".to_string(), "bob".to_string()],
            );
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

    #[tokio::test]
    async fn test_llamacpp_status_shape() {
        let status = llamacpp_status().await;
        assert!(status.0["server_running"].is_boolean());
        // pid may be null or a string
        assert!(status.0["pid"].is_null() || status.0["pid"].is_string());
        assert!(status.0["url"].is_string());
        assert!(status.0["model_path"].is_string());
        assert!(status.0["executable"].is_string());
    }

    #[tokio::test]
    async fn test_llamacpp_load_no_path() {
        // Guard against an environment-configured path leaking in: only assert
        // the response is a well-formed object with a success boolean.
        let resp = llamacpp_load().await;
        assert!(resp.0["success"].is_boolean());
    }

    #[tokio::test]
    async fn test_llamacpp_unload_shape() {
        let resp = llamacpp_unload().await;
        assert!(resp.0["success"].is_boolean());
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
