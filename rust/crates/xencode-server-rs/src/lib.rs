pub mod audit;
pub mod auth;
pub mod routes;
pub mod tokens;
pub mod ws;

use axum::Router;
use std::sync::Arc;

/// Build the full application router with default state.
pub fn build_app() -> Router {
    let state = Arc::new(ws::AppState::new());
    routes::build_router(state)
}

/// Build the application router with a provided state (for testing).
pub fn build_app_with_state(state: Arc<ws::AppState>) -> Router {
    routes::build_router(state)
}

/// Tests for the config-reading routes must not see the developer's real
/// `~/.xencode/config.json`: which local servers happen to be running (or a
/// Colab forward a live run left on the configured port) changes what
/// `/api/models` returns. Pointing `$XCODE_CONFIG_DIR` at a temp dir holding a
/// config whose local endpoints are closed makes those routes deterministic
/// without faking any of their logic.
///
/// The env override is process-global, so the guard also serialises the tests
/// that take it.
#[cfg(test)]
pub(crate) mod testenv {
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::sync::{Mutex, MutexGuard};

    static CONFIG_ENV_LOCK: Mutex<()> = Mutex::new(());

    pub struct IsolatedConfig {
        _lock: MutexGuard<'static, ()>,
        old_dir: Option<OsString>,
        dir: PathBuf,
    }

    impl Drop for IsolatedConfig {
        fn drop(&mut self) {
            match self.old_dir.take() {
                Some(value) => std::env::set_var("XCODE_CONFIG_DIR", value),
                None => std::env::remove_var("XCODE_CONFIG_DIR"),
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    pub fn isolated_config(tag: &str) -> IsolatedConfig {
        let lock = CONFIG_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("xencode-server-test-{tag}-cfg"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create isolated config dir");
        // Port 9 (discard) refuses locally, so the llama.cpp branch of the
        // route runs its real "server offline" path on every machine.
        std::fs::write(
            dir.join("config.json"),
            r#"{"llama_cpp_url": "http://127.0.0.1:9"}"#,
        )
        .expect("write isolated config");
        let old_dir = std::env::var_os("XCODE_CONFIG_DIR");
        std::env::set_var("XCODE_CONFIG_DIR", &dir);
        IsolatedConfig {
            _lock: lock,
            old_dir,
            dir,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// A router plus a freshly-issued token for its state, so auth-gated
    /// routes can be exercised end-to-end.
    async fn app_with_token() -> (Router, String) {
        let state = Arc::new(ws::AppState::new());
        let principal = state.tokens.lock().await.issue("tester");
        (build_app_with_state(state), principal.token)
    }

    async fn json_body(response: axum::response::Response) -> serde_json::Value {
        let body = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn test_health_check() {
        let app = build_app();
        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_api_status() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_api_config() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_models() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn create_session_with_a_valid_token() {
        let (app, token) = app_with_token().await;
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/sessions/create")
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {token}"))
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// The regression this guards: session creation used to be open to anyone.
    #[tokio::test]
    async fn create_session_without_a_token_is_401() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/sessions/create")
                    .header("Content-Type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn create_session_with_a_forged_token_is_401() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/sessions/create")
                    .header("Content-Type", "application/json")
                    .header(
                        "Authorization",
                        "Bearer xencode_00000000000000000000000000000000",
                    )
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_login_then_verify_round_trip() {
        let app = build_app();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/login")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"username":"alice","api_key":null}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        let token = json["token"].as_str().unwrap().to_string();
        assert!(token.starts_with("xencode_"));
        assert_eq!(json["username"], "alice");
        assert!(!json["expires_at"].as_str().unwrap().is_empty());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/verify")
                    .header("Content-Type", "application/json")
                    .body(Body::from(serde_json::json!({"token": token}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        assert_eq!(json["username"], "alice");
        assert!(!json["expires_at"].as_str().unwrap().is_empty());
    }

    /// Login silently ignored `api_key` before; a caller relying on it must
    /// be told it does nothing rather than getting a token that pretends to
    /// be key-backed.
    #[tokio::test]
    async fn auth_login_with_api_key_is_400() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/login")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"username":"bob","api_key":"sk-123"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// The regression this guards: any >20-char `xencode_` string used to
    /// verify successfully.
    #[tokio::test]
    async fn auth_verify_garbage_token_is_401() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/verify")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        r#"{"token":"xencode_abcdefghijklmnopqrstuvwxyz"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_verify_short_token_is_401() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/verify")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"token":"xencode_abc"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn create_session_response_body() {
        let (app, token) = app_with_token().await;
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/sessions/create")
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {token}"))
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        assert!(json["id"].as_str().unwrap().starts_with("xencode-"));
        // The authenticated creator is the session's admin.
        assert_eq!(json["members"], serde_json::json!(["tester"]));
        assert!(!json["created_at"].as_str().unwrap().is_empty());
    }

    /// A created session must be fetchable by its id with the same token.
    #[tokio::test]
    async fn created_session_is_fetchable() {
        let (app, token) = app_with_token().await;
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/sessions/create")
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {token}"))
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let json = json_body(response).await;
        let id = json["id"].as_str().unwrap().to_string();

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/sessions/{id}"))
                    .header("Authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        assert_eq!(json["id"], id);
    }

    #[tokio::test]
    async fn get_session_requires_a_token() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/sessions/whatever")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn get_session_unknown_id_is_404() {
        let (app, token) = app_with_token().await;
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/sessions/no-such-session")
                    .header("Authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_server_status_body() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        assert_eq!(json["online"], true);
        assert_eq!(json["sessions"], 0);
    }

    #[tokio::test]
    async fn test_health_check_body() {
        let app = build_app();
        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        assert_eq!(json["status"], "online");
        assert_eq!(json["service"], "Xencode Server");
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_models_list_body() {
        let _iso = crate::testenv::isolated_config("models-list");
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        let models = json["models"].as_array().unwrap();
        // The always-present remote providers plus the offline fallbacks; the
        // exact count depends on local servers, so assert on the invariant.
        assert!(models.len() >= 2, "remotes always listed: {models:?}");
        let names: Vec<&str> = models.iter().map(|m| m["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"gpt-4o"), "openai row present: {names:?}");
        assert!(
            names.contains(&"claude-3.5-sonnet"),
            "anthropic row present: {names:?}"
        );
    }

    #[tokio::test]
    async fn test_config_body() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
        assert!(json["features"].as_array().unwrap().len() >= 3);
    }

    #[tokio::test]
    async fn test_invalid_route_returns_404() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/nonexistent-route")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// Permissive CORS was removed: no browser client exists, and the header
    /// must not come back by accident.
    #[tokio::test]
    async fn no_cors_headers_are_emitted() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("Origin", "http://example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(!response
            .headers()
            .contains_key("access-control-allow-origin"));
    }

    #[tokio::test]
    async fn llamacpp_load_requires_a_token() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/llamacpp/load")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_build_app_with_state() {
        let state = Arc::new(crate::ws::AppState::new());
        let app = build_app_with_state(state.clone());
        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
