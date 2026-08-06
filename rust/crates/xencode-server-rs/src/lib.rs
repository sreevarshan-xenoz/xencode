pub mod auth;
pub mod routes;
pub mod ws;

use std::sync::Arc;
use axum::Router;

/// Build the full application router with default state.
pub fn build_app() -> Router {
    let state = Arc::new(ws::AppState::new());
    routes::build_router(state)
}

/// Build the application router with a provided state (for testing).
pub fn build_app_with_state(state: Arc<ws::AppState>) -> Router {
    routes::build_router(state)
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
            .oneshot(Request::builder().uri("/api/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_api_config() {
        let app = build_app();
        let response = app
            .oneshot(Request::builder().uri("/api/config").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_models() {
        let app = build_app();
        let response = app
            .oneshot(Request::builder().uri("/api/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_create_session() {
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
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_login_success() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/login")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        r#"{"username":"alice","api_key":null}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["token"].as_str().unwrap().starts_with("xencode_"));
        assert_eq!(json["username"], "alice");
        assert!(json["expires_at"].as_str().unwrap().len() > 0);
    }

    #[tokio::test]
    async fn test_auth_login_with_api_key() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/login")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        r#"{"username":"bob","api_key":"sk-123"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_verify_valid() {
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
        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["valid"], true);
        assert!(!json["session_token"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_auth_verify_invalid_token() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/verify")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        r#"{"token":"bad_token"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_verify_short_token() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/verify")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        r#"{"token":"xencode_abc"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_create_session_response_body() {
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
        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["id"].as_str().unwrap().starts_with("xencode-"));
        assert!(json["members"].as_array().unwrap().is_empty());
        assert!(!json["created_at"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_get_session_not_found() {
        let app = build_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/sessions/no-such-session")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["id"], "no-such-session");
        assert!(json["members"].as_array().unwrap().is_empty());
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
        let body = response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
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
        let body = response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "online");
        assert_eq!(json["service"], "Xencode Server");
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_models_list_body() {
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
        let body = response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let models = json["models"].as_array().unwrap();
        assert_eq!(models.len(), 4);
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
        let body = response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
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

    #[tokio::test]
    async fn test_cors_headers_present() {
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
        // CORS preflight headers should allow all origins
        assert!(response.headers().contains_key("access-control-allow-origin"));
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
