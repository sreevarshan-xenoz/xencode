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
}
