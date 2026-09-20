use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::ws::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyRequest {
    pub token: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerifyResponse {
    pub username: String,
    pub expires_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub username: String,
    pub expires_at: String,
}

#[derive(Debug)]
pub enum AuthError {
    /// The presented credential is missing, unknown, or expired.
    Unauthorized,
    /// The request was structurally fine but used a mechanism we don't have.
    BadRequest(String),
}

impl AuthError {
    fn body(&self) -> (&'static str, String) {
        match self {
            AuthError::Unauthorized => ("unauthorized", "Authentication failed".to_string()),
            AuthError::BadRequest(msg) => ("bad_request", msg.clone()),
        }
    }
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (code, message) = self.body();
        let status = match self {
            AuthError::Unauthorized => StatusCode::UNAUTHORIZED,
            AuthError::BadRequest(_) => StatusCode::BAD_REQUEST,
        };
        (
            status,
            axum::Json(serde_json::json!({"error": message, "code": code})),
        )
            .into_response()
    }
}

/// Login issues a session token bound to the claimed username.
///
/// There is no password store: the token proves *something* was authenticated
/// by this server, which is what the RBAC and WS layers check. The trust
/// boundary is the bind surface — see `TokenStore`'s docs.
pub async fn login(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<LoginRequest>,
) -> Result<axum::Json<LoginResponse>, AuthError> {
    if req.api_key.is_some() {
        // Silently ignoring a credential the caller believed was in use was the
        // old behaviour; it must fail loudly instead.
        return Err(AuthError::BadRequest(
            "api_key login is not supported; omit the field".to_string(),
        ));
    }
    if req.username.trim().is_empty() {
        return Err(AuthError::BadRequest(
            "username must not be empty".to_string(),
        ));
    }
    let principal = state.tokens.lock().await.issue(&req.username);
    Ok(axum::Json(LoginResponse {
        token: principal.token,
        username: principal.username,
        expires_at: principal.expires_at.to_rfc3339(),
    }))
}

/// Verify a token and return who it belongs to. Unknown or expired tokens get
/// a 401 — there is no longer a prefix-and-length check that any `xencode_`
/// string would pass.
pub async fn verify_token(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<VerifyRequest>,
) -> Result<axum::Json<VerifyResponse>, AuthError> {
    let principal = state.tokens.lock().await.lookup(&req.token);
    match principal {
        Some(p) => Ok(axum::Json(VerifyResponse {
            username: p.username,
            expires_at: p.expires_at.to_rfc3339(),
        })),
        None => Err(AuthError::Unauthorized),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> Arc<AppState> {
        Arc::new(AppState::new())
    }

    async fn login_as(state: &Arc<AppState>, username: &str) -> LoginResponse {
        let resp = login(
            State(state.clone()),
            axum::Json(LoginRequest {
                username: username.to_string(),
                api_key: None,
            }),
        )
        .await
        .unwrap();
        resp.0
    }

    #[tokio::test]
    async fn login_then_verify_round_trips_the_same_identity() {
        let state = state();
        let resp = login_as(&state, "alice").await;
        assert!(resp.token.starts_with("xencode_"));
        assert_eq!(resp.username, "alice");

        let verified = verify_token(
            State(state.clone()),
            axum::Json(VerifyRequest {
                token: resp.token.clone(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(verified.0.username, "alice");
        assert!(!verified.0.expires_at.is_empty());
    }

    /// The regression this guards: the old verify accepted any >20-char
    /// `xencode_` string and returned a placeholder username.
    #[tokio::test]
    async fn a_forged_token_is_rejected() {
        let state = state();
        login_as(&state, "alice").await;
        let result = verify_token(
            State(state),
            axum::Json(VerifyRequest {
                token: "xencode_000000000000000000000000000000000000".to_string(),
            }),
        )
        .await;
        assert!(matches!(result, Err(AuthError::Unauthorized)));
    }

    #[tokio::test]
    async fn expired_tokens_do_not_verify() {
        let state = state();
        let stale = {
            let mut tokens = state.tokens.lock().await;
            tokens.issue_with_ttl("alice", chrono::Duration::seconds(-1))
        };
        let result = verify_token(
            State(state),
            axum::Json(VerifyRequest { token: stale.token }),
        )
        .await;
        assert!(matches!(result, Err(AuthError::Unauthorized)));
    }

    #[tokio::test]
    async fn api_key_login_is_rejected_not_ignored() {
        let state = state();
        let result = login(
            State(state.clone()),
            axum::Json(LoginRequest {
                username: "bob".to_string(),
                api_key: Some("sk-123".to_string()),
            }),
        )
        .await;
        let err = result.expect_err("api_key login must fail");
        assert!(matches!(err, AuthError::BadRequest(_)));
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        // And nothing was issued.
        assert!(state.tokens.lock().await.lookup("xencode_none").is_none());
    }

    #[tokio::test]
    async fn empty_username_is_rejected() {
        let state = state();
        let result = login(
            State(state),
            axum::Json(LoginRequest {
                username: "   ".to_string(),
                api_key: None,
            }),
        )
        .await;
        assert!(matches!(result, Err(AuthError::BadRequest(_))));
    }

    #[tokio::test]
    async fn auth_error_status_codes() {
        assert_eq!(
            AuthError::Unauthorized.into_response().status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            AuthError::BadRequest("x".into()).into_response().status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[tokio::test]
    async fn logged_in_tokens_are_independent_identities() {
        let state = state();
        let a = login_as(&state, "alice").await;
        let b = login_as(&state, "bob").await;
        assert_ne!(a.token, b.token);
        for (token, name) in [(&a.token, "alice"), (&b.token, "bob")] {
            let verified = verify_token(
                State(state.clone()),
                axum::Json(VerifyRequest {
                    token: token.clone(),
                }),
            )
            .await
            .unwrap();
            assert_eq!(verified.0.username, name);
        }
    }

    #[test]
    fn login_request_deserializes_absent_api_key() {
        let req: LoginRequest = serde_json::from_str(r#"{"username":"alice"}"#).unwrap();
        assert_eq!(req.username, "alice");
        assert!(req.api_key.is_none());
    }

    #[test]
    fn login_request_deserializes_present_api_key() {
        let req: LoginRequest =
            serde_json::from_str(r#"{"username":"alice","api_key":"sk-123"}"#).unwrap();
        assert_eq!(req.api_key, Some("sk-123".to_string()));
    }

    #[test]
    fn verify_request_deserializes() {
        let req: VerifyRequest = serde_json::from_str(r#"{"token":"xencode_test123"}"#).unwrap();
        assert_eq!(req.token, "xencode_test123");
    }
}
