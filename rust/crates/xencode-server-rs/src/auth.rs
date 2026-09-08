#[cfg(test)]
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyRequest {
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthResponse {
    pub valid: bool,
    pub username: String,
    pub session_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginResponse {
    pub token: String,
    pub username: String,
    pub expires_at: String,
}

#[derive(Debug)]
pub struct AuthError;

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Authentication failed")
    }
}

impl axum::response::IntoResponse for AuthError {
    fn into_response(self) -> axum::response::Response {
        (
            axum::http::StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({
                "error": "Authentication failed"
            })),
        )
            .into_response()
    }
}

/// Simple API key verification (production should use proper JWT).
pub async fn verify_token(
    axum::Json(req): axum::Json<VerifyRequest>,
) -> Result<axum::Json<AuthResponse>, AuthError> {
    if req.token.starts_with("xencode_") && req.token.len() > 20 {
        Ok(axum::Json(AuthResponse {
            valid: true,
            username: "user".to_string(),
            session_token: uuid::Uuid::new_v4().to_string(),
        }))
    } else {
        Err(AuthError)
    }
}

/// Login endpoint — issues a token given a username and optional API key.
pub async fn login(
    axum::Json(req): axum::Json<LoginRequest>,
) -> Result<axum::Json<LoginResponse>, AuthError> {
    let token = format!(
        "xencode_{}",
        uuid::Uuid::new_v4().to_string().replace('-', "")
    );
    let expires = chrono::Utc::now() + chrono::Duration::hours(24);
    Ok(axum::Json(LoginResponse {
        token,
        username: req.username,
        expires_at: expires.to_rfc3339(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_error_display() {
        let err = AuthError;
        assert_eq!(format!("{err}"), "Authentication failed");
    }

    #[test]
    fn test_auth_error_into_response_status() {
        use axum::http::StatusCode;
        let err = AuthError;
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_verify_token_valid() {
        let req = VerifyRequest {
            token: "xencode_abcdefghijklmnopqrstuvwxyz".to_string(),
        };
        let result = verify_token(axum::Json(req)).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(resp.0.valid);
        assert_eq!(resp.0.username, "user");
        assert!(!resp.0.session_token.is_empty());
    }

    #[tokio::test]
    async fn test_verify_token_short_prefix_only() {
        // Token starts with xencode_ but is too short (len <= 20)
        let req = VerifyRequest {
            token: "xencode_abc".to_string(),
        };
        let result = verify_token(axum::Json(req)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_verify_token_too_short() {
        let req = VerifyRequest {
            token: "xencode_".to_string(),
        };
        let result = verify_token(axum::Json(req)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_verify_token_wrong_prefix() {
        let req = VerifyRequest {
            token: "bearer_sometokenvalue".to_string(),
        };
        let result = verify_token(axum::Json(req)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_verify_token_empty() {
        let req = VerifyRequest {
            token: String::new(),
        };
        let result = verify_token(axum::Json(req)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_login_returns_token() {
        let req = LoginRequest {
            username: "testuser".to_string(),
            api_key: None,
        };
        let result = login(axum::Json(req)).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(resp.0.token.starts_with("xencode_"));
        assert_eq!(resp.0.username, "testuser");
        assert!(!resp.0.expires_at.is_empty());
    }

    #[test]
    fn test_login_request_deserialize() {
        let json = r#"{"username":"alice","api_key":"sk-123"}"#;
        let req: LoginRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.username, "alice");
        assert_eq!(req.api_key, Some("sk-123".to_string()));
    }

    #[test]
    fn test_verify_request_serialize_roundtrip() {
        let original = VerifyRequest {
            token: "xencode_test123".to_string(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: VerifyRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.token, original.token);
    }

    #[test]
    fn test_login_response_serialize() {
        let resp = LoginResponse {
            token: "xencode_token".to_string(),
            username: "bob".to_string(),
            expires_at: "2026-06-07T00:00:00+00:00".to_string(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["token"], "xencode_token");
        assert_eq!(json["username"], "bob");
        assert_eq!(json["expires_at"], "2026-06-07T00:00:00+00:00");
    }

    #[test]
    fn test_auth_response_serialize() {
        let resp = AuthResponse {
            valid: true,
            username: "charlie".to_string(),
            session_token: "uuid-here".to_string(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["valid"], true);
        assert_eq!(json["username"], "charlie");
        assert_eq!(json["session_token"], "uuid-here");
    }
}
