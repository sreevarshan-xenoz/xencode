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
        axum::Json(serde_json::json!({
            "error": "Authentication failed"
        }))
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
    let token = format!("xencode_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
    let expires = chrono::Utc::now() + chrono::Duration::hours(24);
    Ok(axum::Json(LoginResponse {
        token,
        username: req.username,
        expires_at: expires.to_rfc3339(),
    }))
}
