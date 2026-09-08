use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::health::{HealthStatus, HealthTracker, ModelHealth};

/// Information about a model hosted in llama.cpp (`llama-server`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlamaCppModelInfo {
    pub id: String,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub owned_by: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    #[serde(default)]
    data: Vec<OpenAIModelEntry>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelEntry {
    id: String,
    #[serde(default)]
    object: Option<String>,
    #[serde(default)]
    owned_by: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LlamaCppProps {
    #[serde(default)]
    default_generation_settings: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    total_slots: Option<u32>,
}

/// Errors from llama.cpp operations.
#[derive(Debug)]
pub enum LlamaCppError {
    NotRunning(String),
    ModelNotFound(String),
    Timeout(String),
    Api(String),
    Parse(String),
}

impl fmt::Display for LlamaCppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlamaCppError::NotRunning(msg) => write!(f, "llama.cpp server not running: {msg}"),
            LlamaCppError::ModelNotFound(name) => write!(f, "model not found in llama.cpp: {name}"),
            LlamaCppError::Timeout(msg) => write!(f, "request to llama.cpp timed out: {msg}"),
            LlamaCppError::Api(msg) => write!(f, "llama.cpp API error: {msg}"),
            LlamaCppError::Parse(msg) => write!(f, "llama.cpp parse error: {msg}"),
        }
    }
}

impl std::error::Error for LlamaCppError {}

/// Advanced sampling and constraint options supported natively by llama.cpp.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlamaCppOptions {
    /// GBNF grammar string for strictly constrained decoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    /// JSON schema for structured JSON output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
    /// Min-P sampling threshold (e.g. 0.05).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    /// Top-K sampling threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Mirostat sampling mode (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<i32>,
    /// Temperature for sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Max tokens to predict.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// Client for interacting with a llama.cpp HTTP server (`llama-server`).
pub struct LlamaCppClient {
    base_url: String,
    pub health_tracker: HealthTracker,
    client: reqwest::Client,
}

impl LlamaCppClient {
    /// Create a new client pointing at the given llama.cpp server.
    pub fn new(base_url: &str, timeout_seconds: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_seconds))
            .build()
            .unwrap_or_default();

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            health_tracker: HealthTracker::new(),
            client,
        }
    }

    /// Create a client with default settings (http://localhost:8080, 30s timeout).
    pub fn default_client() -> Self {
        Self::new("http://localhost:8080", 30)
    }

    /// Check if llama.cpp server is reachable and responsive, returning roundtrip latency in seconds.
    pub async fn ping(&self) -> Result<f64, LlamaCppError> {
        let start = Instant::now();

        // Try /health first
        let health_url = format!("{}/health", self.base_url);
        let resp = self.client.get(&health_url).send().await;

        match resp {
            Ok(r) if r.status().is_success() => Ok(start.elapsed().as_secs_f64()),
            Ok(r) if r.status().as_u16() == 503 => {
                // 503 in llama.cpp indicates server is up but model is currently loading
                Ok(start.elapsed().as_secs_f64())
            }
            _ => {
                // Fallback to /props or /v1/models
                let props_url = format!("{}/props", self.base_url);
                let resp2 = self.client.get(&props_url).send().await.map_err(|e| {
                    if e.is_connect() {
                        LlamaCppError::NotRunning(e.to_string())
                    } else if e.is_timeout() {
                        LlamaCppError::Timeout(e.to_string())
                    } else {
                        LlamaCppError::Api(e.to_string())
                    }
                })?;

                if resp2.status().is_success() {
                    Ok(start.elapsed().as_secs_f64())
                } else {
                    Err(LlamaCppError::Api(format!("HTTP {}", resp2.status())))
                }
            }
        }
    }

    /// List models available on the llama.cpp server.
    pub async fn list_models(&self) -> Result<Vec<LlamaCppModelInfo>, LlamaCppError> {
        let url = format!("{}/v1/models", self.base_url);

        let response = self.client.get(&url).send().await.map_err(|e| {
            if e.is_connect() {
                LlamaCppError::NotRunning(e.to_string())
            } else if e.is_timeout() {
                LlamaCppError::Timeout(e.to_string())
            } else {
                LlamaCppError::Api(e.to_string())
            }
        })?;

        if response.status().is_success() {
            if let Ok(models_resp) = response.json::<OpenAIModelsResponse>().await {
                if !models_resp.data.is_empty() {
                    return Ok(models_resp
                        .data
                        .into_iter()
                        .map(|m| LlamaCppModelInfo {
                            id: m.id,
                            object: m.object,
                            owned_by: m.owned_by,
                        })
                        .collect());
                }
            }
        }

        // Fallback to /props to check loaded model path
        let props_url = format!("{}/props", self.base_url);
        if let Ok(resp) = self.client.get(&props_url).send().await {
            if resp.status().is_success() {
                if let Ok(props) = resp.json::<LlamaCppProps>().await {
                    if let Some(settings) = props.default_generation_settings {
                        if let Some(model_val) = settings.get("model").and_then(|v| v.as_str()) {
                            let model_name = model_val
                                .replace('\\', "/")
                                .split('/')
                                .last()
                                .unwrap_or(model_val)
                                .to_string();
                            return Ok(vec![LlamaCppModelInfo {
                                id: model_name,
                                object: Some("model".to_string()),
                                owned_by: Some("llama.cpp".to_string()),
                            }]);
                        }
                    }
                }
            }
        }

        // If server is responsive, return generic model entry
        if self.ping().await.is_ok() {
            return Ok(vec![LlamaCppModelInfo {
                id: "llamacpp-default".to_string(),
                object: Some("model".to_string()),
                owned_by: Some("llama.cpp".to_string()),
            }]);
        }

        Ok(Vec::new())
    }

    /// Check health of llama.cpp server.
    pub async fn check_health(&mut self, model: &str) -> Result<ModelHealth, LlamaCppError> {
        let start = Instant::now();
        match self.ping().await {
            Ok(response_time) => {
                let health = ModelHealth {
                    status: HealthStatus::Healthy,
                    response_time,
                    last_check: crate::health::current_timestamp(),
                    error_message: None,
                };
                let key = if model.is_empty() { "llamacpp" } else { model };
                self.health_tracker.update(key, health.clone());
                Ok(health)
            }
            Err(e) => {
                let health = ModelHealth {
                    status: HealthStatus::Unavailable,
                    response_time: start.elapsed().as_secs_f64(),
                    last_check: crate::health::current_timestamp(),
                    error_message: Some(e.to_string()),
                };
                let key = if model.is_empty() { "llamacpp" } else { model };
                self.health_tracker.update(key, health.clone());
                Ok(health)
            }
        }
    }

    /// Get configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_client_has_correct_url() {
        let client = LlamaCppClient::default_client();
        assert_eq!(client.base_url(), "http://localhost:8080");
    }

    #[test]
    fn custom_client_trims_trailing_slash() {
        let client = LlamaCppClient::new("http://127.0.0.1:8080/", 15);
        assert_eq!(client.base_url(), "http://127.0.0.1:8080");
    }

    #[test]
    fn options_default_is_empty() {
        let opts = LlamaCppOptions::default();
        assert!(opts.grammar.is_none());
        assert!(opts.json_schema.is_none());
        assert!(opts.min_p.is_none());
    }
}
