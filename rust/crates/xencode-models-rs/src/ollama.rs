use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::health::{HealthStatus, HealthTracker, ModelHealth};

/// Information about an installed Ollama model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub modified_at: String,
}

/// Ollama API response for listing models.
#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<OllamaModel>,
}

/// Single model entry from the Ollama tags API.
#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    digest: String,
    #[serde(default)]
    modified_at: String,
}

/// Errors from Ollama operations.
#[derive(Debug)]
pub enum OllamaError {
    /// Cannot reach the Ollama service.
    NotRunning(String),
    /// Model not found.
    ModelNotFound(String),
    /// Request timed out.
    Timeout(String),
    /// Generic HTTP/API error.
    Api(String),
    /// JSON parsing error.
    Parse(String),
}

impl fmt::Display for OllamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OllamaError::NotRunning(msg) => write!(f, "Ollama not running: {msg}"),
            OllamaError::ModelNotFound(name) => write!(f, "model not found: {name}"),
            OllamaError::Timeout(msg) => write!(f, "request timed out: {msg}"),
            OllamaError::Api(msg) => write!(f, "API error: {msg}"),
            OllamaError::Parse(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for OllamaError {}

/// Client for interacting with the Ollama REST API.
///
/// Mirrors the model management functionality in `xencode/core/models.py`.
pub struct OllamaClient {
    base_url: String,
    pub health_tracker: HealthTracker,
    client: reqwest::Client,
}

impl OllamaClient {
    /// Create a new client pointing at the given Ollama instance.
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

    /// Create a client with default settings (localhost:11434, 30s timeout).
    pub fn default_client() -> Self {
        Self::new("http://localhost:11434", 30)
    }

    /// Check if Ollama service is reachable and responsive, returning latency in seconds.
    pub async fn ping(&self) -> Result<f64, OllamaError> {
        let url = format!("{}/api/version", self.base_url);
        let start = Instant::now();
        let response = self.client.get(&url).send().await.map_err(|e| {
            if e.is_connect() {
                OllamaError::NotRunning(e.to_string())
            } else if e.is_timeout() {
                OllamaError::Timeout(e.to_string())
            } else {
                OllamaError::Api(e.to_string())
            }
        })?;

        if response.status().is_success() {
            Ok(start.elapsed().as_secs_f64())
        } else {
            // Fallback to tags endpoint if /api/version isn't available
            let tags_url = format!("{}/api/tags", self.base_url);
            let resp2 = self.client.get(&tags_url).send().await.map_err(|e| {
                OllamaError::Api(e.to_string())
            })?;
            if resp2.status().is_success() {
                Ok(start.elapsed().as_secs_f64())
            } else {
                Err(OllamaError::Api(format!("HTTP {}", resp2.status())))
            }
        }
    }

    /// List all installed models.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, OllamaError> {
        let url = format!("{}/api/tags", self.base_url);

        let response = self.client.get(&url).send().await.map_err(|e| {
            if e.is_connect() {
                OllamaError::NotRunning(e.to_string())
            } else if e.is_timeout() {
                OllamaError::Timeout(e.to_string())
            } else {
                OllamaError::Api(e.to_string())
            }
        })?;

        let tags: TagsResponse = response
            .json()
            .await
            .map_err(|e| OllamaError::Parse(e.to_string()))?;

        Ok(tags
            .models
            .into_iter()
            .map(|m| ModelInfo {
                name: m.name,
                size: m.size,
                digest: m.digest,
                modified_at: m.modified_at,
            })
            .collect())
    }

    /// Check the health of a specific model by sending a tiny prompt, or ping if model is empty.
    pub async fn check_health(&mut self, model: &str) -> Result<ModelHealth, OllamaError> {
        if model.is_empty() {
            let start = Instant::now();
            return match self.ping().await {
                Ok(response_time) => {
                    let health = ModelHealth {
                        status: HealthStatus::Healthy,
                        response_time,
                        last_check: crate::health::current_timestamp(),
                        error_message: None,
                    };
                    Ok(health)
                }
                Err(e) => {
                    let health = ModelHealth {
                        status: HealthStatus::Unavailable,
                        response_time: start.elapsed().as_secs_f64(),
                        last_check: crate::health::current_timestamp(),
                        error_message: Some(e.to_string()),
                    };
                    Ok(health)
                }
            };
        }

        let url = format!("{}/api/generate", self.base_url);
        let payload = serde_json::json!({
            "model": model,
            "prompt": "hi",
            "stream": false,
            "options": {
                "num_predict": 1
            }
        });

        let start = Instant::now();
        let result = self.client.post(&url).json(&payload).send().await;

        match result {
            Ok(resp) if resp.status().is_success() => {
                let response_time = start.elapsed().as_secs_f64();
                let health = ModelHealth {
                    status: HealthStatus::Healthy,
                    response_time,
                    last_check: crate::health::current_timestamp(),
                    error_message: None,
                };
                self.health_tracker.update(model, health.clone());
                Ok(health)
            }
            Ok(resp) => {
                let status = resp.status();
                let msg = resp.text().await.unwrap_or_default();
                let err_msg = if status.as_u16() == 404 {
                    format!("Model '{model}' not found in Ollama (run 'ollama pull {model}')")
                } else {
                    format!("HTTP {}: {}", status, msg)
                };
                let health = ModelHealth {
                    status: HealthStatus::Error,
                    response_time: start.elapsed().as_secs_f64(),
                    last_check: crate::health::current_timestamp(),
                    error_message: Some(err_msg),
                };
                self.health_tracker.update(model, health.clone());
                Ok(health)
            }
            Err(e) => {
                let status = if e.is_connect() {
                    HealthStatus::Unavailable
                } else {
                    HealthStatus::Error
                };
                let health = ModelHealth {
                    status,
                    response_time: start.elapsed().as_secs_f64(),
                    last_check: crate::health::current_timestamp(),
                    error_message: Some(e.to_string()),
                };
                self.health_tracker.update(model, health.clone());
                Ok(health)
            }
        }
    }

    /// Select the best available model using a preferred-model priority list.
    ///
    /// Mirrors the Python `get_smart_default_model()` logic.
    pub async fn get_smart_default(&self) -> Result<Option<String>, OllamaError> {
        let models = self.list_models().await?;
        if models.is_empty() {
            return Ok(None);
        }

        // Filter out embedding models
        let chat_models: Vec<&ModelInfo> = models
            .iter()
            .filter(|m| !m.name.contains("embed"))
            .collect();

        if chat_models.is_empty() {
            return Ok(Some(models[0].name.clone()));
        }

        // Preferred models in priority order
        let preferred = [
            "qwen2.5:7b",
            "qwen2.5:3b",
            "qwen3:4b",
            "llama3.1:8b",
            "llama3.2:3b",
            "mistral:7b",
            "phi3:mini",
            "gemma2:2b",
        ];

        for pref in &preferred {
            for model in &chat_models {
                if model.name.to_lowercase().contains(pref) {
                    return Ok(Some(model.name.clone()));
                }
            }
        }

        // Fallback to first available chat model
        Ok(Some(chat_models[0].name.clone()))
    }

    /// Get the base URL this client is configured with.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_client_has_correct_url() {
        let client = OllamaClient::default_client();
        assert_eq!(client.base_url(), "http://localhost:11434");
    }

    #[test]
    fn custom_client_trims_trailing_slash() {
        let client = OllamaClient::new("http://myhost:11434/", 10);
        assert_eq!(client.base_url(), "http://myhost:11434");
    }

    // Integration tests below require a running Ollama instance.
    // They are ignored by default and can be run with:
    //   cargo test -- --ignored
    #[tokio::test]
    #[ignore]
    async fn list_models_integration() {
        let client = OllamaClient::default_client();
        let models = client.list_models().await.unwrap();
        println!("Found {} models", models.len());
        for model in &models {
            println!("  - {} ({} bytes)", model.name, model.size);
        }
        // Just verify it doesn't panic; we can't assert on specific models
    }

    #[tokio::test]
    #[ignore]
    async fn smart_default_integration() {
        let client = OllamaClient::default_client();
        let default = client.get_smart_default().await.unwrap();
        println!("Smart default: {:?}", default);
    }

    #[tokio::test]
    #[ignore]
    async fn health_check_integration() {
        let mut client = OllamaClient::default_client();
        let models = client.list_models().await.unwrap();
        if let Some(model) = models.first() {
            let health = client.check_health(&model.name).await.unwrap();
            println!("Health for {}: {:?}", model.name, health);
        }
    }
}
