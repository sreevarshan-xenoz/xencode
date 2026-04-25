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
    timeout_seconds: u64,
    pub health_tracker: HealthTracker,
}

impl OllamaClient {
    /// Create a new client pointing at the given Ollama instance.
    pub fn new(base_url: &str, timeout_seconds: u64) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout_seconds,
            health_tracker: HealthTracker::new(),
        }
    }

    /// Create a client with default settings (localhost:11434, 30s timeout).
    pub fn default_client() -> Self {
        Self::new("http://localhost:11434", 30)
    }

    /// List all installed models.
    pub fn list_models(&self) -> Result<Vec<ModelInfo>, OllamaError> {
        let url = format!("{}/api/tags", self.base_url);
        let response = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_seconds))
            .call()
            .map_err(|e| match e {
                ureq::Error::Transport(ref t) => {
                    if t.kind() == ureq::ErrorKind::ConnectionFailed {
                        OllamaError::NotRunning(e.to_string())
                    } else {
                        OllamaError::Api(e.to_string())
                    }
                }
                _ => OllamaError::Api(e.to_string()),
            })?;

        let tags: TagsResponse = response
            .into_json()
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

    /// Check the health of a specific model by sending a tiny prompt.
    pub fn check_health(&mut self, model: &str) -> Result<ModelHealth, OllamaError> {
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
        let result = ureq::post(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_seconds))
            .send_json(&payload);

        match result {
            Ok(_) => {
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
            Err(ureq::Error::Transport(t)) => {
                let health = ModelHealth {
                    status: if t.kind() == ureq::ErrorKind::ConnectionFailed {
                        HealthStatus::Unavailable
                    } else {
                        HealthStatus::Error
                    },
                    response_time: start.elapsed().as_secs_f64(),
                    last_check: crate::health::current_timestamp(),
                    error_message: Some(t.to_string()),
                };
                self.health_tracker.update(model, health.clone());
                Ok(health)
            }
            Err(e) => {
                let health = ModelHealth {
                    status: HealthStatus::Error,
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
    pub fn get_smart_default(&self) -> Result<Option<String>, OllamaError> {
        let models = self.list_models()?;
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
    #[test]
    #[ignore]
    fn list_models_integration() {
        let client = OllamaClient::default_client();
        let models = client.list_models().unwrap();
        println!("Found {} models", models.len());
        for model in &models {
            println!("  - {} ({} bytes)", model.name, model.size);
        }
        // Just verify it doesn't panic; we can't assert on specific models
    }

    #[test]
    #[ignore]
    fn smart_default_integration() {
        let client = OllamaClient::default_client();
        let default = client.get_smart_default().unwrap();
        println!("Smart default: {:?}", default);
    }

    #[test]
    #[ignore]
    fn health_check_integration() {
        let mut client = OllamaClient::default_client();
        let models = client.list_models().unwrap();
        if let Some(model) = models.first() {
            let health = client.check_health(&model.name).unwrap();
            println!("Health for {}: {:?}", model.name, health);
        }
    }
}
