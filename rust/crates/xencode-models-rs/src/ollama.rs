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
            let resp2 = self
                .client
                .get(&tags_url)
                .send()
                .await
                .map_err(|e| OllamaError::Api(e.to_string()))?;
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

    /// Select the model to use from the ones installed, in the order the advice
    /// table in force on this machine prefers.
    pub async fn get_smart_default(&self) -> Result<Option<String>, OllamaError> {
        self.smart_default(&crate::advice::active_preference())
            .await
    }

    /// [`OllamaClient::get_smart_default`] with the preference order supplied by
    /// the caller — which is how a user's own `model_advice.json` decides,
    /// instead of the order baked into the binary.
    pub async fn smart_default(
        &self,
        preference: &[String],
    ) -> Result<Option<String>, OllamaError> {
        let models = self.list_models().await?;
        Ok(pick_chat_model(&models, preference).map(|m| m.name.clone()))
    }

    /// Get the base URL this client is configured with.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// Choose from the installed models in the order `preference` lists them.
///
/// A preference entry is matched as a substring of the tag, case-insensitively,
/// which is what lets one row (`"qwen3"`) stand for every size of it that
/// happens to be installed (`qwen3:4b`, `qwen3:14b`). Embedding models are
/// skipped: they answer a chat request by producing vectors, which reads as the
/// agent having nothing to say rather than as the wrong model being loaded.
pub fn pick_chat_model<'a>(
    models: &'a [ModelInfo],
    preference: &[String],
) -> Option<&'a ModelInfo> {
    if models.is_empty() {
        return None;
    }
    let chat_models: Vec<&ModelInfo> = models
        .iter()
        .filter(|m| !m.name.to_lowercase().contains("embed"))
        .collect();
    if chat_models.is_empty() {
        return Some(&models[0]);
    }
    for wanted in preference {
        let wanted = wanted.to_lowercase();
        if wanted.is_empty() {
            continue;
        }
        for model in &chat_models {
            if model.name.to_lowercase().contains(&wanted) {
                return Some(model);
            }
        }
    }
    Some(chat_models[0])
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

    fn installed(names: &[&str]) -> Vec<ModelInfo> {
        names
            .iter()
            .map(|name| ModelInfo {
                name: (*name).to_string(),
                size: 0,
                digest: String::new(),
                modified_at: String::new(),
            })
            .collect()
    }

    fn prefs(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_string()).collect()
    }

    #[test]
    fn an_installed_family_is_preferred_over_whatever_happens_to_be_first() {
        let models = installed(&["mistral:7b", "llama3.2:1b", "qwen3:4b"]);
        let picked = pick_chat_model(&models, &prefs(&["qwen3", "llama3.2"]))
            .unwrap()
            .name
            .clone();
        assert_eq!(picked, "qwen3:4b");
    }

    #[test]
    fn the_shipped_preference_finds_a_model_its_row_does_not_name_exactly() {
        // The bug the old list had: it stored full tags like "qwen3:4b" and
        // matched them as substrings, so a machine holding "qwen3:8b" matched
        // nothing and the answer fell back to whatever came first from the API.
        let models = installed(&["nemotron:latest", "qwen3:8b"]);
        let picked = pick_chat_model(&models, &crate::advice::embedded_preference())
            .unwrap()
            .name
            .clone();
        assert_eq!(picked, "qwen3:8b");
    }

    #[test]
    fn a_machine_with_nothing_preferred_still_gets_an_answer_and_not_a_shrug() {
        let models = installed(&["anything:latest", "other:1b"]);
        let picked = pick_chat_model(&models, &prefs(&["qwen3"]))
            .unwrap()
            .name
            .clone();
        assert_eq!(picked, "anything:latest");
        assert!(pick_chat_model(&[], &prefs(&["qwen3"])).is_none());
    }

    #[test]
    fn an_embedding_model_is_not_offered_as_the_chat_default() {
        let models = installed(&["nomic-embed-text:latest", "phi4-mini:latest"]);
        let picked = pick_chat_model(&models, &prefs(&["qwen3"]))
            .unwrap()
            .name
            .clone();
        assert_eq!(picked, "phi4-mini:latest");

        // A machine with only embeddings still gets the first model rather than
        // nothing: refusing to answer is not better than the wrong answer here,
        // because the caller shows what was chosen.
        let only_embeddings = installed(&["nomic-embed-text:latest"]);
        let picked = pick_chat_model(&only_embeddings, &prefs(&["qwen3"]))
            .unwrap()
            .name
            .clone();
        assert_eq!(picked, "nomic-embed-text:latest");
    }

    #[test]
    fn an_empty_preference_row_does_not_swallow_the_choice() {
        // A hand-written advice file with `"ollama_preference": ["", "qwen3"]`
        // used to match the empty string against every tag and return the first.
        let models = installed(&["mistral:7b", "qwen3:4b"]);
        let picked = pick_chat_model(&models, &prefs(&["", "qwen3"]))
            .unwrap()
            .name
            .clone();
        assert_eq!(picked, "qwen3:4b");
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
