use std::fmt;
use std::process::Stdio;
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

/// Completion timing and token usage reported by a llama.cpp server.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlamaCppTimings {
    /// Number of tokens generated.
    pub tokens_generated: u64,
    /// Number of tokens consumed as prompt.
    pub tokens_evaluated: u64,
    /// Generation throughput in tokens per second (tok/s).
    pub predicted_per_second: f64,
    /// Prompt processing throughput in tokens per second.
    pub prompt_per_second: f64,
    /// Total wall clock time of the generation in seconds.
    pub total_seconds: f64,
}

impl LlamaCppTimings {
    /// Compute throughput from a token count and elapsed time.
    pub fn from_elapsed(tokens: u64, elapsed: f64) -> Self {
        Self {
            tokens_generated: tokens,
            predicted_per_second: if elapsed > 0.0 {
                tokens as f64 / elapsed
            } else {
                0.0
            },
            ..Self::default()
        }
    }
}

#[derive(Debug, Deserialize)]
struct LlamaCppProps {
    #[serde(default)]
    default_generation_settings: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    total_slots: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct LlamaCppLoadResponse {
    #[serde(default)]
    error: Option<String>,
}

/// A running `llama-server` process that xencode spawned (auto-start support).
#[derive(Debug)]
pub struct LlamaServerProcess {
    child: std::process::Child,
    pub base_url: String,
}

impl LlamaServerProcess {
    /// The OS process id of the spawned server (if available).
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Check whether the spawned server is still running.
    pub fn is_running(&mut self) -> bool {
        self.child.try_wait().map(|s| s.is_none()).unwrap_or(false)
    }

    /// Terminate the spawned server process.
    pub fn stop(&mut self) -> Result<(), LlamaCppError> {
        self.child
            .kill()
            .map_err(|e| LlamaCppError::Api(format!("failed to stop llama-server: {e}")))
    }
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

    /// Load a GGUF model into the server via POST /v1/models/load.
    ///
    /// `path` is the model identifier as reported by the server (typically a
    /// `--model` or `--alias` value, or a model registry name).
    pub async fn load_model(&self, path: &str) -> Result<(), LlamaCppError> {
        let url = format!("{}/v1/models/load", self.base_url);
        let payload = serde_json::json!({ "model": path });

        let mut resp = self.client.post(&url).json(&payload).send().await.map_err(|e| {
            if e.is_connect() {
                LlamaCppError::NotRunning(e.to_string())
            } else if e.is_timeout() {
                LlamaCppError::Timeout(e.to_string())
            } else {
                LlamaCppError::Api(e.to_string())
            }
        })?;

        if resp.status().is_success() {
            return Ok(());
        }

        // llama.cpp returns HTTP 503 while a model is loading/swapping; poll until it settles.
        let mut attempts = 0;
        while resp.status().as_u16() == 503 && attempts < 40 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            resp = self
                .client
                .post(&url)
                .json(&payload)
                .send()
                .await
                .map_err(|e| LlamaCppError::Api(e.to_string()))?;
            attempts += 1;
        }

        if resp.status().is_success() {
            return Ok(());
        }

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if let Ok(parsed) = serde_json::from_str::<LlamaCppLoadResponse>(&body) {
            if let Some(ref err) = parsed.error {
                return Err(LlamaCppError::Api(err.clone()));
            }
        }
        Err(LlamaCppError::Api(format!("HTTP {status} - {body}")))
    }

    /// Unload the currently loaded model via POST /v1/models/unload.
    pub async fn unload_models(&self) -> Result<(), LlamaCppError> {
        let url = format!("{}/v1/models/unload", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({}))
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() {
                    LlamaCppError::NotRunning(e.to_string())
                } else if e.is_timeout() {
                    LlamaCppError::Timeout(e.to_string())
                } else {
                    LlamaCppError::Api(e.to_string())
                }
            })?;

        if resp.status().is_success() {
            return Ok(());
        }

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(LlamaCppError::Api(format!("unload failed HTTP {status} - {body}")))
    }

    /// Inline request to swap the loaded model. Convenience wrapper around
    /// [`Self::load_model`].
    pub async fn switch_model(&self, path: &str) -> Result<(), LlamaCppError> {
        self.load_model(path).await
    }

    /// Query token-generation timing and usage for a model from the native
    /// `/props` endpoint. Returns `None` when the server does not expose it.
    pub async fn timings(&self) -> Result<Option<LlamaCppTimings>, LlamaCppError> {
        let url = format!("{}/props", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LlamaCppError::Api(e.to_string()))?;
        if !resp.status().is_success() {
            return Ok(None);
        }
        let props = resp
            .json::<serde_json::Value>()
            .await
            .map_err(|e| LlamaCppError::Parse(e.to_string()))?;
        let dgs = props
            .get("default_generation_settings")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let n_predict = dgs
            .get("n_predict")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        Ok(Some(LlamaCppTimings {
            tokens_generated: n_predict,
            ..LlamaCppTimings::default()
        }))
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

/// Start a `llama-server` process hosting the given GGUF model.
///
/// Returns a handle to the spawned process. The process keeps running until
/// [`LlamaServerProcess::stop`] is called (or the child exits on its own).
pub fn start_llama_server(
    executable: &str,
    model_path: &str,
    port: u16,
    extra_args: &[&str],
) -> Result<LlamaServerProcess, LlamaCppError> {
    let mut cmd = std::process::Command::new(executable);
    cmd.arg("--host").arg("127.0.0.1");
    cmd.arg("--port").arg(port.to_string());
    cmd.arg("--model").arg(model_path);
    cmd.args(extra_args);
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());

    let child = cmd
        .spawn()
        .map_err(|e| LlamaCppError::Api(format!("failed to start llama-server: {e}")))?;

    Ok(LlamaServerProcess {
        child,
        base_url: format!("http://127.0.0.1:{}", port),
    })
}

/// Resolve the `llama-server` binary; tries the explicit path supplied by the
/// user, then common names on `PATH`.
pub fn find_llama_server(executable: Option<&str>) -> Option<String> {
    if let Some(exe) = executable {
        if !exe.trim().is_empty() {
            return Some(exe.trim().to_string());
        }
    }
    for candidate in [
        "llama-server",
        "llama-server.exe",
        "llama_server",
        "llama_cli",
    ] {
        if lookup_in_path(candidate) {
            return Some(candidate.to_string());
        }
    }
    None
}

fn lookup_in_path(name: &str) -> bool {
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let full = dir.join(name);
            if full.is_file() {
                return true;
            }
        }
    }
    false
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

    #[test]
    fn timings_from_elapsed_computes_rate() {
        let t = LlamaCppTimings::from_elapsed(100, 2.0);
        assert_eq!(t.tokens_generated, 100);
        assert_eq!(t.predicted_per_second, 50.0);
    }

    #[test]
    fn timings_from_elapsed_zero_time_is_zero_rate() {
        let t = LlamaCppTimings::from_elapsed(100, 0.0);
        assert_eq!(t.predicted_per_second, 0.0);
    }

    #[test]
    fn find_llama_server_prefers_explicit_path() {
        assert_eq!(
            find_llama_server(Some("C:\\tools\\llama-server.exe")),
            Some("C:\\tools\\llama-server.exe".to_string())
        );
        assert_eq!(find_llama_server(Some("  ")), None);
    }

    #[test]
    fn timings_defaults_zero() {
        let t = LlamaCppTimings::default();
        assert_eq!(t.tokens_generated, 0);
        assert_eq!(t.predicted_per_second, 0.0);
        assert_eq!(t.total_seconds, 0.0);
    }
}
