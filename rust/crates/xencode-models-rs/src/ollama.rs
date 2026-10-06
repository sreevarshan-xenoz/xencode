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

/// What a running Ollama reports about one of its own models.
///
/// Read from `POST /api/show`, which answers without loading or running the
/// model. Two things here cannot be known any other way: how large a window the
/// weights were trained for, and which of `tools`, `thinking` and the rest the
/// model claims. Both change what is safe to put in a request — see
/// [`OllamaClient::show_model`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelShow {
    /// The largest context the model was trained for, in tokens. Ollama reports
    /// it under a key prefixed by the architecture — `qwen2.context_length` for
    /// a Qwen file — so the prefix is not part of the lookup. `None` when the
    /// model says nothing, which is not the same as zero.
    pub trained_context_tokens: Option<u32>,
    /// The capabilities as the server words them: `tools`, `thinking`,
    /// `completion`, `embedding`, and whatever a future release adds.
    pub capabilities: Vec<String>,
}

impl ModelShow {
    /// Whether this model may be asked to reason before answering. Asking one
    /// that cannot is not ignored: Ollama 0.34.4 answers such a request with
    /// HTTP 400 and `"<model>" does not support thinking`.
    pub fn can_think(&self) -> bool {
        self.has("thinking")
    }

    /// Whether this model understands a `tools` list in the request.
    pub fn can_use_tools(&self) -> bool {
        self.has("tools")
    }

    fn has(&self, capability: &str) -> bool {
        self.capabilities
            .iter()
            .any(|found| found.eq_ignore_ascii_case(capability))
    }
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

/// Turn transport/connect error messages into user-readable prose, stripping
/// leaked internal REST routes like `/api/show`, `/api/generate`, or `/api/version`.
pub fn sanitize_not_running(msg: &str) -> String {
    if let Some(url_start) = msg.find("url (") {
        let after = &msg[url_start + 5..];
        if let Some(url_end) = after.find(')') {
            let full_url = &after[..url_end];
            let base_url = if let Some(api_pos) = full_url.find("/api/") {
                &full_url[..api_pos]
            } else {
                full_url
            };
            if msg.to_lowercase().contains("refused") || msg.to_lowercase().contains("connect") {
                return format!("connection refused: nothing is listening on {base_url}");
            }
            if msg.to_lowercase().contains("timed out") || msg.to_lowercase().contains("timeout") {
                return format!("request timed out connecting to {base_url}");
            }
            return format!("cannot connect to {base_url}");
        }
    }
    if let Some(pos) = msg.find("/api/") {
        if let Some(end) = msg[pos..].find(|c: char| c.is_whitespace() || c == ')' || c == ':') {
            let mut s = msg[..pos].to_string();
            s.push_str(&msg[pos + end..]);
            return s;
        }
    }
    msg.to_string()
}

impl OllamaError {
    /// The raw transport/system error string without prose sanitization.
    pub fn raw_message(&self) -> &str {
        match self {
            OllamaError::NotRunning(s)
            | OllamaError::ModelNotFound(s)
            | OllamaError::Timeout(s)
            | OllamaError::Api(s)
            | OllamaError::Parse(s) => s,
        }
    }
}

impl fmt::Display for OllamaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OllamaError::NotRunning(msg) => {
                let prose = sanitize_not_running(msg);
                write!(f, "Ollama not running: {prose}")
            }
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
/// Talks to the HTTP routes a running `ollama serve` answers on: `/api/tags`
/// for what is installed, `/api/show` for what one model can do, `/api/chat`
/// for the answers themselves.
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

    /// Ask the server what it knows about one model, without loading it.
    ///
    /// This is the only way to learn, before a request goes out, how large a
    /// window the weights can hold and whether the model can be asked to think.
    /// Both matter because Ollama enforces them at the ends of that range: a
    /// window larger than the model was trained for is quietly reduced to it,
    /// and asking a model that cannot think to do so is refused outright.
    ///
    /// It is also a new surface that can fail — a model that is not installed,
    /// an older server that does not answer this route, or a name that does not
    /// resolve all end here as an error rather than as an empty answer, so a
    /// caller keeps whatever it already knew instead of learning a wrong zero.
    pub async fn show_model(&self, name: &str) -> Result<ModelShow, OllamaError> {
        let url = format!("{}/api/show", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "model": name }))
            .send()
            .await
            .map_err(|e| {
                if e.is_connect() {
                    OllamaError::NotRunning(e.to_string())
                } else if e.is_timeout() {
                    OllamaError::Timeout(e.to_string())
                } else {
                    OllamaError::Api(e.to_string())
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            return Err(if status.as_u16() == 404 {
                OllamaError::ModelNotFound(name.to_string())
            } else {
                OllamaError::Api(format!("HTTP {status}"))
            });
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| OllamaError::Parse(e.to_string()))?;
        Ok(parse_show(&body))
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
                        error_message: Some(sanitize_not_running(&e.to_string())),
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
                    error_message: Some(sanitize_not_running(&e.to_string())),
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

/// Pick out the two things this program acts on from a `/api/show` answer.
///
/// Separated from the request so the shapes a server actually returns can be
/// held against a captured response instead of against a running Ollama. The
/// context length is looked up by suffix rather than by full key because the
/// architecture is the prefix — `qwen2.context_length` here, `llama.context_length`
/// for a Llama file — and a missing one stays `None`: the model not saying is
/// not the same as the model holding nothing.
fn parse_show(body: &serde_json::Value) -> ModelShow {
    let capabilities = body
        .get("capabilities")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    let trained_context_tokens = body
        .get("model_info")
        .and_then(serde_json::Value::as_object)
        .and_then(|info| {
            info.iter().find_map(|(key, value)| {
                if key.ends_with(".context_length") {
                    value.as_u64()
                } else {
                    None
                }
            })
        })
        .and_then(|tokens| u32::try_from(tokens).ok());

    ModelShow {
        trained_context_tokens,
        capabilities,
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

    /// A `/api/show` answer captured from Ollama 0.34.4 on 2026-09-27 for the
    /// Qwen3 1.7B GGUF this program's advice table points at. `template` and
    /// `modelfile` are replaced with a note because they are 4,761 characters of
    /// chat template nothing here reads; every other key and value is what the
    /// server sent, including the tokenizer arrays Ollama leaves empty for a
    /// model it did not convert itself.
    fn show_qwen3_1_7b() -> serde_json::Value {
        serde_json::json!({
            "modelfile": "<elided>",
            "template": "<the chat template, 4761 characters, elided>",
            "details": {
                "parent_model": "qwen3-1.7b.gguf",
                "format": "gguf",
                "family": "qwen3",
                "families": ["qwen3"],
                "parameter_size": "1.7B",
                "quantization_level": "Q4_K_M"
            },
            "model_info": {
                "general.architecture": "qwen3",
                "general.basename": "Qwen3-1.7B",
                "general.file_type": 15,
                "general.parameter_count": 1720574976_u64,
                "general.quantization_version": 2,
                "general.quantized_by": "Unsloth",
                "general.repo_url": "https://huggingface.co/unsloth",
                "general.size_label": "1.7B",
                "general.type": "model",
                "quantize.imatrix.chunks_count": 685,
                "quantize.imatrix.dataset": "unsloth_calibration_Qwen3-1.7B.txt",
                "quantize.imatrix.entries_count": 196,
                "quantize.imatrix.file": "Qwen3-1.7B-GGUF/imatrix_unsloth.dat",
                "qwen3.attention.head_count": 16,
                "qwen3.attention.head_count_kv": 8,
                "qwen3.attention.key_length": 128,
                "qwen3.attention.layer_norm_rms_epsilon": 1e-06,
                "qwen3.attention.value_length": 128,
                "qwen3.block_count": 28,
                "qwen3.context_length": 40960,
                "qwen3.embedding_length": 2048,
                "qwen3.feed_forward_length": 6144,
                "qwen3.rope.freq_base": 1000000,
                "tokenizer.ggml.add_bos_token": false,
                "tokenizer.ggml.eos_token_id": 151645,
                "tokenizer.ggml.merges": [],
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.padding_token_id": 151654,
                "tokenizer.ggml.pre": "qwen2",
                "tokenizer.ggml.token_type": [],
                "tokenizer.ggml.tokens": []
            },
            "capabilities": ["tools", "thinking", "completion"],
            "modified_at": "2026-09-27T13:14:27.619991156+05:30"
        })
    }

    /// The same question for the Qwen2.5 0.5B file, captured from the same
    /// server a few minutes earlier: architecture `qwen2`, so the window arrives
    /// as `qwen2.context_length`, and no `thinking` among the capabilities —
    /// which is the pair these two bodies exist to hold the code against.
    fn show_qwen2_0_5b() -> serde_json::Value {
        serde_json::json!({
            "modelfile": "<elided>",
            "template": "<the chat template, 2509 characters, elided>",
            "details": {
                "parent_model": "qwen25-0.5b.gguf",
                "format": "gguf",
                "family": "qwen2",
                "families": ["qwen2"],
                "parameter_size": "630.17M",
                "quantization_level": "Q4_K_M"
            },
            "model_info": {
                "general.architecture": "qwen2",
                "general.file_type": 15,
                "general.finetune": "qwen2.5-0.5b-instruct",
                "general.parameter_count": 630167424_u64,
                "general.quantization_version": 2,
                "general.size_label": "630M",
                "general.type": "model",
                "general.version": "v0.1",
                "qwen2.attention.head_count": 14,
                "qwen2.attention.head_count_kv": 2,
                "qwen2.attention.layer_norm_rms_epsilon": 1e-06,
                "qwen2.block_count": 24,
                "qwen2.context_length": 32768,
                "qwen2.embedding_length": 896,
                "qwen2.feed_forward_length": 4864,
                "qwen2.rope.freq_base": 1000000,
                "tokenizer.ggml.add_bos_token": false,
                "tokenizer.ggml.bos_token_id": 151643,
                "tokenizer.ggml.eos_token_id": 151645,
                "tokenizer.ggml.merges": [],
                "tokenizer.ggml.model": "gpt2",
                "tokenizer.ggml.padding_token_id": 151643,
                "tokenizer.ggml.pre": "qwen2",
                "tokenizer.ggml.token_type": [],
                "tokenizer.ggml.tokens": []
            },
            "capabilities": ["tools", "completion"],
            "modified_at": "2026-09-27T13:07:44.710647144+05:30"
        })
    }

    #[test]
    fn a_captured_answer_names_the_window_and_says_the_model_can_think() {
        let show = parse_show(&show_qwen3_1_7b());
        assert_eq!(show.trained_context_tokens, Some(40_960));
        assert!(show.can_think());
        assert!(show.can_use_tools());
        assert_eq!(show.capabilities.len(), 3);
    }

    #[test]
    fn the_window_is_found_under_whatever_architecture_prefix_the_file_carries() {
        let show = parse_show(&show_qwen2_0_5b());
        assert_eq!(show.trained_context_tokens, Some(32_768));
        assert!(!show.can_think());
        assert!(show.can_use_tools());
    }

    #[test]
    fn a_capability_is_matched_on_the_word_rather_than_on_its_position_or_case() {
        // Both servers listed `tools` first, so a check that only looked at the
        // head of the array would pass here and fail on the next model.
        let show = parse_show(&serde_json::json!({ "capabilities": ["completion", "Thinking"] }));
        assert!(show.can_think());
        assert!(!show.can_use_tools());
    }

    #[test]
    fn an_answer_saying_nothing_reads_as_unknown_rather_than_as_zero() {
        let show = parse_show(&serde_json::json!({ "details": { "format": "gguf" } }));
        assert_eq!(show.trained_context_tokens, None);
        assert!(show.capabilities.is_empty());
        assert!(!show.can_think());
    }

    #[test]
    fn capabilities_that_are_not_text_are_not_counted() {
        // A server that answers `[null, 42]` says nothing this program should act
        // on; a request built on a capability nobody claimed is worse than one
        // built on nothing.
        let show = parse_show(&serde_json::json!({ "capabilities": [null, 42, "thinking"] }));
        assert_eq!(show.capabilities, vec!["thinking".to_string()]);
        assert!(show.can_think());
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

    #[test]
    fn sanitize_not_running_drops_leaked_routes() {
        let raw_show = "error sending request for url (http://localhost:11434/api/show): tcp connect error: Connection refused (os error 111)";
        let prose_show = sanitize_not_running(raw_show);
        assert!(!prose_show.contains("/api/show"), "prose should not contain /api/show: {prose_show}");
        assert!(prose_show.contains("nothing is listening on http://localhost:11434"), "{prose_show}");

        let err = OllamaError::NotRunning(raw_show.to_string());
        assert_eq!(err.raw_message(), raw_show);
        let displayed = err.to_string();
        assert!(!displayed.contains("/api/show"), "Display should not contain /api/show: {displayed}");
        assert!(displayed.contains("nothing is listening on http://localhost:11434"), "{displayed}");

        let raw_generate = "error sending request for url (http://localhost:11434/api/generate): tcp connect error: Connection refused";
        let prose_generate = sanitize_not_running(raw_generate);
        assert!(!prose_generate.contains("/api/generate"), "{prose_generate}");

        let raw_version = "error sending request for url (http://localhost:11434/api/version): tcp connect error: Connection refused";
        let prose_version = sanitize_not_running(raw_version);
        assert!(!prose_version.contains("/api/version"), "{prose_version}");
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
