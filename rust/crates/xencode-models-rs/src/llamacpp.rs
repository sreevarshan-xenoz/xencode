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

/// Pick the context window out of a `/props` response.
///
/// Measured on `llama-server` b10809: the running window is
/// `default_generation_settings.n_ctx`, and `/props` has **no** top-level
/// `n_ctx` — a build that starts the server with `-c 8192` reports 8192 there
/// and nothing elsewhere. The top-level key is read as a second candidate
/// because releases before that one carried it there; that shape is unverified
/// on this machine, so the nested key stays first and the whole response is
/// checked before either is believed.
///
/// Returns `None` when no candidate is a positive number that fits a `u32`.
pub fn context_window_from_props(props: &serde_json::Value) -> Option<u32> {
    if let Some(nested) = props
        .get("default_generation_settings")
        .and_then(|dgs| dgs.get("n_ctx"))
        .and_then(|v| v.as_u64())
    {
        if let Some(window) = as_window(nested) {
            return Some(window);
        }
    }
    props
        .get("n_ctx")
        .and_then(|v| v.as_u64())
        .and_then(as_window)
}

fn as_window(tokens: u64) -> Option<u32> {
    if tokens == 0 {
        return None;
    }
    u32::try_from(tokens).ok()
}

/// What a running server says about itself, read out of one `/props` response.
///
/// A field is `None` when the server answered without reporting it, which is a
/// different answer from `Some` and stays that way in the caller's wording.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServerReport {
    /// Tokens of context per slot — see [`context_window_from_props`].
    ///
    /// Per slot, not total: measured on b10809, a server started with
    /// `--ctx-size 2048 --parallel 2` reported 1024 here.
    pub context_tokens: Option<u32>,
    /// How many conversations the server serves at once (`total_slots`).
    pub slots: Option<u32>,
}

/// Pick both reported values out of a `/props` response.
///
/// Measured on b10809: `default_generation_settings.n_ctx` and a top-level
/// `total_slots` are both present; the launch's `--cache-type-k`,
/// `--cache-type-v` and `--batch-size` appear nowhere in the response, so a
/// preset that includes them cannot be checked this way.
pub fn report_from_props(props: &serde_json::Value) -> ServerReport {
    ServerReport {
        context_tokens: context_window_from_props(props),
        slots: props
            .get("total_slots")
            .and_then(|v| v.as_u64())
            .and_then(|s| u32::try_from(s).ok())
            .filter(|s| *s > 0),
    }
}

/// Pick the token count out of a `/tokenize` response, for the `text` that was
/// sent to get it.
///
/// Measured on `llama-server` b10809: the response is `{"tokens":[…]}`, one
/// integer per token, so the answer is the length of that array. `None` covers
/// both a body without such an array and — the case that makes `text` part of
/// this function — an empty array given back for text that was not empty. That
/// is what b10809 does to a request using a field name it does not read, and it
/// answers HTTP 200 while doing it, so believing zero here would report a prompt
/// as empty instead of reporting that nobody counted it.
pub fn counted_tokens_from_response(body: &serde_json::Value, text: &str) -> Option<u64> {
    let tokens = body.get("tokens").and_then(|v| v.as_array())?;
    if tokens.is_empty() && !text.is_empty() {
        return None;
    }
    Some(tokens.len() as u64)
}

/// Say what a server xencode started reports about itself, next to what it was
/// started with.
///
/// `/props` is the only witness, and it carries two of the preset's values —
/// the window and the slot count. The cache quantization and batch size appear
/// nowhere in the response, so this says nothing about them either way.
pub fn settings_check_line(label: &str, asked: ServerReport, got: ServerReport) -> String {
    if got.context_tokens.is_none() && got.slots.is_none() {
        return format!("{label}: the server reported no settings, so nothing is verified");
    }
    let summary = |report: ServerReport| match (report.context_tokens, report.slots) {
        (Some(ctx), Some(slots)) => format!("{ctx} tokens of context in {slots} slot(s)"),
        (Some(ctx), None) => format!("{ctx} tokens of context"),
        (None, Some(slots)) => format!("{slots} slot(s)"),
        (None, None) => String::new(),
    };
    let got_summary = summary(got);
    let asked_summary = summary(asked);
    if asked_summary.is_empty() {
        return format!("{label}: {got_summary}");
    }
    // Only a value both sides have can disagree; a server that reports half of
    // what was asked has not contradicted the other half.
    let contradicted = asked
        .context_tokens
        .zip(got.context_tokens)
        .is_some_and(|(want, reported)| want != reported)
        || asked
            .slots
            .zip(got.slots)
            .is_some_and(|(want, reported)| want != reported);
    if !contradicted {
        return format!("{label}: {got_summary}, as asked");
    }
    // A flag written twice is decided by the last value, so a difference usually
    // means `llama_cpp_args` had the last word rather than a preset failing to
    // apply.
    format!(
        "{label}: {got_summary}, not the {asked_summary} asked for — later flags win, so check llama_cpp_args"
    )
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
    /// Sampler seed. llama.cpp draws tokens from this, so a run is only
    /// repeatable if the seed goes over the wire with the temperature — leave it
    /// out and the server picks one per request, which is what it does today.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
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
                                .next_back()
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

        let mut resp = self
            .client
            .post(&url)
            .json(&payload)
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
        Err(LlamaCppError::Api(format!(
            "unload failed HTTP {status} - {body}"
        )))
    }

    /// Inline request to swap the loaded model. Convenience wrapper around
    /// [`Self::load_model`].
    pub async fn switch_model(&self, path: &str) -> Result<(), LlamaCppError> {
        self.load_model(path).await
    }

    /// The context window the server is actually running with.
    ///
    /// Reads `/props`; see [`context_window_from_props`] for which keys are
    /// consulted. `Ok(None)` means the server answered without saying — an old
    /// build, or one started with `--props` disabled — which is a different
    /// answer from "no server", and the caller treats it as "keep guessing".
    pub async fn context_window(&self) -> Result<Option<u32>, LlamaCppError> {
        Ok(self.report().await?.context_tokens)
    }

    /// Everything this client knows how to check about a running server, from
    /// one `/props` request.
    pub async fn report(&self) -> Result<ServerReport, LlamaCppError> {
        let url = format!("{}/props", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| LlamaCppError::Api(e.to_string()))?;
        if !resp.status().is_success() {
            return Ok(ServerReport::default());
        }
        let props = resp
            .json::<serde_json::Value>()
            .await
            .map_err(|e| LlamaCppError::Parse(e.to_string()))?;
        Ok(report_from_props(&props))
    }

    /// Ask until the server says something about itself, giving up after `tries`
    /// attempts `gap` apart and returning whatever it said by then.
    ///
    /// This exists because a server that has accepted a connection is not yet a
    /// server that has loaded a model: `llama-server` answers `/health` with 503
    /// while the weights come in, and `/props` with it — measured starting a
    /// 1.5B GGUF, where the settings were unreadable for the first seconds and
    /// reported `4096 tokens in 1 slot` afterwards. A check taken at that moment
    /// has nothing to say, and a client that reports "unverified" for a server
    /// that is merely still loading teaches the user to ignore it.
    pub async fn report_when_ready(&self, tries: u32, gap: std::time::Duration) -> ServerReport {
        for attempt in 0..tries {
            if let Ok(report) = self.report().await {
                if report.context_tokens.is_some() || report.slots.is_some() {
                    return report;
                }
            }
            if attempt + 1 < tries {
                tokio::time::sleep(gap).await;
            }
        }
        ServerReport::default()
    }

    /// How many tokens the server's own vocabulary says `text` is, asked of its
    /// `/tokenize` endpoint. `Ok(None)` means nobody counted it — see
    /// [`counted_tokens_from_response`] for when an answer is not believed.
    ///
    /// Special tokens are counted as the literal text they are written as:
    /// `parse_special` and `add_bos` are accepted and ignored on b10809, so a
    /// marker like `<|end|>` costs five tokens here instead of one. That only
    /// ever pushes the count up, which is the safe direction for a budget.
    pub async fn count_tokens(&self, text: &str) -> Result<Option<u64>, LlamaCppError> {
        let url = format!("{}/tokenize", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "content": text }))
            .send()
            .await
            .map_err(|e| LlamaCppError::Api(e.to_string()))?;
        if !resp.status().is_success() {
            return Ok(None);
        }
        let body = resp
            .json::<serde_json::Value>()
            .await
            .map_err(|e| LlamaCppError::Parse(e.to_string()))?;
        Ok(counted_tokens_from_response(&body, text))
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
        let n_predict = dgs.get("n_predict").and_then(|v| v.as_u64()).unwrap_or(0);
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

/// Assemble the command line for a server xencode starts itself.
///
/// Order is the whole point: the profile's preset first, then the model alias,
/// then whatever the user put in `llama_cpp_args` — last. `llama-server` runs
/// with the final value given for a flag, measured on b10809 by starting one
/// with `--ctx-size 8192 --ctx-size 2048`: `/props` reported 2048. Anything
/// earlier would let a preset quietly win an argument the user wrote down.
pub fn server_launch_args(
    profile_args: &[String],
    alias: Option<&str>,
    user_args: &[String],
) -> Vec<String> {
    let mut args = profile_args.to_vec();
    if let Some(alias) = alias {
        if !alias.trim().is_empty() && !args.iter().any(|a| a == "--alias") {
            args.push("--alias".to_string());
            args.push(alias.trim().to_string());
        }
    }
    args.extend(user_args.iter().cloned());
    args
}

/// The launch flags for the reasoning setting, or the one sentence explaining
/// why the setting says nothing usable.
///
/// `None`, a blank string and `auto` all say nothing: the model's own chat
/// template decides whether it thinks, which is what every build did before this
/// setting existed. `off` becomes `--reasoning off`; a whole number becomes
/// `--reasoning-budget <n>`.
///
/// **Why this is a launch flag and not a request field.** Measured on
/// `llama-server` b10809 with Qwen3-0.6B-Q4_K_M, asking one question at
/// temperature 0: the same request sent plain, sent with `reasoning_budget: 16`,
/// sent with `reasoning_effort: "minimal"`, and sent with
/// `chat_template_kwargs: {"thinking": false}`. The three that carried a field
/// came back identical to each other — 681 completion tokens, 1981 characters of
/// thinking — so a 16-token budget and an instruction not to think both did
/// nothing. Separately, a request carrying a key that exists nowhere in
/// llama.cpp was answered with HTTP 200 and no log line, which is the general
/// problem: a server accepting a field is not evidence that it read the field.
/// What does bite is the command line, measured the same way on the same model
/// and question:
///
/// | launched with | thinking characters | answer characters |
/// |---|---|---|
/// | (nothing) | 1352 | 354 |
/// | `--reasoning-budget 32` | 98 | 871 |
/// | `--reasoning-budget 0` | 0 | 2577 |
/// | `--reasoning off` | 0 | 358 |
///
/// Read the last two rows together, because they are the trap in this setting:
/// a budget of `0` is not the same as turning thinking off. It enters the
/// thinking block, ends it immediately, and the model goes on to write its
/// reasoning into the answer instead — 2577 characters of it for a one-number
/// question, against 358 from `--reasoning off`. A budget too small to finish a
/// chain does not fail and does not warn; the answer simply comes from a
/// half-finished plan. On this one question the unrestricted run and the
/// `--reasoning-budget 32` run both reached for the wrong number while `off` got
/// it right, which is a single sample on a 0.6B model and is recorded as that,
/// not as a rule about budgets.
///
/// The `/props` endpoint does not echo either flag, so unlike the context window
/// and the slot count there is no read-back: what was asked is visible in the
/// command line the program prints, and the effect is only visible in what comes
/// back.
pub fn reasoning_launch_args(setting: Option<&str>) -> Result<Vec<String>, String> {
    let Some(raw) = setting.map(str::trim) else {
        return Ok(Vec::new());
    };
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return Ok(Vec::new());
    }
    if raw.eq_ignore_ascii_case("off") {
        return Ok(vec!["--reasoning".to_string(), "off".to_string()]);
    }
    match raw.parse::<i64>() {
        Ok(budget) if budget >= 0 => Ok(vec![
            "--reasoning-budget".to_string(),
            budget.to_string(),
        ]),
        _ => Err(format!(
            "llama_cpp_reasoning must be \"auto\", \"off\" or a token budget like \"256\", not {raw:?}"
        )),
    }
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

/// Home-based directories that plausibly hold GGUF models, in preference order.
fn candidate_model_dirs(home: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut dirs = vec![
        home.join(".cache").join("llama.cpp"),
        home.join(".llama").join("models"),
        home.join(".local")
            .join("share")
            .join("llama.cpp")
            .join("models"),
    ];
    dirs.push(home.join("models"));
    dirs.push(home.join("models").join("llama.cpp"));
    dirs.push(std::path::PathBuf::from("models"));
    dirs
}

/// The user's home directory, tolerating missing env vars.
fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("USERPROFILE")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            let drive = std::env::var_os("HOMEDRIVE")?;
            let path = std::env::var_os("HOMEPATH")?;
            Some(
                std::path::PathBuf::from(drive.to_string_lossy().into_owned())
                    .join(path.to_string_lossy().into_owned()),
            )
        })
        .or_else(|| std::env::var_os("HOME").map(std::path::PathBuf::from))
}

/// Discover a GGUF model file to host when `llama_cpp_model_path` is empty.
///
/// Prefers an explicit path; otherwise scans the standard llama.cpp model
/// locations. `hint_name` (e.g. the `qwen3-4b` part of a `llamacpp:qwen3-4b`
/// model id) disambiguates when several GGUFs are present.
pub fn resolve_gguf_model(explicit: Option<&str>, hint_name: Option<&str>) -> Option<String> {
    let dirs = home_dir()
        .map(|h| candidate_model_dirs(&h))
        .unwrap_or_default();
    resolve_gguf_model_in(explicit, hint_name, &dirs)
}

/// Core of [`resolve_gguf_model`], parameterised over candidate directories so
/// it is testable without touching the real home directory.
fn resolve_gguf_model_in(
    explicit: Option<&str>,
    hint_name: Option<&str>,
    dirs: &[std::path::PathBuf],
) -> Option<String> {
    if let Some(explicit) = explicit {
        if !explicit.trim().is_empty() {
            return Some(explicit.trim().to_string());
        }
    }

    // Collect *.gguf files from each candidate dir, plus one level of subdirs
    // (common layout: `models/<model-name>/<model>.gguf`). The match key keeps
    // the containing folder name so a hint can match on either.
    let mut found: Vec<(std::path::PathBuf, String)> = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_gguf = |p: &std::path::Path| {
                p.is_file()
                    && p.extension()
                        .map(|e| e.eq_ignore_ascii_case("gguf"))
                        .unwrap_or(false)
            };
            if is_gguf(&path) {
                let key = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_ascii_lowercase();
                found.push((path, key));
                continue;
            }
            if path.is_dir() {
                let folder_key = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_ascii_lowercase();
                if let Ok(inner) = std::fs::read_dir(&path) {
                    for child in inner.flatten() {
                        let child_path = child.path();
                        if is_gguf(&child_path) {
                            let stem = child_path
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_ascii_lowercase();
                            let key = if folder_key.contains(&stem) || stem.contains(&folder_key) {
                                folder_key.clone()
                            } else {
                                format!("{folder_key}/{stem}")
                            };
                            found.push((child_path, key));
                        }
                    }
                }
            }
        }
    }
    if found.is_empty() {
        return None;
    }
    if let Some(hint) = hint_name {
        let hint_lower = hint.to_ascii_lowercase();
        if let Some(matched) = found.iter().find(|(_, key)| key.contains(&hint_lower)) {
            return Some(matched.0.to_string_lossy().into_owned());
        }
    }
    if found.len() == 1 {
        return Some(found[0].0.to_string_lossy().into_owned());
    }
    None
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
        // Whitespace-only explicit path behaves exactly like no explicit path —
        // independent of whether llama-server happens to be present on PATH.
        assert_eq!(find_llama_server(Some("  ")), find_llama_server(None));
    }

    #[test]
    fn resolve_gguf_prefers_explicit_path() {
        assert_eq!(
            resolve_gguf_model_in(
                Some("D:\\models\\qwen3-4b.gguf"),
                None,
                &[std::path::PathBuf::from("C:\\nonesuch")]
            ),
            Some("D:\\models\\qwen3-4b.gguf".to_string())
        );
    }

    #[test]
    fn resolve_gguf_hint_disambiguates_multiple_files() {
        let dir = std::env::temp_dir().join(format!("xencode-gguf-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("llama3.2.gguf"), b"x").unwrap();
        std::fs::write(dir.join("qwen3-4b.gguf"), b"x").unwrap();

        let found = resolve_gguf_model_in(None, Some("qwen3-4b"), std::slice::from_ref(&dir));
        assert_eq!(
            found,
            Some(dir.join("qwen3-4b.gguf").to_string_lossy().into_owned())
        );

        // No hint + multiple candidates is ambiguous.
        assert_eq!(
            resolve_gguf_model_in(None, None, std::slice::from_ref(&dir)),
            None
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_gguf_hint_matches_nested_model_folder() {
        // Mirrors the real layout: `~/models/<model>/<model>.gguf`.
        let dir = std::env::temp_dir().join(format!("xencode-gguf-nest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("llama3.2")).unwrap();
        std::fs::create_dir_all(dir.join("qwen3-4b")).unwrap();
        std::fs::write(dir.join("llama3.2").join("llama3.2-Q4_K_M.gguf"), b"x").unwrap();
        std::fs::write(dir.join("qwen3-4b").join("Qwen3-4B-Q4_K_M.gguf"), b"x").unwrap();

        assert_eq!(
            resolve_gguf_model_in(None, Some("qwen3-4b"), std::slice::from_ref(&dir)),
            Some(
                dir.join("qwen3-4b")
                    .join("Qwen3-4B-Q4_K_M.gguf")
                    .to_string_lossy()
                    .into_owned()
            )
        );
        // Ambiguous without a hint (one flat file + two nested families found).
        assert_eq!(
            resolve_gguf_model_in(None, None, std::slice::from_ref(&dir)),
            None
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_gguf_single_file_without_hint() {
        let dir = std::env::temp_dir().join(format!("xencode-gguf-single-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("only.gguf"), b"x").unwrap();

        assert_eq!(
            resolve_gguf_model_in(None, None, std::slice::from_ref(&dir)),
            Some(dir.join("only.gguf").to_string_lossy().into_owned())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_gguf_no_files_returns_none() {
        let dir = std::env::temp_dir().join(format!("xencode-gguf-empty-{}", std::process::id()));
        assert_eq!(
            resolve_gguf_model_in(None, None, std::slice::from_ref(&dir)),
            None
        );
    }

    #[test]
    fn timings_defaults_zero() {
        let t = LlamaCppTimings::default();
        assert_eq!(t.tokens_generated, 0);
        assert_eq!(t.predicted_per_second, 0.0);
        assert_eq!(t.total_seconds, 0.0);
    }

    /// `/props` as answered by `llama-server` b10809 started with `-c 8192`,
    /// captured from the running server on this machine and trimmed to the keys
    /// that matter here. Note what is absent: there is no top-level `n_ctx`.
    fn props_b10809() -> serde_json::Value {
        serde_json::json!({
            "bos_token": "<|endoftext|>",
            "build_info": "b10809-5266f24da7",
            "chat_template": "{%- if tools %}",
            "default_generation_settings": {
                "n_ctx": 8192,
                "params": { "n_predict": -1, "temperature": 0.8, "top_k": 40 }
            },
            "endpoint_props": false,
            "eos_token": "<|im_end|>",
            "model_alias": "dolphin",
            "model_ftype": "Q4_K - Medium",
            "model_path": "/models/Dolphin3.0-Qwen2.5-1.5B-Q4_K_M.gguf",
            "total_slots": 4
        })
    }

    #[test]
    fn context_window_is_read_from_the_generation_settings() {
        assert_eq!(context_window_from_props(&props_b10809()), Some(8192));
    }

    #[test]
    fn context_window_falls_back_to_a_top_level_report() {
        let props = serde_json::json!({ "n_ctx": 4096, "total_slots": 1 });
        assert_eq!(context_window_from_props(&props), Some(4096));
    }

    #[test]
    fn context_window_is_absent_when_the_server_does_not_report_one() {
        let empty = serde_json::json!({ "build_info": "b10809", "total_slots": 4 });
        assert_eq!(context_window_from_props(&empty), None);
        let zero = serde_json::json!({ "default_generation_settings": { "n_ctx": 0 } });
        assert_eq!(context_window_from_props(&zero), None);
        let text = serde_json::json!({ "default_generation_settings": { "n_ctx": "8192" } });
        assert_eq!(context_window_from_props(&text), None);
        let huge = serde_json::json!({ "n_ctx": 5_000_000_000_u64 });
        assert_eq!(context_window_from_props(&huge), None);
    }

    #[test]
    fn a_usable_nested_window_beats_an_unusable_top_level_one() {
        let props = serde_json::json!({
            "n_ctx": 0,
            "default_generation_settings": { "n_ctx": 2048 }
        });
        assert_eq!(context_window_from_props(&props), Some(2048));
    }

    /// `/tokenize` as answered by `llama-server` b10809 to the prose sentence
    /// `"The context budget now knows the window it is spending."`, captured
    /// from the running server on this machine.
    #[test]
    fn a_token_count_is_the_length_of_the_token_array() {
        let body = serde_json::json!({
            "tokens": [785, 2266, 8039, 1431, 8788, 279, 3241, 432, 374, 10164, 13]
        });
        assert_eq!(
            counted_tokens_from_response(&body, "The context budget now knows…"),
            Some(11)
        );
    }

    #[test]
    fn a_response_without_a_token_array_is_not_an_answer() {
        for body in [
            serde_json::json!({}),
            serde_json::json!({ "token_count": 11 }),
            serde_json::json!({ "tokens": "785 2266" }),
            serde_json::json!({ "error": "unsupported" }),
        ] {
            assert_eq!(counted_tokens_from_response(&body, "some text"), None);
        }
    }

    /// Zero tokens is a real answer about an empty string and a fake answer
    /// about anything else: b10809 replies `{"tokens":[]}` with HTTP 200 to a
    /// request whose field name it does not read.
    #[test]
    fn zero_tokens_is_only_believed_for_text_that_was_actually_empty() {
        let empty = serde_json::json!({ "tokens": [] });
        assert_eq!(counted_tokens_from_response(&empty, ""), Some(0));
        assert_eq!(counted_tokens_from_response(&empty, "hello"), None);
    }

    /// Live check against a running `llama-server` — needs a real server, so it
    /// is skipped by default. Point it at one with
    /// `XENCODE_TEST_LLAMA_URL=http://127.0.0.1:8099 cargo test -p
    /// xencode-models-rs -- --ignored`, and start that server with a window you
    /// know, since the assertion is only "it reported one".
    #[tokio::test]
    #[ignore]
    async fn a_running_server_reports_its_own_window() {
        let url = std::env::var("XENCODE_TEST_LLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let reported = LlamaCppClient::new(&url, 5)
            .context_window()
            .await
            .expect("server did not answer /props");
        let tokens = reported.expect("server reported no window");
        assert!(tokens > 0, "reported window {tokens} is not usable");
        println!("{url} is running a {tokens}-token window");
    }

    /// The same server counting a sentence it is given, with the arithmetic the
    /// budgeter would have done instead (`ceil(chars / 4)`) printed beside it —
    /// the comparison is the reason to ask. Needs a real server, so it is skipped
    /// by default; see [`a_running_server_reports_its_own_window`] for the URL
    /// variable.
    #[tokio::test]
    #[ignore]
    async fn a_running_server_counts_the_text_it_is_given() {
        let url = std::env::var("XENCODE_TEST_LLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let text = "The context budget now knows the window it is spending.";
        let counted = LlamaCppClient::new(&url, 5)
            .count_tokens(text)
            .await
            .expect("server did not answer /tokenize");
        let tokens = counted.expect("server returned no usable count for non-empty text");
        let chars = text.len();
        let estimated = chars.div_ceil(4) as u64;
        assert!(tokens > 0, "counted {tokens} tokens for {chars} characters");
        println!("{url}: {chars} characters = {tokens} tokens counted, {estimated} estimated");
    }

    /// Waiting costs the wait and nothing else: a server that never answers
    /// leaves the report empty rather than hanging, so the caller can still say
    /// "unverified" and move on. Nothing listens on port 9 here, so the requests
    /// are refused on the loopback interface.
    #[tokio::test]
    async fn waiting_for_a_server_that_never_answers_gives_up_empty() {
        let client = LlamaCppClient::new("http://127.0.0.1:9", 1);
        let report = client
            .report_when_ready(2, std::time::Duration::from_millis(10))
            .await;
        assert_eq!(report, ServerReport::default());
    }

    /// The slot count beside the window, from the same `/props` request. Needs a
    /// real server; see [`a_running_server_reports_its_own_window`] for the URL
    /// variable.
    #[tokio::test]
    #[ignore]
    async fn a_running_server_reports_its_slot_count() {
        let url = std::env::var("XENCODE_TEST_LLAMA_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        let report = LlamaCppClient::new(&url, 5)
            .report()
            .await
            .expect("server did not answer /props");
        let slots = report.slots.expect("server reported no slot count");
        assert!(slots > 0, "reported {slots} slots");
        println!(
            "{url}: {slots} slot(s), {} tokens of context each",
            report.context_tokens.unwrap_or(0)
        );
    }

    /// A flag written twice is not a conflict — `llama-server` runs with the last
    /// value, measured on b10809 by starting one with `--ctx-size 8192
    /// --ctx-size 2048` and reading 2048 back from `/props`. That is what makes
    /// the ordering here load-bearing: the user's own arguments have to be the
    /// later ones or the preset would overrule the person who wrote them.
    #[test]
    fn what_the_user_configured_is_the_last_word() {
        let args = server_launch_args(
            &["--ctx-size".to_string(), "8192".to_string()],
            Some("tiny"),
            &["--ctx-size".to_string(), "32768".to_string()],
        );
        assert_eq!(
            args,
            [
                "--ctx-size",
                "8192",
                "--alias",
                "tiny",
                "--ctx-size",
                "32768",
            ]
        );
    }

    /// An alias already in the preset is not said a second time, and a blank one
    /// is not said at all — `llama-server` takes `--alias` to mean "the next
    /// argument is the name", so a bare one would swallow the flag after it.
    #[test]
    fn an_alias_is_said_once_and_never_blank() {
        let args = server_launch_args(
            &["--alias".to_string(), "preset".to_string()],
            Some("tiny"),
            &[],
        );
        assert_eq!(args, ["--alias", "preset"]);

        let added = server_launch_args(
            &["--ctx-size".to_string(), "4096".to_string()],
            Some("tiny"),
            &[],
        );
        assert_eq!(added, ["--ctx-size", "4096", "--alias", "tiny"]);

        let none = server_launch_args(&["--ctx-size".to_string(), "4096".to_string()], None, &[]);
        assert_eq!(none, ["--ctx-size", "4096"]);

        let blank = server_launch_args(&[], Some("   "), &[]);
        assert!(blank.is_empty(), "{blank:?} carries a blank alias");
    }

    /// The three things the reasoning setting can mean, said as flags. `off` and
    /// a budget are different sentences — `--reasoning off` tells the template
    /// not to think at all, `--reasoning-budget N` lets it think for N tokens —
    /// so they cannot be folded into one "thinking: yes/no" switch.
    #[test]
    fn the_reasoning_setting_becomes_the_flag_that_means_it() {
        let cases: &[(&str, &[&str])] = &[
            ("off", &["--reasoning", "off"]),
            ("OFF", &["--reasoning", "off"]),
            ("256", &["--reasoning-budget", "256"]),
            ("0", &["--reasoning-budget", "0"]),
            ("  1024  ", &["--reasoning-budget", "1024"]),
            ("auto", &[]),
            ("", &[]),
        ];
        for (setting, want) in cases {
            let got = reasoning_launch_args(Some(setting)).expect("{setting} is a reasoning mode");
            assert_eq!(
                got.iter().map(String::as_str).collect::<Vec<_>>(),
                want.to_vec(),
                "for {setting:?}"
            );
        }
        assert!(reasoning_launch_args(None).unwrap().is_empty());
    }

    /// A value that names no mode is refused in the sentence a user can act on,
    /// rather than passed to `llama-server` — which accepts a flag it cannot
    /// parse by refusing to start at all, or (for a budget) by never hearing
    /// about it.
    #[test]
    fn a_reasoning_setting_that_names_nothing_is_refused_with_the_words() {
        for bad in ["-1", "lots", "256.5", "on", "1e3"] {
            let err = reasoning_launch_args(Some(bad))
                .err()
                .unwrap_or_else(|| panic!("{bad:?} was accepted as a reasoning setting"));
            assert!(
                err.contains("\"auto\", \"off\" or a token budget") && err.contains(bad),
                "refusal for {bad:?} does not say what is allowed: {err}"
            );
        }
    }

    /// The three answers a start-up check can give, in the words the user sees.
    #[test]
    fn a_server_either_agrees_disagrees_or_says_nothing() {
        let asked = ServerReport {
            context_tokens: Some(8192),
            slots: Some(1),
        };
        assert_eq!(
            settings_check_line(
                "BALANCED preset",
                asked,
                ServerReport {
                    context_tokens: Some(8192),
                    slots: Some(1)
                }
            ),
            "BALANCED preset: 8192 tokens of context in 1 slot(s), as asked"
        );
        assert_eq!(
            settings_check_line(
                "BALANCED preset",
                asked,
                ServerReport {
                    context_tokens: Some(2048),
                    slots: Some(1)
                }
            ),
            "BALANCED preset: 2048 tokens of context in 1 slot(s), not the 8192 tokens of context in 1 slot(s) asked for — later flags win, so check llama_cpp_args"
        );
        assert_eq!(
            settings_check_line("BALANCED preset", asked, ServerReport::default()),
            "BALANCED preset: the server reported no settings, so nothing is verified"
        );
        // A server that answers half of it has not contradicted the other half.
        assert_eq!(
            settings_check_line(
                "BALANCED preset",
                asked,
                ServerReport {
                    context_tokens: Some(8192),
                    slots: None
                }
            ),
            "BALANCED preset: 8192 tokens of context, as asked"
        );
    }

    /// Shape read straight off a b10809 `/props` response for a server started
    /// with `--ctx-size 4096 --parallel 1`.
    #[test]
    fn a_server_says_its_window_and_its_slots() {
        let props = serde_json::json!({
            "default_generation_settings": { "n_ctx": 4096, "params": { "seed": 4294967295u64 } },
            "total_slots": 1,
            "build_info": "b10809-5266f24da7",
        });
        assert_eq!(
            report_from_props(&props),
            ServerReport {
                context_tokens: Some(4096),
                slots: Some(1),
            }
        );
    }

    /// A slot count that is absent, zero, or not a number says nothing. `0` is
    /// not "no parallelism" — it is a server that has not told us, and the caller
    /// has to be able to tell that apart from a real answer.
    #[test]
    fn a_server_that_does_not_report_its_slots_says_so_by_not_reporting_them() {
        for props in [
            serde_json::json!({ "default_generation_settings": { "n_ctx": 8192 } }),
            serde_json::json!({ "total_slots": 0 }),
            serde_json::json!({ "total_slots": "one" }),
            serde_json::json!({}),
        ] {
            assert_eq!(report_from_props(&props).slots, None, "{props}");
        }
    }
}
