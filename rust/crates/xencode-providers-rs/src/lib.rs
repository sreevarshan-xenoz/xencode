use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use xencode_models_rs::{LlamaCppClient, LlamaCppOptions, LlamaCppTimings, OllamaClient};

pub mod anthropic;
pub mod capabilities;
pub mod compatible;
pub mod egress;
mod frames;
pub mod gemini;
pub mod playback;
pub mod qwen;
pub mod retry;
pub mod tools;

use retry::RetryConfig;

pub use capabilities::{capabilities_for, ModelCapabilities};
pub use compatible::OpenAICompatibleProvider;
pub use egress::{chain_for, classify, provider_for, url_host, Egress, EgressPolicy, RoutingFacts};
pub use tools::{
    advise_tools, background_tools, command_tools, file_tools, plan_tools, AgentStep, AgentTurn,
    ToolCall, ToolDefinition,
};

#[derive(Debug)]
pub enum ProviderError {
    Network(String),
    /// An error response from a provider.
    ///
    /// `status` carries the HTTP status as data. It used to be recoverable only
    /// by searching `message` for the digits, which meant a permanent 400 whose
    /// body happened to mention "500" — a token limit, a request id, a model
    /// name — was retried as if it were a server error.
    Api {
        status: Option<u16>,
        message: String,
    },
    Parse(String),
    /// The prompt was refused before it left this machine, because the route
    /// the model name resolves to needs a permission the configuration has not
    /// given it. Not a network condition: no retry and no fallback candidate
    /// can make an unpermitted route permitted.
    Egress(String),
}

impl ProviderError {
    /// An error response carrying an HTTP status.
    ///
    /// Formats the message as `"{provider} {status} - {body}"`, the convention
    /// already used across the providers.
    pub fn api(provider: &str, status: impl Into<u16>, body: impl fmt::Display) -> Self {
        let status = status.into();
        ProviderError::Api {
            status: Some(status),
            message: format!("{provider} {status} - {body}"),
        }
    }

    /// An API-level error with no HTTP status behind it — a missing key, an
    /// unusable response shape. Never retriable.
    pub fn api_message(message: impl Into<String>) -> Self {
        ProviderError::Api {
            status: None,
            message: message.into(),
        }
    }

    /// The HTTP status, when this error came from an HTTP response.
    pub fn status(&self) -> Option<u16> {
        match self {
            ProviderError::Api { status, .. } => *status,
            _ => None,
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::Network(msg) => write!(f, "network error: {msg}"),
            ProviderError::Api { message, .. } => write!(f, "API error: {message}"),
            ProviderError::Parse(msg) => write!(f, "parse error: {msg}"),
            ProviderError::Egress(msg) => write!(f, "egress denied: {msg}"),
        }
    }
}

impl std::error::Error for ProviderError {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

/// Message body: plain text, or OpenAI-style content parts mixing text with
/// image URLs (data URLs from the analysis crate's `to_data_url`).
///
/// `untagged` keeps the wire shape backward compatible: `Text` serializes as
/// a bare string — every text-only payload is byte-identical to before —
/// while `Parts` serializes as the `[{"type":"text",...},
/// {"type":"image_url",...}]` array OpenAI-compatible endpoints accept.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One element of a multi-part message body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlPart },
}

/// Image payload. `url` is a `data:{mime};base64,{data}` URL (see
/// `to_data_url`); raw base64 without the prefix is also accepted by the
/// Ollama renderer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrlPart {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub detail: Option<String>,
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

impl ChatMessage {
    /// Plain-text message. Every existing `content: string_value` call site
    /// keeps compiling unchanged via `From<String>`.
    pub fn text(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Text(text.into()),
        }
    }

    /// User message carrying one text part plus an image part per URL.
    /// Contract: URLs are data URLs from the image intake pipeline.
    pub fn user_with_images(text: impl Into<String>, urls: Vec<String>) -> Self {
        let text = text.into();
        let mut parts = Vec::with_capacity(urls.len() + 1);
        if !text.is_empty() {
            parts.push(ContentPart::Text { text });
        }
        parts.extend(urls.into_iter().map(|url| ContentPart::ImageUrl {
            image_url: ImageUrlPart { url, detail: None },
        }));
        Self {
            role: "user".to_string(),
            content: MessageContent::Parts(parts),
        }
    }

    /// Concatenated text parts (the whole body for `Text`). What text-only
    /// backends and response parsing consume.
    pub fn text_content(&self) -> String {
        match &self.content {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                })
                .collect(),
        }
    }

    /// Image URLs in part order; empty for text-only messages.
    pub fn image_urls(&self) -> Vec<&str> {
        match &self.content {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.as_str()),
                    ContentPart::Text { .. } => None,
                })
                .collect(),
        }
    }

    pub fn has_images(&self) -> bool {
        match &self.content {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }
}

/// Split a `data:{mime};base64,{data}` URL. Returns `None` for anything else
/// (raw base64, http URLs), letting each renderer choose its fallback.
pub fn split_data_url(url: &str) -> Option<(&str, &str)> {
    url.strip_prefix("data:")?.split_once(";base64,")
}

/// Ollama `/api/chat` message: text content plus a top-level `images` array
/// of raw base64 (Ollama does not take OpenAI content blocks or data-URL
/// prefixes). The `images` key is omitted for text-only messages, so those
/// payloads are byte-identical to before.
pub(crate) fn to_ollama_value(msg: &ChatMessage) -> serde_json::Value {
    let mut value = serde_json::json!({"role": msg.role, "content": msg.text_content()});
    let images: Vec<&str> = msg
        .image_urls()
        .into_iter()
        .map(|u| split_data_url(u).map(|(_, data)| data).unwrap_or(u))
        .collect();
    if !images.is_empty() {
        value["images"] =
            serde_json::Value::Array(images.into_iter().map(serde_json::Value::from).collect());
    }
    value
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: ChatMessage,
}

/// One complete NDJSON line from Ollama's `/api/chat`: forwards the text it
/// carries, replaces `calls` when it carries tool calls, and reports whether
/// the server said it was done. `calls` is `None` on the plain chat path, which
/// has no tools to expect.
fn ingest_ollama_line<F: FnMut(&str)>(
    line: &str,
    text: &mut String,
    calls: Option<&mut Vec<ToolCall>>,
    callback: &mut F,
) -> bool {
    if line.trim().is_empty() {
        return false;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
        return false;
    };
    if let Some(message) = value.get("message") {
        if let Ok(message) = serde_json::from_value::<ChatMessage>(message.clone()) {
            let content = message.text_content();
            if !content.is_empty() {
                callback(&content);
                text.push_str(&content);
            }
        }
        if let Some(offered) = message.get("tool_calls") {
            let parsed = tools::parse_ollama_calls(offered, "call_");
            if !parsed.is_empty() {
                if let Some(calls) = calls {
                    *calls = parsed;
                }
            }
        }
    }
    value.get("done").and_then(|d| d.as_bool()).unwrap_or(false)
}

/// OpenRouter's SSE framing: `data: {json}` lines whose first choice carries
/// the text delta, terminated by a `data: [DONE]` marker.
fn ingest_openrouter_line<F: FnMut(&str)>(line: &str, text: &mut String, callback: &mut F) {
    let line = line.trim();
    if line.is_empty() || line == "data: [DONE]" {
        return;
    }
    let Some(data) = line.strip_prefix("data: ") else {
        return;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
        return;
    };
    let Some(content) = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|choice| choice.get("delta"))
        .and_then(|delta| delta.get("content"))
        .and_then(|content| content.as_str())
    else {
        return;
    };
    if !content.is_empty() {
        callback(content);
        text.push_str(content);
    }
}

/// Extract the inner model identifier if `model` routes to the llama.cpp
/// provider (prefixes `llamacpp:`, `llama.cpp:`, or `llama:`).
///
/// Returns `None` when the model does not target llama.cpp.
fn llamacpp_target(model: &str) -> Option<&str> {
    model
        .strip_prefix("llamacpp:")
        .or_else(|| model.strip_prefix("llama.cpp:"))
        .or_else(|| {
            if model.starts_with("llama:") {
                model.strip_prefix("llama:")
            } else {
                None
            }
        })
}

/// Extract the inner model identifier if `model` routes to the custom
/// OpenAI-compatible endpoint (`remote:<model>`).
fn remote_target(model: &str) -> Option<&str> {
    model.strip_prefix("remote:")
}

/// ProviderManager abstracts over local and cloud models.
///
/// Supports Ollama (local), llama.cpp (local), OpenRouter (cloud), Qwen (cloud),
/// Gemini (cloud), Anthropic (cloud), and any OpenAI-compatible `/chat/completions`
/// server the user pointed at (`remote:`, which is also how a Colab GPU reaches
/// the laptop).
///
/// Features:
/// - Automatic provider routing based on model prefix
/// - Retry with exponential backoff on transient failures
/// - Health tracking across all providers
pub struct ProviderManager {
    ollama_client: OllamaClient,
    llama_cpp_client: Option<LlamaCppClient>,
    openrouter_api_key: Option<String>,
    qwen_api_key: Option<String>,
    gemini_api_key: Option<String>,
    anthropic_api_key: Option<String>,
    /// API root of the `remote:` endpoint; `None` until one is configured.
    remote_base_url: Option<String>,
    remote_api_key: Option<String>,
    retry_config: RetryConfig,
    /// Whether a request may leave this machine. Checked before every route in
    /// front of the network, and again when the agent's fallback chain is built.
    egress: EgressPolicy,
    client: reqwest::Client,
    /// Per-request silence bound (seconds). Streaming responses get a
    /// time-to-first-token cap, then run unbounded once tokens flow;
    /// non-streaming calls get a total cap. `0` disables both.
    request_timeout_secs: u64,
    /// Most recent llama.cpp generation timing (tokens + tok/s), if any.
    llamacpp_timings: Mutex<Option<LlamaCppTimings>>,
}

impl ProviderManager {
    pub fn new(
        ollama_client: OllamaClient,
        openrouter_api_key: Option<String>,
        qwen_api_key: Option<String>,
        gemini_api_key: Option<String>,
        anthropic_api_key: Option<String>,
    ) -> Self {
        let client = reqwest::Client::new();
        Self {
            ollama_client,
            llama_cpp_client: None,
            openrouter_api_key,
            qwen_api_key,
            gemini_api_key,
            anthropic_api_key,
            remote_base_url: None,
            remote_api_key: None,
            retry_config: RetryConfig::default(),
            egress: EgressPolicy::default(),
            client,
            request_timeout_secs: 0,
            llamacpp_timings: Mutex::new(None),
        }
    }

    /// Point the `remote:` prefix at an OpenAI-compatible server. An empty or
    /// whitespace `base_url` leaves the route unconfigured, so a `remote:`
    /// request reports that instead of dialling a guessed host.
    pub fn with_remote(mut self, base_url: &str, api_key: Option<String>) -> Self {
        let base_url = base_url.trim();
        if !base_url.is_empty() {
            self.remote_base_url = Some(base_url.to_string());
            self.remote_api_key = api_key.filter(|key| !key.trim().is_empty());
        }
        self
    }

    /// Set a llama.cpp client for local GGUF/llama-server inference.
    pub fn with_llama_cpp(mut self, client: LlamaCppClient) -> Self {
        self.llama_cpp_client = Some(client);
        self
    }

    /// Set a custom retry configuration.
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Bound silence at `secs` seconds per attempt (each try gets the full
    /// budget; timeouts surface as retriable network errors). Non-streaming
    /// calls get a total cap; streams get a time-to-first-token cap, then
    /// run unbounded once tokens flow. `0` disables both.
    pub fn with_request_timeout(mut self, secs: u64) -> Self {
        self.request_timeout_secs = secs;
        self
    }

    /// Get the current retry configuration.
    pub fn retry_config(&self) -> &RetryConfig {
        &self.retry_config
    }

    /// Generate a response asynchronously (resolves when the full response is ready).
    ///
    /// Routes to the appropriate provider based on the model prefix:
    /// - `anthropic:` → Anthropic Claude API
    /// - `qwen:` → Qwen cloud API
    /// - `google_gemini:` → Google Gemini API
    /// - `llamacpp:` / `llama.cpp:` / `llama:` → llama.cpp server
    /// - `remote:` → the configured OpenAI-compatible endpoint
    /// - contains `/` with OpenRouter key → OpenRouter
    /// - else → local Ollama
    ///
    /// Retries on transient errors (network failures, 5xx, 429) using exponential backoff.
    pub async fn generate(
        &self,
        model: &str,
        messages: &[ChatMessage],
    ) -> Result<String, ProviderError> {
        self.generate_with_options(model, messages, None).await
    }

    /// [`generate`][Self::generate] with explicit llama.cpp sampling options.
    pub async fn generate_with_options(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: Option<&LlamaCppOptions>,
    ) -> Result<String, ProviderError> {
        let model_owned = model.to_string();
        let messages_owned = messages.to_vec();
        let timeout_secs = self.request_timeout_secs;

        retry::retry_async(&self.retry_config, || async {
            if timeout_secs == 0 {
                self.generate_inner(&model_owned, &messages_owned, options)
                    .await
            } else {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(timeout_secs),
                    self.generate_inner(&model_owned, &messages_owned, options),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => Err(ProviderError::Network(format!(
                        "request timed out after {timeout_secs}s"
                    ))),
                }
            }
        })
        .await
    }

    /// Record token-usage timing observed from a llama.cpp completion.
    fn record_llamacpp_timings(&self, tokens: u64, elapsed: f64) {
        *self.llamacpp_timings.lock().unwrap() =
            Some(LlamaCppTimings::from_elapsed(tokens, elapsed));
    }

    /// Most recent llama.cpp generation timing (tokens generated + tok/s).
    pub fn last_llamacpp_timings(&self) -> Option<LlamaCppTimings> {
        self.llamacpp_timings.lock().unwrap().clone()
    }

    /// Read the llama.cpp timings and empty the slot, so a caller that adds up
    /// several requests in one turn counts each generation exactly once.
    pub fn take_llamacpp_timings(&self) -> Option<LlamaCppTimings> {
        self.llamacpp_timings.lock().unwrap().take()
    }

    /// Replace the egress policy. The default allows every route, so a caller
    /// that has not opted in sees exactly the behaviour this gate replaced.
    pub fn with_egress_policy(mut self, policy: EgressPolicy) -> Self {
        self.egress = policy;
        self
    }

    pub fn egress_policy(&self) -> EgressPolicy {
        self.egress
    }

    /// Where this model id would send the prompt, given the current
    /// configuration — the same prefix rules the routers below use.
    pub fn egress_of(&self, model: &str) -> Egress {
        classify(model, self.routing_facts())
    }

    /// The fallback order a turn may walk, and the candidates it skipped.
    ///
    /// A skipped candidate is not an error to swallow: the caller shows it, so
    /// "no fallback ran" is distinguishable from "no fallback was configured".
    pub fn fallback_chain(
        &self,
        primary: &str,
        configured: &[String],
    ) -> (Vec<String>, Vec<String>) {
        chain_for(primary, configured, self.egress, self.routing_facts())
    }

    /// The configuration that decides where a model id resolves to.
    fn routing_facts(&self) -> RoutingFacts<'_> {
        RoutingFacts {
            openrouter_key: self.openrouter_api_key.is_some(),
            remote_host: self.remote_host(),
        }
    }

    /// Host of the configured `remote:` endpoint, if any.
    fn remote_host(&self) -> Option<&str> {
        self.remote_base_url.as_deref().and_then(url_host)
    }

    /// Refuse a route the policy does not permit, before the prompt is built.
    ///
    /// The refusal names the model and the setting that would allow it. A
    /// deny-by-default posture that cannot be lifted without reading the source
    /// is not safety, it is an outage.
    fn check_egress(&self, model: &str) -> Result<(), ProviderError> {
        let egress = self.egress_of(model);
        if egress == Egress::Cloud && !self.egress.allow_cloud {
            return Err(ProviderError::Egress(format!(
                "`{model}` sends this conversation to an internet service and cloud \
                 models are not allowed. Allow them with `xencode config set \
                 allow_cloud_models true`, or choose a model that runs on this machine."
            )));
        }
        Ok(())
    }

    /// The configured OpenAI-compatible endpoint, or a message saying how to
    /// configure one. Never falls back to a default host: a mistyped or absent
    /// URL must fail loudly rather than POST the prompt somewhere else.
    fn remote_provider(&self) -> Result<compatible::OpenAICompatibleProvider, ProviderError> {
        match self.remote_base_url.as_deref() {
            Some(base_url) => Ok(compatible::OpenAICompatibleProvider::new(
                base_url,
                self.remote_api_key.clone(),
            )),
            None => Err(ProviderError::api_message(
                "No remote endpoint configured — set Settings → Remote Endpoint \
                 URL, or run `xencode config set remote_url <url>`"
                    .to_string(),
            )),
        }
    }

    /// Inner generate without retry wrapping (used by retry logic).
    async fn generate_inner(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: Option<&LlamaCppOptions>,
    ) -> Result<String, ProviderError> {
        self.check_egress(model)?;

        // Route based on model prefix
        if let Some(inner_model) = model.strip_prefix("anthropic:") {
            if let Some(ref key) = self.anthropic_api_key {
                let provider = anthropic::AnthropicProvider::new(key.clone(), None, None);
                return provider.generate(inner_model, messages, None).await;
            }
            return Err(ProviderError::api_message(
                "Anthropic API key not configured".to_string(),
            ));
        }

        if let Some(inner_model) = model.strip_prefix("qwen:") {
            if let Some(ref key) = self.qwen_api_key {
                let provider = qwen::QwenProvider::new(key.clone(), None);
                return provider.generate(inner_model, messages).await;
            }
            return Err(ProviderError::api_message(
                "Qwen API key not configured".to_string(),
            ));
        }

        if let Some(inner_model) = model.strip_prefix("google_gemini:") {
            if let Some(ref key) = self.gemini_api_key {
                let provider = gemini::GeminiProvider::new(key.clone(), None);
                return provider.generate(inner_model, messages, None, None).await;
            }
            return Err(ProviderError::api_message(
                "Google Gemini API key not configured".to_string(),
            ));
        }

        // llama.cpp route (models prefixed with "llamacpp:", "llama.cpp:", or "llama:")
        if let Some(inner_model) = llamacpp_target(model) {
            return self.generate_llamacpp(inner_model, messages, options).await;
        }

        // Custom OpenAI-compatible endpoint (models prefixed with "remote:")
        if let Some(inner_model) = remote_target(model) {
            let provider = self.remote_provider()?;
            let rendered = tools::render_history(messages, &[], tools::HistoryStyle::OpenAI);
            return provider.generate(inner_model, &rendered, "Remote").await;
        }

        // OpenRouter route (models with a slash, e.g. "openai/gpt-4")
        if model.contains('/') && self.openrouter_api_key.is_some() {
            return self.generate_nonstream_openrouter(model, messages).await;
        }

        // Default: local Ollama
        let url = format!("{}/api/chat", self.ollama_client.base_url());

        let payload = serde_json::json!({
            "model": model,
            "messages": messages.iter().map(to_ollama_value).collect::<Vec<_>>(),
            "stream": false
        });

        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let body: OllamaResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        Ok(body.message.text_content())
    }

    /// Generate a response and stream it token-by-token.
    ///
    /// Routes to the appropriate provider based on the model prefix:
    /// - `anthropic:` → Anthropic Claude API
    /// - `qwen:` → Qwen cloud API
    /// - `google_gemini:` → Google Gemini API
    /// - `remote:` → the configured OpenAI-compatible endpoint
    /// - contains `/` with OpenRouter key → OpenRouter
    /// - else → local Ollama
    ///
    /// Retries on transient errors (network failures, 5xx, 429) using exponential
    /// backoff — but only until the first token has been delivered to the
    /// callback. Once streaming has started, retrying would re-deliver the same
    /// tokens (duplicated output), so a mid-stream failure is surfaced as-is.
    pub async fn generate_stream<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        self.generate_stream_with_options(model, messages, None, callback)
            .await
    }

    /// [`generate_stream`][Self::generate_stream] with explicit llama.cpp
    /// sampling options passed through to the `llama-server` request.
    pub async fn generate_stream_with_options<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: Option<&LlamaCppOptions>,
        callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        let model_owned = model.to_string();
        let messages_owned = messages.to_vec();

        // Use Mutex for interior mutability so the callback can be shared across retry attempts.
        // The mutex is locked only during the synchronous callback invocation (per token),
        // not across await points, so it remains Send-compatible.
        let cb = Mutex::new(callback);
        // Tracks whether any token has already been delivered to the consumer.
        // A stream must not be retried once tokens have been emitted, because a
        // fresh attempt would re-deliver the same tokens (duplicate output).
        let emitted = AtomicBool::new(false);

        retry::retry_async_with_guard(
            &self.retry_config,
            || emitted.load(Ordering::SeqCst),
            || async {
                let cb_ref = &cb;
                let emitted_ref = &emitted;
                let take_token = |token: &str| {
                    emitted_ref.store(true, Ordering::SeqCst);
                    let mut guard = cb_ref.lock().unwrap();
                    guard(token);
                };
                // Bound silence, not duration: a hung server that never sends
                // a first token fails fast (retriable — nothing was emitted);
                // once tokens flow the attempt runs unbounded to completion.
                if self.request_timeout_secs == 0 {
                    return self
                        .generate_stream_inner(&model_owned, &messages_owned, options, take_token)
                        .await;
                }
                let mut attempt = Box::pin(self.generate_stream_inner(
                    &model_owned,
                    &messages_owned,
                    options,
                    take_token,
                ));
                tokio::select! {
                    result = &mut attempt => result,
                    _ = tokio::time::sleep(std::time::Duration::from_secs(
                        self.request_timeout_secs,
                    )) => {
                        if emitted_ref.load(Ordering::SeqCst) {
                            attempt.await
                        } else {
                            Err(ProviderError::Network(format!(
                                "no response within {}s",
                                self.request_timeout_secs
                            )))
                        }
                    }
                }
            },
        )
        .await
    }

    /// Streaming generation with tool definitions for the agentic loop.
    ///
    /// Returns the visible text plus any model-requested tool calls; the
    /// caller executes them and continues the loop with [`AgentTurn`]
    /// history. `tools` may be empty (plain step, no `tools` key is sent).
    ///
    /// Backends without a natively plumbed tool schema (Gemini, Anthropic)
    /// degrade to single-shot text — no tool calls, no error.
    ///
    /// Unlike [`generate_stream_with_options`], this path never retries:
    /// re-running a step could double-execute tools.
    pub async fn generate_stream_with_tools<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        history: &[AgentTurn],
        tools: &[ToolDefinition],
        options: Option<&LlamaCppOptions>,
        callback: F,
    ) -> Result<AgentStep, ProviderError>
    where
        F: FnMut(&str),
    {
        self.check_egress(model)?;

        if let Some(inner_model) = model.strip_prefix("anthropic:") {
            if let Some(ref key) = self.anthropic_api_key {
                let provider = anthropic::AnthropicProvider::new(key.clone(), None, None);
                let text = provider
                    .generate_stream(inner_model, messages, None, callback)
                    .await?;
                return Ok(AgentStep {
                    text,
                    tool_calls: Vec::new(),
                });
            }
            return Err(ProviderError::api_message(
                "Anthropic API key not configured".to_string(),
            ));
        }

        if let Some(inner_model) = model.strip_prefix("qwen:") {
            if let Some(ref key) = self.qwen_api_key {
                let provider = qwen::QwenProvider::new(key.clone(), None);
                let rendered =
                    tools::render_history(messages, history, tools::HistoryStyle::OpenAI);
                return provider
                    .generate_stream_with_tools(inner_model, &rendered, tools, callback)
                    .await;
            }
            return Err(ProviderError::api_message(
                "Qwen API key not configured".to_string(),
            ));
        }

        if let Some(inner_model) = model.strip_prefix("google_gemini:") {
            if let Some(ref key) = self.gemini_api_key {
                let provider = gemini::GeminiProvider::new(key.clone(), None);
                let text = provider
                    .generate_stream(inner_model, messages, None, None, callback)
                    .await?;
                return Ok(AgentStep {
                    text,
                    tool_calls: Vec::new(),
                });
            }
            return Err(ProviderError::api_message(
                "Google Gemini API key not configured".to_string(),
            ));
        }

        // llama.cpp route (models prefixed with "llamacpp:", "llama.cpp:", or "llama:")
        if let Some(inner_model) = llamacpp_target(model) {
            return self
                .generate_stream_llamacpp_with_tools(
                    inner_model,
                    messages,
                    history,
                    tools,
                    options,
                    callback,
                )
                .await;
        }

        // Custom OpenAI-compatible endpoint, with full tool-calling support
        if let Some(inner_model) = remote_target(model) {
            let provider = self.remote_provider()?;
            let rendered = tools::render_history(messages, history, tools::HistoryStyle::OpenAI);
            return provider
                .generate_stream_with_tools(inner_model, &rendered, tools, "Remote", callback)
                .await;
        }

        // OpenRouter route
        if model.contains('/') && self.openrouter_api_key.is_some() {
            return self
                .generate_stream_openrouter_with_tools(model, messages, history, tools, callback)
                .await;
        }

        // Default: local Ollama
        self.generate_stream_ollama_with_tools(model, messages, history, tools, callback)
            .await
    }

    /// Inner stream generate without retry wrapping.
    async fn generate_stream_inner<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: Option<&LlamaCppOptions>,
        callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        self.check_egress(model)?;

        // Route based on model prefix
        if let Some(inner_model) = model.strip_prefix("anthropic:") {
            if let Some(ref key) = self.anthropic_api_key {
                let provider = anthropic::AnthropicProvider::new(key.clone(), None, None);
                return provider
                    .generate_stream(inner_model, messages, None, callback)
                    .await;
            }
            return Err(ProviderError::api_message(
                "Anthropic API key not configured".to_string(),
            ));
        }

        if let Some(inner_model) = model.strip_prefix("qwen:") {
            if let Some(ref key) = self.qwen_api_key {
                let provider = qwen::QwenProvider::new(key.clone(), None);
                return provider
                    .generate_stream(inner_model, messages, callback)
                    .await;
            }
            return Err(ProviderError::api_message(
                "Qwen API key not configured".to_string(),
            ));
        }

        if let Some(inner_model) = model.strip_prefix("google_gemini:") {
            if let Some(ref key) = self.gemini_api_key {
                let provider = gemini::GeminiProvider::new(key.clone(), None);
                return provider
                    .generate_stream(inner_model, messages, None, None, callback)
                    .await;
            }
            return Err(ProviderError::api_message(
                "Google Gemini API key not configured".to_string(),
            ));
        }

        // llama.cpp route (models prefixed with "llamacpp:", "llama.cpp:", or "llama:")
        if let Some(inner_model) = llamacpp_target(model) {
            return self
                .generate_stream_llamacpp(inner_model, messages, options, callback)
                .await;
        }

        // Custom OpenAI-compatible endpoint
        if let Some(inner_model) = remote_target(model) {
            let provider = self.remote_provider()?;
            let rendered = tools::render_history(messages, &[], tools::HistoryStyle::OpenAI);
            let step = provider
                .generate_stream_with_tools(inner_model, &rendered, &[], "Remote", callback)
                .await?;
            return Ok(step.text);
        }

        // OpenRouter route
        if model.contains('/') && self.openrouter_api_key.is_some() {
            self.generate_stream_openrouter(model, messages, callback)
                .await
        } else {
            self.generate_stream_ollama(model, messages, callback).await
        }
    }

    /// llama.cpp servers serve their single loaded model and ignore the
    /// `model` field of chat completions when it differs. Make model switches
    /// effective: if the requested model is not currently loaded, swap it in.
    /// When the server cannot load the model, surface a clear error instead of
    /// silently answering with the previously-loaded model.
    async fn ensure_llamacpp_model_loaded(&self, model: &str) -> Result<(), ProviderError> {
        let Some(client) = &self.llama_cpp_client else {
            return Ok(());
        };
        match client.list_models().await {
            Ok(loaded) => {
                if loaded.iter().any(|m| m.id == model) {
                    return Ok(());
                }
                client.load_model(model).await.map_err(|e| {
                    ProviderError::api_message(format!(
                        "llama.cpp model '{model}' is not loaded and could not be swapped in: {e}"
                    ))
                })
            }
            Err(_) => Ok(()), // let the chat request surface connectivity errors
        }
    }

    async fn generate_stream_ollama<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        mut callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        let url = format!("{}/api/chat", self.ollama_client.base_url());

        let payload = serde_json::json!({
            "model": model,
            "messages": messages.iter().map(to_ollama_value).collect::<Vec<_>>(),
            "stream": true
        });

        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        // Surface HTTP error statuses (5xx, 429) as retriable Api errors —
        // without this, an error body fails to parse as NDJSON and the failure
        // is silently swallowed as an empty success, defeating the retry layer.
        if !response.status().is_success() {
            let status = response.status();
            let msg = response.text().await.unwrap_or_default();
            return Err(ProviderError::api("Ollama", status, msg));
        }

        let mut stream = response.bytes_stream();
        let mut full_response = String::new();
        let mut lines = frames::FrameLines::default();
        let mut done = false;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
            lines.feed(&chunk, &mut |line| {
                if done {
                    return;
                }
                done = ingest_ollama_line(line, &mut full_response, None, &mut callback);
            });
        }
        if !done {
            lines.finish(&mut |line| {
                if done {
                    return;
                }
                done = ingest_ollama_line(line, &mut full_response, None, &mut callback);
            });
        }

        Ok(full_response)
    }

    /// Ollama `/api/chat` streaming with tool definitions.
    ///
    /// Ollama sends complete `message.tool_calls[]` arrays per NDJSON line
    /// (no delta fragments), so the last non-empty set wins.
    async fn generate_stream_ollama_with_tools<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        history: &[AgentTurn],
        tools: &[ToolDefinition],
        mut callback: F,
    ) -> Result<AgentStep, ProviderError>
    where
        F: FnMut(&str),
    {
        let url = format!("{}/api/chat", self.ollama_client.base_url());

        let mut payload = serde_json::json!({
            "model": model,
            "messages": tools::render_history(messages, history, tools::HistoryStyle::Ollama),
            "stream": true
        });
        if !tools.is_empty() {
            payload["tools"] = tools.iter().map(ToolDefinition::to_api_value).collect();
        }

        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let msg = response.text().await.unwrap_or_default();
            return Err(ProviderError::api("Ollama", status, msg));
        }

        let mut stream = response.bytes_stream();
        let mut text = String::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut lines = frames::FrameLines::default();
        let mut done = false;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
            lines.feed(&chunk, &mut |line| {
                if done {
                    return;
                }
                done = ingest_ollama_line(line, &mut text, Some(&mut calls), &mut callback);
            });
        }
        if !done {
            lines.finish(&mut |line| {
                if done {
                    return;
                }
                done = ingest_ollama_line(line, &mut text, Some(&mut calls), &mut callback);
            });
        }

        Ok(AgentStep {
            text,
            tool_calls: calls,
        })
    }

    /// Non-streaming OpenRouter request (called from `generate()`).
    async fn generate_nonstream_openrouter(
        &self,
        model: &str,
        messages: &[ChatMessage],
    ) -> Result<String, ProviderError> {
        let url = "https://openrouter.ai/api/v1/chat/completions";
        let api_key = self.openrouter_api_key.as_ref().unwrap();

        let payload = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true
        });

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("HTTP-Referer", "http://localhost")
            .header("X-Title", "Xencode")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let msg = response.text().await.unwrap_or_default();
            return Err(ProviderError::api("OpenRouter", status, msg));
        }

        #[derive(Deserialize)]
        struct OpenRouterNonStreamResponse {
            choices: Vec<OpenRouterNonStreamChoice>,
        }
        #[derive(Deserialize)]
        struct OpenRouterNonStreamChoice {
            message: OpenRouterMessage,
        }
        #[derive(Deserialize)]
        struct OpenRouterMessage {
            content: String,
        }

        let body: OpenRouterNonStreamResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        body.choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| ProviderError::Parse("OpenRouter: empty response".to_string()))
    }

    async fn generate_stream_openrouter<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        mut callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        let url = "https://openrouter.ai/api/v1/chat/completions";
        let api_key = self.openrouter_api_key.as_ref().unwrap();

        let payload = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true
        });

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("HTTP-Referer", "http://localhost")
            .header("X-Title", "Xencode")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let msg = response.text().await.unwrap_or_default();
            return Err(ProviderError::api("OpenRouter", status, msg));
        }

        let mut stream = response.bytes_stream();
        let mut full_response = String::new();

        let mut lines = crate::frames::FrameLines::default();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
            lines.feed(&chunk, &mut |line| {
                ingest_openrouter_line(line, &mut full_response, &mut callback)
            });
        }
        lines.finish(&mut |line| ingest_openrouter_line(line, &mut full_response, &mut callback));

        Ok(full_response)
    }

    /// Non-streaming llama.cpp completion request using OpenAI-compatible /v1/chat/completions.
    async fn generate_llamacpp(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: Option<&LlamaCppOptions>,
    ) -> Result<String, ProviderError> {
        let start = Instant::now();
        self.ensure_llamacpp_model_loaded(model).await?;
        let base_url = match &self.llama_cpp_client {
            Some(client) => client.base_url(),
            None => "http://localhost:8080",
        };

        let url = format!("{}/v1/chat/completions", base_url);

        let mut payload = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false
        });

        if let Some(opts) = options {
            merge_llamacpp_options(&mut payload, opts);
        }

        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("llama.cpp request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let msg = response.text().await.unwrap_or_default();
            return Err(ProviderError::api("llama.cpp", status, msg));
        }

        #[derive(Deserialize)]
        struct LlamaCppResponse {
            choices: Vec<LlamaCppChoice>,
            #[serde(default)]
            usage: Option<LlamaCppUsage>,
        }
        #[derive(Deserialize)]
        struct LlamaCppChoice {
            message: ChatMessage,
        }
        #[derive(Deserialize)]
        struct LlamaCppUsage {
            #[serde(default)]
            completion_tokens: u64,
        }

        let body: LlamaCppResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("llama.cpp parse error: {e}")))?;

        let tokens = body
            .usage
            .as_ref()
            .map(|u| u.completion_tokens)
            .unwrap_or(0);
        if tokens > 0 {
            self.record_llamacpp_timings(tokens, start.elapsed().as_secs_f64());
        }

        body.choices
            .first()
            .map(|c| c.message.text_content())
            .ok_or_else(|| ProviderError::Parse("llama.cpp: empty choices".to_string()))
    }

    /// Streaming llama.cpp completion request using SSE /v1/chat/completions.
    async fn generate_stream_llamacpp<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: Option<&LlamaCppOptions>,
        mut callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        let start = Instant::now();
        self.ensure_llamacpp_model_loaded(model).await?;
        let base_url = match &self.llama_cpp_client {
            Some(client) => client.base_url(),
            None => "http://localhost:8080",
        };

        let url = format!("{}/v1/chat/completions", base_url);

        let mut payload = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true
        });

        if let Some(opts) = options {
            merge_llamacpp_options(&mut payload, opts);
        }

        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("llama.cpp request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let msg = response.text().await.unwrap_or_default();
            return Err(ProviderError::api("llama.cpp", status, msg));
        }

        let mut stream = response.bytes_stream();
        let mut full_response = String::new();
        let mut completion_tokens: u64 = 0;
        let mut acc = tools::ToolCallAccumulator::default();
        let mut lines = frames::FrameLines::default();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| ProviderError::Network(format!("llama.cpp stream error: {e}")))?;
            lines.feed(&chunk, &mut |line| {
                compatible::ingest_line(
                    line,
                    &mut full_response,
                    &mut completion_tokens,
                    &mut acc,
                    &mut callback,
                )
            });
        }
        lines.finish(&mut |line| {
            compatible::ingest_line(
                line,
                &mut full_response,
                &mut completion_tokens,
                &mut acc,
                &mut callback,
            )
        });

        if completion_tokens > 0 {
            self.record_llamacpp_timings(completion_tokens, start.elapsed().as_secs_f64());
        }

        Ok(full_response)
    }

    /// Streaming llama.cpp request with tool definitions (agentic loop).
    /// `history` carries prior assistant/tool turns rendered OpenAI-style.
    async fn generate_stream_llamacpp_with_tools<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        history: &[AgentTurn],
        tools: &[ToolDefinition],
        options: Option<&LlamaCppOptions>,
        callback: F,
    ) -> Result<AgentStep, ProviderError>
    where
        F: FnMut(&str),
    {
        // Local-only behavior stays here: model-swap guard, sampling-opts
        // merge, and tok/s timings. HTTP+SSE is the shared adapter core.
        let start = Instant::now();
        self.ensure_llamacpp_model_loaded(model).await?;
        let base_url = match &self.llama_cpp_client {
            Some(client) => client.base_url(),
            None => "http://localhost:8080",
        };

        let url = format!("{}/v1/chat/completions", base_url);
        let rendered = tools::render_history(messages, history, tools::HistoryStyle::OpenAI);

        let mut payload = serde_json::json!({
            "model": model,
            "messages": rendered,
            "stream": true
        });
        if !tools.is_empty() {
            payload["tools"] = tools.iter().map(ToolDefinition::to_api_value).collect();
            payload["tool_choice"] = serde_json::Value::String("auto".to_string());
        }

        if let Some(opts) = options {
            merge_llamacpp_options(&mut payload, opts);
        }

        let outcome = compatible::post_sse_stream(
            &self.client,
            &url,
            None,
            &[],
            &payload,
            "llama.cpp",
            callback,
        )
        .await?;

        if outcome.completion_tokens > 0 {
            self.record_llamacpp_timings(outcome.completion_tokens, start.elapsed().as_secs_f64());
        }

        Ok(outcome.step)
    }

    /// Streaming OpenRouter request with tool definitions (agentic loop).
    /// Thin alias over the generic OpenAI-compatible adapter.
    async fn generate_stream_openrouter_with_tools<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        history: &[AgentTurn],
        tools: &[ToolDefinition],
        callback: F,
    ) -> Result<AgentStep, ProviderError>
    where
        F: FnMut(&str),
    {
        let provider = compatible::OpenAICompatibleProvider::new(
            "https://openrouter.ai/api/v1",
            self.openrouter_api_key.clone(),
        )
        .header("HTTP-Referer", "http://localhost")
        .header("X-Title", "Xencode");
        let rendered = tools::render_history(messages, history, tools::HistoryStyle::OpenAI);
        provider
            .generate_stream_with_tools(model, &rendered, tools, "OpenRouter", callback)
            .await
    }
}

/// Merge llama.cpp sampling options into an OpenAI-compatible chat payload.
fn merge_llamacpp_options(payload: &mut serde_json::Value, opts: &LlamaCppOptions) {
    if let Some(ref grammar) = opts.grammar {
        payload["grammar"] = serde_json::Value::String(grammar.clone());
    }
    if let Some(ref schema) = opts.json_schema {
        payload["json_schema"] = schema.clone();
    }
    if let Some(min_p) = opts.min_p {
        payload["min_p"] = serde_json::json!(min_p);
    }
    if let Some(top_k) = opts.top_k {
        payload["top_k"] = serde_json::json!(top_k);
    }
    if let Some(mirostat) = opts.mirostat {
        payload["mirostat"] = serde_json::json!(mirostat);
    }
    if let Some(temp) = opts.temperature {
        payload["temperature"] = serde_json::json!(temp);
    }
    if let Some(max_tokens) = opts.max_tokens {
        payload["max_tokens"] = serde_json::json!(max_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_chat_message() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "hello".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"hello"}"#);
    }

    #[test]
    fn text_content_round_trips_through_json() {
        let msg = ChatMessage::text("user", "hello");
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content, MessageContent::Text("hello".to_string()));
        assert_eq!(back.text_content(), "hello");
        assert!(!back.has_images());
        assert!(back.image_urls().is_empty());
    }

    #[test]
    fn parts_serialize_as_openai_content_blocks() {
        let msg = ChatMessage::user_with_images(
            "what is this?",
            vec!["data:image/png;base64,iVBORw0KGgo=".to_string()],
        );
        assert!(msg.has_images());
        assert_eq!(msg.image_urls(), vec!["data:image/png;base64,iVBORw0KGgo="]);
        assert_eq!(msg.text_content(), "what is this?");
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(
            json["content"],
            serde_json::json!([
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
            ])
        );
        // And back again — providers returning blocks deserialize too.
        let back: ChatMessage = serde_json::from_value(json).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn split_data_url_parts_mime_and_data() {
        assert_eq!(
            split_data_url("data:image/jpeg;base64,/9j/4AA="),
            Some(("image/jpeg", "/9j/4AA="))
        );
        assert_eq!(split_data_url("aGVsbG8="), None);
        assert_eq!(split_data_url("https://x/y.png"), None);
    }

    #[test]
    fn ollama_value_omits_images_key_for_text() {
        let value = to_ollama_value(&ChatMessage::text("user", "hi"));
        assert_eq!(value, serde_json::json!({"role": "user", "content": "hi"}));
    }

    #[test]
    fn ollama_value_strips_data_url_prefix() {
        let msg = ChatMessage::user_with_images(
            "see",
            vec![
                "data:image/png;base64,AAAA".to_string(),
                "rawbase64==".to_string(),
            ],
        );
        let value = to_ollama_value(&msg);
        assert_eq!(value["content"], "see");
        assert_eq!(value["images"], serde_json::json!(["AAAA", "rawbase64=="]));
    }

    #[test]
    fn llamacpp_target_matches_prefixes() {
        assert_eq!(llamacpp_target("llamacpp:model.gguf"), Some("model.gguf"));
        assert_eq!(llamacpp_target("llama.cpp:model"), Some("model"));
        assert_eq!(llamacpp_target("llama:runner"), Some("runner"));
        assert_eq!(llamacpp_target("qwen2.5:7b"), None);
        assert_eq!(llamacpp_target("llama3.1:8b"), None);
    }

    #[test]
    fn merge_options_populates_payload() {
        let opts = LlamaCppOptions {
            temperature: Some(0.7),
            top_k: Some(40),
            min_p: Some(0.05),
            mirostat: Some(2),
            max_tokens: Some(256),
            grammar: None,
            json_schema: None,
        };
        let mut payload = serde_json::json!({ "model": "m", "stream": true });
        merge_llamacpp_options(&mut payload, &opts);
        assert_eq!(payload["temperature"], 0.7);
        assert_eq!(payload["top_k"], 40);
        assert_eq!(payload["min_p"], 0.05);
        assert_eq!(payload["mirostat"], 2);
        assert_eq!(payload["max_tokens"], 256);
        assert!(payload.get("grammar").is_none());
    }

    #[test]
    fn llamacpp_target_rejects_slash_and_plain() {
        assert_eq!(llamacpp_target("openai/gpt-4o"), None);
        assert_eq!(llamacpp_target(""), None);
    }
}
