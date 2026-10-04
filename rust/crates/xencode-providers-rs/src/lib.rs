use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use xencode_models_rs::{LlamaCppClient, LlamaCppOptions, LlamaCppTimings, OllamaClient};

pub use xencode_models_rs::ModelShow;

pub mod anthropic;
pub mod capabilities;
pub mod compatible;
pub mod egress;
mod frames;
pub mod gemini;
pub mod listing;
pub mod playback;
pub mod qwen;
pub mod retry;
pub mod schema;
pub mod tools;
pub mod traffic;

use retry::RetryConfig;

pub use capabilities::{
    capabilities_for, effective_context_window, ollama_window_asked, ModelCapabilities,
};
pub use compatible::OpenAICompatibleProvider;
pub use egress::{chain_for, classify, provider_for, url_host, Egress, EgressPolicy, RoutingFacts};
pub use tools::{
    advise_tools, background_tools, command_tools, file_tools, plan_tools, repro_tools,
    search_tools, skill_tools, web_tools, AgentStep, AgentTurn, ToolCall, ToolDefinition,
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

/// NVIDIA NIM's OpenAI-compatible root. Fixed per L-11: a first-run user gets
/// real capacity without renting anything, and the route carries no
/// per-deployment URL to misconfigure.
pub const NVIDIA_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";

/// Extract the inner model identifier if `model` routes to NVIDIA NIM
/// (`nvidia:<vendor/model>`, e.g. `nvidia:mistralai/mistral-7b-instruct-v0.3`).
fn nvidia_target(model: &str) -> Option<&str> {
    model.strip_prefix("nvidia:")
}

/// True when `model` is served by a llama.cpp server, so a caller can decide
/// whether asking that server anything (its window, its tokenizer) is worth
/// a request at all. Same rule the provider uses to route, not a copy of it.
pub fn routes_to_llamacpp(model: &str) -> bool {
    matches!(llamacpp_target(model), Some(target) if !target.is_empty())
}

/// True when `model` is served by the local Ollama, so asking Ollama anything
/// about it — what it can do, how big a window its weights hold — is worth a
/// request at all.
///
/// Deliberately narrower than the manager's actual routing, which falls through
/// to Ollama for anything no other provider claims: a model id with a slash in
/// it goes to OpenRouter when a key is configured and to Ollama when it is not,
/// and this function cannot see the key. So a slash id answers `false` here and
/// simply never gets the extra asking, which costs a missing nicety on one odd
/// shape of id rather than a wrong assumption about a server that was never
/// involved.
pub fn routes_to_ollama(model: &str) -> bool {
    !routes_to_llamacpp(model)
        && remote_target(model).is_none()
        && nvidia_target(model).is_none()
        && !model.starts_with("anthropic:")
        && !model.starts_with("qwen:")
        && !model.starts_with("google_gemini:")
        && !model.contains('/')
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
    /// API root of the `nvidia:` endpoint; `Some` once an NVIDIA key is
    /// configured (the URL itself is fixed — see [`NVIDIA_BASE_URL`]).
    nvidia_base_url: Option<String>,
    nvidia_api_key: Option<String>,
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
    /// Where to write down what a request asked and what came back, when the
    /// caller asked for a recording (QA-1). `None` records nothing.
    traffic: Option<traffic::TrafficRecorder>,
    /// What every Ollama request this manager makes should say about the
    /// server's own behaviour — the window, how long the model stays loaded,
    /// and whether it may think.
    ollama_request: OllamaRequest,
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
            nvidia_base_url: None,
            nvidia_api_key: None,
            retry_config: RetryConfig::default(),
            egress: EgressPolicy::default(),
            client,
            request_timeout_secs: 0,
            llamacpp_timings: Mutex::new(None),
            traffic: None,
            ollama_request: OllamaRequest::default(),
        }
    }

    /// Say what every Ollama request should tell the server about itself: the
    /// window the conversation is budgeted for, how long the model should stay
    /// loaded, and whether a reasoning model may think first. The default —
    /// what every caller gets without this — sends none of the three, which
    /// leaves each of them to Ollama's own answer.
    pub fn with_ollama_request(mut self, request: OllamaRequest) -> Self {
        self.ollama_request = request;
        self
    }

    /// The Ollama server-behaviour fields this manager puts on a request, after
    /// whatever `with_ollama_request` was given.
    pub fn ollama_request(&self) -> &OllamaRequest {
        &self.ollama_request
    }

    /// Ask the Ollama server what `model` is: what it can do, and how big a
    /// window its own weights hold. `None` when the question got no useful
    /// answer — no server, a model that is not installed, a build old enough not
    /// to answer — which the caller treats as having learned nothing rather than
    /// as having learned that the model can do nothing.
    pub async fn ollama_show(&self, model: &str) -> Option<ModelShow> {
        if !routes_to_ollama(model) {
            return None;
        }
        self.ollama_client.show_model(model).await.ok()
    }

    /// Ask Ollama about `model`, then keep whatever survives the asking as what
    /// every later request for it will say, and hand back the notes about what
    /// had to be given up. [`ollama_request_for`] holds the rules; this is the
    /// one place that knows both the server to ask and the manager to remember
    /// the answer on, so a caller in the CLI or the TUI does not have to
    /// reproduce the route rules to get them applied.
    pub async fn prepare_ollama_request(
        &mut self,
        model: &str,
        asked: OllamaRequest,
    ) -> Vec<String> {
        if !routes_to_ollama(model) {
            return Vec::new();
        }
        let show = self.ollama_show(model).await;
        let (decided, notes) = ollama_request_for(show.as_ref(), asked);
        self.ollama_request = decided;
        notes
    }

    /// Keep the traffic of every OpenAI-compatible request this manager makes,
    /// so the session it describes can be replayed later
    /// ([`traffic::TrafficRecorder`], [`playback::Playback`]). `None` — which is
    /// what every caller passes unless it is recording — captures nothing.
    pub fn with_traffic(mut self, recorder: Option<traffic::TrafficRecorder>) -> Self {
        self.traffic = recorder;
        self
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

    /// Point the `nvidia:` prefix at NVIDIA NIM. The key is the whole
    /// configuration — the URL is fixed — so an `nvidia:` request without one
    /// reports that instead of dialling anonymously.
    pub fn with_nvidia(mut self, api_key: Option<String>) -> Self {
        let key = api_key.filter(|key| !key.trim().is_empty());
        if key.is_some() {
            self.nvidia_base_url = Some(NVIDIA_BASE_URL.to_string());
            self.nvidia_api_key = key;
        }
        self
    }

    /// Point the `nvidia:` prefix at a non-default OpenAI-compatible root
    /// (a corporate proxy in front of NIM, or a wiremock stand-in in tests).
    /// An empty `base_url` leaves the route as it was.
    pub fn with_nvidia_endpoint(mut self, base_url: &str, api_key: Option<String>) -> Self {
        let base_url = base_url.trim();
        if !base_url.is_empty() {
            self.nvidia_base_url = Some(base_url.to_string());
            self.nvidia_api_key = api_key.filter(|key| !key.trim().is_empty());
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
    /// - `nvidia:` → NVIDIA NIM (fixed OpenAI-compatible endpoint)
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

    /// Record token usage and timing from what a llama.cpp completion reported.
    ///
    /// Every count here is the server's own — see
    /// [`LlamaCppTimings::from_usage`] for what b10809 was measured to say.
    /// Nothing is recorded for a request the server never processed, which is
    /// the case where it reported neither a prompt nor a reply.
    fn maybe_record_llamacpp_timings(&self, counts: compatible::UsageCounts, elapsed: f64) {
        if counts.prompt_tokens == 0 && counts.completion_tokens == 0 {
            return;
        }
        *self.llamacpp_timings.lock().unwrap() = Some(LlamaCppTimings::from_usage(
            counts.completion_tokens,
            counts.prompt_tokens,
            counts.cached_tokens,
            elapsed,
        ));
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

    /// The NVIDIA NIM endpoint, or a message saying how to configure it.
    /// Unlike `remote:`, the URL is fixed — only the key is user-supplied —
    /// so a missing key is the only unconfigured state.
    fn nvidia_provider(&self) -> Result<compatible::OpenAICompatibleProvider, ProviderError> {
        match self.nvidia_base_url.as_deref() {
            Some(base_url) => Ok(compatible::OpenAICompatibleProvider::new(
                base_url,
                self.nvidia_api_key.clone(),
            )
            .with_recorder(self.traffic.clone())),
            None => Err(ProviderError::api_message(
                "No NVIDIA API key configured — run `xencode config set nvidia_api_key <key>` \
                 (or export NVIDIA_NIM_API_KEY) and use `nvidia:<vendor/model>`"
                    .to_string(),
            )),
        }
    }
    /// The configured OpenAI-compatible endpoint, or a message saying how to
    /// configure one. Never falls back to a default host: a mistyped or absent
    /// URL must fail loudly rather than POST the prompt somewhere else.
    fn remote_provider(&self) -> Result<compatible::OpenAICompatibleProvider, ProviderError> {
        match self.remote_base_url.as_deref() {
            Some(base_url) => Ok(compatible::OpenAICompatibleProvider::new(
                base_url,
                self.remote_api_key.clone(),
            )
            .with_recorder(self.traffic.clone())),
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

        // NVIDIA NIM route (models prefixed with "nvidia:")
        if let Some(inner_model) = nvidia_target(model) {
            let provider = self.nvidia_provider()?;
            let rendered = tools::render_history(messages, &[], tools::HistoryStyle::OpenAI);
            return provider.generate(inner_model, &rendered, "Nvidia").await;
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

        let mut payload = serde_json::json!({
            "model": model,
            "messages": messages.iter().map(to_ollama_value).collect::<Vec<_>>(),
            "stream": false
        });
        merge_ollama_request(&mut payload, options, &self.ollama_request);

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
    /// - `nvidia:` → NVIDIA NIM
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

        // NVIDIA NIM route, with full tool-calling support
        if let Some(inner_model) = nvidia_target(model) {
            let provider = self.nvidia_provider()?;
            let rendered = tools::render_history(messages, history, tools::HistoryStyle::OpenAI);
            return provider
                .generate_stream_with_tools(inner_model, &rendered, tools, "Nvidia", callback)
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
        self.generate_stream_ollama_with_tools(model, messages, history, tools, options, callback)
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

        // NVIDIA NIM route
        if let Some(inner_model) = nvidia_target(model) {
            let provider = self.nvidia_provider()?;
            let rendered = tools::render_history(messages, &[], tools::HistoryStyle::OpenAI);
            let step = provider
                .generate_stream_with_tools(inner_model, &rendered, &[], "Nvidia", callback)
                .await?;
            return Ok(step.text);
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
            self.generate_stream_ollama(model, messages, options, callback)
                .await
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
        options: Option<&LlamaCppOptions>,
        mut callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        let url = format!("{}/api/chat", self.ollama_client.base_url());

        let mut payload = serde_json::json!({
            "model": model,
            "messages": messages.iter().map(to_ollama_value).collect::<Vec<_>>(),
            "stream": true
        });
        merge_ollama_request(&mut payload, options, &self.ollama_request);

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
        options: Option<&LlamaCppOptions>,
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
        merge_ollama_request(&mut payload, options, &self.ollama_request);

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

        let status = response.status().as_u16();
        let started = Instant::now();
        let mut stream = response.bytes_stream();
        let mut text = String::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut lines = frames::FrameLines::default();
        let mut done = false;
        // Only read into when a recording is being kept; see `traffic`.
        let mut raw: Vec<u8> = Vec::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
            if self.traffic.is_some() {
                raw.extend_from_slice(&chunk);
            }
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

        if let Some(recorder) = self.traffic.as_ref() {
            if let Ok(response_body) = String::from_utf8(raw) {
                recorder.record(traffic::TrafficPair {
                    method: "POST".to_string(),
                    path: traffic::url_path(&url),
                    request_body: payload.to_string(),
                    status,
                    content_type: "application/x-ndjson".to_string(),
                    response_body,
                    ts_unix_ms: traffic::now_unix_ms(),
                    duration_ms: started.elapsed().as_millis() as u64,
                });
            }
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
            #[serde(default)]
            prompt_tokens: u64,
            #[serde(default)]
            prompt_tokens_details: Option<LlamaCppPromptDetails>,
        }
        #[derive(Deserialize)]
        struct LlamaCppPromptDetails {
            #[serde(default)]
            cached_tokens: u64,
        }

        let body: LlamaCppResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("llama.cpp parse error: {e}")))?;

        let counts = body
            .usage
            .as_ref()
            .map(|u| compatible::UsageCounts {
                completion_tokens: u.completion_tokens,
                prompt_tokens: u.prompt_tokens,
                cached_tokens: u
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0),
            })
            .unwrap_or_default();
        self.maybe_record_llamacpp_timings(counts, start.elapsed().as_secs_f64());

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
        compatible::ask_for_usage(&mut payload);

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
        let mut usage = compatible::UsageCounts::default();
        let mut acc = tools::ToolCallAccumulator::default();
        let mut lines = frames::FrameLines::default();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| ProviderError::Network(format!("llama.cpp stream error: {e}")))?;
            lines.feed(&chunk, &mut |line| {
                compatible::ingest_line(
                    line,
                    &mut full_response,
                    &mut usage,
                    &mut acc,
                    &mut callback,
                )
            });
        }
        lines.finish(&mut |line| {
            compatible::ingest_line(
                line,
                &mut full_response,
                &mut usage,
                &mut acc,
                &mut callback,
            )
        });

        self.maybe_record_llamacpp_timings(usage, start.elapsed().as_secs_f64());

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
        compatible::ask_for_usage(&mut payload);
        if !tools.is_empty() {
            payload["tools"] = tools.iter().map(ToolDefinition::to_api_value).collect();
            payload["tool_choice"] = serde_json::Value::String("auto".to_string());
        }

        if let Some(opts) = options {
            merge_llamacpp_options(&mut payload, opts);
        }

        let outcome = compatible::post_sse_stream(
            &self.client,
            &compatible::SseRequest {
                url: &url,
                api_key: None,
                extra_headers: &[],
                payload: &payload,
                label: "llama.cpp",
                recorder: self.traffic.as_ref(),
            },
            callback,
        )
        .await?;

        self.maybe_record_llamacpp_timings(outcome.usage, start.elapsed().as_secs_f64());

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
///
/// A schema goes out as `response_format`, the shape the chat endpoint documents,
/// rather than as a bare `json_schema` — which is the name the plain-text
/// `/completion` endpoint uses, and the only shape a schema sent to a chat request
/// should not take. What the local server was measured doing with an answer asked
/// for this way is to follow it: `{"answer": "yes", "confidence": 1}` came back to
/// a request whose text said, in as many words, to answer in a sentence and avoid
/// braces.
///
/// One measured interaction is worth knowing before a caller reaches for this: a
/// schema and a `tools` list in the same chat request do not combine. The server
/// honours the schema and stops offering tool calls at all — on the local build
/// here, five requests carrying both produced an answer shaped like the schema and
/// no tool call whatsoever. Both are passed through exactly as asked, because
/// choosing silently between them would be worse than the caller finding out.
fn merge_llamacpp_options(payload: &mut serde_json::Value, opts: &LlamaCppOptions) {
    if let Some(ref grammar) = opts.grammar {
        payload["grammar"] = serde_json::Value::String(grammar.clone());
    }
    if let Some(ref schema) = opts.json_schema {
        let schema = schema::flatten(schema);
        payload["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                // The name is how the endpoint asks which of several schemas this
                // is; one request, one shape, so one name.
                "name": "answer",
                "schema": schema,
            }
        });
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
    if let Some(seed) = opts.seed {
        payload["seed"] = serde_json::json!(seed);
    }
    if let Some(max_tokens) = opts.max_tokens {
        payload["max_tokens"] = serde_json::json!(max_tokens);
    }
}

/// What an Ollama request adds on top of the sampling knobs it shares with
/// llama.cpp: how big a window to be served, how long the model stays loaded
/// afterwards, and whether a reasoning model may think before it answers.
///
/// These are kept apart from [`LlamaCppOptions`] because they are not properties
/// of the answer being sampled. `num_ctx` decides how much conversation fits,
/// `keep_alive` decides how long the weights hold their memory for the next
/// request, and `think` is honoured in the body of an Ollama request — which is
/// the opposite of the llama.cpp equivalent, measured to be accepted and ignored
/// there and reachable only as a launch flag.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OllamaRequest {
    /// `options.num_ctx` — the context window this conversation is budgeted for.
    pub num_ctx: Option<u32>,
    /// `keep_alive` — a duration such as `10m`, or `0` to release the model as
    /// soon as this answer is done.
    pub keep_alive: Option<String>,
    /// `think` — `Some(false)` tells a reasoning model not to reason, `Some(true)`
    /// asks one that can to reason, and `None` leaves the field out altogether and
    /// lets the model's own default decide.
    pub think: Option<bool>,
}

impl OllamaRequest {
    /// Read the two settings a person can write into the config file.
    ///
    /// `reasoning` takes the same words `llama_cpp_reasoning` does — `off`, `on`,
    /// `auto` — with one difference that comes from the server rather than from
    /// taste: Ollama has no partial thinking to ask for. A token budget like
    /// `"256"` is therefore an error here instead of a launch flag, because
    /// sending it would quietly mean something else. `keep_alive` is passed
    /// through as the text Ollama itself parses (`"10m"`, `"30s"`, `"0"`); an
    /// unreadable duration is Ollama's complaint to make, on the request.
    pub fn from_settings(
        reasoning: Option<&str>,
        keep_alive: Option<&str>,
    ) -> Result<Self, String> {
        Ok(Self {
            num_ctx: None,
            keep_alive: keep_alive
                .map(str::trim)
                .filter(|raw| !raw.is_empty())
                .map(|raw| raw.to_string()),
            think: think_from_setting(reasoning)?,
        })
    }
}

/// Turn the words in `ollama_reasoning` into the `think` field, or explain why
/// they cannot mean anything. `None` from this function means "send no `think`
/// at all", which measured behaviour is *not* the same as `Some(false)`: an
/// omitted field leaves the model's own default in place, and Qwen3 defaults to
/// thinking.
fn think_from_setting(raw: Option<&str>) -> Result<Option<bool>, String> {
    let Some(raw) = raw.map(str::trim) else {
        return Ok(None);
    };
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    if raw.eq_ignore_ascii_case("off") {
        return Ok(Some(false));
    }
    if raw.eq_ignore_ascii_case("on") {
        return Ok(Some(true));
    }
    Err(format!(
        "ollama_reasoning must be \"auto\", \"on\" or \"off\", not {raw:?}. Ollama has no way to \
         ask for a thinking budget; use llama_cpp_reasoning with a llama.cpp server for that"
    ))
}

/// Put the sampling knobs and the server-behaviour knobs onto an Ollama
/// `/api/chat` body, under the names that endpoint uses.
///
/// The names differ from the OpenAI-compatible route in three ways that matter:
/// a JSON schema is the whole value of `format` rather than a `response_format`
/// object, a generation cap is `options.num_predict` rather than `max_tokens`,
/// and everything else about how the tokens are chosen lives in one `options`
/// object. A grammar has no counterpart here — Ollama takes a schema and derives
/// its own constrained decoding from it — so `grammar` is deliberately not sent
/// rather than being smuggled into `format`, which would be a different promise.
///
/// `mirostat` is not sent either, and that is a measurement rather than an
/// omission: Ollama 0.34.4 answers a request carrying it with
/// `invalid option provided option=mirostat` and runs the sampler without it, so
/// promising Mirostat on this route would be promising something the server
/// throws away. `temperature`, `top_k`, `min_p`, `seed`, `num_predict` and
/// `num_ctx` were each checked to arrive, by reading the sampler parameters the
/// inference engine prints for the request.
///
/// One interaction to know before a caller puts a schema beside a tools list: the
/// schema wins and the tool call vanishes. Measured on a 1.7B model asked to read
/// a file — with `tools` alone it answered with a `read_file` call, and with
/// `format` added to the same request it answered `{"answer": "I will read the
/// file notes.txt for you."}` and called nothing. Both still go out as asked, as
/// on the llama.cpp route, because a caller choosing silently between them would
/// be worse than one finding this out.
fn merge_ollama_request(
    payload: &mut serde_json::Value,
    opts: Option<&LlamaCppOptions>,
    request: &OllamaRequest,
) {
    let mut options = serde_json::Map::new();
    if let Some(opts) = opts {
        if let Some(ref schema) = opts.json_schema {
            payload["format"] = schema::flatten(schema);
        }
        if let Some(temp) = opts.temperature {
            options.insert("temperature".to_string(), serde_json::json!(temp));
        }
        if let Some(top_k) = opts.top_k {
            options.insert("top_k".to_string(), serde_json::json!(top_k));
        }
        if let Some(min_p) = opts.min_p {
            options.insert("min_p".to_string(), serde_json::json!(min_p));
        }
        if let Some(seed) = opts.seed {
            options.insert("seed".to_string(), serde_json::json!(seed));
        }
        if let Some(max_tokens) = opts.max_tokens {
            options.insert("num_predict".to_string(), serde_json::json!(max_tokens));
        }
    }
    if let Some(num_ctx) = request.num_ctx {
        options.insert("num_ctx".to_string(), serde_json::json!(num_ctx));
    }
    if !options.is_empty() {
        payload["options"] = serde_json::Value::Object(options);
    }
    if let Some(keep_alive) = request.keep_alive.as_deref() {
        payload["keep_alive"] = serde_json::Value::String(keep_alive.to_string());
    }
    if let Some(think) = request.think {
        payload["think"] = serde_json::json!(think);
    }
}

/// Decide what to actually ask Ollama for, once the model has been asked what it
/// can do. `show` is `None` when that question failed — no server, a model that
/// is not installed, a build old enough not to answer it. Having learned nothing
/// is not the same as having learned that the model can do nothing, so the window
/// keeps the number it was given; but the two fields differ in what a wrong guess
/// costs, and only one of them can fail a request outright.
///
/// Both rules below were measured on 0.34.4. A window bigger than the weights
/// were trained for is reduced to it by the server, so a conversation budgeted
/// for the bigger number is refused mid-turn with `exceed_context_size_error`
/// rather than truncated — clamping here is what keeps the two numbers the same.
/// And asking a model that cannot think to think is answered with HTTP 400 and
/// `"<model>" does not support thinking`, which is why `think: true` is sent only
/// to a model that has said it can, and not to one whose answer never arrived.
/// `think: false` needs no such permission: it was accepted by a model with no
/// thinking at all.
///
/// The notes are for whoever reads the transcript: a run that asked for one
/// window and got a smaller one, or asked for reasoning that the model does not
/// have, should say so instead of looking like it did what was asked.
pub fn ollama_request_for(
    show: Option<&xencode_models_rs::ModelShow>,
    asked: OllamaRequest,
) -> (OllamaRequest, Vec<String>) {
    let mut notes = Vec::new();
    let mut decided = OllamaRequest {
        keep_alive: asked.keep_alive.clone(),
        ..Default::default()
    };

    if let Some(num_ctx) = asked.num_ctx {
        match show.and_then(|s| s.trained_context_tokens) {
            Some(ceiling) if ceiling < num_ctx => {
                notes.push(format!(
                    "asked Ollama for a {num_ctx}-token window; this model holds {ceiling}, so that is what it was given"
                ));
                decided.num_ctx = Some(ceiling);
            }
            Some(_) => decided.num_ctx = Some(num_ctx),
            None => {
                // Nothing was learned about the ceiling, so nothing is clamped:
                // the profile's number is what the context is being filled for,
                // and asking for less than that is the one way to be refused.
                decided.num_ctx = Some(num_ctx);
            }
        }
    }

    decided.think = match asked.think {
        Some(true) => {
            let can = show.is_some_and(|s| s.can_think());
            if !can {
                notes.push(
                    "this model was asked to reason first but does not say it can, so nothing was asked of it"
                        .to_string(),
                );
            }
            can.then_some(true)
        }
        other => other,
    };

    (decided, notes)
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
    fn routing_to_llama_cpp_is_what_the_server_probe_asks() {
        assert!(routes_to_llamacpp("llamacpp:model.gguf"));
        assert!(routes_to_llamacpp("llama:dolphin"));
        // An Ollama tag that only looks local, and a bare prefix with no model
        // behind it, must not send anyone to the llama.cpp server.
        assert!(!routes_to_llamacpp("llama3.1:8b"));
        assert!(!routes_to_llamacpp("llama:"));
        assert!(!routes_to_llamacpp("qwen2.5:7b"));
    }

    #[test]
    fn merge_options_populates_payload() {
        let opts = LlamaCppOptions {
            temperature: Some(0.7),
            top_k: Some(40),
            min_p: Some(0.05),
            mirostat: Some(2),
            max_tokens: Some(256),
            seed: Some(42),
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
        assert_eq!(payload["seed"], 42);
        assert!(payload.get("grammar").is_none());
    }

    /// The status quo the seed work replaces: with no seed asked for, nothing
    /// goes over the wire, and llama.cpp draws one per request (`--seed` defaults
    /// to -1 = random). A test that only checked the happy path would let a
    /// stray `"seed": null` through, which the server reads as "random" while
    /// the code looks like it pinned something.
    #[test]
    fn merge_options_omits_seed_and_temperature_entirely_when_unset() {
        let opts = LlamaCppOptions {
            max_tokens: Some(64),
            ..Default::default()
        };
        let mut payload = serde_json::json!({ "model": "m", "stream": true });
        merge_llamacpp_options(&mut payload, &opts);
        assert_eq!(payload["max_tokens"], 64);
        assert!(payload.get("seed").is_none());
        assert!(payload.get("temperature").is_none());
    }

    /// Zero is a value, not an absence: a seed of 0 must still be sent, or the
    /// one run someone pinned to that seed would sample freely.
    #[test]
    fn merge_options_sends_seed_zero() {
        let opts = LlamaCppOptions {
            seed: Some(0),
            ..Default::default()
        };
        let mut payload = serde_json::json!({});
        merge_llamacpp_options(&mut payload, &opts);
        assert_eq!(payload["seed"], 0);
    }

    #[test]
    fn llamacpp_target_rejects_slash_and_plain() {
        assert_eq!(llamacpp_target("openai/gpt-4o"), None);
        assert_eq!(llamacpp_target(""), None);
    }

    fn a_show(trained: Option<u32>, capabilities: &[&str]) -> ModelShow {
        ModelShow {
            trained_context_tokens: trained,
            capabilities: capabilities.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn only_an_id_with_no_provider_prefix_is_asked_of_ollama() {
        assert!(routes_to_ollama("qwen3:4b"));
        assert!(routes_to_ollama("llama3.2"));
        assert!(!routes_to_ollama("llama:qwen3"));
        assert!(!routes_to_ollama("remote:qwen3"));
        assert!(!routes_to_ollama("qwen:qwen-plus"));
        assert!(!routes_to_ollama("google_gemini:gemini-2.5-flash"));
        assert!(!routes_to_ollama("anthropic:claude-sonnet-4-5"));
        // A slashed id may end up at Ollama when no OpenRouter key is set, but
        // this function cannot see the key, so it says no and nothing is asked.
        assert!(!routes_to_ollama("openai/gpt-4o"));
    }

    #[test]
    fn the_reasoning_setting_reads_three_words_and_refuses_a_budget() {
        assert_eq!(think_from_setting(Some("off")).unwrap(), Some(false));
        assert_eq!(think_from_setting(Some("ON")).unwrap(), Some(true));
        assert_eq!(think_from_setting(Some(" auto ")).unwrap(), None);
        assert_eq!(think_from_setting(Some("")).unwrap(), None);
        assert_eq!(think_from_setting(None).unwrap(), None);
        let err = think_from_setting(Some("256")).unwrap_err();
        assert!(
            err.contains("ollama_reasoning") && err.contains("llama_cpp_reasoning"),
            "{err}"
        );
    }

    #[test]
    fn from_settings_passes_a_keep_alive_through_as_the_text_the_server_parses() {
        let request = OllamaRequest::from_settings(Some("off"), Some(" 10m ")).unwrap();
        assert_eq!(request.think, Some(false));
        assert_eq!(request.keep_alive.as_deref(), Some("10m"));
        assert_eq!(request.num_ctx, None);
        // An empty setting is an unset one: sending `keep_alive: ""` would be
        // asking Ollama to interpret nothing as a duration.
        let blank = OllamaRequest::from_settings(None, Some("   ")).unwrap();
        assert_eq!(blank.keep_alive, None);
        assert_eq!(blank.think, None);
    }

    #[test]
    fn merge_puts_every_value_under_the_name_ollama_reads() {
        let opts = LlamaCppOptions {
            temperature: Some(0.2),
            top_k: Some(40),
            min_p: Some(0.05),
            seed: Some(7),
            max_tokens: Some(512),
            json_schema: Some(serde_json::json!({"type": "object"})),
            grammar: Some("root ::= \"x\"".to_string()),
            mirostat: Some(2),
        };
        let request = OllamaRequest {
            num_ctx: Some(8192),
            keep_alive: Some("0".to_string()),
            think: Some(false),
        };
        let mut payload = serde_json::json!({ "model": "qwen3:4b" });
        merge_ollama_request(&mut payload, Some(&opts), &request);

        let options = payload["options"].as_object().unwrap();
        assert_eq!(options["temperature"], 0.2);
        assert_eq!(options["top_k"], 40);
        assert_eq!(options["min_p"], 0.05);
        assert_eq!(options["seed"], 7);
        // A generation cap is `num_predict` here, not `max_tokens`.
        assert_eq!(options["num_predict"], 512);
        assert_eq!(options["num_ctx"], 8192);
        assert_eq!(payload["keep_alive"], "0");
        assert_eq!(payload["think"], false);
        assert_eq!(payload["format"], serde_json::json!({"type": "object"}));
        // And the two things this server cannot be given: a grammar of our own
        // (it derives its constrained decoding from the schema) and Mirostat,
        // which 0.34.4 answers by dropping the option and warning about it.
        assert!(
            payload.get("grammar").is_none() && payload.get("mirostat").is_none(),
            "{payload}"
        );
        assert!(!options.contains_key("mirostat"));
        assert!(!options.contains_key("max_tokens"));
    }

    #[test]
    fn merge_adds_nothing_that_nobody_asked_for() {
        // The whole point of the empty case: without this item every Ollama
        // request carried no `options`, and adding one that says nothing would
        // change what the server does with a model's own defaults.
        let mut payload = serde_json::json!({ "model": "qwen3:4b" });
        merge_ollama_request(&mut payload, None, &OllamaRequest::default());
        assert_eq!(payload, serde_json::json!({ "model": "qwen3:4b" }));

        let mut payload = serde_json::json!({ "model": "qwen3:4b" });
        merge_ollama_request(
            &mut payload,
            Some(&LlamaCppOptions::default()),
            &Default::default(),
        );
        assert!(payload.get("options").is_none(), "{payload}");
        assert!(payload.get("think").is_none());
    }

    #[test]
    fn a_window_bigger_than_the_weights_is_brought_down_and_said_out_loud() {
        let (decided, notes) = ollama_request_for(
            Some(&a_show(Some(40_960), &["thinking", "tools"])),
            OllamaRequest {
                num_ctx: Some(131_072),
                ..Default::default()
            },
        );
        assert_eq!(decided.num_ctx, Some(40_960));
        assert_eq!(notes.len(), 1);
        assert!(
            notes[0].contains("131072") && notes[0].contains("40960"),
            "{notes:?}"
        );
    }

    #[test]
    fn a_window_the_model_can_hold_is_sent_as_asked() {
        let (decided, notes) = ollama_request_for(
            Some(&a_show(Some(40_960), &["thinking"])),
            OllamaRequest {
                num_ctx: Some(8192),
                ..Default::default()
            },
        );
        assert_eq!(decided.num_ctx, Some(8192));
        assert!(notes.is_empty());
    }

    #[test]
    fn a_model_that_never_said_its_window_leaves_the_number_alone() {
        // Not the same as a model that said it holds nothing: guessing a smaller
        // window here would trim context nobody asked to trim.
        let (decided, notes) = ollama_request_for(
            Some(&a_show(None, &[])),
            OllamaRequest {
                num_ctx: Some(8192),
                ..Default::default()
            },
        );
        assert_eq!(decided.num_ctx, Some(8192));
        assert!(notes.is_empty());
        let (no_probe, _) = ollama_request_for(
            None,
            OllamaRequest {
                num_ctx: Some(8192),
                ..Default::default()
            },
        );
        assert_eq!(no_probe.num_ctx, Some(8192));
    }

    #[test]
    fn asking_a_model_that_cannot_think_to_think_would_fail_the_whole_request() {
        // Measured: `think: true` on a model without the capability is answered
        // HTTP 400 `"<model>" does not support thinking`, so it is dropped here
        // rather than sent and lost. Dropping it is also what a failed question
        // does — nothing was learned, so nothing is promised.
        let (decided, notes) = ollama_request_for(
            Some(&a_show(Some(32_768), &["tools", "completion"])),
            OllamaRequest {
                think: Some(true),
                ..Default::default()
            },
        );
        assert_eq!(decided.think, None);
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("does not say it can"), "{notes:?}");

        let (silent, notes) = ollama_request_for(
            None,
            OllamaRequest {
                think: Some(true),
                ..Default::default()
            },
        );
        assert_eq!(silent.think, None);
        assert_eq!(notes.len(), 1);
    }

    #[test]
    fn turning_thinking_off_needs_no_permission_from_the_model() {
        // `think: false` was accepted by a model whose capability list has no
        // thinking in it at all, so it goes through whatever was learned.
        for show in [
            Some(a_show(Some(32_768), &["tools", "completion"])),
            Some(a_show(None, &[])),
            None,
        ] {
            let (decided, notes) = ollama_request_for(
                show.as_ref(),
                OllamaRequest {
                    think: Some(false),
                    keep_alive: Some("5m".to_string()),
                    ..Default::default()
                },
            );
            assert_eq!(decided.think, Some(false));
            assert_eq!(decided.keep_alive.as_deref(), Some("5m"));
            assert!(notes.is_empty());
        }
    }

    #[test]
    fn the_asked_window_is_what_this_machine_can_serve_not_what_the_model_advertises() {
        assert_eq!(ollama_window_asked(Some(256_000), 8_192), 8_192);
        assert_eq!(ollama_window_asked(None, 8_192), 8_192);
        assert_eq!(ollama_window_asked(Some(4_096), 8_192), 4_096);
    }
}
