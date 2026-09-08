use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use xencode_models_rs::{LlamaCppClient, LlamaCppOptions, LlamaCppTimings, OllamaClient};

pub mod anthropic;
pub mod gemini;
pub mod qwen;
pub mod retry;

use retry::RetryConfig;

#[derive(Debug)]
pub enum ProviderError {
    Network(String),
    Api(String),
    Parse(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::Network(msg) => write!(f, "network error: {msg}"),
            ProviderError::Api(msg) => write!(f, "API error: {msg}"),
            ProviderError::Parse(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for ProviderError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: ChatMessage,
    done: bool,
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

/// ProviderManager abstracts over local and cloud models.
///
/// Supports Ollama (local), llama.cpp (local), OpenRouter (cloud), Qwen (cloud), Gemini (cloud),
/// and Anthropic (cloud).
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
    retry_config: RetryConfig,
    client: reqwest::Client,
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
            retry_config: RetryConfig::default(),
            client,
            llamacpp_timings: Mutex::new(None),
        }
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

        retry::retry_async(&self.retry_config, || async {
            self.generate_inner(&model_owned, &messages_owned, options)
                .await
        })
        .await
    }

    /// Record token-usage timing observed from a llama.cpp completion.
    fn record_llamacpp_timings(&self, tokens: u64, elapsed: f64) {
        *self.llamacpp_timings.lock().unwrap() = Some(LlamaCppTimings::from_elapsed(tokens, elapsed));
    }

    /// Most recent llama.cpp generation timing (tokens generated + tok/s).
    pub fn last_llamacpp_timings(&self) -> Option<LlamaCppTimings> {
        self.llamacpp_timings.lock().unwrap().clone()
    }

    /// Inner generate without retry wrapping (used by retry logic).
    async fn generate_inner(
        &self,
        model: &str,
        messages: &[ChatMessage],
        options: Option<&LlamaCppOptions>,
    ) -> Result<String, ProviderError> {
        // Route based on model prefix
        if let Some(inner_model) = model.strip_prefix("anthropic:") {
            if let Some(ref key) = self.anthropic_api_key {
                let provider = anthropic::AnthropicProvider::new(key.clone(), None, None);
                return provider.generate(inner_model, messages, None).await;
            }
            return Err(ProviderError::Api(
                "Anthropic API key not configured".to_string(),
            ));
        }

        if let Some(inner_model) = model.strip_prefix("qwen:") {
            if let Some(ref key) = self.qwen_api_key {
                let provider = qwen::QwenProvider::new(key.clone(), None);
                return provider.generate(inner_model, messages).await;
            }
            return Err(ProviderError::Api(
                "Qwen API key not configured".to_string(),
            ));
        }

        if let Some(inner_model) = model.strip_prefix("google_gemini:") {
            if let Some(ref key) = self.gemini_api_key {
                let provider = gemini::GeminiProvider::new(key.clone(), None);
                return provider.generate(inner_model, messages, None, None).await;
            }
            return Err(ProviderError::Api(
                "Google Gemini API key not configured".to_string(),
            ));
        }

        // llama.cpp route (models prefixed with "llamacpp:", "llama.cpp:", or "llama:")
        if let Some(inner_model) = llamacpp_target(model) {
            return self.generate_llamacpp(inner_model, messages, options).await;
        }

        // OpenRouter route (models with a slash, e.g. "openai/gpt-4")
        if model.contains('/') && self.openrouter_api_key.is_some() {
            return self.generate_nonstream_openrouter(model, messages).await;
        }

        // Default: local Ollama
        let url = format!("{}/api/chat", self.ollama_client.base_url());

        let payload = serde_json::json!({
            "model": model,
            "messages": messages,
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

        Ok(body.message.content)
    }

    /// Generate a response and stream it token-by-token.
    ///
    /// Routes to the appropriate provider based on the model prefix:
    /// - `anthropic:` → Anthropic Claude API
    /// - `qwen:` → Qwen cloud API
    /// - `google_gemini:` → Google Gemini API
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
                self.generate_stream_inner(&model_owned, &messages_owned, options, |token| {
                    emitted_ref.store(true, Ordering::SeqCst);
                    let mut guard = cb_ref.lock().unwrap();
                    guard(token);
                })
                .await
            },
        )
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
        // Route based on model prefix
        if let Some(inner_model) = model.strip_prefix("anthropic:") {
            if let Some(ref key) = self.anthropic_api_key {
                let provider = anthropic::AnthropicProvider::new(key.clone(), None, None);
                return provider
                    .generate_stream(inner_model, messages, None, callback)
                    .await;
            }
            return Err(ProviderError::Api(
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
            return Err(ProviderError::Api(
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
            return Err(ProviderError::Api(
                "Google Gemini API key not configured".to_string(),
            ));
        }

        // llama.cpp route (models prefixed with "llamacpp:", "llama.cpp:", or "llama:")
        if let Some(inner_model) = llamacpp_target(model) {
            return self
                .generate_stream_llamacpp(inner_model, messages, options, callback)
                .await;
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
                    ProviderError::Api(format!(
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
            "messages": messages,
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
            return Err(ProviderError::Api(format!("Ollama {} - {}", status, msg)));
        }

        let mut stream = response.bytes_stream();
        let mut full_response = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
            if let Ok(text) = std::str::from_utf8(&chunk) {
                for line in text.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    if let Ok(parsed) = serde_json::from_str::<OllamaResponse>(line) {
                        callback(&parsed.message.content);
                        full_response.push_str(&parsed.message.content);
                        if parsed.done {
                            return Ok(full_response);
                        }
                    }
                }
            }
        }

        Ok(full_response)
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
            "stream": false
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
            return Err(ProviderError::Api(format!(
                "OpenRouter {} - {}",
                status, msg
            )));
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
            return Err(ProviderError::Api(format!(
                "OpenRouter {} - {}",
                status, msg
            )));
        }

        let mut stream = response.bytes_stream();
        let mut full_response = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
            if let Ok(text) = std::str::from_utf8(&chunk) {
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() || line == "data: [DONE]" {
                        continue;
                    }
                    if let Some(data) = line.strip_prefix("data: ") {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                            if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                                if let Some(first) = choices.first() {
                                    if let Some(delta) = first.get("delta") {
                                        if let Some(content) =
                                            delta.get("content").and_then(|c| c.as_str())
                                        {
                                            callback(content);
                                            full_response.push_str(content);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

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
            return Err(ProviderError::Api(format!("llama.cpp {} - {}", status, msg)));
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

        let tokens = body.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0);
        if tokens > 0 {
            self.record_llamacpp_timings(tokens, start.elapsed().as_secs_f64());
        }

        body.choices
            .first()
            .map(|c| c.message.content.clone())
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
            return Err(ProviderError::Api(format!("llama.cpp {} - {}", status, msg)));
        }

        let mut stream = response.bytes_stream();
        let mut full_response = String::new();
        let mut completion_tokens: u64 = 0;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| ProviderError::Network(format!("llama.cpp stream error: {e}")))?;
            if let Ok(text) = std::str::from_utf8(&chunk) {
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() || line == "data: [DONE]" {
                        continue;
                    }
                    if let Some(data) = line.strip_prefix("data: ") {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                            // The final chunk carries usage (with an empty choices array).
                            if let Some(usage) = json.get("usage").and_then(|u| u.as_object()) {
                                if let Some(t) = usage
                                    .get("completion_tokens")
                                    .and_then(|v| v.as_u64())
                                {
                                    completion_tokens = t;
                                }
                            }
                            if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                                if let Some(first) = choices.first() {
                                    if let Some(delta) = first.get("delta") {
                                        if let Some(content) =
                                            delta.get("content").and_then(|c| c.as_str())
                                        {
                                            callback(content);
                                            full_response.push_str(content);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if completion_tokens > 0 {
            self.record_llamacpp_timings(completion_tokens, start.elapsed().as_secs_f64());
        }

        Ok(full_response)
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
            content: "hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert_eq!(json, r#"{"role":"user","content":"hello"}"#);
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
