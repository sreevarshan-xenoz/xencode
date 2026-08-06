use std::fmt;
use std::sync::Mutex;

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use xencode_models_rs::OllamaClient;

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

/// ProviderManager abstracts over local and cloud models.
///
/// Supports Ollama (local), OpenRouter (cloud), Qwen (cloud), Gemini (cloud),
/// and Anthropic (cloud).
///
/// Features:
/// - Automatic provider routing based on model prefix
/// - Retry with exponential backoff on transient failures
/// - Health tracking across all providers
pub struct ProviderManager {
    ollama_client: OllamaClient,
    openrouter_api_key: Option<String>,
    qwen_api_key: Option<String>,
    gemini_api_key: Option<String>,
    anthropic_api_key: Option<String>,
    retry_config: RetryConfig,
    client: reqwest::Client,
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
            openrouter_api_key,
            qwen_api_key,
            gemini_api_key,
            anthropic_api_key,
            retry_config: RetryConfig::default(),
            client,
        }
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
    /// - contains `/` with OpenRouter key → OpenRouter
    /// - else → local Ollama
    ///
    /// Retries on transient errors (network failures, 5xx, 429) using exponential backoff.
    pub async fn generate(&self, model: &str, messages: &[ChatMessage]) -> Result<String, ProviderError> {
        let model_owned = model.to_string();
        let messages_owned = messages.to_vec();

        retry::retry_async(&self.retry_config, || async {
            self.generate_inner(&model_owned, &messages_owned).await
        }).await
    }

    /// Inner generate without retry wrapping (used by retry logic).
    async fn generate_inner(&self, model: &str, messages: &[ChatMessage]) -> Result<String, ProviderError> {
        // Route based on model prefix
        if let Some(inner_model) = model.strip_prefix("anthropic:") {
            if let Some(ref key) = self.anthropic_api_key {
                let provider = anthropic::AnthropicProvider::new(key.clone(), None, None);
                return provider.generate(inner_model, messages, None).await;
            }
            return Err(ProviderError::Api("Anthropic API key not configured".to_string()));
        }

        if let Some(inner_model) = model.strip_prefix("qwen:") {
            if let Some(ref key) = self.qwen_api_key {
                let provider = qwen::QwenProvider::new(key.clone(), None);
                return provider.generate(inner_model, messages).await;
            }
            return Err(ProviderError::Api("Qwen API key not configured".to_string()));
        }

        if let Some(inner_model) = model.strip_prefix("google_gemini:") {
            if let Some(ref key) = self.gemini_api_key {
                let provider = gemini::GeminiProvider::new(key.clone(), None);
                return provider.generate(inner_model, messages, None, None).await;
            }
            return Err(ProviderError::Api("Google Gemini API key not configured".to_string()));
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

        let response = self.client.post(&url)
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
    /// Retries on transient errors (network failures, 5xx, 429) using exponential backoff.
    pub async fn generate_stream<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
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

        retry::retry_async(&self.retry_config, || async {
            let cb_ref = &cb;
            self.generate_stream_inner(&model_owned, &messages_owned, |token| {
                let mut guard = cb_ref.lock().unwrap();
                guard(token);
            }).await
        }).await
    }

    /// Inner stream generate without retry wrapping.
    async fn generate_stream_inner<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        // Route based on model prefix
        if let Some(inner_model) = model.strip_prefix("anthropic:") {
            if let Some(ref key) = self.anthropic_api_key {
                let provider = anthropic::AnthropicProvider::new(key.clone(), None, None);
                return provider.generate_stream(inner_model, messages, None, callback).await;
            }
            return Err(ProviderError::Api("Anthropic API key not configured".to_string()));
        }

        if let Some(inner_model) = model.strip_prefix("qwen:") {
            if let Some(ref key) = self.qwen_api_key {
                let provider = qwen::QwenProvider::new(key.clone(), None);
                return provider.generate_stream(inner_model, messages, callback).await;
            }
            return Err(ProviderError::Api("Qwen API key not configured".to_string()));
        }

        if let Some(inner_model) = model.strip_prefix("google_gemini:") {
            if let Some(ref key) = self.gemini_api_key {
                let provider = gemini::GeminiProvider::new(key.clone(), None);
                return provider.generate_stream(inner_model, messages, None, None, callback).await;
            }
            return Err(ProviderError::Api("Google Gemini API key not configured".to_string()));
        }

        // OpenRouter route
        if model.contains('/') && self.openrouter_api_key.is_some() {
            self.generate_stream_openrouter(model, messages, callback).await
        } else {
            self.generate_stream_ollama(model, messages, callback).await
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

        let response = self.client.post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(e.to_string()))?;

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

        let response = self.client.post(url)
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
            return Err(ProviderError::Api(format!("OpenRouter {} - {}", status, msg)));
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

        let response = self.client.post(url)
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
            return Err(ProviderError::Api(format!("OpenRouter {} - {}", status, msg)));
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
                                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
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
}
