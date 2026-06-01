use std::fmt;
use futures_util::StreamExt;

use serde::{Deserialize, Serialize};
use xencode_models_rs::OllamaClient;

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
/// Currently supports local Ollama and OpenRouter.
pub struct ProviderManager {
    ollama_client: OllamaClient,
    openrouter_api_key: Option<String>,
    client: reqwest::Client,
}

impl ProviderManager {
    pub fn new(ollama_client: OllamaClient, openrouter_api_key: Option<String>) -> Self {
        let client = reqwest::Client::new();
        Self { ollama_client, openrouter_api_key, client }
    }

    /// Generate a response asynchronously (resolves when the full response is ready).
    pub async fn generate(&self, model: &str, messages: &[ChatMessage]) -> Result<String, ProviderError> {
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
    pub async fn generate_stream<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        // Simple routing based on model prefix or available keys
        // OpenRouter models usually contain a slash, e.g., "anthropic/claude-3-opus"
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
