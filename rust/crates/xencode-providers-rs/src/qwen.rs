use std::fmt;

use futures_util::StreamExt;
use serde::Deserialize;

use crate::{ChatMessage, ProviderError};

/// Qwen AI model provider using the OpenAI-compatible chat API.
///
/// API: `POST https://chat.qwen.ai/v1/chat/completions`
/// Auth: `Authorization: Bearer <api_key>`
///
/// Supports both streaming and non-streaming modes.
pub struct QwenProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct QwenChoice {
    message: ChatMessageContent,
}

#[derive(Debug, Deserialize)]
struct ChatMessageContent {
    content: String,
}

#[derive(Debug, Default, Deserialize)]
struct DeltaContent {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct QwenResponse {
    choices: Vec<QwenChoice>,
}

#[derive(Debug, Deserialize)]
struct QwenStreamChunk {
    choices: Vec<QwenStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct QwenStreamChoice {
    delta: DeltaContent,
}

impl QwenProvider {
    /// Create a new Qwen provider.
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let client = reqwest::Client::new();
        Self {
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://chat.qwen.ai/v1".to_string()),
            client,
        }
    }

    /// Generate a non-streaming response.
    pub async fn generate(
        &self,
        model: &str,
        messages: &[ChatMessage],
    ) -> Result<String, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);

        let payload = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Qwen request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api(format!("Qwen {} - {body}", status)));
        }

        let body: QwenResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("Qwen response parse: {e}")))?;

        body.choices
            .first()
            .map(|c| c.message.content.clone())
            .ok_or_else(|| ProviderError::Parse("Qwen: empty choices in response".to_string()))
    }

    /// Generate a streaming response, calling `callback` for each token.
    pub async fn generate_stream<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        mut callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        let url = format!("{}/chat/completions", self.base_url);

        let payload = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Qwen stream request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api(format!("Qwen {} - {body}", status)));
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
                        if let Ok(parsed) = serde_json::from_str::<QwenStreamChunk>(data) {
                            for choice in &parsed.choices {
                                if !choice.delta.content.is_empty() {
                                    callback(&choice.delta.content);
                                    full_response.push_str(&choice.delta.content);
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

impl fmt::Debug for QwenProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenProvider")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_creation() {
        let provider = QwenProvider::new("sk-test-123".to_string(), None);
        // Just verify no panics
        assert!(provider.api_key == "sk-test-123");
        assert_eq!(provider.base_url, "https://chat.qwen.ai/v1");
    }

    #[test]
    fn custom_base_url() {
        let provider = QwenProvider::new(
            "sk-test".to_string(),
            Some("https://myqwen.example.com/v1".to_string()),
        );
        assert_eq!(provider.base_url, "https://myqwen.example.com/v1");
    }
}
