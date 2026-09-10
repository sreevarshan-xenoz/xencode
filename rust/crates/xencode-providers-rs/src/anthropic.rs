use std::fmt;

use futures_util::StreamExt;
use serde::Deserialize;

use crate::{ChatMessage, ProviderError};

/// Anthropic (Claude) model provider using the Messages API.
///
/// API: `POST https://api.anthropic.com/v1/messages`
/// Auth: `x-api-key` header
///
/// Supports both streaming and non-streaming modes.
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    anthropic_version: String,
    client: reqwest::Client,
}

// ── Anthropic API response types (non-streaming) ─────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    id: Option<String>,
    #[serde(rename = "type")]
    response_type: Option<String>,
    role: Option<String>,
    content: Option<Vec<ContentBlock>>,
    stop_reason: Option<String>,
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct UsageInfo {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorResponse {
    error: Option<AnthropicErrorDetail>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    error_type: Option<String>,
    message: Option<String>,
}

// ── Implementation ──────────────────────────────────────────────────────

impl AnthropicProvider {
    /// Create a new Anthropic provider with the given API key.
    ///
    /// `base_url` defaults to `https://api.anthropic.com`.
    /// `anthropic_version` defaults to `2023-06-01`.
    pub fn new(
        api_key: String,
        base_url: Option<String>,
        anthropic_version: Option<String>,
    ) -> Self {
        let client = reqwest::Client::new();
        Self {
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string()),
            anthropic_version: anthropic_version.unwrap_or_else(|| "2023-06-01".to_string()),
            client,
        }
    }

    /// Convert internal ChatMessage roles to Anthropic roles.
    ///
    /// Anthropic supports:
    /// - `user` → "user"
    /// - `assistant` → "assistant"
    /// - `system` → extracted into a separate `system` field (not in messages array)
    fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<serde_json::Value>) {
        let mut system_prompt: Option<String> = None;
        let mut converted = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    // Accumulate system messages into a single system prompt
                    let existing = system_prompt.get_or_insert_with(String::new);
                    if !existing.is_empty() {
                        existing.push_str("\n\n");
                    }
                    existing.push_str(&msg.content);
                }
                "user" | "assistant" => {
                    converted.push(serde_json::json!({
                        "role": msg.role,
                        "content": msg.content
                    }));
                }
                _ => {
                    // Unknown role — treat as user
                    converted.push(serde_json::json!({
                        "role": "user",
                        "content": msg.content
                    }));
                }
            }
        }

        (system_prompt, converted)
    }

    /// Build the request payload for the Anthropic Messages API.
    fn build_payload(model: &str, messages: &[ChatMessage], max_tokens: u32) -> serde_json::Value {
        let (system, anthy_messages) = Self::convert_messages(messages);

        let mut payload = serde_json::json!({
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthy_messages,
        });

        if let Some(system_text) = system {
            payload["system"] = serde_json::json!(system_text);
        }

        payload
    }

    /// Generate a non-streaming response.
    pub async fn generate(
        &self,
        model: &str,
        messages: &[ChatMessage],
        max_tokens: Option<u32>,
    ) -> Result<String, ProviderError> {
        let url = format!("{}/v1/messages", self.base_url);
        let payload = Self::build_payload(model, messages, max_tokens.unwrap_or(1024));

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Anthropic request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();

            // Try to extract a structured error message
            let detail = serde_json::from_str::<AnthropicErrorResponse>(&body_text)
                .ok()
                .and_then(|e| e.error)
                .and_then(|e| e.message)
                .unwrap_or(body_text);

            return Err(ProviderError::api("Anthropic", status, detail));
        }

        let body: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("Anthropic response parse: {e}")))?;

        // Extract text from all content blocks
        let text = body
            .content
            .unwrap_or_default()
            .into_iter()
            .filter(|block| block.block_type == "text")
            .map(|block| block.text)
            .collect::<Vec<_>>()
            .join("");

        if text.is_empty() {
            return Err(ProviderError::Parse(
                "Anthropic: empty text content in response".to_string(),
            ));
        }

        Ok(text)
    }

    /// Generate a streaming response, calling `callback` for each token/text chunk.
    ///
    /// Uses Anthropic's SSE streaming (`stream: true`) and parses `content_block_delta`
    /// events that contain `text_delta` blocks.
    pub async fn generate_stream<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        max_tokens: Option<u32>,
        mut callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        let url = format!("{}/v1/messages", self.base_url);
        let mut payload = Self::build_payload(model, messages, max_tokens.unwrap_or(1024));
        payload["stream"] = serde_json::json!(true);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Anthropic stream request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::api("Anthropic", status, body));
        }

        let mut stream = response.bytes_stream();
        let mut full_response = String::new();
        let mut event_type = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
            if let Ok(text) = std::str::from_utf8(&chunk) {
                for line in text.lines() {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    // Parse SSE event type lines: `event: content_block_delta`
                    if let Some(event_name) = trimmed.strip_prefix("event: ") {
                        event_type = event_name.to_string();
                        continue;
                    }

                    // Parse SSE data lines: `data: {...}`
                    if let Some(data) = trimmed.strip_prefix("data: ") {
                        // Reset event_type after using it to prevent stale carry-over
                        let current_event = std::mem::take(&mut event_type);

                        // Handle the streaming events
                        if current_event == "content_block_delta" {
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                                if let Some(delta) = parsed.get("delta") {
                                    if delta.get("type").and_then(|t| t.as_str())
                                        == Some("text_delta")
                                    {
                                        if let Some(text_chunk) =
                                            delta.get("text").and_then(|t| t.as_str())
                                        {
                                            if !text_chunk.is_empty() {
                                                callback(text_chunk);
                                                full_response.push_str(text_chunk);
                                            }
                                        }
                                    }
                                }
                            }
                        } else if event_type == "message_start" {
                            // Optionally capture message_id from the initial event
                            // (no text content in message_start)
                        }

                        // event_type was cleared by std::mem::take above
                    }
                }
            }
        }

        Ok(full_response)
    }
}

impl fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_creation() {
        let provider = AnthropicProvider::new("sk-ant-test-123".to_string(), None, None);
        assert_eq!(provider.base_url, "https://api.anthropic.com");
        assert_eq!(provider.anthropic_version, "2023-06-01");
    }

    #[test]
    fn custom_base_url_and_version() {
        let provider = AnthropicProvider::new(
            "sk-ant-test".to_string(),
            Some("https://custom.anthropic.example.com".to_string()),
            Some("2024-01-01".to_string()),
        );
        assert_eq!(provider.base_url, "https://custom.anthropic.example.com");
        assert_eq!(provider.anthropic_version, "2024-01-01");
    }

    #[test]
    fn convert_messages_simple() {
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hi there".to_string(),
            },
        ];
        let (system, msgs) = AnthropicProvider::convert_messages(&messages);
        assert!(system.is_none());
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hello");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "Hi there");
    }

    #[test]
    fn convert_messages_with_system() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];
        let (system, msgs) = AnthropicProvider::convert_messages(&messages);
        assert_eq!(system.unwrap(), "You are a helpful assistant.");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn convert_messages_system_only() {
        let messages = vec![ChatMessage {
            role: "system".to_string(),
            content: "Be concise.".to_string(),
        }];
        let (system, msgs) = AnthropicProvider::convert_messages(&messages);
        assert_eq!(system.unwrap(), "Be concise.");
        assert_eq!(msgs.len(), 0);
    }

    #[test]
    fn convert_messages_multiple_system_accumulates() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Be helpful.".to_string(),
            },
            ChatMessage {
                role: "system".to_string(),
                content: "Be concise.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];
        let (system, msgs) = AnthropicProvider::convert_messages(&messages);
        let system_text = system.unwrap();
        assert!(system_text.contains("Be helpful."));
        assert!(system_text.contains("Be concise."));
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn convert_messages_unknown_role() {
        let messages = vec![ChatMessage {
            role: "unknown".to_string(),
            content: "test".to_string(),
        }];
        let (system, msgs) = AnthropicProvider::convert_messages(&messages);
        assert!(system.is_none());
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn build_payload_with_system() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are Claude.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            },
        ];
        let payload =
            AnthropicProvider::build_payload("claude-3-5-sonnet-20241022", &messages, 2048);
        assert_eq!(payload["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(payload["max_tokens"], 2048);
        assert_eq!(payload["system"], "You are Claude.");
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn build_payload_no_system() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];
        let payload = AnthropicProvider::build_payload("claude-3-haiku-20240307", &messages, 512);
        assert_eq!(payload["model"], "claude-3-haiku-20240307");
        assert_eq!(payload["max_tokens"], 512);
        assert!(payload.get("system").is_none());
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
    }
}
