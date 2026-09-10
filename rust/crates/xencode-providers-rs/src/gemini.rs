use std::fmt;

use futures_util::StreamExt;
use serde::Deserialize;

use crate::{ChatMessage, ProviderError};

/// Google Gemini model provider.
///
/// API: `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
/// Auth: `key={api_key}` query parameter
///
/// Supports both streaming and non-streaming modes.
pub struct GeminiProvider {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

// ── Gemini API response types ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
    #[serde(default)]
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: Option<Content>,
}

#[derive(Debug, Deserialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Deserialize)]
struct Part {
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct PromptFeedback {
    #[serde(default)]
    block_reason: Option<String>,
}

/// Gemini streaming chunk (SSE `data: {...}`).
#[derive(Debug, Deserialize)]
struct GeminiStreamChunk {
    candidates: Option<Vec<StreamCandidate>>,
}

#[derive(Debug, Deserialize)]
struct StreamCandidate {
    content: Option<StreamContent>,
}

#[derive(Debug, Deserialize)]
struct StreamContent {
    parts: Vec<Part>,
}

// ── Implementation ──────────────────────────────────────────────────────────

impl GeminiProvider {
    /// Create a new Gemini provider with the given API key.
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let client = reqwest::Client::new();
        Self {
            api_key,
            base_url: base_url
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".to_string()),
            client,
        }
    }

    /// Convert internal ChatMessage roles to Gemini roles ("user" or "model").
    fn convert_messages(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
        let mut contents = Vec::new();
        // Track whether we need to prepend system content to the first user message
        let mut system_buffer = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    // Accumulate system messages to prepend to the first user message
                    if !system_buffer.is_empty() {
                        system_buffer.push_str("\n\n");
                    }
                    system_buffer.push_str(&msg.content);
                }
                "user" | "assistant" => {
                    let gemini_role = if msg.role == "assistant" {
                        "model"
                    } else {
                        "user"
                    };
                    let mut text = msg.content.clone();

                    // Prepend accumulated system messages to the first user message
                    if gemini_role == "user" && !system_buffer.is_empty() {
                        text = format!("{}\n\n{}", system_buffer, text);
                        system_buffer.clear();
                    }

                    contents.push(serde_json::json!({
                        "role": gemini_role,
                        "parts": [{"text": text}]
                    }));
                }
                _ => {
                    // Unknown role — treat as user
                    contents.push(serde_json::json!({
                        "role": "user",
                        "parts": [{"text": msg.content}]
                    }));
                }
            }
        }

        // If there's leftover system content with no user message, create one
        if !system_buffer.is_empty() {
            contents.insert(
                0,
                serde_json::json!({
                    "role": "user",
                    "parts": [{"text": system_buffer}]
                }),
            );
        }

        contents
    }

    /// Build the request payload for the Gemini API.
    fn build_payload(
        messages: &[ChatMessage],
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> serde_json::Value {
        let contents = Self::convert_messages(messages);

        let mut generation_config = serde_json::json!({
            "candidateCount": 1,
            "stopSequences": [],
        });

        if let Some(mt) = max_tokens {
            generation_config["maxOutputTokens"] = serde_json::json!(mt);
        }
        if let Some(tmp) = temperature {
            generation_config["temperature"] = serde_json::json!(tmp);
        }

        serde_json::json!({
            "contents": contents,
            "generationConfig": generation_config,
            "safetySettings": [
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        })
    }

    /// Generate a non-streaming response.
    pub async fn generate(
        &self,
        model: &str,
        messages: &[ChatMessage],
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<String, ProviderError> {
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        );
        let payload = Self::build_payload(messages, max_tokens, temperature);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Gemini request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::api("Gemini", status, body));
        }

        let body: GeminiResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("Gemini response parse: {e}")))?;

        // Check for blocking
        if let Some(feedback) = &body.prompt_feedback {
            if let Some(ref reason) = feedback.block_reason {
                return Err(ProviderError::api_message(format!(
                    "Gemini request blocked: {reason}"
                )));
            }
        }

        let text = body
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .map(|c| {
                c.parts
                    .into_iter()
                    .map(|p| p.text)
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        Ok(text)
    }

    /// Generate a streaming response, calling `callback` for each token.
    ///
    /// Uses the `streamGenerateContent` endpoint which returns SSE chunks.
    pub async fn generate_stream<F>(
        &self,
        model: &str,
        messages: &[ChatMessage],
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        mut callback: F,
    ) -> Result<String, ProviderError>
    where
        F: FnMut(&str),
    {
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, model, self.api_key
        );
        let payload = Self::build_payload(messages, max_tokens, temperature);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("Gemini stream request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::api("Gemini", status, body));
        }

        let mut stream = response.bytes_stream();
        let mut full_response = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
            if let Ok(text) = std::str::from_utf8(&chunk) {
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    if let Some(data) = line.strip_prefix("data: ") {
                        // Gemini sends `data: {"candidates": [...]}`
                        if let Ok(parsed) = serde_json::from_str::<GeminiStreamChunk>(data) {
                            if let Some(candidates) = parsed.candidates {
                                for candidate in &candidates {
                                    if let Some(ref content) = candidate.content {
                                        for part in &content.parts {
                                            if !part.text.is_empty() {
                                                callback(&part.text);
                                                full_response.push_str(&part.text);
                                            }
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

impl fmt::Debug for GeminiProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiProvider")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_creation() {
        let provider = GeminiProvider::new("AIza-test-123".to_string(), None);
        assert_eq!(
            provider.base_url,
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[test]
    fn custom_base_url() {
        let provider = GeminiProvider::new(
            "test-key".to_string(),
            Some("https://custom.gemini.example.com/v1".to_string()),
        );
        assert_eq!(provider.base_url, "https://custom.gemini.example.com/v1");
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
        let contents = GeminiProvider::convert_messages(&messages);
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Hello");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["text"], "Hi there");
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
        let contents = GeminiProvider::convert_messages(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        // System content should be prepended
        let text = contents[0]["parts"][0]["text"].as_str().unwrap();
        assert!(text.contains("You are a helpful assistant."));
        assert!(text.contains("Hello"));
    }

    #[test]
    fn convert_messages_system_only() {
        let messages = vec![ChatMessage {
            role: "system".to_string(),
            content: "Be concise.".to_string(),
        }];
        let contents = GeminiProvider::convert_messages(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "Be concise.");
    }

    #[test]
    fn convert_messages_unknown_role() {
        let messages = vec![ChatMessage {
            role: "unknown".to_string(),
            content: "test".to_string(),
        }];
        let contents = GeminiProvider::convert_messages(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }
}
