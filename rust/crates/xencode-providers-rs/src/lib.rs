use std::fmt;
use std::io::{BufRead, BufReader};

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
/// Currently only supports local Ollama.
pub struct ProviderManager {
    ollama_client: OllamaClient,
}

impl ProviderManager {
    pub fn new(ollama_client: OllamaClient) -> Self {
        Self { ollama_client }
    }

    /// Generate a response synchronously (blocks until the full response is ready).
    pub fn generate(&self, model: &str, messages: &[ChatMessage]) -> Result<String, ProviderError> {
        let url = format!("{}/api/chat", self.ollama_client.base_url());
        
        let payload = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false
        });

        let response = ureq::post(&url)
            .send_json(&payload)
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let body: OllamaResponse = response
            .into_json()
            .map_err(|e| ProviderError::Parse(e.to_string()))?;

        Ok(body.message.content)
    }

    /// Generate a response and stream it token-by-token.
    pub fn generate_stream<F>(
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

        let response = ureq::post(&url)
            .send_json(&payload)
            .map_err(|e| ProviderError::Network(e.to_string()))?;

        let reader = BufReader::new(response.into_reader());
        let mut full_response = String::new();

        for line in reader.lines() {
            let line = line.map_err(|e| ProviderError::Network(e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }

            if let Ok(chunk) = serde_json::from_str::<OllamaResponse>(&line) {
                callback(&chunk.message.content);
                full_response.push_str(&chunk.message.content);
                
                if chunk.done {
                    break;
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
