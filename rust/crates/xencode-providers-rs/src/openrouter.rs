//! OpenRouter provider client — cloud model access.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use xencode_core_rs::{HealthStatus, ModelResponse, ProviderConfig, ProviderHealth};

use crate::{ModelProvider, ProviderError};

/// OpenRouter model provider client.
pub struct OpenRouterClient {
    config: ProviderConfig,
    http_client: Client,
}

impl OpenRouterClient {
    pub fn new(config: ProviderConfig) -> Self {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .unwrap_or_default();
        Self { config, http_client }
    }
}

#[async_trait]
impl ModelProvider for OpenRouterClient {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn provider_type(&self) -> xencode_core_rs::ProviderType {
        xencode_core_rs::ProviderType::OpenRouter
    }

    async fn generate(&self, model: &str, prompt: &str) -> Result<ModelResponse, ProviderError> {
        if self.config.api_key.is_none() {
            return Err(ProviderError::AuthError("OpenRouter API key not configured".to_string()));
        }

        let start = Instant::now();
        let url = format!("{}/chat/completions", self.config.base_url);
        let api_key = self.config.api_key.as_deref().unwrap_or("");

        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        });

        let response = self.http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    ProviderError::Timeout
                } else {
                    ProviderError::Http(e)
                }
            })?;

        if response.status().is_client_error() {
            return Err(ProviderError::AuthError("Invalid API key or insufficient credits".to_string()));
        }

        let data: OpenRouterResponse = response.json().await.map_err(ProviderError::Http)?;
        let elapsed = start.elapsed();

        Ok(ModelResponse {
            model: model.to_string(),
            content: data.choices.first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default(),
            latency_ms: elapsed.as_millis() as u64,
            tokens_generated: data.usage.as_ref().map(|u| u.total_tokens as u32).unwrap_or(0),
            error: None,
        })
    }

    async fn generate_stream(
        &self,
        _model: &str,
        _prompt: &str,
        _tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<ModelResponse, ProviderError> {
        Err(ProviderError::Provider("Streaming not yet implemented for OpenRouter".to_string()))
    }

    async fn health(&self) -> ProviderHealth {
        let start = Instant::now();

        if self.config.api_key.is_none() {
            return ProviderHealth {
                provider: self.config.name.clone(),
                status: HealthStatus::Degraded,
                latency_ms: 0,
                error_rate: 0.0,
                last_check: chrono::Utc::now(),
                model_count: 0,
                message: Some("API key not configured".to_string()),
            };
        }

        let url = format!("{}/models", self.config.base_url);
        let api_key = self.config.api_key.as_deref().unwrap_or("");

        match self.http_client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await
        {
            Ok(response) => {
                let elapsed = start.elapsed();
                let status = if response.status().is_success() {
                    HealthStatus::Healthy
                } else {
                    HealthStatus::Degraded
                };
                ProviderHealth {
                    provider: self.config.name.clone(),
                    status,
                    latency_ms: elapsed.as_millis() as u64,
                    error_rate: 0.0,
                    last_check: chrono::Utc::now(),
                    model_count: 0,
                    message: None,
                }
            }
            Err(_) => ProviderHealth {
                provider: self.config.name.clone(),
                status: HealthStatus::Unhealthy,
                latency_ms: 0,
                error_rate: 1.0,
                last_check: chrono::Utc::now(),
                model_count: 0,
                message: Some("OpenRouter not reachable".to_string()),
            },
        }
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        Ok(vec![
            "qwen-3-4b".to_string(),
            "mistral-7b".to_string(),
            "codellama-7b".to_string(),
        ])
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
struct Message {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    total_tokens: u64,
}
