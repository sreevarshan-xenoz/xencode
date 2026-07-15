//! Ollama provider client — local LLM inference.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use xencode_core_rs::{HealthStatus, ModelResponse, ProviderConfig, ProviderHealth};

use crate::{ModelProvider, ProviderError};

/// Ollama model provider client.
pub struct OllamaClient {
    config: ProviderConfig,
    http_client: Client,
}

impl OllamaClient {
    pub fn new(config: ProviderConfig) -> Self {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .unwrap_or_default();
        Self { config, http_client }
    }
}

#[async_trait]
impl ModelProvider for OllamaClient {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn provider_type(&self) -> xencode_core_rs::ProviderType {
        xencode_core_rs::ProviderType::Ollama
    }

    async fn generate(&self, model: &str, prompt: &str) -> Result<ModelResponse, ProviderError> {
        let start = Instant::now();
        let url = format!("{}/api/generate", self.config.base_url);
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
        });

        let response = self.http_client
            .post(&url)
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

        let data: OllamaGenerateResponse = response.json().await.map_err(ProviderError::Http)?;
        let elapsed = start.elapsed();

        Ok(ModelResponse {
            model: model.to_string(),
            content: data.response.unwrap_or_default(),
            latency_ms: elapsed.as_millis() as u64,
            tokens_generated: data.eval_count.unwrap_or(0) as u32,
            error: data.error,
        })
    }

    async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<ModelResponse, ProviderError> {
        let start = Instant::now();
        let url = format!("{}/api/generate", self.config.base_url);
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": true,
        });

        let response = self.http_client
            .post(&url)
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

        let mut full_content = String::new();
        let mut stream = response.bytes_stream();
        use futures::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(ProviderError::Http)?;
            if let Ok(text) = String::from_utf8(chunk.to_vec()) {
                for line in text.lines() {
                    if let Ok(resp) = serde_json::from_str::<OllamaStreamResponse>(line) {
                        if let Some(content) = resp.response {
                            full_content.push_str(&content);
                            let _ = tx.send(content).await;
                        }
                        if resp.done {
                            let elapsed = start.elapsed();
                            return Ok(ModelResponse {
                                model: model.to_string(),
                                content: full_content,
                                latency_ms: elapsed.as_millis() as u64,
                                tokens_generated: resp.eval_count.unwrap_or(0) as u32,
                                error: None,
                            });
                        }
                    }
                }
            }
        }

        let elapsed = start.elapsed();
        Ok(ModelResponse {
            model: model.to_string(),
            content: full_content,
            latency_ms: elapsed.as_millis() as u64,
            tokens_generated: 0,
            error: None,
        })
    }

    async fn health(&self) -> ProviderHealth {
        let start = Instant::now();
        let url = format!("{}/api/tags", self.config.base_url);

        match self.http_client.get(&url).send().await {
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
                message: Some("Ollama not reachable".to_string()),
            },
        }
    }

    async fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let url = format!("{}/api/tags", self.config.base_url);
        let response = self.http_client.get(&url).send().await.map_err(ProviderError::Http)?;
        let data: OllamaTagsResponse = response.json().await.map_err(ProviderError::Http)?;
        Ok(data.models.iter().map(|m| m.name.clone()).collect())
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    response: Option<String>,
    error: Option<String>,
    eval_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamResponse {
    response: Option<String>,
    done: bool,
    eval_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}
