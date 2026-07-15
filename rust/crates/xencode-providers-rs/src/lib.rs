//! Model provider implementations — Ollama, OpenRouter, and custom providers.

pub mod ollama;
pub mod openrouter;
pub mod resolver;

pub use ollama::OllamaClient;
pub use openrouter::OpenRouterClient;
pub use resolver::ProviderResolver;

use async_trait::async_trait;
use xencode_core_rs::{ModelResponse, ProviderConfig, ProviderHealth, ProviderType, HealthStatus};

/// Unified trait for all model providers.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn name(&self) -> &str;
    fn provider_type(&self) -> ProviderType;

    /// Generate a response from the model.
    async fn generate(&self, model: &str, prompt: &str) -> Result<ModelResponse, ProviderError>;

    /// Generate a streaming response.
    async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<ModelResponse, ProviderError>;

    /// Check provider health.
    async fn health(&self) -> ProviderHealth;

    /// List available models from this provider.
    async fn list_models(&self) -> Result<Vec<String>, ProviderError>;

    /// Get the provider configuration.
    fn config(&self) -> &ProviderConfig;
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Provider error: {0}")]
    Provider(String),
    #[error("Timeout")]
    Timeout,
    #[error("Rate limited")]
    RateLimited,
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Authentication error: {0}")]
    AuthError(String),
    #[error("Stream error: {0}")]
    StreamError(String),
}

/// Default provider configuration.
pub fn default_providers() -> Vec<ProviderConfig> {
    vec![
        ProviderConfig {
            name: "ollama".to_string(),
            provider_type: ProviderType::Ollama,
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            default_model: "qwen3:4b".to_string(),
            models: Vec::new(),
            timeout_seconds: 60,
            max_retries: 3,
        },
        ProviderConfig {
            name: "openrouter".to_string(),
            provider_type: ProviderType::OpenRouter,
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key: None,
            default_model: "qwen-3-4b".to_string(),
            models: Vec::new(),
            timeout_seconds: 60,
            max_retries: 3,
        },
    ]
}
