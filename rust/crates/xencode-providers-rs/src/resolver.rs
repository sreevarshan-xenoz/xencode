//! Provider resolver — routes requests to the right provider with retry/fallback.

use crate::{ModelProvider, OllamaClient, OpenRouterClient, ProviderError};
use std::sync::Arc;
use xencode_core_rs::{ModelResponse, ProviderConfig, ProviderType};

/// Resolves and routes requests to the appropriate provider.
pub struct ProviderResolver {
    providers: Vec<Box<dyn ModelProvider>>,
}

impl ProviderResolver {
    /// Create a new resolver from provider configs.
    pub fn new(configs: Vec<ProviderConfig>) -> Self {
        let mut providers: Vec<Box<dyn ModelProvider>> = Vec::new();
        for config in configs {
            match config.provider_type {
                ProviderType::Ollama => {
                    providers.push(Box::new(OllamaClient::new(config)));
                }
                ProviderType::OpenRouter => {
                    providers.push(Box::new(OpenRouterClient::new(config)));
                }
                _ => {
                    // Skip unknown providers
                    continue;
                }
            }
        }
        Self { providers }
    }

    /// Find a provider by name.
    pub fn find_provider(&self, name: &str) -> Option<&Box<dyn ModelProvider>> {
        self.providers.iter().find(|p| p.name() == name)
    }

    /// Get all providers.
    pub fn all_providers(&self) -> &[Box<dyn ModelProvider>] {
        &self.providers
    }

    /// Generate a response with automatic fallback.
    pub async fn generate_with_fallback(
        &self,
        provider_name: &str,
        model: &str,
        prompt: &str,
        max_retries: u32,
    ) -> Result<ModelResponse, ProviderError> {
        let mut last_error = None;

        // Try the requested provider first
        for attempt in 0..=max_retries {
            if let Some(provider) = self.find_provider(provider_name) {
                match provider.generate(model, prompt).await {
                    Ok(response) => return Ok(response),
                    Err(e) => {
                        last_error = Some(e);
                        if attempt < max_retries {
                            tokio::time::sleep(std::time::Duration::from_millis(500 * (attempt + 1))).await;
                        }
                    }
                }
            }
        }

        // Fallback: try other providers
        for provider in &self.providers {
            if provider.name() != provider_name {
                match provider.generate(model, prompt).await {
                    Ok(response) => return Ok(response),
                    Err(e) => {
                        last_error = Some(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or(ProviderError::Provider("All providers failed".to_string())))
    }
}
