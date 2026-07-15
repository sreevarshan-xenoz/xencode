//! Model management and registry for AI models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xencode_core_rs::{ModelDefinition, ModelResponse, ProviderType};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Provider error: {0}")]
    Provider(String),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Registry of available models across all providers.
pub struct ModelRegistry {
    models: HashMap<String, Vec<ModelDefinition>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Register models for a provider.
    pub fn register(&mut self, provider: &str, models: Vec<ModelDefinition>) {
        self.models.insert(provider.to_string(), models);
    }

    /// Get all models for a provider.
    pub fn get_provider_models(&self, provider: &str) -> Vec<&ModelDefinition> {
        self.models.get(provider).map(|m| m.iter().collect()).unwrap_or_default()
    }

    /// Get a specific model by name across all providers.
    pub fn find_model(&self, name: &str) -> Option<&ModelDefinition> {
        self.models.values().flatten().find(|m| m.name == name)
    }

    /// Get all models across all providers.
    pub fn all_models(&self) -> Vec<&ModelDefinition> {
        self.models.values().flatten().collect()
    }

    /// List unique model names.
    pub fn list_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.models.values()
            .flatten()
            .map(|m| m.name.clone())
            .collect();
        names.sort();
        names.dedup();
        names
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        registry.register("ollama", vec![
            ModelDefinition {
                name: "qwen3:4b".to_string(),
                provider: "ollama".to_string(),
                display_name: "Qwen 3 (4B)".to_string(),
                capabilities: vec!["code".to_string(), "general".to_string(), "reasoning".to_string()],
                context_length: 32768,
                speed_score: 8,
                quality_score: 7,
            },
            ModelDefinition {
                name: "llama2:7b".to_string(),
                provider: "ollama".to_string(),
                display_name: "Llama 2 (7B)".to_string(),
                capabilities: vec!["general".to_string(), "creative".to_string()],
                context_length: 4096,
                speed_score: 6,
                quality_score: 7,
            },
            ModelDefinition {
                name: "codellama:7b".to_string(),
                provider: "ollama".to_string(),
                display_name: "Code Llama (7B)".to_string(),
                capabilities: vec!["code".to_string(), "reasoning".to_string()],
                context_length: 16384,
                speed_score: 6,
                quality_score: 7,
            },
            ModelDefinition {
                name: "mistral:7b".to_string(),
                provider: "ollama".to_string(),
                display_name: "Mistral (7B)".to_string(),
                capabilities: vec!["general".to_string(), "reasoning".to_string()],
                context_length: 8192,
                speed_score: 7,
                quality_score: 8,
            },
        ]);
        registry
    }
}
