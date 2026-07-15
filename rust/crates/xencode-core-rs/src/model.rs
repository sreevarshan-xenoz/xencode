//! AI model definitions and chat message types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A chat message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// AI model definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub name: String,
    pub provider: String,
    pub display_name: String,
    pub capabilities: Vec<String>,
    pub context_length: u32,
    pub speed_score: u8,
    pub quality_score: u8,
}

impl ModelDefinition {
    pub fn new(name: &str, provider: &str) -> Self {
        Self {
            name: name.to_string(),
            provider: provider.to_string(),
            display_name: name.to_string(),
            capabilities: Vec::new(),
            context_length: 4096,
            speed_score: 5,
            quality_score: 5,
        }
    }
}

/// Model provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub provider_type: ProviderType,
    pub base_url: String,
    pub api_key: Option<String>,
    pub default_model: String,
    pub models: Vec<ModelDefinition>,
    pub timeout_seconds: u64,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProviderType {
    Ollama,
    OpenRouter,
    Qwen,
    Gemini,
    Custom,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::Ollama => write!(f, "ollama"),
            ProviderType::OpenRouter => write!(f, "openrouter"),
            ProviderType::Qwen => write!(f, "qwen"),
            ProviderType::Gemini => write!(f, "gemini"),
            ProviderType::Custom => write!(f, "custom"),
        }
    }
}

/// A response from a model ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResponse {
    pub responses: Vec<ModelResponse>,
    pub final_output: String,
    pub method: String,
    pub confidence: f64,
    pub processing_time_ms: u64,
}

/// Response from a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub model: String,
    pub content: String,
    pub latency_ms: u64,
    pub tokens_generated: u32,
    pub error: Option<String>,
}

/// Tool execution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool: String,
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub duration_ms: u64,
}

/// Health status of a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealth {
    pub provider: String,
    pub status: HealthStatus,
    pub latency_ms: u64,
    pub error_rate: f64,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub model_count: usize,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// System status summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub version: String,
    pub uptime_seconds: u64,
    pub ollama_running: bool,
    pub providers: Vec<ProviderHealth>,
    pub cache_stats: CacheStats,
    pub memory_sessions: usize,
}

/// Cache statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size_bytes: u64,
    pub max_size_bytes: u64,
    pub entry_count: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Available output formats for CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Text,
    Json,
    Yaml,
    Markdown,
    Html,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(OutputFormat::Text),
            "json" => Ok(OutputFormat::Json),
            "yaml" => Ok(OutputFormat::Yaml),
            "markdown" | "md" => Ok(OutputFormat::Markdown),
            "html" => Ok(OutputFormat::Html),
            _ => Err(format!("Unknown output format: {}", s)),
        }
    }
}
