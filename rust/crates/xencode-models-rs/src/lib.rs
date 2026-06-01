pub mod health;
pub mod ollama;

pub use health::{HealthStatus, HealthTracker, ModelHealth};
pub use ollama::{ModelInfo, OllamaClient, OllamaError};
