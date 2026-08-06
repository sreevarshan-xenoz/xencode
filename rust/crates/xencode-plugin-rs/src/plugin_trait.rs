use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during plugin operations.
#[derive(Debug, Error)]
pub enum PluginError {
    #[error("Plugin not found: {0}")]
    NotFound(String),
    #[error("Plugin already registered: {0}")]
    AlreadyRegistered(String),
    #[error("Plugin initialization failed: {0}")]
    InitFailed(String),
    #[error("Plugin event handling failed: {0}")]
    EventFailed(String),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Plugin version incompatible: expected {expected}, got {actual}")]
    VersionMismatch { expected: String, actual: String },
}

/// An event dispatched to plugins by the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEvent {
    pub event_type: String,
    pub data: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl PluginEvent {
    pub fn new(event_type: &str, data: serde_json::Value) -> Self {
        Self {
            event_type: event_type.to_string(),
            data,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// A response returned by a plugin after handling an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginResponse {
    pub plugin_name: String,
    pub data: serde_json::Value,
}

/// The host interface available to plugins.
pub trait PluginHost: Send + Sync {
    fn get_plugin_dir(&self) -> &str;
    fn get_config_value(&self, key: &str) -> Option<String>;
}

/// Trait that all Xencode plugins must implement.
pub trait XencodePlugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn description(&self) -> &str;
    fn initialize(&mut self, host: &dyn PluginHost) -> Result<(), PluginError>;
    fn shutdown(&mut self) -> Result<(), PluginError>;
    fn handle_event(&mut self, event: PluginEvent) -> Result<Option<PluginResponse>, PluginError>;
}
