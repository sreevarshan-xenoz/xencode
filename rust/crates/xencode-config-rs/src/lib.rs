//! Configuration management for Xencode.

use std::path::PathBuf;
use xencode_core_rs::XencodeConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Not found: {0}")]
    NotFound(String),
}

/// Manages Xencode configuration loading, saving, and access.
pub struct ConfigManager {
    config: XencodeConfig,
    config_path: PathBuf,
}

impl ConfigManager {
    /// Load configuration from default paths.
    pub fn load() -> Result<Self, ConfigError> {
        let config_dir = get_config_dir();
        let config_path = config_dir.join("config.toml");

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: XencodeConfig = toml::from_str(&content)
                .map_err(|e| ConfigError::Parse(e.to_string()))?;
            Ok(Self { config, config_path })
        } else {
            let config = XencodeConfig::default();
            let manager = Self { config, config_path };
            let _ = manager.save(); // best-effort initial save
            Ok(manager)
        }
    }

    /// Load from a specific path.
    pub fn load_from(path: &PathBuf) -> Result<Self, ConfigError> {
        if !path.exists() {
            return Err(ConfigError::NotFound(path.to_string_lossy().to_string()));
        }
        let content = std::fs::read_to_string(path)?;
        let config: XencodeConfig = toml::from_str(&content)
            .map_err(|e| ConfigError::Parse(e.to_string()))?;
        Ok(Self {
            config,
            config_path: path.clone(),
        })
    }

    /// Save configuration to disk.
    pub fn save(&self) -> Result<(), ConfigError> {
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(&self.config)
            .map_err(|e| ConfigError::Parse(e.to_string()))?;
        std::fs::write(&self.config_path, content)?;
        Ok(())
    }

    /// Get a reference to the config.
    pub fn get(&self) -> &XencodeConfig {
        &self.config
    }

    /// Get a mutable reference to the config.
    pub fn get_mut(&mut self) -> &mut XencodeConfig {
        &mut self.config
    }

    /// Get the config path.
    pub fn path(&self) -> &PathBuf {
        &self.config_path
    }
}

/// Get the Xencode config directory (~/.xencode).
pub fn get_config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".xencode")
}
