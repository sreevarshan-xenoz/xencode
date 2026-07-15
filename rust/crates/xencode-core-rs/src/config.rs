//! Configuration types for Xencode.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level Xencode configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XencodeConfig {
    pub default_model: String,
    pub theme: ThemeConfig,
    pub providers: Vec<ProviderSettings>,
    pub cache: CacheSettings,
    pub memory: MemorySettings,
    pub telemetry: TelemetrySettings,
    pub plugins: PluginSettings,
}

impl Default for XencodeConfig {
    fn default() -> Self {
        Self {
            default_model: "qwen3:4b".to_string(),
            theme: ThemeConfig::default(),
            providers: vec![
                ProviderSettings {
                    name: "ollama".to_string(),
                    enabled: true,
                    base_url: "http://localhost:11434".to_string(),
                    api_key: None,
                    default_model: "qwen3:4b".to_string(),
                    timeout_seconds: 60,
                    max_retries: 3,
                },
            ],
            cache: CacheSettings::default(),
            memory: MemorySettings::default(),
            telemetry: TelemetrySettings::default(),
            plugins: PluginSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    pub mode: String,
    pub primary: String,
    pub accent: String,
    pub background: String,
    pub foreground: String,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            mode: "dark".to_string(),
            primary: "#00ff88".to_string(),
            accent: "#00aaff".to_string(),
            background: "#1a1b26".to_string(),
            foreground: "#c0caf5".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSettings {
    pub name: String,
    pub enabled: bool,
    pub base_url: String,
    pub api_key: Option<String>,
    pub default_model: String,
    pub timeout_seconds: u64,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSettings {
    pub enabled: bool,
    pub max_size_mb: u64,
    pub ttl_seconds: u64,
    pub compression: bool,
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size_mb: 500,
            ttl_seconds: 3600,
            compression: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySettings {
    pub max_sessions: usize,
    pub max_messages_per_session: usize,
    pub persist: bool,
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            max_sessions: 50,
            max_messages_per_session: 200,
            persist: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySettings {
    pub enabled: bool,
    pub analytics_dir: PathBuf,
}

impl Default for TelemetrySettings {
    fn default() -> Self {
        Self {
            enabled: true,
            analytics_dir: PathBuf::from("~/.xencode/analytics"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSettings {
    pub plugin_dir: PathBuf,
    pub enabled_plugins: Vec<String>,
    pub allow_external: bool,
}

impl Default for PluginSettings {
    fn default() -> Self {
        Self {
            plugin_dir: PathBuf::from("~/.xencode/plugins"),
            enabled_plugins: Vec::new(),
            allow_external: false,
        }
    }
}
