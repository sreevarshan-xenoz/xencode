use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// API key configuration for cloud model providers.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ApiKeys {
    #[serde(default)]
    pub openai_api_key: Option<String>,
    #[serde(default)]
    pub openrouter_api_key: Option<String>,
    #[serde(default)]
    pub google_gemini_api_key: Option<String>,
    #[serde(default)]
    pub qwen_client_id: Option<String>,
    #[serde(default)]
    pub qwen_api_key: Option<String>,
}

/// Top-level Xencode configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct XencodeConfig {
    /// The currently selected default model.
    #[serde(default = "default_model")]
    pub default_model: String,

    /// Active UI theme.
    #[serde(default = "default_theme")]
    pub active_theme: String,

    /// Ollama base URL.
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,

    /// llama.cpp server URL.
    #[serde(default = "default_llama_cpp_url")]
    pub llama_cpp_url: String,

    /// Path to a GGUF model used when auto-starting / loading llama-server.
    #[serde(default = "default_llama_cpp_model_path")]
    pub llama_cpp_model_path: String,

    /// Path to the llama-server executable (empty = resolved from PATH).
    #[serde(default)]
    pub llama_cpp_executable: String,

    /// Extra CLI arguments to pass when auto-starting llama-server.
    #[serde(default)]
    pub llama_cpp_args: Vec<String>,

    /// llama.cpp sampling default: temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_temperature: Option<f64>,

    /// llama.cpp sampling default: top-k.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_top_k: Option<i32>,

    /// llama.cpp sampling default: min-p.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_min_p: Option<f64>,

    /// llama.cpp sampling default: max generated tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_max_tokens: Option<u32>,

    /// Maximum cache size (number of entries).
    #[serde(default = "default_cache_size")]
    pub max_cache_size: usize,

    /// Response timeout in seconds.
    #[serde(default = "default_timeout")]
    pub response_timeout: u64,

    /// Whether caching is enabled.
    #[serde(default = "default_true")]
    pub cache_enabled: bool,

    /// Whether conversation memory is enabled.
    #[serde(default = "default_true")]
    pub memory_enabled: bool,

    /// Maximum memory items to retain.
    #[serde(default = "default_memory_items")]
    pub max_memory_items: usize,

    /// API keys for cloud providers.
    #[serde(default)]
    pub api_keys: ApiKeys,
}

fn default_model() -> String {
    "qwen2.5:7b".to_string()
}

fn default_theme() -> String {
    "ocean".to_string()
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

fn default_llama_cpp_url() -> String {
    "http://localhost:8080".to_string()
}

fn default_llama_cpp_model_path() -> String {
    String::new()
}

fn default_cache_size() -> usize {
    100
}

fn default_timeout() -> u64 {
    30
}

fn default_true() -> bool {
    true
}

fn default_memory_items() -> usize {
    50
}

impl Default for XencodeConfig {
    fn default() -> Self {
        Self {
            default_model: default_model(),
            active_theme: default_theme(),
            ollama_url: default_ollama_url(),
            llama_cpp_url: default_llama_cpp_url(),
            llama_cpp_model_path: default_llama_cpp_model_path(),
            llama_cpp_executable: String::new(),
            llama_cpp_args: Vec::new(),
            llama_cpp_temperature: None,
            llama_cpp_top_k: None,
            llama_cpp_min_p: None,
            llama_cpp_max_tokens: None,
            max_cache_size: default_cache_size(),
            response_timeout: default_timeout(),
            cache_enabled: true,
            memory_enabled: true,
            max_memory_items: default_memory_items(),
            api_keys: ApiKeys::default(),
        }
    }
}

/// Errors from config operations.
#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Json(serde_json::Error),
    NoHomeDir,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(source) => write!(f, "config I/O error: {source}"),
            ConfigError::Json(source) => write!(f, "config parse error: {source}"),
            ConfigError::NoHomeDir => write!(f, "could not determine home directory"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl XencodeConfig {
    /// Returns the path to the xencode config directory (`~/.xencode/`).
    pub fn config_dir() -> Result<PathBuf, ConfigError> {
        dirs::home_dir()
            .map(|home| home.join(".xencode"))
            .ok_or(ConfigError::NoHomeDir)
    }

    /// Returns the path to the config file (`~/.xencode/config.json`).
    pub fn config_path() -> Result<PathBuf, ConfigError> {
        Ok(Self::config_dir()?.join("config.json"))
    }

    /// Load configuration from `~/.xencode/config.json`.
    ///
    /// Returns defaults if the file doesn't exist.
    pub fn load() -> Result<Self, ConfigError> {
        let path = Self::config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&path).map_err(ConfigError::Io)?;
        serde_json::from_str(&content).map_err(ConfigError::Json)
    }

    /// Load configuration from a specific file path.
    pub fn load_from(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        serde_json::from_str(&content).map_err(ConfigError::Json)
    }

    /// Save configuration to `~/.xencode/config.json`.
    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ConfigError::Io)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(ConfigError::Json)?;
        std::fs::write(&path, json).map_err(ConfigError::Io)?;
        Ok(())
    }

    /// Save configuration to a specific file path.
    pub fn save_to(&self, path: impl AsRef<std::path::Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ConfigError::Io)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(ConfigError::Json)?;
        std::fs::write(path, json).map_err(ConfigError::Io)?;
        Ok(())
    }

    /// Serialize the config to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, ConfigError> {
        serde_json::to_string_pretty(self).map_err(ConfigError::Json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn temp_dir() -> PathBuf {
        // A process-wide counter, not a timestamp. Tests in one binary run in
        // parallel threads and can read the same nanosecond, which had them share
        // a directory and clobber each other's assertions.
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-config-test-{unique}"))
    }

    #[test]
    fn default_config_has_expected_values() {
        let config = XencodeConfig::default();
        assert_eq!(config.default_model, "qwen2.5:7b");
        assert_eq!(config.ollama_url, "http://localhost:11434");
        assert_eq!(config.llama_cpp_url, "http://localhost:8080");
        assert_eq!(config.llama_cpp_model_path, "");
        assert_eq!(config.llama_cpp_executable, "");
        assert!(config.llama_cpp_args.is_empty());
        assert!(config.llama_cpp_temperature.is_none());
        assert!(config.llama_cpp_top_k.is_none());
        assert!(config.llama_cpp_min_p.is_none());
        assert!(config.llama_cpp_max_tokens.is_none());
        assert_eq!(config.max_cache_size, 100);
        assert_eq!(config.response_timeout, 30);
        assert!(config.cache_enabled);
        assert!(config.memory_enabled);
        assert_eq!(config.max_memory_items, 50);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = temp_dir();
        let path = dir.join("config.json");

        let mut config = XencodeConfig {
            default_model: "llama3.1:8b".to_string(),
            ..XencodeConfig::default()
        };
        config.api_keys.openai_api_key = Some("sk-test-123".to_string());

        config.save_to(&path).unwrap();
        let loaded = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(loaded, config);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_nonexistent_returns_defaults() {
        // load_from on a nonexistent file should error, but load() falls back to default
        let dir = temp_dir();
        let path = dir.join("nonexistent.json");
        assert!(XencodeConfig::load_from(&path).is_err());
    }

    #[test]
    fn partial_json_gets_defaults_for_missing_fields() {
        let dir = temp_dir();
        let path = dir.join("partial.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, r#"{"default_model": "mistral:7b"}"#).unwrap();

        let config = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(config.default_model, "mistral:7b");
        // Other fields should have defaults
        assert_eq!(config.ollama_url, "http://localhost:11434");
        assert_eq!(config.max_cache_size, 100);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn to_json_produces_valid_json() {
        let config = XencodeConfig::default();
        let json = config.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["default_model"], "qwen2.5:7b");
    }
}
