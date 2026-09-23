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
    /// Bearer token for the custom OpenAI-compatible endpoint (`remote:` models).
    /// Optional: a local `llama-server`, LM Studio or vLLM usually has none.
    #[serde(default)]
    pub remote_api_key: Option<String>,
}

/// Google Colab bridge settings (Milestone K). Everything is opt-in by
/// default: `enabled` is false, `session` empty (the CLI picks a name when the
/// VM is created), local/remote ports default to the bridge plumbing, and the
/// runtime is llama.cpp. `model` is what `xencode colab up` installs on the
/// VM; `weights_source` chooses where it pulls the weights from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColabConfig {
    /// Master switch: the Colab provider is only reachable when true.
    #[serde(default)]
    pub enabled: bool,
    /// Colab session name. Empty = let `xencode colab up` create a session.
    #[serde(default)]
    pub session: String,
    /// Local port the SSH forward exposes the VM's OpenAI endpoint at.
    #[serde(default = "default_colab_local_port")]
    pub local_port: u16,
    /// Port the inference server listens on inside the VM. `0` = the
    /// runtime's native port (llama.cpp 18080 — Colab's own proxy holds 8080
    /// — ollama 11434); set only to override where the VM-side server binds.
    #[serde(default = "default_colab_remote_port")]
    pub remote_port: u16,
    /// Inference runtime started on the VM: "llama.cpp" or "ollama".
    #[serde(default = "default_colab_runtime")]
    pub runtime: String,
    /// Model the bridge installs on the VM: a Hugging Face GGUF repo id for
    /// llama.cpp, a tag for ollama. Empty = the process defaults still apply.
    #[serde(default)]
    pub model: String,
    /// Where the runtime fetches weights: "hf", "drive" or "gcs".
    #[serde(default = "default_colab_weights")]
    pub weights_source: String,
    /// GGUF quantization to serve (llama.cpp): a file-name fragment like
    /// "Q4_K_M". Empty = the bridge default. ollama tags carry their own.
    #[serde(default)]
    pub quant: String,
    /// Intended as "re-establish the forward when xencode starts"; recorded
    /// and round-tripped, but no code acts on it yet — bring-up stays explicit.
    #[serde(default)]
    pub auto_connect: bool,
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

    /// Body layout preset: "classic", "chat-first" or "zen".
    /// Unknown values fall back to "classic" at render time.
    #[serde(default = "default_layout")]
    pub layout: String,

    /// Draw panel borders with rounded corners.
    #[serde(default)]
    pub rounded_borders: bool,

    /// Show vertical scrollbars on scrollable lists (chat, explorer).
    #[serde(default = "default_true")]
    pub show_scrollbars: bool,

    /// Show a line-number gutter in the code editor.
    #[serde(default = "default_true")]
    pub show_line_numbers: bool,

    /// Agent tool-approval mode: "ask", "edit-allow" or "all-allow".
    /// Unknown values fall back to "ask" at decision time.
    #[serde(default = "default_agent_approval")]
    pub agent_approval: String,

    /// How many assistant→tool→assistant rounds one chat turn may take
    /// before tools are withdrawn and the model must answer in prose.
    #[serde(default = "default_agent_max_rounds")]
    pub agent_max_rounds: usize,

    /// Wall-clock seconds a foreground `run_command` may take before it is
    /// killed. Slow work belongs in `background_start`.
    #[serde(default = "default_agent_command_timeout")]
    pub agent_command_timeout: u64,

    /// Alternate models tried in order when the configured model fails before
    /// producing any output (I4-01). A different provider/model can fix what a
    /// permanent error on the primary cannot — a 404 `model not found`, a
    /// wrong key, an outage, a quota ceiling. Empty by default: no fallback,
    /// the error surfaces exactly as today.
    #[serde(default)]
    pub agent_fallback_models: Vec<String>,

    /// Ollama base URL.
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,

    /// llama.cpp server URL.
    #[serde(default = "default_llama_cpp_url")]
    pub llama_cpp_url: String,

    /// API root of a custom OpenAI-compatible server, addressed as
    /// `remote:<model>` — `/chat/completions` is appended to it. Empty means
    /// nothing is configured, and a `remote:` request says so rather than
    /// guessing a host. This is also how a Google Colab (or any SSH-forwarded)
    /// `llama-server` reaches the laptop as `http://127.0.0.1:<port>/v1`.
    #[serde(default)]
    pub remote_base_url: String,

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

    /// Model Context Protocol servers the agent may ask for tools from,
    /// keyed by the name that becomes the `mcp__<name>__<tool>` prefix.
    /// Empty by default: nothing is started, and nothing is reachable,
    /// until the user declares a server.
    #[serde(default)]
    pub mcp_servers: std::collections::BTreeMap<String, McpServer>,

    /// Wall-clock seconds an MCP request (handshake, tool list, tool call)
    /// may take before the server is considered unresponsive.
    #[serde(default = "default_mcp_timeout")]
    pub mcp_timeout: u64,

    /// Pre/post shell hooks around approved agent tool calls (I3-02). Empty by
    /// default: hooks only ever run on configured tools, only after the user
    /// approved the call, and never for a policy `Deny`.
    #[serde(default)]
    pub agent_hooks: AgentHooks,

    /// Named generation settings the TUI's Custom Models panel applies to the
    /// next turn. The panel edits these values and writes them back with
    /// [`XencodeConfig::save`]; nothing is seeded, so an empty list means the
    /// panel has nothing to show rather than something invented.
    #[serde(default)]
    pub model_profiles: Vec<ModelProfile>,

    /// Google Colab bridge settings. Opt-in: every field has a safe default,
    /// so a config that predates the block still loads with the bridge off.
    #[serde(default)]
    pub colab: ColabConfig,
}

/// One saved profile: a model id plus the sampling settings that reach a
/// request. `None` means "send nothing for this knob and let the server's own
/// default apply"; a value is passed to llama.cpp in the request body and, on
/// an applied profile, becomes the session's llama.cpp default for every later
/// turn. There is deliberately no `top_p` here because no provider path in this
/// workspace sends it.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ModelProfile {
    /// Label shown in the panel.
    pub name: String,
    /// Model id in exactly the form `default_model` takes — `ollama:qwen2.5:7b`,
    /// `openrouter:…`, `llamacpp:…` or a bare Ollama tag.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// One declared MCP server: a command we spawn and talk JSON-RPC to over its
/// stdin/stdout. Credentials for a server go in `env`, not in `args`, so they
/// do not end up in a shell history or a `ps` line.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct McpServer {
    /// Executable to spawn (resolved through `PATH` like a shell would).
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
}

/// Pre/post shell hooks (I3-02): commands run around approved agent tool
/// calls, matched per tool name first and then by `*` as the catch-all.
/// A `before` hook that exits non-zero vetoes the call (the tool never runs);
/// an `after` hook runs regardless of the call's outcome.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct AgentHooks {
    /// Tool name (or `*`) → `sh -c` command to run before the approved call.
    #[serde(default)]
    pub before: std::collections::BTreeMap<String, String>,
    /// Tool name (or `*`) → `sh -c` command to run after the call.
    #[serde(default)]
    pub after: std::collections::BTreeMap<String, String>,
}

fn default_model() -> String {
    "qwen2.5:7b".to_string()
}

fn default_theme() -> String {
    "ocean".to_string()
}

fn default_layout() -> String {
    "classic".to_string()
}

fn default_agent_approval() -> String {
    "ask".to_string()
}

fn default_agent_max_rounds() -> usize {
    16
}

fn default_agent_command_timeout() -> u64 {
    30
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

fn default_mcp_timeout() -> u64 {
    30
}

fn default_colab_local_port() -> u16 {
    18000
}

fn default_colab_remote_port() -> u16 {
    0
}

fn default_colab_runtime() -> String {
    "llama.cpp".to_string()
}

fn default_colab_weights() -> String {
    "hf".to_string()
}

impl Default for XencodeConfig {
    fn default() -> Self {
        Self {
            default_model: default_model(),
            active_theme: default_theme(),
            layout: default_layout(),
            rounded_borders: false,
            show_scrollbars: true,
            show_line_numbers: true,
            agent_approval: default_agent_approval(),
            agent_max_rounds: default_agent_max_rounds(),
            agent_command_timeout: default_agent_command_timeout(),
            agent_fallback_models: Vec::new(),
            ollama_url: default_ollama_url(),
            llama_cpp_url: default_llama_cpp_url(),
            remote_base_url: String::new(),
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
            mcp_servers: std::collections::BTreeMap::new(),
            mcp_timeout: default_mcp_timeout(),
            agent_hooks: AgentHooks::default(),
            model_profiles: Vec::new(),
            colab: ColabConfig::default(),
        }
    }
}

impl Default for ColabConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            session: String::new(),
            local_port: default_colab_local_port(),
            remote_port: default_colab_remote_port(),
            runtime: default_colab_runtime(),
            model: String::new(),
            weights_source: default_colab_weights(),
            quant: String::new(),
            auto_connect: false,
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
    /// Returns the path to the xencode config directory: `$XCODE_CONFIG_DIR`
    /// when set (tests and portable installs), else `~/.xencode/`.
    pub fn config_dir() -> Result<PathBuf, ConfigError> {
        if let Ok(dir) = std::env::var("XCODE_CONFIG_DIR") {
            if !dir.is_empty() {
                return Ok(PathBuf::from(dir));
            }
        }
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
    ///
    /// The write is atomic and the resulting file is owner-only: this config
    /// holds provider API keys as plain text, so a partly-written file or a
    /// world-readable one is a leak either way.
    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::config_path()?;
        self.save_to(&path)
    }

    /// Save configuration to a specific file path.
    pub fn save_to(&self, path: impl AsRef<std::path::Path>) -> Result<(), ConfigError> {
        let json = serde_json::to_string_pretty(self).map_err(ConfigError::Json)?;
        xencode_core_rs::write_atomic(path.as_ref(), json.as_bytes()).map_err(ConfigError::Io)?;
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
        assert_eq!(config.layout, "classic");
        assert!(!config.rounded_borders);
        assert!(config.show_scrollbars);
        assert!(config.show_line_numbers);
        assert_eq!(config.agent_approval, "ask");
        assert_eq!(config.agent_max_rounds, 16);
        assert_eq!(config.agent_command_timeout, 30);
        assert!(config.agent_fallback_models.is_empty());
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
        assert!(config.mcp_servers.is_empty());
        assert_eq!(config.mcp_timeout, 30);
        assert!(!config.colab.enabled);
        assert_eq!(config.colab.session, "");
        assert_eq!(config.colab.local_port, 18000);
        assert_eq!(config.colab.remote_port, 0);
        assert_eq!(config.colab.runtime, "llama.cpp");
        assert_eq!(config.colab.weights_source, "hf");
        assert_eq!(config.colab.model, "");
        assert!(config.colab.quant.is_empty(), "empty = bridge default");
        assert!(!config.colab.auto_connect);
    }

    /// The Colab block is opt-in: a config written before it existed must load
    /// with the bridge off, and a non-empty block must survive a round-trip.
    #[test]
    fn colab_block_defaults_off_and_round_trips() {
        let dir = temp_dir();
        let path = dir.join("colab.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, r#"{"default_model": "qwen2.5:7b"}"#).unwrap();
        let mut loaded = XencodeConfig::load_from(&path).unwrap();
        assert!(!loaded.colab.enabled);

        loaded.colab.enabled = true;
        loaded.colab.session = "xencode-t4a".to_string();
        loaded.colab.local_port = 19000;
        loaded.colab.remote_port = 8001;
        loaded.colab.runtime = "ollama".to_string();
        loaded.colab.model = "qwen3:8b".to_string();
        loaded.colab.weights_source = "gcs".to_string();
        loaded.colab.quant = "Q6_K".to_string();
        loaded.colab.auto_connect = true;
        loaded.save_to(&path).unwrap();

        let again = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(again.colab, loaded.colab);
        assert_eq!(again.colab.session, "xencode-t4a");
        assert_eq!(again.colab.runtime, "ollama");
        assert_eq!(again.colab.quant, "Q6_K");
        assert!(again.colab.auto_connect);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// The shipped example config is user-facing documentation, so it must
    /// load through the real loader and carry the same Colab defaults the
    /// bridge uses — an example that drifts teaches the wrong keys.
    #[test]
    fn shipped_example_loads_and_matches_colab_defaults() {
        let example = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../.xencode.example.json");
        let config = XencodeConfig::load_from(&example)
            .unwrap_or_else(|e| panic!("{} does not load: {e}", example.display()));
        assert_eq!(config.colab, ColabConfig::default());
        assert_eq!(config.colab.local_port, 18000);
        assert_eq!(config.colab.remote_port, 0);
        assert!(!config.colab.enabled, "the bridge is opt-in");
        assert!(
            config.remote_base_url.is_empty(),
            "the example must not point at an invented endpoint"
        );
    }

    /// A `remote:` endpoint is opt-in, so an absent URL must stay absent rather
    /// than point somewhere invented — and a config written before the fields
    /// existed must still load.
    #[test]
    fn remote_endpoint_defaults_empty_and_round_trips() {
        let config = XencodeConfig::default();
        assert!(config.remote_base_url.is_empty());
        assert!(config.api_keys.remote_api_key.is_none());

        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("remote.json");
        fs::write(&path, r#"{"default_model": "qwen2.5:7b"}"#).unwrap();
        let mut loaded = XencodeConfig::load_from(&path).unwrap();
        assert!(loaded.remote_base_url.is_empty());

        loaded.remote_base_url = "http://127.0.0.1:18080/v1".to_string();
        loaded.api_keys.remote_api_key = Some("runtime-proxy-token".to_string());
        loaded.save_to(&path).unwrap();
        let again = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(again, loaded);
        assert_eq!(again.remote_base_url, "http://127.0.0.1:18080/v1");
        assert_eq!(
            again.api_keys.remote_api_key.as_deref(),
            Some("runtime-proxy-token")
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    /// MCP servers are declared as a map in config.json, and a server with no
    /// `args`/`env` of its own must still parse — that is the common case.
    #[test]
    fn mcp_servers_parse_from_config_json_and_survive_a_roundtrip() {
        let dir = temp_dir();
        let path = dir.join("mcp.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            &path,
            r#"{
                "mcp_timeout": 5,
                "mcp_servers": {
                    "docs": {"command": "mcp-docs", "args": ["--root", "/docs"],
                              "env": {"DOCS_TOKEN": "t"}},
                    "bare": {"command": "npx"}
                }
            }"#,
        )
        .unwrap();

        let mut config = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(config.mcp_timeout, 5);
        assert_eq!(config.mcp_servers.len(), 2);
        let docs = &config.mcp_servers["docs"];
        assert_eq!(docs.command, "mcp-docs");
        assert_eq!(docs.args, vec!["--root".to_string(), "/docs".to_string()]);
        assert_eq!(docs.env.get("DOCS_TOKEN").map(String::as_str), Some("t"));
        assert_eq!(config.mcp_servers["bare"].args, Vec::<String>::new());

        // Saving must not lose the map, or a server disappears silently.
        config.mcp_timeout = 9;
        let out = dir.join("saved.json");
        config.save_to(&out).unwrap();
        let loaded = XencodeConfig::load_from(&out).unwrap();
        assert_eq!(loaded, config);
        assert_eq!(loaded.mcp_servers, config.mcp_servers);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Hooks ride along in config.json the way MCP servers do: a missing
    /// `agent_hooks` key stays empty (hooks only run when declared), and a
    /// declared one must survive a save/load roundtrip.
    #[test]
    fn agent_hooks_parse_and_roundtrip() {
        let dir = temp_dir();
        let path = dir.join("hooks.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            &path,
            r#"{
                "agent_hooks": {
                    "before": {"edit_file": "cargo fmt --check", "write_file": "cargo check"},
                    "after": {"*": "cargo test --quiet"}
                }
            }"#,
        )
        .unwrap();

        let mut config = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(config.agent_hooks.before["edit_file"], "cargo fmt --check");
        assert_eq!(config.agent_hooks.before["write_file"], "cargo check");
        assert_eq!(config.agent_hooks.after["*"], "cargo test --quiet");

        // Saving must not lose the hooks, or a gate silently disappears.
        config
            .agent_hooks
            .before
            .insert("run_command".to_string(), "true".to_string());
        let out = dir.join("saved.json");
        config.save_to(&out).unwrap();
        let loaded = XencodeConfig::load_from(&out).unwrap();
        assert_eq!(loaded, config);
        assert_eq!(loaded.agent_hooks, config.agent_hooks);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_default_config_has_no_hooks_at_all() {
        let config = XencodeConfig::default();
        assert!(config.agent_hooks.before.is_empty());
        assert!(config.agent_hooks.after.is_empty());
    }

    /// `agent_fallback_models` (I4-01) parses an ordered list and survives a
    /// save/load roundtrip; the default stays empty so a missing key behaves
    /// exactly as before the feature existed.
    #[test]
    fn agent_fallback_models_parse_and_roundtrip() {
        let dir = temp_dir();
        let path = dir.join("fallback.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            &path,
            r#"{"agent_fallback_models": ["qwen2.5:14b", "gemini:gemini-2.0-flash"]}"#,
        )
        .unwrap();

        let mut config = XencodeConfig::load_from(&path).unwrap();
        // Order is the chain's order: primary → fallback₁ → fallback₂.
        assert_eq!(
            config.agent_fallback_models,
            vec!["qwen2.5:14b", "gemini:gemini-2.0-flash"]
        );
        config
            .agent_fallback_models
            .push("ollama/deepseek-coder".to_string());

        let out = dir.join("saved.json");
        config.save_to(&out).unwrap();
        let loaded = XencodeConfig::load_from(&out).unwrap();
        assert_eq!(loaded, config);
        assert_eq!(
            loaded.agent_fallback_models,
            vec![
                "qwen2.5:14b",
                "gemini:gemini-2.0-flash",
                "ollama/deepseek-coder"
            ]
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = temp_dir();
        let path = dir.join("config.json");

        let mut config = XencodeConfig {
            default_model: "llama3.1:8b".to_string(),
            layout: "zen".to_string(),
            rounded_borders: true,
            show_scrollbars: false,
            show_line_numbers: false,
            agent_approval: "edit-allow".to_string(),
            agent_max_rounds: 24,
            agent_command_timeout: 5,
            ..XencodeConfig::default()
        };
        config.api_keys.openai_api_key = Some("sk-test-123".to_string());

        config.save_to(&path).unwrap();
        let loaded = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(loaded, config);

        fs::remove_dir_all(&dir).unwrap();
    }

    /// A profile only exists if it survives a save and comes back out of the
    /// file: the Custom Models panel writes through `save` and reads on startup.
    #[test]
    fn model_profiles_round_trip_and_omit_unset_knobs() {
        let dir = temp_dir();
        let path = dir.join("config.json");
        let config = XencodeConfig {
            model_profiles: vec![
                ModelProfile {
                    name: "tight".to_string(),
                    model: "ollama:qwen2.5:7b".to_string(),
                    temperature: Some(0.2),
                    max_tokens: Some(2048),
                },
                ModelProfile {
                    name: "server default".to_string(),
                    model: "llamacpp:gemma".to_string(),
                    temperature: None,
                    max_tokens: None,
                },
            ],
            ..XencodeConfig::default()
        };
        config.save_to(&path).unwrap();
        assert_eq!(XencodeConfig::load_from(&path).unwrap(), config);

        // A profile that sends no temperature must not write the key at all —
        // an explicit 0.0 would mean greedy decoding, not "unset".
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let second = &json["model_profiles"][1];
        assert!(
            second.get("temperature").is_none() && second.get("max_tokens").is_none(),
            "{second}"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_config_written_before_profiles_still_loads() {
        let dir = temp_dir();
        let path = dir.join("config.json");
        fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, r#"{"default_model":"qwen2.5:7b"}"#).unwrap();
        let config = XencodeConfig::load_from(&path).unwrap();
        assert!(config.model_profiles.is_empty());
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
        assert_eq!(config.layout, "classic");
        assert!(config.show_scrollbars);
        assert!(config.show_line_numbers);
        assert!(!config.rounded_borders);
        assert_eq!(config.agent_approval, "ask");
        assert_eq!(config.agent_max_rounds, 16);
        assert_eq!(config.agent_command_timeout, 30);
        assert_eq!(config.active_theme, "ocean");

        fs::remove_dir_all(&dir).unwrap();
    }

    /// The env override is process-global, so this is the one test that
    /// touches `XCODE_CONFIG_DIR`; no other test in this binary calls
    /// `config_dir()`.
    #[test]
    fn xcode_config_dir_env_overrides_the_default_location() {
        let dir = temp_dir();
        std::env::set_var("XCODE_CONFIG_DIR", &dir);
        let result = (|| {
            let path = XencodeConfig::config_path()?;
            assert_eq!(path, dir.join("config.json"));
            let config = XencodeConfig {
                layout: "chat-first".to_string(),
                ..XencodeConfig::default()
            };
            config.save()?;
            let loaded = XencodeConfig::load()?;
            assert_eq!(loaded, config);
            // An empty override is not an override: home wins again.
            std::env::set_var("XCODE_CONFIG_DIR", "");
            assert_eq!(
                XencodeConfig::config_dir()?,
                dirs::home_dir().unwrap().join(".xencode")
            );
            Ok::<(), ConfigError>(())
        })();
        std::env::remove_var("XCODE_CONFIG_DIR");
        result.unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn to_json_produces_valid_json() {
        let config = XencodeConfig::default();
        let json = config.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["default_model"], "qwen2.5:7b");
    }

    #[cfg(unix)]
    fn mode_of(path: &std::path::Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[cfg(unix)]
    fn set_mode(path: &std::path::Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn saving_config_leaves_it_readable_only_by_the_owner() {
        let dir = temp_dir();
        let path = dir.join("config.json");
        let mut config = XencodeConfig::default();
        config.api_keys.openai_api_key = Some("sk-test-plaintext-key".to_string());
        config.save_to(&path).unwrap();

        assert_eq!(mode_of(&path), 0o600);
        assert_eq!(
            XencodeConfig::load_from(&path)
                .unwrap()
                .api_keys
                .openai_api_key
                .as_deref(),
            Some("sk-test-plaintext-key")
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn saving_config_tightens_a_file_that_was_world_readable() {
        let dir = temp_dir();
        let path = dir.join("config.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, r#"{"default_model": "qwen2.5:7b"}"#).unwrap();
        set_mode(&path, 0o644);
        assert_eq!(mode_of(&path), 0o644);

        XencodeConfig::default().save_to(&path).unwrap();

        assert_eq!(mode_of(&path), 0o600);
        fs::remove_dir_all(&dir).unwrap();
    }
}
