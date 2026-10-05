use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::secrets::{SecretProblem, SecretProvider};

/// The shape of `config.json` that this binary writes.
///
/// Every rung of the ladder in [`migrate`] raises a file's version by one, so a
/// file at this number is exactly what the current code expects, and a file
/// above it was written by a newer xencode. A file carrying no such key —
/// every config written before the key existed — is
/// [`LEGACY_CONFIG_VERSION`], and adopting it is the first rung.
pub const CURRENT_CONFIG_VERSION: u32 = 1;

/// What a file with no `config_version` key is: written before the key existed.
pub const LEGACY_CONFIG_VERSION: u32 = 0;

/// How many `config.json.bak.<time>` files a save leaves behind. Bounded on
/// purpose: the settings panel saves on every change, so an unkept history would
/// fill the config directory with copies of a file that holds secrets.
pub const CONFIG_BACKUPS_KEPT: usize = 5;

/// The version a parsed file declares, read from the raw JSON rather than from
/// the struct — the whole question is what the *file* claims, before any of it
/// is dropped for being unknown to this binary.
fn declared_version(value: &serde_json::Value) -> u32 {
    value
        .get("config_version")
        .and_then(serde_json::Value::as_u64)
        .map(|found| found as u32)
        .unwrap_or(LEGACY_CONFIG_VERSION)
}

/// What kind of JSON value is at the top level, in words a person editing the
/// file can act on. Used by both refusals that come from a shape rather than a
/// version, so the read path and the write path describe the same file the same
/// way.
fn json_shape(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "true or false",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Bring a file written by version `from` up to [`CURRENT_CONFIG_VERSION`], one
/// rung at a time.
///
/// Each step is total: it takes the file as its predecessor wrote it and leaves
/// one the next step can read, touching only what that step is about. There is
/// no failure path by design — a file this binary cannot read at all is
/// rejected by [`XencodeConfig::load`] before the ladder runs, never partway
/// through it, so nothing half-migrated can be written back.
fn migrate(value: &mut serde_json::Value, from: u32) {
    // 0 -> 1: adopt versioning. A file written before the key existed gains it
    // and nothing else changes; this rung is what every existing config takes
    // on its first read by a versioned binary.
    if from < 1 {
        value["config_version"] = serde_json::Value::Number(CURRENT_CONFIG_VERSION.into());
    }
}

/// Stop a save that would replace a file this binary cannot read.
///
/// Checks the shape only, not the version: the caller asks that separately, and
/// the two refusals say different things. A missing file and an empty one pass —
/// there is nothing there to destroy — while any other bytes must at least be a
/// JSON object before they may be overwritten.
fn refuse_to_replace_unreadable(path: &std::path::Path) -> Result<(), ConfigError> {
    let existing = match std::fs::read(path) {
        Ok(found) => found,
        Err(problem) if problem.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(problem) => return Err(ConfigError::Io(problem)),
    };
    let existing = String::from_utf8_lossy(&existing);
    if existing.trim().is_empty() {
        return Ok(());
    }
    let value: serde_json::Value = match serde_json::from_str(&existing) {
        Ok(found) => found,
        Err(problem) => {
            return Err(ConfigError::Corrupt {
                path: path.to_path_buf(),
                problem: problem.to_string(),
            })
        }
    };
    if !value.is_object() {
        return Err(ConfigError::NotAConfig {
            path: path.to_path_buf(),
            found: json_shape(&value),
        });
    }
    Ok(())
}

/// Copy the file at `path` to `<name>.bak.<UTC time>` before `bytes` replaces it,
/// then trim the directory to the newest [`CONFIG_BACKUPS_KEPT`] copies.
///
/// Does nothing when there is no file yet, and when the file already holds these
/// exact bytes — the interface saves whenever a setting changes, and a key that
/// is put back where it was must not cost a copy. The copy is created owner-only,
/// the same rule the config file itself follows: a backup of a file holding API
/// keys holds those keys.
fn keep_backup_if_changed(path: &std::path::Path, bytes: &[u8]) -> Result<(), ConfigError> {
    let existing = match std::fs::read(path) {
        Ok(found) => found,
        Err(problem) if problem.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(problem) => return Err(ConfigError::Io(problem)),
    };
    if existing == bytes {
        return Ok(());
    }
    let name = path
        .file_name()
        .map(|found| found.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config.json".to_string());
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let stamp = utc_stamp(now.as_secs() as i64, now.subsec_nanos());
    // Same-second saves are possible, so a taken name gets a counter rather than
    // a silent overwrite of the older copy.
    let mut target = dir.join(format!("{name}.bak.{stamp}"));
    let mut attempt = 2;
    while target.exists() {
        target = dir.join(format!("{name}.bak.{stamp}-{attempt}"));
        attempt += 1;
    }
    write_owner_only(&target, &existing)?;
    prune_backups(dir, &name)
}

/// Write `bytes` to `path` with the owner-only mode applied as the file is
/// created, never afterwards: chmod after the write leaves a window in which the
/// copy is readable by everyone, which is the window SE-1 exists to close.
#[cfg(unix)]
fn write_owner_only(path: &std::path::Path, bytes: &[u8]) -> Result<(), ConfigError> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(ConfigError::Io)?;
    std::io::Write::write_all(&mut file, bytes).map_err(ConfigError::Io)
}

#[cfg(not(unix))]
fn write_owner_only(path: &std::path::Path, bytes: &[u8]) -> Result<(), ConfigError> {
    std::fs::write(path, bytes).map_err(ConfigError::Io)
}

/// Delete the oldest copies beyond the kept count. The timestamps are fixed
/// width and in UTC, so the names sort in the order they were written.
fn prune_backups(dir: &std::path::Path, name: &str) -> Result<(), ConfigError> {
    let prefix = format!("{name}.bak.");
    let mut copies: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(ConfigError::Io)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .map(|found| found.to_string_lossy().starts_with(&prefix))
                .unwrap_or(false)
        })
        .collect();
    copies.sort();
    while copies.len() > CONFIG_BACKUPS_KEPT {
        let oldest = copies.remove(0);
        std::fs::remove_file(&oldest).map_err(ConfigError::Io)?;
    }
    Ok(())
}

/// `20261003T095901.123456789Z` from seconds and the sub-second part of the same
/// instant. Howard Hinnant's `civil_from_days` in reverse of the
/// `days_from_civil` already used by the advice table, so no calendar library is
/// pulled in for a file name.
///
/// The nanoseconds are what make the naming safe to prune: a name that has been
/// deleted is never handed out again, so the newest copy can't be written under
/// the oldest copy's name and then trimmed as the oldest.
fn utc_stamp(secs: i64, nanos: u32) -> String {
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { year + 1 } else { year };
    format!(
        "{year:04}{m:02}{d:02}T{:02}{:02}{:02}.{nanos:09}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

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
    /// Bearer token for NVIDIA NIM (`nvidia:` models,
    /// `https://integrate.api.nvidia.com/v1`). Resolved by
    /// [`ApiKeys::secret`], which also honours `NVIDIA_NIM_API_KEY` and
    /// `API_KEY_NVIDIA` so the key can live outside this file.
    #[serde(default)]
    pub nvidia_api_key: Option<String>,
    /// Key for the Brave Search API (`https://api.search.brave.com`). It is not a
    /// model provider: this credential is only ever sent to Brave's search host,
    /// and only once `search_provider` says so.
    #[serde(default)]
    pub brave_api_key: Option<String>,
    /// Key for Tavily (`https://api.tavily.com`), the other paid search backend.
    /// Same rule as [`ApiKeys::brave_api_key`] — a search key, never a route.
    #[serde(default)]
    pub tavily_api_key: Option<String>,
}

impl ApiKeys {
    /// The credential in force for one provider: the stored value if the file
    /// holds one (running it if the value is a `command:` reference), otherwise
    /// the provider's environment variable.
    ///
    /// A reference that fails is a [`SecretProblem`] naming the command — an empty
    /// key and a broken keyring are different situations, and treating them alike
    /// would mean a provider answering "no key configured" when the real cause is
    /// that `secret-tool` is not on `PATH`.
    pub fn secret(&self, provider: SecretProvider) -> Result<Option<String>, SecretProblem> {
        crate::secrets::resolve(provider.stored(self), provider.env_vars())
    }

    /// Whether the provider has a credential, without reading it: no command
    /// reference is run, so this is safe to ask while a panel is drawing.
    pub fn has_secret(&self, provider: SecretProvider) -> bool {
        crate::secrets::is_present(provider.stored(self), provider.env_vars())
    }

    /// Where the credential comes from, for `config show` and `doctor`. Never the
    /// value itself.
    pub fn secret_source(&self, provider: SecretProvider) -> Option<String> {
        crate::secrets::describe(provider.stored(self), provider.env_vars())
    }

    /// Store a credential for a provider. Blank unsets it. A value beginning with
    /// [`SECRET_COMMAND_PREFIX`](crate::secrets::SECRET_COMMAND_PREFIX) is kept as
    /// written and treated as the command to read the secret from, so nothing but
    /// the reference lands on disk.
    pub fn set_secret(&mut self, provider: SecretProvider, value: &str) {
        let stored = (!value.trim().is_empty()).then(|| value.trim().to_string());
        match provider {
            SecretProvider::OpenAi => self.openai_api_key = stored,
            SecretProvider::OpenRouter => self.openrouter_api_key = stored,
            SecretProvider::Gemini => self.google_gemini_api_key = stored,
            SecretProvider::Qwen => self.qwen_api_key = stored,
            SecretProvider::Remote => self.remote_api_key = stored,
            SecretProvider::Nvidia => self.nvidia_api_key = stored,
            SecretProvider::Brave => self.brave_api_key = stored,
            SecretProvider::Tavily => self.tavily_api_key = stored,
        }
    }
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
    /// Which shape this file is written in. See [`CURRENT_CONFIG_VERSION`].
    ///
    /// A file with no key here was written before the key existed, and
    /// [`migrate`] stamps it — which is why the deserialising default below is
    /// [`LEGACY_CONFIG_VERSION`] while a config built in memory starts at
    /// [`CURRENT_CONFIG_VERSION`]. The two are meant to differ: one describes a
    /// file on disk, the other a config this binary made.
    #[serde(default = "legacy_config_version")]
    pub config_version: u32,

    /// The currently selected default model.
    #[serde(default = "default_model")]
    pub default_model: String,

    /// Active UI theme.
    #[serde(default = "default_theme")]
    pub active_theme: String,

    /// Body layout: "classic", "chat-first", "zen", or the name of a template
    /// in [`XencodeConfig::layout_templates`]. Unknown values fall back to
    /// "classic" at render time.
    #[serde(default = "default_layout")]
    pub layout: String,

    /// How much project context a run is allowed to spend: "low", "balanced" or
    /// "high". "auto" (the default) picks one from the memory this machine
    /// reports, and says which it picked and why — on the command line, and in
    /// the TUI's `/ctx`. The reason the setting exists at all is that the probe
    /// reads system memory, not the GPU a machine may or may not have, so it can
    /// be wrong about a box it has never seen; naming a profile here replaces the
    /// guess instead of arguing with it. An unrecognised value is reported and
    /// ignored rather than treated as a profile.
    #[serde(default = "default_hardware_profile")]
    pub hardware_profile: String,

    /// Draw panel borders with rounded corners.
    #[serde(default)]
    pub rounded_borders: bool,

    /// Show vertical scrollbars on scrollable lists (chat, explorer).
    #[serde(default = "default_true")]
    pub show_scrollbars: bool,

    /// Show a line-number gutter in the code editor.
    #[serde(default = "default_true")]
    pub show_line_numbers: bool,

    /// Read the mouse at all: the wheel scrolls the pane under it, a click
    /// focuses one, and a drag of a pane's edge resizes it.
    ///
    /// This is the escape hatch for what those three cost. A terminal that
    /// reports mouse events to the program stops doing its own click-and-drag
    /// text selection, so with this on, selecting transcript text with the
    /// mouse means holding whatever key your terminal reserves for that
    /// (usually `Shift`) — and turning this off hands the mouse back whole,
    /// taking the wheel and the click away with it.
    #[serde(default = "default_true")]
    pub mouse_capture: bool,

    /// Agent tool-approval mode: "ask", "edit-allow", "all-allow", "plan"
    /// (read-only, enforced) or "autonomous" (local writes free, off-box denied).
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

    /// How many times one chat turn may send a failing project check (the
    /// discovered `cargo test`/`cargo clippy` commands) back to the model for
    /// another repair attempt before the turn ends and the task is reported
    /// as incomplete. 0 turns the repair loop off entirely.
    #[serde(default = "default_agent_repair_max_iters")]
    pub agent_repair_max_iters: usize,

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

    /// How much an Ollama model is allowed to think before it answers: `off`
    /// for `think: false`, `on` for `think: true`, anything else — including
    /// `auto` and leaving this out — sends no `think` field at all and the
    /// model's own default decides.
    ///
    /// Unlike `llama_cpp_reasoning`, which is a launch flag because a running
    /// `llama-server` ignores the same fields in a request body, this one is
    /// sent per request and Ollama honours it there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ollama_reasoning: Option<String>,

    /// How long Ollama should hold a model loaded in memory after a request
    /// finishes (`"10m"`, `"30s"`, or `"0"` to unload it right away). Unset
    /// sends nothing, which leaves the decision to Ollama's own default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ollama_keep_alive: Option<String>,

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

    /// Where `llama_cpp_model_path` comes from when it is not on disk yet: an
    /// HTTPS URL of the GGUF itself, not of a page that links it. Bring-up
    /// fetches it into that path, resuming a partial file instead of
    /// restarting one. Empty means xencode has nowhere to get it and says so.
    #[serde(default)]
    pub llama_cpp_model_url: String,

    /// The SHA256 the file at `llama_cpp_model_path` must hash to. Empty means
    /// nothing is pinned: the download still reports what it received, but no
    /// claim is made that the bytes are the bytes anyone intended.
    ///
    /// The value has to come from outside the transfer to be worth anything —
    /// `xencode models advise` prints one from the shipped table, or copy it
    /// off the page you chose the model from.
    #[serde(default)]
    pub llama_cpp_model_sha256: String,

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

    /// llama.cpp sampling default: the seed the sampler draws from. Unset means
    /// nothing is sent and the server chooses one per request, so a run cannot be
    /// repeated. Set it — and `llama_cpp_temperature: 0.0` — when a result has to
    /// be reproducible; a negative value asks the server to keep choosing, the
    /// way `-1` does on its own command line. What a turn was asked to use is
    /// written to the project's `metrics.jsonl` by the TUI, so a claim of
    /// repeatability can be checked against the row rather than against memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_seed: Option<i64>,

    /// llama.cpp sampling default: max generated tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_max_tokens: Option<u32>,

    /// How much a self-started `llama-server` is allowed to think before it
    /// answers: `off` to tell a thinking model not to, or a token budget as text
    /// (`"256"`) to cut thinking short. Anything else — including `auto` and
    /// leaving this out — says nothing at launch and the model's own template
    /// decides, which is what every build did before this existed.
    ///
    /// This is a launch setting, not a request setting: a running server ignores
    /// the same fields in a request body. `reasoning_launch_args` in
    /// `xencode-models-rs` records what was measured, including the fact that a
    /// budget too small to finish a chain does not fail — it just answers from a
    /// half-finished plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_reasoning: Option<String>,

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

    /// What one conversation session may spend before the TUI warns, in
    /// micro-dollars ($1.00 = 1_000_000). Unset by default, because a price is
    /// only known if `pricing.json` says so — see `/cost`. This is a warning
    /// threshold, not a stop: nothing refuses a request over it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_budget_usd_micros: Option<u64>,

    /// What a kilowatt-hour costs where this machine runs, in cents, used to put
    /// a price on the electricity a local generation drew — the counter reading
    /// lives in `xencode_context_rs::power`. Set it from your own bill: a typical
    /// figure is 12 to 30 cents.
    ///
    /// Unset means the watt-hours are shown without a price beside them, which is
    /// a different statement from "this cost nothing": the machine drew the power
    /// either way, and the only unknown is what the meter charge was. A negative
    /// or not-a-number value is refused when it is written down, because it would
    /// price a turn below zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_cents_per_kwh: Option<f64>,

    /// What one calendar day may spend, in prompt plus completion tokens. The
    /// four `budget_*_per_day` settings are one mechanism seen from four angles,
    /// so see [`XencodeConfig::budget_minutes_per_day`] for what happens when one
    /// is crossed and for why none of them stops a turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens_per_day: Option<u64>,

    /// What one calendar day may draw at the wall, in watt-hours, counted from the
    /// machine's own energy counter. On a machine that publishes no counter this
    /// cap can never be reached, because there is no reading to reach it with, and
    /// xencode does not guess one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_energy_wh_per_day: Option<u64>,

    /// What one calendar day's tokens may cost at the rates in `pricing.json`, in
    /// micro-dollars ($1.00 = 1_000_000). Provider spend alone: the electricity a
    /// local turn drew is what [`XencodeConfig::budget_energy_wh_per_day`] weighs,
    /// and the two are different documents that are never added together. Models
    /// the price table does not know count as nothing against this cap, so an
    /// unpriced day cannot cross it — a floor is allowed to fire, not to report
    /// the day as cheap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_usd_micros_per_day: Option<u64>,

    /// What one calendar day's turns may take in the aggregate, in minutes —
    /// the wall-clock of the turns themselves, not the time the interface sat
    /// open with nothing asked of it.
    ///
    /// Crossing any of the four caps buys down the *next* turn instead of
    /// refusing anything: the hardware profile steps one rung down, which shrinks
    /// the context it fills, the files it retrieves and the size of each of them.
    /// A refusal landing between an edit and the check that was supposed to catch
    /// it is how these tools lose people's work, so the budget is never allowed
    /// to fire in the middle of a turn — it is read at the boundary, from the
    /// records the last turn wrote. Once the profile is at its lowest rung there
    /// is nothing further to give up, which is said once, and the day goes on
    /// being spent. Unset by default, which is not a cap at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_minutes_per_day: Option<u64>,

    /// Whether a prompt may be sent to an internet service at all.
    ///
    /// This is consent, not credentials: `api_keys` says who you are to a cloud
    /// provider, this says whether your conversation may reach one. They are
    /// deliberately separate — a configured key must not be read as permission
    /// to leave the machine, or the status-bar indicator would be describing
    /// something other than the rule actually in force. Off by default, and off
    /// for existing configurations that predate it, which is the point.
    #[serde(default)]
    pub allow_cloud_models: bool,

    /// Whether the agent may fetch a crate's documentation from `crates.io` or
    /// `docs.rs` when no copy of it is on this machine.
    ///
    /// Separate from [`XencodeConfig::allow_cloud_models`] because they guard
    /// different things: that one is about a *prompt* leaving, this one is
    /// about a text file arriving. Asking crates.io for the readme of a version
    /// named in `Cargo.lock` sends no code, no path and no question — but it is
    /// still a request out, and an agent that made one whenever a model got
    /// curious would be a program with an unplugged network switch. So the
    /// documented path stays offline, and this opens the fallback on purpose.
    #[serde(default)]
    pub allow_online_docs: bool,

    /// Whether a model missing from `pricing.json` may be priced from a fetched
    /// catalogue — OpenRouter's public model listing — instead of being reported
    /// as having no price.
    ///
    /// A third network switch, next to [`XencodeConfig::allow_online_docs`], and
    /// off like it. Nothing dials the listing on its own: the fetch happens when
    /// `xencode prices fetch` is run, and what that writes is a plain file under
    /// `.xencode/cache/` with the moment it was read in it. Turning this on only
    /// says that file may be consulted. It is worth separating even from the docs
    /// switch because a fetched price is a number that goes wrong quietly — the
    /// catalogue changes and nothing here is told — which is why every report that
    /// uses one also says how old it is, and why a rate you write by hand always
    /// outranks a rate that was looked up.
    #[serde(default)]
    pub price_lookup: bool,

    /// Whether the agent is offered `web_fetch` at all: one read of a page or
    /// an API response whose address the model picked.
    ///
    /// The fourth network switch, and the only one where the *model* chooses the
    /// address. [`XencodeConfig::allow_online_docs`] dials two named hosts for a
    /// file whose version a lock file already pinned; this one goes wherever a
    /// tool call says, which is why it is off by default, why offering it is a
    /// separate decision from every approval mode, and why each call still asks:
    /// a permission to fetch one page is not a permission to fetch the web.
    /// What it cannot reach even when on is written down in
    /// `xencode_analysis_rs::web`: private networks, the cloud's metadata
    /// address, and any host that resolves only to those.
    #[serde(default)]
    pub allow_web_fetch: bool,

    /// Which search engine the agent's `web_search` tool is allowed to ask, by its
    /// exact name — `none`, `wikipedia`, `searxng`, `brave` or `tavily`.
    ///
    /// `none` is the default, and it is not a disappointing default: it is the
    /// honest one. The keyless options people assume are free were measured and
    /// are gone — DuckDuckGo's keyless endpoints answer this machine a bot CAPTCHA
    /// or **410 Gone**, and public SearXNG instances answer a `format=json` request
    /// with an HTML page or **429**. So a search tool only exists when a person
    /// names an engine, and the `none` answer says what the alternatives were
    /// rather than silently failing every week.
    #[serde(default = "default_search_provider")]
    pub search_provider: String,

    /// The address of the SearXNG instance to ask, used only when
    /// [`XencodeConfig::search_provider`] is `searxng`. It has to be an instance
    /// you run, because that is the only kind that serves JSON. The same address
    /// rule as `web_fetch` applies to it: whatever this holds is resolved and
    /// refused before a connection, so it cannot point at a private network or the
    /// cloud's metadata service.
    #[serde(default)]
    pub search_searxng_url: String,

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

    /// Write down every model call an agent turn makes, so the turn can be run
    /// again. Off by default, and it has to stay that way for a person who has
    /// not asked for it: a recording holds the whole prompt and the full output
    /// of every tool, which is the most sensitive copy of a session this program
    /// can make. What is written goes to `.xencode/cache/sessions/` — the tree
    /// already kept out of version control — and `xencode replay <run-id>` reads
    /// it back.
    #[serde(default)]
    pub session_recording: bool,

    /// Run each `run_command`, `background_start` and shell hook inside a
    /// `bubblewrap` sandbox: the workspace and `~/.cargo` stay writable, the
    /// rest of the home (including `~/.ssh`) and the wider filesystem are not
    /// mounted in, and the network namespace is dropped unless the call asks
    /// for it. Off by default, because it changes what an approved command can
    /// reach and a build that fetches a dependency will not run with the net
    /// off — turning it on is a decision, not a surprise. There is deliberately
    /// no silent fallback: when this is on and `bwrap` is not installed, the
    /// command is refused with the reason rather than quietly run unsandboxed.
    #[serde(default)]
    pub run_command_sandbox: bool,

    /// Named generation settings the TUI's Custom Models panel applies to the
    /// next turn. The panel edits these values and writes them back with
    /// [`XencodeConfig::save`]; nothing is seeded, so an empty list means the
    /// panel has nothing to show rather than something invented.
    #[serde(default)]
    pub model_profiles: Vec<ModelProfile>,

    /// Let a profile take a turn on its own, when the prompt reads as the kind of
    /// work that profile is marked for — see [`ModelProfile::for_task`]. Off by
    /// default, because with it off the only thing that decides a turn's model is
    /// the one the user last chose, which is what every configuration written
    /// before this setting expects.
    ///
    /// This is not a classifier: the reading it acts on is the same whole-word
    /// rule the retrieval uses — `shape_of` in xencode-context-rs — so what can
    /// take a turn is only what that rule can tell apart.
    #[serde(default)]
    pub model_routing: bool,

    /// Body layouts the user authored, keyed by the name they are chosen by
    /// (V-5), alongside the three shipped presets `classic`, `chat-first` and
    /// `zen`. Each value is a layout tree: a `{"leaf": {"slot": …, "focus": …}}`
    /// or a `{"split": {"horizontal": …, "parts": [[child, share], …]}}` with
    /// shares `{"percent": n}`, `{"min": n}` or `{"length": n}`. Slots are
    /// `explorer`, `editor`, `chat`, `input` and `terminal`; foci are
    /// `explorer`, `editor` and `chat`.
    ///
    /// The value is left as raw JSON on purpose. This file is hand-edited, and
    /// a template that does not parse has to be refused *by name at the moment
    /// it is chosen* — not by failing the whole config, which would silently
    /// reset every other setting to its default. It also means a config
    /// written for a newer xencode (a node kind this build has never heard of)
    /// still opens here, the same promise `model_profiles` makes about an
    /// unknown task word. The tree is parsed and validated where it is used:
    /// `xencode_tui_rs::templates`.
    #[serde(default)]
    pub layout_templates: std::collections::BTreeMap<String, serde_json::Value>,

    /// Named views the user stored or hand-wrote (V-4), keyed by slot name:
    /// `Code`, `Chat`, `Terminal`, `Focus`, `Review`, `Split` — the six seeded
    /// names — and `7`, `8`, `9` for the unseeded chord slots. A value is the
    /// same layout tree a `layout_templates` entry holds, in the same
    /// vocabulary; `xencode_tui_rs::views` reads this map.
    ///
    /// Raw JSON for the same reason as templates, and one more: the map is
    /// written back out by `Ctrl+Shift+<digit>`, so a hand-added entry —
    /// or one from a future build that knows a node kind this one does not —
    /// must survive the round trip rather than fail the load. An empty map
    /// changes nothing: the seeded views live in code, and a view never
    /// becomes the only way to reach a pane.
    #[serde(default)]
    pub layout_views: std::collections::BTreeMap<String, serde_json::Value>,

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
    /// The kind of turn this profile is for, as one word: `bugfix` for a prompt
    /// that says something is broken, `general` for one that says nothing of the
    /// kind. Left out, the profile only ever applies by hand.
    ///
    /// `general` is wider than it sounds, and deliberately said so here: the rule
    /// notices words for broken code and is silent about everything else, so a
    /// profile marked `general` claims every turn that is not read as a bugfix —
    /// a rename, a question, an edit as large as any. That is what makes it
    /// suitable for a cheaper reading model and what makes it a second default,
    /// which is why it is not the value a configuration starts with.
    ///
    /// A word the reading does not have — `refactor`, `feature`, a misspelling —
    /// is kept as written and matches nothing, so the profile stays applicable by
    /// hand. Nothing is refused at load: a config that carries a word a newer
    /// version understands must still open this one. The Custom Models panel's
    /// `f` offers only the words that can match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub for_task: Option<String>,
}

/// One declared MCP server, reached one of two ways: a `command` we spawn and
/// talk JSON-RPC to over its stdin/stdout, or a hosted `url` we post each
/// message to. Exactly one of the two is set — a declaration with neither has
/// nothing to connect to, and one with both would be a coin flip.
///
/// Credentials for a spawned server go in `env`, and for an endpoint in
/// `headers`, rather than in `args`: an argument ends up in a shell history and
/// in `ps`, and a header value is never printed back.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct McpServer {
    /// Executable to spawn (resolved through `PATH` like a shell would).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
    /// `http://` or `https://` address of a hosted server: one POST per message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Sent with every request to `url` — where `authorization: Bearer …` goes.
    #[serde(default)]
    pub headers: std::collections::BTreeMap<String, String>,
}

impl McpServer {
    /// The command to spawn, when the declaration names one.
    pub fn spawn_command(&self) -> Option<&str> {
        self.command
            .as_deref()
            .filter(|command| !command.trim().is_empty())
    }

    /// The address to post to, when the declaration names one.
    pub fn endpoint_url(&self) -> Option<&str> {
        self.url.as_deref().filter(|url| !url.trim().is_empty())
    }

    /// What is wrong with a declaration that does not name exactly one way to
    /// reach the server, in words for the person who wrote it. `None` when the
    /// declaration is usable.
    pub fn misconfigured(&self) -> Option<String> {
        match (
            self.spawn_command().is_some(),
            self.endpoint_url().is_some(),
        ) {
            (true, false) => None,
            (false, true) => None,
            (false, false) => {
                Some("names neither a \"command\" to spawn nor a \"url\" to reach".to_string())
            }
            (true, true) => Some(
                "names both a \"command\" and a \"url\"; a server is one or the other".to_string(),
            ),
        }
    }
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

/// What a file that never carried `config_version` is told it is.
fn legacy_config_version() -> u32 {
    LEGACY_CONFIG_VERSION
}

fn default_theme() -> String {
    "ocean".to_string()
}

/// The default for [`XencodeConfig::search_provider`]. Spelled out as a function
/// rather than left to `#[serde(default)]` because the field is a `String`, and
/// the value serde would reach for on an absent key is `""`. That reads the same
/// way here — nothing is searched — but a file written back out should carry the
/// name a person can see and change, not an empty string.
fn default_search_provider() -> String {
    "none".to_string()
}

fn default_layout() -> String {
    "classic".to_string()
}

fn default_hardware_profile() -> String {
    "auto".to_string()
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

fn default_agent_repair_max_iters() -> usize {
    3
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
            config_version: CURRENT_CONFIG_VERSION,
            default_model: default_model(),
            active_theme: default_theme(),
            layout: default_layout(),
            hardware_profile: default_hardware_profile(),
            rounded_borders: false,
            show_scrollbars: true,
            show_line_numbers: true,
            mouse_capture: true,
            agent_approval: default_agent_approval(),
            agent_max_rounds: default_agent_max_rounds(),
            agent_command_timeout: default_agent_command_timeout(),
            agent_repair_max_iters: default_agent_repair_max_iters(),
            agent_fallback_models: Vec::new(),
            ollama_url: default_ollama_url(),
            ollama_reasoning: None,
            ollama_keep_alive: None,
            llama_cpp_url: default_llama_cpp_url(),
            remote_base_url: String::new(),
            llama_cpp_model_path: default_llama_cpp_model_path(),
            llama_cpp_executable: String::new(),
            llama_cpp_model_url: String::new(),
            llama_cpp_model_sha256: String::new(),
            llama_cpp_args: Vec::new(),
            llama_cpp_temperature: None,
            llama_cpp_top_k: None,
            llama_cpp_min_p: None,
            llama_cpp_seed: None,
            llama_cpp_max_tokens: None,
            llama_cpp_reasoning: None,
            max_cache_size: default_cache_size(),
            response_timeout: default_timeout(),
            cache_enabled: true,
            memory_enabled: true,
            max_memory_items: default_memory_items(),
            cost_budget_usd_micros: None,
            power_cents_per_kwh: None,
            budget_tokens_per_day: None,
            budget_energy_wh_per_day: None,
            budget_usd_micros_per_day: None,
            budget_minutes_per_day: None,
            allow_cloud_models: false,
            allow_online_docs: false,
            run_command_sandbox: false,
            price_lookup: false,
            allow_web_fetch: false,
            search_provider: default_search_provider(),
            search_searxng_url: String::new(),
            api_keys: ApiKeys::default(),
            mcp_servers: std::collections::BTreeMap::new(),
            mcp_timeout: default_mcp_timeout(),
            agent_hooks: AgentHooks::default(),
            session_recording: false,
            model_profiles: Vec::new(),
            model_routing: false,
            layout_templates: std::collections::BTreeMap::new(),
            layout_views: std::collections::BTreeMap::new(),
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
    /// The file is valid JSON but not an object of settings. Deserialising it
    /// anyway would hand back a fully defaulted config — serde reads a struct
    /// from a JSON array positionally, and with every field carrying a default
    /// an empty array is indistinguishable from "the user set nothing" — so the
    /// difference between a corrupt file and an empty one would be lost, and the
    /// next save would write defaults over it.
    NotAConfig {
        /// Which file, and what it held instead.
        path: PathBuf,
        found: &'static str,
    },
    /// The file is not readable JSON at all: truncated mid-write, hand-edited
    /// into a broken shape, or half of a merge conflict. Unlike
    /// [`ConfigError::Json`], this one names the file, because the read path has
    /// to be able to say *which* file to go and fix — and because a caller that
    /// treats this as "no config yet" would write defaults over a file holding
    /// someone's keys.
    Corrupt {
        /// Which file, and where in it the parser gave up.
        path: PathBuf,
        problem: String,
    },
    /// The file on disk declares a config shape newer than this binary writes.
    /// Reading it would guess at fields that do not exist yet, and writing it
    /// would delete whatever the newer xencode stored, so neither happens.
    NewerFile {
        /// Which file, named in the message because the person has to edit it.
        path: PathBuf,
        /// What that file claims its version is.
        found: u32,
        /// The highest version this binary understands
        /// ([`CURRENT_CONFIG_VERSION`]).
        known: u32,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(source) => write!(f, "config I/O error: {source}"),
            ConfigError::Json(source) => write!(f, "config parse error: {source}"),
            ConfigError::NoHomeDir => write!(f, "could not determine home directory"),
            ConfigError::NotAConfig { path, found } => write!(
                f,
                "{} holds a JSON {found}, where an object of settings was expected. That is not a \
                 configuration and it was not read — a file like this has no settings in it, and \
                 reading it as one would quietly hand back every default. Restore the file from a \
                 backup, or write a new one with `xencode config set <key> <value>`.",
                path.display()
            ),
            ConfigError::Corrupt { path, problem } => write!(
                f,
                "{} is not readable JSON: {problem}. Nothing was read from it and nothing was \
                 written to it, so whatever the file held is still there. Repair it by hand or \
                 restore a `config.json.bak.<time>` copy from beside it.",
                path.display()
            ),
            ConfigError::NewerFile { path, found, known } => write!(
                f,
                "{} declares config version {found}, and this xencode only knows versions up to \
                 {known}. It was not read, and nothing was written to it — an older binary \
                 cannot see the fields a newer one added and would drop them on save. Run the \
                 xencode that wrote this file, or point XCODE_CONFIG_DIR at a config this one \
                 can read.",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

impl XencodeConfig {
    /// Returns the directory the settings file lives in: `$XCODE_CONFIG_DIR`
    /// when set (tests and portable installs), else the location
    /// [`crate::paths`] resolves for settings — `~/.config/xencode`, or
    /// `~/.xencode` for a person who has never moved off it.
    pub fn config_dir() -> Result<PathBuf, ConfigError> {
        crate::paths::settings_dir()
    }

    /// Returns the path to the config file (`<settings dir>/config.json`).
    pub fn config_path() -> Result<PathBuf, ConfigError> {
        Ok(Self::config_dir()?.join("config.json"))
    }

    /// Load configuration from the file [`Self::config_path`] names.
    ///
    /// Returns defaults if the file doesn't exist.
    pub fn load() -> Result<Self, ConfigError> {
        let path = Self::config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&path).map_err(ConfigError::Io)?;
        Self::parse(&path, &content)
    }

    /// Load configuration from a specific file path.
    pub fn load_from(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        Self::parse(path, &content)
    }

    /// Turn file bytes into a config, after checking the version the file
    /// declares and running the ladder up to this binary's.
    ///
    /// The version is checked on the raw JSON because the fields that decide it
    /// are exactly the ones the struct below cannot see. A file whose top level
    /// is not an object is refused rather than deserialised positionally, and
    /// one that is not JSON at all is refused with the file named, because the
    /// alternative for the caller is `unwrap_or_default()` — defaults written
    /// back over the file on the next save.
    ///
    /// An empty or whitespace-only file is treated as "no settings yet", not as
    /// damage: it is what `touch` leaves, and there is nothing in it to lose. A
    /// file with bytes in it is a different case, and so is guarded at the write.
    fn parse(path: &std::path::Path, content: &str) -> Result<Self, ConfigError> {
        if content.trim().is_empty() {
            return Ok(Self::default());
        }
        let mut value: serde_json::Value = match serde_json::from_str(content) {
            Ok(found) => found,
            Err(problem) => {
                return Err(ConfigError::Corrupt {
                    path: path.to_path_buf(),
                    problem: problem.to_string(),
                })
            }
        };
        if !value.is_object() {
            return Err(ConfigError::NotAConfig {
                path: path.to_path_buf(),
                found: json_shape(&value),
            });
        }
        let found = declared_version(&value);
        if found > CURRENT_CONFIG_VERSION {
            return Err(ConfigError::NewerFile {
                path: path.to_path_buf(),
                found,
                known: CURRENT_CONFIG_VERSION,
            });
        }
        migrate(&mut value, found);
        serde_json::from_value(value).map_err(ConfigError::Json)
    }

    /// What the file at `path` says its config version is, without loading it.
    ///
    /// `None` means there is no readable object to ask — no file, an unreadable
    /// one, or one that is not a JSON object — which a caller reports as the
    /// shape it found rather than as a version. A readable file with no
    /// `config_version` key is [`LEGACY_CONFIG_VERSION`]: written before the key
    /// existed.
    pub fn version_of(path: impl AsRef<std::path::Path>) -> Option<u32> {
        let content = std::fs::read_to_string(path).ok()?;
        let value: serde_json::Value = serde_json::from_str(&content).ok()?;
        value.is_object().then(|| declared_version(&value))
    }

    /// Save configuration to the file [`Self::config_path`] names.
    ///
    /// The write is atomic and the resulting file is owner-only: this config
    /// holds provider API keys as plain text, so a partly-written file or a
    /// world-readable one is a leak either way.
    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::config_path()?;
        self.save_to(&path)
    }

    /// Save configuration to a specific file path.
    ///
    /// Two things about the file already there stop the write, both for the same
    /// reason: this config holds API keys, and a save is the moment they can be
    /// lost. The first is version — a newer one is refused, because plenty of
    /// call sites fall back to defaults when a load fails and without this check
    /// they would save those defaults over a config this binary could not read.
    /// The second is shape — a file that is not readable JSON, or not an object,
    /// is refused too, since it is exactly as unreadable to the writer as it was
    /// to the reader, and the fallback would silently reset every setting.
    ///
    /// What is about to be replaced is copied to a timestamped
    /// `config.json.bak.<time>` first, unless the bytes are already identical —
    /// the interface saves whenever a setting changes, and copying a file it has
    /// not changed would crowd out the copies of the ones it has.
    pub fn save_to(&self, path: impl AsRef<std::path::Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        refuse_to_replace_unreadable(path)?;
        self.force_save_to(path)
    }

    /// Save past the shape check, for the one command whose job is to discard
    /// what the file holds: `xencode config reset`. Refusing there would be a
    /// dead end — an unreadable config is exactly when someone reaches for reset,
    /// and the damaged bytes still go to a timestamped backup first. The version
    /// check is *not* skipped: a newer xencode's file is not ours to overwrite,
    /// reset included.
    pub fn force_save_to(&self, path: impl AsRef<std::path::Path>) -> Result<(), ConfigError> {
        let path = path.as_ref();
        if let Some(found) = Self::version_of(path) {
            if found > CURRENT_CONFIG_VERSION {
                return Err(ConfigError::NewerFile {
                    path: path.to_path_buf(),
                    found,
                    known: CURRENT_CONFIG_VERSION,
                });
            }
        }
        let json = serde_json::to_string_pretty(self).map_err(ConfigError::Json)?;
        keep_backup_if_changed(path, json.as_bytes())?;
        xencode_core_rs::write_atomic(path, json.as_bytes()).map_err(ConfigError::Io)?;
        Ok(())
    }

    /// The configuration as `xencode config show` prints it.
    ///
    /// Credentials are replaced by where they come from. Serialising the struct as
    /// it stands put every API key in plain view on the terminal, and that output
    /// is what people paste into a bug report, pipe into a file, or screenshot —
    /// including a key that had deliberately been kept out of `config.json` in an
    /// environment variable. A command reference is shown as stored, because the
    /// reference is not the secret and it is the thing to go and edit.
    pub fn to_json(&self) -> Result<String, ConfigError> {
        let mut value = serde_json::to_value(self).map_err(ConfigError::Json)?;
        if let Some(keys) = value
            .get_mut("api_keys")
            .and_then(|found| found.as_object_mut())
        {
            for provider in SecretProvider::ALL {
                let field = provider.json_field();
                let shown = self.api_keys.secret_source(provider);
                *keys
                    .entry(field.to_string())
                    .or_insert(serde_json::Value::Null) = match shown {
                    Some(label) => serde_json::Value::String(label),
                    None => serde_json::Value::Null,
                };
            }
        }
        serde_json::to_string_pretty(&value).map_err(ConfigError::Json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secrets::SECRET_COMMAND_PREFIX;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Mutex, MutexGuard};

    /// Tests touching `NVIDIA_NIM_API_KEY` must not run concurrently: the
    /// variable is process-global and two tests swapping it would read each
    /// other's values.
    static NIM_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_nim_env() -> MutexGuard<'static, ()> {
        NIM_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

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
    fn nvidia_key_prefers_config_over_environment_and_blank_is_unset() {
        let _guard = lock_nim_env();
        let previous = std::env::var_os("NVIDIA_NIM_API_KEY");
        let previous_alias = std::env::var_os("API_KEY_NVIDIA");
        std::env::remove_var("API_KEY_NVIDIA");
        std::env::set_var("NVIDIA_NIM_API_KEY", "env-key");
        // Environment alone resolves.
        assert_eq!(
            ApiKeys::default()
                .secret(SecretProvider::Nvidia)
                .unwrap()
                .as_deref(),
            Some("env-key")
        );
        // A configured key wins over the environment.
        let configured = ApiKeys {
            nvidia_api_key: Some("config-key".to_string()),
            ..Default::default()
        };
        assert_eq!(
            configured
                .secret(SecretProvider::Nvidia)
                .unwrap()
                .as_deref(),
            Some("config-key")
        );
        // Blank on either side counts as unset, never as a shadow.
        let blank_config = ApiKeys {
            nvidia_api_key: Some("   ".to_string()),
            ..Default::default()
        };
        assert_eq!(
            blank_config
                .secret(SecretProvider::Nvidia)
                .unwrap()
                .as_deref(),
            Some("env-key")
        );
        std::env::remove_var("NVIDIA_NIM_API_KEY");
        assert_eq!(
            ApiKeys::default().secret(SecretProvider::Nvidia).unwrap(),
            None
        );
        // The alias fills the same provider, so the naming rule works for NVIDIA
        // as well as its first-shipped variable name.
        std::env::set_var("API_KEY_NVIDIA", "alias-key");
        assert_eq!(
            ApiKeys::default()
                .secret(SecretProvider::Nvidia)
                .unwrap()
                .as_deref(),
            Some("alias-key")
        );
        match previous {
            Some(value) => std::env::set_var("NVIDIA_NIM_API_KEY", value),
            None => std::env::remove_var("NVIDIA_NIM_API_KEY"),
        }
        match previous_alias {
            Some(value) => std::env::set_var("API_KEY_NVIDIA", value),
            None => std::env::remove_var("API_KEY_NVIDIA"),
        }
    }

    #[test]
    fn config_show_names_where_each_credential_lives_and_never_the_value() {
        let _guard = lock_nim_env();
        let previous = std::env::var_os("API_KEY_OPENAI");
        std::env::remove_var("API_KEY_OPENAI");
        let config = XencodeConfig {
            api_keys: ApiKeys {
                openai_api_key: Some("sk-live-value-not-for-printing".to_string()),
                google_gemini_api_key: Some(
                    "command:secret-tool lookup service xencode".to_string(),
                ),
                ..Default::default()
            },
            ..Default::default()
        };
        let shown = config.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&shown).unwrap();
        assert_eq!(
            parsed["api_keys"]["openai_api_key"],
            serde_json::Value::String("set in config.json (value not shown)".to_string()),
            "a stored key is named, not printed: {shown}"
        );
        assert_eq!(
            parsed["api_keys"]["google_gemini_api_key"],
            serde_json::Value::String(
                "command reference — command:secret-tool lookup service xencode".to_string()
            ),
            "the reference is the thing to go edit, so it is shown: {shown}"
        );
        assert_eq!(
            parsed["api_keys"]["qwen_api_key"],
            serde_json::Value::Null,
            "an absent key stays absent"
        );
        assert!(
            !shown.contains("sk-live-value-not-for-printing"),
            "the value appears nowhere in what is printed: {shown}"
        );
        // Everything that is not a credential is printed as it stands.
        assert_eq!(parsed["default_model"], config.default_model);
        assert_eq!(parsed["config_version"], CURRENT_CONFIG_VERSION);

        // An environment-only credential is named by variable.
        std::env::set_var("API_KEY_OPENAI", "env-live-value-not-for-printing");
        let shown = XencodeConfig::default().to_json().unwrap();
        assert!(
            shown.contains("set in the environment as API_KEY_OPENAI"),
            "the tier is stated: {shown}"
        );
        assert!(
            !shown.contains("env-live-value-not-for-printing"),
            "and the value is not: {shown}"
        );
        match previous {
            Some(value) => std::env::set_var("API_KEY_OPENAI", value),
            None => std::env::remove_var("API_KEY_OPENAI"),
        }
    }

    #[test]
    fn what_a_panel_can_ask_while_it_draws_costs_no_command() {
        let dir =
            std::env::temp_dir().join(format!("xencode-config-presence-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("was-run");
        let keys = ApiKeys {
            openrouter_api_key: Some(format!(
                "{SECRET_COMMAND_PREFIX}/bin/sh -c 'touch {}; echo key'",
                marker.display()
            )),
            ..Default::default()
        };
        assert!(keys.has_secret(SecretProvider::OpenRouter));
        assert!(
            !marker.exists(),
            "presence must not run the helper a panel is only asking about"
        );
        assert_eq!(
            keys.secret_source(SecretProvider::OpenRouter)
                .unwrap()
                .split(" — ")
                .next()
                .unwrap(),
            "command reference"
        );
        let _ = fs::remove_dir_all(&dir);
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
        // Nothing is written down about a session until the user asks.
        assert!(!config.session_recording);
        // No shell command is fenced until the user asks; on it needs bwrap and
        // changes what an approved command can reach.
        assert!(!config.run_command_sandbox);
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
        assert!(config.llama_cpp_seed.is_none());
        assert!(config.llama_cpp_max_tokens.is_none());
        // A model that thinks is left alone until the user says otherwise.
        assert!(config.llama_cpp_reasoning.is_none());
        // The same for a model served by Ollama, where the ask rides in the
        // request instead of the launch flags, and for how long it stays loaded.
        assert!(config.ollama_reasoning.is_none());
        assert!(config.ollama_keep_alive.is_none());
        // No saved profile takes a turn on its own until the user turns that on.
        assert!(!config.model_routing);
        assert!(config.model_profiles.is_empty());
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
            !config.allow_cloud_models,
            "the example must show the posture the product ships with"
        );
        assert!(
            !config.allow_online_docs,
            "the example must show the posture the product ships with"
        );
        assert!(
            config.remote_base_url.is_empty(),
            "the example must not point at an invented endpoint"
        );
        // RS-2: an example that named an engine would teach the wrong posture —
        // the whole point of the setting is that nothing dials out until asked.
        assert_eq!(
            config.search_provider, "none",
            "the example must show search switched off"
        );
        assert!(config.search_searxng_url.is_empty());
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

    /// Cloud access is consent, not capability: it starts off, a config written
    /// before the key existed loads with it off, and an explicit `true` survives
    /// being saved and read back. Holding a provider key is not the same thing —
    /// that is what the separate `api_keys` block is for.
    #[test]
    fn cloud_models_are_disallowed_until_the_config_says_otherwise() {
        assert!(
            !XencodeConfig::default().allow_cloud_models,
            "cloud access must be asked for, not assumed"
        );

        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("egress.json");
        fs::write(&path, r#"{"default_model": "qwen2.5:7b"}"#).unwrap();
        let mut loaded = XencodeConfig::load_from(&path).unwrap();
        assert!(!loaded.allow_cloud_models);
        assert!(
            loaded.api_keys.openrouter_api_key.is_none(),
            "the test asserts on a config with no cloud key at all, so a default \
             of `true` cannot be mistaken for one"
        );

        loaded.allow_cloud_models = true;
        loaded.save_to(&path).unwrap();
        let again = XencodeConfig::load_from(&path).unwrap();
        assert!(again.allow_cloud_models);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Documentation fetches are their own consent, and the two switches do not
    /// imply each other: a machine that lets prompts to a cloud provider may
    /// still keep `read_docs` on local files, and one that fetches a readme has
    /// not thereby agreed to send a conversation off-box.
    #[test]
    fn online_docs_are_the_other_switch_and_stay_off_by_themselves() {
        let config = XencodeConfig::default();
        assert!(!config.allow_online_docs);
        assert!(!config.allow_cloud_models);

        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("docs.json");
        // A config from before the key existed loads, with it off.
        fs::write(&path, r#"{"default_model": "qwen2.5:7b"}"#).unwrap();
        let mut loaded = XencodeConfig::load_from(&path).unwrap();
        assert!(!loaded.allow_online_docs);

        loaded.allow_online_docs = true;
        loaded.save_to(&path).unwrap();
        let again = XencodeConfig::load_from(&path).unwrap();
        assert!(again.allow_online_docs);
        assert!(
            !again.allow_cloud_models,
            "one switch must not open the other"
        );

        loaded.allow_online_docs = false;
        loaded.allow_cloud_models = true;
        loaded.save_to(&path).unwrap();
        let both = XencodeConfig::load_from(&path).unwrap();
        assert!(both.allow_cloud_models);
        assert!(!both.allow_online_docs);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// MCP servers are declared as a map in config.json, and a server with no
    /// `args`/`env` of its own must still parse — that is the common case. A
    /// hosted one is declared by its url and the headers it is reached with.
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
                    "bare": {"command": "npx"},
                    "hosted": {"url": "https://mcp.example.com/v1/mcp",
                                "headers": {"authorization": "Bearer p-1"}}
                }
            }"#,
        )
        .unwrap();

        let mut config = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(config.mcp_timeout, 5);
        assert_eq!(config.mcp_servers.len(), 3);
        let docs = &config.mcp_servers["docs"];
        assert_eq!(docs.command.as_deref(), Some("mcp-docs"));
        assert_eq!(docs.args, vec!["--root".to_string(), "/docs".to_string()]);
        assert_eq!(docs.env.get("DOCS_TOKEN").map(String::as_str), Some("t"));
        assert_eq!(config.mcp_servers["bare"].args, Vec::<String>::new());
        let hosted = &config.mcp_servers["hosted"];
        assert_eq!(
            hosted.endpoint_url(),
            Some("https://mcp.example.com/v1/mcp")
        );
        assert_eq!(
            hosted.headers.get("authorization").map(String::as_str),
            Some("Bearer p-1")
        );
        // A url is not a command: nothing here would try to spawn `hosted`.
        assert_eq!(hosted.spawn_command(), None);

        // Saving must not lose the map, or a server disappears silently.
        config.mcp_timeout = 9;
        let out = dir.join("saved.json");
        config.save_to(&out).unwrap();
        let loaded = XencodeConfig::load_from(&out).unwrap();
        assert_eq!(loaded, config);
        assert_eq!(loaded.mcp_servers, config.mcp_servers);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A declaration that names no way to reach the server, or names both, is a
    /// mistake the reader has to be told about rather than one we guess at.
    #[test]
    fn a_server_must_name_exactly_one_of_command_and_url() {
        let usable = |server: McpServer| assert_eq!(server.misconfigured(), None, "{server:?}");
        let unusable = |server: McpServer| {
            assert!(server.misconfigured().is_some(), "{server:?} looks usable");
        };

        usable(McpServer {
            command: Some("mcp-docs".into()),
            ..Default::default()
        });
        usable(McpServer {
            url: Some("https://example.com/mcp".into()),
            ..Default::default()
        });
        unusable(McpServer::default());
        // Empty and whitespace-only are the same as absent: there is nothing to
        // spawn and nowhere to post.
        unusable(McpServer {
            command: Some("   ".into()),
            ..Default::default()
        });
        unusable(McpServer {
            command: Some("npx".into()),
            url: Some("https://example.com/mcp".into()),
            ..Default::default()
        });
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
                    for_task: Some("bugfix".to_string()),
                },
                ModelProfile {
                    name: "server default".to_string(),
                    model: "llamacpp:gemma".to_string(),
                    temperature: None,
                    max_tokens: None,
                    for_task: None,
                },
            ],
            model_routing: true,
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
            second.get("temperature").is_none()
                && second.get("max_tokens").is_none()
                && second.get("for_task").is_none(),
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
        assert!(!config.model_routing);
        // A file written before the mouse had a setting keeps the mouse: the
        // default is the behavior every existing install already has.
        assert!(config.mouse_capture);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_refusal_to_read_the_mouse_is_remembered() {
        // The setting exists so the terminal can be given its own text
        // selection back; that is only worth something if saying no once keeps
        // saying no across a save.
        let dir = temp_dir();
        let path = dir.join("config.json");
        fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, r#"{"mouse_capture":false}"#).unwrap();
        let config = XencodeConfig::load_from(&path).unwrap();
        assert!(!config.mouse_capture);
        config.save_to(&path).unwrap();
        assert!(!XencodeConfig::load_from(&path).unwrap().mouse_capture);
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
        // A config written before the profile was selectable says nothing about
        // it, which means "let the machine decide" rather than "no profile".
        assert_eq!(config.hardware_profile, "auto");
        // The same goes for what to ask of Ollama: a file written before these
        // existed asks nothing, and asks nothing in the same way an empty string
        // would not — it leaves the model's own default in charge.
        assert!(config.ollama_reasoning.is_none());
        assert!(config.ollama_keep_alive.is_none());
        // No saved profile takes a turn on its own until the user turns that on.
        assert!(!config.model_routing);
        assert!(config.model_profiles.is_empty());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_config_can_declare_layout_templates_and_a_broken_one_still_loads() {
        let dir = temp_dir();
        let path = dir.join("config.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            &path,
            r#"{
                "layout": "side",
                "model": "unused",
                "layout_templates": {
                    "side": {"split": {"horizontal": true, "parts": [
                        [{"leaf": {"slot": "editor", "focus": "editor"}}, {"percent": 70}],
                        [{"leaf": {"slot": "chat", "focus": "chat"}}, {"percent": 30}]
                    ]}},
                    "future": {"zones": [{"kind": "editor"}]}
                }
            }"#,
        )
        .unwrap();

        let config = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(config.layout, "side");
        assert_eq!(config.layout_templates.len(), 2);
        // A template this build cannot read stays exactly as written, and costs
        // nothing else in the file: every other setting is still loaded. That is
        // the whole reason these are raw JSON rather than a typed map.
        assert_eq!(
            config.layout_templates["future"]["zones"][0]["kind"],
            "editor"
        );
        assert_eq!(config.ollama_url, "http://localhost:11434");

        // And it round-trips: saving keeps both templates, future shape and all.
        config.save_to(&path).unwrap();
        let again = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(again, config);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_config_written_before_templates_existed_has_none() {
        let dir = temp_dir();
        let path = dir.join("config.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(&path, r#"{"default_model": "mistral:7b"}"#).unwrap();
        let config = XencodeConfig::load_from(&path).unwrap();
        assert!(config.layout_templates.is_empty());
        // An empty map is not written out as an empty block on the way back
        // either: nothing is invented for a config that declared nothing.
        let json = config.to_json().unwrap();
        assert!(json.contains("\"layout_templates\": {}"), "{json}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_config_can_declare_named_views_and_an_unknown_shape_still_loads() {
        let dir = temp_dir();
        let path = dir.join("config.json");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            &path,
            r#"{
                "layout_views": {
                    "Code": {"leaf": {"slot": "editor", "focus": "editor"}},
                    "9": {"zones": [{"kind": "editor"}]}
                }
            }"#,
        )
        .unwrap();

        let config = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(config.layout_views.len(), 2);
        // A view this build cannot read is carried as written and costs
        // nothing else — the same promise templates make.
        assert_eq!(config.layout_views["9"]["zones"][0]["kind"], "editor");
        config.save_to(&path).unwrap();
        let again = XencodeConfig::load_from(&path).unwrap();
        assert_eq!(again, config);
        fs::remove_dir_all(&dir).unwrap();
    }

    /// The env override is process-global, so this test holds the same lock the
    /// paths tests use; it cannot assume which directory the machine's home
    /// resolves to, because [`crate::paths`] answers that from what already
    /// exists on disk.
    #[test]
    fn xcode_config_dir_env_overrides_the_default_location() {
        let _guard = crate::paths::env_lock();
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
            // An empty override is not an override: the directory this machine's
            // layout resolves to wins again, whichever one that is.
            std::env::set_var("XCODE_CONFIG_DIR", "");
            assert_ne!(XencodeConfig::config_dir()?, dir);
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

    #[test]
    fn a_config_written_before_the_version_existed_still_loads_and_is_stamped() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        // A file exactly as an older xencode wrote it: no version key at all.
        let path = dir.join("legacy.json");
        fs::write(&path, r#"{"default_model":"ollama:qwen2.5:7b"}"#).unwrap();

        let loaded = XencodeConfig::load_from(&path).expect("a legacy config must still load");
        assert_eq!(loaded.default_model, "ollama:qwen2.5:7b");
        // The ladder took it to this binary's shape on the way in.
        assert_eq!(loaded.config_version, CURRENT_CONFIG_VERSION);
        // And the file itself still says what it is until something writes it.
        assert_eq!(
            XencodeConfig::version_of(&path),
            Some(LEGACY_CONFIG_VERSION),
            "reading a config must not mutate it"
        );

        // The first save is where the adoption lands.
        loaded.save_to(dir.join("saved.json")).unwrap();
        assert_eq!(
            XencodeConfig::version_of(dir.join("saved.json")),
            Some(CURRENT_CONFIG_VERSION)
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_file_from_a_newer_xencode_is_refused_with_both_numbers_named() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("newer.json");
        let bytes = br#"{"config_version":99,"default_model":"ollama:qwen3:8b","a_field_from_the_future":{"nested":[1,2,3]}}"#;
        fs::write(&path, bytes).unwrap();

        let error = XencodeConfig::load_from(&path)
            .expect_err("a config newer than this binary must not be read as if it were older");
        let ConfigError::NewerFile { found, known, .. } = &error else {
            panic!("expected a newer-file refusal, got {error:?}");
        };
        assert_eq!(*found, 99);
        assert_eq!(*known, CURRENT_CONFIG_VERSION);
        // The message has to be enough on its own: which file, what it claims,
        // what this binary knows, and what to do.
        let text = error.to_string();
        assert!(text.contains("newer.json"), "{text}");
        assert!(text.contains("99"), "{text}");
        assert!(text.contains(&CURRENT_CONFIG_VERSION.to_string()), "{text}");
        assert!(text.contains("XCODE_CONFIG_DIR"), "{text}");

        // Nothing was read, and nothing at all was written.
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn saving_does_not_overwrite_a_config_this_binary_cannot_read() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let bytes = br#"{"config_version":7,"default_model":"ollama:qwen3:8b"}"#;
        fs::write(&path, bytes).unwrap();

        // The point of checking at the write: a call site that fell back to
        // defaults after the failed load would otherwise save right over the
        // newer file.
        let error = XencodeConfig::default()
            .save_to(&path)
            .expect_err("a newer config on disk must not be replaced by defaults");
        assert!(matches!(error, ConfigError::NewerFile { found: 7, .. }));
        assert_eq!(fs::read(&path).unwrap(), bytes, "the file is untouched");

        // A legacy file is not protected, and is not harmed either: it is the
        // shape this binary is able to write.
        let legacy = dir.join("legacy.json");
        fs::write(&legacy, r#"{"default_model":"qwen2.5:7b"}"#).unwrap();
        XencodeConfig::default().save_to(&legacy).unwrap();
        assert_eq!(
            XencodeConfig::version_of(&legacy),
            Some(CURRENT_CONFIG_VERSION)
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_file_that_is_not_json_names_itself_and_says_where_it_broke() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        // The damage a hand edit makes: a trailing comma, and a file that is
        // otherwise full of the person's settings.
        let path = dir.join("config.json");
        let bytes = br#"{"default_model":"ollama:qwen2.5:7b","agent_approval":"ask",}"#;
        fs::write(&path, bytes).unwrap();

        let error = XencodeConfig::load_from(&path)
            .expect_err("bytes that are not JSON must not be read as if they were a config");
        let ConfigError::Corrupt { problem, .. } = &error else {
            panic!("expected a corrupt-file refusal, got {error:?}");
        };
        let text = error.to_string();
        assert!(text.contains("config.json"), "{text}");
        assert!(text.contains("trailing comma"), "{text}");
        assert!(
            !problem.is_empty(),
            "the parser's own words are the useful part"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn saving_refuses_to_replace_a_file_it_could_not_read() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let bytes = br#"{"default_model":"ollama:qwen2.5:7b","agent_approval":"ask",}"#;
        fs::write(&path, bytes).unwrap();

        // The whole defect this closes: the command loads, falls back to
        // defaults because the load failed, and saves — printing success while
        // it replaces someone's settings with the default block.
        let error = XencodeConfig::default()
            .save_to(&path)
            .expect_err("an unreadable config must not be replaced by defaults");
        assert!(
            matches!(error, ConfigError::Corrupt { .. }),
            "expected a corrupt-file refusal, got {error:?}"
        );
        assert_eq!(fs::read(&path).unwrap(), bytes, "the file is untouched");

        // Same guard for a file that is valid JSON of the wrong kind: reading it
        // back as a config would hand every setting the default value.
        let array = dir.join("array.json");
        fs::write(&array, b"[]").unwrap();
        let error = XencodeConfig::default()
            .save_to(&array)
            .expect_err("a JSON array is not a config to overwrite");
        assert!(
            matches!(error, ConfigError::NotAConfig { found: "array", .. }),
            "expected a not-a-config refusal, got {error:?}"
        );
        assert_eq!(fs::read(&array).unwrap(), b"[]");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_empty_config_is_no_settings_yet_rather_than_damage() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        // What `touch` leaves, or a create that got no bytes written. There is
        // nothing in it to protect, so refusing to save would be a dead end.
        for contents in ["", "   \n"] {
            let path = dir.join("empty.json");
            fs::write(&path, contents).unwrap();
            let loaded = XencodeConfig::load_from(&path).expect("an empty file reads as no config");
            assert_eq!(loaded, XencodeConfig::default());
            XencodeConfig::default().save_to(&path).unwrap();
            assert_eq!(
                XencodeConfig::version_of(&path),
                Some(CURRENT_CONFIG_VERSION),
                "the save went through"
            );
            fs::remove_file(&path).unwrap();
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn resetting_still_works_on_a_file_nothing_else_can_write_over() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let bytes = br#"{"default_model":"ollama:qwen2.5:7b","agent_approval":"ask",}"#;
        fs::write(&path, bytes).unwrap();

        // `xencode config reset` is the way out of an unreadable config, so it
        // is allowed past the shape check — and the broken bytes are kept, which
        // is what makes repairing them by hand possible afterwards.
        XencodeConfig::default().force_save_to(&path).unwrap();
        let after = fs::read(&path).unwrap();
        assert_ne!(after, bytes);
        assert!(String::from_utf8(after)
            .unwrap()
            .contains("\"agent_approval\""));

        let backups: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| {
                entry
                    .ok()
                    .map(|found| found.file_name().to_string_lossy().into_owned())
            })
            .filter(|name| name.starts_with("config.json.bak."))
            .collect();
        assert_eq!(backups.len(), 1, "the unreadable file was copied aside");
        let kept = fs::read(dir.join(&backups[0])).unwrap();
        assert_eq!(kept, bytes, "the copy holds what was there, damage and all");

        // What it refuses to skip is the version: a newer xencode's file is not
        // this one to overwrite, reset included.
        let newer = dir.join("newer.json");
        fs::write(&newer, br#"{"config_version":9}"#).unwrap();
        assert!(matches!(
            XencodeConfig::default().force_save_to(&newer),
            Err(ConfigError::NewerFile { found: 9, .. })
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn every_rung_of_the_ladder_is_walked_and_the_ladder_is_total() {
        // A minimal file at each version this binary has ever written must reach
        // the current shape, and a file already current must not be changed.
        for from in LEGACY_CONFIG_VERSION..=CURRENT_CONFIG_VERSION {
            let mut value: serde_json::Value = serde_json::json!({"default_model": "qwen2.5:7b"});
            if from > LEGACY_CONFIG_VERSION {
                value["config_version"] = serde_json::Value::Number(from.into());
            }
            migrate(&mut value, from);
            assert_eq!(
                declared_version(&value),
                CURRENT_CONFIG_VERSION,
                "version {from} did not reach the current shape"
            );
            let parsed: XencodeConfig =
                serde_json::from_value(value).expect("every rung produces a readable config");
            assert_eq!(parsed.default_model, "qwen2.5:7b");
            if from == CURRENT_CONFIG_VERSION {
                // A file already at this shape is left exactly as it is: the
                // ladder has no step to take, so it changes nothing.
                assert_eq!(parsed.config_version, from);
            }
        }
        assert_eq!(
            CURRENT_CONFIG_VERSION, 1,
            "the ladder above has one rung; a new version needs its own step"
        );
    }

    #[test]
    fn a_config_that_is_not_an_object_is_refused_rather_than_defaulted() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("not-an-object.json");
        fs::write(&path, "[]").unwrap();

        // Serde would read an array as a struct given positionally, so an empty
        // one becomes "every field at its default". That is how a corrupt file
        // turns into a config that looks like the user asked for nothing, and
        // then into defaults written over their real file.
        let error = XencodeConfig::load_from(&path)
            .expect_err("an array is not a config")
            .to_string();
        assert!(error.contains("holds a JSON array"), "{error}");
        assert!(error.contains("not-an-object.json"), "{error}");
        assert_eq!(XencodeConfig::version_of(&path), None);
        assert_eq!(
            XencodeConfig::version_of(dir.join("no-such-file.json")),
            None,
            "a file that is not there has no version"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_version_a_config_is_built_with_is_the_version_it_is_saved_as() {
        // The field's deserialising default is deliberately lower than this: an
        // absent key means a legacy file. What must never drift is that the
        // config this binary creates and the file it writes agree.
        assert_eq!(
            XencodeConfig::default().config_version,
            CURRENT_CONFIG_VERSION
        );
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        XencodeConfig::default().save_to(&path).unwrap();
        assert_eq!(
            XencodeConfig::version_of(&path),
            Some(XencodeConfig::default().config_version)
        );
        assert_eq!(
            XencodeConfig::load_from(&path).unwrap(),
            XencodeConfig::default()
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_stamp_is_the_utc_minute_the_epoch_second_falls_in() {
        // Numbers taken from an independent clock, not from this function.
        assert_eq!(utc_stamp(0, 0), "19700101T000000.000000000Z");
        assert_eq!(utc_stamp(951_782_400, 0), "20000229T000000.000000000Z");
        assert_eq!(utc_stamp(1_766_236_439, 0), "20251220T131359.000000000Z");
        assert_eq!(utc_stamp(1_791_021_541, 0), "20261003T095901.000000000Z");
        assert_eq!(utc_stamp(-86_401, 0), "19691230T235959.000000000Z");
        // The sub-second part is fixed width, so names sort in the order written.
        assert_eq!(utc_stamp(1_791_021_541, 7), "20261003T095901.000000007Z");
        assert_eq!(
            utc_stamp(1_791_021_541, 123_456_789),
            "20261003T095901.123456789Z"
        );
    }

    #[test]
    fn a_save_keeps_the_file_it_replaced_and_an_identical_save_keeps_nothing() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");

        let first = XencodeConfig {
            default_model: "qwen3:0.6b".to_string(),
            ..XencodeConfig::default()
        };
        first.save_to(&path).unwrap();
        // Nothing was replaced, so nothing is backed up.
        assert!(backups(&dir).is_empty(), "first save has no predecessor");

        let original = fs::read(&path).unwrap();
        let mut second = first.clone();
        second.default_model = "qwen3:4b".to_string();
        second.save_to(&path).unwrap();

        let copies = backups(&dir);
        assert_eq!(copies.len(), 1, "the replaced file is kept, once");
        assert_eq!(fs::read(&copies[0]).unwrap(), original);
        // The copy holds the same secrets as the file it came from.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&copies[0]).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "the backup is owner-only");
        }

        // Saving the same bytes again is not a change, so it adds no copy —
        // this is what stops the interface writing a copy every time it saves.
        second.save_to(&path).unwrap();
        second.save_to(&path).unwrap();
        assert_eq!(backups(&dir).len(), 1, "an unchanged save backs nothing up");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_backup_history_stays_bounded() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let mut before_last = String::new();
        for n in 0..(CONFIG_BACKUPS_KEPT as u32 + 4) {
            let config = XencodeConfig {
                default_model: format!("model-{n}"),
                ..XencodeConfig::default()
            };
            before_last = fs::read_to_string(&path).unwrap_or_default();
            config.save_to(&path).unwrap();
        }
        let copies = backups(&dir);
        assert_eq!(
            copies.len(),
            CONFIG_BACKUPS_KEPT,
            "the newest {} are kept",
            CONFIG_BACKUPS_KEPT
        );
        // What is kept is the newest, and the newest of them is the file as the
        // last save found it.
        let newest = fs::read_to_string(&copies[copies.len() - 1]).unwrap();
        assert_eq!(
            newest,
            before_last,
            "the newest copy is the state the last save replaced; copies were {:?}",
            copies
                .iter()
                .map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .collect::<Vec<_>>()
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Every `config.json.bak.*` in the directory, oldest name first.
    fn backups(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut found: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().starts_with("config.json.bak."))
                    .unwrap_or(false)
            })
            .collect();
        found.sort();
        found
    }
}
