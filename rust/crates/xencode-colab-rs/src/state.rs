//! Persistent bridge state (`~/.xencode/colab.json`): what the lifecycle
//! commands brought up, so `status`/`down` know what to report and tear down.
//! Pure JSON + filesystem — `unsafe` is forbidden here.
#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use xencode_config_rs::XencodeConfig;

/// Filename of the state file inside the xencode config dir.
pub const STATE_FILENAME: &str = "colab.json";

/// A recorded bridge: the forward pid(s), the session it points at, and what
/// the forward exposes. All optional — a partial or hand-edited file must load
/// rather than panic (`up` writes the full set; nothing requires it back).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColabState {
    /// Session name of the bridged Colab runtime.
    pub session: Option<String>,
    /// PID of the local `ssh -L` forward process.
    pub forward_pid: Option<u32>,
    /// PID of an explicit keep-alive process, when one is spawned.
    pub keepalive_pid: Option<u32>,
    /// Local port the forward exposes.
    pub local_port: Option<u16>,
    /// Port the inference server listens on inside the VM.
    pub remote_port: Option<u16>,
    /// Runtime started on the VM ("llama.cpp" | "ollama").
    pub runtime: Option<String>,
    /// Model id served on the VM.
    pub model: Option<String>,
    /// When the bridge was brought up (RFC3339, local).
    pub started_at: Option<String>,
    /// The URL requests flow through (what `remote_url` is set to).
    pub url: Option<String>,
}

impl ColabState {
    /// Path of the state file: `$XCODE_CONFIG_DIR` when set, else
    /// `~/.xencode/`.
    pub fn state_path() -> Result<PathBuf, String> {
        let dir = XencodeConfig::config_dir().map_err(|e| e.to_string())?;
        Ok(dir.join(STATE_FILENAME))
    }

    /// Load the state file. `Ok(None)` when it does not exist or is empty; a
    /// malformed file is an error (we never silently discard the forward's
    /// recorded pid).
    pub fn load() -> Result<Option<Self>, String> {
        let path = Self::state_path()?;
        let raw = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("could not read {}: {e}", path.display())),
        };
        if raw.trim().is_empty() {
            return Ok(None);
        }
        serde_json::from_str(&raw)
            .map(Some)
            .map_err(|e| format!("could not parse {}: {e}", path.display()))
    }

    /// Write the state file (creating the config dir) via the shared atomic
    /// helper, so a crash mid-write never leaves a torn JSON and the file
    /// holding the SSH key path stays owner-readable.
    pub fn save(&self) -> Result<(), String> {
        let path = Self::state_path()?;
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize state: {e}"))?;
        xencode_core_rs::write_atomic(&path, json.as_bytes())
            .map_err(|e| format!("could not write {}: {e}", path.display()))
    }

    /// Remove the state file. Missing file is not an error.
    pub fn clear() -> Result<(), String> {
        let path = Self::state_path()?;
        match fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("could not remove {}: {e}", path.display())),
        }
    }

    /// True when a non-empty forward pid is recorded. Distinct from "the
    /// bridge works" — probes the pid only, status does the URL check.
    pub fn has_forward(&self) -> bool {
        self.forward_pid.is_some()
    }
}

/// Convenience wrappers kept symmetrical with the module themes.
/// Load [`ColabState`] (see [`ColabState::load`]).
pub fn load_state() -> Result<Option<ColabState>, String> {
    ColabState::load()
}

/// Save [`ColabState`] (see [`ColabState::save`]).
pub fn save_state(state: &ColabState) -> Result<(), String> {
    state.save()
}

/// Remove the state file (see [`ColabState::clear`]).
pub fn remove_state() -> Result<(), String> {
    ColabState::clear()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{temp_dir, with_env};

    #[test]
    fn missing_state_file_loads_as_none() {
        let xcode_dir = temp_dir("state-missing");
        let _g = with_env(&std::env::temp_dir(), &xcode_dir);
        let state = ColabState::load().expect("no file -> None, not error");
        assert!(state.is_none());
    }

    #[test]
    fn state_round_trips_through_the_config_dir() {
        let xcode_dir = temp_dir("state-rt");
        let _g = with_env(&std::env::temp_dir(), &xcode_dir);

        let state = ColabState {
            session: Some("xencode-t4".to_string()),
            forward_pid: Some(4242),
            keepalive_pid: None,
            local_port: Some(18000),
            remote_port: Some(8000),
            runtime: Some("llama.cpp".to_string()),
            model: Some("qwen3:8b".to_string()),
            started_at: Some("2026-09-23T10:00:00Z".to_string()),
            url: Some("http://127.0.0.1:18000/v1".to_string()),
        };
        state.save().expect("save state");

        let path = ColabState::state_path().expect("state path");
        assert!(path.is_file(), "state file written under config dir");
        assert_eq!(
            path.file_name().map(|n| n.to_string_lossy().to_string()),
            Some(STATE_FILENAME.to_string())
        );

        let loaded = ColabState::load().expect("load state");
        assert_eq!(loaded, Some(state));
        assert!(loaded.unwrap().has_forward());
    }

    #[test]
    fn clear_removes_even_when_already_absent() {
        let xcode_dir = temp_dir("state-clear");
        let _g = with_env(&std::env::temp_dir(), &xcode_dir);
        ColabState::save(&ColabState::default()).expect("save");
        ColabState::clear().expect("clear");
        assert!(!ColabState::state_path().unwrap().exists());
        ColabState::clear().expect("clear again is a no-op");
    }

    #[test]
    fn malformed_state_is_an_error_not_silent_reset() {
        let xcode_dir = temp_dir("state-malformed");
        let _g = with_env(&std::env::temp_dir(), &xcode_dir);
        fs::write(ColabState::state_path().unwrap(), "{ not json").expect("write junk");
        let err = ColabState::load().expect_err("junk -> error");
        assert!(
            err.contains("could not parse"),
            "error names the file: {err}"
        );
    }
}
