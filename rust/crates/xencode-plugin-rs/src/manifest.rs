use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The only capabilities a manifest can ask for in this build, and the only
/// ones `PluginRuntime::load` will honour. A plugin has no code of its own, so
/// there is nothing else it could do: it can add text ahead of the agent's
/// system prompt (`prompt`) and declare hooks that run a shell command
/// (`hooks`). A manifest naming anything else is refused rather than ignored,
/// so a declared permission always means the host checked something.
pub const KNOWN_PERMISSIONS: &[&str] = &["prompt", "hooks"];

/// Hooks a plugin declares, in the same `tool name (or `*`) → shell command`
/// shape as the `agent_hooks` block in `config.json`. A plugin's entry wins
/// only where the config is silent; see `PluginRuntime::load`, which merges
/// them so the agent loop keeps exactly one hook path.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginHooks {
    #[serde(default)]
    pub before: BTreeMap<String, String>,
    #[serde(default)]
    pub after: BTreeMap<String, String>,
}

impl PluginHooks {
    pub fn is_empty(&self) -> bool {
        self.before.is_empty() && self.after.is_empty()
    }
}

/// Metadata and declared behaviour of a plugin, read from `plugin.json` (or
/// `manifest.json`) in a plugin directory.
///
/// This build cannot load executable plugin code — there is no dynamic linking
/// here — so a manifest is the whole plugin: the text it adds to the agent's
/// system prompt and the hooks it declares. Anything a manifest does not
/// declare does not happen, and `entry_point` is gone with the Python runtime
/// it named.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
    /// Versions this plugin accepts; `*` (the default) accepts any.
    #[serde(default = "any_version")]
    pub xencode_version: String,
    /// Capabilities this plugin asks the host to grant, checked at load. A
    /// manifest that adds a prompt prefix must name `prompt`; one that declares
    /// hooks must name `hooks`. Anything it uses without asking — or any name
    /// outside `KNOWN_PERMISSIONS` — refuses the load. See `PluginRuntime::load`.
    #[serde(default)]
    pub permissions: Vec<String>,
    /// Prepended to the agent's system prompt by every turn started after load.
    #[serde(default)]
    pub prompt_prefix: String,
    #[serde(default)]
    pub hooks: PluginHooks,
}

fn any_version() -> String {
    "*".to_string()
}

impl PluginManifest {
    /// Load a manifest from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Load a manifest from a JSON file.
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Serialize this manifest to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self)
    }

    /// Check if this plugin is compatible with the given xencode version.
    pub fn is_compatible_with(&self, xencode_version: &str) -> bool {
        self.xencode_version == xencode_version || self.xencode_version == "*"
    }

    /// Whether the plugin adds text ahead of the agent's system prompt. Matches
    /// the trim rule `PluginRuntime::load` uses to decide a prefix is present.
    pub fn uses_prompt(&self) -> bool {
        !self.prompt_prefix.trim().is_empty()
    }

    /// Whether the plugin declares any hook — each of which runs a shell command
    /// in the workspace, which is why it needs its own permission.
    pub fn uses_hooks(&self) -> bool {
        !self.hooks.before.is_empty() || !self.hooks.after.is_empty()
    }

    /// The capabilities the plugin exercises but did not declare. Non-empty
    /// means the plugin would change agent behaviour it never asked to be
    /// allowed to change, so the load is refused.
    pub fn undeclared_permissions(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.uses_prompt() && !self.permissions.iter().any(|p| p == "prompt") {
            missing.push("prompt");
        }
        if self.uses_hooks() && !self.permissions.iter().any(|p| p == "hooks") {
            missing.push("hooks");
        }
        missing
    }

    /// Declared permission names the host has no meaning for. A plugin asking
    /// for `network` in a build that cannot grant it is told so and refused,
    /// rather than having the claim read and dropped.
    pub fn unknown_permissions(&self) -> Vec<&str> {
        self.permissions
            .iter()
            .map(|p| p.as_str())
            .filter(|p| !KNOWN_PERMISSIONS.contains(p))
            .collect()
    }
}

impl Default for PluginManifest {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: "0.1.0".to_string(),
            description: String::new(),
            author: String::new(),
            license: "MIT".to_string(),
            dependencies: Vec::new(),
            xencode_version: "*".to_string(),
            permissions: Vec::new(),
            prompt_prefix: String::new(),
            hooks: PluginHooks::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_from_json() {
        let json = r#"{
            "name": "test-plugin",
            "version": "1.0.0",
            "description": "A test plugin",
            "author": "Test Author",
            "license": "MIT",
            "dependencies": [],
            "xencode_version": "2.1.0",
            "permissions": ["prompt"]
        }"#;
        let manifest = PluginManifest::from_json(json).unwrap();
        assert_eq!(manifest.name, "test-plugin");
        assert_eq!(manifest.version, "1.0.0");
        assert!(manifest.prompt_prefix.is_empty());
        assert!(manifest.hooks.is_empty());
    }

    #[test]
    fn using_a_contribution_without_its_permission_is_reported_missing() {
        let mut m = PluginManifest {
            name: "guardrails".to_string(),
            version: "1".to_string(),
            prompt_prefix: "Read the tests first.".to_string(),
            ..Default::default()
        };
        m.hooks
            .after
            .insert("*".to_string(), "cargo fmt".to_string());
        // Both capabilities are exercised but neither was asked for.
        assert!(m.uses_prompt());
        assert!(m.uses_hooks());
        assert_eq!(m.undeclared_permissions(), vec!["prompt", "hooks"]);
        // Declaring one clears only that one.
        m.permissions = vec!["hooks".to_string()];
        assert_eq!(m.undeclared_permissions(), vec!["prompt"]);
        // Declaring both clears the field.
        m.permissions = vec!["prompt".to_string(), "hooks".to_string()];
        assert!(m.undeclared_permissions().is_empty());
    }

    #[test]
    fn an_inert_manifest_asks_for_nothing_and_misses_nothing() {
        let m = PluginManifest {
            name: "noop".to_string(),
            version: "1".to_string(),
            ..Default::default()
        };
        assert!(!m.uses_prompt());
        assert!(!m.uses_hooks());
        assert!(m.undeclared_permissions().is_empty());
        assert!(m.unknown_permissions().is_empty());
    }

    #[test]
    fn a_permission_name_the_host_does_not_know_is_flagged() {
        let m = PluginManifest {
            name: "guardrails".to_string(),
            version: "1".to_string(),
            permissions: vec!["prompt".to_string(), "network".to_string()],
            ..Default::default()
        };
        assert_eq!(m.unknown_permissions(), vec!["network"]);
    }

    /// A manifest that declares nothing but a name and a version is still a
    /// loadable plugin — it just changes nothing.
    #[test]
    fn a_minimal_manifest_loads() {
        let manifest =
            PluginManifest::from_json(r#"{ "name": "noop", "version": "1.0.0" }"#).unwrap();
        assert_eq!(manifest.xencode_version, "*");
        assert!(manifest.hooks.before.is_empty());
    }

    #[test]
    fn declared_prefix_and_hooks_round_trip() {
        let json = r#"{
            "name": "guardrails",
            "version": "1.0.0",
            "prompt_prefix": "Always run the tests before answering.",
            "hooks": {
                "before": { "write_file": "cargo check" },
                "after": { "*": "cargo fmt" }
            }
        }"#;
        let manifest = PluginManifest::from_json(json).unwrap();
        assert_eq!(
            manifest.prompt_prefix,
            "Always run the tests before answering."
        );
        assert_eq!(manifest.hooks.before["write_file"], "cargo check");
        assert_eq!(manifest.hooks.after["*"], "cargo fmt");
        let back = PluginManifest::from_json(&manifest.to_json().unwrap()).unwrap();
        assert_eq!(back, manifest);
    }

    #[test]
    fn test_version_compatibility() {
        let manifest = PluginManifest {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            xencode_version: "2.1.0".to_string(),
            ..Default::default()
        };
        assert!(manifest.is_compatible_with("2.1.0"));
        assert!(!manifest.is_compatible_with("2.0.0"));
    }

    #[test]
    fn test_wildcard_compatibility() {
        let manifest = PluginManifest {
            xencode_version: "*".to_string(),
            ..Default::default()
        };
        assert!(manifest.is_compatible_with("any-version"));
    }
}
