use serde::{Deserialize, Serialize};

/// Metadata describing a plugin, typically read from a `plugin.yaml` or `plugin.json` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub entry_point: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
    pub xencode_version: String,
    #[serde(default)]
    pub permissions: Vec<String>,
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
        serde_json::to_string_pretty(self)
    }

    /// Check if this plugin is compatible with the given xencode version.
    pub fn is_compatible_with(&self, xencode_version: &str) -> bool {
        self.xencode_version == xencode_version || self.xencode_version == "*"
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
            "entry_point": "plugin.py",
            "dependencies": [],
            "xencode_version": "2.1.0",
            "permissions": ["read"]
        }"#;
        let manifest = PluginManifest::from_json(json).unwrap();
        assert_eq!(manifest.name, "test-plugin");
        assert_eq!(manifest.version, "1.0.0");
    }

    #[test]
    fn test_version_compatibility() {
        let manifest = PluginManifest {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            description: "".to_string(),
            author: "".to_string(),
            license: "MIT".to_string(),
            entry_point: "plugin.py".to_string(),
            dependencies: vec![],
            xencode_version: "2.1.0".to_string(),
            permissions: vec![],
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

impl Default for PluginManifest {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: "0.1.0".to_string(),
            description: String::new(),
            author: String::new(),
            license: "MIT".to_string(),
            entry_point: "plugin.py".to_string(),
            dependencies: Vec::new(),
            xencode_version: "*".to_string(),
            permissions: Vec::new(),
        }
    }
}
