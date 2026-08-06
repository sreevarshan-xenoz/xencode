use std::path::PathBuf;

use crate::manifest::PluginManifest;
use crate::plugin_trait::PluginError;

/// Discovers and loads plugins from a directory.
pub struct PluginRegistry {
    plugin_dir: PathBuf,
}

impl PluginRegistry {
    pub fn new(plugin_dir: PathBuf) -> Self {
        Self { plugin_dir }
    }

    /// Discover all plugin manifests in the plugin directory.
    /// Looks for `plugin.json` or `plugin.yaml` files in subdirectories.
    pub fn discover(&self) -> Vec<PluginManifest> {
        let mut manifests = Vec::new();
        if !self.plugin_dir.exists() {
            return manifests;
        }

        let entries = match std::fs::read_dir(&self.plugin_dir) {
            Ok(entries) => entries,
            Err(_) => return manifests,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            // Look for manifest files
            for manifest_name in &["plugin.json", "plugin.yaml", "manifest.json"] {
                let manifest_path = path.join(manifest_name);
                if manifest_path.exists() {
                    if let Ok(manifest) = PluginManifest::from_file(&manifest_path) {
                        manifests.push(manifest);
                    }
                }
            }
        }

        manifests
    }

    /// Load a plugin from its manifest.
    /// For now returns the manifest; actual plugin loading requires dynamic linking.
    pub fn load_plugin(&self, manifest: &PluginManifest) -> Result<PluginManifest, PluginError> {
        let plugin_path = self.plugin_dir.join(&manifest.name);
        if !plugin_path.exists() {
            return Err(PluginError::NotFound(manifest.name.clone()));
        }
        Ok(manifest.clone())
    }

    /// Get the full path to a plugin's directory.
    pub fn plugin_path(&self, name: &str) -> PathBuf {
        self.plugin_dir.join(name)
    }

    /// Check if a plugin exists in the registry.
    pub fn has_plugin(&self, name: &str) -> bool {
        self.plugin_dir.join(name).exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_discover_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = PluginRegistry::new(tmp.path().to_path_buf());
        let manifests = registry.discover();
        assert!(manifests.is_empty());
    }

    #[test]
    fn test_discover_with_plugin() {
        let tmp = tempfile::tempdir().unwrap();
        let plugin_dir = tmp.path().join("my-plugin");
        fs::create_dir_all(&plugin_dir).unwrap();

        let manifest = PluginManifest {
            name: "my-plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Test".to_string(),
            author: "Author".to_string(),
            license: "MIT".to_string(),
            entry_point: "plugin.py".to_string(),
            dependencies: vec![],
            xencode_version: "*".to_string(),
            permissions: vec!["read".to_string()],
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        fs::write(plugin_dir.join("plugin.json"), json).unwrap();

        let registry = PluginRegistry::new(tmp.path().to_path_buf());
        let manifests = registry.discover();
        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].name, "my-plugin");
    }
}
