use std::path::{Component, Path, PathBuf};

use crate::manifest::PluginManifest;
use crate::plugin_trait::PluginError;

/// True if `name` is usable as a directory name directly under the plugin dir.
///
/// Plugin names reach us from manifest files, which for a third-party plugin are
/// attacker-controlled. `Path::join` happily walks out of the plugin directory
/// given `../..`, and replaces the base entirely given an absolute path, so a
/// name is only accepted when it is exactly one normal path component.
fn is_safe_plugin_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    )
}

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

            // Look for manifest files. JSON only: a `plugin.yaml` used to be on
            // this list while the parser was `serde_json`, so it could never
            // load and never said why.
            for manifest_name in &["plugin.json", "manifest.json"] {
                let manifest_path = path.join(manifest_name);
                if manifest_path.exists() {
                    if let Ok(manifest) = PluginManifest::from_file(&manifest_path) {
                        // Drop a manifest whose name can't be turned into a path
                        // inside the plugin dir, so it never reaches load_plugin.
                        if is_safe_plugin_name(&manifest.name) {
                            manifests.push(manifest);
                        }
                    }
                }
            }
        }

        manifests
    }

    /// Build the plugin a manifest describes.
    ///
    /// There is no dynamic linking in this build, so a manifest *is* the
    /// plugin: its prompt prefix and declared hooks, wrapped in the trait the
    /// host registers. A manifest whose name escapes the plugin directory, or
    /// whose directory is not there, is refused here rather than loaded.
    pub fn load_plugin(
        &self,
        manifest: &PluginManifest,
    ) -> Result<crate::runtime::ManifestPlugin, PluginError> {
        let plugin_path = self
            .plugin_path(&manifest.name)
            .ok_or_else(|| PluginError::InvalidName(manifest.name.clone()))?;
        if !plugin_path.exists() {
            return Err(PluginError::NotFound(manifest.name.clone()));
        }
        Ok(crate::runtime::ManifestPlugin::new(manifest.clone()))
    }

    /// Get the full path to a plugin's directory.
    ///
    /// Returns `None` for a name that would escape the plugin directory.
    pub fn plugin_path(&self, name: &str) -> Option<PathBuf> {
        is_safe_plugin_name(name).then(|| self.plugin_dir.join(name))
    }

    /// Check if a plugin exists in the registry.
    ///
    /// A name that would escape the plugin directory is never present.
    pub fn has_plugin(&self, name: &str) -> bool {
        self.plugin_path(name).is_some_and(|path| path.exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin_trait::XencodePlugin as _;
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
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&manifest).unwrap();
        fs::write(plugin_dir.join("plugin.json"), json).unwrap();

        let registry = PluginRegistry::new(tmp.path().to_path_buf());
        let manifests = registry.discover();
        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].name, "my-plugin");
    }

    fn manifest_named(name: &str) -> PluginManifest {
        PluginManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: "Test".to_string(),
            author: "Author".to_string(),
            license: "MIT".to_string(),
            permissions: vec!["read".to_string()],
            ..Default::default()
        }
    }

    /// Names that must never be turned into a path under the plugin directory.
    const UNSAFE_NAMES: &[&str] = &[
        "../evil",
        "../../etc/passwd",
        "..",
        ".",
        "",
        "nested/plugin",
        "a/../../b",
        #[cfg(unix)]
        "/etc/passwd",
        #[cfg(windows)]
        r"C:\Windows",
    ];

    #[test]
    fn plugin_path_rejects_names_that_escape_the_plugin_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = PluginRegistry::new(tmp.path().to_path_buf());

        for name in UNSAFE_NAMES {
            assert!(
                registry.plugin_path(name).is_none(),
                "plugin_path accepted {name:?}"
            );
        }
    }

    #[test]
    fn plugin_path_accepts_a_plain_name() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = PluginRegistry::new(tmp.path().to_path_buf());

        assert_eq!(
            registry.plugin_path("my-plugin"),
            Some(tmp.path().join("my-plugin"))
        );
    }

    #[test]
    fn has_plugin_is_false_for_escaping_names_even_when_the_target_exists() {
        let tmp = tempfile::tempdir().unwrap();
        // A real directory that a traversal would otherwise reach: the plugin
        // dir's own sibling.
        let outside = tmp.path().join("outside");
        fs::create_dir_all(&outside).unwrap();
        let registry = PluginRegistry::new(tmp.path().join("plugins"));
        fs::create_dir_all(tmp.path().join("plugins")).unwrap();

        assert!(tmp.path().join("outside").exists());
        assert!(!registry.has_plugin("../outside"));
    }

    #[test]
    fn load_plugin_rejects_an_escaping_manifest_name() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = PluginRegistry::new(tmp.path().to_path_buf());

        let err = registry
            .load_plugin(&manifest_named("../evil"))
            .expect_err("load_plugin accepted a traversing name");
        assert!(matches!(err, PluginError::InvalidName(_)), "got {err:?}");
    }

    #[test]
    fn load_plugin_rejects_a_manifest_whose_directory_is_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let registry = PluginRegistry::new(tmp.path().to_path_buf());

        let err = registry
            .load_plugin(&manifest_named("not-installed"))
            .expect_err("a manifest with no directory is not a plugin");
        assert!(matches!(err, PluginError::NotFound(_)), "got {err:?}");
    }

    /// J-08: the registry hands back something the host can register, not a
    /// copy of the manifest it was given.
    #[test]
    fn load_plugin_builds_the_plugin_a_manifest_describes() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("guardrails");
        fs::create_dir_all(&dir).unwrap();
        let mut manifest = manifest_named("guardrails");
        manifest.prompt_prefix = "Run the tests first.".to_string();
        manifest
            .hooks
            .before
            .insert("write_file".to_string(), "cargo check".to_string());

        let registry = PluginRegistry::new(tmp.path().to_path_buf());
        let plugin = registry.load_plugin(&manifest).unwrap();
        assert_eq!(plugin.name(), "guardrails");
        assert_eq!(plugin.version(), "1.0.0");
        let (hooks, prefix) = plugin.declared();
        assert_eq!(prefix, "Run the tests first.");
        assert_eq!(hooks.before["write_file"], "cargo check");
    }

    #[test]
    fn discover_skips_a_manifest_whose_name_escapes_the_plugin_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let plugin_dir = tmp.path().join("looks-innocent");
        fs::create_dir_all(&plugin_dir).unwrap();

        let json = serde_json::to_string_pretty(&manifest_named("../../evil")).unwrap();
        fs::write(plugin_dir.join("plugin.json"), json).unwrap();

        let registry = PluginRegistry::new(tmp.path().to_path_buf());
        assert!(registry.discover().is_empty());
    }
}
