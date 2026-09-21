use std::path::Path;

use crate::host::{BasicHost, Host};
use crate::manifest::{PluginHooks, PluginManifest};
use crate::plugin_trait::{PluginError, PluginEvent, PluginResponse, XencodePlugin};
use crate::registry::PluginRegistry;

/// What happened to one manifest at load time. The CLI and the TUI both show
/// these, so "installed" and "took hold" are never the same word by accident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadReport {
    pub name: String,
    pub version: String,
    /// Registered with the host and initialized.
    pub loaded: bool,
    /// Why not, in the manifest's or the host's own words.
    pub reason: Option<String>,
    /// Whether it contributes text to the agent's system prompt.
    pub prompt_prefix: bool,
    pub before_hooks: usize,
    pub after_hooks: usize,
}

impl LoadReport {
    pub fn summary(&self) -> String {
        if !self.loaded {
            return format!(
                "{} v{} — NOT LOADED: {}",
                self.name,
                self.version,
                self.reason.clone().unwrap_or_else(|| "unknown".to_string())
            );
        }
        let mut parts = Vec::new();
        if self.prompt_prefix {
            parts.push("prompt prefix".to_string());
        }
        if self.before_hooks > 0 {
            parts.push(format!("{} before hook(s)", self.before_hooks));
        }
        if self.after_hooks > 0 {
            parts.push(format!("{} after hook(s)", self.after_hooks));
        }
        if parts.is_empty() {
            parts.push("declares nothing".to_string());
        }
        format!(
            "{} v{} — loaded: {}",
            self.name,
            self.version,
            parts.join(", ")
        )
    }
}

/// A plugin built from a manifest. It has no code of its own: what it adds to
/// the session is the prompt prefix and the hooks the runtime read off it at
/// load time. `handle_event` answers nothing because there is no plugin code
/// to answer — this build does not load executable plugins, and saying so here
/// is the point of the type.
#[derive(Debug)]
pub struct ManifestPlugin {
    manifest: PluginManifest,
    initialized: bool,
}

impl ManifestPlugin {
    pub fn new(manifest: PluginManifest) -> Self {
        Self {
            manifest,
            initialized: false,
        }
    }

    pub fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    pub fn declared(&self) -> (PluginHooks, String) {
        (
            self.manifest.hooks.clone(),
            self.manifest.prompt_prefix.clone(),
        )
    }
}

impl XencodePlugin for ManifestPlugin {
    fn name(&self) -> &str {
        &self.manifest.name
    }

    fn version(&self) -> &str {
        &self.manifest.version
    }

    fn description(&self) -> &str {
        &self.manifest.description
    }

    fn initialize(
        &mut self,
        _host: &dyn crate::plugin_trait::PluginHost,
    ) -> Result<(), PluginError> {
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), PluginError> {
        self.initialized = false;
        Ok(())
    }

    fn handle_event(&mut self, _event: PluginEvent) -> Result<Option<PluginResponse>, PluginError> {
        if !self.initialized {
            return Err(PluginError::InitFailed(self.manifest.name.clone()));
        }
        Ok(None)
    }
}

/// The plugins this session actually took hold: registered with a host,
/// initialized, and flattened into the two things the agent loop consumes.
pub struct PluginRuntime {
    host: Host,
    reports: Vec<LoadReport>,
    prompt_prefix: String,
    hooks: PluginHooks,
    dir: std::path::PathBuf,
}

impl PluginRuntime {
    /// Discover manifests under `dir` and load the ones this build can honour.
    ///
    /// A manifest whose `xencode_version` does not accept `xencode_version` is
    /// reported and skipped rather than applied. Load order is name order, so
    /// two plugins that both edit the prompt produce the same text every run.
    pub fn load(dir: &Path, xencode_version: &str) -> Self {
        let registry = PluginRegistry::new(dir.to_path_buf());
        let (mut host, _rx) = Host::new(&dir.display().to_string());
        let mut reports = Vec::new();
        let mut prefix = String::new();
        let mut hooks = PluginHooks::default();

        let mut manifests = registry.discover();
        manifests.sort_by(|a, b| a.name.cmp(&b.name));

        for manifest in manifests {
            let name = manifest.name.clone();
            if !manifest.is_compatible_with(xencode_version) {
                reports.push(not_loaded(
                    &manifest,
                    Some(format!(
                        "needs xencode {xencode_version} (declared {})",
                        manifest.xencode_version
                    )),
                ));
                continue;
            }
            let declared = (manifest.hooks.clone(), manifest.prompt_prefix.clone());
            let built = match registry.load_plugin(&manifest) {
                Ok(plugin) => plugin,
                Err(e) => {
                    reports.push(not_loaded(&manifest, Some(e.to_string())));
                    continue;
                }
            };
            let host_ctx = BasicHost::new(&dir.display().to_string());
            match host.register(Box::new(built), manifest.clone(), &host_ctx) {
                Ok(()) => {
                    let (plugin_hooks, plugin_prefix) = declared;
                    if !plugin_prefix.trim().is_empty() {
                        if !prefix.is_empty() {
                            prefix.push('\n');
                        }
                        prefix.push_str(plugin_prefix.trim());
                    }
                    merge_hooks(&mut hooks, &plugin_hooks);
                    reports.push(LoadReport {
                        name,
                        version: manifest.version.clone(),
                        loaded: true,
                        reason: None,
                        prompt_prefix: !plugin_prefix.trim().is_empty(),
                        before_hooks: plugin_hooks.before.len(),
                        after_hooks: plugin_hooks.after.len(),
                    });
                }
                Err(e) => reports.push(not_loaded(&manifest, Some(e.to_string()))),
            }
        }

        Self {
            host,
            reports,
            prompt_prefix: prefix,
            hooks,
            dir: dir.to_path_buf(),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// A runtime that loaded nothing, for a caller that must not look at the
    /// user's plugin directory: `App::for_tests()` uses this so a plugin someone
    /// installed on their own machine cannot change what a test asserts.
    pub fn empty(dir: impl Into<std::path::PathBuf>) -> Self {
        let dir = dir.into();
        let (host, _rx) = Host::new(&dir.display().to_string());
        Self {
            host,
            reports: Vec::new(),
            prompt_prefix: String::new(),
            hooks: PluginHooks::default(),
            dir,
        }
    }

    /// Text the loaded plugins add ahead of the agent's system prompt.
    pub fn prompt_prefix(&self) -> &str {
        &self.prompt_prefix
    }

    /// Hooks the loaded plugins declared, with nothing from the config mixed
    /// in — the caller decides precedence.
    pub fn hooks(&self) -> &PluginHooks {
        &self.hooks
    }

    pub fn reports(&self) -> &[LoadReport] {
        &self.reports
    }

    pub fn loaded_count(&self) -> usize {
        self.host.plugin_count()
    }
}

/// A plugin's entry only lands where the merged map is still silent: an
/// explicit config hook outranks a plugin's, and the exact-tool form outranks
/// a `*` from another plugin.
fn merge_hooks(into: &mut PluginHooks, from: &PluginHooks) {
    for table in [
        (&mut into.before, &from.before),
        (&mut into.after, &from.after),
    ] {
        for (tool, command) in table.1 {
            table
                .0
                .entry(tool.clone())
                .or_insert_with(|| command.clone());
        }
    }
}

fn not_loaded(manifest: &PluginManifest, reason: Option<String>) -> LoadReport {
    LoadReport {
        name: manifest.name.clone(),
        version: manifest.version.clone(),
        loaded: false,
        reason,
        prompt_prefix: false,
        before_hooks: 0,
        after_hooks: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn plugin_dir(root: &Path, name: &str, json: &str) {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("plugin.json"), json).unwrap();
    }

    fn write(root: &Path, name: &str, version: &str, xencode_version: Option<&str>) {
        let xv = xencode_version.unwrap_or("*");
        plugin_dir(
            root,
            name,
            &format!(
                r#"{{ "name": "{name}", "version": "{version}", "xencode_version": "{xv}" }}"#
            ),
        );
    }

    #[test]
    fn a_manifest_becomes_a_registered_plugin() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "noop", "1.0.0", None);

        let runtime = PluginRuntime::load(tmp.path(), "0.1.0");
        assert_eq!(runtime.loaded_count(), 1);
        assert_eq!(runtime.reports().len(), 1);
        assert!(runtime.reports()[0].loaded);
        assert_eq!(
            runtime.reports()[0].summary(),
            "noop v1.0.0 — loaded: declares nothing"
        );
        assert!(runtime.prompt_prefix().is_empty());
        assert!(runtime.hooks().is_empty());
        assert!(runtime.host.get_plugin("noop").is_some());
    }

    #[test]
    fn declared_prefix_and_hooks_reach_the_agent_loop_inputs() {
        let tmp = tempfile::tempdir().unwrap();
        plugin_dir(
            tmp.path(),
            "guardrails",
            r#"{
                "name": "guardrails",
                "version": "2.0.0",
                "prompt_prefix":  "  Run the tests before answering.  ",
                "hooks": { "before": { "write_file": "cargo check" }, "after": { "*": "cargo fmt" } }
            }"#,
        );

        let runtime = PluginRuntime::load(tmp.path(), "0.1.0");
        assert_eq!(runtime.prompt_prefix(), "Run the tests before answering.");
        assert_eq!(runtime.hooks().before["write_file"], "cargo check");
        assert_eq!(runtime.hooks().after["*"], "cargo fmt");
        assert_eq!(
            runtime.reports()[0].summary(),
            "guardrails v2.0.0 — loaded: prompt prefix, 1 before hook(s), 1 after hook(s)"
        );
    }

    /// A manifest pinned to a version this build is not must be reported, not
    /// quietly applied.
    #[test]
    fn an_incompatible_manifest_is_skipped_with_its_own_declaration() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "future", "1.0.0", Some("9.9.9"));

        let runtime = PluginRuntime::load(tmp.path(), "0.1.0");
        assert_eq!(runtime.loaded_count(), 0);
        assert!(!runtime.reports()[0].loaded);
        let summary = runtime.reports()[0].summary();
        assert!(summary.contains("NOT LOADED"), "{summary}");
        assert!(summary.contains("needs xencode 0.1.0"), "{summary}");
        assert!(summary.contains("declared 9.9.9"), "{summary}");
    }

    /// Two plugins both editing the prompt must produce the same text every
    /// run, whichever order the directory walk returned them in.
    #[test]
    fn load_order_is_the_plugin_name_so_the_prompt_is_stable() {
        let tmp = tempfile::tempdir().unwrap();
        for (name, prefix) in [("zeta", "Z last"), ("alpha", "A first")] {
            plugin_dir(
                tmp.path(),
                name,
                &format!(r#"{{ "name": "{name}", "version": "1", "prompt_prefix": "{prefix}" }}"#),
            );
        }

        let runtime = PluginRuntime::load(tmp.path(), "0.1.0");
        assert_eq!(runtime.prompt_prefix(), "A first\nZ last");
        let names: Vec<&str> = runtime.reports().iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "zeta"]);
    }

    #[test]
    fn the_first_plugin_to_declare_a_hook_keeps_it() {
        let tmp = tempfile::tempdir().unwrap();
        for (name, command) in [("aa", "cargo check"), ("bb", "cargo clippy")] {
            plugin_dir(
                tmp.path(),
                name,
                &format!(
                    r#"{{ "name": "{name}", "version": "1", "hooks": {{ "before": {{ "*": "{command}" }} }} }}"#
                ),
            );
        }

        let runtime = PluginRuntime::load(tmp.path(), "0.1.0");
        assert_eq!(runtime.hooks().before["*"], "cargo check");
    }

    #[test]
    fn an_empty_or_absent_plugin_dir_loads_nothing_and_says_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let runtime = PluginRuntime::load(&tmp.path().join("nope"), "0.1.0");
        assert_eq!(runtime.loaded_count(), 0);
        assert!(runtime.reports().is_empty());
        assert!(runtime.prompt_prefix().is_empty());
        assert!(runtime.hooks().is_empty());
    }

    /// A manifest plugin has no code, so it answers no events — but it answers
    /// only while the host has it initialized.
    #[test]
    fn a_manifest_plugin_answers_no_events() {
        let mut plugin = ManifestPlugin::new(PluginManifest {
            name: "guardrails".to_string(),
            version: "1.0.0".to_string(),
            ..Default::default()
        });
        let err = plugin
            .handle_event(PluginEvent::new("agent:turn", serde_json::Value::Null))
            .unwrap_err();
        assert!(matches!(err, PluginError::InitFailed(_)), "{err:?}");

        plugin
            .initialize(&BasicHost::new("/tmp"))
            .expect("initialize");
        assert_eq!(plugin.name(), "guardrails");
        assert_eq!(plugin.version(), "1.0.0");
        assert!(plugin.description().is_empty());
        assert!(plugin
            .handle_event(PluginEvent::new("agent:turn", serde_json::Value::Null))
            .unwrap()
            .is_none());
        plugin.shutdown().expect("shut down");
        assert!(plugin
            .handle_event(PluginEvent::new("agent:turn", serde_json::Value::Null))
            .is_err());
    }

    #[test]
    fn merging_hooks_never_overwrites_an_earlier_declaration() {
        let mut into = PluginHooks::default();
        into.before
            .insert("*".to_string(), "from config".to_string());
        let mut from = PluginHooks::default();
        from.before
            .insert("*".to_string(), "from plugin".to_string());
        from.after
            .insert("edit_file".to_string(), "cargo fmt".to_string());

        merge_hooks(&mut into, &from);
        assert_eq!(into.before["*"], "from config");
        assert_eq!(into.after["edit_file"], "cargo fmt");
    }
}
