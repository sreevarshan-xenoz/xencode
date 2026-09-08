use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::info;

use crate::manifest::PluginManifest;
use crate::plugin_trait::{PluginError, PluginEvent, PluginHost, PluginResponse, XencodePlugin};

/// The plugin host manages loaded plugins and routes events to them.
pub struct Host {
    plugins: HashMap<String, Box<dyn XencodePlugin>>,
    manifests: HashMap<String, PluginManifest>,
    event_queue: mpsc::Sender<PluginEvent>,
    #[allow(dead_code)]
    plugin_dir: String,
}

/// A basic host implementation that stores configuration values.
pub struct BasicHost {
    plugin_dir: String,
    config_values: HashMap<String, String>,
}

impl BasicHost {
    pub fn new(plugin_dir: &str) -> Self {
        Self {
            plugin_dir: plugin_dir.to_string(),
            config_values: HashMap::new(),
        }
    }

    pub fn set_config(&mut self, key: &str, value: &str) {
        self.config_values
            .insert(key.to_string(), value.to_string());
    }
}

impl PluginHost for BasicHost {
    fn get_plugin_dir(&self) -> &str {
        &self.plugin_dir
    }

    fn get_config_value(&self, key: &str) -> Option<String> {
        self.config_values.get(key).cloned()
    }
}

impl Host {
    /// Create a new plugin host with the given plugin directory.
    pub fn new(plugin_dir: &str) -> (Self, mpsc::Receiver<PluginEvent>) {
        let (tx, rx) = mpsc::channel(256);
        (
            Self {
                plugins: HashMap::new(),
                manifests: HashMap::new(),
                event_queue: tx,
                plugin_dir: plugin_dir.to_string(),
            },
            rx,
        )
    }

    /// Register a plugin with the host.
    pub fn register(
        &mut self,
        plugin: Box<dyn XencodePlugin>,
        manifest: PluginManifest,
        host: &dyn PluginHost,
    ) -> Result<(), PluginError> {
        let name = plugin.name().to_string();
        if self.plugins.contains_key(&name) {
            return Err(PluginError::AlreadyRegistered(name));
        }

        // Initialize the plugin
        let mut p = plugin;
        p.initialize(host)?;

        self.plugins.insert(name.clone(), p);
        self.manifests.insert(name.clone(), manifest);
        info!("Plugin registered: {name}");
        Ok(())
    }

    /// Unregister a plugin by name.
    pub fn unregister(&mut self, name: &str) -> Result<(), PluginError> {
        if let Some(mut plugin) = self.plugins.remove(name) {
            plugin.shutdown()?;
            self.manifests.remove(name);
            info!("Plugin unregistered: {name}");
            Ok(())
        } else {
            Err(PluginError::NotFound(name.to_string()))
        }
    }

    /// Emit an event to the event queue.
    pub fn emit(&self, event: PluginEvent) {
        let _ = self.event_queue.try_send(event);
    }

    /// Get a reference to the event queue sender.
    pub fn event_sender(&self) -> &mpsc::Sender<PluginEvent> {
        &self.event_queue
    }

    /// Route an incoming event to all registered plugins.
    pub fn route_event(
        &mut self,
        event: &PluginEvent,
    ) -> Vec<Result<Option<PluginResponse>, PluginError>> {
        let mut results = Vec::new();
        for (name, plugin) in &mut self.plugins {
            match plugin.handle_event(event.clone()) {
                Ok(response) => {
                    if response.is_some() {
                        info!("Plugin {name} handled event: {}", event.event_type);
                    }
                    results.push(Ok(response));
                }
                Err(e) => {
                    tracing::warn!("Plugin {name} failed to handle event: {e}");
                    results.push(Err(e));
                }
            }
        }
        results
    }

    /// List all registered plugin manifests.
    pub fn list_plugins(&self) -> Vec<&PluginManifest> {
        self.manifests.values().collect()
    }

    /// Get a specific plugin by name.
    pub fn get_plugin(&self, name: &str) -> Option<&dyn XencodePlugin> {
        self.plugins.get(name).map(|p| p.as_ref())
    }

    /// Get the number of registered plugins.
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }
}
