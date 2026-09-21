pub mod host;
pub mod manifest;
pub mod plugin_trait;
pub mod registry;
pub mod runtime;

pub use host::{BasicHost, Host};
pub use manifest::{PluginHooks, PluginManifest};
pub use plugin_trait::{PluginError, PluginEvent, PluginHost, PluginResponse, XencodePlugin};
pub use registry::PluginRegistry;
pub use runtime::{LoadReport, ManifestPlugin, PluginRuntime};

/// The directory installed plugins live in: `$XCODE_PLUGIN_DIR` when set (tests
/// and portable installs), else `<data dir>/xencode/plugins` — the same path
/// `xencode plugin install` writes to, so what you install is what loads.
pub fn default_plugin_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("XCODE_PLUGIN_DIR") {
        if !dir.is_empty() {
            return std::path::PathBuf::from(dir);
        }
    }
    dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("xencode")
        .join("plugins")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default must be the directory `xencode plugin install` writes to, or
    /// an installed plugin would never load. When the environment overrides it,
    /// that override is the contract instead, so there is nothing to assert.
    #[test]
    fn default_plugin_dir_is_the_install_location() {
        if std::env::var("XCODE_PLUGIN_DIR").is_ok() {
            return;
        }
        let dir = default_plugin_dir();
        assert_eq!(dir.file_name().unwrap(), "plugins");
        assert_eq!(dir.parent().unwrap().file_name().unwrap(), "xencode");
    }
}
