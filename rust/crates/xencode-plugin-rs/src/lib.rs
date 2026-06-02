pub mod host;
pub mod manifest;
pub mod plugin_trait;
pub mod registry;

pub use host::{BasicHost, Host};
pub use manifest::PluginManifest;
pub use plugin_trait::{PluginError, PluginEvent, PluginHost, PluginResponse, XencodePlugin};
pub use registry::PluginRegistry;
