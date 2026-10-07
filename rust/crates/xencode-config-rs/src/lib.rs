pub mod composition;
pub mod config;
pub mod files;
pub mod paths;
pub mod remotes;
pub mod secrets;

pub use composition::{CompositionProfile, CompositionSummary, KNOWN_CAPABILITIES};
pub use config::{
    AgentHooks, ApiKeys, ColabConfig, ConfigError, McpServer, ModelProfile, XencodeConfig,
    CURRENT_CONFIG_VERSION, LEGACY_CONFIG_VERSION,
};
pub use files::{create_file, delete_file, read_file, write_file, FileError};
pub use secrets::{
    describe as secret_describe, is_present as secret_is_present, is_secret_reference,
    resolve as secret_resolve, SecretProblem, SecretProvider, SECRET_COMMAND_PREFIX,
    SECRET_HELPER_TIMEOUT,
};
