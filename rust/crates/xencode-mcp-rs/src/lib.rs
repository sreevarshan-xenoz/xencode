//! A client for the Model Context Protocol, as far as xencode uses it: the
//! stdio transport, and the tools, resources and prompts a server declares.
//!
//! ```no_run
//! use std::time::Duration;
//! use xencode_mcp_rs::{McpClient, ServerSpec};
//!
//! # async fn demo() -> Result<(), xencode_mcp_rs::McpError> {
//! let spec = ServerSpec::new("docs", "mcp-docs-server").args(&["--root".into(), "/docs".into()]);
//! let client = McpClient::start(&spec, Duration::from_secs(30)).await?;
//! for tool in client.list_tools().await? {
//!     println!("{}: {}", tool.name, tool.description);
//! }
//! client.shutdown().await;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod error;
pub mod protocol;

pub use client::{McpClient, ServerSpec};
pub use error::McpError;
pub use protocol::{
    Feature, McpPrompt, McpResource, McpTool, PromptArgument, PromptMessage, ResourceContent,
    ServerCapabilities, ToolOutcome, PROTOCOL_VERSION,
};
