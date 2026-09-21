//! Error surface for the MCP client.
//!
//! Every variant names the server it came from: a tool loop that reports
//! "connection closed" without saying *which* of the configured servers died
//! is not actionable for the model or the user.

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum McpError {
    /// The server process could not be started (bad command, not executable).
    #[error("cannot start MCP server `{server}`: {reason}")]
    Spawn { server: String, reason: String },
    /// The server accepted no answer within the request budget.
    #[error("MCP server `{server}` did not answer `{method}` within {secs}s")]
    Timeout {
        server: String,
        method: String,
        secs: u64,
    },
    /// The server's stdout ended: it exited or was killed mid-request.
    #[error("MCP server `{server}` closed the connection")]
    Closed { server: String },
    /// The bytes on stdout were not the protocol we speak.
    #[error("MCP server `{server}` broke protocol: {message}")]
    Protocol { server: String, message: String },
    /// A well-formed JSON-RPC error response. The method is not recoverable
    /// here (a response carries only its id), so the message stands alone.
    #[error("MCP server `{server}` returned error {code}: {message}")]
    Server {
        server: String,
        code: i64,
        message: String,
    },
}

impl McpError {
    /// Which server is at fault, for the caller's transcript line.
    pub fn server(&self) -> &str {
        match self {
            McpError::Spawn { server, .. }
            | McpError::Timeout { server, .. }
            | McpError::Closed { server, .. }
            | McpError::Protocol { server, .. }
            | McpError::Server { server, .. } => server,
        }
    }
}
