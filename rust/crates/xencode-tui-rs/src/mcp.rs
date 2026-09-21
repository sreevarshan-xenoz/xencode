//! The TUI's side of MCP (I3-01): what to offer the model, and where a call
//! goes once it is approved.
//!
//! [`xencode_mcp_rs`] speaks the protocol; this module owns the live sessions,
//! the names the model sees, and the strings the transcript shows. Servers are
//! only started by `/mcp` — a configured-but-broken server must not stall
//! startup, and a turn should not pay for a handshake it may not need.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use xencode_mcp_rs::McpClient;
pub use xencode_mcp_rs::ServerSpec;
use xencode_providers_rs::ToolDefinition;

/// Prefix that marks a tool as coming from a server rather than from xencode.
pub const MCP_PREFIX: &str = "mcp__";

/// A server's declaration in config, turned into what the client needs.
pub fn spec_from_config(name: &str, server: &xencode_config_rs::McpServer) -> ServerSpec {
    ServerSpec {
        name: name.to_string(),
        command: server.command.clone(),
        args: server.args.clone(),
        env: server
            .env
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    }
}

/// A name a provider will accept as a function: ASCII alphanumerics, `_` and
/// `-`, at most 64 characters including the prefix.
fn sanitize(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `mcp__<server>__<tool>`, shortened from the tool end if the whole thing will
/// not fit. The server name wins over the tool name because the transcript and
/// the session map are both keyed by server.
pub fn full_tool_name(server: &str, tool: &str) -> String {
    let server = sanitize(server);
    let mut tool = sanitize(tool);
    let budget = 64usize.saturating_sub(MCP_PREFIX.len() + server.len() + 2);
    if tool.len() > budget {
        let mut cut = budget;
        while cut > 0 && !tool.is_char_boundary(cut) {
            cut -= 1;
        }
        tool.truncate(cut);
    }
    format!("{MCP_PREFIX}{server}__{tool}")
}

/// The `(server, tool)` pair behind a prefixed name, if it is one. Both halves
/// are already sanitized, so this is the comparison to make against a key.
pub fn split_full_name(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_prefix(MCP_PREFIX)?;
    let (server, tool) = rest.split_once("__")?;
    if server.is_empty() || tool.is_empty() {
        return None;
    }
    Some((server, tool))
}

/// Whether a model-visible tool name comes from a server.
pub fn is_mcp_tool(name: &str) -> bool {
    split_full_name(name).is_some()
}

/// What connecting one server produced. Every field is something the server
/// actually reported; a failed connect says why in the server's own words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub server: String,
    pub connected: bool,
    pub tools: usize,
    pub detail: String,
}

struct Session {
    client: Arc<McpClient>,
    /// `(name the model sees, name the server knows)`, in the server's order.
    tools: Vec<(String, String)>,
}

/// The connected servers for this session, shared between `App` (which draws
/// `/mcp` output) and every agent loop (which calls tools).
pub struct McpHub {
    sessions: tokio::sync::Mutex<BTreeMap<String, Session>>,
    /// Snapshot of what to offer the model, readable without awaiting so a turn
    /// can be armed synchronously.
    offered: std::sync::RwLock<Vec<ToolDefinition>>,
}

impl Default for McpHub {
    fn default() -> Self {
        Self::new()
    }
}

impl McpHub {
    pub fn new() -> Self {
        McpHub {
            sessions: tokio::sync::Mutex::new(BTreeMap::new()),
            offered: std::sync::RwLock::new(Vec::new()),
        }
    }

    /// Start every listed server that is not already running, replacing any
    /// session with the same name. One [`Report`] per server, in spec order.
    pub async fn connect(&self, specs: &[ServerSpec], timeout: Duration) -> Vec<Report> {
        let mut reports = Vec::new();
        for spec in specs {
            let key = sanitize(&spec.name);
            if let Some(tools) = self.running_tools(&key).await {
                reports.push(Report {
                    server: spec.name.clone(),
                    connected: true,
                    tools,
                    detail: "already running".to_string(),
                });
                continue;
            }
            reports.push(self.connect_one(spec, key, timeout).await);
        }
        reports
    }

    async fn connect_one(&self, spec: &ServerSpec, key: String, timeout: Duration) -> Report {
        let failed = |detail: String| Report {
            server: spec.name.clone(),
            connected: false,
            tools: 0,
            detail,
        };
        let client = match McpClient::start(spec, timeout).await {
            Ok(client) => client,
            Err(e) => return failed(e.to_string()),
        };
        let tools = match client.list_tools().await {
            Ok(tools) => tools,
            Err(e) => {
                client.shutdown().await;
                return failed(format!("started, but {}; it is not running", e));
            }
        };

        // Two tools of one server can sanitize to the same visible name; the
        // second is dropped rather than shadowing the first, and said so.
        let mut named: Vec<(String, String)> = Vec::new();
        let mut definitions: Vec<ToolDefinition> = Vec::new();
        let mut clashes = 0usize;
        for tool in &tools {
            let visible = full_tool_name(&key, &tool.name);
            if named.iter().any(|(seen, _)| seen == &visible) {
                clashes += 1;
                continue;
            }
            named.push((visible.clone(), tool.name.clone()));
            definitions.push(ToolDefinition {
                name: visible,
                description: format!(
                    "MCP tool `{}` from server `{}`. {}",
                    tool.name,
                    spec.name,
                    if tool.description.is_empty() {
                        "The server supplied no description."
                    } else {
                        tool.description.as_str()
                    }
                ),
                parameters: tool.input_schema.clone(),
            });
        }
        let count = definitions.len();
        let detail = if clashes > 0 {
            format!("{count} tool(s), {clashes} dropped for a clashing name")
        } else if count == 0 {
            "no tools".to_string()
        } else {
            format!("{count} tool(s)")
        };
        self.add_session(key, Arc::new(client), named, definitions)
            .await;
        Report {
            server: spec.name.clone(),
            connected: true,
            tools: count,
            detail,
        }
    }

    async fn running_tools(&self, key: &str) -> Option<usize> {
        self.sessions
            .lock()
            .await
            .get(key)
            .map(|session| session.tools.len())
    }

    async fn add_session(
        &self,
        key: String,
        client: Arc<McpClient>,
        tools: Vec<(String, String)>,
        definitions: Vec<ToolDefinition>,
    ) {
        let previous = {
            let mut sessions = self.sessions.lock().await;
            sessions.insert(key.clone(), Session { client, tools })
        };
        // A server replaced by its own new session stops as soon as we drop its
        // pipes; the caller must not end up with two of one name.
        if let Some(previous) = previous {
            previous.client.shutdown().await;
        }
        let mut offered = self.write_offered();
        offered.retain(|definition| {
            split_full_name(&definition.name).map(|(server, _)| server) != Some(key.as_str())
        });
        offered.extend(definitions);
    }

    fn write_offered(&self) -> std::sync::RwLockWriteGuard<'_, Vec<ToolDefinition>> {
        self.offered
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// What to offer the model this turn. xencode's own tools are added by the
    /// caller; these come after them.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.offered
            .read()
            .map(|offered| offered.clone())
            .unwrap_or_default()
    }

    pub fn is_empty(&self) -> bool {
        self.definitions().is_empty()
    }

    /// Run one prefixed call. The string always says what happened, including
    /// when the server or the tool is gone.
    pub async fn call(&self, full_name: &str, arguments: serde_json::Value) -> String {
        let Some((key, _)) = split_full_name(full_name) else {
            return format!("error: {full_name} is not an MCP tool name");
        };
        // The lock is released before awaiting: one slow server must not
        // block another one's calls, or `/mcp` status, for the whole timeout.
        let session = {
            let sessions = self.sessions.lock().await;
            sessions
                .get(key)
                .map(|session| (Arc::clone(&session.client), session.tools.clone()))
        };
        let Some((client, tools)) = session else {
            return format!("error: MCP server `{key}` is not running — start it with /mcp");
        };
        let Some(real) = tools
            .iter()
            .find(|(visible, _)| visible == full_name)
            .map(|(_, real)| real.clone())
        else {
            return format!("error: MCP server `{key}` does not offer {full_name}");
        };
        match client.call_tool(&real, arguments).await {
            Ok(text) => text,
            Err(e) => format!("error: {e}"),
        }
    }

    /// One line per running server for `/mcp`, including what it printed that
    /// was not the protocol. No servers means a line saying so.
    pub async fn status_lines(&self) -> Vec<String> {
        let sessions = self.sessions.lock().await;
        if sessions.is_empty() {
            return vec![
                "no MCP servers running — declare them under \"mcp_servers\" in config.json, then run /mcp"
                    .to_string(),
            ];
        }
        sessions
            .values()
            .map(|session| {
                let stray = session.client.stray_lines();
                let noise = if stray > 0 {
                    format!(", {stray} non-protocol line(s) ignored")
                } else {
                    String::new()
                };
                format!(
                    "{}: {} tool(s){noise}",
                    session.client.name(),
                    session.tools.len(),
                )
            })
            .collect()
    }

    /// Kill everything and withdraw every tool. Returns how many servers were
    /// running, so the transcript can state it.
    pub async fn stop_all(&self) -> usize {
        let drained: Vec<Session> = {
            let mut sessions = self.sessions.lock().await;
            let sessions = std::mem::take(&mut *sessions);
            sessions.into_values().collect()
        };
        let count = drained.len();
        for session in drained {
            session.client.shutdown().await;
        }
        self.write_offered().clear();
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    #[test]
    fn prefixed_names_survive_the_providers_name_rule() {
        let name = full_tool_name("my.server", "get!page");
        assert_eq!(name, "mcp__my_server__get_page");
        assert!(
            name.len() <= 64,
            "providers reject longer function names: {name}"
        );
        assert_eq!(split_full_name(&name), Some(("my_server", "get_page")));
        assert!(is_mcp_tool(&name));
    }

    #[test]
    fn an_over_long_tool_is_cut_from_its_own_end() {
        let name = full_tool_name("srv", &"u".repeat(90));
        assert_eq!(name.len(), 64);
        assert!(name.starts_with("mcp__srv__uuu"));
        // The server part is never what gets truncated.
        assert_eq!(
            split_full_name(&name).map(|(server, _)| server),
            Some("srv")
        );
    }

    #[test]
    fn only_well_formed_prefixed_names_are_mcp() {
        for name in [
            "read_file",
            "mcp__",
            "mcp__srv__",
            "mcp______tool",
            "mcp_srv_tool",
        ] {
            assert!(!is_mcp_tool(name), "{name} is not an MCP tool name");
        }
    }

    #[tokio::test]
    async fn an_empty_hub_offers_nothing_and_says_so() {
        let hub = McpHub::new();
        assert!(hub.is_empty());
        assert!(hub.definitions().is_empty());
        let lines = hub.status_lines().await;
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("no MCP servers running"), "{lines:?}");
        // A call against nothing fails in words, not by panic.
        assert!(hub
            .call("mcp__gone__tool", serde_json::json!({}))
            .await
            .starts_with("error:"));
    }

    #[tokio::test]
    async fn connecting_a_dead_server_reports_failure_without_breaking_the_hub() {
        let hub = McpHub::new();
        let specs = vec![ServerSpec::new("ghost", "/nonexistent/mcp-server")];
        let reports = hub.connect(&specs, Duration::from_secs(2)).await;
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].connected);
        assert!(reports[0].detail.contains("ghost"), "{:?}", reports[0]);
        assert!(hub.is_empty(), "a failed server must offer no tools");
        // Asking again reports the same truth rather than piling up sessions.
        let again = hub.connect(&specs, Duration::from_secs(2)).await;
        assert_eq!(again, reports);
    }

    #[tokio::test]
    async fn a_call_to_a_server_that_was_never_started_names_it() {
        let hub = McpHub::new();
        let said = hub.call("mcp__absent__tool", serde_json::json!({})).await;
        assert!(said.contains("absent"), "{said}");
        assert!(
            said.contains("/mcp"),
            "the model is told how to fix it: {said}"
        );
    }

    #[tokio::test]
    async fn stopping_everything_withdraws_the_tools() {
        let hub = McpHub::new();
        assert_eq!(hub.stop_all().await, 0, "nothing was running");
        assert!(hub.is_empty());
    }

    /// A server that really exists, spawned by path from a temp dir: the whole
    /// bridge — handshake, names offered to the model, a routed call, teardown.
    /// No ports, nothing installed.
    const FIXTURE: &str = r#"#!/bin/sh
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    case "$line" in
        *'"method":"initialize"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"0"}}}\n' "$id"
            ;;
        *'"method":"tools/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Returns its argument","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},{"name":"echo","description":"Shadow of the first, same visible name"}]}}\n' "$id"
            ;;
        *'"method":"tools/call"'*)
            text=$(printf '%s' "$line" | sed -n 's/.*"text":"\([^"]*\)".*/\1/p')
            printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"echo: %s"}]}}\n' "$id" "$text"
            ;;
    esac
done
"#;

    /// Write the fixture where no other test can collide with it.
    fn fixture_spec() -> (PathBuf, ServerSpec) {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let unique = format!(
            "xencode-mcp-hub-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let dir = std::env::temp_dir().join(&unique);
        std::fs::create_dir_all(&dir).expect("create fixture dir");
        let path = dir.join("server.sh");
        std::fs::write(&path, FIXTURE).expect("write fixture");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("make fixture executable");
        let spec =
            ServerSpec::new("fixture", "/bin/sh").args(&[path.to_string_lossy().to_string()]);
        (dir, spec)
    }

    #[tokio::test]
    async fn a_live_server_becomes_offered_tools_that_route_back() {
        let (dir, spec) = fixture_spec();
        let hub = McpHub::new();
        let reports = hub
            .connect(std::slice::from_ref(&spec), Duration::from_secs(5))
            .await;
        assert!(reports[0].connected, "{:?}", reports[0]);
        // Both tools sanitize to the same visible name, so only the first
        // stands — and the report says the other was dropped for it.
        assert_eq!(reports[0].tools, 1, "{:?}", reports[0]);
        assert!(reports[0].detail.contains("clashing"), "{:?}", reports[0]);

        let offered = hub.definitions();
        assert_eq!(offered.len(), 1);
        assert_eq!(offered[0].name, "mcp__fixture__echo");
        assert!(
            offered[0].description.contains("server `fixture`"),
            "{}",
            offered[0].description
        );
        // The server's own schema reaches the model untouched.
        assert_eq!(
            offered[0].parameters["properties"]["text"]["type"],
            "string"
        );

        assert_eq!(
            hub.call("mcp__fixture__echo", serde_json::json!({"text": "routed"}))
                .await,
            "echo: routed"
        );
        // A name the server never listed is refused by us, not sent to it.
        let missing = hub.call("mcp__fixture__lie", serde_json::json!({})).await;
        assert!(missing.starts_with("error:"), "{missing}");
        assert!(missing.contains("does not offer"), "{missing}");

        let status = hub.status_lines().await;
        assert_eq!(status, vec!["fixture: 1 tool(s)".to_string()]);

        // Reconnecting a running server is a no-op that reports itself as such.
        let again = hub.connect(&[spec], Duration::from_secs(5)).await;
        assert_eq!(again[0].detail, "already running");

        assert_eq!(hub.stop_all().await, 1);
        assert!(hub.is_empty(), "a stopped server offers nothing");
        assert!(hub
            .call("mcp__fixture__echo", serde_json::json!({}))
            .await
            .contains("not running"));
        std::fs::remove_dir_all(&dir).expect("clean fixture dir");
    }
}
