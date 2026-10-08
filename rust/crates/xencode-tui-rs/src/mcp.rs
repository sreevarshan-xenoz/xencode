//! The TUI's side of MCP (I3-01, finished by M-6): what to offer the model,
//! where a call goes once it is approved, and how a running server's documents
//! and prepared prompts are read back to the person who started it.
//!
//! [`xencode_mcp_rs`] speaks the protocol; this module owns the live sessions,
//! the names the model sees, and the strings the transcript shows. Servers are
//! only started by `/mcp` — a configured-but-broken server must not stall
//! startup, and a turn should not pay for a handshake it may not need.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use xencode_mcp_rs::{McpClient, McpError, McpPrompt, McpResource};
pub use xencode_mcp_rs::{ServerSpec, Transport};
use xencode_providers_rs::ToolDefinition;

/// Prefix that marks a tool as coming from a server rather than from xencode.
pub const MCP_PREFIX: &str = "mcp__";

/// Longest function name a provider accepts. Both directions respect it: the
/// names we import from a server and the names we publish as one.
pub const MAX_TOOL_NAME: usize = 64;

/// A server's declaration in config, turned into what the client needs. A
/// declaration that names neither a command to spawn nor an address to post to
/// — or both — comes back as the sentence a person can act on instead of a
/// guess about which half they meant.
pub fn spec_from_config(
    name: &str,
    server: &xencode_config_rs::McpServer,
) -> Result<ServerSpec, String> {
    if let Some(problem) = server.misconfigured() {
        return Err(problem);
    }
    let spec = match server.endpoint_url() {
        Some(url) => {
            let mut spec = ServerSpec::http(name, url);
            for (key, value) in &server.headers {
                spec = spec.header(key.clone(), value.clone());
            }
            spec
        }
        None => {
            // `misconfigured()` said exactly one of the two is set, so the
            // command a spawned server needs is here.
            let command = server.spawn_command().unwrap_or_default();
            ServerSpec::new(name, command).args(&server.args).env(
                server
                    .env
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            )
        }
    };
    Ok(spec)
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
    let budget = MAX_TOOL_NAME.saturating_sub(MCP_PREFIX.len() + server.len() + 2);
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

/// A name xencode itself publishes when it is the server rather than the client.
///
/// `full_tool_name` is what a *caller* of xencode would see: our six names with
/// `mcp__xencode__` in front of them, and that prefix leaves only a limited
/// budget for the name behind it. So a published name goes through the same two
/// steps the client applies to a tool it imports — sanitize, then fit inside 64 —
/// and there is one rule for both directions rather than a server that advertises
/// something a client cannot represent.
pub fn advertised_tool_name(raw: &str) -> String {
    // Everything left after sanitizing is ASCII, so a byte cut cannot land
    // inside a character.
    let mut name = sanitize(raw);
    name.truncate(name.len().min(MAX_TOOL_NAME));
    name
}

/// What connecting one server produced. Every field is something the server
/// actually reported; a failed connect says why in the server's own words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub server: String,
    pub connected: bool,
    pub tools: usize,
    pub detail: String,
    /// What the server offered besides tools, one line each, named — a count of
    /// resources you cannot name is not something you can use.
    pub offers: Vec<String>,
}

/// The lines a server's resources and prompts render to. Shared by the connect
/// report and `/mcp status` so the two cannot describe the same session
/// differently. A set the handshake did not declare says so rather than
/// reporting zero, because "none" and "not offered" are different facts.
fn offer_lines(resources: &[McpResource], prompts: &[McpPrompt], notes: &[String]) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for note in notes {
        lines.push(format!("  {note}"));
    }
    for resource in resources {
        let named = if resource.name.is_empty() || resource.name == resource.uri {
            String::new()
        } else {
            format!(" — {}", resource.name)
        };
        // The uri leads, because it is what `/mcp read` has to be given.
        lines.push(format!("  resource {}{named}", resource.uri));
    }
    for prompt in prompts {
        let args = if prompt.arguments.is_empty() {
            "no arguments".to_string()
        } else {
            prompt
                .arguments
                .iter()
                .map(|argument| {
                    if argument.required {
                        format!("{}=", argument.name)
                    } else {
                        format!("[{}=]", argument.name)
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        lines.push(format!("  prompt {} ({args})", prompt.name));
    }
    lines
}

/// A set the server does not have, in the words a session line is built from.
/// "Not declared" and "declared it, then could not answer for it" are different
/// facts and are not collapsed into one.
fn capability_note(feature: &str, error: &McpError) -> String {
    match error {
        McpError::NotOffered { .. } => format!("{feature} — not declared by this server"),
        other => format!("{feature} — asked for, and it said: {other}"),
    }
}

/// A running session seen the way both `/mcp` and `/mcp status` need it: the
/// counts, what was refused, and the names behind them. Drawing it in one place
/// keeps the two commands from describing the same server differently.
struct SessionView<'a> {
    name: &'a str,
    endpoint: String,
    tools: usize,
    stray: usize,
    resources: &'a [McpResource],
    prompts: &'a [McpPrompt],
    refused: &'a [String],
}

impl<'a> SessionView<'a> {
    fn of(session: &'a Session) -> Self {
        SessionView {
            name: session.client.name(),
            endpoint: session.client.endpoint(),
            tools: session.tools.len(),
            stray: session.client.stray_lines(),
            resources: &session.resources,
            prompts: &session.prompts,
            refused: &session.refused,
        }
    }

    /// What this server has, counted. What it said it does not have is a separate
    /// line, so a count is never read as a refusal.
    fn counts(&self) -> String {
        let mut parts = vec![if self.tools == 0 {
            "no tools".to_string()
        } else {
            format!("{} tool(s)", self.tools)
        }];
        if !self.resources.is_empty() {
            parts.push(format!("{} resource(s)", self.resources.len()));
        }
        if !self.prompts.is_empty() {
            parts.push(format!("{} prompt(s)", self.prompts.len()));
        }
        parts.join(", ")
    }

    fn offers(&self) -> Vec<String> {
        offer_lines(self.resources, self.prompts, self.refused)
    }

    /// What connecting produced, for `/mcp`. `detail` is the sentence the
    /// command leads with; the names come along underneath.
    fn report(&self, server: &str, detail: impl Into<String>) -> Report {
        Report {
            server: server.to_string(),
            connected: true,
            tools: self.tools,
            detail: detail.into(),
            offers: self.offers(),
        }
    }

    /// The line `/mcp status` draws: counts and endpoint first, then every
    /// resource and prompt by name underneath.
    fn lines(&self) -> Vec<String> {
        let noise = if self.stray > 0 {
            format!(", {} non-protocol line(s) ignored", self.stray)
        } else {
            String::new()
        };
        let mut lines = vec![format!(
            "{}: {}{noise} · {}",
            self.name,
            self.counts(),
            self.endpoint
        )];
        lines.extend(self.offers());
        lines
    }
}

struct Session {
    client: Arc<McpClient>,
    /// `(name the model sees, name the server knows)`, in the server's order.
    tools: Vec<(String, String)>,
    resources: Vec<McpResource>,
    prompts: Vec<McpPrompt>,
    /// A set the server does not have and why, in words — so a session showing
    /// no resources says whether that was declared, asked for and refused, or
    /// never asked at all.
    refused: Vec<String>,
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
            let running = self
                .sessions
                .lock()
                .await
                .get(&key)
                .map(|session| SessionView::of(session).report(&spec.name, "already running"));
            if let Some(report) = running {
                reports.push(report);
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
            offers: Vec::new(),
        };
        let client = match McpClient::start(spec, timeout).await {
            Ok(client) => client,
            Err(e) => return failed(e.to_string()),
        };

        // A server that never said it had tools is not a broken one — a server
        // offering only documents, or only prompts, is a working connection. So
        // the refusal to send `tools/list` is recorded and the session stays up.
        // What is fatal is a server that claimed the set and then failed to
        // answer for it: with no tools, resources or prompts we have nothing.
        let mut refused: Vec<String> = Vec::new();
        let tools = match client.list_tools().await {
            Ok(tools) => tools,
            Err(e @ McpError::NotOffered { .. }) => {
                refused.push(capability_note("tools", &e));
                Vec::new()
            }
            Err(e) => {
                client.shutdown().await;
                return failed(format!("started, but {}; it is not running", e));
            }
        };
        let resources = match client.list_resources().await {
            Ok(resources) => resources,
            Err(e) => {
                refused.push(capability_note("resources", &e));
                Vec::new()
            }
        };
        let prompts = match client.list_prompts().await {
            Ok(prompts) => prompts,
            Err(e) => {
                refused.push(capability_note("prompts", &e));
                Vec::new()
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
        let session = Session {
            client: Arc::new(client),
            tools: named,
            resources,
            prompts,
            refused,
        };
        let view = SessionView::of(&session);
        let detail = if clashes > 0 {
            format!("{}, {clashes} dropped for a clashing name", view.counts())
        } else {
            view.counts()
        };
        let report = Report {
            server: spec.name.clone(),
            connected: true,
            tools: session.tools.len(),
            detail,
            offers: view.offers(),
        };
        self.add_session(key, session, definitions).await;
        report
    }

    async fn add_session(&self, key: String, session: Session, definitions: Vec<ToolDefinition>) {
        let previous = {
            let mut sessions = self.sessions.lock().await;
            sessions.insert(key.clone(), session)
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

    /// Read one resource out of a running server by the uri the server itself
    /// listed. A set the handshake did not declare is refused here too, in the
    /// same words as everywhere else, so `/mcp read` cannot ask a tools-only
    /// server for a document.
    pub async fn read_resource(&self, server: &str, uri: &str) -> Result<String, String> {
        let key = sanitize(server);
        let client = {
            let sessions = self.sessions.lock().await;
            sessions.get(&key).map(|s| Arc::clone(&s.client))
        };
        let Some(client) = client else {
            return Err(format!(
                "MCP server `{server}` is not running — start it with /mcp"
            ));
        };
        let contents = client.read_resource(uri).await.map_err(|e| e.to_string())?;
        if contents.is_empty() {
            return Ok(format!(
                "MCP server `{server}` read {uri} and it was empty."
            ));
        }
        Ok(contents
            .iter()
            .map(|content| content.rendered())
            .collect::<Vec<_>>()
            .join("\n"))
    }

    /// Ask a running server for one of its prompts, with the arguments it asked
    /// for as `name=value` pairs. The messages come back with the role the
    /// server gave each one.
    pub async fn get_prompt(
        &self,
        server: &str,
        name: &str,
        arguments: &[(String, String)],
    ) -> Result<String, String> {
        let key = sanitize(server);
        let client = {
            let sessions = self.sessions.lock().await;
            sessions.get(&key).map(|s| Arc::clone(&s.client))
        };
        let Some(client) = client else {
            return Err(format!(
                "MCP server `{server}` is not running — start it with /mcp"
            ));
        };
        let messages = client
            .get_prompt(name, arguments)
            .await
            .map_err(|e| e.to_string())?;
        if messages.is_empty() {
            return Ok(format!(
                "MCP server `{server}` has no prompt called `{name}`."
            ));
        }
        Ok(messages
            .iter()
            .map(|message| format!("{}: {}", message.role, message.text))
            .collect::<Vec<_>>()
            .join("\n"))
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

    /// One line per running server for `/mcp`: what it has, where it is reached,
    /// what non-protocol noise it has printed, and then every resource and
    /// prompt it offered by name. An endpoint's credentials are masked the way a
    /// provider key is, so this is safe to read aloud. No servers means a line
    /// saying so.
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
            .flat_map(|session| SessionView::of(session).lines())
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
pub(crate) mod tests {
    // The live-server tests spawn `#!/bin/sh` fixtures and run only on Unix, so
    // the fixtures themselves are unused when the module is built anywhere else.
    #![cfg_attr(not(unix), allow(dead_code, unused_imports))]
    use super::*;
    #[cfg(unix)]
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
    /// No ports, nothing installed. It declares all three sets, because that is
    /// what a person points a server at for.
    const FIXTURE: &str = r#"#!/bin/sh
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    case "$line" in
        *'"method":"initialize"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{},"resources":{},"prompts":{}},"serverInfo":{"name":"fixture","version":"0"}}}\n' "$id"
            ;;
        *'"method":"tools/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Returns its argument","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},{"name":"echo","description":"Shadow of the first, same visible name"}]}}\n' "$id"
            ;;
        *'"method":"tools/call"'*)
            text=$(printf '%s' "$line" | sed -n 's/.*"text":"\([^"]*\)".*/\1/p')
            printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"echo: %s"}]}}\n' "$id" "$text"
            ;;
        *'"method":"resources/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"resources":[{"uri":"file:///notes/todo.md","name":"todo","description":"What is left","mimeType":"text/markdown"}]}}\n' "$id"
            ;;
        *'"method":"resources/read"'*)
            uri=$(printf '%s' "$line" | sed -n 's/.*"uri":"\([^"]*\)".*/\1/p')
            case "$uri" in
                *todo.md)
                    printf '{"jsonrpc":"2.0","id":%s,"result":{"contents":[{"uri":"%s","mimeType":"text/markdown","text":"- fix the mask"}]}}\n' "$id" "$uri"
                    ;;
                *)
                    printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32002,"message":"no such resource"}}\n' "$id"
                    ;;
            esac
            ;;
        *'"method":"prompts/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"prompts":[{"name":"review","description":"Read a patch","arguments":[{"name":"patch","description":"the diff","required":true},{"name":"style"}]}]}}\n' "$id"
            ;;
        *'"method":"prompts/get"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"messages":[{"role":"user","content":{"type":"text","text":"review this patch"}}]}}\n' "$id"
            ;;
    esac
done
"#;

    /// A server with documents and no tools and no prompts: the session has to
    /// survive being asked for the sets it never claimed.
    const DOCUMENTS_FIXTURE: &str = r#"#!/bin/sh
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    case "$line" in
        *'"method":"initialize"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{"resources":{}},"serverInfo":{"name":"docs","version":"0"}}}\n' "$id"
            ;;
        *'"method":"resources/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"resources":[{"uri":"handbook://sre","name":"sre handbook"}]}}\n' "$id"
            ;;
        *'"method":"resources/read"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"contents":[{"uri":"handbook://sre","text":"page one"}]}}\n' "$id"
            ;;
    esac
done
"#;

    /// Write a fixture where no other test can collide with it.
    #[cfg(unix)]
    fn fixture_server(name: &str, script: &str) -> (PathBuf, ServerSpec) {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let unique = format!(
            "xencode-mcp-hub-{}-{}-{name}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let dir = std::env::temp_dir().join(&unique);
        std::fs::create_dir_all(&dir).expect("create fixture dir");
        let path = dir.join("server.sh");
        std::fs::write(&path, script).expect("write fixture");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .expect("make fixture executable");
        let spec = ServerSpec::new(name, "/bin/sh").args(&[path.to_string_lossy().to_string()]);
        (dir, spec)
    }

    #[cfg(unix)]
    fn fixture_spec() -> (PathBuf, ServerSpec) {
        fixture_server("fixture", FIXTURE)
    }

    /// The documents-only server, for a test in another module that drives the
    /// real `/mcp` command handler against a server that is actually running.
    #[cfg(unix)]
    pub(crate) fn live_documents_server() -> (PathBuf, ServerSpec) {
        fixture_server("docs", DOCUMENTS_FIXTURE)
    }

    #[tokio::test]
    #[cfg(unix)]
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
        assert_eq!(status.len(), 3, "{status:?}");
        assert!(
            status[0].starts_with("fixture: 1 tool(s), 1 resource(s), 1 prompt(s) · /bin/sh "),
            "the line names how the server is reached: {}",
            status[0]
        );
        // A count is not something a person can act on; the names are. The
        // resource is shown by the uri it has to be read with, and the prompt by
        // the arguments it wants, with the optional one bracketed.
        assert_eq!(status[1], "  resource file:///notes/todo.md — todo");
        assert_eq!(status[2], "  prompt review (patch= [style=])");

        // And both are readable from the session that listed them.
        assert_eq!(
            hub.read_resource("fixture", "file:///notes/todo.md").await,
            Ok("- fix the mask".to_string())
        );
        assert_eq!(
            hub.get_prompt(
                "fixture",
                "review",
                &[("patch".into(), "@@ -1 +1 @@".into())]
            )
            .await,
            Ok("user: review this patch".to_string())
        );
        let nowhere = hub.read_resource("fixture", "file:///no/such").await;
        assert!(nowhere.is_err(), "{nowhere:?}");

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

    /// A server that offers documents and prompts and no tools is a working
    /// connection, not a failed one: `tools/list` is never sent, the session
    /// stays up, and what it does have is listed by name.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_server_without_tools_stays_connected_and_says_it_has_none() {
        let (dir, spec) = fixture_server("docs", DOCUMENTS_FIXTURE);
        let hub = McpHub::new();
        let reports = hub
            .connect(std::slice::from_ref(&spec), Duration::from_secs(5))
            .await;
        assert!(reports[0].connected, "{:?}", reports[0]);
        assert_eq!(reports[0].tools, 0, "{:?}", reports[0]);
        assert_eq!(
            reports[0].detail, "no tools, 1 resource(s)",
            "{:?}",
            reports[0]
        );
        assert_eq!(
            reports[0].offers,
            [
                "  tools — not declared by this server".to_string(),
                "  prompts — not declared by this server".to_string(),
                "  resource handbook://sre — sre handbook".to_string(),
            ],
            "{:?}",
            reports[0].offers
        );
        // Nothing to offer the model, and that is the server's own doing.
        assert!(hub.definitions().is_empty());
        // The session is live: its document reads back over the same pipes.
        assert_eq!(
            hub.read_resource("docs", "handbook://sre").await,
            Ok("page one".to_string())
        );
        // A prompt it never declared is refused before anything is sent.
        assert!(hub
            .get_prompt("docs", "standup", &[])
            .await
            .unwrap_err()
            .contains("does not offer prompts"));
        std::fs::remove_dir_all(&dir).expect("clean fixture dir");
    }

    /// Asking a server that is not running for one of its documents is refused
    /// in the same words everywhere else uses, not by a panic or a hang.
    #[tokio::test]
    async fn reading_from_a_server_that_is_not_running_names_it() {
        let hub = McpHub::new();
        assert!(hub
            .read_resource("absent", "file:///x")
            .await
            .unwrap_err()
            .contains("absent"));
        assert!(hub
            .get_prompt("absent", "review", &[])
            .await
            .unwrap_err()
            .contains("/mcp"));
    }

    /// A config entry is turned into the transport it names: a command to spawn
    /// with its arguments and environment, or an address to post to with the
    /// headers it is authenticated by.
    #[test]
    fn a_spawned_server_and_a_hosted_one_become_different_specs() {
        let spawned = spec_from_config(
            "docs",
            &xencode_config_rs::McpServer {
                command: Some("mcp-docs".into()),
                args: vec!["--root".into(), "/docs".into()],
                env: std::collections::BTreeMap::from([("DOCS_TOKEN".into(), "t".into())]),
                ..Default::default()
            },
        )
        .expect("a command is a way to reach a server");
        assert_eq!(
            spawned.transport,
            Transport::Stdio {
                command: "mcp-docs".to_string(),
                args: vec!["--root".to_string(), "/docs".to_string()],
                env: vec![("DOCS_TOKEN".to_string(), "t".to_string())],
            }
        );

        let hosted = spec_from_config(
            "hosted",
            &xencode_config_rs::McpServer {
                url: Some("https://mcp.example.com/v1/mcp".into()),
                headers: std::collections::BTreeMap::from([(
                    "authorization".into(),
                    "Bearer p-1".into(),
                )]),
                ..Default::default()
            },
        )
        .expect("a url is a way to reach a server");
        assert_eq!(
            hosted.transport,
            Transport::Http {
                url: "https://mcp.example.com/v1/mcp".to_string(),
                headers: vec![("authorization".to_string(), "Bearer p-1".to_string())],
            }
        );
        // An argument belongs to a process. Handing one to an endpoint would be
        // a silent nothing, so the builder leaves a url spec as it is.
        assert_eq!(hosted.clone().args(&["--root".into()]), hosted);
    }

    /// A declaration that cannot be read as one server is refused in words, in
    /// the same places: nothing is spawned and nothing is posted.
    #[test]
    fn a_declaration_that_names_no_way_to_reach_the_server_is_refused() {
        for (name, server) in [
            (
                "empty",
                xencode_config_rs::McpServer {
                    ..Default::default()
                },
            ),
            (
                "both",
                xencode_config_rs::McpServer {
                    command: Some("npx".into()),
                    url: Some("https://example.com/mcp".into()),
                    ..Default::default()
                },
            ),
        ] {
            let problem = spec_from_config(name, &server)
                .err()
                .unwrap_or_else(|| panic!("{name} declares no way to reach a server"));
            assert!(
                problem.contains("\"command\"") && problem.contains("\"url\""),
                "{name}: {problem}"
            );
        }
    }
}
