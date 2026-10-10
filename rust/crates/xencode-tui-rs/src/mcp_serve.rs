//! xencode as an MCP server: `xencode mcp serve` (M-5).
//!
//! The client side (`crate::mcp`) speaks to somebody else's tools; this is the
//! other direction — an editor, script, or agent connecting to *us* over a pipe
//! and asking for `read_file` and friends.
//!
//! The one thing that makes this different from the TUI is who approves. A
//! client of a stdio server has no prompt to answer, so there is nobody to say
//! yes to a write. Rather than hang every mutating call or wave it through, the
//! server decides with [`HeadlessPolicy`]: read-only tools run, everything else
//! is refused unless the operator named it on the command line, and a refusal
//! says which flag would have allowed it. Nothing here consults or alters the
//! interactive gate — see [`crate::agent_tools::classify`], which stays exactly
//! as strict as it was.

use std::path::PathBuf;
use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock, Implementation, InitializeResult,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, ServiceExt};
use xencode_providers_rs::{command_tools, file_tools, ToolCall, ToolDefinition};

use crate::agent_tools::{
    execute_tool_call, tool_class, Headless, HeadlessPolicy, TaskRuntime, ToolClass,
};
use crate::mcp::advertised_tool_name;

/// `serverInfo.name`, which is also the server name a client prefixes our tool
/// names with (`mcp__xencode__read_file` once it imports them).
pub const SERVER_NAME: &str = "xencode";

/// The six tools xencode publishes, in this order. Chosen because they are the
/// ones a caller outside the TUI would reasonably want and because every one of
/// them has a schema the agent loop already honours; nothing new is invented for
/// the server.
const EXPOSED: [&str; 6] = [
    "read_file",
    "list_dir",
    "search_files",
    "write_file",
    "edit_file",
    "run_command",
];

/// The names as published, each already inside the shared length limit.
pub fn exposed_tool_names() -> Vec<String> {
    EXPOSED
        .iter()
        .map(|name| advertised_tool_name(name))
        .collect()
}

/// The schema of one published tool, taken from the same definition the model
/// is given — the server does not keep a second copy that can disagree.
fn tool_definitions() -> Vec<ToolDefinition> {
    let all: Vec<ToolDefinition> = file_tools().into_iter().chain(command_tools()).collect();
    EXPOSED
        .iter()
        .filter_map(|want| all.iter().find(|def| &def.name == want).cloned())
        .collect()
}

/// What one published call produced, before it becomes anything protocol-shaped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub text: String,
    /// The call did not do what it was asked — refused, unknown tool, or the
    /// tool's own failure. Maps to MCP `isError`, so a client sees a failed
    /// result rather than a successful one containing the word "error".
    pub failed: bool,
}

/// The server: a workspace, the launch-time permission decision, and the same
/// background-task runtime the agent loop uses so `run_command` is the real
/// thing rather than a second implementation.
pub struct XencodeServer {
    root: PathBuf,
    policy: HeadlessPolicy,
    tasks: TaskRuntime,
    version: String,
    /// The lead's team tools (TM-2), when started with `--team`.
    team: Option<std::sync::Arc<crate::mcp_team::TeamClient>>,
}

impl XencodeServer {
    pub fn new(
        root: PathBuf,
        version: impl Into<String>,
        policy: HeadlessPolicy,
        tasks: TaskRuntime,
    ) -> Self {
        Self {
            root,
            policy,
            tasks,
            version: version.into(),
            team: None,
        }
    }

    /// Also publish the lead's team tools (TM-2).
    pub fn with_team(mut self) -> Self {
        self.team = Some(crate::mcp_team::TeamClient::new(&self.root));
        self
    }

    /// What `tools/list` returns: the agent loop's own name, description and
    /// JSON Schema for each tool, with the name put through the shared length
    /// limit and the read-only hint set from the same tool class the approval
    /// gate uses. Whether a call will be *refused* is not a property of the tool
    /// and does not go here — the handshake instructions say what this launch
    /// permits.
    pub fn advertised_tools(&self) -> Vec<Tool> {
        let team = if self.team.is_some() {
            crate::mcp_team::team_tool_definitions()
        } else {
            Vec::new()
        };
        tool_definitions()
            .into_iter()
            .chain(team)
            .map(|def| {
                let schema = match def.parameters {
                    serde_json::Value::Object(map) => Arc::new(map),
                    _ => Arc::new(serde_json::Map::new()),
                };
                let mut tool = Tool::new(advertised_tool_name(&def.name), def.description, schema);
                let read_only = if def.name.starts_with("team_") {
                    matches!(
                        def.name.as_str(),
                        "team_agents" | "team_status" | "team_result"
                    )
                } else {
                    tool_class(&def.name) == ToolClass::ReadOnly
                };
                tool.annotations = Some(rmcp::model::ToolAnnotations::new().read_only(read_only));
                tool
            })
            .collect()
    }

    /// The whole of one call: decide, then do. Kept free of protocol types so
    /// the gate can be exercised without a client on the other end of the pipe.
    pub async fn invoke(
        &self,
        name: &str,
        arguments: serde_json::Map<String, serde_json::Value>,
    ) -> Reply {
        if let Some(team) = &self.team {
            if crate::mcp_team::TEAM_TOOLS.contains(&name) {
                return match team.call(name, &arguments).await {
                    Ok(text) => Reply {
                        text,
                        failed: false,
                    },
                    Err(text) => Reply { text, failed: true },
                };
            }
        }
        if !EXPOSED.contains(&name) {
            return Reply {
                text: format!(
                    "xencode does not publish a tool called `{name}`; it publishes {}",
                    EXPOSED.join(", ")
                ),
                failed: true,
            };
        }
        match self.policy.decide(&self.root, name, &arguments) {
            Headless::Refused { reason } => Reply {
                text: reason,
                failed: true,
            },
            Headless::Allow => {
                let call = ToolCall {
                    id: format!("mcp-{name}"),
                    name: name.to_string(),
                    arguments: serde_json::Value::Object(arguments),
                };
                let text = execute_tool_call(&self.tasks, &self.root, &call).await;
                // The executor reports a failure as `error: …` rather than a
                // result type; a client deserves the difference.
                let failed = text.starts_with("error:");
                Reply { text, failed }
            }
        }
    }
}

impl ServerHandler for XencodeServer {
    fn get_info(&self) -> InitializeResult {
        // What this launch will actually run, said once at handshake instead of
        // only when a call gets refused.
        let instructions = if self.policy.is_read_only() {
            format!(
                "{SERVER_NAME} is serving {} read-only. Calls that would change files or \
                 run a command are refused; the server can be restarted with --allow <tool> \
                 for a named tool. A `path` or `cwd` outside that directory is always refused.",
                self.root.display()
            )
        } else {
            format!(
                "{SERVER_NAME} is serving {} with writes permitted for the tools named \
                 at launch. A `path` or `cwd` reaching outside that directory is refused; \
                 a permitted command is run as the caller wrote it and is not checked \
                 that way.",
                self.root.display()
            )
        };
        InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(SERVER_NAME, self.version.clone()))
            .with_instructions(instructions)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        let result = ListToolsResult {
            tools: self.advertised_tools(),
            ..Default::default()
        };
        Ok(result)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, McpError> {
        let arguments = request.arguments.unwrap_or_default();
        let reply = self.invoke(&request.name, arguments).await;
        let blocks = vec![ContentBlock::text(reply.text)];
        let result = if reply.failed {
            // A refusal the caller should read is an error *result*, not a
            // protocol error: clients render the latter opaquely and the reason
            // would be lost.
            CallToolResult::error(blocks)
        } else {
            CallToolResult::success(blocks)
        };
        Ok(result.into())
    }
}

/// Run the server until the client closes the pipe. Errors come back as strings
/// because the CLI prints them; stdout is the protocol channel, so nothing here
/// may write to it.
pub async fn serve(
    root: PathBuf,
    version: &str,
    policy: HeadlessPolicy,
    tasks: TaskRuntime,
    team: bool,
) -> Result<(), String> {
    let server = XencodeServer::new(root, version, policy, tasks);
    let server = if team { server.with_team() } else { server };
    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|error| format!("could not start the MCP server on stdio: {error}"))?;
    service
        .waiting()
        .await
        .map_err(|error| format!("the MCP server stopped unexpectedly: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_tools::new_task_runtime;

    /// A real directory with a real file in it: every assertion below is about
    /// bytes that exist on disk.
    fn workspace(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "xencode-mcp-serve-{label}-{}-{}",
            std::process::id(),
            unique_counter()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("hello.txt"), "one\ntwo\nthree\n").unwrap();
        dir
    }

    fn unique_counter() -> u32 {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        N.fetch_add(1, Ordering::SeqCst)
    }

    fn server(root: PathBuf, allowed: &[&str]) -> XencodeServer {
        XencodeServer::new(
            root,
            "0.1.0-test",
            HeadlessPolicy::new(allowed.iter().map(|name| name.to_string())),
            new_task_runtime(),
        )
    }

    fn args(json: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        json.as_object().unwrap().clone()
    }

    #[tokio::test]
    async fn publishes_the_six_tools_with_real_schemas() {
        let dir = workspace("list");
        let tools = server(dir.clone(), &[]).advertised_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        assert_eq!(
            names,
            [
                "read_file",
                "list_dir",
                "search_files",
                "write_file",
                "edit_file",
                "run_command"
            ]
        );
        for tool in &tools {
            // A schema that is not an object makes an OpenAI-style request
            // invalid, and every one of these tools takes named arguments.
            assert_eq!(
                tool.input_schema.get("type").and_then(|v| v.as_str()),
                Some("object"),
                "{} has no object schema",
                tool.name
            );
            assert!(
                tool.description
                    .as_ref()
                    .is_some_and(|d| !d.trim().is_empty()),
                "{} publishes no description",
                tool.name
            );
        }
        // The description is the agent-loop one, not a rewritten copy.
        let read_file = tools.iter().find(|t| t.name == "read_file").unwrap();
        assert!(
            read_file
                .description
                .as_ref()
                .unwrap()
                .contains("crate:<name>"),
            "the schema drifted from what the model is given"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_read_executes_for_real_and_returns_the_file() {
        let dir = workspace("read");
        let reply = server(dir.clone(), &[])
            .invoke("read_file", args(serde_json::json!({"path": "hello.txt"})))
            .await;
        assert!(!reply.failed, "{}", reply.text);
        // The numbered-line render the agent loop produces, from the real file.
        assert_eq!(reply.text, "1\tone\n2\ttwo\n3\tthree");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn default_launch_refuses_every_write_and_says_which_flag_fixes_it() {
        let dir = workspace("refuse");
        let s = server(dir.clone(), &[]);
        let reply = s
            .invoke(
                "write_file",
                args(serde_json::json!({"path": "new.txt", "content": "hi"})),
            )
            .await;
        assert!(reply.failed);
        assert!(reply.text.contains("--allow write_file"), "{}", reply.text);
        // Refused means untouched: the file is not there.
        assert!(!dir.join("new.txt").exists());

        let reply = s
            .invoke(
                "run_command",
                args(serde_json::json!({"command": "touch pwned"})),
            )
            .await;
        assert!(reply.failed && reply.text.contains("--allow run_command"));
        assert!(!dir.join("pwned").exists());

        // An unknown tool is not quietly allowed; it is named back.
        let reply = s
            .invoke("delete_everything", args(serde_json::json!({})))
            .await;
        assert!(reply.failed);
        assert!(reply.text.contains("does not publish"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_allowed_tool_actually_writes_the_file() {
        let dir = workspace("allow");
        let s = server(dir.clone(), &["write_file"]);
        let reply = s
            .invoke(
                "write_file",
                args(serde_json::json!({"path": "made.txt", "content": "written\n"})),
            )
            .await;
        assert!(!reply.failed, "{}", reply.text);
        assert_eq!(
            std::fs::read_to_string(dir.join("made.txt")).unwrap(),
            "written\n"
        );
        // Still only that tool: naming write_file did not open the shell.
        let reply = s
            .invoke("run_command", args(serde_json::json!({"command": "true"})))
            .await;
        assert!(reply.failed && reply.text.contains("--allow run_command"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_path_outside_the_workspace_is_refused_even_for_an_allowed_tool() {
        let dir = workspace("escape");
        let outside = format!(
            "../{}-outside.txt",
            dir.file_name().unwrap().to_str().unwrap()
        );
        let s = server(dir.clone(), &["write_file", "run_command"]);
        let reply = s
            .invoke(
                "write_file",
                args(serde_json::json!({"path": outside, "content": "x"})),
            )
            .await;
        assert!(reply.failed);
        assert!(
            reply.text.contains("outside this workspace"),
            "{}",
            reply.text
        );
        let reply = s
            .invoke(
                "run_command",
                args(serde_json::json!({"command": "id", "cwd": "/etc"})),
            )
            .await;
        assert!(reply.failed, "a permitted tool must not move the boundary");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_command_that_fails_reports_its_exit_rather_than_being_a_refusal() {
        let dir = workspace("fail");
        let s = server(dir.clone(), &["run_command"]);
        let reply = s
            .invoke(
                "run_command",
                args(serde_json::json!({"command": "echo oops >&2; exit 4"})),
            )
            .await;
        // The tool ran, so this is not a policy refusal — no flag would change
        // the answer, and the caller must not be told one would.
        assert!(!reply.text.contains("--allow"), "{}", reply.text);
        assert!(!reply.text.starts_with("error:"), "{}", reply.text);
        // The command's own outcome reaches the client in the executor's words:
        // the exit code and what it printed to stderr.
        assert!(reply.text.contains("exit 4"), "{}", reply.text);
        assert!(reply.text.contains("oops"), "{}", reply.text);

        // A call the executor itself could not make is the other case, and it
        // does come back as a failed result.
        let reply = s
            .invoke(
                "read_file",
                args(serde_json::json!({"path": "no-such.txt"})),
            )
            .await;
        assert!(reply.failed, "{}", reply.text);
        assert!(reply.text.starts_with("error:"), "{}", reply.text);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn the_read_only_hint_describes_the_tool_not_the_launch() {
        let dir = workspace("hint");
        let read_only = |tools: Vec<Tool>, name: &str| {
            tools
                .iter()
                .find(|t| t.name == name)
                .unwrap()
                .annotations
                .as_ref()
                .unwrap()
                .read_only_hint
        };
        let locked = server(dir.clone(), &[]).advertised_tools();
        assert_eq!(read_only(locked.clone(), "read_file"), Some(true));
        assert_eq!(
            read_only(locked, "write_file"),
            Some(false),
            "a refused write is still announced as a write, so the client knows what it asked for"
        );
        let open = server(dir.clone(), &["write_file"]).advertised_tools();
        assert_eq!(
            read_only(open, "write_file"),
            Some(false),
            "allowing the tool changes what happens, not what the tool is"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_handshake_says_what_this_launch_will_actually_run() {
        let dir = workspace("info");
        let locked = ServerHandler::get_info(&server(dir.clone(), &[]));
        let text = locked.instructions.clone().unwrap_or_default();
        assert!(text.contains("read-only"), "{text}");
        assert!(!text.contains("with writes permitted"), "{text}");

        let open = ServerHandler::get_info(&server(dir.clone(), &["write_file"]));
        let text = open.instructions.clone().unwrap_or_default();
        assert!(text.contains("with writes permitted"), "{text}");
        assert!(text.contains("outside that directory is refused"), "{text}");

        // It names the directory it is holding, and identifies itself as the
        // server a client will prefix tool names with.
        assert_eq!(locked.server_info.name, SERVER_NAME);
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            locked
                .instructions
                .as_ref()
                .unwrap()
                .contains(&dir.display().to_string()),
            "the handshake does not say which workspace it means"
        );
        assert!(locked.capabilities.tools.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn published_names_fit_the_limit_the_client_publishes_under() {
        for name in exposed_tool_names() {
            assert!(
                name.len() <= crate::mcp::MAX_TOOL_NAME,
                "{name} is longer than a provider accepts"
            );
        }
        // The client prefixes what we publish, so the two rules have to hold
        // together: a long name is cut the same way by both, and a name with
        // characters a provider rejects is sanitized rather than dropped.
        let ugly = "read file/ünïcode!".repeat(6);
        let published = advertised_tool_name(&ugly);
        assert_eq!(published.len(), crate::mcp::MAX_TOOL_NAME);
        assert!(
            published
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "{published}"
        );
        let imported = crate::mcp::full_tool_name(SERVER_NAME, &published);
        assert_eq!(
            imported.len(),
            crate::mcp::MAX_TOOL_NAME,
            "importing our own published name overflows the limit"
        );
        assert_eq!(
            crate::mcp::split_full_name(&imported).unwrap().0,
            SERVER_NAME
        );
    }

    #[test]
    fn worker_adapter_registry_resolves_mcp_adapter() {
        let registry = WorkerAdapterRegistry::default();
        assert!(registry.available().contains(&"mcp"));
        let adapter = registry.get("mcp").expect("mcp adapter must be mounted");
        assert_eq!(adapter.id(), "mcp");
        assert_eq!(adapter.protocol(), "mcp-stdio-v1");
        assert_eq!(adapter.exposed_tools().len(), 6);
    }
}

/// A worker adapter interface exposing tool execution capabilities (M-5, AF-3).
pub trait WorkerAdapter: Send + Sync {
    /// Identifier of the adapter (e.g. "mcp").
    fn id(&self) -> &'static str;
    /// Description of the adapter.
    fn description(&self) -> &'static str;
    /// Protocol used by the adapter.
    fn protocol(&self) -> &'static str;
    /// List tools exposed by this adapter.
    fn exposed_tools(&self) -> Vec<String>;
}

/// Statically linked MCP stdio worker adapter (M-5).
#[derive(Debug, Default, Clone)]
pub struct McpWorkerAdapter;

impl WorkerAdapter for McpWorkerAdapter {
    fn id(&self) -> &'static str {
        "mcp"
    }

    fn description(&self) -> &'static str {
        "Model Context Protocol stdio server exposing workspace tools"
    }

    fn protocol(&self) -> &'static str {
        "mcp-stdio-v1"
    }

    fn exposed_tools(&self) -> Vec<String> {
        exposed_tool_names()
    }
}

/// Statically linked mount point for worker adapters (AF-3).
pub struct WorkerAdapterRegistry {
    adapters: std::collections::BTreeMap<&'static str, Box<dyn WorkerAdapter>>,
}

impl Default for WorkerAdapterRegistry {
    fn default() -> Self {
        let mut reg = Self::new();
        reg.register(Box::new(McpWorkerAdapter));
        reg
    }
}

impl WorkerAdapterRegistry {
    pub fn new() -> Self {
        Self {
            adapters: std::collections::BTreeMap::new(),
        }
    }

    pub fn register(&mut self, adapter: Box<dyn WorkerAdapter>) {
        self.adapters.insert(adapter.id(), adapter);
    }

    pub fn get(&self, id: &str) -> Option<&dyn WorkerAdapter> {
        self.adapters.get(id).map(|a| a.as_ref())
    }

    pub fn available(&self) -> Vec<&'static str> {
        self.adapters.keys().copied().collect()
    }
}
