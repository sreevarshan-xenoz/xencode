//! The wire shapes of Model Context Protocol, as far as xencode uses it:
//! JSON-RPC 2.0 requests and the methods behind tools, resources and prompts
//! (`initialize` / `tools/list` / `tools/call` / `resources/list` /
//! `resources/read` / `prompts/list` / `prompts/get`).
//!
//! Everything here is pure functions over [`serde_json::Value`] so the parsing
//! is testable without a process on the other end of a pipe. Servers in the
//! wild disagree about key spellings, so the readers accept both the spec form
//! and the snake_case drift, and a tool we cannot make sense of is dropped
//! rather than guessed at.

use serde_json::{json, Value};

/// Newest protocol revision we claim to speak. A server that cannot match it
/// replies with its own `protocolVersion`; we do not enforce either value,
/// because the tool methods are identical across the 2024-11-05 / 2025-03-26 /
/// 2025-06-18 revisions xencode targets.
pub const PROTOCOL_VERSION: &str = "2025-03-26";

/// `clientInfo.name` — what a server log shows as its caller.
pub const CLIENT_NAME: &str = "xencode";

/// One tool a server offers. `input_schema` is the server's own JSON Schema,
/// passed through to the model untouched: it is not ours to simplify.
#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// One of the three method sets a server can declare. Named in a handshake so
/// the client knows which calls to send and the user knows what a server
/// offered when it did not offer much.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    Tools,
    Resources,
    Prompts,
}

impl Feature {
    /// The key this set is declared under in `capabilities`, and the word a
    /// refusal is written with.
    pub fn key(self) -> &'static str {
        match self {
            Feature::Tools => "tools",
            Feature::Resources => "resources",
            Feature::Prompts => "prompts",
        }
    }
}

/// What the server answered it can do.
///
/// Resources and prompts belong to the *server*: a client cannot get them by
/// declaring them, only by asking once the handshake says they are there.
/// `declared_any` keeps the two ways of saying "nothing" apart — a server that
/// named its capabilities and left resources out genuinely has none, while one
/// that sent an empty or absent block has told us nothing and is still worth a
/// try. Refusing the second case would break connections that work today.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ServerCapabilities {
    pub tools: bool,
    pub resources: bool,
    pub prompts: bool,
    /// Whether the handshake declared any capability block at all.
    pub declared_any: bool,
}

impl ServerCapabilities {
    /// Whether to send a method of this set. Silence from the server is not a
    /// refusal, so an undeclared handshake lets the call through and the
    /// server's answer settles it.
    pub fn offers(&self, feature: Feature) -> bool {
        if !self.declared_any {
            return true;
        }
        match feature {
            Feature::Tools => self.tools,
            Feature::Resources => self.resources,
            Feature::Prompts => self.prompts,
        }
    }

    /// The sets this server named, as `["tools", "resources"]`, in the order a
    /// reader expects them. A server that declared nothing reads back as
    /// nothing, which is the truth rather than a guess.
    pub fn declared(&self) -> Vec<&'static str> {
        let mut named: Vec<&'static str> = Vec::new();
        if self.tools {
            named.push(Feature::Tools.key());
        }
        if self.resources {
            named.push(Feature::Resources.key());
        }
        if self.prompts {
            named.push(Feature::Prompts.key());
        }
        named
    }
}

pub fn request(id: u64, method: &str, params: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    })
}

/// A JSON-RPC message that expects no answer (`notifications/initialized`).
pub fn notification(method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "method": method, "params": params })
}

pub fn initialize_params(client_version: &str) -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        // Client capabilities are roots, sampling and elicitation — the three
        // ways a server can call back into us. xencode answers none of them, so
        // declaring them would invite requests we would drop on the floor.
        // Tools, resources and prompts are declared by the *server* and read
        // back with [`parse_server_capabilities`].
        "capabilities": {},
        "clientInfo": { "name": CLIENT_NAME, "version": client_version },
    })
}

pub fn call_tool_params(name: &str, arguments: Value) -> Value {
    json!({ "name": name, "arguments": arguments })
}

pub fn read_resource_params(uri: &str) -> Value {
    json!({ "uri": uri })
}

/// `prompts/get` parameters. The arguments are string-valued by spec, so they
/// travel as pairs rather than as a JSON value the caller has to shape. An
/// argument-less prompt still gets an object: a strict server rejects `null`.
pub fn get_prompt_params(name: &str, arguments: &[(String, String)]) -> Value {
    let mut sent = serde_json::Map::new();
    for (key, value) in arguments {
        sent.insert(key.clone(), Value::String(value.clone()));
    }
    json!({ "name": name, "arguments": Value::Object(sent) })
}

/// Read `result.capabilities` from an `initialize` response.
pub fn parse_server_capabilities(result: &Value) -> ServerCapabilities {
    let declared = match result.get("capabilities") {
        Some(Value::Object(map)) => map,
        _ => return ServerCapabilities::default(),
    };
    // A key carried with a non-null value is a declaration, whether its body is
    // `{}` or `{"listChanged": true}`; `"resources": null` is not one.
    let named = |key: &str| declared.get(key).is_some_and(|v| !v.is_null());
    ServerCapabilities {
        tools: named(Feature::Tools.key()),
        resources: named(Feature::Resources.key()),
        prompts: named(Feature::Prompts.key()),
        declared_any: declared.values().any(|v| !v.is_null()),
    }
}

/// Read `result.tools[]` from a `tools/list` response.
pub fn parse_tools(result: &Value) -> Vec<McpTool> {
    let Some(entries) = result.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let name = entry.get("name").and_then(Value::as_str)?;
            if name.trim().is_empty() {
                return None;
            }
            Some(McpTool {
                name: name.to_string(),
                description: entry
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                input_schema: entry
                    .get("inputSchema")
                    .or_else(|| entry.get("input_schema"))
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object"})),
            })
        })
        .collect()
}

/// One resource a server exposes. `uri` is what `resources/read` is called
/// with, so it is kept exactly as the server wrote it.
#[derive(Debug, Clone, PartialEq)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    pub mime_type: String,
}

/// Read `result.resources[]` from a `resources/list` response. A resource with
/// no uri is not readable, so it is dropped rather than shown as one.
pub fn parse_resources(result: &Value) -> Vec<McpResource> {
    let Some(entries) = result.get("resources").and_then(Value::as_array) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let uri = entry.get("uri").and_then(Value::as_str)?;
            if uri.trim().is_empty() {
                return None;
            }
            Some(McpResource {
                uri: uri.to_string(),
                name: readable_name(entry, uri),
                description: text_of(entry, "description"),
                mime_type: entry
                    .get("mimeType")
                    .or_else(|| entry.get("mime_type"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            })
        })
        .collect()
}

/// The body of one resource from a `resources/read` response.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceContent {
    pub uri: String,
    pub mime_type: String,
    /// The text, when the server sent text. A `blob` is base64 bytes for a
    /// format no text model can read, so its length is kept and the bytes are
    /// not — handing them over would put noise in a context window.
    pub text: Option<String>,
    pub blob_len: Option<usize>,
}

impl ResourceContent {
    /// What a text-only viewer shows: the body, or the shape of what it cannot
    /// show. A binary resource never reads as an empty one. The number is the
    /// length of the base64 the server sent, not of the bytes behind it.
    pub fn rendered(&self) -> String {
        match &self.text {
            Some(text) => text.clone(),
            None => format!(
                "[{}: {} bytes of base64 not shown]",
                if self.mime_type.is_empty() {
                    "unknown type"
                } else {
                    &self.mime_type
                },
                self.blob_len.unwrap_or(0)
            ),
        }
    }
}

/// Read `result.contents[]` from a `resources/read` response. One uri can come
/// back as several pieces, so all of them are kept.
pub fn parse_resource_contents(result: &Value) -> Vec<ResourceContent> {
    let entries: Vec<Value> = match result.get("contents") {
        Some(Value::Array(entries)) => entries.clone(),
        // Some servers answer a single body where the spec wants a list.
        Some(object @ Value::Object(_)) => vec![object.clone()],
        _ => Vec::new(),
    };
    entries
        .iter()
        .filter_map(|entry| {
            let text = entry.get("text").and_then(Value::as_str);
            let blob = entry.get("blob").and_then(Value::as_str);
            if text.is_none() && blob.is_none() {
                return None;
            }
            Some(ResourceContent {
                uri: entry
                    .get("uri")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                mime_type: entry
                    .get("mimeType")
                    .or_else(|| entry.get("mime_type"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                text: text.map(str::to_string),
                blob_len: blob.map(str::len),
            })
        })
        .collect()
}

/// One argument a prompt wants. `required` drives what we can send at all.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptArgument {
    pub name: String,
    pub description: String,
    pub required: bool,
}

/// One prompt a server offers: a prepared message the user can ask for by name.
#[derive(Debug, Clone, PartialEq)]
pub struct McpPrompt {
    pub name: String,
    pub description: String,
    pub arguments: Vec<PromptArgument>,
}

/// Read `result.prompts[]` from a `prompts/list` response.
pub fn parse_prompts(result: &Value) -> Vec<McpPrompt> {
    let Some(entries) = result.get("prompts").and_then(Value::as_array) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let name = entry.get("name").and_then(Value::as_str)?;
            if name.trim().is_empty() {
                return None;
            }
            let arguments = entry
                .get("arguments")
                .and_then(Value::as_array)
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(|entry| {
                            let name = entry.get("name").and_then(Value::as_str)?;
                            Some(PromptArgument {
                                name: name.to_string(),
                                description: text_of(entry, "description"),
                                required: entry
                                    .get("required")
                                    .and_then(Value::as_bool)
                                    .unwrap_or(false),
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Some(McpPrompt {
                name: name.to_string(),
                description: text_of(entry, "description"),
                arguments,
            })
        })
        .collect()
}

/// One message a `prompts/get` produced, in the role the server gave it.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptMessage {
    pub role: String,
    pub text: String,
}

/// Read `result.messages[]` from a `prompts/get` response. A message whose
/// content is an image or an embedded resource is still returned, named the way
/// a tool result would be, so the user sees what came back rather than an
/// empty prompt.
pub fn parse_prompt(result: &Value) -> Vec<PromptMessage> {
    let Some(messages) = result.get("messages").and_then(Value::as_array) else {
        return Vec::new();
    };
    messages
        .iter()
        .map(|message| PromptMessage {
            role: message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            text: render_block(message.get("content").unwrap_or(&Value::Null)),
        })
        .collect()
}

/// A `name` worth showing, falling back to the uri a resource is called by.
fn readable_name(entry: &Value, fallback: &str) -> String {
    match entry.get("name").and_then(Value::as_str) {
        Some(name) if !name.trim().is_empty() => name.to_string(),
        _ => fallback.to_string(),
    }
}

fn text_of(entry: &Value, key: &str) -> String {
    entry
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// What a `tools/call` produced. `text` is what the model reads back.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutcome {
    pub text: String,
    /// The server said the call failed. Stated as a failure, never smoothed
    /// into a success string the model would act on.
    pub is_error: bool,
}
/// Flatten a `CallToolResult` into text for the model.
///
/// Only `text` blocks carry content a text model can use; everything else is
/// named and dropped, because silently discarding an image would let the model
/// believe it saw the tool's whole answer.
pub fn parse_tool_result(result: &Value) -> ToolOutcome {
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut parts: Vec<String> = result
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| blocks.iter().map(render_block).collect())
        .unwrap_or_default();

    if parts.is_empty() {
        if let Some(structured) = result.get("structuredContent") {
            if !structured.is_null() {
                parts.push(structured.to_string());
            }
        }
    }

    ToolOutcome {
        text: parts.join("\n"),
        is_error,
    }
}

/// One content block flattened to what a text model can read — the same rule a
/// tool result and a prompt message follow, so the two cannot drift apart.
/// Only `text` carries content a text model can use; everything else is named
/// and dropped, because silently discarding an image would let the model
/// believe it saw the whole answer.
fn render_block(block: &Value) -> String {
    match block.get("type").and_then(Value::as_str) {
        Some("text") => block
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        Some("image") => describe_binary("image", block),
        Some("audio") => describe_binary("audio", block),
        Some("resource") => describe_resource(block),
        Some("resource_link") => block
            .get("uri")
            .and_then(Value::as_str)
            .map(|uri| format!("[resource link: {uri}]"))
            .unwrap_or_else(|| "[resource link without uri]".to_string()),
        Some(other) => format!("[unsupported content: {other}]"),
        // A block with no discriminator at all: show its JSON rather than
        // pretend the call returned nothing.
        None => format!("[unrecognized content: {block}]"),
    }
}

fn describe_binary(kind: &str, block: &Value) -> String {
    let mime = block
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or("unknown type");
    let size = block
        .get("data")
        .and_then(Value::as_str)
        .map(|b64| b64.len())
        .unwrap_or(0);
    format!("[{kind}: {mime}, {size} bytes of base64 not shown]")
}

fn describe_resource(block: &Value) -> String {
    let Some(resource) = block.get("resource") else {
        return "[resource without body]".to_string();
    };
    let uri = resource
        .get("uri")
        .and_then(Value::as_str)
        .unwrap_or("<no uri>");
    match resource.get("text").and_then(Value::as_str) {
        Some(text) => format!("[resource {uri}]\n{text}"),
        None => format!("[resource {uri}: binary blob not shown]"),
    }
}

/// Pull the code/message out of a JSON-RPC error envelope.
pub fn parse_error(error: &Value) -> (i64, String) {
    let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unspecified error")
        .to_string();
    (code, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_carry_the_id_the_response_must_echo() {
        let value = request(7, "tools/list", json!({}));
        assert_eq!(value["jsonrpc"], "2.0");
        assert_eq!(value["id"], 7);
        assert_eq!(value["method"], "tools/list");
        // A notification is the same shape with the id removed.
        assert!(notification("notifications/initialized", json!({}))
            .get("id")
            .is_none());
    }

    #[test]
    fn initialize_announces_our_version_and_identity() {
        let params = initialize_params("0.1.0");
        assert_eq!(params["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(params["clientInfo"]["name"], CLIENT_NAME);
        assert_eq!(params["clientInfo"]["version"], "0.1.0");
        // We answer no server request of any kind, so we declare no client
        // capability. Resources and prompts are the server's to declare.
        assert_eq!(params["capabilities"], json!({}));
    }

    #[test]
    fn a_handshake_says_which_method_sets_the_server_has() {
        let caps = parse_server_capabilities(&json!({
            "capabilities": {"tools": {}, "resources": {"listChanged": true}, "prompts": null}
        }));
        assert!(caps.tools);
        assert!(caps.resources, "an empty object is a declaration");
        assert!(!caps.prompts, "\"prompts\": null is not one");
        assert_eq!(caps.declared(), ["tools", "resources"]);
        assert!(!caps.offers(Feature::Prompts));
    }

    #[test]
    fn a_silent_handshake_is_not_taken_as_a_refusal() {
        for response in [
            json!({}),
            json!({"capabilities": {}}),
            json!({"capabilities": "not an object"}),
        ] {
            let caps = parse_server_capabilities(&response);
            assert!(!caps.declared_any, "{response}");
            assert!(
                !caps.tools && !caps.resources && !caps.prompts,
                "{response}"
            );
            // Asking costs one round trip and may still work; deciding for the
            // server on no evidence would not.
            assert!(caps.offers(Feature::Resources), "{response}");
        }
    }

    #[test]
    fn a_declared_absence_stops_the_call_from_being_sent() {
        let caps = parse_server_capabilities(&json!({"capabilities": {"tools": {}}}));
        assert!(caps.offers(Feature::Tools));
        assert!(!caps.offers(Feature::Prompts));
    }

    #[test]
    fn resources_are_read_by_the_uri_they_must_be_read_with() {
        let result = json!({"resources": [
            {"uri": "file:///notes/todo.md", "name": "todo",
             "description": "What is left", "mimeType": "text/markdown"},
            {"uri": "db://rows", "mimeType": "application/json"},
            {"name": "nowhere — no uri to read it with"},
            {"uri": "   ", "name": "blank uri — dropped"}
        ]});
        let resources = parse_resources(&result);
        assert_eq!(
            resources.iter().map(|r| r.uri.as_str()).collect::<Vec<_>>(),
            ["file:///notes/todo.md", "db://rows"]
        );
        assert_eq!(resources[0].name, "todo");
        assert_eq!(resources[0].mime_type, "text/markdown");
        assert!(resources[0].description.starts_with("What is left"));
        // A resource that ships no name is still called by its uri.
        assert_eq!(resources[1].name, "db://rows");
        assert!(parse_resources(&json!({})).is_empty());
    }

    #[test]
    fn a_text_resource_keeps_its_body_and_a_binary_one_its_size() {
        let contents = parse_resource_contents(&json!({"contents": [
            {"uri": "file:///a", "mimeType": "text/plain", "text": "body of a\n"},
            {"uri": "db://rows", "mimeType": "application/octet-stream", "blob": "AQIDBA=="},
            {"uri": "file:///empty", "mimeType": "text/plain"}
        ]}));
        assert_eq!(contents.len(), 2, "a piece with neither text nor blob");
        // An empty body stays empty rather than reading as binary.
        assert_eq!(contents[0].text.as_deref(), Some("body of a\n"));
        assert_eq!(contents[0].rendered(), "body of a\n");
        assert_eq!(contents[1].blob_len, Some(8));
        assert_eq!(
            contents[1].rendered(),
            "[application/octet-stream: 8 bytes of base64 not shown]"
        );
        // A single body where the spec wants a list is still one resource.
        let one = parse_resource_contents(&json!({
            "contents": {"uri": "file:///b", "text": "just one"}
        }));
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].rendered(), "just one");
        // No mime stated still says how much is being withheld.
        let nameless = parse_resource_contents(&json!({"contents": [{"uri": "x", "blob": "AA"}]}));
        assert_eq!(
            nameless[0].rendered(),
            "[unknown type: 2 bytes of base64 not shown]"
        );
    }

    #[test]
    fn prompts_come_with_the_arguments_they_want_named() {
        let prompts = parse_prompts(&json!({"prompts": [
            {"name": "review", "description": "Read a patch", "arguments": [
                {"name": "patch", "description": "the diff", "required": true},
                {"name": "style"}
            ]},
            {"name": "noargs"},
            {"name": "  ", "description": "blank name — dropped"}
        ]}));
        assert_eq!(
            prompts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            ["review", "noargs"]
        );
        assert_eq!(prompts[0].arguments.len(), 2);
        assert!(prompts[0].arguments[0].required);
        assert!(!prompts[0].arguments[1].required);
        assert_eq!(prompts[0].arguments[1].name, "style");
        assert!(prompts[1].arguments.is_empty());
        assert!(parse_prompts(&json!({})).is_empty());
    }

    #[test]
    fn prompt_arguments_travel_as_strings_under_their_names() {
        let params = get_prompt_params(
            "review",
            &[("patch".to_string(), "@@ -1 +1 @@".to_string())],
        );
        assert_eq!(params["name"], "review");
        assert_eq!(params["arguments"]["patch"], "@@ -1 +1 @@");
        // A prompt that takes nothing still gets an object, never `null`.
        assert_eq!(get_prompt_params("noargs", &[])["arguments"], json!({}));
    }

    #[test]
    fn prompt_messages_keep_their_role_and_name_what_is_not_text() {
        let messages = parse_prompt(&json!({"messages": [
            {"role": "user", "content": {"type": "text", "text": "review this"}},
            {"role": "assistant", "content": {"type": "image", "mimeType": "image/png", "data": "AAAA"}}
        ]}));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].text, "review this");
        assert_eq!(
            messages[1].text,
            "[image: image/png, 4 bytes of base64 not shown]"
        );
        assert!(parse_prompt(&json!({})).is_empty());
    }

    #[test]
    fn tools_are_read_from_both_schema_spellings() {
        let result = json!({"tools": [
            {"name": "fetch", "description": "Gets a page",
             "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}}},
            {"name": "snake", "input_schema": {"type": "object"}},
            {"name": "  ", "description": "nameless — dropped"},
            {"description": "no name at all — dropped"}
        ]});
        let tools = parse_tools(&result);
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            ["fetch", "snake"]
        );
        assert_eq!(tools[0].input_schema["properties"]["url"]["type"], "string");
        assert!(tools[0].description.starts_with("Gets"));
        // A tool that ships no schema still needs an object shape, or the
        // OpenAI-style request becomes invalid.
        assert_eq!(tools[1].input_schema, json!({"type": "object"}));
    }

    #[test]
    fn a_missing_or_wrongly_typed_tools_key_is_no_tools() {
        assert!(parse_tools(&json!({})).is_empty());
        assert!(parse_tools(&json!({"tools": "not an array"})).is_empty());
    }

    #[test]
    fn text_blocks_are_joined_and_the_error_flag_survives() {
        let outcome = parse_tool_result(&json!({
            "content": [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}],
            "isError": true
        }));
        assert_eq!(outcome.text, "line one\nline two");
        assert!(outcome.is_error);
    }

    #[test]
    fn unshivable_content_is_named_rather_than_dropped() {
        let outcome = parse_tool_result(&json!({
            "content": [
                {"type": "image", "mimeType": "image/png", "data": "AAAA"},
                {"type": "resource", "resource": {"uri": "file:///x", "text": "body"}},
                {"type": "resource_link", "uri": "https://example.com"},
                {"type": "mystery"},
                {"no_type": true}
            ]
        }));
        assert!(outcome
            .text
            .contains("[image: image/png, 4 bytes of base64 not shown]"));
        assert!(outcome.text.contains("[resource file:///x]\nbody"));
        assert!(outcome
            .text
            .contains("[resource link: https://example.com]"));
        assert!(outcome.text.contains("[unsupported content: mystery]"));
        assert!(outcome.text.contains("[unrecognized content:"));
        assert!(!outcome.is_error, "absent isError means success");
    }

    #[test]
    fn structured_content_rescues_an_empty_content_list() {
        let outcome = parse_tool_result(&json!({"content": [], "structuredContent": {"n": 2}}));
        assert_eq!(outcome.text, "{\"n\":2}");
    }

    #[test]
    fn error_envelopes_keep_their_code() {
        let (code, message) = parse_error(&json!({"code": -32601, "message": "no such tool"}));
        assert_eq!(code, -32601);
        assert_eq!(message, "no such tool");
        assert_eq!(
            parse_error(&json!({})),
            (0, "unspecified error".to_string())
        );
    }
}
