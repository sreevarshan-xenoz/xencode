//! The wire shapes of Model Context Protocol, as far as xencode uses it:
//! JSON-RPC 2.0 requests and the three tool methods
//! (`initialize` / `tools/list` / `tools/call`).
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
        "capabilities": {},
        "clientInfo": { "name": CLIENT_NAME, "version": client_version },
    })
}

pub fn call_tool_params(name: &str, arguments: Value) -> Value {
    json!({ "name": name, "arguments": arguments })
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
    let mut parts: Vec<String> = Vec::new();

    if let Some(blocks) = result.get("content").and_then(Value::as_array) {
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => parts.push(
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                ),
                Some("image") => parts.push(describe_binary("image", block)),
                Some("audio") => parts.push(describe_binary("audio", block)),
                Some("resource") => parts.push(describe_resource(block)),
                Some("resource_link") => parts.push(
                    block
                        .get("uri")
                        .and_then(Value::as_str)
                        .map(|uri| format!("[resource link: {uri}]"))
                        .unwrap_or_else(|| "[resource link without uri]".to_string()),
                ),
                Some(other) => parts.push(format!("[unsupported content: {other}]")),
                // A block with no discriminator at all: show its JSON rather
                // than pretend the call returned nothing.
                None => parts.push(format!("[unrecognized content: {block}]")),
            }
        }
    }

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
        assert_eq!(params["capabilities"], json!({}));
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
