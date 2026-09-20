//! Tool calling for the agentic loop.
//!
//! `llama-server` started with `--agent` (or `--tools …`) renders tool
//! definitions into the chat template so the model emits OpenAI-style
//! `tool_calls`. The server does NOT execute them for API clients —
//! execution stays client-side (see `xencode-agent-rs`), which keeps one
//! sandbox and one approval policy across every provider.
//!
//! Wire shapes differ per backend:
//! - OpenAI-compatible (llama.cpp, OpenRouter, Qwen): `tools[]` in the
//!   request, `message.tool_calls[]` / streaming `delta.tool_calls[]` out.
//! - Ollama native `/api/chat`: the same `tools[]` schema in, complete
//!   `message.tool_calls[]` arrays per NDJSON line out (no delta fragments).
//! - Gemini / Anthropic native APIs: not yet plumbed — the loop degrades to
//!   single-shot text on those routes (no regression vs. today).

use std::collections::HashMap;

/// One callable function offered to the model (OpenAI function schema).
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema object for the arguments
    /// (`{"type":"object","properties":{…}}`).
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    /// `{"type":"function","function":{…}}` value. Ollama's `/api/chat`
    /// accepts the same schema.
    pub fn to_api_value(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// One model-requested call, normalized across backends.
///
/// `arguments` preserves the wire shape (string for OpenAI-compatible,
/// object for Ollama) so history can be echoed back verbatim; use
/// [`ToolCall::arguments_object`] to execute.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// OpenAI `tool_calls[].id` — generated (`call_{index}`) when the backend
    /// supplies none (Ollama), so results can always be paired.
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Arguments as an object: strings are parsed, objects pass through,
    /// anything else becomes `{}`.
    pub fn arguments_object(&self) -> serde_json::Map<String, serde_json::Value> {
        match &self.arguments {
            serde_json::Value::Object(map) => map.clone(),
            serde_json::Value::String(s) => serde_json::from_str(s).unwrap_or_default(),
            _ => serde_json::Map::new(),
        }
    }

    /// Arguments rendered as a JSON string for OpenAI-style echo/history.
    pub fn arguments_json(&self) -> String {
        match &self.arguments {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
        }
    }
}

/// Extra turns of an agentic exchange: the assistant turn that requested
/// tools, and each tool's result.
#[derive(Debug, Clone)]
pub enum AgentTurn {
    Assistant { text: String, calls: Vec<ToolCall> },
    ToolResult { id: String, content: String },
}

/// One streaming step: visible text plus any requested calls.
#[derive(Debug, Clone, Default)]
pub struct AgentStep {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HistoryStyle {
    OpenAI,
    Ollama,
}

/// Render base chat turns + agent turns into a backend-ready message array.
pub(crate) fn render_history(
    base: &[crate::ChatMessage],
    extra: &[AgentTurn],
    style: HistoryStyle,
) -> serde_json::Value {
    let mut out: Vec<serde_json::Value> = base
        .iter()
        .map(|m| match style {
            // OpenAI-family backends take content blocks natively; the
            // untagged enum serializes text as a bare string.
            HistoryStyle::OpenAI => serde_json::json!({"role": m.role, "content": m.content}),
            // Ollama wants text plus a top-level `images` array instead.
            HistoryStyle::Ollama => crate::to_ollama_value(m),
        })
        .collect();
    for turn in extra {
        match turn {
            AgentTurn::Assistant { text, calls } => {
                let mut msg = serde_json::json!({"role": "assistant", "content": text});
                if !calls.is_empty() {
                    msg["tool_calls"] = match style {
                        HistoryStyle::OpenAI => serde_json::Value::Array(
                            calls
                                .iter()
                                .map(|c| {
                                    serde_json::json!({
                                        "id": c.id,
                                        "type": "function",
                                        "function": {
                                            "name": c.name,
                                            "arguments": c.arguments_json(),
                                        }
                                    })
                                })
                                .collect(),
                        ),
                        HistoryStyle::Ollama => serde_json::Value::Array(
                            calls
                                .iter()
                                .map(|c| {
                                    serde_json::json!({
                                        "function": {
                                            "name": c.name,
                                            "arguments": serde_json::Value::Object(
                                                c.arguments_object(),
                                            ),
                                        }
                                    })
                                })
                                .collect(),
                        ),
                    };
                }
                out.push(msg);
            }
            AgentTurn::ToolResult { id, content } => {
                let mut msg = serde_json::json!({"role": "tool", "content": content});
                if style == HistoryStyle::OpenAI {
                    msg["tool_call_id"] = serde_json::Value::String(id.clone());
                }
                out.push(msg);
            }
        }
    }
    serde_json::Value::Array(out)
}

/// Parse an Ollama-style `tool_calls` array
/// (`[{function: {name, arguments}}]`, no ids).
pub(crate) fn parse_ollama_calls(value: &serde_json::Value, id_prefix: &str) -> Vec<ToolCall> {
    let Some(arr) = value.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .enumerate()
        .filter_map(|(i, entry)| {
            let function = entry.get("function")?;
            let name = function.get("name")?.as_str()?;
            if name.is_empty() {
                return None;
            }
            let arguments = function
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            Some(ToolCall {
                id: format!("{id_prefix}{i}"),
                name: name.to_string(),
                arguments,
            })
        })
        .collect()
}

#[derive(Debug, Default)]
struct PartialCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Accumulates streaming `delta.tool_calls[]` fragments keyed by `index`.
/// The model sends the id/name first and the arguments JSON in pieces.
/// Entries without an explicit index (some proxies omit it) join the slot
/// of the same call id, else the in-progress anonymous slot, else a fresh
/// slot — parallel calls never melt into one garbled call.
#[derive(Debug, Default)]
pub(crate) struct ToolCallAccumulator {
    partial: HashMap<u32, PartialCall>,
    last_slot: Option<u32>,
}

impl ToolCallAccumulator {
    /// Slot for one delta entry: explicit index wins; otherwise id-match,
    /// anonymous continuation, or a fresh slot past every seen index.
    fn slot_for(&mut self, entry: &serde_json::Value) -> u32 {
        if let Some(i) = entry.get("index").and_then(|v| v.as_u64()) {
            let key = i as u32;
            self.last_slot = Some(key);
            return key;
        }
        let id = entry
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());
        if let Some(id) = id {
            if let Some((&key, _)) = self
                .partial
                .iter()
                .find(|(_, part)| part.id.as_deref() == Some(id))
            {
                self.last_slot = Some(key);
                return key;
            }
        } else if let Some(key) = self.last_slot {
            if self.partial.get(&key).is_some_and(|part| part.id.is_none()) {
                return key;
            }
        }
        let key = self
            .partial
            .keys()
            .max()
            .map(|m| m.saturating_add(1))
            .unwrap_or(0);
        self.last_slot = Some(key);
        key
    }

    pub(crate) fn ingest(&mut self, value: &serde_json::Value) {
        let Some(arr) = value.as_array() else {
            return;
        };
        for entry in arr {
            let index = self.slot_for(entry);
            let part = self.partial.entry(index).or_default();
            if part.id.is_none() {
                part.id = entry
                    .get("id")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
            }
            if let Some(function) = entry.get("function") {
                if part.name.is_none() {
                    part.name = function
                        .get("name")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(str::to_string);
                }
                if let Some(args) = function.get("arguments").and_then(|v| v.as_str()) {
                    part.arguments.push_str(args);
                }
            }
        }
    }

    pub(crate) fn finish(self) -> Vec<ToolCall> {
        let mut indexed: Vec<(u32, PartialCall)> = self.partial.into_iter().collect();
        indexed.sort_by_key(|(i, _)| *i);
        indexed
            .into_iter()
            .filter_map(|(i, part)| {
                let name = part.name?;
                if name.is_empty() {
                    return None;
                }
                Some(ToolCall {
                    id: part.id.unwrap_or_else(|| format!("call_{i}")),
                    name,
                    arguments: serde_json::Value::String(part.arguments),
                })
            })
            .collect()
    }
}

/// Shared OpenAI-SSE chunk handler: forwards text through the callback and
/// accumulates any `delta.tool_calls` fragments. Returns `true` when the
/// chunk carried the terminal `[DONE]` marker.
pub(crate) fn ingest_oai_chunk(
    json: &serde_json::Value,
    text: &mut String,
    acc: &mut ToolCallAccumulator,
    callback: &mut impl FnMut(&str),
) {
    let Some(choices) = json.get("choices").and_then(|c| c.as_array()) else {
        return;
    };
    let Some(first) = choices.first() else {
        return;
    };
    let Some(delta) = first.get("delta") else {
        return;
    };
    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
        callback(content);
        text.push_str(content);
    }
    if let Some(calls) = delta.get("tool_calls") {
        acc.ingest(calls);
    }
}

/// Tool surface for the background task registry (Milestone D, D1-02).
///
/// Schemas only — execution lives with the caller that owns a
/// [`xencode_core_rs::TaskManager`](../xencode_core_rs/struct.TaskManager.html),
/// so one approval policy covers every provider.
pub fn background_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "background_start".to_string(),
            description: "Start a shell command as a background task and return its \
                          task id immediately, without waiting for it to finish. Use \
                          for long-running work (builds, tests, servers)."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short label shown in the task list"
                    },
                    "command": {
                        "type": "string",
                        "description": "Shell command line, run through sh -c"
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Directory to run the command in (e.g. a git \
                                        worktree path); defaults to the session directory"
                    }
                },
                "required": ["command"]
            }),
        },
        ToolDefinition {
            name: "background_poll".to_string(),
            description: "Check a background task: its status (running, exited with a \
                          code, or killed) plus the most recent output lines."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Task id returned by background_start"
                    }
                },
                "required": ["id"]
            }),
        },
        ToolDefinition {
            name: "background_stop".to_string(),
            description: "Kill a running background task. Finishing a task this way \
                          is permanent; the record can then be removed."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Task id returned by background_start"
                    }
                },
                "required": ["id"]
            }),
        },
    ]
}

/// Tool surface for repository insights (Milestone F, F3-02).
///
/// Schema only — execution lives in the TUI's agent tool loop, which owns
/// the workspace snapshot.
pub fn advise_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "repo_advise".to_string(),
        description: "Report deterministic repository insights from the \
                      project's .xencode symbol snapshot: broken imports, \
                      import cycles, hub files and orphan files. Read-only; \
                      requires the project index (/init). Returns at most 40 \
                      findings; pass a path filter to narrow them."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Only report findings whose file path \
                                    contains this substring"
                }
            }
        }),
    }]
}

/// The workspace file tools (Milestone I, I1-02): read, list, search,
/// write and a precise old→new edit. Schemas stay deliberately small —
/// local models call fewer, flatter tools far more reliably. Execution
/// lives in the TUI's agent tool loop behind the permission policy.
pub fn file_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a UTF-8 text file in the workspace and return \
                          its lines with numbers. Returns 200 lines per call; \
                          pass offset (1-based line number) to page through \
                          longer files."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, relative to the workspace \
                                        root (e.g. src/main.rs)"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "First line to return, 1-based; \
                                        defaults to 1"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum lines to return (default \
                                        200, cap 2000)"
                    }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "list_dir".to_string(),
            description: "List the entries of a directory in the workspace. \
                          Directories end with a slash. Returns at most 300 \
                          entries."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the \
                                        workspace root; empty or omitted \
                                        means the root itself"
                    }
                }
            }),
        },
        ToolDefinition {
            name: "search_files".to_string(),
            description: "Search the workspace for a regular expression \
                          (matched line by line) and return \
                          path:line:content matches, at most 100 hits. Skips \
                          .git, target, node_modules and binary files."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression, e.g. fn\\s+main"
                    },
                    "path": {
                        "type": "string",
                        "description": "Restrict to this file or directory \
                                        (relative); omit for the whole \
                                        workspace"
                    }
                },
                "required": ["pattern"]
            }),
        },
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Create or overwrite a UTF-8 text file in the \
                          workspace with the given content. Parent \
                          directories are created. Returns a unified diff \
                          of the change."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace \
                                        root"
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete new content of the file"
                    }
                },
                "required": ["path", "content"]
            }),
        },
        ToolDefinition {
            name: "edit_file".to_string(),
            description: "Replace an exact string in an existing workspace \
                          file. Fails when old is missing or appears more \
                          than once — then pass more surrounding context, \
                          or all=true to replace every occurrence. Returns \
                          a unified diff."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace \
                                        root"
                    },
                    "old": {
                        "type": "string",
                        "description": "The exact text to replace"
                    },
                    "new": {
                        "type": "string",
                        "description": "The replacement text"
                    },
                    "all": {
                        "type": "boolean",
                        "description": "Replace every occurrence instead of \
                                        requiring a unique match"
                    }
                },
                "required": ["path", "old", "new"]
            }),
        },
    ]
}

/// Tool surface for running a command and waiting for it (Milestone I,
/// I2-02). This is the "did my change actually work?" tool: the output comes
/// back to the model as produced, so the timeout and the output cap are
/// enforced by the executor rather than by the schema.
pub fn command_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "run_command".to_string(),
        description: "Run a shell command in the workspace root and wait for it to \
                      finish, returning its combined output and exit code. Use this to \
                      build, test or inspect after a change. Long-running work belongs \
                      in background_start instead."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command line, run through sh -c in the \
                                    workspace root"
                }
            },
            "required": ["command"]
        }),
    }]
}

/// The agent's visible todo list (Milestone I, I2-03). One call replaces the
/// whole plan, so the model never has to track indices, and the tool touches
/// no files — it is presentation only. A model that ignores it loses the
/// checklist and nothing else.
pub fn plan_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "update_plan".to_string(),
        description: "Replace your task list for the current request. Call it \
                      once when you decide the approach and again whenever an \
                      item's status changes; keep it to at most 12 short \
                      items. Statuses: pending, in_progress, done. This \
                      changes no files and runs no commands."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "The complete plan, in the order you will do it",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "One short line describing the step"
                            },
                            "status": {
                                "type": "string",
                                "description": "pending | in_progress | done \
                                                (default pending)"
                            }
                        },
                        "required": ["text"]
                    }
                }
            },
            "required": ["items"]
        }),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_command_takes_one_string_argument() {
        let tools = command_tools();
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            ["run_command"]
        );
        let value = tools[0].to_api_value();
        let props = value["function"]["parameters"]["properties"]
            .as_object()
            .unwrap();
        assert_eq!(props.len(), 1);
        assert_eq!(props["command"]["type"], "string");
        assert_eq!(value["function"]["parameters"]["required"][0], "command");
    }

    #[test]
    fn update_plan_takes_one_array_of_text_and_status() {
        let tools = plan_tools();
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            ["update_plan"]
        );
        let value = tools[0].to_api_value();
        let params = &value["function"]["parameters"];
        assert_eq!(params["required"][0], "items");
        // The only structured argument we offer, so keep its object flat:
        // text + status, and only text is required.
        let item = &params["properties"]["items"]["items"];
        let props = item["properties"].as_object().unwrap();
        assert_eq!(props.len(), 2);
        assert!(props.contains_key("text") && props.contains_key("status"));
        assert_eq!(item["required"][0], "text");
    }

    #[test]
    fn file_tools_have_minimal_openai_function_schemas() {
        let tools = file_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "read_file",
                "list_dir",
                "search_files",
                "write_file",
                "edit_file"
            ]
        );
        for tool in &tools {
            let value = tool.to_api_value();
            assert_eq!(value["type"], "function");
            assert_eq!(value["function"]["name"], tool.name);
            // Every argument is a plain scalar — local models handle these.
            let props = value["function"]["parameters"]["properties"]
                .as_object()
                .unwrap();
            for (_, schema) in props {
                let ty = schema["type"].as_str().unwrap();
                assert!(
                    matches!(ty, "string" | "integer" | "boolean"),
                    "{}: {ty}",
                    tool.name
                );
            }
        }
        assert_eq!(tools[0].parameters["required"][0], "path");
        assert_eq!(tools[4].parameters["required"][2], "new");
    }

    #[test]
    fn advise_tool_is_a_valid_openai_function_schema() {
        let tools = advise_tools();
        assert_eq!(tools[0].name, "repo_advise");
        let value = tools[0].to_api_value();
        assert_eq!(value["type"], "function");
        assert_eq!(value["function"]["name"], "repo_advise");
        let params = &value["function"]["parameters"];
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["filter"].is_object());
        // Everything is optional.
        assert!(params.get("required").is_none());
        assert!(!tools[0].description.is_empty());
    }

    #[test]
    fn background_tools_are_valid_openai_function_schemas() {
        let tools = background_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            ["background_start", "background_poll", "background_stop"]
        );
        for tool in &tools {
            let value = tool.to_api_value();
            assert_eq!(value["type"], "function");
            assert_eq!(value["function"]["name"], tool.name.as_str());
            let params = &value["function"]["parameters"];
            assert_eq!(params["type"], "object");
            assert!(
                params["properties"].is_object()
                    && !params["properties"].as_object().unwrap().is_empty()
            );
            assert!(!tool.description.is_empty());
        }
        // Only `background_start` has an optional argument.
        let start = &tools[0].parameters["required"];
        assert_eq!(start.as_array().unwrap().len(), 1);
        for tool in &tools[1..] {
            let required = tool.parameters["required"].as_array().unwrap().clone();
            assert_eq!(required.len(), 1);
            assert_eq!(required[0], "id");
        }
    }

    #[test]
    fn definition_renders_openai_function_schema() {
        let def = ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        };
        let v = def.to_api_value();
        assert_eq!(v["type"], "function");
        assert_eq!(v["function"]["name"], "read_file");
        assert_eq!(v["function"]["parameters"]["required"][0], "path");
    }

    #[test]
    fn parse_ollama_object_args_and_generated_ids() {
        let v = serde_json::json!([
            {"function": {"name": "read_file", "arguments": {"path": "x"}}},
        ]);
        let calls = parse_ollama_calls(&v, "call_");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(
            calls[0].arguments_object()["path"],
            serde_json::Value::String("x".to_string())
        );
    }

    #[test]
    fn accumulator_concatenates_split_arguments_across_chunks() {
        let mut acc = ToolCallAccumulator::default();
        acc.ingest(&serde_json::json!([
            {"index": 0, "id": "a1", "function": {"name": "read_file", "arguments": "{\"pa"}}
        ]));
        acc.ingest(&serde_json::json!([
            {"index": 0, "function": {"arguments": "th\":\"x\"}"}}
        ]));
        acc.ingest(&serde_json::json!([
            {"index": 1, "id": "a2", "function": {"name": "grep_search", "arguments": "{}"}}
        ]));
        let calls = acc.finish();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "a1");
        assert_eq!(calls[0].arguments_json(), "{\"path\":\"x\"}");
        assert_eq!(calls[1].id, "a2");
    }

    #[test]
    fn accumulator_demuxes_index_less_parallel_calls() {
        // Proxies that omit `index`: same-id fragments join, distinct calls
        // stay separate instead of melting into one garbled call.
        let mut acc = ToolCallAccumulator::default();
        acc.ingest(&serde_json::json!([
            {"id": "a1", "function": {"name": "read_file", "arguments": "{\"pa"}},
            {"id": "a2", "function": {"name": "grep_search", "arguments": "{}"}},
        ]));
        acc.ingest(&serde_json::json!([
            {"id": "a1", "function": {"arguments": "th\":\"x\"}"}},
        ]));
        let calls = acc.finish();
        assert_eq!(calls.len(), 2);
        let a1 = calls.iter().find(|c| c.id == "a1").unwrap();
        assert_eq!(a1.arguments_json(), "{\"path\":\"x\"}");
        assert!(calls.iter().any(|c| c.id == "a2"));
    }

    #[test]
    fn render_openai_history_pairs_ids() {
        let base = vec![crate::ChatMessage {
            role: "user".to_string(),
            content: "hi".into(),
        }];
        let extra = vec![
            AgentTurn::Assistant {
                text: "".to_string(),
                calls: vec![ToolCall {
                    id: "a1".to_string(),
                    name: "read_file".to_string(),
                    arguments: serde_json::json!("{\"path\":\"x\"}"),
                }],
            },
            AgentTurn::ToolResult {
                id: "a1".to_string(),
                content: "contents".into(),
            },
        ];
        let v = render_history(&base, &extra, HistoryStyle::OpenAI);
        assert_eq!(v[0]["role"], "user");
        assert_eq!(v[1]["tool_calls"][0]["id"], "a1");
        assert_eq!(
            v[1]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"x\"}"
        );
        assert_eq!(v[2]["role"], "tool");
        assert_eq!(v[2]["tool_call_id"], "a1");
    }

    #[test]
    fn render_ollama_history_keeps_object_args_without_ids() {
        let extra = vec![AgentTurn::Assistant {
            text: "".to_string(),
            calls: vec![ToolCall {
                id: "call_0".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "x"}),
            }],
        }];
        let v = render_history(&[], &extra, HistoryStyle::Ollama);
        assert_eq!(v[0]["tool_calls"][0]["function"]["arguments"]["path"], "x");
        assert!(v[0]["tool_calls"][0].get("id").is_none());
    }

    #[test]
    fn render_history_maps_images_per_backend_style() {
        let base = vec![crate::ChatMessage::user_with_images(
            "see",
            vec!["data:image/png;base64,AAAA".to_string()],
        )];
        let openai = render_history(&base, &[], HistoryStyle::OpenAI);
        assert_eq!(
            openai[0]["content"],
            serde_json::json!([
                {"type": "text", "text": "see"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ])
        );
        let ollama = render_history(&base, &[], HistoryStyle::Ollama);
        assert_eq!(ollama[0]["content"], "see");
        assert_eq!(ollama[0]["images"], serde_json::json!(["AAAA"]));
    }
}
