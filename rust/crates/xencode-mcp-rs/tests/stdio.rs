//! End-to-end tests over a real stdio pipe.
//!
//! The servers are POSIX shell scripts written into a temp dir at test time and
//! spawned by path: no ports, no network, nothing installed. Each fixture
//! answers the JSON-RPC methods it is asked about, echoing the request id, so
//! the client's request/response pairing is exercised for real.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use xencode_mcp_rs::{McpClient, McpError, ServerSpec};

/// Answers the methods xencode uses against a server that has tools alone.
/// Prints one line of log noise on stdout (which the spec forbids but servers
/// do) and one on stderr. It also answers `resources/list` and `prompts/list`
/// with a sentinel: the handshake declares neither, so a call that reaches this
/// fixture would prove the negotiation was never consulted.
const TALKING: &str = r#"#!/bin/sh
echo "fixture: noise on stdout, not json-rpc"
echo "fixture: and a line on stderr" >&2
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    case "$line" in
        *'"method":"initialize"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"0.0.1"}}}\n' "$id"
            ;;
        *'"method":"tools/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"echo","description":"Returns its argument","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},{"name":"boom","description":"Always fails"}]}}\n' "$id"
            ;;
        *'"method":"resources/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"resources":[{"uri":"asked-anyway","name":"asked anyway"}]}}\n' "$id"
            ;;
        *'"method":"prompts/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"prompts":[{"name":"asked-anyway"}]}}\n' "$id"
            ;;
        *'"method":"tools/call"'*)
            case "$line" in
                *'"name":"boom"'*)
                    printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"cannot boom right now"}],"isError":true}}\n' "$id"
                    ;;
                *)
                    text=$(printf '%s' "$line" | sed -n 's/.*"text":"\([^"]*\)".*/\1/p')
                    printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"echo: %s"}]}}\n' "$id" "$text"
                    ;;
            esac
            ;;
    esac
done
"#;

/// Declares all three method sets and answers for each of them.
const OFFERING: &str = r##"#!/bin/sh
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    case "$line" in
        *'"method":"initialize"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{},"resources":{"listChanged":true},"prompts":{}},"serverInfo":{"name":"fixture","version":"0.0.1"}}}\n' "$id"
            ;;
        *'"method":"resources/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"resources":[{"uri":"file:///notes/todo.md","name":"todo","description":"What is left to do","mimeType":"text/markdown"},{"uri":"db://rows","name":"rows","mimeType":"application/octet-stream"}]}}\n' "$id"
            ;;
        *'"method":"resources/read"'*)
            case "$line" in
                *'file:///notes/todo.md'*)
                    printf '{"jsonrpc":"2.0","id":%s,"result":{"contents":[{"uri":"file:///notes/todo.md","mimeType":"text/markdown","text":"# todo\\nship the next release\\n"}]}}\n' "$id"
                    ;;
                *)
                    printf '{"jsonrpc":"2.0","id":%s,"result":{"contents":[{"uri":"db://rows","mimeType":"application/octet-stream","blob":"AQIDBA=="}]}}\n' "$id"
                    ;;
            esac
            ;;
        *'"method":"prompts/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"prompts":[{"name":"review","description":"Review a patch","arguments":[{"name":"patch","description":"the diff to read","required":true}]},{"name":"standup"}]}}\n' "$id"
            ;;
        *'"method":"prompts/get"'*)
            patch=$(printf '%s' "$line" | sed -n 's/.*"patch":"\([^"]*\)".*/\1/p')
            printf '{"jsonrpc":"2.0","id":%s,"result":{"description":"Review a patch","messages":[{"role":"user","content":{"type":"text","text":"review: %s"}}]}}\n' "$id" "$patch"
            ;;
    esac
done
"##;

/// Says nothing about what it has and answers `resources/list` anyway — a
/// handshake that declares nothing is not the same as a server that has nothing.
const UNDECLARED: &str = r#"#!/bin/sh
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    case "$line" in
        *'"method":"initialize"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"fixture","version":"0.0.1"}}}\n' "$id"
            ;;
        *'"method":"resources/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"resources":[{"uri":"file:///silent/offering","name":"offered without saying"}]}}\n' "$id"
            ;;
    esac
done
"#;

/// Reads requests and never answers one: every call must hit its deadline.
const SILENT: &str = "#!/bin/sh\nwhile IFS= read -r line; do :; done\n";

/// Refuses everything after the handshake, with a JSON-RPC error envelope.
const REJECT: &str = r#"#!/bin/sh
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    case "$line" in
        *'"method":"initialize"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"0.0.1"}}}\n' "$id"
            ;;
        *'"id":'*)
            printf '{"jsonrpc":"2.0","id":%s,"error":{"code":-32601,"message":"method not found"}}\n' "$id"
            ;;
    esac
done
"#;

/// Write the script, make it executable, and describe how to run it.
fn fixture(dir: &Path, name: &str, script: &str) -> ServerSpec {
    let path: PathBuf = dir.join(format!("{name}.sh"));
    std::fs::write(&path, script).expect("write fixture");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod fixture");
    ServerSpec::new(name, "/bin/sh").args(&[path.to_string_lossy().to_string()])
}

async fn talking(dir: &Path) -> McpClient {
    McpClient::start(&fixture(dir, "talking", TALKING), Duration::from_secs(5))
        .await
        .expect("the fixture server should start and initialize")
}

async fn offering(dir: &Path) -> McpClient {
    McpClient::start(&fixture(dir, "offering", OFFERING), Duration::from_secs(5))
        .await
        .expect("the fixture server should start and initialize")
}

#[tokio::test]
async fn handshake_lists_tools_and_calls_one() {
    let dir = tempfile::tempdir().unwrap();
    let client = talking(dir.path()).await;

    let tools = client.list_tools().await.expect("tools/list");
    assert_eq!(
        tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["echo", "boom"]
    );
    assert_eq!(tools[0].description, "Returns its argument");
    assert_eq!(
        tools[0].input_schema["properties"]["text"]["type"],
        "string"
    );
    // The tool with no schema of its own still reaches the model as an object.
    assert_eq!(tools[1].input_schema["type"], "object");

    assert_eq!(
        client.stray_lines(),
        1,
        "the stdout noise must be counted, not treated as fatal"
    );
    assert!(client.stderr_tail().contains("a line on stderr"));

    assert_eq!(
        client
            .call_tool("echo", serde_json::json!({"text": "hello"}))
            .await
            .expect("tools/call"),
        "echo: hello"
    );

    // A server-reported failure stays a failure in the text the model reads.
    let failed = client
        .call_tool("boom", serde_json::json!({}))
        .await
        .expect("the call reached the server");
    assert!(
        failed.starts_with("MCP tool `boom` reported an error:"),
        "{failed}"
    );
    assert!(failed.contains("cannot boom right now"), "{failed}");

    client.shutdown().await;
}

#[tokio::test]
async fn arguments_are_forwarded_as_an_object_even_when_we_have_none() {
    let dir = tempfile::tempdir().unwrap();
    let client = talking(dir.path()).await;
    // `Value::Null` must not reach the server as `"arguments": null`: the spec
    // wants an object, and a strict server rejects anything else.
    let said = client
        .call_tool("echo", serde_json::Value::Null)
        .await
        .expect("call accepted");
    assert_eq!(said, "echo: ");
}

#[tokio::test]
async fn a_server_that_never_answers_hits_its_deadline_naming_itself() {
    let dir = tempfile::tempdir().unwrap();
    let error = McpClient::start(
        &fixture(dir.path(), "silent", SILENT),
        Duration::from_secs(2),
    )
    .await
    .err()
    .expect("no initialize reply is coming");
    match error {
        McpError::Timeout {
            server,
            method,
            secs,
        } => {
            assert_eq!(server, "silent");
            assert_eq!(method, "initialize");
            assert_eq!(secs, 2, "the caller's own budget, reported honestly");
        }
        other => panic!("expected a timeout, got {other:?}"),
    }
}

#[tokio::test]
async fn a_server_that_exits_during_handshake_reports_its_own_words() {
    let dir = tempfile::tempdir().unwrap();
    let spec = fixture(
        dir.path(),
        "dead",
        "#!/bin/sh\necho \"I refuse to speak JSON\" >&2\nexit 1\n",
    );
    let error = McpClient::start(&spec, Duration::from_secs(5))
        .await
        .err()
        .expect("a server that exits cannot be initialized");
    let text = error.to_string();
    assert!(text.contains("dead"), "{text}");
    // The stderr tail is the only clue there is, so it has to come along.
    assert!(text.contains("I refuse to speak JSON"), "{text}");
}

#[tokio::test]
async fn an_unstartable_command_is_reported_not_panicked() {
    let spec = ServerSpec::new("missing", "/nonexistent/xencode-mcp-server");
    match McpClient::start(&spec, Duration::from_secs(2))
        .await
        .err()
        .expect("nothing to start")
    {
        McpError::Spawn { server, reason } => {
            assert_eq!(server, "missing");
            assert!(!reason.is_empty(), "the OS message is the useful part");
        }
        other => panic!("expected a spawn failure, got {other:?}"),
    }
}

#[tokio::test]
async fn a_rejected_method_surfaces_as_a_server_error_with_its_code() {
    let dir = tempfile::tempdir().unwrap();
    let client = McpClient::start(
        &fixture(dir.path(), "reject", REJECT),
        Duration::from_secs(5),
    )
    .await
    .expect("the handshake succeeds; only later calls are refused");
    match client.list_tools().await.expect_err("the server says no") {
        McpError::Server {
            server,
            code,
            message,
        } => {
            assert_eq!(server, "reject");
            assert_eq!(code, -32601);
            assert_eq!(message, "method not found");
        }
        other => panic!("expected a JSON-RPC error, got {other:?}"),
    }
}

/// The handshake is kept, not thrown away: what it declared decides what we
/// send afterwards.
#[tokio::test]
async fn a_server_that_declared_tools_only_is_told_so_and_never_quizzed() {
    let dir = tempfile::tempdir().unwrap();
    let client = talking(dir.path()).await;
    assert!(client.capabilities().tools);
    assert!(!client.capabilities().resources);
    assert!(!client.capabilities().prompts);
    assert_eq!(client.capabilities().declared(), ["tools"]);

    // The fixture would have answered `resources/list` and `prompts/list` with
    // a resource named "asked-anyway". Getting a refusal back is the evidence
    // that no such request left us.
    macro_rules! refused {
        ($call:expr, $feature:expr) => {
            match $call.await.expect_err("this server never said it had them") {
                McpError::NotOffered {
                    server,
                    feature: named,
                } => {
                    assert_eq!(server, "talking");
                    assert_eq!(named, $feature);
                }
                other => panic!("expected a not-offered refusal, got {other:?}"),
            }
        };
    }
    refused!(client.list_resources(), "resources");
    refused!(client.read_resource("asked-anyway"), "resources");
    refused!(client.list_prompts(), "prompts");
    refused!(client.get_prompt("asked-anyway", &[]), "prompts");
    // Tools, which it did declare, still go through the same gate and work.
    assert_eq!(client.list_tools().await.expect("tools/list").len(), 2);
    client.shutdown().await;
}

#[tokio::test]
async fn resources_and_prompts_are_read_from_a_server_that_offers_them() {
    let dir = tempfile::tempdir().unwrap();
    let client = offering(dir.path()).await;
    assert_eq!(
        client.capabilities().declared(),
        ["tools", "resources", "prompts"]
    );

    let resources = client.list_resources().await.expect("resources/list");
    assert_eq!(
        resources.iter().map(|r| r.uri.as_str()).collect::<Vec<_>>(),
        ["file:///notes/todo.md", "db://rows"]
    );
    assert_eq!(resources[0].name, "todo");
    assert_eq!(resources[0].description, "What is left to do");
    assert_eq!(resources[0].mime_type, "text/markdown");

    let body = client
        .read_resource("file:///notes/todo.md")
        .await
        .expect("resources/read");
    assert_eq!(body.len(), 1);
    assert_eq!(body[0].rendered(), "# todo\nship the next release\n");

    // A binary piece says how much it withheld rather than looking empty.
    let binary = client
        .read_resource("db://rows")
        .await
        .expect("binary read");
    assert_eq!(binary[0].text, None);
    assert_eq!(binary[0].blob_len, Some(8));
    assert_eq!(
        binary[0].rendered(),
        "[application/octet-stream: 8 bytes of base64 not shown]"
    );

    let prompts = client.list_prompts().await.expect("prompts/list");
    assert_eq!(
        prompts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
        ["review", "standup"]
    );
    assert_eq!(prompts[0].arguments[0].name, "patch");
    assert!(prompts[0].arguments[0].required);
    assert!(prompts[1].arguments.is_empty());

    // The argument reached the server as a named string, and the prompt came
    // back with the role the server wrote it in.
    let messages = client
        .get_prompt(
            "review",
            &[("patch".to_string(), "-old line +new line".to_string())],
        )
        .await
        .expect("prompts/get");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].text, "review: -old line +new line");

    client.shutdown().await;
}

#[tokio::test]
async fn a_server_that_declares_nothing_is_still_asked() {
    let dir = tempfile::tempdir().unwrap();
    let client = McpClient::start(
        &fixture(dir.path(), "undeclared", UNDECLARED),
        Duration::from_secs(5),
    )
    .await
    .expect("the handshake succeeds");
    assert!(!client.capabilities().declared_any);
    // An empty capability block told us nothing, so it cannot be a refusal: the
    // call goes and the server's own answer settles it.
    let resources = client
        .list_resources()
        .await
        .expect("a server that stayed silent may still have resources");
    assert_eq!(resources[0].uri, "file:///silent/offering");
    client.shutdown().await;
}
