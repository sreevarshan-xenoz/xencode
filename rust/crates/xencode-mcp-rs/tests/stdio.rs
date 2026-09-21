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

/// Answers the three methods xencode uses. Prints one line of log noise on
/// stdout (which the spec forbids but servers do) and one on stderr.
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
