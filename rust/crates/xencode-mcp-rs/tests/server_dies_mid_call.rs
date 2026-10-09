//! QA-6: an MCP server process that dies in the middle of a tool call.
//!
//! The server is a real child process — a POSIX shell script run by `sh` from
//! `PATH`, which is Git's shell on Windows — that completes the handshake and
//! lists its tools, then exits when a tool is called. The call must fail at
//! once, naming the server and carrying what it said on stderr, instead of
//! waiting out its deadline.

use std::path::Path;
use std::time::{Duration, Instant};

use xencode_mcp_rs::{McpClient, ServerSpec};

const DIES_ON_CALL: &str = r#"
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
    case "$line" in
        *'"method":"initialize"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"fixture","version":"0.0.1"}}}\n' "$id"
            ;;
        *'"method":"tools/list"'*)
            printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"work","description":"Dies instead"}]}}\n' "$id"
            ;;
        *'"method":"tools/call"'*)
            echo "out of memory while working" >&2
            exit 9
            ;;
    esac
done
"#;

fn sh_on_path() -> bool {
    std::process::Command::new("sh")
        .args(["-c", "exit 0"])
        .status()
        .is_ok_and(|s| s.success())
}

fn spec(dir: &Path) -> ServerSpec {
    let path = dir.join("dies.sh");
    std::fs::write(&path, DIES_ON_CALL).expect("write the server script");
    ServerSpec::new("dies", "sh").args(&[path.to_string_lossy().to_string()])
}

#[tokio::test]
async fn a_server_that_dies_mid_call_fails_the_call_at_once_with_its_last_words() {
    if !sh_on_path() {
        eprintln!("skipped: no `sh` on PATH to run the server script with");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    // A long deadline, so a failure that waited for it would show in the time.
    let client = McpClient::start(&spec(dir.path()), Duration::from_secs(30))
        .await
        .expect("the handshake completes before the server dies");
    let tools = client.list_tools().await.expect("tools/list");
    assert_eq!(tools[0].name, "work");

    let started = Instant::now();
    let error = client
        .call_tool("work", serde_json::json!({}))
        .await
        .expect_err("the server died; the call cannot have succeeded");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the call waited {:?} instead of noticing the server had gone",
        started.elapsed()
    );
    let text = error.to_string();
    assert!(text.contains("closed the connection"), "{text}");
    assert!(text.contains("dies"), "{text}");
    assert!(
        text.contains("out of memory while working"),
        "{text} / tail now: {:?}",
        client.stderr_tail()
    );

    // The next call on the dead connection fails the same way, without hanging.
    let again = client.call_tool("work", serde_json::json!({})).await;
    assert!(again.is_err());
}
