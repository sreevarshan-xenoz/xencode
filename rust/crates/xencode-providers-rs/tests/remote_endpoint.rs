//! Integration tests for the `remote:` route — a user-configured
//! OpenAI-compatible `/chat/completions` endpoint (LM Studio, vLLM, a proxy, or
//! an SSH-forwarded llama-server on a Google Colab GPU).
//!
//! The route is only useful if it really posts to the configured host, sends
//! the bearer token when there is one and none when there is not, and streams.
//! wiremock supplies the server, so no network or GPU is involved.

use wiremock::matchers::{body_json, header, header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use xencode_models_rs::OllamaClient;
use xencode_providers_rs::{ChatMessage, ProviderManager, ToolDefinition};

fn test_messages() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".into(),
    }]
}

/// Manager pointed at `base_url`, with a key only when one is given. Ollama is
/// unreachable so any accidental fall-through shows up as an Ollama error.
fn manager_for(base_url: &str, key: Option<&str>) -> ProviderManager {
    ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        None,
        None,
        None,
        None,
    )
    .with_remote(base_url, key.map(|s| s.to_string()))
}

fn completion_body(text: &str) -> serde_json::Value {
    serde_json::json!({ "choices": [{ "message": { "role": "assistant", "content": text } }] })
}

#[tokio::test]
async fn remote_prefix_without_an_endpoint_explains_how_to_set_one() {
    let manager = ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        None,
        None,
        None,
        None,
    );
    let error = manager
        .generate("remote:qwen2.5:7b", &test_messages())
        .await
        .expect_err("an unconfigured endpoint must not fall through to Ollama");
    let text = error.to_string();
    assert!(
        text.contains("No remote endpoint configured") && text.contains("remote_url"),
        "unhelpful error: {text}"
    );
}

#[tokio::test]
async fn blank_or_whitespace_base_url_leaves_the_route_unconfigured() {
    let manager = manager_for("   ", Some("k"));
    let error = manager
        .generate("remote:m", &test_messages())
        .await
        .expect_err("whitespace is not an endpoint");
    assert!(
        error.to_string().contains("No remote endpoint configured"),
        "{error}"
    );
}

#[tokio::test]
async fn remote_prefix_posts_to_the_configured_base_url_and_returns_the_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer colab-token"))
        .and(body_json(serde_json::json!({
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("from the GPU")))
        .expect(1)
        .mount(&server)
        .await;

    let url = format!("{}/v1", server.uri());
    let manager = manager_for(&url, Some("colab-token"));
    let reply = manager
        .generate("remote:qwen2.5:7b", &test_messages())
        .await
        .expect("remote route should answer");
    assert_eq!(reply, "from the GPU");
}

#[tokio::test]
async fn a_keyless_endpoint_sends_no_authorization_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header_exists("Authorization"))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("unexpected")))
        .mount(&server)
        .await;

    let manager = manager_for(&format!("{}/v1", server.uri()), None);
    let error = manager
        .generate("remote:m", &test_messages())
        .await
        .expect_err("the only mock requires an Authorization header, so a keyless call must 404");
    // 404 from wiremock's default handler proves no header-matched mock ran.
    assert!(error.to_string().contains("404"), "{error}");
}

#[tokio::test]
async fn remote_prefix_streams_tokens_and_ignores_the_done_marker() {
    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: [DONE]\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_bytes(sse),
        )
        .mount(&server)
        .await;

    let manager = manager_for(&format!("{}/v1", server.uri()), None);
    let mut seen = String::new();
    let text = {
        let seen = &mut seen;
        manager
            .generate_stream("remote:m", &test_messages(), |token| seen.push_str(token))
            .await
            .expect("stream should complete")
    };
    assert_eq!(text, "Hello");
    assert_eq!(seen, "Hello");
}

#[tokio::test]
async fn remote_stream_reports_which_endpoint_failed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(503).set_body_json(serde_json::json!({"error": "busy"})),
        )
        .mount(&server)
        .await;

    let manager = manager_for(&format!("{}/v1", server.uri()), None);
    let error = manager
        .generate("remote:m", &test_messages())
        .await
        .expect_err("503 is an error");
    assert!(
        error.to_string().contains("Remote"),
        "error should name the endpoint that failed: {error}"
    );
}

#[tokio::test]
async fn remote_agent_turn_returns_the_tools_the_model_asked_for() {
    let server = MockServer::start().await;
    let sse = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
        "\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.rs\\\"\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{",
        "\"arguments\":\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_bytes(sse),
        )
        .mount(&server)
        .await;

    let tools = vec![ToolDefinition {
        name: "read_file".to_string(),
        description: "Read a file".to_string(),
        parameters: serde_json::json!({"type": "object"}),
    }];
    let manager = manager_for(&format!("{}/v1", server.uri()), None);
    let step = manager
        .generate_stream_with_tools("remote:m", &test_messages(), &[], &tools, None, |_| {})
        .await
        .expect("tool-call stream should complete");

    assert_eq!(step.tool_calls.len(), 1);
    assert_eq!(step.tool_calls[0].name, "read_file");
    assert_eq!(step.tool_calls[0].arguments_json(), r#"{"path":"a.rs"}"#);
}
