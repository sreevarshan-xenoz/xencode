//! Real loopback HTTP tests for the `bitnet:` route.
//!
//! bitnet.cpp is built on llama.cpp, so its HTTP server is llama-server and the
//! route reuses the OpenAI-compatible client. These tests drive that client
//! against a real listener on 127.0.0.1: they assert the path, the exact JSON
//! body, the absence of an API key, the reply text, SSE deltas and tool-call
//! shapes. No model generation is fabricated here: a real BitNet runtime and a
//! model small enough for this machine's 2 GB GPU are not available, so only the
//! HTTP contract is verified.

use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use xencode_models_rs::OllamaClient;
use xencode_providers_rs::{ChatMessage, Egress, ProviderError, ProviderManager, ToolDefinition};

fn test_messages() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".into(),
    }]
}

/// Manager pointed at `base_url`, with Ollama on an unreachable port so any
/// accidental fall-through is a network error instead of a quiet success.
fn manager_for(base_url: &str) -> ProviderManager {
    ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        None,
        None,
        None,
        None,
    )
    .with_bitnet(base_url)
}

fn completion_body(text: &str) -> serde_json::Value {
    serde_json::json!({ "choices": [{ "message": { "role": "assistant", "content": text } }] })
}

#[tokio::test]
async fn bitnet_prefix_without_an_endpoint_explains_how_to_set_one() {
    let manager = ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        None,
        None,
        None,
        None,
    );
    let error = manager
        .generate("bitnet:m", &test_messages())
        .await
        .expect_err("an unconfigured endpoint must not fall through to Ollama");
    let text = error.to_string();
    assert!(
        text.contains("No BitNet endpoint configured") && text.contains("bitnet_url"),
        "unhelpful error: {text}"
    );
    // A missing setting is not a retryable network condition.
    assert!(matches!(error, ProviderError::Api { status: None, .. }));
}

#[tokio::test]
async fn blank_or_whitespace_base_url_leaves_the_route_unconfigured() {
    let manager = manager_for("   ");
    let error = manager
        .generate("bitnet:m", &test_messages())
        .await
        .expect_err("whitespace is not an endpoint");
    assert!(
        error.to_string().contains("No BitNet endpoint configured"),
        "{error}"
    );
}

#[tokio::test]
async fn a_non_local_bitnet_endpoint_is_refused_before_any_connection() {
    let manager = manager_for("https://bitnet.example.invalid/v1");
    assert_eq!(manager.egress_of("bitnet:m"), Egress::Local);

    let error = manager
        .generate("bitnet:m", &test_messages())
        .await
        .expect_err("remote BitNet servers are not part of this slice");
    assert!(
        matches!(error, ProviderError::Egress(_)),
        "expected an egress refusal, got {error}"
    );
    let text = error.to_string();
    assert!(
        text.contains("not on this machine")
            && text.contains("loopback")
            && text.contains("bitnet_url"),
        "unhelpful refusal: {text}"
    );
}

#[tokio::test]
async fn bitnet_prefix_posts_the_exact_openai_body_to_chat_completions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("from bitnet.cpp")))
        .expect(1)
        .mount(&server)
        .await;

    let manager = manager_for(&format!("{}/v1", server.uri()));
    let reply = manager
        .generate("bitnet:m", &test_messages())
        .await
        .expect("the BitNet route should answer");
    assert_eq!(reply, "from bitnet.cpp");

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1, "exactly one POST to the local server");
    assert!(
        requests[0].headers.get("authorization").is_none(),
        "a keyless local server must not receive an Authorization header"
    );
}

#[tokio::test]
async fn bitnet_prefix_streams_tokens_and_ignores_the_done_marker() {
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
        .expect(1)
        .mount(&server)
        .await;

    let manager = manager_for(&format!("{}/v1", server.uri()));
    let mut seen = String::new();
    let text = {
        let seen = &mut seen;
        manager
            .generate_stream("bitnet:m", &test_messages(), |token| seen.push_str(token))
            .await
            .expect("the BitNet stream should complete")
    };
    assert_eq!(text, "Hello");
    assert_eq!(seen, "Hello");
}

#[tokio::test]
async fn bitnet_agent_turn_returns_the_tool_call_shape_from_the_stream() {
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
        .expect(1)
        .mount(&server)
        .await;

    let tools = vec![ToolDefinition {
        name: "read_file".to_string(),
        description: "Read a file".to_string(),
        parameters: serde_json::json!({"type": "object"}),
    }];
    let manager = manager_for(&format!("{}/v1", server.uri()));
    let step = manager
        .generate_stream_with_tools("bitnet:m", &test_messages(), &[], &tools, None, |_| {})
        .await
        .expect("the BitNet tool-call stream should complete");

    assert_eq!(step.tool_calls.len(), 1);
    assert_eq!(step.tool_calls[0].name, "read_file");
    assert_eq!(step.tool_calls[0].arguments_json(), r#"{"path":"a.rs"}"#);
}
