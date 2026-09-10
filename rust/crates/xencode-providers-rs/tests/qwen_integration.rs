//! Integration tests for the Qwen provider.
//!
//! These tests use `wiremock` to simulate the Qwen HTTP API, so no real
//! API key or network access is required.

use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use xencode_providers_rs::qwen::QwenProvider;
use xencode_providers_rs::{ChatMessage, ProviderError};

/// Helper: create a standard set of chat messages used in most tests.
fn test_messages() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: "What is Rust?".to_string(),
    }]
}

/// Helper: the standard JSON body the Qwen provider should send for a
/// non-streaming request.
fn expected_body(model: &str, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [
            {"role": "user", "content": "What is Rust?"}
        ],
        "stream": stream,
    })
}

/// Helper: a valid non-streaming OpenAI-compatible response payload.
fn success_response(text: &str) -> serde_json::Value {
    serde_json::json!({
        "choices": [
            {
                "message": {
                    "content": text,
                    "role": "assistant"
                }
            }
        ]
    })
}

// ── Non-streaming tests ──────────────────────────────────────────────────

#[tokio::test]
async fn qwen_generate_basic() {
    let mock_server = MockServer::start().await;

    // Expected request
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer test-api-key"))
        .and(header("Content-Type", "application/json"))
        .and(body_json(expected_body("qwen2.5:72b", false)))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(success_response("Rust is a systems programming language.")),
        )
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = QwenProvider::new(
        "test-api-key".to_string(),
        Some(format!("{}/v1", mock_server.uri())),
    );

    let result = provider.generate("qwen2.5:72b", &test_messages()).await;

    assert!(result.is_ok(), "generate() failed: {:?}", result.err());
    assert_eq!(result.unwrap(), "Rust is a systems programming language.");
}

#[tokio::test]
async fn qwen_generate_http_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = QwenProvider::new(
        "invalid-key".to_string(),
        Some(format!("{}/v1", mock_server.uri())),
    );

    let result = provider.generate("qwen2.5:72b", &test_messages()).await;

    assert!(result.is_err(), "expected error for 401 response");
    match result.unwrap_err() {
        ProviderError::Api { message: msg, .. } => {
            assert!(msg.contains("401"), "expected 401 in error: {msg}");
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
}

#[tokio::test]
async fn qwen_generate_empty_choices() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": []
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = QwenProvider::new(
        "test-key".to_string(),
        Some(format!("{}/v1", mock_server.uri())),
    );

    let result = provider.generate("qwen2.5:72b", &test_messages()).await;

    assert!(result.is_err(), "expected error for empty choices");
    match result.unwrap_err() {
        ProviderError::Parse(msg) => {
            assert!(msg.contains("empty"), "expected Parse error: {msg}");
        }
        other => panic!("expected Parse error, got: {other:?}"),
    }
}

#[tokio::test]
async fn qwen_generate_network_error() {
    // Use a port that won't be listening
    let provider = QwenProvider::new(
        "test-key".to_string(),
        Some("http://127.0.0.1:1/v1".to_string()),
    );

    let result = provider.generate("qwen2.5:72b", &test_messages()).await;

    assert!(result.is_err(), "expected network error");
    match result.unwrap_err() {
        ProviderError::Network(msg) => {
            assert!(
                msg.contains("Qwen") || msg.contains("failed") || msg.contains("error"),
                "expected Network error: {msg}"
            );
        }
        other => panic!("expected Network error, got: {other:?}"),
    }
}

// ── Streaming tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn qwen_stream_basic() {
    let mock_server = MockServer::start().await;

    // The Qwen streaming API returns SSE-like lines.
    // We send back multiple chunks.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer test-api-key"))
        .and(body_json(expected_body("qwen2.5:72b", true)))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Rust\"}}]}\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\" is\"}}]}\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\" safe\"}}]}\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\".\"}}]}\n\
                     data: [DONE]\n",
        ))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = QwenProvider::new(
        "test-api-key".to_string(),
        Some(format!("{}/v1", mock_server.uri())),
    );

    let mut tokens = Vec::new();
    let result = provider
        .generate_stream("qwen2.5:72b", &test_messages(), |token| {
            tokens.push(token.to_string());
        })
        .await;

    assert!(
        result.is_ok(),
        "generate_stream() failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap(), "Rust is safe.");
    assert_eq!(tokens, vec!["Rust", " is", " safe", "."]);
}

#[tokio::test]
async fn qwen_stream_http_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("Rate limited"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = QwenProvider::new(
        "test-key".to_string(),
        Some(format!("{}/v1", mock_server.uri())),
    );

    let result = provider
        .generate_stream("qwen2.5:72b", &test_messages(), |_| {})
        .await;

    assert!(result.is_err(), "expected error for 429");
    match result.unwrap_err() {
        ProviderError::Api { message: msg, .. } => {
            assert!(msg.contains("429"), "expected 429 in error: {msg}");
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
}

#[tokio::test]
async fn qwen_stream_network_error() {
    let provider = QwenProvider::new(
        "test-key".to_string(),
        Some("http://127.0.0.1:1/v1".to_string()),
    );

    let result = provider
        .generate_stream("qwen2.5:72b", &test_messages(), |_| {})
        .await;

    assert!(result.is_err(), "expected network error");
    match result.unwrap_err() {
        ProviderError::Network(msg) => {
            assert!(
                msg.contains("Qwen") || msg.contains("failed") || msg.contains("error"),
                "expected Network error: {msg}"
            );
        }
        other => panic!("expected Network error, got: {other:?}"),
    }
}
