//! Integration tests for the Gemini provider.
//!
//! These tests use `wiremock` to simulate the Gemini HTTP API, so no real
//! API key or network access is required.

use wiremock::matchers::{body_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use xencode_providers_rs::gemini::GeminiProvider;
use xencode_providers_rs::{ChatMessage, ProviderError};

/// Helper: standard test messages.
fn test_messages() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: "Tell me about Rust.".to_string(),
    }]
}

/// Helper: expected Gemini request body for the standard test messages.
fn expected_body() -> serde_json::Value {
    serde_json::json!({
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Tell me about Rust."}]
            }
        ],
        "generationConfig": {
            "candidateCount": 1,
            "stopSequences": [],
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    })
}

/// Helper: a valid non-streaming Gemini response.
fn success_response(text: &str) -> serde_json::Value {
    serde_json::json!({
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}]
                }
            }
        ]
    })
}

// ── Non-streaming tests ──────────────────────────────────────────────────

#[tokio::test]
async fn gemini_generate_basic() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-pro:generateContent"))
        .and(query_param("key", "test-gemini-key"))
        .and(header("Content-Type", "application/json"))
        .and(body_json(expected_body()))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response(
            "Rust is a systems programming language focused on safety.",
        )))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = GeminiProvider::new(
        "test-gemini-key".to_string(),
        Some(format!("{}/v1beta", mock_server.uri())),
    );

    let result = provider
        .generate("gemini-pro", &test_messages(), None, None)
        .await;

    assert!(result.is_ok(), "generate() failed: {:?}", result.err());
    assert_eq!(
        result.unwrap(),
        "Rust is a systems programming language focused on safety."
    );
}

#[tokio::test]
async fn gemini_generate_with_max_tokens() {
    let mock_server = MockServer::start().await;

    // Expected body with max_tokens set
    let mut custom_body = expected_body();
    custom_body["generationConfig"]["maxOutputTokens"] = serde_json::json!(512);

    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-pro:generateContent"))
        .and(query_param("key", "test-key"))
        .and(body_json(custom_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response("Short answer.")))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = GeminiProvider::new(
        "test-key".to_string(),
        Some(format!("{}/v1beta", mock_server.uri())),
    );

    let result = provider
        .generate("gemini-pro", &test_messages(), Some(512), None)
        .await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Short answer.");
}

#[tokio::test]
async fn gemini_generate_http_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-pro:generateContent"))
        .respond_with(ResponseTemplate::new(403).set_body_string("Forbidden"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = GeminiProvider::new(
        "bad-key".to_string(),
        Some(format!("{}/v1beta", mock_server.uri())),
    );

    let result = provider
        .generate("gemini-pro", &test_messages(), None, None)
        .await;

    assert!(result.is_err(), "expected error for 403");
    match result.unwrap_err() {
        ProviderError::Api { message: msg, .. } => {
            assert!(msg.contains("403"), "expected 403 in error: {msg}");
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
}

#[tokio::test]
async fn gemini_generate_blocked_response() {
    let mock_server = MockServer::start().await;

    // Gemini can block requests and return a prompt_feedback with block_reason
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-pro:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": [],
            "prompt_feedback": {
                "block_reason": "SAFETY"
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = GeminiProvider::new(
        "test-key".to_string(),
        Some(format!("{}/v1beta", mock_server.uri())),
    );

    let result = provider
        .generate("gemini-pro", &test_messages(), None, None)
        .await;

    assert!(result.is_err(), "expected error for blocked request");
    match result.unwrap_err() {
        ProviderError::Api { message: msg, .. } => {
            assert!(
                msg.contains("blocked") || msg.contains("SAFETY"),
                "expected blocked error: {msg}"
            );
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
}

#[tokio::test]
async fn gemini_generate_empty_candidates() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-pro:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "candidates": []
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = GeminiProvider::new(
        "test-key".to_string(),
        Some(format!("{}/v1beta", mock_server.uri())),
    );

    let result = provider
        .generate("gemini-pro", &test_messages(), None, None)
        .await;

    // Empty candidates returns empty string (not an error for Gemini)
    assert!(
        result.is_ok(),
        "empty candidates should return empty string"
    );
    assert_eq!(result.unwrap(), "");
}

#[tokio::test]
async fn gemini_generate_network_error() {
    let provider = GeminiProvider::new(
        "test-key".to_string(),
        Some("http://127.0.0.1:1/v1beta".to_string()),
    );

    let result = provider
        .generate("gemini-pro", &test_messages(), None, None)
        .await;

    assert!(result.is_err(), "expected network error");
    match result.unwrap_err() {
        ProviderError::Network(msg) => {
            assert!(
                msg.contains("Gemini") || msg.contains("failed"),
                "expected Network error: {msg}"
            );
        }
        other => panic!("expected Network error, got: {other:?}"),
    }
}

// ── Streaming tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_stream_basic() {
    let mock_server = MockServer::start().await;

    // Gemini streaming uses SSE data chunks with candidates.
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-pro:streamGenerateContent"))
        .and(query_param("alt", "sse"))
        .and(query_param("key", "test-key"))
        .and(body_json(expected_body()))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Rust\"}]}}]}\n\
                     data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" is\"}]}}]}\n\
                     data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" fast\"}]}}]}\n",
        ))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = GeminiProvider::new(
        "test-key".to_string(),
        Some(format!("{}/v1beta", mock_server.uri())),
    );

    let mut tokens = Vec::new();
    let result = provider
        .generate_stream("gemini-pro", &test_messages(), None, None, |token| {
            tokens.push(token.to_string());
        })
        .await;

    assert!(
        result.is_ok(),
        "generate_stream() failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap(), "Rust is fast");
    assert_eq!(tokens, vec!["Rust", " is", " fast"]);
}

#[tokio::test]
async fn gemini_stream_http_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-pro:streamGenerateContent"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .expect(1)
        .mount(&mock_server)
        .await;

    let provider = GeminiProvider::new(
        "test-key".to_string(),
        Some(format!("{}/v1beta", mock_server.uri())),
    );

    let result = provider
        .generate_stream("gemini-pro", &test_messages(), None, None, |_| {})
        .await;

    assert!(result.is_err(), "expected error for 500");
    match result.unwrap_err() {
        ProviderError::Api { message: msg, .. } => {
            assert!(msg.contains("500"), "expected 500 in error: {msg}");
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
}

#[tokio::test]
async fn gemini_stream_network_error() {
    let provider = GeminiProvider::new(
        "test-key".to_string(),
        Some("http://127.0.0.1:1/v1beta".to_string()),
    );

    let result = provider
        .generate_stream("gemini-pro", &test_messages(), None, None, |_| {})
        .await;

    assert!(result.is_err(), "expected network error");
    match result.unwrap_err() {
        ProviderError::Network(msg) => {
            assert!(
                msg.contains("Gemini") || msg.contains("failed"),
                "expected Network error: {msg}"
            );
        }
        other => panic!("expected Network error, got: {other:?}"),
    }
}
