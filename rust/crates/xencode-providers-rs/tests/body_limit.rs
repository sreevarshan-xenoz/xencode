//! A provider that sends an unbounded body must not be able to exhaust memory.
//! These use a real local HTTP server with a body just over the 8 MiB cap.

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use xencode_models_rs::OllamaClient;
use xencode_providers_rs::{ChatMessage, ProviderManager};

fn messages() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: "hi".into(),
    }]
}

fn manager(uri: &str) -> ProviderManager {
    ProviderManager::new(OllamaClient::new(uri, 60), None, None, None, None)
}

#[tokio::test]
async fn an_oversized_non_streaming_answer_is_refused() {
    let server = MockServer::start().await;
    // One byte over the cap. The payload need not be valid JSON: the size
    // check must fire before anything tries to parse it.
    let body = "x".repeat(8 * 1024 * 1024 + 1);
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let err = manager(&server.uri())
        .generate("llama3.1:8b", &messages())
        .await
        .expect_err("an unbounded body must not be buffered");
    assert!(err.to_string().contains("exceeded"), "{err}");
}

#[tokio::test]
async fn an_oversized_stream_is_refused() {
    let server = MockServer::start().await;
    let body = "x".repeat(8 * 1024 * 1024 + 1);
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let err = manager(&server.uri())
        .generate_stream("llama3.1:8b", &messages(), |_| {})
        .await
        .expect_err("an unbounded stream must not be accumulated");
    assert!(err.to_string().contains("exceeded"), "{err}");
}
