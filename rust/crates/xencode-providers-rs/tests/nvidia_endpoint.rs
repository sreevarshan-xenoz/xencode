//! Integration tests for the `nvidia:` route — NVIDIA NIM's fixed
//! OpenAI-compatible `/chat/completions` endpoint behind a model prefix.
//!
//! The route is only useful if `nvidia:<vendor/model>` really posts to
//! `integrate.api.nvidia.com` with the bearer token, reports a missing key
//! instead of dialling anonymously, and classifies as cloud. wiremock
//! supplies the server, so no key and no network are involved — and no test
//! here ever writes a real key.

use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use xencode_models_rs::OllamaClient;
use xencode_providers_rs::{ChatMessage, ProviderManager, NVIDIA_BASE_URL};

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
    .with_nvidia_endpoint(base_url, key.map(|s| s.to_string()))
}

fn completion_body(text: &str) -> serde_json::Value {
    serde_json::json!({ "choices": [{ "message": { "role": "assistant", "content": text } }] })
}

#[tokio::test]
async fn nvidia_prefix_without_a_key_explains_how_to_set_one() {
    let manager = ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        None,
        None,
        None,
        None,
    );
    let error = manager
        .generate(
            "nvidia:mistralai/mistral-7b-instruct-v0.3",
            &test_messages(),
        )
        .await
        .expect_err("a keyless route must not fall through to Ollama");
    let text = error.to_string();
    assert!(
        text.contains("No NVIDIA API key configured") && text.contains("nvidia_api_key"),
        "unhelpful error: {text}"
    );
}

#[tokio::test]
async fn blank_key_leaves_the_route_unconfigured() {
    let manager = ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        None,
        None,
        None,
        None,
    )
    .with_nvidia(Some("   ".to_string()));
    let error = manager
        .generate("nvidia:m", &test_messages())
        .await
        .expect_err("whitespace is not a key");
    assert!(
        error.to_string().contains("No NVIDIA API key configured"),
        "{error}"
    );
}

#[tokio::test]
async fn nvidia_prefix_posts_the_inner_model_with_the_bearer_token() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer fake-nim-key"))
        .and(body_json(serde_json::json!({
            "model": "mistralai/mistral-7b-instruct-v0.3",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(completion_body("from NIM")))
        .expect(1)
        .mount(&server)
        .await;

    let url = format!("{}/v1", server.uri());
    let manager = manager_for(&url, Some("fake-nim-key"));
    let reply = manager
        .generate(
            "nvidia:mistralai/mistral-7b-instruct-v0.3",
            &test_messages(),
        )
        .await
        .expect("nvidia route should answer");
    assert_eq!(reply, "from NIM");
}

#[tokio::test]
async fn nvidia_failures_name_the_endpoint_that_failed() {
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
        .generate("nvidia:m", &test_messages())
        .await
        .expect_err("503 is an error");
    assert!(
        error.to_string().contains("Nvidia"),
        "error should name the endpoint that failed: {error}"
    );
}

#[test]
fn the_nvidia_root_is_the_documented_integrate_host() {
    assert_eq!(NVIDIA_BASE_URL, "https://integrate.api.nvidia.com/v1");
}
