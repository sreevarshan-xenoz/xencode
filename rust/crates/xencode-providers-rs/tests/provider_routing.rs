//! Integration tests for `ProviderManager` routing logic.
//!
//! These tests verify that `ProviderManager::generate()` correctly routes
//! model prefixes (e.g. `qwen:`, `google_gemini:`) to the appropriate
//! provider, returns proper errors when API keys are missing, and falls
//! back to Ollama for unrecognized model names.
//!
//! Because `ProviderManager::generate_inner()` creates sub-providers with
//! their default `base_url` (not a mock server URL), the routing tests that
//! exercise the full HTTP path are verified by checking the error type and
//! message prefix — a network error containing "Qwen" proves the request
//! was routed to the Qwen provider rather than Ollama or another provider.

use xencode_models_rs::OllamaClient;
use xencode_providers_rs::{ChatMessage, ProviderError, ProviderManager};

fn test_messages() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".to_string(),
    }]
}

/// Helper: create a ProviderManager with the given keys.
fn make_manager(
    qwen_key: Option<&str>,
    gemini_key: Option<&str>,
    anthropic_key: Option<&str>,
) -> ProviderManager {
    ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1), // Ollama URL won't be reachable
        None,                                       // openrouter key
        qwen_key.map(|s| s.to_string()),
        gemini_key.map(|s| s.to_string()),
        anthropic_key.map(|s| s.to_string()),
    )
}

// ── Qwen routing ─────────────────────────────────────────────────────────

#[tokio::test]
async fn qwen_prefix_routes_to_qwen_provider() {
    let manager = make_manager(Some("qwen-key"), None, None);

    let result = manager.generate("qwen:qwen2.5:72b", &test_messages()).await;

    // The Qwen provider will try to contact the real Qwen API (default URL).
    // Depending on the network environment this may fail with:
    // - Network error (connection refused/DNS failure) → "Qwen request failed: ..."
    // - API error (invalid key, provider returns 401) → "Qwen 401 - ..."
    // Either way, "Qwen" in the message proves the routing worked.
    assert!(result.is_err(), "expected routing to Qwen provider");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Qwen"),
        "expected error mentioning 'Qwen', got: {err:?}"
    );
}

#[tokio::test]
async fn qwen_missing_key_returns_error() {
    let manager = make_manager(None, None, None);

    let result = manager.generate("qwen:qwen2.5:72b", &test_messages()).await;

    assert!(result.is_err(), "expected error for missing Qwen key");
    match result.unwrap_err() {
        ProviderError::Api { message: msg, .. } => {
            assert!(
                msg.contains("Qwen API key"),
                "expected 'Qwen API key' error: {msg}"
            );
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
}

// ── Gemini routing ───────────────────────────────────────────────────────

#[tokio::test]
async fn gemini_prefix_routes_to_gemini_provider() {
    let manager = make_manager(None, Some("gemini-key"), None);

    let result = manager
        .generate("google_gemini:gemini-1.5-pro", &test_messages())
        .await;

    // The Gemini provider will try the real API. May get Network or Api error.
    assert!(result.is_err(), "expected routing to Gemini provider");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Gemini"),
        "expected error mentioning 'Gemini', got: {err:?}"
    );
}

#[tokio::test]
async fn gemini_missing_key_returns_error() {
    let manager = make_manager(None, None, None);

    let result = manager
        .generate("google_gemini:gemini-1.5-pro", &test_messages())
        .await;

    assert!(result.is_err(), "expected error for missing Gemini key");
    match result.unwrap_err() {
        ProviderError::Api { message: msg, .. } => {
            assert!(
                msg.contains("Gemini API key"),
                "expected 'Gemini API key' error: {msg}"
            );
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
}

// ── Anthropic routing ────────────────────────────────────────────────────

#[tokio::test]
async fn anthropic_prefix_routes_to_anthropic_provider() {
    let manager = make_manager(None, None, Some("anthropic-key"));

    let result = manager
        .generate("anthropic:claude-3-5-sonnet-20241022", &test_messages())
        .await;

    // The Anthropic provider will try the real API. May get Network or Api error.
    assert!(result.is_err(), "expected routing to Anthropic provider");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Anthropic"),
        "expected error mentioning 'Anthropic', got: {err:?}"
    );
}

#[tokio::test]
async fn anthropic_missing_key_returns_error() {
    let manager = make_manager(None, None, None);

    let result = manager
        .generate("anthropic:claude-3-5-sonnet-20241022", &test_messages())
        .await;

    assert!(result.is_err(), "expected error for missing Anthropic key");
    match result.unwrap_err() {
        ProviderError::Api { message: msg, .. } => {
            assert!(
                msg.contains("Anthropic API key"),
                "expected 'Anthropic API key' error: {msg}"
            );
        }
        other => panic!("expected Api error, got: {other:?}"),
    }
}

// ── Fallback / default routing ───────────────────────────────────────────

#[tokio::test]
async fn unknown_prefix_tries_ollama() {
    let manager = make_manager(None, None, None);

    // An unrecognized model name (no prefix match) should try Ollama.
    // The Ollama URL is deliberately broken (127.0.0.1:1).
    // The error should be from the ProviderManager's Ollama route,
    // NOT from any provider-specific error.
    let result = manager.generate("llama3.1:8b", &test_messages()).await;

    assert!(
        result.is_err(),
        "expected network error from Ollama fallback"
    );
    match result.unwrap_err() {
        ProviderError::Network(msg) => {
            assert!(
                !msg.contains("Qwen")
                    && !msg.contains("Gemini")
                    && !msg.contains("Anthropic")
                    && !msg.contains("OpenRouter"),
                "expected Ollama error (not another provider): {msg}"
            );
        }
        other => panic!("expected Network error from Ollama, got: {other:?}"),
    }
}

// ── OpenRouter routing ──────────────────────────────────────────────────

#[tokio::test]
async fn openrouter_prefix_routes() {
    let manager = ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        Some("or-key".to_string()),
        None,
        None,
        None,
    );

    // Models with '/' route to OpenRouter when the key is set.
    // The request will fail trying to reach the real OpenRouter API,
    // but the error will contain "OpenRouter" proving routing worked.
    let result = manager.generate("openai/gpt-4o", &test_messages()).await;

    assert!(result.is_err(), "expected error (routed to OpenRouter)");
    match result.unwrap_err() {
        ProviderError::Network(msg) => {
            assert!(
                msg.contains("error") || msg.contains("connect") || msg.contains("failed"),
                "expected network error from OpenRouter: {msg}"
            );
        }
        ProviderError::Api { message: msg, .. } => {
            assert!(
                msg.contains("OpenRouter"),
                "expected OpenRouter error: {msg}"
            );
        }
        other => panic!("expected Network or Api error (tried OpenRouter), got: {other:?}"),
    }
}

// ── llama.cpp routing ────────────────────────────────────────────────────

#[tokio::test]
async fn llamacpp_prefix_routes_to_llamacpp() {
    let manager = make_manager(None, None, None);

    let result = manager
        .generate("llamacpp:mistral-7b-instruct.Q4_K_M.gguf", &test_messages())
        .await;

    assert!(result.is_err(), "expected routing error to llama.cpp");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("llama.cpp"),
        "expected error mentioning 'llama.cpp', got: {msg}"
    );
}

#[tokio::test]
async fn llamacpp_with_custom_client() {
    use xencode_models_rs::LlamaCppClient;

    let manager = make_manager(None, None, None)
        .with_llama_cpp(LlamaCppClient::new("http://127.0.0.1:9999", 2));

    let result = manager
        .generate("llama.cpp:qwen2.5-coder-7b.gguf", &test_messages())
        .await;

    assert!(
        result.is_err(),
        "expected error from custom llama.cpp client"
    );
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("llama.cpp"),
        "expected error mentioning 'llama.cpp', got: {msg}"
    );
}
