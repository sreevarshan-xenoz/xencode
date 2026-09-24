//! The egress gate: what a refusal refuses, and what a fallback may not do.
//!
//! These run against `ProviderManager` rather than the classifier alone,
//! because the point of the change is that the decision is made *before* any
//! connection is opened. Every manager here points its local routes at an
//! unreachable port, so "it tried" and "it refused" are distinguishable by the
//! error type alone — no server, no network.

use xencode_models_rs::OllamaClient;
use xencode_providers_rs::{ChatMessage, Egress, EgressPolicy, ProviderError, ProviderManager};

fn test_messages() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello".into(),
    }]
}

fn manager() -> ProviderManager {
    ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        None,
        Some("qwen-key".to_string()),
        None,
        None,
    )
}

#[tokio::test]
async fn a_denied_cloud_route_is_refused_before_it_is_dialled() {
    let manager = manager().with_egress_policy(EgressPolicy { allow_cloud: false });

    let result = manager.generate("qwen:qwen2.5:72b", &test_messages()).await;

    // Not a network error mentioning Qwen: the request never left.
    assert!(
        matches!(result, Err(ProviderError::Egress(_))),
        "expected an egress refusal, got {result:?}"
    );
    assert!(
        result.unwrap_err().to_string().contains("egress denied"),
        "the refusal should say so in the message the user sees"
    );
}

#[tokio::test]
async fn a_denied_cloud_route_still_allows_local_models() {
    let manager = manager().with_egress_policy(EgressPolicy { allow_cloud: false });

    let result = manager.generate("qwen2.5:7b", &test_messages()).await;

    // Routed to Ollama and failed there — the policy is about destinations,
    // not a global kill switch.
    assert!(
        matches!(result, Err(ProviderError::Network(_))),
        "expected the local route to run and fail on the unreachable port, got {result:?}"
    );
}

#[tokio::test]
async fn with_no_policy_said_anything_everything_is_permitted() {
    // The default is the pre-existing behaviour: nothing is refused, so adding
    // the gate cannot regress a working configuration.
    let manager = manager();
    assert!(manager.egress_policy().allow_cloud);
    let result = manager.generate("qwen:qwen2.5:72b", &test_messages()).await;
    assert!(
        !matches!(result, Err(ProviderError::Egress(_))),
        "the default policy must not refuse anything, got {result:?}"
    );
}

#[test]
fn a_slashed_model_is_only_cloud_when_openrouter_is_configured() {
    let without_key = ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        None,
        None,
        None,
        None,
    );
    let with_key = ProviderManager::new(
        OllamaClient::new("http://127.0.0.1:1", 1),
        Some("or-key".to_string()),
        None,
        None,
        None,
    );
    assert_eq!(without_key.egress_of("openai/gpt-4o"), Egress::Local);
    assert_eq!(with_key.egress_of("openai/gpt-4o"), Egress::Cloud);
}

#[test]
fn a_remote_endpoint_is_judged_by_the_host_it_points_at() {
    let local = manager().with_remote("http://127.0.0.1:8080/v1", None);
    let off_machine = manager().with_remote("https://inference.example.invalid/v1", None);
    assert_eq!(local.egress_of("remote:qwen2.5-coder"), Egress::Local);
    assert_eq!(off_machine.egress_of("remote:qwen2.5-coder"), Egress::Cloud);
}

#[test]
fn a_local_turn_cannot_borrow_a_cloud_fallback() {
    let configured: Vec<String> = ["anthropic:claude-3-5-sonnet", "llama3.2:3b"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let (chain, skipped) = manager().fallback_chain("qwen2.5:7b", &configured);

    assert_eq!(
        chain,
        vec!["qwen2.5:7b".to_string(), "llama3.2:3b".to_string()]
    );
    assert_eq!(skipped, vec!["anthropic:claude-3-5-sonnet".to_string()]);
}

/// The same candidate list through the chain builder this replaced, in one
/// test, so the leak and its closure are shown side by side rather than
/// asserted separately: the textual order put a cloud provider one transient
/// Ollama error away from the whole conversation.
#[test]
fn the_textual_chain_is_where_the_leak_came_from() {
    let configured: Vec<String> = ["anthropic:claude-3-5-sonnet", "llama3.2:3b"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let unfiltered = xencode_providers_rs::retry::fallback_chain("qwen2.5:7b", &configured);
    assert_eq!(unfiltered.len(), 3, "primary plus both alternates");
    assert!(unfiltered.contains(&"anthropic:claude-3-5-sonnet".to_string()));

    let (filtered, _) = manager().fallback_chain("qwen2.5:7b", &configured);
    assert!(!filtered.contains(&"anthropic:claude-3-5-sonnet".to_string()));
}

#[test]
fn a_refused_cloud_turn_does_not_gain_a_local_fallback_either() {
    // Cloud primary, policy denies cloud: the primary stays in the chain so the
    // router can name it in the refusal, and the local candidate is dropped for
    // being a different class — silently swapping the user's model is not this
    // function's to decide.
    let deny = manager().with_egress_policy(EgressPolicy { allow_cloud: false });
    let configured: Vec<String> = ["qwen2.5:7b"].iter().map(|s| s.to_string()).collect();

    let (chain, skipped) = deny.fallback_chain("qwen:qwen3-max", &configured);

    assert_eq!(chain, vec!["qwen:qwen3-max".to_string()]);
    assert_eq!(skipped, vec!["qwen2.5:7b".to_string()]);
    assert!(
        !xencode_providers_rs::retry::is_fallback_eligible(&ProviderError::Egress(
            "no".to_string()
        )),
        "the refusal must end the turn, not hand it to the next candidate"
    );
}
