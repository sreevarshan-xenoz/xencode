//! Egress classification and policy — where a model id actually sends a prompt.
//!
//! Routing has always been decided by prefix, in three copies of the same
//! if-chain inside `ProviderManager`. This module reads the same prefix rules
//! once, in one place, and answers a question the routers never asked: does
//! this request leave the machine?
//!
//! Two rules are enforced from here:
//!   - [`EgressPolicy::check`] — a request may not go to an internet service
//!     unless the policy allows cloud routes at all.
//!   - [`chain_for`] — a fallback candidate may not change the turn's egress
//!     class. Without this, a local model that happens to be down moves the
//!     whole conversation to a cloud provider, which is the defect this module
//!     exists to close: the fallback chain was built from raw strings and every
//!     error but a decode failure advanced it.
//!
//! The classification is deliberately conservative: a route whose destination
//! cannot be proven local counts as cloud. That is why `remote:` looks at the
//! configured host rather than trusting its prefix.

use crate::{llamacpp_target, remote_target, ProviderError};

/// Where a request ends up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Egress {
    /// A server on this machine: Ollama, llama.cpp, or a `remote:` endpoint
    /// whose host is local.
    Local,
    /// Anything else — an internet service, or an endpoint we cannot place.
    Cloud,
}

impl Egress {
    pub fn label(self) -> &'static str {
        match self {
            Egress::Local => "local",
            Egress::Cloud => "off-machine",
        }
    }
}

/// What a request is allowed to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EgressPolicy {
    /// Whether a prompt may reach an internet service at all.
    pub allow_cloud: bool,
}

impl Default for EgressPolicy {
    /// The permissive policy, which filters nothing. A library or a test that
    /// has no opinion about where prompts go keeps the behaviour this gate
    /// replaced; the product itself builds its policy from the user's setting
    /// with [`EgressPolicy::new`].
    fn default() -> Self {
        Self { allow_cloud: true }
    }
}

impl EgressPolicy {
    /// The policy a setting implies: `true` permits cloud routes, `false`
    /// confines every prompt to this machine. Deliberately a separate value from
    /// "a cloud API key exists" — a key says who you are to a provider, not
    /// that your conversation may reach it.
    pub const fn new(allow_cloud: bool) -> Self {
        Self { allow_cloud }
    }

    /// Refuse a request whose destination the policy does not permit.
    pub fn check(&self, egress: Egress) -> Result<(), ProviderError> {
        if egress == Egress::Cloud && !self.allow_cloud {
            return Err(ProviderError::Egress(
                "cloud models are not allowed by the egress policy".to_string(),
            ));
        }
        Ok(())
    }
}

/// The configuration facts that change where a model id resolves to. A model
/// containing `/` only reaches OpenRouter when an OpenRouter key exists; a
/// `remote:` model reaches wherever the configured URL points.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoutingFacts<'a> {
    pub openrouter_key: bool,
    /// Host portion of the configured `remote:` endpoint, if one is set.
    pub remote_host: Option<&'a str>,
}

/// Classify a model id by the same prefix rules the routers use, in the same
/// order — a change to one must be a change to the other.
pub fn classify(model: &str, facts: RoutingFacts<'_>) -> Egress {
    if model.starts_with("anthropic:")
        || model.starts_with("qwen:")
        || model.starts_with("google_gemini:")
    {
        return Egress::Cloud;
    }
    if llamacpp_target(model).is_some() {
        return Egress::Local;
    }
    if remote_target(model).is_some() {
        return match facts.remote_host {
            Some(host) if host_is_local(host) => Egress::Local,
            _ => Egress::Cloud,
        };
    }
    if model.contains('/') && facts.openrouter_key {
        return Egress::Cloud;
    }
    Egress::Local
}

/// The host of an HTTP(S) URL, without scheme, port or path.
pub fn url_host(url: &str) -> Option<&str> {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = rest.split('/').next().unwrap_or("");
    let host = authority
        .rsplit_once('@')
        .map(|(_, h)| h)
        .unwrap_or(authority);
    // An IPv6 literal keeps its brackets; everything after the last `]` (or the
    // first plain `:`) is a port.
    if let Some(open) = host.strip_prefix('[') {
        return open.split_once(']').map(|(h, _)| h);
    }
    Some(host.split(':').next().unwrap_or(""))
}

/// True for a host that cannot leave this machine.
pub fn host_is_local(host: &str) -> bool {
    let host = host.trim_matches(['[', ']']);
    host.eq_ignore_ascii_case("localhost")
        || host.eq_ignore_ascii_case("localhost.localdomain")
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host == "::1"
        || host == "0.0.0.0"
        || host == "::"
        || host.strip_prefix("127.").is_some_and(|rest| {
            !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit() || b == b'.')
        })
}

/// The fallback order a turn may actually walk: the primary plus any configured
/// candidate that leaves the prompt in the same place the primary would, with
/// everything else reported back so the caller can say so out loud.
pub fn chain_for(
    primary: &str,
    configured: &[String],
    policy: EgressPolicy,
    facts: RoutingFacts<'_>,
) -> (Vec<String>, Vec<String>) {
    let full = crate::retry::fallback_chain(primary, configured);
    let Some(first) = full.first() else {
        return (Vec::new(), Vec::new());
    };
    let primary_egress = classify(first, facts);
    let mut kept = Vec::with_capacity(full.len());
    let mut dropped = Vec::new();
    for (index, candidate) in full.iter().enumerate() {
        let egress = classify(candidate, facts);
        let allowed = if index == 0 {
            // The primary is the user's own choice: dropping it here would
            // silently run a different model than they asked for. The policy
            // check at the router refuses the turn instead.
            true
        } else {
            egress == primary_egress && policy.check(egress).is_ok()
        };
        if allowed {
            kept.push(candidate.clone());
        } else {
            dropped.push(candidate.clone());
        }
    }
    (kept, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn cloud_key() -> RoutingFacts<'static> {
        RoutingFacts {
            openrouter_key: true,
            remote_host: None,
        }
    }

    #[test]
    fn cloud_prefixes_classify_as_cloud() {
        for model in [
            "anthropic:claude-3-5-sonnet",
            "qwen:qwen3-max",
            "google_gemini:gemini-2.0-flash",
        ] {
            assert_eq!(
                classify(model, RoutingFacts::default()),
                Egress::Cloud,
                "{model}"
            );
        }
    }

    #[test]
    fn local_servers_and_bare_models_classify_as_local() {
        let no_key = RoutingFacts::default();
        for model in [
            "qwen2.5:7b",
            "llama3.2:3b",
            "llamacpp:qwen2.5-coder",
            "llama.cpp:qwen2.5-coder",
            "llama:qwen2.5-coder",
        ] {
            assert_eq!(classify(model, no_key), Egress::Local, "{model}");
        }
    }

    #[test]
    fn a_slashed_model_is_cloud_only_when_openrouter_is_configured() {
        // With no key the router falls through to local Ollama, so "where it
        // goes" really does depend on configuration, not just the name.
        assert_eq!(
            classify("openai/gpt-4o", RoutingFacts::default()),
            Egress::Local
        );
        assert_eq!(classify("openai/gpt-4o", cloud_key()), Egress::Cloud);
    }

    #[test]
    fn remote_follows_the_configured_host_not_its_prefix() {
        let local = RoutingFacts {
            openrouter_key: false,
            remote_host: Some("127.0.0.1"),
        };
        assert_eq!(classify("remote:qwen2.5-coder", local), Egress::Local);
        let tunnel = RoutingFacts {
            openrouter_key: false,
            remote_host: Some("inference.example.com"),
        };
        assert_eq!(classify("remote:qwen2.5-coder", tunnel), Egress::Cloud);
        // Nothing configured at all: the route would fail, and it is not
        // provably local.
        assert_eq!(
            classify("remote:qwen2.5-coder", RoutingFacts::default()),
            Egress::Cloud
        );
    }

    #[test]
    fn hosts_are_read_out_of_urls_and_looked_at_closely() {
        assert_eq!(url_host("http://127.0.0.1:11434/v1"), Some("127.0.0.1"));
        assert_eq!(
            url_host("https://api.openrouter.ai/api/v1"),
            Some("api.openrouter.ai")
        );
        assert_eq!(url_host("localhost:8080"), Some("localhost"));
        assert_eq!(url_host("http://[::1]:8080/v1"), Some("::1"));
        assert_eq!(
            url_host("http://user@example.invalid:443/v1"),
            Some("example.invalid")
        );
        assert!(host_is_local("localhost"));
        assert!(host_is_local("127.0.0.5"));
        assert!(host_is_local("::1"));
        assert!(host_is_local("box.local"));
        // A name that merely starts like a loopback address is not one.
        assert!(!host_is_local("127.example.com"));
        assert!(!host_is_local("localhost.evil.invalid"));
    }

    #[test]
    fn the_policy_refuses_cloud_and_permits_local() {
        let deny = EgressPolicy { allow_cloud: false };
        assert!(deny.check(Egress::Cloud).is_err());
        assert!(deny.check(Egress::Local).is_ok());
        assert!(EgressPolicy::default().check(Egress::Cloud).is_ok());
    }

    #[test]
    fn a_local_turn_cannot_fall_back_to_a_cloud_provider() {
        let configured: Vec<String> = ["qwen2.5:7b", "anthropic:claude-3-5-sonnet", "llama3.2:3b"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (chain, dropped) = chain_for(
            "qwen2.5:7b",
            &configured,
            EgressPolicy::default(),
            RoutingFacts::default(),
        );
        assert_eq!(
            chain,
            vec!["qwen2.5:7b".to_string(), "llama3.2:3b".to_string()]
        );
        assert_eq!(dropped, vec!["anthropic:claude-3-5-sonnet".to_string()]);
    }

    #[test]
    fn a_cloud_turn_keeps_its_cloud_fallbacks() {
        let configured: Vec<String> = ["google_gemini:gemini-2.0-flash", "qwen:qwen3-max"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (chain, dropped) = chain_for(
            "anthropic:claude-3-5-sonnet",
            &configured,
            EgressPolicy::default(),
            RoutingFacts::default(),
        );
        assert_eq!(chain.len(), 3);
        assert!(dropped.is_empty());
    }

    #[test]
    fn a_denied_policy_drops_cloud_candidates_from_a_cloud_turn() {
        let configured: Vec<String> = ["google_gemini:gemini-2.0-flash"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let deny = EgressPolicy { allow_cloud: false };
        let (chain, dropped) = chain_for(
            "anthropic:claude-3-5-sonnet",
            &configured,
            deny,
            RoutingFacts::default(),
        );
        // The primary stays: refusing the turn is the router's job, and
        // quietly swapping the model the user picked is not anyone's.
        assert_eq!(chain, vec!["anthropic:claude-3-5-sonnet".to_string()]);
        assert_eq!(dropped, vec!["google_gemini:gemini-2.0-flash".to_string()]);
    }

    #[test]
    fn an_empty_primary_still_returns_a_usable_chain() {
        let (chain, dropped) = chain_for("", &[], EgressPolicy::default(), RoutingFacts::default());
        assert!(chain.is_empty());
        assert!(dropped.is_empty());
    }
}
