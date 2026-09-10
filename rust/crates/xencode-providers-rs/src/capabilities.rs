//! Model capabilities: what the context budgeter needs to know per model.
//!
//! Resolution is offline and conservative by design:
//!
//! - `context_window` is `Some` only for model families whose window is
//!   stable public knowledge. Everything else — local servers (where the
//!   effective window is the server's `--ctx-size`), Ollama (where it is the
//!   server's `num_ctx`), ambiguous cloud families — resolves to `None`,
//!   meaning "defer to the hardware profile default". A wrong number is
//!   worse than no number: over-claiming silently drops context, and the
//!   table cannot track every monthly model release.
//! - `supports_tools` is `true` exactly for the routes with plumbed
//!   `tool_calls` support, so the future agent loop can skip tool offers
//!   where they would be ignored.
//!
//! Precedence at the call site is therefore:
//! `known model window → hardware profile default`.

/// What the context budgeter and agent loop need to know about a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Effective context window in tokens, when confidently known.
    /// `None` = unknown → caller falls back to the hardware profile.
    pub context_window: Option<u32>,
    /// Max output tokens, when confidently known. `None` almost everywhere
    /// for now (local servers cap via `--n-predict`; cloud caps vary).
    pub max_output_tokens: Option<u32>,
    /// This route parses and echoes `tool_calls` (llama.cpp, Ollama,
    /// OpenRouter, Qwen). Gemini/Anthropic degrade to single-shot text.
    pub supports_tools: bool,
    /// Conservative: `false` until a vision pipeline exists.
    pub supports_vision: bool,
    /// Best-effort reasoning-model heuristic, advisory only.
    pub supports_reasoning: bool,
    /// All routes stream.
    pub supports_streaming: bool,
}

/// Resolve capabilities for a model id in any routed form
/// (`llamacpp:…`, `qwen:…`, `openrouter` `a/b`, plain Ollama tags, …).
/// Never fails and never touches the network.
pub fn capabilities_for(model: &str) -> ModelCapabilities {
    let lower = model.to_lowercase();
    ModelCapabilities {
        context_window: known_context_window(&lower),
        max_output_tokens: None,
        supports_tools: route_supports_tools(model),
        supports_vision: false,
        supports_reasoning: is_reasoning_family(&lower),
        supports_streaming: true,
    }
}

/// Stable, public-knowledge context windows. Deliberately narrow: ambiguous
/// families (generic `qwen`, `gemini`, `mistral`, plain `llama`) and every
/// local route resolve to `None` (see module docs).
fn known_context_window(lower_model: &str) -> Option<u32> {
    // Order matters: specific families before generic substrings.
    if lower_model.contains("qwen3-coder") {
        return Some(256_000);
    }
    if lower_model.contains("claude") {
        return Some(200_000);
    }
    if lower_model.contains("llama-3.1") || lower_model.contains("llama-3.3") {
        return Some(128_000);
    }
    if lower_model.contains("gpt-oss") {
        return Some(128_000);
    }
    if lower_model.contains("gpt-4o") {
        return Some(128_000);
    }
    if lower_model.contains("devstral") {
        return Some(128_000);
    }
    if lower_model.contains("deepseek") {
        return Some(64_000);
    }
    None
}

/// True exactly for the routes whose streaming path parses `tool_calls`.
fn route_supports_tools(model: &str) -> bool {
    if model.starts_with("anthropic:") || model.starts_with("google_gemini:") {
        return false;
    }
    // Everything else routes through llama.cpp, Ollama, OpenRouter, or Qwen,
    // all of which carry native tool calls.
    true
}

/// Best-effort reasoning-model sniffing for UI hints, not for budgeting.
fn is_reasoning_family(lower_model: &str) -> bool {
    [
        "-r1",
        "reasoner",
        "reasoning",
        "thinking",
        "qwq",
        "/o1",
        "/o3",
        "o1-",
        "o3-",
    ]
    .iter()
    .any(|m| lower_model.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_families_resolve_windows() {
        assert_eq!(
            capabilities_for("anthropic:claude-sonnet-4").context_window,
            Some(200_000)
        );
        assert_eq!(
            capabilities_for("openrouter/anthropic/claude-opus-4").context_window,
            Some(200_000)
        );
        assert_eq!(
            capabilities_for("openrouter/deepseek/deepseek-chat").context_window,
            Some(64_000)
        );
        assert_eq!(
            capabilities_for("llama-3.1:70b").context_window,
            Some(128_000)
        );
        assert_eq!(
            capabilities_for("qwen:qwen3-coder-480b").context_window,
            Some(256_000)
        );
        assert_eq!(
            capabilities_for("openrouter/openai/gpt-oss-120b").context_window,
            Some(128_000)
        );
    }

    #[test]
    fn ambiguous_and_local_routes_defer_to_profile() {
        // Local servers: the effective window is the server's --ctx-size /
        // num_ctx, unknowable statically — must be None, never a guess.
        assert_eq!(capabilities_for("llamacpp:qwen3-4b").context_window, None);
        assert_eq!(capabilities_for("qwen2.5:7b").context_window, None);
        // Ambiguous cloud families: windows span 32k–1M across variants.
        assert_eq!(
            capabilities_for("google_gemini:gemini-2.0-flash").context_window,
            None
        );
        assert_eq!(capabilities_for("qwen:qwen-max").context_window, None);
        assert_eq!(
            capabilities_for("some-future-model-xyz").context_window,
            None
        );
    }

    #[test]
    fn tool_support_matches_plumbed_routes() {
        assert!(capabilities_for("llamacpp:qwen3-4b").supports_tools);
        assert!(capabilities_for("qwen2.5:7b").supports_tools);
        assert!(capabilities_for("openai/gpt-4o").supports_tools);
        assert!(capabilities_for("qwen:qwen-max").supports_tools);
        assert!(!capabilities_for("anthropic:claude-sonnet-4").supports_tools);
        assert!(!capabilities_for("google_gemini:gemini-2.0-flash").supports_tools);
    }

    #[test]
    fn streaming_everywhere_vision_nowhere() {
        for m in [
            "llamacpp:qwen3-4b",
            "anthropic:claude-sonnet-4",
            "google_gemini:gemini-2.0-flash",
        ] {
            let caps = capabilities_for(m);
            assert!(caps.supports_streaming);
            assert!(!caps.supports_vision);
        }
    }
}
