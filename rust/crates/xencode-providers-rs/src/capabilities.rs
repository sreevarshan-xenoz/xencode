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
//! The local route is the one place the table cannot answer, and the server
//! can: a running `llama-server` reports the window it was started with, so
//! [`effective_context_window`] prefers that over any table entry when the
//! model actually goes to a llama.cpp server. Ollama does not report a window
//! per process, so on that route the number carried here is the one this
//! program asked Ollama to use — see [`crate::ollama_window_asked`].
//!
//! Precedence at the call site is therefore:
//! `window the serving side runs with (llama.cpp) or was asked to use (Ollama)
//! → known model window → hardware profile default`.

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

/// Resolve the window to budget against, preferring what the server says.
///
/// `server_reported` is the running window read from a llama.cpp server's
/// `/props` (see `xencode_models_rs::llamacpp::context_window_from_props`).
/// It governs **only** when the model actually routes to a llama.cpp server —
/// that number describes the process it was read from, and applying a window
/// learned elsewhere would budget for space the model does not have. For any
/// other route the reported value is dropped and the static table answers.
///
/// On the llama.cpp route the reported value wins *over the table*, not just
/// over a `None`: `llama:llama-3.1-8b` is a family the table knows as 128k
/// while the server may well have been started with `-c 8192`. A window
/// larger than the server's is what silently truncates context.
///
/// The Ollama route takes a reported number too, with one difference in where
/// it came from: Ollama is not asked what window it is running with, because it
/// answers that per model rather than per process, and the value carried here is
/// the window this program decided to request on every request for that model
/// (`options.num_ctx`, which Ollama then honours). The two sources must not be
/// mixed up in a caller's head — one is a measurement of a running server, the
/// other is the thing we asked for and can therefore rely on.
pub fn effective_context_window(model: &str, server_reported: Option<u32>) -> Option<u32> {
    if crate::routes_to_llamacpp(model) || crate::routes_to_ollama(model) {
        if let Some(reported) = server_reported {
            return Some(reported);
        }
    }
    capabilities_for(model).context_window
}

/// What to ask a local Ollama for as the window, before the model itself gets a
/// say (see [`crate::ollama_request_for`], which reduces this further to what
/// the weights were trained on).
///
/// Never the family table's number on its own: `qwen3-coder` is a 256,000-token
/// model in the table and a set of weights someone is loading on a laptop here,
/// and asking Ollama for a window the machine cannot hold spends the turn on a
/// server that is trying to allocate a cache it has no room for. The profile is
/// the measured answer about this machine, so it is the ceiling; the table only
/// gets to lower the number, never raise it above what the machine reported.
pub fn ollama_window_asked(table: Option<u32>, profile_tokens: u32) -> u32 {
    table.unwrap_or(profile_tokens).min(profile_tokens)
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
    fn a_window_reported_by_the_server_governs_the_local_route() {
        for model in ["llamacpp:qwen3-4b", "llama.cpp:qwen3-4b", "llama:dolphin"] {
            assert_eq!(
                effective_context_window(model, Some(8192)),
                Some(8192),
                "{model}"
            );
        }
        // With nothing reported, the local route is as clueless as before.
        assert_eq!(effective_context_window("llama:dolphin", None), None);
    }

    #[test]
    fn a_reported_window_beats_the_family_table_on_the_local_route() {
        // The table knows the llama-3.1 family as 128k; a server started with
        // `-c 8192` has 8192 and no more. Budgeting for the family's window
        // here is what silently drops context, so the measurement wins.
        assert_eq!(
            capabilities_for("llama:llama-3.1-8b").context_window,
            Some(128_000)
        );
        assert_eq!(
            effective_context_window("llama:llama-3.1-8b", Some(8192)),
            Some(8192)
        );
    }

    #[test]
    fn a_window_from_a_different_kind_of_server_is_ignored() {
        // That 8192 describes a llama.cpp process. Applying it to a hosted
        // route would budget for space the API does not run out of the same way.
        assert_eq!(
            effective_context_window("anthropic:claude-sonnet-4", Some(8192)),
            Some(200_000)
        );
        assert_eq!(
            effective_context_window("openrouter/deepseek/deepseek-chat", Some(8192)),
            Some(64_000)
        );
    }

    #[test]
    fn a_window_asked_of_ollama_governs_the_ollama_route() {
        // `qwen2.5:7b` is an Ollama tag the table has nothing to say about, and
        // for a model that goes there the number carried in is the window this
        // program asked Ollama to serve — the same number the request body
        // carries. Budgeting against anything else is how a turn ends up filling
        // a window the server was not told to open.
        assert_eq!(
            effective_context_window("qwen2.5:7b", Some(8192)),
            Some(8192)
        );
        // A model the table does know still defers to the window in use here:
        // `qwen3-coder` is 256k as a service, and 8192 as the request said.
        assert_eq!(
            effective_context_window("qwen3-coder:30b", Some(8192)),
            Some(8192)
        );
        // Nothing decided yet, nothing to prefer.
        assert_eq!(effective_context_window("qwen2.5:7b", None), None);
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
