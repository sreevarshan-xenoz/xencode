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
//!
//! Vision is the one capability the offline table cannot know and the serving
//! side can: a running `llama-server` reports `modalities.vision` in `/props`,
//! and Ollama lists a `vision` capability in `/api/show`. [`capabilities_with_report`]
//! folds whatever the server said into the table, treating silence as "no" so
//! an image is never sent to a model that has not positively claimed to see.
//! With no server report the table's conservative `false` stands.

use xencode_models_rs::llamacpp::ServerReport;

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
    /// `true` only when the serving side positively reported vision support.
    /// The offline table never claims it: a wrong "yes" sends a picture to a
    /// model that reads it as text, which is worse than an honest refusal.
    pub supports_vision: bool,
    /// The server's per-launch `media_marker`, read from `/props` when present.
    /// Random per launch, so only a value actually reported is kept here.
    pub media_marker: Option<String>,
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
        media_marker: None,
        supports_reasoning: is_reasoning_family(&lower),
        supports_streaming: true,
    }
}

/// Fold what the serving side reported into the table's answer for a model.
///
/// The offline table cannot know whether a given set of weights can see; only
/// the process serving them can. On the llama.cpp route the `/props` answer
/// carries `modalities.vision` and a per-launch `media_marker`; on the Ollama
/// route `/api/show` lists a `vision` capability. Both take precedence over the
/// table, and both are conservative: a field that is absent, or a server that
/// cannot be reached, leaves `supports_vision` at the table's `false`.
pub fn capabilities_with_report(
    model: &str,
    llamacpp: Option<&ServerReport>,
    ollama_vision: Option<bool>,
) -> ModelCapabilities {
    let mut caps = capabilities_for(model);
    if crate::routes_to_llamacpp(model) {
        if let Some(report) = llamacpp {
            caps.supports_vision = report.supports_vision.unwrap_or(false);
            caps.media_marker = report.media_marker.clone();
        }
    } else if crate::routes_to_ollama(model) {
        caps.supports_vision = ollama_vision.unwrap_or(false);
    }
    caps
}

/// Decide whether an image-bearing turn may go out, before any provider call.
///
/// Returns a named, actionable refusal when the turn carries images and the
/// selected model has not been confirmed to see them; `None` when the turn may
/// proceed (no images, or vision support was reported). This is a pure check —
/// it touches no network and constructs no provider, so a refusal here is
/// guaranteed to happen before a request could be sent.
pub fn image_vision_refusal(
    model: &str,
    vision_supported: bool,
    has_images: bool,
) -> Option<String> {
    (has_images && !vision_supported).then(|| {
        format!(
            "model {model} does not report vision support; use --model with a vision-capable model"
        )
    })
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
    fn bitnet_is_a_local_tool_capable_route_with_no_static_window() {
        assert_eq!(
            capabilities_for("bitnet:BitNet-b1.58-2B").context_window,
            None
        );
        let caps = capabilities_for("bitnet:m");
        assert!(caps.supports_tools);
        assert!(caps.supports_streaming);
        assert!(!caps.supports_vision);
        // A window measured on a different server is never borrowed for a
        // BitNet route — the weights and the process that serves them differ.
        assert_eq!(effective_context_window("bitnet:m", Some(8192)), None);
    }

    #[test]
    fn reports_from_other_servers_do_not_describe_a_bitnet_model() {
        let caps = capabilities_with_report(
            "bitnet:m",
            Some(&llamacpp_report(
                Some(true),
                Some("<__media_vISIONMARKERcaptured__>"),
            )),
            Some(true),
        );
        assert!(
            !caps.supports_vision,
            "a /props answer describes the llama.cpp process, not the BitNet one"
        );
        assert!(caps.media_marker.is_none());
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

    /// A `/props` answer with the vision fields as reported by b10809: vision
    /// false and a per-launch marker. Provenance: `NEXT_PLAN_TASKS.md` §Z-1,
    /// observed verbatim on 2026-10-05. Parsed by the models crate in its own
    /// tests; here it is the input to the capability mapping.
    fn llamacpp_report(vision: Option<bool>, marker: Option<&str>) -> ServerReport {
        ServerReport {
            context_tokens: Some(7168),
            slots: Some(4),
            supports_vision: vision,
            media_marker: marker.map(str::to_string),
        }
    }

    #[test]
    fn a_vision_report_overrides_the_tables_conservative_default() {
        let model = "llamacpp:smolvlm2";
        let caps = capabilities_with_report(
            model,
            Some(&llamacpp_report(
                Some(true),
                Some("<__media_vISIONMARKERcaptured__>"),
            )),
            None,
        );
        assert!(caps.supports_vision);
        assert_eq!(
            caps.media_marker.as_deref(),
            Some("<__media_vISIONMARKERcaptured__>")
        );
        // Still the table's answer for everything the server did not speak to.
        assert!(caps.supports_tools);
        assert!(caps.supports_streaming);

        // A reported "no" is a real answer and overrides the table too.
        let caps = capabilities_with_report(
            model,
            Some(&llamacpp_report(
                Some(false),
                Some("<__media_vDInAurkiv92eIIfmePeq5FKhw1ceqZi__>"),
            )),
            None,
        );
        assert!(!caps.supports_vision);
        assert_eq!(
            caps.media_marker.as_deref(),
            Some("<__media_vDInAurkiv92eIIfmePeq5FKhw1ceqZi__>")
        );
    }

    #[test]
    fn silence_from_the_serving_side_stays_a_conservative_no() {
        let caps = capabilities_with_report("llamacpp:qwen3-4b", None, None);
        assert!(!caps.supports_vision);
        assert_eq!(caps.media_marker, None);
        // A server that answered but did not report the fields is the same.
        let caps = capabilities_with_report(
            "llamacpp:qwen3-4b",
            Some(&llamacpp_report(None, None)),
            None,
        );
        assert!(!caps.supports_vision);

        // Ollama's `vision` capability is folded in the same way, and only on
        // an Ollama route. No live vision model was available here, so this
        // checks the boolean the models crate parses, not an image answer.
        assert!(capabilities_with_report("qwen2.5-vl:7b", None, Some(true)).supports_vision);
        assert!(!capabilities_with_report("qwen2.5-vl:7b", None, Some(false)).supports_vision);
        assert!(!capabilities_with_report("qwen2.5-vl:7b", None, None).supports_vision);
        // A vision report from a llama.cpp server is not applied to a cloud
        // route: it describes a different process.
        assert!(
            !capabilities_with_report(
                "anthropic:claude-sonnet-4",
                Some(&llamacpp_report(Some(true), None)),
                None
            )
            .supports_vision
        );
    }

    #[test]
    fn an_image_on_a_text_only_model_is_refused_by_name() {
        let model = "llamacpp:qwen3-4b";
        let refusal = image_vision_refusal(model, false, true)
            .expect("an image on a text-only model must be refused before sending");
        assert_eq!(
            refusal,
            format!("model {model} does not report vision support; use --model with a vision-capable model")
        );
        assert!(refusal.contains(model));
        assert!(refusal.contains("--model"));

        // No images: nothing to refuse, whatever the model claims. Confirmed
        // vision: the image may go out.
        assert_eq!(image_vision_refusal(model, false, false), None);
        assert_eq!(image_vision_refusal("llamacpp:smolvlm2", true, true), None);
    }
}
