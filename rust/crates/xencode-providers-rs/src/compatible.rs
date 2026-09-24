//! One generic OpenAI-compatible chat provider (Roo-style "OpenAI Compatible").
//!
//! Covers OpenRouter, Qwen cloud, LM Studio, vLLM, LiteLLM gateways, Groq,
//! Together, custom endpoints — anything speaking `/chat/completions` with
//! SSE `delta` chunks. New providers become config rows, not new files.
//!
//! Backends with genuinely different wire formats or local-only behavior
//! (Ollama native, llama.cpp, Anthropic, Gemini) keep their dedicated paths;
//! the llama.cpp tools wrapper reuses [`post_sse_stream`] for HTTP+SSE while
//! keeping its model-swap guard, sampling-opts merge, and timings.

use futures_util::StreamExt;

use crate::tools::{self, ToolCallAccumulator};
use crate::{AgentStep, ProviderError, ToolDefinition};

/// Outcome of one streamed `/chat/completions` call.
#[derive(Debug)]
pub(crate) struct StreamOutcome {
    pub step: AgentStep,
    pub completion_tokens: u64,
}

/// Generic client for any OpenAI-compatible `/chat/completions` endpoint.
#[derive(Debug, Clone)]
pub struct OpenAICompatibleProvider {
    base_url: String,
    api_key: Option<String>,
    extra_headers: Vec<(String, String)>,
    client: reqwest::Client,
}

impl OpenAICompatibleProvider {
    /// `base_url` is the API root (e.g. `https://openrouter.ai/api/v1`);
    /// `/chat/completions` is appended. `api_key` may be `None` for
    /// keyless local endpoints (LM Studio, vLLM defaults).
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key,
            extra_headers: Vec::new(),
            client: reqwest::Client::new(),
        }
    }

    /// Extra header sent on every request (e.g. OpenRouter's
    /// `HTTP-Referer` / `X-Title`).
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.extra_headers
            .push((name.to_string(), value.to_string()));
        self
    }

    /// Full chat-completions URL, tolerating a trailing slash on the base.
    pub fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    /// Request payload: rendered messages plus the `tools` array and
    /// `tool_choice: auto` only when tools are actually offered.
    pub fn build_payload(
        &self,
        model: &str,
        rendered_messages: &serde_json::Value,
        tools: &[ToolDefinition],
    ) -> serde_json::Value {
        let mut payload = serde_json::json!({
            "model": model,
            "messages": rendered_messages,
            "stream": true,
        });
        if !tools.is_empty() {
            payload["tools"] = tools.iter().map(ToolDefinition::to_api_value).collect();
            payload["tool_choice"] = serde_json::Value::String("auto".to_string());
        }
        payload
    }

    /// Stream one tool-aware chat step. `rendered_messages` must already
    /// include any agent-turn history (`tools::render_history`).
    pub async fn generate_stream_with_tools<F>(
        &self,
        model: &str,
        rendered_messages: &serde_json::Value,
        tools: &[ToolDefinition],
        label: &str,
        callback: F,
    ) -> Result<AgentStep, ProviderError>
    where
        F: FnMut(&str),
    {
        let payload = self.build_payload(model, rendered_messages, tools);
        Ok(post_sse_stream(
            &self.client,
            &self.completions_url(),
            self.api_key.as_deref(),
            &self.extra_headers,
            &payload,
            label,
            callback,
        )
        .await?
        .step)
    }
    /// One non-streaming completion: `stream: false`, and the first choice's
    /// message content. A reply carrying only tool calls has no content, which
    /// is an empty string here rather than an error.
    pub async fn generate(
        &self,
        model: &str,
        rendered_messages: &serde_json::Value,
        label: &str,
    ) -> Result<String, ProviderError> {
        let payload = serde_json::json!({
            "model": model,
            "messages": rendered_messages,
            "stream": false,
        });
        let mut request = self.client.post(self.completions_url()).json(&payload);
        if let Some(ref key) = self.api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        for (name, value) in &self.extra_headers {
            request = request.header(name.as_str(), value.as_str());
        }

        let response = request
            .send()
            .await
            .map_err(|e| ProviderError::Network(format!("{label} request failed: {e}")))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProviderError::api(label, status, body));
        }
        let value: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ProviderError::Parse(format!("{label} response: {e}")))?;
        Ok(first_choice_text(&value))
    }
}

/// `choices[0].message.content` as a string, or empty when absent or null.
fn first_choice_text(value: &serde_json::Value) -> String {
    value["choices"]
        .get(0)
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .unwrap_or_default()
        .to_string()
}

/// Shared HTTP+SSE core for every OpenAI-compatible backend: POST the
/// payload, forward text through the callback, accumulate `delta.tool_calls`,
/// and harvest the terminal `usage` chunk when the server sends one.
pub(crate) async fn post_sse_stream<F>(
    client: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
    extra_headers: &[(String, String)],
    payload: &serde_json::Value,
    label: &str,
    mut callback: F,
) -> Result<StreamOutcome, ProviderError>
where
    F: FnMut(&str),
{
    let mut request = client.post(url).json(payload);
    if let Some(key) = api_key {
        request = request.header("Authorization", format!("Bearer {key}"));
    }
    for (name, value) in extra_headers {
        request = request.header(name.as_str(), value.as_str());
    }

    let response = request
        .send()
        .await
        .map_err(|e| ProviderError::Network(format!("{label} stream request failed: {e}")))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(ProviderError::api(label, status, body));
    }

    let mut stream = response.bytes_stream();
    let mut text = String::new();
    let mut completion_tokens: u64 = 0;
    let mut acc = ToolCallAccumulator::default();
    let mut lines = crate::frames::FrameLines::default();

    // The stream is read as bytes, not as lines: a `data:` line can be spread
    // over two reads, and one read can hold several. See `frames`.
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
        lines.feed(&chunk, &mut |line| {
            ingest_line(
                line,
                &mut text,
                &mut completion_tokens,
                &mut acc,
                &mut callback,
            )
        });
    }
    lines.finish(&mut |line| {
        ingest_line(
            line,
            &mut text,
            &mut completion_tokens,
            &mut acc,
            &mut callback,
        )
    });

    Ok(StreamOutcome {
        step: AgentStep {
            text,
            tool_calls: acc.finish(),
        },
        completion_tokens,
    })
}

/// One complete line of an OpenAI-style SSE body.
pub(crate) fn ingest_line<F: FnMut(&str)>(
    line: &str,
    text: &mut String,
    completion_tokens: &mut u64,
    acc: &mut ToolCallAccumulator,
    callback: &mut F,
) {
    let line = line.trim();
    if line.is_empty() || line == "data: [DONE]" {
        return;
    }
    let Some(data) = line.strip_prefix("data: ") else {
        return;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
        return;
    };
    // The final chunk carries usage (with an empty choices array).
    if let Some(usage) = json.get("usage").and_then(|u| u.as_object()) {
        if let Some(tokens) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
            *completion_tokens = tokens;
        }
    }
    tools::ingest_oai_chunk(&json, text, acc, callback);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }]
    }

    #[test]
    fn completions_url_tolerates_trailing_slash() {
        let p = OpenAICompatibleProvider::new("https://x.example/v1/", None);
        assert_eq!(p.completions_url(), "https://x.example/v1/chat/completions");
        let p = OpenAICompatibleProvider::new("https://x.example/v1", None);
        assert_eq!(p.completions_url(), "https://x.example/v1/chat/completions");
    }

    #[test]
    fn payload_omits_tools_key_when_no_tools_offered() {
        let p = OpenAICompatibleProvider::new("https://x.example/v1", None);
        let payload = p.build_payload("m", &serde_json::json!([]), &[]);
        assert_eq!(payload["model"], "m");
        assert!(payload.get("tools").is_none());
        assert!(payload.get("tool_choice").is_none());
    }

    #[test]
    fn payload_includes_tools_and_auto_choice() {
        let p = OpenAICompatibleProvider::new("https://x.example/v1", None);
        let payload = p.build_payload("m", &serde_json::json!([]), &sample_tools());
        assert_eq!(payload["tools"][0]["function"]["name"], "read_file");
        assert_eq!(payload["tool_choice"], "auto");
    }

    #[test]
    fn first_choice_text_reads_content_and_tolerates_a_tool_only_reply() {
        assert_eq!(
            first_choice_text(&serde_json::json!({"choices":[{"message":{"content":"hi"}}]})),
            "hi"
        );
        // A reply that only asked for a tool call has no content to show.
        assert_eq!(
            first_choice_text(&serde_json::json!({"choices":[{"message":{"content":null}}]})),
            ""
        );
        assert_eq!(first_choice_text(&serde_json::json!({"choices": []})), "");
        assert_eq!(first_choice_text(&serde_json::json!({})), "");
    }
}
