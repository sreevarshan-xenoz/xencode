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
    pub usage: UsageCounts,
}

/// What a server said the request cost, from the `usage` object the last chunk of
/// a stream carries. All three are the server's own counts: `prompt_tokens` is
/// the whole prompt including the chat template, and `cached_tokens` is how much
/// of it was already in the server's memory — measured on `llama-server` b10809,
/// where a first-seen prompt of 3,202 tokens reported 1 cached and the same prompt
/// sent again reported 3,201.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct UsageCounts {
    pub completion_tokens: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
}

/// Generic client for any OpenAI-compatible `/chat/completions` endpoint.
#[derive(Debug, Clone)]
pub struct OpenAICompatibleProvider {
    base_url: String,
    api_key: Option<String>,
    extra_headers: Vec<(String, String)>,
    client: reqwest::Client,
    /// Set when this endpoint's traffic is being captured; see
    /// [`crate::traffic`].
    recorder: Option<crate::traffic::TrafficRecorder>,
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
            recorder: None,
        }
    }

    /// Capture every request this endpoint makes, so the session can be served
    /// back later. See [`crate::traffic::TrafficRecorder`].
    pub fn with_recorder(mut self, recorder: Option<crate::traffic::TrafficRecorder>) -> Self {
        self.recorder = recorder;
        self
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
            &SseRequest {
                url: &self.completions_url(),
                api_key: self.api_key.as_deref(),
                extra_headers: &self.extra_headers,
                payload: &payload,
                label,
                recorder: self.recorder.as_ref(),
            },
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

/// One streaming request: where it goes, how it authenticates, what it asks for
/// and what to do with the bytes. Grouped because a call that streams needs all
/// of it, and passing eight arguments in order is how a header ends up where the
/// payload belongs.
pub(crate) struct SseRequest<'a> {
    pub url: &'a str,
    pub api_key: Option<&'a str>,
    pub extra_headers: &'a [(String, String)],
    pub payload: &'a serde_json::Value,
    pub label: &'a str,
    /// Keep the bytes as they arrived for a recording of this run; `None` asks
    /// for nothing to be kept.
    pub recorder: Option<&'a crate::traffic::TrafficRecorder>,
}

/// Shared HTTP+SSE core for every OpenAI-compatible backend: POST the
/// payload, forward text through the callback, accumulate `delta.tool_calls`,
/// and harvest the terminal `usage` chunk when the server sends one.
pub(crate) async fn post_sse_stream<F>(
    client: &reqwest::Client,
    request: &SseRequest<'_>,
    mut callback: F,
) -> Result<StreamOutcome, ProviderError>
where
    F: FnMut(&str),
{
    let SseRequest {
        url,
        api_key,
        extra_headers,
        payload,
        label,
        recorder,
    } = *request;
    let mut http = client.post(url).json(payload);
    if let Some(key) = api_key {
        http = http.header("Authorization", format!("Bearer {key}"));
    }
    for (name, value) in extra_headers {
        http = http.header(name.as_str(), value.as_str());
    }

    let started = std::time::Instant::now();
    let request_body = payload.to_string();
    let response = http
        .send()
        .await
        .map_err(|e| ProviderError::Network(format!("{label} stream request failed: {e}")))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(ProviderError::api(label, status, body));
    }

    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let mut stream = response.bytes_stream();
    let mut text = String::new();
    let mut usage = UsageCounts::default();
    let mut acc = ToolCallAccumulator::default();
    let mut ended = false;
    let mut lines = crate::frames::FrameLines::default();
    // Kept only for a recording: the bytes as they arrived, before the frame
    // reader had anything to say about where a line begins.
    let mut raw: Vec<u8> = Vec::new();

    // The stream is read as bytes, not as lines: a `data:` line can be spread
    // over two reads, and one read can hold several. See `frames`.
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| ProviderError::Network(e.to_string()))?;
        if recorder.is_some() {
            raw.extend_from_slice(&chunk);
        }
        lines.feed(&chunk, &mut |line| {
            ended |= ingest_line(line, &mut text, &mut usage, &mut acc, &mut callback)
        });
    }
    lines.finish(&mut |line| {
        ended |= ingest_line(line, &mut text, &mut usage, &mut acc, &mut callback)
    });
    if !ended {
        return Err(cut_off(label));
    }

    // A body that is not text is not recorded at all: half of it written down
    // would come back later as an answer the model never gave.
    if let Some(recorder) = recorder {
        if let Ok(response_body) = String::from_utf8(raw) {
            recorder.record(crate::traffic::TrafficPair {
                method: "POST".to_string(),
                path: crate::traffic::url_path(url),
                request_body,
                status,
                content_type,
                response_body,
                ts_unix_ms: crate::traffic::now_unix_ms(),
                duration_ms: started.elapsed().as_millis() as u64,
            });
        }
    }

    Ok(StreamOutcome {
        step: AgentStep {
            text,
            tool_calls: acc.finish(),
        },
        usage,
    })
}

/// The error for an answer whose stream closed before the server said it was
/// done. Without it, a server killed mid-answer handed back half an answer as
/// if it were the whole one.
pub(crate) fn cut_off(label: &str) -> ProviderError {
    ProviderError::Network(format!(
        "{label} closed the connection before finishing its answer"
    ))
}

/// One complete line of an OpenAI-style SSE body. Returns true when the line
/// says the answer is over: the `data: [DONE]` marker, or a choice carrying a
/// `finish_reason`. A stream that closes without either was cut off (QA-6).
pub(crate) fn ingest_line<F: FnMut(&str)>(
    line: &str,
    text: &mut String,
    usage: &mut UsageCounts,
    acc: &mut ToolCallAccumulator,
    callback: &mut F,
) -> bool {
    let line = line.trim();
    if line == "data: [DONE]" {
        return true;
    }
    if line.is_empty() {
        return false;
    }
    let Some(data) = line.strip_prefix("data: ") else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else {
        return false;
    };
    let finished = json
        .get("choices")
        .and_then(|c| c.as_array())
        .is_some_and(|choices| {
            choices
                .iter()
                .any(|c| c.get("finish_reason").is_some_and(|r| !r.is_null()))
        });
    // The final chunk carries usage (with an empty choices array). A server only
    // sends that chunk when asked — see `ask_for_usage`.
    if let Some(counts) = json.get("usage").and_then(|u| u.as_object()) {
        if let Some(tokens) = counts.get("completion_tokens").and_then(|v| v.as_u64()) {
            usage.completion_tokens = tokens;
        }
        if let Some(tokens) = counts.get("prompt_tokens").and_then(|v| v.as_u64()) {
            usage.prompt_tokens = tokens;
        }
        if let Some(tokens) = counts
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
        {
            usage.cached_tokens = tokens;
        }
    }
    tools::ingest_oai_chunk(&json, text, acc, callback);
    finished
}

/// Ask an OpenAI-style server to say what the request cost.
///
/// Without this the streaming answer carries no `usage` object at all, which was
/// measured on `llama-server` b10809: the same request sent as a stream returned
/// six chunks and no token counts, and with this key added it returned a seventh
/// carrying `prompt_tokens: 2222, cached_tokens: 2221`. A stream that ends without
/// usage is then the same case as before — nothing is invented for it.
///
/// Only the llama.cpp requests ask. How a hosted server reacts to an unknown key
/// in a stream request is unverified here, and an answer that stops arriving is
/// worse than a token count we do not have.
pub(crate) fn ask_for_usage(payload: &mut serde_json::Value) {
    payload["stream_options"] = serde_json::json!({ "include_usage": true });
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

    /// The last chunk of a streamed answer carries the counts, with no text in
    /// it. A chunk with neither is a heartbeat, not an answer.
    #[test]
    fn the_final_stream_chunk_carries_the_servers_counts() {
        let mut text = String::new();
        let mut usage = UsageCounts::default();
        let mut acc = ToolCallAccumulator::default();
        ingest_line(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
            &mut text,
            &mut usage,
            &mut acc,
            &mut |_| {},
        );
        ingest_line(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":2222,\
             \"completion_tokens\":7,\
             \"prompt_tokens_details\":{\"cached_tokens\":2221}}}",
            &mut text,
            &mut usage,
            &mut acc,
            &mut |_| {},
        );
        assert_eq!(text, "hi");
        assert_eq!(usage.prompt_tokens, 2222);
        assert_eq!(usage.cached_tokens, 2221);
        assert_eq!(usage.completion_tokens, 7);
    }

    /// A server that answers without a `usage` object at all — which is what a
    /// stream looks like when it was never asked for one — leaves the counts at
    /// zero rather than inventing them.
    #[test]
    fn a_stream_that_reports_nothing_costs_nothing_as_far_as_we_know() {
        let mut text = String::new();
        let mut usage = UsageCounts::default();
        let mut acc = ToolCallAccumulator::default();
        ingest_line(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
            &mut text,
            &mut usage,
            &mut acc,
            &mut |_| {},
        );
        assert_eq!(usage, UsageCounts::default());
    }

    /// Sending the key is not optional: measured on `llama-server` b10809, the
    /// same request as a stream carried no usage at all until this was added.
    #[test]
    fn a_streamed_llama_request_asks_for_its_own_cost() {
        let mut payload = serde_json::json!({"model": "m", "stream": true});
        ask_for_usage(&mut payload);
        assert_eq!(payload["stream_options"]["include_usage"], true);
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
