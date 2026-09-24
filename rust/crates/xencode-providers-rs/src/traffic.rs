//! Capturing what a provider was asked and what it actually answered.
//!
//! [`crate::playback`] can only replay a conversation that was written down
//! first, and a down-again paraphrase of a model's answer is worth nothing: the
//! faults it is supposed to pin live in the bytes between the server and the
//! reader. So a [`TrafficRecorder`] sits at the HTTP boundary of the shared
//! OpenAI-compatible request path and keeps the payload that was serialised and
//! the response body that arrived, as received, before either is parsed. That is
//! the same shape a hand-made cassette holds, so [`TrafficRecorder::cassette`]
//! converts one captured session into a [`Cassette`] that
//! [`crate::playback::Playback`] can serve again on a real socket.
//!
//! Nothing is recorded unless a caller attaches a recorder to the manager. A
//! captured request holds the whole prompt, so recordings belong in a directory
//! the project keeps out of version control, not in a source tree.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::playback::{
    Capture, Cassette, Interaction, RecordedResponse, RequestMatcher, CASSETTE_FORMAT,
};

/// One request that left this process and the answer that came back, both as
/// the wire had them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficPair {
    /// Always `POST` for the paths that are captured today.
    pub method: String,
    /// The path the request was made against, without the host.
    pub path: String,
    /// The request body exactly as it was serialised.
    pub request_body: String,
    pub status: u16,
    pub content_type: String,
    /// The response body exactly as it was received, `data:` lines and all.
    pub response_body: String,
    /// When the request was sent, in milliseconds since the Unix epoch.
    pub ts_unix_ms: u64,
    /// How long the whole exchange took. A replay reads this out of the
    /// recording instead of timing itself, which is what lets two replays of
    /// one session be compared byte for byte at all.
    pub duration_ms: u64,
}

impl TrafficPair {
    /// What a cassette has to match on so that it answers the question it
    /// recorded and not the next one that happens to arrive: the newest
    /// message of the request, taken from the serialised body rather than from
    /// the parsed value, so the text compared is the text that was written.
    pub fn needle(&self) -> Option<String> {
        request_needle(&self.request_body)
    }
}

/// The text a cassette has to see in a request before answering it, taken from
/// a serialised request body. Public because a replay rebuilt from a session
/// recording (see `xencode_context_rs::session`) needs the same matcher a
/// capture would have written.
pub fn request_needle(request_body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(request_body).ok()?;
    let content = value
        .get("messages")?
        .as_array()?
        .last()?
        .get("content")?
        .as_str()?;
    let escaped = serde_json::to_string(content).ok()?;
    let inner = escaped.trim_matches('"');
    // The tail, not the head: the newest message is the one a replay can most
    // easily get wrong, because it is usually what a tool just printed.
    let mut start = inner.len().saturating_sub(NEEDLE_MAX_BYTES);
    while start < inner.len() && !inner.is_char_boundary(start) {
        start += 1;
    }
    let needle = &inner[start..];
    (!needle.is_empty()).then(|| needle.to_string())
}

/// How much of the newest message a cassette matches on.
const NEEDLE_MAX_BYTES: usize = 200;

/// A list of exchanges, filled in as they happen and drained by whoever asked
/// for the recording. Cloning it hands out another handle to the same list, so
/// a recorder can outlive the request that wrote to it.
#[derive(Clone, Default, Debug)]
pub struct TrafficRecorder {
    pairs: Arc<Mutex<Vec<TrafficPair>>>,
    /// Monotonic count of everything ever recorded, so a caller that drains the
    /// list between two rounds can still number the lines it writes.
    total: Arc<AtomicU64>,
}

impl TrafficRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Called by the request path for one completed exchange. Ignores the
    /// recording if the body cannot be read as text, because a recording with
    /// half a response in it would be replayed as a shorter answer and read as
    /// the model having stopped early.
    pub(crate) fn record(&self, pair: TrafficPair) {
        let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
        pairs.push(pair);
        self.total.fetch_add(1, Ordering::Relaxed);
    }

    /// Take everything recorded so far and leave the list empty.
    pub fn take(&self) -> Vec<TrafficPair> {
        let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *pairs)
    }

    /// How many exchanges are waiting to be taken.
    pub fn pending(&self) -> usize {
        self.pairs.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Everything ever recorded by this handle, for a caller that wants to
    /// number its lines without holding them.
    pub fn recorded_so_far(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Turn the captured exchanges into a cassette this crate can serve again.
    ///
    /// The provenance block is the caller's to supply, because only the caller
    /// knows what the bytes were produced by — and a recording that does not
    /// say where it came from cannot be trusted later.
    pub fn cassette(&self, captured: Capture) -> io::Result<Cassette> {
        let mut interactions = Vec::new();
        for pair in self.pairs.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            interactions.push(Interaction {
                request: RequestMatcher {
                    method: pair.method.clone(),
                    path: pair.path.clone(),
                    body_contains: pair.needle().into_iter().collect(),
                    body: Some(pair.request_body.clone()),
                },
                response: RecordedResponse {
                    status: pair.status,
                    content_type: pair.content_type.clone(),
                    body: pair.response_body.clone(),
                },
            });
        }
        let cassette = Cassette {
            format: CASSETTE_FORMAT.to_string(),
            captured,
            interactions,
        };
        cassette.validate()?;
        Ok(cassette)
    }
}

/// The path of a URL, with the scheme, host, port and query left off — which is
/// what a cassette matches a replayed request against.
pub fn url_path(url: &str) -> String {
    let after_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let path = after_scheme
        .split_once('/')
        .map(|(_, rest)| rest)
        .unwrap_or("");
    format!("/{}", path.split(['?', '#']).next().unwrap_or_default())
}

/// The clock a capture stamps with, in milliseconds since the Unix epoch.
pub(crate) fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(content: &str) -> TrafficPair {
        TrafficPair {
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            request_body: serde_json::json!({
                "model": "m",
                "messages": [
                    {"role": "system", "content": "instructions"},
                    {"role": "user", "content": content}
                ]
            })
            .to_string(),
            status: 200,
            content_type: "text/event-stream".to_string(),
            response_body: "data: [DONE]\n".to_string(),
            ts_unix_ms: 1,
            duration_ms: 2,
        }
    }

    #[test]
    fn the_matcher_is_taken_from_the_body_that_was_written_not_from_the_value() {
        // A newline inside a tool's output is two characters in the body and
        // one in the parsed string; matching on the parsed one would never hit.
        let captured = pair("first line\nsecond line");
        let needle = captured.needle().expect("a matcher");
        assert!(captured.request_body.contains(&needle), "{needle}");
        assert!(needle.contains("second line"));
    }

    #[test]
    fn a_request_with_no_message_content_has_no_matcher() {
        let mut captured = pair("hi");
        captured.request_body = serde_json::json!({"model": "m"}).to_string();
        assert_eq!(captured.needle(), None);
    }

    #[tokio::test]
    async fn a_capture_turns_into_a_cassette_that_serves_the_same_bytes() {
        let recorder = TrafficRecorder::new();
        recorder.record(pair("what is the capital of France"));
        recorder.record(pair("and its population"));
        let cassette = recorder
            .cassette(Capture {
                at_unix_ms: 1_700_000_000_000,
                server: "test".to_string(),
                model: "m".to_string(),
                tool: "unit test".to_string(),
                note: "written by a test, not captured from a server".to_string(),
            })
            .expect("cassette");
        assert_eq!(cassette.interactions.len(), 2);
        assert_eq!(cassette.interactions[0].response.body, "data: [DONE]\n");
        assert_eq!(cassette.interactions[1].request.body_contains.len(), 1);
        let taken = recorder.take();
        assert_eq!(taken.len(), 2);
        assert_eq!(taken[0].path, "/v1/chat/completions");
        assert_eq!(recorder.pending(), 0, "taking leaves nothing behind");
        assert_eq!(
            recorder.recorded_so_far(),
            2,
            "a drain does not unrecord what happened"
        );
    }

    #[test]
    fn a_capture_keeps_the_path_a_request_was_made_against() {
        // The cassette that replays a capture matches on path, so this has to
        // agree with what a client asking the same URL sends.
        assert_eq!(
            url_path("http://127.0.0.1:9/v1/chat/completions"),
            "/v1/chat/completions"
        );
        assert_eq!(
            url_path("https://x.example/chat/completions"),
            "/chat/completions"
        );
        assert_eq!(
            url_path("http://h/v1/chat/completions?i=1"),
            "/v1/chat/completions"
        );
        assert_eq!(url_path("http://h/api/chat"), "/api/chat");
    }

    #[test]
    fn a_second_handle_writes_to_the_same_capture() {
        // The recorder is handed to a request path that outlives the caller's
        // borrow, so a clone has to be the same list rather than a copy.
        let recorder = TrafficRecorder::new();
        let other = recorder.clone();
        recorder.record(pair("one"));
        assert_eq!(other.pending(), 1);
        assert_eq!(other.take().len(), 1);
        assert_eq!(recorder.pending(), 0);
    }
}
