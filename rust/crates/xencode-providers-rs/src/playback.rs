//! Replaying recorded provider traffic over a real socket.
//!
//! An in-process fake never exercises the parts of a client that go wrong:
//! where one TCP read ends and the next begins, whether a JSON object arrives
//! in a single piece, whether a multi-byte character is cut in half. Those
//! faults only show up below the HTTP boundary, which is what this module is
//! for. [`Playback`] listens on a real loopback port and answers with the body
//! that a real server once sent, so the caller's HTTP client, its chunked
//! transfer decoding, its byte-stream reading and its tool execution all run
//! for real. Nothing in between is stubbed.
//!
//! A [`Cassette`] is a committed recording. The `captured` block states where
//! the bytes came from, because a recording is only worth trusting if you know
//! it is one: every cassette in `tests/fixtures` was written by `curl` against
//! a `llama-server` process running on this machine, from a local model file,
//! with no network involved. They are captures of traffic, not hand-written
//! expectations — the difference matters when a recording disagrees with what
//! you assumed the server would say.

use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};

/// The only format version this reader understands. A cassette that names a
/// different one is refused instead of being guessed at.
pub const CASSETTE_FORMAT: &str = "xencode-cassette/1";

/// Where a recording came from. Kept next to the bytes it describes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capture {
    /// When the traffic was recorded, in milliseconds since the Unix epoch.
    pub at_unix_ms: u64,
    /// The server that produced it, as specifically as it will say.
    pub server: String,
    /// The model it was asked to run.
    pub model: String,
    /// How the bytes were taken.
    pub tool: String,
    /// Anything a later reader needs to judge the recording by — the machine
    /// it ran on, whether the model was reasoning, what licence it carries.
    #[serde(default)]
    pub note: String,
}

/// What a request has to look like for this interaction to answer it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMatcher {
    pub method: String,
    /// Compared against the path the client asked for, exactly.
    pub path: String,
    /// Each of these has to appear somewhere in the request body. Used to keep
    /// a cassette from answering a question that was never asked.
    #[serde(default)]
    pub body_contains: Vec<String>,
    /// The request body as it was recorded, when the capture kept it. A test
    /// can resend these bytes and expect [`RecordedResponse::body`] back,
    /// which makes a recording a question-and-answer pair rather than a
    /// suggestion about what the server might have said.
    #[serde(default)]
    pub body: Option<String>,
}

/// The recorded answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedResponse {
    pub status: u16,
    pub content_type: String,
    /// The body exactly as it was received, `data:` lines and all.
    pub body: String,
}

/// One recorded request and the answer that went with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    pub request: RequestMatcher,
    pub response: RecordedResponse,
}

/// A recorded session: where it came from, and what was asked and answered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cassette {
    pub format: String,
    pub captured: Capture,
    #[serde(default)]
    pub interactions: Vec<Interaction>,
}

impl Cassette {
    /// Read a cassette from disk and check it is a format this code can serve.
    pub fn load(path: &std::path::Path) -> io::Result<Cassette> {
        let text = std::fs::read_to_string(path)?;
        let cassette: Cassette = serde_json::from_str(&text).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} is not a readable cassette: {e}", path.display()),
            )
        })?;
        cassette.validate()?;
        Ok(cassette)
    }

    /// Refuse a cassette this code cannot serve honestly: a format it does not
    /// know, no interactions, or an interaction with nothing to answer.
    pub fn validate(&self) -> io::Result<()> {
        if self.format != CASSETTE_FORMAT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "cassette says format {:?}, this build reads {:?}",
                    self.format, CASSETTE_FORMAT
                ),
            ));
        }
        if self.interactions.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cassette records no interactions",
            ));
        }
        for interaction in &self.interactions {
            if interaction.response.body.is_empty() && interaction.response.status < 400 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "the recorded answer for {} {} has no body: a recorded empty body \
                         would be replayed as a successful request that said nothing",
                        interaction.request.method, interaction.request.path
                    ),
                ));
            }
            if let Some(recorded) = &interaction.request.body {
                for needle in &interaction.request.body_contains {
                    if !recorded.contains(needle) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "the matcher for {} {} asks for {:?}, which is not in the \
                                 request that was actually recorded: resending that request \
                                 would be refused by this very cassette",
                                interaction.request.method, interaction.request.path, needle
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Start serving this cassette on a loopback port.
    pub fn play(&self) -> io::Result<Playback> {
        Playback::start(self, Delivery::default())
    }

    /// Start serving with a chosen shape for the writes, so a caller can ask
    /// for the recording to arrive in pieces the way a busy network sends it.
    pub fn play_with(&self, delivery: Delivery) -> io::Result<Playback> {
        Playback::start(self, delivery)
    }
}

/// How the recorded bytes are handed to the client once they leave the
/// cassette. The recording itself never changes — only where the writes stop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Delivery {
    /// One write per recorded line. The client still has to reassemble the
    /// stream, but every `data:` object arrives whole.
    #[default]
    LinePerLine,
    /// Each recorded line is cut in two near its middle and the halves are
    /// written separately, so a JSON object arrives in pieces. If the cut
    /// lands inside a multi-byte character, the character is cut too — which
    /// is what a real stream does to text like `サーバー`.
    SplitMidLine,
}

/// A running cassette server. The listener closes when this is dropped.
#[derive(Debug)]
pub struct Playback {
    addr: SocketAddr,
    /// Requests the cassette could not answer, for the caller to report.
    misses: std::sync::Arc<AtomicUsize>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Playback {
    fn start(cassette: &Cassette, delivery: Delivery) -> io::Result<Playback> {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let next = std::sync::Arc::new(AtomicUsize::new(0));
        let interactions = cassette.interactions.clone();
        let (tx, mut rx) = tokio::sync::oneshot::channel::<()>();
        let misses = std::sync::Arc::new(AtomicUsize::new(0));
        let misses_task = misses.clone();

        tokio::spawn(async move {
            let listener = match tokio::net::TcpListener::from_std(listener) {
                Ok(listener) => listener,
                Err(_) => return,
            };
            loop {
                tokio::select! {
                    _ = &mut rx => break,
                    accepted = listener.accept() => {
                        if let Ok((socket, _peer)) = accepted {
                            let _ = socket.set_nodelay(true);
                            let interactions = interactions.clone();
                            let next = next.clone();
                            let misses = misses_task.clone();
                            tokio::spawn(async move {
                                let _ = serve_one(socket, &interactions, &next, &misses, delivery).await;
                            });
                        }
                    }
                }
            }
        });

        Ok(Playback {
            addr,
            misses,
            shutdown: Some(tx),
        })
    }

    /// The base URL to point a provider at, without a trailing slash.
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// How many requests the cassette refused to answer. A non-zero count here
    /// means a test replayed something that was never recorded.
    pub fn misses(&self) -> usize {
        self.misses.load(Ordering::Relaxed)
    }
}

impl Drop for Playback {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

/// Send a request to a running [`Playback`] and return what the production
/// streaming reader makes of the recorded answer.
///
/// This is the whole point of the module: the request goes out over a real
/// socket, the response comes back as real HTTP with real chunked transfer
/// encoding, and every byte is parsed by the same code that talks to a real
/// server. Only the far end is a recording.
pub async fn replay<F>(
    base_url: &str,
    path: &str,
    request: &serde_json::Value,
    callback: F,
) -> Result<crate::AgentStep, crate::ProviderError>
where
    F: FnMut(&str),
{
    let outcome = crate::compatible::post_sse_stream(
        &reqwest::Client::new(),
        &crate::compatible::SseRequest {
            url: &format!("{base_url}{path}"),
            api_key: None,
            extra_headers: &[],
            payload: request,
            label: "cassette replay",
            // A replay is the far end of a recording, not the making of one.
            recorder: None,
        },
        callback,
    )
    .await?;
    Ok(outcome.step)
}

/// The request a cassette recorded, ready to send again. `None` when the
/// capture kept only the answer.
pub fn recorded_request(interaction: &Interaction) -> Option<serde_json::Value> {
    interaction
        .request
        .body
        .as_deref()
        .and_then(|body| serde_json::from_str(body).ok())
}

/// Read one request, answer it from the recording, and hang up.
async fn serve_one(
    mut socket: tokio::net::TcpStream,
    interactions: &[Interaction],
    cursor: &AtomicUsize,
    misses: &AtomicUsize,
    delivery: Delivery,
) -> io::Result<()> {
    use tokio::io::AsyncWriteExt;

    let (method, path, body) = read_request(&mut socket).await?;

    let chosen = pick(interactions, cursor, &method, &path, &body);
    let response = match chosen {
        Some(interaction) => &interaction.response,
        None => {
            misses.fetch_add(1, Ordering::Relaxed);
            socket
                .write_all(
                    b"HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\
                      Connection: close\r\n\r\nnothing recorded for this request",
                )
                .await?;
            socket.flush().await?;
            return Ok(());
        }
    };

    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\n\
         Connection: close\r\n\r\n",
        response.status,
        reason(response.status),
        response.content_type
    );
    socket.write_all(head.as_bytes()).await?;
    // The body is written in pieces rather than as one buffer, so the client
    // has to reassemble it: where a `data:` object ends and the next begins is
    // decided by the reader, not by the recording.
    for piece in pieces(&response.body, delivery) {
        socket
            .write_all(format!("{:x}\r\n", piece.len()).as_bytes())
            .await?;
        socket.write_all(piece).await?;
        socket.write_all(b"\r\n").await?;
        socket.flush().await?;
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    socket.write_all(b"0\r\n\r\n").await?;
    socket.flush().await?;
    Ok(())
}

/// The recorded body, cut up the way [`Delivery`] says. Every piece is
/// verbatim recording: cutting only chooses where a write stops, and a piece
/// is allowed to stop in the middle of a character because that is what a real
/// stream does.
fn pieces(body: &str, delivery: Delivery) -> Vec<&[u8]> {
    let mut out = Vec::new();
    for line in body.split_inclusive('\n') {
        if line.is_empty() {
            continue;
        }
        let bytes = line.as_bytes();
        match delivery {
            Delivery::LinePerLine => out.push(bytes),
            Delivery::SplitMidLine => {
                let cut = mid_cut(line);
                if cut == 0 || cut == bytes.len() {
                    out.push(bytes);
                } else {
                    out.push(&bytes[..cut]);
                    out.push(&bytes[cut..]);
                }
            }
        }
    }
    out
}

/// A byte offset near the middle of `line`, preferring one that falls inside a
/// multi-byte character so the cut splits a character and not just a line.
fn mid_cut(line: &str) -> usize {
    let bytes = line.as_bytes();
    let middle = line.len() / 2;
    if middle == 0 {
        return 0;
    }
    for radius in 0..middle.min(64) {
        for candidate in [middle + radius, middle - radius] {
            // 0b10xxxxxx marks a continuation byte: a character still going.
            if candidate < bytes.len() && bytes[candidate] & 0xC0 == 0x80 {
                return candidate;
            }
        }
    }
    middle
}

/// Take the next interaction that matches, in recorded order. Matching is by
/// method and path; `body_contains` is checked so a cassette cannot quietly
/// answer a different question than the one it recorded.
fn pick<'a>(
    interactions: &'a [Interaction],
    cursor: &AtomicUsize,
    method: &str,
    path: &str,
    body: &str,
) -> Option<&'a Interaction> {
    let start = cursor.load(Ordering::Relaxed);
    for (index, interaction) in interactions.iter().enumerate().skip(start) {
        let asked = interaction.request.method.eq_ignore_ascii_case(method)
            && interaction.request.path == path
            && interaction
                .request
                .body_contains
                .iter()
                .all(|needle| body.contains(needle));
        if asked {
            cursor.store(index + 1, Ordering::Relaxed);
            return Some(interaction);
        }
    }
    None
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> io::Result<(String, String, String)> {
    use tokio::io::AsyncReadExt;

    let mut raw = Vec::new();
    let mut byte = [0u8; 1];
    let mut header_end = None;
    let mut content_length = 0usize;
    loop {
        let read = socket.read(&mut byte).await?;
        if read == 0 {
            break;
        }
        raw.push(byte[0]);
        if header_end.is_none() {
            if let Some(position) = find(&raw, b"\r\n\r\n") {
                header_end = Some(position + 4);
                if let Some(length) = content_length_of(&raw[..position]) {
                    content_length = length;
                }
            }
        }
        if let Some(end) = header_end {
            if raw.len() >= end + content_length {
                break;
            }
        }
    }

    let header_end = header_end.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "request had no header terminator",
        )
    })?;
    let head = String::from_utf8_lossy(&raw[..header_end]).into_owned();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let full_path = parts.next().unwrap_or_default().to_string();
    let path = full_path
        .split(['?', '#'])
        .next()
        .unwrap_or_default()
        .to_string();
    let body = String::from_utf8_lossy(&raw[header_end..header_end + content_length])
        .trim_end_matches('\u{0}')
        .to_string();
    Ok((method, path, body))
}

fn content_length_of(head: &[u8]) -> Option<usize> {
    let text = String::from_utf8_lossy(head).into_owned();
    for line in text.lines().skip(1) {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("content-length") {
            return value.trim().parse().ok();
        }
    }
    None
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|part| part == needle)
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        429 => "Too Many Requests",
        _ => "Status",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cassette(body: &str) -> Cassette {
        Cassette {
            format: CASSETTE_FORMAT.to_string(),
            captured: Capture {
                at_unix_ms: 0,
                server: "test".to_string(),
                model: "test-model".to_string(),
                tool: "unit test".to_string(),
                note: String::new(),
            },
            interactions: vec![Interaction {
                request: RequestMatcher {
                    method: "POST".to_string(),
                    path: "/v1/chat/completions".to_string(),
                    body_contains: vec!["hello".to_string()],
                    body: None,
                },
                response: RecordedResponse {
                    status: 200,
                    content_type: "text/event-stream".to_string(),
                    body: body.to_string(),
                },
            }],
        }
    }

    async fn ask(base: &str, body: &str) -> reqwest::Result<String> {
        reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .body(body.to_string())
            .send()
            .await?
            .text()
            .await
    }

    #[test]
    fn a_cassette_from_another_format_is_refused_rather_than_guessed_at() {
        let mut recorded = cassette("data: [DONE]\n");
        recorded.format = "xencode-cassette/2".to_string();
        let error = recorded.validate().unwrap_err();
        assert!(error.to_string().contains("xencode-cassette/2"), "{error}");
    }

    #[tokio::test]
    async fn a_successful_recording_with_an_empty_body_is_refused() {
        // Replaying it would look like a model that answered nothing, which is
        // how a broken fixture gets mistaken for a broken product.
        assert!(cassette("").validate().is_err());
    }

    #[tokio::test]
    async fn the_recording_answers_on_a_real_socket_in_recorded_order() {
        let recorded =
            cassette("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\ndata: [DONE]\n");
        let playback = recorded.play().expect("listener");
        let text = ask(&playback.base_url(), r#"{"prompt":"hello"}"#)
            .await
            .expect("request reached the recorder");
        assert_eq!(
            text,
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\ndata: [DONE]\n"
        );
        assert_eq!(playback.misses(), 0);
    }

    #[tokio::test]
    async fn a_question_that_was_never_recorded_is_not_answered_with_something_else() {
        let recorded = cassette("data: [DONE]\n");
        let playback = recorded.play().expect("listener");
        let status = reqwest::Client::new()
            .post(format!("{}/v1/chat/completions", playback.base_url()))
            .body(r#"{"prompt":"something nobody asked"}"#)
            .send()
            .await
            .expect("request reached the recorder")
            .status();
        assert_eq!(status, reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(playback.misses(), 1);
    }

    #[test]
    fn cutting_for_delivery_moves_no_bytes_and_can_split_a_character() {
        let body = "data: {\"delta\":{\"content\":\"サーバーが起動しました\"}}\n\ndata: [DONE]\n";
        let whole = body.as_bytes();
        let line_by_line = pieces(body, Delivery::LinePerLine);
        let split = pieces(body, Delivery::SplitMidLine);
        assert_eq!(line_by_line.concat(), whole);
        assert_eq!(
            split.concat(),
            whole,
            "delivery may choose where a write stops, but it must not add, \
             drop or reorder a byte"
        );
        assert!(split.len() > line_by_line.len());
        // One of those pieces is not valid text on its own: the cut landed
        // inside a character, which is the case a byte-level reader survives
        // and a text-level one does not.
        assert!(split
            .iter()
            .any(|piece| std::str::from_utf8(piece).is_err()));
    }

    #[test]
    fn a_matcher_its_own_recording_would_not_satisfy_is_refused() {
        let mut recorded = cassette("data: [DONE]\n");
        recorded.interactions[0].request.body = Some(r#"{"model":"m"}"#.to_string());
        recorded.interactions[0].request.body_contains[0] = "hello".to_string();
        let error = recorded.validate().unwrap_err();
        assert!(
            error.to_string().contains("not in the request"),
            "expected an unplayable-matcher complaint, got: {error}"
        );
    }
}
