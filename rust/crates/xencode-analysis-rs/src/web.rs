//! Web extraction for research (multimodal step 2) — deterministic page intake.
//!
//! Zero-LLM fetching of a URL into plain text for the context pipeline:
//!
//!   - [`fetch_url`] — GET with a timeout, an `Xencode` user agent, an
//!     [`MAX_PAGE_BYTES`] cap, and an HTML/text content-type gate (binary
//!     formats like PDF are a separate backlog item: document parsing).
//!   - [`extract_text`] — pure HTML→text: drops `script`/`style`/`noscript`/
//!     `template` blocks and comments, strips tags, decodes entities,
//!     collapses whitespace, and pulls the `<title>`.
//!
//! Only `http`/`https` URLs are accepted. Everything is unit-tested without
//! network access except [`fetch_url`], which is tested against a one-shot
//! `TcpListener` server spun up inside the test.

use serde::{Deserialize, Serialize};

/// Refuse pages larger than this — research snippets must fit the context
/// pipeline; a 2 MiB cap stops a stray dump from blowing it up.
pub const MAX_PAGE_BYTES: usize = 2 * 1024 * 1024;

/// Default per-request timeout for research fetches.
pub const FETCH_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("only http/https URLs can be fetched: {0}")]
    BadScheme(String),
    #[error("request failed: {0}")]
    Network(String),
    #[error("server returned status {0}")]
    Status(u16),
    #[error("page exceeds the {1}-byte cap")]
    TooLarge(String, usize),
    #[error("unsupported content type for text extraction: {0}")]
    UnsupportedType(String),
}

/// A fetched page reduced to research-ready text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchedPage {
    pub url: String,
    pub title: Option<String>,
    pub text: String,
    pub bytes: u64,
}

/// GET `url` and reduce it to a [`FetchedPage`]. Async (reqwest); the HTML
/// parsing itself is the pure [`extract_text`].
pub async fn fetch_url(url: &str) -> Result<FetchedPage, FetchError> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(FetchError::BadScheme(url.to_string()));
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .user_agent("Xencode/1.0 (research extraction)")
        .build()
        .map_err(|e| FetchError::Network(e.to_string()))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| FetchError::Network(e.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(FetchError::Status(status.as_u16()));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if !is_textual_content(&content_type) {
        return Err(FetchError::UnsupportedType(content_type));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|e| FetchError::Network(e.to_string()))?;
    if bytes.len() > MAX_PAGE_BYTES {
        return Err(FetchError::TooLarge(url.to_string(), MAX_PAGE_BYTES));
    }
    let body = String::from_utf8_lossy(&bytes);
    let (title, text) = extract_text(&body);
    Ok(FetchedPage {
        url: url.to_string(),
        title,
        text,
        bytes: bytes.len() as u64,
    })
}

/// True for content types text extraction handles: HTML and plain text.
/// Empty (server sent none) is treated as HTML — most origins serve pages.
fn is_textual_content(content_type: &str) -> bool {
    let mime = content_type.split(';').next().unwrap_or("").trim();
    mime.is_empty()
        || mime == "text/html"
        || mime == "text/plain"
        || mime == "application/xhtml+xml"
}

/// Reduce an HTML (or plain-text) document to `(title, text)`. Pure —
/// unit-tested over hand-crafted markup, no fixtures.
pub fn extract_text(html: &str) -> (Option<String>, String) {
    let title = extract_title(html);
    let mut text = html.to_string();
    // Drop non-rendered blocks wholesale (non-greedy, dot-matches-newline).
    for tag in ["script", "style", "noscript", "template"] {
        let re = regex::Regex::new(&format!("(?is)<{tag}\\b.*?</{tag}\\s*>")).unwrap();
        text = re.replace_all(&text, " ").into_owned();
    }
    // Comments, then any remaining tag.
    let comment = regex::Regex::new("(?s)<!--.*?-->").unwrap();
    text = comment.replace_all(&text, " ").into_owned();
    let tag = regex::Regex::new("<[^>]*>").unwrap();
    text = tag.replace_all(&text, " ").into_owned();
    let decoded = decode_entities(&text);
    let collapsed = regex::Regex::new(r"\s+")
        .unwrap()
        .replace_all(&decoded, " ");
    (title, collapsed.trim().to_string())
}

/// Contents of the first `<title>` tag, tags stripped and trimmed.
fn extract_title(html: &str) -> Option<String> {
    let re = regex::Regex::new("(?is)<title\\b.*?>(.*?)</title\\s*>").unwrap();
    let raw = re.captures(html)?.get(1)?.as_str();
    let tag = regex::Regex::new("<[^>]*>").unwrap();
    let title = tag.replace_all(raw, "").trim().to_string();
    if title.is_empty() {
        None
    } else {
        Some(decode_entities(&title))
    }
}

/// Decode the entities extraction actually meets in the wild: the five
/// named basics plus decimal/hex numeric references. Unknown entities are
/// left verbatim rather than dropped.
fn decode_entities(text: &str) -> String {
    let named = regex::Regex::new("&(amp|lt|gt|quot|apos|nbsp);").unwrap();
    let mut out = named
        .replace_all(text, |caps: &regex::Captures| {
            match &caps[1] {
                "amp" => "&",
                "lt" => "<",
                "gt" => ">",
                "quot" => "\"",
                "apos" => "'",
                "nbsp" => " ",
                _ => unreachable!(),
            }
            .to_string()
        })
        .into_owned();
    let decimal = regex::Regex::new("&#([0-9]+);").unwrap();
    out = decimal
        .replace_all(&out, |caps: &regex::Captures| {
            caps[1]
                .parse::<u32>()
                .ok()
                .and_then(char::from_u32)
                .map(|c| c.to_string())
                .unwrap_or_else(|| caps[0].to_string())
        })
        .into_owned();
    let hex = regex::Regex::new("&#x([0-9a-fA-F]+);").unwrap();
    hex.replace_all(&out, |caps: &regex::Captures| {
        u32::from_str_radix(&caps[1], 16)
            .ok()
            .and_then(char::from_u32)
            .map(|c| c.to_string())
            .unwrap_or_else(|| caps[0].to_string())
    })
    .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_tags_scripts_and_comments() {
        let (title, text) = extract_text(
            "<html><head><title>T</title><style>.a{color:red}</style>\
             <script>alert(1)</script></head>\
             <body><!-- hi --><h1>Hello</h1><p>world</p></body></html>",
        );
        assert_eq!(title.as_deref(), Some("T"));
        assert_eq!(text, "T Hello world");
    }

    #[test]
    fn decodes_named_and_numeric_entities() {
        let (_, text) = extract_text("<p>A &amp; B &lt;C&gt; &#65; &#x42; &unknown;</p>");
        assert_eq!(text, "A & B <C> A B &unknown;");
    }

    #[test]
    fn collapses_whitespace_and_handles_missing_title() {
        let (title, text) = extract_text("<p>one\n\n   two\t\tthree</p>");
        assert_eq!(title, None);
        assert_eq!(text, "one two three");
    }

    #[test]
    fn title_entities_and_inner_tags_cleaned() {
        let (title, _) = extract_text("<title>A &amp; <b>B</b></title><p>x</p>");
        assert_eq!(title.as_deref(), Some("A & B"));
    }

    #[test]
    fn plain_text_passes_through_untouched() {
        let (title, text) = extract_text("just some text");
        assert_eq!(title, None);
        assert_eq!(text, "just some text");
    }

    #[test]
    fn content_type_gate_accepts_pages_rejects_binaries() {
        assert!(is_textual_content("text/html; charset=utf-8"));
        assert!(is_textual_content("text/plain"));
        assert!(is_textual_content("application/xhtml+xml"));
        assert!(is_textual_content(""));
        assert!(!is_textual_content("application/pdf"));
        assert!(!is_textual_content("image/png"));
        assert!(!is_textual_content("application/octet-stream"));
    }

    /// One-shot local HTTP server: deterministic `fetch_url` coverage with
    /// no external network.
    async fn serve_once(body: &'static str, content_type: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            use tokio::io::AsyncWriteExt as _;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{addr}/")
    }

    #[tokio::test]
    async fn fetch_url_extracts_served_page() {
        let url = serve_once(
            "<html><head><title>Hi</title></head><body><p>hello <b>web</b></p></body></html>",
            "text/html",
        )
        .await;
        let page = fetch_url(&url).await.unwrap();
        assert_eq!(page.url, url);
        assert_eq!(page.title.as_deref(), Some("Hi"));
        assert_eq!(page.text, "Hi hello web");
        assert!(page.bytes > 0);
    }

    #[tokio::test]
    async fn fetch_url_rejects_scheme_and_type() {
        assert!(matches!(
            fetch_url("ftp://x/y").await,
            Err(FetchError::BadScheme(_))
        ));
        let url = serve_once("%PDF-1.4", "application/pdf").await;
        assert!(matches!(
            fetch_url(&url).await,
            Err(FetchError::UnsupportedType(_))
        ));
    }
}
