//! Web extraction for research (multimodal step 2) — deterministic page intake.
//!
//! Zero-LLM fetching of a URL into plain text for the context pipeline:
//!
//!   - [`fetch_url`] — GET with a timeout, an `Xencode` user agent, an
//!     [`MAX_PAGE_BYTES`] cap, and an HTML/text content-type gate (binary
//!     formats like PDF are a separate backlog item: document parsing).
//!   - [`fetch_url_guarded`] — the same request for a caller who did not choose
//!     the URL: a scheme check, a refusal of every address inside a private
//!     network or the cloud's metadata range, re-checked at each redirect, with
//!     the redirects followed by this module rather than the HTTP library.
//!   - [`extract_text`] — pure HTML→text: drops `script`/`style`/`noscript`/
//!     `template` blocks and comments, strips tags, decodes entities,
//!     collapses whitespace, and pulls the `<title>`.
//!
//! Only `http`/`https` URLs are accepted. Everything is unit-tested without
//! network access except [`fetch_url`], which is tested against a one-shot
//! `TcpListener` server spun up inside the test.

use serde::{Deserialize, Serialize};
use std::net::{IpAddr, ToSocketAddrs};

/// Refuse pages larger than this — research snippets must fit the context
/// pipeline; a 2 MiB cap stops a stray dump from blowing it up.
pub const MAX_PAGE_BYTES: usize = 2 * 1024 * 1024;

/// Default per-request timeout for research fetches.
pub const FETCH_TIMEOUT_SECS: u64 = 20;

/// How many redirects one fetch will follow. A number rather than "as many as
/// the server sends": a chain that never lands is a way to spend the caller's
/// time, and every hop is a place a new host can appear.
pub const MAX_REDIRECTS: usize = 5;

/// How much of a page one caller is handed back. [`MAX_PAGE_BYTES`] protects the
/// fetch from a huge download; this protects the *reader* — a terminal screen or
/// a model's context window — from a page that fits the first limit and still
/// runs to half a million characters.
pub const DEFAULT_TEXT_CAP_CHARS: usize = 30_000;

/// Cut `text` to `cap` characters, returning what is kept and how many were
/// dropped. A count rather than a yes/no, because the two callers say what is
/// missing in different words — one to a person at a terminal, one to a model —
/// and both have to say how much.
pub fn cap_chars(text: &str, cap: usize) -> (String, usize) {
    let total = text.chars().count();
    if total <= cap {
        return (text.to_string(), 0);
    }
    (text.chars().take(cap).collect(), total - cap)
}

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
    /// The address is inside a range a fetch nobody chose must not reach, or
    /// could not be placed outside one.
    #[error("refusing to fetch {0}: {1}")]
    Blocked(String, String),
    #[error("more than {0} redirects — the chain never landed")]
    TooManyRedirects(usize),
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
///
/// This is the path for a URL a person chose — `xencode fetch <url>`, `xencode
/// prices fetch`. It caps size and rejects binaries, but it will fetch
/// `http://10.0.0.8/` if told to, because the person who typed the address is
/// the authority on it. Anything chosen somewhere else uses
/// [`fetch_url_guarded`].
pub async fn fetch_url(url: &str) -> Result<FetchedPage, FetchError> {
    fetch(url, false).await
}

/// [`fetch_url`] with the local-network guard on, for a caller who did not pick
/// the address.
///
/// An agent's fetch is chosen by a model, and a model is handed addresses by the
/// pages, issues and files it reads. Without this, the tool is an internal
/// network probe with a nice interface: the same request that returns a
/// documentation page returns whatever a service on `10.0.0.8:8080` answers, and
/// `169.254.169.254` answers with this machine's cloud credentials. So the host
/// is resolved and checked before the connection is made, and re-checked at
/// every redirect — a `Location:` header is exactly as untrusted as the URL that
/// carried it, which is why the redirects are walked here instead of left to
/// reqwest's redirect policy.
pub async fn fetch_url_guarded(url: &str) -> Result<FetchedPage, FetchError> {
    fetch(url, true).await
}

async fn fetch(url: &str, guarded: bool) -> Result<FetchedPage, FetchError> {
    // Both paths, not just the guarded one: a scheme reqwest cannot dial is a
    // mistake in the address, and it should be said so rather than arriving as
    // a transport failure.
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(FetchError::BadScheme(url.to_string()));
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .user_agent("Xencode/1.0 (research extraction)")
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| FetchError::Network(e.to_string()))?;
    let mut target = url.to_string();
    let mut hops = 0;
    let response = loop {
        if guarded {
            guard_destination(&target)?;
        }
        let answered = client
            .get(&target)
            .send()
            .await
            .map_err(|e| FetchError::Network(e.to_string()))?;
        if !answered.status().is_redirection() {
            break answered;
        }
        // A 3xx with no `Location` is an answer, not a step: report the status
        // the server actually sent rather than invent somewhere to go.
        let Some(location) = answered.headers().get(reqwest::header::LOCATION) else {
            break answered;
        };
        let location = location
            .to_str()
            .map_err(|e| FetchError::Network(e.to_string()))?;
        let joined = reqwest::Url::parse(&target)
            .map_err(|_| FetchError::BadScheme(target.clone()))?
            .join(location)
            .map_err(|e| FetchError::Network(e.to_string()))?
            .to_string();
        hops += 1;
        if hops > MAX_REDIRECTS || joined == target {
            return Err(FetchError::TooManyRedirects(MAX_REDIRECTS));
        }
        target = joined;
    };
    intake(response, url).await
}

/// Reduce a response to a [`FetchedPage`], the part both entry points share.
async fn intake(response: reqwest::Response, requested: &str) -> Result<FetchedPage, FetchError> {
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
    // Where the request actually ended up, which is the last redirect target
    // rather than the address we were handed. `bytes()` consumes the response,
    // so this is read while it is still there.
    let landed = response.url().to_string();
    let bytes = response
        .bytes()
        .await
        .map_err(|e| FetchError::Network(e.to_string()))?;
    if bytes.len() > MAX_PAGE_BYTES {
        return Err(FetchError::TooLarge(requested.to_string(), MAX_PAGE_BYTES));
    }
    let body = String::from_utf8_lossy(&bytes).into_owned();
    let (title, text) = if is_json_content(&content_type) {
        // JSON has no markup to strip, and stripping it would be harmful: the tag
        // regex eats everything between two `<` characters, and an entity like
        // `&lt;` inside a code sample is not markup at all. Hand the document
        // over as it came.
        (None, body.trim().to_string())
    } else if content_type.starts_with("text/plain") {
        // The previous gate ran plain text through the HTML stripper, which
        // collapsed its line breaks. A log, a README and a stack trace are read
        // for their lines.
        (None, body.trim().to_string())
    } else {
        extract_text(&body)
    };
    Ok(FetchedPage {
        url: landed,
        title,
        text,
        bytes: bytes.len() as u64,
    })
}

/// The host of one URL has to resolve, and every address it resolves to has to
/// be one a stranger may be pointed at. Failing to resolve is a refusal rather
/// than a pass: an unresolvable name is also exactly what a host that only
/// exists on a company's internal DNS looks like from out here.
///
/// Public because the approval prompt asks the same question before the person
/// answers, and a second copy of the rule is one that can drift from the first.
pub fn guard_destination(url: &str) -> Result<(), FetchError> {
    let parsed = reqwest::Url::parse(url).map_err(|_| FetchError::BadScheme(url.to_string()))?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => return Err(FetchError::BadScheme(other.to_string())),
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| FetchError::Blocked(url.to_string(), "no host to connect to".to_string()))?
        .to_string();
    let port = parsed.port_or_known_default().unwrap_or(80);
    // An IPv6 literal arrives from `host_str()` with its brackets still on.
    let bare = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(&host);
    if let Ok(ip) = bare.parse::<IpAddr>() {
        return match denied_range(ip) {
            Some(why) => Err(FetchError::Blocked(
                url.to_string(),
                format!("{ip} is {why}"),
            )),
            None => Ok(()),
        };
    }
    let resolved: Vec<IpAddr> = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| FetchError::Blocked(url.to_string(), format!("{host} does not resolve ({e}) — an address only internal DNS knows is what this guard is for")))?
        .map(|addr| addr.ip())
        .collect();
    if resolved.is_empty() {
        return Err(FetchError::Blocked(
            url.to_string(),
            format!("{host} resolves to nothing"),
        ));
    }
    if let Some(ip) = resolved
        .iter()
        .find_map(|ip| denied_range(*ip).map(|why| (*ip, why)))
    {
        return Err(FetchError::Blocked(
            url.to_string(),
            format!("{host} resolves to {}, which is {}", ip.0, ip.1),
        ));
    }
    Ok(())
}

/// Why this address is off-limits to a fetch nobody chose, or `None` when it is
/// fair game.
///
/// Loopback is deliberately **allowed**. The ranges a model-chosen fetch must
/// not reach are the ones that are not this machine: another host on a private
/// network, the carrier-grade range in front of one, and the link-local address
/// every cloud answers instance metadata and credentials from. `127.0.0.1` is a
/// service on this box — which `xencode fetch` and this crate's own tests
/// address — and a guard that refused it would be one nobody could run.
fn denied_range(ip: IpAddr) -> Option<String> {
    // Checked first so the answer does not depend on which branch a loopback
    // address happens to fall into: `::1` starts with a zero segment, and a
    // range test written on the raw words would refuse it as the unspecified
    // address.
    if ip.is_loopback() {
        return None;
    }
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            if a == 10 {
                Some("the 10.0.0.0/8 private network".to_string())
            } else if a == 172 && (16..=31).contains(&b) {
                Some("the 172.16.0.0/12 private network".to_string())
            } else if a == 192 && b == 168 {
                Some("the 192.168.0.0/16 private network".to_string())
            } else if a == 169 && b == 254 {
                Some(
                    "the link-local range, which is where a cloud serves its instance metadata \
                     and credentials"
                        .to_string(),
                )
            } else if a == 100 && (64..=127).contains(&b) {
                Some("the carrier-grade NAT range, inside a provider's network".to_string())
            } else if a == 0 {
                Some("the `this host on this network` range".to_string())
            } else if a >= 224 {
                Some("a multicast group".to_string())
            } else {
                None
            }
        }
        IpAddr::V6(v6) => {
            let first = v6.segments()[0];
            if first & 0xfe00 == 0xfc00 {
                Some("the IPv6 unique-local range".to_string())
            } else if first & 0xffc0 == 0xfe80 {
                Some("the IPv6 link-local range".to_string())
            } else if first == 0 {
                Some("the unspecified address".to_string())
            } else if first & 0xff00 == 0xff00 {
                Some("an IPv6 multicast group".to_string())
            } else {
                None
            }
        }
    }
}

/// True for content types text extraction handles: HTML, plain text and JSON.
/// An empty header (the server sent none) is treated as HTML — most origins
/// serve pages.
///
/// JSON is here because the useful machine-readable half of the web answers it:
/// an API response, a security advisory feed and a package registry are all
/// `application/json`, and a fetcher whose gate refused it could not read any of
/// them. The suffixed profiles (`application/feed+json`, `application/geo+json`)
/// are the same document with a name on it.
fn is_textual_content(content_type: &str) -> bool {
    let mime = content_type.split(';').next().unwrap_or("").trim();
    mime.is_empty()
        || mime == "text/html"
        || mime == "text/plain"
        || mime == "application/xhtml+xml"
        || mime.starts_with("text/")
        || is_json_content(content_type)
}

/// Whether this content type is a JSON document, suffixed profiles included.
fn is_json_content(content_type: &str) -> bool {
    let mime = content_type.split(';').next().unwrap_or("").trim();
    mime == "application/json" || mime.ends_with("+json")
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
        assert!(is_textual_content("text/csv"));
        assert!(is_textual_content("application/json"));
        assert!(is_textual_content("application/vnd.api+json"));
        assert!(is_textual_content("application/feed+json; charset=utf-8"));
        assert!(!is_textual_content("application/pdf"));
        assert!(!is_textual_content("image/png"));
        assert!(!is_textual_content("application/octet-stream"));
        assert!(!is_textual_content("application/zip"));
    }

    #[test]
    fn denied_ranges_cover_private_link_local_and_multicast() {
        for addr in [
            "10.0.0.8",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "100.127.255.255",
            "0.0.0.0",
            "224.0.0.5",
            "239.10.10.10",
        ] {
            let ip: IpAddr = addr.parse().unwrap();
            assert!(denied_range(ip).is_some(), "{addr} must be refused");
        }
        for addr in ["fd00::1", "fc12:3456::78", "fe80::1", "::", "ff02::1"] {
            let ip: IpAddr = addr.parse().unwrap();
            assert!(denied_range(ip).is_some(), "{addr} must be refused");
        }
        // Adjacent to a denied range but outside it: the boundaries matter as
        // much as the blocks, because a range that is one octet too wide
        // refuses the public internet's own addresses.
        for addr in [
            "172.15.0.1",
            "172.32.0.1",
            "100.63.255.255",
            "100.128.0.1",
            "168.254.1.1",
        ] {
            let ip: IpAddr = addr.parse().unwrap();
            assert!(denied_range(ip).is_none(), "{addr} must be allowed");
        }
    }

    #[test]
    fn loopback_is_allowed_by_design_and_public_addresses_are() {
        for addr in ["127.0.0.1", "::1", "93.184.216.34", "2001:4860:4860::8888"] {
            let ip: IpAddr = addr.parse().unwrap();
            assert!(denied_range(ip).is_none(), "{addr} must be allowed");
        }
    }

    #[test]
    fn guard_refuses_a_metadata_ip_and_an_unresolvable_name() {
        let err = guard_destination("http://169.254.169.254/latest/meta-data/")
            .expect_err("the cloud metadata address must be refused");
        assert!(
            matches!(&err, FetchError::Blocked(_, why) if why.contains("169.254.169.254")),
            "{err}"
        );
        // IPv6 literals arrive bracketed from the URL parser; the bare form must
        // be checked too or the guard is one syntax away from being useless.
        assert!(guard_destination("http://[fd00::1]/").is_err());
        assert!(guard_destination("http://[fd00::1]:8080/x").is_err());
        assert!(matches!(
            guard_destination("ftp://example.com/"),
            Err(FetchError::BadScheme(_))
        ));
        // A name that resolves nowhere is a refusal, not a pass.
        let err = guard_destination("http://xencode-not-a-real-host.invalid./")
            .expect_err("an unresolvable host must be refused");
        assert!(matches!(&err, FetchError::Blocked(_, _)), "{err}");
    }

    #[test]
    fn guard_allows_a_public_address_and_the_local_test_server() {
        guard_destination("http://127.0.0.1:8080/").expect("loopback is allowed");
        // No network here, so this asserts only that a public literal is not in
        // a denied range — the resolve step is what needs the internet.
        assert!(denied_range("93.184.216.34".parse::<IpAddr>().unwrap()).is_none());
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

    /// A server that never stops redirecting: every request gets a 302 to a new
    /// path, so a fetcher that follows redirects blindly has no end to reach.
    async fn serve_endless_redirects() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt as _;
            let mut hop = 0usize;
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                hop += 1;
                let response = format!(
                    "HTTP/1.1 302 Found\r\nlocation: /hop{hop}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}/")
    }

    /// A server that answers the first request with a redirect to a given
    /// address — the shape of the attack the per-hop check exists for.
    async fn serve_redirect_to(location: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            use tokio::io::AsyncWriteExt as _;
            let response = format!(
                "HTTP/1.1 302 Found\r\nlocation: {location}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
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
        assert!(matches!(
            fetch_url_guarded("ftp://x/y").await,
            Err(FetchError::BadScheme(_))
        ));
        let url = serve_once("%PDF-1.4", "application/pdf").await;
        assert!(matches!(
            fetch_url(&url).await,
            Err(FetchError::UnsupportedType(_))
        ));
    }

    #[tokio::test]
    async fn json_is_returned_as_it_arrived() {
        // The old gate refused JSON outright; the fix must not trade that for
        // feeding JSON through the HTML stripper, whose tag regex eats
        // everything between two `<` characters.
        let body = r#"{"field":"a < b","nested":{"ok":[1,2,3]}}"#;
        let url = serve_once(body, "application/json").await;
        let page = fetch_url(&url).await.unwrap();
        assert_eq!(page.text, body);
        assert!(serde_json::from_str::<serde_json::Value>(&page.text).is_ok());
    }

    #[tokio::test]
    async fn plain_text_keeps_its_line_breaks() {
        let url = serve_once("first line\nsecond line\nthird", "text/plain").await;
        let page = fetch_url(&url).await.unwrap();
        assert_eq!(page.text, "first line\nsecond line\nthird");
    }

    #[tokio::test]
    async fn guarded_fetch_serves_a_local_page() {
        // Loopback is allowed on purpose, and this is the proof the guard has
        // not been widened into something that cannot run at all.
        let url = serve_once("<p>guarded</p>", "text/html").await;
        let page = fetch_url_guarded(&url).await.unwrap();
        assert_eq!(page.text, "guarded");
    }

    #[tokio::test]
    async fn a_redirect_into_the_metadata_range_is_refused() {
        let url = serve_redirect_to("http://169.254.169.254/latest/meta-data/").await;
        let err = fetch_url_guarded(&url)
            .await
            .expect_err("a redirect must not smuggle the fetch into a denied range");
        assert!(
            matches!(&err, FetchError::Blocked(_, why) if why.contains("169.254.169.254")),
            "{err}"
        );
    }

    /// A server for two hops: the first request is redirected, the second is
    /// answered with a page. Sequential by nature — a fetch makes one request
    /// at a time — so the hop order is the test's assertion.
    async fn serve_redirect_then_page(path: &'static str) -> (String, String) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt as _;
            let body =
                "<html><head><title>Landed</title></head><body><p>second hop</p></body></html>";
            for hop in 0..2 {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let response = if hop == 0 {
                    format!(
                        "HTTP/1.1 302 Found\r\nlocation: {path}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    )
                } else {
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                };
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        (format!("http://{addr}/"), format!("http://{addr}{path}"))
    }

    #[tokio::test]
    async fn a_redirect_is_followed_and_the_landing_url_is_reported() {
        let (url, landed) = serve_redirect_then_page("/landed").await;
        let page = fetch_url(&url).await.unwrap();
        assert_eq!(page.url, landed);
        assert_eq!(page.title.as_deref(), Some("Landed"));
        assert_eq!(page.text, "Landed second hop");
    }

    #[tokio::test]
    async fn a_guarded_fetch_follows_a_redirect_it_is_allowed_to_take() {
        // Both hops are loopback, which the guard permits on purpose. The
        // landing URL is the second hop, so this also proves the guarded path
        // tracks where it actually ended up rather than what it was asked for.
        let (url, landed) = serve_redirect_then_page("/landed").await;
        let page = fetch_url_guarded(&url)
            .await
            .expect("a guarded fetch must still be able to reach a local server");
        assert_eq!(page.url, landed);
        assert_eq!(page.text, "Landed second hop");
    }

    #[tokio::test]
    async fn a_redirect_chain_that_never_lands_is_stopped() {
        let url = serve_endless_redirects().await;
        let err = fetch_url(&url)
            .await
            .expect_err("an endless redirect chain must end in an error");
        assert!(
            matches!(&err, FetchError::TooManyRedirects(n) if *n == MAX_REDIRECTS),
            "{err}"
        );
    }
}
