//! Reading a model catalogue that prices itself (CX-4).
//!
//! One job: bring back the bytes of OpenRouter's public model listing, so that
//! whatever xencode reports as a price can be traced to a document rather than to
//! a list baked into this binary. Nothing here interprets those bytes — that is
//! `PriceLookup` in the context crate, which is tested against a listing whose
//! shape came from a real answer. This module only dials, checks the answer is
//! worth reading, and hands it over.
//!
//! No key is sent and none is needed: the same listing is published for anybody
//! to read. The request still carries an `Xencode` user agent, because a server
//! that has to decide about an unnamed caller decides badly.

/// Where the listing is read from. Overridable at the call site so a test can
/// serve a real HTTP answer locally instead of pretending to.
pub const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

/// Refuse a listing larger than this. The document fetched here was 764,719
/// bytes covering 466 models; eight megabytes is the same listing several times
/// over, which is a server that has started answering a different question.
pub const LISTING_BYTES_CAP: usize = 8 * 1024 * 1024;

// A build that lowers the cap under the real document should fail here rather
// than fail to price anything, quietly, later.
const _: () = assert!(LISTING_BYTES_CAP > 764_719);

/// How long to wait for the listing. A price lookup is never on the path of a
/// turn, so this can afford to wait out a slow server without holding anything up.
pub const LISTING_TIMEOUT_SECS: u64 = 20;

/// Why a listing could not be read.
#[derive(Debug, PartialEq, Eq)]
pub enum ListingError {
    /// The address is not one a price could be fetched from.
    BadUrl(String),
    /// The request did not complete.
    Unreachable(String),
    /// The server answered, but not with the listing.
    Status(u16),
    /// The answer was larger than [`LISTING_BYTES_CAP`].
    TooLarge(usize),
}

impl std::fmt::Display for ListingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ListingError::BadUrl(url) => write!(f, "only http/https can be fetched: {url}"),
            ListingError::Unreachable(why) => {
                write!(f, "the price listing could not be reached: {why}")
            }
            ListingError::Status(code) => {
                write!(f, "the price listing answered {code} instead of a body")
            }
            ListingError::TooLarge(cap) => {
                write!(
                    f,
                    "the price listing is over the {cap}-byte limit; refusing it"
                )
            }
        }
    }
}

impl std::error::Error for ListingError {}

/// GET the catalogue at `url` and return its body as text.
pub async fn fetch_listing(url: &str) -> Result<String, ListingError> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(ListingError::BadUrl(url.to_string()));
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(LISTING_TIMEOUT_SECS))
        .user_agent(concat!(
            "Xencode/",
            env!("CARGO_PKG_VERSION"),
            " (price lookup)"
        ))
        .build()
        .map_err(|e| ListingError::Unreachable(e.to_string()))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| ListingError::Unreachable(e.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        return Err(ListingError::Status(status.as_u16()));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|e| ListingError::Unreachable(e.to_string()))?;
    if bytes.len() > LISTING_BYTES_CAP {
        return Err(ListingError::TooLarge(LISTING_BYTES_CAP));
    }
    // Lossy on purpose: a stray byte in a description field is not a reason to
    // lose the prices, and the parser that reads this asks for nothing but the
    // numbers and the model ids.
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// The listing this build reads prices out of.
pub async fn fetch_openrouter_listing() -> Result<String, ListingError> {
    fetch_listing(OPENROUTER_MODELS_URL).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real HTTP server on a loopback port, answering once. Not a mock: the
    /// request goes over a socket and the response is parsed by reqwest.
    async fn serve(status_line: &'static str, body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt as _;
            let (mut stream, _) = listener.accept().await.unwrap();
            // Read the request line so the client gets its write out of the way
            // before the answer arrives.
            let mut buf = [0u8; 1024];
            let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await;
            let response = format!(
                "{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
        format!("http://{addr}/api/v1/models")
    }

    #[tokio::test]
    async fn a_served_listing_is_handed_over_byte_for_byte() {
        let body = r#"{"data":[{"id":"openai/gpt-5","pricing":{"prompt":"0.00000125"}}]}"#;
        let url = serve("HTTP/1.1 200 OK", body).await;
        assert_eq!(fetch_listing(&url).await.unwrap(), body);
    }

    #[tokio::test]
    async fn an_answer_that_is_not_the_listing_is_refused_with_its_own_number() {
        let url = serve("HTTP/1.1 503 Service Unavailable", "{}").await;
        assert_eq!(
            fetch_listing(&url).await.unwrap_err(),
            ListingError::Status(503)
        );
    }

    #[tokio::test]
    async fn an_address_that_is_not_a_web_origin_is_refused_before_anything_is_dialed() {
        for bad in ["file:///etc/passwd", "ftp://example/models"] {
            assert_eq!(
                fetch_listing(bad).await.unwrap_err(),
                ListingError::BadUrl(bad.to_string()),
                "{bad}"
            );
        }
    }

    #[tokio::test]
    async fn a_server_that_never_answers_is_a_reason_and_not_a_hang() {
        // Bind a port and drop it, so the connection is refused rather than
        // swallowed by something slower.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let error = fetch_listing(&format!("http://{addr}/api/v1/models"))
            .await
            .unwrap_err();
        assert!(matches!(error, ListingError::Unreachable(_)), "{error:?}");
        assert!(
            error.to_string().contains("could not be reached"),
            "{error}"
        );
    }

    #[test]
    fn the_listing_this_build_reads_is_the_public_one_needing_no_key() {
        assert_eq!(OPENROUTER_MODELS_URL, "https://openrouter.ai/api/v1/models");
    }
}
