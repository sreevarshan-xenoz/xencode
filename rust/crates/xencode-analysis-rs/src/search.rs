//! One search call, behind a setting that is off by default.
//!
//! The obvious version of this feature — "just use DuckDuckGo, it is free" — was
//! measured rather than assumed, and it is not there. `lite.duckduckgo.com`
//! answers this machine with a *"Unfortunately, bots use DuckDuckGo too"*
//! CAPTCHA (`cc=botnet`) one time and a real results page another, and
//! `duckduckgo.com/developer/search-api` is **410 Gone**. Public SearXNG
//! instances answer `format=json` with an HTML document (200, `<!doctype html>`)
//! or with **429**, which matches SearXNG's own documentation that public
//! instances disable JSON. MDN's JSON search endpoint is **404**. So every
//! provider here is one a person names in their own config, `none` is the
//! default, and the keyless option that actually works is Wikipedia's own API.
//!
//! What this module guarantees is the part that is not about which engine you
//! picked: the address is resolved and refused before the connection is opened,
//! using [`crate::web::guard_destination`], so a self-hosted `search_url` cannot
//! point the request at `169.254.169.254` or a host on a private network, and a
//! credential is never sent anywhere except the host that provider's key is for.

use thiserror::Error;

/// One result, reduced to what a model can act on: where it came from and what
/// the engine said about it. Nothing here fetches the page — reading a result is
/// a separate decision with its own approval, which is what keeps a list of
/// links from being a way to browse the web without saying so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Error)]
pub enum SearchError {
    /// Nothing is configured, or what is configured is missing the half it
    /// needs. The message names the setting to change rather than failing as a
    /// transport error, because "no search provider" and "the search provider
    /// could not be reached" are different things a person can act on.
    #[error("{0}")]
    NotConfigured(String),
    #[error("search query is empty")]
    EmptyQuery,
    #[error("search query is longer than {0} characters")]
    QueryTooLong(usize),
    #[error("cannot search {0}: {1}")]
    Blocked(String, String),
    #[error("the search provider answered with status {0}")]
    Status(u16),
    #[error("the search provider's answer is not the JSON its API returns: {0}")]
    BadAnswer(String),
    #[error("the search request failed: {0}")]
    Network(String),
    #[error("the search provider redirected {0} times and never answered")]
    TooManyRedirects(usize),
    #[error("the search provider's answer is larger than {0} bytes")]
    TooLarge(usize),
}

/// The provider names a config may hold. Public because `xencode config set`
/// checks a typed name against this list rather than accepting anything and
/// letting the first search fail with it.
pub const PROVIDER_NAMES: [&str; 5] = ["none", "wikipedia", "searxng", "brave", "tavily"];

/// How many results one call may ask for. Every provider in here caps a page at
/// a similar size; the limit exists so `max_results` cannot be used to pull a
/// hundred links into the context window in one call.
pub const MAX_RESULTS: usize = 10;

/// The longest query accepted, in characters. A search box is not a place to put
/// a file, and a provider that silently truncates one turns a question into a
/// different question.
pub const MAX_QUERY_CHARS: usize = 400;

/// A search engine the person has named in their config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provider {
    /// Nothing. The default, and the answer for a machine that has not opted in.
    None,
    /// Wikipedia's `action=query&list=search` API. Keyless and free, and the one
    /// keyless option measured here that still answers JSON. It answers about
    /// things, not about a crate's API, so it is a reference rather than a
    /// documentation search.
    Wikipedia,
    /// A SearXNG instance the person runs, at that exact address.
    Searxng { base: String },
    /// Brave Search API, with the key the person supplied.
    Brave { key: String },
    /// Tavily, with the key the person supplied.
    Tavily { key: String },
}

impl Provider {
    /// Read the provider out of the settings that name it.
    ///
    /// `key` is the credential for whichever provider was chosen — Brave and
    /// Tavily need one, the others ignore it — so a caller resolves the right one
    /// and does not have to know which provider ends up using it. A `searxng`
    /// with no address configured is [`SearchError::NotConfigured`], not a
    /// provider that fails on first use: the mistake is in the config, and it is
    /// worth saying before a request is attempted.
    pub fn parse(name: &str, searxng_url: &str, key: Option<String>) -> Result<Self, SearchError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "none" => Ok(Self::None),
            "wikipedia" => Ok(Self::Wikipedia),
            "searxng" => {
                let base = searxng_url.trim().trim_end_matches('/');
                if base.is_empty() {
                    return Err(SearchError::NotConfigured(
                        "search_provider is `searxng` but search_searxng_url is empty: point it at \
                         an instance you run, e.g. `xencode config set search_searxng_url \
                         http://127.0.0.1:8888`"
                            .to_string(),
                    ));
                }
                Ok(Self::Searxng {
                    base: base.to_string(),
                })
            }
            "brave" => Ok(Self::Brave {
                key: need_key(name, key)?,
            }),
            "tavily" => Ok(Self::Tavily {
                key: need_key(name, key)?,
            }),
            other => Err(SearchError::NotConfigured(format!(
                "`{other}` is not a search provider: the names are {}",
                PROVIDER_NAMES
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }

    /// The one word that names this engine to a person: what a search answer is
    /// headed with, and what `config show` holds in `search_provider`. Written
    /// out rather than derived from the enum's own `Debug`, because `Debug` of a
    /// key-bearing variant prints the credential.
    pub fn slug(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Wikipedia => "wikipedia",
            Self::Searxng { .. } => "searxng",
            Self::Brave { .. } => "brave",
            Self::Tavily { .. } => "tavily",
        }
    }
}

fn need_key(name: &str, key: Option<String>) -> Result<String, SearchError> {
    match key.filter(|k| !k.trim().is_empty()) {
        Some(key) => Ok(key),
        None => Err(SearchError::NotConfigured(format!(
            "search_provider is `{name}` and it needs a key of your own: `xencode config set {name}_api_key …`, \
             or put it in the `{}` environment variable",
            env_var_for(name)
        ))),
    }
}

fn env_var_for(name: &str) -> &'static str {
    match name {
        "brave" => "API_KEY_BRAVE",
        _ => "API_KEY_TAVILY",
    }
}

/// Search `query` and return at most `max_results` hits.
///
/// [`Provider::None`] answers [`SearchError::NotConfigured`] rather than an empty
/// list: a search that found nothing and a machine that has no search are
/// different answers, and a model that is told "no results" will go looking
/// again.
pub async fn search(
    provider: &Provider,
    query: &str,
    max_results: usize,
) -> Result<Vec<Hit>, SearchError> {
    let query = query.trim();
    if query.is_empty() {
        return Err(SearchError::EmptyQuery);
    }
    if query.chars().count() > MAX_QUERY_CHARS {
        return Err(SearchError::QueryTooLong(MAX_QUERY_CHARS));
    }
    let want = max_results.clamp(1, MAX_RESULTS);
    let value = match provider {
        Provider::None => {
            return Err(SearchError::NotConfigured(
                "no search provider is configured: `xencode config set search_provider wikipedia` \
                 uses Wikipedia's API, and `searxng`, `brave` and `tavily` need an address or a key \
                 of your own"
                    .to_string(),
            ))
        }
        Provider::Wikipedia => {
            let url = format!(
                "https://en.wikipedia.org/w/api.php?action=query&list=search&srlimit={want}\
                 &srsearch={}&format=json",
                percent_encode(query)
            );
            get_json(&url, &[]).await?
        }
        Provider::Searxng { base } => {
            let url = format!(
                "{base}/search?q={}&format=json&categories=general&pageno=1",
                percent_encode(query)
            );
            get_json(&url, &[]).await?
        }
        Provider::Brave { key } => {
            let url = format!(
                "https://api.search.brave.com/res/v1/web/search?q={}&count={want}",
                percent_encode(query)
            );
            get_json(&url, &[("X-Subscription-Token", key.clone())]).await?
        }
        Provider::Tavily { key } => {
            let body = serde_json::json!({
                "query": query,
                "max_results": want,
                "search_depth": "basic",
            });
            post_json(
                "https://api.tavily.com/search",
                &body,
                &[("Authorization", format!("Bearer {key}"))],
            )
            .await?
        }
    };
    Ok(match provider {
        Provider::Wikipedia => wikipedia_hits(&value),
        Provider::Searxng { .. } => searxng_hits(&value),
        Provider::Brave { .. } => brave_hits(&value),
        Provider::Tavily { .. } => tavily_hits(&value),
        Provider::None => Vec::new(),
    })
}

/// Wikipedia's own answer: `query.search[]`, each with a `title`, a `pageid` and
/// a `snippet` whose matches are wrapped in `<span class="searchmatch">`. The
/// markup goes away with [`crate::web::extract_text`], which is the same
/// reduction a fetched page gets — the alternative is handing a model a sentence
/// with tags in it and hoping it reads around them.
fn wikipedia_hits(value: &serde_json::Value) -> Vec<Hit> {
    let mut hits = Vec::new();
    let Some(rows) = value
        .get("query")
        .and_then(|q| q.get("search"))
        .and_then(|s| s.as_array())
    else {
        return hits;
    };
    for row in rows {
        let Some(title) = row.get("title").and_then(|t| t.as_str()) else {
            continue;
        };
        hits.push(Hit {
            title: title.to_string(),
            // `pageid` is the address that does not break when a title is
            // redirected or renamed; the canonical form is kept as the readable
            // part of the URL because a model has to be able to say what it read.
            url: format!(
                "https://en.wikipedia.org/wiki/{}",
                percent_encode_path(title)
            ),
            snippet: crate::web::extract_text(
                row.get("snippet").and_then(|s| s.as_str()).unwrap_or(""),
            )
            .1
            .trim()
            .to_string(),
        });
    }
    hits
}

/// SearXNG's documented JSON: a top-level `results[]` of `title`, `url` and
/// `content`. `engine` and `parsed_url` are also there and are not used: the
/// instance's own `url` is what the next fetch will be pointed at, and hiding a
/// rewrite behind an `engine` label is worse than showing it.
fn searxng_hits(value: &serde_json::Value) -> Vec<Hit> {
    rows(value, "results", &["content", "snippet"])
}

/// Brave Search API: `web.results[]` with `title`, `url` and `description`.
fn brave_hits(value: &serde_json::Value) -> Vec<Hit> {
    let Some(web) = value.get("web") else {
        return Vec::new();
    };
    rows(web, "results", &["description", "snippet"])
}

/// Tavily: a top-level `results[]` with `title`, `url` and `content`, already
/// summarised by the service.
fn tavily_hits(value: &serde_json::Value) -> Vec<Hit> {
    rows(value, "results", &["content"])
}

fn rows(value: &serde_json::Value, key: &str, snippet_keys: &[&str]) -> Vec<Hit> {
    let mut hits = Vec::new();
    let Some(rows) = value.get(key).and_then(|r| r.as_array()) else {
        return hits;
    };
    for row in rows {
        let (Some(title), Some(url)) = (
            row.get("title").and_then(|t| t.as_str()),
            row.get("url").and_then(|u| u.as_str()),
        ) else {
            continue;
        };
        let snippet = snippet_keys
            .iter()
            .find_map(|k| row.get(*k).and_then(|s| s.as_str()))
            .unwrap_or("")
            .to_string();
        hits.push(Hit {
            title: title.to_string(),
            url: url.to_string(),
            snippet: crate::web::extract_text(&snippet).1.trim().to_string(),
        });
    }
    hits
}

/// GET a provider's API and parse its body as JSON.
///
/// The address is guarded on every hop, the same rule [`crate::web`] applies to a
/// fetched page: a `search_searxng_url` is a URL the model's caller typed once,
/// and a redirect out of it is as untrusted as a redirect out of any page.
async fn get_json(url: &str, headers: &[(&str, String)]) -> Result<serde_json::Value, SearchError> {
    request(url, None, headers).await
}

async fn post_json(
    url: &str,
    body: &serde_json::Value,
    headers: &[(&str, String)],
) -> Result<serde_json::Value, SearchError> {
    request(url, Some(body), headers).await
}

async fn request(
    url: &str,
    body: Option<&serde_json::Value>,
    headers: &[(&str, String)],
) -> Result<serde_json::Value, SearchError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            crate::web::FETCH_TIMEOUT_SECS,
        ))
        .user_agent("Xencode/1.0 (research extraction)")
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| SearchError::Network(e.to_string()))?;
    let mut target = url.to_string();
    let mut hops = 0;
    let answered = loop {
        crate::web::guard_destination(&target).map_err(blocked(&target))?;
        let mut call = match body {
            None => client.get(&target),
            Some(body) => client.post(&target).json(body),
        };
        for (name, value) in headers {
            call = call.header(*name, value);
        }
        let response = call
            .send()
            .await
            .map_err(|e| SearchError::Network(e.to_string()))?;
        if !response.status().is_redirection() {
            break response;
        }
        // A POST that redirects is not followed: the key would go to wherever the
        // `Location:` says, and a search provider has no reason to do that.
        if body.is_some() {
            return Err(SearchError::Blocked(
                target,
                "the provider answered a key-bearing request with a redirect, and the key is not \
                 sent on"
                    .to_string(),
            ));
        }
        let Some(location) = response.headers().get(reqwest::header::LOCATION) else {
            break response;
        };
        let location = location
            .to_str()
            .map_err(|e| SearchError::Network(e.to_string()))?;
        let joined = reqwest::Url::parse(&target)
            .map_err(|_| SearchError::Blocked(target.clone(), "not a URL".to_string()))?
            .join(location)
            .map_err(|e| SearchError::Network(e.to_string()))?
            .to_string();
        hops += 1;
        if hops > crate::web::MAX_REDIRECTS || joined == target {
            return Err(SearchError::TooManyRedirects(crate::web::MAX_REDIRECTS));
        }
        target = joined;
    };
    let status = answered.status();
    if !status.is_success() {
        // The body of an error is not quoted into the answer: a gateway page from
        // a provider can carry anything, and this is a model-facing string.
        return Err(SearchError::Status(status.as_u16()));
    }
    let content_type = answered
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = answered
        .bytes()
        .await
        .map_err(|e| SearchError::Network(e.to_string()))?;
    if bytes.len() > crate::web::MAX_PAGE_BYTES {
        return Err(SearchError::TooLarge(crate::web::MAX_PAGE_BYTES));
    }
    let text = String::from_utf8_lossy(&bytes).into_owned();
    serde_json::from_str::<serde_json::Value>(&text).map_err(|_| {
        SearchError::BadAnswer(if content_type.is_empty() {
            "no content type was given".to_string()
        } else {
            format!("its content type was `{content_type}`")
        })
    })
}

fn blocked(target: &str) -> impl Fn(crate::web::FetchError) -> SearchError + '_ {
    let target = target.to_string();
    move |e| match e {
        crate::web::FetchError::Blocked(_, why) => {
            SearchError::Blocked(target.clone(), why.to_string())
        }
        other => SearchError::Blocked(target.clone(), other.to_string()),
    }
}

/// Percent-encode a query for a `+`-free `application/x-www-form-urlencoded`
/// value: spaces become `%20` rather than `+`, because `+` means a literal plus
/// to at least one of these APIs and the difference is a different question.
fn percent_encode(raw: &str) -> String {
    encode(raw, |c| {
        c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~')
    })
}

/// The same, for a path segment, where `/` separates rather than names.
fn percent_encode_path(raw: &str) -> String {
    encode(raw.replace(' ', "_").as_str(), |c| {
        c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~')
    })
}

fn encode(raw: &str, keep: impl Fn(char) -> bool) -> String {
    let mut out = String::with_capacity(raw.len());
    for byte in raw.as_bytes() {
        let ch = *byte as char;
        if keep(ch) {
            out.push(ch);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_provider_is_named_by_a_key_of_the_five_and_its_extra_half() {
        assert_eq!(Provider::parse("none", "", None).unwrap(), Provider::None);
        // Empty is the same answer as "none", because a config that has the key
        // absent reads as an empty string here and nothing should go out.
        assert_eq!(
            Provider::parse("  ", "", None).unwrap(),
            Provider::None,
            "an unset provider must not search"
        );
        assert_eq!(
            Provider::parse("Wikipedia", "", None).unwrap(),
            Provider::Wikipedia
        );
        // Trailing slashes are what people type; the request adds its own path.
        assert_eq!(
            Provider::parse("searxng", "http://127.0.0.1:8888/", None).unwrap(),
            Provider::Searxng {
                base: "http://127.0.0.1:8888".to_string()
            }
        );
        let missing_url = Provider::parse("searxng", "", None)
            .unwrap_err()
            .to_string();
        assert!(missing_url.contains("search_searxng_url"), "{missing_url}");
        // A key-based provider with no key is refused at the config, before a
        // request that would fail with a 401 the person cannot act on.
        let no_key = Provider::parse("brave", "", None).unwrap_err().to_string();
        assert!(no_key.contains("API_KEY_BRAVE"), "{no_key}");
        assert!(no_key.contains("brave_api_key"), "{no_key}");
        let blank_key = Provider::parse("tavily", "", Some("   ".to_string()))
            .unwrap_err()
            .to_string();
        assert!(blank_key.contains("API_KEY_TAVILY"), "{blank_key}");
        assert_eq!(
            Provider::parse("tavily", "", Some("tvly-not-a-real-key".to_string())).unwrap(),
            Provider::Tavily {
                key: "tvly-not-a-real-key".to_string()
            }
        );
        let unknown = Provider::parse("google", "", None).unwrap_err().to_string();
        assert!(
            unknown.contains("`google` is not a search provider"),
            "{unknown}"
        );
    }

    #[tokio::test]
    async fn no_search_and_an_empty_question_are_answered_without_a_request() {
        // `none` is not "zero results": a model told there were no results goes
        // looking again, and it is the configuration that is missing.
        let none = search(&Provider::None, "rust ownership", 5)
            .await
            .unwrap_err()
            .to_string();
        assert!(none.contains("no search provider is configured"), "{none}");
        assert!(
            matches!(
                search(&Provider::Wikipedia, "   ", 5).await,
                Err(SearchError::EmptyQuery)
            ),
            "an empty query reaches the network"
        );
        let long = "a".repeat(MAX_QUERY_CHARS + 1);
        assert!(matches!(
            search(&Provider::Wikipedia, &long, 5).await,
            Err(SearchError::QueryTooLong(_))
        ));
    }

    /// The response Wikipedia's API actually sent, captured from
    /// `en.wikipedia.org/w/api.php?action=query&list=search&srlimit=3&format=json`
    /// on 2026-10-04 (HTTP 200, `content-type: application/json; charset=utf-8`,
    /// 1261 bytes), trimmed to its first row. The markup in `snippet` and the
    /// `&#039;` entity are the parts worth testing: they are what a search-match
    /// highlight looks like on the wire.
    const WIKIPEDIA_LIVE_ANSWER: &str = r#"{"query":{"searchinfo":{"totalhits":1402},"search":[{"ns":0,"title":"Substructural type system","pageid":14554100,"size":13207,"wordcount":1362,"snippet":"that transfer <span class=\"searchmatch\">ownership</span> – moving, where <span class=\"searchmatch\">ownership</span> is the responsibility to free the resource. Uses that don&#039;t transfer <span class=\"searchmatch\">ownership</span> – <span class=\"searchmatch\">borrowing</span> – are not in"}]}}"#;

    #[test]
    fn a_real_wikipedia_answer_becomes_a_title_a_link_and_plain_text() {
        let value: serde_json::Value = serde_json::from_str(WIKIPEDIA_LIVE_ANSWER).unwrap();
        let hits = wikipedia_hits(&value);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Substructural type system");
        assert_eq!(
            hits[0].url,
            "https://en.wikipedia.org/wiki/Substructural_type_system"
        );
        assert!(
            !hits[0].snippet.contains("searchmatch"),
            "the highlight markup was handed to the model: {}",
            hits[0].snippet
        );
        assert!(
            hits[0]
                .snippet
                .contains("ownership is the responsibility to free"),
            "{}",
            hits[0].snippet
        );
        // `&#039;` is an apostrophe in the document, and an entity left in a
        // snippet is a string the model quotes wrong.
        assert!(!hits[0].snippet.contains("&#039;"), "{}", hits[0].snippet);
    }

    #[test]
    fn a_title_that_is_not_a_plain_path_still_becomes_one_address() {
        // Percent-encoding the underscored title is what keeps `C++` and `A / B`
        // from arriving as two path segments or a broken link.
        let value: serde_json::Value =
            serde_json::from_str(r#"{"query":{"search":[{"title":"C++/CLI","pageid":1}]}}"#)
                .unwrap();
        assert_eq!(
            wikipedia_hits(&value)[0].url,
            "https://en.wikipedia.org/wiki/C%2B%2B%2FCLI"
        );
    }

    #[test]
    fn a_query_with_spaces_is_encoded_for_the_api_that_read_it() {
        assert_eq!(percent_encode("rust ownership"), "rust%20ownership");
        assert_eq!(percent_encode("borrow&move"), "borrow%26move");
        assert_eq!(percent_encode("café"), "caf%C3%A9");
    }

    /// A SearXNG instance answering its own documented JSON, on a real socket.
    /// This is the one networked provider whose address the person chooses, so it
    /// is the one that can be exercised end to end here — and the reason the
    /// public instances are not the default: measured the same day,
    /// `searx.be/search?format=json` answers **200 with `<!doctype html>`** and
    /// `paulgo.io` answers **429**.
    async fn serve_searxng(
        body: &'static str,
        content_type: &'static str,
    ) -> (String, std::sync::Arc<std::sync::Mutex<String>>) {
        use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let received = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let got = received.clone();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (read_half, mut write_half) = stream.into_split();
            let mut lines = BufReader::new(read_half).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.is_empty() {
                    break;
                }
                got.lock().unwrap().push_str(&line);
                got.lock().unwrap().push('\n');
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = write_half.write_all(response.as_bytes()).await;
            let _ = write_half.flush().await;
        });
        (format!("http://{addr}"), received)
    }

    #[tokio::test]
    async fn an_instance_of_ones_own_is_queried_over_a_real_socket() {
        let body = r#"{"query":"rust ownership","results":[{"title":"Ownership - The Rust Programming Language","url":"https://doc.example.org/book/ch04-01-what-is-ownership.html","content":"In Rust, ownership means each value has one <b>owner</b>…","engine":"duckduckgo"},{"title":"Second result","url":"https://elsewhere.example.org/a","content":null,"engine":"wikipedia"}]}"#;
        let (base, received) = serve_searxng(body, "application/json").await;
        let provider = Provider::parse("searxng", &base, None).unwrap();
        let hits = search(&provider, "rust ownership", 5).await.unwrap();
        let asked = received.lock().unwrap().clone();
        // The instance has to be asked for JSON, and for the words as one query:
        // a provider that silently sent `+` for a space is asking a different
        // question.
        assert!(
            asked.contains("GET /search?q=rust%20ownership&format=json"),
            "asked for: {asked}"
        );
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!(
            hits[0].url,
            "https://doc.example.org/book/ch04-01-what-is-ownership.html"
        );
        assert!(
            !hits[0].snippet.contains("<b>"),
            "markup survived: {}",
            hits[0].snippet
        );
        // A result with no snippet is still a result — the title and the address
        // are what the next call needs.
        assert_eq!(hits[1].snippet, "");
    }

    #[tokio::test]
    async fn an_instance_that_answers_html_instead_of_json_is_said_to_have() {
        // This is exactly what a public instance does to a `format=json` request,
        // and it must not read as "the search returned nothing".
        let (base, _) = serve_searxng(
            "<!doctype html><html><body>results</body></html>",
            "text/html",
        )
        .await;
        let provider = Provider::parse("searxng", &base, None).unwrap();
        let err = search(&provider, "rust ownership", 5).await.unwrap_err();
        let text = err.to_string();
        assert!(text.contains("not the JSON its API returns"), "{text}");
        assert!(text.contains("text/html"), "{text}");
    }

    #[tokio::test]
    async fn a_search_address_cannot_be_pointed_at_this_machine() {
        // The instance URL is typed once into a config file, and a host that later
        // resolves inward is refused the same way a fetched page is.
        let err = search(
            &Provider::Searxng {
                base: "http://169.254.169.254".to_string(),
            },
            "rust ownership",
            5,
        )
        .await
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("cannot search"), "{text}");
        assert!(text.contains("link-local"), "{text}");
    }

    #[test]
    fn a_brave_or_tavily_answer_is_read_from_its_own_shape() {
        // `web.results[]` for Brave and a top-level `results[]` for Tavily, with
        // the field each one names for the snippet.
        let brave: serde_json::Value = serde_json::from_str(
            r#"{"web":{"results":[{"title":"What is ownership?","url":"https://a.example/x","description":"The <em>owner</em> of a value frees it."},{"title":"No description","url":"https://b.example/y"}]}}"#,
        )
        .unwrap();
        let hits = brave_hits(&brave);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url, "https://a.example/x");
        assert!(
            !hits[0].snippet.contains("<em>"),
            "markup survived: {}",
            hits[0].snippet
        );
        assert_eq!(hits[1].snippet, "");
        // A Brave error body has no `web` in it, and reads as no hits rather than
        // as a panic in the parser.
        let error_body: serde_json::Value =
            serde_json::from_str(r#"{"error":{"code":"VALIDATION"}}"#).unwrap();
        assert!(brave_hits(&error_body).is_empty());

        let tavily: serde_json::Value = serde_json::from_str(
            r#"{"results":[{"title":"Ownership","url":"https://c.example/z","content":"Each value has one owner at a time.","score":0.81}]}"#,
        )
        .unwrap();
        let hits = tavily_hits(&tavily);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].snippet, "Each value has one owner at a time.");
    }

    #[tokio::test]
    async fn a_provider_that_answers_with_an_error_status_says_the_status() {
        // A 429 or a 401 must not be quoted back as a page — an error document can
        // carry anything, and this answer goes into a model's context. Measured
        // live: `paulgo.io/search?format=json` answers **429** and
        // `api.tavily.com/search` without a key answers **401**.
        use tokio::io::AsyncWriteExt as _;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = stream
                .write_all(
                    b"HTTP/1.1 429 Too Many Requests\r\ncontent-type: text/plain\r\n\
                      content-length: 0\r\nconnection: close\r\n\r\n",
                )
                .await;
        });
        let err = search(
            &Provider::Searxng {
                base: format!("http://{addr}"),
            },
            "rust ownership",
            5,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, SearchError::Status(429)),
            "a rate limit read as something else: {err}"
        );
    }

    /// The keyless provider, asked for real. Ignored because it reaches out — run
    /// it with `cargo test -p xencode-analysis-rs -- --ignored` on a machine that
    /// means to check. This is the one option that survived the sweep that killed
    /// the others: DuckDuckGo's keyless endpoints answer a CAPTCHA page or **410
    /// Gone**, and public SearXNG instances answer `format=json` with HTML.
    #[tokio::test]
    #[ignore]
    async fn wikipedia_answers_a_real_question_with_no_key_at_all() {
        let hits = search(&Provider::Wikipedia, "rust ownership", 3)
            .await
            .expect("Wikipedia's search API answered");
        assert!(!hits.is_empty(), "the API returned no rows at all");
        for hit in &hits {
            assert!(!hit.title.is_empty(), "{hit:?}");
            assert!(
                hit.url.starts_with("https://en.wikipedia.org/wiki/"),
                "{hit:?}"
            );
            assert!(!hit.snippet.contains('<'), "{hit:?}");
        }
    }
}
