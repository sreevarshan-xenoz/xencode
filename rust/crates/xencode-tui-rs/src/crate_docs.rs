//! `read_docs` — one dependency's own documentation, at a stated version.
//!
//! RS-3 made a locked crate's *source* readable and left the choosing to the
//! model. Choosing is the weakest step for a small one: it has to know that
//! `serde` documents itself in `README.md` and not in `Cargo.toml`, that
//! `docs/` exists in some crates and not others, and that `crates.io` is
//! where an unpinned version's readme lives. This module does that part: it
//! decides which file is the crate's documentation, in a fixed order, and it
//! says which version the bytes came from.
//!
//! Everything starts locally, because the local copy is the version this
//! project actually builds. The network is a fallback for the two cases where
//! there is no local copy at all — a crate that is pinned but never unpacked,
//! or a version asked for by hand — and it is opt-in (`allow_online_docs`).
//!
//! The two endpoints were checked by calling them, not by reading about them:
//!   - `crates.io/api/v1/crates/<name>/<version>/readme` answers **302** to
//!     `static.crates.io/readmes/<name>/<name>-<version>.html`, and what comes
//!     back is a rendered-markdown *fragment* (serde 1.0.229: 3,510 bytes,
//!     starting at `<p>`, no `<html>` wrapper). Without a version the same
//!     endpoint answers **400** — it cannot be asked for "the latest" — so a
//!     version is required here too. A version that does not exist still
//!     answers 302, and the object store then answers **403**.
//!   - `docs.rs/crate/<name>/<version>/source/<path>` is a whole web page:
//!     49,883 bytes of highlighted HTML for `serde` 1.0.229's `Cargo.toml`,
//!     whose text is 1,969 bytes. The file is the one `<pre><code>` block
//!     inside `<div id="source-code">`, so it is recoverable, and the line
//!     numbers sit in a sibling `<pre id="line-numbers">` that must not be read
//!     as code; a version or path that does not exist returns 404 (6,647 bytes)
//!     with no such block, which is how "not found" is told apart from an empty
//!     file.

use std::path::{Path, PathBuf};

use crate::crate_sources::{self, CrateSource};

/// How much of a document the model is handed per call. Head, not tail: a
/// readme answers "what is this" in its first screen, and the rest is
/// addressable through `read_file` with a `crate:` address and an offset.
pub const DOC_BYTE_CAP: usize = 8 * 1024;

/// How many other documentation files one answer names.
pub const DOC_LIST_CAP: usize = 12;

/// A documentation read is worth waiting for but not worth stalling a turn.
pub const FETCH_TIMEOUT_SECS: u64 = 15;

/// Both services ask for something identifying; a bare client gets a 403 from
/// crates.io's API for some routes.
pub const USER_AGENT: &str = "xencode-read-docs (+https://github.com/xencode)";

/// The names tried, in order, when the crate's own `Cargo.toml` does not say
/// which file is its readme. Case matters here: a crate with both `README.md`
/// and `readme.md` is unpacked on a case-sensitive filesystem and the first
/// one is the real file.
const README_NAMES: &[&str] = &[
    "README.md",
    "README.markdown",
    "README.mdx",
    "README.rst",
    "README.txt",
    "README",
    "readme.md",
    "Readme.md",
];

/// The file a crate documents itself with, and where it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocFile {
    /// Crate-relative, with forward slashes — what to pass back as `path`.
    pub rel: String,
    pub full: PathBuf,
}

/// The local copy of a crate, and how it relates to this project's lock file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalCopy {
    pub source: CrateSource,
    /// Whether `Cargo.lock` pins exactly the version that was found.
    pub pinned: bool,
    /// What the lock does pin for this name, empty when it pins nothing.
    pub lock_versions: Vec<String>,
}

impl LocalCopy {
    /// The line that opens every local answer. A version is never implicit:
    /// an answer quoted from a version this project does not build has to
    /// carry that fact with it.
    pub fn label(&self) -> String {
        if self.pinned {
            return self.source.label();
        }
        let pinned = if self.lock_versions.is_empty() {
            "does not name it".to_string()
        } else {
            format!("pins {}", self.lock_versions.join(", "))
        };
        format!(
            "{} {} — read from cargo's local copy, which is not what this project's \
             Cargo.lock {}",
            self.source.name, self.source.version, pinned
        )
    }
}

/// Whether the local copy could be read, and if not, why in one sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Local {
    Found(LocalCopy),
    Absent(String),
}

impl Local {
    /// The reason this copy is not on disk, when it is not.
    pub fn absent(&self) -> Option<&str> {
        match self {
            Local::Found(_) => None,
            Local::Absent(why) => Some(why),
        }
    }
}

/// Which characters a crates.io package name may contain. Registry names are
/// already restricted to this set; the check exists because the name is put
/// into a URL, and a model-supplied string must not become a path or a host.
fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+'))
}

/// Versions carry `+build` metadata (`1.0.0+3.2.1`), which is legal in a URL
/// path segment and means exactly itself.
fn valid_version(version: &str) -> bool {
    !version.is_empty()
        && version.len() <= 32
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+'))
}

/// Where crates.io keeps a version's rendered readme.
pub fn readme_url(name: &str, version: &str) -> Result<String, String> {
    if !valid_name(name) {
        return Err(format!(
            "{name:?} is not a crate name this tool can look up"
        ));
    }
    if !valid_version(version) {
        return Err(format!(
            "{version:?} is not a version this tool can look up"
        ));
    }
    Ok(format!(
        "https://crates.io/api/v1/crates/{name}/{version}/readme"
    ))
}

/// Where docs.rs keeps one file of a version's source, as a web page.
pub fn source_url(name: &str, version: &str, path: &str) -> Result<String, String> {
    if !valid_name(name) {
        return Err(format!(
            "{name:?} is not a crate name this tool can look up"
        ));
    }
    if !valid_version(version) {
        return Err(format!(
            "{version:?} is not a version this tool can look up"
        ));
    }
    if path.is_empty() || !crate_sources::path_is_contained(path) {
        return Err(format!(
            "{path:?} is not a file inside a crate: keep it relative and inside the package"
        ));
    }
    Ok(format!(
        "https://docs.rs/crate/{name}/{version}/source/{path}"
    ))
}

/// The versions `Cargo.lock` pins for `name` in the project at `root`, which
/// is what an answer decides between "pass a version" and "run `cargo fetch`".
pub fn locked_versions(root: &Path, name: &str) -> Vec<String> {
    crate_sources::locked_packages_for(root)
        .into_iter()
        .filter(|(locked, _)| locked == name)
        .map(|(_, version)| version)
        .collect()
}

/// The local copy of `name`, preferring `version` when one is given.
///
/// With no version this is RS-3's rule unchanged: the version comes from the
/// lock file, and a lock that pins two of them is reported rather than
/// guessed at. With a version given, that version is read if cargo has
/// unpacked it anywhere — which is allowed but labelled, because the model
/// asked for a version this project may not build.
pub fn find_local(root: &Path, name: &str, version: Option<&str>) -> Local {
    if !valid_name(name) {
        return Local::Absent(format!(
            "{name:?} is not a crate name this tool can look up"
        ));
    }
    let locked = locked_versions(root, name);
    let dirs = match crate_sources::cargo_home() {
        Some(home) => crate_sources::registry_src_dirs(&home),
        None => {
            return Local::Absent(
                "cannot find cargo's home directory, so no unpacked crate source can be \
                 located on this machine"
                    .to_string(),
            )
        }
    };
    if let Some(version) = version {
        if !valid_version(version) {
            return Local::Absent(format!(
                "{version:?} is not a version this tool can look up"
            ));
        }
        let dir_name = format!("{name}-{version}");
        let found = dirs.iter().map(|d| d.join(&dir_name)).find(|d| d.is_dir());
        return match found {
            Some(dir) => Local::Found(LocalCopy {
                pinned: locked.iter().any(|locked| locked == version),
                lock_versions: locked,
                source: CrateSource {
                    name: name.to_string(),
                    version: version.to_string(),
                    dir,
                },
            }),
            None => Local::Absent(format!(
                "cargo has not unpacked {name} {version} on this machine"
            )),
        };
    }
    if locked.is_empty() {
        return Local::Absent(format!(
            "this project's Cargo.lock does not name {name}, so there is no version of it to \
             read from here"
        ));
    }
    let pairs: Vec<(String, String)> = locked
        .iter()
        .map(|version| (name.to_string(), version.clone()))
        .collect();
    match crate_sources::resolve_crate_in(&dirs, &pairs, name, "") {
        Ok((_, source)) => Local::Found(LocalCopy {
            pinned: true,
            lock_versions: locked,
            source,
        }),
        Err(why) => Local::Absent(format!(
            "{why}; read_docs can also take an explicit version"
        )),
    }
}

/// The `readme = "…"` target a crate's own manifest names, if it names one
/// that exists. `cargo package` refuses a manifest readme that is not in the
/// published file list, so inside an unpacked crate this is the answer.
fn manifest_readme(dir: &Path) -> Option<DocFile> {
    let text = std::fs::read_to_string(dir.join("Cargo.toml")).ok()?;
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("readme") else {
            continue;
        };
        let Some(rest) = rest.trim().strip_prefix('=') else {
            continue;
        };
        let Some(value) = rest.trim().strip_prefix('"').and_then(|v| {
            let end = v.find('"')?;
            Some(v[..end].to_string())
        }) else {
            continue;
        };
        if value.is_empty() || value == "false" || !crate_sources::path_is_contained(&value) {
            continue;
        }
        let full = dir.join(&value);
        if full.is_file() {
            return Some(DocFile {
                rel: value.replace('\\', "/"),
                full,
            });
        }
    }
    None
}

/// Which file documents the crate unpacked at `dir`, in a fixed order: the
/// manifest's own `readme` key, then the conventional names.
pub fn pick_doc_file(dir: &Path) -> Option<DocFile> {
    if let Some(from_manifest) = manifest_readme(dir) {
        return Some(from_manifest);
    }
    // Compare against the names actually in the directory. Asking the disk
    // whether `README.md` exists is answered "yes" for `readme.md` on Windows
    // and macOS, which would report a name the crate does not have.
    let present: std::collections::HashSet<std::ffi::OsString> = std::fs::read_dir(dir)
        .map(|entries| entries.flatten().map(|e| e.file_name()).collect())
        .unwrap_or_default();
    for name in README_NAMES {
        let full = dir.join(name);
        if present.contains(std::ffi::OsStr::new(name)) && full.is_file() {
            return Some(DocFile {
                rel: name.to_string(),
                full,
            });
        }
    }
    None
}

/// The other files in the crate that read like documentation: markdown and
/// text at the top level and one directory down, excluding the one just
/// returned. Named so the next call can ask for one of them instead of
/// searching.
pub fn other_doc_files(dir: &Path, except: &str) -> Vec<String> {
    let mut listed: Vec<String> = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return listed;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir {
            // One level down: `docs/` and `guide/` hold what a readme links
            // to, and a crate's own top-level file list is not the whole story.
            let Ok(nested) = std::fs::read_dir(&path) else {
                continue;
            };
            for sub in nested.flatten() {
                let full = sub.path();
                if !full.is_file() || !is_doc_name(&full) {
                    continue;
                }
                if let Some(rel) = relative(dir, &full) {
                    listed.push(rel);
                }
            }
            continue;
        }
        if !is_doc_name(&path) {
            continue;
        }
        if let Some(rel) = relative(dir, &path) {
            listed.push(rel);
        }
    }
    listed.retain(|rel| rel != except);
    listed.sort();
    listed.dedup();
    if listed.len() > DOC_LIST_CAP {
        listed.truncate(DOC_LIST_CAP);
        listed.push("…".to_string());
    }
    listed
}

fn relative(dir: &Path, full: &Path) -> Option<String> {
    Some(
        full.strip_prefix(dir)
            .ok()?
            .to_string_lossy()
            .replace('\\', "/"),
    )
}

fn is_doc_name(path: &Path) -> bool {
    let Some(name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".md")
        || lower.ends_with(".markdown")
        || lower.ends_with(".rst")
        || lower.starts_with("readme")
        || lower.starts_with("changelog")
        || lower.starts_with("license")
        || lower.starts_with("contributing")
}

/// Cut a document to the bytes one answer carries, keeping the head, and say
/// in `how_to_get_the_rest` — the caller's words, because only the caller
/// knows whether the remainder is a page away or a URL — where the rest is.
pub fn cap_doc(text: &str, how_to_get_the_rest: &str) -> String {
    if text.len() <= DOC_BYTE_CAP {
        return text.to_string();
    }
    let mut end = DOC_BYTE_CAP;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = text[..end].to_string();
    // Drop a partial last line so the cut never lands mid-word.
    if let Some(nl) = out.rfind('\n') {
        out.truncate(nl);
    }
    out.push_str(&format!(
        "\n… cut to the first {} bytes of {} — {how_to_get_the_rest}\n",
        DOC_BYTE_CAP,
        text.len()
    ));
    out
}

/// Unescape the entities both services emit in text. `&amp;` last, so an
/// `&lt;` written as `&amp;lt;` in the source stays visible as `&lt;`.
fn unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
}

/// Collapse runs of blank lines and trailing space, so a page of markup does
/// not turn into a page of whitespace.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0usize;
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    while out.ends_with("\n\n") {
        out.pop();
    }
    out
}

/// Turn the HTML crates.io renders a readme into, or a docs.rs source page,
/// back into text. Block ends become line breaks, list items get a dash, and
/// a link keeps both its label and its address — a readme that says "see the
/// examples" is worth nothing if the URL is dropped.
pub fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2 + 16);
    let mut rest = html;
    let mut anchor_at: Option<usize> = None;
    let mut href = String::new();
    loop {
        let Some(lt) = rest.find('<') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..lt]);
        let Some(gt) = rest[lt..].find('>') else {
            out.push_str(&rest[lt..]);
            break;
        };
        let tag = &rest[lt + 1..lt + gt];
        rest = &rest[lt + gt + 1..];
        let closing = tag.starts_with('/');
        let body = tag.trim_start_matches('/').trim_start();
        let name: String = body
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        match name.as_str() {
            "a" if !closing => {
                anchor_at = Some(out.len());
                href = attribute(body, "href").unwrap_or_default();
            }
            "a" if closing => {
                if let Some(at) = anchor_at.take() {
                    let label = out[at..].trim();
                    if !href.is_empty() && !label.is_empty() && !label.ends_with(&href) {
                        out.push_str(" (");
                        out.push_str(&href);
                        out.push(')');
                    }
                }
                href.clear();
            }
            "li" if !closing => {
                end_block(&mut out);
                out.push_str("- ");
            }
            "br" => out.push('\n'),
            "p" | "div" | "ul" | "ol" | "table" | "tr" | "blockquote" | "section" | "article"
            | "header" | "footer" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "pre" => {
                end_block(&mut out);
            }
            _ => {}
        }
    }
    tidy(&unescape(&out))
}

/// A block-level tag ends a line of prose, so the text that follows it starts
/// on a fresh line instead of running into the previous paragraph.
fn end_block(out: &mut String) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
}

/// A `name="value"` attribute from the inside of a start tag.
fn attribute(tag: &str, key: &str) -> Option<String> {
    let (_, rest) = tag.split_once(key)?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let quote = rest.chars().next().filter(|c| matches!(c, '"' | '\''))?;
    let inner = &rest[quote.len_utf8()..];
    let end = inner.find(quote)?;
    Some(unescape(&inner[..end]))
}

/// The one `<pre><code>` block inside docs.rs's `<div id="source-code">`, as
/// text. `None` is how "that version or file does not exist" arrives: a 404
/// page carries no such block, so it cannot be mistaken for an empty file.
pub fn docs_rs_source(page: &str) -> Option<String> {
    let start = page.find("id=\"source-code\"")?;
    let rest = &page[start..];
    let code = rest.find("<pre>")? + "<pre>".len();
    let body = &rest[code..];
    let end = body.find("</pre>")?;
    let inner = &body[..end];
    // The highlighter wraps every token in a span and leaves the file's own
    // newlines literal, so dropping the tags restores the source.
    let stripped = strip_tags(inner);
    Some(tidy(&unescape(&stripped)))
}

fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        out.push_str(&rest[..lt]);
        match rest[lt..].find('>') {
            Some(gt) => rest = &rest[lt + gt + 1..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Fetch one of the two documented URLs. The status is returned with the
/// body: crates.io answers a version that does not exist with a redirect to
/// an object store that refuses it, so a 403 here means "no such readme" and
/// must not be reported as a network failure.
pub async fn fetch(url: &str) -> Result<(u16, String), String> {
    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .build()
        .map_err(|e| format!("cannot set up the request: {e}"))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("{url}: {e}"))?;
    let status = response.status().as_u16();
    let body = response.text().await.map_err(|e| format!("{url}: {e}"))?;
    Ok((status, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xencode-docs-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A scratch registry holding one unpacked crate with the files a
    /// published package really has. Returns the crate directory and the
    /// scratch root to delete afterwards — never a parent of it.
    fn unpacked(name: &str, version: &str, files: &[(&str, &str)]) -> (PathBuf, PathBuf) {
        let root = scratch(&format!("{name}-{version}"));
        let dir = root.join(format!("{name}-{version}"));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            format!("[package]\nname = \"{name}\"\nversion = \"{version}\"\n"),
        )
        .unwrap();
        std::fs::write(dir.join("src/lib.rs"), "fn f() {}\n").unwrap();
        for (path, text) in files {
            let full = dir.join(path);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(full, text).unwrap();
        }
        (dir, root)
    }

    #[test]
    fn the_manifest_decides_which_file_is_the_readme() {
        let (dir, root) = unpacked(
            "twodocs",
            "1.0.0",
            &[
                ("README.md", "# not the readme\n"),
                ("docs/readme.markdown", "# the real one\n"),
            ],
        );
        let mut manifest = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap();
        manifest.push_str("readme = \"docs/readme.markdown\"\n");
        std::fs::write(dir.join("Cargo.toml"), manifest).unwrap();

        let picked = pick_doc_file(&dir).expect("manifest readme");
        assert_eq!(picked.rel, "docs/readme.markdown");
        assert!(picked.full.is_file());
        // A `readme` key that is a boolean, or that points out of the crate,
        // is not a file to read.
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"twodocs\"\nversion = \"1.0.0\"\nreadme = false\n",
        )
        .unwrap();
        assert_eq!(
            pick_doc_file(&dir).expect("fallback").rel,
            "README.md",
            "a manifest that names no file falls back to the conventional name"
        );
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"twodocs\"\nversion = \"1.0.0\"\nreadme = \"../../../etc/passwd\"\n",
        )
        .unwrap();
        assert_eq!(
            pick_doc_file(&dir).expect("refused escape").rel,
            "README.md"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_candidate_order_is_fixed_and_case_sensitive() {
        let (dir, root) = unpacked(
            "cased",
            "1.0.0",
            &[("readme.md", "lower\n"), ("README.rst", "rst\n")],
        );
        assert_eq!(pick_doc_file(&dir).expect("one").rel, "README.rst");
        std::fs::remove_dir_all(&root).ok();

        let (none, root) = unpacked("nodoc", "1.0.0", &[]);
        assert!(pick_doc_file(&none).is_none());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn other_documentation_is_named_so_the_next_call_can_ask_for_it() {
        let (dir, root) = unpacked(
            "manydocs",
            "1.0.0",
            &[
                ("README.md", "# top\n"),
                ("CHANGELOG.md", "# changes\n"),
                ("LICENSE-APACHE", "apache\n"),
                ("docs/guide.md", "# guide\n"),
                ("src/lib.rs", "fn f() {}\n"),
            ],
        );
        let others = other_doc_files(&dir, "README.md");
        assert!(others.contains(&"CHANGELOG.md".to_string()), "{others:?}");
        assert!(
            others.contains(&"docs/guide.md".to_string()),
            "one level down counts: {others:?}"
        );
        assert!(others.contains(&"LICENSE-APACHE".to_string()), "{others:?}");
        assert!(!others.contains(&"README.md".to_string()), "{others:?}");
        assert!(!others.iter().any(|p| p.ends_with("lib.rs")), "{others:?}");
        assert!(
            !others.iter().any(|p| p.ends_with("Cargo.toml")),
            "{others:?}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_long_document_is_cut_at_the_front_and_says_so() {
        let text = "line\n".repeat(4000); // 20 KiB
        let cut = cap_doc(
            &text,
            "read_file with path=\"crate:big/README.md\" pages the rest",
        );
        assert!(cut.len() < text.len(), "{}", cut.len());
        assert!(cut.starts_with("line\nline\n"), "{}", &cut[..20]);
        assert!(
            cut.contains("cut to the first 8192 bytes of 20000"),
            "{cut}"
        );
        assert!(
            cut.contains("crate:big/README.md"),
            "must say how to get the rest: {cut}"
        );
        // An exact fit is handed over untouched.
        let small = "hello\n".to_string();
        assert_eq!(cap_doc(&small, "nowhere"), small);
    }

    #[test]
    fn a_rendered_readme_comes_back_as_text_that_keeps_its_links() {
        // Shape taken from crates.io's own fragment for serde 1.0.229.
        let html = "<p><strong>Serde</strong> is a framework.</p>\n<ul>\n\
                    <li><a href=\"https://serde.rs\" rel=\"nofollow\">An overview</a></li>\n\
                    </ul>\n<p>Use <code>Vec&lt;u8&gt;</code> &amp; friends.</p>";
        let text = html_to_text(html);
        assert!(text.starts_with("Serde is a framework."), "{text}");
        assert!(text.contains("- An overview (https://serde.rs)"), "{text}");
        assert!(text.contains("Vec<u8> & friends"), "{text}");
        assert!(!text.contains("nofollow"), "{text}");
        assert!(!text.contains("<strong>"), "{text}");
    }

    #[test]
    fn a_docs_rs_page_yields_the_file_and_a_404_yields_nothing() {
        let page = "<html><head><style>x{}</style></head><body>\
                    <pre id=\"line-numbers\"><code><a>1</a>\n2\n</code></pre>\
                    <div id=\"source-code\" class=\"source-code\"><pre><code>\
                    <span class=\"syntax-text\"># Title</span>\
                    <span class=\"n\">\n</span>\
                    <span>body &amp; more &lt;x&gt;</span>\
                    </code></pre></div></body></html>";
        let file = docs_rs_source(page).expect("the source block");
        assert!(file.starts_with("# Title\n"), "{file:?}");
        assert!(file.contains("body & more <x>"), "{file:?}");
        assert!(!file.contains("syntax-text"), "{file:?}");
        assert!(!file.contains('1'), "line numbers stay out: {file:?}");
        assert_eq!(
            docs_rs_source("<html><head><title>404 Not Found</title></head>"),
            None
        );
    }

    #[test]
    fn the_two_endpoints_are_built_from_a_stated_version_only() {
        assert_eq!(
            readme_url("serde", "1.0.229").unwrap(),
            "https://crates.io/api/v1/crates/serde/1.0.229/readme"
        );
        assert_eq!(
            source_url("serde", "1.0.229", "docs/guide.md").unwrap(),
            "https://docs.rs/crate/serde/1.0.229/source/docs/guide.md"
        );
        // A build-metadata version is legal and stays as written.
        assert!(readme_url("llvm-sys", "18.1.0+llvm-18.1.0")
            .unwrap()
            .ends_with("/llvm-sys/18.1.0+llvm-18.1.0/readme"));
        for bad in [
            readme_url("serde/../evil", "1.0.0"),
            readme_url("serde", ""),
            readme_url("serde", "1.0.0 2"),
            source_url("serde", "1.0.0", "../../etc/passwd"),
            source_url("serde", "1.0.0", "/etc/passwd"),
            source_url("serde", "1.0.0", ""),
        ] {
            assert!(bad.is_err(), "{bad:?}");
        }
    }

    #[test]
    fn an_unpinned_local_copy_says_which_version_the_lock_wanted() {
        let source = CrateSource {
            name: "serde".to_string(),
            version: "1.0.219".to_string(),
            dir: PathBuf::from("/tmp/serde-1.0.219"),
        };
        let copy = LocalCopy {
            source: source.clone(),
            pinned: true,
            lock_versions: vec!["1.0.219".to_string()],
        };
        assert_eq!(
            copy.label(),
            "serde 1.0.219 — the version this project's Cargo.lock pins"
        );
        let other = LocalCopy {
            pinned: false,
            lock_versions: vec!["1.0.229".to_string()],
            ..copy.clone()
        };
        assert_eq!(
            other.label(),
            "serde 1.0.219 — read from cargo's local copy, which is not what this project's \
             Cargo.lock pins 1.0.229"
        );
        let absent = LocalCopy {
            pinned: false,
            lock_versions: vec![],
            ..copy
        };
        assert!(
            absent.label().contains("Cargo.lock does not name it"),
            "{}",
            absent.label()
        );
    }

    #[test]
    fn a_crate_with_no_local_copy_reports_why_in_one_line() {
        // No Cargo.lock in this scratch project, and no version offered.
        let root = scratch("nolock");
        let local = find_local(&root, "serde", None);
        let why = local.absent().expect("absent").to_string();
        assert!(why.contains("does not name serde"), "{why}");
        std::fs::remove_dir_all(&root).ok();

        // A version that is not on disk: named exactly, rather than guessed
        // at from whatever else of that crate is unpacked.
        let root = scratch("nolock2");
        let local = find_local(&root, "serde", Some("9.9.9"));
        let why = local.absent().expect("absent").to_string();
        assert!(why.contains("serde 9.9.9"), "{why}");
        assert!(why.contains("not unpacked"), "{why}");
        std::fs::remove_dir_all(&root).ok();
    }
}
