//! Per-turn agent trace (§Milestone W1, EV-2) — appended to `cache/turns.jsonl`.
//!
//! `metrics.jsonl` answers "how full was the window"; this answers "what did
//! the agent actually do on that turn, and what did it cost". One row per
//! turn, written when the loop finishes, beside the metrics rows it belongs to.
//!
//! The deliberate omissions matter more than the fields:
//!   - **No prompt text.** A prompt is the user's own words and often contains
//!     the credential or client detail they were asking about. What is
//!     recorded is a digest ([`prompt_digest`]), enough to tell two turns
//!     apart and to spot a repeated prompt, not enough to read back.
//!   - **Tool arguments, but only as far as they explain the call.** They carry
//!     file paths and command lines, which is exactly what a wrong tool choice
//!     is diagnosed from, so they are recorded. What is *not* recorded is the
//!     payload a call carries rather than points at — the bytes of a file being
//!     written, the text to replace, the plan itself — which stands in for its
//!     size. Everything that survives goes through [`redact_secrets`] and a
//!     length cap, as the output does.
//!   - **Tool output only as a short, redacted tail.** It is the field that
//!     genuinely cannot be sanitised by omission — a `cat` of a secrets file is
//!     exactly what you want to see and exactly what must not be kept — so it
//!     goes through [`redact_secrets`] before it is cut, and only the last few
//!     hundred characters survive. The redaction is pattern-based: it removes
//!     the shapes of credentials we know about, it does not prove a string is
//!     harmless.
//!   - **No invented numbers.** `prompt_tokens` and `est_cost_micros` stay
//!     `null` unless something measured them. Today llama.cpp reports generated
//!     tokens and nothing reports cost, so those are `null` on every row.

use crate::metrics::{MetricSource, MetricsIdentity};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

/// Characters of tool output kept per call, after redaction.
pub const TRACE_TAIL_CAP: usize = 300;

/// Characters of a tool's arguments kept per call, after bulk values have been
/// replaced by their size and the rest has been redacted.
pub const TRACE_ARGUMENTS_CAP: usize = 240;

/// What one tool call did on a turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolTrace {
    pub name: String,
    /// `done` / `denied` / `refused` / `failed`, as the agent loop reports it.
    pub outcome: String,
    /// The arguments the model chose, reduced to what explains the call:
    /// references kept, payloads replaced by their size, credentials removed,
    /// and the whole thing cut to [`TRACE_ARGUMENTS_CAP`]. `None` when the call
    /// took no arguments at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    /// Last [`TRACE_TAIL_CAP`] characters of the result, credentials removed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail: Option<String>,
}

/// Argument keys whose value *is* the payload of the call rather than a
/// reference to it. Their text is the conversation's content — a file body, a
/// search-and-replace pair, a list of steps — which is what the omissions above
/// exist to keep out; how big it was is enough to read the call.
const BULK_ARGUMENT_KEYS: &[&str] = &["content", "old", "new", "items"];

/// One row of `.xencode/cache/turns.jsonl`: everything the agent did between
/// the user pressing Enter and the answer finishing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TurnTrace {
    /// UTC epoch millis when the turn ended.
    pub ts_unix_ms: u64,
    /// Wall-clock milliseconds the loop ran for.
    pub duration_ms: u64,
    /// Requests the loop made, including the final tool-less one.
    pub rounds: u32,
    /// Tool calls executed, in the order they were asked for.
    #[serde(default)]
    pub tools: Vec<ToolTrace>,
    /// `true` when the loop stopped on a provider error instead of an answer.
    #[serde(default)]
    pub failed: bool,
    /// What the provider said when the loop stopped on an error, with anything
    /// shaped like a credential taken out. `null` for a turn that ended normally,
    /// and for one written before this field existed.
    #[serde(default)]
    pub error: Option<String>,
    /// Digest of the prompt that started the turn — never the prompt itself.
    #[serde(default)]
    pub prompt_sha256: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    /// Whether the prompt left this machine.
    #[serde(default)]
    pub source: Option<MetricSource>,
    /// Tokens the server reported generating, summed over the turn's requests.
    /// `null` when the provider reports no usage at all (Ollama does not).
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    /// Prompt tokens as counted locally by the context assembler. `null`
    /// until a provider reports them; a local estimate is not a measurement.
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    /// Money. `null` everywhere: nothing in this program prices a request yet,
    /// and an estimate in this column would be mistaken for one later.
    #[serde(default)]
    pub est_cost_micros: Option<u64>,
    /// The workspace files the context assembler put in front of the model for
    /// this turn, best-match first — the ones that survived its budget, not the
    /// candidates it trimmed. Their bodies are not stored, only where they came
    /// from, so a turn can be asked "what was it looking at" without the trace
    /// becoming a copy of the repository. Note that the identically named column
    /// on a `metrics.jsonl` row is a *count* of these, not the list.
    #[serde(default)]
    pub retrieved_files: Vec<String>,
    /// `true` when the prompt that started this turn carried the `[d]` decision
    /// marker, which is what makes a turn survive compaction (§11). Read off
    /// the user's own words, never from anything the model wrote about its
    /// reasoning: a model's explanation of why it chose something is a
    /// narrative, not the internals of the choice. A row written before this
    /// field existed reads as `false`, which means "not marked", not "declined
    /// to say".
    #[serde(default)]
    pub is_decision: bool,
    /// The digest of the instruction set this turn ran under, from the prompt
    /// registry. `prompt_sha256` above identifies the user's words; this
    /// identifies what the program told the model to do with them, so a turn from
    /// before a prompt edit and one after are not compared as if they were alike.
    #[serde(default)]
    pub prompt_version: Option<String>,
    /// What the checks run after this turn's edits found (EVd-3), when any ran.
    /// `None` when the turn edited nothing or no check applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checks: Option<ChecksVerdict>,
}

/// The checks-ran verdict (EVd-3), in JUnit's terms: a check that did not run
/// did not pass. `failed` is a subset of `ran`; `skipped` lists checks that were
/// due and never ran — a denial, a timeout, no toolchain, or an earlier
/// failure that stopped the run. Nothing here says "verified": an exit code
/// says the checks passed, not that the change is right.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChecksVerdict {
    pub ran: Vec<String>,
    pub failed: Vec<String>,
    pub skipped: Vec<String>,
    /// Where the output is kept: `tools[N]` names this turn's own tool row
    /// whose tail holds the last check's output.
    #[serde(default)]
    pub evidence_ref: Option<String>,
}

impl ChecksVerdict {
    /// Every check that was due ran, and none failed.
    pub fn all_passed(&self) -> bool {
        !self.ran.is_empty() && self.failed.is_empty() && self.skipped.is_empty()
    }

    /// One line for `/trace`: counts first, then the names that matter.
    pub fn summary(&self) -> String {
        let mut line = format!(
            "checks: {} ran, {} failed, {} skipped",
            self.ran.len(),
            self.failed.len(),
            self.skipped.len()
        );
        if !self.failed.is_empty() {
            line.push_str(&format!(" — failed: {}", self.failed.join(", ")));
        }
        if !self.skipped.is_empty() {
            line.push_str(&format!(" — not run: {}", self.skipped.join(", ")));
        }
        if let Some(evidence) = &self.evidence_ref {
            line.push_str(&format!(" (output: {evidence})"));
        }
        line
    }
}

impl TurnTrace {
    /// A row for a turn that just ended, with no tools and no identity yet.
    pub fn new(duration_ms: u64, rounds: u32) -> Self {
        Self {
            ts_unix_ms: crate::conversation::now_millis(),
            duration_ms,
            rounds,
            tools: Vec::new(),
            failed: false,
            error: None,
            prompt_sha256: None,
            session_id: None,
            model: None,
            provider: None,
            source: None,
            completion_tokens: None,
            prompt_tokens: None,
            est_cost_micros: None,
            retrieved_files: Vec::new(),
            is_decision: false,
            prompt_version: Some(crate::prompts::set_version().to_string()),
            checks: None,
        }
    }

    /// Stamp which conversation, model and route this turn belonged to.
    pub fn with_identity(mut self, identity: MetricsIdentity) -> Self {
        self.session_id = identity.session_id;
        self.model = identity.model;
        self.provider = identity.provider;
        self.source = identity.source;
        self
    }
}

/// First 16 hex characters of the SHA-256 of a prompt. Long enough to
/// distinguish the turns of a session, short enough to read in a TUI row; the
/// prompt itself is never stored.
pub fn prompt_digest(prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    let sum = hasher.finalize();
    let mut out = String::with_capacity(16);
    for byte in sum.iter().take(8) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

static PEM_KEY_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    pattern(r"(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----")
});
/// An assignment whose *name* looks like a credential, in code, in JSON or in
/// a shell line: `AWS_SECRET_ACCESS_KEY='…'`, `"api_key": "…"`, `password: …`.
/// The name is matched as one whole identifier and then checked for a
/// secret-shaped word, because `SECRET` inside such a name is not followed by
/// a word boundary and would escape a keyword-only pattern. The value may lead
/// with a scheme word so `Authorization: Bearer <jwt>` loses the token, not
/// just the word "Bearer".
static KEYED_VALUE_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    pattern(
        r#"(?i)\b([a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*["']?\s*[=:]\s*)("[^"]*"|'[^']*'|(?:[a-z][a-z0-9-]* )?[^\s,;]{3,})"#,
    )
});
static BEARER_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| pattern(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/-]{8,}"));
static PREFIXED_TOKEN_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    pattern(
        r"\b(?:sk|gh[supbor]|github_pat|glpat|hf|pt|xox[baprs])[-_][A-Za-z0-9_\-]{6,}|\bAIza[0-9A-Za-z_\-]{20,}|\bya29\.[A-Za-z0-9_\-]{6,}|\bAKIA[0-9A-Z]{12,}",
    )
});

/// Words that make an assigned name worth hiding. Deliberately over-broad:
/// redacting a harmless `key_count` costs a readable row, leaving a real key
/// costs a secret.
const SECRET_NAME_PARTS: &[&str] = &[
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "authorization",
    "credential",
    "apikey",
    "api_key",
    "api-key",
    "oauth",
    "key",
];

fn pattern(source: &str) -> regex::Regex {
    regex::Regex::new(source).expect("a redaction pattern that does not compile")
}

/// Replace anything shaped like a credential with `[redacted]`, keeping the
/// name it was assigned to so the row is still readable.
/// Longest provider error kept on a trace row.
pub const TRACE_ERROR_CAP: usize = 300;

/// A provider's error message made fit for a trace row, which is kept on disk
/// and read by other tools. Only the first line is kept, capped at
/// [`TRACE_ERROR_CAP`] characters, because an error body can echo part of the
/// request; credential shapes are redacted; and any URL loses its
/// `user:password@` and its query string, where an API key can ride.
pub fn redact_error_for_trace(text: &str) -> String {
    static USERINFO: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)\b(https?://)[^/\s@]+@").expect("valid")
    });
    static QUERY: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)\b(https?://[^\s?#]+)\?[^\s]*").expect("valid")
    });
    let first = text.lines().next().unwrap_or("").trim();
    let redacted = redact_secrets(first);
    let no_userinfo = USERINFO.replace_all(&redacted, "$1");
    let clean = QUERY.replace_all(&no_userinfo, "$1?[query removed]");
    let mut out: String = clean.chars().take(TRACE_ERROR_CAP).collect();
    if clean.chars().count() > TRACE_ERROR_CAP {
        out.push('…');
    }
    out
}

pub fn redact_secrets(text: &str) -> String {
    let text = PEM_KEY_RE.replace_all(text, "[redacted private key]");
    // The value is replaced, the `key =` part is kept: knowing that a turn
    // looked at `OPENAI_API_KEY` is useful, knowing its value is not.
    let text = KEYED_VALUE_RE.replace_all(&text, |caps: &regex::Captures| {
        let name = caps[1].to_lowercase();
        if SECRET_NAME_PARTS.iter().any(|part| name.contains(part)) {
            format!("{}\"[redacted]\"", &caps[1])
        } else {
            caps[0].to_string()
        }
    });
    let text = BEARER_RE.replace_all(&text, |caps: &regex::Captures| {
        format!("{}[redacted]", &caps[1])
    });
    PREFIXED_TOKEN_RE
        .replace_all(&text, "[redacted]")
        .into_owned()
}

/// Whether text holds anything shaped like a credential, by the same patterns
/// [`redact_secrets`] scrubs with — one predicate, one pattern list, so the
/// taint gate (SE-4) and the scrubber cannot disagree about what a secret
/// looks like. A `KEYED_VALUE` match only counts when its name is
/// secret-shaped, exactly as the scrubber insists.
pub fn contains_secret(text: &str) -> bool {
    if PEM_KEY_RE.is_match(text) {
        return true;
    }
    if BEARER_RE.is_match(text) {
        return true;
    }
    if PREFIXED_TOKEN_RE.is_match(text) {
        return true;
    }
    KEYED_VALUE_RE.captures_iter(text).any(|caps| {
        caps.get(1).is_some_and(|name| {
            let name = name.as_str().to_lowercase();
            SECRET_NAME_PARTS.iter().any(|part| name.contains(part))
        })
    })
}

/// Every byte range in `text` that holds a credential, by the same four shapes
/// [`redact_secrets`] removes — one pattern list, so the redactor and the
/// scrubber and the taint gate can never disagree about what a secret is. The
/// ranges are sorted and merged (a private-key block that also trips a
/// prefixed-token match yields one span, not two), and each covers the *value*
/// only: a bearer prefix and a `KEY =` name are left standing so the row still
/// reads, exactly as [`redact_secrets`] keeps them.
///
/// This is the locate step the placeholder/restore map (PR-3) builds on: it
/// needs not just "is there a secret" but "replace this exact slice and
/// remember what it was".
pub fn secret_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for m in PEM_KEY_RE.find_iter(text) {
        spans.push((m.start(), m.end()));
    }
    for m in PREFIXED_TOKEN_RE.find_iter(text) {
        spans.push((m.start(), m.end()));
    }
    // Bearer: the match leads with `bearer `, which stays; only its token goes.
    for caps in BEARER_RE.captures_iter(text) {
        if let (Some(whole), Some(prefix)) = (caps.get(0), caps.get(1)) {
            spans.push((prefix.end(), whole.end()));
        }
    }
    // A keyed value counts only when the *name* is secret-shaped, and then the
    // value (group 2) is what is hidden, not the `name =`.
    for caps in KEYED_VALUE_RE.captures_iter(text) {
        if let (Some(name), Some(value)) = (caps.get(1), caps.get(2)) {
            let lowered = name.as_str().to_lowercase();
            if SECRET_NAME_PARTS.iter().any(|part| lowered.contains(part)) {
                spans.push((value.start(), value.end()));
            }
        }
    }
    spans.sort_unstable();
    // Merge overlaps and adjacency, keeping the widest reach of any covering run.
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// One credential-shaped value found in file content, located so a person can
/// go look at the line. `kind` names which of the credential shapes matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretHit {
    /// 1-based line number within the scanned text.
    pub line: usize,
    pub kind: &'static str,
}

/// The opening marker of a private-key block, matched per line. The full PEM
/// regex spans lines, so a line-scanner needs a marker of its own to say where
/// a key begins.
static PEM_BEGIN_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| pattern(r"(?i)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"));

/// Where in `text` credentials sit, by the same shapes [`contains_secret`]
/// detects and [`redact_secrets`] scrubs — one pattern list, three uses, so the
/// content scanner cannot disagree with the taint gate or the transcript
/// scrubber about what a secret looks like. A single line can carry more than
/// one shape; each is reported once. This is deliberately the *broad* list: it
/// catches a bare `sk-…` token or a pasted private key that a name-gated scan
/// misses, and it flags a harmless `key_count=500` too — the allowlist and the
/// fixture skip are what keep that from becoming noise.
pub fn scan_secrets(text: &str) -> Vec<SecretHit> {
    let mut hits = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let mut kinds: Vec<&'static str> = Vec::new();
        if PEM_BEGIN_RE.is_match(line) {
            kinds.push("private key");
        }
        if PREFIXED_TOKEN_RE.is_match(line) {
            kinds.push("API key");
        }
        if BEARER_RE.is_match(line) {
            kinds.push("bearer token");
        }
        if KEYED_VALUE_RE.captures_iter(line).any(|caps| {
            caps.get(1).is_some_and(|name| {
                let name = name.as_str().to_lowercase();
                SECRET_NAME_PARTS.iter().any(|part| name.contains(part))
            })
        }) {
            kinds.push("secret assignment");
        }
        for kind in kinds {
            hits.push(SecretHit {
                line: line_no,
                kind,
            });
        }
    }
    hits
}

/// Built-in path shapes whose content is not secret-scanned, because a
/// credential-looking string there is documentation or a test fixture rather
/// than a live leak: an `examples/`, `testdata/`, `fixtures/` or `samples/`
/// directory, or a `*.example` / `*.sample` / `*.template` file. This mirrors
/// the walker's own `.example` exemption so the two never disagree.
pub fn path_skips_secret_scan(rel: &str) -> bool {
    let rel = rel.replace('\\', "/").to_ascii_lowercase();
    if rel.split('/').filter(|seg| !seg.is_empty()).any(|seg| {
        matches!(
            seg,
            "examples" | "example" | "testdata" | "fixtures" | "fixture" | "samples" | "sample"
        )
    }) {
        return true;
    }
    let file = rel.rsplit('/').next().unwrap_or("");
    file.ends_with(".example")
        || file.ends_with(".sample")
        || file.ends_with(".template")
        || file.ends_with(".tpl")
        || file.ends_with(".dist")
}

/// Load the user's secret-scan allowlist from `.xencode/cache/secrets-allowlist`.
/// Each non-blank line that is not a `#` comment is a repo-relative path prefix
/// or file name whose content is left alone. An unreadable file is no entries —
/// the scan then reports everything, which is the safe direction.
pub fn load_secret_allowlist(xencode_dir: &Path) -> Vec<String> {
    std::fs::read_to_string(xencode_dir.join("cache").join("secrets-allowlist"))
        .map(|text| {
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty() && !line.starts_with('#'))
                .map(|line| line.replace('\\', "/").to_ascii_lowercase())
                .collect()
        })
        .unwrap_or_default()
}

/// Whether `rel` is excused from the content scan by the user's allowlist:
/// the entry matches as a whole path, a leading directory, or a named file in
/// any directory. Case- and separator-insensitive, like the built-in skip.
pub fn allowlisted_by(rel: &str, entries: &[String]) -> bool {
    let rel = rel.replace('\\', "/").to_ascii_lowercase();
    entries.iter().any(|entry| {
        let entry = entry.trim_matches('/');
        !entry.is_empty()
            && (rel == entry
                || rel.starts_with(&format!("{entry}/"))
                || rel.ends_with(&format!("/{entry}"))
                || rel.contains(&format!("/{entry}/")))
    })
}

/// The last `cap` bytes of `text` on a character boundary, as one line, with
/// credentials removed. Redaction runs first so a secret cannot survive by
/// being cut in half.
pub fn tail_preview(text: &str, cap: usize) -> String {
    let redacted = redact_secrets(text);
    let mut start = 0;
    if redacted.len() > cap {
        start = redacted.len() - cap;
        while !redacted.is_char_boundary(start) {
            start += 1;
        }
    }
    let mut tail: String = redacted[start..]
        .chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect();
    while tail.starts_with(' ') {
        tail.remove(0);
    }
    while tail.ends_with(' ') {
        tail.pop();
    }
    tail
}

/// The first `cap` bytes of `text` on a character boundary, as one line,
/// followed by `…` when anything was dropped. Unlike a tool's output — where
/// the reason a command failed is at the end — arguments lead with what they
/// point at, so the front is the part worth keeping.
fn head_line(text: &str, cap: usize) -> String {
    let mut end = text.len().min(cap);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let cut_short = end < text.len();
    let mut line: String = text[..end]
        .chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect();
    while line.ends_with(' ') {
        line.pop();
    }
    if cut_short {
        line.push('…');
    }
    line
}

/// What a call's arguments were, in the form the trace keeps: references kept,
/// payloads replaced by their size, credentials removed, one line, cut to
/// [`TRACE_ARGUMENTS_CAP`]. `None` for a call that took no arguments, so an
/// absent field and an empty one mean different things.
pub fn arguments_preview(arguments: &serde_json::Value) -> Option<String> {
    let mut object = match arguments {
        serde_json::Value::Object(map) => map.clone(),
        // A backend that streams its arguments may hand them over as a JSON
        // string that has not been parsed yet.
        serde_json::Value::String(text) => {
            serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(text)
                .unwrap_or_default()
        }
        _ => serde_json::Map::new(),
    };
    if object.is_empty() {
        return None;
    }
    for (key, value) in object.iter_mut() {
        if BULK_ARGUMENT_KEYS.contains(&key.as_str()) {
            *value = serde_json::Value::String(bulk_note(value));
        }
    }
    let rendered = serde_json::to_string(&serde_json::Value::Object(object))
        .unwrap_or_else(|_| "{}".to_string());
    Some(head_line(&redact_secrets(&rendered), TRACE_ARGUMENTS_CAP))
}

/// How a payload argument is described in place of its text. Counted in items
/// when it is a list, in bytes otherwise.
fn bulk_note(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Array(items) => format!("[{} items]", items.len()),
        serde_json::Value::String(text) => format!("[{} bytes]", text.len()),
        other => format!("[{} bytes]", other.to_string().len()),
    }
}

pub fn trace_path(xencode_dir: &Path) -> std::path::PathBuf {
    xencode_dir.join("cache").join("turns.jsonl")
}

/// Append one turn to `turns.jsonl` (append-only, like the metrics file).
pub fn append_trace(xencode_dir: &Path, trace: &TurnTrace) -> std::io::Result<()> {
    let path = trace_path(xencode_dir);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let line = serde_json::to_string(trace)?;
    use std::io::Write;
    writeln!(file, "{line}")
}

/// The most recent `limit` turns, oldest first — the window the `/trace`
/// pane shows. A row cut off by a crash is dropped rather than failing the
/// read, same as the metrics file.
pub fn read_recent_traces(xencode_dir: &Path, limit: usize) -> Vec<TurnTrace> {
    let rows = xencode_core_rs::read_jsonl_tolerant::<TurnTrace>(&trace_path(xencode_dir)).rows;
    let skip = rows.len().saturating_sub(limit);
    rows.into_iter().skip(skip).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn temp_dir() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-trace-test-{unique}"))
    }

    /// `contains_secret` and `redact_secrets` agree: whatever one flags,
    /// the other scrubs, because they read the same patterns.
    #[test]
    fn secret_detection_matches_secret_scrubbing() {
        let secrets = [
            "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
            "Authorization: Bearer FAKEJwtNotARealToken123",
            "OPENAI_API_KEY=\"sk-FAKE-NOT-A-REAL-TEST-KEY\"",
            "AKIAFAKEFAKEFAKEFA",
        ];
        for secret in secrets {
            assert!(contains_secret(secret), "{secret} must taint");
            assert!(
                !redact_secrets(secret).contains("sk-FAKE-NOT-A-REAL-TEST-KEY"),
                "and must scrub"
            );
        }
        for plain in ["fn main() {}", "the monkey ate a secret banana", "count=5"] {
            assert!(!contains_secret(plain), "{plain} must not taint");
        }
        // Deliberately over-broad, as the scrubber's own docs admit: a
        // harmless `key_count` taints, because leaving a real key costs
        // an exfiltration and taint only buys a prompt.
        assert!(contains_secret("key_count=500"));
    }

    /// The content scanner locates a secret by line, so a scan can point at it.
    #[test]
    fn scan_secrets_locates_a_bare_token_a_bearer_and_a_pem_marker() {
        let text = "use reqwest;\n\
                    fn auth() {\n\
                    \x20   let c = \"sk-proj-FAKE-NOT-A-REAL-TEST-KEY\";\n\
                    \x20   let h = format!(\"Authorization: Bearer FAKEJwtNotAReal.abc\");\n\
                    \x20   let pem = \"-----BEGIN OPENSSH PRIVATE KEY-----\";\n\
                    }\n";
        let hits = scan_secrets(text);
        let found = |line: usize, kind: &str| hits.iter().any(|h| h.line == line && h.kind == kind);
        // Line 3: a bare `sk-…` token — no secret-shaped *name*, so a
        // name-gated scanner misses it, but the broad prefix list catches it.
        assert!(found(3, "API key"), "{hits:?}");
        // Line 4: a bearer token.
        assert!(found(4, "bearer token"), "{hits:?}");
        // Line 5: the opening line of a private key.
        assert!(found(5, "private key"), "{hits:?}");
    }

    /// The scanner agrees with the taint predicate: what one flags, the other
    /// sees, because both read the same patterns.
    #[test]
    fn scan_secrets_agrees_with_contains_secret() {
        for text in [
            "AWS_SECRET_ACCESS_KEY=FAKE_NOT_A_REAL_SECRET",
            "token: \"ghp_FAKE_NOT_A_REAL_TEST_KEY\"",
        ] {
            assert!(contains_secret(text));
            assert!(!scan_secrets(text).is_empty(), "{text} must locate");
        }
        for plain in ["fn main() {}", "let count = 5; // secrets are fun"] {
            assert!(!contains_secret(plain));
            assert!(scan_secrets(plain).is_empty(), "{plain} must not fire");
        }
    }

    #[test]
    fn the_secret_allowlist_skips_example_trees_and_honours_a_prefix() {
        // Built-in: fixture/sample trees and `.example` files are documentation,
        // not live leaks.
        assert!(path_skips_secret_scan("examples/config.rs"));
        assert!(path_skips_secret_scan("tests/fixtures/keys.json"));
        assert!(path_skips_secret_scan(".env.example"));
        assert!(!path_skips_secret_scan("src/keys.rs"));
        // A user allowlist line matches a whole path, a leading directory, or a
        // named file anywhere.
        let entries: Vec<String> = vec!["integration/legacy.rs".to_string(), "vendor".to_string()];
        assert!(allowlisted_by("integration/legacy.rs", &entries));
        assert!(allowlisted_by("src/deep/vendor/key.rs", &entries));
        assert!(!allowlisted_by("src/keys.rs", &entries));
        // An unreadable allowlist file yields no entries, never a crash.
        let missing = temp_dir();
        assert!(load_secret_allowlist(&missing).is_empty());
    }

    #[test]
    fn a_trace_error_keeps_one_short_line_with_no_credential_in_it() {
        let said = "request to https://someone:FAKE-NOT-A-REAL-PASS@llm.example.test/v1?key=FAKE-NOT-A-REAL-TEST-KEY failed
second line echoing the prompt";
        let kept = redact_error_for_trace(said);
        assert_eq!(
            kept,
            "request to https://llm.example.test/v1?[query removed] failed"
        );
        let long = "x".repeat(TRACE_ERROR_CAP + 50);
        assert_eq!(
            redact_error_for_trace(&long).chars().count(),
            TRACE_ERROR_CAP + 1
        );
    }

    #[test]
    fn a_turn_round_trips_through_the_file() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let trace = TurnTrace {
            duration_ms: 4_200,
            rounds: 3,
            tools: vec![ToolTrace {
                name: "read_file".to_string(),
                outcome: "done".to_string(),
                arguments: Some("{\"path\":\"notes.txt\"}".to_string()),
                tail: Some("fn main() {".to_string()),
            }],
            prompt_sha256: Some(prompt_digest("why does the build fail")),
            completion_tokens: Some(412),
            retrieved_files: vec!["notes.txt".to_string(), "src/main.rs".to_string()],
            is_decision: true,
            ..TurnTrace::new(0, 0)
        };
        append_trace(&xencode, &trace).unwrap();

        let rows = read_recent_traces(&xencode, 50);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rounds, 3);
        assert_eq!(rows[0].duration_ms, 4_200);
        assert_eq!(rows[0].tools[0].name, "read_file");
        assert_eq!(
            rows[0].tools[0].arguments.as_deref(),
            Some("{\"path\":\"notes.txt\"}")
        );
        assert_eq!(rows[0].completion_tokens, Some(412));
        assert_eq!(rows[0].retrieved_files, vec!["notes.txt", "src/main.rs"]);
        assert!(rows[0].is_decision, "the prompt carried the [d] marker");
        // The turn says which instructions it ran under, not just what the user typed.
        assert_eq!(
            rows[0].prompt_version.as_deref(),
            Some(crate::prompts::set_version())
        );
        // Nothing prices a request yet, so nothing pretends to.
        assert_eq!(rows[0].est_cost_micros, None);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_reader_keeps_only_the_newest_rows() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        for round in 0..6 {
            let mut trace = TurnTrace::new(1_000, 1);
            trace.rounds = round;
            append_trace(&xencode, &trace).unwrap();
        }
        let rows = read_recent_traces(&xencode, 3);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].rounds, 3);
        assert_eq!(rows[2].rounds, 5);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_turn_written_by_an_older_build_still_reads() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        fs::create_dir_all(xencode.join("cache")).unwrap();
        // The shape a build before the identity fields wrote: no session, no
        // model, no provider, no source.
        fs::write(
            trace_path(&xencode),
            "{\"ts_unix_ms\":1,\"duration_ms\":9,\"rounds\":1,\"tools\":[]}\n",
        )
        .unwrap();
        let rows = read_recent_traces(&xencode, 50);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].duration_ms, 9);
        assert!(rows[0].session_id.is_none());
        assert_eq!(rows[0].completion_tokens, None);
        // A turn from before prompts were versioned claims no prompt set, which is
        // right: that build did not know what its instructions were worth.
        assert_eq!(rows[0].prompt_version, None);
        // The fields added later are absent from an old row, so they read as
        // unmarked and as nothing retrieved. That is what a row written before
        // they existed can say, not a claim that the turn had none.
        assert!(rows[0].retrieved_files.is_empty());
        assert!(!rows[0].is_decision);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_checks_verdict_says_what_ran_failed_and_never_ran() {
        let verdict = ChecksVerdict {
            ran: vec!["cargo test".into(), "cargo clippy".into()],
            failed: vec!["cargo clippy".into()],
            skipped: vec![],
            evidence_ref: Some("tools[3]".into()),
        };
        assert!(!verdict.all_passed());
        assert_eq!(
            verdict.summary(),
            "checks: 2 ran, 1 failed, 0 skipped — failed: cargo clippy (output: tools[3])"
        );
        // A check that never ran is not a pass.
        let unrun = ChecksVerdict {
            skipped: vec!["cargo test".into()],
            ..Default::default()
        };
        assert!(!unrun.all_passed());
        assert!(unrun.summary().contains("not run: cargo test"));
        assert!(!unrun.summary().contains("verified"));

        // A row written before the field existed reads as "no checks".
        let old: TurnTrace =
            serde_json::from_str(r#"{"ts_unix_ms":1,"duration_ms":2,"rounds":1}"#).unwrap();
        assert_eq!(old.checks, None);
        // And a row without checks does not grow an empty field on disk.
        let line = serde_json::to_string(&TurnTrace::new(1, 1)).unwrap();
        assert!(!line.contains("checks"), "{line}");
    }

    #[test]
    fn a_missing_or_half_written_trace_file_reads_as_what_it_can() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        assert!(read_recent_traces(&xencode, 50).is_empty());

        let mut trace = TurnTrace::new(10, 1);
        trace.rounds = 7;
        append_trace(&xencode, &trace).unwrap();
        use std::io::Write;
        fs::OpenOptions::new()
            .append(true)
            .open(trace_path(&xencode))
            .unwrap()
            .write_all(b"{\"rounds\":9,\"dur")
            .unwrap();
        let rows = read_recent_traces(&xencode, 50);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rounds, 7);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_same_prompt_always_digests_alike_and_a_changed_one_differs() {
        assert_eq!(
            prompt_digest("rename the helper"),
            prompt_digest("rename the helper")
        );
        assert_ne!(
            prompt_digest("rename the helper"),
            prompt_digest("Rename the helper")
        );
        assert_eq!(prompt_digest("anything").len(), 16);
    }

    #[test]
    fn credentials_are_removed_whatever_shape_they_arrive_in() {
        let cases = [
            ("OPENAI_API_KEY=sk-FAKE-NOT-A-REAL-TEST-KEY", "sk-FAKE-NOT-A-REAL"),
            (
                "export AWS_SECRET_ACCESS_KEY='FAKE_NOT_A_REAL_SECRET_KEY'",
                "FAKE_NOT_A_REAL",
            ),
            (
                "curl -H \"Authorization: Bearer FAKEJwtNotARealToken123\"",
                "FAKEJwtNotAReal",
            ),
            (
                "token: ghp_FAKE_NOT_A_REAL_TEST_KEY",
                "ghp_FAKE_NOT_A_REAL",
            ),
            (
                "{\"AWS_SECRET_ACCESS_KEY\": \"FAKE_NOT_A_REAL_SECRET_KEY\"}",
                "FAKE_NOT_A_REAL",
            ),
            ("slack=xoxb-FAKE-NOT-A-REAL-TEST", "xoxb-FAKE-NOT-A-REAL"),
            (
                "key=AIzaSyFAKE_NOT_A_REAL_GOOGLE_TEST",
                "AIzaSyFAKE_NOT_A_REAL",
            ),
            (
                "aws_key_id AKIAFAKEFAKEFAKEFA here",
                "AKIAFAKE",
            ),
            (
                "-----BEGIN RSA PRIVATE KEY-----\nFAKEKEYBODYnotAReal123\n-----END RSA PRIVATE KEY-----",
                "FAKEKEYBODYnotAReal",
            ),
        ];
        for (input, secret) in cases {
            let out = redact_secrets(input);
            assert!(!out.contains(secret), "{input}\n -> {out}");
            assert!(out.contains("redacted"), "{input}\n -> {out}");
        }
        // The name survives, so the row still says what the turn touched.
        assert_eq!(
            redact_secrets("password=hunter2hunter2"),
            "password=\"[redacted]\""
        );
    }

    #[test]
    fn ordinary_output_survives_redaction() {
        let text = "3 files changed, 12 insertions(+)\ntoken_count is not a credential";
        assert_eq!(redact_secrets(text), text);
    }

    #[test]
    fn a_preview_is_one_line_and_ends_where_the_secret_was() {
        let noisy = "build log\n".repeat(50) + "final error: OPENAI_API_KEY=sk-abcdefghijklmnop\n";
        let preview = tail_preview(&noisy, TRACE_TAIL_CAP);
        assert!(!preview.contains('\n'));
        assert!(!preview.contains("sk-abcdefghijklmnop"));
        assert!(preview.contains("[redacted]"));
        assert!(preview.len() <= TRACE_TAIL_CAP);
        // A short result is kept whole.
        assert_eq!(tail_preview("ok", 100), "ok");
        // Cutting on a character boundary must not panic on multi-byte text.
        let unicode = "é".repeat(400) + "tail";
        assert!(tail_preview(&unicode, 50).ends_with("tail"));
    }

    #[test]
    fn arguments_are_kept_as_far_as_they_explain_the_call() {
        // Where a call pointed is kept whole.
        let read = arguments_preview(&serde_json::json!({"path": "src/main.rs", "limit": 40}))
            .expect("a call with arguments");
        assert!(read.contains("src/main.rs"), "{read}");
        assert!(read.contains("40"), "{read}");
        // What a call carried is kept as its size, not its text.
        let body = "fn main() {}\n".repeat(100);
        let write = arguments_preview(&serde_json::json!({
            "path": "src/main.rs",
            "content": body,
        }))
        .expect("a call with arguments");
        assert!(write.contains("src/main.rs"), "{write}");
        assert!(write.contains("[1300 bytes]"), "{write}");
        assert!(!write.contains("fn main()"), "{write}");
        // An edit keeps both sizes and the path; the text either way is in the file.
        let edit = arguments_preview(&serde_json::json!({"path": "a.rs", "old": "x", "new": "yy"}))
            .expect("a call with arguments");
        assert!(
            edit.contains("[1 bytes]") && edit.contains("[2 bytes]"),
            "{edit}"
        );
        // A list is counted, not spelled out.
        let plan = arguments_preview(&serde_json::json!({"items": [
            {"text": "read the failing test", "status": "pending"},
            {"text": "fix it", "status": "pending"},
            {"text": "run the suite", "status": "pending"},
        ]}))
        .expect("a call with arguments");
        assert_eq!(plan, "{\"items\":\"[3 items]\"}");
        assert!(!plan.contains("failing test"), "{plan}");
    }

    #[test]
    fn a_call_that_took_no_arguments_records_nothing() {
        assert!(arguments_preview(&serde_json::json!({})).is_none());
        assert!(arguments_preview(&serde_json::Value::Null).is_none());
        // The shape a streamed call arrives in before its arguments are parsed.
        let parsed = arguments_preview(&serde_json::Value::String(
            "{\"path\":\"notes.txt\"}".to_string(),
        ))
        .expect("a string of arguments is still a call's arguments");
        assert!(parsed.contains("notes.txt"), "{parsed}");
        // Not JSON at all: nothing to keep, and nothing to fail on.
        assert!(arguments_preview(&serde_json::Value::String("half a cal".to_string())).is_none());
    }

    #[test]
    fn a_credential_inside_an_argument_is_removed_and_a_long_argument_is_cut() {
        let preview = arguments_preview(&serde_json::json!({
            "command": "curl -H \"Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\" https://api.example"
        }))
        .expect("a call with arguments");
        assert!(
            !preview.contains("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
            "{preview}"
        );
        assert!(preview.contains("[redacted]"), "{preview}");
        assert!(
            preview.contains("curl"),
            "the command itself stays readable"
        );

        let long =
            arguments_preview(&serde_json::json!({"pattern": "a".repeat(2_000)})).expect("pattern");
        assert!(long.ends_with('…'), "{long}");
        // The cap is characters of the kept preview; the ellipsis marks the cut.
        assert!(long.len() <= TRACE_ARGUMENTS_CAP + '…'.len_utf8());
        // A cut that would land inside a multi-byte character does not panic.
        let unicode =
            arguments_preview(&serde_json::json!({"pattern": "é".repeat(400)})).expect("pattern");
        assert!(unicode.ends_with('…'));
    }

    #[test]
    fn a_trace_row_carries_the_same_identity_as_a_metrics_row() {
        let identity = MetricsIdentity {
            session_id: Some("session_1".to_string()),
            model: Some("anthropic:claude-3-5-sonnet".to_string()),
            provider: Some("anthropic".to_string()),
            source: Some(MetricSource::Cloud),
        };
        let trace = TurnTrace::new(5, 2).with_identity(identity);
        assert_eq!(trace.session_id.as_deref(), Some("session_1"));
        assert_eq!(trace.provider.as_deref(), Some("anthropic"));
        assert_eq!(trace.source, Some(MetricSource::Cloud));
    }
}
