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
            ("OPENAI_API_KEY=sk-abc123def456ghi789", "sk-abc123"),
            (
                "export AWS_SECRET_ACCESS_KEY='wJalrXUtnFEMI/K7MDENG'",
                "wJalrXUtnFEMI",
            ),
            (
                "curl -H \"Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9\"",
                "eyJhbGciOiJIUzI1NiI",
            ),
            (
                "token: ghp_0123456789abcdef0123456789abcdef0123",
                "ghp_0123456789abcdef",
            ),
            (
                "{\"AWS_SECRET_ACCESS_KEY\": \"wJalrXUtnFEMI/K7MDENG\"}",
                "wJalrXUtnFEMI",
            ),
            ("slack=xoxb-1234567890-abcdefghij", "xoxb-1234567890"),
            (
                "key=AIzaSyA1234567890abcdefghijklmnopqrstuv",
                "AIzaSyA1234567890",
            ),
            (
                "aws_key_id AKIAIOSFODNN7EXAMPLE here",
                "AKIAIOSFODNN7EXAMPLE",
            ),
            (
                "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
                "MIIEpAIBAAKCAQEA",
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
