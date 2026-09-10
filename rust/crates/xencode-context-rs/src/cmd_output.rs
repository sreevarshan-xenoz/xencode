//! Tool-output compression (§12) — the hidden context killer.
//!
//! Rule: **the conversation never retains raw tool output.** A command like
//! `cargo test` that prints 500 lines becomes a 4-line summary in the prompt
//! plus an on-disk blob that is re-read only when the model asks for it:
//!
//! ```text
//! cargo test → 500 lines → stored as .xencode/cache/cmd/<sha256>.txt
//!
//! conversation retains only:
//! COMMAND: cargo test
//! RESULT: FAILED (3 errors)
//! FILES: auth.rs, router.rs
//! DETAIL: <up to 4 lines>
//! RAW: cmd/<hash>.txt
//! ```
//!
//! Content is deduplicated by SHA-256 of the raw output, so identical runs
//! reuse one blob and never balloon the cache.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

/// Detail lines the summary carries — the model's own one-liner if available.
pub const DETAIL_LINES_CAP: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CmdRecord {
    pub command: String,
    pub ts_unix_ms: u64,
    /// Raw output stored as `cache/cmd/<hash>.txt`.
    pub log_sha256: String,
    /// Characters of raw output captured (for cheap size reporting).
    pub chars: u64,
    /// Short textual outcome, e.g. `FAILED (3 errors)` or `0 tests failed`.
    pub result_summary: String,
    /// Files implicated (parsed from the output, may be empty).
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CmdIndex {
    /// sha256 → record. One blob per unique output.
    #[serde(default)]
    pub blobs: BTreeMap<String, CmdRecord>,
    #[serde(default)]
    pub total_chars: u64,
}

impl CmdIndex {
    pub fn from_disk(xencode_dir: &Path) -> Option<Self> {
        crate::index::read_json(&index_path(xencode_dir))
    }

    pub fn save_to(&self, xencode_dir: &Path) -> io::Result<()> {
        if let Some(parent) = index_path(xencode_dir).parent() {
            fs::create_dir_all(parent)?;
        }
        crate::index::write_atomic(&index_path(xencode_dir), self)
    }
}

pub fn index_path(xencode_dir: &Path) -> std::path::PathBuf {
    xencode_dir.join("cache").join("cmd").join("index.json")
}

/// Capture raw command output: hash it, store `cache/cmd/<hash>.txt` (one blob
/// per unique content) and return a summary record. Returns `None` when the
/// output is empty (nothing worth retaining).
pub fn capture_output(
    xencode_dir: &Path,
    command: &str,
    raw_output: &str,
    result_summary: &str,
    files: &[String],
) -> io::Result<Option<CmdRecord>> {
    if raw_output.is_empty() {
        return Ok(None);
    }
    let mut idx = CmdIndex::from_disk(xencode_dir).unwrap_or_default();
    let hash = sha256_hex(raw_output);
    let blob_dir = xencode_dir.join("cache").join("cmd");
    fs::create_dir_all(&blob_dir)?;

    let blob_path = blob_dir.join(format!("{hash}.txt"));
    if !blob_path.exists() {
        fs::write(&blob_path, raw_output)?;
        idx.total_chars += raw_output.len() as u64;
    }

    let record = CmdRecord {
        command: command.to_string(),
        ts_unix_ms: crate::conversation::now_millis(),
        log_sha256: hash.clone(),
        chars: raw_output.len() as u64,
        result_summary: result_summary.to_string(),
        files: files.to_vec(),
    };
    idx.blobs.insert(hash, record.clone());
    idx.save_to(xencode_dir)?;
    Ok(Some(record))
}

/// Re-read the raw blob on demand (`RAW: cmd/<hash>.txt`).
pub fn read_raw(xencode_dir: &Path, log_sha256: &str) -> Option<String> {
    let path = xencode_dir
        .join("cache")
        .join("cmd")
        .join(format!("{log_sha256}.txt"));
    fs::read_to_string(path).ok()
}

/// Render the §12 summary block that enters the conversation (not the raw
/// output). `detail_lines` is whatever the model already summarized of the
/// output; it is always trimmed to `DETAIL_LINES_CAP`.
pub fn render_summary(record: &CmdRecord, detail_lines: &[&str]) -> String {
    let mut out = format!("COMMAND: {}\n", record.command);
    out.push_str(&format!("RESULT: {}\n", record.result_summary));
    if !record.files.is_empty() {
        out.push_str(&format!("FILES: {}\n", record.files.join(", ")));
    }
    if !detail_lines.is_empty() {
        out.push_str("DETAIL:");
        for line in detail_lines.iter().take(DETAIL_LINES_CAP) {
            out.push_str("\n  ");
            out.push_str(line);
        }
        out.push('\n');
    }
    out.push_str(&format!("RAW: cmd/{}.txt\n", record.log_sha256));
    out
}

/// Deterministic hex SHA-256 (Windows line endings normalized to `\n` so the
/// same output hashes identically across CRLF/LF checkouts).
pub fn sha256_hex(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n");
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Naive "which files does this output mention" — lines in `src/**` or ending
/// in a known source extension. Cost cap so giant outputs don't stall the loop.
pub fn deduce_files(raw_output: &str, max_files: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in raw_output.lines().take(2_000) {
        for token in line.split_whitespace() {
            let token = token.trim_end_matches([':', ')', ',', ';']);
            let looks_like_src = token.starts_with("src/")
                || token.starts_with("tests/")
                || token.starts_with("lib/")
                || token.starts_with("rust/")
                || token.contains('/')
                    && token
                        .rsplit_once('.')
                        .map(|(_, ext)| {
                            matches!(
                                ext,
                                "rs" | "py" | "ts" | "js" | "go" | "c" | "h" | "cpp" | "hpp"
                            )
                        })
                        .unwrap_or(false);
            if looks_like_src && token.len() < 120 && !out.contains(&token.to_string()) {
                out.push(token.to_string());
                if out.len() >= max_files {
                    return out;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn temp_dir() -> PathBuf {
        // A process-wide counter, not a timestamp. Tests in one binary run in
        // parallel threads and can read the same nanosecond, which had them share
        // a directory and clobber each other's assertions.
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-cmd-test-{unique}"))
    }

    #[test]
    fn captures_dedupes_and_re_reads_raw() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        let out =
            "error[E0308]: mismatched types\n  --> src/auth.rs:42:10\n  --> src/router.rs:9:3\n";
        let files = deduce_files(out, 5);
        assert_eq!(files, vec!["src/auth.rs:42:10", "src/router.rs:9:3"]);

        let r1 = capture_output(&xencode, "cargo check", out, "FAILED (2 errors)", &files)
            .unwrap()
            .unwrap();
        let r2 = capture_output(&xencode, "cargo check", out, "FAILED (2 errors)", &files)
            .unwrap()
            .unwrap();
        assert_eq!(r1.log_sha256, r2.log_sha256);

        let idx = CmdIndex::from_disk(&xencode).unwrap();
        assert_eq!(idx.blobs.len(), 1);
        assert_eq!(idx.total_chars, out.len() as u64);

        assert_eq!(read_raw(&xencode, &r1.log_sha256).as_deref(), Some(out));
        // Blob path is cache/cmd/<hash>.txt (matches the RAW: reference).
        assert!(xencode
            .join("cache")
            .join("cmd")
            .join(format!("{}.txt", r1.log_sha256))
            .exists());

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn empty_output_is_not_retained() {
        let dir = temp_dir();
        let xencode = dir.join(".xencode");
        assert!(capture_output(&xencode, "cargo fmt", "", "", &[])
            .unwrap()
            .is_none());
        assert!(
            !index_path(&xencode).exists()
                || CmdIndex::from_disk(&xencode).unwrap().blobs.is_empty()
        );
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn summary_respects_detail_cap_and_hash_is_crlf_stable() {
        let record = CmdRecord {
            command: "cargo test".to_string(),
            ts_unix_ms: 1,
            log_sha256: sha256_hex("a\r\nb"),
            chars: 10,
            result_summary: "FAILED (3 errors)".to_string(),
            files: vec!["auth.rs".to_string(), "router.rs".to_string()],
        };
        let detail_strings: Vec<String> = (0..8).map(|i| format!("detail {i}")).collect();
        let details: Vec<&str> = detail_strings.iter().map(|s| s.as_str()).collect();
        let s = render_summary(&record, &details);
        assert!(s.starts_with("COMMAND: cargo test"));
        assert!(s.contains("RESULT: FAILED (3 errors)"));
        assert!(s.contains("FILES: auth.rs, router.rs"));
        assert!(s.contains("RAW: cmd/"));
        assert_eq!(s.matches("detail").count(), DETAIL_LINES_CAP);
        // Same logical content hashes identically across CRLF and LF.
        assert_eq!(sha256_hex("a\r\nb"), sha256_hex("a\nb"));
    }
}
