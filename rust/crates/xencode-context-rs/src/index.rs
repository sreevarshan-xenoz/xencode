//! On-disk index formats for the context engine.
//!
//! - `index/files.json`   — per-file inventory (language, size, LOC, flags)
//! - `index/manifest.json`— invalidation snapshot (git HEAD + mtimes)
//!
//! All writes are atomic: content lands in a `.tmp` file, then an atomic
//! rename replaces the destination, so a killed process never leaves a
//! half-written index (kill-safe resume).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Schema version for the index files (bump on breaking schema changes).
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileEntry {
    /// Repo-relative path with `/` separators.
    pub path: String,
    pub language: String,
    pub size: u64,
    pub loc: u64,
    pub ext: String,
    pub important: bool,
    /// Secret-detected file: listed, never read.
    pub secret: bool,
    /// Binary-detected file: listed, never read.
    pub binary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesIndex {
    pub version: u32,
    pub generated_at: u64,
    pub files: Vec<FileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub version: u32,
    pub git_head: Option<String>,
    pub branch: Option<String>,
    pub dirty: u64,
    /// Files that existed at scan time but were skipped (ignored/excluded).
    /// Persisted so a fresh resume can report an accurate count.
    #[serde(default)]
    pub skipped: u64,
    /// Relative path (`/` separators) → modified time (epoch millis).
    pub mtime_map: BTreeMap<String, u64>,
    pub indexed_at: u64,
}

impl Default for FilesIndex {
    fn default() -> Self {
        Self {
            version: VERSION,
            generated_at: 0,
            files: Vec::new(),
        }
    }
}

/// Serialize `value` to `path` atomically (temp file + rename).
pub fn write_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(value)?;
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, json.as_bytes())?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Write a plain-text string atomically (used by `state.md`).
pub fn write_str_atomic(path: &Path, text: &str) -> Result<(), std::io::Error> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, text.as_bytes())?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Deserialize `path` as JSON; `None` when missing or corrupt.
pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Write files under `index/` inside a project.
pub fn file_index_path(xencode_dir: &Path) -> std::path::PathBuf {
    xencode_dir.join("index").join("files.json")
}

pub fn manifest_path(xencode_dir: &Path) -> std::path::PathBuf {
    xencode_dir.join("index").join("manifest.json")
}

pub fn symbols_json_path(xencode_dir: &Path) -> std::path::PathBuf {
    xencode_dir.join("index").join("symbols.json")
}

pub fn deps_json_path(xencode_dir: &Path) -> std::path::PathBuf {
    xencode_dir.join("index").join("deps.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("xencode-index-test-{stamp}"))
    }

    #[test]
    fn round_trips_files_index_atomically() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("files.json");

        let idx = FilesIndex {
            version: VERSION,
            generated_at: 42,
            files: vec![FileEntry {
                path: "src/main.rs".to_string(),
                language: "rust".to_string(),
                size: 100,
                loc: 12,
                ext: "rs".to_string(),
                important: false,
                secret: false,
                binary: false,
            }],
        };
        write_atomic(&path, &idx).unwrap();

        // No leftover temp artifacts.
        assert!(!dir.join("files.tmp").exists());

        let back: FilesIndex = read_json(&path).unwrap();
        assert_eq!(back.files.len(), 1);
        assert_eq!(back.files[0].path, "src/main.rs");
        assert_eq!(back.version, VERSION);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn manifest_round_trips_with_mtime_map() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.json");

        let mut m = Manifest::default();
        m.version = VERSION;
        m.git_head = Some("abc123".to_string());
        m.mtime_map.insert("a.rs".to_string(), 111);
        m.mtime_map.insert("b.rs".to_string(), 222);
        write_atomic(&path, &m).unwrap();

        let back: Manifest = read_json(&path).unwrap();
        assert_eq!(back.git_head.as_deref(), Some("abc123"));
        assert_eq!(back.mtime_map.len(), 2);

        fs::remove_dir_all(dir).unwrap();
    }
}