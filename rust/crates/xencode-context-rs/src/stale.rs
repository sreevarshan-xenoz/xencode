//! Stale-file tracking (M2) — the Cline `FileContextTracker` idea.
//!
//! When a file enters the model's context we pin its content hash at "loaded"
//! time. Later, before the model edits or re-reads, we compare the current
//! hash. If they differ the model may be reasoning about obsolete content:
//!
//! ```text
//! src/router.rs
//!   indexed/loaded: <hash at load>
//!   current:        <hash now>
//!   status:         STALE  → "⚠ changed since loaded — re-read before editing"
//! ```
//!
//! State is persisted to `.xencode/cache/loaded.json`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io;
use std::path::Path;

/// `Clean` = hash unchanged since load · `Stale` = changed · 
/// `Missing` = never tracked or gone from disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileStateKind {
    Clean,
    Stale,
    Missing,
}

#[derive(Debug, Clone)]
pub struct TrackedFile {
    pub path: String,
    pub state: FileStateKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadedRecord {
    pub hash: String,
    pub loaded_at: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadedState {
    /// Repo-relative path (`/`) → hash recorded when the file was loaded.
    #[serde(default)]
    pub files: BTreeMap<String, LoadedRecord>,
}

impl LoadedState {
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

pub struct FileContextTracker {
    /// `.xencode/cache/loaded.json`
    pub path: std::path::PathBuf,
    pub state: LoadedState,
}

impl FileContextTracker {
    pub fn new(xencode_dir: &Path) -> Self {
        Self {
            path: xencode_dir.join("cache").join("loaded.json"),
            state: LoadedState::default(),
        }
    }

    /// Load persisted state from disk (missing/corrupt → empty state).
    pub fn load_from_disk(&mut self) -> &mut Self {
        self.state = crate::index::read_json(&self.path).unwrap_or_default();
        self
    }

    /// Record the current hash of each file as "loaded". Files that no longer
    /// exist are left untouched (they surface as `Missing` on the next check).
    pub fn mark_loaded(&mut self, root: &Path, paths: &[&str]) {
        let now = now_millis();
        for path in paths {
            let Some(hash) = file_hash(&root.join(path)) else {
                continue;
            };
            self.state.files.insert(
                path.to_string(),
                LoadedRecord {
                    hash,
                    loaded_at: now,
                },
            );
        }
    }

    /// Compare current hashes against the pinned ones.
    pub fn check(&self, root: &Path, paths: &[&str]) -> Vec<TrackedFile> {
        let mut out = Vec::new();
        for path in paths {
            let Some(rec) = self.state.files.get(*path) else {
                out.push(TrackedFile {
                    path: path.to_string(),
                    state: FileStateKind::Missing,
                });
                continue;
            };
            let state = match file_hash(&root.join(path)) {
                None => FileStateKind::Missing,
                Some(hash) if hash != rec.hash => FileStateKind::Stale,
                Some(_) => FileStateKind::Clean,
            };
            out.push(TrackedFile {
                path: path.to_string(),
                state,
            });
        }
        out
    }

    pub fn check_all(&self, root: &Path) -> Vec<TrackedFile> {
        let paths: Vec<&str> = self.state.files.keys().map(|s| s.as_str()).collect();
        self.check(root, &paths)
    }

    /// Human-readable warnings for anything not clean, ready for the model:
    /// `⚠ src/router.rs changed since it was loaded — re-read before relying on it.`
    pub fn status_text(&self, root: &Path, paths: &[&str]) -> String {
        let mut lines: Vec<String> = Vec::new();
        for tracked in self.check(root, paths) {
            match tracked.state {
                FileStateKind::Clean => {}
                FileStateKind::Stale => lines.push(format!(
                    "⚠ {} changed since it was loaded — re-read it before relying on the cached content.",
                    tracked.path
                )),
                FileStateKind::Missing => lines.push(format!(
                    "⚠ {} was loaded but no longer exists on disk.",
                    tracked.path
                )),
            }
        }
        lines.join("\n")
    }

    pub fn save(&self) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::index::write_atomic(&self.path, &self.state)
    }
}

/// FNV-1a 64-bit over the file bytes, hex-encoded — deterministic across runs.
fn file_hash(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Some(format!("{hash:016x}"))
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("xencode-stale-test-{stamp}"))
    }

    #[test]
    fn detects_stale_and_missing_after_mark_loaded() {
        let root = temp_dir();
        fs::create_dir_all(root.join("cache")).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/a.rs"), "v1").unwrap();
        fs::write(root.join("src/b.rs"), "b").unwrap();

        let mut tracker = FileContextTracker::new(&root.join(".xencode"));
        tracker.load_from_disk();
        tracker.mark_loaded(&root, &["src/a.rs", "src/b.rs", "src/gone.rs"]);

        // untouched → clean
        let all_clean = tracker.check(&root, &["src/a.rs", "src/b.rs"]);
        assert!(all_clean
            .iter()
            .all(|t| t.state == FileStateKind::Clean));

        // edit a.rs → stale
        fs::write(root.join("src/a.rs"), "v2").unwrap();
        let after = tracker.check(&root, &["src/a.rs"]);
        assert_eq!(after[0].state, FileStateKind::Stale);

        // never-marked file → missing
        let missing = tracker.check(&root, &["src/gone.rs"]);
        assert_eq!(missing[0].state, FileStateKind::Missing);

        // status_text surfaces a warning
        let text = tracker.status_text(&root, &["src/a.rs", "src/gone.rs"]);
        assert!(text.contains("changed since it was loaded"));
        assert!(text.contains("no longer exists"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn survives_round_trip_to_disk() {
        let root = temp_dir();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/a.rs"), "content").unwrap();

        let xencode = root.join(".xencode");
        {
            let mut tracker = FileContextTracker::new(&xencode);
            tracker.load_from_disk();
            tracker.mark_loaded(&root, &["src/a.rs"]);
            tracker.save().unwrap();
        }
        // New instance loads the pinned hash.
        let mut reloaded = FileContextTracker::new(&xencode);
        reloaded.load_from_disk();
        let tracked = reloaded.check(&root, &["src/a.rs"]);
        assert_eq!(tracked[0].state, FileStateKind::Clean);

        fs::remove_dir_all(root).unwrap();
    }
}