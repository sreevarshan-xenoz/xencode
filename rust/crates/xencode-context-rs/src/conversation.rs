//! Canonical conversation transcript (M3 — context lifecycle).
//!
//! The transcript lives in `.xencode/cache/transcript/current.json` and is the
//! **canonical** record of the conversation. The prompt assembled per request
//! (§10) is only a *projection*, so compaction may discard projection messages
//! without ever destroying history.
//!
//! Raw snapshots are copied to `.xencode/cache/transcript/<ts>.json` before
//! any rewrite (§11 "Raw transcript is dumped … before rewrite").

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub ts_unix_ms: u64,
    /// True when `content` carries the `[d]` decision marker (§11) — such
    /// entries are never dropped by soft compaction.
    #[serde(default)]
    pub is_decision: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transcript {
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub updated_at_unix_ms: u64,
    #[serde(default)]
    pub entries: Vec<TranscriptEntry>,
}

impl Transcript {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            updated_at_unix_ms: now_millis(),
            entries: Vec::new(),
        }
    }

    /// Append a message, stamping the timestamp and decision marker.
    pub fn add(&mut self, role: &str, content: &str) {
        self.entries.push(TranscriptEntry {
            role: role.to_string(),
            content: content.to_string(),
            ts_unix_ms: now_millis(),
            is_decision: crate::state::has_decision_marker(content),
        });
        self.updated_at_unix_ms = now_millis();
    }

    /// The `n` most recent entries.
    pub fn recent(&self, n: usize) -> Vec<&TranscriptEntry> {
        let skip = self.entries.len().saturating_sub(n);
        self.entries[skip..].iter().collect()
    }

    /// Entries that must survive any compaction (decision-marked).
    pub fn decisions(&self) -> Vec<&TranscriptEntry> {
        self.entries.iter().filter(|e| e.is_decision).collect()
    }

    /// Load the canonical store (missing/corrupt → empty).
    pub fn from_disk(path: &Path) -> Option<Self> {
        crate::index::read_json(path)
    }

    pub fn save_to(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::index::write_atomic(path, self)
    }

    /// Canonical location: `.xencode/cache/transcript/current.json`.
    pub fn current_path(xencode_dir: &Path) -> std::path::PathBuf {
        xencode_dir.join("cache").join("transcript").join("current.json")
    }

    /// Copy the canonical store to a timestamped snapshot
    /// (`.xencode/cache/transcript/<ts>.json`). Returns the snapshot path.
    /// Call this **before** any compaction rewrite (§11).
    pub fn snapshot(&self, xencode_dir: &Path) -> io::Result<std::path::PathBuf> {
        let dir = xencode_dir.join("cache").join("transcript");
        std::fs::create_dir_all(&dir)?;
        let stamp = now_millis();
        let path = dir.join(format!("{stamp}.json"));
        self.save_to(&path)?;
        Ok(path)
    }
}

pub fn now_millis() -> u64 {
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
        std::env::temp_dir().join(format!("xencode-conv-test-{stamp}"))
    }

    fn seeded() -> Transcript {
        let mut t = Transcript::new("s1");
        for i in 0..10 {
            let marker = if i == 1 { " [d]" } else { "" };
            t.add("user", &format!("message {i}{marker}"));
            t.add("assistant", &format!("reply {i}"));
        }
        t
    }

    #[test]
    fn tags_decision_marker() {
        let mut t = Transcript::new("s1");
        t.add("user", "Use actix-web [d]");
        t.add("assistant", "ok");
        assert!(t.entries[0].is_decision);
        assert!(!t.entries[1].is_decision);
    }

    #[test]
    fn recent_keeps_last_n() {
        let t = seeded();
        let recent = t.recent(3);
        assert_eq!(recent.len(), 3);
        assert!(recent[0].content.contains("8"));
        assert_every(recent, |e| e.content.contains("8") || e.content.contains("9"));
    }

    #[test]
    fn decisions_survive_anywhere() {
        let mut t = Transcript::new("s1");
        t.add("user", "first");
        t.add("assistant", "DECISION [d]");
        t.add("user", "third");
        let d = t.decisions();
        assert_eq!(d.len(), 1);
        assert!(d[0].content.contains("DECISION"));
    }

    #[test]
    fn save_load_round_trip_and_snapshot() {
        let root = temp_dir();
        let xencode = root.join(".xencode");
        let t = seeded();
        t.save_to(&Transcript::current_path(&xencode)).unwrap();
        let loaded = Transcript::from_disk(&Transcript::current_path(&xencode)).unwrap();
        assert_eq!(loaded.entries.len(), 20);
        assert_eq!(loaded.session_id, "s1");

        let snap = t.snapshot(&xencode).unwrap();
        assert!(snap.exists());
        assert_ne!(snap, Transcript::current_path(&xencode));
        let snapped = Transcript::from_disk(&snap).unwrap();
        assert_eq!(snapped.entries.len(), 20);

        fs::remove_dir_all(root).unwrap();
    }

    fn assert_every(v: Vec<&TranscriptEntry>, f: impl Fn(&TranscriptEntry) -> bool) {
        assert!(v.iter().all(|e| f(e)));
    }
}