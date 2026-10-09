//! Live session status files (DK-1).
//!
//! Every running xencode session keeps one small JSON file in
//! `<state dir>/live/`, rewritten when its state changes and every few
//! seconds as a heartbeat. The floating badge reads the folder. Files are
//! written to a temporary name and renamed into place, so a reader never sees
//! half a file from a writer that is still running.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The format version this build writes and reads.
pub const VERSION: u32 = 1;
/// Longest headline written, in characters.
pub const HEADLINE_MAX: usize = 120;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveState {
    Idle,
    Working,
    NeedsYou,
    Finished,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LiveSource {
    Chat,
    Bytebot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiveStatus {
    pub version: u32,
    pub session_id: String,
    pub pid: u32,
    /// Project folder, with forward slashes on every platform.
    pub project: String,
    pub model: String,
    pub source: LiveSource,
    pub state: LiveState,
    /// Already redacted and clipped by the writer.
    pub headline: String,
    pub changed_at: u64,
    pub heartbeat_at: u64,
}

/// `<state dir>/live`. Follows `XCODE_CONFIG_DIR` like every other xencode folder.
pub fn live_dir() -> Result<PathBuf, String> {
    xencode_config_rs::paths::state_dir()
        .map(|dir| dir.join("live"))
        .map_err(|e| e.to_string())
}

pub fn status_path(dir: &Path, session_id: &str) -> PathBuf {
    dir.join(format!("{session_id}.json"))
}

/// Write `status` atomically: a temporary file in the same folder, renamed over
/// the old one.
pub fn write_status(dir: &Path, status: &LiveStatus) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let target = status_path(dir, &status.session_id);
    let temp = dir.join(format!(".{}.{}.tmp", status.session_id, std::process::id()));
    let body = serde_json::to_string_pretty(status).map_err(std::io::Error::other)?;
    std::fs::write(&temp, body)?;
    std::fs::rename(&temp, &target).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })
}

pub fn remove_status(dir: &Path, session_id: &str) {
    let _ = std::fs::remove_file(status_path(dir, session_id));
}

/// Delete status files whose heartbeat stopped more than `older_than` seconds
/// before `now`: their sessions were killed or crashed and cannot remove
/// their own file. Returns how many were deleted. Files that cannot be read
/// are left alone, because nothing proves their session is dead.
pub fn prune_dead(dir: &Path, now: u64, older_than: u64) -> usize {
    let mut removed = 0;
    for read in read_all(dir) {
        if let Read::Ok(status) = read {
            if now.saturating_sub(status.heartbeat_at) > older_than {
                remove_status(dir, &status.session_id);
                removed += 1;
            }
        }
    }
    removed
}

/// One file's outcome. A file that cannot be used is reported, not hidden.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Read {
    Ok(LiveStatus),
    Skipped { file: PathBuf, reason: String },
}

/// Every `*.json` status file in `dir`, sorted by session. A missing folder
/// means no sessions.
pub fn read_all(dir: &Path) -> Vec<Read> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let file = entry.path();
        if file.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let text = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(e) => {
                out.push(Read::Skipped {
                    file,
                    reason: format!("unreadable: {e}"),
                });
                continue;
            }
        };
        let version = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v.get("version").and_then(|v| v.as_u64()));
        match version {
            Some(v) if v == u64::from(VERSION) => match serde_json::from_str(&text) {
                Ok(status) => out.push(Read::Ok(status)),
                Err(e) => out.push(Read::Skipped {
                    file,
                    reason: format!("not a status file: {e}"),
                }),
            },
            Some(v) => out.push(Read::Skipped {
                file,
                reason: format!("written by a newer xencode (format {v}); update this badge"),
            }),
            None => out.push(Read::Skipped {
                file,
                reason: "not a status file".to_string(),
            }),
        }
    }
    out.sort_by_key(sort_key);
    out
}

fn sort_key(r: &Read) -> String {
    match r {
        Read::Ok(s) => s.session_id.clone(),
        Read::Skipped { file, .. } => file.to_string_lossy().into_owned(),
    }
}

/// The first line of `text`, clipped to `HEADLINE_MAX` characters and ending
/// in `…` when clipped.
pub fn clip_headline(text: &str) -> String {
    let one_line = text.lines().next().unwrap_or("").trim();
    if one_line.chars().count() <= HEADLINE_MAX {
        return one_line.to_string();
    }
    let mut out: String = one_line.chars().take(HEADLINE_MAX - 1).collect();
    out.push('…');
    out
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(id: &str) -> LiveStatus {
        LiveStatus {
            version: VERSION,
            session_id: id.to_string(),
            pid: 42,
            project: "E:/xencode".to_string(),
            model: "llamacpp:qwen3-4b".to_string(),
            source: LiveSource::Chat,
            state: LiveState::NeedsYou,
            headline: "waiting for you to allow: write_file src/auth.rs".to_string(),
            changed_at: 1_760_000_123,
            heartbeat_at: 1_760_000_125,
        }
    }

    #[test]
    fn a_written_status_reads_back_identically_and_uses_the_documented_names() {
        let dir = tempfile::tempdir().unwrap();
        write_status(dir.path(), &sample("s1")).unwrap();
        let text = std::fs::read_to_string(status_path(dir.path(), "s1")).unwrap();
        assert!(text.contains("\"state\": \"needs_you\""), "{text}");
        assert!(text.contains("\"source\": \"chat\""), "{text}");
        let all = read_all(dir.path());
        assert_eq!(all.len(), 1);
        match &all[0] {
            Read::Ok(status) => assert_eq!(status, &sample("s1")),
            Read::Skipped { reason, .. } => panic!("skipped: {reason}"),
        }
    }

    #[test]
    fn writing_twice_leaves_one_file_and_no_temporary_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = sample("s1");
        write_status(dir.path(), &s).unwrap();
        s.state = LiveState::Working;
        write_status(dir.path(), &s).unwrap();
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["s1.json".to_string()]);
    }

    #[test]
    fn unknown_version_and_broken_json_are_skipped_with_a_reason() {
        let dir = tempfile::tempdir().unwrap();
        let mut newer = serde_json::to_value(sample("new")).unwrap();
        newer["version"] = serde_json::json!(2);
        std::fs::write(dir.path().join("new.json"), newer.to_string()).unwrap();
        std::fs::write(dir.path().join("half.json"), "{\"version\": 1, \"sess").unwrap();
        std::fs::write(dir.path().join("notes.txt"), "not a status").unwrap();
        let all = read_all(dir.path());
        assert_eq!(all.len(), 2, "only .json files are considered");
        for r in &all {
            match r {
                Read::Skipped { reason, .. } => assert!(!reason.is_empty()),
                Read::Ok(s) => panic!("read {s:?}"),
            }
        }
    }

    #[test]
    fn a_missing_folder_reads_as_no_sessions() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_all(&dir.path().join("absent")).is_empty());
    }

    #[test]
    fn headlines_are_clipped_on_a_character_boundary() {
        let long = "é".repeat(200);
        let clipped = clip_headline(&long);
        assert_eq!(clipped.chars().count(), HEADLINE_MAX);
        assert!(clipped.ends_with('…'));
        assert_eq!(clip_headline("short"), "short");
    }

    #[test]
    fn only_files_silent_for_longer_than_the_limit_are_pruned() {
        let dir = tempfile::tempdir().unwrap();
        let mut dead = sample("dead");
        dead.heartbeat_at = 1_000;
        let mut alive = sample("alive");
        alive.heartbeat_at = 1_000 + 590;
        write_status(dir.path(), &dead).unwrap();
        write_status(dir.path(), &alive).unwrap();
        std::fs::write(dir.path().join("broken.json"), "{").unwrap();
        assert_eq!(prune_dead(dir.path(), 1_000 + 601, 600), 1);
        assert!(!status_path(dir.path(), "dead").exists());
        assert!(status_path(dir.path(), "alive").exists());
        assert!(
            dir.path().join("broken.json").exists(),
            "a file that cannot be read is not judged dead"
        );
    }

    #[test]
    fn removing_a_status_deletes_only_that_session() {
        let dir = tempfile::tempdir().unwrap();
        write_status(dir.path(), &sample("a")).unwrap();
        write_status(dir.path(), &sample("b")).unwrap();
        remove_status(dir.path(), "a");
        assert!(!status_path(dir.path(), "a").exists());
        assert!(status_path(dir.path(), "b").exists());
    }
}
