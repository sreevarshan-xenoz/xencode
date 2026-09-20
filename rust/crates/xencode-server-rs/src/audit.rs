//! Append-only JSONL mirror of the workspace audit trail.
//!
//! The trail itself lives in `WorkspaceManager` (in memory, ordered by a
//! monotonic `seq`); this sink persists every event it has not seen yet to
//! disk, one JSON object per line. Losing audit writes must never take down
//! the session plane, so the first IO error warns once and disables the
//! sink — the in-memory trail stays complete either way.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use tracing::warn;
use xencode_collaboration_rs::{AuditEvent, WorkspaceManager};

struct State {
    /// `Some(None)` means the sink gave up after an IO error; `None` means
    /// no file was ever configured (or it is not open yet).
    file: Option<Option<File>>,
    path: Option<PathBuf>,
    /// Highest `seq` already written, so wiring is one `sync_from` call
    /// per mutation site and restarts append without duplicates.
    last_seq: u64,
}

/// Persists `WorkspaceManager` audit events to a JSONL file.
pub struct AuditSink {
    state: Mutex<State>,
}

impl AuditSink {
    /// A sink that does no IO at all — the default for tests and for
    /// servers started with `--audit-path none`.
    pub fn disabled() -> Self {
        Self {
            state: Mutex::new(State {
                file: None,
                path: None,
                last_seq: 0,
            }),
        }
    }

    /// A sink appending to `path`, created on first write.
    pub fn to_file(path: impl Into<PathBuf>) -> Self {
        Self {
            state: Mutex::new(State {
                file: None,
                path: Some(path.into()),
                last_seq: 0,
            }),
        }
    }

    pub fn is_enabled(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.path.is_some() && !matches!(&state.file, Some(None))
    }

    /// Number of events this sink has written.
    pub fn written(&self) -> u64 {
        self.state.lock().unwrap().last_seq
    }

    /// Mirror every event in the manager's trail newer than the last one
    /// this sink wrote. Safe to call after any mutation; cheap no-op
    /// otherwise.
    pub fn sync_from(&self, workspaces: &WorkspaceManager) {
        let events: Vec<AuditEvent> = workspaces
            .audit_log()
            .iter()
            .filter(|e| e.seq > self.state.lock().unwrap().last_seq)
            .cloned()
            .collect();
        if events.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        let wrote_all = events
            .iter()
            .all(|event| Self::write_line(&mut state, event));
        if wrote_all {
            state.last_seq = events[events.len() - 1].seq;
        }
    }

    /// Returns false once the sink is dead. The lock is held across the
    /// write on purpose: audit lines must not interleave mid-record.
    fn write_line(state: &mut State, event: &AuditEvent) -> bool {
        let Some(path) = state.path.clone() else {
            return false;
        };
        if matches!(&state.file, Some(None)) {
            return false;
        }
        let opened = match &mut state.file {
            Some(f) => f.as_mut().map(|f| f as &mut dyn Write),
            None => match OpenOptions::new().append(true).create(true).open(&path) {
                Ok(file) => {
                    state.file = Some(Some(file));
                    state
                        .file
                        .as_mut()
                        .unwrap()
                        .as_mut()
                        .map(|f| f as &mut dyn Write)
                }
                Err(e) => {
                    warn!("audit sink disabled: cannot open {}: {e}", path.display());
                    state.file = Some(None);
                    None
                }
            },
        };
        let Some(file) = opened else {
            return false;
        };
        let line = match serde_json::to_string(event) {
            Ok(line) => line,
            Err(e) => {
                warn!(
                    "audit sink disabled: cannot serialize event {}: {e}",
                    event.seq
                );
                state.file = Some(None);
                return false;
            }
        };
        match file
            .write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
        {
            Ok(()) => true,
            Err(e) => {
                warn!(
                    "audit sink disabled: write to {} failed: {e}",
                    path.display()
                );
                state.file = Some(None);
                false
            }
        }
    }
}

impl Default for AuditSink {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager_with_trail() -> WorkspaceManager {
        let mut wm = WorkspaceManager::new();
        wm.create_workspace_with_id("s1", "session", "alice");
        wm.add_member("s1", "alice", "bob", xencode_collaboration_rs::Role::Editor)
            .unwrap();
        wm
    }

    fn read_lines(path: &std::path::Path) -> Vec<serde_json::Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).expect("each audit line must be valid JSON"))
            .collect()
    }

    #[test]
    fn a_file_sink_writes_one_json_line_per_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let sink = AuditSink::to_file(&path);

        sink.sync_from(&manager_with_trail());

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 2, "created + member_added, one line each");
        assert_eq!(lines[0]["action"], "workspace_created");
        assert_eq!(lines[1]["action"], "member_added");
        assert_eq!(lines[1]["actor"], "alice");
    }

    /// The restart story: a fresh sink on the same file appends rather
    /// than truncates, so history survives the server going away.
    #[test]
    fn a_second_sink_appends_to_the_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");

        AuditSink::to_file(&path).sync_from(&manager_with_trail());
        let mut wm2 = WorkspaceManager::new();
        wm2.create_workspace_with_id("s2", "later", "carol");
        AuditSink::to_file(&path).sync_from(&wm2);

        let lines = read_lines(&path);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2]["target"], "s2");
    }

    #[test]
    fn a_sink_does_not_rewrite_events_it_already_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let sink = AuditSink::to_file(&path);
        let wm = manager_with_trail();

        sink.sync_from(&wm);
        sink.sync_from(&wm);
        sink.sync_from(&wm);

        assert_eq!(read_lines(&path).len(), 2);
        assert_eq!(sink.written(), 2);
    }

    /// An impossible path must degrade to a warning, not a panic — and
    /// after that the sink stays quiet.
    #[test]
    fn an_unwritable_path_disables_the_sink_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        // A directory where a file should be: every open fails.
        let blocked = dir.path().join("blocked.jsonl");
        std::fs::create_dir(&blocked).unwrap();
        let sink = AuditSink::to_file(&blocked);
        assert!(sink.is_enabled());

        sink.sync_from(&manager_with_trail());

        assert!(
            !sink.is_enabled(),
            "the failed open should disable the sink"
        );
        assert_eq!(sink.written(), 0);
        // Still no panic on further events.
        sink.sync_from(&manager_with_trail());
    }

    #[test]
    fn a_disabled_sink_does_no_io() {
        let sink = AuditSink::disabled();
        assert!(!sink.is_enabled());

        sink.sync_from(&manager_with_trail());

        assert_eq!(sink.written(), 0);
        assert!(!sink.is_enabled());
    }

    /// The event the sink persists must survive the sink's own serde
    /// round trip: `seq` and `at` included, since auditors sort by one and
    /// cross-reference the other.
    #[test]
    fn persisted_lines_carry_the_full_event() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit.jsonl");
        let sink = AuditSink::to_file(&path);
        let wm = manager_with_trail();

        sink.sync_from(&wm);

        let lines = read_lines(&path);
        let in_memory = wm.audit_log();
        for (line, event) in lines.iter().zip(in_memory) {
            assert_eq!(line["seq"], event.seq);
            assert_eq!(line["at"], event.at);
            assert_eq!(line["actor"], event.actor.to_string());
            assert_eq!(line["target"], event.target);
            assert!(line["detail"].is_string());
        }
        assert!(
            lines
                .windows(2)
                .all(|w| w[0]["seq"].as_u64() < w[1]["seq"].as_u64()),
            "seq must be strictly increasing on disk"
        );
    }
}
