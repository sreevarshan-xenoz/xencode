//! The terminal app's side of the live status feed (DK-1): one file per
//! session, redacted before it is written, refreshed every five seconds, and
//! removed when the session ends.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use xencode_live_rs::{LiveSource, LiveState, LiveStatus};

const HEARTBEAT: Duration = Duration::from_secs(5);

pub struct LiveFeed {
    dir: PathBuf,
    status: LiveStatus,
    last_write: Option<Instant>,
}

impl LiveFeed {
    pub fn new(dir: PathBuf, session_id: String, project: &Path) -> LiveFeed {
        let project = std::fs::canonicalize(project)
            .unwrap_or_else(|_| project.to_path_buf())
            .to_string_lossy()
            .trim_start_matches(r"\\?\")
            .replace('\\', "/");
        let now = xencode_live_rs::now_secs();
        // Sessions that were killed never removed their own file; ten minutes
        // without a heartbeat is the same limit the badge forgets them at.
        xencode_live_rs::prune_dead(&dir, now, 600);
        LiveFeed {
            dir,
            status: LiveStatus {
                version: xencode_live_rs::VERSION,
                session_id,
                pid: std::process::id(),
                project,
                model: String::new(),
                source: LiveSource::Chat,
                state: LiveState::Idle,
                headline: String::new(),
                changed_at: now,
                heartbeat_at: now,
            },
            last_write: None,
        }
    }

    pub fn path(&self) -> PathBuf {
        xencode_live_rs::status_path(&self.dir, &self.status.session_id)
    }

    /// Record a state. The headline is redacted and clipped here, so no caller
    /// can write a secret by forgetting to.
    pub fn set(&mut self, state: LiveState, source: LiveSource, headline: &str, model: &str) {
        let headline =
            xencode_live_rs::clip_headline(&xencode_context_rs::redact_secrets(headline));
        let changed = self.status.state != state
            || self.status.source != source
            || self.status.headline != headline
            || self.status.model != model;
        if !changed && self.last_write.is_some() {
            return;
        }
        let now = xencode_live_rs::now_secs();
        self.status.state = state;
        self.status.source = source;
        self.status.headline = headline;
        self.status.model = model.to_string();
        self.status.changed_at = now;
        self.write(now);
    }

    /// Called every main-loop iteration; writes at most every five seconds.
    pub fn heartbeat(&mut self) {
        if self.last_write.is_some_and(|t| t.elapsed() < HEARTBEAT) {
            return;
        }
        self.write(xencode_live_rs::now_secs());
    }

    fn write(&mut self, now: u64) {
        self.status.heartbeat_at = now;
        // A full disk or a read-only state folder costs the badge, never the
        // session, so a failed write is not reported to the person.
        let _ = xencode_live_rs::write_status(&self.dir, &self.status);
        self.last_write = Some(Instant::now());
    }
}

impl Drop for LiveFeed {
    fn drop(&mut self) {
        xencode_live_rs::remove_status(&self.dir, &self.status.session_id);
    }
}
