//! Real-time workspace watcher (M4) — proactive warnings for loaded context.
//!
//! [`stale`] only reports drift when something asks it to. This module turns the
//! OS file events into a first-class push: it watches the workspace root with
//! the `notify` crate, ignores VCS/build/cache directories, folds OS event kinds
//! into the coarse [`WatchKind`] tri-state, and debounces bursts so an editor
//! save (which fires several events in a few milliseconds) coalesces into one
//! [`WatchEvent`].
//!
//! The TUI/CLI drains [`WatcherSession::next_batch`] on a background task; the
//! returned list feeds the same `⚠ ...` proactive warnings [`stale`] emits, but
//! without waiting for a model turn to observe them.

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

/// Directories that are never reported to callers. Mirrors the TUI's scan
/// defaults plus the usual build/cache noise.
pub const DEFAULT_EXCLUDED_DIRS: &[&str] = &[
    ".git",
    ".xencode",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
    ".pytest_cache",
];

/// How long the channel must stay silent before a queued batch is emitted.
pub const DEFAULT_DEBOUNCE: Duration = Duration::from_millis(250);

/// Upper bound on a caller's quiet window. A long window is not a preference:
/// it delays every warning by that much without coalescing anything extra,
/// because the events from one save arrive within a few milliseconds.
pub const MAX_QUIET_WINDOW: Duration = Duration::from_millis(1000);

/// How long a batch may be held while events keep arriving before it is
/// reported anyway. Without this, a `git checkout` or a build that touches
/// files faster than the quiet window starves the caller: the watcher sees
/// constant activity and reports nothing until the workspace goes quiet.
pub const MAX_STORM_LATENCY: Duration = Duration::from_millis(2000);

/// Coarse event tri-state — a normalized version of `notify::EventKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WatchKind {
    Created,
    Modified,
    Removed,
}

/// A normalized, debounced filesystem event. `path` is workspace-relative
/// (always `/`-separated, matching how the rest of the crate keys files).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchEvent {
    pub path: String,
    pub kind: WatchKind,
}

/// Maps a raw `notify` event kind onto the coarse tri-state. Platform noise
/// (e.g. macOS emitting `Access` alongside `Modify`, atomic-rename temp files)
/// intentionally collapses into [`WatchKind::Modified`] — callers only care
/// that *something about the file changed*.
pub fn map_kind(kind: &notify::EventKind) -> WatchKind {
    use notify::EventKind::*;
    match kind {
        Create(_) => WatchKind::Created,
        Remove(_) => WatchKind::Removed,
        Modify(_) | Access(_) | Other | Any => WatchKind::Modified,
    }
}

/// True when `path` (repo-relative, `/`-separated) lives inside an excluded
/// directory. Matches any path segment, so a nested `node_modules` is skipped
/// the same way the toplevel one is.
pub fn should_ignore(path: &str, excluded: &[impl AsRef<str>]) -> bool {
    if excluded.is_empty() {
        return false;
    }
    path.split('/')
        .any(|seg| excluded.iter().any(|d| d.as_ref() == seg))
}

/// Debounce accumulator. Raw events are merged by path until the channel goes
/// quiet for [`DEFAULT_DEBOUNCE`]; then the caller drains the survivors.
#[derive(Debug, Default)]
pub struct Debounce {
    pending: BTreeMap<String, WatchKind>,
}

impl Debounce {
    /// Fold one raw event in. When a torn editor save reports `Removed` then
    /// `Created` for the same path inside one debounce window, the net result is
    /// [`WatchKind::Modified`] — the file is still there and it changed.
    pub fn push(&mut self, event: WatchEvent) {
        if event.kind == WatchKind::Created
            && self.pending.get(&event.path) == Some(&WatchKind::Removed)
        {
            self.pending.insert(event.path, WatchKind::Modified);
        } else {
            self.pending.insert(event.path, event.kind);
        }
    }

    /// Take everything accumulated so far, oldest path first.
    pub fn drain(&mut self) -> Vec<WatchEvent> {
        std::mem::take(&mut self.pending)
            .into_iter()
            .map(|(path, kind)| WatchEvent { path, kind })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

/// The pull side of the watcher. Wrap `notify`'s push channel and sit on a
/// background task (the CLI/TUI already spawns one per headed feature).
pub struct WatcherSession {
    rx: Receiver<WatchEvent>,
    debounce: Debounce,
}

impl WatcherSession {
    /// Block up to `quiet_for`. Returns the debounced batch the moment the
    /// channel has been silent that long (empty when nothing arrived). A busy
    /// workspace cannot postpone this forever: `quiet_for` is capped at
    /// [`MAX_QUIET_WINDOW`], and a batch that has been growing for
    /// [`MAX_STORM_LATENCY`] is reported even while events are still arriving.
    pub fn next_batch(&mut self, quiet_for: Duration) -> Vec<WatchEvent> {
        let quiet_for = quiet_for.min(MAX_QUIET_WINDOW);
        let deadline = Instant::now() + MAX_STORM_LATENCY;
        loop {
            match self.rx.recv_timeout(quiet_for) {
                Ok(event) => {
                    self.debounce.push(event);
                    if Instant::now() >= deadline {
                        return self.debounce.drain();
                    }
                }
                Err(_) => return self.debounce.drain(),
            }
        }
    }
}

/// The full watcher: owns the `notify` watcher (kept alive for the session's
/// lifetime) plus the drained, debounced event stream.
pub struct WorkspaceWatcher {
    session: WatcherSession,
    _watcher: RecommendedWatcher,
}

impl WorkspaceWatcher {
    /// Start watching `root` recursively. `excluded` defaults to
    /// [`DEFAULT_EXCLUDED_DIRS`] when empty.
    pub fn spawn(root: &Path, excluded: &[&str]) -> Result<WorkspaceWatcher, notify::Error> {
        let excluded: Vec<String> = if excluded.is_empty() {
            DEFAULT_EXCLUDED_DIRS
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            excluded.iter().map(|s| s.to_string()).collect()
        };
        let (tx, rx): (Sender<WatchEvent>, Receiver<WatchEvent>) = mpsc::channel();
        let root = root.to_path_buf();
        let tx_root = root.clone();
        let tx_excluded = excluded.clone();

        let mut watcher =
            notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                let Ok(event) = res else { return };
                let Some(path) = event
                    .paths
                    .iter()
                    .map(|p| p.strip_prefix(&tx_root).unwrap_or(p))
                    .find(|rel| !should_ignore(&rel.to_string_lossy(), &tx_excluded))
                else {
                    return;
                };
                let _ = tx.send(WatchEvent {
                    path: path.to_string_lossy().replace('\\', "/"),
                    kind: map_kind(&event.kind),
                });
            })?;
        watcher.watch(&root, RecursiveMode::Recursive)?;

        Ok(WorkspaceWatcher {
            session: WatcherSession {
                rx,
                debounce: Debounce::default(),
            },
            _watcher: watcher,
        })
    }

    /// Drain one debounced batch (see [`WatcherSession::next_batch`]).
    pub fn next_batch(&mut self, quiet_for: Duration) -> Vec<WatchEvent> {
        self.session.next_batch(quiet_for)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::time::Instant;

    fn temp_dir() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-watch-test-{unique}"))
    }

    #[test]
    fn maps_notify_kinds_to_coarse_tristate() {
        use notify::event::{CreateKind, ModifyKind, RemoveKind};
        use notify::EventKind;
        assert_eq!(
            map_kind(&EventKind::Create(CreateKind::File)),
            WatchKind::Created
        );
        assert_eq!(
            map_kind(&EventKind::Modify(ModifyKind::Name(
                notify::event::RenameMode::Any
            ))),
            WatchKind::Modified
        );
        assert_eq!(
            map_kind(&EventKind::Remove(RemoveKind::File)),
            WatchKind::Removed
        );
    }

    #[test]
    fn ignore_skips_excluded_segments() {
        let ex = DEFAULT_EXCLUDED_DIRS;
        assert!(should_ignore(".git/config", ex));
        assert!(should_ignore("node_modules/x/y.js", ex));
        assert!(should_ignore("src/__pycache__/m.pyc", ex));
        assert!(should_ignore("target/debug/xencode", ex));
        assert!(!should_ignore("src/main.rs", ex));
        assert!(!should_ignore("README.md", ex));
        // no exclusions → nothing is ignored
        assert!(!should_ignore("target/x", &config_empty()));
    }

    fn config_empty() -> Vec<&'static str> {
        Vec::new()
    }

    #[test]
    fn debounce_merges_removed_then_created_into_modified() {
        let mut d = Debounce::default();
        d.push(WatchEvent {
            path: "src/a.rs".into(),
            kind: WatchKind::Removed,
        });
        d.push(WatchEvent {
            path: "src/a.rs".into(),
            kind: WatchKind::Created,
        });
        d.push(WatchEvent {
            path: "src/b.rs".into(),
            kind: WatchKind::Modified,
        });
        let batch = d.drain();
        assert_eq!(batch.len(), 2);
        let a = batch.iter().find(|e| e.path == "src/a.rs").unwrap();
        assert_eq!(a.kind, WatchKind::Modified);
        assert!(d.is_empty());
    }

    #[test]
    fn a_storm_is_reported_instead_of_waiting_for_quiet() {
        // Events keep arriving far faster than the quiet window, which is what
        // a build or a large checkout looks like from here. The old behaviour
        // was to coalesce forever and report nothing.
        let (tx, rx) = mpsc::channel();
        let mut session = WatcherSession {
            rx,
            debounce: Debounce::default(),
        };
        std::thread::spawn(move || {
            for i in 0..200 {
                if tx
                    .send(WatchEvent {
                        path: format!("src/gen_{i}.rs"),
                        kind: WatchKind::Modified,
                    })
                    .is_err()
                {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        });

        let started = Instant::now();
        let batch = session.next_batch(Duration::from_millis(50));
        let elapsed = started.elapsed();

        assert!(!batch.is_empty(), "storm produced no events at all");
        assert!(
            elapsed < Duration::from_millis(3_500),
            "waited {elapsed:?} for a batch while events were still arriving; \
             the storm deadline never fired"
        );
    }

    #[test]
    fn an_overlong_quiet_window_is_capped() {
        // A caller asking for a 10 s window must not hold a change for 10 s:
        // bursts arrive within milliseconds, so the extra time only delays the
        // warning without merging anything additional.
        // The sender is deliberately kept alive: a closed channel returns
        // immediately, which would prove nothing about how long the wait is.
        let (tx, rx) = mpsc::channel();
        let held_open = tx.clone();
        let mut session = WatcherSession {
            rx,
            debounce: Debounce::default(),
        };
        tx.send(WatchEvent {
            path: "src/a.rs".into(),
            kind: WatchKind::Modified,
        })
        .unwrap();

        let started = Instant::now();
        let batch = session.next_batch(Duration::from_secs(10));
        let elapsed = started.elapsed();

        assert_eq!(batch.len(), 1);
        assert!(
            elapsed < Duration::from_secs(2),
            "waited out the caller's 10 s window ({elapsed:?}) instead of capping \
             the quiet period at {MAX_QUIET_WINDOW:?}"
        );
        drop(held_open);
    }

    #[test]
    fn session_coalesces_a_typical_editor_save() {
        let root = temp_dir();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/a.rs"), "v1").unwrap();

        let mut watcher = WorkspaceWatcher::spawn(&root, DEFAULT_EXCLUDED_DIRS).unwrap();

        // A save emits several events almost simultaneously; they must collapse
        // into one batched watch event for the same path.
        let a = root.join("src/a.rs");
        for _ in 0..3 {
            fs::write(&a, "v2").unwrap();
        }

        let deadline = Instant::now() + Duration::from_secs(5);
        let batch = loop {
            let batch = watcher.next_batch(Duration::from_millis(50));
            if !batch.is_empty() {
                break batch;
            }
            assert!(
                Instant::now() < deadline,
                "watcher never delivered an event"
            );
            std::thread::sleep(Duration::from_millis(20));
        };

        let a_events: Vec<_> = batch
            .iter()
            .filter(|e| e.path == "src/a.rs")
            .map(|e| e.kind)
            .collect();
        assert_eq!(a_events, vec![WatchKind::Modified]);

        fs::remove_dir_all(root).unwrap();
    }
}
