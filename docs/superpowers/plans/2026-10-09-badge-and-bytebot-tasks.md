# Floating badge and ByteBot tasks — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every running xencode session publishes its live state to a file, a floating GPUI badge shows the most urgent state across sessions, and ByteBot becomes a queue of tasks with the original Bytebot project's states, a help step, a review step, slash commands and in-panel model switching.

**Architecture:** A new tiny crate, `xencode-live-rs`, owns the status file format, reading, writing, and the badge's pure decision logic. The terminal app writes through it. The badge is a separate Cargo workspace at `rust/badge/` so the main workspace and its CI never compile GPUI. ByteBot task records live in a new TUI module, `bytebot_tasks.rs`, and drive the existing agent loop.

**Tech Stack:** Rust 1.99 (stable), serde/serde_json, tokio, ratatui (existing TUI), GPUI pinned to zed-industries/zed commit `f0b38ac75ea769b75287ff1370242d88b3878c2d`, `notify` 8 for folder watching.

**Spec:** `docs/superpowers/specs/2026-10-09-desktop-badge-and-bytebot-tasks-design.md`

## Global Constraints

- Rust only; no Python, no mocks anywhere, including tests (`AGENTS.md`).
- One plan ID per commit, named in the subject; commit messages in plain English (`AGENTS.md` commit rule).
- Never push.
- Status file: `<state dir>/live/<session id>.json`, `"version": 1`, states exactly `idle`, `working`, `needs_you`, `finished`, `failed`; headline at most 120 characters, redacted with `xencode_context_rs::redact_secrets`, never built from the prompt or model text.
- Heartbeat every 5 seconds; a session is "not responding" after 20 seconds without one, and ignored after 10 minutes.
- Badge urgency order: `needs_you`, `failed`, `working`, `finished`, `idle`. `finished` fades after 60 seconds.
- ByteBot task states exactly: `pending`, `running`, `needs_help`, `needs_review`, `completed`, `cancelled`, `failed`.
- Every `cargo test` run on this machine sets `XCODE_CONFIG_DIR` to a fresh temporary folder, has a hard `timeout -k`, uses `-j 4`, and is followed by a leftover-process check.
- `rust/target` above 20 GiB → `cargo clean`. The badge workspace builds into `rust/badge/target`; measure it too.
- macOS is built in CI only and is never checked off as verified from this machine.

## Review Focus

1. **Two terminal sessions at once** — each must write its own file, and the badge must show both rows, not the last writer's. Test in Task 4 (`two_sessions_are_two_rows`).
2. **A status file from a newer xencode** (`"version": 2`) or a half-written file — the badge must skip it with a reason, never crash or show garbage. Test in Task 1 (`unknown_version_and_broken_json_are_skipped_with_a_reason`).
3. **A secret inside a tool summary** (for example `run_command curl -H "Authorization: Bearer sk-FAKE-NOT-A-REAL-TEST-KEY"`) must never reach the status file. Test in Task 2 (`a_secret_in_a_tool_summary_never_reaches_the_file`).
4. **Undo of a ByteBot task after a later chat turn changed files** — rewinding would undo the wrong turn, so it must refuse and say why. Test in Task 10 (`undo_refuses_when_a_later_turn_changed_files`).
5. **xencode exits while a task waits for help** — the record must not stay `needs_help` forever. Test in Task 8 (`a_task_left_running_or_waiting_is_failed_on_restart`).

---

## File structure

| File | Responsibility |
|---|---|
| `rust/crates/xencode-live-rs/` (new crate) | Status type, atomic write, read-all, live folder path, badge aggregation (pure) |
| `rust/crates/xencode-tui-rs/src/live_status.rs` (new) | The terminal app's writer: redaction, heartbeat, state transitions |
| `rust/crates/xencode-tui-rs/src/bytebot_tasks.rs` (new) | ByteBot task record, states, on-disk store, queue |
| `rust/crates/xencode-tui-rs/src/app.rs` | Hooks into turn lifecycle, `set_model`, `dispatch_prompt`, task queue driving |
| `rust/crates/xencode-tui-rs/src/agent_tools.rs` | `ask_user` tool class and pause |
| `rust/crates/xencode-providers-rs/src/tools.rs` | `ask_user` tool definition |
| `rust/crates/xencode-tui-rs/src/keymap.rs`, `ui.rs` | ByteBot panel keys and drawing |
| `rust/crates/xencode-cli/src/main.rs` | `xencode badge` subcommand |
| `rust/crates/xencode-config-rs/src/config.rs` | `badge_autostart` setting |
| `rust/badge/` (new, own workspace) | GPUI badge binary |

---

### Task 1: `xencode-live-rs` — the status file (DK-1, part 1)

**Files:**
- Create: `rust/crates/xencode-live-rs/Cargo.toml`, `rust/crates/xencode-live-rs/src/lib.rs`
- Modify: `rust/Cargo.toml` (workspace `members`)

**Interfaces:**
- Produces:
  - `pub enum LiveState { Idle, Working, NeedsYou, Finished, Failed }` (serde `snake_case`)
  - `pub enum LiveSource { Chat, Bytebot }` (serde `snake_case`)
  - `pub struct LiveStatus { version: u32, session_id: String, pid: u32, project: String, model: String, source: LiveSource, state: LiveState, headline: String, changed_at: u64, heartbeat_at: u64 }` (all `pub`)
  - `pub const VERSION: u32 = 1;` `pub const HEADLINE_MAX: usize = 120;`
  - `pub fn live_dir() -> Result<PathBuf, String>`
  - `pub fn status_path(dir: &Path, session_id: &str) -> PathBuf`
  - `pub fn write_status(dir: &Path, status: &LiveStatus) -> std::io::Result<()>` (atomic)
  - `pub fn remove_status(dir: &Path, session_id: &str)`
  - `pub enum Read { Ok(LiveStatus), Skipped { file: PathBuf, reason: String } }`
  - `pub fn read_all(dir: &Path) -> Vec<Read>`
  - `pub fn clip_headline(text: &str) -> String`
  - `pub fn now_secs() -> u64`

- [ ] **Step 1: Create the crate and add it to the workspace**

`rust/crates/xencode-live-rs/Cargo.toml`:

```toml
[package]
name = "xencode-live-rs"
version = "0.1.0"
edition = "2021"
description = "Live session status files shared by the xencode terminal app and the floating badge"
license = "MIT"

[dependencies]
xencode-config-rs = { path = "../xencode-config-rs" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
tempfile = "3"
```

Check `edition` and `license` against `rust/crates/xencode-core-rs/Cargo.toml` and copy whatever that crate uses. Add `"crates/xencode-live-rs",` to `members` in `rust/Cargo.toml` after `"crates/xencode-config-rs",`.

- [ ] **Step 2: Write the failing tests** in `src/lib.rs` under `#[cfg(test)] mod tests`:

```rust
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
    fn removing_a_status_deletes_only_that_session() {
        let dir = tempfile::tempdir().unwrap();
        write_status(dir.path(), &sample("a")).unwrap();
        write_status(dir.path(), &sample("b")).unwrap();
        remove_status(dir.path(), "a");
        assert!(!status_path(dir.path(), "a").exists());
        assert!(status_path(dir.path(), "b").exists());
    }
}
```

- [ ] **Step 3: Run them to watch them fail**

Run: `cd rust && XCODE_CONFIG_DIR=$(mktemp -d) timeout -k 10 600 cargo test -p xencode-live-rs -j 4`
Expected: compile errors naming `LiveStatus`, `write_status` and the rest.

- [ ] **Step 4: Implement** above the tests in `src/lib.rs`:

```rust
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

/// One file's outcome. A file that cannot be used is reported, not hidden.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Read {
    Ok(LiveStatus),
    Skipped { file: PathBuf, reason: String },
}

/// Every `*.json` status file in `dir`. A missing folder means no sessions.
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
                out.push(Read::Skipped { file, reason: format!("unreadable: {e}") });
                continue;
            }
        };
        let version = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v.get("version").and_then(|v| v.as_u64()));
        match version {
            Some(v) if v == u64::from(VERSION) => match serde_json::from_str(&text) {
                Ok(status) => out.push(Read::Ok(status)),
                Err(e) => out.push(Read::Skipped { file, reason: format!("not a status file: {e}") }),
            },
            Some(v) => out.push(Read::Skipped {
                file,
                reason: format!("written by a newer xencode (format {v}); update this badge"),
            }),
            None => out.push(Read::Skipped { file, reason: "not a status file".to_string() }),
        }
    }
    out.sort_by(|a, b| key(a).cmp(key(b)));
    out
}

fn key(r: &Read) -> &str {
    match r {
        Read::Ok(s) => &s.session_id,
        Read::Skipped { file, .. } => file.to_str().unwrap_or(""),
    }
}

/// Clip to `HEADLINE_MAX` characters, ending in `…` when clipped.
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: same as Step 3. Expected: 6 passed.

- [ ] **Step 6: Clippy, then commit**

```bash
cd rust && cargo clippy -p xencode-live-rs --all-targets -j 4 && cargo fmt -p xencode-live-rs
git add rust/Cargo.toml rust/Cargo.lock rust/crates/xencode-live-rs
git commit   # subject: "DK-1: a small crate for the live session status files the badge will read"
```

---

### Task 2: the terminal app writes its status (DK-1, part 2)

**Files:**
- Create: `rust/crates/xencode-tui-rs/src/live_status.rs`
- Modify: `rust/crates/xencode-tui-rs/Cargo.toml` (add `xencode-live-rs = { path = "../xencode-live-rs" }`), `src/lib.rs` (`pub mod live_status;`), `src/app.rs`
- Test: `rust/crates/xencode-tui-rs/tests/live_status_feed.rs`

**Interfaces:**
- Consumes: Task 1's `LiveStatus`, `LiveState`, `LiveSource`, `write_status`, `remove_status`, `clip_headline`, `now_secs`.
- Produces:
  - `pub struct LiveFeed` with `pub fn new(dir: PathBuf, session_id: String, project: &Path) -> LiveFeed`
  - `pub fn set(&mut self, state: LiveState, source: LiveSource, headline: &str, model: &str)` — redacts, clips, writes only when state, source or headline changed
  - `pub fn heartbeat(&mut self)` — rewrites when 5 s have passed since the last write
  - `pub fn path(&self) -> PathBuf`
  - `impl Drop for LiveFeed` — removes the file
  - App field `pub live: Option<crate::live_status::LiveFeed>`; `None` in `App::for_tests()`, `Some` in `App::new()`.
  - App helper `pub(crate) fn live_set(&mut self, state: LiveState, source: LiveSource, headline: &str)` — no-op when `live` is `None`.

- [ ] **Step 1: Write the failing integration test** `tests/live_status_feed.rs`:

```rust
//! DK-1: the terminal app's status file follows the turn lifecycle.
use xencode_live_rs::{read_all, LiveState, Read};
use xencode_tui_rs::app::App;
use xencode_tui_rs::live_status::LiveFeed;

fn state_of(dir: &std::path::Path) -> (LiveState, String) {
    match read_all(dir).into_iter().next().expect("a status file") {
        Read::Ok(s) => (s.state, s.headline),
        Read::Skipped { reason, .. } => panic!("{reason}"),
    }
}

fn app_with_feed(dir: &std::path::Path) -> App<'static> {
    let mut app = App::for_tests();
    app.live = Some(LiveFeed::new(dir.to_path_buf(), "s1".into(), std::path::Path::new(".")));
    app
}

#[test]
fn the_file_follows_working_needs_you_finished_and_failed() {
    let dir = tempfile::tempdir().unwrap();
    let mut app = app_with_feed(dir.path());
    app.live_turn_started();
    assert_eq!(state_of(dir.path()).0, LiveState::Working);
    app.live_tool_started("write_file src/auth.rs");
    assert_eq!(state_of(dir.path()).1, "write_file src/auth.rs");
    app.live_approval_waiting("write_file src/auth.rs");
    let (state, headline) = state_of(dir.path());
    assert_eq!(state, LiveState::NeedsYou);
    assert_eq!(headline, "waiting for you to allow: write_file src/auth.rs");
    app.live_turn_ended(None);
    assert_eq!(state_of(dir.path()).0, LiveState::Finished);
    app.live_turn_started();
    app.live_turn_ended(Some("connection refused\nmore detail"));
    let (state, headline) = state_of(dir.path());
    assert_eq!(state, LiveState::Failed);
    assert_eq!(headline, "connection refused");
}

#[test]
fn a_secret_in_a_tool_summary_never_reaches_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut app = app_with_feed(dir.path());
    app.live_tool_started("run_command curl -H \"Authorization: Bearer sk-FAKE-NOT-A-REAL-TEST-KEY\"");
    let text = std::fs::read_to_string(dir.path().join("s1.json")).unwrap();
    assert!(!text.contains("sk-FAKE-NOT-A-REAL-TEST-KEY"), "{text}");
}

#[test]
fn dropping_the_feed_removes_the_file() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut app = app_with_feed(dir.path());
        app.live_turn_started();
        assert!(dir.path().join("s1.json").exists());
    }
    assert!(!dir.path().join("s1.json").exists());
}
```

Add `tempfile` to the TUI crate's dev-dependencies if missing (it is already there) and `xencode-live-rs` to `[dependencies]`.

- [ ] **Step 2: Run to watch it fail**

Run: `cd rust && XCODE_CONFIG_DIR=$(mktemp -d) timeout -k 10 900 cargo test -p xencode-tui-rs -j 4 --test live_status_feed`
Expected: errors for `live_status`, `live`, `live_turn_started`.

- [ ] **Step 3: Implement `src/live_status.rs`**

```rust
//! The terminal app's side of the live status feed (DK-1): one file per
//! session, redacted before it is written, refreshed every five seconds.

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
        let headline = xencode_live_rs::clip_headline(&xencode_context_rs::redact_secrets(headline));
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
        // A full disk or a read-only state folder costs the badge, never the session.
        let _ = xencode_live_rs::write_status(&self.dir, &self.status);
        self.last_write = Some(Instant::now());
    }
}

impl Drop for LiveFeed {
    fn drop(&mut self) {
        xencode_live_rs::remove_status(&self.dir, &self.status.session_id);
    }
}
```

- [ ] **Step 4: Wire it into `App`** (`src/app.rs`)

1. Field, beside `bytebot_stop` (~line 847):
```rust
    /// This session's live status file (DK-1), read by the floating badge.
    /// `None` in tests that do not ask for one.
    pub live: Option<crate::live_status::LiveFeed>,
```
2. In the struct literal of `with_config_and_memory` (where `bytebot_stop: None,` is), add `live: None,`.
3. In `App::new()` (line 2233), after `memory.start_session(None);` keep the returned id and, after `app` is built:
```rust
        let session_id = memory.start_session(None);
        // ... existing lines ...
        if let Ok(dir) = xencode_live_rs::live_dir() {
            let root = std::env::current_dir().unwrap_or_default();
            app.live = Some(crate::live_status::LiveFeed::new(dir, session_id, &root));
        }
```
(`start_session` returns the id: `pub fn start_session(&mut self, session_id: Option<String>) -> String`.)
4. Helpers on `impl App`, next to `push_toast`:
```rust
    pub(crate) fn live_set(&mut self, state: xencode_live_rs::LiveState, source: xencode_live_rs::LiveSource, headline: &str) {
        let model = self.config.default_model.clone();
        if let Some(feed) = self.live.as_mut() {
            feed.set(state, source, headline, &model);
        }
    }
    fn live_source(&self) -> xencode_live_rs::LiveSource {
        if self.bytebot_running { xencode_live_rs::LiveSource::Bytebot } else { xencode_live_rs::LiveSource::Chat }
    }
    pub fn live_turn_started(&mut self) {
        let source = self.live_source();
        self.live_set(xencode_live_rs::LiveState::Working, source, "thinking");
    }
    pub fn live_tool_started(&mut self, summary: &str) {
        let source = self.live_source();
        self.live_set(xencode_live_rs::LiveState::Working, source, summary);
    }
    pub fn live_approval_waiting(&mut self, summary: &str) {
        let source = self.live_source();
        self.live_set(xencode_live_rs::LiveState::NeedsYou, source, &format!("waiting for you to allow: {summary}"));
    }
    /// `error` is `None` for a normal end.
    pub fn live_turn_ended(&mut self, error: Option<&str>) {
        let source = self.live_source();
        match error {
            None => self.live_set(xencode_live_rs::LiveState::Finished, source, "done"),
            Some(e) => self.live_set(xencode_live_rs::LiveState::Failed, source, e),
        }
    }
    pub fn live_stopped(&mut self) {
        let source = self.live_source();
        self.live_set(xencode_live_rs::LiveState::Idle, source, "stopped");
    }
```
5. Call sites (line numbers from the 2026-10-09 code map; search the quoted code if they moved):
   - `submit_message`, right after `self.is_generating = true;` (~3923): `self.live_turn_started();`
   - `arm_bytebot`, right after `self.bytebot_running = true;` (~4893): `self.live_turn_started();`
   - `run_app` drain loop, `[TOOL]` branch (~13044): when `body.starts_with("→ ")`, call `app.live_tool_started(body.trim_start_matches("→ ").trim())`.
   - `bytebot_event` (~5341), `call:` branch: `self.live_tool_started(summary)`.
   - `run_app` approval drain, after `app.approval_queue.push_back((request, responder));` (~13094): `app.live_approval_waiting(&summary)` with `summary` cloned from `request.summary` before the push.
   - `resolve_approval` (~3492), at the end: if the queue is now empty and `is_generating || bytebot_running`, `self.live_set(Working, source, "continuing")`.
   - `TURN_ERROR_PREFIX` branch (~13062): store the body in a new field `pub(crate) live_turn_error: Option<String>` (initialised `None`).
   - `append_generation`, inside `if text == "[DONE]"` (~4490): `let err = self.live_turn_error.take(); self.live_turn_ended(err.as_deref());`
   - `[BYTEBOT_DONE]` branch (~12543): `let failed = app.bytebot_steps.iter().any(|(_, s)| s == "failed"); app.live_turn_ended(failed.then_some("a step failed"));` — replaced by task states in Task 8.
   - `[STOPPED]` branch (~13057): `app.live_stopped();`
   - Main loop, right after `app.say_secret_problems();` (~13306): `if let Some(feed) = app.live.as_mut() { feed.heartbeat(); }`
6. Make `live_turn_started`, `live_tool_started`, `live_approval_waiting`, `live_turn_ended`, `live_stopped` `pub` so the integration test can drive them (they are the lifecycle the TUI already calls).

- [ ] **Step 5: Run the test to verify it passes**

Run: the Step 2 command. Expected: 3 passed.

- [ ] **Step 6: Watch it live**

Build and start the real TUI in Windows Terminal with `XCODE_CONFIG_DIR` pointing at a fresh folder. In a second shell, `type <dir>\state\live\*.json` (check `xencode paths` for the state folder): after typing `/help` the file exists with `idle`; Ctrl+C removes it. Write down what was seen for the commit body.

- [ ] **Step 7: Full TUI suite against the 2026-10-09 baseline (0 failures), clippy, commit**

Subject: `DK-1: each running session writes its live status for the floating badge to read`.

---

### Task 3: `badge_autostart` and `xencode badge` (DK-2, part 1)

**Files:**
- Modify: `rust/crates/xencode-config-rs/src/config.rs`, `rust/crates/xencode-tui-rs/src/focus.rs`, `keymap.rs`, `ui.rs`, `app.rs`, `rust/crates/xencode-cli/src/main.rs`, `rust/crates/xencode-live-rs/src/lib.rs`
- Test: in `xencode-live-rs` (`find_badge`), keymap test for the settings row list.

**Interfaces:**
- Produces:
  - `XencodeConfig.badge_autostart: bool` (serde default false)
  - `xencode_live_rs::BADGE_EXE: &str` = `"xencode-badge"` (+ `.exe` on Windows via `std::env::consts::EXE_SUFFIX`)
  - `pub fn find_badge(beside: Option<&Path>) -> Option<PathBuf>` — next to `beside` first, then `PATH` via `xencode_core_rs::sys::which`
  - `pub fn spawn_badge(exe: &Path) -> std::io::Result<()>` — detached: on Windows `creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)` (`0x00000008 | 0x00000200`), on Unix `process_group(0)`; stdin/stdout/stderr null.
  - CLI `Commands::Badge` → `run_badge()`.

- [ ] **Step 1: Failing tests** in `xencode-live-rs`:

```rust
    #[test]
    fn the_badge_next_to_xencode_is_found_first() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join(format!("xencode-badge{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&exe, b"").unwrap();
        assert_eq!(find_badge(Some(dir.path())), Some(exe));
    }

    #[test]
    fn no_badge_anywhere_is_none() {
        let dir = tempfile::tempdir().unwrap();
        // PATH may hold a real badge on a developer machine; only the folder case is asserted.
        let found = find_badge(Some(dir.path()));
        assert!(found.is_none_or(|p| !p.starts_with(dir.path())));
    }
```
(`xencode-live-rs` gains `xencode-core-rs = { path = "../xencode-core-rs" }` for `sys::which`.)

- [ ] **Step 2: Run to fail; Step 3: implement**

```rust
pub fn badge_exe_name() -> String {
    format!("xencode-badge{}", std::env::consts::EXE_SUFFIX)
}

pub fn find_badge(beside: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = beside {
        let candidate = dir.join(badge_exe_name());
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    xencode_core_rs::sys::which("xencode-badge").ok()
}

/// Start the badge so it outlives this process and owns no console.
pub fn spawn_badge(exe: &Path) -> std::io::Result<()> {
    let mut cmd = std::process::Command::new(exe);
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    cmd.spawn().map(|_| ())
}
```
Check `xencode_core_rs::sys::which`'s exact signature first (`grep -n "pub fn which" rust/crates/xencode-core-rs/src/sys.rs`) and adapt the `.ok()`.

- [ ] **Step 4: Config and Settings row**
  - `config.rs`: after `allow_external_workers` (694–695) add `/// Start the floating badge when the terminal app starts (DK-2).` `#[serde(default)] pub badge_autostart: bool,` and `badge_autostart: false,` in `Default` (~1094).
  - `focus.rs` `SETTINGS_ITEMS`: after the "Mouse Capture" row add `SettingRow { label: "Floating Badge", section: "Display", kind: SettingKind::Toggle },`.
  - `keymap.rs` `settings_toggle` (1288): `"Floating Badge" => &mut app.config.badge_autostart,`.
  - `ui.rs` toggle read (1302–1313): `"Floating Badge" => app.config.badge_autostart,`.
  - Update the Display-section label list asserted in the keymap test (~3398–3407).
  - `App::new()`: after the live feed is created, `if app.config.badge_autostart { app.start_badge(); }` where
```rust
    pub(crate) fn start_badge(&mut self) {
        let beside = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()));
        match xencode_live_rs::find_badge(beside.as_deref()) {
            Some(exe) => {
                if let Err(e) = xencode_live_rs::spawn_badge(&exe) {
                    self.push_toast(crate::toast::ToastKind::Warning, format!("Could not start the badge: {e}"));
                }
            }
            None => self.push_toast(
                crate::toast::ToastKind::Warning,
                "Floating Badge is on but xencode-badge was not found next to xencode or on PATH".to_string(),
            ),
        }
    }
```
  Starting it twice is safe: the badge refuses a second copy (Task 5, Step 4).

- [ ] **Step 5: CLI.** In `Commands` (main.rs 160) add:
```rust
    /// Start the floating badge that shows what running xencode sessions are doing
    Badge,
```
and the arm `Commands::Badge => run_badge(),` with
```rust
fn run_badge() -> Result<(), String> {
    let beside = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let exe = xencode_live_rs::find_badge(beside.as_deref()).ok_or_else(|| {
        "xencode-badge was not found next to xencode or on PATH. Build it with: \
         cargo build --release --manifest-path rust/badge/Cargo.toml"
            .to_string()
    })?;
    xencode_live_rs::spawn_badge(&exe).map_err(|e| format!("could not start {}: {e}", exe.display()))?;
    println!("Started {}", exe.display());
    Ok(())
}
```
Match the return type other arms use (check `Commands::Paths` dispatch at 2297) and add `xencode-live-rs` to the CLI's dependencies.

- [ ] **Step 6: Tests pass, `xencode badge` without a badge built prints the "not found" message (run it), clippy, docs** — `CLI_GUIDE.md` and `README.md` gain `xencode badge` and the Floating Badge setting, verified against `xencode badge --help`. Commit: `DK-2: the xencode badge command and a setting to start the badge with the terminal app`.

---

### Task 4: the badge's decisions, without a window (DK-2, part 2)

**Files:**
- Create: `rust/crates/xencode-live-rs/src/badge.rs` (module `pub mod badge;` in `lib.rs`)

The spec placed this in the badge crate; it lives here instead so its tests run in the main workspace and its CI, which never builds GPUI. Record that deviation in the commit body.

**Interfaces:**
- Produces:
  - `pub enum Shown { Nothing, Working, NeedsYou, Finished, Failed }`
  - `pub struct Row { pub project: String, pub model: String, pub state_words: String, pub headline: String, pub since: String, pub responding: bool }`
  - `pub struct BadgeView { pub shown: Shown, pub stale_dot: bool, pub rows: Vec<Row>, pub skipped: Vec<String> }`
  - `pub fn view(reads: &[Read], now: u64) -> BadgeView`
  - consts `STALE_AFTER: u64 = 20`, `FORGET_AFTER: u64 = 600`, `FINISHED_FADES_AFTER: u64 = 60`

- [ ] **Step 1: Failing tests** (in `badge.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LiveSource, LiveState, LiveStatus, Read, VERSION};

    fn s(id: &str, state: LiveState, changed: u64, beat: u64) -> Read {
        Read::Ok(LiveStatus {
            version: VERSION, session_id: id.into(), pid: 1, project: format!("E:/{id}"),
            model: "m".into(), source: LiveSource::Chat, state, headline: format!("{id} doing"),
            changed_at: changed, heartbeat_at: beat,
        })
    }

    #[test]
    fn the_most_urgent_state_wins() {
        let now = 1000;
        let reads = [s("a", LiveState::Working, 990, 999), s("b", LiveState::NeedsYou, 990, 999), s("c", LiveState::Failed, 990, 999)];
        assert_eq!(view(&reads, now).shown, Shown::NeedsYou);
        let reads = [s("a", LiveState::Working, 990, 999), s("c", LiveState::Failed, 990, 999)];
        assert_eq!(view(&reads, now).shown, Shown::Failed);
    }

    #[test]
    fn two_sessions_are_two_rows() {
        let v = view(&[s("a", LiveState::Working, 990, 999), s("b", LiveState::Idle, 990, 999)], 1000);
        assert_eq!(v.rows.len(), 2);
        assert_eq!(v.rows[0].state_words, "working");
        assert_eq!(v.rows[1].state_words, "idle");
    }

    #[test]
    fn finished_fades_after_a_minute_but_its_row_stays() {
        let v = view(&[s("a", LiveState::Finished, 1000 - FINISHED_FADES_AFTER - 1, 999)], 1000);
        assert_eq!(v.shown, Shown::Nothing);
        assert_eq!(v.rows[0].state_words, "finished");
        let v = view(&[s("a", LiveState::Finished, 990, 999)], 1000);
        assert_eq!(v.shown, Shown::Finished);
    }

    #[test]
    fn a_silent_session_is_not_responding_then_forgotten() {
        let v = view(&[s("a", LiveState::Working, 900, 1000 - STALE_AFTER - 1)], 1000);
        assert!(v.stale_dot);
        assert!(!v.rows[0].responding);
        assert_eq!(v.rows[0].state_words, "not responding");
        assert_eq!(v.shown, Shown::Nothing, "a dead session is not 'working'");
        let v = view(&[s("a", LiveState::Working, 0, 1000 - FORGET_AFTER - 1)], 1000);
        assert!(v.rows.is_empty());
    }

    #[test]
    fn skipped_files_are_reported_in_words() {
        let reads = [Read::Skipped { file: "x.json".into(), reason: "written by a newer xencode (format 2); update this badge".into() }];
        let v = view(&reads, 1000);
        assert_eq!(v.shown, Shown::Nothing);
        assert_eq!(v.skipped, vec!["x.json: written by a newer xencode (format 2); update this badge".to_string()]);
    }

    #[test]
    fn since_is_written_in_minutes_and_seconds() {
        assert_eq!(since(65), "1m 5s ago");
        assert_eq!(since(5), "5s ago");
        assert_eq!(since(7300), "2h 1m ago");
    }
}
```

- [ ] **Step 2: Run to fail. Step 3: implement**

```rust
//! What the floating badge shows (DK-2), decided without a window so it can be
//! tested anywhere.

use crate::{LiveState, Read};

pub const STALE_AFTER: u64 = 20;
pub const FORGET_AFTER: u64 = 600;
pub const FINISHED_FADES_AFTER: u64 = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Shown { Nothing, Working, NeedsYou, Finished, Failed }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub project: String,
    pub model: String,
    pub state_words: String,
    pub headline: String,
    pub since: String,
    pub responding: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadgeView {
    pub shown: Shown,
    pub stale_dot: bool,
    pub rows: Vec<Row>,
    pub skipped: Vec<String>,
}

/// Urgency: needs you, failed, working, finished, then nothing.
fn rank(shown: Shown) -> u8 {
    match shown {
        Shown::NeedsYou => 4,
        Shown::Failed => 3,
        Shown::Working => 2,
        Shown::Finished => 1,
        Shown::Nothing => 0,
    }
}

pub fn view(reads: &[Read], now: u64) -> BadgeView {
    let mut v = BadgeView { shown: Shown::Nothing, stale_dot: false, rows: Vec::new(), skipped: Vec::new() };
    for r in reads {
        let s = match r {
            Read::Ok(s) => s,
            Read::Skipped { file, reason } => {
                v.skipped.push(format!("{}: {reason}", file.display()));
                continue;
            }
        };
        let silent_for = now.saturating_sub(s.heartbeat_at);
        if silent_for > FORGET_AFTER {
            continue;
        }
        let responding = silent_for <= STALE_AFTER;
        let age = now.saturating_sub(s.changed_at);
        let shown = if !responding {
            v.stale_dot = true;
            Shown::Nothing
        } else {
            match s.state {
                LiveState::Idle => Shown::Nothing,
                LiveState::Working => Shown::Working,
                LiveState::NeedsYou => Shown::NeedsYou,
                LiveState::Failed => Shown::Failed,
                LiveState::Finished if age <= FINISHED_FADES_AFTER => Shown::Finished,
                LiveState::Finished => Shown::Nothing,
            }
        };
        if rank(shown) > rank(v.shown) {
            v.shown = shown;
        }
        let state_words = if responding {
            match s.state {
                LiveState::Idle => "idle",
                LiveState::Working => "working",
                LiveState::NeedsYou => "needs you",
                LiveState::Finished => "finished",
                LiveState::Failed => "failed",
            }
        } else {
            "not responding"
        };
        v.rows.push(Row {
            project: s.project.clone(),
            model: s.model.clone(),
            state_words: state_words.to_string(),
            headline: s.headline.clone(),
            since: since(age),
            responding,
        });
    }
    v
}

pub fn since(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s ago"),
        60..=3599 => format!("{}m {}s ago", secs / 60, secs % 60),
        _ => format!("{}h {}m ago", secs / 3600, (secs % 3600) / 60),
    }
}
```

Note: the spec says "failed stays until hovered". The view reports `Failed` while the session's state is `failed`; clearing it on hover is window state, done in Task 5 (`acknowledged_failures`).

- [ ] **Step 4: Pass, clippy, commit** — `DK-2: what the badge shows, decided and tested without a window`.

---

### Task 5: the GPUI badge window (DK-2, part 3)

**Files:**
- Create: `rust/badge/Cargo.toml`, `rust/badge/src/main.rs`, `rust/badge/src/view.rs`, `rust/badge/README.md`
- Modify: `rust/Cargo.toml` (`exclude = ["badge"]` under `[workspace]`), `.github/workflows/ci.yml` (new job), `.gitignore` if `rust/badge/target` is not already covered.

**Interfaces:**
- Consumes: `xencode_live_rs::{live_dir, read_all, now_secs, badge::{view, BadgeView, Shown}}`.

- [ ] **Step 1: Feasibility first — a transparent, always-on-top 40-pixel window**

`rust/badge/Cargo.toml`:
```toml
[package]
name = "xencode-badge"
version = "0.1.0"
edition = "2021"
publish = false

[workspace]

[dependencies]
gpui = { git = "https://github.com/zed-industries/zed", rev = "f0b38ac75ea769b75287ff1370242d88b3878c2d" }
gpui_platform = { git = "https://github.com/zed-industries/zed", rev = "f0b38ac75ea769b75287ff1370242d88b3878c2d", features = ["font-kit"] }
xencode-live-rs = { path = "../crates/xencode-live-rs" }
notify = "8"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[profile.dev]
debug = "line-tables-only"

[profile.dev.package."*"]
debug = false
```

`src/main.rs` (spike version):
```rust
use gpui::{div, prelude::*, px, rgb, size, App, Bounds, Context, Window, WindowBackgroundAppearance, WindowBounds, WindowKind, WindowOptions};

struct Dot;
impl Render for Dot {
    fn render(&mut self, _w: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size(px(40.)).rounded_full().bg(rgb(0x3b82f6))
    }
}

fn main() {
    gpui_platform::application().run(|cx: &mut App| {
        let display = cx.primary_display().expect("a display");
        let screen = display.bounds();
        let origin = gpui::point(screen.origin.x + screen.size.width - px(48.), screen.origin.y + screen.size.height / 2.);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds { origin, size: size(px(40.), px(40.)) })),
                titlebar: None,
                kind: WindowKind::PopUp,
                is_movable: true,
                is_resizable: false,
                is_minimizable: false,
                focus: false,
                window_background: WindowBackgroundAppearance::Transparent,
                ..Default::default()
            },
            |_, cx| cx.new(|_| Dot),
        )
        .expect("the badge window");
    });
}
```

Run: `cd rust/badge && timeout -k 10 3600 cargo build -j 4` (first build fetches the Zed repository; expect a long build), then `./target/debug/xencode-badge.exe`.
Expected: a blue dot at the right edge, above other windows, with no frame or background.

**Stop rule (from the spec):** if this does not build with the GNU toolchain, try once with `cargo +stable-x86_64-pc-windows-msvc build` (Visual Studio 2022 Build Tools are installed). If GPUI cannot show a transparent always-on-top window on Windows, stop the task and report to the owner before considering any other GUI library. Measure `du -sh rust/badge/target` after the build.

API names above (`primary_display`, `display.bounds()`, `rounded_full`, `point`) were read from GPUI's source on 2026-10-09; if the pinned revision differs, read `crates/gpui/examples/window_positioning.rs` in the fetched checkout (`~/.cargo/git/checkouts/zed-*/f0b38ac/`) and follow it.

- [ ] **Step 2: Commit the spike as the starting point** — `DK-2: a transparent always-on-top badge window builds and shows on Windows` (body says what was seen, the toolchain used and the target folder size).

- [ ] **Step 3: The real view** (`src/view.rs`): a `Badge` entity holding `BadgeView`, `hovered: bool`, `acknowledged: HashSet<String>` (projects whose failure was hovered), `spin: f32`.
  - Resting: 40-pixel circle. Fill by `Shown`: Nothing `0x4b5563` at 60 % opacity; Working `0x3b82f6` with a ring drawn as a thicker border whose colour rotates by `spin`; NeedsYou `0xf59e0b`, opacity pulsing between 0.7 and 1.0; Finished `0x22c55e` with a "✓" label; Failed `0xef4444` with "!" until hovered; `stale_dot` adds an 8-pixel grey dot at the top right.
  - Hover (`on_hover`): resize the window to 360 × (48 + 56 × rows) pixels, anchored to the edge, and draw one row per `Row`: project (bold), `state_words`, model, headline, `since`. When `rows` is empty: "No xencode session is running." `skipped` entries are listed in grey at the bottom. Leaving the card shrinks it back. Hovering marks every failed project in `acknowledged`, and `Shown::Failed` is drawn as `Nothing` when all failed rows are acknowledged.
  - Refresh: a `notify` watcher on `live_dir()` sends a message to the app on any change; also a 2-second timer (`cx.spawn` + `Timer::after`). Each refresh is `view(&read_all(&dir), now_secs())`. Animation frames run only while Working or NeedsYou is shown.
  - Drag: the window is movable; on mouse up, snap to the nearest screen edge and save `{ "edge": "right", "offset": 0.42 }` to `<settings dir>/badge.json` (`xencode_config_rs::paths::settings_dir()`), read back on start.

- [ ] **Step 4: One badge per user.** At start, open `<live dir>/badge.lock` and call `File::try_lock()` (stable since Rust 1.89). If it fails, exit 0 silently. Keep the file handle alive for the process lifetime.

- [ ] **Step 5: Watch every state on this machine.** Start the badge, then a real TUI session with a scripted or real model: idle (dim), a turn running (ring), an approval waiting (amber pulse), the turn ending (green, fading after a minute), a turn failing against a stopped server (red until hovered), killing the TUI with Task Manager (grey dot after 20 s, gone after 10 min), and two TUI sessions at once (two rows). Run `xencode badge` twice and confirm one badge. Write down each observation.

- [ ] **Step 6: CI job** in `.github/workflows/ci.yml`:
```yaml
  badge:
    name: badge (${{ matrix.os }})
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --manifest-path rust/badge/Cargo.toml
```
Copy the `actions/checkout` and toolchain step versions the existing jobs use.

- [ ] **Step 7: Docs and commit** — `rust/badge/README.md` (build, run, what each look means), README and USER_MANUAL sections, CHANGELOG entry. Plan file: DK-1 and DK-2 checked with what was watched; the macOS line stays unchecked. Commit: `DK-2: the floating badge shows every session's state and a card on hover`.

---

### Task 6: one way to change model, `/model`, and the panel list (BT-5)

**Files:** `rust/crates/xencode-tui-rs/src/app.rs`, `keymap.rs`, `ui.rs`, `help.rs`; test in `app.rs` tests.

**Interfaces:**
- Produces: `pub fn set_model(&mut self, model: &str, tx: mpsc::UnboundedSender<String>)`; App fields `pub bytebot_model_picker: bool`, `pub bytebot_model_selected: usize`; `"/model"` in `SLASH_COMMANDS` and help `COMMANDS`.

- [ ] **Step 1: Failing tests** (in `app.rs` `mod tests`):
```rust
    #[tokio::test]
    async fn changing_model_forgets_the_old_models_window() {
        let mut app = App::for_tests();
        let (tx, _rx) = mpsc::unbounded_channel();
        app.config.default_model = "qwen2.5:7b".into();
        app.ollama_window = Some(32_768);
        app.server_context_window = Some(8192);
        app.set_model("qwen3:4b", tx);
        assert_eq!(app.config.default_model, "qwen3:4b");
        assert_eq!(app.ollama_window, None);
        assert_eq!(app.server_context_window, None);
    }

    #[tokio::test]
    async fn slash_model_switches_and_lists() {
        let mut app = App::for_tests();
        let (tx, _rx) = mpsc::unbounded_channel();
        app.available_models = vec!["a:1".into(), "b:2".into()];
        app.chat_input.insert_str("/model b:2");
        app.submit_message(tx.clone());
        assert_eq!(app.config.default_model, "b:2");
        app.chat_input.insert_str("/model");
        app.submit_message(tx);
        let last = app.messages.last().unwrap().content.clone();
        assert!(last.contains("a:1") && last.contains("b:2") && last.contains("current: b:2"), "{last}");
    }
```
- [ ] **Step 2: Run to fail; Step 3: implement**
```rust
    /// Every model change goes through here (BT-5): the Models screen, `/model`
    /// and the ByteBot panel's list. The context window measured for the old
    /// model is forgotten, so the next turn is not budgeted for it.
    pub fn set_model(&mut self, model: &str, tx: mpsc::UnboundedSender<String>) {
        self.config.default_model = model.to_string();
        self.save_config();
        self.server_context_window = None;
        self.ollama_window = None;
        if let Some(inner) = llama_model_target(model) {
            self.llamacpp_control("switch", Some(inner.to_string()), tx);
        }
    }
```
  - keymap.rs `key_model_selector` Enter (978): replace the body with `app.set_model(&model, tx.clone()); app.focus = FocusArea::ChatInput;`.
  - `submit_message` chain: before `/help` (~3898) add a `/model` branch: with an argument, `self.set_model(arg, tx)` and a system line "Model: {arg}"; without, a system line listing `available_models` joined by ", " with "current: {default_model}" and "Type /model <name> to switch."
  - Add `"/model"` to `SLASH_COMMANDS` and `("/model [name]", "switch model, or list the ones found")` to help `COMMANDS`.
- [ ] **Step 4: The panel list.** In `key_bytebot` (1439): when `bytebot_model_picker` is true, Up/Down move `bytebot_model_selected` within `available_models`, Enter calls `app.set_model(..)` and closes the picker, Esc closes it. In `run_bytebot`, a command exactly `/model` (no argument) opens the picker instead of running; `/model <name>` calls `set_model`. ui.rs `draw_bytebot_panel`: when the picker is open, draw the list over the step rows with "↑↓ Enter Esc" in the title and the current model marked. Test: `bytebot_panel_model_picker_switches_without_leaving` — open the panel, type `/model`, Enter, Down, Enter; assert `focus == ByteBotPanel` and the model changed.
- [ ] **Step 5: Pass, full crate suite, clippy, docs (USER_MANUAL key table, QUICK_START command list, CHANGELOG), commit** — `BT-5: /model and a model list in the ByteBot panel; a model change forgets the old model's context window`.

---

### Task 7: slash commands from the ByteBot panel (BT-4)

**Files:** `app.rs` (`submit_message` split), `keymap.rs` tests.

**Interfaces:**
- Produces: `pub fn dispatch_prompt(&mut self, prompt: String, tx: mpsc::UnboundedSender<String>)` — everything `submit_message` did after reading and clearing the composer; `submit_message` becomes "read composer, clear it, history, then `dispatch_prompt`".

- [ ] **Step 1: Failing test**
```rust
    #[tokio::test]
    async fn a_slash_command_in_the_bytebot_panel_runs_the_command_not_a_task() {
        let mut app = App::for_tests();
        let (tx, _rx) = mpsc::unbounded_channel();
        app.chat_input.insert_str("my draft");
        app.bytebot_command = "/help".into();
        app.run_bytebot(tx);
        assert!(app.help_visible, "/help opened the help overlay");
        assert!(!app.bytebot_running, "no task was started");
        assert_eq!(app.chat_input.lines().join(""), "my draft", "the chat draft is untouched");
    }
```
- [ ] **Step 2: Run to fail. Step 3:** split `submit_message` at the point right after the input is reset and history recorded (the code map puts the chain at ~3693): move the rest into `dispatch_prompt(prompt, tx)`. In `run_bytebot`, before `arm_bytebot`: if `task.starts_with('/') && !task.starts_with("/model")`, clear `bytebot_command`, push a log line `"ran {first word} — its output is in the chat"`, and call `self.dispatch_prompt(task, tx)`; return.
- [ ] **Step 4: Pass; the existing slash-command tests still pass unchanged; commit** — `BT-4: slash commands typed in the ByteBot panel run like they do in chat`.

---

### Task 8: ByteBot task records and the queue (BT-1)

**Files:**
- Create: `rust/crates/xencode-tui-rs/src/bytebot_tasks.rs` (`pub mod bytebot_tasks;` in `lib.rs`)
- Modify: `app.rs`, `ui.rs` (`draw_bytebot_panel` 3170–3430)

**Interfaces:**
- Produces:
  - `pub enum TaskState { Pending, Running, NeedsHelp, NeedsReview, Completed, Cancelled, Failed }` (serde `snake_case`), `fn words(self) -> &'static str`
  - `pub struct ByteBotTask { pub id: String, pub text: String, pub model: String, pub state: TaskState, pub steps: Vec<(String, String)>, pub changed_files: Vec<String>, pub question: Option<String>, pub note: Option<String>, pub turn: Option<usize>, pub created_at: u64, pub started_at: Option<u64>, pub ended_at: Option<u64> }`
  - `pub struct TaskStore { dir: PathBuf }` with `pub fn new(xencode_dir: &Path) -> TaskStore` (dir = `<xencode>/bytebot/tasks`), `pub fn save(&self, t: &ByteBotTask) -> io::Result<()>` (atomic, like Task 1), `pub fn load_all(&self) -> Vec<ByteBotTask>` (oldest first), `pub fn recover(&self) -> usize` (marks `Running`/`NeedsHelp` as `Failed` with note "xencode exited during this task", returns how many)
  - App: `pub bytebot_tasks: Vec<ByteBotTask>`, `bytebot_store: Option<TaskStore>`, `fn bytebot_current(&mut self) -> Option<&mut ByteBotTask>`, `fn bytebot_start_next(&mut self, tx)`.

- [ ] **Step 1: Failing tests** in `bytebot_tasks.rs`:
```rust
    #[test]
    fn a_task_round_trips_and_states_use_the_original_names() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let t = ByteBotTask::new("fix the login bug", "llamacpp:qwen3-4b");
        store.save(&t).unwrap();
        let text = std::fs::read_to_string(dir.path().join("bytebot/tasks").join(format!("{}.json", t.id))).unwrap();
        assert!(text.contains("\"state\": \"pending\""), "{text}");
        assert_eq!(store.load_all(), vec![t]);
    }

    #[test]
    fn a_task_left_running_or_waiting_is_failed_on_restart() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let mut a = ByteBotTask::new("a", "m"); a.state = TaskState::Running;
        let mut b = ByteBotTask::new("b", "m"); b.state = TaskState::NeedsHelp;
        let c = ByteBotTask::new("c", "m");
        for t in [&a, &b, &c] { store.save(t).unwrap(); }
        assert_eq!(store.recover(), 2);
        let all = store.load_all();
        assert!(all.iter().filter(|t| t.state == TaskState::Failed).all(|t| t.note.as_deref() == Some("xencode exited during this task")));
        assert_eq!(all.iter().find(|t| t.text == "c").unwrap().state, TaskState::Pending);
    }
```
  and in `app.rs` tests, with the scripted server (`serve_scripted_answers`, app.rs ~13376; copy the set-up of an existing ByteBot test such as the one around 17175):
```rust
    // Enter while a task runs queues the next one; tasks run oldest first;
    // a task that changes nothing ends Completed (NeedsReview comes in BT-3).
    async fn tasks_queue_and_run_in_order()
```
  Write that test in full by copying the nearest existing ByteBot scripted-server test and asserting `bytebot_tasks` states after each `[BYTEBOT_DONE]` drain.
- [ ] **Step 2: Run to fail. Step 3: implement** `bytebot_tasks.rs` (ids: `format!("{}-{:04x}", now_secs(), rand)` — use `std::process::id() ^ nanos` as the low bits to avoid a new dependency), the atomic save copied from Task 1's `write_status`, and the App wiring:
  - `run_bytebot`: create a `ByteBotTask` (Pending, current model), save it, push it; if no task is running, `bytebot_start_next`.
  - `bytebot_start_next`: first Pending → Running, `started_at`, `turn = Some(self.checkpoints.turns())` before `arm_bytebot`, then spawn `agent_rounds` exactly as `run_bytebot` does today.
  - `[BYTEBOT_DONE]` branch: finish the current task — Failed if any step failed or an `err:` event arrived (note = the error), else Completed; copy `bytebot_steps` into the record; save; feed the badge with `live_turn_ended`; then `bytebot_start_next`.
  - `bytebot_event` steps also append to the current task record.
  - Esc on a running task (UX-14's stop flag): when `[STOPPED]`/`[BYTEBOT_DONE]` arrives after a stop, the state is Cancelled.
  - `App::new()`: `bytebot_store = Some(TaskStore::new(&<project>/.xencode))`, `recover()`, load into `bytebot_tasks`; if it recovered any, a toast "N ByteBot task(s) were interrupted when xencode last exited".
  - ui.rs: a task list above the step rows: one line per task, newest last, `state words` + text, the running one highlighted with `›`.
- [ ] **Step 4: Pass, crate suite, clippy, commit** — `BT-1: ByteBot tasks are kept as records with the original project's states and run one at a time`.

---

### Task 9: the agent can ask for help (BT-2)

**Files:** `xencode-providers-rs/src/tools.rs`, `agent_tools.rs`, `app.rs`, `keymap.rs`, `ui.rs`.

**Interfaces:**
- Produces: `pub fn ask_user_tool() -> ToolDefinition` in providers `tools.rs`; `ToolClass` for `ask_user` = ReadOnly (it changes nothing; the question is the gate); `ApprovalCtx.ask: Option<mpsc::UnboundedSender<(String, oneshot::Sender<String>)>>`; App field `bytebot_help: Option<(String, oneshot::Sender<String>)>`.

- [ ] **Step 1: Failing test** (scripted server returns one `ask_user` call, then a final answer):
```rust
    // The task pauses in NeedsHelp with the question shown; answering resumes it
    // and the answer reaches the model as the tool result; "/done" tells the
    // model the person did the step themselves.
    async fn a_task_that_asks_for_help_waits_for_the_answer()
```
  Assert: state NeedsHelp and `question == Some("Which database?")` after the call; badge feed `needs_you`; after typing "postgres" + Enter in the panel, the next request body sent to the scripted server contains `"postgres"`; final state Completed. A second test types `/done` and asserts the tool result text `"The person did this step themselves. Continue from there."`
- [ ] **Step 2: Run to fail. Step 3: implement**
  - `tools.rs`:
```rust
pub fn ask_user_tool() -> ToolDefinition {
    ToolDefinition {
        name: "ask_user".to_string(),
        description: "Stop and ask the person one question when you cannot continue without their answer. The task pauses until they reply.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "question": { "type": "string", "description": "One clear question" } },
            "required": ["question"]
        }),
    }
}
```
  - Offered only when `sink == LoopSink::ByteBot`: in `agent_rounds`, after `offered_tools_with(..)`, `if sink == LoopSink::ByteBot { tools.push(xencode_providers_rs::ask_user_tool()); }`.
  - `agent_tools.rs` `execute_tool_call_plan` match (3829): an `"ask_user"` arm is unreachable there because it needs the context; instead handle it in `execute_tool_call_approved` (4360) before the permission check:
```rust
    if call.name == "ask_user" {
        let question = call.arguments_object().get("question").and_then(|q| q.as_str()).unwrap_or("").to_string();
        let Some(ask) = ctx.ask.as_ref() else {
            return "error: ask_user is only available to ByteBot tasks".to_string();
        };
        let (reply, answer) = oneshot::channel();
        if ask.send((question, reply)).is_err() {
            return UNATTENDED_RESULT.to_string();
        }
        return match answer.await {
            Ok(text) if text.trim() == "/done" => "The person did this step themselves. Continue from there.".to_string(),
            Ok(text) => format!("The person answered: {text}"),
            Err(_) => "The person did not answer; the task was cancelled.".to_string(),
        };
    }
```
  - App: an `ask_tx`/`ask_rx` pair created like `approval_tx`/`approval_rx`; `arm_bytebot` sets `run.approval.ask = Some(self.ask_tx.clone())`; every other `ApprovalCtx` construction sets `ask: None`. `run_app` drains `ask_rx` next to the approval drain: set the current task NeedsHelp with the question, save, `live_set(NeedsYou, Bytebot, question)`, store the sender in `bytebot_help`.
  - `key_bytebot`: when `bytebot_help` is Some, Enter sends `bytebot_command` through the stored sender (taking it), clears the command, sets the task back to Running, `live_tool_started("answer received")`. Esc while waiting drops the sender (the tool returns "did not answer") and sets the stop flag, so the task ends Cancelled.
  - ui.rs: while waiting, draw the question in amber above the command box with "type an answer, or /done if you did it yourself · Esc cancels".
- [ ] **Step 4: Pass, crate suite (approval tests unchanged), clippy, commit** — `BT-2: a ByteBot task can stop and ask the person a question, or let them take over`.

---

### Task 10: review before a task counts as done (BT-3)

**Files:** `app.rs`, `keymap.rs`, `ui.rs`.

**Interfaces:**
- Consumes: `CheckpointStore::{turns, group_paths, rewind}` (agent_tools.rs 4587–4656), the task's `turn`.
- Produces: `fn bytebot_accept(&mut self)`, `fn bytebot_undo(&mut self) -> Result<RewindReport, String>`.

- [ ] **Step 1: Failing tests**
```rust
    // A task whose turn wrote files ends NeedsReview listing them; `a` with an
    // empty command box makes it Completed.
    async fn a_task_that_changed_files_waits_for_review()
    // `u` rewinds the task's group: the file is back to its old bytes, state Cancelled, note "changes undone".
    async fn undo_restores_the_files_and_cancels_the_task()
    // A later chat turn also wrote a file: `u` refuses with
    // "Undo is only possible while this task's changes are the latest; use /rewind to step back through later turns first."
    // and nothing on disk changes.
    async fn undo_refuses_when_a_later_turn_changed_files()
```
  Use the scripted server with a `write_file` call and `agent_approval = "edit-allow"`, as the plugin hook test does (app.rs ~14666).
- [ ] **Step 2: Run to fail. Step 3: implement**
  - `[BYTEBOT_DONE]` normal end: `let paths = self.checkpoints.group_paths(turn)`; non-empty → NeedsReview, `changed_files` = paths relative to the project with forward slashes, `live_set(NeedsYou, Bytebot, "review N changed file(s)")`; empty → Completed.
  - `bytebot_accept`: NeedsReview → Completed, save, `live_turn_ended(None)`.
  - `bytebot_undo`: refuse unless the task's turn is the newest group holding writes. Check by comparing `self.checkpoints.pending_paths(1)` with the task's `changed_files` (absolute paths): equal means the latest write group is this task's. Then `self.checkpoints.rewind(1)`, set Cancelled with note "changes undone", save, `live_stopped()`.
  - `key_bytebot`: with `bytebot_command` empty and the current reviewable task in NeedsReview, `a` → accept, `u` → undo (result as a log line). Otherwise letters type as usual.
  - ui.rs: while reviewing, list `changed_files` and "a accept · u undo".
- [ ] **Step 4: Pass, crate suite, clippy, commit** — `BT-3: a ByteBot task that changed files waits for the person to accept or undo it`.

---

### Task 11: close out

- [ ] **Step 1:** Full `cargo test --workspace -j 4` with `XCODE_CONFIG_DIR` set; compare failures with the Windows list under `PL-2` (other crates' failures are known and open). Record counts.
- [ ] **Step 2:** One live run end to end with a real local model: start llama.cpp as `NEXT_PLAN_TASKS.md` §SM-2 describes, run a ByteBot task that asks for help and edits a file, answer, review, accept; watch the badge go working → needs you → working → needs you → finished. Stop the server immediately. Write down what was seen.
- [ ] **Step 3:** `README.md`, `QUICK_START.md`, `CLI_GUIDE.md`, `docs/USER_MANUAL.md`: ByteBot tasks, `/model`, `/done`, `a`/`u`, the badge — only what was run. `NEXT_PLAN_TASKS.md`: check off DK-1, DK-2 (Windows), BT-1 … BT-5 with evidence; DK-2's macOS line stays open; counts updated. `CHANGELOG.md`.
- [ ] **Step 4:** `cargo clean` in `rust/` (the workspace suite ran) and report `rust/badge/target` size; remove `xencode-*` temp folders this session created.
- [ ] **Step 5:** Commit — `DK-1, DK-2, BT-1 to BT-5: manuals, plan and changelog for the badge and ByteBot tasks`.
