# Orchestrator Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turning orchestrator mode on replaces the conversation with a rail of agents and a grid of tiles, each tile running an installed coding agent's real terminal interface in its own git worktree, hosted by the project's engine so the agents outlive the window.

**Architecture:** A new crate `xencode-term-rs` owns a PTY session (`portable-pty`), a `vt100` screen with scrollback, an attention detector over the byte stream, and a function that draws a `vt100` screen into a ratatui buffer. The engine gains a `TermHost` keyed by team worker id; tiles are team workers of kind `terminal`, so worktrees, the checked merge, the audit line and the Agents view apply unchanged. Windows talk to it over new `ClientMsg::Term` / `EngineMsg::TermOutput` / `EngineMsg::TermReply` messages and keep their own `vt100` parser per tile.

**Tech Stack:** Rust 2021, ratatui 0.29, crossterm 0.28, portable-pty 0.9.0, vt100 0.16.2, tokio, the existing engine protocol (JSON lines) and `xencode-team-rs`.

**Spec:** `docs/superpowers/specs/2026-10-11-orchestrator-mode-design.md`

## Global Constraints

- Rust only, under `rust/crates/*`; no mocks anywhere — every PTY test runs a real shell (`cmd`/`powershell` on Windows, `sh` on Unix).
- ratatui stays at 0.29; `tui-term` is not used (it needs `ratatui-core` 0.1).
- `portable-pty = "0.9"`, `vt100 = "0.16"`; scrollback 5,000 lines.
- A tile runs the vendor's own CLI with the person's own environment and sign-in; the Local Only posture does not block a tile, and the picker says "runs <vendor>'s own CLI with your sign-in".
- The prefix key is Ctrl+A (`orchestrator.prefix`); a doubled Ctrl+A sends a literal Ctrl+A.
- Working = output in the last 2 s; idle = none for 2 s; needs you = BEL or OSC 9 / OSC 777 since the person last typed into it; done / failed = exit 0 / non-zero. No screen-text heuristics.
- A session's size is the size the most recent `Resize` asked for.
- One commit per plan item, naming its ID, in plain English; `cargo fmt --all`, clippy `-D warnings`, the workspace suite against the known Windows baseline before each commit.
- Never start a local GPU workload; the live run with installed agents is watched by the person, not a CI test.

## Review Focus

1. **A program that floods output** (`yes`, a build log): the window stays responsive — output is coalesced per frame and the screen keeps only its scrollback.
2. **A tile narrower than the agent's own minimum width**: drawn clipped, never a panic; resize sends the real size so the agent re-lays out.
3. **Unicode and wide characters in a tile** (CJK, emoji, combining marks): `vt100` cell widths drive the draw; a wide cell takes two columns and nothing writes past the tile edge.
4. **The engine restarts while tiles run**: tiles say `engine restarting`; the rail re-reads; worktrees are adopted as stopped workers; nothing is lost on disk.
5. **The child ignores SIGTERM / closes stdin**: stop ends the whole process tree; the tile reports `exited`.

---

### Task 1: The terminal crate (OM-1)

**Files:**
- Create: `rust/crates/xencode-term-rs/Cargo.toml`, `src/lib.rs`, `src/session.rs`, `src/screen.rs`, `src/attention.rs`, `src/render.rs`
- Modify: `rust/Cargo.toml` (workspace member)
- Test: `rust/crates/xencode-term-rs/tests/real_shell.rs`

**Interfaces:**
- Produces:
  - `pub struct Session` with `pub fn spawn(cmd: &CommandSpec, size: Size) -> Result<Session, TermError>`, `pub fn write(&self, bytes: &[u8]) -> Result<(), TermError>`, `pub fn resize(&self, size: Size) -> Result<(), TermError>`, `pub fn try_read(&self) -> Vec<u8>` (drains what the reader thread collected), `pub fn exit_status(&self) -> Option<i32>`, `pub fn kill(&self)` (whole tree).
  - `pub struct CommandSpec { pub program: String, pub args: Vec<String>, pub cwd: PathBuf, pub env: Vec<(String, String)> }`; `pub struct Size { pub cols: u16, pub rows: u16 }`.
  - `pub struct Screen` wrapping `vt100::Parser` (scrollback 5,000): `pub fn new(size: Size) -> Self`, `pub fn feed(&mut self, bytes: &[u8])`, `pub fn resize(&mut self, size: Size)`, `pub fn replay(&self) -> Vec<u8>` (`contents_formatted`), `pub fn scroll(&mut self, rows: isize)`.
  - `pub struct Attention` with `pub fn observe(&mut self, bytes: &[u8], now: Instant)`, `pub fn typed(&mut self)`, `pub fn state(&self, now: Instant, exited: Option<i32>) -> TileState`; `pub enum TileState { Working, Idle, NeedsYou, Done, Failed }`.
  - `pub fn draw(screen: &vt100::Screen, area: Rect, buf: &mut Buffer)` — every cell's symbol, fg, bg, bold/italic/underline/inverse; wide cells take two columns; nothing past `area`.

- [ ] **Step 1: Failing tests** (`tests/real_shell.rs`), all against a real shell:

```rust
fn shell() -> CommandSpec {
    if cfg!(windows) {
        CommandSpec { program: "cmd".into(), args: vec!["/Q".into()], cwd: std::env::temp_dir(), env: vec![] }
    } else {
        CommandSpec { program: "sh".into(), args: vec![], cwd: std::env::temp_dir(), env: vec![] }
    }
}

fn wait_for(session: &Session, screen: &mut Screen, needle: &str) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        screen.feed(&session.try_read());
        if screen.text().contains(needle) { return true; }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

#[test]
fn output_reaches_the_screen_and_input_reaches_the_program() {
    let size = Size { cols: 80, rows: 24 };
    let session = Session::spawn(&shell(), size).unwrap();
    let mut screen = Screen::new(size);
    session.write(b"echo XENCODE_TERM_OK\r").unwrap();
    assert!(wait_for(&session, &mut screen, "XENCODE_TERM_OK"), "{}", screen.text());
    session.kill();
}

#[test]
fn a_child_that_exits_reports_its_code() {
    let session = Session::spawn(&shell(), Size { cols: 80, rows: 24 }).unwrap();
    session.write(if cfg!(windows) { b"exit 3\r" } else { b"exit 3\n" }).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while session.exit_status().is_none() && Instant::now() < deadline { std::thread::sleep(Duration::from_millis(50)); }
    assert_eq!(session.exit_status(), Some(3));
}

#[test]
fn a_bell_raises_needs_you_and_typing_clears_it() {
    let mut a = Attention::default();
    let t = Instant::now();
    a.observe(b"done\x07", t);
    assert_eq!(a.state(t, None), TileState::NeedsYou);
    a.typed();
    assert_eq!(a.state(t, None), TileState::Working);
    a.observe(b"\x1b]9;finished\x07", t);
    assert_eq!(a.state(t, None), TileState::NeedsYou);
    assert_eq!(Attention::default().state(t + Duration::from_secs(3), None), TileState::Idle);
}

#[test]
fn replay_reproduces_the_screen() {
    let size = Size { cols: 40, rows: 10 };
    let mut first = Screen::new(size);
    first.feed(b"\x1b[31mred\x1b[0m and plain\r\nsecond line");
    let mut second = Screen::new(size);
    second.feed(&first.replay());
    assert_eq!(first.text(), second.text());
}

#[test]
fn the_screen_draws_into_a_buffer_and_stops_at_the_edge() {
    let mut s = Screen::new(Size { cols: 20, rows: 3 });
    s.feed("wide 漢字 cells and a long line past the edge".as_bytes());
    let area = Rect::new(0, 0, 10, 3);
    let mut buf = Buffer::empty(Rect::new(0, 0, 12, 3));
    xencode_term_rs::draw(s.vt(), area, &mut buf);
    assert_eq!(buf[(10, 0)].symbol(), " ", "nothing past the tile");
}
```

- [ ] **Step 2: Run** `cargo test -p xencode-term-rs` — Expected: FAIL to compile (crate empty).
- [ ] **Step 3: Implement** `session.rs` (portable-pty `native_pty_system().openpty`, `CommandBuilder` with cwd/env, a reader thread pushing into `Arc<Mutex<Vec<u8>>>`, `child.wait` on a thread storing the code, kill via `child.kill()` plus the process group on Unix / job object on Windows), `screen.rs` (`vt100::Parser::new(rows, cols, 5000)`, `text()` = `screen().contents()`), `attention.rs` (scan for `0x07` outside OSC, `ESC ] 9 ;`, `ESC ] 777 ;`; last-output instant), `render.rs` (map `vt100::Color` to ratatui `Color`, honour `is_wide`/`is_wide_continuation`).
- [ ] **Step 4: Run** the crate tests on Windows and on Linux (WSL `~/xencode` copy). Expected: PASS on both.
- [ ] **Step 5: Commit** `OM-1: a terminal crate runs a real program in a pseudo-terminal and draws its screen`.

### Task 2: The engine hosts terminal sessions (OM-2)

**Files:**
- Create: `rust/crates/xencode-tui-rs/src/engine/term.rs` (`TermHost`)
- Modify: `engine/proto.rs` (`TermRequest`, `ClientMsg::Term`, `EngineMsg::TermOutput`, `EngineMsg::TermReply`), `engine/server.rs` (dispatch, `has_work` counts live sessions, fan-out to watchers), `engine/team.rs` (worker kind `terminal`: worktree + branch as for ACP workers, no ACP connection), `xencode-team-rs/src/lib.rs` (`WorkerSnapshot.kind`, serde default `"acp"`), `engine/link.rs` (window side: send/receive), `app.rs` (`term_tiles: BTreeMap<String, TileView>` with a local `Screen`)
- Test: `rust/crates/xencode-cli/tests/engine_term_cli.rs` (real `xencode engine` process, real shell)

**Interfaces:**
- Consumes: `Session`, `Screen`, `Attention`, `TileState` (Task 1).
- Produces: `TermRequest::{Agents, Open { agent: String }, Input { id, bytes_b64 }, Resize { id, cols, rows }, Watch { ids }, Stop { id }}`; `EngineMsg::TermOutput { id, bytes_b64 }`; `EngineMsg::TermReply { req: u64, ok: bool, body: serde_json::Value }`; `App::term_open(agent)`, `App::term_input(id, bytes)`, `App::term_resize(id, cols, rows)`, `App::term_watch_all()`.

- [ ] **Step 1: Failing end-to-end test:** start `xencode engine` on a scratch git repository (the existing engine test helpers), connect a window, `Open { agent: "shell" }` (a built-in roster entry `shell` that runs the platform shell, used for tests and for a plain terminal tile), `Watch`, `Input "echo TERM_E2E\r"`, read `TermOutput` until the text appears; disconnect; reconnect; `Watch` again and assert the replay already contains `TERM_E2E`; the worker shows `kind: "terminal"` and a worktree under `<repo>-team/`.
- [ ] **Step 2: Run** — Expected: FAIL (no `Term` message).
- [ ] **Step 3: Implement** the host (one tokio task per session polling `try_read` every 16 ms, coalescing bytes per tick; the host's own `Screen` per session for replay; `Attention` state folded into the worker snapshot), the protocol, the team kind, and the window side.
- [ ] **Step 4: Run** the new test plus `team_cli`, `engine_cli` and the TUI crate on Windows and Linux. Expected: PASS.
- [ ] **Step 5: Commit** `OM-2: the engine runs terminal agents in their own worktrees and windows watch and type into them`.

### Task 3: The mode: picker, rail and tiles (OM-3)

**Files:**
- Create: `rust/crates/xencode-tui-rs/src/orchestrator_view.rs` (layout, rail, tiles, picker)
- Modify: `ui.rs` (`draw_frame`: when `app.mode == Orchestrator` draw the orchestrator view), `keymap.rs` (prefix handling before every other stage except approvals), `app.rs` (`orch_focus`, `orch_zoom`, `orch_tile_all`, `orch_prefix_armed`, picker state), config (`orchestrator.agents`, `orchestrator.prefix`)
- Test: `tests/orchestrator_screens.rs`

**Interfaces:**
- Consumes: `term_tiles`, `term_open`, `term_input`, `term_resize` (Task 2); `xencode_term_rs::draw`.
- Produces: `pub fn tile_rects(area: Rect, tiles: usize, zoom: Option<usize>, all: bool, rail: bool) -> (Option<Rect>, Vec<Rect>)`.

- [ ] **Step 1: Failing tests:** `tile_rects` at 160×48 with 3 tiles gives a rail of 22 columns and a 2×2 grid with one empty cell; with `zoom = Some(1)` one tile fills the area under a one-row strip; under 100 columns there is no rail and one tile; the picker lists every roster agent with installed/missing and the sign-in line; with the mode on and a real `shell` tile, typed letters reach the tile's PTY (its screen shows them), Ctrl+A then `z` zooms, Ctrl+A Ctrl+A writes `0x01` to the tile, Ctrl+A `0` focuses xencode's own conversation tile.
- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** the layout, rail rows (number, glyph, name, elapsed, branch, +/−, cost), the picker (remembers the choice in config), keys and focus; each tile's `Resize` is sent when its rect changes.
- [ ] **Step 4: Run** the crate tests and an any-size sweep of the orchestrator view (4×3 up to 200×60). Expected: PASS.
- [ ] **Step 5: Commit** `OM-3: orchestrator mode shows a rail of agents and their live screens in tiles`.

### Task 4: Attention, inbox and notification (OM-4)

**Files:** `orchestrator_view.rs` (yellow frame, Needs you group, INBOX), `shell.rs` (`attention_title` counts tiles that need the person), `app.rs` run loop (OSC 9 once per new needs-you while unfocused)
**Test:** `tests/orchestrator_screens.rs`

- [ ] **Step 1: Failing tests:** a real `shell` tile told to `printf '\a'` (Unix) / `echo ^G` (Windows) moves to Needs you within 2 s, its frame turns the warning colour, the title reads `xencode · 1 needs you`; typing into it clears it.
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS on both platforms).
- [ ] **Step 5: Commit** `OM-4: an agent that rings or sends a notification is marked as needing you`.

### Task 5: Merge, stop, ask all, tile 0 and small terminals (OM-5)

**Files:** `orchestrator_view.rs`, `keymap.rs`, `engine/term.rs` (stop = kill tree, keep worktree), `app.rs`
**Test:** `tests/orchestrator_screens.rs`, `xencode-cli/tests/engine_term_cli.rs`

- [ ] **Step 1: Failing tests:** in a scratch repository a real `shell` tile writes a file in its worktree and commits it; Ctrl+A `m` opens the Changes view on that diff, Enter runs the checked merge and the file lands on the base branch with an audit line; Ctrl+A `x` asks, then the process ends and the worktree stays; Ctrl+A `a` with `echo hi` writes to every working or idle tile.
- [ ] **Step 2–4:** run (FAIL), implement, run (PASS).
- [ ] **Step 5: Commit** `OM-5: a tile's work is merged from the rail through the checked merge, and tiles can be stopped or all asked at once`.

### Task 6: Manuals and a watched live run (OM-6)

- [ ] **Step 1:** Render the orchestrator view at 160×48 with real `shell` tiles (temporary capture test, not committed) for the manuals.
- [ ] **Step 2:** README, QUICK_START, CLI_GUIDE, USER_MANUAL: the mode, the picker, the keys after Ctrl+A, attention, merge and stop, the settings `orchestrator.agents` and `orchestrator.prefix`, what differs on Windows.
- [ ] **Step 3:** With the person present, a live run with the installed agents (claude, codex, opencode, …), written down in NEXT_PLAN_TASKS with what worked and what did not. Not a CI test.
- [ ] **Step 4: Commit** `OM-6: the manuals describe orchestrator mode from a watched run with real agents`.
