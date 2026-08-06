# Rust TUI Foundation — Phase 1 Design

**Date:** 2026-06-19
**Status:** Approved (pending spec review)
**Crate:** `xencode-tui-rs`
**Type:** Refactor + 3 critical bug fixes (no new features, no new backends)

---

## Context and Goal

The end goal is a **full Python-TUI → Rust-TUI migration** so the Python Textual TUI can be retired entirely in favor of the higher-performance Rust TUI. That larger effort is decomposed into five phases, each with its own spec → plan → build cycle:

| Phase | Scope |
|-------|-------|
| **1 (this spec)** | Refactor monolith into module tree; fix 3 critical wiring bugs |
| 2 | Wire feasible panels to real Rust backends |
| 3 | Port missing Python-only features (onboarding, diff viewer, commit wizard, etc.) |
| 4 | Build hard infrastructure (voice STT, collaboration sync, profiler crate, lessons, translate) |
| 5 | Cut-over: Rust becomes default `xencode`; remove Python TUI |

**Decisions confirmed with the user:**
- Parity depth: **fully real backends** (panels call real Rust crates, not mocks).
- Hard panels: **build everything real** (voice, collab, profiler, lessons, translate all eventually real).
- Cutover: **Rust first, then remove Python** — the Rust TUI must stay functional and building at every step; Python TUI is only removed after 100% feature parity is verified.

This spec covers **Phase 1 only**.

### Why Phase 1 exists

The current Rust TUI is two monolithic files:
- `src/app.rs` — 2100 lines (all state + the entire event loop + message router)
- `src/ui.rs` — 2186 lines (all rendering for 25 `draw_*` functions)
- `src/lib.rs` — 4 lines

These files are already at the edge of maintainability. Phases 2–4 will add onboarding modals, diff viewers, a commit wizard, a code-execution panel, real STT, a collaboration client, and a profiler crate. Continuing to extend the monolith would make each future change high-risk and untestable. Phase 1 pays the structural debt once so that all subsequent phases are tractable.

Phase 1 deliberately does **not** wire any panel to a new backend or add any new feature. That keeps the diff reviewable and the regression risk contained to code moves plus three well-understood bug fixes.

---

## Current State (baseline, verified)

- `cargo build --workspace` exits 0 (clean).
- 23 `FocusArea` variants in `app.rs`.
- 25 `draw_*` functions in `ui.rs`.
- 13 feature panels already render (not "coming soon" — the project doc is stale on this point), but most run on hardcoded mock data and several have wiring bugs.
- Critical bugs confirmed by audit (these are the only behavioral changes Phase 1 makes):
  1. **ByteBot Enter never executes** — `app.rs:1693-1701`: the `ByteBotPanel` Enter handler only recalls history (and history is always empty, see bug 4). The fully-implemented `run_bytebot()` (`app.rs:832`) is never invoked from the panel. The `/bytebot` chat-command path works; only the dedicated panel is dead.
  2. **Settings cursor indices swapped** — `app.rs:1761-1774`: the Enter handler maps `settings_cursor == 7` → URL editing and `== 6` → factory reset, but the settings list (`ui.rs:466-475`) indexes `6 = Ollama URL`, `7 = Factory Reset`. The Left/Right handlers (`app.rs:1909, 1948`) correctly use `== 7` for URL, so arrow-keys and Enter disagree on the same row. Selecting "Ollama URL" + Enter wipes the config; selecting "Factory Reset" + Enter drops into URL-edit mode.
  3. **Terminal pane renders a stub** — `ui.rs:385-395`: `draw_terminal()` renders hardcoded "Terminal emulation coming soon" and ignores `app.terminal_output`. The backend (`spawn_terminal`, `write_terminal`, the spawned `terminal_child`, the `[TERM_EXEC]` reader at `app.rs:1446-1501`) all function; only display is missing.
  4. **ByteBot history never populated** — `bytebot_history` is read by the recall feature (`ui.rs:1102`, `app.rs:1626, 1696`) but nothing ever pushes to it. Fixed as part of bug 1.

Non-critical issues (NOT addressed in Phase 1 — deferred to later phases): learning mode stuck on lesson 1, multi-language translate non-functional, custom models can't edit/persist, terminal-assistant query field cosmetic, model list hardcoded, theme dots cap at 4, profiler single-shot, mouse clicks only route to 3 panes.

---

## Module Architecture

### Target file tree

```
src/
├── lib.rs                  # re-exports; run_app() entry (unchanged signature)
├── app.rs                  # App struct (shared state only) + event dispatch (~400 lines)
├── theme.rs                # ThemeColors + 7 themes (extracted from app.rs:66-186)
├── focus.rs                # FocusArea enum + navigate_feature() (extracted from app.rs)
├── layout.rs               # draw(), draw_header, draw_status_bar, draw_body, centered_rect
├── editor.rs               # TextArea wrapper + open_file_in_editor/save_editor
├── terminal.rs             # terminal spawn/exec/read + real render (fixes stub)
├── input.rs                # TextInput helper (cursor + insert/backspace) shared by text fields
├── panels/
│   ├── mod.rs              # Panel trait + dispatch helpers
│   ├── chat.rs             # chat input + messages (owns draw_input, draw_messages logic)
│   ├── file_explorer.rs
│   ├── code_review.rs
│   ├── model_selector.rs
│   ├── settings.rs         # ← fixes index-swap bug (bug 2)
│   ├── feature_navigator.rs
│   ├── git_commit.rs
│   ├── performance_dashboard.rs
│   ├── provider_health.rs
│   ├── project_analyzer.rs
│   ├── bytebot.rs          # ← fixes "Enter doesn't execute" (bug 1) + history (bug 4)
│   ├── collaboration_hub.rs
│   ├── voice_interface.rs
│   ├── terminal_assistant.rs
│   ├── security_auditor.rs
│   ├── performance_profiler.rs
│   ├── custom_models.rs
│   ├── learning_mode.rs
│   └── multi_language.rs
└── widgets/
    ├── mod.rs
    ├── gauge.rs            # the repeated bar()/gauge_block() helpers
    └── spinner.rs          # the repeated spinner-frame arrays
```

### The `Panel` trait

The central abstraction. Every overlay/feature panel implements it. The giant `match app.focus { ... }` blocks in both `app.rs` (key handling) and `ui.rs` (rendering) collapse into single dispatch calls.

```rust
use ratatui::Frame;
use ratatui::layout::Rect;
use crossterm::event::KeyEvent;
use crate::app::App;
use crate::channel::ChannelSender;

pub trait Panel {
    /// Which focus area activates this panel.
    fn focus_area(&self) -> FocusArea;

    /// Window title (border title) shown when this panel is open.
    fn title(&self, app: &App) -> String;

    /// Render the panel as an overlay (panels draw into a centered popup area,
    /// which they compute themselves or receive via a helper).
    fn render(&self, f: &mut Frame, app: &App);

    /// Handle a key press while this panel is focused.
    /// Returns `true` if the key was consumed (prevents fall-through to global handlers
    /// for that key). Must NOT handle global Ctrl+ shortcuts — those are dispatched
    /// before the panel sees the key.
    fn handle_key(&mut self, app: &mut App, key: KeyEvent, tx: &ChannelSender) -> bool;

    /// Process an async channel message routed to this panel.
    /// Each panel owns its message prefix (e.g. "[BYTEBOT]", "[SECURITY]") and this
    /// method is called only for messages with that prefix (after prefix-stripping),
    /// or with a sentinel like "[DONE]" when the panel is active.
    fn handle_message(&mut self, app: &mut App, token: &str);

    /// Hint text shown in the status bar when this panel is focused.
    fn hint(&self) -> &str;
}
```

Design notes on the trait:

- **Render takes `&App` + `&Self`** (both immutable). Panels own their own state in their struct; they mutate it in `handle_key`/`handle_message`, never during render. This matches the existing pattern where `draw_*` functions take `&App`.
- **`handle_key` returns `bool`** so the dispatcher can decide whether to fall through. Global shortcuts (Ctrl+C, Ctrl+, etc.) are handled in `app.rs` *before* the panel is consulted, so panels never need to re-implement quit/switch.
- **Message routing**: `run_app` keeps a small table mapping message prefix to panel ref. On each `rx.try_recv()`, it matches the prefix and calls `panels.<specific>.handle_message(&mut app, body)`. Sentinels like `[DONE]`, `[HEALTH_DONE]`, `[BYTEBOT_DONE]` route to the currently-generating panel.
- The core non-overlay surfaces (chat input, file explorer, code editor) are **not** `Panel`s — they're always-on panes in the 3-column body layout, handled directly in `layout.rs`. Only the overlay feature panels implement `Panel`. (This avoids forcing a popup model onto the persistent body.)
- **Cross-panel focus switches**: a panel's `handle_key` can return a `PanelAction::SwitchFocus(FocusArea)` to request the dispatcher change `app.focus`. The dispatcher performs the switch — panels never touch each other directly.

### State split: `App` (shared) + `Panels` (per-panel)

Per the borrow approach above, there are **two top-level structs** owned side by side in `run_app`:

**`App`** — shared state only:
`focus`, `input_mode`, `config`, `theme`, `memory`, `file_tree`, `selected_file`, `file_scroll_offset`, `attached_files`, `opened_file`, `editor`, `editor_dirty`, `git_status`, `git_branch`, `available_models`, `selected_model`, `messages`, `chat_scroll`, `is_generating`, `is_reviewing`, `spinner_tick`, `show_terminal`, `session_start_time`, `total_llm_calls`, `average_latency`, `ollama_health_entries` (shared by dashboard + provider health), `last_health_check`, `health_check_in_progress`, plus the terminal handle fields.

**`Panels`** — one struct grouping all per-panel state (each panel's fields are its private struct fields):

Moves into `Panels` (not into `App`):
- `bytebot_*` (8 fields) → `ByteBotPanel`
- `collab_*` (9 fields) → `CollaborationHubPanel`
- `voice_*` (8 fields) → `VoiceInterfacePanel`
- `term_asst_*` (6 fields) → `TerminalAssistantPanel`
- `sec_*` (7 fields) → `SecurityAuditorPanel`
- `profiler_*` (5 fields) → `PerformanceProfilerPanel`
- `models_*` (5 fields) → `CustomModelsPanel`
- `learn_*` (13 fields) → `LearningModePanel`
- `lang_*` (7 fields) → `MultiLanguagePanel`
- `commit_*` (2 fields) → `GitCommitPanel`
- `settings_*` (4 fields) → `SettingsPanel`
- `feature_nav_selected` → `FeatureNavigatorPanel`

**Ownership / borrow approach** (concrete, not hand-wavy — this is the riskiest part so it's pinned down):

The core aliasing problem: a panel needs `&mut` to both its own state AND shared `App` state (ByteBot pushes to `app.messages`; Settings mutates `app.config` + `app.theme`; Security reads `app.file_tree`). Naively passing `&mut App` while the panel is itself a field of `App` is a borrow error.

**Chosen design: separate `Panels` struct, NOT owned by `App`.**

```rust
// Panel-local state lives OUTSIDE App, fully decoupled.
pub struct Panels {
    pub bytebot: ByteBotPanel,
    pub security: SecurityAuditorPanel,
    pub settings: SettingsPanel,
    // ... one field per overlay panel
}

pub struct App {
    // shared state only — no panel fields here
    focus: FocusArea,
    config: XencodeConfig,
    theme: ThemeColors,
    memory: ConversationMemory,
    file_tree: Vec<String>,
    messages: Vec<UiMessage>,
    // ...
}

// In run_app: own both side by side, never nested.
let mut app = App::new();
let mut panels = Panels::new();

// Dispatcher: pass &mut App (shared) AND &mut panels.<specific> (panel-local).
// No aliasing — they are different variables.
match app.focus {
    FocusArea::ByteBotPanel => {
        panels.bytebot.handle_key(&mut app, key, tx);
    }
    // ...
}
```

Each panel's `handle_key`/`handle_message` takes `&mut self` (panel state) + `&mut App` (shared state). Because `app` and `panels.bytebot` are **separate stack variables**, there is no aliasing — the borrow checker is satisfied trivially. This is simpler and more flexible than the disjoint-field-borrow alternative and needs no `PanelCtx`/`AppMut` indirection.

Fallback (only if Step 3 reveals a problem): if a panel ever needs to switch focus to another panel (e.g. FeatureNavigator → target panel), which requires mutating a *different* panel field, the dispatcher returns an enum `PanelAction { SwitchFocus(FocusArea), Quit, ... }` from `handle_key` and `run_app` performs the cross-panel mutation centrally. Panels never touch siblings directly.

This pattern matches what the current `run_app` loop does (one `&mut app` match arm per focus area) — the refactor just splits the panel-local fields out of `App` into `Panels`, making ownership explicit.

---

## Migration Strategy

The TUI must remain **functional and building at every step**. Each numbered step is one commit; `cargo build --workspace` must be green after each. No step is allowed to leave the crate in a non-compiling state.

### Step 0 — Extract leaf modules (pure code move, zero behavior change)
Create `theme.rs`, `focus.rs`, `widgets/gauge.rs`, `widgets/spinner.rs`, `input.rs`. Move the `ThemeColors` impl (`app.rs:66-186`), the `FocusArea` enum + `navigate_feature()` (`app.rs:25-64, 587-604`), the inline `bar()`/`gauge_block()` closures (duplicated across `ui.rs` performance/security/profiler panels), and the spinner-frame `char` arrays (duplicated 8+ times). Update imports. **Verify: build green, smoke-test 3 panels render identically.**

### Step 1 — Extract `layout.rs` and `editor.rs` (pure code move)
Move `draw()`, `draw_header`, `draw_status_bar`, `draw_body`, `centered_rect` into `layout.rs`. Move `open_file_in_editor`, `save_editor`, the `UiMessage` struct into `editor.rs`. **Verify: build green, full layout renders identically.**

### Step 2 — Extract `terminal.rs` and fix bug 3 (render stub)
Move `spawn_terminal`, `close_terminal`, `write_terminal`, the `terminal_child` field + the `[TERM_EXEC]` reader. Replace the `draw_terminal` stub with a real renderer: decode `terminal_output` (`Vec<Vec<u8>>`) as ANSI → ratatui lines via the existing `ansi-to-tui` dependency, render the input line + cursor. **Verify: build green, toggle Ctrl+T and confirm output displays; run `dir` or `ls` and see real output.**

### Step 3 — Introduce `Panel` trait + migrate 3 simplest panels
Add `panels/mod.rs` with the trait + dispatch helpers. Migrate `project_analyzer`, `feature_navigator`, `performance_dashboard` first — they are read-only (no async messages, minimal key handling), making them the safest validators of the trait shape. Move their `draw_*` fn, their state fields, their key/message handling into `panels/*.rs`. **Verify: build green, each panel opens/responds identically.**

### Step 4 — Migrate message-driven panels (fixes bugs 1 + 4)
Migrate one at a time: `security_auditor`, `bytebot`, `performance_profiler`, `collaboration_hub`, `voice_interface`, `terminal_assistant`. Each carries its `[TAG]` message handler into `handle_message`. During `bytebot` migration, fix bugs 1 + 4:
- Enter handler calls `run_bytebot(tx)` (guarded on non-empty command), then pushes the command onto `bytebot_history`.
- Up/Enter recall now has history to read.
**Verify after each: build green, open the panel, trigger its flow, confirm identical (or bug-fixed) behavior.**

### Step 5 — Migrate interactive/edit panels (fixes bug 2)
Migrate `settings`, `git_commit`, `custom_models`, `learning_mode`, `multi_language`. During `settings` migration, fix bug 2: swap the Enter-handler cursor checks so `== 6` → URL editing, `== 7` → factory reset, matching the list order and the Left/Right handlers. **Verify: build green; for settings, confirm cursor 6 → URL edit and cursor 7 → reset (manual repro).**

### Step 6 — Migrate the core overlay panels
Migrate `code_review`, `model_selector`. These share more `App` state (selected file, default model) but fit the trait cleanly. **Verify: build green.**

### Step 7 — Final cleanup
Delete `ui.rs` (all rendering now lives in `layout.rs` + `panels/`). `app.rs` is now dispatch-only (~400 lines). Run `cargo clippy --workspace` to fix warnings, run full smoke test across all 23 focus areas. **Verify: build green, clippy clean, every panel reachable and responsive.**

---

## Bug Fix Specifications (acceptance criteria)

These are the only behavioral changes. Each is verified explicitly.

### Bug 1 + 4 — ByteBot execute + history
- **Location**: `panels/bytebot.rs` (after migration from `app.rs:1693-1701`).
- **Fix**: In `handle_key` for `KeyCode::Enter` when `focus == ByteBotPanel` and `!bytebot_running`:
  1. If `bytebot_command.trim().is_empty()`, do nothing.
  2. Else call `run_bytebot(tx)` (which sets `bytebot_running = true`, seeds steps, clears the command field).
  3. Push the trimmed command onto `bytebot_history` before `run_bytebot` clears it.
- **Current broken behavior** (must not regress): Enter currently only recalls history; it never executes.
- **Acceptance**: type a command, press Enter → steps animate, progress bar fills, log populates; pressing Up later recalls the command from the history list.

### Bug 2 — Settings index swap
- **Location**: `panels/settings.rs` (after migration from `app.rs:1761-1774`).
- **Fix**: swap the two `settings_cursor` checks in the Enter handler:
  - `settings_cursor == 6` → enter URL-editing mode (set `settings_url_editing = true`, seed `settings_url_buffer` from `config.ollama_url`).
  - `settings_cursor == 7` → factory reset (restore `XencodeConfig::default()`, recompute theme, set `settings_reset_active`, save, return to ChatInput).
- **Consistency requirement**: after the fix, Enter matches the Left/Right handlers (`app.rs:1909, 1948`) which already use `== 6` for URL — both Enter and arrows agree on every row.
- **Acceptance**: move the settings cursor to the "Ollama URL" item (array index 6) + Enter → enters URL editing (yellow buffer). Move to "Factory Reset" (array index 7) + Enter → resets config. No accidental config wipe from selecting URL.

### Bug 3 — Terminal render stub
- **Location**: `terminal.rs` render function (after migration from `ui.rs:385-395`).
- **Fix**: render the popup using real data:
  - Decode each `Vec<u8>` in `terminal_output` as UTF-8 lossy, apply ANSI escape parsing (`ansi-to-tui` crate, already a dependency), produce ratatui `Line`s.
  - Render the input prompt line: `"> "` + `terminal_input` with a visible cursor at `terminal_cursor`.
  - Apply `terminal_scroll` as a vertical scroll offset.
- **Acceptance**: Ctrl+T opens pane, type `dir` (Windows) / `ls` (Unix), Enter → real command output appears in the pane. Previously showed "coming soon."

---

## Testing & Verification

Phase 1 is a refactor, so behavior must not change except for the three intentional fixes.

### Per-step verification (every commit)
1. `cargo build --workspace` — must exit 0.
2. `cargo clippy --workspace` — must be clean (catches unused imports, move errors, shadowed bindings introduced during extraction).
3. Manual smoke test: launch TUI via `cargo run -p xencode-cli -- tui`, open each panel touched in that step, confirm it renders and responds identically to pre-refactor.

### Unit tests for the bug fixes
Added in the panel modules that own each fix:
- `panels/bytebot.rs`: `test_enter_executes_bytebot` — construct a `ByteBotPanel`, set `bytebot_command`, send Enter, assert `bytebot_running == true` and `bytebot_history` contains the command.
- `panels/bytebot.rs`: `test_enter_empty_command_noop` — empty command + Enter leaves `bytebot_running == false`.
- `panels/settings.rs`: `test_cursor6_opens_url_edit` — `settings_cursor == 6` + Enter sets `settings_url_editing == true`, does NOT reset config.
- `panels/settings.rs`: `test_cursor7_factory_resets` — `settings_cursor == 7` + Enter restores defaults and sets `settings_reset_active`.

The terminal-render fix (bug 3) is visual and verified manually; no unit test (rendering output is not easily asserted without a snapshot harness, which is out of scope).

### What is explicitly NOT tested in Phase 1
- Pure code-move extractions (steps 0–1, 3–7 structural parts): covered by smoke test. No unit tests for moved code.
- Full panel behavior: deferred to Phase 2 once panels have real backends worth testing.
- Snapshot/visual regression testing: out of scope for Phase 1.

---

## Out of Scope (explicitly deferred)

The following are real issues but are NOT addressed in Phase 1. They belong to later phases:

| Issue | Phase |
|-------|-------|
| Wire security auditor to real `analysis-rs` scanner | Phase 2 |
| Wire project analyzer to real `core-rs` stats | Phase 2 |
| Real Ollama model list (currently hardcoded) | Phase 2 |
| Real git diff/commit wizard | Phase 3 |
| Onboarding modal | Phase 3 |
| Diff viewer, code-execution panel | Phase 3 |
| Learning mode lesson progression | Phase 4 |
| Multi-language real translation | Phase 4 |
| Custom-models real persistence | Phase 2 |
| Voice real STT | Phase 4 |
| Collaboration real sync (WS client + `collaboration-rs`) | Phase 4 |
| Real profiler crate | Phase 4 |
| Provider retry/backoff, HealthTracker propagation | Phase 2 |
| Removing the Python TUI | Phase 5 |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Borrow-checker friction from panels needing both `&mut self` and shared `App` state | Medium | Use disjoint-field borrows; where ugly, pass narrow `&mut` references to the specific shared fields a panel needs. Validate the pattern in Step 3 (simplest panels) before applying to complex ones. |
| Behavior drift during pure code moves (steps 0–1) | Low | Smoke test after each; no logic changes in those steps. |
| Trait shape wrong, forcing rework after several panels migrated | Medium | Migrate the 3 simplest panels first (Step 3) to validate the trait before committing the complex panels. |
| Message routing regression (sentinels like `[DONE]` mis-routed) | Medium | Keep the prefix table explicit and unit-testable; the ByteBot/Settings tests cover the two highest-risk routing paths. |
| Terminal ANSI rendering differs from the captured bytes on Windows | Low-Medium | `ansi-to-tui` is already a dependency and handles standard escapes; smoke-test with `dir`/`echo` on the user's Windows environment. |

---

## Success Criteria for Phase 1

1. `cargo build --workspace` exits 0.
2. `cargo clippy --workspace` is clean.
3. `xencode-tui-rs/src/ui.rs` is deleted; all rendering lives in `layout.rs` + `panels/`.
4. `app.rs` is dispatch-only (target ≤ 500 lines, down from 2100).
5. All 23 `FocusArea`s reachable and responsive in a full smoke test.
6. The 4 bug-fix unit tests pass.
7. The 3 critical bugs are fixed and verified by manual repro (ByteBot executes, settings rows match, terminal shows real output).
8. No new features added, no backends wired — the diff is structural + fixes only.
