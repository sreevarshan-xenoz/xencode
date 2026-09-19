# TUI UI Improvement Backlog

> Inventory of `rust/crates/xencode-tui-rs/` verified against the tree on
> 2026-09-19. Every item cites `file:line` — re-verify before starting work,
> as these line numbers drift. Grouped by theme, rough priority top → bottom.

## 1. Dead / drifted modules (do first — everything else builds on these)

- [ ] **Abandoned refactor: unused modules.** `widgets/` (spinner, gauge),
  `channel.rs`, `input.rs`, `focus.rs`, `theme.rs` are declared in `lib.rs`
  but imported by nothing (only live imports: `app.rs:27→ui`, `ui.rs:924→review`,
  `app.rs:284→review`). Either wire them in or delete them.
- [ ] **focus.rs has drifted from app.rs**: missing `ReviewDashboard`, 13 vs 14
  `FEATURE_LIST` entries (`focus.rs:14-36` vs `app.rs:42-66`). Single source of truth.
- [ ] **Theme duplicated verbatim**: `theme.rs:6-125` vs `app.rs:85-205`.
- [ ] **Inline spinners/gauges duplicate widgets/**: 11 hardcoded frame arrays in
  `ui.rs` (376, 422, 893, 1367, 1512, 1604, 1774, 1898, 2089, 2118, 2484) and a
  local `bar` closure (`ui.rs:1055`) instead of `widgets::gauge::bar`.

## 2. Correctness bugs

- [ ] **ByteBot Enter clobbers typed command** with history entry instead of
  executing it (`app.rs:3659-3667`).
- [ ] **Blocking `git commit -am` on the UI thread** — TUI freezes on slow repos
  (`app.rs:3650-3652`); move to background task (Milestone D-1 infra).
- [ ] **`FocusArea::Terminal` unreachable**: never assigned, pane is a
  "coming soon" stub (`app.rs:3463`, `ui.rs:464-465`). Wire it or remove it.
- [ ] **`file_scroll_offset` set but never used** (`app.rs:272`) — file explorer
  scrolling silently broken.
- [ ] **Unbounded scroll**: `provider_health_scroll`/`security_scroll` have no
  max clamp (`app.rs:3591, 3594`) — scrolling past the end renders blank.
- [ ] **CodeReview output not scrollable** (`ui.rs:917-921`, no `.scroll`) —
  long diffs unreachable.
- [ ] **Mouse wheel dead in 12+ focus areas**: only 7 areas handle wheel events,
  rest fall into `_ => {}` (`app.rs:4293`).

## 3. Theme & visual consistency

- [ ] **Theme leakage**: ~63 raw `Color::` uses in `ui.rs`, ~40 hardcoded named
  colors at render sites (293, 485, 504-508, 738-742, 1254-55, 1598-1611,
  1983-85, 2391-94, …) instead of `ThemeColors` slots. Funnel through theme.
- [ ] **No light theme**: 7 dark palettes only (`theme.rs:6-125`). Add a light
  variant + `xencode config` toggle (theme cycling is triplicated today:
  `ui.rs:574`, `app.rs:4026, 4108`).
- [ ] **Magic-number layout**: fixed 20/50/30 body split (`ui.rs:178-210`) and a
  *second* set of 20%/70% constants re-derived in mouse click handling
  (`app.rs:4322-4324`) — one layout fn should produce both.
- [ ] **Settings nav count hardcoded** `14` (`app.rs:3565`) vs the actual
  settings tuple list (`ui.rs:687`) — new rows silently unreachable.

## 4. Chat & rendering UX

- [ ] **No markdown rendering** for streaming chat: tokens appended as plain
  `Line`s with `Wrap` (`ui.rs:357-370, 401-404`). Code blocks are indistinguishable
  from prose — biggest visible win. (ratatui has no native md; hand-roll a
  block/heading/inline-code splitter or add `tui-markdown`-style renderer.)
- [ ] **No toasts/notifications layer**: file-watch warnings are injected as fake
  system chat lines (`app.rs:440-449, 1319`) — pollutes conversation history.
  Add a transient overlay.
- [ ] **No input history recall**: Up scrolls chat instead (`app.rs:3524-3526`);
  no autocomplete anywhere (grep: no hits). Chat is single-line `String+cursor`
  (`app.rs:265-266`) — multiline input via the existing `tui_textarea` (already
  used by the code editor, `app.rs:275`) would fix paste/wrapping/editing.
- [ ] **No help overlay**: no `F1`/`?` screen anywhere (grep: no hits); hints are
  only per-focus status lines (`ui.rs:141-163`). Keybinding help panel needed,
  especially with one giant key match (`app.rs:3366-4252`) and no keymap table.

## 5. Structure / maintainability

- [ ] **Giant key-match in `run_app`** (887 lines, `app.rs:3366-4252`): extract a
  keymap table or per-focus `handle_key` fns.
- [ ] **One `match app.focus` overlay render** with 34 `draw_*` fns in `ui.rs`
  (33-53): fine short-term; consider a `Panel` trait if panels keep growing.
- [ ] **Resize is a no-op handler** (`app.rs:4340`) — works because layout
  recomputes per frame, but scroll offsets aren't clamped on shrink; add clamping.

## Suggested order

1 (dead code) → 2 (bugs) → 4 (markdown + help overlay = biggest user-visible win)
→ 3 (theme) → 5 (refactors, ride along with feature work).
