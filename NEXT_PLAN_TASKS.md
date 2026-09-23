# Next Plan — Task Checklist

> Working task list for the active backlog. See [NEXT_PLAN.md](NEXT_PLAN.md) for
> the milestone overview and [docs/ROADMAP.md](docs/ROADMAP.md) for the long-term roadmap.
> All items target the Rust workspace (`rust/crates/*`) per `AGENTS.md`. Verified 2026-09-19.

## Rust Migration — Complete ✅

- [x] Port core, config, cache, memory crates — `xencode-core-rs`, `-config-rs`, `-cache-rs`, `-memory-rs`
- [x] TUI foundation (ratatui, `rust/crates/xencode-tui-rs/`)
- [x] Model providers with retries (`rust/crates/xencode-providers-rs/`)
- [x] Repo-wide context / RAG — `xencode-context-rs` (index, embed, retrieve, budget)
- [x] Server / collaboration / plugin crates — `xencode-server-rs`, `-collaboration-rs`, `-plugin-rs`
- [x] Analysis + security scanning — `xencode-analysis-rs`
- [x] Tool-calling + model capabilities — `generate_stream_with_tools`, `ModelCapabilities`
- [x] CLI subcommands — scan, config, models, cache, query, memory, tasks, worktree, advise, server, analyze, fetch, review, plugin, llamacpp, tui
- [x] Workspace gates green — 15 crates, 805 tests passing, 4 ignored, zero warnings

## Real-Time Intelligence (Phase 3+)

- [x] Real-time file watcher with proactive warnings (`rust/crates/xencode-context-rs/`, new watcher module) — `WorkspaceWatcher` on `notify`, debounced; TUI surfaces warnings for tracked/attached/open files
- [x] Live refactor suggestions on the symbol graph (`advise` module: cycles, hubs, orphans; `/advise [filter]` in the TUI)
- [x] Proactive bug warnings ("you just introduced a bug") (`broken_imports`, `affected_dependents` → dependents named in watch warnings)

## Multimodal UX

- [x] Image input pipeline (`rust/crates/xencode-analysis-rs/` `images` module: magic-byte detect, header dimensions, data URLs; `analyze` CLI surfaces inventory; `MessageContent` parts flow end-to-end through Ollama/Anthropic/Gemini/OpenAI-compatible renderers; TUI attach sends images as message parts)
- [x] Document parsing into context (`documents` module in `rust/crates/xencode-context-rs/`: PDF via pdf-extract, DOCX via zip+w:t runs, size/char caps; TUI attach inlines extracted text)
- [x] Web extraction for research (`web` module: timeout/capped fetch, content-type gate, HTML→text; `fetch` CLI subcommand)

## Secure Team Mode

- [x] Audit log coverage for team actions (`workspace` manager: sequenced `AuditEvent`s incl. denials, `audit_log`/`events_for`)
- [x] Workspace RBAC hardening (`rust/crates/xencode-collaboration-rs/workspace.rs`: Admin-gated membership, self-leave, last-admin guard)
- [x] Collaboration session security review and polish (Milestone G: real
  bearer tokens with expiry, RBAC-backed first-frame WS auth, enforced session
  size, local-first bind with opt-in TLS, persistent JSONL audit, and a TUI
  hub that actually connects — details in G1–G3 below)

## Git Loop Completion

- [x] PR review dashboard / PR-level review browsing in the TUI
  (per-file Code Review panel exists in `rust/crates/xencode-tui-rs/`;
  CLI triage shipped: `review` command over rename-aware diff helpers;
  TUI per-file diff browsing shipped: `review` module + `ReviewDashboard`
  focus (`Ctrl+Y`, Feature Navigator entry) with base toggle HEAD<->main)

## Milestone D — Background Tasks & Worktree Support (drafted 2026-09-19)

> Ground rules for every item: Rust-only under `rust/crates/*`; unit tests with each
> module; `cargo test --workspace` green + zero warnings before commit; one atomic
> commit per task; docs (`README.md`/`CLI_GUIDE.md` + `CHANGELOG.md`) updated in the
> final docs task.

### D1 — Background task core

- [x] D1-01 `tasks` module in `xencode-core-rs`: `TaskRecord` (id, name, command, pid,
  status Running/Exited(s)/Killed, capped output tail), tokio-backed spawn, registry
  with `start`/`poll`/`stop`/`list`/`remove`. Pure state transitions unit-testable
  without real processes. Shipped as two layers: `TaskStore`/`TaskRecord` (pure, 6
  unit tests) and `TaskManager` (owns `tokio::process` children, stdout+stderr drained
  by reader tasks into a 500-line cap; `remove` refuses Running *before* touching the
  child map so `kill_on_drop` can't silent-kill a live task). 5 `#[tokio::test]`
  lifecycle tests cover echo→Exited(0)+output, `exit 3`, sleep→stop→Killed→rm, and
  NotFound/AlreadyFinished paths.
- [x] D1-02 Tool surface: `background_start`/`background_poll`/`background_stop` as
  `ToolDefinition`s in `xencode-providers-rs/tools.rs` shapes, and wire tool-call
  **execution** into the agent turn loop (`generate_stream_with_tools` exists in the
  providers; the TUI loop currently never handles a `ToolCall` — that lands here).
  Shipped: `background_tools()` schema builder in providers; new TUI `agent_tools.rs`
  with a shared `TaskRuntime` (`Arc<tokio::sync::Mutex<TaskManager>>` on `App`) and
  `execute_tool_call`; the chat loop now runs up to `MAX_TOOL_ROUNDS` (8) tool rounds
  per turn before a final tool-less answer, echoing each call/result into chat as a
  `⚙` system line. Backends without native tool support degrade as before (schemas
  only — Anthropic/Gemini return no calls).

### D2 — Background tasks UX

- [x] D2-01 TUI `tasks.rs` panel modeled on `review.rs`: `FocusArea::TaskManager`,
  Feature Navigator entry + global hotkey, task list with status colors, Enter →
  output detail pane (scrollable), `x` → stop, `d` → remove finished.
  Shipped in `ui.rs::draw_task_manager` (panel lives with the other overlays; no
  separate `tasks.rs` was warranted at this size): `Ctrl+K` toggle + Navigator
  entry #14, status-colored rows (▶ info / ✓ success / ✗ danger / ⊘ warning) read
  via non-blocking `tasks_snapshot()`, `x`/`d` dispatch `[TASKS]stop|id` /
  `[TASKS]rm|id` through the event channel so keys never block the UI thread,
  detail scroll re-clamped on resize (shared line builders, E6-02 pattern).
- [x] D2-02 CLI: `xencode tasks list | start <cmd> | poll <id> | stop <id> | rm <id>`.
  Since a CLI process can't see the TUI's in-memory registry, this ships a
  file-backed twin (`xencode-core-rs::tasks_file`): records in `.xencode/tasks/tasks.json`,
  each command wrapped in an `sh` script whose EXIT trap writes `<id>.exit` (survives
  `exit N` inside the command), merged stdout+stderr in `<id>.out`, status derived on
  read from exit file + killed flag + `/proc` liveness (zombies count as dead). `stop`
  signals `kill` and marks the record; `rm` refuses running tasks. Root is the current
  directory's `.xencode/tasks` (same layout as `xencode init`).
- [x] D2-03 Tests: registry lifecycle (start→exit→rm, stop of dead task, output cap)
  + headless frame renders for the panel. Most of this landed with D1/D2 and was
  verified rather than duplicated: output cap (store eviction unit test + bounded
  `poll` tail), stop-of-dead / double-stop (`AlreadyFinished`, both registries),
  file-registry full lifecycle incl. persisted ids, atomic saves and torn-read
  safety (7 `tasks_file` tests), headless renders (populated `Ctrl+K` panel across
  sizes and detail modes, `x`/`d` key dispatch). Added here: the natural
  start→exit(0)→remove lifecycle on `TaskManager` (killed-path removal was covered,
  clean-exit removal was not) and the receive side of the `[TASKS]` protocol
  (`handle_tasks_command` junk-body no-ops + real stop→rm against the registry).
  The `xencode tasks` CLI itself is covered by manual end-to-end runs (list/start/
  poll/stop/rm + error exits), not unit tests — `run_tasks` is thin glue over the
  registry against the real cwd.

### D3 — Worktree support

- [x] D3-01 Git helpers in `xencode-context-rs` (`worktree.rs`, alongside `gitinfo.rs`):
  parse `git worktree list --porcelain` (pure fn), add/remove shells with explicit
  args; dirty marker reuse via `dirty_paths`.
  Shipped as `parse_worktree_list` (pure; handles spaces in paths, detached/locked/
  prunable/bare markers, `is_main` on the first block) plus `worktree_list`/
  `worktree_add`/`worktree_remove` shells built from explicit arg vectors — `git_stdout`
  made `pub(crate)` in `gitinfo.rs` so worktrees report failures exactly like diffs.
  Dirty state per worktree is a caller-side `dirty_paths(wt.path)` check (D3-02);
  tests include a real `git init` → add → remove round trip.
- [x] D3-02 TUI `WorktreePanel`: list worktrees (path, branch, HEAD, dirty flag),
  add (path+branch prompt), remove with confirm; guard: main worktree never
  removable. Same panel pattern as D2-01.
  `Ctrl+O` toggle + Navigator entry #16; rows show ★ main / ⚡ dirty (via
  `dirty_paths`) / ◯ clean with branch, short HEAD and path. `a` opens a two-stage
  prompt (path → branch, empty branch lets git name it), `d` → y/N confirm —
  removal of the main worktree is refused before any git call, and dirty linked
  worktrees stay protected by git itself (no `--force`). `r` refreshes; git calls
  are synchronous, matching the `Ctrl+G` refresh precedent.
- [x] D3-03 Agent integration: background tasks (D1) take an optional `cwd` so a task
  can run inside a chosen worktree; worktree list included in the context bundle.
  `TaskManager::start_with_cwd` (plain `start` delegates with `None`), the
  `background_start` tool schema gained an optional string `cwd`, and the tool result
  echoes `in <dir>` so the model knows where it ran (missing directory = `Spawn`
  error string, no panic). `git_summary_text` now appends a `Worktrees:` block —
  path, branch, `[dirty]`, `[main]`, capped at 8 — but only when the repo actually
  has more than one worktree.
- [x] D3-04 CLI: `xencode worktree list | add <path> [<branch>] | remove <path>`.
  Thin glue over the D3-01 helpers against the current directory: add with a branch
  checks that branch out (git's error if it doesn't exist), without one lets git name
  a new branch after the directory; remove is refused for the main checkout and for
  dirty worktrees (git's own guard — no `--force`). Verified live end-to-end (add ×2,
  list, dirty-remove error + clean remove) in a scratch repo; no unit tests added —
  the parsing/shell layers are already covered in `worktree.rs`.

### D4 — Close-out

- [x] D4-01 Docs sweep: README structure + CLI_GUIDE subcommands, NEXT_PLAN.md counts,
  CHANGELOG entry; `cargo test --workspace` count recorded.

  Live figures at commit: **494 tests passed, 0 failed, 4 ignored, 13 crates**;
  `cargo clippy --workspace --all-targets` clean. Sweep covered: README (23 panels,
  Tasks/Worktree rows, counts), CLI_GUIDE (`tasks`/`worktree` sections, essentials
  keys), USER_MANUAL (command reference re-synced to real `--help`, Background Tasks
  + worktree docs), NEXT_PLAN.md (Milestone D marked complete; stale claims fixed —
  watcher and multimodal inputs are shipped, PR-review backlog entry corrected,
  team-workflow item reworded against the real `xencode-collaboration-rs`), and the
  stale CLI-subcommand checklist line above refreshed from `xencode --help`.

## Milestone E — UI Systematic Fixes (drafted 2026-09-19)

> Findings were audited 2026-09-19 and are inlined in each E-task below with
> file:line citations (the standalone `docs/UI_IMPROVEMENTS.md` backlog was
> removed by request in `5fcf571`; recoverable from history if ever needed). Same ground rules as Milestone D (Rust-only,
> tests per task, zero warnings, atomic commits, docs in close-out).
> Dependency note: E2-02 (non-blocking git commit) is trivial on its own but becomes
> free if Milestone D's D1-01 task core lands first — sequence accordingly.

### E1 — Reconcile dead modules (UI doc §1) — foundation for everything else

- [x] E1-01 Single source of truth for focus/feature list: moved the live
  `FocusArea`/`InputMode`/`FEATURE_LIST`/`navigate_feature` from `app.rs` into
  `focus.rs` (deleted the drifted 13-entry copy); `app.rs` re-exports, `ReviewDashboard`
  already reachable via Feature Navigator.
- [x] E1-02 Single theme source: deleted the verbatim duplicate in `app.rs`
  (byte-identical diff verified first), `app.rs` now re-exports `theme::ThemeColors`.
- [x] E1-03 Adopt `widgets/`: all 11 inline spinner frame arrays in `ui.rs` now use
  `widgets::spinner::frame`/`SPINNER_FRAMES`; the `bar` closure and the bytebot/voice
  gauge duplicates now use `widgets::gauge::bar`. Bars with different shapes
  (profiler ticks, learning gauges) intentionally left per `gauge.rs` doc note.
- [x] E1-04 Gate: dead modules deleted — `channel.rs`, `input.rs` (zero references
  verified). `src/` is now: app, focus, review, theme, ui, widgets — all live.
  Workspace after deletion: 417 passed, 0 failed, zero warnings (README counts synced).

### E2 — Correctness bugs (UI doc §2)

- [x] E2-01 ByteBot Enter executes the typed command instead of replacing it with
  history (`app.rs:3659-3667`); regression test `bytebot_enter_executes_typed_command_not_history`.
  Fix landed in `8956914`.
- [x] E2-02 Git commit off the UI thread (`app.rs:3650-3652`) — `tokio::process`
  spawn; result reported as a `[GIT_COMMIT_OK/ERR]` system chat line via the existing
  event channel, then `refresh_git()`. Helper `first_output_line` unit-tested.
- [x] E2-03 Removed dead `FocusArea::Terminal` (never assigned): enum variant
  (`focus.rs`), dead close-strip focus reset (`app.rs` Ctrl+T arm), and the
  `small_terminal_render` FOCI entry. The Ctrl+T terminal strip pane itself is
  untouched (driven by `show_terminal`, not focus). That test's FOCI list also gained
  the missing `ReviewDashboard` entry — verified it renders at all swept sizes.
- [x] E2-04 Scroll fixes: deleted never-read `file_scroll_offset` (explorer list
  already auto-scrolls to selection via `ListState`); provider-health/security/code-review
  paragraphs clamp their stored scroll at render time with `ui::clamp_scroll`
  (unit-tested) so scrolling past the end no longer renders blank; CodeReview output
  is now scrollable via `review_scroll` + ↑/↓, reset on each new review.
- [x] E2-05 Mouse wheel now drives every focus area that has scroll state:
  Settings/ModelSelector/CustomModels cursors and the (new, E2-04) CodeReview scroll
  joined the existing 7. Remaining areas are single-screen panels with no scrollable
  content — wheel is a deliberate no-op there.
- [x] E2-06 (found while auditing keys for E3-02) Text fields were untypable: the
  universal `space`/`i`/`/`/`m`/`s`/`q`/`?` arms fired before the char-insert arm, so
  GitCommit/ByteBot/settings-URL buffers couldn't contain a space or those letters, and
  `q` quit mid-message. Guarded by a new `App::text_entry_active()` (unit-tested); also
  fixed Left-arrow *deleting* (it now moves the cursor back — Backspace is the delete key)
  and made the VoiceInterface `m`-mute reachable. Was still open at the time:
  `s` remained a Settings shortcut globally, so the SecurityAuditor/CustomModels
  `s` branches stayed unreachable — the binding conflict (plus the missed `j`/`k`
  swallowing in text fields) was resolved in E6-01 via focus-first dispatch.

### E3 — High-impact UX (UI doc §4, first half)

- [x] E3-01 Markdown rendering in chat: new `markdown.rs` (no new deps) renders
  fenced code (with `┌─ lang`/`└─` frame, correct even mid-stream when the fence is
  still open), headings, bullets/ordered lists, block quotes, rules, and inline
  `code`/**bold**/*italic* as styled `Line`s; unclosed inline markers stay literal.
  Assistant messages in `draw_messages` use it; user/system stay plain. 6 unit tests
  + the small-terminal render sweep cover it.
- [x] E3-02 Help overlay (`?` or `F1`): new `help.rs` lists real keybindings —
  the current panel's keys, plus universal/global/editing sections — from an
  audited inventory of the actual `run_app` handler (no fiction). Modal: Esc/`?`/`F1`
  close, ↑↓/jk scroll (`help_scroll`, render-clamped), all other keys swallowed.
  `?:help` added to the default status hint. 2 unit tests + a help-overlay render sweep.
- [x] E3-03 Toast/notification layer: new `toast.rs` — file-watch warnings now push a
  transient top-right overlay (`Toast`, 6s TTL, dedup-refreshed, render-capped to 4)
  instead of being injected as fake system chat lines. Pruned each loop tick; the
  repeat-suppression in `watch_warning_for` now compares the last visible warning toast.
  3 unit tests + a toast-exercised render sweep.

### E4 — Theme & layout consistency (UI doc §3)

- [x] E4-01 All raw `Color::` uses in `ui.rs` (exactly 63) funneled through new
  semantic `ThemeColors` slots: `success`/`warning`/`danger`/`info`/
  `accent_secondary` (diff markers, severities, ok/fail states, provider
  badges). Slots are initialized to the same ANSI colors in all 7 themes, so
  this is behavior-identical — themes (incl. the E4-04 light theme) now have
  real knobs to override. `ui.rs` now contains zero `Color::` constants (all
  `Color::Rgb` literals live in `theme.rs`); the only other raw constants are
  two `DarkGray` editor line-number styles in `app.rs` `App::new`/file-open
  (set before any theme is applied).
- [x] E4-02 One layout function feeding both render and mouse-hit testing:
  `ui::body_chunks` now owns the File Explorer/Code Editor/Chat split (20/50/30),
  `draw_body` renders its rects and the click handler focuses panels via
  `ui::body_hit_test` — the duplicated `term_width * 20 / 100` / `* 70 / 100`
  maths in `app.rs` is gone, so render and hit-testing cannot drift. Unit test
  tiles every column at 9 widths and asserts each hit lands on the pane drawn
  under it (panes also proven gap-free).
- [x] E4-03 Settings nav bound from the actual row list, not the hardcoded `14`.
  New `focus::SETTINGS_ROWS` (14 row names) is the single source of truth: `ui.rs`
  builds its padded labels from it and its value array is typed
  `[&str; SETTINGS_ROWS.len()]` so row/value drift is a compile error; the section
  ranges and the reset-row check derive from it too. Both nav bounds in `app.rs`
  (↓ key and wheel-down) use `SETTINGS_ROWS.len()`. One render test walks the
  cursor over every row.
- [x] E4-04 Light theme + dedupe of the theme-cycle lists. `theme::THEME_NAMES`
  (now 8 entries incl. `light`) is the single list: the duplicated 7-theme arrays
  in `draw_settings` and both Settings ←/→ key arms are gone, replaced by a pure
  `theme::cycle_theme(active, forward)` helper (unit-tested, wraps both ways,
  unknown names no-op). The light theme overrides every `ThemeColors` slot
  (white bg, dark accents, `Dark*`-style RGB status colors) and is settable via
  the existing `active_theme` config string — no whitelist in config-rs to
  update. Fixed a pre-existing display bug while there: the settings Theme row
  showed only 4 of the theme dots (`0..4` hardcoded). 2 theme unit tests +
  a light-theme render sweep over all panels.

### E5 — Input UX (UI doc §4, second half)

- [x] E5-01 Multiline chat input: `App::input` String + manual byte cursor
  (which corrupted the cursor on non-ASCII input) replaced by `App::chat_input`,
  a `tui_textarea::TextArea` (already the editor's dep). Enter submits,
  Alt+Enter inserts a newline, Ctrl+J works on terminals that mangle Alt
  (crossterm reports it as LF, which the textarea turns into a newline). All
  manual insert/backspace/arrow/Home/End handling deleted — the textarea owns
  editing, and arrow keys move between lines while editing while plain
  Up/Down in normal mode still scroll chat (unchanged). Tab still inserts 4
  spaces; box renders theme-styled (restyled on theme change/factory reset)
  with a scroll-adjusted cursor. Input title + help overlay document the
  chords. 1 unit test: multiline submit keeps `\n` in the prompt and clears
  the box.
- [x] E5-02 Input history + slash autocomplete. Sent prompts land in
  `App::input_history` (200-entry cap, adjacent duplicates skipped); Alt+Up/
  Down recalls them while editing (clamps at both ends, restores the stashed
  draft when returning past the newest), while plain Up/Down still scroll the
  chat in normal mode and navigate lines while editing. Tab on a lone `/...`
  first line completes the command via pure `complete_slash_token` (longest
  common prefix over `SLASH_COMMANDS` = the four commands `submit_message`
  actually intercepts; Tab on `/` toasts the list); anywhere else Tab still
  inserts 4 spaces. Help overlay gained a Slash-commands section (test pins it
  against SLASH_COMMANDS). 3 new tests (completion matrix, recall walk +
  dedup, help sync).

### E6 — Structure (UI doc §5) — ride along, last

- [x] E6-01 The ~900-line key `match` is gone from `run_app`: new `keymap.rs`
  holds `handle_key` = modal-help guard → global Ctrl chord table → per-focus
  `key_*` handlers (`key_settings`, `key_git_commit`, `key_security`, …19 of
  them) → remaining universal chords (`i / m s ? q`). Not a pure move — three
  deliberate, tested behavior changes came out of the restructure:
  (1) the E2-06 `s` conflict is resolved — focus handlers run before the global
  `s`, so SecurityAuditor sort-toggle and CustomModels save-profile are live
  (`s` still opens Settings everywhere else);
  (2) text fields are now 100% typable: `j`/`k` no longer move row cursors or
  recall history while typing in GitCommit/ByteBot/Settings URL editing (they
  insert — the old Up|`k`/Down|`j` arms swallowed them; ByteBot recall is ↑
  only now);
  (3) Settings ↑/↓ still navigate rows while editing, but `j`/`k` go to the
  buffer. 12 pin-tests in `keymap::tests` + help overlay entries updated.
- [x] E6-02 Clamp scroll offsets on `Event::Resize`: `ui::clamp_scrolls_on_resize`
  now runs on every Resize event. Panels already clamped at render time, but the
  *stored* offsets stayed oversized and re-exposed blank scroll-past-the-end when
  the terminal grew back. Chat/CodeReview/ProviderHealth/SecurityAuditor/Help
  offsets are all recomputed against the same geometry the draw functions use —
  via the shared line builders (`chat_lines`, `review_display_text`,
  `provider_health_lines` and the new `security_findings_lines` extraction), so
  clamp and render can no longer disagree. Unit test pins shrink-clamps, never
  re-inflates on grow, and reaches 0 when everything fits.

### E7 — Close-out

- [x] E7-01 Close-out sweep: CHANGELOG has an entry per E-item; counts
  refreshed to the live 455-test total. Fiction removed from the manuals:
  QUICK_START's `/help` `/models` `/model` `/clear` `/exit` don't exist in
  the TUI (only `/init` `/ctx` `/advise` `/bytebot` are intercepted — that
  list, plus the new editing keys, is what it now shows), USER_MANUAL's
  shortcut table was rewritten from the actual `keymap.rs` bindings
  (`Ctrl+H` runs a check, it doesn't open a dashboard; `i`/`e` semantics
  corrected; help overlay + multiline/history/Tab-completion keys added),
  the stale `h:refresh` status-bar hint for ProviderHealth (unbound key)
  fixed to `Ctrl+H`, README panel count corrected 17 → 21 (the actual
  `FocusArea` enum size).

> Status legend: `[x]` done, `[ ]` todo.

## Milestone F — Live Refactor Insights (complete 2026-09-20)

Goal (backlog item 2): make the deterministic `advise` engine *live* — the symbol/dep
snapshot stays current between `/init` runs, insights get a dedicated panel, and both
the user (`xencode advise`) and the model (a `repo_advise` tool) can reach them.

### F1 — Live graph

- [x] F1-01 `xencode-context-rs`: `refresh_rust_file(root, rel_path)` — re-extract
  symbols for one already-indexed `.rs` file from its current bytes (drop the record if
  the file is gone), rebuild `deps.json` via `build_graph`, and keep `files.json` +
  `manifest.json` entries (size/loc/mtime) consistent; a path absent from the index is
  a no-op (new files still need `/init`). Atomic writes via `write_atomic`. Unit tests
  on temp `.xencode` dirs: edit changes edges, delete drops record + edges, unknown
  file no-op.
  Shipped as `refresh::refresh_rust_file` returning `RefreshOutcome::{Updated(graph),
  NoOp}` (explicit enum instead of `Option<Vec<_>>` — reads better at call sites).
  Also added `.xencode` to the watcher's `DEFAULT_EXCLUDED_DIRS` so snapshot rewrites
  never feed watcher events back into a refresh. 4 unit tests on temp workspaces
  (edit rewrites edges + index sizes, removal drops record/entry/edges and keeps the
  manifest consistent, unknown/non-Rust no-op leaves bytes untouched, absolute paths
  accepted and a refreshed snapshot makes the next `/init` report `fresh`).
- [x] F1-02 TUI watcher wiring: `handle_watch_event` refreshes the snapshot for
  modified/removed `.rs` files before computing affected dependents, so toasts and
  `/advise` see the current graph without re-running `/init`. Tests with temp dirs.
  Shipped as the pure `live_refresh_snapshot(root, kind, path)` gate/helper — only
  modified|removed on `.rs` reach the disk, and refresh errors are swallowed so a
  broken snapshot can never kill the warning path. One integration-style test on a
  temp workspace pins edit-removes-edge, the kind/extension/unknown-path gates, and
  removal dropping the symbol record.

### F2 — Insights panel

- [x] F2-01 AdvisePanel (`Ctrl+L` — not the drafted `Ctrl+S`, which is taken by
  editor-save/GitCommit, and not `Ctrl+I`, which terminals alias to Tab): computes
  `advise()` from the (now live) snapshot on
  open; rows colored per kind (⚠ broken import / 🔁 cycle / 🧶 hub / 🕸 orphan),
  ↑↓/jk select, Enter → full-message detail, `o` opens the advised file in the editor,
  `r` recomputes, Esc unwinds detail → panel → chat. Feature Navigator entry #17,
  help overlay, focus/Esc plumbing, keymap tests + small-terminal render test.

  Findings list is a stateful `List` (viewport follows the selection; no manual
  scroll state), so the only resize-clamped offset is the wrapped detail body.
  `/advise` now renders through the same `refresh_advise()` the panel uses, so
  chat and panel can never disagree. 2 keymap tests + 1 render sweep (all five
  advice kinds, list/detail × populated/empty, oversized scroll); workspace at 502.

### F3 — Surfaces

- [x] F3-01 CLI: `xencode advise [FILTER] [--json] [--limit 40]` over the
  `.xencode` snapshot of the current directory.
  Shipped with the filter as positional `[FILTER]` (substring of the
  finding's file path), not the drafted `--filter` flag; `--limit 0` shows
  all findings; missing index is an actionable `error: no project index in
  … — start the TUI and run /init first` with exit 1, not an empty report.
  Shares the snapshot read path with the TUI (`compute_advise` mirrors
  `refresh_advise`). Unit test on a temp workspace (cycle + orphan
  present, filter narrows and non-matching filter empties, no-index error)
  plus a live smoke on a scratch repo: numbered text table, `--json`
  array, positional filter, `… +N more` overflow line, `--limit 0`,
  error/exit-1 path, and `--help` text.
- [x] F3-02 Agentic tool: `repo_advise` callable by the model in the chat tool
  loop (schema `advise_tools()` in `xencode-providers-rs` next to
  `background_tools`, executor in `agent_tools.rs`) returning the compact
  advice report (optional `filter` arg; capped at 40 findings with a
  `… +N more` line; missing index comes back as an `error: …` string, never
  a panic). Tools are offered on every round except the final tool-less one.
  Instead of a third copy of the snapshot reader, this shipped as
  `advise_from_snapshot` in `xencode_context_rs` — the TUI panel
  (`refresh_advise`) and the CLI (`compute_advise`) now read through it too,
  with a `ContextError::NoIndex` variant carrying the user-facing message.
  Verified by unit tests against real on-disk `init_project` snapshots in
  temp dirs (cycle found, filter empties to the clean report, no-index
  error, cap boundary) plus a providers schema test; not driven through a
  live model run.

### F4 — Close-out

- [x] F4-01 Docs sweep: README/CLI_GUIDE/USER_MANUAL/NEXT_PLAN synced to what shipped,
  live test/crate counts recorded, CHANGELOG entry.
  Close-out pass re-verified every Milestone F claim against the tree: 13 crates
  and 508 tests match a fresh `cargo test --workspace` at this commit; `Ctrl+L`
  appears in the help overlay, both key tables and the CLI guide; the insights
  panel is FocusArea #24 / Feature Navigator entry #17; `xencode advise` matches
  `--help` exactly (positional `[FILTER]`, `--json`, `--limit` default 40,
  0 = all). NEXT_PLAN's backlog now carries only team-mode hardening.
> Status legend: `[x]` done, `[ ]` todo.

## Milestone G — Team Mode Hardening (drafted 2026-09-20)

Goal (backlog item 4): the collaboration RBAC/audit crate is currently orphaned —
zero consumers — while the server runs its own unauthenticated session model
(identity = username in the WS URL, decorative `/auth` routes, permissive CORS,
plain HTTP on 0.0.0.0) and the TUI hub is a pure simulation (hardcoded members,
fake telemetry, a false "WebSocket (TLS)" label). G wires the server to real
tokens + RBAC + persisted audit, hardens the bind posture with opt-in TLS, and
turns the hub into an actual WebSocket client.

### G1 — Server security core

- [x] G1-01 collaboration-rs: close the last-admin *demotion* hole (removal was
  guarded; role-change downgrades were not), audit both guards with `Denied`
  events, add `join()` (idempotent self-join as Editor), `create_workspace_with_id`
  (server-chosen ids), and `log_denied` for denials decided outside the mutators.
  5 new unit tests (24 in crate); workspace at 513.
- [x] G1-02 server-rs: real `TokenStore` (random `xencode_<uuid v4 hex>`, 24 h TTL,
  prune-on-issue, constant-time compare) + `Authed` bearer extractor; `/auth/login`
  rejects the silently-ignored `api_key` with a 400, `/auth/verify` becomes a real
  lookup; sessions + llamacpp load/unload require auth; unknown session ids now 404;
  permissive CORS deleted; `/api/llamacpp/status` and public `/api/config` stop
  leaking model/executable/args paths. 13 net-new tests (79 in server-rs);
  workspace at 526. The WS route still takes the username in its path and is
  unauthenticated — first-frame WS auth is G1-03.
- [x] G1-03 server-rs: WS route loses the username (`/ws/{session_id}`), first-frame
  `auth` with 5 s timeout, close codes 4401/4403/4404/4409; joins go through
  `WorkspaceManager::join`; `activity` relay is Editor+ and server-stamped;
  `MAX_SESSION_MEMBERS` (10) enforced, not advisory; shared `wire.rs` in
  collaboration-rs; the parallel in-memory session map is deleted. Duplex
  (no-ports) wire tests. Presence (`SyncCoordinator`, per connection) and
  membership (`WorkspaceManager`, per identity) are now distinct: roles survive
  disconnects, `members` frames carry live peers. 12 e2e handshake tests +
  5 wire round-trip tests (server-rs 88, collaboration-rs 29); workspace at
  540. The 5 s auth timeout itself is not timing-tested — every rejection path
  is covered, but a dedicated test would add 5 s to the suite for one branch.
- [x] G1-04 server-rs: `AuditSink` mirroring every mutation + denial to
  `~/.xencode/audit.jsonl` (`--audit-path`, `none` disables); one line per event,
  append-not-truncate across restarts, write failure disables the sink without
  taking down the session plane. Sink lives in server-rs (`audit.rs`), the
  collaboration crate stays IO-free; `AppState::new` defaults to a disabled
  sink so no test touches disk, and the CLI flag that points it at
  `~/.xencode/audit.jsonl` arrives with G2-01. Join refusals (no session 4404,
  session full 4409) now also write `Denied` events — the actor is
  authenticated by then. `AuditAction` serializes snake_case to match its
  `Display`. 6 sink unit tests + 1 wire-up e2e (audit lines on disk: order,
  strict seq, actor); workspace at 547.

### G2 — Network posture

- [x] G2-01 cli: `xencode server` gains `--host` (default 127.0.0.1), `--cert`/`--key`
  (rustls via axum-server), `--audit-path`, `--allow-insecure-public`;
  non-loopback bind without TLS refuses to start (pure `resolve_bind` unit matrix,
  incl. IPv6 `::1`); banner prints honest `ws://`/`wss://` + audit status.
  `--audit-path` defaults to `~/.xencode/audit.jsonl` (G1-04's sink is now
  actually wired), `none` disables; cert-without-key errors; certificates beat
  the escape hatch (no warning when TLS is on). 7 pure unit tests (no sockets);
  live smoke: loopback run answered `/` and `/auth/login`, `--host 0.0.0.0`
  refused with both ways out named. Hosts are IPs or `localhost` only — no
  DNS resolution behind the user's back. Workspace at 555.

### G3 — The hub becomes real

- [x] G3-01 tui: `collab_client.rs` worker (tokio-tungstenite) — login, connect,
  auth frame, read loop translated through pure `frame_to_tokens` into the
  `[COLLAB]` ingestion grammar; 30 s ping keepalive with a 60 s liveness
  deadline (timings constant-defined, not real-time tested), manual retry only
  (no auto-reconnect); simulated session task and the two dead state fields
  (`collab_commit_stream`, `collab_shared_files`) deleted, `member:`/`sync:`/
  `pending:` grammar replaced by `session:`, one `members:<json>` snapshot tag
  and `error:`. The worker owns login (and creates a session via
  `POST /sessions/create` when the id is empty), so there is no `collab_token`
  app field — tokens never leave the worker. 10 unit tests incl. a duplex
  `run_session` e2e (auth frame → auth_ok → 4409 close translated honestly);
  no server-rs dev-dep in the TUI — that would be a dependency cycle.
  Workspace at 565.
- [x] G3-02 tui: real hub UX — `c` create, `j` join-by-id, `Enter` connect,
  `r` retry, `Tab` field cycle, `Esc` disconnect+abort; fake port/latency/"TLS"
  telemetry replaced with server/transport/session/role from real state;
  help overlay matches handled keys; render sweep covers the active branch.
  The form is an in-panel editing mode (no new FocusArea): `Tab` is intercepted
  for the hub only and cycles server/user/session; while editing, keystrokes
  type into the selected field (`text_entry_active` guards the global chords;
  'q' types instead of quitting) and Esc unwinds edit → disconnect → close.
  Deleted telemetry: timestamp-derived port, "Protocol: WebSocket (TLS)",
  `<15ms` latency, hardcoded alice/bob/carol rows and the dead
  `collab_pending_changes` field. Rendered instead: real server URL, honest
  transport line (`ws (no TLS)` unless the URL is https), session id, live
  member rows with role badges from the `members:` snapshot, connected-for
  time, and the worker's last error verbatim. Ctrl+W closes the hub and
  aborts the worker too, so no socket is orphaned. The help table lists
  exactly the seven keys the hub binds (pinned by an exact-list test); a
  buffer-text render test covers idle-form/editing/ws/wss/error states plus
  a small-size sweep of the active branch. Workspace at 571.

### G4 — Close-out

- [x] G4-01 Docs sweep: CLI_GUIDE server section (flags, refusal rule, token
  flow, in-memory-sessions note), USER_MANUAL hub keys, QUICK_START/README/
  CHANGELOG synced, `docs/api_documentation.md` WS path corrected, live counts.
  The CLI_GUIDE server section had already shipped with G2-01 and was
  re-verified against `run_server`. This pass: USER_MANUAL gained a
  Collaboration Hub key table (c/j/Enter/r/Tab/Esc) and lost its fake claims
  (`0.0.0.0` default example banner, "credential vault/SQLite/refresh-token/
  email-verification" bullet), its command reference now lists `advise`
  verbatim from `--help`, and the Collaboration Server section documents the
  real posture (loopback default, TLS pairing rule, first-frame auth, audit
  path, in-memory sessions). `docs/api_documentation.md` had no username-in-URL
  WS path to fix — instead its entirely-fake JWT/`/api/v1` auth matrix was
  replaced with the real route table (public vs bearer vs WS close codes)
  behind a banner marking the Python module sections below as legacy.
  QUICK_START gained a Team collaboration quickstart; README's CRDT-sync
  claim (crdt.rs is unwired, deferred) replaced with token auth/RBAC/audit.
  `sync.rs::join_session` now records that credentials are enforced upstream
  in `xencode-server-rs`. Counts refreshed with this commit's gates.

Deferred on purpose (recorded, not skipped): passwords/IdPs (login = identity
claim; the bind surface is the perimeter), join approval (knowledge of the
session id is the invite), CRDT document sync (`crdt.rs` stays unwired),
session recovery after restart, TUI auto-reconnect, private-CA wss trust,
outgoing editor-activity producer, configurable max session size.

## Milestone H — TUI Polish + Selectable Layouts (complete 2026-09-20)

- [x] H1-01 — config schema: `layout`, `rounded_borders`, `show_scrollbars`,
  `show_line_numbers` (flat keys, serde defaults, CLI `config set` arms) and
  `XCODE_CONFIG_DIR` override for the config dir. Commit `70884aa`
- [x] H1-02 — `layout.rs`: pure `compute_layout` engine, presets
  classic/chat-first/zen, unknown→classic fallback, terminal-drop under
  18 rows, hit-test maps hidden-pane clicks to visible neighbours. `33129d7`
- [x] H1-03 — wired the engine: draw, mouse and resize-clamp read the same
  geometry; `App.last_body_focus`/`last_layout` latch at draw time. `268f221`
- [x] H1-04 — declarative `SETTINGS_ITEMS` table replaces magic row indices;
  new Display rows; `App::save_config` choke point; fixed the settings bug
  where rowing away kept the old edit buffer armed. `04be29a`
- [x] H1-05 — `Ctrl+U` cycles presets live with a toast; Tab ring skips
  hidden panes (zen deliberately keeps the full ring — Tab flips its panes).
  `8ff96bc`
- [x] H1-06 — `panel_block` + `panel_border_set`: every framed panel honors
  `rounded_borders`; markdown code-fence decoration stays square by design.
  `5c3739c`
- [x] H1-07 — config-driven scrollbars (chat & explorer, ≥24 cols) and the
  editor line-number gutter + current-line highlight (≥45 cols, gate in one
  place). `6ce7a53`
- [x] H1-08 — header revamp: ` ✦ xencode [layout] ⎇branch model` + focus
  badge, width ladder 72/60/40. `8d1dce0`
- [x] H1-09 — small-terminal pass: emoji-free titles <30 cols, popup
  80 %-floor when collapsed, toasts never cover the input. `ce14b5b`
- [x] H1-10 — layout×panel×size render sweep (incl. unknown preset and zen
  per body focus, all toggles on) + Ctrl+U help/keymap pin. `79f1b4a`
- [x] H1-11 — close-out docs (this entry): USER_MANUAL layout table + keys,
  CLI_GUIDE config keys + `XCODE_CONFIG_DIR`, QUICK_START layout line,
  README counts, CHANGELOG Added/Changed. Live figures at this commit:
  **591 tests passed, 13 crates, zero clippy warnings**

## Milestone I — Real Agentic Features (in progress)

- [x] I1-01 — permission policy core: `classify`/`ApprovalMode`/`ToolClass`
  in `agent_tools`, hard-deny outside the workspace / `.git/` / config dir,
  `agent_approval` config key + Settings Cycle row + CLI `config set`,
  session grants on `App`. **596 tests passed, zero clippy warnings**
- [x] I1-02 — file tool definitions + executors (read/list/search/write/edit,
  unified diffs via `similar`), all hard-denied outside the workspace at the
  executor itself; `background_start` resolves relative `cwd` against the
  workspace root. **603 tests passed, zero clippy warnings**
- [x] I1-03 — approval overlay: topmost modal with diff/command preview,
  `y`/`a`/`n`/`Esc` + `k`/`j` scroll, session grants, FIFO queue for stacked
  calls, and a `⚙ … · approved/denied` line in the chat log for every answer.
  **611 tests passed, zero clippy warnings**
- [x] I1-04 — file tools offered in the chat loop; every call routed through
  the policy (approve → execute, deny → `error:` the model must not retry),
  session grants shared with the loop, `agent_max_rounds` config (default 16).
  **617 tests passed, zero clippy warnings**
- [x] I2-01 — checkpoints + `/rewind`: pre-bytes snapshot of every approved write/edit, grouped per turn (session-only, git untouched), `/rewind [turns]`, editor buffer refreshed unless unsaved edits are at stake. **624 tests passed, zero clippy warnings**
- [x] I2-02 — `run_command` tool: foreground `sh -c` in the workspace root, `agent_command_timeout` (default 30 s, Settings row + CLI), 8 KiB tail of combined output, exit status first, approval prompt shows the literal command line. **630 tests passed, zero clippy warnings**
- [x] I2-03 — plan/TODO visibility: `update_plan(items)` posts the model's todo list (≤12 steps, read-only class so it never prompts), rendered as a `☰ Plan 2/5` strip above the transcript with `✓`/`▶`/`·` glyphs, compact at 6 steps with a hidden-count line; `/plan` pins, `/plan clear` drops; tolerant parsing of weak-model shapes, failed update leaves the visible plan alone. **641 tests passed, zero clippy warnings**
- [x] I2-04 — real ByteBot: `run_bytebot` drives the same agent loop as chat (background + file + command + plan tools) with the delegated task as its first message, so `bytebot_steps` are the **actual tool calls** with live status and the progress bar is the share of them that came back — it can regress, and the closing line says `N/M call(s) completed`. Honours `agent_approval` (every mutating call prompts in `ask` mode), shares the checkpoint groups so one `/rewind` undoes a whole run, and prints the provider's real error instead of invented output; the scripted sleeps and fabricated "All tests pass" rows are gone. **645 tests passed, zero clippy warnings**
- [x] I3-01 — MCP stdio client (`xencode-mcp-rs`): servers declared under
  `mcp_servers` in config (`command`/`args`/`env`, credentials only in env),
  exposed to the model as `mcp__<server>__<tool>` (sanitized, ≤64 chars) behind
  the same approval gate as every other tool (`External` class, always
  `y`/`n`, never waved through by autonomy), started **only** on user request
  via `/mcp`, with `/mcp status` and `/mcp stop`; a broken or missing server
  fails in its own words in a `[MCP]✗` line without stalling the TUI, and a
  stopped server's tools are withdrawn. `mcp_timeout` config (default 30 s,
  CLI `config set mcp_timeout`). **671 tests passed, zero clippy warnings**
- [x] I3-02 — pre/post tool-use hooks: `agent_hooks` config declares
  `before`/`after` shell commands (exact tool name, or `*` for every tool),
  run via `sh -c` in the workspace root on **approved** agent tool calls. A
  failing `before` hook vetoes the call before anything runs (file untouched,
  no rewind checkpoint, result is `error: pre-hook vetoed ...`); a passing one
  has its output prepended to the result. The `after` hook runs regardless of
  the call's outcome and its output is appended. Hook output follows the same
  cap as `run_command` (stderr merged, tail kept). **677 tests passed, zero clippy warnings**
- [x] I3-03 — `/spawn` subagent in a git worktree: `/spawn <task> [#branch]`
  creates a sibling `git worktree add` (`<dir>-spawn-<id>[-<branch>]`, branch
  `#branch` or generated `xencode/spawn-<id>`), then runs the same delegated
  agent loop as `/bytebot` isolated in that worktree (its own tool_root and a
  fresh checkpoint store, so `/rewind` never touches it). Live status streams
  are surfaced per call, and on completion the agent's final answer is posted
  back in the main chat as `(spawn #<id> · <task>)` with a
  `⏺ spawn #<id> done/failed — branch … at …, N/M call(s) completed` line.
  `/spawn status` lists every registered run with worktree locations. The
  user's task runs unimpeded in the main chat while the subagent works.
  **682 tests passed, zero clippy warnings**
- [x] I4-01 — provider fallback chain: primary model first, then the ordered
  `agent_fallback_models` alternates (CLI `config set agent_fallback_models
  a,b`, comma list). Each candidate gets exactly one attempt and the chain
  advances only when the error is fallback-eligible (`is_fallback_eligible`:
  transport/provider failure, never a parse error) and **nothing has streamed
  yet** — once a token arrives the turn stays on the provider that spoke it and
  the real error surfaces as before. A `[FALLBACK]` system note is drained into
  the transcript so a mid-chain bump is visible. `fallback_chain` dedupes the
  primary out of the alternates. Shipped with 690 tests passing
  (`providers-rs` +6, `config-rs` +1, CLI +1, TUI +2); the commit message
  claimed the checklist was ticked here but it was not — corrected by 793cb6f's
  close-out. **Correction (89e609a):** `is_fallback_eligible` had no caller in
  88ef42b — the loop advanced on *any* pre-stream failure, so a decode error
  burned the whole chain. The gate is now wired (`should_advance_fallback`) and
  covered.
- [x] I4-02 — close-out docs + honesty sweep. Every manual was re-read against
  the tree and the fiction removed: the README **ensemble reasoning** claim
  (replaced by the sequential chain), a credential vault and `xencode vault
  init|migrate|status` (no such subcommands — keys are plain JSON in
  `~/.xencode/config.json`), a v2.1.0 README badge (the crate is 0.1.0),
  "language-aware AST analysis" (per-language heuristics), in-chat
  `/help /models /model /project /status /clear /exit` (the eight real slash
  commands), a first-run setup wizard, analytics/monitoring/API claims that no
  route implements, k8s `postgres.yaml` + `DATABASE_URL` and
  `monitoring/prometheus.yml` as if they were wired (they are not), and a
  Documentation table now split into current vs archive.
  `.xencode.example.json` — pure Python-era shape (providers map, `ensemble`,
  `compression: lzma`) — is now the real flat `config.json`, validated through
  `XencodeConfig::load_from`. `CONTRIBUTING.md` moved off Python (`venv`,
  `requirements.txt`, `pytest`, `ruff`, `mypy`, `bandit`, the `dev` branch) to
  the cargo gates on `main`. Recorded as a known gap: providers-rs has an
  Anthropic client but `ApiKeys` has no `anthropic_api_key` and both entry
  points pass `None`, so `anthropic:…` models cannot authenticate. Named the
  seven scripted TUI panels, the manifest-only plugin surface, and the
  hardcoded entries in `GET /api/models`. Verifying the sweep's own claim that
  tests are hermetic turned up the rest of it: `save_config()` was reachable
  from tests and rewrote the developer's `~/.xencode/config.json`, now gated by
  `App::for_tests()` (f49e374).
  **692 tests passed, zero clippy warnings, `cargo fmt --check` clean**

## Milestone I — complete ✅

## Milestone J — every panel tells the truth — complete ✅

**Decided 2026-09-21.** The I4-02 sweep left seven TUI panels playing hardcoded
phrase lists — the same failure mode I2-04 fixed in ByteBot — plus a plugin
surface that reads manifests and loads nothing. The rule for this milestone is
the ByteBot rule: **a panel may only show data that came from the machine, the
provider, or the repo, and when it cannot get that data it says so in the
panel's own words.** No seeded arrays, no invented results.

Ordering is cheapest-real-first; each item is its own commit and must keep the
workspace gates green.

- [x] J-01 — **Security auditor** (2026-09-21): `Enter` walks the workspace with
  the existing `scanner::scan_tree` file list on a blocking thread and runs
  `VulnerabilityScanner::scan_file` on each readable file, streaming
  `[SECURITY]finding:` lines as they come and `[SECURITY]progress:` per file.
  Deliberately **not** `CodeAnalyzer::analyze_file` — the analyzer emits style
  and maintainability issues only, so adding it would have shown non-security
  findings under a security heading. Secret files from the walk are reported as
  Medium findings. Completed with real counts: the panel's totals come from the
  scan itself, unreadable files and a failed walk surface as
  `[SECURITY]note:` / `[SECURITY]failed:` log lines instead of silence, and the
  200-line display cap never affects the reported totals. Covered by
  `security_scan_streams_real_findings`, which asserts the scripted
  `config.py` / "tests passed" strings are gone. **685 tests passed.**
- [x] J-02 — **Performance profiler** (2026-09-21): the six fake functions are
  gone. `Enter` now reports measurements: this process's CPU (two
  `/proc/self/stat` reads 250 ms apart, since CPU is a rate) and resident memory
  (`/proc/self/statm` against `/proc/meminfo`), plus what the session already
  knows — average turn latency, llama.cpp tokens/s and token counts, per-provider
  health latency or its error, and the last 6 rows of `.xencode/cache/metrics.jsonl`
  (KV reuse, prompt tokens, tok/s, retrieved files). A gauge with no data behind
  it renders `n/a`, never a zero, and the panel notes when there has been no
  turn, no health check or no metrics file. Dropped the `fastrand` dependency —
  the scripted gauge numbers were its only user. Also fixed a layout bug the
  rewrite exposed: the Gauges box fitted 3 of its 5 lines, so Memory and Latency
  had never been visible. No new OS dependencies. Covered by three profiler
  tests plus a render test that asserts the real rows and the `n/a`.
  **689 tests passed.**
- [x] J-03 — **Terminal assistant** (2026-09-21): the scripted five-command
  list — including `docker system prune -af` and `rm -rf node_modules`, which
  this panel showed as "suggestions" with a risk badge — is gone. The panel now
  opens as a question field: type what you want to do, `Enter` makes **one**
  provider call per question, and the prompt gives the model something real to
  aim at (the workspace path, its top-level entries from the context index, the
  current git branch) and asks for a JSON array of `{command, risk, why}`,
  capped at 8. `parse_term_suggestions` tolerates fences, prose and a bare
  object; a reply with no commands in it is printed as the reply ("The model
  did not answer with commands. It said: …") instead of being turned into
  invented suggestions. Risk labels only ever escalate: a command matching
  `DESTRUCTIVE_PATTERNS` is shown as `destructive` whatever the model claimed,
  and `f` filters by that label. Running a selection goes through the agent's
  own gate — `execute_tool_call_approved` with the session's `ApprovalCtx`, so
  the same policy, modal, hooks and checkpoint group as a model-issued
  `run_command`, and `approval_ctx()` is now the single place that context is
  built. Nothing on screen is a claim about an outcome: a denial is history as
  `error: the user denied…`, and an unanswered prompt (no listener) denies too.
  Provider errors surface in the panel, flattened to one line. Also fixed: the
  `[TERM]output:` protocol arm had no sender left. Covered by five tests —
  parsing/risk escalation/cap, filter rows, empty question sends nothing,
  and two gate tests that assert a denied command never touches the disk.
  **695 tests passed.**
- [x] J-04 — **Multi-language** (2026-09-21): `Enter`/`d` runs the context
  engine's own `scan_tree` on the default workspace root in `spawn_blocking` and
  streams `[LANG]row` tokens, so the table is **languages present here** — files,
  lines (the scanner's `count_loc`, i.e. blanks and comment-leading lines
  excluded) and share of those lines, sorted by lines then name. Below it the
  walk's own notes: `N files · M lines · K skipped by ignore rules`, and an
  explicit line for secret/binary files, which the walker lists but never reads,
  so they contribute files and **zero** lines; unreadable files are called out
  too. The right-hand list is `scanner::Language::ALL` — a new const, added so a
  panel can name the languages the scanner actually supports instead of keeping
  a copy of the enum — with `▸` marking whatever the walk found; a test asserts
  `ALL` and `language_for_extension` agree in both directions and that every
  `as_str()` is a unique lowercase name. Translation is one real model call
  through the same single-shot provider path the terminal assistant uses: `Tab`
  picks From / To / Text, typing edits the selected field, `Enter` asks, and the
  reply is printed verbatim or the provider's error is printed as an error
  (`lang_translate_error` draws it in the failure colour). Empty text answers
  "Nothing to translate" without spending a request. Dead state removed with the
  scripted table: `lang_active`, `lang_supported`, and the hardcoded
  `lang_detection_results` / detection-legend tuples. Covered by three tests —
  a temp-dir walk that pins the numbers and proves `src/app.py` / `components.tsx`
  can no longer appear, an empty-text/no-request test with the field-editing
  cycle, and a render test at 160×44. **700 tests passed.**
- [x] J-05 — **Custom models** (2026-09-21): the four seeded profiles
  (`Code Assistant`, `Creative Writer`, `Bug Hunter`, `Code Reviewer`) and the
  dead `models_editing` / `models_saving` / `models_test_output` state are gone.
  The panel now lists `model_profiles` from `config.json` — a new
  `xencode_config_rs::ModelProfile { name, model, temperature, max_tokens }`
  with `#[serde(default)]`, so a config written before this still loads and an
  empty list renders "None yet — press `n`" instead of samples. `n` adds a
  profile from the current session settings, `-`/`+` moves temperature over
  0.0–2.0 in 0.1 steps, `←`/`→` steps `max_tokens` along a fixed ladder
  (64…8192) — an unset knob starts from the value the session would send
  anyway, so the first step is off a real baseline. `Enter` applies to the next
  turn (in memory only: `default_model` + the llama.cpp knobs, the model
  selector's position, and a `switch` to the llama.cpp server when the id points
  at it); `s` is the only key that writes `config.json`, through `XencodeConfig::save`
  rather than the silent `save_config()`, and reports `wrote N profile(s)` /
  `config.json unchanged: {error}` — or says persistence is off when the session
  has it off. `t` makes one request with exactly that profile's settings through
  the same `SingleShot` path the terminal assistant and translator use, so the
  status line carries the provider's own reply or its own error, flattened to
  one line. **`top_p` was dropped from the plan**: no provider path in this
  workspace sends it — `merge_llamacpp_options` is the only place sampling knobs
  reach a request, and it takes temperature and max tokens for llama.cpp. The
  panel says that out loud rather than pretending Ollama and cloud endpoints
  obey the sliders, and an unset knob renders as "unset — the server decides".
  Covered by three keymap tests (edit/apply ≠ save, empty list applies nothing,
  save round-trips through a temp `XCODE_CONFIG_DIR` — shared with the settings
  test behind a new `CONFIG_DIR` mutex so neither can race the other or touch
  the real config), two config round-trip tests (unset knobs are absent from the
  JSON; a pre-profiles config still loads), and two render tests at 160×44.
  **706 tests passed.**
- [x] J-06 — **Learning mode** (2026-09-21): the hardcoded "Rust Ownership
  Basics" lesson is gone — five sentences about a language feature, a
  `calculate_length` snippet that is not in this repo, an "exercise" nobody could
  submit, and a quiz whose correct option was whatever happened to be first
  (`learn_quiz_correct = selected == 0`, and the render even coloured option 0
  green). `Enter` now reads `.xencode/index/symbols.json` and queues the files
  that **declare something** — most declarations first, ties by path, capped at
  5 — so the queue is the index's own list; a file the extractor found nothing in
  is not a lesson, and no index at all means the panel prints "No project index —
  run /init first" and spends no request. What goes on screen per lesson is the
  file's own text (capped at a line boundary by `cap_at_line`, with a note saying
  how many of how many bytes were sent) plus the declarations the index recorded;
  `learn_show` does all of that with no provider involved, and `learn_ask_current`
  then sends **that same text** — what the panel shows is what the model was
  given. One call per lesson asks for `{explain, question, options, answer, why}`;
  `parse_lesson_quiz` tolerates fences, prose, a quoted answer index, and
  non-string options, and refuses a reply with no usable key (no explanation,
  fewer than 2 options, an index past the end) — that reply is printed as the
  reply rather than replaced by a canned question. Grading is against the model's
  key, the panel names which option was the key, shows its `why`, and keeps the
  model's sentences under "The model says:" so they cannot be read as facts the
  tool verified. `p`/`n` walk the queue, `r` re-asks, and every character is
  handled in the panel so `n`/`p`/`r` cannot fall through to a global chord
  (E2-06). Also fixed a layout bug the rewrite exposed: the header box fitted 1 of
  its 2 lines, so the lesson count had never been visible. Dead state removed:
  `learn_exercise`, `learn_progress_pct` (the +20%-per-correct-answer score).
  Covered by six app tests (queue order + no-index reason off a real `init_project`
  run, the show path incl. an unreadable file, grading against a non-first key,
  junk reply reported, parsing rules, no request when nothing is indexed) and two
  render tests at 160×44. **714 tests passed.**
- [x] J-07 — **Voice interface**: `arecord`/`pw-record` capture on Enter, level
  bar driven by RMS computed from the captured PCM (real meters), then
  transcription through a whisper CLI if one is on `PATH`; with no STT backend
  the panel reports the clip it recorded and that no transcription engine is
  installed — never a canned transcript.
  **Done** (`xencode-tui-rs/src/voice.rs`, new module): the recorder is the first
  of `arecord`, `pw-record`, `parec` found on `PATH` (a `which` that walks `$PATH`
  itself — there was no PATH lookup anywhere in the workspace to reuse), spawned
  with a fixed argv list that streams raw S16_LE mono 16 kHz on stdout. No shell
  string is built, so this stays in the same category as the `git` and
  `llama-server` spawns elsewhere and nowhere near the agent's `run_command`
  approval path. Every 100 ms chunk (3200 bytes) yields one meter reading — RMS
  over the samples, ×8 as a stated display gain because a voice RMS sits near
  0.02 and an unamplified bar reads dead — so the bar, the peak and the clip
  length are all arithmetic over bytes the recorder sent. Enter again, or Esc
  while a capture runs, ends it early and keeps what it has; `m`/Space is a real
  mute, draining the pipe and discarding audio rather than producing a silent
  clip. The PCM goes into `<root>/.xencode/voice/clip-<unix>.wav` through a
  hand-built 44-byte RIFF header (Python's `wave` opens it: 1 channel, 16-bit,
  16000 Hz). Text appears in the transcript only from a whisper CLI's stdout;
  with none installed the panel states that and names the clip, and a transcriber
  that fails, prints nothing, or dies contributes its own words or nothing at
  all. Removed with this: the four canned phrase/result pairs (including
  `run tests → ✅ 142 tests passed, 0 failed`), the fixed 0.3–0.9 level cycle,
  and the dead `voice_commands`, `voice_confidence`, `voice_language` state, plus
  the "speaking" status this product has no text-to-speech to earn. Verified live
  against the real microphone path, not just fakes: `arecord` was found, a
  capture stopped at ~2 s reported RMS readings that started saturated and decayed
  to ~0.12, wrote a valid 1.85 s WAV, and printed the no-engine note; the
  substituted-recorder tests use `cat` on a prepared PCM file, so no test needs a
  microphone or mutates `PATH`. Covered by nine module tests (RMS incl. the
  trailing odd byte, meter scaling, WAV header field by field, ms from byte
  count, chunk-per-reading with levels and totals, muted capture keeps nothing,
  an unstartable recorder names itself, empty output, the missing-engine note)
  and ten app/render tests (level token parses or changes nothing, peak holds,
  clip report vs. no engine leaves the transcript empty, malformed clip said out
  loud, failed capture stops the meter, mute reaches the flag the reader thread
  reads, empty capture, and three panel renders at 160×44).
  **733 tests passed.**
- [x] J-08 — **Plugin runtime** (2026-09-21): manifests now load.
  `PluginRuntime::load(dir, version)` discovers `plugin.json` /
  `manifest.json`, skips a manifest whose `xencode_version` does not accept this
  build (reported, not silently dropped), registers each remaining one as a
  `ManifestPlugin` with the `Host`, and flattens the session into the two things
  a plugin may actually change: a trimmed `prompt_prefix` and `before`/`after`
  hooks. There is **no** dynamic linking and no plugin code — `entry_point` went
  with the Python runtime it named, and `ManifestPlugin::handle_event` answers
  nothing. Precedence is one rule with one path: a plugin's hook only lands
  where `agent_hooks` in config.json is silent (`session_hooks()`), and plugin
  text goes into the `system:` argument of `assemble_chat` for chat turns and
  delegated runs alike, loaded once per session so the KV-stable head stays
  byte-stable. `App::new()` loads from `default_plugin_dir()`
  (`$XCODE_PLUGIN_DIR`, else `<data dir>/xencode/plugins` — the same directory
  the CLI installs into); `App::for_tests()` loads from an empty one, so a
  plugin on a developer's machine cannot move a test assertion.
  Verified live, not only in tests: `xencode plugin install` printed
  `guardrails v1.2.0 — loaded: prompt prefix, 1 before hook(s), 1 after hook(s)`
  beside a pinned manifest reported as `NOT LOADED: needs xencode 0.1.0
  (declared 9.9.9)`, `plugin remove ../../etc` was rejected as an invalid name,
  and the running TUI's `/plugin` reported `1 loaded, 2 reported` with the same
  two verdicts. Covered by 8 runtime tests, 5 TUI tests (including
  `a_plugin_hook_runs_around_a_gated_tool_call`, which asserts the plugin's
  command ran around a real gated `write_file` — policy `edit-allow`, the gate
  itself untouched) and the manifest/registry suite. **751 tests passed.**

Not in scope: `AnthropicProvider` stays as-is by explicit decision (documented
gap, no key field); `crdt.rs` stays unwired (settled Milestone G decision); the
provider-health and diff/worktree/task/insight panels are already real.

## Docs archive purge — complete ✅

**2026-09-21, after J-08.** The manuals were honest about the code but the repo
still carried 9,371 lines of documentation describing a product this tree is not,
linked from the entry-point docs. Deleted rather than archived:
`DOCUMENTATION.md` (dual-stack Python + Rust, credential vault, ensemble
reasoning, 12 crates/65 tests), `PRD.md`, `project details.md`,
`docs/FEATURES.md`, `docs/ARCHITECTURE_DIAGRAMS.md` (an "API Gateway", a
"Connection Pool Module" and a "Distributed Cache" — the Python package shape),
`BROWSER_LOGIN_PLAN.md`, `docs/superpowers/` (executed migration plans, their
specs and `- [ ]` agent-worker scratch, linked from nothing) and
`images/4-6.jpg`.

The files that stayed were re-checked against the tree, because the rule is
"no fiction in manuals" and those docs had drifted:
- `docs/ROADMAP.md` rewritten from the code. Its Phase 2 section listed
  `xencode --git-commit`, `--git-review`, `--git-diff-analyze`,
  `--git-branch suggest` and `/analyze` `/smart` `/context` chat commands that
  the Rust CLI never defined (`Commands` in `xencode-cli/src/main.rs` has
  `review`, not `--git-*`) — replaced with the real entry points:
  `xencode review [--base main] [--format text|json]`, `Ctrl+R` per-file review,
  `Ctrl+Y` PR dashboard, and `Ctrl+S` for the git commit panel, whose text
  ("Staging N files…") and `[GIT_COMMIT_OK]` / `[GIT_COMMIT_ERR]` results come
  from `ui.rs` / `keymap.rs`. Phase 3 and the Review Dashboard are now marked
  shipped (Milestones E/F, per-file dashboard) instead of "not yet built", the
  `ModelManager` class that does not exist is gone, the unreachable Anthropic
  route left the architecture line, and the measured block is today's numbers.
- `docs/api_documentation.md` trimmed to the server: the `xencode.core.*` module
  reference, the benchmarking suite and the Python examples describe names no
  file in `rust/` defines.
- `docs/INSTALL_MANUAL.md` and README's diagram pointer retargeted off the
  deleted files; README's "Historical archive" table is now a paragraph saying
  those files were deleted, and its feature bullet stopped advertising Kubernetes
  assets (`k8s/` is gone; `Dockerfile` + `docker-compose.yml` remain).

---

## Milestone K — a GPU you do not own: remote providers + Google Colab — active 🚧

> Research verified by live experiment on 2026-09-23 against this machine and a
> real free-tier Colab account, not from blog posts. Phase 0 (the Settings →
> Providers editing surface) landed the same day; the routing half had landed
> the day before.

### Why

The laptop is an i5-1035G1 (4c/8t) with 15 GB RAM and an Iris Plus G1 / MX250.
Even a 7B model is a struggle on it, so *local* inference is a dead end here.
The free GPUs Google Colab hands out (a T4 has 16 GB) are the cheapest way to
make the agent usable, and the same plumbing unlocks any remote
OpenAI-compatible server.

### What was proven, and what was disproven

- **Proven: `colab ssh` is a working SSH bridge.** `google-colab-cli` 0.7.x
  exposes `colab ssh --proxy-mode`, an OpenSSH `ProxyCommand` WebSocket bridge.
  Measured here: runtime allocated by `colab new`, banner `SSH-2.0-OpenSSH_9.6p1`,
  login accepted **as `root`** with an ed25519 key (`colab`, `sree`, `user` are
  all refused), and a server listening on `127.0.0.1:8000` in the VM answered
  through `ssh -N -L 18000:127.0.0.1:8000` from the laptop with **HTTP 200 in
  0.59 s**. No tunnel provider, no public URL, nothing for a stranger to hit.
- **Disproven: Colab does not publish VM ports.** Every guess at
  `https://<port>-<runtime>.prod.colab.dev` returned a bare `404` — with the
  session JWT as `X-Colab-Runtime-Proxy-Token`, as the `colab-runtime-proxy-token`
  query param, both, or neither — *even while a listener was up on that port*.
  Only port 8080 routes, and it routes to Colab's own agent (that is what the
  SSH bridge rides). Guides that sell "Colab + cloudflared" are solving a problem
  this path does not have.

### The connectivity decision that decides everything (checked every corner)

- **Public tunnels (ngrok / cloudflare / pinggy) burned inside the VM get you
  suspended on the free tier.** Colab's own FAQ lists "remote control such as
  SSH shells, remote desktops" and "bypassing the notebook UI to interact
  primarily via a web UI" as disallowed on free managed runtimes, and real
  users report Google locking the *whole account* for the ngrok-backdoor
  pattern. The same FAQ explicitly states a paid subscription or a positive
  compute-unit balance *removes* those restrictions.
- **The front door that exists:** nothing is tunneled; the move is
  `colab ssh --proxy-mode` riding Google's authenticated proxy, exactly what
  the official Colab-in-VS-Code extension does (its auth is a localhost
  loopback OAuth exchange, not a poke through the VM firewall).
- **Version trap:** `google-colab-cli` **0.6.0 on PyPI is missing the `ssh`
  subcommand** (upstream issue #102; present only since 0.7.x). Preflight must
  assert >= 0.7.0. The CLI is Linux/macOS-only, so Windows users fall back to
  typing a public-tunnel URL (paid tier) into Settings → Remote URL.
- **Sister project, not a dependency:** Google also ships an official **Colab
  MCP server** (`googlecolab/colab-mcp`, Mar 2026) for driving notebooks as a
  code-execution *workspace* for agents. Different feature (inference backend
  vs execution sandbox); note for a future `xencode-mcp-rs` integration, out of
  scope here.

### What is in the tree today

- **K-1a (landed, `eca4204`):** `remote:` routes any OpenAI-compatible endpoint
  through `OpenAICompatibleProvider` (`providers-rs src/compatible.rs:26`,
  dispatch at `lib.rs:472`) in all three `ProviderManager` paths; config fields
  are `remote_base_url` (`config.rs:90`) + `api_keys.remote_api_key` (doc comment
  there names the Colab SSH-forward case explicitly); `xencode tests/remote_endpoint.rs`
  fakes a Colab endpoint with a `Bearer colab-token` via wiremock.
- **K-1b (landed, `68f2a4a`):** Settings → Providers edits real values — Remote
  URL as a text row, Remote/Gemini/Qwen/OpenRouter keys as masked `Secret` rows
  (bullets while editing, last-four tail on display; empty commit clears, Esc
  discards the buffer). Endpoint rows refresh the picker, all provider rows
  re-run the health check on save. Persistence stays behind `App::save_config()`
  so `for_tests()` writes a temp dir, never the real config.
- **K-2a (landed, `5739ebb` + `173a36b`):** `xencode colab preflight` gates the
  bridge in one pass: the `colab` CLI on PATH, version >= 0.7.0 (the 0.6.0
  version trap is caught twice — by the version string and by a functional
  `colab ssh --help` probe), backend auth (`colab sessions`), ssh/ssh-keygen on
  PATH, and an ed25519 key pair under the config dir (`--generate-key`). The
  report prints a fix line per failing check and exits non-zero. Lives in the
  new `xencode-colab-rs` crate (8 hermetic tests with fake colab CLIs).
- **Still missing:** the rest of the Colab *lifecycle* — provision, bootstrap
  the inference server, hold the forward, survive reconnects, report health,
  tear down — plus the Settings section that drives it and the model-picker
  surfacing.

### Tasks

- [x] **K-1a — custom endpoint in config + routing** (`eca4204`).
- [x] **K-1b — Settings can edit providers, masked keys included**
      (`68f2a4a`). The three kinds are now editable in one panel: Local
      (Ollama / llama.cpp), Cloud (Gemini / Qwen / OpenRouter), Remote / Colab.
- [x] **K-2a — `xencode colab preflight`.** Report, in one pass: CLI present and
      >= 0.7.0, auth works (`colab sessions`), an ed25519 key exists under
      `~/.xencode/` (generate it if asked), and what to run when a check fails.
      The `xencode-colab-rs` crate was born here so its preflight gate has a
      home (`173a36b`; CLI wiring `5739ebb`).
- [x] **K-2b — orchestration + state in `xencode-colab-rs` (crate exists from
      K-2a).** Wraps the optional `colab`/`ssh` tools (same "report itself
      unpowered" pattern as J-07's recorders; `which()`/`run()` and the preflight
      checks are already there). New `ColabConfig` in `xencode-config-rs`:
      `enabled, session, local_port, remote_port, runtime ("llama.cpp"|"ollama"),
      model, weights_source ("hf"|"drive"|"gcs"), auto_connect` — all
      `#[serde(default)]`. State in `~/.xencode/colab.json`: session, ssh /
      forward / keep-alive pids, ports, runtime, model, started_at, url.
- [x] **K-2c — `xencode colab up|status|down`.** `up`: `colab new --gpu T4`,
      SSH bootstrap (pinned `llama-server` + GGUF on `127.0.0.1:8080`, **or**
      `ollama serve` on 11434 — the ollama choice makes tags flow into the
      model picker for free via the existing `refresh_models()`), hold `-N -L`,
      keep-alive per the official CLI, write `colab.json`, point
      `remote_base_url` (or `llama_cpp_url` / `ollama_url`) at the forward.
      `status`: parse `colab sessions` + GET `/v1/models` on the forward.
      `down`: kill forward, `colab stop`, clear state.
- [ ] **K-3 — survivability.** A provider-health row for the forward
      (`draw_provider_health` / `run_health_check` currently have no Remote/Colab
      entry), dead-VM / 12-hour-reap detection, one-key reconnect.
- [ ] **K-4 — tests + docs.** Config round-trip for the Colab block, wiremock
      fake OpenAI Colab endpoint (extend `remote_endpoint.rs`), masked-key
      rendering (baseline exists from K-1b; the fake `colab` CLI on `$PATH`
      landed with K-2a); then README / QUICK_START / CLI_GUIDE / USER_MANUAL /
      CHANGELOG and the `.xencode.example.json` in the same pass.

### Standing constraints for this milestone

`colab` and `ssh` stay *optional external tools* (like `arecord` and the whisper
CLI in J-07) — the product remains Rust and the feature reports itself unpowered
when they are missing rather than shelling out blindly. Session names are
validated, never interpolated into a shell. Nothing about the approval gate
changes: a remote model is still a model, and agent tools still go through
`execute_tool_call_approved`. The default connectivity path is the official
`colab ssh` bridge (no public URL, ToS-safe on every tier); public tunnels stay
documented as an advanced paid-tier-only way to fill in Settings → Remote URL
manually — the account-risk choice is the user's, never baked in.
