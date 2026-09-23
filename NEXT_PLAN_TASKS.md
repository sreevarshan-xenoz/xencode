# Next Plan — Task Checklist

> Working task list for the active backlog. See [NEXT_PLAN.md](NEXT_PLAN.md) for
> the milestone overview and [docs/ROADMAP.md](docs/ROADMAP.md) for the long-term roadmap.
> All items target the Rust workspace (`rust/crates/*`) per `AGENTS.md`. Verified 2026-09-23.

## Rust Migration — Complete ✅

- [x] Port core, config, cache, memory crates — `xencode-core-rs`, `-config-rs`, `-cache-rs`, `-memory-rs`
- [x] TUI foundation (ratatui, `rust/crates/xencode-tui-rs/`)
- [x] Model providers with retries (`rust/crates/xencode-providers-rs/`)
- [x] Repo-wide context / RAG — `xencode-context-rs` (index, embed, retrieve, budget)
- [x] Server / collaboration / plugin crates — `xencode-server-rs`, `-collaboration-rs`, `-plugin-rs`
- [x] Analysis + security scanning — `xencode-analysis-rs`
- [x] Tool-calling + model capabilities — `generate_stream_with_tools`, `ModelCapabilities`
- [x] CLI subcommands — scan, config, models, cache, query, memory, tasks, worktree, colab, advise, server, analyze, fetch, review, plugin, llamacpp, tui
- [x] Workspace gates green — 15 crates, 815 tests passing, 4 ignored, zero warnings

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

## Milestone K — a GPU you do not own: remote providers + Google Colab — complete ✅

> Every item below was finished and the whole path was proven against a real
> free-tier Colab VM on 2026-09-23: a T4 was rented, llama.cpp served a Q4_K_M
> GGUF on it, `xencode query -m 'remote:…'` answered through the SSH forward,
> Provider Health went green, `--reconnect` recovered a killed tunnel in 9 s,
> and `down` left no session and no orphan. No step of that was mocked.

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
  new `xencode-colab-rs` crate (49 hermetic tests, every one of them driving a
  fake `colab`/`ssh` script on `$PATH` — no network, no real VM).
- **Still open, deliberately:** `colab_auto_connect` is stored and round-tripped
  but nothing reads it — bring-up stays an explicit `xencode colab up`. The
  `drive`/`gcs` weights sources are validated and refused with a fix message
  rather than implemented; `hf` covers the case the milestone was for.

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
      SSH bootstrap (pinned prebuilt `llama-server` + GGUF on
      `127.0.0.1:18080` — Colab's own proxy holds 8080 — **or**
      `ollama serve` on 11434 — the ollama choice makes tags flow into the
      model picker for free via the existing `refresh_models()`), hold
      `ssh -N -l root -L`, keep-alive per the official CLI, write `colab.json`,
      point `remote_base_url` (or `llama_cpp_url` / `ollama_url`) at the
      forward. `status`: parse `colab sessions` + GET `/v1/models` on the
      forward. `down`: kill forward, `colab stop`, clear state.
- [x] **K-2d — the same path, run for real (`c014c30`, `692c494`, `2a27501`).**
      Brought up an actual free-tier T4 and served a real GGUF through the
      forward. Six things only a live run shows: the ssh login must be **root**
      (Colab injects the key for root only), port **18080** (8080 is occupied
      by Colab's node proxy), `READY` must mean *serving* — the model needs
      ~40 s to load, so a spawn-time `READY` raced the probe — a dead
      bridge slot (`Already-active SSH session` / `banner exchange`) has to be
      waited out instead of failing the bring-up, `--reconnect` should try the
      forward before re-downloading anything, and the detached forward must not
      inherit stderr or it keeps a caller's pipeline open forever. Also
      `--quant` / `config colab_quant`, and the two `xencode-server-rs` models
      tests made hermetic so a live bridge can't change what they assert.
- [x] **K-3 — survivability (reap + reconnect).** 12-hour-reap detection in
      `colab status` (started-at age vs endpoint, reaped hint) and one-key
      reconnect (`xencode colab up --reconnect`): fast path reuses a live
      endpoint with zero colab/ssh calls, otherwise re-creates a reaped
      session / re-runs the bootstrap / re-spawns the forward and re-probes.
- [x] **K-3 — survivability (health row).** A provider-health row for the
      forward (`draw_provider_health` / `run_health_check` now have a Remote
      entry: seeded like the keyed providers, probes `{remote_base_url}/models`,
      Connection Details lists the Remote URI, detail line under the row shows
      the forward URL).
- [x] **K-4 — tests + docs.** Config round-trip for the Colab block, wiremock
      fake OpenAI Colab endpoint (extend `remote_endpoint.rs`), masked-key
      rendering (baseline exists from K-1b; the fake `colab` CLI on `$PATH`
      landed with K-2a); then README / QUICK_START / CLI_GUIDE / USER_MANUAL /
      CHANGELOG and the `.xencode.example.json` in the same pass.
      Done: the shipped example now has a test that loads it through the real
      loader and pins its Colab defaults, `mask_secret`'s boundary is covered,
      and the manuals describe what the live run actually did — root login,
      18080, READY-means-serving, bridge-slot retries, forward-first
      reconnect, `--quant`, and the `colab_enabled` gate that `up` refuses
      without.

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

## Milestone L — any machine you can SSH into, and an agent that finishes its own work (planned 2026-09-23)

> Status: **planned, nothing built.** Two independent tracks. The agent track
> (L-7..L-9) ships first because it improves every provider at once, including
> the free Colab path that already works.

### Why

Milestone K turned out not to be a Colab feature. What actually got built was
five generic ones wearing a Colab costume — provision a box, install and start
an OpenAI-compatible server on loopback, hold a private tunnel to it, persist
state across the provider killing the box, reconnect with one key. Of
`xencode-colab-rs`'s 2,968 lines, roughly 600 are Colab-specific (`colab new`,
`colab sessions`, the one-bridge 429 slot, the 12-hour reap). The rest is
bootstrap/poll/retry/reconnect logic any second backend would reuse verbatim.

Separately, the agent still stops short of finishing: after an edit it reports
success without running the project's tests, an `edit_file` whose `old` string
isn't unique hard-fails with no recovery path, and `RequestMetrics` already
records tokens per request that nobody ever reads.

### Research: which remote backends were considered and rejected

Web research on 2026-09-23 across twenty-plus GPU backends, two independent
passes. Both are cited below; where they disagree, that is said rather than
averaged.

- **Architecturally impossible for this pattern** — Kaggle, Hugging Face
  Spaces, Replicate, Modal, Salad. No SSH or no private port path, so the model
  server is reachable only over a public URL — the exact ngrok-shaped free-tier
  ToS violation Milestone K ruled out. Modal's "tunnels" are unauthenticated
  public TLS URLs; Salad is container-groups only with no user VM.
- **No GPU, or the wrong shape** — Oracle Always Free is Arm **CPU** only (and
  the A1 allotment was cut to 2 OCPU / 12 GB); GCP/AWS/Azure have no free GPU
  and need card + quota ceremony; Hetzner sells GPUs only as flat monthly
  dedicated servers, which is not a burst `up` command.
- **Viable, and still not built** — Vast.ai, Lightning AI, RunPod, Fly.io
  Machines, DigitalOcean/Vultr/Scaleway. Each is prepaid-credit or
  spot-preemptible: a machine that can be reclaimed mid-response and a zero
  balance that deletes the volume. **The two research passes directly contradict
  each other on RunPod** — one found root SSH plus A5000 at ~$0.16/hr and ranked
  it the best paid backend, the other read RunPod's SSH docs and found that port
  forwarding specifically requires a public IPv4 on the pod, with proxy "Basic
  SSH" not supporting forwards. That cannot be settled without a paid account,
  and "expose a public IP on a box running our model server" is a security
  regression, so the feature is not built on a contested fact. Stopped-but-not-
  deleted pods and detached network volumes also bill silently — an
  abandonment-billing hazard this project has to support for real users.
- **What replaces all of them: `bring-your-own-SSH`.** L-2 gives "I have a 24 GB
  workstation / a Mac Studio / a server under my desk" with zero new vendor
  surface, zero billing hazard, zero ToS risk — and it is the thing that
  actually forces the L-1 trait extraction to be real rather than theoretical.

Priced facts above are from vendor pages as of 2026-09-23 and will drift. The
Metal path in L-3 is **UNVERIFIED**: it cannot be tested on this machine, which
has no Apple hardware, so it ships as a detected-and-reported branch, not as a
promised runtime.

### Do not build (decided, with reasons)

- **Managed GPU-cloud backends** — see the RunPod contradiction and the
  billing-abandonment risk above.
- **An embedding / vector index, or Cursor-style remote repo indexing.**
  `xencode-context-rs` has deterministic retrieval *and* a retrieval eval that
  measures it. Swapping in embeddings would regress quality silently.
- **A llama-swap clone** (multi-model VRAM juggling) — solves a server
  administration problem, not an agent problem.
- **Speculative decoding** — a ~25-30% tok/s win for a large, fragile launch
  surface on hardware that is already the bottleneck.
- **Terminal computer-use, cloud PR-review bots, A2A.**
- **A new plugin/extension format of our own** — see the plugin track, which
  lands separately once its research is in.

### Standing constraints for this milestone

Unchanged from Milestone K: Rust-only under `rust/crates/*`; external tools
(`ssh`, `colab`, a future vendor CLI) stay optional and the feature reports
itself unpowered when they are missing; nothing is interpolated into a shell
unvalidated; the approval gate still guards every agent tool call and a remote
model is still just a model; **no public tunnels**; live verification against a
real machine, not mocks, and `down`/teardown so nothing is left billing. Keys
stay plain strings in `config.json` — there is no encrypted vault, so L-4's
loopback-only and `--api-key` work is what keeps a *remote* server from being
reachable by anyone else on that machine.

### Tasks

Numbered by dependency, not by ship order. **Ship order: Track A (L-7 → L-9)
first**, then Track R (L-1 → L-6); L-10 → L-12 are polish after either.

#### Track R — any machine you can SSH into

- [ ] **L-1 — extract a `Backend` trait from `xencode-colab-rs`.** Split the
      generic half of `orchestrate.rs` + `lifecycle.rs` (bootstrap,
      poll-until-serving, forward spawn/hold, state file, reconnect) behind a
      trait with three seams: `provision()` (create/attach the compute),
      `transport()` (return the argv that carries `ssh -N -L`-style forwarding),
      `reap_hint()` (the provider-specific "why is it gone" line, e.g. Colab's
      12 h). Colab becomes impl #1 in the same crate; `state.rs` gains a
      `backend` field so `colab.json` migrates without losing a live bridge. No
      user-visible behavior change in this step.
      **Done-when:** `xencode colab up|status|down|reconnect` behaves
      byte-identically to today (re-run the live T4 bring-up, not just the 49
      hermetic tests), and the Colab-specific file is under ~800 lines.
- [ ] **L-2 — `xencode remote add|list|use|up|status|down`, the BYO-SSH
      backend.** `add` records a host (`user@host[:port]`, optional
      `~/.ssh/config` alias) plus a runtime choice into a per-host profile; `up`
      runs the L-3 probe, pushes the same bootstrap L-1 keeps generic, starts
      the server on loopback, holds the forward, writes state, points
      `remote_base_url` at it. Reuses `point_config_at_forward` unchanged so the
      `remote:<model>` route works with no provider changes.
      **Done-when:** proven end-to-end against a **second real machine** — bring
      the same GGUF up somewhere that is not Colab, answer a real
      `xencode query -m 'remote:…'`, `down` cleanly, and `status` after a killed
      tunnel says so instead of lying.
- [ ] **L-3 — remote capability probe.** Run detection **on the box** —
      `nvidia-smi`, `lspci` for AMD, `system_profiler`/`sysctl` for Apple, plus
      RAM and free disk — and pick the runtime from the result: CUDA tarball →
      `llama-server`; AMD → `ollama` (it handles ROCm where the CUDA build
      cannot); Apple → Metal; otherwise CPU, with the model size capped
      accordingly. Report the chosen branch and the numbers it was chosen from.
      **Done-when:** forcing each branch (real CUDA box, real CPU-only box;
      Metal **UNVERIFIED** — no Apple hardware here) yields a server that
      actually answers `/v1/models`, and a wrong-branch guess is impossible
      because the decision is printed.
- [ ] **L-4 — SSH hardening: TOFU pinning + connection reuse.** First contact
      pins the host key into xencode's own `known_hosts` under the config dir
      (`accept-new` semantics, refuse on change); `ControlMaster`/`ControlPersist`
      plus `ServerAliveInterval` make the bootstrap's many short `ssh` calls
      share one connection. Never set `StrictHostKeyChecking=no`, and if the
      remote's uid isn't ours, pass llama.cpp `--api-key` and re-check that the
      server bound to `127.0.0.1` and not `0.0.0.0`.
      **Done-when:** a host-key change is refused with an actionable message,
      the bootstrap makes one TCP connection instead of ~10, and a second user
      on the remote box cannot read the endpoint.
- [ ] **L-5 — `xencode hw probe`: local hardware → launch flags.** Read
      VRAM/RAM/cores from `lspci`, `/proc/meminfo`, `/sys/class/drm` and emit a
      recommended quant, context size and `-ngl` layer count — including the two
      silent killers the numbers hide: KV cache on top of weights, and a
      concurrent `cargo` build evicting the mmap. Feed the same generator into
      L-2/L-3 so model choice stops being a guess.
      **Done-when:** its recommendation for this laptop (i5-1035G1, 15 GB,
      MX250) is a size that actually loads and answers, and every field it reads
      is documented.
- [ ] **L-6 — budget preflight and OOM recovery.** Before launch, refuse a
      model whose measured footprint exceeds available memory and say what would
      fit. On an exit-137 / killed-server signature, step quant or context down
      and retry once, then escalate with a concrete "or `xencode remote add …` /
      `xencode colab up`" suggestion instead of hanging.
      **Done-when:** deliberately asking for a model two sizes too big produces
      the refusal before download, and a real OOM produces the stepped-down
      retry rather than a silent hang.

#### Track A — the agent finishes its own work

- [ ] **L-7 — test/lint auto-repair loop with an exit-code "done" gate.** After
      the agent's edits, run the project's own test and lint commands (discovered
      from `Cargo.toml`/the workspace, over the existing approval gate), feed
      failures back to the model, and iterate a bounded number of times. The
      gate is the process exit code, never the model's claim that it is done.
      **Done-when:** a seeded compile error in a scratch crate is repaired by
      the loop and terminates on `cargo test` genuinely exiting 0; the iteration
      cap is configurable and its exhaustion is reported as an incomplete task,
      not as success; the approval gate is unchanged.
- [ ] **L-8 — `edit_file` failure fallback.** Today a non-unique or
      non-matching `old` string hard-fails. Return structured "why it didn't
      match" — the count of matches plus each candidate with line numbers and
      surrounding context — so the next turn self-corrects, and retry once
      automatically when the failure was ambiguity rather than absence.
      **Done-when:** an ambiguous edit on a real duplicated line converges in
      one retry, and the existing exact-match contract stays exact — no silent
      fuzzy write.
- [ ] **L-9 — cost metering over the metrics that already exist.** Aggregate
      `RequestMetrics` (prompt/cached/completion tokens, tok/s, context usage,
      compaction) into per-model and per-session spend with a configurable
      budget: a `/cost` command, a TUI status row, and a warning as the budget is
      crossed.
      **Done-when:** numbers come from written `metrics.jsonl` records and match
      a hand-summed session, an unknown price is shown as unknown rather than
      invented, and a price table update is data, not code.

#### Track E — extend both (after either track lands)

- [ ] **L-10 — resumable, disk-aware GGUF download.** Check free disk before
      starting, resume a partial file instead of restarting, and surface progress
      in the TUI — it is the longest and least observable step of a bring-up.
      **Done-when:** killing a download mid-file and re-running `up` resumes it,
      and a too-small target refuses before writing anything.
- [ ] **L-11 — free hosted inference routes (`groq:…`, `nvidia:…`).** Route
      through the existing OpenAI-compatible path with a per-prefix base URL so a
      first-run user gets real capacity without renting anything. Groq's free
      tier is roughly 30 RPM / 1,000 req/day behind an **8K TPM** ceiling that is
      smaller than one agent prompt; NVIDIA NIM is roughly 40 RPM behind a 429
      backoff. **Rate numbers are web-verified as of 2026-09-23 and will drift**
      — keep them in a dated table, not in prose.
      **Done-when:** a real prompt is answered through each route from a clean
      config, the 8K-TPM trap is handled and documented rather than surfacing as
      a confusing 429, and no key is ever written by a test.
- [ ] **L-12 — LSP diagnostics loop.** After edits, pull real compiler
      diagnostics from an LSP server rather than only `cargo check`, so
      non-cargo languages get the same L-7 treatment.
      **Done-when:** it demonstrably catches something `cargo check` does not,
      in a language the agent currently edits blind — otherwise it does not ship.

## Milestone M — stop being an island: hooks, skills, plugins, MCP, ACP (planned 2026-09-23)

> Status: **planned, nothing built.** Ask from the same research pass as L: "what
> other features like the Colab bridge could we add — and what about plugins, MCP
> and third-party support?"

### Why

The Colab bridge was valuable because it connected xencode to compute it did not
own. The same instinct applied to software says the bigger isolation is protocol
-level: xencode cannot be extended by anyone else's tooling, and cannot be *used*
by anyone else's agent. Before planning that, the extension surface was read
rather than assumed. What is actually in the tree:

| Seam | Real state (verified 2026-09-23) |
|---|---|
| **Plugins** | Manifest only. `PluginManifest` (`xencode-plugin-rs/src/manifest.rs:32-53`) carries name/version/`xencode_version` pin, `prompt_prefix`, `before`/`after` hooks. `PluginRuntime::load` merges those two outputs into the agent loop (`app.rs:2362-2383`). **`handle_event` returns `Ok(None)` unconditionally** (`runtime.rs:113-118`) — no plugin has ever received an event. The `XencodePlugin`/host traits exist but nothing dynamically loads through them. |
| **Plugin `permissions`** | Parsed (`manifest.rs:47`) and **never checked anywhere**. A manifest can claim capabilities the host does not enforce. |
| **MCP** | Client only, **stdio only, tools only** — resources/prompts/sampling deliberately not negotiated (`xencode-mcp-rs/src/client.rs:1,9`, `lib.rs:2`). Hand-rolled JSON-RPC; no `rmcp` or any MCP crate in any `Cargo.toml`. Tools surface as `mcp__<server>__<tool>` capped at 64 chars (`xencode-tui-rs/src/mcp.rs:51-63`). Servers start only on `/mcp`; config `mcp_servers` is command/args/env with no HTTP, no headers, no auth (`config.rs:190,239`). **No MCP server mode.** |
| **Hooks** | Real and load-bearing: `agent_hooks.before/after` maps a tool name or `*` to `sh -c` (`config.rs:252-260`), runs post-approval in the workspace root (`agent_tools.rs:874-954`), and a failing `before` hook vetoes (`:1206`). **But the command is static** — `run_hook` spawns `sh -c` with piped stdout/stderr and never a stdin write, so a hook cannot learn which tool ran or what it was about to change. |
| **Approval gate** | `classify()` (`agent_tools.rs:209-248`) with ask/edit-allow/all-allow modes, hard-deny outside the workspace / `.git` / config dir, MCP tools always `External → Ask`, session-scoped grants. No persistent allowlist. |
| **Subagents** | `/spawn` runs a subagent in a git worktree (`app.rs:167-177`) — execution exists, declarative agent definitions do not. |
| **AGENTS.md** | Read into the context head alongside `anchor.md` (`xencode-context-rs/src/context.rs:96-103`). |
| **Collaboration** | Server routes, bearer auth, RBAC, audit and WS are wired, and the TUI hub connects for real. `crdt.rs` is referenced by nothing but its own `lib.rs` re-export — still the settled Milestone G deferral. |
| **Absent, verified** | No LSP anywhere. No sandbox (no seccomp/landlock — worktree isolation is all there is). No OTel. No custom user slash commands. No skills format. No downloadable themes or statusline (8 hardcoded themes). |

### The strategy that falls out of that table

Every candidate here has a compatibility version that is cheaper than an
invention, and in 2026 the ecosystem has already picked winners on three of them:
**Claude-Code-shaped hook events with a JSON payload on stdin** (Codex copied
it, so the shape is the de-facto standard), **`SKILL.md` for skills** (Cursor and
others converged), and **git-repo `marketplace.json` for distribution** — which
means a registry is a git repo, not a binary service worth building. So the plan
is: make xencode readable by other tools' conventions first, then expose xencode
itself over MCP and ACP. Do not invent a fourth format.

Two surfaces are genuinely new capability rather than compatibility, and they are
the two biggest items: **MCP server mode** (another agent calls xencode's
read/search/edit/run tools) and **ACP** (xencode's agent runs inside Zed, Neovim,
or Emacs instead of only a terminal). ACP is real and growing in 2026 — Zed and
JetBrains both ship it and a Rust SDK exists at 0.2.x — but the spec is pre-1.0,
so it is last, not first.

### The trap that decides the order of the big two

Headless MCP and ACP callers **cannot answer an approval prompt**. xencode's
approval gate is the thing that makes it safe to run `run_command` at all, so a
naive `mcp serve` either hangs every write into a timeout or quietly auto-
approves — the second option is not a feature gap, it is a vulnerability. M-5 and
M-7 therefore inherit a prerequisite they cannot skip: an explicit, non-
interactive permission policy (per-tool, session-scoped, defaulting to read-only)
that is auditable and that never widens the interactive TUI path.

### Do not build (decided, with reasons)

- **A plugin marketplace or registry service.** Distribution that already works
  is a git repo plus a manifest; building a server for it is maintaining infra,
  not product.
- **A new plugin config or skill format.** `plugin.json` and `SKILL.md` are the
  conventions; a third dialect buys incompatibility and nothing else.
- **Dynamic native or WASM plugin code loading.** Turns every third-party plugin
  into arbitrary code execution with no sandbox behind it (there is none), for
  demand nobody has asked for. The manifest model stays.
- **Completing the CRDT wiring.** Still no product pull — settled since G.
- **OTel/telemetry now.** L-9's cost metering over the existing `metrics.jsonl`
  answers the question people actually ask.
- **A full command sandbox** (Landlock/seccomp/bubblewrap). Reconsidered only if
  M-5 ships and exposes write tools to external callers; that is the one change
  that would make it load-bearing rather than nice.

### Tasks

Small-to-large, and deliberately: M-1..M-4 are compatibility work that makes
xencode usable by tooling people already have. M-5..M-7 are the new surfaces.

- [ ] **M-1 — give hooks their payload.** Write `{tool, args, phase, session_id,
      workspace}` JSON to the hook process's stdin in `run_hook`, keeping the
      existing non-zero-exit veto and the current output annotation. Adopt the
      event names other agents already use so a hook written for one runs here.
      **Done-when:** a hook script that reads stdin can name the tool and veto a
      specific `write_file` by its path — and no secret ever travels in argv,
      where `/proc` would leak it.
- [ ] **M-2 — enforce what a manifest declares.** Either check `permissions` at
      load and refuse or degrade with a clear message, or delete the field. A
      parsed-but-ignored security-relevant field is worse than an absent one.
      **Done-when:** `/plugin` reports the enforced decision, and a test proves a
      disallowed capability cannot reach the agent loop.
- [ ] **M-3 — skills: `SKILL.md` loader.** Discover `~/.xencode/skills/*/SKILL.md`
      and `.xencode/skills/*/SKILL.md`, parse frontmatter, inject only the
      name/description menu into the prompt head, and load a full body on demand
      through the existing plugin prompt plumbing. `/skills` lists what loaded.
      **Done-when:** installing a skill measurably changes behavior on a real
      prompt, and a directory of 30 skills costs the prompt a menu, not 30 bodies.
- [ ] **M-4 — `xencode plugin install <git-url>`.** Clone, pin the commit, verify
      the manifest, show a diff of what it declares before it can contribute a
      prompt prefix, then copy into the config dir. `remove` and `update`
      complete the cycle.
      **Done-when:** install → load → the prefix is visible in `/plugin`, an
      unpinned install says which commit it pinned, and a manifest that gains a
      new `prompt_prefix` on update is shown as a diff rather than applied
      silently.
- [ ] **M-5 — `xencode mcp serve`: xencode as an MCP server.** Expose `read_file`,
      `list_dir`, `search_files`, `write_file`, `edit_file`, `run_command` over
      stdio using the official `rmcp` SDK (replacing the hand-rolled client only
      if it earns it — the client stays as-is unless a shared dependency makes
      that free). **Requires the non-interactive permission policy above.**
      **Done-when:** an external MCP client lists and calls xencode's tools
      read-only against a real workspace, every write path is refused by default
      with an actionable reason, and the 64-char tool-name limit is handled the
      same way the client already handles it.
- [ ] **M-6 — finish the MCP client: resources, prompts, and HTTP with headers.**
      Negotiate the capabilities the client currently declines, and add an
      HTTP/SSE transport with auth headers so a hosted server is reachable.
      **Done-when:** one real third-party server is connected over each
      transport, a resource and a prompt from it surface in the TUI, and a token
      in config is masked in every render the same way `mask_secret` masks keys.
- [ ] **M-7 — `xencode acp`: run the agent inside an editor.** Put the existing
      turn loop behind the ACP Rust SDK over stdio, mapping xencode's approval
      requests onto ACP permission requests and the plan/tool stream onto ACP
      session updates.
      **Done-when:** a real chat plus an approved edit round-trips inside Zed (or
      a second ACP client), with a **live** ACP version pinned in `Cargo.toml` —
      the spec is pre-1.0 and this item is explicitly allowed to be blocked by
      upstream churn rather than half-shipped.

## Milestone N — the full option space (research appendix, drafted 2026-09-23)

> **This is a map, not a queue.** Six research passes (Sept 2026) covered the
> areas L and M did not. Everything they surfaced is recorded here — including
> the items later passes will cut — so triage is a deliberate, revisitable step
> rather than something that happens implicitly while planning. Ranking lives at
> the bottom; the record comes first.
>
> Every "what exists today" claim below was read out of the tree, not assumed,
> with file:line. Web claims carry their source; anything the passes could not
> confirm is marked **UNVERIFIED**.

### N-0 — Eleven facts about the current tree that changed how the options read

1. **There is no AST parsing anywhere.** Symbol extraction is four regexes over
   Rust text (`xencode-context-rs/src/symbols.rs:55-68`), whose own header comment
   says "Tree-sitter may replace the extraction layer later" (`symbols.rs:16`).
   `scan_tree` is a file walker with extension-based language tagging
   (`scanner.rs:187,342`). The analyzer is per-line regex
   (`xencode-analysis-rs/src/analyzer.rs:32-100`). `tree-sitter`, `syn` and `quote`
   appear in zero `Cargo.toml`s. So every structural claim the refactor-insights
   panel makes is textual underneath, and `edit_file`/`search_files` are
   exact-string and per-line regex (`agent_tools.rs:22,445-516`).
2. **Nothing evaluates the agent.** `eval.rs` measures *file-retrieval* quality
   only — recall@k, precision@k, MRR over a gold set at `.xencode/eval/gold.json`,
   A/B'd from `/ctx eval` (`eval.rs:6-9,66-82`; `app.rs:3381-3397`). There is no
   end-to-end agent benchmark, zero snapshot/insta crates in the workspace, no
   golden full-run regression test, and no model A/B on real tasks. The system
   prompt is one frozen string, `AGENT_SYSTEM_PROMPT` (`context.rs:33`), with no
   versioning.
3. **A cloned repo's `AGENTS.md` is delivered in the trusted position.** It is
   concatenated into the stable prefix between the system identity line and
   `STABLE_END_MARKER` (`context.rs:96-119`), and that line reads "Follow the
   project guidelines below exactly" (`context.rs:33-35`). `grep` for
   `untrust|sanitiz|injection|defang` across `context-rs`, `agent_tools.rs`,
   `xencode-mcp-rs` and `xencode-plugin-rs` returns **zero hits**: no
   prompt-injection guard exists.
4. **`~/.xencode/config.json` is written with default permissions.**
   `XencodeConfig::save()` uses `std::fs::write` with no `set_permissions` in the
   call path (`config.rs:444-454`), so plaintext `api_keys` land world-readable
   under the usual umask. `set_permissions` is used elsewhere (`mcp.rs:452`,
   `xencode-colab-rs/src/lib.rs:98`), so this is an omission, not a constraint.
5. **The approval gate inspects path *arguments*, never command *contents* — and
   that is the correct design, with a consequence.** `classify` hard-denies only
   `path`/`cwd` args outside the workspace or in `.git`/config dir
   (`agent_tools.rs:217-227`, `path_allowed` `:186-205`, lexical, symlinks
   unresolved). `ToolClass::Shell` is `Ask` in every mode except `all-allow`
   (`:238`), so a human sees the literal command string. `ReadOnly → Allow` is
   unconditional and there is no taint tracking across turns. The exposure is not
   "no gate"; it is that the gate's only defense for a shell call is a human
   reading a string — and fact 3 is the vector aimed at that reading.
6. **Images are never decoded.** `images.rs` detects format by magic bytes
   (`:91`), reads header dimensions (`:134`), caps at 20 MiB (`:23`) and encodes a
   data URL (`:302`) — no decode, no resize, no recompress. A 4K screenshot goes
   to the provider as-is. This is a token-budget defect, not a missing feature.
7. **Scanned PDFs silently yield empty text**, and the code knows it:
   `documents.rs:16-18` lists OCR as a follow-up.
8. **The TUI cannot show an image.** ratatui 0.29 + crossterm 0.28 only; no
   graphics protocol (kitty/sixel/iTerm), no OSC-52, no clipboard integration, no
   screenshot capture. The markdown renderer is hand-rolled with **no syntax
   highlighting** (`markdown.rs:1-5`). 8 hardcoded themes (`theme.rs:7-16`), 3
   layout presets (`layout.rs:14`), mouse scroll/click hit-testing exists
   (`app.rs:6138-6280`).
9. **Voice is record-only.** `voice.rs` is real — `arecord` subprocess (`:115`),
   16 kHz S16_LE WAV, 15 s cap (`:26`), transcription by shelling to a whisper CLI
   on PATH (`:145-157`). **No TTS anywhere** (zero hits for
   tts/speak/piper/kokoro/sherpa).
10. **Memory has no cross-session retrieval.** `xencode-memory-rs` stores
    session-keyed raw message lists to `~/.xencode/conversation_memory.json`,
    50-message cap with the oldest drained (`lib.rs:9,176-179`); `xencode memory`
    lists/inspects sessions. No distilled facts, no relevance retrieval.
11. **Compaction is already better than most products, and its byproducts are
    useful.** Soft at ≥70% usage (drop oldest 30%, `[d]`-decision entries always
    survive, no model call), hard at ≥90% (one model call folding into a layered
    markdown summary with the last 6 turns verbatim) — `compact.rs:17-18,54-116`,
    parsed back into `ContextState` (`:129`). Before a rewrite it snapshots the
    canonical transcript to `.xencode/cache/transcript/<ts>.json`
    (`compact.rs:8-11`) — a free corpus of **recorded real traffic**, not mocks.
    Context is 7 budget-capped tiers (`context.rs:20-25`).

**The constraint that quietly limits several features here:** `context.rs:29-80`
builds a byte-identical prefix (SYSTEM + `AGENTS.md` + `anchor.md`, closed by the
marker, SHA-256 drift-checked) *specifically so llama.cpp can reuse the KV
prefix*. Any feature that varies the per-turn file set — nested instruction files,
prompt hot-swapping, cross-session memory injected into the head — voids that
reuse unless it is placed below the marker or the cache boundary is made explicit.
`RequestMetrics.cached_tokens` (`metrics.rs:24-42`) already measures the damage.

**And one from N-0 worth keeping straight:** the `VulnerabilityScanner`'s
five regex families are mostly Python-flavored (`security.rs:11-40,49-220` —
`os.system`/`eval`, md5/sha1/DES/RC4, SQL concat, path traversal, SSRF), it shells
out to nothing (`cargo audit`/`deny`/`semgrep`/`gitleaks`: zero hits), and it never
opens secret-named files — by design, twice: the walker never reads them
(`scanner.rs:47`) and the scan loop skips `entry.is_secret` (`app.rs:887-889`). So a
key in a file that does *not* look like a secret is content-scanned while `.env`
never is.

### N-1 — Code intelligence and structural editing

- **CI-1 `ast_edit` agent tool via ast-grep as a subprocess** — structural
  search/replace with metavar patterns, `--json` results, `fix` mode.
  Effort S (~1 day). Trap: needs the ast-grep binary (same "reports itself
  unpowered" pattern as `colab`/`ssh`), and a wrong pattern matches zero sites
  and looks like success. Done-when: `fn $A($B) -> $C` rewrites 3 seeded call
  sites in one call and the diff equals a hand-check.
- **CI-2 tree-sitter symbol extraction replacing the `symbols.rs` regexes**
  (Rust + TS + Python grammars). M (~1 wk). Trap: grammar/runtime ABI version
  pinning and a C toolchain requirement at build time. Done-when: a strict
  superset of the regex symbols on this repo, zero false positives inside macros
  or comments.
- **CI-3 `edit_symbol(path, symbol, new_body)`** — range-scoped replacement
  validated by reparse. M, after CI-2. Trap: tree-sitter error recovery hides
  broken output; must reject a file whose edited region parses with ERROR nodes.
  Done-when: a corrupt-edit test proves the rejection.
- **CI-4 codemod mode** — the agent emits one ast-grep YAML rule, xencode applies
  it repo-wide behind a preview diff + approval. S-M after CI-1. Trap: repo-wide
  apply on a dirty tree. Done-when: a 20-site rename in one tool call.
- **CI-5 Rust toolchain kit as gated tools** — `cargo fix --allow-dirty
  --clippy`, `clippy --message-format=json` summarized, `cargo fmt`,
  `cargo-shear`. S each. Trap: `cargo fix` overwrites edits made since the last
  build — sequence after build-green, before commit. Done-when: the L-7 loop
  drives a clippy count to 0 with JSON evidence before/after.
- **CI-6 `what_breaks` impact analysis** — reverse-dependency list for an edit
  target from the existing dep graph. M. Trap: regex-grade accuracy on call
  sites, so label the confidence explicitly. Done-when: editing `symbols.rs`
  surfaces its known consumers.
- **CI-7 DAP debug loop through lldb-dap over MCP, not a custom client** —
  breakpoint / continue / inspect-locals as tools. M-L (~1-2 wk). Trap: a stateful
  subprocess against stateless tool calls, timeouts, and debug binaries required.
  Done-when: the agent stops at a breakpoint in a failing test and prints a
  local's value. (There is no usable Rust DAP *client* crate; Microsoft's `dap`
  builds adapters. LLDB shipping an MCP mode is **UNVERIFIED** — the docs page
  could not be deep-fetched.)

### N-2 — Developer workflow and product surface

- **WF-1 NDJSON event stream mode** (`query --stream --format ndjson`): token and
  tool events, versioned. S-M. This is the primitive other tooling builds on —
  `claude -p --output-format stream-json` and `codex exec --json` prove the shape,
  and today `--format json` emits one blob. Trap: schema churn; version it
  explicitly. Done-when: a script pipes it and CLI_GUIDE documents every event.
- **WF-2 GitHub PR surface over REST** (`xencode pr create|review <n>`, comment
  threads into the existing ReviewDashboard, `GH_TOKEN` or device flow; no `gh`
  dependency). M. Trap: auth sprawl, rate limits, vendor lock-in. Done-when:
  a PR opens from a TUI branch and a review thread reads back, without `gh`.
- **WF-3 session resume/naming + redacted transcript export** (`--resume <name>`,
  `/share`). S-M. Trap: transcripts contain secrets — reuse `mask_secret`.
  Done-when: resume restores context across processes and export passes a
  redaction test.
- **WF-4 build/test autodiscovery** — probe README/CI files/`justfile`/`mise`,
  write the recipe into `anchor.md`, and *prove* it by running it. M. Trap: false
  confidence; require exit 0. Done-when: a fresh clone yields a working
  test command from one command.
- **WF-5 `cargo-dist` release pipeline + binstall + AUR.** S. Trap: signing keys
  in CI. Done-when: a tag produces `cargo binstall xencode`.
- **WF-6 shell completions + man page generated from clap.** S. Trap: drift —
  generate them in CI, never by hand. Done-when: tab completion works in fish/zsh.
- **WF-7 CI watchdog** (`xencode ci watch` over Actions REST feeding failed logs
  into the agent). M. Trap: polling and pagination cost. Done-when: after a push
  the terminal reports pass/fail plus one failed-test summary unaided.
- **WF-8 `xencode-action`** — the review command as a GitHub Action commenting on
  PRs. M. Trap: token scoping is a supply-chain risk; depends on WF-2.
- **WF-9 signed-commit passthrough.** S. Trap: GPG agent env inside a TUI.
  Done-when: agent-made commits verify on GitHub.
- **WF-10 stacked-diff assist over worktrees.** L. Trap: rebase correctness, and
  GitHub's stacked-PRs preview may commoditize it — revisit after that matures.

### N-3 — Agent quality, evaluation, context engineering

- **EV-1 local task-eval harness** — 10-30 seeded scratch git repos, a `task.md`
  each, graded by `cargo test`/exit code, run headless through the real agent
  loop. M (~1-2 wk). Trap: grade the diff, not the chat; and the agent can read
  its own grader. Done-when: a pass rate is reported and the suite runs in CI
  without network. **This is the item that makes L-7 measurable instead of
  plausible.**
- **EV-2 turn trace + TUI inspector** — per-turn JSONL beside `metrics.jsonl`
  (prompt hash, tools, rounds, tokens, cost) and a `/trace` pane. S (2-4 d).
  Trap: tool output contains secrets. Done-when: the last 50 turns are browsable
  with per-task token totals.
- **EV-3 prompt registry with versioning** — named prompt files, version hashed
  into metrics and eval rows. S (2-3 d). Trap: see the KV-prefix constraint in
  N-0. Done-when: `/ctx` shows the active version and eval output groups by it.
- **EV-4 cross-session memory with relevance retrieval** — distilled facts scored
  through the existing `retrieve()` signals. M (~1 wk). Trap: stale facts poison
  context; needs expiry and review. Done-when: an eval-style test shows the right
  fact injected within budget.
- **EV-5 sub-directory instruction files** — walk from the edited file's dir to
  root, budget-capped. S (2-3 d). Trap: the KV-prefix contract. Done-when: an
  edit in a nested dir provably loads its directives.
- **EV-6 notes-to-self scratchpad** — a `write_note` tool in a compaction-exempt
  tier, like `[d]`. S (2-4 d). Trap: unbounded growth; cap and fold into hard
  compaction. Done-when: notes survive a hard compaction and appear in assembly.
- **EV-7 failure reflection → human-promoted lesson** — after N failed rounds or a
  `/rewind`, the agent *drafts* a lesson and the user approves it into
  `AGENTS.md`. S (3-4 d). Trap: auto-commit is drift, not learning.
  Done-when: the draft cannot land without explicit approval.
- **EV-8 playback regression below the HTTP boundary** — recorded *real* provider
  responses replayed through real tool execution in seeded temp repos. M (~1 wk).
  Trap: house-rule optics — the fixtures must be documented as captures of real
  traffic (the `compact.rs` transcript snapshots are already such a corpus), not
  as mocks. Done-when: a previously-flaky bug is pinned by a committed fixture.
- **EV-9 API prompt-cache accounting** — `cache_control` breakpoints at the stable
  prefix; `cached_tokens` is already metered. S-M (3-5 d). Trap: markers in the
  wrong place void reuse — measure with the existing kv-reuse ratio.
  Done-when: a multi-turn session shows a measured cost drop.
- **EV-10 judge-assisted eval reports**, LLM ranking only near-miss outcomes that
  an exit code already graded. S (3 d). Trap: position/verbosity/self-preference
  bias are documented; the judge may never flip an exit-code failure.
- **EV-11 tamper-evident audit log** — hash-chain `audit.jsonl` (prev-hash + seq;
  the `seq` exists, the chain does not). S. Trap: do not build a Merkle tree.
  Done-when: a verify command detects a mid-file edit.

### N-4 — Security and the agent's own trust model

Ranked by what the tree actually shows, not by how alarming it sounds.

- **SE-1 `chmod 0600` on config save** (fact 4). Tiny. Done-when: a test asserts
  the mode, and an already-existing world-readable file is tightened on save.
- **SE-2 untrusted-content marking** — every repo file, command output, MCP result
  and `git log` string carries a source attribute; the system prompt states that
  only user turns carry instructions. S (~200 lines in `context-rs`). Trap:
  markers are hygiene, not a wall — models do ignore them. Done-when: markers
  survive compaction and are covered by tests.
- **SE-3 the `AGENTS.md` trust split** (fact 3) — a repo-provided `AGENTS.md` is
  *data* until the user trusts that content hash once, with a persistent trust
  store. M. Trap: it breaks the exact workflow `AGENTS.md` exists for, so the
  prompt must be unmissable and the decision durable. **Done-when: an untrusted
  `AGENTS.md` cannot raise permissions, proven by test.**
- **SE-4 lethal-trifecta gate in `classify`** — if a turn has read
  `~/.ssh`/`~/.xencode`/env/secret-named content, a later network-writing command
  escalates or denies. M. Trap: cross-turn taint tracking is leaky; keep it a
  coarse session bit. Done-when: a planted exfil scenario is blocked by test.
- **SE-5 secret *content* scanning** — scan into `run_security_scan` and every
  `write_file`/`edit_file` approval, redacting transcript copies. M. Trap: false
  positives on fixtures need an allowlist file; a pure-Rust library option is
  early-stage (**UNVERIFIED** quality). Done-when: a planted key is caught and
  `examples/` is ignored.
- **SE-6 `xencode deps` supply-chain report** — shell to `cargo deny` (+
  `cargo-shear`), parse JSON, stream findings like the security scan; lockfile
  diffing on a PR. S-M. Trap: report only — auto-fixing dependencies is how the
  supply chain becomes the attack. Done-when: it runs on this workspace.
- **SE-7 Landlock/bubblewrap wrapper for `run_command`** — workspace + `~/.cargo`
  writable, network off by default, per-command `--net` approval. L. Trap: silent
  fallback when the kernel lacks Landlock; the test matrix is painful. The
  `landlock` crate is alive (0.4.x). Done-when: `cat ~/.ssh/id_rsa` fails *inside*
  an approved command, on this machine.

Context for the ordering: a Jan-2026 SoK catalogued 42 injection techniques and
found adaptive attacks still exceeding 85% success against *filter-based*
defenses, with the consensus that architectural mitigations beat detection
([arXiv 2601.17548](https://arxiv.org/abs/2601.17548),
[OWASP cheat sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html),
[CaMeL](https://simonw.substack.com/p/camel-offers-a-promising-new-direction)).
That is why SE-2/SE-3/SE-4 are ranked above SE-7 despite being less glamorous.

### N-5 — Multimodal and the terminal surface

- **MM-1 image resize/recompress before send** (fact 6): decode, cap ~1568 px,
  JPEG q80. S-M. Done-when: attached bytes are under ~1 MiB and a token-count
  before/after is in the commit message. **A fix, not a feature.**
- **MM-2 screenshot→attach hotkey** via `grim`/`spectacle` into the existing
  attach path. M (~200 lines). Trap: Wayland-only tooling — degrade with a clear
  message the way `voice.rs` does. Done-when: one keypress lands a screenshot in
  the attached block and the provider accepts it.
- **MM-3 local VLM vision over the existing bridge** — a Qwen2.5-VL-class `mmproj`
  model on the Colab T4 or a `remote:` host, plus a `describe_image` tool. M.
  Trap: there is no "is this model vision-capable" gate today, so failures are
  opaque; add one. Done-when: `xencode query` on a real screenshot answers
  through `remote:`.
- **MM-4 inline image preview** via `ratatui-image` (official org; kitty/sixel/
  iTerm with a half-block fallback). M. Trap: must not break the narrow-terminal
  render tests. Done-when: renders on a supporting terminal, degrades to a
  metadata line otherwise.
- **MM-5 Wayland image paste** (`wl-paste -t image/png`) with text keeping
  priority. S. Done-when: paste does the obvious thing for both.
- **MM-6 clipboard copy of code blocks/diffs** — `wl-copy` with an OSC-52
  fallback that is probed, never assumed. S.
- **MM-7 syntax highlighting in the markdown renderer** (`two-face` +
  `tree-sitter-highlight`, or syntect). M. Trap: `render_markdown` returns
  `Vec<Line>` and must stay streaming-safe.
- **MM-8 `xencode report`** — one self-contained HTML file from a session or scan,
  with an embedded SVG chart (`plotters`). M. Trap: stop at HTML; PDF/DOCX is the
  scope creep that kills it.
- **MM-9 Mermaid diagram generation** — model emits Mermaid, validate and render
  to SVG for the report (the `merman` crate renders Mermaid headlessly in Rust;
  a CLI alternative exists but its URL was not verified this pass); no terminal
  render. M. Trap: the grammar is large, so invalid output falls back to a code
  block.
- **MM-10 OCR for scanned PDFs** — closes `documents.rs:16-18`'s own TODO, via a
  VLM (MM-3) rather than a Tesseract dependency. S-M after MM-3.
- **MM-11 browser verification as a documented Playwright-MCP recipe** — the
  existing stdio MCP client makes this nearly free and adds no Python. S-M,
  mostly docs. Trap: say plainly that Playwright MCP is a Node subprocess.
  Done-when: the agent loads a dev server, screenshots, attaches.

### N-6 — Local-first, tailnet, cross-device, and non-code work

`tailscale` is installed and up on this machine, and Tailscale's own guidance to
bind services to localhost (so identity headers can't be forged) is already
xencode's loopback-by-default posture. Since the 1.52 redesign, `tailscale serve`
is not HTTP-only — `--tcp`, `--tls-terminated-tcp`, `--proxy-protocol` and a
Layer-3 mode are documented, so raw TCP is a supported use case; plain `ssh -L` to
a `100.x` address also works today. What Serve adds for free is auto-TLS on
`host.tailnet.ts.net`, MagicDNS, ACLs and identity headers that make an API key
redundant.

- **LF-1 `xencode tail serve up|status|down`** — publish a model endpoint (or a
  build log / HTML report) to the tailnet, reusing the colab `preflight`/`state`
  shape with `tailscale` optional and reporting itself unpowered. M (~500 lines).
  Trap: Serve config **persists across reboots** (only `serve disable` clears
  it), a team/family tailnet is not "only me", and tagged devices get no identity
  headers. Done-when: a phone browser reaches the laptop's `/v1/models` over
  `ts.net`, and `status` prints the *actual* ACL granting access instead of
  asserting privacy.
- **LF-2 approval round-trip to a phone** — self-hosted ntfy + nonce, **timeout =
  deny**. M. Trap: treating "the user is away" as license to auto-approve is the
  inverse of the point. Done-when: an unreachable prompt denies after N minutes
  and the transcript names who was asked. (Claude Code's Remote Control expires
  dialogs at 5 min to the no-action default — the right prior art.)
- **LF-3 outbound-only controller window over the tailnet** — a phone as a *window*
  into a local session, queueing prompts across drops, gate intact. L. Trap: two
  keyboards, and any third-party relay breaks local-first.
- **LF-4 `xencode run --detach`** — durable queue, resume-after-crash, and stop
  conditions on wall-clock and cost as well as rounds. L. Trap: keep status
  *derived* (exit file + `/proc`, as `tasks.json` already does) rather than
  stored, or it lies after a crash; lid-close suspend is the silent killer here.
  Done-when: SIGKILL mid-job then `--resume` continues from the last completed
  round, and each cap verifiably stops a runaway.
- **LF-5 `sql` tool + DB panel with read-only enforced in the driver** —
  `rusqlite` bundled / `sqlx` for Postgres, `SQLITE_OPEN_READ_ONLY`, a role with
  `default_transaction_read_only=on`, statement timeout, row/byte caps, no
  multi-statement. S-M. Trap: regex on the SQL string is theater. Done-when: an
  adversarial prompt cannot mutate a checksummed database.
- **LF-6 session bundle on git as the multi-machine bus.** S. Trap: concurrent
  writers — sessions are JSONL, so sync plus a lock is enough; do not design a
  protocol.
- **LF-7 weight provenance** — SHA256 + HF revision pin, verify-on-load, and a
  visible verified/unsigned badge in the model panel. S-M. Trap: hashing after
  download proves nothing about the *source*; say what it does and does not
  establish.
- **LF-8 offline conformance suite** — a CI run with networking off proving every
  panel degrades honestly instead of rendering zeros for measurements that never
  happened. S. This is the testable version of "local-first" as a product claim.
- **LF-9 Google Workspace MCP as opt-in, read-only** (Gmail/Calendar MCP servers
  exist but are Developer Preview, need a GCP project, and relay through a cloud).
  S. Trap: it contradicts the local-first story, so it ships labeled as the
  exception it is. Done-when: it lists mail and events and refuses to send.

### Cross-cutting do-not-build register (consolidated from L, M and N)

Managed GPU clouds (L); embeddings/vector index and Cursor-style remote indexing
(L, N-3); llama-swap clone (L); speculative decoding (L); terminal computer-use
clicking loops and hosted computer-use with live display (N-5); cloud PR bots as a
substitute for the local review loop (L); A2A (L); a plugin marketplace service or
a new plugin/skill format (M); dynamic native or WASM plugin loading (M); full
OTLP stack when JSONL is inspectable (M, N-3); completing the CRDT wiring (M, N-6);
regex prompt-injection *detectors* — sub-50% mitigation per the SoK (N-4); a
container or microVM sandbox, which kills the local-first premise (N-4);
`cargo audit` where deny/OSV supersede it (N-4); auto-updating dependencies (N-4);
an agent that sends email (N-6); Telegram/Discord as a primary control path — an
inbound-capable agent is RCE for anyone who can message it (N-6); Tailscale Funnel
for any model or session endpoint — public by definition, three ports, no identity
headers (N-6); Syncthing integration — Android-side maintainer churn and invisible
conflict resolution (N-6); a bespoke always-on daemon when systemd user units give
cgroups, journald and a watchdog free (N-6); a native VS Code/Zed extension since
M-7 ACP covers both (N-2); Docker-based CI replay (N-2); a `gh` subprocess wrapper
with its untestable auth surface (N-2); real PDF/DOCX/PPTX generation (N-5);
TTS voice-out (N-5); user theme files (N-5); an embedded rust-analyzer/LSP host for
call graphs, where index startup dwarfs the agent loop (N-1); a hand-rolled DAP
client (N-1); libcst/jscodeshift integration, which violates the Rust-only
directive (N-1); cargo-mutants in the default loop (N-1); local cross-encoder
reranking (N-3); multi-agent A/B swarm harnesses (N-3); golden-transcript replay
that *replaces* real connections (N-3); an autonomous self-modifying system prompt
(N-3); SWE-bench/Terminal-Bench runner integration (N-3); model attestation and
SLSA-for-weights, still an IETF draft-00 with nothing to verify against (N-6).

### Triage status

**Not yet ranked.** The ranking pass is deliberately a separate, later decision —
this appendix records the option space so the cut is visible and revisitable
rather than implicit. Inputs it will have to weigh: the three defect-shaped items
(SE-1, MM-1, and the `AGENTS.md` trust position SE-3) versus capability-shaped
items; EV-1 as the measurement that makes L-7's claim checkable; and the
KV-prefix constraint in N-0 as a tax on several otherwise cheap features.
Milestones L and M already carry a ranking; this appendix does not.

### Where to re-check this appendix (primary sources, consulted 2026-09-23)

Repo facts are verifiable by the file:line above. The external claims came from:

- **Code intelligence** — [ast-grep](https://github.com/ast-grep/ast-grep) and
  [its docs](https://ast-grep.github.io/) ·
  [tree-sitter Rust migration post](https://ast-grep.github.io/blog/tree-sitter-rust-migration) ·
  [grammar ABI versioning discussion](https://github.com/tree-sitter/tree-sitter/discussions/1768) ·
  [`cargo fix`](https://doc.rust-lang.org/cargo/commands/cargo-fix.html) ·
  [`cargo-shear`](https://github.com/Boshen/cargo-shear) ·
  [LLDB MCP docs](https://lldb.llvm.org/use/mcp.html) ·
  [codemods on Martin Fowler](https://martinfowler.com/articles/codemods-api-refactoring.html)
- **Dev workflow** — [cargo-dist](https://axodotdev.github.io/cargo-dist/) ·
  [mise devcontainer generation](https://mise.jdx.dev/cli/generate/devcontainer.html) ·
  [ACP v2 draft](https://agentclientprotocol.com/announcements/acp-v2-draft) ·
  [State of CLI coding agents mid-2026](https://blog.arcbjorn.com/state-of-cli-coding-agents-2026) ·
  [GitHub stacked PRs](https://explainx.ai/blog/github-stacked-pull-requests-public-preview-july-2026) ·
  [claude-code-action](https://github.com/anthropics/claude-code-action)
- **Eval and context** — [Terminal-Bench](https://www.tbench.ai/) ·
  [harness scaling, arXiv 2601.11868](https://arxiv.org/abs/2601.11868) ·
  [LLM-as-judge reliability](https://deepeval.com/blog/llm-as-a-judge) ·
  [Claude compaction docs](https://platform.claude.com/docs/en/build-with-claude/compaction) ·
  [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) ·
  [llama.cpp prompt caching issue 19494](https://github.com/ggml-org/llama.cpp/issues/19494) ·
  [From Memory to Skills](https://arxiv.org/html/2607.16621v1)
- **Security** — [injection SoK, arXiv 2601.17548](https://arxiv.org/abs/2601.17548) ·
  [Unit 42 on in-the-wild agent injection](https://unit42.paloaltonetworks.com/ai-agent-prompt-injection/) ·
  [OWASP prompt-injection prevention cheat sheet](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html) ·
  [CaMeL summary](https://simonw.substack.com/p/camel-offers-a-promising-new-direction) ·
  [gitleaks](https://github.com/gitleaks/gitleaks) ·
  [cargo audit vs deny vs vet workflow](https://safeguard.sh/resources/blog/cargo-audit-deny-advisories-workflow) ·
  [Claude Code sandboxing](https://code.claude.com/docs/en/sandboxing) ·
  [bubblewrap layered sandboxing](https://labs.esokia.com/post/sandboxing-claude-code-cli-linux-bubblewrap/)
- **Multimodal** — [ratatui-image](https://github.com/ratatui/ratatui-image) ·
  [llama.cpp multimodal / mmproj](https://www.aidoczh.com/llama-cpp/docs/models/multimodal/) ·
  [PaddleOCR-VL](https://arxiv.org/html/2510.14528v2) ·
  [chromiumoxide vs the alternatives](https://dev.to/vhub_systems_ed5641f65d59/puppeteer-in-rust-chromiumoxide-and-headlesschrome-vs-the-python-alternative-4ji0) ·
  [Anthropic computer-use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool) ·
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ·
  [typst on docs.rs](https://docs.rs/typst)
- **Local-first** — [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve) ·
  [serve CLI reference (`--tcp`, `--tls-terminated-tcp`)](https://tailscale.com/docs/reference/tailscale-cli/serve) ·
  [Tailscale Funnel limits](https://tailscale.com/docs/features/tailscale-funnel) ·
  [self-host a local AI stack on a tailnet](https://tailscale.com/blog/self-host-a-local-ai-stack) ·
  [Claude Code Remote Control](https://code.claude.com/docs/en/remote-control) ·
  [ntfy.sh](https://ntfy.sh/) ·
  [Mutagen](https://mutagen.io/documentation/synchronization/) ·
  [Syncthing-Android maintainer handover](https://news.ycombinator.com/item?id=46184730) ·
  [Google Workspace MCP servers (Developer Preview)](https://developers.google.com/workspace/guides/configure-mcp-servers) ·
  [model provenance attestation draft-00](https://datatracker.ietf.org/doc/draft-sharif-ai-model-lifecycle-attestation/00/)

Two notes on trusting this list: several 2026 pages could not be deep-fetched
during the passes (quota), so any figure the appendix marks **UNVERIFIED** is
snippet-level and must be re-read before it is relied on. Vendor rate limits
(L-11, free-tier quotas) drift by design in particular.

