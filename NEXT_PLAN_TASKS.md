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
- [x] CLI subcommands — scan, config, models, cache, query, memory, server, analyze, plugin, llamacpp, tui
- [x] Workspace gates green — 13 crates, 417 tests passing, zero warnings

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
- [ ] Collaboration session security review and polish

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

- [ ] D1-01 `tasks` module in `xencode-core-rs`: `TaskRecord` (id, name, command, pid,
  status Running/Exited(s)/Killed, capped output tail), tokio-backed spawn, registry
  with `start`/`poll`/`stop`/`list`/`remove`. Pure state transitions unit-testable
  without real processes.
- [ ] D1-02 Tool surface: `background_start`/`background_poll`/`background_stop` as
  `ToolDefinition`s in `xencode-providers-rs/tools.rs` shapes, and wire tool-call
  **execution** into the agent turn loop (`generate_stream_with_tools` exists in the
  providers; the TUI loop currently never handles a `ToolCall` — that lands here).

### D2 — Background tasks UX

- [ ] D2-01 TUI `tasks.rs` panel modeled on `review.rs`: `FocusArea::TaskManager`,
  Feature Navigator entry + global hotkey, task list with status colors, Enter →
  output detail pane (scrollable), `x` → stop, `d` → remove finished.
- [ ] D2-02 CLI: `xencode tasks list | start <cmd> | poll <id> | stop <id> | rm <id>`.
- [ ] D2-03 Tests: registry lifecycle (start→exit→rm, stop of dead task, output cap)
  + headless frame renders for the panel.

### D3 — Worktree support

- [ ] D3-01 Git helpers in `xencode-context-rs` (`worktree.rs`, alongside `gitinfo.rs`):
  parse `git worktree list --porcelain` (pure fn), add/remove shells with explicit
  args; dirty marker reuse via `dirty_paths`.
- [ ] D3-02 TUI `WorktreePanel`: list worktrees (path, branch, HEAD, dirty flag),
  add (path+branch prompt), remove with confirm; guard: main worktree never
  removable. Same panel pattern as D2-01.
- [ ] D3-03 Agent integration: background tasks (D1) take an optional `cwd` so a task
  can run inside a chosen worktree; worktree list included in the context bundle.
- [ ] D3-04 CLI: `xencode worktree list | add <path> [<branch>] | remove <path>`.

### D4 — Close-out

- [ ] D4-01 Docs sweep: README structure + CLI_GUIDE subcommands, NEXT_PLAN.md counts,
  CHANGELOG entry; `cargo test --workspace` count recorded.

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
- [ ] E2-04 Scroll fixes: use or delete `file_scroll_offset` (`app.rs:272`); clamp
  provider-health/security scroll to row count (`app.rs:3591, 3594`); make
  CodeReview diff output scrollable (`ui.rs:917-921`).
- [ ] E2-05 Mouse wheel support for the remaining focus areas (currently 7 of 18;
  `_ => {}` at `app.rs:4293`).

### E3 — High-impact UX (UI doc §4, first half)

- [ ] E3-01 Markdown rendering in chat: block/heading/inline-code/URL splitter
  rendered as styled `Line`s (no new deps unless justified); code fences visually
  distinct while streaming. Headless render tests.
- [ ] E3-02 Help overlay (`F1` or `?`): keybindings per current focus area,
  generated from the real key handler — no fiction.
- [ ] E3-03 Toast/notification layer: file-watch warnings become transient overlays
  (`app.rs:440-449`), not fake system chat lines.

### E4 — Theme & layout consistency (UI doc §3)

- [ ] E4-01 Funnel ~63 raw `Color::` uses in `ui.rs` through `ThemeColors` slots
  (add slots as needed: diff add/remove, severity levels, panel border dim).
- [ ] E4-02 One layout function feeding both render and mouse-hit testing — kill
  the duplicated 20%/50%/30% and 20%/70% magic numbers (`ui.rs:178-210`,
  `app.rs:4322-4324`).
- [ ] E4-03 Settings nav bound from the actual row list, not the hardcoded `14`
  (`app.rs:3565` vs `ui.rs:687`).
- [ ] E4-04 Light theme + config toggle; dedupe the triplicated theme-cycle lists
  (`ui.rs:574`, `app.rs:4026, 4108`).

### E5 — Input UX (UI doc §4, second half)

- [ ] E5-01 Multiline chat input via `tui_textarea` (already a dep for the editor),
  Alt+Enter sends newline, Enter submits; keep single-line feel by default.
- [ ] E5-02 Input history recall (e.g. Alt+Up/Down; plain Up/Down keep scrolling
  chat) + slash-command autocomplete on `/`.

### E6 — Structure (UI doc §5) — ride along, last

- [ ] E6-01 Split the 887-line key `match` (`app.rs:3366-4252`) into per-focus
  `handle_key` fns + a global chord table; pure move, no behavior change, tests
  pin existing bindings first.
- [ ] E6-02 Clamp scroll offsets on `Event::Resize` shrink (`app.rs:4340`).

### E7 — Close-out

- [ ] E7-01 Docs: CHANGELOG entries as items land; `CLI_GUIDE.md`/`USER_MANUAL.md`
  TUI keys section updated for help overlay + new input behavior; counts refreshed.

> Status legend: `[x]` done, `[ ]` todo.