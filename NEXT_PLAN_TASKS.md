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
- [x] Workspace gates green — 13 crates, 421 tests passing, zero warnings

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

> Status legend: `[x]` done, `[ ]` todo.