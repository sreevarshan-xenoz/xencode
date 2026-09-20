# Next Plan — Xencode Roadmap Tracking

> Companion doc to [docs/ROADMAP.md](docs/ROADMAP.md). This file tracks the
> current focus and immediate next milestones for Xencode. **Verified against
> the tree on 2026-09-20** — see [docs/RUST_MIGRATION_STATUS.md](docs/RUST_MIGRATION_STATUS.md).

## Current Status

- ✅ **Milestone A complete**: Reliability hardening (transport retries, diagnostics, model lock, vault, smoke gate)
- ✅ **Milestone B complete**: Agentic MVP stability (workflow loop, auto-fix suggestions, hotkeys, voice MVP)
- ✅ **Milestone C complete**: Deep dev workflow (git automation, diff panel, replay, NL terminal safety)
- ✅ **Milestone D complete**: Background tasks & worktrees (task registry + `Ctrl+K` panel + agentic `background_*` tools + `xencode tasks`/`worktree` CLI + `Ctrl+O` worktree panel)
- ✅ **Milestone F complete**: Live refactor insights (watcher-driven snapshot refresh + `Ctrl+L` insights panel + `/advise` + `xencode advise` CLI + `repo_advise` agent tool)
- ✅ **Rust migration complete**: 13 crates, 508 tests passing, zero warnings — the Rust workspace is the only active codebase
- 🚧 **Active backlog**: team-mode hardening

## Active Backlog (all Rust — tracked in [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md))

0. **Milestone D — Background tasks & worktree support** — ✅ complete 2026-09-20:
   a background-task registry + TUI task panel + `tasks`/`worktree` CLI, and git
   worktree creation/browsing so agent runs can be isolated per task.
1. **Real-time file watcher + proactive warnings** — shipped (Milestone E): a
   debounced `WorkspaceWatcher` (notify) feeds the TUI, which warns — as toasts —
   only about tracked/attached/open files, enriched with dep-graph dependents.
2. **Refactor suggestions** — ✅ complete 2026-09-20 as **Milestone F — Live
   Refactor Insights**: the watcher refreshes the symbol/dep snapshot per edited
   `.rs` file, a `Ctrl+L` insights panel + `/advise` + `xencode advise` CLI +
   `repo_advise` agent tool all read one shared `advise_from_snapshot` path.
3. **Multimodal inputs** — shipped (Milestone E): image analysis and TUI attach as
   per-backend message parts, plus PDF/DOCX text extraction into the context bundle.
4. **Secure team workflows hardening** — workspace RBAC and audit events (membership
   changes, last-admin guard, denials) already ship in `xencode-collaboration-rs`;
   remaining hardening on top, and the TUI hub session still runs simulated members.
5. **Git loop completion** — PR review browsing in the TUI: shipped (Milestone E,
   `Ctrl+Y` per-file dashboard); remaining ideas live in later milestones.

## Done (recent Rust work, not new work)

- **Repo-wide context indexing + routing intelligence** — `xencode-context-rs`
  (index/embed/retrieve + per-turn context assembly and budgeting) and
  `xencode-providers-rs` (routing, `ModelCapabilities`, fallback).
- **Fallback governance** — status-code-driven retriability + retry budgets
  (`rust/crates/xencode-providers-rs/retry.rs`) and the Provider Health TUI panel.
- **Tool-calling plumbing** — `ToolDefinition`/`ToolCall`/`AgentTurn` +
  `generate_stream_with_tools` across ollama, llamacpp, openrouter, qwen.
- **Context-window budgeting** — live turns budgeted from the model's real window.

## Task Breakdown

Day-to-day task tracking lives in [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md).