# Next Plan — Xencode Roadmap Tracking

> Companion doc to [docs/ROADMAP.md](docs/ROADMAP.md). This file tracks the
> current focus and immediate next milestones for Xencode. **Verified against
> the tree on 2026-09-19** — see [docs/RUST_MIGRATION_STATUS.md](docs/RUST_MIGRATION_STATUS.md).

## Current Status

- ✅ **Milestone A complete**: Reliability hardening (transport retries, diagnostics, model lock, vault, smoke gate)
- ✅ **Milestone B complete**: Agentic MVP stability (workflow loop, auto-fix suggestions, hotkeys, voice MVP)
- ✅ **Milestone C complete**: Deep dev workflow (git automation, diff panel, replay, NL terminal safety)
- ✅ **Rust migration complete**: 13 crates, 331 tests passing, zero warnings — the Rust workspace is the only active codebase
- 🚧 **Active backlog**: real-time intelligence (file watcher), multimodal UX, team-mode hardening, PR review dashboard

## Active Backlog (all Rust — tracked in [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md))

1. **Real-time file watcher + proactive warnings** — watch the workspace and surface
   warnings/refactor hints as files change (new capability; no watcher exists today).
2. **Refactor suggestions** — build on the `xencode-context-rs` index/symbol graph
   (`rust/crates/xencode-context-rs/`) to propose live improvements.
3. **Multimodal inputs** — image/document input paths (no Rust implementation yet).
4. **Secure team workflows hardening** — audit-log coverage for team/RBAC actions and
   workspace-permission tightening on top of `rust/crates/xencode-collaboration-rs/`.
5. **Git loop completion** — PR review browsing in the TUI; a per-file Code Review
   panel exists (`rust/crates/xencode-tui-rs/`), a PR-level dashboard does not.

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