# Next Plan — Xencode Roadmap Tracking

> Companion doc to [docs/ROADMAP.md](docs/ROADMAP.md). This file tracks the
> current focus and immediate next milestones for Xencode.

## Current Status (Feb 2026)

- ✅ **Milestone A complete**: Reliability hardening (transport retries, diagnostics, model lock, vault, smoke gate)
- ✅ **Milestone B complete**: Agentic MVP stability (workflow loop, auto-fix suggestions, hotkeys, voice MVP)
- ✅ **Milestone C complete**: Deep dev workflow (git automation, diff panel, replay, NL terminal safety)
- 🚧 **Active backlog**: Phase 3+ intelligence, fallback governance, multimodal UX, secure team mode

## Active Backlog

1. **Repo-wide context indexing + routing intelligence** — RAG/vector indexing across the
   whole workspace, fed into model routing decisions.
2. **Smart fallback policy governance** — provider health UX, retry budgets, and policy-driven
   fallback routing (see `xencode/routing/`).
3. **Multimodal inputs** — image/document/voice input paths (see `xencode/multimodal/`).
4. **Secure team workflows** — workspaces, collaboration, RBAC hardening (see `xencode/workspace/`
   and `xencode/collaboration/`).

## Near-Term Milestones

- **Phase 3 intelligence**: real-time file watching, proactive warnings, refactor suggestions.
- **Rust migration**: continue porting core subsystems to the Rust workspace (`rust/`), guided by
  [docs/RUST_MIGRATION_PLAN.md](docs/RUST_MIGRATION_PLAN.md) and [docs/RUST_MIGRATION_STATUS.md](docs/RUST_MIGRATION_STATUS.md).
- **Git loop completion**: review dashboard, PR review browsing in the TUI.

## Task Breakdown

Day-to-day task tracking lives in [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md).
