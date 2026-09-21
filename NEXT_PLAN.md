# Next Plan — Xencode Roadmap Tracking

> Companion doc to [docs/ROADMAP.md](docs/ROADMAP.md). This file tracks the
> current focus and immediate next milestones for Xencode. **Verified against
> the tree on 2026-09-21** — day-to-day detail in
> [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md).

## Current Status

- ✅ **Milestone A complete**: Reliability hardening (transport retries, diagnostics, model lock, smoke gate)
- ✅ **Milestone B complete**: Agentic MVP stability (workflow loop, auto-fix suggestions, hotkeys) + the voice panel, scripted until J-07
- ✅ **Milestone C complete**: Deep dev workflow (git automation, diff panel, replay, NL terminal safety)
- ✅ **Milestone D complete**: Background tasks & worktrees (task registry + `Ctrl+K` panel + agentic `background_*` tools + `xencode tasks`/`worktree` CLI + `Ctrl+O` worktree panel)
- ✅ **Milestone F complete**: Live refactor insights (watcher-driven snapshot refresh + `Ctrl+L` insights panel + `/advise` + `xencode advise` CLI + `repo_advise` agent tool)
- ✅ **Milestone G complete**: Team mode hardening (bearer-token auth + RBAC on HTTP and WS, first-frame WS identity, JSONL audit trail, local-first bind with opt-in TLS, real TUI Collaboration Hub)
- ✅ **Milestone I complete** (2026-09-21): approval-gated agent tool loop with checkpoints and `/rewind`, plan visibility, real ByteBot delegation, MCP stdio tool servers, pre/post tool hooks, `/spawn` in a worktree, and a sequential provider fallback chain — closed out by **I4-02**, the manuals-vs-implementation honesty sweep
- ✅ **Rust migration complete**: 14 crates, 751 tests passing, zero warnings — the Rust workspace is the only active codebase
- ✅ **Milestone J complete** (2026-09-21): every panel tells the truth.
  The I4-02 sweep left seven scripted TUI panels and a manifest-only plugin
  surface; dead Rust, the Python-era tooling configs and the unused k8s /
  Prometheus assets are already gone. **J-01 to J-08 are done** — the security
  auditor scans the workspace, the profiler measures, the terminal assistant
  asks a model and runs what it picks through the agent's approval gate, the
  multi-language panel tabulates a real `scan_tree` walk and translates
  through a real model call, the custom models panel edits real
  `model_profiles` in `config.json`, the learning mode panel teaches files
  the project index actually found, the voice panel records from the
  microphone and keeps a WAV, transcribing only when a whisper CLI exists,
  and a discovered, version-compatible `plugin.json` now registers with the
  host and reaches every agent turn through its prompt prefix and hooks —
  with `/plugin` and `xencode plugin list` both reporting what actually took
  hold. No TUI panel ships scripted content, and no manifest claims a
  capability this build lacks. Task breakdown and the done-when rule for each
  item: [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md) § Milestone J.
- ⏸️ **Deliberately parked, not gaps to "fix"**: `AnthropicProvider` stays
  unreachable until an `anthropic_api_key` is a decision someone makes, and
  `crdt.rs` stays unwired (settled Milestone G deferral).

## Backlog (all shipped — tracked in [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md))

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
4. **Secure team workflows hardening** — ✅ complete 2026-09-20 as **Milestone G —
   Team Mode Hardening**: the orphaned RBAC/audit crate is wired into the server
   behind real bearer tokens, WS identity moved from the URL to a first `auth`
   frame with close codes for every refusal, joins/denials persist to a JSONL
   audit log, `xencode server` binds loopback by default with opt-in TLS — and
   the TUI Collaboration Hub stopped simulating: it connects for real, showing
   live members with their roles.
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