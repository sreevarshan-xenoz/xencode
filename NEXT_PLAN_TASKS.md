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
- [x] Workspace gates green — 13 crates, 331 tests passing, zero warnings

## Real-Time Intelligence (Phase 3+)

- [x] Real-time file watcher with proactive warnings (`rust/crates/xencode-context-rs/`, new watcher module) — `WorkspaceWatcher` on `notify`, debounced; TUI surfaces warnings for tracked/attached/open files
- [x] Live refactor suggestions on the symbol graph (`advise` module: cycles, hubs, orphans; `/advise [filter]` in the TUI)
- [x] Proactive bug warnings ("you just introduced a bug") (`broken_imports`, `affected_dependents` → dependents named in watch warnings)

## Multimodal UX

- [ ] Image input pipeline (`rust/crates/xencode-analysis-rs/` or a new `-multimodal` crate)
- [ ] Document parsing into context (`rust/crates/xencode-context-rs/`)
- [ ] Web extraction for research

## Secure Team Mode

- [ ] Audit log coverage for team actions (`rust/crates/xencode-server-rs/` or `-collaboration-rs/`)
- [ ] Workspace RBAC hardening (`rust/crates/xencode-collaboration-rs/workspace.rs`)
- [ ] Collaboration session security review and polish

## Git Loop Completion

- [ ] PR review dashboard / PR-level review browsing in the TUI
  (per-file Code Review panel exists in `rust/crates/xencode-tui-rs/`)

> Status legend: `[x]` done, `[ ]` todo.