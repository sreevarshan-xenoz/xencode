# Next Plan — Task Checklist

> Working task list for the active backlog. See [NEXT_PLAN.md](NEXT_PLAN.md) for
> the milestone overview and [docs/ROADMAP.md](docs/ROADMAP.md) for the long-term roadmap.

## Phase 3+ Intelligence

- [ ] Repo-wide context indexing (vector store + graph store, `xencode/rag/`)
- [ ] Routing intelligence: prompt router + provider health awareness
- [ ] Real-time file watcher with proactive warnings
- [ ] Live refactor suggestions

## Fallback Governance

- [ ] Provider health dashboard/panel (TUI)
- [ ] Retry budget policies per provider (`xencode/routing/retry_budget.py`)
- [ ] Fallback policy configuration UX (`xencode/routing/fallback_config.py`)

## Multimodal UX

- [ ] Image input pipeline (`xencode/multimodal/image_analyzer.py`)
- [ ] Document parsing into context (`xencode/multimodal/document_parser.py`)
- [ ] Web extraction for research (`xencode/multimodal/web_extractor.py`)

## Secure Team Mode

- [ ] Workspace RBAC hardening (`xencode/workspace/workspace_security.py`)
- [ ] Collaboration session security (`xencode/collaboration/rbac.py`)
- [ ] Audit log coverage for team actions (`xencode/audit/`)

## Rust Migration

- [ ] Port cache, config, memory crates (in progress, `rust/`)
- [ ] TUI foundation (ratatui, `rust/crates/xencode-tui-rs/`)
- [ ] Provider transport with retries (`rust/crates/xencode-providers-rs/`)
- [ ] Server/analysis/collaboration/plugin crates

> Status legend: `[x]` done, `[ ]` todo, `[/]` in progress.
