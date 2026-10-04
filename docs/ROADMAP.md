# 🚀 Xencode Next-Level Roadmap (High Impact Strategy)

## 🎯 Vision: The AI Developer Operating System
Transform Xencode from a tool into the **system** developers use for 80% of their daily workflow: Coding & Git.

---

## ⚡ Execution Plan: "Depth Over Breadth"

> **Verified against the tree on 2026-10-03** — 16 crates and 2141 tests passing
> (`cargo test --workspace`). Every line
> below is marked with what the code does today, and the entry points are the
> real ones (`xencode --help`, `?` in the TUI).
> [`NEXT_PLAN_TASKS.md`](../NEXT_PLAN_TASKS.md) is the day-to-day record.

### Phase 1: The Foundation (✅ FROZEN / COMPLETE)
*Core infrastructure is feature-complete. No further expansion here.*
- [x] **Multi-model conversations** - Switch models mid-chat ✨
- [x] **Context-aware responses** - Conversation memory + per-turn context assembly ✨
- [x] **Smart model selection** - Routing from `ModelCapabilities`, sequential fallback across `agent_fallback_models` ✨
- [x] **Project context awareness** - Repo-wide index + retrieval with a token budget (`xencode-context-rs`) ✨
- [x] **Code analysis system** - Per-language heuristics + pattern-based OWASP scanner (`xencode-analysis-rs`) ✨
- [x] **Core types** - `ConversationMemory` (`xencode-memory-rs`), `ResponseCache` (`xencode-cache-rs`), model profiles + health (`xencode-models-rs`)

---

### Phase 2: The Perfect Git Loop
*Goal: The world's best AI-powered Git assistant.*

#### 1. Review (the core loop)
- [x] **PR Reviewer** - `xencode review [--base main] [--format text|json]` — rename-aware diff triage with per-file analysis
- [x] **Code Review panel** - `Ctrl+R`, per-file AI review of the current file (`xencode-tui-rs`)
- [x] **Review Dashboard** - `Ctrl+Y`, per-file PR-level browsing (base toggle HEAD ↔ main)
- [ ] **AI-generated commit message** - not built: the `Ctrl+S` git commit panel takes a message you type and runs `git commit -am` off the UI thread; nothing proposes the text from the diff

#### 2. Git in the TUI
- [x] **Interactive Diff Viewer** - ratatui diff panel for reviewing changes before committing
- [x] **Commit panel** - `Ctrl+S`: states how many files it will stage
  (`Staging N files...`), runs `git commit -am` off the UI thread and reports the
  result back as a `[GIT_COMMIT_OK]` / `[GIT_COMMIT_ERR]` chat line
- [ ] **Branch Assistant** - not built as a CLI verb; worktrees are (`xencode worktree`, `/spawn`)

---

### Phase 3: ⚡ The Offline Copilot (✅ Shipped as Milestones E/F)
*Goal: Real-time assistance within the loop.*
- [x] **Real-time File Watcher** — debounced `WorkspaceWatcher` (notify) feeds the TUI (Milestone E)
- [x] **Proactive Warnings** — toasts for tracked/attached/open files, enriched with dep-graph dependents (Milestone E)
- [x] **Refactor Suggestions** — `Ctrl+L` insights panel, `/advise`, `xencode advise` and the `repo_advise` agent tool, all reading one `advise_from_snapshot` path (Milestone F)

---

### 📦 Icebox / Long-Term Vision
*Great ideas saved for later to maintain laser focus.*
- **Voice output (text-to-speech)** — input is real since J-07: the panel records
  through `arecord`/`pw-record`/`parec`, meters from the captured PCM and keeps a
  WAV, and transcribes with a whisper CLI when one is installed. Nothing speaks,
  so the panel has no "speaking" state
- **Anthropic without OpenRouter** — `xencode-providers-rs` has an Anthropic
  client, but `ApiKeys` has no `anthropic_api_key` and both entry points pass
  `None`, so an `anthropic:…` id cannot authenticate. Deliberately parked: adding
  the key field is a decision, not a bug
- ~~**Plugin System**~~ — done in Rust (`xencode-plugin-rs`). A `plugin.json` *is*
  the plugin: no dynamic linking, no plugin code. A version-compatible manifest
  loads and reaches every agent turn through its `prompt_prefix` and
  `before`/`after` hooks, which fill only the gaps `agent_hooks` in config.json
  left open
- **Agent Orchestration** (multi-agent debugging) — `/spawn` already runs a
  delegated agent loop in its own git worktree; nothing coordinates several runs
- **CRDT session sync** — `crdt.rs` exists and is deliberately unwired (settled
  Milestone G decision: session state stays in-memory)
- **Commit message generation** — see Phase 2 above
- **VS Code Extension** (Separate product)
- **Web Interface** (Separate product)

## 🛠️ Technical Architecture Evolution

### Current Architecture
```
xencode (Rust binary) → providers-rs → Ollama / llama.cpp / Gemini / Qwen / OpenRouter
                                   ↘ retry + sequential fallback (agent_fallback_models)
                                   ↘ remote:… → any OpenAI-compatible endpoint, incl. the
                                     Colab VM reached through an SSH forward (colab-rs)
```

### Target Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Xencode Ecosystem                    │
├─────────────────────────────────────────────────────────┤
│  CLI Interface  │  TUI (ratatui)  │  API Server (axum)  │
├─────────────────────────────────────────────────────────┤
│        Core Engine (rust/crates/xencode-core-rs)        │
├─────────────────────────────────────────────────────────┤
│ Context │ Cache │ Models │ Providers │ Plugins │ Memory │
├─────────────────────────────────────────────────────────┤
│    Ollama   │   llama.cpp   │   Cloud Providers   │ RAG │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Implementation Progress

### ✅ Shipped since this file was written
1. **Approval-gated agent tool loop** (Milestone I) — 11 tools, a modal prompt per
   mutating call, checkpoints + `/rewind`, plan visibility, pre/post tool hooks,
   `/spawn` in a worktree, MCP stdio servers, sequential provider fallback
2. **Background tasks & worktrees** (Milestone D) — task registry, `Ctrl+K` panel,
   `xencode tasks` / `xencode worktree`
3. **Live refactor insights** (Milestone F) and **team-mode hardening**
   (Milestone G) — bearer tokens + RBAC + a JSONL audit trail on HTTP and WS,
   loopback-by-default bind with opt-in TLS
4. **Every TUI panel tells the truth** (Milestone J, J-01…J-08) — the seven
   scripted panels were replaced with real scans, measurements, model calls and
   microphone capture, and plugin manifests now load
5. **A run that can be looked at afterwards** (Milestone R, W1) — a rollup over the
   metrics log, per-turn traces read back by `/trace`, a seed that travels with a
   llama.cpp request, the instructions a model is given now living as versioned
   prompt files whose digest is recorded with every metric, trace and eval score,
   and `xencode replay <run-id>`, which runs a recorded turn again over a loopback
   port from the bytes it was made of — same stream reader, same permission gate,
   tools really executing, no model answering

### 🚀 Current plan status

Milestone V's TUIOS-informed layout work is partially delivered. The tree-backed
layout, resizing, persistence, templates and related TUI improvements are tracked
in [`NEXT_PLAN_TASKS.md`](../NEXT_PLAN_TASKS.md). **V-9**, the session layout
transition inspector, shipped on 2026-09-29 as `Ctrl+0`. **V-10**, which would
open a pane for observed worker permission requests and report worker state, is
not complete: it waits for measured and normalized worker events from AR-1,
AR-4, AR-5 and AR-9. V-11's automatic rearrangement remains parked. Do not read
the TUIOS adaptation as complete while V-10 is still open.

The ordered backlog and its dependency waves remain in
[`NEXT_PLAN_TASKS.md`](../NEXT_PLAN_TASKS.md); the milestone summary is in
[`NEXT_PLAN.md`](../NEXT_PLAN.md). Research appendices are evidence and candidate
records, not a list of shipped behavior.

**Free compute, re-read from source on 2026-10-04.** Milestone L's 2026-09-23
backend survey put Kaggle in the "no SSH, therefore unreachable" bucket. Reading
the projects that actually run a model server on free GPUs changed that: Kaggle
runs notebook code as root, a prebuilt Linux CUDA build of `llama-server` exists
so the community's 26-minute compile is optional, `--api-key` is one flag rather
than a project, and the weights-and-binary cache that makes a session start in a
minute is a private Kaggle dataset rather than Google Drive (which Kaggle cannot
mount). The open problem is transport, not capacity, and every public example
found solves it by exposing an unauthenticated endpoint to the internet — which
this project's own rules forbid. Separately, AMD's **$100 Developer Cloud credit**
gives a root-SSH MI300X VM, which the existing bring-your-own-SSH path already
handles. Both are recorded as **L-13 → L-15** with their numbers dated and their
unverified points marked; none of it is shipped behavior, and each is gated on a
probe that has not run.

## 📊 Success Metrics & Current Status

### 🎯 Target Metrics
- **Developer Productivity**: Reduce coding time by 40%
- **Code Quality**: Improve code review efficiency by 60%
- **User Adoption**: 1000+ active users within 6 months
- **Feature Usage**: 80% of users use 3+ advanced features
- **Performance**: Sub-second response times for all operations

### 📈 Metrics as measured (2026-09-29)
- **Rust Migration**: 16/16 crates — complete; the Python stack is deleted
- **Test Suite**: 1694 passing, 0 failing (`cargo test --workspace`)
- **Compilation**: `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo fmt --all --check` clean
- **Code Quality**: per-language heuristics + pattern-based OWASP scanner
  (`xencode-analysis-rs`)
- **Context Awareness**: repo-wide indexing + per-turn retrieval (`xencode-context-rs`)
- **Model Intelligence**: offline `ModelCapabilities` + status-driven fallback routing
