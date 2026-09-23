# 🚀 Xencode Next-Level Roadmap (High Impact Strategy)

## 🎯 Vision: The AI Developer Operating System
Transform Xencode from a tool into the **system** developers use for 80% of their daily workflow: Coding & Git.

---

## ⚡ Execution Plan: "Depth Over Breadth"

> **Verified against the tree on 2026-09-23** — 15 crates, 815 tests. Every line
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

### 🚀 Next up
No *committed* milestone. **Milestone K** (remote providers + the Google Colab GPU
bridge) closed on 2026-09-23, verified against a live free-tier T4 rather than a
mock; nothing has been green-lit since.

What exists is planning, tracked in
[`NEXT_PLAN_TASKS.md`](../NEXT_PLAN_TASKS.md): two planned tracks — **L** (any
machine you can SSH into, and an agent that finishes its own work) and **M**
(ecosystem compatibility: hooks, skills, agents-as-markdown, MCP server, ACP) —
plus five research appendices — **N**, **O**, **P**,
**Q** (an external hundred-proposal review, dispositioned item by item) and **S** (an
external thirty-eight-proposal review of running other vendors' coding agents, checked
against what those agents already do) —
which between them record 208 candidate features from N–Q, 26 further tasks from S, a
do-not-build register for each family, and the defects found while checking claims
against the code. The
appendices were deliberately never ranked by worth, and still are not. The
**order** was settled afterwards as **Milestone R**: eighteen dependency waves
over all 284 recorded plan items — the 208 research candidates, the 31 further Q
IDs that turned out to be folds or refinements of candidates already counted, the
19 committed tasks in L and M, and S's 26 — sequenced by what would otherwise inherit another
item's broken measurement — correctness first, then observability, the model
substrate, code intelligence, verification, trust, knowledge, autonomy, and the
product surface last. What remains an owner decision is valuation within a wave
(effort, defect closure, daily-driver value, how much of the local-first story
each item protects), not sequence. The icebox above still feeds it.

## 📊 Success Metrics & Current Status

### 🎯 Target Metrics
- **Developer Productivity**: Reduce coding time by 40%
- **Code Quality**: Improve code review efficiency by 60%
- **User Adoption**: 1000+ active users within 6 months
- **Feature Usage**: 80% of users use 3+ advanced features
- **Performance**: Sub-second response times for all operations

### 📈 Metrics as measured (2026-09-23)
- **Rust Migration**: 15/15 crates — complete; the Python stack is deleted, and
  Milestone K added `xencode-colab-rs` as the 15th
- **Test Suite**: 815 passing, 0 failing, 4 ignored (`cargo test --workspace`)
- **Compilation**: `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo fmt --all --check` clean
- **Code Quality**: per-language heuristics + pattern-based OWASP scanner
  (`xencode-analysis-rs`)
- **Context Awareness**: repo-wide indexing + per-turn retrieval (`xencode-context-rs`)
- **Model Intelligence**: offline `ModelCapabilities` + status-driven fallback routing
