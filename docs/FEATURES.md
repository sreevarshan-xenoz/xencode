# Xencode Future Features

> Wild ideas for post-Rust-migration development.

---

## Status Legend

| Tag | Meaning |
|-----|---------|
| `ready` | Migration complete, ready for feature work |
| `exploration-needed` | Need more design before spec |
| `nice-to-have` | Lower priority, opportunistic |
| `blocked` | Depends on another feature |

---

## ✅ Rust Migration Complete

The full Rust migration (Phases 5–8) is now **complete** — 12 crates, 65 tests, zero warnings.

All features below are now on the `ready` track. The foundation (scan, config, cache, memory, providers, TUI, CLI, analysis, server, collaboration, plugin) is implemented.

---

## 1. Multi-Model Arena Mode
**Tag:** `ready`  
**Priority:** High

### What
Run the same prompt through N local models simultaneously and render a
side-by-side diff panel in the TUI. Each model gets its own pane with
streaming token output + latency/metrics strip at the bottom.

### Why
- Evaluate which local model performs best on your specific codebase/tasks
- Great for choosing the right model per feature area (fast vs. verbose vs. refactor)
- No equivalent tool exists in the local-LLM space — this is a differentiator

### Core UX
- `/arena "explain this function"` → splits screen into N panes
- Each pane: model name, streaming response, elapsed time, token count
- Keyboard: `Tab` cycles focus between panes, `Up/Down` navigates history
- Final state: highlight winner by a configurable metric (speed, length, manual vote)

### Technical Notes
- Uses existing `ProviderManager.generate_stream` per model
- Needs a new `ArenaManager` that fans out requests and aggregates results
- New `FocusArea::ArenaPane(n)` enum variant required
- Storage: arena results persisted to `~/.xencode/arena/` for replay
- Could extend to "judge mode" — pick one model as arbiter and score others vs it

---

## 2. Git Autopilot Agent
**Tag:** `ready`  
**Priority:** High

### What
Autonomous git loop that watches your workspace for changes, drafts commit
messages from the actual diff context, and presents an "intent feed" in the
TUI — you review and approve before anything runs. Opens PRs via `gh`.

### Why
- The most tedious part of dev workflow is commit message writing and PR hygiene
- Current tooling (commitizen, git-cz) uses generic templates
- An AI that sees your actual diff + message history → genuinely great commits without
  you having to explain context
- Fits naturally into the TUI as a persistent sidebar panel

### Core UX
- `/autopilot on` → starts watching filesystem with `notify` crate
- Changes trigger a staged intent: list of changed files + proposed commit message
- TUI shows intent feed: each entry is `[proposed] <commit msg> (+3 files, -12 lines)`
- You: `Enter` to approve, `e` to edit message, `d` to dismiss
- On approve: `git add --patch` (hunk-level) or `git add -A`, then commit
- Auto-PR via `gh` CLI if branch differs from base
- `/autopilot off` to disable

### Technical Notes
- Filesystem watcher via `notify` crate (same binary, no system dependencies)
- Parse `git diff --cached` and `git diff` to build context for the AI
- Intent feed is a state machine: `Watching → PendingReview → Approved → Committed`
- Uses existing Ollama client — prompt includes diff + recent commit history
- Git credentials: reuse `gh auth status` token, no new secret management needed
- Branch naming: auto-generate from ticket/PR title via AI
- PR body: AI-generated from commit log of branch

### Edge Cases
- Conflict detection before merge → surface in TUI with resolution options
- Large diffs → chunk into reviewable hunks (existing `/patch` code can be borrowed)
- Multiple pending intents → queue them, process in order
- Detached HEAD → warn and block autopilot

---

## 3. Project Genome Browser
**Tag:** `exploration-needed`  
**Priority:** Medium

### What
Transform the flat file explorer into an **interactive dependency graph**.
Files/imports become nodes, relationships become edges. Keyboard-navigable
with drill-down at each node. Like a mini IDE code map built into the TUI.

### Why
- Most devs navigate code by grepping — this makes the structure visible
- Great for onboarding to a new codebase
- Could power "jump to caller", "find all usages", "trace import chain"
- Would make the TUI the central hub for understanding code, not just chatting

### Core UX
- `g` toggle → enters Genome view from file explorer
- Nodes: files (circles), folders (squares grouped under folders)
- Edges: `import`/`require`/`use` statements
- Arrow styles: solid = direct import, dashed = transitive
- Keyboard: `h/j` navigate nodes, `Enter` drill into a file, `Tab` toggle
  relationship mode (imports ↔ dependents), `l` collapse back
- Color by: language, file size, change frequency (from git log), function count

### Technical Notes
- Build graph from existing `scan_workspace` output
- Parse import/require/use statements per-language (Rust `use`, Python `import`,
  JS `require`/`import`, Go `import`)
- Use existing `git diff` history to tag nodes with change frequency
- Graph layout: simple force-directed or hierarchical — `petgraph` crate supports this
- Render via ratatui using block drawing for nodes + ASCII edges, or a simpler
  tree-collapse mode if graphs are too complex for terminal

---

## 4. Voice-First Coding Mode
**Tag:** `exploration-needed`  
**Priority:** Low-Medium

### What
A `/voice` toggle in the TUI: use your mic to speak to the AI, and have
the response read back to you via TTS. A pair-programming loop without ever
touching a keyboard.

### Why
- Developers spend a lot of time hands-off — on walks, cooking, commuting
- Complements, doesn't replace, the text interface
- Impressive demo factor

### Technical Notes
- STT: Whisper.cpp via `whisper-rs` crate (fully local, no cloud)
- TTS: `kaldi` or `espeak` subprocess for simple speech output
- Audio capture: `cpal` crate or platform-specific (Windows: `winapi`)

---

## 5. Context Time-Travel Replay
**Tag:** `exploration-needed`  
**Priority:** Medium

### What
A session replay debugger for AI reasoning. After a chat session ends,
you can scrub through it like a video timeline and see what context
the AI had at each step.

### Why
- AI gave a bad answer? Now you can see *why*
- Extremely powerful for debugging AI behavior, trust-building, and training
- Turns the TUI into a research tool for your own coding workflow

### Technical Notes
- `ConversationMemory` already stores history — needs a richer snapshot model
- New `SessionReplay` struct alongside `ConversationMemory`
- Replay UI: separate `ReplayMode` in TUI with timeline widget

---

## 6. Smart Fallback Health Dashboard
**Tag:** `exploration-needed`  
**Priority:** Medium

### What
A live TUI panel showing the real-time health of every configured model
provider — latency, error rate, retry count, cache hit rate — with
automated fallback governance (auto-disable sick providers, auto-escalate to cloud).

### Why
- Ollama going down is a silent failure right now
- Users don't know which model is actually responding or why

### Core UX
- `Ctrl+H` opens the Health Dashboard overlay (already wired in TUI)
- Status badges: `🟢 Healthy`, `🟡 Degraded`, `🔴 Disabled`, `☁️ Cloud Fallback`
- Auto-governance toggle: ON = system silently switches providers

### Technical Notes
- Extend `ProviderManager` with health metrics collection
- Metrics: request latency p50/p95, error count, timeout count, fallback count
- Health check pings at configurable intervals (default: 60s)

---

## 7. Workspace RAG + Context Indexing
**Tag:** `exploration-needed`  
**Priority:** High

### What
Offline RAG index of the current workspace. When you ask about code,
the system indexes your repo locally (using Ollama embeddings), and
retrieves relevant chunks — letting the AI answer questions about code
it hasn't seen in the current conversation.

### Why
- "What does this codebase do?" → currently needs a long-running chat
- With RAG, you get instant codebase-aware answers from local embeddings
- Offline-first: everything runs on Ollama, no cloud needed

### Technical Notes
- Embeddings: `nomic-embed-text` via Ollama (already in xencode-analysis-rs)
- Chunking: language-aware (Rust fn boundaries, Python class boundaries) — already in xencode-analysis-rs
- Vector store: in-memory with cosine similarity search — already in xencode-analysis-rs

> **Note:** The core building blocks (chunking, embeddings, vector store) already exist in `xencode-analysis-rs`. This feature needs the glue layer + TUI integration.

---

## 8. Secure Team Mode
**Tag:** `exploration-needed`  
**Priority:** Low

### What
A shared Xencode instance (or CLI tool) for engineering teams.
Each member has their own config/model cache. The AI can answer
questions using the shared team's codebase RAG index.

### Why
- Solo tools don't scale to teams
- Shared context bank + shared local models = huge efficiency multiplier
- Teams don't want their data in the cloud

### Technical Notes
- Team server: `xencode server` (already implemented in xencode-server-rs)
- Auth: uses existing auth layer from xencode-server-rs
- Shared index: compute once, serve many (needs Workspace RAG first — Feature #7)

---

## Feature Dependency Graph

```
  Arena Mode ───────────────┐
  ↑ (uses ProviderManager)  │
                             │
  Git Autopilot ─────────────┤
  ↑ (uses Ollama + git)      │ (all ready for implementation)
                             │
  Workspace RAG ─────────────┤
  ↑ (chunk/embed/store done) │
                             │
  Context Time-Travel ───────┤
  ↑ (extends Memory)         │
                             │
  Project Genome ────────────┘
  ↑ (extends scan)

  Health Dashboard ──────────┐
  ↑ (extends ProviderManager)│ (can run in parallel)
                             │
  Secure Team Mode ──────────┘
  ↑ (needs Health + RAG)
```

---

## Next Steps

1. ✅ **Rust migration complete** — foundation is in place
2. 🔲 **Pick a feature** from the list above and follow the brainstorming → design → implementation workflow
3. 🔲 **Set up integration tests** for the TUI before adding major features

---

_Last updated: 2026-06-02 | Branch: `main` | Rust migration fully complete_

---

> 📄 **Reference:** [`xencode-codebase-reference.html`](../xencode-codebase-reference.html) — Complete codebase reference with Rust crate details, TUI panel statuses, test coverage, architecture diagrams, backlog items, and phase-by-phase migration tracking.