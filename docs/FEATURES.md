# Xencode Future Features

> All wild ideas go here. Revisit after Rust migration is complete.

---

## Status Legend

| Tag | Meaning |
|-----|---------|
| `post-migration` | Implement after full Rust TUI migration |
| `exploration-needed` | Need more design before spec |
| `nice-to-have` | Lower priority, opportunistic |
| `blocked` | Depends on another feature |

---

## 1. Multi-Model Arena Mode
**Tag:** `post-migration`  
**Priority:** High  
**Owner:** TBD

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
**Tag:** `post-migration`  
**Priority:** High  
**Owner:** TBD

### What
Autonomous git loop that watches your workspace for changes, drafts commit
messages from the actual diff context, and presents an "intent feed" in the
TUI — you review and approve before anything runs. Opens PRs via `gh`.

### Why
- The most tedious part of dev workflow is commit message writing and PR hygiene
- Current tooling (commitizen, git-cz) uses generic templates
- An AI that sees your actualdiff + message history → genuinely great commits without
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
**Blocked by:** Core workspace scan already exists in `xencode_core_rs`

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
- **Exploration mode**: start focused, expand relationships on demand

### Exploration Items
- Is force-directed layout feasible in terminal without a GUI? Maybe simpler:
  collapsible tree with relationship expansion
- Performance: large repos (50k+ files) need indexing + lazy loading
- Should this be `post-migration` or can it be a Rust-only feature first?

---

## 4. Voice-First Coding Mode
**Tag:** `exploration-needed`  
**Priority:** Low-Medium  
**Owner:** TBD

### What
A `/voice` toggle in the TUI: use your mic to speak to the AI, and have
the response read back to you via TTS. A pair-programming loop without ever
touching a keyboard.

### Why
- Developers spend a lot of time hands-off — on walks, cooking, commuting
- Idea: "debug this todo list" while making coffee
- Complements, doesn't replace, the text interface
- Impressive demo factor

### Core UX
- `/voice on` → starts listening (WebRTC-compatible via `cpal` + `hound` or
  a speech-to-text API)
- Speak prompt → transcribed via Whisper.cpp (local) or cloud STT → sent to Ollama
- Response streamed to TTS engine (e.g., `rodio`, `kaldi` via subprocess)
- `Ctrl+C` interrupts and stops audio
- Toggle off → back to text mode

### Technical Notes
- STT: Whisper.cpp via `whisper-rs` crate (fully local, no cloud)
- TTS: `kaldi` or `espeak` subprocess for simple speech output
- Audio capture: `cpal` crate or platform-specific (Windows: `winapi`)
- Fallback: if local STT fails, prompt user to use cloud (OpenAI Whisper or similar)
- Mode indicator in TUI: waveform/stereo bars when recording

### Exploration Items
- Does local Whisper model work well enough for coding jargon?
- Should TTS be optional (audio-only) or also stream text in TUI?
- Mic hotword detection vs. push-to-talk? Push-to-talk safer for terminal environments

---

## 5. Context Time-Travel Replay
**Tag:** `exploration-needed`  
**Priority:** Medium  
**Owner:** TBD

### What
A session replay debugger for AI reasoning. After a chat session ends,
you can scrub through it like a video timeline and see what context
(conversation history, retrieved files, memory) the AI had *at each step*.

### Why
- AI gave a bad answer? Now you can see *why* — was it bad context?
  wrong memory retrieval? wrong model prompt?
- Extremely powerful for debugging AI behavior, trust-building, and training
- Turns the TUI into a research tool for your own coding workflow

### Core UX
- `/replay` → opens replay mode after session ends
- Timeline scrubber: one stop per AI turn
- At each stop: shows prompt sent + full context bundle at that moment
  (memory retrieval results, file reads, config state, model params)
- `←/→` to scrub, `Enter` to expand a context item, `q` to exit
- Final screen: "Why was this wrong?" — AI classifier on worst turns
- Export: save session as JSON for sharing/debugging

### Technical Notes
- `ConversationMemory` already stores history — needs a richer snapshot model
  that captures context at each turn (not just messages)
- New `SessionReplay` struct alongside `ConversationMemory`
- Snapshots: turn number, timestamp, messages so far, retrieved docs,
  file reads, config snapshot, model + temperature used
- Replay UI: separate `ReplayMode` in TUI with timeline widget
- Serialization: snapshots + messages → JSON, stored in `~/.xencode/replays/`
- Optional: diff view showing what changed in context between turn N and turn N+1

### Exploration Items
- How much context is too much to display? Need smart summaries per snapshot
- Can this integrate with Arena Mode? "Why did Model A win here vs. Model B?"

---

## 6. Smart Fallback Health Dashboard
**Tag:** `exploration-needed`  
**Priority:** Medium  
**Blocker:** Depends on existing provider health work in `xencode-providers-rs`

### What
A live TUI panel showing the real-time health of every configured model
provider — latency, error rate, retry count, cache hit rate — with
automated fallback governance (auto-disable sick providers, auto-escalate to cloud).

### Why
- Ollama going down is a silent failure right now
- Users don't know which model is actually responding or why the response
  came from Cloud instead of Local
- Would turn "it works / it doesn't" into observable engineering

### Core UX
- `Ctrl+h` opens the Health Dashboard as an overlay panel
- Shows: provider card per Model (name, last response time, error count,
  cache hits, current status badge)
- Status badges: `🟢 Healthy`, `🟡 Degraded`, `🔴 Disabled`, `☁️ Cloud Fallback`
- Auto-governance toggle: ON = system silently switches providers
- Manual mode: you see a prompt before each fallback

### Technical Notes
- Extend `ProviderManager` with health metrics collection
- Metrics: request latency p50/p95, error count, timeout count, fallback count
- Health check pings at configurable intervals (default: 60s)
- Store health history in `xencode-cache-rs` for trend analysis
- Governance rules as config: `providers[].health.threshold_error_rate = 0.3`
- TUI: existing ratatui dashboard, update via tokio channel

---

## 7. Workspace RAG + Context Indexing
**Tag:** `exploration-needed`  
**Priority:** High  
**Blocker:** Needs Phase 3 intelligence work  
**Owner:** TBD

### What
Offline RAG index of the current workspace. When you ask about code,
the system indexes your repo locally (using Ollama embeddings), and
retrieves relevant chunks — letting the AI answer questions about code
it hasn't seen in the current conversation.

### Why
- "What does this codebase do?" → currently needs a long-running chat
- With RAG, you get instant codebase-aware answers from local embeddings
- Offline-first: everything runs on Ollama, no cloud needed
- Complements the existing `ConversationMemory` with **semantic** retrieval

### Core UX
- `/index` → builds initial workspace index (show progress bar in TUI)
- `/index update` → incremental re-index on file changes
- `/index off` → disable
- Query behavior: transparently injects top-K relevant chunks into prompt
- Index stats shown in status bar: `📚 1,247 chunks indexed`

### Technical Notes
- Embeddings: `nomic-embed-text` via Ollama (small, fast, already local)
- Chunking: language-aware (Rust fn boundaries, Python class boundaries,
  generic line-based fallback)
- Vector store: `xencode-memory-rs` extends to store embeddings (Qdrant-like
  HNSW, but lightweight — can use `r3bl_hnsw` or simple cosine top-K scan)
- Index file format: JSON with file path + chunk + embedding vector
- Re-index triggers: manual, on startup, or via file watcher
- Query path: embed question → top-K ANN search → inject into context

---

## 8. Secure Team Mode
**Tag:** `exploration-needed`  
**Priority:** Low  
**Owner:** TBD

### What
A shared Xencode instance (or CLI tool) for engineering teams.
Each member has their own config/model cache. The AI can answer
questions using the shared team's codebase RAG index without leaking
personal project state.

### Why
- Solo tools don't scale to teams
- Shared context bank + shared local models = huge efficiency multiplier
- Teams don't want their data in the cloud

### Core UX
- Team server: `xencode team serve` (uses existing FastAPI or new Rust HTTP layer)
- Members register with invite token
- Shared workspace index visible to all members
- Personal memory remains private
- Per-user audit log in team dashboard

### Technical Notes
- Auth: shared secret or per-user token (stored in team vault)
- Model layer: team runs models, members connect via HTTP
- Shared index: compute once, serve many (workspace RAG needed first — Feature #7)
- Isolation: personal memory per user, team memory shared (separate stores)
- Deployment: single Docker image for team server

---

## Feature Dependency Graph

```
Arena Mode ───────────────────────────────┐
  ↑ (uses ProviderManager.generate_stream) │
  │                                       │
Git Autopilot ────────────────────────────┤ (all post-migration)
  ↑ (uses Ollama + git diff parsing)      │
  │                                       │
Workspace RAG ────────────────────────────┤
  ↑                                       │
  │                                       │
Context Time-Travel ──────────────────────┤
  ↑ (extends ConversationMemory)          │
  │                                       │
Project Genome Browser ───────────────────┤
  ↑ (extends scan_workspace)              │
  │                                       │
Voice Mode ───────────────────────────────┘
  ↑ (uses Ollama)

Health Dashboard ─────────────────────────┐
  ↑ (extends ProviderManager)             │ (can run in parallel)
  │                                       │
Secure Team Mode ─────────────────────────┘
  ↑ (needs Health + RAG as prerequisites)
```

---

## Next Steps

Before implementing any feature:

1. **Finish Rust migration** — migration must be stable and tested first
2. **Set up integration tests** for the TUI — no feature should break existing flows
3. **Choose feature #1** from this list based on team priorities
4. Follow the brainstorming → design → implementation workflow for that feature

---

_Last updated: 2026-04-26 | Branch: `total-migration-rust`_