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
- ✅ **Rust migration complete**: 15 crates, 815 tests passing, 4 ignored, zero warnings — the Rust workspace is the only active codebase (Milestone K added `xencode-colab-rs`)
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
- ✅ **Docs archive purge** (2026-09-21, after J-08): 9,371 lines of documentation
  describing the retired Python product are deleted, not filed away —
  `DOCUMENTATION.md`, `PRD.md`, `project details.md`, `docs/FEATURES.md`,
  `docs/ARCHITECTURE_DIAGRAMS.md`, `BROWSER_LOGIN_PLAN.md`, `docs/superpowers/`
  and three unused screenshots. The survivors were re-checked against the tree
  and `docs/ROADMAP.md` was rewritten from the code (its `xencode --git-*` flags
  and `/smart`-style chat commands were never in the Rust CLI).
- ✅ **Milestone K complete** (2026-09-23): a GPU you do not own. `remote:…`
  routes any OpenAI-compatible endpoint, Settings edits provider endpoints and
  masked keys, and `xencode colab up|status|down|preflight` rents a Colab VM,
  installs llama.cpp (CUDA) or Ollama on it and tunnels the endpoint to
  `127.0.0.1` over the official `colab ssh` bridge — no public URL. Verified end
  to end against a live free-tier T4: a real GGUF answered `xencode query -m
  'remote:…'` through the forward, Provider Health went green, `--reconnect`
  rebuilt a killed tunnel in 9 s, and `down` left nothing billing. That run, not
  a unit test, is what fixed the root login, the 18080 port and the
  `READY`-means-serving gate. Tasks: [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md)
  § Milestone K.
- 🔜 **Milestone M planned** (2026-09-23): *stop being an island — the plugin,
  hooks, skills and MCP track.* Research against the tree found the honest state:
  a "plugin" today is a `plugin.json` manifest that can only contribute a prompt
  prefix and two hooks — `PluginRuntime::handle_event` returns `None`
  unconditionally, so no plugin ever receives an event — and the manifest's
  `permissions` field is parsed and **never enforced**. MCP is a client only,
  stdio only, tools only, hand-rolled with no `rmcp` dependency. Hooks are
  real and can veto, but the command string is static: a hook is never told
  which tool ran or with what arguments. The track fixes compatibility with the
  conventions the rest of the ecosystem already settled on (hook stdin payloads,
  `SKILL.md`, agents-as-markdown, git-installable plugins) before adding the
  two big new surfaces — xencode as an MCP *server* and xencode over ACP. Tasks:
  [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md) § Milestone M.
- 🗺️ **Milestone N recorded** (2026-09-23): the option-space appendix, not a
  queue. Six more research passes — code intelligence, developer workflow, agent
  evaluation, security/injection defense, multimodal surfaces, and the
  local-first/tailnet cluster — are written up in full with roughly fifty
  candidate features, each with effort, trap and done-when, plus a consolidated
  do-not-build register. It is deliberately **unranked**: ranking is a separate
  pass, so the cut is visible and revisitable instead of implicit in planning.
  It also surfaced eleven verified facts about the current tree, three of which
  read as defects rather than missing features — repo-controlled `AGENTS.md`
  text sits in the position we instruct the model to obey exactly,
  `config.json` with plaintext keys is written world-readable, and attached
  images go to a provider undecoded and unresized. Appendix:
  [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md) § Milestone N.
- ⏸️ **Deliberately parked, not gaps to "fix"**: `AnthropicProvider` stays
  unreachable until an `anthropic_api_key` is a decision someone makes, and
  `crdt.rs` stays unwired (settled Milestone G deferral).
- 🔜 **Milestone L planned** (2026-09-23): *any machine you can SSH into, and an
  agent that finishes its own work.* Two tracks — the remote-backend track
  generalizes the Colab bridge into a `Backend` trait plus a BYO-SSH `xencode
  remote` command and hardware/OOM guards; the agent track ships a test/lint
  auto-repair loop, an `edit_file` failure fallback, and cost metering over the
  metrics that already exist. Research behind the shape (including the paid GPU
  clouds deliberately **not** integrated) is in
  [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md) § Milestone L.

## Backlog (A–K all shipped — tracked in [NEXT_PLAN_TASKS.md](NEXT_PLAN_TASKS.md))

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