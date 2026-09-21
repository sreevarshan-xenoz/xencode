# Xencode User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Rust CLI](#rust-cli)
5. [Rust TUI](#rust-tui)
6. [Command Reference](#command-reference)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

## Introduction

Xencode is an AI-powered development assistant that talks to local language
models through Ollama and llama.cpp (and, on opt-in, to cloud providers). It
provides an approval-gated agent loop over your workspace, pattern-based code
and security analysis, repository insights, a collaboration server and a
plugin system, with a focus on privacy and offline operation.

### Architecture
Xencode is a **Rust-only** workspace (`rust/crates/*`, 14 crates):
- **Rust core** — Primary CLI/TUI, server, code analysis, security scanning, plugin system, multi-provider routing (Ollama, llama.cpp, Gemini, Qwen, OpenRouter) with retry middleware, and collaboration sync

The Rust binary (`xencode`) is the entry point.

### Key Features
- **Rust TUI**: Ratatui-based terminal interface with 24 focus areas — 17 of them reachable from the `Ctrl+F` feature navigator
- **Multi-Provider AI Routing**: Ollama and llama.cpp locally + Gemini, Qwen and OpenRouter (any OpenAI-compatible model id) in the cloud, with status-code-driven retry middleware and a sequential `agent_fallback_models` chain
- **Code Analysis**: per-language heuristics (Python, JS/TS, Rust) + a pattern-based OWASP Top 10 scanner
- **HTTP/WebSocket Server**: Axum-based collaboration server with token auth, role-based access control, a JSONL audit trail and local-first bind defaults (TLS opt-in)
- **Plugin System**: Plugin trait, host, registry with lifecycle management
- **Conversation Memory & Cache**: Persistent session history + two-tier (memory and disk) cache with LRU eviction
- **Team Mode**: `xencode server` issues real bearer tokens (`POST /auth/login`), enforces Viewer/Editor/Admin roles on both HTTP and the WebSocket, and appends every action to an audit log

> **Not implemented, whatever a panel shows.** Four TUI panels are still
> interface mockups with scripted content: Voice Interface, Custom Models,
> Learning Mode and Multi-Language. They render and respond to keys; nothing
> listens. Three are real. The **Security Auditor** — `Enter` walks the
> workspace and runs the same `VulnerabilityScanner` behind `xencode analyze`,
> so an empty result means the scanner found nothing, not that the panel is a
> demo. The **Performance Profiler** — `Enter` samples this process's CPU and
> resident memory from `/proc`, reads the turn latency, llama.cpp timings and
> provider health the session already measured, and lists the last rows of
> `.xencode/metrics.jsonl`; anything with no data behind it shows `n/a` instead
> of a number. The **Terminal Assistant** — you type what you want to do, it
> makes one call to the configured model for candidate commands, and a command
> you pick runs through the agent's approval gate exactly as the model's own
> `run_command` would. None of the remaining four has a real backend
> yet — the closest working equivalents are the chat itself (a real model call)
> and `xencode advise` for repo insights. The performance dashboard, a separate
> panel, reports real session metrics.

## Installation

### Rust Binary

```bash
# Build from source (requires a stable Rust toolchain)
cd rust && cargo build --release -p xencode-cli
./target/release/xencode --help
```

### Prerequisites
- **Rust** (stable; the workspace uses `LazyLock`, so 1.80+) — for building from source
- **Ollama** installed and running (`ollama serve`)
- **A model** installed: `ollama pull qwen3:4b`
- **4GB+ RAM** recommended

## Getting Started

### Rust CLI

The Rust binary is the primary entry point:

```bash
# Verify installation
xencode --help

# Launch TUI (default experience)
xencode tui

# Scan workspace
xencode scan . --max-depth 2

# List models
xencode models list

# Analyze code for issues and vulnerabilities
xencode analyze src/
xencode analyze src/main.rs --format json

# Start the collaboration server
xencode server --port 8765

# Manage plugins
xencode plugin list
xencode plugin install ./my-plugin/
xencode plugin remove my-plugin

# Run a quick query
xencode query "Explain clean architecture"

# View conversation memory
xencode memory list

# Manage background tasks
xencode tasks start "cargo test" --name tests
xencode tasks list
xencode tasks poll 1 --lines 20

# Check cache stats
xencode cache stats

# Show config
xencode config show
```

### Rust TUI

Launch the interactive TUI with `xencode tui`:

```
┌─────────────────┬──────────────────────────────────────┬──────────────────┐
│  File Explorer   │          Code Editor                  │    Chat Panel    │
│    (Tab)        │                                       │    (Tab)        │
│                 │                                       │                  │
│ src/            │   // Edit your code here             │  You > Hello!    │
│   main.rs      │                                        │                  │
│   lib.rs       │                                        │  AI > Hi there!  │
│ tests/          │                                       │                  │
└─────────────────┴──────────────────────────────────────┴──────────────────┘

```

**TUI keyboard shortcuts:**

Global (work in every panel, any mode):

| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit |
| `Ctrl+G` | Refresh git status |
| `Ctrl+E` | File explorer |
| `Ctrl+S` | Save editor / Git commit panel |
| `Ctrl+R` | AI code review (current file) |
| `Ctrl+Y` | PR review dashboard |
| `Ctrl+K` | Background tasks panel |
| `Ctrl+O` | Worktree panel (list/add/remove git worktrees) |
| `Ctrl+L` | Insights panel (live refactor suggestions & warnings) |
| `Ctrl+F` | Feature Navigator (17 panels) |
| `Ctrl+D` | Performance dashboard |
| `Ctrl+P` | Project analyzer |
| `Ctrl+B` | ByteBot agent panel |
| `Ctrl+H` | Run provider health check |
| `Ctrl+T` | Toggle embedded terminal strip |
| `Ctrl+U` | Cycle body layout preset: classic → chat-first → zen |
| `Ctrl+W` | Close panel → chat |
| `Ctrl+,` | Settings |

Panel-independent (Normal mode):

| Key | Action |
|-----|--------|
| `?` / `F1` | Keybinding help overlay (lists the focused panel's keys) |
| `Tab` | Cycle explorer → editor → chat |
| `Esc` | Close popup / leave edit mode |
| `i` or `/` | Edit the chat input |
| `m` | Model selector (`r` refresh, `l`/`u` llama.cpp load/unload) |
| `s` | Settings (except in panels that bind `s` themselves: Security Auditor sort, Custom Models save) |
| `q` | Quit |
| `e` | Code editor: start typing edits |
| `↑ ↓` / `j k` | Scroll or move selection (all scrollable panels) |
| Mouse wheel | Scroll the panel under the cursor |

Chat editing (after `i`):

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Alt+Enter` / `Ctrl+J` | Insert newline (multiline prompts) |
| `Alt+↑` / `Alt+↓` | Recall previous / next sent prompt |
| `Tab` | Complete a `/` command, else insert 4 spaces |

Agent approval prompt (modal — while it is open these are the only keys that
do anything):

| Key | Action |
|-----|--------|
| `y` | Allow this call |
| `a` | Allow, and allow everything of this class for the rest of the session |
| `n` / `Esc` | Deny (the model is told it was denied and must not retry unchanged) |
| `k` / `j` | Scroll the diff or command preview |

**Feature Navigator panels (Ctrl+F → select → Enter):**

| Panel | Description |
|-------|-------------|
| Performance Dashboard | Session stats, file breakdown (real metrics) |
| Provider Health | Health checks with status icons (real requests) |
| Project Analyzer | Workspace file type analysis (real scan) |
| Git Commit | Type a message, Enter runs `git commit -am` (tracked modifications only — untracked files are never added); result returns as a chat line |
| ByteBot Agent | Step-through autonomous task execution (real tool loop) |
| Collaboration Hub | Real WebSocket client: create/join sessions, live members with roles, server errors verbatim |
| Voice Interface | 🎭 Mockup — a scripted session plays audio levels and transcript lines; no microphone is opened |
| Terminal Assistant | Real delegation: type what you want to do and `Enter` makes one call to the configured model, which answers with up to 8 `command · risk · why` rows built from the workspace it was told about. `f` filters by risk, `j`/`k` select, `y`/`Enter` runs the selection — through the agent's approval gate, so the same modal and policy as a model-issued `run_command`, and the outcome (a denial included) is listed in the panel's history. A provider error is printed rather than papered over |
| Security Auditor | Real scan: `Enter` walks the workspace with the context engine and runs `VulnerabilityScanner::scan_file` on every readable file; findings stream in with their CWE ids and a log line gives the true totals. Files the walker lists as secret are reported, never read |
| Performance Profiler | Real measurement: `Enter` reads this process's CPU (two `/proc/self/stat` samples 250 ms apart) and resident memory, then the session's own numbers — average turn latency, llama.cpp tokens/s, per-provider health, and the last 6 rows of `.xencode/metrics.jsonl` (KV reuse, prompt tokens, tok/s, retrieved files). A gauge with nothing to show reads `n/a` |
| Custom Models | 🎭 Mockup — sample profiles and sliders; saving writes nothing to config |
| Learning Mode | 🎭 Mockup — one hardcoded Rust ownership lesson |
| Multi-Language | 🎭 Mockup — a fixed sample language-detection table; nothing is detected |
| PR Review | Per-file diff browsing, base toggle (real git diff) |
| Background Tasks | Live registry of background commands (stop/remove) |
| Worktrees | Git worktree list, add (path+branch) and remove with confirm (real git) |
| Insights | Refactor findings from the live symbol graph (broken imports, cycles, hubs, orphans) |

**Collaboration Hub keys (while the panel is focused):**

| Key | Action |
|-----|--------|
| `c` | Create a new session and connect (the server assigns the id) |
| `j` | Edit the session id to join an existing session |
| `Enter` | Connect with the fields as shown / finish editing a field |
| `r` | Retry: hang up and dial the same server/session again |
| `Tab` | Cycle the edited field: server → user → session |
| `Esc` | Stop editing → disconnect (stops the client) → close the panel |

The hub talks to `xencode server` (see Collaboration Server): it logs in
over HTTP, authenticates the WebSocket with the first `auth` frame, and
renders only real state — transport (`ws (no TLS)` / `wss (TLS)`), session
id, live members with their roles, connected-for time, and the server's
last error. There is no auto-reconnect; `r` is manual by design.

### Layouts & Display

The body layout is a preset — pick it with `Ctrl+U` (live, with a toast) or
on the Settings panel (`s`). All choices persist to `~/.xencode/config.json`.

| Preset | Shape |
|--------|-------|
| `classic` | explorer 20 % / editor 50 % / chat 30 % (the default look) |
| `chat-first` | explorer hidden, editor 25 %, chat 75 % |
| `zen` | one pane fills the body — explorer or editor when focused, otherwise chat |

Display settings (Settings panel rows, same keys as `xencode config set`):

| Row / config key | Default | Effect |
|------------------|---------|--------|
| `Layout` / `layout` | `classic` | preset above; unknown values fall back to classic |
| `Rounded Borders` / `rounded_borders` | off | rounded panel corners |
| `Show Scrollbars` / `show_scrollbars` | on | vertical scrollbar on chat & explorer (panes ≥ 24 cols) |
| `Line Numbers` / `show_line_numbers` | on | editor gutter + current-line highlight (editor ≥ 45 cols) |
| `Agent Approval` / `agent_approval` | `ask` | how the chat agent may use its tools: `ask` prompts before mutating tools, `edit-allow` auto-approves file edits but still prompts for shell, `all-allow` auto-approves everything inside the workspace (paths outside it, `.git/` and the config dir are always refused) |
| `Command Timeout` / `agent_command_timeout` | 30 s | how long the agent's `run_command` may run before it is killed; the panel steps 5–300 s, `config set` accepts 1–600 |

The header reads ` ✦ xencode [layout] ⎇branch model` on the left with the
focused panel name on the right; on narrow terminals parts drop out in that
order (layout chip below 72 columns, branch/model below 60, badge below 40).
Tab cycles only the panes the current layout shows.

### Agent Tools & Approvals

While it is answering, the chat agent can call tools; each round is shown in
the transcript as `⚙ <tool> <arguments>`, and the result of every call is fed
back to the model before it continues.

| Tool | What it does | Class |
|------|--------------|-------|
| `read_file(path, offset?, limit?)` | Paged, line-numbered file text (200 lines by default) | read-only |
| `list_dir(path?)` | Directory listing, `/` marks directories | read-only |
| `search_files(pattern, path?)` | Regex search over the tree (skips `target/`, `node_modules/`, dot-dirs; 100 hits) | read-only |
| `repo_advise(filter?)` | Findings from the project index | read-only |
| `update_plan(items)` | Post or refresh the todo list the user watches | read-only |
| `background_poll(id)` / `background_stop(id)` | Output / cancel of a background task | read-only |
| `write_file(path, content)` | Create or replace a file (answers with the unified diff) | file change |
| `edit_file(path, old, new, all?)` | Exact string replace; refuses an ambiguous match unless `all` | file change |
| `run_command(command)` | `sh -c` in the project root, waits and returns the exit status plus output | shell command |
| `background_start(command, cwd?, name?)` | Start a shell command in the background (`Ctrl+K` panel) | shell command |

`file change` and `shell command` calls stop at the approval prompt described
in the key table above, unless `agent_approval` says otherwise. Paths are
relative to the project root; anything resolving outside it, anything under
`.git/`, and anything in the config directory is refused in every mode without
prompting — the transcript shows `⚙✗ … · refused: outside the workspace`. A
denial is reported to the model as an error it must not retry unchanged, so
the agent explains or re-plans instead of looping.

`agent_max_rounds` (default 16) caps how many tool rounds one turn may take;
after that the model is asked for a prose answer with no tools offered.

`run_command` is what closes the edit → test → fix loop: the prompt shows the
literal command line (never a diff, because there is no proposed file change to
show) and the answer comes back as `$ <command>`, `exit <code>` and the
combined stdout+stderr. Only the last 8 KiB of output is kept — when a build
fails the reason is at the end — and the cap is announced in the result.
`agent_command_timeout` (default 30 seconds, Settings row `Command Timeout`,
or `xencode config set agent_command_timeout 45`) kills a command that runs
too long; a killed command returns no output at all, and the model is told to
use `background_start` for anything that slow.

For multi-step work the model can post a todo list with `update_plan`, and it
appears above the chat transcript as a bordered `☰ Plan 2/5` strip: `✓` done
(struck through), `▶` the step in progress, `·` not started. Up to 12 steps are
kept, the compact strip shows the first 6 and points at `/plan` for the rest.
**`/plan`** pins the full list (or shrinks it again), **`/plan clear`** drops
it. Posting a plan is read-only — it never costs an approval, in any mode — and
a malformed update is refused with a message the model can act on, leaving the
plan already on screen untouched. A plan is a window into the agent's turn, not
a contract: weak local models may skip it entirely, and nothing is enforced
from it.

Every approved write or edit is snapshotted first, so **`/rewind [turns]`**
puts the files back: `/rewind` undoes the last turn that changed anything,
`/rewind 3` the last three. Files the agent created are deleted, files it
changed return byte-for-byte, and turns that wrote nothing are skipped
instead of counted. The snapshots live in memory only — quitting discards
them, and git is never touched, so your own commits remain the durable
history. `/rewind` refuses to run while the agent is still generating — a chat
turn or a ByteBot run — and
if the rewound file is open in the editor with unsaved edits, it warns rather
than throwing your work away. It covers the `write_file` / `edit_file` tools
only — a `run_command` or `background_start` that touched files behind our
back is not undone, which is why every snapshot refusal is stated in the
transcript instead of quietly claimed.

### ByteBot (delegated runs)

`/bytebot <task>` in the chat, or `Ctrl+B` and type it in the panel, hands the
task to the *same* tool loop the chat uses with the task as its only user turn:
the project's guidelines, git state and retrieved context, no chat history. It
is not a separate engine and it is not a demo —

- the **Execution Steps** list is the call log: one row per tool call the model
  actually made, `⏳` while it runs, then `✅ done`, `✗ denied` (you said no at
  the prompt), `✗ refused` (the policy blocked it) or `❌ failed`. A step never
  appears before the call exists.
- the progress bar is finished calls over calls made, so it can move
  backwards when the model starts another one; it is not a promise of work
  remaining.
- the approval gate is exactly your `agent_approval` setting: in `ask` mode a
  delegated run prompts at every edit and every command, which is the point —
  autonomy is a setting, not a hidden override.
- its writes are checkpointed like any other turn, so one `/rewind` puts the
  whole run's file changes back, and a plan it posts shows in the same strip.
- if the provider fails, the panel prints the error it got and the open step
  goes `failed`. Nothing is reported as done that xencode did not observe.

### Slash Commands (the only ones)

Type `/` in the chat input and press `Tab` to list them:

```
/init [abort|status]    Index the project (context snapshot under .xencode/)
/ctx <sub>              Context engine: status/track/compact/eval/kv/archive
/advise [filter]        Repository insights from that index
/bytebot <task>         Delegate the task to the agent loop
/plan [clear]           Pin the agent's todo list, or clear it
/rewind [turns]         Undo the agent's file writes for this session
/mcp [status|stop]      Start the MCP servers declared in config, or query them
/spawn <task> [#branch] Run a subagent in a fresh sibling git worktree
```

There is no `/help`, `/clear`, `/exit` or `/models`: press `?` (or `F1`) for
the keybinding overlay, `Ctrl+C` (or `q`) to quit, and `m` for the model
selector.

### First Run

There is no setup wizard. `xencode` opens the TUI against
`~/.xencode/config.json` (created with defaults on first save), reads
`default_model` and talks to the Ollama server at `ollama_url`. If Ollama is
not running or no model is installed, the model list is simply empty and chat
turns fail with the provider's error — `xencode models list` and `Ctrl+H`
tell you which. `install.sh` is the step that installs Ollama and pulls the
starter model; inside the TUI nothing is installed for you.

## Advanced Features

### Code Analysis

Analyze source code for style issues, bugs, and security vulnerabilities:

```bash
# Analyze a directory (recursive; junk dirs like target/ are skipped)
xencode analyze src/

# Analyze a single file with JSON output
xencode analyze src/main.rs --format json

# Image files take the intake path: format, dimensions and byte size
xencode analyze ./assets/logo.png --format json
```

### Security Scanning

The Rust analyzer includes OWASP-focused vulnerability scanning:
- Hardcoded secrets (passwords, API keys, tokens)
- SQL injection patterns
- Command injection risks
- Weak cryptography (MD5, SHA1)
- Path traversal vulnerabilities
- SSRF patterns

These are **regular expressions over source text**. There is no dataflow
analysis and no CVE database: the scanner never consults your dependency tree,
so a vulnerable library version is invisible to it.

### Collaboration Server

Start the team server — local-first by default:

```bash
xencode server                        # http://127.0.0.1:8765, ws://
xencode server --port 9000 --audit-path none
xencode server --host 0.0.0.0 --cert fullchain.pem --key privkey.pem   # https + wss
```

- Binds `127.0.0.1` unless `--host` says otherwise; a non-loopback plain
  bind refuses to start unless `--allow-insecure-public` is given (and
  warns loudly even then). TLS needs both `--cert` and `--key`.
- Clients obtain a token from `POST /auth/login` (the username is an
  identity claim — the bind surface is the perimeter) and authenticate
  each WebSocket with a first `auth` frame; the URL carries only the
  session id (`/ws/{session_id}`).
- Roles come from the workspace ledger: the session creator is Admin,
  joiners are Editor, and only Editor-or-above may relay activity on the
  WebSocket (Viewers receive). All mutating HTTP endpoints require the
  bearer token. Every action appends one JSONL line to
  `~/.xencode/audit.jsonl` (`--audit-path` to move, `none` to disable).
- Sessions live in memory only: after a server restart the peers are
  gone — only the audit log survives.

### Plugin System

Extend functionality with plugins:

```bash
# List installed plugins
xencode plugin list

# Install a plugin from a directory
xencode plugin install ./my-plugin/

# Remove a plugin
xencode plugin remove my-plugin
```

Plugins implement the `XencodePlugin` trait with lifecycle methods:
- `initialize` — Called when plugin is loaded
- `handle_event` — Process an event and return a response
- `shutdown` — Clean up resources

### Conversation Memory

```bash
# List all conversation sessions
xencode memory list

# Show specific session
xencode memory show <session-id>
```

### Cache Management

```bash
# Show cache stats (hits, misses, evictions)
xencode cache stats

# Clear all cached responses
xencode cache clear
```

### Background Tasks

```bash
# Start a task that survives this process (state in .xencode/tasks/)
xencode tasks start "cargo test" --name tests

# List tasks with derived status (running / exited(code) / killed)
xencode tasks list
xencode tasks list --json

# Show a task's status and trailing output
xencode tasks poll 1 --lines 20

# Stop a running task; remove a finished one (refused while running)
xencode tasks stop 1
xencode tasks rm 1
```

## Command Reference

### Rust CLI Commands

```
Usage: xencode [COMMAND]

Commands:
  scan      Scan a workspace and list all entries
  config    Configuration management
  models    Local model management (Ollama & llama.cpp)
  cache     Response cache management
  query     Send a query to a model
  memory    Manage conversation memory
  tasks     Manage background tasks (file-backed, survives this process)
  worktree  Manage git worktrees of the current repository
  advise    Repository insights from the .xencode snapshot: broken imports, import cycles, hub files and orphans
  server    Start the collaboration server
  analyze   Analyze code for issues and vulnerabilities
  fetch     Fetch a web page and extract research-ready text
  review    Review the diff between a base branch and HEAD, file by file
  plugin    Manage plugins
  llamacpp  llama.cpp server management (status/start/stop/load/unload)
  tui       Launch the Terminal User Interface
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

## Examples

### Example 1: Code Analysis
```bash
$ xencode analyze rust/crates/xencode-config-rs/src/config.rs
Analysis of: rust/crates/xencode-config-rs/src/config.rs
   Issues: 34
  [low] Ln8: Public item missing documentation -- Add /// doc comment explaining purpose and usage
  [medium] Ln409: Unwrap may cause panic on None/Err -- Use proper error handling with match or ? operator
  [medium] Ln421: Unwrap may cause panic on None/Err -- Use proper error handling with match or ? operator
  ...
```
Findings carry `issue_type`, `severity` (`Low`/`Medium`/`High`/`Critical`),
`file_path`, `line_number`, `message`, `suggestion` and `code_snippet` —
`--format json` emits them as an array of those objects.

### Example 2: Collaboration Server
```bash
# Start the server (local-first: binds 127.0.0.1, plain ws://)
$ xencode server
🚀 Xencode server starting on http://127.0.0.1:8765
   WebSocket: ws://127.0.0.1:8765/ws/{session_id}
   audit: ~/.xencode/audit.jsonl

# In another terminal, check health
$ curl http://localhost:8765/
{"status":"online","service":"Xencode Server","version":"0.1.0"}

# List models as the server sees them
$ curl http://localhost:8765/api/models
{"models":[{"name":"qwen3:4b","provider":"ollama","type":"local","size":...,"modified_at":"..."},
           {"name":"gpt-4o","provider":"openai","type":"remote"},
           {"name":"claude-3.5-sonnet","provider":"anthropic","type":"remote"}]}
```
`/api/models` lists what Ollama and llama.cpp actually report **and** appends
two hardcoded remote entries; when both local servers are offline it falls back
to two sample Ollama names. Treat it as a demo payload, not a model registry.
Routes: `/`, `/sessions/create`, `/sessions/{id}`, `/ws/{session_id}`,
`/auth/login`, `/auth/verify`, `/api/config`, `/api/models`, `/api/status`,
`/api/llamacpp/{status,load,unload}`.

### Example 3: Running the TUI
```bash
$ xencode tui

# The TUI opens with 3 panels: File Explorer | Code Editor | Chat
# Press Ctrl+F to open the Feature Navigator
# Select any panel with arrow keys + Enter
# Press Esc to close an overlay
# Press Tab to cycle focus between main panels
```

### Example 4: Plugin Management
```bash
$ xencode plugin list
No plugins installed in: ~/.local/share/xencode/plugins
Use 'xencode plugin install <path>' to install a plugin.

$ xencode plugin install ./my-custom-plugin/
✅ Plugin 'my-custom-plugin' installed successfully.

$ xencode plugin list
📦 Installed Plugins (from ~/.local/share/xencode/plugins):
  my-custom-plugin v1.0.0 — <manifest description> (by <manifest author>)
```
The listing reads the plugin **manifest**; the `XencodePlugin` trait, host and
event routing live in `xencode-plugin-rs` as a library — the CLI does not
instantiate or run plugin code.

### Example 5: Short Query
```bash
$ xencode query "What does this Rust code do?"
[Query runs against configured model...]
```

## Troubleshooting

### Common Issues

#### Ollama Not Running
**Problem:** "Cannot connect to Ollama service"
**Solution:** 
1. Start Ollama: `ollama serve`
2. Verify it's running: `curl http://localhost:11434/api/tags`

#### No Models Available
**Problem:** "No models found"
**Solution:**
1. Check available models: `ollama list`
2. Install a model: `ollama pull qwen3:4b`

#### Slow Responses
**Problem:** Long response times
**Solution:**
1. Check model health: `xencode models health <name>`
2. List installed models: `xencode models list`
3. Check system resources: `htop` or Task Manager

#### File Operation Errors
**Problem:** "Permission denied" or "File not found"
**Solution:**
1. Check file permissions
2. Verify file paths are correct
3. Ensure you have read/write permissions for the directory

### Performance Tips

1. **Use Caching**: Xencode caches responses to speed up repeated queries
2. **Choose Efficient Models**: Smaller models often respond faster
3. **Manage Memory**: Clear old sessions if experiencing slowdowns
4. **Optimize Prompts**: Clear, specific prompts yield faster responses

### Getting Help

- Press `?` (or `F1`) in the TUI for the keybinding overlay; `xencode --help`
  and `xencode <subcommand> --help` are the CLI reference
- `Ctrl+H` in the TUI runs a provider health check; `xencode models default`
  shows which model the picker chose
- Report issues on GitHub: https://github.com/sreevarshan-xenoz/xencode/issues
- Join discussions: https://github.com/sreevarshan-xenoz/xencode/discussions

## Best Practices

### Effective Prompting
- Be specific about what you need
- Provide context when relevant
- Break complex tasks into smaller queries
- Ask for explanations of code you don't understand

### Security
- Don't share sensitive information in prompts
- Verify code suggestions before running
- Keep your system updated
- Review file operations before confirming

### Productivity
- Use conversation memory to maintain context
- Leverage file operations for code generation
- Switch models based on task requirements
- Export important conversations for reference

---

For more information, visit the official documentation at https://github.com/sreevarshan-xenoz/xencode