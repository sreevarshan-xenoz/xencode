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
plugin system, with a focus on privacy and local-first operation.

### Architecture
Xencode is a **Rust-only** workspace (`rust/crates/*`, 15 crates):
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

> **No panel is a mockup.** Seven TUI panels used to play hardcoded phrase lists;
> all seven now do work.
> The **Security Auditor** — `Enter` walks the
> workspace and runs the same `VulnerabilityScanner` behind `xencode analyze`,
> so an empty result means the scanner found nothing, not that the panel is a
> demo. The **Performance Profiler** — `Enter` samples this process's CPU and
> resident memory from `/proc`, reads the turn latency, llama.cpp timings and
> provider health the session already measured, and lists the last rows of
> `.xencode/cache/metrics.jsonl`; anything with no data behind it shows `n/a` instead
> of a number. The **Terminal Assistant** — you type what you want to do, it
> makes one call to the configured model for candidate commands, and a command
> you pick runs through the agent's approval gate exactly as the model's own
> `run_command` would. The **Multi-Language** panel — `Enter` (or `d`) walks the
> workspace with the same `scan_tree` the context engine uses and tabulates the
> languages that are actually present: file count, non-blank non-comment lines
> and share, with notes for what was skipped, listed as secret (counted, never
> read) or binary. Its list of language names is the scanner's own `Language`
> enum, and `Tab` + typing fill in From / To / Text before `Enter` makes one
> model call for the translation — a provider error is shown as an error rather
> than a fake answer. The **Custom Models** panel — the list is
> `model_profiles` from `config.json`, `Enter` applies a profile to the next
> turn, `s` is the only key that writes the file, and `t` shows the provider's
> own reply or its own error. The **Learning Mode** panel — `Enter` queues the
> files `.xencode/index/symbols.json` says declare something, shows the chosen
> file's own text and its recorded declarations, and asks the model once to
> teach that file and set a quiz with its answer key; `p`/`n` walk the queue.
> With no project index the panel says so and spends no request. The **Voice
> Interface** — `Enter` opens the microphone through the first of `arecord`,
> `pw-record` or `parec` found on `PATH`, and the meter, the peak and the clip
> length are computed from the PCM that came out of it; `Enter` again (or `Esc`)
> stops early and keeps what it has, and `m` mutes for real — the audio is
> discarded, not recorded quietly. The clip is written to
> `.xencode/voice/clip-<unix>.wav`. Words appear in the transcript only when a
> whisper CLI is installed and produced them; with none, the panel says so and
> names the clip. The closest working equivalents for everything else are the
> chat itself (a real model call)
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

# Rent a Colab GPU and serve a model through it
xencode config set colab_enabled true
xencode colab preflight --generate-key
xencode colab up
xencode colab status
xencode colab down

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
| Provider Health | Health checks with status icons (real requests): Ollama, llama.cpp and each keyed API get a row, and so does the Remote/Colab forward — it probes `remote_base_url`'s `/models` (the same endpoint `xencode colab status` waits on) and shows the forward URI under the row, while Connection Details lists the configured Remote URI or says it is not configured. With no remote URL set the row reads "Remote URL not configured (Settings → Remote URL)" instead of hiding |
| Project Analyzer | Workspace file type analysis (real scan) |
| Git Commit | Type a message, Enter runs `git commit -am` (tracked modifications only — untracked files are never added); result returns as a chat line |
| ByteBot Agent | Step-through autonomous task execution (real tool loop) |
| Collaboration Hub | Real WebSocket client: create/join sessions, live members with roles, server errors verbatim |
| Voice Interface | Real capture: `Enter` spawns the first recorder on `PATH` — `arecord`, `pw-record` or `parec` — streaming raw 16-bit mono 16 kHz, and the level bar, the peak and the clip length are RMS over the bytes it actually sent (one reading per 100 ms chunk). `Enter` again, or `Esc` while it is recording, ends the capture early and keeps the clip; `m`/`Space` mutes, which discards audio instead of saving a silent file. The clip is written to `.xencode/voice/clip-<unix>.wav`. The transcript stays empty unless a whisper CLI (`whisper`, `whisper-cpp`, `whisper-cli`) is installed — then its stdout is the text, and its failure is shown as its failure. With no engine the panel says so and names the clip it kept |
| Terminal Assistant | Real delegation: type what you want to do and `Enter` makes one call to the configured model, which answers with up to 8 `command · risk · why` rows built from the workspace it was told about. `f` filters by risk, `j`/`k` select, `y`/`Enter` runs the selection — through the agent's approval gate, so the same modal and policy as a model-issued `run_command`, and the outcome (a denial included) is listed in the panel's history. A provider error is printed rather than papered over |
| Security Auditor | Real scan: `Enter` walks the workspace with the context engine and runs `VulnerabilityScanner::scan_file` on every readable file; findings stream in with their CWE ids and a log line gives the true totals. Files the walker lists as secret are reported, never read |
| Performance Profiler | Real measurement: `Enter` reads this process's CPU (two `/proc/self/stat` samples 250 ms apart) and resident memory, then the session's own numbers — average turn latency, llama.cpp tokens/s, per-provider health, and the last 6 rows of `.xencode/metrics.jsonl` (KV reuse, prompt tokens, tok/s, retrieved files). A gauge with nothing to show reads `n/a` |
| Custom Models | Real profiles: the panel lists `model_profiles` from `config.json` (empty on a fresh install, which it says out loud rather than filling with samples). `n` adds one seeded from the current session settings, `-`/`+` moves temperature (0.0–2.0) and `←`/`→` steps the token budget along 64…8192; `Enter` applies it to the next turn — model id plus both knobs, and a llama.cpp profile also asks the server to load that model — without touching disk; `s` writes the whole list through `XencodeConfig::save`; `t` sends one request with exactly that profile's settings and prints the provider's real reply or its real error. Unset knobs render as "the server decides" and are sent as nothing |
| Learning Mode | Real lessons: `Enter` reads `.xencode/index/symbols.json` and queues the files that actually declare something — most declarations first, ties by path, five at a time. The panel shows that file's own text (capped at a line boundary, with a note saying how much was sent) and the declarations the index recorded, then makes one model call asking it to teach that file and reply with `{explain, question, options, answer, why}`; the model's sentences are labelled as its own. `p`/`n` walk the queue, `r` re-asks, `←→` + `Enter` answer the quiz, and grading uses the model's answer key — a reply with no usable key in it is printed as the reply, never replaced by a canned question. No project index → the panel says "run /init first" and sends nothing |
| Multi-Language | Real detection + real translation: `Enter`/`d` runs the context engine's `scan_tree` over the workspace and lists each language actually found with files, lines (blanks and comment leads excluded) and share, plus notes for skipped, secret-listed (never read) and binary files. The language list is `scanner::Language`, with `▸` marking what the walk found. `Tab` selects From / To / Text, typing edits it, `Enter` makes one model call and prints its reply — or the provider's own error |
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
/plugin [reload]        Report which plugins took effect / re-scan the dir
/spawn <task> [#branch] Run a subagent in a fresh sibling git worktree
/trace [turns]          What the recent agent turns did, from the local turn log
/cost                   Tokens, KV-cache reuse, speed and spend for this project
```

There is no `/help`, `/clear`, `/exit` or `/models`: press `?` (or `F1`) for
the keybinding overlay, `Ctrl+C` (or `q`) to quit, and `m` for the model
selector.

**`/trace [turns]` — looking back at what the agent actually did.** Each finished
agent turn appends one line to `.xencode/cache/turns.jsonl` inside the project:
how long it took, how many rounds the loop ran, the model and server that served
it, which tools it called and how each one ended, with the arguments each call
was made from, which workspace files were put in front of the model, and whether
the turn was marked as a decision. `/trace` prints the newest 50 of those lines
(fewer with `/trace 5`), starting with a total. It reads a local file, so it
answers even with every model server down. A marked turn prints as `#3 [d] …`,
the same marker that keeps its transcript entries through compaction; it comes
from `[d]` in the words you typed, never from anything a model said about its own
reasoning. Each line also says what that turn was shown to read
(`read for context: src/main.rs, notes.txt +1 more`), and a call that did not
finish prints the arguments it was given before the end of its output.

What it does not keep is deliberate: no prompt text (only a short digest of it)
and no full tool output — just a brief tail of each output, scrubbed of anything
that looks like a key, token or password before it is stored. Arguments are kept
only as far as they explain the call: a path, a pattern, a command line stays,
while the body of a file being written, the text an edit replaces and a plan's
steps are recorded as their size. The file holds every call's arguments;
`/trace` prints them for the calls that did not finish — denied, refused or
failed — because listing them all would push the interesting lines off the
screen. And because most local servers report
no token usage at all, the token column stays empty unless one did, and cost is
never estimated; `/trace` says which of those is the case rather than showing a
number it invented.

**`/cost` — what the recorded turns add up to.** Every turn that assembles a
context appends one row to `.xencode/cache/metrics.jsonl` in the project, and
those rows are folded into `.xencode/cache/metrics-rollup.json` so the answer
does not depend on how large the log has grown. `/cost` prints the fold: how many
records and sessions it covers and over what span, tokens prompted and generated,
the share of the prompt served from the KV cache, p50 and p95 generation and
prompt-evaluation speed over the newest 512 records that reported a rate, then
the breakdown per session (the current one marked with `→`) and per model. Like
`/trace` it reads local files and asks no model anything, so it answers with
every server down.

Money only exists if you say what things cost. Put a table in the project at
`.xencode/pricing.json`:

```json
{
  "models": {
    "qwen2.5:7b": { "input_usd_per_mtok": 0.0, "output_usd_per_mtok": 0.0 },
    "openai/gpt-4o": {
      "input_usd_per_mtok": 2.5,
      "output_usd_per_mtok": 10.0,
      "cached_input_usd_per_mtok": 1.25
    }
  }
}
```

Prices are dollars per million tokens, and editing that file changes the figures
on the next `/cost` — nothing is compiled in, and a price list update is a data
change. What the report refuses to do is guess: a model with no entry is listed
as `price unknown for N model`, a partly priced session reports `at least $x (no
price for N model)` rather than a total, a missing file says so, and a line that
did not parse is named in the report instead of being quietly dropped. Cache
reads priced separately are set with `cached_input_usd_per_mtok`; leave it out and
they are billed at the input price, which the report states.

Set `cost_budget_usd_micros` in `~/.xencode/config.json` (micro-dollars, so
`5000000` is $5.00) and the status bar carries this session's spend, refreshed
when each turn finishes — `💸 $0.42/$5.00` once every model it used has a price,
or `💸 13120 tok` while one does not — and crossing the budget prints one warning
in the transcript. The budget is a warning, not a stop: nothing refuses a request
over it.

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
  `~/.xencode/audit.jsonl` (`--audit-path` to move, `none` to disable). Each
  line is linked to the one before it, so `xencode audit verify` can say
  whether the file was edited afterwards.
- Sessions live in memory only: after a server restart the peers are
  gone — only the audit log survives.

### Google Colab GPU bridge

Rent a Colab VM and serve a model from it, without exposing anything: the VM's
OpenAI endpoint arrives on your laptop through the official `colab ssh` bridge,
and Xencode points the Remote provider at it.

```bash
xencode config set colab_enabled true
xencode colab preflight --generate-key
xencode colab up
xencode colab status
xencode colab down
```

- Prerequisites are the `colab` CLI (`google-colab-cli` >= 0.7.0, which is the
  first release with `colab ssh`) and `gcloud` application-default
  credentials; `preflight` checks both plus `ssh`, and prints a runnable fix
  line per failing check. It creates `~/.xencode/colab_ed25519` with
  `--generate-key`.
- `up` creates the session if absent, installs llama.cpp (CUDA build when the
  VM reports a GPU) or Ollama, serves one GGUF from Hugging Face on
  `127.0.0.1:18080` inside the VM, and holds a forward at
  `http://127.0.0.1:18000/v1`. It reports ready only once `/v1/models`
  actually answers.
- After that the VM is an ordinary provider: pick it with `m` in the TUI,
  address it as `remote:<served-model-id>` (`/v1/models` on the forward lists
  the ids), and watch the Remote row in Provider Health (Ctrl+F).
- Free-tier VMs last about 12 hours and their disk is wiped. `status` says
  when the recorded VM is old enough to have been reaped; `xencode colab up
  --reconnect` rebuilds from `~/.xencode/colab.json` — reusing the forward if
  the endpoint still answers, re-creating the VM if not.
- `down` kills the forward, runs `colab stop` and clears state. Run it: an
  unreleased VM keeps consuming compute units.
- No public URL exists by design — Colab's terms forbid tunnel brokers on the
  free tier, and the forward keeps the endpoint bound to loopback.

### Plugin System

A plugin is a directory holding `plugin.json` (or `manifest.json`) in
`$XCODE_PLUGIN_DIR`, else `<data dir>/xencode/plugins` — the same directory the
TUI loads at startup.

```bash
# Report each plugin and whether it actually loads
xencode plugin list

# Install a plugin from a directory (prints the same verdict)
xencode plugin install ./my-plugin/

# Remove a plugin
xencode plugin remove my-plugin
```

This build loads no executable plugin code, so a manifest *is* the plugin, and
it can declare exactly two effects, both applied to every agent turn:

- `prompt_prefix` — text placed ahead of the agent's system prompt.
- `hooks` — `before` and `after` maps of tool name (or `*`) to an `sh -c`
  command, the same shape as `agent_hooks` in `config.json`, run around
  approval-gated tool calls. Where both declare the same tool, **config.json
  wins**.

`xencode_version` (default `*`) pins the versions a plugin accepts; one that
does not match is reported as `NOT LOADED` rather than skipped silently. Inside
the TUI, `/plugin` prints the same report and `/plugin reload` re-scans the
directory after an install.

The `XencodePlugin` trait, host and event routing live in `xencode-plugin-rs`
as a library; `ManifestPlugin` is the implementation the runtime registers for
each manifest, and it answers no events — there is no plugin code to answer
them.

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

### Audit Log Integrity

```bash
# Check the session server's audit log for records changed after writing
xencode audit verify

# Check a log somewhere else
xencode audit verify /path/to/audit.jsonl
```

Every record the server appends carries a digest of its own contents and the
digest of the record before it, so editing, deleting or moving a line is
reported on a specific line and the command exits non-zero. A record written
before this was in place is counted and named as unprovable rather than
silently passed over. Truncating the end of the log is not something the file
can detect on its own, and neither is a full rewrite that recomputes every
digest.

### Scripted Queries

```bash
# One JSON event per line, read as it arrives
xencode query "Summarise this crate" --format ndjson | jq -j 'select(.type == "token") | .text'
```

`-j`, not `-r`: `jq -r` appends a newline after every piece it prints, and an
answer that arrived in 47 pieces gains 47 line breaks it never had.

`xencode query` writes plain words by default. `--format ndjson` writes a
`start` line (model, client, whether the prompt stayed on this machine,
conversation id), a `token` line per piece of the answer as it arrives, and one
closing `done` or `error` line. Every line carries `"v": 1`, and the token lines
always add up to the answer the `done` line reports — including when the reply
came from the response cache. `CLI_GUIDE.md` documents each field, the version
rules, and the one shell trap that eats line breaks in a naive consumer.

### Replaying a recorded run

```bash
xencode config set session_recording true    # then run an agent turn in the TUI
xencode replay --list                        # what has been recorded, newest first
xencode replay 1790240197                    # an id, or enough of it to be unique
xencode replay 1790240197 --run-tools        # and let the recorded commands run again
```

With `session_recording` on, each model call of an agent turn is written to
`.xencode/cache/sessions/<run-id>.jsonl` as it happens — the request, the response
bytes exactly as they arrived, and what each tool returned. `xencode replay` serves
those bytes again on a loopback port while the real agent loop, the real stream
reader, the real permission gate and the real tools run against them, then writes
`tool_calls.jsonl` naming each call, its arguments, its outcome and a digest of its
result. No model answers a replay, so it needs neither a server nor a provider
account, and two replays of one recording write the same file down to the byte,
because every time in it comes from the recording rather than the clock.

The gate is not bypassed: without `--run-tools` a call the recording shows needed
approval comes back `denied`, and the report says which model call it stopped at
and exits non-zero. Only the routes whose bytes this program reads itself are
recordable — Ollama, llama.cpp, a `remote:` endpoint and OpenRouter — and asking
for a recording of an Anthropic, Gemini or Qwen model is refused with the reason.
`CLI_GUIDE.md` documents the flags and what the ledger holds.

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
  audit     The session server's audit log
  query     Send a query to a model
  memory    Manage conversation memory
  tasks     Manage background tasks (file-backed, survives this process)
  worktree  Manage git worktrees of the current repository
  colab     Google Colab bridge: preflight, then up / status / down for a model server running on a Colab VM
  advise    Repository insights from the .xencode snapshot: broken imports, import cycles, hub files and orphans
  server    Start the collaboration server
  analyze   Analyze code for issues and vulnerabilities
  fetch     Fetch a web page and extract research-ready text
  review    Review the diff between a base branch and HEAD, file by file
  replay    Run a recorded session again from the bytes it was made of
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
```console
$ xencode plugin install ./guardrails/
✅ Plugin 'guardrails' installed to /home/me/.local/share/xencode/plugins/guardrails.
   guardrails v1.2.0 — loaded: prompt prefix, 1 before hook(s), 1 after hook(s)

$ xencode plugin list
📦 Plugins in /home/me/.local/share/xencode/plugins (xencode 0.1.0):
  future v9.9.9 — NOT LOADED: needs xencode 0.1.0 (declared 9.9.9)
  guardrails v1.2.0 — loaded: prompt prefix, 1 before hook(s), 1 after hook(s)
  1 of 2 loaded — a loaded plugin's prompt prefix and hooks apply to every agent turn.

$ xencode plugin remove ../../etc
error: Invalid plugin name: ../../etc
```
Every line above is real output. An empty plugin directory says
`No plugins installed in: <dir>` and how to install one.

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