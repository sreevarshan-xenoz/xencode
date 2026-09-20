# 🤖 Xencode CLI Guide

The command-line interface for the Xencode AI assistant (Rust binary).
Running `xencode` with no subcommand launches the TUI.

## 🚀 Installation

```bash
# From source
git clone https://github.com/sreevarshan-xenoz/xencode
cd xencode
./install.sh        # Linux/macOS — builds + installs the binary

# Or build directly
cd rust && cargo build --release -p xencode-cli
cp target/release/xencode ~/.local/bin/
```

## 🎯 Quick Start

```bash
# Launch the TUI (default)
xencode

# One-shot query
xencode query "Explain clean code principles"

# Analyze a path
xencode analyze ./src

# Show version
xencode --version
```

## 📋 Command Reference

### `xencode` / `xencode tui`
Launch the immersive terminal UI (default when no subcommand is given).

TUI keys — press `?` (or `F1`) in the TUI for the live, panel-aware
keybinding overlay; the authoritative list lives there. Essentials:
`Tab` cycles explorer/editor/chat · `i` edits chat (`Enter` sends,
`Alt+Enter`/`Ctrl+J` newline, `Alt+↑/↓` history, `Tab` completes `/`
commands) · `m` model selector · `s` settings · `e` edit focused file ·
`Ctrl+R` AI review · `Ctrl+Y` PR review · `Ctrl+K` background tasks · `Ctrl+O` worktrees · `Ctrl+L` insights · `Ctrl+B` ByteBot ·
`Ctrl+H` health check · `Ctrl+G` git refresh · `Ctrl+W` close panel ·
`Ctrl+C` or `q` quit. Slash commands: `/init`, `/ctx`, `/advise`,
`/bytebot`.

### `xencode query <prompt>`
Send a one-shot query to the configured model.

```bash
xencode query "Explain microservices architecture"

# Pin a model, skip the cache, attach a session
xencode query "Explain async programming" \
  --model qwen3:4b \
  --no-cache \
  --session-id demo

# llama.cpp sampling controls
xencode query "Write a haiku" \
  --temperature 0.7 \
  --top-k 40 \
  --min-p 0.05 \
  --max-tokens 256
```

### `xencode analyze <path> [--format text|json]`
Analyze a file or directory for code issues and vulnerabilities. Image
files take the intake path: format, dimensions, and byte size are reported
(`--format json` returns the `ImageMeta` for a single image).

Directory mode walks the full tree (junk dirs like `target/` skipped),
analyzes every non-image file, and reports skip counts. Directory JSON is
a documented object — single-file shapes are unchanged:

```bash
xencode analyze ./src
xencode analyze ./assets/logo.png --format json
xencode analyze ./src --format json | jq '{issues: (.issues|length), images: (.images|length), skipped}'
```

### `xencode scan [path] [--hidden] [--max-depth N] [--format text|json]`
List workspace entries (kind, size, path) as TSV or JSON.

```bash
xencode scan . --max-depth 2
xencode scan . --format json | jq '.[].path'
```

### `xencode models <action>`
Local model management (Ollama & llama.cpp).

```bash
xencode models list       # All installed Ollama models
xencode models health <name>  # Check one model
xencode models default     # Show the smart-selected default
```

### `xencode llamacpp <action>`
llama.cpp server management: `status`, `start`, `stop`, `load`, `unload`.

```bash
xencode llamacpp status
xencode llamacpp start --model mymodel.gguf --port 8080
```

### `xencode config <action>`
Configuration management. Config lives in `~/.xencode/config.json`;
set `XCODE_CONFIG_DIR` to point Xencode at a different directory.

```bash
xencode config show
xencode config set default_model qwen3:4b
xencode config reset
```

`config set` keys (values are validated; `config show` prints the JSON):

| Key | Type | Notes |
|-----|------|-------|
| `default_model` | string | e.g. `qwen3:4b` |
| `ollama_url`, `llama_cpp_url` | string | provider endpoints |
| `llama_cpp_model_path`, `llama_cpp_executable` | string | llama.cpp paths |
| `llama_cpp_args` | string | split on whitespace |
| `max_cache_size`, `response_timeout`, `max_memory_items` | number | |
| `cache_enabled`, `memory_enabled` | bool | `true`/`false` |
| `layout` | string | TUI body preset: `classic`, `chat-first`, `zen` (unknown → classic at render) |
| `rounded_borders` | bool | rounded panel corners |
| `show_scrollbars` | bool | scrollbars on chat & explorer panes |
| `show_line_numbers` | bool | editor line-number gutter + current-line highlight |

### `xencode cache <action>`
Response cache management (`stats`, `clear`, …).

```bash
xencode cache stats
```

### `xencode memory <action>`
Conversation memory management (`list`, …).

```bash
xencode memory list
```

### `xencode tasks <action>`
File-backed background tasks. State lives in `.xencode/tasks/` under the
current directory, so tasks started here are visible to later `xencode
tasks` runs in the same project (the TUI's `Ctrl+K` panel keeps its own
in-process registry). Status is derived on read from the task's exit file,
killed flag, and process liveness.

```bash
xencode tasks start "cargo test" --name tests   # → "started task 1 (pid …)"
xencode tasks list                              # table; --json for machine output
xencode tasks poll 1 --lines 20                 # status + trailing stdout/stderr
xencode tasks stop 1                            # signal a running task
xencode tasks rm 1                              # forget a finished task (refuses running)
```

### `xencode worktree <action>`
Git worktrees of the repository at the current directory: `list`,
`add <path> [<branch>]` (existing branch or commit to check out; when
omitted git names the new branch after the directory), `remove <path>`
(dirty worktrees are refused by git itself, the main checkout is never
removable).

```bash
xencode worktree add ../feature-x        # new branch "feature-x"
xencode worktree add ../hotfix main      # check out existing branch
xencode worktree list
xencode worktree remove ../feature-x
```

### `xencode advise [FILTER] [--json] [--limit 40]`
Repository insights from the `.xencode` snapshot written by the TUI's
`/init`: broken imports, import cycles, hub files and orphans.
`FILTER` is a positional substring matched against each finding's file
path; `--limit 0` shows everything. Errors with exit 1 when the project
has no index yet.

```bash
xencode advise                      # top 40 findings
xencode advise src/auth             # only findings touching that path
xencode advise --json --limit 0     # full machine-readable report
```

### `xencode server [OPTIONS]`
Start the collaboration HTTP/WebSocket server. Sessions live in memory
(the audit log is the only thing that survives a restart); clients
authenticate on the WebSocket with a token from `POST /auth/login`, and
the first WS frame must be the `auth` frame — the URL carries no
identity (`/ws/{session_id}`).

```bash
xencode server                          # http://127.0.0.1:8765, ws://
xencode server --port 9000 --audit-path none
xencode server --host 0.0.0.0 --cert fullchain.pem --key privkey.pem   # https + wss
```

| Flag | Meaning |
|---|---|
| `--port <PORT>` | Listen port (default `8765`) |
| `--host <HOST>` | Bind address (default `127.0.0.1`; IP or `localhost`) |
| `--cert <PEM>` / `--key <PEM>` | TLS material — both or neither; enables `https://`/`wss://` |
| `--audit-path <PATH\|none>` | JSONL audit trail (default `~/.xencode/audit.jsonl`; `none` disables) |
| `--allow-insecure-public` | Escape hatch: bind a non-loopback address over plain ws:// |

Posture rules, enforced at startup: a non-loopback `--host` without TLS
refuses to start unless `--allow-insecure-public` is given (then it
binds with a loud clear-text warning); `--cert` without `--key` (or the
reverse) is an error; the banner prints the real scheme — `ws://` stays
`ws://`, only certificates earn `wss://`.

### `xencode plugin <action>`
Plugin management: `list`, `install <path>`, `remove <name>`.

```bash
xencode plugin list
```

### `xencode fetch <url> [--format text|json]`
Fetch a web page and extract research-ready text (title + body, scripts
and markup stripped). `--format json` returns the full `FetchedPage`.

```bash
xencode fetch https://example.com
xencode fetch https://example.com --format json | jq .title
```
Text output caps at 30k chars with a truncation trailer; `--format json`
returns the full body. Invalid `--json-schema` values are rejected up
front instead of degrading silently.

### `xencode review [--base main] [--format text|json]`
PR-level diff triage: files changed between the base and HEAD with line
counts, plus working-tree analysis per file (code issues, image
inventory). `--base HEAD` reviews uncommitted changes. Unanalyzable files
(deleted, binary) get visible notes, never silence.

```bash
xencode review --base main
xencode review --base HEAD --format json | jq '.files[] | {path, issues: (.issues|length)}'
```

## 🎯 Usage Examples

### Development Workflow
```bash
# 1. Index the project (inside the TUI)
xencode
# › /init

# 2. Check model health
xencode models list

# 3. Query for code help
xencode query "How do I parse JSON in Rust?"

# 4. Analyze before committing
xencode analyze ./src --format text
```

### Scripting
```bash
#!/bin/bash
# Ask and fail loudly on error
if ! xencode query "$1" --no-cache; then
    echo "❌ Query failed" >&2
    exit 1
fi
```

### JSON output for tooling
```bash
xencode analyze ./assets/logo.png --format json | jq .format
```
