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
`/bytebot`, `/plan` (pin or clear the agent's todo list),
`/rewind` (undo the agent's file changes for this session),
`/mcp` (connect every MCP server declared in config; `/mcp status`,
`/mcp stop`), `/plugin` (report which plugins loaded and what they changed;
`/plugin reload` re-scans the plugin directory), and
`/spawn <task> [#branch]` (run a subagent in a fresh
git worktree next to the project, e.g. `proj-spawn-1` on branch
`xencode/spawn-1`; a `#branch` suffix names the branch). The spawned
agent's live steps stream in the transcript, its final answer is posted
back with `(spawn #<id> · <task>)`, and `/spawn status` lists every
registered run with its worktree location. Your main chat keeps working
while the subagent works.

### `xencode query <prompt>`
Send a one-shot query to the configured model.

```bash
xencode query "Explain microservices architecture"

# Pin a model, skip the cache, attach a session
xencode query "Explain async programming" \
  --model qwen3:4b \
  --no-cache \
  --session demo

# llama.cpp sampling controls
xencode query "Write a haiku" \
  --temperature 0.7 \
  --top-k 40 \
  --min-p 0.05 \
  --max-tokens 256

# Structured output: --grammar takes a GBNF file or string, --json-schema a
# JSON schema. An invalid --json-schema is rejected up front (exit 1) rather
# than silently degrading to a plain completion.
xencode query "List three file formats" --json-schema '{"type":"object"}'
```
Sampling flags apply when the resolved model is served by llama.cpp; the
prompt alone (no flags) goes to the configured default model.

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
Local model management.

```bash
xencode models list           # Ollama models + models served by llama.cpp
xencode models health <name>  # Check one model's health
xencode models default        # Show the smart-selected default
```

### `xencode llamacpp <action>`
llama.cpp server management: `status`, `start`, `stop`, `load`, `unload`,
`list`, `set-path`.

```bash
xencode llamacpp status
xencode llamacpp start --model mymodel.gguf --port 8080 [--exec /path/to/llama-server]
xencode llamacpp set-path ~/models/mymodel.gguf   # persist the GGUF path
xencode llamacpp list                             # models on a running server
xencode llamacpp stop
```

### `xencode colab <action>`
Google Colab bridge: run the inference server on a Colab VM (T4 GPU etc.)
and reach it from this machine. The only supported transport is the official
`google-colab-cli` `colab ssh --proxy-mode` WebSocket SSH bridge — never a
public tunnel (Colab's free tier forbids ngrok/cloudflared-style tunnels and
suspends accounts that use them). The CLI is Linux/macOS only; Windows users
type a public-tunnel URL (paid tier) into Settings → Remote URL instead.

```bash
xencode config set colab_enabled true  # opt in — `up` refuses while the bridge is off
xencode colab preflight                # is the bridge usable? (exit 0 when green)
xencode colab preflight --generate-key # also create ~/.xencode/colab_ed25519 if missing
xencode colab up                       # create the VM, install the runtime, hold the tunnel
xencode colab up --reconnect           # rebuild a broken bridge from colab.json (one key)
xencode colab status                   # is the forward/session/endpoint alive?
xencode colab down                     # kill the forward, colab stop, clear state
```

`preflight` checks in one pass: the `colab` CLI on PATH, version >= 0.7.0
(0.6.0 shipped without the `ssh` subcommand), backend auth via
`colab sessions`, ssh/ssh-keygen on PATH, and the ed25519 key pair. Every
failing check prints a runnable fix line.

Prerequisites — the bridge rides two tools Xencode does not ship:

```bash
# 1. the official Google CLI (>= 0.7.0), however you install it, on PATH
colab --version
# 2. Google application-default credentials, with all four scopes the two
#    backends need — userinfo.email for the session backend and colaboratory
#    for the keep-alive RPC, or calls fail with 401/403 that look like a
#    permissions bug. gcloud refuses a list missing cloud-platform.
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```

`xencode colab preflight` is the source of truth for what is missing; run it
before blaming the tunnel.

`up` is the happy-path bring-up: `colab new --gpu <gpu> -s <name>` when the
session is absent, pushes an ssh bootstrap that installs the runtime bound to
`127.0.0.1` only *inside* the VM, holds an `ssh -N -l root -L` forward, waits
until `/v1/models` answers, and writes `~/.xencode/colab.json` — then points
the provider URLs at the forward (`llama_cpp_url`/`ollama_url` for the runtime,
`remote_base_url` for the OpenAI-compatible remote). The ssh user is `root`
because Colab injects the bridge key for root only. Flags override config;
`config colab_*` keys fill the rest:

```bash
xencode colab up                      # uses colab.session / colab.runtime / colab.model
xencode colab up --reconnect          # rebuild a broken bridge from colab.json (one key)
xencode colab up --runtime ollama     # tag flow into the model picker; respins the VM
xencode colab up --gpu L4 --model Qwen/Qwen2.5-7B-Instruct-GGUF
xencode colab up --local-port 18001   # laptop side of the forward
xencode colab up --remote-port 18080  # VM-side port (0 = runtime-native)
xencode colab up --weights hf         # llama.cpp weights from Hugging Face
xencode colab up --quant Q6_K         # which GGUF quant to serve (default Q4_K_M)
```

`up` refuses unless the bridge is switched on (`xencode config set
colab_enabled true`), and defaults the session name to `xencode-vm` and the
model to `Qwen/Qwen2.5-7B-Instruct-GGUF` when neither a flag nor config supplies
one.

`up` is patient by design: Colab gives a runtime exactly one SSH bridge and the
slot of a bridge that just died takes a while to free, so both the bootstrap and
the forward retry through that window (`Already-active SSH session` /
`banner exchange` errors, up to 8 attempts 20 s apart). `READY` from the VM
means the server actually serves — after the download it polls `/v1/models`
inside the VM before reporting, so a model that needs ~40 s to load on a T4
cannot look up-and-fail. The bootstrap budget is 40 minutes.

`up --reconnect` is the one-key repair path driven by `colab.json`: if the
forward's endpoint already answers `/v1/models` it returns immediately (no
colab or ssh calls at all — a dead forward pid is re-spawned and re-probed
before anything is re-fetched); otherwise it re-creates the session if the VM
was reaped server-side (never when the session still exists), re-runs the
bootstrap, and re-spawns the forward, then re-probes and rewrites state.
Without a `colab.json` it errors with a pointer to a full `xencode colab up`.

`runtime` chooses what is installed on the VM: `llama.cpp` (pinned prebuilt
llama.cpp release — CUDA build when `nvidia-smi` answers, the plain x64 build
otherwise — serving one GGUF fetched from Hugging Face) or `ollama`
(`ollama serve` + the pull — its tags then flow into the model picker for free
via the existing provider list). llama.cpp listens on `127.0.0.1:18080` by
default because Colab's own runtime proxy permanently holds `8080` on the VM.
`weights` is `hf` for llama.cpp; `drive`/`gcs` are refused with a fix message.
Session names are validated before they touch a shell (`[A-Za-z0-9_-]`,
1–64 chars).

`status` never fails hard — it reports three cells (forward pid alive,
session listed by `colab sessions`, and a `/v1/models` probe on the forward)
so it stays scriptable while fully degraded. When the VM is older than 12
hours and the endpoint is down it flags a likely Colab reaper and prints the
one-key fix (`xencode colab up --reconnect`). `down` is idempotent: kills the
recorded forward pid, runs `colab stop -s <name>`, and clears state; with no
`colab.json` it reports `nothing to tear down`.

Once `up` is green the tunnel is an ordinary provider — there is no Colab-specific
client path. `up` writes `llama_cpp_url` (or `ollama_url` for that runtime) and
`remote_base_url = <forward>/v1`, so the `remote:` prefix, the TUI model picker
and the Remote row in Provider Health (Ctrl+F) all speak through the same
forward:

```bash
xencode query -m 'remote:/root/xencode-llama/model.gguf' "Capital of France?"
curl -s http://127.0.0.1:18000/v1/models     # what the VM actually serves
```

llama.cpp reports the GGUF path it was started with as its model id, so on the
`llama.cpp` runtime that id is `$HOME/xencode-llama/model.gguf` inside the VM —
`/root/...` because Colab injects the bridge key for root. Read it from
`/v1/models` rather than assuming it.

### `xencode config <action>`
Configuration management. Config lives in `~/.xencode/config.json`;
set `XCODE_CONFIG_DIR` to point Xencode at a different directory.

```bash
xencode config show
xencode config set default_model qwen3:4b
xencode config set mcp_timeout 30
xencode config reset
```

`config set` keys (values are validated; `config show` prints the JSON):
`mcp_servers`, `agent_hooks` and `model_profiles` are nested structures, so they are edited directly in the JSON instead, or managed in the TUI where a panel exists for them.

| Key | Type | Notes |
|-----|------|-------|
| `default_model` | string | e.g. `qwen3:4b` |
| `ollama_url`, `llama_cpp_url` | string | provider endpoints |
| `remote_url`, `remote_key` | string | Remote / Colab endpoint (any OpenAI-compatible server, e.g. `http://127.0.0.1:18000/v1` + its bearer token); empty `remote_key` clears it |
| `colab_enabled` | bool | Gates the whole Colab bridge; `false` → `xencode colab *` refuses |
| `colab_session` | string | Session name for `xencode colab up`; empty = create one |
| `colab_runtime` | string | `llama.cpp` or `ollama` — what gets installed on the VM |
| `colab_model` | string | HF GGUF repo (llama.cpp) or ollama tag |
| `colab_quant` | string | GGUF quant to serve; empty = `Q4_K_M` |
| `colab_weights_source` | string | `hf` (llama.cpp); `drive`/`gcs` accepted but refused at bring-up |
| `colab_local_port`, `colab_remote_port` | number | Laptop side of the forward / VM-side port (`0` = runtime-native: llama.cpp `18080`, ollama `11434`) |
| `colab_auto_connect` | bool | Persisted but not acted on yet — nothing reconnects without an explicit `xencode colab up` |
| `llama_cpp_model_path`, `llama_cpp_executable` | string | llama.cpp paths |
| `llama_cpp_args` | string | split on whitespace |
| `max_cache_size`, `response_timeout`, `max_memory_items` | number | |
| `cache_enabled`, `memory_enabled` | bool | `true`/`false` |
| `layout` | string | TUI body preset: `classic`, `chat-first`, `zen` (unknown → classic at render) |
| `rounded_borders` | bool | rounded panel corners |
| `show_scrollbars` | bool | scrollbars on chat & explorer panes |
| `show_line_numbers` | bool | editor line-number gutter + current-line highlight |
| `agent_approval` | string | agent tool-approval mode: `ask`, `edit-allow`, `all-allow` (unknown → `ask`) |
| `agent_max_rounds` | integer | assistant→tool rounds allowed per chat turn before the model must answer in prose (`1`–`64`, default `16`) |
| `agent_command_timeout` | integer | seconds the agent's foreground `run_command` may take before it is killed (`1`–`600`, default `30`); slow work belongs in `background_start` |
| `agent_fallback_models` | list | comma-separated ordered alternates for the agent's turns (I4-01), e.g. `xencode config set agent_fallback_models "qwen2.5:14b,google_gemini:gemini-2.0-flash"`. The configured default model is always tried first, so this list holds only fallbacks (duplicates of it are dropped). A candidate is abandoned — and the chain moves on — only when it failed **before emitting any token** and the error is not our own response-decode failure; a token already on screen, or a `Parse` error, fixes the model in place. Each candidate gets one attempt per step and the transcript records a `[FALLBACK]` line when the chain moves. A candidate that would send the conversation somewhere the primary would not — a cloud API as the alternate for a local model, or the reverse — is never tried, and the transcript names it as skipped instead; a `remote:` endpoint counts as local only when its configured URL points at this machine (`localhost`, `127.x`, `::1`, `.local`). `xencode query` is single-shot and does not use this chain. An empty list (the default) disables fallback. |
| `mcp_timeout` | integer | seconds a server may take to handshake and answer before it is reported failed (`1`–`300`, default `30`) |
| `model_profiles` | list of objects | saved profiles the TUI's Custom Models panel (J-05) shows: `{ "name": "...", "model": "ollama:qwen2.5:7b", "temperature": 0.2, "max_tokens": 2048 }`. `temperature` and `max_tokens` are optional — omit them and the panel renders "unset — the server decides" and sends nothing. `model` takes exactly the form `default_model` does. `Enter` applies a profile to the next turn; `s` in the panel writes the whole list back here. There is no `top_p`: no provider path in this workspace sends it, and only llama.cpp receives these two knobs in the request body |
| `mcp_servers` | object | MCP stdio servers to offer as tools: `"name" → { "command": "...", "args": [...], "env": {...} }` (credentials go in `env`, never `args`); nothing is started until you run `/mcp` |
| `agent_hooks` | object | shell hooks around **approved** agent tool calls: `"before"` and `"after"` maps from an exact tool name (or `"*"` for every tool) to a command run via `sh -c` in the workspace root. A failing `before` hook vetoes the call (nothing runs, no rewind point, output shown as `error: pre-hook vetoed this call`); a passing one has its output prepended to the result. The `after` hook always runs and its output is appended. Hook output is capped like `run_command` (stderr merged, tail kept) |

Edit on `agent_hooks` directly in the JSON (`config set` has no nested-map key):

```json
{
  "agent_hooks": {
    "before": {
      "write_file": "git status --short",
      "run_command": "echo 'about to run a shell command'"
    },
    "after": {
      "*": "true"
    }
  }
}
```

### `xencode cache <action>`
Response cache management: `stats`, `clear`.

```bash
xencode cache stats
xencode cache clear
```

### `xencode memory <action>`
Conversation memory (persisted under `~/.xencode`): `list`, `show <session>`.

```bash
xencode memory list
xencode memory show <session-id>
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
The dependency graph behind them is built from `use` statements, `mod`
declarations and `impl Trait for Type` blocks, so a file an orphan report
names is one none of those reach in either direction.
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
`list`, `install <path>`, `remove <name>`. Plugins live in `$XCODE_PLUGIN_DIR`,
else `<data dir>/xencode/plugins` — the same directory the TUI loads from at
startup, so the two never disagree.

A plugin is a directory holding `plugin.json` (or `manifest.json`). This build
loads no executable plugin code: the manifest is the whole plugin, and the two
things it can declare are a `prompt_prefix` (placed ahead of the agent's system
prompt on every turn) and `hooks` — `before`/`after` maps of tool name (or `*`)
to an `sh -c` command, the same shape as `agent_hooks` in config.json. A
plugin's hook only lands where config.json is silent, so your own config always
outranks it.

```json
{
  "name": "guardrails",
  "version": "1.2.0",
  "prompt_prefix": "Run cargo test before answering.",
  "xencode_version": "*",
  "hooks": { "before": { "write_file": "echo pre" }, "after": { "*": "echo post" } }
}
```

`list` runs the load and reports what took hold instead of just listing
directories:

```console
$ xencode plugin list
📦 Plugins in /tmp/j08-probe/plugins (xencode 0.1.0):
  future v9.9.9 — NOT LOADED: needs xencode 0.1.0 (declared 9.9.9)
  guardrails v1.2.0 — loaded: prompt prefix, 1 before hook(s), 1 after hook(s)
  1 of 2 loaded — a loaded plugin's prompt prefix and hooks apply to every agent turn.
```

`install <path>` copies the directory (or a single manifest) under that name and
prints the same one-line verdict, so an install nothing can load is visible
immediately. `remove <name>` deletes only that one directory; a name containing
a path separator or `..` is rejected rather than resolved.

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
