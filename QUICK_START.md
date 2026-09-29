# 🚀 Xencode - Quick Start Guide

## Installation

```bash
# Already installed? Skip to Usage!

# If not, clone and run the installer:
git clone https://github.com/sreevarshan-xenoz/xencode
cd xencode
./install.sh        # Linux/macOS — builds the Rust binary, checks Ollama
```

Windows (PowerShell): `.\install.ps1`

### Prebuilt binaries (from the first tagged release on)

Releases are built by cargo-dist for five targets with checksums, plus shell
and PowerShell installers — nothing here exists until a `v*` tag is pushed:

```bash
cargo binstall xencode-cli
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/sreevarshan-xenoz/xencode/releases/latest/download/xencode-cli-installer.sh | sh
```

**What happens:**
1. ✅ Checks Rust toolchain, curl, git
2. ✅ Builds the release binary (`cargo build --release -p xencode-cli`)
3. ✅ Checks Ollama (installs + starts it if missing)
4. ✅ Pulls the starter model (`qwen3:4b`)
5. ✅ Smoke-tests the binary and installs the `xencode` command

## First Run

```bash
xencode
```

Launches the immersive TUI. Run `/init` once per project for project-aware
answers, then just ask.

Local models need nothing else. For cloud models, put the keys in the
`api_keys` object of `~/.xencode/config.json` (`xencode config show` prints
it; the file is plain JSON, there is no vault) and select a cloud model with
`xencode config set default_model …`. Xencode writes that file `0600`, and any
`config set` re-saves it that way — if you created it by hand, `chmod 600` it
yourself. See [CLI_GUIDE.md](CLI_GUIDE.md) for the accepted model prefixes.

## Usage

### TUI (default experience)
```bash
xencode
```
Full-screen terminal UI: chat, file explorer (Space attaches files —
images ride as message parts the model actually sees, shrunk first to a
1568-pixel long edge and recompressed to JPEG when they have no transparency,
PDFs/DOCXs parse to
text), `/ctx` retrieval,
`/advise` repo insights, `/plan` for the agent's todo list, `/rewind` to undo agent edits, `/mcp` to bring up your MCP tool servers, `/plugin` to see which plugins took effect, `/spawn` to run a subagent in its own git worktree, `/trace` to look back at what recent turns actually did, `/cost` for what those
turns add up to in tokens and money, model picker, and more. Ask about a library
and the agent can read the upstream source of the exact version your
`Cargo.lock` pins, already on disk — address it as `crate:serde/src/de.rs`.
Those reads are read-only, and the answer names the version it came from.
`read_docs` is the same idea for prose: it answers with the readme a crate points
at itself, plus a list of the other documents it ships. `lookup_advisory` answers
the security question — what is known to be wrong with this crate, judged against
the version your lock file pins — from the advisory corpora on disk, which you
download once:

```bash
xencode advisories sync                 # the one command that needs network here
xencode advisories check --path rust    # offline, over every locked package
```

```text
$ xencode advisories check --path rust
/home/sree/Projects/xencode/rust/Cargo.lock — 419 locked packages; 4 of them are named by 5 advisory record(s):
  lru 0.12.5
    RUSTSEC-2026-0002 [rustsec] 2026-01-07 — affected, 0.16.3 is offered as safe
      https://github.com/jeromefroe/lru-rs/pull/224
    RUSTSEC-2026-0253 [rustsec] 2026-05-12 — affected, 0.18.2 is offered as safe
      https://github.com/jeromefroe/lru-rs/pull/238
  …
```

### One-shot query
```bash
xencode query "what is recursion?"
```
Get an instant answer without entering the TUI.

### Replay a recorded run
```bash
xencode config set session_recording true   # then run an agent turn in the TUI
xencode replay --list                       # what has been recorded, newest first
xencode replay 1790240197                   # the shortest prefix that is unique
```
Turn on `session_recording` and every model call of an agent turn is written down
as it happened — the request, the response bytes as they arrived, and what each
tool returned. `xencode replay` serves those bytes again on a loopback port while
the real agent loop, the real stream reader, the real permission gate and the real
tools run against them, and writes `tool_calls.jsonl` saying which call came back
the same and which did not. No model answers a replay, so it needs no server and
no provider account. What is recorded is the TUI's agent turns: `xencode query`
writes no recording. The tools stay gated: a call the recording says needed
approval comes back `denied` unless you pass `--run-tools`. Ollama, llama.cpp, a
`remote:` endpoint and OpenRouter can be recorded; models served by Anthropic,
Gemini or Qwen cannot, and the command says so rather than writing a paraphrase.

### Ask what this machine can serve
```bash
xencode hw probe
```
Memory, cores, the compute devices `llama-server` itself reports — not the ones
PCI pretends to hold — what the model file's cache costs per token, and the flags
to start a local server with. It prints the `config set` line that keeps them and
changes nothing.

### Time this repository's git history
```bash
xencode history status
xencode history setup      # write the commit-graph and multi-pack-index, then re-time
```
`status` times the history queries — commit subjects, the paths each commit
touched, the reachable-commit count, a blame — from `git` processes that just ran,
and reports whether the two indexes that make them cheap are present. `setup`
writes them and measures again. Expect it to report no speed-up: on an 813-commit
repository only the commit count moved (2.7 ms → 2.0 ms), and the command says so
rather than inventing a win.

### Analyze a path
```bash
xencode analyze ./src
```
Code issues + security findings, plus an inventory of any images found.

### Repository insights
```bash
xencode advise            # needs the /init snapshot from the TUI
xencode advise src/auth --json
```
Broken imports, import cycles, hub files and orphans, read straight from
the `.xencode` index.

### Team collaboration
```bash
xencode server           # local-first: http://127.0.0.1:8765, ws://
```
Then in the TUI open the Collaboration Hub (Ctrl+F → Collaboration Hub):
`c` creates and connects a session, `j` edits in a session id to join one,
`Tab` cycles the server/user/session fields, `r` retries, `Esc` hangs up.
The panel shows only the real connection state — transport, session id,
live members with roles. Flags, tokens, TLS and the audit log are in
`CLI_GUIDE.md`.

### Rented GPU (Google Colab)
No GPU locally? Rent one for the length of a session and Xencode talks to it
over an SSH tunnel. Requires the official `google-colab-cli` (>= 0.7.0) on PATH
and Google application-default credentials — both are checked for you, and
`CLI_GUIDE.md` → `xencode colab <action>` has the setup commands.

```bash
xencode config set colab_enabled true    # the bridge is opt-in; up refuses without it
xencode colab preflight --generate-key   # CLI version, auth, ssh key — fix lines for anything missing
xencode colab up                         # T4 + llama.cpp + Qwen/Qwen2.5-7B-Instruct-GGUF by default
xencode query -m 'remote:/root/xencode-llama/model.gguf' "Capital of France?"
xencode colab status                     # forward, session, endpoint + a 12h-reap warning
xencode colab down                       # ALWAYS run this: an unstopped VM keeps billing
```

`up` writes `remote_base_url` (and the matching local runtime URL) into
`~/.xencode/config.json`, so after it succeeds the VM is just another provider:
select it in the TUI model picker (`m`) or watch it in Provider Health
(Ctrl+F → Provider Health). Colab gives a free-tier VM about 12 hours and wipes
its disk, so when the row goes red run `xencode colab up --reconnect` — it
reuses the VM if it lives and re-creates it if it was reaped, from the recorded
`colab.json` alone.

## Commands

### In the TUI
```
/init [abort|status]  - Generate & control project docs
/ctx <sub>            - Context engine (status/track/compact/eval/kv/archive/prompts)
/advise [filter]      - Repository insights (cycles, hubs, orphans, broken imports)
/bytebot <task>       - Delegate a task to the autonomous agent
/plan [clear]         - Pin the agent's todo list, or clear it
/rewind [turns]       - Undo the agent's file changes for this session
/mcp [status|stop]    - Connect all MCP servers declared in config, or ask about/stop them
/spawn <task> [#branch] - Run a subagent in a fresh git worktree (status with /spawn status)
/plugin [reload]      - Show which plugins took effect, or re-scan the plugin dir
/trace [turns]        - Replay what the last agent turns did, from the local turn log
/cost                 - Tokens, KV-cache reuse, speed and spend for this project
```
Those are the only slash commands. Tab completes them; typing a lone `/`
and pressing Tab lists them. Press `?` (or `F1`) any time for the full
keybinding help overlay. `Ctrl+U` cycles the body layout (classic →
chat-first → zen, then any layouts you declared in `layout_templates`);
`Alt+Left`/`Alt+Right` grows or shrinks the focused pane by five points, and a
divider between two side-by-side panes drags to do the same by hand — dragging
needs the mouse, and **Settings → `Mouse Capture`** gives it back to the
terminal when a plain drag should select text instead.
The same choice lives on the Settings panel (`s`) and the same list drives
both, alongside Rounded Borders / Show Scrollbars / Line Numbers toggles —
everything persists to `~/.xencode/config.json`. A layout you resized comes
back next start from `~/.xencode/layout.json` (private, `0600`); `Ctrl+U`
clears it, and the file never stores your conversation or model state — only
where the panes were. `Ctrl+0` lists every one of those changes this session has
been through, each named by what caused it, with the pane widths before and
after on `Enter`; it is session memory, and reading it moves nothing.

`Ctrl+1`…`Ctrl+9` recall a *view* — a saved arrangement with the pane it was
focused on. Six slots arrive filled: `Code`, `Chat`, `Terminal`, `Focus`,
`Review`, `Split`; `Ctrl+Shift+<digit>` stores the screen you are looking at
into any of the nine, including the empty `7`–`9`. The view you were on comes
back next start. Stored views are `layout_views` in `config.json`; `Ctrl+U`
leaves a view without renaming it, and nothing here is a gate — every panel a
view shows is still reachable on its own keys.

## Examples

### Example 1: Basic Chat
```bash
$ xencode

You › what is recursion?
Xencode › [streams answer in real-time]

Ctrl+C (or q) quits.
```

### Example 2: Switch Models
```bash
$ xencode

Press m              # opens the model selector (refreshes the list)
↑ ↓ to pick qwen2.5:7b, Enter to set it as default (r re-refreshes)
You › explain async/await
Xencode › [uses the new model]
```

### Example 3: Project Context
```bash
$ cd /path/to/your/project
$ xencode

You › /init
[Indexes the project]

You › how can I improve this code?
Xencode › [includes project context in response]
```

## Tips

1. **Maximize terminal** for best experience
2. **Install multiple models** for flexibility
3. **Press Ctrl+H** to run a provider health check
4. **Run `/init`** in each project directory for auto-context
5. **Press `?`** for the keybinding help overlay; `m` opens the model selector

## Troubleshooting

### Ollama Not Running
```bash
ollama serve
# or
systemctl start ollama
```

### No Models
```bash
ollama pull qwen3:4b
```

### Slow Responses
Press `m` in the TUI, select a smaller model (e.g. `phi3:mini`) and hit
Enter — the selection is saved as the default.

## That's It!

**You're ready to use Xencode!** 🎉

```bash
xencode
```

**Your immersive AI assistant awaits!** 🤖✨
