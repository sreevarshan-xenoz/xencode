# 🚀 Xencode - Quick Start Guide

## Installation

```bash
# Already installed? Skip to Usage!

# If not, clone and run the installer:
git clone <your-repo>
cd xencode
./install.sh        # Linux/macOS — builds the Rust binary, checks Ollama
```

Windows (PowerShell): `.\install.ps1`

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

## Usage

### TUI (default experience)
```bash
xencode
```
Full-screen terminal UI: chat, file explorer (Space attaches files —
images ride as message parts the model actually sees, PDFs/DOCXs parse to
text), `/ctx` retrieval,
`/advise` repo insights, `/plan` for the agent's todo list, `/rewind` to undo agent edits, `/mcp` to bring up your MCP tool servers, `/spawn` to run a subagent in its own git worktree, model picker, and more.

### One-shot query
```bash
xencode query "what is recursion?"
```
Get an instant answer without entering the TUI.

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

## Commands

### In the TUI
```
/init [abort|status]  - Generate & control project docs
/ctx <sub>            - Context engine (status/track/compact/eval/kv/archive)
/advise [filter]      - Repository insights (cycles, hubs, orphans, broken imports)
/bytebot <task>       - Delegate a task to the autonomous agent
/plan [clear]         - Pin the agent's todo list, or clear it
/rewind [turns]       - Undo the agent's file changes for this session
/mcp [status|stop]    - Connect all MCP servers declared in config, or ask about/stop them
/spawn <task> [#branch] - Run a subagent in a fresh git worktree (status with /spawn status)
```
Those are the only slash commands. Tab completes them; typing a lone `/`
and pressing Tab lists them. Press `?` (or `F1`) any time for the full
keybinding help overlay. `Ctrl+U` cycles the body layout (classic →
chat-first → zen); the same choice lives on the Settings panel (`s`),
alongside Rounded Borders / Show Scrollbars / Line Numbers toggles —
everything persists to `~/.xencode/config.json`.

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
