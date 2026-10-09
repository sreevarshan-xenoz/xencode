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

Launches the immersive TUI, from a terminal — that is where it draws. Run it in a
pipe, a redirect or a CI step and it says so, and names the commands that need no
terminal at all. Run `/init` once per project for project-aware
answers, then just ask.

Local models need nothing else. For cloud models, `xencode config set
openai_api_key <key>` stores the key in the `api_keys` object of `config.json`
in the settings directory (`$XDG_CONFIG_HOME/xencode`, or `~/.xencode` before it
has been migrated) — the value is never printed back, and `config show`
says only where each credential came from. To keep a key out of that file, store
a reference to a program that prints it (`xencode config set qwen_api_key
"command:pass show xencode/qwen"`) or leave it unset and export `API_KEY_QWEN`.
There is no vault. Xencode writes the file `0600`, and any `config set` re-saves
it that way — if you created it by hand, `chmod 600` it yourself. Select a cloud
model with `xencode config set default_model …`, and see [CLI_GUIDE.md](CLI_GUIDE.md)
for the accepted model prefixes.

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
`/advise` repo insights, `/impact <file>` blast-radius fan-out panel, `/workers` for the fleet — the workers
this session launched, the roles your recipes name, the task registry, the recorded runs, the newest
events and the approvals waiting, each figure naming the row it was read from, `/plan` for the agent's todo list, `/rewind` to undo agent edits, `/lesson` to read the lesson a rewind, a run of failing checks or a refused call drafted and approve your own words into `AGENTS.md`, `/gate bugfix` to hold the agent off a production file until it has reproduced the bug, `/mcp` to bring up your MCP tool servers, `/plugin` to see which plugins took effect, `/skills` to see which `SKILL.md` skills loaded and what they cost the prompt, `/spawn` to run a subagent in its own git worktree, `/trace` to look back at what recent turns actually did, `/cost` for what those
turns add up to in tokens and money — naming which document each rate was read out
of, since `.xencode/pricing.json` is yours and a price `xencode prices fetch`
looked up is somebody else's list with a date on it — and, for an answer from a local model, a `⚡` line saying what the turn drew at the wall in watt-hours and what that costs at your own `$/kWh`, today's usage against any daily caps you set, model picker, and more. Ask about a library
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

### Fence the agent's shell off from your home
```bash
xencode config set run_command_sandbox true   # needs `bwrap` installed
```
Off by default. On, each `run_command`, `background_start` and shell hook runs in
a `bubblewrap` namespace: the workspace and `~/.cargo` stay writable, the rest of
the home (`~/.ssh` keys included) is replaced by an empty directory so it is gone
rather than hidden, and the network is off unless that one command passes `net`.
It is the guard against a command or a hook exfiltrating a file the approval gate
could not see. With `bwrap` missing the command is refused, never run
unsandboxed. A build that downloads a dependency will not run with the net off,
which is why enabling it is a choice, not the default.

### Let the agent read one page you have not chosen
```bash
xencode config set allow_web_fetch true
```
This is what offers the agent the `web_fetch` tool — the only one whose address
comes from the model. Off by default, and turning it on only offers the tool:
every call still stops at the approval prompt, which shows the address and
whether the fetch would even be allowed, and `a` (allow for the session) does not
apply here, because a yes about one page is not a yes about the next host. The
address is resolved and refused before connecting and again at each redirect, so
a private network or the cloud's metadata service stays unreachable even after a
`y`; `127.0.0.1` is allowed, so a local dev server is fetchable. Answers are text
capped at 30 000 characters. A page the server says is missing gets one more
request on the same address — its root `/llms.txt`, the index some documentation
sites publish for models — labelled as that index rather than the page, and most
sites have none, which is reported as a plain miss. The `web_fetch` section of
[CLI_GUIDE.md](CLI_GUIDE.md) shows the prompt, the refusal and a real answer.

### Give the agent a search engine to ask
```bash
xencode config set search_provider wikipedia   # or searxng, brave, tavily
```
`web_search` finds addresses; `web_fetch` reads one. Nothing is offered until you
name an engine, and the default is `none`. Wikipedia is the one keyless engine
that answers — about people, places and concepts, not the whole web. `searxng` is
an instance you run (`xencode config set search_searxng_url http://127.0.0.1:8888`),
and `brave` and `tavily` are a paid API behind `brave_api_key` / `tavily_api_key`,
each sent only to its own host. There is no default public instance because the
free options were checked and are not there: DuckDuckGo answers this machine with
its bot CAPTCHA or `410 Gone`, public SearXNG instances won't serve JSON, MDN's
JSON search endpoint is `404`. The question leaves the machine, so every call asks
— `a` (allow for the session) does not apply — and what comes back is titles,
links and the engine's snippets, none of them read. The `web_search` section of
[CLI_GUIDE.md](CLI_GUIDE.md) shows the prompt and the real answer.

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

### A project you have just cloned
```bash
xencode bootstrap --check   # report what is missing, create nothing
xencode bootstrap           # write it
```
A fresh clone has no `AGENTS.md`, no `.xencode/anchor.md` and no example settings
file, and the first two are read into every prompt. `bootstrap` writes all three
from what git and the directory itself report — no build, no test, no model call —
so it names no command and leaves the one blank a person has to fill in. A file
that already exists is never replaced; there is no `--force`. Afterwards run
`xencode anchor` to replace the anchor with build and test commands it actually
verified.

### Your own lines in `AGENTS.md` survive the budget
`AGENTS.md` reaches the model under a 1,200-token ceiling that keeps the front of
the file, so a long one used to lose its tail on every turn — and `## Lessons`,
where `/lesson approve` puts the sentence you wrote, *is* the tail. `## Lessons`
and `## Preferences` are now lifted out before that cut and ride a 300-token
budget of their own. To see what the head costs and whose words a turn is made of:
```bash
/ctx kv        # the head's size, its hash, and whether two turns agree on it
/egress        # the turn broken down by whose bytes it is, in tokens
```

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
`config.json` in the settings directory, so after it succeeds the VM is just
another provider:
select it in the TUI model picker (`m`) or watch it in Provider Health
(Ctrl+F → Provider Health). Colab gives a free-tier VM about 12 hours and wipes
its disk, so when the row goes red run `xencode colab up --reconnect` — it
reuses the VM if it lives and re-creates it if it was reaped, from the recorded
`colab.json` alone.

### Remote inference hosts (SSH)
Have your own machine with a GPU? Record host profiles and manage which one to use over SSH (`L-2`):

```bash
# Record a remote host profile (user@host[:port] or ~/.ssh/config alias)
xencode remote add lab dev@192.168.1.100:2222 --runtime llama.cpp --model qwen2.5:7b

# List recorded profiles and pick the active one
xencode remote list
xencode remote use lab
xencode remote show
```

## Commands

### In the TUI
```
/init [abort|status]  - Generate & control project docs
/ctx <sub>            - Context engine (status/track/compact/eval/kv/archive/fold/promote/drop/prompts)
/advise [filter]      - Repository insights (cycles, hubs, orphans, broken imports)
/impact <file>        - Blast radius of one file (crates · files · churn, in a fan-out panel)
/workers              - Worker panel: fleet, recipe roles, tasks, graph, costs, logs, approvals — every figure traces to a real row, and a worker xencode cannot observe reads as unknown, not idle (`Ctrl+A`)
/orchestrator <verb>  - The same state as a mode: `on`/`off` (`Ctrl+Space` flips the badge), `status`, `agents`, `tasks`, `graph`, `logs`, `costs`, `permissions`, `inspect`, `retry`, `stop`, `attach`. Leaving the mode leaves the session as it was found; `attach` hands this real terminal to a vendor's own running session and comes back with the exit status that process returned
/bytebot <task>       - Delegate a task to the autonomous agent
/plan [clear]         - Pin the agent's todo list, or clear it
/rewind [turns] [--force] - Undo the agent's file changes for this session (refuses files you edited by hand, unless --force)
/lesson [status|set <words>|approve|drop] - The lesson a rewind, three failing checks or a refused call drafted; the lesson line stays blank until you write it, and `/lesson approve` is the only thing that appends to AGENTS.md
/gate [bugfix [paths] | off] - Read, open or close the reproduction gate: with it open on a bug, the agent cannot touch a production file until `reproduce_bug` has been seen failing on unchanged code
/mcp [status|stop]    - Connect all MCP servers declared in config, or ask about/stop them (read a listed resource with `/mcp read <server> <uri>`, ask for a prompt with `/mcp prompt <server> <name>`)
/spawn <task> [#branch] - Run a subagent in a fresh git worktree (status with /spawn status); name its files with @path to lease them, so a second spawn on the same file waits
/plugin [reload]      - Show which plugins took effect, the prompt text and pinned commit each one contributes, or re-scan the plugin dir
/skills [reload]      - Show which SKILL.md skills loaded, what they refuse and what the prompt pays for them, or re-scan both skill directories
/trace [turns]        - Replay what the last agent turns did, from the local turn log
/cost                 - Tokens, KV-cache reuse, speed and spend for this project, naming where each price came from
/doctor [env|deps]    - Probe machine resources, GPUs, memory and environment facts
/verify [skip...]     - Run the machine-checkable checklist — fmt, lint, test
/hotspots [limit]     - Rank files by churn, size and bus factor
/agents               - Inventory the coding-agent CLIs installed on PATH
/trust [status|forget] [path] - Follow an AGENTS.md as instructions — the workspace's, or a directory's like src/auth/AGENTS.md — or report/withdraw (trust is per content hash)
/egress [text]         - Show where the next turn would send your prompt and what redaction holds back, without sending it
/goto <destination>    - Switch focus directly to any panel destination by name (also `/nav`); every feature stays reachable whatever disclosure level is set
/level [1-4]           - Read or set the progressive disclosure tier — 1 Core, 2 Workflow, 3 Advanced, 4 All — which governs what the feature navigator (`Ctrl+F`) and welcome line show. A first start opens at 2; the command palette (`Ctrl+X`) always lists everything
/help                  - Open the help overlay (the same as `?`). A `/word` that is not a command is answered with "Unknown command" instead of being sent to the model
```
Those are the only slash commands. Tab completes them; typing a lone `/`
and pressing Tab lists them. Press `?` (or `F1`) any time for the full
keybinding help overlay, or `Ctrl+X` for the command palette, which finds
any panel, command or setting by a few typed words. `Ctrl+U` cycles the body layout (classic →
chat-first → zen, then any layouts you declared in `layout_templates`);
`Alt+Left`/`Alt+Right` grows or shrinks the focused pane by five points, and a
divider between two side-by-side panes drags to do the same by hand — dragging
needs the mouse, and **Settings → `Mouse Capture`** gives it back to the
terminal when a plain drag should select text instead.
The same choice lives on the Settings panel (`s`) and the same list drives
both, alongside Rounded Borders / Show Scrollbars / Line Numbers toggles —
everything persists to `config.json` in the settings directory. A layout you
resized comes back next start from `layout.json` beside it (private, `0600`); `Ctrl+U`
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
↑ ↓ to pick an installed model reported by Ollama or llama.cpp; Enter sets it as default
If the list is empty, install/start a local model server or configure the cloud model directly.
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
