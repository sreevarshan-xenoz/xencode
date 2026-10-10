# The team orchestrator: one lead agent, several worker agents, one project — design

Date: 2026-10-10. Status: awaiting owner review. Plan IDs: `TM-1` … `TM-6`.

## 1. What the owner asked for, and what was decided

Said by the owner (2026-10-10): "they can set claude as their orchestrator and control other
models like gemini agy codex and all and work on a same project".

Decided with the owner during this design:

- **First user:** a solo developer, with their own subscriptions or API keys and/or local
  models.
- **The person picks the lead.** Any agent that can use an MCP server — Claude Code, Codex,
  Gemini CLI, Antigravity — leads, and directs the others through tools xencode provides.
- **Each worker works in its own git worktree.**
- **The lead merges a worker's result when the checks pass**, without asking the person.
- **Sign-in:** API keys by default; a subscription login only when the person turns it on for
  that agent, after xencode shows that vendor's terms line.
- **First workers:** Claude Code, Codex, Gemini CLI, Antigravity, and xencode itself (local or
  configured models).
- **Watching:** a Team section in the terminal app's worker panel, plus the badge.
- **Approach A:** the per-project engine runs the workers as ACP clients; the lead reaches them
  through new tools on `xencode mcp serve`; a merge lands only after the checks pass on the
  merged result.
- **The person, not the lead, answers the workers' permission prompts** (or gives a worker
  trust).

Set aside: xencode's own agent as the only lead (the owner wants to choose the lead), and
one-shot command-line workers (`claude -p`, `codex exec`: no follow-ups, and their approvals
happen inside each vendor's tool where xencode can neither see nor answer them).

## 2. Facts this design rests on (checked 2026-10-10)

**Agents** (ACP Registry index, 41 agents, and each project's docs):

| Agent | ACP | Start command | API-key sign-in | Login | Terms on third-party use of a login |
|---|---|---|---|---|---|
| Claude Code | adapter (Anthropic/Zed/JetBrains), registry `claude-acp` 0.89.1 | `npx -y @agentclientprotocol/claude-agent-acp@0.89.1` | `ANTHROPIC_API_KEY` | `/login`, `CLAUDE_CODE_OAUTH_TOKEN` | Anthropic "does not permit third-party developers to … route requests through Free, Pro, or Max plan credentials on behalf of their users" (code.claude.com/docs/en/legal-and-compliance); the Agent SDK page says the same unless approved. Unclear, leaning no, for the adapter on a login. |
| Codex | adapter (OpenAI/JetBrains/Zed), registry `codex-acp` 2.2.2 | `npx -y @agentclientprotocol/codex-acp@2.2.2` | `CODEX_API_KEY` / `OPENAI_API_KEY` | ChatGPT login offered by the adapter | No restriction found; the contract text could not be read (403). |
| Gemini CLI | built in, registry `gemini` | `gemini --acp` | `GEMINI_API_KEY` | cached Google login | "Directly accessing the services powering Gemini CLI … using third-party software" is a violation (geminicli.com/docs/resources/tos-privacy). Unclear for driving the real binary. |
| Antigravity | Google's own server, registry `antigravity-acp` 1.3.0 | `agy_acp_server` (`.exe` on Windows, `.par` elsewhere; Linux adds `--uid=`) | `GEMINI_API_KEY` **and** `"modelProvider": "gemini"` in `~/.gemini/antigravity-cli/settings.json` | Google login | "Using third party software … to access the Service (e.g. using OpenClaw with Antigravity OAuth)" can suspend the account (antigravity.google/terms §6). Its docs never mention ACP. |

All four can use an MCP server as a client: `claude mcp add`, `codex mcp add`, `gemini mcp add`,
Antigravity's `mcp_config.json`.

**The official Rust crate** `agent-client-protocol` 3.3.0 (already pinned by `M-7`) has a
client side that starts an agent as a child process (`AcpAgent::new(AcpAgentConfig::new(cmd)
.args(..).env(..))`, feature `process`) and ends its whole process tree. Its built-in
`claude_agent()`/`codex()` use `@latest`; this design pins the registry versions instead.

**What exists in xencode** (mapped 2026-10-10):

- The roster (`xencode-agents-rs/src/roster.rs`) knows ten vendor agents, but only runs them as
  one-shot probes (`xencode interop`). Nothing delegates real work to a vendor agent.
- `/spawn` (`xencode-tui-rs/src/app.rs`, `spawn_worktree`) makes a sibling worktree on
  `xencode/spawn-N` and runs xencode's own agent there; `@path` claims a file through
  `.xencode/leases.json`. It is typed by a person, cannot be cancelled mid-run, and does not merge.
- `xencode merge land` (`xencode-analysis-rs/src/merge_decision.rs`) merges into the base, then
  runs the test commands, and only reports a failure: the merge is not undone.
- `xencode mcp serve` offers six file tools and no orchestration tool.
- The worker panel (`worker_panel.rs`) and the badge (`xencode-live-rs`) exist; nothing feeds
  them vendor workers. Per-worker cost is not recorded anywhere.
- `xencode acp` (`M-7`) makes xencode itself an ACP agent, so it can be a worker too.

## 3. Shape

- **`xencode-team-rs`** (new crate): the worker runtime. It starts each worker as an ACP client
  in its own worktree, sends it the task as a prompt, and keeps what the worker sends — text,
  tool calls, plan, changed files, usage — plus its state.
- **It runs inside the per-project engine** (EN-1 … EN-4). Workers belong to the project, outlive
  the lead's terminal and every window, and their permission prompts are engine approvals:
  first answer from any window counts.
- **`xencode mcp serve --team`** adds the orchestration tools to the existing MCP server. Each
  call goes to the engine over the same local socket the terminal uses. Setup for a lead, e.g.
  `claude mcp add xencode -- xencode mcp serve --team`.

## 4. The lead's tools

| Tool | Does |
|---|---|
| `team_agents` | The workers this machine can start: installed or not, signed in by key or by an opted-in login, and which |
| `team_start {agent, task, base?}` | New worktree off `base` (default: the current branch), starts the agent on the task; returns a worker id |
| `team_status {id?}` | One worker or all: state (working, needs you, done, failed, stopped), last message, files changed, tokens and cost so far |
| `team_result {id}` | The worker's final answer and its diff against `base` |
| `team_message {id, text}` | A follow-up prompt in that worker's session |
| `team_stop {id}` | Stops it; its worktree is kept |
| `team_merge {id}` | The checked merge (§6) |

The lead never answers a worker's permission prompt: the person does, in the Team panel or any
window, or gives a worker trust (`team_start {…, trust: "edits"}` is refused unless the person
set it in `team.toml`). A model does not approve another model's commands.

## 5. Worktrees, sign-in, sessions, cost

- **Worktrees:** `<repo>-team/<id>` on branch `xencode/team/<id>`, made with `/spawn`'s worktree
  code. A stop keeps it; a landed merge removes it and its branch; `xencode team clean` removes
  stopped workers' worktrees. A file named with `@path` in the task is claimed first, as `/spawn`
  does.
- **Sign-in:** per the table in §2. Keys come from xencode's secret store (as `xencode config`
  uses), never the repo. A login is used only after the person runs `xencode team login-optin
  <agent>`, which prints that vendor's terms line and asks for a yes. An agent that cannot sign
  in is shown so by `team_agents`, and `team_start` refuses with the fix in words.
- **Versions:** the adapters are pinned to the registry's versions; changing one is a code
  change.
- **Sessions:** one ACP session per worker, cwd its worktree. `team_message` continues it. A
  first start that downloads an adapter says so instead of looking stuck.
- **Cost:** per worker, tokens from the agent's usage updates and a price from the price table
  for API-key work; a login is shown as "on your plan, not priced", never as $0.
  `team_status`, the Team panel and `xencode orchestrator costs` show it.
- **Limits:** at most 4 workers running at once by default (`team.toml`), refused in words past
  it. The Local Only posture refuses outside agents as it does today; xencode workers stay
  allowed.

## 6. The checked merge (`team_merge`)

1. A scratch worktree at the base branch's current commit; the worker's branch is merged there.
2. A conflict stops it: the conflicting files are returned, for the lead to resolve (message
   the worker, or start a fresh one).
3. The checks run in the scratch worktree: the project's test and build commands, found as
   `xencode verify` finds them, or the `checks` list in `.xencode/team.toml`.
4. Green, and the base has not moved: the base moves forward to that merge commit. The person's
   working copy is never touched before the result is green. If the base moved, back to 1.
5. **No checks found:** no automatic merge; the person is asked (a merge that lands without
   checks would land on nothing).
6. Every attempt, landed or refused, goes to the audit log with the check output. A landed
   merge is undone with `git revert` or `/rewind`.

## 7. The Team panel and the badge

A **Team** section in the existing worker panel (`/workers`): one row per worker — agent, task,
state, cost, last line. Enter opens its transcript; `s` stops it; `m` sends a message; `a`
answers its waiting approval. A worker waiting for the person makes the badge show "needs you"
(`LiveSource` gains `Worker`).

## 8. Errors

| Problem | What happens |
|---|---|
| The agent's command is not installed | `team_agents` marks it missing; `team_start` refuses with the install command |
| The worker crashes | `failed`, with its last error; worktree kept |
| The engine restarts | workers whose process is gone are marked `stopped`; worktrees kept |
| Past the worker limit | refused in words |
| The base branch has uncommitted changes | `team_merge` refuses: the person's working copy is not merged over |
| A check never ends | the check's timeout (`team.toml`, default 20 minutes) fails it |

## 9. Testing and what counts as verified

No mocks (`AGENTS.md`):

- Real worktrees and a real git repository; real check commands; a real `xencode acp` worker
  with a model address nobody listens on, for start, status, stop, message, result and merge.
- The merge is tested end to end with real commits: green checks land, red checks do not, a
  conflict is reported, a moved base is retried, no checks means asking.
- Each outside agent is tested live only when the person has set that vendor's API key and asks
  for it, because it costs money; those tests are `#[ignore]` with that reason.
- Runs that use the local GPU are started only after asking the owner, every time.
- The last stage watches a real lead (Claude Code with an API key, if the owner provides one)
  start two workers, follow them and land one merge.

## 10. Stages

| Stage | What | Done when |
|---|---|---|
| `TM-1` | Worker runtime in the engine: an ACP worker started in its own worktree, its session recorded, stop and message | a real `xencode acp` worker starts, reports, is messaged and stopped, in a worktree |
| `TM-2` | `xencode mcp serve --team` and the seven tools | an MCP client starts a worker and reads its status and result through the tools |
| `TM-3` | The checked merge | green lands, red does not, conflicts and moved bases handled, no checks means asking |
| `TM-4` | Team panel and the badge | the panel shows workers and answers a waiting approval |
| `TM-5` | Sign-in, login opt-in, cost; the four outside agents | each outside agent starts with an API key when one is set (live, on request) |
| `TM-6` | Manuals | README, Quick Start, CLI guide, user manual, changelog, plan |

## 11. Risks

- Vendors' terms on logins may change, or be read differently; keys stay the default and the
  opt-in shows the terms line each time it is turned on.
- Adapter versions move fast; they are pinned and updated deliberately.
- Antigravity's ACP server is undocumented outside the registry; it is the most likely to break,
  and is last in `TM-5`.
- Automatic merging is only as safe as the project's checks; a project without checks falls back
  to asking.
