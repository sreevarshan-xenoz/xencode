# `xencode acp`: xencode as an agent inside Zed — design

Date: 2026-10-10. Status: awaiting owner review. Plan ID: `M-7`, built in stages `M-7a` … `M-7e`.

## 1. What the owner asked for, and what was decided

Said by the owner (2026-10-09 and 2026-10-10):

- A desktop app for Windows and Mac for people who prefer a GUI, with Zed as the inspiration.
- After weighing a Zed fork, the owner chose: "the acp is better and less work". xencode runs
  **inside stock Zed** as an external agent over the Agent Client Protocol. There is no fork.

Decided with the owner during this design:

- **No fork.** People install normal Zed and add xencode as an agent. xencode stays MIT; nothing
  of Zed is copied or redistributed.
- **`xencode acp` is a window onto the per-project engine** (EN-1 … EN-4), like the terminal app.
- **A new crate, `xencode-acp-rs`, using the official `agent-client-protocol` crate.**
- **Listing xencode in the ACP Registry is a later step**, not part of M-7.

Considered and set aside: a renamed Zed fork with xencode panels (GPL-3.0 for the app, a build
of ~249 crates, upkeep against Zed's daily changes), and an own GPUI app (DK-4/DK-5, still in the
plan, not started).

## 2. Facts this design rests on (checked 2026-10-10)

- **Protocol:** <https://github.com/agentclientprotocol/agent-client-protocol> (Apache-2.0,
  moved from `zed-industries`). Protocol version is the integer `1`; a draft v2 exists.
  JSON-RPC 2.0, one message per line, over the agent's standard input and output. The editor
  starts the agent as a child process. Standard output must carry only protocol messages;
  standard error may carry logs.
- **Rust crate:** `agent-client-protocol` 3.3.0, published 2026-10-09, Apache-2.0, edition
  2024, minimum Rust 1.88. The `stdio` feature must be turned on. tokio is not required by the
  crate; handlers must be `Send`, so it runs on our multi-threaded runtime. Agents are built with
  `Agent.builder()` and typed handlers (`on_receive_request`), connected with `Stdio::new()`.
  Handlers run inside the dispatch loop, so a long turn is moved off it with
  `connection.spawn(…)`. There is no `Agent` trait in 3.x.
- **Required methods:** `initialize`, `session/new`, `session/prompt`, `session/cancel` (a
  notification), and the agent's `session/update` notifications.
- **Model choice:** there is no `session/set_model`. An agent offers "config options" in the
  `session/new` response (`{id, name, category: "model", type: "select", currentValue,
  options}`); the editor changes one with `session/set_config_option`, and the agent answers
  with the full list. Whether Zed's model picker shows `category: "model"` options today is not
  verified; it is checked live in `M-7c`.
- **Permission:** `session/request_permission {sessionId, toolCall, options: [{optionId, name,
  kind}]}`, kinds `allow_once`, `allow_always`, `reject_once`, `reject_always`; the answer is
  `selected` with an option id, or `cancelled`.
- **Tool calls:** `toolCallId`, `title`, `kind` (`read`, `edit`, `delete`, `move`, `search`,
  `execute`, `think`, `fetch`, `switch_mode`, `other`), `status` (`pending`, `in_progress`,
  `completed`, `failed`), `content` (text, `diff {path, oldText?, newText}`, terminal),
  `locations [{path, line?}]`.
- **Slash commands:** the agent sends `available_commands_update` with `{name, description,
  input?}`; a command arrives as ordinary prompt text starting with `/`.
- **Zed setup:** `"agent_servers": {"xencode": {"type": "custom", "command": "xencode",
  "args": ["acp"], "env": {}}}` in Zed's settings; the agent appears in the Agent Panel's
  new-thread menu.
- **Registry** (later): <https://github.com/agentclientprotocol/registry>; a pull request adds
  `<id>/agent.json` (binaries per platform, licence, …) and a 16×16 monochrome `icon.svg`. Its
  CI checks that the agent returns at least one sign-in method from `initialize`.
- **This machine:** Zed is installed (`%LOCALAPPDATA%\Programs\Zed`), so the last stage is
  watched in real Zed on Windows 11.

## 3. Shape

`xencode acp` is a new subcommand. When Zed opens a session for a folder, the command connects
to that folder's engine, or starts one, the same way the terminal app does (`engine::link`), and
from then on it is one more window: it sends the engine `ClientMsg`s and draws nothing, turning
the engine's `EngineMsg`s and view changes into ACP updates instead.

What this gives Zed users without new work: a task goes on when Zed closes; Zed and a terminal on
the same project share one conversation; a prompt answered in either counts once (first answer
wins); the badge shows the session.

Code: a new crate `rust/crates/xencode-acp-rs`, depending on `agent-client-protocol` (pinned,
`stdio` feature) and on `xencode-tui-rs` for the engine link and protocol types. The CLI gains
`xencode acp`, which hands standard input and output to the crate. One process can hold several
ACP sessions; each session is bound to one project folder and one engine connection.

## 4. How ACP maps onto xencode

| ACP | xencode |
|---|---|
| `initialize` | Answers protocol version 1, `agentInfo` naming xencode and its version, prompt capabilities `image` and `embeddedContext`, and one sign-in method, "Use xencode's settings". `authenticate` with it checks that a model is configured and, if none is, fails with a message naming `xencode setup` and `xencode config`. |
| `session/new {cwd, mcpServers}` | Connects to or starts the engine for `cwd`. Returns a session id and a `model` config option listing the models xencode can use, current one selected. Sends `available_commands_update` with the commands that run in the engine (EN-2 ruling 2). `mcpServers` are accepted and not used in M-7; a line on standard error says so. |
| `session/set_config_option` (model) | `ClientMsg::SetModel`, the same path as `/model` (it also forgets the old model's measured context window). Answers with the full option list. |
| `session/prompt` | `ClientMsg::SubmitChat` with the text; images and embedded files go with it the way the terminal attaches them. Waits for the turn to end and answers `end_turn`; `cancelled` after a stop; `max_turn_requests` when the round cap ended the turn. |
| Streamed text and thinking | `agent_message_chunk` and `agent_thought_chunk`, made from the growth of the turn's last assistant message in the engine's view changes. |
| Tool calls | `tool_call` when a tool starts and `tool_call_update` as it runs and ends, with the ACP kind: `read_file`, `list_dir`, `read_docs` → `read`; `write_file`, `edit_file`, `edit_symbol`, `ast_edit` → `edit`; `search_files`, `find_refs`, `web_search` → `search`; `run_command`, `background_start` → `execute`; `web_fetch` → `fetch`; every other tool → `other`. An edit carries a `diff` with the file's text before and after, so Zed shows its own diff. |
| Approval prompt (`ApprovalRequested`) | `session/request_permission` with allow once, always allow and reject, answered with `ClientMsg::AnswerApproval`. If another window answered first (`ApprovalResolved`), the pending request is answered `cancelled`. |
| `ask_user` question (`QuestionAsked`) | Zed's question dialog through `elicitation/create` when the client declares it supports one; otherwise the question is sent as agent text and the next prompt is taken as the answer (`ClientMsg::AnswerQuestion`). |
| ByteBot | `/bytebot <task>` enqueues a task (`ClientMsg::EnqueueTask`). Its steps are sent as an ACP `plan`. Its review — keep or undo the changed files — is a permission request with "Keep" and "Undo", answered with `ClientMsg::Review`. |
| `session/cancel` | `ClientMsg::Stop` for the session's running work. |
| Files | xencode keeps editing files on disk itself and does not use the editor's `fs/*` methods in M-7. Zed reloads a file changed on disk and warns when it has unsaved edits. |

Not in M-7: `session/load` (old sessions), the editor's `terminal/*` methods, MCP servers passed
by the editor, and the Registry listing.

## 5. Errors

- The engine cannot be started or the connection is lost: the running prompt fails with a
  JSON-RPC error that says what happened in words; the next prompt tries to reconnect. Nothing
  waits forever.
- A client asking for a protocol version other than 1 is answered with 1, as the protocol
  allows; the client decides whether to go on.
- Standard output carries only protocol messages. Every log line goes to standard error, and
  `xencode acp` installs nothing that prints to standard output. A test checks that every line
  the agent writes parses as JSON-RPC.
- A request xencode does not support gets the crate's "method not found".

## 6. Testing and what counts as verified

No mocks, per `AGENTS.md`:

- Tests start the real `xencode acp` binary and drive it with the official crate's **client**
  side — a real second ACP client, which M-7's done-when allows. The engine behind it is the
  real engine.
- Failing model calls use an address nobody listens on, as the engine tests do.
- Chat, edits and approvals run against a **real llama.cpp server**; those tests are ignored
  unless `XENCODE_LIVE_LLAMACPP_URL` names one (the local Qwen3-4B, or a `colab up` endpoint).
- Each stage ends by watching it in the real Zed on this machine; `M-7e` watches all of it
  together.

## 7. Stages

| Stage | What | Done when |
|---|---|---|
| `M-7a` | Crate, `xencode acp`, `initialize`, sign-in method, `session/new` on the engine, prompt with streamed text, cancel | A real client gets streamed text from a real model, then `end_turn`; a cancel ends with `cancelled` |
| `M-7b` | Tool calls with kinds and diffs; approvals as permission requests | An approved edit writes the file and shows a diff; a rejected one writes nothing |
| `M-7c` | Model config option and the slash-command list | Choosing another model in the client changes the engine's model |
| `M-7d` | `ask_user` questions; ByteBot as plan updates and its review as Keep or Undo | A ByteBot task runs and its review is answered from the client |
| `M-7e` | `README.md`, `QUICK_START.md`, `CLI_GUIDE.md`, `docs/USER_MANUAL.md`, changelog, plan; watched in real Zed | A chat, an approved edit, a model switch and a ByteBot task are seen working in Zed on Windows |

## 8. Risks

- The protocol and crate move fast (83 crate versions; v2 in draft). The crate version is
  pinned, and M-7's own done-when already allows it to be blocked by upstream churn rather than
  half-shipped.
- Whether Zed shows a `model` config option as a model picker is not verified until `M-7c`. If it
  does not, the model is changed with `/model` and the gap is written down.
- The engine's view sends whole transcript lines, not token deltas. Streaming is made from the
  growth of the last message; if that turns out too coarse in Zed, the engine's `Event` stream is
  used instead.
