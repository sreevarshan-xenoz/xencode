# The engine as its own process (DK-3) — design

Date: 2026-10-09. Status: awaiting owner review. Plan IDs: `EN-1` … `EN-4` (stages of `DK-3`)
and `RA-1` (a bug found while mapping).

## 1. What the owner asked for, and what was decided

Part 3 of the desktop-app design (`docs/superpowers/specs/2026-10-09-desktop-badge-and-bytebot-tasks-design.md`,
§3): xencode's engine moves out of the terminal app into a background process, so the terminal
app, the coming desktop app and the floating badge are windows onto one engine.

Decided with the owner on 2026-10-09:

- **Lifetime.** The engine runs while any window is connected *or* any task is working, and exits
  on its own when neither is true. Closing the terminal mid-task leaves ByteBot working, the badge
  showing it, and a new window attaches to it.
- **Connection.** A local socket: a named pipe on Windows, a socket file on macOS and Linux. No
  network port.
- **First-version scope.** All agent work: chat turns, ByteBot tasks and their queue, approvals,
  ByteBot questions, file snapshots and undo, conversation memory, and the badge's status file.
  Panels (file tree, git, model list, settings) stay in each window for now.
- The three design sections below were approved one by one in conversation.

## 2. Facts this design rests on (read from the code, 2026-10-09)

- `agent_rounds` (`xencode-tui-rs/src/app.rs`) takes an owned `AgentRun` and holds no reference to
  `App`. It talks back through three channels: a `String` token channel (about 25 kinds of token:
  raw text, `[TOOL]…`, `[BYTEBOT]…`, `[TURNERR]`, `[STOPPED]`/`[BYTEBOT_STOPPED]`, `[DONE]`, …), the
  approval channel (`ApprovalRequest` + a oneshot answer) and ByteBot's question channel. Shared
  state reaches it through `Arc`s: grants, `CheckpointStore`, plan, MCP hub, skills, taint, repro
  gate, approval rows, task runtime.
- The typed `AgentEvent` (`xencode-agents-rs/src/protocol.rs`) is serialisable but has one
  in-process publisher; the loop does not use it.
- `ConversationMemory` rewrites `conversation_memory.json` whole with no locking, so two processes
  writing it would race. `CheckpointStore` lives in memory; a git copy sits on
  `refs/heads/xencode/ckpt`.
- `xencode-server-rs` serves collaboration frames over a loopback WebSocket; it carries no agent
  traffic. No named pipe or Unix socket exists in the tree.
- `xencode run --detach` uses `fork` and is Unix-only (`detached.rs`); the badge launcher
  (`xencode_live_rs::spawn_badge`) is the only detached start that works on Windows.
- Plan ruling O-8 rejects an always-on daemon, an in-binary scheduler, and a second approval
  channel. This design keeps all three out: the engine exits when idle, schedules nothing, and holds
  the one approval queue.

## 3. The engine and its life

- **One engine per project folder** (the workspace root, canonicalised). Two windows on the same
  project share it; different projects get different engines. It runs as `xencode engine
  --project <root>`, a subcommand of the same binary.
- **Start.** A window connects to its project's socket; if nothing answers it starts the engine
  detached (the shared launcher from `EN-4`) and retries for up to 5 seconds, then says plainly
  that the engine could not be started and why.
- **Stop.** With no window connected and no task running, the engine exits after 10 seconds (so a
  window that is restarting can reconnect). A task waiting on a question or a review with no window
  connected waits up to 30 minutes; then a question is withdrawn and its task cancelled, and a
  review is completed with its changes kept (the same rule startup already applies); then the
  engine exits.
- **Crash.** A window that loses its engine says so in words and offers to start a new one. Task
  records on disk survive; a task that was running is failed by the existing startup rule.
- **One writer.** The engine is the only process that writes conversation memory, task records and
  the live status file for its project, which removes today's two-process race. Only one engine
  can own a project: it holds an exclusive lock file in `<state dir>/engine/<fingerprint>.lock`.

## 4. The connection and the messages

- **Socket name.** `<fingerprint>` is the first 16 hex characters of the SHA-256 of the canonical
  project path. Windows: `\\.\pipe\xencode-engine-<user>-<fingerprint>`. Unix:
  `<state dir>/engine/<fingerprint>.sock`.
- **Access.** On Windows the pipe is created with a security descriptor that grants access to the
  current user's SID only, and with remote clients rejected; the default descriptor would also let
  other local accounts read it. On Unix the socket file is created mode `0600` inside a `0700`
  folder.
- **Framing.** One JSON object per line. The first message each way is `hello` with a protocol
  version; a mismatch is answered with an error message naming both versions, then the connection
  closes.
- **Window → engine:** `submit_chat {prompt}`, `enqueue_task {text}`, `answer_question {id, text}`,
  `answer_approval {id, answer}`, `stop {target: chat|bytebot}`, `review {accept|undo}`,
  `set_model {name}`, `goodbye`.
- **Engine → window:** `snapshot` on connect (transcript tail, task list, pending approvals and
  questions, current model, running state), then `event` messages as things happen, plus
  `approval_requested {id, request}`, `approval_resolved {id, answer, by}`, `question_asked {id,
  text}`, `question_answered {id, by}`.
- **Approvals across windows.** The engine holds the single queue. Every window shows the prompt;
  the first answer wins; the others receive `approval_resolved` naming the window that answered.
  Questions work the same way.
- **First version carries today's tokens.** `event` wraps the loop's existing token strings
  unchanged, so the terminal app's token handling stays exactly as it is. Replacing them with typed
  `AgentEvent`s is a later, separate item.

## 5. Stages

| ID | Stage | Visible result |
|---|---|---|
| `EN-1` | New crate `xencode-engine-rs` owns all agent work, moved out of `app.rs`. The terminal app drives it through in-process channels carrying exactly the messages of §4. | None: same behaviour, existing tests unchanged |
| `EN-2` | The socket transport, `xencode engine`, start-on-demand and reconnect. The terminal app uses the socket by default; tests may run in-process. | Closing the terminal no longer stops a task |
| `EN-3` | Several windows per engine: snapshot on connect, shared approvals and questions, the lifetime rules of §3, and the engine writing the badge's status file. | A second terminal sees and can answer the first one's task |
| `EN-4` | A detached-process launcher that works on Windows, shared by the badge, the engine and `xencode run --detach`. | `xencode run --detach` works on Windows |

Typed events and the desktop app's shell (`DK-4`) follow these.

## 6. Testing and what counts as verified

- No mocks. Tests use real sockets and, for `EN-2` onward, a real `xencode engine` process.
- `EN-1`: the existing TUI suite passes unchanged; that is its proof.
- `EN-2`/`EN-3`: two clients racing to answer one approval (exactly one wins, the other is told);
  a client disconnecting mid-task and a new one attaching and seeing the snapshot; the engine
  exiting after the idle delay; the engine killed mid-task and the client saying so; a version
  mismatch refused in words.
- Windows pipe access: a connection from another local account must be refused. This needs a second
  account on the test machine; without one it stays marked unverified, per `AGENTS.md`.
- macOS and Linux socket behaviour is covered by CI; nobody will have watched it on a Mac.

## 7. Risks

- **Untangling `app.rs`** (about 21,500 lines) is the largest job. `EN-1` is a refactor with no
  visible change, held to the existing tests, and is done before any socket exists.
- **Approval UX with several windows** can surprise; the "answered in another window" message is
  part of `EN-3`'s done-when, not a polish item.
- **Windows named-pipe security** is easy to get wrong silently; the explicit descriptor is a
  requirement with its own test.

## 8. Found while mapping, not part of this work

- **`RA-1`:** `run_agent` (the headless entry point, `app.rs`) looks for `[SPAWN:0:finish:` but the
  loop sends `[SPAWN]0:finish:`, so `final_answer` is always empty; `rounds` is also hard-coded to
  1. A separate fix.
- `AF-2`'s "shipped" note says the engine publishes tool events on the bus; only `resolve_approval`
  does. Worth correcting in the plan when typed events are taken up.
