# EN-2 — the engine in its own process — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `xencode engine` runs the agent work for one project in its own process, reached over a local socket; the terminal app starts it on demand, works as a window onto it, and closing the terminal no longer stops a running task.

**Architecture:** The EN-1 messages travel as JSON lines over a named pipe (Windows) or a socket file (Unix), restricted to the user's own account. The engine process holds a headless `App` and runs `engine::handle` and `engine::pump` for every connected window. Each window receives *view* messages — the agent state it draws, sent as changes — and applies them to its own `App`; it no longer runs agent work itself.

**Tech Stack:** Rust 1.99, tokio (named pipes, Unix sockets, mpsc), serde, windows-sys (security descriptors), sha2.

**Spec:** `docs/superpowers/specs/2026-10-09-engine-process-design.md` (§3 life, §4 connection, §5 EN-2).

## Global Constraints

- No mocks; real sockets and a real `xencode engine` process in integration tests (`AGENTS.md`).
- Socket: `\\.\pipe\xencode-engine-<user>-<fp>` on Windows, `<state dir>/engine/<fp>.sock` (mode `0600`, folder `0700`) elsewhere; `<fp>` = first 16 hex of SHA-256 of the canonical project path. Windows pipe: DACL granting only the current user's SID, remote clients rejected.
- One engine per project: exclusive lock `<state dir>/engine/<fp>.lock`.
- An engine with no window for 10 s and nothing running and no question waiting exits. (The 30-minute rule for waiting questions and reviews is EN-3.)
- A window connecting gets `hello` then a full view (the snapshot). Protocol version stays 1 plus the new `view` and `note` messages.
- Plan-item commits, plain English, never push; `XCODE_CONFIG_DIR` set for every test run; `-j 4`; leftover-process check; `cargo clean` above 20 GiB.

## Rulings carried from writing this plan (gaps in the design)

1. **How a window stays in step.** The design said windows hold the screen and the engine holds agent work, but not how the screen learns agent state. The engine sends `view` messages: changes to the transcript (new or changed lines from an index on), `is_generating`, ByteBot's running flag, steps, log, progress and tasks, the approval prompts, the waiting question, the plan strip, the model, the LLM call count and the llama.cpp timings — exactly the agent fields `ui.rs` reads. A window applies them; it does not apply the loop's `event` tokens itself (they stay on the wire for the badge and the desktop app).
2. **Where slash commands run.** Commands that change agent state run in the engine: plain chat, `/bytebot`, `/spawn`, `/rewind`, `/gate`, `/plan`, `/lesson`, `/ctx`, `/model`, `/trust`, `/mcp`, `/plugin`, `/skills`. Every other command changes only the window and runs there. Lines a window-side command adds to the transcript are sent to the engine as `note` messages, so every window's transcript is the same.
3. **The detached launcher is generalised now** (`xencode_live_rs::spawn_detached`), because starting the engine needs it; EN-4 then uses it for `xencode run --detach` on Windows.
4. **In-process mode stays** for tests and as a fallback: `xencode tui --in-process`, or automatically (with a warning) when the engine cannot be started.

## Review Focus

1. **Two windows typing at once** — the engine serialises messages; neither window's message is lost. Test in Task 4 (`two_clients_both_reach_the_engine`).
2. **A window that disconnects mid-stream** — the engine keeps running the task and does not panic on the dead writer. Test in Task 4 (`a_task_outlives_its_window`).
3. **A stale socket file or lock from a crashed engine** — a new engine starts. Test in Task 4 (`a_stale_socket_file_is_replaced`).
4. **A huge transcript** — view changes carry only what changed. Test in Task 3 (`a_streamed_reply_sends_only_its_own_line`).
5. **The engine killed while a window is attached** — the window says so in words and falls back. Test in Task 5 (`a_lost_engine_is_said_in_words`).

---

### Task 1: where the engine listens, and a detached launcher

**Files:** create `xencode-tui-rs/src/engine/address.rs`; modify `xencode-live-rs/src/lib.rs` (`spawn_detached`, `spawn_badge` uses it).

**Interfaces:**
- `pub fn fingerprint(project: &Path) -> String` (16 hex chars of SHA-256 of the canonical path with forward slashes and, on Windows, lower-cased).
- `pub enum Address { Pipe(String), Socket(PathBuf) }` with `pub fn for_project(project: &Path) -> Result<Address, String>` and `pub fn lock_path(project: &Path) -> Result<PathBuf, String>`.
- `xencode_live_rs::spawn_detached(exe: &Path, args: &[&str]) -> io::Result<u32>` (returns the pid); `spawn_badge(exe)` = `spawn_detached(exe, &[])`.

Tests: the same folder written two ways (`E:\x` and `e:/x/`) gives one fingerprint; two folders differ; the pipe name contains the user name and fingerprint; the socket path is under `<state dir>/engine/`; `spawn_detached` starts a real short-lived process (`cmd /c exit 0` or `true`) and returns its pid.

Commit: `EN-2: where a project's engine listens, and a launcher that starts it detached on every platform`.

### Task 2: the transport (`engine/transport.rs`)

**Interfaces:**
- `pub struct Listener` with `pub async fn bind(addr: &Address) -> io::Result<Listener>` and `pub async fn accept(&mut self) -> io::Result<Conn>`.
- `pub struct Conn` with `pub async fn send(&mut self, line: &str) -> io::Result<()>`, `pub async fn recv(&mut self) -> io::Result<Option<String>>` (None at end of stream), and `pub fn split(self) -> (ConnReader, ConnWriter)`.
- `pub async fn connect(addr: &Address) -> io::Result<Conn>`.
- Windows: the first pipe instance is created with a security descriptor from SDDL `D:P(A;;GA;;;<current user SID>)` (`ConvertStringSecurityDescriptorToSecurityDescriptorW`; SID from `GetTokenInformation(TokenUser)` + `ConvertSidToStringSidW`), `reject_remote_clients(true)`, and every later instance the same; a busy pipe on connect is retried for up to 2 s.
- Unix: remove a socket file nobody answers on, bind, chmod `0600`, folder `0700`.

Tests: a real round trip of three lines both ways; `recv` returns `None` after the peer drops; on Windows, the pipe's DACL read back with `GetSecurityInfo` names the current user's SID and no `WD`/`AU`/`BU` entries; on Unix, the mode is `0600`. A second-account connection test is not possible on this machine and is left recorded as unverified.

Commit: `EN-2: the local socket windows and the engine talk over, open only to the user's own account`.

### Task 3: the view (`engine/view.rs`)

**Interfaces:**
- `pub struct View` (serde): `messages_from: usize`, `messages: Vec<UiMessage>`, `generating: bool`, `bytebot_running: bool`, `bytebot_steps`, `bytebot_log`, `bytebot_progress: f64`, `tasks: Vec<ViewTask>` (`ByteBotTask` plus `this_session`), `approvals: Vec<ApprovalView>` (with the tool class), `question: Option<(u64, String)>`, `plan: Vec<PlanItem>`, `model: String`, `llm_calls: u64`, `timings: Option<…>` (whatever `last_llamacpp_timings` holds, made serialisable).
- `pub struct Watcher` remembering the last view sent: `pub fn full(&mut self, app: &App) -> View` and `pub fn changes(&mut self, app: &App) -> Option<View>` (None when nothing changed; `messages_from` = first changed line).
- `pub fn apply(app: &mut App, view: View)` — sets those fields on a window's `App`; approvals become `app.remote_approvals` (requests rebuilt for drawing), the question sets `question_id`/`question_text`.
- `EngineMsg::View { view: View }`, `ClientMsg::Note { role: String, content: String }`.
- `UiMessage`, `PlanItem`, the timings type gain `Serialize, Deserialize`.

Tests: `full` then `apply` onto a fresh `App` reproduces every field; `a_streamed_reply_sends_only_its_own_line` (a 200-line transcript whose last line grows sends `messages_from = 199` and one line); no change gives `None`; `pending_approval()` on a window returns the rebuilt request.

Commit: `EN-2: the view of agent state a window draws, sent as changes`.

### Task 4: `xencode engine` (`engine/server.rs`, CLI)

**Interfaces:**
- `pub async fn serve(project: PathBuf) -> Result<(), String>`: take the lock (or exit 0 saying another engine runs), `App::new()` with the project as working folder, bind, then one loop over: new connections, lines from connections (`hello` → `Hello` + full `View`; `note` → appended to the transcript; anything else → `handle`), and a 33 ms tick (`pump`, then each connection gets the `event`s and, if changed, a `View`). Replies to one window's message go to that window; `ApprovalResolved`/`QuestionAnswered` go to all. Idle exit per the constraints.
- CLI: `xencode engine [--project <dir>]` (default: current folder).

Tests (`xencode-cli/tests/engine_cli.rs`, real binary, settings folder with `llama_cpp_url = http://127.0.0.1:9` and model `llamacpp:none`):
`hello_gets_a_snapshot`, `two_clients_both_reach_the_engine`, `a_task_outlives_its_window` (client adds a ByteBot task, disconnects; a second client connecting later sees it ended in `failed` with the connection error, proving the engine kept it), `an_idle_engine_exits` (no client, nothing running: the process exits within 15 s), `a_second_engine_for_the_same_project_steps_aside`, `a_stale_socket_file_is_replaced` (Unix only; Windows pipes vanish with their process).

Commit: `EN-2: xencode engine runs a project's agent work in its own process`.

### Task 5: the terminal app as a window onto the engine

**Interfaces:**
- `pub struct EngineLink` (connection plus an outgoing channel) on `App` as `pub engine_link: Option<EngineLink>`.
- `engine::act` sends over the link when there is one, otherwise handles in process as today.
- `pub fn runs_in_engine(line: &str) -> bool` per ruling 2; `dispatch_prompt` in a window sends engine lines as `SubmitChat` and runs the rest locally, sending any transcript lines they add as `note`s.
- `run_app` (window mode): connect, or start the engine with `spawn_detached(current_exe, ["engine", "--project", root])` and retry for 5 s; on failure fall back to in-process with a warning. Each frame applies incoming `View`s; a lost connection becomes the warning "the engine stopped (…); starting a new one" and one reconnect attempt.
- `xencode tui --in-process` forces the old mode.
- The window's `App` has no live status file of its own (the engine writes it) and does not write conversation memory.

Tests: `runs_in_engine` for every command in `SLASH_COMMANDS`; a window `App` linked to a real engine (spawned from the test) sends `/bytebot …` and receives views showing the task; `a_lost_engine_is_said_in_words` (engine process killed; the next frame shows the warning).

Commit: `EN-2: the terminal app works as a window onto the engine, starting it when needed`.

### Task 6: watch it, document it, close out

- Live: start `xencode` (window) in a console; check the engine process appears; add a ByteBot task through a test client against a real llama.cpp server, close the window, and see the engine finish the task and the badge follow it; then the engine exits on its own. Stop the model server.
- `README.md`, `CLI_GUIDE.md` (`xencode engine`, `xencode tui --in-process`), `docs/USER_MANUAL.md`, `CHANGELOG.md`, `NEXT_PLAN_TASKS.md` (EN-2 checked with what was watched).
- Workspace suite; `cargo clean`; temp folders.

Commit: `EN-2: manuals, plan and changelog for the engine process`.
