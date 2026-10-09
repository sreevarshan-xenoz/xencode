# Desktop app, floating badge and ByteBot tasks — design

Date: 2026-10-09. Status: awaiting owner review. Plan IDs: `DK-1` … `DK-5`, `BT-1` … `BT-5`
(Milestone AJ in `NEXT_PLAN_TASKS.md`).

## 1. What the owner asked for

Said by the owner, in their words summarised:

- A desktop app for Windows and Mac, for people who prefer a GUI to the terminal, with
  everything the terminal app has. Zed (`zed-industries/zed`) is the inspiration.
- A small xencode logo floating at the edge of the screen while xencode runs and the person
  does other work. Hovering it shows what is happening; it signals when the agent needs the
  person and when the work is done. Most useful with ByteBot runs.
- ByteBot "completed the way the original project does it". The owner left the reading of
  that to us: "choose which is better and useful".
- The desktop app and the badge are designed together.

Decided with the owner during this design (2026-10-09):

- **One engine, many windows.** xencode's engine becomes a background process; the terminal
  app, the desktop app and the badge are all clients of it.
- **ByteBot borrows the original project's task model** (task states, a task list, a step
  where the person takes over). It stays xencode's coding agent. It does not control a
  desktop.
- **The status feed is one small file per running session.**

Assumed, and open to correction:

- Everything is Rust, as `AGENTS.md` requires. The GUI library is GPUI.
- The badge ships before the full desktop app, because it is useful with the terminal app
  people already run.

## 2. Facts this design rests on (checked 2026-10-09)

- `bytebot-ai/bytebot`: 11,077 stars, Apache-2.0, **archived**, last push 2025-09-12. It runs
  an agent inside a Docker Ubuntu desktop and acts by screenshot, mouse and keyboard. Its task
  states (`packages/bytebot-agent/prisma/schema.prisma`): `PENDING`, `RUNNING`, `NEEDS_HELP`,
  `NEEDS_REVIEW`, `COMPLETED`, `CANCELLED`, `FAILED`.
- xencode's ByteBot (`I2-04`) is the ordinary agent loop reported to its own panel. One run at a
  time, no persistence, no states beyond running and not running. Its command box sends every
  line to the model, including `/init`. It has no way to change model.
- GPUI: Apache-2.0, crate `gpui` 0.2.2 on crates.io (published 2025-10-22). Zed's copy changes
  daily, so this design pins a git revision rather than the year-old crate. Zed's editor code is
  GPL; nothing of it is copied, only GPUI is used as a library.
- Nothing in xencode today emits a desktop notification, a bell or a window title, and nothing
  outside the TUI process can see its state. `xencode-server-rs` serves collaboration sessions,
  not the TUI's live state.
- Choosing a model sets `default_model` and returns to chat; it does not clear the context
  window measured for the previous model (`server_context_window`, `ollama_window`), so the
  first turn after a switch can be budgeted for the old model.

## 3. The parts, in build order

| Part | IDs | What ships | Designed here |
|---|---|---|---|
| 1 | `DK-1`, `DK-2` | Status feed from the terminal app; the floating badge | In full |
| 2 | `BT-1` … `BT-5` | ByteBot task list and states; slash commands and model switching in its panel | In full |
| 3 | `DK-3` | The engine moves into a background process; the terminal app becomes its first client | Named only |
| 4 | `DK-4` | Desktop app shell on GPUI: chat, approvals, model picker, settings | Named only |
| 5 | `DK-5` | The remaining terminal panels ported to the desktop app, in groups | Named only |

Parts 3 to 5 each get their own design document before any code. Nothing in parts 1 and 2 is
thrown away by them: the engine process takes over writing the status feed, and the badge keeps
reading the same files.

## 4. Part 1 — the status feed (`DK-1`)

### 4.1 Where and what

One file per running session: `<state dir>/live/<session id>.json`, where `<state dir>` is
`xencode_config_rs::paths::state_dir()` (so `XCODE_CONFIG_DIR` moves it, as it moves everything
else). Written atomically: to a temporary file in the same folder, then renamed over the old one.

```json
{
  "version": 1,
  "session_id": "1760000000-sess",
  "pid": 12345,
  "project": "E:/xencode",
  "model": "llamacpp:qwen3-4b",
  "source": "chat",
  "state": "needs_you",
  "headline": "waiting for you to allow: write_file src/auth.rs",
  "changed_at": 1760000123,
  "heartbeat_at": 1760000125
}
```

- `state` is one of `idle`, `working`, `needs_you`, `finished`, `failed`.
- `source` is `chat` or `bytebot`: which loop the state is about.
- `headline` is at most 120 characters. It is built from the tool call's approval summary or
  the step being run, **never from the prompt or the model's text**, and it is passed through
  `redact_secrets` (`xencode-context-rs/src/trace.rs`) before it is written. The file lives
  outside the project, but it is still a file on disk another program reads.
- `project` and paths use forward slashes on every platform.

### 4.2 When it changes

| Event in the TUI | `state` | `headline` |
|---|---|---|
| A chat turn or ByteBot task starts | `working` | the first step, or "thinking" |
| A tool call starts | `working` | its approval summary |
| An approval prompt is waiting | `needs_you` | "waiting for you to allow: …" |
| A ByteBot task is in needs help or needs review (part 2) | `needs_you` | the question, or "review N changed files" |
| A turn or task ends normally | `finished` | "done in M:SS" |
| A turn or task ends on a provider or tool error | `failed` | the error's first line, redacted |
| Esc stops a turn | `idle` | "stopped" |

`heartbeat_at` is rewritten every 5 seconds from the TUI's main loop, with no other change. A
clean exit deletes the file. A file whose heartbeat is older than 20 seconds belongs to a
session that died; the badge shows it as "not responding" and ignores it after 10 minutes. The
badge never deletes another program's file.

### 4.3 Code

A new module `rust/crates/xencode-tui-rs/src/live_status.rs` owns the type, the write and the
redaction. The TUI calls it from the places that already change `is_generating`,
`bytebot_running` and the approval queue, so there is one writer per session and no new thread.

## 5. Part 1 — the floating badge (`DK-2`)

### 5.1 Shape

A new binary crate `rust/crates/xencode-badge`, built on GPUI pinned to one git revision.

- `xencode badge` (new CLI subcommand) starts it detached. It is found next to the `xencode`
  executable, then on `PATH`; if neither has it, the command says so and how to build it.
- A setting, `badge_autostart` (default off), makes the TUI start it on launch if it is not
  already running. A lock file in `<state dir>/live/` keeps it to one badge per user.

### 5.2 Behaviour

- A round logo about 40 pixels across, borderless, always on top, snapped to the right edge of
  the screen. It can be dragged along any edge; the position is saved in
  `<settings dir>/badge.json`.
- It shows the most urgent state among all live sessions, in this order: `needs_you`, `failed`,
  `working`, `finished`, `idle`.

| State | Look |
|---|---|
| No session running | Dim logo |
| `working` | A slowly turning ring |
| `needs_you` | Amber, a gentle pulse |
| `finished` | A green tick that fades back after 60 seconds |
| `failed` | A red mark that stays until hovered |
| A session not responding | A grey dot on the logo |

- Hovering slides out a card with one row per session: project, model, state in words, the
  headline, and how long since it changed. The state is always written in words, never colour
  alone.
- Version one does not focus a terminal window on click, and sends no desktop notifications.

### 5.3 Code

- `badge_model.rs`: plain Rust, no GPUI. Reads the folder, parses each file, applies the stale
  rules and the urgency order, and returns what to draw. All of the badge's decisions live here
  and are tested here.
- `main.rs` and the view: GPUI window, drawing and hover only. It watches the folder with the
  `notify` crate and re-reads every 2 seconds as a fallback, because file watching is unreliable
  on some network and synced folders.

## 6. Part 2 — ByteBot tasks (`BT-1` … `BT-5`)

### 6.1 `BT-1` — a task record and a queue

- Each task is one file: `<project>/.xencode/bytebot/tasks/<task id>.json`, holding the task
  text, the model, the state, the step rows, the changed files, and the times it was created,
  started and ended. `.xencode` is already ignored by git.
- States, as in the original project: `pending`, `running`, `needs_help`, `needs_review`,
  `completed`, `cancelled`, `failed`.
- Enter in the panel while a task runs adds a `pending` task. Tasks run one at a time, oldest
  first. The panel lists them with their state in words; the list survives a restart, and a task
  found `running` after a restart is marked `failed` with "xencode exited during this task".

### 6.2 `BT-2` — needs help, and taking over

- A new tool, `ask_user`, with one argument, `question`, is offered to ByteBot runs only. Calling
  it moves the task to `needs_help` and pauses the loop at that point, the same way an approval
  prompt pauses it today.
- The panel shows the question and an answer box. The answer is returned to the model as the
  tool's result, and the task goes back to `running`.
- **Take over**: instead of answering, the person can do the work themselves and then type
  `/done` in the answer box. The model is told "the person did this step themselves" and
  continues. A slash command is used rather than a key chord because every printable key in the
  panel already goes to its text box.
- Esc while a task waits for help cancels it, as Esc cancels a running task.

### 6.3 `BT-3` — needs review

- When a task ends normally and its checkpoint group changed files, it goes to `needs_review`
  instead of `completed`. The panel lists the changed files.
- With the command box empty, `a` accepts and moves it to `completed`; `u` undoes the task's
  whole checkpoint group with the existing rewind machinery and moves it to `cancelled`, noting
  "changes undone". With text in the box, both letters type as usual.
- A task that changed nothing goes straight to `completed`.

### 6.4 `BT-4` — slash commands in the panel

A line in the panel's command box that starts with `/` runs through the same command handler as
chat (`submit_message`'s slash chain), not to the model. Output goes where that command already
writes; the panel shows one line saying the command ran and where its output is. `/init` from the
panel therefore does exactly what `/init` in chat does.

### 6.5 `BT-5` — model switching without leaving the panel

- A new command, `/model`, available in chat and in the ByteBot panel. With a name it switches;
  with none it opens a model list inside the panel (arrows and Enter), using the same list as the
  Models screen.
- Every model change — the Models screen, `/model`, the panel list — goes through one new
  function, `App::set_model`, which also clears the context window measured for the previous
  model and swaps the llama.cpp server's loaded model when needed. That closes the gap in §2.
- A task already running keeps the model it started with; the record says which.

### 6.6 Feeding the badge

`needs_help` and `needs_review` write `needs_you`; `completed` writes `finished`; `failed` writes
`failed`; `cancelled` writes `idle`.

## 7. Testing and what counts as verified

- **Status feed.** Tests drive a real `App` through each row of §4.2 and read the real file back,
  including a stale heartbeat and a headline holding a secret-shaped string, which must be
  redacted. They run with `XCODE_CONFIG_DIR` set to a temporary folder.
- **Badge model.** Ordinary unit tests over real files in a temporary folder: urgency order,
  stale and dead sessions, unreadable or newer-version files (skipped with a reason, not
  crashing).
- **Badge window.** Run on this Windows machine against a live terminal session, through each
  state, and the result written down with what was seen. **macOS is built in CI only and stays
  marked "not yet watched on a Mac"** until someone runs it on one; per `AGENTS.md` it is not
  checked off.
- **ByteBot tasks.** Each state change tested against the real agent loop with the scripted model
  server the suite already uses (`serve_scripted_answers`). One live run with a real local model
  for the whole path, with the llama.cpp server stopped as soon as it ends.
- No mocks anywhere, as `AGENTS.md` requires.

## 8. Not in this design

- Desktop control by screenshot, mouse and keyboard. The plan rejected it in September on
  reliability grounds and nothing here reopens that.
- Connecting to a Bytebot server: the project is archived.
- Desktop pop-up notifications, focusing a terminal window from the badge, and sound.
- Parts 3 to 5 beyond their names and order.

## 9. Risks

- **GPUI's published crate is a year behind its source.** Pinning a git revision avoids that but
  means updating deliberately. If GPUI cannot do a transparent always-on-top window on Windows,
  `DK-2` stops and reports it before any other GUI library is considered.
- **The engine split (part 3) is the largest single job** in the programme and touches the
  20,000-line `app.rs`. It gets its own design so that the terminal app keeps working at every
  commit.
- **macOS cannot be verified from this machine.**
