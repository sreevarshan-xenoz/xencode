# EN-3 — several windows per engine, and the engine's lifetime rules — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Several windows share one engine as equals — each sees who answered a prompt it was also showing — and an engine left with nobody to answer it finishes its waits by the design's 30-minute rule instead of waiting forever; the engine is the only process that writes the project's conversation memory, task records and badge status file.

**Architecture:** EN-2 already gives every window a full view on connect, sends every window the same prompts, and lets only the first answer to a prompt win. EN-3 adds: names that tell windows apart, an "answered in another window" notice, a protocol mismatch that ends the connection, an unattended-wait rule run by the engine on each tick, and a window constructor that opens nothing for writing.

**Tech Stack:** Rust, tokio, the EN-2 engine modules in `xencode-tui-rs/src/engine/`.

**Spec:** `docs/superpowers/specs/2026-10-09-engine-process-design.md` (§3 Stop and One writer, §4 Approvals across windows and Framing, §6, §7).

## Global Constraints

- No mocks: real engine processes, real pipes, real `App`s with real channels. Anything needing a model that produces tool calls runs against a real llama.cpp server and is `#[ignore]`d with the reason, run here by hand, and recorded as watched.
- Plan-item commits in plain English; never push; `XCODE_CONFIG_DIR` set for every test run; `-j 4`; leftover-process check; `cargo clean` above 20 GiB.
- Waiting with no window: a question is withdrawn and its task cancelled; a review is completed with its changes kept; then the engine exits by the idle rule. The limit is 30 minutes.

## Rulings carried from writing this plan

1. **An approval left waiting with no window is denied** when the limit passes. The spec names questions and reviews only; an approval is the same kind of wait (a person must answer), and denying is the safe answer: nothing is written.
2. **The limit is a documented option**, `xencode engine --wait-limit <seconds>` (default 1800), so a person can shorten it and the end-to-end test can watch it act in seconds.
3. **Windows name themselves** `terminal <pid>` so a notice can say which window answered.
4. The "answered in another window" notice is an info toast in the window that did not answer: `answered in <window>: <answer>` for an approval, `answered in <window>` for a question.

## Review Focus

1. Two windows answering one approval within the same tick — exactly one answer counts, the other window is told.
2. A window connecting while the unattended clock runs — the clock stops; it does not cancel a question a window is now showing.
3. A question answered in the last second before the limit — not withdrawn afterwards.
4. A window started before any engine exists — it writes no memory, task record or status file of its own.
5. A version mismatch — the error is said, then the connection ends, and no message from that window is acted on.

---

### Task 1: windows are told who answered

**Files:** `engine/link.rs` (client name, notice), `engine/server.rs` (unchanged broadcast), `app.rs` (`run_app` client name).

- `link::open(addr, client)` already takes a name; `run_app` and `connect_or_start` pass `format!("terminal {}", std::process::id())`. `EngineLink` keeps its own name.
- `pub fn answered_elsewhere(msg: &EngineMsg, me: &str) -> Option<String>`: `ApprovalResolved { by, answer, .. }` with `by != me` → `answered in {by}: {allow|allow for the session|deny}`; `QuestionAnswered { by, .. }` with `by != me` → `answered in {by}`; anything else → `None`.
- `link::frame` shows that text as an info toast.

Tests: `answered_elsewhere` for both messages, for the window's own answers (None), and for other messages; a live `#[ignore]` test (`two_windows_racing_for_one_approval`) against a real llama.cpp server named by `XENCODE_LIVE_LLAMACPP_URL`: a task asked to write a file raises an approval; two windows answer at once; exactly one `approval_resolved` is sent, both windows receive it naming the same winner, the loser's own answer gets "that approval is no longer waiting".

Commit: `EN-3: a window is told when another window answered a prompt it was showing`.

### Task 2: a protocol mismatch ends the connection

**Files:** `engine/server.rs`.

- A `hello` with another version is answered with the error naming both versions, then the window is dropped (its writer drains the error first).

Test (`engine_cli.rs`): `a_window_speaking_another_version_is_told_and_let_go` — hello with version 999; the error names 1 and 999; the next read is the end of the connection; a later `note` on the same connection never appears in another window's transcript.

Commit: `EN-3: a window speaking another protocol version is told so and disconnected`.

### Task 3: nothing waits forever for a window that is not there

**Files:** `engine/mod.rs` (`unattended`), `engine/server.rs` (clock, `--wait-limit`), `xencode-cli/src/main.rs`.

- `pub fn unattended(app: &mut App, tx) -> Vec<String>`: withdraws a waiting question (`bytebot_withdraw_question`, the task ends cancelled), denies every waiting approval, and accepts a task waiting for review (`bytebot_accept`, changes kept). Returns one line per thing it did, added to the transcript as system lines.
- The server keeps `unattended_since: Option<Instant>`, set when no window is connected and something waits on a person (question, approval or review), cleared when a window connects or nothing waits. When it passes the limit, `unattended` runs once.
- `serve(project, wait_limit: Duration)`; CLI `xencode engine --wait-limit <seconds>` (default 1800).
- `has_work` stays as it is; a task waiting for review does not keep the engine alive (EN-2 behaviour, kept).

Tests: unit tests on a real `App` with real channels — a waiting question is withdrawn and its task cancelled; a waiting approval's responder receives `Denied`; a task waiting for review becomes completed with its changed files still on disk. Live `#[ignore]` test `a_wait_with_no_window_ends_by_the_limit` with `--wait-limit 5`: the approval is denied about five seconds after the last window left, and the engine then exits.

Commit: `EN-3: an engine with no window ends a wait after a limit instead of waiting forever`.

### Task 4: the engine is the only writer

**Files:** `app.rs` (`App::for_window`, `run_app` order), `engine/link.rs`.

- `pub fn for_window() -> App` builds the settings, plugins, skills and arrangement as `App::new` does, but with conversation memory that is not saved, no live status file, no task store and no recovery pass over task records.
- `run_app` connects (or starts the engine) first, then builds `App::for_window()` on success or `App::new()` when it runs the work itself. `become_window` no longer needs to drop the status file and store.

Tests: `a_window_app_writes_nothing_of_the_engines` (the only test in its own CLI test file, `engine_window_writes.rs`, so `XCODE_CONFIG_DIR` and the working folder can be set for that whole test process before anything reads them): build `App::for_window()` in an empty project folder with a fresh settings folder; it has no status file, no task store and memory that is not saved; after it has run a window-side command, the settings folder holds no conversation memory file and no `live/` file, and the project holds no `.xencode/bytebot` folder.

Commit: `EN-3: only the engine writes conversation memory, task records and the badge's status file`.

### Task 5: watch it, document it, close out

- Live, real llama.cpp (Qwen3-4B): two windows on one engine race an approval; one window closes with a question waiting and `--wait-limit 20`; watch the question withdrawn, the task cancelled and the engine exit; the badge's status file follows.
- `CLI_GUIDE.md` (`--wait-limit`, what happens to waits with no window), `docs/USER_MANUAL.md`, `README.md`, `CHANGELOG.md`, `NEXT_PLAN_TASKS.md` (EN-3 checked with what was watched; test counts).
- Workspace suite; `cargo clean`; temp folders.

Commit: `EN-3: manuals, plan and changelog for several windows and the wait limit`.
