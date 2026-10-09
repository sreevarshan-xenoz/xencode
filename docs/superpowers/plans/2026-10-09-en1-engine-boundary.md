# EN-1 — the engine boundary, in process — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every agent action the terminal app takes goes through one typed message (`ClientMsg`) handled by an engine module, and everything the agent loop reports comes back as a typed message (`EngineMsg`), with no change anyone can see — so `EN-2` only has to carry those messages over a socket.

**Architecture:** A new module `xencode-tui-rs/src/engine/` holds the wire protocol (`proto.rs`) and the engine's two entry points: `handle(app, ClientMsg)` for actions and `pump(app, rx)` for the loop's output. The engine's state is the existing `App` core, used headless; nothing moves crate yet. The token handling inside `run_app` and the approval/question drains are first pulled out into `App` methods unchanged, so the engine can call them.

**Tech Stack:** Rust 1.99, serde/serde_json, tokio mpsc/oneshot, the existing xencode-tui-rs crate.

**Spec:** `docs/superpowers/specs/2026-10-09-engine-process-design.md` (§4 messages, §5 stage EN-1).

## Global Constraints

- No visible behaviour change: the existing `xencode-tui-rs` suite (847 passed, 6 ignored, 0 failed on 2026-10-09) must pass unchanged, except where a test is rewritten only to call a moved function by its new path.
- No mocks (`AGENTS.md`). One plan ID per commit, plain-English messages, never push.
- Protocol: one JSON object per line, `"version": 1` in `hello`; message names exactly as spec §4: `submit_chat`, `enqueue_task`, `answer_question`, `answer_approval`, `stop`, `review`, `set_model`, `goodbye`; `snapshot`, `event`, `approval_requested`, `approval_resolved`, `question_asked`, `question_answered`, plus `error`.
- Every `cargo test` here sets `XCODE_CONFIG_DIR` to a fresh folder, uses `timeout -k`, `-j 4`, then a leftover-process check; `cargo clean` above 20 GiB.

## Ruling carried from writing this plan

The spec's EN-1 names a new crate `xencode-engine-rs`. The agent loop depends on about a dozen
modules of `xencode-tui-rs` (agent_tools, mcp, sandbox, ckptgit, reprogate, bytebot_tasks,
live_status, detached, …), so a separate crate would mean moving most of the crate at once. The
CLI already depends on `xencode-tui-rs`, so `xencode engine` (EN-2) can live there too. EN-1
therefore builds the engine as the module `xencode-tui-rs/src/engine/`; extracting a crate waits
until the desktop app needs the protocol types without the TUI, and then only `proto.rs` moves.

## Review Focus

1. **An approval answered by id after it was already answered** (two windows later) must not answer the next request in the queue. Test in Task 4 (`a_stale_approval_id_answers_nothing`).
2. **A question answered after it was withdrawn** (Esc) must be refused, not panic. Test in Task 4 (`an_answer_to_a_withdrawn_question_is_refused`).
3. **An unknown or newer message** on decode must give a readable error, not a crash. Test in Task 3 (`unknown_messages_are_refused_in_words`).
4. **Tokens that start a new run inside `apply_token`** (`[SPAWN]` scheduling a queued worker, `[BYTEBOT_DONE]` starting the next task) must still start them after the move. Covered by the existing suite plus Task 1's test.
5. **The order of a frame** — tokens, then approvals, then questions — must stay as `run_app` had it, or an approval could appear before the text that led to it. Test in Task 5 (`pump_keeps_the_frame_order`).

---

### Task 1: `App::apply_token` — the token handling, moved out of `run_app`

**Files:** Modify `rust/crates/xencode-tui-rs/src/app.rs`.

**Interfaces:**
- Produces: `pub fn apply_token(&mut self, token: &str, tx: &mpsc::UnboundedSender<String>)`.

- [ ] **Step 1: Failing test** in `app.rs` tests:
```rust
    /// EN-1: the loop's tokens are applied by one App method, the same one
    /// `run_app` calls, so the engine can call it too.
    #[tokio::test]
    async fn apply_token_is_what_the_main_loop_does_with_a_token() {
        let mut app = App::for_tests();
        let (tx, _rx) = mpsc::unbounded_channel();
        app.is_generating = true;
        app.apply_token("hello ", &tx);
        app.apply_token("world", &tx);
        app.apply_token("[TOOL]→ read_file a.rs", &tx);
        app.apply_token("[STOPPED]", &tx);
        app.apply_token("[DONE]", &tx);
        let text: Vec<String> = app.messages.iter().map(|m| m.content.clone()).collect();
        assert!(text.iter().any(|t| t == "hello world"), "{text:?}");
        assert!(text.iter().any(|t| t == "⚙→ read_file a.rs"), "{text:?}");
        assert!(text.iter().any(|t| t == "■ Turn stopped."), "{text:?}");
        assert!(!app.is_generating);
    }
```
- [ ] **Step 2: Run, expect "no method named `apply_token`".**
- [ ] **Step 3: Move.** Cut the body of `while let Ok(token) = rx.try_recv() { … }` in `run_app` (from `if let Some(body) = token.strip_prefix("[REVIEW]")` to the final `else { app.append_generation(&token); }`) into the new method, replacing `app.` with `self.` and `&token` with `token`; `tx.clone()` stays. The loop becomes:
```rust
        while let Ok(token) = rx.try_recv() {
            signals.messages += 1;
            app.apply_token(&token, &tx);
        }
```
- [ ] **Step 4: Test passes; full `xencode-tui-rs` suite unchanged (847/0/6); clippy clean.**
- [ ] **Step 5: Commit** — `EN-1: the main loop's handling of agent messages becomes one App method the engine can call`.

### Task 2: `App::drain_agent_channels` — approvals and questions, moved out of `run_app`

**Files:** Modify `app.rs`.

**Interfaces:**
- Produces: `pub fn drain_agent_channels(&mut self) -> usize` (number of approval requests queued, for `signals.approvals`).

- [ ] **Step 1: Failing test:**
```rust
    #[tokio::test]
    async fn drain_agent_channels_queues_approvals_and_questions() {
        let mut app = App::for_tests();
        let request = crate::agent_tools::ApprovalRequest {
            tool: "write_file".into(),
            class: crate::agent_tools::ToolClass::Edit,
            summary: "write_file a.rs".into(),
            preview: String::new(),
            draft: Default::default(),
        };
        let (responder, _answer) = tokio::sync::oneshot::channel();
        app.approval_tx.send((request, responder)).unwrap();
        let (reply, _q) = tokio::sync::oneshot::channel();
        app.ask_tx.send(("Which port?".into(), reply)).unwrap();
        assert_eq!(app.drain_agent_channels(), 1);
        assert_eq!(app.pending_approval().map(|r| r.summary.as_str()), Some("write_file a.rs"));
        assert!(app.bytebot_help.is_some());
    }
```
- [ ] **Step 2: Run, expect missing method.**
- [ ] **Step 3: Move** the two drains after the token loop in `run_app` (the `waiting` approval drain with `live_approval_waiting`, and the `asked` question drain) into the method, returning the approval count; `run_app` sets `signals.approvals += app.drain_agent_channels();`. Check `ApprovalRequest`'s real fields with `grep -n "pub struct ApprovalRequest" -A8 src/agent_tools.rs` and adjust the test's literal.
- [ ] **Step 4: Pass; suite unchanged; clippy.** **Step 5: Commit** — `EN-1: approvals and ByteBot questions are queued by one App method`.

### Task 3: the wire protocol (`engine/proto.rs`)

**Files:** Create `src/engine/mod.rs` (`pub mod proto;`), `src/engine/proto.rs`; `src/lib.rs` gets `pub mod engine;`.

**Interfaces:**
- Produces:
```rust
pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMsg {
    Hello { version: u32, client: String },
    SubmitChat { prompt: String },
    EnqueueTask { text: String },
    AnswerQuestion { id: u64, text: String },
    AnswerApproval { id: u64, answer: WireAnswer },
    Stop { target: StopTarget },
    Review { decision: ReviewDecision },
    SetModel { name: String },
    Goodbye,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireAnswer { Allow, AllowForSession, Deny }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopTarget { Chat, Bytebot }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewDecision { Accept, Undo }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalView { pub id: u64, pub tool: String, pub class: String, pub summary: String, pub preview: String }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineMsg {
    Hello { version: u32, session_id: String },
    Snapshot { model: String, generating: bool, bytebot_running: bool, approvals: Vec<ApprovalView>, question: Option<(u64, String)> },
    Event { token: String },
    ApprovalRequested { approval: ApprovalView },
    ApprovalResolved { id: u64, answer: WireAnswer, by: String },
    QuestionAsked { id: u64, text: String },
    QuestionAnswered { id: u64, by: String },
    Error { message: String },
}

pub fn encode(msg: &impl Serialize) -> String;            // one line, no trailing newline inside
pub fn decode_client(line: &str) -> Result<ClientMsg, String>;
pub fn decode_engine(line: &str) -> Result<EngineMsg, String>;
pub fn version_mismatch(theirs: u32) -> String;           // words naming both versions
```
- [ ] **Step 1: Failing tests** in `proto.rs`: every `ClientMsg` and `EngineMsg` variant round-trips through `encode`/`decode_*` and `encode` contains no `'\n'`; `"type":"submit_chat"` appears in the encoded `SubmitChat`; `unknown_messages_are_refused_in_words` (`{"type":"launch_rockets"}` and `not json` both give an `Err` whose text names what was wrong); `version_mismatch(2)` contains "1" and "2".
- [ ] **Step 2: Run, fail. Step 3: implement** (serde derives above; `encode` = `serde_json::to_string`; decoders map serde errors to `format!("not a message xencode understands: {e}")`).
- [ ] **Step 4: Pass; clippy. Step 5: Commit** — `EN-1: the messages windows and the engine will exchange, as typed JSON lines`.

### Task 4: the engine's action side (`engine::handle`)

**Files:** `src/engine/mod.rs`; `app.rs` (approval and question ids).

**Interfaces:**
- Consumes: Task 3's types; `App::{dispatch_prompt, run_bytebot (via bytebot_enqueue), bytebot_answer, bytebot_withdraw_question, resolve_approval, bytebot_accept, bytebot_undo, set_model, live_stopped}`; the stop flags `turn_stop`, `bytebot_stop`.
- Produces:
  - App fields `pub(crate) approval_ids: VecDeque<u64>` (parallel to `approval_queue`), `pub(crate) question_id: Option<u64>`, `next_agent_id: u64`; `drain_agent_channels` assigns ids as it queues.
  - `pub fn handle(app: &mut App, msg: ClientMsg, tx: &mpsc::UnboundedSender<String>, by: &str) -> Vec<EngineMsg>`.

Mapping:
| ClientMsg | Engine does | Returns |
|---|---|---|
| `SubmitChat{prompt}` | `app.dispatch_prompt(prompt, tx.clone())` | `[]` |
| `EnqueueTask{text}` | `app.bytebot_enqueue(&text, tx.clone())` | `[]` |
| `AnswerQuestion{id,text}` | if `question_id == Some(id)`: set `bytebot_command = text`, `bytebot_answer()` → `QuestionAnswered{id, by}`; else `Error` "that question is no longer waiting" | |
| `AnswerApproval{id,answer}` | if `approval_ids.front() == Some(&id)`: `resolve_approval(answer.into())`, pop id → `ApprovalResolved{id,answer,by}`; else `Error` "that approval is no longer waiting" | |
| `Stop{Chat}` | set `turn_stop` if generating | `[]` |
| `Stop{Bytebot}` | set `bytebot_stop` if running, and `bytebot_withdraw_question()` | `[]` |
| `Review{Accept}` / `Review{Undo}` | `bytebot_accept` / `bytebot_undo` (an `Err` becomes `Error{message}`) | |
| `SetModel{name}` | `app.set_model(&name, tx.clone())` | `[]` |
| `Hello` | `version_mismatch` if `version != PROTOCOL_VERSION`, else `Hello{version, session_id}` | |
| `Goodbye` | nothing | `[]` |

- [ ] **Step 1: Failing tests** in `engine/mod.rs` (`#[tokio::test]`, `App::for_tests()`, dead server `http://127.0.0.1:9` as in the ByteBot tests): one per row above, plus `a_stale_approval_id_answers_nothing` (two queued approvals; answering the first id twice resolves only the first, the second call returns `Error`, the second request still waits) and `an_answer_to_a_withdrawn_question_is_refused`.
- [ ] **Step 2: Fail. Step 3: implement** (ids assigned in `drain_agent_channels`; `resolve_approval` also pops `approval_ids`, so a resolution from the TUI's own keys keeps the two queues aligned).
- [ ] **Step 4: Pass; suite unchanged; clippy. Step 5: Commit** — `EN-1: one engine function carries out every agent action a window can ask for`.

### Task 5: the engine's output side (`engine::pump`)

**Files:** `src/engine/mod.rs`, `app.rs` (`run_app`).

**Interfaces:**
- Produces: `pub fn pump(app: &mut App, rx: &mut mpsc::UnboundedReceiver<String>, tx: &mpsc::UnboundedSender<String>) -> Pump` where `pub struct Pump { pub messages: usize, pub approvals: usize, pub out: Vec<EngineMsg> }`. It applies every waiting token (`apply_token`) emitting `Event{token}` for each, then `drain_agent_channels` emitting `ApprovalRequested`/`QuestionAsked` for each new one — tokens first, then approvals, then questions, as `run_app` did.

- [ ] **Step 1: Failing tests:** tokens become `Event`s in order; a queued approval becomes `ApprovalRequested` with its id, summary and class label; `pump_keeps_the_frame_order` (a token, an approval and a question sent before one pump come out in that order).
- [ ] **Step 2: Fail. Step 3: implement; `run_app` replaces its token loop and drain call with `let pumped = engine::pump(&mut app, &mut rx, &tx); signals.messages += pumped.messages; signals.approvals += pumped.approvals;` (the `out` messages are not needed in process yet).
- [ ] **Step 4: Pass; suite unchanged; clippy. Step 5: Commit** — `EN-1: the engine turns everything the agent loop reports into typed messages`.

### Task 6: the terminal app's actions go through the engine

**Files:** `app.rs`, `keymap.rs`.

Every agent action the terminal app takes calls `engine::handle(app, ClientMsg::…, tx, "terminal")` instead of the App method directly:
- chat Enter: `submit_message` keeps reading the composer and history, then `engine::handle(SubmitChat{prompt})`;
- ByteBot Enter: `run_bytebot` routes slash lines as now and plain tasks via `EnqueueTask`; a waiting question is answered via `AnswerQuestion{id}`;
- approval keys y / a / n: `AnswerApproval{id: front id}`;
- Esc / Ctrl+C stop: `Stop{Chat}` and `Stop{Bytebot}`;
- `a` / `u` review: `Review`;
- Models screen Enter and `/model <name>`: `SetModel`.

- [ ] **Step 1:** Before changing call sites, list them: `grep -n "dispatch_prompt\|bytebot_enqueue\|bytebot_answer\|resolve_approval\|bytebot_accept\|bytebot_undo\|set_model(" src/app.rs src/keymap.rs` — record the list in the ledger.
- [ ] **Step 2:** Change each call site. The existing keymap, approval and ByteBot tests are the failing-then-passing guard: run them after each group of call sites.
- [ ] **Step 3:** Afterwards the same grep shows those methods called only from `engine/mod.rs` and from tests; anything else is a missed route.
- [ ] **Step 4:** Full suite unchanged; clippy. **Step 5: Commit** — `EN-1: the terminal app asks the engine for every agent action instead of acting directly`.

### Task 7: close out

- [ ] Full `cargo test --workspace` (Windows): the 35 known failures elsewhere only. `NEXT_PLAN_TASKS.md` EN-1 checked with what was proved; `CHANGELOG.md` gets a short "Changed (internal)" note; `cargo clean`; temp folders cleaned.
- [ ] Commit — `EN-1: plan and changelog record the engine boundary`.
