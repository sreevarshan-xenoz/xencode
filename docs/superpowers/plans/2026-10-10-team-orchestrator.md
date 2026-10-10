# Team orchestrator (TM-1 … TM-6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A lead agent chosen by the person (Claude Code, Codex, Gemini CLI, Antigravity) directs worker agents on one project through xencode: each worker runs in its own git worktree inside the project's engine, and its work merges on its own only when the project's checks pass on the merged result.

**Architecture:** A new crate `xencode-team-rs` holds everything that does not need the terminal app: worktrees, the ACP worker runtime (the official crate's client side), the checked merge, the agent table. The engine (`xencode-tui-rs/src/engine`) hosts a `Team` of workers and answers new `ClientMsg::Team*` requests; workers' permission prompts become ordinary engine approvals. `xencode mcp serve --team` connects to the engine as a window and turns seven MCP tools into those requests. The worker panel and the badge read the engine's view.

**Tech Stack:** Rust, tokio, `agent-client-protocol = "=3.3.0"` (features `stdio`, `process`), `rmcp` (already used by `mcp_serve.rs`), git on PATH.

**Spec:** `docs/superpowers/specs/2026-10-10-team-orchestrator-design.md`

## Global Constraints

- Rust only; **no mocks** in product code or tests (`AGENTS.md`). Tests use real git repositories, real worktrees, real check commands, and a real `xencode acp` worker whose model address nobody listens on (`http://127.0.0.1:9`) or that stalls (`stalled_settings()` pattern from `acp_cli.rs`).
- **Never run anything on the owner's local GPU** (memory `ask-before-gpu`). Model-backed live runs go to Colab (`xencode colab up` from WSL) or to vendor APIs with the owner's keys, and only when the owner asks.
- Outside agents' live tests are `#[ignore = "needs <VENDOR>_API_KEY and costs money"]`; never run without the owner asking.
- Adapters are pinned: `@agentclientprotocol/claude-agent-acp@0.89.1`, `@agentclientprotocol/codex-acp@2.2.2`, `gemini --acp`, Antigravity `agy_acp_server` (registry `antigravity-acp` 1.3.0).
- Worktree layout: `<parent>/<repo>-team/<id>` on branch `xencode/team/<id>`; worker ids are `w1`, `w2`, … per engine.
- Default worker limit 4; default check timeout 1200 s; both overridable in `.xencode/team.toml`.
- Every `cargo test` run sets `XCODE_CONFIG_DIR` to a fresh temp folder, uses `timeout -k`, `-j 4`; test the changed crate while iterating, `cargo test --workspace` before each commit; `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings -A clippy::format-in-format-args` clean on Windows **and** on Linux (WSL clone at `~/xencode`), because CI's Linux job compiles `cfg(unix)` tests Windows skips.
- One commit per stage (`TM-1` … `TM-6`), subject naming the ID, plain-English body, ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push unless asked.

## Review Focus

1. **A worker's agent process dies mid-task** (crash, killed): its state becomes `failed` with the last error, its worktree is kept, and no request from the lead hangs. Pinned in Task 1.
2. **The base branch moves while a merge's checks run** (the person commits meanwhile): the merge does not land on stale results; it starts over from the new base. Pinned in Task 4.
3. **The person's working copy has uncommitted changes when a merge lands**: the merge refuses rather than merging over them. Pinned in Task 4.
4. **The lead asks about a worker id that does not exist, or calls a tool while no engine runs**: an error in words, not a hang or a panic. Pinned in Task 3.
5. **Two workers' approvals arrive at once**: each is its own engine approval with its own id, and answering one does not answer the other. Pinned in Task 2.

---

## File Structure

- `rust/crates/xencode-team-rs/` (new): `src/lib.rs` (types: `WorkerId`, `WorkerState`, `WorkerSnapshot`), `src/worktree.rs` (create, remove, diff), `src/agents.rs` (the agent table and launch specs), `src/worker.rs` (one ACP worker), `src/merge.rs` (the checked merge), `src/config.rs` (`.xencode/team.toml`).
- `rust/crates/xencode-tui-rs/src/engine/proto.rs` — `ClientMsg::Team(TeamRequest)`, `EngineMsg::TeamReply { req, reply }`.
- `rust/crates/xencode-tui-rs/src/engine/team.rs` (new) — the engine's `Team`: owns workers, turns their permission prompts into approvals, answers requests.
- `rust/crates/xencode-tui-rs/src/engine/view.rs` — `View.team: Option<Vec<WorkerSnapshot>>`.
- `rust/crates/xencode-tui-rs/src/mcp_serve.rs` — `--team` tools.
- `rust/crates/xencode-tui-rs/src/worker_panel.rs` — a Team section.
- `rust/crates/xencode-live-rs/src/lib.rs` — `LiveSource::Worker`.
- `rust/crates/xencode-cli/src/main.rs` — `mcp serve --team`, `team login-optin <agent>`, `team clean`.
- Tests: unit tests in each new file; `rust/crates/xencode-cli/tests/team_cli.rs` (real engine, real `xencode acp` workers).

---

### Task 1: Worktrees and one ACP worker (TM-1)

**Files:** Create `xencode-team-rs` (`Cargo.toml`, `src/lib.rs`, `src/worktree.rs`, `src/worker.rs`); add to `rust/Cargo.toml` members. Test: unit tests in `worktree.rs`; `rust/crates/xencode-cli/tests/team_cli.rs`.

**Interfaces (produced):**

```rust
// lib.rs
pub type WorkerId = String; // "w1", "w2", …
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState { Starting, Working, NeedsYou, Done, Failed, Stopped }
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerSnapshot {
    pub id: WorkerId, pub agent: String, pub task: String, pub state: WorkerState,
    pub branch: String, pub worktree: PathBuf, pub last_message: String,
    pub answer: String, pub error: Option<String>, pub files_changed: Vec<String>,
    pub tokens: Option<u64>, pub cost_micros: Option<u64>, pub on_plan: bool,
}
// worktree.rs
pub fn create(root: &Path, id: &str, base: &str) -> Result<(PathBuf, String), String>; // (path, branch)
pub fn remove(root: &Path, path: &Path, branch: &str) -> Result<(), String>;
pub fn changed_files(worktree: &Path, base: &str) -> Result<Vec<String>, String>;
pub fn diff(worktree: &Path, base: &str) -> Result<String, String>;
// worker.rs
pub struct LaunchSpec { pub program: PathBuf, pub args: Vec<String>, pub env: Vec<(String, String)> }
pub enum WorkerEvent { Changed(WorkerSnapshot), Permission { worker: WorkerId, summary: String, tool: String, answer: tokio::sync::oneshot::Sender<bool> } }
pub struct WorkerHandle { /* id, cmd sender */ }
impl WorkerHandle {
    pub fn start(id: WorkerId, agent: &str, task: &str, spec: LaunchSpec, worktree: PathBuf, branch: String,
                 events: tokio::sync::mpsc::UnboundedSender<WorkerEvent>) -> WorkerHandle;
    pub fn message(&self, text: &str) -> Result<(), String>; // refused while Working
    pub fn stop(&self);
}
```

The worker runs `agent_client_protocol::Client.builder()` with handlers for `SessionNotification` (text → `last_message`/`answer`, tool calls counted, `usage_update` → tokens) and `RequestPermissionRequest` (→ `WorkerEvent::Permission`, awaits the oneshot; `true` → the first `allow_once` option, `false` → `reject_once`), and `connect_with(AcpAgent::new(AcpAgentConfig::new(program).args(args).envs(env)), |c| …)` whose closure: `initialize`, `session/new {cwd: worktree}`, `session/prompt {task}`, then loops on a command channel (`Message(text)` → another prompt; `Stop` → `session/cancel` then return). Each state change sends `WorkerEvent::Changed`. A connection error or the agent process ending sets `Failed` with the error text (Review Focus 1).

- [ ] **Step 1: Failing tests.** `worktree.rs`: in a real temp git repo with one commit, `create` makes `<parent>/<repo>-team/w1` on `xencode/team/w1`; a file written there shows in `changed_files` and `diff`; `remove` deletes both. `team_cli.rs`: `a_worker_runs_its_task_in_its_own_worktree` — start a worker whose `LaunchSpec` is `env!("CARGO_BIN_EXE_xencode") acp` with `XCODE_CONFIG_DIR` pointing at `settings()` (unreachable model); events reach `Done` with `answer` containing `127.0.0.1:9`, `worktree` exists and is on `xencode/team/w1`. `a_stopped_worker_says_stopped` (stalled settings; `stop()` → `Stopped` within 30 s). `a_worker_whose_agent_dies_is_failed` (stalled settings; kill the child `xencode acp` process found by its parent; → `Failed` with an error, worktree kept). `a_message_continues_the_same_session` (after `Done`, `message("again")` → `Working` → `Done` again).
- [ ] **Step 2: Run, see them fail** (compile errors).
- [ ] **Step 3: Implement** as in Interfaces. Git through `std::process::Command::new("git")` with `-C`.
- [ ] **Step 4: Run them; then clippy on Linux (WSL).**
- [ ] **Step 5:** no commit yet — Task 1 and Task 2 are `TM-1`.

### Task 2: The engine hosts a team (TM-1)

**Files:** `engine/proto.rs`, new `engine/team.rs`, `engine/mod.rs` (`handle` routes `ClientMsg::Team`), `engine/server.rs` (pumps worker events each tick), `engine/view.rs`. Test: `team_cli.rs`.

**Interfaces:**

```rust
// proto.rs
pub enum TeamRequest {
    Agents, Start { agent: String, task: String, base: Option<String> }, Status { id: Option<String> },
    Result { id: String }, Message { id: String, text: String }, Stop { id: String }, Merge { id: String },
}
ClientMsg::Team { req: u64, request: TeamRequest }
EngineMsg::TeamReply { req: u64, ok: bool, body: serde_json::Value }   // sent only to the asking window
// view.rs
pub team: Option<Vec<xencode_team_rs::WorkerSnapshot>>
```

`engine/team.rs`: `Team { workers: BTreeMap<String, (WorkerHandle, WorkerSnapshot)>, events: UnboundedReceiver<WorkerEvent>, next: u32, limit: usize, pending: HashMap<u64 approval id, oneshot::Sender<bool>> }`. `Team::request(&mut self, app, request) -> Result<Value, String>`; `Team::pump(&mut self, app)` drains events: `Changed` updates the snapshot; `Permission` pushes an engine approval (the same queue `ApprovalRequested` uses) with tool `"worker <id>: <tool>"` and the summary, keyed by its own approval id (Review Focus 5); `AnswerApproval` for such an id sends the oneshot. `Start` refuses past the limit and in Local Only posture for outside agents; Task 2 supports only `agent: "xencode"` (spec: `current_exe() acp`), outside agents come in Task 6.

- [ ] **Step 1: Failing test** `team_cli.rs`: start a real engine (the `Engine` helper pattern from `engine_cli.rs`), connect a window, send `Team{Start{agent:"xencode", task:"say hi"}}`, get `TeamReply{ok:true, body:{"id":"w1"}}`; watch `View.team` reach `w1` `Done`; `Status{id:"w9"}` → `ok:false` with "no worker w9"; `two_workers_approvals_are_separate` (Review Focus 5): two xencode workers whose model really asks for a `write_file` — a model is needed for that, so this test is `#[ignore = "needs XENCODE_LIVE_LLAMACPP_URL"]` and runs against a Colab server (`xencode colab up` from WSL), only when the owner asks; it checks the two approvals have different ids and answering one leaves the other waiting.
- [ ] **Step 2–4:** fail, implement, pass; Linux clippy.
- [ ] **Step 5: Commit `TM-1`** ("TM-1: the engine runs worker agents, each in its own worktree"), after `cargo test --workspace` and docs: `CHANGELOG.md` Unreleased entry, `NEXT_PLAN_TASKS.md` TM table.

### Task 3: The lead's tools (TM-2)

**Files:** `mcp_serve.rs` (`--team` adds `team_agents`, `team_start`, `team_status`, `team_result`, `team_message`, `team_stop`, `team_merge`), `main.rs` (`McpAction::Serve { team: bool }`). Test: `mcp_serve.rs` unit tests calling `invoke` against a real engine; `team_cli.rs` drives `xencode mcp serve --team` over stdio with the rmcp client.

The team tools connect to the engine for `--workspace` with `engine::link::connect_or_start_as(addr, start, "mcp <pid>")` once, keep the link, send `ClientMsg::Team{req, request}`, and wait (≤ 30 s, `team_merge` ≤ check timeout + 60 s) for the `TeamReply` with that `req`, skipping other messages (they belong to other windows' turns, as `M-7`'s drain fix learned). No engine and it cannot be started → the tool fails with "cannot reach the engine for <folder>: …" (Review Focus 4).

- [ ] **Step 1: Failing tests:** `invoke("team_start", {agent:"xencode", task:"say hi"})` → text with `w1`; `invoke("team_status", {id:"w1"})` eventually `done`; `invoke("team_status", {id:"nope"})` → failed reply "no worker nope"; the six file tools are still refused or allowed exactly as before.
- [ ] **Step 2–4.**
- [ ] **Step 5: Commit `TM-2`**, docs for `mcp serve --team` (CLI guide, regenerated completions from the WSL build).

### Task 4: The checked merge (TM-3)

**Files:** `xencode-team-rs/src/merge.rs`, `src/config.rs`; `engine/team.rs` (`Merge`). Test: unit tests in `merge.rs` with real repos.

```rust
pub enum Checks { Commands(Vec<String>), Cargo, None }
pub fn checks_for(root: &Path) -> Checks; // team.toml `checks = [...]` first, else Cargo.toml → Cargo, else None
pub enum MergeOutcome { Landed { commit: String }, ChecksFailed { output: String }, Conflict { files: Vec<String> },
                        NeedsPerson { why: String }, Refused { why: String } }
pub fn checked_merge(root: &Path, base: &str, branch: &str, checks: &Checks, timeout: Duration) -> MergeOutcome;
```

Steps of `checked_merge`: refuse if `git -C root status --porcelain` is not empty (Review Focus 3); note base's commit; scratch worktree at that commit (`<repo>-team/merge-<n>`); `git merge --no-ff branch` there (conflict → list `git diff --name-only --diff-filter=U`); run checks there (`Cargo` → `xencode_analysis_rs::toolchain::run_checklist(scratch, &[], secs)`, `Commands` → each with `sh -c`/`cmd /C`, timeout); green → if base still at the noted commit, `git -C root merge --ff-only <scratch HEAD>` (the person's checked-out branch moves; their files change only now), else repeat once from the new base (Review Focus 2), still moving → `Refused`; `None` → `NeedsPerson` (the engine turns it into an approval: "land w1 with no checks?"). Remove the scratch worktree in every outcome. Write an audit line (`.xencode/audit.jsonl` through the existing audit writer) for every outcome.

- [ ] **Step 1: Failing tests** (all real git, real commands): green `Commands(["git --version"])` lands; red `Commands(["exit 1"])` → `ChecksFailed`, base unchanged; conflicting edits → `Conflict` with the file; a commit to base between the scratch merge and landing (inject by making the check command itself commit to base: `git -C <root> commit --allow-empty -m moved` the first time) → retried and lands on the new base; dirty working copy → `Refused`; no checks → `NeedsPerson`.
- [ ] **Step 2–4.**
- [ ] **Step 5: Commit `TM-3`**.

### Task 5: Team panel and badge (TM-4)

**Files:** `worker_panel.rs` (a `Team` section rendering `View.team`; keys Enter/s/m/a), `app.rs` (keys send `ClientMsg::Team` / `AnswerApproval`), `xencode-live-rs` (`LiveSource::Worker`; the engine's live status counts a `NeedsYou` worker). Test: a real `App` window linked to a real engine (the `window_onto` pattern in `engine_cli.rs`) shows a started worker's row; pressing `s` stops it.

- [ ] Steps 1–5 as above; commit `TM-4`.

### Task 6: The outside agents, sign-in and cost (TM-5)

**Files:** `xencode-team-rs/src/agents.rs`, `main.rs` (`team login-optin <agent>`, `team clean`), `engine/team.rs` (`Agents`, outside `Start`).

```rust
pub struct AgentSpec { pub name: &'static str, pub program: &'static str, pub args: &'static [&'static str],
                       pub key_vars: &'static [&'static str], pub terms_line: &'static str, pub terms_url: &'static str }
pub const AGENTS: &[AgentSpec]; // claude-code, codex, gemini, antigravity, xencode — values from the spec's §2 table
pub fn availability(spec: &AgentSpec, optins: &[String]) -> Availability; // Missing{install} | Key{var} | Login | NoSignIn{fix}
```

Keys are read from the environment and from xencode's secret store (the same lookup `xencode config` uses for provider keys) and passed only through the worker's `env`. A login is used only when `team-optins.json` in xencode's settings folder names the agent — written by `xencode team login-optin <agent>` after printing `terms_line` and `terms_url` and reading a `yes`. Cost: `usage_update` tokens × the price table (`xencode_context_rs::pricing`) for key sign-in; `on_plan = true` for a login.

- [ ] **Step 1: Failing tests:** `availability` for a program not on PATH → `Missing` with the install line; with `GEMINI_API_KEY` set (fake value `FAKE-NOT-A-REAL-KEY`) and `gemini` present → `Key`; `login-optin` refused without `yes`; live `#[ignore = "needs ANTHROPIC_API_KEY and costs money"]` etc. per agent: start, `Done`, answer non-empty.
- [ ] **Step 2–4;** live runs only when the owner provides keys and asks.
- [ ] **Step 5: Commit `TM-5`**.

### Task 7: Manuals and the whole flow (TM-6)

`README.md`, `QUICK_START.md` ("Lead several agents"), `CLI_GUIDE.md` (`mcp serve --team`, `team login-optin`, `team clean`, `.xencode/team.toml`), `docs/USER_MANUAL.md`, `CHANGELOG.md`, `NEXT_PLAN_TASKS.md`. If the owner provides an API key: a real lead (Claude Code) starts two xencode workers on a test repo, follows them, and lands one merge — watched, with what happened written down.

- [ ] Steps: docs from `--help` and observed behaviour only; workspace tests; counts; commit `TM-6`; `cargo clean` if `rust/target` > 20 GiB.
