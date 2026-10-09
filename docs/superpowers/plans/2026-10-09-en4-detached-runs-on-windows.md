# EN-4 — `xencode run --detach` on Windows — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `xencode run --detach`, `--resume --detach` and `--stop` work on Windows, using the detached launcher the badge and the engine already share.

**Architecture:** On Unix a detached run forks and the child runs the worker in-process (AE-7); that stays. Windows has no fork, so there the run starts the same binary again as its hidden worker (`xencode run --child <id> --xencode-dir <dir>`, which already exists), detached from the console, in the run's working folder, with its output appended to the run's log. Stopping writes the `stop` file as today and then ends the worker with `TerminateProcess`, the Windows counterpart of the `SIGTERM` Unix sends.

**Tech Stack:** Rust, `std::process::Command` with Windows creation flags, `windows-sys` (already a dependency of `xencode-tui-rs` on Windows).

**Spec:** `docs/superpowers/specs/2026-10-09-engine-process-design.md` §5 (`EN-4`: "A detached-process launcher that works on Windows, shared by the badge, the engine and `xencode run --detach`"; visible result "`xencode run --detach` works on Windows").

## Global Constraints

- No mocks: real processes, real files, a real `xencode` binary in the integration tests.
- Plan-item commits in plain English; never push; `XCODE_CONFIG_DIR` set for every test run; `-j 4`; leftover-process check.
- Unix keeps forking: AE-7 chose not to re-execute the binary there, and nothing here changes that.

## Review Focus

1. A worker that outlives the command that started it — closing the console must not end it.
2. A log file that already exists (a resumed run) — appended to, never truncated.
3. A stop for a worker that already ended — reported, no error.
4. A project path with spaces — passed as one argument.
5. A run started from a folder other than the project — the worker's working folder is the run's tree, and `--xencode-dir` points at the project's `.xencode`.

---

### Task 1: the launcher takes a working folder and a log

**Files:** `rust/crates/xencode-live-rs/src/lib.rs`.

- `pub fn spawn_detached_logged(exe: &Path, args: &[&str], cwd: &Path, log: &Path) -> io::Result<u32>`: as `spawn_detached`, but runs in `cwd` and appends standard output and standard error to `log` (created if missing).
- `spawn_detached` keeps its behaviour.

Tests: a real short-lived process started with a log that already holds a line writes its own line after it; the working folder it reports is `cwd` (a folder with a space in its name).

Commit: `EN-4: the detached launcher can run in a given folder and write to a log`.

### Task 2: `xencode run --detach` on Windows

**Files:** `rust/crates/xencode-tui-rs/src/detached.rs` (`spawn_child` for Windows).

- `#[cfg(windows)] spawn_child(exe, run_id, xencode_dir, cwd, log)` = `spawn_detached_logged(exe, ["run", "--child", run_id, "--xencode-dir", xencode_dir], cwd, log)`.
- Other non-Unix platforms keep the refusal.

Tests (`xencode-cli/tests/run_detach_cli.rs`, real binary, settings with a model server address nobody listens on): `run --detach` prints the run id and pid and returns at once; the run then finishes on its own and `run --show <id>` reports it ended with the connection error in words; the log holds the worker's output. On Unix the same test exercises the fork path.

Commit: `EN-4: xencode run --detach works on Windows by starting the worker as its own process`.

### Task 3: `xencode run --stop` on Windows

**Files:** `rust/crates/xencode-tui-rs/src/detached.rs` (`stop_child`).

- On Windows, after writing the `stop` file, open the process with `PROCESS_TERMINATE` and call `TerminateProcess`; then wait as today. A process that is already gone is reported as stopped.

Tests: a real long-lived process (`ping -n 30 127.0.0.1`) in a run folder: `stop_child` ends it within the wait, the `stop` file exists, and the status reads stopped; `stop_child` on a pid that has already exited says so without error.

Commit: `EN-4: xencode run --stop ends a detached run on Windows`.

### Task 4: watch it, document it, close out

- Live: `xencode run --detach` against a real llama.cpp server with a task that takes several rounds; close the console that started it; watch the log grow and the run finish; start another and stop it with `--stop`.
- `CLI_GUIDE.md` (`run --detach` on Windows), `README.md`, `CHANGELOG.md`, `NEXT_PLAN_TASKS.md` (EN-4 and DK-3 checked with what was watched).

Commit: `EN-4: manuals, plan and changelog for detached runs on Windows`.
