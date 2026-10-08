//! Background task registry (Milestone D, D1-01).
//!
//! Two layers: [`TaskStore`] is pure state (records, capped output tail,
//! status transitions) and fully unit-testable without spawning anything;
//! [`TaskManager`] sits on top and owns the real `tokio::process` children.
//! Output is drained by reader tasks into a shared buffer so `poll` never
//! blocks on a chatty command.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

/// Lines kept per task; the oldest lines are dropped first.
pub const MAX_OUTPUT_LINES: usize = 500;
/// Default wall-clock limit for a background command.
pub const DEFAULT_TASK_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Running,
    Exited(i32),
    Killed,
    TimedOut,
}

impl TaskStatus {
    pub fn label(&self) -> String {
        match self {
            TaskStatus::Running => "running".to_string(),
            TaskStatus::Exited(code) => format!("exited({code})"),
            TaskStatus::Killed => "killed".to_string(),
            TaskStatus::TimedOut => "timed out".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskRecord {
    pub id: u64,
    pub name: String,
    pub command: String,
    pub pid: Option<u32>,
    pub status: TaskStatus,
    pub started_at: u64,
    pub finished_at: Option<u64>,
    output: Vec<String>,
}

impl TaskRecord {
    fn new(id: u64, name: String, command: String) -> Self {
        Self {
            id,
            name,
            command,
            pid: None,
            status: TaskStatus::Running,
            started_at: unix_now(),
            finished_at: None,
            output: Vec::new(),
        }
    }

    pub fn output(&self) -> &[String] {
        &self.output
    }

    /// Append a line, evicting oldest lines past the cap.
    pub fn push_output(&mut self, line: impl Into<String>) {
        self.output.push(line.into());
        let excess = self.output.len().saturating_sub(MAX_OUTPUT_LINES);
        if excess > 0 {
            self.output.drain(..excess);
        }
    }

    /// Record a terminal status exactly once; late polls are no-ops.
    pub fn finish(&mut self, status: TaskStatus) {
        if matches!(self.status, TaskStatus::Running) {
            self.status = status;
            self.finished_at = Some(unix_now());
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Pure id/task bookkeeping, shared by the manager and (later) the TUI panel.
#[derive(Default)]
pub struct TaskStore {
    next_id: u64,
    tasks: Vec<TaskRecord>,
}

impl TaskStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            tasks: Vec::new(),
        }
    }

    pub fn allocate_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn insert(&mut self, record: TaskRecord) {
        if record.id >= self.next_id {
            self.next_id = record.id + 1;
        }
        self.tasks.push(record);
    }

    pub fn get(&self, id: u64) -> Option<&TaskRecord> {
        self.tasks.iter().find(|t| t.id == id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut TaskRecord> {
        self.tasks.iter_mut().find(|t| t.id == id)
    }

    pub fn list(&self) -> &[TaskRecord] {
        &self.tasks
    }

    pub fn append_output(&mut self, id: u64, line: impl Into<String>) {
        if let Some(task) = self.get_mut(id) {
            task.push_output(line);
        }
    }

    /// Remove a finished task. Running tasks are refused — stop them first.
    pub fn remove(&mut self, id: u64) -> Result<TaskRecord, TaskError> {
        match self.tasks.iter().position(|t| t.id == id) {
            None => Err(TaskError::NotFound(id)),
            Some(idx) if matches!(self.tasks[idx].status, TaskStatus::Running) => {
                Err(TaskError::StillRunning(id))
            }
            Some(idx) => Ok(self.tasks.remove(idx)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TaskError {
    #[error("no such task: {0}")]
    NotFound(u64),
    #[error("task {0} is still running (stop it first)")]
    StillRunning(u64),
    #[error("task {0} already finished")]
    AlreadyFinished(u64),
    #[error("spawn failed: {0}")]
    Spawn(std::io::Error),
}

/// Owns real child processes keyed by task id.
pub struct TaskManager {
    store: TaskStore,
    children: HashMap<u64, Child>,
    deadlines: HashMap<u64, tokio::time::Instant>,
    watchdogs: HashMap<u64, tokio::task::JoinHandle<()>>,
    timed_out: HashMap<u64, Arc<AtomicBool>>,
    outputs: HashMap<u64, Arc<Mutex<Vec<String>>>>,
    /// The tasks reading each child's stdout and stderr. A child can be reaped
    /// before they reach the end of its output, so a finished task waits for
    /// them (see [`READER_SETTLE`]) before its output is read.
    readers: HashMap<u64, Vec<tokio::task::JoinHandle<()>>>,
}

/// Longest a poll waits, once a task has finished, for its output readers to
/// reach the end of what it wrote.
const READER_SETTLE: Duration = Duration::from_secs(2);

impl Default for TaskManager {
    fn default() -> Self {
        Self::new()
    }
}

/// What to actually execute for a background task: a program and its argument
/// list. Normally `SpawnSpec::shell` (`sh -c command`); the agent passes a
/// `bwrap …` spec when the SE-7 sandbox is on, so isolation reaches background
/// tasks too while the recorded command stays the human-readable one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnSpec {
    pub program: String,
    pub args: Vec<String>,
}

impl SpawnSpec {
    /// The plain `sh -c command` spawn.
    pub fn shell(command: &str) -> SpawnSpec {
        SpawnSpec {
            program: "sh".to_string(),
            args: vec!["-c".to_string(), command.to_string()],
        }
    }
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            store: TaskStore::new(),
            children: HashMap::new(),
            deadlines: HashMap::new(),
            watchdogs: HashMap::new(),
            timed_out: HashMap::new(),
            outputs: HashMap::new(),
            readers: HashMap::new(),
        }
    }

    pub fn store(&self) -> &TaskStore {
        &self.store
    }

    /// Spawn `command` through `sh -c` and start draining its output.
    pub async fn start(&mut self, name: &str, command: &str) -> Result<u64, TaskError> {
        self.start_with_timeout(name, command, DEFAULT_TASK_TIMEOUT)
            .await
    }

    /// Spawn a command with a caller-selected wall-clock limit.
    pub async fn start_with_timeout(
        &mut self,
        name: &str,
        command: &str,
        timeout: Duration,
    ) -> Result<u64, TaskError> {
        self.start_with_cwd_and_timeout(name, command, None, timeout)
            .await
    }

    /// Like [`start`](Self::start) but runs the command in `cwd` (D3-03:
    /// lets a task build/test inside a chosen git worktree). A missing
    /// directory surfaces as `Spawn` from the OS.
    pub async fn start_with_cwd(
        &mut self,
        name: &str,
        command: &str,
        cwd: Option<&std::path::Path>,
    ) -> Result<u64, TaskError> {
        self.start_with_cwd_and_timeout(name, command, cwd, DEFAULT_TASK_TIMEOUT)
            .await
    }

    /// Spawn a command in a directory with a caller-selected wall-clock limit.
    pub async fn start_with_cwd_and_timeout(
        &mut self,
        name: &str,
        command: &str,
        cwd: Option<&std::path::Path>,
        timeout: Duration,
    ) -> Result<u64, TaskError> {
        self.start_spawning(name, command, cwd, timeout, &SpawnSpec::shell(command))
            .await
    }

    /// Like [`start_with_cwd_and_timeout`](Self::start_with_cwd_and_timeout) but
    /// spawning through a caller-resolved program and argument list instead of a
    /// bare `sh -c`. `command` is still what the record and `poll` display;
    /// `spawn` is what actually executes. The agent uses this to run a background
    /// task inside the SE-7 `bwrap` sandbox while keeping the recorded command
    /// readable.
    pub async fn start_spawning(
        &mut self,
        name: &str,
        command: &str,
        cwd: Option<&std::path::Path>,
        timeout: Duration,
        spawn: &SpawnSpec,
    ) -> Result<u64, TaskError> {
        let mut cmd = Command::new(&spawn.program);
        cmd.args(&spawn.args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        if let Some(cwd) = cwd {
            cmd.current_dir(cwd);
        }
        let mut child = cmd.spawn().map_err(TaskError::Spawn)?;

        let id = self.store.allocate_id();
        let mut record = TaskRecord::new(id, name.to_string(), command.to_string());
        record.pid = child.id();

        // Merge stdout+stderr into one capped tail via reader tasks, so
        // `poll` only has to look at `try_wait` and the shared buffer.
        let buffer: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let mut readers = Vec::new();
        if let Some(stream) = child.stdout.take() {
            readers.push(spawn_reader(stream, Arc::clone(&buffer)));
        }
        if let Some(stream) = child.stderr.take() {
            readers.push(spawn_reader(stream, Arc::clone(&buffer)));
        }

        self.store.insert(record);
        self.readers.insert(id, readers);
        self.children.insert(id, child);
        let deadline = tokio::time::Instant::now() + timeout;
        self.deadlines.insert(id, deadline);
        let timed_out = Arc::new(AtomicBool::new(false));
        let watchdog_flag = Arc::clone(&timed_out);
        #[cfg(unix)]
        if let Some(pid) = self.store.get(id).and_then(|task| task.pid) {
            self.watchdogs.insert(
                id,
                tokio::spawn(async move {
                    tokio::time::sleep_until(deadline).await;
                    watchdog_flag.store(true, Ordering::Release);
                    unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
                }),
            );
        }
        // Windows has no process groups to signal, so the timeout ends the
        // task's own process; without this a task there would never time out.
        #[cfg(not(unix))]
        if let Some(pid) = self.store.get(id).and_then(|task| task.pid) {
            self.watchdogs.insert(
                id,
                tokio::spawn(async move {
                    tokio::time::sleep_until(deadline).await;
                    watchdog_flag.store(true, Ordering::Release);
                    let _ = tokio::task::spawn_blocking(move || crate::sys::terminate(pid)).await;
                }),
            );
        }
        self.timed_out.insert(id, timed_out);
        self.outputs.insert(id, buffer);
        Ok(id)
    }

    /// Reap the child if it exited and return an up-to-date snapshot.
    pub async fn poll(&mut self, id: u64) -> Result<TaskRecord, TaskError> {
        self.reap(id)?;
        if self
            .timed_out
            .get(&id)
            .is_some_and(|flag| flag.load(Ordering::Acquire))
            || self
                .deadlines
                .get(&id)
                .is_some_and(|deadline| tokio::time::Instant::now() >= *deadline)
        {
            self.kill_task(id, TaskStatus::TimedOut).await?;
        }
        self.settle_output(id).await;
        self.drain_output(id);
        self.snapshot(id)
    }

    /// Once a task has finished, wait for its output readers to reach the end of
    /// what the child wrote, so the last lines are not lost because the exit was
    /// seen first. Bounded by [`READER_SETTLE`]: a grandchild left running in the
    /// background can hold the pipe open, and a poll must not wait on it.
    async fn settle_output(&mut self, id: u64) {
        if self.children.contains_key(&id) {
            return; // still running: there is no end of output to wait for yet
        }
        let Some(readers) = self.readers.remove(&id) else {
            return;
        };
        let deadline = tokio::time::Instant::now() + READER_SETTLE;
        for reader in readers {
            let _ = tokio::time::timeout_at(deadline, reader).await;
        }
    }

    /// Kill a running task. `AlreadyFinished` for anything already reaped —
    /// callers treat that as idempotent success.
    pub async fn stop(&mut self, id: u64) -> Result<(), TaskError> {
        self.reap(id)?;
        self.kill_task(id, TaskStatus::Killed).await
    }

    async fn kill_task(&mut self, id: u64, reason: TaskStatus) -> Result<(), TaskError> {
        if let Some(watchdog) = self.watchdogs.remove(&id) {
            watchdog.abort();
        }
        let Some(child) = self.children.get_mut(&id) else {
            return match self.store.get(id) {
                Some(_) => Err(TaskError::AlreadyFinished(id)),
                None => Err(TaskError::NotFound(id)),
            };
        };
        #[cfg(unix)]
        if let Some(pid) = child.id() {
            // The command starts in a fresh process group; signal the group so
            // shell-spawned commands and vendor descendants are included.
            unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
        }
        #[cfg(not(unix))]
        let _ = child.start_kill();
        let status = child.wait().await.ok();
        if let (Some(status), Some(task)) = (status, self.store.get_mut(id)) {
            if matches!(task.status, TaskStatus::Running) {
                if status.success() {
                    task.finish(TaskStatus::Exited(status.code().unwrap_or(0)));
                } else {
                    task.finish(reason);
                }
            }
        }
        self.drain_output(id);
        self.children.remove(&id);
        self.deadlines.remove(&id);
        self.timed_out.remove(&id);
        Ok(())
    }

    pub fn list(&self) -> &[TaskRecord] {
        self.store.list()
    }

    pub fn snapshot(&self, id: u64) -> Result<TaskRecord, TaskError> {
        self.store.get(id).cloned().ok_or(TaskError::NotFound(id))
    }

    /// Remove a finished task's record entirely.
    pub fn remove(&mut self, id: u64) -> Result<TaskRecord, TaskError> {
        // Checked before touching the child map: dropping a `kill_on_drop`
        // child would silently kill a still-running task.
        if let Some(task) = self.store.get(id) {
            if matches!(task.status, TaskStatus::Running) {
                return Err(TaskError::StillRunning(id));
            }
        }
        self.children.remove(&id);
        self.deadlines.remove(&id);
        self.timed_out.remove(&id);
        if let Some(watchdog) = self.watchdogs.remove(&id) {
            watchdog.abort();
        }
        self.outputs.remove(&id);
        self.readers.remove(&id);
        self.store.remove(id)
    }

    fn reap(&mut self, id: u64) -> Result<(), TaskError> {
        let Some(child) = self.children.get_mut(&id) else {
            return if self.store.get(id).is_some() {
                Ok(())
            } else {
                Err(TaskError::NotFound(id))
            };
        };
        if let Ok(Some(status)) = child.try_wait() {
            let code = status.code().unwrap_or(-1);
            let timed_out = self
                .timed_out
                .get(&id)
                .is_some_and(|flag| flag.load(Ordering::Acquire));
            if let Some(task) = self.store.get_mut(id) {
                task.finish(if timed_out {
                    TaskStatus::TimedOut
                } else {
                    TaskStatus::Exited(code)
                });
            }
            self.children.remove(&id);
            self.deadlines.remove(&id);
            self.timed_out.remove(&id);
            if let Some(watchdog) = self.watchdogs.remove(&id) {
                watchdog.abort();
            }
        }
        Ok(())
    }

    fn drain_output(&mut self, id: u64) {
        let Some(buffer) = self.outputs.get(&id) else {
            return;
        };
        let pending: Vec<String> = {
            let mut buf = buffer.lock().unwrap_or_else(|e| e.into_inner());
            buf.drain(..).collect()
        };
        for line in pending {
            self.store.append_output(id, line);
        }
    }
}

impl Drop for TaskManager {
    fn drop(&mut self) {
        for watchdog in self.watchdogs.values() {
            watchdog.abort();
        }
        #[cfg(unix)]
        for child in self.children.values() {
            if let Some(pid) = child.id() {
                // `kill_on_drop` handles the shell itself; signal the group
                // here as well so its descendants cannot outlive the manager.
                unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
            }
        }
    }
}

fn spawn_reader<S>(stream: S, buffer: Arc<Mutex<Vec<String>>>) -> tokio::task::JoinHandle<()>
where
    S: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut lines = BufReader::new(stream).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let mut buf = buffer.lock().unwrap_or_else(|e| e.into_inner());
            buf.push(line);
            let excess = buf.len().saturating_sub(MAX_OUTPUT_LINES);
            if excess > 0 {
                buf.drain(..excess);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: u64) -> TaskRecord {
        TaskRecord::new(id, "test".into(), "true".into())
    }

    /// Whether a pid can still run code. A killed child that has not yet been
    /// collected reads as `Z`, and Linux has a second transient before that —
    /// `X`, exit-dead — between a thread exiting and its parent noticing. Both
    /// are terminal; `R` and `S` are not, and a process in either of those was
    /// not killed.
    #[cfg(unix)]
    fn state_is_dead(state: Option<&str>) -> bool {
        matches!(state, Some("Z") | Some("X"))
    }

    #[cfg(unix)]
    fn pid_is_dead(pid: i32) -> bool {
        match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
            Err(_) => true,
            Ok(contents) => state_is_dead(contents.split_whitespace().nth(2)),
        }
    }

    /// Poll until the task leaves Running. Output readers run on their own
    /// task, so lines can land one poll after the exit is reaped.
    async fn await_exit(m: &mut TaskManager, id: u64) -> TaskRecord {
        for _ in 0..100 {
            let rec = m.poll(id).await.unwrap();
            if !matches!(rec.status, TaskStatus::Running) {
                return rec;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("task {id} never exited");
    }

    #[test]
    fn push_output_evicts_oldest_past_cap() {
        let mut rec = record(1);
        for i in 0..(MAX_OUTPUT_LINES + 100) {
            rec.push_output(format!("l{i}"));
        }
        assert_eq!(rec.output().len(), MAX_OUTPUT_LINES);
        assert_eq!(rec.output().first().unwrap(), "l100");
        assert_eq!(
            rec.output().last().unwrap(),
            &format!("l{}", MAX_OUTPUT_LINES + 99)
        );
    }

    #[test]
    fn finish_only_from_running_and_stamps_time() {
        let mut rec = record(1);
        assert_eq!(rec.status, TaskStatus::Running);
        assert!(rec.finished_at.is_none());
        rec.finish(TaskStatus::Exited(0));
        assert_eq!(rec.status, TaskStatus::Exited(0));
        assert!(rec.finished_at.is_some());
        rec.finish(TaskStatus::Killed);
        assert_eq!(rec.status, TaskStatus::Exited(0));
    }

    #[test]
    fn status_labels() {
        assert_eq!(TaskStatus::Running.label(), "running");
        assert_eq!(TaskStatus::Exited(2).label(), "exited(2)");
        assert_eq!(TaskStatus::Killed.label(), "killed");
    }

    #[test]
    fn store_ids_never_reuse_and_insert_bumps_counter() {
        let mut store = TaskStore::new();
        assert_eq!(store.allocate_id(), 1);
        assert_eq!(store.allocate_id(), 2);
        store.insert(record(10));
        assert_eq!(store.allocate_id(), 11);
    }

    #[test]
    fn store_remove_refuses_running_accepts_finished() {
        let mut store = TaskStore::new();
        store.insert(record(1));
        assert!(matches!(store.remove(1), Err(TaskError::StillRunning(1))));
        assert!(store.get(1).is_some(), "refused remove must keep record");
        let mut rec = record(2);
        rec.finish(TaskStatus::Exited(0));
        store.insert(rec);
        store.remove(2).unwrap();
        assert!(store.get(2).is_none());
        assert!(matches!(store.remove(99), Err(TaskError::NotFound(99))));
    }

    #[test]
    fn store_append_output_silently_ignores_unknown_id() {
        let mut store = TaskStore::new();
        store.append_output(42, "dropped");
        assert!(store.list().is_empty());
    }

    #[tokio::test]
    async fn spawn_echo_exits_zero_with_output() {
        let mut m = TaskManager::new();
        let id = m.start("echo", "echo hi").await.unwrap();
        assert_eq!(id, 1);
        assert!(m.list()[0].pid.is_some());
        let mut rec = await_exit(&mut m, id).await;
        for _ in 0..100 {
            if rec.output().contains(&"hi".to_string()) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            rec = m.poll(id).await.unwrap();
        }
        assert_eq!(rec.status, TaskStatus::Exited(0));
        assert_eq!(rec.output(), ["hi"]);
    }

    /// D3-03: `start_with_cwd` runs the command inside the given directory.
    #[tokio::test]
    async fn start_with_cwd_runs_in_that_directory() {
        let dir = std::env::temp_dir().join(format!("xencode-cwd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut m = TaskManager::new();
        let id = m.start_with_cwd("pwd", "pwd", Some(&dir)).await.unwrap();
        let mut rec = await_exit(&mut m, id).await;
        for _ in 0..100 {
            if !rec.output().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            rec = m.poll(id).await.unwrap();
        }
        // macOS leaves /tmp symlinked; compare against the canonical path.
        let got = std::path::PathBuf::from(rec.output()[0].trim());
        assert_eq!(got.canonicalize().unwrap(), dir.canonicalize().unwrap());
        // A directory that doesn't exist must surface as a spawn error.
        assert!(matches!(
            m.start_with_cwd("nope", "pwd", Some(&dir.join("missing")))
                .await,
            Err(TaskError::Spawn(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn exit_code_is_captured() {
        let mut m = TaskManager::new();
        let id = m.start("fail", "exit 3").await.unwrap();
        let rec = await_exit(&mut m, id).await;
        assert_eq!(rec.status, TaskStatus::Exited(3));
    }

    #[tokio::test]
    async fn stop_running_task_kills_it() {
        let mut m = TaskManager::new();
        let id = m.start("sleep", "sleep 30").await.unwrap();
        assert_eq!(m.poll(id).await.unwrap().status, TaskStatus::Running);
        m.stop(id).await.unwrap();
        assert_eq!(m.snapshot(id).unwrap().status, TaskStatus::Killed);
        // Finished tasks are removable; a second stop is AlreadyFinished.
        assert!(matches!(
            m.stop(id).await,
            Err(TaskError::AlreadyFinished(1))
        ));
        m.remove(id).unwrap();
        assert!(m.list().is_empty());
    }

    /// The first poll that reports a task finished already holds everything it
    /// printed. The exit used to be seen before the readers reached the end of
    /// the pipe, so a finished task could report no output at all — which is how
    /// `orchestrator retry` returned an empty tail on CI once. Thirty runs in a
    /// row, because a race shows up as a rate, not as one failure.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_finished_task_reports_all_its_output_on_the_first_poll() {
        for attempt in 0..30 {
            let mut m = TaskManager::new();
            let id = m.start("print", "printf 'a\\nb\\nc\\n'").await.unwrap();
            let rec = loop {
                let rec = m.poll(id).await.unwrap();
                if !matches!(rec.status, TaskStatus::Running) {
                    break rec;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            };
            assert_eq!(rec.output(), ["a", "b", "c"], "attempt {attempt}");
            m.remove(id).unwrap();
        }
    }

    // The command backgrounds a POSIX `sleep` and the death check reads `/proc`,
    // so this only measures anything on Linux.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn wall_clock_limit_kills_task_and_marks_timeout() {
        let mut m = TaskManager::new();
        let id = m
            .start_with_timeout(
                "bounded",
                "sleep 30 & echo $!; wait",
                Duration::from_millis(100),
            )
            .await
            .unwrap();
        // Bounded, so a shell that never prints the pid fails the test instead
        // of hanging the whole suite.
        let mut child_pid = None;
        for _ in 0..500 {
            let rec = m.poll(id).await.unwrap();
            if let Some(pid) = rec
                .output()
                .last()
                .and_then(|l| l.trim().parse::<i32>().ok())
            {
                child_pid = Some(pid);
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let child_pid = child_pid.expect("the task printed its child's pid within 5 seconds");
        tokio::time::sleep(Duration::from_millis(140)).await;
        // The watchdog must enforce the limit without depending on another
        // poll call to notice that the deadline has passed.
        assert!(
            pid_is_dead(child_pid),
            "the time limit left the descendant runnable"
        );
        let rec = m.poll(id).await.unwrap();
        assert_eq!(rec.status, TaskStatus::TimedOut);
        assert!(rec.finished_at.is_some());
        m.remove(id).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn exit_dead_and_zombie_are_the_states_that_mean_a_child_is_done() {
        assert!(state_is_dead(Some("Z")));
        assert!(state_is_dead(Some("X")));
        assert!(!state_is_dead(Some("R")));
        assert!(!state_is_dead(Some("S")));
        assert!(!state_is_dead(Some("D")));
        assert!(!state_is_dead(None));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stop_kills_shell_descendants_in_its_process_group() {
        let mut m = TaskManager::new();
        let id = m
            .start("parent and child", "sleep 30 & echo $!; wait")
            .await
            .unwrap();
        let child_pid = loop {
            let rec = m.poll(id).await.unwrap();
            if let Some(line) = rec.output().last() {
                if let Ok(pid) = line.trim().parse::<i32>() {
                    break pid;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        m.stop(id).await.unwrap();
        for _ in 0..100 {
            if pid_is_dead(child_pid) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            pid_is_dead(child_pid),
            "descendant {child_pid} was still runnable after its process group was killed"
        );
    }

    #[tokio::test]
    async fn remove_refuses_running_task() {
        let mut m = TaskManager::new();
        let id = m.start("sleep", "sleep 30").await.unwrap();
        assert!(matches!(m.remove(id), Err(TaskError::StillRunning(1))));
        // Cleanup: actually kill it so the test process is gone.
        m.stop(id).await.unwrap();
    }

    /// The full natural lifecycle: start → exit(0) → remove. The stop-path
    /// tests only remove killed tasks.
    #[tokio::test]
    async fn naturally_finished_task_can_be_removed() {
        let mut m = TaskManager::new();
        let id = m.start("bye", "echo bye").await.unwrap();
        let rec = await_exit(&mut m, id).await;
        assert_eq!(rec.status, TaskStatus::Exited(0));
        m.remove(id).unwrap();
        assert!(m.list().is_empty());
        assert!(matches!(m.remove(id), Err(TaskError::NotFound(_))));
    }

    #[tokio::test]
    async fn unknown_ids_report_not_found() {
        let mut m = TaskManager::new();
        assert!(matches!(m.poll(7).await, Err(TaskError::NotFound(7))));
        assert!(matches!(m.stop(7).await, Err(TaskError::NotFound(7))));
        assert!(matches!(m.snapshot(7), Err(TaskError::NotFound(7))));
        assert!(matches!(m.remove(7), Err(TaskError::NotFound(7))));
    }
}
