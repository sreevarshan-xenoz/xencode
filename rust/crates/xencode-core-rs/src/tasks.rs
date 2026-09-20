//! Background task registry (Milestone D, D1-01).
//!
//! Two layers: [`TaskStore`] is pure state (records, capped output tail,
//! status transitions) and fully unit-testable without spawning anything;
//! [`TaskManager`] sits on top and owns the real `tokio::process` children.
//! Output is drained by reader tasks into a shared buffer so `poll` never
//! blocks on a chatty command.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

/// Lines kept per task; the oldest lines are dropped first.
pub const MAX_OUTPUT_LINES: usize = 500;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Running,
    Exited(i32),
    Killed,
}

impl TaskStatus {
    pub fn label(&self) -> String {
        match self {
            TaskStatus::Running => "running".to_string(),
            TaskStatus::Exited(code) => format!("exited({code})"),
            TaskStatus::Killed => "killed".to_string(),
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
    outputs: HashMap<u64, Arc<Mutex<Vec<String>>>>,
}

impl Default for TaskManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskManager {
    pub fn new() -> Self {
        Self {
            store: TaskStore::new(),
            children: HashMap::new(),
            outputs: HashMap::new(),
        }
    }

    pub fn store(&self) -> &TaskStore {
        &self.store
    }

    /// Spawn `command` through `sh -c` and start draining its output.
    pub async fn start(&mut self, name: &str, command: &str) -> Result<u64, TaskError> {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(TaskError::Spawn)?;

        let id = self.store.allocate_id();
        let mut record = TaskRecord::new(id, name.to_string(), command.to_string());
        record.pid = child.id();

        // Merge stdout+stderr into one capped tail via reader tasks, so
        // `poll` only has to look at `try_wait` and the shared buffer.
        let buffer: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        if let Some(stream) = child.stdout.take() {
            spawn_reader(stream, Arc::clone(&buffer));
        }
        if let Some(stream) = child.stderr.take() {
            spawn_reader(stream, Arc::clone(&buffer));
        }

        self.store.insert(record);
        self.children.insert(id, child);
        self.outputs.insert(id, buffer);
        Ok(id)
    }

    /// Reap the child if it exited and return an up-to-date snapshot.
    pub async fn poll(&mut self, id: u64) -> Result<TaskRecord, TaskError> {
        self.reap(id)?;
        self.drain_output(id);
        self.snapshot(id)
    }

    /// Kill a running task. `AlreadyFinished` for anything already reaped —
    /// callers treat that as idempotent success.
    pub async fn stop(&mut self, id: u64) -> Result<(), TaskError> {
        let Some(child) = self.children.get_mut(&id) else {
            return match self.store.get(id) {
                Some(_) => Err(TaskError::AlreadyFinished(id)),
                None => Err(TaskError::NotFound(id)),
            };
        };
        let _ = child.start_kill();
        self.reap(id)?;
        // If it exited before the kill landed, keep the real exit status;
        // otherwise label it Killed.
        if let Some(task) = self.store.get_mut(id) {
            if matches!(task.status, TaskStatus::Running) {
                task.status = TaskStatus::Killed;
                task.finished_at = Some(unix_now());
            }
        }
        self.drain_output(id);
        self.children.remove(&id);
        Ok(())
    }

    pub fn list(&self) -> &[TaskRecord] {
        self.store.list()
    }

    pub fn snapshot(&self, id: u64) -> Result<TaskRecord, TaskError> {
        self.store
            .get(id)
            .cloned()
            .ok_or(TaskError::NotFound(id))
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
        self.outputs.remove(&id);
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
            if let Some(task) = self.store.get_mut(id) {
                task.finish(TaskStatus::Exited(code));
            }
            self.children.remove(&id);
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

fn spawn_reader<S>(stream: S, buffer: Arc<Mutex<Vec<String>>>)
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
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: u64) -> TaskRecord {
        TaskRecord::new(id, "test".into(), "true".into())
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
        assert!(matches!(
            store.remove(99),
            Err(TaskError::NotFound(99))
        ));
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
