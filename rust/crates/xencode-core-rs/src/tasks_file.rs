//! Cross-process background task registry (Milestone D, D2-02).
//!
//! [`TaskManager`](super::TaskManager) owns live child handles and only
//! lives as long as its process, so a CLI invocation cannot see the TUI's
//! registry. This module persists the same idea on disk: task records in
//! `tasks.json`, each command wrapped in a `<id>.sh` script that writes its
//! exit code to `<id>.exit`, and merged stdout/stderr in `<id>.out`. Status
//! is *derived* on read (exit file, killed flag, `/proc` liveness) rather
//! than stored, so it stays honest even if the wrapper dies hard.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::tasks::{TaskError, TaskStatus};

/// One persisted task. `status` is deliberately absent — it is computed by
/// [`FileTaskRegistry::status`] from the on-disk side files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileTask {
    pub id: u64,
    pub name: String,
    pub command: String,
    pub pid: u32,
    pub started_at: u64,
    /// Set when `stop` was requested; wins over the exit file so a killed
    /// task reads as `Killed` even though the wrapper reports a code.
    pub killed: bool,
}

/// Registry backed by `<root>/tasks.json` plus per-task side files.
#[derive(Debug, Clone)]
pub struct FileTaskRegistry {
    root: PathBuf,
}

impl FileTaskRegistry {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn tasks_file(&self) -> PathBuf {
        self.root.join("tasks.json")
    }

    fn script_path(&self, id: u64) -> PathBuf {
        self.root.join(format!("{id}.sh"))
    }

    fn exit_path(&self, id: u64) -> PathBuf {
        self.root.join(format!("{id}.exit"))
    }

    fn out_path(&self, id: u64) -> PathBuf {
        self.root.join(format!("{id}.out"))
    }

    /// All persisted tasks, ordered by id. A missing file means no tasks.
    pub fn list(&self) -> Result<Vec<FileTask>, TaskError> {
        let raw = match fs::read_to_string(self.tasks_file()) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(TaskError::Spawn(e)),
        };
        let tasks: Vec<FileTask> =
            serde_json::from_str(&raw).map_err(|e| TaskError::Spawn(std::io::Error::other(e)))?;
        Ok(tasks)
    }

    fn save(&self, tasks: &[FileTask]) -> Result<(), TaskError> {
        fs::create_dir_all(&self.root).map_err(TaskError::Spawn)?;
        let json = serde_json::to_string_pretty(tasks)
            .map_err(|e| TaskError::Spawn(std::io::Error::other(e)))?;
        // Atomic replace: other processes read this file at any moment and
        // must never see a half-written JSON document.
        crate::atomic::write_atomic(&self.tasks_file(), json.as_bytes()).map_err(TaskError::Spawn)
    }

    /// Spawn `command` detached from this process and record it.
    pub fn start(&self, name: &str, command: &str) -> Result<FileTask, TaskError> {
        let mut tasks = self.list()?;
        let id = tasks.iter().map(|t| t.id).max().unwrap_or(0) + 1;
        fs::create_dir_all(&self.root).map_err(TaskError::Spawn)?;

        let script = self.script_path(id);
        let exit = self.exit_path(id);
        // EXIT trap, not a trailing line: an `exit N` inside the command
        // would skip anything written after it. The exit path arrives as
        // $1 so no shell quoting of the path is needed.
        let wrapper = format!("#!/bin/sh\ntrap 'echo $? > \"$1\"' EXIT\n{command}\n");
        fs::write(&script, wrapper).map_err(TaskError::Spawn)?;

        let out = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.out_path(id))
            .map_err(TaskError::Spawn)?;
        let child = Command::new("sh")
            .arg(&script)
            .arg(exit.display().to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::from(out.try_clone().map_err(TaskError::Spawn)?))
            .stderr(Stdio::from(out))
            .spawn()
            .map_err(TaskError::Spawn)?;

        let task = FileTask {
            id,
            name: if name.is_empty() {
                command.to_string()
            } else {
                name.to_string()
            },
            command: command.to_string(),
            pid: child.id(),
            started_at: unix_now_secs(),
            killed: false,
        };
        // std::process::Child leaves the child running when dropped, which
        // is exactly what we want here.
        drop(child);
        tasks.push(task.clone());
        self.save(&tasks)?;
        Ok(task)
    }

    fn find<'a>(&self, tasks: &'a [FileTask], id: u64) -> Result<&'a FileTask, TaskError> {
        tasks
            .iter()
            .find(|t| t.id == id)
            .ok_or(TaskError::NotFound(id))
    }

    /// Read the wrapper's exit code file, if it exists.
    fn exit_code(&self, id: u64) -> Option<i32> {
        let raw = fs::read_to_string(self.exit_path(id)).ok()?;
        raw.trim().parse::<i32>().ok()
    }

    /// Derive the task's current status from on-disk state. `task` must be
    /// the stored record (fresh from `list`), not a pre-`stop` clone.
    pub fn status(&self, task: &FileTask) -> TaskStatus {
        match self.exit_code(task.id) {
            Some(_) if task.killed => TaskStatus::Killed,
            Some(code) => TaskStatus::Exited(code),
            None if pid_alive(task.pid) => TaskStatus::Running,
            None => TaskStatus::Killed,
        }
    }

    /// Last `tail` lines of merged stdout+stderr.
    pub fn output(&self, id: u64, tail: usize) -> Vec<String> {
        let raw = match fs::read_to_string(self.out_path(id)) {
            Ok(raw) => raw,
            Err(_) => return Vec::new(),
        };
        let mut lines: Vec<String> = raw.lines().map(str::to_string).collect();
        if lines.len() > tail {
            lines = lines.split_off(lines.len() - tail);
        }
        lines
    }

    /// Signal a running task to stop. Errors: unknown id or already gone.
    pub fn stop(&self, id: u64) -> Result<(), TaskError> {
        let mut tasks = self.list()?;
        let task = self.find(&tasks, id)?.clone();
        if self.status(&task) != TaskStatus::Running {
            return Err(TaskError::AlreadyFinished(id));
        }
        kill_pid(task.pid).map_err(TaskError::Spawn)?;
        let entry = tasks
            .iter_mut()
            .find(|t| t.id == id)
            .ok_or(TaskError::NotFound(id))?;
        entry.killed = true;
        self.save(&tasks)
    }

    /// Forget a finished task and delete its side files.
    pub fn remove(&self, id: u64) -> Result<(), TaskError> {
        let tasks = self.list()?;
        let task = self.find(&tasks, id)?;
        if self.status(task) == TaskStatus::Running {
            return Err(TaskError::StillRunning(id));
        }
        let rest: Vec<FileTask> = tasks.into_iter().filter(|t| t.id != id).collect();
        for path in [self.script_path(id), self.exit_path(id), self.out_path(id)] {
            let _ = fs::remove_file(path);
        }
        self.save(&rest)
    }

    /// Snapshot for the CLI: every task with its derived status.
    pub fn poll(&self) -> Result<BTreeMap<u64, (FileTask, TaskStatus)>, TaskError> {
        Ok(self
            .list()?
            .into_iter()
            .map(|t| {
                let status = self.status(&t);
                (t.id, (t, status))
            })
            .collect())
    }
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A pid is alive if its `/proc` entry exists and is not a zombie (a
/// dropped, unreaped child lingers as `Z` until we exit).
pub fn pid_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        match fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => match stat.rfind(')') {
                Some(i) => stat.as_bytes().get(i + 2) != Some(&(b'Z')),
                None => true,
            },
            Err(_) => false,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        true
    }
}

fn kill_pid(pid: u32) -> std::io::Result<std::process::Child> {
    #[cfg(unix)]
    let spawned = Command::new("kill").arg(pid.to_string()).spawn();
    #[cfg(windows)]
    let spawned = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .spawn();
    #[cfg(not(any(unix, windows)))]
    let spawned = Err(std::io::Error::other("no kill mechanism on this platform"));
    let mut child = spawned?;
    // Reap the killer so no zombies linger; its exit code is best-effort
    // confirmation the signal was deliverable (ESRCH shows up as exit 1).
    let _ = child.wait();
    Ok(child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SEQ: AtomicUsize = AtomicUsize::new(0);

    /// Each test gets its own directory under the system temp dir so the
    /// real `~/.xencode` state is never touched.
    fn temp_root(label: &str) -> PathBuf {
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!(
            "xencode-tasksfile-{label}-{}-{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn wait_until<F: Fn() -> bool>(f: F) -> bool {
        for _ in 0..200 {
            if f() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn empty_registry_lists_nothing() {
        let reg = FileTaskRegistry::new(temp_root("empty"));
        assert!(reg.list().unwrap().is_empty());
        assert!(reg.poll().unwrap().is_empty());
        assert!(reg.output(1, 10).is_empty());
    }

    #[test]
    fn ids_are_unique_and_persist_across_instances() {
        let root = temp_root("ids");
        let reg = FileTaskRegistry::new(&root);
        let a = reg.start("a", "exit 0").unwrap();
        let b = reg.start("b", "exit 0").unwrap();
        assert_eq!((a.id, b.id), (1, 2));
        // A fresh instance (simulating another CLI process) sees the same rows.
        let reopened = FileTaskRegistry::new(&root);
        let ids: Vec<u64> = reopened.list().unwrap().into_iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![1, 2]);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn echo_task_exits_zero_with_output() {
        let reg = FileTaskRegistry::new(temp_root("echo"));
        let task = reg.start("greet", "echo hi").unwrap();
        assert!(wait_until(|| reg.exit_code(task.id).is_some()));
        assert_eq!(reg.status(&task), TaskStatus::Exited(0));
        assert_eq!(reg.output(task.id, 10), vec!["hi".to_string()]);
        reg.remove(task.id).unwrap();
        assert!(reg.list().unwrap().is_empty());
    }

    #[test]
    fn nonzero_exit_code_is_recorded() {
        let reg = FileTaskRegistry::new(temp_root("code3"));
        let task = reg.start("fail", "exit 3").unwrap();
        assert!(wait_until(|| reg.status(&task) == TaskStatus::Exited(3)));
        let _ = fs::remove_dir_all(reg.root());
    }

    #[test]
    fn stop_marks_killed_and_remove_refuses_running() {
        let reg = FileTaskRegistry::new(temp_root("stop"));
        let task = reg.start("sleeper", "sleep 30").unwrap();
        assert_eq!(reg.status(&task), TaskStatus::Running);
        assert!(
            matches!(reg.remove(task.id), Err(TaskError::StillRunning(_))),
            "remove must refuse a running task"
        );
        reg.stop(task.id).unwrap();
        // Re-read the stored record: `stop` persisted the killed flag, and a
        // killed task must read as Killed whether or not its EXIT trap got
        // to write an exit code before dying.
        assert!(wait_until(|| {
            let stored = reg
                .list()
                .unwrap()
                .into_iter()
                .find(|t| t.id == task.id)
                .unwrap();
            reg.status(&stored) == TaskStatus::Killed
        }));
        // A second stop sees a finished task.
        assert!(
            matches!(reg.stop(task.id), Err(TaskError::AlreadyFinished(_))),
            "second stop must report already-finished"
        );
        reg.remove(task.id).unwrap();
        assert!(reg.list().unwrap().is_empty());
        assert!(!reg.out_path(task.id).exists());
    }

    #[test]
    fn unknown_ids_and_dead_pids() {
        let reg = FileTaskRegistry::new(temp_root("unknown"));
        assert!(matches!(reg.stop(99), Err(TaskError::NotFound(99))));
        assert!(matches!(reg.remove(99), Err(TaskError::NotFound(99))));
        // A record whose pid long vanished and never wrote an exit file
        // reads as Killed; hand-craft the row since start() can't make one.
        let ghost = FileTask {
            id: 7,
            name: "ghost".into(),
            command: "true".into(),
            pid: u32::MAX,
            started_at: unix_now_secs(),
            killed: false,
        };
        reg.save(std::slice::from_ref(&ghost)).unwrap();
        assert_eq!(reg.status(&ghost), TaskStatus::Killed);
        assert!(matches!(reg.stop(7), Err(TaskError::AlreadyFinished(7))));
        let _ = fs::remove_dir_all(reg.root());
    }

    #[test]
    fn output_tail_is_bounded() {
        let reg = FileTaskRegistry::new(temp_root("tail"));
        let task = reg
            .start("seq", "for i in $(seq 1 100); do echo line$i; done")
            .unwrap();
        assert!(wait_until(|| reg.output(task.id, usize::MAX).len() >= 100));
        let tail = reg.output(task.id, 3);
        assert_eq!(tail, vec!["line98", "line99", "line100"]);
        let _ = fs::remove_dir_all(reg.root());
    }
}
