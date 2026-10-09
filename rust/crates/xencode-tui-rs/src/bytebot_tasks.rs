//! ByteBot tasks (BT-1): each task the person hands to ByteBot is a record on
//! disk with the original Bytebot project's states, kept in
//! `<project>/.xencode/bytebot/tasks/<id>.json` so the list survives a restart.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The original project's task states (`bytebot-ai/bytebot`, `TaskStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Pending,
    Running,
    NeedsHelp,
    NeedsReview,
    Completed,
    Cancelled,
    Failed,
}

impl TaskState {
    /// The state as the panel writes it.
    pub fn words(self) -> &'static str {
        match self {
            TaskState::Pending => "pending",
            TaskState::Running => "running",
            TaskState::NeedsHelp => "needs help",
            TaskState::NeedsReview => "needs review",
            TaskState::Completed => "completed",
            TaskState::Cancelled => "cancelled",
            TaskState::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteBotTask {
    pub id: String,
    pub text: String,
    /// The model the task runs with, fixed when it starts.
    pub model: String,
    pub state: TaskState,
    /// One row per tool call: what was called and how it ended.
    #[serde(default)]
    pub steps: Vec<(String, String)>,
    /// Files the task changed, relative to the project, forward slashes.
    #[serde(default)]
    pub changed_files: Vec<String>,
    /// The question it is waiting on while it needs help (BT-2).
    #[serde(default)]
    pub question: Option<String>,
    /// Why it ended the way it did, when that needs saying.
    #[serde(default)]
    pub note: Option<String>,
    /// The checkpoint group its writes went into, for undo (BT-3).
    #[serde(default)]
    pub turn: Option<usize>,
    pub created_at: u64,
    #[serde(default)]
    pub started_at: Option<u64>,
    #[serde(default)]
    pub ended_at: Option<u64>,
}

impl ByteBotTask {
    pub fn new(text: &str, model: &str) -> ByteBotTask {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        // Seconds for order, then the low bits of the clock's nanoseconds and
        // a per-process counter, so two tasks made in one second differ.
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = format!(
            "{}-{:05x}{:03x}",
            now.as_secs(),
            now.subsec_nanos() & 0xfffff,
            n & 0xfff
        );
        ByteBotTask {
            id,
            text: text.to_string(),
            model: model.to_string(),
            state: TaskState::Pending,
            steps: Vec::new(),
            changed_files: Vec::new(),
            question: None,
            note: None,
            turn: None,
            created_at: now.as_secs(),
            started_at: None,
            ended_at: None,
        }
    }
}

/// The folder of task records for one project.
pub struct TaskStore {
    dir: PathBuf,
}

impl TaskStore {
    /// `xencode_dir` is the project's `.xencode` folder.
    pub fn new(xencode_dir: &Path) -> TaskStore {
        TaskStore {
            dir: xencode_dir.join("bytebot").join("tasks"),
        }
    }

    /// Write one record atomically: a temporary file renamed over the old one.
    pub fn save(&self, task: &ByteBotTask) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let target = self.dir.join(format!("{}.json", task.id));
        let temp = self.dir.join(format!(".{}.tmp", task.id));
        let body = serde_json::to_string_pretty(task).map_err(std::io::Error::other)?;
        std::fs::write(&temp, body)?;
        std::fs::rename(&temp, &target).inspect_err(|_| {
            let _ = std::fs::remove_file(&temp);
        })
    }

    /// Every readable record, oldest first. A record that cannot be read is
    /// skipped rather than stopping the list from loading.
    pub fn load_all(&self) -> Vec<ByteBotTask> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut tasks: Vec<ByteBotTask> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
            .filter_map(|p| serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok())
            .collect();
        tasks.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        tasks
    }

    /// Mark tasks that were running or waiting for help when xencode last
    /// exited as failed; nothing can resume them. Returns how many.
    pub fn recover(&self) -> usize {
        let mut recovered = 0;
        for mut task in self.load_all() {
            if matches!(task.state, TaskState::Running | TaskState::NeedsHelp) {
                task.state = TaskState::Failed;
                task.question = None;
                task.note = Some("xencode exited during this task".to_string());
                if self.save(&task).is_ok() {
                    recovered += 1;
                }
            }
        }
        recovered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_task_round_trips_and_states_use_the_original_names() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let t = ByteBotTask::new("fix the login bug", "llamacpp:qwen3-4b");
        store.save(&t).unwrap();
        let text = std::fs::read_to_string(
            dir.path()
                .join("bytebot")
                .join("tasks")
                .join(format!("{}.json", t.id)),
        )
        .unwrap();
        assert!(text.contains("\"state\": \"pending\""), "{text}");
        assert_eq!(store.load_all(), vec![t]);
    }

    #[test]
    fn tasks_load_oldest_first() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let mut a = ByteBotTask::new("a", "m");
        a.created_at = 200;
        let mut b = ByteBotTask::new("b", "m");
        b.created_at = 100;
        store.save(&a).unwrap();
        store.save(&b).unwrap();
        let texts: Vec<_> = store.load_all().into_iter().map(|t| t.text).collect();
        assert_eq!(texts, vec!["b".to_string(), "a".to_string()]);
    }

    #[test]
    fn a_task_left_running_or_waiting_is_failed_on_restart() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let mut a = ByteBotTask::new("a", "m");
        a.state = TaskState::Running;
        let mut b = ByteBotTask::new("b", "m");
        b.state = TaskState::NeedsHelp;
        let c = ByteBotTask::new("c", "m");
        for t in [&a, &b, &c] {
            store.save(t).unwrap();
        }
        assert_eq!(store.recover(), 2);
        let all = store.load_all();
        for t in all.iter().filter(|t| t.text != "c") {
            assert_eq!(t.state, TaskState::Failed);
            assert_eq!(t.note.as_deref(), Some("xencode exited during this task"));
        }
        assert_eq!(
            all.iter().find(|t| t.text == "c").unwrap().state,
            TaskState::Pending
        );
    }

    #[test]
    fn two_tasks_made_at_once_get_different_ids() {
        let a = ByteBotTask::new("a", "m");
        let b = ByteBotTask::new("b", "m");
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn a_broken_record_is_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        store.save(&ByteBotTask::new("ok", "m")).unwrap();
        std::fs::write(dir.path().join("bytebot/tasks/broken.json"), "{").unwrap();
        assert_eq!(store.load_all().len(), 1);
    }
}
