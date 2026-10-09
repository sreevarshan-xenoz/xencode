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
    /// Made by the person in this session. Never written or read: a record
    /// loaded from disk is always `false`, and only `true` tasks may start.
    #[serde(skip)]
    pub this_session: bool,
}

impl ByteBotTask {
    pub fn new(text: &str, model: &str) -> ByteBotTask {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        // Seconds, the full nanoseconds and a per-process counter, all
        // zero-padded, so ids sort in the order the tasks were made: the
        // queue reloads in that order after a restart.
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = task_id(now.as_secs(), now.subsec_nanos(), n);
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
            this_session: true,
        }
    }
}

/// Whether `id` has the shape `task_id` makes: digits and dashes only.
fn is_task_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_digit() || c == '-')
}

/// A task id that sorts in creation order: seconds, nanoseconds
/// and a per-process counter, each zero-padded to a fixed width.
fn task_id(secs: u64, nanos: u32, counter: u32) -> String {
    format!("{secs:012}-{nanos:09}-{:04}", counter % 10_000)
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
    /// Secrets are redacted from the text, steps and note before writing, and
    /// an id that is not one this module makes is refused, so a record can
    /// never name a file outside the tasks folder.
    pub fn save(&self, task: &ByteBotTask) -> std::io::Result<()> {
        if !is_task_id(&task.id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("not a task id: {:?}", task.id),
            ));
        }
        let mut task = task.clone();
        task.text = xencode_context_rs::redact_secrets(&task.text);
        for (call, outcome) in &mut task.steps {
            *call = xencode_context_rs::redact_secrets(call);
            *outcome = xencode_context_rs::redact_secrets(outcome);
        }
        task.note = task.note.map(|n| xencode_context_rs::redact_secrets(&n));
        task.question = task
            .question
            .map(|q| xencode_context_rs::redact_secrets(&q));
        let task = &task;
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
            .filter_map(|p| {
                serde_json::from_str::<ByteBotTask>(&std::fs::read_to_string(p).ok()?).ok()
            })
            .filter(|t| is_task_id(&t.id))
            .collect();
        tasks.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        tasks
    }

    /// Settle tasks left open when xencode last exited. Returns every task with
    /// its settled state, decided here in memory so a record that cannot be
    /// rewritten still never comes back pending, and how many were
    /// interrupted mid-run. A task that was running or waiting for help
    /// cannot resume, so it is failed. A task still pending is cancelled, not
    /// run: its text was read from disk rather than typed by the person at the
    /// keyboard now, and a record can be planted in that folder.
    pub fn recover(&self) -> (Vec<ByteBotTask>, usize) {
        let mut interrupted = 0;
        let mut settled = Vec::new();
        for mut task in self.load_all() {
            match task.state {
                TaskState::Running | TaskState::NeedsHelp => {
                    task.state = TaskState::Failed;
                    task.question = None;
                    task.note = Some("xencode exited during this task".to_string());
                    let _ = self.save(&task);
                    interrupted += 1;
                }
                TaskState::Pending => {
                    task.state = TaskState::Cancelled;
                    task.note = Some(
                        "not started before xencode exited; type it again to run it".to_string(),
                    );
                    let _ = self.save(&task);
                }
                _ => {}
            }
            settled.push(task);
        }
        (settled, interrupted)
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
        assert!(
            !text.contains("this_session"),
            "the session flag is never written"
        );
        // Everything comes back except the session flag, which is never read.
        let mut expected = t.clone();
        expected.this_session = false;
        assert_eq!(store.load_all(), vec![expected]);
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
        let (all, interrupted) = store.recover();
        assert_eq!(interrupted, 2);
        for t in all.iter().filter(|t| t.text != "c") {
            assert_eq!(t.state, TaskState::Failed);
            assert_eq!(t.note.as_deref(), Some("xencode exited during this task"));
        }
        // A task still pending from an earlier session is never started on
        // its own: its text came from disk, not from the person now at the
        // keyboard, and a record can be planted there.
        let c = all.iter().find(|t| t.text == "c").unwrap();
        assert_eq!(c.state, TaskState::Cancelled);
        assert!(c.note.as_deref().is_some_and(|n| n.contains("not started")));
    }

    #[test]
    fn a_pending_task_is_settled_even_when_its_record_cannot_be_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let t = ByteBotTask::new("planted", "m");
        store.save(&t).unwrap();
        let file = dir
            .path()
            .join("bytebot/tasks")
            .join(format!("{}.json", t.id));
        let original = std::fs::metadata(&file).unwrap().permissions();
        let mut readonly = original.clone();
        readonly.set_readonly(true);
        std::fs::set_permissions(&file, readonly).unwrap();
        let (tasks, _) = store.recover();
        std::fs::set_permissions(&file, original).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(
            tasks[0].state,
            TaskState::Cancelled,
            "the in-memory list does not trust the failed write"
        );
    }

    #[test]
    fn only_tasks_made_in_this_session_can_start() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let typed = ByteBotTask::new("typed now", "m");
        assert!(typed.this_session);
        store.save(&typed).unwrap();
        let loaded = store.load_all();
        assert!(
            !loaded[0].this_session,
            "a record read from disk is never this session's"
        );
    }

    #[test]
    fn a_record_whose_id_is_a_path_is_never_loaded_or_written() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let mut evil = ByteBotTask::new("x", "m");
        evil.id = "../../escaped".to_string();
        assert!(
            store.save(&evil).is_err(),
            "an id that is a path is refused"
        );
        assert!(!dir.path().join("escaped.json").exists());
        let planted = serde_json::to_string(&evil).unwrap();
        std::fs::create_dir_all(dir.path().join("bytebot/tasks")).unwrap();
        std::fs::write(dir.path().join("bytebot/tasks/planted.json"), planted).unwrap();
        assert!(
            store.load_all().is_empty(),
            "a planted record with a path id is skipped"
        );
    }

    #[test]
    fn secrets_never_reach_a_task_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let mut t = ByteBotTask::new("deploy with key sk-FAKE-NOT-A-REAL-TEST-KEY", "m");
        t.steps.push((
            "run_command curl -H \"Authorization: Bearer sk-FAKE-NOT-A-REAL-TEST-KEY\"".into(),
            "done".into(),
        ));
        t.note = Some("failed: sk-FAKE-NOT-A-REAL-TEST-KEY rejected".into());
        store.save(&t).unwrap();
        let text = std::fs::read_to_string(
            dir.path()
                .join("bytebot/tasks")
                .join(format!("{}.json", t.id)),
        )
        .unwrap();
        assert!(!text.contains("sk-FAKE-NOT-A-REAL-TEST-KEY"), "{text}");
    }

    #[test]
    fn ids_sort_in_creation_order_across_every_digit_boundary() {
        // Pairs made one after the other; each later id must sort after.
        let pairs = [
            ((5, 1_048_575, 0), (5, 1_048_576, 1)),
            ((5, 999_999_999, 7), (6, 0, 8)),
            ((5, 10, 9_999), (5, 11, 0)),
            ((9, 0, 0), (10, 0, 1)),
        ];
        for ((s1, n1, c1), (s2, n2, c2)) in pairs {
            let (a, b) = (task_id(s1, n1, c1), task_id(s2, n2, c2));
            assert!(a < b, "{a} does not sort before {b}");
        }
    }

    #[test]
    fn tasks_made_in_one_second_reload_in_the_order_they_were_made() {
        let dir = tempfile::tempdir().unwrap();
        let store = TaskStore::new(dir.path());
        let made: Vec<ByteBotTask> = (0..20)
            .map(|i| ByteBotTask::new(&format!("task {i}"), "m"))
            .collect();
        for t in made.iter().rev() {
            store.save(t).unwrap();
        }
        let ids: Vec<_> = made.iter().map(|t| t.id.clone()).collect();
        let reloaded: Vec<_> = store.load_all().into_iter().map(|t| t.id).collect();
        assert_eq!(reloaded, ids);
        let mut unique = ids.clone();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "two tasks share an id");
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
