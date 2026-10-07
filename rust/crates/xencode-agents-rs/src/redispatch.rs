//! OR-7: Re-dispatch on failure.
//!
//! When an autonomous worker dies or is terminated, its task is re-queued onto
//! another agent using the worker continuation package (`AR-7`).
//!
//! # Critical Invariants
//!
//! 1. Failure reasons are captured strictly from the process (signal, exit code, timeout),
//!    never from worker prose or self-reported claims.
//! 2. The code diff is preserved directly from the repository without loss.
//! 3. The re-dispatch retry is written to the ledger as a second attempt on the same task.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::worker_package::{
    resume_task_from_package, ObservedDiff, ResumptionOutcome, WorkerPackage, WorkerStopReason,
};

/// An attempt logged in the task attempt ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskAttemptEntry {
    /// Identifier of the task.
    pub task_id: String,
    /// 1-based attempt counter for this task.
    pub attempt: usize,
    /// Which worker executed this attempt.
    pub worker: String,
    /// Objective process stop reason.
    pub stop_reason: WorkerStopReason,
    /// Whether the attempt concluded successfully.
    pub passed: bool,
    /// Changed files preserved across this attempt.
    pub changed_files: Vec<String>,
    /// Timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
    /// Path to the continuation package used or produced, if any.
    pub package_path: Option<String>,
}

/// Ledger tracking execution attempts per task.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskAttemptLedger {
    pub entries: Vec<TaskAttemptEntry>,
}

impl TaskAttemptLedger {
    /// Load the ledger entries from a JSONL file on disk.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self {
                entries: Vec::new(),
            });
        }
        let content = fs::read_to_string(path)
            .map_err(|e| format!("failed to read ledger from {}: {e}", path.display()))?;
        let mut entries = Vec::new();
        for (idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let entry: TaskAttemptEntry = serde_json::from_str(line).map_err(|e| {
                format!(
                    "failed to parse ledger entry line {} in {}: {e}",
                    idx + 1,
                    path.display()
                )
            })?;
            entries.push(entry);
        }
        Ok(Self { entries })
    }

    /// Append an entry to the JSONL ledger file.
    pub fn append_to_file(path: &Path, entry: &TaskAttemptEntry) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create ledger directory: {e}"))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("failed to open ledger file {}: {e}", path.display()))?;

        let serialized = serde_json::to_string(entry)
            .map_err(|e| format!("failed to serialize ledger entry: {e}"))?;
        writeln!(file, "{serialized}").map_err(|e| format!("failed to write ledger entry: {e}"))?;
        Ok(())
    }

    /// Return all attempts logged for a specific task.
    pub fn attempts_for_task<'a>(
        &'a self,
        task_id: &'a str,
    ) -> impl Iterator<Item = &'a TaskAttemptEntry> {
        self.entries.iter().filter(move |e| e.task_id == task_id)
    }
}

/// Complete outcome of re-dispatching a failed worker onto a second agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedispatchOutcome {
    pub task_id: String,
    pub attempts: usize,
    pub initial_worker: String,
    pub initial_stop_reason: WorkerStopReason,
    pub replacement_worker: String,
    pub all_tests_passed: bool,
    pub preserved_diff: ObservedDiff,
    pub ledger_entries: Vec<TaskAttemptEntry>,
    pub resumption_outcome: ResumptionOutcome,
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Re-queue a failed worker task onto another agent using an AR-7 continuation package.
///
/// 1. Verifies the initial worker died (stop reason must be non-zero or abnormal termination).
/// 2. Builds a continuation package capturing the uncommitted repository diff and process stop reason.
/// 3. Logs Attempt 1 in the ledger.
/// 4. Resumes the task using `replacement_worker` without losing the diff.
/// 5. Logs Attempt 2 in the ledger.
#[allow(clippy::too_many_arguments)]
pub fn redispatch_failed_worker<F>(
    repo_dir: &Path,
    task_id: &str,
    initial_worker: &str,
    replacement_worker: &str,
    stop_reason: WorkerStopReason,
    verification_test_cmds: &[&str],
    ledger_path: &Path,
    execute_resumption: F,
) -> Result<RedispatchOutcome, String>
where
    F: FnOnce(&str, &Path) -> Result<i32, String>,
{
    // 1. Initial stop reason must be a real process failure, not a clean zero exit
    if stop_reason.is_clean_exit() {
        return Err(format!(
            "cannot re-dispatch task '{task_id}': worker '{initial_worker}' exited cleanly with code 0"
        ));
    }

    // 2. Build worker continuation package from the repo state
    let package = WorkerPackage::build(
        repo_dir,
        task_id,
        format!("Auto-generated continuation package after {initial_worker} failure"),
        initial_worker,
        stop_reason.clone(),
        verification_test_cmds,
        Vec::new(),
    )?;

    let pkg_dir = repo_dir.join(".xencode").join("packages");
    fs::create_dir_all(&pkg_dir).map_err(|e| e.to_string())?;
    let pkg_path = pkg_dir.join(format!("{task_id}.json"));
    package.save_to_file(&pkg_path)?;
    let pkg_path_str = pkg_path.to_string_lossy().to_string();

    // 3. Log Attempt 1 in the ledger
    let attempt1 = TaskAttemptEntry {
        task_id: task_id.to_string(),
        attempt: 1,
        worker: initial_worker.to_string(),
        stop_reason: stop_reason.clone(),
        passed: false,
        changed_files: package.observed_diff.changed_files.clone(),
        timestamp_ms: now_unix_ms(),
        package_path: Some(pkg_path_str.clone()),
    };
    TaskAttemptLedger::append_to_file(ledger_path, &attempt1)?;

    // 4. Resume the task on the replacement worker using the continuation package
    let resumption_outcome = resume_task_from_package(
        &package,
        repo_dir,
        replacement_worker,
        verification_test_cmds,
        execute_resumption,
    )?;

    // 5. Log Attempt 2 (the retry) in the ledger
    let attempt2_exit_code = if resumption_outcome.all_tests_passed {
        0
    } else {
        1
    };
    let attempt2 = TaskAttemptEntry {
        task_id: task_id.to_string(),
        attempt: 2,
        worker: replacement_worker.to_string(),
        stop_reason: WorkerStopReason::ExitCode {
            code: attempt2_exit_code,
        },
        passed: resumption_outcome.all_tests_passed,
        changed_files: package.observed_diff.changed_files.clone(),
        timestamp_ms: now_unix_ms(),
        package_path: Some(pkg_path_str),
    };
    TaskAttemptLedger::append_to_file(ledger_path, &attempt2)?;

    Ok(RedispatchOutcome {
        task_id: task_id.to_string(),
        attempts: 2,
        initial_worker: initial_worker.to_string(),
        initial_stop_reason: stop_reason,
        replacement_worker: replacement_worker.to_string(),
        all_tests_passed: resumption_outcome.all_tests_passed,
        preserved_diff: package.observed_diff,
        ledger_entries: vec![attempt1, attempt2],
        resumption_outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn init_git_repo(path: &Path) {
        let run = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(path)
                .args(args)
                .status()
                .expect("git setup failed");
            assert!(status.success());
        };
        run(&["init"]);
        run(&["config", "user.email", "test@example.com"]);
        run(&["config", "user.name", "Tester"]);
        fs::write(path.join("app.rs"), "fn main() {}\n").unwrap();
        run(&["add", "app.rs"]);
        run(&["commit", "-m", "initial commit"]);
    }

    #[test]
    fn killed_worker_task_completes_on_replacement_with_diff_preserved_and_ledger_retry() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);

        // Initial worker modifies app.rs before dying
        let modified_code = "fn main() {\n    println!(\"worker 1 was here\");\n}\n";
        fs::write(repo.join("app.rs"), modified_code).unwrap();

        // The process was killed by SIGKILL (signal 9)
        let stop_reason = WorkerStopReason::Signal { signal: 9 };
        let ledger_file = repo.join(".xencode").join("task_ledger.jsonl");

        let outcome = redispatch_failed_worker(
            repo,
            "task-failover-42",
            "first-worker",
            "second-worker",
            stop_reason,
            &["test -f app.rs", "grep -q 'worker 1 was here' app.rs"],
            &ledger_file,
            |context, _workdir| {
                // Replacement worker inspects context and verifies diff
                assert!(context.contains("app.rs"));
                assert!(context.contains("worker 1 was here"));
                assert!(context.contains("terminated by signal 9"));
                Ok(0)
            },
        )
        .unwrap();

        // 1. Task completed elsewhere
        assert_eq!(outcome.initial_worker, "first-worker");
        assert_eq!(outcome.replacement_worker, "second-worker");
        assert!(outcome.all_tests_passed);
        assert_eq!(outcome.attempts, 2);

        // 2. Diff was preserved without losing work
        assert_eq!(
            outcome.preserved_diff.changed_files,
            vec!["app.rs".to_string()]
        );
        let current_content = fs::read_to_string(repo.join("app.rs")).unwrap();
        assert_eq!(current_content, modified_code);

        // 3. Retry is visible in the ledger as a second attempt on one task
        let ledger = TaskAttemptLedger::load_from_file(&ledger_file).unwrap();
        let attempts: Vec<&TaskAttemptEntry> =
            ledger.attempts_for_task("task-failover-42").collect();
        assert_eq!(attempts.len(), 2, "must have exactly 2 attempts in ledger");

        assert_eq!(attempts[0].attempt, 1);
        assert_eq!(attempts[0].worker, "first-worker");
        assert!(!attempts[0].passed);
        assert_eq!(
            attempts[0].stop_reason,
            WorkerStopReason::Signal { signal: 9 }
        );
        assert_eq!(attempts[0].changed_files, vec!["app.rs".to_string()]);

        assert_eq!(attempts[1].attempt, 2);
        assert_eq!(attempts[1].worker, "second-worker");
        assert!(attempts[1].passed);
        assert_eq!(
            attempts[1].stop_reason,
            WorkerStopReason::ExitCode { code: 0 }
        );
        assert_eq!(attempts[1].changed_files, vec!["app.rs".to_string()]);
    }

    #[test]
    fn cannot_redispatch_clean_zero_exit_worker() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);

        let clean_exit = WorkerStopReason::ExitCode { code: 0 };
        let ledger_file = repo.join(".xencode").join("task_ledger.jsonl");

        let err = redispatch_failed_worker(
            repo,
            "task-clean",
            "worker-1",
            "worker-2",
            clean_exit,
            &["true"],
            &ledger_file,
            |_ctx, _dir| Ok(0),
        )
        .unwrap_err();

        assert!(err.contains("exited cleanly"));
    }
}
