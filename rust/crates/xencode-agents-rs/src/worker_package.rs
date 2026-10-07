//! AR-7: The worker continuation package.
//!
//! When an autonomous coding worker finishes or stops — whether it completed,
//! errored, timed out, or was interrupted — what the *next* worker needs to know
//! must be constructed purely from verifiable, observed facts:
//!
//! 1. The git diff: actual changed files and unified patch against repository HEAD.
//! 2. Verification test results: commands xencode executed and their real exit codes.
//! 3. The last normalized events: real tool invocations and outputs from the event stream.
//! 4. Why the previous worker stopped: the real process exit code, signal, or cancellation.
//!
//! # Critical Invariant: No Self-Reported Progress or Completion Claims
//!
//! A worker claiming "I finished 75%" or "the task is completely done" is an unverified
//! assertion, not a fact. The continuation package contains NO self-reported progress
//! percentage or completion flags. Any attempted JSON payload carrying progress claims
//! is strictly rejected. Downstream resumption operates solely from the observed code
//! state and verification exit codes.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::protocol::AgentEvent;

/// Objective reason why the previous worker process stopped running.
///
/// Captured from process exit status or supervisory monitor, never from worker prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum WorkerStopReason {
    ExitCode { code: i32 },
    Signal { signal: i32 },
    TimedOut { duration_secs: u64 },
    Cancelled { reason: String },
    StalledNoOutput { idle_secs: u64 },
    Unknown,
}

impl WorkerStopReason {
    pub fn is_clean_exit(&self) -> bool {
        matches!(self, Self::ExitCode { code: 0 })
    }

    pub fn summary(&self) -> String {
        match self {
            Self::ExitCode { code } => format!("process exited with code {code}"),
            Self::Signal { signal } => format!("process terminated by signal {signal}"),
            Self::TimedOut { duration_secs } => format!("process timed out after {duration_secs}s"),
            Self::Cancelled { reason } => format!("process cancelled: {reason}"),
            Self::StalledNoOutput { idle_secs } => {
                format!("process stalled with no output for {idle_secs}s")
            }
            Self::Unknown => "process stopped for unknown reason".to_string(),
        }
    }
}

/// The diff observed directly by inspecting git in the target repository.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedDiff {
    pub changed_files: Vec<String>,
    pub patch: String,
    pub insertions: usize,
    pub deletions: usize,
}

impl ObservedDiff {
    pub fn empty() -> Self {
        Self {
            changed_files: Vec::new(),
            patch: String::new(),
            insertions: 0,
            deletions: 0,
        }
    }

    /// Inspect a repository on disk and construct the observed diff against HEAD.
    pub fn from_repo(repo_dir: &Path) -> Result<Self, String> {
        let status_output = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_dir)
            .args(["status", "--porcelain"])
            .output()
            .map_err(|e| format!("git status failed in {}: {e}", repo_dir.display()))?;

        let status_str = String::from_utf8_lossy(&status_output.stdout);
        let mut changed_files = Vec::new();
        for line in status_str.lines() {
            if line.len() > 3 {
                let path_part = &line[3..];
                let actual_path = if let Some((_, dest)) = path_part.split_once(" -> ") {
                    dest.trim()
                } else {
                    path_part.trim()
                };
                if !actual_path.is_empty() && !changed_files.iter().any(|f| f == actual_path) {
                    changed_files.push(actual_path.to_string());
                }
            }
        }

        let diff_output = std::process::Command::new("git")
            .arg("-C")
            .arg(repo_dir)
            .args(["diff", "HEAD"])
            .output()
            .or_else(|_| {
                std::process::Command::new("git")
                    .arg("-C")
                    .arg(repo_dir)
                    .arg("diff")
                    .output()
            })
            .map_err(|e| format!("git diff failed in {}: {e}", repo_dir.display()))?;

        let patch = String::from_utf8_lossy(&diff_output.stdout).to_string();

        let mut insertions = 0;
        let mut deletions = 0;
        for line in patch.lines() {
            if line.starts_with("+++") || line.starts_with("---") {
                continue;
            }
            if line.starts_with('+') {
                insertions += 1;
            } else if line.starts_with('-') {
                deletions += 1;
            }
        }

        Ok(Self {
            changed_files,
            patch,
            insertions,
            deletions,
        })
    }
}

/// A test command executed by xencode, with its exit code and output excerpt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedTestRun {
    pub command: String,
    pub exit_code: i32,
    pub output_tail: String,
    pub passed: bool,
}

impl VerifiedTestRun {
    /// Execute a test command in the repository and observe its exit code and output.
    pub fn run(repo_dir: &Path, command_str: &str) -> Result<Self, String> {
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(command_str)
            .current_dir(repo_dir)
            .output()
            .map_err(|e| format!("failed to execute '{command_str}': {e}"))?;

        let exit_code = output.status.code().unwrap_or(1);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = if stderr.is_empty() {
            stdout.to_string()
        } else if stdout.is_empty() {
            stderr.to_string()
        } else {
            format!("{stdout}\n{stderr}")
        };

        let lines: Vec<&str> = combined.lines().collect();
        let tail_lines = if lines.len() > 30 {
            &lines[lines.len() - 30..]
        } else {
            &lines[..]
        };
        let output_tail = tail_lines.join("\n");

        Ok(Self {
            command: command_str.to_string(),
            exit_code,
            output_tail,
            passed: exit_code == 0,
        })
    }
}

/// The worker continuation package: what the next worker needs to know.
///
/// Contains strictly observed facts. Any extraneous or self-reported progress
/// fields are rejected during deserialization via `deny_unknown_fields`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerPackage {
    pub task_id: String,
    pub task_description: String,
    pub previous_agent: String,
    pub stop_reason: WorkerStopReason,
    pub observed_diff: ObservedDiff,
    pub verified_tests: Vec<VerifiedTestRun>,
    pub recent_events: Vec<AgentEvent>,
    pub created_at_unix_ms: u64,
}

impl WorkerPackage {
    /// Build a continuation package from observed workspace facts.
    pub fn build(
        repo_dir: &Path,
        task_id: impl Into<String>,
        task_description: impl Into<String>,
        previous_agent: impl Into<String>,
        stop_reason: WorkerStopReason,
        test_commands: &[&str],
        recent_events: Vec<AgentEvent>,
    ) -> Result<Self, String> {
        let observed_diff = ObservedDiff::from_repo(repo_dir)?;
        let mut verified_tests = Vec::new();
        for cmd in test_commands {
            verified_tests.push(VerifiedTestRun::run(repo_dir, cmd)?);
        }
        let created_at_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Ok(Self {
            task_id: task_id.into(),
            task_description: task_description.into(),
            previous_agent: previous_agent.into(),
            stop_reason,
            observed_diff,
            verified_tests,
            recent_events,
            created_at_unix_ms,
        })
    }

    /// Format "what the next worker needs to know" as clear, actionable briefing text.
    pub fn resumption_context(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("# Task Continuation: {}\n\n", self.task_id));
        out.push_str("## Target Task\n");
        out.push_str(&self.task_description);
        out.push_str("\n\n");

        out.push_str("## Previous Worker Exit\n");
        out.push_str(&format!(
            "- Worker: `{}`\n- Observed stop: {}\n\n",
            self.previous_agent,
            self.stop_reason.summary()
        ));

        out.push_str(&format!(
            "## Observed Repository State ({} changed files, +{} -{} lines)\n",
            self.observed_diff.changed_files.len(),
            self.observed_diff.insertions,
            self.observed_diff.deletions
        ));
        if self.observed_diff.changed_files.is_empty() {
            out.push_str("No repository modifications observed.\n\n");
        } else {
            out.push_str("Modified files:\n");
            for f in &self.observed_diff.changed_files {
                out.push_str(&format!("- {f}\n"));
            }
            if !self.observed_diff.patch.is_empty() {
                out.push_str("\n```diff\n");
                out.push_str(&self.observed_diff.patch);
                if !self.observed_diff.patch.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str("```\n\n");
            }
        }

        out.push_str("## Verification Test Results\n");
        if self.verified_tests.is_empty() {
            out.push_str("No verification test runs executed.\n\n");
        } else {
            for t in &self.verified_tests {
                let status = if t.passed {
                    "PASSED (exit 0)"
                } else {
                    "FAILED"
                };
                out.push_str(&format!(
                    "- `{}` -> exit {} [{status}]\n",
                    t.command, t.exit_code
                ));
                if !t.passed && !t.output_tail.is_empty() {
                    out.push_str("  Output excerpt:\n```\n");
                    for line in t.output_tail.lines().take(15) {
                        out.push_str(&format!("  {line}\n"));
                    }
                    out.push_str("```\n");
                }
            }
            out.push('\n');
        }

        out.push_str("## Recent Event Stream (Tail)\n");
        if self.recent_events.is_empty() {
            out.push_str("No previous events recorded.\n\n");
        } else {
            for ev in &self.recent_events {
                out.push_str(&format!("- {}\n", ev.name()));
            }
            out.push('\n');
        }

        out.push_str("## Continuation Directives\n");
        out.push_str(
            "1. No self-reported progress percentages or completion assertions are accepted.\n",
        );
        out.push_str("2. Continue from the observed diff and failing test diagnostics above.\n");
        out.push_str("3. All verification tests must exit with code 0 to achieve completion.\n");

        out
    }

    /// Save the package atomically to disk with owner-only 0600 permissions.
    pub fn save_to_file(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("cannot serialize worker package: {e}"))?;
        crate::capture::write_atomic(path, json.as_bytes())
    }

    /// Load the package from disk, strictly validating schema and rejecting progress claims.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("invalid worker package JSON in {}: {e}", path.display()))
    }
}

/// Outcome of a second worker resuming a task from a worker continuation package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResumptionOutcome {
    pub task_id: String,
    pub resuming_agent: String,
    pub runner_exit_code: i32,
    pub all_tests_passed: bool,
    pub post_test_runs: Vec<VerifiedTestRun>,
    pub updated_diff: ObservedDiff,
}

/// Resume a real task from a worker package alone.
///
/// Executes the second worker with the context prepared from the package,
/// then independently verifies repository state and test outcomes.
pub fn resume_task_from_package<F>(
    package: &WorkerPackage,
    repo_dir: &Path,
    resuming_agent: &str,
    test_commands: &[&str],
    worker_runner: F,
) -> Result<ResumptionOutcome, String>
where
    F: FnOnce(&str, &Path) -> Result<i32, String>,
{
    let context = package.resumption_context();
    let runner_exit_code = worker_runner(&context, repo_dir)?;

    let mut post_test_runs = Vec::new();
    for cmd in test_commands {
        post_test_runs.push(VerifiedTestRun::run(repo_dir, cmd)?);
    }

    let all_tests_passed = !post_test_runs.is_empty() && post_test_runs.iter().all(|t| t.passed);
    let updated_diff = ObservedDiff::from_repo(repo_dir)?;

    Ok(ResumptionOutcome {
        task_id: package.task_id.clone(),
        resuming_agent: resuming_agent.to_string(),
        runner_exit_code,
        all_tests_passed,
        post_test_runs,
        updated_diff,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn init_git_repo(path: &Path) {
        let run = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(path)
                .args(args)
                .status()
                .expect("git command failed");
            assert!(status.success());
        };

        run(&["init"]);
        run(&["config", "user.email", "tester@example.com"]);
        run(&["config", "user.name", "Tester"]);
        fs::write(path.join("file.txt"), "hello world\n").unwrap();
        run(&["add", "file.txt"]);
        run(&["commit", "-m", "initial commit"]);
    }

    #[test]
    fn package_built_from_real_repo_observes_diff_and_exit_codes() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);

        // Modify file.txt
        fs::write(repo.join("file.txt"), "hello world\nsecond line\n").unwrap();

        // Build package
        let package = WorkerPackage::build(
            repo,
            "task-42",
            "Add second line to file.txt",
            "codex",
            WorkerStopReason::ExitCode { code: 1 },
            &["test -f file.txt"],
            vec![],
        )
        .unwrap();

        assert_eq!(package.task_id, "task-42");
        assert_eq!(package.previous_agent, "codex");
        assert_eq!(package.stop_reason, WorkerStopReason::ExitCode { code: 1 });
        assert_eq!(package.observed_diff.changed_files, vec!["file.txt"]);
        assert!(package.observed_diff.insertions >= 1);
        assert_eq!(package.verified_tests.len(), 1);
        assert!(package.verified_tests[0].passed);
        assert_eq!(package.verified_tests[0].exit_code, 0);

        let context = package.resumption_context();
        assert!(context.contains("Task Continuation: task-42"));
        assert!(context.contains("codex"));
        assert!(context.contains("process exited with code 1"));
        assert!(context.contains("file.txt"));
        assert!(context.contains("No self-reported progress percentages"));
    }

    #[test]
    fn package_rejects_payload_with_self_reported_progress_claim() {
        // Construct JSON attempting to inject self-reported progress percentage
        let fraudulent_json = r#"{
            "task_id": "task-99",
            "task_description": "Fix memory leak",
            "previous_agent": "claude",
            "stop_reason": { "kind": "exit_code", "code": 1 },
            "observed_diff": {
                "changed_files": [],
                "patch": "",
                "insertions": 0,
                "deletions": 0
            },
            "verified_tests": [],
            "recent_events": [],
            "created_at_unix_ms": 1000,
            "progress_percent": 75,
            "completion_claim": "almost finished"
        }"#;

        let res: Result<WorkerPackage, _> = serde_json::from_str(fraudulent_json);
        assert!(
            res.is_err(),
            "package must refuse unverified progress or completion claims"
        );
        let err = res.unwrap_err().to_string();
        assert!(
            err.contains("unknown field `progress_percent`")
                || err.contains("unknown field `completion_claim`")
        );
    }

    #[test]
    fn atomic_save_and_load_package() {
        let dir = tempdir().unwrap();
        let pkg_path = dir.path().join("task-10.json");

        let package = WorkerPackage {
            task_id: "task-10".to_string(),
            task_description: "Refactor router".to_string(),
            previous_agent: "agy".to_string(),
            stop_reason: WorkerStopReason::TimedOut { duration_secs: 120 },
            observed_diff: ObservedDiff {
                changed_files: vec!["src/router.rs".to_string()],
                patch: "+ new route".to_string(),
                insertions: 1,
                deletions: 0,
            },
            verified_tests: vec![VerifiedTestRun {
                command: "cargo check".to_string(),
                exit_code: 101,
                output_tail: "error: syntax error".to_string(),
                passed: false,
            }],
            recent_events: vec![],
            created_at_unix_ms: 5555,
        };

        package.save_to_file(&pkg_path).unwrap();
        assert!(pkg_path.exists());

        let loaded = WorkerPackage::load_from_file(&pkg_path).unwrap();
        assert_eq!(loaded, package);
        assert_eq!(
            loaded.stop_reason,
            WorkerStopReason::TimedOut { duration_secs: 120 }
        );
    }

    #[test]
    fn second_worker_resumes_real_task_from_package_alone() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);

        // Worker 1 starts: edits file.txt into a failing state
        fs::write(repo.join("file.txt"), "incomplete work\n").unwrap();

        // Test command that checks if file contains "completed work"
        let test_cmd = "grep -q 'completed work' file.txt";

        // Build package representing worker 1 stopping with exit code 1
        let package = WorkerPackage::build(
            repo,
            "task-feature-x",
            "Ensure file.txt contains 'completed work'",
            "worker-1",
            WorkerStopReason::ExitCode { code: 1 },
            &[test_cmd],
            vec![],
        )
        .unwrap();

        // The initial test must have failed
        assert_eq!(package.verified_tests[0].exit_code, 1);
        assert!(!package.verified_tests[0].passed);

        // Package is saved and loaded
        let pkg_file = dir.path().join("continuation.json");
        package.save_to_file(&pkg_file).unwrap();
        let loaded_package = WorkerPackage::load_from_file(&pkg_file).unwrap();

        // Worker 2 resumes the task from loaded_package alone:
        // Reads context, implements the required change
        let outcome = resume_task_from_package(
            &loaded_package,
            repo,
            "worker-2",
            &[test_cmd],
            |context, target_dir| {
                // Assert that worker 2 receives the factual briefing without self-reported progress claims
                assert!(context.contains("task-feature-x"));
                assert!(context.contains("worker-1"));
                assert!(context.contains("incomplete work"));

                // Worker 2 writes the required text to pass the test
                fs::write(target_dir.join("file.txt"), "completed work\n").unwrap();
                Ok(0)
            },
        )
        .unwrap();

        assert_eq!(outcome.task_id, "task-feature-x");
        assert_eq!(outcome.resuming_agent, "worker-2");
        assert_eq!(outcome.runner_exit_code, 0);
        assert!(outcome.all_tests_passed);
        assert_eq!(outcome.post_test_runs.len(), 1);
        assert_eq!(outcome.post_test_runs[0].exit_code, 0);
        assert!(outcome.post_test_runs[0].passed);
    }
}
