//! OR-4 — leases, not shared checkouts.
//!
//! Two workers must never share a working tree checkout. Each worker is granted a
//! dedicated worktree lease with a declared file set.
//!
//! # Conflict Detection at Scheduling Time
//!
//! Shared file conflicts are detected and resolved BEFORE launching workers, not
//! discovered at merge time after conflicting modifications have landed on disk.
//! If two workers request overlapping files, the second worker is told to wait
//! before it is launched. Once the first worker releases its lease, the waiting
//! worker is unblocked and granted its dedicated worktree.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Information about a conflict between two workers' declared file sets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseConflict {
    /// The worker requesting the lease.
    pub requesting_worker: String,
    /// The task id requesting the lease.
    pub requesting_task: String,
    /// The specific file path that caused the conflict.
    pub conflicting_file: String,
    /// The worker that currently holds the active lease on this file.
    pub held_by_worker: String,
    /// The task id that currently holds the active lease on this file.
    pub held_by_task: String,
}

impl std::fmt::Display for LeaseConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "file '{}' requested by worker '{}' for task '{}' is actively held by worker '{}' for task '{}'",
            self.conflicting_file,
            self.requesting_worker,
            self.requesting_task,
            self.held_by_worker,
            self.held_by_task
        )
    }
}

/// An active lease granted to a worker for an exclusive worktree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerLease {
    /// Unique lease identifier.
    pub lease_id: String,
    /// Worker identifier (e.g. `codex`, `claude`, `agent-1`).
    pub worker_id: String,
    /// Task identifier (e.g. `task-auth`, `task-split`).
    pub task_id: String,
    /// Path to the dedicated worktree directory allocated for this worker.
    pub worktree_path: PathBuf,
    /// Git branch associated with this lease worktree.
    pub branch: String,
    /// The declared file set permitted for modification by this worker.
    pub declared_files: Vec<String>,
    /// Creation timestamp in milliseconds since Unix epoch.
    pub created_at_unix_ms: u64,
}

/// Decision made by the lease scheduler upon receiving a lease request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum LeaseDecision {
    /// The lease was granted. A dedicated worktree was allocated and the worker may launch.
    Granted(WorkerLease),
    /// A declared file conflicts with an active lease. The worker must wait before being launched.
    WaitBeforeLaunch { conflict: LeaseConflict },
    /// The lease request was rejected (e.g. illegal path climbing or invalid arguments).
    Refused { reason: String },
}

/// A request waiting in the queue because of a declared file conflict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaitingRequest {
    pub worker_id: String,
    pub task_id: String,
    pub branch: String,
    pub declared_files: Vec<String>,
    pub conflict: LeaseConflict,
}

/// The result of attempting to schedule and run a worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleOutcome<R> {
    /// The worker was granted a lease and executed.
    Executed {
        lease: WorkerLease,
        result: R,
        unblocked_next: Option<WorkerLease>,
    },
    /// The worker was told to wait before launch due to a declared file conflict.
    WaitingBeforeLaunch { conflict: LeaseConflict },
    /// The request was refused.
    Refused { reason: String },
}

/// The registry managing worker leases and scheduling queues.
#[derive(Debug, Clone)]
pub struct LeaseRegistry {
    repo_root: PathBuf,
    active_leases: BTreeMap<String, WorkerLease>,
    waiting_queue: Vec<WaitingRequest>,
}

impl LeaseRegistry {
    pub fn new(repo_root: PathBuf) -> Self {
        Self {
            repo_root,
            active_leases: BTreeMap::new(),
            waiting_queue: Vec::new(),
        }
    }

    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    pub fn active_leases(&self) -> Vec<&WorkerLease> {
        self.active_leases.values().collect()
    }

    pub fn waiting_queue(&self) -> &[WaitingRequest] {
        &self.waiting_queue
    }

    /// Normalize relative path and reject climbing outside workspace.
    fn normalize_path(path_str: &str) -> Result<String, String> {
        let p = Path::new(path_str);
        if p.is_absolute() {
            return Err(format!("declared file cannot be absolute: '{path_str}'"));
        }
        let mut clean = Vec::new();
        for comp in p.components() {
            match comp {
                Component::ParentDir => {
                    return Err(format!(
                        "declared file cannot climb out of workspace: '{path_str}'"
                    ))
                }
                Component::Normal(part) => clean.push(part.to_string_lossy().to_string()),
                Component::CurDir => {}
                _ => {}
            }
        }
        if clean.is_empty() {
            return Err("declared file cannot be empty".to_string());
        }
        Ok(clean.join("/"))
    }

    /// Request a lease for a worker.
    ///
    /// If any declared file conflicts with an active lease, returns `WaitBeforeLaunch`
    /// and enqueues the request. The worker is NOT launched.
    pub fn request_lease(
        &mut self,
        worker_id: &str,
        task_id: &str,
        branch: &str,
        declared_files: &[String],
    ) -> Result<LeaseDecision, String> {
        if declared_files.is_empty() {
            return Ok(LeaseDecision::Refused {
                reason: "declared file set cannot be empty; worker must declare target files"
                    .to_string(),
            });
        }

        let mut normalized_files = Vec::new();
        for f in declared_files {
            match Self::normalize_path(f) {
                Ok(clean) => {
                    if !normalized_files.contains(&clean) {
                        normalized_files.push(clean);
                    }
                }
                Err(err) => return Ok(LeaseDecision::Refused { reason: err }),
            }
        }

        // Check for conflicts against active leases
        for active in self.active_leases.values() {
            for req_file in &normalized_files {
                if active.declared_files.contains(req_file) {
                    let conflict = LeaseConflict {
                        requesting_worker: worker_id.to_string(),
                        requesting_task: task_id.to_string(),
                        conflicting_file: req_file.clone(),
                        held_by_worker: active.worker_id.clone(),
                        held_by_task: active.task_id.clone(),
                    };
                    self.waiting_queue.push(WaitingRequest {
                        worker_id: worker_id.to_string(),
                        task_id: task_id.to_string(),
                        branch: branch.to_string(),
                        declared_files: normalized_files,
                        conflict: conflict.clone(),
                    });
                    return Ok(LeaseDecision::WaitBeforeLaunch { conflict });
                }
            }
        }

        // No conflicts: allocate dedicated worktree and grant lease
        let lease = self.provision_worktree(worker_id, task_id, branch, normalized_files)?;
        self.active_leases
            .insert(lease.lease_id.clone(), lease.clone());
        Ok(LeaseDecision::Granted(lease))
    }

    fn provision_worktree(
        &self,
        worker_id: &str,
        task_id: &str,
        branch: &str,
        declared_files: Vec<String>,
    ) -> Result<WorkerLease, String> {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let safe_worker = worker_id.replace(['/', ' '], "-");
        let safe_task = task_id.replace(['/', ' '], "-");
        let lease_id = format!("lease-{safe_worker}-{safe_task}");

        let repo_name = self
            .repo_root
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("repo");
        let parent = self.repo_root.parent().unwrap_or(&self.repo_root);
        let worktree_path = parent.join(format!("{repo_name}-{lease_id}-{timestamp_ms}"));

        // If repo_root has .git directory, use real git worktree add
        if self.repo_root.join(".git").exists() {
            let status = std::process::Command::new("git")
                .arg("-C")
                .arg(&self.repo_root)
                .args(["worktree", "add"])
                .arg("-b")
                .arg(branch)
                .arg(&worktree_path)
                .status()
                .or_else(|_| {
                    std::process::Command::new("git")
                        .arg("-C")
                        .arg(&self.repo_root)
                        .args(["worktree", "add"])
                        .arg(&worktree_path)
                        .arg(branch)
                        .status()
                })
                .map_err(|e| format!("git worktree add failed: {e}"))?;

            if !status.success() {
                // If branch creation failed (e.g. branch exists), try attaching without -b
                let status_retry = std::process::Command::new("git")
                    .arg("-C")
                    .arg(&self.repo_root)
                    .args(["worktree", "add", "--detach"])
                    .arg(&worktree_path)
                    .status()
                    .map_err(|e| format!("git worktree add --detach failed: {e}"))?;
                if !status_retry.success() {
                    return Err(format!(
                        "failed to create git worktree at {}",
                        worktree_path.display()
                    ));
                }
            }
        } else {
            // Non-git filesystem isolation fallback
            std::fs::create_dir_all(&worktree_path)
                .map_err(|e| format!("cannot create worktree dir: {e}"))?;
        }

        Ok(WorkerLease {
            lease_id,
            worker_id: worker_id.to_string(),
            task_id: task_id.to_string(),
            worktree_path,
            branch: branch.to_string(),
            declared_files,
            created_at_unix_ms: timestamp_ms,
        })
    }

    /// Release an active lease and unblock any waiting worker whose conflicts are cleared.
    pub fn release_lease(&mut self, lease_id: &str) -> Result<Option<WorkerLease>, String> {
        let removed = self.active_leases.remove(lease_id);
        if let Some(ref lease) = removed {
            // Clean up git worktree if git checkout
            if self.repo_root.join(".git").exists() {
                let _ = std::process::Command::new("git")
                    .arg("-C")
                    .arg(&self.repo_root)
                    .args(["worktree", "remove", "--force"])
                    .arg(&lease.worktree_path)
                    .output();
            } else {
                let _ = std::fs::remove_dir_all(&lease.worktree_path);
            }
        }

        // Check if any waiting request is now unblocked
        let mut unblocked_idx = None;
        for (i, req) in self.waiting_queue.iter().enumerate() {
            let has_conflict = self.active_leases.values().any(|active| {
                req.declared_files
                    .iter()
                    .any(|f| active.declared_files.contains(f))
            });
            if !has_conflict {
                unblocked_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = unblocked_idx {
            let req = self.waiting_queue.remove(idx);
            let next_lease = self.provision_worktree(
                &req.worker_id,
                &req.task_id,
                &req.branch,
                req.declared_files,
            )?;
            self.active_leases
                .insert(next_lease.lease_id.clone(), next_lease.clone());
            return Ok(Some(next_lease));
        }

        Ok(None)
    }

    /// Schedule a worker task with lease protection.
    ///
    /// If there is a declared file conflict, the worker is NOT launched and is told
    /// to wait. If granted, the worker runner closure executes inside the dedicated worktree.
    pub fn schedule_and_run<F, R>(
        &mut self,
        worker_id: &str,
        task_id: &str,
        branch: &str,
        declared_files: &[String],
        worker_runner: F,
    ) -> Result<ScheduleOutcome<R>, String>
    where
        F: FnOnce(&WorkerLease) -> Result<R, String>,
    {
        match self.request_lease(worker_id, task_id, branch, declared_files)? {
            LeaseDecision::WaitBeforeLaunch { conflict } => {
                // The worker runner closure is NOT called. The worker was told to wait before launch.
                Ok(ScheduleOutcome::WaitingBeforeLaunch { conflict })
            }
            LeaseDecision::Refused { reason } => Ok(ScheduleOutcome::Refused { reason }),
            LeaseDecision::Granted(lease) => {
                let result = worker_runner(&lease)?;
                let unblocked_next = self.release_lease(&lease.lease_id)?;
                Ok(ScheduleOutcome::Executed {
                    lease,
                    result,
                    unblocked_next,
                })
            }
        }
    }
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
                .expect("git setup failed");
            assert!(status.success());
        };
        run(&["init"]);
        run(&["config", "user.email", "tester@example.com"]);
        run(&["config", "user.name", "Tester"]);
        fs::write(path.join("file1.rs"), "// file 1\n").unwrap();
        fs::write(path.join("file2.rs"), "// file 2\n").unwrap();
        run(&["add", "file1.rs", "file2.rs"]);
        run(&["commit", "-m", "initial commit"]);
    }

    #[test]
    fn two_workers_requesting_same_file_tells_second_to_wait_before_launch() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);

        let mut registry = LeaseRegistry::new(repo.to_path_buf());

        // 1. Worker 1 requests lease for file1.rs
        let mut worker_1_ran = false;
        let outcome_1 = registry
            .schedule_and_run(
                "worker-alpha",
                "task-1",
                "branch-1",
                &["file1.rs".to_string()],
                |lease| {
                    worker_1_ran = true;
                    assert!(lease.worktree_path.exists());
                    assert_eq!(lease.worker_id, "worker-alpha");
                    // Worker 1 writes in its dedicated worktree
                    fs::write(lease.worktree_path.join("file1.rs"), "// edited by alpha\n")
                        .unwrap();
                    Ok("alpha-done")
                },
            )
            .unwrap();

        assert!(worker_1_ran);
        assert!(matches!(outcome_1, ScheduleOutcome::Executed { .. }));

        // Now test concurrent conflict: Worker 1 actively holds file1.rs
        let lease_1 = match registry.request_lease(
            "worker-1",
            "task-edit-1",
            "feature-1",
            &["file1.rs".to_string()],
        ) {
            Ok(LeaseDecision::Granted(l)) => l,
            other => panic!("expected granted lease, got: {other:?}"),
        };

        // Worker 2 asks for the exact same file file1.rs while worker 1 is active
        let mut worker_2_launched = false;
        let outcome_2 = registry
            .schedule_and_run(
                "worker-2",
                "task-edit-2",
                "feature-2",
                &["file1.rs".to_string()],
                |_lease| {
                    worker_2_launched = true;
                    Ok("should not run")
                },
            )
            .unwrap();

        // Verification of done-when: worker 2 was told to wait before it was launched!
        assert!(
            !worker_2_launched,
            "Worker 2 must NOT be launched when asking for an actively leased file!"
        );
        match outcome_2 {
            ScheduleOutcome::WaitingBeforeLaunch { conflict } => {
                assert_eq!(conflict.conflicting_file, "file1.rs");
                assert_eq!(conflict.held_by_worker, "worker-1");
                assert_eq!(conflict.requesting_worker, "worker-2");
            }
            other => panic!("expected WaitingBeforeLaunch, got {other:?}"),
        }

        assert_eq!(registry.waiting_queue().len(), 1);

        // 3. Independent Worker 3 asks for file2.rs (no conflict) and is granted immediately
        let mut worker_3_launched = false;
        let outcome_3 = registry
            .schedule_and_run(
                "worker-3",
                "task-edit-3",
                "feature-3",
                &["file2.rs".to_string()],
                |lease| {
                    worker_3_launched = true;
                    assert!(lease.worktree_path.exists());
                    Ok("worker-3-success")
                },
            )
            .unwrap();

        assert!(
            worker_3_launched,
            "Worker 3 with disjoint file set must launch concurrently"
        );
        assert!(matches!(outcome_3, ScheduleOutcome::Executed { .. }));

        // 4. Worker 1 releases its lease -> Worker 2 is unblocked and granted lease!
        let unblocked = registry.release_lease(&lease_1.lease_id).unwrap();
        assert!(unblocked.is_some());
        let lease_2 = unblocked.unwrap();
        assert_eq!(lease_2.worker_id, "worker-2");
        assert_eq!(lease_2.task_id, "task-edit-2");
        assert!(lease_2.worktree_path.exists());
        assert_eq!(registry.waiting_queue().len(), 0);

        // Clean up Worker 2's lease
        let _ = registry.release_lease(&lease_2.lease_id);
    }

    #[test]
    fn invalid_path_climbing_is_refused_at_scheduling_time() {
        let dir = tempdir().unwrap();
        let mut registry = LeaseRegistry::new(dir.path().to_path_buf());

        let decision = registry
            .request_lease(
                "worker-rogue",
                "task-escape",
                "branch-bad",
                &["../../etc/passwd".to_string()],
            )
            .unwrap();

        match decision {
            LeaseDecision::Refused { reason } => {
                assert!(reason.contains("climb out of workspace"));
            }
            other => panic!("expected Refused, got {other:?}"),
        }
    }
}
