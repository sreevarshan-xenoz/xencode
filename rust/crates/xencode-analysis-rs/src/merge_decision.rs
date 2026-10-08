//! OR-5 — the merge decision.
//!
//! Autonomous workers develop on isolated branches. Merging them into the base
//! repository requires:
//!
//! 1. `git merge-tree` conflict detection: speculative merge inspection without
//!    touching the working tree or index.
//! 2. A rendered conflict report showing the exact conflicting files and diff markers.
//! 3. An evidence-backed verdict per branch holding the worker's checks.
//! 4. A human gate to land anything: nothing merges without a decision a named human made.
//! 5. Post-integration verification: after integration, tests are re-run on the
//!    merged tree and kept distinct from the workers' own runs.
//! 6. `OR-17`'s veto: a review or verification outcome recorded against a branch
//!    blocks the land, and [`crate::veto`] holds what that takes.

use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::veto::{open_vetoes_for, Veto};

/// Result of checking a candidate branch with git merge-tree against a base branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergePrecheck {
    /// Whether the branch merges cleanly without conflicts into the base.
    pub clean: bool,
    /// Files with conflicts, if any.
    pub conflict_files: Vec<String>,
    /// The rendered conflict diff showing conflict markers.
    pub rendered_conflict: Option<String>,
}

/// An individual check executed on a branch by a worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchCheck {
    pub name: String,
    pub exit_code: i32,
    pub passed: bool,
    pub evidence_ref: String,
}

/// Specification for a branch to be evaluated for merging.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchSpec {
    pub branch: String,
    pub worker: String,
    pub task_id: String,
    pub checks: Vec<BranchCheck>,
}

/// The evidence-backed verdict for a single candidate branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchVerdict {
    pub branch: String,
    pub worker: String,
    pub task_id: String,
    pub commit_sha: String,
    pub changed_files: Vec<String>,
    pub worker_checks: Vec<BranchCheck>,
    pub precheck: MergePrecheck,
    /// Every review or verification outcome still blocking this branch (`OR-17`).
    /// The plan carries them so a reader sees the reason and who may lift it,
    /// and `execute_merge` re-reads them from disk so a plan that left them out
    /// cannot land the branch either.
    pub vetoes: Vec<Veto>,
    pub eligible: bool,
}

/// An overall merge plan evaluating one or more candidate branches against a base branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergePlan {
    pub base_branch: String,
    pub branches: Vec<BranchVerdict>,
    pub all_clean: bool,
    pub all_worker_checks_passed: bool,
}

/// A human decision authorizing or rejecting a merge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanMergeApproval {
    /// The full name or identifier of the human making the decision (e.g. "Alice").
    pub approved_by: String,
    /// Whether the merge is granted or rejected.
    pub decision: bool,
    /// Rationale or note provided by the human.
    pub reason: String,
    /// Timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

impl HumanMergeApproval {
    pub fn new(approved_by: impl Into<String>, decision: bool, reason: impl Into<String>) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            approved_by: approved_by.into(),
            decision,
            reason: reason.into(),
            timestamp_ms,
        }
    }
}

/// A verification test executed on the combined integration tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrationCheck {
    pub command: String,
    pub exit_code: i32,
    pub passed: bool,
    pub output_tail: String,
}

/// The final outcome of a merge operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeOutcome {
    pub base_branch: String,
    pub merged_branches: Vec<String>,
    pub approved_by: String,
    pub post_integration_checks: Vec<IntegrationCheck>,
    pub all_post_checks_passed: bool,
    pub integration_commit: Option<String>,
}

/// Speculatively evaluate whether `candidate_branch` merges cleanly into `base_branch`
/// using `git merge-tree`, capturing rendered conflicts if present.
pub fn precheck_branch(
    repo_root: &Path,
    base_branch: &str,
    candidate_branch: &str,
) -> Result<MergePrecheck, String> {
    // 1. Try modern git merge-tree --write-tree
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["merge-tree", "--write-tree", base_branch, candidate_branch])
        .output()
        .or_else(|_| {
            // Fallback to 3-way merge-tree
            let mb = Command::new("git")
                .arg("-C")
                .arg(repo_root)
                .args(["merge-base", base_branch, candidate_branch])
                .output()?;
            let base_sha = String::from_utf8_lossy(&mb.stdout).trim().to_string();
            Command::new("git")
                .arg("-C")
                .arg(repo_root)
                .args(["merge-tree", &base_sha, base_branch, candidate_branch])
                .output()
        })
        .map_err(|e| format!("git merge-tree failed in {}: {e}", repo_root.display()))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = format!("{stdout}\n{stderr}");

    let has_conflict = !output.status.success()
        || combined.contains("CONFLICT")
        || combined.contains("<<<<<<<")
        || combined.contains("changed in both");

    if !has_conflict {
        return Ok(MergePrecheck {
            clean: true,
            conflict_files: Vec::new(),
            rendered_conflict: None,
        });
    }

    // Extract conflicting file paths
    let mut conflict_files = Vec::new();
    for line in combined.lines() {
        if let Some(rest) = line.strip_prefix("CONFLICT (content): Merge conflict in ") {
            conflict_files.push(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("CONFLICT (add/add): Merge conflict in ") {
            conflict_files.push(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("CONFLICT (modify/delete): ") {
            conflict_files.push(rest.trim().to_string());
        } else if line.starts_with("changed in both") {
            // 3-way merge-tree output format
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(path) = parts.last() {
                conflict_files.push(path.to_string());
            }
        }
    }

    // Extract rendered conflict excerpt
    let conflict_excerpt = if combined.contains("<<<<<<<") {
        let mut excerpt = Vec::new();
        let mut in_conflict = false;
        for line in combined.lines() {
            if line.contains("<<<<<<<") {
                in_conflict = true;
            }
            if in_conflict {
                excerpt.push(line);
                if line.contains(">>>>>>>") {
                    in_conflict = false;
                }
            }
        }
        if excerpt.is_empty() {
            combined.lines().take(40).collect::<Vec<_>>().join("\n")
        } else {
            excerpt.join("\n")
        }
    } else {
        combined.lines().take(40).collect::<Vec<_>>().join("\n")
    };

    Ok(MergePrecheck {
        clean: false,
        conflict_files,
        rendered_conflict: Some(conflict_excerpt),
    })
}

/// Construct a comprehensive merge plan evaluating candidate branches against a base branch.
pub fn build_merge_plan(
    repo_root: &Path,
    base_branch: &str,
    branch_specs: &[BranchSpec],
) -> Result<MergePlan, String> {
    let mut branch_verdicts = Vec::new();
    let mut all_clean = true;
    let mut all_worker_checks_passed = true;

    for spec in branch_specs {
        // Find commit SHA
        let sha_out = Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["rev-parse", &spec.branch])
            .output()
            .map_err(|e| format!("git rev-parse failed for {}: {e}", spec.branch))?;
        let commit_sha = if sha_out.status.success() {
            String::from_utf8_lossy(&sha_out.stdout).trim().to_string()
        } else {
            return Err(format!(
                "branch '{}' does not exist in repository",
                spec.branch
            ));
        };

        // Find changed files against base branch
        let diff_out = Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["diff", "--name-only", base_branch, &spec.branch])
            .output()
            .map_err(|e| format!("git diff failed: {e}"))?;
        let changed_files: Vec<String> = String::from_utf8_lossy(&diff_out.stdout)
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();

        // Run git merge-tree precheck
        let precheck = precheck_branch(repo_root, base_branch, &spec.branch)?;
        if !precheck.clean {
            all_clean = false;
        }

        // OR-17: a review or verification outcome recorded against this branch is
        // a check that failed. It arrives from the veto file rather than from the
        // caller's spec, because the party that was blocked is not the party that
        // gets to say which checks existed.
        let vetoes = open_vetoes_for(repo_root, &spec.branch)?;
        let mut checks = spec.checks.clone();
        for veto in &vetoes {
            checks.push(BranchCheck {
                name: format!("veto:{}", veto.id),
                exit_code: 1,
                passed: false,
                evidence_ref: veto.evidence_ref(),
            });
        }

        let checks_ok = !checks.is_empty() && checks.iter().all(|c| c.passed);
        if !checks_ok {
            all_worker_checks_passed = false;
        }

        let eligible = precheck.clean && checks_ok;

        branch_verdicts.push(BranchVerdict {
            branch: spec.branch.clone(),
            worker: spec.worker.clone(),
            task_id: spec.task_id.clone(),
            commit_sha,
            changed_files,
            worker_checks: checks,
            precheck,
            vetoes,
            eligible,
        });
    }

    Ok(MergePlan {
        base_branch: base_branch.to_string(),
        branches: branch_verdicts,
        all_clean,
        all_worker_checks_passed,
    })
}

/// Execute a merge operation governed by a named human approval gate.
///
/// Four things have to hold before anything is written: a named human decided
/// yes, no veto is open on any branch in the plan, every recorded worker check
/// passed, and the tree merges clean. The veto check reads the disk again rather
/// than trusting the plan, so a stale or hand-built plan is not a way around it.
///
/// Re-runs post-integration verification tests on the resulting integrated tree.
pub fn execute_merge(
    repo_root: &Path,
    plan: &MergePlan,
    approval: &HumanMergeApproval,
    post_integration_test_cmds: &[&str],
) -> Result<MergeOutcome, String> {
    // 1. HUMAN APPROVAL GATE: Nothing merges without a decision a named human made!
    if approval.approved_by.trim().is_empty() {
        return Err("merge refused: no named human approved the merge decision (human approval gate required)".to_string());
    }
    if !approval.decision {
        return Err(format!(
            "merge refused by {}: {}",
            approval.approved_by, approval.reason
        ));
    }

    // 2. VETO GATE (`OR-17`): re-read from disk, not from the plan in hand. A
    //    caller that built a plan before the veto landed — or built one that
    //    simply left the field out — still cannot merge the branch.
    let mut blocked: Vec<String> = Vec::new();
    for verdict in &plan.branches {
        for veto in open_vetoes_for(repo_root, &verdict.branch)? {
            blocked.push(format!(
                "{} on '{}' — {}",
                veto.summary_line(),
                verdict.branch,
                veto.who_may_clear()
            ));
        }
    }
    if !blocked.is_empty() {
        return Err(format!(
            "merge refused: {} open veto(es), and a veto is not the worker's to lift\n{}",
            blocked.len(),
            blocked.join("\n")
        ));
    }

    // 3. WORKER-CHECK GATE: every check the plan recorded has to have passed.
    //    This flag was computed and printed and then never consulted; an empty
    //    check list is not a pass either, so "nothing proven" cannot read as
    //    "green" — the same rule the result envelope holds itself to.
    if !plan.all_worker_checks_passed {
        let failures: Vec<String> = plan
            .branches
            .iter()
            .flat_map(|verdict| {
                let branch = verdict.branch.clone();
                verdict
                    .worker_checks
                    .iter()
                    .filter(|check| !check.passed)
                    .map(move |check| {
                        format!(
                            "{}: {} exited {} ({})",
                            branch, check.name, check.exit_code, check.evidence_ref
                        )
                    })
            })
            .collect();
        let unknown: Vec<String> = plan
            .branches
            .iter()
            .filter(|verdict| verdict.worker_checks.is_empty())
            .map(|verdict| verdict.branch.clone())
            .collect();
        let mut detail = String::new();
        if !failures.is_empty() {
            detail += &format!("failed checks:\n{}", failures.join("\n"));
        }
        if !unknown.is_empty() {
            detail += &format!(
                "{}branches with no recorded check: {}",
                if detail.is_empty() { "" } else { "\n" },
                unknown.join(", ")
            );
        }
        return Err(format!("merge refused: {detail}"));
    }

    // 4. CONFLICT GATE
    if !plan.all_clean {
        let conflicts: Vec<String> = plan
            .branches
            .iter()
            .filter(|b| !b.precheck.clean)
            .map(|b| format!("{} (conflicts: {:?})", b.branch, b.precheck.conflict_files))
            .collect();
        return Err(format!(
            "merge refused: plan contains branches with merge conflicts: {}",
            conflicts.join(", ")
        ));
    }

    // 5. Ensure base branch is checked out
    let checkout = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["checkout", &plan.base_branch])
        .output()
        .map_err(|e| format!("git checkout failed: {e}"))?;
    if !checkout.status.success() {
        return Err(format!(
            "failed to checkout base branch '{}': {}",
            plan.base_branch,
            String::from_utf8_lossy(&checkout.stderr).trim()
        ));
    }

    // 6. Merge each candidate branch
    let mut merged_branches = Vec::new();
    for bv in &plan.branches {
        let msg = format!(
            "Merge branch '{}' into '{}' (approved by {})",
            bv.branch, plan.base_branch, approval.approved_by
        );
        let merge_res = Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .args(["merge", "--no-ff", "-m", &msg, &bv.branch])
            .output()
            .map_err(|e| format!("failed to merge branch '{}': {e}", bv.branch))?;

        if !merge_res.status.success() {
            return Err(format!(
                "merge failed for branch '{}': {}",
                bv.branch,
                String::from_utf8_lossy(&merge_res.stderr).trim()
            ));
        }
        merged_branches.push(bv.branch.clone());
    }

    // 7. POST-INTEGRATION VERIFICATION: re-run tests on the integrated tree,
    //    which is a different run from the one each worker did on its own.
    let mut post_integration_checks = Vec::new();
    for cmd_str in post_integration_test_cmds {
        let out = Command::new("sh")
            .arg("-c")
            .arg(cmd_str)
            .current_dir(repo_root)
            .output()
            .map_err(|e| format!("failed to run post-integration test '{cmd_str}': {e}"))?;

        let exit_code = out.status.code().unwrap_or(1);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        let combined = format!("{stdout}\n{stderr}");
        let tail = combined
            .lines()
            .rev()
            .take(15)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");

        post_integration_checks.push(IntegrationCheck {
            command: cmd_str.to_string(),
            exit_code,
            passed: exit_code == 0,
            output_tail: tail,
        });
    }

    let all_post_checks_passed =
        post_integration_checks.is_empty() || post_integration_checks.iter().all(|c| c.passed);

    // Get integration commit SHA
    let head_sha = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());

    Ok(MergeOutcome {
        base_branch: plan.base_branch.clone(),
        merged_branches,
        approved_by: approval.approved_by.clone(),
        post_integration_checks,
        all_post_checks_passed,
        integration_commit: head_sha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn init_git_repo(path: &Path) {
        let run = |args: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(path)
                .args(args)
                .status()
                .expect("git setup failed");
            assert!(status.success());
        };
        run(&["init"]);
        run(&["config", "user.email", "merger@example.com"]);
        run(&["config", "user.name", "Merger"]);
        fs::write(path.join("base.txt"), "base file\n").unwrap();
        run(&["add", "base.txt"]);
        run(&["commit", "-m", "initial base commit"]);
    }

    fn create_branch_with_commit(repo: &Path, branch: &str, file: &str, content: &str) {
        let run = |args: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(repo)
                .args(args)
                .status()
                .expect("git branch failed");
            assert!(status.success());
        };
        run(&["checkout", "master"]);
        run(&["checkout", "-b", branch]);
        fs::write(repo.join(file), content).unwrap();
        run(&["add", file]);
        run(&["commit", "-m", &format!("commit on {branch}")]);
        run(&["checkout", "master"]);
    }

    #[test]
    fn precheck_detects_clean_and_conflicting_branches_with_rendered_diff() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);

        // Branch 1: clean modification to file1.txt
        create_branch_with_commit(repo, "branch-clean", "file1.txt", "clean content\n");

        // Branch 2: divergent modification to base.txt (created before master commits its own change)
        create_branch_with_commit(
            repo,
            "branch-conflict",
            "base.txt",
            "conflicting base edit\n",
        );

        // Base edit to base.txt on master creating a divergent 3-way conflict with branch-conflict
        let run = |args: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(repo)
                .args(args)
                .status()
                .unwrap();
            assert!(status.success());
        };
        run(&["checkout", "master"]);
        fs::write(repo.join("base.txt"), "base modified by master\n").unwrap();
        run(&["commit", "-am", "master modified base"]);

        // 1. Precheck clean branch
        let clean_check = precheck_branch(repo, "master", "branch-clean").unwrap();
        assert!(clean_check.clean, "branch-clean must be detected as clean");
        assert!(clean_check.conflict_files.is_empty());
        assert!(clean_check.rendered_conflict.is_none());

        // 2. Precheck conflicting branch
        let conflict_check = precheck_branch(repo, "master", "branch-conflict").unwrap();
        assert!(
            !conflict_check.clean,
            "branch-conflict must be detected as conflicting"
        );
        assert!(conflict_check.rendered_conflict.is_some());
        let rendered = conflict_check.rendered_conflict.unwrap();
        assert!(rendered.contains("base.txt") || rendered.contains("CONFLICT"));
    }

    #[test]
    fn merge_refuses_without_named_human_decision() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);

        create_branch_with_commit(repo, "feature-x", "fx.txt", "feature x\n");

        let plan = build_merge_plan(
            repo,
            "master",
            &[BranchSpec {
                branch: "feature-x".to_string(),
                worker: "codex".to_string(),
                task_id: "task-x".to_string(),
                checks: vec![BranchCheck {
                    name: "unit-test".to_string(),
                    exit_code: 0,
                    passed: true,
                    evidence_ref: "logs/test.log".to_string(),
                }],
            }],
        )
        .unwrap();

        // 1. Attempt merge with empty approved_by
        let unapproved = HumanMergeApproval::new("", true, "no human named");
        let res = execute_merge(repo, &plan, &unapproved, &["true"]);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("no named human approved"));

        // 2. Attempt merge with decision = false
        let rejected = HumanMergeApproval::new("Alice", false, "not ready for merge");
        let res2 = execute_merge(repo, &plan, &rejected, &["true"]);
        assert!(res2.is_err());
        assert!(res2.unwrap_err().contains("merge refused by Alice"));
    }

    #[test]
    fn clean_four_branch_merge_shows_post_integration_tests() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);

        // Create 4 distinct branches with disjoint files
        create_branch_with_commit(repo, "feat-1", "mod1.txt", "content 1\n");
        create_branch_with_commit(repo, "feat-2", "mod2.txt", "content 2\n");
        create_branch_with_commit(repo, "feat-3", "mod3.txt", "content 3\n");
        create_branch_with_commit(repo, "feat-4", "mod4.txt", "content 4\n");

        let specs = vec![
            BranchSpec {
                branch: "feat-1".to_string(),
                worker: "worker-1".to_string(),
                task_id: "t1".to_string(),
                checks: vec![BranchCheck {
                    name: "test-w1".to_string(),
                    exit_code: 0,
                    passed: true,
                    evidence_ref: "ev1".to_string(),
                }],
            },
            BranchSpec {
                branch: "feat-2".to_string(),
                worker: "worker-2".to_string(),
                task_id: "t2".to_string(),
                checks: vec![BranchCheck {
                    name: "test-w2".to_string(),
                    exit_code: 0,
                    passed: true,
                    evidence_ref: "ev2".to_string(),
                }],
            },
            BranchSpec {
                branch: "feat-3".to_string(),
                worker: "worker-3".to_string(),
                task_id: "t3".to_string(),
                checks: vec![BranchCheck {
                    name: "test-w3".to_string(),
                    exit_code: 0,
                    passed: true,
                    evidence_ref: "ev3".to_string(),
                }],
            },
            BranchSpec {
                branch: "feat-4".to_string(),
                worker: "worker-4".to_string(),
                task_id: "t4".to_string(),
                checks: vec![BranchCheck {
                    name: "test-w4".to_string(),
                    exit_code: 0,
                    passed: true,
                    evidence_ref: "ev4".to_string(),
                }],
            },
        ];

        let plan = build_merge_plan(repo, "master", &specs).unwrap();
        assert!(plan.all_clean, "all four branches must be clean");
        assert!(plan.all_worker_checks_passed);
        assert_eq!(plan.branches.len(), 4);

        // A named human approves
        let approval = HumanMergeApproval::new("Alice", true, "Verified all 4 arms ready");

        // Execute merge with post-integration test commands
        let post_test_cmds = &[
            "test -f mod1.txt && test -f mod2.txt",
            "test -f mod3.txt && test -f mod4.txt",
            "grep -q 'content 4' mod4.txt",
        ];

        let outcome = execute_merge(repo, &plan, &approval, post_test_cmds).unwrap();

        assert_eq!(outcome.merged_branches.len(), 4);
        assert_eq!(outcome.approved_by, "Alice");
        assert!(outcome.all_post_checks_passed);
        assert_eq!(outcome.post_integration_checks.len(), 3);
        assert_eq!(outcome.post_integration_checks[0].exit_code, 0);
        assert_eq!(outcome.post_integration_checks[1].exit_code, 0);
        assert_eq!(outcome.post_integration_checks[2].exit_code, 0);
        assert!(outcome.post_integration_checks[0].passed);

        // Verify that all 4 files actually exist on disk in master
        assert!(repo.join("mod1.txt").exists());
        assert!(repo.join("mod2.txt").exists());
        assert!(repo.join("mod3.txt").exists());
        assert!(repo.join("mod4.txt").exists());
    }

    /// A passing plan built by the worker's own side of the pipeline, ready to
    /// land, with one clean branch on it.
    fn landable_plan(repo: &Path) -> MergePlan {
        build_merge_plan(
            repo,
            "master",
            &[BranchSpec {
                branch: "feature-x".to_string(),
                worker: "codex".to_string(),
                task_id: "task-x".to_string(),
                checks: vec![BranchCheck {
                    name: "unit-test".to_string(),
                    exit_code: 0,
                    passed: true,
                    evidence_ref: "logs/test.log".to_string(),
                }],
            }],
        )
        .unwrap()
    }

    /// The whole point of `OR-17`: a plan that says everything is fine is not
    /// enough, because the plan is not where the block lives. It was built before
    /// anyone objected, and `execute_merge` reads the veto file again.
    #[test]
    fn a_veto_lands_on_a_plan_that_already_said_it_was_ready() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);
        create_branch_with_commit(repo, "feature-x", "fx.txt", "feature x\n");

        let plan = landable_plan(repo);
        assert!(
            plan.all_worker_checks_passed,
            "the plan itself looks clean before the veto"
        );
        let approval = HumanMergeApproval::new("Alice", true, "looks good to me");
        assert!(
            execute_merge(repo, &plan, &approval, &["true"]).is_ok(),
            "and it lands while nothing objects"
        );

        // Back to a clean base and try again, this time with someone in the way.
        git(repo, &["reset", "--hard", "HEAD~1"]);
        let veto = crate::veto::record_veto(
            repo,
            "feature-x",
            "codex",
            crate::veto::VetoSource::Review,
            "Alice",
            "the retry swallows the error",
        )
        .unwrap();

        let err = execute_merge(repo, &plan, &approval, &["true"]).unwrap_err();
        assert!(err.contains("merge refused: 1 open veto"), "{err}");
        assert!(err.contains(&veto.id), "{err}");
        assert!(
            err.contains("the retry swallows the error"),
            "the reason is in the refusal: {err}"
        );
        assert!(
            err.contains("not codex, which is the worker this veto blocks"),
            "and so is who may clear it: {err}"
        );
        assert!(
            !repo.join("fx.txt").exists(),
            "the branch did not reach master"
        );
    }

    /// The plan is the reader's view of the block, so it has to carry it: an
    /// injected failed check, the veto attached, the branch ineligible.
    #[test]
    fn the_plan_shows_the_veto_and_stops_calling_the_branch_eligible() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);
        create_branch_with_commit(repo, "feature-x", "fx.txt", "feature x\n");
        crate::veto::record_veto(
            repo,
            "feature-x",
            "codex",
            crate::veto::VetoSource::Verification,
            "ci",
            "clippy exited 101",
        )
        .unwrap();

        let plan = landable_plan(repo);
        assert!(!plan.all_worker_checks_passed);
        let verdict = &plan.branches[0];
        assert!(!verdict.eligible);
        assert!(
            verdict.precheck.clean,
            "it still merges cleanly, and that is not enough"
        );
        assert_eq!(verdict.vetoes.len(), 1);
        assert_eq!(verdict.vetoes[0].reason, "clippy exited 101");
        assert!(verdict
            .worker_checks
            .iter()
            .any(|c| c.name == "veto:veto-0001" && !c.passed));
    }

    /// The half that was missing all along: `all_worker_checks_passed` was
    /// computed, printed and then never consulted. A failed check now blocks,
    /// and so does the absence of any check — nothing proven is nothing passed.
    #[test]
    fn a_failed_check_blocks_and_so_does_a_branch_with_no_checks_at_all() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);
        create_branch_with_commit(repo, "feature-x", "fx.txt", "feature x\n");
        let approval = HumanMergeApproval::new("Alice", true, "looks good to me");

        let red = build_merge_plan(
            repo,
            "master",
            &[BranchSpec {
                branch: "feature-x".to_string(),
                worker: "codex".to_string(),
                task_id: "task-x".to_string(),
                checks: vec![BranchCheck {
                    name: "unit-test".to_string(),
                    exit_code: 101,
                    passed: false,
                    evidence_ref: "logs/test.log".to_string(),
                }],
            }],
        )
        .unwrap();
        let err = execute_merge(repo, &red, &approval, &["true"]).unwrap_err();
        assert!(err.contains("merge refused: failed checks"), "{err}");
        assert!(err.contains("unit-test exited 101"), "{err}");
        assert!(!repo.join("fx.txt").exists());

        let unproven = build_merge_plan(
            repo,
            "master",
            &[BranchSpec {
                branch: "feature-x".to_string(),
                worker: "codex".to_string(),
                task_id: "task-x".to_string(),
                checks: vec![],
            }],
        )
        .unwrap();
        let err = execute_merge(repo, &unproven, &approval, &["true"]).unwrap_err();
        assert!(
            err.contains("no recorded check"),
            "an empty check list is not a pass: {err}"
        );
        assert!(!repo.join("fx.txt").exists());
    }

    /// A veto is a block, not a sentence: once someone with standing clears it,
    /// the same branch lands.
    #[test]
    fn clearing_the_veto_by_a_name_that_is_not_the_blocked_worker_lets_it_land() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);
        create_branch_with_commit(repo, "feature-x", "fx.txt", "feature x\n");
        crate::veto::record_veto(
            repo,
            "feature-x",
            "codex",
            crate::veto::VetoSource::Review,
            "Alice",
            "the retry swallows the error",
        )
        .unwrap();
        let plan = landable_plan(repo);
        let approval = HumanMergeApproval::new("Alice", true, "looks good to me");

        // The blocked party cannot be the one who unblocks itself, and saying so
        // does not change the record.
        assert!(crate::veto::clear_veto(repo, "veto-0001", "codex", None).is_err());
        assert!(execute_merge(repo, &plan, &approval, &["true"]).is_err());

        crate::veto::clear_veto(repo, "veto-0001", "Grace", None).unwrap();
        let after = landable_plan(repo);
        assert!(
            after.all_worker_checks_passed,
            "the veto is gone from the plan"
        );
        let outcome = execute_merge(repo, &after, &approval, &["true"]).unwrap();
        assert_eq!(outcome.merged_branches, vec!["feature-x".to_string()]);
        assert!(repo.join("fx.txt").exists());
    }

    /// A veto file that will not parse has to stop the plan, not clear it.
    #[test]
    fn a_corrupt_veto_file_stops_the_plan_rather_than_passing_it() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        init_git_repo(repo);
        create_branch_with_commit(repo, "feature-x", "fx.txt", "feature x\n");
        crate::veto::record_veto(
            repo,
            "feature-x",
            "codex",
            crate::veto::VetoSource::Review,
            "Alice",
            "x",
        )
        .unwrap();
        std::fs::write(crate::veto::veto_path(repo), "{ truncated").unwrap();

        let err = build_merge_plan(
            repo,
            "master",
            &[BranchSpec {
                branch: "feature-x".to_string(),
                worker: "codex".to_string(),
                task_id: "task-x".to_string(),
                checks: vec![BranchCheck {
                    name: "unit-test".to_string(),
                    exit_code: 0,
                    passed: true,
                    evidence_ref: "e".to_string(),
                }],
            }],
        )
        .unwrap_err();
        assert!(err.contains("unreadable veto list"), "{err}");
    }

    fn git(repo: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .status()
            .expect("git is on PATH");
        assert!(status.success(), "git {args:?} failed");
    }
}
