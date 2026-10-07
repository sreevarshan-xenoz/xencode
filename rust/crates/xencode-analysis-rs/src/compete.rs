//! Competing candidate implementations over git worktrees (AF-5).
//!
//! Run two or three candidate implementations on isolated branches, verify
//! each one with the toolchain checklist, and render an objective table of
//! `{ran, skipped, failed, evidence-ref}` per arm with NO composite score.
//!
//! A person then picks an arm, which checks out the selected implementation
//! while preserving the alternative candidate branch and its evidence on disk.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::toolchain;

/// One row in an arm's verification checklist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmCheckRow {
    /// Check identifier (`test`, `lint`, `fmt`).
    pub name: String,
    /// Whether the check executed.
    pub ran: bool,
    /// Whether the check was skipped.
    pub skipped: bool,
    /// Whether the check ran and exited non-zero.
    pub failed: bool,
    /// Relative or absolute path to the preserved evidence artifact.
    pub evidence_ref: String,
}

/// The result and verification status of one candidate arm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmResult {
    /// Short identifier for the arm (e.g. `arm-a`, `arm-1`).
    pub arm_id: String,
    /// Human-readable strategy or label.
    pub label: String,
    /// The git branch where the candidate implementation lives.
    pub branch: String,
    /// Working directory for this arm's worktree.
    pub worktree_path: PathBuf,
    /// Artifact session identifier.
    pub session: String,
    /// Machine verification rows for this arm.
    pub checks: Vec<ArmCheckRow>,
}

/// A complete competing candidate run comparing 2 or 3 arms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompetingReport {
    /// Unique run identifier.
    pub run_id: String,
    /// The original prompt or question with defensible candidate approaches.
    pub prompt: String,
    /// The candidate arms evaluated.
    pub arms: Vec<ArmResult>,
    /// The arm chosen by human decision, if picked.
    pub picked_arm: Option<String>,
    /// Creation timestamp in milliseconds since Unix epoch.
    pub created_at_unix_ms: u64,
}

/// Outcome of picking an arm by human action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PickOutcome {
    /// Run identifier.
    pub run_id: String,
    /// Identifier of the selected arm.
    pub picked_arm_id: String,
    /// Git branch checked out as the active implementation.
    pub picked_branch: String,
    /// Other candidate branches preserved on disk.
    pub other_branches: Vec<String>,
    /// Preserved evidence artifact directories on disk.
    pub preserved_evidence: Vec<PathBuf>,
}

/// Specification for a single candidate arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateArmSpec {
    /// Short identifier (e.g. `arm-a`).
    pub arm_id: String,
    /// Descriptive strategy label.
    pub label: String,
    /// Optional custom branch name (defaults to `compete/{run_id}/{arm_id}`).
    pub branch: Option<String>,
    /// Optional command to execute inside the arm's worktree.
    pub command: Option<String>,
    /// Optional file writes to apply in the arm's worktree: `(relative_path, content)`.
    pub file_edits: Vec<(PathBuf, String)>,
}

/// Configuration for running competing candidate arms.
#[derive(Debug, Clone)]
pub struct CompetingConfig {
    /// The question or requirement with competing answers.
    pub prompt: String,
    /// Candidate arms to evaluate (must be 2 or 3).
    pub arm_specs: Vec<CandidateArmSpec>,
    /// Check names to skip (e.g. `test`, `lint`, `fmt`).
    pub skip_checks: Vec<String>,
    /// Verification timeout in seconds.
    pub timeout_secs: u64,
}

/// Render the competing candidate report as a plain table.
///
/// In strict accordance with AF-5, this prints `{ran, skipped, failed, evidence-ref}`
/// per arm with NO composite score, aggregate percentages, or ranks.
pub fn format_competing_table(report: &CompetingReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("Competing Arms Run: {}\n", report.run_id));
    out.push_str(&format!("Prompt: {}\n", report.prompt));
    if let Some(ref picked) = report.picked_arm {
        out.push_str(&format!("Picked Arm: {}\n", picked));
    }
    out.push('\n');

    for arm in &report.arms {
        out.push_str(&format!("Arm: {} [branch: {}]\n", arm.arm_id, arm.branch));
        if !arm.label.is_empty() && arm.label != arm.arm_id {
            out.push_str(&format!("  Strategy: {}\n", arm.label));
        }
        out.push_str("  check   ran     skipped failed  evidence-ref\n");
        for row in &arm.checks {
            out.push_str(&format!(
                "  {:<7} {:<7} {:<7} {:<7} {}\n",
                row.name, row.ran, row.skipped, row.failed, row.evidence_ref,
            ));
        }
        out.push('\n');
    }
    out.trim_end().to_string()
}

/// Run two or three candidate implementations in isolated worktrees and record verification.
pub fn run_competing_arms(
    repo_root: &Path,
    config: &CompetingConfig,
) -> Result<CompetingReport, String> {
    let arm_specs = if config.arm_specs.is_empty() {
        vec![
            CandidateArmSpec {
                arm_id: "arm-a".to_string(),
                label: "Candidate Approach A".to_string(),
                branch: None,
                command: None,
                file_edits: Vec::new(),
            },
            CandidateArmSpec {
                arm_id: "arm-b".to_string(),
                label: "Candidate Approach B".to_string(),
                branch: None,
                command: None,
                file_edits: Vec::new(),
            },
        ]
    } else {
        config.arm_specs.clone()
    };

    if arm_specs.len() < 2 || arm_specs.len() > 3 {
        return Err(format!(
            "competing arms must have 2 or 3 candidate implementations (cost and scale bounded), got {}",
            arm_specs.len()
        ));
    }

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let run_id = format!("compete-{timestamp_ms}");

    let repo_name = repo_root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("repo");
    let parent = repo_root.parent().unwrap_or(repo_root);

    let mut arm_results = Vec::new();

    for spec in &arm_specs {
        let branch = spec
            .branch
            .clone()
            .unwrap_or_else(|| format!("compete/{}/{}", run_id, spec.arm_id));
        let session = format!("{}-{}", run_id, spec.arm_id);
        let worktree_dir = parent.join(format!("{repo_name}-{run_id}-{}", spec.arm_id));

        // Create isolated worktree on its own branch
        xencode_context_rs::worktree_add(repo_root, &worktree_dir, Some(&branch), true)
            .map_err(|e| format!("failed to create worktree for arm '{}': {}", spec.arm_id, e))?;

        // Apply file edits if specified
        if !spec.file_edits.is_empty() {
            for (rel_path, content) in &spec.file_edits {
                let full_path = worktree_dir.join(rel_path);
                if let Some(p) = full_path.parent() {
                    let _ = std::fs::create_dir_all(p);
                }
                std::fs::write(&full_path, content)
                    .map_err(|e| format!("failed writing {}: {}", rel_path.display(), e))?;
            }
        }

        // Apply command if specified. A candidate that failed to build must not
        // be verified anyway: the rows would describe a tree nobody produced.
        if let Some(ref cmd) = spec.command {
            let ran = std::process::Command::new("sh")
                .args(["-c", cmd])
                .current_dir(&worktree_dir)
                .output()
                .map_err(|e| {
                    format!(
                        "failed to execute candidate command for arm '{}': {}",
                        spec.arm_id, e
                    )
                })?;
            if !ran.status.success() {
                let code = ran
                    .status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "a signal".to_string());
                return Err(format!(
                    "candidate command for arm '{}' ({cmd}) exited with {code}: {}",
                    spec.arm_id,
                    String::from_utf8_lossy(&ran.stderr).trim()
                ));
            }
        }

        // Record candidate artifact in repo if no files or commands were specified
        if spec.file_edits.is_empty() && spec.command.is_none() {
            let doc_path =
                worktree_dir.join(format!("CANDIDATE_{}.md", spec.arm_id.to_uppercase()));
            let _ = std::fs::write(
                &doc_path,
                format!(
                    "# Candidate Implementation: {}\n\nPrompt: {}\nArm ID: {}\n",
                    spec.label, config.prompt, spec.arm_id
                ),
            );
        }

        // Commit candidate implementation in the branch worktree if dirty
        let status = std::process::Command::new("git")
            .current_dir(&worktree_dir)
            .args(["status", "--porcelain"])
            .output()
            .map_err(|e| format!("failed git status in worktree: {e}"))?;

        if !status.stdout.is_empty() {
            let staged = std::process::Command::new("git")
                .current_dir(&worktree_dir)
                .args(["add", "-A"])
                .output()
                .map_err(|e| format!("failed to stage arm '{}': {e}", spec.arm_id))?;
            if !staged.status.success() {
                return Err(format!(
                    "could not stage arm '{}': {}",
                    spec.arm_id,
                    String::from_utf8_lossy(&staged.stderr).trim()
                ));
            }
            let committed = std::process::Command::new("git")
                .current_dir(&worktree_dir)
                .args(["commit", "-m", &format!("arm: {}", spec.label)])
                .output()
                .map_err(|e| format!("failed to commit arm '{}': {e}", spec.arm_id))?;
            // A silently uncommitted arm leaves a branch that holds the base
            // code, so both arms would verify the same tree and look compared.
            if !committed.status.success() {
                return Err(format!(
                    "could not commit arm '{}' to branch '{}': {}",
                    spec.arm_id,
                    branch,
                    String::from_utf8_lossy(&committed.stderr).trim()
                ));
            }
        }

        // Run verification checklist for this arm's worktree and session
        let checklist = toolchain::run_checklist_for_session(
            &worktree_dir,
            &config.skip_checks,
            config.timeout_secs,
            Some(&session),
        )
        .map_err(|e| {
            format!(
                "verification checklist failed for arm '{}': {}",
                spec.arm_id, e
            )
        })?;

        // Ensure artifact logs are synchronized to repo_root's artifact directory so they persist
        let wt_art = worktree_dir
            .join(xencode_context_rs::XENCODE_DIR)
            .join(xencode_context_rs::artifacts::ARTIFACTS_DIR)
            .join(&session);
        let root_art = repo_root
            .join(xencode_context_rs::XENCODE_DIR)
            .join(xencode_context_rs::artifacts::ARTIFACTS_DIR)
            .join(&session);
        if wt_art.is_dir() {
            let _ = std::fs::create_dir_all(&root_art);
            if let Ok(entries) = std::fs::read_dir(&wt_art) {
                for entry in entries.flatten() {
                    let dest = root_art.join(entry.file_name());
                    let _ = std::fs::copy(entry.path(), dest);
                }
            }
        }

        let checks = checklist
            .checks
            .into_iter()
            .map(|c| {
                let failed = c.ran && !c.passed();
                ArmCheckRow {
                    name: c.name,
                    ran: c.ran,
                    skipped: !c.ran,
                    failed,
                    evidence_ref: c.evidence_ref,
                }
            })
            .collect();

        arm_results.push(ArmResult {
            arm_id: spec.arm_id.clone(),
            label: spec.label.clone(),
            branch,
            worktree_path: worktree_dir,
            session,
            checks,
        });
    }

    let report = CompetingReport {
        run_id,
        prompt: config.prompt.clone(),
        arms: arm_results,
        picked_arm: None,
        created_at_unix_ms: timestamp_ms as u64,
    };

    save_competing_run(repo_root, &report)?;
    Ok(report)
}

/// Pick an arm by human choice.
///
/// Switches the active repository to the picked arm's branch while leaving
/// all other branches and their evidence artifacts on disk.
pub fn pick_arm(repo_root: &Path, run_id: &str, arm_id: &str) -> Result<PickOutcome, String> {
    let mut report = load_competing_run(repo_root, run_id)?;
    let picked = report
        .arms
        .iter()
        .find(|a| a.arm_id == arm_id)
        .cloned()
        .ok_or_else(|| {
            let valid = report
                .arms
                .iter()
                .map(|a| a.arm_id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            format!("arm '{arm_id}' not found in run '{run_id}'. Available arms: {valid}")
        })?;

    // If the picked branch is still attached to its temporary worktree, unlink that worktree
    // so the root checkout can switch to the picked branch.
    if picked.worktree_path.exists() {
        let _ = xencode_context_rs::worktree_remove(repo_root, &picked.worktree_path, true);
    }

    // Switch/checkout to picked branch in repo_root
    let checkout = std::process::Command::new("git")
        .current_dir(repo_root)
        .args(["checkout", &picked.branch])
        .output()
        .map_err(|e| format!("git checkout failed: {e}"))?;
    if !checkout.status.success() {
        return Err(format!(
            "could not switch to branch '{}': {}",
            picked.branch,
            String::from_utf8_lossy(&checkout.stderr).trim()
        ));
    }

    // Verify other candidate branches and their evidence are preserved on disk
    let mut other_branches = Vec::new();
    let mut preserved_evidence = Vec::new();

    for other in report.arms.iter().filter(|a| a.arm_id != arm_id) {
        let check_branch = std::process::Command::new("git")
            .current_dir(repo_root)
            .args(["rev-parse", "--verify", &other.branch])
            .output()
            .map_err(|e| format!("failed verifying branch {}: {e}", other.branch))?;
        if check_branch.status.success() {
            other_branches.push(other.branch.clone());
        }

        let evidence_dir = repo_root
            .join(xencode_context_rs::XENCODE_DIR)
            .join(xencode_context_rs::artifacts::ARTIFACTS_DIR)
            .join(&other.session);
        if evidence_dir.exists() {
            preserved_evidence.push(evidence_dir);
        }
    }

    report.picked_arm = Some(arm_id.to_string());
    save_competing_run(repo_root, &report)?;

    Ok(PickOutcome {
        run_id: run_id.to_string(),
        picked_arm_id: arm_id.to_string(),
        picked_branch: picked.branch,
        other_branches,
        preserved_evidence,
    })
}

/// Persist a competing run report under `.xencode/compete/<run_id>.json`.
pub fn save_competing_run(repo_root: &Path, report: &CompetingReport) -> Result<(), String> {
    let dir = repo_root
        .join(xencode_context_rs::XENCODE_DIR)
        .join("compete");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("could not create compete directory: {e}"))?;
    let path = dir.join(format!("{}.json", report.run_id));
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| format!("could not serialize competing report: {e}"))?;
    std::fs::write(&path, json.as_bytes())
        .map_err(|e| format!("could not write competing report {}: {e}", path.display()))
}

/// Load a previously saved competing run report.
pub fn load_competing_run(repo_root: &Path, run_id: &str) -> Result<CompetingReport, String> {
    let path = repo_root
        .join(xencode_context_rs::XENCODE_DIR)
        .join("compete")
        .join(format!("{run_id}.json"));
    if !path.exists() {
        return Err(format!("competing run '{run_id}' not found"));
    }
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read competing report {}: {e}", path.display()))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("corrupt competing report in {}: {e}", path.display()))
}

/// List all competing candidate run reports ordered newest first.
pub fn list_competing_runs(repo_root: &Path) -> Result<Vec<CompetingReport>, String> {
    let dir = repo_root
        .join(xencode_context_rs::XENCODE_DIR)
        .join("compete");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut reports = Vec::new();
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("could not read compete directory: {e}"))?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(content) = std::fs::read_to_string(&p) {
                if let Ok(report) = serde_json::from_str::<CompetingReport>(&content) {
                    reports.push(report);
                }
            }
        }
    }
    reports.sort_by_key(|a| std::cmp::Reverse(a.created_at_unix_ms));
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A committed single-crate project in a temporary directory, with a git
    /// identity configured so an arm's commit can actually be made.
    fn sample_repo(tag: &str) -> (tempfile::TempDir, PathBuf) {
        let temp = tempfile::Builder::new().prefix(tag).tempdir().unwrap();
        let repo = temp.path().join("repo");
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname = \"compete-sample\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(repo.join("src/lib.rs"), "pub fn answer() -> u32 { 0 }\n").unwrap();
        for args in [
            vec!["init", "-b", "main", "."],
            vec!["config", "user.name", "Compete Test"],
            vec!["config", "user.email", "compete-test@example.invalid"],
            vec!["add", "-A"],
            vec!["commit", "-m", "initial commit"],
        ] {
            let out = std::process::Command::new("git")
                .args(&args)
                .current_dir(&repo)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        (temp, repo)
    }

    fn arm(arm_id: &str) -> CandidateArmSpec {
        CandidateArmSpec {
            arm_id: arm_id.to_string(),
            label: arm_id.to_string(),
            branch: None,
            command: None,
            file_edits: Vec::new(),
        }
    }

    /// Every check skipped, so nothing but the arm's own build steps can fail.
    fn nothing_verified(config: CompetingConfig) -> CompetingConfig {
        CompetingConfig {
            skip_checks: vec!["fmt".into(), "lint".into(), "test".into()],
            ..config
        }
    }

    /// A candidate whose build command exits non-zero never reaches the
    /// checklist: verification rows would describe a tree nobody built.
    #[test]
    fn a_failing_candidate_command_stops_the_run_and_records_nothing() {
        let (_temp, repo) = sample_repo("compete-cmd-failed");
        let broken = CandidateArmSpec {
            command: Some("echo generating >&2; exit 3".to_string()),
            ..arm("broken")
        };
        let config = nothing_verified(CompetingConfig {
            prompt: "Which parser?".to_string(),
            arm_specs: vec![broken, arm("sound")],
            skip_checks: Vec::new(),
            timeout_secs: 30,
        });

        let err = run_competing_arms(&repo, &config).unwrap_err();
        assert!(err.contains("candidate command for arm 'broken'"), "{err}");
        assert!(
            err.contains("exited with 3"),
            "the exit code must be reported: {err}"
        );
        assert!(
            err.contains("generating"),
            "the command's own error must be carried: {err}"
        );
        assert!(
            list_competing_runs(&repo).unwrap().is_empty(),
            "a run that never finished must not be recorded as if it had"
        );
    }

    /// If the arm's commit is refused, its branch keeps the base code, and the
    /// two arms would verify identical trees while looking compared.
    #[test]
    fn an_arm_that_cannot_be_committed_is_reported_rather_than_verified_empty() {
        use std::os::unix::fs::PermissionsExt;
        let (_temp, repo) = sample_repo("compete-commit-refused");
        let hook = repo.join(".git/hooks/pre-commit");
        std::fs::write(&hook, "#!/bin/sh\nexit 1\n").unwrap();
        std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();

        let edited = |arm_id: &str| CandidateArmSpec {
            file_edits: vec![(
                PathBuf::from("src/lib.rs"),
                format!("pub fn answer() -> u32 {{ {} }}\n", arm_id.len()),
            )],
            ..arm(arm_id)
        };
        let config = nothing_verified(CompetingConfig {
            prompt: "Const or static?".to_string(),
            arm_specs: vec![edited("first"), edited("second")],
            skip_checks: Vec::new(),
            timeout_secs: 30,
        });

        let err = run_competing_arms(&repo, &config).unwrap_err();
        assert!(err.contains("could not commit arm 'first'"), "{err}");
        assert!(list_competing_runs(&repo).unwrap().is_empty(), "{err}");
    }

    #[test]
    fn format_competing_table_renders_checks_without_composite_score() {
        let report = CompetingReport {
            run_id: "compete-123".to_string(),
            prompt: "Choose token-bucket vs leaky-bucket algorithm".to_string(),
            arms: vec![
                ArmResult {
                    arm_id: "arm-a".to_string(),
                    label: "Token Bucket".to_string(),
                    branch: "compete/123/arm-a".to_string(),
                    worktree_path: PathBuf::from("/tmp/wt-a"),
                    session: "123-arm-a".to_string(),
                    checks: vec![
                        ArmCheckRow {
                            name: "fmt".to_string(),
                            ran: true,
                            skipped: false,
                            failed: false,
                            evidence_ref: ".xencode/artifacts/123-arm-a/fmt.log".to_string(),
                        },
                        ArmCheckRow {
                            name: "test".to_string(),
                            ran: true,
                            skipped: false,
                            failed: false,
                            evidence_ref: ".xencode/artifacts/123-arm-a/test.log".to_string(),
                        },
                    ],
                },
                ArmResult {
                    arm_id: "arm-b".to_string(),
                    label: "Leaky Bucket".to_string(),
                    branch: "compete/123/arm-b".to_string(),
                    worktree_path: PathBuf::from("/tmp/wt-b"),
                    session: "123-arm-b".to_string(),
                    checks: vec![
                        ArmCheckRow {
                            name: "fmt".to_string(),
                            ran: true,
                            skipped: false,
                            failed: false,
                            evidence_ref: ".xencode/artifacts/123-arm-b/fmt.log".to_string(),
                        },
                        ArmCheckRow {
                            name: "test".to_string(),
                            ran: true,
                            skipped: false,
                            failed: true,
                            evidence_ref: ".xencode/artifacts/123-arm-b/test.log".to_string(),
                        },
                    ],
                },
            ],
            picked_arm: None,
            created_at_unix_ms: 1000,
        };

        let rendered = format_competing_table(&report);

        // Required headers
        assert!(rendered.contains("check   ran     skipped failed  evidence-ref"));
        assert!(rendered.contains("Arm: arm-a [branch: compete/123/arm-a]"));
        assert!(rendered.contains("Arm: arm-b [branch: compete/123/arm-b]"));
        assert!(rendered.contains("Strategy: Token Bucket"));
        assert!(rendered.contains("Strategy: Leaky Bucket"));

        // Evidence and check status
        assert!(rendered.contains(".xencode/artifacts/123-arm-a/test.log"));
        assert!(rendered.contains(".xencode/artifacts/123-arm-b/test.log"));

        // Explicitly assert NO composite score or percentage
        let lower = rendered.to_lowercase();
        assert!(
            !lower.contains("score"),
            "table must have NO composite score"
        );
        assert!(!lower.contains("%"), "table must have no percentages");
        assert!(
            !lower.contains("winner"),
            "table must have no winner declaration"
        );
        assert!(!lower.contains("grade"), "table must have no grade");
    }

    /// `compete list` promises newest first, so the ordering is what the
    /// function under test produces, not what the caller happens to sort.
    #[test]
    fn saved_runs_list_newest_first_and_load_back_by_id() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        let report = |ts: u64| CompetingReport {
            run_id: format!("compete-{ts}"),
            prompt: format!("prompt at {ts}"),
            arms: Vec::new(),
            picked_arm: None,
            created_at_unix_ms: ts,
        };
        for ts in [1000_u64, 3000, 2000] {
            save_competing_run(repo, &report(ts)).unwrap();
        }

        let listed = list_competing_runs(repo).unwrap();
        assert_eq!(
            listed
                .iter()
                .map(|r| r.created_at_unix_ms)
                .collect::<Vec<_>>(),
            vec![3000, 2000, 1000],
            "the newest run must be first"
        );

        let loaded = load_competing_run(repo, "compete-2000").unwrap();
        assert_eq!(loaded.prompt, "prompt at 2000");

        let missing = load_competing_run(repo, "compete-1999999").unwrap_err();
        assert!(missing.contains("not found"), "{missing}");

        // A repository with no runs yet lists empty rather than failing.
        let fresh = tempfile::tempdir().unwrap();
        assert!(list_competing_runs(fresh.path()).unwrap().is_empty());
    }

    #[test]
    fn arm_count_validation_enforces_two_or_three_arms() {
        let repo = PathBuf::from("/tmp/does-not-matter");
        let single_arm_cfg = CompetingConfig {
            prompt: "test".to_string(),
            arm_specs: vec![CandidateArmSpec {
                arm_id: "arm-1".to_string(),
                label: "one".to_string(),
                branch: None,
                command: None,
                file_edits: vec![],
            }],
            skip_checks: vec![],
            timeout_secs: 10,
        };
        let err = run_competing_arms(&repo, &single_arm_cfg).unwrap_err();
        assert!(err.contains("must have 2 or 3 candidate implementations"));

        let four_arm_cfg = CompetingConfig {
            prompt: "test".to_string(),
            arm_specs: (1..=4)
                .map(|i| CandidateArmSpec {
                    arm_id: format!("arm-{i}"),
                    label: format!("label {i}"),
                    branch: None,
                    command: None,
                    file_edits: vec![],
                })
                .collect(),
            skip_checks: vec![],
            timeout_secs: 10,
        };
        let err4 = run_competing_arms(&repo, &four_arm_cfg).unwrap_err();
        assert!(err4.contains("must have 2 or 3 candidate implementations"));
    }

    #[test]
    fn run_competing_arms_two_branches_and_pick_leaves_other_branch_and_evidence() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        std::fs::create_dir_all(&repo).unwrap();

        let run = |cmd: &[&str], dir: &Path| {
            let out = std::process::Command::new(cmd[0])
                .args(&cmd[1..])
                .current_dir(dir)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "failed {:?}: {}",
                cmd,
                String::from_utf8_lossy(&out.stderr)
            );
        };

        run(&["git", "init", "-b", "main"], &repo);
        run(&["git", "config", "user.name", "Test User"], &repo);
        run(&["git", "config", "user.email", "test@example.com"], &repo);

        std::fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname = \"compete-sample\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(repo.join("src/lib.rs"), "pub fn answer() -> u32 { 0 }\n").unwrap();

        run(&["git", "add", "."], &repo);
        run(&["git", "commit", "-m", "initial commit"], &repo);

        let config = CompetingConfig {
            prompt: "Return 42 via iterative vs recursive implementation".to_string(),
            arm_specs: vec![
                CandidateArmSpec {
                    arm_id: "iterative".to_string(),
                    label: "Iterative approach".to_string(),
                    branch: None,
                    command: None,
                    file_edits: vec![(
                        PathBuf::from("src/lib.rs"),
                        "pub fn answer() -> u32 { 42 }\n#[test]\nfn test_ans() { assert_eq!(answer(), 42); }\n".to_string(),
                    )],
                },
                CandidateArmSpec {
                    arm_id: "recursive".to_string(),
                    label: "Recursive approach".to_string(),
                    branch: None,
                    command: None,
                    file_edits: vec![(
                        PathBuf::from("src/lib.rs"),
                        "pub fn answer() -> u32 { 42 }\n#[test]\nfn test_ans() { assert_eq!(answer(), 42); }\n".to_string(),
                    )],
                },
            ],
            skip_checks: vec!["fmt".to_string(), "lint".to_string()],
            timeout_secs: 60,
        };

        let report = run_competing_arms(&repo, &config).unwrap();
        assert_eq!(report.arms.len(), 2);
        assert_eq!(report.arms[0].arm_id, "iterative");
        assert_eq!(report.arms[1].arm_id, "recursive");

        let branch_1 = &report.arms[0].branch;
        let branch_2 = &report.arms[1].branch;

        let check_b1 = std::process::Command::new("git")
            .current_dir(&repo)
            .args(["rev-parse", "--verify", branch_1])
            .output()
            .unwrap();
        assert!(check_b1.status.success(), "branch 1 must exist");

        let check_b2 = std::process::Command::new("git")
            .current_dir(&repo)
            .args(["rev-parse", "--verify", branch_2])
            .output()
            .unwrap();
        assert!(check_b2.status.success(), "branch 2 must exist");

        assert!(!report.arms[0].checks.is_empty());
        assert!(!report.arms[1].checks.is_empty());

        let table = format_competing_table(&report);
        assert!(table.contains("check   ran     skipped failed  evidence-ref"));
        assert!(table.contains(branch_1));
        assert!(table.contains(branch_2));
        assert!(!table.to_lowercase().contains("score"));

        let outcome = pick_arm(&repo, &report.run_id, "iterative").unwrap();
        assert_eq!(outcome.picked_arm_id, "iterative");
        assert_eq!(outcome.picked_branch, *branch_1);

        let check_b2_after = std::process::Command::new("git")
            .current_dir(&repo)
            .args(["rev-parse", "--verify", branch_2])
            .output()
            .unwrap();
        assert!(
            check_b2_after.status.success(),
            "unpicked branch 2 must remain on disk"
        );

        assert!(outcome.other_branches.contains(branch_2));
        assert!(!outcome.preserved_evidence.is_empty());
        for ev in &outcome.preserved_evidence {
            assert!(
                ev.exists(),
                "preserved evidence path must exist on disk: {}",
                ev.display()
            );
        }
    }
}
