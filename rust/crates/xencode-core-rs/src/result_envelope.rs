//! The result envelope (`OR-16`).
//!
//! Every finished task produces one machine-readable record instead of a paragraph.
//! The whole trust problem of a multi-agent workflow is in one sentence: a reviewer
//! reading an implementer's *"yeah, authentication is done"* and taking it at face
//! value. So this record keeps **what the worker claims** and **what actually
//! happened** in different fields, and only the second half is quotable to a human.
//!
//! It is `EVd-3`'s checks-ran verdict extended, not a rival ledger. The commands
//! carry the same shape `EVd-3` settled on — a command that ran and an exit code,
//! plus a pointer to where its output lives — with JUnit's `skipped ≠ passed` rule
//! preserved (nothing that did not run can pass). On top of that evidence the
//! envelope adds the fields a task handoff needs and the claims it must not be
//! mistaken for: the agent, the task, the files the diff actually shows changed,
//! a handoff state, and — held apart — the worker's own assertions.
//!
//! A reviewing agent is handed [`ReviewerView`], which carries the evidence and a
//! clearly-labelled list of unverified claims. It is never handed the implementer's
//! prose as if it were fact.

use serde::{Deserialize, Serialize};

/// How the task ended. A worker choosing the word "completed" is only a claim
/// until the evidence half agrees — this field is where the *decided* status
/// lives, which the launch path sets from the contract and checks, not from the
/// worker's report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishStatus {
    /// The contract passed and every check that ran exited zero.
    Completed,
    /// It ran and something failed — a check, the build, the contract.
    Failed,
    /// It could not be judged complete: verification never ran, or a veto blocks
    /// it. Distinct from `Failed` so "unproven" is never read as "broken", and
    /// never as "done".
    Blocked,
}

impl FinishStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Blocked => "blocked",
        }
    }
}

/// A command that ran during the task, with its real exit code and where its
/// output was kept. This is `EVd-3`'s slot: `ran` is implicit (a command here
/// ran — one that was skipped has no entry), the exit code is the verdict, and
/// `evidence_ref` points at the artifact rather than restating it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RanCommand {
    pub command: String,
    pub exit_code: i32,
    pub evidence_ref: String,
}

impl RanCommand {
    /// Passed means it ran and exited zero — nothing else, the same rule as
    /// `EVd-3`. An exit code of nonzero is a fail however the worker frames it.
    pub fn passed(&self) -> bool {
        self.exit_code == 0
    }
}

/// What the worker asserted about its own work. Stored, but never promoted to
/// fact: a claim is the thing the evidence is checked *against*, not a substitute
/// for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub text: String,
}

/// The evidence half: only things xencode observed itself — a diff, an exit code,
/// a saved artifact. This is the field a human or reviewer may quote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evidence {
    /// Files the diff shows changed, taken from git rather than from the worker.
    pub changed_files: Vec<String>,
    /// The commands that ran, each with its real exit code.
    pub commands: Vec<RanCommand>,
    /// Pointers to saved artifacts (logs, reports) behind the commands.
    pub artifact_refs: Vec<String>,
}

/// Where the task passes next, so a fleet can chain without a human reading
/// prose to find out. `NeedsReview` and `Blocked` are both states that mean "do
/// not land yet", and differ in whether a reviewer or an author is expected to
/// move it. What actually refuses a merge today is the veto these names point
/// at (`OR-17`, `xencode-analysis-rs/src/veto.rs`); nothing consumes the
/// envelope itself yet, which is what `AE-1` is for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum Handoff {
    /// Nothing more to do on this task; downstream work may proceed.
    Done,
    /// A human or a reviewer must look before it lands.
    NeedsReview { reason: String },
    /// It cannot proceed — a veto, a missing dependency, an unbuilt check.
    Blocked { reason: String },
}

/// The record one finished task produces. Claims and evidence are separate
/// fields by construction; there is no way to read the envelope's "what
/// happened" without going through [`ResultEnvelope::evidence`], which never
/// consults the claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResultEnvelope {
    pub status: FinishStatus,
    pub agent: String,
    pub task: String,
    pub evidence: Evidence,
    /// Held apart from evidence on purpose. The reviewer sees these as
    /// assertions, not facts.
    pub claims: Vec<Claim>,
    pub handoff: Handoff,
}

/// What a reviewing agent is handed: the evidence, the decided status, and the
/// worker's claims only under an explicit "unverified" label. Constructing this
/// from an envelope is the one supported way to forward a result onward, which
/// is what keeps prose from sneaking back in as fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewerView<'a> {
    pub status: FinishStatus,
    pub agent: &'a str,
    pub task: &'a str,
    pub evidence: &'a Evidence,
    /// The same claims, but never folded into `evidence` — a reviewer must be
    /// able to tell which is which.
    pub unverified_claims: &'a [Claim],
}

/// A claim phrase that must never be presented as fact. Used to prove the
/// separation actually holds in the quotable view.
impl ResultEnvelope {
    /// The evidence rendered for a human. Built only from [`Evidence`] — a
    /// worker's own claim text can never appear here, so quoting this cannot
    /// launder an assertion into fact.
    pub fn evidence_quotable(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "changed files ({}):\n",
            self.evidence.changed_files.len()
        ));
        for f in &self.evidence.changed_files {
            out.push_str(&format!("  {f}\n"));
        }
        out.push_str(&format!(
            "commands run ({}):\n",
            self.evidence.commands.len()
        ));
        for c in &self.evidence.commands {
            out.push_str(&format!(
                "  {} -> exit {} ({})\n",
                c.command, c.exit_code, c.evidence_ref
            ));
        }
        if !self.evidence.artifact_refs.is_empty() {
            out.push_str("artifacts:\n");
            for a in &self.evidence.artifact_refs {
                out.push_str(&format!("  {a}\n"));
            }
        }
        out
    }

    /// Every command that ran exited zero AND at least one ran. Mirrors
    /// `EVd-3`: an empty command list is *not* a pass — nothing proven is nothing
    /// passed — and a skipped check has no entry, so it can never inflate this.
    pub fn all_checks_passed(&self) -> bool {
        !self.evidence.commands.is_empty() && self.evidence.commands.iter().all(RanCommand::passed)
    }

    /// The record handed to a reviewing agent: evidence and status, with claims
    /// kept as a separately-labelled list.
    pub fn for_reviewer(&self) -> ReviewerView<'_> {
        ReviewerView {
            status: self.status,
            agent: &self.agent,
            task: &self.task,
            evidence: &self.evidence,
            unverified_claims: &self.claims,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(
        status: FinishStatus,
        commands: Vec<RanCommand>,
        claims: Vec<&str>,
    ) -> ResultEnvelope {
        ResultEnvelope {
            status,
            agent: "claude".into(),
            task: "split-retrieval".into(),
            evidence: Evidence {
                changed_files: vec!["src/auth.rs".into()],
                commands,
                artifact_refs: vec!["logs/test.log".into()],
            },
            claims: claims
                .into_iter()
                .map(|t| Claim { text: t.into() })
                .collect(),
            handoff: Handoff::Done,
        }
    }

    /// Claims and evidence sit in different fields: a claim never reaches the
    /// quotable-to-a-human projection, so "authentication is done" cannot be
    /// read back as though the run proved it.
    #[test]
    fn a_claim_never_appears_in_the_quotable_evidence() {
        let e = envelope(
            FinishStatus::Completed,
            vec![RanCommand {
                command: "cargo test".into(),
                exit_code: 0,
                evidence_ref: "logs/test.log".into(),
            }],
            vec!["yeah, authentication is done"],
        );
        let quotable = e.evidence_quotable();
        assert!(
            quotable.contains("cargo test"),
            "the command that ran is quotable"
        );
        assert!(
            quotable.contains("src/auth.rs"),
            "the changed file is quotable"
        );
        assert!(
            !quotable.to_lowercase().contains("authentication is done"),
            "a worker claim leaked into the evidence a human is shown: {quotable}"
        );
    }

    /// A reviewing agent is handed the record, with claims present but fenced
    /// off as unverified — never merged into the evidence.
    #[test]
    fn the_reviewer_is_handed_the_record_with_claims_fenced_off() {
        let e = envelope(FinishStatus::Blocked, Vec::new(), vec!["all tests green"]);
        let view = e.for_reviewer();
        assert_eq!(view.status, FinishStatus::Blocked);
        assert!(
            view.evidence.commands.is_empty(),
            "reviewer sees that nothing ran"
        );
        assert_eq!(
            view.unverified_claims.len(),
            1,
            "the claim is still visible…"
        );
        // …but on a field distinct from `evidence`, so it reads as an assertion
        // rather than as something the run proved.
        assert_eq!(view.evidence.artifact_refs.len(), 1);
        assert!(view
            .evidence
            .changed_files
            .contains(&"src/auth.rs".to_string()));
    }

    /// `EVd-3` semantics survive: passing needs a command that ran and exited
    /// zero. No commands (nothing proven) is not a pass, and a nonzero exit is a
    /// fail however the worker words it.
    #[test]
    fn nothing_proven_is_not_passed_and_a_nonzero_exit_is_failed() {
        let no_checks = envelope(FinishStatus::Completed, Vec::new(), vec!["trust me"]);
        assert!(
            !no_checks.all_checks_passed(),
            "an empty command list proves nothing"
        );

        let green = envelope(
            FinishStatus::Completed,
            vec![RanCommand {
                command: "cargo test".into(),
                exit_code: 0,
                evidence_ref: "l".into(),
            }],
            vec![],
        );
        assert!(green.all_checks_passed());

        let red = envelope(
            FinishStatus::Failed,
            vec![
                RanCommand {
                    command: "cargo test".into(),
                    exit_code: 0,
                    evidence_ref: "l".into(),
                },
                RanCommand {
                    command: "cargo clippy".into(),
                    exit_code: 101,
                    evidence_ref: "l".into(),
                },
            ],
            vec!["clippy is fine"],
        );
        assert!(
            !red.all_checks_passed(),
            "one nonzero exit fails the whole thing"
        );
    }

    /// The record is real, not assembled from strings: commands are executed for
    /// their genuine exit codes and changed files come from a genuine `git diff`,
    /// then the envelope must reflect both — and still keep the claim out.
    #[test]
    fn a_real_command_run_and_a_real_git_diff_fill_the_envelope() {
        let dir = std::env::temp_dir().join(format!("xencode-envelope-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        git(&dir, &["init", "-q"]);
        git(&dir, &["config", "user.email", "t@example.com"]);
        git(&dir, &["config", "user.name", "test"]);
        std::fs::write(dir.join("src/a.rs"), b"v1").unwrap();
        std::fs::write(dir.join("src/b.rs"), b"v1").unwrap();
        git(&dir, &["add", "-A"]);
        git(&dir, &["commit", "-qm", "seed"]);
        std::fs::write(dir.join("src/a.rs"), b"v2").unwrap(); // only a.rs changes

        // A command that really runs and really exits 0; one that really exits 7.
        let pass = std::process::Command::new("sh")
            .args(["-c", "true"])
            .output()
            .unwrap();
        let fail = std::process::Command::new("sh")
            .args(["-c", "exit 7"])
            .output()
            .unwrap();
        let changed: Vec<String> =
            String::from_utf8_lossy(&git_bytes(&dir, &["diff", "--name-only"]))
                .lines()
                .map(|l| l.to_string())
                .collect();

        let e = ResultEnvelope {
            status: FinishStatus::Failed,
            agent: "claude".into(),
            task: "real".into(),
            evidence: Evidence {
                changed_files: changed.clone(),
                commands: vec![
                    RanCommand {
                        command: "true".into(),
                        exit_code: pass.status.code().unwrap(),
                        evidence_ref: "logs/true.log".into(),
                    },
                    RanCommand {
                        command: "exit 7".into(),
                        exit_code: fail.status.code().unwrap(),
                        evidence_ref: "logs/seven.log".into(),
                    },
                ],
                artifact_refs: vec![],
            },
            claims: vec![Claim {
                text: "everything passed".into(),
            }],
            handoff: Handoff::Blocked {
                reason: "a real command exited 7".into(),
            },
        };

        assert_eq!(
            changed,
            vec!["src/a.rs".to_string()],
            "git reports only the file that changed"
        );
        assert_eq!(
            e.evidence.commands[0].exit_code, 0,
            "the real exit code of `true`"
        );
        assert_eq!(
            e.evidence.commands[1].exit_code, 7,
            "the real exit code of `exit 7`"
        );
        assert!(
            !e.all_checks_passed(),
            "a genuinely nonzero exit fails the envelope"
        );
        assert!(
            e.evidence_quotable().contains("exit 7"),
            "the real code is surfaced to a human"
        );
        assert!(
            !e.evidence_quotable().contains("everything passed"),
            "the claim stays out"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn git(dir: &std::path::Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .expect("git is on PATH");
        assert!(status.success(), "git {args:?} failed");
    }

    fn git_bytes(dir: &std::path::Path, args: &[&str]) -> Vec<u8> {
        let output = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git is on PATH");
        assert!(output.status.success(), "git {args:?} failed");
        output.stdout
    }
}
