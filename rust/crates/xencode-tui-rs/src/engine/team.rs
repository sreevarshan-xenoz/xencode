//! The engine's team of worker agents (TM-1): it starts workers in their own
//! worktrees, keeps what each has said, answers the team requests windows and
//! the lead send, and turns a worker's permission prompt into an ordinary
//! engine approval, so the person answers it in any window.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};
use xencode_team_rs::merge::{self, Checks, MergeOutcome};
use xencode_team_rs::worker::{LaunchSpec, WorkerEvent, WorkerHandle};
use xencode_team_rs::{worktree, MergeState, WorkerSnapshot, WorkerState};

use crate::agent_tools::{ApprovalAnswer, ApprovalDraft, ApprovalRequest, ToolClass};
use crate::engine::proto::TeamRequest;

/// Workers running at once, unless `.xencode/team.toml` says otherwise.
pub const DEFAULT_LIMIT: usize = 4;

/// The team an engine runs.
pub struct Team {
    workers: BTreeMap<String, WorkerHandle>,
    shown: BTreeMap<String, WorkerSnapshot>,
    events_tx: mpsc::UnboundedSender<WorkerEvent>,
    events: mpsc::UnboundedReceiver<WorkerEvent>,
    next: u32,
    limit: usize,
    /// The branch each worker started from.
    bases: BTreeMap<String, String>,
    /// Each worker's merge, kept apart from what the worker reports.
    merges: BTreeMap<String, MergeState>,
}

/// The approvals channel: a request and where its answer goes.
pub type Approvals = mpsc::UnboundedSender<(ApprovalRequest, oneshot::Sender<ApprovalAnswer>)>;

impl Default for Team {
    fn default() -> Self {
        let (events_tx, events) = mpsc::unbounded_channel();
        Team {
            workers: BTreeMap::new(),
            shown: BTreeMap::new(),
            events_tx,
            events,
            next: 0,
            limit: DEFAULT_LIMIT,
            bases: BTreeMap::new(),
            merges: BTreeMap::new(),
        }
    }
}

impl Team {
    /// Every worker as last seen, in id order.
    pub fn snapshots(&self) -> Vec<WorkerSnapshot> {
        self.shown
            .values()
            .cloned()
            .map(|s| self.with_merge(s))
            .collect()
    }

    fn with_merge(&self, mut s: WorkerSnapshot) -> WorkerSnapshot {
        s.merge = self.merges.get(&s.id).cloned();
        s
    }

    /// Take in what the workers reported. A permission prompt is raised on
    /// `approvals`, the channel the agent loop's prompts use, so it gets an
    /// id, reaches every window and is answered once.
    pub fn pump(
        &mut self,
        approvals: &mpsc::UnboundedSender<(ApprovalRequest, oneshot::Sender<ApprovalAnswer>)>,
    ) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                WorkerEvent::Changed(snapshot) => {
                    self.shown.insert(snapshot.id.clone(), snapshot);
                }
                WorkerEvent::Merged { worker, outcome } => {
                    self.merges.insert(worker, MergeState::Finished(outcome));
                }
                WorkerEvent::Permission {
                    worker,
                    summary,
                    tool,
                    answer,
                } => {
                    let request = ApprovalRequest {
                        tool: format!("worker {worker}: {tool}"),
                        // Another agent's tool: xencode can neither preview,
                        // checkpoint nor undo it.
                        class: ToolClass::External,
                        summary: format!("{worker} asks: {summary}"),
                        preview: String::new(),
                        draft: ApprovalDraft::default(),
                    };
                    let (tx, rx) = oneshot::channel();
                    if approvals.send((request, tx)).is_err() {
                        let _ = answer.send(false);
                        continue;
                    }
                    tokio::spawn(async move {
                        let yes = matches!(
                            rx.await,
                            Ok(ApprovalAnswer::Approved | ApprovalAnswer::ApprovedForSession)
                        );
                        let _ = answer.send(yes);
                    });
                }
            }
        }
    }

    /// Carry out one team request for the project at `root`.
    pub fn request(
        &mut self,
        root: &Path,
        request: TeamRequest,
        approvals: &Approvals,
    ) -> Result<Value, String> {
        match request {
            TeamRequest::Agents => Ok(json!([{ "name": "xencode", "available": true }])),
            TeamRequest::Start { agent, task, base } => self.start(root, &agent, &task, base),
            TeamRequest::Status { id: Some(id) } => {
                let s = self.find(&id)?;
                serde_json::to_value(s).map_err(|e| e.to_string())
            }
            TeamRequest::Status { id: None } => {
                serde_json::to_value(self.snapshots()).map_err(|e| e.to_string())
            }
            TeamRequest::Result { id } => {
                let s = self.find(&id)?;
                let diff = worktree::diff(&s.worktree, &base_of(root))?;
                Ok(json!({ "id": id, "state": s.state, "answer": s.answer, "diff": diff }))
            }
            TeamRequest::Message { id, text } => {
                self.handle(&id)?.message(&text)?;
                Ok(json!({ "id": id, "sent": true }))
            }
            TeamRequest::Stop { id } => {
                self.handle(&id)?.stop();
                Ok(json!({ "id": id, "stopping": true }))
            }
            TeamRequest::Merge { id } => self.merge(root, &id, approvals),
        }
    }

    fn find(&self, id: &str) -> Result<WorkerSnapshot, String> {
        match self.workers.get(id) {
            Some(handle) => Ok(self.with_merge(handle.snapshot())),
            None => Err(format!("no worker {id}")),
        }
    }

    /// Start the checked merge of worker `id`; its outcome arrives as the
    /// worker's `merge`.
    fn merge(&mut self, root: &Path, id: &str, approvals: &Approvals) -> Result<Value, String> {
        let s = self.find(id)?;
        if !matches!(s.state, WorkerState::Done | WorkerState::Stopped) {
            return Err(format!(
                "{id} is {:?}; it can be merged once its turn is done or it is stopped",
                s.state
            ));
        }
        if matches!(s.merge, Some(MergeState::Running)) {
            return Err(format!("{id} is being merged already"));
        }
        if matches!(
            s.merge,
            Some(MergeState::Finished(MergeOutcome::Landed { .. }))
        ) {
            return Err(format!("{id} has landed already"));
        }
        let base = self.bases.get(id).cloned().unwrap_or_else(|| base_of(root));
        self.merges.insert(id.to_string(), MergeState::Running);
        let settings = merge::team_settings(root);
        let timeout = std::time::Duration::from_secs(settings.check_timeout_secs.unwrap_or(1200));
        let stop = self.handle(id)?.stopper();
        let (root, id_owned, approvals, events) = (
            root.to_path_buf(),
            id.to_string(),
            approvals.clone(),
            self.events_tx.clone(),
        );
        tokio::spawn(async move {
            let outcome = merge_worker(&root, &id_owned, &s, &base, timeout, &approvals).await;
            if matches!(outcome, MergeOutcome::Landed { .. }) {
                // Its work is in; the worker ends and its worktree goes. On
                // Windows a folder whose files a process holds open cannot be
                // removed, and the worker's own engine keeps the worktree's
                // files until it exits, idle, after the worker ends; so the
                // removal is tried for longer than that.
                stop();
                for _ in 0..(2 * crate::engine::server::IDLE_EXIT.as_secs() as usize * 4 + 40) {
                    if worktree::remove(&root, &s.worktree, &s.branch).is_ok()
                        || !s.worktree.exists()
                    {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                }
            }
            let _ = events.send(WorkerEvent::Merged {
                worker: id_owned,
                outcome,
            });
        });
        Ok(json!({ "id": id, "merging": true }))
    }

    fn handle(&self, id: &str) -> Result<&WorkerHandle, String> {
        self.workers
            .get(id)
            .ok_or_else(|| format!("no worker {id}"))
    }

    fn start(
        &mut self,
        root: &Path,
        agent: &str,
        task: &str,
        base: Option<String>,
    ) -> Result<Value, String> {
        let spec = launch_spec(agent)?;
        let running = self
            .workers
            .values()
            .filter(|h| {
                !matches!(
                    h.snapshot().state,
                    xencode_team_rs::WorkerState::Done
                        | xencode_team_rs::WorkerState::Failed
                        | xencode_team_rs::WorkerState::Stopped
                )
            })
            .count();
        let limit = merge::team_settings(root).limit.unwrap_or(self.limit);
        if running >= limit {
            return Err(format!(
                "{running} workers are already running, the limit is {limit}; stop one or raise `limit` in .xencode/team.toml"
            ));
        }
        self.next += 1;
        let id = format!("w{}", self.next);
        let base = base.unwrap_or_else(|| base_of(root));
        let (path, branch) = worktree::create(root, &id, &base)?;
        self.bases.insert(id.clone(), base.clone());
        let handle = WorkerHandle::start(
            id.clone(),
            agent,
            task,
            spec,
            path.clone(),
            branch.clone(),
            self.events_tx.clone(),
        );
        self.shown.insert(id.clone(), handle.snapshot());
        self.workers.insert(id.clone(), handle);
        Ok(json!({ "id": id, "branch": branch, "worktree": path }))
    }
}

/// The whole merge of one worker: commit its work, merge it with the checks,
/// ask the person when there are none, and remove its worktree once landed.
async fn merge_worker(
    root: &Path,
    id: &str,
    s: &WorkerSnapshot,
    base: &str,
    timeout: std::time::Duration,
    approvals: &Approvals,
) -> MergeOutcome {
    // `asked` is the commit the person was asked about: once they answer,
    // exactly that commit is merged, or nothing if the work moved since.
    let run = |checks: Checks, asked: Option<String>| {
        let (root, wt, base, task, id) = (
            root.to_path_buf(),
            s.worktree.clone(),
            base.to_string(),
            s.task.clone(),
            id.to_string(),
        );
        tokio::task::spawn_blocking(move || {
            let message = task
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("work by a worker agent")
                .to_string();
            let committed = match merge::commit_work(&wt, &message) {
                Ok(committed) => committed,
                Err(why) => return (MergeOutcome::Refused { why }, None),
            };
            let tip = match worktree::commit_of(&wt, "HEAD") {
                Ok(tip) => tip,
                Err(why) => return (MergeOutcome::Refused { why }, None),
            };
            match asked {
                None => {
                    let outcome = merge::checked_merge(&root, &base, &tip, &checks, timeout);
                    (outcome, Some(tip))
                }
                Some(asked) if committed || tip != asked => (
                    MergeOutcome::Refused {
                        why: format!(
                            "{id}'s work changed after you were asked; merge it again to be                              asked about what it is now"
                        ),
                    },
                    None,
                ),
                Some(asked) => (
                    merge::checked_merge_person_allowed(&root, &base, &asked, &checks, timeout),
                    None,
                ),
            }
        })
    };
    let refused = |e: tokio::task::JoinError| (MergeOutcome::Refused { why: e.to_string() }, None);
    let checks = merge::checks_for(root);
    let unchecked = matches!(checks, Checks::None);
    let (first, tip) = run(checks.clone(), None).await.unwrap_or_else(refused);
    match (first, tip) {
        (MergeOutcome::NeedsPerson { why }, Some(tip)) => {
            let short: String = tip.chars().take(10).collect();
            let request = ApprovalRequest {
                tool: format!("merge {id}"),
                class: ToolClass::External,
                summary: if unchecked {
                    format!("land {id} (commit {short}) on {base} without checks? ({why})")
                } else {
                    format!("land {id} (commit {short}) on {base} once its checks pass? ({why})")
                },
                preview: String::new(),
                draft: ApprovalDraft::default(),
            };
            let (tx, rx) = oneshot::channel();
            if approvals.send((request, tx)).is_err() {
                return MergeOutcome::NeedsPerson { why };
            }
            match rx.await {
                Ok(ApprovalAnswer::Approved | ApprovalAnswer::ApprovedForSession) => {
                    // With no checks the person's yes is the check; otherwise
                    // the checks still run.
                    let checks = if unchecked {
                        Checks::Commands(Vec::new())
                    } else {
                        checks
                    };
                    run(checks, Some(tip)).await.unwrap_or_else(refused).0
                }
                _ => MergeOutcome::Refused {
                    why: format!("the person chose not to land {id}"),
                },
            }
        }
        (other, _) => other,
    }
}

/// The branch the project has checked out, which workers start from.
fn base_of(root: &Path) -> String {
    std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "HEAD".to_string())
}

/// How to start `agent`. TM-1 knows xencode itself; the outside agents come
/// with sign-in in TM-5.
fn launch_spec(agent: &str) -> Result<LaunchSpec, String> {
    match agent {
        "xencode" => {
            let exe: PathBuf = std::env::current_exe()
                .map_err(|e| format!("cannot find the xencode program: {e}"))?;
            Ok(LaunchSpec {
                program: exe,
                args: vec!["acp".to_string()],
                env: Vec::new(),
            })
        }
        other => Err(format!(
            "xencode cannot start an agent called `{other}`; it knows: xencode"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Security review: the person says yes to the work they were asked about;
    /// if the worker's work changed while the question waited, that is not
    /// what lands.
    #[tokio::test]
    async fn work_that_changed_after_the_person_was_asked_does_not_land() {
        let git = |dir: &Path, args: &[&str]| {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(args)
                .output()
                .unwrap();
            assert!(out.status.success(), "{args:?}");
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("proj");
        std::fs::create_dir(&root).unwrap();
        git(&root, &["init", "-q", "-b", "main"]);
        git(&root, &["config", "user.email", "t@example.invalid"]);
        git(&root, &["config", "user.name", "t"]);
        git(&root, &["config", "core.autocrlf", "false"]);
        std::fs::write(root.join("a.txt"), "one\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "first"]);
        let (wt, branch) = worktree::create(&root, "w1", "main").unwrap();
        std::fs::write(wt.join("a.txt"), "asked about\n").unwrap();
        let s = WorkerSnapshot {
            id: "w1".into(),
            agent: "xencode".into(),
            task: "change a".into(),
            state: WorkerState::Done,
            branch,
            worktree: wt.clone(),
            last_message: String::new(),
            answer: String::new(),
            error: None,
            tool_calls: 0,
            tokens: None,
            cost_micros: None,
            on_plan: false,
            merge: None,
        };
        let before = git(&root, &["rev-parse", "main"]);
        let (approvals, mut queue) = mpsc::unbounded_channel();
        let merging = {
            let (root, s) = (root.clone(), s.clone());
            tokio::spawn(async move {
                merge_worker(
                    &root,
                    "w1",
                    &s,
                    "main",
                    std::time::Duration::from_secs(60),
                    &approvals,
                )
                .await
            })
        };
        // The project has no checks, so the person is asked.
        let (_request, answer) = queue.recv().await.unwrap();
        std::fs::write(wt.join("a.txt"), "changed while asking\n").unwrap();
        answer.send(ApprovalAnswer::Approved).unwrap();
        let outcome = merging.await.unwrap();
        match &outcome {
            MergeOutcome::Refused { why } => assert!(why.contains("changed"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(git(&root, &["rev-parse", "main"]), before);
    }

    /// Security review (authorization scope): a worker's prompt is asked as an
    /// ordinary approval, and "always allow" answers that one prompt only —
    /// nothing is remembered, so the next prompt from any worker or from the
    /// person's own agent is asked again.
    #[tokio::test]
    async fn always_allow_on_a_worker_prompt_answers_that_prompt_only() {
        let mut team = Team::default();
        let (approvals, mut queue) = mpsc::unbounded_channel();
        for n in 0..2 {
            let (tx, rx) = oneshot::channel();
            team.events_tx
                .send(WorkerEvent::Permission {
                    worker: "w1".into(),
                    summary: format!("write notes {n}"),
                    tool: "edit".into(),
                    answer: tx,
                })
                .unwrap();
            team.pump(&approvals);
            let (request, answer) = queue.try_recv().expect("asked as an approval");
            assert_eq!(request.class, ToolClass::External);
            assert!(
                request.summary.starts_with("w1 asks"),
                "{}",
                request.summary
            );
            answer.send(ApprovalAnswer::ApprovedForSession).unwrap();
            assert!(rx.await.unwrap(), "that prompt was allowed");
        }
        // The second prompt above was queued and asked again: nothing was
        // allowed in advance.
        assert!(queue.try_recv().is_err());
    }
}
