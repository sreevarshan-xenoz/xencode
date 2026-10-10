//! The engine's team of worker agents (TM-1): it starts workers in their own
//! worktrees, keeps what each has said, answers the team requests windows and
//! the lead send, and turns a worker's permission prompt into an ordinary
//! engine approval, so the person answers it in any window.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};
use xencode_team_rs::worker::{LaunchSpec, WorkerEvent, WorkerHandle};
use xencode_team_rs::{worktree, WorkerSnapshot};

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
}

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
        }
    }
}

impl Team {
    /// Every worker as last seen, in id order.
    pub fn snapshots(&self) -> Vec<WorkerSnapshot> {
        self.shown.values().cloned().collect()
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
    pub fn request(&mut self, root: &Path, request: TeamRequest) -> Result<Value, String> {
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
            TeamRequest::Merge { .. } => Err("merging lands in TM-3".to_string()),
        }
    }

    fn find(&self, id: &str) -> Result<WorkerSnapshot, String> {
        match self.workers.get(id) {
            Some(handle) => Ok(handle.snapshot()),
            None => Err(format!("no worker {id}")),
        }
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
        if running >= self.limit {
            return Err(format!(
                "{running} workers are already running, the limit is {}; stop one or raise `limit` in .xencode/team.toml",
                self.limit
            ));
        }
        self.next += 1;
        let id = format!("w{}", self.next);
        let base = base.unwrap_or_else(|| base_of(root));
        let (path, branch) = worktree::create(root, &id, &base)?;
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
