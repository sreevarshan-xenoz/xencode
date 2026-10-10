//! The engine's team of worker agents (TM-1): it starts workers in their own
//! worktrees, keeps what each has said, answers the team requests windows and
//! the lead send, and turns a worker's permission prompt into an ordinary
//! engine approval, so the person answers it in any window.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};
use xencode_config_rs::XencodeConfig;
use xencode_team_rs::agents::{self, AgentSpec, Availability, SignIn, StoredKey};
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
    /// Whether the worktrees an earlier engine left were looked for yet.
    adopted: bool,
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
            adopted: false,
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
                    self.shown.insert(snapshot.id.clone(), *snapshot);
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
                        tool: format!("worker {worker}: {}", shown(&tool)),
                        // Another agent's tool: xencode can neither preview,
                        // checkpoint nor undo it.
                        class: ToolClass::External,
                        summary: format!("{worker} asks: {}", shown(&summary)),
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
        config: &XencodeConfig,
    ) -> Result<Value, String> {
        if !self.adopted {
            self.adopted = true;
            self.adopt(root);
        }
        match request {
            TeamRequest::Agents => Ok(agents_listing(config)),
            TeamRequest::Start { agent, task, base } => {
                let spec = launch_for(&agent, config)?;
                self.start(root, &agent, &task, base, spec)
            }
            TeamRequest::Clean => self.clean(root),
            TeamRequest::Status { id: Some(id) } => {
                let s = self.find(&id)?;
                serde_json::to_value(s).map_err(|e| e.to_string())
            }
            TeamRequest::Status { id: None } => {
                serde_json::to_value(self.snapshots()).map_err(|e| e.to_string())
            }
            TeamRequest::Result { id } => {
                let s = self.find(&id)?;
                let base = self
                    .bases
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| base_of(root));
                let diff = worktree::diff(&s.worktree, &base)?;
                Ok(json!({ "id": id, "state": s.state, "answer": s.answer, "diff": diff }))
            }
            TeamRequest::Message { id, text } => {
                // A landing stops the worker and removes its worktree; it is
                // not given new work meanwhile.
                if matches!(self.merges.get(&id), Some(MergeState::Running)) {
                    return Err(format!(
                        "{id} is being merged; message it once the merge ends"
                    ));
                }
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

    /// Remove the worktrees of workers that are not running: stopped,
    /// failed, or left by an engine that has ended. One with changes nobody
    /// committed is kept and said.
    fn clean(&mut self, root: &Path) -> Result<Value, String> {
        let mut removed = Vec::new();
        let mut kept = Vec::new();
        // A folder a removal left behind (git no longer lists it) is xencode's
        // own leftover, finished off unless its worker still runs.
        for (id, path) in worktree::leftover_folders(root)? {
            if self.workers.get(&id).is_some_and(|h| {
                !matches!(
                    h.snapshot().state,
                    WorkerState::Stopped | WorkerState::Failed
                )
            }) {
                continue;
            }
            match worktree::remove(root, &path, &worktree::branch_for(&id)) {
                Ok(()) => {
                    self.workers.remove(&id);
                    self.shown.remove(&id);
                    removed.push(id);
                }
                Err(why) => kept.push(json!({ "id": id, "why": why })),
            }
        }
        for (id, path, branch) in worktree::team_worktrees(root)? {
            if let Some(handle) = self.workers.get(&id) {
                let state = handle.snapshot().state;
                if !matches!(state, WorkerState::Stopped | WorkerState::Failed) {
                    kept.push(
                        json!({ "id": id, "why": format!("it is {state:?}; stop it first") }),
                    );
                    continue;
                }
            }
            match worktree::has_changes(&path) {
                Ok(false) => {}
                Ok(true) => {
                    kept.push(json!({
                        "id": id,
                        "why": format!("{} has changes nobody committed", path.display())
                    }));
                    continue;
                }
                Err(why) => {
                    kept.push(json!({ "id": id, "why": why }));
                    continue;
                }
            }
            match worktree::remove(root, &path, &branch) {
                Ok(()) => {
                    self.workers.remove(&id);
                    self.shown.remove(&id);
                    removed.push(id);
                }
                Err(why) => kept.push(json!({ "id": id, "why": why })),
            }
        }
        // Scratch worktrees merges left; never while a merge runs.
        if !self
            .merges
            .values()
            .any(|m| matches!(m, MergeState::Running))
        {
            removed.extend(worktree::sweep_scratch(root));
        }
        Ok(json!({ "removed": removed, "kept": kept }))
    }

    /// Take in the workers an earlier engine on this project left: each
    /// worktree it made is shown as a stopped worker, which can be merged or
    /// cleaned, and the next worker gets a number after theirs.
    fn adopt(&mut self, root: &Path) {
        let mut found: Vec<(String, PathBuf, String)> =
            worktree::team_worktrees(root).unwrap_or_default();
        for (id, path) in worktree::leftover_folders(root).unwrap_or_default() {
            let branch = worktree::branch_for(&id);
            found.push((id, path, branch));
        }
        for (id, path, branch) in found {
            let Ok(n) = id[1..].parse::<u32>() else {
                continue;
            };
            self.next = self.next.max(n);
            if self.shown.contains_key(&id) {
                continue;
            }
            self.shown.insert(
                id.clone(),
                WorkerSnapshot {
                    id,
                    agent: "unknown".to_string(),
                    task: String::new(),
                    state: WorkerState::Stopped,
                    branch,
                    worktree: path,
                    last_message: "left by an engine that ended".to_string(),
                    answer: String::new(),
                    error: None,
                    tool_calls: 0,
                    tokens: None,
                    cost_micros: None,
                    on_plan: false,
                    merge: None,
                },
            );
        }
    }

    /// Whether a worker is still going (starting, working or waiting on the
    /// person) or a merge is running: the engine does not end while one is,
    /// since workers belong to the project, not to the lead's session.
    pub fn has_work(&self) -> bool {
        self.merges
            .values()
            .any(|m| matches!(m, MergeState::Running))
            || self.workers.values().any(|h| {
                matches!(
                    h.snapshot().state,
                    WorkerState::Starting | WorkerState::Working | WorkerState::NeedsYou
                )
            })
    }

    fn find(&self, id: &str) -> Result<WorkerSnapshot, String> {
        match (self.workers.get(id), self.shown.get(id)) {
            (Some(handle), _) => Ok(self.with_merge(handle.snapshot())),
            // One an earlier engine left: stopped, with no process to ask.
            (None, Some(s)) => Ok(self.with_merge(s.clone())),
            (None, None) => Err(format!("no worker {id}")),
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
        // A worker an earlier engine left has no process to stop, and
        // nothing says who made its worktree.
        let stop = self.workers.get(id).map(|h| h.stopper());
        let unverified = stop.is_none();
        let (root, id_owned, approvals, events) = (
            root.to_path_buf(),
            id.to_string(),
            approvals.clone(),
            self.events_tx.clone(),
        );
        tokio::spawn(async move {
            let outcome = if unverified {
                match left_worker_allowed(&id_owned, &s, &base, &approvals).await {
                    Ok(tip) => {
                        merge_worker(&root, &id_owned, &s, &base, timeout, &approvals, Some(tip))
                            .await
                    }
                    Err(refused) => refused,
                }
            } else {
                merge_worker(&root, &id_owned, &s, &base, timeout, &approvals, None).await
            };
            audit_merge(&id_owned, &s, &base, &outcome);
            if matches!(outcome, MergeOutcome::Landed { .. }) {
                // Its work is in; the worker ends and its worktree goes. On
                // Windows a folder whose files a process holds open cannot be
                // removed, and the worker's own engine keeps the worktree's
                // files until it exits, idle, after the worker ends; so the
                // removal is tried for longer than that.
                if let Some(stop) = &stop {
                    stop();
                }
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
        spec: LaunchSpec,
    ) -> Result<Value, String> {
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
        if !worktree::is_branch(root, &base) {
            self.next -= 1;
            return Err(format!(
                "`{base}` is not a branch here; a worker's work lands on a branch, so its base \
                 must be one"
            ));
        }
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
    pinned: Option<String>,
) -> MergeOutcome {
    // `asked` is the commit the person was asked about: once they answer,
    // exactly that commit is merged, or nothing if the work moved since.
    // `pinned` is one they were asked about before the merge began.
    let run = |checks: Checks, asked: Option<String>| {
        let (root, wt, base, task, id, pinned) = (
            root.to_path_buf(),
            s.worktree.clone(),
            base.to_string(),
            s.task.clone(),
            id.to_string(),
            pinned.clone(),
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
            let moved = |since: &str| committed || tip != since;
            let changed = || MergeOutcome::Refused {
                why: format!(
                    "{id}'s work changed after you were asked; merge it again to be asked \
                     about what it is now"
                ),
            };
            match asked {
                None if pinned.as_deref().is_some_and(moved) => (changed(), None),
                None => {
                    let outcome = merge::checked_merge(&root, &base, &tip, &checks, timeout);
                    (outcome, Some(tip))
                }
                Some(asked) if moved(&asked) => (changed(), None),
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
        (MergeOutcome::NeedsPerson { why, files }, Some(tip)) => {
            let short: String = tip.chars().take(10).collect();
            let request = ApprovalRequest {
                tool: format!("merge {id}"),
                class: ToolClass::External,
                summary: if unchecked {
                    format!("land {id} (commit {short}) on {base} without checks? ({why})")
                } else {
                    format!("land {id} (commit {short}) on {base} once its checks pass? ({why})")
                },
                // Every file the question is about, in full.
                preview: files.join("\n"),
                draft: ApprovalDraft::default(),
            };
            let (tx, rx) = oneshot::channel();
            if approvals.send((request, tx)).is_err() {
                return MergeOutcome::NeedsPerson { why, files };
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

/// Ask the person before merging a worker an earlier engine left: xencode
/// did not start it here, and a worker running git itself could have made
/// such a worktree. Its work is committed first and the question names that
/// commit; `Ok` with it after a yes, so only it is merged. The checks still
/// run.
async fn left_worker_allowed(
    id: &str,
    s: &WorkerSnapshot,
    base: &str,
    approvals: &Approvals,
) -> Result<String, MergeOutcome> {
    let wt = s.worktree.clone();
    let tip = tokio::task::spawn_blocking(move || {
        merge::commit_work(&wt, "work by a worker an earlier engine left")?;
        worktree::commit_of(&wt, "HEAD")
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r)
    .map_err(|why| MergeOutcome::Refused { why })?;
    let short: String = tip.chars().take(10).collect();
    let request = ApprovalRequest {
        tool: format!("merge {id}"),
        class: ToolClass::External,
        summary: format!(
            "merge {id} (commit {short}) onto {base}? It was left by an earlier engine, so \
             xencode cannot say who made it; read {} first",
            s.branch
        ),
        preview: s.worktree.display().to_string(),
        draft: ApprovalDraft::default(),
    };
    let (tx, rx) = oneshot::channel();
    let refused = MergeOutcome::Refused {
        why: format!("{id} was left by an earlier engine and the person did not agree to merge it"),
    };
    if approvals.send((request, tx)).is_err() {
        return Err(refused);
    }
    match rx.await {
        Ok(ApprovalAnswer::Approved | ApprovalAnswer::ApprovedForSession) => Ok(tip),
        _ => Err(refused),
    }
}

/// Text a worker's agent chose, as the person's question shows it: escaped,
/// so a character that reorders or hides text shows as its code, and clipped.
fn shown(text: &str) -> String {
    const LIMIT: usize = 300;
    let escaped = text.escape_debug().to_string();
    if escaped.chars().count() > LIMIT {
        let clipped: String = escaped.chars().take(LIMIT).collect();
        format!("{clipped}…")
    } else {
        escaped
    }
}

/// One line in the chained audit trail (`xencode audit verify`) for every
/// merge's end: which worker, onto which branch, and what happened. A
/// trail that cannot be written is said on the engine's standard error; the
/// merge itself has already ended.
fn audit_merge(id: &str, s: &WorkerSnapshot, base: &str, outcome: &MergeOutcome) {
    let detail = match outcome {
        MergeOutcome::Landed { commit } => {
            format!("landed {commit} from {} ({})", s.branch, s.agent)
        }
        MergeOutcome::ChecksFailed { .. } => format!("checks failed; {} kept off {base}", s.branch),
        MergeOutcome::Conflict { files } => format!("conflict in {} file(s)", files.len()),
        MergeOutcome::NeedsPerson { why, .. } => format!("needs the person: {why}"),
        MergeOutcome::Refused { why } => format!("refused: {why}"),
    };
    let outcome_word = serde_json::to_value(outcome)
        .ok()
        .and_then(|v| {
            v.get("outcome")
                .and_then(|o| o.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default();
    let event = xencode_collaboration_rs::AuditEvent {
        seq: 0,
        at: xencode_collaboration_rs::audit_stamp(),
        actor: format!("lead agent, worker {id}"),
        action: xencode_collaboration_rs::AuditAction::TeamMerge,
        target: base.to_string(),
        detail: format!("{outcome_word}: {detail}"),
    };
    let written = xencode_config_rs::paths::state_dir()
        .map_err(|e| e.to_string())
        .and_then(|dir| {
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            xencode_server_rs::audit::AuditSink::to_file(dir.join("audit.jsonl"))
                .append_external(&event)
        });
    if let Err(why) = written {
        eprintln!("could not write the audit line for {id}'s merge: {why}");
    }
}

/// Whether an approval is one the team asked for: a worker's permission
/// prompt or a merge without checks. The badge says those are a worker's.
pub fn is_team_approval(request: &ApprovalRequest) -> bool {
    ["worker w", "merge w"]
        .iter()
        .any(|p| request.tool.starts_with(p))
}

/// The branch the project has checked out, which workers start from.
fn base_of(root: &Path) -> String {
    worktree::current_branch(root).unwrap_or_else(|| "HEAD".to_string())
}

/// How to start `agent`. TM-1 knows xencode itself; the outside agents come
/// with sign-in in TM-5.
/// What `team_agents` says: xencode, then each outside agent and whether it
/// can start here now.
fn agents_listing(config: &XencodeConfig) -> Value {
    let mut out = vec![json!({ "name": "xencode", "available": true, "sign_in": "none needed" })];
    for spec in agents::AGENTS {
        let mut row = match describe(spec, config) {
            Availability::Ready {
                sign_in: SignIn::Key { var },
                ..
            } => {
                json!({ "name": spec.name, "available": true, "sign_in": format!("API key ({var})") })
            }
            Availability::Ready {
                sign_in: SignIn::Login,
                ..
            } => json!({
                "name": spec.name,
                "available": true,
                "sign_in": "your own login (turned on with `xencode team login-optin`); on your plan, not priced"
            }),
            Availability::Missing { install } => {
                json!({ "name": spec.name, "available": false, "missing": install })
            }
            Availability::NoSignIn { fix } => {
                json!({ "name": spec.name, "available": false, "sign_in": fix })
            }
        };
        if !config.allow_external_workers {
            row["available"] = json!(false);
            row["refused"] = json!(POSTURE_REFUSAL);
        }
        out.push(row);
    }
    Value::Array(out)
}

const POSTURE_REFUSAL: &str = "Local Only refuses another vendor's agent; \
     `xencode config set allow_external_workers true` allows it";

/// Whether an outside agent can start here now, read from this machine.
fn describe(spec: &AgentSpec, config: &XencodeConfig) -> Availability {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let program = agents::find_program(spec.program, &path);
    let key = key_for(spec, config);
    let settings_dir = XencodeConfig::config_dir().ok();
    let opted_in = settings_dir
        .as_deref()
        .map(|d| agents::optins(d).iter().any(|n| n == spec.name))
        .unwrap_or(false);
    let home = dirs::home_dir().map(|h| agents::antigravity_settings(&h));
    agents::availability(
        spec,
        program,
        key.as_ref().map(|(var, _)| var.as_str()),
        opted_in,
        home.as_deref(),
    )
}

/// The API key for `spec`: from its own environment variables, else from
/// xencode's settings, handed on under the variable the agent reads.
fn key_for(spec: &AgentSpec, config: &XencodeConfig) -> Option<(String, String)> {
    for var in spec.key_vars {
        if let Ok(value) = std::env::var(var) {
            if !value.trim().is_empty() {
                return Some((var.to_string(), value));
            }
        }
    }
    let provider = match spec.stored_key? {
        StoredKey::OpenAi => xencode_config_rs::SecretProvider::OpenAi,
        StoredKey::Gemini => xencode_config_rs::SecretProvider::Gemini,
    };
    let value = config.api_keys.secret(provider).ok().flatten()?;
    let var = spec.key_vars.last()?;
    Some((var.to_string(), value))
}

/// How to start `agent`, refused in words when it cannot start.
fn launch_for(agent: &str, config: &XencodeConfig) -> Result<LaunchSpec, String> {
    if agent == "xencode" {
        return launch_spec(agent);
    }
    let Some(spec) = agents::find(agent) else {
        let known: Vec<&str> = std::iter::once("xencode")
            .chain(agents::AGENTS.iter().map(|a| a.name))
            .collect();
        return Err(format!(
            "xencode cannot start an agent called `{agent}`; it knows: {}",
            known.join(", ")
        ));
    };
    if !config.allow_external_workers {
        return Err(format!("{agent}: {POSTURE_REFUSAL}"));
    }
    match describe(spec, config) {
        Availability::Ready { program, sign_in } => {
            let env: Vec<(String, String)> = match sign_in {
                SignIn::Key { .. } => key_for(spec, config).into_iter().collect(),
                SignIn::Login => Vec::new(),
            };
            let xencodes: Vec<&str> = xencode_config_rs::SecretProvider::ALL
                .iter()
                .flat_map(|p| p.env_vars().iter().copied())
                .collect();
            let hidden = agents::hidden_vars(spec, &sign_in, &xencodes);
            let exe = std::env::current_exe()
                .map_err(|e| format!("cannot find the xencode program: {e}"))?;
            let dir = launch_dir()?;
            Ok(wrapped(
                exe,
                &dir,
                &hidden,
                &program,
                spec.args,
                env,
                matches!(sign_in, SignIn::Login),
            ))
        }
        Availability::Missing { install } => Err(format!("{agent} cannot start: {install}")),
        Availability::NoSignIn { fix } => Err(format!("{agent} cannot sign in: {fix}")),
    }
}

/// The empty folder outside agents' programs run in: xencode's own, never a
/// worker's tree, so a `.npmrc` or `node_modules` there cannot change which
/// adapter `npx` runs.
fn launch_dir() -> Result<PathBuf, String> {
    let dir = XencodeConfig::config_dir()
        .map_err(|e| e.to_string())?
        .join("team-launch");
    std::fs::create_dir_all(&dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
    Ok(dir)
}

/// An outside agent's program, started through `xencode team exec` so it runs
/// in `dir` with the `hidden` variables removed.
fn wrapped(
    exe: PathBuf,
    dir: &Path,
    hidden: &[String],
    program: &Path,
    args: &[&str],
    env: Vec<(String, String)>,
    on_plan: bool,
) -> LaunchSpec {
    let mut all = vec![
        "team".to_string(),
        "exec".to_string(),
        "--cwd".to_string(),
        dir.to_string_lossy().to_string(),
    ];
    for var in hidden {
        all.push("--unset".to_string());
        all.push(var.clone());
    }
    all.push("--".to_string());
    all.push(program.to_string_lossy().to_string());
    all.extend(args.iter().map(|a| a.to_string()));
    LaunchSpec {
        program: exe,
        args: all,
        env,
        on_plan,
    }
}

fn launch_spec(agent: &str) -> Result<LaunchSpec, String> {
    match agent {
        "xencode" => {
            let exe: PathBuf = std::env::current_exe()
                .map_err(|e| format!("cannot find the xencode program: {e}"))?;
            Ok(LaunchSpec {
                program: exe,
                args: vec!["acp".to_string()],
                env: Vec::new(),
                on_plan: false,
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

    /// Security review: an outside agent runs through `xencode team exec`, in
    /// xencode's own folder, with every other key removed.
    #[test]
    fn an_outside_agent_runs_outside_the_worktree_without_other_keys() {
        let spec = wrapped(
            PathBuf::from("/bin/xencode"),
            Path::new("/cfg/team-launch"),
            &[
                "ANTHROPIC_API_KEY".to_string(),
                "OPENAI_API_KEY".to_string(),
            ],
            Path::new("/usr/bin/gemini"),
            &["--acp"],
            vec![("GEMINI_API_KEY".into(), "FAKE-NOT-A-REAL-KEY".into())],
            false,
        );
        assert_eq!(spec.program, PathBuf::from("/bin/xencode"));
        let args = spec.args.join(" ");
        assert!(
            args.starts_with("team exec --cwd /cfg/team-launch --unset ANTHROPIC_API_KEY --unset OPENAI_API_KEY -- "),
            "{args}"
        );
        assert!(args.ends_with(" --acp"), "{args}");
        assert_eq!(spec.env.len(), 1);
    }

    /// TM-4: the badge says a worker is waiting only for the team's own
    /// approvals, not for the person's agent's tools.
    #[test]
    fn the_teams_approvals_are_told_apart() {
        let ask = |tool: &str| ApprovalRequest {
            tool: tool.to_string(),
            class: ToolClass::External,
            summary: String::new(),
            preview: String::new(),
            draft: ApprovalDraft::default(),
        };
        assert!(is_team_approval(&ask("worker w1: edit")));
        assert!(is_team_approval(&ask("merge w2")));
        assert!(!is_team_approval(&ask("write_file")));
        assert!(!is_team_approval(&ask("merge_branches")));
    }

    /// Review: a worker being merged is not given new work.
    #[test]
    fn a_message_to_a_worker_being_merged_is_refused() {
        let mut team = Team {
            adopted: true,
            ..Team::default()
        };
        team.merges.insert("w1".into(), MergeState::Running);
        let (approvals, _queue) = mpsc::unbounded_channel();
        let err = team
            .request(
                Path::new("."),
                TeamRequest::Message {
                    id: "w1".into(),
                    text: "more".into(),
                },
                &approvals,
                &XencodeConfig::default(),
            )
            .unwrap_err();
        assert!(err.contains("being merged"), "{err}");
    }

    /// Security review: a worker's agent chooses its permission text, which
    /// goes into the person's question; it cannot restyle or hide parts of it.
    #[tokio::test]
    async fn a_workers_permission_text_cannot_restyle_the_question() {
        let mut team = Team::default();
        let (approvals, mut queue) = mpsc::unbounded_channel();
        let (answer, _wait) = oneshot::channel();
        team.events_tx
            .send(WorkerEvent::Permission {
                worker: "w1".into(),
                summary: format!("read a file\u{202e}etirw{}", "x".repeat(500)),
                tool: "edit\u{200b}".into(),
                answer,
            })
            .unwrap();
        team.pump(&approvals);
        let (request, _) = queue.recv().await.unwrap();
        assert!(!request.summary.contains('\u{202e}'), "{}", request.summary);
        assert!(request.summary.contains("\\u{202e}"), "{}", request.summary);
        assert!(request.summary.chars().count() < 340, "clipped");
        assert!(!request.tool.contains('\u{200b}'), "{}", request.tool);
    }

    /// Security review: a worktree an earlier engine left cannot be traced
    /// to anyone (a worker could have made one itself), so merging it asks the
    /// person first, and a no lands nothing.
    #[tokio::test]
    async fn merging_a_worker_left_by_an_earlier_engine_asks_the_person_first() {
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
        // Checks that would pass, so only the question stands in the way.
        std::fs::create_dir_all(root.join(".xencode")).unwrap();
        std::fs::write(
            root.join(".xencode").join("team.toml"),
            "checks = [\"git --version\"]\n",
        )
        .unwrap();
        std::fs::write(root.join(".gitignore"), ".xencode/\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "first"]);
        let (wt, _branch) = worktree::create(&root, "w7", "main").unwrap();
        std::fs::write(wt.join("planted.txt"), "x\n").unwrap();
        let before = git(&root, &["rev-parse", "main"]);

        let mut team = Team::default();
        let (approvals, mut queue) = mpsc::unbounded_channel();
        let config = XencodeConfig::default();
        let started = team.request(
            &root,
            TeamRequest::Merge { id: "w7".into() },
            &approvals,
            &config,
        );
        assert!(started.is_ok(), "{started:?}");
        let (request, answer) = queue.recv().await.unwrap();
        assert!(
            request.summary.contains("earlier engine"),
            "{}",
            request.summary
        );
        answer.send(ApprovalAnswer::Denied).unwrap();
        let start = std::time::Instant::now();
        loop {
            team.pump(&approvals);
            let s = team.find("w7").unwrap();
            if let Some(MergeState::Finished(outcome)) = s.merge {
                assert!(
                    matches!(outcome, MergeOutcome::Refused { .. }),
                    "{outcome:?}"
                );
                break;
            }
            assert!(start.elapsed() < std::time::Duration::from_secs(30));
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert_eq!(git(&root, &["rev-parse", "main"]), before);
    }

    /// Security review: the yes to merging a left worker is for the commit
    /// it named; work added while the question waited is not merged.
    #[tokio::test]
    async fn a_left_worker_that_changes_after_the_yes_is_not_merged() {
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
        std::fs::create_dir_all(root.join(".xencode")).unwrap();
        std::fs::write(
            root.join(".xencode").join("team.toml"),
            "checks = [\"git --version\"]\n",
        )
        .unwrap();
        std::fs::write(root.join(".gitignore"), ".xencode/\n").unwrap();
        git(&root, &["add", "."]);
        git(&root, &["commit", "-q", "-m", "first"]);
        let (wt, _branch) = worktree::create(&root, "w7", "main").unwrap();
        std::fs::write(wt.join("shown.txt"), "x\n").unwrap();
        let before = git(&root, &["rev-parse", "main"]);

        let mut team = Team::default();
        let (approvals, mut queue) = mpsc::unbounded_channel();
        let started = team.request(
            &root,
            TeamRequest::Merge { id: "w7".into() },
            &approvals,
            &XencodeConfig::default(),
        );
        assert!(started.is_ok(), "{started:?}");
        let (request, answer) = queue.recv().await.unwrap();
        assert!(request.summary.contains("commit "), "{}", request.summary);
        std::fs::write(wt.join("added-while-asking.txt"), "y\n").unwrap();
        answer.send(ApprovalAnswer::Approved).unwrap();
        let start = std::time::Instant::now();
        loop {
            team.pump(&approvals);
            if let Some(MergeState::Finished(outcome)) = team.find("w7").unwrap().merge {
                match outcome {
                    MergeOutcome::Refused { why } => assert!(why.contains("changed"), "{why}"),
                    other => panic!("{other:?}"),
                }
                break;
            }
            assert!(start.elapsed() < std::time::Duration::from_secs(30));
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert_eq!(git(&root, &["rev-parse", "main"]), before);
    }

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
                    None,
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
