//! One worker agent driven over the Agent Client Protocol (TM-1): the agent
//! runs as a child process with its worktree as the session folder; what it
//! says is kept in a [`WorkerSnapshot`], every change is sent as a
//! [`WorkerEvent`], and its permission prompts are handed to whoever answers
//! them (the engine).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::schema::v1::{
    CancelNotification, ContentBlock, InitializeRequest, NewSessionRequest, PermissionOptionKind,
    PromptRequest, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionNotification, SessionUpdate, StopReason, TextContent,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{AcpAgent, AcpAgentConfig, Agent, Client, ConnectionTo};
use tokio::sync::{mpsc, oneshot};

use crate::{WorkerId, WorkerSnapshot, WorkerState};

/// How to start a worker's agent.
#[derive(Debug, Clone)]
pub struct LaunchSpec {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

/// What a worker reports.
#[derive(Debug)]
pub enum WorkerEvent {
    /// Its state or what it said changed.
    Changed(Box<WorkerSnapshot>),
    /// Its merge ended (TM-3).
    Merged {
        worker: WorkerId,
        outcome: crate::merge::MergeOutcome,
    },
    /// It asks permission for a tool; `answer` takes yes or no.
    Permission {
        worker: WorkerId,
        summary: String,
        tool: String,
        answer: oneshot::Sender<bool>,
    },
}

enum Command {
    Message(String),
    Stop,
}

/// A running worker.
pub struct WorkerHandle {
    commands: mpsc::UnboundedSender<Command>,
    snapshot: Arc<Mutex<WorkerSnapshot>>,
}

/// How long a cancelled turn may take to answer before the worker is ended.
const CANCEL_WAIT: Duration = Duration::from_secs(20);

impl WorkerHandle {
    /// Start the agent in `worktree` and give it `task`.
    pub fn start(
        id: WorkerId,
        agent: &str,
        task: &str,
        spec: LaunchSpec,
        worktree: PathBuf,
        branch: String,
        events: mpsc::UnboundedSender<WorkerEvent>,
    ) -> WorkerHandle {
        let snapshot = Arc::new(Mutex::new(WorkerSnapshot {
            id,
            agent: agent.to_string(),
            task: task.to_string(),
            state: WorkerState::Starting,
            branch,
            worktree,
            last_message: String::new(),
            answer: String::new(),
            error: None,
            tool_calls: 0,
            tokens: None,
            cost_micros: None,
            on_plan: false,
            merge: None,
        }));
        let (commands, receiver) = mpsc::unbounded_channel();
        let shared = Shared {
            snapshot: Arc::clone(&snapshot),
            events,
        };
        let task = task.to_string();
        tokio::spawn(async move {
            let ending = run(spec, task, receiver, shared.clone()).await;
            shared.update(|s| match ending {
                Ok(state) => s.state = state,
                Err(why) => {
                    s.state = WorkerState::Failed;
                    s.error = Some(why);
                }
            });
        });
        WorkerHandle { commands, snapshot }
    }

    /// What the worker is and has said, now.
    pub fn snapshot(&self) -> WorkerSnapshot {
        self.snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// A follow-up prompt in the same session. Refused while a turn runs.
    pub fn message(&self, text: &str) -> Result<(), String> {
        let state = self.snapshot().state;
        if state != WorkerState::Done {
            return Err(format!(
                "the worker is {state:?}; a message can be sent once its turn is done"
            ));
        }
        self.commands
            .send(Command::Message(text.to_string()))
            .map_err(|_| "the worker has ended".to_string())
    }

    /// Stop the worker; its worktree is kept.
    pub fn stop(&self) {
        let _ = self.commands.send(Command::Stop);
    }

    /// Something that stops this worker later, from another task.
    pub fn stopper(&self) -> impl Fn() + Send + Sync + 'static {
        let commands = self.commands.clone();
        move || {
            let _ = commands.send(Command::Stop);
        }
    }
}

/// The snapshot and the event channel, shared with the protocol handlers.
#[derive(Clone)]
struct Shared {
    snapshot: Arc<Mutex<WorkerSnapshot>>,
    events: mpsc::UnboundedSender<WorkerEvent>,
}

impl Shared {
    fn update(&self, change: impl FnOnce(&mut WorkerSnapshot)) {
        let copy = {
            let mut s = self.snapshot.lock().unwrap_or_else(|e| e.into_inner());
            change(&mut s);
            s.clone()
        };
        let _ = self.events.send(WorkerEvent::Changed(Box::new(copy)));
    }

    fn id(&self) -> WorkerId {
        self.snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .id
            .clone()
    }
}

/// Run the worker to its end; `Ok` with how it ended, `Err` with why it
/// failed.
async fn run(
    spec: LaunchSpec,
    task: String,
    mut commands: mpsc::UnboundedReceiver<Command>,
    shared: Shared,
) -> Result<WorkerState, String> {
    let worktree = shared
        .snapshot
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .worktree
        .clone();
    let mut config = AcpAgentConfig::new(spec.program.clone()).args(spec.args.clone());
    for (k, v) in &spec.env {
        config = config.env(k.clone(), v.clone());
    }
    let on_update = shared.clone();
    let on_permission = shared.clone();
    let session = shared.clone();
    Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _c| {
                on_update.update(|s| record(s, &n.update));
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |req: RequestPermissionRequest, responder, connection| {
                let shared = on_permission.clone();
                connection.spawn(async move {
                    let (tx, rx) = oneshot::channel();
                    let summary = req
                        .tool_call
                        .fields
                        .title
                        .clone()
                        .unwrap_or_else(|| "a tool".to_string());
                    let tool = format!("{:?}", req.tool_call.fields.kind).to_lowercase();
                    shared.update(|s| s.state = WorkerState::NeedsYou);
                    let _ = shared.events.send(WorkerEvent::Permission {
                        worker: shared.id(),
                        summary,
                        tool,
                        answer: tx,
                    });
                    let yes = rx.await.unwrap_or(false);
                    shared.update(|s| s.state = WorkerState::Working);
                    let wanted = if yes {
                        PermissionOptionKind::AllowOnce
                    } else {
                        PermissionOptionKind::RejectOnce
                    };
                    let outcome = req
                        .options
                        .iter()
                        .find(|o| o.kind == wanted)
                        .map(|o| {
                            RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                                o.option_id.clone(),
                            ))
                        })
                        .unwrap_or(RequestPermissionOutcome::Cancelled);
                    responder.respond(RequestPermissionResponse::new(outcome))
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(
            AcpAgent::new(config),
            async move |c: ConnectionTo<Agent>| {
                c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                let opened = c
                    .send_request(NewSessionRequest::new(worktree))
                    .block_task()
                    .await?;
                let sid = opened.session_id;
                let mut next = Some(task);
                loop {
                    let Some(text) = next.take() else {
                        // Idle between turns: a message starts the next one.
                        match commands.recv().await {
                            Some(Command::Message(text)) => {
                                next = Some(text);
                                continue;
                            }
                            Some(Command::Stop) | None => return Ok(WorkerState::Stopped),
                        }
                    };
                    session.update(|s| {
                        s.state = WorkerState::Working;
                        s.answer.clear();
                    });
                    let turn = c.send_request(PromptRequest::new(
                        sid.clone(),
                        vec![ContentBlock::Text(TextContent::new(text))],
                    ));
                    let mut turn = Box::pin(turn.block_task());
                    let ended = tokio::select! {
                        ended = &mut turn => ended,
                        command = commands.recv() => {
                            match command {
                                Some(Command::Stop) | None => {
                                    c.send_notification(CancelNotification::new(sid.clone()))?;
                                    let _ = tokio::time::timeout(CANCEL_WAIT, turn).await;
                                    return Ok(WorkerState::Stopped);
                                }
                                // Refused by `message` while a turn runs.
                                Some(Command::Message(_)) => turn.await,
                            }
                        }
                    };
                    let ended = ended?;
                    if ended.stop_reason == StopReason::Cancelled {
                        return Ok(WorkerState::Stopped);
                    }
                    session.update(|s| s.state = WorkerState::Done);
                }
            },
        )
        .await
        .map_err(|e| format!("the worker's agent ended: {e}"))
}

/// Fold one session update into the snapshot.
fn record(s: &mut WorkerSnapshot, update: &SessionUpdate) {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => {
            if let ContentBlock::Text(t) = &chunk.content {
                s.answer.push_str(&t.text);
                if let Some(line) = s.answer.lines().rev().find(|l| !l.trim().is_empty()) {
                    s.last_message = line.trim().to_string();
                }
            }
        }
        SessionUpdate::ToolCall(_) => s.tool_calls += 1,
        _ => {}
    }
}
