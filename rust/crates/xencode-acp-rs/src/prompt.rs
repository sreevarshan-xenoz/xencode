//! A prompt from the editor, run as one turn on the session's engine: its
//! text (M-7a), its tool calls and approval prompts (M-7b).

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol::schema::v1::{
    Content, ContentBlock, ContentChunk, Diff, EmbeddedResourceResource, PermissionOption,
    PermissionOptionKind, Plan, PlanEntry, PlanEntryPriority, PlanEntryStatus, PromptRequest,
    PromptResponse, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SessionId, SessionNotification, SessionUpdate, StopReason, TextContent, ToolCall,
    ToolCallContent, ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
    ToolKind,
};
use agent_client_protocol::{Client, ConnectionTo, Responder};
use tokio::sync::Notify;
use xencode_tui_rs::engine::link::{EngineLink, LinkEvent};
use xencode_tui_rs::engine::proto::{
    ApprovalView, ClientMsg, EngineMsg, ReviewDecision, StopTarget, WireAnswer,
};

use crate::kinds;
use crate::session::{self, Session};
use crate::turn::{Ending, Out, ToolEnd, ToolStart, Turn};

/// JSON-RPC's "internal error" code, for a turn the engine could not finish.
pub const INTERNAL: i32 = -32603;

/// The prompt as the one text xencode's chat takes. Attached files are
/// added as fenced blocks under their address; links are named; images and
/// audio are not sent in M-7 and are said on standard error.
pub fn prompt_text(blocks: &[ContentBlock]) -> String {
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text(t) => parts.push(t.text.clone()),
            ContentBlock::ResourceLink(link) => parts.push(format!("(file: {})", link.uri)),
            ContentBlock::Resource(embedded) => match &embedded.resource {
                EmbeddedResourceResource::TextResourceContents(file) => {
                    parts.push(format!("{}:\n```\n{}\n```", file.uri, file.text))
                }
                _ => eprintln!("xencode acp: a binary attachment was left out"),
            },
            _ => eprintln!("xencode acp: images and audio are not sent to the model yet"),
        }
    }
    parts.join("\n\n")
}

fn error(code: i32, message: impl Into<String>) -> agent_client_protocol::Error {
    agent_client_protocol::Error::new(code, message.into())
}

/// Start the prompt's turn. Answers at once with an error when the session
/// is unknown or busy or its engine cannot be reached; otherwise the turn
/// runs in its own task and answers when it ends.
pub async fn start(
    sessions: &session::Sessions,
    req: PromptRequest,
    responder: Responder<PromptResponse>,
    connection: ConnectionTo<Client>,
) -> Result<(), agent_client_protocol::Error> {
    let id = req.session_id.to_string();
    let text = prompt_text(&req.prompt);
    if let Some(why) = crate::options::window_only(&text) {
        return responder.respond_with_error(error(crate::INVALID, why));
    }
    let bytebot_task = text.strip_prefix("/bytebot").map(|t| t.trim().to_string());
    if bytebot_task.as_deref() == Some("") {
        return responder.respond_with_error(error(
            crate::INVALID,
            "say what ByteBot should do: /bytebot <task>",
        ));
    }
    let Some(state) = sessions.get(&id) else {
        return responder.respond_with_error(error(crate::INVALID, format!("no session {id}")));
    };
    let (link, cancel, project, msg, turn) = {
        let mut s = state.lock().await;
        if s.busy {
            return responder.respond_with_error(error(
                crate::INVALID,
                "a turn is already running in this session; wait for it or cancel it",
            ));
        }
        // What the engine sent while no turn was reading belongs to other
        // windows' turns; it is read now and not mistaken for this one's.
        let mut drained = Vec::new();
        let mut link = s.link.take();
        if let Some(open) = link.as_mut() {
            match open.drain() {
                Ok(msgs) => drained = msgs,
                Err(_) => link = None,
            }
        }
        let link = match link {
            Some(link) => link,
            None => match session::connect(&s.project).await {
                Ok((link, _view)) => link,
                Err(why) => {
                    return responder.respond_with_error(error(
                        INTERNAL,
                        format!("cannot reach the engine: {why}"),
                    ))
                }
            },
        };
        let (msg, turn) = match route(s.question.clone(), &drained, bytebot_task, text) {
            Ok(routed) => {
                s.question = None;
                routed
            }
            Err(why) => {
                // Refused before anything was sent: the question still waits.
                s.link = Some(link);
                return responder.respond_with_error(error(crate::INVALID, why));
            }
        };
        s.busy = true;
        s.cancel = Arc::new(Notify::new());
        (link, Arc::clone(&s.cancel), s.project.clone(), msg, turn)
    };
    if !link.send(&msg) {
        finish(&state, None, None).await;
        return responder.respond_with_error(error(
            INTERNAL,
            "the engine was lost before the prompt reached it",
        ));
    }
    let run = TurnRun {
        connection: connection.clone(),
        session_id: req.session_id.clone(),
        link,
        cancel,
        tools: Tools::new(project),
        asking: None,
        waiting: VecDeque::new(),
        bytebot: turn.follows().map(str::to_string),
    };
    connection.spawn(async move {
        let (link, answer) = run.run(turn).await;
        let question = match &answer {
            Ok(Finish::Asked(qid, followed)) => Some((*qid, followed.clone())),
            _ => None,
        };
        finish(&state, link, question).await;
        match answer {
            Ok(Finish::Ended(Ending::Done)) | Ok(Finish::Asked(..)) => {
                responder.respond(PromptResponse::new(StopReason::EndTurn))
            }
            Ok(Finish::Ended(Ending::Stopped)) => {
                responder.respond(PromptResponse::new(StopReason::Cancelled))
            }
            Ok(Finish::Ended(Ending::Failed(why))) | Err(why) => {
                responder.respond_with_error(error(INTERNAL, why))
            }
        }
    })
}

/// What this prompt sends the engine, and how its turn is followed. A
/// waiting ByteBot question takes the message as its answer, unless the
/// question was answered in another window meanwhile (`drained` says so); a
/// new ByteBot task while a question waits is refused, because it would
/// queue behind the task that asked.
pub fn route(
    question: Option<(u64, String)>,
    drained: &[EngineMsg],
    bytebot_task: Option<String>,
    text: String,
) -> Result<(ClientMsg, Turn), String> {
    let question = question.filter(|(qid, _)| {
        !drained
            .iter()
            .any(|m| matches!(m, EngineMsg::QuestionAnswered { id, .. } if id == qid))
    });
    match (question, bytebot_task) {
        (Some(_), Some(_)) => Err(
            "ByteBot is waiting for your answer to its question; answer it first, or stop it"
                .to_string(),
        ),
        (Some((qid, followed)), None) => Ok((
            ClientMsg::AnswerQuestion { id: qid, text },
            Turn::bytebot(&followed),
        )),
        (None, Some(task)) => {
            let turn = Turn::bytebot(&task);
            Ok((ClientMsg::EnqueueTask { text: task }, turn))
        }
        (None, None) => Ok((ClientMsg::SubmitChat { prompt: text }, Turn::new())),
    }
}

/// How a turn finished: it ended, or a ByteBot task asked a question and
/// waits for the person's next message (M-7d).
enum Finish {
    Ended(Ending),
    Asked(u64, String),
}

/// The editor's answer to a permission request, when it comes.
type PermissionAnswer = Pin<
    Box<
        dyn Future<Output = Result<RequestPermissionResponse, agent_client_protocol::Error>> + Send,
    >,
>;

/// What a permission request is about.
#[derive(Clone, Copy, PartialEq)]
enum Asking {
    /// An approval prompt, by id.
    Approval(u64),
    /// A ByteBot task's review: keep or undo its changes.
    Review,
}

/// One running turn: reads the engine, tells the editor, and carries the
/// editor's answers back.
struct TurnRun {
    connection: ConnectionTo<Client>,
    session_id: SessionId,
    link: EngineLink,
    cancel: Arc<Notify>,
    tools: Tools,
    /// The request being asked about, and the editor's answer to come.
    asking: Option<(Asking, PermissionAnswer)>,
    /// Approvals that came while another request was open.
    waiting: VecDeque<ApprovalView>,
    /// The words of the ByteBot task this turn follows, if it does.
    bytebot: Option<String>,
}

impl TurnRun {
    /// Run to the turn's end. Gives back the link (`None` once the engine
    /// was lost) and how the turn finished, or why it could not.
    async fn run(mut self, mut turn: Turn) -> (Option<EngineLink>, Result<Finish, String>) {
        loop {
            let event = tokio::select! {
                event = self.link.next() => event,
                _ = self.cancel.notified() => {
                    self.stop();
                    turn.stopped_by_the_person();
                    continue;
                }
                answered = Self::answer_of(&mut self.asking), if self.asking.is_some() => {
                    self.answered(answered);
                    continue;
                }
            };
            let msg = match event {
                LinkEvent::Msg(msg) => msg,
                LinkEvent::Lost(why) => {
                    return (None, Err(format!("the engine was lost: {why}")));
                }
            };
            if let EngineMsg::ApprovalResolved { id, .. } = &msg {
                self.resolved_elsewhere(*id);
            }
            for out in turn.feed(&msg) {
                let sent = match out {
                    Out::Text(text) => self.say(text),
                    Out::ToolStart(start) => {
                        let update = self.tools.started(start);
                        self.notify(update)
                    }
                    Out::ToolEnd(end) => {
                        let update = self.tools.ended(end);
                        self.notify(update)
                    }
                    Out::Permission(approval) => {
                        self.permission(approval);
                        Ok(())
                    }
                    Out::Plan(steps) => self.notify(SessionUpdate::Plan(plan_of(&steps))),
                    Out::Review => {
                        let answer = self.ask_review();
                        self.asking = Some((Asking::Review, answer));
                        Ok(())
                    }
                    Out::Question(qid, question) => {
                        let followed = self.bytebot.clone().unwrap_or_default();
                        if let Err(e) = self.say(format!(
                            "xencode asks: {question}\n\nReply in your next message."
                        )) {
                            return (
                                Some(self.link),
                                Err(format!("cannot reach the editor: {e}")),
                            );
                        }
                        return (Some(self.link), Ok(Finish::Asked(qid, followed)));
                    }
                    Out::Ended(ending) => return (Some(self.link), Ok(Finish::Ended(ending))),
                };
                if let Err(e) = sent {
                    return (
                        Some(self.link),
                        Err(format!("cannot reach the editor: {e}")),
                    );
                }
            }
        }
    }

    async fn answer_of(
        asking: &mut Option<(Asking, PermissionAnswer)>,
    ) -> Result<RequestPermissionResponse, agent_client_protocol::Error> {
        match asking {
            Some((_, answer)) => answer.as_mut().await,
            None => std::future::pending().await,
        }
    }

    fn notify(&self, update: SessionUpdate) -> Result<(), agent_client_protocol::Error> {
        self.connection
            .send_notification(SessionNotification::new(self.session_id.clone(), update))
    }

    fn say(&self, text: String) -> Result<(), agent_client_protocol::Error> {
        self.notify(SessionUpdate::AgentMessageChunk(ContentChunk::new(
            ContentBlock::Text(TextContent::new(text)),
        )))
    }

    /// The editor cancelled. An open approval is answered no, so the agent
    /// loop is not left waiting on it; an open ByteBot review keeps the
    /// changes, as the engine does when a review waits with nobody there
    /// (nothing is thrown away on a stop); then the run is stopped.
    fn stop(&mut self) {
        match self.asking.take() {
            Some((Asking::Approval(id), _)) => {
                self.link.send(&ClientMsg::AnswerApproval {
                    id,
                    answer: WireAnswer::Deny,
                });
            }
            Some((Asking::Review, _)) => {
                self.link.send(&ClientMsg::Review {
                    decision: ReviewDecision::Accept,
                });
            }
            None => {}
        }
        self.waiting.clear();
        let target = if self.bytebot.is_some() {
            StopTarget::Bytebot
        } else {
            StopTarget::Chat
        };
        self.link.send(&ClientMsg::Stop { target });
    }

    fn permission(&mut self, approval: ApprovalView) {
        if self.asking.is_none() {
            let answer = self.ask(&approval);
            self.asking = Some((Asking::Approval(approval.id), answer));
        } else {
            self.waiting.push_back(approval);
        }
    }

    fn answered(
        &mut self,
        answer: Result<RequestPermissionResponse, agent_client_protocol::Error>,
    ) {
        match self.asking.take() {
            Some((Asking::Approval(id), _)) => {
                self.link.send(&ClientMsg::AnswerApproval {
                    id,
                    answer: wire_answer(answer),
                });
            }
            Some((Asking::Review, _)) => {
                self.link.send(&ClientMsg::Review {
                    decision: review_decision(answer),
                });
            }
            None => {}
        }
        self.ask_next();
    }

    /// Another window answered approval `id` first; its answer counts and
    /// the editor's, if it comes, is not used.
    fn resolved_elsewhere(&mut self, id: u64) {
        if self
            .asking
            .as_ref()
            .is_some_and(|(open, _)| *open == Asking::Approval(id))
        {
            self.asking = None;
            self.ask_next();
        } else {
            self.waiting.retain(|a| a.id != id);
        }
    }

    fn ask_next(&mut self) {
        if self.asking.is_some() {
            return;
        }
        if let Some(next) = self.waiting.pop_front() {
            let answer = self.ask(&next);
            self.asking = Some((Asking::Approval(next.id), answer));
        }
    }

    fn request(&self, call: ToolCallUpdate, options: Vec<PermissionOption>) -> PermissionAnswer {
        Box::pin(
            self.connection
                .send_request(RequestPermissionRequest::new(
                    self.session_id.clone(),
                    call,
                    options,
                ))
                .block_task(),
        )
    }

    /// Ask the editor about one approval prompt.
    fn ask(&self, approval: &ApprovalView) -> PermissionAnswer {
        let call = ToolCallUpdate::new(
            format!("approval-{}", approval.id),
            ToolCallUpdateFields::new()
                .title(approval.summary.clone())
                .kind(kinds::kind_of(&approval.tool)),
        );
        self.request(
            call,
            vec![
                PermissionOption::new("allow", "Allow once", PermissionOptionKind::AllowOnce),
                PermissionOption::new(
                    "allow-session",
                    "Always allow this session",
                    PermissionOptionKind::AllowAlways,
                ),
                PermissionOption::new("deny", "Reject", PermissionOptionKind::RejectOnce),
            ],
        )
    }

    /// Ask the editor whether to keep the ByteBot task's changes.
    fn ask_review(&self) -> PermissionAnswer {
        let call = ToolCallUpdate::new(
            "bytebot-review",
            ToolCallUpdateFields::new()
                .title(format!(
                    "ByteBot finished: {}. Keep its changes?",
                    self.bytebot.as_deref().unwrap_or("its task")
                ))
                .kind(ToolKind::Edit),
        );
        self.request(
            call,
            vec![
                PermissionOption::new("keep", "Keep the changes", PermissionOptionKind::AllowOnce),
                PermissionOption::new("undo", "Undo the changes", PermissionOptionKind::RejectOnce),
            ],
        )
    }
}

/// ByteBot's steps as an ACP plan: a step running now is in progress, one
/// that ended is completed, with how it ended when that was not done.
fn plan_of(steps: &[(String, String)]) -> Plan {
    Plan::new(
        steps
            .iter()
            .map(|(call, state)| {
                let (status, content) = match state.as_str() {
                    "running" => (PlanEntryStatus::InProgress, call.clone()),
                    "done" => (PlanEntryStatus::Completed, call.clone()),
                    other => (PlanEntryStatus::Completed, format!("{call} ({other})")),
                };
                PlanEntry::new(content, PlanEntryPriority::Medium, status)
            })
            .collect(),
    )
}

/// The editor's review answer: only a chosen "keep" keeps the changes.
fn review_decision(
    answer: Result<RequestPermissionResponse, agent_client_protocol::Error>,
) -> ReviewDecision {
    match answer.map(|r| r.outcome) {
        Ok(RequestPermissionOutcome::Selected(chosen)) if &*chosen.option_id.0 == "keep" => {
            ReviewDecision::Accept
        }
        _ => ReviewDecision::Undo,
    }
}

/// The editor's choice as the engine takes it. Anything but a chosen
/// "allow" is a no.
pub fn wire_answer(
    answer: Result<RequestPermissionResponse, agent_client_protocol::Error>,
) -> WireAnswer {
    match answer.map(|r| r.outcome) {
        Ok(RequestPermissionOutcome::Selected(chosen)) => match &*chosen.option_id.0 {
            "allow" => WireAnswer::Allow,
            "allow-session" => WireAnswer::AllowForSession,
            _ => WireAnswer::Deny,
        },
        _ => WireAnswer::Deny,
    }
}

/// The tool calls of one turn, for their updates: an edit remembers the
/// file's text from before it ran, so its end can carry a diff.
struct Tools {
    project: PathBuf,
    before: HashMap<String, (String, Option<String>)>,
}

impl Tools {
    fn new(project: PathBuf) -> Tools {
        Tools {
            project,
            before: HashMap::new(),
        }
    }

    fn started(&mut self, start: ToolStart) -> SessionUpdate {
        let kind = kinds::kind_of(&start.name);
        let path = kinds::path_of(&start.arguments);
        let title = match &path {
            Some(p) => format!("{} {p}", start.name),
            None => start.name.clone(),
        };
        let mut call = ToolCall::new(start.id.clone(), title)
            .kind(kind)
            .status(ToolCallStatus::InProgress)
            .raw_input(start.arguments.clone());
        if let Some(p) = &path {
            call = call.locations(vec![ToolCallLocation::new(self.project.join(p))]);
            if kind == ToolKind::Edit {
                let old = kinds::readable(&self.project.join(p));
                self.before.insert(start.id.clone(), (p.clone(), old));
            }
        }
        SessionUpdate::ToolCall(call)
    }

    fn ended(&mut self, end: ToolEnd) -> SessionUpdate {
        let status = if end.outcome == "done" {
            ToolCallStatus::Completed
        } else {
            ToolCallStatus::Failed
        };
        let mut content = Vec::new();
        if let Some((path, old)) = self.before.remove(&end.id) {
            let full = self.project.join(&path);
            if status == ToolCallStatus::Completed {
                if let Some(new) = kinds::readable(&full) {
                    // No old text: the edit made a new file.
                    content.push(ToolCallContent::Diff(Diff::new(full, new).old_text(old)));
                }
            }
        }
        if content.is_empty() && !end.preview.is_empty() {
            content.push(ToolCallContent::Content(Content::new(ContentBlock::Text(
                TextContent::new(end.preview),
            ))));
        }
        SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            end.id,
            ToolCallUpdateFields::new().status(status).content(content),
        ))
    }
}

/// Wake the session's running turn, which then stops it on the engine.
pub async fn cancel(sessions: &session::Sessions, session_id: &str) {
    if let Some(state) = sessions.get(session_id) {
        let s = state.lock().await;
        if s.busy {
            s.cancel.notify_one();
        }
    }
}

async fn finish(
    state: &tokio::sync::Mutex<Session>,
    link: Option<EngineLink>,
    question: Option<(u64, String)>,
) {
    let mut s = state.lock().await;
    s.busy = false;
    s.link = link;
    s.question = question;
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::SelectedPermissionOutcome;

    fn waiting() -> Option<(u64, String)> {
        Some((7, "write a note".to_string()))
    }

    #[test]
    fn a_waiting_question_takes_the_next_message_as_its_answer() {
        let (msg, turn) = route(waiting(), &[], None, "blue".into()).unwrap();
        assert_eq!(
            msg,
            ClientMsg::AnswerQuestion {
                id: 7,
                text: "blue".into()
            }
        );
        assert_eq!(turn.follows(), Some("write a note"));
    }

    /// Review finding 3: a question answered in another window meanwhile.
    #[test]
    fn a_question_answered_elsewhere_leaves_the_message_as_chat() {
        let answered = [EngineMsg::QuestionAnswered {
            id: 7,
            by: "terminal 1".into(),
        }];
        let (msg, turn) = route(waiting(), &answered, None, "hello".into()).unwrap();
        assert_eq!(
            msg,
            ClientMsg::SubmitChat {
                prompt: "hello".into()
            }
        );
        assert_eq!(turn.follows(), None);
    }

    /// Review finding 4.
    #[test]
    fn a_new_bytebot_task_while_a_question_waits_is_refused() {
        let why = route(
            waiting(),
            &[],
            Some("another".into()),
            "/bytebot another".into(),
        )
        .unwrap_err();
        assert!(why.contains("answer"), "{why}");
    }

    #[test]
    fn text_blocks_are_joined_and_links_named() {
        let blocks = vec![
            ContentBlock::Text(TextContent::new("fix this")),
            ContentBlock::Text(TextContent::new("please")),
        ];
        assert_eq!(prompt_text(&blocks), "fix this\n\nplease");
    }

    #[test]
    fn only_a_chosen_allow_lets_a_tool_run() {
        let chose = |id: &str| {
            Ok(RequestPermissionResponse::new(
                RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(id.to_string())),
            ))
        };
        assert_eq!(wire_answer(chose("allow")), WireAnswer::Allow);
        assert_eq!(
            wire_answer(chose("allow-session")),
            WireAnswer::AllowForSession
        );
        assert_eq!(wire_answer(chose("deny")), WireAnswer::Deny);
        assert_eq!(wire_answer(chose("something-else")), WireAnswer::Deny);
        assert_eq!(
            wire_answer(Ok(RequestPermissionResponse::new(
                RequestPermissionOutcome::Cancelled
            ))),
            WireAnswer::Deny
        );
    }

    #[test]
    fn a_finished_edit_carries_its_diff_and_a_new_file_has_no_old_text() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "old").unwrap();
        let mut tools = Tools::new(dir.path().to_path_buf());
        let start = |id: &str, path: &str| ToolStart {
            id: id.into(),
            name: "write_file".into(),
            arguments: serde_json::json!({ "path": path }),
        };
        let end = |id: &str| ToolEnd {
            id: id.into(),
            // The label the agent loop gives a finished call.
            outcome: "done".into(),
            preview: "ok".into(),
        };
        tools.started(start("e1", "a.txt"));
        std::fs::write(dir.path().join("a.txt"), "new").unwrap();
        let diff_of = |update: SessionUpdate| match update {
            SessionUpdate::ToolCallUpdate(u) => match u.fields.content.unwrap().remove(0) {
                ToolCallContent::Diff(d) => d,
                other => panic!("not a diff: {other:?}"),
            },
            other => panic!("not an update: {other:?}"),
        };
        let d = diff_of(tools.ended(end("e1")));
        assert_eq!(d.old_text.as_deref(), Some("old"));
        assert_eq!(d.new_text, "new");

        tools.started(start("e2", "fresh.txt"));
        std::fs::write(dir.path().join("fresh.txt"), "made").unwrap();
        let d = diff_of(tools.ended(end("e2")));
        assert_eq!(d.old_text, None);
        assert_eq!(d.new_text, "made");
    }
}
