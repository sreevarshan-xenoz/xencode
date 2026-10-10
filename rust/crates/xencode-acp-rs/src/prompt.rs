//! A prompt from the editor, run as one turn on the session's engine: its
//! text (M-7a), its tool calls and approval prompts (M-7b).

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use agent_client_protocol::schema::v1::{
    Content, ContentBlock, ContentChunk, Diff, EmbeddedResourceResource, PermissionOption,
    PermissionOptionKind, PromptRequest, PromptResponse, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse, SessionId, SessionNotification,
    SessionUpdate, StopReason, TextContent, ToolCall, ToolCallContent, ToolCallLocation,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::{Client, ConnectionTo, Responder};
use tokio::sync::Notify;
use xencode_tui_rs::engine::link::{EngineLink, LinkEvent};
use xencode_tui_rs::engine::proto::{ApprovalView, ClientMsg, EngineMsg, StopTarget, WireAnswer};

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
    let Some(state) = sessions.get(&id) else {
        return responder.respond_with_error(error(crate::INVALID, format!("no session {id}")));
    };
    let (link, cancel, project) = {
        let mut s = state.lock().await;
        if s.busy {
            return responder.respond_with_error(error(
                crate::INVALID,
                "a turn is already running in this session; wait for it or cancel it",
            ));
        }
        let link = match s.link.take() {
            Some(link) => link,
            None => match session::connect(&s.project).await {
                Ok(link) => link,
                Err(why) => {
                    return responder.respond_with_error(error(
                        INTERNAL,
                        format!("cannot reach the engine: {why}"),
                    ))
                }
            },
        };
        s.busy = true;
        s.cancel = Arc::new(Notify::new());
        (link, Arc::clone(&s.cancel), s.project.clone())
    };
    let text = prompt_text(&req.prompt);
    if !link.send(&ClientMsg::SubmitChat { prompt: text }) {
        finish(&state, None).await;
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
    };
    connection.spawn(async move {
        let (link, answer) = run.run().await;
        finish(&state, link).await;
        match answer {
            Ok(Ending::Done) => responder.respond(PromptResponse::new(StopReason::EndTurn)),
            Ok(Ending::Stopped) => responder.respond(PromptResponse::new(StopReason::Cancelled)),
            Ok(Ending::Failed(why)) | Err(why) => {
                responder.respond_with_error(error(INTERNAL, why))
            }
        }
    })
}

/// The editor's answer to a permission request, when it comes.
type PermissionAnswer = Pin<
    Box<
        dyn Future<Output = Result<RequestPermissionResponse, agent_client_protocol::Error>> + Send,
    >,
>;

/// One running turn: reads the engine, tells the editor, and carries the
/// editor's answers back.
struct TurnRun {
    connection: ConnectionTo<Client>,
    session_id: SessionId,
    link: EngineLink,
    cancel: Arc<Notify>,
    tools: Tools,
    /// The approval being asked about, and the editor's answer to come.
    asking: Option<(u64, PermissionAnswer)>,
    /// Approvals that came while another was being asked about.
    waiting: VecDeque<ApprovalView>,
}

impl TurnRun {
    /// Run to the turn's end. Gives back the link (`None` once the engine
    /// was lost) and how the turn ended, or why it could not finish.
    async fn run(mut self) -> (Option<EngineLink>, Result<Ending, String>) {
        let mut turn = Turn::new();
        loop {
            let event = tokio::select! {
                event = self.link.next() => event,
                _ = self.cancel.notified() => {
                    self.stop();
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
                    Out::Text(text) => self.notify(SessionUpdate::AgentMessageChunk(
                        ContentChunk::new(ContentBlock::Text(TextContent::new(text))),
                    )),
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
                    // Questions: M-7d.
                    Out::Question(..) => Ok(()),
                    Out::Ended(ending) => return (Some(self.link), Ok(ending)),
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
        asking: &mut Option<(u64, PermissionAnswer)>,
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

    /// The editor cancelled. An open approval is answered no first, so the
    /// agent loop is not left waiting on it, and then the turn is stopped.
    fn stop(&mut self) {
        if let Some((id, _)) = self.asking.take() {
            self.link.send(&ClientMsg::AnswerApproval {
                id,
                answer: WireAnswer::Deny,
            });
        }
        self.waiting.clear();
        self.link.send(&ClientMsg::Stop {
            target: StopTarget::Chat,
        });
    }

    fn permission(&mut self, approval: ApprovalView) {
        if self.asking.is_none() {
            let answer = self.ask(&approval);
            self.asking = Some((approval.id, answer));
        } else {
            self.waiting.push_back(approval);
        }
    }

    fn answered(
        &mut self,
        answer: Result<RequestPermissionResponse, agent_client_protocol::Error>,
    ) {
        if let Some((id, _)) = self.asking.take() {
            self.link.send(&ClientMsg::AnswerApproval {
                id,
                answer: wire_answer(answer),
            });
        }
        self.ask_next();
    }

    /// Another window answered approval `id` first; its answer counts and
    /// the editor's, if it comes, is not used.
    fn resolved_elsewhere(&mut self, id: u64) {
        if self.asking.as_ref().is_some_and(|(open, _)| *open == id) {
            self.asking = None;
            self.ask_next();
        } else {
            self.waiting.retain(|a| a.id != id);
        }
    }

    fn ask_next(&mut self) {
        if let Some(next) = self.waiting.pop_front() {
            let answer = self.ask(&next);
            self.asking = Some((next.id, answer));
        }
    }

    /// Ask the editor about one approval prompt.
    fn ask(&self, approval: &ApprovalView) -> PermissionAnswer {
        let call = ToolCallUpdate::new(
            format!("approval-{}", approval.id),
            ToolCallUpdateFields::new()
                .title(approval.summary.clone())
                .kind(kinds::kind_of(&approval.tool)),
        );
        let options = vec![
            PermissionOption::new("allow", "Allow once", PermissionOptionKind::AllowOnce),
            PermissionOption::new(
                "allow-session",
                "Always allow this session",
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new("deny", "Reject", PermissionOptionKind::RejectOnce),
        ];
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

async fn finish(state: &tokio::sync::Mutex<Session>, link: Option<EngineLink>) {
    let mut s = state.lock().await;
    s.busy = false;
    s.link = link;
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1::SelectedPermissionOutcome;

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
