//! A prompt from the editor, run as one turn on the session's engine (M-7a).

use std::sync::Arc;

use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, EmbeddedResourceResource, PromptRequest, PromptResponse, SessionId,
    SessionNotification, SessionUpdate, StopReason, TextContent,
};
use agent_client_protocol::{Client, ConnectionTo, Responder};
use tokio::sync::Notify;
use xencode_tui_rs::engine::link::LinkEvent;
use xencode_tui_rs::engine::proto::{ClientMsg, StopTarget};

use crate::session::{self, Session};
use crate::turn::{Ending, Out, Turn};

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
    let (mut link, cancel) = {
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
        (link, Arc::clone(&s.cancel))
    };
    let text = prompt_text(&req.prompt);
    if !link.send(&ClientMsg::SubmitChat { prompt: text }) {
        finish(&state, None).await;
        return responder.respond_with_error(error(
            INTERNAL,
            "the engine was lost before the prompt reached it",
        ));
    }
    let session_id = req.session_id.clone();
    connection.clone().spawn(async move {
        let mut turn = Turn::new();
        let answer = loop {
            let event = tokio::select! {
                event = link.next() => event,
                _ = cancel.notified() => {
                    link.send(&ClientMsg::Stop { target: StopTarget::Chat });
                    continue;
                }
            };
            let msg = match event {
                LinkEvent::Msg(msg) => msg,
                LinkEvent::Lost(why) => {
                    finish(&state, None).await;
                    return responder.respond_with_error(error(
                        INTERNAL,
                        format!("the engine was lost: {why}"),
                    ));
                }
            };
            let mut ended = None;
            for out in turn.feed(&msg) {
                match out {
                    Out::Text(text) => say(&connection, &session_id, text)?,
                    Out::Ended(ending) => ended = Some(ending),
                    // Approvals and questions: M-7b and M-7d.
                    Out::Permission(_) | Out::Question(..) => {}
                }
            }
            if let Some(ending) = ended {
                break ending;
            }
        };
        finish(&state, Some(link)).await;
        match answer {
            Ending::Done => responder.respond(PromptResponse::new(StopReason::EndTurn)),
            Ending::Stopped => responder.respond(PromptResponse::new(StopReason::Cancelled)),
            Ending::Failed(why) => responder.respond_with_error(error(INTERNAL, why)),
        }
    })
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
    link: Option<xencode_tui_rs::engine::link::EngineLink>,
) {
    let mut s = state.lock().await;
    s.busy = false;
    s.link = link;
}

fn say(
    connection: &ConnectionTo<Client>,
    session_id: &SessionId,
    text: String,
) -> Result<(), agent_client_protocol::Error> {
    connection.send_notification(SessionNotification::new(
        session_id.clone(),
        SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(TextContent::new(
            text,
        )))),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_blocks_are_joined_and_links_named() {
        let blocks = vec![
            ContentBlock::Text(TextContent::new("fix this")),
            ContentBlock::Text(TextContent::new("please")),
        ];
        assert_eq!(prompt_text(&blocks), "fix this\n\nplease");
    }
}
