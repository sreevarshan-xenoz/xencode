//! What the engine sends during a turn, as ACP updates (M-7a). Pure: it
//! takes engine messages and gives back what to tell the editor, so it is
//! tested with real engine messages and no connection.

use xencode_tui_rs::bytebot_tasks::TaskState;
use xencode_tui_rs::engine::proto::{ApprovalView, EngineMsg};

/// Something to tell the editor.
#[derive(Debug, Clone, PartialEq)]
pub enum Out {
    /// Streamed answer text.
    Text(String),
    /// A tool call began.
    ToolStart(ToolStart),
    /// A tool call ended.
    ToolEnd(ToolEnd),
    /// An approval prompt to ask the editor about.
    Permission(ApprovalView),
    /// A question from the agent (`ask_user`), with its id.
    Question(u64, String),
    /// ByteBot's steps so far: what was called and how it ended (M-7d).
    Plan(Vec<(String, String)>),
    /// The ByteBot task changed files and waits for keep or undo.
    Review,
    /// The turn is over.
    Ended(Ending),
}

/// A tool call as the agent loop began it (`[TOOLCALL]`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct ToolStart {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// How a tool call ended (`[TOOLEND]`): its outcome label and the start of
/// its result.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct ToolEnd {
    pub id: String,
    pub outcome: String,
    #[serde(default)]
    pub preview: String,
}

/// How a turn ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Ending {
    Done,
    Stopped,
    Failed(String),
}

/// One turn's worth of engine messages, read in order.
#[derive(Debug, Default)]
pub struct Turn {
    stopped: bool,
    /// For a ByteBot turn, the task being followed.
    bytebot: Option<Followed>,
}

/// The ByteBot task a turn follows. The task list may still hold an older,
/// finished task with the same words, so the turn ends only after it has
/// seen this one waiting or running.
#[derive(Debug, Default)]
struct Followed {
    text: String,
    started: bool,
    review_asked: bool,
}

impl Turn {
    pub fn new() -> Turn {
        Turn::default()
    }

    /// The words of the ByteBot task this turn follows, if it does.
    pub fn follows(&self) -> Option<&str> {
        self.bytebot.as_ref().map(|f| f.text.as_str())
    }

    /// A turn that follows the ByteBot task with these words to its end.
    pub fn bytebot(text: &str) -> Turn {
        Turn {
            bytebot: Some(Followed {
                text: text.to_string(),
                ..Followed::default()
            }),
            ..Turn::default()
        }
    }

    pub fn feed(&mut self, msg: &EngineMsg) -> Vec<Out> {
        if let (Some(followed), EngineMsg::View { view }) = (self.bytebot.as_mut(), msg) {
            let mut out = Vec::new();
            if let Some(steps) = &view.bytebot_steps {
                out.push(Out::Plan(steps.clone()));
            }
            if let Some(tasks) = &view.tasks {
                if let Some(ended) = followed.read(tasks) {
                    out.push(ended);
                }
            }
            return out;
        }
        match msg {
            EngineMsg::Event { token } => self.token(token),
            EngineMsg::ApprovalRequested { approval } => vec![Out::Permission(approval.clone())],
            EngineMsg::QuestionAsked { id, text } => vec![Out::Question(*id, text.clone())],
            EngineMsg::Error { message } => vec![Out::Ended(Ending::Failed(message.clone()))],
            _ => Vec::new(),
        }
    }

    fn token(&mut self, token: &str) -> Vec<Out> {
        if token == "[DONE]" {
            let ending = if self.stopped {
                Ending::Stopped
            } else {
                Ending::Done
            };
            return vec![Out::Ended(ending)];
        }
        if token == "[STOPPED]" {
            self.stopped = true;
            return Vec::new();
        }
        if let Some(body) = token.strip_prefix("[BYTEBOT]") {
            // ByteBot's narration and its errors are what the person reads;
            // its calls and their ends arrive as steps in the view.
            return match body
                .strip_prefix("log:")
                .or_else(|| body.strip_prefix("err:"))
            {
                Some(text) => vec![Out::Text(format!("{text}\n"))],
                None => Vec::new(),
            };
        }
        if let Some(body) = token.strip_prefix("[TOOLCALL]") {
            return serde_json::from_str(body)
                .map(Out::ToolStart)
                .into_iter()
                .collect();
        }
        if let Some(body) = token.strip_prefix("[TOOLEND]") {
            return serde_json::from_str(body)
                .map(Out::ToolEnd)
                .into_iter()
                .collect();
        }
        if let Some(why) = token.strip_prefix("[TURNERR]") {
            // The model call failed; the terminal shows it in the chat, and
            // so does the editor. The turn still ends with `[DONE]`.
            return vec![Out::Text(why.to_string())];
        }
        if is_tagged(token) {
            // A line for the terminal's panels, not answer text.
            return Vec::new();
        }
        vec![Out::Text(token.to_string())]
    }
}

impl Followed {
    /// What the followed task's state now says to do, if anything.
    fn read(&mut self, tasks: &[xencode_tui_rs::engine::view::ViewTask]) -> Option<Out> {
        let task = tasks
            .iter()
            .rev()
            .find(|t| t.this_session && t.task.text == self.text)?;
        match task.task.state {
            TaskState::Pending | TaskState::Running | TaskState::NeedsHelp => {
                self.started = true;
                None
            }
            TaskState::NeedsReview => {
                self.started = true;
                if self.review_asked {
                    None
                } else {
                    self.review_asked = true;
                    Some(Out::Review)
                }
            }
            TaskState::Completed | TaskState::Failed if self.started => {
                Some(Out::Ended(Ending::Done))
            }
            // Undo after a review marks the task cancelled: the person
            // chose it, so the turn ended as it should.
            TaskState::Cancelled if self.review_asked => Some(Out::Ended(Ending::Done)),
            TaskState::Cancelled if self.started => Some(Out::Ended(Ending::Stopped)),
            _ => None,
        }
    }
}

/// Whether `token` starts with an engine tag such as `[TOOL]` or
/// `[LLAMACPP_MSG]`: capital letters and underscores in brackets.
fn is_tagged(token: &str) -> bool {
    let Some(rest) = token.strip_prefix('[') else {
        return false;
    };
    match rest.find(']') {
        Some(end) if end > 0 => rest[..end]
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_'),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xencode_tui_rs::bytebot_tasks::{ByteBotTask, TaskState};
    use xencode_tui_rs::engine::proto::EngineMsg;
    use xencode_tui_rs::engine::view::{View, ViewTask};

    fn ev(token: &str) -> EngineMsg {
        EngineMsg::Event {
            token: token.to_string(),
        }
    }

    fn run(turn: &mut Turn, msgs: &[EngineMsg]) -> Vec<Out> {
        msgs.iter().flat_map(|m| turn.feed(m)).collect()
    }

    #[test]
    fn streamed_text_then_done_is_text_then_the_end() {
        let mut turn = Turn::new();
        let out = run(&mut turn, &[ev("Hel"), ev("lo"), ev("[DONE]")]);
        assert_eq!(
            out,
            vec![
                Out::Text("Hel".into()),
                Out::Text("lo".into()),
                Out::Ended(Ending::Done)
            ]
        );
    }

    #[test]
    fn a_stop_ends_the_turn_as_stopped() {
        let mut turn = Turn::new();
        let out = run(&mut turn, &[ev("par"), ev("[STOPPED]"), ev("[DONE]")]);
        assert_eq!(out.last(), Some(&Out::Ended(Ending::Stopped)));
    }

    #[test]
    fn tagged_lines_are_not_text() {
        let mut turn = Turn::new();
        for token in [
            "[TOOL]→ read_file({})",
            "[TIMINGS]{}",
            "[CTXSTATS]3",
            "[LLAMACPP_MSG]x",
        ] {
            assert!(turn.feed(&ev(token)).is_empty(), "{token}");
        }
    }

    #[test]
    fn text_that_starts_with_a_bracket_is_still_text() {
        let mut turn = Turn::new();
        assert_eq!(
            turn.feed(&ev("[1, 2, 3] is a list")),
            vec![Out::Text("[1, 2, 3] is a list".into())]
        );
    }

    #[test]
    fn a_failed_model_call_is_said_as_text() {
        let mut turn = Turn::new();
        let out = run(
            &mut turn,
            &[
                ev("[TURNERR]llamacpp:none could not answer: connection refused"),
                ev("[DONE]"),
            ],
        );
        assert_eq!(
            out,
            vec![
                Out::Text("llamacpp:none could not answer: connection refused".into()),
                Out::Ended(Ending::Done)
            ]
        );
    }

    #[test]
    fn a_tool_call_is_seen_start_and_end() {
        let mut turn = Turn::new();
        let out = run(
            &mut turn,
            &[
                ev(
                    r#"[TOOLCALL]{"id":"c1","name":"write_file","arguments":{"path":"n.txt","content":"hi"}}"#,
                ),
                ev(r#"[TOOLEND]{"id":"c1","outcome":"done","preview":"wrote n.txt"}"#),
            ],
        );
        assert_eq!(
            out,
            vec![
                Out::ToolStart(ToolStart {
                    id: "c1".into(),
                    name: "write_file".into(),
                    arguments: serde_json::json!({"path": "n.txt", "content": "hi"}),
                }),
                Out::ToolEnd(ToolEnd {
                    id: "c1".into(),
                    outcome: "done".into(),
                    preview: "wrote n.txt".into(),
                }),
            ]
        );
    }

    #[test]
    fn a_tool_token_that_does_not_parse_is_dropped() {
        let mut turn = Turn::new();
        assert!(turn.feed(&ev("[TOOLCALL]{not json")).is_empty());
    }

    fn view_with_task(text: &str, state: TaskState) -> EngineMsg {
        let mut task = ByteBotTask::new(text, "llamacpp:none");
        task.state = state;
        EngineMsg::View {
            view: Box::new(View {
                tasks: Some(vec![ViewTask {
                    task,
                    this_session: true,
                }]),
                ..View::default()
            }),
        }
    }

    #[test]
    fn a_bytebot_task_shows_its_steps_and_asks_for_review_then_ends() {
        let mut turn = Turn::bytebot("write a note");
        let steps = EngineMsg::View {
            view: Box::new(View {
                bytebot_steps: Some(vec![("write_file(n.txt)".into(), "running".into())]),
                ..View::default()
            }),
        };
        assert_eq!(
            turn.feed(&steps),
            vec![Out::Plan(vec![(
                "write_file(n.txt)".into(),
                "running".into()
            )])]
        );
        assert!(turn
            .feed(&view_with_task("write a note", TaskState::Running))
            .is_empty());
        assert_eq!(
            turn.feed(&ev("[BYTEBOT]log:The file is written.")),
            vec![Out::Text("The file is written.\n".into())]
        );
        assert_eq!(
            turn.feed(&view_with_task("write a note", TaskState::NeedsReview)),
            vec![Out::Review]
        );
        // Asked once, however many views repeat the state.
        assert!(turn
            .feed(&view_with_task("write a note", TaskState::NeedsReview))
            .is_empty());
        assert_eq!(
            turn.feed(&view_with_task("write a note", TaskState::Completed)),
            vec![Out::Ended(Ending::Done)]
        );
    }

    #[test]
    fn an_undone_review_is_a_normal_end() {
        let mut turn = Turn::bytebot("write a note");
        turn.feed(&view_with_task("write a note", TaskState::NeedsReview));
        // Undo marks the task cancelled; the person chose it, nothing stopped.
        assert_eq!(
            turn.feed(&view_with_task("write a note", TaskState::Cancelled)),
            vec![Out::Ended(Ending::Done)]
        );
    }

    #[test]
    fn an_older_task_with_the_same_words_does_not_end_the_new_one() {
        let mut turn = Turn::bytebot("write a note");
        // The finished one from before is in the list first.
        assert!(turn
            .feed(&view_with_task("write a note", TaskState::Completed))
            .is_empty());
        assert!(turn
            .feed(&view_with_task("write a note", TaskState::Pending))
            .is_empty());
        assert_eq!(
            turn.feed(&view_with_task("write a note", TaskState::Failed)),
            vec![Out::Ended(Ending::Done)]
        );
    }

    #[test]
    fn a_cancelled_bytebot_task_ends_as_stopped_and_its_error_is_said() {
        let mut turn = Turn::bytebot("write a note");
        assert_eq!(
            turn.feed(&ev("[BYTEBOT]err:the model server is unreachable")),
            vec![Out::Text("the model server is unreachable\n".into())]
        );
        turn.feed(&view_with_task("write a note", TaskState::Running));
        assert_eq!(
            turn.feed(&view_with_task("write a note", TaskState::Cancelled)),
            vec![Out::Ended(Ending::Stopped)]
        );
    }

    #[test]
    fn an_engine_error_fails_the_turn_in_its_words() {
        let mut turn = Turn::new();
        let out = turn.feed(&EngineMsg::Error {
            message: "model server unreachable".into(),
        });
        assert_eq!(
            out,
            vec![Out::Ended(Ending::Failed(
                "model server unreachable".into()
            ))]
        );
    }
}
