//! What the engine sends during a turn, as ACP updates (M-7a). Pure: it
//! takes engine messages and gives back what to tell the editor, so it is
//! tested with real engine messages and no connection.

use xencode_tui_rs::engine::proto::{ApprovalView, EngineMsg};

/// Something to tell the editor.
#[derive(Debug, Clone, PartialEq)]
pub enum Out {
    /// Streamed answer text.
    Text(String),
    /// An approval prompt to ask the editor about.
    Permission(ApprovalView),
    /// A question from the agent (`ask_user`), with its id.
    Question(u64, String),
    /// The turn is over.
    Ended(Ending),
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
}

impl Turn {
    pub fn new() -> Turn {
        Turn::default()
    }

    pub fn feed(&mut self, msg: &EngineMsg) -> Vec<Out> {
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
    use xencode_tui_rs::engine::proto::EngineMsg;

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
