//! The messages windows and the engine exchange: one JSON object per line.
//!
//! Each side opens with `hello`, carrying `PROTOCOL_VERSION`; a mismatch is
//! answered with an error naming both versions. Message names are the ones
//! the design gives (`docs/superpowers/specs/2026-10-09-engine-process-design.md`, §4).

use serde::{Deserialize, Serialize};

/// The protocol version this build speaks.
pub const PROTOCOL_VERSION: u32 = 1;

/// What a window asks the engine to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMsg {
    Hello {
        version: u32,
        client: String,
    },
    SubmitChat {
        prompt: String,
    },
    EnqueueTask {
        text: String,
    },
    AnswerQuestion {
        id: u64,
        text: String,
    },
    AnswerApproval {
        id: u64,
        answer: WireAnswer,
    },
    Stop {
        target: StopTarget,
    },
    Review {
        decision: ReviewDecision,
    },
    SetModel {
        name: String,
    },
    /// A line a window-side command added to the transcript, so every
    /// window's transcript stays the same.
    Note {
        role: String,
        content: String,
    },
    Goodbye,
}

/// An answer to an approval prompt, as it travels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireAnswer {
    Allow,
    AllowForSession,
    Deny,
}

impl From<WireAnswer> for crate::agent_tools::ApprovalAnswer {
    fn from(answer: WireAnswer) -> Self {
        match answer {
            WireAnswer::Allow => crate::agent_tools::ApprovalAnswer::Approved,
            WireAnswer::AllowForSession => crate::agent_tools::ApprovalAnswer::ApprovedForSession,
            WireAnswer::Deny => crate::agent_tools::ApprovalAnswer::Denied,
        }
    }
}

impl From<crate::agent_tools::ApprovalAnswer> for WireAnswer {
    fn from(answer: crate::agent_tools::ApprovalAnswer) -> Self {
        match answer {
            crate::agent_tools::ApprovalAnswer::Approved => WireAnswer::Allow,
            crate::agent_tools::ApprovalAnswer::ApprovedForSession => WireAnswer::AllowForSession,
            crate::agent_tools::ApprovalAnswer::Denied => WireAnswer::Deny,
        }
    }
}

/// Which run a stop is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopTarget {
    Chat,
    Bytebot,
}

/// The person's decision on a ByteBot task waiting for review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewDecision {
    Accept,
    Undo,
}

/// An approval prompt as a window shows it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalView {
    pub id: u64,
    pub tool: String,
    /// The tool's class in words, as the prompt names it.
    pub class: String,
    pub summary: String,
    pub preview: String,
}

/// What the engine tells a window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineMsg {
    Hello {
        version: u32,
        session_id: String,
    },
    /// The agent state a window draws: all of it when the window connects,
    /// then only what changed.
    View {
        view: Box<super::view::View>,
    },
    /// One message from the agent loop, as `App::apply_token` takes it.
    Event {
        token: String,
    },
    ApprovalRequested {
        approval: ApprovalView,
    },
    ApprovalResolved {
        id: u64,
        answer: WireAnswer,
        by: String,
    },
    QuestionAsked {
        id: u64,
        text: String,
    },
    QuestionAnswered {
        id: u64,
        by: String,
    },
    Error {
        message: String,
    },
}

/// One message as one line of JSON, without the line ending.
pub fn encode(msg: &impl Serialize) -> String {
    // Serialising these plain enums cannot fail; an empty object is never
    // sent in practice and would be refused by the other side in words.
    serde_json::to_string(msg).unwrap_or_else(|_| "{}".to_string())
}

pub fn decode_client(line: &str) -> Result<ClientMsg, String> {
    serde_json::from_str(line).map_err(|e| format!("not a message xencode understands: {e}"))
}

pub fn decode_engine(line: &str) -> Result<EngineMsg, String> {
    serde_json::from_str(line).map_err(|e| format!("not a message xencode understands: {e}"))
}

/// What to say to a window speaking another protocol version.
pub fn version_mismatch(theirs: u32) -> String {
    format!(
        "this xencode speaks protocol version {PROTOCOL_VERSION} and the other side speaks \
         version {theirs}; update the older one"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client_samples() -> Vec<ClientMsg> {
        vec![
            ClientMsg::Hello {
                version: PROTOCOL_VERSION,
                client: "terminal".into(),
            },
            ClientMsg::SubmitChat {
                prompt: "fix the bug\nplease".into(),
            },
            ClientMsg::EnqueueTask {
                text: "write a note".into(),
            },
            ClientMsg::AnswerQuestion {
                id: 3,
                text: "postgres".into(),
            },
            ClientMsg::AnswerApproval {
                id: 4,
                answer: WireAnswer::AllowForSession,
            },
            ClientMsg::Stop {
                target: StopTarget::Bytebot,
            },
            ClientMsg::Review {
                decision: ReviewDecision::Undo,
            },
            ClientMsg::SetModel {
                name: "llamacpp:qwen3-4b".into(),
            },
            ClientMsg::Note {
                role: "system".into(),
                content: "Theme: dark".into(),
            },
            ClientMsg::Goodbye,
        ]
    }

    fn engine_samples() -> Vec<EngineMsg> {
        let approval = ApprovalView {
            id: 4,
            tool: "write_file".into(),
            class: "file edit".into(),
            summary: "write_file a.rs".into(),
            preview: "+hello".into(),
        };
        vec![
            EngineMsg::Hello {
                version: PROTOCOL_VERSION,
                session_id: "session_1".into(),
            },
            EngineMsg::View {
                view: Box::new(super::super::view::View {
                    messages_from: 2,
                    generating: Some(true),
                    approvals: Some(vec![approval.clone()]),
                    question: Some(Some((3, "Which port?".into()))),
                    ..Default::default()
                }),
            },
            EngineMsg::Event {
                token: "[TOOL]→ read_file a.rs".into(),
            },
            EngineMsg::ApprovalRequested { approval },
            EngineMsg::ApprovalResolved {
                id: 4,
                answer: WireAnswer::Deny,
                by: "terminal".into(),
            },
            EngineMsg::QuestionAsked {
                id: 3,
                text: "Which port?".into(),
            },
            EngineMsg::QuestionAnswered {
                id: 3,
                by: "desktop".into(),
            },
            EngineMsg::Error {
                message: "no".into(),
            },
        ]
    }

    #[test]
    fn every_message_round_trips_on_one_line() {
        for msg in client_samples() {
            let line = encode(&msg);
            assert!(!line.contains('\n'), "{line}");
            assert_eq!(decode_client(&line).unwrap(), msg);
        }
        for msg in engine_samples() {
            let line = encode(&msg);
            assert!(!line.contains('\n'), "{line}");
            assert_eq!(decode_engine(&line).unwrap(), msg);
        }
    }

    #[test]
    fn message_names_are_the_ones_the_design_gives() {
        let line = encode(&ClientMsg::SubmitChat { prompt: "x".into() });
        assert!(line.contains("\"type\":\"submit_chat\""), "{line}");
        let line = encode(&EngineMsg::ApprovalRequested {
            approval: ApprovalView {
                id: 1,
                tool: String::new(),
                class: String::new(),
                summary: String::new(),
                preview: String::new(),
            },
        });
        assert!(line.contains("\"type\":\"approval_requested\""), "{line}");
    }

    #[test]
    fn unknown_messages_are_refused_in_words() {
        let err = decode_client(r#"{"type":"launch_rockets"}"#).unwrap_err();
        assert!(err.contains("not a message xencode understands"), "{err}");
        assert!(decode_engine("not json").is_err());
    }

    #[test]
    fn a_version_mismatch_names_both_versions() {
        let text = version_mismatch(2);
        assert!(text.contains('1') && text.contains('2'), "{text}");
    }
}
