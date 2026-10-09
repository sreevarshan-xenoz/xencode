//! The engine (EN-1): every agent action a window asks for, and everything
//! the agent loop reports back, as typed messages. In this stage the engine
//! runs inside the terminal app; `EN-2` carries the same messages over a
//! local socket to an engine in its own process.

pub mod proto;

use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use crate::app::App;
use proto::{ClientMsg, EngineMsg, ReviewDecision, StopTarget, PROTOCOL_VERSION};

/// Carry out one action a window asked for. `by` names the window, so the
/// others can be told who answered. Returns what the engine says back.
pub fn handle(
    app: &mut App,
    msg: ClientMsg,
    tx: &mpsc::UnboundedSender<String>,
    by: &str,
) -> Vec<EngineMsg> {
    match msg {
        ClientMsg::Hello { version, .. } => {
            if version == PROTOCOL_VERSION {
                vec![EngineMsg::Hello {
                    version: PROTOCOL_VERSION,
                    session_id: app.memory.current_session().cloned().unwrap_or_default(),
                }]
            } else {
                vec![EngineMsg::Error {
                    message: proto::version_mismatch(version),
                }]
            }
        }
        ClientMsg::SubmitChat { prompt } => {
            app.dispatch_prompt(prompt, tx.clone());
            Vec::new()
        }
        ClientMsg::EnqueueTask { text } => {
            app.bytebot_enqueue(&text, tx.clone());
            Vec::new()
        }
        ClientMsg::AnswerQuestion { id, text } => {
            if app.question_id != Some(id) {
                return vec![EngineMsg::Error {
                    message: "that question is no longer waiting".to_string(),
                }];
            }
            if text.trim().is_empty() {
                return vec![EngineMsg::Error {
                    message: "an empty answer was not sent; the question is still waiting"
                        .to_string(),
                }];
            }
            app.bytebot_command = text;
            app.bytebot_answer();
            vec![EngineMsg::QuestionAnswered {
                id,
                by: by.to_string(),
            }]
        }
        ClientMsg::AnswerApproval { id, answer } => {
            if app.approval_ids.front() != Some(&id) {
                return vec![EngineMsg::Error {
                    message: "that approval is no longer waiting".to_string(),
                }];
            }
            app.resolve_approval(answer.into());
            vec![EngineMsg::ApprovalResolved {
                id,
                answer,
                by: by.to_string(),
            }]
        }
        ClientMsg::Stop { target } => {
            match target {
                StopTarget::Chat => {
                    if app.is_generating {
                        if let Some(stop) = app.turn_stop.as_ref() {
                            stop.store(true, Ordering::Relaxed);
                        }
                    }
                }
                StopTarget::Bytebot => {
                    if app.bytebot_running {
                        if let Some(stop) = app.bytebot_stop.as_ref() {
                            stop.store(true, Ordering::Relaxed);
                        }
                    }
                    // A task waiting on a question would never see the stop.
                    app.bytebot_withdraw_question();
                }
            }
            Vec::new()
        }
        ClientMsg::Review { decision } => {
            let done = match decision {
                ReviewDecision::Accept => {
                    if app.bytebot_reviewing().is_none() {
                        Err("no ByteBot task is waiting for review".to_string())
                    } else {
                        app.bytebot_accept(tx.clone());
                        Ok(())
                    }
                }
                ReviewDecision::Undo => app.bytebot_undo(tx.clone()).map(|_| ()),
            };
            match done {
                Ok(()) => Vec::new(),
                Err(message) => vec![EngineMsg::Error { message }],
            }
        }
        ClientMsg::SetModel { name } => {
            app.set_model(&name, tx.clone());
            Vec::new()
        }
        ClientMsg::Goodbye => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::proto::*;
    use super::*;
    use crate::app::App;
    use tokio::sync::{mpsc, oneshot};

    fn engine_app() -> App<'static> {
        let mut app = App::for_tests();
        // Nothing listens on port 9: runs fail in the background, and only
        // what the engine does with each message is checked.
        app.config.default_model = "llamacpp:none".into();
        app.config.llama_cpp_url = "http://127.0.0.1:9".into();
        app
    }

    fn request(summary: &str) -> crate::agent_tools::ApprovalRequest {
        crate::agent_tools::ApprovalRequest {
            tool: "write_file".into(),
            class: crate::agent_tools::ToolClass::Edit,
            summary: summary.into(),
            preview: String::new(),
            draft: Default::default(),
        }
    }

    #[tokio::test]
    async fn submit_chat_and_enqueue_task_start_their_runs() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        let out = handle(
            &mut app,
            ClientMsg::SubmitChat {
                prompt: "hello".into(),
            },
            &tx,
            "t",
        );
        assert!(out.is_empty());
        assert!(app.is_generating);
        handle(
            &mut app,
            ClientMsg::EnqueueTask {
                text: "a job".into(),
            },
            &tx,
            "t",
        );
        assert!(app.bytebot_running);
    }

    #[tokio::test]
    async fn an_approval_is_answered_by_its_id() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        let (first, first_answer) = oneshot::channel();
        let (second, _second_answer) = oneshot::channel();
        app.approval_tx.send((request("first"), first)).unwrap();
        app.approval_tx.send((request("second"), second)).unwrap();
        app.drain_agent_channels();
        let ids: Vec<u64> = app.approval_ids.iter().copied().collect();
        assert_eq!(ids.len(), 2);
        let out = handle(
            &mut app,
            ClientMsg::AnswerApproval {
                id: ids[0],
                answer: WireAnswer::Allow,
            },
            &tx,
            "desktop",
        );
        assert_eq!(
            out,
            vec![EngineMsg::ApprovalResolved {
                id: ids[0],
                answer: WireAnswer::Allow,
                by: "desktop".into()
            }]
        );
        assert_eq!(
            first_answer.await.unwrap(),
            crate::agent_tools::ApprovalAnswer::Approved
        );
        assert_eq!(
            app.pending_approval().map(|r| r.summary.as_str()),
            Some("second")
        );
    }

    #[tokio::test]
    async fn a_stale_approval_id_answers_nothing() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        let (first, _a) = oneshot::channel();
        let (second, _b) = oneshot::channel();
        app.approval_tx.send((request("first"), first)).unwrap();
        app.approval_tx.send((request("second"), second)).unwrap();
        app.drain_agent_channels();
        let id = *app.approval_ids.front().unwrap();
        let answer = ClientMsg::AnswerApproval {
            id,
            answer: WireAnswer::Deny,
        };
        handle(&mut app, answer.clone(), &tx, "terminal");
        let out = handle(&mut app, answer, &tx, "desktop");
        assert!(
            matches!(out.as_slice(), [EngineMsg::Error { .. }]),
            "{out:?}"
        );
        assert_eq!(
            app.pending_approval().map(|r| r.summary.as_str()),
            Some("second"),
            "the second request was not answered by the first one's id"
        );
    }

    #[tokio::test]
    async fn a_question_is_answered_by_its_id() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        handle(
            &mut app,
            ClientMsg::EnqueueTask {
                text: "pick a db".into(),
            },
            &tx,
            "t",
        );
        let (reply, answer) = oneshot::channel();
        app.ask_tx.send(("Which database?".into(), reply)).unwrap();
        app.drain_agent_channels();
        let id = app.question_id.expect("the question has an id");
        let out = handle(
            &mut app,
            ClientMsg::AnswerQuestion {
                id,
                text: "postgres".into(),
            },
            &tx,
            "desktop",
        );
        assert_eq!(
            out,
            vec![EngineMsg::QuestionAnswered {
                id,
                by: "desktop".into()
            }]
        );
        assert_eq!(answer.await.unwrap(), "postgres");
        assert_eq!(app.question_id, None);
    }

    #[tokio::test]
    async fn an_empty_answer_is_refused_and_the_question_keeps_waiting() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        handle(
            &mut app,
            ClientMsg::EnqueueTask {
                text: "pick a db".into(),
            },
            &tx,
            "t",
        );
        let (reply, _answer) = oneshot::channel();
        app.ask_tx.send(("Which database?".into(), reply)).unwrap();
        app.drain_agent_channels();
        let id = app.question_id.unwrap();
        let out = handle(
            &mut app,
            ClientMsg::AnswerQuestion {
                id,
                text: "  ".into(),
            },
            &tx,
            "t",
        );
        assert!(
            matches!(out.as_slice(), [EngineMsg::Error { .. }]),
            "{out:?}"
        );
        assert_eq!(app.question_id, Some(id));
    }

    #[tokio::test]
    async fn an_answer_to_a_withdrawn_question_is_refused() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        handle(
            &mut app,
            ClientMsg::EnqueueTask {
                text: "pick a db".into(),
            },
            &tx,
            "t",
        );
        let (reply, answer) = oneshot::channel();
        app.ask_tx.send(("Which database?".into(), reply)).unwrap();
        app.drain_agent_channels();
        let id = app.question_id.unwrap();
        handle(
            &mut app,
            ClientMsg::Stop {
                target: StopTarget::Bytebot,
            },
            &tx,
            "t",
        );
        assert!(answer.await.is_err(), "the stop withdrew the question");
        let out = handle(
            &mut app,
            ClientMsg::AnswerQuestion {
                id,
                text: "late".into(),
            },
            &tx,
            "t",
        );
        assert!(
            matches!(out.as_slice(), [EngineMsg::Error { .. }]),
            "{out:?}"
        );
    }

    #[tokio::test]
    async fn stop_chat_sets_the_chat_turns_stop() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        handle(
            &mut app,
            ClientMsg::SubmitChat {
                prompt: "hello".into(),
            },
            &tx,
            "t",
        );
        handle(
            &mut app,
            ClientMsg::Stop {
                target: StopTarget::Chat,
            },
            &tx,
            "t",
        );
        let stop = app.turn_stop.as_ref().expect("a running turn has a stop");
        assert!(stop.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[tokio::test]
    async fn review_with_nothing_to_review_is_an_error_in_words() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        let out = handle(
            &mut app,
            ClientMsg::Review {
                decision: ReviewDecision::Undo,
            },
            &tx,
            "t",
        );
        match out.as_slice() {
            [EngineMsg::Error { message }] => {
                assert!(message.contains("waiting for review"), "{message}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn set_model_changes_the_model() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        handle(
            &mut app,
            ClientMsg::SetModel {
                name: "qwen3:4b".into(),
            },
            &tx,
            "t",
        );
        assert_eq!(app.config.default_model, "qwen3:4b");
    }

    #[tokio::test]
    async fn hello_checks_the_version() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        let out = handle(
            &mut app,
            ClientMsg::Hello {
                version: PROTOCOL_VERSION,
                client: "terminal".into(),
            },
            &tx,
            "terminal",
        );
        assert!(
            matches!(
                out.as_slice(),
                [EngineMsg::Hello {
                    version: PROTOCOL_VERSION,
                    ..
                }]
            ),
            "{out:?}"
        );
        let out = handle(
            &mut app,
            ClientMsg::Hello {
                version: 99,
                client: "x".into(),
            },
            &tx,
            "x",
        );
        match out.as_slice() {
            [EngineMsg::Error { message }] => assert!(message.contains("99"), "{message}"),
            other => panic!("{other:?}"),
        }
    }
}
