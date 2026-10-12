//! The engine (EN-1): every agent action a window asks for, and everything
//! the agent loop reports back, as typed messages. In this stage the engine
//! runs inside the terminal app; `EN-2` carries the same messages over a
//! local socket to an engine in its own process.

/// What the engine answers to an answer for a question that was answered
/// or withdrawn already. Windows that answer late recognise it by this text.
pub const QUESTION_GONE: &str = "that question is no longer waiting";

/// As [`QUESTION_GONE`], for an approval prompt.
pub const APPROVAL_GONE: &str = "that approval is no longer waiting";

pub mod address;
pub mod link;
pub mod proto;
pub mod server;
pub mod team;
pub mod term;
pub mod transport;
pub mod view;

use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use crate::app::App;
use proto::{ClientMsg, EngineMsg, ReviewDecision, StopTarget, PROTOCOL_VERSION};

/// The terminal app's way of asking the engine for an agent action. In a
/// window onto an engine (EN-2) the message goes over the link; otherwise it
/// goes through `handle` as the window "terminal", and an error the engine
/// answers with is shown as a warning toast.
pub fn act(app: &mut App, msg: ClientMsg, tx: &mpsc::UnboundedSender<String>) {
    if app.is_window() {
        link::act(app, msg, tx);
        return;
    }
    for reply in handle(app, msg, tx, "terminal") {
        match reply {
            EngineMsg::Error { message } => {
                app.push_toast(crate::toast::ToastKind::Warning, message);
            }
            EngineMsg::TeamReply { ok, body, .. } => team_toast(app, ok, &body),
            _ => {}
        }
    }
}

/// The engine's answer to a team request this app sent (TM-4), as a toast:
/// what was done, or why not, in the engine's words.
pub fn team_toast(app: &mut App, ok: bool, body: &serde_json::Value) {
    let text = match body {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let clipped: String = text.chars().take(160).collect();
    if ok {
        app.push_toast(crate::toast::ToastKind::Info, format!("team: {clipped}"));
    } else {
        app.push_toast(crate::toast::ToastKind::Warning, format!("team: {clipped}"));
    }
}

/// What one `pump` did: how many loop messages and approval prompts arrived,
/// and the messages to send to every window.
pub struct Pump {
    pub messages: usize,
    pub approvals: usize,
    pub out: Vec<EngineMsg>,
    /// What each terminal agent wrote, for the windows watching it (OM-2).
    pub term: Vec<(String, Vec<u8>)>,
}

/// Take in everything the agent loops reported since the last call, in the
/// order the main loop always used: messages first (each applied to the app,
/// then sent on as an `Event`), then new approval prompts, then a new
/// question. Approvals and the question come with their ids.
pub fn pump(
    app: &mut App,
    rx: &mut mpsc::UnboundedReceiver<String>,
    tx: &mpsc::UnboundedSender<String>,
) -> Pump {
    let mut out = Vec::new();
    let mut messages = 0;
    let approvals_tx = app.approval_tx.clone();
    app.team.pump(&approvals_tx);
    let term = app.term_host.pump();
    // In process there is no window to send to: the app's own tile screens
    // are fed here, as a window's are from `TermOutput`.
    if !app.is_window() {
        for (id, bytes) in &term {
            app.feed_tile(id, bytes);
        }
        app.term_tiles = app.term_host.tiles();
    }
    while let Ok(token) = rx.try_recv() {
        messages += 1;
        app.apply_token(&token, tx);
        out.push(EngineMsg::Event { token });
    }
    let question_before = app.question_id;
    let approvals = app.drain_agent_channels();
    let first_new = app.approval_queue.len() - approvals;
    for (i, (request, _)) in app.approval_queue.iter().enumerate().skip(first_new) {
        out.push(EngineMsg::ApprovalRequested {
            approval: approval_view(app.approval_ids[i], request),
        });
    }
    if app.question_id != question_before {
        if let (Some(id), Some(text)) = (app.question_id, app.question_text.clone()) {
            out.push(EngineMsg::QuestionAsked { id, text });
        }
    }
    Pump {
        messages,
        approvals,
        out,
        term,
    }
}

/// Carry out one request about the terminal agents (OM-2).
fn term_request(app: &mut App, request: proto::TermRequest) -> Result<serde_json::Value, String> {
    use proto::TermRequest;
    use xencode_term_rs::Size;
    match request {
        TermRequest::Agents => Ok(serde_json::to_value(term::picker()).unwrap_or_default()),
        TermRequest::Open { agent, cols, rows } => {
            let root = xencode_context_rs::default_root();
            let tile = app.term_host.open(&root, &agent, Size { cols, rows })?;
            Ok(serde_json::to_value(tile).unwrap_or_default())
        }
        TermRequest::Input { id, data } => {
            let bytes = proto::unb64(&data)?;
            app.term_host.input(&id, &bytes)?;
            Ok(serde_json::Value::Null)
        }
        TermRequest::Resize { id, cols, rows } => {
            app.term_host.resize(&id, Size { cols, rows })?;
            Ok(serde_json::Value::Null)
        }
        TermRequest::Watch { ids } => {
            let mut replays = serde_json::Map::new();
            for id in ids {
                if let Some(bytes) = app.term_host.replay(&id) {
                    replays.insert(id, serde_json::Value::String(proto::b64(&bytes)));
                }
            }
            Ok(serde_json::json!({ "replays": replays }))
        }
        TermRequest::Stop { id } => {
            app.term_host.stop(&id)?;
            Ok(serde_json::Value::Null)
        }
        TermRequest::Merge { id } => {
            // The program is stopped first, so nothing writes into the
            // worktree while it is merged and the folder can go afterwards.
            app.term_host.stop(&id)?;
            let (worker, base) = app
                .term_host
                .worker(&id)
                .ok_or_else(|| format!("there is no terminal agent `{id}`"))?;
            let root = xencode_context_rs::default_root();
            app.team.adopt_terminal(&root, worker, base);
            let approvals = app.approval_tx.clone();
            app.team.request(
                &root,
                proto::TeamRequest::Merge { id },
                &approvals,
                &app.config,
            )
        }
    }
}

fn approval_view(id: u64, request: &crate::agent_tools::ApprovalRequest) -> proto::ApprovalView {
    proto::ApprovalView {
        id,
        tool: request.tool.clone(),
        class: request.class_label().to_string(),
        summary: request.summary.clone(),
        preview: request.preview.clone(),
    }
}

/// Whether something waits for a person: a question, an approval or a task
/// waiting for review.
pub fn waits_on_a_person(app: &App) -> bool {
    app.question_id.is_some() || !app.approval_queue.is_empty() || app.bytebot_reviewing().is_some()
}

/// What the engine does when no window has been connected for the wait
/// limit while something waited for a person (EN-3): a question is
/// withdrawn and its task stopped, every approval is denied, and a task
/// waiting for review is completed with its changes kept. Each thing done
/// is said in the transcript; the lines are returned.
pub fn unattended(app: &mut App, tx: &mpsc::UnboundedSender<String>) -> Vec<String> {
    let mut said = Vec::new();
    if app.question_id.is_some() {
        handle(
            app,
            ClientMsg::Stop {
                target: StopTarget::Bytebot,
            },
            tx,
            "engine",
        );
        said.push(
            "No window answered ByteBot's question in time: the question was withdrawn and its task stopped."
                .to_string(),
        );
    }
    let approvals = app.approval_queue.len();
    if approvals > 0 {
        while !app.approval_queue.is_empty() {
            app.resolve_approval(crate::agent_tools::ApprovalAnswer::Denied);
        }
        said.push(format!(
            "No window answered {approvals} approval prompt(s) in time: each was denied, and nothing was written."
        ));
    }
    if app.bytebot_reviewing().is_some() {
        app.bytebot_accept(tx.clone());
        said.push(
            "No window reviewed ByteBot's finished task in time: it was completed and its changes kept."
                .to_string(),
        );
    }
    for line in &said {
        app.push_system_message(line.clone());
    }
    said
}

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
                    message: QUESTION_GONE.to_string(),
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
                    message: APPROVAL_GONE.to_string(),
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
        ClientMsg::SetPolicy {
            approval,
            cloud_models,
            external_workers,
        } => {
            // Only a person's terminal window decides what the agent may do;
            // the reader a lead agent's MCP tools connect through, or any
            // other client, is refused.
            if !by.starts_with("terminal") {
                return vec![EngineMsg::Error {
                    message: format!(
                        "{by} cannot change the approval mode or cloud access; \
                         only a terminal window can"
                    ),
                }];
            }
            // Read back through the parser, so an unknown word becomes `ask`.
            let approval = crate::agent_tools::ApprovalMode::parse(&approval)
                .config_name()
                .to_string();
            let changed = app.config.agent_approval != approval
                || app.config.allow_cloud_models != cloud_models
                || app.config.allow_external_workers != external_workers;
            app.config.agent_approval = approval;
            app.config.allow_cloud_models = cloud_models;
            app.config.allow_external_workers = external_workers;
            if changed {
                // Said in the conversation, so every window sees who changed it.
                let onoff = |on: bool| if on { "on" } else { "off" };
                app.messages.push(crate::app::UiMessage {
                    role: "system".into(),
                    content: format!(
                        "⚙ approval mode {}, cloud models {}, external workers {} — set by {by}",
                        app.config.agent_approval,
                        onoff(cloud_models),
                        onoff(external_workers)
                    ),
                });
            }
            Vec::new()
        }
        ClientMsg::Term { req, request } => {
            // A terminal agent runs whatever is typed into it, with no
            // approval: only a person's terminal window may open, type into,
            // watch, resize or stop one, never the reader a lead agent's MCP
            // tools connect through, or any other client.
            if !by.starts_with("terminal") {
                return vec![EngineMsg::TermReply {
                    req,
                    ok: false,
                    body: serde_json::Value::String(format!(
                        "{by} cannot use the terminal agents; only a terminal window can"
                    )),
                }];
            }
            let (ok, body) = match term_request(app, request) {
                Ok(body) => (true, body),
                Err(why) => (false, serde_json::Value::String(why)),
            };
            vec![EngineMsg::TermReply { req, ok, body }]
        }
        ClientMsg::ResumeTasks => {
            app.bytebot_start_next(tx.clone());
            Vec::new()
        }
        ClientMsg::Note { role, content } => {
            app.messages.push(crate::app::UiMessage { role, content });
            Vec::new()
        }
        ClientMsg::Team { req, request } => {
            let root = xencode_context_rs::default_root();
            let approvals = app.approval_tx.clone();
            let (ok, body) = match app.team.request(&root, request, &approvals, &app.config) {
                Ok(body) => (true, body),
                Err(why) => (false, serde_json::Value::String(why)),
            };
            vec![EngineMsg::TeamReply { req, ok, body }]
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

    #[tokio::test]
    async fn a_question_nobody_answers_is_withdrawn_and_its_task_stopped() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        let (help_tx, mut help_rx) = oneshot::channel::<String>();
        app.bytebot_help = Some(help_tx);
        app.question_id = Some(3);
        app.question_text = Some("Which port?".into());
        app.bytebot_running = true;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        app.bytebot_stop = Some(stop.clone());

        let said = unattended(&mut app, &tx);
        assert_eq!(app.question_id, None);
        assert!(stop.load(Ordering::Relaxed), "the waiting task is stopped");
        assert!(
            help_rx.try_recv().is_err(),
            "the question's answer channel is closed"
        );
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(said[0].contains("question"), "{said:?}");
        assert!(app.messages.iter().any(|m| m.content == said[0]));
    }

    #[tokio::test]
    async fn approvals_nobody_answers_are_denied() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut answers = Vec::new();
        for (i, summary) in ["write a.rs", "write b.rs"].into_iter().enumerate() {
            let (answer_tx, answer_rx) = oneshot::channel();
            app.approval_queue.push_back((request(summary), answer_tx));
            app.approval_ids.push_back(i as u64 + 1);
            answers.push(answer_rx);
        }
        let said = unattended(&mut app, &tx);
        for mut answer in answers {
            assert_eq!(
                answer.try_recv().unwrap(),
                crate::agent_tools::ApprovalAnswer::Denied
            );
        }
        assert!(app.approval_queue.is_empty() && app.approval_ids.is_empty());
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(said[0].contains("2 approval"), "{said:?}");
    }

    #[tokio::test]
    async fn a_review_nobody_answers_is_completed_with_its_changes_kept() {
        use crate::bytebot_tasks::{ByteBotTask, TaskState};
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut task = ByteBotTask::new("write a note", "llamacpp:none");
        task.state = TaskState::NeedsReview;
        task.changed_files = vec!["note.md".into()];
        task.this_session = true;
        app.bytebot_tasks.push(task);
        let said = unattended(&mut app, &tx);
        assert_eq!(app.bytebot_tasks[0].state, TaskState::Completed);
        assert_eq!(
            app.bytebot_tasks[0].changed_files,
            vec!["note.md".to_string()]
        );
        assert!(said[0].contains("kept"), "{said:?}");
    }

    #[tokio::test]
    async fn nothing_waiting_means_nothing_done() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        assert!(unattended(&mut app, &tx).is_empty());
        assert!(!waits_on_a_person(&app));
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
    async fn pump_turns_tokens_into_events_and_reports_counts() {
        let mut app = engine_app();
        let (tx, mut rx) = mpsc::unbounded_channel();
        app.is_generating = true;
        tx.send("hello ".to_string()).unwrap();
        tx.send("[DONE]".to_string()).unwrap();
        let pumped = pump(&mut app, &mut rx, &tx);
        assert_eq!(pumped.messages, 2);
        assert_eq!(pumped.approvals, 0);
        assert_eq!(
            pumped.out,
            vec![
                EngineMsg::Event {
                    token: "hello ".into()
                },
                EngineMsg::Event {
                    token: "[DONE]".into()
                },
            ]
        );
        assert!(
            !app.is_generating,
            "the tokens were applied, not only forwarded"
        );
    }

    #[tokio::test]
    async fn pump_keeps_the_frame_order() {
        let mut app = engine_app();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let (asker, _a) = oneshot::channel();
        app.ask_tx.send(("Which port?".into(), asker)).unwrap();
        let (responder, _b) = oneshot::channel();
        app.approval_tx
            .send((request("write_file a.rs"), responder))
            .unwrap();
        tx.send("[TOOL]→ write_file a.rs".to_string()).unwrap();
        let pumped = pump(&mut app, &mut rx, &tx);
        assert_eq!(pumped.approvals, 1);
        let id = app.approval_ids[0];
        let question = app.question_id.unwrap();
        assert_eq!(
            pumped.out,
            vec![
                EngineMsg::Event {
                    token: "[TOOL]→ write_file a.rs".into()
                },
                EngineMsg::ApprovalRequested {
                    approval: ApprovalView {
                        id,
                        tool: "write_file".into(),
                        class: crate::agent_tools::ToolClass::Edit.overlay_label().into(),
                        summary: "write_file a.rs".into(),
                        preview: String::new(),
                    }
                },
                EngineMsg::QuestionAsked {
                    id: question,
                    text: "Which port?".into()
                },
            ]
        );
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

    /// Review C1: the approval mode, cloud access and external workers a
    /// window sets are the ones the engine, which runs the tools, enforces.
    #[tokio::test]
    async fn the_policy_a_window_sets_is_the_one_the_engine_enforces() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        assert_eq!(app.config.agent_approval, "ask");
        handle(
            &mut app,
            ClientMsg::SetPolicy {
                approval: "plan".into(),
                cloud_models: true,
                external_workers: true,
            },
            &tx,
            "terminal 1",
        );
        assert_eq!(app.config.agent_approval, "plan");
        assert!(app.config.allow_cloud_models && app.config.allow_external_workers);
        // A word the engine does not know is the most guarded mode, not a typo
        // that stays in force.
        handle(
            &mut app,
            ClientMsg::SetPolicy {
                approval: "yolo".into(),
                cloud_models: false,
                external_workers: false,
            },
            &tx,
            "terminal 1",
        );
        assert_eq!(app.config.agent_approval, "ask");
        assert!(!app.config.allow_cloud_models);
    }

    /// The security review of the policy message: only a person's terminal
    /// window may change what the agent may do — not the reader a lead agent's
    /// MCP tools connect through — and every change is said in the
    /// conversation, naming the window, so all windows see it.
    #[tokio::test]
    async fn only_a_terminal_window_changes_the_policy_and_the_change_is_said() {
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        let loosen = || ClientMsg::SetPolicy {
            approval: "autonomous".into(),
            cloud_models: true,
            external_workers: true,
        };
        let replies = handle(&mut app, loosen(), &tx, "reader 4242");
        assert!(
            matches!(replies.as_slice(), [EngineMsg::Error { .. }]),
            "{replies:?}"
        );
        assert_eq!(app.config.agent_approval, "ask");
        assert!(!app.config.allow_cloud_models && !app.config.allow_external_workers);
        handle(&mut app, loosen(), &tx, "terminal 77");
        assert_eq!(app.config.agent_approval, "autonomous");
        let said = app
            .messages
            .last()
            .map(|m| m.content.clone())
            .unwrap_or_default();
        assert!(
            said.contains("autonomous") && said.contains("terminal 77"),
            "{said}"
        );
        // Sending what is already in force says nothing again.
        let before = app.messages.len();
        handle(&mut app, loosen(), &tx, "terminal 77");
        assert_eq!(app.messages.len(), before);
    }

    /// The security review of OM-2: a terminal agent runs whatever is typed
    /// into it with no approval, so only a person's terminal window may open,
    /// type into, watch, resize or stop one — not the reader a lead agent's
    /// MCP tools connect through, which a model drives.
    #[tokio::test]
    async fn only_a_terminal_window_can_use_the_terminal_agents() {
        use crate::engine::proto::TermRequest;
        let mut app = engine_app();
        let (tx, _rx) = mpsc::unbounded_channel();
        for request in [
            TermRequest::Open {
                agent: "shell".into(),
                cols: 80,
                rows: 24,
            },
            TermRequest::Input {
                id: "t1".into(),
                data: proto::b64(b"echo hi\r"),
            },
            TermRequest::Watch {
                ids: vec!["t1".into()],
            },
            TermRequest::Stop { id: "t1".into() },
        ] {
            let replies = handle(
                &mut app,
                ClientMsg::Term { req: 7, request },
                &tx,
                "reader 4242",
            );
            assert!(
                matches!(replies.as_slice(), [EngineMsg::TermReply { ok: false, .. }]),
                "{replies:?}"
            );
        }
        assert!(app.term_host.tiles().is_empty(), "nothing was started");
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
