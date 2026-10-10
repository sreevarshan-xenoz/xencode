//! M-7: the real `xencode acp`, driven by the official ACP crate's client
//! side — a second ACP client, as M-7's done-when allows.

use agent_client_protocol::schema::v1::{AuthenticateRequest, InitializeRequest};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{AcpAgent, AcpAgentConfig, Agent, Client, ConnectionTo};

/// Settings whose model server address has nothing listening on it.
pub fn settings() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("no-llama-server-here");
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "default_model": "llamacpp:none",
            "llama_cpp_url": "http://127.0.0.1:9",
            "llama_cpp_executable": missing.to_string_lossy(),
        })
        .to_string(),
    )
    .unwrap();
    dir
}

/// `xencode acp` with `config` as its settings folder.
pub fn agent(config: &std::path::Path) -> AcpAgent {
    AcpAgent::new(
        AcpAgentConfig::new(env!("CARGO_BIN_EXE_xencode"))
            .arg("acp")
            .env("XCODE_CONFIG_DIR", config.to_string_lossy().to_string()),
    )
}

#[tokio::test]
async fn the_handshake_names_xencode_and_offers_its_settings_as_sign_in() {
    let config = settings();
    Client
        .builder()
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            let init = c
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            assert_eq!(init.protocol_version, ProtocolVersion::V1);
            assert_eq!(init.agent_info.as_ref().unwrap().name, "xencode");
            assert_eq!(init.auth_methods.len(), 1);
            assert_eq!(init.auth_methods[0].id().to_string(), "xencode-settings");
            // A model is configured, so signing in with the settings works.
            c.send_request(AuthenticateRequest::new("xencode-settings"))
                .block_task()
                .await?;
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn signing_in_without_a_model_says_how_to_set_one() {
    let config = tempfile::tempdir().unwrap();
    std::fs::write(
        config.path().join("config.json"),
        r#"{"default_model": ""}"#,
    )
    .unwrap();
    Client
        .builder()
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let err = c
                .send_request(AuthenticateRequest::new("xencode-settings"))
                .block_task()
                .await
                .expect_err("no model is set");
            assert!(format!("{err:?}").contains("xencode config"), "{err:?}");
            Ok(())
        })
        .await
        .unwrap();
}

use agent_client_protocol::schema::v1::{
    CancelNotification, ContentBlock, NewSessionRequest, PromptRequest, SessionNotification,
    SessionUpdate, StopReason, TextContent,
};
use std::sync::{Arc, Mutex};

type Seen = Arc<Mutex<Vec<SessionUpdate>>>;

/// Everything the agent said as message text, in order.
fn said(seen: &Seen) -> String {
    seen.lock()
        .unwrap()
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::AgentMessageChunk(chunk) => match &chunk.content {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

fn text(prompt: &str) -> Vec<ContentBlock> {
    vec![ContentBlock::Text(TextContent::new(prompt))]
}

/// Settings whose model server accepts connections and never answers, so a
/// turn stays running until it is stopped. The listener is returned so it
/// lives as long as the test.
fn stalled_settings() -> (tempfile::TempDir, std::net::TcpListener) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "default_model": "llamacpp:none",
            "llama_cpp_url": url,
            "llama_cpp_executable": dir.path().join("none").to_string_lossy(),
        })
        .to_string(),
    )
    .unwrap();
    (dir, listener)
}

#[tokio::test]
async fn a_prompt_reaches_the_engine_and_the_unreachable_model_is_said() {
    let config = settings();
    let project = tempfile::tempdir().unwrap();
    let cwd = project.path().to_path_buf();
    let seen: Seen = Default::default();
    let record = Arc::clone(&seen);
    Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _c| {
                record.lock().unwrap().push(n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let s = c
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let r = c
                .send_request(PromptRequest::new(s.session_id, text("say hi")))
                .block_task()
                .await?;
            assert_eq!(r.stop_reason, StopReason::EndTurn);
            Ok(())
        })
        .await
        .unwrap();
    let text = said(&seen);
    assert!(
        text.contains("127.0.0.1:9"),
        "the failure reached the client: {text}"
    );
}

/// Review Focus 1: Zed lets a person send while a turn runs.
#[tokio::test]
async fn a_second_prompt_while_one_runs_is_refused_and_a_cancel_stops_the_first() {
    let (config, _stalled) = stalled_settings();
    let project = tempfile::tempdir().unwrap();
    let cwd = project.path().to_path_buf();
    Client
        .builder()
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let s = c
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let first = c.send_request(PromptRequest::new(s.session_id.clone(), text("one")));
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let second = c
                .send_request(PromptRequest::new(s.session_id.clone(), text("two")))
                .block_task()
                .await
                .expect_err("one turn at a time");
            assert!(
                format!("{second:?}").contains("already running"),
                "{second:?}"
            );
            c.send_notification(CancelNotification::new(s.session_id))?;
            let r = tokio::time::timeout(std::time::Duration::from_secs(30), first.block_task())
                .await
                .expect("the cancel ended the turn")?;
            assert_eq!(r.stop_reason, StopReason::Cancelled);
            Ok(())
        })
        .await
        .unwrap();
}

/// Review Focus 3.
#[tokio::test]
async fn a_folder_that_does_not_exist_or_is_a_file_is_refused() {
    let config = settings();
    let project = tempfile::tempdir().unwrap();
    let missing = project.path().join("missing");
    let file = project.path().join("a-file.txt");
    std::fs::write(&file, "x").unwrap();
    Client
        .builder()
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let err = c
                .send_request(NewSessionRequest::new(missing))
                .block_task()
                .await
                .expect_err("no such folder");
            assert!(format!("{err:?}").contains("missing"), "{err:?}");
            let err = c
                .send_request(NewSessionRequest::new(file))
                .block_task()
                .await
                .expect_err("a file");
            assert!(format!("{err:?}").contains("not a folder"), "{err:?}");
            Ok(())
        })
        .await
        .unwrap();
}

/// Review Focus 2: the engine dies; the prompt says so instead of hanging,
/// and the next prompt gets a new engine.
#[tokio::test]
async fn a_lost_engine_ends_the_prompt_with_an_error_and_the_next_gets_a_new_one() {
    let (config, _stalled) = stalled_settings();
    let project = tempfile::tempdir().unwrap();
    let cwd = project.path().to_path_buf();
    // An engine this test owns, so it can be killed; `xencode acp` finds it.
    let mut engine = std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(["engine", "--project"])
        .arg(project.path())
        .env("XCODE_CONFIG_DIR", config.path())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    {
        use std::io::BufRead;
        let mut line = String::new();
        std::io::BufReader::new(engine.stdout.as_mut().unwrap())
            .read_line(&mut line)
            .unwrap();
        assert!(line.contains("listening on"), "{line}");
    }
    let engine = Arc::new(Mutex::new(engine));
    let killer = Arc::clone(&engine);
    Client
        .builder()
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let s = c
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let first = c.send_request(PromptRequest::new(s.session_id.clone(), text("one")));
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            {
                let mut child = killer.lock().unwrap();
                child.kill().unwrap();
                child.wait().unwrap();
            }
            let err = tokio::time::timeout(std::time::Duration::from_secs(20), first.block_task())
                .await
                .expect("the loss ended the prompt")
                .expect_err("an error, not a normal end");
            assert!(format!("{err:?}").contains("engine"), "{err:?}");
            // The next prompt starts a new engine; it is stopped at once.
            let next = c.send_request(PromptRequest::new(s.session_id.clone(), text("two")));
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            c.send_notification(CancelNotification::new(s.session_id))?;
            let r = tokio::time::timeout(std::time::Duration::from_secs(30), next.block_task())
                .await
                .expect("the new engine took the prompt")?;
            assert_eq!(r.stop_reason, StopReason::Cancelled);
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "needs a real llama.cpp server: set XENCODE_LIVE_LLAMACPP_URL"]
async fn a_real_model_streams_its_answer() {
    let url = std::env::var("XENCODE_LIVE_LLAMACPP_URL").expect("XENCODE_LIVE_LLAMACPP_URL");
    let model = std::env::var("XENCODE_LIVE_MODEL").unwrap_or_else(|_| "llamacpp:live".into());
    let config = tempfile::tempdir().unwrap();
    std::fs::write(
        config.path().join("config.json"),
        serde_json::json!({"default_model": model, "llama_cpp_url": url}).to_string(),
    )
    .unwrap();
    let project = tempfile::tempdir().unwrap();
    let cwd = project.path().to_path_buf();
    let seen: Seen = Default::default();
    let record = Arc::clone(&seen);
    Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _c| {
                record.lock().unwrap().push(n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let s = c
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let r = c
                .send_request(PromptRequest::new(
                    s.session_id,
                    text("Count from 1 to 10, separated by spaces, and write nothing else."),
                ))
                .block_task()
                .await?;
            assert_eq!(r.stop_reason, StopReason::EndTurn);
            Ok(())
        })
        .await
        .unwrap();
    let chunks = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|u| matches!(u, SessionUpdate::AgentMessageChunk(_)))
        .count();
    let text = said(&seen);
    eprintln!("live answer in {chunks} chunks: {text}");
    assert!(chunks >= 2, "streamed, not sent whole: {chunks}");
    // What the model says is the model's business; that it arrives, in
    // pieces, is xencode's.
    assert!(!text.trim().is_empty(), "an answer arrived");
}

use agent_client_protocol::schema::v1::{
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, ToolCallContent, ToolKind,
};

/// Settings for a real llama.cpp server named by `XENCODE_LIVE_LLAMACPP_URL`.
fn live_settings() -> tempfile::TempDir {
    let url = std::env::var("XENCODE_LIVE_LLAMACPP_URL").expect("XENCODE_LIVE_LLAMACPP_URL");
    let model = std::env::var("XENCODE_LIVE_MODEL").unwrap_or_else(|_| "llamacpp:live".into());
    let config = tempfile::tempdir().unwrap();
    std::fs::write(
        config.path().join("config.json"),
        serde_json::json!({"default_model": model, "llama_cpp_url": url}).to_string(),
    )
    .unwrap();
    config
}

/// How the test's client answers permission requests.
#[derive(Clone, Copy)]
enum Answer {
    Pick(&'static str),
    /// Never answer; the test cancels instead.
    Never,
}

const WRITE_NOTE: &str =
    "Use the write_file tool to create the file notes.txt containing exactly: hi. \
Do not do anything else.";

/// Run one prompt on a fresh session in `project`, answering permission
/// requests with `answer`. Returns the updates seen, the permission
/// requests seen and the stop reason.
async fn live_turn(
    project: &std::path::Path,
    answer: Answer,
    cancel_on_permission: bool,
) -> (Vec<SessionUpdate>, usize, StopReason) {
    let config = live_settings();
    let cwd = project.to_path_buf();
    let seen: Seen = Default::default();
    let record = Arc::clone(&seen);
    let asked = Arc::new(Mutex::new(0usize));
    let counted = Arc::clone(&asked);
    let (asked_tx, asked_rx) = tokio::sync::watch::channel(false);
    let mut asked_rx = asked_rx;
    let stop = Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _c| {
                record.lock().unwrap().push(n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |_req: RequestPermissionRequest, responder, connection| {
                *counted.lock().unwrap() += 1;
                let _ = asked_tx.send(true);
                match answer {
                    Answer::Pick(id) => responder.respond(RequestPermissionResponse::new(
                        RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                            id.to_string(),
                        )),
                    )),
                    Answer::Never => connection.spawn(async move {
                        // Answered only when the turn is cancelled, as the
                        // protocol asks of a client.
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        responder.respond(RequestPermissionResponse::new(
                            RequestPermissionOutcome::Cancelled,
                        ))
                    }),
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let s = c
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let turn = c.send_request(PromptRequest::new(s.session_id.clone(), text(WRITE_NOTE)));
            if cancel_on_permission {
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(300),
                    asked_rx.wait_for(|asked| *asked),
                )
                .await;
                c.send_notification(CancelNotification::new(s.session_id))?;
            }
            let r = turn.block_task().await?;
            Ok(r.stop_reason)
        })
        .await
        .unwrap();
    let updates = seen.lock().unwrap().clone();
    let asked = *asked.lock().unwrap();
    (updates, asked, stop)
}

#[tokio::test]
#[ignore = "needs a real llama.cpp server: set XENCODE_LIVE_LLAMACPP_URL"]
async fn an_approved_edit_writes_the_file_and_shows_a_diff() {
    let project = tempfile::tempdir().unwrap();
    let (updates, asked, stop) = live_turn(project.path(), Answer::Pick("allow"), false).await;
    eprintln!(
        "asked {asked} time(s); stop {stop:?}; {} updates",
        updates.len()
    );
    assert_eq!(stop, StopReason::EndTurn);
    assert!(asked >= 1, "the edit asked first");
    assert_eq!(
        std::fs::read_to_string(project.path().join("notes.txt"))
            .unwrap()
            .trim(),
        "hi"
    );
    assert!(
        updates
            .iter()
            .any(|u| matches!(u, SessionUpdate::ToolCall(c) if c.kind == ToolKind::Edit)),
        "an edit tool call was shown"
    );
    let diff = updates.iter().find_map(|u| match u {
        SessionUpdate::ToolCallUpdate(u) => {
            u.fields.content.as_ref()?.iter().find_map(|c| match c {
                ToolCallContent::Diff(d) => Some(d.clone()),
                _ => None,
            })
        }
        _ => None,
    });
    let diff = diff.unwrap_or_else(|| {
        let tools: Vec<_> = updates
            .iter()
            .filter(|u| {
                matches!(
                    u,
                    SessionUpdate::ToolCall(_) | SessionUpdate::ToolCallUpdate(_)
                )
            })
            .collect();
        panic!("the edit's diff was sent: {tools:#?}")
    });
    assert!(diff.path.ends_with("notes.txt"), "{:?}", diff.path);
    assert_eq!(diff.new_text.trim(), "hi");
    assert_eq!(diff.old_text, None, "a new file");
}

#[tokio::test]
#[ignore = "needs a real llama.cpp server: set XENCODE_LIVE_LLAMACPP_URL"]
async fn a_rejected_edit_writes_nothing() {
    let project = tempfile::tempdir().unwrap();
    let (_updates, asked, stop) = live_turn(project.path(), Answer::Pick("deny"), false).await;
    eprintln!("asked {asked} time(s); stop {stop:?}");
    assert!(asked >= 1, "the edit asked first");
    assert!(!project.path().join("notes.txt").exists());
}

/// Review Focus 4.
#[tokio::test]
#[ignore = "needs a real llama.cpp server: set XENCODE_LIVE_LLAMACPP_URL"]
async fn a_cancel_while_permission_is_open_stops_the_turn() {
    let project = tempfile::tempdir().unwrap();
    let (_updates, asked, stop) = live_turn(project.path(), Answer::Never, true).await;
    eprintln!("asked {asked} time(s); stop {stop:?}");
    assert!(asked >= 1, "the edit asked first");
    assert_eq!(stop, StopReason::Cancelled);
    assert!(!project.path().join("notes.txt").exists());
}

use agent_client_protocol::schema::v1::{
    SessionConfigKind, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigSelectOptions, SetSessionConfigOptionRequest,
};

/// The model option's current value and choices, from a list of options.
fn model_option(options: &[SessionConfigOption]) -> (String, Vec<String>) {
    let option = options
        .iter()
        .find(|o| o.id.to_string() == "model")
        .expect("a model option");
    assert_eq!(option.category, Some(SessionConfigOptionCategory::Model));
    match &option.kind {
        SessionConfigKind::Select(select) => {
            let choices = match &select.options {
                SessionConfigSelectOptions::Ungrouped(list) => {
                    list.iter().map(|o| o.value.to_string()).collect()
                }
                other => panic!("grouped options: {other:?}"),
            };
            (select.current_value.to_string(), choices)
        }
        other => panic!("not a select: {other:?}"),
    }
}

#[tokio::test]
async fn a_new_session_offers_the_model_and_choosing_one_changes_the_engines_model() {
    let config = settings();
    let project = tempfile::tempdir().unwrap();
    let cwd = project.path().to_path_buf();
    Client
        .builder()
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let s = c
                .send_request(NewSessionRequest::new(cwd.clone()))
                .block_task()
                .await?;
            let (current, choices) = model_option(s.config_options.as_deref().unwrap_or(&[]));
            assert_eq!(current, "llamacpp:none");
            assert!(
                choices.contains(&"llamacpp:none".to_string()),
                "{choices:?}"
            );

            let changed = c
                .send_request(SetSessionConfigOptionRequest::new(
                    s.session_id.clone(),
                    "model",
                    agent_client_protocol::schema::v1::SessionConfigOptionValue::value_id(
                        "llamacpp:other",
                    ),
                ))
                .block_task()
                .await?;
            assert_eq!(model_option(&changed.config_options).0, "llamacpp:other");

            // The engine itself changed: a second session on the same folder
            // is told the engine's model.
            let again = c
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            assert_eq!(
                model_option(again.config_options.as_deref().unwrap_or(&[])).0,
                "llamacpp:other"
            );

            let err = c
                .send_request(SetSessionConfigOptionRequest::new(
                    s.session_id,
                    "mode",
                    agent_client_protocol::schema::v1::SessionConfigOptionValue::value_id("x"),
                ))
                .block_task()
                .await
                .expect_err("only the model option exists");
            assert!(format!("{err:?}").contains("only the model"), "{err:?}");
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn the_command_list_names_bytebot_and_a_terminal_panel_command_is_refused() {
    let config = settings();
    let project = tempfile::tempdir().unwrap();
    let cwd = project.path().to_path_buf();
    let seen: Seen = Default::default();
    let record = Arc::clone(&seen);
    Client
        .builder()
        .on_receive_notification(
            async move |n: SessionNotification, _c| {
                record.lock().unwrap().push(n.update);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(agent(config.path()), async move |c: ConnectionTo<Agent>| {
            c.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let s = c
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let err = tokio::time::timeout(
                std::time::Duration::from_secs(20),
                c.send_request(PromptRequest::new(s.session_id, text("/init")))
                    .block_task(),
            )
            .await
            .expect("refused at once, not run")
            .expect_err("a terminal panel command");
            assert!(format!("{err:?}").contains("xencode tui"), "{err:?}");
            Ok(())
        })
        .await
        .unwrap();
    let names: Vec<String> = seen
        .lock()
        .unwrap()
        .iter()
        .filter_map(|u| match u {
            SessionUpdate::AvailableCommandsUpdate(list) => Some(
                list.available_commands
                    .iter()
                    .map(|c| c.name.clone())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect();
    assert!(names.contains(&"bytebot".to_string()), "{names:?}");
    assert!(names.contains(&"model".to_string()), "{names:?}");
    assert!(!names.contains(&"init".to_string()), "{names:?}");
}
