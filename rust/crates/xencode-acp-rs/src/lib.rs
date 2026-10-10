//! `xencode acp` (M-7): xencode as an agent for editors that speak the Agent
//! Client Protocol, such as Zed. The editor starts this process and talks to
//! it with JSON-RPC over standard input and output; each session works
//! through its folder's engine (EN-2), as the terminal app does. Standard
//! output carries only protocol messages; anything else goes to standard
//! error.

pub mod kinds;
pub mod options;
pub mod prompt;
pub mod session;
pub mod turn;

use agent_client_protocol::schema::v1::{
    AgentCapabilities, AuthMethod, AuthMethodAgent, AuthenticateRequest, AuthenticateResponse,
    AvailableCommandsUpdate, CancelNotification, Implementation, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PromptCapabilities, PromptRequest,
    SessionId, SessionNotification, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{Agent, Stdio};

/// The one sign-in method: xencode's own settings. The ACP Registry wants
/// every agent to offer at least one.
pub const AUTH_METHOD_ID: &str = "xencode-settings";

/// JSON-RPC's "invalid params" code, for requests xencode cannot carry out
/// as asked.
pub const INVALID: i32 = -32602;

/// Serve the protocol on standard input and output until the editor closes
/// them.
pub async fn serve_stdio() -> Result<(), String> {
    let sessions = session::Sessions::default();
    let for_new = sessions.clone();
    let for_prompt = sessions.clone();
    let for_cancel = sessions.clone();
    let for_options = sessions.clone();
    Agent
        .builder()
        .name("xencode")
        .on_receive_request(
            async move |_init: InitializeRequest, responder, _connection| {
                // xencode speaks version 1; a client asking for another is
                // answered with 1 and decides whether to go on.
                responder.respond(
                    InitializeResponse::new(ProtocolVersion::V1)
                        .agent_info(Implementation::new("xencode", env!("CARGO_PKG_VERSION")))
                        .agent_capabilities(
                            AgentCapabilities::new().prompt_capabilities(
                                PromptCapabilities::new().embedded_context(true),
                            ),
                        )
                        .auth_methods(vec![AuthMethod::Agent(AuthMethodAgent::new(
                            AUTH_METHOD_ID,
                            "Use xencode's settings",
                        ))]),
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |_auth: AuthenticateRequest, responder, _connection| match model_problem() {
                None => responder.respond(AuthenticateResponse::new()),
                Some(why) => {
                    responder.respond_with_error(agent_client_protocol::Error::new(INVALID, why))
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |req: NewSessionRequest, responder, connection| {
                if !req.mcp_servers.is_empty() {
                    eprintln!(
                        "xencode acp: the editor's {} tool server(s) are not used; xencode uses its own",
                        req.mcp_servers.len()
                    );
                }
                // Starting an engine and asking model servers take seconds;
                // the dispatch loop must not wait for them.
                let sessions = for_new.clone();
                let notify = connection.clone();
                connection.spawn(async move {
                    match session::Session::open(&req.cwd).await {
                        Ok(session) => {
                            let option =
                                options::model_option(&session.model, &options::models().await);
                            let id = SessionId::new(sessions.add(session));
                            responder.respond(
                                NewSessionResponse::new(id.clone()).config_options(vec![option]),
                            )?;
                            notify.send_notification(SessionNotification::new(
                                id,
                                SessionUpdate::AvailableCommandsUpdate(
                                    AvailableCommandsUpdate::new(options::commands()),
                                ),
                            ))
                        }
                        Err(why) => responder
                            .respond_with_error(agent_client_protocol::Error::new(INVALID, why)),
                    }
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |req: SetSessionConfigOptionRequest, responder, connection| {
                let fail = |why: String| agent_client_protocol::Error::new(INVALID, why);
                if req.config_id.to_string() != options::MODEL_OPTION {
                    return responder.respond_with_error(fail(
                        "xencode offers only the model option".to_string(),
                    ));
                }
                let Some(state) = for_options.get(&req.session_id.to_string()) else {
                    return responder
                        .respond_with_error(fail(format!("no session {}", req.session_id)));
                };
                // The switch waits on the engine; the dispatch loop must not.
                connection.spawn(async move {
                    match options::set_model(&state, &req.value).await {
                        Ok(model) => {
                            let option = options::model_option(&model, &options::models().await);
                            responder.respond(SetSessionConfigOptionResponse::new(vec![option]))
                        }
                        Err(why) => responder.respond_with_error(fail(why)),
                    }
                })
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            async move |req: PromptRequest, responder, connection| {
                prompt::start(&for_prompt, req, responder, connection).await
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_notification(
            async move |note: CancelNotification, _connection| {
                prompt::cancel(&for_cancel, &note.session_id.to_string()).await;
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_to(Stdio::new())
        .await
        .map_err(|e| format!("the editor connection ended: {e}"))
}

/// Why xencode cannot answer yet, or `None` when a model is set.
fn model_problem() -> Option<String> {
    match xencode_config_rs::XencodeConfig::load() {
        Ok(config) if !config.default_model.trim().is_empty() => None,
        Ok(_) => {
            Some("no model is set: run `xencode config set default_model <model>`".to_string())
        }
        Err(e) => Some(format!(
            "xencode's settings could not be read ({e}); check them with `xencode config`"
        )),
    }
}
