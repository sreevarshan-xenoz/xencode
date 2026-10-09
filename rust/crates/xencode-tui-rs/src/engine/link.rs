//! The terminal app as a window onto the engine (EN-2): the connection, the
//! split between what runs in the engine and what runs in the window, and
//! what a window does each frame with what the engine sent.

use std::io;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use super::address::Address;
use super::proto::{self, ClientMsg, EngineMsg, PROTOCOL_VERSION};
use super::transport::connect;
use super::view::{self, View};
use crate::app::App;

/// How long a window waits for an engine it started to answer.
pub const START_WAIT: Duration = Duration::from_secs(5);

/// Commands that change agent state, and so run in the engine. Plain chat
/// does too. Every other command changes only the window and runs there.
pub const ENGINE_COMMANDS: &[&str] = &[
    "/bytebot", "/spawn", "/rewind", "/gate", "/plan", "/lesson", "/ctx", "/model", "/trust",
    "/mcp", "/plugin", "/skills",
];

/// Commands that run in the window: they draw a panel of the window's own,
/// hand the terminal over, or only read.
pub const WINDOW_COMMANDS: &[&str] = &[
    "/init",
    "/advise",
    "/impact",
    "/trace",
    "/cost",
    "/doctor",
    "/verify",
    "/hotspots",
    "/agents",
    "/workers",
    "/orchestrator",
    "/egress",
    "/goto",
    "/level",
    "/help",
];

/// Whether a submitted line runs in the engine: plain chat and the engine's
/// commands do. A line that only looks like a path (`/usr/lib is missing`)
/// is chat.
pub fn runs_in_engine(line: &str) -> bool {
    match crate::app::unknown_slash_command(line.trim_start()) {
        Some(word) => ENGINE_COMMANDS.contains(&word),
        None => true,
    }
}

/// The name a terminal window gives the engine, so other windows can be
/// told which window answered (EN-3).
pub fn window_name() -> String {
    format!("terminal {}", std::process::id())
}

/// What a window says when another window answered a prompt it was also
/// showing: `None` for its own answers and for every other message.
pub fn answered_elsewhere(msg: &EngineMsg, me: &str) -> Option<String> {
    match msg {
        EngineMsg::ApprovalResolved { by, answer, .. } if by != me => {
            let said = match answer {
                proto::WireAnswer::Allow => "allow",
                proto::WireAnswer::AllowForSession => "allow for the session",
                proto::WireAnswer::Deny => "deny",
            };
            Some(format!("answered in {by}: {said}"))
        }
        EngineMsg::QuestionAnswered { by, .. } if by != me => Some(format!("answered in {by}")),
        _ => None,
    }
}

/// What the reading task hands the window.
enum Incoming {
    Msg(EngineMsg),
    Lost(String),
}

/// A window's connection to its engine.
pub struct EngineLink {
    /// The name this window gave the engine.
    name: String,
    out: mpsc::UnboundedSender<String>,
    incoming: mpsc::UnboundedReceiver<Incoming>,
    /// How long the transcript is as the engine last set it. Lines past it
    /// were added by the window and go to the engine as notes.
    engine_len: usize,
}

impl EngineLink {
    /// Send a message. `false` once the connection is gone.
    pub fn send(&self, msg: &ClientMsg) -> bool {
        self.out.send(proto::encode(msg)).is_ok()
    }
}

/// Connect to the engine at `addr` and say hello. Returns the link and the
/// full view the engine answers with.
pub async fn open(addr: &Address, client: &str) -> Result<(EngineLink, View), String> {
    let mut conn = connect(addr)
        .await
        .map_err(|e| format!("cannot reach the engine at {addr}: {e}"))?;
    let hello = ClientMsg::Hello {
        version: PROTOCOL_VERSION,
        client: client.to_string(),
    };
    conn.send(&proto::encode(&hello))
        .await
        .map_err(|e| format!("cannot talk to the engine: {e}"))?;
    let mut greeted = false;
    let view = loop {
        let line = tokio::time::timeout(START_WAIT, conn.recv())
            .await
            .map_err(|_| "the engine did not answer in time".to_string())?
            .map_err(|e| format!("cannot read from the engine: {e}"))?
            .ok_or_else(|| "the engine closed the connection".to_string())?;
        match proto::decode_engine(&line)? {
            EngineMsg::Hello { .. } => greeted = true,
            EngineMsg::View { view } if greeted => break *view,
            EngineMsg::Error { message } => return Err(message),
            _ => {}
        }
    };

    let (mut reader, mut writer) = conn.split();
    let (in_tx, incoming) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            let why = match reader.recv().await {
                Ok(Some(line)) => match proto::decode_engine(&line) {
                    Ok(msg) => {
                        if in_tx.send(Incoming::Msg(msg)).is_err() {
                            return;
                        }
                        continue;
                    }
                    Err(e) => e,
                },
                Ok(None) => "it closed the connection".to_string(),
                Err(e) => e.to_string(),
            };
            let _ = in_tx.send(Incoming::Lost(why));
            return;
        }
    });
    let (out, mut queue) = mpsc::unbounded_channel::<String>();
    tokio::spawn(async move {
        while let Some(line) = queue.recv().await {
            if writer.send(&line).await.is_err() {
                return;
            }
        }
    });
    Ok((
        EngineLink {
            name: client.to_string(),
            out,
            incoming,
            engine_len: 0,
        },
        view,
    ))
}

/// Connect to the engine at `addr`, or call `start` to start one and wait up
/// to `START_WAIT` for it to answer.
pub async fn connect_or_start(
    addr: &Address,
    start: &(dyn Fn() -> io::Result<()> + Send + Sync),
) -> Result<(EngineLink, View), String> {
    if let Ok(linked) = open(addr, &window_name()).await {
        return Ok(linked);
    }
    start().map_err(|e| format!("cannot start the engine: {e}"))?;
    let begun = Instant::now();
    loop {
        match open(addr, &window_name()).await {
            Ok(linked) => return Ok(linked),
            Err(why) if begun.elapsed() >= START_WAIT => return Err(why),
            Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}

/// Make `app` a window onto the engine behind `link`, starting from `view`.
/// The window keeps no status file and writes no task records: the engine
/// does both.
pub fn become_window(app: &mut App, mut link: EngineLink, view: View) {
    app.live = None;
    app.bytebot_store = None;
    app.messages.clear();
    view::apply(app, view);
    link.engine_len = app.messages.len();
    app.engine_link = Some(link);
}

/// What one frame took from the engine.
#[derive(Debug, Default)]
pub struct Frame {
    /// Views applied: the screen needs drawing.
    pub views: usize,
    /// Why the connection ended, when it did. The link is gone.
    pub lost: Option<String>,
}

/// Each frame in a window: lines the window added to the transcript go to
/// the engine as notes, then everything the engine sent is applied.
pub fn frame(app: &mut App) -> Frame {
    let mut done = Frame::default();
    if app.engine_link.is_none() {
        reconnected(app);
        if app.engine_link.is_some() {
            done.views += 1;
        }
    }
    let Some(mut link) = app.engine_link.take() else {
        return done;
    };
    if app.messages.len() > link.engine_len {
        for line in app.messages.split_off(link.engine_len) {
            link.send(&ClientMsg::Note {
                role: line.role,
                content: line.content,
            });
        }
    }
    while let Ok(incoming) = link.incoming.try_recv() {
        match incoming {
            Incoming::Msg(EngineMsg::View { view }) => {
                view::apply(app, *view);
                link.engine_len = app.messages.len();
                done.views += 1;
            }
            Incoming::Msg(EngineMsg::Error { message }) => {
                app.push_toast(crate::toast::ToastKind::Warning, message);
            }
            Incoming::Msg(msg) => {
                if let Some(said) = answered_elsewhere(&msg, &link.name) {
                    app.push_toast(crate::toast::ToastKind::Info, said);
                }
            }
            Incoming::Lost(why) => {
                done.lost = Some(why);
                return done;
            }
        }
    }
    app.engine_link = Some(link);
    done
}

/// What a window says when its engine went away.
pub fn lost_warning(why: &str) -> String {
    format!("the engine stopped ({why}); starting a new one")
}

/// How a window starts an engine: shared with the reconnecting task.
pub type Starter = std::sync::Arc<dyn Fn() -> io::Result<()> + Send + Sync>;

/// A reconnection running in the background, checked by `frame`.
pub type Reconnecting = tokio::sync::oneshot::Receiver<Result<(EngineLink, View), String>>;

/// After the engine went away: say so, and connect to a new one once, in the
/// background so the window keeps drawing and taking keys meanwhile. `frame`
/// picks up the result. When that fails too the window carries on with the
/// agent work itself.
pub fn recover(app: &mut App<'_>, addr: &Address, start: Starter, why: &str) {
    app.push_toast(crate::toast::ToastKind::Warning, lost_warning(why));
    let addr = addr.clone();
    let (done, waiting) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        // One outer limit, so a slow start and a slow answer cannot add up.
        let result = tokio::time::timeout(START_WAIT * 2, connect_or_start(&addr, &*start))
            .await
            .unwrap_or_else(|_| Err("no engine answered in time".to_string()));
        let _ = done.send(result);
    });
    app.engine_reconnect = Some(waiting);
}

/// Take in a finished reconnection, if one was running.
fn reconnected(app: &mut App) {
    let Some(waiting) = app.engine_reconnect.as_mut() else {
        return;
    };
    let result = match waiting.try_recv() {
        Ok(result) => result,
        Err(tokio::sync::oneshot::error::TryRecvError::Empty) => return,
        Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
            Err("the reconnection stopped".to_string())
        }
    };
    app.engine_reconnect = None;
    match result {
        Ok((link, view)) => become_window(app, link, view),
        Err(e) => {
            app.take_over_engine_work();
            app.push_toast(
                crate::toast::ToastKind::Warning,
                format!("no engine could be started ({e}); this window now runs the agent work itself and keeps its own records"),
            );
        }
    }
}

/// `engine::act` in a window: the engine's work goes over the link; a
/// command that runs in the window runs here. What the window shows changes
/// at once where waiting for the engine would let a key be pressed twice.
pub(super) fn act(app: &mut App, msg: ClientMsg, tx: &mpsc::UnboundedSender<String>) {
    if let ClientMsg::SubmitChat { prompt } = &msg {
        if !runs_in_engine(prompt) {
            super::handle(app, msg, tx, "terminal");
            return;
        }
    }
    match &msg {
        ClientMsg::AnswerApproval { .. } => {
            app.approval_ids.pop_front();
            app.remote_approvals.pop_front();
            app.approval_scroll = 0;
        }
        ClientMsg::AnswerQuestion { text, .. } if !text.trim().is_empty() => {
            app.question_id = None;
            app.question_text = None;
            app.bytebot_command.clear();
            app.bytebot_cursor = 0;
        }
        ClientMsg::EnqueueTask { .. } => {
            app.bytebot_command.clear();
            app.bytebot_cursor = 0;
        }
        _ => {}
    }
    let sent = app.engine_link.as_ref().is_some_and(|link| link.send(&msg));
    if !sent {
        app.push_toast(
            crate::toast::ToastKind::Warning,
            "the engine is not connected; that was not sent".to_string(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_slash_command_runs_in_exactly_one_place() {
        for command in crate::app::SLASH_COMMANDS {
            let engine = ENGINE_COMMANDS.contains(command);
            let window = WINDOW_COMMANDS.contains(command);
            assert!(
                engine != window,
                "{command} must be listed once, in ENGINE_COMMANDS or WINDOW_COMMANDS"
            );
            assert_eq!(runs_in_engine(&format!("{command} x")), engine, "{command}");
            assert_eq!(runs_in_engine(command), engine, "{command}");
        }
        for command in ENGINE_COMMANDS.iter().chain(WINDOW_COMMANDS) {
            assert!(
                crate::app::SLASH_COMMANDS.contains(command),
                "{command} is not a command"
            );
        }
    }

    #[test]
    fn a_window_is_told_who_else_answered() {
        use crate::engine::proto::WireAnswer;
        let me = "terminal 7";
        let approval = |by: &str, answer| EngineMsg::ApprovalResolved {
            id: 1,
            answer,
            by: by.into(),
        };
        assert_eq!(
            answered_elsewhere(&approval("terminal 9", WireAnswer::Allow), me).as_deref(),
            Some("answered in terminal 9: allow")
        );
        assert_eq!(
            answered_elsewhere(&approval("desktop", WireAnswer::AllowForSession), me).as_deref(),
            Some("answered in desktop: allow for the session")
        );
        assert_eq!(
            answered_elsewhere(&approval("terminal 9", WireAnswer::Deny), me).as_deref(),
            Some("answered in terminal 9: deny")
        );
        let question = |by: &str| EngineMsg::QuestionAnswered {
            id: 2,
            by: by.into(),
        };
        assert_eq!(
            answered_elsewhere(&question("terminal 9"), me).as_deref(),
            Some("answered in terminal 9")
        );
        // A window's own answers, and anything else, say nothing.
        assert_eq!(
            answered_elsewhere(&approval(me, WireAnswer::Allow), me),
            None
        );
        assert_eq!(answered_elsewhere(&question(me), me), None);
        let other = EngineMsg::Error {
            message: "x".into(),
        };
        assert_eq!(answered_elsewhere(&other, me), None);
    }

    #[test]
    fn a_window_names_itself_by_its_process() {
        assert_eq!(window_name(), format!("terminal {}", std::process::id()));
    }

    #[test]
    fn chat_runs_in_the_engine_even_when_it_starts_with_a_slash() {
        assert!(runs_in_engine("fix the failing test"));
        assert!(runs_in_engine("/usr/lib is missing"));
        assert!(runs_in_engine("  /bytebot write a note"));
        assert!(!runs_in_engine("/help"));
        assert!(!runs_in_engine("/nav files"));
    }
}
