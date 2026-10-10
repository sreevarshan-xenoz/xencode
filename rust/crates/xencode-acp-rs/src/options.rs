//! The model choice and the command list an editor shows for xencode
//! (M-7c).

use std::time::Duration;

use agent_client_protocol::schema::v1::{
    AvailableCommand, SessionConfigOption, SessionConfigOptionCategory, SessionConfigOptionValue,
    SessionConfigSelectOption,
};
use xencode_tui_rs::engine::link::{LinkEvent, ENGINE_COMMANDS, WINDOW_COMMANDS};
use xencode_tui_rs::engine::proto::{ClientMsg, EngineMsg};

use crate::session::Session;

/// The one option xencode offers.
pub const MODEL_OPTION: &str = "model";

/// What each command that runs in the engine does, in one sentence.
const COMMAND_HELP: &[(&str, &str)] = &[
    (
        "/bytebot",
        "Run a task with ByteBot: it works step by step and asks for review",
    ),
    ("/spawn", "Start a helper agent on a separate task"),
    ("/rewind", "Undo the last turn's file changes"),
    ("/gate", "Open or close a tool gate for this session"),
    ("/plan", "Show or change the plan"),
    ("/lesson", "Record a lesson for later sessions"),
    ("/ctx", "Show what is in the model's context"),
    ("/model", "Switch the model"),
    ("/trust", "Change which tools run without asking"),
    ("/mcp", "Connect or list tool servers"),
    ("/plugin", "Manage plugins"),
    ("/skills", "List or load skills"),
];

/// The commands an editor offers after `/`: those that run in the engine.
/// Commands that draw a terminal panel are left out.
pub fn commands() -> Vec<AvailableCommand> {
    ENGINE_COMMANDS
        .iter()
        .map(|name| {
            let help = COMMAND_HELP
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, h)| *h)
                .unwrap_or("A xencode command");
            AvailableCommand::new(name.trim_start_matches('/'), help)
        })
        .collect()
}

/// Why a prompt cannot run here, when it starts with a command that draws a
/// terminal panel.
pub fn window_only(prompt: &str) -> Option<String> {
    let word = prompt.split_whitespace().next()?;
    WINDOW_COMMANDS
        .contains(&word)
        .then(|| format!("`{word}` draws a terminal panel; run it in `xencode tui`"))
}

/// The model option: `current` selected among `models`, which always
/// includes it.
pub fn model_option(current: &str, models: &[String]) -> SessionConfigOption {
    let mut names: Vec<String> = models.to_vec();
    if !current.is_empty() && !names.iter().any(|m| m == current) {
        names.push(current.to_string());
    }
    let choices: Vec<SessionConfigSelectOption> = names
        .iter()
        .map(|m| SessionConfigSelectOption::new(m.clone(), m.clone()))
        .collect();
    SessionConfigOption::select(MODEL_OPTION, "Model", current.to_string(), choices)
        .category(SessionConfigOptionCategory::Model)
}

/// The models to offer, as the terminal's Models screen finds them.
pub async fn models() -> Vec<String> {
    match xencode_config_rs::XencodeConfig::load() {
        Ok(config) => xencode_tui_rs::app::discover_models(&config).await,
        Err(_) => Vec::new(),
    }
}

/// Change the engine's model to `value`, waiting until the engine says it
/// changed. Refused while a turn holds the engine link.
pub async fn set_model(
    state: &tokio::sync::Mutex<Session>,
    value: &SessionConfigOptionValue,
) -> Result<String, String> {
    let SessionConfigOptionValue::ValueId { value } = value else {
        return Err("the model option takes a model name".to_string());
    };
    let name = value.to_string();
    // The link is taken out for the switch, as a turn does, so the
    // session's lock is not held while the engine works: a message or a
    // stop for this session is answered at once in the meantime.
    let mut link = {
        let mut s = state.lock().await;
        if s.busy {
            return Err(
                "a turn is running in this session; change the model when it ends".to_string(),
            );
        }
        match s.link.take() {
            Some(link) => {
                s.busy = true;
                link
            }
            None => {
                return Err("the engine was lost; send a message to reconnect".to_string());
            }
        }
    };
    let changed = if link.send(&ClientMsg::SetModel { name: name.clone() }) {
        // Switching can take seconds: the engine first tries to start the
        // llama.cpp server for a llama.cpp model.
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                match link.next().await {
                    LinkEvent::Msg(EngineMsg::View { view })
                        if view.model.as_deref() == Some(&name) =>
                    {
                        return Ok(())
                    }
                    LinkEvent::Msg(EngineMsg::Error { message }) => return Err(message),
                    LinkEvent::Msg(_) => {}
                    LinkEvent::Lost(why) => return Err(format!("the engine was lost: {why}")),
                }
            }
        })
        .await
        .unwrap_or_else(|_| Err("the engine did not confirm the new model".to_string()))
    } else {
        Err("the engine was lost; send a message to reconnect".to_string())
    };
    let mut s = state.lock().await;
    s.busy = false;
    if changed
        .as_ref()
        .is_err_and(|why| why.starts_with("the engine was lost"))
    {
        s.link = None;
    } else {
        s.link = Some(link);
    }
    if changed.is_ok() {
        s.model = name;
    }
    changed.map(|()| s.model.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_current_model_is_always_a_choice() {
        let option = model_option("llamacpp:mine", &["qwen3:8b".to_string()]);
        let json = serde_json::to_value(&option).unwrap();
        assert_eq!(json["currentValue"], "llamacpp:mine");
        let values: Vec<&str> = json["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["value"].as_str().unwrap())
            .collect();
        assert_eq!(values, vec!["qwen3:8b", "llamacpp:mine"]);
        assert_eq!(json["category"], "model");
    }

    #[test]
    fn panel_commands_are_refused_and_engine_commands_offered() {
        assert!(window_only("/init").unwrap().contains("xencode tui"));
        assert!(window_only("/advise this").is_some());
        assert_eq!(window_only("/bytebot write a note"), None);
        assert_eq!(window_only("fix the bug"), None);
        let names: Vec<String> = commands().into_iter().map(|c| c.name).collect();
        assert!(names.contains(&"bytebot".to_string()));
        assert!(!names.iter().any(|n| n.starts_with('/')));
    }
}
