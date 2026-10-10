//! The lead agent's tools (TM-2): `xencode mcp serve --team` publishes seven
//! `team_*` tools that pass the lead's requests to the project's engine, which
//! runs the workers. The server connects to the engine as one more window
//! (`mcp <pid>`), starting it when none runs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Map, Value};
use xencode_providers_rs::ToolDefinition;

use crate::engine::address::Address;
use crate::engine::link::{self, EngineLink, LinkEvent};
use crate::engine::proto::{ClientMsg, EngineMsg, TeamRequest};

/// How long an ordinary team request may take.
const WAIT: Duration = Duration::from_secs(30);
/// A merge runs the project's checks first.
/// How long `team_merge` waits for a merge: every check may run to its
/// time limit (`.xencode/team.toml`), plus five minutes for git.
fn merge_wait(root: &Path) -> Duration {
    let settings = xencode_team_rs::merge::team_settings(root);
    let checks = settings.checks.len().max(1) as u64;
    let each = settings.check_timeout_secs.unwrap_or(1200);
    Duration::from_secs(checks.saturating_mul(each).saturating_add(300))
}

/// The published tool names, in this order.
pub const TEAM_TOOLS: [&str; 7] = [
    "team_agents",
    "team_start",
    "team_status",
    "team_result",
    "team_message",
    "team_stop",
    "team_merge",
];

/// What each team tool is, for the lead.
pub fn team_tool_definitions() -> Vec<ToolDefinition> {
    let id = json!({"type": "string", "description": "The worker's id, such as w1"});
    let def =
        |name: &str, description: &str, properties: Value, required: &[&str]| ToolDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters: json!({"type": "object", "properties": properties, "required": required}),
        };
    vec![
        def("team_agents", "List the worker agents this machine can start, and how each would sign in.", json!({}), &[]),
        def(
            "team_start",
            "Start a worker agent on a task, in its own git worktree off the base branch (default: the branch checked out). Returns the worker's id.",
            json!({
                "agent": {"type": "string", "description": "Which agent: see team_agents"},
                "task": {"type": "string", "description": "What the worker should do, in full"},
                "base": {"type": "string", "description": "The branch to start from (optional)"}
            }),
            &["agent", "task"],
        ),
        def(
            "team_status",
            "A worker's state (starting, working, needs_you, done, failed, stopped), last message, tool calls and cost; without an id, every worker.",
            json!({"id": id}),
            &[],
        ),
        def("team_result", "A worker's latest answer and its diff against the base branch.", json!({"id": id}), &["id"]),
        def(
            "team_message",
            "Send a follow-up to a worker whose turn is done; it continues in the same session.",
            json!({"id": id, "text": {"type": "string"}}),
            &["id", "text"],
        ),
        def("team_stop", "Stop a worker; its worktree is kept.", json!({"id": id}), &["id"]),
        def(
            "team_merge",
            "Merge a worker's branch into the base branch, landing it only if the project's checks pass on the merged result.",
            json!({"id": id}),
            &["id"],
        ),
    ]
}

/// The team request a tool call stands for.
pub fn request_for(name: &str, args: &Map<String, Value>) -> Result<TeamRequest, String> {
    let text = |key: &str| -> Result<String, String> {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("{name} needs `{key}`"))
    };
    let optional = |key: &str| args.get(key).and_then(Value::as_str).map(str::to_string);
    Ok(match name {
        "team_agents" => TeamRequest::Agents,
        "team_start" => TeamRequest::Start {
            agent: text("agent")?,
            task: text("task")?,
            base: optional("base"),
        },
        "team_status" => TeamRequest::Status { id: optional("id") },
        "team_result" => TeamRequest::Result { id: text("id")? },
        "team_message" => TeamRequest::Message {
            id: text("id")?,
            text: text("text")?,
        },
        "team_stop" => TeamRequest::Stop { id: text("id")? },
        "team_merge" => TeamRequest::Merge { id: text("id")? },
        other => return Err(format!("no team tool called `{other}`")),
    })
}

/// The worker agents of the project's engine, when one runs; `None` when
/// none does. Nothing is started to answer: a reading never starts work.
pub async fn running_team(root: &Path) -> Option<Vec<xencode_team_rs::WorkerSnapshot>> {
    let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let addr = Address::for_project(&root).ok()?;
    let (link, view) = link::open(&addr, &format!("reader {}", std::process::id()))
        .await
        .ok()?;
    let _ = link.send(&crate::engine::proto::ClientMsg::Goodbye);
    Some(view.team.unwrap_or_default())
}

/// The server's connection to the project's engine, made on first use.
pub struct TeamClient {
    root: PathBuf,
    link: tokio::sync::Mutex<Option<EngineLink>>,
    next: AtomicU64,
}

impl TeamClient {
    pub fn new(root: &Path) -> Arc<TeamClient> {
        let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        Arc::new(TeamClient {
            root,
            link: tokio::sync::Mutex::new(None),
            next: AtomicU64::new(0),
        })
    }

    /// Carry out one tool call: its text, or why it failed in words.
    pub async fn call(&self, name: &str, args: &Map<String, Value>) -> Result<String, String> {
        let request = request_for(name, args)?;
        if let TeamRequest::Merge { id } = &request {
            return self.merge(id.clone()).await;
        }
        let body = self.ask(request, WAIT).await?;
        Ok(serde_json::to_string_pretty(&body).unwrap_or_else(|_| body.to_string()))
    }

    /// Start a merge and wait for its outcome: landed is a success, anything
    /// else is said as a failed call with the reason.
    async fn merge(&self, id: String) -> Result<String, String> {
        self.ask(TeamRequest::Merge { id: id.clone() }, WAIT)
            .await?;
        let start = std::time::Instant::now();
        loop {
            let status = self
                .ask(
                    TeamRequest::Status {
                        id: Some(id.clone()),
                    },
                    WAIT,
                )
                .await?;
            if let Some(finished) = status.get("merge").and_then(|m| m.get("finished")) {
                let text = serde_json::to_string_pretty(finished).unwrap_or_default();
                return if finished.get("outcome").and_then(Value::as_str) == Some("landed") {
                    Ok(text)
                } else {
                    Err(text)
                };
            }
            let wait = merge_wait(&self.root);
            if start.elapsed() > wait {
                return Err(format!(
                    "{id}'s merge did not finish within {} s; it is still running, and \
                     team_status shows its outcome when it ends",
                    wait.as_secs()
                ));
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Send one request to the engine and return its answer.
    pub async fn request(&self, request: TeamRequest) -> Result<Value, String> {
        self.ask(request, WAIT).await
    }

    async fn ask(&self, request: TeamRequest, wait: Duration) -> Result<Value, String> {
        let mut guard = self.link.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        let link = guard.as_mut().expect("connected above");
        // What arrived since the last call belongs to other windows' turns.
        if link.drain().is_err() {
            *guard = Some(self.connect().await?);
        }
        let link = guard.as_mut().expect("connected above");
        let req = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        if !link.send(&ClientMsg::Team { req, request }) {
            *guard = None;
            return Err("the engine connection was lost; try again".to_string());
        }
        let answer = tokio::time::timeout(wait, async {
            loop {
                match link.next().await {
                    LinkEvent::Msg(EngineMsg::TeamReply { req: r, ok, body }) if r == req => {
                        return Ok((ok, body))
                    }
                    LinkEvent::Msg(_) => {}
                    LinkEvent::Lost(why) => return Err(format!("the engine was lost: {why}")),
                }
            }
        })
        .await
        .map_err(|_| format!("the engine did not answer within {} s", wait.as_secs()))?;
        match answer {
            Ok((true, body)) => Ok(body),
            Ok((false, body)) => Err(body
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| body.to_string())),
            Err(why) => {
                *guard = None;
                Err(why)
            }
        }
    }

    async fn connect(&self) -> Result<EngineLink, String> {
        let addr = Address::for_project(&self.root)?;
        let folder = self.root.to_string_lossy().to_string();
        let start: link::Starter = Arc::new(move || -> std::io::Result<()> {
            let exe = std::env::current_exe()?;
            xencode_live_rs::spawn_detached(&exe, &["engine", "--project", &folder]).map(|_| ())
        });
        let name = format!("mcp {}", std::process::id());
        link::connect_or_start_as(&addr, &*start, &name)
            .await
            .map(|(link, _view)| link)
            .map_err(|why| format!("cannot reach the engine for {}: {why}", self.root.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review: `team_merge` waits as long as the checks may run, not a fixed
    /// time shorter than them.
    #[test]
    fn the_merge_wait_covers_every_check_running_to_its_limit() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(merge_wait(dir.path()), Duration::from_secs(1500));
        std::fs::create_dir_all(dir.path().join(".xencode")).unwrap();
        std::fs::write(
            dir.path().join(".xencode").join("team.toml"),
            "checks = [\"a\", \"b\", \"c\"]\ncheck_timeout_secs = 1200\n",
        )
        .unwrap();
        assert_eq!(merge_wait(dir.path()), Duration::from_secs(3 * 1200 + 300));
    }
}
