//! The view of agent state a window draws (EN-2). The engine holds the
//! agent work; each window holds only the screen, and learns what to draw
//! from `view` messages: everything when it connects, then only what
//! changed since the last one.

use serde::{Deserialize, Deserializer, Serialize};

use crate::agent_tools::{ApprovalDraft, ApprovalRequest, PlanItem, ToolClass};
use crate::app::{App, UiMessage};
use crate::bytebot_tasks::ByteBotTask;
use crate::engine::proto::ApprovalView;
use xencode_models_rs::llamacpp::LlamaCppTimings;

/// A ByteBot task as a window sees it, with the "made in this session" flag
/// that the task record on disk never carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewTask {
    #[serde(flatten)]
    pub task: ByteBotTask,
    pub this_session: bool,
}

/// Agent state for a window. The transcript from `messages_from` on is
/// replaced by `messages`; every other field is `None` when it did not
/// change.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct View {
    pub messages_from: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<UiMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generating: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytebot_running: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytebot_steps: Option<Vec<(String, String)>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytebot_log: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytebot_progress: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tasks: Option<Vec<ViewTask>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approvals: Option<Vec<ApprovalView>>,
    /// `Some(None)` says the question was answered or withdrawn.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present"
    )]
    pub question: Option<Option<(u64, String)>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<Vec<PlanItem>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_calls: Option<u64>,
    /// The team of worker agents (TM).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub team: Option<Vec<xencode_team_rs::WorkerSnapshot>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present"
    )]
    pub timings: Option<Option<LlamaCppTimings>>,
}

/// A field that is present, even as `null`, is a change: `Some(None)`.
fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// Everything but the transcript, as last sent.
#[derive(Clone, PartialEq)]
struct Agent {
    generating: bool,
    bytebot_running: bool,
    bytebot_steps: Vec<(String, String)>,
    bytebot_log: Vec<String>,
    bytebot_progress: f64,
    tasks: Vec<ViewTask>,
    approvals: Vec<ApprovalView>,
    question: Option<(u64, String)>,
    plan: Vec<PlanItem>,
    model: String,
    llm_calls: u64,
    timings: Option<LlamaCppTimings>,
    team: Vec<xencode_team_rs::WorkerSnapshot>,
}

impl Agent {
    fn read(app: &App) -> Agent {
        Agent {
            generating: app.is_generating,
            bytebot_running: app.bytebot_running,
            bytebot_steps: app.bytebot_steps.clone(),
            bytebot_log: app.bytebot_log.clone(),
            bytebot_progress: app.bytebot_progress,
            tasks: app
                .bytebot_tasks
                .iter()
                .map(|task| ViewTask {
                    // The flag travels beside the task, never inside it.
                    task: ByteBotTask {
                        this_session: false,
                        ..task.clone()
                    },
                    this_session: task.this_session,
                })
                .collect(),
            approvals: app
                .approval_queue
                .iter()
                .zip(app.approval_ids.iter())
                .map(|((request, _), id)| super::approval_view(*id, request))
                .collect(),
            question: match (app.question_id, &app.question_text) {
                (Some(id), Some(text)) => Some((id, text.clone())),
                _ => None,
            },
            plan: crate::agent_tools::plan_items(&app.agent_plan),
            model: app.config.default_model.clone(),
            llm_calls: app.total_llm_calls,
            timings: app.last_llamacpp_timings.clone(),
            team: app.team.snapshots(),
        }
    }
}

/// Remembers what was last sent to one window, so the next view carries
/// only the difference.
#[derive(Default)]
pub struct Watcher {
    sent: Vec<UiMessage>,
    last: Option<Agent>,
}

impl Watcher {
    /// Everything, for a window that just connected.
    pub fn full(&mut self, app: &App) -> View {
        let agent = Agent::read(app);
        self.sent = app.messages.clone();
        let view = View {
            messages_from: 0,
            messages: app.messages.clone(),
            generating: Some(agent.generating),
            bytebot_running: Some(agent.bytebot_running),
            bytebot_steps: Some(agent.bytebot_steps.clone()),
            bytebot_log: Some(agent.bytebot_log.clone()),
            bytebot_progress: Some(agent.bytebot_progress),
            tasks: Some(agent.tasks.clone()),
            approvals: Some(agent.approvals.clone()),
            question: Some(agent.question.clone()),
            plan: Some(agent.plan.clone()),
            model: Some(agent.model.clone()),
            llm_calls: Some(agent.llm_calls),
            timings: Some(agent.timings.clone()),
            team: Some(agent.team.clone()),
        };
        self.last = Some(agent);
        view
    }

    /// What changed since the last view, or `None` when nothing did.
    pub fn changes(&mut self, app: &App) -> Option<View> {
        let Some(last) = self.last.take() else {
            return Some(self.full(app));
        };
        let now = Agent::read(app);
        let mut view = View::default();
        let mut changed = false;

        let common = self
            .sent
            .iter()
            .zip(app.messages.iter())
            .take_while(|(old, new)| old == new)
            .count();
        if common < self.sent.len() || common < app.messages.len() {
            changed = true;
            view.messages_from = common;
            view.messages = app.messages[common..].to_vec();
            self.sent.truncate(common);
            self.sent.extend(view.messages.iter().cloned());
        } else {
            view.messages_from = app.messages.len();
        }

        macro_rules! field {
            ($name:ident) => {
                if now.$name != last.$name {
                    view.$name = Some(now.$name.clone());
                    changed = true;
                }
            };
        }
        field!(generating);
        field!(bytebot_running);
        field!(bytebot_steps);
        field!(bytebot_log);
        field!(bytebot_progress);
        field!(tasks);
        field!(approvals);
        field!(question);
        field!(plan);
        field!(model);
        field!(llm_calls);
        field!(timings);
        field!(team);

        self.last = Some(now);
        changed.then_some(view)
    }
}

/// Set a window's agent state from a view.
pub fn apply(app: &mut App, view: View) {
    let from = view.messages_from.min(app.messages.len());
    app.messages.truncate(from);
    app.messages.extend(view.messages);
    if let Some(value) = view.generating {
        app.is_generating = value;
    }
    if let Some(value) = view.team {
        app.team_view = value;
    }
    if let Some(value) = view.bytebot_running {
        app.bytebot_running = value;
    }
    if let Some(value) = view.bytebot_steps {
        app.bytebot_steps = value;
    }
    if let Some(value) = view.bytebot_log {
        app.bytebot_log = value;
    }
    if let Some(value) = view.bytebot_progress {
        app.bytebot_progress = value;
    }
    if let Some(tasks) = view.tasks {
        app.bytebot_tasks = tasks
            .into_iter()
            .map(
                |ViewTask {
                     mut task,
                     this_session,
                 }| {
                    task.this_session = this_session;
                    task
                },
            )
            .collect();
    }
    if let Some(approvals) = view.approvals {
        app.approval_ids = approvals.iter().map(|a| a.id).collect();
        app.remote_approvals = approvals
            .into_iter()
            .map(|a| {
                (
                    a.id,
                    ApprovalRequest {
                        tool: a.tool,
                        class: ToolClass::from_overlay_label(&a.class),
                        summary: a.summary,
                        preview: a.preview,
                        draft: ApprovalDraft::default(),
                    },
                )
            })
            .collect();
    }
    if let Some(question) = view.question {
        app.question_id = question.as_ref().map(|(id, _)| *id);
        app.question_text = question.map(|(_, text)| text);
    }
    if let Some(plan) = view.plan {
        if let Ok(mut items) = app.agent_plan.lock() {
            *items = plan;
        }
    }
    if let Some(model) = view.model {
        app.config.default_model = model;
    }
    if let Some(value) = view.llm_calls {
        app.total_llm_calls = value;
    }
    if let Some(value) = view.timings {
        app.last_llamacpp_timings = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_tools::{ApprovalDraft, ApprovalRequest, PlanItem, PlanStatus, ToolClass};
    use crate::app::{App, UiMessage};
    use crate::bytebot_tasks::{ByteBotTask, TaskState};

    fn line(role: &str, content: &str) -> UiMessage {
        UiMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    fn busy_engine() -> App<'static> {
        let mut app = App::for_tests();
        app.messages = vec![line("user", "fix it"), line("assistant", "on it")];
        app.is_generating = true;
        app.bytebot_running = true;
        app.bytebot_steps = vec![("read_file a.rs".into(), "ok".into())];
        app.bytebot_log = vec!["started".into()];
        app.bytebot_progress = 0.5;
        app.bytebot_tasks = vec![ByteBotTask {
            id: "000000000001-000000001-0001".into(),
            text: "write a note".into(),
            model: "llamacpp:none".into(),
            state: TaskState::Running,
            steps: vec![("write_file note.md".into(), "ok".into())],
            changed_files: vec!["note.md".into()],
            question: None,
            note: None,
            turn: Some(2),
            created_at: 1,
            started_at: Some(2),
            ended_at: None,
            pid: Some(42),
            this_session: true,
        }];
        let (answer_tx, _answer_rx) = tokio::sync::oneshot::channel();
        app.approval_queue.push_back((
            ApprovalRequest {
                tool: "write_file".into(),
                class: ToolClass::Edit,
                summary: "write_file a.rs".into(),
                preview: "+hello".into(),
                draft: ApprovalDraft::default(),
            },
            answer_tx,
        ));
        app.approval_ids.push_back(7);
        app.question_id = Some(9);
        app.question_text = Some("Which port?".into());
        *app.agent_plan.lock().unwrap() = vec![PlanItem {
            text: "read the code".into(),
            status: PlanStatus::InProgress,
        }];
        app.config.default_model = "llamacpp:qwen".into();
        app.total_llm_calls = 3;
        app.last_llamacpp_timings = Some(xencode_models_rs::llamacpp::LlamaCppTimings {
            tokens_generated: 10,
            predicted_per_second: 25.0,
            ..Default::default()
        });
        app
    }

    #[test]
    fn a_full_view_applied_to_a_window_reproduces_the_agent_state() {
        let engine = busy_engine();
        let mut window = App::for_tests();
        apply(&mut window, Watcher::default().full(&engine));

        let texts = |app: &App| {
            app.messages
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(texts(&window), texts(&engine));
        assert!(window.is_generating);
        assert!(window.bytebot_running);
        assert_eq!(window.bytebot_steps, engine.bytebot_steps);
        assert_eq!(window.bytebot_log, engine.bytebot_log);
        assert_eq!(window.bytebot_progress, 0.5);
        assert_eq!(window.bytebot_tasks, engine.bytebot_tasks);
        assert!(window.bytebot_tasks[0].this_session);
        assert_eq!(window.question_id, Some(9));
        assert_eq!(window.question_text.as_deref(), Some("Which port?"));
        assert_eq!(
            crate::agent_tools::plan_items(&window.agent_plan),
            crate::agent_tools::plan_items(&engine.agent_plan)
        );
        assert_eq!(window.config.default_model, "llamacpp:qwen");
        assert_eq!(window.total_llm_calls, 3);
        assert_eq!(window.last_llamacpp_timings, engine.last_llamacpp_timings);

        let shown = window.pending_approval().expect("the approval is shown");
        assert_eq!(shown.tool, "write_file");
        assert_eq!(shown.class, ToolClass::Edit);
        assert_eq!(shown.summary, "write_file a.rs");
        assert_eq!(shown.preview, "+hello");
        assert_eq!(window.approval_ids.front(), Some(&7));
    }

    #[test]
    fn a_streamed_reply_sends_only_its_own_line() {
        let mut engine = App::for_tests();
        engine.messages = (0..200)
            .map(|i| line("assistant", &format!("line {i}")))
            .collect();
        let mut watcher = Watcher::default();
        watcher.full(&engine);

        engine.messages[199].content.push_str(" and more");
        let view = watcher.changes(&engine).expect("the growing line is sent");
        assert_eq!(view.messages_from, 199);
        assert_eq!(view.messages.len(), 1);
        assert_eq!(view.messages[0].content, "line 199 and more");
        assert!(view.generating.is_none() && view.tasks.is_none() && view.model.is_none());
    }

    #[test]
    fn nothing_changed_sends_nothing() {
        let engine = busy_engine();
        let mut watcher = Watcher::default();
        watcher.full(&engine);
        assert!(watcher.changes(&engine).is_none());
    }

    #[test]
    fn a_shortened_transcript_shortens_the_window() {
        let mut engine = App::for_tests();
        engine.messages = (0..20).map(|i| line("user", &format!("{i}"))).collect();
        let mut watcher = Watcher::default();
        let mut window = App::for_tests();
        apply(&mut window, watcher.full(&engine));

        engine.messages.truncate(5);
        let view = watcher.changes(&engine).expect("the cut is sent");
        assert_eq!((view.messages_from, view.messages.len()), (5, 0));
        apply(&mut window, view);
        assert_eq!(window.messages.len(), 5);
    }

    #[test]
    fn a_withdrawn_question_and_answered_approval_leave_the_window() {
        let mut engine = busy_engine();
        let mut watcher = Watcher::default();
        let mut window = App::for_tests();
        apply(&mut window, watcher.full(&engine));

        engine.question_id = None;
        engine.question_text = None;
        engine.approval_queue.clear();
        engine.approval_ids.clear();
        apply(
            &mut window,
            watcher.changes(&engine).expect("both are sent"),
        );
        assert_eq!(window.question_id, None);
        assert!(window.pending_approval().is_none());
        assert!(window.approval_ids.is_empty());
    }

    #[test]
    fn a_view_travels_as_one_line() {
        let view = Watcher::default().full(&busy_engine());
        let line = crate::engine::proto::encode(&crate::engine::proto::EngineMsg::View {
            view: Box::new(view.clone()),
        });
        assert!(!line.contains('\n'));
        match crate::engine::proto::decode_engine(&line).unwrap() {
            crate::engine::proto::EngineMsg::View { view: back } => assert_eq!(*back, view),
            other => panic!("came back as {other:?}"),
        }
    }
}
