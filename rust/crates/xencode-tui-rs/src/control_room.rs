//! The orchestrator control room (`X-3`).
//!
//! This is the control room as a *projection*, not a fifth panel. It reads two
//! things and nothing else: the normalised [`AgentEvent`] model every vendor
//! adapter produces (`AR-9`/`AR-10`), and the scheduler's own record of what it
//! ran (`OR-2`'s [`ScheduleReport`]). It never touches a raw vendor stream, a
//! process handle, or a pipe — those live behind the adapters and the launch
//! path, and by the time an event reaches here it is already a value in the
//! common model. That is the whole point: a change to how one vendor formats
//! its output cannot break this surface, because this surface is written only
//! against the event variants, not against any vendor's bytes.
//!
//! Two rules hold it honest.
//!
//! - **A worker's pane is named by its task and agent, never by where it sits.**
//!   Position is what the layout tree (`V-1`) reshuffles; the task is what the
//!   operator is actually tracking. Every projection output carries an
//!   [`WorkerRef`] so a card, a timeline row and an approval all say *"codex —
//!   split the retrieval tier"* regardless of ordering.
//! - **A number is traced or it is unknown, never idle** (`OR-12`'s rule). When
//!   the events prove a count, it is shown; when the run simply never reported
//!   something — a vendor that does not emit permission requests at all, a node
//!   the scheduler has not timed — it renders as [`Trace::Unknown`], because "we
//!   were never told" and "we looked and found nothing" are different facts, and
//!   collapsing them onto zero would put a clean number on the screen that no
//!   evidence backs.

use xencode_agents_rs::protocol::{AgentEvent, Origin};
use xencode_core_rs::scheduler::ScheduleReport;

/// Which worker a projection row belongs to. Carried on every output so a pane
/// is identified by task and agent, not by position in a list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRef {
    /// The agent's own name, as the roster knows it — `codex`, `claude`.
    pub agent: String,
    /// The task it was launched on, in words the operator wrote.
    pub task: String,
    /// The scheduler node this worker maps to, if any. Timing comes from here;
    /// a worker with no node has no recorded duration.
    pub node: Option<String>,
}

impl WorkerRef {
    /// The label shown on the pane: agent first, then task. `"codex — split the
    /// retrieval tier"`. It is the same string no matter where the layout tree
    /// places the pane.
    pub fn label(&self) -> String {
        format!("{} — {}", self.agent, self.task)
    }
}

/// A projected number: backed by a stored event or the scheduler's own record,
/// or honestly unknown. See the module docs — an unobserved value is `Unknown`,
/// never a `0` that reads as "checked and empty".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Trace<T> {
    Known(T),
    Unknown,
}

impl<T> Trace<T> {
    /// The value if there is one, `None` if unknown. Lets a caller fall back to
    /// a rendered `—` instead of inventing a number.
    pub fn value(&self) -> Option<&T> {
        match self {
            Trace::Known(v) => Some(v),
            Trace::Unknown => None,
        }
    }

    /// Whether the screen may print a real number here.
    pub fn is_known(&self) -> bool {
        matches!(self, Trace::Known(_))
    }
}

/// A worker's state as the events describe it. Derived only from event variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardStatus {
    /// It finished. `outcome` is whatever the terminal event carried, if it
    /// carried one.
    Completed { outcome: Option<String> },
    /// It hit an error. Carries the event's message.
    Failed { message: String },
    /// It has begun and not ended.
    Running,
    /// Nothing has been reported about it — no events at all. Not "idle": we
    /// have simply never been told it exists in a stream.
    Unknown,
}

/// One fleet card per worker — the headline state the operator scans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FleetCard {
    pub worker: WorkerRef,
    pub status: CardStatus,
    /// Prose the worker emitted, counted from the stream.
    pub messages: usize,
    /// Files that changed, taken from `FileChanged` — which xencode populates
    /// from its own diff, never from a worker's claim.
    pub files_changed: Vec<String>,
    /// Tools that started and have not produced output yet.
    pub tools_in_flight: usize,
    /// Whether a human is being waited on. Unknown for a run whose vendor never
    /// raises permissions and which has not yet ended.
    pub needs_approval: Trace<bool>,
    /// Wall-clock duration, only when the scheduler recorded both ends. The
    /// event model carries no timestamps, so this is `Unknown` unless the
    /// matching node finished.
    pub elapsed_ms: Trace<u64>,
    /// Whether any figure on this card was filled by xencode's own bookkeeping
    /// rather than witnessed in the worker's stream. A card built entirely from
    /// synthesised events is not evidence the worker said any of it.
    pub partly_synthesised: bool,
}

/// A single point on a worker's timeline: the event's kind and a short detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineEntry {
    pub worker: WorkerRef,
    /// The normalised event name (`message`, `tool_started`, …). This is the
    /// only view of the event the surface keeps — never the raw vendor line.
    pub kind: &'static str,
    pub detail: String,
}

/// A permission the worker raised that has not been resolved by the run ending.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingApproval {
    pub worker: WorkerRef,
    pub tool: String,
    pub call_id: Option<String>,
}

/// The task-graph projection over scheduler state — launch order, per-node
/// status, and the numbers `OR-2` computed. Every value here traces to the
/// report; nothing is inferred from a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphView {
    pub launch_order: Vec<String>,
    /// `(node id, status label)` for every node the scheduler recorded an
    /// outcome for.
    pub nodes: Vec<(String, String)>,
    pub binding: &'static str,
    pub peak_concurrency: usize,
    pub capacity: usize,
    pub elapsed_ms: u64,
    pub bottleneck: Option<String>,
}

/// One worker's event stream, as the launch path hands it over after
/// normalisation.
pub struct Stream<'a> {
    pub worker: WorkerRef,
    pub events: &'a [AgentEvent],
}

/// The control room. A pure function of events plus scheduler state; it holds no
/// process handle and reads no vendor output.
pub struct ControlRoom<'a> {
    streams: Vec<Stream<'a>>,
    report: Option<&'a ScheduleReport>,
}

impl<'a> ControlRoom<'a> {
    pub fn new(streams: Vec<Stream<'a>>, report: Option<&'a ScheduleReport>) -> Self {
        Self { streams, report }
    }

    /// One card per worker, in the order given. The card is named by its
    /// [`WorkerRef`], so reordering the input reorders the list but never
    /// re-identifies a pane.
    pub fn fleet(&self) -> Vec<FleetCard> {
        self.streams
            .iter()
            .map(|s| {
                let mut messages = 0usize;
                let mut files_changed: Vec<String> = Vec::new();
                let mut started_tools = 0usize;
                let mut output_tools = 0usize;
                let mut status = CardStatus::Unknown;
                let mut needs_approval = Trace::Unknown;
                let mut partly_synthesised = false;

                for ev in s.events {
                    if ev.origin() == Origin::Synthesised {
                        partly_synthesised = true;
                    }
                    match ev {
                        AgentEvent::Message { .. } => messages += 1,
                        AgentEvent::FileChanged { path, .. } => {
                            if !files_changed.iter().any(|f| f == path) {
                                files_changed.push(path.clone());
                            }
                        }
                        AgentEvent::ToolStarted { .. } => started_tools += 1,
                        AgentEvent::ToolOutput { .. } => output_tools += 1,
                        AgentEvent::PermissionRequested { .. } => {
                            needs_approval = Trace::Known(true);
                        }
                        AgentEvent::PermissionDenied { .. } => {
                            needs_approval = Trace::Known(false);
                        }
                        AgentEvent::Error { message, .. } => {
                            status = CardStatus::Failed {
                                message: message.clone(),
                            };
                        }
                        AgentEvent::Completed { outcome, .. } => {
                            status = CardStatus::Completed {
                                outcome: outcome.clone(),
                            };
                            // A finished run cannot still be awaiting a human.
                            needs_approval = Trace::Known(false);
                        }
                        AgentEvent::SessionEnded { .. } => {
                            // The session closed. If nothing later re-opened it,
                            // there is no live approval either.
                            needs_approval = Trace::Known(false);
                        }
                        AgentEvent::SessionStarted { .. } | AgentEvent::ToolRequested { .. } => {
                            if status == CardStatus::Unknown {
                                status = CardStatus::Running;
                            }
                        }
                    }
                }

                let elapsed_ms = match (s.worker.node.as_deref(), self.report) {
                    (Some(id), Some(r)) => r
                        .outcome(id)
                        .filter(|o| o.finished_ms > 0 && o.started_ms > 0)
                        .map(|o| Trace::Known(o.finished_ms.saturating_sub(o.started_ms)))
                        .unwrap_or(Trace::Unknown),
                    _ => Trace::Unknown,
                };

                FleetCard {
                    worker: s.worker.clone(),
                    status,
                    messages,
                    files_changed,
                    tools_in_flight: started_tools.saturating_sub(output_tools),
                    needs_approval,
                    elapsed_ms,
                    partly_synthesised,
                }
            })
            .collect()
    }

    /// Every event across every worker, each tagged with the worker it came
    /// from. Rows are labelled by task-and-agent, not by stream position.
    pub fn timeline(&self) -> Vec<TimelineEntry> {
        let mut out = Vec::new();
        for s in &self.streams {
            for ev in s.events {
                out.push(TimelineEntry {
                    worker: s.worker.clone(),
                    kind: ev.name(),
                    detail: detail_of(ev),
                });
            }
        }
        out
    }

    /// The permissions still outstanding: raised, and not closed out by the run
    /// completing or ending. Each carries its worker's name.
    pub fn pending_approvals(&self) -> Vec<PendingApproval> {
        let mut out = Vec::new();
        for s in &self.streams {
            let ended = s.events.iter().any(|e| {
                matches!(
                    e,
                    AgentEvent::Completed { .. } | AgentEvent::SessionEnded { .. }
                )
            });
            if ended {
                continue;
            }
            let denied_tools: Vec<&str> = s
                .events
                .iter()
                .filter_map(|e| match e {
                    AgentEvent::PermissionDenied { tool, .. } => Some(tool.as_str()),
                    _ => None,
                })
                .collect();
            for ev in s.events {
                if let AgentEvent::PermissionRequested { tool, call_id, .. } = ev {
                    if !denied_tools.contains(&tool.as_str()) {
                        out.push(PendingApproval {
                            worker: s.worker.clone(),
                            tool: tool.clone(),
                            call_id: call_id.clone(),
                        });
                    }
                }
            }
        }
        out
    }

    /// The task graph, read straight from the scheduler's report. When there is
    /// no report the view is empty rather than fabricated.
    pub fn graph(&self) -> Option<GraphView> {
        let r = self.report?;
        Some(GraphView {
            launch_order: r.launch_order.clone(),
            nodes: r
                .outcomes
                .iter()
                .map(|(id, o)| (id.clone(), o.status.label()))
                .collect(),
            binding: r.binding.label(),
            peak_concurrency: r.peak_concurrency,
            capacity: r.capacity,
            elapsed_ms: r.elapsed_ms,
            bottleneck: r.bottleneck.clone(),
        })
    }
}

/// The short detail line for a timeline row. Only ever drawn from the normalised
/// event's own fields — never a raw vendor payload.
fn detail_of(ev: &AgentEvent) -> String {
    match ev {
        AgentEvent::Message { text, .. } => truncate(text),
        AgentEvent::ToolRequested { tool, .. }
        | AgentEvent::ToolStarted { tool, .. }
        | AgentEvent::PermissionRequested { tool, .. } => tool.clone(),
        AgentEvent::PermissionDenied { tool, reason, .. } => match reason {
            Some(r) => format!("{tool}: {r}"),
            None => format!("{tool}: denied"),
        },
        AgentEvent::ToolOutput { tool, output, .. } => match output {
            Some(o) => format!("{tool}: {}", truncate(o)),
            None => tool.clone(),
        },
        AgentEvent::FileChanged { path, .. } => path.clone(),
        AgentEvent::Error { message, .. } => truncate(message),
        AgentEvent::Completed { outcome, .. } => outcome.clone().unwrap_or_default(),
        AgentEvent::SessionStarted { session_id, .. } => session_id.clone().unwrap_or_default(),
        AgentEvent::SessionEnded { reason, .. } => reason.clone().unwrap_or_default(),
    }
}

fn truncate(s: &str) -> String {
    const CAP: usize = 120;
    if s.chars().count() <= CAP {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(CAP).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xencode_agents_rs::protocol::Origin;

    fn wr(agent: &str, task: &str, node: Option<&str>) -> WorkerRef {
        WorkerRef {
            agent: agent.into(),
            task: task.into(),
            node: node.map(str::to_string),
        }
    }

    fn msg(text: &str) -> AgentEvent {
        AgentEvent::Message {
            text: text.into(),
            origin: Origin::Observed,
        }
    }

    /// The grep-verifiable claim of the done-when, made into a test that watches
    /// itself: the only agents-rs import in this file's real code is the
    /// `protocol` module, and no process handle or vendor-output mechanism
    /// appears. A future edit that reaches for a pipe or a raw capture fails
    /// here before it can ship.
    #[test]
    fn the_projection_imports_only_the_protocol_and_nothing_that_reads_a_vendor() {
        let src = include_str!("control_room.rs");
        // Only the code, not this test module, which names the tokens to ban.
        let code = src.split("#[cfg(test)]").next().unwrap();

        for (n, line) in code.lines().enumerate() {
            if line.contains("xencode_agents_rs") {
                assert!(
                    line.contains("::protocol"),
                    "line {n} imports agents-rs outside the protocol: {line}"
                );
            }
        }
        for forbidden in [
            "std::process",
            "Command",
            ".output()",
            ".spawn",
            "PipeReader",
            "capture::",
            "probe::",
            "contract::",
            "envelope::",
            "reqwest",
            "serde_json::from_str",
        ] {
            assert!(
                !code.contains(forbidden),
                "the view layer must not touch {forbidden:?} — that would let a vendor change break it"
            );
        }
        // And it does use the protocol, or the test above would pass vacuously.
        assert!(code.contains("xencode_agents_rs::protocol"));
    }

    /// A pane is named by its task and agent, never by position: two workers
    /// keep their identities when the input order flips.
    #[test]
    fn a_pane_is_labelled_by_task_and_agent_not_where_it_sits() {
        let a = wr("codex", "split the retrieval tier", None);
        let b = wr("claude", "write the migration", None);
        let ev_a = vec![msg("on it")];
        let ev_b = vec![msg("working")];
        let forward = ControlRoom::new(
            vec![
                Stream {
                    worker: a.clone(),
                    events: &ev_a,
                },
                Stream {
                    worker: b.clone(),
                    events: &ev_b,
                },
            ],
            None,
        );
        let flipped = ControlRoom::new(
            vec![
                Stream {
                    worker: b.clone(),
                    events: &ev_b,
                },
                Stream {
                    worker: a.clone(),
                    events: &ev_a,
                },
            ],
            None,
        );
        let find = |cards: &[FleetCard], label: &str| {
            cards
                .iter()
                .find(|c| c.worker.label() == label)
                .cloned()
                .unwrap()
        };
        assert_eq!(a.label(), "codex — split the retrieval tier");
        // The card for `codex` is byte-identical whichever order it came in, so
        // the layout tree repositioning panes cannot change what they say.
        assert_eq!(
            find(&forward.fleet(), &a.label()),
            find(&flipped.fleet(), &a.label())
        );
    }

    /// A worker that never reported anything is unknown, not idle: no status, no
    /// duration, no approval verdict — every unobserved figure stays unknown,
    /// while the message count (a fact about the whole stream we did see) is a
    /// genuine zero.
    #[test]
    fn an_unreported_worker_is_unknown_never_idle() {
        let room = ControlRoom::new(
            vec![Stream {
                worker: wr("kiro-cli", "triage", None),
                events: &[],
            }],
            None,
        );
        let card = &room.fleet()[0];
        assert_eq!(card.status, CardStatus::Unknown, "no events, no status");
        assert!(
            card.needs_approval == Trace::Unknown,
            "never raised, never ended: unknown, not false"
        );
        assert!(card.elapsed_ms == Trace::Unknown, "no node, no timing");
        assert_eq!(card.messages, 0, "the stream we saw held zero messages");
    }

    /// Every figure the events *do* back is known and correct: messages counted,
    /// files taken from the diff, a running approval, and a duration that appears
    /// only because the scheduler timed the node.
    #[test]
    fn each_traced_number_maps_to_the_event_that_proves_it() {
        use xencode_core_rs::scheduler::{Binding, NodeOutcome};
        let events = vec![
            AgentEvent::SessionStarted {
                session_id: Some("s1".into()),
                origin: Origin::Observed,
            },
            msg("first"),
            msg("second"),
            AgentEvent::FileChanged {
                path: "src/a.rs".into(),
                origin: Origin::Observed,
            },
            AgentEvent::ToolStarted {
                tool: "edit".into(),
                call_id: Some("c1".into()),
                origin: Origin::Observed,
            },
            AgentEvent::PermissionRequested {
                tool: "shell".into(),
                call_id: Some("c2".into()),
                origin: Origin::Observed,
            },
        ];
        // A real scheduler report that timed node "n1" from 100ms to 650ms.
        let mut outcomes = std::collections::BTreeMap::new();
        outcomes.insert(
            "n1".to_string(),
            NodeOutcome {
                id: "n1".into(),
                status: xencode_core_rs::tasks::TaskStatus::Exited(0),
                started_ms: 100,
                finished_ms: 650,
            },
        );
        let report = ScheduleReport {
            launch_order: vec!["n1".into()],
            outcomes,
            peak_concurrency: 1,
            capacity: 2,
            binding: Binding::Workers,
            bottleneck: None,
            critical_path: vec!["n1".into()],
            elapsed_ms: 650,
        };

        let room = ControlRoom::new(
            vec![Stream {
                worker: wr("codex", "split the retrieval tier", Some("n1")),
                events: &events,
            }],
            Some(&report),
        );
        let card = &room.fleet()[0];
        assert_eq!(card.messages, 2);
        assert_eq!(card.files_changed, vec!["src/a.rs".to_string()]);
        assert_eq!(card.tools_in_flight, 1);
        assert_eq!(card.needs_approval, Trace::Known(true));
        assert_eq!(card.elapsed_ms, Trace::Known(550));
        assert!(!card.partly_synthesised);
    }

    /// A duration is unknown when the scheduler never timed the node, even if the
    /// worker is clearly mid-run — the event stream carries no clock.
    #[test]
    fn duration_is_unknown_without_a_timed_node() {
        let ev = vec![msg("still going")];
        let room = ControlRoom::new(
            vec![Stream {
                worker: wr("codex", "open", Some("missing-node")),
                events: &ev,
            }],
            None,
        );
        assert_eq!(room.fleet()[0].elapsed_ms, Trace::Unknown);
    }

    /// Timeline rows and pending approvals both carry the worker's name, and an
    /// approval is dropped once the run completes — nothing resolves on position.
    #[test]
    fn approvals_are_attributed_to_their_worker_and_close_when_it_finishes() {
        let ask = AgentEvent::PermissionRequested {
            tool: "shell".into(),
            call_id: Some("x".into()),
            origin: Origin::Observed,
        };
        let open_events = vec![ask.clone()];
        let mut open = ControlRoom::new(
            vec![Stream {
                worker: wr("claude", "add auth", None),
                events: &open_events,
            }],
            None,
        );
        let pending = open.pending_approvals();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].worker.label(), "claude — add auth");
        assert_eq!(pending[0].tool, "shell");

        // Once it completes, the same worker has nothing pending and its card
        // says approval is no longer needed.
        let events = vec![
            ask,
            AgentEvent::Completed {
                outcome: Some("ok".into()),
                origin: Origin::Observed,
            },
        ];
        open = ControlRoom::new(
            vec![Stream {
                worker: wr("claude", "add auth", None),
                events: &events,
            }],
            None,
        );
        assert!(open.pending_approvals().is_empty());
        assert_eq!(open.fleet()[0].needs_approval, Trace::Known(false));

        // The timeline shows the run as normalised kinds only, never raw bytes.
        let tl = open.timeline();
        assert_eq!(tl[0].kind, "permission_requested");
        assert_eq!(tl[1].kind, "completed");
        assert_eq!(tl[0].worker.label(), "claude — add auth");
    }

    /// The graph projection reads scheduler state and nothing from a stream; with
    /// no report it yields no view rather than an invented one.
    #[test]
    fn the_graph_view_reads_only_the_schedulers_record() {
        use xencode_core_rs::scheduler::Binding;
        let no_report = ControlRoom::new(Vec::new(), None);
        assert!(
            no_report.graph().is_none(),
            "no report, no fabricated graph"
        );

        let report = ScheduleReport {
            launch_order: vec!["n1".into(), "n2".into()],
            outcomes: std::collections::BTreeMap::new(),
            peak_concurrency: 2,
            capacity: 4,
            binding: Binding::Workers,
            bottleneck: Some("n2".into()),
            critical_path: vec!["n1".into(), "n2".into()],
            elapsed_ms: 900,
        };
        let room = ControlRoom::new(Vec::new(), Some(&report));
        let g = room.graph().unwrap();
        assert_eq!(g.capacity, 4);
        assert_eq!(g.binding, "workers");
        assert_eq!(g.bottleneck.as_deref(), Some("n2"));
        assert_eq!(g.launch_order, vec!["n1".to_string(), "n2".to_string()]);
    }
}
