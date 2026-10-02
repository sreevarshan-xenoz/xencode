//! The worker-event to window bridge (`V-10`).
//!
//! This is the thin layer between the orchestrator's normalised events and the
//! screen. It takes what [`ControlRoom`] already projects from `AR-9`/`AR-10`'s
//! `AgentEvent` streams and `OR-2`'s scheduler state, and turns it into the
//! [`AgentPane`] values the window already knows how to draw — so no parallel
//! pane model is invented and no raw vendor byte reaches the surface.
//!
//! Three rules define it, and each is the line a later session is most likely
//! to cross:
//!
//! - **A permission request opens a pane naming the request, without moving
//!   focus.** The pane appears because the bridge reports state; grabbing the
//!   keyboard is not part of that, and the bridge has no way to do it — it takes
//!   no mutable app, focus, or layout, and a test over its own source enforces
//!   that none of those words appears in what it can run.
//! - **A worker xencode cannot observe renders as `unknown`, never as `idle` or
//!   `working`.** `OR-12`'s rule restated for this surface: a panel may show
//!   only what the machine, the provider or the repo gave it, and when it has
//!   nothing it says so plainly rather than dressing a gap up as a state.
//! - **The bridge never moves, resizes or reorders an existing pane.** That is
//!   `V-11`'s parked behaviour, and the easiest thing to add by accident. The
//!   bridge only ever produces new pane values; it is given no collection to
//!   mutate, and the source scan checks for the mutating calls it must not make.

use crate::control_room::{CardStatus, ControlRoom, FleetCard, Trace};
use crate::view::{AgentPane, PaneKind};

/// Bridge the projected fleet into panes the window draws. Pure: it reads the
/// projection and returns owned pane values, holding nothing that could move a
/// pane already on screen.
pub fn bridge(room: &ControlRoom) -> Vec<AgentPane> {
    let mut panes = Vec::new();

    let cards = room.fleet();
    if !cards.is_empty() {
        let rows = cards.iter().map(status_row).collect();
        panes.push(AgentPane {
            kind: PaneKind::Agents,
            title: format!("Workers ({})", cards.len()),
            rows,
        });
    }

    // A permission request opens this pane; a run that has none produces no such
    // pane at all, so it appears and disappears with the request, focus untouched.
    let approvals = room.pending_approvals();
    if !approvals.is_empty() {
        let rows = approvals
            .iter()
            .map(|a| format!("{} — awaiting approval: {}", a.worker.label(), a.tool))
            .collect();
        panes.push(AgentPane {
            kind: PaneKind::Agents,
            title: "Awaiting approval".to_string(),
            rows,
        });
    }

    if let Some(graph) = room.graph() {
        let mut rows = vec![
            format!("launch order: {}", join(&graph.launch_order)),
            format!("binding: {}", graph.binding),
            format!(
                "concurrency: {} of {} at peak",
                graph.peak_concurrency, graph.capacity
            ),
        ];
        if let Some(b) = graph.bottleneck {
            rows.push(format!("bottleneck: {b}"));
        }
        panes.push(AgentPane {
            kind: PaneKind::Monitor,
            title: "Task graph".to_string(),
            rows,
        });
    }

    panes
}

/// One worker's line. The status word comes from what the events proved; a
/// worker that reported nothing is `unknown`, and an unmeasured duration is
/// never shown as a number.
fn status_row(card: &FleetCard) -> String {
    let status = match &card.status {
        CardStatus::Completed { outcome } => match outcome {
            Some(o) => format!("completed ({o})"),
            None => "completed".to_string(),
        },
        CardStatus::Failed { message } => format!("failed ({message})"),
        CardStatus::Running => "running".to_string(),
        // The rule: never "idle", never "working" — just what is true.
        CardStatus::Unknown => "unknown".to_string(),
    };
    let elapsed = match &card.elapsed_ms {
        Trace::Known(ms) => format!(", {ms}ms"),
        Trace::Unknown => ", duration unknown".to_string(),
    };
    format!("{} — {status}{elapsed}", card.worker.label())
}

fn join(items: &[String]) -> String {
    if items.is_empty() {
        "—".to_string()
    } else {
        items.join(" → ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_room::{Stream, WorkerRef};
    use xencode_agents_rs::protocol::{AgentEvent, Origin};

    fn wr(agent: &str, task: &str) -> WorkerRef {
        WorkerRef {
            agent: agent.into(),
            task: task.into(),
            node: None,
        }
    }

    fn ask(tool: &str) -> AgentEvent {
        AgentEvent::PermissionRequested {
            tool: tool.into(),
            call_id: Some("c1".into()),
            origin: Origin::Observed,
        }
    }

    /// The bridge's whole promise is that it cannot touch focus or rearrange an
    /// existing pane, so its source is scanned for the very mechanisms that
    /// would break that — and for a real call into the projection so the scan
    /// cannot pass by being empty.
    #[test]
    fn the_bridge_can_report_state_but_never_moves_a_pane_or_focus() {
        let src = include_str!("worker_bridge.rs");
        let code = src.split("#[cfg(test)]").next().unwrap();
        // No handle to the mutable UI: the bridge must not name focus or layout
        // as things it changes, and must not perform reordering primitives.
        for banned in [
            "app.focus",
            "self.focus",
            ".focus =",
            "set_focus",
            "compute_layout",
            "BodyLayout",
            ".sort",
            ".swap(",
            ".reverse(",
            ".remove(",
            ".insert(",
            "std::process",
            "Command",
        ] {
            assert!(
                !code.contains(banned),
                "the bridge must not touch {banned:?} — that would let it move a pane or read a vendor"
            );
        }
        // It does drive the projection, so the guards above are not vacuous.
        assert!(code.contains("room.fleet()"));
        assert!(code.contains("room.pending_approvals()"));
    }

    /// A worker that stops on a permission request opens a pane naming that
    /// request; nothing in that path changes focus.
    #[test]
    fn a_permission_request_opens_a_pane_naming_it() {
        let events = vec![ask("shell")];
        let room = ControlRoom::new(
            vec![Stream {
                worker: wr("codex", "split the retrieval tier"),
                events: &events,
            }],
            None,
        );
        let panes = bridge(&room);
        let approval = panes
            .iter()
            .find(|p| p.title == "Awaiting approval")
            .expect("a pending request must open a pane");
        assert_eq!(
            approval.rows,
            vec!["codex — split the retrieval tier — awaiting approval: shell".to_string()]
        );
    }

    /// When the run finishes the request closes, so the pane goes away — it
    /// tracks the state, it does not persist a stale prompt.
    #[test]
    fn a_finished_run_closes_the_approval_pane() {
        let events = vec![
            ask("shell"),
            AgentEvent::Completed {
                outcome: Some("ok".into()),
                origin: Origin::Observed,
            },
        ];
        let room = ControlRoom::new(
            vec![Stream {
                worker: wr("codex", "add auth"),
                events: &events,
            }],
            None,
        );
        assert!(
            !bridge(&room).iter().any(|p| p.title == "Awaiting approval"),
            "no live request, no approval pane"
        );
    }

    /// A worker xencode cannot observe is `unknown` on the pane — never `idle`,
    /// never `working`, the two words the done-when forbids dressing a gap in.
    #[test]
    fn an_unobservable_worker_reads_unknown_never_idle_or_working() {
        let room = ControlRoom::new(
            vec![Stream {
                worker: wr("kiro-cli", "triage"),
                events: &[],
            }],
            None,
        );
        let panes = bridge(&room);
        let workers = panes
            .iter()
            .find(|p| p.title.starts_with("Workers"))
            .unwrap();
        let line = &workers.rows[0];
        assert!(line.contains("unknown"), "must render unknown: {line}");
        assert!(!line.contains("idle"), "must not fake idle: {line}");
        assert!(!line.contains("working"), "must not fake working: {line}");
        // An unmeasured duration is called unknown, not shown as a number.
        assert!(line.contains("duration unknown"), "no clock yet: {line}");
    }

    /// A worker whose run is observed reports its real state, so the unknown
    /// rule is not just "always print unknown".
    #[test]
    fn an_observed_worker_reports_its_real_state() {
        let events = vec![
            AgentEvent::SessionStarted {
                session_id: None,
                origin: Origin::Observed,
            },
            AgentEvent::Message {
                text: "hello".into(),
                origin: Origin::Observed,
            },
        ];
        let room = ControlRoom::new(
            vec![Stream {
                worker: wr("claude", "add auth"),
                events: &events,
            }],
            None,
        );
        let panes = bridge(&room);
        let workers = panes
            .iter()
            .find(|p| p.title.starts_with("Workers"))
            .unwrap();
        assert!(
            workers.rows[0].contains("— running"),
            "started and open: running, was {:?}",
            workers.rows[0]
        );
        // The status is not the unknown fallback; only the (still unmeasured)
        // duration is, which is a different, honest gap.
        assert!(
            !workers.rows[0].contains("— unknown"),
            "an observable worker must not read as an unknown status"
        );
    }

    /// Opening a pane is not a focus change: the bridge is given no handle to
    /// the focused area, so an approval surfaces while input stays where it was.
    #[test]
    fn surfacing_an_approval_does_not_move_focus() {
        // `bridge` cannot even be handed the app: its one argument is a
        // projection it only reads. That type signature is the guarantee; this
        // test pins it so a future widening to `&mut App` is caught here.
        let takes_only_a_projection: fn(&ControlRoom) -> Vec<AgentPane> = bridge;
        let events = vec![ask("write")];
        let room = ControlRoom::new(
            vec![Stream {
                worker: wr("codex", "x"),
                events: &events,
            }],
            None,
        );
        let _ = takes_only_a_projection(&room);
    }
}
