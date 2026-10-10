//! The worker panel (`OR-12`): seven readings laid end to end, each one naming
//! the row its figures came from.
//!
//! The sections are the fleet of workers xencode launched, the roles a team
//! recipe names that xencode does not launch itself, the worker agents a lead
//! directs through the project's engine (TM-4), the task registry, what the
//! recorded team runs say about the graph and the energy of a run, the event
//! timeline, and the approvals waiting on a human. Everything the panel can
//! print is a value some other part of xencode already holds or wrote: a
//! [`FleetCard`] from the normalised event streams, a [`TaskRecord`] from the
//! registry, a [`TeamRun`] from `.xencode/team-runs/`, an [`Estimate`] quoted
//! from a run of that exact recipe.
//!
//! Three rules hold it honest, and all three are the done-when.
//!
//! - **Every number traces to a row.** Each row carries its own list of sources,
//!   and `Enter` shows them beside the line they describe. A figure with nothing
//!   behind it is not a figure, so it is not printed; the row says what is
//!   unknown instead, in the same sentence that names why.
//! - **A worker xencode cannot observe is `unknown`, never `idle`.** `idle` is a
//!   claim about what a worker is doing. A role in a recipe that has never run
//!   here, or a stream that reported nothing, has no such claim to make: the
//!   row says so and gives no numbers at all, because a zero would read as a
//!   measurement.
//! - **A worker the posture refuses is `refused`, which is not `unknown`.** Under
//!   the shipped Local-Only profile a role assigned to another vendor's agent
//!   will not be launched at all (`OR-13`), so the row says that instead of
//!   leaving it as a missing reading, and quotes the setting that would open the
//!   rule.
//!
//! Nothing here touches a file, a process or a socket. `App::refresh_worker_panel`
//! does the reading when the panel opens and when the user asks it to, and hands
//! the values over, so this projection is testable on data and never redraws off
//! a disk.

use std::path::Path;

use xencode_core_rs::tasks::TaskRecord;
use xencode_core_rs::team_runs::{Estimate, RunFile};

use crate::app::SpendSnapshot;
use crate::control_room::{CardStatus, FleetCard, PendingApproval, TimelineEntry, Trace};

/// The seven headings, in the order the panel lays them out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelSection {
    Agents,
    Team,
    Tasks,
    Graph,
    Costs,
    Logs,
    Approvals,
}

impl PanelSection {
    /// Every section, in the order the panel lays them out. `/orchestrator status`
    /// (`OR-14`) walks this list so a section with nothing in it is named as one
    /// rather than going missing from the report.
    pub const ALL: [PanelSection; 7] = [
        PanelSection::Agents,
        PanelSection::Team,
        PanelSection::Tasks,
        PanelSection::Graph,
        PanelSection::Costs,
        PanelSection::Logs,
        PanelSection::Approvals,
    ];

    /// The heading as the screen shows it. Lower case, because it is a label in
    /// a row rather than a title of its own panel.
    pub fn title(self) -> &'static str {
        match self {
            PanelSection::Agents => "agents",
            PanelSection::Team => "team",
            PanelSection::Tasks => "tasks",
            PanelSection::Graph => "graph",
            PanelSection::Costs => "costs",
            PanelSection::Logs => "logs",
            PanelSection::Approvals => "approvals",
        }
    }

    /// What the section reads, for the row that heads it. A reader who selects
    /// the heading is asking where these numbers come from.
    fn source(self) -> &'static str {
        match self {
            PanelSection::Agents => {
                "fleet cards from the normalised event streams, plus every role a recipe in \
                 `.xencode/teams/` names — a role xencode did not launch has no stream, so it \
                 shows no figures"
            }
            PanelSection::Team => {
                "the worker agents the project's engine runs for a lead agent \
                 (`xencode mcp serve --team`), as the engine reported them; `s` stops the \
                 selected worker and `m` merges it when the project's checks pass"
            }
            PanelSection::Tasks => {
                "the task registry, read without waiting on it: a locked \
                                    registry is reported as unknown, never as an empty list"
            }
            PanelSection::Graph => {
                "one row per recorded run in `.xencode/team-runs/`, taken \
                                    from the scheduler's own report of that run"
            }
            PanelSection::Costs => {
                "this session's spend from the records on disk, the energy \
                                    each recorded run measured, and for a recipe that has run \
                                    here, a quote for the next run labelled as the quote it is"
            }
            PanelSection::Logs => {
                "the newest events across every stream, each already \
                                   normalised into the common model by its adapter"
            }
            PanelSection::Approvals => {
                "the queue the tool loop is parked on, and any \
                                        permission a worker raised that nothing has answered"
            }
        }
    }

    /// The header row for the section, counting the rows that follow it.
    fn header(self, rows: &[PanelRow]) -> PanelRow {
        PanelRow {
            section: self,
            is_header: true,
            line: format!("── {} ({}) ──", self.title(), rows.len()),
            sources: vec![self.source().to_string()],
        }
    }
}

/// One line of the panel: what it says, and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelRow {
    pub section: PanelSection,
    /// The section's own row. It carries the count, so it is not one of the
    /// readings and is styled apart.
    pub is_header: bool,
    pub line: String,
    /// One entry per figure on the line, naming the event, record or file that
    /// holds it. This is what `Enter` opens.
    pub sources: Vec<String>,
}

impl PanelRow {
    /// The line and its sources together, for the detail view.
    pub fn detail(&self) -> String {
        if self.sources.is_empty() {
            return self.line.clone();
        }
        let mut out = format!("{}\n\nRead from\n", self.line);
        for source in &self.sources {
            out.push_str("  ");
            out.push_str(source);
            out.push('\n');
        }
        out.trim_end().to_string()
    }
}

/// Lay the seven sections out in order, each behind its own heading. A section with
/// nothing in it gets no heading: the heading carries the count, and a heading
/// over an empty list would be a figure with nothing behind it. `App` reads the
/// missing heading as "this section found nothing to say", which is what
/// `/orchestrator status` (`OR-14`) prints in that case.
pub fn sections(parts: [(PanelSection, Vec<PanelRow>); 7]) -> Vec<PanelRow> {
    let mut out = Vec::new();
    for (section, rows) in parts {
        if !rows.is_empty() {
            out.push(section.header(&rows));
        }
        out.extend(rows);
    }
    out
}

/// Keep one section of an already-built panel, heading and all (`OR-14`). This is
/// a filter over rows that were read, not a second reading: `/orchestrator graph`
/// and `/workers` show the same figures about the same runs, because they are the
/// same `Vec<PanelRow>` with one comparison applied.
pub fn only(rows: Vec<PanelRow>, section: Option<PanelSection>) -> Vec<PanelRow> {
    match section {
        None => rows,
        Some(wanted) => rows
            .into_iter()
            .filter(|row| row.section == wanted)
            .collect(),
    }
}

/// One card per worker xencode launched. The figures are the card's, and each
/// one is named with the event that proves it.
pub fn fleet_rows(cards: &[FleetCard]) -> Vec<PanelRow> {
    cards
        .iter()
        .map(|card| {
            let status = match &card.status {
                CardStatus::Completed { outcome: Some(o) } => format!("completed ({o})"),
                CardStatus::Completed { outcome: None } => "completed".to_string(),
                CardStatus::Failed { message } => format!("failed ({message})"),
                CardStatus::Running => "running".to_string(),
                // The rule: no event has arrived, so the panel claims nothing
                // about what the worker is doing.
                CardStatus::Unknown => "unknown".to_string(),
            };
            let duration = match &card.elapsed_ms {
                Trace::Known(ms) => format!(", {ms}ms"),
                Trace::Unknown => ", duration unknown".to_string(),
            };
            let files = if card.files_changed.is_empty() {
                "no files changed".to_string()
            } else {
                format!("{} file(s) changed", card.files_changed.len())
            };
            let approval = match &card.needs_approval {
                Trace::Known(true) => "waiting on you".to_string(),
                Trace::Known(false) => "not waiting".to_string(),
                Trace::Unknown => "approval state unknown".to_string(),
            };
            let mut sources = vec![
                match &card.status {
                    CardStatus::Unknown => {
                        "status unknown: the stream this panel was handed holds no events at \
                         all, so nothing about what the worker is doing is claimed"
                            .to_string()
                    }
                    _ => "status: derived only from the normalised event variants this stream \
                          carried, never from a word the worker wrote about itself"
                        .to_string(),
                },
                format!("{} message(s): counted from Message events", card.messages),
                format!(
                    "{files}: FileChanged events, which xencode fills from the diff it read, \
                     not from a claim"
                ),
                format!(
                    "{} tool(s) in flight: ToolStarted minus ToolOutput",
                    card.tools_in_flight
                ),
                format!(
                    "{approval}: PermissionRequested, closed only by a denial or the run ending"
                ),
            ];
            sources.push(match (&card.worker.node, &card.elapsed_ms) {
                (Some(node), Trace::Known(ms)) => format!(
                    "{ms}ms: the scheduler's record of node `{node}` — launched and finished, \
                     both ends timed"
                ),
                (Some(node), Trace::Unknown) => format!(
                    "duration unknown: node `{node}` has no record with both ends timed, and \
                     the event model carries no timestamps"
                ),
                (None, _) => "duration unknown: this worker is not tied to a scheduler node, \
                              so nothing timed it"
                    .to_string(),
            });
            if card.partly_synthesised {
                sources.push(
                    "some of these events were written by xencode's own bookkeeping rather \
                     than witnessed in the worker's stream"
                        .to_string(),
                );
            }
            PanelRow {
                section: PanelSection::Agents,
                is_header: false,
                line: format!("{} — {status}{duration}, {files}", card.worker.label()),
                sources,
            }
        })
        .collect()
}

/// A role a team recipe names. xencode holds the recipe and nothing else: the
/// worker it names is one that `xencode team run` would launch, so from here it
/// is unobservable and gets no figures.
pub struct PlannedRole {
    pub recipe: String,
    pub worker: String,
    pub role: String,
    pub path: String,
    /// The checks that gate this role, as the recipe wrote them. An empty list is
    /// the recipe's own answer and is shown as such.
    pub gates: Vec<String>,
    pub needs: Vec<String>,
    /// Whether the agent roster has a row for this worker — the same answer the
    /// router is handed, and answered by the caller because this projection reads
    /// no files and no `PATH`.
    pub external: bool,
}

/// One row per role, refused first when the posture refuses the worker it names.
///
/// A refusal is a stronger statement than `unknown`: an unobserved role might
/// yet run, while a refused role will not be launched at all under the posture
/// that is in force, and the row quotes that posture and the setting that opens
/// it (`OR-13`). A worker the roster cannot place is not claimed as xencode's
/// own — the row is the ordinary one, and its sources say the question went
/// unanswered.
pub fn planned_role_rows(
    roles: &[PlannedRole],
    profile: &xencode_core_rs::Profile,
) -> Vec<PanelRow> {
    roles
        .iter()
        .map(|role| {
            let gates = if role.gates.is_empty() {
                "no check gates it".to_string()
            } else {
                format!("{} check(s)", role.gates.len())
            };
            let refusal = profile.check_worker(&role.worker, role.external);
            let mut sources = vec![
                "status, duration, messages and cost: unknown — xencode holds no event \
                 stream for a role it did not launch, so it prints none of them"
                    .to_string(),
                format!("{gates}: the `gate` list in {}", role.path),
                format!(
                    "waits on {}: the `needs` list in {}",
                    if role.needs.is_empty() {
                        "nothing".to_string()
                    } else {
                        role.needs.join(", ")
                    },
                    role.path
                ),
                format!("recipe `{}`", role.recipe),
            ];
            if role.external {
                sources.push(format!(
                    "worker `{}`: the agent roster has a row for it, which is xencode's list of \
                     another vendor's coding-agent CLI",
                    role.worker
                ));
            } else {
                sources.push(format!(
                    "whose worker `{}` is: the roster has no row for it, so this is not a claim \
                     that it is xencode's own — nothing here checked whether it exists",
                    role.worker
                ));
            }
            if let Err(refusal) = refusal {
                sources.push(format!("posture: {}", profile.worker_rule()));
                sources.push(refusal.why);
                return PanelRow {
                    section: PanelSection::Agents,
                    is_header: false,
                    line: format!(
                        "{} — {}: refused by the {} posture, not launched",
                        role.worker,
                        role.role,
                        profile.name()
                    ),
                    sources,
                };
            }
            PanelRow {
                section: PanelSection::Agents,
                is_header: false,
                line: format!("{} — {}: unknown, not idle", role.worker, role.role),
                sources,
            }
        })
        .collect()
}

/// A file or directory the panel was pointed at and could not read. Leaving it
/// out would make the section look like it had seen everything there was.
pub fn unreadable_row(section: PanelSection, what: &str, problem: &str) -> PanelRow {
    PanelRow {
        section,
        is_header: false,
        line: format!("{what} — unreadable, so nothing below it is listed"),
        sources: vec![format!(
            "it is there and xencode could not read it: {problem}"
        )],
    }
}

/// The task registry. `None` means the lock is held by a running turn, which is
/// a different fact from "no tasks" and gets its own row rather than a zero.
pub fn task_rows(snapshot: Option<&[TaskRecord]>) -> Vec<PanelRow> {
    let Some(tasks) = snapshot else {
        return vec![PanelRow {
            section: PanelSection::Tasks,
            is_header: false,
            line: "tasks: unknown — a turn holds the registry".to_string(),
            sources: vec![
                "the registry is read without waiting, because the panel must not block the \
                 event loop while a tool call owns it; nothing was counted this frame, so no \
                 count is shown"
                    .to_string(),
            ],
        }];
    };
    if tasks.is_empty() {
        return vec![PanelRow {
            section: PanelSection::Tasks,
            is_header: false,
            line: "tasks: none recorded".to_string(),
            sources: vec![
                "the registry answered the read and held no rows — a counted zero, not the \
                 locked one above"
                    .to_string(),
            ],
        }];
    }
    tasks
        .iter()
        .map(|task| {
            let pid = match task.pid {
                Some(pid) => format!("pid {pid}"),
                None => "no pid recorded".to_string(),
            };
            let end = match task.finished_at {
                Some(at) => format!("ended at {at}"),
                None => "still open".to_string(),
            };
            PanelRow {
                section: PanelSection::Tasks,
                is_header: false,
                line: format!(
                    "#{} {} — {} ({pid})",
                    task.id,
                    task.name,
                    task.status.label()
                ),
                sources: vec![
                    format!(
                        "{}: the status the registry holds for row #{}",
                        task.status.label(),
                        task.id
                    ),
                    format!("{pid}: recorded when the process was spawned"),
                    format!("started at {}: the registry's own stamp", task.started_at),
                    format!(
                        "{end}: {}",
                        match task.finished_at {
                            Some(_) => "written when the task was reaped",
                            None => "the task has not been reaped, so no end time exists to show",
                        }
                    ),
                    format!(
                        "{} output line(s) held, oldest evicted past the cap",
                        task.output().len()
                    ),
                    format!("command: {}", task.command),
                ],
            }
        })
        .collect()
}

/// One row per recorded run, newest last so the list reads forward in time. The
/// figures are the run's, taken from the report the scheduler left in the file.
pub fn graph_rows(runs: &[RunFile], dir: &Path) -> Vec<PanelRow> {
    if runs.is_empty() {
        return vec![PanelRow {
            section: PanelSection::Graph,
            is_header: false,
            line: format!(
                "graph: nothing measured — no run recorded under {}",
                dir.display()
            ),
            sources: vec![
                "a team that has never run here leaves no file, and the live session reports no \
                 scheduler timings of its own, so there is no order, no peak and no elapsed to \
                 print"
                    .to_string(),
            ],
        }];
    }
    runs.iter()
        .map(|file| match &file.run {
            Err(problem) => PanelRow {
                section: PanelSection::Graph,
                is_header: false,
                line: format!(
                    "{} — the record is there and cannot be read",
                    file.path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("?")
                ),
                sources: vec![
                    format!("{}", file.path.display()),
                    format!("why it failed: {problem}"),
                ],
            },
            Ok(run) => {
                let order = if run.launch_order.is_empty() {
                    "nothing launched".to_string()
                } else {
                    run.launch_order.join(" → ")
                };
                let energy = match run.watt_hours {
                    Some(wh) => format!("{wh:.3} Wh"),
                    None => "energy unknown".to_string(),
                };
                let mut sources = vec![
                    format!("{}", file.path.display()),
                    format!(
                        "launch order {}: the scheduler's report, written by the run itself",
                        run.launch_order.len()
                    ),
                    format!(
                        "peak {} of capacity {}: what the queue recorded while it ran, and the \
                         number it was capped at",
                        run.peak_concurrency, run.capacity
                    ),
                    format!("binding: the record's `{}`", run.binding),
                    format!(
                        "elapsed {}ms: measured end to end by that run",
                        run.elapsed_ms
                    ),
                    format!(
                        "approved by {} at {}",
                        run.approved_by, run.started_at_unix_ms
                    ),
                    format!("{energy}: what the machine's counter drew while the run lasted"),
                ];
                for role in &run.roles {
                    sources.push(format!(
                        "role {}: {}, {}ms to {}ms — the queue's own timing",
                        role.name, role.status, role.started_ms, role.finished_ms
                    ));
                }
                PanelRow {
                    section: PanelSection::Graph,
                    is_header: false,
                    line: format!(
                        "{} — {}, peak {}/{} ({}), {}ms",
                        run.run_id,
                        order,
                        run.peak_concurrency,
                        run.capacity,
                        run.binding,
                        run.elapsed_ms
                    ),
                    sources,
                }
            }
        })
        .collect()
}

/// A recipe's quote for a next run, and the file it was read from so the row can
/// name its own source.
pub struct Quote {
    pub recipe: String,
    pub path: String,
    pub estimate: Option<Estimate>,
}

pub fn cost_rows(
    spend: Option<&SpendSnapshot>,
    runs: &[RunFile],
    quotes: &[Quote],
) -> Vec<PanelRow> {
    let mut rows = Vec::new();
    match spend {
        None => rows.push(PanelRow {
            section: PanelSection::Costs,
            is_header: false,
            line: "this session: nothing on record yet".to_string(),
            sources: vec![
                "the spend line is refreshed when a turn finishes and reads the records on \
                 disk; no turn has finished, so there is no figure to show and no zero is \
                 invented for one"
                    .to_string(),
            ],
        }),
        Some(spend) => {
            let price = match (spend.micros, spend.priced) {
                (Some(_), _) => spend.line.clone(),
                (None, true) => "priced, and the price came to nothing".to_string(),
                (None, false) => "no price is known for what was used".to_string(),
            };
            rows.push(PanelRow {
                section: PanelSection::Costs,
                is_header: false,
                line: format!("this session: {price}, {} token(s)", spend.tokens),
                sources: vec![
                    format!(
                        "{}: settled from the run records at the turn's end, never estimated \
                         mid-turn",
                        if spend.micros.is_some() {
                            "the cost shown"
                        } else {
                            "no cost"
                        }
                    ),
                    format!(
                        "{} token(s): prompt plus completion on record",
                        spend.tokens
                    ),
                    format!(
                        "prices: {}",
                        if spend.priced {
                            "at least one model used this session is priced in the table"
                        } else {
                            "the price table holds nothing this session used, so the cost is \
                             unknown rather than free"
                        }
                    ),
                ],
            });
        }
    }

    for file in runs {
        let name = file
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();
        let Ok(run) = file.run.as_ref() else { continue };
        let energy = match (run.watt_hours, run.cents_per_kwh) {
            (Some(wh), Some(cents)) => match run.cost_micros(Some(cents)) {
                Some(micros) => format!(
                    "{wh:.3} Wh at {} c/kWh, {}",
                    cents,
                    xencode_context_rs::format_usd(micros)
                ),
                None => format!("{wh:.3} Wh, priced to nothing"),
            },
            (Some(wh), None) => format!("{wh:.3} Wh, no tariff recorded, so no price"),
            (None, _) => "the machine drew no energy counter".to_string(),
        };
        rows.push(PanelRow {
            section: PanelSection::Costs,
            is_header: false,
            line: format!("{name}: {energy}"),
            sources: vec![
                format!("{}", file.path.display()),
                format!(
                    "watt-hours: this machine's counter over the {}ms the run lasted",
                    run.elapsed_ms
                ),
                format!(
                    "tariff: {}",
                    match run.cents_per_kwh {
                        Some(_) =>
                            "the one set when the run happened, kept in the record so \
                                    the energy is not frozen at yesterday's price",
                        None => "absent, which is why no price is shown",
                    }
                ),
            ],
        });
    }

    for quote in quotes {
        rows.push(match &quote.estimate {
            Some(estimate) => PanelRow {
                section: PanelSection::Costs,
                is_header: false,
                line: format!(
                    "{}: a next run quoted at {}ms from {}",
                    quote.recipe, estimate.wall_clock_ms, estimate.run_id
                ),
                sources: vec![
                    "an estimate, not a measurement: it is one past run of this exact \
                     recipe, and the plan for the next one cannot know what it has not run"
                        .to_string(),
                    format!("quoted from {}", quote.path),
                    format!(
                        "{} role(s), peak {}, {} — the fingerprint of the recipe is what lets \
                         that run stand for this one",
                        estimate.roles,
                        estimate.peak_concurrency,
                        match estimate.watt_hours {
                            Some(wh) => format!("{wh:.3} Wh"),
                            None => "no energy measured".to_string(),
                        }
                    ),
                ],
            },
            None => PanelRow {
                section: PanelSection::Costs,
                is_header: false,
                line: format!("{}: no quote — never run on this machine", quote.recipe),
                sources: vec![
                    format!("read from {}", quote.path),
                    "no record carries this recipe's fingerprint, so there is no wall clock to \
                     quote and no typical number is filled in"
                        .to_string(),
                ],
            },
        });
    }
    rows
}

/// The newest events across every stream. `keep` bounds the list; the rows it
/// cost are named, so the screen never reads as the whole history.
pub fn log_rows(entries: &[TimelineEntry], keep: usize) -> Vec<PanelRow> {
    let mut rows = Vec::new();
    if entries.is_empty() {
        // The other five sections each report a measured nothing; a section that
        // simply went missing would be the panel skipping part of its own answer.
        return vec![PanelRow {
            section: PanelSection::Logs,
            is_header: false,
            line: "logs: nothing has reported yet".to_string(),
            sources: vec![
                "the timeline this session's control room built is empty — no worker has put an \
                 event on it, so the read counted zero rather than the panel passing over it"
                    .to_string(),
            ],
        }];
    }
    let hidden = entries.len().saturating_sub(keep);
    for (i, entry) in entries.iter().enumerate().skip(hidden) {
        rows.push(PanelRow {
            section: PanelSection::Logs,
            is_header: false,
            line: format!(
                "{} — {}: {}",
                entry.worker.label(),
                entry.kind,
                entry.detail
            ),
            sources: vec![format!(
                "event {} of {} in that worker's stream, normalised by its adapter; the raw \
                 vendor line is not kept anywhere in this panel",
                i + 1,
                entries.len()
            )],
        });
    }
    if hidden > 0 {
        rows.insert(
            0,
            PanelRow {
                section: PanelSection::Logs,
                is_header: false,
                line: format!("{hidden} earlier event(s) not shown"),
                sources: vec![format!(
                    "the panel lists the newest {keep} of {} rows; the rest are still in the \
                     streams they came from",
                    entries.len()
                )],
            },
        );
    }
    rows
}

/// One row per worker agent the project's engine runs (TM-4). Each row starts
/// with the worker's id, which is what `s` and `m` act on.
pub fn team_rows(workers: &[xencode_team_rs::WorkerSnapshot]) -> Vec<PanelRow> {
    use xencode_team_rs::{merge::MergeOutcome, MergeState, WorkerState};
    if workers.is_empty() {
        return vec![PanelRow {
            section: PanelSection::Team,
            is_header: false,
            line: "team: no worker agents are running".to_string(),
            sources: vec![
                "the project's engine runs none; a lead agent starts them with `team_start` \
                 through `xencode mcp serve --team`"
                    .to_string(),
            ],
        }];
    }
    workers
        .iter()
        .map(|w| {
            let state = match w.state {
                WorkerState::Starting => "starting",
                WorkerState::Working => "working",
                WorkerState::NeedsYou => "needs you",
                WorkerState::Done => "done",
                WorkerState::Failed => "failed",
                WorkerState::Stopped => "stopped",
            };
            let mut line = format!(
                "{} · {} · {state} · {} tool call(s)",
                w.id, w.agent, w.tool_calls
            );
            match &w.merge {
                Some(MergeState::Running) => line.push_str(" · merging"),
                Some(MergeState::Finished(outcome)) => line.push_str(match outcome {
                    MergeOutcome::Landed { .. } => " · landed",
                    MergeOutcome::ChecksFailed { .. } => " · checks failed",
                    MergeOutcome::Conflict { .. } => " · merge conflict",
                    MergeOutcome::NeedsPerson { .. } => " · merge needs you",
                    MergeOutcome::Refused { .. } => " · merge refused",
                }),
                None => {}
            }
            let said = w.error.as_deref().unwrap_or(&w.last_message);
            if !said.is_empty() {
                let clipped: String = said.chars().take(80).collect();
                line.push_str(" — ");
                line.push_str(&clipped);
            }
            let mut sources = vec![
                format!("worker {} in the project's engine, task: {}", w.id, w.task),
                format!(
                    "its worktree {} on branch {}",
                    w.worktree.display(),
                    w.branch
                ),
            ];
            if let Some(MergeState::Finished(outcome)) = &w.merge {
                sources.push(format!(
                    "its merge: {}",
                    serde_json::to_string(outcome).unwrap_or_default()
                ));
            }
            PanelRow {
                section: PanelSection::Team,
                is_header: false,
                line,
                sources,
            }
        })
        .collect()
}

/// The worker a Team row is about, read back from the id it starts with.
pub fn team_worker(row: &PanelRow) -> Option<&str> {
    if row.section != PanelSection::Team || row.is_header {
        return None;
    }
    let id = row.line.split(' ').next()?;
    (id.starts_with('w') && id.len() > 1 && id[1..].chars().all(|c| c.is_ascii_digit()))
        .then_some(id)
}

/// The approvals waiting on a person, from both places that can hold one.
pub fn approval_rows(live: &[String], raised: &[PendingApproval]) -> Vec<PanelRow> {
    if live.is_empty() && raised.is_empty() {
        return vec![PanelRow {
            section: PanelSection::Approvals,
            is_header: false,
            line: "approvals: nothing is waiting".to_string(),
            sources: vec![
                "both places that can hold one were read: the queue the tool loop parks on is \
                 empty, and no worker has a permission request open that a denial or the end of \
                 its run has not closed"
                    .to_string(),
            ],
        }];
    }
    let mut rows: Vec<PanelRow> = live
        .iter()
        .map(|line| PanelRow {
            section: PanelSection::Approvals,
            is_header: false,
            line: line.clone(),
            sources: vec![
                "the live queue: a tool call in this session is parked on the answer, and the \
                 modal shows the bytes at stake"
                    .to_string(),
            ],
        })
        .collect();
    rows.extend(raised.iter().map(|ask| PanelRow {
        section: PanelSection::Approvals,
        is_header: false,
        line: format!("{} — awaiting approval: {}", ask.worker.label(), ask.tool),
        sources: vec![format!(
            "a PermissionRequested for `{}` in that worker's stream, with no denial and no end \
             of run after it{}",
            ask.tool,
            match &ask.call_id {
                Some(id) => format!(" (call {id})"),
                None => String::new(),
            }
        )],
    }));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_room::{Stream, WorkerRef};
    use xencode_agents_rs::protocol::{AgentEvent, Origin};
    use xencode_core_rs::team_runs::{RoleRun, TeamRun};

    fn wr(agent: &str, task: &str, node: Option<&str>) -> WorkerRef {
        WorkerRef {
            agent: agent.into(),
            task: task.into(),
            node: node.map(str::to_string),
        }
    }

    fn room_with(events: Vec<AgentEvent>, agent: &str) -> Vec<FleetCard> {
        let room = crate::control_room::ControlRoom::new(
            vec![Stream {
                worker: wr(agent, "split the retrieval tier", None),
                events: &events,
            }],
            None,
        );
        room.fleet()
    }

    /// The done-when's second half, watched at the surface it is written for: a
    /// worker with no events gets the word unknown and no figures at all.
    #[test]
    fn an_unobserved_worker_row_says_unknown_and_prints_no_numbers() {
        let rows = fleet_rows(&room_with(Vec::new(), "codex"));
        let line = &rows[0].line;
        assert!(line.contains("unknown"), "{line}");
        for banned in ["idle", "working", "waiting"] {
            assert!(
                !line.contains(banned),
                "{line} claims a state it cannot know"
            );
        }
        assert!(
            line.contains("duration unknown"),
            "an unmeasured clock is named, not zeroed: {line}"
        );
        // `0 file(s)` would read as a measurement of nothing. The row says what
        // it counted instead.
        assert!(line.contains("no files changed"), "{line}");
        assert!(
            rows[0]
                .sources
                .iter()
                .any(|s| s.contains("no events at all")),
            "the sources must say why the status is unknown: {:?}",
            rows[0].sources
        );
    }

    /// A role a recipe names and xencode did not launch is the clearest case of
    /// an unobservable worker: it has a name, a file, and no state whatsoever. A
    /// worker the roster cannot place is the case the posture has nothing to say
    /// about, so its row says the question went unanswered rather than that the
    /// worker is xencode's own.
    #[test]
    fn a_recipe_role_xencode_did_not_launch_is_unknown_and_carries_no_figures() {
        let rows = planned_role_rows(
            &[PlannedRole {
                recipe: "rust-fix".into(),
                worker: "my-own-runner".into(),
                role: "survey".into(),
                path: ".xencode/teams/rust-fix.toml".into(),
                gates: vec!["cargo test".into()],
                needs: vec![],
                external: false,
            }],
            &xencode_core_rs::Profile::LOCAL_ONLY,
        );
        let line = &rows[0].line;
        assert_eq!(line, "my-own-runner — survey: unknown, not idle");
        assert!(
            rows[0]
                .sources
                .iter()
                .any(|s| s.starts_with("status, duration, messages and cost: unknown")),
            "{:?}",
            rows[0].sources
        );
        assert!(
            rows[0].sources.iter().any(|s| s.contains("no row for it")
                && s.contains("not a claim that it is xencode's own")),
            "a worker nobody could place must not read as an accepted one: {:?}",
            rows[0].sources
        );
        // The one count the recipe really does carry is the gate list's.
        assert!(rows[0]
            .sources
            .iter()
            .any(|s| s == "1 check(s): the `gate` list in .xencode/teams/rust-fix.toml"));
    }

    /// The refusal (`OR-13`): a role assigned to another vendor's agent is not an
    /// unobserved role any more — under the posture in force it will not be
    /// launched at all. The row says which posture refused it, and the sources
    /// carry the rule and the setting that opens it.
    #[test]
    fn a_role_the_posture_refuses_is_said_as_refused_and_names_its_way_out() {
        let refused = planned_role_rows(
            &[PlannedRole {
                recipe: "rust-fix".into(),
                worker: "codex".into(),
                role: "survey".into(),
                path: ".xencode/teams/rust-fix.toml".into(),
                gates: vec!["cargo test".into()],
                needs: vec![],
                external: true,
            }],
            &xencode_core_rs::Profile::LOCAL_ONLY,
        );
        let row = &refused[0];
        assert_eq!(
            row.line,
            "codex — survey: refused by the Local Only posture, not launched"
        );
        assert!(
            row.sources
                .iter()
                .any(|s| s.starts_with("posture: worker routes:")),
            "{:?}",
            row.sources
        );
        assert!(
            row.sources
                .iter()
                .any(|s| s.contains("xencode has a roster row for `codex`")),
            "the reason is on screen: {:?}",
            row.sources
        );
        assert_eq!(
            row.sources
                .iter()
                .filter(|s| s.contains("xencode config set allow_external_workers true"))
                .count(),
            1,
            "the setting is stated once on the row, by the rule line, and not repeated \
             under the reason: {:?}",
            row.sources
        );
        // Opening the rule turns the very same role back into an unobserved one,
        // which is what distinguishes a refusal from a missing reading.
        let allowed = planned_role_rows(
            &[first_refused_shape()],
            &xencode_core_rs::Profile::new(false, true),
        );
        assert_eq!(allowed[0].line, "codex — survey: unknown, not idle");
        assert!(
            allowed[0]
                .sources
                .iter()
                .any(|s| s.starts_with("worker `codex`: the agent roster has a row")),
            "{:?}",
            allowed[0].sources
        );
    }

    /// The same role, for the two halves of the test above, so the only thing that
    /// changes between a refusal and a pass is the posture.
    fn first_refused_shape() -> PlannedRole {
        PlannedRole {
            recipe: "rust-fix".into(),
            worker: "codex".into(),
            role: "survey".into(),
            path: ".xencode/teams/rust-fix.toml".into(),
            gates: vec!["cargo test".into()],
            needs: vec![],
            external: true,
        }
    }

    /// A locked registry is not an empty one. The two rows must differ in words,
    /// because a reader decides differently on each.
    #[test]
    fn a_locked_registry_reads_unknown_and_an_empty_one_reads_a_counted_zero() {
        let locked = &task_rows(None)[0];
        assert!(locked.line.contains("unknown"), "{}", locked.line);
        assert!(
            !locked.line.contains('0'),
            "the locked case must print no count: {}",
            locked.line
        );
        let empty = &task_rows(Some(&[]))[0];
        assert_eq!(empty.line, "tasks: none recorded");
        assert!(
            empty.sources[0].contains("counted zero"),
            "{:?}",
            empty.sources
        );
        assert_ne!(locked.line, empty.line);
    }

    /// A recorded run's figures all name the file they were read from, and a
    /// record that cannot be parsed is shown rather than dropped.
    #[test]
    fn a_recorded_run_names_its_file_and_an_unreadable_one_stays_on_screen() {
        let run = TeamRun {
            run_id: "rust-fix-1".into(),
            recipe: "rust-fix".into(),
            recipe_file: "teams/rust-fix.toml".into(),
            fingerprint: "abc".into(),
            approved_by: "sree".into(),
            started_at_unix_ms: 10,
            elapsed_ms: 900,
            capacity: 4,
            binding: "workers".into(),
            peak_concurrency: 2,
            launch_order: vec!["survey".into(), "fix".into()],
            roles: vec![RoleRun {
                name: "survey".into(),
                status: "exited(0)".into(),
                started_ms: 0,
                finished_ms: 200,
            }],
            watt_hours: Some(1.5),
            cents_per_kwh: Some(12.0),
        };
        let dir = Path::new("/tmp/.xencode/team-runs");
        let rows = graph_rows(
            &[
                RunFile {
                    path: dir.join("rust-fix-1.json"),
                    run: Ok(run),
                },
                RunFile {
                    path: dir.join("broken.json"),
                    run: Err("expected value at line 1".into()),
                },
            ],
            dir,
        );
        assert!(rows[0].line.contains("survey → fix"), "{}", rows[0].line);
        assert!(rows[0].line.contains("2/4"), "{}", rows[0].line);
        assert!(
            rows[0]
                .sources
                .iter()
                .any(|s| s.contains("role survey: exited(0), 0ms to 200ms")),
            "{:?}",
            rows[0].sources
        );
        assert!(rows[0]
            .sources
            .iter()
            .any(|s| s.ends_with("rust-fix-1.json")));
        assert!(rows[1].line.contains("cannot be read"), "{}", rows[1].line);
    }

    /// With no records at all the graph section says there is nothing measured,
    /// and does not slip in a peak or an elapsed.
    #[test]
    fn with_no_records_the_graph_says_nothing_was_measured() {
        let rows = graph_rows(&[], Path::new(".xencode/team-runs"));
        assert_eq!(rows.len(), 1);
        assert!(rows[0].line.starts_with("graph: nothing measured"));
        assert!(
            !rows[0].line.contains('0') && !rows[0].line.contains("peak"),
            "no number without a run behind it: {}",
            rows[0].line
        );
    }

    /// A quote for a next run is labelled the estimate it is, and names the run
    /// it was taken from. A recipe that has never run gets no number at all.
    #[test]
    fn a_quote_is_called_an_estimate_and_names_the_run_it_came_from() {
        let quotes = vec![
            Quote {
                recipe: "rust-fix".into(),
                path: ".xencode/teams/rust-fix.toml".into(),
                estimate: Some(Estimate {
                    run_id: "rust-fix-1".into(),
                    approved_by: "sree".into(),
                    started_at_unix_ms: 10,
                    wall_clock_ms: 900,
                    peak_concurrency: 2,
                    watt_hours: None,
                    roles: 3,
                }),
            },
            Quote {
                recipe: "docs-sweep".into(),
                path: ".xencode/teams/docs-sweep.toml".into(),
                estimate: None,
            },
        ];
        let rows = cost_rows(None, &[], &quotes);
        let quoted = &rows[1];
        assert!(
            quoted.line.contains("quoted at 900ms from rust-fix-1"),
            "{}",
            quoted.line
        );
        assert!(
            quoted
                .sources
                .iter()
                .any(|s| s.starts_with("an estimate, not a measurement")),
            "{:?}",
            quoted.sources
        );
        let never = &rows[2];
        assert_eq!(
            never.line,
            "docs-sweep: no quote — never run on this machine"
        );
        assert!(
            !never.line.chars().any(|c| c.is_ascii_digit()),
            "no number fills in for a run that never happened: {}",
            never.line
        );
    }

    /// An unpriced session is unknown, not free: the row says no price is known
    /// and the sources say the table holds nothing, rather than showing `$0`.
    #[test]
    fn a_session_with_no_price_is_unknown_and_never_free() {
        let spend = SpendSnapshot {
            line: String::new(),
            micros: None,
            tokens: 4200,
            priced: false,
        };
        let rows = cost_rows(Some(&spend), &[], &[]);
        assert!(
            rows[0].line.contains("no price is known"),
            "{}",
            rows[0].line
        );
        assert!(
            !rows[0].line.contains("$0"),
            "a zero is not a price: {}",
            rows[0].line
        );
        assert!(rows[0].line.contains("4200 token(s)"), "{}", rows[0].line);
    }

    /// Every figure the fleet row prints is a count of something the events
    /// proved, so the sources name the event each one came from.
    #[test]
    fn each_figure_on_a_fleet_row_names_the_event_that_proves_it() {
        let cards = room_with(
            vec![AgentEvent::Message {
                text: "on it".into(),
                origin: Origin::Observed,
            }],
            "claude",
        );
        let rows = fleet_rows(&cards);
        assert!(rows[0].line.contains("claude — split the retrieval tier"));
        let sources = rows[0].sources.join("\n");
        for expected in [
            "1 message(s): counted from Message events",
            "FileChanged events",
            "ToolStarted minus ToolOutput",
            "duration unknown",
        ] {
            assert!(
                sources.contains(expected),
                "missing {expected} in\n{sources}"
            );
        }
    }

    /// The empty approvals section is a measured nothing — both places were read.
    /// TM-4: a worker's row starts with its id, which `s` and `m` act on, and
    /// says its state and its merge; no workers is a row that says so.
    #[test]
    fn a_team_row_names_its_worker_its_state_and_its_merge() {
        use xencode_team_rs::{merge::MergeOutcome, MergeState, WorkerSnapshot, WorkerState};
        let worker = WorkerSnapshot {
            id: "w12".into(),
            agent: "xencode".into(),
            task: "fix the parser".into(),
            state: WorkerState::NeedsYou,
            branch: "xencode/team/w12".into(),
            worktree: "/p/proj-team/w12".into(),
            last_message: "may I edit src/parse.rs?".into(),
            answer: String::new(),
            error: None,
            tool_calls: 3,
            tokens: None,
            cost_micros: None,
            on_plan: false,
            merge: Some(MergeState::Finished(MergeOutcome::ChecksFailed {
                output: "1 test failed".into(),
            })),
        };
        let rows = team_rows(std::slice::from_ref(&worker));
        assert_eq!(
            rows[0].line,
            "w12 · xencode · needs you · 3 tool call(s) · checks failed — may I edit src/parse.rs?"
        );
        assert_eq!(team_worker(&rows[0]), Some("w12"));
        assert!(
            rows[0].detail().contains("1 test failed"),
            "{}",
            rows[0].detail()
        );

        let none = team_rows(&[]);
        assert_eq!(none[0].line, "team: no worker agents are running");
        assert_eq!(team_worker(&none[0]), None);
        let heading = sections([
            (PanelSection::Agents, vec![]),
            (PanelSection::Team, rows),
            (PanelSection::Tasks, vec![]),
            (PanelSection::Graph, vec![]),
            (PanelSection::Costs, vec![]),
            (PanelSection::Logs, vec![]),
            (PanelSection::Approvals, vec![]),
        ]);
        assert_eq!(heading[0].line, "── team (1) ──");
        assert_eq!(team_worker(&heading[0]), None, "a heading is no worker");
    }

    /// The heading, the rows and the count in it all trace to the same list.
    #[test]
    fn nothing_waiting_is_reported_as_a_checked_nothing() {
        let rows = approval_rows(&[], &[]);
        assert_eq!(rows[0].line, "approvals: nothing is waiting");
        assert!(
            rows[0].sources[0].contains("both places"),
            "{:?}",
            rows[0].sources
        );
        let laid = sections([
            (PanelSection::Agents, vec![]),
            (PanelSection::Team, vec![]),
            (PanelSection::Tasks, vec![]),
            (PanelSection::Graph, vec![]),
            (PanelSection::Costs, vec![]),
            (PanelSection::Logs, vec![]),
            (PanelSection::Approvals, rows),
        ]);
        assert_eq!(laid.len(), 2, "one heading and one row");
        assert!(laid[0].is_header);
        assert_eq!(laid[0].line, "── approvals (1) ──");
        assert!(laid[0].sources[0].contains("is parked on"));
    }

    /// A section with no rows gets no heading: a heading counting zero rows would
    /// claim the panel looked and found nothing in a section it never read.
    #[test]
    fn an_empty_section_is_left_out_rather_than_counted() {
        let laid = sections([
            (PanelSection::Agents, vec![]),
            (PanelSection::Team, vec![]),
            (PanelSection::Tasks, task_rows(Some(&[]))),
            (PanelSection::Graph, vec![]),
            (PanelSection::Costs, vec![]),
            (PanelSection::Logs, vec![]),
            (PanelSection::Approvals, vec![]),
        ]);
        assert_eq!(laid.len(), 2);
        assert_eq!(laid[0].line, "── tasks (1) ──");
    }

    /// A session where no worker has put anything on the timeline still has a log
    /// section, said as the counted nothing it is. Every other builder reports its
    /// own nothing this way; a section that only appears once something happens is a
    /// part of the answer the panel withholds until it is convenient.
    #[test]
    fn a_timeline_with_no_events_reports_that_rather_than_going_missing() {
        let rows = log_rows(&[], 30);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].line, "logs: nothing has reported yet");
        assert!(
            rows[0].sources[0].contains("counted zero"),
            "{:?}",
            rows[0].sources
        );
        let laid = sections([
            (
                PanelSection::Agents,
                vec![PanelRow {
                    section: PanelSection::Agents,
                    is_header: false,
                    line: "subagent #1 — a task, not started".to_string(),
                    sources: vec!["the fleet this session launched".to_string()],
                }],
            ),
            (PanelSection::Team, team_rows(&[])),
            (PanelSection::Tasks, task_rows(Some(&[]))),
            (
                PanelSection::Graph,
                graph_rows(&[], std::path::Path::new("/nowhere")),
            ),
            (PanelSection::Costs, cost_rows(None, &[], &[])),
            (PanelSection::Logs, rows),
            (PanelSection::Approvals, approval_rows(&[], &[])),
        ]);
        let headed: Vec<PanelSection> = laid
            .iter()
            .filter(|row| row.is_header)
            .map(|row| row.section)
            .collect();
        assert_eq!(
            headed,
            Vec::from(PanelSection::ALL),
            "a fresh session shows all seven headings, so `/orchestrator status` and the panel \
             agree"
        );
    }

    /// The detail view is the trace: the line, then every figure with its source.
    #[test]
    fn the_detail_prints_the_line_then_where_each_figure_came_from() {
        let detail = PanelRow {
            section: PanelSection::Logs,
            is_header: false,
            line: "one".into(),
            sources: vec!["two".into(), "three".into()],
        }
        .detail();
        assert_eq!(detail, "one\n\nRead from\n  two\n  three");
    }

    /// The log list is bounded, and the bound is stated rather than hidden.
    #[test]
    fn a_bounded_log_says_how_many_rows_it_left_out() {
        let mut events = Vec::new();
        for i in 0..5 {
            events.push(AgentEvent::Message {
                text: format!("m{i}"),
                origin: Origin::Observed,
            });
        }
        let room = crate::control_room::ControlRoom::new(
            vec![Stream {
                worker: wr("codex", "t", None),
                events: &events,
            }],
            None,
        );
        let entries = room.timeline();
        assert_eq!(entries.len(), 5);
        let rows = log_rows(&entries, 2);
        assert_eq!(rows.len(), 3, "two events plus the line about the rest");
        assert!(
            rows[0].line.contains("3 earlier event(s) not shown"),
            "{}",
            rows[0].line
        );
        assert!(
            rows[1].sources[0].contains("event 4 of 5"),
            "{:?}",
            rows[1].sources
        );
        assert!(rows[2].line.contains("m4"), "{}", rows[2].line);
    }

    /// A recipe file that is there and cannot be read is still a row, so the
    /// panel never looks like it has seen every recipe.
    #[test]
    fn an_unreadable_file_is_named_instead_of_skipped() {
        let row = unreadable_row(PanelSection::Agents, "oops.toml", "missing field `roles`");
        assert!(row.line.contains("oops.toml"), "{}", row.line);
        assert!(row.line.contains("unreadable"), "{}", row.line);
        assert!(row.sources[0].contains("missing field `roles`"));
        assert_eq!(row.section, PanelSection::Agents);
    }
}
