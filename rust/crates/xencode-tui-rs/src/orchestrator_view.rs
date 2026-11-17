//! Orchestrator mode (OM-3): a rail of every agent on the left, and tiles
//! with their live screens. Tile 0 is xencode's own conversation; the others
//! are terminal agents the engine runs, each in a worktree of its own.
//!
//! Keys go to the focused tile. Ctrl+A is the prefix for the mode's own
//! commands: a digit focuses that tile, `z` zooms, `t` tiles every agent, `n`
//! opens the picker, `x` stops the focused agent, Ctrl+A sends a Ctrl+A.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

use crate::app::App;

/// Columns the rail takes.
pub const RAIL_WIDTH: u16 = 24;
/// Below this width there is no rail and only the focused tile is drawn.
pub const RAIL_MIN_WIDTH: u16 = 100;
/// Tiles drawn at once unless every agent is asked for (`t`).
pub const GRID_TILES: usize = 4;

/// The mode's own state.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OrchState {
    /// The focused tile: 0 is the conversation, `n` the n-th terminal agent.
    pub focus: usize,
    pub zoom: bool,
    pub all: bool,
    /// Ctrl+A was pressed; the next key is a command.
    pub prefix: bool,
    /// The picker, open with the row under the cursor and the chosen names.
    pub picker: Option<Picker>,
    /// `x` was pressed once on this tile; a second `x` stops it.
    pub stop_armed: Option<String>,
    /// `m` was pressed once on this tile; a second `m` stops it and merges.
    pub merge_armed: Option<String>,
    /// The line `a` (ask all) is typing, sent to every agent on Enter.
    pub ask: Option<String>,
    /// The size each tile was last told, so a resize is only sent on change.
    pub sizes: std::collections::BTreeMap<String, (u16, u16)>,
    /// Sizes to send to the engine on the next frame.
    pub pending_resize: Vec<(String, u16, u16)>,
    /// Tiles whose output this window already asked for.
    pub watched: std::collections::BTreeSet<String>,
}

/// The list of agents a new tile can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picker {
    pub agents: Vec<crate::engine::term::PickerAgent>,
    pub cursor: usize,
    pub chosen: std::collections::BTreeSet<String>,
}

/// One thing waiting on the person (OM-7). Built only from real `App` state,
/// and shared by the Inbox and the window title so they cannot disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Waiting {
    /// A question the engine waits to have answered (EN-1).
    Question { id: u64, text: String },
    /// An approval prompt, local or from an engine window (OM-4, EN-2).
    /// The id can be absent in state built outside the normal tool loop.
    Approval { id: Option<u64>, summary: String },
    /// A ByteBot task whose changes wait for review (BT-3).
    Review { task_id: String, text: String },
    /// A terminal agent that rang or sent a notification (OM-4). `n` is its
    /// number in the rail, where 0 is xencode's own conversation.
    Tile { n: usize, id: String, agent: String },
}

impl Waiting {
    /// The key used to notice this item newly waiting, for the one desktop
    /// notification and the attention count in the window title.
    pub fn key(&self) -> String {
        match self {
            Waiting::Question { id, .. } => format!("question {id}"),
            Waiting::Approval { id: Some(id), .. } => format!("approval {id}"),
            Waiting::Approval { id: None, .. } => "approval".to_string(),
            Waiting::Review { task_id, .. } => format!("review {task_id}"),
            Waiting::Tile { id, .. } => format!("tile {id}"),
        }
    }

    /// The Inbox row, before the `+N` for further approvals is appended.
    pub fn label(&self) -> String {
        match self {
            Waiting::Question { text, .. } => format!("? {text}"),
            Waiting::Approval { summary, .. } => format!("? {summary}"),
            Waiting::Review { text, .. } => format!("! {text} needs review"),
            Waiting::Tile { n, agent, .. } => format!("! {n} {agent} called"),
        }
    }
}

/// Everything that waits on the person, in the order the Inbox shows it:
/// the question, pending approvals, a ByteBot review, then calling agents.
/// Pure: it reads `App` state and changes nothing. This is the one definition
/// used by both the Inbox and `shell::waiting_keys`/`attention_title`.
pub fn waiting_on_user(app: &App) -> Vec<Waiting> {
    let mut out = Vec::new();
    if let Some(id) = app.question_id {
        out.push(Waiting::Question {
            id,
            text: app.question_text.clone().unwrap_or_default(),
        });
    }
    for (i, (request, _)) in app.approval_queue.iter().enumerate() {
        out.push(Waiting::Approval {
            id: app.approval_ids.get(i).copied(),
            summary: crate::transcript::clean(&request.summary),
        });
    }
    for (id, request) in &app.remote_approvals {
        out.push(Waiting::Approval {
            id: Some(*id),
            summary: crate::transcript::clean(&request.summary),
        });
    }
    if let Some(i) = app.bytebot_reviewing() {
        let task = &app.bytebot_tasks[i];
        out.push(Waiting::Review {
            task_id: task.id.clone(),
            text: task.text.clone(),
        });
    }
    for (i, tile) in app.term_tiles.iter().enumerate() {
        if tile.state == "needs_you" {
            out.push(Waiting::Tile {
                n: i + 1,
                id: tile.id.clone(),
                agent: tile.agent.clone(),
            });
        }
    }
    out
}

/// The Inbox rows for `waiting`. Every approval after the first is folded
/// into a `+N` on the first row, as the Inbox has always done (OM-4).
fn inbox_lines(app: &App, waiting: &[Waiting]) -> Vec<Line<'static>> {
    let approvals = waiting
        .iter()
        .filter(|w| matches!(w, Waiting::Approval { .. }))
        .count();
    let mut shown = 0;
    let mut lines = Vec::new();
    for item in waiting {
        let mut label = item.label();
        if let Waiting::Approval { .. } = item {
            shown += 1;
            if shown > 1 {
                continue;
            }
            if approvals > 1 {
                label.push_str(&format!(" +{}", approvals - 1));
            }
        }
        lines.push(Line::from(Span::styled(
            label,
            Style::default().fg(app.theme.warning),
        )));
    }
    lines
}

/// Where the rail and each tile go in `area`, for `tiles` tiles (the
/// conversation included). A tile not drawn gets an empty rect.
pub fn tile_rects(
    area: Rect,
    tiles: usize,
    focus: usize,
    zoom: bool,
    all: bool,
) -> (Option<Rect>, Vec<Rect>) {
    let mut rects = vec![Rect::default(); tiles];
    if tiles == 0 || area.width == 0 || area.height == 0 {
        return (None, rects);
    }
    let focus = focus.min(tiles - 1);
    let rail_shown = area.width >= RAIL_MIN_WIDTH;
    let (rail, body) = if rail_shown {
        (
            Some(Rect {
                width: RAIL_WIDTH,
                ..area
            }),
            Rect {
                x: area.x + RAIL_WIDTH,
                width: area.width - RAIL_WIDTH,
                ..area
            },
        )
    } else {
        (None, area)
    };
    // Zoomed, or too narrow for a grid: one tile fills the body.
    if zoom || !rail_shown {
        rects[focus] = body;
        return (rail, rects);
    }
    // The tiles drawn: every one with `t`, otherwise the first four, with the
    // focused tile always among them.
    let mut shown: Vec<usize> = if all {
        (0..tiles).collect()
    } else {
        (0..tiles.min(GRID_TILES)).collect()
    };
    if !shown.contains(&focus) {
        shown.pop();
        shown.push(focus);
    }
    let n = shown.len();
    let cols = (n as f64).sqrt().ceil() as u16;
    let rows = (n as u16).div_ceil(cols);
    let cell_w = body.width / cols.max(1);
    let cell_h = body.height / rows.max(1);
    for (i, tile) in shown.into_iter().enumerate() {
        let c = i as u16 % cols;
        let r = i as u16 / cols;
        // The last column and row take what the division left over.
        let w = if c == cols - 1 {
            body.width - cell_w * c
        } else {
            cell_w
        };
        let h = if r == rows - 1 {
            body.height - cell_h * r
        } else {
            cell_h
        };
        rects[tile] = Rect {
            x: body.x + cell_w * c,
            y: body.y + cell_h * r,
            width: w,
            height: h,
        };
    }
    (rail, rects)
}

/// The bytes a terminal sends for `key`, as a program in the tile expects
/// them. `None` for a key a terminal has no bytes for.
pub fn key_bytes(key: KeyEvent) -> Option<Vec<u8>> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let mut bytes = match key.code {
        KeyCode::Char(c) if ctrl => {
            let lower = c.to_ascii_lowercase();
            match lower {
                'a'..='z' => vec![lower as u8 - b'a' + 1],
                ' ' | '@' | '2' => vec![0],
                '[' | '3' => vec![0x1b],
                '\\' | '4' => vec![0x1c],
                ']' | '5' => vec![0x1d],
                '^' | '6' => vec![0x1e],
                '_' | '7' | '/' => vec![0x1f],
                _ => return None,
            }
        }
        KeyCode::Char(c) => c.to_string().into_bytes(),
        KeyCode::Enter => vec![b'\r'],
        KeyCode::Tab => vec![b'\t'],
        KeyCode::BackTab => b"\x1b[Z".to_vec(),
        KeyCode::Backspace => vec![0x7f],
        KeyCode::Esc => vec![0x1b],
        KeyCode::Up => b"\x1b[A".to_vec(),
        KeyCode::Down => b"\x1b[B".to_vec(),
        KeyCode::Right => b"\x1b[C".to_vec(),
        KeyCode::Left => b"\x1b[D".to_vec(),
        KeyCode::Home => b"\x1b[H".to_vec(),
        KeyCode::End => b"\x1b[F".to_vec(),
        KeyCode::PageUp => b"\x1b[5~".to_vec(),
        KeyCode::PageDown => b"\x1b[6~".to_vec(),
        KeyCode::Insert => b"\x1b[2~".to_vec(),
        KeyCode::Delete => b"\x1b[3~".to_vec(),
        KeyCode::F(n) => match n {
            1 => b"\x1bOP".to_vec(),
            2 => b"\x1bOQ".to_vec(),
            3 => b"\x1bOR".to_vec(),
            4 => b"\x1bOS".to_vec(),
            5 => b"\x1b[15~".to_vec(),
            6 => b"\x1b[17~".to_vec(),
            7 => b"\x1b[18~".to_vec(),
            8 => b"\x1b[19~".to_vec(),
            9 => b"\x1b[20~".to_vec(),
            10 => b"\x1b[21~".to_vec(),
            11 => b"\x1b[23~".to_vec(),
            12 => b"\x1b[24~".to_vec(),
            _ => return None,
        },
        _ => return None,
    };
    // Alt is the escape prefix a terminal puts in front.
    if alt {
        bytes.insert(0, 0x1b);
    }
    Some(bytes)
}

/// The glyph a rail row and a tile title show for an agent's state.
fn glyph(state: &str) -> &'static str {
    match state {
        "needs_you" => "!",
        "working" => "●",
        "idle" => "○",
        "done" => "✓",
        "failed" => "✕",
        _ => "·",
    }
}

/// The tile's title: its number, the agent, its state.
fn title(n: usize, name: &str, state: &str) -> String {
    format!(
        " {n} {} {name} · {} ",
        glyph(state),
        state.replace('_', " ")
    )
}

/// Draw the mode over the whole of `area`.
pub fn draw(f: &mut Frame, app: &mut App, area: Rect) {
    let tiles = app.term_tiles.clone();
    let count = tiles.len() + 1;
    app.orch.focus = app.orch.focus.min(count - 1);
    let (rail, rects) = tile_rects(area, count, app.orch.focus, app.orch.zoom, app.orch.all);
    if let Some(rail) = rail {
        draw_rail(f, app, rail, &tiles);
    }
    for (i, rect) in rects.iter().enumerate() {
        if rect.width < 3 || rect.height < 3 {
            continue;
        }
        let focused = i == app.orch.focus;
        let (name, state) = if i == 0 {
            (
                "xencode".to_string(),
                if app.is_generating { "working" } else { "idle" }.to_string(),
            )
        } else {
            let t = &tiles[i - 1];
            (t.agent.clone(), t.state.clone())
        };
        let color = if state == "needs_you" {
            app.theme.warning
        } else if focused {
            app.theme.accent
        } else {
            app.theme.border
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_set(crate::widgets::panel_border_set(app.config.rounded_borders))
            .border_style(Style::default().fg(color))
            .title(Span::styled(
                title(i, &name, &state),
                Style::default().fg(color).add_modifier(if focused {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
            ));
        let inner = block.inner(*rect);
        f.render_widget(Clear, *rect);
        f.render_widget(block, *rect);
        if i == 0 {
            crate::shell::draw(f, app, inner);
            continue;
        }
        let tile = &tiles[i - 1];
        // The program is told the size of the tile that draws it.
        let size = (inner.width, inner.height);
        if app.orch.sizes.get(&tile.id) != Some(&size) && size.0 > 0 && size.1 > 0 {
            app.orch.sizes.insert(tile.id.clone(), size);
            app.orch
                .pending_resize
                .push((tile.id.clone(), size.0, size.1));
        }
        match app.term_screens.get(&tile.id) {
            Some(screen) => xencode_term_rs::draw(screen.vt(), inner, f.buffer_mut()),
            None => f.render_widget(
                Paragraph::new(Span::styled(
                    "starting…",
                    Style::default().fg(app.theme.message_system),
                )),
                inner,
            ),
        }
        if let Some(code) = tile.exit {
            let note = Rect {
                y: inner.y + inner.height.saturating_sub(1),
                height: 1,
                ..inner
            };
            f.render_widget(
                Paragraph::new(Span::styled(
                    format!(" exited (code {code}) "),
                    Style::default().fg(app.theme.danger),
                )),
                note,
            );
        }
    }
    if app.orch.picker.is_some() {
        draw_picker(f, app, area);
    }
    if let Some(text) = app.orch.ask.as_ref() {
        let row = Rect {
            y: area.y + area.height.saturating_sub(1),
            height: 1.min(area.height),
            ..area
        };
        f.render_widget(Clear, row);
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    " ask every agent › ",
                    Style::default()
                        .fg(app.theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(text.clone(), Style::default().fg(app.theme.fg)),
                Span::styled(
                    "  enter sends · esc cancels",
                    Style::default().fg(app.theme.message_system),
                ),
            ])),
            row,
        );
    }
}

fn draw_rail(f: &mut Frame, app: &App, area: Rect, tiles: &[crate::engine::term::TileInfo]) {
    let dim = Style::default().fg(app.theme.message_system);
    let mut lines: Vec<Line<'static>> = Vec::new();
    let row = |n: usize, name: &str, state: &str, branch: &str, focused: bool| {
        let style = if state == "needs_you" {
            Style::default()
                .fg(app.theme.warning)
                .add_modifier(Modifier::BOLD)
        } else if focused {
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(app.theme.fg)
        };
        let mut spans = vec![Span::styled(format!("{n} {} {name}", glyph(state)), style)];
        if !branch.is_empty() {
            let short = branch.rsplit('/').next().unwrap_or(branch).to_string();
            spans.push(Span::styled(format!(" {short}"), dim));
        }
        Line::from(spans)
    };
    let groups = [
        ("Needs you", &["needs_you"][..]),
        ("Working", &["working"][..]),
        ("Idle", &["idle"][..]),
        ("Done", &["done", "failed"][..]),
    ];
    let conversation_state = if app.is_generating { "working" } else { "idle" };
    for (heading, states) in groups {
        let mut rows: Vec<Line<'static>> = Vec::new();
        if states.contains(&conversation_state) {
            rows.push(row(
                0,
                "xencode",
                conversation_state,
                "",
                app.orch.focus == 0,
            ));
        }
        for (i, t) in tiles.iter().enumerate() {
            if states.contains(&t.state.as_str()) {
                rows.push(row(
                    i + 1,
                    &t.agent,
                    &t.state,
                    &t.branch,
                    app.orch.focus == i + 1,
                ));
            }
        }
        if !rows.is_empty() {
            lines.push(Line::from(Span::styled(
                heading.to_string(),
                Style::default().add_modifier(Modifier::BOLD),
            )));
            lines.extend(rows);
            lines.push(Line::from(""));
        }
    }
    lines.push(Line::from(Span::styled("+ new agent (ctrl+a n)", dim)));
    // What waits on the person, from the same list the title counts (OM-7).
    let inbox = inbox_lines(app, &waiting_on_user(app));
    if !inbox.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Inbox".to_string(),
            Style::default().add_modifier(Modifier::BOLD),
        )));
        lines.extend(inbox);
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(crate::widgets::panel_border_set(app.config.rounded_borders))
        .border_style(Style::default().fg(app.theme.border))
        .title(Span::styled(
            " Agents ",
            Style::default().fg(app.theme.accent),
        ));
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_picker(f: &mut Frame, app: &App, area: Rect) {
    let Some(picker) = app.orch.picker.as_ref() else {
        return;
    };
    let width = area.width.min(70);
    let height = (picker.agents.len() as u16 + 6).min(area.height);
    let rect = Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    };
    let dim = Style::default().fg(app.theme.message_system);
    let mut lines: Vec<Line<'static>> = Vec::new();
    for (i, agent) in picker.agents.iter().enumerate() {
        let mark = if picker.chosen.contains(&agent.name) {
            "[x]"
        } else {
            "[ ]"
        };
        let style = if i == picker.cursor {
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD)
        } else if agent.installed {
            Style::default().fg(app.theme.fg)
        } else {
            dim
        };
        let status = if agent.installed {
            "installed"
        } else {
            "missing"
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{mark} {:<14}", agent.name), style),
            Span::styled(status.to_string(), dim),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "each runs its own CLI with your sign-in, in its own worktree",
        dim,
    )));
    lines.push(Line::from(Span::styled(
        "space choose · enter open · esc close",
        dim,
    )));
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(crate::widgets::panel_border_set(app.config.rounded_borders))
        .border_style(Style::default().fg(app.theme.accent))
        .title(Span::styled(
            " Open agents ",
            Style::default().fg(app.theme.accent),
        ));
    f.render_widget(Clear, rect);
    f.render_widget(Paragraph::new(lines).block(block), rect);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approval(summary: &str) -> crate::agent_tools::ApprovalRequest {
        crate::agent_tools::ApprovalRequest {
            tool: "run_shell".into(),
            class: crate::agent_tools::ToolClass::Shell,
            summary: summary.into(),
            preview: String::new(),
            draft: Default::default(),
        }
    }

    fn review_task(text: &str) -> crate::bytebot_tasks::ByteBotTask {
        let mut task = crate::bytebot_tasks::ByteBotTask::new(text, "test-model");
        task.state = crate::bytebot_tasks::TaskState::NeedsReview;
        task
    }

    fn tile(id: &str, state: &str) -> crate::engine::term::TileInfo {
        crate::engine::term::TileInfo {
            id: id.into(),
            agent: "shell".into(),
            branch: String::new(),
            worktree: String::new(),
            state: state.into(),
            cols: 80,
            rows: 24,
            exit: None,
        }
    }

    #[test]
    fn a_pending_question_appears_with_its_text() {
        let mut app = App::for_tests();
        app.question_id = Some(7);
        app.question_text = Some("Which database should I use?".into());
        let waiting = waiting_on_user(&app);
        assert_eq!(
            waiting,
            vec![Waiting::Question {
                id: 7,
                text: "Which database should I use?".into()
            }]
        );
        let rows = inbox_lines(&app, &waiting);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].to_string().contains("Which database should I use?"));
    }

    #[test]
    fn a_task_waiting_for_review_appears() {
        let mut app = App::for_tests();
        app.bytebot_tasks.push(review_task("write the parser"));
        let waiting = waiting_on_user(&app);
        assert_eq!(waiting.len(), 1);
        assert!(matches!(
            &waiting[0],
            Waiting::Review { text, .. } if text == "write the parser"
        ));
        let rows = inbox_lines(&app, &waiting);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].to_string().contains("write the parser"));
    }

    #[test]
    fn approvals_and_calling_tiles_still_appear_with_the_plus_summary() {
        let mut app = App::for_tests();
        let (first, _answer) = tokio::sync::oneshot::channel();
        app.approval_queue
            .push_back((approval("run the tests"), first));
        app.approval_ids.push_back(11);
        let (second, _answer) = tokio::sync::oneshot::channel();
        app.approval_queue
            .push_back((approval("commit the change"), second));
        app.approval_ids.push_back(12);
        app.term_tiles = vec![tile("t1", "needs_you"), tile("t2", "working")];
        let waiting = waiting_on_user(&app);
        assert_eq!(
            waiting,
            vec![
                Waiting::Approval {
                    id: Some(11),
                    summary: "run the tests".into()
                },
                Waiting::Approval {
                    id: Some(12),
                    summary: "commit the change".into()
                },
                Waiting::Tile {
                    n: 1,
                    id: "t1".into(),
                    agent: "shell".into()
                },
            ]
        );
        let rows: Vec<String> = inbox_lines(&app, &waiting)
            .iter()
            .map(|line| line.to_string())
            .collect();
        assert_eq!(rows, ["? run the tests +1", "! 1 shell called"]);
    }

    #[test]
    fn nothing_waiting_yields_no_inbox_rows() {
        let app = App::for_tests();
        assert!(waiting_on_user(&app).is_empty());
        assert!(inbox_lines(&app, &waiting_on_user(&app)).is_empty());
    }

    #[test]
    fn attention_title_counts_the_question_and_the_review_once_each() {
        let mut app = App::for_tests();
        assert!(!crate::shell::attention_title(&app).contains("need you"));
        app.question_id = Some(3);
        app.question_text = Some("pick one".into());
        assert_eq!(crate::shell::attention_title(&app), "xencode · needs you");
        app.bytebot_tasks.push(review_task("ship it"));
        assert_eq!(crate::shell::attention_title(&app), "xencode · 2 need you");
        assert_eq!(
            crate::shell::waiting_keys(&app).len(),
            waiting_on_user(&app).len()
        );
    }

    #[test]
    fn a_wide_screen_has_a_rail_and_a_grid_of_four() {
        let (rail, rects) = tile_rects(Rect::new(0, 0, 160, 48), 3, 0, false, false);
        assert_eq!(rail.map(|r| r.width), Some(RAIL_WIDTH));
        let body_w = 160 - RAIL_WIDTH;
        // Three tiles in a two-by-two grid: two on top, one under.
        assert_eq!(rects[0], Rect::new(RAIL_WIDTH, 0, body_w / 2, 24));
        assert_eq!(
            rects[1],
            Rect::new(RAIL_WIDTH + body_w / 2, 0, body_w - body_w / 2, 24)
        );
        assert_eq!(rects[2], Rect::new(RAIL_WIDTH, 24, body_w / 2, 24));
    }

    #[test]
    fn zoom_gives_the_focused_tile_the_whole_body() {
        let (rail, rects) = tile_rects(Rect::new(0, 0, 160, 48), 3, 1, true, false);
        assert!(rail.is_some());
        assert_eq!(rects[1], Rect::new(RAIL_WIDTH, 0, 160 - RAIL_WIDTH, 48));
        assert_eq!(rects[0], Rect::default());
    }

    #[test]
    fn a_narrow_screen_draws_only_the_focused_tile() {
        let (rail, rects) = tile_rects(Rect::new(0, 0, 80, 24), 3, 2, false, false);
        assert!(rail.is_none());
        assert_eq!(rects[2], Rect::new(0, 0, 80, 24));
        assert_eq!(rects[0], Rect::default());
    }

    #[test]
    fn past_four_tiles_the_focused_one_is_always_drawn_and_t_draws_all() {
        let (_, rects) = tile_rects(Rect::new(0, 0, 160, 48), 7, 6, false, false);
        assert_ne!(rects[6], Rect::default(), "the focused tile is drawn");
        assert_eq!(rects.iter().filter(|r| **r != Rect::default()).count(), 4);
        let (_, all) = tile_rects(Rect::new(0, 0, 160, 48), 7, 6, false, true);
        assert_eq!(all.iter().filter(|r| **r != Rect::default()).count(), 7);
    }

    #[test]
    fn keys_become_the_bytes_a_terminal_sends() {
        let k = |code, mods| key_bytes(KeyEvent::new(code, mods));
        assert_eq!(
            k(KeyCode::Char('x'), KeyModifiers::NONE),
            Some(b"x".to_vec())
        );
        assert_eq!(k(KeyCode::Char('c'), KeyModifiers::CONTROL), Some(vec![3]));
        assert_eq!(k(KeyCode::Char('a'), KeyModifiers::CONTROL), Some(vec![1]));
        assert_eq!(k(KeyCode::Enter, KeyModifiers::NONE), Some(b"\r".to_vec()));
        assert_eq!(k(KeyCode::Up, KeyModifiers::NONE), Some(b"\x1b[A".to_vec()));
        assert_eq!(
            k(KeyCode::Char('b'), KeyModifiers::ALT),
            Some(b"\x1bb".to_vec())
        );
        assert_eq!(
            k(KeyCode::Char('é'), KeyModifiers::NONE),
            Some("é".as_bytes().to_vec())
        );
    }
}
