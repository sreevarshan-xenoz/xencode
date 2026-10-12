//! The conversation's side panel (TD-2): what changed, the plan, how full the
//! context is, the agents at work, and the state of the project — each read
//! from state the app already keeps, so nothing here runs a command per frame.

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::app::App;
use xencode_team_rs::WorkerState;

/// Most rows a section lists before it says how many more there are.
const MAX_ROWS: usize = 6;

/// One titled part of the panel.
pub struct Section {
    pub title: &'static str,
    pub rows: Vec<Line<'static>>,
}

fn dim(app: &App, text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(
        text.into(),
        Style::default().fg(app.theme.message_system),
    ))
}

/// `+N more` once a list passes `MAX_ROWS`.
fn cap(app: &App, mut rows: Vec<Line<'static>>) -> Vec<Line<'static>> {
    if rows.len() > MAX_ROWS {
        let more = rows.len() - MAX_ROWS;
        rows.truncate(MAX_ROWS);
        rows.push(dim(app, format!("+{more} more")));
    }
    rows
}

/// `path` cut to `room` columns from the left, so the file's name survives:
/// `…/xencode-cli/src/main.rs`.
pub fn path_tail(path: &str, room: usize) -> String {
    let count = path.chars().count();
    if count <= room {
        return path.to_string();
    }
    let keep = room.saturating_sub(1);
    let tail: String = path.chars().skip(count - keep).collect();
    format!("…{tail}")
}

fn changes(app: &App, width: u16) -> Section {
    let mut files: Vec<(&String, &String)> = app.git_status.iter().collect();
    files.sort_by(|a, b| a.0.cmp(b.0));
    let rows = if files.is_empty() {
        vec![dim(app, "no changes yet")]
    } else {
        let rows = files
            .into_iter()
            .map(|(path, code)| {
                let code = code.trim();
                let color = match code.chars().next() {
                    Some('A') | Some('?') => app.theme.success,
                    Some('D') => app.theme.danger,
                    _ => app.theme.warning,
                };
                let code = if code == "??" { "+" } else { code };
                Line::from(vec![
                    Span::styled(format!("{code:<2} "), Style::default().fg(color)),
                    Span::styled(
                        path_tail(path, (width as usize).saturating_sub(4)),
                        Style::default().fg(app.theme.fg),
                    ),
                ])
            })
            .collect();
        cap(app, rows)
    };
    Section {
        title: "Changes",
        rows,
    }
}

fn plan(app: &App) -> Section {
    let items = crate::agent_tools::plan_items(&app.agent_plan);
    let rows = if items.is_empty() {
        vec![dim(app, "no plan yet")]
    } else {
        let rows = items
            .into_iter()
            .map(|item| {
                let style = match item.status {
                    crate::agent_tools::PlanStatus::Done => {
                        Style::default().fg(app.theme.message_system)
                    }
                    crate::agent_tools::PlanStatus::InProgress => Style::default()
                        .fg(app.theme.accent)
                        .add_modifier(Modifier::BOLD),
                    crate::agent_tools::PlanStatus::Pending => Style::default().fg(app.theme.fg),
                };
                Line::from(Span::styled(
                    format!("{} {}", item.status.glyph(), item.text),
                    style,
                ))
            })
            .collect();
        cap(app, rows)
    };
    Section { title: "Plan", rows }
}

/// Tokens as people read them: `950`, `12.3k`, `128k`.
pub fn short_tokens(n: u64) -> String {
    if n < 1000 {
        n.to_string()
    } else if n < 100_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        format!("{}k", n / 1000)
    }
}

fn context(app: &App, width: u16) -> Section {
    let mut rows = Vec::new();
    match (
        crate::shell::context_window(app),
        crate::shell::context_percent(app),
    ) {
        (Some(window), Some(pct)) => {
            let bar_width = width.saturating_sub(10).max(4) as usize;
            let filled = (bar_width as u64 * pct / 100) as usize;
            let color = if pct >= 85 {
                app.theme.danger
            } else if pct >= 60 {
                app.theme.warning
            } else {
                app.theme.success
            };
            rows.push(Line::from(vec![
                Span::styled("█".repeat(filled), Style::default().fg(color)),
                Span::styled(
                    "░".repeat(bar_width - filled),
                    Style::default().fg(app.theme.message_system),
                ),
                Span::styled(format!(" {pct}%"), Style::default().fg(color)),
            ]));
            rows.push(dim(
                app,
                format!(
                    "{} / {} tokens",
                    short_tokens(app.last_prompt_tokens),
                    short_tokens(u64::from(window))
                ),
            ));
        }
        _ => rows.push(dim(app, "not measured yet")),
    }
    rows.push(Line::from(Span::styled(
        app.config.default_model.clone(),
        Style::default().fg(app.theme.accent),
    )));
    if let Some(spend) = &app.spend {
        rows.push(dim(app, spend.line.clone()));
    }
    rows.push(dim(
        app,
        format!("approval: {}", crate::shell::approval_label(app)),
    ));
    Section {
        title: "Context",
        rows,
    }
}

fn worker_glyph(state: WorkerState) -> &'static str {
    match state {
        WorkerState::Starting => "◌",
        WorkerState::Working => "●",
        WorkerState::NeedsYou => "!",
        WorkerState::Done => "✓",
        WorkerState::Failed => "✕",
        WorkerState::Stopped => "■",
    }
}

fn agents(app: &App) -> Section {
    let mut workers = app.team_shown();
    // The ones waiting on the person come first: that is why they look here.
    workers.sort_by_key(|w| w.state != WorkerState::NeedsYou);
    let mut rows: Vec<Line<'static>> = workers
        .iter()
        .map(|w| {
            let color = match w.state {
                WorkerState::NeedsYou => app.theme.warning,
                WorkerState::Failed => app.theme.danger,
                WorkerState::Done => app.theme.success,
                WorkerState::Working | WorkerState::Starting => app.theme.accent,
                WorkerState::Stopped => app.theme.message_system,
            };
            Line::from(vec![
                Span::styled(
                    format!("{} ", worker_glyph(w.state)),
                    Style::default().fg(color),
                ),
                Span::styled(
                    format!("{} {}", w.agent, w.task),
                    Style::default().fg(app.theme.fg),
                ),
            ])
        })
        .collect();
    for spawn in app.spawns.iter().filter(|s| s.running) {
        rows.push(Line::from(vec![
            Span::styled("● ", Style::default().fg(app.theme.accent)),
            Span::styled(
                format!("spawn {} {}", spawn.id, spawn.task),
                Style::default().fg(app.theme.fg),
            ),
        ]));
    }
    let rows = if rows.is_empty() {
        vec![dim(app, "none running · ctrl+space")]
    } else {
        cap(app, rows)
    };
    Section {
        title: "Agents",
        rows,
    }
}

fn project(app: &App) -> Section {
    let mut rows = Vec::new();
    let branch = if app.git_branch.is_empty() {
        "no git branch".to_string()
    } else {
        app.git_branch.clone()
    };
    let changed = app.git_status.len();
    rows.push(Line::from(vec![
        Span::styled(branch, Style::default().fg(app.theme.fg)),
        Span::styled(
            if changed == 0 {
                " · clean".to_string()
            } else {
                format!(" · {changed} changed")
            },
            Style::default().fg(app.theme.message_system),
        ),
    ]));
    let tools = app.mcp.definitions().len();
    if tools > 0 {
        rows.push(dim(app, format!("MCP: {tools} tools")));
    }
    Section {
        title: "Project",
        rows,
    }
}

/// The panel's sections, in the order they are drawn.
pub fn sections(app: &App, width: u16) -> Vec<Section> {
    vec![
        changes(app, width),
        plan(app),
        context(app, width),
        agents(app),
        project(app),
    ]
}

/// Cut a line to `width` columns, keeping its styles.
fn clip(line: Line<'static>, width: usize) -> Line<'static> {
    let mut left = width;
    let mut spans = Vec::new();
    for span in line.spans {
        if left == 0 {
            break;
        }
        let text: String = span.content.chars().take(left).collect();
        left -= text.chars().count();
        spans.push(Span::styled(text, span.style));
    }
    Line::from(spans)
}

/// Draw the panel into `area`.
pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let title = if app.git_branch.is_empty() {
        " xencode ".to_string()
    } else {
        format!(" xencode · {} ", app.git_branch)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(crate::widgets::panel_border_set(app.config.rounded_borders))
        .border_style(Style::default().fg(app.theme.border))
        .title(Span::styled(title, Style::default().fg(app.theme.accent)));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(1) as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();
    for (i, section) in sections(app, inner.width).into_iter().enumerate() {
        if i > 0 {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            section.title,
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::BOLD),
        )));
        lines.extend(section.rows.into_iter().map(|row| clip(row, width)));
    }
    let padded = Rect {
        x: inner.x + 1,
        width: inner.width.saturating_sub(1),
        ..inner
    };
    f.render_widget(Paragraph::new(lines), padded);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_read_the_way_people_say_them() {
        assert_eq!(short_tokens(950), "950");
        assert_eq!(short_tokens(12_300), "12.3k");
        assert_eq!(short_tokens(128_000), "128k");
    }

    #[test]
    fn a_long_path_keeps_its_file_name() {
        assert_eq!(path_tail("src/a.rs", 20), "src/a.rs");
        assert_eq!(path_tail("rust/crates/xencode-cli/src/main.rs", 12), "…src/main.rs");
    }

    #[test]
    fn a_long_row_is_cut_to_the_panel() {
        let line = Line::from(vec![Span::raw("ab"), Span::raw("cdef")]);
        let cut = clip(line, 4);
        let text: String = cut.spans.iter().map(|s| s.content.to_string()).collect();
        assert_eq!(text, "abcd");
    }
}
