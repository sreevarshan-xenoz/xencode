//! The conversation screen (TD-2): the transcript fills the screen, an
//! approval waiting on the person is drawn inside it, the input box is pinned
//! to the bottom with one status line under it, and a slim side panel sits on
//! the right when there is room.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::app::App;
use crate::focus::InputMode;
use crate::widgets::spinner;

/// Columns the side panel takes.
pub const SIDE_WIDTH: u16 = 32;
/// Below this width `side_panel = "auto"` hides the panel.
pub const SIDE_MIN_WIDTH: u16 = 100;
/// Most rows of draft the input box grows to, not counting its frame.
const INPUT_MAX_ROWS: usize = 6;
/// The hints on the right of the status line, longest first; the first that
/// fits is drawn.
const HINTS: [&str; 3] = [
    "ctrl+p commands · ctrl+b panel · ? keys",
    "ctrl+p · ? keys",
    "?",
];
/// The same in a view (TD-4).
const VIEW_HINTS: [&str; 3] = ["tab next tab · esc back · ctrl+p", "tab · esc back", "esc"];

/// Where each part of the conversation screen goes this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShellAreas {
    pub transcript: Rect,
    pub approval: Option<Rect>,
    pub input: Rect,
    pub status: Rect,
    pub side: Option<Rect>,
    /// The transcript's last row with text on it, counted from its top: what
    /// lies below is empty, which is where a toast can go without covering
    /// anything. Known once the frame is drawn; `areas` leaves it 0.
    pub used: u16,
}

/// Whether the conversation screen is the one on show: the Conversation
/// view, including under a panel that belongs to no view (TD-4).
pub fn active(app: &App) -> bool {
    app.screen() == crate::screens::Screen::Conversation
}

/// Whether the side panel shows at `width` columns.
pub fn side_visible(app: &App, width: u16) -> bool {
    match app.config.side_panel.as_str() {
        "off" => false,
        // "on" still gives way when the conversation would be squeezed out.
        "on" => app.side_panel_open && width >= SIDE_WIDTH + 40,
        _ => app.side_panel_open && width >= SIDE_MIN_WIDTH,
    }
}

/// The approval mode in the words the status line uses.
pub fn approval_label(app: &App) -> &'static str {
    match app.config.agent_approval.as_str() {
        "edit-allow" => "auto-edit",
        "all-allow" => "allow all",
        "plan" => "plan",
        "autonomous" => "autonomous",
        _ => "ask",
    }
}

/// The model's context window, when a server has said what it is.
pub fn context_window(app: &App) -> Option<u32> {
    app.server_context_window
        .or(app.ollama_window)
        .filter(|w| *w > 0)
}

/// The share of the context window the last prompt took, when both are known.
pub fn context_percent(app: &App) -> Option<u64> {
    let window = u64::from(context_window(app)?);
    if app.last_prompt_tokens == 0 {
        return None;
    }
    Some((app.last_prompt_tokens * 100 / window).min(100))
}

/// Where a prompt is allowed to go (PR-2), in the words the old status bar
/// used, read from the same policy the router enforces.
pub fn egress_label(app: &App) -> &'static str {
    if app.egress_policy().allow_cloud {
        "cloud allowed"
    } else {
        "local only"
    }
}

/// The notification a terminal that is not in front gets once when an
/// approval starts waiting: OSC 9, which Windows Terminal, iTerm2, WezTerm and
/// kitty show as a desktop notification and other terminals ignore.
pub const NOTIFY_SEQUENCE: &str = "\x1b]9;xencode needs you\x1b\\";

/// Everything waiting on the person, one key each: every approval, and
/// every terminal agent that rang or sent a notification (OM-4). A new key
/// is what a notification is sent for.
pub fn waiting_keys(app: &App) -> std::collections::BTreeSet<String> {
    let mut keys = std::collections::BTreeSet::new();
    for id in &app.approval_ids {
        keys.insert(format!("approval {id}"));
    }
    for (id, _) in &app.remote_approvals {
        keys.insert(format!("approval {id}"));
    }
    if app.pending_approval().is_some() && keys.is_empty() {
        keys.insert("approval".to_string());
    }
    for tile in app.term_tiles.iter().filter(|t| t.state == "needs_you") {
        keys.insert(format!("tile {}", tile.id));
    }
    keys
}

/// The window title: the project, or a call for the person while something
/// waits on their answer.
pub fn attention_title(app: &App) -> String {
    match waiting_keys(app).len() {
        0 => {}
        1 => return "xencode · needs you".to_string(),
        n => return format!("xencode · {n} need you"),
    }
    let project = xencode_context_rs::default_root()
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    format!("xencode · {project}")
}

/// `text` cut into rows of at most `width` columns, every character kept: a
/// command's tail past the box's edge is exactly what must not be hidden.
pub fn hard_wrap(text: &str, width: usize) -> Vec<String> {
    use unicode_width::UnicodeWidthChar;
    let width = width.max(1);
    let mut rows = vec![String::new()];
    let mut used = 0;
    for c in text.chars() {
        let w = c.width().unwrap_or(0);
        if used + w > width && used > 0 {
            rows.push(String::new());
            used = 0;
        }
        rows.last_mut().unwrap().push(c);
        used += w;
    }
    rows
}

/// The approval prompt at `width` columns of text: its title, the call and
/// the change wrapped so nothing is cut off, and the keys line.
fn approval_lines(app: &App, width: usize) -> Option<(String, Vec<Line<'static>>, Line<'static>)> {
    let request = app.pending_approval()?;
    let waiting = app.approval_queue.len() + app.remote_approvals.len();
    let more = if waiting > 1 {
        format!(" · +{} more", waiting - 1)
    } else {
        String::new()
    };
    let title = format!(" Allow {}?{more} ", request.class_label());
    let bold = Style::default()
        .fg(app.theme.fg)
        .add_modifier(Modifier::BOLD);
    let mut body: Vec<Line<'static>> =
        hard_wrap(&crate::transcript::visible(&request.summary), width)
            .into_iter()
            .map(|row| Line::from(Span::styled(row, bold)))
            .collect();
    for line in request.preview.lines() {
        let color = if line.starts_with("@@") {
            app.theme.accent
        } else if line.starts_with('+') {
            app.theme.success
        } else if line.starts_with('-') {
            app.theme.danger
        } else {
            app.theme.message_system
        };
        for row in hard_wrap(&crate::transcript::visible(line), width) {
            body.push(Line::from(Span::styled(row, Style::default().fg(color))));
        }
    }
    let key = |k: &'static str| {
        Span::styled(
            k,
            Style::default()
                .fg(app.theme.warning)
                .add_modifier(Modifier::BOLD),
        )
    };
    let word = |w: &'static str| Span::styled(w, Style::default().fg(app.theme.message_system));
    let keys = Line::from(vec![
        key("y"),
        word(" allow · "),
        key("a"),
        word(" always for this kind · "),
        key("n"),
        word(" deny"),
    ]);
    Some((title, body, keys))
}

/// Where everything goes in `area`.
pub fn areas(area: Rect, app: &App) -> ShellAreas {
    let draft_rows = app.chat_input.lines().len().clamp(1, INPUT_MAX_ROWS) as u16;
    // The frame always keeps one row of transcript and the status line.
    let input_height = (draft_rows + 2).min(area.height.saturating_sub(2)).max(1);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(area);
    let (top, input, status) = (rows[0], rows[1], rows[2]);
    let (top, side) = if side_visible(app, area.width) && top.height >= 3 {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(SIDE_WIDTH)])
            .split(top);
        (cols[0], Some(cols[1]))
    } else {
        (top, None)
    };
    let (transcript, approval) = match approval_lines(app, approval_text_width(top.width)) {
        Some((_, body, _)) if top.height >= 6 => {
            // The change gets up to two thirds of the rows; past that it scrolls.
            let wanted = body.len() as u16 + 3;
            let height = wanted.min(top.height * 2 / 3).max(4);
            let split = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(height)])
                .split(top);
            (split[0], Some(split[1]))
        }
        Some(_) => (Rect { height: 0, ..top }, Some(top)),
        None => (top, None),
    };
    ShellAreas {
        transcript,
        approval,
        input,
        status,
        side,
        used: 0,
    }
}

/// Draw the conversation screen into `area`.
pub fn draw(f: &mut Frame, app: &mut App, area: Rect) -> ShellAreas {
    app.last_body_area = area;
    let mut a = areas(area, app);
    // The mouse and the scroll clamp find the transcript and the input here.
    app.last_layout = crate::layout::BodyLayout {
        chat: Some(a.transcript),
        input: Some(a.input),
        ..Default::default()
    };
    a.used = draw_transcript(f, app, a.transcript, a.side.is_none());
    if let Some(rect) = a.approval {
        draw_approval(f, app, rect);
    }
    if let Some(rect) = a.side {
        crate::sidepanel::draw(f, app, rect);
    }
    draw_mentions(f, app, a.transcript);
    draw_input(f, app, a.input);
    draw_status(f, app, a.status, &HINTS);
    a
}

/// The transcript's lines at `width` columns, the welcome when empty.
pub fn transcript_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    if app.messages.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            " What should xencode do?",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::BOLD),
        )));
        let dim = Style::default().fg(app.theme.message_system);
        for row in crate::transcript::wrap_indented(
            "/ commands · @ files · ! shell · ctrl+p finds anything · ? keys",
            width as usize,
            1,
        ) {
            lines.push(Line::from(Span::styled(row, dim)));
        }
        let project = xencode_context_rs::default_root()
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let model = if app.config.default_model.is_empty() {
            "no model chosen yet — ctrl+l picks one".to_string()
        } else {
            app.config.default_model.clone()
        };
        for row in
            crate::transcript::wrap_indented(&format!("{model} · {project}"), width as usize, 1)
        {
            lines.push(Line::from(Span::styled(row, dim)));
        }
    } else {
        lines = crate::transcript::lines(
            &app.messages,
            &app.theme,
            &crate::transcript::RenderOpts {
                width,
                expand_tools: app.config.tool_output == "expand",
                expanded: app.tool_expanded.clone(),
            },
        );
    }
    // Messages typed during the turn, waiting for it to end (TD-3).
    for queued in &app.queued {
        let queued = crate::transcript::clean(queued);
        let first = queued.lines().next().unwrap_or_default();
        for row in crate::transcript::wrap_indented(&format!("queued: {first}"), width as usize, 2)
        {
            lines.push(Line::from(Span::styled(
                row,
                Style::default().fg(app.theme.message_system),
            )));
        }
    }
    if app.is_generating {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  {} working · esc stops", spinner::frame(app.spinner_tick)),
            Style::default().fg(app.theme.accent),
        )));
    }
    lines
}

/// The transcript's text area inside its rect: one column of margin each side.
fn text_rect(area: Rect) -> Rect {
    Rect {
        x: area.x + 1.min(area.width),
        width: area.width.saturating_sub(2),
        ..area
    }
}

/// Keep `app.chat_scroll` inside what the transcript at `area` can scroll.
pub fn clamp_scroll(app: &mut App, area: Rect) {
    let text = text_rect(area);
    let total = transcript_lines(app, text.width).len() as u16;
    app.chat_scroll = app.chat_scroll.min(total.saturating_sub(text.height));
}

/// Returns the rows from the top of `area` down to its last line of text.
fn draw_transcript(f: &mut Frame, app: &mut App, area: Rect, with_plan: bool) -> u16 {
    if area.height == 0 || area.width < 3 {
        return 0;
    }
    let top = area.y;
    // Without the side panel the plan sits over the transcript, as before.
    let area = match crate::ui::plan_block_height(app, area) {
        rows if with_plan && rows > 0 => {
            let split = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(rows), Constraint::Min(1)])
                .split(area);
            crate::ui::draw_plan_strip(f, app, split[0]);
            split[1]
        }
        _ => area,
    };
    let text = text_rect(area);
    let lines = transcript_lines(app, text.width);
    let total = lines.len().min(u16::MAX as usize) as u16;
    // Reading further up (`chat_scroll` above 0) while new lines arrive: the
    // view moves up with them, so the lines being read stay put (spec §4.3).
    if app.chat_scroll > 0 && total > app.transcript_rows {
        app.chat_scroll = app.chat_scroll.saturating_add(total - app.transcript_rows);
    }
    app.transcript_rows = total;
    let max_scroll = total.saturating_sub(text.height);
    app.chat_scroll = app.chat_scroll.min(max_scroll);
    // `chat_scroll` counts up from the bottom, so 0 follows the newest line.
    let scroll = max_scroll - app.chat_scroll;
    f.render_widget(Paragraph::new(lines).scroll((scroll, 0)), text);
    // A scrollbar in the right margin, only once there is something to
    // scroll to, and only where `show_scrollbars` and the width allow it.
    if app.config.show_scrollbars && area.width >= 24 && max_scroll > 0 {
        let bar = Rect {
            x: area.x + area.width - 1,
            width: 1,
            ..area
        };
        let mut state =
            ratatui::widgets::ScrollbarState::new(total as usize).position(scroll as usize);
        f.render_stateful_widget(crate::ui::scrollbar_bar(app), bar, &mut state);
    }
    (text.y - top) + total.min(text.height)
}

/// Columns of text inside the approval box drawn `width` wide: its border and
/// one column of padding on each side.
fn approval_text_width(width: u16) -> usize {
    width.saturating_sub(4).max(1) as usize
}

pub(crate) fn draw_approval(f: &mut Frame, app: &App, area: Rect) {
    let Some((title, body, keys)) = approval_lines(app, approval_text_width(area.width)) else {
        return;
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(crate::widgets::panel_border_set(app.config.rounded_borders))
        .border_style(Style::default().fg(app.theme.warning))
        .title(Span::styled(title, Style::default().fg(app.theme.warning)));
    // The keys are always the last row. A change longer than the rows above
    // them scrolls with ↑↓ (or j/k), and a row of its own says how much is
    // out of sight, so nothing is approved unseen without the box saying so.
    let mut rows = (area.height.saturating_sub(2) as usize).saturating_sub(1);
    if body.len() > rows {
        rows = rows.saturating_sub(1);
    }
    let hidden = body.len().saturating_sub(rows);
    let scroll = app.approval_scroll.min(hidden);
    let below = hidden - scroll;
    let mut lines: Vec<Line<'static>> = body.into_iter().skip(scroll).take(rows).collect();
    if hidden > 0 {
        let mut note = Vec::new();
        if scroll > 0 {
            note.push(format!("↑ {scroll} above"));
        }
        if below > 0 {
            note.push(format!("↓ {below} more"));
        }
        note.push("↑↓ scroll".to_string());
        lines.push(Line::from(Span::styled(
            note.join(" · "),
            Style::default().fg(app.theme.warning),
        )));
    }
    lines.push(keys);
    f.render_widget(
        Paragraph::new(lines).block(block.padding(ratatui::widgets::Padding::horizontal(1))),
        area,
    );
}

/// The files matching the `@word` being typed, in a box sitting on the input
/// (TD-3). Tab puts the first one in.
fn draw_mentions(f: &mut Frame, app: &App, transcript: Rect) {
    let matches = app.mention_matches();
    if matches.is_empty() || transcript.height < 4 || transcript.width < 20 {
        return;
    }
    let rows = (matches.len() as u16).min(transcript.height - 2);
    let width = transcript.width.min(64);
    let area = Rect {
        x: transcript.x,
        y: transcript.y + transcript.height - rows - 2,
        width,
        height: rows + 2,
    };
    let lines: Vec<Line<'static>> = matches
        .into_iter()
        .take(rows as usize)
        .enumerate()
        .map(|(i, path)| {
            let style = if i == 0 {
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(app.theme.fg)
            };
            Line::from(Span::styled(crate::transcript::clean(&path), style))
        })
        .collect();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(crate::widgets::panel_border_set(app.config.rounded_borders))
        .border_style(Style::default().fg(app.theme.border))
        .title(Span::styled(
            " files · tab inserts ",
            Style::default().fg(app.theme.message_system),
        ));
    f.render_widget(ratatui::widgets::Clear, area);
    f.render_widget(
        Paragraph::new(lines).block(block.padding(ratatui::widgets::Padding::horizontal(1))),
        area,
    );
}

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let editing = app.input_mode == InputMode::Editing;
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(crate::widgets::panel_border_set(app.config.rounded_borders))
        .border_style(Style::default().fg(if editing {
            app.theme.accent
        } else {
            app.theme.border
        }));
    let inner = block.inner(area);
    // One column of air between the frame and the text.
    let inner = Rect {
        x: inner.x + 1.min(inner.width),
        width: inner.width.saturating_sub(2),
        ..inner
    };
    f.render_widget(block, area);
    let empty = app.chat_input.lines().iter().all(|l| l.is_empty());
    if empty {
        let hint = if app.is_generating {
            "xencode is working · type to queue a message · esc stops"
        } else {
            "Ask xencode…   / commands   @ files   ! shell"
        };
        f.render_widget(
            Paragraph::new(Span::styled(
                hint,
                Style::default().fg(app.theme.message_system),
            )),
            inner,
        );
    } else {
        f.render_widget(&app.chat_input, inner);
    }
    if editing {
        let (row, col) = app.chat_input.cursor();
        let x = inner.x.saturating_add(col as u16);
        let y = inner.y.saturating_add(row as u16);
        if x < inner.x + inner.width && y < inner.y + inner.height {
            f.set_cursor_position((x, y));
        }
    }
}

/// Every item the status line could show, in the configured order.
fn all_status_items(app: &App) -> Vec<String> {
    let mut out = Vec::new();
    // Orchestrator mode changes what the agent may do, so it is always said,
    // first, whatever `status_line` lists; coding mode is the quiet default.
    if app.mode == crate::focus::Mode::Orchestrator {
        out.push("orchestrator".to_string());
    }
    for item in &app.config.status_line {
        match item.as_str() {
            "model" if !app.config.default_model.is_empty() => {
                out.push(app.config.default_model.clone())
            }
            "provider" => out.push(egress_label(app).to_string()),
            "approval" => out.push(approval_label(app).to_string()),
            "context" => {
                if let Some(pct) = context_percent(app) {
                    out.push(format!("{pct}% context"));
                }
            }
            "cost" => {
                if let Some(micros) = app.spend.as_ref().and_then(|s| s.micros) {
                    out.push(format!("${:.2}", micros as f64 / 1_000_000.0));
                }
            }
            "branch" if !app.git_branch.is_empty() => out.push(app.git_branch.clone()),
            _ => {}
        }
    }
    out
}

fn width_of(items: &[String]) -> usize {
    items.iter().map(|i| i.chars().count()).sum::<usize>() + items.len().saturating_sub(1) * 3
}

/// The status line's left items that fit in `width` columns beside the
/// shortest hint, dropped from the end.
pub fn status_items(app: &App, width: u16) -> Vec<String> {
    let mut items = all_status_items(app);
    let room = (width as usize).saturating_sub(HINTS[HINTS.len() - 1].len() + 3);
    while !items.is_empty() && width_of(&items) > room {
        items.pop();
    }
    if let Some(first) = items.first_mut() {
        if first.chars().count() > room {
            *first = first.chars().take(room).collect();
        }
    }
    items
}

/// The status line under a view (TD-4): the same items, the view's keys.
pub fn draw_view_status(f: &mut Frame, app: &App, area: Rect) {
    draw_status(f, app, area, &VIEW_HINTS);
}

fn draw_status(f: &mut Frame, app: &App, area: Rect, hints: &[&str]) {
    let width = area.width as usize;
    let items = status_items(app, area.width);
    let left = width_of(&items) + 1;
    let hint = hints
        .iter()
        .find(|h| left + h.chars().count() + 2 <= width)
        .copied()
        .unwrap_or("");
    let mut spans = vec![Span::raw(" ")];
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(
                " · ",
                Style::default().fg(app.theme.message_system),
            ));
        }
        let color = if i == 0 {
            app.theme.accent
        } else {
            app.theme.message_system
        };
        spans.push(Span::styled(item.clone(), Style::default().fg(color)));
    }
    if !hint.is_empty() {
        let gap = width.saturating_sub(left + hint.chars().count() + 1);
        spans.push(Span::raw(" ".repeat(gap)));
        spans.push(Span::styled(
            hint,
            Style::default().fg(app.theme.message_system),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_status_items_drop_from_the_end_until_they_fit() {
        let mut app = App::for_tests();
        app.config.default_model = "ollama:qwen2.5:7b".into();
        app.git_branch = "main".into();
        let wide = status_items(&app, 200);
        assert_eq!(wide, ["ollama:qwen2.5:7b", "local only", "ask", "main"]);
        let narrow = status_items(&app, 30);
        assert_eq!(narrow, ["ollama:qwen2.5:7b"]);
        assert!(status_items(&app, 4).len() <= 1);
    }

    #[test]
    fn orchestrator_mode_is_named_first_on_the_status_line() {
        let mut app = App::for_tests();
        assert!(!status_items(&app, 200).iter().any(|i| i == "orchestrator"));
        app.mode = crate::focus::Mode::Orchestrator;
        assert_eq!(
            status_items(&app, 200).first().map(String::as_str),
            Some("orchestrator")
        );
        // It is the last item a narrow terminal drops.
        assert_eq!(status_items(&app, 20), ["orchestrator"]);
    }

    #[test]
    fn the_title_counts_everything_waiting_on_the_person() {
        let mut app = App::for_tests();
        let tile = |id: &str, state: &str| crate::engine::term::TileInfo {
            id: id.into(),
            agent: "shell".into(),
            branch: String::new(),
            worktree: String::new(),
            state: state.into(),
            cols: 80,
            rows: 24,
            exit: None,
        };
        app.term_tiles = vec![
            tile("t1", "needs_you"),
            tile("t2", "working"),
            tile("t3", "needs_you"),
        ];
        assert_eq!(attention_title(&app), "xencode · 2 need you");
        let keys = waiting_keys(&app);
        assert!(keys.contains("tile t1") && keys.contains("tile t3") && !keys.contains("tile t2"));
    }

    #[test]
    fn the_title_calls_for_the_person_while_an_approval_waits() {
        let mut app = App::for_tests();
        assert!(attention_title(&app).starts_with("xencode · "));
        assert_ne!(attention_title(&app), "xencode · needs you");
        let (responder, _answer) = tokio::sync::oneshot::channel();
        app.approval_queue.push_back((
            crate::agent_tools::ApprovalRequest {
                tool: "run_shell".into(),
                class: crate::agent_tools::ToolClass::Shell,
                summary: "run_shell cargo test".into(),
                preview: "cargo test".into(),
                draft: Default::default(),
            },
            responder,
        ));
        assert_eq!(attention_title(&app), "xencode · needs you");
    }
}
