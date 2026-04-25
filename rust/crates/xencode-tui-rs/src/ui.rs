use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, InputMode};

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Min(1), Constraint::Length(3)].as_ref())
        .split(f.area());

    draw_messages(f, app, chunks[0]);
    draw_input(f, app, chunks[1]);
}

fn draw_messages(f: &mut Frame, app: &App, area: Rect) {
    let mut text = Vec::new();

    for msg in &app.messages {
        let (role_color, role_name) = match msg.role.as_str() {
            "user" => (Color::Cyan, "You"),
            "assistant" => (Color::Green, "Xencode"),
            _ => (Color::Gray, "System"),
        };

        text.push(Line::from(vec![Span::styled(
            format!("{}: ", role_name),
            Style::default().fg(role_color).add_modifier(Modifier::BOLD),
        )]));

        // Basic line splitting for content
        for line in msg.content.lines() {
            text.push(Line::from(Span::raw(line)));
        }
        text.push(Line::from(Span::raw("")));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Chat History ");

    // We can calculate offset to keep the latest messages in view
    let text_lines = text.len() as u16;
    let height = area.height.saturating_sub(2);
    let scroll = text_lines.saturating_sub(height);

    let paragraph = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

    f.render_widget(paragraph, area);
}

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let mode_color = match app.input_mode {
        InputMode::Normal => Color::Gray,
        InputMode::Editing => Color::Yellow,
    };

    let title = match app.input_mode {
        InputMode::Normal => " Input (Press 'i' to edit, 'q' to quit) ",
        InputMode::Editing => {
            if app.is_generating {
                " Generating... "
            } else {
                " Input (Press 'Esc' to exit edit mode, 'Enter' to submit) "
            }
        }
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(mode_color))
        .title(title);

    let content = if app.is_generating {
        "Please wait..."
    } else {
        &app.input
    };

    let paragraph = Paragraph::new(content).block(block);
    f.render_widget(paragraph, area);

    // Render cursor
    if app.input_mode == InputMode::Editing && !app.is_generating {
        // Simple cursor positioning (doesn't handle line wraps yet)
        f.set_cursor_position((
            area.x + 1 + app.input.len() as u16,
            area.y + 1,
        ));
    }
}
