use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, FocusArea, InputMode};

pub fn draw(f: &mut Frame, app: &App) {
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .margin(1)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(75)].as_ref())
        .split(f.area());

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)].as_ref())
        .split(main_chunks[1]);

    draw_file_explorer(f, app, main_chunks[0]);
    draw_messages(f, app, right_chunks[0]);
    draw_input(f, app, right_chunks[1]);

    if app.focus == FocusArea::ModelSelector {
        draw_model_selector(f, app, f.area());
    }
}

fn draw_model_selector(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(50, 50, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(" Select Model (Enter to confirm) ");

    let items: Vec<ListItem> = app.available_models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let style = if i == app.selected_model {
                Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            
            let prefix = if model == &app.config.default_model {
                "[*] "
            } else {
                "    "
            };

            ListItem::new(Line::from(Span::styled(format!("{}{}", prefix, model), style)))
        })
        .collect();

    let list = List::new(items).block(block);

    let mut state = ListState::default();
    state.select(Some(app.selected_model));

    f.render_widget(Clear, popup_area); // This clears the background
    f.render_stateful_widget(list, popup_area, &mut state);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ].as_ref())
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ].as_ref())
        .split(popup_layout[1])[1]
}

fn draw_file_explorer(f: &mut Frame, app: &App, area: Rect) {
    let border_color = if app.focus == FocusArea::FileExplorer {
        Color::Yellow
    } else {
        Color::Gray
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(" Workspace Explorer ");

    let items: Vec<ListItem> = app.file_tree
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let style = if i == app.selected_file {
                Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            
            let prefix = if app.attached_files.contains(path) {
                "[x] "
            } else {
                "    "
            };

            ListItem::new(Line::from(Span::styled(format!("{}{}", prefix, path), style)))
        })
        .collect();

    let list = List::new(items).block(block);

    // Manual scrolling for now using list state
    let mut state = ListState::default();
    state.select(Some(app.selected_file));

    f.render_stateful_widget(list, area, &mut state);
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
        InputMode::Normal => if app.focus == FocusArea::ChatInput { Color::White } else { Color::Gray },
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
