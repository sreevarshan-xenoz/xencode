use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, FocusArea, InputMode};

pub fn draw(f: &mut Frame, app: &App) {
    let block = Block::default()
        .style(Style::default().bg(app.theme.bg).fg(app.theme.fg));
    f.render_widget(block, f.area());

    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .margin(1)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(75)].as_ref())
        .split(f.area());

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)].as_ref())
        .split(main_chunks[1]);

    let mut message_area = right_chunks[0];
    draw_file_explorer(f, app, main_chunks[0]);
    if app.focus == FocusArea::Terminal {
        let vert_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
            .split(right_chunks[0]);
        message_area = vert_chunks[0];
        draw_terminal(f, app, vert_chunks[1]);
    }

    draw_messages(f, app, message_area);
    draw_input(f, app, right_chunks[1]);

    if app.focus == FocusArea::ModelSelector {
        draw_model_selector(f, app, f.area());
    } else if app.focus == FocusArea::Settings {
        draw_settings(f, app, f.area());
    } else if app.focus == FocusArea::CodeReview {
        draw_code_review(f, app, f.area());
    }
}

fn draw_terminal(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border_active))
        .title(" Terminal (Coming Soon) ");
    
    let text = Paragraph::new("This feature requires a fully interactive PTY.\nFor now, use 'Ctrl+T' to toggle this pane.")
        .block(block)
        .style(Style::default().fg(app.theme.fg));
    
    f.render_widget(text, area);
}

fn draw_settings(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(60, 60, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border_active))
        .title(" Settings (Ctrl+, to close) ");

    let settings_text = format!(
        "Active Theme: {}\nDefault Model: {}\nOllama URL: {}\nOpenRouter API Key: {}",
        app.config.active_theme,
        app.config.default_model,
        app.config.ollama_url,
        if app.config.api_keys.openrouter_api_key.is_some() { "Set" } else { "Not Set" }
    );

    let text = Paragraph::new(settings_text)
        .block(block)
        .style(Style::default().fg(app.theme.fg));

    f.render_widget(Clear, popup_area);
    f.render_widget(text, popup_area);
}

fn draw_code_review(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(80, 80, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border_active))
        .title(" Code Review (Ctrl+R to close) ");

    let review_text = if let Some(file_path) = app.file_tree.get(app.selected_file) {
        format!("Reviewing: {}\n\nPress 'Enter' to initiate an AI review of this file.", file_path)
    } else {
        "Select a file in the File Explorer first.".to_string()
    };

    let text = Paragraph::new(review_text)
        .block(block)
        .style(Style::default().fg(app.theme.fg));

    f.render_widget(Clear, popup_area);
    f.render_widget(text, popup_area);
}

fn draw_model_selector(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(50, 50, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border_active))
        .title(" Select Model (Enter to confirm) ");

    let items: Vec<ListItem> = app.available_models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let style = if i == app.selected_model {
                Style::default().fg(app.theme.highlight_fg).bg(app.theme.highlight).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(app.theme.fg)
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
    let border_style = if app.focus == FocusArea::FileExplorer {
        Style::default().fg(app.theme.border_active)
    } else {
        Style::default().fg(app.theme.border)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" Workspace Explorer ");

    let items: Vec<ListItem> = app.file_tree
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let style = if i == app.selected_file {
                Style::default().fg(app.theme.highlight_fg).bg(app.theme.highlight).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(app.theme.fg)
            };
            
            let attached_prefix = if app.attached_files.contains(path) {
                "[x] "
            } else {
                "    "
            };

            let git_indicator = if let Some(status) = app.git_status.get(path) {
                format!("[{}] ", status.chars().next().unwrap_or('?'))
            } else {
                "    ".to_string()
            };

            let display_text = format!("{}{}{}", attached_prefix, git_indicator, path);

            ListItem::new(Line::from(Span::styled(display_text, style)))
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
        let (role_name, style) = match msg.role.as_str() {
            "user" => ("You", Style::default().fg(app.theme.message_user)),
            "assistant" => ("Xencode", Style::default().fg(app.theme.message_assistant)),
            _ => ("System", Style::default().fg(app.theme.message_system)),
        };

        text.push(Line::from(vec![Span::styled(
            format!("{}: ", role_name),
            style.add_modifier(Modifier::BOLD),
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
    let border_style = if app.focus == FocusArea::ChatInput {
        Style::default().fg(app.theme.border_active)
    } else {
        Style::default().fg(app.theme.border)
    };

    let title = if app.is_generating {
        " Thinking... "
    } else {
        match app.input_mode {
            InputMode::Normal => " Input (Press 'i' to edit, 'q' to quit) ",
            InputMode::Editing => " Editing (Press 'Enter' to send, 'Esc' to exit) ",
        }
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
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
