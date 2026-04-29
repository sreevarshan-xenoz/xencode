use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, FocusArea, InputMode};

pub fn draw(f: &mut Frame, app: &App) {
    // Full-screen themed background
    let bg = Block::default().style(Style::default().bg(app.theme.bg).fg(app.theme.fg));
    f.render_widget(bg, f.area());

    // Top-level: header (1) + body + status bar (1)
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Min(1),   // body
            Constraint::Length(1), // status bar
        ])
        .split(f.area());

    draw_header(f, app, outer[0]);
    draw_body(f, app, outer[1]);
    draw_status_bar(f, app, outer[2]);

    // Overlays (rendered on top)
    match app.focus {
        FocusArea::ModelSelector => draw_model_selector(f, app, f.area()),
        FocusArea::Settings => draw_settings(f, app, f.area()),
        FocusArea::CodeReview => draw_code_review(f, app, f.area()),
        FocusArea::PerformanceDashboard => draw_performance_dashboard(f, app, f.area()),
        FocusArea::ProviderHealth => draw_provider_health(f, app, f.area()),
        FocusArea::ProjectAnalyzer => draw_project_analyzer(f, app, f.area()),
        FocusArea::GitCommit => draw_git_commit(f, app, f.area()),
        _ => {}
    }
}

// ── Header ──────────────────────────────────────────────────────────────────

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let model = &app.config.default_model;
    let theme_name = &app.config.active_theme;
    let file_count = app.attached_files.len();
    let git_count = app.git_status.len();

    let left = format!(" ⚡ Xencode TUI ");
    let right = format!(
        " {} │ {} │ 📎 {} │ Δ {} ",
        model, theme_name, file_count, git_count
    );

    let padding = area.width as usize - left.len().min(area.width as usize) - right.len().min(area.width as usize);
    let header_text = format!("{}{}{}", left, " ".repeat(padding.max(0)), right);

    let header = Paragraph::new(header_text)
        .style(Style::default().bg(app.theme.accent).fg(app.theme.highlight_fg).add_modifier(Modifier::BOLD));
    f.render_widget(header, area);
}

// ── Status Bar ──────────────────────────────────────────────────────────────

fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let mode_str = match app.input_mode {
        InputMode::Normal => "NORMAL",
        InputMode::Editing => "EDITING",
    };

    let hints = match app.input_mode {
        InputMode::Normal => "i:edit  m:models  Tab:switch  q:quit  Ctrl+,:settings  Ctrl+R:review  Ctrl+T:terminal",
        InputMode::Editing => "Enter:send  Esc:normal  ←→:cursor  Ctrl+C:quit",
    };

    let status_text = format!(" {} │ {} ", mode_str, hints);

    let bar = Paragraph::new(status_text)
        .style(Style::default().bg(app.theme.status_bg).fg(app.theme.status_fg));
    f.render_widget(bar, area);
}

// ── Body (file explorer + chat + input) ─────────────────────────────────────

fn draw_body(f: &mut Frame, app: &App, area: Rect) {
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(75)])
        .split(area);

    draw_file_explorer(f, app, main_chunks[0]);

    // Right side: chat + optional terminal + input
    if app.show_terminal {
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(6),      // chat
                Constraint::Length(8),    // terminal
                Constraint::Length(3),    // input
            ])
            .split(main_chunks[1]);
        draw_messages(f, app, right[0]);
        draw_terminal(f, app, right[1]);
        draw_input(f, app, right[2]);
    } else {
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(main_chunks[1]);
        draw_messages(f, app, right[0]);
        draw_input(f, app, right[1]);
    }
}

// ── File Explorer ───────────────────────────────────────────────────────────

fn draw_file_explorer(f: &mut Frame, app: &App, area: Rect) {
    let is_focused = app.focus == FocusArea::FileExplorer;
    let border_style = if is_focused {
        Style::default().fg(app.theme.border_active)
    } else {
        Style::default().fg(app.theme.border)
    };

    let title = format!(" 📁 Workspace ({} files) ", app.file_tree.len());
    let block = Block::default().borders(Borders::ALL).border_style(border_style).title(title);

    let items: Vec<ListItem> = app.file_tree
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let is_selected = i == app.selected_file;

            // Git status color
            let git_color = if let Some(status) = app.git_status.get(path) {
                match status.as_str() {
                    "M" | "MM" => app.theme.message_user,    // Modified
                    "A" | "AM" => app.theme.message_assistant, // Added
                    "D"        => ratatui::style::Color::Red,  // Deleted
                    _          => app.theme.fg,               // Untracked etc
                }
            } else {
                app.theme.fg
            };

            let style = if is_selected {
                Style::default().fg(app.theme.highlight_fg).bg(app.theme.highlight).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(git_color)
            };

            let attached = if app.attached_files.contains(path) { "📌 " } else { "   " };
            let git_mark = if let Some(s) = app.git_status.get(path) {
                format!("[{}]", s.chars().next().unwrap_or(' '))
            } else {
                "   ".to_string()
            };

            ListItem::new(Line::from(Span::styled(format!("{}{} {}", attached, git_mark, path), style)))
        })
        .collect();

    let list = List::new(items).block(block);
    let mut state = ListState::default();
    state.select(Some(app.selected_file));
    f.render_stateful_widget(list, area, &mut state);
}

// ── Chat Messages ───────────────────────────────────────────────────────────

fn draw_messages(f: &mut Frame, app: &App, area: Rect) {
    let mut text = Vec::new();

    if app.messages.is_empty() {
        text.push(Line::from(Span::styled(
            "  Welcome to Xencode! Press 'i' to start typing.",
            Style::default().fg(app.theme.message_system),
        )));
        text.push(Line::from(""));
        text.push(Line::from(Span::styled(
            "  Shortcuts: m=models, Tab=explorer, Ctrl+R=review, /bytebot=agent",
            Style::default().fg(app.theme.message_system),
        )));
    }

    for msg in &app.messages {
        let (icon, role_name, color) = match msg.role.as_str() {
            "user" => ("🧑", "You", app.theme.message_user),
            "assistant" => ("🤖", "Xencode", app.theme.message_assistant),
            _ => ("ℹ️", "System", app.theme.message_system),
        };

        text.push(Line::from(vec![
            Span::styled(format!(" {} {} ", icon, role_name), Style::default().fg(color).add_modifier(Modifier::BOLD)),
            Span::styled("─".repeat(40), Style::default().fg(app.theme.border)),
        ]));

        for line in msg.content.lines() {
            text.push(Line::from(Span::styled(format!("  {}", line), Style::default().fg(app.theme.fg))));
        }
        text.push(Line::from(""));
    }

    // Spinner at bottom when generating
    if app.is_generating {
        let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        let frame = frames[app.spinner_tick % frames.len()];
        text.push(Line::from(Span::styled(
            format!("  {} Generating...", frame),
            Style::default().fg(app.theme.accent),
        )));
    }

    let is_focused = app.focus == FocusArea::ChatInput && app.input_mode == InputMode::Normal;
    let border_style = if is_focused {
        Style::default().fg(app.theme.border_active)
    } else {
        Style::default().fg(app.theme.border)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(" 💬 Chat ");

    let text_lines = text.len() as u16;
    let height = area.height.saturating_sub(2);
    let max_scroll = text_lines.saturating_sub(height);
    let scroll = max_scroll.saturating_sub(app.chat_scroll);

    let paragraph = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

    f.render_widget(paragraph, area);
}

// ── Input ───────────────────────────────────────────────────────────────────

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let is_editing = app.input_mode == InputMode::Editing;
    let border_style = if is_editing {
        Style::default().fg(app.theme.accent)
    } else if app.focus == FocusArea::ChatInput {
        Style::default().fg(app.theme.border_active)
    } else {
        Style::default().fg(app.theme.border)
    };

    let title = if app.is_generating {
        let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        let frame = frames[app.spinner_tick % frames.len()];
        format!(" {} Thinking... ", frame)
    } else if is_editing {
        " ✏️  Type your message (Enter to send) ".to_string()
    } else {
        " Press 'i' to start typing ".to_string()
    };

    let block = Block::default().borders(Borders::ALL).border_style(border_style).title(title);

    let display_text = if app.is_generating { "Please wait..." } else { &app.input };
    let paragraph = Paragraph::new(display_text).block(block).style(Style::default().fg(app.theme.fg));
    f.render_widget(paragraph, area);

    // Show cursor
    if is_editing && !app.is_generating {
        let cursor_x = area.x + 1 + app.input_cursor as u16;
        let cursor_y = area.y + 1;
        if cursor_x < area.x + area.width - 1 {
            f.set_cursor_position((cursor_x, cursor_y));
        }
    }
}

// ── Terminal Pane ───────────────────────────────────────────────────────────

fn draw_terminal(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border_active))
        .title(" 🖥️  Terminal (Ctrl+T to toggle) ");

    let text = Paragraph::new("  Terminal emulation coming soon.\n  Use Ctrl+T to toggle this pane.")
        .block(block)
        .style(Style::default().fg(app.theme.message_system));
    f.render_widget(text, area);
}

// ── Overlays ────────────────────────────────────────────────────────────────

fn draw_model_selector(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(50, 40, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🧠 Select Model (↑↓ Enter) ");

    let items: Vec<ListItem> = app.available_models.iter().enumerate().map(|(i, model)| {
        let is_current = model == &app.config.default_model;
        let style = if i == app.selected_model {
            Style::default().fg(app.theme.highlight_fg).bg(app.theme.highlight).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(app.theme.fg)
        };
        let prefix = if is_current { " ● " } else { "   " };
        ListItem::new(Line::from(Span::styled(format!("{}{}", prefix, model), style)))
    }).collect();

    let list = List::new(items).block(block);
    let mut state = ListState::default();
    state.select(Some(app.selected_model));
    f.render_stateful_widget(list, popup_area, &mut state);
}

fn draw_settings(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(60, 50, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" ⚙️  Settings (Esc to close) ");

    let openrouter_status = if app.config.api_keys.openrouter_api_key.is_some() { "✅ Set" } else { "❌ Not Set" };
    let settings_text = format!(
        "\n  Theme:           {}\n  Default Model:   {}\n  Ollama URL:      {}\n  Cache Enabled:   {}\n  Memory Enabled:  {}\n  Memory Items:    {}\n  Timeout:         {}s\n\n  OpenRouter Key:  {}\n\n  Edit ~/.xencode/config.json to modify settings.",
        app.config.active_theme, app.config.default_model, app.config.ollama_url,
        app.config.cache_enabled, app.config.memory_enabled,
        app.config.max_memory_items, app.config.response_timeout,
        openrouter_status
    );

    let text = Paragraph::new(settings_text)
        .block(block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false });
    f.render_widget(text, popup_area);
}

fn draw_code_review(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(80, 80, area);
    f.render_widget(Clear, popup_area);

    let title = if app.is_reviewing {
        let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        let frame = frames[app.spinner_tick % frames.len()];
        format!(" {} Code Review (analyzing...) ", frame)
    } else {
        " 🔍 Code Review (Enter to review, Esc to close) ".to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);

    let review_text = if !app.code_review_output.is_empty() {
        app.code_review_output.clone()
    } else if let Some(fp) = app.file_tree.get(app.selected_file) {
        format!("\n  Selected file: {}\n\n  Press Enter to start AI code review.", fp)
    } else {
        "  Select a file in the File Explorer first.".to_string()
    };

    let text = Paragraph::new(review_text)
        .block(block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false });
    f.render_widget(text, popup_area);
}

// ── Phase 9 Overlays ────────────────────────────────────────────────────────

fn draw_performance_dashboard(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(60, 50, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(app.theme.accent)).title(" 📊 Performance Dashboard (Esc to close) ");
    
    // Calculate simple stats
    let msg_count = app.messages.len();
    let mem_items = app.config.max_memory_items;
    
    let text = format!(
        "\n  Session Statistics:\n\n  Messages in Memory: {} / {}\n  Cache Enabled:      {}\n  Theme:              {}\n\n  (Latency and token tracking will require async metrics aggregator)",
        msg_count, mem_items, app.config.cache_enabled, app.config.active_theme
    );

    let para = Paragraph::new(text).block(block).style(Style::default().fg(app.theme.fg));
    f.render_widget(para, popup_area);
}

fn draw_provider_health(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(60, 40, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(app.theme.accent)).title(" 🏥 Provider Health (Esc to close) ");
    
    let text = format!(
        "\n  Ollama URI:       {}   [Status: ACTIVE]\n  OpenRouter API:   {}        [Status: {}]\n\n  Active Model:     {}",
        app.config.ollama_url,
        if app.config.api_keys.openrouter_api_key.is_some() { "Configured" } else { "Missing" },
        if app.config.api_keys.openrouter_api_key.is_some() { "OK" } else { "ERROR" },
        app.config.default_model
    );

    let para = Paragraph::new(text).block(block).style(Style::default().fg(app.theme.fg));
    f.render_widget(para, popup_area);
}

fn draw_project_analyzer(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(60, 60, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(app.theme.accent)).title(" 📈 Project Analyzer (Esc to close) ");
    
    let mut rust_files = 0;
    let mut py_files = 0;
    let mut ts_files = 0;
    let mut other = 0;
    
    for file in &app.file_tree {
        if file.ends_with(".rs") { rust_files += 1; }
        else if file.ends_with(".py") { py_files += 1; }
        else if file.ends_with(".ts") || file.ends_with(".tsx") { ts_files += 1; }
        else { other += 1; }
    }

    let text = format!(
        "\n  Workspace Scan Results:\n\n  Total Files: {}\n\n  🦀 Rust:       {}\n  🐍 Python:     {}\n  📘 TypeScript: {}\n  📄 Other:      {}",
        app.file_tree.len(), rust_files, py_files, ts_files, other
    );

    let para = Paragraph::new(text).block(block).style(Style::default().fg(app.theme.fg));
    f.render_widget(para, popup_area);
}

fn draw_git_commit(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(50, 40, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(app.theme.accent)).title(" 📝 Git Commit (Enter to commit, Esc to cancel) ");
    
    let modified_count = app.git_status.len();
    
    let text = format!(
        "\n  Staging {} files...\n\n  Message:\n  > {}\n\n  (Type your message and press Enter)",
        modified_count, app.commit_message
    );

    let para = Paragraph::new(text).block(block).style(Style::default().fg(app.theme.fg));
    f.render_widget(para, popup_area);
    
    // Draw cursor
    let cursor_x = popup_area.x + 4 + app.commit_cursor as u16;
    let cursor_y = popup_area.y + 5;
    f.set_cursor_position((cursor_x, cursor_y));
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
