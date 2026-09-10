use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use xencode_models_rs::current_timestamp;

use crate::app::{App, FocusArea, InputMode, FEATURE_LIST};

pub fn draw(f: &mut Frame, app: &App) {
    // Full-screen themed background
    let bg = Block::default().style(Style::default().bg(app.theme.bg).fg(app.theme.fg));
    f.render_widget(bg, f.area());

    // Top-level: header (1) + body + status bar (1)
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Min(1),    // body
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
        FocusArea::FeatureNavigator => draw_feature_navigator(f, app, f.area()),
        FocusArea::ByteBotPanel => draw_bytebot_panel(f, app, f.area()),
        FocusArea::CollaborationHub => draw_collaboration_hub(f, app, f.area()),
        FocusArea::VoiceInterface => draw_voice_interface(f, app, f.area()),
        FocusArea::TerminalAssistant => draw_terminal_assistant(f, app, f.area()),
        FocusArea::SecurityAuditor => draw_security_auditor(f, app, f.area()),
        FocusArea::PerformanceProfiler => draw_performance_profiler(f, app, f.area()),
        FocusArea::CustomModels => draw_custom_models(f, app, f.area()),
        FocusArea::LearningMode => draw_learning_mode(f, app, f.area()),
        FocusArea::MultiLanguage => draw_multi_language(f, app, f.area()),
        _ => {}
    }

    // Project context overlay (driven from chat via /init), always on top.
    if app.init_visible {
        draw_project_init_panel(f, app, f.area());
    }
}

// ── Header ──────────────────────────────────────────────────────────────────

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let model = &app.config.default_model;
    let theme_name = &app.config.active_theme;
    let file_count = app.attached_files.len();
    let git_count = app.git_status.len();

    let left = " ⚡ Xencode TUI ".to_string();
    let right = format!(
        " {} │ {} │ 📎 {} │ Δ {} {}",
        model,
        theme_name,
        file_count,
        git_count,
        if app.init_running { " ⏳init " } else { "" }
    );

    // Display width, not byte length: the header holds multi-byte glyphs.
    // Saturating so a narrow terminal cannot underflow the padding.
    let padding = (area.width as usize)
        .saturating_sub(Line::raw(&left).width())
        .saturating_sub(Line::raw(&right).width());
    let header_text = format!("{}{}{}", left, " ".repeat(padding), right);

    let header = Paragraph::new(header_text).style(
        Style::default()
            .bg(app.theme.accent)
            .fg(app.theme.highlight_fg)
            .add_modifier(Modifier::BOLD),
    );
    f.render_widget(header, area);
}

// ── Status Bar ──────────────────────────────────────────────────────────────

fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let mode_str = match app.input_mode {
        InputMode::Normal => "NORMAL",
        InputMode::Editing => "EDITING",
    };

    // Git branch
    let branch_str = format!(" \u{1F9F0} {}", app.git_branch);

    // Provider health indicator
    let ollama_ok = app
        .ollama_health_entries
        .get("ollama")
        .map(|(s, _, _)| s == "healthy")
        .unwrap_or(false);
    let health_icon = if ollama_ok { "\u{2705}" } else { "\u{2753}" };

    // Uptime
    let uptime_secs = (xencode_models_rs::current_timestamp() - app.session_start_time).max(0.0);
    let uptime_mins = (uptime_secs / 60.0) as u64;
    let uptime_secs_rem = (uptime_secs % 60.0) as u64;
    let uptime_str = if uptime_mins > 0 {
        format!("{}m {}s", uptime_mins, uptime_secs_rem)
    } else {
        format!("{}s", uptime_secs_rem)
    };

    // File count
    let files_str = format!("\u{1F4C4} {}", app.file_tree.len());

    // Build status text chunks
    let left_parts = match app.input_mode {
        InputMode::Normal => format!(
            " {}  {} | {}  {}  {}  \u{394} {}  | ",
            mode_str,
            branch_str,
            health_icon,
            uptime_str,
            files_str,
            app.git_status.len()
        ),
        InputMode::Editing => format!(" {}  | ", mode_str),
    };

    // Context-sensitive hints based on current focus
    let hints = if app.input_mode == InputMode::Editing {
        "Enter:send  Esc:normal  \u{2190}\u{2192}:cursor"
    } else {
        match app.focus {
            FocusArea::Settings => "\u{2191}\u{2193}:nav  \u{2190}\u{2192}:change  Enter:save  Esc:close",
            FocusArea::FileExplorer => "\u{2191}\u{2193}:select  Enter:open  Space:attach  m:models",
            FocusArea::CodeEditor => "e:edit  \u{2191}\u{2193}:scroll  Ctrl+S:save  Tab:next",
            FocusArea::ModelSelector => "\u{2191}\u{2193}:select  Enter:confirm  Esc:close",
            FocusArea::CodeReview => "Enter:review  Esc:close",
            FocusArea::FeatureNavigator => "\u{2191}\u{2193}:nav  Enter:open  Esc:close",
            FocusArea::ByteBotPanel => "Enter:run  Esc:close  Type command above",
            FocusArea::ProviderHealth => "h:refresh  Esc:close",
            FocusArea::PerformanceDashboard | FocusArea::ProjectAnalyzer |
            FocusArea::GitCommit | FocusArea::CollaborationHub |
            FocusArea::VoiceInterface | FocusArea::TerminalAssistant |
            FocusArea::SecurityAuditor | FocusArea::PerformanceProfiler |
            FocusArea::CustomModels | FocusArea::LearningMode |
            FocusArea::MultiLanguage => "Enter:start  Esc:close",
            _ => "i:edit  m:models  s:settings  Tab:switch  Ctrl+R:review  Ctrl+T:terminal  Ctrl+B:bytebot  Ctrl+D:dashboard  Ctrl+P:analyzer",
        }
    };

    let status_text = format!("{}{}", left_parts, hints);

    let bar = Paragraph::new(status_text).style(
        Style::default()
            .bg(app.theme.status_bg)
            .fg(app.theme.status_fg),
    );
    f.render_widget(bar, area);
}

// ── Body (file explorer + chat + input) ─────────────────────────────────────

fn draw_body(f: &mut Frame, app: &App, area: Rect) {
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20), // File Explorer
            Constraint::Percentage(50), // Code Editor
            Constraint::Percentage(30), // Chat
        ])
        .split(area);

    draw_file_explorer(f, app, main_chunks[0]);
    draw_code_editor(f, app, main_chunks[1]);

    // Right side: chat + optional terminal + input
    if app.show_terminal {
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(6),    // chat
                Constraint::Length(8), // terminal
                Constraint::Length(3), // input
            ])
            .split(main_chunks[2]);
        draw_messages(f, app, right[0]);
        draw_terminal(f, app, right[1]);
        draw_input(f, app, right[2]);
    } else {
        let right = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(main_chunks[2]);
        draw_messages(f, app, right[0]);
        draw_input(f, app, right[1]);
    }
}

fn draw_code_editor(f: &mut Frame, app: &App, area: Rect) {
    let is_focused = app.focus == FocusArea::CodeEditor;
    let is_editing = is_focused && app.input_mode == InputMode::Editing;

    let border_style = if is_editing {
        Style::default().fg(app.theme.accent)
    } else if is_focused {
        Style::default().fg(app.theme.border_active)
    } else {
        Style::default().fg(app.theme.border)
    };

    let dirty_mark = if app.editor_dirty { " [modified]" } else { "" };
    let mode_mark = if is_editing { " EDITING" } else { "" };

    let title = if let Some(ref fp) = app.opened_file {
        format!(" 📝 {}{}{} ", fp, dirty_mark, mode_mark)
    } else {
        " 📝 Code Editor (Select a file & press Enter) ".to_string()
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(title);

    if app.opened_file.is_some() {
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(&app.editor, inner);
    } else {
        let text = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No file opened.",
                Style::default().fg(app.theme.message_system),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  Select a file in the Explorer and press Enter.",
                Style::default().fg(app.theme.message_system),
            )),
            Line::from(Span::styled(
                "  Press 'e' to enter edit mode, Ctrl+S to save.",
                Style::default().fg(app.theme.message_system),
            )),
        ];
        let paragraph = Paragraph::new(text).block(block);
        f.render_widget(paragraph, area);
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
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(title);

    let items: Vec<ListItem> = app
        .file_tree
        .iter()
        .enumerate()
        .map(|(i, path)| {
            let is_selected = i == app.selected_file;

            // Git status color
            let git_color = if let Some(status) = app.git_status.get(path) {
                match status.as_str() {
                    "M" | "MM" => app.theme.message_user,      // Modified
                    "A" | "AM" => app.theme.message_assistant, // Added
                    "D" => ratatui::style::Color::Red,         // Deleted
                    _ => app.theme.fg,                         // Untracked etc
                }
            } else {
                app.theme.fg
            };

            let style = if is_selected {
                Style::default()
                    .fg(app.theme.highlight_fg)
                    .bg(app.theme.highlight)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(git_color)
            };

            let attached = if app.attached_files.contains(path) {
                "📌 "
            } else {
                "   "
            };
            let git_mark = if let Some(s) = app.git_status.get(path) {
                format!("[{}]", s.chars().next().unwrap_or(' '))
            } else {
                "   ".to_string()
            };

            ListItem::new(Line::from(Span::styled(
                format!("{}{} {}", attached, git_mark, path),
                style,
            )))
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
            Span::styled(
                format!(" {} {} ", icon, role_name),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled("─".repeat(40), Style::default().fg(app.theme.border)),
        ]));

        for line in msg.content.lines() {
            text.push(Line::from(Span::styled(
                format!("  {}", line),
                Style::default().fg(app.theme.fg),
            )));
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

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(title);

    let display_text = if app.is_generating {
        "Please wait..."
    } else {
        &app.input
    };
    let paragraph = Paragraph::new(display_text)
        .block(block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(paragraph, area);

    // Show cursor
    if is_editing && !app.is_generating {
        let cursor_x = area.x + 1 + app.input_cursor as u16;
        let cursor_y = area.y + 1;
        if cursor_x < area.x + area.width.saturating_sub(1) {
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

    let text =
        Paragraph::new("  Terminal emulation coming soon.\n  Use Ctrl+T to toggle this pane.")
            .block(block)
            .style(Style::default().fg(app.theme.message_system));
    f.render_widget(text, area);
}

// ── Overlays ────────────────────────────────────────────────────────────────

fn draw_model_selector(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(55, 50, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🧠 Select Model (↑↓ Enter · 'r' Refresh · 'l' Load · 'u' Unload · Esc Close) ");

    let items: Vec<ListItem> = if app.available_models.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "   ⚠️  No models detected. Ensure Ollama is running, or press 'r' to refresh.",
            Style::default().fg(ratatui::style::Color::Yellow),
        )))]
    } else {
        app.available_models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let is_current = model == &app.config.default_model;
                let style = if i == app.selected_model {
                    Style::default()
                        .fg(app.theme.highlight_fg)
                        .bg(app.theme.highlight)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(app.theme.fg)
                };
                let prefix = if is_current { " ● " } else { "   " };
                let (badge, badge_color) =
                    if model.starts_with("llamacpp:") || model.starts_with("llama.cpp:") {
                        (" [llamacpp]", ratatui::style::Color::Yellow)
                    } else if model.contains('/') || model.starts_with("qwen-") {
                        (" [cloud]", ratatui::style::Color::Magenta)
                    } else {
                        (" [ollama]", ratatui::style::Color::Cyan)
                    };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{}{}", prefix, model), style),
                    Span::styled(badge, Style::default().fg(badge_color)),
                ]))
            })
            .collect()
    };

    let list = List::new(items).block(block);
    let mut state = ListState::default();
    if !app.available_models.is_empty() {
        state.select(Some(app.selected_model));
    }
    f.render_stateful_widget(list, popup_area, &mut state);

    // Footer status: show llama.cpp model path / load-unload feedback
    let footer_y = popup_area.bottom();
    if footer_y < area.height {
        let status = if !app.llamacpp_action_msg.is_empty() {
            app.llamacpp_action_msg.clone()
        } else if app.config.llama_cpp_model_path.is_empty() {
            "Hint: set a GGUF path in Settings → Llama.cpp Model for 'l'/'u'".to_string()
        } else {
            format!("Llama.cpp GGUF: {}", app.config.llama_cpp_model_path)
        };
        let width = popup_area.width.saturating_sub(2).max(1) as usize;
        let status = if status.chars().count() > width {
            let s: String = status.chars().take(width).collect();
            s + "..."
        } else {
            status
        };
        f.render_widget(
            Paragraph::new(Span::styled(
                status,
                Style::default().fg(app.theme.message_system),
            )),
            Rect::new(popup_area.x, footer_y, popup_area.width, 1),
        );
    }
}

fn draw_settings(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(65, 80, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" ⚙️  Settings (↑↓ select, ←→ change, Enter edit/save, Esc close) ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(24),
            Constraint::Length(7),
            Constraint::Length(4),
        ])
        .split(inner);

    // ── Settings List ──────────────────────────────────────────────────────
    let themes = [
        "ocean",
        "midnight",
        "forest",
        "terminal",
        "dracula",
        "solarized",
        "nord",
    ];
    let theme_pos = themes
        .iter()
        .position(|t| *t == app.config.active_theme)
        .unwrap_or(0);
    let theme_indicators: String = (0..4)
        .map(|i| if i == theme_pos { "●" } else { "○" })
        .collect::<Vec<_>>()
        .join(" ");

    let cache_str = if app.config.cache_enabled {
        "✅ Enabled".to_string()
    } else {
        "❌ Disabled".to_string()
    };
    let memory_str = if app.config.memory_enabled {
        "✅ Enabled".to_string()
    } else {
        "❌ Disabled".to_string()
    };
    let cache_size_str = format!("{} entries", app.config.max_cache_size);
    let memory_items_str = format!("{} entries", app.config.max_memory_items);
    let timeout_str = format!("{}s", app.config.response_timeout);
    let theme_str = format!("{}  [{}]", app.config.active_theme, theme_indicators);

    let ollama_url_str = if app.settings_url_editing && app.settings_cursor == 6 {
        format!(
            "{}| (type to edit, Enter to confirm)",
            &app.settings_url_buffer[..app.settings_url_cursor]
        )
    } else {
        format!("{}  (Enter to edit)", app.config.ollama_url)
    };

    let llamacpp_url_str = if app.settings_url_editing && app.settings_cursor == 7 {
        format!(
            "{}| (type to edit, Enter to confirm)",
            &app.settings_url_buffer[..app.settings_url_cursor]
        )
    } else {
        format!("{}  (Enter to edit)", app.config.llama_cpp_url)
    };

    let llamacpp_model_path_str = if app.settings_url_editing && app.settings_cursor == 8 {
        format!(
            "{}| (type to edit, Enter to confirm)",
            &app.settings_url_buffer[..app.settings_url_cursor]
        )
    } else if app.config.llama_cpp_model_path.is_empty() {
        "⚠️  Not set (Enter to edit)".to_string()
    } else {
        format!("{}  (Enter to edit)", app.config.llama_cpp_model_path)
    };

    let fmt_opt = |v: &Option<f64>| match v {
        Some(x) => format!("{x}"),
        None => "default".to_string(),
    };
    let fmt_opt_i32 = |v: &Option<i32>| match v {
        Some(x) => format!("{x}"),
        None => "default".to_string(),
    };
    let fmt_opt_u32 = |v: &Option<u32>| match v {
        Some(x) => format!("{x}"),
        None => "default".to_string(),
    };

    let llama_temp_str = if app.settings_url_editing && app.settings_cursor == 9 {
        format!(
            "{}| (Enter to confirm)",
            &app.settings_url_buffer[..app.settings_url_cursor]
        )
    } else {
        fmt_opt(&app.config.llama_cpp_temperature)
    };
    let llama_topk_str = if app.settings_url_editing && app.settings_cursor == 10 {
        format!(
            "{}| (Enter to confirm)",
            &app.settings_url_buffer[..app.settings_url_cursor]
        )
    } else {
        fmt_opt_i32(&app.config.llama_cpp_top_k)
    };
    let llama_minp_str = if app.settings_url_editing && app.settings_cursor == 11 {
        format!(
            "{}| (Enter to confirm)",
            &app.settings_url_buffer[..app.settings_url_cursor]
        )
    } else {
        fmt_opt(&app.config.llama_cpp_min_p)
    };
    let llama_maxtokens_str = if app.settings_url_editing && app.settings_cursor == 12 {
        format!(
            "{}| (Enter to confirm)",
            &app.settings_url_buffer[..app.settings_url_cursor]
        )
    } else {
        fmt_opt_u32(&app.config.llama_cpp_max_tokens)
    };

    let reset_label = if app.settings_reset_active {
        "✅ Reset to defaults!"
    } else {
        "⚠️  Reset to defaults (Enter to confirm)"
    };

    let settings_values: [(&str, &str); 14] = [
        ("Theme            ", &theme_str),
        ("Cache Enabled    ", &cache_str),
        ("Memory Enabled   ", &memory_str),
        ("Max Cache Size   ", &cache_size_str),
        ("Memory Items     ", &memory_items_str),
        ("Response Timeout ", &timeout_str),
        ("Ollama URL       ", &ollama_url_str),
        ("Llama.cpp URL    ", &llamacpp_url_str),
        ("Llama.cpp Model  ", &llamacpp_model_path_str),
        ("Llama Temp       ", &llama_temp_str),
        ("Llama Top-K      ", &llama_topk_str),
        ("Llama Min-P      ", &llama_minp_str),
        ("Llama Max Tokens ", &llama_maxtokens_str),
        ("Factory Reset    ", reset_label),
    ];

    let sections: [(usize, usize, &str); 6] = [
        (0, 1, "  Display"),
        (1, 3, "  Performance"),
        (3, 6, "  Limits"),
        (6, 8, "  Connection"),
        (8, 13, "  llama.cpp"),
        (13, 14, "  Actions"),
    ];

    let mut settings_lines: Vec<Line> = Vec::new();
    for (s_idx, &(start, end, section_name)) in sections.iter().enumerate() {
        settings_lines.push(Line::from(Span::styled(
            section_name,
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )));

        for (idx, &(label, value)) in settings_values.iter().enumerate().take(end).skip(start) {
            let is_selected = app.settings_cursor == idx;
            let style = if is_selected {
                Style::default()
                    .fg(app.theme.highlight_fg)
                    .bg(app.theme.highlight)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(app.theme.fg)
            };
            let pointer = if is_selected { " ▶ " } else { "   " };
            let is_reset = idx == 13;
            let is_url_item = idx == 6 || idx == 7 || idx == 8;
            let is_num_item = idx == 9 || idx == 10 || idx == 11 || idx == 12;
            let value_color = if is_reset && is_selected {
                ratatui::style::Color::Red
            } else if is_reset {
                app.theme.message_system
            } else if (is_url_item || is_num_item) && app.settings_url_editing && is_selected {
                ratatui::style::Color::Yellow
            } else {
                app.theme.accent
            };
            settings_lines.push(Line::from(vec![
                Span::styled(format!("{}{}", pointer, label), style),
                Span::styled(value, Style::default().fg(value_color)),
            ]));
        }
        if s_idx + 1 < sections.len() {
            settings_lines.push(Line::from(""));
        }
    }

    let settings_para = Paragraph::new(settings_lines).style(Style::default().fg(app.theme.fg));
    f.render_widget(settings_para, chunks[0]);

    // ── Provider Status ────────────────────────────────────────────────────
    let mut provider_lines: Vec<Line> = Vec::new();
    provider_lines.push(Line::from(Span::styled(
        "  API Providers",
        Style::default()
            .fg(app.theme.fg)
            .add_modifier(Modifier::UNDERLINED),
    )));
    provider_lines.push(Line::from(""));

    let checks: [(&str, bool, &str); 5] = [
        ("Ollama   ", true, &app.config.ollama_url),
        ("Llama.cpp ", true, &app.config.llama_cpp_url),
        (
            "OpenRouter",
            app.config.api_keys.openrouter_api_key.is_some(),
            if app.config.api_keys.openrouter_api_key.is_some() {
                "✅ Key set"
            } else {
                "❌ No key"
            },
        ),
        (
            "Gemini   ",
            app.config.api_keys.google_gemini_api_key.is_some(),
            if app.config.api_keys.google_gemini_api_key.is_some() {
                "✅ Key set"
            } else {
                "❌ No key"
            },
        ),
        (
            "Qwen     ",
            app.config.api_keys.qwen_api_key.is_some(),
            if app.config.api_keys.qwen_api_key.is_some() {
                "✅ Key set"
            } else {
                "❌ No key"
            },
        ),
    ];

    for &(name, configured, detail) in &checks {
        let icon = if configured { "✅" } else { "❌" };
        let color = if configured {
            ratatui::style::Color::Green
        } else {
            ratatui::style::Color::Red
        };
        provider_lines.push(Line::from(vec![
            Span::styled(format!("  {}  {}", icon, name), Style::default().fg(color)),
            Span::styled(detail, Style::default().fg(app.theme.message_system)),
        ]));
    }
    provider_lines.push(Line::from(""));
    provider_lines.push(Line::from(Span::styled(
        format!(
            "  Active Model: {}  (press 'm' to change)",
            app.config.default_model
        ),
        Style::default().fg(app.theme.message_system),
    )));

    let provider_para = Paragraph::new(provider_lines).style(Style::default().fg(app.theme.fg));
    f.render_widget(provider_para, chunks[1]);

    // ── Keyboard Shortcuts ─────────────────────────────────────────────────
    let shortcut_lines = vec![
        Line::from(Span::styled(
            "  Keyboard Shortcuts",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  ↑↓",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Navigate  ", Style::default().fg(app.theme.fg)),
            Span::styled(
                "←→",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Change value  ", Style::default().fg(app.theme.fg)),
        ]),
        Line::from(vec![
            Span::styled(
                "  Enter",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Save & close  ", Style::default().fg(app.theme.fg)),
            Span::styled(
                "Esc",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("    Close", Style::default().fg(app.theme.fg)),
        ]),
        Line::from(vec![
            Span::styled(
                "  s",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("       Open/close  ", Style::default().fg(app.theme.fg)),
            Span::styled(
                "m",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("       Change model", Style::default().fg(app.theme.fg)),
        ]),
    ];

    let shortcuts_para = Paragraph::new(shortcut_lines).style(Style::default().fg(app.theme.fg));
    f.render_widget(shortcuts_para, chunks[2]);
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
        format!(
            "\n  Selected file: {}\n\n  Press Enter to start AI code review.",
            fp
        )
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
    let popup_area = centered_rect(70, 60, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 📊 Performance Dashboard (Esc to close) ");

    // Compute workspace breakdown
    let mut rust_files = 0u64;
    let mut py_files = 0u64;
    let mut ts_files = 0u64;
    let mut other_files = 0u64;
    for file in &app.file_tree {
        if file.ends_with(".rs") {
            rust_files += 1;
        } else if file.ends_with(".py") {
            py_files += 1;
        } else if file.ends_with(".ts") || file.ends_with(".tsx") {
            ts_files += 1;
        } else {
            other_files += 1;
        }
    }
    let total_files = app.file_tree.len() as u64;

    // Session uptime
    let uptime_secs = (current_timestamp() - app.session_start_time).max(0.0);
    let hours = (uptime_secs / 3600.0) as u64;
    let mins = ((uptime_secs % 3600.0) / 60.0) as u64;
    let secs = (uptime_secs % 60.0) as u64;

    let uptime_str = if hours > 0 {
        format!("{}h {}m {}s", hours, mins, secs)
    } else if mins > 0 {
        format!("{}m {}s", mins, secs)
    } else {
        format!("{}s", secs)
    };

    // Bar chart helper (simple ASCII)
    fn bar(value: u64, max: u64, width: usize) -> String {
        if max == 0 {
            return " ".repeat(width);
        }
        let filled = ((value as f64 / max as f64) * width as f64).round() as usize;
        let filled = filled.min(width);
        let empty = width.saturating_sub(filled);
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }

    let msg_count = app.messages.len() as u64;
    let bar_width = 20usize;
    let max_bar = total_files.max(msg_count).max(1);

    let git_count = app.git_status.len() as u64;
    let avg_latency = if app.average_latency > 0.0 {
        app.average_latency
    } else {
        0.0
    };

    // Determine health status emoji for current model
    let model_health_icon = if let Some((status, _, _)) = app.ollama_health_entries.get("ollama") {
        match status.as_str() {
            "healthy" => "\u{2705}",
            _ => "\u{2753}",
        }
    } else {
        "\u{2753}"
    };

    let lines = vec![
        Line::from(Span::styled(
            format!(
                " Session Uptime:          {}  |  Messages: {}  |  LLM Calls: {}",
                uptime_str, msg_count, app.total_llm_calls
            ),
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            " Workspace Breakdown",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from(format!(
            "   Total Files: {}  |  Git Changes: {}",
            total_files, git_count
        )),
        Line::from(""),
        Line::from(format!(
            "   {} Rust:        {}  {}",
            "\u{1F980}",
            rust_files,
            bar(rust_files, max_bar, bar_width)
        )),
        Line::from(format!(
            "   {} Python:      {}  {}",
            "\u{1F40D}",
            py_files,
            bar(py_files, max_bar, bar_width)
        )),
        Line::from(format!(
            "   {} TypeScript:  {}  {}",
            "\u{1F4D8}",
            ts_files,
            bar(ts_files, max_bar, bar_width)
        )),
        Line::from(format!(
            "   {} Other:       {}  {}",
            "\u{1F4C4}",
            other_files,
            bar(other_files, max_bar, bar_width)
        )),
        Line::from(""),
        Line::from(Span::styled(
            " Configuration",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from(format!(
            "   Model:      {}  {}",
            model_health_icon, app.config.default_model
        )),
        Line::from(format!("   Theme:      {}", app.config.active_theme)),
        Line::from(format!(
            "   Cache:      {}  ({})",
            if app.config.cache_enabled {
                "\u{2705} Enabled"
            } else {
                "\u{274C} Disabled"
            },
            app.config.max_memory_items
        )),
        Line::from(format!("   Timeout:    {}s", app.config.response_timeout)),
        Line::from(""),
        Line::from(Span::styled(
            " Latency & Performance",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from(format!(
            "   Avg Latency:      {} ms",
            if avg_latency > 0.0 {
                format!("{:.0}", avg_latency)
            } else {
                "N/A".to_string()
            }
        )),
        Line::from(format!("   Total LLM Calls:  {}", app.total_llm_calls)),
        Line::from(""),
        Line::from(Span::styled(
            " System Utilization (simulated)",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from(format!(
            "   CPU:     [{}{}]  {:.0}%",
            "\u{2588}".repeat((app.profiler_gauge_cpu / 10.0) as usize),
            "\u{2591}".repeat(10usize.saturating_sub((app.profiler_gauge_cpu / 10.0) as usize)),
            app.profiler_gauge_cpu
        )),
        Line::from(format!(
            "   Memory:  [{}{}]  {:.0}%",
            "\u{2588}".repeat((app.profiler_gauge_mem / 10.0) as usize),
            "\u{2591}".repeat(10usize.saturating_sub((app.profiler_gauge_mem / 10.0) as usize)),
            app.profiler_gauge_mem
        )),
        Line::from(""),
        Line::from(Span::styled(
            " Session Timeline",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from(format!(
            "   {}  Started TUI session",
            if app.session_start_time > 0.0 {
                "\u{25B6}"
            } else {
                "\u{25CB}"
            }
        )),
        Line::from(format!(
            "   {}  Workspace scanned ({} files)",
            "\u{25B6}",
            app.file_tree.len()
        )),
        Line::from(format!(
            "   {}  {} LLM calls made",
            "\u{25B6}", app.total_llm_calls
        )),
        Line::from(""),
        Line::from(Span::styled(
            " Press 'h' to refresh health checks. Esc to close.",
            Style::default().fg(app.theme.message_system),
        )),
    ];

    let para = Paragraph::new(lines)
        .block(block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(para, popup_area);
}

fn draw_provider_health(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(70, 55, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🏥 Provider Health (Esc to close · h to refresh) ");

    let mut lines: Vec<Line> = Vec::new();

    // Helper: format a health status line with color
    let provider_name = |provider: &str| -> String {
        match provider {
            "ollama" => format!("{} Ollama", "\u{1F916}"),
            "llamacpp" => format!("{} llama.cpp", "\u{1F999}"),
            "openrouter" => format!("{} OpenRouter", "\u{1F310}"),
            _ => provider.to_string(),
        }
    };

    for (provider, (status, latency, error)) in &app.ollama_health_entries {
        let status_icon = match status.as_str() {
            "healthy" => "\u{2705}",
            "error" | "unavailable" => "\u{274C}",
            _ => "\u{2753}",
        };
        let status_color = match status.as_str() {
            "healthy" => ratatui::style::Color::Green,
            "error" | "unavailable" => ratatui::style::Color::Red,
            _ => app.theme.message_system,
        };
        let latency_str = if *latency > 0.0 {
            format!("{:.0} ms", latency)
        } else {
            "---".to_string()
        };
        let error_str = if let Some(msg) = error {
            if !msg.is_empty() {
                format!("  Error: {}", msg)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        lines.push(Line::from(vec![
            Span::styled(
                format!("  {}  {}    ", status_icon, provider_name(provider)),
                Style::default()
                    .fg(app.theme.fg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("[{}]", status),
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  Latency: {}", latency_str),
                Style::default().fg(app.theme.fg),
            ),
        ]));
        if provider == "llamacpp" {
            if let Some(ts) = app.last_llamacpp_timings.as_ref() {
                lines.push(Line::from(Span::styled(
                    format!(
                        "        ⚡ {:.0} tok/s · {} tokens generated (last request)",
                        ts.predicted_per_second, ts.tokens_generated
                    ),
                    Style::default().fg(ratatui::style::Color::Yellow),
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    "        ⚡ No generation stats yet",
                    Style::default().fg(app.theme.message_system),
                )));
            }
        }
        if !error_str.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("        {}", error_str),
                Style::default().fg(ratatui::style::Color::Red),
            )));
        }
        lines.push(Line::from(""));
    }

    // Add provider-specific detail cards
    lines.push(Line::from(Span::styled(
        " Connection Details",
        Style::default()
            .fg(app.theme.fg)
            .add_modifier(Modifier::UNDERLINED),
    )));
    lines.push(Line::from(format!(
        "   Ollama URI:    {}",
        app.config.ollama_url
    )));
    lines.push(Line::from(format!(
        "   Llama.cpp URI: {}",
        app.config.llama_cpp_url
    )));
    lines.push(Line::from(format!(
        "   OpenRouter:   {}",
        if app.config.api_keys.openrouter_api_key.is_some() {
            "\u{2705} Key configured"
        } else {
            "\u{274C} No API key"
        }
    )));
    lines.push(Line::from(""));

    lines.push(Line::from(Span::styled(
        " Active Model",
        Style::default()
            .fg(app.theme.fg)
            .add_modifier(Modifier::UNDERLINED),
    )));
    lines.push(Line::from(format!("   {}", app.config.default_model)));
    lines.push(Line::from(""));

    // Last check info
    if app.last_health_check > 0.0 {
        let elapsed = (current_timestamp() - app.last_health_check).max(0.0);
        lines.push(Line::from(Span::styled(
            format!(" Last health check: {:.0}s ago", elapsed),
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            " No health check performed yet. Press 'h' to run.",
            Style::default().fg(app.theme.message_system),
        )));
    }
    if app.health_check_in_progress {
        let frames = [
            '\u{280B}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283C}', '\u{2834}', '\u{2826}',
            '\u{2827}', '\u{2807}', '\u{280F}',
        ];
        let frame = frames[app.spinner_tick % frames.len()];
        lines.push(Line::from(Span::styled(
            format!(" {} Checking provider status...", frame),
            Style::default().fg(app.theme.accent),
        )));
    }

    let para = Paragraph::new(lines)
        .block(block)
        .style(Style::default().fg(app.theme.fg))
        .scroll((app.provider_health_scroll, 0));
    f.render_widget(para, popup_area);
}

fn draw_project_analyzer(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(60, 60, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 📈 Project Analyzer (Esc to close) ");

    let mut rust_files = 0;
    let mut py_files = 0;
    let mut ts_files = 0;
    let mut other = 0;

    for file in &app.file_tree {
        if file.ends_with(".rs") {
            rust_files += 1;
        } else if file.ends_with(".py") {
            py_files += 1;
        } else if file.ends_with(".ts") || file.ends_with(".tsx") {
            ts_files += 1;
        } else {
            other += 1;
        }
    }

    let text = format!(
        "\n  Workspace Scan Results:\n\n  Total Files: {}\n\n  🦀 Rust:       {}\n  🐍 Python:     {}\n  📘 TypeScript: {}\n  📄 Other:      {}",
        app.file_tree.len(), rust_files, py_files, ts_files, other
    );

    let para = Paragraph::new(text)
        .block(block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(para, popup_area);
}

fn draw_git_commit(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(50, 40, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 📝 Git Commit (Enter to commit, Esc to cancel) ");

    let modified_count = app.git_status.len();

    let text = format!(
        "\n  Staging {} files...\n\n  Message:\n  > {}\n\n  (Type your message and press Enter)",
        modified_count, app.commit_message
    );

    let para = Paragraph::new(text)
        .block(block)
        .style(Style::default().fg(app.theme.fg));
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

fn draw_feature_navigator(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(50, 60, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🚀 Feature Navigator (↑↓ Enter, Esc to close) ");

    let items: Vec<ListItem> = FEATURE_LIST
        .iter()
        .enumerate()
        .map(|(i, (name, desc))| {
            let style = if i == app.feature_nav_selected {
                Style::default()
                    .fg(app.theme.highlight_fg)
                    .bg(app.theme.highlight)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(app.theme.fg)
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!(" {} ", name), style),
                Span::styled(
                    format!("— {}", desc),
                    Style::default().fg(app.theme.message_system),
                ),
            ]))
        })
        .collect();

    let list = List::new(items).block(block);
    let mut state = ListState::default();
    state.select(Some(app.feature_nav_selected));
    f.render_stateful_widget(list, popup_area, &mut state);
}

// ── ByteBot Agent Panel ─────────────────────────────────────────────────────

fn draw_bytebot_panel(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(72, 70, area);
    f.render_widget(Clear, popup_area);

    let status_icon = if app.bytebot_running {
        let frames = [
            '\u{280B}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283C}', '\u{2834}', '\u{2826}',
            '\u{2827}', '\u{2807}', '\u{280F}',
        ];
        frames[app.spinner_tick % frames.len()]
    } else {
        '\u{25C9}'
    };
    let status_str = if app.bytebot_running {
        " Running"
    } else {
        " Idle"
    };

    let title = format!(
        " {} ByteBot Agent [{}]{} (Enter:run, Esc:close) ",
        "\u{1F916}", status_icon, status_str
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    // Layout: command input (3) | steps (rest) split into steps (40%) + log (60%)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // command input
            Constraint::Min(1),    // steps + log
        ])
        .split(inner);

    // ── Command Input ────────────────────────────────────────────────────────
    let cmd_focused = !app.bytebot_running;
    let cmd_border = if cmd_focused {
        Style::default().fg(app.theme.accent)
    } else {
        Style::default().fg(app.theme.border)
    };
    let cmd_block = Block::default()
        .borders(Borders::ALL)
        .border_style(cmd_border)
        .title(" ⌨️  Command (e.g., 'update deps', 'analyze tests') ");
    let display_cmd = if app.bytebot_running {
        "Executing... press Esc to return".to_string()
    } else if app.bytebot_command.is_empty() {
        "Type a command and press Enter".to_string()
    } else {
        app.bytebot_command.clone()
    };
    let cmd_para = Paragraph::new(display_cmd)
        .block(cmd_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(cmd_para, chunks[0]);

    // Show cursor in command input
    if cmd_focused {
        let cursor_x = chunks[0].x + 1 + app.bytebot_cursor as u16;
        let cursor_y = chunks[0].y + 1;
        if cursor_x < chunks[0].x + chunks[0].width.saturating_sub(1) {
            f.set_cursor_position((cursor_x, cursor_y));
        }
    }

    // ── Steps + Log ──────────────────────────────────────────────────────────
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // steps
            Constraint::Percentage(60), // log
        ])
        .split(chunks[1]);

    // Steps panel
    let mut steps_lines: Vec<Line> = Vec::new();
    if !app.bytebot_steps.is_empty() || app.bytebot_running {
        steps_lines.push(Line::from(Span::styled(
            " Execution Steps",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )));
        steps_lines.push(Line::from(""));

        for (i, (step_name, status)) in app.bytebot_steps.iter().enumerate() {
            let (icon, color) = match status.as_str() {
                "done" => ("\u{2705}".to_string(), ratatui::style::Color::Green),
                "running" => {
                    let running_icon: String = if app.bytebot_running {
                        let frames = [
                            '\u{280B}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283C}', '\u{2834}',
                            '\u{2826}', '\u{2827}', '\u{2807}', '\u{280F}',
                        ];
                        frames[app.spinner_tick % frames.len()].to_string()
                    } else {
                        "\u{23F3}".to_string()
                    };
                    (running_icon, ratatui::style::Color::Yellow)
                }
                "failed" => ("\u{274C}".to_string(), ratatui::style::Color::Red),
                _ => ("\u{25CB}".to_string(), app.theme.message_system), // pending / unknown
            };
            steps_lines.push(Line::from(vec![
                Span::styled(format!(" {} ", icon), Style::default().fg(color)),
                Span::styled(
                    format!("Step {}: {}", i + 1, step_name),
                    Style::default().fg(if status == "done" || status == "running" {
                        app.theme.fg
                    } else {
                        app.theme.message_system
                    }),
                ),
            ]));
        }

        // Progress bar
        if app.bytebot_running || app.bytebot_progress > 0.0 {
            steps_lines.push(Line::from(""));
            let bar_width = 20usize;
            let filled = (app.bytebot_progress * bar_width as f64).round() as usize;
            let filled = filled.min(bar_width);
            let empty = bar_width.saturating_sub(filled);
            let pct = (app.bytebot_progress * 100.0).round();
            let bar = format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty));
            let bar_str = format!("  {} {:.0}%", bar, pct);
            steps_lines.push(Line::from(Span::styled(
                bar_str,
                Style::default().fg(app.theme.accent),
            )));
        }
    } else {
        // Empty state
        steps_lines.push(Line::from(""));
        steps_lines.push(Line::from(Span::styled(
            "  Enter a command above and press Enter",
            Style::default().fg(app.theme.message_system),
        )));
        steps_lines.push(Line::from(Span::styled(
            "  to start ByteBot execution.",
            Style::default().fg(app.theme.message_system),
        )));
    }

    let steps_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" \u{1F4CB} Steps ");
    let steps_para = Paragraph::new(steps_lines)
        .block(steps_block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false });
    f.render_widget(steps_para, bottom[0]);

    // Log panel
    let mut log_lines: Vec<Line> = Vec::new();
    log_lines.push(Line::from(Span::styled(
        " Execution Log",
        Style::default()
            .fg(app.theme.fg)
            .add_modifier(Modifier::UNDERLINED),
    )));
    log_lines.push(Line::from(""));

    if app.bytebot_log.is_empty() {
        log_lines.push(Line::from(Span::styled(
            "  No execution log yet.",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        for entry in &app.bytebot_log {
            // Color based on content prefix
            let color = if entry.starts_with('✅') {
                ratatui::style::Color::Green
            } else if entry.starts_with('❌') || entry.starts_with("failed") {
                ratatui::style::Color::Red
            } else {
                app.theme.fg
            };
            log_lines.push(Line::from(Span::styled(
                format!("  {}", entry),
                Style::default().fg(color),
            )));
        }
    }

    let log_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" \u{1F4DD} Log ");
    let log_para = Paragraph::new(log_lines)
        .block(log_block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false });
    f.render_widget(log_para, bottom[1]);

    // Command history
    let history_lines: Vec<Line> = if !app.bytebot_history.is_empty() {
        let mut hl = vec![
            Line::from(Span::styled(
                " History (press Up to recall)",
                Style::default()
                    .fg(app.theme.fg)
                    .add_modifier(Modifier::UNDERLINED),
            )),
            Line::from(""),
        ];
        for cmd in app.bytebot_history.iter().rev().take(5) {
            hl.push(Line::from(Span::styled(
                format!("  \u{25B6} {}", cmd),
                Style::default().fg(app.theme.message_system),
            )));
        }
        hl
    } else {
        vec![
            Line::from(Span::styled(
                " History",
                Style::default()
                    .fg(app.theme.fg)
                    .add_modifier(Modifier::UNDERLINED),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  No previous commands",
                Style::default().fg(app.theme.message_system),
            )),
        ]
    };

    // Add history as a small section below the split panels if there's room.
    // Guard on `popup_area` itself — the arithmetic below is in its coordinate
    // space, and a narrow popup would underflow `width - 2`.
    if bottom[1].height > 8 && popup_area.height > 7 && popup_area.width > 2 {
        let hist_bottom_y = popup_area.y + popup_area.height - 7;
        let hist_area = Rect {
            x: popup_area.x + 1,
            y: hist_bottom_y,
            width: popup_area.width - 2,
            height: 6,
        };
        f.render_widget(Clear, hist_area);
        let hist_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(app.theme.border))
            .title(" History ");
        let inner_hist = hist_block.inner(hist_area);
        f.render_widget(hist_block, hist_area);
        let hist_para = Paragraph::new(history_lines).style(Style::default().fg(app.theme.fg));
        f.render_widget(hist_para, inner_hist);
    }
}

// ── Project Context (/init) Overlay ────────────────────────────────────────

fn draw_project_init_panel(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(72, 62, area);
    f.render_widget(Clear, popup_area);

    let status_icon: String = if app.init_running {
        let frames = [
            '\u{280B}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283C}', '\u{2834}', '\u{2826}',
            '\u{2827}', '\u{2807}', '\u{280F}',
        ];
        frames[app.spinner_tick % frames.len()].to_string()
    } else {
        "\u{2705}".to_string()
    };
    let status = if app.init_running {
        " Running"
    } else {
        " Done"
    };

    let title = format!(
        " \u{26A1} Project Context /init [{}]{}  (Esc:close, /init abort) ",
        status_icon, status
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);
    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);

    // Steps (40%) + log (60%)
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(chunks[0]);

    let mut steps_lines: Vec<Line> = Vec::new();
    steps_lines.push(Line::from(Span::styled(
        " Structural pass 1 — zero LLM calls",
        Style::default()
            .fg(app.theme.fg)
            .add_modifier(Modifier::UNDERLINED),
    )));
    steps_lines.push(Line::from(""));
    for (name, status) in &app.init_steps {
        let (icon, color) = match status.as_str() {
            "done" => ("\u{2705}".to_string(), ratatui::style::Color::Green),
            "running" => ("\u{23F3}".to_string(), ratatui::style::Color::Yellow),
            "failed" => ("\u{274C}".to_string(), ratatui::style::Color::Red),
            _ => ("\u{25CB}".to_string(), app.theme.message_system),
        };
        steps_lines.push(Line::from(vec![
            Span::styled(format!(" {} ", icon), Style::default().fg(color)),
            Span::styled(name.clone(), Style::default().fg(app.theme.fg)),
        ]));
    }

    let steps_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" Steps ");
    f.render_widget(
        Paragraph::new(steps_lines)
            .block(steps_block)
            .style(Style::default().fg(app.theme.fg)),
        body[0],
    );

    // Log lines, most recent at the bottom.
    let max_lines = body[1].height.saturating_sub(2) as usize;
    let count = std::cmp::min(max_lines, app.init_log.len());
    let start = app.init_log.len() - count;
    let log_lines: Vec<Line> = app.init_log[start..]
        .iter()
        .map(|entry| {
            let color = if entry.contains('\u{274C}') {
                ratatui::style::Color::Red
            } else if entry.contains('\u{2705}') {
                ratatui::style::Color::Green
            } else {
                app.theme.fg
            };
            Line::from(Span::styled(
                format!("  {}", entry),
                Style::default().fg(color),
            ))
        })
        .collect();
    let log_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" Log ");
    f.render_widget(
        Paragraph::new(log_lines)
            .block(log_block)
            .style(Style::default().fg(app.theme.fg))
            .wrap(Wrap { trim: false }),
        body[1],
    );

    // Progress bar.
    let bar_width = chunks[1].width.saturating_sub(2);
    let filled = ((app.init_progress * bar_width as f64) as u16).min(bar_width);
    let mut bar = String::new();
    for _ in 0..filled {
        bar.push('\u{2588}');
    }
    for _ in filled..bar_width {
        bar.push(' ');
    }
    let pct = (app.init_progress * 100.0).round() as u64;
    let progress_para = Paragraph::new(Span::styled(
        format!(" {bar} ({pct}%)"),
        Style::default().fg(app.theme.accent),
    ));
    f.render_widget(progress_para, chunks[1]);
}

// ── Collaboration Hub Panel ────────────────────────────────────────────────

fn draw_collaboration_hub(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(75, 72, area);
    f.render_widget(Clear, popup_area);

    let status_icon: String = match app.collab_sync_status.as_str() {
        "connected" | "synced" => "✅".to_string(),
        "syncing" | "connecting" => {
            let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            frames[app.spinner_tick % frames.len()].to_string()
        }
        "error" | "disconnected" => "❌".to_string(),
        _ => "❓".to_string(),
    };

    let title = format!(
        " 👥 Collaboration Hub [{}] (Enter:start, Esc:close) ",
        status_icon
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    // Layout: session info (3) | body split into members (35%) + activity (65%)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // session info bar
            Constraint::Min(1),    // body
        ])
        .split(inner);

    // ── Session Info Bar ─────────────────────────────────────────────────────
    let session_active = app.collab_session_active;
    let session_border = if session_active {
        Style::default().fg(app.theme.accent)
    } else {
        Style::default().fg(app.theme.border)
    };
    let session_block = Block::default()
        .borders(Borders::ALL)
        .border_style(session_border)
        .title(" \u{1F310} Session ");

    let session_display = if !session_active {
        " Press Enter to start a collaboration session".to_string()
    } else {
        let status_color = match app.collab_sync_status.as_str() {
            "synced" | "connected" => "\u{2705}",
            "syncing" | "connecting" => "\u{1F504}",
            "error" | "disconnected" => "\u{274C}",
            _ => "\u{2753}",
        };
        format!(
            " ID: {}   Status: {} {}   Pending: {}   Members: {}",
            app.collab_session_id,
            status_color,
            app.collab_sync_status,
            app.collab_pending_changes,
            app.collab_members.len(),
        )
    };
    let session_para = Paragraph::new(session_display)
        .block(session_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(session_para, chunks[0]);

    // ── Body: Members (35%) + Activity (65%) ─────────────────────────────────
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(35), // members
            Constraint::Percentage(65), // activity
        ])
        .split(chunks[1]);

    // Members panel
    let mut member_lines: Vec<Line> = Vec::new();
    if !app.collab_members.is_empty() {
        member_lines.push(Line::from(Span::styled(
            " Team Members",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )));
        member_lines.push(Line::from(""));

        for (name, status, connection) in &app.collab_members {
            let (status_icon, status_color) = match status.as_str() {
                "online" => ("\u{25CF}", ratatui::style::Color::Green),
                "away" => ("\u{25CB}", ratatui::style::Color::Yellow),
                "busy" => ("\u{25A0}", ratatui::style::Color::Red),
                _ => ("?", app.theme.message_system),
            };
            let role_badge = match name.as_str() {
                "You (local)" => " [Admin]",
                "alice" => " [Editor]",
                "bob" => " [Viewer]",
                "carol" => " [Editor]",
                _ => "",
            };
            member_lines.push(Line::from(vec![
                Span::styled(
                    format!(" {} ", status_icon),
                    Style::default().fg(status_color),
                ),
                Span::styled(
                    format!("{}{}  {}", name, role_badge, connection),
                    Style::default().fg(app.theme.fg),
                ),
            ]));
        }
    } else {
        member_lines.push(Line::from(""));
        member_lines.push(Line::from(Span::styled(
            "  No team members yet.",
            Style::default().fg(app.theme.message_system),
        )));
        member_lines.push(Line::from(Span::styled(
            "  Press Enter to start a session.",
            Style::default().fg(app.theme.message_system),
        )));
    }

    // Connection info at bottom of members panel
    if session_active {
        member_lines.push(Line::from(""));
        member_lines.push(Line::from(Span::styled(
            " Connection",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )));
        member_lines.push(Line::from(""));
        member_lines.push(Line::from(format!(
            "   Port:  {:04}",
            (current_timestamp() as u64 % 60000 + 8000)
        )));
        member_lines.push(Line::from("   Protocol: WebSocket (TLS)"));
        member_lines.push(Line::from("   Latency: <15ms"));
        if app.collab_last_sync > 0.0 {
            let elapsed = (current_timestamp() - app.collab_last_sync).max(0.0);
            member_lines.push(Line::from(format!("   Last sync: {:.0}s ago", elapsed)));
        }
    }

    let member_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" \u{1F465} Members ");
    let member_para = Paragraph::new(member_lines)
        .block(member_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(member_para, body[0]);

    // Activity/Log panel
    let mut activity_lines: Vec<Line> = Vec::new();
    activity_lines.push(Line::from(Span::styled(
        " Activity Feed",
        Style::default()
            .fg(app.theme.fg)
            .add_modifier(Modifier::UNDERLINED),
    )));
    activity_lines.push(Line::from(""));

    if app.collab_activity_log.is_empty() {
        activity_lines.push(Line::from(Span::styled(
            "  No activity yet. Start a session to begin.",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        for entry in &app.collab_activity_log {
            let color = if entry.starts_with('\u{2705}') || entry.starts_with('\u{1F504}') {
                ratatui::style::Color::Green
            } else if entry.starts_with('\u{274C}') || entry.starts_with('\u{26A0}') {
                ratatui::style::Color::Red
            } else if entry.starts_with('\u{1F4E4}') || entry.starts_with('\u{1F4E5}') {
                ratatui::style::Color::Cyan
            } else if entry.starts_with('\u{1F464}') {
                ratatui::style::Color::Yellow
            } else {
                app.theme.fg
            };
            activity_lines.push(Line::from(Span::styled(
                format!("  {}", entry),
                Style::default().fg(color),
            )));
        }

        // Sync status indicator at bottom
        if app.collab_sync_status == "syncing" || app.collab_sync_status == "connecting" {
            activity_lines.push(Line::from(""));
            let frames = [
                '\u{280B}', '\u{2819}', '\u{2839}', '\u{2838}', '\u{283C}', '\u{2834}', '\u{2826}',
                '\u{2827}', '\u{2807}', '\u{280F}',
            ];
            let frame = frames[app.spinner_tick % frames.len()];
            activity_lines.push(Line::from(Span::styled(
                format!(" {} Synchronizing...", frame),
                Style::default().fg(app.theme.accent),
            )));
        }
    }

    let activity_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" \u{1F4AC} Activity ");
    let activity_para = Paragraph::new(activity_lines)
        .block(activity_block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false });
    f.render_widget(activity_para, body[1]);
}

// ── Voice Interface Panel ──────────────────────────────────────────────────

fn draw_voice_interface(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(65, 60, area);
    f.render_widget(Clear, popup_area);

    let status_icon = match app.voice_status.as_str() {
        "listening" => "🎤",
        "processing" => {
            let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let idx = app.spinner_tick % frames.len();
            // Return as &str slice
            if idx < 5 {
                "🔄"
            } else {
                "⚡"
            }
        }
        "speaking" => "🔊",
        _ => "🎙️",
    };

    let title = format!(
        " 🎙️ Voice Interface [{}] (Enter:start, Esc:close) ",
        status_icon
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // audio meter + status
            Constraint::Length(6), // recent commands
            Constraint::Min(1),    // transcript
        ])
        .split(inner);

    // ── Audio Meter + Status ────────────────────────────────────────────────
    let status_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📡 Audio Input ");

    let bar_width = 30usize;
    let filled = (app.voice_level * bar_width as f64).round() as usize;
    let filled = filled.min(bar_width);
    let empty = bar_width.saturating_sub(filled);
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
    let meter = format!(
        " Level: [{}] {:.0}%\n Status: {} {}",
        bar,
        app.voice_level * 100.0,
        status_icon,
        app.voice_status,
    );
    let status_para = Paragraph::new(meter)
        .block(status_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(status_para, chunks[0]);

    // ── Recent Commands ─────────────────────────────────────────────────────
    let cmd_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📋 Recent Commands ");

    let mut cmd_lines: Vec<Line> = Vec::new();
    if app.voice_commands.is_empty() {
        cmd_lines.push(Line::from(Span::styled(
            "  Press Enter to start voice recognition",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        for (cmd, result) in app.voice_commands.iter().rev().take(3) {
            cmd_lines.push(Line::from(Span::styled(
                format!("  🗣️  {}", cmd),
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            )));
            cmd_lines.push(Line::from(Span::styled(
                format!("     {}", result),
                Style::default().fg(app.theme.fg),
            )));
        }
    }
    let cmd_para = Paragraph::new(cmd_lines)
        .block(cmd_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(cmd_para, chunks[1]);

    // ── Transcript Log ──────────────────────────────────────────────────────
    let trans_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📝 Transcript ");

    let mut trans_lines: Vec<Line> = Vec::new();
    if app.voice_transcript.is_empty() {
        trans_lines.push(Line::from(Span::styled(
            "  No speech detected yet.",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        for entry in &app.voice_transcript {
            let color = if entry.starts_with('✅') {
                ratatui::style::Color::Green
            } else if entry.starts_with('❌') {
                ratatui::style::Color::Red
            } else {
                app.theme.fg
            };
            trans_lines.push(Line::from(Span::styled(
                format!("  {}", entry),
                Style::default().fg(color),
            )));
        }
    }
    let trans_para = Paragraph::new(trans_lines)
        .block(trans_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(trans_para, chunks[2]);
}

// ── Terminal Assistant Panel ────────────────────────────────────────────────

fn draw_terminal_assistant(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(75, 65, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 💡 Terminal Assistant (Enter:refresh, Esc:close) ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // status
            Constraint::Min(1),    // suggestions
            Constraint::Length(4), // history
        ])
        .split(inner);

    // ── Status ──────────────────────────────────────────────────────────────
    let status_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🤖 AI Shell Helper ");
    let status_text = if !app.term_asst_output.is_empty() {
        &app.term_asst_output
    } else {
        "Press Enter to load command suggestions"
    };
    let status_para = Paragraph::new(status_text)
        .block(status_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(status_para, chunks[0]);

    // ── Command Suggestions ─────────────────────────────────────────────────
    let sugg_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📋 Suggested Commands ");

    let mut sugg_lines: Vec<Line> = Vec::new();
    if app.term_asst_suggestions.is_empty() {
        sugg_lines.push(Line::from(Span::styled(
            "  Press Enter to load suggestions. Ask questions like:",
            Style::default().fg(app.theme.message_system),
        )));
        sugg_lines.push(Line::from(Span::styled(
            "    • 'how to find large files'",
            Style::default().fg(app.theme.message_system),
        )));
        sugg_lines.push(Line::from(Span::styled(
            "    • 'check disk usage'",
            Style::default().fg(app.theme.message_system),
        )));
        sugg_lines.push(Line::from(Span::styled(
            "    • 'find all python files'",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        for suggestion in &app.term_asst_suggestions {
            let color = if suggestion.starts_with("✅") {
                ratatui::style::Color::Green
            } else if suggestion.starts_with("⚠️") {
                ratatui::style::Color::Yellow
            } else {
                app.theme.fg
            };
            sugg_lines.push(Line::from(Span::styled(
                format!("  {}", suggestion),
                Style::default().fg(color),
            )));
        }
    }
    let sugg_para = Paragraph::new(sugg_lines)
        .block(sugg_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(sugg_para, chunks[1]);

    // ── History ─────────────────────────────────────────────────────────────
    let hist_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📜 Execution History ");
    let hist_text = if app.term_asst_history.is_empty() {
        "  No commands executed yet."
    } else {
        "  Executed commands will appear here."
    };
    let hist_para = Paragraph::new(hist_text)
        .block(hist_block)
        .style(Style::default().fg(app.theme.message_system));
    f.render_widget(hist_para, chunks[2]);
}

// ── Security Auditor Panel ──────────────────────────────────────────────────

fn draw_security_auditor(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(75, 70, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🛡️ Security Auditor (Enter:scan, Esc:close) ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // summary cards
            Constraint::Min(1),    // findings list
        ])
        .split(inner);

    // ── Summary Cards ───────────────────────────────────────────────────────
    let (critical, high, medium, low) = app.sec_scan_summary;
    let total = critical + high + medium + low;
    let severity_bar = |count: u32, max: u32, color: ratatui::style::Color| -> Line {
        let w = 15usize;
        let filled = if max > 0 {
            (count as f64 / max as f64 * w as f64).round() as usize
        } else {
            0
        };
        let filled = filled.min(w);
        let empty = w.saturating_sub(filled);
        Line::from(Span::styled(
            format!("  {}  {}{}", count, "█".repeat(filled), "░".repeat(empty)),
            Style::default().fg(color),
        ))
    };

    let max_sev = critical.max(high).max(medium).max(low).max(1);

    let summary_lines = vec![
        Line::from(Span::styled(
            format!(
                " Vulnerability Scan  |  Total: {}  |  Progress: {:.0}%",
                total,
                app.sec_scan_progress * 100.0
            ),
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        severity_bar(critical, max_sev, ratatui::style::Color::Red),
        severity_bar(high, max_sev, ratatui::style::Color::Yellow),
        severity_bar(medium, max_sev, ratatui::style::Color::Cyan),
        severity_bar(low, max_sev, ratatui::style::Color::Green),
    ];

    let summary_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📊 Risk Summary ");
    let summary_para = Paragraph::new(summary_lines)
        .block(summary_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(summary_para, chunks[0]);

    // ── Findings List ───────────────────────────────────────────────────────
    let find_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🔎 Findings ");

    let mut find_lines: Vec<Line> = Vec::new();
    if app.sec_scan_results.is_empty() && !app.sec_scan_active {
        find_lines.push(Line::from(Span::styled(
            "  Press Enter to start a vulnerability scan.",
            Style::default().fg(app.theme.message_system),
        )));
    } else if app.sec_scan_results.is_empty() && app.sec_scan_active {
        find_lines.push(Line::from(Span::styled(
            "  Scanning...",
            Style::default().fg(app.theme.accent),
        )));
    } else {
        for (severity, category, location) in &app.sec_scan_results {
            let (icon, color) = match severity.as_str() {
                "Critical" => ("🔴", ratatui::style::Color::Red),
                "High" => ("🟡", ratatui::style::Color::Yellow),
                "Medium" => ("🔵", ratatui::style::Color::Cyan),
                _ => ("🟢", ratatui::style::Color::Green),
            };
            find_lines.push(Line::from(vec![
                Span::styled(format!(" {} ", icon), Style::default().fg(color)),
                Span::styled(
                    format!("[{}] {} — {}", severity, category, location),
                    Style::default()
                        .fg(app.theme.fg)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
        }
    }

    // Log entries at bottom of findings
    if !app.sec_scan_log.is_empty() {
        find_lines.push(Line::from(""));
        find_lines.push(Line::from(Span::styled(
            " Scan Log",
            Style::default()
                .fg(app.theme.message_system)
                .add_modifier(Modifier::UNDERLINED),
        )));
        for entry in app.sec_scan_log.iter().rev().take(5) {
            let color = if entry.starts_with('❌') {
                ratatui::style::Color::Red
            } else if entry.starts_with('⚠') {
                ratatui::style::Color::Yellow
            } else if entry.starts_with('🚨') {
                ratatui::style::Color::Red
            } else {
                app.theme.fg
            };
            find_lines.push(Line::from(Span::styled(
                format!("  {}", entry),
                Style::default().fg(color),
            )));
        }
    }

    let find_para = Paragraph::new(find_lines)
        .block(find_block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false })
        .scroll((app.security_scroll, 0));
    f.render_widget(find_para, chunks[1]);
}

// ── Performance Profiler Panel ──────────────────────────────────────────────

fn draw_performance_profiler(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(72, 68, area);
    f.render_widget(Clear, popup_area);

    let status_indicator = if app.profiler_running {
        let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        frames[app.spinner_tick % frames.len()]
    } else {
        '●'
    };

    let title = format!(
        " ⚡ Performance Profiler [{}] (Enter:profile, Esc:close) ",
        status_indicator
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // gauges
            Constraint::Min(1),    // function list
        ])
        .split(inner);

    // ── Gauges ──────────────────────────────────────────────────────────────
    fn gauge_block(title: &str, value: f64, max: f64, color: ratatui::style::Color) -> Line<'_> {
        let w = 15usize;
        let pct = if max > 0.0 {
            (value / max * 100.0).min(100.0).round()
        } else {
            0.0
        };
        let filled = (pct / 100.0 * w as f64).round() as usize;
        let filled = filled.min(w);
        let empty = w.saturating_sub(filled);
        Line::from(Span::styled(
            format!(
                " {}  [{}{}]  {:.0}%",
                title,
                "█".repeat(filled),
                "░".repeat(empty),
                pct
            ),
            Style::default().fg(color),
        ))
    }

    let gauge_lines = vec![
        Line::from(Span::styled(
            " System Metrics",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from(""),
        gauge_block(
            "CPU     ",
            app.profiler_gauge_cpu,
            100.0,
            ratatui::style::Color::Cyan,
        ),
        gauge_block(
            "Memory  ",
            app.profiler_gauge_mem,
            100.0,
            ratatui::style::Color::Magenta,
        ),
        gauge_block(
            "Latency ",
            app.profiler_gauge_latency,
            500.0,
            ratatui::style::Color::Yellow,
        ),
    ];

    let gauge_block_w = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📈 Gauges ");
    let gauge_para = Paragraph::new(gauge_lines)
        .block(gauge_block_w)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(gauge_para, chunks[0]);

    // ── Function List ───────────────────────────────────────────────────────
    let func_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📊 Profiled Functions ");

    let mut func_lines: Vec<Line> = Vec::new();
    if app.profiler_functions.is_empty() && !app.profiler_running {
        func_lines.push(Line::from(Span::styled(
            "  Press Enter to start profiling.",
            Style::default().fg(app.theme.message_system),
        )));
        func_lines.push(Line::from(Span::styled(
            "  Profiles CPU time, memory, and call frequency.",
            Style::default().fg(app.theme.message_system),
        )));
    } else if app.profiler_functions.is_empty() && app.profiler_running {
        func_lines.push(Line::from(Span::styled(
            "  Profiling in progress...",
            Style::default().fg(app.theme.accent),
        )));
    } else {
        // Header
        func_lines.push(Line::from(vec![
            Span::styled(
                "  Function",
                Style::default()
                    .fg(app.theme.fg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "        Time(ms)  Mem(MB)  Calls",
                Style::default().fg(app.theme.fg),
            ),
        ]));
        func_lines.push(Line::from(Span::styled(
            "  ".to_owned() + &"─".repeat(45),
            Style::default().fg(app.theme.border),
        )));

        for (name, time_ms, mem_mb, calls) in &app.profiler_functions {
            let hot = *time_ms > 200.0;
            let color = if hot {
                ratatui::style::Color::Red
            } else {
                app.theme.fg
            };
            let hot_mark = if hot { " 🔥" } else { "  " };
            func_lines.push(Line::from(Span::styled(
                format!(
                    "  {:<15} {:>8.1} {:>8.1} {:>6}{}",
                    name, time_ms, mem_mb, calls, hot_mark
                ),
                Style::default().fg(color),
            )));
        }
    }

    let func_para = Paragraph::new(func_lines)
        .block(func_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(func_para, chunks[1]);
}

// ── Custom Models Panel ─────────────────────────────────────────────────────

fn draw_custom_models(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(70, 65, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🧩 Custom Models (↑↓ select, Esc:close) ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40), // profiles list
            Constraint::Percentage(60), // details
        ])
        .split(inner);

    // ── Profile List ────────────────────────────────────────────────────────
    let list_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📋 Profiles ");

    let mut profile_items: Vec<ListItem> = Vec::new();
    for (i, (name, provider, temp, tokens, _top_p)) in app.models_profiles.iter().enumerate() {
        let style = if i == app.models_selected {
            Style::default()
                .fg(app.theme.highlight_fg)
                .bg(app.theme.highlight)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(app.theme.fg)
        };
        let provider_icon = if provider == "ollama" { "🦙" } else { "🌐" };
        profile_items.push(ListItem::new(Line::from(Span::styled(
            format!(" {} {}  t={:.1}  tk={}", provider_icon, name, temp, tokens),
            style,
        ))));
    }

    if profile_items.is_empty() {
        profile_items.push(ListItem::new(Line::from(Span::styled(
            "  Press Enter to load profiles",
            Style::default().fg(app.theme.message_system),
        ))));
    }

    let profile_list = List::new(profile_items).block(list_block);
    let mut list_state = ratatui::widgets::ListState::default();
    list_state.select(Some(app.models_selected));
    f.render_stateful_widget(profile_list, chunks[0], &mut list_state);

    // ── Profile Details ─────────────────────────────────────────────────────
    let detail_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🔧 Parameters ");

    let mut detail_lines: Vec<Line> = Vec::new();
    if let Some((name, provider, temp, tokens, top_p)) =
        app.models_profiles.get(app.models_selected)
    {
        detail_lines.push(Line::from(Span::styled(
            format!("  {} (via {})", name, provider),
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        detail_lines.push(Line::from(""));

        // Temperature slider
        let slider_w = 20usize;
        let t_filled = ((temp / 2.0) * slider_w as f64).round() as usize;
        let t_filled = t_filled.min(slider_w);
        let t_empty = slider_w.saturating_sub(t_filled);
        detail_lines.push(Line::from(format!(
            "  Temperature: [{}{}] {:.2}",
            "█".repeat(t_filled),
            "░".repeat(t_empty),
            temp
        )));

        // Max tokens
        let tk_filled = ((*tokens as f64 / 8192.0) * slider_w as f64).round() as usize;
        let tk_filled = tk_filled.min(slider_w);
        let tk_empty = slider_w.saturating_sub(tk_filled);
        detail_lines.push(Line::from(format!(
            "  Max Tokens:  [{}{}] {}",
            "█".repeat(tk_filled),
            "░".repeat(tk_empty),
            tokens
        )));

        // Top-P
        let p_filled = ((top_p / 1.0) * slider_w as f64).round() as usize;
        let p_filled = p_filled.min(slider_w);
        let p_empty = slider_w.saturating_sub(p_filled);
        detail_lines.push(Line::from(format!(
            "  Top-P:       [{}{}] {:.2}",
            "█".repeat(p_filled),
            "░".repeat(p_empty),
            top_p
        )));

        detail_lines.push(Line::from(""));
        if !app.models_test_output.is_empty() {
            detail_lines.push(Line::from(Span::styled(
                format!("  Test: {}", app.models_test_output),
                Style::default().fg(app.theme.message_system),
            )));
        } else {
            detail_lines.push(Line::from(Span::styled(
                "  Use ↑↓ to select profiles.",
                Style::default().fg(app.theme.message_system),
            )));
        }
    }

    let detail_para = Paragraph::new(detail_lines)
        .block(detail_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(detail_para, chunks[1]);
}

// ── Learning Mode Panel ─────────────────────────────────────────────────────

fn draw_learning_mode(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(70, 70, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 📚 Learning Mode (Enter:start, Esc:close) ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header + progress
            Constraint::Min(1),    // content
        ])
        .split(inner);

    // ── Header + Progress ───────────────────────────────────────────────────
    let header_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🎯 Lesson Progress ");

    let bar_w = 25usize;
    let filled = (app.learn_progress_pct / 100.0 * bar_w as f64).round() as usize;
    let filled = filled.min(bar_w);
    let empty = bar_w.saturating_sub(filled);
    let progress_str = format!(
        "  {} / {}  |  [{}{}]  {:.0}%",
        app.learn_current_lesson,
        app.learn_total_lessons,
        "█".repeat(filled),
        "░".repeat(empty),
        app.learn_progress_pct,
    );

    let header_lines = vec![
        Line::from(Span::styled(
            format!(
                "  Lesson {}: {}",
                app.learn_current_lesson, app.learn_lesson_title
            ),
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            progress_str,
            Style::default().fg(app.theme.fg),
        )),
    ];
    let header_para = Paragraph::new(header_lines)
        .block(header_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(header_para, chunks[0]);

    // ── Lesson Content ──────────────────────────────────────────────────────
    let content_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📖 Content ");

    let mut content_lines: Vec<Line> = Vec::new();
    if app.learn_content.is_empty() {
        content_lines.push(Line::from(Span::styled(
            "  Press Enter to start learning.",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        // Lesson points
        for point in &app.learn_content {
            content_lines.push(Line::from(Span::styled(
                format!("  • {}", point),
                Style::default().fg(app.theme.fg),
            )));
        }
        content_lines.push(Line::from(""));

        // Code example
        if !app.learn_code_example.is_empty() {
            content_lines.push(Line::from(Span::styled(
                "  Code Example:",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            )));
            for line in app.learn_code_example.lines() {
                content_lines.push(Line::from(Span::styled(
                    format!("    {}", line),
                    Style::default().fg(ratatui::style::Color::Green),
                )));
            }
            content_lines.push(Line::from(""));
        }

        // Exercise prompt
        if !app.learn_exercise.is_empty() {
            content_lines.push(Line::from(Span::styled(
                "  Exercise:",
                Style::default()
                    .fg(ratatui::style::Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )));
            content_lines.push(Line::from(Span::styled(
                format!("    {}", app.learn_exercise),
                Style::default().fg(ratatui::style::Color::Yellow),
            )));
        }

        // Quiz section
        if app.learn_quiz_active {
            content_lines.push(Line::from(""));
            content_lines.push(Line::from(Span::styled(
                "  Quiz:",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            )));
            content_lines.push(Line::from(Span::styled(
                format!("    {}", app.learn_quiz_question),
                Style::default().fg(app.theme.fg),
            )));
            content_lines.push(Line::from(""));
            for (i, option) in app.learn_quiz_options.iter().enumerate() {
                let is_selected = i == app.learn_quiz_selected;
                let prefix = if is_selected && !app.learn_quiz_answered {
                    "  \u{25B6} "
                } else {
                    "    "
                };
                let style = if is_selected && !app.learn_quiz_answered {
                    Style::default()
                        .fg(app.theme.highlight_fg)
                        .bg(app.theme.highlight)
                } else if app.learn_quiz_answered && i == 0 {
                    Style::default().fg(ratatui::style::Color::Green)
                } else if app.learn_quiz_answered && is_selected && !app.learn_quiz_correct {
                    Style::default().fg(ratatui::style::Color::Red)
                } else {
                    Style::default().fg(app.theme.fg)
                };
                content_lines.push(Line::from(Span::styled(
                    format!("{}{}", prefix, option),
                    style,
                )));
            }
            if app.learn_quiz_answered {
                content_lines.push(Line::from(""));
                if app.learn_quiz_correct {
                    content_lines.push(Line::from(Span::styled(
                        "  Correct! +20% progress",
                        Style::default().fg(ratatui::style::Color::Green),
                    )));
                } else {
                    content_lines.push(Line::from(Span::styled(
                        "  Not quite. Try again next time!",
                        Style::default().fg(ratatui::style::Color::Yellow),
                    )));
                }
            } else {
                content_lines.push(Line::from(""));
                content_lines.push(Line::from(Span::styled(
                    "  Select with \u{2190}\u{2192}, confirm with Enter",
                    Style::default().fg(app.theme.message_system),
                )));
            }
        }
    }

    let content_para = Paragraph::new(content_lines)
        .block(content_block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false });
    f.render_widget(content_para, chunks[1]);
}

// ── Multi-Language Panel ────────────────────────────────────────────────────

fn draw_multi_language(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(70, 65, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🌐 Multi-Language (Enter:detect, Esc:close) ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // detection + supported
            Constraint::Percentage(50), // translation
        ])
        .split(inner);

    // ── Left: Detection + Supported Languages ───────────────────────────────
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(55), // detection
            Constraint::Percentage(45), // supported
        ])
        .split(chunks[0]);

    // Detection results
    let detect_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🔍 Language Detection ");

    let mut detect_lines: Vec<Line> = Vec::new();
    if app.lang_detection_results.is_empty() {
        detect_lines.push(Line::from(Span::styled(
            "  Press Enter to scan workspace.",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        for (file, lang, conf) in &app.lang_detection_results {
            let color = match conf.trim_end_matches('%').parse::<f64>().unwrap_or(0.0) {
                c if c >= 99.0 => ratatui::style::Color::Green,
                c if c >= 90.0 => ratatui::style::Color::Cyan,
                _ => ratatui::style::Color::Yellow,
            };
            detect_lines.push(Line::from(vec![
                Span::styled(format!("  {:<15}", file), Style::default().fg(app.theme.fg)),
                Span::styled(format!("{}  {}", lang, conf), Style::default().fg(color)),
            ]));
        }
    }
    let detect_para = Paragraph::new(detect_lines)
        .block(detect_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(detect_para, left[0]);

    // Supported languages
    let supp_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🌍 Supported Languages ");

    let mut supp_lines: Vec<Line> = Vec::new();
    for (lang, status) in &app.lang_supported {
        let color = match status.as_str() {
            "✅" => ratatui::style::Color::Green,
            "🔄" => ratatui::style::Color::Yellow,
            _ => ratatui::style::Color::Red,
        };
        supp_lines.push(Line::from(vec![
            Span::styled(format!("  {} ", status), Style::default().fg(color)),
            Span::styled(lang, Style::default().fg(app.theme.fg)),
        ]));
    }
    let supp_para = Paragraph::new(supp_lines)
        .block(supp_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(supp_para, left[1]);

    // ── Right: Translation ──────────────────────────────────────────────────
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1), // translate input/output
        ])
        .split(chunks[1]);

    let trans_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🔄 Quick Translate ");

    let mut trans_lines: Vec<Line> = Vec::new();
    trans_lines.push(Line::from(Span::styled(
        format!(
            "  Source: {}  Target: {}",
            app.lang_translate_source, app.lang_translate_target
        ),
        Style::default().fg(app.theme.accent),
    )));
    trans_lines.push(Line::from(""));
    trans_lines.push(Line::from(Span::styled(
        "  Input:  (type and press Enter)",
        Style::default().fg(app.theme.fg),
    )));
    if app.lang_translate_input.is_empty() {
        trans_lines.push(Line::from(Span::styled(
            "    > ",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        trans_lines.push(Line::from(Span::styled(
            format!("    > {}", app.lang_translate_input),
            Style::default().fg(app.theme.fg),
        )));
    }
    if !app.lang_translate_output.is_empty() {
        trans_lines.push(Line::from(""));
        trans_lines.push(Line::from(Span::styled(
            format!("  Output: {}", app.lang_translate_output),
            Style::default().fg(ratatui::style::Color::Green),
        )));
    }
    let trans_para = Paragraph::new(trans_lines)
        .block(trans_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(trans_para, right[0]);
}
