use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Clear, List, ListItem, ListState, Paragraph, Scrollbar,
        ScrollbarOrientation, ScrollbarState, Wrap,
    },
    Frame,
};

use xencode_models_rs::current_timestamp;

use crate::app::App;
use crate::focus::{
    mask_secret, FocusArea, InputMode, SettingKind, FEATURE_LIST, SETTINGS_ITEMS,
    SETTINGS_LABEL_WIDTH,
};
use crate::layout::compute_layout;
use crate::widgets::{gauge, panel_border_set, spinner};

pub fn draw(f: &mut Frame, app: &mut App) {
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
        FocusArea::ReviewDashboard => draw_review_dashboard(f, app, f.area()),
        FocusArea::TaskManager => draw_task_manager(f, app, f.area()),
        FocusArea::WorktreePanel => draw_worktree_panel(f, app, f.area()),
        FocusArea::AdvisePanel => draw_advise_panel(f, app, f.area()),
        _ => {}
    }

    // Project context overlay (driven from chat via /init), always on top.
    if app.init_visible {
        draw_project_init_panel(f, app, f.area());
    }

    // Transient toasts (file-watch warnings) float over the body (E3-03).
    if !app.toasts.is_empty() {
        draw_toasts(f, app, outer[1]);
    }

    // Help overlay is modal; the approval prompt outranks even that.
    if app.help_visible {
        draw_help_overlay(f, app, f.area());
    }
    if app.pending_approval().is_some() {
        draw_approval_overlay(f, app, f.area());
    }
}

/// The agent approval prompt (I1-03): topmost and modal, showing the call,
/// its class, and the exact bytes at stake (diff or command line).
fn draw_approval_overlay(f: &mut Frame, app: &App, area: Rect) {
    let Some(request) = app.pending_approval() else {
        return;
    };
    let popup_area = centered_rect(66, 74, area);
    f.render_widget(Clear, popup_area);

    let queue_note = if app.approval_queue.len() > 1 {
        format!(" · +{} more", app.approval_queue.len() - 1)
    } else {
        String::new()
    };
    let title = format!(" ⚙ Allow {}?{} ", request.class_label(), queue_note);
    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.warning))
        .title(title);

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        request.summary.clone(),
        Style::default()
            .fg(app.theme.fg)
            .add_modifier(Modifier::BOLD),
    )));
    if !request.preview.is_empty() {
        lines.push(Line::from(""));
        for line in request.preview.lines() {
            let style = if line.starts_with('+') {
                Style::default().fg(app.theme.success)
            } else if line.starts_with('-') {
                Style::default().fg(app.theme.danger)
            } else if line.starts_with("@@") {
                Style::default().fg(app.theme.accent)
            } else {
                Style::default().fg(app.theme.message_system)
            };
            lines.push(Line::from(Span::styled(line.to_string(), style)));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "y:allow · a:allow all like this · n/Esc:deny",
        Style::default().fg(app.theme.warning),
    )));

    let max_scroll = clamp_scroll(lines.len(), popup_area.height);
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll(((app.approval_scroll as u16).min(max_scroll), 0));
    f.render_widget(para, popup_area);
}

fn draw_toasts(f: &mut Frame, app: &App, area: Rect) {
    let lines = crate::toast::render_lines(&app.toasts, &app.theme);
    if lines.is_empty() || area.width < 12 || area.height < 3 {
        return;
    }
    let text_w = lines.iter().map(|l| l.width()).max().unwrap_or(0) as u16;
    let width = (text_w + 2).min(area.width.saturating_sub(2));
    // Never overlay the input strip: the toast's last row must stay above
    // the body's bottom three rows, so it shrinks to one line or steps
    // aside entirely on short screens (H1-09).
    let height = (lines.len() as u16 + 2).min(area.height.saturating_sub(5));
    if height < 3 {
        return;
    }
    let popup = Rect {
        x: area.x + area.width.saturating_sub(width + 1),
        y: area.y + 1,
        width,
        height,
    }
    .intersection(area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 📣 ");
    f.render_widget(Clear, popup);
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(para, popup);
}

fn draw_help_overlay(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(70, 85, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" ⌨ Keybindings — Esc or ? to close ");

    let all = crate::help::help_lines(app.focus, &app.theme);
    let max_scroll = clamp_scroll(all.len(), popup_area.height);
    let para = Paragraph::new(all)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((app.help_scroll.min(max_scroll), 0));
    f.render_widget(para, popup_area);
}

// ── Header ──────────────────────────────────────────────────────────────────

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    // Width ladder (H1-08): brand always, layout chip from 72 cols,
    // branch + model from 60, focus badge from 40.
    let w = area.width as usize;
    let mut left = String::from(" ✦ xencode ");
    if w >= 72 {
        left.push_str(&format!(
            "[{}] ",
            crate::layout::effective_layout(&app.config.layout)
        ));
    }
    if w >= 60 {
        left.push_str(&format!("⎇ {} ", app.git_branch));
        left.push_str(&app.config.default_model);
        if app.init_running {
            left.push_str(" ⏳init");
        }
    }
    let right = if w >= 40 {
        format!(" {} ", app.focus.display_name())
    } else {
        String::new()
    };

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
    let hints = if app.pending_approval().is_some() {
        "y:allow  a:allow like this  n/Esc:deny  k/j:scroll diff"
    } else if app.input_mode == InputMode::Editing {
        "Enter:send  Esc:normal  \u{2190}\u{2192}:cursor"
    } else {
        match app.focus {
            FocusArea::Settings => "\u{2191}\u{2193}:nav  \u{2190}\u{2192}:change  Enter:save  Esc:close",
            FocusArea::FileExplorer => "\u{2191}\u{2193}:select  Enter:open  Space:attach  m:models",
            FocusArea::CodeEditor => "e:edit  \u{2191}\u{2193}:scroll  Ctrl+S:save  Tab:next",
            FocusArea::ModelSelector => "\u{2191}\u{2193}:select  Enter:confirm  Esc:close",
            FocusArea::CodeReview => "Enter:review  Esc:close",
            FocusArea::ReviewDashboard => "↑↓:file  u/d:scroll  b:base  Enter:reload  Esc:close",
            FocusArea::TaskManager => {
                if app.tasks_detail {
                    "↑↓:scroll  Enter:back to list  Esc:close"
                } else {
                    "↑↓:select  Enter:output  x:stop  d:remove  Esc:close"
                }
            }
            FocusArea::WorktreePanel => match app.worktree_prompt {
                crate::focus::WorktreePrompt::None =>
                    "a:add  d:remove  r:refresh  Esc:close",
                crate::focus::WorktreePrompt::AddPath =>
                    "path: <type>  Enter:next  Esc:cancel",
                crate::focus::WorktreePrompt::AddBranch =>
                    "branch (Enter=auto): <type>  Enter:run  Esc:cancel",
                crate::focus::WorktreePrompt::ConfirmRemove =>
                    "y:confirm  n/Esc:cancel",
            },
            FocusArea::AdvisePanel => {
                if app.advise_detail {
                    "↑↓:scroll  Enter:back to list  Esc:close"
                } else {
                    "↑↓:select  Enter:detail  o:open file  r:recompute  Esc:close"
                }
            }
            FocusArea::FeatureNavigator => "\u{2191}\u{2193}:nav  Enter:open  Esc:close",
            FocusArea::ByteBotPanel => "Enter:run  Esc:close  Type command above",
            FocusArea::ProviderHealth => "Ctrl+H:run check  Esc:close",
            FocusArea::TerminalAssistant => {
                "type:ask  Enter:ask/run  j/k:select  y:run  f:filter  Esc:close"
            }
            FocusArea::PerformanceDashboard | FocusArea::ProjectAnalyzer |
            FocusArea::GitCommit | FocusArea::CollaborationHub |
            FocusArea::SecurityAuditor | FocusArea::PerformanceProfiler => "Enter:start  Esc:close",
            FocusArea::VoiceInterface => "Enter:record  Enter again:stop  m:mute  Esc:close",
            FocusArea::LearningMode => {
                "Enter:lessons  p/n:file  ←→+Enter:quiz  r:re-ask  Esc:close"
            }
            FocusArea::CustomModels => {
                "n:new  -/+:temp  ←/→:tokens  Enter:apply  s:save  t:test  Esc:close"
            }
            FocusArea::MultiLanguage => {
                "Enter:detect  Tab:pick a field  type:edit  Enter:translate  Esc:close"
            }
            _ => "i:edit  m:models  s:settings  ?:help  Tab:switch  Ctrl+R:review  Ctrl+Y:pr-review  Ctrl+T:terminal  Ctrl+B:bytebot  Ctrl+D:dashboard  Ctrl+P:analyzer",
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

// ── Body (explorer + editor + chat + input) ────────────────────────────────

/// Renders the body from the layout engine's resolved rects (H1-03). The
/// geometry lives in exactly one place (`layout::compute_layout`); the mouse
/// handler and resize clamp call the same function. `last_layout` is
/// recorded here so the Tab focus-ring can see what the user can see, and
/// `last_body_focus` is latched for zen's focus-follows target.
fn draw_body(f: &mut Frame, app: &mut App, area: Rect) {
    if matches!(
        app.focus,
        FocusArea::FileExplorer | FocusArea::CodeEditor | FocusArea::ChatInput
    ) {
        app.last_body_focus = app.focus;
    }
    let layout = compute_layout(
        area,
        &app.config.layout,
        app.show_terminal,
        app.last_body_focus,
    );
    app.last_layout = layout;

    if let Some(rect) = layout.explorer {
        draw_file_explorer(f, app, rect);
    }
    if let Some(rect) = layout.editor {
        draw_code_editor(f, app, rect);
    }
    if let Some(rect) = layout.chat {
        draw_messages(f, app, rect);
    }
    if let Some(rect) = layout.terminal {
        draw_terminal(f, app, rect);
    }
    if let Some(rect) = layout.input {
        draw_input(f, app, rect);
    }
}

/// How a pane's border should read, in priority order.
#[derive(Clone, Copy)]
enum PaneState {
    /// The user is typing in this pane right now.
    Editing,
    /// Keyboard focus lives here, normal mode.
    Focused,
    /// Always-live pane (e.g. the terminal strip).
    Active,
    /// Unfocused.
    Plain,
}

/// The framed panel block every body pane builds (H1-06): one place that
/// applies the rounded-borders preference and the focus border convention.
fn panel_block(app: &App, title: String, state: PaneState) -> Block<'static> {
    let border = match state {
        PaneState::Editing => Style::default().fg(app.theme.accent),
        PaneState::Focused | PaneState::Active => Style::default().fg(app.theme.border_active),
        PaneState::Plain => Style::default().fg(app.theme.border),
    };
    Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(border)
        .title(title)
}

fn draw_code_editor(f: &mut Frame, app: &mut App, area: Rect) {
    let is_focused = app.focus == FocusArea::CodeEditor;
    let is_editing = is_focused && app.input_mode == InputMode::Editing;

    // The width gate for the gutter lives here and only here: below ~45
    // columns the numbers would eat too much of the code area.
    let gutter = app.config.show_line_numbers && area.width >= 45;
    if gutter {
        if app.editor.line_number_style().is_none() {
            app.editor
                .set_line_number_style(Style::default().fg(ratatui::style::Color::DarkGray));
        }
    } else {
        app.editor.remove_line_number();
    }
    app.editor.set_cursor_line_style(if is_editing {
        Style::default().add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    });

    let dirty_mark = if app.editor_dirty { " [modified]" } else { "" };
    let mode_mark = if is_editing { " EDITING" } else { "" };
    // Below ~30 columns the emoji costs more than it tells.
    let icon = if area.width < 30 { "" } else { " 📝" };

    let title = if let Some(ref fp) = app.opened_file {
        format!("{icon} {fp}{dirty_mark}{mode_mark} ")
    } else {
        format!("{icon} Code Editor (Select a file & press Enter) ")
    };

    let block = panel_block(
        app,
        title,
        if is_editing {
            PaneState::Editing
        } else if is_focused {
            PaneState::Focused
        } else {
            PaneState::Plain
        },
    );

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

    let title = if area.width < 30 {
        format!(" Files ({} files) ", app.file_tree.len())
    } else {
        format!(" 📁 Workspace ({} files) ", app.file_tree.len())
    };
    let block = panel_block(
        app,
        title,
        if is_focused {
            PaneState::Focused
        } else {
            PaneState::Plain
        },
    );

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
                    "D" => app.theme.danger,                   // Deleted
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

    let show_bar = app.config.show_scrollbars && area.width >= 24;
    let (list_area, bar_area) = if show_bar {
        let c = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(area);
        (c[0], Some(c[1]))
    } else {
        (area, None)
    };

    let list = List::new(items).block(block);
    let mut state = ListState::default();
    state.select(Some(app.selected_file));
    f.render_stateful_widget(list, list_area, &mut state);

    if let Some(bar) = bar_area {
        let mut sb_state = ScrollbarState::new(app.file_tree.len()).position(state.offset());
        f.render_stateful_widget(scrollbar_bar(app), bar, &mut sb_state);
    }
}

/// The shared vertical scrollbar style for body panes (H1-07): accent
/// thumb, no arrow heads — the arrow glyphs would dominate a 1-column bar.
fn scrollbar_bar(app: &App) -> Scrollbar<'static> {
    Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(None)
        .end_symbol(None)
        .thumb_style(Style::default().fg(app.theme.accent))
        .track_style(Style::default().fg(app.theme.border))
}

// ── Chat Messages ───────────────────────────────────────────────────────────

/// The chat transcript as rendered lines. Shared by `draw_messages` and the
/// resize clamp so both agree on the content length.
fn chat_lines(app: &App) -> Vec<Line<'static>> {
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

        if msg.role == "assistant" {
            text.extend(crate::markdown::render_markdown(&msg.content, &app.theme));
        } else {
            for line in msg.content.lines() {
                text.push(Line::from(Span::styled(
                    format!("  {}", line),
                    Style::default().fg(app.theme.fg),
                )));
            }
        }
        text.push(Line::from(""));
    }

    // Spinner at bottom when generating
    if app.is_generating {
        let frame = spinner::frame(app.spinner_tick);
        text.push(Line::from(Span::styled(
            format!("  {} Generating...", frame),
            Style::default().fg(app.theme.accent),
        )));
    }
    text
}

/// Rows the agent's plan strip takes off the top of the chat pane — 0 when
/// there is no plan, or when the pane is too short to give any up. Shared by
/// `draw_messages` and the resize clamp so the transcript's scroll math sees
/// the geometry the user does (E6-02 discipline).
fn plan_block_height(app: &App, area: Rect) -> u16 {
    const MIN_TRANSCRIPT_ROWS: u16 = 6;
    let items = crate::agent_tools::plan_items(&app.agent_plan);
    if items.is_empty() {
        return 0;
    }
    let shown = plan_visible_items(app, items.len());
    let rows = shown + usize::from(items.len() > shown);
    let height = (rows + 2).min(u16::MAX as usize) as u16; // + borders
    if area.height < height + MIN_TRANSCRIPT_ROWS {
        return 0;
    }
    height
}

/// How many steps the strip renders before pointing at `/plan`.
fn plan_visible_items(app: &App, total: usize) -> usize {
    if app.plan_pinned {
        total
    } else {
        total.min(crate::agent_tools::PLAN_COMPACT_ITEMS)
    }
}

fn draw_plan_strip(f: &mut Frame, app: &App, area: Rect) {
    let items = crate::agent_tools::plan_items(&app.agent_plan);
    if items.is_empty() || area.height < 3 {
        return;
    }
    let shown = plan_visible_items(app, items.len());
    let done = items
        .iter()
        .filter(|item| item.status == crate::agent_tools::PlanStatus::Done)
        .count();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_set(panel_border_set(app.config.rounded_borders))
        .border_style(Style::default().fg(app.theme.border))
        .title(format!(" ☰ Plan {done}/{} ", items.len()));
    // One line per step, clipped to the pane: a wrapped step would push the
    // next one out of a block whose height was computed without wrapping.
    let width = (area.width as usize).saturating_sub(4);
    let mut lines: Vec<Line<'static>> = items
        .iter()
        .take(shown)
        .map(|item| {
            let (glyph, style) = match item.status {
                crate::agent_tools::PlanStatus::Done => (
                    "✓",
                    Style::default()
                        .fg(app.theme.message_system)
                        .add_modifier(Modifier::CROSSED_OUT),
                ),
                crate::agent_tools::PlanStatus::InProgress => (
                    "▶",
                    Style::default()
                        .fg(app.theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                crate::agent_tools::PlanStatus::Pending => ("·", Style::default().fg(app.theme.fg)),
            };
            Line::from(vec![
                Span::styled(format!("{glyph} "), style),
                Span::styled(
                    crate::agent_tools::truncate_one_line(&item.text, width),
                    style,
                ),
            ])
        })
        .collect();
    if items.len() > shown {
        lines.push(Line::from(Span::styled(
            format!("  … {} more — /plan", items.len() - shown),
            Style::default().fg(app.theme.message_system),
        )));
    }
    f.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_messages(f: &mut Frame, app: &App, area: Rect) {
    let text = chat_lines(app);

    let is_focused = app.focus == FocusArea::ChatInput && app.input_mode == InputMode::Normal;

    // The agent's todo list sits above the transcript, outside the scroll: a
    // plan you have to scroll up to find is not a plan (I2-03).
    let plan_rows = plan_block_height(app, area);
    let area = if plan_rows == 0 {
        area
    } else {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(plan_rows), Constraint::Min(3)])
            .split(area);
        draw_plan_strip(f, app, split[0]);
        split[1]
    };

    let block = panel_block(
        app,
        if area.width < 30 {
            " Chat ".to_string()
        } else {
            " 💬 Chat ".to_string()
        },
        if is_focused {
            PaneState::Focused
        } else {
            PaneState::Plain
        },
    );

    let text_lines = text.len() as u16;
    let height = area.height.saturating_sub(2);
    let max_scroll = text_lines.saturating_sub(height);
    let scroll = max_scroll.saturating_sub(app.chat_scroll);

    let show_bar = app.config.show_scrollbars && area.width >= 24;
    let (text_area, bar_area) = if show_bar {
        let c = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(area);
        (c[0], Some(c[1]))
    } else {
        (area, None)
    };

    let paragraph = Paragraph::new(text)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

    f.render_widget(paragraph, text_area);

    if let Some(bar) = bar_area {
        // Same logical-line model the paragraph scroll uses above.
        let mut sb_state = ScrollbarState::new(text_lines as usize).position(scroll as usize);
        f.render_stateful_widget(scrollbar_bar(app), bar, &mut sb_state);
    }
}

// ── Input ───────────────────────────────────────────────────────────────────

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let is_editing = app.input_mode == InputMode::Editing;

    let title = if app.is_generating {
        let frame = spinner::frame(app.spinner_tick);
        format!(" {} Thinking... ", frame)
    } else if is_editing {
        if area.width < 30 {
            " Enter:send · Alt+Enter: newline ".to_string()
        } else {
            " ✏️  Enter to send · Alt+Enter (or Ctrl+J) for newline ".to_string()
        }
    } else {
        " Press 'i' to start typing ".to_string()
    };

    let block = panel_block(
        app,
        title,
        if is_editing {
            PaneState::Editing
        } else if app.focus == FocusArea::ChatInput {
            PaneState::Focused
        } else {
            PaneState::Plain
        },
    );

    if app.is_generating {
        let paragraph = Paragraph::new("Please wait...")
            .block(block)
            .style(Style::default().fg(app.theme.fg));
        f.render_widget(paragraph, area);
    } else {
        let inner = block.inner(area);
        f.render_widget(block, area);
        f.render_widget(&app.chat_input, inner);

        // Show cursor (position is scroll-adjusted by the textarea).
        if is_editing {
            let (row, col) = app.chat_input.cursor();
            let x = inner.x + col as u16;
            let y = inner.y + row as u16;
            if x < inner.x + inner.width && y < inner.y + inner.height {
                f.set_cursor_position((x, y));
            }
        }
    }
}

// ── Terminal Pane ───────────────────────────────────────────────────────────

fn draw_terminal(f: &mut Frame, app: &App, area: Rect) {
    let block = panel_block(
        app,
        if area.width < 30 {
            " Terminal (Ctrl+T) ".to_string()
        } else {
            " 🖥️  Terminal (Ctrl+T to toggle) ".to_string()
        },
        PaneState::Active,
    );

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
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🧠 Select Model (↑↓ Enter · 'r' Refresh · 'l' Load · 'u' Unload · Esc Close) ");

    let items: Vec<ListItem> = if app.available_models.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "   ⚠️  No models detected. Ensure Ollama is running, or press 'r' to refresh.",
            Style::default().fg(app.theme.warning),
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
                        (" [llamacpp]", app.theme.warning)
                    } else if model.contains('/') || model.starts_with("qwen-") {
                        (" [cloud]", app.theme.accent_secondary)
                    } else {
                        (" [ollama]", app.theme.info)
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

/// The right-hand value column for one settings row, formatted per its
/// `SettingKind` (and label, for the rows that carry units or types).
fn setting_display(app: &App, idx: usize) -> String {
    use crate::focus::SettingKind;
    let row = &SETTINGS_ITEMS[idx];
    let editing_here = app.settings_url_editing && app.settings_cursor == idx;
    match &row.kind {
        SettingKind::Cycle(options) => {
            let current = match row.label {
                "Theme" => &app.config.active_theme,
                "Layout" => &app.config.layout,
                "Agent Approval" => &app.config.agent_approval,
                _ => options[0],
            };
            let pos = options.iter().position(|o| *o == current).unwrap_or(0);
            let dots: Vec<&str> = (0..options.len())
                .map(|i| if i == pos { "●" } else { "○" })
                .collect();
            format!("{}  [{}]", current, dots.join(" "))
        }
        SettingKind::Toggle => {
            let on = match row.label {
                "Rounded Borders" => app.config.rounded_borders,
                "Show Scrollbars" => app.config.show_scrollbars,
                "Line Numbers" => app.config.show_line_numbers,
                "Cache Enabled" => app.config.cache_enabled,
                "Memory Enabled" => app.config.memory_enabled,
                _ => false,
            };
            if on {
                "✅ Enabled".to_string()
            } else {
                "❌ Disabled".to_string()
            }
        }
        SettingKind::Stepped { .. } => match row.label {
            "Max Cache Size" => format!("{} entries", app.config.max_cache_size),
            "Memory Items" => format!("{} entries", app.config.max_memory_items),
            "Response Timeout" => format!("{}s", app.config.response_timeout),
            "Command Timeout" => format!("{}s", app.config.agent_command_timeout),
            _ => String::new(),
        },
        SettingKind::Text => {
            let value = match row.label {
                "Ollama URL" => &app.config.ollama_url,
                "Llama.cpp URL" => &app.config.llama_cpp_url,
                "Llama.cpp Model" => &app.config.llama_cpp_model_path,
                "Remote URL" => &app.config.remote_base_url,
                _ => &app.config.ollama_url,
            };
            if editing_here {
                format!(
                    "{}| (type to edit, Enter to confirm)",
                    &app.settings_url_buffer[..app.settings_url_cursor]
                )
            } else if value.is_empty() {
                "⚠️  Not set (Enter to edit)".to_string()
            } else {
                format!("{}  (Enter to edit)", value)
            }
        }
        SettingKind::Secret => {
            if editing_here {
                // Bullets, not characters: the key must never reach the screen.
                format!(
                    "{}| (Enter to save, Esc to cancel)",
                    "•".repeat(app.settings_url_buffer.chars().count())
                )
            } else {
                match crate::app::secret_value(&app.config, row.label) {
                    Some(key) => format!("{}  (Enter to edit)", mask_secret(key)),
                    None => "⚠️  Not set (Enter to edit)".to_string(),
                }
            }
        }
        SettingKind::Number => {
            let shown = match row.label {
                "Llama Temp" => app.config.llama_cpp_temperature.map(|v| v.to_string()),
                "Llama Top-K" => app.config.llama_cpp_top_k.map(|v| v.to_string()),
                "Llama Min-P" => app.config.llama_cpp_min_p.map(|v| v.to_string()),
                "Llama Max Tokens" => app.config.llama_cpp_max_tokens.map(|v| v.to_string()),
                _ => None,
            };
            if editing_here {
                format!(
                    "{}| (Enter to confirm)",
                    &app.settings_url_buffer[..app.settings_url_cursor]
                )
            } else {
                shown.unwrap_or_else(|| "default".to_string())
            }
        }
        SettingKind::Action => {
            if app.settings_reset_active {
                "✅ Reset to defaults!".to_string()
            } else {
                "⚠️  Reset to defaults (Enter to confirm)".to_string()
            }
        }
    }
}

fn draw_settings(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(65, 80, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
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
    // One line per row of `focus::SETTINGS_ITEMS`: sections, values and
    // colors all derive from the table, never from row numbers (H1-04).
    let width = SETTINGS_LABEL_WIDTH;
    let mut settings_lines: Vec<Line> = Vec::new();
    let mut prev_section: Option<&str> = None;
    for (idx, row) in SETTINGS_ITEMS.iter().enumerate() {
        if prev_section != Some(row.section) {
            if prev_section.is_some() {
                settings_lines.push(Line::from(""));
            }
            settings_lines.push(Line::from(Span::styled(
                format!("  {}", row.section),
                Style::default()
                    .fg(app.theme.fg)
                    .add_modifier(Modifier::UNDERLINED),
            )));
            prev_section = Some(row.section);
        }

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
        let value_color = if row.kind == SettingKind::Action {
            if is_selected {
                app.theme.danger
            } else {
                app.theme.message_system
            }
        } else if matches!(row.kind, SettingKind::Text | SettingKind::Number)
            && app.settings_url_editing
            && is_selected
        {
            app.theme.warning
        } else {
            app.theme.accent
        };
        settings_lines.push(Line::from(vec![
            Span::styled(format!("{pointer}{:<width$}", row.label), style),
            Span::styled(setting_display(app, idx), Style::default().fg(value_color)),
        ]));
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
            app.theme.success
        } else {
            app.theme.danger
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

/// The code-review popup body text; shared with the resize clamp.
fn review_display_text(app: &App) -> String {
    if !app.code_review_output.is_empty() {
        app.code_review_output.clone()
    } else if let Some(fp) = app.file_tree.get(app.selected_file) {
        format!(
            "\n  Selected file: {}\n\n  Press Enter to start AI code review.",
            fp
        )
    } else {
        "  Select a file in the File Explorer first.".to_string()
    }
}

fn draw_code_review(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(80, 80, area);
    f.render_widget(Clear, popup_area);

    let title = if app.is_reviewing {
        let frame = spinner::frame(app.spinner_tick);
        format!(" {} Code Review (analyzing...) ", frame)
    } else {
        " 🔍 Code Review (Enter to review, Esc to close) ".to_string()
    };

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);

    let review_text = review_display_text(app);
    let review_rows = review_text.lines().count();
    let text = Paragraph::new(review_text)
        .block(block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false })
        .scroll((
            app.review_scroll
                .min(clamp_scroll(review_rows, popup_area.height)),
            0,
        ));
    f.render_widget(text, popup_area);
}

fn draw_review_dashboard(f: &mut Frame, app: &App, area: Rect) {
    use crate::review::format_review_file_line;

    let popup_area = centered_rect(90, 85, area);
    f.render_widget(Clear, popup_area);

    let dash = &app.review_dash;
    let title = format!(
        " 🔍 PR Review — {}...HEAD ({} files) ",
        dash.base,
        dash.files.len()
    );
    let outer = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);
    f.render_widget(outer, popup_area);
    let inner = popup_area.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    if inner.width < 3 || inner.height < 2 {
        return;
    }

    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(inner);

    // Left: changed-file list with the cursor row highlighted.
    let file_lines: Vec<Line> = if dash.files.is_empty() {
        vec![Line::from("  (no changes)")]
    } else {
        dash.files
            .iter()
            .enumerate()
            .map(|(i, file)| {
                let marker = if i == dash.selected { "> " } else { "  " };
                let row = format!("{}{}", marker, format_review_file_line(file));
                if i == dash.selected {
                    Line::from(Span::styled(
                        row,
                        Style::default()
                            .fg(app.theme.highlight_fg)
                            .bg(app.theme.highlight)
                            .add_modifier(Modifier::BOLD),
                    ))
                } else {
                    Line::from(row)
                }
            })
            .collect()
    };
    let file_list = Paragraph::new(file_lines)
        .block(
            Block::default()
                .borders(Borders::RIGHT)
                .border_style(Style::default().fg(app.theme.border))
                .title(" Files "),
        )
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(file_list, panes[0]);

    // Right: unified diff of the selected file (or error/empty state).
    let body = if let Some(err) = &dash.error {
        format!("Error: {err}")
    } else if dash.diff_text.is_empty() {
        if dash.files.is_empty() {
            "No changes for this base.\n\nPress b to switch base (HEAD <-> main).".to_string()
        } else {
            "(empty diff)".to_string()
        }
    } else {
        dash.diff_text.clone()
    };
    let diff = Paragraph::new(body)
        .block(
            Block::default()
                .border_style(Style::default().fg(app.theme.border))
                .title(" Diff "),
        )
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false })
        .scroll((dash.scroll, 0));
    f.render_widget(diff, panes[1]);
}

/// Detail-pane body for the selected task. Shared by the renderer and the
/// resize clamp so they count the same lines (E6-02 discipline).
fn task_detail_text(tasks: &[xencode_core_rs::TaskRecord], selected: usize) -> String {
    match tasks.get(selected) {
        None => "(no task selected)".to_string(),
        Some(rec) if rec.output().is_empty() => format!(
            "{} — no output yet (still {})",
            rec.command,
            rec.status.label()
        ),
        Some(rec) => rec.output().join("\n"),
    }
}

/// Inner rows of the task panel popup at terminal height `area_height`.
fn task_panel_inner_height(area_height: u16) -> u16 {
    centered_rect(85, 75, Rect::new(0, 0, 1, area_height))
        .inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        })
        .height
}

fn draw_task_manager(f: &mut Frame, app: &App, area: Rect) {
    use xencode_core_rs::TaskStatus;

    let popup_area = centered_rect(85, 75, area);
    f.render_widget(Clear, popup_area);

    // Registry read without blocking; `None` only during a tool call, and
    // an empty list is the honest fallback for that one frame.
    let tasks = app.tasks_snapshot().unwrap_or_default();
    let running = tasks
        .iter()
        .filter(|t| matches!(t.status, TaskStatus::Running))
        .count();
    let outer = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(format!(
            " ⏳ Background Tasks — {running} running / {} total ",
            tasks.len()
        ));
    f.render_widget(outer, popup_area);
    let inner = popup_area.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    if inner.width < 3 || inner.height < 2 {
        return;
    }

    if app.tasks_detail {
        let body = task_detail_text(&tasks, app.tasks_selected);
        let rows = body.lines().count();
        let text = Paragraph::new(body)
            .style(Style::default().fg(app.theme.fg))
            .wrap(Wrap { trim: false })
            .scroll((
                (app.tasks_scroll as u16).min(clamp_scroll(rows, inner.height)),
                0,
            ));
        f.render_widget(text, inner);
        return;
    }

    let rows: Vec<Line> = if tasks.is_empty() {
        vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No background tasks yet.",
                Style::default().fg(app.theme.fg),
            )),
            Line::from(Span::styled(
                "  Ask the model to run a command in the background",
                Style::default().fg(app.theme.border),
            )),
            Line::from(Span::styled(
                "  (background_start) — it shows up here.",
                Style::default().fg(app.theme.border),
            )),
        ]
    } else {
        let selected = app.tasks_selected.min(tasks.len() - 1);
        tasks
            .iter()
            .enumerate()
            .map(|(i, rec)| task_line(rec, i == selected, app))
            .collect()
    };
    let list = Paragraph::new(rows).style(Style::default().fg(app.theme.fg));
    f.render_widget(list, inner);
}

fn task_line(
    rec: &xencode_core_rs::TaskRecord,
    selected: bool,
    app: &App,
) -> ratatui::text::Line<'static> {
    use xencode_core_rs::TaskStatus;
    let (icon, color) = match &rec.status {
        TaskStatus::Running => ("▶", app.theme.info),
        TaskStatus::Exited(0) => ("✓", app.theme.success),
        TaskStatus::Exited(_) => ("✗", app.theme.danger),
        TaskStatus::Killed => ("⊘", app.theme.warning),
    };
    let marker = if selected { "> " } else { "  " };
    let row = format!(
        "{marker}{icon} #{:<3} {:<11} {} — {}",
        rec.id,
        rec.status.label(),
        rec.name,
        rec.command.replace('\n', " "),
    );
    if selected {
        ratatui::text::Line::from(Span::styled(
            row,
            Style::default()
                .fg(app.theme.highlight_fg)
                .bg(app.theme.highlight)
                .add_modifier(Modifier::BOLD),
        ))
    } else {
        ratatui::text::Line::from(Span::styled(row, Style::default().fg(color)))
    }
}

fn draw_worktree_panel(f: &mut Frame, app: &App, area: Rect) {
    use crate::focus::WorktreePrompt;

    let popup_area = centered_rect(80, 65, area);
    f.render_widget(Clear, popup_area);

    let outer = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(format!(" 🌳 Worktrees — {} ", app.worktrees.len()));
    f.render_widget(outer, popup_area);
    let inner = popup_area.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    if inner.width < 3 || inner.height < 2 {
        return;
    }

    // List rows (prompt mode keeps showing them, dimmed) + action lines.
    let mut rows: Vec<Line> = Vec::new();
    if app.worktrees.is_empty() {
        rows.push(Line::from(Span::styled(
            if app.worktree_status.is_empty() {
                "  No worktrees found — this is the only checkout."
            } else {
                app.worktree_status.as_str()
            },
            Style::default().fg(app.theme.fg),
        )));
    } else {
        let selected = app.worktree_selected.min(app.worktrees.len() - 1);
        for (i, wt) in app.worktrees.iter().enumerate() {
            let dirty = app.worktree_dirty.get(i).copied().unwrap_or(false);
            let (icon, color) = if wt.is_main {
                ("★", app.theme.accent)
            } else if dirty {
                ("⚡", app.theme.warning)
            } else {
                ("◯", app.theme.success)
            };
            let marker = if i == selected { "> " } else { "  " };
            let text = format!(
                "{marker}{icon} {:<10} {:<24} {}",
                wt.display_branch(),
                wt.short_head(),
                wt.path.display()
            );
            if i == selected {
                rows.push(Line::from(Span::styled(
                    text,
                    Style::default()
                        .fg(app.theme.highlight_fg)
                        .bg(app.theme.highlight)
                        .add_modifier(Modifier::BOLD),
                )));
            } else {
                rows.push(Line::from(Span::styled(text, Style::default().fg(color))));
            }
        }
    }

    let mut lines: Vec<Line> = vec![Line::from(Span::styled(
        "  legend: ★ main (never removable)  ⚡ dirty  ◯ clean",
        Style::default().fg(app.theme.border),
    ))];
    lines.extend(rows);
    lines.push(Line::from(""));

    let prompt_line = match app.worktree_prompt {
        WorktreePrompt::None => None,
        WorktreePrompt::AddPath => Some(format!("  new path: {}▏", app.worktree_path_buf)),
        WorktreePrompt::AddBranch => Some(format!(
            "  branch ({}): {}▏",
            if app.worktree_path_buf.is_empty() {
                "?"
            } else {
                app.worktree_path_buf.as_str()
            },
            app.worktree_branch_buf
        )),
        WorktreePrompt::ConfirmRemove => Some(match app.worktrees.get(app.worktree_selected) {
            Some(wt) => format!(
                "  remove {} ? (git refuses dirty worktrees) y/n",
                wt.path.display()
            ),
            None => "  nothing to remove".to_string(),
        }),
    };
    if let Some(p) = prompt_line {
        lines.push(Line::from(Span::styled(
            p,
            Style::default()
                .fg(app.theme.message_assistant)
                .add_modifier(Modifier::BOLD),
        )));
    } else if !app.worktree_status.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("  {}", app.worktree_status),
            Style::default().fg(app.theme.warning),
        )));
    }

    let text = Paragraph::new(lines).style(Style::default().fg(app.theme.fg));
    f.render_widget(text, inner);
}

/// AdvisePanel (F2-01): the stateful `List` gives the (potentially long)
/// findings list a viewport that follows the selection for free; only the
/// wrapped detail body needs manual scrolling.
fn draw_advise_panel(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(85, 70, area);
    f.render_widget(Clear, popup_area);

    let outer = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(format!(
            " 💡 Insights — {} finding(s) ",
            app.advise_items.len()
        ));
    f.render_widget(outer, popup_area);
    let inner = popup_area.inner(ratatui::layout::Margin {
        horizontal: 1,
        vertical: 1,
    });
    if inner.width < 3 || inner.height < 2 {
        return;
    }

    if app.advise_detail {
        let body = advise_detail_text(app);
        let rows = body.lines().count();
        let text = Paragraph::new(body)
            .style(Style::default().fg(app.theme.fg))
            .wrap(Wrap { trim: false })
            .scroll((
                (app.advise_scroll as u16).min(clamp_scroll(rows, inner.height)),
                0,
            ));
        f.render_widget(text, inner);
        return;
    }

    if app.advise_items.is_empty() {
        let why = if app.advise_status.is_empty() {
            "Nothing to flag — the symbol graph is clean."
        } else {
            app.advise_status.as_str()
        };
        let text = Paragraph::new(why).style(Style::default().fg(app.theme.fg));
        f.render_widget(text, inner);
        return;
    }

    let items: Vec<ListItem> = app
        .advise_items
        .iter()
        .map(|a| {
            let (icon, color) = advise_kind_style(a.kind, app);
            ListItem::new(Line::from(Span::styled(
                format!("{icon} {}", a.message),
                Style::default().fg(color),
            )))
        })
        .collect();
    let list = List::new(items).highlight_style(
        Style::default()
            .fg(app.theme.highlight_fg)
            .bg(app.theme.highlight)
            .add_modifier(Modifier::BOLD),
    );
    let mut state = ListState::default();
    state.select(Some(app.advise_selected.min(app.advise_items.len() - 1)));
    f.render_stateful_widget(list, inner, &mut state);
}

fn advise_kind_style(
    kind: xencode_context_rs::AdviceKind,
    app: &App,
) -> (&'static str, ratatui::style::Color) {
    use xencode_context_rs::AdviceKind;
    match kind {
        AdviceKind::BrokenImport => ("⚠", app.theme.danger),
        AdviceKind::Cycle => ("🔁", app.theme.warning),
        AdviceKind::Hub => ("🧶", app.theme.info),
        AdviceKind::Orphan => ("🕸", app.theme.border),
        AdviceKind::AffectedDependent => ("↳", app.theme.success),
    }
}

fn advise_detail_text(app: &App) -> String {
    let Some(a) = app.advise_items.get(app.advise_selected) else {
        return "Nothing selected — Esc back to the list.".to_string();
    };
    let (icon, _) = advise_kind_style(a.kind, app);
    format!(
        "{icon} {}\nkind: {:?}\nfile: {}\n\nPress o to open the file in the editor, Enter/Esc back to the list.",
        a.message, a.kind, a.file
    )
}

// ── Phase 9 Overlays ────────────────────────────────────────────────────────

fn draw_performance_dashboard(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(70, 60, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
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
    let bar = gauge::bar;

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
            " System Utilization (last profiler run)",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )),
        util_line("CPU", app.profiler_gauge_cpu, 100.0, "%"),
        util_line(
            "Memory",
            app.profiler_gauge_mem,
            app.profiler_gauge_mem_total.unwrap_or(0.0),
            " MB",
        ),
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

/// The provider-health popup content; shared with the resize clamp.
fn provider_health_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();

    // Helper: format a health status line with color
    let provider_name = |provider: &str| -> String {
        match provider {
            "ollama" => format!("{} Ollama", "\u{1F916}"),
            "llamacpp" => format!("{} llama.cpp", "\u{1F999}"),
            "openrouter" => format!("{} OpenRouter", "\u{1F310}"),
            "remote" => format!("{} Remote", "\u{1F30D}"),
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
            "healthy" => app.theme.success,
            "error" | "unavailable" => app.theme.danger,
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
                    Style::default().fg(app.theme.warning),
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    "        ⚡ No generation stats yet",
                    Style::default().fg(app.theme.message_system),
                )));
            }
        }
        if provider == "remote" && !app.config.remote_base_url.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("        \u{1F504} {}", app.config.remote_base_url),
                Style::default().fg(app.theme.message_system),
            )));
        }
        if !error_str.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("        {}", error_str),
                Style::default().fg(app.theme.danger),
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
        "   Remote URI:    {}",
        if app.config.remote_base_url.is_empty() {
            "(not configured)".to_string()
        } else {
            app.config.remote_base_url.clone()
        }
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
        let frame = spinner::frame(app.spinner_tick);
        lines.push(Line::from(Span::styled(
            format!(" {} Checking provider status...", frame),
            Style::default().fg(app.theme.accent),
        )));
    }
    lines
}

fn draw_provider_health(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(70, 55, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🏥 Provider Health (Esc to close · h to refresh) ");

    let lines = provider_health_lines(app);
    let lines_len = lines.len();
    let para = Paragraph::new(lines)
        .block(block)
        .style(Style::default().fg(app.theme.fg))
        .scroll((
            app.provider_health_scroll
                .min(clamp_scroll(lines_len, popup_area.height)),
            0,
        ));
    f.render_widget(para, popup_area);
}

fn draw_project_analyzer(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(60, 60, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
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
        .border_set(panel_border_set(app.config.rounded_borders))
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

/// Max useful vertical scroll (in lines) for `rows` of content inside a
/// bordered area of height `area_height`. Scrolling past this renders blank
/// space, so render sites clamp the stored offset with it (E2-04).
fn clamp_scroll(rows: usize, area_height: u16) -> u16 {
    let visible = (area_height as usize).saturating_sub(2); // borders
    rows.saturating_sub(visible).min(u16::MAX as usize) as u16
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
    let raw = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1];

    // On tiny terminals the percentage collapses to a 1-2 cell box where
    // borders eat all the content. Such popups get at least 80 % coverage
    // (H1-09); usable-sized ones keep their designed percentage.
    let floor = |dim: u16| ((dim as u32 * 80) / 100) as u16;
    let width = if raw.width < 16 {
        raw.width.max(floor(r.width)).min(r.width)
    } else {
        raw.width
    };
    let height = if raw.height < 6 {
        raw.height.max(floor(r.height)).min(r.height)
    } else {
        raw.height
    };
    Rect {
        x: r.x + (r.width - width) / 2,
        y: r.y + (r.height - height) / 2,
        width,
        height,
    }
    .intersection(r)
}

/// Re-clamp every stored scroll offset against the new terminal size.
/// Panels already clamp at render time, but the *stored* values would stay
/// oversized and re-expose blank scroll-past-the-end when the terminal grows
/// back. Geometry mirrors the draw functions exactly (E6-02).
pub fn clamp_scrolls_on_resize(app: &mut App, width: u16, height: u16) {
    let area = Rect::new(0, 0, width, height);

    // Chat pane: outer [header 1 | body | status 1], then the layout
    // engine's chat rect — the exact one draw_body renders (H1-03).
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(area)[1];
    let layout = compute_layout(
        body,
        &app.config.layout,
        app.show_terminal,
        app.last_body_focus,
    );
    if let Some(chat_area) = layout.chat {
        let rows = chat_area
            .height
            .saturating_sub(plan_block_height(app, chat_area));
        app.chat_scroll = app
            .chat_scroll
            .min(clamp_scroll(chat_lines(app).len(), rows));
    }

    // Code Review popup.
    let review_area = centered_rect(80, 80, area);
    let review_rows = review_display_text(app).lines().count();
    app.review_scroll = app
        .review_scroll
        .min(clamp_scroll(review_rows, review_area.height));

    // Provider Health popup.
    let health_area = centered_rect(70, 55, area);
    app.provider_health_scroll = app.provider_health_scroll.min(clamp_scroll(
        provider_health_lines(app).len(),
        health_area.height,
    ));

    // Security Auditor findings list (inner column, below the summary cards).
    let sec_inner = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .inner(centered_rect(75, 70, area));
    let sec_list = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(1)])
        .split(sec_inner)[1];
    app.security_scroll = app.security_scroll.min(clamp_scroll(
        security_findings_lines(app).len(),
        sec_list.height,
    ));

    // Help overlay.
    let help_area = centered_rect(70, 85, area);
    app.help_scroll = app.help_scroll.min(clamp_scroll(
        crate::help::help_lines(app.focus, &app.theme).len(),
        help_area.height,
    ));

    // Approval prompt (I1-03): summary + preview + footer, the same row
    // count the overlay builds.
    let approval_rows = app
        .pending_approval()
        .map(|request| match &request.preview {
            preview if preview.is_empty() => 3,
            preview => 3 + preview.lines().count(),
        });
    if let Some(rows) = approval_rows {
        app.approval_scroll = app
            .approval_scroll
            .min(clamp_scroll(rows, centered_rect(66, 74, area).height) as usize);
    }

    // Background Tasks detail pane (only scrolled state the panel keeps).
    if app.tasks_detail {
        let tasks = app.tasks_snapshot().unwrap_or_default();
        let selected = app.tasks_selected.min(tasks.len().saturating_sub(1));
        let body = task_detail_text(&tasks, selected);
        app.tasks_scroll = app
            .tasks_scroll
            .min(clamp_scroll(body.lines().count(), task_panel_inner_height(height)) as usize);
    }

    // Insights detail pane (the findings list viewport follows the
    // selection on its own via the stateful List).
    if app.advise_detail {
        let inner = centered_rect(85, 70, area).inner(ratatui::layout::Margin {
            horizontal: 1,
            vertical: 1,
        });
        let rows = advise_detail_text(app).lines().count();
        app.advise_scroll = app
            .advise_scroll
            .min(clamp_scroll(rows, inner.height) as usize);
    }
}

fn draw_feature_navigator(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(50, 60, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
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
        spinner::frame(app.spinner_tick)
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
        .border_set(panel_border_set(app.config.rounded_borders))
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
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(cmd_border)
        .title(" ⌨️  Command (e.g., 'update deps', 'analyze tests') ");
    let display_cmd = if app.bytebot_command.is_empty() {
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
                "done" => ("\u{2705}".to_string(), app.theme.success),
                "running" => {
                    let running_icon: String = if app.bytebot_running {
                        spinner::frame(app.spinner_tick).to_string()
                    } else {
                        "\u{23F3}".to_string()
                    };
                    (running_icon, app.theme.warning)
                }
                // A refusal is its own outcome: the call was never made, which
                // is not the same as one that ran and failed.
                "denied" | "refused" => ("\u{2717}".to_string(), app.theme.warning),
                _ => ("\u{274C}".to_string(), app.theme.danger), // failed / unknown
            };
            let suffix = match status.as_str() {
                "done" | "running" => String::new(),
                other => format!(" · {other}"),
            };
            steps_lines.push(Line::from(vec![
                Span::styled(format!(" {} ", icon), Style::default().fg(color)),
                Span::styled(
                    format!("Step {}: {}{}", i + 1, step_name, suffix),
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
            let pct = (app.bytebot_progress * 100.0).round();
            let bar = gauge::bar(
                (app.bytebot_progress * 100.0).round() as u64,
                100,
                bar_width,
            );
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
        .border_set(panel_border_set(app.config.rounded_borders))
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
                app.theme.success
            } else if entry.starts_with('❌') || entry.starts_with("failed") {
                app.theme.danger
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
        .border_set(panel_border_set(app.config.rounded_borders))
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
            .border_set(panel_border_set(app.config.rounded_borders))
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
        spinner::frame(app.spinner_tick).to_string()
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
        .border_set(panel_border_set(app.config.rounded_borders))
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
            "done" => ("\u{2705}".to_string(), app.theme.success),
            "running" => ("\u{23F3}".to_string(), app.theme.warning),
            "failed" => ("\u{274C}".to_string(), app.theme.danger),
            _ => ("\u{25CB}".to_string(), app.theme.message_system),
        };
        steps_lines.push(Line::from(vec![
            Span::styled(format!(" {} ", icon), Style::default().fg(color)),
            Span::styled(name.clone(), Style::default().fg(app.theme.fg)),
        ]));
    }

    let steps_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
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
                app.theme.danger
            } else if entry.contains('\u{2705}') {
                app.theme.success
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
        .border_set(panel_border_set(app.config.rounded_borders))
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

/// "admin" → "Admin" — server roles arrive lowercase from the wire.
fn capitalize_role(role: &str) -> String {
    let mut chars = role.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn draw_collaboration_hub(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(75, 72, area);
    f.render_widget(Clear, popup_area);

    let status_icon: String = match app.collab_sync_status.as_str() {
        "connected" => "✅".to_string(),
        "connecting" => spinner::frame(app.spinner_tick).to_string(),
        "disconnected" => "○".to_string(),
        _ => "❓".to_string(),
    };

    let title = format!(
        " 👥 Collaboration Hub [{}] (c:create j:join ⏎:connect r:retry Tab:field Esc:back) ",
        status_icon
    );
    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
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
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(session_border)
        .title(" \u{1F310} Session ");

    let session_display = if !session_active {
        // The form: three fields, the edited one marked and cursorred.
        let cursor = |selected: bool| {
            if selected {
                "▸"
            } else {
                " "
            }
        };
        use crate::focus::CollabField;
        let session_shown = if app.collab_session_id.is_empty() {
            "(new)"
        } else {
            app.collab_session_id.as_str()
        };
        let mut line = format!(
            " {}Server: {}   {}User: {}   {}Session: {}",
            cursor(app.collab_editing && app.collab_field == CollabField::Server),
            app.collab_server_url,
            cursor(app.collab_editing && app.collab_field == CollabField::Username),
            app.collab_username,
            cursor(app.collab_editing && app.collab_field == CollabField::Session),
            session_shown,
        );
        if app.collab_editing {
            line.push_str(" ▏");
        } else {
            line.push_str("   c/⏎ to connect");
        }
        line
    } else {
        let status_color = match app.collab_sync_status.as_str() {
            "connected" => "\u{2705}",
            "connecting" => "\u{1F504}",
            "disconnected" => "\u{274C}",
            _ => "\u{2753}",
        };
        format!(
            " ID: {}   Status: {} {}   Members: {}",
            app.collab_session_id,
            status_color,
            app.collab_sync_status,
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

        for (name, role, _connection) in &app.collab_members {
            // Presence is the server's members list: everyone in it is
            // connected right now; the badge is their workspace role.
            let badge = capitalize_role(role);
            let label = if badge.is_empty() {
                name.clone()
            } else {
                format!("{name} [{badge}]")
            };
            member_lines.push(Line::from(vec![
                Span::styled(" \u{25CF} ", Style::default().fg(app.theme.success)),
                Span::styled(label, Style::default().fg(app.theme.fg)),
            ]));
        }
    } else {
        member_lines.push(Line::from(""));
        member_lines.push(Line::from(Span::styled(
            if session_active {
                "  Waiting for the server…"
            } else {
                "  Not connected. c: new session, j: join one."
            },
            Style::default().fg(app.theme.message_system),
        )));
    }

    // Connection info at bottom of members panel — only what is real.
    member_lines.push(Line::from(""));
    member_lines.push(Line::from(Span::styled(
        " Connection",
        Style::default()
            .fg(app.theme.fg)
            .add_modifier(Modifier::UNDERLINED),
    )));
    member_lines.push(Line::from(""));
    member_lines.push(Line::from(format!("   Server: {}", app.collab_server_url)));
    let transport = if app.collab_server_url.starts_with("https://") {
        "wss (TLS)"
    } else {
        "ws (no TLS)"
    };
    member_lines.push(Line::from(format!("   Transport: {transport}")));
    if session_active && app.collab_last_sync > 0.0 {
        let elapsed = (current_timestamp() - app.collab_last_sync).max(0.0);
        member_lines.push(Line::from(format!("   Connected: {elapsed:.0}s")));
    }
    if !app.collab_error.is_empty() {
        member_lines.push(Line::from(Span::styled(
            format!("   ⚠ {}", app.collab_error),
            Style::default().fg(app.theme.danger),
        )));
    }

    let member_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
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
                app.theme.success
            } else if entry.starts_with('\u{274C}') || entry.starts_with('\u{26A0}') {
                app.theme.danger
            } else if entry.starts_with('\u{1F4E4}') || entry.starts_with('\u{1F4E5}') {
                app.theme.info
            } else if entry.starts_with('\u{1F464}') {
                app.theme.warning
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
            let frame = spinner::frame(app.spinner_tick);
            activity_lines.push(Line::from(Span::styled(
                format!(" {} Synchronizing...", frame),
                Style::default().fg(app.theme.accent),
            )));
        }
    }

    let activity_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
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

    let status_icon = if app.voice_muted && app.voice_busy {
        "🔇"
    } else {
        match app.voice_status.as_str() {
            "listening" => "🎤",
            "processing" => "🔄",
            _ => "🎙️",
        }
    };

    let title = format!(
        " 🎙️ Voice Interface [{}] (Enter:record/stop  m:mute  Esc:close) ",
        status_icon
    );
    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // meter + status
            Constraint::Length(6), // what the session produced
            Constraint::Min(1),    // transcript
        ])
        .split(inner);

    // ── Audio Meter + Status ────────────────────────────────────────────────
    let status_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📡 Audio Input ");

    let bar_width = 30usize;
    let shown_level = if app.voice_muted {
        0.0
    } else {
        app.voice_level
    };
    let bar = gauge::bar((shown_level * 100.0).round() as u64, 100, bar_width);
    let meter = format!(
        " Level: [{}] {:.0}%\n Peak: {:.0}%  Length: {}  {}",
        bar,
        shown_level * 100.0,
        app.voice_peak * 100.0,
        crate::voice::format_pcm_ms(app.voice_pcm_bytes),
        if app.voice_muted {
            "muted — audio discarded"
        } else {
            app.voice_status.as_str()
        },
    );
    let status_para = Paragraph::new(meter)
        .block(status_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(status_para, chunks[0]);

    // ── What this session did ───────────────────────────────────────────────
    let session_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📋 Session ");

    let mut session_lines: Vec<Line> = Vec::new();
    if !app.voice_active {
        session_lines.push(Line::from(Span::styled(
            "  Enter records from the microphone and the meter follows the",
            Style::default().fg(app.theme.message_system),
        )));
        session_lines.push(Line::from(Span::styled(
            "  level of what the recorder actually sent.",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        session_lines.push(Line::from(Span::styled(
            format!(
                "  Recorder: {}",
                if app.voice_recorder.is_empty() {
                    "none found on PATH"
                } else {
                    app.voice_recorder.as_str()
                }
            ),
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        let note_color = if app.voice_note.contains("No speech-to-text")
            || app.voice_note.contains("No recorder")
            || app.voice_note.contains("no audio")
            || app.voice_note.contains("failed")
            || app.voice_note.contains("exited")
            || app.voice_note.contains("heard nothing")
        {
            app.theme.warning
        } else {
            app.theme.message_system
        };
        for line in app.voice_note.lines().take(3) {
            session_lines.push(Line::from(Span::styled(
                format!("  {line}"),
                Style::default().fg(note_color),
            )));
        }
        session_lines.push(Line::from(Span::styled(
            "  Transcripts appear only from an installed speech engine.",
            Style::default().fg(app.theme.message_system),
        )));
    }
    let session_para = Paragraph::new(session_lines)
        .block(session_block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false });
    f.render_widget(session_para, chunks[1]);

    // ── Transcript Log ──────────────────────────────────────────────────────
    let trans_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📝 Transcript ");

    let mut trans_lines: Vec<Line> = Vec::new();
    if app.voice_transcript.is_empty() {
        trans_lines.push(Line::from(Span::styled(
            if app.voice_busy {
                "  Listening — nothing transcribed yet."
            } else {
                "  No speech text. Nothing here is invented; see Session for why."
            },
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        for entry in &app.voice_transcript {
            trans_lines.push(Line::from(Span::styled(
                format!("  🗣️  {entry}"),
                Style::default().fg(app.theme.fg),
            )));
        }
    }
    let trans_para = Paragraph::new(trans_lines)
        .block(trans_block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false });
    f.render_widget(trans_para, chunks[2]);
}

// ── Terminal Assistant Panel ────────────────────────────────────────────────

fn draw_terminal_assistant(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(75, 65, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 💡 Terminal Assistant (i:type, Enter:ask/run, f:filter, Esc:close) ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // query + last status line
            Constraint::Min(1),    // suggestions
            Constraint::Length(4), // history
        ])
        .split(inner);

    // ── Query / status ──────────────────────────────────────────────────────
    let status_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(format!(
            " 🤖 {} · asking through the agent's approval gate ",
            app.config.default_model
        ));
    // While typing, this box *is* the input field: the caret is drawn where
    // the next character lands, so the user can see what they are asking.
    let mut status_lines: Vec<Line> = Vec::new();
    if app.term_asst_typing {
        let mut spans = vec![
            Span::styled("  › ", Style::default().fg(app.theme.accent)),
            Span::styled(
                app.term_asst_query.clone(),
                Style::default().fg(app.theme.fg),
            ),
        ];
        if !app.term_asst_busy {
            spans.push(Span::styled("▏", Style::default().fg(app.theme.accent)));
        }
        status_lines.push(Line::from(spans));
    } else if !app.term_asst_query.is_empty() {
        status_lines.push(Line::from(Span::styled(
            format!("  › {}", app.term_asst_query),
            Style::default().fg(app.theme.message_system),
        )));
    }
    let idle_hint = "Type what you want to do, then Enter asks the model for commands.";
    let output = if app.term_asst_output.is_empty() {
        if app.term_asst_typing && app.term_asst_query.is_empty() {
            idle_hint
        } else {
            ""
        }
    } else {
        &app.term_asst_output
    };
    if !output.is_empty() {
        status_lines.push(Line::from(Span::styled(
            format!("  {output}"),
            Style::default().fg(if app.term_asst_busy {
                app.theme.accent
            } else if app.term_asst_output.is_empty() {
                app.theme.message_system
            } else {
                app.theme.warning
            }),
        )));
    }
    let status_para = Paragraph::new(status_lines)
        .block(status_block)
        .wrap(Wrap { trim: false });
    f.render_widget(status_para, chunks[0]);

    // ── Command suggestions ─────────────────────────────────────────────────
    let sugg_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(format!(
            " 📋 Suggested Commands {} ",
            if app.term_risk_filter == "All" {
                String::new()
            } else {
                format!("· filtered: {}", app.term_risk_filter)
            }
        ));

    let visible = app.term_visible_rows();
    let mut sugg_lines: Vec<Line> = Vec::new();
    if app.term_asst_busy && app.term_asst_suggestions.is_empty() {
        sugg_lines.push(Line::from(Span::styled(
            "  Waiting for the model…",
            Style::default().fg(app.theme.message_system),
        )));
    } else if visible.is_empty() {
        sugg_lines.push(Line::from(Span::styled(
            if app.term_asst_suggestions.is_empty() {
                "  No suggestions yet — type what you want to do and press Enter."
            } else {
                "  Nothing matches this filter; press f to show the rest."
            },
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        for (row, &idx) in visible.iter().enumerate() {
            let (command, risk, why) = &app.term_asst_suggestions[idx];
            let selected = row == app.term_asst_selected.min(visible.len().saturating_sub(1));
            let risky = risk.eq_ignore_ascii_case("destructive");
            sugg_lines.push(Line::from(vec![
                Span::styled(
                    if selected { "▶ " } else { "  " },
                    Style::default().fg(app.theme.accent),
                ),
                Span::styled(
                    if risky { "[risk] " } else { "[ok]   " },
                    Style::default().fg(if risky {
                        app.theme.danger
                    } else {
                        app.theme.success
                    }),
                ),
                Span::styled(
                    command.clone(),
                    Style::default().fg(app.theme.fg).add_modifier(if selected {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
                ),
            ]));
            if !why.is_empty() {
                sugg_lines.push(Line::from(Span::styled(
                    format!("   {}", why),
                    Style::default().fg(app.theme.message_system),
                )));
            }
        }
    }
    let sugg_para = Paragraph::new(sugg_lines)
        .block(sugg_block)
        .wrap(Wrap { trim: false });
    f.render_widget(sugg_para, chunks[1]);

    // ── History ─────────────────────────────────────────────────────────────
    let hist_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📜 Commands This Session ");
    // Newest last, and the inner rows are the ones that ran — through the
    // approval gate, so an outcome here is a real result or a real denial.
    let hist_lines: Vec<Line> = if app.term_asst_history.is_empty() {
        vec![Line::from(Span::styled(
            "  Nothing has been run from this panel yet.",
            Style::default().fg(app.theme.message_system),
        ))]
    } else {
        app.term_asst_history
            .iter()
            .rev()
            .take(2)
            .rev()
            .map(|(command, risk, result)| {
                Line::from(Span::styled(
                    format!("  {} [{}] → {}", command, risk, result),
                    Style::default().fg(app.theme.fg),
                ))
            })
            .collect()
    };
    let hist_para = Paragraph::new(hist_lines)
        .block(hist_block)
        .wrap(Wrap { trim: false });
    f.render_widget(hist_para, chunks[2]);
}

// ── Security Auditor Panel ──────────────────────────────────────────────────

/// The security-auditor findings list (findings + scan-log tail); shared by
/// `draw_security_auditor` and the resize clamp.
fn security_findings_lines(app: &App) -> Vec<Line<'static>> {
    let mut find_lines: Vec<Line> = Vec::new();
    if app.sec_scan_results.is_empty() && !app.sec_scan_active {
        let idle = if app.sec_scan_progress >= 1.0 {
            "  No findings in the last scan. Press Enter to scan again."
        } else {
            "  Press Enter to start a vulnerability scan."
        };
        find_lines.push(Line::from(Span::styled(
            idle,
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
                "Critical" => ("🔴", app.theme.danger),
                "High" => ("🟡", app.theme.warning),
                "Medium" => ("🔵", app.theme.info),
                _ => ("🟢", app.theme.success),
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
                app.theme.danger
            } else if entry.starts_with('⚠') {
                app.theme.warning
            } else if entry.starts_with('🚨') {
                app.theme.danger
            } else {
                app.theme.fg
            };
            find_lines.push(Line::from(Span::styled(
                format!("  {}", entry),
                Style::default().fg(color),
            )));
        }
    }
    find_lines
}

fn draw_security_auditor(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(75, 70, area);
    f.render_widget(Clear, popup_area);

    let scanned = if app.sec_scan_path.is_empty() {
        String::from(" 🛡️ Security Auditor (Enter:scan, Esc:close) ")
    } else {
        format!(
            " 🛡️ Security Auditor · {} (Enter:scan, Esc:close) ",
            app.sec_scan_path
        )
    };
    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(scanned);

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
        severity_bar(critical, max_sev, app.theme.danger),
        severity_bar(high, max_sev, app.theme.warning),
        severity_bar(medium, max_sev, app.theme.info),
        severity_bar(low, max_sev, app.theme.success),
    ];

    let summary_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📊 Risk Summary ");
    let summary_para = Paragraph::new(summary_lines)
        .block(summary_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(summary_para, chunks[0]);

    // ── Findings List ───────────────────────────────────────────────────────
    let find_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🔎 Findings ");

    let find_lines = security_findings_lines(app);
    let find_len = find_lines.len();
    let find_para = Paragraph::new(find_lines)
        .block(find_block)
        .style(Style::default().fg(app.theme.fg))
        .wrap(Wrap { trim: false })
        .scroll((
            app.security_scroll
                .min(clamp_scroll(find_len, chunks[1].height)),
            0,
        ));
    f.render_widget(find_para, chunks[1]);
}

// ── Performance Profiler Panel ──────────────────────────────────────────────

/// One measured utilization line: a bar scaled against `max` and the true
/// value, or `n/a` when the measurement does not exist yet. A missing number
/// is never drawn as a zero.
fn util_line(label: &str, value: Option<f64>, max: f64, unit: &str) -> Line<'static> {
    let bar = |pct: f64| {
        let filled = ((pct / 100.0).clamp(0.0, 1.0) * 10.0).round() as usize;
        format!(
            "[{}{}]",
            "█".repeat(filled),
            "░".repeat(10usize.saturating_sub(filled))
        )
    };
    let text = match value {
        Some(v) if max > 0.0 => format!(
            "   {:<7} {}  {:>6.0}{}  ({:.0}%)",
            label,
            bar(v / max * 100.0),
            v,
            unit,
            v / max * 100.0
        ),
        Some(v) => format!("   {:<7} {:>6.0}{}  (no scale)", label, v, unit),
        None => format!("   {:<7} {:>6}{}", label, "n/a", unit),
    };
    Line::from(text)
}

fn draw_performance_profiler(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(72, 68, area);
    f.render_widget(Clear, popup_area);

    let status_indicator = if app.profiler_running {
        spinner::frame(app.spinner_tick)
    } else {
        '●'
    };

    let title = format!(
        " ⚡ Performance Profiler [{}] (Enter:profile, Esc:close) ",
        status_indicator
    );
    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(title);

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7), // gauges (5 lines + borders)
            Constraint::Min(1),    // measurements
        ])
        .split(inner);

    // ── Gauges ──────────────────────────────────────────────────────────────
    fn gauge_line(
        title: &str,
        value: Option<f64>,
        max: f64,
        unit: &str,
        color: ratatui::style::Color,
    ) -> Line<'static> {
        let w = 15usize;
        let (bar, tail) = match value {
            Some(v) if max > 0.0 => {
                let pct = (v / max * 100.0).min(100.0);
                let filled = (pct / 100.0 * w as f64).round() as usize;
                (
                    format!(
                        "[{}{}]",
                        "█".repeat(filled.min(w)),
                        "░".repeat(w.saturating_sub(filled))
                    ),
                    format!("{:.1}{}", v, unit),
                )
            }
            _ => ("[".to_owned() + &"░".repeat(w) + "]", "n/a".to_string()),
        };
        Line::from(Span::styled(
            format!(" {}  {}  {}", title, bar, tail),
            Style::default().fg(color),
        ))
    }

    let gauge_lines = vec![
        Line::from(Span::styled(
            " This Process",
            Style::default()
                .fg(app.theme.fg)
                .add_modifier(Modifier::UNDERLINED),
        )),
        Line::from(""),
        gauge_line(
            "CPU     ",
            app.profiler_gauge_cpu,
            100.0,
            "%",
            app.theme.info,
        ),
        gauge_line(
            "Memory  ",
            app.profiler_gauge_mem,
            app.profiler_gauge_mem_total.unwrap_or(0.0),
            " MB",
            app.theme.accent_secondary,
        ),
        gauge_line(
            "Latency ",
            app.profiler_gauge_latency,
            500.0,
            " ms",
            app.theme.warning,
        ),
    ];

    let gauge_block_w = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📈 Gauges ");
    let gauge_para = Paragraph::new(gauge_lines)
        .block(gauge_block_w)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(gauge_para, chunks[0]);

    // ── Measurements ────────────────────────────────────────────────────────
    let func_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📊 Measurements ");

    let mut func_lines: Vec<Line> = Vec::new();
    if app.profiler_rows.is_empty() && !app.profiler_running {
        func_lines.push(Line::from(Span::styled(
            "  Press Enter to measure this session.",
            Style::default().fg(app.theme.message_system),
        )));
        func_lines.push(Line::from(Span::styled(
            " Process usage, turn latency, provider health, recorded metrics.",
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        func_lines.push(Line::from(vec![
            Span::styled(
                "  Source",
                Style::default()
                    .fg(app.theme.fg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "  Metric                    Value",
                Style::default().fg(app.theme.fg),
            ),
        ]));
        func_lines.push(Line::from(Span::styled(
            "  ".to_owned() + &"─".repeat(45),
            Style::default().fg(app.theme.border),
        )));
        for (source, metric, value) in app.profiler_rows.iter().take(24) {
            func_lines.push(Line::from(Span::styled(
                format!("  {:<12} {:<24} {}", source, metric, value),
                Style::default().fg(app.theme.fg),
            )));
        }
    }
    for note in app.profiler_notes.iter().rev().take(4) {
        func_lines.push(Line::from(Span::styled(
            format!("  · {}", note),
            Style::default().fg(app.theme.message_system),
        )));
    }

    let func_para = Paragraph::new(func_lines)
        .block(func_block)
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(func_para, chunks[1]);
}

// ── Custom Models Panel ─────────────────────────────────────────────────────

/// One of the two parameter bars. `None` renders as `unset`, not as zero: a
/// profile that sends nothing lets the server decide, and showing `0.00` would
/// claim the opposite.
fn param_bar(fraction: f64, width: usize) -> String {
    let filled = ((fraction.clamp(0.0, 1.0)) * width as f64).round() as usize;
    let filled = filled.min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

fn draw_custom_models(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(74, 70, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🧩 Custom Models (n:new, -/+ ←/→ tune, Enter:apply, s:save, t:test) ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(38), // profiles list
            Constraint::Percentage(62), // details
        ])
        .split(inner);

    // ── Profile List ────────────────────────────────────────────────────────
    let list_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(format!(" 📋 Profiles ({}) ", app.model_profiles.len()));

    let mut profile_items: Vec<ListItem> = Vec::new();
    for (i, profile) in app.model_profiles.iter().enumerate() {
        let style = if i == app.models_selected {
            Style::default()
                .fg(app.theme.highlight_fg)
                .bg(app.theme.highlight)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(app.theme.fg)
        };
        let edited = if app.models_dirty && i == app.models_selected {
            " ·unsaved"
        } else {
            ""
        };
        profile_items.push(ListItem::new(vec![
            Line::from(Span::styled(format!(" {}{}", profile.name, edited), style)),
            Line::from(Span::styled(
                format!("   {}", profile.model),
                Style::default().fg(app.theme.message_system),
            )),
        ]));
    }

    if profile_items.is_empty() {
        profile_items.push(ListItem::new(Line::from(Span::styled(
            "  None yet — press n",
            Style::default().fg(app.theme.message_system),
        ))));
    }

    let profile_list = List::new(profile_items).block(list_block);
    let mut list_state = ratatui::widgets::ListState::default();
    list_state.select(Some(app.models_selected));
    f.render_stateful_widget(profile_list, chunks[0], &mut list_state);

    // ── Profile Details ─────────────────────────────────────────────────────
    let detail_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🔧 Parameters ");

    let slider_w = 20usize;
    let mut detail_lines: Vec<Line> = Vec::new();
    if let Some(profile) = app.selected_model_profile() {
        detail_lines.push(Line::from(Span::styled(
            profile.name.clone(),
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        detail_lines.push(Line::from(format!("  model    {}", profile.model)));
        detail_lines.push(Line::from(""));

        let temp_line = match profile.temperature {
            Some(temp) => format!(
                "  temp     [{}] {:.2}",
                param_bar(temp / 2.0, slider_w),
                temp
            ),
            None => "  temp     unset — the server decides".to_string(),
        };
        detail_lines.push(Line::from(temp_line));
        let tokens_line = match profile.max_tokens {
            Some(tokens) => format!(
                "  tokens   [{}] {}",
                param_bar(tokens as f64 / 8192.0, slider_w),
                tokens
            ),
            None => "  tokens   unset — the server decides".to_string(),
        };
        detail_lines.push(Line::from(tokens_line));
    } else {
        detail_lines.push(Line::from(Span::styled(
            "  No model_profiles in config.json yet.",
            Style::default().fg(app.theme.warning),
        )));
        detail_lines.push(Line::from(""));
        detail_lines.push(Line::from("  Press n to start one from the model this"));
        detail_lines.push(Line::from("  session uses now, then s to keep it."));
    }

    detail_lines.push(Line::from(""));
    for hint in [
        "  -/+ temp · ←/→ tokens · n new profile",
        "  Enter apply · s save · t test",
        "  A llama.cpp server gets both numbers;",
        "  Ollama and cloud use their own defaults.",
    ] {
        detail_lines.push(Line::from(Span::styled(
            hint,
            Style::default().fg(app.theme.message_system),
        )));
    }

    if !app.models_status.is_empty() {
        let failed = app.models_status.starts_with("test failed")
            || app.models_status.contains("unchanged")
            || app.models_status.contains("nothing was written");
        detail_lines.push(Line::from(""));
        detail_lines.push(Line::from(Span::styled(
            format!("  {}", app.models_status),
            Style::default().fg(if failed {
                app.theme.danger
            } else {
                app.theme.success
            }),
        )));
    }

    let detail_para = Paragraph::new(detail_lines)
        .block(detail_block)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(app.theme.fg));
    f.render_widget(detail_para, chunks[1]);
}

// ── Learning Mode Panel ─────────────────────────────────────────────────────

fn draw_learning_mode(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(70, 70, area);
    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 📚 Learning Mode — lessons from this repo ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            // Two border rows plus both header lines: at Length(3) the queue
            // count was clipped and the panel showed only the file name.
            Constraint::Length(4),
            Constraint::Min(1), // lesson text, model's words, quiz
        ])
        .split(inner);

    // ── Header + Progress ───────────────────────────────────────────────────
    let header_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🎯 Lesson Queue ");

    let bar_w = 25usize;
    let (seen, total) = (app.learn_current_lesson, app.learn_total_lessons);
    // Progress means "how far through the queue the index built", not a score.
    let pct = if total > 0 {
        seen as f64 * 100.0 / total as f64
    } else {
        0.0
    };
    let filled = ((pct / 100.0) * bar_w as f64).round() as usize;
    let filled = filled.min(bar_w);
    let empty = bar_w.saturating_sub(filled);
    let progress_str = if total == 0 {
        "  The project index queued 0 lessons.".to_string()
    } else {
        format!(
            "  lesson {seen} of {total}  |  [{}{}]  {:.0}% of the queue",
            "█".repeat(filled),
            "░".repeat(empty),
            pct,
        )
    };

    let header_lines = vec![
        Line::from(Span::styled(
            if app.learn_lesson_title.is_empty() {
                "  No file chosen yet".to_string()
            } else {
                format!("  {}", app.learn_lesson_title)
            },
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
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 📖 Content ");

    let mut content_lines: Vec<Line> = Vec::new();
    if !app.learn_status.is_empty() {
        let bad = app.learn_status.starts_with("provider said")
            || app.learn_status.starts_with("The model did not")
            || app.learn_status.starts_with("The index names")
            || app.learn_status.starts_with("No project index");
        content_lines.push(Line::from(Span::styled(
            format!("  {}", app.learn_status),
            Style::default().fg(if bad {
                app.theme.danger
            } else {
                app.theme.message_system
            }),
        )));
        content_lines.push(Line::from(""));
    }
    if app.learn_content.is_empty() {
        content_lines.push(Line::from(Span::styled(
            if app.learn_active {
                "  Nothing queued. Enter builds lessons from .xencode/index."
            } else {
                "  Enter queues the files this workspace's index says declare something,"
            },
            Style::default().fg(app.theme.message_system),
        )));
        if !app.learn_active {
            content_lines.push(Line::from(Span::styled(
                "  then asks the model to teach that file.",
                Style::default().fg(app.theme.message_system),
            )));
        }
    } else {
        // Facts the index and the file itself produced, before any model text.
        for point in &app.learn_content {
            content_lines.push(Line::from(Span::styled(
                format!("  • {}", point),
                Style::default().fg(app.theme.fg),
            )));
        }

        if !app.learn_explain.is_empty() {
            content_lines.push(Line::from(""));
            content_lines.push(Line::from(Span::styled(
                "  The model says:",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            )));
            for sentence in &app.learn_explain {
                content_lines.push(Line::from(Span::styled(
                    format!("    {sentence}"),
                    Style::default().fg(app.theme.fg),
                )));
            }
        }

        // Code example
        if !app.learn_code_example.is_empty() {
            content_lines.push(Line::from(""));
            content_lines.push(Line::from(Span::styled(
                "  From the file:",
                Style::default()
                    .fg(app.theme.accent)
                    .add_modifier(Modifier::BOLD),
            )));
            for line in app.learn_code_example.lines() {
                content_lines.push(Line::from(Span::styled(
                    format!("    {}", line),
                    Style::default().fg(app.theme.success),
                )));
            }
        }

        // Quiz section
        if app.learn_quiz_active {
            content_lines.push(Line::from(""));
            content_lines.push(Line::from(Span::styled(
                "  Quiz (the model's question and answer key):",
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
                let is_key = app.learn_quiz_answered && app.learn_quiz_answer == Some(i);
                let prefix = if is_selected && !app.learn_quiz_answered {
                    "  \u{25B6} "
                } else if is_key {
                    "  \u{2713} "
                } else {
                    "    "
                };
                let style = if is_selected && !app.learn_quiz_answered {
                    Style::default()
                        .fg(app.theme.highlight_fg)
                        .bg(app.theme.highlight)
                } else if is_key {
                    Style::default().fg(app.theme.success)
                } else if app.learn_quiz_answered && is_selected && !app.learn_quiz_correct {
                    Style::default().fg(app.theme.danger)
                } else {
                    Style::default().fg(app.theme.fg)
                };
                content_lines.push(Line::from(Span::styled(
                    format!("{}{}", prefix, option),
                    style,
                )));
            }
            content_lines.push(Line::from(""));
            if app.learn_quiz_answered {
                let verdict = if app.learn_quiz_correct {
                    "correct"
                } else {
                    "not the key"
                };
                content_lines.push(Line::from(Span::styled(
                    format!(
                        "  You picked {} — the model's key was option {}; {}.",
                        app.learn_quiz_selected + 1,
                        app.learn_quiz_answer.map(|a| a + 1).unwrap_or(0),
                        verdict,
                    ),
                    Style::default().fg(if app.learn_quiz_correct {
                        app.theme.success
                    } else {
                        app.theme.warning
                    }),
                )));
                if !app.learn_quiz_why.is_empty() {
                    content_lines.push(Line::from(Span::styled(
                        format!("  Why: {}", app.learn_quiz_why),
                        Style::default().fg(app.theme.fg),
                    )));
                }
            } else {
                content_lines.push(Line::from(Span::styled(
                    "  Select with \u{2190}\u{2192}, confirm with Enter",
                    Style::default().fg(app.theme.message_system),
                )));
            }
        } else if app.learn_busy {
            content_lines.push(Line::from(""));
            content_lines.push(Line::from(Span::styled(
                "  Waiting for the model's quiz…",
                Style::default().fg(app.theme.message_system),
            )));
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
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.accent))
        .title(" 🌐 Multi-Language (Enter:detect  Tab:type  Enter:translate  Esc:close) ");

    let inner = block.inner(popup_area);
    f.render_widget(block, popup_area);

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(52), // what this workspace is written in
            Constraint::Percentage(48), // what the scanner knows + translation
        ])
        .split(inner);

    // ── Left: the walk's own numbers ────────────────────────────────────────
    let detect_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(format!(
            " 🔍 This workspace {} ",
            if app.lang_scan_path.is_empty() {
                String::new()
            } else {
                format!("· {}", app.lang_scan_path)
            }
        ));

    let mut detect_lines: Vec<Line> = Vec::new();
    if app.lang_busy && app.lang_detection_results.is_empty() {
        detect_lines.push(Line::from(Span::styled(
            "  Walking the workspace…",
            Style::default().fg(app.theme.message_system),
        )));
    } else if app.lang_detection_results.is_empty() {
        detect_lines.push(Line::from(Span::styled(
            if app.lang_notes.is_empty() {
                "  Nothing counted yet — press Enter to walk the workspace."
            } else {
                "  The walk listed no files. The reason is below."
            },
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        detect_lines.push(Line::from(Span::styled(
            format!(
                "  {:<12}{:>6}{:>9}{:>8}",
                "language", "files", "lines", "share"
            ),
            Style::default()
                .fg(app.theme.accent)
                .add_modifier(Modifier::BOLD),
        )));
        for (language, files, lines, share) in &app.lang_detection_results {
            detect_lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:<12}{:>6}{:>9}", language, files, lines),
                    Style::default().fg(app.theme.fg),
                ),
                Span::styled(
                    format!("{:>5.1}%", share),
                    Style::default().fg(app.theme.info),
                ),
            ]));
        }
    }
    // The walk's caveats ride under the table: an ignored-tree or an unreadable
    // file has to be visible, not silently missing from the totals.
    for note in &app.lang_notes {
        detect_lines.push(Line::from(Span::styled(
            format!("  {note}"),
            Style::default().fg(app.theme.warning),
        )));
    }
    let detect_para = Paragraph::new(detect_lines)
        .block(detect_block)
        .wrap(Wrap { trim: false });
    f.render_widget(detect_para, chunks[0]);

    // ── Right: the scanner's vocabulary, then the translator ────────────────
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9), // recognised languages
            Constraint::Min(5),    // translation
        ])
        .split(chunks[1]);

    let seen: std::collections::HashSet<&'static str> = xencode_context_rs::scanner::Language::ALL
        .iter()
        .map(|language| language.as_str())
        .filter(|name| {
            app.lang_detection_results
                .iter()
                .any(|(found, _, _, _)| found == name)
        })
        .collect();
    let supp_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(format!(
            " 🌍 Languages the scanner names ({}) {} ",
            xencode_context_rs::scanner::Language::ALL.len(),
            if app.lang_detection_results.is_empty() {
                "· nothing detected yet"
            } else {
                "· ▸ present here"
            }
        ));
    // Four per row: this is the enum's own contents, so the list can never
    // promise a language the walker cannot name.
    let names: Vec<&'static str> = xencode_context_rs::scanner::Language::ALL
        .iter()
        .map(|language| language.as_str())
        .collect();
    let mut supp_lines: Vec<Line> = Vec::new();
    for group in names.chunks(4) {
        let mut spans = Vec::new();
        for name in group {
            let present = seen.contains(name);
            spans.push(Span::styled(
                format!("{:>2}{:<12}", if present { "▸" } else { " " }, name),
                Style::default().fg(if present {
                    app.theme.success
                } else {
                    app.theme.message_system
                }),
            ));
        }
        supp_lines.push(Line::from(spans));
    }
    let supp_para = Paragraph::new(supp_lines).block(supp_block);
    f.render_widget(supp_para, right[0]);

    let trans_block = Block::default()
        .border_set(panel_border_set(app.config.rounded_borders))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(app.theme.border))
        .title(" 🔄 Translate (one model call) ");
    let field = |label: &str, value: &str, edited: bool| -> Line {
        let mut spans = vec![Span::styled(
            format!("  {label:<6}"),
            Style::default().fg(if edited {
                app.theme.accent
            } else {
                app.theme.message_system
            }),
        )];
        spans.push(Span::styled(
            if value.is_empty() && !edited {
                "—".to_string()
            } else {
                value.to_string()
            },
            Style::default().fg(app.theme.fg).add_modifier(if edited {
                Modifier::BOLD
            } else {
                Modifier::empty()
            }),
        ));
        if edited {
            spans.push(Span::styled("▏", Style::default().fg(app.theme.accent)));
        }
        Line::from(spans)
    };
    let editing = app.lang_editing;
    let mut trans_lines = vec![
        field(
            "From",
            &app.lang_translate_source,
            editing == Some(crate::focus::LangField::Source),
        ),
        field(
            "To",
            &app.lang_translate_target,
            editing == Some(crate::focus::LangField::Target),
        ),
        field(
            "Text",
            &app.lang_translate_input,
            editing == Some(crate::focus::LangField::Input),
        ),
        Line::from(""),
    ];
    if app.lang_translate_output.is_empty() {
        trans_lines.push(Line::from(Span::styled(
            if app.lang_busy {
                "  Waiting for the model…"
            } else {
                "  Tab to a field, type, Enter asks the model. The reply appears here."
            },
            Style::default().fg(app.theme.message_system),
        )));
    } else {
        for line in app.lang_translate_output.lines() {
            trans_lines.push(Line::from(Span::styled(
                format!("  {line}"),
                Style::default().fg(if app.lang_translate_error {
                    app.theme.danger
                } else {
                    app.theme.success
                }),
            )));
        }
    }
    let trans_para = Paragraph::new(trans_lines)
        .block(trans_block)
        .wrap(Wrap { trim: false });
    f.render_widget(trans_para, right[1]);
}

#[cfg(test)]
mod tests {
    use super::{clamp_scroll, App};

    /// K-3: the health popup renders the Remote/Colab forward row — status,
    /// latency, the configured URI in Connection Details, and (when set) the
    /// URL right under the row.
    #[test]
    fn provider_health_lists_the_remote_forward_row() {
        use super::provider_health_lines;

        let mut app = App::for_tests();
        app.ollama_health_entries
            .insert("remote".to_string(), ("healthy".to_string(), 42.5, None));
        app.config.remote_base_url = "http://127.0.0.1:18000/v1".to_string();

        let lines = provider_health_lines(&app);
        let text: String = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("Remote"), "row rendered: {text}");
        assert!(text.contains("healthy"), "status rendered: {text}");
        assert!(text.contains("42 ms"), "latency rendered: {text}");
        assert!(
            text.contains("http://127.0.0.1:18000/v1"),
            "forward URI rendered: {text}"
        );
        assert!(text.contains("Remote URI:"), "details row: {text}");
    }

    #[test]
    fn clamp_scroll_bounds_to_scrollable_rows() {
        // 10 rows in a 12-tall bordered area: everything fits, no scroll.
        assert_eq!(clamp_scroll(10, 12), 0);
        // 20 rows in a 10-tall area: 10 scrollable lines past the 8 visible.
        assert_eq!(clamp_scroll(20, 10), 12);
        // Degenerate heights must not underflow.
        assert_eq!(clamp_scroll(5, 0), 5);
        assert_eq!(clamp_scroll(0, 24), 0);
    }

    #[test]
    fn centered_rect_lifts_useless_popups_but_not_designed_ones() {
        use super::centered_rect;
        use ratatui::layout::Rect;

        // 20x8: 60/50 % would give a 12x4 border-only shell — it grows to
        // the 80 % floor instead (H1-09).
        let r = centered_rect(60, 50, Rect::new(0, 0, 20, 8));
        assert!(r.width >= 16 && r.height >= 6, "{r:?}");
        assert!(r.intersection(Rect::new(0, 0, 20, 8)) == r);

        // A normal terminal keeps the requested percentage untouched.
        let big = centered_rect(50, 50, Rect::new(0, 0, 100, 40));
        assert_eq!((big.width, big.height), (50, 20));

        // 1x1 stays inside the screen without underflowing.
        let tiny = centered_rect(70, 70, Rect::new(0, 0, 1, 1));
        assert!(tiny.width <= 1 && tiny.height <= 1, "{tiny:?}");
    }

    #[test]
    fn classic_hit_test_follows_body_layout() {
        use super::FocusArea;
        use crate::layout::compute_layout;
        use ratatui::layout::Rect;

        for width in [1u16, 7, 20, 33, 61, 80, 100, 120, 240] {
            let area = Rect::new(0, 0, width, 24);
            let layout = compute_layout(area, "classic", false, FocusArea::ChatInput);
            let (explorer, editor, chat) = (
                layout.explorer.unwrap(),
                layout.editor.unwrap(),
                layout.chat.unwrap(),
            );
            // The three panes tile the full width: no dead columns, no overlap.
            assert_eq!(explorer.x, 0);
            assert_eq!(editor.x, explorer.right());
            assert_eq!(chat.x, editor.right());
            assert_eq!(chat.right(), area.right());

            // Every column hits exactly the pane rendered under it.
            for col in 0..width {
                let expected = if col < explorer.right() {
                    FocusArea::FileExplorer
                } else if col < editor.right() {
                    FocusArea::CodeEditor
                } else {
                    FocusArea::ChatInput
                };
                assert_eq!(
                    layout.hit_test(col),
                    Some(expected),
                    "width {width} col {col}"
                );
            }
        }
    }

    #[test]
    fn resize_clamp_shrinks_stored_scrolls_and_never_inflates() {
        use super::clamp_scrolls_on_resize;
        use crate::app::UiMessage;

        let mut app = App::for_tests();
        for i in 0..20 {
            app.messages.push(UiMessage {
                role: "user".to_string(),
                content: format!("line {i}"),
            });
        }
        app.chat_scroll = 9999;
        app.review_scroll = 9999;
        app.provider_health_scroll = 9999;
        app.security_scroll = 9999;
        app.help_scroll = 9999;

        let scrolls = |app: &App| {
            [
                app.chat_scroll,
                app.review_scroll,
                app.provider_health_scroll,
                app.security_scroll,
                app.help_scroll,
            ]
        };

        clamp_scrolls_on_resize(&mut app, 60, 12);
        let shrunk = scrolls(&app);
        for v in shrunk.iter() {
            assert!(*v < 9999, "oversized scroll survived a shrink");
        }
        // The 60-line transcript overflows a 12-row terminal, so the chat
        // offset must clamp to a positive maximum rather than vanish to 0.
        assert!(shrunk[0] > 0);

        clamp_scrolls_on_resize(&mut app, 200, 100);
        let grown = scrolls(&app);
        for (before, after) in shrunk.iter().zip(grown.iter()) {
            assert!(*after <= *before, "re-inflated on grow");
        }
        // Everything fits in a 200x100 terminal.
        assert_eq!(grown, [0; 5]);
    }
}
