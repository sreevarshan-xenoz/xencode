//! Keyboard dispatch (E6-01): the event loop's 900-line key `match` lives
//! here as a global chord table plus per-focus `key_*` handlers.
//!
//! Order of evaluation in Normal mode is: Tab/Esc (always universal) →
//! focus handler → remaining global chords (`i` `/` `m` `s` `?` `q`).
//! Focus handlers run before the global chords on purpose: a panel owns
//! its own letter keys, which is what makes SecurityAuditor `s` and
//! CustomModels `s` reachable again (the E2-06 binding conflict).

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::sync::mpsc;

use xencode_config_rs::XencodeConfig;

use crate::app::{first_output_line, llama_model_target, App};
use crate::focus::{navigate_feature, FocusArea, InputMode, FEATURE_LIST};
use crate::theme::ThemeColors;

type Tx = mpsc::UnboundedSender<String>;

/// What the event loop should do after a key was handled.
#[derive(Debug, PartialEq, Eq)]
pub enum KeyFlow {
    Continue,
    Quit,
}

/// Entry point called from `Event::Key` in the run loop.
pub fn handle_key(app: &mut App, key: KeyEvent, tx: &Tx) -> KeyFlow {
    // The approval prompt is topmost and modal: it answers y/a/n/Esc,
    // scrolls the diff, and swallows everything else — quit chords
    // included — exactly like the help overlay.
    if app.pending_approval().is_some() {
        return approval_modal_key(app, key);
    }
    // The help overlay is modal: Esc/?/F1 close it, every other key is
    // swallowed while it is open.
    if app.help_visible {
        return help_modal_key(app, key);
    }
    // Global Ctrl chords work in ALL input modes.
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        if let Some(flow) = global_ctrl_chord(app, key, tx) {
            return flow;
        }
    }
    match app.input_mode {
        InputMode::Editing => editing_key(app, key, tx),
        InputMode::Normal => normal_key(app, key, tx),
    }
}

fn quit() -> KeyFlow {
    KeyFlow::Quit
}

fn done() -> KeyFlow {
    KeyFlow::Continue
}

/// The approval prompt answers with four keys and nothing else. Enter is
/// deliberately swallowed rather than treated as a deny: a stray Return in
/// the terminal should never be read as an answer.
fn approval_modal_key(app: &mut App, key: KeyEvent) -> KeyFlow {
    use crate::agent_tools::ApprovalAnswer;
    match key.code {
        KeyCode::Char('y') | KeyCode::Char('Y') => {
            app.resolve_approval(ApprovalAnswer::Approved);
        }
        KeyCode::Char('a') | KeyCode::Char('A') => {
            app.resolve_approval(ApprovalAnswer::ApprovedForSession);
        }
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
            app.resolve_approval(ApprovalAnswer::Denied);
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.approval_scroll = app.approval_scroll.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') | KeyCode::PageDown => {
            app.approval_scroll += 1;
        }
        _ => {}
    }
    done()
}

fn help_modal_key(app: &mut App, key: KeyEvent) -> KeyFlow {
    match key.code {
        KeyCode::Esc | KeyCode::Char('?') | KeyCode::F(1) => {
            app.help_visible = false;
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.help_scroll = app.help_scroll.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.help_scroll += 1;
        }
        _ => {}
    }
    done()
}

/// The global Ctrl chord table. `Some` = consumed, `None` = fall through to
/// the input-mode handlers (e.g. Ctrl+J reaches the chat textarea as a
/// newline).
fn global_ctrl_chord(app: &mut App, key: KeyEvent, tx: &Tx) -> Option<KeyFlow> {
    match key.code {
        KeyCode::Char('c') => return Some(quit()),
        KeyCode::Char('g') => {
            app.refresh_git();
        }
        KeyCode::Char(',') => {
            app.focus = if app.focus == FocusArea::Settings {
                FocusArea::ChatInput
            } else {
                FocusArea::Settings
            };
        }
        KeyCode::Char('b') => {
            if !app.bytebot_running && app.focus != FocusArea::ByteBotPanel {
                app.focus = FocusArea::ByteBotPanel;
            } else if app.focus == FocusArea::ByteBotPanel {
                app.focus = FocusArea::ChatInput;
            }
        }
        KeyCode::Char('d') => {
            app.focus = if app.focus == FocusArea::PerformanceDashboard {
                FocusArea::ChatInput
            } else {
                FocusArea::PerformanceDashboard
            };
        }
        KeyCode::Char('p') => {
            app.focus = if app.focus == FocusArea::ProjectAnalyzer {
                FocusArea::ChatInput
            } else {
                FocusArea::ProjectAnalyzer
            };
        }
        KeyCode::Char('e') => {
            app.focus = if app.focus == FocusArea::FileExplorer {
                FocusArea::ChatInput
            } else {
                FocusArea::FileExplorer
            };
        }
        KeyCode::Char('w') => {
            // Closing the hub with a live socket must not orphan the worker —
            // Esc is the documented hang-up, Ctrl+W does the same first.
            if app.focus == FocusArea::CollaborationHub {
                app.collab_editing = false;
                app.collab_disconnect();
            }
            // Close current panel and return to ChatInput
            match app.focus {
                FocusArea::ByteBotPanel
                | FocusArea::CollaborationHub
                | FocusArea::VoiceInterface
                | FocusArea::TerminalAssistant
                | FocusArea::SecurityAuditor
                | FocusArea::PerformanceProfiler
                | FocusArea::CustomModels
                | FocusArea::LearningMode
                | FocusArea::MultiLanguage
                | FocusArea::ProviderHealth
                | FocusArea::PerformanceDashboard
                | FocusArea::ProjectAnalyzer
                | FocusArea::GitCommit
                | FocusArea::CodeReview
                | FocusArea::ReviewDashboard
                | FocusArea::TaskManager
                | FocusArea::WorktreePanel
                | FocusArea::AdvisePanel
                | FocusArea::FeatureNavigator
                | FocusArea::ModelSelector
                | FocusArea::Settings => {
                    app.focus = FocusArea::ChatInput;
                }
                _ => {}
            }
        }
        KeyCode::Char('r') => {
            app.focus = if app.focus == FocusArea::CodeReview {
                FocusArea::ChatInput
            } else {
                FocusArea::CodeReview
            };
        }
        KeyCode::Char('y') => {
            if app.focus == FocusArea::ReviewDashboard {
                app.focus = FocusArea::ChatInput;
            } else {
                let base = app.review_dash.base.clone();
                app.review_dash.open(&base);
                app.focus = FocusArea::ReviewDashboard;
            }
        }
        KeyCode::Char('k') => {
            // Background task panel (D2-01): opens at the top of the list.
            if app.focus == FocusArea::TaskManager {
                app.focus = FocusArea::ChatInput;
            } else {
                app.tasks_selected = 0;
                app.tasks_detail = false;
                app.tasks_scroll = 0;
                app.focus = FocusArea::TaskManager;
            }
        }
        KeyCode::Char('o') => {
            // Worktree panel (D3-02): re-reads git on every open.
            if app.focus == FocusArea::WorktreePanel {
                app.focus = FocusArea::ChatInput;
            } else {
                app.worktree_selected = 0;
                app.worktree_prompt = crate::focus::WorktreePrompt::None;
                app.worktree_status.clear();
                app.refresh_worktrees();
                app.focus = FocusArea::WorktreePanel;
            }
        }
        KeyCode::Char('l') => {
            // Insights panel (F2-01): re-runs the deterministic analyses on
            // every open — they're pure and the snapshot is kept live.
            // (Ctrl+S stays with editor-save/GitCommit; Ctrl+I is Tab's
            // alias, so the panel lives on L.)
            if app.focus == FocusArea::AdvisePanel {
                app.focus = FocusArea::ChatInput;
            } else {
                app.advise_selected = 0;
                app.advise_detail = false;
                app.advise_scroll = 0;
                app.advise_status.clear();
                app.refresh_advise();
                app.focus = FocusArea::AdvisePanel;
            }
        }
        KeyCode::Char('t') => {
            app.show_terminal = !app.show_terminal;
        }
        KeyCode::Char('u') => {
            // Live layout-preset cycling (H1-05): pure geometry, so this
            // never touches pane state — open file, scrolls and messages
            // all survive the switch.
            let next = crate::layout::cycle_layout(&app.config.layout, true);
            app.config.layout = next.clone();
            app.save_config();
            app.push_toast(crate::toast::ToastKind::Info, format!("Layout: {next}"));
        }
        KeyCode::Char('h') => {
            if !app.health_check_in_progress {
                app.run_health_check(tx.clone());
            }
        }
        KeyCode::Char('f') => {
            app.focus = if app.focus == FocusArea::FeatureNavigator {
                FocusArea::ChatInput
            } else {
                FocusArea::FeatureNavigator
            };
        }
        KeyCode::Char('s') => {
            if app.focus == FocusArea::CodeEditor {
                app.save_editor();
            } else {
                app.focus = if app.focus == FocusArea::GitCommit {
                    FocusArea::ChatInput
                } else {
                    FocusArea::GitCommit
                };
            }
        }
        _ => return None,
    };
    Some(done())
}

// ── Normal mode ────────────────────────────────────────────────────────────

fn normal_key(app: &mut App, key: KeyEvent, tx: &Tx) -> KeyFlow {
    // Keys with no focus-specific meaning at all.
    match key.code {
        KeyCode::Tab => {
            // The Collaboration Hub owns Tab: it cycles the form field
            // instead of the body focus ring.
            if app.focus == FocusArea::CollaborationHub {
                app.collab_cycle_field();
                return done();
            }
            app.focus = next_body_focus(app);
            return done();
        }
        KeyCode::Esc => {
            on_esc(app);
            return done();
        }
        _ => {}
    }

    // The focused panel owns its keys first; unresolved keys fall through
    // to the global chords below.
    if focus_key(app, key, tx) {
        return done();
    }
    global_chord(app, key, tx)
}

/// Tab's ring: explorer → editor → chat, but only through the panes the
/// layout actually renders (H1-05) — tabbing to a pane hidden by chat-first
/// would leave the user staring at an unhighlighted screen. Zen is the
/// exception: it renders one pane at a time *from the focus*, so the full
/// ring is how the user flips between them. Before the first draw (or with
/// no layout recorded) every pane counts as visible, which is classic.
/// From any overlay panel, Tab lands back on the chat input.
fn next_body_focus(app: &App) -> FocusArea {
    use crate::focus::FocusArea::*;
    let ring = [FileExplorer, CodeEditor, ChatInput];
    if crate::layout::effective_layout(&app.config.layout) == "zen" {
        return match app.focus {
            FileExplorer => CodeEditor,
            CodeEditor => ChatInput,
            _ => FileExplorer,
        };
    }
    let is_visible = |f| match f {
        FileExplorer => app.last_layout.explorer.is_some(),
        CodeEditor => app.last_layout.editor.is_some(),
        ChatInput => app.last_layout.chat.is_some() || app.last_layout.input.is_some(),
        _ => false,
    };
    let active: Vec<FocusArea> = ring.iter().copied().filter(|f| is_visible(*f)).collect();
    let ring_ref: &[FocusArea] = if active.is_empty() { &ring } else { &active };
    match ring_ref.iter().position(|f| *f == app.focus) {
        Some(i) => ring_ref[(i + 1) % ring_ref.len()],
        None => ChatInput,
    }
}

/// Per-focus dispatch: each handler returns true if it consumed the key.
fn focus_key(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    match app.focus {
        FocusArea::Settings => key_settings(app, key, tx),
        FocusArea::FileExplorer => key_file_explorer(app, key),
        FocusArea::CodeEditor => key_code_editor(app, key),
        FocusArea::ModelSelector => key_model_selector(app, key, tx),
        FocusArea::ChatInput => key_chat(app, key),
        FocusArea::GitCommit => key_git_commit(app, key, tx),
        FocusArea::ByteBotPanel => key_bytebot(app, key, tx),
        FocusArea::SecurityAuditor => key_security(app, key, tx),
        FocusArea::CodeReview => key_code_review(app, key, tx),
        FocusArea::ReviewDashboard => key_review_dashboard(app, key),
        FocusArea::TaskManager => key_task_manager(app, key, tx),
        FocusArea::WorktreePanel => key_worktree_panel(app, key),
        FocusArea::AdvisePanel => key_advise_panel(app, key),
        FocusArea::ProviderHealth => key_provider_health(app, key),
        FocusArea::LearningMode => key_learning(app, key),
        FocusArea::CustomModels => key_custom_models(app, key),
        FocusArea::VoiceInterface => key_voice(app, key, tx),
        FocusArea::TerminalAssistant => key_terminal_assistant(app, key, tx),
        FocusArea::CollaborationHub => key_collab(app, key, tx),
        FocusArea::PerformanceProfiler => key_profiler(app, key, tx),
        FocusArea::FeatureNavigator => key_feature_nav(app, key),
        FocusArea::MultiLanguage => key_multi_language(app, key),
        _ => false,
    }
}

/// Global chords that apply wherever the focus handler left the key alone.
/// All are suppressed while a text field is being typed into (E2-06), so
/// the letters remain typable.
fn global_chord(app: &mut App, key: KeyEvent, tx: &Tx) -> KeyFlow {
    match key.code {
        KeyCode::Char('i') | KeyCode::Char('/') if !app.text_entry_active() => {
            app.input_mode = InputMode::Editing;
            app.focus = FocusArea::ChatInput;
        }
        KeyCode::Char('m') if !app.text_entry_active() => {
            app.focus = if app.focus == FocusArea::ModelSelector {
                FocusArea::ChatInput
            } else {
                app.refresh_models(tx.clone());
                FocusArea::ModelSelector
            };
        }
        KeyCode::Char('s') if !app.text_entry_active() => {
            app.focus = if app.focus == FocusArea::Settings {
                FocusArea::ChatInput
            } else {
                FocusArea::Settings
            };
        }
        KeyCode::Char('?') | KeyCode::F(1) if !app.text_entry_active() => {
            app.help_visible = true;
            app.help_scroll = 0;
        }
        KeyCode::Char('q') if !app.text_entry_active() => return quit(),
        _ => {}
    }
    done()
}

fn on_esc(app: &mut App) {
    if app.init_visible {
        app.init_visible = false;
        return;
    }
    match app.focus {
        FocusArea::Settings => {
            if app.settings_url_editing {
                app.settings_url_editing = false;
            } else {
                app.settings_reset_active = false;
                app.save_config();
                app.focus = FocusArea::ChatInput;
            }
        }
        FocusArea::WorktreePanel => {
            // Esc unwinds the prompt one stage, closes the panel at the list.
            match app.worktree_prompt {
                crate::focus::WorktreePrompt::None => app.focus = FocusArea::ChatInput,
                _ => cancel_worktree_prompt(app),
            }
        }
        FocusArea::AdvisePanel => {
            // Esc unwinds detail → list → chat.
            if app.advise_detail {
                app.advise_detail = false;
                app.advise_scroll = 0;
            } else {
                app.focus = FocusArea::ChatInput;
            }
        }
        FocusArea::CollaborationHub => {
            // Esc unwinds the hub: field editing → live session (hang up,
            // panel stays open idle) → close.
            if app.collab_editing {
                app.collab_editing = false;
            } else if app.collab_session_active {
                app.collab_disconnect();
            } else {
                app.focus = FocusArea::ChatInput;
            }
        }
        FocusArea::ModelSelector
        | FocusArea::CodeReview
        | FocusArea::PerformanceDashboard
        | FocusArea::ProviderHealth
        | FocusArea::ProjectAnalyzer
        | FocusArea::GitCommit
        | FocusArea::FeatureNavigator
        | FocusArea::ByteBotPanel
        | FocusArea::VoiceInterface
        | FocusArea::TerminalAssistant
        | FocusArea::SecurityAuditor
        | FocusArea::PerformanceProfiler
        | FocusArea::CustomModels
        | FocusArea::LearningMode
        | FocusArea::MultiLanguage
        | FocusArea::ReviewDashboard
        | FocusArea::TaskManager => {
            app.focus = FocusArea::ChatInput;
        }
        FocusArea::CodeEditor => {
            app.input_mode = InputMode::Normal;
        }
        _ => {}
    }
}

// ── Per-focus handlers ─────────────────────────────────────────────────────

fn key_chat(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.chat_scroll = app.chat_scroll.saturating_add(1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.chat_scroll = app.chat_scroll.saturating_sub(1);
        }
        _ => return false,
    }
    true
}

fn key_file_explorer(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            if app.selected_file > 0 {
                app.selected_file -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.selected_file + 1 < app.file_tree.len() {
                app.selected_file += 1;
            }
        }
        KeyCode::Enter => {
            if let Some(fp) = app.file_tree.get(app.selected_file).cloned() {
                app.open_file_in_editor(&fp);
            }
        }
        KeyCode::Char(' ') => {
            if let Some(fp) = app.file_tree.get(app.selected_file) {
                let fp = fp.clone();
                if app.attached_files.contains(&fp) {
                    app.attached_files.remove(&fp);
                } else {
                    app.attached_files.insert(fp);
                }
            }
        }
        _ => return false,
    }
    true
}

fn key_code_editor(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.editor.scroll((-1, 0));
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.editor.scroll((1, 0));
        }
        KeyCode::Char('e') => {
            app.input_mode = InputMode::Editing;
        }
        _ => return false,
    }
    true
}

fn key_model_selector(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            if app.selected_model > 0 {
                app.selected_model -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.selected_model + 1 < app.available_models.len() {
                app.selected_model += 1;
            }
        }
        KeyCode::Enter => {
            if let Some(model) = app.available_models.get(app.selected_model) {
                app.config.default_model = model.clone();
                let _ = app.config.save();
                // llama.cpp servers only serve their loaded model, so kick
                // off a server-side swap so the next generation uses it.
                if let Some(inner) = llama_model_target(model) {
                    app.llamacpp_control("switch", Some(inner.to_string()), tx.clone());
                }
                app.focus = FocusArea::ChatInput;
            }
        }
        KeyCode::Char('r') => {
            app.refresh_models(tx.clone());
        }
        KeyCode::Char('l') => {
            // Load the selected model via llama.cpp
            let target = app
                .available_models
                .get(app.selected_model)
                .and_then(|m| llama_model_target(m));
            app.llamacpp_control("load", target.map(|s| s.to_string()), tx.clone());
        }
        KeyCode::Char('u') => {
            app.llamacpp_control("unload", None, tx.clone());
        }
        _ => return false,
    }
    true
}

/// The row under the settings cursor (the list is a compile-time constant,
/// so the clamp can never panic).
fn settings_current(app: &App) -> &'static crate::focus::SettingRow {
    let items = crate::focus::SETTINGS_ITEMS;
    &items[app.settings_cursor.min(items.len() - 1)]
}

/// Rows whose value is typed into the buffer (Text/Number kinds). Only
/// these claim ←/→ for cursor movement while editing.
fn settings_row_typable(app: &App) -> bool {
    matches!(
        settings_current(app).kind,
        crate::focus::SettingKind::Text | crate::focus::SettingKind::Number
    )
}

fn key_settings(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    let row_count = crate::focus::SETTINGS_ITEMS.len();
    // While a value is being typed, every character belongs to the buffer —
    // including j/k, which would otherwise move the row cursor (E6-01).
    if app.settings_url_editing {
        match key.code {
            KeyCode::Char(c) => {
                app.settings_url_buffer.insert(app.settings_url_cursor, c);
                app.settings_url_cursor += 1;
            }
            KeyCode::Backspace if app.settings_url_cursor > 0 => {
                app.settings_url_cursor -= 1;
                app.settings_url_buffer.remove(app.settings_url_cursor);
            }
            KeyCode::Up => {
                // Moving rows abandons the uncommitted edit: the buffer
                // belongs to one row's field, and silently retargeting it
                // to whatever row the cursor landed on would commit it to
                // the wrong config value.
                app.settings_url_editing = false;
                if app.settings_cursor > 0 {
                    app.settings_cursor -= 1;
                }
            }
            KeyCode::Down => {
                app.settings_url_editing = false;
                if app.settings_cursor + 1 < row_count {
                    app.settings_cursor += 1;
                }
            }
            KeyCode::Left if settings_row_typable(app) && app.settings_url_cursor > 0 => {
                app.settings_url_cursor -= 1;
            }
            KeyCode::Right
                if settings_row_typable(app)
                    && app.settings_url_cursor < app.settings_url_buffer.len() =>
            {
                app.settings_url_cursor += 1;
            }
            KeyCode::Enter => settings_enter(app, tx),
            _ => return false,
        }
        return true;
    }
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            if app.settings_cursor > 0 {
                app.settings_cursor -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.settings_cursor + 1 < row_count {
                app.settings_cursor += 1;
            }
        }
        KeyCode::Enter => settings_enter(app, tx),
        KeyCode::Left => settings_step(app, -1),
        KeyCode::Right => settings_step(app, 1),
        _ => return false,
    }
    true
}

fn settings_enter(app: &mut App, tx: &Tx) {
    use crate::focus::SettingKind;
    let row_label = settings_current(app).label;
    if app.settings_url_editing {
        // Commit the edit for the row that owns the buffer.
        let buf = app.settings_url_buffer.clone();
        match row_label {
            "Ollama URL" => app.config.ollama_url = buf,
            "Llama.cpp URL" => app.config.llama_cpp_url = buf,
            "Llama.cpp Model" => app.config.llama_cpp_model_path = buf,
            "Llama Temp" => {
                app.config.llama_cpp_temperature =
                    buf.trim().parse::<f64>().ok().filter(|x| x.is_finite());
            }
            "Llama Top-K" => {
                app.config.llama_cpp_top_k = buf.trim().parse().ok();
            }
            "Llama Min-P" => {
                app.config.llama_cpp_min_p =
                    buf.trim().parse::<f64>().ok().filter(|x| x.is_finite());
            }
            "Llama Max Tokens" => {
                app.config.llama_cpp_max_tokens = buf.trim().parse().ok();
            }
            _ => {}
        }
        app.settings_url_editing = false;
        app.save_config();
        // Refresh models and health check when a URL/model path changed.
        if matches!(
            row_label,
            "Ollama URL" | "Llama.cpp URL" | "Llama.cpp Model"
        ) {
            app.refresh_models(tx.clone());
            app.run_health_check(tx.clone());
        }
        return;
    }
    match settings_current(app).kind {
        SettingKind::Text | SettingKind::Number => {
            // Start editing this row's value.
            app.settings_url_editing = true;
            app.settings_url_buffer = settings_edit_seed(app, row_label);
            app.settings_url_cursor = app.settings_url_buffer.len();
        }
        SettingKind::Action => {
            // Factory Reset
            app.config = XencodeConfig::default();
            app.theme = ThemeColors::get(&app.config.active_theme);
            app.style_chat_input();
            app.settings_reset_active = true;
            app.settings_cursor = 0;
            app.save_config();
            app.refresh_models(tx.clone());
            app.run_health_check(tx.clone());
            app.focus = FocusArea::ChatInput;
        }
        _ => {
            app.save_config();
            app.focus = FocusArea::ChatInput;
        }
    }
}

/// The current stored value for a typed-edit row, as the initial buffer.
fn settings_edit_seed(app: &App, label: &str) -> String {
    match label {
        "Ollama URL" => app.config.ollama_url.clone(),
        "Llama.cpp URL" => app.config.llama_cpp_url.clone(),
        "Llama.cpp Model" => app.config.llama_cpp_model_path.clone(),
        "Llama Temp" => app
            .config
            .llama_cpp_temperature
            .map(|v| v.to_string())
            .unwrap_or_default(),
        "Llama Top-K" => app
            .config
            .llama_cpp_top_k
            .map(|v| v.to_string())
            .unwrap_or_default(),
        "Llama Min-P" => app
            .config
            .llama_cpp_min_p
            .map(|v| v.to_string())
            .unwrap_or_default(),
        "Llama Max Tokens" => app
            .config
            .llama_cpp_max_tokens
            .map(|v| v.to_string())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// One ←/→ adjustment for an integer row: add/subtract `step`, clamped to
/// `[min, max]`; downward never crosses below the `min + step` floor the
/// historical per-row guards enforced.
fn stepped(v: u64, dir: i32, step: u64, min: u64, max: u64) -> u64 {
    if dir > 0 {
        v.saturating_add(step).min(max)
    } else if v >= min + step {
        v - step
    } else {
        v
    }
}

fn settings_cycle(app: &mut App, label: &str, options: &'static [&'static str], dir: i32) {
    let current = match label {
        "Theme" => app.config.active_theme.clone(),
        "Layout" => app.config.layout.clone(),
        "Agent Approval" => app.config.agent_approval.clone(),
        _ => return,
    };
    let len = options.len();
    let pos = options.iter().position(|o| *o == current).unwrap_or(0);
    let next = if dir > 0 {
        (pos + 1) % len
    } else {
        (pos + len - 1) % len
    };
    let value = options[next];
    match label {
        "Theme" => {
            app.config.active_theme = value.to_string();
            app.theme = ThemeColors::get(&app.config.active_theme);
            app.style_chat_input();
        }
        "Layout" => app.config.layout = value.to_string(),
        "Agent Approval" => app.config.agent_approval = value.to_string(),
        _ => {}
    }
}

fn settings_toggle(app: &mut App, label: &str) -> bool {
    let flag = match label {
        "Rounded Borders" => &mut app.config.rounded_borders,
        "Show Scrollbars" => &mut app.config.show_scrollbars,
        "Line Numbers" => &mut app.config.show_line_numbers,
        "Cache Enabled" => &mut app.config.cache_enabled,
        "Memory Enabled" => &mut app.config.memory_enabled,
        _ => return false,
    };
    *flag = !*flag;
    true
}

/// ←/→ on the row under the cursor. Every behavior derives from the row's
/// `SettingKind` in `focus::SETTINGS_ITEMS` — never from the row number.
fn settings_step(app: &mut App, dir: i32) {
    use crate::focus::SettingKind;
    let label = settings_current(app).label;
    match settings_current(app).kind {
        SettingKind::Cycle(options) => {
            settings_cycle(app, label, options, dir);
            app.save_config();
        }
        SettingKind::Toggle => {
            if settings_toggle(app, label) {
                app.save_config();
            }
        }
        SettingKind::Stepped { step, min, max } => {
            match label {
                "Max Cache Size" => {
                    app.config.max_cache_size =
                        stepped(app.config.max_cache_size as u64, dir, step, min, max) as usize
                }
                "Memory Items" => {
                    app.config.max_memory_items =
                        stepped(app.config.max_memory_items as u64, dir, step, min, max) as usize
                }
                "Response Timeout" => {
                    app.config.response_timeout =
                        stepped(app.config.response_timeout, dir, step, min, max)
                }
                _ => return,
            }
            app.save_config();
        }
        _ => {}
    }
}

fn key_git_commit(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    match key.code {
        KeyCode::Enter => {
            if !app.commit_message.trim().is_empty() {
                // Off the UI thread (E2-02): slow repos must not freeze the
                // TUI. The result arrives as a [GIT_COMMIT_*] chat line.
                let msg = app.commit_message.clone();
                app.commit_message.clear();
                app.commit_cursor = 0;
                app.focus = FocusArea::ChatInput;
                let ctx = tx.clone();
                tokio::spawn(async move {
                    let out = tokio::process::Command::new("git")
                        .args(["commit", "-am", &msg])
                        .output()
                        .await;
                    let (tag, body) = match out {
                        Ok(o) if o.status.success() => {
                            ("[GIT_COMMIT_OK]", first_output_line(&o.stdout))
                        }
                        Ok(o) => ("[GIT_COMMIT_ERR]", first_output_line(&o.stderr)),
                        Err(e) => ("[GIT_COMMIT_ERR]", e.to_string()),
                    };
                    let _ = ctx.send(format!("{tag}{body}"));
                });
            }
        }
        KeyCode::Char(c) => {
            // All characters, j/k included, go into the message (E6-01).
            app.commit_message.insert(app.commit_cursor, c);
            app.commit_cursor += 1;
        }
        KeyCode::Backspace if app.commit_cursor > 0 => {
            app.commit_cursor -= 1;
            app.commit_message.remove(app.commit_cursor);
        }
        KeyCode::Left if app.commit_cursor > 0 => {
            // Move cursor left; Backspace is the delete key.
            app.commit_cursor -= 1;
        }
        KeyCode::Right if app.commit_cursor < app.commit_message.len() => {
            app.commit_cursor += 1;
        }
        _ => return false,
    }
    true
}

fn key_bytebot(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    match key.code {
        KeyCode::Up if !app.bytebot_running && !app.bytebot_history.is_empty() => {
            app.bytebot_command = app.bytebot_history.last().unwrap().clone();
            app.bytebot_cursor = app.bytebot_command.len();
        }
        KeyCode::Enter => {
            // Executes what's typed; history recall is ↑.
            app.run_bytebot(tx.clone());
        }
        KeyCode::Char(c) => {
            app.bytebot_command.insert(app.bytebot_cursor, c);
            app.bytebot_cursor += 1;
        }
        KeyCode::Backspace if app.bytebot_cursor > 0 => {
            app.bytebot_cursor -= 1;
            app.bytebot_command.remove(app.bytebot_cursor);
        }
        KeyCode::Left if app.bytebot_cursor > 0 => {
            app.bytebot_cursor -= 1;
        }
        KeyCode::Right if app.bytebot_cursor < app.bytebot_command.len() => {
            app.bytebot_cursor += 1;
        }
        _ => return false,
    }
    true
}

fn key_security(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            if app.security_scroll > 0 {
                app.security_scroll -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.security_scroll += 1;
        }
        KeyCode::Enter => {
            if !app.sec_scan_active {
                app.start_security_scan(tx.clone());
            }
        }
        KeyCode::Char(' ') => {
            // Cycle severity filter
            app.sec_filter_severity = match app.sec_filter_severity.as_str() {
                "All" => "Critical",
                "Critical" => "High",
                "High" => "Medium",
                "Medium" => "Low",
                _ => "All",
            }
            .to_string();
        }
        // Sort toggle — previously shadowed by the global Settings `s`
        // (the E2-06 conflict, resolved by focus-first dispatch).
        KeyCode::Char('s') => {
            app.sec_sort_mode = if app.sec_sort_mode == "severity" {
                "category".to_string()
            } else {
                "severity".to_string()
            };
        }
        _ => return false,
    }
    true
}

fn key_code_review(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.review_scroll = app.review_scroll.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.review_scroll += 1;
        }
        KeyCode::Enter => {
            if !app.is_reviewing {
                app.submit_review(tx.clone());
            }
        }
        _ => return false,
    }
    true
}

fn key_review_dashboard(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => app.review_dash.move_selection(-1),
        KeyCode::Down | KeyCode::Char('j') => app.review_dash.move_selection(1),
        KeyCode::Enter => app.review_dash.reload(),
        // Toggle diff base: working tree <-> main.
        KeyCode::Char('b') => app.review_dash.toggle_base(),
        // Scroll the diff pane.
        KeyCode::Char('u') => app.review_dash.scroll_by(-10),
        KeyCode::Char('d') => app.review_dash.scroll_by(10),
        _ => return false,
    }
    true
}

fn key_task_manager(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    fn select(app: &mut App, delta: i32) {
        let count = app.tasks_snapshot().map(|t| t.len()).unwrap_or(0);
        let next = app.tasks_selected as i64 + delta as i64;
        app.tasks_selected = next.clamp(0, count.saturating_sub(1) as i64) as usize;
    }
    match key.code {
        KeyCode::Up | KeyCode::Char('k') if app.tasks_detail => {
            app.tasks_scroll = app.tasks_scroll.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') if app.tasks_detail => {
            app.tasks_scroll += 1;
        }
        KeyCode::Up | KeyCode::Char('k') => select(app, -1),
        KeyCode::Down | KeyCode::Char('j') => select(app, 1),
        KeyCode::Enter => {
            app.tasks_detail = !app.tasks_detail;
            app.tasks_scroll = 0;
        }
        KeyCode::Char('x') if !app.tasks_detail => {
            if let Some(id) = app
                .tasks_snapshot()
                .and_then(|t| t.get(app.tasks_selected).map(|r| r.id))
            {
                let _ = tx.send(format!("[TASKS]stop|{id}"));
            }
        }
        KeyCode::Char('d') if !app.tasks_detail => {
            if let Some(id) = app
                .tasks_snapshot()
                .and_then(|t| t.get(app.tasks_selected).map(|r| r.id))
            {
                let _ = tx.send(format!("[TASKS]rm|{id}"));
            }
        }
        _ => return false,
    }
    true
}

fn cancel_worktree_prompt(app: &mut App) {
    app.worktree_prompt = crate::focus::WorktreePrompt::None;
    app.worktree_path_buf.clear();
    app.worktree_branch_buf.clear();
}

/// WorktreePanel (D3-02): list keys are inert while a prompt is open —
/// the prompt swallows everything (typed buffers, Enter, Esc).
fn key_worktree_panel(app: &mut App, key: KeyEvent) -> bool {
    use crate::focus::WorktreePrompt::*;
    if app.worktree_prompt != None {
        return key_worktree_prompt(app, key);
    }
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.worktree_selected = app.worktree_selected.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.worktree_selected + 1 < app.worktrees.len() {
                app.worktree_selected += 1;
            }
        }
        KeyCode::Char('a') => {
            app.worktree_prompt = AddPath;
            app.worktree_path_buf.clear();
            app.worktree_branch_buf.clear();
            app.worktree_status.clear();
        }
        KeyCode::Char('d') => {
            if app.worktrees.is_empty() {
                return true;
            }
            app.worktree_prompt = ConfirmRemove;
            app.worktree_status.clear();
        }
        KeyCode::Char('r') => {
            app.worktree_status.clear();
            app.refresh_worktrees();
        }
        _ => return false,
    }
    true
}

fn key_worktree_prompt(app: &mut App, key: KeyEvent) -> bool {
    use crate::focus::WorktreePrompt::*;
    match (app.worktree_prompt, key.code) {
        (_, KeyCode::Esc) => cancel_worktree_prompt(app),
        (ConfirmRemove, KeyCode::Char('y')) => app.worktree_do_remove(),
        (ConfirmRemove, KeyCode::Char('n')) => app.worktree_prompt = None,
        (AddPath, KeyCode::Enter) => {
            if app.worktree_path_buf.trim().is_empty() {
                app.worktree_status = "path must not be empty".to_string();
            } else {
                app.worktree_prompt = AddBranch;
            }
        }
        (AddBranch, KeyCode::Enter) => app.worktree_do_add(),
        (_, KeyCode::Enter) => app.worktree_prompt = None,
        (_, KeyCode::Backspace) => {
            let buf = match app.worktree_prompt {
                AddPath => &mut app.worktree_path_buf,
                _ => &mut app.worktree_branch_buf,
            };
            buf.pop();
        }
        (_, KeyCode::Char(c)) => {
            let buf = match app.worktree_prompt {
                AddPath => &mut app.worktree_path_buf,
                _ => &mut app.worktree_branch_buf,
            };
            buf.push(c);
        }
        _ => {}
    }
    true
}

/// AdvisePanel (F2-01): a read-only insights list. Enter toggles the detail
/// view (where j/k scroll the message), `o` opens the advised file in the
/// editor, `r` recomputes from the live snapshot.
fn key_advise_panel(app: &mut App, key: KeyEvent) -> bool {
    let count = app.advise_items.len();
    match key.code {
        KeyCode::Up | KeyCode::Char('k') if app.advise_detail => {
            app.advise_scroll = app.advise_scroll.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') if app.advise_detail => {
            app.advise_scroll += 1;
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.advise_selected = app.advise_selected.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.advise_selected + 1 < count {
                app.advise_selected += 1;
            }
        }
        KeyCode::Enter => {
            app.advise_detail = !app.advise_detail;
            app.advise_scroll = 0;
        }
        KeyCode::Char('r') => {
            app.advise_selected = 0;
            app.advise_detail = false;
            app.advise_scroll = 0;
            app.advise_status.clear();
            app.refresh_advise();
        }
        KeyCode::Char('o') if !app.advise_detail => {
            if let Some(path) = app
                .advise_items
                .get(app.advise_selected)
                .map(|a| a.file.clone())
            {
                app.open_file_in_editor(&path);
            }
        }
        _ => return false,
    }
    true
}

fn key_provider_health(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            if app.provider_health_scroll > 0 {
                app.provider_health_scroll -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.provider_health_scroll += 1;
        }
        _ => return false,
    }
    true
}

fn key_learning(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Enter => {
            if !app.learn_active {
                app.start_learning_mode();
            } else if app.learn_quiz_active && !app.learn_quiz_answered {
                // Check quiz answer
                app.learn_quiz_answered = true;
                // Simple check: first option is correct
                app.learn_quiz_correct = app.learn_quiz_selected == 0;
                if app.learn_quiz_correct {
                    app.learn_progress_pct = (app.learn_progress_pct + 20.0).min(100.0);
                }
            }
        }
        KeyCode::Left
            if app.learn_quiz_active && !app.learn_quiz_answered && app.learn_quiz_selected > 0 =>
        {
            app.learn_quiz_selected -= 1;
        }
        KeyCode::Right
            if app.learn_quiz_active
                && !app.learn_quiz_answered
                && app.learn_quiz_selected + 1 < app.learn_quiz_options.len() =>
        {
            app.learn_quiz_selected += 1;
        }
        _ => return false,
    }
    true
}

fn key_custom_models(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            if app.models_selected > 0 {
                app.models_selected -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.models_selected + 1 < app.models_profiles.len() {
                app.models_selected += 1;
            }
        }
        KeyCode::Enter => {
            if !app.models_editing {
                app.start_custom_models();
            }
        }
        KeyCode::Left if app.models_editing && app.models_selected > 0 => {
            app.models_selected -= 1;
        }
        KeyCode::Right
            if app.models_editing && app.models_selected + 1 < app.models_profiles.len() =>
        {
            app.models_selected += 1;
        }
        // Save profile — previously shadowed by the global Settings `s`.
        KeyCode::Char('s') if app.models_editing => {
            app.models_saving = true;
            app.models_test_output = "Profile saved!".to_string();
        }
        _ => return false,
    }
    true
}

fn key_voice(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    match key.code {
        KeyCode::Enter => {
            if !app.voice_active {
                app.start_voice_session(tx.clone());
            }
        }
        // Both Space (status bar hint) and `m` toggle mute (E2-06/E6-01:
        // focus-first dispatch means the global `m` never fires here).
        KeyCode::Char(' ') | KeyCode::Char('m') => {
            app.voice_muted = !app.voice_muted;
        }
        _ => return false,
    }
    true
}

fn key_terminal_assistant(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    match key.code {
        KeyCode::Enter => {
            if !app.term_asst_active {
                app.start_terminal_assistant(tx.clone());
            }
        }
        KeyCode::Char(' ') => {
            // Cycle risk filter
            app.term_risk_filter = match app.term_risk_filter.as_str() {
                "All" => "Safe",
                "Safe" => "Destructive",
                _ => "All",
            }
            .to_string();
        }
        _ => return false,
    }
    true
}

/// Collaboration Hub (G3-02): a real client, so the keys are a form, not a
/// toy. Idle: `c` creates a session (server assigns the id), `j` edits the
/// session id to join, `Enter`/`Tab` connect or edit fields. Live: only `r`
/// (retry) and `Esc` (disconnect, handled in `on_esc`) do anything.
fn key_collab(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    if app.collab_editing {
        match key.code {
            KeyCode::Enter => app.collab_editing = false,
            KeyCode::Backspace => app.collab_edit_backspace(),
            KeyCode::Char(c) => app.collab_edit_char(c),
            _ => {}
        }
        return true;
    }
    match key.code {
        KeyCode::Char('c') if !app.collab_session_active => {
            // "create" means "connect and let the server name the session".
            app.collab_session_id.clear();
            app.start_collab_session(tx.clone());
        }
        KeyCode::Char('j') if !app.collab_session_active => {
            app.collab_field = crate::focus::CollabField::Session;
            app.collab_editing = true;
        }
        KeyCode::Enter if !app.collab_session_active => {
            app.start_collab_session(tx.clone());
        }
        KeyCode::Char('r') => app.collab_retry(tx.clone()),
        _ => return false,
    }
    true
}

fn key_profiler(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    if key.code == KeyCode::Enter && !app.profiler_active {
        app.start_profiler(tx.clone());
        return true;
    }
    false
}

fn key_multi_language(app: &mut App, key: KeyEvent) -> bool {
    if key.code == KeyCode::Enter && !app.lang_active {
        app.start_multi_language();
        return true;
    }
    false
}

fn key_feature_nav(app: &mut App, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            if app.feature_nav_selected > 0 {
                app.feature_nav_selected -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.feature_nav_selected + 1 < FEATURE_LIST.len() {
                app.feature_nav_selected += 1;
            }
        }
        KeyCode::Enter => {
            app.focus = navigate_feature(app.feature_nav_selected);
        }
        _ => return false,
    }
    true
}

// ── Editing mode ───────────────────────────────────────────────────────────

fn editing_key(app: &mut App, key: KeyEvent, tx: &Tx) -> KeyFlow {
    // If the code editor is focused, forward input to its textarea.
    if app.focus == FocusArea::CodeEditor {
        match key.code {
            KeyCode::Esc => {
                app.input_mode = InputMode::Normal;
            }
            _ => {
                app.editor.input(key);
                app.editor_dirty = true;
            }
        }
        return done();
    }
    // Chat input: tui_textarea owns cursor/edits. Enter sends; Alt+Enter
    // (or Ctrl+J, which survives terminals that mangle Alt) adds a newline
    // instead.
    match key.code {
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => {
            app.chat_input.insert_newline();
        }
        KeyCode::Char('j') if key.modifiers == KeyModifiers::CONTROL => {
            app.chat_input.insert_newline();
        }
        KeyCode::Enter => {
            if !app.is_generating {
                app.submit_message(tx.clone());
                app.chat_scroll = 0;
            }
        }
        KeyCode::Esc => {
            app.input_mode = InputMode::Normal;
        }
        KeyCode::Up if key.modifiers.contains(KeyModifiers::ALT) => {
            app.recall_history(-1);
        }
        KeyCode::Down if key.modifiers.contains(KeyModifiers::ALT) => {
            app.recall_history(1);
        }
        KeyCode::Tab => {
            if !app.complete_slash_draft() {
                app.chat_input.insert_str("    ");
            }
        }
        _ => {
            app.chat_input.input(key);
        }
    }
    done()
}

#[cfg(test)]
mod tests {
    use super::{handle_key, KeyFlow};
    use crate::app::{App, InputMode};
    use crate::focus::FocusArea;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use tokio::sync::mpsc;
    use xencode_config_rs::XencodeConfig;

    fn press(app: &mut App, code: KeyCode) -> KeyFlow {
        let (tx, _rx) = mpsc::unbounded_channel();
        handle_key(app, KeyEvent::new(code, KeyModifiers::NONE), &tx)
    }

    fn press_with_mods(app: &mut App, code: KeyCode, mods: KeyModifiers) -> KeyFlow {
        let (tx, _rx) = mpsc::unbounded_channel();
        handle_key(app, KeyEvent::new(code, mods), &tx)
    }

    fn app_with(focus: FocusArea) -> App<'static> {
        let mut app = App::new();
        app.focus = focus;
        app.input_mode = InputMode::Normal;
        app
    }

    #[test]
    fn tab_cycles_body_focus() {
        let mut app = app_with(FocusArea::ChatInput);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::FileExplorer);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::CodeEditor);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }

    #[test]
    fn tab_ring_skips_panes_the_layout_hides() {
        use crate::layout::compute_layout;
        let mut app = app_with(FocusArea::ChatInput);
        app.config.layout = "chat-first".into();
        app.last_layout = compute_layout(
            ratatui::layout::Rect::new(0, 1, 80, 22),
            "chat-first",
            false,
            FocusArea::ChatInput,
        );
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::CodeEditor);
        press(&mut app, KeyCode::Tab);
        assert_eq!(
            app.focus,
            FocusArea::ChatInput,
            "the hidden explorer must stay out of the ring"
        );
    }

    #[test]
    fn tab_flips_through_zen_panes() {
        // Zen shows one pane at a time *following the focus*, so its ring is
        // deliberately the full one: Tab is how you flip panes.
        use crate::layout::compute_layout;
        let mut app = app_with(FocusArea::ChatInput);
        app.config.layout = "zen".into();
        app.last_layout = compute_layout(
            ratatui::layout::Rect::new(0, 1, 80, 22),
            "zen",
            false,
            FocusArea::ChatInput,
        );
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::FileExplorer);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::CodeEditor);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }

    /// H1-10 pin: the help overlay advertises Ctrl+U, so the chord must
    /// actually be handled — help and keymap may not drift apart.
    #[test]
    fn ctrl_u_is_documented_and_handled() {
        assert!(
            crate::help::GLOBAL.iter().any(|(key, _)| *key == "Ctrl+U"),
            "Ctrl+U missing from the GLOBAL help table"
        );
        let mut app = app_with(FocusArea::ChatInput);
        let start = app.config.layout.clone();
        press_with_mods(&mut app, KeyCode::Char('u'), KeyModifiers::CONTROL);
        assert_ne!(app.config.layout, start, "Ctrl+U did not cycle the layout");
    }

    #[test]
    fn plain_q_quits_but_not_while_typing() {
        let mut app = app_with(FocusArea::ChatInput);
        assert_eq!(press(&mut app, KeyCode::Char('q')), KeyFlow::Quit);
        let mut app = app_with(FocusArea::GitCommit);
        assert_eq!(press(&mut app, KeyCode::Char('q')), KeyFlow::Continue);
        assert_eq!(app.commit_message, "q");
    }

    #[test]
    fn help_overlay_is_modal_and_swallows_quit_chords() {
        let mut app = app_with(FocusArea::ChatInput);
        press(&mut app, KeyCode::Char('?'));
        assert!(app.help_visible);
        assert_eq!(press(&mut app, KeyCode::Char('q')), KeyFlow::Continue);
        assert_eq!(press(&mut app, KeyCode::Char('?')), KeyFlow::Continue);
        assert!(!app.help_visible);
    }

    /// Queue an approval prompt the way the tool task would, and hand back
    /// the receiving end so a test can assert what the task was woken with.
    fn queue_approval(
        app: &mut App,
        tool: &str,
        class: crate::agent_tools::ToolClass,
    ) -> tokio::sync::oneshot::Receiver<crate::agent_tools::ApprovalAnswer> {
        use crate::agent_tools::ApprovalRequest;
        let (responder, answer) = tokio::sync::oneshot::channel();
        app.approval_queue.push_back((
            ApprovalRequest {
                tool: tool.into(),
                class,
                summary: format!("{tool} src/lib.rs"),
                preview: "+fn hello() {}\n".into(),
            },
            responder,
        ));
        answer
    }

    #[test]
    fn approval_modal_is_topmost_and_swallows_quit_chords() {
        use crate::agent_tools::{ApprovalAnswer, ToolClass};
        let mut app = app_with(FocusArea::ChatInput);
        let mut answer = queue_approval(&mut app, "write_file", ToolClass::Edit);
        // Help can open underneath, but the prompt still owns the keys.
        press(&mut app, KeyCode::Char('?'));
        assert_eq!(press(&mut app, KeyCode::Char('q')), KeyFlow::Continue);
        assert_eq!(
            press_with_mods(&mut app, KeyCode::Char('c'), KeyModifiers::CONTROL),
            KeyFlow::Continue
        );
        assert!(
            !app.help_visible,
            "q/Ctrl+C/? must not reach the handlers below the prompt"
        );
        assert_eq!(
            app.pending_approval().map(|r| r.tool.as_str()),
            Some("write_file")
        );
        // Enter is deliberately neither an allow nor a deny.
        assert_eq!(press(&mut app, KeyCode::Enter), KeyFlow::Continue);
        assert!(answer.try_recv().is_err());
        assert_eq!(press(&mut app, KeyCode::Char('y')), KeyFlow::Continue);
        assert_eq!(answer.try_recv(), Ok(ApprovalAnswer::Approved));
        assert!(app.pending_approval().is_none());
    }

    #[test]
    fn approval_keys_answer_in_queue_order_and_record_the_transcript() {
        use crate::agent_tools::{ApprovalAnswer, ToolClass};
        let mut app = app_with(FocusArea::ChatInput);
        let mut first = queue_approval(&mut app, "write_file", ToolClass::Edit);
        let mut second = queue_approval(&mut app, "edit_file", ToolClass::Edit);
        assert_eq!(press(&mut app, KeyCode::Char('n')), KeyFlow::Continue);
        assert_eq!(first.try_recv(), Ok(ApprovalAnswer::Denied));
        assert_eq!(
            app.pending_approval().map(|r| r.tool.as_str()),
            Some("edit_file"),
            "the queue must stay FIFO"
        );
        assert_eq!(press(&mut app, KeyCode::Esc), KeyFlow::Continue);
        assert_eq!(second.try_recv(), Ok(ApprovalAnswer::Denied));
        let logged: Vec<&str> = app
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(
            logged,
            vec![
                "⚙ write_file src/lib.rs · denied",
                "⚙ edit_file src/lib.rs · denied"
            ]
        );
    }

    #[test]
    fn approval_a_grants_the_tool_class_for_the_session_only() {
        use crate::agent_tools::{ApprovalAnswer, ToolClass};
        let mut app = app_with(FocusArea::ChatInput);
        let mut answer = queue_approval(&mut app, "edit_file", ToolClass::Edit);
        assert_eq!(press(&mut app, KeyCode::Char('a')), KeyFlow::Continue);
        assert_eq!(answer.try_recv(), Ok(ApprovalAnswer::ApprovedForSession));
        assert!(app.agent_grants.contains(&ToolClass::Edit));
        // Shell commands are a separate class: granting edits says nothing
        // about running commands.
        assert!(!app.agent_grants.contains(&ToolClass::Shell));
        assert!(app.pending_approval().is_none());
        // Answering an empty queue is a no-op, not a panic.
        assert_eq!(press(&mut app, KeyCode::Char('y')), KeyFlow::Continue);
    }

    #[test]
    fn approval_scroll_keys_page_the_diff_and_reset_on_answer() {
        use crate::agent_tools::ToolClass;
        let mut app = app_with(FocusArea::ChatInput);
        let _answer = queue_approval(&mut app, "write_file", ToolClass::Edit);
        press(&mut app, KeyCode::Char('j'));
        press(&mut app, KeyCode::PageDown);
        assert_eq!(app.approval_scroll, 2);
        press(&mut app, KeyCode::Char('k'));
        assert_eq!(app.approval_scroll, 1);
        press(&mut app, KeyCode::Char('y'));
        assert_eq!(app.approval_scroll, 0);
        // Up at the top saturates instead of underflowing.
        let _answer = queue_approval(&mut app, "write_file", ToolClass::Edit);
        press(&mut app, KeyCode::Up);
        assert_eq!(app.approval_scroll, 0);
    }

    #[test]
    fn chat_k_j_scroll_transcript() {
        let mut app = app_with(FocusArea::ChatInput);
        press(&mut app, KeyCode::Char('k'));
        assert_eq!(app.chat_scroll, 1);
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.chat_scroll, 0);
    }

    #[test]
    fn security_s_toggles_sort_instead_of_opening_settings() {
        // The E2-06 conflict: global Settings `s` used to shadow this.
        let mut app = app_with(FocusArea::SecurityAuditor);
        assert_eq!(app.sec_sort_mode, "severity");
        press(&mut app, KeyCode::Char('s'));
        assert_eq!(app.focus, FocusArea::SecurityAuditor);
        assert_eq!(app.sec_sort_mode, "category");
        // Space still cycles the severity filter.
        press(&mut app, KeyCode::Char(' '));
        assert_eq!(app.sec_filter_severity, "Critical");
    }

    #[test]
    fn custom_models_s_saves_only_while_editing() {
        let mut app = app_with(FocusArea::CustomModels);
        app.models_editing = false;
        press(&mut app, KeyCode::Char('s'));
        // Not editing: `s` keeps its global meaning.
        assert_eq!(app.focus, FocusArea::Settings);

        let mut app = app_with(FocusArea::CustomModels);
        app.models_editing = true;
        press(&mut app, KeyCode::Char('s'));
        assert_eq!(app.focus, FocusArea::CustomModels);
        assert!(app.models_saving);
    }

    #[test]
    fn commit_message_is_fully_typable_including_j_k_and_globals() {
        let mut app = app_with(FocusArea::GitCommit);
        for c in "jail/m s?".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.commit_message, "jail/m s?");
        // Backspace deletes, Left only moves the cursor (E2-06): with the
        // cursor one step back, the deleted char is the `s`, not the `?`.
        press(&mut app, KeyCode::Left);
        assert_eq!(app.commit_cursor, app.commit_message.len() - 1);
        press(&mut app, KeyCode::Backspace);
        assert_eq!(app.commit_message, "jail/m ?");
        // Cursor at end: one more Backspace removes the `?`.
        press(&mut app, KeyCode::Right);
        press(&mut app, KeyCode::Backspace);
        assert_eq!(app.commit_message, "jail/m ");
    }

    #[test]
    fn settings_url_editing_types_j_k_instead_of_moving_row_cursor() {
        let mut app = app_with(FocusArea::Settings);
        let url_row = crate::focus::settings_row_index("Ollama URL");
        app.settings_cursor = url_row;
        app.settings_url_editing = true;
        app.settings_url_buffer = "http://localhost:11434".to_string();
        app.settings_url_cursor = app.settings_url_buffer.len();
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.settings_url_buffer, "http://localhost:11434j");
        assert_eq!(app.settings_cursor, url_row);
    }

    #[test]
    fn settings_table_shape_is_stable() {
        use crate::focus::{SettingKind, SETTINGS_ITEMS};
        assert_eq!(SETTINGS_ITEMS.first().unwrap().label, "Theme");
        assert_eq!(SETTINGS_ITEMS.last().unwrap().label, "Factory Reset");
        assert_eq!(SETTINGS_ITEMS.last().unwrap().kind, SettingKind::Action);
        // The H1-04 polish rows sit right after Theme in Display.
        let display: Vec<&str> = SETTINGS_ITEMS
            .iter()
            .filter(|r| r.section == "Display")
            .map(|r| r.label)
            .collect();
        assert_eq!(
            display,
            [
                "Theme",
                "Layout",
                "Rounded Borders",
                "Show Scrollbars",
                "Line Numbers"
            ]
        );
        // I1-01: the agent policy row is a three-option Cycle in its own section.
        let agent = SETTINGS_ITEMS
            .iter()
            .find(|r| r.label == "Agent Approval")
            .expect("Agent Approval row exists");
        assert_eq!(agent.section, "Agent");
        assert_eq!(
            agent.kind,
            SettingKind::Cycle(crate::agent_tools::APPROVAL_MODE_NAMES)
        );
    }

    #[test]
    fn settings_edit_moving_rows_abandons_the_buffer() {
        // The edit buffer belongs to one row's field; rowing away must not
        // leave it armed to commit into whatever row the cursor lands on.
        let mut app = app_with(FocusArea::Settings);
        app.settings_cursor = crate::focus::settings_row_index("Ollama URL");
        press(&mut app, KeyCode::Enter);
        assert!(app.settings_url_editing);
        let original = app.config.ollama_url.clone();
        press(&mut app, KeyCode::Char('X'));
        press(&mut app, KeyCode::Down);
        assert!(!app.settings_url_editing);
        assert_eq!(app.config.ollama_url, original, "uncommitted edit leaked");
        assert_eq!(
            app.settings_cursor,
            crate::focus::settings_row_index("Llama.cpp URL")
        );
    }

    #[test]
    fn settings_steps_persist_through_the_config_dir() {
        // The only test that touches XCODE_CONFIG_DIR (process-global):
        // pointing saves at a temp dir means this test — and any concurrent
        // save — never writes the user's real ~/.xencode.
        use crate::focus::{settings_row_index, SETTINGS_ITEMS};

        let dir =
            std::env::temp_dir().join(format!("xencode-settings-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("XCODE_CONFIG_DIR", &dir);

        let mut app = app_with(FocusArea::Settings);
        app.config = XencodeConfig::default();

        // Layout row: → cycles presets and wraps back around.
        app.settings_cursor = settings_row_index("Layout");
        for _ in 0..crate::layout::LAYOUT_NAMES.len() {
            press(&mut app, KeyCode::Right);
        }
        assert_eq!(app.config.layout, "classic", "cycle must wrap");
        press(&mut app, KeyCode::Right);
        assert_eq!(app.config.layout, "chat-first");

        // A typo'd choice snaps into the cycle instead of wedging the row.
        app.config.layout = "bogus".into();
        press(&mut app, KeyCode::Right);
        assert_eq!(app.config.layout, "chat-first");

        // Toggles flip.
        app.settings_cursor = settings_row_index("Rounded Borders");
        press(&mut app, KeyCode::Right);
        assert!(app.config.rounded_borders);

        // Agent Approval cycles the three modes and wraps (I1-01).
        app.settings_cursor = settings_row_index("Agent Approval");
        assert_eq!(app.config.agent_approval, "ask");
        for expected in ["edit-allow", "all-allow", "ask", "edit-allow"] {
            press(&mut app, KeyCode::Right);
            assert_eq!(app.config.agent_approval, expected, "cycle must wrap");
        }

        // Stepped rows clamp at the floor.
        app.settings_cursor = settings_row_index("Response Timeout");
        app.config.response_timeout = 10;
        press(&mut app, KeyCode::Left);
        assert_eq!(app.config.response_timeout, 5);
        press(&mut app, KeyCode::Left);
        assert_eq!(app.config.response_timeout, 5, "stepped below min");

        // Navigation bounds derive from the table.
        app.settings_cursor = SETTINGS_ITEMS.len() - 1;
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.settings_cursor, SETTINGS_ITEMS.len() - 1);

        // The Ctrl+U chord cycles presets live and toasts the name (H1-05).
        press_with_mods(&mut app, KeyCode::Char('u'), KeyModifiers::CONTROL);
        assert_eq!(app.config.layout, "zen", "chat-first → zen");
        assert!(
            app.toasts.iter().any(|t| t.message.contains("zen")),
            "the chord must toast the new layout"
        );

        // Esc closes Settings and persists everything above.
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.focus, FocusArea::ChatInput);
        let saved = XencodeConfig::load_from(dir.join("config.json")).unwrap();
        assert_eq!(saved.layout, "zen");
        assert!(saved.rounded_borders);
        assert_eq!(saved.response_timeout, 5);
        assert_eq!(saved.agent_approval, "edit-allow");

        // The chord keeps cycling (and saving) from the chat pane.
        press_with_mods(&mut app, KeyCode::Char('u'), KeyModifiers::CONTROL);
        assert_eq!(app.config.layout, "classic", "zen wraps to classic");
        let saved = XencodeConfig::load_from(dir.join("config.json")).unwrap();
        assert_eq!(saved.layout, "classic");

        std::env::remove_var("XCODE_CONFIG_DIR");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn voice_m_mutes_without_opening_model_selector() {
        let mut app = app_with(FocusArea::VoiceInterface);
        press(&mut app, KeyCode::Char('m'));
        assert!(app.voice_muted);
        assert_eq!(app.focus, FocusArea::VoiceInterface);
    }

    #[test]
    fn esc_closes_panels_and_dismisses_init_overlay_first() {
        // ModelSelector chosen over Settings: Settings' Esc persists config,
        // and unit tests must not touch the user's config file.
        let mut app = app_with(FocusArea::ModelSelector);
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.focus, FocusArea::ChatInput);

        let mut app = app_with(FocusArea::ModelSelector);
        app.init_visible = true;
        press(&mut app, KeyCode::Esc);
        assert!(!app.init_visible);
        assert_eq!(app.focus, FocusArea::ModelSelector);
    }

    #[test]
    fn ctrl_chords_work_in_every_mode() {
        // Ctrl+T toggles the embedded terminal even in chat Editing mode.
        let mut app = app_with(FocusArea::ChatInput);
        app.input_mode = InputMode::Editing;
        press_with_mods(&mut app, KeyCode::Char('t'), KeyModifiers::CONTROL);
        assert!(app.show_terminal);
        assert_eq!(app.input_mode, InputMode::Editing);
        assert_eq!(
            press_with_mods(&mut app, KeyCode::Char('c'), KeyModifiers::CONTROL),
            KeyFlow::Quit
        );
    }

    #[test]
    fn ctrl_j_inserts_a_newline_in_chat_editing() {
        let mut app = app_with(FocusArea::ChatInput);
        app.input_mode = InputMode::Editing;
        press_with_mods(&mut app, KeyCode::Char('j'), KeyModifiers::CONTROL);
        assert_eq!(app.chat_input.lines().len(), 2);
    }

    #[test]
    fn ctrl_k_toggles_task_panel_and_resets_cursor() {
        let mut app = app_with(FocusArea::ChatInput);
        app.tasks_selected = 4;
        app.tasks_detail = true;
        press_with_mods(&mut app, KeyCode::Char('k'), KeyModifiers::CONTROL);
        assert_eq!(app.focus, FocusArea::TaskManager);
        assert_eq!(
            (app.tasks_selected, app.tasks_detail, app.tasks_scroll),
            (0, false, 0)
        );
        press_with_mods(&mut app, KeyCode::Char('k'), KeyModifiers::CONTROL);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }

    #[test]
    fn task_panel_enter_toggles_detail_and_arrows_change_role() {
        let mut app = app_with(FocusArea::TaskManager);
        press(&mut app, KeyCode::Enter);
        assert!(app.tasks_detail);
        // In detail view j/k scroll the output, not the selection.
        press(&mut app, KeyCode::Char('j'));
        assert_eq!((app.tasks_scroll, app.tasks_selected), (1, 0));
        press(&mut app, KeyCode::Enter);
        assert!(!app.tasks_detail);
        // Empty registry: list selection stays pinned at 0.
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.tasks_selected, 0);
        // Esc closes like any panel.
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }

    #[tokio::test]
    async fn task_panel_x_and_d_target_the_selected_task() {
        let mut app = app_with(FocusArea::TaskManager);
        app.task_runtime
            .lock()
            .await
            .start("sleeper", "sleep 30")
            .await
            .unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let key = |code| crossterm::event::KeyEvent::new(code, KeyModifiers::NONE);
        handle_key(&mut app, key(KeyCode::Char('x')), &tx);
        assert_eq!(rx.try_recv().unwrap(), "[TASKS]stop|1");
        handle_key(&mut app, key(KeyCode::Char('d')), &tx);
        assert_eq!(rx.try_recv().unwrap(), "[TASKS]rm|1");
        // Detail view: x/d are not list actions and must not fire.
        app.tasks_detail = true;
        handle_key(&mut app, key(KeyCode::Char('d')), &tx);
        assert!(rx.try_recv().is_err());
        app.task_runtime.lock().await.stop(1).await.unwrap();
    }

    fn worktree(path: &str, branch: &str, main: bool) -> xencode_context_rs::WorktreeInfo {
        xencode_context_rs::WorktreeInfo {
            path: std::path::PathBuf::from(path),
            head: "0123456789abcdef0123456789abcdef01234567".into(),
            branch: Some(branch.into()),
            detached: false,
            bare: false,
            locked: None,
            prunable: None,
            is_main: main,
        }
    }

    #[test]
    fn ctrl_o_toggles_worktree_panel_and_prompts_reset() {
        let mut app = app_with(FocusArea::ChatInput);
        app.worktree_prompt = crate::focus::WorktreePrompt::AddPath;
        press_with_mods(&mut app, KeyCode::Char('o'), KeyModifiers::CONTROL);
        assert_eq!(app.focus, FocusArea::WorktreePanel);
        assert_eq!(app.worktree_prompt, crate::focus::WorktreePrompt::None);
        // Read-only listing of the repo the tests run in: at least the main
        // worktree, and no side effects.
        assert!(!app.worktrees.is_empty());
        press_with_mods(&mut app, KeyCode::Char('o'), KeyModifiers::CONTROL);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }

    #[test]
    fn worktree_add_prompt_captures_text_and_esc_cancels() {
        let mut app = app_with(FocusArea::WorktreePanel);
        app.worktrees = vec![worktree("/repo", "main", true)];
        press(&mut app, KeyCode::Char('a'));
        assert_eq!(app.worktree_prompt, crate::focus::WorktreePrompt::AddPath);
        // Letters go to the buffer, not to global shortcuts ('q' must not quit).
        for c in "feature".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.worktree_path_buf, "feature");
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.worktree_prompt, crate::focus::WorktreePrompt::AddBranch);
        press(&mut app, KeyCode::Backspace);
        assert!(app.worktree_branch_buf.is_empty());
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.worktree_prompt, crate::focus::WorktreePrompt::None);
        assert!(app.worktree_path_buf.is_empty());
        // Panel itself stays open; Esc at the list closes it.
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }

    #[test]
    fn worktree_remove_refuses_main_and_confirms_others() {
        let mut app = app_with(FocusArea::WorktreePanel);
        app.worktrees = vec![
            worktree("/repo", "main", true),
            worktree("/repo-wt", "feat", false),
        ];
        app.worktree_dirty = vec![false, true];
        press(&mut app, KeyCode::Down);
        assert_eq!(app.worktree_selected, 1);
        press(&mut app, KeyCode::Up);
        assert_eq!(app.worktree_selected, 0);
        // Main: 'y' is refused before any git call happens.
        press(&mut app, KeyCode::Char('d'));
        press(&mut app, KeyCode::Char('y'));
        assert_eq!(app.worktree_status, "main worktree is not removable");
        assert_eq!(app.worktrees.len(), 2);
        // 'n' cancels without touching git.
        press(&mut app, KeyCode::Char('d'));
        press(&mut app, KeyCode::Char('n'));
        assert_eq!(app.worktree_prompt, crate::focus::WorktreePrompt::None);
    }

    #[tokio::test]
    async fn collab_create_connects_and_esc_hangs_up() {
        let mut app = app_with(FocusArea::CollaborationHub);
        press(&mut app, KeyCode::Char('c'));
        assert!(app.collab_session_active);
        assert!(app.collab_worker.is_some());
        // "create" leaves the id empty: the server names the session and
        // reports it back through the session: token.
        assert!(app.collab_session_id.is_empty());
        // Live already: neither 'c' nor Enter starts a second connection.
        press(&mut app, KeyCode::Char('c'));
        press(&mut app, KeyCode::Enter);
        assert!(app.collab_session_active && app.collab_worker.is_some());
        // Esc disconnects but keeps the panel open…
        press(&mut app, KeyCode::Esc);
        assert!(!app.collab_session_active);
        assert!(app.collab_worker.is_none());
        assert_eq!(app.focus, FocusArea::CollaborationHub);
        // …'r' reconnects…
        press(&mut app, KeyCode::Char('r'));
        assert!(app.collab_session_active && app.collab_worker.is_some());
        // …and Ctrl+W closes the panel without orphaning the worker.
        press_with_mods(&mut app, KeyCode::Char('w'), KeyModifiers::CONTROL);
        assert_eq!(app.focus, FocusArea::ChatInput);
        assert!(!app.collab_session_active);
        assert!(app.collab_worker.is_none());
        // Re-opening the hub (via the Feature Navigator in real use) finds
        // it idle; Esc from idle closes again.
        app.focus = FocusArea::CollaborationHub;
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }

    #[tokio::test]
    async fn collab_form_edits_only_the_selected_field() {
        use crate::focus::CollabField;
        let mut app = app_with(FocusArea::CollaborationHub);
        press(&mut app, KeyCode::Tab); // Server → Username, entering edit mode
        assert!(app.collab_editing);
        assert_eq!(app.collab_field, CollabField::Username);
        // Typing belongs to the field: 'q' must not quit.
        app.collab_username.clear();
        press(&mut app, KeyCode::Char('q'));
        assert_eq!(app.collab_username, "q");
        assert_eq!(app.focus, FocusArea::CollaborationHub);
        press(&mut app, KeyCode::Backspace);
        assert!(app.collab_username.is_empty());
        press(&mut app, KeyCode::Enter); // ends editing, keeps the value
        assert!(!app.collab_editing);
        // 'j' is the join entry point: straight into the Session field.
        press(&mut app, KeyCode::Char('j'));
        assert!(app.collab_editing);
        assert_eq!(app.collab_field, CollabField::Session);
        for c in "xencode-7".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        assert_eq!(app.collab_session_id, "xencode-7");
        // Tab while editing cycles the field without leaving edit mode.
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.collab_field, CollabField::Server);
        assert!(app.collab_editing);
    }

    #[tokio::test]
    async fn collab_r_retries_from_idle_and_from_live() {
        let mut app = app_with(FocusArea::CollaborationHub);
        press(&mut app, KeyCode::Char('r'));
        assert!(app.collab_session_active); // retry from idle = connect
        press(&mut app, KeyCode::Char('r'));
        assert!(app.collab_session_active); // retry while live = reconnect
        assert!(app.collab_worker.is_some());
    }

    #[test]
    fn collab_help_lists_only_keys_the_hub_handles() {
        let keys: Vec<&str> = crate::help::panel_bindings(FocusArea::CollaborationHub)
            .iter()
            .map(|(key, _)| *key)
            .collect();
        assert_eq!(
            keys,
            vec!["c", "j", "Enter", "r", "Tab", "type", "Esc"],
            "the help overlay and key_collab have drifted apart"
        );
    }

    #[test]
    fn ctrl_l_toggles_advise_panel() {
        let mut app = app_with(FocusArea::ChatInput);
        press_with_mods(&mut app, KeyCode::Char('l'), KeyModifiers::CONTROL);
        assert_eq!(app.focus, FocusArea::AdvisePanel);
        assert!(!app.advise_detail);
        press_with_mods(&mut app, KeyCode::Char('l'), KeyModifiers::CONTROL);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }

    #[test]
    fn advise_panel_detail_navigation_open_and_esc_unwind() {
        use xencode_context_rs::AdviceKind;
        let item = |kind, file: &str| xencode_context_rs::Advice {
            file: file.to_string(),
            kind,
            message: format!("message for {file}"),
        };
        let mut app = app_with(FocusArea::AdvisePanel);
        app.advise_items = vec![
            item(AdviceKind::BrokenImport, "src/app.rs"),
            item(AdviceKind::Cycle, "src/b.rs"),
        ];
        press(&mut app, KeyCode::Down);
        assert_eq!(app.advise_selected, 1);
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.advise_selected, 1, "clamped at the end");
        press(&mut app, KeyCode::Char('k'));
        assert_eq!(app.advise_selected, 0);
        // Detail mode: j/k scroll, 'o' is inert, Esc returns to the list.
        press(&mut app, KeyCode::Enter);
        assert!(app.advise_detail);
        press(&mut app, KeyCode::Down);
        assert_eq!(app.advise_scroll, 1);
        press(&mut app, KeyCode::Char('o'));
        assert_eq!(app.opened_file, None);
        press(&mut app, KeyCode::Esc);
        assert!(!app.advise_detail);
        assert_eq!(app.focus, FocusArea::AdvisePanel);
        // List mode: 'o' opens the advised file, Esc closes the panel.
        press(&mut app, KeyCode::Char('o'));
        assert_eq!(app.opened_file.as_deref(), Some("src/app.rs"));
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }
}
