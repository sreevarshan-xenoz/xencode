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
            app.focus = match app.focus {
                FocusArea::FileExplorer => FocusArea::CodeEditor,
                FocusArea::CodeEditor => FocusArea::ChatInput,
                FocusArea::ChatInput => FocusArea::FileExplorer,
                _ => FocusArea::ChatInput,
            };
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
                let _ = app.config.save();
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
        FocusArea::ModelSelector
        | FocusArea::CodeReview
        | FocusArea::PerformanceDashboard
        | FocusArea::ProviderHealth
        | FocusArea::ProjectAnalyzer
        | FocusArea::GitCommit
        | FocusArea::FeatureNavigator
        | FocusArea::ByteBotPanel
        | FocusArea::CollaborationHub
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

fn key_settings(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
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
                if app.settings_cursor > 0 {
                    app.settings_cursor -= 1;
                }
            }
            KeyCode::Down => {
                if app.settings_cursor + 1 < crate::focus::SETTINGS_ROWS.len() {
                    app.settings_cursor += 1;
                }
            }
            KeyCode::Left
                if (6..=12).contains(&app.settings_cursor) && app.settings_url_cursor > 0 =>
            {
                app.settings_url_cursor -= 1;
            }
            KeyCode::Right
                if (6..=12).contains(&app.settings_cursor)
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
            if app.settings_cursor + 1 < crate::focus::SETTINGS_ROWS.len() {
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
    if app.settings_url_editing {
        // Commit the edit for whichever row is under the cursor.
        match app.settings_cursor {
            6 => app.config.ollama_url = app.settings_url_buffer.clone(),
            7 => app.config.llama_cpp_url = app.settings_url_buffer.clone(),
            8 => app.config.llama_cpp_model_path = app.settings_url_buffer.clone(),
            9 => {
                app.config.llama_cpp_temperature = app
                    .settings_url_buffer
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|x| x.is_finite());
            }
            10 => {
                app.config.llama_cpp_top_k = app.settings_url_buffer.trim().parse().ok();
            }
            11 => {
                app.config.llama_cpp_min_p = app
                    .settings_url_buffer
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|x| x.is_finite());
            }
            12 => {
                app.config.llama_cpp_max_tokens = app.settings_url_buffer.trim().parse().ok();
            }
            _ => {}
        }
        app.settings_url_editing = false;
        let _ = app.config.save();
        // Refresh models and health check when a URL/model path changed.
        if app.settings_cursor <= 8 {
            app.refresh_models(tx.clone());
            app.run_health_check(tx.clone());
        }
        return;
    }
    let row = app.settings_cursor;
    if (6..=12).contains(&row) {
        // Start editing this row's text value.
        app.settings_url_editing = true;
        app.settings_url_buffer = match row {
            6 => app.config.ollama_url.clone(),
            7 => app.config.llama_cpp_url.clone(),
            8 => app.config.llama_cpp_model_path.clone(),
            9 => app
                .config
                .llama_cpp_temperature
                .map(|v| v.to_string())
                .unwrap_or_default(),
            10 => app
                .config
                .llama_cpp_top_k
                .map(|v| v.to_string())
                .unwrap_or_default(),
            11 => app
                .config
                .llama_cpp_min_p
                .map(|v| v.to_string())
                .unwrap_or_default(),
            _ => app
                .config
                .llama_cpp_max_tokens
                .map(|v| v.to_string())
                .unwrap_or_default(),
        };
        app.settings_url_cursor = app.settings_url_buffer.len();
    } else if row == 13 {
        // Factory reset
        app.config = XencodeConfig::default();
        app.theme = ThemeColors::get(&app.config.active_theme);
        app.style_chat_input();
        app.settings_reset_active = true;
        app.settings_cursor = 0;
        let _ = app.config.save();
        app.refresh_models(tx.clone());
        app.run_health_check(tx.clone());
        app.focus = FocusArea::ChatInput;
    } else {
        let _ = app.config.save();
        app.focus = FocusArea::ChatInput;
    }
}

fn settings_step(app: &mut App, dir: i32) {
    match app.settings_cursor {
        0 => {
            if let Some(next) = crate::theme::cycle_theme(&app.config.active_theme, dir > 0) {
                app.config.active_theme = next;
                app.theme = ThemeColors::get(&app.config.active_theme);
                app.style_chat_input();
                let _ = app.config.save();
            }
        }
        1 => {
            app.config.cache_enabled = !app.config.cache_enabled;
            let _ = app.config.save();
        }
        2 => {
            app.config.memory_enabled = !app.config.memory_enabled;
            let _ = app.config.save();
        }
        3 => {
            if dir < 0 {
                if app.config.max_cache_size >= 20 {
                    app.config.max_cache_size -= 10;
                    let _ = app.config.save();
                }
            } else {
                app.config.max_cache_size = app.config.max_cache_size.saturating_add(10).min(1000);
                let _ = app.config.save();
            }
        }
        4 => {
            if dir < 0 {
                if app.config.max_memory_items >= 10 {
                    app.config.max_memory_items -= 5;
                    let _ = app.config.save();
                }
            } else {
                app.config.max_memory_items =
                    app.config.max_memory_items.saturating_add(5).min(500);
                let _ = app.config.save();
            }
        }
        5 => {
            if dir < 0 {
                if app.config.response_timeout >= 10 {
                    app.config.response_timeout -= 5;
                    let _ = app.config.save();
                }
            } else {
                app.config.response_timeout =
                    app.config.response_timeout.saturating_add(5).min(300);
                let _ = app.config.save();
            }
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

fn key_collab(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    if key.code == KeyCode::Enter && !app.collab_session_active {
        app.start_collab_session(tx.clone());
        return true;
    }
    false
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
        app.settings_cursor = 6;
        app.settings_url_editing = true;
        app.settings_url_buffer = "http://localhost:11434".to_string();
        app.settings_url_cursor = app.settings_url_buffer.len();
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.settings_url_buffer, "http://localhost:11434j");
        assert_eq!(app.settings_cursor, 6);
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
