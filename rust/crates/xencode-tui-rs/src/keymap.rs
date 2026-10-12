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

use crate::app::{llama_model_target, App};
use crate::focus::{FocusArea, InputMode};
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
        return approval_modal_key(app, key, tx);
    }
    // The help overlay is modal: Esc/?/F1 close it, every other key is
    // swallowed while it is open.
    if app.help_visible {
        return help_modal_key(app, key);
    }
    // The command palette is modal too (AG-3): everything typed goes into its
    // own query, so a half-written prompt in the composer is never touched.
    if app.palette_visible {
        return palette_modal_key(app, key, tx);
    }
    // The agent stack overlay is modal the same way: Esc closes, Ctrl+N
    // advances while it is open, everything else is swallowed so typing never
    // lands in chat behind a panel the user is reading.
    if app.agent_stack_visible {
        match key.code {
            KeyCode::Esc => {
                app.agent_stack_visible = false;
            }
            _ => {
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    if let KeyCode::Char('n') | KeyCode::Char('N') = key.code {
                        advance_agent_stack(app);
                    }
                }
            }
        }
        return done();
    }
    // Orchestrator mode (OM-3): keys go to the focused tile, Ctrl+A is the
    // prefix for the mode's own commands.
    if let Some(flow) = orchestrator_key(app, key, tx) {
        return flow;
    }
    // Esc stops a running turn before it does anything else (UX-14). The
    // screen has promised this for a while ("Esc to stop it first"); the
    // text already streamed stays, and the loop ends at its next boundary.
    // The second Esc, with the stop already asked for, is the ordinary key.
    if key.code == KeyCode::Esc && key.modifiers.is_empty() && stop_running_turn(app, tx) {
        return done();
    }
    // Only a second bare `q` confirms a quit; any other key withdraws it.
    if !(key.code == KeyCode::Char('q') && key.modifiers.is_empty()) {
        app.quit_armed = false;
    }
    // The same for a second Ctrl+C on an empty input.
    if !(key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)) {
        app.ctrl_c_armed = false;
    }
    // The conversation is always ready to type (TD-3): there is no mode to
    // switch into first, so a letter is a letter there.
    if app.focus == FocusArea::ChatInput {
        app.input_mode = InputMode::Editing;
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

/// A key in orchestrator mode. `None` lets the ordinary handling have it:
/// outside the mode, on the conversation's tile, and for Ctrl+Space.
fn orchestrator_key(app: &mut App, key: KeyEvent, tx: &Tx) -> Option<KeyFlow> {
    use crate::focus::Mode;
    if app.mode != Mode::Orchestrator || app.screen() != crate::screens::Screen::Conversation {
        return None;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    // The picker takes every key while it is open.
    if let Some(picker) = app.orch.picker.as_mut() {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => picker.cursor = picker.cursor.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                picker.cursor = (picker.cursor + 1).min(picker.agents.len().saturating_sub(1))
            }
            KeyCode::Char(' ') => {
                if let Some(agent) = picker.agents.get(picker.cursor) {
                    if agent.installed && !picker.chosen.remove(&agent.name) {
                        picker.chosen.insert(agent.name.clone());
                    }
                }
            }
            KeyCode::Enter => app.open_orch_chosen(tx),
            KeyCode::Esc => app.orch.picker = None,
            _ => {}
        }
        return Some(done());
    }
    let tiles = app.term_tiles.len() + 1;
    if app.orch.prefix {
        app.orch.prefix = false;
        let focused_id = app
            .orch
            .focus
            .checked_sub(1)
            .and_then(|i| app.term_tiles.get(i))
            .map(|t| (t.id.clone(), t.agent.clone()));
        if !matches!(key.code, KeyCode::Char('x')) {
            app.orch.stop_armed = None;
        }
        match key.code {
            KeyCode::Char(d) if d.is_ascii_digit() => {
                let n = d as usize - '0' as usize;
                if n < tiles {
                    app.orch.focus = n;
                }
            }
            KeyCode::Char('a') if ctrl => {
                if let Some((id, _)) = focused_id {
                    app.term_type(&id, &[1], tx);
                }
            }
            KeyCode::Char('z') => app.orch.zoom = !app.orch.zoom,
            KeyCode::Char('t') => {
                app.orch.all = !app.orch.all;
                app.orch.zoom = false;
            }
            KeyCode::Char('n') => app.open_orch_picker(),
            KeyCode::Char('x') => {
                if let Some((id, agent)) = focused_id {
                    if app.orch.stop_armed.as_deref() == Some(id.as_str()) {
                        app.orch.stop_armed = None;
                        crate::engine::act(
                            app,
                            crate::engine::proto::ClientMsg::Term {
                                req: 0,
                                request: crate::engine::proto::TermRequest::Stop { id },
                            },
                            tx,
                        );
                    } else {
                        app.orch.stop_armed = Some(id);
                        app.push_toast(
                            crate::toast::ToastKind::Info,
                            format!("ctrl+a x again stops {agent}; its worktree stays"),
                        );
                    }
                }
            }
            KeyCode::Left | KeyCode::Up | KeyCode::Char('h') | KeyCode::Char('k') => {
                app.orch.focus = (app.orch.focus + tiles - 1) % tiles;
            }
            KeyCode::Right | KeyCode::Down | KeyCode::Char('l') | KeyCode::Char('j') => {
                app.orch.focus = (app.orch.focus + 1) % tiles;
            }
            KeyCode::Char('?') => {
                app.help_visible = true;
                app.help_scroll = 0;
            }
            _ => {}
        }
        return Some(done());
    }
    if ctrl && key.code == KeyCode::Char('a') {
        app.orch.prefix = true;
        return Some(done());
    }
    // Ctrl+Space leaves the mode; the conversation's tile is the ordinary
    // conversation.
    if (ctrl && key.code == KeyCode::Char(' ')) || app.orch.focus == 0 {
        return None;
    }
    let tile = app
        .term_tiles
        .get(app.orch.focus - 1)
        .map(|t| t.id.clone())?;
    if let Some(bytes) = crate::orchestrator_view::key_bytes(key) {
        app.term_type(&tile, &bytes, tx);
    }
    Some(done())
}

fn quit() -> KeyFlow {
    KeyFlow::Quit
}

/// Ask every running agent loop to stop: the chat turn and the ByteBot task
/// each carry a flag the round loop races against. Returns `true` when at
/// least one flag was newly set, so the caller can swallow the key; a flag
/// already set means the stop was asked for and the key means what it
/// usually does.
fn stop_running_turn(app: &mut App, tx: &Tx) -> bool {
    use crate::engine::proto::{ClientMsg, StopTarget};
    let unset = |flag: &Option<std::sync::Arc<std::sync::atomic::AtomicBool>>| {
        flag.as_ref()
            .is_some_and(|f| !f.load(std::sync::atomic::Ordering::Relaxed))
    };
    // A window onto an engine (EN-2) holds no stop flags of its own: what is
    // running is what the engine's view says.
    let window = app.is_window();
    let chat = app.is_generating && (window || unset(&app.turn_stop));
    // A task waiting on a question sits inside that tool call and would never
    // see the stop; the engine withdraws the question too (BT-2).
    let bytebot =
        app.bytebot_running && (window || unset(&app.bytebot_stop) || app.bytebot_help.is_some());
    if chat {
        crate::engine::act(
            app,
            ClientMsg::Stop {
                target: StopTarget::Chat,
            },
            tx,
        );
    }
    if bytebot {
        crate::engine::act(
            app,
            ClientMsg::Stop {
                target: StopTarget::Bytebot,
            },
            tx,
        );
    }
    let stopped = chat || bytebot;
    if stopped {
        app.push_toast(
            crate::toast::ToastKind::Info,
            "Stopping the turn — what it already said stays".to_string(),
        );
    }
    stopped
}

/// Keys while the command palette is open. Printable characters extend the
/// query; arrows (or Ctrl+P / Ctrl+N) move the highlight; Enter chooses; Esc
/// or Ctrl+X closes it and leaves focus and mode exactly as they were.
fn palette_modal_key(app: &mut App, key: KeyEvent, tx: &Tx) -> KeyFlow {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc => app.palette_visible = false,
        KeyCode::Char('x') | KeyCode::Char('X') if ctrl => app.palette_visible = false,
        KeyCode::Enter => app.choose_palette_entry(tx),
        KeyCode::Up => palette_move(app, -1),
        KeyCode::Char('p') if ctrl => palette_move(app, -1),
        KeyCode::Down => palette_move(app, 1),
        KeyCode::Char('n') if ctrl => palette_move(app, 1),
        KeyCode::Char('u') if ctrl => {
            app.palette_query.clear();
            app.palette_selected = 0;
        }
        KeyCode::Backspace => {
            app.palette_query.pop();
            app.palette_selected = 0;
        }
        KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => {
            app.palette_query.push(c);
            app.palette_selected = 0;
        }
        _ => {}
    }
    done()
}

/// Move the palette highlight one row, staying inside the matches.
fn palette_move(app: &mut App, step: isize) {
    let last = app.palette_matches().len().saturating_sub(1);
    app.palette_selected = app.palette_selected.saturating_add_signed(step).min(last);
}

fn done() -> KeyFlow {
    KeyFlow::Continue
}

/// The approval prompt answers with four keys and nothing else. Enter is
/// deliberately swallowed rather than treated as a deny: a stray Return in
/// the terminal should never be read as an answer.
fn approval_modal_key(app: &mut App, key: KeyEvent, tx: &Tx) -> KeyFlow {
    // A letter typed in the moment the box appears was meant for the draft:
    // it goes there, and only a key pressed once the box has been on screen
    // for half a second answers it.
    // Other keys (Esc denies, the arrows scroll) still work: none of them is
    // typed by accident in the middle of a word, and none of them allows.
    if app.approval_settling() {
        if let KeyCode::Char(c) = key.code {
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                && app.focus == FocusArea::ChatInput
            {
                app.chat_input.insert_char(c);
            }
            return done();
        }
    }
    use crate::engine::proto::{ClientMsg, WireAnswer};
    let answer = |app: &mut App, answer: WireAnswer| {
        if let Some(&id) = app.approval_ids.front() {
            crate::engine::act(app, ClientMsg::AnswerApproval { id, answer }, tx);
        }
    };
    match key.code {
        KeyCode::Char('y') | KeyCode::Char('Y') => answer(app, WireAnswer::Allow),
        KeyCode::Char('a') | KeyCode::Char('A') => answer(app, WireAnswer::AllowForSession),
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => answer(app, WireAnswer::Deny),
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
fn advance_agent_stack(app: &mut App) {
    let count = app.agent_stack_panes().len().max(1);
    app.agent_stack_index = (app.agent_stack_index + 1) % count;
}

fn global_ctrl_chord(app: &mut App, key: KeyEvent, tx: &Tx) -> Option<KeyFlow> {
    // While a prompt is being typed, the readline keys belong to the
    // composer: Ctrl+A/E/K/U/W used to open panels.
    if app.input_mode == InputMode::Editing
        && app.focus == FocusArea::ChatInput
        && matches!(key.code, KeyCode::Char('a' | 'k' | 'u' | 'w'))
    {
        return None;
    }
    // Ctrl+E too, while there is text to move to the end of; on an empty
    // input it opens the files view (TD-3), which loses nothing.
    if app.input_mode == InputMode::Editing
        && app.focus == FocusArea::ChatInput
        && key.code == KeyCode::Char('e')
        && !app.chat_input.is_empty()
    {
        return None;
    }
    match key.code {
        KeyCode::Char(' ') => {
            // The product-mode toggle (`X-2`). Claimed here, on the pre-focus
            // global stage that runs before focus routing, precisely so it cannot
            // steal the bare `Space` the File Explorer and Security panels
            // each bind to their own action: a `Ctrl+Space` never reaches those
            // handlers, and a plain `Space` (no CONTROL) still does. Flipping the
            // mode reads and writes one `App.mode` field — every task, agent,
            // session, worktree, diff, approval and git fact lives once on `App`,
            // so a round trip between the modes cannot copy or drop any of them.
            app.toggle_mode();
        }
        KeyCode::Char('n') | KeyCode::Char('N') => {
            // Agent stack: if tiled in the body layout, advance it directly;
            // otherwise open/advance the overlay. One chord, discoverable in
            // the help table.
            if app.last_layout.agents.is_some() {
                advance_agent_stack(app);
                return Some(done());
            }
            if !app.agent_stack_visible {
                app.agent_stack_visible = true;
                app.agent_stack_index = 0;
            } else {
                advance_agent_stack(app);
            }
            return Some(done());
        }
        KeyCode::Char('c') => {
            // A running turn is cancelled first. Then, in the conversation, a
            // draft is cleared; an empty input asks once, and the next Ctrl+C
            // quits (TD-3). Elsewhere an idle session quits, as before.
            if stop_running_turn(app, tx) {
                app.push_toast(
                    crate::toast::ToastKind::Info,
                    "Ctrl+C again quits".to_string(),
                );
                return Some(done());
            }
            if app.focus == FocusArea::ChatInput {
                if !app.chat_input.is_empty() {
                    app.reset_chat_input();
                    return Some(done());
                }
                if !app.ctrl_c_armed {
                    app.ctrl_c_armed = true;
                    app.push_toast(
                        crate::toast::ToastKind::Info,
                        "Ctrl+C again quits".to_string(),
                    );
                    return Some(done());
                }
            }
            return Some(quit());
        }
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
            // On the conversation it shows or hides the side panel (TD-2);
            // ByteBot is still a Ctrl+X search away.
            if app.focus == FocusArea::ChatInput && crate::shell::active(app) {
                app.side_panel_open = !app.side_panel_open;
            } else if !app.bytebot_running && app.focus != FocusArea::ByteBotPanel {
                app.focus = FocusArea::ByteBotPanel;
            } else if app.focus == FocusArea::ByteBotPanel {
                app.focus = FocusArea::ChatInput;
            }
        }
        KeyCode::Char('p') => {
            // The palette (TD-3): the seven views, every panel, command and
            // setting. The project analyzer moved into the Project view.
            app.open_palette();
        }
        KeyCode::Char('e') => {
            // The files view, and from its explorer back (TD-3); from the
            // editor it is the explorer, as it always was.
            if app.focus == FocusArea::FileExplorer {
                app.open_screen(crate::screens::Screen::Conversation);
            } else {
                app.open_screen(crate::screens::Screen::Files);
            }
        }
        KeyCode::Char('w') => {
            // Close current panel and return to ChatInput
            match app.focus {
                FocusArea::ByteBotPanel
                | FocusArea::SecurityAuditor
                | FocusArea::CustomModels
                | FocusArea::MultiLanguage
                | FocusArea::ProviderHealth
                | FocusArea::ProjectAnalyzer
                | FocusArea::GitCommit
                | FocusArea::CodeReview
                | FocusArea::ReviewDashboard
                | FocusArea::TaskManager
                | FocusArea::WorktreePanel
                | FocusArea::AdvisePanel
                | FocusArea::ImpactPanel
                | FocusArea::WorkerPanel
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
            // Unfold the newest tool call's output, or fold it again (TD-3).
            // The worktrees moved into the Agents view.
            let latest = app.latest_tool_id();
            app.tool_expanded = if app.tool_expanded.is_some() && app.tool_expanded == latest {
                None
            } else {
                latest
            };
        }
        KeyCode::Char('a') => {
            // The worker panel (`OR-12`): re-reads the streams, the registry and
            // the records on disk on every open, because a panel of numbers that
            // is quietly stale is worse than one that says it has not looked.
            if app.focus == FocusArea::WorkerPanel {
                app.focus = FocusArea::ChatInput;
            } else {
                app.refresh_worker_panel();
                app.focus = FocusArea::WorkerPanel;
            }
        }
        KeyCode::Char('l') => {
            // The Models view (TD-3); insights moved into the Project view.
            if app.screen() == crate::screens::Screen::Models {
                app.open_screen(crate::screens::Screen::Conversation);
            } else {
                app.open_screen_with(crate::screens::Screen::Models, Some(tx));
            }
        }
        KeyCode::Char('t') => {
            // The terminal strip under the editor in the Files view.
            app.show_terminal = !app.show_terminal;
        }
        KeyCode::Char('h') => {
            if !app.health_check_in_progress {
                app.run_health_check(tx.clone());
            }
        }
        KeyCode::Char('x') | KeyCode::Char('X') => {
            // The command palette (AG-3). Ctrl+K, the chord most editors use,
            // was already the background tasks panel here and also deletes to
            // the end of the line in the composer, so the palette takes the
            // unbound Ctrl+X, Emacs' "run a command" key, and nothing moved.
            app.open_palette();
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
    // In a view of several tabs, Tab and Shift+Tab move between them (TD-4).
    // The files view keeps its own pane ring, and the languages panel its
    // own use of Tab, which Shift+Tab still steps past.
    let screen = app.screen();
    let tabbed = !matches!(
        screen,
        crate::screens::Screen::Conversation | crate::screens::Screen::Files
    ) && screen.panels().len() > 1;
    if tabbed
        && (key.code == KeyCode::BackTab
            || (key.code == KeyCode::Tab && app.focus != FocusArea::MultiLanguage))
    {
        let step = if key.code == KeyCode::BackTab { -1 } else { 1 };
        let next = screen.step_tab(app.focus, step);
        app.open_panel(next, Some(tx));
        return done();
    }
    match key.code {
        KeyCode::Tab => {
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
/// frame actually drew (H1-05) — in the Files view that is the tree and the
/// editor. Before the first draw (or with no layout recorded) every pane
/// counts as visible.
/// From any overlay panel, Tab lands back on the chat input.
fn next_body_focus(app: &App) -> FocusArea {
    use crate::focus::FocusArea::*;
    let default_ring = [FileExplorer, CodeEditor, ChatInput];
    let ring = [FileExplorer, CodeEditor, ByteBotPanel, ChatInput];
    let is_visible = |f| match f {
        FileExplorer => app.last_layout.explorer.is_some(),
        CodeEditor => app.last_layout.editor.is_some(),
        ByteBotPanel => app.last_layout.agents.is_some(),
        ChatInput => app.last_layout.chat.is_some() || app.last_layout.input.is_some(),
        _ => false,
    };
    let active: Vec<FocusArea> = ring.iter().copied().filter(|f| is_visible(*f)).collect();
    let ring_ref: &[FocusArea] = if active.is_empty() {
        &default_ring
    } else {
        &active
    };
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
        FocusArea::ImpactPanel => key_impact_panel(app, key),
        FocusArea::WorkerPanel => key_worker_panel(app, key, tx),
        FocusArea::ProviderHealth => key_provider_health(app, key),
        FocusArea::CustomModels => key_custom_models(app, key, tx),
        FocusArea::MultiLanguage => key_multi_language(app, key, tx),
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
            // `/` is the start of a command, so it stays typed: dropping it sent
            // `/init` to the model as "init". Only into an empty draft, so a
            // half-written prompt is not changed by the key that reopens it.
            if key.code == KeyCode::Char('/') && app.chat_input.is_empty() {
                app.chat_input.insert_char('/');
            }
        }
        KeyCode::Char('m') if !app.text_entry_active() => {
            app.focus = if app.focus == FocusArea::ModelSelector {
                FocusArea::ChatInput
            } else {
                // The panel's footer reports the configured file's checksum
                // state, and a line from an earlier load would cover it for the
                // rest of the session.
                app.llamacpp_action_msg.clear();
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
        KeyCode::Char('q') if !app.text_entry_active() => {
            // One stray `q` must not end the session: the first asks, the
            // second quits, and any other key in between keeps going.
            if app.quit_armed {
                return quit();
            }
            app.quit_armed = true;
            app.push_toast(
                crate::toast::ToastKind::Info,
                "Press q again to quit — any other key keeps the session".to_string(),
            );
            return done();
        }
        _ => {}
    }
    done()
}

fn on_esc(app: &mut App) {
    if app.init_visible {
        app.init_visible = false;
        return;
    }
    if app.focus == FocusArea::ByteBotPanel && app.bytebot_model_picker {
        app.bytebot_model_picker = false;
        return;
    }
    match app.focus {
        // The Files view, like every other view, gives way to the
        // conversation on Esc (the editor's first Esc leaves editing mode).
        FocusArea::FileExplorer | FocusArea::CodeEditor => {
            app.open_screen(crate::screens::Screen::Conversation);
        }
        FocusArea::Settings => {
            if app.settings_url_editing {
                forget_secret(app);
                app.settings_url_editing = false;
            } else {
                app.settings_reset_active = false;
                app.settings_reset_armed = false;
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
        FocusArea::ImpactPanel => {
            // Esc unwinds one stage at a time: inspect (detail) back to the
            // tree, then a previous target from the descend stack, then close
            // to chat. The user's ← is the same unwind as Esc for the target,
            // so the two are interchangeable and neither silently drops the
            // history the user built by descending.
            if app.impact_detail {
                app.impact_detail = false;
                app.impact_scroll = 0;
            } else if let Some(previous) = app.impact_history.pop() {
                app.refresh_impact_for(&previous);
            } else {
                app.focus = FocusArea::ChatInput;
            }
        }
        FocusArea::WorkerPanel => {
            // Esc unwinds two stages: a row's trace closes before the panel does,
            // and closing the panel leaves every worker where it was. Reading
            // the fleet changes nothing about it (`OR-12`).
            if app.workers_detail {
                app.workers_detail = false;
                app.workers_scroll = 0;
            } else {
                app.focus = FocusArea::ChatInput;
            }
        }
        FocusArea::ModelSelector
        | FocusArea::CodeReview
        | FocusArea::ProviderHealth
        | FocusArea::ProjectAnalyzer
        | FocusArea::GitCommit
        | FocusArea::ByteBotPanel
        | FocusArea::SecurityAuditor
        | FocusArea::CustomModels
        | FocusArea::MultiLanguage
        | FocusArea::ReviewDashboard
        | FocusArea::TaskManager => {
            app.focus = FocusArea::ChatInput;
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
            if let Some(model) = app.available_models.get(app.selected_model).cloned() {
                crate::engine::act(
                    app,
                    crate::engine::proto::ClientMsg::SetModel { name: model },
                    tx,
                );
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

/// Rows whose value is typed into the buffer (Text/Number/Secret kinds). Only
/// these claim ←/→ for cursor movement while editing.
fn settings_row_typable(app: &App) -> bool {
    matches!(
        settings_current(app).kind,
        crate::focus::SettingKind::Text
            | crate::focus::SettingKind::Number
            | crate::focus::SettingKind::Secret
    )
}

/// A finished Secret edit must not leave the plaintext key sitting in the
/// editor buffer once the row no longer owns it.
fn forget_secret(app: &mut App) {
    if matches!(
        settings_current(app).kind,
        crate::focus::SettingKind::Secret
    ) {
        app.settings_url_buffer.clear();
        app.settings_url_cursor = 0;
    }
}

fn key_settings(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    let row_count = crate::focus::SETTINGS_ITEMS.len();
    // An armed Factory Reset (TX-5) is only ever performed by the very next key
    // being Enter on that row; anything else takes the arming back.
    if key.code != KeyCode::Enter {
        app.settings_reset_armed = false;
    }
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
                forget_secret(app);
                app.settings_url_editing = false;
                if app.settings_cursor > 0 {
                    app.settings_cursor -= 1;
                }
            }
            KeyCode::Down => {
                forget_secret(app);
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
        KeyCode::Left => settings_step(app, -1, tx),
        KeyCode::Right => settings_step(app, 1, tx),
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
        if matches!(settings_current(app).kind, SettingKind::Secret) {
            // An emptied key field means "remove the key", not "store nothing".
            crate::app::set_secret_value(
                &mut app.config,
                row_label,
                (!buf.trim().is_empty()).then(|| buf.trim().to_string()),
            );
        } else {
            match row_label {
                "Ollama URL" => app.config.ollama_url = buf,
                "Llama.cpp URL" => app.config.llama_cpp_url = buf,
                "Llama.cpp Model" => app.config.llama_cpp_model_path = buf,
                "Remote URL" => app.config.remote_base_url = buf.trim().to_string(),
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
                "Llama Seed" => {
                    app.config.llama_cpp_seed = buf.trim().parse().ok();
                }
                "Llama Max Tokens" => {
                    app.config.llama_cpp_max_tokens = buf.trim().parse().ok();
                }
                _ => {}
            }
        }
        forget_secret(app);
        app.settings_url_editing = false;
        app.save_config();
        // A changed endpoint changes the model picker and the health panel;
        // a changed key only changes whether a provider answers.
        if matches!(
            row_label,
            "Ollama URL" | "Llama.cpp URL" | "Llama.cpp Model"
        ) {
            app.refresh_models(tx.clone());
        }
        if matches!(
            row_label,
            "Ollama URL"
                | "Llama.cpp URL"
                | "Llama.cpp Model"
                | "Remote URL"
                | "Remote Key"
                | "Gemini Key"
                | "Qwen Key"
                | "OpenRouter Key"
        ) {
            app.run_health_check(tx.clone());
        }
        return;
    }
    match settings_current(app).kind {
        SettingKind::Text | SettingKind::Number | SettingKind::Secret => {
            // Start editing this row's value.
            app.settings_url_editing = true;
            app.settings_url_buffer = settings_edit_seed(app, row_label);
            app.settings_url_cursor = app.settings_url_buffer.len();
        }
        SettingKind::Action => {
            // Factory Reset replaces every setting — URLs, model, keys kept in the
            // config — so one Enter only arms it; the second performs it (TX-5).
            if !app.settings_reset_armed {
                app.settings_reset_armed = true;
                return;
            }
            app.settings_reset_armed = false;
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
        "Remote URL" => app.config.remote_base_url.clone(),
        "Remote Key" | "Gemini Key" | "Qwen Key" | "OpenRouter Key" => {
            crate::app::secret_value(&app.config, label)
                .map(str::to_string)
                .unwrap_or_default()
        }
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
        "Llama Seed" => app
            .config
            .llama_cpp_seed
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
        "Cloud Models" => &mut app.config.allow_cloud_models,
        "External Workers" => &mut app.config.allow_external_workers,
        "Mouse Capture" => &mut app.config.mouse_capture,
        "Floating Badge" => &mut app.config.badge_autostart,
        _ => return false,
    };
    *flag = !*flag;
    true
}

/// ←/→ on the row under the cursor. Every behavior derives from the row's
/// `SettingKind` in `focus::SETTINGS_ITEMS` — never from the row number.
fn settings_step(app: &mut App, dir: i32, tx: &Tx) {
    use crate::focus::SettingKind;
    let label = settings_current(app).label;
    // These three decide what the agent may do; the engine that runs the
    // tools is told after the change below (C1).
    let policy = matches!(
        label,
        "Agent Approval" | "Cloud Models" | "External Workers"
    );
    match settings_current(app).kind {
        SettingKind::Cycle(options) => {
            settings_cycle(app, label, options, dir);
            app.save_config();
        }
        SettingKind::Toggle => {
            if settings_toggle(app, label) {
                if label == "External Workers" {
                    // Opening this hands work to a program whose traffic xencode
                    // cannot see, so the cost of the flip is said at the keystroke
                    // that makes it.
                    app.push_toast(
                        crate::toast::ToastKind::Info,
                        if app.config.allow_external_workers {
                            "external workers on: a team role may be handed to another \
                             vendor's agent"
                        } else {
                            "external workers off: a name on the agent roster is refused \
                             by that name"
                        }
                        .to_string(),
                    );
                }
                if label == "Mouse Capture" {
                    // The row is the escape hatch for what the mouse costs, so
                    // the cost is said at the keystroke rather than in a manual
                    // nobody has open at the moment text will not select.
                    app.push_toast(
                        crate::toast::ToastKind::Info,
                        if app.config.mouse_capture {
                            "mouse on: the wheel scrolls and a click picks a pane"
                        } else {
                            "mouse off: the terminal keeps its own text selection"
                        }
                        .to_string(),
                    );
                }
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
                "Command Timeout" => {
                    app.config.agent_command_timeout =
                        stepped(app.config.agent_command_timeout, dir, step, min, max)
                }
                "Disclosure Level" => {
                    let next =
                        stepped(app.config.disclosure_level as u64, dir, step, min, max) as u8;
                    app.set_disclosure_level(crate::focus::DisclosureLevel::from_u8(next));
                }
                _ => return,
            }
            app.save_config();
        }
        _ => {}
    }
    if policy {
        app.push_policy(tx);
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
                // Same directory the old bare call inherited: only the signing
                // environment is new, not where the commit lands.
                let repo = std::env::current_dir().unwrap_or(std::path::PathBuf::from("."));
                tokio::spawn(async move {
                    // Signing-capable and repo-rooted: the commit runs where the
                    // repository is, with the signing environment passed through,
                    // so a configured signature works and a missing key fails
                    // with words instead of hanging on pinentry.
                    let out = tokio::task::spawn_blocking(move || {
                        crate::gitsign::commit_signed(&repo, &msg, 120)
                    })
                    .await
                    .map_err(|e| format!("commit task failed: {e}"));
                    let (tag, body) = match out {
                        Ok(Ok(line)) => ("[GIT_COMMIT_OK]", line),
                        Ok(Err(reason)) | Err(reason) => ("[GIT_COMMIT_ERR]", reason),
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
    if app.bytebot_model_picker {
        // The panel's model list (BT-5) takes the arrows and Enter; Esc is
        // handled by `on_esc`, which closes the list before the panel.
        let count = app.available_models.len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                app.bytebot_model_selected = app.bytebot_model_selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if app.bytebot_model_selected + 1 < count {
                    app.bytebot_model_selected += 1;
                }
            }
            KeyCode::Enter => {
                if let Some(model) = app
                    .available_models
                    .get(app.bytebot_model_selected)
                    .cloned()
                {
                    crate::engine::act(
                        app,
                        crate::engine::proto::ClientMsg::SetModel {
                            name: model.clone(),
                        },
                        tx,
                    );
                    app.bytebot_log.push(format!("model: {model}"));
                }
                app.bytebot_model_picker = false;
            }
            _ => {}
        }
        return true;
    }
    match key.code {
        KeyCode::Up if !app.bytebot_running && !app.bytebot_history.is_empty() => {
            app.bytebot_command = app.bytebot_history.last().unwrap().clone();
            app.bytebot_cursor = app.bytebot_command.len();
        }
        KeyCode::Enter => {
            // A waiting question (BT-2) takes the line as its answer;
            // otherwise Enter runs or queues what is typed. History is ↑.
            if let Some(id) = app
                .question_id
                .filter(|_| app.bytebot_help.is_some() || app.is_window())
            {
                let text = app.bytebot_command.clone();
                crate::engine::act(
                    app,
                    crate::engine::proto::ClientMsg::AnswerQuestion { id, text },
                    tx,
                );
            } else {
                app.run_bytebot(tx.clone());
            }
        }
        KeyCode::Char('n') | KeyCode::Char('N')
            if app.last_layout.agents.is_some() && app.bytebot_command.is_empty() =>
        {
            advance_agent_stack(app);
        }
        // BT-3: with nothing typed, `a` accepts and `u` undoes a task that is
        // waiting for review; with text in the box they type as usual.
        KeyCode::Char('a')
            if app.bytebot_command.is_empty() && app.bytebot_reviewing().is_some() =>
        {
            crate::engine::act(
                app,
                crate::engine::proto::ClientMsg::Review {
                    decision: crate::engine::proto::ReviewDecision::Accept,
                },
                tx,
            );
        }
        KeyCode::Char('u')
            if app.bytebot_command.is_empty() && app.bytebot_reviewing().is_some() =>
        {
            let replies = crate::engine::handle(
                app,
                crate::engine::proto::ClientMsg::Review {
                    decision: crate::engine::proto::ReviewDecision::Undo,
                },
                tx,
                "terminal",
            );
            for reply in replies {
                if let crate::engine::proto::EngineMsg::Error { message } = reply {
                    app.bytebot_log.push(message.clone());
                    app.push_toast(crate::toast::ToastKind::Warning, message);
                }
            }
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
        KeyCode::Char('p') => {
            if let Some(item) = app.advise_items.get(app.advise_selected) {
                app.propose_task(item.offer_task());
            }
        }
        _ => return false,
    }
    true
}

/// ImpactPanel (`QD-2`): the fan-out over QD-1's three layers. `↑/↓` move the
/// cursor over the flattened rows, `Enter` opens the selected row to read its
/// evidence (crate, hop, the `use`/`mod`/`impl` that resolved, and — where git
/// history is available — the co-change count with the target), `→` re-roots
/// the query at the selected consumer file and pushes the previous target onto
/// the descend stack, `←` pops that stack, and `r` re-runs the query in place.
/// Nothing here recomputes on its own: the user drives the frontier.
fn key_impact_panel(app: &mut App, key: KeyEvent) -> bool {
    use xencode_context_rs::ImpactRow;
    let rows: Vec<ImpactRow> = app
        .impact_tree
        .as_ref()
        .map(|t| t.rows())
        .unwrap_or_default();
    let count = rows.len();
    match key.code {
        KeyCode::Up | KeyCode::Char('k') if app.impact_detail => {
            app.impact_scroll = app.impact_scroll.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') if app.impact_detail => {
            app.impact_scroll += 1;
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.impact_selected = app.impact_selected.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.impact_selected + 1 < count {
                app.impact_selected += 1;
            }
        }
        KeyCode::Enter => {
            if count > 0 {
                app.impact_detail = !app.impact_detail;
                app.impact_scroll = 0;
            }
        }
        KeyCode::Right => {
            // Descend only onto a File row — a crate header or the target
            // itself has no consumer meaning, so the arrow does nothing there
            // rather than re-querying something the user did not pick.
            if let Some(ImpactRow::File { path, .. }) = rows.get(app.impact_selected) {
                let current = app
                    .impact_tree
                    .as_ref()
                    .map(|t| t.target.clone())
                    .unwrap_or_default();
                let next = path.clone();
                if !next.is_empty() && next != current {
                    app.impact_history.push(current);
                    app.impact_selected = 0;
                    app.impact_detail = false;
                    app.impact_scroll = 0;
                    app.refresh_impact_for(&next);
                }
            }
        }
        KeyCode::Left => {
            if let Some(previous) = app.impact_history.pop() {
                app.impact_selected = 0;
                app.impact_detail = false;
                app.impact_scroll = 0;
                app.refresh_impact_for(&previous);
            }
        }
        KeyCode::Char('r') => {
            if let Some(target) = app.impact_tree.as_ref().map(|t| t.target.clone()) {
                app.impact_selected = 0;
                app.impact_detail = false;
                app.impact_scroll = 0;
                app.refresh_impact_for(&target);
            }
        }
        KeyCode::Char('o') if !app.impact_detail => {
            if let Some(ImpactRow::File { path, .. }) = rows.get(app.impact_selected) {
                let path = path.clone();
                app.open_file_in_editor(&path);
            }
        }
        _ => return false,
    }
    true
}

fn key_worker_panel(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    let count = app.workers_rows.len();
    // A Team row's worker, for `s` and `m` (TM-4).
    let worker = app
        .workers_rows
        .get(app.workers_selected)
        .and_then(crate::worker_panel::team_worker)
        .map(str::to_string);
    match key.code {
        KeyCode::Up | KeyCode::Char('k') if app.workers_detail => {
            app.workers_scroll = app.workers_scroll.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') if app.workers_detail => {
            app.workers_scroll += 1;
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.workers_selected = app.workers_selected.saturating_sub(1);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.workers_selected + 1 < count {
                app.workers_selected += 1;
            }
        }
        KeyCode::Enter if count > 0 => {
            // The row's own trace opens: which record, event or file each
            // figure on the line came from.
            app.workers_detail = !app.workers_detail;
            app.workers_scroll = 0;
        }
        KeyCode::Char('r') => {
            app.refresh_worker_panel();
        }
        KeyCode::Char('s') if worker.is_some() => {
            let id = worker.unwrap_or_default();
            app.team_command(crate::engine::proto::TeamRequest::Stop { id }, tx);
        }
        KeyCode::Char('m') if worker.is_some() => {
            let id = worker.unwrap_or_default();
            app.team_command(crate::engine::proto::TeamRequest::Merge { id }, tx);
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

/// Custom models are a real form now (J-05): the list is `model_profiles` from
/// config, `n` starts one from the session's current settings, `←`/`→` and
/// `-`/`+` move its parameters in memory, `f` says what kind of turn may take it
/// without being asked (MI-7), `Enter` applies them to the next turn, `s` is the
/// only key that writes config.json, and `t` asks the provider itself. Every
/// character is handled here, so no letter — `n` and `s` above all — can fall
/// through to a global chord and open another panel (E2-06).
fn key_custom_models(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    let count = app.model_profiles.len();
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            if app.models_selected > 0 {
                app.models_selected -= 1;
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if app.models_selected + 1 < count {
                app.models_selected += 1;
            }
        }
        KeyCode::Enter => app.apply_model_profile(tx.clone()),
        KeyCode::Char('n') => app.add_model_profile(),
        KeyCode::Char('f') => app.cycle_model_profile_task(),
        KeyCode::Char('-') => app.adjust_model_temperature(-0.1),
        KeyCode::Char('+') | KeyCode::Char('=') => app.adjust_model_temperature(0.1),
        KeyCode::Left => app.step_model_max_tokens(false),
        KeyCode::Right => app.step_model_max_tokens(true),
        KeyCode::Char('s') => app.save_model_profiles(),
        KeyCode::Char('t') => app.test_model_profile(tx.clone()),
        KeyCode::Char(_) => {}
        _ => return false,
    }
    true
}

/// Multi-Language (J-04): `Enter`/`d` walks the workspace, `Tab` selects a form
/// field and letters then type into it, `Enter` translates. Detection is the
/// context engine's own walk; translation is one call to the configured model.
fn key_multi_language(app: &mut App, key: KeyEvent, tx: &Tx) -> bool {
    if app.lang_editing.is_some() {
        match key.code {
            KeyCode::Enter => app.translate_text(tx.clone()),
            KeyCode::Tab => app.cycle_lang_field(),
            KeyCode::Esc => app.lang_editing = None,
            KeyCode::Backspace => app.lang_backspace(),
            KeyCode::Char(c) => app.lang_char(c),
            _ => return false,
        }
        return true;
    }
    match key.code {
        KeyCode::Enter | KeyCode::Char('d') if !app.lang_busy => {
            app.start_language_scan(tx.clone())
        }
        KeyCode::Tab => app.cycle_lang_field(),
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
        // Readline editing in the composer (TX-9).
        KeyCode::Char('a') if key.modifiers == KeyModifiers::CONTROL => {
            app.chat_input.move_cursor(tui_textarea::CursorMove::Head);
        }
        KeyCode::Char('e') if key.modifiers == KeyModifiers::CONTROL => {
            app.chat_input.move_cursor(tui_textarea::CursorMove::End);
        }
        KeyCode::Char('k') if key.modifiers == KeyModifiers::CONTROL => {
            app.chat_input.delete_line_by_end();
        }
        KeyCode::Char('u') if key.modifiers == KeyModifiers::CONTROL => {
            app.chat_input.delete_line_by_head();
        }
        KeyCode::Char('w') if key.modifiers == KeyModifiers::CONTROL => {
            app.chat_input.delete_word();
        }
        // Enter sends; while a turn runs the message waits its turn, and a
        // line starting with `!` is a shell command (TD-3).
        KeyCode::Enter if app.focus == FocusArea::ChatInput => {
            app.send_or_queue(tx.clone());
        }
        KeyCode::Enter => {
            if !app.is_generating {
                app.submit_message(tx.clone());
                app.chat_scroll = 0;
            }
        }
        // `?` on an empty input is the key help; in a sentence it is a
        // question mark.
        KeyCode::Char('?')
            if app.focus == FocusArea::ChatInput
                && app.chat_input.is_empty()
                && key.modifiers.difference(KeyModifiers::SHIFT).is_empty() =>
        {
            app.help_visible = true;
            app.help_scroll = 0;
        }
        // The transcript scrolls by half a screen from the keyboard; the
        // wheel does the same (TD-3: k and j are letters here now).
        KeyCode::PageUp if app.focus == FocusArea::ChatInput => {
            let step = app.last_layout.chat.map_or(10, |r| (r.height / 2).max(1));
            app.chat_scroll = app.chat_scroll.saturating_add(step);
        }
        KeyCode::PageDown if app.focus == FocusArea::ChatInput => {
            let step = app.last_layout.chat.map_or(10, |r| (r.height / 2).max(1));
            app.chat_scroll = app.chat_scroll.saturating_sub(step);
        }
        // Shift+Tab steps the approval mode: ask, auto-edit, plan (TD-3).
        KeyCode::BackTab if app.focus == FocusArea::ChatInput => app.cycle_approval_mode(tx),
        // F1 types nothing, so it opens help even while typing a prompt.
        KeyCode::F(1) => {
            app.help_visible = true;
            app.help_scroll = 0;
        }
        // In the conversation Esc never leaves the input (TD-3); twice in a
        // row puts the last message back in it to edit and send again.
        KeyCode::Esc if app.focus == FocusArea::ChatInput => {
            let now = std::time::Instant::now();
            let again = app
                .last_esc_at
                .is_some_and(|at| now.duration_since(at) < std::time::Duration::from_millis(600));
            if again {
                app.last_esc_at = None;
                app.recall_last_message();
            } else {
                app.last_esc_at = Some(now);
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
            // An `@word` takes the best matching file first (TD-3).
            if !app.complete_mention() && !app.complete_slash_draft() {
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

    /// One guard for every test that repoints `XCODE_CONFIG_DIR`, which is
    /// process-global: without it, two persistence tests would send each
    /// other's writes to the wrong directory.
    static CONFIG_DIR: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Points `XCODE_CONFIG_DIR` at a test directory until dropped, then puts
    /// back whatever was there before, so the sandbox the test run was started
    /// with is never taken away from the tests that run after this one.
    struct ConfigDirOverride(Option<std::ffi::OsString>);

    impl ConfigDirOverride {
        fn set(dir: &std::path::Path) -> Self {
            let before = std::env::var_os("XCODE_CONFIG_DIR");
            std::env::set_var("XCODE_CONFIG_DIR", dir);
            Self(before)
        }
    }

    impl Drop for ConfigDirOverride {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var("XCODE_CONFIG_DIR", value),
                None => std::env::remove_var("XCODE_CONFIG_DIR"),
            }
        }
    }

    fn press(app: &mut App, code: KeyCode) -> KeyFlow {
        let (tx, _rx) = mpsc::unbounded_channel();
        handle_key(app, KeyEvent::new(code, KeyModifiers::NONE), &tx)
    }

    fn press_with_mods(app: &mut App, code: KeyCode, mods: KeyModifiers) -> KeyFlow {
        let (tx, _rx) = mpsc::unbounded_channel();
        handle_key(app, KeyEvent::new(code, mods), &tx)
    }

    #[test]
    fn slash_opens_the_composer_with_the_slash_typed() {
        let mut app = app_with(FocusArea::FileExplorer);
        app.input_mode = InputMode::Normal;
        press(&mut app, KeyCode::Char('/'));
        assert_eq!(app.input_mode, InputMode::Editing);
        assert_eq!(
            app.chat_input.lines().join(
                "
"
            ),
            "/"
        );
        for c in "init".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        assert_eq!(
            app.chat_input.lines().join(
                "
"
            ),
            "/init"
        );

        // A draft already there is left as it was, when `/` comes from a panel.
        app.focus = FocusArea::FileExplorer;
        app.input_mode = InputMode::Normal;
        press(&mut app, KeyCode::Char('/'));
        assert_eq!(
            app.chat_input.lines().join(
                "
"
            ),
            "/init"
        );
    }

    #[test]
    fn ctrl_n_opens_advances_and_esc_closes_the_agent_stack() {
        let mut app = app_with(FocusArea::ChatInput);
        assert!(!app.agent_stack_visible);
        press_with_mods(&mut app, KeyCode::Char('n'), KeyModifiers::CONTROL);
        assert!(app.agent_stack_visible);
        assert_eq!(app.agent_stack_index, 0);
        // Three panes: the fourth advance wraps to the front.
        press_with_mods(&mut app, KeyCode::Char('n'), KeyModifiers::CONTROL);
        assert_eq!(app.agent_stack_index, 1);
        press_with_mods(&mut app, KeyCode::Char('n'), KeyModifiers::CONTROL);
        press_with_mods(&mut app, KeyCode::Char('n'), KeyModifiers::CONTROL);
        assert_eq!(app.agent_stack_index, 0, "wraps instead of growing");
        press_with_mods(&mut app, KeyCode::Esc, KeyModifiers::empty());
        assert!(!app.agent_stack_visible);
    }

    /// The mode toggle is claimed on the pre-focus global stage, so a
    /// `Ctrl+Space` must flip `App.mode` without firing the bare-`Space` action
    /// of whichever panel is focused — and a plain `Space` must still reach that
    /// same panel's handler afterwards.
    #[test]
    fn ctrl_space_toggles_mode_without_stealing_a_panels_own_space() {
        // File Explorer: bare Space attaches the selected file.
        let mut app = app_with(FocusArea::FileExplorer);
        app.file_tree = vec!["notes.txt".into()];
        app.selected_file = 0;
        app.attached_files.clear();
        assert_eq!(app.mode, crate::focus::Mode::Coding);
        press_with_mods(&mut app, KeyCode::Char(' '), KeyModifiers::CONTROL);
        assert_eq!(
            app.mode,
            crate::focus::Mode::Orchestrator,
            "Ctrl+Space flips mode"
        );
        assert!(
            app.attached_files.is_empty(),
            "Ctrl+Space must not attach the file"
        );
        press(&mut app, KeyCode::Char(' '));
        assert_eq!(
            app.attached_files.len(),
            1,
            "plain Space still reaches the handler"
        );

        // Security Auditor: bare Space cycles the severity filter.
        let mut app = app_with(FocusArea::SecurityAuditor);
        app.sec_filter_severity = "All".into();
        press_with_mods(&mut app, KeyCode::Char(' '), KeyModifiers::CONTROL);
        assert_eq!(app.mode, crate::focus::Mode::Orchestrator);
        assert_eq!(
            app.sec_filter_severity, "All",
            "Ctrl+Space must not cycle the filter"
        );
        press(&mut app, KeyCode::Char(' '));
        assert_eq!(
            app.sec_filter_severity, "Critical",
            "plain Space still cycles"
        );
    }

    /// Both modes read the one shared state on `App`; no mode owns a copy, so a
    /// round trip cannot drop or duplicate anything a worker has put there.
    #[test]
    fn a_mode_round_trip_leaves_the_shared_state_identical() {
        let mut app = app_with(FocusArea::ChatInput);
        app.file_tree = vec!["a.rs".into(), "b.rs".into()];
        app.selected_file = 1;
        app.attached_files.insert("a.rs".into());
        app.sec_filter_severity = "High".into();
        app.input_history.push("a prompt".into());
        app.git_branch = "topic/x".into();
        app.git_status.insert("src/main.rs".into(), "M".into());

        let before_tree = app.file_tree.clone();
        let before_selected = app.selected_file;
        let before_attached = app.attached_files.clone();
        let before_severity = app.sec_filter_severity.clone();
        let before_history = app.input_history.clone();
        let before_branch = app.git_branch.clone();
        let before_git_status = app.git_status.clone();
        let before_worktrees = app.worktrees.clone();

        assert_eq!(app.mode, crate::focus::Mode::Coding);
        app.toggle_mode();
        assert_eq!(app.mode, crate::focus::Mode::Orchestrator);
        app.toggle_mode();
        assert_eq!(
            app.mode,
            crate::focus::Mode::Coding,
            "round trip returns to Coding"
        );

        assert_eq!(app.file_tree, before_tree);
        assert_eq!(app.selected_file, before_selected);
        assert_eq!(app.attached_files, before_attached);
        assert_eq!(app.sec_filter_severity, before_severity);
        assert_eq!(app.input_history, before_history);
        assert_eq!(app.git_branch, before_branch);
        assert_eq!(app.git_status, before_git_status);
        assert_eq!(
            app.worktrees, before_worktrees,
            "the tracked worktrees are untouched"
        );
    }

    fn app_with(focus: FocusArea) -> App<'static> {
        let mut app = App::for_tests();
        app.focus = focus;
        app.input_mode = InputMode::Normal;
        app
    }

    #[test]
    fn tab_cycles_body_focus() {
        let mut app = app_with(FocusArea::FileExplorer);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::CodeEditor);
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::ChatInput);
        // In the conversation Tab completes or indents; it does not leave (TD-3).
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::ChatInput);
    }

    #[test]
    fn tab_ring_skips_panes_the_layout_hides() {
        // The Files view draws the tree and the editor and no chat pane, so Tab
        // moves between those two and never onto a pane that is not there.
        let mut app = app_with(FocusArea::CodeEditor);
        app.last_layout = app.body_layout(ratatui::layout::Rect::new(0, 1, 80, 22));
        assert!(app.last_layout.chat.is_none());
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusArea::FileExplorer);
        press(&mut app, KeyCode::Tab);
        assert_eq!(
            app.focus,
            FocusArea::CodeEditor,
            "the missing chat pane must stay out of the ring"
        );
    }

    #[test]
    fn an_empty_session_opens_in_the_composer() {
        let mut app = App::for_tests();
        app.messages.retain(|m| m.role != "user");
        app.input_mode = InputMode::Normal;
        app.focus = FocusArea::FileExplorer;
        app.start_in_composer_when_empty();
        assert_eq!(app.input_mode, InputMode::Editing);
        assert_eq!(app.focus, FocusArea::ChatInput);
        // So a typed sentence is text, not global keys.
        for c in "quit smoke".chars() {
            assert_eq!(press(&mut app, KeyCode::Char(c)), KeyFlow::Continue);
        }
        assert_eq!(app.chat_input.lines().join(""), "quit smoke");

        // A restored conversation keeps the normal-mode start.
        let mut app = App::for_tests();
        app.messages.push(crate::app::UiMessage {
            role: "user".into(),
            content: "hi".into(),
        });
        app.input_mode = InputMode::Normal;
        app.start_in_composer_when_empty();
        assert_eq!(app.input_mode, InputMode::Normal);
    }

    #[test]
    fn readline_keys_edit_the_prompt_instead_of_opening_panels() {
        let mut app = app_with(FocusArea::ChatInput);
        app.input_mode = InputMode::Editing;
        for c in "hello world".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        let ctrl = |app: &mut App, c| {
            press_with_mods(app, KeyCode::Char(c), KeyModifiers::CONTROL);
        };
        ctrl(&mut app, 'w');
        assert_eq!(app.chat_input.lines().join(""), "hello ");
        ctrl(&mut app, 'a');
        press(&mut app, KeyCode::Char('>'));
        assert_eq!(app.chat_input.lines().join(""), ">hello ");
        ctrl(&mut app, 'e');
        press(&mut app, KeyCode::Char('!'));
        assert_eq!(app.chat_input.lines().join(""), ">hello !");
        ctrl(&mut app, 'u');
        assert_eq!(app.chat_input.lines().join(""), "");
        for c in "abc".chars() {
            press(&mut app, KeyCode::Char(c));
        }
        ctrl(&mut app, 'a');
        ctrl(&mut app, 'k');
        assert_eq!(app.chat_input.lines().join(""), "");
        assert_eq!(app.focus, FocusArea::ChatInput);
        assert_eq!(app.input_mode, InputMode::Editing);
    }

    #[test]
    fn ctrl_c_stops_a_running_turn_before_it_quits() {
        let mut app = app_with(FocusArea::ChatInput);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        app.turn_stop = Some(stop.clone());
        app.is_generating = true;
        let ctrl_c =
            |app: &mut App| press_with_mods(app, KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(ctrl_c(&mut app), KeyFlow::Continue);
        assert!(stop.load(std::sync::atomic::Ordering::Relaxed));
        // Then, on an empty input, one asks and the next quits (TD-3).
        assert_eq!(ctrl_c(&mut app), KeyFlow::Continue, "the second one asks");
        assert_eq!(ctrl_c(&mut app), KeyFlow::Quit, "the third one quits");

        let mut idle = app_with(FocusArea::ChatInput);
        assert_eq!(ctrl_c(&mut idle), KeyFlow::Continue);
        assert_eq!(ctrl_c(&mut idle), KeyFlow::Quit);
    }

    fn type_text(app: &mut App, text: &str) {
        for c in text.chars() {
            press(app, KeyCode::Char(c));
        }
    }

    fn ctrl(app: &mut App, c: char) -> KeyFlow {
        press_with_mods(app, KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    #[test]
    fn the_palette_opens_over_a_draft_and_closing_it_changes_nothing() {
        let mut app = app_with(FocusArea::ChatInput);
        app.input_mode = InputMode::Editing;
        type_text(&mut app, "half a prompt");
        ctrl(&mut app, 'x');
        assert!(app.palette_visible);
        // Typing goes to the palette's query, not to the composer.
        type_text(&mut app, "quit");
        assert_eq!(app.palette_query, "quit");
        assert!(
            !app.quit_armed,
            "q typed into the palette must not arm quitting"
        );
        press(&mut app, KeyCode::Esc);
        assert!(!app.palette_visible);
        assert_eq!(app.chat_input.lines().join(""), "half a prompt");
        assert_eq!(app.focus, FocusArea::ChatInput);
        assert_eq!(app.input_mode, InputMode::Editing);
    }

    #[test]
    fn the_palette_reaches_a_panel_a_setting_and_a_command() {
        let mut app = app_with(FocusArea::ChatInput);
        ctrl(&mut app, 'x');
        type_text(&mut app, "worktree");
        press(&mut app, KeyCode::Enter);
        assert!(!app.palette_visible);
        assert_eq!(app.focus, FocusArea::WorktreePanel);

        let mut app = app_with(FocusArea::ChatInput);
        ctrl(&mut app, 'x');
        type_text(&mut app, "ollama");
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.focus, FocusArea::Settings);
        assert_eq!(
            crate::focus::SETTINGS_ITEMS[app.settings_cursor].label,
            "Ollama URL"
        );

        // A command is staged in the composer, and a draft is kept in history.
        let mut app = app_with(FocusArea::ChatInput);
        app.input_mode = InputMode::Editing;
        type_text(&mut app, "my draft");
        ctrl(&mut app, 'x');
        type_text(&mut app, "rewind");
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.chat_input.lines().join(""), "/rewind ");
        assert_eq!(
            app.input_history.last().map(String::as_str),
            Some("my draft")
        );
        assert_eq!(app.input_mode, InputMode::Editing);
        assert!(
            app.messages.is_empty(),
            "staging a command must not send it"
        );
    }

    #[test]
    fn the_palette_highlight_moves_and_stays_inside_the_matches() {
        let mut app = app_with(FocusArea::ChatInput);
        ctrl(&mut app, 'x');
        press(&mut app, KeyCode::Up);
        assert_eq!(app.palette_selected, 0);
        press(&mut app, KeyCode::Down);
        ctrl(&mut app, 'n');
        assert_eq!(app.palette_selected, 2);
        ctrl(&mut app, 'p');
        assert_eq!(app.palette_selected, 1);
        // Typing resets the highlight to the best match.
        type_text(&mut app, "zzqqxxj");
        assert_eq!(app.palette_selected, 0);
        press(&mut app, KeyCode::Down);
        assert_eq!(app.palette_selected, 0, "no matches, nowhere to move");
        // Enter on no match does nothing and keeps the palette open.
        press(&mut app, KeyCode::Enter);
        assert!(app.palette_visible);
        ctrl(&mut app, 'x');
        assert!(!app.palette_visible, "Ctrl+X closes it again");
    }

    #[test]
    fn ctrl_k_is_still_the_background_tasks_panel() {
        // From a panel: in the conversation Ctrl+K deletes to the end of the line.
        let mut app = app_with(FocusArea::FileExplorer);
        ctrl(&mut app, 'k');
        assert_eq!(app.focus, FocusArea::TaskManager);
        assert!(!app.palette_visible);
    }

    #[test]
    fn esc_stops_a_running_turn_before_it_does_anything_else() {
        let mut app = app_with(FocusArea::ChatInput);
        app.input_mode = InputMode::Editing;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        app.turn_stop = Some(stop.clone());
        app.is_generating = true;
        assert_eq!(press(&mut app, KeyCode::Esc), KeyFlow::Continue);
        assert!(
            stop.load(std::sync::atomic::Ordering::Relaxed),
            "Esc did not set the stop flag"
        );
        assert_eq!(
            app.input_mode,
            InputMode::Editing,
            "the first Esc only stops the turn"
        );
        // A second Esc, with the stop already asked for, is the ordinary key,
        // and in the conversation that never leaves the input (TD-3).
        assert_eq!(press(&mut app, KeyCode::Esc), KeyFlow::Continue);
        assert_eq!(app.input_mode, InputMode::Editing);

        // A ByteBot run is stopped the same way.
        let mut app = app_with(FocusArea::ByteBotPanel);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        app.bytebot_stop = Some(stop.clone());
        app.bytebot_running = true;
        assert_eq!(press(&mut app, KeyCode::Esc), KeyFlow::Continue);
        assert!(stop.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(
            app.focus,
            FocusArea::ByteBotPanel,
            "the panel stays open while it stops"
        );

        // Idle, Esc is the ordinary key.
        let mut idle = app_with(FocusArea::ChatInput);
        idle.input_mode = InputMode::Editing;
        press(&mut idle, KeyCode::Esc);
        assert_eq!(idle.input_mode, InputMode::Editing);
    }

    #[test]
    fn plain_q_asks_once_then_quits_but_not_while_typing() {
        // In the conversation q is a letter (TD-3).
        let mut app = app_with(FocusArea::ChatInput);
        assert_eq!(press(&mut app, KeyCode::Char('q')), KeyFlow::Continue);
        assert_eq!(app.chat_input.lines().join(""), "q");

        let mut app = app_with(FocusArea::FileExplorer);
        assert_eq!(press(&mut app, KeyCode::Char('q')), KeyFlow::Continue);
        assert!(app.quit_armed, "the first q asks");
        assert_eq!(press(&mut app, KeyCode::Char('q')), KeyFlow::Quit);

        // Any other key in between withdraws the question.
        let mut app = app_with(FocusArea::FileExplorer);
        press(&mut app, KeyCode::Char('q'));
        press(&mut app, KeyCode::Down);
        assert!(!app.quit_armed);
        assert_eq!(press(&mut app, KeyCode::Char('q')), KeyFlow::Continue);

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
        use crate::agent_tools::{ApprovalDraft, ApprovalRequest};
        let (responder, answer) = tokio::sync::oneshot::channel();
        // Through the channel the agent loop uses, so the prompt gets its id
        // the way a real one does (EN-1).
        app.approval_tx
            .send((
                ApprovalRequest {
                    tool: tool.into(),
                    class,
                    draft: ApprovalDraft::default(),
                    summary: format!("{tool} src/lib.rs"),
                    preview: "+fn hello() {}\n".into(),
                },
                responder,
            ))
            .unwrap();
        app.drain_agent_channels();
        seen(app);
        answer
    }

    /// The approval in front has been on screen a second, as a person
    /// answering it would have seen it.
    fn seen(app: &mut App) {
        app.approval_shown = app.front_approval_id().map(|id| {
            (
                id,
                std::time::Instant::now() - std::time::Duration::from_secs(1),
            )
        });
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
        assert!(app.agent_grants.lock().unwrap().contains(&ToolClass::Edit));
        // Shell commands are a separate class: granting edits says nothing
        // about running commands.
        assert!(!app.agent_grants.lock().unwrap().contains(&ToolClass::Shell));
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
    fn page_up_and_down_scroll_the_transcript() {
        // TD-3: k and j are letters in the conversation now.
        let mut app = app_with(FocusArea::ChatInput);
        press(&mut app, KeyCode::PageUp);
        assert!(app.chat_scroll > 0);
        press(&mut app, KeyCode::PageDown);
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

    /// J-05: the list is config's, so `s` in this panel means "write config"
    /// and never falls through to the global Settings chord (E2-06). The
    /// parameter keys edit the profile in memory; nothing reaches the disk
    /// without `s`, and `s` says so when persistence is off.
    #[test]
    fn custom_models_keys_edit_apply_and_save() {
        use xencode_config_rs::ModelProfile;
        let mut app = app_with(FocusArea::CustomModels);
        app.model_profiles = vec![ModelProfile {
            name: "tight".to_string(),
            model: "ollama:qwen2.5:7b".to_string(),
            temperature: None,
            max_tokens: None,
            for_task: None,
        }];
        app.models_dirty = true;

        press(&mut app, KeyCode::Char('s'));
        assert_eq!(app.focus, FocusArea::CustomModels, "s is not Settings here");
        assert!(
            app.models_status.contains("nothing was written"),
            "{}",
            app.models_status
        );
        assert!(app.models_dirty, "an unsaved edit is still unsaved");

        press(&mut app, KeyCode::Char('+'));
        assert_eq!(app.model_profiles[0].temperature, Some(1.1));
        press(&mut app, KeyCode::Right);
        assert_eq!(app.model_profiles[0].max_tokens, Some(1024));
        press(&mut app, KeyCode::Left);
        assert_eq!(app.model_profiles[0].max_tokens, Some(512));

        press(&mut app, KeyCode::Enter);
        assert_eq!(app.config.default_model, "ollama:qwen2.5:7b");
        assert_eq!(app.config.llama_cpp_temperature, Some(1.1));
        assert_eq!(app.config.llama_cpp_max_tokens, Some(512));
        // The `s` above snapshotted the profile before it was tuned; apply
        // must not have rewritten it. Only `s` moves the saved list.
        assert_eq!(
            app.config.model_profiles[0].temperature, None,
            "apply ≠ save"
        );
        assert_eq!(app.config.model_profiles[0].max_tokens, None);
    }

    /// MI-7: `f` says which kind of turn this profile may take on its own. It
    /// steps through only the words the prompt reading can produce, so a profile
    /// cannot end up marked for a kind of turn no turn is ever read as, and the
    /// key is handled in this panel rather than falling through to a global
    /// chord (E2-06).
    #[test]
    fn custom_models_f_marks_the_profile_for_a_kind_of_turn() {
        use xencode_config_rs::ModelProfile;
        let mut app = app_with(FocusArea::CustomModels);
        app.model_profiles = vec![ModelProfile {
            name: "fixer".to_string(),
            model: "ollama:qwen2.5:7b".to_string(),
            temperature: None,
            max_tokens: None,
            for_task: None,
        }];
        app.models_dirty = false;

        press(&mut app, KeyCode::Char('f'));
        assert_eq!(app.focus, FocusArea::CustomModels, "f is not a chord");
        assert_eq!(app.model_profiles[0].for_task.as_deref(), Some("bugfix"));
        assert!(app.models_status.contains("fixer"), "{}", app.models_status);

        press(&mut app, KeyCode::Char('f'));
        assert_eq!(
            app.model_profiles[0].for_task.as_deref(),
            Some("general"),
            "the second state is the wide one, and its message has to say so"
        );
        assert!(app.models_status.contains("wide net"));

        press(&mut app, KeyCode::Char('f'));
        assert_eq!(app.model_profiles[0].for_task, None);
        assert!(app.models_status.contains("by hand only"));
        assert!(app.models_dirty, "marking a profile is an unsaved edit");
        assert!(
            app.config
                .model_profiles
                .iter()
                .all(|saved| saved.name != "fixer"),
            "f edits memory; only s writes config"
        );
    }

    /// An empty list is the real state of a fresh config: nothing to apply, and
    /// `n` starts from the session's own settings rather than a seeded row. `n`
    /// and `s` must not reach a global chord such as the one that opens
    /// Settings (E2-06).
    #[test]
    fn custom_models_without_profiles_says_so_instead_of_applying_nothing() {
        let mut app = app_with(FocusArea::CustomModels);
        let before = app.config.default_model.clone();
        assert!(
            app.model_profiles.is_empty(),
            "a fresh config seeds nothing"
        );
        press(&mut app, KeyCode::Enter);
        assert!(app.models_status.contains("no model_profiles"));
        assert_eq!(app.config.default_model, before);
        press(&mut app, KeyCode::Char('+'));
        assert!(app.model_profiles.is_empty());

        press(&mut app, KeyCode::Char('n'));
        assert_eq!(app.focus, FocusArea::CustomModels, "n stays in the panel");
        assert_eq!(app.model_profiles.len(), 1);
        assert_eq!(app.model_profiles[0].model, before);
        assert!(app.models_dirty);

        press(&mut app, KeyCode::Char('s'));
        assert_eq!(app.focus, FocusArea::CustomModels, "s is not Settings");
        assert_eq!(app.config.model_profiles, app.model_profiles);
    }

    /// `s` is the panel's only disk write, so this checks the file: what the
    /// panel showed after editing is what came back out of config.json.
    #[test]
    fn custom_models_save_writes_the_profiles_it_shows() {
        let _guard = CONFIG_DIR.lock().unwrap_or_else(|e| e.into_inner());
        let dir =
            std::env::temp_dir().join(format!("xencode-profiles-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let _config_dir = ConfigDirOverride::set(&dir);

        let mut app = app_with(FocusArea::CustomModels);
        // Opts back into persistence — into the temp XCODE_CONFIG_DIR above.
        app.persist_config = true;
        app.config = XencodeConfig::default();

        press(&mut app, KeyCode::Char('n'));
        press(&mut app, KeyCode::Char('-'));
        press(&mut app, KeyCode::Right);
        press(&mut app, KeyCode::Char('s'));

        let saved = XencodeConfig::load_from(dir.join("config.json")).unwrap();
        assert_eq!(saved.model_profiles, app.model_profiles);
        assert_eq!(saved.model_profiles.len(), 1);
        let profile = &saved.model_profiles[0];
        assert_eq!(profile.model, "qwen2.5:7b");
        assert_eq!(profile.temperature, Some(0.9));
        assert_eq!(profile.max_tokens, Some(1024));
        assert!(!app.models_dirty, "a successful save clears the marker");
        assert!(
            app.models_status.contains("wrote 1 profile"),
            "{}",
            app.models_status
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A config from a newer xencode is not overwritten, and the person has to
    /// find out: every setting they just changed now lives only in memory. The
    /// refusal says it once, not once per keystroke.
    #[test]
    fn a_refused_config_save_says_so_once_and_leaves_the_file_alone() {
        let _guard = CONFIG_DIR.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("xencode-newer-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, b"{\"config_version\": 99, \"default_model\": \"x\"}").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _config_dir = ConfigDirOverride::set(&dir);

        let mut app = App::for_tests();
        app.persist_config = true;

        let refusals = |app: &App| {
            app.messages
                .iter()
                .filter(|m| m.content.contains("config.json unchanged"))
                .count()
        };

        app.save_config();
        app.save_config();
        app.save_config();

        assert_eq!(refusals(&app), 1, "one refusal, not one per keystroke");
        let line = app
            .messages
            .iter()
            .find(|m| m.content.contains("config.json unchanged"))
            .expect("the refusal reaches the transcript");
        assert_eq!(line.role, "system");
        assert!(line.content.contains("99"), "{}", line.content);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            bytes,
            "the newer file is untouched"
        );

        // Once the file is one this binary can write, saving works and the
        // guard resets, so a later refusal would be reported again.
        std::fs::remove_file(&path).unwrap();
        app.save_config();
        assert!(app.last_config_save_note.is_none());
        assert!(path.is_file(), "the config is written at last");
        assert_eq!(refusals(&app), 1, "a success adds nothing");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The same refusal for the damage a hand edit makes — the case that used to
    /// end in a silent reset. The settings panel loads with defaults when the
    /// file cannot be read, and the next change it saves would have written that
    /// default block over the person's own file.
    #[test]
    fn a_settings_save_stops_at_a_config_that_is_not_readable_json() {
        let _guard = CONFIG_DIR.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("xencode-broken-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, b"{\"default_model\": \"ollama:mine\",}").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _config_dir = ConfigDirOverride::set(&dir);

        let mut app = App::for_tests();
        app.persist_config = true;
        app.save_config();

        let line = app
            .messages
            .iter()
            .find(|m| m.content.contains("config.json unchanged"))
            .expect("the refusal reaches the transcript");
        assert!(
            line.content.contains("not readable JSON"),
            "{}",
            line.content
        );
        // Named by path and by what broke, because the person has to go edit it.
        assert!(line.content.contains("config.json"), "{}", line.content);
        assert!(line.content.contains("trailing comma"), "{}", line.content);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            bytes,
            "the unreadable file is untouched"
        );

        std::fs::remove_dir_all(&dir).unwrap();
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

    /// TX-5: one Enter on Factory Reset only arms it; the second performs it;
    /// any other key in between takes the arming back.
    #[tokio::test]
    async fn factory_reset_needs_a_second_enter_and_any_other_key_disarms_it() {
        let mut app = App::for_tests();
        app.config.default_model = "kept-model".to_string();
        app.focus = FocusArea::Settings;
        app.settings_cursor = crate::focus::SETTINGS_ITEMS.len() - 1;

        press(&mut app, KeyCode::Enter);
        assert!(app.settings_reset_armed, "the first Enter arms it");
        assert_eq!(
            app.config.default_model, "kept-model",
            "and changes nothing"
        );

        press(&mut app, KeyCode::Up);
        press(&mut app, KeyCode::Down);
        assert!(
            !app.settings_reset_armed,
            "moving away takes the arming back"
        );
        press(&mut app, KeyCode::Enter);
        assert_eq!(
            app.config.default_model, "kept-model",
            "a fresh first Enter only arms again"
        );

        press(&mut app, KeyCode::Enter);
        assert!(!app.settings_reset_armed);
        assert_ne!(
            app.config.default_model, "kept-model",
            "the second Enter reset it"
        );
    }

    #[test]
    fn a_and_u_review_a_task_only_when_the_command_box_is_empty() {
        use crate::bytebot_tasks::{ByteBotTask, TaskState};
        let mut app = app_with(FocusArea::ByteBotPanel);
        let mut task = ByteBotTask::new("write a note", "m");
        task.state = TaskState::NeedsReview;
        task.changed_files = vec!["note.txt".into()];
        app.bytebot_tasks = vec![task];

        // With text in the box, the letters type.
        type_text(&mut app, "fix ua");
        assert_eq!(app.bytebot_command, "fix ua");
        assert_eq!(app.bytebot_tasks[0].state, TaskState::NeedsReview);

        app.bytebot_command.clear();
        app.bytebot_cursor = 0;
        press(&mut app, KeyCode::Char('a'));
        assert_eq!(app.bytebot_tasks[0].state, TaskState::Completed);
        assert!(app.bytebot_command.is_empty(), "the key was not typed");
    }

    #[test]
    fn enter_on_an_empty_box_does_not_answer_a_question() {
        let mut app = app_with(FocusArea::ByteBotPanel);
        app.bytebot_running = true;
        let (reply, mut answer) = tokio::sync::oneshot::channel();
        app.bytebot_help = Some(reply);
        app.question_id = Some(1);
        type_text(&mut app, "   ");
        press(&mut app, KeyCode::Enter);
        assert!(
            answer.try_recv().is_err(),
            "a blank line was sent as the answer"
        );
        assert!(app.bytebot_help.is_some(), "the question is still waiting");
        type_text(&mut app, "postgres");
        press(&mut app, KeyCode::Enter);
        assert_eq!(answer.try_recv().unwrap(), "postgres");
    }

    #[test]
    fn enter_answers_a_bytebot_question_and_esc_withdraws_it() {
        let mut app = app_with(FocusArea::ByteBotPanel);
        app.bytebot_running = true;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        app.bytebot_stop = Some(stop.clone());
        let (reply, mut answer) = tokio::sync::oneshot::channel();
        app.bytebot_help = Some(reply);
        app.question_id = Some(1);
        type_text(&mut app, "postgres");
        press(&mut app, KeyCode::Enter);
        assert_eq!(answer.try_recv().unwrap(), "postgres");
        assert!(
            app.bytebot_tasks.is_empty(),
            "the answer is not queued as a task"
        );

        // Esc stops the task, and the waiting question must let go too, or the
        // loop would sit inside the tool call and never see the stop.
        let (reply, mut answer) = tokio::sync::oneshot::channel();
        app.bytebot_help = Some(reply);
        app.question_id = Some(1);
        press(&mut app, KeyCode::Esc);
        assert!(stop.load(std::sync::atomic::Ordering::Relaxed));
        assert!(answer.try_recv().is_err(), "the question was withdrawn");
    }

    #[test]
    fn the_bytebot_panel_switches_model_without_leaving() {
        let mut app = app_with(FocusArea::ByteBotPanel);
        app.available_models = vec!["a:1".into(), "b:2".into()];
        app.config.default_model = "a:1".into();
        type_text(&mut app, "/model");
        press(&mut app, KeyCode::Enter);
        assert!(app.bytebot_model_picker, "/model alone opens the list");
        assert!(!app.bytebot_running, "no task started");
        assert_eq!(
            app.bytebot_model_selected, 0,
            "the list starts on the current model"
        );
        press(&mut app, KeyCode::Down);
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.config.default_model, "b:2");
        assert!(!app.bytebot_model_picker);
        assert_eq!(app.focus, FocusArea::ByteBotPanel);

        // Esc closes the list and changes nothing; `/model <name>` switches directly.
        type_text(&mut app, "/model");
        press(&mut app, KeyCode::Enter);
        press(&mut app, KeyCode::Esc);
        assert!(!app.bytebot_model_picker);
        assert_eq!(app.config.default_model, "b:2");
        assert_eq!(app.focus, FocusArea::ByteBotPanel);
        type_text(&mut app, "/model a:1");
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.config.default_model, "a:1");
        assert!(app.bytebot_command.is_empty());
    }

    #[test]
    fn the_floating_badge_row_turns_autostart_on_and_off() {
        let mut app = app_with(FocusArea::Settings);
        assert!(!app.config.badge_autostart, "off by default");
        assert!(super::settings_toggle(&mut app, "Floating Badge"));
        assert!(app.config.badge_autostart);
        assert!(super::settings_toggle(&mut app, "Floating Badge"));
        assert!(!app.config.badge_autostart);
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
                "Rounded Borders",
                "Show Scrollbars",
                "Line Numbers",
                "Mouse Capture",
                "Floating Badge",
                "Disclosure Level",
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
        // I2-02: the command budget sits under it, and the section stays
        // contiguous — Settings prints a header only when the section changes.
        let agent_rows: Vec<&str> = SETTINGS_ITEMS
            .iter()
            .filter(|r| r.section == "Agent")
            .map(|r| r.label)
            .collect();
        assert_eq!(agent_rows, ["Agent Approval", "Command Timeout"]);
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

    /// The seed row behaves like the other typed number rows: blank means
    /// nothing is sent, what is stored comes back when the row is reopened, and
    /// text that is not a number unsets it rather than committing a zero the
    /// server would honour as a real seed.
    #[test]
    fn the_seed_row_edits_and_commits_like_the_other_number_rows() {
        let mut app = app_with(FocusArea::Settings);
        app.config.llama_cpp_seed = None;
        app.settings_cursor = crate::focus::settings_row_index("Llama Seed");
        press(&mut app, KeyCode::Enter);
        assert!(app.settings_url_editing);
        assert_eq!(
            app.settings_url_buffer, "",
            "an unset seed must not show an invented value"
        );
        for ch in "1234".chars() {
            press(&mut app, KeyCode::Char(ch));
        }
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.config.llama_cpp_seed, Some(1234));

        press(&mut app, KeyCode::Enter);
        assert_eq!(app.settings_url_buffer, "1234");
        app.settings_url_buffer.clear();
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.config.llama_cpp_seed, None);

        app.config.llama_cpp_seed = Some(9);
        press(&mut app, KeyCode::Enter);
        app.settings_url_buffer = "twelve".to_string();
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.config.llama_cpp_seed, None);
    }

    #[test]
    fn the_cloud_models_row_switches_consent_and_sits_below_the_keys() {
        use crate::focus::{settings_row_index, SettingKind, SETTINGS_ITEMS};

        let mut app = app_with(FocusArea::Settings);
        app.config = XencodeConfig::default();
        assert!(
            !app.egress_policy().allow_cloud,
            "a session starts confined to this machine"
        );

        app.settings_cursor = settings_row_index("Cloud Models");
        press(&mut app, KeyCode::Right);
        assert!(app.egress_policy().allow_cloud, "the row is the switch");
        press(&mut app, KeyCode::Right);
        assert!(
            !app.egress_policy().allow_cloud,
            "and it can be turned back off"
        );

        // It ends the Providers section, below every key row: a key is
        // transport for a cloud provider, this row is permission to use one,
        // and the panel should read in that order.
        let providers: Vec<&str> = SETTINGS_ITEMS
            .iter()
            .filter(|row| row.section == "Providers")
            .map(|row| row.label)
            .collect();
        assert_eq!(providers.last().copied(), Some("External Workers"));
        let consent = SETTINGS_ITEMS
            .iter()
            .find(|row| row.label == "Cloud Models")
            .unwrap();
        assert_eq!(consent.kind, SettingKind::Toggle);
    }

    /// The worker rule is its own consent, so its row must move its own flag and
    /// nothing else: opening it cannot open the egress policy, and the flip is
    /// said out loud because what it permits is a program xencode cannot watch.
    #[test]
    fn the_external_workers_row_switches_work_not_egress_and_says_what_it_did() {
        use crate::focus::{settings_row_index, SettingKind, SETTINGS_ITEMS};

        let mut app = app_with(FocusArea::Settings);
        app.config = XencodeConfig::default();
        assert_eq!(
            app.config.profile(),
            xencode_core_rs::Profile::LOCAL_ONLY,
            "a session starts with both rules of the posture closed"
        );

        app.settings_cursor = settings_row_index("External Workers");
        press(&mut app, KeyCode::Right);
        assert!(
            app.config.allow_external_workers,
            "the row is the switch for work"
        );
        assert!(
            !app.egress_policy().allow_cloud,
            "and it is not a way to open the model rule"
        );
        assert!(app
            .toasts
            .iter()
            .any(|t| t.message.contains("another vendor's agent")),);

        press(&mut app, KeyCode::Right);
        assert!(!app.config.allow_external_workers);
        assert_eq!(app.config.profile(), xencode_core_rs::Profile::LOCAL_ONLY);
        assert!(app
            .toasts
            .iter()
            .any(|t| t.message.contains("refused by that name")));

        let row = SETTINGS_ITEMS
            .iter()
            .find(|row| row.label == "External Workers")
            .unwrap();
        assert_eq!(row.kind, SettingKind::Toggle);
    }

    #[test]
    fn settings_steps_persist_through_the_config_dir() {
        // Pointing saves at a temp dir means this test — and any concurrent
        // save — never writes the user's real ~/.xencode.
        use crate::focus::{settings_row_index, SETTINGS_ITEMS};

        let _guard = CONFIG_DIR.lock().unwrap_or_else(|e| e.into_inner());

        let dir =
            std::env::temp_dir().join(format!("xencode-settings-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let _config_dir = ConfigDirOverride::set(&dir);

        let mut app = app_with(FocusArea::Settings);
        // The point of this test is the round trip through disk, so it opts
        // back into persistence — into the temp `XCODE_CONFIG_DIR` above.
        app.persist_config = true;
        app.config = XencodeConfig::default();

        // Toggles flip.
        app.settings_cursor = settings_row_index("Rounded Borders");
        press(&mut app, KeyCode::Right);
        // Rounded is the default (TD-2), so the first press turns it off.
        assert!(!app.config.rounded_borders);

        // Agent Approval cycles the five modes and wraps (I1-01, MD-1).
        app.settings_cursor = settings_row_index("Agent Approval");
        assert_eq!(app.config.agent_approval, "ask");
        for expected in [
            "edit-allow",
            "all-allow",
            "plan",
            "autonomous",
            "ask",
            "edit-allow",
        ] {
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

        // I2-02: the agent's foreground-command budget is a stepped row too,
        // so it is adjustable without leaving the TUI.
        app.settings_cursor = settings_row_index("Command Timeout");
        assert_eq!(app.config.agent_command_timeout, 30, "config default");
        press(&mut app, KeyCode::Right);
        assert_eq!(app.config.agent_command_timeout, 35);
        for _ in 0..20 {
            press(&mut app, KeyCode::Left);
        }
        assert_eq!(
            app.config.agent_command_timeout, 5,
            "stepped below min at the row's own floor"
        );

        // Navigation bounds derive from the table.
        app.settings_cursor = SETTINGS_ITEMS.len() - 1;
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.settings_cursor, SETTINGS_ITEMS.len() - 1);

        // Esc closes Settings and persists everything above.
        press(&mut app, KeyCode::Esc);
        assert_eq!(app.focus, FocusArea::ChatInput);
        let saved = XencodeConfig::load_from(dir.join("config.json")).unwrap();
        assert!(!saved.rounded_borders);
        assert_eq!(saved.response_timeout, 5);
        assert_eq!(saved.agent_approval, "edit-allow");
        assert_eq!(saved.agent_command_timeout, 5);

        let _ = std::fs::remove_dir_all(&dir);
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
        // On an empty input the first Ctrl+C asks and the second quits.
        press_with_mods(&mut app, KeyCode::Char('c'), KeyModifiers::CONTROL);
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
        let mut app = app_with(FocusArea::FileExplorer);
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
    fn opening_the_worktree_panel_resets_its_prompt() {
        // TD-3: Ctrl+O unfolds tool output now; the panel is in the Agents view.
        let mut app = app_with(FocusArea::ChatInput);
        app.worktree_prompt = crate::focus::WorktreePrompt::AddPath;
        app.open_panel(FocusArea::WorktreePanel, None);
        assert_eq!(app.focus, FocusArea::WorktreePanel);
        assert_eq!(app.worktree_prompt, crate::focus::WorktreePrompt::None);
        // Read-only listing of the repo the tests run in: at least the main
        // worktree, and no side effects.
        assert!(!app.worktrees.is_empty());
        press(&mut app, KeyCode::Esc);
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
    async fn ctrl_l_toggles_the_models_view() {
        // TD-3: insights moved into the Project view; Ctrl+L is the models.
        let mut app = app_with(FocusArea::ChatInput);
        press_with_mods(&mut app, KeyCode::Char('l'), KeyModifiers::CONTROL);
        assert_eq!(app.focus, FocusArea::ModelSelector);
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
