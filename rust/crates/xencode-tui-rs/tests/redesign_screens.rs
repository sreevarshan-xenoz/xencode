//! The redesigned screens (TD): drawn with the real `ui::draw` into a test
//! terminal and read back as rows of text.

use ratatui::{backend::TestBackend, Terminal};
use xencode_tui_rs::{app::App, ui::draw};

/// Each row of the screen, as the person would read it.
fn screen(app: &mut App<'static>, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal.draw(|f| draw(f, app)).unwrap();
    let buf = terminal.backend().buffer().clone();
    (0..h)
        .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
        .collect()
}

/// TD-1: a tool call is one `⎿` row on the screen, with its outcome, and no
/// `⚙` line repeats it.
#[tokio::test]
async fn a_tool_call_is_drawn_once_as_one_row() {
    let mut app = App::for_tests();
    app.toasts.clear();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    app.is_generating = true;
    app.apply_token("[TOOL]→ read_file src/a.rs", &tx);
    app.apply_token(
        r#"[TOOLCALL]{"id":"c1","name":"read_file","arguments":{"path":"src/a.rs"}}"#,
        &tx,
    );
    app.apply_token("[TOOL]← fn a() {}", &tx);
    app.apply_token(
        r#"[TOOLEND]{"id":"c1","outcome":"done","preview":"fn a() {}"}"#,
        &tx,
    );
    app.apply_token("[DONE]", &tx);
    let rows = screen(&mut app, 160, 40);
    let tool_rows: Vec<&String> = rows.iter().filter(|r| r.contains("⎿")).collect();
    assert_eq!(tool_rows.len(), 1, "{rows:#?}");
    assert!(tool_rows[0].contains("read_file src/a.rs") && tool_rows[0].contains("✓"));
    assert!(!rows.iter().any(|r| r.contains("⚙")), "{rows:#?}");
}

/// An edit waiting for the person's yes, with a real diff in it.
fn queue_edit_approval(app: &mut App<'static>) {
    let (responder, _answer) = tokio::sync::oneshot::channel();
    app.approval_queue.push_back((
        xencode_tui_rs::agent_tools::ApprovalRequest {
            tool: "edit_file".into(),
            class: xencode_tui_rs::agent_tools::ToolClass::Edit,
            summary: "edit_file src/parser.rs".into(),
            preview:
                "@@ -40,3 +40,4 @@\n     let level = self.indent();\n+    self.close_level();\n"
                    .into(),
            draft: Default::default(),
        },
        responder,
    ));
}

/// TD-2: the input box sits on the last rows with the status line under it.
#[test]
fn the_input_sits_on_the_last_rows_and_the_status_line_under_it() {
    let mut app = App::for_tests();
    app.toasts.clear();
    let rows = screen(&mut app, 100, 30);
    assert!(rows[29].contains("ctrl+p"), "{:?}", rows[29]);
    assert!(
        rows[26].contains("╭") && rows[28].contains("╰"),
        "{rows:#?}"
    );
    assert!(
        rows[29].contains(&app.config.default_model),
        "{:?}",
        rows[29]
    );
}

/// TD-2: the side panel shows at wide sizes and hides under 100 columns.
#[test]
fn the_side_panel_shows_at_wide_sizes_and_hides_under_100_columns() {
    let mut app = App::for_tests();
    app.toasts.clear();
    app.side_panel_open = true;
    let wide = screen(&mut app, 160, 40).join("\n");
    assert!(
        wide.contains("Context") && wide.contains("Changes"),
        "{wide}"
    );
    let narrow = screen(&mut app, 80, 24).join("\n");
    assert!(!narrow.contains("Context"), "{narrow}");
    app.side_panel_open = false;
    let closed = screen(&mut app, 160, 40).join("\n");
    assert!(!closed.contains("Context"), "{closed}");
}

/// Review focus 4: a terminal resized from wide to narrow keeps the frame
/// whole — nothing panics, the input box stays, the status line fits.
#[test]
fn a_resize_from_wide_to_narrow_keeps_the_frame_whole() {
    let mut app = App::for_tests();
    app.side_panel_open = true;
    for w in [160u16, 120, 100, 99, 80, 60, 40, 20] {
        let rows = screen(&mut app, w, 24);
        assert!(
            rows.iter().any(|r| r.contains("╰")),
            "input box lost at {w}"
        );
        assert!(!rows[23].trim().is_empty(), "status line lost at {w}");
    }
}

/// TD-2: an approval is drawn in the conversation with its diff and keys.
#[test]
fn an_approval_is_drawn_in_the_conversation_with_its_diff() {
    let mut app = App::for_tests();
    app.toasts.clear();
    queue_edit_approval(&mut app);
    let text = screen(&mut app, 120, 36).join("\n");
    assert!(
        text.contains("src/parser.rs") && text.contains("+    self.close_level();"),
        "{text}"
    );
    assert!(
        text.contains("y allow") && text.contains("n deny"),
        "{text}"
    );
}

/// A shell command waiting for the person's yes.
fn queue_shell_approval(app: &mut App<'static>, summary: &str, preview: &str) {
    let (responder, _answer) = tokio::sync::oneshot::channel();
    app.approval_queue.push_back((
        xencode_tui_rs::agent_tools::ApprovalRequest {
            tool: "run_command".into(),
            class: xencode_tui_rs::agent_tools::ToolClass::Shell,
            summary: summary.into(),
            preview: preview.into(),
            draft: Default::default(),
        },
        responder,
    ));
}

/// TD-2: a command longer than the box is wrapped, not cut: its tail is the
/// part that must not be hidden from the person approving it.
#[test]
fn a_long_command_is_shown_whole_in_the_approval() {
    let mut app = App::for_tests();
    app.toasts.clear();
    let command = format!("echo {} TAILMARK", "x".repeat(150));
    queue_shell_approval(&mut app, "run_command", &command);
    let text = screen(&mut app, 80, 30).join("\n");
    assert!(text.contains("TAILMARK"), "{text}");
}

/// TD-2: a change longer than the box says how much is out of sight, and
/// scrolls to show the rest.
#[test]
fn a_long_change_says_how_much_is_hidden_and_scrolls() {
    let mut app = App::for_tests();
    app.toasts.clear();
    let preview: Vec<String> = (1..=60).map(|i| format!("+line {i:02}")).collect();
    queue_shell_approval(&mut app, "write_file big.rs", &preview.join("\n"));
    let text = screen(&mut app, 100, 30).join("\n");
    assert!(
        text.contains("+line 01") && !text.contains("+line 60"),
        "{text}"
    );
    assert!(
        text.contains("more") && text.contains("↑↓ scroll"),
        "{text}"
    );
    app.approval_scroll = 1000;
    let text = screen(&mut app, 100, 30).join("\n");
    assert!(
        text.contains("+line 60") && text.contains("above"),
        "{text}"
    );
    assert!(text.contains("y allow"), "{text}");
}

/// TD-2: the summary of a call is drawn without terminal escape codes, so a
/// model cannot hide part of what it asks for.
#[test]
fn an_approval_summary_carries_no_escape_codes() {
    let mut app = App::for_tests();
    app.toasts.clear();
    queue_shell_approval(&mut app, "run_command \u{1b}[8mhidden part", "ls");
    let text = screen(&mut app, 100, 30).join("\n");
    assert!(text.contains("hidden part"), "{text}");
    assert!(!text.contains('\u{1b}'), "{text:?}");
}

/// TD-2: a fresh screen says what to do in one line, not a list of keys.
#[test]
fn the_empty_conversation_says_what_to_do_in_one_line() {
    let mut app = App::for_tests();
    app.toasts.clear();
    let text = screen(&mut app, 120, 36).join("\n");
    assert!(text.contains("What should xencode do?"), "{text}");
    assert!(!text.contains("Shortcuts:"), "{text}");
}

// ── TD-3: always-ready keys, the seven views and the palette ────────────────

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use xencode_tui_rs::screens::Screen;

fn key(
    app: &mut App<'static>,
    code: KeyCode,
    mods: KeyModifiers,
) -> xencode_tui_rs::keymap::KeyFlow {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    xencode_tui_rs::keymap::handle_key(app, KeyEvent::new(code, mods), &tx)
}

fn type_text(app: &mut App<'static>, text: &str) {
    for c in text.chars() {
        key(app, KeyCode::Char(c), KeyModifiers::NONE);
    }
}

fn draft(app: &App<'static>) -> String {
    app.chat_input.lines().join("\n")
}

/// TD-3: letters are text in the conversation, not shortcuts.
#[test]
fn typing_goes_to_the_input_without_pressing_i() {
    let mut app = App::for_tests();
    type_text(&mut app, "fix q s m");
    assert_eq!(draft(&app), "fix q s m");
    assert_eq!(app.screen(), Screen::Conversation);
}

/// TD-3: Ctrl+P opens the palette, and every view is in it.
#[test]
fn ctrl_p_opens_the_palette_with_every_view() {
    let mut app = App::for_tests();
    key(&mut app, KeyCode::Char('p'), KeyModifiers::CONTROL);
    assert!(app.palette_visible);
    let entries = app.palette_matches();
    for s in Screen::ALL {
        assert!(
            entries
                .iter()
                .any(|e| e.target == xencode_tui_rs::palette::PaletteTarget::Screen(s)),
            "{s:?}"
        );
    }
}

/// TD-3: choosing a view in the palette opens it. (The Models view loads the
/// model list in the background, so this runs on a runtime.)
#[tokio::test]
async fn choosing_a_view_in_the_palette_opens_it() {
    let mut app = App::for_tests();
    key(&mut app, KeyCode::Char('p'), KeyModifiers::CONTROL);
    type_text(&mut app, "models");
    key(&mut app, KeyCode::Enter, KeyModifiers::NONE);
    assert!(!app.palette_visible);
    assert_eq!(app.screen(), Screen::Models);
}

/// TD-3: a half-written message survives answering an approval. The prompt
/// arrives the way the agent's does, through the app's approval channel.
#[tokio::test]
async fn a_draft_survives_an_approval() {
    let mut app = App::for_tests();
    type_text(&mut app, "half a mess");
    let (responder, answer) = tokio::sync::oneshot::channel();
    app.approval_tx
        .send((
            xencode_tui_rs::agent_tools::ApprovalRequest {
                tool: "edit_file".into(),
                class: xencode_tui_rs::agent_tools::ToolClass::Edit,
                summary: "edit_file src/parser.rs".into(),
                preview: "+    self.close_level();".into(),
                draft: Default::default(),
            },
            responder,
        ))
        .unwrap();
    app.drain_agent_channels();
    assert!(app.pending_approval().is_some());
    key(&mut app, KeyCode::Char('n'), KeyModifiers::NONE);
    assert!(app.pending_approval().is_none());
    assert_eq!(
        answer.await.unwrap(),
        xencode_tui_rs::agent_tools::ApprovalAnswer::Denied
    );
    assert_eq!(draft(&app), "half a mess");
}

/// TD-3: Esc twice puts the last message back in the input to edit.
#[test]
fn esc_esc_puts_the_last_message_back_in_the_input() {
    let mut app = App::for_tests();
    app.messages.push(xencode_tui_rs::app::UiMessage {
        role: "user".into(),
        content: "earlier ask".into(),
    });
    key(&mut app, KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(draft(&app), "", "one Esc changes nothing");
    key(&mut app, KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(draft(&app), "earlier ask");
    assert_eq!(app.screen(), Screen::Conversation);
}

/// TD-3: Esc leaves a view and comes back to the conversation.
#[test]
fn esc_leaves_a_view_and_returns_to_the_conversation() {
    let mut app = App::for_tests();
    app.open_screen(Screen::Settings);
    assert_eq!(app.screen(), Screen::Settings);
    key(&mut app, KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(app.screen(), Screen::Conversation);
}

/// TD-3: Ctrl+C clears the input, then asks, then quits.
#[test]
fn ctrl_c_clears_the_input_then_asks_then_quits() {
    use xencode_tui_rs::keymap::KeyFlow;
    let mut app = App::for_tests();
    type_text(&mut app, "abc");
    let flow = key(&mut app, KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert!(!matches!(flow, KeyFlow::Quit));
    assert_eq!(draft(&app), "");
    let flow = key(&mut app, KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert!(
        !matches!(flow, KeyFlow::Quit),
        "the first press on an empty input asks"
    );
    assert!(app
        .toasts
        .iter()
        .any(|t| t.message.contains("Ctrl+C again quits")));
    let flow = key(&mut app, KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert!(matches!(flow, KeyFlow::Quit));
}

/// TD-3: Shift+Tab steps through ask → auto-edit → plan.
#[test]
fn shift_tab_cycles_the_approval_mode() {
    let mut app = App::for_tests();
    assert_eq!(app.config.agent_approval, "ask");
    for expected in ["edit-allow", "plan", "ask"] {
        key(&mut app, KeyCode::BackTab, KeyModifiers::SHIFT);
        assert_eq!(app.config.agent_approval, expected);
    }
}

/// TD-3: Ctrl+O unfolds the latest tool call's output, and folds it again.
#[test]
fn ctrl_o_unfolds_the_latest_tool_output() {
    let mut app = App::for_tests();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    app.is_generating = true;
    app.apply_token("[TOOL]→ read_file a.rs", &tx);
    app.apply_token(
        r#"[TOOLCALL]{"id":"c7","name":"read_file","arguments":{}}"#,
        &tx,
    );
    app.apply_token(
        r#"[TOOLEND]{"id":"c7","outcome":"done","preview":"1\n2\n3\n4\n5\n6"}"#,
        &tx,
    );
    key(&mut app, KeyCode::Char('o'), KeyModifiers::CONTROL);
    assert_eq!(app.tool_expanded.as_deref(), Some("c7"));
    key(&mut app, KeyCode::Char('o'), KeyModifiers::CONTROL);
    assert_eq!(app.tool_expanded, None);
}

/// TD-3: Ctrl+E opens the files view from an empty input, and is still
/// end-of-line while there is text.
#[test]
fn ctrl_e_opens_files_from_an_empty_input() {
    let mut app = App::for_tests();
    type_text(&mut app, "ab");
    key(&mut app, KeyCode::Home, KeyModifiers::NONE);
    key(&mut app, KeyCode::Char('e'), KeyModifiers::CONTROL);
    assert_eq!(app.screen(), Screen::Conversation);
    assert_eq!(app.chat_input.cursor(), (0, 2));
    let mut app = App::for_tests();
    key(&mut app, KeyCode::Char('e'), KeyModifiers::CONTROL);
    assert_eq!(app.screen(), Screen::Files);
}

/// TD-3: Ctrl+L opens the models view.
#[tokio::test]
async fn ctrl_l_opens_the_models_view() {
    let mut app = App::for_tests();
    key(&mut app, KeyCode::Char('l'), KeyModifiers::CONTROL);
    assert_eq!(app.screen(), Screen::Models);
}

/// TD-3: `?` on an empty input is help; anywhere else it is a question mark.
#[test]
fn a_question_mark_opens_help_only_on_an_empty_input() {
    let mut app = App::for_tests();
    type_text(&mut app, "why");
    key(&mut app, KeyCode::Char('?'), KeyModifiers::NONE);
    assert_eq!(draft(&app), "why?");
    assert!(!app.help_visible);
    let mut app = App::for_tests();
    key(&mut app, KeyCode::Char('?'), KeyModifiers::NONE);
    assert!(app.help_visible);
}

/// TD-3: a message sent while a turn runs waits, and goes once it ends.
#[tokio::test]
async fn a_message_sent_while_a_turn_runs_is_queued_and_sent_after() {
    let mut app = App::for_tests();
    app.toasts.clear();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    app.is_generating = true;
    type_text(&mut app, "/help");
    key(&mut app, KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(app.queued, vec!["/help".to_string()]);
    assert_eq!(draft(&app), "");
    let shown = screen(&mut app, 120, 30).join("\n");
    assert!(shown.contains("queued: /help"), "{shown}");
    app.apply_token("[DONE]", &tx);
    assert!(app.queued.is_empty());
    assert!(
        app.messages
            .iter()
            .any(|m| m.role == "user" && m.content == "/help"),
        "{:?}",
        app.messages
    );
}

/// TD-3: `!cmd` runs a shell command through the same approval as the agent's.
#[tokio::test]
async fn a_bang_command_asks_before_it_runs() {
    let mut app = App::for_tests();
    type_text(&mut app, "!echo hi");
    key(&mut app, KeyCode::Enter, KeyModifiers::NONE);
    let mut waited = 0;
    while app.pending_approval().is_none() && waited < 100 {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        app.drain_agent_channels();
        waited += 1;
    }
    let request = app
        .pending_approval()
        .unwrap_or_else(|| panic!("no approval for the command: {:?}", app.messages));
    assert_eq!(request.class, xencode_tui_rs::agent_tools::ToolClass::Shell);
    assert!(request.summary.contains("echo hi"), "{}", request.summary);
}

/// TD-3: `@` followed by text lists the project's files that match, and Tab
/// puts the best one in.
#[test]
fn an_at_word_lists_matching_files() {
    let mut app = App::for_tests();
    app.file_tree = vec![
        "src/main.rs".into(),
        "Cargo.toml".into(),
        "docs/cargo-notes.md".into(),
    ];
    type_text(&mut app, "look at @cargo");
    let matches = app.mention_matches();
    assert_eq!(
        matches.first().map(String::as_str),
        Some("Cargo.toml"),
        "{matches:?}"
    );
    key(&mut app, KeyCode::Tab, KeyModifiers::NONE);
    assert_eq!(draft(&app), "look at @Cargo.toml ");
}
