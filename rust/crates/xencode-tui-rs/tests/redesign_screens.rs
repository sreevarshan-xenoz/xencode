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
    app.apply_token(r#"[TOOLEND]{"id":"c1","outcome":"done","preview":"fn a() {}"}"#, &tx);
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
            preview: "@@ -40,3 +40,4 @@\n     let level = self.indent();\n+    self.close_level();\n"
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
    assert!(rows[29].contains("ctrl+x"), "{:?}", rows[29]);
    assert!(rows[26].contains("╭") && rows[28].contains("╰"), "{rows:#?}");
    assert!(rows[29].contains(&app.config.default_model), "{:?}", rows[29]);
}

/// TD-2: the side panel shows at wide sizes and hides under 100 columns.
#[test]
fn the_side_panel_shows_at_wide_sizes_and_hides_under_100_columns() {
    let mut app = App::for_tests();
    app.toasts.clear();
    app.side_panel_open = true;
    let wide = screen(&mut app, 160, 40).join("\n");
    assert!(wide.contains("Context") && wide.contains("Changes"), "{wide}");
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
        assert!(rows.iter().any(|r| r.contains("╰")), "input box lost at {w}");
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
    assert!(text.contains("src/parser.rs") && text.contains("+    self.close_level();"), "{text}");
    assert!(text.contains("y allow") && text.contains("n deny"), "{text}");
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
