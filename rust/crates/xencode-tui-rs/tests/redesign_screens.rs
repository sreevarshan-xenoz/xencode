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
