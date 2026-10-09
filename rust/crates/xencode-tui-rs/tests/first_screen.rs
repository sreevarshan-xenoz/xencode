//! What a new user sees first, at the 80-column size many terminals open at
//! (TX-6, TX-7): help reachable from the status bar without being cut off,
//! a welcome that says how to start and where to find everything, and the
//! state of the provider actually in use — not Ollama's whatever the model.

use ratatui::{backend::TestBackend, Terminal};
use xencode_tui_rs::app::{App, InputMode};
use xencode_tui_rs::ui::draw;

fn rendered(app: &mut App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| draw(frame, app)).unwrap();
    let buffer = terminal.backend().buffer();
    (0..buffer.area().height)
        .map(|y| {
            (0..buffer.area().width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn help_is_on_the_status_bar_at_80_columns() {
    let mut app = App::for_tests();
    app.input_mode = InputMode::Normal;
    let screen = rendered(&mut app, 80, 24);
    assert!(screen.contains("?:help"), "got {screen}");
}

#[test]
fn an_empty_session_says_how_to_start_and_where_everything_is() {
    let mut app = App::for_tests();
    app.messages.clear();
    app.config.default_model = "llamacpp:qwen3-4b".to_string();
    let screen = rendered(&mut app, 120, 40);
    assert!(screen.contains("Welcome to Xencode"), "got {screen}");
    // The chat pane wraps these, so their words are checked line by line.
    let lines = xencode_tui_rs::ui::welcome_lines(&app).join(
        "
",
    );
    assert!(lines.contains("type a prompt and press Enter"), "{lines}");
    assert!(
        lines.contains("F1 or ? keys · / commands · Ctrl+F all panels"),
        "{lines}"
    );
    // The provider named is the model's own, not Ollama.
    assert!(
        lines.contains("Model: llamacpp:qwen3-4b (llamacpp "),
        "{lines}"
    );
}

#[test]
fn the_provider_state_follows_its_health_check() {
    let mut app = App::for_tests();
    app.config.default_model = "llamacpp:qwen3-4b".to_string();
    app.ollama_health_entries
        .insert("llamacpp".to_string(), ("healthy".to_string(), 12.0, None));
    assert_eq!(app.provider_status(), "llamacpp ready");
    app.ollama_health_entries.insert(
        "llamacpp".to_string(),
        (
            "unavailable".to_string(),
            0.0,
            Some("connection refused".into()),
        ),
    );
    assert_eq!(app.provider_status(), "llamacpp not running");
}
