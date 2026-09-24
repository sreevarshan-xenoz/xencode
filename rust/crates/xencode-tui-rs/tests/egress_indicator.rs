//! The status bar has to say which egress rule is in force (PR-2), because a
//! local-first product that never states its posture leaves the user guessing
//! whether a prompt stayed on this machine.
//!
//! Rendered rather than asserted on a function: the point is that the sentence
//! reaches the screen, and changes when the setting does.

use ratatui::{backend::TestBackend, Terminal};
use xencode_tui_rs::app::{App, InputMode};
use xencode_tui_rs::ui::draw;

/// Everything the app paints into a normal-sized terminal, as text.
fn rendered(app: &mut App) -> String {
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
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
fn the_status_bar_says_the_session_is_confined_to_this_machine() {
    let mut app = App::for_tests();
    app.input_mode = InputMode::Normal;
    assert!(
        !app.config.allow_cloud_models,
        "the test is about the shipped default"
    );

    let screen = rendered(&mut app);

    assert!(
        screen.contains("local only"),
        "the bar should state the rule, got {screen:?}"
    );
    assert!(
        !screen.contains("cloud allowed"),
        "it should not describe a permission nobody granted, got {screen:?}"
    );
}

#[test]
fn the_status_bar_changes_when_cloud_access_is_granted() {
    let mut app = App::for_tests();
    app.input_mode = InputMode::Normal;
    app.config.allow_cloud_models = true;

    let screen = rendered(&mut app);

    assert!(
        screen.contains("cloud allowed"),
        "an opted-in session should say so, got {screen:?}"
    );
    assert!(
        !screen.contains("local only"),
        "the bar must not claim a rule the setting no longer applies, got {screen:?}"
    );
}
