//! BT-5: the ByteBot panel draws its model list, marking the current model.
use ratatui::{backend::TestBackend, Terminal};
use xencode_tui_rs::app::{App, FocusArea};
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
fn the_panel_lists_models_and_marks_the_current_one() {
    let mut app = App::for_tests();
    app.focus = FocusArea::ByteBotPanel;
    app.available_models = vec!["llamacpp:qwen3-4b".into(), "qwen2.5:7b".into()];
    app.config.default_model = "qwen2.5:7b".into();
    app.bytebot_model_picker = true;
    app.bytebot_model_selected = 0;
    let screen = rendered(&mut app, 120, 40);
    assert!(screen.contains("Model"), "{screen}");
    assert!(screen.contains("llamacpp:qwen3-4b"), "{screen}");
    assert!(screen.contains("qwen2.5:7b (current)"), "{screen}");
    assert!(
        screen.contains("› llamacpp:qwen3-4b"),
        "the lit row is marked: {screen}"
    );

    app.available_models.clear();
    let screen = rendered(&mut app, 120, 40);
    assert!(screen.contains("No models found yet"), "{screen}");
}
