//! BT-1: the ByteBot panel lists its tasks with their states in words.
use ratatui::{backend::TestBackend, Terminal};
use xencode_tui_rs::app::{App, FocusArea};
use xencode_tui_rs::bytebot_tasks::{ByteBotTask, TaskState};
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

fn task(text: &str, state: TaskState) -> ByteBotTask {
    let mut t = ByteBotTask::new(text, "m");
    t.state = state;
    t
}

#[test]
fn the_panel_lists_tasks_with_their_states() {
    let mut app = App::for_tests();
    app.focus = FocusArea::ByteBotPanel;
    app.bytebot_tasks = vec![
        task("first job", TaskState::Completed),
        task("second job", TaskState::Running),
        task("third job", TaskState::Pending),
    ];
    let screen = rendered(&mut app, 140, 45);
    assert!(screen.contains("Tasks"), "{screen}");
    assert!(screen.contains("completed  first job"), "{screen}");
    assert!(screen.contains("› running  second job"), "{screen}");
    assert!(screen.contains("pending  third job"), "{screen}");
}
