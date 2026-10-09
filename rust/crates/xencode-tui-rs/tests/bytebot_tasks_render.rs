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

#[test]
fn a_waiting_question_is_shown_with_how_to_answer_it() {
    let mut app = App::for_tests();
    app.focus = FocusArea::ByteBotPanel;
    let mut waiting = task("pick a database", TaskState::NeedsHelp);
    waiting.question = Some("Which database?".into());
    app.bytebot_tasks = vec![waiting];
    let screen = rendered(&mut app, 140, 45);
    assert!(screen.contains("Needs help: Which database?"), "{screen}");
    assert!(screen.contains("/done"), "{screen}");
    assert!(screen.contains("needs help  pick a database"), "{screen}");
}

#[test]
fn a_task_waiting_for_review_lists_its_files_and_the_keys() {
    let mut app = App::for_tests();
    app.focus = FocusArea::ByteBotPanel;
    let mut reviewing = task("write a note", TaskState::NeedsReview);
    reviewing.changed_files = vec!["src/lib.rs".into(), "note.txt".into()];
    app.bytebot_tasks = vec![reviewing];
    let screen = rendered(&mut app, 140, 45);
    assert!(screen.contains("Review 2 changed file(s)"), "{screen}");
    assert!(screen.contains("src/lib.rs"), "{screen}");
    assert!(screen.contains("a accepts"), "{screen}");
    assert!(screen.contains("u undoes"), "{screen}");
}
