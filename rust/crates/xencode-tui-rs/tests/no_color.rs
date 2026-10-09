//! `NO_COLOR` (https://no-color.org) and a focus mark that is not colour
//! alone (TX-8), checked on a real rendered frame.

use ratatui::style::{Color, Modifier};
use ratatui::{backend::TestBackend, Terminal};
use xencode_tui_rs::app::{App, InputMode};
use xencode_tui_rs::focus::FocusArea;
use xencode_tui_rs::ui::draw;

fn frame(app: &mut App) -> ratatui::buffer::Buffer {
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal.draw(|f| draw(f, app)).unwrap();
    terminal.backend().buffer().clone()
}

fn text(buffer: &ratatui::buffer::Buffer) -> String {
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
fn no_color_leaves_no_colour_and_keeps_marks_as_reverse_video() {
    let mut app = App::for_tests();
    app.input_mode = InputMode::Normal;
    app.no_color = true;
    let buffer = frame(&mut app);
    for cell in buffer.content.iter() {
        assert_eq!(cell.fg, Color::Reset, "a coloured foreground: {cell:?}");
        assert_eq!(cell.bg, Color::Reset, "a coloured background: {cell:?}");
    }
    // The status bar had its own background; it is still told apart.
    let last = buffer.area().height - 1;
    assert!(
        buffer[(2, last)].modifier.contains(Modifier::REVERSED),
        "the status bar lost its mark"
    );
}

#[test]
fn colour_is_kept_when_no_color_is_not_set() {
    let mut app = App::for_tests();
    app.no_color = false;
    let buffer = frame(&mut app);
    assert!(buffer.content.iter().any(|c| c.fg != Color::Reset));
}

#[test]
fn the_focused_pane_is_marked_in_its_title() {
    let mut app = App::for_tests();
    app.input_mode = InputMode::Normal;
    app.focus = FocusArea::FileExplorer;
    let screen = text(&frame(&mut app));
    let marked: Vec<&str> = screen.lines().filter(|l| l.contains('▶')).collect();
    assert!(
        marked
            .iter()
            .any(|l| l.contains("▶ Files") || l.contains("▶Files")),
        "the explorer title is not marked: {marked:?}"
    );
    assert_eq!(
        screen.matches('▶').count(),
        1,
        "only the focused pane is marked: {screen}"
    );
}
