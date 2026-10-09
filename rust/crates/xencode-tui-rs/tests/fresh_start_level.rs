//! AG-4: what a person sees the first time xencode starts, with no settings
//! file at all. The app is built by `App::new()`, the same path `xencode tui`
//! takes, against an empty settings directory, and the counts below are read
//! off the rendered screen, not from the destination table.
//!
//! This file holds one test because it sets `XCODE_CONFIG_DIR` for the whole
//! test process; a second test here would race it.

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

#[tokio::test]
async fn a_fresh_start_is_simple_and_says_where_everything_else_is() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("XCODE_CONFIG_DIR", dir.path());
    // Plugins live elsewhere unless told; keep this run away from real ones.
    std::env::set_var("XCODE_PLUGIN_DIR", dir.path().join("plugins"));

    let mut app = App::new();
    let level = app.config.disclosure_level;

    let first = rendered(&mut app, 120, 40);
    let shortcuts = xencode_tui_rs::focus::first_run_shortcuts_line(app.active_disclosure_level());
    let shortcut_count = shortcuts.split(", ").filter(|s| !s.is_empty()).count();

    app.focus = FocusArea::FeatureNavigator;
    let navigator = rendered(&mut app, 120, 40);
    let listed: Vec<&str> = app
        .palette_items()
        .iter()
        .map(|(name, _, _)| *name)
        .filter(|name| navigator.contains(name))
        .collect();

    println!("FRESH START: disclosure level {level}");
    println!("FRESH START: {shortcut_count} shortcuts on the first screen: {shortcuts}");
    println!(
        "FRESH START: {} panels on screen in the Ctrl+F navigator: {}",
        listed.len(),
        listed.join(", ")
    );

    assert_eq!(level, 2, "a fresh start should open at the Workflow level");
    assert!(
        first.contains("Ctrl+X"),
        "the first screen must say how to reach what the level hides:\n{first}"
    );
    assert_eq!(listed.len(), app.palette_items().len());

    // `/level 4` goes through `set_disclosure_level`; every panel comes back
    // into the navigator, and the choice is saved in the settings file.
    app.set_disclosure_level(xencode_tui_rs::focus::DisclosureLevel::Level4);
    let all = xencode_tui_rs::focus::DESTINATIONS.len() - 1; // not the navigator itself
    assert_eq!(app.palette_items().len(), all);
    let saved = std::fs::read_to_string(dir.path().join("config.json")).unwrap();
    assert!(saved.contains("\"disclosure_level\": 4"), "{saved}");
}
