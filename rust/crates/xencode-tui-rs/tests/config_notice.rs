//! The first frame of a session whose `config.json` could not be read.
//!
//! `DF-1` stopped xencode from overwriting an unreadable settings file; the
//! interactive screen was left starting on defaults with nothing said, so the
//! first thing a person heard was a save refusal *after* they had already
//! changed settings they thought were live. These tests drive the real
//! constructor and the real renderer, because the claim is about the screen.

use ratatui::{backend::TestBackend, Terminal};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use xencode_tui_rs::app::App;
use xencode_tui_rs::ui::draw;

/// Repointing `XCODE_CONFIG_DIR` is process-global, so the cases take a lock.
static LOCK: Mutex<()> = Mutex::new(());

/// Deliberately broken JSON, one trailing comma — the hand edit that used to be
/// answered with a silent reset.
const BROKEN: &str = "{\"default_model\":\"ollama:MYMARKERMODEL\",\"temperature\":0.7,}";

/// The notice the overlay carries, in full.
const NOTICE: &str = "settings not read — this session starts on defaults";

fn temp_config_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-tui-config-notice-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Point the process at a config directory holding `body`, for as long as the
/// returned guard is held.
fn with_config(body: &str) -> (PathBuf, MutexGuard<'static, ()>) {
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = temp_config_dir("dir");
    std::fs::write(dir.join("config.json"), body).unwrap();
    std::env::set_var("XCODE_CONFIG_DIR", &dir);
    (dir, guard)
}

fn leave(dir: &Path, guard: MutexGuard<'static, ()>) {
    std::env::remove_var("XCODE_CONFIG_DIR");
    std::fs::remove_dir_all(dir).ok();
    drop(guard);
}

/// Everything the app paints into a terminal one frame deep.
fn first_frame(app: &mut App, width: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, 30)).unwrap();
    terminal
        .draw(|frame| draw(frame, app))
        .expect("the first frame renders");
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

/// The damage `DF-1` stopped is now said twice on purpose: once in the overlay,
/// short enough to read at a glance, and once in the chat, where the full refusal
/// — which file, where it broke, what to do — outlives the toast.
#[test]
fn a_session_on_an_unreadable_config_says_so_on_the_first_frame() {
    let (dir, guard) = with_config(BROKEN);
    let mut app = App::new();
    let screen = first_frame(&mut app, 100);
    assert!(
        screen.lines().any(|l| l.contains(NOTICE)),
        "no notice on the first frame:\n{screen}"
    );
    // The whole refusal, in the transcript the person can scroll back to.
    let said = app
        .messages
        .iter()
        .find(|m| m.role == "system" && m.content.contains("settings not read"))
        .expect("the chat carries no explanation of the notice");
    assert!(said.content.contains("config.json"), "{}", said.content);
    assert!(
        said.content.contains("is not readable JSON"),
        "{}",
        said.content
    );
    assert!(said.content.contains("trailing comma"), "{}", said.content);
    assert!(
        said.content
            .contains(dir.join("config.json").to_str().unwrap()),
        "the full path is what lets them go fix it: {}",
        said.content
    );
    // It is a notice, not a block: the session opened with its panels.
    assert!(
        screen.contains("Chat"),
        "the screen did not open:\n{screen}"
    );
    // And their bytes are still the bytes they wrote.
    assert_eq!(
        std::fs::read_to_string(dir.join("config.json")).unwrap(),
        BROKEN
    );
    leave(&dir, guard);
}

/// An ordinary terminal is eighty columns wide, and the overlay draws a box
/// inside it. The notice has to survive that, because a clipped sentence is the
/// failure mode this item is about.
#[test]
fn the_notice_arrives_in_full_on_an_eighty_column_terminal() {
    let (dir, guard) = with_config(BROKEN);
    let mut app = App::new();
    let screen = first_frame(&mut app, 80);
    let line = screen
        .lines()
        .find(|l| l.contains("settings not read"))
        .unwrap_or_else(|| panic!("no notice at eighty columns:\n{screen}"));
    assert!(line.contains("this session starts on defaults"), "{line}");
    // The reason is not the overlay's job at this width; it is in the chat, and
    // the chat's copy is on the screen too — wrapped across rows, so checked as
    // the words the person reads rather than as one unbroken string.
    assert!(screen.contains("readable JSON"), "{screen}");
    assert!(screen.contains("trailing comma"), "{screen}");
    assert!(screen.contains("Repair it by hand"), "{screen}");
    leave(&dir, guard);
}

/// Nothing is said when the file is readable — a notice on every launch would
/// train the person to ignore it.
#[test]
fn a_readable_config_opens_silently_and_is_used() {
    const GOOD: &str = "{\"default_model\":\"ollama:MYMARKERMODEL\"}";
    let (dir, guard) = with_config(GOOD);
    let mut app = App::new();
    assert_eq!(
        app.config.default_model, "ollama:MYMARKERMODEL",
        "the file was not read"
    );
    let screen = first_frame(&mut app, 100);
    assert!(!screen.contains("settings not read"), "{screen}");
    assert!(
        !app.messages
            .iter()
            .any(|m| m.content.contains("settings not read")),
        "a good config was announced as broken"
    );
    // A notice path must not become a write path.
    assert_eq!(
        std::fs::read_to_string(dir.join("config.json")).unwrap(),
        GOOD
    );
    leave(&dir, guard);
}
