//! OM-3: orchestrator mode drawn and driven for real. Each test runs in a
//! scratch git repository of its own (the terminal agents' worktrees go
//! beside it), opens real shells in pseudo-terminals in process, and reads
//! the screen back.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{backend::TestBackend, Terminal};
use xencode_tui_rs::{app::App, ui::draw};

/// The project is the current directory, which is the whole process's, so
/// the tests take turns.
static CWD: Mutex<()> = Mutex::new(());

struct Scratch {
    _outer: tempfile::TempDir,
    before: std::path::PathBuf,
    _guard: MutexGuard<'static, ()>,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.before);
    }
}

/// A repository with one commit, made the current directory.
fn scratch() -> Scratch {
    let guard = CWD.lock().unwrap_or_else(|e| e.into_inner());
    let outer = tempfile::tempdir().unwrap();
    let root = outer.path().join("proj");
    std::fs::create_dir(&root).unwrap();
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "-q", "-b", "main"]);
    git(&["config", "user.email", "t@example.invalid"]);
    git(&["config", "user.name", "t"]);
    std::fs::write(root.join("a.txt"), "one\n").unwrap();
    git(&["add", "."]);
    git(&["commit", "-q", "-m", "first"]);
    let before = std::env::current_dir().unwrap();
    std::env::set_current_dir(&root).unwrap();
    Scratch {
        _outer: outer,
        before,
        _guard: guard,
    }
}

fn key(app: &mut App<'static>, code: KeyCode, mods: KeyModifiers) {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    xencode_tui_rs::keymap::handle_key(app, KeyEvent::new(code, mods), &tx);
}

fn ctrl_a(app: &mut App<'static>, then: KeyCode) {
    key(app, KeyCode::Char('a'), KeyModifiers::CONTROL);
    key(app, then, KeyModifiers::NONE);
}

fn screen(app: &mut App<'static>, w: u16, h: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal.draw(|f| draw(f, app)).unwrap();
    let buf = terminal.backend().buffer().clone();
    (0..h)
        .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// One turn of the frame loop, in process: the engine's work, the tiles'
/// needs sent, a frame drawn.
fn frame(app: &mut App<'static>) -> String {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    xencode_tui_rs::engine::pump(app, &mut rx, &tx);
    let text = screen(app, 160, 48);
    app.orch_flush(&tx);
    text
}

fn frames_until(app: &mut App<'static>, what: &str, done: impl Fn(&str) -> bool) -> String {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let text = frame(app);
        if done(&text) {
            return text;
        }
        assert!(
            Instant::now() < deadline,
            "gave up waiting for {what}:\n{text}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn enter_mode(app: &mut App<'static>) {
    key(app, KeyCode::Char(' '), KeyModifiers::CONTROL);
}

/// What is typed is not what the shell prints.
fn typed() -> &'static str {
    if cfg!(windows) {
        "echo XEN%COMSPEC:~0,0%CODE_OM3"
    } else {
        "echo XEN''CODE_OM3"
    }
}

#[tokio::test]
async fn entering_the_mode_opens_a_picker_of_every_agent() {
    let _here = scratch();
    let mut app = App::for_tests();
    app.toasts.clear();
    enter_mode(&mut app);
    let text = screen(&mut app, 160, 48);
    assert!(text.contains("Open agents"), "{text}");
    assert!(text.contains("shell") && text.contains("claude"), "{text}");
    assert!(
        text.contains("installed") && text.contains("your sign-in"),
        "{text}"
    );
    key(&mut app, KeyCode::Esc, KeyModifiers::NONE);
    let text = screen(&mut app, 160, 48);
    assert!(!text.contains("Open agents"), "{text}");
    assert!(
        text.contains("Agents") && text.contains("xencode"),
        "the rail: {text}"
    );
}

#[tokio::test]
async fn a_shell_tile_takes_the_keys_and_draws_its_screen() {
    let _here = scratch();
    let mut app = App::for_tests();
    app.toasts.clear();
    enter_mode(&mut app);
    // The shell is the picker's first row.
    key(&mut app, KeyCode::Char(' '), KeyModifiers::NONE);
    key(&mut app, KeyCode::Enter, KeyModifiers::NONE);
    frames_until(&mut app, "the tile", |_| true);
    assert_eq!(app.term_tiles.len(), 1, "one shell tile");
    // Ctrl+A 1 focuses it; what is typed goes to the shell.
    ctrl_a(&mut app, KeyCode::Char('1'));
    for c in typed().chars() {
        key(&mut app, KeyCode::Char(c), KeyModifiers::NONE);
    }
    key(&mut app, KeyCode::Enter, KeyModifiers::NONE);
    let text = frames_until(&mut app, "the shell's output", |t| {
        t.contains("XENCODE_OM3")
    });
    assert!(text.contains("1 ") && text.contains("shell"), "{text}");
    // Ctrl+A z zooms the tile, Ctrl+A 0 goes back to the conversation.
    ctrl_a(&mut app, KeyCode::Char('z'));
    assert!(app.orch.zoom);
    ctrl_a(&mut app, KeyCode::Char('0'));
    assert_eq!(app.orch.focus, 0);
    // In the conversation's tile a letter is the conversation's.
    key(&mut app, KeyCode::Char('q'), KeyModifiers::NONE);
    assert_eq!(app.chat_input.lines().join(""), "q");
    // Ctrl+A x twice stops the shell; its worktree stays.
    ctrl_a(&mut app, KeyCode::Char('1'));
    ctrl_a(&mut app, KeyCode::Char('x'));
    ctrl_a(&mut app, KeyCode::Char('x'));
    let deadline = Instant::now() + Duration::from_secs(20);
    while !app_tile_ended(&app) {
        assert!(Instant::now() < deadline, "the shell kept running");
        frame(&mut app);
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn app_tile_ended(app: &App<'static>) -> bool {
    app.term_tiles
        .first()
        .is_some_and(|t| matches!(t.state.as_str(), "done" | "failed"))
}

#[tokio::test]
async fn ctrl_space_leaves_the_mode_and_the_tiles_keep_running() {
    let _here = scratch();
    let mut app = App::for_tests();
    app.toasts.clear();
    enter_mode(&mut app);
    key(&mut app, KeyCode::Char(' '), KeyModifiers::NONE);
    key(&mut app, KeyCode::Enter, KeyModifiers::NONE);
    frames_until(&mut app, "the tile", |_| true);
    enter_mode(&mut app);
    let text = screen(&mut app, 160, 48);
    assert!(!text.contains("+ new agent"), "the rail is gone: {text}");
    // The conversation's side panel names the agent still running.
    assert!(text.contains("shell t1"), "{text}");
    assert_eq!(app.term_tiles.len(), 1);
    assert!(app.term_tiles[0].exit.is_none(), "the shell still runs");
}

#[tokio::test]
async fn the_mode_draws_at_any_size() {
    let _here = scratch();
    let mut app = App::for_tests();
    app.toasts.clear();
    enter_mode(&mut app);
    key(&mut app, KeyCode::Char(' '), KeyModifiers::NONE);
    key(&mut app, KeyCode::Enter, KeyModifiers::NONE);
    frames_until(&mut app, "the tile", |_| true);
    for (w, h) in [
        (200u16, 60u16),
        (160, 48),
        (100, 30),
        (99, 30),
        (80, 24),
        (40, 12),
        (10, 4),
        (4, 3),
    ] {
        let drawn = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            screen(&mut app, w, h);
        }));
        assert!(drawn.is_ok(), "{w}x{h}");
    }
}

/// OM-4: a real program in a tile that rings the bell is marked as needing
/// the person: its rail row moves to Needs you, it is in the Inbox, and the
/// window title counts it; typing into it answers it.
#[tokio::test]
async fn a_tile_that_rings_needs_you_until_it_is_typed_into() {
    let _here = scratch();
    let mut app = App::for_tests();
    app.toasts.clear();
    enter_mode(&mut app);
    key(&mut app, KeyCode::Char(' '), KeyModifiers::NONE);
    key(&mut app, KeyCode::Enter, KeyModifiers::NONE);
    frames_until(&mut app, "the tile", |_| true);
    ctrl_a(&mut app, KeyCode::Char('1'));
    let ring = if cfg!(windows) {
        "powershell -NoProfile -Command \"[char]7\""
    } else {
        "printf '\\a'"
    };
    for c in ring.chars() {
        key(&mut app, KeyCode::Char(c), KeyModifiers::NONE);
    }
    key(&mut app, KeyCode::Enter, KeyModifiers::NONE);
    let text = frames_until(&mut app, "the bell", |t| t.contains("Needs you"));
    assert!(text.contains("Inbox"), "{text}");
    assert_eq!(
        xencode_tui_rs::shell::attention_title(&app),
        "xencode · needs you"
    );
    // Typing into the tile answers whatever it was asking.
    key(&mut app, KeyCode::Char(' '), KeyModifiers::NONE);
    frames_until(&mut app, "the call to clear", |t| !t.contains("Needs you"));
    assert_ne!(
        xencode_tui_rs::shell::attention_title(&app),
        "xencode · needs you"
    );
}

fn type_line(app: &mut App<'static>, text: &str) {
    for c in text.chars() {
        key(app, KeyCode::Char(c), KeyModifiers::NONE);
    }
    key(app, KeyCode::Enter, KeyModifiers::NONE);
}

fn open_shell(app: &mut App<'static>) {
    // The picker opens on entering the mode, or with Ctrl+A n; the shell is
    // its first row and stays chosen once chosen.
    if app.orch.picker.is_none() {
        ctrl_a(app, KeyCode::Char('n'));
    }
    if !app.config.orchestrator_agents.iter().any(|a| a == "shell") {
        key(app, KeyCode::Char(' '), KeyModifiers::NONE);
    }
    key(app, KeyCode::Enter, KeyModifiers::NONE);
    frames_until(app, "the tile", |_| true);
}

/// OM-5: a tile's work lands on the base branch from the rail, through the
/// checked merge: the first Ctrl+A m says what will happen, the second stops
/// the agent and merges; with no checks set, the person is asked first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_tiles_work_is_merged_from_the_rail_through_the_checked_merge() {
    let here = scratch();
    let root = std::env::current_dir().unwrap();
    let mut app = App::for_tests();
    app.toasts.clear();
    enter_mode(&mut app);
    open_shell(&mut app);
    ctrl_a(&mut app, KeyCode::Char('1'));
    let worktree = std::path::PathBuf::from(&app.term_tiles[0].worktree);
    type_line(&mut app, "echo made by the tile> tile.txt");
    let deadline = Instant::now() + Duration::from_secs(20);
    while !worktree.join("tile.txt").exists() {
        assert!(Instant::now() < deadline, "the tile never wrote its file");
        frame(&mut app);
        std::thread::sleep(Duration::from_millis(50));
    }
    ctrl_a(&mut app, KeyCode::Char('m'));
    assert!(
        app.toasts
            .iter()
            .any(|t| t.message.contains("ctrl+a m again")),
        "the first press says what will happen"
    );
    ctrl_a(&mut app, KeyCode::Char('m'));
    // No checks are set in this repository, so the merge asks the person.
    let deadline = Instant::now() + Duration::from_secs(60);
    while app.pending_approval().is_none() {
        assert!(Instant::now() < deadline, "the merge never asked");
        frame(&mut app);
        std::thread::sleep(Duration::from_millis(50));
    }
    frame(&mut app);
    std::thread::sleep(Duration::from_millis(600));
    frame(&mut app);
    key(&mut app, KeyCode::Char('y'), KeyModifiers::NONE);
    let landed = |root: &std::path::Path| {
        std::process::Command::new("git")
            .args(["show", "main:tile.txt"])
            .current_dir(root)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };
    let deadline = Instant::now() + Duration::from_secs(120);
    while !landed(&root) {
        assert!(
            Instant::now() < deadline,
            "the tile's work never landed on main"
        );
        frame(&mut app);
        std::thread::sleep(Duration::from_millis(100));
    }
    drop(app);
    drop(here);
}

/// OM-5: Ctrl+A a sends one line to every working or idle agent.
#[tokio::test]
async fn ask_all_sends_one_line_to_every_agent() {
    let _here = scratch();
    let mut app = App::for_tests();
    app.toasts.clear();
    enter_mode(&mut app);
    open_shell(&mut app);
    open_shell(&mut app);
    assert_eq!(app.term_tiles.len(), 2);
    ctrl_a(&mut app, KeyCode::Char('a'));
    let line = if cfg!(windows) {
        "echo XEN%COMSPEC:~0,0%CODE_ALL"
    } else {
        "echo XEN''CODE_ALL"
    };
    type_line(&mut app, line);
    let ids: Vec<String> = app.term_tiles.iter().map(|t| t.id.clone()).collect();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        frame(&mut app);
        let both = ids.iter().all(|id| {
            app.term_screens
                .get(id)
                .is_some_and(|s| s.text().contains("XENCODE_ALL"))
        });
        if both {
            break;
        }
        assert!(Instant::now() < deadline, "not every agent got the line");
        std::thread::sleep(Duration::from_millis(50));
    }
}
