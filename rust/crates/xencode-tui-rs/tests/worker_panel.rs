//! The worker panel (`OR-12`) as the operator drives it.
//!
//! These are the panel's own done-when, checked through the keyboard and the
//! slash command rather than through the projection: every number on screen
//! traces to a row xencode already holds or a record on disk, and a worker
//! xencode cannot observe is reported as unknown instead of idle. The disk
//! reads are real — a recipe written to a scratch project directory, read back
//! through `load_recipes` — because the panel's whole claim is that it shows
//! what is actually there.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::sync::mpsc;
use xencode_tui_rs::app::{App, FocusArea, SpawnRecord};
use xencode_tui_rs::keymap::handle_key;

/// One test at a time, because the panel reads its recipe and run directories
/// relative to the process working directory and `the_roles...` test moves it
/// into a scratch project. The guard is taken by every test here, so no other
/// test in this binary can read a half-set-up directory.
fn cwd_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn press(app: &mut App, code: KeyCode) {
    let (tx, _rx) = mpsc::unbounded_channel();
    handle_key(app, KeyEvent::new(code, KeyModifiers::NONE), &tx);
}

fn press_ctrl(app: &mut App, ch: char) {
    let (tx, _rx) = mpsc::unbounded_channel();
    handle_key(
        app,
        KeyEvent::new(KeyCode::Char(ch), KeyModifiers::CONTROL),
        &tx,
    );
}

fn submit(app: &mut App, prompt: &str) {
    let (tx, _rx) = mpsc::unbounded_channel();
    app.chat_input.insert_str(prompt);
    app.submit_message(tx);
}

fn rows(app: &App) -> Vec<String> {
    app.workers_rows.iter().map(|r| r.line.clone()).collect()
}

fn row_containing(app: &App, needle: &str) -> String {
    rows(app)
        .into_iter()
        .find(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("no row mentions {needle:?}; rows are: {:#?}", rows(app)))
}

fn spawn(id: u64, task: &str, events: Vec<xencode_agents_rs::protocol::AgentEvent>) -> SpawnRecord {
    SpawnRecord {
        id,
        branch: format!("subagent-{id}"),
        path: PathBuf::from("/tmp/nowhere"),
        task: task.to_string(),
        running: events.is_empty(),
        failed: false,
        steps: Vec::new(),
        events,
    }
}

/// The panel as a keystroke leaves it: read once, then in focus.
fn open(app: &mut App) {
    app.refresh_worker_panel();
    app.focus = FocusArea::WorkerPanel;
}

#[test]
fn ctrl_a_opens_the_panel_over_the_current_state_and_closes_again() {
    let _guard = cwd_guard();
    let mut app = App::for_tests();
    app.spawns
        .push(spawn(1, "split the retrieval tier", Vec::new()));
    assert_eq!(app.focus, FocusArea::ChatInput);
    assert!(
        app.workers_rows.is_empty(),
        "the panel reads only when it is opened"
    );

    press_ctrl(&mut app, 'a');
    assert_eq!(app.focus, FocusArea::WorkerPanel);
    assert!(
        !app.workers_rows.is_empty(),
        "opening the panel read the state it lists"
    );
    assert!(
        row_containing(&app, "split the retrieval tier").contains("unknown"),
        "the one worker with no events must be named on screen"
    );

    press_ctrl(&mut app, 'a');
    assert_eq!(app.focus, FocusArea::ChatInput);
}

#[test]
fn slash_workers_opens_the_panel_and_asks_no_model() {
    let _guard = cwd_guard();
    let mut app = App::for_tests();
    submit(&mut app, "/workers");
    assert_eq!(app.focus, FocusArea::WorkerPanel);
    // The command is a read, so it must not have sent a turn anywhere: the
    // conversation still holds only the usage-free reply the loop writes.
    assert!(
        app.messages
            .iter()
            .all(|m| m.role != "assistant" || !m.content.contains("workers")),
        "the panel is a screen, not a conversation"
    );
}

#[test]
fn every_row_names_the_record_it_was_read_from() {
    let _guard = cwd_guard();
    let mut app = App::for_tests();
    app.spawns.push(spawn(2, "harden the parser", Vec::new()));
    app.refresh_worker_panel();
    assert!(
        app.workers_rows.len() >= 2,
        "a heading and the worker row, at least"
    );
    for row in &app.workers_rows {
        assert!(
            !row.sources.is_empty(),
            "{} prints with no source behind it",
            row.line
        );
        let detail = row.detail();
        assert!(
            detail.contains(&row.line),
            "{}: detail starts with the line",
            row.line
        );
        assert!(
            detail.contains("Read from"),
            "{}: detail names the sources",
            row.line
        );
    }
}

#[test]
fn a_worker_with_no_events_in_its_stream_reads_unknown_and_shows_no_figures() {
    let _guard = cwd_guard();
    let mut app = App::for_tests();
    app.spawns.push(spawn(3, "survey the crate", Vec::new()));
    app.refresh_worker_panel();
    let line = row_containing(&app, "subagent #3 — survey the crate");
    let claim = line
        .strip_prefix("subagent #3 — survey the crate — ")
        .unwrap_or_else(|| panic!("unexpected row shape: {line}"));
    assert!(claim.contains("unknown"), "{claim}");
    assert!(
        !claim.contains("idle") && !claim.contains("working"),
        "{claim}: the stream says nothing, so neither does the panel"
    );
    assert!(
        !claim.chars().any(|c| c.is_ascii_digit()),
        "{claim}: an unobserved worker must not carry invented numbers"
    );
}

#[test]
fn enter_opens_the_trace_and_esc_closes_the_trace_before_the_panel() {
    let _guard = cwd_guard();
    let mut app = App::for_tests();
    open(&mut app);
    press(&mut app, KeyCode::Enter);
    assert!(app.workers_detail);
    app.workers_scroll = 4;
    press(&mut app, KeyCode::Down);
    assert_eq!(app.workers_scroll, 5, "in detail, ↓ scrolls the trace");
    press(&mut app, KeyCode::Esc);
    assert!(!app.workers_detail);
    assert_eq!(app.workers_scroll, 0);
    assert_eq!(app.focus, FocusArea::WorkerPanel);
    press(&mut app, KeyCode::Esc);
    assert_eq!(app.focus, FocusArea::ChatInput);
}

#[test]
fn the_cursor_walks_the_rows_and_stops_at_the_last_one() {
    let _guard = cwd_guard();
    let mut app = App::for_tests();
    open(&mut app);
    let count = app.workers_rows.len();
    assert!(count > 1);
    for _ in 0..count + 5 {
        press(&mut app, KeyCode::Down);
    }
    assert_eq!(app.workers_selected, count - 1, "no walking off the end");
    for _ in 0..count + 5 {
        press(&mut app, KeyCode::Up);
    }
    assert_eq!(app.workers_selected, 0);
}

#[test]
fn r_re_reads_so_a_worker_that_started_after_the_panel_opened_shows_up() {
    let _guard = cwd_guard();
    let mut app = App::for_tests();
    app.spawns.push(spawn(4, "first", Vec::new()));
    open(&mut app);
    let before = rows(&app);
    app.spawns.push(spawn(5, "second", Vec::new()));
    app.workers_selected = 2;
    app.workers_detail = true;
    press(&mut app, KeyCode::Char('r'));
    let after = rows(&app);
    assert!(
        after.iter().any(|line| line.contains("second")),
        "re-read rows: {after:#?}"
    );
    assert_eq!(after.len(), before.len() + 1, "one more worker row");
    assert!(
        after[0].contains("agents (2)"),
        "the heading counts what it heads: {:?}",
        after[0]
    );
    assert_eq!(app.workers_selected, 0, "a re-read starts at the top");
    assert!(!app.workers_detail);
    assert_eq!(
        app.focus,
        FocusArea::WorkerPanel,
        "r re-reads, it does not close"
    );
}

/// Writing a real team recipe and reading it back is the only way to check the
/// panel's sharpest claim: a role xencode does not launch is one it cannot
/// observe, so it must appear as unknown rather than as an idle worker — and
/// must carry no figures, because a zero would read as a measurement.
#[test]
fn the_roles_a_recipe_names_but_xencode_did_not_launch_are_unknown_and_uncounted() {
    let _guard = cwd_guard();
    let root = std::env::temp_dir().join(format!("xencode-workers-panel-{}", std::process::id()));
    let teams = root.join(".xencode").join("teams");
    std::fs::create_dir_all(&teams).unwrap();
    let recipe = teams.join("docs-sweep.toml");
    std::fs::write(
        &recipe,
        r#"name = "docs-sweep"

[[roles]]
name = "survey"
worker = "opencode"
gate = ["lint"]
command = "echo survey"

[[roles]]
name = "integrate"
worker = "claude"
gate = []
needs = ["survey"]
command = "echo integrate"

[capacity]
workers = 2
verification_throughput = 2
"#,
    )
    .unwrap();

    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(&root).unwrap();
    let mut app = App::for_tests();
    app.refresh_worker_panel();
    let all = rows(&app);
    let details: Vec<String> = app.workers_rows.iter().map(|r| r.detail()).collect();
    std::env::set_current_dir(&previous).unwrap();
    let _ = std::fs::remove_dir_all(&root);

    // The role rows themselves: named by the recipe, claimed by nobody.
    let survey = row_containing(&app, "opencode — survey: unknown, not idle");
    let integrate = row_containing(&app, "claude — integrate: unknown, not idle");
    for line in [&survey, &integrate] {
        let tail = line.split_once(": ").expect("the row states its status").1;
        assert!(
            tail.starts_with("unknown, not idle"),
            "{line}: the row must say unknown and say it in words the reader can act on"
        );
        assert!(
            !tail.chars().any(|c| c.is_ascii_digit()),
            "{line}: an unlaunched role carries no figures"
        );
    }
    // The recipe's own words, traced to the file they were read from.
    let path = recipe.display().to_string();
    let survey_detail = details
        .iter()
        .find(|d| d.contains("opencode — survey"))
        .expect("the survey row's detail");
    assert!(survey_detail.contains("1 check(s)"), "{survey_detail}");
    assert!(survey_detail.contains(&path), "{survey_detail}");
    let integrate_detail = details
        .iter()
        .find(|d| d.contains("claude — integrate"))
        .expect("the integrate row's detail");
    assert!(
        integrate_detail.contains("no check gates it"),
        "{integrate_detail}"
    );
    assert!(
        integrate_detail.contains("waits on survey"),
        "{integrate_detail}: the edge is the recipe's own `needs` list"
    );
    assert!(
        survey_detail.contains("xencode holds no event stream for a role it did not launch"),
        "{survey_detail}"
    );
    // Nothing has ever run in this scratch project, so no cost is claimed and
    // no graph is drawn.
    let quote = row_containing(&app, "docs-sweep: no quote — never run on this machine");
    assert!(
        !quote.chars().any(|c| c.is_ascii_digit()),
        "{quote}: a recipe with no recorded run quotes no figures"
    );
    assert!(
        all.iter()
            .any(|line| line.contains("graph: nothing measured")),
        "{all:#?}"
    );
}
