//! A ByteBot task end to end against a real local model (DK-1, BT-1 … BT-3).
//!
//! Ignored by default: it needs a running llama.cpp server. Run it with
//!
//! ```text
//! XENCODE_LIVE_LLAMA_URL=http://127.0.0.1:8090 XENCODE_LIVE_MODEL=llamacpp:qwen3-4b \
//!   cargo test -p xencode-tui-rs --test live_bytebot_real_model -- --ignored --nocapture
//! ```
//!
//! Everything is real: the model, the agent loop, the tools writing to disk,
//! the task record and the live status file a floating badge can watch. Only
//! the keyboard is replaced, by the same calls the panel's keys make, and the
//! token routing is the same few branches `run_app` uses for ByteBot.

use std::time::{Duration, Instant};

use xencode_tui_rs::app::App;
use xencode_tui_rs::bytebot_tasks::{TaskState, TaskStore};

/// Feed one token the way `run_app`'s loop does for a ByteBot run.
fn route(app: &mut App, token: &str, tx: &tokio::sync::mpsc::UnboundedSender<String>) {
    if let Some(body) = token.strip_prefix("[BYTEBOT]") {
        app.bytebot_event(body);
    } else if token == "[BYTEBOT_DONE]" {
        app.bytebot_run_finished(tx.clone());
    } else if token == "[STOPPED]" {
        app.live_stopped();
    }
}

fn state_of(app: &App) -> TaskState {
    app.bytebot_tasks.last().map(|t| t.state).unwrap()
}

/// Drain tokens and questions until `done` says so or the time runs out.
async fn run_until(
    app: &mut App<'_>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>,
    tx: &tokio::sync::mpsc::UnboundedSender<String>,
    limit: Duration,
    done: impl Fn(&App) -> bool,
) -> bool {
    let started = Instant::now();
    while started.elapsed() < limit {
        while let Ok(token) = rx.try_recv() {
            route(app, &token, tx);
        }
        let mut asked = Vec::new();
        if let Some(ask) = app.ask_rx.as_mut() {
            while let Ok(question) = ask.try_recv() {
                asked.push(question);
            }
        }
        for (question, reply) in asked {
            app.bytebot_needs_help(question, reply);
        }
        if done(app) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

#[tokio::test]
#[ignore = "needs a running llama.cpp server; see the file's documentation"]
async fn a_bytebot_task_asks_writes_and_waits_for_review_with_a_real_model() {
    let url = std::env::var("XENCODE_LIVE_LLAMA_URL").expect("XENCODE_LIVE_LLAMA_URL");
    let model = std::env::var("XENCODE_LIVE_MODEL").unwrap_or("llamacpp:qwen3-4b".into());
    let workspace = tempfile::tempdir().unwrap();
    std::env::set_current_dir(workspace.path()).unwrap();

    let mut app = App::for_tests();
    app.config.default_model = model;
    app.config.llama_cpp_url = url;
    app.config.agent_approval = "edit-allow".into();
    app.bytebot_store = Some(TaskStore::new(&workspace.path().join(".xencode")));
    if let Ok(dir) = xencode_live_rs::live_dir() {
        app.live = Some(xencode_tui_rs::live_status::LiveFeed::new(
            dir,
            "live-check".into(),
            workspace.path(),
        ));
    }
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    app.bytebot_command = "First call the ask_user tool to ask me which file name to use. \
                           Do not guess. After I answer, create that file containing the \
                           single line: hello from bytebot"
        .into();
    app.run_bytebot(tx.clone());
    println!("LIVE: task started, state {:?}", state_of(&app));

    let asked = run_until(&mut app, &mut rx, &tx, Duration::from_secs(300), |a| {
        state_of(a) == TaskState::NeedsHelp || state_of(a) != TaskState::Running
    })
    .await;
    println!(
        "LIVE: phase 1 ended (reached={asked}) in state {:?}, question {:?}",
        state_of(&app),
        app.bytebot_tasks.last().unwrap().question
    );
    if state_of(&app) == TaskState::NeedsHelp {
        // Leave time to see the badge turn amber.
        tokio::time::sleep(Duration::from_secs(8)).await;
        app.bytebot_command = "notes.md".into();
        app.bytebot_answer();
        println!("LIVE: answered notes.md");
    }

    let finished = run_until(&mut app, &mut rx, &tx, Duration::from_secs(300), |a| {
        !matches!(state_of(a), TaskState::Running | TaskState::NeedsHelp)
    })
    .await;
    let task = app.bytebot_tasks.last().unwrap().clone();
    println!(
        "LIVE: run ended (reached={finished}) in state {:?}; changed {:?}; note {:?}; steps {:?}",
        task.state, task.changed_files, task.note, task.steps
    );
    let written = std::fs::read_to_string(workspace.path().join("notes.md")).ok();
    println!("LIVE: notes.md on disk: {written:?}");

    if task.state == TaskState::NeedsReview {
        // Leave time to see the badge ask for the review.
        tokio::time::sleep(Duration::from_secs(8)).await;
        app.bytebot_accept(tx.clone());
        println!("LIVE: accepted, state {:?}", state_of(&app));
        tokio::time::sleep(Duration::from_secs(8)).await;
    }
    let records = TaskStore::new(&workspace.path().join(".xencode")).load_all();
    println!(
        "LIVE: record on disk says {:?}",
        records.last().map(|t| t.state)
    );
    std::env::set_current_dir(std::env::temp_dir()).unwrap();
}
