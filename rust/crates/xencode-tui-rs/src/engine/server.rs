//! `xencode engine` (EN-2): one process per project that holds the agent
//! work and serves it to every window connected over the local socket.
//!
//! The engine owns a headless `App`. A window's messages go through
//! `engine::handle`; every tick the agent loop's reports go through
//! `engine::pump`, and each window is sent what changed in the view it draws.
//! Answers to one window go to that window, except that an approval or a
//! question being answered is told to every window.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use super::address::{lock_path, Address};
use super::proto::{self, ClientMsg, EngineMsg};
use super::transport::{Conn, Listener};
use super::view::Watcher;
use crate::app::App;

/// How long an engine with no window and nothing to do waits before it
/// exits.
pub const IDLE_EXIT: Duration = Duration::from_secs(10);

/// How long something may wait for a person while no window is connected
/// before the engine ends the wait itself (EN-3): thirty minutes.
pub const WAIT_LIMIT: Duration = Duration::from_secs(30 * 60);

/// How often the agent loop's reports are taken in and views sent: the
/// terminal's own frame interval.
const TICK: Duration = Duration::from_millis(33);

/// Messages queued for one window before it counts as stalled: about eight
/// seconds of views at the tick rate, more than a window that is reading
/// ever falls behind. A stalled window (a suspended terminal) is let go; it
/// gets a full view again when it reconnects.
const WINDOW_QUEUE: usize = 256;

/// When the engine ends waits for a person itself (EN-3). The clock runs
/// while no window is connected and something waits; at the limit the
/// waits are ended. After that, until a window connects, a new wait (the
/// model asking again after a denial, say) is ended at once rather than
/// given a whole new limit: nobody is coming to answer it.
#[derive(Default)]
struct UnattendedClock {
    since: Option<Instant>,
    fired: bool,
}

impl UnattendedClock {
    /// Whether the waits should be ended now. `watched`: a window is
    /// connected; `waiting`: something waits for a person.
    fn tick(&mut self, now: Instant, watched: bool, waiting: bool, limit: Duration) -> bool {
        if watched {
            *self = UnattendedClock::default();
            return false;
        }
        if !waiting {
            self.since = None;
            return false;
        }
        if self.fired {
            return true;
        }
        let since = *self.since.get_or_insert(now);
        if now.duration_since(since) >= limit {
            self.fired = true;
            self.since = None;
            return true;
        }
        false
    }
}

/// One connected window.
struct Window {
    name: String,
    out: mpsc::Sender<String>,
    watcher: Watcher,
    greeted: bool,
    /// The reading task, then the writing task.
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl Window {
    /// Queue a message for the window. `false` once its connection is gone
    /// or it has stopped reading.
    fn send(&self, msg: &EngineMsg) -> bool {
        self.out.try_send(proto::encode(msg)).is_ok()
    }
}

impl Window {
    /// Stop reading from the window, and close its connection once what is
    /// already queued for it has been written.
    fn close_after_sending(mut self) {
        let reading = self.tasks.remove(0);
        reading.abort();
        // The writing task ends when the queue is dropped with this window,
        // after writing what the queue holds; it is not stopped here.
        self.tasks.clear();
    }
}

impl Drop for Window {
    /// Letting a window go closes its connection, even when its writer is
    /// stuck on a pipe nobody reads.
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

/// The project folder as the person would write it: resolved, and on
/// Windows without the `\\?\` prefix.
fn display(project: &Path) -> String {
    project
        .to_string_lossy()
        .trim_start_matches(r"\\?\")
        .to_string()
}

/// Run the engine for `project` until it has had no window and nothing to do
/// for `IDLE_EXIT`. Returns at once, after saying so, if another engine for
/// the same project is already running.
pub async fn serve(project: PathBuf, wait_limit: Duration) -> Result<(), String> {
    let project = std::fs::canonicalize(&project)
        .map_err(|e| format!("cannot open the project folder {}: {e}", project.display()))?;
    let shown = display(&project);

    let lock_at = lock_path(&project)?;
    if let Some(folder) = lock_at.parent() {
        std::fs::create_dir_all(folder)
            .map_err(|e| format!("cannot create {}: {e}", folder.display()))?;
    }
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_at)
        .map_err(|e| format!("cannot open {}: {e}", lock_at.display()))?;
    if lock.try_lock().is_err() {
        println!("An engine for {shown} is already running.");
        return Ok(());
    }

    std::env::set_current_dir(&project).map_err(|e| format!("cannot work in {shown}: {e}"))?;
    let addr = Address::for_project(&project)?;
    let listener = Listener::bind(&addr)
        .await
        .map_err(|e| format!("cannot listen on {addr}: {e}"))?;

    let mut app = App::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    app.refresh_models(tx.clone());
    app.maybe_auto_start_llama(tx.clone());

    println!("xencode engine for {shown} listening on {addr}");
    let _ = std::io::stdout().flush();

    let (conn_tx, mut conn_rx) = mpsc::unbounded_channel::<Conn>();
    tokio::spawn(accept_windows(listener, conn_tx));
    let (line_tx, mut line_rx) = mpsc::unbounded_channel::<(u64, Option<String>)>();

    let mut windows: BTreeMap<u64, Window> = BTreeMap::new();
    let mut next_id = 0u64;
    let mut idle_since: Option<Instant> = None;
    let mut clock = UnattendedClock::default();
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            // A window arriving is taken in before a tick that might end a
            // wait the window was about to see.
            biased;
            Some(conn) = conn_rx.recv() => {
                next_id += 1;
                windows.insert(next_id, open_window(next_id, conn, line_tx.clone()));
            }
            Some((id, line)) = line_rx.recv() => {
                match line {
                    Some(line) => on_line(&mut app, &mut windows, id, &line, &tx),
                    None => {
                        windows.remove(&id);
                    }
                }
            }
            _ = tick.tick() => {
                let pumped = super::pump(&mut app, &mut rx, &tx);
                let mut gone = Vec::new();
                for (id, window) in windows.iter_mut().filter(|(_, w)| w.greeted) {
                    let mut alive = pumped.out.iter().all(|msg| window.send(msg));
                    if let Some(view) = window.watcher.changes(&app) {
                        alive &= window.send(&EngineMsg::View { view: Box::new(view) });
                    }
                    if !alive {
                        gone.push(*id);
                    }
                }
                for id in gone {
                    windows.remove(&id);
                }
                if let Some(feed) = app.live.as_mut() {
                    feed.heartbeat();
                }

                // Only a window that said hello can answer anything.
                let watched = windows.values().any(|w| w.greeted);
                let waiting = super::waits_on_a_person(&app);
                if clock.tick(Instant::now(), watched, waiting, wait_limit) {
                    super::unattended(&mut app, &tx);
                }

                if windows.is_empty() && !has_work(&app) {
                    let since = *idle_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= IDLE_EXIT {
                        break;
                    }
                } else {
                    idle_since = None;
                }
            }
        }
    }
    drop(lock);
    Ok(())
}

/// Whether anything is running or waiting on the person.
fn has_work(app: &App) -> bool {
    app.is_generating
        || app.bytebot_running
        || app.question_id.is_some()
        || !app.approval_queue.is_empty()
        || app.bytebot_reviewing().is_some()
        || app
            .bytebot_tasks
            .iter()
            .any(|task| task.this_session && task.state == crate::bytebot_tasks::TaskState::Pending)
}

/// Hand every new connection to the main loop.
async fn accept_windows(mut listener: Listener, conns: mpsc::UnboundedSender<Conn>) {
    loop {
        match listener.accept().await {
            Ok(conn) => {
                if conns.send(conn).is_err() {
                    return;
                }
            }
            Err(e) => {
                eprintln!("xencode engine: a window could not connect: {e}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// Give a connection a reading task and a writing task of its own, so a
/// slow or vanished window never holds up the engine.
fn open_window(id: u64, conn: Conn, lines: mpsc::UnboundedSender<(u64, Option<String>)>) -> Window {
    let (mut reader, mut writer) = conn.split();
    let reading = tokio::spawn(async move {
        loop {
            match reader.recv().await {
                Ok(Some(line)) => {
                    if lines.send((id, Some(line))).is_err() {
                        return;
                    }
                }
                _ => {
                    let _ = lines.send((id, None));
                    return;
                }
            }
        }
    });
    let (out, mut queue) = mpsc::channel::<String>(WINDOW_QUEUE);
    let writing = tokio::spawn(async move {
        while let Some(line) = queue.recv().await {
            if writer.send(&line).await.is_err() {
                return;
            }
        }
    });
    Window {
        name: format!("window {id}"),
        out,
        watcher: Watcher::default(),
        greeted: false,
        tasks: vec![reading, writing],
    }
}

/// One line from window `id`.
fn on_line(
    app: &mut App,
    windows: &mut BTreeMap<u64, Window>,
    id: u64,
    line: &str,
    tx: &mpsc::UnboundedSender<String>,
) {
    let Some(window) = windows.get_mut(&id) else {
        return;
    };
    let msg = match proto::decode_client(line) {
        Ok(msg) => msg,
        Err(message) => {
            window.send(&EngineMsg::Error { message });
            return;
        }
    };
    match msg {
        ClientMsg::Hello { ref client, .. } => {
            let name = client.clone();
            let replies = super::handle(app, msg, tx, &name);
            let greeted = replies
                .iter()
                .any(|reply| matches!(reply, EngineMsg::Hello { .. }));
            for reply in &replies {
                window.send(reply);
            }
            if !greeted {
                // Nothing more from a window speaking another protocol is
                // acted on: it has been told why, and is let go.
                if let Some(window) = windows.remove(&id) {
                    window.close_after_sending();
                }
                return;
            }
            {
                window.name = name;
                window.greeted = true;
                let view = window.watcher.full(app);
                window.send(&EngineMsg::View {
                    view: Box::new(view),
                });
            }
        }
        _ if !window.greeted => {
            window.send(&EngineMsg::Error {
                message: "say hello first: the engine needs to know the protocol version"
                    .to_string(),
            });
        }
        ClientMsg::Goodbye => {
            windows.remove(&id);
        }
        msg => {
            let name = window.name.clone();
            for reply in super::handle(app, msg, tx, &name) {
                let for_everyone = matches!(
                    reply,
                    EngineMsg::ApprovalResolved { .. } | EngineMsg::QuestionAnswered { .. }
                );
                if for_everyone {
                    // The window that answered knows; the others are told
                    // who did. Names are what clients call themselves, so
                    // they are shown, never used to tell windows apart.
                    for (_, other) in windows
                        .iter()
                        .filter(|(other, w)| **other != id && w.greeted)
                    {
                        other.send(&reply);
                    }
                } else if let Some(window) = windows.get(&id) {
                    window.send(&reply);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMIT: Duration = Duration::from_secs(60);

    #[test]
    fn a_wait_ends_only_once_the_limit_has_passed() {
        let start = Instant::now();
        let mut clock = UnattendedClock::default();
        assert!(!clock.tick(start, false, true, LIMIT));
        assert!(!clock.tick(start + Duration::from_secs(59), false, true, LIMIT));
        assert!(clock.tick(start + LIMIT, false, true, LIMIT));
    }

    #[test]
    fn once_nobody_came_a_new_wait_ends_at_once() {
        let start = Instant::now();
        let mut clock = UnattendedClock::default();
        clock.tick(start, false, true, LIMIT);
        assert!(clock.tick(start + LIMIT, false, true, LIMIT));
        // The denial is taken in; the model asks again a little later.
        assert!(!clock.tick(start + LIMIT + Duration::from_secs(1), false, false, LIMIT));
        assert!(clock.tick(start + LIMIT + Duration::from_secs(2), false, true, LIMIT));
    }

    #[test]
    fn a_window_arriving_starts_the_clock_over() {
        let start = Instant::now();
        let mut clock = UnattendedClock::default();
        clock.tick(start, false, true, LIMIT);
        assert!(clock.tick(start + LIMIT, false, true, LIMIT));
        assert!(!clock.tick(start + LIMIT, true, true, LIMIT));
        // The window left again: the full limit applies to the next wait.
        let later = start + LIMIT * 2;
        assert!(!clock.tick(later, false, true, LIMIT));
        assert!(!clock.tick(later + Duration::from_secs(59), false, true, LIMIT));
        assert!(clock.tick(later + LIMIT, false, true, LIMIT));
    }

    #[test]
    fn a_wait_answered_before_the_limit_starts_the_clock_over() {
        let start = Instant::now();
        let mut clock = UnattendedClock::default();
        clock.tick(start, false, true, LIMIT);
        assert!(!clock.tick(start + Duration::from_secs(50), false, false, LIMIT));
        assert!(!clock.tick(start + Duration::from_secs(70), false, true, LIMIT));
        assert!(clock.tick(start + Duration::from_secs(130), false, true, LIMIT));
    }
}
