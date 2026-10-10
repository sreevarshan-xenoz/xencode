//! One ACP session: one project folder and its engine link (M-7a). A session
//! is one more window onto the folder's engine (EN-2), named `acp <pid>`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use xencode_tui_rs::engine::address::Address;
use xencode_tui_rs::engine::link::{self, EngineLink};
use xencode_tui_rs::engine::view::View;

/// Every open session of this `xencode acp` process, by session id.
#[derive(Clone, Default)]
pub struct Sessions {
    map: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<Session>>>>>,
    next: Arc<AtomicU64>,
}

impl Sessions {
    /// Keep `session` and return its new id.
    pub fn add(&self, session: Session) -> String {
        let n = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let id = format!("acp-{}-{n}", std::process::id());
        self.map
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id.clone(), Arc::new(tokio::sync::Mutex::new(session)));
        id
    }

    pub fn get(&self, id: &str) -> Option<Arc<tokio::sync::Mutex<Session>>> {
        self.map
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .cloned()
    }
}

/// One session's state.
pub struct Session {
    /// The project folder, as given by the editor and checked.
    pub project: PathBuf,
    /// The engine connection; `None` while a turn holds it, or after the
    /// engine was lost (the next prompt connects again).
    pub link: Option<EngineLink>,
    /// Whether a turn is running.
    pub busy: bool,
    /// Wakes the running turn when the editor cancels it.
    pub cancel: Arc<Notify>,
    /// The engine's model, as last set or told.
    pub model: String,
    /// A ByteBot question waiting for the person's next message: its id and
    /// the words of the task that asked (M-7d).
    pub question: Option<(u64, String)>,
}

impl Session {
    /// Check `cwd` and reach its engine, starting one if none answers.
    pub async fn open(cwd: &Path) -> Result<Session, String> {
        let project = checked_folder(cwd)?;
        let (link, view) = connect(&project).await?;
        Ok(Session {
            project,
            link: Some(link),
            busy: false,
            cancel: Arc::new(Notify::new()),
            model: view.model.unwrap_or_default(),
            question: None,
        })
    }
}

/// `cwd` made absolute, or why it cannot be a session's folder.
pub fn checked_folder(cwd: &Path) -> Result<PathBuf, String> {
    let meta = std::fs::metadata(cwd)
        .map_err(|_| format!("the folder {} does not exist", cwd.display()))?;
    if !meta.is_dir() {
        return Err(format!("{} is not a folder", cwd.display()));
    }
    std::fs::canonicalize(cwd).map_err(|e| format!("cannot open {}: {e}", cwd.display()))
}

/// Connect to the engine for `project`, starting it detached when none
/// answers, as the terminal app does. Gives back the engine's full view
/// too.
pub async fn connect(project: &Path) -> Result<(EngineLink, View), String> {
    let addr = Address::for_project(project)?;
    let folder = project.to_string_lossy().to_string();
    let start: link::Starter = Arc::new(move || -> std::io::Result<()> {
        let exe = std::env::current_exe()?;
        xencode_live_rs::spawn_detached(&exe, &["engine", "--project", &folder]).map(|_| ())
    });
    let name = format!("acp {}", std::process::id());
    link::connect_or_start_as(&addr, &*start, &name).await
}
