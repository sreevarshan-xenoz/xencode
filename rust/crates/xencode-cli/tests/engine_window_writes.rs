//! EN-3: a terminal window onto an engine writes nothing the engine owns —
//! no conversation memory, no badge status file, no ByteBot task records.
//! This file holds one test, so the process's settings folder and working
//! folder can be set before anything reads them.

use std::path::Path;

fn files_under(dir: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(files_under(&path));
        } else {
            found.push(path.to_string_lossy().replace('\\', "/"));
        }
    }
    found
}

#[test]
fn a_window_app_writes_nothing_of_the_engines() {
    let settings = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    let before = std::env::current_dir().unwrap();
    // The only test in this process, set before any thread reads it.
    std::env::set_var("XCODE_CONFIG_DIR", settings.path());
    std::env::set_current_dir(project.path()).unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let mut app = xencode_tui_rs::app::App::for_window();
        assert!(app.live.is_none(), "a window keeps no status file");
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        app.dispatch_prompt("/help".into(), tx);
        drop(app);
    });
    drop(runtime);
    std::env::set_current_dir(before).unwrap();

    let written = files_under(settings.path());
    assert!(
        !written
            .iter()
            .any(|f| f.ends_with("conversation_memory.json") || f.contains("/live/")),
        "the window wrote what the engine owns: {written:?}"
    );
    assert!(
        !project.path().join(".xencode").join("bytebot").exists(),
        "the window made a task folder"
    );
}
