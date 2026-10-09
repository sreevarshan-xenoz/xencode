//! EN-3: a window whose engine is gone for good takes over what the engine
//! kept: conversation memory saved to disk, the badge status file and the
//! ByteBot task records. This file holds one test, so the process's
//! settings folder and working folder can be set before anything reads them.

#[test]
fn a_window_left_without_an_engine_keeps_its_own_records() {
    let settings = tempfile::tempdir().unwrap();
    let project = tempfile::tempdir().unwrap();
    let before = std::env::current_dir().unwrap();
    // The only test in this process, set before any thread reads it.
    std::env::set_var("XCODE_CONFIG_DIR", settings.path());
    std::env::set_current_dir(project.path()).unwrap();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let live_files = runtime.block_on(async {
        let mut app = xencode_tui_rs::app::App::for_window();
        assert!(app.live.is_none() && app.bytebot_store.is_none());
        app.take_over_engine_work();
        assert!(
            app.live.is_some(),
            "the window writes the badge status file now"
        );
        assert!(
            app.bytebot_store.is_some(),
            "and keeps ByteBot task records"
        );
        let live = settings.path().join("live");
        let count = std::fs::read_dir(&live).map(|d| d.count()).unwrap_or(0);
        drop(app);
        count
    });
    drop(runtime);
    std::env::set_current_dir(before).unwrap();
    assert_eq!(live_files, 1, "one status file while the window runs");
}
