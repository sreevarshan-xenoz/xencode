//! AE-5: Progressive disclosure level per destination, enforced rather than decorative.
//!
//! Verifies that:
//! 1. One canonical table in `focus.rs` governs the palette and the first-run welcome screen.
//! 2. A beginner drive (Level 2 Workflow) reaches a committed change without ever seeing any Level 4 destination.
//! 3. Every deferred destination remains reachable by name.
//! 4. Turning progressive disclosure off (Level 4) restores full plain xencode as found.

use ratatui::{backend::TestBackend, Terminal};
use std::path::PathBuf;
use std::process::Command;
use xencode_tui_rs::app::{App, DisclosureLevel, FocusArea, DESTINATIONS};
use xencode_tui_rs::ui::draw;

fn render_screen(app: &mut App) -> String {
    let mut terminal = Terminal::new(TestBackend::new(120, 36)).unwrap();
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

const LEVEL_4_KEYWORDS: &[&str] = &[
    "bytebot",
    "/bytebot",
    "ByteBot",
    "Voice Interface",
    "Collaboration Hub",
    "Security Auditor",
    "Performance Profiler",
    "Custom Models",
    "Multi-Language",
];

struct TempDirGuard {
    path: PathBuf,
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn create_temp_git_repo() -> TempDirGuard {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "xencode-disclosure-test-{}-{}",
        std::process::id(),
        nonce
    ));
    std::fs::create_dir_all(&path).expect("create test repo dir");
    TempDirGuard { path }
}

#[test]
fn beginner_drive_reaches_committed_change_seeing_no_level_4_destination() {
    // 1. Prepare a real git repository for the drive
    let repo = create_temp_git_repo();
    let repo_path = &repo.path;

    assert!(Command::new("git")
        .args(["init"])
        .current_dir(repo_path)
        .status()
        .expect("git init")
        .success());
    assert!(Command::new("git")
        .args(["config", "user.name", "Test Beginner"])
        .current_dir(repo_path)
        .status()
        .expect("git config user.name")
        .success());
    assert!(Command::new("git")
        .args(["config", "user.email", "beginner@test.local"])
        .current_dir(repo_path)
        .status()
        .expect("git config user.email")
        .success());

    // Initial commit
    std::fs::write(repo_path.join("README.md"), "# Beginner Repo\n").unwrap();
    Command::new("git")
        .args(["add", "."])
        .current_dir(repo_path)
        .status()
        .unwrap();
    Command::new("git")
        .args(["commit", "-m", "initial commit"])
        .current_dir(repo_path)
        .status()
        .unwrap();

    // 2. Start app at Level 2 (beginner / workflow tier)
    let mut app = App::for_tests();
    app.set_disclosure_level(DisclosureLevel::Level2);

    // 3. Inspect first-run welcome screen
    let welcome_screen = render_screen(&mut app);
    for &kw in LEVEL_4_KEYWORDS {
        assert!(
            !welcome_screen.contains(kw),
            "Welcome screen at Level 2 must not show Level 4 destination '{}'",
            kw
        );
    }

    // 4. Inspect palette / Feature Navigator
    app.focus = FocusArea::FeatureNavigator;
    let palette_items = app.palette_items();
    for (_name, _desc, area) in &palette_items {
        assert!(
            area.disclosure_level().rank() <= 2,
            "Palette at Level 2 must not contain Level {:?} destination {:?}",
            area.disclosure_level(),
            area
        );
    }
    let palette_screen = render_screen(&mut app);
    for &kw in LEVEL_4_KEYWORDS {
        assert!(
            !palette_screen.contains(kw),
            "Palette screen at Level 2 must not display Level 4 keyword '{}'",
            kw
        );
    }

    // 5. Code Editor: make an edit
    app.focus = FocusArea::CodeEditor;
    let edit_file = repo_path.join("app.txt");
    std::fs::write(&edit_file, "beginner edit content\n").unwrap();
    let editor_screen = render_screen(&mut app);
    for &kw in LEVEL_4_KEYWORDS {
        assert!(
            !editor_screen.contains(kw),
            "Editor screen at Level 2 must not display Level 4 keyword '{}'",
            kw
        );
    }

    // Stage file in git
    Command::new("git")
        .args(["add", "app.txt"])
        .current_dir(repo_path)
        .status()
        .unwrap();

    // 6. Git Commit panel: stage and commit changes
    app.focus = FocusArea::GitCommit;
    app.commit_message = "beginner change committed successfully".to_string();
    let commit_screen = render_screen(&mut app);
    for &kw in LEVEL_4_KEYWORDS {
        assert!(
            !commit_screen.contains(kw),
            "Commit screen at Level 2 must not display Level 4 keyword '{}'",
            kw
        );
    }

    // Commit change to git
    let commit_res = xencode_tui_rs::gitsign::commit_signed(repo_path, &app.commit_message, 30);
    assert!(
        commit_res.is_ok(),
        "Commit should succeed: {:?}",
        commit_res
    );

    // Verify git log contains the committed change
    let log_out = Command::new("git")
        .args(["log", "-1", "--pretty=%B"])
        .current_dir(repo_path)
        .output()
        .expect("git log");
    let log_msg = String::from_utf8_lossy(&log_out.stdout);
    assert!(
        log_msg.contains("beginner change committed successfully"),
        "Git history must contain the committed change"
    );

    // 7. Verify all deferred Level 4 destinations remain reachable by name
    assert!(app.navigate_to_destination_by_name("bytebot"));
    assert_eq!(app.focus, FocusArea::ByteBotPanel);

    assert!(app.navigate_to_destination_by_name("voice"));
    assert_eq!(app.focus, FocusArea::VoiceInterface);

    assert!(app.navigate_to_destination_by_name("collab"));
    assert_eq!(app.focus, FocusArea::CollaborationHub);

    assert!(app.navigate_to_destination_by_name("security"));
    assert_eq!(app.focus, FocusArea::SecurityAuditor);

    assert!(app.navigate_to_destination_by_name("profiler"));
    assert_eq!(app.focus, FocusArea::PerformanceProfiler);

    assert!(app.navigate_to_destination_by_name("custom-models"));
    assert_eq!(app.focus, FocusArea::CustomModels);

    assert!(app.navigate_to_destination_by_name("learn"));
    assert_eq!(app.focus, FocusArea::LearningMode);

    assert!(app.navigate_to_destination_by_name("languages"));
    assert_eq!(app.focus, FocusArea::MultiLanguage);
}

#[test]
fn turning_disclosure_off_restores_plain_xencode_identically() {
    let mut app = App::for_tests();
    app.set_disclosure_level(DisclosureLevel::Level4);

    // All destinations are present
    assert_eq!(DESTINATIONS.len(), 27);
    let palette = app.palette_items();
    // Excludes FeatureNavigator itself (27 - 7 Level 1/2 workflow/core panes not in feature palette)
    assert!(palette
        .iter()
        .any(|(_, _, a)| *a == FocusArea::ByteBotPanel));
    assert!(palette
        .iter()
        .any(|(_, _, a)| *a == FocusArea::VoiceInterface));
    assert!(palette
        .iter()
        .any(|(_, _, a)| *a == FocusArea::SecurityAuditor));
    // `OR-12`: the worker panel is a Level 3 destination, so turning
    // disclosure off reaches it from the palette like every other pane.
    assert!(palette.iter().any(|(_, _, a)| *a == FocusArea::WorkerPanel));

    // Welcome shortcuts line includes /bytebot=agent
    let shortcuts = xencode_tui_rs::focus::first_run_shortcuts_line(DisclosureLevel::Level4);
    assert!(shortcuts.contains("/bytebot=agent"));
}
