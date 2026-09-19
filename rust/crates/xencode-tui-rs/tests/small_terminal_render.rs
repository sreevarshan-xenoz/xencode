//! Every panel must render without panicking at any terminal size.
//!
//! Layout maths in the TUI is done on `u16`, so a subtraction that is safe at
//! 80x24 underflows in a terminal a couple of columns wide and takes the whole
//! app down. This sweep is the cheapest way to keep that from coming back:
//! it renders every `FocusArea` across a grid of small sizes and fails with the
//! exact size/panel combinations that panicked.

use ratatui::{backend::TestBackend, Terminal};
use xencode_tui_rs::app::{App, FocusArea, UiMessage};
use xencode_tui_rs::ui::draw;

/// Every overlay reachable from `draw`, plus the base layouts.
const FOCI: &[(&str, FocusArea)] = &[
    ("ChatInput", FocusArea::ChatInput),
    ("FileExplorer", FocusArea::FileExplorer),
    ("CodeEditor", FocusArea::CodeEditor),
    ("ModelSelector", FocusArea::ModelSelector),
    ("Settings", FocusArea::Settings),
    ("CodeReview", FocusArea::CodeReview),
    ("PerformanceDashboard", FocusArea::PerformanceDashboard),
    ("ProviderHealth", FocusArea::ProviderHealth),
    ("ProjectAnalyzer", FocusArea::ProjectAnalyzer),
    ("GitCommit", FocusArea::GitCommit),
    ("FeatureNavigator", FocusArea::FeatureNavigator),
    ("ByteBotPanel", FocusArea::ByteBotPanel),
    ("CollaborationHub", FocusArea::CollaborationHub),
    ("VoiceInterface", FocusArea::VoiceInterface),
    ("TerminalAssistant", FocusArea::TerminalAssistant),
    ("SecurityAuditor", FocusArea::SecurityAuditor),
    ("PerformanceProfiler", FocusArea::PerformanceProfiler),
    ("CustomModels", FocusArea::CustomModels),
    ("LearningMode", FocusArea::LearningMode),
    ("MultiLanguage", FocusArea::MultiLanguage),
    ("ReviewDashboard", FocusArea::ReviewDashboard),
];

/// An app carrying enough content that data-dependent branches actually render
/// (empty-state paths skip most of the layout maths we care about here).
fn populated(focus: FocusArea) -> App<'static> {
    let mut app = App::new();
    app.focus = focus;
    app.messages.push(UiMessage {
        role: "user".into(),
        content: "hello world ".repeat(20),
    });
    app.messages.push(UiMessage {
        role: "assistant".into(),
        content: "```rust\nfn x() {}\n```\nlong ".repeat(10),
    });
    app.bytebot_log.push("✅ did a thing".into());
    app.bytebot_log.push("❌ failed a thing".into());
    app.bytebot_history.push("run tests".into());
    app.attached_files.insert("./src/main.rs".into());
    app.input = "some input text that is fairly long".into();
    // Keep the toast overlay exercised in every panel/size combination too.
    app.toasts.push(xencode_tui_rs::toast::Toast {
        message: "src/x.rs changed on disk — affects main.rs".into(),
        kind: xencode_tui_rs::toast::ToastKind::Warning,
        expires: f64::MAX,
    });
    app
}

/// Dense through the region where layout maths breaks down, then a few sizes
/// spanning up to a normal terminal. A full 40x30 sweep finds nothing extra and
/// costs ~85s; this covers the same failures in a fraction of the time.
const WIDTHS: &[u16] = &[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 61, 80];
const HEIGHTS: &[u16] = &[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 30];

#[test]
fn renders_at_any_terminal_size() {
    let mut failures = Vec::new();

    for (name, focus) in FOCI {
        let app = populated(*focus);
        for &width in WIDTHS {
            for &height in HEIGHTS {
                let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal.draw(|f| draw(f, &app)).unwrap();
                }));
                if rendered.is_err() {
                    failures.push(format!("{name} at {width}x{height}"));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} panel/size combinations panicked: {failures:#?}",
        failures.len()
    );
}

/// The help overlay is topmost and modal; it must survive the same size sweep.
#[test]
fn renders_help_overlay_at_any_terminal_size() {
    let mut failures = Vec::new();
    let mut app = populated(FocusArea::ChatInput);
    app.help_visible = true;
    app.help_scroll = 5; // exercise the clamped scroll path
    for &width in WIDTHS {
        for &height in HEIGHTS {
            let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| draw(f, &app)).unwrap();
            }));
            if rendered.is_err() {
                failures.push(format!("help at {width}x{height}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// E4-03 regression: the settings navigation bound derives from
/// SETTINGS_ROWS, so every row index must be a renderable cursor position.
#[test]
fn settings_panel_renders_with_cursor_on_every_row() {
    for row in 0..xencode_tui_rs::focus::SETTINGS_ROWS.len() {
        let mut app = populated(FocusArea::Settings);
        app.settings_cursor = row;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| draw(f, &app)).unwrap();
    }
}

/// E4-04: the light theme must drive every panel without panicking.
#[test]
fn every_panel_renders_with_light_theme() {
    for (name, focus) in FOCI {
        let mut app = populated(*focus);
        app.theme = xencode_tui_rs::app::ThemeColors::get("light");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
            terminal.draw(|f| draw(f, &app)).unwrap();
        }));
        assert!(result.is_ok(), "{name} failed under light theme");
    }
}
