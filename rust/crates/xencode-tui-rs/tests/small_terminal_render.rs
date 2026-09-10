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
    ("Terminal", FocusArea::Terminal),
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
