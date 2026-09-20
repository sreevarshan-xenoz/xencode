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
    ("TaskManager", FocusArea::TaskManager),
    ("WorktreePanel", FocusArea::WorktreePanel),
    ("AdvisePanel", FocusArea::AdvisePanel),
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
    app.chat_input
        .insert_str("some input text that is fairly long");
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

/// D2-01: the empty-state sweep above never sees registry rows. This renders
/// the list and detail views over a live (`sleep`) and a finished+output
/// (`echo`) task, across the sizes where layout maths breaks.
#[test]
fn task_panel_renders_populated_registry() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let mut app = populated(FocusArea::TaskManager);
        let id_running = app
            .task_runtime
            .lock()
            .await
            .start("sleeper", "sleep 30")
            .await
            .unwrap();
        let id_done = app
            .task_runtime
            .lock()
            .await
            .start("echo", "echo hi")
            .await
            .unwrap();
        for _ in 0..200 {
            let settled = {
                let mut m = app.task_runtime.lock().await;
                let rec = m.poll(id_done).await.unwrap();
                !matches!(rec.status, xencode_core_rs::TaskStatus::Running)
                    && !rec.output().is_empty()
            };
            if settled {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        for (selected, detail) in [(0usize, false), (1, false), (0, true), (1, true)] {
            app.tasks_selected = selected;
            app.tasks_detail = detail;
            for &width in &[20, 61, 80] {
                for &height in &[8, 16, 24] {
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal.draw(|f| draw(f, &app)).unwrap_or_else(|_| {
                        panic!("render {width}x{height} sel={selected} detail={detail}")
                    });
                }
            }
        }
        app.task_runtime
            .lock()
            .await
            .stop(id_running)
            .await
            .unwrap();
        assert!(id_running < id_done);
    });
}

fn fixture_worktrees() -> Vec<xencode_context_rs::WorktreeInfo> {
    let wt = |path: &str, branch: Option<&str>, detached: bool, main: bool| {
        xencode_context_rs::WorktreeInfo {
            path: path.into(),
            head: "0123456789abcdef0123456789abcdef01234567".into(),
            branch: branch.map(str::to_string),
            detached,
            bare: false,
            locked: None,
            prunable: None,
            is_main: main,
        }
    };
    vec![
        wt("/home/u/proj", Some("main"), false, true),
        wt(
            "/home/u/proj with space/feat",
            Some("feature/x"),
            false,
            false,
        ),
        wt("/home/u/proj/detached", None, true, false),
    ]
}

/// WorktreePanel across list, both prompt stages and confirm mode at tiny
/// and normal sizes, on populated and empty registries. Fixtures only —
/// this never shells out to git.
#[test]
fn worktree_panel_renders_all_prompt_stages() {
    use xencode_tui_rs::focus::WorktreePrompt;
    let stages = [
        (WorktreePrompt::None, "", ""),
        (WorktreePrompt::AddPath, "/home/u/proj-ne", ""),
        (WorktreePrompt::AddBranch, "/home/u/proj-new", "feat"),
        (WorktreePrompt::ConfirmRemove, "", ""),
    ];
    for (prompt, path, branch) in stages {
        for has_rows in [true, false] {
            let mut app = populated(FocusArea::WorktreePanel);
            if has_rows {
                app.worktrees = fixture_worktrees();
                app.worktree_dirty = vec![false, true, false];
            }
            app.worktree_prompt = prompt;
            app.worktree_path_buf = path.into();
            app.worktree_branch_buf = branch.into();
            app.worktree_status = "main worktree is not removable".into();
            for &width in &[20, 61, 80] {
                for &height in &[8, 16, 24] {
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal.draw(|f| draw(f, &app)).unwrap_or_else(|_| {
                        panic!("render {width}x{height} {prompt:?} has_rows={has_rows}")
                    });
                }
            }
        }
    }
}

/// AdvisePanel across list/detail × populated/empty, with every advice kind
/// and an oversized detail scroll (clamp path). Fixtures only — the panel
/// never touches `.xencode` at draw time.
#[test]
fn advise_panel_renders_list_detail_and_empty() {
    use xencode_context_rs::{Advice, AdviceKind};
    let items = vec![
        Advice {
            file: "src/a.rs".into(),
            kind: AdviceKind::BrokenImport,
            message: "⚠ src/a.rs imports `crate::gone::Thing`, which resolves to nothing in this workspace — did a module move or get renamed?".into(),
        },
        Advice {
            file: "src/b.rs".into(),
            kind: AdviceKind::Cycle,
            message: "🔁 import cycle: src/b.rs → src/c.rs → src/b.rs".into(),
        },
        Advice {
            file: "src/big.rs".into(),
            kind: AdviceKind::Hub,
            message: "🧶 src/big.rs depends on 12 files".into(),
        },
        Advice {
            file: "src/dead.rs".into(),
            kind: AdviceKind::Orphan,
            message: "🕸 src/dead.rs has no workspace imports in either direction".into(),
        },
        Advice {
            file: "src/d.rs".into(),
            kind: AdviceKind::AffectedDependent,
            message: "↳ you changed src/d.rs — src/e.rs depends on it".into(),
        },
    ];
    for with_items in [true, false] {
        for detail in [true, false] {
            let mut app = populated(FocusArea::AdvisePanel);
            if with_items {
                app.advise_items = items.clone();
                app.advise_selected = items.len() - 1;
            }
            app.advise_detail = detail;
            app.advise_scroll = 999;
            app.advise_status = "No project index — run /init first.".into();
            for &width in &[20, 61, 80] {
                for &height in &[8, 16, 24] {
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal.draw(|f| draw(f, &app)).unwrap_or_else(|_| {
                        panic!("render {width}x{height} detail={detail} items={with_items}")
                    });
                }
            }
        }
    }
}
