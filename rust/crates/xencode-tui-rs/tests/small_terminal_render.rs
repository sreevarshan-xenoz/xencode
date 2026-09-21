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
    let mut app = App::for_tests();
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
    // I2-04: step rows are real tool calls, so the panel renders every
    // outcome a call can have.
    app.bytebot_steps = vec![
        ("read_file src/app.rs".into(), "done".into()),
        ("edit_file src/app.rs".into(), "running".into()),
        ("run_command cargo test".into(), "denied".into()),
        ("write_file NOTES.md".into(), "failed".into()),
    ];
    app.bytebot_progress = 0.25;
    app.bytebot_history.push("run tests".into());
    app.attached_files.insert("./src/main.rs".into());
    app.chat_input
        .insert_str("some input text that is fairly long");
    // J-03: these rows are a model's reply, so the sweep renders both risk
    // labels, the selection marker and a real outcome in the history.
    app.term_asst_query = "free some disk".into();
    app.term_asst_typing = false;
    app.term_asst_output = "Asking qwen2.5:7b — free some disk".into();
    app.term_asst_suggestions = vec![
        (
            "du -sh *".into(),
            "safe".into(),
            "sizes of everything here".into(),
        ),
        (
            "rm -rf ./target".into(),
            "destructive".into(),
            String::new(),
        ),
    ];
    app.term_asst_selected = 1;
    app.term_asst_history = vec![(
        "df -h".into(),
        "safe".into(),
        "error: the user denied this action".into(),
    )];
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
        let mut app = populated(*focus);
        for &width in WIDTHS {
            for &height in HEIGHTS {
                let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal.draw(|f| draw(f, &mut app)).unwrap();
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
                terminal.draw(|f| draw(f, &mut app)).unwrap();
            }));
            if rendered.is_err() {
                failures.push(format!("help at {width}x{height}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

fn queue_prompt(app: &mut App<'static>, tool: &str, class: xencode_tui_rs::agent_tools::ToolClass) {
    let (responder, _answer) = tokio::sync::oneshot::channel();
    app.approval_queue.push_back((
        xencode_tui_rs::agent_tools::ApprovalRequest {
            tool: tool.into(),
            class,
            summary: format!("{tool} src/lib.rs"),
            preview: "@@ -1,2 +1,3 @@\n-fn old() {}\n+fn hello() {}\n+fn world() {}\n".into(),
        },
        responder,
    ));
}

/// I1-03: the approval prompt is topmost and modal, so it must survive the
/// same size sweep — including a stacked queue and a long diff at 1 row.
#[test]
fn renders_approval_overlay_at_any_terminal_size() {
    use xencode_tui_rs::agent_tools::ToolClass;
    let mut failures = Vec::new();
    let mut app = populated(FocusArea::ChatInput);
    app.help_visible = true; // drawn underneath: the prompt must still paint
    queue_prompt(&mut app, "write_file", ToolClass::Edit);
    queue_prompt(&mut app, "run_command", ToolClass::Shell);
    for &width in WIDTHS {
        for &height in HEIGHTS {
            let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| draw(f, &mut app)).unwrap();
            }));
            if rendered.is_err() {
                failures.push(format!("approval at {width}x{height}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn approval_overlay_shows_the_call_class_diff_and_queue() {
    use xencode_tui_rs::agent_tools::ToolClass;
    let mut app = populated(FocusArea::ChatInput);
    queue_prompt(&mut app, "write_file", ToolClass::Edit);
    let text = render_text(&mut app, 100, 30);
    assert!(text.contains("Allow file change?"), "{text}");
    assert!(text.contains("write_file src/lib.rs"), "{text}");
    assert!(text.contains("fn hello"), "{text}");
    assert!(text.contains("y:allow"), "{text}");

    queue_prompt(&mut app, "run_command", ToolClass::Shell);
    let text = render_text(&mut app, 100, 30);
    assert!(text.contains("+1 more"), "{text}");
}

fn seed_plan(app: &mut App<'static>, items: serde_json::Value) {
    let posted = xencode_tui_rs::agent_tools::apply_plan(&app.agent_plan, &items);
    assert!(posted.starts_with("plan updated:"), "{posted}");
}

/// I2-03: the strip eats rows out of the transcript, so it has to survive the
/// same sweep — including a pinned 12-step list in a pane far too short for
/// it, where the honest answer is to drop the strip, not crush the chat.
#[test]
fn renders_plan_strip_at_any_terminal_size() {
    let mut failures = Vec::new();
    let mut app = populated(FocusArea::ChatInput);
    seed_plan(
        &mut app,
        serde_json::json!([
            {"text": "read the failing test", "status": "done"},
            {"text": "fix the ✂ parser", "status": "in_progress"},
            {"text": "x".repeat(400), "status": "pending"},
            {"text": "run cargo test", "status": "pending"},
        ]),
    );
    for &width in WIDTHS {
        for &height in HEIGHTS {
            let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| draw(f, &mut app)).unwrap();
            }));
            if rendered.is_err() {
                failures.push(format!("plan at {width}x{height}"));
            }
        }
    }
    app.plan_pinned = true;
    let many: Vec<serde_json::Value> = (1..=xencode_tui_rs::agent_tools::PLAN_MAX_ITEMS)
        .map(|i| serde_json::json!({"text": format!("item {i:02}")}))
        .collect();
    seed_plan(&mut app, serde_json::Value::Array(many));
    for &width in WIDTHS {
        for &height in HEIGHTS {
            let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| draw(f, &mut app)).unwrap();
            }));
            if rendered.is_err() {
                failures.push(format!("pinned plan at {width}x{height}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn plan_strip_shows_progress_and_points_at_the_hidden_steps() {
    let mut app = populated(FocusArea::ChatInput);
    // Geometry this test asserts on belongs to the classic preset, not to
    // whatever the developer happens to have in ~/.xencode.
    app.config.layout = "classic".into();
    // The strip lives at the top of a pane; `populated`'s pinned toast is a
    // top-right overlay, so drop it to read the strip's own cells.
    app.toasts.clear();
    let items: Vec<serde_json::Value> = (1..=8)
        .map(|i| {
            serde_json::json!({
                "text": format!("item {i:02}"),
                "status": if i == 1 { "done" } else if i == 2 { "in_progress" } else { "pending" },
            })
        })
        .collect();
    seed_plan(&mut app, serde_json::Value::Array(items));

    let text = render_text(&mut app, 100, 30);
    assert!(text.contains("☰ Plan 1/8"), "{text}");
    assert!(text.contains("✓ item 01"), "{text}");
    assert!(text.contains("▶ item 02"), "{text}");
    assert!(text.contains("· item 03"), "{text}");
    assert!(text.contains("… 2 more — /plan"), "{text}");
    assert!(
        !text.contains("item 07"),
        "the compact strip shows the first few steps, not all of them: {text}"
    );

    app.plan_pinned = true;
    let text = render_text(&mut app, 100, 40);
    assert!(text.contains("item 07"), "{text}");
    assert!(
        !text.contains("more — /plan"),
        "a pinned plan has nothing hidden: {text}"
    );

    // A short pane would rather drop the strip than starve the transcript.
    let text = render_text(&mut app, 100, 10);
    assert!(!text.contains("☰ Plan"), "{text}");
}

/// I2-04: every step row is a call the model made, refusal included. Nothing
/// on this screen comes from a script, so no "Applying changes…" row can
/// appear before a change was made and no "all tests pass" can be claimed.
#[test]
fn bytebot_panel_shows_real_calls_and_their_outcomes() {
    let mut app = populated(FocusArea::ByteBotPanel);
    app.toasts.clear();
    let text = render_text(&mut app, 160, 40);
    assert!(text.contains("Step 1: read_file src/app.rs"), "{text}");
    assert!(
        text.contains("Step 3: run_command cargo test · denied"),
        "{text}"
    );
    assert!(
        text.contains("Step 4: write_file NOTES.md · failed"),
        "{text}"
    );
    assert!(text.contains("25%"), "the bar is calls finished: {text}");
    assert!(text.contains("did a thing"), "{text}");
    for scripted in ["Analyzing workspace", "Applying changes", "All tests pass"] {
        assert!(!text.contains(scripted), "{scripted} is not real: {text}");
    }
}

/// J-03: the list on this screen is whatever the model answered, and the
/// history is whatever the gate let run. The five canned commands this panel
/// used to show — including `docker system prune -af` — cannot come back.
#[test]
fn terminal_assistant_renders_the_reply_and_the_outcome() {
    let mut app = populated(FocusArea::TerminalAssistant);
    app.toasts.clear();
    let text = render_text(&mut app, 160, 40);
    assert!(text.contains("free some disk"), "the question: {text}");
    assert!(text.contains("du -sh *"), "{text}");
    assert!(text.contains("sizes of everything here"), "{text}");
    assert!(text.contains("rm -rf ./target"), "{text}");
    assert!(
        text.contains("[risk] rm -rf ./target"),
        "the dangerous row is marked: {text}"
    );
    assert!(text.contains("[ok]   du -sh *"), "{text}");
    assert!(text.contains("df -h"), "the run history: {text}");
    assert!(
        text.contains("the user denied"),
        "a denial is shown as a denial: {text}"
    );
    for canned in [
        "Press Enter to load suggestions",
        "check disk usage",
        "find all python files",
        "docker system prune",
        "rm -rf node_modules",
    ] {
        assert!(
            !text.contains(canned),
            "{canned} is scripted, not real: {text}"
        );
    }
}

/// E4-03 regression: the settings navigation bound derives from
/// SETTINGS_ITEMS, so every row index must be a renderable cursor position.
#[test]
fn settings_panel_renders_with_cursor_on_every_row() {
    for row in 0..xencode_tui_rs::focus::SETTINGS_ITEMS.len() {
        let mut app = populated(FocusArea::Settings);
        app.settings_cursor = row;
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
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
            terminal.draw(|f| draw(f, &mut app)).unwrap();
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
                    terminal.draw(|f| draw(f, &mut app)).unwrap_or_else(|_| {
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
                    terminal.draw(|f| draw(f, &mut app)).unwrap_or_else(|_| {
                        panic!("render {width}x{height} {prompt:?} has_rows={has_rows}")
                    });
                }
            }
        }
    }
}

fn render_text(app: &mut App<'static>, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|f| draw(f, app)).unwrap();
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect()
}

/// G3-02: the hub renders the real connection state — no fabricated port,
/// no "Protocol: WebSocket (TLS)" over plain ws, no "<15ms" latency. Every
/// claim on screen comes from app state the worker actually wrote.
#[test]
fn collaboration_hub_renders_real_state_not_theatre() {
    let mut app = populated(FocusArea::CollaborationHub);
    app.collab_server_url = "http://127.0.0.1:8765".into();
    app.collab_session_id = "xencode-abc123".into();
    app.collab_session_active = true;
    app.collab_sync_status = "connected".into();
    app.collab_members = vec![
        ("alice".into(), "admin".into(), "connected".into()),
        ("bob".into(), "viewer".into(), "connected".into()),
    ];
    app.collab_last_sync = xencode_models_rs::current_timestamp() - 5.0;

    // Wide enough that the 35%-columns panel shows the whole URL; the
    // truncation at smaller sizes is a layout fact, not a missing feature —
    // the sweep at the bottom covers those sizes for panics.
    let text = render_text(&mut app, 140, 40);
    assert!(text.contains("Server: http://127.0.0.1:8765"), "{text}");
    assert!(text.contains("ws (no TLS)"), "{text}");
    assert!(text.contains("xencode-abc123"), "{text}");
    assert!(text.contains("alice [Admin]"), "{text}");
    assert!(text.contains("bob [Viewer]"), "{text}");
    assert!(text.contains("Connected: 5s"), "{text}");
    // The deleted theatre, gone for good on a plain-ws connection.
    assert!(!text.contains("Protocol: WebSocket"), "{text}");
    assert!(!text.contains("Latency"), "{text}");
    assert!(!text.contains("Port:"), "{text}");
    assert!(!text.contains("carol"), "{text}");

    // wss transport is claimed only when the server URL is https.
    app.collab_server_url = "https://team.example.com".into();
    let text = render_text(&mut app, 140, 40);
    assert!(text.contains("wss (TLS)"), "{text}");

    // Errors surface verbatim instead of being papered over.
    app.collab_error = "connection refused".into();
    let text = render_text(&mut app, 140, 40);
    assert!(text.contains("connection refused"), "{text}");

    // Idle form: the three editable fields and no fake session line.
    let mut idle = populated(FocusArea::CollaborationHub);
    idle.collab_server_url = "http://127.0.0.1:8765".into();
    idle.collab_username = "sree".into();
    idle.collab_editing = true;
    idle.collab_field = xencode_tui_rs::focus::CollabField::Session;
    let text = render_text(&mut idle, 140, 40);
    assert!(text.contains("Server: http://127.0.0.1:8765"), "{text}");
    assert!(text.contains("User: sree"), "{text}");
    assert!(text.contains("Session: (new)"), "{text}");
    assert!(!text.contains("Status:"), "{text}");

    // And the size sweep for the active branch, which the FOCI pass renders
    // only in its idle state.
    for &width in &[20, 40, 61, 80] {
        for &height in &[8, 16, 24] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal
                .draw(|f| draw(f, &mut app))
                .unwrap_or_else(|_| panic!("hub render failed at {width}x{height}"));
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
                    terminal.draw(|f| draw(f, &mut app)).unwrap_or_else(|_| {
                        panic!("render {width}x{height} detail={detail} items={with_items}")
                    });
                }
            }
        }
    }
}

/// H1-06: the `rounded_borders` preference reaches every framed panel at
/// once — body panes, overlays and toasts. Mixed corner sets would mean a
/// construction site was missed by `panel_block`/`panel_border_set`.
/// Messages are cleared because markdown code-fence decoration (`┌─ rust`)
/// is content styling and stays square by design.
#[test]
fn rounded_borders_switch_every_panel_corner() {
    for (name, focus) in FOCI {
        let mut app = populated(*focus);
        app.messages.clear();
        app.config.rounded_borders = true;
        let text = render_text(&mut app, 100, 30);
        assert!(
            text.contains('╭'),
            "{name}: no rounded corner found in:\n{text}"
        );
        assert!(
            !text.contains('┌'),
            "{name}: square corner survived:\n{text}"
        );

        app.config.rounded_borders = false;
        let text = render_text(&mut app, 100, 30);
        assert!(text.contains('┌'), "{name}: square corner missing");
        assert!(
            !text.contains('╭'),
            "{name}: rounded corner leaked:\n{text}"
        );
    }
}

/// H1-07: the scrollbar preference and the editor gutter actually reach the
/// screen, and both respect their gates (no bars in narrow panes, no gutter
/// below 45 editor columns).
#[test]
fn scrollbars_and_gutter_follow_config_and_width() {
    // Chat: 120-wide zen leaves a 1-column bar; the long transcript makes a
    // thumb smaller than the track, so the double-line glyph must appear.
    let mut app = populated(FocusArea::ChatInput);
    app.config.layout = "zen".into();
    let text = render_text(&mut app, 120, 24);
    assert!(
        text.contains('║'),
        "scrollbar missing when enabled:\n{text}"
    );
    app.config.show_scrollbars = false;
    let text = render_text(&mut app, 120, 24);
    assert!(!text.contains('║'), "scrollbar leaked when disabled");

    // Editor gutter: needs show_line_numbers AND ≥45 columns.
    let mut dir = std::env::temp_dir();
    dir.push(format!("xencode_gutter_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sample.rs");
    std::fs::write(
        &path,
        (1..=12)
            .map(|i| format!("line{i} contents"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let mut app = populated(FocusArea::CodeEditor);
    app.config.layout = "zen".into();
    app.open_file_in_editor(path.to_str().unwrap());

    let text = render_text(&mut app, 120, 24);
    assert!(text.contains("1 line1"), "gutter missing:\n{text}");
    assert!(text.contains("12 line12"), "two-digit gutter missing");

    app.config.show_line_numbers = false;
    let text = render_text(&mut app, 120, 24);
    assert!(text.contains("line1 contents"));
    assert!(!text.contains("1 line1"), "gutter leaked when disabled");

    // Wide toggle, narrow pane: classic's 40-column editor must drop the
    // gutter below the 45-column gate.
    app.config.show_line_numbers = true;
    app.config.layout = "classic".into();
    let text = render_text(&mut app, 80, 24);
    assert!(!text.contains("1 line1"), "gutter shown under 45 columns");

    let _ = std::fs::remove_dir_all(&dir);
}

/// H1-08: the header carries brand + layout chip + branch/model + focus
/// badge, and each part drops out at its own width rung. Asserts on the
/// header row only — body panels legitimately say "Chat" etc.
#[test]
fn header_ladder_drops_chips_as_width_shrinks() {
    let header = |app: &mut App<'static>, width: u16| -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, 24)).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .take(width as usize)
            .map(|c| c.symbol())
            .collect()
    };

    let mut app = populated(FocusArea::ChatInput);
    app.config.layout = "zen".into();
    app.config.default_model = "testmodel:1".into();
    app.git_branch = "testbranch".into();

    let text = header(&mut app, 100);
    assert!(text.contains("✦ xencode"), "{text}");
    assert!(text.contains("[zen]"), "{text}");
    assert!(text.contains("⎇ testbranch"), "{text}");
    assert!(text.contains("testmodel:1"), "{text}");
    assert!(text.contains("Chat"), "{text}");

    let text = header(&mut app, 70);
    assert!(
        !text.contains("[zen]"),
        "layout chip survived at 70: {text}"
    );
    assert!(text.contains("⎇ testbranch"), "{text}");

    let text = header(&mut app, 55);
    assert!(
        !text.contains("testbranch"),
        "branch survived at 55: {text}"
    );
    assert!(!text.contains("testmodel"), "model survived at 55: {text}");

    let text = header(&mut app, 30);
    assert!(text.contains("✦"), "brand vanished at 30: {text}");
    assert!(!text.contains("Chat"), "focus badge survived at 30: {text}");
}

/// H1-09: the toast overlay must never sit on top of the input strip. On
/// short screens it steps aside (skips) rather than covering the editor.
#[test]
fn toast_stays_clear_of_the_input_on_short_screens() {
    let mut app = populated(FocusArea::ChatInput);
    let text = render_text(&mut app, 60, 6);
    assert!(
        !text.contains("src/x.rs changed on disk"),
        "toast covered the input at 60x6:\n{text}"
    );
    let text = render_text(&mut app, 60, 10);
    assert!(
        text.contains("src/x.rs changed on disk"),
        "toast vanished at 60x10"
    );
}

/// H1-10: the layout engine's acceptance sweep — every preset (plus an
/// unknown one that must degrade to classic) × every panel × sizes, with
/// all display toggles on, plus zen rendered against each body focus.
#[test]
fn every_layout_preset_renders_at_any_size() {
    const LAYOUTS: &[&str] = &["classic", "chat-first", "zen", "bogus-name"];
    const SIZES: &[(u16, u16)] = &[
        (4, 4),
        (7, 8),
        (20, 4),
        (20, 16),
        (40, 8),
        (40, 24),
        (61, 16),
        (80, 6),
        (80, 24),
        (120, 30),
    ];
    let mut failures = Vec::new();

    for layout in LAYOUTS {
        for (name, focus) in FOCI {
            let mut app = populated(*focus);
            app.config.layout = (*layout).into();
            app.config.rounded_borders = true;
            app.config.show_scrollbars = true;
            app.config.show_line_numbers = true;
            for &(width, height) in SIZES {
                let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal.draw(|f| draw(f, &mut app)).unwrap();
                }));
                if rendered.is_err() {
                    failures.push(format!("{layout}/{name} at {width}x{height}"));
                }
            }
        }
    }

    // Zen's target pane follows the last body focus — pin all three.
    for body_focus in [
        FocusArea::ChatInput,
        FocusArea::FileExplorer,
        FocusArea::CodeEditor,
    ] {
        let mut app = populated(FocusArea::Settings); // overlay must not steal zen's slot
        app.last_body_focus = body_focus;
        app.config.layout = "zen".into();
        for &(width, height) in SIZES {
            let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal.draw(|f| draw(f, &mut app)).unwrap();
            }));
            if rendered.is_err() {
                failures.push(format!("zen/{body_focus:?} at {width}x{height}"));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} layout/panel/size combinations panicked: {failures:#?}",
        failures.len()
    );
}

/// J-01/J-02: these two panels show measurements, so the render must carry the
/// real row text — and a gauge with nothing behind it reads `n/a`, not zero.
#[test]
fn measured_panels_render_real_rows() {
    let mut app = populated(FocusArea::SecurityAuditor);
    app.toasts.clear();
    app.sec_scan_results.push((
        "Critical".into(),
        "hardcoded-secret".into(),
        "src/db.rs:2".into(),
    ));
    app.sec_scan_summary = (1, 0, 0, 0);
    app.sec_scan_log.push("1 findings across 41 files".into());
    let text = render_text(&mut app, 110, 34);
    assert!(text.contains("[Critical] hardcoded-secret"), "{text}");
    assert!(text.contains("Total: 1"), "{text}");
    assert!(text.contains("41 files"), "{text}");

    let mut app = populated(FocusArea::PerformanceProfiler);
    app.toasts.clear();
    app.profiler_rows = vec![(
        "process".into(),
        "resident set".into(),
        "22.6 MB of 15763 MB".into(),
    )];
    app.profiler_gauge_cpu = Some(16.0);
    let text = render_text(&mut app, 110, 34);
    assert!(text.contains("resident set"), "{text}");
    assert!(text.contains("22.6 MB"), "{text}");
    assert!(
        text.contains("n/a"),
        "latency has no measurement yet and must not render as 0: {text}"
    );
}
