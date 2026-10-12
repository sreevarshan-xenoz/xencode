//! Every panel must render without panicking at any terminal size.
//!
//! Layout maths in the TUI is done on `u16`, so a subtraction that is safe at
//! 80x24 underflows in a terminal a couple of columns wide and takes the whole
//! app down. This sweep is the cheapest way to keep that from coming back:
//! it renders every `FocusArea` across a grid of small sizes and fails with the
//! exact size/panel combinations that panicked.

use ratatui::{backend::TestBackend, Terminal};
use xencode_tui_rs::{
    app::{App, FocusArea, UiMessage},
    ui::draw,
};

/// Every overlay reachable from `draw`, plus the base layouts.
const FOCI: &[(&str, FocusArea)] = &[
    ("ChatInput", FocusArea::ChatInput),
    ("FileExplorer", FocusArea::FileExplorer),
    ("CodeEditor", FocusArea::CodeEditor),
    ("ModelSelector", FocusArea::ModelSelector),
    ("Settings", FocusArea::Settings),
    ("CodeReview", FocusArea::CodeReview),
    ("ProviderHealth", FocusArea::ProviderHealth),
    ("ProjectAnalyzer", FocusArea::ProjectAnalyzer),
    ("GitCommit", FocusArea::GitCommit),
    ("ByteBotPanel", FocusArea::ByteBotPanel),
    ("SecurityAuditor", FocusArea::SecurityAuditor),
    ("CustomModels", FocusArea::CustomModels),
    ("MultiLanguage", FocusArea::MultiLanguage),
    ("ReviewDashboard", FocusArea::ReviewDashboard),
    ("TaskManager", FocusArea::TaskManager),
    ("WorktreePanel", FocusArea::WorktreePanel),
    ("AdvisePanel", FocusArea::AdvisePanel),
    ("ImpactPanel", FocusArea::ImpactPanel),
    ("WorkerPanel", FocusArea::WorkerPanel),
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
    // J-04: the multi-language table is a walk's own numbers, so the sweep
    // covers a counted language, a zero-line one and an error-colored reply.
    app.lang_scan_path = "/tmp/workspace".into();
    app.lang_detection_results = vec![("rust".into(), 21, 4200, 78.4), ("toml".into(), 1, 0, 0.0)];
    app.lang_notes = vec!["4 files · 6 lines · 2 skipped by ignore rules".into()];
    app.lang_translate_source = "auto".into();
    app.lang_translate_target = "Tamil".into();
    app.lang_translate_input = "good morning".into();
    app.lang_translate_output = "vāṅkai maṇippu".into();
    app.lang_editing = Some(xencode_tui_rs::focus::LangField::Target);
    // J-05: profiles are config's rows, so the sweep covers a tuned profile,
    // an unset one (drawn as "unset", never as 0) and a status line. MI-7 adds
    // the kind of turn a profile may take on its own, which is the difference
    // between the two rows here.
    app.model_profiles = vec![
        xencode_config_rs::ModelProfile {
            name: "tight".into(),
            model: "ollama:qwen2.5:7b".into(),
            temperature: Some(0.2),
            max_tokens: Some(2048),
            for_task: Some("bugfix".into()),
        },
        xencode_config_rs::ModelProfile {
            name: "server default".into(),
            model: "llamacpp:gemma-3n-e4b".into(),
            temperature: None,
            max_tokens: None,
            for_task: None,
        },
    ];
    app.models_selected = 0;
    app.models_dirty = true;
    app.models_status = "next turn uses ollama:qwen2.5:7b · temperature 0.2".into();
    // QD-2: the impact panel walks target/crate/file rows and each of the
    // three churn states (real count, zero, unknown). Seed one of each so the
    // sweep paints every branch of `draw_impact_panel` without touching
    // `.xencode` or `cargo metadata`.
    app.impact_tree = Some(xencode_context_rs::ImpactTree {
        target: "crates/alpha/src/lib.rs".into(),
        target_crate: Some("alpha".into()),
        groups: vec![
            xencode_context_rs::ImpactGroup {
                crate_name: "alpha".into(),
                crate_hop: 0,
                direct: false,
                kind: None,
                files: vec![xencode_context_rs::ImpactFile {
                    path: "crates/alpha/src/inner.rs".into(),
                    file_hops: 1,
                    via: vec!["use crate::thing".into()],
                    churn: Some(0),
                }],
            },
            xencode_context_rs::ImpactGroup {
                crate_name: "beta".into(),
                crate_hop: 1,
                direct: true,
                kind: Some("normal".into()),
                files: vec![
                    xencode_context_rs::ImpactFile {
                        path: "crates/beta/src/lib.rs".into(),
                        file_hops: 1,
                        via: vec!["use alpha::thing".into()],
                        churn: Some(5),
                    },
                    xencode_context_rs::ImpactFile {
                        path: "crates/beta/src/other.rs".into(),
                        file_hops: 2,
                        via: vec!["use alpha::lib::thing".into()],
                        churn: None,
                    },
                ],
            },
        ],
        max_hops: 2,
        crate_count: 2,
        file_count: 3,
        churn: xencode_context_rs::ChurnSummary {
            own_commits: 12,
            partner_count: 1,
            total_partner_commits: 5,
            known: true,
        },
        basis: "edges are resolved `use`/`mod`/`impl` names, not call sites".into(),
    });
    app.impact_selected = 1;
    app.impact_status.clear();
    // OR-12: the worker panel is a list of rows plus a provenance detail view,
    // so the sweep needs rows with several sources and a section that is empty.
    // The rows are the real projection's output over fixture cards — no fake
    // renderer is involved.
    let cards = vec![xencode_tui_rs::control_room::FleetCard {
        worker: xencode_tui_rs::control_room::WorkerRef {
            agent: "codex".into(),
            task: "survey the crate".into(),
            node: Some("n1".into()),
        },
        status: xencode_tui_rs::control_room::CardStatus::Running,
        messages: 3,
        files_changed: vec!["src/app.rs".into(), "src/ui.rs".into()],
        tools_in_flight: 1,
        needs_approval: xencode_tui_rs::control_room::Trace::Known(true),
        elapsed_ms: xencode_tui_rs::control_room::Trace::Known(1200),
        partly_synthesised: true,
    }];
    app.workers_rows = xencode_tui_rs::worker_panel::sections([
        (
            xencode_tui_rs::worker_panel::PanelSection::Agents,
            xencode_tui_rs::worker_panel::fleet_rows(&cards),
        ),
        (
            xencode_tui_rs::worker_panel::PanelSection::Team,
            xencode_tui_rs::worker_panel::team_rows(&[]),
        ),
        (
            xencode_tui_rs::worker_panel::PanelSection::Tasks,
            xencode_tui_rs::worker_panel::task_rows(None),
        ),
        (
            xencode_tui_rs::worker_panel::PanelSection::Graph,
            xencode_tui_rs::worker_panel::graph_rows(
                &[],
                std::path::Path::new(".xencode/team-runs"),
            ),
        ),
        (
            xencode_tui_rs::worker_panel::PanelSection::Costs,
            xencode_tui_rs::worker_panel::cost_rows(None, &[], &[]),
        ),
        (
            xencode_tui_rs::worker_panel::PanelSection::Logs,
            xencode_tui_rs::worker_panel::log_rows(&[], 30),
        ),
        (
            xencode_tui_rs::worker_panel::PanelSection::Approvals,
            xencode_tui_rs::worker_panel::approval_rows(&[], &[]),
        ),
    ]);
    app.workers_selected = 2;
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

/// The ByteBot panel's model list (BT-5) is drawn over its step rows and must
/// survive the same sweep, with models and without.
#[test]
fn renders_the_bytebot_model_list_at_any_terminal_size() {
    let mut failures = Vec::new();
    for models in [
        vec![],
        vec!["a:1".to_string(), "b:2".to_string(), "c:3".to_string()],
    ] {
        let mut app = populated(FocusArea::ByteBotPanel);
        app.available_models = models.clone();
        app.bytebot_model_picker = true;
        app.bytebot_model_selected = models.len().saturating_sub(1);
        for &width in WIDTHS {
            for &height in HEIGHTS {
                let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal.draw(|f| draw(f, &mut app)).unwrap();
                }));
                if rendered.is_err() {
                    failures.push(format!("{} models at {width}x{height}", models.len()));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The command palette is modal too, with a query line above its list; both
/// an empty query and one that matches nothing must survive the size sweep.
#[test]
fn renders_command_palette_at_any_terminal_size() {
    let mut failures = Vec::new();
    for query in ["", "set", "zzqqxxj"] {
        let mut app = populated(FocusArea::ChatInput);
        app.palette_visible = true;
        app.palette_query = query.to_string();
        app.palette_selected = 3;
        for &width in WIDTHS {
            for &height in HEIGHTS {
                let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal.draw(|f| draw(f, &mut app)).unwrap();
                }));
                if rendered.is_err() {
                    failures.push(format!("palette {query:?} at {width}x{height}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
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
            // A rendered stub, not a gated call: nothing is bound to re-check.
            draft: Default::default(),
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
    assert!(text.contains("y allow"), "{text}");

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
    // The strip lives at the top of a pane; `populated`'s pinned toast is a
    // top-right overlay, so drop it to read the strip's own cells.
    app.toasts.clear();
    // The strip is how the plan shows when the side panel does not (TD-2).
    app.side_panel_open = false;
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

/// J-04: the language table is the walk's numbers and the reply is the model's
/// text. The detection list and the "supported languages" table this panel used
/// to show were both hardcoded.
#[test]
fn multi_language_panel_shows_the_walk_and_the_form() {
    let mut app = populated(FocusArea::MultiLanguage);
    app.toasts.clear();
    let text = render_text(&mut app, 160, 44);
    assert!(text.contains("/tmp/workspace"), "the scanned path: {text}");
    assert!(text.contains("rust"), "{text}");
    assert!(text.contains("78.4%"), "{text}");
    assert!(text.contains("2 skipped by ignore rules"), "{text}");
    // The scanner's own vocabulary, from `Language::ALL`.
    assert!(text.contains("Languages the scanner names (25)"), "{text}");
    assert!(text.contains("typescript"), "{text}");
    assert!(text.contains("▸rust"), "present here: {text}");
    assert!(text.contains("Tamil"), "the edited field: {text}");
    assert!(text.contains("good morning"), "{text}");
    assert!(text.contains("vāṅkai maṇippu"), "the reply: {text}");
    for canned in [
        "src/app.py",
        "components.tsx",
        "Supported Languages",
        "98.7%",
        "Press Enter to scan workspace.",
    ] {
        assert!(
            !text.contains(canned),
            "{canned} is scripted, not real: {text}"
        );
    }
}

/// J-05: the rows are config's profiles and the parameters are the ones a
/// request will carry. The four seeded profiles this panel used to open with
/// are gone, and an unset knob reads "unset" instead of drawing a zero bar.
#[test]
fn custom_models_panel_shows_config_profiles() {
    let mut app = populated(FocusArea::CustomModels);
    app.toasts.clear();
    let text = render_text(&mut app, 160, 44);
    assert!(text.contains("tight"), "{text}");
    assert!(text.contains("ollama:qwen2.5:7b"), "the model id: {text}");
    assert!(text.contains("unsaved"), "the dirty marker: {text}");
    assert!(text.contains("temperature 0.2"), "the status line: {text}");
    assert!(text.contains("0.20"), "{text}");
    assert!(text.contains("2048"), "{text}");
    assert!(
        text.contains("for turn bugfix"),
        "MI-7: which turn may take this profile on its own: {text}"
    );
    for canned in [
        "Code Assistant",
        "Creative Writer",
        "Bug Hunter",
        "Code Reviewer",
        "Top-P",
        "Profile saved!",
        "Press Enter to load profiles",
    ] {
        assert!(!text.contains(canned), "{canned} is scripted: {text}");
    }

    // The second row carries no parameters, which must read as the server
    // deciding rather than as 0.
    app.models_selected = 1;
    let text = render_text(&mut app, 160, 44);
    assert!(text.contains("unset"), "{text}");
    assert!(text.contains("llamacpp:gemma-3n-e4b"), "{text}");
    assert!(
        text.contains("by hand only"),
        "a profile with no turn marked for it says so rather than looking broken: {text}"
    );
    assert!(
        text.contains("model_routing is off"),
        "a marking that cannot fire yet must say so: {text}"
    );
    assert!(
        text.contains("xencode config set model_routing true"),
        "and say where to turn it on, since no key in this panel does: {text}"
    );
}

/// A fresh config has no profiles, and the panel says so rather than inventing
/// rows to fill the space.
#[test]
fn custom_models_panel_renders_the_empty_config_honestly() {
    let mut app = App::for_tests();
    app.focus = FocusArea::CustomModels;
    app.toasts.clear();
    let text = render_text(&mut app, 160, 44);
    assert!(text.contains("No model_profiles in config.json"), "{text}");
    assert!(text.contains("press n"), "how to get one: {text}");
    for canned in ["Code Assistant", "Top-P"] {
        assert!(!text.contains(canned), "{canned} is scripted: {text}");
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
    // Chat: the long transcript makes a thumb smaller than the track, so the
    // double-line glyph must appear.
    let mut app = populated(FocusArea::ChatInput);
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
    // The Files view gives the editor 70% of the width: 49 columns at 70.
    let mut app = populated(FocusArea::CodeEditor);
    // The pinned toast sits over the top right, which is the editor here.
    app.toasts.clear();
    app.open_file_in_editor(path.to_str().unwrap());

    let text = render_text(&mut app, 70, 24);
    assert!(text.contains("1 line1"), "gutter missing:\n{text}");
    assert!(text.contains("12 line12"), "two-digit gutter missing");

    app.config.show_line_numbers = false;
    let text = render_text(&mut app, 70, 24);
    assert!(text.contains("line1 contents"));
    assert!(!text.contains("1 line1"), "gutter leaked when disabled");

    // Toggle on, narrow pane: at 60 columns the editor has 42, under the
    // 45-column gate, so the gutter drops while the code still shows.
    app.config.show_line_numbers = true;
    let text = render_text(&mut app, 60, 24);
    assert!(text.contains("line1 contents"), "editor missing:\n{text}");
    assert!(!text.contains("1 line1"), "gutter shown under 45 columns");

    let _ = std::fs::remove_dir_all(&dir);
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

/// J-01: the security panel shows measurements, so the render must carry the
/// real row text.
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
}

/// OR-12: the worker panel has two views and both must survive a tiny
/// terminal — the sectioned list, and one row's provenance detail, which is
/// the only place in the TUI that prints several wrapped sentences per row.
/// An empty panel is swept too, because that is what the first `/workers` of
/// a session sees.
#[test]
fn worker_panel_renders_list_detail_and_empty() {
    for with_rows in [true, false] {
        for detail in [true, false] {
            let mut app = populated(FocusArea::WorkerPanel);
            if !with_rows {
                app.workers_rows.clear();
            }
            app.workers_detail = detail;
            app.workers_scroll = 999;
            for &width in &[20, 61, 80, 120] {
                for &height in &[8, 16, 24, 30] {
                    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                    terminal.draw(|f| draw(f, &mut app)).unwrap_or_else(|_| {
                        panic!("render {width}x{height} detail={detail} rows={with_rows}")
                    });
                }
            }
        }
    }
}

/// Every `FocusArea` is in the sweep table, so a panel added later cannot skip
/// the small-terminal checks by being left out of `FOCI`.
#[test]
fn the_sweep_covers_every_focus_area() {
    for area in FocusArea::ALL {
        assert!(
            FOCI.iter().any(|(_, foci)| *foci == area),
            "{area:?} is not in FOCI"
        );
    }
    assert_eq!(FOCI.len(), FocusArea::ALL.len());
}
