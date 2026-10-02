//! ImpactPanel behavior (`QD-2`).
//!
//! The panel is a walk over a fixed tree: it does not re-query on its own, so
//! the assertions below are about cursor and stack state, not about file
//! contents. A real workspace query is checked once at the bottom, in
//! `the_slash_command_opens_the_panel_over_a_real_workspace_file`, to prove the
//! `/impact` command and `change_impact` are actually wired together — the
//! keyboard tests use a hand-built tree so they can pin each row exactly.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tokio::sync::mpsc;
use xencode_context_rs::{ChurnSummary, ImpactFile, ImpactGroup, ImpactRow, ImpactTree};
use xencode_tui_rs::app::{App, FocusArea};
use xencode_tui_rs::keymap::handle_key;

fn press(app: &mut App, code: KeyCode) {
    let (tx, _rx) = mpsc::unbounded_channel();
    handle_key(app, KeyEvent::new(code, KeyModifiers::NONE), &tx);
}

fn submit(app: &mut App, prompt: &str) {
    let (tx, _rx) = mpsc::unbounded_channel();
    app.chat_input.insert_str(prompt);
    app.submit_message(tx);
}

/// A three-row tree: target · one crate header · two file rows. Churn varies
/// so a test can check `Some(0)` vs `Some(n)` vs `None` reaches the render
/// path — the panel's contract is "no co-change" and "unknown" are different
/// sentences, not both "0".
fn tree() -> ImpactTree {
    ImpactTree {
        target: "crates/alpha/src/lib.rs".into(),
        target_crate: Some("alpha".into()),
        groups: vec![ImpactGroup {
            crate_name: "beta".into(),
            crate_hop: 1,
            direct: true,
            kind: Some("normal".into()),
            files: vec![
                ImpactFile {
                    path: "crates/beta/src/lib.rs".into(),
                    file_hops: 1,
                    via: vec!["use alpha::thing".into()],
                    churn: Some(4),
                },
                ImpactFile {
                    path: "crates/beta/src/other.rs".into(),
                    file_hops: 2,
                    via: vec!["use alpha::lib::thing".into()],
                    churn: None,
                },
            ],
        }],
        max_hops: 2,
        crate_count: 1,
        file_count: 2,
        churn: ChurnSummary {
            own_commits: 10,
            partner_count: 1,
            total_partner_commits: 4,
            known: true,
        },
        basis: "edges are resolved `use`/`mod`/`impl` names, not call sites".into(),
    }
}

fn app_with_tree() -> App<'static> {
    let mut app = App::for_tests();
    app.focus = FocusArea::ImpactPanel;
    app.impact_tree = Some(tree());
    app
}

#[test]
fn slash_impact_with_no_argument_shows_usage_and_does_not_move_focus() {
    let mut app = App::for_tests();
    assert_eq!(app.focus, FocusArea::ChatInput);
    submit(&mut app, "/impact");
    assert_eq!(app.focus, FocusArea::ChatInput, "usage is not a query");
    assert!(app.impact_tree.is_none());
    let last = app.messages.last().expect("a reply was written");
    assert_eq!(last.role, "system");
    assert!(
        last.content.contains("usage: /impact <file>"),
        "the usage line should name the arg, got {:?}",
        last.content
    );
}

#[test]
fn arrow_down_walks_rows_and_arrow_up_stops_at_zero() {
    let mut app = app_with_tree();
    assert_eq!(app.impact_selected, 0);
    press(&mut app, KeyCode::Down);
    assert_eq!(app.impact_selected, 1);
    press(&mut app, KeyCode::Down);
    assert_eq!(app.impact_selected, 2);
    press(&mut app, KeyCode::Down);
    assert_eq!(app.impact_selected, 3);
    // 1 target + 1 crate header + 2 file rows = 4 rows, so the cursor clamps.
    press(&mut app, KeyCode::Down);
    assert_eq!(app.impact_selected, 3);
    for _ in 0..10 {
        press(&mut app, KeyCode::Up);
    }
    assert_eq!(app.impact_selected, 0);
}

#[test]
fn enter_opens_detail_and_pressing_it_again_returns_to_the_list() {
    let mut app = app_with_tree();
    assert!(!app.impact_detail);
    press(&mut app, KeyCode::Enter);
    assert!(app.impact_detail);
    // Detail-mode ↑/↓ scroll the paragraph, not the list cursor.
    app.impact_scroll = 3;
    press(&mut app, KeyCode::Down);
    assert_eq!(app.impact_scroll, 4);
    press(&mut app, KeyCode::Enter);
    assert!(!app.impact_detail);
    assert_eq!(app.impact_scroll, 0, "reopening resets the scroll");
}

#[test]
fn right_arrow_descends_onto_a_file_row_and_pushes_the_previous_target() {
    let mut app = app_with_tree();
    // Rows: 0 Target, 1 Crate, 2 File beta/src/lib.rs, 3 File beta/src/other.rs
    app.impact_selected = 1;
    press(&mut app, KeyCode::Right);
    assert!(
        app.impact_history.is_empty(),
        "→ is a no-op on a crate header, not a re-query the user did not ask for"
    );
    app.impact_selected = 0;
    press(&mut app, KeyCode::Right);
    assert!(
        app.impact_history.is_empty(),
        "→ is a no-op on the target row too"
    );
    app.impact_selected = 2;
    press(&mut app, KeyCode::Right);
    assert_eq!(
        app.impact_history,
        vec!["crates/alpha/src/lib.rs".to_string()],
        "the target we left is what ← pops"
    );
    assert_eq!(
        app.impact_selected, 0,
        "the new tree starts at its own target"
    );
    assert!(
        !app.impact_detail,
        "a descend closes detail with the old row"
    );
}

#[test]
fn left_arrow_pops_the_descend_stack_and_a_second_press_is_inert() {
    let mut app = app_with_tree();
    app.impact_history.push("crates/alpha/src/lib.rs".into());
    app.impact_selected = 3;
    app.impact_detail = true;
    app.impact_scroll = 5;
    press(&mut app, KeyCode::Left);
    assert!(app.impact_history.is_empty());
    assert_eq!(app.impact_selected, 0);
    assert!(!app.impact_detail);
    assert_eq!(app.impact_scroll, 0);
    // With no stack left, ← does nothing (it does not close the panel).
    press(&mut app, KeyCode::Left);
    assert_eq!(app.focus, FocusArea::ImpactPanel);
}

#[test]
fn r_re_runs_the_query_on_the_current_target_without_touching_the_history() {
    let mut app = app_with_tree();
    app.impact_history
        .push("crates/alpha/src/earlier.rs".into());
    app.impact_selected = 2;
    app.impact_detail = true;
    app.impact_scroll = 4;
    press(&mut app, KeyCode::Char('r'));
    assert_eq!(
        app.impact_history.len(),
        1,
        "r is a re-query in place, not a navigation — the ← stack must survive"
    );
    assert_eq!(app.impact_selected, 0, "the cursor returns to the top");
    assert!(!app.impact_detail, "detail closes with the re-query");
    assert_eq!(app.impact_scroll, 0, "the detail scroll resets too");
    assert_eq!(
        app.focus,
        FocusArea::ImpactPanel,
        "r does not close the panel"
    );
}

#[test]
fn esc_unwinds_detail_then_descend_stack_then_closes_to_chat() {
    let mut app = app_with_tree();
    app.impact_history = vec!["crates/alpha/src/lib.rs".into()];
    app.impact_detail = true;
    // Stage 1: leave detail; panel stays focused, stack untouched.
    press(&mut app, KeyCode::Esc);
    assert!(!app.impact_detail);
    assert_eq!(app.focus, FocusArea::ImpactPanel);
    assert_eq!(app.impact_history.len(), 1);
    // Stage 2: pop the descend stack.
    press(&mut app, KeyCode::Esc);
    assert!(app.impact_history.is_empty());
    assert_eq!(app.focus, FocusArea::ImpactPanel);
    // Stage 3: close to chat.
    press(&mut app, KeyCode::Esc);
    assert_eq!(app.focus, FocusArea::ChatInput);
}

#[test]
fn rows_are_the_projection_the_panel_walks_not_a_recomputed_tree() {
    // Regression guard against a renderer walking a different order than the
    // keymap does. The rows the panel highlights must be exactly the rows the
    // keyboard cursor moves through.
    let t = tree();
    let rows = t.rows();
    assert_eq!(rows.len(), 4);
    assert!(matches!(rows[0], ImpactRow::Target { .. }));
    assert!(matches!(&rows[1], ImpactRow::Crate { crate_name, .. } if crate_name == "beta"));
    assert!(
        matches!(&rows[2], ImpactRow::File { path, churn: Some(4), .. }
        if path == "crates/beta/src/lib.rs")
    );
    assert!(matches!(&rows[3], ImpactRow::File { path, churn: None, .. }
        if path == "crates/beta/src/other.rs"));
}

#[test]
fn o_is_only_bound_while_the_cursor_sits_on_a_file_row() {
    // `open_file_in_editor` reads the path (or fills the editor with an
    // "Unable to read" line when it can't). The test's tree uses a fake path
    // that does not exist, so `o` on a File row must be the only one that
    // touches the editor buffer; on Target or Crate rows, `o` is inert.
    let mut app = app_with_tree();
    let before: Vec<String> = app.editor.lines().to_vec();

    app.impact_selected = 0; // Target
    press(&mut app, KeyCode::Char('o'));
    let after_target: Vec<String> = app.editor.lines().to_vec();
    assert_eq!(
        after_target, before,
        "o on the target row must not open anything"
    );

    app.impact_selected = 1; // Crate
    press(&mut app, KeyCode::Char('o'));
    let after_crate: Vec<String> = app.editor.lines().to_vec();
    assert_eq!(
        after_crate, before,
        "o on a crate header must not open anything"
    );

    app.impact_selected = 2; // File
    press(&mut app, KeyCode::Char('o'));
    let after_file: Vec<String> = app.editor.lines().to_vec();
    assert_ne!(after_file, before, "o on a file row must reach the editor");
    assert!(
        after_file
            .iter()
            .any(|l| l.contains("crates/beta/src/lib.rs")),
        "the editor names the file the cursor was on, got {after_file:?}"
    );
}

/// Live check against this workspace: the real `/impact <file>` path runs
/// `cargo metadata`, the file-symbol graph and `git log` for one file we know
/// exists. This is the item's done-when for "the panel is wired to QD-1", not
/// a hand-built projection. We only assert the wiring opened and produced a
/// tree — the actual numbers depend on this workspace's live shape.
#[test]
fn the_slash_command_opens_the_panel_over_a_real_workspace_file() {
    use std::path::PathBuf;
    // The test binary runs with cwd inside the crate, so move to workspace
    // root for `default_root()` in `refresh_impact_for`.
    let ws = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("rust/")
        .to_path_buf();
    std::env::set_current_dir(&ws).unwrap();

    let mut app = App::for_tests();
    submit(&mut app, "/impact crates/xencode-core-rs/src/lib.rs");
    assert_eq!(app.focus, FocusArea::ImpactPanel);
    let t = app
        .impact_tree
        .as_ref()
        .unwrap_or_else(|| panic!("expected a tree, got status {:?}", app.impact_status));
    assert!(
        t.target.ends_with("crates/xencode-core-rs/src/lib.rs"),
        "target widened to {}",
        t.target
    );
}
