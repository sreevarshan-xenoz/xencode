//! Persisting the window arrangement across restarts (`V-6`).
//!
//! One file, `<config dir>/layout.json`, holding the arrangement and nothing
//! else: the resized tree with its ratios, which pane was focused, and the
//! layout name the arrangement was taken from. Not the work — no transcripts,
//! no model state, no tool state, nothing a worker owns. Those belong to the
//! memory crate and LF-4, and copying them here is how a session grows two
//! sources of truth.
//!
//! Four existing decisions bind this file completely, and all four are cheap
//! to obey: `DB-1`'s `write_atomic` is the only writer (atomic, `0600`, no
//! torn reads), `SE-1`'s owner-only mode comes with it, and `DB-2`'s version
//! ladder is the compatibility story — except that `DB-2` has not shipped, so
//! this file carries its own `version` field and refuses anything newer with
//! an explanation instead of parsing it. The O register's rule that JSONL
//! suffices and SQLite is declined is why this is one small JSON document and
//! not a store.
//!
//! A saved arrangement is only ever a restored `custom_view` — exactly what
//! a resize chord produces (`V-3`). `Ctrl+U` clearing it clears the saved
//! tree too, and a layout name changed anywhere (config edit, the Settings
//! row) makes the stored arrangement stale on the next start: the config
//! name is what the user asked for, so it wins and the old tree is dropped.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::app::App;
use crate::focus::FocusArea;
use crate::view::ViewState;

/// The version this build writes, and the highest it can read. The next
/// shape change bumps it; anything above it in a file means the file was
/// written by a newer xencode, and that is said rather than guessed at.
pub const ARRANGEMENT_VERSION: u32 = 1;

/// The name of the file inside the config directory.
pub const ARRANGEMENT_FILE: &str = "layout.json";

/// What a restart owes the window arrangement, and nothing more.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arrangement {
    pub version: u32,
    /// The layout name the arrangement belongs to. The name itself lives in
    /// config (which already persists it); it is recorded here so a restored
    /// tree can be recognised as belonging to a layout the user has since
    /// changed away from.
    pub name: String,
    /// Which pane held focus when the app closed.
    pub focus: FocusArea,
    /// The resized tree, if there was one. `None` means presets decide —
    /// writing that explicitly is the point: clearing an arrangement must
    /// overwrite the old tree, not leave it to come back next start.
    #[serde(default)]
    pub view: Option<ViewState>,
}

/// Why a stored arrangement could not be read. Every variant is a sentence
/// that can be shown as-is: the user's screen geometry should come back, and
/// when it does not, the reason belongs on the screen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadError {
    /// A file this build cannot parse, in the user-visible words. Covers a
    /// newer version (reported as such, before any parsing is attempted),
    /// hand-edited garbage, and a tree holding geometry this vocabulary
    /// cannot store.
    Refused(String),
    /// The file could not be read at all. The OS sentence is carried, because
    /// "permission denied" and "not a file" deserve different follow-ups.
    Io(String),
}

/// Where the arrangement file lives: beside `config.json`, in whatever
/// directory config uses (so `XENCODE_CONFIG_DIR` moves both together).
pub fn path() -> Option<PathBuf> {
    xencode_config_rs::XencodeConfig::config_dir()
        .ok()
        .map(|dir| dir.join(ARRANGEMENT_FILE))
}

/// Read one arrangement file. A file whose version is newer than this build
/// is answered before it is parsed — "written by a newer xencode" is the
/// truth, where a field error would be a guess.
pub fn read_from(path: &Path) -> Result<Arrangement, ReadError> {
    let bytes = std::fs::read(path).map_err(|e| ReadError::Io(e.to_string()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| ReadError::Refused(e.to_string()))?;
    let stored = value.get("version").and_then(|v| v.as_u64()).unwrap_or(0);
    if stored > ARRANGEMENT_VERSION as u64 {
        return Err(ReadError::Refused(format!(
            "{ARRANGEMENT_FILE} was written by a newer xencode (version {stored}, \
             this build reads up to {ARRANGEMENT_VERSION}) — not restored"
        )));
    }
    serde_json::from_value(value).map_err(|e| ReadError::Refused(e.to_string()))
}

/// Write one arrangement file, through `DB-1`'s helper so it is `0600` and
/// cannot be seen half-written.
pub fn write_to(path: &Path, arrangement: &Arrangement) -> Result<(), ReadError> {
    let json =
        serde_json::to_vec_pretty(arrangement).map_err(|e| ReadError::Refused(e.to_string()))?;
    xencode_core_rs::write_atomic(path, &json).map_err(|e| ReadError::Io(e.to_string()))
}

/// Snapshot what a running app would restore: the tree it is holding (or the
/// explicit absence of one), the focused pane, and the layout name both
/// belong to.
pub fn capture(app: &App) -> Arrangement {
    Arrangement {
        version: ARRANGEMENT_VERSION,
        name: app.config.layout.clone(),
        focus: app.last_body_focus,
        view: app.custom_view.clone(),
    }
}

/// What restoring an arrangement into a starting app did, or why nothing was
/// restored. The app decides whether and where to say these; the point is
/// that a silent non-restoration is not an option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Restored {
    /// Nothing to do: no file has ever been written.
    Nothing,
    /// The file was unreadable or from a newer build; nothing was touched.
    Skipped(String),
    /// The file was fine but its layout name is no longer the configured
    /// one — the user changed layouts since the last start, so the config
    /// wins and the stored tree was dropped.
    Stale,
    /// The arrangement is on screen.
    Applied,
}

/// Restore one read arrangement into a starting app. Geometry and focus come
/// back; a name that no longer matches the config drops the tree, because
/// config is what the user asked for most recently.
pub fn apply(app: &mut App, arrangement: &Arrangement) -> Restored {
    if arrangement.name != app.config.layout {
        return Restored::Stale;
    }
    if let Some(view) = &arrangement.view {
        app.custom_view = Some(view.clone());
    }
    app.last_body_focus = arrangement.focus;
    Restored::Applied
}

/// Startup entry point: read [`path()`] and restore what it holds.
pub fn load_into(app: &mut App) -> Restored {
    let Some(path) = path() else {
        return Restored::Nothing;
    };
    if !path.exists() {
        return Restored::Nothing;
    }
    match read_from(&path) {
        Ok(arrangement) => apply(app, &arrangement),
        Err(ReadError::Refused(why)) => Restored::Skipped(why),
        Err(ReadError::Io(why)) => Restored::Skipped(format!("{ARRANGEMENT_FILE}: {why}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::focus::FocusArea;
    use crate::view::{classic_tree, nudge_focused};
    use ratatui::layout::Rect;

    fn temp_path(label: &str) -> PathBuf {
        // Same convention as the config crate's tests: a process-wide counter,
        // not a timestamp, because parallel threads can share a nanosecond.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "xencode-arrangement-{label}-{}-{unique}.json",
            std::process::id()
        ))
    }

    fn resized_app() -> App<'static> {
        let mut app = App::for_tests();
        app.config.layout = "classic".to_string();
        let area = Rect::new(0, 1, 100, 22);
        let mut tree = classic_tree(area, false);
        assert!(nudge_focused(&mut tree, FocusArea::CodeEditor, 5));
        app.custom_view = Some(ViewState::new(tree));
        app.last_body_focus = FocusArea::CodeEditor;
        app
    }

    #[test]
    fn a_saved_arrangement_round_trips_with_geometry_and_focus() {
        let app = resized_app();
        let before = app
            .custom_view
            .as_ref()
            .expect("resized")
            .render(Rect::new(0, 1, 100, 22));
        let path = temp_path("roundtrip");
        let arrangement = capture(&app);
        write_to(&path, &arrangement).expect("write");

        let restored = read_from(&path).expect("read");
        assert_eq!(restored.name, "classic");
        assert_eq!(restored.focus, FocusArea::CodeEditor);
        let mut fresh = App::for_tests();
        fresh.config.layout = "classic".to_string();
        assert_eq!(apply(&mut fresh, &restored), Restored::Applied);
        let after = fresh
            .custom_view
            .as_ref()
            .expect("the tree came back")
            .render(Rect::new(0, 1, 100, 22));
        assert_eq!(before, after, "same rects, same panes");
        assert_eq!(fresh.last_body_focus, FocusArea::CodeEditor);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_file_is_owner_only_and_carries_the_version() {
        let path = temp_path("mode");
        let app = resized_app();
        write_to(&path, &capture(&app)).expect("write");
        let bytes = std::fs::read(&path).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["version"], ARRANGEMENT_VERSION);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600,
                "SE-1: this records what you were working on"
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_newer_version_is_rejected_with_an_explanation_not_parsed() {
        let path = temp_path("future");
        std::fs::write(&path, br#"{"version": 99, "name": "classic"}"#).unwrap();
        let err = read_from(&path).expect_err("must refuse");
        match err {
            ReadError::Refused(why) => {
                assert!(why.contains("newer xencode"), "{why}");
                assert!(why.contains("version 99"), "{why}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_corrupt_file_is_refused_with_the_reason_and_restores_nothing() {
        let path = temp_path("corrupt");
        std::fs::write(&path, b"{ not json").unwrap();
        let mut app = App::for_tests();
        let err = read_from(&path).expect_err("must refuse");
        assert!(matches!(err, ReadError::Refused(_)));
        std::fs::remove_file(&path).ok();

        // The same sentence the app would show; nothing was applied either way.
        assert_eq!(app.custom_view, None);
        assert_eq!(
            apply(
                &mut app,
                &Arrangement {
                    version: ARRANGEMENT_VERSION,
                    name: "no-such-layout".to_string(),
                    focus: FocusArea::ChatInput,
                    view: Some(ViewState::new(classic_tree(
                        Rect::new(0, 1, 100, 22),
                        false
                    ))),
                }
            ),
            Restored::Stale
        );
        assert_eq!(app.custom_view, None, "a stale name restores nothing");
    }

    #[test]
    fn clearing_an_arrangement_overwrites_the_stored_tree() {
        // Ctrl+U clears the tree; if the file kept the old one, the next
        // start would resurrect an arrangement the user threw away.
        let path = temp_path("clear");
        let mut app = resized_app();
        write_to(&path, &capture(&app)).expect("first write");
        app.custom_view = None;
        write_to(&path, &capture(&app)).expect("clear write");
        let restored = read_from(&path).expect("read");
        assert_eq!(restored.view, None, "the clear is what is stored now");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_hand_edited_tree_that_does_not_build_is_refused_by_name() {
        // The tree codec shares its vocabulary with templates, so a stored
        // zero share must fail the same way a template's does — a sentence,
        // not a half-built tree on screen.
        let path = temp_path("badtree");
        std::fs::write(
            &path,
            br#"{
                "version": 1,
                "name": "classic",
                "focus": "chatinput",
                "view": {"root": {"split": {"horizontal": true, "parts": [
                    [{"leaf": {"slot": "editor", "focus": "editor"}}, {"percent": 0}],
                    [{"leaf": {"slot": "chat", "focus": "chat"}}, {"percent": 100}]
                ]}}}
            }"#,
        )
        .unwrap();
        let err = read_from(&path).expect_err("must refuse");
        match err {
            ReadError::Refused(why) => assert!(why.contains("zero share"), "{why}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stored_words_are_the_words_templates_already_speak() {
        // One vocabulary for hand-written config templates and the file the
        // app writes itself: the same JSON a template reader accepts, this
        // codec must accept, and vice versa.
        let classic_json = serde_json::to_value(crate::templates::preset_template("classic"))
            .expect("template encodes");
        let node =
            crate::templates::template_from_value(&classic_json).expect("the twin builds a tree");
        let encoded = serde_json::to_value(&node).expect("the tree encodes");
        let decoded: crate::view::LayoutNode =
            serde_json::from_value(encoded.clone()).expect("and decodes back");
        assert_eq!(decoded, node);
        // The shape says so in the same words a template uses.
        assert!(encoded.get("split").is_some(), "{encoded}");
        let parts = encoded["split"]["parts"].as_array().unwrap();
        let pair = parts[0].as_array().unwrap();
        let share = &pair[1];
        assert!(share.get("percent").is_some(), "{share}");

        // And the other direction: the tree JSON is a vocabulary a config
        // template reader already accepts, so a hand-written file and an
        // app-written file cannot drift into two dialects.
        let edited = r#"{"leaf": {"slot": "editor", "focus": "editor"}}"#;
        let value: serde_json::Value = serde_json::from_str(edited).unwrap();
        crate::templates::template_from_value(&value).expect("template reader accepts it");
        let node: crate::view::LayoutNode =
            serde_json::from_value(value).expect("tree codec accepts it");
        assert_eq!(
            node,
            crate::view::LayoutNode::Leaf(crate::view::Pane {
                slot: crate::view::BodySlot::Editor,
                focus: FocusArea::CodeEditor,
            })
        );
    }
}
