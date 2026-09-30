//! Named views, saved and recalled (`V-4`).
//!
//! A view is a saved window arrangement with a name on it: switching to one
//! restores its exact geometry and its focused pane, in one chord
//! (`Ctrl+1`…`Ctrl+9`), and storing one takes the arrangement currently on
//! screen (`Ctrl+Shift+<digit>`). The value is modest and real — "I want the
//! diff and the chat together" is arranged once instead of resized every
//! session.
//!
//! Six of the nine slots are seeded with an arrangement built in code, so
//! the chord does something from the first start with an empty config. A
//! stored view lives in `XencodeConfig::layout_views` under its slot name,
//! written through the same config path every other setting uses. The file
//! format for that map is config's own, not a new one — the boundary `V-5`
//! set for templates holds here too: no view directory.
//!
//! **Views are optional, additive, and never the only way to reach a
//! panel.** `Ctrl+T`, `Ctrl+U` and the resize chords work exactly the same
//! with no view ever named; the Feature Navigator (`Ctrl+F`) remains the
//! discovery path. A view is a shortcut, not a gate.
//!
//! They are called *views*, not workspaces: Milestone G owns that word for
//! live collaboration (Finding 5).
//!
//! The relationship to `V-6`: this module names arrangements, that one
//! remembers the live one across a restart. A stored view is the user's own
//! config, so it needs no new file; `layout.json` carries only which view was
//! active, so a restart comes back on the view it left on.

use crate::focus::FocusArea;
use crate::view::{chat_column, leaf, BodySlot, LayoutNode, TERMINAL_MIN_HEIGHT};
use ratatui::layout::{Constraint, Rect};

/// The views a config declares: slot name → raw JSON tree. Aliased so the
/// signatures here read as "the config's views", the way `templates::Templates`
/// does for layouts.
pub type Views = std::collections::BTreeMap<String, serde_json::Value>;

/// The number of view chords: `Ctrl+1`…`Ctrl+9`.
pub const VIEW_SLOTS: usize = 9;

fn split_horizontal(parts: Vec<(LayoutNode, Constraint)>) -> LayoutNode {
    LayoutNode::Split {
        horizontal: true,
        parts,
    }
}

fn column(area: Rect, show_terminal: bool) -> LayoutNode {
    chat_column(show_terminal, area.height >= TERMINAL_MIN_HEIGHT)
}

fn code_tree(area: Rect, show_terminal: bool) -> LayoutNode {
    split_horizontal(vec![
        (
            leaf(BodySlot::Explorer, FocusArea::FileExplorer),
            Constraint::Percentage(15),
        ),
        (
            leaf(BodySlot::Editor, FocusArea::CodeEditor),
            Constraint::Percentage(60),
        ),
        (column(area, show_terminal), Constraint::Percentage(25)),
    ])
}

fn chat_tree(area: Rect, show_terminal: bool) -> LayoutNode {
    split_horizontal(vec![
        (
            leaf(BodySlot::Editor, FocusArea::CodeEditor),
            Constraint::Percentage(20),
        ),
        (column(area, show_terminal), Constraint::Percentage(80)),
    ])
}

/// Editor narrow, chat wide, terminal strip in the tree. Every custom tree
/// already decides the strip at switch time — the flag cannot re-shape a tree
/// that exists — so this is what `render` does with a resized arrangement,
/// named. `Ctrl+T` still owns the flag for the presets.
fn terminal_tree(area: Rect, _show_terminal: bool) -> LayoutNode {
    split_horizontal(vec![
        (
            leaf(BodySlot::Editor, FocusArea::CodeEditor),
            Constraint::Percentage(20),
        ),
        (column(area, true), Constraint::Percentage(80)),
    ])
}

/// The whole body is the chat column: transcript, terminal strip, input.
fn focus_tree(area: Rect, show_terminal: bool) -> LayoutNode {
    column(area, show_terminal)
}

/// Files down one side of the code, transcript across the bottom — the shape
/// for reading a diff while the review panel renders in the chat column.
fn review_tree(area: Rect, show_terminal: bool) -> LayoutNode {
    LayoutNode::Split {
        horizontal: false,
        parts: vec![
            (
                split_horizontal(vec![
                    (
                        leaf(BodySlot::Explorer, FocusArea::FileExplorer),
                        Constraint::Percentage(25),
                    ),
                    (
                        leaf(BodySlot::Editor, FocusArea::CodeEditor),
                        Constraint::Percentage(75),
                    ),
                ]),
                Constraint::Percentage(60),
            ),
            (column(area, show_terminal), Constraint::Percentage(40)),
        ],
    }
}

/// Half the screen for the code, half for the conversation.
fn split_tree(area: Rect, show_terminal: bool) -> LayoutNode {
    split_horizontal(vec![
        (
            leaf(BodySlot::Editor, FocusArea::CodeEditor),
            Constraint::Percentage(50),
        ),
        (column(area, show_terminal), Constraint::Percentage(50)),
    ])
}

/// Code editor on the left, agent stack in the middle, chat column on the right.
fn agents_tree(area: Rect, show_terminal: bool) -> LayoutNode {
    split_horizontal(vec![
        (
            leaf(BodySlot::Editor, FocusArea::CodeEditor),
            Constraint::Percentage(40),
        ),
        (
            leaf(BodySlot::Agents, FocusArea::ByteBotPanel),
            Constraint::Percentage(30),
        ),
        (column(area, show_terminal), Constraint::Percentage(30)),
    ])
}

/// One seeded view: its chord name, the pane it focuses, and the tree it
/// draws. The shapes are code, not config data — same reason the presets are
/// builders: they must exist before any file declares them.
#[derive(Debug)]
pub struct ViewDef {
    /// The name the slot answers to, and the key it is stored under.
    pub name: &'static str,
    /// The body pane this view focuses when it is switched to.
    pub focus: FocusArea,
    /// The arrangement. Takes whether the terminal strip is shown, like the
    /// preset builders do.
    pub tree: fn(area: Rect, show_terminal: bool) -> LayoutNode,
}

/// The seeded views, in chord order `Ctrl+1`…`Ctrl+7`. Slots 8–9 have no
/// seed: they answer only once something is stored there.
pub const SEEDED: [ViewDef; 7] = [
    ViewDef {
        name: "Code",
        focus: FocusArea::CodeEditor,
        tree: code_tree,
    },
    ViewDef {
        name: "Chat",
        focus: FocusArea::ChatInput,
        tree: chat_tree,
    },
    ViewDef {
        name: "Terminal",
        focus: FocusArea::ChatInput,
        tree: terminal_tree,
    },
    ViewDef {
        name: "Focus",
        focus: FocusArea::ChatInput,
        tree: focus_tree,
    },
    ViewDef {
        name: "Review",
        focus: FocusArea::CodeEditor,
        tree: review_tree,
    },
    ViewDef {
        name: "Split",
        focus: FocusArea::CodeEditor,
        tree: split_tree,
    },
    ViewDef {
        name: "Agents",
        focus: FocusArea::ByteBotPanel,
        tree: agents_tree,
    },
];

/// The name a slot answers to: the seeded name for 1–6, the digit as a word
/// otherwise. This is the key a stored view is kept under.
pub fn slot_name(slot: usize) -> String {
    SEEDED
        .get(slot - 1)
        .map(|def| def.name.to_string())
        .unwrap_or_else(|| slot.to_string())
}

/// The name to show for a slot: the seeded name where there is one, `view
/// {n}` otherwise.
pub fn slot_label(slot: usize) -> String {
    SEEDED
        .get(slot - 1)
        .map(|def| def.name.to_string())
        .unwrap_or_else(|| format!("view {slot}"))
}

/// What a `Ctrl+<digit>` chord asks for: recall a view, or store the
/// arrangement on screen into one. `Shift` is the difference, and a digit
/// outside 1–9 is not a view chord at all — `0` included, which is one short
/// of the focus ring rather than a slot. Digits reach the app as `Char('1')`
/// with CONTROL held (the kitty/CSI-u encoding the letter chords already
/// rely on); a terminal that sends the old keypress form yields a
/// non-canonical codepoint and falls through unhandled.
#[derive(Debug, PartialEq, Eq)]
pub enum ViewChord {
    /// Put this view on screen.
    Switch(usize),
    /// Save the current arrangement under this view's name.
    Store(usize),
}

pub fn chord(code: char, shift: bool) -> Option<ViewChord> {
    let digit = code.to_digit(10)? as usize;
    if digit == 0 || digit > VIEW_SLOTS {
        return None;
    }
    Some(if shift {
        ViewChord::Store(digit)
    } else {
        ViewChord::Switch(digit)
    })
}

/// The view slots, in chord order: the six seeded names then `7`…`9`, the
/// slots that answer only once something is stored. The Settings row reads
/// this so it cannot disagree with the chord.
pub fn slot_list() -> Vec<String> {
    (1..=VIEW_SLOTS).map(slot_name).collect()
}

/// What a slot switches to: a tree stored in config, which wins, or the
/// seeded shape in code.
#[derive(Debug)]
pub enum Seed<'a> {
    /// A view stored (or hand-written) in config for this slot name.
    Declared(&'a serde_json::Value),
    /// The built-in arrangement for this slot.
    BuiltIn(&'static ViewDef),
}

/// Resolve a slot to what it should draw. A config entry for a seeded name
/// overrides the seed — that is what `Ctrl+Shift+<digit>` wrote — and slots
/// with no entry and no seed have nothing to switch to.
pub fn seed_for<'a>(declared: &'a Views, slot: usize) -> Option<Seed<'a>> {
    let name = slot_name(slot);
    if let Some(value) = declared.get(&name) {
        return Some(Seed::Declared(value));
    }
    SEEDED.get(slot - 1).map(Seed::BuiltIn)
}

/// Why switching to `slot` would not show what its name says, if it would
/// not: nothing stored in an unseeded slot, or a stored tree that refuses to
/// build, said with the template's own reason (the vocabulary is shared, so
/// the sentence is). `None` means the view has a shape to put on screen.
pub fn problem(declared: &Views, slot: usize) -> Option<String> {
    match seed_for(declared, slot) {
        None => Some(format!(
            "{} holds no view yet — Ctrl+Shift+{slot} stores the screen as one",
            slot_label(slot)
        )),
        Some(Seed::BuiltIn(_)) => None,
        Some(Seed::Declared(value)) => match crate::templates::template_from_value(value) {
            Ok(_) => None,
            Err(e) => Some(format!("view {} refused: {}", slot_name(slot), e.0)),
        },
    }
}

/// The arrangement a slot puts on screen, with the pane it focuses: a stored
/// tree read from config, or the seeded shape built at `area`. `None` is the
/// empty-slot case — [`problem`] names it, and the caller shows that rather
/// than guessing. Assumes the slot has been checked, which is why it can
/// unwrap the tree build.
pub fn shape(
    declared: &Views,
    slot: usize,
    area: Rect,
    show_terminal: bool,
) -> Option<(LayoutNode, FocusArea)> {
    match seed_for(declared, slot)? {
        Seed::BuiltIn(def) => Some(((def.tree)(area, show_terminal), def.focus)),
        Seed::Declared(value) => {
            let node = crate::templates::template_from_value(value).ok()?;
            let focus = primary_focus(&node);
            Some((node, focus))
        }
    }
}

/// The pane a stored tree considers primary: the first leaf in reading order
/// that holds a body pane — the same order the panes are drawn in, so the
/// focus lands where the eye does. A tree made only of fixed strips has no
/// body leaf to name and stays with the chat input.
pub fn primary_focus(node: &LayoutNode) -> FocusArea {
    fn walk(node: &LayoutNode) -> Option<FocusArea> {
        match node {
            LayoutNode::Leaf(pane) => match pane.slot {
                BodySlot::Explorer | BodySlot::Editor | BodySlot::Agents | BodySlot::Chat => {
                    Some(pane.focus)
                }
                BodySlot::Input | BodySlot::Terminal => None,
            },
            LayoutNode::Split { parts, .. } => parts.iter().find_map(|(child, _)| walk(child)),
            LayoutNode::Tabbed { panes, .. } | LayoutNode::Stack { panes, .. } => {
                panes.iter().find_map(|pane| match pane.slot {
                    BodySlot::Explorer | BodySlot::Editor | BodySlot::Agents | BodySlot::Chat => {
                        Some(pane.focus)
                    }
                    BodySlot::Input | BodySlot::Terminal => None,
                })
            }
        }
    }
    walk(node).unwrap_or(FocusArea::ChatInput)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::to_body_layout;

    fn area() -> Rect {
        Rect::new(0, 1, 120, 40)
    }

    fn declared(entries: &[(&str, &str)]) -> Views {
        entries
            .iter()
            .map(|(name, json)| {
                (
                    name.to_string(),
                    serde_json::from_str(json).expect("test view is valid JSON"),
                )
            })
            .collect()
    }

    const SIDE: &str = r#"{"split": {"horizontal": true, "parts": [
        [{"leaf": {"slot": "editor", "focus": "editor"}}, {"percent": 70}],
        [{"leaf": {"slot": "chat", "focus": "chat"}}, {"percent": 30}]
    ]}}"#;

    #[test]
    fn six_seeded_views_have_distinct_shapes_and_named_focus() {
        // The done-when's "switches in one chord" rests on each seed being a
        // real arrangement: its slots present, its focused pane named.
        for def in SEEDED.iter() {
            let tree = (def.tree)(area(), false);
            let leaves = crate::view::render(&tree, area());
            assert!(!leaves.is_empty(), "{} draws nothing", def.name);
            assert!(
                leaves.iter().any(|(p, _)| p.focus == def.focus),
                "{} does not contain its focused pane {:?}",
                def.name,
                def.focus
            );
        }
        let code = to_body_layout(&crate::view::render(
            &(SEEDED[0].tree)(area(), false),
            area(),
        ));
        assert!(code.explorer.is_some() && code.editor.is_some() && code.chat.is_some());
        assert_eq!(code.explorer.unwrap().width, 18, "15% of 120");
        let focus = to_body_layout(&crate::view::render(
            &(SEEDED[3].tree)(area(), false),
            area(),
        ));
        assert!(focus.explorer.is_none() && focus.editor.is_none());
        assert!(focus.chat.is_some() && focus.input.is_some());
        let terminal = to_body_layout(&crate::view::render(
            &(SEEDED[2].tree)(area(), false),
            area(),
        ));
        assert!(
            terminal.terminal.is_some(),
            "the Terminal view shows the strip even when it was off"
        );
    }

    #[test]
    fn a_stored_view_replaces_the_seed_and_an_unseeded_slot_needs_one() {
        let views = declared(&[("Code", SIDE), ("8", SIDE)]);
        // Config wins over the seed for a seeded name…
        match seed_for(&views, 1) {
            Some(Seed::Declared(_)) => {}
            other => panic!("expected the stored Code view, got {other:?}"),
        }
        // …and is the only source for slots 8–9.
        assert!(matches!(seed_for(&views, 8), Some(Seed::Declared(_))));
        assert!(seed_for(&views, 9).is_none());
        // Without a stored entry, a seeded slot falls back to its shape.
        assert!(matches!(seed_for(&Views::new(), 1), Some(Seed::BuiltIn(_))));
        assert!(matches!(seed_for(&Views::new(), 6), Some(Seed::BuiltIn(_))));
        assert!(matches!(seed_for(&Views::new(), 7), Some(Seed::BuiltIn(_))));
        assert!(seed_for(&Views::new(), 8).is_none());
    }

    #[test]
    fn a_broken_stored_view_is_refused_with_the_templates_sentence() {
        let views = declared(&[(
            "Code",
            r#"{"leaf": {"slot": "sidebar", "focus": "editor"}}"#,
        )]);
        let why = problem(&views, 1).expect("refused");
        assert!(why.contains("unknown slot"), "{why}");
        assert!(why.contains("Code"), "{why}");
        // A seed is never refused; it has no file that could be wrong.
        assert_eq!(problem(&Views::new(), 1), None);
        // An empty slot says what is missing and which chord fixes it, rather
        // than switching to a geometry the user never asked for.
        let why = problem(&Views::new(), 9).expect("empty");
        assert!(why.contains("view 9 holds no view yet"), "{why}");
        assert!(why.contains("Ctrl+Shift+9"), "{why}");
    }

    #[test]
    fn chords_split_recall_from_store() {
        assert_eq!(chord('1', false), Some(ViewChord::Switch(1)));
        assert_eq!(chord('9', false), Some(ViewChord::Switch(9)));
        assert_eq!(chord('4', true), Some(ViewChord::Store(4)));
        assert_eq!(chord('0', false), None, "zero is not a slot");
        assert_eq!(chord('a', false), None);
    }

    #[test]
    fn a_stored_tree_focuses_its_first_body_pane_and_a_seed_its_named_one() {
        let views = declared(&[("8", SIDE)]);
        // The stored 70/30 editor-chat split: focus goes to the editor, the
        // first pane in reading order that holds one.
        let (node, focus) = shape(&views, 8, area(), false).expect("stored");
        assert_eq!(focus, FocusArea::CodeEditor);
        let body = to_body_layout(&crate::view::render(&node, area()));
        assert_eq!(body.editor.unwrap().width, 84, "70% of 120");
        assert!(body.explorer.is_none(), "a view draws its own tree");
        // The seed states its focus outright, and a slot with nothing stored
        // yields nothing rather than a guess.
        assert_eq!(
            shape(&Views::new(), 1, area(), false).map(|(_, f)| f),
            Some(FocusArea::CodeEditor)
        );
        assert_eq!(shape(&Views::new(), 8, area(), false).map(|(_, f)| f), None);
    }

    #[test]
    fn slot_names_are_stable_and_in_chord_order() {
        assert_eq!(
            slot_list(),
            vec![
                "Code".to_string(),
                "Chat".to_string(),
                "Terminal".to_string(),
                "Focus".to_string(),
                "Review".to_string(),
                "Split".to_string(),
                "Agents".to_string(),
                "8".to_string(),
                "9".to_string(),
            ]
        );
    }

    #[test]
    fn the_seven_seeds_are_seven_different_arrangements() {
        // "Seven seeded views" is only worth a chord each if each one moves the
        // panes somewhere new. Compared by rendered geometry, not by name.
        let mut shapes: Vec<Vec<(BodySlot, Rect)>> = Vec::new();
        for def in SEEDED.iter() {
            let mut leaves: Vec<(BodySlot, Rect)> =
                crate::view::render(&(def.tree)(area(), false), area())
                    .into_iter()
                    .map(|(pane, rect)| (pane.slot, rect))
                    .collect();
            leaves.sort_by_key(|(slot, rect)| (rect.x, rect.y, slot_title(slot)));
            assert!(
                !shapes.contains(&leaves),
                "{} repeats another view's geometry",
                def.name
            );
            shapes.push(leaves);
        }
        assert_eq!(shapes.len(), 7);
    }

    fn slot_title(slot: &BodySlot) -> &'static str {
        match slot {
            BodySlot::Explorer => "explorer",
            BodySlot::Editor => "editor",
            BodySlot::Chat => "chat",
            BodySlot::Input => "input",
            BodySlot::Terminal => "terminal",
            BodySlot::Agents => "agents",
        }
    }
}
