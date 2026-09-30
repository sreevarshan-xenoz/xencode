//! Layout templates, and the one list of layout names (`V-5`).
//!
//! Presets stop being cages and become starting points: the three shipped
//! arrangements, `classic`, `chat-first` and `zen`, and the templates a user
//! authors in `XencodeConfig::layout_templates` are looked up in **one name
//! registry**, so `Ctrl+U`, the header chip and the Settings row cannot
//! disagree about what exists. A new arrangement arrives as data — a
//! `LayoutTemplate` value in the config file — with no code change.
//!
//! `classic` and `chat-first` are also described here as data
//! ([`preset_template`]), and a test proves the data twin renders the same
//! rects as the builder the preset uses, so "the presets are templates" is
//! checked rather than claimed. `zen` has no twin: its single pane follows the
//! focused area, and a shape made of words cannot depend on that.
//!
//! Two rules hold the line:
//!
//! - A name that is neither a preset nor a declared template renders `classic`,
//!   which is the fallback `layout::effective_layout` has always documented.
//! - A declared template that cannot become a tree is refused **with the
//!   reason** ([`problem`]) instead of guessing a layout the user did not ask
//!   for, and it does not break loading the rest of the config: the config
//!   carries these as raw JSON, exactly so a name this build cannot read stays
//!   a refused name rather than a lost configuration file.
//!
//! Deliberately not a template *directory*: a file format for layouts is a
//! compatibility promise, and user-supplied appearance files are already out.
//! Templates live in config and inherit whatever versioning config gets.

use std::collections::BTreeMap;

use crate::focus::FocusArea;
use crate::view::{BodySlot, LayoutNode, Pane};
use ratatui::layout::{Constraint, Rect};
use serde::{Deserialize, Serialize};

/// The layout templates a config declares: name → raw JSON tree. Aliased so
/// the registry's signatures read as "the config's templates" rather than as a
/// map of JSON.
pub type Templates = BTreeMap<String, serde_json::Value>;

/// The shipped presets, in cycle order. `layout::LAYOUT_NAMES` re-exports this,
/// so the three names exist in one place.
pub const PRESET_NAMES: &[&str] = &["classic", "chat-first", "zen"];

/// How a split divides its area: the three kinds the shipped presets already
/// use, so a hand-written template speaks the same vocabulary as the builders
/// instead of a weaker one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Share {
    /// A percentage of the area — the presets' 20/50/30.
    Percent(u16),
    /// At least this many cells, sharing whatever is left.
    Min(u16),
    /// Exactly this many cells: the chat input's three rows, the embedded
    /// terminal's eight.
    Length(u16),
}

impl Share {
    /// The constraint this share becomes, in the same ratatui vocabulary the
    /// preset builders use.
    fn constraint(self) -> Constraint {
        match self {
            Share::Percent(percent) => Constraint::Percentage(percent),
            Share::Min(min) => Constraint::Min(min),
            Share::Length(length) => Constraint::Length(length),
        }
    }

    /// Zero lays out nothing. A child that cannot be seen is refused at load
    /// rather than silently drawn at zero width.
    fn is_zero(self) -> bool {
        matches!(self, Share::Percent(0) | Share::Min(0) | Share::Length(0))
    }
}

/// A serializable layout shape: leaves name slots, splits name shares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayoutTemplate {
    /// One pane: `{ "leaf": { "slot": "editor", "focus": "editor" } }`.
    Leaf {
        /// `explorer`, `editor`, `chat`, `input`, or `terminal`.
        slot: String,
        /// `explorer`, `editor`, or `chat`. Terminal and input regions read as
        /// chat, the same contract as hit-testing.
        focus: String,
    },
    /// Children side by side or stacked, each with the share it takes.
    Split {
        /// `true` for side-by-side, `false` for stacked.
        horizontal: bool,
        /// `(child, share)`.
        parts: Vec<(Box<LayoutTemplate>, Share)>,
    },
}

/// A template that failed to become a tree, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateError(pub String);

fn parse_slot(slot: &str) -> Result<BodySlot, TemplateError> {
    match slot {
        "explorer" => Ok(BodySlot::Explorer),
        "editor" => Ok(BodySlot::Editor),
        "chat" => Ok(BodySlot::Chat),
        "input" => Ok(BodySlot::Input),
        "terminal" => Ok(BodySlot::Terminal),
        "agents" => Ok(BodySlot::Agents),
        _ => Err(TemplateError(format!(
            "unknown slot {slot:?}: want explorer, editor, chat, input, terminal, or agents"
        ))),
    }
}

fn parse_focus(focus: &str) -> Result<FocusArea, TemplateError> {
    match focus {
        "explorer" => Ok(FocusArea::FileExplorer),
        "editor" => Ok(FocusArea::CodeEditor),
        "agents" => Ok(FocusArea::ByteBotPanel),
        "chat" => Ok(FocusArea::ChatInput),
        _ => Err(TemplateError(format!(
            "unknown focus {focus:?}: want explorer, editor, agents, or chat"
        ))),
    }
}

/// Build a tree from a template. Shares are validated, not normalized: zero
/// shares and a split with fewer than two children are refused, because a
/// template that cannot lay out must fail when it is chosen, not mid-frame.
pub fn template_to_node(template: &LayoutTemplate) -> Result<LayoutNode, TemplateError> {
    match template {
        LayoutTemplate::Leaf { slot, focus } => Ok(LayoutNode::Leaf(Pane {
            slot: parse_slot(slot)?,
            focus: parse_focus(focus)?,
        })),
        LayoutTemplate::Split { horizontal, parts } => {
            if parts.len() < 2 {
                return Err(TemplateError(
                    "a split needs at least two children".to_string(),
                ));
            }
            let mut children = Vec::new();
            for (child, share) in parts {
                if share.is_zero() {
                    return Err(TemplateError(
                        "a zero share lays out nothing; remove the child instead".to_string(),
                    ));
                }
                children.push((template_to_node(child)?, share.constraint()));
            }
            Ok(LayoutNode::Split {
                horizontal: *horizontal,
                parts: children,
            })
        }
    }
}

/// Read one config value as a template and build it: the shape first, then the
/// tree. One entry point, so a typo in a node name and a typo in a slot name
/// are refused the same way, with the reason.
pub fn template_from_value(value: &serde_json::Value) -> Result<LayoutNode, TemplateError> {
    let template: LayoutTemplate = serde_json::from_value(value.clone())
        .map_err(|e| TemplateError(format!("not a layout tree: {e}")))?;
    template_to_node(&template)
}

/// The preset a name is, if it is one. `None` covers both "a template from
/// config" and "no such layout"; [`problem`] tells those two apart.
pub fn preset_name(name: &str) -> Option<&'static str> {
    PRESET_NAMES.iter().copied().find(|preset| *preset == name)
}

/// Is this name backed by a template declared in config? Such a name renders
/// through the tree the config describes, which is the one shape a preset name
/// cannot produce.
pub fn is_declared(templates: &Templates, name: &str) -> bool {
    preset_name(name).is_none() && templates.contains_key(name)
}

/// Every layout name on offer, in cycle order: the shipped presets first, then
/// the declared templates in name order. A `BTreeMap` iterates sorted, so the
/// cycle is the same on every machine and every run. A template that reuses a
/// preset's name is skipped rather than listed twice — the preset wins, and
/// [`problem`] says so when that name is the active one.
pub fn names(templates: &Templates) -> Vec<String> {
    let mut out: Vec<String> = PRESET_NAMES.iter().map(|n| n.to_string()).collect();
    out.extend(
        templates
            .keys()
            .filter(|name| preset_name(name).is_none())
            .cloned(),
    );
    out
}

/// Move one step through [`names`], wrapping at both ends. An unknown current
/// name starts the cycle at the first preset, the way `cycle_layout` always
/// did.
pub fn cycle_name(templates: &Templates, active: &str, forward: bool) -> String {
    let all = names(templates);
    let len = all.len();
    let pos = all.iter().position(|name| name == active).unwrap_or(0);
    let next = if forward {
        (pos + 1) % len
    } else {
        (pos + len - 1) % len
    };
    all[next].clone()
}

/// The name to show for the layout being rendered: the name itself when it is
/// a preset or a template that builds, and `classic` otherwise — because that
/// is what the user is looking at.
pub fn effective_name(templates: &Templates, name: &str) -> String {
    if problem(templates, name).is_none() {
        name.to_string()
    } else {
        "classic".to_string()
    }
}

/// Why `name` is not rendering as written, if it is not: an unknown name, a
/// template that refuses to build, or a template shadowed by a preset name.
/// `None` means the name renders as asked. Returned as a sentence the caller
/// can show without rewording it.
pub fn problem(templates: &Templates, name: &str) -> Option<String> {
    if let Some(preset) = preset_name(name) {
        return templates.contains_key(name).then(|| {
            format!("layout template {name:?} is ignored: {name:?} is the shipped {preset} preset")
        });
    }
    match templates.get(name) {
        Some(value) => match template_from_value(value) {
            Ok(_) => None,
            Err(e) => Some(format!(
                "layout template {name:?} refused: {} — rendering classic",
                e.0
            )),
        },
        None => Some(format!(
            "unknown layout {name:?} — rendering classic; known names: {}",
            names(templates).join(", ")
        )),
    }
}

/// The tree for one configured layout name.
///
/// A preset builds through the functions the pixel-parity sweep proves; a
/// declared template builds from data; anything else renders `classic`. This
/// never fails, because a frame cannot fail: the reason a name did not render
/// as written comes from [`problem`], said at the moment the name is chosen.
pub fn tree(
    templates: &Templates,
    name: &str,
    area: Rect,
    show_terminal: bool,
    body_focus: FocusArea,
) -> LayoutNode {
    if let Some(preset) = preset_name(name) {
        return crate::view::preset_tree(area, preset, show_terminal, body_focus);
    }
    if let Some(value) = templates.get(name) {
        if let Ok(node) = template_from_value(value) {
            return node;
        }
    }
    crate::view::classic_tree(area, show_terminal)
}

/// The data shape of a preset, for the presets a template can describe:
/// `classic` and `chat-first`. `zen` returns `None`, and that is the honest
/// answer rather than a gap — its one pane follows the focused area, which is
/// a session fact, not something a shape made of words can state.
///
/// The twin is the preset *with the embedded terminal shown*: whether the
/// terminal is shown at all is a session flag (`Ctrl+T`), not a shape. For the
/// same reason the twin does not decide that a short area drops the terminal —
/// `view::render` drops it, so a short area agrees with the preset builder,
/// which leaves the leaf out. A test compares the two across the size sweep
/// rather than trusting this sentence.
pub fn preset_template(name: &str) -> Option<LayoutTemplate> {
    let leaf = |slot: &str, focus: &str| {
        Box::new(LayoutTemplate::Leaf {
            slot: slot.to_string(),
            focus: focus.to_string(),
        })
    };
    // chat, optional terminal, input — the column every preset with chat uses.
    // `chat` takes what is left over the two fixed rows: `Min(1)` here, and
    // the builder says `Min(6)`, a floor that can never bind while the terminal
    // is shown (the column keeps at least seven rows there) and that the
    // builder's short branch drops to 1 as well. One geometry, said once.
    let chat_column = |chat_share: Share| {
        (
            Box::new(LayoutTemplate::Split {
                horizontal: false,
                parts: vec![
                    (leaf("chat", "chat"), Share::Min(1)),
                    (leaf("terminal", "chat"), Share::Length(8)),
                    (leaf("input", "chat"), Share::Length(3)),
                ],
            }),
            chat_share,
        )
    };
    match name {
        "classic" => Some(LayoutTemplate::Split {
            horizontal: true,
            parts: vec![
                (leaf("explorer", "explorer"), Share::Percent(20)),
                (leaf("editor", "editor"), Share::Percent(50)),
                chat_column(Share::Percent(30)),
            ],
        }),
        "chat-first" => Some(LayoutTemplate::Split {
            horizontal: true,
            parts: vec![
                (leaf("editor", "editor"), Share::Percent(25)),
                chat_column(Share::Percent(75)),
            ],
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{render, to_body_layout};

    /// The `classic` twin as a config value: the same shape, written the way a
    /// user would write it in `layout_templates`.
    const EDITOR_FIRST: &str = r#"{"split": {"horizontal": true, "parts": [
        [{"leaf": {"slot": "editor", "focus": "editor"}}, {"percent": 60}],
        [{"leaf": {"slot": "chat", "focus": "chat"}}, {"percent": 40}]
    ]}}"#;

    fn templates(entries: &[(&str, &str)]) -> Templates {
        entries
            .iter()
            .map(|(name, json)| {
                (
                    name.to_string(),
                    serde_json::from_str(json).expect("test template is valid JSON"),
                )
            })
            .collect()
    }

    fn area() -> Rect {
        Rect::new(0, 1, 100, 22)
    }

    #[test]
    fn a_template_from_config_builds_without_code() {
        // The done-when in miniature: this arrangement arrives as a config
        // value, and nothing in this module knew its shape beforehand.
        let declared = templates(&[("editor-first", EDITOR_FIRST)]);
        let node = tree(
            &declared,
            "editor-first",
            area(),
            false,
            FocusArea::ChatInput,
        );
        let leaves = render(&node, area());
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].0.slot, BodySlot::Editor);
        assert_eq!(leaves[1].0.slot, BodySlot::Chat);
        assert_eq!(leaves[0].1.width, 60);
        assert_eq!(leaves[1].1.width, 40);
        // Nothing has changed for a config that declares nothing.
        assert_eq!(
            tree(
                &Templates::new(),
                "editor-first",
                area(),
                false,
                FocusArea::ChatInput
            ),
            crate::view::classic_tree(area(), false)
        );
    }

    #[test]
    fn unknown_slots_and_foci_are_refused_with_reasons() {
        let bad_slot = templates(&[("bad", r#"{"leaf": {"slot": "sidebar", "focus": "editor"}}"#)]);
        let why = problem(&bad_slot, "bad").expect("refused");
        assert!(why.contains("unknown slot"), "{why}");
        assert!(why.contains("rendering classic"), "{why}");
        let bad_focus = templates(&[(
            "bad",
            r#"{"leaf": {"slot": "editor", "focus": "terminal"}}"#,
        )]);
        let why = problem(&bad_focus, "bad").expect("refused");
        assert!(why.contains("unknown focus"), "{why}");
    }

    #[test]
    fn degenerate_shapes_are_refused_at_load() {
        // A split of one, a split of none, a zero share, a shape that is not a
        // tree: each would draw something the user did not ask for, so each is
        // refused by name and renders classic instead of guessing.
        for (name, json) in [
            (
                "one-child",
                r#"{"split": {"horizontal": true, "parts": [[{"leaf": {"slot": "chat", "focus": "chat"}}, {"percent": 100}]]}}"#,
            ),
            (
                "no-children",
                r#"{"split": {"horizontal": true, "parts": []}}"#,
            ),
            (
                "zero-share",
                r#"{"split": {"horizontal": true, "parts": [[{"leaf": {"slot": "editor", "focus": "editor"}}, {"percent": 0}], [{"leaf": {"slot": "chat", "focus": "chat"}}, {"percent": 100}]]}}"#,
            ),
            ("not-a-tree", r#"{"grid": {"cells": 4}}"#),
            ("a-string", r#""editor""#),
        ] {
            let declared = templates(&[(name, json)]);
            assert!(problem(&declared, name).is_some(), "{name} must refuse");
            assert_eq!(
                tree(&declared, name, area(), false, FocusArea::ChatInput),
                crate::view::classic_tree(area(), false),
                "{name} must render classic rather than guess"
            );
        }
    }

    #[test]
    fn shapes_a_newer_build_invents_are_refused_not_fatal() {
        // The config carries templates as raw JSON so a name this build cannot
        // read is a refused name, not a lost configuration file. This is what
        // that promise looks like from here — and it names the reason.
        let declared = templates(&[("future", r#"{"zones": [{"kind": "editor"}]}"#)]);
        let why = problem(&declared, "future").expect("refused");
        assert!(why.contains("not a layout tree"), "{why}");
    }

    #[test]
    fn presets_and_declared_templates_are_one_list_of_names() {
        let declared = templates(&[
            ("editor-first", EDITOR_FIRST),
            ("aquarium", EDITOR_FIRST),
            ("classic", EDITOR_FIRST),
        ]);
        // Presets in cycle order first, then the declared names sorted — and a
        // template that reuses a preset's name is not listed twice.
        assert_eq!(
            names(&declared),
            vec![
                "classic".to_string(),
                "chat-first".to_string(),
                "zen".to_string(),
                "aquarium".to_string(),
                "editor-first".to_string(),
            ]
        );
        // The preset wins, and says so.
        let why = problem(&declared, "classic").expect("shadowed template");
        assert!(why.contains("shipped classic preset"), "{why}");
        assert!(is_declared(&declared, "editor-first"));
        assert!(!is_declared(&declared, "zen"));
        assert!(!is_declared(&declared, "typo"));
    }

    #[test]
    fn cycling_covers_declared_templates_and_wraps() {
        let declared = templates(&[("editor-first", EDITOR_FIRST)]);
        // The three presets cycle exactly as they always have.
        assert_eq!(cycle_name(&declared, "classic", true), "chat-first");
        assert_eq!(cycle_name(&declared, "chat-first", true), "zen");
        // Then the declared template is one more stop, and the cycle wraps.
        assert_eq!(cycle_name(&declared, "zen", true), "editor-first");
        assert_eq!(cycle_name(&declared, "editor-first", true), "classic");
        assert_eq!(cycle_name(&declared, "classic", false), "editor-first");
        // An unknown current name starts from the top rather than wedging.
        assert_eq!(cycle_name(&declared, "typo", true), "chat-first");
        assert_eq!(cycle_name(&Templates::new(), "classic", false), "zen");
    }

    #[test]
    fn an_unknown_name_falls_back_to_classic_and_says_so() {
        let declared = templates(&[("editor-first", EDITOR_FIRST)]);
        assert_eq!(effective_name(&declared, "typo"), "classic");
        let why = problem(&declared, "typo").expect("unknown");
        assert!(why.contains("unknown layout"), "{why}");
        // The names it offers are the ones that exist, presets and templates
        // alike — the sentence is usable as written.
        assert!(why.contains("editor-first"), "{why}");
        // A name that renders as asked is reported as itself.
        assert_eq!(effective_name(&declared, "editor-first"), "editor-first");
        assert_eq!(effective_name(&declared, "zen"), "zen");
        assert!(problem(&declared, "editor-first").is_none());
        assert!(problem(&declared, "zen").is_none());
        // A template that refuses does not get to label the pane it did not
        // draw: the header says classic, and `problem` carries the story.
        let broken = templates(&[(
            "broken",
            r#"{"leaf": {"slot": "sidebar", "focus": "editor"}}"#,
        )]);
        assert_eq!(effective_name(&broken, "broken"), "classic");
        assert!(problem(&broken, "broken").is_some());
    }

    #[test]
    fn the_preset_twin_renders_what_the_preset_builds() {
        // "The three shipped presets are templates": two of them have a data
        // twin, and this is the check rather than the claim. Zen is the honest
        // exception — one pane following the focus is not a shape.
        assert!(preset_template("zen").is_none());
        let sizes = [
            Rect::new(0, 1, 80, 22),
            Rect::new(0, 1, 60, 19),
            Rect::new(0, 1, 40, 9),
            Rect::new(0, 1, 20, 4),
            Rect::new(0, 1, 3, 1),
            Rect::new(0, 1, 1, 1),
        ];
        for preset in ["classic", "chat-first"] {
            let twin = preset_template(preset).expect("twin");
            let node = template_to_node(&twin).expect("the twin is a tree");
            for area in sizes {
                for focus in [FocusArea::ChatInput, FocusArea::FileExplorer] {
                    for show_terminal in [false, true] {
                        let from_data = to_body_layout(&render(&node, area));
                        let from_preset =
                            crate::layout::compute_layout(area, preset, show_terminal, focus);
                        // A hidden terminal is a session flag, not a shape: the
                        // twin describes the preset with it shown. Where the
                        // preset itself leaves the terminal out for a short
                        // area, the two must still agree — which is what
                        // `render`'s drop rule is for.
                        let short = area.height < crate::view::TERMINAL_MIN_HEIGHT;
                        if show_terminal || short {
                            assert_eq!(
                                from_data, from_preset,
                                "{preset} t={show_terminal} f={focus:?} {area:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}
