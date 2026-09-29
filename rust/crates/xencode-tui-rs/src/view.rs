//! Layout tree and views (`V-1`), alongside — not instead of — presets.
//!
//! `compute_layout` in [`crate::layout`] stays the shipped geometry path. This
//! module introduces the tree that will eventually replace the preset match:
//! `LayoutNode` (`Leaf`, `Split`, `Tabbed`, `Stack`) plus `Pane`, rendered by
//! one function every surface shares. The three presets are re-expressed as
//! builder functions, and a sweep test proves the tree renders them
//! pixel-identically to `compute_layout` — until that proof exists for every
//! consumer, both paths live and the flag (still presets) chooses.
//!
//! Three rules from the plan, enforced here:
//!
//! - One geometry source: [`render`] feeds drawing, [`hit_test_tree`] feeds
//!   the mouse, so the two cannot drift. Tabbed and Stack resolve to the
//!   active child, and the tests cover those node kinds, not just widths.
//! - [`FocusArea`](crate::focus::FocusArea) stays the focus model. A pane
//!   carries one; the tree never invents a second focus concept.
//! - `Tabbed` and `Stack` resolve the active pane to the whole area. The
//!   visual tab bar is V-5's chrome, not geometry, so it lives there.

use crate::focus::FocusArea;
use crate::layout::BodyLayout;
use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Which body slot a pane fills. Mirrors [`BodyLayout`]'s fields so a rendered
/// tree folds back into one without loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodySlot {
    Explorer,
    Editor,
    Chat,
    Input,
    Terminal,
}

/// One surface: what it shows and what focus it carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pane {
    /// The body slot this pane fills.
    pub slot: BodySlot,
    /// Focus model, unchanged from presets: the tree carries focus areas, it
    /// does not replace them.
    pub focus: FocusArea,
}

/// A layout: leaves, splits, or one-visible-child groups.
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutNode {
    /// One pane filling its whole area.
    Leaf(Pane),
    /// Children side by side (`horizontal`) or stacked, sized by constraints.
    Split {
        horizontal: bool,
        parts: Vec<(LayoutNode, Constraint)>,
    },
    /// Several panes sharing one area, one visible. Geometry resolves the
    /// active child to the whole area; the tab chrome that names the rest is
    /// V-5's work, not this function's.
    Tabbed { panes: Vec<Pane>, active: usize },
    /// Like `Tabbed` without the chrome: one visible child, no selector row.
    Stack { panes: Vec<Pane>, active: usize },
}

/// A named arrangement: the root plus nothing else yet. V-6 names views and
/// V-9 persists them; both extend this struct rather than replacing it.
#[derive(Debug, Clone)]
pub struct ViewState {
    /// The arrangement.
    pub root: LayoutNode,
}

impl ViewState {
    /// A view showing `root`.
    pub fn new(root: LayoutNode) -> Self {
        Self { root }
    }

    /// Replace the arrangement. Pane state lives in `App`, not here, so
    /// swapping trees never loses an open file, a scroll, or a message.
    pub fn set_root(&mut self, root: LayoutNode) {
        self.root = root;
    }

    /// Leaf panes with their rects, in render order.
    pub fn render(&self, area: Rect) -> Vec<(Pane, Rect)> {
        render(&self.root, area)
    }
}

/// Leaf panes with their rects, in render order. Splits use the same ratatui
/// primitives as `compute_layout`, so identical inputs give identical rects.
pub fn render(node: &LayoutNode, area: Rect) -> Vec<(Pane, Rect)> {
    match node {
        LayoutNode::Leaf(pane) => vec![(*pane, area)],
        LayoutNode::Split { horizontal, parts } => {
            let direction = if *horizontal {
                Direction::Horizontal
            } else {
                Direction::Vertical
            };
            let constraints: Vec<Constraint> = parts.iter().map(|(_, c)| *c).collect();
            let areas = Layout::default()
                .direction(direction)
                .constraints(constraints)
                .split(area);
            let mut out = Vec::new();
            for ((child, _), rect) in parts.iter().zip(areas.iter()) {
                out.extend(render(child, *rect));
            }
            out
        }
        LayoutNode::Tabbed { panes, active } | LayoutNode::Stack { panes, active } => {
            if panes.is_empty() {
                Vec::new()
            } else {
                // Out-of-range actives clamp to the last pane rather than
                // panic: an index from a removed tab must not crash the frame.
                let index = (*active).min(panes.len() - 1);
                vec![(panes[index], area)]
            }
        }
    }
}

/// Fold rendered leaves back into a [`BodyLayout`]. Preset trees hold each
/// slot at most once; a hand-built tree repeating a slot keeps the first, and
/// says so here rather than in a comment somewhere downstream.
pub fn to_body_layout(leaves: &[(Pane, Rect)]) -> BodyLayout {
    let mut layout = BodyLayout::default();
    for (pane, rect) in leaves {
        let slot = match pane.slot {
            BodySlot::Explorer => &mut layout.explorer,
            BodySlot::Editor => &mut layout.editor,
            BodySlot::Chat => &mut layout.chat,
            BodySlot::Input => &mut layout.input,
            BodySlot::Terminal => &mut layout.terminal,
        };
        if slot.is_none() {
            *slot = Some(*rect);
        }
    }
    layout
}

/// Which body panel owns terminal column `column`, walking the tree.
///
/// Same contract as [`BodyLayout::hit_test`](crate::layout::BodyLayout::hit_test):
/// explorer, editor and chat regions left to right, clicks beyond the last
/// visible region falling to the nearest visible neighbour. Only the active
/// child of a `Tabbed` or `Stack` participates, which is what makes switching
/// the active tab move the clicks with it.
pub fn hit_test_tree(node: &LayoutNode, area: Rect, column: u16) -> Option<FocusArea> {
    let mut visible: Vec<(Rect, FocusArea)> = Vec::new();
    for (pane, rect) in render(node, area) {
        match pane.slot {
            BodySlot::Explorer | BodySlot::Editor | BodySlot::Chat => {
                visible.push((rect, pane.focus));
            }
            BodySlot::Input | BodySlot::Terminal => {}
        }
    }
    let mut last: Option<FocusArea> = None;
    for (rect, focus) in visible {
        if column < rect.right() {
            return Some(focus);
        }
        last = Some(focus);
    }
    last
}

fn leaf(slot: BodySlot, focus: FocusArea) -> LayoutNode {
    LayoutNode::Leaf(Pane { slot, focus })
}

/// The chat column's vertical stack, shared by every preset that shows chat.
/// Same splits as `split_chat_column`, so the tree and the preset cannot drift:
/// chat, optional terminal, input.
fn chat_column(show_terminal: bool, tall_enough: bool) -> LayoutNode {
    if show_terminal && tall_enough {
        LayoutNode::Split {
            horizontal: false,
            parts: vec![
                (
                    leaf(BodySlot::Chat, FocusArea::ChatInput),
                    Constraint::Min(6),
                ),
                (
                    leaf(BodySlot::Terminal, FocusArea::ChatInput),
                    Constraint::Length(8),
                ),
                (
                    leaf(BodySlot::Input, FocusArea::ChatInput),
                    Constraint::Length(3),
                ),
            ],
        }
    } else {
        LayoutNode::Split {
            horizontal: false,
            parts: vec![
                (
                    leaf(BodySlot::Chat, FocusArea::ChatInput),
                    Constraint::Min(1),
                ),
                (
                    leaf(BodySlot::Input, FocusArea::ChatInput),
                    Constraint::Length(3),
                ),
            ],
        }
    }
}

/// The `classic` preset as a tree: 20/50/30 columns, chat column on the right.
pub fn classic_tree(area: Rect, show_terminal: bool) -> LayoutNode {
    LayoutNode::Split {
        horizontal: true,
        parts: vec![
            (
                leaf(BodySlot::Explorer, FocusArea::FileExplorer),
                Constraint::Percentage(20),
            ),
            (
                leaf(BodySlot::Editor, FocusArea::CodeEditor),
                Constraint::Percentage(50),
            ),
            (
                chat_column(show_terminal, area.height >= 18),
                Constraint::Percentage(30),
            ),
        ],
    }
}

/// The `chat-first` preset as a tree: editor left, chat column right.
pub fn chat_first_tree(area: Rect, show_terminal: bool) -> LayoutNode {
    LayoutNode::Split {
        horizontal: true,
        parts: vec![
            (
                leaf(BodySlot::Editor, FocusArea::CodeEditor),
                Constraint::Percentage(25),
            ),
            (
                chat_column(show_terminal, area.height >= 18),
                Constraint::Percentage(75),
            ),
        ],
    }
}

/// The `zen` preset as a tree: one pane fills the body.
pub fn zen_tree(area: Rect, show_terminal: bool, body_focus: FocusArea) -> LayoutNode {
    match body_focus {
        FocusArea::FileExplorer => leaf(BodySlot::Explorer, FocusArea::FileExplorer),
        FocusArea::CodeEditor => leaf(BodySlot::Editor, FocusArea::CodeEditor),
        _ => chat_column(show_terminal, area.height >= 18),
    }
}

/// Any preset as a tree. Unknown names fall back to classic, the same contract
/// as `effective_layout`.
pub fn preset_tree(
    area: Rect,
    preset: &str,
    show_terminal: bool,
    body_focus: FocusArea,
) -> LayoutNode {
    match preset {
        "chat-first" => chat_first_tree(area, show_terminal),
        "zen" => zen_tree(area, show_terminal, body_focus),
        _ => classic_tree(area, show_terminal),
    }
}

/// Semantic engineering surfaces. Names for what a pane *is for*, not where
/// it sits: the tree decides geometry, this decides meaning. The agent event
/// flow (V-11's descendant) will address panes by these kinds — an
/// `AgentStarted` finds `Agents`, a `VerificationFailed` finds `Verify` — so
/// the vocabulary lands before the wiring that will use it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaneKind {
    Code,
    Agents,
    Review,
    Verify,
    Git,
    Debug,
    Research,
    Terminal,
    Monitor,
}

impl PaneKind {
    /// Short human name for tab strips and overlay headers.
    pub fn title(self) -> &'static str {
        match self {
            Self::Code => "Code",
            Self::Agents => "Agents",
            Self::Review => "Review",
            Self::Verify => "Verify",
            Self::Git => "Git",
            Self::Debug => "Debug",
            Self::Research => "Research",
            Self::Terminal => "Terminal",
            Self::Monitor => "Monitor",
        }
    }

    /// What the pane is for, in one line. The overlay shows it so a kind is
    /// never a mystery abbreviation.
    pub fn hint(self) -> &'static str {
        match self {
            Self::Code => "editor and explorer",
            Self::Agents => "spawned subagents and their state",
            Self::Review => "diffs awaiting a decision",
            Self::Verify => "checks, evidence, and verdicts",
            Self::Git => "branches, worktrees, and commits",
            Self::Debug => "failing tests and their output",
            Self::Research => "retrieved context and sources",
            Self::Terminal => "the embedded shell",
            Self::Monitor => "cost, health, and background tasks",
        }
    }
}

/// One agent-stack pane: a kind, a title, and its state rows. Built from App
/// state the loop already holds — no new event flow, no new subscription.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentPane {
    /// What this pane is.
    pub kind: PaneKind,
    /// Header line, e.g. `Subagents (2)`.
    pub title: String,
    /// State rows, visible without input. Never empty: an idle pane says so.
    pub rows: Vec<String>,
}

/// The three agent panes, always three, from state the app already has.
///
/// `spawns` are `(branch, state line)` per subagent, `bytebot` are
/// `(step, status)` pairs, `approvals` are one line per queued request. Empty
/// sources yield idle rows rather than missing panes, so the stack shape is
/// stable and switching never lands on nothing.
pub fn agent_panes(
    spawns: &[(String, String)],
    bytebot: &[(String, String)],
    approvals: &[String],
) -> Vec<AgentPane> {
    let mut subagents: Vec<String> = spawns
        .iter()
        .map(|(branch, state)| format!("{branch} — {state}"))
        .collect();
    if subagents.is_empty() {
        subagents.push("idle — try `/spawn <task>`".to_string());
    }
    let mut steps: Vec<String> = bytebot
        .iter()
        .map(|(step, status)| format!("{step}: {status}"))
        .collect();
    if steps.is_empty() {
        steps.push("idle — no ByteBot run".to_string());
    }
    let mut pending: Vec<String> = approvals.to_vec();
    if pending.is_empty() {
        pending.push("none pending".to_string());
    }
    vec![
        AgentPane {
            kind: PaneKind::Agents,
            title: format!("Subagents ({})", spawns.len()),
            rows: subagents,
        },
        AgentPane {
            kind: PaneKind::Agents,
            title: format!("ByteBot ({})", bytebot.len()),
            rows: steps,
        },
        AgentPane {
            kind: PaneKind::Agents,
            title: format!("Approvals ({})", approvals.len()),
            rows: pending,
        },
    ]
}

/// Advance the first `Stack` or `Tabbed` node found depth-first. Returns
/// whether anything moved: no stack means no-op, never an error.
pub fn cycle_active(node: &mut LayoutNode) -> bool {
    match node {
        LayoutNode::Leaf(_) => false,
        LayoutNode::Split { parts, .. } => parts.iter_mut().any(|(child, _)| cycle_active(child)),
        LayoutNode::Tabbed { panes, active } | LayoutNode::Stack { panes, active } => {
            if panes.is_empty() {
                false
            } else {
                *active = (*active + 1) % panes.len();
                true
            }
        }
    }
}

/// The active index of the first `Stack` or `Tabbed` node, depth-first.
/// `None` when the tree holds no stack — readers use this rather than tracking
/// a parallel index that could disagree with the tree.
pub fn stack_active(node: &LayoutNode) -> Option<usize> {
    match node {
        LayoutNode::Leaf(_) => None,
        LayoutNode::Split { parts, .. } => parts.iter().find_map(|(child, _)| stack_active(child)),
        LayoutNode::Tabbed { panes, active } | LayoutNode::Stack { panes, active } => {
            if panes.is_empty() {
                None
            } else {
                Some((*active).min(panes.len() - 1))
            }
        }
    }
}

/// Text for the stack overlay: tab strip plus the active pane's rows. Pure so
/// the content is testable without a terminal; the draw code only lays it out.
pub fn stack_overlay_text(panes: &[AgentPane], active: usize) -> Vec<String> {
    if panes.is_empty() {
        return vec!["no agent panes".to_string()];
    }
    let active = active.min(panes.len() - 1);
    let mut out = Vec::new();
    let strip: Vec<String> = panes
        .iter()
        .enumerate()
        .map(|(i, pane)| {
            if i == active {
                format!("[{}]", pane.title)
            } else {
                pane.title.clone()
            }
        })
        .collect();
    out.push(strip.join("  "));
    out.push(format!(
        "{} — {}",
        panes[active].kind.title(),
        panes[active].kind.hint()
    ));
    out.extend(panes[active].rows.iter().cloned());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{compute_layout, LAYOUT_NAMES};

    fn rect(w: u16, h: u16) -> Rect {
        Rect::new(0, 1, w, h)
    }

    fn areas() -> Vec<Rect> {
        vec![
            rect(80, 22),
            rect(60, 19),
            rect(40, 9),
            rect(20, 4),
            rect(3, 1),
            rect(1, 1),
        ]
    }

    fn focuses() -> Vec<FocusArea> {
        vec![
            FocusArea::ChatInput,
            FocusArea::FileExplorer,
            FocusArea::CodeEditor,
        ]
    }

    #[test]
    fn the_tree_renders_every_preset_pixel_identically() {
        // The disposition's proof: until this holds for every consumer, both
        // paths live and presets choose. If it ever fails, the tree is wrong,
        // never the shipped path.
        for area in areas() {
            for preset in LAYOUT_NAMES {
                for focus in focuses() {
                    for show_terminal in [false, true] {
                        let tree = preset_tree(area, preset, show_terminal, focus);
                        let from_tree = to_body_layout(&render(&tree, area));
                        let from_preset = compute_layout(area, preset, show_terminal, focus);
                        assert_eq!(
                            from_tree, from_preset,
                            "{preset} t={show_terminal} f={focus:?} {area:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tree_hit_testing_agrees_with_presets_everywhere() {
        for area in areas() {
            for preset in LAYOUT_NAMES {
                for focus in focuses() {
                    for show_terminal in [false, true] {
                        let tree = preset_tree(area, preset, show_terminal, focus);
                        let from_preset = compute_layout(area, preset, show_terminal, focus);
                        let mut column = 0;
                        while column < area.width.saturating_add(2) {
                            assert_eq!(
                                hit_test_tree(&tree, area, column),
                                from_preset.hit_test(column),
                                "{preset} col={column} {area:?}"
                            );
                            column += 1;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tabbed_and_stack_resolve_the_active_child() {
        let area = rect(80, 22);
        let panes = vec![
            Pane {
                slot: BodySlot::Editor,
                focus: FocusArea::CodeEditor,
            },
            Pane {
                slot: BodySlot::Chat,
                focus: FocusArea::ChatInput,
            },
        ];
        // Tabbed and Stack resolve identically by geometry; the difference is
        // chrome V-5 owns, so both shapes run the same assertions.
        for tabbed in [true, false] {
            let make = |active: usize| {
                if tabbed {
                    LayoutNode::Tabbed {
                        panes: panes.clone(),
                        active,
                    }
                } else {
                    LayoutNode::Stack {
                        panes: panes.clone(),
                        active,
                    }
                }
            };
            // Active editor: full area, editor focus.
            let node = make(0);
            assert_eq!(render(&node, area), vec![(panes[0], area)]);
            assert_eq!(hit_test_tree(&node, area, 0), Some(FocusArea::CodeEditor));
            // Switching moves both the rect owner and the clicks.
            let node = make(1);
            assert_eq!(render(&node, area), vec![(panes[1], area)]);
            assert_eq!(hit_test_tree(&node, area, 79), Some(FocusArea::ChatInput));
            // Out-of-range actives clamp instead of panicking.
            let node = make(99);
            assert_eq!(render(&node, area), vec![(panes[1], area)]);
        }
        // Empty groups render nothing and resolve nothing.
        let empty = LayoutNode::Tabbed {
            panes: Vec::new(),
            active: 0,
        };
        assert!(render(&empty, area).is_empty());
        assert_eq!(hit_test_tree(&empty, area, 0), None);
    }

    #[test]
    fn views_swap_trees_without_losing_anything() {
        let area = rect(80, 22);
        let mut view = ViewState::new(classic_tree(area, false));
        assert_eq!(view.render(area).len(), 4);
        view.set_root(zen_tree(area, false, FocusArea::CodeEditor));
        assert_eq!(view.render(area).len(), 1);
    }

    #[test]
    fn pane_kinds_name_themselves() {
        assert_eq!(PaneKind::Agents.title(), "Agents");
        assert!(PaneKind::Verify.hint().contains("verdict"));
        // Nine kinds, each distinct, so a kind is never an alias by accident.
        let mut titles = std::collections::BTreeSet::new();
        for kind in [
            PaneKind::Code,
            PaneKind::Agents,
            PaneKind::Review,
            PaneKind::Verify,
            PaneKind::Git,
            PaneKind::Debug,
            PaneKind::Research,
            PaneKind::Terminal,
            PaneKind::Monitor,
        ] {
            assert!(titles.insert(kind.title()), "duplicate title");
        }
    }

    #[test]
    fn three_agent_panes_always_three() {
        let panes = agent_panes(&[], &[], &[]);
        assert_eq!(panes.len(), 3);
        for pane in &panes {
            assert!(!pane.rows.is_empty(), "an idle pane says so");
        }
        let panes = agent_panes(
            &[("feat".to_string(), "1/2 call(s) completed".to_string())],
            &[("fetch".to_string(), "done".to_string())],
            &["edit_file: main.rs".to_string()],
        );
        assert_eq!(panes[0].title, "Subagents (1)");
        assert!(panes[0].rows[0].contains("feat"));
        assert!(panes[1].rows[0].contains("fetch: done"));
        assert!(panes[2].rows[0].contains("edit_file"));
    }

    #[test]
    fn cycling_moves_the_first_stack_depth_first() {
        let mut tree = LayoutNode::Split {
            horizontal: true,
            parts: vec![
                (
                    leaf(BodySlot::Editor, FocusArea::CodeEditor),
                    Constraint::Percentage(50),
                ),
                (
                    LayoutNode::Stack {
                        panes: vec![
                            Pane {
                                slot: BodySlot::Chat,
                                focus: FocusArea::ChatInput,
                            },
                            Pane {
                                slot: BodySlot::Terminal,
                                focus: FocusArea::ChatInput,
                            },
                        ],
                        active: 0,
                    },
                    Constraint::Percentage(50),
                ),
            ],
        };
        assert!(cycle_active(&mut tree));
        assert_eq!(stack_active(&tree), Some(1));
        assert!(cycle_active(&mut tree));
        assert_eq!(stack_active(&tree), Some(0));
        // No stack anywhere: no-op, never an error.
        let mut plain = leaf(BodySlot::Editor, FocusArea::CodeEditor);
        assert!(!cycle_active(&mut plain));
    }

    #[test]
    fn the_overlay_marks_the_active_pane() {
        let panes = agent_panes(&[], &[("fetch".to_string(), "done".to_string())], &[]);
        let text = stack_overlay_text(&panes, 1);
        assert!(text[0].contains("[ByteBot (1)]"), "{text:?}");
        assert!(text.iter().any(|l| l.contains("fetch: done")), "{text:?}");
        assert!(stack_overlay_text(&[], 0) == vec!["no agent panes".to_string()]);
    }

    #[test]
    fn unknown_preset_names_fall_back_to_classic() {
        let area = rect(80, 22);
        let tree = preset_tree(area, "typo", false, FocusArea::ChatInput);
        let from_tree = to_body_layout(&render(&tree, area));
        assert_eq!(
            from_tree,
            compute_layout(area, "classic", false, FocusArea::ChatInput)
        );
    }
}
