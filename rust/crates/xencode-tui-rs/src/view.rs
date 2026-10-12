//! The layout tree (`V-1`): `LayoutNode` (`Leaf`, `Split`, `Tabbed`, `Stack`)
//! plus `Pane`, rendered by one function every surface shares. The Files view
//! builds its one fixed tree from these (`App::body_tree`).
//!
//! Two rules, enforced here:
//!
//! - One geometry source: [`render`] feeds drawing, [`hit_test_tree`] (and its
//!   cell-wise sibling [`hit_test_tree_point`]) feeds the mouse, so the two
//!   cannot drift. Tabbed and Stack resolve to the active child.
//! - [`FocusArea`](crate::focus::FocusArea) stays the focus model. A pane
//!   carries one; the tree never invents a second focus concept.

use crate::focus::FocusArea;
use crate::layout::BodyLayout;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};

/// Which body slot a pane fills. Mirrors [`BodyLayout`]'s fields so a rendered
/// tree folds back into one without loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodySlot {
    Explorer,
    Editor,
    Chat,
    Input,
    Terminal,
    Agents,
}

/// One surface: what it shows and what focus it carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pane {
    /// The body slot this pane fills.
    pub slot: BodySlot,
    /// The focus area a click on this pane gives: the tree carries focus
    /// areas, it does not replace them.
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
    /// active child to the whole area.
    Tabbed { panes: Vec<Pane>, active: usize },
    /// Like `Tabbed` without the chrome: one visible child, no selector row.
    Stack { panes: Vec<Pane>, active: usize },
}

/// Leaf panes with their rects, in render order. A stacked split shorter than
/// [`TERMINAL_MIN_HEIGHT`] drops its terminal children *before* the split is
/// computed, so their rows go to the pane above instead of squeezing it to
/// nothing.
pub fn render(node: &LayoutNode, area: Rect) -> Vec<(Pane, Rect)> {
    render_inner(node, area)
}

/// Is this node a leaf holding the embedded terminal?
fn is_terminal_leaf(node: &LayoutNode) -> bool {
    matches!(
        node,
        LayoutNode::Leaf(Pane {
            slot: BodySlot::Terminal,
            ..
        })
    )
}

fn render_inner(node: &LayoutNode, area: Rect) -> Vec<(Pane, Rect)> {
    match node {
        LayoutNode::Leaf(pane) => vec![(*pane, area)],
        LayoutNode::Split { horizontal, parts } => {
            let mut out = Vec::new();
            for (child, rect) in shown_children(*horizontal, parts, area) {
                out.extend(render_inner(child, rect));
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

/// Fold rendered leaves back into a [`BodyLayout`]. A tree that repeats a slot
/// keeps the first.
pub fn to_body_layout(leaves: &[(Pane, Rect)]) -> BodyLayout {
    let mut layout = BodyLayout::default();
    for (pane, rect) in leaves {
        let slot = match pane.slot {
            BodySlot::Explorer => &mut layout.explorer,
            BodySlot::Editor => &mut layout.editor,
            BodySlot::Agents => &mut layout.agents,
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
/// Explorer, editor and chat regions left to right, clicks beyond the last
/// visible region falling to the nearest visible neighbour. Only the active
/// child of a `Tabbed` or `Stack` participates, which is what makes switching
/// the active tab move the clicks with it.
pub fn hit_test_tree(node: &LayoutNode, area: Rect, column: u16) -> Option<FocusArea> {
    let mut visible: Vec<(Rect, FocusArea)> = Vec::new();
    for (pane, rect) in render(node, area) {
        match pane.slot {
            BodySlot::Explorer | BodySlot::Editor | BodySlot::Agents | BodySlot::Chat => {
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

/// Which body panel owns the cell at (`row`, `column`), walking the tree.
///
/// [`hit_test_tree`] answers a column-only version of this question, which is
/// the right question for a side-by-side split. Where panes are stacked a
/// column is not enough: the pane whose rect contains the cell wins. A cell that lands on no
/// pane — the input strip, the terminal, a hidden tab — falls back to the
/// column rule, so a click keeps landing where it always did.
pub fn hit_test_tree_point(
    node: &LayoutNode,
    area: Rect,
    row: u16,
    column: u16,
) -> Option<FocusArea> {
    for (pane, rect) in render(node, area) {
        let claims = matches!(
            pane.slot,
            BodySlot::Explorer | BodySlot::Editor | BodySlot::Agents | BodySlot::Chat
        );
        if claims && rect.contains(Position { x: column, y: row }) {
            return Some(pane.focus);
        }
    }
    hit_test_tree(node, area, column)
}

/// One pane leaf.
pub fn leaf(slot: BodySlot, focus: FocusArea) -> LayoutNode {
    LayoutNode::Leaf(Pane { slot, focus })
}

/// Below this many rows, a terminal pane would squeeze its column to nothing
/// useful, so the renderer drops it here, not in the draw code.
pub const TERMINAL_MIN_HEIGHT: u16 = 18;

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
    /// State rows, visible without input. Never empty: an empty source says so.
    pub rows: Vec<String>,
}

/// The three agent panes, always three, from state the app already has.
///
/// `spawns` are `(branch, state line)` per subagent, `bytebot` are
/// `(step, status)` pairs, `approvals` are one line per queued request. Empty
/// sources yield a row saying what is absent rather than missing panes, so the
/// stack shape is stable and switching never lands on nothing. An empty source
/// is not an idle worker: `idle` would claim something exists that is merely
/// not busy, when the honest reading is that xencode spawned nothing (`OR-12`).
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
        subagents.push("nothing spawned — try `/spawn <task>`".to_string());
    }
    let mut steps: Vec<String> = bytebot
        .iter()
        .map(|(step, status)| format!("{step}: {status}"))
        .collect();
    if steps.is_empty() {
        steps.push("no ByteBot run recorded".to_string());
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

/// The children a split actually draws, with their rects.
///
/// Drawing and hit-testing both come through here, so a click lands on the
/// pane the frame drew.
fn shown_children<'a>(
    horizontal: bool,
    parts: &'a [(LayoutNode, Constraint)],
    area: Rect,
) -> Vec<(&'a LayoutNode, Rect)> {
    let shown: Vec<&'a (LayoutNode, Constraint)> = parts
        .iter()
        .filter(|(child, _)| {
            horizontal || area.height >= TERMINAL_MIN_HEIGHT || !is_terminal_leaf(child)
        })
        .collect();
    let areas = Layout::default()
        .direction(if horizontal {
            Direction::Horizontal
        } else {
            Direction::Vertical
        })
        .constraints(shown.iter().map(|(_, c)| *c).collect::<Vec<_>>())
        .split(area);
    shown
        .iter()
        .map(|pair| &pair.0)
        .zip(areas.iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: u16, h: u16) -> Rect {
        Rect::new(0, 1, w, h)
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
        // Tabbed and Stack resolve identically by geometry, so both shapes run
        // the same assertions.
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
            assert!(!pane.rows.is_empty(), "an empty pane says so");
            // Nothing was spawned, so nothing can be reported as merely idle.
            for row in &pane.rows {
                assert!(
                    !row.contains("idle"),
                    "{row} claims a worker that isn't there"
                );
            }
        }
        assert_eq!(panes[0].rows[0], "nothing spawned — try `/spawn <task>`");
        assert_eq!(panes[1].rows[0], "no ByteBot run recorded");
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
    fn a_stacked_tree_hit_tests_by_point() {
        // Editor above chat: both leaves span the full width, so a column can
        // never tell them apart and only the cell can.
        let area = rect(80, 22);
        let tree = LayoutNode::Split {
            horizontal: false,
            parts: vec![
                (
                    leaf(BodySlot::Editor, FocusArea::CodeEditor),
                    Constraint::Percentage(50),
                ),
                (
                    leaf(BodySlot::Chat, FocusArea::ChatInput),
                    Constraint::Percentage(50),
                ),
            ],
        };
        let leaves = render(&tree, area);
        let editor_rect = leaves[0].1;
        let chat_rect = leaves[1].1;
        assert_eq!(
            hit_test_tree_point(&tree, area, editor_rect.y, 40),
            Some(FocusArea::CodeEditor)
        );
        assert_eq!(
            hit_test_tree_point(&tree, area, chat_rect.y, 40),
            Some(FocusArea::ChatInput)
        );
    }

    #[test]
    fn a_short_stacked_split_drops_the_terminal_it_cannot_fit() {
        // The chat column with the terminal named: at 10 rows the terminal leaf
        // is dropped before layout, so its eight rows stay with the chat pane.
        let tree = LayoutNode::Split {
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
        };
        let short = render(&tree, rect(80, 10));
        assert!(
            !short
                .iter()
                .any(|(pane, _)| pane.slot == BodySlot::Terminal),
            "no terminal at 10 rows: {short:?}"
        );
        let from_tree = to_body_layout(&short);
        assert_eq!(from_tree.chat.map(|r| r.height), Some(7));
        assert_eq!(from_tree.input.map(|r| r.height), Some(3));

        // The same column at 22 rows keeps the terminal.
        let tall = to_body_layout(&render(&tree, rect(80, 22)));
        assert_eq!(tall.terminal.map(|r| r.height), Some(8));
    }
}
