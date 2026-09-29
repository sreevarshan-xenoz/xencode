//! Layout tree and views (`V-1`), alongside — not instead of — presets at
//! first, and since `V-5` the only thing a frame draws.
//!
//! `LayoutNode` (`Leaf`, `Split`, `Tabbed`, `Stack`) plus `Pane`, rendered by
//! one function every surface shares. The three presets are expressed as
//! builder functions, and a sweep proves the tree renders them
//! pixel-identically to `crate::layout::compute_layout` — which is why that
//! function now exists as the *reference* the sweep compares against, rather
//! than as a second path a frame can take. [`crate::templates`] is the one
//! place a layout name resolves: a preset to its builder here, a name the user
//! declared in config to the data constructor there.
//!
//! Three rules from the plan, enforced here:
//!
//! - One geometry source: [`render`] feeds drawing, [`hit_test_tree`] (and its
//!   cell-wise sibling [`hit_test_tree_point`]) feeds the mouse, so the two
//!   cannot drift. Tabbed and Stack resolve to the active child, and the tests
//!   cover those node kinds, not just widths.
//! - [`FocusArea`](crate::focus::FocusArea) stays the focus model. A pane
//!   carries one; the tree never invents a second focus concept.
//! - `Tabbed` and `Stack` resolve the active pane to the whole area. The
//!   visual tab bar is V-5's chrome, not geometry, so it lives there.

use crate::focus::FocusArea;
use crate::layout::BodyLayout;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use serde::{Deserialize, Serialize};

/// Which body slot a pane fills. Mirrors [`BodyLayout`]'s fields so a rendered
/// tree folds back into one without loss.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BodySlot {
    Explorer,
    Editor,
    Chat,
    Input,
    Terminal,
}

/// One surface: what it shows and what focus it carries. Never serialized
/// directly — trees go to disk as [`PaneCode`], whose focus is one of the
/// three words a template speaks (`explorer`, `editor`, `chat`), mapped in
/// [`encode_focus`]/[`decode_focus`] rather than by `FocusArea` itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pane {
    /// The body slot this pane fills.
    pub slot: BodySlot,
    /// Focus model, unchanged from presets: the tree carries focus areas, it
    /// does not replace them.
    pub focus: FocusArea,
}

/// A layout: leaves, splits, or one-visible-child groups.
///
/// Serialized by hand through [`TreeCode`], not derived: a tree must decode
/// through the same validating constructor a config template goes through,
/// and a derived impl cannot refuse a zero share or an unknown slot word.
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

/// A named arrangement: the root plus nothing else yet. `V-6` persists this
/// and `V-4` adds names to it; both extend the struct rather than replacing it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

/// A tree that refuses to be read back, with the sentence why. Mirrors
/// [`crate::templates::TemplateError`]: a stored arrangement that this build
/// cannot understand is reported by name, never guessed at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeError(pub String);

impl std::fmt::Display for TreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// How a split child takes its share — the same three words a config template
/// speaks (`crate::templates::Share`), so the file on disk and the file a user
/// hand-writes are one vocabulary. `Ratio` exists only on the encode side: a
/// tree holding one cannot be saved, and says so, rather than being written
/// with a word no template can state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ShareCode {
    Percent(u16),
    Min(u16),
    Length(u16),
    /// Encode-only: `serde(skip)` makes a stored `ratio` a refusal on read,
    /// which is the correct answer to geometry this vocabulary cannot keep.
    #[serde(skip)]
    Ratio(u32, u32),
}

fn encode_constraint(constraint: Constraint) -> ShareCode {
    match constraint {
        Constraint::Percentage(v) => ShareCode::Percent(v),
        Constraint::Min(v) => ShareCode::Min(v),
        Constraint::Length(v) => ShareCode::Length(v),
        Constraint::Ratio(n, d) => ShareCode::Ratio(n, d),
        // `max` and `fill` are words no template speaks; saving either is
        // inventing geometry a rebuilt tree will not reproduce. Routed
        // through the encode-only `Ratio`, so the save refuses them by name.
        Constraint::Fill(v) => ShareCode::Ratio(v as u32, 0),
        Constraint::Max(v) => ShareCode::Ratio(v as u32, 1),
    }
}

fn decode_constraint(share: ShareCode) -> Result<Constraint, TreeError> {
    match share {
        ShareCode::Percent(v) => Ok(Constraint::Percentage(v)),
        ShareCode::Min(v) => Ok(Constraint::Min(v)),
        ShareCode::Length(v) => Ok(Constraint::Length(v)),
        ShareCode::Ratio(_, _) => Err(TreeError(
            "a split child shares with a word this build does not store".to_string(),
        )),
    }
}

/// The JSON shape of a pane: slot and focus as the words a template uses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PaneCode {
    slot: String,
    focus: String,
}

fn encode_pane(pane: Pane) -> PaneCode {
    PaneCode {
        slot: encode_slot(pane.slot),
        focus: encode_focus(pane.focus),
    }
}

impl Pane {
    fn decode(code: &PaneCode) -> Result<Self, TreeError> {
        Ok(Pane {
            slot: decode_slot(&code.slot)?,
            focus: decode_focus(&code.focus)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SplitCode {
    horizontal: bool,
    parts: Vec<(TreeCode, ShareCode)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct GroupCode {
    panes: Vec<PaneCode>,
    active: usize,
}

/// The on-disk shape of a tree, written by encoding a live one and read back
/// with validation. Tagged in the same lowercase words a config template
/// uses (`crate::templates::LayoutTemplate`), so the file the app writes and
/// the file a user hand-writes are one vocabulary. `Tabbed` and `Stack` are
/// the shapes V-2's stack overlay cycles inside; a resized preset tree never
/// contains them today, and this still stores one honestly if it ever does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum TreeCode {
    Leaf(PaneCode),
    Split(SplitCode),
    Tabbed(GroupCode),
    Stack(GroupCode),
}

fn encode_slot(slot: BodySlot) -> String {
    match slot {
        BodySlot::Explorer => "explorer",
        BodySlot::Editor => "editor",
        BodySlot::Chat => "chat",
        BodySlot::Input => "input",
        BodySlot::Terminal => "terminal",
    }
    .to_string()
}

fn encode_focus(focus: FocusArea) -> String {
    match focus {
        FocusArea::FileExplorer => "explorer",
        FocusArea::CodeEditor => "editor",
        _ => "chat",
    }
    .to_string()
}

fn decode_slot(word: &str) -> Result<BodySlot, TreeError> {
    match word {
        "explorer" => Ok(BodySlot::Explorer),
        "editor" => Ok(BodySlot::Editor),
        "chat" => Ok(BodySlot::Chat),
        "input" => Ok(BodySlot::Input),
        "terminal" => Ok(BodySlot::Terminal),
        other => Err(TreeError(format!(
            "unknown slot {other:?}: want explorer, editor, chat, input, or terminal"
        ))),
    }
}

fn decode_focus(word: &str) -> Result<FocusArea, TreeError> {
    match word {
        "explorer" => Ok(FocusArea::FileExplorer),
        "editor" => Ok(FocusArea::CodeEditor),
        "chat" => Ok(FocusArea::ChatInput),
        other => Err(TreeError(format!(
            "unknown focus {other:?}: want explorer, editor, or chat"
        ))),
    }
}

fn encode_node(node: &LayoutNode) -> TreeCode {
    match node {
        LayoutNode::Leaf(pane) => TreeCode::Leaf(encode_pane(*pane)),
        LayoutNode::Split { horizontal, parts } => TreeCode::Split(SplitCode {
            horizontal: *horizontal,
            parts: parts
                .iter()
                .map(|(child, constraint)| (encode_node(child), encode_constraint(*constraint)))
                .collect(),
        }),
        LayoutNode::Tabbed { panes, active } => TreeCode::Tabbed(GroupCode {
            panes: panes.iter().cloned().map(encode_pane).collect(),
            active: *active,
        }),
        LayoutNode::Stack { panes, active } => TreeCode::Stack(GroupCode {
            panes: panes.iter().cloned().map(encode_pane).collect(),
            active: *active,
        }),
    }
}

fn decode_node(code: &TreeCode) -> Result<LayoutNode, TreeError> {
    match code {
        TreeCode::Leaf(pane) => Ok(LayoutNode::Leaf(Pane::decode(pane)?)),
        TreeCode::Split(split) => {
            if split.parts.len() < 2 {
                return Err(TreeError("a split needs at least two children".to_string()));
            }
            let mut parts = Vec::new();
            for (child, share) in &split.parts {
                if matches!(
                    share,
                    ShareCode::Percent(0) | ShareCode::Min(0) | ShareCode::Length(0)
                ) {
                    return Err(TreeError(
                        "a zero share lays out nothing; remove the child instead".to_string(),
                    ));
                }
                let constraint = decode_constraint(*share)?;
                parts.push((decode_node(child)?, constraint));
            }
            Ok(LayoutNode::Split {
                horizontal: split.horizontal,
                parts,
            })
        }
        TreeCode::Tabbed(group) | TreeCode::Stack(group) => {
            let mut panes = Vec::new();
            for pane in &group.panes {
                panes.push(Pane::decode(pane)?);
            }
            let node = if matches!(code, TreeCode::Tabbed(_)) {
                LayoutNode::Tabbed {
                    panes,
                    active: group.active,
                }
            } else {
                LayoutNode::Stack {
                    panes,
                    active: group.active,
                }
            };
            Ok(node)
        }
    }
}

impl serde::Serialize for LayoutNode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        encode_node(self).serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for LayoutNode {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Same contract as a config template: the shape is read, then
        // validated, and a refusal is a sentence rather than a guess.
        let code = TreeCode::deserialize(deserializer)?;
        decode_node(&code).map_err(serde::de::Error::custom)
    }
}

/// Leaf panes with their rects, in render order. Splits use the same ratatui
/// primitives as `compute_layout`, so identical inputs give identical rects.
/// A stacked split shorter than [`TERMINAL_MIN_HEIGHT`] drops its terminal
/// children *before* the split is computed — the same rule the preset builders
/// apply through `chat_column`, so a resized tree, a template and a preset
/// agree about when the terminal vanishes. Their rows go back to the chat pane
/// above instead of squeezing it to nothing.
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
            let shown: Vec<&(LayoutNode, Constraint)> = parts
                .iter()
                .filter(|(child, _)| {
                    *horizontal || area.height >= TERMINAL_MIN_HEIGHT || !is_terminal_leaf(child)
                })
                .collect();
            let direction = if *horizontal {
                Direction::Horizontal
            } else {
                Direction::Vertical
            };
            let constraints: Vec<Constraint> = shown.iter().map(|(_, c)| *c).collect();
            let areas = Layout::default()
                .direction(direction)
                .constraints(constraints)
                .split(area);
            let mut out = Vec::new();
            for (child, rect) in shown.iter().zip(areas.iter()) {
                out.extend(render_inner(&child.0, *rect));
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

/// Which body panel owns the cell at (`row`, `column`), walking the tree.
///
/// [`hit_test_tree`] answers a column-only version of this question, which is
/// the right question while every body split is side-by-side — the three
/// presets are. A template may split top from bottom, and there a column is not
/// enough: the pane whose rect contains the cell wins. A cell that lands on no
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
            BodySlot::Explorer | BodySlot::Editor | BodySlot::Chat
        );
        if claims && rect.contains(Position { x: column, y: row }) {
            return Some(pane.focus);
        }
    }
    hit_test_tree(node, area, column)
}

fn leaf(slot: BodySlot, focus: FocusArea) -> LayoutNode {
    LayoutNode::Leaf(Pane { slot, focus })
}

/// Below this many rows, a terminal pane would squeeze its column to nothing
/// useful, so the renderer drops it — the same rule as `split_chat_column`,
/// in the same shared place, not in the draw code.
pub const TERMINAL_MIN_HEIGHT: u16 = 18;

/// The chat column's vertical stack, shared by every preset that shows chat.
/// Same splits as `split_chat_column`, so the tree and the preset cannot drift:
/// chat, optional terminal, input. The terminal leaf is always built when
/// asked for; [`render`] drops it in short areas.
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
                chat_column(show_terminal, area.height >= TERMINAL_MIN_HEIGHT),
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
                chat_column(show_terminal, area.height >= TERMINAL_MIN_HEIGHT),
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
        _ => chat_column(show_terminal, area.height >= TERMINAL_MIN_HEIGHT),
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

/// Minimum share of a split any pane keeps, in percentage points. Below this
/// a pane is a sliver its content cannot use, and a resize that produced one
/// would be a collapse wearing an adjustment's clothes.
pub const MIN_PANE_PERCENT: u16 = 10;

/// Grow the branch holding the focused pane by `delta` percentage points,
/// taking from a sibling in the nearest enclosing horizontal split.
///
/// Only `Percentage` constraints move: `Length` and `Min` children are fixed
/// chrome (input rows, terminal height), and resizing fixed chrome is how a
/// three-row input becomes a zero-row one. Both sides clamp at
/// [`MIN_PANE_PERCENT`]; a move that cannot happen returns `false` rather
/// than a partial one.
pub fn nudge_focused(node: &mut LayoutNode, focus: FocusArea, delta: i16) -> bool {
    fn path_to(node: &LayoutNode, focus: FocusArea, path: &mut Vec<usize>) -> bool {
        match node {
            LayoutNode::Leaf(pane) => pane.focus == focus,
            LayoutNode::Split { parts, .. } => {
                for (index, (child, _)) in parts.iter().enumerate() {
                    path.push(index);
                    if path_to(child, focus, path) {
                        return true;
                    }
                    path.pop();
                }
                false
            }
            LayoutNode::Tabbed { .. } | LayoutNode::Stack { .. } => false,
        }
    }

    fn child_at<'a>(node: &'a mut LayoutNode, path: &[usize]) -> Option<&'a mut LayoutNode> {
        let mut current = node;
        for index in path {
            match current {
                LayoutNode::Split { parts, .. } => {
                    current = &mut parts.get_mut(*index)?.0;
                }
                _ => return None,
            }
        }
        Some(current)
    }

    let mut path = Vec::new();
    if !path_to(node, focus, &mut path) {
        return false;
    }
    // Walk from the leaf up to the nearest enclosing horizontal split.
    while let Some(index) = path.pop() {
        let Some(parent) = child_at(node, &path) else {
            return false;
        };
        let LayoutNode::Split { horizontal, parts } = parent else {
            continue;
        };
        if !*horizontal {
            continue;
        }
        let donor = if index + 1 < parts.len() {
            index + 1
        } else {
            index.saturating_sub(1)
        };
        if donor == index || parts.len() < 2 {
            return false;
        }
        let (grow_pct, give_pct) = match (&parts[index].1, &parts[donor].1) {
            (Constraint::Percentage(grow), Constraint::Percentage(give)) => (*grow, *give),
            _ => return false,
        };
        let room = give_pct.saturating_sub(MIN_PANE_PERCENT) as i16;
        let moved = delta.clamp(-(grow_pct as i16 - MIN_PANE_PERCENT as i16), room);
        if moved == 0 {
            return false;
        }
        parts[index].1 = Constraint::Percentage((grow_pct as i16 + moved) as u16);
        parts[donor].1 = Constraint::Percentage((give_pct as i16 - moved) as u16);
        return true;
    }
    false
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

    fn classic_widths(tree: &LayoutNode, area: Rect) -> Vec<u16> {
        render(tree, area)
            .iter()
            .filter(|(pane, _)| {
                pane.slot == BodySlot::Explorer
                    || pane.slot == BodySlot::Editor
                    || pane.slot == BodySlot::Chat
            })
            .map(|(_, rect)| rect.width)
            .collect()
    }

    #[test]
    fn nudging_grows_the_focused_pane_from_its_neighbour() {
        let area = rect(100, 22);
        let mut tree = classic_tree(area, false);
        let before = classic_widths(&tree, area);
        assert!(nudge_focused(&mut tree, FocusArea::CodeEditor, 5));
        let after = classic_widths(&tree, area);
        assert!(after[1] > before[1], "editor grew: {before:?} -> {after:?}");
        // Editor is the middle child: it grows from its next sibling, the
        // chat column, while the explorer is untouched.
        assert_eq!(after[0], before[0], "explorer untouched");
        assert!(
            after[2] < before[2],
            "chat column paid: {before:?} -> {after:?}"
        );
        assert!(nudge_focused(&mut tree, FocusArea::CodeEditor, -5));
        assert_eq!(classic_widths(&tree, area), before, "shrinking restores");
    }

    #[test]
    fn nudging_clamps_at_the_documented_minimum() {
        let area = rect(100, 22);
        let mut tree = classic_tree(area, false);
        for _ in 0..30 {
            nudge_focused(&mut tree, FocusArea::CodeEditor, 5);
        }
        // Explorer clamped at the minimum: thirty presses cannot collapse it,
        // and a thirty-first press reports that nothing moved.
        let widths = classic_widths(&tree, area);
        assert!(widths[0] >= 10, "explorer survived: {widths:?}");
        assert!(
            !nudge_focused(&mut tree, FocusArea::CodeEditor, 5),
            "clamped means no move to report"
        );
    }

    #[test]
    fn nudging_a_missing_focus_is_a_no_op() {
        let area = rect(100, 22);
        let mut tree = chat_first_tree(area, false);
        // No explorer leaf in chat-first: nothing to grow.
        assert!(!nudge_focused(&mut tree, FocusArea::FileExplorer, 5));
    }

    #[test]
    fn resized_trees_survive_tiny_areas() {
        // The done-when's resize clause: a resized tree at 20x4 renders inside
        // its area without panicking, whatever the ratios became.
        let area = rect(100, 22);
        let mut tree = classic_tree(area, false);
        for _ in 0..10 {
            nudge_focused(&mut tree, FocusArea::CodeEditor, 5);
        }
        for tiny in [rect(20, 4), rect(3, 1), rect(1, 1)] {
            for (pane, r) in render(&tree, tiny) {
                assert_eq!(r.intersection(tiny), r, "{:?} escapes {tiny:?}", pane.slot);
            }
        }
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

    #[test]
    fn a_point_hit_test_agrees_with_the_column_rule_for_every_preset() {
        // The mouse path now asks by cell, because a template may stack panes.
        // For the presets — which split side by side — the answer may not move:
        // every cell lands on the pane the shipped column rule names.
        for area in areas() {
            for preset in LAYOUT_NAMES {
                for focus in focuses() {
                    for show_terminal in [false, true] {
                        let tree = preset_tree(area, preset, show_terminal, focus);
                        let shipped = compute_layout(area, preset, show_terminal, focus);
                        for row in area.y..area.bottom() {
                            for column in 0..area.width.saturating_add(2) {
                                assert_eq!(
                                    hit_test_tree_point(&tree, area, row, column),
                                    shipped.hit_test(column),
                                    "{preset} row={row} col={column} {area:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
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
        // This is what makes a data template agree with the preset builders,
        // which decide the same thing while building (proved against
        // `compute_layout` in `templates.rs`).
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
