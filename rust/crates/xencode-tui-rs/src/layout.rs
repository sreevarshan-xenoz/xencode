//! The named body rectangles one frame drew.
//!
//! The Files view's panes come from one fixed tree (`App::body_tree`), folded
//! into this struct by `view::to_body_layout` so the mouse, the Tab ring and the
//! scroll keys read the same rectangles the draw used. A pane that is not on
//! screen is `None`, not a zero-sized rect: callers skip drawing it, and the Tab
//! ring never lands on it.

use ratatui::layout::Rect;

/// Every body-region rect for one frame, already resolved: `None` means that
/// pane is not on screen. `terminal` is `None` when the embedded terminal is
/// off *or* the body is too short to host it — that decision is made once, in
/// `view::render`, not in the draw code.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct BodyLayout {
    pub explorer: Option<Rect>,
    pub editor: Option<Rect>,
    pub chat: Option<Rect>,
    pub input: Option<Rect>,
    pub terminal: Option<Rect>,
    pub agents: Option<Rect>,
}
