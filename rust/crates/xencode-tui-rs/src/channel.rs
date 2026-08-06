//! Channel sender wrapper and panel-action enum.
//!
//! New infrastructure introduced in Step 0 of the foundation refactor.
//! The `ChannelSender` newtype lets the `Panel` trait reference a concrete
//! sender type without importing the full `tokio::sync::mpsc` path everywhere.
//! `PanelAction` is the return type of `Panel::handle_key`, letting panels
//! request cross-panel focus switches (performed centrally by the dispatcher).

use tokio::sync::mpsc;

use crate::focus::FocusArea;

/// Wrapper around the unbounded channel sender.
#[derive(Clone)]
pub struct ChannelSender {
    pub tx: mpsc::UnboundedSender<String>,
}

impl ChannelSender {
    pub fn new(tx: mpsc::UnboundedSender<String>) -> Self {
        Self { tx }
    }

    pub fn send(&self, msg: impl Into<String>) {
        let _ = self.tx.send(msg.into());
    }
}

/// Action returned by a panel's `handle_key`, for the dispatcher to execute.
/// Panels cannot change `app.focus` or touch sibling panels directly; they
/// request it here and the dispatcher performs the mutation centrally.
pub enum PanelAction {
    /// Key was consumed; no further action needed.
    Consumed,
    /// Key was not consumed; fall through to default handlers.
    NotConsumed,
    /// Switch focus to a different area.
    SwitchFocus(FocusArea),
    /// Request to quit the app.
    Quit,
}
