//! Transient toast notifications (E3-03).
//!
//! File-watch warnings surface here instead of being injected as fake system
//! chat lines, so they stop polluting the conversation history.

use crate::theme::ThemeColors;
use ratatui::{
    style::Style,
    text::{Line, Span},
};

pub const TOAST_TTL_SECS: f64 = 6.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Warning,
    Info,
    Success,
}

#[derive(Clone)]
pub struct Toast {
    pub message: String,
    pub kind: ToastKind,
    pub expires: f64,
}

/// Push with dedup: an identical active toast is refreshed, not doubled.
pub fn push(list: &mut Vec<Toast>, message: String, kind: ToastKind, now: f64) {
    list.retain(|t| !(t.kind == kind && t.message == message));
    list.push(Toast {
        message,
        kind,
        expires: now + TOAST_TTL_SECS,
    });
}

pub fn prune(list: &mut Vec<Toast>, now: f64) {
    list.retain(|t| t.expires > now);
}

pub fn last_of_kind(list: &[Toast], kind: ToastKind) -> Option<&str> {
    list.iter()
        .rev()
        .find(|t| t.kind == kind)
        .map(|t| t.message.as_str())
}

/// Overlay lines, newest first, capped so toasts never eat the screen.
pub fn render_lines(list: &[Toast], theme: &ThemeColors) -> Vec<Line<'static>> {
    list.iter()
        .rev()
        .take(4)
        .map(|t| {
            let (icon, color) = match t.kind {
                ToastKind::Warning => ("⚠", theme.message_user),
                ToastKind::Info => ("ℹ", theme.message_system),
                ToastKind::Success => ("✓", theme.message_assistant),
            };
            Line::from(Span::styled(
                format!(" {icon} {}", t.message),
                Style::default().fg(color),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_dedups_and_refreshes() {
        let mut list = Vec::new();
        push(&mut list, "x changed".into(), ToastKind::Warning, 100.0);
        push(&mut list, "x changed".into(), ToastKind::Warning, 103.0);
        assert_eq!(list.len(), 1);
        assert!((list[0].expires - 109.0).abs() < f64::EPSILON);
    }

    #[test]
    fn prune_removes_only_expired() {
        let mut list = Vec::new();
        push(&mut list, "old".into(), ToastKind::Info, 0.0);
        push(&mut list, "new".into(), ToastKind::Info, 10.0);
        prune(&mut list, 10.0 + TOAST_TTL_SECS / 2.0);
        assert_eq!(last_of_kind(&list, ToastKind::Info), Some("new"));
        prune(&mut list, 1000.0);
        assert!(list.is_empty());
    }

    #[test]
    fn render_caps_at_four_newest_first() {
        let mut list = Vec::new();
        for i in 0..6 {
            push(&mut list, format!("t{i}"), ToastKind::Warning, i as f64);
        }
        let lines = render_lines(&list, &ThemeColors::get("ocean"));
        assert_eq!(lines.len(), 4);
        assert!(lines[0].spans[0].content.contains("t5"));
    }
}
