//! Body layout: the one geometry source for draw, mouse hit-testing and
//! resize scroll-clamping.
//!
//! `compute_layout` is pure — presets are plain geometry, so switching the
//! layout can never lose pane state (open file, scrolls, messages live in
//! `App`, not here). Hidden panes are `None`, not zero-sized rects: callers
//! must skip drawing them, and clicks in their regions fall to the nearest
//! visible neighbour.

use crate::focus::FocusArea;
use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Layout presets in cycle order (←/→ on the Layout settings row, Ctrl+U).
pub const LAYOUT_NAMES: &[&str] = &["classic", "chat-first", "zen"];

/// The preset to render: an unknown configured value (typo, hand-edited
/// config) falls back to "classic" — same contract as unknown themes.
pub fn effective_layout(name: &str) -> &'static str {
    LAYOUT_NAMES
        .iter()
        .find(|l| **l == name)
        .map_or("classic", |&l| l)
}

/// Next layout name when cycling by one step (`forward`). Unknown values
/// start the cycle from "classic".
pub fn cycle_layout(active: &str, forward: bool) -> String {
    let len = LAYOUT_NAMES.len();
    let pos = LAYOUT_NAMES.iter().position(|l| *l == active).unwrap_or(0);
    let next = if forward {
        (pos + 1) % len
    } else {
        (pos + len - 1) % len
    };
    LAYOUT_NAMES[next].to_string()
}

/// Every body-region rect for one frame, already resolved: `None` means
/// the preset hides that pane entirely. `terminal` is `None` when the
/// embedded terminal is off *or* the chat column is too short to host it —
/// that decision lives here, once, not in the draw code.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct BodyLayout {
    pub explorer: Option<Rect>,
    pub editor: Option<Rect>,
    pub chat: Option<Rect>,
    pub input: Option<Rect>,
    pub terminal: Option<Rect>,
}

/// Below this many rows, a terminal pane would squeeze the chat column to
/// nothing useful, so it is dropped.
const TERMINAL_MIN_CHAT_HEIGHT: u16 = 18;

/// The chat column's vertical split (messages / optional terminal / input),
/// shared by every preset that shows chat.
fn split_chat_column(area: Rect, show_terminal: bool) -> (Rect, Option<Rect>, Rect) {
    if show_terminal && area.height >= TERMINAL_MIN_CHAT_HEIGHT {
        let c = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(6),    // chat
                Constraint::Length(8), // terminal
                Constraint::Length(3), // input
            ])
            .split(area);
        (c[0], Some(c[1]), c[2])
    } else {
        let c = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(area);
        (c[0], None, c[1])
    }
}

/// `body_focus` is the last focus that pointed at a body pane (the app
/// tracks it so overlay focus doesn't flicker zen's target); only
/// `FileExplorer` and `CodeEditor` claim the whole body, everything else
/// gets the chat column.
pub fn compute_layout(
    area: Rect,
    preset: &str,
    show_terminal: bool,
    body_focus: FocusArea,
) -> BodyLayout {
    match effective_layout(preset) {
        "chat-first" => {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(25), Constraint::Percentage(75)])
                .split(area);
            let (chat, terminal, input) = split_chat_column(cols[1], show_terminal);
            BodyLayout {
                explorer: None,
                editor: Some(cols[0]),
                chat: Some(chat),
                input: Some(input),
                terminal,
            }
        }
        "zen" => match body_focus {
            FocusArea::FileExplorer => BodyLayout {
                explorer: Some(area),
                editor: None,
                chat: None,
                input: None,
                terminal: None,
            },
            FocusArea::CodeEditor => BodyLayout {
                explorer: None,
                editor: Some(area),
                chat: None,
                input: None,
                terminal: None,
            },
            _ => {
                let (chat, terminal, input) = split_chat_column(area, show_terminal);
                BodyLayout {
                    explorer: None,
                    editor: None,
                    chat: Some(chat),
                    input: Some(input),
                    terminal,
                }
            }
        },
        _ => {
            // classic — the historical 20/50/30 split, pixel-for-pixel.
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(20),
                    Constraint::Percentage(50),
                    Constraint::Percentage(30),
                ])
                .split(area);
            let (chat, terminal, input) = split_chat_column(cols[2], show_terminal);
            BodyLayout {
                explorer: Some(cols[0]),
                editor: Some(cols[1]),
                chat: Some(chat),
                input: Some(input),
                terminal,
            }
        }
    }
}

impl BodyLayout {
    /// Which body panel owns terminal column `column`? Clicks landing on a
    /// hidden pane's region move to the nearest visible neighbour, so the
    /// mouse can never select something off-screen.
    pub fn hit_test(&self, column: u16) -> Option<FocusArea> {
        let visible: [Option<(Rect, FocusArea)>; 3] = [
            self.explorer.map(|r| (r, FocusArea::FileExplorer)),
            self.editor.map(|r| (r, FocusArea::CodeEditor)),
            self.chat.map(|r| (r, FocusArea::ChatInput)),
        ];
        let mut last: Option<(Rect, FocusArea)> = None;
        for pane in visible.into_iter().flatten() {
            if column < pane.0.right() {
                return Some(pane.1);
            }
            last = Some(pane);
        }
        last.map(|(_, focus)| focus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: u16, h: u16) -> Rect {
        Rect::new(0, 1, w, h) // y=1 like the real body row under the header
    }

    fn present(layout: &BodyLayout) -> Vec<(&'static str, Rect)> {
        let mut v = Vec::new();
        if let Some(r) = layout.explorer {
            v.push(("explorer", r));
        }
        if let Some(r) = layout.editor {
            v.push(("editor", r));
        }
        if let Some(r) = layout.chat {
            v.push(("chat", r));
        }
        if let Some(r) = layout.input {
            v.push(("input", r));
        }
        if let Some(r) = layout.terminal {
            v.push(("terminal", r));
        }
        v
    }

    #[test]
    fn unknown_names_fall_back_and_cycles_wrap() {
        assert_eq!(effective_layout("bogus"), "classic");
        assert_eq!(effective_layout("zen"), "zen");
        assert_eq!(cycle_layout("classic", true), "chat-first");
        assert_eq!(cycle_layout("chat-first", true), "zen");
        assert_eq!(cycle_layout("zen", true), "classic");
        assert_eq!(cycle_layout("classic", false), "zen");
        // Unknown starts from classic rather than getting stuck.
        assert_eq!(cycle_layout("bogus", true), "chat-first");
    }

    #[test]
    fn classic_reproduces_the_legacy_split() {
        // Today's body_chunks() at 80 wide: 20/50/30 percent. A legacy
        // reference computed with the same ratatui constraints the old code
        // used, so a regression here changes the default look.
        let area = rect(80, 22);
        let legacy: Vec<Rect> = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(20),
                Constraint::Percentage(50),
                Constraint::Percentage(30),
            ])
            .split(area)
            .to_vec();
        let layout = compute_layout(area, "classic", false, FocusArea::ChatInput);
        assert_eq!(layout.explorer, Some(legacy[0]));
        assert_eq!(layout.editor, Some(legacy[1]));
        assert_eq!(layout.chat.unwrap().width, legacy[2].width);
        assert_eq!(layout.input.map(|r| r.height), Some(3));
        assert_eq!(layout.terminal, None);
    }

    #[test]
    fn show_terminal_splits_chat_and_drops_when_too_short() {
        let tall = compute_layout(rect(80, 22), "classic", true, FocusArea::ChatInput);
        assert!(tall.terminal.is_some());
        assert_eq!(tall.terminal.unwrap().height, 8);
        assert_eq!(tall.input.unwrap().height, 3);

        let short = compute_layout(rect(80, 10), "classic", true, FocusArea::ChatInput);
        assert_eq!(
            short.terminal, None,
            "a squashed terminal is worse than none"
        );
        assert_eq!(short.input.unwrap().height, 3);
    }

    #[test]
    fn zen_follows_body_focus_and_hides_the_rest() {
        let area = rect(80, 22);
        let chat = compute_layout(area, "zen", false, FocusArea::ChatInput);
        assert_eq!(chat.explorer, None);
        assert_eq!(chat.editor, None);
        // chat column minus the 3-row input
        assert_eq!(chat.chat, Some(Rect::new(0, 1, 80, 19)));

        let explorer = compute_layout(area, "zen", false, FocusArea::FileExplorer);
        assert_eq!(explorer.explorer, Some(area));
        assert_eq!(explorer.input, None, "no chat input in zen-explorer");

        let editor = compute_layout(area, "zen", false, FocusArea::CodeEditor);
        assert_eq!(editor.editor, Some(area));
    }

    #[test]
    fn chat_first_hides_explorer_only() {
        let layout = compute_layout(rect(80, 22), "chat-first", true, FocusArea::ChatInput);
        assert_eq!(layout.explorer, None);
        assert!(layout.editor.is_some() && layout.chat.is_some());
        assert!(layout.terminal.is_some());
        // chat column is the 75 % share, so it keeps a wide terminal
        assert!(layout.chat.unwrap().width > layout.editor.unwrap().width);
    }

    #[test]
    fn every_preset_fits_its_area_without_overlap_at_any_size() {
        let areas = [
            rect(80, 22),
            rect(60, 19),
            rect(40, 9),
            rect(20, 4),
            rect(3, 1),
            rect(1, 1),
        ];
        for area in areas {
            for preset in LAYOUT_NAMES {
                for focus in [
                    FocusArea::ChatInput,
                    FocusArea::FileExplorer,
                    FocusArea::CodeEditor,
                ] {
                    for show_terminal in [false, true] {
                        let layout = compute_layout(area, preset, show_terminal, focus);
                        let rects = present(&layout);
                        for (name, r) in &rects {
                            assert_eq!(
                                r.intersection(area),
                                *r,
                                "{preset}/{name} {r:?} escapes {area:?}"
                            );
                        }
                        for (i, (a, ra)) in rects.iter().enumerate() {
                            for (b, rb) in &rects[i + 1..] {
                                assert!(
                                    ra.intersection(*rb).is_empty(),
                                    "{preset} t={show_terminal} f={focus:?}: {a} and {b} overlap"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn hit_test_skips_hidden_panes() {
        let layout = compute_layout(rect(80, 22), "chat-first", false, FocusArea::ChatInput);
        // Column 0 is inside the hidden explorer's old territory -> editor.
        assert_eq!(layout.hit_test(0), Some(FocusArea::CodeEditor));
        let editor_right = layout.editor.unwrap().right();
        assert_eq!(layout.hit_test(editor_right), Some(FocusArea::ChatInput));
        assert_eq!(layout.hit_test(u16::MAX), Some(FocusArea::ChatInput));

        let zen = compute_layout(rect(80, 22), "zen", false, FocusArea::FileExplorer);
        assert_eq!(zen.hit_test(79), Some(FocusArea::FileExplorer));

        let classic = compute_layout(rect(80, 22), "classic", false, FocusArea::ChatInput);
        assert_eq!(classic.hit_test(0), Some(FocusArea::FileExplorer));
        assert_eq!(
            classic.hit_test(classic.editor.unwrap().x),
            Some(FocusArea::CodeEditor)
        );
    }
}
