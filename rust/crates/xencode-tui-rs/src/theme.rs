//! Theme color definitions for the TUI.
//!
//! Holds the `ThemeColors` struct, the named theme palettes, and the shared
//! `THEME_NAMES`/`cycle_theme` used by the settings panel and key handler.

/// Themes available in the picker, in cycle order (←/→ on the Theme row).
pub const THEME_NAMES: &[&str] = &[
    "ocean",
    "midnight",
    "forest",
    "terminal",
    "dracula",
    "solarized",
    "nord",
    "light",
];

/// Next theme name when cycling by one step (`forward`), or `None` if the
/// active theme is unknown (nothing sensible to cycle to).
pub fn cycle_theme(active: &str, forward: bool) -> Option<String> {
    let len = THEME_NAMES.len();
    let pos = THEME_NAMES.iter().position(|t| *t == active)?;
    let next = if forward {
        (pos + 1) % len
    } else {
        (pos + len - 1) % len
    };
    Some(THEME_NAMES[next].to_string())
}

#[derive(Clone, Copy)]
pub struct ThemeColors {
    pub bg: ratatui::style::Color,
    pub fg: ratatui::style::Color,
    pub accent: ratatui::style::Color,
    pub border: ratatui::style::Color,
    pub border_active: ratatui::style::Color,
    pub highlight: ratatui::style::Color,
    pub highlight_fg: ratatui::style::Color,
    pub message_user: ratatui::style::Color,
    pub message_assistant: ratatui::style::Color,
    pub message_system: ratatui::style::Color,
    pub status_bg: ratatui::style::Color,
    pub status_fg: ratatui::style::Color,
    /// Semantic status colors — the palette panels use instead of raw ANSI
    /// constants (diff add/remove, severities, ok/fail states).
    pub success: ratatui::style::Color,
    pub warning: ratatui::style::Color,
    pub danger: ratatui::style::Color,
    pub info: ratatui::style::Color,
    pub accent_secondary: ratatui::style::Color,
}

impl ThemeColors {
    pub fn get(name: &str) -> Self {
        match name {
            "midnight" => Self {
                bg: ratatui::style::Color::Rgb(15, 17, 26),
                fg: ratatui::style::Color::Rgb(230, 230, 230),
                accent: ratatui::style::Color::Magenta,
                border: ratatui::style::Color::Rgb(60, 60, 80),
                border_active: ratatui::style::Color::Magenta,
                highlight: ratatui::style::Color::Magenta,
                highlight_fg: ratatui::style::Color::White,
                message_user: ratatui::style::Color::Magenta,
                message_assistant: ratatui::style::Color::Cyan,
                message_system: ratatui::style::Color::DarkGray,
                status_bg: ratatui::style::Color::Rgb(30, 30, 50),
                status_fg: ratatui::style::Color::Rgb(200, 200, 220),
                success: ratatui::style::Color::Green,
                warning: ratatui::style::Color::Yellow,
                danger: ratatui::style::Color::Red,
                info: ratatui::style::Color::Cyan,
                accent_secondary: ratatui::style::Color::Magenta,
            },
            "forest" => Self {
                bg: ratatui::style::Color::Rgb(15, 29, 20),
                fg: ratatui::style::Color::Rgb(233, 245, 234),
                accent: ratatui::style::Color::Green,
                border: ratatui::style::Color::Rgb(40, 70, 40),
                border_active: ratatui::style::Color::Green,
                highlight: ratatui::style::Color::Green,
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Green,
                message_assistant: ratatui::style::Color::Yellow,
                message_system: ratatui::style::Color::DarkGray,
                status_bg: ratatui::style::Color::Rgb(20, 40, 25),
                status_fg: ratatui::style::Color::Rgb(200, 230, 200),
                success: ratatui::style::Color::Green,
                warning: ratatui::style::Color::Yellow,
                danger: ratatui::style::Color::Red,
                info: ratatui::style::Color::Cyan,
                accent_secondary: ratatui::style::Color::Magenta,
            },
            "terminal" => Self {
                bg: ratatui::style::Color::Rgb(0, 17, 0),
                fg: ratatui::style::Color::Rgb(128, 255, 128),
                accent: ratatui::style::Color::Rgb(128, 255, 128),
                border: ratatui::style::Color::Rgb(0, 64, 0),
                border_active: ratatui::style::Color::Rgb(128, 255, 128),
                highlight: ratatui::style::Color::Rgb(0, 128, 0),
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Rgb(255, 255, 255),
                message_assistant: ratatui::style::Color::Rgb(128, 255, 128),
                message_system: ratatui::style::Color::Rgb(0, 128, 0),
                status_bg: ratatui::style::Color::Rgb(0, 30, 0),
                status_fg: ratatui::style::Color::Rgb(128, 255, 128),
                success: ratatui::style::Color::Green,
                warning: ratatui::style::Color::Yellow,
                danger: ratatui::style::Color::Red,
                info: ratatui::style::Color::Cyan,
                accent_secondary: ratatui::style::Color::Magenta,
            },
            "dracula" => Self {
                bg: ratatui::style::Color::Rgb(40, 42, 54),
                fg: ratatui::style::Color::Rgb(248, 248, 242),
                accent: ratatui::style::Color::Rgb(255, 121, 198),
                border: ratatui::style::Color::Rgb(68, 71, 90),
                border_active: ratatui::style::Color::Rgb(255, 121, 198),
                highlight: ratatui::style::Color::Rgb(189, 147, 249),
                highlight_fg: ratatui::style::Color::Rgb(40, 42, 54),
                message_user: ratatui::style::Color::Rgb(255, 121, 198),
                message_assistant: ratatui::style::Color::Rgb(80, 250, 123),
                message_system: ratatui::style::Color::Rgb(98, 114, 164),
                status_bg: ratatui::style::Color::Rgb(30, 31, 41),
                status_fg: ratatui::style::Color::Rgb(248, 248, 242),
                success: ratatui::style::Color::Green,
                warning: ratatui::style::Color::Yellow,
                danger: ratatui::style::Color::Red,
                info: ratatui::style::Color::Cyan,
                accent_secondary: ratatui::style::Color::Magenta,
            },
            "solarized" => Self {
                bg: ratatui::style::Color::Rgb(0, 43, 54),
                fg: ratatui::style::Color::Rgb(131, 148, 150),
                accent: ratatui::style::Color::Rgb(38, 139, 210),
                border: ratatui::style::Color::Rgb(7, 54, 66),
                border_active: ratatui::style::Color::Rgb(38, 139, 210),
                highlight: ratatui::style::Color::Rgb(42, 161, 152),
                highlight_fg: ratatui::style::Color::Rgb(0, 43, 54),
                message_user: ratatui::style::Color::Rgb(38, 139, 210),
                message_assistant: ratatui::style::Color::Rgb(133, 153, 0),
                message_system: ratatui::style::Color::Rgb(88, 110, 117),
                status_bg: ratatui::style::Color::Rgb(0, 30, 38),
                status_fg: ratatui::style::Color::Rgb(147, 161, 161),
                success: ratatui::style::Color::Green,
                warning: ratatui::style::Color::Yellow,
                danger: ratatui::style::Color::Red,
                info: ratatui::style::Color::Cyan,
                accent_secondary: ratatui::style::Color::Magenta,
            },
            "nord" => Self {
                bg: ratatui::style::Color::Rgb(46, 52, 64),
                fg: ratatui::style::Color::Rgb(216, 222, 233),
                accent: ratatui::style::Color::Rgb(136, 192, 208),
                border: ratatui::style::Color::Rgb(59, 66, 82),
                border_active: ratatui::style::Color::Rgb(136, 192, 208),
                highlight: ratatui::style::Color::Rgb(94, 129, 172),
                highlight_fg: ratatui::style::Color::Rgb(236, 239, 244),
                message_user: ratatui::style::Color::Rgb(136, 192, 208),
                message_assistant: ratatui::style::Color::Rgb(163, 190, 140),
                message_system: ratatui::style::Color::Rgb(97, 108, 135),
                status_bg: ratatui::style::Color::Rgb(36, 41, 51),
                status_fg: ratatui::style::Color::Rgb(216, 222, 233),
                success: ratatui::style::Color::Green,
                warning: ratatui::style::Color::Yellow,
                danger: ratatui::style::Color::Red,
                info: ratatui::style::Color::Cyan,
                accent_secondary: ratatui::style::Color::Magenta,
            },
            "light" => Self {
                bg: ratatui::style::Color::Rgb(255, 255, 255),
                fg: ratatui::style::Color::Rgb(35, 35, 35),
                accent: ratatui::style::Color::Rgb(0, 0, 139),
                border: ratatui::style::Color::Rgb(205, 205, 205),
                border_active: ratatui::style::Color::Rgb(0, 0, 139),
                highlight: ratatui::style::Color::Rgb(215, 228, 245),
                highlight_fg: ratatui::style::Color::Rgb(20, 30, 55),
                message_user: ratatui::style::Color::Rgb(0, 0, 139),
                message_assistant: ratatui::style::Color::Rgb(0, 100, 0),
                message_system: ratatui::style::Color::DarkGray,
                status_bg: ratatui::style::Color::Rgb(232, 232, 236),
                status_fg: ratatui::style::Color::Rgb(45, 45, 55),
                success: ratatui::style::Color::Rgb(0, 100, 0),
                warning: ratatui::style::Color::Rgb(160, 120, 0),
                danger: ratatui::style::Color::Rgb(139, 0, 0),
                info: ratatui::style::Color::Rgb(0, 110, 110),
                accent_secondary: ratatui::style::Color::Rgb(120, 40, 120),
            },
            // "ocean" and default
            _ => Self {
                bg: ratatui::style::Color::Rgb(11, 27, 43),
                fg: ratatui::style::Color::Rgb(234, 244, 255),
                accent: ratatui::style::Color::Cyan,
                border: ratatui::style::Color::Rgb(40, 60, 80),
                border_active: ratatui::style::Color::Cyan,
                highlight: ratatui::style::Color::Cyan,
                highlight_fg: ratatui::style::Color::Black,
                message_user: ratatui::style::Color::Cyan,
                message_assistant: ratatui::style::Color::Green,
                message_system: ratatui::style::Color::DarkGray,
                status_bg: ratatui::style::Color::Rgb(18, 40, 60),
                status_fg: ratatui::style::Color::Rgb(200, 220, 240),
                success: ratatui::style::Color::Green,
                warning: ratatui::style::Color::Yellow,
                danger: ratatui::style::Color::Red,
                info: ratatui::style::Color::Cyan,
                accent_secondary: ratatui::style::Color::Magenta,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{cycle_theme, ThemeColors, THEME_NAMES};
    use ratatui::style::Color;

    #[test]
    fn cycle_wraps_both_directions() {
        assert_eq!(cycle_theme("ocean", true).as_deref(), Some("midnight"));
        assert_eq!(cycle_theme("ocean", false).as_deref(), Some("light"));
        assert_eq!(cycle_theme("light", true).as_deref(), Some("ocean"));
        assert_eq!(cycle_theme("not-a-theme", true), None);
    }

    #[test]
    fn every_named_theme_resolves_to_its_own_palette() {
        // A misspelled arm in `ThemeColors::get` would silently fall through
        // to the ocean default; distinct (bg, accent) pairs catch that.
        let palettes: Vec<(Color, Color)> = THEME_NAMES
            .iter()
            .map(|name| {
                let t = ThemeColors::get(name);
                (t.bg, t.accent)
            })
            .collect();
        for (i, a) in palettes.iter().enumerate() {
            for (j, b) in palettes.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        a, b,
                        "themes {:?} and {:?} share a palette",
                        THEME_NAMES[i], THEME_NAMES[j]
                    );
                }
            }
        }
        // Unknown names still fall back to ocean (documented behavior).
        assert_eq!(ThemeColors::get("bogus").bg, ThemeColors::get("ocean").bg);
    }
}
