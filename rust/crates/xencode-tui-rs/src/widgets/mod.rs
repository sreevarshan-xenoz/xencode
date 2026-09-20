//! Shared rendering widgets extracted from `ui.rs` (Step 0).
//!
//! - `spinner`: the Braille spinner frames duplicated ~10× across `ui.rs`.
//! - `gauge`: the `bar()` ASCII-bar helper duplicated across dashboard/profiler panels.

pub mod gauge;
pub mod spinner;

/// Border glyphs for every bordered panel, chosen by the user's
/// `rounded_borders` preference (H1-06). One function so a panel can never
/// opt out of the setting by building its own set.
pub fn panel_border_set(rounded: bool) -> ratatui::symbols::border::Set {
    if rounded {
        ratatui::symbols::border::ROUNDED
    } else {
        ratatui::symbols::border::PLAIN
    }
}
