//! ASCII progress-bar helper.
//!
//! Extracted from the inline `bar()` closure in `draw_performance_dashboard`
//! (`ui.rs:659`). That closure took `(value: u64, max: u64, width: usize)`.
//! Other panels use slightly different signatures (`gauge_block` with title/color,
//! inline `filled/empty` blocks) — those are left in place for now to avoid
//! changing pixels; they'll be unified in a later pass once panels migrate.

/// Render an ASCII progress bar string of the given width.
///
/// Returns `width` characters: `filled` `█` followed by `empty` `░`.
/// If `max == 0`, returns `width` spaces.
pub fn bar(value: u64, max: u64, width: usize) -> String {
    if max == 0 {
        return " ".repeat(width);
    }
    let filled = ((value as f64 / max as f64) * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bar_half() {
        assert_eq!(bar(5, 10, 10), "█████░░░░░");
    }

    #[test]
    fn test_bar_full() {
        assert_eq!(bar(10, 10, 6), "██████");
    }

    #[test]
    fn test_bar_zero_max() {
        assert_eq!(bar(5, 0, 6), "      ");
    }

    #[test]
    fn test_bar_overflow_clamps() {
        assert_eq!(bar(20, 10, 8), "████████");
    }
}
