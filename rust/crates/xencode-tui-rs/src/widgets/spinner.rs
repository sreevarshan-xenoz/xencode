//! Braille spinner frames.
//!
//! Extracted from `ui.rs` where the same `['⠋', '⠙', ...]` array was duplicated
//! ~10× across dashboard, provider-health, bytebot, collab, voice, terminal,
//! security, profiler, and learning panels.

/// Braille spinner frames, cycled by `spinner_tick`.
pub const SPINNER_FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Get the spinner character for the given tick.
pub fn frame(tick: usize) -> char {
    SPINNER_FRAMES[tick % SPINNER_FRAMES.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_wraps_around() {
        assert_eq!(frame(0), '⠋');
        assert_eq!(frame(SPINNER_FRAMES.len()), '⠋');
        assert_eq!(frame(SPINNER_FRAMES.len() + 1), '⠙');
    }
}
