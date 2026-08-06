//! Shared rendering widgets extracted from `ui.rs` (Step 0).
//!
//! - `spinner`: the Braille spinner frames duplicated ~10× across `ui.rs`.
//! - `gauge`: the `bar()` ASCII-bar helper duplicated across dashboard/profiler panels.

pub mod gauge;
pub mod spinner;
