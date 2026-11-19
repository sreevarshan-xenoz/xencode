//! Crate root of the relationship fixture: declares every module.
//!
//! This file is read from disk by the relationship tests and mapped to the
//! logical path `src/lib.rs`. Its `mod` declarations are real edges.

mod store;
mod service;
mod traits;
mod view;
