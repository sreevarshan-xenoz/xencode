//! Measuring what the coding-agent CLIs on this machine actually do.
//!
//! This crate exists for `AR-1` and follows the boundary §S-11 set for it: it
//! knows agents and nothing about plans. Orchestration — what to ask them, in
//! what order, and what happens when they disagree — is `OR-*` and lives
//! elsewhere.
//!
//! # The one rule that shapes everything here
//!
//! **`AR-1` says "nothing may be inferred from documentation."** So a cell in the
//! matrix is never quietly filled from a help screen or a web page. Every cell
//! carries how it was learned, in [`Provenance`], and a cell that was read rather
//! than observed says so in the report. A matrix where half the cells are
//! transcribed from `--help` is not a measurement, and the whole reason for
//! running the CLIs is that the transcripts disagree with the help text.
//!
//! The second half of the same rule: **a run that failed is recorded as a
//! failure.** An agent that refuses because it is not signed in is evidence about
//! that agent — it fails fast, with a clear message, on stderr, and costs
//! nothing. That is a *cell*, not a gap.
//!
//! # What a run may and may not cost
//!
//! Nothing here sends a model request unless the operator passes [`ProbeOptions::`]
//! allowing it, and even then the task is read-only by construction. An agent
//! that needs an account and has none stops at the auth check, which is still a
//! useful observation and is recorded as one.

pub mod probe;
pub mod roster;

pub use probe::{ProbeOptions, ProbeReport, RunCapture};
pub use roster::{
    inventory, provenance_of_help_cell, AgentSpec, InstallSource, InstalledAgent, ROSTER,
};
