//! A team of worker agents on one project (TM-1 … TM-6): each worker is an
//! outside agent (or xencode itself) driven over the Agent Client Protocol in
//! its own git worktree; its work merges only when the project's checks pass
//! on the merged result. The engine hosts the team; this crate holds what does
//! not need the terminal app.

pub mod agents;
pub mod merge;
pub mod worker;
pub mod worktree;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A worker's id within one engine: `w1`, `w2`, …
pub type WorkerId = String;

/// Where a worker is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    Starting,
    Working,
    NeedsYou,
    Done,
    Failed,
    Stopped,
}

/// What a window or the lead is told about one worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerSnapshot {
    pub id: WorkerId,
    pub agent: String,
    pub task: String,
    pub state: WorkerState,
    pub branch: String,
    pub worktree: PathBuf,
    /// The last line the worker said.
    pub last_message: String,
    /// Everything the worker said in its latest turn.
    pub answer: String,
    pub error: Option<String>,
    pub tool_calls: u32,
    pub tokens: Option<u64>,
    pub cost_micros: Option<u64>,
    /// Signed in with the person's own plan, so not priced.
    pub on_plan: bool,
    /// Its merge, once one was asked for (TM-3).
    #[serde(default)]
    pub merge: Option<MergeState>,
}

/// Where a worker's merge is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeState {
    Running,
    Finished(merge::MergeOutcome),
}
