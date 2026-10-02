pub mod atomic;
pub mod jsonl;
pub mod result_envelope;
pub mod rustc_json;
pub mod scheduler;
pub mod task_contract;
pub mod tasks;
pub mod tasks_file;
pub mod workspace;

pub use atomic::write_atomic;
pub use jsonl::{read_jsonl_tolerant, JsonlRead};
pub use result_envelope::{
    Claim, Evidence, FinishStatus, Handoff, RanCommand, ResultEnvelope, ReviewerView,
};
pub use rustc_json::{
    cargo_json_command, parse as parse_rustc_json, BuildReport, Diagnostic, Suggestion,
    CARGO_JSON_FLAG, EXPLANATION_CAP, MAX_DIAGNOSTICS, MAX_EXPLANATIONS, MAX_HELPS,
};
pub use scheduler::{
    Binding, GraphError, NodeId, NodeOutcome, ScheduleError, ScheduleReport, Scheduler, TaskGraph,
    TaskNode,
};
pub use task_contract::{Breach, TaskContract};
pub use tasks::{
    TaskError, TaskManager, TaskRecord, TaskStatus, TaskStore, DEFAULT_TASK_TIMEOUT,
    MAX_OUTPUT_LINES,
};
pub use tasks_file::{FileTask, FileTaskRegistry};
pub use workspace::{scan_workspace, EntryKind, ScanOptions, WorkspaceEntry, WorkspaceScanError};
