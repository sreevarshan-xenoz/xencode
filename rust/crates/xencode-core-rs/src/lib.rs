pub mod atomic;
pub mod chain;
pub mod decompose;
pub mod jsonl;
pub mod lease;
pub mod profile;
pub mod result_envelope;
pub mod routing;
pub mod rustc_json;
pub mod scheduler;
pub mod sys;
pub mod task_contract;
pub mod tasks;
pub mod tasks_file;
pub mod team;
pub mod team_runs;
pub mod workspace;

pub use atomic::write_atomic;
pub use decompose::{decide, Reference, Split, SplitDecision, SplitError, SplitScore, Subtask};
pub use jsonl::{read_jsonl_tolerant, JsonlRead};
pub use lease::{
    LeaseConflict, LeaseDecision, LeaseRegistry, ScheduleOutcome, WaitingRequest, WorkerLease,
};
pub use profile::{Profile, WorkerRefusal};
pub use result_envelope::{
    Claim, Evidence, FinishStatus, Handoff, RanCommand, ResultEnvelope, ReviewerView,
};
pub use routing::{
    CandidateEvaluation, CapabilityRouter, Fact, Provenance, RejectionReason, RoutingDecision,
    StepNote, TaskRequirement, WorkerCandidate,
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
pub use team::{
    load_recipes, Capacity, RecipeError, RecipeFile, RoleSpec, TeamRecipe, GATE_CHECKS, RECIPES_DIR,
};
pub use team_runs::{
    estimate as estimate_from_runs, fingerprint as recipe_fingerprint, load_runs, Estimate,
    RoleRun, RunFile, RunObservation, TeamRun, RUNS_DIR,
};
pub use workspace::{scan_workspace, EntryKind, ScanOptions, WorkspaceEntry, WorkspaceScanError};
