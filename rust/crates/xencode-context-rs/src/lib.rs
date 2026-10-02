//! Local-first project context engine (M0: deterministic structural pass).
//!
//! Reads a repository on disk and produces the `.xencode/` scaffold plus the
//! retrieval index (`index/files.json`, `index/manifest.json`) without any LLM
//! calls. Later milestones add symbol extraction, LLM summaries and retrieval.
//!
//! Pass 1 is entirely deterministic and testable; `/init` in the TUI drives it
//! through a progress callback that receives tagged lines:
//!   - `phase_start:<name>`
//!   - `phase_done:<name>`
//!   - `log:<message>`
//!
//! The single entry point is [`init_project`].

pub mod advise;
pub mod anchor;
pub mod artifacts;
pub mod budget;
pub mod cochange;
pub mod compact;
pub mod context;
pub mod conversation;
pub mod crate_graph;
pub mod doctor;
pub mod documents;
pub mod editing;
pub mod embed;
pub mod eval;
pub mod gitinfo;
pub mod histdigest;
pub mod history;
pub mod hotspots;
pub mod hwprobe;
pub mod impact;
pub mod impact_tree;
pub mod index;
pub mod init;
pub mod ledger;
pub mod metrics;
pub(crate) mod parse;
pub mod perf;
pub mod pricing;
pub mod prompts;
pub mod refresh;
pub mod releasenotes;
pub mod repo_map;
pub mod retrieve;
pub mod rollup;
pub mod scanner;
pub mod seeds;
pub mod session;
pub mod shape;
pub mod stale;
pub mod state;
pub mod symbols;
pub mod trace;
pub mod tsymbols;
pub mod verify;
pub mod watcher;
pub mod worktree;

pub use advise::{
    advise, advise_from_snapshot, affected_dependents, broken_imports, find_cycles, hub_files,
    orphan_files, Advice, AdviceKind, AFFECTED_MAX_HOPS, HUB_MIN_OUT,
};
pub use anchor::{
    discover, is_current, prove, render, write_anchor, Discovery, Kind, Provenance, Recipe, Verdict,
};
pub use budget::{
    est_tokens, fill_target, truncate_tail_to_tokens, truncate_to_tokens, ContextCaps,
    HardwareProfile, ProfileDecision, PromptOverhead, TOKENS_PER_RETRIEVED_FILE,
};
pub use cochange::{
    load_history, mine_commit_history, parse_commit_log, save_history, CommitHistory, FileCommits,
    COCHANGE_BONUS, COMMIT_LOG_LIMIT, HUB_COMMIT_DIVISOR, MASS_COMMIT_FILES, RECENCY_BONUS,
    RECENCY_WINDOW_DAYS, TOP_PARTNERS,
};
pub use compact::{
    hard_compact_prompt, parse_hard_compact_reply, should_compact, soft_compact, CompactReport,
    CompactionKind,
};
pub use context::{
    assemble_chat, assemble_prompt, collect_live_context, git_summary_text, read_retrieved_bodies,
    stable_system_text, ChatAssembly, ChatInput, ChatTurn, LiveContext, RetrievedBlock, TierDoc,
    HISTORY_TURN_OVERHEAD_TOKENS, STABLE_END_MARKER,
};
pub use conversation::{Transcript, TranscriptEntry};
pub use crate_graph::{cargo_metadata, crate_of_file, parse_crate_graph, CrateGraph, EdgeKind};
pub use doctor::{
    check_cache_writable, check_git, check_index, check_mcp, check_metrics, check_provider,
    probe_env, resolve_on_path, tcp_reachable, EnvFacts, SelfCheck,
};
pub use documents::{
    is_document_path, parse_document, parse_document_bytes, DocError, DocKind, DocText,
    MAX_DOC_BYTES, MAX_DOC_CHARS,
};
pub use editing::{replace_symbol_body, EditFailure};
pub use embed::{hybrid_select, pseudo_document, tokenize, Bm25, LEXICAL_WEIGHT};
pub use eval::{
    append_eval_run, comparable_previous_eval_run, compare_shapes, default_gold, eval_log_path,
    evaluate, evaluate_with, gold_by_shape, gold_from_disk, previous_eval_run, read_eval_runs,
    EvalItem, EvalReport, EvalRun, EvalRunRecord, ShapeComparison,
};
pub use gitinfo::{
    changed_paths_between, current_git_info, dirty_paths, git_diff_file, git_diff_numstat,
    git_file_set, is_git_repo, parse_numstat, DiffFile, GitInfo, MAX_DIFF_CHARS,
};
pub use histdigest::{history_digest, recent_subjects, DIGEST_CHAR_CAP};
pub use history::{
    default_blame_target, history_setup, history_status, CommitGraph, HistorySetup, HistoryStatus,
    PackIndex, QueryTime,
};
pub use hotspots::{
    hotspots, owners_for, parse_churn, parse_codeowners, pattern_matches, FileHistory, OwnerRule,
};
pub use impact::{
    change_impact, impact, impact_from_filesystem, impact_from_snapshot, removal_from_graph,
    removal_impact, ChangeImpact, ImpactReport, ImpactedFile, RemovalImpact, DECLARED_CAP,
    IMPACT_MAX_HOPS,
};
pub use impact_tree::{
    impact_tree, ChurnSummary, ImpactFile, ImpactGroup, ImpactRow, ImpactTree, UNCLAIMED,
};
pub use index::{
    deps_json_path, file_index_path, history_json_path, read_json, symbols_json_path, write_atomic,
    write_str_atomic, FileEntry, FilesIndex, Manifest, VERSION,
};
pub use init::{init_project, ContextError, InitSummary, XENCODE_DIR};
pub use ledger::{
    append_ledger, digest_hex, ledger_for_session, ledger_path, read_ledger, LedgerEntry, RunClass,
};
pub use metrics::{
    append_metrics, metrics_path, read_metrics, read_metrics_since, read_metrics_tail,
    CompactAction, MetricSource, MetricsIdentity, RequestMetrics,
};
pub use pricing::{
    cost_of, format_usd, pricing_path, CostReport, ModelCost, ModelPrice, PriceTable,
};
pub use refresh::{refresh_rust_file, RefreshOutcome};
pub use repo_map::{
    rank_repo_map, repo_map_text, MapRow, REPO_MAP_CAP_TOKENS, REPO_MAP_MAX_FILES,
    REPO_MAP_MAX_HOPS, REPO_MAP_MAX_SYMBOLS,
};
pub use retrieve::{retrieve, word_tokens, RetrievalIndex, RetrieveOptions, RetrievedFile};
pub use rollup::{
    read_rollup, refresh_rollup, rollup_path, write_rollup, MetricsRollup, Percentiles,
    ProfileSample, SessionTotals, TokenTotals, RATE_SAMPLE_WINDOW, ROLLUP_VERSION,
};
pub use scanner::{
    detect_language, language_for_extension, language_has_semantic_tier, scan_tree,
    semantic_tier_refusal, Language, ScanOptions, ScanOutcome,
};
pub use seeds::{
    all_seed_cases, seed_case, write_seed, BugShape, GraderRun, SeedCase, SeedEdit, SeedFile,
    SeededTask,
};
pub use session::{
    list_session_ids, new_run_id, read_session, resolve_run_id, session_path, sessions_dir,
    RecordedCall, RecordedRun, RecordedToolCall, Session, SessionLine, SessionWriter,
    SESSION_FORMAT,
};
pub use shape::{shape_of, ShapeRead, TaskShape};
pub use stale::{FileContextTracker, FileStateKind, LoadedRecord, LoadedState, TrackedFile};
pub use state::{has_decision_marker, ContextState};
pub use symbols::{
    build_graph, dependency_map, dependent_map, extract_rust_symbols, rank_files, resolve_import,
    DepEdge, PerFileSymbols,
};
pub use tsymbols::extract as extract_tree_symbols;
pub use verify::{
    nextest_available, run as run_tests, Engine, Options as TestOptions, Outcome as TestOutcome,
};

pub use trace::{
    append_trace, arguments_preview, prompt_digest, read_recent_traces, redact_secrets,
    tail_preview, trace_path, ToolTrace, TurnTrace, TRACE_ARGUMENTS_CAP, TRACE_TAIL_CAP,
};
pub use watcher::{
    map_kind, should_ignore, Debounce, WatchEvent, WatchKind, WatcherSession, WorkspaceWatcher,
    DEFAULT_DEBOUNCE, DEFAULT_EXCLUDED_DIRS,
};
pub use worktree::{
    parse_worktree_list, worktree_add, worktree_add_detached, worktree_list, worktree_remove,
    WorktreeInfo,
};

use std::path::Path;

/// Root directory used by the context engine inside a project.
pub fn default_root() -> std::path::PathBuf {
    std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf())
}
