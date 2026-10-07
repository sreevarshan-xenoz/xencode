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
pub mod bootstrap;
pub mod budget;
pub mod cli_impact;
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
pub mod factgc;
pub mod factrank;
pub mod factverify;
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
pub mod lesson;
pub mod metrics;
pub mod notes;
pub(crate) mod parse;
pub mod perf;
pub mod power;
pub mod pricing;
pub mod prompts;
pub mod redact;
pub mod refresh;
pub mod releasenotes;
pub mod repo_map;
pub mod retrieve;
pub mod rollup;
pub mod runledger;
pub mod scanner;
pub mod seeds;
pub mod session;
pub mod shape;
pub mod source;
pub mod stale;
pub mod state;
pub mod symbols;
pub mod trace;
pub mod trust;
pub mod tsymbols;
pub mod verify;
pub mod watcher;
pub mod worktree;

pub use advise::{
    advise, advise_from_snapshot, affected_dependents, broken_imports, find_cycles, hub_files,
    orphan_files, Advice, AdviceKind, FailingCheckObservation, ProposedTask, AFFECTED_MAX_HOPS,
    HUB_MIN_OUT,
};
pub use anchor::{
    discover, is_current, prove, render, write_anchor, AnchorMeta, ANCHOR_META_FILE,
    ANCHOR_STALE_AGE_DAYS, anchor_age_days, read_anchor_meta, read_anchor_meta_from_dir,
    write_anchor_meta, Discovery, Kind, Provenance, Recipe, Verdict,
};
pub use bootstrap::{
    bootstrap, offered_files, BootstrapEntry, BootstrapKind, BootstrapReport, OfferedFile,
    ProjectFacts,
};
pub use budget::{
    est_tokens, fill_target, truncate_tail_to_tokens, truncate_to_tokens, BudgetBreach,
    BudgetDimension, ContextCaps, DailyBudgets, HardwareProfile, ProfileDecision, PromptOverhead,
    TOKENS_PER_RETRIEVED_FILE,
};
pub use cli_impact::{detect_cli_impact, CliImpact, CommandDocRef, CommandImpact, StaleDocRef};
pub use cochange::{
    load_history, mine_commit_history, parse_commit_log, save_history, CommitHistory, FileCommits,
    COCHANGE_BONUS, COMMIT_LOG_LIMIT, HUB_COMMIT_DIVISOR, MASS_COMMIT_FILES, RECENCY_BONUS,
    RECENCY_WINDOW_DAYS, TOP_PARTNERS,
};
pub use compact::{
    audit_durable_facts, believed_state, disagreement_note, drop_stale_facts, fact_prose,
    fold_state_from_reply, hard_compact_prompt, parse_hard_compact_reply, promote_state_candidate,
    read_state_candidate, record_checks, should_compact, soft_compact, stamp_provenance,
    state_candidate_path, write_state_candidate, CompactReport, CompactionKind, DroppedFact,
    FactDisagreement, FactProblem, FoldRefusal, FoldReport, PromoteRefusal, StaleFacts,
    DISAGREEMENT_LINES_SHOWN, STATE_CANDIDATE_FILE, STATE_FILE_CAP_TOKENS, STATE_FILE_FACT_CAP,
};
pub use context::{
    assemble_chat, assemble_prompt, collect_live_context, git_summary_text, read_retrieved_bodies,
    stable_system_text, ChatAssembly, ChatInput, ChatTurn, LiveContext, RetrievedBlock, TierDoc,
    HISTORY_TURN_OVERHEAD_TOKENS, STABLE_END_MARKER,
};
pub use conversation::{Transcript, TranscriptEntry};
pub use crate_graph::{cargo_metadata, crate_of_file, parse_crate_graph, CrateGraph, EdgeKind};
pub use doctor::{
    check_cache_writable, check_config, check_config_version, check_dir_size, check_free_disk,
    check_git, check_index, check_mcp, check_metrics, check_model, check_permissions,
    check_provider, dir_usage, file_mode, format_bytes, probe_env, resolve_on_path, tcp_reachable,
    ConfigRead, EnvFacts, SelfCheck,
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
pub use factgc::{
    age_words, collect_gc, now_ms as gc_now_ms, queue_exists, read_tombstones, record_stale_facts,
    tombstone_path, FactTombstone, GcReport, Recording, RETIRE_AFTER_MONTHS, TOMBSTONE_FILE,
};
pub use factverify::{
    evidence_path, evidence_rows, read_evidence, record_evidence, CheckRecord, CheckVerdict,
    EvidenceRow, FactEvidence, EVIDENCE_FILE, REVISIONS_KEPT,
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
pub use lesson::{
    approve_lesson, asks_for_words, check_streak, denied_call, draft_lesson, drop_lesson,
    lesson_candidate_path, read_lesson, render_lesson, set_lesson, ApprovedLesson, Evidence,
    LessonDraft, LessonRefusal, LessonWrite, DENIED_SOURCE, LESSON_CANDIDATE_FILE,
    LESSON_CHECK_STREAK, LESSON_EVIDENCE_CAP,
};
pub use metrics::{
    append_metrics, metrics_path, read_metrics, read_metrics_since, read_metrics_tail,
    CompactAction, MetricSource, MetricsIdentity, RequestMetrics,
};
pub use notes::{
    append_note, note_lines, notes_path, read_notes, NoteWrite, NOTES_FILE, NOTES_MAX_LINES,
};
pub use pricing::{
    cost_of, format_usd, listing_provenance, lookup_path, pricing_path, CostReport, ModelCost,
    ModelPrice, PriceLookup, PriceSource, PriceTable, OPENROUTER_SOURCE, PRICE_TTL_DAYS,
};
pub use refresh::{refresh_rust_file, RefreshOutcome};
pub use repo_map::{
    rank_repo_map, repo_map_text, MapRow, REPO_MAP_CAP_TOKENS, REPO_MAP_MAX_FILES,
    REPO_MAP_MAX_HOPS, REPO_MAP_MAX_SYMBOLS,
};
pub use retrieve::{retrieve, word_tokens, RetrievalIndex, RetrieveOptions, RetrievedFile};
pub use rollup::{
    local_day_key, read_rollup, refresh_rollup, rollup_path, trim_metrics, write_rollup, DayTotals,
    MetricsRollup, ModelRates, Percentiles, ProfileSample, SessionTotals, TokenTotals, DAYS_KEPT,
    METRICS_KEEP_BYTES, MODEL_RATE_SAMPLE_WINDOW, RATE_SAMPLE_WINDOW, ROLLUP_VERSION,
};
pub use runledger::{
    accountability_trailer, append_run, read_runs, recent_runs, run_by_id, run_evidence, runs_path,
    ApprovalDecision, ApprovalRow, RunEvidence, RunRecord, RUNS_FILE, RUNS_WINDOW,
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
pub use source::{
    SourceClass, ATTACHED_DATA_NOTE, DATA_TOKEN, REPO_DATA_NOTE, UNTRUSTED_AGENTS_BANNER,
};
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

pub use redact::{Redactor, Vault};
pub use trace::{
    allowlisted_by, append_trace, arguments_preview, contains_secret, load_secret_allowlist,
    path_skips_secret_scan, prompt_digest, read_recent_traces, redact_secrets, scan_secrets,
    secret_spans, tail_preview, trace_path, SecretHit, ToolTrace, TurnTrace, TRACE_ARGUMENTS_CAP,
    TRACE_TAIL_CAP,
};
pub use trust::{
    agents_content_is_trusted, agents_sha256, read_agents_md, read_scoped_agents_md,
    resolve_agents_path, trust_agents, trust_agents_at, untrust_agents, untrust_agents_at,
    UNTRUSTED_BANNER,
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
