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
pub mod budget;
pub mod compact;
pub mod context;
pub mod conversation;
pub mod documents;
pub mod embed;
pub mod eval;
pub mod gitinfo;
pub mod index;
pub mod init;
pub mod metrics;
pub mod refresh;
pub mod retrieve;
pub mod scanner;
pub mod stale;
pub mod state;
pub mod symbols;
pub mod watcher;
pub mod worktree;

pub use advise::{
    advise, advise_from_snapshot, affected_dependents, broken_imports, find_cycles, hub_files,
    orphan_files, Advice, AdviceKind, AFFECTED_MAX_HOPS, HUB_MIN_OUT,
};
pub use budget::{est_tokens, truncate_tail_to_tokens, truncate_to_tokens, HardwareProfile};
pub use compact::{
    hard_compact_prompt, parse_hard_compact_reply, should_compact, soft_compact, CompactReport,
    CompactionKind,
};
pub use context::{
    assemble_chat, assemble_prompt, collect_live_context, git_summary_text, read_retrieved_bodies,
    stable_system_text, ChatAssembly, ChatInput, ChatTurn, LiveContext, RetrievedBlock, TierDoc,
    AGENT_SYSTEM_PROMPT, HISTORY_TURN_OVERHEAD_TOKENS, STABLE_END_MARKER,
};
pub use conversation::{Transcript, TranscriptEntry};
pub use documents::{
    is_document_path, parse_document, parse_document_bytes, DocError, DocKind, DocText,
    MAX_DOC_BYTES, MAX_DOC_CHARS,
};
pub use embed::{hybrid_rerank, pseudo_document, tokenize, Bm25};
pub use eval::{default_gold, evaluate, gold_from_disk, EvalItem, EvalReport, EvalRun};
pub use gitinfo::{
    changed_paths_between, current_git_info, dirty_paths, git_diff_file, git_diff_numstat,
    git_file_set, is_git_repo, parse_numstat, DiffFile, GitInfo, MAX_DIFF_CHARS,
};
pub use index::{
    deps_json_path, file_index_path, read_json, symbols_json_path, write_atomic, write_str_atomic,
    FileEntry, FilesIndex, Manifest, VERSION,
};
pub use init::{init_project, ContextError, InitSummary, XENCODE_DIR};
pub use metrics::{append_metrics, read_metrics, CompactAction, RequestMetrics};
pub use refresh::{refresh_rust_file, RefreshOutcome};
pub use retrieve::{retrieve, word_tokens, RetrievalIndex, RetrieveOptions, RetrievedFile};
pub use scanner::{scan_tree, Language, ScanOptions, ScanOutcome};
pub use stale::{FileContextTracker, FileStateKind, LoadedRecord, LoadedState, TrackedFile};
pub use state::{has_decision_marker, ContextState};
pub use symbols::{
    build_graph, dependency_map, dependent_map, extract_rust_symbols, rank_files, resolve_import,
    DepEdge, PerFileSymbols,
};
pub use watcher::{
    map_kind, should_ignore, Debounce, WatchEvent, WatchKind, WatcherSession, WorkspaceWatcher,
    DEFAULT_DEBOUNCE, DEFAULT_EXCLUDED_DIRS,
};
pub use worktree::{
    parse_worktree_list, worktree_add, worktree_list, worktree_remove, WorktreeInfo,
};

use std::path::Path;

/// Root directory used by the context engine inside a project.
pub fn default_root() -> std::path::PathBuf {
    std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf())
}
