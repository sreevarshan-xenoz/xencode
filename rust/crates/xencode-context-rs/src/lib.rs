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

pub mod budget;
pub mod cmd_output;
pub mod compact;
pub mod context;
pub mod conversation;
pub mod embed;
pub mod eval;
pub mod gitinfo;
pub mod index;
pub mod init;
pub mod metrics;
pub mod retrieve;
pub mod scanner;
pub mod stale;
pub mod state;
pub mod symbols;

pub use budget::{est_tokens, truncate_tail_to_tokens, truncate_to_tokens, HardwareProfile};
pub use cmd_output::{
    capture_output, deduce_files, read_raw, render_summary, CmdIndex, CmdRecord, DETAIL_LINES_CAP,
};
pub use compact::{
    hard_compact_prompt, parse_hard_compact_reply, should_compact, soft_compact, CompactReport,
    CompactionKind,
};
pub use context::{
    assemble_prompt, git_summary_text, read_retrieved_bodies, RetrievedBlock, TierDoc, STABLE_END_MARKER,
};
pub use conversation::{Transcript, TranscriptEntry};
pub use embed::{cosine, hybrid_rerank, pseudo_document, tokenize, Bm25, Embedder};
pub use eval::{default_gold, evaluate, gold_from_disk, EvalItem, EvalReport, EvalRun};
pub use gitinfo::{changed_paths_between, current_git_info, dirty_paths, git_file_set, is_git_repo, GitInfo};
pub use index::{write_atomic, write_str_atomic, FileEntry, FilesIndex, Manifest, VERSION};
pub use init::{init_project, ContextError, InitSummary, XENCODE_DIR};
pub use metrics::{append_metrics, read_metrics, CompactAction, RequestMetrics};
pub use retrieve::{retrieve, word_tokens, RetrievedFile, RetrieveOptions, RetrievalIndex};
pub use scanner::{scan_tree, Language, ScanOutcome, ScanOptions};
pub use stale::{FileContextTracker, FileStateKind, LoadedRecord, LoadedState, TrackedFile};
pub use state::{has_decision_marker, ContextState};
pub use symbols::{
    build_graph, dependency_map, dependent_map, expand_dependencies, extract_rust_symbols,
    rank_files, resolve_import, DepEdge, PerFileSymbols,
};

use std::path::Path;
use std::sync::atomic::AtomicBool;

/// Root directory used by the context engine inside a project.
pub fn default_root() -> std::path::PathBuf {
    std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf())
}

/// Shared cancellation flag type passed to [`init_project`].
pub type CancelFlag = std::sync::Arc<AtomicBool>;