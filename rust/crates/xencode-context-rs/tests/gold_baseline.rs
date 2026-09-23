//! The retrieval baseline for this workspace, measured against the whole indexed
//! repo rather than a toy haystack. Ignored by default because it builds a real
//! project index (about a second) and writes it into `.xencode/`, which is
//! git-ignored but shared with the TUI.
//!
//! Run it to see what the deterministic retriever and the BM25 rerank actually
//! score on the built-in gold set:
//!
//! ```text
//! cargo test -p xencode-context-rs --test gold_baseline -- --ignored --nocapture
//! ```
//!
//! The numbers it prints are the ones a retrieval change is expected to move.
//! They are reported, never asserted: this repo's file set changes and a fixed
//! threshold would turn the measurement into a fixture.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("crate sits at <root>/rust/crates/<name>")
        .to_path_buf()
}

#[test]
#[ignore]
fn gold_scores_against_the_real_index() {
    let root = workspace_root();
    xencode_context_rs::init_project(&root, std::sync::Arc::new(AtomicBool::new(false)), |_| {})
        .expect("init_project on the real workspace");

    let xencode = root.join(xencode_context_rs::XENCODE_DIR);
    let index = xencode_context_rs::RetrievalIndex::load(&xencode).expect("index on disk");
    println!(
        "indexed {} files, {} with symbols, {} dependency edges",
        index.files.len(),
        index.symbols.len(),
        index.deps.len()
    );

    let gold = xencode_context_rs::default_gold();
    let no_changes = HashSet::new();
    for (label, rerank) in [("deterministic", false), ("hybrid rerank", true)] {
        let report = xencode_context_rs::evaluate(&index, &gold, 5, &no_changes, rerank);
        println!(
            "{label:14}  recall@1={:.3}  recall@5={:.3}  MRR={:.3}",
            report.recall_at[0], report.recall_at[4], report.mrr
        );
        for (query, _, rank, _) in &report.hits {
            let shown = if *rank == usize::MAX {
                "outside top 5".to_string()
            } else {
                format!("rank {rank}")
            };
            println!("  {shown:>14}  {query}");
        }
    }
}
