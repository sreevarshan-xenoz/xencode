//! The retrieval baseline for this workspace, measured against the whole indexed
//! repo rather than a toy haystack. Ignored by default because it builds a real
//! project index (about a second) and writes it into `.xencode/`, which is
//! git-ignored but shared with the TUI.
//!
//! Run it to see what the structural retriever and the two hybrid arms
//! actually score on the built-in gold set:
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
    // Three arms, so the run prices each addition separately: structural only,
    // plus a lexical arm over path and symbols, plus the same with the files'
    // documentation prose indexed too.
    let arms: [(&str, xencode_context_rs::RetrieveOptions); 3] = [
        ("deterministic      ", Default::default()),
        (
            "+ text (path+symbol)",
            xencode_context_rs::RetrieveOptions {
                lexical: true,
                lexical_docs: false,
                ..Default::default()
            },
        ),
        (
            "+ text + doc prose ",
            xencode_context_rs::RetrieveOptions {
                lexical: true,
                ..Default::default()
            },
        ),
    ];
    for (label, opts) in &arms {
        let started = std::time::Instant::now();
        let report = xencode_context_rs::evaluate_with(&index, &gold, 5, &no_changes, opts);
        println!(
            "{label}  recall@1={:.3}  recall@5={:.3}  MRR={:.3}  {:.1} ms/query",
            report.recall_at[0],
            report.recall_at[4],
            report.mrr,
            started.elapsed().as_secs_f64() * 1000.0 / gold.len().max(1) as f64,
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
