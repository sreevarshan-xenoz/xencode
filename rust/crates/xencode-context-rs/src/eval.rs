//! Retrieval evaluation (M5) — measure before you embed (§18).
//!
//! Deterministic retrieval is the baseline; semantic retrieval only earns its
//! way in if it beats it. `evaluate_retrieval` runs a gold set of
//! `query → expected files` through the pipeline and reports:
//!
//! - recall@k / precision@k for k = 1..=top_k
//! - mean reciprocal rank (MRR)
//! - per-query hit rank, for eyeballing systematic misses
//!
//! Gold sets: `.xencode/eval/gold.json` (overrides the built-in sample, which
//! is aimed at the Xencode repo itself). The built-in sample mixes probes whose
//! answer is in the file name with probes that need the vocabulary a file
//! declares, so a change that only sharpens filename matching cannot pass it
//! unchanged. `cargo test -p xencode-context-rs --test gold_baseline --
//! --ignored --nocapture` prints what the current retriever scores on it.

use crate::retrieve::{retrieve, RetrievalIndex, RetrieveOptions, RetrievedFile};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalItem {
    pub query: String,
    /// Repo-relative paths that SHOULD surface in the top-K.
    #[serde(default)]
    pub expected: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRun {
    #[serde(default)]
    pub items: Vec<EvalItem>,
    #[serde(default)]
    pub top_k: usize,
    /// Whether the run applied the BM25 hybrid rerank stage.
    #[serde(default)]
    pub reranked: bool,
}

#[derive(Debug, Clone, Default)]
pub struct EvalReport {
    pub queries: usize,
    pub top_k: usize,
    pub reranked: bool,
    /// recall@k for k = 1..=top_k (1-indexed slots).
    pub recall_at: Vec<f64>,
    /// precision@k averaged over queries that have expectations.
    pub precision_at: Vec<f64>,
    /// MRR over queries that have expectations (0.0 on a total miss).
    pub mrr: f64,
    /// (query, expected, first-hit rank, every-ranked path).
    pub hits: Vec<(String, Vec<String>, usize, Vec<String>)>,
}

impl EvalReport {
    /// First expected file surfaced in `ranked`, else `top_k + 1`.
    fn first_hit_rank(expected: &[String], ranked: &[RetrievedFile]) -> usize {
        for (i, r) in ranked.iter().enumerate() {
            if expected.iter().any(|p| p == &r.path) {
                return i + 1;
            }
        }
        usize::MAX
    }
}

/// The built-in sanity gold set for the Xencode repo — real queries about this
/// codebase whose answers are unambiguous.
pub fn default_gold() -> Vec<EvalItem> {
    let run: EvalRun =
        serde_json::from_str(include_str!("eval/gold.json")).expect("built-in gold set must parse");
    run.items
}

/// Load the gold set from `.xencode/eval/gold.json`, falling back to the
/// built-in sample.
pub fn gold_from_disk(xencode_dir: &std::path::Path) -> Vec<EvalItem> {
    let path = xencode_dir.join("eval").join("gold.json");
    if let Ok(text) = std::fs::read_to_string(&path) {
        if let Ok(run) = serde_json::from_str::<EvalRun>(&text) {
            return run.items;
        }
    }
    default_gold()
}

/// Run the gold set through retrieval and measure recall@k / precision@k / MRR.
///
/// `rerank` selects the arm: `false` is the deterministic structural pipeline,
/// `true` runs the same pipeline with the lexical arm added at the candidate
/// stage (`RetrieveOptions::lexical`), which is what would ship. The A/B is
/// deliberately option-driven rather than global so both arms can be measured
/// in one process, with and without documentation text.
pub fn evaluate_with(
    index: &RetrievalIndex,
    gold: &[EvalItem],
    top_k: usize,
    dirty: &HashSet<String>,
    options: &RetrieveOptions,
) -> EvalReport {
    let opts = RetrieveOptions {
        top_k,
        ..options.clone()
    };
    let mut recall = vec![0.0; top_k];
    let mut precision = vec![0.0; top_k];
    let mut mrr_numer = 0.0;
    let mut mrr_queries = 0usize;
    let mut hits: Vec<(String, Vec<String>, usize, Vec<String>)> = Vec::new();

    for item in gold {
        let ranked = retrieve(&item.query, index, dirty, &opts);
        let ranked_paths: Vec<String> = ranked.iter().map(|r| r.path.clone()).collect();
        if item.expected.is_empty() {
            hits.push((
                item.query.clone(),
                item.expected.clone(),
                EvalReport::first_hit_rank(&item.expected, &ranked),
                ranked_paths,
            ));
            continue;
        }
        // recall@k / precision@k: a query's expectation is satisfied if any
        // expected file appears in the top-k.
        for k in 1..=top_k {
            let window: Vec<String> = ranked.iter().take(k).map(|r| r.path.clone()).collect();
            let hits_expected = item
                .expected
                .iter()
                .filter(|p| window.iter().any(|w| w == *p))
                .count();
            recall[k - 1] += if hits_expected > 0 { 1.0 } else { 0.0 };
            precision[k - 1] += hits_expected as f64 / k as f64;
        }
        mrr_queries += 1;
        let rank = EvalReport::first_hit_rank(&item.expected, &ranked);
        if rank != usize::MAX {
            mrr_numer += 1.0 / rank as f64;
        }
        hits.push((
            item.query.clone(),
            item.expected.clone(),
            rank,
            ranked_paths,
        ));
    }

    let n = gold.len().max(1) as f64;
    let with_expected = gold
        .iter()
        .filter(|g| !g.expected.is_empty())
        .count()
        .max(1) as f64;
    EvalReport {
        queries: gold.len(),
        top_k,
        reranked: opts.lexical,
        recall_at: recall.iter().map(|v| v / n).collect(),
        precision_at: precision.iter().map(|v| v / with_expected).collect(),
        mrr: if mrr_queries > 0 {
            mrr_numer / mrr_queries as f64
        } else {
            0.0
        },
        hits,
    }
}

/// Run the gold set through either the deterministic pipeline or, with
/// `rerank`, the hybrid one that scores candidates with BM25 — including each
/// file's documentation prose — before the top-K is cut.
pub fn evaluate(
    index: &RetrievalIndex,
    gold: &[EvalItem],
    top_k: usize,
    dirty: &HashSet<String>,
    rerank: bool,
) -> EvalReport {
    evaluate_with(
        index,
        gold,
        top_k,
        dirty,
        &RetrieveOptions {
            lexical: rerank,
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::FileEntry;
    use crate::symbols::PerFileSymbols;

    fn file(path: &str) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            language: "rust".to_string(),
            size: 0,
            loc: 0,
            ext: "rs".to_string(),
            important: false,
            secret: false,
            binary: false,
        }
    }

    fn sample_index() -> RetrievalIndex {
        let mut idx = RetrievalIndex {
            files: vec![
                file("crates/xencode-context-rs/src/retrieve.rs"),
                file("crates/xencode-context-rs/src/budget.rs"),
                file("crates/xencode-context-rs/src/stale.rs"),
                file("crates/xencode-tui-rs/src/app.rs"),
            ],
            ..Default::default()
        };
        idx.symbols.insert(
            "crates/xencode-context-rs/src/retrieve.rs".into(),
            PerFileSymbols {
                structs: vec![
                    "RetrievalIndex".into(),
                    "RetrieveOptions".into(),
                    "RetrievedFile".into(),
                ],
                functions: vec!["retrieve".into(), "word_tokens".into()],
                imports: vec![],
                exports: vec![],
                mods: vec![],
                ..Default::default()
            },
        );
        idx.symbols.insert(
            "crates/xencode-context-rs/src/budget.rs".into(),
            PerFileSymbols {
                structs: vec!["HardwareProfile".into()],
                functions: vec!["est_tokens".into(), "truncate_to_tokens".into()],
                imports: vec![],
                exports: vec![],
                mods: vec![],
                ..Default::default()
            },
        );
        idx.symbols.insert(
            "crates/xencode-tui-rs/src/app.rs".into(),
            PerFileSymbols {
                structs: vec!["App".into()],
                functions: vec!["submit_message".into()],
                imports: vec![],
                exports: vec![],
                mods: vec![],
                ..Default::default()
            },
        );
        idx
    }

    fn gold() -> Vec<EvalItem> {
        vec![
            EvalItem {
                query: "retrieve top files by score".to_string(),
                expected: vec!["crates/xencode-context-rs/src/retrieve.rs".to_string()],
            },
            EvalItem {
                query: "token budget truncation estimates".to_string(),
                expected: vec!["crates/xencode-context-rs/src/budget.rs".to_string()],
            },
            EvalItem {
                query: "submit the chat message".to_string(),
                expected: vec!["crates/xencode-tui-rs/src/app.rs".to_string()],
            },
        ]
    }

    #[test]
    fn evaluate_scores_recall_and_mrr_on_gold() {
        let idx = sample_index();
        let dirty = HashSet::new();
        let rep = evaluate(&idx, &gold(), 5, &dirty, false);
        assert_eq!(rep.queries, 3);
        // Every query finds its answer.
        assert_eq!(rep.recall_at[0], 1.0);
        assert_eq!(rep.mrr, 1.0);
        assert_eq!(rep.precision_at[0], 1.0);
    }

    #[test]
    fn evaluate_ranks_misses_as_zero_reciprocal() {
        let idx = sample_index();
        let gold = vec![EvalItem {
            query: "unrelated websocket framing".to_string(),
            expected: vec!["crates/nope.rs".to_string()],
        }];
        let rep = evaluate(&idx, &gold, 3, &HashSet::new(), false);
        assert_eq!(rep.mrr, 0.0);
        assert_eq!(rep.recall_at[0], 0.0);
        assert_eq!(rep.hits[0].2, usize::MAX);
    }

    #[test]
    fn rerank_can_only_improve_mrr_here() {
        let idx = sample_index();
        let base = evaluate(&idx, &gold(), 5, &HashSet::new(), false);
        let reranked = evaluate(&idx, &gold(), 5, &HashSet::new(), true);
        assert!(reranked.mrr >= base.mrr);
    }

    #[test]
    fn every_gold_answer_is_reachable_from_its_own_file() {
        // The corpus must not rot in either direction: each expected path has to
        // be a real file in this workspace (`cmd_output.rs` was not, and a gold
        // entry naming a file that does not exist can never be hit, so it drags
        // every score down while looking like a retrieval failure), and its
        // query has to describe that file well enough for the deterministic
        // signals — its own name plus the symbols it declares — to put it first.
        //
        // The haystack is the answer plus three unrelated files. Being first
        // here says the pairing is sound; it deliberately says nothing about
        // whether the answer survives a whole repo, which is the part the
        // measurement in `tests/gold_baseline.rs` reports.
        let root = workspace_root();
        let gold = default_gold();
        assert!(
            gold.len() >= 16,
            "the built-in corpus widened once; it must not shrink back"
        );

        for item in &gold {
            assert_eq!(
                item.expected.len(),
                1,
                "one answer per probe: {}",
                item.query
            );
            let path = &item.expected[0];
            let on_disk = root.join(path);
            assert!(
                on_disk.is_file(),
                "gold entry points at a missing file: {path}"
            );

            let symbols = crate::extract_rust_symbols(
                &std::fs::read_to_string(&on_disk)
                    .unwrap_or_else(|e| panic!("unreadable gold answer {path}: {e}")),
            );

            let mut idx = RetrievalIndex {
                files: [
                    "rust/crates/xencode-core-rs/src/tasks.rs",
                    "rust/crates/xencode-tui-rs/src/focus.rs",
                    "rust/crates/xencode-analysis-rs/src/web.rs",
                ]
                .into_iter()
                .map(file)
                .chain(std::iter::once(file(path)))
                .collect(),
                ..Default::default()
            };
            idx.symbols.insert(path.clone(), symbols);

            let rep = evaluate(&idx, std::slice::from_ref(item), 5, &HashSet::new(), false);
            assert_eq!(
                rep.hits[0].2, 1,
                "query {:?} does not describe {}; ranked {:?}",
                item.query, path, rep.hits[0].3
            );
        }
    }

    /// This workspace's root, three levels up from the crate directory.
    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("crate sits at <root>/rust/crates/<name>")
            .to_path_buf()
    }
}
