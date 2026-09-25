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
    /// What kind of work this probe is a request for, as the word
    /// [`crate::TaskShape`] prints it — `bugfix`, or nothing at all. Left out, the probe
    /// is run with whatever shape the arm being measured carries; a gold query is
    /// a prompt without a session behind it, so most probes honestly have no
    /// label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
}

impl EvalItem {
    /// The labelled shape, or `None` when the probe carries no label. A label
    /// that is not a shape — a typo, or a word from a future version of the
    /// table — reads as no label too, so one bad row cannot end the run; the
    /// built-in set is checked word by word by a test.
    pub fn task_shape(&self) -> Option<crate::TaskShape> {
        self.shape.as_deref().and_then(crate::TaskShape::parse)
    }

    /// The same probe with its label removed, which is how a measurement asks
    /// the retriever to score it as though the shape had never been read.
    pub fn without_shape(&self) -> Self {
        Self {
            shape: None,
            ..self.clone()
        }
    }
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
        // A probe's own label wins over the arm's shape, because the label is
        // the claim being tested: "a bug-report-shaped query retrieves better
        // with the bugfix weights". A probe with no label is scored with
        // whatever shape the arm carries.
        let item_options = RetrieveOptions {
            shape: item.task_shape().unwrap_or(opts.shape),
            ..opts.clone()
        };
        let ranked = retrieve(&item.query, index, dirty, &item_options);
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

/// The gold set split by the shape each probe was written as, in a fixed order.
///
/// Probes carrying no label are `general` ones: the weight table as it stands is
/// exactly what they ask for, so they belong to the partition that changes
/// nothing rather than being left out of the measurement.
pub fn gold_by_shape(gold: &[EvalItem]) -> Vec<(crate::TaskShape, Vec<EvalItem>)> {
    let mut partitions: Vec<(crate::TaskShape, Vec<EvalItem>)> =
        [crate::TaskShape::General, crate::TaskShape::Bugfix]
            .into_iter()
            .map(|shape| (shape, Vec::new()))
            .collect();
    for item in gold {
        let shape = item.task_shape().unwrap_or(crate::TaskShape::General);
        if let Some((_, bucket)) = partitions.iter_mut().find(|(s, _)| *s == shape) {
            bucket.push(item.clone());
        }
    }
    partitions.retain(|(_, bucket)| !bucket.is_empty());
    partitions
}

/// One shape's gold probes measured twice: once with the shape's weights turned
/// off, once with them on.
#[derive(Debug, Clone)]
pub struct ShapeComparison {
    pub shape: crate::TaskShape,
    pub queries: usize,
    /// Every probe scored with `general` weights, labels ignored.
    pub untuned: EvalReport,
    /// Every probe scored as its own label says, which is what the product does.
    pub tuned: EvalReport,
}

impl ShapeComparison {
    /// What the shape's weights did to mean reciprocal rank on its own probes.
    pub fn mrr_delta(&self) -> f64 {
        self.tuned.mrr - self.untuned.mrr
    }
}

/// Measure each shape on its own probes, against the same probes with the shape
/// weights switched off.
///
/// This is the check a bias has to pass before it ships. The plan's warning was
/// that a per-shape weight table is a story unless the gold set can be
/// partitioned by shape and measured per partition, because an overall number
/// can rise on one kind of query while the shape quietly damages another.
/// `options` is the arm under test — the lexical stage on or off — and is held
/// identical across both sides of every comparison, so the only difference
/// between `untuned` and `tuned` is the shape.
pub fn compare_shapes(
    index: &RetrievalIndex,
    gold: &[EvalItem],
    top_k: usize,
    dirty: &HashSet<String>,
    options: &RetrieveOptions,
) -> Vec<ShapeComparison> {
    gold_by_shape(gold)
        .into_iter()
        .map(|(shape, items)| {
            let queries = items.len();
            let untuned_items: Vec<EvalItem> = items.iter().map(EvalItem::without_shape).collect();
            let untuned = evaluate_with(
                index,
                &untuned_items,
                top_k,
                dirty,
                &RetrieveOptions {
                    shape: crate::TaskShape::General,
                    ..options.clone()
                },
            );
            let tuned = evaluate_with(
                index,
                &items,
                top_k,
                dirty,
                &RetrieveOptions {
                    shape,
                    ..options.clone()
                },
            );
            ShapeComparison {
                shape,
                queries,
                untuned,
                tuned,
            }
        })
        .collect()
}

/// One retrieval-eval run as recorded on disk.
///
/// Eval numbers only mean something next to the two things that produced them:
/// the gold set and the instructions the build carries. A score stored on its own
/// invites the next person to compare it with a run whose prompts had been edited
/// in between, and call the difference a retrieval change.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalRunRecord {
    /// UTC epoch millis when the run finished.
    pub ts_unix_ms: u64,
    /// Which arm was measured, in the label the runner gave it.
    pub arm: String,
    pub top_k: usize,
    /// How many gold queries were asked.
    pub queries: usize,
    /// The prompt-set digest this build sent, from [`crate::prompts`].
    pub prompt_version: String,
    pub mrr: f64,
    /// recall@k for k = 1..=top_k, same shape as [`EvalReport::recall_at`].
    #[serde(default)]
    pub recall_at: Vec<f64>,
}

impl EvalRunRecord {
    /// One row for a finished arm. The prompt version is taken from the build
    /// rather than passed in, because the build is the only honest source for it:
    /// a caller that got it wrong would record a comparison that cannot be made.
    pub fn from_report(arm: &str, report: &EvalReport) -> Self {
        Self {
            ts_unix_ms: crate::conversation::now_millis(),
            arm: arm.to_string(),
            top_k: report.top_k,
            queries: report.queries,
            prompt_version: crate::prompts::set_version().to_string(),
            mrr: report.mrr,
            recall_at: report.recall_at.clone(),
        }
    }
}

pub fn eval_log_path(xencode_dir: &std::path::Path) -> std::path::PathBuf {
    xencode_dir.join("cache").join("eval.jsonl")
}

/// Append one run to `cache/eval.jsonl`. Nothing is trimmed: the point of the log
/// is that a score from an older build is still there to be compared, or refused.
pub fn append_eval_run(
    xencode_dir: &std::path::Path,
    record: &EvalRunRecord,
) -> std::io::Result<()> {
    use std::io::Write;
    let path = eval_log_path(xencode_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writeln!(file, "{}", serde_json::to_string(record)?)
}

/// Every recorded run, oldest first. A line that does not parse is skipped rather
/// than ending the read, so one interrupted write cannot hide the history.
pub fn read_eval_runs(xencode_dir: &std::path::Path) -> Vec<EvalRunRecord> {
    xencode_core_rs::read_jsonl_tolerant(&eval_log_path(xencode_dir)).rows
}

/// The newest recorded run of `arm` at `top_k`, whatever its prompts were — which
/// is why the caller is shown the version rather than being handed a delta.
pub fn previous_eval_run<'a>(
    runs: &'a [EvalRunRecord],
    arm: &str,
    top_k: usize,
) -> Option<&'a EvalRunRecord> {
    runs.iter().rev().find(|r| r.top_k == top_k && r.arm == arm)
}

/// The newest recorded run of `arm` at `top_k` that a score can honestly be
/// compared with: same prompts, or nothing.
pub fn comparable_previous_eval_run<'a>(
    runs: &'a [EvalRunRecord],
    arm: &str,
    top_k: usize,
    prompt_version: &str,
) -> Option<&'a EvalRunRecord> {
    runs.iter()
        .rev()
        .find(|r| r.top_k == top_k && r.arm == arm && r.prompt_version == prompt_version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::FileEntry;
    use crate::symbols::PerFileSymbols;
    use crate::TaskShape;

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
        let probe = |query: &str, expected: &str| EvalItem {
            query: query.to_string(),
            expected: vec![expected.to_string()],
            shape: None,
        };
        vec![
            probe(
                "retrieve top files by score",
                "crates/xencode-context-rs/src/retrieve.rs",
            ),
            probe(
                "token budget truncation estimates",
                "crates/xencode-context-rs/src/budget.rs",
            ),
            probe(
                "submit the chat message",
                "crates/xencode-tui-rs/src/app.rs",
            ),
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
            shape: None,
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
    fn a_label_that_is_not_a_shape_is_no_label() {
        let item = EvalItem {
            query: "harden the token check".to_string(),
            expected: vec![],
            shape: Some("secure".to_string()),
        };
        // `secure` is a word the plan uses for a shape nobody has priced. A gold
        // set carrying a future label must not stop the run over it.
        assert_eq!(item.task_shape(), None);
        assert_eq!(item.without_shape().shape, None);
        assert_eq!(
            EvalItem {
                query: String::new(),
                expected: vec![],
                shape: Some("bugfix".to_string()),
            }
            .task_shape(),
            Some(crate::TaskShape::Bugfix)
        );
    }

    #[test]
    fn a_shape_bias_is_measured_as_a_bias_and_not_as_a_different_retriever() {
        // Both files match the query identically on path and symbol; only one
        // holds a test whose name says what the prompt says. So the difference
        // between the two arms can only be the shape — which is the whole point
        // of `compare_shapes`, and the reason its inputs are held fixed.
        let idx = RetrievalIndex {
            files: vec![file("src/a_tail.rs"), file("src/z_tail.rs")],
            symbols: std::collections::BTreeMap::from([
                (
                    "src/a_tail.rs".to_string(),
                    PerFileSymbols {
                        functions: vec!["tail".to_string()],
                        ..Default::default()
                    },
                ),
                (
                    "src/z_tail.rs".to_string(),
                    PerFileSymbols {
                        functions: vec!["tail".to_string()],
                        tests: vec!["tail_keeps_the_last_line".to_string()],
                        ..Default::default()
                    },
                ),
            ]),
            ..Default::default()
        };
        let bugfix = EvalItem {
            query: "tail keeps the last line fix".to_string(),
            expected: vec!["src/z_tail.rs".to_string()],
            shape: Some("bugfix".to_string()),
        };
        let plain = EvalItem {
            query: "the tail of a file".to_string(),
            expected: vec!["src/a_tail.rs".to_string()],
            shape: None,
        };
        let comparisons = structural_comparison(&idx, &[bugfix, plain]);
        let shaped = comparisons
            .iter()
            .find(|c| c.shape == crate::TaskShape::Bugfix)
            .expect("the bugfix partition");
        assert_eq!(shaped.queries, 1);
        // Without the bias the tied pair is ordered by path, so the answer is second.
        assert_eq!(shaped.untuned.mrr, 0.5);
        assert_eq!(shaped.tuned.mrr, 1.0);
        assert!(shaped.mrr_delta() > 0.0);
        // The control: probes with no label are the `general` partition, whose
        // weights are the table as it stands. A nonzero delta here would mean the
        // two arms differed in something besides the shape.
        let control = comparisons
            .iter()
            .find(|c| c.shape == crate::TaskShape::General)
            .expect("the general partition");
        assert_eq!(control.untuned.mrr, 1.0, "the control should retrieve");
        assert_eq!(control.mrr_delta(), 0.0);
    }

    /// `compare_shapes` under the structural arm, which is what a unit test can
    /// hold fixed — the lexical arm reaches into file text on disk.
    fn structural_comparison(index: &RetrievalIndex, gold: &[EvalItem]) -> Vec<ShapeComparison> {
        compare_shapes(index, gold, 5, &HashSet::new(), &RetrieveOptions::default())
    }

    #[test]
    fn every_probe_is_a_shape_the_words_in_the_probe_would_also_give() {
        // A label the retriever would never reach on its own measures a pairing
        // the product cannot produce, so a gain on it would be a gain in a
        // fiction. Every probe is therefore checked against the reading a prompt
        // gets on its own — including the unlabelled ones, which must really read
        // as `general`, or the general partition would quietly contain shaped
        // queries measured with the wrong weights.
        let gold = default_gold();
        let mut labelled = 0;
        for item in &gold {
            let read = crate::shape_of(&item.query);
            let shape = match &item.shape {
                None => TaskShape::General,
                Some(word) => {
                    labelled += 1;
                    item.task_shape()
                        .unwrap_or_else(|| panic!("unknown shape label {word:?} in {}", item.query))
                }
            };
            assert_eq!(
                read.shape, shape,
                "probe {:?} is labelled {shape} but its own words read as {} ({:?})",
                item.query, read.shape, read.reasons
            );
        }
        assert!(
            labelled >= 4,
            "the built-in set must be partitionable, not merely labelled: {labelled} of {} \
             probes carry a shape",
            gold.len()
        );
        assert!(
            gold.iter()
                .any(|i| i.task_shape() == Some(TaskShape::Bugfix)),
            "no probe asks about bugfix work, so the one priced bias has no partition to be \
             measured on"
        );
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
        // here says the pairing is sound: the probe names something the answer
        // file carries, its declared symbols or — for a prose file, which
        // declares none — the fact that the project keeps its rules and manuals
        // in it. It deliberately says nothing about whether the answer survives a
        // whole repo, which is the part the measurement in
        // `tests/gold_baseline.rs` reports.
        let root = workspace_root();
        let gold = default_gold();
        assert!(
            gold.len() >= 24,
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

    fn record(arm: &str, top_k: usize, prompt_version: &str, mrr: f64, ts: u64) -> EvalRunRecord {
        EvalRunRecord {
            ts_unix_ms: ts,
            arm: arm.to_string(),
            top_k,
            queries: 12,
            prompt_version: prompt_version.to_string(),
            mrr,
            recall_at: vec![mrr],
        }
    }

    #[test]
    fn a_run_written_to_disk_comes_back_and_carries_its_prompts() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("xencode-eval-log-test-{unique}"));
        let older = record("hybrid", 5, "aaaaaaaaaaaa", 0.41, 1);
        let newer = record("hybrid", 5, "bbbbbbbbbbbb", 0.38, 2);
        append_eval_run(&dir, &older).unwrap();

        let runs = read_eval_runs(&dir);
        assert_eq!(
            runs,
            vec![older.clone()],
            "the log reads back what was written"
        );
        // The first run of a new prompt set has nothing to be compared against:
        // the caller must say so rather than reach past the prompt change.
        assert_eq!(
            comparable_previous_eval_run(&runs, "hybrid", 5, "bbbbbbbbbbbb"),
            None,
            "a run under different prompts was offered as comparable"
        );
        assert_eq!(previous_eval_run(&runs, "hybrid", 5), Some(&older));

        append_eval_run(&dir, &newer).unwrap();
        let runs = read_eval_runs(&dir);
        assert_eq!(runs.len(), 2, "appends accumulate instead of replacing");
        assert_eq!(runs[1], newer);
        assert_eq!(runs[1].recall_at, vec![0.38]);
        // Same arm and depth, older prompts: still findable, so the version can be
        // reported to the user as the reason a delta was withheld.
        assert_eq!(
            comparable_previous_eval_run(&runs, "hybrid", 5, "aaaaaaaaaaaa"),
            Some(&older)
        );
        // A different arm or depth is a different measurement, not a delta.
        assert_eq!(previous_eval_run(&runs, "structural", 5), None);
        assert_eq!(previous_eval_run(&runs, "hybrid", 10), None);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
