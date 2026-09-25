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
//!
//! Each arm is also appended to `.xencode/cache/eval.jsonl` with the prompt-set
//! digest this build carries, so a later run can say whether a difference came
//! from the retriever or from a prompt edit — and refuse to compare when it was
//! the prompts that moved.
//!
//! After the arms, the per-shape biases are measured probe-partition by
//! probe-partition against the same probes with the bias off, which is the only
//! way a weight that helps one kind of turn and hurts another can be seen.

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
    // What this build tells a model to do, as one digest. Retrieval does not read
    // the prompts, so a score only means something next to the gold set and this
    // digest — a later run can tell whether the difference is the retriever.
    let prompts_now = xencode_context_rs::prompts::set_version();
    println!(
        "prompt set {prompts_now} ({} prompts)",
        xencode_context_rs::prompts::registry().len()
    );
    let prior = xencode_context_rs::read_eval_runs(&xencode);
    // Three arms, so the run prices each addition separately: structural only,
    // plus a lexical arm over path and symbols, plus the same with the files'
    // documentation prose indexed too. A probe that carries a `shape` is scored
    // as that shape in every arm, so what is compared here is the lexical stage
    // and nothing else; the shape's own contribution is measured per partition
    // below.
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
        // The comparison is with the newest earlier run of this arm at this depth
        // that was measured under the same prompt set. Anything else is stated as
        // not comparable rather than quietly reported as progress.
        let label = label.trim();
        match xencode_context_rs::comparable_previous_eval_run(&prior, label, 5, prompts_now) {
            Some(prev) => println!(
                "  {label}  MRR {:.3} → {:.3} ({:+.3}) under prompts {prompts_now}",
                prev.mrr,
                report.mrr,
                report.mrr - prev.mrr
            ),
            None => {
                let why = match xencode_context_rs::previous_eval_run(&prior, label, 5) {
                    Some(other) => format!(
                        "the last run of this arm used prompts {}",
                        other.prompt_version
                    ),
                    None => "no earlier run of this arm is recorded".to_string(),
                };
                println!("  {label}  no comparison available — {why}");
            }
        }
        for (query, _, rank, _) in &report.hits {
            let shown = if *rank == usize::MAX {
                "outside top 5".to_string()
            } else {
                format!("rank {rank}")
            };
            println!("  {shown:>14}  {query}");
        }
        xencode_context_rs::append_eval_run(
            &xencode,
            &xencode_context_rs::EvalRunRecord::from_report(label, &report),
        )
        .expect("write the eval run to the log");
    }
    println!(
        "recorded to {}",
        xencode_context_rs::eval_log_path(&xencode).display()
    );
    // The shape biases last, on both arms. Each shape is scored against the same
    // probes with its own weight turned off, so a bias that only pays for itself
    // on one kind of turn cannot hide inside an overall average — and a bias that
    // pays for nothing shows up as 0.000 rather than as a rumour. The structural
    // arm is measured too because the hybrid arm already solves most probes, and
    // a bias that can only help where the stronger arm is not running is a
    // different claim from one that helps in the shipped configuration.
    let tests_indexed: usize = index.symbols.values().map(|s| s.tests.len()).sum();
    let files_with_tests = index
        .symbols
        .values()
        .filter(|s| !s.tests.is_empty())
        .count();
    println!(
        "\nshape biases, per partition ({} test names over {files_with_tests} files indexed):",
        tests_indexed
    );
    let arms_by_name = [
        (
            "deterministic",
            xencode_context_rs::RetrieveOptions::default(),
        ),
        (
            "+ text + doc prose",
            xencode_context_rs::RetrieveOptions {
                lexical: true,
                lexical_docs: true,
                ..Default::default()
            },
        ),
    ];
    for (arm, opts) in &arms_by_name {
        println!("  on the {arm} arm:");
        for part in &xencode_context_rs::compare_shapes(&index, &gold, 5, &no_changes, opts) {
            if part.shape == xencode_context_rs::TaskShape::General {
                println!(
                    "    {:<8} {:>2} probes · control, no weight moves: MRR {:.3}",
                    part.shape.as_str(),
                    part.queries,
                    part.tuned.mrr
                );
                continue;
            }
            println!(
                "    {:<8} {:>2} probes · MRR {:.3} → {:.3} ({:+.3}) with the bias",
                part.shape.as_str(),
                part.queries,
                part.untuned.mrr,
                part.tuned.mrr,
                part.mrr_delta()
            );
        }
    }
}
