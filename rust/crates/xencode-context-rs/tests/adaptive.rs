//! Adaptive context decisions against real files (XI-3).
//!
//! The relationship cases use the checked-in Rust fixtures from XI-2: their
//! symbols, tests and dependency edges come from the same extractor and graph
//! builder the product uses, and the history comes from the same `git log`
//! parser. The benchmark case uses the checked-in gold probes and reads their
//! answer files off this workspace. Nothing here is a canned answer keyed on a
//! query string.
//!
//! ```text
//! cargo test -p xencode-context-rs --test adaptive -- --nocapture
//! ```

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use xencode_context_rs::budget::HardwareProfile;
use xencode_context_rs::cochange::LOG_MARKER;
use xencode_context_rs::{
    assemble_chat, assemble_chat_adaptive, build_graph, decide, extract_rust_symbols,
    impact_evidence, parse_commit_log, retrieve, shape_of, ContextCaps, FileEntry, ImpactEvidence,
    PerFileSymbols, Relation, RelationKind, RelationScope, RelationSource, RetrievalIndex,
    RetrieveOptions, RetrievedBlock, RetrievedFile, TaskShape, MASS_COMMIT_FILES, TOP_PARTNERS,
};

const FIXTURE_FILES: [(&str, &str); 5] = [
    ("src/lib.rs", include_str!("fixtures/relationships/lib.rs")),
    (
        "src/traits.rs",
        include_str!("fixtures/relationships/traits.rs"),
    ),
    (
        "src/store.rs",
        include_str!("fixtures/relationships/store.rs"),
    ),
    (
        "src/service.rs",
        include_str!("fixtures/relationships/service.rs"),
    ),
    (
        "src/view.rs",
        include_str!("fixtures/relationships/view.rs"),
    ),
];

fn fixture_index() -> RetrievalIndex {
    let symbols: BTreeMap<String, PerFileSymbols> = FIXTURE_FILES
        .iter()
        .map(|(path, text)| (path.to_string(), extract_rust_symbols(text)))
        .collect();
    let rust_files: Vec<String> = FIXTURE_FILES.iter().map(|(p, _)| p.to_string()).collect();
    let deps = build_graph(&rust_files, &symbols);
    let log = format!(
        "{m}1000\n\nsrc/store.rs\nsrc/service.rs\n\
         {m}2000\n\nsrc/store.rs\nsrc/lib.rs\n\
         {m}3000\n\nsrc/view.rs\nsrc/traits.rs\n",
        m = LOG_MARKER
    );
    let history = parse_commit_log(&log, LOG_MARKER, MASS_COMMIT_FILES, TOP_PARTNERS);
    RetrievalIndex {
        files: FIXTURE_FILES
            .iter()
            .map(|(path, _)| FileEntry {
                path: path.to_string(),
                language: "rust".to_string(),
                size: 0,
                loc: 0,
                ext: "rs".to_string(),
                important: false,
                secret: false,
                binary: false,
            })
            .collect(),
        symbols,
        deps,
        mtimes: BTreeMap::new(),
        history,
    }
}

fn fixture_body(path: &str) -> String {
    let text = FIXTURE_FILES
        .iter()
        .find(|(p, _)| *p == path)
        .map(|(_, text)| *text)
        .unwrap_or("// no fixture body");
    format!("File: {path}\n```rust\n{text}\n```")
}

fn fixture_blocks(paths: &[&str]) -> Vec<RetrievedBlock> {
    paths
        .iter()
        .enumerate()
        .map(|(i, path)| RetrievedBlock {
            path: path.to_string(),
            score: (10 - i) as u64,
            body: fixture_body(path),
        })
        .collect()
}

fn caps() -> ContextCaps {
    ContextCaps::from_profile(HardwareProfile::Low)
}

fn input_with(
    retrieved: Vec<RetrievedBlock>,
    prompt: &'static str,
    history: &'static [(String, String)],
) -> xencode_context_rs::ChatInput<'static> {
    xencode_context_rs::ChatInput {
        profile: HardwareProfile::Low,
        context_window: None,
        system: "You are a coding agent. Be concise.",
        agents_md: Some("# Rules\n- Rust first\n"),
        anchor_md: Some("# Anchor\n"),
        scoped_md: None,
        state_md: None,
        notes_md: None,
        git_summary: "",
        repo_map: "",
        retrieved,
        attached_block: "",
        history,
        prompt,
    }
}

#[test]
fn a_bugfix_plan_surfaces_real_xi2_relations() {
    let index = fixture_index();
    // The changed file is a real fixture with a dependent (service.rs) and two
    // tests that name its symbols, all derived from the indexes above.
    let evidence = impact_evidence(&index, "src/store.rs");
    assert!(evidence
        .dependents
        .iter()
        .any(|r| r.object == "src/service.rs"));
    assert!(evidence.tests.iter().any(|r| r.object == "save"));

    let query = "fix the broken save";
    let retrieved = retrieve(
        query,
        &index,
        &HashSet::new(),
        &RetrieveOptions::for_live_chat(3, TaskShape::Bugfix),
    );
    let plan = decide(
        shape_of(query).shape,
        query,
        caps(),
        &retrieved,
        Some(&evidence),
    );
    assert_eq!(
        plan.order,
        xencode_context_rs::RetrievalOrder::EvidenceFirst
    );
    assert!(plan.dependents >= 1, "{plan:?}");
    assert!(plan.symbol_facts >= 1, "{plan:?}");
    assert!(plan.include_tests);

    let applied = assemble_chat_adaptive(
        input_with(
            fixture_blocks(&["src/store.rs", "src/view.rs", "src/lib.rs"]),
            query,
            &[],
        ),
        &plan,
        Some(&evidence),
    );
    assert!(applied.evidence_added);
    let text = applied.assembly.prompt_text();
    assert!(text.contains("Impact evidence"), "{text}");
    assert!(text.contains("dependent: src/service.rs"), "{text}");
    assert!(
        text.contains("test: save_persists_value names save"),
        "{text}"
    );
    assert!(
        text.contains("affected_dependents over deps.json"),
        "{text}"
    );
    assert!(text.contains("symbols.json tests"), "{text}");
    println!(
        "bugfix evidence block sourced from 2 real relations; {} retrieved body/ies kept",
        applied.retrieval_kept
    );
}

#[test]
fn the_stable_prefix_is_unchanged_and_redaction_still_applies() {
    let index = fixture_index();
    let evidence = impact_evidence(&index, "src/store.rs");
    let query = "fix the broken save";
    let retrieved = retrieve(
        query,
        &index,
        &HashSet::new(),
        &RetrieveOptions::for_live_chat(3, TaskShape::Bugfix),
    );
    let plan = decide(
        TaskShape::Bugfix,
        query,
        caps(),
        &retrieved,
        Some(&evidence),
    );
    assert!(
        plan.retrieval_depth >= 1,
        "the secret-bearing block must be kept"
    );

    let mut blocks = fixture_blocks(&["src/store.rs", "src/view.rs"]);
    // An approved-shaped fake secret: it trips our own detector but sits off a
    // real vendor signature. It must not reach the model through any tier.
    blocks[0]
        .body
        .push_str("\n// key = sk-FAKE-NOT-A-REAL-TEST-KEY\n");

    let baseline = assemble_chat(input_with(blocks.clone(), query, &[]));
    let applied = assemble_chat_adaptive(input_with(blocks, query, &[]), &plan, Some(&evidence));

    assert_eq!(baseline.turns[0].content, applied.assembly.turns[0].content);
    assert!(!baseline.turns[0].content.contains("sk-FAKE-NOT-A-REAL"));
    let adaptive_text = applied.assembly.prompt_text();
    assert!(
        !adaptive_text.contains("sk-FAKE-NOT-A-REAL"),
        "{adaptive_text}"
    );
    assert!(!applied.assembly.vault.is_empty());
    assert!(applied.assembly.vault.placeholders().count() >= 1);
    println!(
        "stable prefix byte-identical ({} chars); secret replaced in dynamic tiers",
        applied.assembly.turns[0].content.len()
    );
}

#[test]
fn the_same_inputs_produce_the_same_plan_and_the_same_turns() {
    let index = fixture_index();
    let evidence = impact_evidence(&index, "src/store.rs");
    let query = "the save is broken";
    let retrieved = retrieve(
        query,
        &index,
        &HashSet::new(),
        &RetrieveOptions::for_live_chat(3, TaskShape::Bugfix),
    );
    let a = decide(
        TaskShape::Bugfix,
        query,
        caps(),
        &retrieved,
        Some(&evidence),
    );
    let b = decide(
        TaskShape::Bugfix,
        query,
        caps(),
        &retrieved,
        Some(&evidence),
    );
    assert_eq!(a, b);

    let blocks = fixture_blocks(&["src/store.rs", "src/service.rs"]);
    let x = assemble_chat_adaptive(input_with(blocks.clone(), query, &[]), &a, Some(&evidence));
    let y = assemble_chat_adaptive(input_with(blocks, query, &[]), &b, Some(&evidence));
    assert_eq!(x.assembly.turns, y.assembly.turns);
    assert_eq!(x.assembly.total_tokens, y.assembly.total_tokens);
}

#[test]
fn a_small_context_cannot_over_fetch() {
    let small = ContextCaps {
        top_k: 1,
        content_cap_chars: ContextCaps::MIN_CONTENT_CAP_CHARS,
    };
    let many: Vec<RetrievedFile> = (0..12)
        .map(|i| RetrievedFile {
            path: format!("f{i}.rs"),
            score: 20 - i,
            reasons: vec![],
        })
        .collect();
    let plan = decide(TaskShape::General, "short question", small, &many, None);
    assert_eq!(plan.retrieval_depth, 1);
    let applied = assemble_chat_adaptive(
        input_with(
            fixture_blocks(&["src/store.rs", "src/view.rs"]),
            "short question",
            &[],
        ),
        &plan,
        None,
    );
    assert_eq!(applied.retrieval_kept, 1);
    assert!(!applied.evidence_added);
}

#[test]
fn missing_measurements_are_not_turned_into_edges() {
    let unknown_only: ImpactEvidence = ImpactEvidence {
        changed_file: "src/store.rs".into(),
        dependents: vec![],
        tests: vec![],
        co_changes: vec![],
        unknown: vec![Relation {
            kind: RelationKind::DependedOnBy,
            subject: "src/store.rs".into(),
            object: "<unknown>".into(),
            source: RelationSource::Unknown,
            scope: RelationScope::Unknown,
            evidence: "no indexed Rust file depends on it".into(),
        }],
    };
    let plan = decide(
        TaskShape::Bugfix,
        "fix it",
        caps(),
        &[],
        Some(&unknown_only),
    );
    assert_eq!(plan.order, xencode_context_rs::RetrievalOrder::ByScore);
    assert_eq!(plan.dependents, 0);
    assert_eq!(plan.symbol_facts, 0);
    assert!(plan.evidence.iter().any(|e| e.why.contains("unknown")));
    let applied = assemble_chat_adaptive(
        input_with(fixture_blocks(&["src/store.rs"]), "fix it", &[]),
        &plan,
        Some(&unknown_only),
    );
    assert!(!applied.evidence_added);
}

#[test]
fn evidence_survives_a_tight_budget_because_it_leads_the_retrieval_list() {
    let index = fixture_index();
    let evidence = impact_evidence(&index, "src/store.rs");
    let plan = decide(
        TaskShape::Bugfix,
        "fix the broken save",
        caps(),
        &[],
        Some(&evidence),
    );
    // A long but bounded state tier eats most of the Low window; the compact
    // evidence block is still first in the retrieval list and survives.
    let long_state = "context note line\n".repeat(40);
    let mut input = input_with(
        fixture_blocks(&["src/store.rs", "src/view.rs"]),
        "fix it",
        &[],
    );
    input.state_md = Some(Box::leak(long_state.into_boxed_str()));
    let applied = assemble_chat_adaptive(input, &plan, Some(&evidence));
    let text = applied.assembly.prompt_text();
    assert!(
        text.contains("Impact evidence") && text.contains("dependent: src/"),
        "critical evidence must not be the first thing a tight budget drops: {text}"
    );
}

// ── Honest benchmark: existing assembly vs the adaptive arm ────────────────

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("crate sits at <root>/rust/crates/<name>")
        .to_path_buf()
}

/// Build a real index from the files the checked-in gold answers name. Bodies,
/// symbols and language come off disk; no answer is fabricated.
fn index_of_gold_answers(gold: &[xencode_context_rs::EvalItem]) -> RetrievalIndex {
    let root = workspace_root();
    let mut index = RetrievalIndex::default();
    for item in gold {
        for path in &item.expected {
            if index.files.iter().any(|file| file.path == *path) {
                continue;
            }
            let body = std::fs::read_to_string(root.join(path))
                .unwrap_or_else(|e| panic!("gold answer {path} is unreadable: {e}"));
            let ext = PathBuf::from(path)
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("")
                .to_string();
            index.files.push(FileEntry {
                path: path.clone(),
                language: if ext == "rs" {
                    "rust".to_string()
                } else {
                    ext.clone()
                },
                size: body.len() as u64,
                loc: body.lines().count() as u64,
                ext: ext.clone(),
                important: false,
                secret: false,
                binary: false,
            });
            if ext == "rs" {
                index
                    .symbols
                    .insert(path.clone(), extract_rust_symbols(&body));
            }
        }
    }
    index
}

fn body_for_gold_file(path: &str) -> String {
    let root = workspace_root();
    let text = std::fs::read_to_string(root.join(path)).unwrap_or_default();
    let language = if path.ends_with(".rs") {
        "rust"
    } else {
        "text"
    };
    let kept: String = text.chars().take(2_000).collect();
    format!("File: {path}\n```{language}\n{kept}\n```")
}

///Precision/recall of the files a turn actually injects, judged against the
///probe's expected paths. Counts are per probe, macro-averaged.
#[derive(Default, Clone, Copy)]
struct ContextScore {
    precision: f64,
    recall: f64,
    included: usize,
    tokens: u64,
    micros: f64,
}

fn score_turn(included: &[String], expected: &[String]) -> (f64, f64) {
    let hits = expected.iter().filter(|e| included.contains(e)).count();
    let precision = if included.is_empty() {
        0.0
    } else {
        hits as f64 / included.len() as f64
    };
    let recall = if expected.is_empty() {
        0.0
    } else {
        hits as f64 / expected.len() as f64
    };
    (precision, recall)
}

#[test]
fn adaptive_against_existing_assembly_on_the_gold_set() {
    let gold = xencode_context_rs::default_gold();
    let index = index_of_gold_answers(&gold);
    let opts = RetrieveOptions::for_live_chat(5, TaskShape::General);
    let caps = ContextCaps::from_profile(HardwareProfile::Balanced);
    let mut baseline_total = ContextScore::default();
    let mut adaptive_total = ContextScore::default();
    let mut scored = 0usize;

    for item in &gold {
        let shape = item.task_shape().unwrap_or(TaskShape::General);
        let retrieved = retrieve(
            &item.query,
            &index,
            &HashSet::new(),
            &RetrieveOptions {
                shape,
                ..opts.clone()
            },
        );
        // Real XI-2 evidence for the gold answer's own file, never a canned map.
        let subject = item.expected.first().map(String::as_str).unwrap_or("");
        let evidence = if subject.is_empty() {
            None
        } else {
            Some(impact_evidence(&index, subject))
        };
        let plan = decide(shape, &item.query, caps, &retrieved, evidence.as_ref());

        let blocks: Vec<RetrievedBlock> = retrieved
            .iter()
            .map(|r| RetrievedBlock {
                path: r.path.clone(),
                score: r.score,
                body: body_for_gold_file(&r.path),
            })
            .collect();

        let started = std::time::Instant::now();
        let base = assemble_chat(input_with(blocks.clone(), "gold probe", &[]));
        let base_micros = started.elapsed().as_secs_f64() * 1_000_000.0;
        let started = std::time::Instant::now();
        let adapted = assemble_chat_adaptive(
            input_with(blocks, "gold probe", &[]),
            &plan,
            evidence.as_ref(),
        );
        let adaptive_micros = started.elapsed().as_secs_f64() * 1_000_000.0;

        let base_files: Vec<String> = base.retrieved_files.clone();
        let adapted_files: Vec<String> = adapted
            .assembly
            .retrieved_files
            .iter()
            .filter(|path| path.as_str() != "impact-evidence")
            .cloned()
            .collect();
        let (bp, br) = score_turn(&base_files, &item.expected);
        let (ap, ar) = score_turn(&adapted_files, &item.expected);
        baseline_total.precision += bp;
        baseline_total.recall += br;
        baseline_total.included += base_files.len();
        baseline_total.tokens += base.total_tokens;
        baseline_total.micros += base_micros;
        adaptive_total.precision += ap;
        adaptive_total.recall += ar;
        adaptive_total.included += adapted_files.len();
        adaptive_total.tokens += adapted.assembly.total_tokens;
        adaptive_total.micros += adaptive_micros;
        scored += 1;
    }

    let n = scored.max(1) as f64;
    let report = xencode_context_rs::evaluate_with(&index, &gold, 5, &HashSet::new(), &opts);
    // XI-9 record: one versioned record per arm; compare() still reports the
    // numeric deltas and says plainly that different arms are not one suite.
    let config = xencode_context_rs::BenchmarkConfig {
        model: None,
        provider: None,
        hardware_profile: Some("BALANCED".to_string()),
        top_k: 5,
        seed: Some(0),
        temperature: Some(0.0),
        grader: None,
    };
    let corpus = xencode_context_rs::BenchmarkCorpus::from_checked_in(gold.clone());
    // Three deterministic repeats so the record is structurally complete under
    // XI-9's MIN_REPEATS. The arms are still different configurations, so the
    // comparison must refuse to call this an improvement.
    let make_runs = |tokens: u64, micros: f64, report: &xencode_context_rs::EvalReport| {
        (0..xencode_context_rs::MIN_REPEATS)
            .map(|run_index| {
                let mut row = xencode_context_rs::RequestMetrics::new(
                    "BALANCED",
                    HardwareProfile::Balanced.ctx_tokens() as u32,
                );
                row.prompt_tokens = tokens.min(u32::MAX as u64) as u32;
                row.elapsed_ms = Some(micros / 1000.0).map(|v| v as u64);
                row.prompt_version = Some(xencode_context_rs::prompts::set_version().to_string());
                let mut run = xencode_context_rs::BenchmarkRun::new(
                    run_index,
                    report.clone(),
                    vec![],
                    &[],
                    std::slice::from_ref(&row),
                );
                // The token figure is the assembler's estimate, not a server
                // count: keep it on the run without pretending it was metered.
                run.task.tokens = Some(tokens);
                run
            })
            .collect()
    };
    let baseline = xencode_context_rs::BenchmarkRecord::new(
        "xi3-context",
        "existing-assembly",
        xencode_context_rs::EvidenceKind::HarnessValidation,
        config.clone(),
        corpus.clone(),
        make_runs(
            baseline_total.tokens / scored.max(1) as u64,
            baseline_total.micros,
            &report,
        ),
    );
    let adaptive = xencode_context_rs::BenchmarkRecord::new(
        "xi3-context",
        "adaptive-plan",
        xencode_context_rs::EvidenceKind::HarnessValidation,
        config,
        corpus,
        make_runs(
            adaptive_total.tokens / scored.max(1) as u64,
            adaptive_total.micros,
            &report,
        ),
    );
    let comparison = xencode_context_rs::compare(&baseline, &adaptive);
    let bp = baseline_total.precision / n;
    let br = baseline_total.recall / n;
    let ap = adaptive_total.precision / n;
    let ar = adaptive_total.recall / n;
    let bt = baseline_total.tokens / scored.max(1) as u64;
    let at = adaptive_total.tokens / scored.max(1) as u64;
    println!(
        "gold set: {scored} probes; retrieval MRR {:.3}, recall@1 {:.3}, recall@5 {:.3} \
         (both arms reuse the same retrieve(); index holds only the {}-file gold answer set, \
         a smaller haystack than the full-repo baseline)",
        report.mrr,
        report.recall_at[0],
        report.recall_at[4],
        index.files.len()
    );
    println!(
        "context precision@5: baseline {:.3} -> adaptive {:.3} ({:+.1}% relative); \
         context recall@5: baseline {:.3} -> adaptive {:.3} ({:+.1}% relative)",
        bp,
        ap,
        if bp > 0.0 {
            (ap - bp) / bp * 100.0
        } else {
            0.0
        },
        br,
        ar,
        if br > 0.0 {
            (ar - br) / br * 100.0
        } else {
            0.0
        }
    );
    println!(
        "estimated tokens/turn: baseline {bt} -> adaptive {at} ({:+}); \
         decision+assembly CPU: {:.0} us -> {:.0} us",
        at as i64 - bt as i64,
        baseline_total.micros / n,
        adaptive_total.micros / n
    );
    println!(
        "XI-9 comparison: insufficient_evidence={} reasons={:?}",
        comparison.insufficient_evidence, comparison.reasons
    );
    for metric in [
        xencode_context_rs::Metric::Tokens,
        xencode_context_rs::Metric::LatencyMs,
        xencode_context_rs::Metric::RetrievalMrr,
    ] {
        let delta = comparison
            .deltas
            .iter()
            .find(|d| d.metric == metric)
            .unwrap();
        println!("  delta {}: {:?}", metric.name(), delta.delta);
    }
    println!(
        "release gates: +10% precision {}, <=5% recall regression {}, \
         no token increase without task gain: task gain UNMEASURED (needs GPU/model)",
        if ap >= bp * 1.10 { "MET" } else { "NOT MET" },
        if br > 0.0 && (br - ar) / br <= 0.05 {
            "MET"
        } else {
            "NOT MET"
        }
    );
    // The two arms are different configurations, so XI-9 must refuse to call
    // this an improvement. This is a report, not a claim.
    assert!(comparison.insufficient_evidence);
    assert!(!comparison.improvement_claimed);
    assert!(
        report.mrr > 0.0,
        "the shipped retrieval arm must be measured"
    );
}
