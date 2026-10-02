//! QO-4 — the hot paths, timed ten samples each.
//!
//! The workload is this repository, read from disk: the same tree the indexer
//! walks and retrieval ranks. Nothing here is synthetic, so a number in a
//! baseline means "this machine, this tree, this code" and nothing more. Because
//! the corpus is live, every baseline records the file count it measured against
//! and `xencode perf` refuses to compare runs over different trees.

use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::{BTreeMap, HashSet};
use std::hint::black_box;
use std::path::PathBuf;

use xencode_context_rs::{budget, compact, conversation, embed, index, retrieve, scanner, symbols};

/// The workspace root, found by walking up from the directory the bench runs in.
fn workspace() -> PathBuf {
    let mut dir = std::env::current_dir().expect("bench has no working directory");
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("crates").is_dir() {
            return dir;
        }
        if !dir.pop() {
            panic!("no Cargo.toml with a crates/ directory above the bench's working directory");
        }
    }
}

/// The tree as the indexer sees it, produced by the production scan rather than
/// assembled by hand, plus the source text of every Rust file in it.
struct Corpus {
    root: PathBuf,
    files: Vec<index::FileEntry>,
    paths: Vec<String>,
    contents: BTreeMap<String, String>,
    symbols: BTreeMap<String, symbols::PerFileSymbols>,
    deps: Vec<symbols::DepEdge>,
}

impl Corpus {
    fn load() -> Self {
        let root = workspace();
        let outcome = scanner::scan_tree(&root, &scanner::ScanOptions::default())
            .expect("the workspace root scans");
        let files: Vec<index::FileEntry> = outcome
            .files
            .iter()
            .map(|e| index::FileEntry {
                path: e.path.clone(),
                language: e.language.as_str().to_string(),
                size: e.size,
                loc: e.loc,
                ext: e.ext.clone(),
                important: e.important,
                secret: e.is_secret,
                binary: e.is_binary,
            })
            .collect();

        let mut paths = Vec::new();
        let mut contents = BTreeMap::new();
        for entry in &files {
            if entry.secret || entry.binary || entry.ext != "rs" {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(root.join(&entry.path)) {
                paths.push(entry.path.clone());
                contents.insert(entry.path.clone(), text);
            }
        }
        paths.sort();
        assert!(
            paths.len() > 100,
            "the corpus is this workspace's own Rust sources; {} files is not that",
            paths.len()
        );

        let symbols = extract_all(&paths, &contents);
        let deps = symbols::build_graph(&paths, &symbols);
        Self {
            root,
            files,
            paths,
            contents,
            symbols,
            deps,
        }
    }

    fn index(&self) -> retrieve::RetrievalIndex {
        retrieve::RetrievalIndex {
            files: self.files.clone(),
            symbols: self.symbols.clone(),
            deps: self.deps.clone(),
            mtimes: BTreeMap::new(),
            history: Default::default(),
        }
    }

    /// How many files this run measured against — the corpus identity a baseline
    /// has to agree with before its numbers mean anything.
    fn file_count(&self) -> usize {
        self.paths.len()
    }
}

fn extract_all(
    paths: &[String],
    contents: &BTreeMap<String, String>,
) -> BTreeMap<String, symbols::PerFileSymbols> {
    let mut out = BTreeMap::new();
    for path in paths {
        let text = contents.get(path).expect("corpus entry");
        out.insert(path.clone(), symbols::extract_rust_symbols(text));
    }
    out
}

/// Index build: the walk, the per-file symbol pass, the dependency graph.
fn bench_index_build(c: &mut Criterion, corpus: &Corpus) {
    let options = scanner::ScanOptions::default();
    let root = corpus.root.clone();
    c.bench_function("index_build/scan_tree", move |b| {
        b.iter(|| {
            let outcome = scanner::scan_tree(black_box(&root), &options).expect("scan");
            outcome.files.len()
        })
    });

    let paths = corpus.paths.clone();
    let contents = corpus.contents.clone();
    c.bench_function("index_build/extract_symbols", move |b| {
        b.iter(|| extract_all(black_box(&paths), &contents).len())
    });

    let paths_for_graph = corpus.paths.clone();
    let symbols_for_graph = corpus.symbols.clone();
    c.bench_function("index_build/build_graph", move |b| {
        b.iter(|| symbols::build_graph(black_box(&paths_for_graph), &symbols_for_graph).len())
    });
}

/// Retrieval: the arm a live turn runs, and the lexical pass it depends on.
fn bench_retrieve(c: &mut Criterion, corpus: &Corpus) {
    let idx = corpus.index();
    let changed: HashSet<String> = HashSet::new();
    let indexable: Vec<usize> = (0..idx.files.len()).collect();
    let terms = embed::tokenize("compact transcript entries");
    let queries = [
        "compact the conversation transcript",
        "provider api key resolution",
        "render the chat frame",
    ];

    let retrieve_idx = idx.clone();
    let mut next = 0usize;
    c.bench_function("retrieve/hybrid_top10", move |b| {
        b.iter(|| {
            let query = black_box(queries[next % queries.len()]);
            next += 1;
            let options = retrieve::RetrieveOptions::for_live_chat(10, Default::default());
            retrieve::retrieve(query, &retrieve_idx, &changed, &options).len()
        })
    });

    c.bench_function("retrieve/bm25_build_and_score", move |b| {
        b.iter(|| {
            let bm25 = embed::Bm25::over(black_box(&idx), &indexable, true);
            bm25.score(0, &terms)
        })
    });
}

/// Compaction: the deterministic fold that runs when the window fills.
fn bench_compaction(c: &mut Criterion, corpus: &Corpus) {
    let mut transcript = conversation::Transcript::new("bench");
    let mut block = String::new();
    for path in &corpus.paths {
        block.push_str(corpus.contents.get(path).expect("corpus entry"));
        block.push('\n');
        if block.len() > 4000 {
            transcript.entries.push(conversation::TranscriptEntry {
                role: "assistant".to_string(),
                content: block.clone(),
                ts_unix_ms: 0,
                is_decision: transcript.entries.len().is_multiple_of(7),
            });
            block.clear();
            if transcript.entries.len() >= 240 {
                break;
            }
        }
    }
    assert!(
        transcript.entries.len() >= 40,
        "the transcript is built from real file bodies; {} entries is not one",
        transcript.entries.len()
    );

    c.bench_function("compaction/soft_compact", move |b| {
        b.iter(|| {
            let mut copy = black_box(transcript.clone());
            let report = compact::soft_compact(&mut copy, 0.4);
            (report.before, report.after, report.dropped)
        })
    });
}

/// The token trimmer every tier cap goes through, over real code.
fn bench_truncation(c: &mut Criterion, corpus: &Corpus) {
    let bodies: Vec<String> = corpus.paths.to_vec();
    let contents = corpus.contents.clone();
    c.bench_function("truncation/truncate_to_tokens", move |b| {
        b.iter(|| {
            let mut kept = 0usize;
            for path in &bodies {
                let text = contents.get(path).expect("corpus entry");
                let (head, tokens) = budget::truncate_to_tokens(black_box(text), 300, true);
                kept += head.len() + tokens as usize;
            }
            kept
        })
    });
}

fn bench_all(c: &mut Criterion) {
    let corpus = Corpus::load();
    // The corpus identity is printed so a recorded baseline can be checked
    // against the run that produced it; `xencode perf` reads this line back.
    println!(
        "xencode-perf-corpus: files={} root={}",
        corpus.file_count(),
        corpus.root.display()
    );
    bench_index_build(c, &corpus);
    bench_retrieve(c, &corpus);
    bench_compaction(c, &corpus);
    bench_truncation(c, &corpus);
}

criterion_group!(
    name = hot_paths;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(2))
        .warm_up_time(std::time::Duration::from_millis(300));
    targets = bench_all
);
criterion_main!(hot_paths);
