//! The mechanism test for default-off term-association expansion (SE-4).
//!
//! The corpus under `fixtures/cooccur_corpus/src/` is made of real Rust source
//! files with declarations and documentation comments. The test reads them from
//! disk and runs the real symbol extractor and the real retrieval pipeline; no
//! answer or score is canned here.
//!
//! The prompt is about a `gateway`. The answer file, `src/resilience.rs`, says
//! nothing about a gateway: its symbols and prose use `backoff`, `retry`,
//! `circuit` and `breaker` instead. Those words co-occur with `gateway` in the
//! other files of this corpus, so an association index built from the corpus
//! can add them to the query and let BM25 reach a file whose own text shares no
//! word with the prompt. The test prints both ranked lists and the measured
//! PMI values; it asserts the mechanism (the answer enters / rank improves),
//! not any claim about answer quality.

use std::collections::HashSet;
use std::path::PathBuf;

use xencode_context_rs::{
    extract_rust_symbols, retrieve, tokenize, CooccurrenceIndex, FileEntry, RetrievalIndex,
    RetrieveOptions,
};

const PROMPT: &str = "gateway failure";
const ANSWER: &str = "src/resilience.rs";

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cooccur_corpus")
}

fn build_index() -> RetrievalIndex {
    let root = corpus_dir();
    let mut index = RetrievalIndex::default();
    // Read in a fixed order so `index.files` and every printed score are
    // deterministic. This is the same inventory an indexer would write.
    let mut paths: Vec<PathBuf> = std::fs::read_dir(root.join("src"))
        .expect("fixture source directory exists")
        .map(|entry| entry.expect("readable directory entry").path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("rs"))
        .collect();
    paths.sort();
    for path in paths {
        let rel = format!(
            "src/{}",
            path.file_name()
                .and_then(|n| n.to_str())
                .expect("utf8 name")
        );
        let content = std::fs::read_to_string(&path).expect("read real source file");
        let symbols = extract_rust_symbols(&content);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("rs")
            .to_string();
        index.files.push(FileEntry {
            path: rel.clone(),
            language: "rust".to_string(),
            size: content.len() as u64,
            loc: content.lines().count() as u64,
            ext,
            important: false,
            secret: false,
            binary: false,
        });
        index.symbols.insert(rel, symbols);
    }
    index
}

fn rank_of(list: &[xencode_context_rs::RetrievedFile], path: &str) -> Option<usize> {
    list.iter().position(|r| r.path == path).map(|i| i + 1)
}

fn print_ranked(label: &str, list: &[xencode_context_rs::RetrievedFile]) {
    println!("{label} ({} files):", list.len());
    for (i, r) in list.iter().enumerate() {
        println!(
            "  {:>2}. {:<24} score={:<3} {}",
            i + 1,
            r.path,
            r.score,
            r.reasons.join(", ")
        );
    }
}

#[test]
fn association_brings_in_a_file_that_shares_no_prompt_word() {
    let index = build_index();
    assert_eq!(index.files.len(), 6, "all six real source files were read");

    let answer = index
        .symbols
        .get(ANSWER)
        .expect("the answer file has an extracted symbol inventory");
    let prompt_tokens: Vec<String> = tokenize(PROMPT);
    let answer_tokens = xencode_context_rs::pseudo_document(ANSWER, answer, true);
    for token in &prompt_tokens {
        assert!(
            !answer_tokens.contains(token),
            "the answer must share no prompt token, but shares {token}"
        );
    }
    println!(
        "prompt tokens: {:?}; answer pseudo-doc tokens: {:?}",
        prompt_tokens, answer_tokens
    );

    let cooc = CooccurrenceIndex::build(&index, true);
    for term in &prompt_tokens {
        println!("PMI associates of {term:?}: {:?}", cooc.associates(term));
    }
    let expanded = cooc.expand(
        &prompt_tokens,
        xencode_context_rs::DEFAULT_PER_TERM,
        xencode_context_rs::DEFAULT_MAX_TERMS,
    );
    println!("expanded query terms: {:?}", expanded);
    assert!(
        !expanded.is_empty(),
        "the corpus must actually teach the prompt terms some associates"
    );
    for term in &expanded {
        assert!(
            answer_tokens.contains(term),
            "the answer should carry the learned term {term}"
        );
    }

    let changed = HashSet::new();
    let lexical_only = RetrieveOptions {
        lexical: true,
        ..Default::default()
    };
    assert!(
        !lexical_only.assoc,
        "association must be off in the default"
    );
    let with_assoc = RetrieveOptions {
        lexical: true,
        assoc: true,
        ..Default::default()
    };

    let base = retrieve(PROMPT, &index, &changed, &lexical_only);
    let assoc = retrieve(PROMPT, &index, &changed, &with_assoc);
    print_ranked("BM25 only", &base);
    print_ranked("BM25 + association", &assoc);

    let base_rank = rank_of(&base, ANSWER);
    let assoc_rank = rank_of(&assoc, ANSWER);
    println!(
        "answer {ANSWER}: BM25 rank {:?} -> with association rank {:?}",
        base_rank, assoc_rank
    );
    let base_score = base
        .iter()
        .find(|r| r.path == ANSWER)
        .map(|r| r.score)
        .unwrap_or(0);
    let assoc_score = assoc
        .iter()
        .find(|r| r.path == ANSWER)
        .map(|r| r.score)
        .unwrap_or(0);
    println!("answer score: {base_score} -> {assoc_score}");

    let rank_number = |r: Option<usize>| r.unwrap_or(usize::MAX);
    assert!(
        rank_number(assoc_rank) < rank_number(base_rank),
        "association must improve the answer's rank or bring it in at all"
    );
    assert!(
        assoc_score > base_score,
        "the expanded query must raise the answer's lexical score"
    );
}
