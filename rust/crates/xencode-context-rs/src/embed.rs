//! Semantic retrieval stage (M5) — hybrid lexical scoring + reranking.
//!
//! The plan defers real embeddings until deterministic retrieval has been
//! measured (§18 "evaluate before you embed"). This module provides the
//! lexical half of the hybrid and the extension point where an embedding
//! model plugs in later:
//!
//! - `Bm25` scores a file against the query over its **pseudo-document**:
//!   path segments + declared symbols (structs/functions/imports/exports).
//!   No content reads, no model, fully deterministic.
//! - `hybrid_rerank` reorders the structural top-K by a BM25 boost so that a
//!   filename that merely mentions the query ranks below the file whose
//!   symbols actually do the work.
//! - `Embedder` is the trait an Ollama/llama.cpp embeddings endpoint will
//!   implement in a later pass; `None` keeps the engine purely lexical.

use crate::retrieve::{RetrievedFile, RetrievalIndex};
use std::collections::HashMap;

/// Very small stop list. The pseudo-docs are already tiny (paths + symbols),
/// so we only strip high-frequency structural noise.
const STOP: &[&str] = &[
    "the", "and", "for", "with", "use", "using", "when", "how", "why", "what", "does",
    "file", "files", "fn", "struct", "let", "pub", "src", "lib", "mod", "rs", "py", "ts",
    "json", "xencode", "context", "module", "impl",
];

const K1: f64 = 1.2;
const B: f64 = 0.75;

/// Lowercase word tokenizer; tokens must be ≥2 chars and not in [`STOP`].
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        let t = raw.trim().to_lowercase();
        if t.len() >= 2 && !STOP.contains(&t.as_str()) {
            out.push(t);
        }
    }
    out
}

/// Terms a symbol contributes: `login_page` → `login`, `page`; camelCase names
/// stay as one token plus their lowercased self.
fn symbol_tokens(name: &str) -> Vec<String> {
    let mut toks: Vec<String> = tokenize(name);
    for part in name.split(&['_', '-']) {
        toks.extend(tokenize(part));
        // camel/snake boundary: split on uppercase transitions.
        let mut cur = String::new();
        let mut prev_lower = false;
        for ch in part.chars() {
            if ch.is_uppercase() && prev_lower {
                if cur.len() >= 2 {
                    toks.push(cur.to_lowercase());
                }
                cur.clear();
            }
            prev_lower = ch.is_lowercase() || ch.is_ascii_digit();
            cur.push(ch);
        }
        if cur.len() >= 2 {
            toks.push(cur.to_lowercase());
        }
    }
    toks
}

/// The pseudo-document for one indexed file: path segments (boosted) + every
/// symbol identifier it declares. Deterministic ordering.
pub fn pseudo_document(path: &str, symbols: &crate::symbols::PerFileSymbols) -> Vec<String> {
    let mut toks = Vec::new();
    for seg in path.split(&['/', '.']) {
        toks.extend(symbol_tokens(seg));
    }
    // Path tokens count double: path hits are cheap, reliable signal.
    toks.extend(tokenize(path));
    for name in &symbols.structs {
        toks.extend(symbol_tokens(name));
    }
    for name in &symbols.functions {
        toks.extend(symbol_tokens(name));
    }
    for name in &symbols.imports {
        toks.extend(symbol_tokens(name));
    }
    for name in &symbols.exports {
        toks.extend(symbol_tokens(name));
    }
    toks
}

/// BM25 over pseudo-documents. Built once per (index, candidates); the classic
/// `k1=1.2, b=0.75` defaults keep it comparable across queries.
pub struct Bm25 {
    idf: HashMap<String, f64>,
    tfs: Vec<HashMap<String, u32>>,
    lens: Vec<u32>,
    avg_dl: f64,
}

impl Bm25 {
    /// Build over a set of (path, symbols) pseudo-documents, aligned by index.
    pub fn build(index: &RetrievalIndex) -> Self {
        let mut tfs = Vec::with_capacity(index.files.len());
        let mut lens = Vec::with_capacity(index.files.len());
        let mut df: HashMap<String, u32> = HashMap::new();
        let n = index.files.len().max(1) as f64;

        for file in &index.files {
            let syms = index.symbols.get(&file.path);
            let toks = pseudo_document(&file.path, syms.unwrap_or(&crate::symbols::PerFileSymbols::default()));
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in &toks {
                let count = tf.entry(t.clone()).or_insert(0);
                *count += 1;
            }
            for t in tf.keys() {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
            lens.push(toks.len() as u32);
            tfs.push(tf);
        }
        let total: u64 = lens.iter().map(|l| *l as u64).sum();
        let avg_dl = total as f64 / n;

        let mut idf = HashMap::new();
        for (term, d) in df {
            let d = d as f64;
            idf.insert(term, (1.0 + (n - d + 0.5) / (d + 0.5)).ln());
        }

        Self { idf, tfs, lens, avg_dl }
    }

    /// BM25 score of document `doc_idx` for `query_terms`.
    pub fn score(&self, doc_idx: usize, query_terms: &[String]) -> f64 {
        let Some(tf_map) = self.tfs.get(doc_idx) else {
            return 0.0;
        };
        let len = self.lens[doc_idx] as f64;
        let mut s = 0.0;
        for term in query_terms {
            let Some(&tf) = tf_map.get(term) else { continue };
            let Some(idf) = self.idf.get(term) else { continue };
            let tf = tf as f64;
            let denom = tf + K1 * (1.0 - B + B * len / self.avg_dl.max(1e-3));
            s += idf * (tf * (K1 + 1.0) / denom);
        }
        s
    }

    pub fn max_score_per_doc(&self, query_terms: &[String]) -> Vec<f64> {
        (0..self.tfs.len()).map(|i| self.score(i, query_terms)).collect()
    }
}

/// Reorder structural top-K with a BM25 boost (hybrid). The structural score
/// stays the ceiling — BM25 only reorders within `results` — so a strong
/// symbol/path hit cannot be displaced by a keyword-only match.
pub fn hybrid_rerank(index: &RetrievalIndex, results: &[RetrievedFile], query: &str) -> Vec<RetrievedFile> {
    let terms = tokenize(query);
    let bm25 = Bm25::build(index);
    // Map result ordinal → BM25 score.
    let mut boosted: Vec<(u64, String, Vec<String>)> = Vec::new();
    let mut doc_idxs: HashMap<String, usize> = HashMap::new();
    for (i, f) in index.files.iter().enumerate() {
        doc_idxs.insert(f.path.clone(), i);
    }
    for r in results {
        let b = doc_idxs.get(&r.path).map(|&i| bm25.score(i, &terms)).unwrap_or(0.0);
        // +8 × bm25 keeps values on a comparable scale to the structural table.
        let combined = r.score + (b * 8.0).round() as u64;
        let mut reasons = r.reasons.clone();
        reasons.push(format!("bm25 {b:.2}"));
        boosted.push((combined, r.path.clone(), reasons));
    }
    boosted.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    boosted
        .into_iter()
        .map(|(score, path, reasons)| RetrievedFile { path, score, reasons })
        .collect()
}

/// Extension point for real embedding models (M5+, post-evaluation). Returning
/// `None` keeps the pipeline purely lexical; a future Ollama / llama.cpp
/// embeddings client implements `Embedder` and `hybrid_score` folds the
/// cosine similarity in without touching retrieval.
pub trait Embedder {
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
}

/// Cosine similarity between two normalized vectors; `None` when either side is
/// empty — the hybrid falls back to the pure lexical score.
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
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

    #[test]
    fn tokenize_splits_punctuation_and_skips_stopwords() {
        assert_eq!(tokenize("how does auth flow?"), vec!["auth", "flow"]);
        assert!(tokenize("the and for").is_empty());
    }

    #[test]
    fn bm25_prefers_symbol_dense_file() {
        let mut idx = RetrievalIndex {
            files: vec![file("web/static/style.css"), file("src/auth.rs")],
            ..Default::default()
        };
        idx.symbols.insert(
            "web/static/style.css".to_string(),
            PerFileSymbols { structs: vec![], functions: vec!["layout()".to_string()], imports: vec![], exports: vec![] },
        );
        idx.symbols.insert(
            "src/auth.rs".to_string(),
            PerFileSymbols { structs: vec!["LoginPage".to_string()], functions: vec!["authenticate()".to_string(), "sessions()".to_string()], imports: vec![], exports: vec![] },
        );
        let bm25 = Bm25::build(&idx);
        let terms = tokenize("login page authenticate");
        let css = bm25.score(0, &terms);
        let auth = bm25.score(1, &terms);
        assert!(auth > css, "auth file should beat the css file on semantic signal: {css} vs {auth}");
    }

    #[test]
    fn hybrid_rerank_moves_symbol_match_ahead_of_filename_substring() {
        let mut idx = RetrievalIndex {
            files: vec![file("web/style/login.ts"), file("src/login.rs")],
            ..Default::default()
        };
        idx.symbols.insert(
            "web/style/login.ts".to_string(),
            PerFileSymbols { structs: vec![], functions: vec![], imports: vec![], exports: vec![] },
        );
        idx.symbols.insert(
            "src/login.rs".to_string(),
            PerFileSymbols { structs: vec![], functions: vec!["perform_login()".to_string()], imports: vec![], exports: vec![] },
        );
        // Structural scores are equal (both filename exactly "login"); rerank
        // must promote the file that actually performs the login.
        let results = vec![
            RetrievedFile { path: "web/style/login.ts".to_string(), score: 10, reasons: vec![] },
            RetrievedFile { path: "src/login.rs".to_string(), score: 10, reasons: vec![] },
        ];
        let reranked = hybrid_rerank(&idx, &results, "login perform login");
        assert_eq!(reranked[0].path, "src/login.rs");
        assert!(reranked[0].reasons.iter().any(|r| r.starts_with("bm25")));
    }

    #[test]
    fn cosine_matches_and_rejects_mismatched_vectors() {
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), Some(0.0));
        assert!(cosine(&[1.0, 1.0], &[1.0, 1.0]).unwrap() > 0.99);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(cosine(&[], &[]), None);
    }
}