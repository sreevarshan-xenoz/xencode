//! Semantic retrieval stage (M5) — hybrid lexical scoring over the index.
//!
//! The plan defers real embeddings until deterministic retrieval has been
//! measured (§18 "evaluate before you embed"). This module provides the
//! lexical half of the hybrid and the extension point where an embedding
//! model plugs in later:
//!
//! - `Bm25` scores a file against the query over its **pseudo-document**:
//!   path segments + declared symbols (structs/functions/imports/exports) +
//!   the file's documentation prose. No model, fully deterministic.
//! - `hybrid_select` runs that arm over the whole index and keeps the top-K by
//!   structural score plus lexical score together, so a file only its prose
//!   describes can still be reached. It replaces a `hybrid_rerank` stage that
//!   only reordered the top-K it was handed and so could never promote a file
//!   the structural pass had missed. A real embeddings model would fold into
//!   `hybrid_select`; nothing is wired until the eval says it pays.

use crate::retrieve::{RetrievalIndex, RetrievedFile};
use std::collections::HashMap;

/// Very small stop list. The pseudo-docs were written against path and symbol
/// tokens, so we only strip high-frequency structural noise.
const STOP: &[&str] = &[
    "the", "and", "for", "with", "use", "using", "when", "how", "why", "what", "does", "file",
    "files", "fn", "struct", "let", "pub", "src", "lib", "mod", "rs", "py", "ts", "json",
    "xencode", "context", "module", "impl",
];

const K1: f64 = 1.2;
const B: f64 = 0.75;

/// How much a BM25 point is worth next to a structural point. The structural
/// table runs from 3 (the floor) through 6/8/10 signals to 16 (the symbol cap);
/// a typical strong lexical match scores 2–6, so ×4 lets prose place a file
/// alongside one weak structural signal and a dense match outrank a filename
/// substring, without letting vocabulary alone beat an exact filename hit.
pub const LEXICAL_WEIGHT: f64 = 4.0;

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
/// symbol identifier it declares + its documentation prose. `docs` is the only
/// arm that carries language a person would type at a question rather than a
/// name the compiler sees; `with_docs = false` drops it, which is what lets the
/// eval measure what the prose is actually worth. Deterministic ordering.
pub fn pseudo_document(
    path: &str,
    symbols: &crate::symbols::PerFileSymbols,
    with_docs: bool,
) -> Vec<String> {
    let mut toks = Vec::new();
    for seg in path.split(&['/', '.']) {
        toks.extend(symbol_tokens(seg));
    }
    // Path tokens count double: path hits are cheap, reliable signal.
    toks.extend(tokenize(path));
    for names in [
        &symbols.structs,
        &symbols.enums,
        &symbols.traits,
        &symbols.types,
        &symbols.functions,
        &symbols.impls,
        &symbols.mods,
        &symbols.imports,
        &symbols.exports,
    ] {
        for name in names {
            toks.extend(symbol_tokens(name));
        }
    }
    if with_docs {
        toks.extend(tokenize(&symbols.docs));
    }
    toks
}

/// BM25 over pseudo-documents. Built once per (index, candidates); the classic
/// `k1=1.2, b=0.75` defaults keep it comparable across queries.
pub struct Bm25 {
    idf: HashMap<String, f64>,
    /// Term frequencies and lengths, keyed by the position in
    /// `RetrievalIndex::files` rather than by build order, so a subset built
    /// over candidates still answers `score(doc_idx, …)`.
    tfs: HashMap<usize, HashMap<String, u32>>,
    lens: HashMap<usize, u32>,
    avg_dl: f64,
}

impl Bm25 {
    /// Build over every indexed file, aligned by index with `index.files`.
    pub fn build(index: &RetrievalIndex) -> Self {
        let idx: Vec<usize> = (0..index.files.len()).collect();
        Self::over(index, &idx, true)
    }

    /// Build over the given subset of `index.files`, addressed by those same
    /// indices. Inverse document frequency and average length still come from
    /// the subset, because a term's rarity only means something relative to the
    /// documents being ranked against each other.
    pub fn over(index: &RetrievalIndex, docs: &[usize], with_docs: bool) -> Self {
        let mut tfs = HashMap::with_capacity(docs.len());
        let mut lens = HashMap::with_capacity(docs.len());
        let mut df: HashMap<String, u32> = HashMap::new();
        let n = docs.len().max(1) as f64;

        for &i in docs {
            let file = &index.files[i];
            let syms = index.symbols.get(&file.path);
            let toks = pseudo_document(
                &file.path,
                syms.unwrap_or(&crate::symbols::PerFileSymbols::default()),
                with_docs,
            );
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in &toks {
                let count = tf.entry(t.clone()).or_insert(0);
                *count += 1;
            }
            for t in tf.keys() {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
            lens.insert(i, toks.len() as u32);
            tfs.insert(i, tf);
        }
        let total: u64 = lens.values().map(|l| *l as u64).sum();
        let avg_dl = total as f64 / n;

        let mut idf = HashMap::new();
        for (term, d) in df {
            let d = d as f64;
            idf.insert(term, (1.0 + (n - d + 0.5) / (d + 0.5)).ln());
        }

        Self {
            idf,
            tfs,
            lens,
            avg_dl,
        }
    }

    /// BM25 score of document `doc_idx` for `query_terms`.
    pub fn score(&self, doc_idx: usize, query_terms: &[String]) -> f64 {
        let (Some(tf_map), Some(&len)) = (self.tfs.get(&doc_idx), self.lens.get(&doc_idx)) else {
            return 0.0;
        };
        let len = len as f64;
        let mut s = 0.0;
        for term in query_terms {
            let Some(&tf) = tf_map.get(term) else {
                continue;
            };
            let Some(idf) = self.idf.get(term) else {
                continue;
            };
            let tf = tf as f64;
            let denom = tf + K1 * (1.0 - B + B * len / self.avg_dl.max(1e-3));
            s += idf * (tf * (K1 + 1.0) / denom);
        }
        s
    }
}

/// The lexical arm as a candidate stage: score **every** indexable file with
/// BM25 and keep the top-`k` by structural score plus a
/// [`LEXICAL_WEIGHT`]-scaled lexical one.
///
/// This starts from the whole index rather than from a list it is handed, so a
/// file whose *text* answers the query can enter the candidate set on that
/// basis alone. A file with no structural signal still has to clear `floor` on
/// the lexical boost, which is what keeps a stray common word from dragging the
/// index into the prompt.
///
/// `structural` is the scored structural candidate list, in any order. The
/// result is sorted by combined score, ties by path.
pub fn hybrid_select(
    index: &RetrievalIndex,
    structural: &[(String, u64, Vec<String>)],
    query: &str,
    k: usize,
    with_docs: bool,
    floor: u64,
) -> Vec<RetrievedFile> {
    let indexable: Vec<usize> = index
        .files
        .iter()
        .enumerate()
        .filter(|(_, f)| !f.secret && !f.binary)
        .map(|(i, _)| i)
        .collect();
    let bm25 = Bm25::over(index, &indexable, with_docs);
    let terms = tokenize(query);

    let mut structural_of: HashMap<&str, (u64, &Vec<String>)> = HashMap::new();
    for (path, score, reasons) in structural {
        structural_of.insert(path.as_str(), (*score, reasons));
    }

    let mut scored: Vec<(u64, String, Vec<String>)> = Vec::new();
    for &i in &indexable {
        let path = &index.files[i].path;
        let b = bm25.score(i, &terms);
        let boost = (b * LEXICAL_WEIGHT).round() as u64;
        let (base, reasons) = structural_of
            .get(path.as_str())
            .map(|(s, r)| (*s, (*r).clone()))
            .unwrap_or((0, Vec::new()));
        if base + boost < floor.max(crate::retrieve::MIN_SCORE) {
            continue;
        }
        let mut reasons = reasons;
        if boost > 0 {
            reasons.push(format!("text match {b:.2}"));
        }
        scored.push((base + boost, path.clone(), reasons));
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(k)
        .map(|(score, path, reasons)| RetrievedFile {
            path,
            score,
            reasons,
        })
        .collect()
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
            PerFileSymbols {
                structs: vec![],
                functions: vec!["layout()".to_string()],
                imports: vec![],
                exports: vec![],
                mods: vec![],
                ..Default::default()
            },
        );
        idx.symbols.insert(
            "src/auth.rs".to_string(),
            PerFileSymbols {
                structs: vec!["LoginPage".to_string()],
                functions: vec!["authenticate()".to_string(), "sessions()".to_string()],
                imports: vec![],
                exports: vec![],
                mods: vec![],
                ..Default::default()
            },
        );
        let bm25 = Bm25::build(&idx);
        let terms = tokenize("login page authenticate");
        let css = bm25.score(0, &terms);
        let auth = bm25.score(1, &terms);
        assert!(
            auth > css,
            "auth file should beat the css file on semantic signal: {css} vs {auth}"
        );
    }

    #[test]
    fn hybrid_select_promotes_a_file_no_structural_signal_surfaced() {
        // The whole point of moving the lexical arm from a rerank stage into
        // candidate selection: a file whose name says nothing about the query
        // but whose documentation does must now be reachable at all.
        let mut idx = RetrievalIndex {
            files: vec![
                file("web/style/login.ts"),
                file("src/session_store.rs"),
                file("src/misc.rs"),
            ],
            ..Default::default()
        };
        idx.symbols.insert(
            "src/session_store.rs".to_string(),
            PerFileSymbols {
                docs: "Keeps a signed cookie alive across restarts and refreshes it.".to_string(),
                ..Default::default()
            },
        );
        let picked = hybrid_select(&idx, &[], "cookie refresh signed", 5, true, 3);
        assert_eq!(picked[0].path, "src/session_store.rs");
        assert!(picked[0].reasons.iter().any(|r| r.starts_with("text")));
    }

    #[test]
    fn doc_prose_can_be_switched_off_and_stops_counting() {
        let mut idx = RetrievalIndex {
            files: vec![file("src/a.rs"), file("src/b.rs")],
            ..Default::default()
        };
        idx.symbols.insert(
            "src/a.rs".to_string(),
            PerFileSymbols {
                docs: "quux".to_string(),
                ..Default::default()
            },
        );
        assert!(hybrid_select(&idx, &[], "quux", 5, true, 3)
            .iter()
            .any(|r| r.path == "src/a.rs"));
        assert!(
            hybrid_select(&idx, &[], "quux", 5, false, 3).is_empty(),
            "with the prose arm off, a doc-only term must match nothing"
        );
    }

    #[test]
    fn hybrid_select_keeps_the_structural_winner_and_reports_both_scores() {
        let mut idx = RetrievalIndex {
            files: vec![file("web/style/login.ts"), file("src/login.rs")],
            ..Default::default()
        };
        idx.symbols.insert(
            "src/login.rs".to_string(),
            PerFileSymbols {
                functions: vec!["perform_login".to_string()],
                docs: "Verifies credentials and opens a session.".to_string(),
                ..Default::default()
            },
        );
        // Both files score identically on the filename; the one whose symbols
        // and text do the work must win.
        let structural = vec![
            (
                "web/style/login.ts".to_string(),
                10u64,
                vec!["filename exact".to_string()],
            ),
            (
                "src/login.rs".to_string(),
                10u64,
                vec!["filename exact".to_string()],
            ),
        ];
        let picked = hybrid_select(&idx, &structural, "login perform credentials", 5, true, 3);
        assert_eq!(picked[0].path, "src/login.rs");
        assert!(picked[0].reasons.iter().any(|r| r.starts_with("text")));
        assert_eq!(picked[0].reasons[0], "filename exact");
    }
}
