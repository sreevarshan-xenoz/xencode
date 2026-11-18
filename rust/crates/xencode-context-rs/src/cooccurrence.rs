//! Corpus-derived term association — an optional, model-free expansion of the
//! lexical retrieval query.
//!
//! This module does **not** embed anything and does not know what a word means.
//! It measures one narrow fact about this repository: which terms occur in the
//! same file's [`crate::embed::pseudo_document`] as which other terms. Terms
//! that share files more often than chance would predict earn a positive
//! pointwise mutual information (PMI) score; at query time the strongest
//! associates of the prompt's own terms are appended to the BM25 query.
//!
//! What this can and cannot do:
//!
//! - It can pull a file into the candidate set when that file uses *different
//!   words* for the same subject and those words co-occur with the prompt's
//!   words elsewhere in the corpus. That is an association learned from this
//!   repository's text, nothing more.
//! - It is **not** a neural embedding and **not** synonymy. `retry` and
//!   `backoff` are linked here only because the source files of this repository
//!   put them together, not because anything understands what they mean. A
//!   repository whose files never mention both words gets no expansion.
//! - Real embedding quality is blocked in this workspace: there is no embedding
//!   model, crate or runtime in the dependency graph, and SCIP indexing was
//!   measured too slow (190 s) to use at query time. No improved eval pass rate
//!   is claimed for the association arm; the gold comparison is reported, not
//!   asserted.
//!
//! The index is pure Rust — `BTreeMap`/`BTreeSet` only, no new dependency, no
//! network, no process, no model — and every ranking is tie-broken by term so
//! the same corpus always yields the same expansion. It is off by default in
//! [`crate::retrieve::RetrieveOptions`].

use crate::embed::pseudo_document;
use crate::retrieve::RetrievalIndex;
use crate::symbols::PerFileSymbols;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A pair of terms must appear together in at least this many pseudo-documents
/// before its PMI is trusted. Two single co-occurrences are one file's chosen
/// vocabulary, not a corpus fact.
pub const MIN_CO_DF: u32 = 2;

/// How many associates are taken per prompt term when no other value is given.
pub const DEFAULT_PER_TERM: usize = 3;

/// Cap on the number of expanded terms added to a query. Without a cap a short
/// prompt about a hub term could drag half the vocabulary into BM25.
pub const DEFAULT_MAX_TERMS: usize = 8;

/// A term-to-associates index built from a corpus of pseudo-documents.
///
/// The map is keyed by a term and holds `(associated term, PMI)` pairs sorted by
/// descending PMI, then by ascending term name. Both orderings are total, so a
/// corpus produces one deterministic index.
#[derive(Debug, Clone, Default)]
pub struct CooccurrenceIndex {
    associates: BTreeMap<String, Vec<(String, f64)>>,
    documents: usize,
}

impl CooccurrenceIndex {
    /// Build the index over the pseudo-documents of an index's files, using the
    /// exact same token source as the BM25 arm (`with_docs` is the
    /// `lexical_docs` retrieval option). Secret and binary files are skipped,
    /// just as `hybrid_select` skips them.
    pub fn build(index: &RetrievalIndex, with_docs: bool) -> Self {
        Self::build_from_documents(&Self::pseudo_documents(index, with_docs))
    }

    /// Build only the rows for `focus` terms, from the same pseudo-documents.
    /// Retrieval uses this at query time: only the prompt's own terms can be
    /// expanded, so counting every other pair in a large index is wasted work.
    /// Every score produced here is identical to that term's row in a full
    /// [`Self::build`] — the same document frequencies and the same co-occurrence
    /// counts are used, and only unrelated pairs are skipped.
    pub fn build_focused(index: &RetrievalIndex, with_docs: bool, focus: &[String]) -> Self {
        Self::build_from_documents_focused(&Self::pseudo_documents(index, with_docs), focus)
    }

    fn pseudo_documents(index: &RetrievalIndex, with_docs: bool) -> Vec<BTreeSet<String>> {
        let mut documents = Vec::with_capacity(index.files.len());
        for file in &index.files {
            if file.secret || file.binary {
                continue;
            }
            let empty = PerFileSymbols::default();
            let symbols = index.symbols.get(&file.path).unwrap_or(&empty);
            let tokens = pseudo_document(&file.path, symbols, with_docs);
            documents.push(tokens.into_iter().collect::<BTreeSet<String>>());
        }
        documents
    }

    /// The pure core: PMI over a set of term sets, one set per document.
    ///
    /// Document probabilities are used (`P(t) = df(t)/N`,
    /// `P(a,b) = co_df(a,b)/N`), so `PMI(a,b) = ln(co_df(a,b) * N /
    /// (df(a) * df(b)))`. Only positive PMI with at least [`MIN_CO_DF`] shared
    /// documents is kept: a non-positive score means the two terms occur
    /// together no more often than their individual frequencies predict, which
    /// is no association at all.
    pub fn build_from_documents(documents: &[BTreeSet<String>]) -> Self {
        Self::compute(documents, None)
    }

    /// The pure form of [`Self::build_focused`]: only the rows for `focus` are
    /// kept, and their scores match the full build exactly.
    pub fn build_from_documents_focused(documents: &[BTreeSet<String>], focus: &[String]) -> Self {
        let focus_set: BTreeSet<&str> = focus.iter().map(String::as_str).collect();
        Self::compute(documents, Some(&focus_set))
    }

    fn compute(documents: &[BTreeSet<String>], focus: Option<&BTreeSet<&str>>) -> Self {
        let n = documents.len().max(1) as f64;

        // Terms become integers before any pair counting. A real index has
        // hundreds of files and thousands of distinct terms; packing two ids
        // into one u64 instead of cloning and comparing Strings for every pair
        // keeps the build fast enough to run at query time. Output order never
        // depends on this map: the final lists are sorted below.
        let vocabulary: BTreeSet<&str> = documents
            .iter()
            .flat_map(|doc| doc.iter().map(String::as_str))
            .collect();
        let id_terms: Vec<&str> = vocabulary.iter().copied().collect();
        let term_id: BTreeMap<&str, u32> = id_terms
            .iter()
            .enumerate()
            .map(|(id, &term)| (term, id as u32))
            .collect();
        let width = id_terms.len().max(1) as u64;

        let mut df = vec![0u32; id_terms.len()];
        let mut id_docs: Vec<Vec<u32>> = Vec::with_capacity(documents.len());
        for doc in documents {
            let mut ids: Vec<u32> = doc.iter().map(|term| term_id[term.as_str()]).collect();
            ids.sort_unstable();
            ids.dedup();
            for &id in &ids {
                df[id as usize] += 1;
            }
            id_docs.push(ids);
        }

        let focus_ids: Option<BTreeSet<u32>> = focus.map(|terms| {
            terms
                .iter()
                .filter_map(|term| term_id.get(term).copied())
                .collect()
        });
        let mut co_df: HashMap<u64, u32> = HashMap::new();
        for ids in &id_docs {
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    if let Some(wanted) = &focus_ids {
                        if !wanted.contains(&ids[i]) && !wanted.contains(&ids[j]) {
                            continue;
                        }
                    }
                    let key = ids[i] as u64 * width + ids[j] as u64;
                    *co_df.entry(key).or_insert(0) += 1;
                }
            }
        }

        let mut associates: BTreeMap<String, Vec<(String, f64)>> = BTreeMap::new();
        for (key, shared) in co_df {
            if shared < MIN_CO_DF {
                continue;
            }
            let a = (key / width) as usize;
            let b = (key % width) as usize;
            if let Some(wanted) = &focus_ids {
                if !wanted.contains(&(a as u32)) && !wanted.contains(&(b as u32)) {
                    continue;
                }
            }
            let da = df[a] as f64;
            let db = df[b] as f64;
            if da == 0.0 || db == 0.0 {
                continue;
            }
            let pmi = (shared as f64 * n / (da * db)).ln();
            if !pmi.is_finite() || pmi <= 0.0 {
                continue;
            }
            let ta = id_terms[a].to_string();
            let tb = id_terms[b].to_string();
            associates
                .entry(ta.clone())
                .or_default()
                .push((tb.clone(), pmi));
            associates.entry(tb).or_default().push((ta, pmi));
        }
        for list in associates.values_mut() {
            // Total order: strongest association first, term name breaks ties.
            list.sort_by(|x, y| y.1.total_cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
        }

        Self {
            associates,
            documents: documents.len(),
        }
    }

    /// The associates of one term, strongest first. Empty when the term never
    /// cleared the co-occurrence and positive-PMI bar.
    pub fn associates(&self, term: &str) -> &[(String, f64)] {
        self.associates.get(term).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Number of pseudo-documents this index was built from.
    pub fn documents(&self) -> usize {
        self.documents
    }

    /// Terms strongly associated with any of `terms`, excluding the terms
    /// themselves. At most `per_term` associates are taken for each prompt
    /// term, and at most `max_terms` are returned in total. The output is a
    /// sorted, deduplicated `Vec` so adding it to a query is deterministic.
    pub fn expand(&self, terms: &[String], per_term: usize, max_terms: usize) -> Vec<String> {
        let original: BTreeSet<&str> = terms.iter().map(String::as_str).collect();
        let mut chosen: BTreeSet<String> = BTreeSet::new();
        for term in terms {
            let mut taken = 0;
            for (associate, _) in self.associates(term) {
                if taken >= per_term || chosen.len() >= max_terms {
                    break;
                }
                if original.contains(associate.as_str()) {
                    continue;
                }
                if chosen.insert(associate.clone()) {
                    taken += 1;
                }
            }
        }
        chosen.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(terms: &[&str]) -> BTreeSet<String> {
        terms.iter().map(|t| (*t).to_string()).collect()
    }

    #[test]
    fn pmi_ranks_a_shared_term_above_a_rare_accident() {
        // `alpha` and `beta` share two of four documents and both are absent
        // elsewhere often enough for that to beat chance; `alpha` and `gamma`
        // share only one, below MIN_CO_DF. A term in every document would score
        // PMI 0 with everything, so none of these is universal.
        let docs = vec![
            doc(&["alpha", "beta"]),
            doc(&["alpha", "beta"]),
            doc(&["alpha", "gamma"]),
            doc(&["delta", "gamma"]),
        ];
        let idx = CooccurrenceIndex::build_from_documents(&docs);
        let names: Vec<&str> = idx
            .associates("alpha")
            .iter()
            .map(|(term, _)| term.as_str())
            .collect();
        assert_eq!(names, vec!["beta"]);
        assert!(idx.associates("gamma").is_empty());
        assert_eq!(idx.documents(), 4);
    }

    #[test]
    fn expansion_is_deterministic_deduplicated_and_capped() {
        let docs = vec![
            doc(&["query", "a", "b"]),
            doc(&["query", "a", "b"]),
            doc(&["query", "a", "b"]),
            doc(&["other", "c"]),
        ];
        let idx = CooccurrenceIndex::build_from_documents(&docs);
        let terms = vec!["query".to_string()];
        let first = idx.expand(&terms, 2, 2);
        assert_eq!(first, vec!["a", "b"]);
        assert_eq!(first, idx.expand(&terms, 2, 2));

        // Terms already in the prompt are never added back, and a zero cap
        // returns nothing rather than ignoring the caller's limit.
        assert!(idx
            .expand(&["query".to_string(), "a".to_string()], 0, 2)
            .is_empty());
        assert!(idx.expand(&terms, 2, 0).is_empty());
    }

    #[test]
    fn focused_rows_match_the_full_build_exactly() {
        let docs = vec![
            doc(&["query", "a", "b"]),
            doc(&["query", "a", "b"]),
            doc(&["query", "a", "b"]),
            doc(&["other", "c", "a"]),
            doc(&["other", "c", "b"]),
            doc(&["query", "c"]),
            // Neither word shares a document with `query` or `other`.
            doc(&["lonely", "solo"]),
        ];
        let full = CooccurrenceIndex::build_from_documents(&docs);
        let focus = vec!["query".to_string(), "other".to_string()];
        let focused = CooccurrenceIndex::build_from_documents_focused(&docs, &focus);
        for term in &focus {
            assert_eq!(focused.associates(term), full.associates(term), "{term}");
        }
        assert!(focused.associates("lonely").is_empty());
        assert!(focused.associates("solo").is_empty());
        assert_eq!(focused.documents(), full.documents());
    }

    #[test]
    fn a_corpus_with_no_repeated_pairs_expands_nothing() {
        let docs = vec![doc(&["solo"]), doc(&["alone"]), doc(&["single"])];
        let idx = CooccurrenceIndex::build_from_documents(&docs);
        assert!(idx.associates("solo").is_empty());
        assert!(idx.expand(&["solo".to_string()], 3, 8).is_empty());
    }
}
