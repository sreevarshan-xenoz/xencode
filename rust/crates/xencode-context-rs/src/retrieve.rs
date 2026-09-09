//! Deterministic retrieval (M2) — §9 of the architecture spec.
//!
//! Zero embeddings, zero LLM: score every indexed file against the user's
//! latest prompt with the spec's signal table, expand the dependency graph a
//! few hops from strong seeds, and return the top-K candidates:
//!
//! | Signal                  | Weight |
//! |-------------------------|--------|
//! | filename exact match    | +10    |
//! | filename substring      | +6     |
//! | path segment match      | +5     |
//! | symbol hit              | +8/unique (capped +16) |
//! | git-changed file        | +4     |
//! | dependency hop from seed| +3/hop (≤3 hops)       |
//!
//! Empty/absent queries seed from git-changed + most recently touched files.

use crate::index::{FileEntry, FilesIndex, Manifest};
use crate::symbols::{DepEdge, PerFileSymbols};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

/// Files scoring below this are never injected (spec `Minimum threshold`).
pub const MIN_SCORE: u64 = 3;
/// Files scoring ≥ this become seeds for dependency expansion.
pub const SEED_THRESHOLD: u64 = 10;
/// Max hops of dependency expansion seeded by strong files.
pub const DEFAULT_EXPAND_HOPS: usize = 3;
/// Per-unique-symbol-hit bonus, capped at this total.
pub const SYMBOL_CAP: u64 = 16;

/// A retrieved candidate with an explanation of why it scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievedFile {
    pub path: String,
    pub score: u64,
    /// Human-readable reason labels (e.g. `filename exact`, `symbol: User`).
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RetrieveOptions {
    pub top_k: usize,
    pub min_score: u64,
    pub expand_hops: usize,
    /// Files used as forced seeds when the query is empty (changed + recent).
    pub empty_query_seeds: usize,
}

impl Default for RetrieveOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: MIN_SCORE,
            expand_hops: DEFAULT_EXPAND_HOPS,
            empty_query_seeds: 6,
        }
    }
}

/// Everything retrieval needs from the on-disk index.
#[derive(Debug, Clone, Default)]
pub struct RetrievalIndex {
    pub files: Vec<FileEntry>,
    pub symbols: BTreeMap<String, PerFileSymbols>,
    pub deps: Vec<DepEdge>,
    /// Relative path (`/`) → modified epoch-millis, from the manifest.
    pub mtimes: BTreeMap<String, u64>,
}

impl RetrievalIndex {
    /// Load the retrieval index from `.xencode/` (missing/corrupt → `None`).
    pub fn load(xencode_dir: &Path) -> Option<RetrievalIndex> {
        let files: FilesIndex = crate::index::read_json(&crate::index::file_index_path(xencode_dir))?;
        let symbols: BTreeMap<String, PerFileSymbols> =
            crate::index::read_json(&crate::index::symbols_json_path(xencode_dir))?;
        let deps: Vec<DepEdge> = crate::index::read_json(&crate::index::deps_json_path(xencode_dir))?;
        let manifest: Manifest = crate::index::read_json(&crate::index::manifest_path(xencode_dir))?;
        Some(RetrievalIndex {
            files: files.files,
            symbols,
            deps,
            mtimes: manifest.mtime_map,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }
}

/// Tokenize on non-alphanumerics and camel boundaries, lowercased.
/// `refresh_token` → `["refresh", "token"]`, `UserService` → `["user", "service"]`.
pub fn word_tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|c| !c.is_empty())
    {
        let chars: Vec<char> = chunk.chars().collect();
        let mut cur = String::new();
        let mut prev_lower = false;
        for &c in &chars {
            if c.is_uppercase() && prev_lower && !cur.is_empty() {
                out.push(cur.clone());
                cur.clear();
            }
            prev_lower = c.is_lowercase();
            cur.push(c.to_ascii_lowercase());
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

/// Deterministically score every candidate file against the query.
pub fn retrieve(
    query: &str,
    index: &RetrievalIndex,
    changed: &HashSet<String>,
    options: &RetrieveOptions,
) -> Vec<RetrievedFile> {
    let q = query.trim();
    let query_words: Vec<String> = if q.is_empty() {
        Vec::new()
    } else {
        word_tokens(q)
            .into_iter()
            .filter(|w| w.len() >= 2)
            .collect()
    };

    // forced seeds for an empty query: git-changed files + most recently
    // touched (by manifest mtime).
    let forced_seeds: Vec<String> = if query_words.is_empty() {
        let mut seeds: Vec<String> = changed.iter().cloned().collect();
        let mut recent: Vec<(String, u64)> = index
            .mtimes
            .iter()
            .map(|(p, t)| (p.clone(), *t))
            .collect();
        recent.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        seeds.extend(recent.into_iter().take(options.empty_query_seeds).map(|(p, _)| p));
        seeds.sort();
        seeds.dedup();
        seeds
    } else {
        Vec::new()
    };

    let mut scores: BTreeMap<String, Score> = BTreeMap::new();

    for entry in &index.files {
        if entry.secret || entry.binary {
            continue;
        }
        let (total, reasons) = score_file(entry, &query_words, &index, changed, &forced_seeds);
        if total >= MIN_SCORE {
            scores.insert(entry.path.clone(), Score::new(total, reasons));
        }
    }

    // Dependency expansion: every seed's 1..=hop neighbours accumulate +3/hop.
    let fwd = forward_map(&index.deps);
    let mut seeds: BTreeSet<String> = scores
        .iter()
        .filter(|(_, s)| s.total >= SEED_THRESHOLD)
        .map(|(p, _)| p.clone())
        .collect();
    seeds.extend(forced_seeds.iter().cloned());

    let mut seen: HashSet<String> = HashSet::new();
    let mut frontier: Vec<String> = seeds.iter().cloned().collect();
    for hop in 1..=options.expand_hops {
        let mut next: Vec<String> = Vec::new();
        for file in frontier {
            for dep in fwd.get(&file).into_iter().flatten() {
                if dep == &file {
                    continue;
                }
                let score = scores.entry(dep.clone()).or_insert_with(|| Score::new(0, vec![]));
                score.total += 3;
                score.reasons.push(format!("dep hop {hop} via {file}"));
                if seen.insert(dep.clone()) {
                    next.push(dep.clone());
                }
            }
        }
        frontier = next;
    }

    let mut results: Vec<RetrievedFile> = scores
        .into_iter()
        .filter(|(_, s)| s.total >= options.min_score)
        .map(|(path, s)| RetrievedFile {
            path,
            score: s.total,
            reasons: s.reasons,
        })
        .collect();
    results.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
    });
    results.truncate(options.top_k);
    results
}

struct Score {
    total: u64,
    reasons: Vec<String>,
}

impl Score {
    fn new(total: u64, reasons: Vec<String>) -> Self {
        Self { total, reasons }
    }
}

fn score_file(
    entry: &FileEntry,
    query_words: &[String],
    index: &RetrievalIndex,
    changed: &HashSet<String>,
    forced_seeds: &[String],
) -> (u64, Vec<String>) {
    let mut total: u64 = 0;
    let mut reasons: Vec<String> = Vec::new();
    let path = &entry.path;
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    let name_stem = name
        .rsplit_once('.')
        .map(|(stem, _)| stem.to_string())
        .unwrap_or(name.clone());
    let q_lower = query_words.join(" ");

    // Empty-query forced seeds are already >= threshold; still cheap here.
    if forced_seeds.contains(path) {
        return (SEED_THRESHOLD, vec!["seed: changed/recent".to_string()]);
    }

    if query_words.is_empty() {
        return (0, vec![]);
    }

    // filename exact (+10)
    if path.to_ascii_lowercase() == q_lower || name_stem == q_lower {
        total += 10;
        reasons.push("filename exact".to_string());
    }
    // filename substring (+6)
    if !reasons.iter().any(|r| r == "filename exact") {
        if query_words.iter().any(|w| name.contains(w)) {
            total += 6;
            reasons.push("filename substring".to_string());
        }
    }
    // path segment match (+5)
    if query_words.iter().any(|w| {
        path
            .split('/')
            .any(|seg| seg.to_ascii_lowercase().contains(w))
    }) {
        total += 5;
        reasons.push("path segment".to_string());
    }
    // symbol hit (+8/unique, capped +16)
    if let Some(syms) = index.symbols.get(path) {
        let mut hits: Vec<String> = Vec::new();
        for symbol in syms
            .functions
            .iter()
            .chain(syms.structs.iter())
            .chain(syms.exports.iter())
        {
            let lower = symbol.to_ascii_lowercase();
            if query_words.iter().any(|w| lower.contains(w)) && !hits.contains(symbol) {
                hits.push(symbol.clone());
            }
        }
        if !hits.is_empty() {
            let bonus = hits.len().min(SYMBOL_CAP as usize / 8) as u64 * 8;
            total += bonus.min(SYMBOL_CAP);
            for h in hits.into_iter().take(2) {
                reasons.push(format!("symbol: {h}"));
            }
            reasons.push("symbol hit(s)".to_string());
        }
    }
    // git-changed (+4)
    if changed.contains(path) {
        total += 4;
        reasons.push("git-changed".to_string());
    }
    (total, reasons)
}

/// Forward map: file → directly imported files.
fn forward_map(deps: &[DepEdge]) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for e in deps {
        map.entry(e.from.clone()).or_default().push(e.to.clone());
    }
    for v in map.values_mut() {
        v.sort();
        v.dedup();
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, loc: u64) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            language: "rust".to_string(),
            size: 0,
            loc,
            ext: "rs".to_string(),
            important: false,
            secret: false,
            binary: false,
        }
    }

    fn symbols(path: &str, fns: &[&str]) -> (String, PerFileSymbols) {
        (
            path.to_string(),
            PerFileSymbols {
                structs: vec![],
                functions: fns.iter().map(|s| s.to_string()).collect(),
                imports: vec![],
                exports: vec![],
            },
        )
    }

    fn sample_index() -> RetrievalIndex {
        RetrievalIndex {
            files: vec![
                file("src/auth.rs", 100),
                file("src/database.rs", 200),
                file("src/session.rs", 50),
                file("src/jwt.rs", 40),
                file("README.md", 30),
            ],
            symbols: BTreeMap::from([
                ("src/auth.rs".to_string(), {
                    let (_, s) = symbols("src/auth.rs", &["authenticate", "refresh_token"]);
                    s
                }),
                ("src/database.rs".to_string(), {
                    let (_, s) = symbols("src/database.rs", &["connect", "query"]);
                    s
                }),
            ]),
            deps: vec![
                DepEdge { from: "src/auth.rs".into(), to: "src/database.rs".into(), via: "crate::database".into() },
                DepEdge { from: "src/auth.rs".into(), to: "src/jwt.rs".into(), via: "crate::jwt".into() },
            ],
            mtimes: BTreeMap::new(),
        }
    }

    #[test]
    fn word_tokens_split_camels_and_snakes() {
        assert_eq!(word_tokens("UserService"), vec!["user".to_string(), "service".to_string()]);
        assert_eq!(word_tokens("refresh_token"), vec!["refresh".to_string(), "token".to_string()]);
        assert_eq!(word_tokens("auth.rs::Connect"), vec!["auth".to_string(), "rs".to_string(), "connect".to_string()]);
    }

    #[test]
    fn retrieves_by_filename_symbol_and_dependency_hops() {
        let idx = sample_index();
        let changed = HashSet::new();
        let opts = RetrieveOptions::default();

        let results = retrieve("auth", &idx, &changed, &opts);
        assert!(!results.is_empty());
        // auth.rs: filename exact (+10) + symbol authenticate (+8) = 18 → seed.
        let auth = &results[0];
        assert_eq!(auth.path, "src/auth.rs");
        // filename exact (+10) + path segment "auth" (+5) + symbol
        // "authenticate" contains "auth" (+8) = 23 → seed.
        assert_eq!(auth.score, 23);
        // database.rs reached 1 hop (+3) and jwt.rs (+3) both >= 3 → injected.
        let paths: Vec<&str> = results.iter().map(|r| r.path.as_str()).collect();
        assert!(paths.contains(&"src/database.rs"));
        assert!(paths.contains(&"src/jwt.rs"));
        assert!(paths.len() <= opts.top_k);
    }

    #[test]
    fn git_changed_boost_and_threshold_filtering() {
        let idx = sample_index();
        let changed = HashSet::from(["src/database.rs".to_string()]);
        let opts = RetrieveOptions::default();

        let results = retrieve("nonsensequeryzz", &idx, &changed, &opts);
        // database.rs: git-changed (+4) → injected; everything else below 3.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "src/database.rs");
        assert_eq!(results[0].score, 4);
    }

    #[test]
    fn empty_query_seeds_from_changed_and_recent() {
        let idx = sample_index();
        let changed = HashSet::from(["src/session.rs".to_string()]);
        let mut opts = RetrieveOptions::default();
        opts.top_k = 10;

        let results = retrieve("", &idx, &changed, &opts);
        let paths: Vec<&str> = results.iter().map(|r| r.path.as_str()).collect();
        assert!(paths.contains(&"src/session.rs"));
    }

    #[test]
    fn no_rust_symbols_still_discovers_by_path() {
        let idx = RetrievalIndex {
            files: vec![file("web/static/style.css", 10)],
            symbols: BTreeMap::new(),
            deps: vec![],
            mtimes: BTreeMap::new(),
        };
        let results = retrieve(
            "style",
            &idx,
            &HashSet::new(),
            &RetrieveOptions::default(),
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "web/static/style.css");
        // filename exact (+10) + path segment contain "style" (+5).
        assert_eq!(results[0].score, 15);
    }

    #[test]
    fn load_returns_none_without_index() {
        assert!(RetrievalIndex::load(Path::new("C:/definitely/missing/xencode")).is_none());
    }
}