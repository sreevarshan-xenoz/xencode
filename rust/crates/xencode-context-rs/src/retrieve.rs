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
//! | test named after the prompt, on a bugfix turn | +8 |
//! | history changes it alongside a seed | +5 (co-change) |
//! | committed with the newest work      | +2 (recency)   |
//!
//! The last two rows are [`crate::cochange`], and they are off by default because
//! they measured as nothing: on this repository's 25 gold probes, added to the
//! shipped arm, co-change moved recall@1 by 0.000, recall@5 by 0.000 and mean
//! reciprocal rank by -0.002 at a weight of 5, and all three by 0.000 at a weight
//! of 2 (where a file cannot enter on history alone). Two reasons, both visible in
//! the numbers: the text arm already reaches every file the history could name, so
//! there is no introduction left for history to do; and a hub file — one edited
//! alongside everything — turns the term into a popularity contest, which is what
//! the first, unfiltered run cost 0.440 of recall@1. [`crate::shape`] records the
//! two other biases that measured 0.000 on their own probes and were not shipped.
//!
//! A [`crate::shape::TaskShape`] widens one row of this table: on a turn whose
//! prompt says something is broken, a file whose own test names use the words of
//! the prompt gets [`TEST_NAME_BONUS`]. `General` changes nothing at all. Two
//! other biases were built and measured on this repository's gold set — a wider
//! symbol cap for a rename, a lift for the project's rule and manifest files on a
//! new-feature turn — and both moved mean reciprocal rank by 0.000 on their own
//! probes, so they are gone; see [`crate::shape`].
//!
//! Empty/absent queries seed from git-changed + most recently touched files.
//!
//! With `RetrieveOptions::lexical` the same candidates are additionally scored
//! by BM25 over their pseudo-documents — path, declared symbols and the file's
//! documentation prose — before the top-K is cut, so the text of a file can
//! carry it into the injection set instead of only reordering files that
//! already made it. That hybrid stage is off by default; `/ctx eval` runs both.

use crate::cochange::{COCHANGE_BONUS, RECENCY_BONUS};
use crate::index::{FileEntry, FilesIndex, Manifest};
use crate::shape::TaskShape;
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
/// How much a file whose own test names match the prompt is worth on a turn that
/// is chasing something broken. One hit is enough to earn it: a test name is a
/// sentence, so a second match on the same subject says little more.
///
/// This is the only shape bias that survived being measured. On the four
/// bugfix probes of this repository's gold set it raised mean reciprocal rank
/// from 0.050 to 0.237 over deterministic retrieval; on the hybrid arm that ships
/// it changed nothing, because that arm already ranks all four first.
pub const TEST_NAME_BONUS: u64 = 8;

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
    /// Score the candidate set with BM25 and rank by structural + lexical score
    /// before cutting to `top_k`, instead of cutting on structure alone.
    pub lexical: bool,
    /// What the lexical arm may do to a candidate list, and the knob the eval
    /// turns to price the documentation prose.
    pub lexical_docs: bool,
    /// The kind of work this turn looks like. `General` leaves every weight in
    /// [`score_file`] exactly as it is; the other shapes each move one arm of
    /// that table, and nothing else. See [`crate::shape`].
    pub shape: TaskShape,
    /// Spend the co-change term: a file this repository's history changes
    /// alongside one of the turn's seeds. Off in [`Default`] and off in
    /// [`RetrieveOptions::for_live_chat`], because it measured 0.000 recall and
    /// -0.002 mean reciprocal rank on the gold set; the eval runs it so a
    /// repository with a weaker text arm can check the number for itself.
    pub cochange: bool,
    /// Spend the recency term, on files something else already scored. Never
    /// brings a file in on its own: "committed lately" is not an answer.
    pub recency: bool,
}

impl Default for RetrieveOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: MIN_SCORE,
            expand_hops: DEFAULT_EXPAND_HOPS,
            empty_query_seeds: 6,
            lexical: false,
            lexical_docs: true,
            shape: TaskShape::General,
            cochange: false,
            recency: false,
        }
    }
}

impl RetrieveOptions {
    /// What the product does with a query: the lexical arm is on by default,
    /// because it measured better on this repository's gold set (mean
    /// reciprocal rank 0.767 against 0.329 over 25 probes), and `XCODE_HYBRID=0`
    /// switches it back off.
    ///
    /// Deliberately *not* the `Default`, which stays the structural baseline so
    /// the eval can run both arms in one process. Every live caller goes through
    /// here, so the `/ctx find` preview cannot disagree with what a turn
    /// actually sends — including which [`TaskShape`] it was retrieved as.
    pub fn for_live_chat(top_k: usize, shape: TaskShape) -> Self {
        let lexical = match std::env::var("XCODE_HYBRID") {
            // Anything that is not an explicit "no" keeps the arm on.
            Ok(v) => !matches!(v.trim(), "0" | "false" | "no" | "off"),
            Err(_) => true,
        };
        Self {
            top_k,
            lexical,
            shape,
            ..Default::default()
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
    /// Which files this repository's history changes together, from
    /// `history.json`. Empty when the repository has no mined history, which is
    /// what every index written before it existed looks like to retrieval.
    pub history: crate::cochange::CommitHistory,
}

impl RetrievalIndex {
    /// Load the retrieval index from `.xencode/` (missing/corrupt → `None`).
    pub fn load(xencode_dir: &Path) -> Option<RetrievalIndex> {
        let files: FilesIndex =
            crate::index::read_json(&crate::index::file_index_path(xencode_dir))?;
        let symbols: BTreeMap<String, PerFileSymbols> =
            crate::index::read_json(&crate::index::symbols_json_path(xencode_dir))?;
        let deps: Vec<DepEdge> =
            crate::index::read_json(&crate::index::deps_json_path(xencode_dir))?;
        let manifest: Manifest =
            crate::index::read_json(&crate::index::manifest_path(xencode_dir))?;
        let history = crate::cochange::load_history(xencode_dir);
        Some(RetrievalIndex {
            files: files.files,
            symbols,
            deps,
            mtimes: manifest.mtime_map,
            history,
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
        let mut recent: Vec<(String, u64)> =
            index.mtimes.iter().map(|(p, t)| (p.clone(), *t)).collect();
        recent.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        seeds.extend(
            recent
                .into_iter()
                .take(options.empty_query_seeds)
                .map(|(p, _)| p),
        );
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
        let (total, reasons) = score_file(
            entry,
            &query_words,
            index,
            changed,
            &forced_seeds,
            options.shape,
        );
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
                let score = scores
                    .entry(dep.clone())
                    .or_insert_with(|| Score::new(0, vec![]));
                score.total += 3;
                score.reasons.push(format!("dep hop {hop} via {file}"));
                if seen.insert(dep.clone()) {
                    next.push(dep.clone());
                }
            }
        }
        frontier = next;
    }

    // History: what this repository's commits change together. Applied after
    // the dependency walk and on the same seed set, so it points at the files
    // that live near the turn's subject without importing them.
    if !index.history.is_empty() && (options.cochange || options.recency) {
        // A mined partner list outlives the files in it: a pair recorded three
        // years ago can name a path that has since been deleted, and injecting
        // that path costs a slot in the prompt for an empty body.
        let indexed: HashSet<&str> = index.files.iter().map(|f| f.path.as_str()).collect();
        // One award per candidate file, from its strongest pair. A file that is
        // a historical partner of five of this turn's seeds is not five times
        // the evidence of one that is a partner of one — it is the same fact,
        // "this changes with that", seen repeatedly — and an uncapped sum lets
        // any repository's well-connected files outrank the answer. (Measured:
        // uncapped, `README.md` reached 575 against the right file's 157 here.)
        let mut paired: BTreeMap<&str, (u32, &str)> = BTreeMap::new();
        if options.cochange {
            for path in &seeds {
                let partners = index
                    .history
                    .get(path.as_str())
                    .map(|record| record.partners.as_slice())
                    .unwrap_or(&[]);
                for (partner, count) in partners {
                    let partner = partner.as_str();
                    if partner == path.as_str() || !indexed.contains(partner) {
                        continue;
                    }
                    let best = paired.entry(partner).or_insert((0, path.as_str()));
                    if *count > best.0 {
                        *best = (*count, path.as_str());
                    }
                }
            }
        }
        for (partner, (count, seed)) in paired {
            let score = scores
                .entry(partner.to_string())
                .or_insert_with(|| Score::new(0, vec![]));
            score.total += COCHANGE_BONUS;
            score
                .reasons
                .push(format!("co-change: {count} commit(s) with {seed}"));
        }
        if options.recency {
            for (path, score) in scores.iter_mut() {
                // Only files something else already valued: recency on its own
                // says the file was worked on, not that it answers the query.
                if score.total > 0 && index.history.is_recent(path.as_str()) {
                    score.total += RECENCY_BONUS;
                    score.reasons.push("recently committed".to_string());
                }
            }
        }
    }

    let candidates: Vec<(String, u64, Vec<String>)> = scores
        .into_iter()
        .map(|(path, s)| (path, s.total, s.reasons))
        .collect();
    if options.lexical && !query_words.is_empty() {
        // The lexical arm ranks the whole index, so a file only its text
        // describes can enter the candidates rather than only reorder those
        // that structural scoring already surfaced.
        return crate::embed::hybrid_select(
            index,
            &candidates,
            query,
            options.top_k,
            options.lexical_docs,
            options.min_score,
        );
    }
    let mut results: Vec<RetrievedFile> = candidates
        .into_iter()
        .filter(|(_, total, _)| *total >= options.min_score)
        .map(|(path, score, reasons)| RetrievedFile {
            path,
            score,
            reasons,
        })
        .collect();
    results.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
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
    shape: TaskShape,
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
    if !reasons.iter().any(|r| r == "filename exact")
        && query_words.iter().any(|w| name.contains(w))
    {
        total += 6;
        reasons.push("filename substring".to_string());
    }
    // path segment match (+5)
    if query_words.iter().any(|w| {
        path.split('/')
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
            .chain(syms.enums.iter())
            .chain(syms.traits.iter())
            .chain(syms.types.iter())
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
    // The one shape arm. It widens a signal the table above already has rather
    // than inventing a new one.
    if shape == TaskShape::Bugfix {
        if let Some(syms) = index.symbols.get(path) {
            if test_names_match(syms, query_words) {
                total += TEST_NAME_BONUS;
                reasons.push("a test of this behaviour lives here".to_string());
            }
        }
    }
    (total, reasons)
}

/// English words that appear inside a test's name as grammar rather than as its
/// subject.
///
/// A test name is a sentence, and sentences are mostly `the`, `is`, `and`,
/// `with`. Matching those would make the bonus fire on nearly every prompt — a
/// count of this workspace's test names found `the` in 329 of them, `a` in 283,
/// `and` in 236, `is` in 176 — which is the same no-information the marker
/// itself carries: 100 of the 127 Rust files here contain a test, so "this file
/// has a test" says nothing and neither does "this test mentions `with`".
/// Anything three characters or shorter falls out on the same reasoning without
/// needing to be listed.
fn is_grammar_word(word: &str) -> bool {
    word.len() <= 3
        || matches!(
            word,
            "a" | "an"
                | "the"
                | "and"
                | "or"
                | "nor"
                | "but"
                | "if"
                | "then"
                | "than"
                | "as"
                | "at"
                | "by"
                | "for"
                | "from"
                | "in"
                | "into"
                | "of"
                | "off"
                | "on"
                | "onto"
                | "out"
                | "over"
                | "per"
                | "so"
                | "to"
                | "up"
                | "upon"
                | "via"
                | "with"
                | "within"
                | "without"
                | "this"
                | "that"
                | "these"
                | "those"
                | "there"
                | "here"
                | "they"
                | "them"
                | "their"
                | "we"
                | "you"
                | "he"
                | "she"
                | "his"
                | "her"
                | "its"
                | "our"
                | "is"
                | "are"
                | "was"
                | "were"
                | "be"
                | "been"
                | "being"
                | "am"
                | "do"
                | "does"
                | "did"
                | "done"
                | "have"
                | "has"
                | "had"
                | "not"
                | "no"
                | "none"
                | "all"
                | "any"
                | "each"
                | "every"
                | "both"
                | "other"
                | "some"
                | "such"
                | "only"
                | "same"
                | "too"
                | "very"
                | "just"
                | "also"
                | "still"
                | "more"
                | "most"
                | "one"
                | "two"
        )
}

/// Whether any test declared in this file is named after the words the prompt
/// used.
///
/// One shared word is enough, because a test name states a behaviour —
/// `soft_compact_drops_oldest_30pct_but_keeps_decisions` — and that is the
/// vocabulary a broken thing gets reported in. Pieces are matched whole rather
/// than as substrings, unlike the symbol arm above, because a symbol is an
/// identifier and a test name is English: `fix` is a substring of `fixture`, and
/// the two are not the same subject.
fn test_names_match(symbols: &PerFileSymbols, query_words: &[String]) -> bool {
    symbols.tests.iter().any(|test| {
        test.to_ascii_lowercase()
            .split('_')
            .any(|piece| !is_grammar_word(piece) && query_words.iter().any(|word| word == piece))
    })
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
    use crate::cochange::{CommitHistory, FileCommits};

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
                mods: vec![],
                ..Default::default()
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
                DepEdge {
                    from: "src/auth.rs".into(),
                    to: "src/database.rs".into(),
                    via: "crate::database".into(),
                },
                DepEdge {
                    from: "src/auth.rs".into(),
                    to: "src/jwt.rs".into(),
                    via: "crate::jwt".into(),
                },
            ],
            mtimes: BTreeMap::new(),
            history: Default::default(),
        }
    }

    #[test]
    fn word_tokens_split_camels_and_snakes() {
        assert_eq!(
            word_tokens("UserService"),
            vec!["user".to_string(), "service".to_string()]
        );
        assert_eq!(
            word_tokens("refresh_token"),
            vec!["refresh".to_string(), "token".to_string()]
        );
        assert_eq!(
            word_tokens("auth.rs::Connect"),
            vec!["auth".to_string(), "rs".to_string(), "connect".to_string()]
        );
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
        let opts = RetrieveOptions {
            top_k: 10,
            ..Default::default()
        };

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
            history: Default::default(),
        };
        let results = retrieve("style", &idx, &HashSet::new(), &RetrieveOptions::default());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "web/static/style.css");
        // filename exact (+10) + path segment contain "style" (+5).
        assert_eq!(results[0].score, 15);
    }

    #[test]
    fn lexical_mode_reaches_a_file_the_structural_pass_missed() {
        // `session_store` shares no word with the query, declares no matching
        // symbol and is not near any seed, so the structural pipeline cannot
        // surface it at all. Only its documentation can.
        let idx = RetrievalIndex {
            files: vec![
                file("src/auth.rs", 100),
                file("src/session_store.rs", 60),
                file("src/jwt.rs", 40),
            ],
            symbols: BTreeMap::from([(
                "src/session_store.rs".to_string(),
                PerFileSymbols {
                    docs: "Holds the signed cookie across a restart.".to_string(),
                    ..Default::default()
                },
            )]),
            deps: vec![],
            mtimes: BTreeMap::new(),
            history: Default::default(),
        };
        let changed = HashSet::new();
        let query = "signed cookie restart";

        let structural = retrieve(query, &idx, &changed, &RetrieveOptions::default());
        let paths: Vec<&str> = structural.iter().map(|r| r.path.as_str()).collect();
        assert!(
            !paths.contains(&"src/session_store.rs"),
            "the structural arm should not find it: {paths:?}"
        );

        let hybrid = retrieve(
            query,
            &idx,
            &changed,
            &RetrieveOptions {
                lexical: true,
                ..Default::default()
            },
        );
        assert_eq!(
            hybrid.first().map(|r| r.path.as_str()),
            Some("src/session_store.rs")
        );
        assert!(hybrid[0].reasons.iter().any(|r| r.starts_with("text")));
    }

    #[test]
    fn load_returns_none_without_index() {
        assert!(RetrievalIndex::load(Path::new("C:/definitely/missing/xencode")).is_none());
    }

    /// One file, its symbols, and whether the scanner would call it a guide.
    fn shaped_file(
        path: &str,
        symbols: PerFileSymbols,
        important: bool,
    ) -> (FileEntry, (String, PerFileSymbols)) {
        let mut entry = file(path, 100);
        entry.important = important;
        (entry, (path.to_string(), symbols))
    }

    fn score_of(
        files: Vec<(FileEntry, (String, PerFileSymbols))>,
        query: &str,
        shape: TaskShape,
    ) -> Vec<RetrievedFile> {
        let idx = RetrievalIndex {
            files: files.iter().map(|(entry, _)| entry.clone()).collect(),
            symbols: files
                .into_iter()
                .map(|(_, (path, symbols))| (path, symbols))
                .collect(),
            deps: vec![],
            mtimes: BTreeMap::new(),
            history: Default::default(),
        };
        retrieve(
            query,
            &idx,
            &HashSet::new(),
            &RetrieveOptions {
                shape,
                ..Default::default()
            },
        )
    }

    fn declares(path: &str, fns: &[&str], tests: &[&str]) -> (FileEntry, (String, PerFileSymbols)) {
        shaped_file(
            path,
            PerFileSymbols {
                functions: fns.iter().map(|s| s.to_string()).collect(),
                tests: tests.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            },
            false,
        )
    }

    #[test]
    fn a_bugfix_prompt_prefers_the_file_that_tests_the_behaviour() {
        // Both files declare a `compact` and both are named for one of the two
        // subjects in the prompt, so structurally they are level; only one holds
        // a test whose name is the sentence the bug report paraphrases.
        let files = vec![
            declares(
                "src/compact.rs",
                &["compact"],
                &["soft_compact_keeps_decisions"],
            ),
            declares("src/decisions.rs", &["compact"], &[]),
        ];
        let query = "compact decisions fix";
        let general = score_of(files.clone(), query, TaskShape::General);
        let bugfix = score_of(files, query, TaskShape::Bugfix);
        assert_eq!(
            general
                .iter()
                .find(|r| r.path == "src/compact.rs")
                .unwrap()
                .score,
            general
                .iter()
                .find(|r| r.path == "src/decisions.rs")
                .unwrap()
                .score,
            "the baseline should not already separate them: {general:?}"
        );
        let tested = bugfix.iter().find(|r| r.path == "src/compact.rs").unwrap();
        assert!(
            tested
                .reasons
                .iter()
                .any(|r| r == "a test of this behaviour lives here"),
            "{:?}",
            tested.reasons
        );
        assert_eq!(bugfix.first().unwrap().path, "src/compact.rs");
    }

    #[test]
    fn a_test_name_saying_only_grammar_earns_nothing() {
        // `that`, `is`, `not`, `it` are in most test names and most prompts, so a
        // match on them alone is the same no-signal as "this file has a test".
        let files = vec![declares("src/thing.rs", &["thing"], &["that_is_not_it"])];
        let results = score_of(
            files,
            "why does that thing fail, is it not there",
            TaskShape::Bugfix,
        );
        assert!(
            !results[0]
                .reasons
                .iter()
                .any(|r| r == "a test of this behaviour lives here"),
            "{:?}",
            results[0].reasons
        );
    }

    #[test]
    fn no_bias_but_the_test_name_one_moves_a_weight() {
        // A wider symbol cap for a rename and a lift for the files the project
        // writes its rules in were both built, both measured 0.000 on their own
        // probes of this repository's gold set, and both taken out. This is the
        // guard that keeps them out: nothing besides [`TEST_NAME_BONUS`] may
        // change what a file is worth, and the cap stays 16 however many symbols
        // agree with the prompt.
        let files = vec![
            declares(
                "src/auth.rs",
                &["auth", "authenticate", "authorize", "auth_session"],
                &[],
            ),
            shaped_file("README.md", PerFileSymbols::default(), true),
        ];
        for shape in [TaskShape::General, TaskShape::Bugfix] {
            let auth = &score_of(files.clone(), "rename auth everywhere", shape)[0];
            assert_eq!(auth.path, "src/auth.rs");
            // filename substring (+6) + path segment (+5) + 16's worth of symbols.
            assert_eq!(auth.score, 11 + SYMBOL_CAP, "{shape} widened the cap");
        }
        let guide = score_of(
            files,
            "add a readme section for the new command",
            TaskShape::General,
        )
        .into_iter()
        .find(|r| r.path == "README.md")
        .expect("the readme is reachable by its own name");
        assert_eq!(guide.score, 11, "name (+6) and path (+5), and nothing else");
    }

    /// This repository's history, as the index would hand it over: `src/auth.rs`
    /// and `src/rotation.rs` are committed together and nothing else touches
    /// either, and only `auth.rs` was committed inside the recency window.
    /// `src/deleted.rs` is the partner whose file no longer exists.
    fn history_index() -> RetrievalIndex {
        let newest = 1_700_000_000u64;
        RetrievalIndex {
            files: vec![
                file("src/auth.rs", 100),
                file("src/rotation.rs", 20),
                file("src/unrelated.rs", 20),
            ],
            history: CommitHistory::from(BTreeMap::from([
                (
                    "src/auth.rs".to_string(),
                    FileCommits {
                        partners: vec![
                            ("src/rotation.rs".to_string(), 4),
                            ("src/deleted.rs".to_string(), 3),
                        ],
                        commits: 4,
                        last_commit: newest,
                        hub: false,
                    },
                ),
                (
                    "src/rotation.rs".to_string(),
                    FileCommits {
                        partners: vec![("src/auth.rs".to_string(), 4)],
                        commits: 4,
                        last_commit: newest - 40 * 86_400,
                        hub: false,
                    },
                ),
            ])),
            ..Default::default()
        }
    }

    fn paths(results: &[RetrievedFile]) -> Vec<String> {
        results.iter().map(|r| r.path.clone()).collect()
    }

    #[test]
    fn history_terms_are_off_unless_the_caller_asks() {
        let index = history_index();
        let without = retrieve("auth", &index, &HashSet::new(), &RetrieveOptions::default());
        assert_eq!(paths(&without), vec!["src/auth.rs".to_string()]);
    }

    #[test]
    fn co_change_reaches_a_file_the_query_never_named() {
        let index = history_index();
        let with = retrieve(
            "auth",
            &index,
            &HashSet::new(),
            &RetrieveOptions {
                cochange: true,
                ..Default::default()
            },
        );
        let rotation = with
            .iter()
            .find(|r| r.path == "src/rotation.rs")
            .expect("the partner file is reachable through history");
        assert_eq!(rotation.score, COCHANGE_BONUS, "history alone");
        assert!(
            rotation
                .reasons
                .iter()
                .any(|r| r == "co-change: 4 commit(s) with src/auth.rs"),
            "{:?}",
            rotation.reasons
        );
        // A file with neither a textual nor a historical link stays out.
        assert!(!paths(&with).contains(&"src/unrelated.rs".to_string()));
    }

    #[test]
    fn a_partner_deleted_since_the_commit_is_not_injected() {
        let index = history_index();
        let with = retrieve(
            "auth",
            &index,
            &HashSet::new(),
            &RetrieveOptions {
                cochange: true,
                ..Default::default()
            },
        );
        assert!(
            !paths(&with).contains(&"src/deleted.rs".to_string()),
            "the mined pair is real but the file is gone: {:?} would waste a slot",
            paths(&with)
        );
    }

    #[test]
    fn recency_lifts_a_file_already_scored_and_never_arrives_on_its_own() {
        let index = history_index();
        let recent = retrieve(
            "auth",
            &index,
            &HashSet::new(),
            &RetrieveOptions {
                recency: true,
                ..Default::default()
            },
        );
        let auth = &recent[0];
        assert_eq!(auth.path, "src/auth.rs");
        assert_eq!(
            auth.score,
            15 + RECENCY_BONUS,
            "filename exact (+10) and path segment (+5), plus the newest commit"
        );
        assert!(auth.reasons.contains(&"recently committed".to_string()));
        // rotation.rs scores nothing without the co-change term, so the fact
        // that it was committed — 40 days ago, and outside the window anyway —
        // cannot put it in the prompt by itself.
        assert_eq!(paths(&recent), vec!["src/auth.rs".to_string()]);
    }

    #[test]
    fn a_partner_outside_the_recency_window_gets_the_pair_bonus_only() {
        let index = history_index();
        let both = retrieve(
            "auth",
            &index,
            &HashSet::new(),
            &RetrieveOptions {
                cochange: true,
                recency: true,
                ..Default::default()
            },
        );
        let by_path: BTreeMap<&str, u64> =
            both.iter().map(|r| (r.path.as_str(), r.score)).collect();
        assert_eq!(by_path["src/auth.rs"], 15 + RECENCY_BONUS);
        assert_eq!(
            by_path["src/rotation.rs"], COCHANGE_BONUS,
            "committed 40 days ago: paired, not recent"
        );
    }

    /// Two files that name the same query word, so both are seeds and both are
    /// historical partners of `src/rotation.rs`.
    fn two_seed_index() -> RetrievalIndex {
        RetrievalIndex {
            files: vec![
                file("src/author.rs", 100),
                file("src/authority.rs", 100),
                file("src/rotation.rs", 20),
            ],
            history: CommitHistory::from(BTreeMap::from([
                (
                    "src/author.rs".to_string(),
                    FileCommits {
                        partners: vec![("src/rotation.rs".to_string(), 2)],
                        commits: 2,
                        last_commit: 1_700_000_000,
                        hub: false,
                    },
                ),
                (
                    "src/authority.rs".to_string(),
                    FileCommits {
                        partners: vec![("src/rotation.rs".to_string(), 5)],
                        commits: 5,
                        last_commit: 1_700_000_000,
                        hub: false,
                    },
                ),
                (
                    "src/rotation.rs".to_string(),
                    FileCommits {
                        partners: vec![
                            ("src/authority.rs".to_string(), 5),
                            ("src/author.rs".to_string(), 2),
                        ],
                        commits: 7,
                        last_commit: 1_700_000_000,
                        hub: false,
                    },
                ),
            ])),
            ..Default::default()
        }
    }

    #[test]
    fn several_seeds_that_name_the_same_partner_award_the_bonus_once() {
        let with = retrieve(
            "author",
            &two_seed_index(),
            &HashSet::new(),
            &RetrieveOptions {
                cochange: true,
                ..Default::default()
            },
        );
        let ranked = paths(&with);
        assert!(
            ranked.contains(&"src/author.rs".to_string())
                && ranked.contains(&"src/authority.rs".to_string()),
            "both files are seeds, or the test proves nothing: {ranked:?}"
        );
        let rotation = with
            .iter()
            .find(|r| r.path == "src/rotation.rs")
            .expect("the partner is reachable");
        assert_eq!(
            rotation.score, COCHANGE_BONUS,
            "two seeds paired with it, one award: {:?}",
            rotation.reasons
        );
        assert_eq!(
            rotation
                .reasons
                .iter()
                .filter(|r| r.starts_with("co-change"))
                .count(),
            1,
            "{:?}",
            rotation.reasons
        );
        assert!(
            rotation
                .reasons
                .iter()
                .any(|r| r == "co-change: 5 commit(s) with src/authority.rs"),
            "the strongest pair is the one named: {:?}",
            rotation.reasons
        );
    }

    #[test]
    fn asking_for_history_the_index_does_not_have_changes_nothing() {
        let mut index = history_index();
        index.history = CommitHistory::default();
        let empty = retrieve(
            "auth",
            &index,
            &HashSet::new(),
            &RetrieveOptions {
                cochange: true,
                recency: true,
                ..Default::default()
            },
        );
        assert_eq!(paths(&empty), vec!["src/auth.rs".to_string()]);
        assert_eq!(empty[0].score, 15, "no window to be recent in");
    }
}
