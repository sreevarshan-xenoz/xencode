//! The symbol-only repo map (AC-6) — one cheap, whole-repository orientation
//! line per file, for the turns whose budget cannot hold file bodies.
//!
//! A `Low` machine retrieves three files of ~8 KiB and a prompt on a
//! 2048-token server retrieves one, so the model sees a handful of functions
//! and nothing else about a repository of a thousand files. This tier is the
//! "nothing else" part: paths and the names they declare, no bodies, ranked so
//! that the files *near the work being asked about* come first — hop distance
//! over the dependency graph from the seed files, then how many other files
//! depend on the row (a PageRank-lite signal already computed by the same edge
//! list `impact` walks).
//!
//! Pure and disk-free like the rest of the budget logic: everything it needs is
//! already in [`RetrievalIndex`].

use crate::retrieve::RetrievalIndex;
use crate::symbols::DepEdge;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// The whole tier's ceiling. Low's fill target is 2 457 tokens and a map that
/// ate a tenth of it would be spending the orientation budget rather than
/// adding to it; 300 tokens is what 12 rows of names cost with room to spare.
pub const REPO_MAP_CAP_TOKENS: u64 = 300;
/// Rows before the map stops being a map and becomes a file listing.
pub const REPO_MAP_MAX_FILES: usize = 12;
/// Declared names per row, with `+N more` when the file declares further.
/// Three is measured, not chosen by taste: at this repository's index size,
/// five names per row filled the tier's ceiling in five rows and named the gold
/// answer in 8 of 25 queries; three names per row fits eight rows and reaches 9.
/// Names are cheap, files are the point.
pub const REPO_MAP_MAX_SYMBOLS: usize = 3;
/// How far from a seed file a row may sit and still be called near it. Two hops
/// is "the module the question touches and what that module uses".
pub const REPO_MAP_MAX_HOPS: usize = 2;

/// One row of the map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapRow {
    pub path: String,
    /// Declared names, capped to [`REPO_MAP_MAX_SYMBOLS`].
    pub symbols: Vec<String>,
    /// Hidden further names the row had to leave out.
    pub hidden_symbols: usize,
    /// Hop distance from the nearest seed file; `usize::MAX` when the file is
    /// not reachable within [`REPO_MAP_MAX_HOPS`].
    pub hops: usize,
    /// How many indexed files depend on this one.
    pub in_degree: usize,
}

/// The names a file declares, in the order the inventory holds them. Types
/// before functions, because on a capped row the type is usually what the
/// functions around it are built on.
fn declared_names(symbols: &crate::symbols::PerFileSymbols) -> Vec<String> {
    let mut names: Vec<String> = symbols.structs.clone();
    names.extend(symbols.enums.iter().cloned());
    names.extend(symbols.traits.iter().cloned());
    names.extend(symbols.types.iter().cloned());
    // Test names are in `functions` too, and they are sentences: on a tier whose
    // whole cost is names, one test name buys fewer rows of the map than the
    // declaration it happens to test.
    names.extend(
        symbols
            .functions
            .iter()
            .filter(|f| !symbols.tests.contains(f))
            .cloned(),
    );
    names
}

/// The map as ranked rows, nearest-and-most-depended-on first.
///
/// `seeds` are the files the turn is already about — retrieval hits and the
/// working-tree changes — so the map points away from what the model can see
/// rather than repeating it.
pub fn rank_repo_map(index: &RetrievalIndex, seeds: &[String]) -> Vec<MapRow> {
    let in_degree = in_degrees(&index.deps);
    let hops = hop_distances(&index.deps, seeds);

    let mut rows: Vec<MapRow> = index
        .files
        .iter()
        .filter(|f| !f.secret && !f.binary)
        .filter_map(|f| {
            // A row with no names says only that the file exists, which the
            // retrieval tier already answers better.
            let mut symbols = index
                .symbols
                .get(&f.path)
                .map(declared_names)
                .unwrap_or_default();
            if symbols.is_empty() {
                return None;
            }
            let hidden = symbols.len().saturating_sub(REPO_MAP_MAX_SYMBOLS);
            symbols.truncate(REPO_MAP_MAX_SYMBOLS);
            Some(MapRow {
                path: f.path.clone(),
                symbols,
                hidden_symbols: hidden,
                hops: hops.get(&f.path).copied().unwrap_or(usize::MAX),
                in_degree: in_degree.get(&f.path).copied().unwrap_or(0),
            })
        })
        .collect();

    rows.sort_by(|a, b| {
        a.hops
            .cmp(&b.hops)
            .then_with(|| b.in_degree.cmp(&a.in_degree))
            .then_with(|| a.path.cmp(&b.path))
    });
    rows.truncate(REPO_MAP_MAX_FILES);
    rows
}

/// The tier text: header plus one indented row per file, capped to
/// [`REPO_MAP_CAP_TOKENS`]. Empty string when the index has nothing to say.
pub fn repo_map_text(index: &RetrievalIndex, seeds: &[String]) -> String {
    let rows = rank_repo_map(index, seeds);
    let named = named_files(index);
    if rows.is_empty() {
        return String::new();
    }
    let mut text = String::from(
        "Repo map — files nearest the current work, most depended-on first, \
         names only:\n",
    );
    // Rows are added whole or not at all: a character-truncated map ends with a
    // half-written path, which is worse than no path at all. The note that says
    // how much was left out is reserved for before the last row is admitted.
    let mut used = crate::budget::est_tokens(text.len(), false);
    let note_cost = crate::budget::est_tokens(REPO_MAP_PARTIAL_NOTE.len(), false);
    let mut emitted = 0usize;
    for row in &rows {
        let near = match row.hops {
            usize::MAX => String::new(),
            0 => " [the current work]".to_string(),
            hops => format!(" [{hops} hop(s) from the current work]"),
        };
        let mut names = row.symbols.join(", ");
        if row.hidden_symbols > 0 {
            names.push_str(&format!(", +{} more", row.hidden_symbols));
        }
        let line = format!("  • {}{near}: {names}\n", row.path);
        let cost = crate::budget::est_tokens(line.len(), false);
        if used + cost + note_cost > REPO_MAP_CAP_TOKENS {
            break;
        }
        used += cost;
        emitted += 1;
        text.push_str(&line);
    }
    if emitted == 0 {
        // Even one row did not fit; the tier is not worth its header.
        return String::new();
    }
    if named > emitted {
        text.push_str(&REPO_MAP_PARTIAL_NOTE.replace("{n}", &(named - emitted).to_string()));
    }
    text
}

/// Files the map could have said something about: indexable, and declaring at
/// least one name. Everything the map leaves out is counted against this.
fn named_files(index: &RetrievalIndex) -> usize {
    index
        .files
        .iter()
        .filter(|f| !f.secret && !f.binary)
        .filter(|f| {
            index
                .symbols
                .get(&f.path)
                .is_some_and(|s| !declared_names(s).is_empty())
        })
        .count()
}

/// Said aloud when the cap cut the map short, so a small model does not conclude
/// the repository is as small as the list it was given.
const REPO_MAP_PARTIAL_NOTE: &str = "  … +{n} more files in the index, not listed\n";

/// How many files depend on each file, from the same edge list `impact` walks.
fn in_degrees(deps: &[DepEdge]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for edge in deps {
        if edge.from != edge.to {
            *counts.entry(edge.to.clone()).or_insert(0) += 1;
        }
    }
    counts
}

/// Breadth-first hop distance from the seed set over the dependency edges,
/// treated as undirected: a file the seed uses is as relevant to orienting
/// around it as a file that uses the seed. Seeds are distance 0 — the map may
/// still name them, since a seed that did not make the retrieval cut is exactly
/// the file the model should be pointed at. Distances past [`REPO_MAP_MAX_HOPS`]
/// are not recorded, so an unreachable path falls to the end of the ranking.
fn hop_distances(deps: &[DepEdge], seeds: &[String]) -> BTreeMap<String, usize> {
    let mut neighbours: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in deps {
        if edge.from == edge.to {
            continue;
        }
        neighbours
            .entry(edge.from.as_str())
            .or_default()
            .push(edge.to.as_str());
        neighbours
            .entry(edge.to.as_str())
            .or_default()
            .push(edge.from.as_str());
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut queue: VecDeque<(&str, usize)> = VecDeque::new();
    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    for seed in seeds {
        if seen.insert(seed.as_str()) {
            out.insert(seed.clone(), 0);
            queue.push_back((seed.as_str(), 0));
        }
    }
    while let Some((path, hop)) = queue.pop_front() {
        if hop >= REPO_MAP_MAX_HOPS {
            continue;
        }
        for next in neighbours.get(path).map(Vec::as_slice).unwrap_or_default() {
            if seen.insert(next) {
                out.insert(next.to_string(), hop + 1);
                queue.push_back((next, hop + 1));
            }
        }
    }
    out
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

    fn symbols(names: &[&str]) -> PerFileSymbols {
        PerFileSymbols {
            structs: names.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn index(files: &[(&str, &[&str])], deps: &[(&str, &str)]) -> RetrievalIndex {
        RetrievalIndex {
            files: files.iter().map(|(p, _)| file(p)).collect(),
            symbols: files
                .iter()
                .map(|(p, s)| (p.to_string(), symbols(s)))
                .collect(),
            deps: deps
                .iter()
                .map(|(from, to)| DepEdge {
                    from: from.to_string(),
                    to: to.to_string(),
                    via: "crate::x".to_string(),
                })
                .collect(),
            mtimes: Default::default(),
        }
    }

    #[test]
    fn a_file_everything_depends_on_leads_the_map() {
        let index = index(
            &[
                ("src/leaf.rs", &["Thing"]),
                ("src/a.rs", &["A"]),
                ("src/b.rs", &["B"]),
            ],
            &[("src/a.rs", "src/leaf.rs"), ("src/b.rs", "src/leaf.rs")],
        );
        let rows = rank_repo_map(&index, &[]);
        assert_eq!(rows[0].path, "src/leaf.rs");
        assert_eq!(rows[0].in_degree, 2);
        assert_eq!(rows[1].in_degree, 0);
    }

    #[test]
    fn distance_from_the_seed_outranks_raw_in_degree() {
        // `far` is depended on by five files and is unreachable from the seed;
        // `hub` is two hops away and depends on nothing. The map's purpose is
        // orientation around the current work, so proximity wins.
        let files = [
            ("src/seed.rs", &["Seed"][..]),
            ("src/near.rs", &["Near"]),
            ("src/hub.rs", &["Hub"]),
            ("src/far.rs", &["Far"]),
            ("src/u1.rs", &["U1"]),
            ("src/u2.rs", &["U2"]),
            ("src/u3.rs", &["U3"]),
            ("src/u4.rs", &["U4"]),
            ("src/u5.rs", &["U5"]),
        ];
        let deps = [
            ("src/seed.rs", "src/near.rs"),
            ("src/near.rs", "src/hub.rs"),
            ("src/u1.rs", "src/far.rs"),
            ("src/u2.rs", "src/far.rs"),
            ("src/u3.rs", "src/far.rs"),
            ("src/u4.rs", "src/far.rs"),
            ("src/u5.rs", "src/far.rs"),
        ];
        let index = index(&files, &deps);
        let rows = rank_repo_map(&index, &["src/seed.rs".to_string()]);
        let order: Vec<&str> = rows.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(
            order,
            vec![
                "src/seed.rs",
                "src/near.rs",
                "src/hub.rs",
                "src/far.rs",
                "src/u1.rs",
                "src/u2.rs",
                "src/u3.rs",
                "src/u4.rs",
                "src/u5.rs",
            ]
        );
        assert_eq!(rows[0].hops, 0, "the seed itself is on the map");
        assert_eq!(rows[2].hops, 2);
        assert_eq!(rows[3].in_degree, 5, "and it still ranks below a hub");
    }

    #[test]
    fn a_row_lists_the_names_the_tier_allows_and_counts_the_rest() {
        let names = ["One", "Two", "Three", "Four", "Five", "Six", "Seven"];
        let index = index(&[("src/big.rs", &names[..])], &[]);
        let text = repo_map_text(&index, &[]);
        let shown = names[..REPO_MAP_MAX_SYMBOLS].join(", ");
        assert!(
            text.contains(&format!(
                "{shown}, +{} more",
                names.len() - REPO_MAP_MAX_SYMBOLS
            )),
            "{text}"
        );
        assert!(!text.contains(names[REPO_MAP_MAX_SYMBOLS]), "{text}");
    }

    #[test]
    fn secret_and_nameless_files_are_never_rows() {
        let mut index = index(&[("src/a.rs", &["A"]), ("src/empty.rs", &[])], &[]);
        index.files.push(FileEntry {
            secret: true,
            ..file("src/.env")
        });
        index
            .symbols
            .insert("src/.env".to_string(), symbols(&["LEAKED"]));
        let text = repo_map_text(&index, &[]);
        assert!(!text.contains("src/.env"), "{text}");
        assert!(!text.contains("src/empty.rs"), "{text}");
        assert!(text.contains("src/a.rs"), "{text}");
    }

    #[test]
    fn an_empty_index_produces_no_tier() {
        let index = index(&[], &[]);
        assert_eq!(repo_map_text(&index, &[]), "");
    }

    #[test]
    fn the_map_stays_under_its_own_token_cap() {
        let wide = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let owned: Vec<[&str; 5]> = vec![[wide; 5]; 200];
        let files: Vec<(String, &[&str])> = (0..200)
            .map(|i| (format!("src/deep/module_{i}/file.rs"), &owned[i][..]))
            .collect();
        let index = RetrievalIndex {
            files: files.iter().map(|(p, _)| file(p)).collect(),
            symbols: files.iter().map(|(p, s)| (p.clone(), symbols(s))).collect(),
            deps: Vec::new(),
            mtimes: Default::default(),
        };
        let text = repo_map_text(&index, &[]);
        let tokens = crate::budget::est_tokens(text.len(), false);
        assert!(
            tokens <= REPO_MAP_CAP_TOKENS,
            "{tokens} tokens over {REPO_MAP_CAP_TOKENS}: {text}"
        );
        let rows = text.lines().filter(|l| l.starts_with("  •")).count();
        assert!(
            rows > 0 && rows < REPO_MAP_MAX_FILES,
            "{rows} rows:\n{text}"
        );
        // Wide names mean the cap is reached before the 12-row window: every
        // row that survives is complete, and the count says what was dropped.
        assert!(
            text.contains(&format!(
                "+{} more files in the index, not listed",
                200 - rows
            )),
            "{text}"
        );
        // Every admitted row is whole: the same five names, nothing cut off at
        // the end of the tier.
        let hidden = 5 - REPO_MAP_MAX_SYMBOLS;
        assert!(
            text.lines()
                .filter(|l| l.starts_with("  •"))
                .all(|l| l.ends_with(&format!(", +{hidden} more"))),
            "a capped row must be whole:\n{text}"
        );
    }
}
