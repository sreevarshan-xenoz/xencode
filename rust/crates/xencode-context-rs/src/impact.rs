//! Who has to be re-checked when one file changes.
//!
//! The question a model asks right before an edit, and the index already holds
//! the answer in a limited form: `deps.json` is a set of file→file edges, each
//! one made because some file wrote a `use` path, a `mod` declaration or an
//! `impl Trait for Type` that resolves to the other file. Walking those edges
//! backwards gives the consumers of a file without opening a single source file.
//!
//! What that list is *not* is the reason this module says so out loud in its
//! output. An edge is a name resolved through a module path, not a type-checked
//! call site: a file appears here because it links to the edited file, which is
//! weaker than proving it calls the thing being edited. Nothing here reads the
//! bodies of files, so nothing here can claim more than the graph holds.

use crate::symbols::{DepEdge, PerFileSymbols};
use std::collections::{BTreeMap, BTreeSet};

/// How far back the walk goes. The same bound [`crate::affected_dependents`]
/// uses, so the two surfaces cannot disagree about what "affected" means.
pub const IMPACT_MAX_HOPS: usize = crate::advise::AFFECTED_MAX_HOPS;

/// Cap on the declared names shown, because the point is the surface, not a
/// listing of a large file.
pub const DECLARED_CAP: usize = 40;

/// One file on the consumer side of the edited file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactedFile {
    pub file: String,
    /// Every path the index resolved between the two files, so the link can be
    /// checked rather than trusted.
    pub via: Vec<String>,
    /// `1` links to the edited file itself, through its own `use`, `mod` or
    /// `impl`; `2` and `3` are reached through a file that does.
    pub hops: usize,
    /// Whether this file's own `use` statements name the symbol that was asked
    /// about. Always `false` when no symbol was asked about.
    pub uses_symbol: bool,
}

/// The whole answer for one edited file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactReport {
    /// The file as the index stores it, which a tail match may have widened.
    pub target: String,
    /// The symbol asked about, if any.
    pub symbol: Option<String>,
    /// Names the target declares, types first, for the reader to aim at.
    pub declared: Vec<String>,
    /// Declared names beyond the [`DECLARED_CAP`] shown here.
    pub declared_more: usize,
    /// Everything reported, closest first.
    pub files: Vec<ImpactedFile>,
    /// The snapshot's own size, so an empty list reads as a fact about this
    /// index rather than a promise about the code.
    pub indexed_files: usize,
    pub edges: usize,
}

impl ImpactReport {
    /// Files that link to the target themselves.
    pub fn direct(&self) -> impl Iterator<Item = &ImpactedFile> {
        self.files.iter().filter(|f| f.hops == 1)
    }

    /// Files reached through those.
    pub fn through(&self) -> impl Iterator<Item = &ImpactedFile> {
        self.files.iter().filter(|f| f.hops > 1)
    }

    /// The confidence statement the caller owes the reader: what an edge is
    /// here, and what it therefore does not prove.
    pub fn basis(&self) -> String {
        format!(
            "From the index of {} Rust files and {} resolved edges. An edge means a \
             file wrote a `use` path, a `mod` declaration or an `impl Trait for Type` \
             that resolves to this one — name resolution through module paths, not a \
             type-checked call site. A file listed here links this one; only reading it \
             would show whether it touches what you are editing.",
            self.indexed_files, self.edges
        )
    }
}

/// The names a file declares, types before functions. A `#[test]` function is
/// left out: it has no consumer, and a file whose real surface is a dozen public
/// names would otherwise be reported as declaring one name per test in it.
fn declared_surface(sym: &PerFileSymbols) -> Vec<String> {
    let mut types: BTreeSet<String> = BTreeSet::new();
    for group in [&sym.structs, &sym.enums, &sym.traits, &sym.types] {
        types.extend(group.iter().cloned());
    }
    let mut rest: BTreeSet<String> = BTreeSet::new();
    for group in [&sym.functions, &sym.exports, &sym.mods] {
        rest.extend(group.iter().cloned());
    }
    let tests: BTreeSet<&String> = sym.tests.iter().collect();
    rest.retain(|name| !types.contains(name) && !tests.contains(name));
    let mut out: Vec<String> = types.into_iter().collect();
    out.extend(rest);
    out
}

/// Does one `use` payload name `symbol`?
///
/// The payload is the statement with `use` and `;` removed, so this reads path
/// segments rather than substrings: `crate::symbols::rap` names `rap` and
/// `crate::symbols::wrap` does not, even though one contains the other. A brace
/// group is read member by member and `as` is taken off the end, because
/// `use x::build_graph as bg` names `build_graph`.
fn payload_names_symbol(payload: &str, symbol: &str) -> bool {
    payload.split("::").any(|segment| {
        segment
            .split(" as ")
            .next()
            .unwrap_or(segment)
            .split(['{', '}', ','])
            .any(|name| name.trim() == symbol)
    })
}

/// Which of one file's own `use` statements name the symbol. A file that reaches
/// the target through `mod` or `impl` alone has nothing here, which is correctly
/// "links the module, does not name it".
fn names_symbol(symbols: &BTreeMap<String, PerFileSymbols>, file: &str, symbol: &str) -> bool {
    symbols
        .get(file)
        .map(|sym| {
            sym.imports
                .iter()
                .any(|payload| payload_names_symbol(payload, symbol))
        })
        .unwrap_or(false)
}

/// Which indexed paths match what was asked for: the stored path exactly, or as
/// its tail, so `symbols.rs` finds `crates/x/src/symbols.rs` however the caller
/// spells it.
fn matching_targets(all: impl Iterator<Item = String>, asked: &str) -> Vec<String> {
    let asked = asked
        .trim()
        .trim_start_matches("./")
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_string();
    if asked.is_empty() {
        return Vec::new();
    }
    let as_suffix = format!("/{asked}");
    all.filter(|key| *key == asked || key.ends_with(&as_suffix))
        .collect()
}

/// Walk the edges backwards from `target`, one hop at a time, never revisiting a
/// file — which is also what stops a cycle from being walked for ever.
fn dependents_by_hop(
    graph: &[DepEdge],
    target: &str,
    max_hops: usize,
) -> Vec<(String, usize, Vec<String>)> {
    let mut reverse: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    for edge in graph {
        if edge.from == edge.to {
            continue;
        }
        reverse
            .entry(edge.to.as_str())
            .or_default()
            .push((edge.from.as_str(), edge.via.as_str()));
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    seen.insert(target.to_string());
    let mut out: Vec<(String, usize, Vec<String>)> = Vec::new();
    let mut frontier: Vec<String> = vec![target.to_string()];
    for hop in 1..=max_hops {
        let mut next: Vec<String> = Vec::new();
        for file in &frontier {
            let Some(links) = reverse.get(file.as_str()) else {
                continue;
            };
            // One entry per depending file, carrying every path between them:
            // the same consumer reached two ways is not two consumers.
            let mut by_from: BTreeMap<&str, Vec<String>> = BTreeMap::new();
            for (from, via) in links {
                by_from.entry(from).or_default().push(via.to_string());
            }
            for (from, mut vias) in by_from {
                if !seen.insert(from.to_string()) {
                    continue;
                }
                vias.sort();
                vias.dedup();
                out.push((from.to_string(), hop, vias));
                next.push(from.to_string());
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    out.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// The report for one target, from a graph and symbol map already in memory.
/// Split from [`impact_from_snapshot`] so the walk can be tested without a disk.
pub fn impact(
    graph: &[DepEdge],
    symbols: &BTreeMap<String, PerFileSymbols>,
    target: &str,
    symbol: Option<&str>,
    max_hops: usize,
) -> ImpactReport {
    let surface = symbols
        .get(target)
        .map(declared_surface)
        .unwrap_or_default();
    let files = dependents_by_hop(graph, target, max_hops)
        .into_iter()
        .map(|(file, hops, via)| ImpactedFile {
            uses_symbol: symbol
                .map(|name| names_symbol(symbols, &file, name))
                .unwrap_or(false),
            file,
            via,
            hops,
        })
        .collect();
    ImpactReport {
        target: target.to_string(),
        symbol: symbol.map(|s| s.to_string()),
        declared_more: surface.len().saturating_sub(DECLARED_CAP),
        declared: surface.into_iter().take(DECLARED_CAP).collect(),
        files,
        indexed_files: symbols.len(),
        edges: graph.len(),
    }
}

/// The same report read from the `.xencode` snapshot written by
/// [`crate::init_project`] — the single read path so no surface can disagree
/// with the index on disk. `Err(NoIndex)` when there is no index,
/// `Err(NotIndexed)` when the index has no such file (naming the closest),
/// `Err(AmbiguousTarget)` when a tail matches several files.
pub fn impact_from_snapshot(
    root: &std::path::Path,
    asked: &str,
    symbol: Option<&str>,
) -> Result<ImpactReport, crate::ContextError> {
    use crate::index::{deps_json_path, read_json, symbols_json_path};
    let xencode = root.join(crate::init::XENCODE_DIR);
    let symbols: BTreeMap<String, PerFileSymbols> =
        read_json(&symbols_json_path(&xencode)).unwrap_or_default();
    if symbols.is_empty() {
        return Err(crate::ContextError::NoIndex(xencode));
    }
    let matches = matching_targets(symbols.keys().cloned(), asked);
    let target = match matches.as_slice() {
        [] => Err(crate::ContextError::NotIndexed {
            asked: asked.to_string(),
            near: near_names(&symbols, asked),
            indexed: symbols.len(),
        }),
        [only] => Ok(only.clone()),
        many => Err(crate::ContextError::AmbiguousTarget {
            asked: asked.to_string(),
            matches: many.to_vec(),
        }),
    }?;
    let graph: Vec<DepEdge> = read_json(&deps_json_path(&xencode)).unwrap_or_default();
    Ok(impact(&graph, &symbols, &target, symbol, IMPACT_MAX_HOPS))
}

/// Paths in the index sharing the asked-for file name, so a wrong directory is
/// answered with the right path instead of "not found".
fn near_names(symbols: &BTreeMap<String, PerFileSymbols>, asked: &str) -> Vec<String> {
    let leaf = asked.rsplit('/').next().unwrap_or(asked);
    let mut near: Vec<String> = symbols
        .keys()
        .filter(|key| key.rsplit('/').next().unwrap_or(key) == leaf)
        .cloned()
        .collect();
    if near.is_empty() {
        near = symbols.keys().cloned().collect();
    }
    near.sort();
    near.truncate(6);
    near
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::{build_graph, extract_rust_symbols};

    fn symbols_of(files: &[(&str, &str)]) -> BTreeMap<String, PerFileSymbols> {
        files
            .iter()
            .map(|(p, c)| (p.to_string(), extract_rust_symbols(c)))
            .collect()
    }

    fn paths_of(files: &[(&str, &str)]) -> Vec<String> {
        files.iter().map(|(p, _)| p.to_string()).collect()
    }

    fn files_in(pairs: &[(&str, &str)]) -> (Vec<DepEdge>, BTreeMap<String, PerFileSymbols>) {
        // `src/lib.rs` is part of every fixture: the crate root is what makes a
        // `crate::…` path resolve to a file at all, and an empty one adds no edges.
        let mut all: Vec<(&str, &str)> = vec![("src/lib.rs", "")];
        all.extend_from_slice(pairs);
        let symbols = symbols_of(&all);
        let graph = build_graph(&paths_of(&all), &symbols);
        (graph, symbols)
    }

    fn listed(report: &ImpactReport) -> Vec<(&str, usize)> {
        report
            .files
            .iter()
            .map(|f| (f.file.as_str(), f.hops))
            .collect()
    }

    #[test]
    fn a_file_that_uses_another_is_reported_as_its_dependent_with_the_path() {
        let (graph, symbols) = files_in(&[
            ("src/leaf.rs", "pub fn build() {}\npub struct Thing;\n"),
            (
                "src/mid.rs",
                "use crate::leaf::build;\npub fn middle() {}\n",
            ),
            ("src/top.rs", "use crate::mid::middle;\nfn top() {}\n"),
        ]);
        assert_eq!(graph.len(), 2, "{graph:?}");
        let report = impact(&graph, &symbols, "src/leaf.rs", None, IMPACT_MAX_HOPS);
        assert_eq!(listed(&report), vec![("src/mid.rs", 1), ("src/top.rs", 2)]);
        assert_eq!(report.files[0].via, vec!["crate::leaf"]);
        assert_eq!(report.edges, 2);
        assert_eq!(report.indexed_files, 4);
    }

    #[test]
    fn a_file_that_declares_a_module_is_a_dependent_of_that_module() {
        // A `mod` line carries no `use`, so a module tree would otherwise be files
        // with no edges between them at all.
        let files = [
            ("src/lib.rs", "mod leaf;\n"),
            ("src/leaf.rs", "pub fn build() {}\n"),
        ];
        let symbols = symbols_of(&files);
        let graph = build_graph(&paths_of(&files), &symbols);
        assert_eq!(graph.len(), 1, "{graph:?}");
        let report = impact(&graph, &symbols, "src/leaf.rs", None, IMPACT_MAX_HOPS);
        assert_eq!(listed(&report), vec![("src/lib.rs", 1)]);
        assert_eq!(report.files[0].via, vec!["mod leaf"]);
    }

    #[test]
    fn the_thing_a_file_implements_links_back_to_the_file_that_declares_it() {
        let (graph, symbols) = files_in(&[
            ("src/spell.rs", "pub trait Speak { fn say(&self) -> String; }\n"),
            (
                "src/sayer.rs",
                "use crate::spell::Speak;\nstruct S;\nimpl Speak for S {\n    fn say(&self) -> String { String::new() }\n}\n",
            ),
        ]);
        // Two edges, because this file both writes the `use` and implements the
        // trait. An impl alone is a real edge: Rust lets you write one with no
        // `use` of the trait in sight.
        assert_eq!(graph.len(), 2, "{graph:?}");
        let report = impact(&graph, &symbols, "src/spell.rs", None, IMPACT_MAX_HOPS);
        assert_eq!(listed(&report), vec![("src/sayer.rs", 1)]);
        assert_eq!(report.files[0].via, vec!["crate::spell", "impl Speak"]);
        let impl_only = {
            let symbols = symbols_of(&[
                ("src/lib.rs", ""),
                (
                    "src/sayer.rs",
                    "struct S;\nimpl Speak for S {\n    fn say(&self) -> String { String::new() }\n}\n",
                ),
                ("src/spell.rs", "pub trait Speak { fn say(&self) -> String; }\n"),
            ]);
            let graph = build_graph(
                &paths_of(&[
                    ("src/lib.rs", ""),
                    ("src/spell.rs", ""),
                    ("src/sayer.rs", ""),
                ]),
                &symbols,
            );
            impact(&graph, &symbols, "src/spell.rs", None, IMPACT_MAX_HOPS)
        };
        assert_eq!(listed(&impl_only), vec![("src/sayer.rs", 1)]);
        assert_eq!(impl_only.files[0].via, vec!["impl Speak"]);
    }

    #[test]
    fn what_is_declared_is_offered_so_the_reader_can_aim() {
        let (graph, symbols) =
            files_in(&[("src/leaf.rs", "pub fn build() {}\npub struct Thing;\n")]);
        let report = impact(&graph, &symbols, "src/leaf.rs", None, IMPACT_MAX_HOPS);
        assert_eq!(report.declared, vec!["Thing", "build"]);
        assert_eq!(report.declared_more, 0);
    }

    #[test]
    fn a_test_function_is_not_part_of_the_surface_a_consumer_could_reach() {
        let (graph, symbols) =
            files_in(&[("src/t.rs", "#[test]\nfn checks() {}\npub fn build() {}\n")]);
        let report = impact(&graph, &symbols, "src/t.rs", None, IMPACT_MAX_HOPS);
        assert_eq!(report.declared, vec!["build"]);
    }

    #[test]
    fn asking_a_symbol_names_which_consumers_write_that_name_in_their_use() {
        let (graph, symbols) = files_in(&[
            ("src/leaf.rs", "pub fn build() {}\npub fn other() {}\n"),
            (
                "src/mid.rs",
                "use crate::leaf::build;\npub fn middle() {}\n",
            ),
            (
                "src/braces.rs",
                "use crate::leaf::{build, other};\nfn b() {}\n",
            ),
            (
                "src/alias.rs",
                "use crate::leaf::build as make;\nfn a() {}\n",
            ),
        ]);
        let report = impact(
            &graph,
            &symbols,
            "src/leaf.rs",
            Some("build"),
            IMPACT_MAX_HOPS,
        );
        let naming: Vec<&str> = report
            .files
            .iter()
            .filter(|f| f.uses_symbol)
            .map(|f| f.file.as_str())
            .collect();
        assert_eq!(naming, vec!["src/alias.rs", "src/braces.rs", "src/mid.rs"]);
        let other = impact(
            &graph,
            &symbols,
            "src/leaf.rs",
            Some("other"),
            IMPACT_MAX_HOPS,
        );
        let naming_other: Vec<&str> = other
            .files
            .iter()
            .filter(|f| f.uses_symbol)
            .map(|f| f.file.as_str())
            .collect();
        assert_eq!(naming_other, vec!["src/braces.rs"]);
    }

    #[test]
    fn a_segment_that_merely_starts_with_the_name_is_not_a_hit() {
        assert!(payload_names_symbol("crate::symbols::rap", "rap"));
        assert!(!payload_names_symbol("crate::symbols::wrap", "rap"));
        assert!(payload_names_symbol("crate::a::build as make", "build"));
        assert!(payload_names_symbol("crate::a::{one, build}", "build"));
        assert!(!payload_names_symbol("crate::a::{one, two}", "build"));
    }

    #[test]
    fn a_file_with_no_consumers_says_so_without_inventing_any() {
        let (graph, symbols) = files_in(&[
            ("src/leaf.rs", "pub fn build() {}\n"),
            (
                "src/mid.rs",
                "use crate::leaf::build;\npub fn middle() {}\n",
            ),
            ("src/top.rs", "use crate::mid::middle;\nfn top() {}\n"),
        ]);
        let report = impact(&graph, &symbols, "src/top.rs", None, IMPACT_MAX_HOPS);
        assert!(report.files.is_empty());
        // The honest reading is that this file is a leaf of this graph, not that
        // nothing anywhere depends on it — hence the size of the index alongside.
        assert_eq!(report.indexed_files, 4);
        assert_eq!(report.edges, 2);
        assert!(report.basis().contains("not a type-checked call site"));
    }

    #[test]
    fn the_walk_stops_at_the_hop_bound_and_never_loops() {
        let (graph, symbols) = files_in(&[
            ("src/a.rs", "use crate::b::bx;\npub fn ax() {}\n"),
            ("src/b.rs", "use crate::c::cx;\npub fn bx() {}\n"),
            ("src/c.rs", "use crate::a::ax;\npub fn cx() {}\n"),
        ]);
        assert_eq!(graph.len(), 3, "{graph:?}");
        let one = impact(&graph, &symbols, "src/a.rs", None, 1);
        assert_eq!(listed(&one), vec![("src/c.rs", 1)]);
        let three = impact(&graph, &symbols, "src/a.rs", None, 3);
        assert_eq!(listed(&three), vec![("src/c.rs", 1), ("src/b.rs", 2)]);
    }

    #[test]
    fn a_path_asked_by_its_tail_finds_the_indexed_file() {
        let (_, symbols) = files_in(&[
            ("crates/x/src/symbols.rs", "pub fn build() {}\n"),
            ("crates/y/src/app.rs", "use x::symbols::build;\nfn a() {}\n"),
        ]);
        let mut keys: Vec<String> = symbols.keys().cloned().collect();
        keys.sort();
        assert_eq!(
            matching_targets(keys.iter().cloned(), "src/symbols.rs"),
            vec!["crates/x/src/symbols.rs"]
        );
        assert_eq!(
            matching_targets(keys.iter().cloned(), "symbols.rs"),
            vec!["crates/x/src/symbols.rs"]
        );
        assert!(matching_targets(keys.iter().cloned(), "nope.rs").is_empty());
        // Asking for the whole stored path, or the same path with `./` or a
        // trailing slash, is the same file.
        assert_eq!(
            matching_targets(keys.iter().cloned(), "./crates/x/src/symbols.rs/"),
            vec!["crates/x/src/symbols.rs"]
        );
    }

    #[test]
    fn one_tail_shared_by_two_files_is_refused_rather_than_guessed() {
        let (_, symbols) = files_in(&[
            ("crates/x/src/model.rs", "pub struct A;\n"),
            ("crates/y/src/model.rs", "pub struct B;\n"),
        ]);
        let matches = matching_targets(symbols.keys().cloned(), "src/model.rs");
        assert_eq!(
            matches,
            vec!["crates/x/src/model.rs", "crates/y/src/model.rs"]
        );
    }

    #[test]
    fn near_names_points_at_the_path_the_index_actually_has() {
        let (_, symbols) = files_in(&[
            ("crates/x/src/symbols.rs", "pub fn build() {}\n"),
            ("crates/y/src/main.rs", "fn main() {}\n"),
        ]);
        assert_eq!(
            near_names(&symbols, "src/missing/symbols.rs"),
            vec!["crates/x/src/symbols.rs"]
        );
        // Nothing shares the name at all: the answer is a short listing, not nothing.
        assert_eq!(near_names(&symbols, "absent.rs").len(), 3);
    }

    #[test]
    fn the_same_consumer_reached_two_ways_is_one_consumer_carrying_both_paths() {
        let (_, symbols) = files_in(&[("src/leaf.rs", "pub struct Thing;\npub fn build() {}\n")]);
        let graph = vec![
            DepEdge {
                from: "src/mid.rs".to_string(),
                to: "src/leaf.rs".to_string(),
                via: "crate::leaf".to_string(),
            },
            DepEdge {
                from: "src/mid.rs".to_string(),
                to: "src/leaf.rs".to_string(),
                via: "super::leaf".to_string(),
            },
        ];
        let report = impact(&graph, &symbols, "src/leaf.rs", None, IMPACT_MAX_HOPS);
        assert_eq!(report.files.len(), 1);
        assert_eq!(report.files[0].via, vec!["crate::leaf", "super::leaf"]);
        assert_eq!(report.declared, vec!["Thing", "build"]);
    }

    fn temp_dir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "xencode-impact-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn snapshot(
        graph: &[DepEdge],
        symbols: &BTreeMap<String, PerFileSymbols>,
    ) -> std::path::PathBuf {
        let dir = temp_dir();
        let index = dir.join(crate::init::XENCODE_DIR).join("index");
        std::fs::create_dir_all(&index).unwrap();
        crate::index::write_atomic(&index.join("symbols.json"), symbols).unwrap();
        crate::index::write_atomic(&index.join("deps.json"), &graph.to_vec()).unwrap();
        dir
    }

    #[test]
    fn a_snapshot_on_disk_answers_the_same_question_as_the_graph_in_memory() {
        let (graph, symbols) = files_in(&[
            ("src/leaf.rs", "pub fn build() {}\n"),
            (
                "src/mid.rs",
                "use crate::leaf::build;\npub fn middle() {}\n",
            ),
            ("src/top.rs", "use crate::mid::middle;\nfn top() {}\n"),
        ]);
        let dir = snapshot(&graph, &symbols);
        let report = impact_from_snapshot(&dir, "src/leaf.rs", None).unwrap();
        assert_eq!(listed(&report), vec![("src/mid.rs", 1), ("src/top.rs", 2)]);
        let by_tail = impact_from_snapshot(&dir, "leaf.rs", None).unwrap();
        assert_eq!(by_tail.target, "src/leaf.rs");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_name_the_index_does_not_hold_is_answered_with_the_names_it_does() {
        let (graph, symbols) = files_in(&[("src/leaf.rs", "pub fn build() {}\n")]);
        let dir = snapshot(&graph, &symbols);
        let missing = impact_from_snapshot(&dir, "src/deep/leaf.rs", None).unwrap_err();
        let said = missing.to_string();
        assert!(said.contains("nothing in the project index"), "{said}");
        assert!(said.contains("src/leaf.rs"), "{said}");
        let absent = impact_from_snapshot(&dir, "nothing_here.rs", None).unwrap_err();
        // Nothing shares the name, so the answer is a short listing rather than silence.
        assert!(absent.to_string().contains("src/leaf.rs"), "{absent}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_tail_matching_two_indexed_files_is_refused_with_both_paths() {
        let files = [
            ("crates/x/src/model.rs", "pub struct A;\n"),
            ("crates/y/src/model.rs", "pub struct B;\n"),
        ];
        let symbols = symbols_of(&files);
        let graph = build_graph(&paths_of(&files), &symbols);
        let dir = snapshot(&graph, &symbols);
        let said = impact_from_snapshot(&dir, "src/model.rs", None)
            .unwrap_err()
            .to_string();
        assert!(
            said.contains("matches more than one indexed file"),
            "{said}"
        );
        assert!(said.contains("crates/x/src/model.rs"), "{said}");
        assert!(said.contains("crates/y/src/model.rs"), "{said}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn with_no_index_at_all_the_answer_names_the_command_that_makes_one() {
        let dir = temp_dir();
        let said = impact_from_snapshot(&dir, "src/leaf.rs", None)
            .unwrap_err()
            .to_string();
        assert!(said.contains("run /init first"), "{said}");
    }
}
