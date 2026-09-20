//! Repository insights (M4) — live refactor suggestions + proactive warnings.
//!
//! Pure, deterministic analyses over the symbol graph from [`crate::symbols`].
//! Zero LLM calls: everything here runs on every save if the TUI wants it to.
//!
//!   - [`find_cycles`] — import cycles (`a ↔ b`), the classic "extract a third
//!     module" refactor trigger.
//!   - [`hub_files`] — high fan-out files worth splitting.
//!   - [`orphan_files`] — workspace files with no edges at all (dead code?).
//!   - [`broken_imports`] — workspace-anchored (`crate::`/`self::`/`super::`)
//!     imports that resolve to nothing: the "you just renamed a module and
//!     broke its importers" warning. Plain `use serde::…` imports are skipped —
//!     they name external crates, not workspace files.
//!   - [`affected_dependents`] — reverse reachability from a set of changed
//!     files: "you changed X — Y and Z depend on it, re-check them." Fed by
//!     the [`crate::watcher`] event stream.
//!
//! [`advise`] aggregates the file-level findings into sorted [`Advice`].

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use crate::symbols::{DepEdge, PerFileSymbols};

/// Out-degree at or above which a file is flagged as a hub.
pub const HUB_MIN_OUT: usize = 8;
/// Default BFS depth for [`affected_dependents`].
pub const AFFECTED_MAX_HOPS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdviceKind {
    BrokenImport,
    Cycle,
    AffectedDependent,
    Hub,
    Orphan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Advice {
    pub file: String,
    pub kind: AdviceKind,
    pub message: String,
}

/// Import cycles as sorted file lists, smallest path first (canonical rotation
/// so the same cycle is never reported twice from different entry points).
/// Self-loops are impossible by construction (see `dependency_map`) and skipped.
pub fn find_cycles(graph: &[DepEdge]) -> Vec<Vec<String>> {
    let mut adj: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for e in graph {
        if e.from == e.to {
            continue;
        }
        adj.entry(e.from.as_str())
            .or_default()
            .insert(e.to.as_str());
        adj.entry(e.to.as_str()).or_default();
    }
    // Iterative DFS with gray/black coloring. Each back edge (to a gray node)
    // yields exactly the stack slice forming that cycle.
    let mut color: HashMap<&str, u8> = HashMap::new();
    let mut stack: Vec<&str> = Vec::new();
    let mut found: BTreeSet<Vec<String>> = BTreeSet::new();
    let starts: Vec<&str> = adj.keys().copied().collect();
    for start in starts {
        if color.get(start).copied().unwrap_or(0) != 0 {
            continue;
        }
        let mut work: Vec<(&str, bool)> = vec![(start, false)];
        while let Some((node, exiting)) = work.pop() {
            if exiting {
                color.insert(node, 2);
                stack.pop();
                continue;
            }
            match color.get(node).copied().unwrap_or(0) {
                1 => {
                    if let Some(pos) = stack.iter().position(|&n| n == node) {
                        let mut cyc: Vec<String> =
                            stack[pos..].iter().map(|s| s.to_string()).collect();
                        if let Some(min) = cyc
                            .iter()
                            .enumerate()
                            .min_by(|a, b| a.1.cmp(b.1))
                            .map(|(i, _)| i)
                        {
                            cyc.rotate_left(min);
                        }
                        found.insert(cyc);
                    }
                    continue;
                }
                2 => continue,
                _ => {}
            }
            color.insert(node, 1);
            stack.push(node);
            work.push((node, true));
            if let Some(neigh) = adj.get(node) {
                for &n in neigh.iter().rev() {
                    work.push((n, false));
                }
            }
        }
    }
    found.into_iter().collect()
}

/// Files depending on at least `min_out` others, most-coupled first
/// (ties broken by path). `(file, out-degree)` pairs.
pub fn hub_files(graph: &[DepEdge], min_out: usize) -> Vec<(String, usize)> {
    let fwd = crate::symbols::dependency_map(graph);
    let mut hubs: Vec<(String, usize)> = fwd
        .into_iter()
        .filter(|(_, deps)| deps.len() >= min_out)
        .map(|(f, deps)| (f, deps.len()))
        .collect();
    hubs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    hubs
}

/// Rust files with no dependency edges in either direction, sorted.
/// Entry points (`lib.rs`/`main.rs`/`mod.rs`) are never orphans.
pub fn orphan_files(files: &[String], graph: &[DepEdge]) -> Vec<String> {
    let mut touched: HashSet<&str> = HashSet::new();
    for e in graph {
        touched.insert(e.from.as_str());
        touched.insert(e.to.as_str());
    }
    let mut orphans: Vec<String> = files
        .iter()
        .filter(|f| !touched.contains(f.as_str()))
        .filter(|f| !matches!(f.rsplit('/').next(), Some("lib.rs" | "main.rs" | "mod.rs")))
        .cloned()
        .collect();
    orphans.sort();
    orphans
}

/// Workspace-anchored imports that resolve to no workspace file, as sorted
/// `(file, import)` pairs. Only `crate::`/`self::`/`super::` imports are
/// considered — a bare `use serde::Serialize` names an external crate and a
/// missing workspace file behind a workspace qualifier almost always means a
/// move/rename broke the importer.
pub fn broken_imports(
    rust_files: &[String],
    symbols: &BTreeMap<String, PerFileSymbols>,
) -> Vec<(String, String)> {
    use crate::symbols::Qualifier;
    let file_set: HashSet<&str> = rust_files.iter().map(|s| s.as_str()).collect();
    let roots = crate::symbols::crate_roots(rust_files);
    let mut ordered: Vec<&String> = symbols.keys().collect();
    ordered.sort();
    let mut out: Vec<(String, String)> = Vec::new();
    for file in ordered {
        let Some(sym) = symbols.get(file) else {
            continue;
        };
        let root = crate::symbols::crate_root_for(file, &roots);
        for import in &sym.imports {
            let pairs = crate::symbols::parse_import(import);
            let anchored: Vec<_> = pairs
                .iter()
                .filter(|(q, _)| {
                    matches!(
                        q,
                        Qualifier::Crate | Qualifier::SelfMod | Qualifier::Super { .. }
                    )
                })
                .collect();
            if anchored.is_empty() {
                continue;
            }
            let broken = anchored.iter().any(|(q, segs)| {
                crate::symbols::resolve_pair(file, q, segs, root, &file_set).is_none()
            });
            if broken {
                out.push((file.clone(), import.clone()));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Reverse reachability: for each changed file, the sorted files that
/// (transitively) depend on it within `max_hops` — the "re-check these"
/// set behind "you just introduced a bug". Files with no dependents are
/// absent from the map.
pub fn affected_dependents(
    graph: &[DepEdge],
    changed: &[&str],
    max_hops: usize,
) -> BTreeMap<String, Vec<String>> {
    let rev = crate::symbols::dependent_map(graph);
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for &seed in changed {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let mut current = vec![seed.to_string()];
        for _ in 0..max_hops {
            let mut next: Vec<String> = Vec::new();
            for file in &current {
                if let Some(dependents) = rev.get(file) {
                    for dep in dependents {
                        if dep != seed && visited.insert(dep.clone()) {
                            next.push(dep.clone());
                        }
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            current = next;
        }
        if !visited.is_empty() {
            out.insert(seed.to_string(), visited.into_iter().collect());
        }
    }
    out
}

/// Aggregate every file-level finding into deterministically sorted advice
/// (by kind, then file). The watcher-driven [`affected_dependents`] query is
/// intentionally *not* part of this snapshot — it needs the live changed set.
pub fn advise(
    rust_files: &[String],
    symbols: &BTreeMap<String, PerFileSymbols>,
    graph: &[DepEdge],
) -> Vec<Advice> {
    let mut out: Vec<Advice> = Vec::new();
    for (file, import) in broken_imports(rust_files, symbols) {
        out.push(Advice {
            file: file.clone(),
            kind: AdviceKind::BrokenImport,
            message: format!(
                "⚠ {file} imports `{import}`, which resolves to nothing in this workspace — did a module move or get renamed?"
            ),
        });
    }
    for cycle in find_cycles(graph) {
        let chain = cycle.join(" → ");
        out.push(Advice {
            file: cycle[0].clone(),
            kind: AdviceKind::Cycle,
            message: format!(
                "🔁 import cycle: {chain} → {} — extract the shared piece into a third module.",
                cycle[0]
            ),
        });
    }
    for (file, n) in hub_files(graph, HUB_MIN_OUT) {
        out.push(Advice {
            file: file.clone(),
            kind: AdviceKind::Hub,
            message: format!(
                "🧶 {file} depends on {n} files — consider splitting it to lower coupling."
            ),
        });
    }
    for file in orphan_files(rust_files, graph) {
        out.push(Advice {
            file: file.clone(),
            kind: AdviceKind::Orphan,
            message: format!(
                "🕸 {file} has no workspace imports in either direction — dead code, or missing from the build?"
            ),
        });
    }
    out.sort_by(|a, b| (a.kind, &a.file).cmp(&(b.kind, &b.file)));
    out
}

/// Recompute [`advise`] from the `.xencode` snapshot written by
/// [`crate::init_project`] in `root`. This is the single read path shared by
/// the TUI panel, the `xencode advise` CLI and the agentic `repo_advise`
/// tool, so the surfaces can never disagree about what the snapshot says.
/// `Err(NoIndex)` when the symbol map is missing or empty.
pub fn advise_from_snapshot(root: &Path) -> Result<Vec<Advice>, crate::ContextError> {
    use crate::index::{deps_json_path, file_index_path, read_json, symbols_json_path};
    let xencode = root.join(crate::init::XENCODE_DIR);
    let symbols: BTreeMap<String, PerFileSymbols> =
        read_json(&symbols_json_path(&xencode)).unwrap_or_default();
    if symbols.is_empty() {
        return Err(crate::ContextError::NoIndex(xencode));
    }
    let graph: Vec<DepEdge> = read_json(&deps_json_path(&xencode)).unwrap_or_default();
    let index: Option<crate::index::FilesIndex> = read_json(&file_index_path(&xencode));
    let rust_files: Vec<String> = index
        .map(|i| {
            i.files
                .into_iter()
                .filter(|f| f.language == "rust")
                .map(|f| f.path)
                .collect()
        })
        .unwrap_or_default();
    Ok(advise(&rust_files, &symbols, &graph))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::{build_graph, extract_rust_symbols};

    fn symbols_from(files: &[(&str, &str)]) -> BTreeMap<String, PerFileSymbols> {
        files
            .iter()
            .map(|(path, content)| (path.to_string(), extract_rust_symbols(content)))
            .collect()
    }

    fn rust_paths(files: &[(&str, &str)]) -> Vec<String> {
        files.iter().map(|(p, _)| p.to_string()).collect()
    }

    fn two_cycle_repo() -> (Vec<String>, BTreeMap<String, PerFileSymbols>, Vec<DepEdge>) {
        let files = [
            ("src/lib.rs", ""),
            ("src/a.rs", "use crate::b::bf;\npub fn af() {}\n"),
            ("src/b.rs", "use crate::a::af;\npub fn bf() {}\n"),
        ];
        let paths = rust_paths(&files);
        let syms = symbols_from(&files);
        let graph = build_graph(&paths, &syms);
        assert_eq!(graph.len(), 2);
        (paths, syms, graph)
    }

    #[test]
    fn detects_a_two_file_cycle_once() {
        let (_, _, graph) = two_cycle_repo();
        assert_eq!(find_cycles(&graph), vec![vec!["src/a.rs", "src/b.rs"]]);
    }

    #[test]
    fn advise_from_snapshot_reads_disk_and_errors_without_index() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "xencode-advise-snap-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // No snapshot yet → NoIndex with the /init hint, not an empty report.
        let err = advise_from_snapshot(&root).unwrap_err();
        assert!(matches!(err, crate::ContextError::NoIndex(_)), "{err:?}");
        assert!(err.to_string().contains("no project index"), "{err}");

        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "mod a;\nmod b;\n").unwrap();
        std::fs::write(root.join("src/a.rs"), "use crate::b::bee;\n").unwrap();
        std::fs::write(root.join("src/b.rs"), "use crate::a::ay;\n").unwrap();
        std::fs::File::create(root.join("Cargo.toml")).unwrap();
        crate::init_project(&root, std::sync::Arc::new(AtomicBool::new(false)), |_| {})
            .expect("init");
        let items = advise_from_snapshot(&root).unwrap();
        assert!(
            items
                .iter()
                .any(|i| i.kind == AdviceKind::Cycle && i.file == "src/a.rs"),
            "{items:?}"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn no_cycle_on_a_clean_chain() {
        let files = [
            ("src/lib.rs", ""),
            ("src/a.rs", "use crate::b::bf;\npub fn af() {}\n"),
            ("src/b.rs", "pub fn bf() {}\n"),
        ];
        let paths = rust_paths(&files);
        let graph = build_graph(&paths, &symbols_from(&files));
        assert!(find_cycles(&graph).is_empty());
    }

    #[test]
    fn flags_hubs_above_threshold_most_coupled_first() {
        let edges: Vec<DepEdge> = (0..9)
            .map(|i| DepEdge {
                from: "src/hub.rs".into(),
                to: format!("src/d{i}.rs"),
                via: format!("crate::d{i}"),
            })
            .chain([DepEdge {
                from: "src/small.rs".into(),
                to: "src/d0.rs".into(),
                via: "crate::d0".into(),
            }])
            .collect();
        assert_eq!(hub_files(&edges, 8), vec![("src/hub.rs".to_string(), 9)]);
        assert!(hub_files(&edges, 10).is_empty());
    }

    #[test]
    fn flags_orphans_but_not_entry_points() {
        let files = [
            ("src/lib.rs", ""),
            ("src/main.rs", ""),
            ("src/used.rs", "pub fn u() {}\n"),
            ("src/user.rs", "use crate::used::u;\npub fn x() {}\n"),
            ("src/dead.rs", "pub fn d() {}\n"),
        ];
        let paths = rust_paths(&files);
        let graph = build_graph(&paths, &symbols_from(&files));
        // lib.rs/main.rs have no edges either but are entry points.
        assert_eq!(orphan_files(&paths, &graph), vec!["src/dead.rs"]);
    }

    #[test]
    fn flags_broken_workspace_imports_not_external_crates() {
        let files = [
            ("src/lib.rs", ""),
            (
                "src/a.rs",
                "use crate::gone::Thing;\nuse super::also_gone::x;\nuse serde::Serialize;\npub fn af() {}\n",
            ),
        ];
        let paths = rust_paths(&files);
        let syms = symbols_from(&files);
        let broken = broken_imports(&paths, &syms);
        assert_eq!(broken.len(), 2);
        assert!(broken.contains(&("src/a.rs".to_string(), "crate::gone::Thing".to_string())));
        assert!(broken.contains(&("src/a.rs".to_string(), "super::also_gone::x".to_string())));
        assert!(!broken.iter().any(|(_, i)| i.contains("serde")));
    }

    #[test]
    fn affected_dependents_walks_reverse_edges() {
        // a imports b, b imports c: changing c affects b and (transitively) a.
        let edges = vec![
            DepEdge {
                from: "src/a.rs".into(),
                to: "src/b.rs".into(),
                via: "crate::b".into(),
            },
            DepEdge {
                from: "src/b.rs".into(),
                to: "src/c.rs".into(),
                via: "crate::c".into(),
            },
        ];
        let affected = affected_dependents(&edges, &["src/c.rs"], 3);
        assert_eq!(
            affected.get("src/c.rs"),
            Some(&vec!["src/a.rs".to_string(), "src/b.rs".to_string()])
        );
        // Hop limit respected: 1 hop from c reaches only b.
        let near = affected_dependents(&edges, &["src/c.rs"], 1);
        assert_eq!(near.get("src/c.rs"), Some(&vec!["src/b.rs".to_string()]));
        // A file nobody depends on is absent.
        assert!(affected_dependents(&edges, &["src/a.rs"], 3).is_empty());
    }

    #[test]
    fn advise_aggregates_sorted_by_kind_then_file() {
        let (paths, syms, graph) = two_cycle_repo();
        let all = advise(&paths, &syms, &graph);
        let kinds: Vec<AdviceKind> = all.iter().map(|a| a.kind).collect();
        assert!(kinds.contains(&AdviceKind::Cycle));
        let mut sorted = all.clone();
        sorted.sort_by(|a, b| (a.kind, &a.file).cmp(&(b.kind, &b.file)));
        assert_eq!(all, sorted);
        assert!(all
            .iter()
            .find(|a| a.kind == AdviceKind::Cycle)
            .unwrap()
            .message
            .contains("src/a.rs"));
    }
}
