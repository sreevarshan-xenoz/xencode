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
    let target = resolve_target(&matches, asked, &symbols)?;
    let graph: Vec<DepEdge> = read_json(&deps_json_path(&xencode)).unwrap_or_default();
    Ok(impact(&graph, &symbols, &target, symbol, IMPACT_MAX_HOPS))
}

/// Build the whole graph from the current filesystem — scan, extract, link — and
/// answer the same question, with no `.xencode` index on disk at all. This is
/// what lets `xencode impact` run headless: the watcher-driven surfaces need an
/// index a TUI `/init` wrote, but the reverse-reachability walk only ever needed
/// the edges, and the edges are pure functions of the files. The graph is
/// rebuilt per call rather than cached, because a read-only question that has to
/// build an index to answer it is building a snapshot it was told not to need.
pub fn impact_from_filesystem(
    root: &std::path::Path,
    asked: &str,
    symbol: Option<&str>,
) -> Result<ImpactReport, crate::ContextError> {
    let (graph, symbols) = headless_graph(root)?;
    let matches = matching_targets(symbols.keys().cloned(), asked);
    let target = resolve_target(&matches, asked, &symbols)?;
    Ok(impact(&graph, &symbols, &target, symbol, IMPACT_MAX_HOPS))
}

/// Scan `root`, read every first-tier Rust file, and build the dependency graph
/// in memory. Shared by [`impact_from_filesystem`] and [`change_impact`] so the
/// two can never disagree about what the edges are.
pub(crate) fn headless_graph(
    root: &std::path::Path,
) -> Result<(Vec<DepEdge>, BTreeMap<String, PerFileSymbols>), crate::ContextError> {
    use crate::scanner::{language_for_extension, scan_tree, ScanOptions};
    use crate::symbols::{build_graph, extract_rust_symbols};
    let root = root
        .canonicalize()
        .map_err(|source| crate::ContextError::Io {
            path: root.to_path_buf(),
            source,
        })?;
    // The same git-filtered scan `/init` runs, so a headless answer walks the
    // tracked tree, not the build directory or anything ignored. `git_file_set`
    // is `None` outside a repository, which makes the scan exclusion-based.
    let git_filter = crate::gitinfo::git_file_set(&root);
    let scan = scan_tree(
        &root,
        &ScanOptions {
            git_filter,
            cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        },
    )?;
    let rust_files: Vec<String> = scan
        .files
        .iter()
        .filter(|e| {
            language_for_extension(&e.ext).has_semantic_tier() && !e.is_secret && !e.is_binary
        })
        .map(|e| e.path.clone())
        .collect();
    let mut symbols = BTreeMap::new();
    for path in &rust_files {
        let full = root.join(path);
        let content = std::fs::read_to_string(&full)
            .map_err(|source| crate::ContextError::Io { path: full, source })?;
        symbols.insert(path.clone(), extract_rust_symbols(&content));
    }
    let graph = build_graph(&rust_files, &symbols);
    Ok((graph, symbols))
}

/// Turn the tail-match set into one stored path, or the error that explains why
/// it is not one. Lifted out of [`impact_from_snapshot`] so the headless and
/// snapshot paths refuse a missing or ambiguous file identically.
fn resolve_target(
    matches: &[String],
    asked: &str,
    symbols: &BTreeMap<String, PerFileSymbols>,
) -> Result<String, crate::ContextError> {
    match matches {
        [] => Err(crate::ContextError::NotIndexed {
            asked: asked.to_string(),
            near: near_names(symbols, asked),
            indexed: symbols.len(),
        }),
        [only] => Ok(only.clone()),
        many => Err(crate::ContextError::AmbiguousTarget {
            asked: asked.to_string(),
            matches: many.to_vec(),
        }),
    }
}

/// Everything `xencode impact` reports for one file (`QD-1`): the crate it lives
/// in and what depends on that crate (exact, from `cargo metadata`), the files
/// that link it (a hop-capped prediction over the symbol graph), and the files
/// its history is coupled to (churn). Three layers, three different strengths of
/// evidence, kept apart so a reader never mistakes a predicted edge for a proven
/// one or a crate-level fact for a line-level one.
pub struct ChangeImpact {
    /// The file as the graph stores it, widened from what was asked.
    pub target: String,
    /// The workspace member the file belongs to, when it belongs to one.
    pub crate_name: Option<String>,
    /// Members that depend on that crate, transitively, each with its hop.
    pub reverse_crates: Vec<(String, usize)>,
    /// Members that link it directly, with the dependency kind of that edge.
    pub direct_crates: Vec<(String, String)>,
    /// The file-level prediction, already hop-capped.
    pub files: ImpactReport,
    /// Files most changed alongside this one, strongest coupling first.
    pub cochange: Vec<(String, u32)>,
    /// Commits this file appeared in — its own churn.
    pub own_commits: u32,
    /// `None` when there is no git history here at all, which is not the same as
    /// a file nobody edits together with: an empty coupling list with history
    /// present means "not co-changed", `None` means "no way to know".
    pub history_known: bool,
    /// The workspace member each consumer file lives in, keyed by the file's
    /// path as `files` reports it. Populated once from `cargo metadata` alongside
    /// the crate layer above, so a projection of this report — the fan-out panel
    /// (`QD-2`) and any later surface — can group by crate without a second
    /// subprocess. A file the manifest does not claim (a build artifact, a stray
    /// `.rs` outside any member) is simply absent from this map.
    pub file_crates: BTreeMap<String, String>,
    /// The CLI commands and manual documentation impact (AE-4), when the target
    /// defines CLI commands.
    pub cli_impact: Option<crate::cli_impact::CliImpact>,
}

/// The three-layer answer for one file. `file` may be a repo-relative or absolute
/// path, or a tail; the same resolution rules the file-graph uses. `cargo
/// metadata` and `git log` run for real; if either is unavailable the layer that
/// needed it reports honestly absent rather than fabricating a closure.
pub fn change_impact(
    root: &std::path::Path,
    file: &str,
) -> Result<ChangeImpact, crate::ContextError> {
    let root = root
        .canonicalize()
        .map_err(|source| crate::ContextError::Io {
            path: root.to_path_buf(),
            source,
        })?;
    let (graph, symbols) = headless_graph(&root)?;
    let matches = matching_targets(symbols.keys().cloned(), file);
    let target = resolve_target(&matches, file, &symbols)?;

    // The crate layer: exact, offline, from the manifests themselves.
    let metadata = crate::crate_graph::cargo_metadata(&root).unwrap_or_default();
    let crate_graph = crate::crate_graph::parse_crate_graph(&metadata).unwrap_or_default();
    // Place the file by its real path on disk, so the answer matches the graph node.
    let crate_name = crate::crate_graph::crate_of_file(&crate_graph, &root.join(&target));
    let (reverse_crates, direct_crates) = match &crate_name {
        Some(name) => (
            crate_graph.reverse_closure(name),
            crate_graph
                .direct_dependents(name)
                .into_iter()
                .map(|(d, k)| (d, k.label().to_string()))
                .collect(),
        ),
        None => (Vec::new(), Vec::new()),
    };

    // The churn layer: one git log over history, coupling read straight off it.
    // `git log --name-only` keys every file from the top of the repository, but
    // the graph and `cargo metadata` above speak from the workspace, which here
    // lives one directory below that top. Without this step the lookup misses
    // every file in a nested workspace and reports "no history" for a repo that
    // plainly has it, so put the target on git's base and bring the answer back.
    let prefix = git_prefix(&root);
    let history = crate::cochange::mine_commit_history(&root);
    let (cochange, own_commits, history_known) = match &history {
        Some(h) => {
            let entry = h.get(&format!("{prefix}{target}"));
            (
                entry
                    .map(|e| {
                        e.partners
                            .iter()
                            .map(|(p, n)| (to_workspace(p, &prefix), *n))
                            .collect()
                    })
                    .unwrap_or_default(),
                entry.map(|e| e.commits).unwrap_or(0),
                true,
            )
        }
        None => (Vec::new(), 0, false),
    };

    // One call to the file layer, reused for both the report itself and the
    // crate map every consumer file belongs to. Calling `impact` twice would
    // walk the same BFS twice for no reason.
    let report = impact(&graph, &symbols, &target, None, IMPACT_MAX_HOPS);
    let mut file_crates = BTreeMap::new();
    for f in &report.files {
        if let Some(name) = crate::crate_graph::crate_of_file(&crate_graph, &root.join(&f.file)) {
            file_crates.insert(f.file.clone(), name);
        }
    }

    let repo_root = crate::cli_impact::find_repo_root(&root);
    let cli_impact = crate::cli_impact::detect_cli_impact(&repo_root, &root.join(&target), None);

    Ok(ChangeImpact {
        files: report,
        target,
        crate_name,
        reverse_crates,
        direct_crates,
        cochange,
        own_commits,
        history_known,
        file_crates,
        cli_impact,
    })
}

/// The removal counterfactual (`QD-5`): the same graph with one node deleted, and
/// what that costs. [`change_impact`] answers "who must be re-checked if I edit
/// this file"; this answers the stronger question "what breaks if I take it out,"
/// where editing and deleting are genuinely different — a change may be absorbed
/// by a consumer, a deletion cannot. Two things fall out of removing the node, and
/// they are the two directions of an edge:
///
/// - **broken links**: the files that write a `use`, `mod` or `impl` resolving to
///   the removed file. Their name no longer points anywhere.
/// - **newly dead files**: files reachable from a crate root today that stop being
///   reachable once the node and its edges are deleted — the modules that only this
///   file pulled into the build, at any depth. With the one path that declared them
///   gone, nothing compiles them any more.
///
/// Both are exact against the symbol graph the file layer already builds; neither
/// reads a file body, so neither can claim more than a resolved name. Entry points
/// (`lib.rs` / `main.rs` / `mod.rs`) are never reported dead: they are roots, not
/// leaves, and an unreferenced root is a crate, not orphaned code.
pub struct RemovalImpact {
    /// The file as the graph stores it, widened from what was asked.
    pub target: String,
    /// Files whose `use`/`mod`/`impl` resolved to the target, now left dangling.
    pub breaks: Vec<String>,
    /// Files that stop being reached at all once the target is removed.
    pub orphans: Vec<String>,
    /// The graph's own size, so an empty pair of lists reads as a fact about this
    /// graph rather than a promise about the code.
    pub indexed_files: usize,
    pub edges: usize,
}

/// Compute the removal counterfactual over a graph in memory. Split from
/// [`removal_impact`] so the graph arithmetic is testable without a filesystem.
pub fn removal_from_graph(graph: &[DepEdge], files: &[String], target: &str) -> RemovalImpact {
    let is_entry = |f: &str| matches!(f.rsplit('/').next(), Some("lib.rs" | "main.rs" | "mod.rs"));

    // Who points at the target: those links dangle when it goes.
    let mut breaks: Vec<String> = graph
        .iter()
        .filter(|e| e.to == target)
        .map(|e| e.from.clone())
        .collect();
    breaks.sort();
    breaks.dedup();

    // Newly dead: files reachable from an entry point today that are not reachable
    // once the target and its edges are gone. The delta of the two reachability
    // sets, minus entry points (a root is not dead code) and the target itself.
    let before = reachable_from_entries(graph, None);
    let after = reachable_from_entries(graph, Some(target));
    let orphans: Vec<String> = before
        .difference(&after)
        .filter(|f| f.as_str() != target && !is_entry(f))
        .cloned()
        .collect();

    RemovalImpact {
        target: target.to_string(),
        breaks,
        orphans,
        indexed_files: files.len(),
        edges: graph.len(),
    }
}

/// Files reachable from any entry point (`lib.rs` / `main.rs` / `mod.rs`),
/// following edges outward. When `skip` is `Some(name)`, that node and every edge
/// touching it are deleted first — the graph as it stands with one file removed.
fn reachable_from_entries(graph: &[DepEdge], skip: Option<&str>) -> BTreeSet<String> {
    let is_entry = |f: &str| matches!(f.rsplit('/').next(), Some("lib.rs" | "main.rs" | "mod.rs"));
    let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for e in graph {
        if Some(e.from.as_str()) == skip || Some(e.to.as_str()) == skip {
            continue;
        }
        adj.entry(e.from.as_str()).or_default().push(e.to.as_str());
    }
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut stack: Vec<&str> = Vec::new();
    for e in graph {
        if is_entry(&e.from) && Some(e.from.as_str()) != skip && seen.insert(e.from.clone()) {
            stack.push(e.from.as_str());
        }
    }
    while let Some(node) = stack.pop() {
        for next in adj.get(node).into_iter().flatten() {
            if seen.insert((*next).to_string()) {
                stack.push(next);
            }
        }
    }
    seen
}

/// The removal counterfactual for one file, run headless off the git-tracked
/// tree. `file` follows the same resolution rules as [`change_impact`].
pub fn removal_impact(
    root: &std::path::Path,
    file: &str,
) -> Result<RemovalImpact, crate::ContextError> {
    let (graph, symbols) = headless_graph(root)?;
    let matches = matching_targets(symbols.keys().cloned(), file);
    let target = resolve_target(&matches, file, &symbols)?;
    let files: Vec<String> = symbols.keys().cloned().collect();
    Ok(removal_from_graph(&graph, &files, &target))
}

/// The workspace's path relative to the git top level, with a trailing slash,
/// or empty when the workspace *is* the top level (or git cannot say). This is
/// the difference between `cargo metadata` and `git log --name-only`: the first
/// speaks from `--manifest-dir`, the second always from the repository's top.
fn git_prefix(root: &std::path::Path) -> String {
    let toplevel = crate::gitinfo::git_stdout(root, &["rev-parse", "--show-toplevel"])
        .ok()
        .and_then(|s| std::fs::canonicalize(s.trim()).ok());
    match toplevel.and_then(|tl| root.strip_prefix(tl).ok().map(|r| r.to_path_buf())) {
        Some(rel) if !rel.as_os_str().is_empty() => {
            format!("{}/", rel.to_string_lossy().replace('\\', "/"))
        }
        _ => String::new(),
    }
}

/// Bring one git-history path back onto the workspace base the rest of the
/// report uses, so a co-change partner reads the same way as the file layer.
fn to_workspace(path: &str, prefix: &str) -> String {
    match path.strip_prefix(prefix) {
        Some(rest) => rest.to_string(),
        None => path.to_string(),
    }
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

    /// The headless-index trap, fixed: the same file that the snapshot path
    /// refuses (no `.xencode` on disk) is answered by building the graph from the
    /// files themselves. This is the reason `xencode impact` runs with no index
    /// and no TUI `/init`.
    #[test]
    fn the_headless_answer_works_where_the_snapshot_would_refuse() {
        let dir = temp_dir();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/lib.rs"), "mod leaf;\nmod mid;\n").unwrap();
        std::fs::write(dir.join("src/leaf.rs"), "pub fn build() {}\n").unwrap();
        std::fs::write(
            dir.join("src/mid.rs"),
            "use crate::leaf::build;\npub fn m() { let _ = build(); }\n",
        )
        .unwrap();
        // No `.xencode` exists, so the snapshot read has nothing to work from…
        assert!(
            impact_from_snapshot(&dir, "src/leaf.rs", None).is_err(),
            "with no index the snapshot path refuses"
        );
        // …while the headless walk reads the files and answers anyway.
        let report = impact_from_filesystem(&dir, "src/leaf.rs", None).expect("headless runs");
        let got: Vec<&str> = report.files.iter().map(|f| f.file.as_str()).collect();
        assert!(got.contains(&"src/mid.rs"), "mid links leaf: {got:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Write a two-member cargo workspace into `repo`/`ws_rel` (so `ws_rel` may put
    /// the workspace in a subdirectory of the git repository, the way this very repo
    /// does), `beta` depending on `alpha` both in its manifest and in a real `use`,
    /// then commit it in `repo` so the churn layer has genuine git history to read.
    fn write_workspace_in(repo: &std::path::Path, ws_rel: &str) {
        let dir = repo.join(ws_rel);
        let write = |rel: &str, body: &str| {
            let full = dir.join(rel);
            std::fs::create_dir_all(full.parent().unwrap()).unwrap();
            std::fs::write(full, body).unwrap();
        };
        write(
            "Cargo.toml",
            "[workspace]\nresolver = \"2\"\nmembers = [\"crates/alpha\", \"crates/beta\"]\n",
        );
        write(
            "crates/alpha/Cargo.toml",
            "[package]\nname = \"alpha\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        );
        write(
            "crates/alpha/src/lib.rs",
            "mod inner;\npub fn thing() -> u32 { 1 }\n",
        );
        write(
            "crates/alpha/src/inner.rs",
            "use crate::thing;\npub fn call() -> u32 { thing() }\n",
        );
        write(
            "crates/beta/Cargo.toml",
            "[package]\nname = \"beta\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nalpha = { path = \"../alpha\" }\n",
        );
        write(
            "crates/beta/src/lib.rs",
            "use alpha::thing;\npub fn use_it() -> u32 { thing() }\n",
        );
        // A real commit, so `git log` returns a real history rather than nothing.
        let git = |args: &[&str]| {
            let ok = std::process::Command::new("git")
                .args(args)
                .current_dir(repo)
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            assert!(ok, "git {args:?} failed in the temp repository");
        };
        git(&["init", "-q"]);
        git(&["add", "-A"]);
        git(&[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "-m",
            "seed",
        ]);
    }

    /// The flat case: the workspace is the repository root.
    fn write_workspace(dir: &std::path::Path) {
        write_workspace_in(dir, "");
    }

    /// `QD-1` end to end against a real workspace: the file resolves to its crate,
    /// and the two evidence layers each prove the thing only it can. The **crate**
    /// layer (exact, from `cargo metadata`) is the only one that sees the cross-crate
    /// dependency `beta → alpha`; the **file** layer (a hop-capped prediction over the
    /// symbol graph) is the one that sees the intra-crate link between `inner.rs` and
    /// the `lib.rs` that declares it. They are deliberately different sets — reading
    /// them as one would mistake a manifest edge for a source edge.
    #[test]
    fn change_impact_keeps_the_crate_layer_and_the_file_layer_apart() {
        let dir = temp_dir();
        write_workspace(&dir);
        let impact = change_impact(&dir, "crates/alpha/src/inner.rs").expect("runs headless");
        assert_eq!(impact.crate_name.as_deref(), Some("alpha"));
        // The crate layer: cross-crate, exact, from the manifests.
        assert!(
            impact.reverse_crates.iter().any(|(n, _)| n == "beta"),
            "cargo metadata closure should hold beta: {:?}",
            impact.reverse_crates
        );
        let direct: Vec<&str> = impact
            .direct_crates
            .iter()
            .map(|(d, _)| d.as_str())
            .collect();
        assert_eq!(direct, vec!["beta"], "and names the beta edge");
        // The file layer: intra-crate, a prediction over resolved names. `lib.rs`
        // declares `mod inner`, so it is the file that links inner.rs.
        let files: Vec<&str> = impact.files.files.iter().map(|f| f.file.as_str()).collect();
        assert!(
            files.iter().any(|f| f.ends_with("alpha/src/lib.rs")),
            "lib declares mod inner, so the graph lists it: {files:?}"
        );
        // And the file layer cannot see beta, which never names inner at the
        // source level — that relationship is the crate layer's alone.
        assert!(
            !files.iter().any(|f| f.ends_with("beta/src/lib.rs")),
            "a cross-crate consumer is the crate layer's, not a graph edge: {files:?}"
        );
        // The churn layer really ran: it read the commit we made.
        assert!(impact.history_known, "git history is present and read");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// `QD-1` regression: a workspace nested below the git top level — the layout
    /// this very repository uses. `git log --name-only` keys files from the
    /// repository root (`rust/crates/alpha/src/inner.rs`) while the graph and
    /// `cargo metadata` name them from the workspace (`crates/alpha/src/inner.rs`).
    /// Without reconciling those bases the churn lookup misses every file and the
    /// report says "no commits" for history that plainly exists, so here the same
    /// commit that touched `inner.rs` must still show up as its own churn.
    #[test]
    fn a_nested_workspace_reads_churn_from_the_repository_top_level() {
        let dir = temp_dir();
        write_workspace_in(&dir, "rust");
        let impact =
            change_impact(&dir.join("rust"), "crates/alpha/src/inner.rs").expect("runs headless");
        assert_eq!(impact.target, "crates/alpha/src/inner.rs");
        assert!(impact.history_known, "git history is present and read");
        assert_eq!(
            impact.own_commits, 1,
            "the seed commit touched inner.rs, counted from the repo root's keys"
        );
        // And the partner is reported on the workspace base, not git's top level,
        // so it reads the same way as the file layer above it.
        assert!(
            impact
                .cochange
                .iter()
                .any(|(p, _)| p == "crates/alpha/src/lib.rs"),
            "inner co-changed with its lib, on the workspace base: {:?}",
            impact.cochange
        );
        assert!(
            !impact.cochange.iter().any(|(p, _)| p.starts_with("rust/")),
            "partners are never left on the repo-root base: {:?}",
            impact.cochange
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn edge(from: &str, to: &str) -> DepEdge {
        DepEdge {
            from: from.to_string(),
            to: to.to_string(),
            via: format!("crate::{to}"),
        }
    }

    /// `QD-5`: deleting a node cuts its incoming links (who broke) and can strand
    /// the modules only it pulled in (what becomes dead). Here `a.rs` is the sole
    /// thing that declares `helper.rs`, and is one of two that reach `shared.rs`;
    /// removing it must break `lib.rs` (which `mod`-ed `a`), orphan `helper.rs`
    /// (nothing else declares it), and leave `shared.rs` alive (lib still reaches it).
    #[test]
    fn removing_a_file_breaks_its_importers_and_strands_only_its_own_children() {
        let graph = vec![
            edge("src/lib.rs", "src/a.rs"),
            edge("src/a.rs", "src/helper.rs"),
            edge("src/a.rs", "src/shared.rs"),
            edge("src/lib.rs", "src/shared.rs"),
        ];
        let files = ["src/lib.rs", "src/a.rs", "src/helper.rs", "src/shared.rs"]
            .map(str::to_string)
            .to_vec();
        let out = removal_from_graph(&graph, &files, "src/a.rs");
        assert_eq!(out.breaks, vec!["src/lib.rs".to_string()], "lib mod-ed a");
        assert_eq!(
            out.orphans,
            vec!["src/helper.rs".to_string()],
            "only a declared helper; shared stays live through lib"
        );
    }

    /// A module that is itself an entry point (`mod.rs`) is never reported as dead
    /// code: it is a root, not a leaf, and calling an unreferenced root "orphaned"
    /// would be a category error. Here `a.rs` pulls in two children — one a `mod.rs`,
    /// one a plain file — and only the plain file can become dead when `a` goes.
    #[test]
    fn an_entry_point_is_never_reported_as_orphaned_by_removal() {
        let graph = vec![
            edge("src/lib.rs", "src/a.rs"),
            edge("src/a.rs", "src/inner/mod.rs"),
            edge("src/a.rs", "src/orphan_me.rs"),
        ];
        let files = [
            "src/lib.rs",
            "src/a.rs",
            "src/inner/mod.rs",
            "src/orphan_me.rs",
        ]
        .map(str::to_string)
        .to_vec();
        let out = removal_from_graph(&graph, &files, "src/a.rs");
        assert_eq!(out.breaks, vec!["src/lib.rs".to_string()], "lib mod-ed a");
        assert_eq!(
            out.orphans,
            vec!["src/orphan_me.rs".to_string()],
            "a plain child strands; the mod.rs under it is a root and stays"
        );
        assert!(
            !out.orphans.iter().any(|f| f.ends_with("mod.rs")),
            "a mod.rs is never dead code: {:?}",
            out.orphans
        );
    }

    /// Removing a leaf — a file nothing else declares — strands nothing: only the
    /// files that import it break. An analysis that reported the whole crate as
    /// dead from deleting one leaf would be useless, so this pins the empty case.
    #[test]
    fn removing_a_leaf_orphans_nothing() {
        let graph = vec![edge("src/lib.rs", "src/leaf.rs")];
        let files = vec!["src/lib.rs".to_string(), "src/leaf.rs".to_string()];
        let out = removal_from_graph(&graph, &files, "src/leaf.rs");
        assert_eq!(out.breaks, vec!["src/lib.rs".to_string()]);
        assert!(
            out.orphans.is_empty(),
            "leaf declares no children: {:?}",
            out.orphans
        );
    }

    /// `QD-5` against the real nested workspace on disk: the headless graph finds
    /// `inner.rs`'s importer (`alpha/src/lib.rs`, which writes `mod inner;`), so
    /// removing inner must name that break — proving the filesystem path resolves
    /// the target and runs the same arithmetic the unit tests pin.
    #[test]
    fn removal_runs_headless_against_a_real_workspace() {
        let dir = temp_dir();
        write_workspace_in(&dir, "rust");
        let out =
            removal_impact(&dir.join("rust"), "crates/alpha/src/inner.rs").expect("runs headless");
        assert_eq!(out.target, "crates/alpha/src/inner.rs");
        assert!(
            out.breaks.iter().any(|f| f.ends_with("alpha/src/lib.rs")),
            "lib declares mod inner, so deleting inner breaks lib: {:?}",
            out.breaks
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
