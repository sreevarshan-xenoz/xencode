//! Repository intelligence (M1): symbol extraction + dependency graph.
//!
//! Pass 2 of the context engine, still fully deterministic and zero-LLM.
//! This is the seed of the Xencode "repository map" (Aider-style):
//!
//!   - regex extraction of `structs` / `functions` / `imports` / `exports`
//!     from Rust source files (`symbols.json`)
//!   - module resolution (`crate::`, `super::`, `self::`, plain-relative)
//!     against the known Rust file set, producing a file→file
//!     dependency graph (`deps.json`)
//!   - deterministic centrality ranking + dependency expansion, which M2's
//!     retrieval/budgeter consumes as its structural signal
//!
//! Resolution is intentionally cheap and best-effort: `use` statements that
//! point at external crates (not resolvable to a file in this workspace) are
//! simply not turned into edges. Tree-sitter may replace the extraction layer
//! later; the on-disk schemas are stable.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Symbol inventory for one source file (`symbols.json` value).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PerFileSymbols {
    #[serde(default)]
    pub structs: Vec<String>,
    #[serde(default)]
    pub functions: Vec<String>,
    /// Raw `use` statements (trimmed, `use`/`;` removed).
    #[serde(default)]
    pub imports: Vec<String>,
    /// Names re-exported via `pub use`.
    #[serde(default)]
    pub exports: Vec<String>,
}

/// A resolved file→file dependency edge (`deps.json` entry).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DepEdge {
    pub from: String,
    pub to: String,
    /// Canonical module path through which the dependency was resolved
    /// (e.g. `crate::db::conn`, `super::models`, `util`).
    pub via: String,
}

struct RegexCache {
    structs: Regex,
    functions: Regex,
    imports: Regex,
    exports: Regex,
}

fn regexes() -> &'static RegexCache {
    static CACHE: std::sync::OnceLock<RegexCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| RegexCache {
        structs: Regex::new(
            r"(?m)^\s*pub(?:\([^)]*\))?\s+struct\s+([A-Za-z_][A-Za-z0-9_]*)",
        )
        .unwrap(),
        functions: Regex::new(
            r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)",
        )
        .unwrap(),
        imports: Regex::new(r"(?m)^\s*(?:pub\s+)?use\s+([^;]+);").unwrap(),
        exports: Regex::new(r"(?m)^\s*pub\s+use\s+([A-Za-z_][A-Za-z0-9_]*)(?:::|;)").unwrap(),
    })
}

/// Extract a symbol inventory from a Rust file's text.
/// All lists are sorted + deduped so output is deterministic.
pub fn extract_rust_symbols(content: &str) -> PerFileSymbols {
    let c = regexes();

    let mut structs: Vec<String> = c
        .structs
        .captures_iter(content)
        .map(|m| m[1].to_string())
        .collect();
    let mut functions: Vec<String> = c
        .functions
        .captures_iter(content)
        .map(|m| m[1].to_string())
        .collect();
    let mut imports: Vec<String> = c
        .imports
        .captures_iter(content)
        .map(|m| m[1].trim().to_string())
        .collect();
    let mut exports: Vec<String> = c
        .exports
        .captures_iter(content)
        .map(|m| m[1].to_string())
        .collect();

    for list in [&mut structs, &mut functions, &mut imports, &mut exports] {
        list.sort();
        list.dedup();
    }

    PerFileSymbols {
        structs,
        functions,
        imports,
        exports,
    }
}

/// Which namespace an import resolves relative to.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Qualifier {
    /// `crate::` — anchored at the crate root (dir of `lib.rs`/`main.rs`).
    Crate,
    /// `self::` — anchored at the current module's dir.
    SelfMod,
    /// `super::` (possibly repeated) — one or more dirs up from the module.
    Super { up: usize },
    /// No qualifier — same-dir relative (ambiguous with external crates, so
    /// only resolves when a matching file actually exists).
    Plain,
}

/// Parse a raw `use`-statement text (already stripped of `use` / `;`)
/// into a qualifier + remaining path segments.
fn parse_import(raw: &str) -> Option<(Qualifier, Vec<String>)> {
    let mut s = raw.trim();
    if s.is_empty() || s.contains('{') || s.contains('}') || s.contains('*') {
        return None;
    }
    if let Some(ix) = s.find(" as ") {
        s = s[..ix].trim();
    }
    s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut segs: Vec<String> = s
        .split("::")
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    if segs.is_empty() {
        return None;
    }
    match segs[0].as_str() {
        "crate" => {
            if segs.len() < 2 {
                return None;
            }
            Some((Qualifier::Crate, segs.split_off(1)))
        }
        "self" => {
            if segs.len() < 2 {
                return None;
            }
            Some((Qualifier::SelfMod, segs.split_off(1)))
        }
        "super" => {
            let mut up = 0usize;
            while segs.first().map(|s| s.as_str()) == Some("super") {
                segs.remove(0);
                up += 1;
            }
            if segs.is_empty() {
                return None;
            }
            Some((Qualifier::Super { up }, segs))
        }
        _ => Some((Qualifier::Plain, segs)),
    }
}

/// Repo-relative dir of a file (`src/app/foo.rs` → `src/app`, `foo.rs` → ``).
fn dir_of(file: &str) -> String {
    match file.rsplit_once('/') {
        Some((dir, _)) => dir.to_string(),
        None => String::new(),
    }
}

/// Parent of a repo-relative dir (`src/app` → `src`, `` → ``).
fn parent_of(dir: &str) -> String {
    match dir.rsplit_once('/') {
        Some((parent, _)) => parent.to_string(),
        None => String::new(),
    }
}

/// Join a base dir and a module path with `/`, handling empty base.
fn join_rel(base: &str, mod_path: &str, is_mod: bool) -> String {
    let rel = if base.is_empty() {
        mod_path.to_string()
    } else {
        format!("{base}/{mod_path}")
    };
    if is_mod {
        format!("{rel}/mod.rs")
    } else {
        format!("{rel}.rs")
    }
}

/// The anchored directory an import resolves from.
fn module_base(file: &str, qual: &Qualifier) -> String {
    let is_mod_rs = file.rsplit('/').next() == Some("mod.rs");
    match qual {
        // `crate::` base is supplied separately (root of the enclosing crate).
        Qualifier::Crate => String::new(),
        Qualifier::SelfMod | Qualifier::Plain => dir_of(file),
        Qualifier::Super { up } => {
            let mut dir = if is_mod_rs {
                parent_of(&dir_of(file))
            } else {
                dir_of(file)
            };
            for _ in 1..*up {
                dir = parent_of(&dir);
            }
            dir
        }
    }
}

/// Resolve an import inside `file` to a concrete repo-relative file path.
/// Returns `(path, via)` where `via` is the canonical module path resolved.
pub fn resolve_import(
    file: &str,
    import: &str,
    crate_root: Option<&str>,
    files: &HashSet<&str>,
) -> Option<(String, String)> {
    let (qual, segs) = parse_import(import)?;
    let base = match &qual {
        Qualifier::Crate => crate_root?.to_string(),
        other => module_base(file, other),
    };
    if base.is_empty() && crate_root.is_none() {
        return None;
    }
    for i in (1..=segs.len()).rev() {
        let mod_path = segs[..i].join("/");
        if mod_path.is_empty() {
            continue;
        }
        for candidate in [
            join_rel(&base, &mod_path, false),
            join_rel(&base, &mod_path, true),
        ] {
            if files.contains(candidate.as_str()) {
                let via = match &qual {
                    Qualifier::Crate => format!("crate::{}", segs[..i].join("::")),
                    Qualifier::Super { up } => {
                        let mut v = String::new();
                        for _ in 0..*up {
                            v.push_str("super::");
                        }
                        v.push_str(&segs[..i].join("::"));
                        v
                    }
                    _ => segs[..i].join("::"),
                };
                return Some((candidate, via));
            }
        }
    }
    None
}

/// Find the crate-root dirs (`dir of lib.rs`/`main.rs`) among the rust files.
fn crate_roots(rust_files: &[String]) -> Vec<String> {
    let mut roots: BTreeSet<String> = BTreeSet::new();
    for f in rust_files {
        let name = f.rsplit('/').next().unwrap_or(f);
        if name == "lib.rs" || name == "main.rs" {
            roots.insert(dir_of(f));
        }
    }
    roots.into_iter().collect()
}

/// Longest crate root that contains `file`; empty-string root is the fallback.
fn crate_root_for<'a>(file: &str, roots: &'a [String]) -> Option<&'a str> {
    let mut candidates: Vec<&String> = roots
        .iter()
        .filter(|r| r.is_empty() || file.starts_with(&format!("{r}/")))
        .collect();
    candidates.sort_by_key(|r| r.len());
    candidates.last().map(|r| r.as_str())
}

/// Build the file→file dependency graph from extracted symbols.
/// Edges are sorted + deduped for deterministic output.
pub fn build_graph(
    rust_files: &[String],
    symbols: &BTreeMap<String, PerFileSymbols>,
) -> Vec<DepEdge> {
    let file_set: HashSet<&str> = rust_files.iter().map(|s| s.as_str()).collect();
    let roots = crate_roots(rust_files);

    let mut edges: Vec<DepEdge> = Vec::new();
    for file in rust_files {
        let Some(sym) = symbols.get(file) else {
            continue;
        };
        let root = crate_root_for(file, &roots);
        for import in &sym.imports {
            if let Some((to, via)) = resolve_import(file, import, root, &file_set) {
                edges.push(DepEdge {
                    from: file.clone(),
                    to,
                    via,
                });
            }
        }
    }

    edges.sort_by(|a, b| (&a.from, &a.to, &a.via).cmp(&(&b.from, &b.to, &b.via)));
    edges.dedup();
    edges
}

/// Forward map: file → files it depends on (self-imports skipped against
/// itself; a file depending on itself yields no self-loop by construction
/// since `use` in a file cannot refer to that same file).
pub fn dependency_map(graph: &[DepEdge]) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for e in graph {
        map.entry(e.from.clone()).or_default().push(e.to.clone());
    }
    for v in map.values_mut() {
        v.sort();
        v.dedup();
    }
    map
}

/// Reverse map: file → files that depend on it.
pub fn dependent_map(graph: &[DepEdge]) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for e in graph {
        map.entry(e.to.clone()).or_default().push(e.from.clone());
    }
    for v in map.values_mut() {
        v.sort();
        v.dedup();
    }
    map
}

/// Deterministic centrality ranking: `2·in-degree + out-degree`, sorted by
/// score descending then path ascending. Guard/entry files with no edges get
/// score `0` but stay present, so retrieval can still discover them lexically.
pub fn rank_files(graph: &[DepEdge], files: &[String]) -> Vec<(String, u64)> {
    let mut in_cnt: BTreeMap<&str, u64> = BTreeMap::new();
    let mut out_cnt: BTreeMap<&str, u64> = BTreeMap::new();
    for e in graph {
        *in_cnt.entry(e.to.as_str()).or_insert(0) += 1;
        *out_cnt.entry(e.from.as_str()).or_insert(0) += 1;
    }
    let mut ranked: Vec<(String, u64)> = files
        .iter()
        .map(|f| {
            let score =
                in_cnt.get(f.as_str()).unwrap_or(&0) * 2 + out_cnt.get(f.as_str()).unwrap_or(&0);
            (f.clone(), score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
}

/// BFS over forward dependency edges from `seeds`, up to `max_hops` deep.
/// Returns every reachable file (excluding the seeds), sorted.
pub fn expand_dependencies(graph: &[DepEdge], seeds: &[&str], max_hops: usize) -> BTreeSet<String> {
    let fwd = dependency_map(graph);
    let mut visited: BTreeSet<String> = BTreeSet::new();
    let mut current: Vec<String> = seeds.iter().map(|s| s.to_string()).collect();
    for _ in 0..max_hops {
        if current.is_empty() {
            break;
        }
        let mut next: Vec<String> = Vec::new();
        for file in &current {
            if let Some(deps) = fwd.get(file) {
                for dep in deps {
                    if !visited.contains(dep) && dep != file {
                        visited.insert(dep.clone());
                        next.push(dep.clone());
                    }
                }
            }
        }
        current = next;
    }
    visited
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symbols_from(files: &[(&str, &str)]) -> BTreeMap<String, PerFileSymbols> {
        files
            .iter()
            .map(|(path, content)| (path.to_string(), extract_rust_symbols(content)))
            .collect()
    }

    fn rust_paths(files: &[(&str, &str)]) -> Vec<String> {
        files.iter().map(|(p, _)| p.to_string()).collect()
    }

    #[test]
    fn extracts_structs_functions_imports_exports() {
        let content = r#"
pub struct User {
    id: u64,
}

#[derive(Debug)]
pub struct Session;

pub fn connect() {}

pub async fn refresh() {}

use std::collections::HashMap;

pub use database::Pool;

fn helper() {}
"#;
        let sym = extract_rust_symbols(content);
        assert_eq!(sym.structs, vec!["Session".to_string(), "User".to_string()]);
        assert_eq!(
            sym.functions,
            vec![
                "connect".to_string(),
                "helper".to_string(),
                "refresh".to_string()
            ]
        );
        assert_eq!(
            sym.imports,
            vec![
                "database::Pool".to_string(),
                "std::collections::HashMap".to_string()
            ]
        );
        assert_eq!(sym.exports, vec!["database".to_string()]);
    }

    #[test]
    fn builds_crate_resolved_dependency_graph() {
        let files = [
            ("lib.rs", "pub mod db;\npub mod api;\n"),
            ("db/mod.rs", "pub mod conn;\n"),
            (
                "db/conn.rs",
                "use crate::api::models::User;\npub fn connect() {}\n",
            ),
            ("api/mod.rs", "pub mod models;\n"),
            (
                "api/models.rs",
                "use crate::db::conn::connect;\npub struct User {}\n",
            ),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);

        let graph = build_graph(&rust_files, &symbols);
        let edges: Vec<(String, String, String)> = graph
            .iter()
            .map(|e| (e.from.clone(), e.to.clone(), e.via.clone()))
            .collect();

        assert_eq!(
            edges,
            vec![
                (
                    "api/models.rs".to_string(),
                    "db/conn.rs".to_string(),
                    "crate::db::conn".to_string()
                ),
                (
                    "db/conn.rs".to_string(),
                    "api/models.rs".to_string(),
                    "crate::api::models".to_string()
                ),
            ]
        );
    }

    #[test]
    fn resolves_super_self_and_plain_relative() {
        let files = [
            ("src/lib.rs", "pub mod app;\npub mod srv;\n"),
            ("src/app/mod.rs", "pub mod foo;\npub mod bar;\n"),
            (
                "src/app/foo.rs",
                "use super::bar::Baz;\nuse crate::srv::serve;\npub struct Foo {}\n",
            ),
            ("src/app/bar.rs", "pub struct Bar;\n"),
            ("src/srv/mod.rs", "pub fn serve() {}\n"),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);

        let graph = build_graph(&rust_files, &symbols);
        let edges_a: Vec<(String, String)> = graph
            .iter()
            .filter(|e| e.from == "src/app/foo.rs")
            .map(|e| (e.to.clone(), e.via.clone()))
            .collect();

        assert_eq!(
            edges_a,
            vec![
                ("src/app/bar.rs".to_string(), "super::bar".to_string()),
                ("src/srv/mod.rs".to_string(), "crate::srv".to_string()),
            ]
        );
    }

    #[test]
    fn resolves_module_tree_with_mod_and_sibling_files() {
        let files = [
            ("src/lib.rs", "pub mod core;\n"),
            ("src/core/mod.rs", "pub mod util;\npub use util::helper;\n"),
            (
                "src/core/util.rs",
                "use super::trait_b::Thing;\nuse self::sibling::S;\npub fn helper() {}\n",
            ),
            ("src/core/sibling.rs", "pub struct S;\n"),
            ("src/core/trait_b/mod.rs", "pub struct Thing;\n"),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);

        let graph = build_graph(&rust_files, &symbols);
        let edges: Vec<(String, String, String)> = graph
            .iter()
            .map(|e| (e.from.clone(), e.to.clone(), e.via.clone()))
            .collect();

        // src/core/mod.rs `pub use util::helper` resolves to util.rs.
        assert!(edges.contains(&(
            "src/core/mod.rs".to_string(),
            "src/core/util.rs".to_string(),
            "util".to_string()
        )));
        // util.rs super::trait_b -> sibling module dir (non-mod.rs file = one
        // dir up from the file itself).
        assert!(edges.contains(&(
            "src/core/util.rs".to_string(),
            "src/core/trait_b/mod.rs".to_string(),
            "super::trait_b".to_string()
        )));
        // util.rs self::sibling -> file in the same dir.
        assert!(edges.contains(&(
            "src/core/util.rs".to_string(),
            "src/core/sibling.rs".to_string(),
            "sibling".to_string()
        )));
        assert_eq!(edges.len(), 3, "only the three resolved edges expected");
    }

    #[test]
    fn ignores_external_crate_and_wildcard_imports() {
        let files = [
            ("src/lib.rs", "pub mod app;\n"),
            (
                "src/app.rs",
                "use serde::{Deserialize, Serialize};\nuse tokio::time::sleep;\nuse crate::app::other;\npub fn x() {}\n",
            ),
            ("src/app/other.rs", "pub fn other() {}\n"),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);

        let graph = build_graph(&rust_files, &symbols);
        assert_eq!(graph.len(), 1);
        assert_eq!(graph[0].from, "src/app.rs");
        assert_eq!(graph[0].to, "src/app/other.rs");
        // Longest module prefix wins: `app::other` resolves through the
        // existing `src/app/other.rs` module file.
        assert_eq!(graph[0].via, "crate::app::other");
    }

    #[test]
    fn ranks_by_centrality_deterministically() {
        let files = [
            ("lib.rs", "pub mod db;\npub mod api;\n"),
            (
                "db/conn.rs",
                "use crate::api::models::User;\npub fn connect() {}\n",
            ),
            ("db/mod.rs", "pub mod conn;\n"),
            (
                "api/models.rs",
                "use crate::db::conn::connect;\npub struct User {}\n",
            ),
            ("api/mod.rs", "pub mod models;\n"),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);
        let graph = build_graph(&rust_files, &symbols);

        let ranked = rank_files(&graph, &rust_files);
        let ranked_again = rank_files(&graph, &rust_files);
        assert_eq!(ranked, ranked_again, "ranking must be deterministic");

        // Both participating files have in-degree 1 (score 2) + out-degree 1
        // (score 3 total); the rest have score 0.
        let by_path: BTreeMap<String, u64> = ranked.into_iter().collect();
        assert_eq!(by_path["db/conn.rs"], 3);
        assert_eq!(by_path["api/models.rs"], 3);
        assert_eq!(by_path["lib.rs"], 0);
        // Ranking order: highest score first; ties by path.
        let order: Vec<String> = ranked_again.into_iter().map(|(p, _)| p).collect();
        assert_eq!(order[0], "api/models.rs".to_string());
        assert_eq!(order[1], "db/conn.rs".to_string());
    }

    #[test]
    fn expands_dependency_hops_and_reverse_lookup() {
        let files = [
            ("lib.rs", "pub mod a;\npub mod b;\npub mod c;\n"),
            ("a.rs", "use crate::b::f;\npub fn af() {}\n"),
            ("b.rs", "use crate::c::g;\npub fn f() {}\n"),
            ("c.rs", "pub fn g() {}\n"),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);
        let graph = build_graph(&rust_files, &symbols);

        let one_hop = expand_dependencies(&graph, &["a.rs"], 1);
        assert_eq!(
            one_hop.into_iter().collect::<Vec<_>>(),
            vec!["b.rs".to_string()]
        );
        let two_hops = expand_dependencies(&graph, &["a.rs"], 2);
        assert_eq!(
            two_hops.into_iter().collect::<Vec<_>>(),
            vec!["b.rs".to_string(), "c.rs".to_string()]
        );

        let deps = dependent_map(&graph);
        assert_eq!(deps["c.rs"], vec!["b.rs".to_string()]);
        let fwd = dependency_map(&graph);
        assert_eq!(fwd["b.rs"], vec!["c.rs".to_string()]);
    }

    #[test]
    fn empty_content_yields_empty_inventory() {
        let sym = extract_rust_symbols("// nothing here\n");
        assert!(sym.structs.is_empty());
        assert!(sym.functions.is_empty());
        assert!(sym.imports.is_empty());
        assert!(sym.exports.is_empty());
    }
}
