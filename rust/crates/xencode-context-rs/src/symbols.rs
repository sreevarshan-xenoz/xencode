//! Repository intelligence (M1): symbol extraction + dependency graph.
//!
//! Pass 2 of the context engine, still fully deterministic and zero-LLM.
//! This is the seed of the Xencode "repository map" (Aider-style):
//!
//!   - regex extraction of `structs` / `enums` / `traits` / `impls` / `types` /
//!     `functions` / `imports` / `exports` / `mods`, plus the file's
//!     documentation prose in `docs`, from Rust source files (`symbols.json`)
//!   - module resolution (`crate::`, `super::`, `self::`, plain-relative)
//!     against the known Rust file set, producing a file→file
//!     dependency graph (`deps.json`), from `use` statements, `mod`
//!     declarations and `impl Trait for Type` blocks
//!   - deterministic centrality ranking + dependency expansion, which M2's
//!     retrieval/budgeter consumes as its structural signal
//!
//! Resolution is intentionally cheap and best-effort: `use` statements that
//! point at external crates (not resolvable to a file in this workspace) are
//! simply not turned into edges, and neither is an `impl` of a trait that two
//! indexed files both define — a name with no single owner is refused, not
//! guessed. Tree-sitter may replace the extraction layer later; the on-disk
//! schemas are stable.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Symbol inventory for one source file (`symbols.json` value).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PerFileSymbols {
    /// `struct` declarations, public or not.
    #[serde(default)]
    pub structs: Vec<String>,
    #[serde(default)]
    pub functions: Vec<String>,
    /// Raw `use` statements (trimmed, `use`/`;` removed).
    #[serde(default)]
    pub imports: Vec<String>,
    /// Names made visible by `pub use`, i.e. the last path segment of each
    /// re-export (`pub use database::Pool` → `Pool`, not `database`).
    #[serde(default)]
    pub exports: Vec<String>,
    /// Module declarations: the names in `mod x;` / `pub mod x;`. These carry no
    /// `use` keyword, so they are invisible to `imports`, but they are the edge
    /// from a crate root (or a `mod.rs`) to the file that module lives in —
    /// without them a module tree contributes no edges at all. Inline
    /// `mod name { … }` blocks declare nothing new and are not collected.
    #[serde(default)]
    pub mods: Vec<String>,
    /// `enum` declarations, public or not.
    #[serde(default)]
    pub enums: Vec<String>,
    /// Traits *defined* here. Paired with `impls` to resolve which file a trait
    /// is implemented from.
    #[serde(default)]
    pub traits: Vec<String>,
    /// Trait names this file implements (`impl MyTrait for Foo` → `MyTrait`),
    /// excluding inherent `impl Foo {}` blocks.
    #[serde(default)]
    pub impls: Vec<String>,
    /// `type Alias = …` declarations.
    #[serde(default)]
    pub types: Vec<String>,
    /// The file's documentation comments joined into one string, capped at
    /// [`DOC_TEXT_CAP`] bytes. Symbol names and file names are identifiers
    /// someone wrote for the compiler; this is the only prose in the inventory,
    /// and it is what a lexical index can match a question phrased in words
    /// against. Doc tests (`/// ```), attribute docs (`#![doc = …]`) and the
    /// plain `//` comments that merely happen to start with three slashes are
    /// left out.
    #[serde(default)]
    pub docs: String,
}

/// How much documentation text one file contributes. Enough for a module header
/// and the doc comment of every public item in an average file; small enough
/// that the whole text of this workspace's index stays well under a megabyte.
pub const DOC_TEXT_CAP: usize = 1200;

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
    mods: Regex,
    enums: Regex,
    traits: Regex,
    impls: Regex,
    types: Regex,
}

/// Prefixes that can appear before `fn`, `struct`, `enum`, `trait` or `type` in
/// any order: visibility, `const`, `async`, `unsafe`, `default`, `auto`, and
/// `extern` with an optional ABI (`extern "C" fn`).
const ITEM_PREFIX: &str = r#"(?:pub(?:\([^)]*\))?\s+|crate\s+|const\s+|async\s+|unsafe\s+|default\s+|auto\s+|extern(?:\s*"[^"]*")?\s+)*"#;

fn regexes() -> &'static RegexCache {
    static CACHE: std::sync::OnceLock<RegexCache> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| RegexCache {
        // `pub` is not required: a private `struct` is still a declaration this
        // file owns, and hiding it made the inventory of any module-private type
        // empty.
        structs: Regex::new(&format!(
            r"(?m)^\s*{ITEM_PREFIX}struct\s+([A-Za-z_][A-Za-z0-9_]*)"
        ))
        .unwrap(),
        functions: Regex::new(&format!(
            r"(?m)^\s*{ITEM_PREFIX}fn\s+([A-Za-z_][A-Za-z0-9_]*)"
        ))
        .unwrap(),
        imports: Regex::new(r"(?m)^\s*(?:pub\s+)?use\s+([^;]+);").unwrap(),
        // The whole path, not its first segment: the name a `pub use` re-exports
        // is what the crate exports, and `export_names` reads it off the end.
        exports: Regex::new(r"(?m)^\s*pub\s+use\s+([^;]+);").unwrap(),
        // Trailing `;` is load-bearing: `mod tests { … }` is an inline namespace,
        // not a file declaration. Read as one it would claim any module that
        // happens to share the block's name — `tests.rs` is common enough — as a
        // child of every file that has a test block.
        mods: Regex::new(
            r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
        )
        .unwrap(),
        enums: Regex::new(&format!(
            r"(?m)^\s*{ITEM_PREFIX}enum\s+([A-Za-z_][A-Za-z0-9_]*)"
        ))
        .unwrap(),
        traits: Regex::new(&format!(
            r"(?m)^\s*{ITEM_PREFIX}trait\s+([A-Za-z_][A-Za-z0-9_]*)"
        ))
        .unwrap(),
        // The trait of an `impl Trait for Type`, and nothing else: an inherent
        // `impl Type {}` names a type this file can already reach, so it is not a
        // dependency. Generic arguments on either side are dropped with it.
        impls: Regex::new(r"(?m)^\s*(?:unsafe\s+)?impl(?:<[^{;]*?>)?\s+([A-Za-z_][A-Za-z0-9_:]*)\s*(?:<[^{;]*?>)?\s+for\b").unwrap(),
        types: Regex::new(&format!(
            r"(?m)^\s*{ITEM_PREFIX}type\s+([A-Za-z_][A-Za-z0-9_]*)\s*="
        ))
        .unwrap(),
    })
}

/// Extract a symbol inventory from a Rust file's text.
/// All lists are sorted + deduped so output is deterministic.
pub fn extract_rust_symbols(content: &str) -> PerFileSymbols {
    let c = regexes();

    let collect = |re: &Regex| -> Vec<String> {
        let mut names: Vec<String> = re
            .captures_iter(content)
            .map(|m| m[1].trim().to_string())
            .collect();
        names.sort();
        names.dedup();
        names
    };

    // `exports` is the one list that is not a bare capture: a `pub use` can name
    // several items at once, so its payload expands to more than one entry.
    let mut exports: Vec<String> = c
        .exports
        .captures_iter(content)
        .flat_map(|m| export_names(&m[1]))
        .collect();
    exports.sort();
    exports.dedup();

    let mut impls: Vec<String> = c
        .impls
        .captures_iter(content)
        .filter_map(|m| trait_impl_target(&m[1]))
        .collect();
    impls.sort();
    impls.dedup();

    PerFileSymbols {
        structs: collect(&c.structs),
        functions: collect(&c.functions),
        imports: collect(&c.imports),
        exports,
        mods: collect(&c.mods),
        enums: collect(&c.enums),
        traits: collect(&c.traits),
        impls,
        types: collect(&c.types),
        docs: doc_text(content),
    }
}

/// The documentation prose of a file: every `//!` and `///` line, in source
/// order, joined and cut to [`DOC_TEXT_CAP`] bytes on a character boundary.
/// Fenced code samples inside a doc comment are skipped — they are Rust, not
/// an explanation of what the file does, and indexing them would put the same
/// vocabulary in every file that shows an example.
fn doc_text(content: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    let mut in_code = false;
    for line in content.lines() {
        let Some(text) = doc_line(line) else {
            continue;
        };
        if text.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if !in_code && !text.is_empty() {
            parts.push(text);
        }
    }
    let joined = parts.join(" ");
    if joined.len() <= DOC_TEXT_CAP {
        return joined;
    }
    let mut cut = joined[..DOC_TEXT_CAP].to_string();
    while !cut.is_char_boundary(cut.len()) {
        cut.pop();
    }
    if let Some(at) = cut.rfind(' ') {
        cut.truncate(at);
    }
    cut
}

/// A line's documentation text with the comment markers removed, or `None` if
/// the line is not one. `//!` is a module/crate header, `///` documents the item
/// below it; `//!/// …` nests, so the marker is trimmed twice.
fn doc_line(line: &str) -> Option<&str> {
    let s = line.trim_start();
    let rest = s.strip_prefix("//!").or_else(|| s.strip_prefix("///"))?;
    Some(rest.trim_start_matches(['!', '/']).trim())
}

/// The names a `pub use` payload re-exports: the last segment of each path
/// (`database::Pool` → `Pool`), honouring ` as ` aliases and brace groups
/// (`foo::{A, B::C}` → `A`, `C`). A glob (`foo::*`) re-exports names this file
/// does not list, so it contributes none.
fn export_names(raw: &str) -> Vec<String> {
    let s = raw.trim();
    if s.is_empty() {
        return Vec::new();
    }
    let items: Vec<String> = if let Some((head, tail)) = s.split_once('{') {
        let Some((inner, _)) = tail.split_once('}') else {
            return Vec::new();
        };
        inner
            .split(',')
            .map(|item| format!("{head}{item}"))
            .collect()
    } else {
        vec![s.to_string()]
    };
    items
        .iter()
        .filter_map(|item| {
            let item = item.trim();
            if item.is_empty() || item.contains('*') || item.contains('{') || item.contains('}') {
                return None;
            }
            let last = match item.rsplit_once(" as ") {
                Some((_, alias)) => alias,
                None => item.rsplit("::").next()?,
            };
            let last = last.trim();
            if is_ident(last) {
                Some(last.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The trait an `impl Trait for Type` implements, from the trait path's last
/// segment (`std::fmt::Debug` → `Debug`).
fn trait_impl_target(path: &str) -> Option<String> {
    let last = path.rsplit("::").next()?.trim();
    is_ident(last).then(|| last.to_string())
}

/// The directory a file's own submodules live in. `src/lib.rs`, `src/main.rs`
/// and `src/api/mod.rs` all describe modules whose children sit beside them, so
/// for those it is the file's directory; for any other file `src/wire.rs` the
/// children live in the directory named after it (`src/wire/`).
fn module_dir(file: &str) -> String {
    let base = dir_of(file);
    let name = file.rsplit('/').next().unwrap_or(file);
    if matches!(name, "mod.rs" | "lib.rs" | "main.rs") {
        base
    } else {
        let stem = name.strip_suffix(".rs").unwrap_or(name);
        if base.is_empty() {
            stem.to_string()
        } else {
            format!("{base}/{stem}")
        }
    }
}

/// The file a `mod name;` declaration in `file` refers to, if it is one of the
/// known files. Rust accepts either `dir/name.rs` or `dir/name/mod.rs` for a
/// module declared in `dir/`, where `dir` is the directory the declaring file's
/// *module* owns (see `module_dir`).
fn resolve_mod_decl(file: &str, name: &str, files: &HashSet<&str>) -> Option<String> {
    let base = module_dir(file);
    [join_rel(&base, name, false), join_rel(&base, name, true)]
        .into_iter()
        .find(|candidate| files.contains(candidate.as_str()))
}

/// Which namespace an import resolves relative to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Qualifier {
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
/// into qualifier + segment pairs. Brace groups expand one level:
/// `crate::ui::{a, b::c}` yields the `crate::ui::a` and `crate::ui::b::c`
/// pairs. Globs and nested groups cannot resolve statically and are
/// skipped — never guessed.
pub(crate) fn parse_import(raw: &str) -> Vec<(Qualifier, Vec<String>)> {
    let mut s = raw.trim();
    if s.is_empty() || s.contains('*') {
        return Vec::new();
    }
    if let Some(ix) = s.find(" as ") {
        s = s[..ix].trim();
    }
    if s.is_empty() {
        return Vec::new();
    }
    if let Some((head, tail)) = s.split_once('{') {
        let Some((items, _)) = tail.split_once('}') else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for item in items.split(',') {
            let item = item.trim();
            if item.is_empty() || item.contains('{') || item.contains('}') || item.contains('*') {
                continue;
            }
            out.extend(single_import(&format!("{head}{item}")));
        }
        return out;
    }
    single_import(s)
}

/// Parse one qualifier path (no braces) into a single pair.
fn single_import(s: &str) -> Vec<(Qualifier, Vec<String>)> {
    let mut segs: Vec<String> = s
        .split("::")
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    if segs.is_empty() {
        return Vec::new();
    }
    match segs[0].as_str() {
        "crate" => {
            if segs.len() < 2 {
                return Vec::new();
            }
            vec![(Qualifier::Crate, segs.split_off(1))]
        }
        "self" => {
            if segs.len() < 2 {
                return Vec::new();
            }
            vec![(Qualifier::SelfMod, segs.split_off(1))]
        }
        "super" => {
            let mut up = 0usize;
            while segs.first().map(|s| s.as_str()) == Some("super") {
                segs.remove(0);
                up += 1;
            }
            if segs.is_empty() {
                return Vec::new();
            }
            vec![(Qualifier::Super { up }, segs)]
        }
        _ => vec![(Qualifier::Plain, segs)],
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
        // `self::` is the current module, whose children live in `module_dir`.
        // For `src/wire.rs` that is `src/wire`, not the `src` a bare path tries.
        Qualifier::SelfMod => module_dir(file),
        Qualifier::Plain => dir_of(file),
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
/// Brace groups try each pair in order; bare (`Plain`) paths try the file's
/// own dir first, then the crate root (Rust 2018 uniform paths) — a match
/// only counts when a real file exists, so external crates never resolve.
pub fn resolve_import(
    file: &str,
    import: &str,
    crate_root: Option<&str>,
    files: &HashSet<&str>,
) -> Option<(String, String)> {
    let pairs = parse_import(import);
    if pairs.is_empty() {
        return None;
    }
    for (qual, segs) in &pairs {
        if let Some(hit) = resolve_pair(file, qual, segs, crate_root, files) {
            return Some(hit);
        }
    }
    None
}

/// Resolve one qualifier pair against a base dir, then (for bare paths) the
/// crate root. Factored out so `broken_imports` can check pairs individually.
pub(crate) fn resolve_pair(
    file: &str,
    qual: &Qualifier,
    segs: &[String],
    crate_root: Option<&str>,
    files: &HashSet<&str>,
) -> Option<(String, String)> {
    let base = match qual {
        Qualifier::Crate => crate_root?.to_string(),
        other => module_base(file, other),
    };
    if base.is_empty() && crate_root.is_none() {
        return None;
    }
    if let Some(hit) = try_base(&base, qual, segs, files) {
        return Some(hit);
    }
    if matches!(qual, Qualifier::Plain) {
        if let Some(root) = crate_root {
            if root != base {
                return try_base(root, qual, segs, files);
            }
        }
    }
    None
}

/// Try longest-prefix file matches (`a/b/Item` → `a/b.rs` before `a.rs`)
/// under one base dir.
fn try_base(
    base: &str,
    qual: &Qualifier,
    segs: &[String],
    files: &HashSet<&str>,
) -> Option<(String, String)> {
    for i in (1..=segs.len()).rev() {
        let mod_path = segs[..i].join("/");
        if mod_path.is_empty() {
            continue;
        }
        for candidate in [
            join_rel(base, &mod_path, false),
            join_rel(base, &mod_path, true),
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
pub(crate) fn crate_roots(rust_files: &[String]) -> Vec<String> {
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
pub(crate) fn crate_root_for<'a>(file: &str, roots: &'a [String]) -> Option<&'a str> {
    let mut candidates: Vec<&String> = roots
        .iter()
        .filter(|r| r.is_empty() || file.starts_with(&format!("{r}/")))
        .collect();
    candidates.sort_by_key(|r| r.len());
    candidates.last().map(|r| r.as_str())
}

/// The file that declares trait `name`, when exactly one known file does.
/// Two files declaring the same trait name is a real possibility in a
/// multi-crate workspace, and guessing between them would invent an edge, so
/// an ambiguous name yields none.
fn resolve_trait_decl<'a>(
    file: &str,
    name: &str,
    trait_files: &'a BTreeMap<&'a str, Vec<&'a str>>,
) -> Option<&'a str> {
    let [only] = trait_files.get(name)?.as_slice() else {
        return None;
    };
    (*only != file).then_some(*only)
}

/// Build the file→file dependency graph from extracted symbols.
/// Edges are sorted + deduped for deterministic output.
pub fn build_graph(
    rust_files: &[String],
    symbols: &BTreeMap<String, PerFileSymbols>,
) -> Vec<DepEdge> {
    let file_set: HashSet<&str> = rust_files.iter().map(|s| s.as_str()).collect();
    let roots = crate_roots(rust_files);

    // Trait name → the files defining it, for `impl Trait for Type` edges.
    let mut trait_files: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (file, sym) in symbols {
        for name in &sym.traits {
            trait_files
                .entry(name.as_str())
                .or_default()
                .push(file.as_str());
        }
    }

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
        // Module declarations, in file order after the imports so the reason a
        // tree is connected ("this file declares that module") stays separate
        // from "this file names that path in a use statement".
        for name in &sym.mods {
            if let Some(to) = resolve_mod_decl(file, name, &file_set) {
                edges.push(DepEdge {
                    from: file.clone(),
                    to,
                    via: format!("mod {name}"),
                });
            }
        }
        // An `impl Trait for Type` depends on the file that declares the trait.
        // Rust lets you write that impl with no `use` of the trait in sight (a
        // prelude or a re-export can bring it in), so imports alone never showed
        // the relationship.
        for name in &sym.impls {
            if let Some(to) = resolve_trait_decl(file, name, &trait_files) {
                edges.push(DepEdge {
                    from: file.clone(),
                    to: to.to_string(),
                    via: format!("impl {name}"),
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
    fn extracts_declared_items_regardless_of_visibility() {
        let content = r#"
pub struct User {
    id: u64,
}

struct Session;

#[derive(Debug)]
pub enum Choice { A, B }

pub trait Speak { fn say(&self) -> String; }

type Handle = User;

pub fn connect() {}

pub async fn refresh() {}

const fn tick() -> u64 { 0 }

pub unsafe extern "C" fn callback() {}

use std::collections::HashMap;

// These examples are deliberately not `crate::`-anchored: this text lives in a
// file the indexer reads, so an anchored path here would be reported as one of
// this workspace's own broken imports.
pub use database::Pool;
pub use net::{http::Client, ws};
pub use legacy::Old as Newer;
pub use bundle::*;

fn helper() {}
"#;
        let sym = extract_rust_symbols(content);
        // A private `struct` is still something this file declares.
        assert_eq!(sym.structs, vec!["Session".to_string(), "User".to_string()]);
        assert_eq!(sym.enums, vec!["Choice".to_string()]);
        assert_eq!(sym.traits, vec!["Speak".to_string()]);
        assert_eq!(sym.types, vec!["Handle".to_string()]);
        // `const fn` and `extern "C" fn` are functions too; the `fn say` inside
        // the trait's own line is not.
        assert_eq!(
            sym.functions,
            vec![
                "callback".to_string(),
                "connect".to_string(),
                "helper".to_string(),
                "refresh".to_string(),
                "tick".to_string(),
            ]
        );
        assert_eq!(
            sym.imports,
            vec![
                "bundle::*".to_string(),
                "database::Pool".to_string(),
                "legacy::Old as Newer".to_string(),
                "net::{http::Client, ws}".to_string(),
                "std::collections::HashMap".to_string(),
            ]
        );
        // The exported *name*, not the module it came from; a glob exports names
        // this file cannot list, so it contributes none.
        assert_eq!(
            sym.exports,
            vec![
                "Client".to_string(),
                "Newer".to_string(),
                "Pool".to_string(),
                "ws".to_string(),
            ]
        );
    }

    #[test]
    fn trait_impls_name_the_trait_and_not_the_type() {
        let content = r#"
pub struct Dog;
pub struct LoudDog;
pub struct Kit;

impl Dog { pub fn bark(&self) {} }

impl Speak for Dog {}
impl crate::speak::Speak for LoudDog {}
impl<T: Clone> Pack<T> for Kit {}
impl std::fmt::Debug for Dog { fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result {} }
"#;
        let sym = extract_rust_symbols(content);
        // The inherent `impl Dog` is absent and the generic arguments are
        // stripped, so what is left is the trait each impl depends on.
        assert_eq!(
            sym.impls,
            vec!["Debug".to_string(), "Pack".to_string(), "Speak".to_string(),]
        );
    }

    #[test]
    fn implements_trait_reaches_the_file_that_declares_it() {
        let files = [
            ("src/lib.rs", "pub mod speak;\npub mod dog;\n"),
            (
                "src/speak.rs",
                "pub trait Speak { fn say(&self) -> String; }\npub struct Echo;\nimpl Speak for Echo {}\n",
            ),
            (
                "src/dog.rs",
                "pub struct Dog;\nimpl crate::speak::Speak for Dog {}\n",
            ),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);
        let graph = build_graph(&rust_files, &symbols);
        let edges: Vec<(&str, &str, &str)> = graph
            .iter()
            .filter(|e| e.via.starts_with("impl "))
            .map(|e| (e.from.as_str(), e.to.as_str(), e.via.as_str()))
            .collect();
        // dog.rs implements a trait it never `use`s, and speak.rs implements its
        // own trait in place — only the first is an edge.
        assert_eq!(edges, vec![("src/dog.rs", "src/speak.rs", "impl Speak")]);
    }

    #[test]
    fn an_ambiguous_trait_name_yields_no_edge() {
        // Two crates in one workspace may each declare `Shape`; guessing which one
        // an impl refers to would invent a dependency.
        let files = [
            ("a/lib.rs", "pub mod shape;\npub mod widget;\n"),
            ("a/shape.rs", "pub trait Shape {}\n"),
            ("b/lib.rs", "pub mod shape;\n"),
            ("b/shape.rs", "pub trait Shape {}\n"),
            (
                "a/widget.rs",
                "pub struct Widget;\nimpl b::shape::Shape for Widget {}\n",
            ),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);
        let graph = build_graph(&rust_files, &symbols);
        assert!(
            !graph.iter().any(|e| e.via.starts_with("impl ")),
            "{graph:?}"
        );
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
                // Module declarations: `pub mod x;` is the only thing that
                // connects a crate root to its modules, and a `mod.rs` to its
                // submodules, so a module tree is invisible without them.
                (
                    "api/mod.rs".to_string(),
                    "api/models.rs".to_string(),
                    "mod models".to_string()
                ),
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
                (
                    "db/mod.rs".to_string(),
                    "db/conn.rs".to_string(),
                    "mod conn".to_string()
                ),
                (
                    "lib.rs".to_string(),
                    "api/mod.rs".to_string(),
                    "mod api".to_string()
                ),
                (
                    "lib.rs".to_string(),
                    "db/mod.rs".to_string(),
                    "mod db".to_string()
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
                "use super::bar::Baz;\nuse crate::srv::serve;\nuse self::detail::D;\npub struct Foo {}\n",
            ),
            ("src/app/bar.rs", "pub struct Bar;\n"),
            ("src/app/foo/detail.rs", "pub struct D;\n"),
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

        // `self::detail` is `src/app/foo/detail.rs`: the module `foo` owns a
        // directory, so its children are not the file's siblings.
        assert_eq!(
            edges_a,
            vec![
                ("src/app/bar.rs".to_string(), "super::bar".to_string()),
                ("src/app/foo/detail.rs".to_string(), "detail".to_string()),
                ("src/srv/mod.rs".to_string(), "crate::srv".to_string()),
            ]
        );
    }

    #[test]
    fn resolves_module_tree_with_mod_and_qualified_imports() {
        let files = [
            ("src/lib.rs", "pub mod core;\n"),
            ("src/core/mod.rs", "pub mod util;\npub use util::helper;\n"),
            (
                "src/core/util.rs",
                "use super::trait_b::Thing;\nuse self::sibling::S;\npub fn helper() {}\n",
            ),
            ("src/core/util/sibling.rs", "pub struct S;\n"),
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
        // util.rs self::sibling -> a child of the `util` module, which lives in
        // the directory named after the file.
        assert!(edges.contains(&(
            "src/core/util.rs".to_string(),
            "src/core/util/sibling.rs".to_string(),
            "sibling".to_string()
        )));
        // `pub mod core;` in lib.rs and `pub mod util;` in core/mod.rs are the
        // two module-declaration edges; the three below come from `use`.
        assert!(edges.contains(&(
            "src/lib.rs".to_string(),
            "src/core/mod.rs".to_string(),
            "mod core".to_string()
        )));
        assert!(edges.contains(&(
            "src/core/mod.rs".to_string(),
            "src/core/util.rs".to_string(),
            "mod util".to_string()
        )));
        assert_eq!(edges.len(), 5, "only the five resolved edges expected");
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
        // Two, not one: `pub mod app;` in lib.rs is a real edge. The point of
        // this test is that nothing external or globbed sneaks in.
        assert_eq!(graph.len(), 2, "{graph:?}");
        let resolved = graph
            .iter()
            .find(|e| e.via != "mod app")
            .expect("the crate-internal import should resolve");
        assert_eq!(resolved.from, "src/app.rs");
        assert_eq!(resolved.to, "src/app/other.rs");
        // Longest module prefix wins: `app::other` resolves through the
        // existing `src/app/other.rs` module file.
        assert_eq!(resolved.via, "crate::app::other");
    }

    #[test]
    fn expands_brace_groups_and_falls_back_to_crate_root() {
        let files = [
            ("src/main.rs", "fn main() {}\n"),
            (
                "src/app/bar.rs",
                "use models::Foo;\nuse crate::ui::{Theme, Widget};\nuse serde::Deserialize;\nuse foo::*;\npub fn bar() {}\n",
            ),
            ("src/models.rs", "pub struct Foo {}\n"),
            ("src/ui.rs", "pub struct Theme {}\n"),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);
        let file_set: HashSet<&str> = rust_files.iter().map(|s| s.as_str()).collect();
        let roots = crate_roots(&rust_files);
        let root = crate_root_for("src/app/bar.rs", &roots);
        assert_eq!(root, Some("src"));

        // 2018 sibling import: dir-relative miss, crate-root hit.
        let hit = resolve_import("src/app/bar.rs", "models::Foo", root, &file_set);
        assert_eq!(hit.map(|(to, _)| to).as_deref(), Some("src/models.rs"));
        // Brace group: both pairs resolve through the ui.rs prefix.
        let brace = resolve_import(
            "src/app/bar.rs",
            "crate::ui::{Theme, Widget}",
            root,
            &file_set,
        );
        assert_eq!(brace.map(|(to, _)| to).as_deref(), Some("src/ui.rs"));
        // External crate and glob: no file match, no edge.
        assert!(resolve_import("src/app/bar.rs", "serde::Deserialize", root, &file_set).is_none());
        assert!(resolve_import("src/app/bar.rs", "foo::*", root, &file_set).is_none());

        // End to end: the new edges appear in the graph.
        let graph = build_graph(&rust_files, &symbols);
        let tos: Vec<&str> = graph.iter().map(|e| e.to.as_str()).collect();
        assert!(tos.contains(&"src/models.rs"), "{tos:?}");
        assert!(tos.contains(&"src/ui.rs"), "{tos:?}");
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

        // A file is worth 2 per dependents and 1 per thing it imports. With the
        // module tree visible, that ranking changes shape: the two files that
        // import each other are each imported by a third file now (their
        // `mod.rs`), so they rise from 3 to 5; each `mod.rs` has one submodule
        // and one parent; and `lib.rs`, which imports nothing but declares two
        // modules, is no longer a zero-scoring leaf.
        let by_path: BTreeMap<String, u64> = ranked.into_iter().collect();
        assert_eq!(by_path["db/conn.rs"], 5);
        assert_eq!(by_path["api/models.rs"], 5);
        assert_eq!(by_path["api/mod.rs"], 3);
        assert_eq!(by_path["db/mod.rs"], 3);
        assert_eq!(by_path["lib.rs"], 2);
        // Ranking order: highest score first; ties by path.
        let order: Vec<String> = ranked_again.into_iter().map(|(p, _)| p).collect();
        assert_eq!(order[0], "api/models.rs".to_string());
        assert_eq!(order[1], "db/conn.rs".to_string());
        assert_eq!(order[2], "api/mod.rs".to_string());
    }

    #[test]
    fn dependency_and_dependent_maps_are_inverses() {
        let files = [
            ("lib.rs", "pub mod a;\npub mod b;\npub mod c;\n"),
            ("a.rs", "use crate::b::f;\npub fn af() {}\n"),
            ("b.rs", "use crate::c::g;\npub fn f() {}\n"),
            ("c.rs", "pub fn g() {}\n"),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);
        let graph = build_graph(&rust_files, &symbols);

        let deps = dependent_map(&graph);
        // c.rs is named by b.rs's import and declared by lib.rs's `pub mod c;`.
        assert_eq!(deps["c.rs"], vec!["b.rs".to_string(), "lib.rs".to_string()]);
        let fwd = dependency_map(&graph);
        assert_eq!(fwd["b.rs"], vec!["c.rs".to_string()]);
    }

    #[test]
    fn a_module_declared_by_a_plain_file_lives_under_the_directory_named_after_it() {
        // `src/wire.rs` is the module `crate::wire`, so the module it declares is
        // `src/wire/frame.rs` — not a sibling `src/frame.rs`.
        let files = [
            ("src/lib.rs", "pub mod wire;\n"),
            ("src/wire.rs", "mod frame;\npub struct Wire;\n"),
            ("src/wire/frame.rs", "pub struct Frame;\n"),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);
        let graph = build_graph(&rust_files, &symbols);
        let edges: Vec<(&str, &str, &str)> = graph
            .iter()
            .map(|e| (e.from.as_str(), e.to.as_str(), e.via.as_str()))
            .collect();
        assert_eq!(
            edges,
            vec![
                ("src/lib.rs", "src/wire.rs", "mod wire"),
                ("src/wire.rs", "src/wire/frame.rs", "mod frame"),
            ]
        );
    }

    #[test]
    fn inline_module_blocks_declare_no_file_and_yield_no_edge() {
        // Nearly every Rust file has `mod tests { … }`. If that counted as a
        // module declaration the graph would gain one self-ish edge per file.
        let files = [
            (
                "src/lib.rs",
                "pub mod api;\nmod tests {\n    fn t() {}\n}\n",
            ),
            ("src/api.rs", "pub struct Api;\n"),
        ];
        let rust_files = rust_paths(&files);
        let symbols = symbols_from(&files);
        assert_eq!(symbols["src/lib.rs"].mods, vec!["api".to_string()]);

        let graph = build_graph(&rust_files, &symbols);
        let edges: Vec<(&str, &str, &str)> = graph
            .iter()
            .map(|e| (e.from.as_str(), e.to.as_str(), e.via.as_str()))
            .collect();
        assert_eq!(edges, vec![("src/lib.rs", "src/api.rs", "mod api")]);
    }

    #[test]
    fn empty_content_yields_empty_inventory() {
        let sym = extract_rust_symbols("// nothing here\n");
        assert!(sym.structs.is_empty());
        assert!(sym.functions.is_empty());
        assert!(sym.imports.is_empty());
        assert!(sym.exports.is_empty());
        assert!(sym.mods.is_empty());
        assert!(sym.enums.is_empty());
        assert!(sym.traits.is_empty());
        assert!(sym.impls.is_empty());
        assert!(sym.types.is_empty());
        // An ordinary `//` comment is not documentation, and its text is not.
        assert!(sym.docs.is_empty());
    }

    #[test]
    fn doc_comments_are_collected_as_prose() {
        let content = "//! Scores a file against a query.\n\
             //!\n\
             //! ```\n\
             //! let example = 1;\n\
             //! ```\n\
             use std::fmt;\n\
             /// Loads the index from disk.\n\
             ///\n\
             /// Panics when the path is missing.\n\
             pub fn load(&self) {}\n\
             // an ordinary comment, not documentation\n\
             #[doc = \"generated by a macro\"]\n\
             pub struct Index;\n";
        let sym = extract_rust_symbols(content);
        assert_eq!(
            sym.docs,
            "Scores a file against a query. Loads the index from disk. Panics when the path is missing."
        );
    }

    #[test]
    fn doc_text_is_capped_on_a_word() {
        // Each line is longer than the whole cap, so this also proves a single
        // over-long line does not wipe the file's text.
        let filler = "word ".repeat(400);
        let content = format!("//! {filler}\n");
        let sym = extract_rust_symbols(&content);
        assert!(
            !sym.docs.is_empty() && sym.docs.len() <= DOC_TEXT_CAP,
            "{} bytes is not a capped-but-non-empty result",
            sym.docs.len()
        );
        assert!(!sym.docs.ends_with(' '));
        assert!(sym.docs.split(' ').all(|w| w == "word"));
    }
}
