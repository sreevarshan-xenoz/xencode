//! The semantic tier (`LSP-2`): who uses a file, answered from the index
//! `rust-analyzer scip` writes rather than from `use` paths read as text.
//!
//! The name tier in [`crate::impact`] links two files when one writes a module
//! path that resolves to the other. That is name resolution, not type checking:
//! it cannot see a method called through a trait object, a re-export, or a type
//! reached through inference, and it links a whole file when only one name in it
//! is used. rust-analyzer's SCIP index records every occurrence of every symbol
//! with the file that defines it, so an edge here means "this file refers to a
//! symbol that file defines" — and the edge carries the symbol's name.
//!
//! The cost is time. On this repository's workspace (16 crates) one run took 190
//! seconds and wrote 30.7 MB, measured on 2026-10-08. So the index is built on
//! request, never on the way to an answer someone is waiting for, and an index
//! that no longer matches the tree is reported as stale and not used: a stale
//! semantic answer is worse than an honest name-level one.

use crate::impact::{ImpactReport, ImpactTier, ImpactedFile, DECLARED_CAP};
use crate::symbols::DepEdge;
use protobuf::Message;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Where the index lives, under the workspace's own `.xencode`.
pub const SCIP_DIR: &str = "scip";
const INDEX_FILE: &str = "index.scip";
const META_FILE: &str = "meta.json";

/// Longest a `rust-analyzer scip` run may take before it is stopped. Three times
/// the measured run on this repository, so a slower machine still finishes.
pub const SCIP_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// What a finished run recorded about itself, written beside the index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScipMeta {
    /// Seconds since the epoch when the run started. A file changed after this
    /// is newer than the index.
    pub started_unix: u64,
    /// `git rev-parse HEAD` when the run started, if this is a repository.
    pub head: Option<String>,
    /// `rust-analyzer --version`.
    pub rust_analyzer: String,
    /// Wall-clock seconds the run took.
    pub seconds: f64,
    /// Every document in the index, as the index names it (relative to the
    /// workspace, `/`-separated). Kept here so freshness can be checked without
    /// reading the index itself.
    pub files: Vec<String>,
}

/// Why an index cannot be used.
#[derive(Debug)]
pub enum ScipError {
    /// `rust-analyzer` is not on `PATH`, or did not answer `--version`.
    NoRustAnalyzer(String),
    /// The run started and did not produce an index.
    RunFailed(String),
    /// The run was stopped at [`SCIP_TIMEOUT`].
    TimedOut(Duration),
    /// There is no index, or its files could not be read.
    Missing(PathBuf),
    /// The index exists but does not describe the tree as it is now.
    Stale(String),
    /// The bytes on disk are not a SCIP index.
    Unreadable(String),
    Io(PathBuf, std::io::Error),
}

impl std::fmt::Display for ScipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScipError::NoRustAnalyzer(why) => write!(
                f,
                "rust-analyzer is not available ({why}); install it with \
                 `rustup component add rust-analyzer`"
            ),
            ScipError::RunFailed(why) => write!(f, "rust-analyzer scip failed: {why}"),
            ScipError::TimedOut(after) => write!(
                f,
                "rust-analyzer scip was stopped after {} seconds without finishing",
                after.as_secs()
            ),
            ScipError::Missing(path) => write!(
                f,
                "no semantic index at {} — build one with `xencode impact <file> --semantic`",
                path.display()
            ),
            ScipError::Stale(why) => write!(f, "the semantic index is stale: {why}"),
            ScipError::Unreadable(why) => write!(f, "the semantic index cannot be read: {why}"),
            ScipError::Io(path, source) => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for ScipError {}

fn scip_dir(workspace: &Path) -> PathBuf {
    workspace.join(crate::init::XENCODE_DIR).join(SCIP_DIR)
}

/// The index file for a workspace.
pub fn index_path(workspace: &Path) -> PathBuf {
    scip_dir(workspace).join(INDEX_FILE)
}

fn meta_path(workspace: &Path) -> PathBuf {
    scip_dir(workspace).join(META_FILE)
}

/// The record of the last finished run, if there is one.
pub fn read_meta(workspace: &Path) -> Option<ScipMeta> {
    let text = std::fs::read_to_string(meta_path(workspace)).ok()?;
    serde_json::from_str(&text).ok()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn rust_analyzer_version() -> Result<String, ScipError> {
    let output = std::process::Command::new("rust-analyzer")
        .arg("--version")
        .output()
        .map_err(|e| ScipError::NoRustAnalyzer(e.to_string()))?;
    if !output.status.success() {
        // The rustup proxy answers this way when the component is missing.
        let why = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(ScipError::NoRustAnalyzer(if why.is_empty() {
            format!("`rust-analyzer --version` exited {}", output.status)
        } else {
            why
        }));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Run `rust-analyzer scip` over `workspace` and record the run beside the
/// index. Blocks for as long as the run takes, up to `timeout`.
///
/// The new index is written to a temporary name and moved into place only once
/// the run succeeded, so a failed or stopped run leaves the previous index and
/// its record as they were.
pub fn generate(workspace: &Path, timeout: Duration) -> Result<ScipMeta, ScipError> {
    let rust_analyzer = rust_analyzer_version()?;
    let dir = scip_dir(workspace);
    std::fs::create_dir_all(&dir).map_err(|e| ScipError::Io(dir.clone(), e))?;
    let partial = dir.join(format!("{INDEX_FILE}.partial"));
    let log_path = dir.join("rust-analyzer.log");
    let log = std::fs::File::create(&log_path).map_err(|e| ScipError::Io(log_path.clone(), e))?;
    let started_unix = now_unix();
    let head = crate::gitinfo::current_git_info(workspace).map(|info| info.head);
    let started = Instant::now();
    let mut child = std::process::Command::new("rust-analyzer")
        .arg("scip")
        .arg(workspace)
        .arg("--output")
        .arg(&partial)
        .stdin(std::process::Stdio::null())
        .stdout(
            log.try_clone()
                .map_err(|e| ScipError::Io(log_path.clone(), e))?,
        )
        .stderr(log)
        .spawn()
        .map_err(|e| ScipError::NoRustAnalyzer(e.to_string()))?;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&partial);
                return Err(ScipError::TimedOut(timeout));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(e) => return Err(ScipError::RunFailed(e.to_string())),
        }
    };
    let seconds = started.elapsed().as_secs_f64();
    if !status.success() || !partial.is_file() {
        let tail = std::fs::read_to_string(&log_path)
            .unwrap_or_default()
            .lines()
            .rev()
            .take(5)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" | ");
        let _ = std::fs::remove_file(&partial);
        return Err(ScipError::RunFailed(format!(
            "exited {status}; last lines of {}: {tail}",
            log_path.display()
        )));
    }
    let index = read_index(&partial)?;
    let files = index
        .documents
        .iter()
        .map(|d| normalise(&d.relative_path))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let meta = ScipMeta {
        started_unix,
        head,
        rust_analyzer,
        seconds,
        files,
    };
    let final_path = index_path(workspace);
    std::fs::rename(&partial, &final_path).map_err(|e| ScipError::Io(final_path.clone(), e))?;
    let meta_file = meta_path(workspace);
    let text = serde_json::to_string_pretty(&meta).expect("the record serialises");
    std::fs::write(&meta_file, text).map_err(|e| ScipError::Io(meta_file, e))?;
    Ok(meta)
}

fn normalise(path: &str) -> String {
    path.replace('\\', "/")
}

/// Whether a path from the index or its record stays inside the workspace.
///
/// Both files sit in the project's `.xencode`, which a cloned repository can
/// ship, so their paths are input like any other: an absolute path or a `..`
/// would have `find_refs` read and print lines from files outside the project.
fn stays_inside(path: &str) -> bool {
    let path = normalise(path);
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains(':')
        && path.split('/').all(|part| part != "..")
}

fn read_index(path: &Path) -> Result<scip::types::Index, ScipError> {
    let bytes = std::fs::read(path).map_err(|e| ScipError::Io(path.to_path_buf(), e))?;
    let index = scip::types::Index::parse_from_bytes(&bytes)
        .map_err(|e| ScipError::Unreadable(e.to_string()))?;
    if let Some(doc) = index
        .documents
        .iter()
        .find(|d| !stays_inside(&d.relative_path))
    {
        return Err(ScipError::Unreadable(format!(
            "it names `{}`, which is outside the workspace, so none of it is used",
            doc.relative_path
        )));
    }
    Ok(index)
}

/// Whether the index on disk still describes `workspace`, and if not, why not.
///
/// Three things make it stale: `HEAD` moved; the set of Rust files git knows
/// about is not the set the index holds; or an indexed file was modified after
/// the run started. The third is checked by modification time, which a checkout
/// also updates, so a branch switch that kept `HEAD`'s file list is still seen.
pub fn freshness(workspace: &Path) -> Result<ScipMeta, ScipError> {
    let index = index_path(workspace);
    let meta = match read_meta(workspace) {
        Some(meta) if index.is_file() => meta,
        _ => return Err(ScipError::Missing(index)),
    };
    if let Some(outside) = meta.files.iter().find(|f| !stays_inside(f)) {
        return Err(ScipError::Unreadable(format!(
            "its record names `{outside}`, which is outside the workspace, so none of it is used"
        )));
    }
    if let Some(indexed_head) = &meta.head {
        let now = crate::gitinfo::current_git_info(workspace).map(|info| info.head);
        if now.as_deref() != Some(indexed_head.as_str()) {
            return Err(ScipError::Stale(format!(
                "it was built at commit {}, and HEAD is now {}",
                short(indexed_head),
                now.as_deref().map(short).unwrap_or("unknown")
            )));
        }
    }
    let indexed: BTreeSet<&str> = meta.files.iter().map(String::as_str).collect();
    if let Some(tracked) = crate::gitinfo::git_file_set(workspace) {
        let rust: BTreeSet<String> = tracked
            .into_iter()
            .filter(|p| p.ends_with(".rs"))
            .map(|p| normalise(&p))
            .collect();
        if let Some(added) = rust.iter().find(|p| !indexed.contains(p.as_str())) {
            return Err(ScipError::Stale(format!(
                "{added} is in the tree and not in the index"
            )));
        }
    }
    for file in &meta.files {
        let path = workspace.join(file);
        let modified = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        match modified {
            None => {
                return Err(ScipError::Stale(format!(
                    "{file} is in the index and no longer on disk"
                )))
            }
            Some(secs) if secs > meta.started_unix => {
                return Err(ScipError::Stale(format!(
                    "{file} changed after the index was built"
                )))
            }
            Some(_) => {}
        }
    }
    Ok(meta)
}

fn short(sha: &str) -> &str {
    &sha[..sha.len().min(10)]
}

/// The file-level reading of one SCIP index.
#[derive(Debug, Default, Clone)]
pub struct ScipGraph {
    /// One edge per (referring file, defining file, symbol name).
    pub edges: Vec<DepEdge>,
    /// The names each file defines, in the order the index lists them.
    pub defined: BTreeMap<String, Vec<String>>,
    /// Every document in the index.
    pub files: BTreeSet<String>,
}

/// The name a person would use for a SCIP symbol: its last descriptor
/// (`impact` for a function, `ImpactReport` for a type). Local symbols, which
/// never leave their file, have none.
///
/// Neither do modules (`crate/`, `impact/` — SCIP ends a namespace with `/`).
/// Writing `crate::helper` refers to the crate root's module symbol, which
/// lives in `lib.rs`; counting that as a use of `lib.rs` linked every file that
/// writes `crate::` to it, and on this repository turned the second hop of one
/// answer into more than a hundred files that use nothing `lib.rs` defines.
fn display_name(symbol: &str) -> Option<String> {
    if symbol.is_empty() || symbol.ends_with('/') || scip::symbol::is_local_symbol(symbol) {
        return None;
    }
    let parsed = scip::symbol::parse_symbol(symbol).ok()?;
    let name = parsed.descriptors.last()?.name.clone();
    (!name.is_empty()).then_some(name)
}

const DEFINITION: i32 = scip::types::SymbolRole::Definition as i32;

/// Turn an index into file-to-file edges. A reference in file A to a symbol
/// defined in file B is an edge from A to B carrying the symbol's name; a file's
/// references to its own symbols are not edges.
pub fn graph_from_index(index: &scip::types::Index) -> ScipGraph {
    let mut definers: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    let mut defined: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut files = BTreeSet::new();
    for doc in &index.documents {
        let file = normalise(&doc.relative_path);
        files.insert(file.clone());
        for occ in &doc.occurrences {
            if occ.symbol_roles & DEFINITION == 0 {
                continue;
            }
            let Some(name) = display_name(&occ.symbol) else {
                continue;
            };
            definers
                .entry(occ.symbol.as_str())
                .or_default()
                .insert(file.clone());
            let names = defined.entry(file.clone()).or_default();
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    let mut edges: BTreeSet<(String, String, String)> = BTreeSet::new();
    for doc in &index.documents {
        let from = normalise(&doc.relative_path);
        for occ in &doc.occurrences {
            if occ.symbol_roles & DEFINITION != 0 {
                continue;
            }
            let Some(targets) = definers.get(occ.symbol.as_str()) else {
                continue;
            };
            let Some(name) = display_name(&occ.symbol) else {
                continue;
            };
            for to in targets {
                if *to != from {
                    edges.insert((from.clone(), to.clone(), name.clone()));
                }
            }
        }
    }
    ScipGraph {
        edges: edges
            .into_iter()
            .map(|(from, to, via)| DepEdge { from, to, via })
            .collect(),
        defined,
        files,
    }
}

/// Read the index for `workspace`, refusing one that is missing or stale.
pub fn load(workspace: &Path) -> Result<(ScipGraph, ScipMeta), ScipError> {
    let meta = freshness(workspace)?;
    let index = read_index(&index_path(workspace))?;
    Ok((graph_from_index(&index), meta))
}

/// Who uses `target`, from the semantic index, walked up to `max_hops`.
///
/// With a `symbol`, the first hop is narrowed to the files that refer to a
/// symbol of that name defined in `target` — the question the name tier can
/// only approximate — and the walk continues from those files alone.
pub fn semantic_impact(
    graph: &ScipGraph,
    target: &str,
    symbol: Option<&str>,
    max_hops: usize,
) -> ImpactReport {
    let mut reverse: BTreeMap<&str, BTreeMap<&str, BTreeSet<&str>>> = BTreeMap::new();
    for edge in &graph.edges {
        reverse
            .entry(edge.to.as_str())
            .or_default()
            .entry(edge.from.as_str())
            .or_default()
            .insert(edge.via.as_str());
    }
    let mut seen: BTreeSet<String> = BTreeSet::from([target.to_string()]);
    let mut files: Vec<ImpactedFile> = Vec::new();
    let mut frontier: Vec<String> = vec![target.to_string()];
    for hop in 1..=max_hops {
        let mut next = Vec::new();
        for file in &frontier {
            let Some(users) = reverse.get(file.as_str()) else {
                continue;
            };
            for (from, names) in users {
                if hop == 1 {
                    if let Some(symbol) = symbol {
                        if !names.contains(symbol) {
                            continue;
                        }
                    }
                }
                if !seen.insert(from.to_string()) {
                    continue;
                }
                files.push(ImpactedFile {
                    file: from.to_string(),
                    via: names.iter().map(|n| n.to_string()).collect(),
                    hops: hop,
                    uses_symbol: hop == 1 && symbol.is_some(),
                });
                next.push(from.to_string());
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    files.sort_by(|a, b| a.hops.cmp(&b.hops).then_with(|| a.file.cmp(&b.file)));
    let declared = graph.defined.get(target).cloned().unwrap_or_default();
    ImpactReport {
        target: target.to_string(),
        symbol: symbol.map(str::to_string),
        declared_more: declared.len().saturating_sub(DECLARED_CAP),
        declared: declared.into_iter().take(DECLARED_CAP).collect(),
        files,
        indexed_files: graph.files.len(),
        edges: graph.edges.len(),
        tier: ImpactTier::Semantic,
    }
}

/// Resolve `asked` against the index's own file list, by full path or by tail,
/// and answer from a fresh index. The same refusals as the name tier: nothing
/// matches, or more than one file does.
pub fn semantic_impact_for(
    workspace: &Path,
    asked: &str,
    symbol: Option<&str>,
    max_hops: usize,
) -> Result<Result<ImpactReport, crate::ContextError>, ScipError> {
    let (graph, _meta) = load(workspace)?;
    let asked_norm = asked
        .trim()
        .trim_start_matches("./")
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_string();
    let tail = format!("/{asked_norm}");
    let matches: Vec<&String> = graph
        .files
        .iter()
        .filter(|f| **f == asked_norm || f.ends_with(&tail))
        .collect();
    Ok(match matches.as_slice() {
        [] => Err(crate::ContextError::NotIndexed {
            asked: asked.to_string(),
            near: Vec::new(),
            indexed: graph.files.len(),
        }),
        [only] => Ok(semantic_impact(&graph, only, symbol, max_hops)),
        many => Err(crate::ContextError::AmbiguousTarget {
            asked: asked.to_string(),
            matches: many.iter().map(|s| s.to_string()).collect(),
        }),
    })
}

/// How a document counts the characters in its positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Utf8,
    Utf16,
    Utf32,
}

/// One place a symbol is written, exactly where rust-analyzer put it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolSite {
    /// The document, relative to the workspace, `/`-separated.
    pub file: String,
    /// Zero-based line.
    pub line: usize,
    /// Start and end on that line, in `encoding` units.
    pub start: usize,
    pub end: usize,
    pub encoding: Encoding,
    /// The declaration itself, rather than a use of it.
    pub definition: bool,
}

/// One symbol with a given name, every place it is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedSymbol {
    /// The SCIP symbol string, which is what tells two same-named items apart.
    pub symbol: String,
    /// `function`, `type`, `value`, `macro` or `item`, from the symbol's suffix.
    pub kind: &'static str,
    /// Documents that declare it.
    pub defined_in: Vec<String>,
    /// Declarations and uses, in document order.
    pub sites: Vec<SymbolSite>,
}

fn kind_of(symbol: &str) -> &'static str {
    if symbol.ends_with(").") {
        "function"
    } else if symbol.ends_with('#') {
        "type"
    } else if symbol.ends_with('!') {
        "macro"
    } else if symbol.ends_with('.') {
        "value"
    } else {
        "item"
    }
}

fn single_line(occ: &scip::types::Occurrence) -> Option<(usize, usize, usize)> {
    match occ.range.as_slice() {
        [line, start, end] => Some((*line as usize, *start as usize, *end as usize)),
        [line, start, end_line, end] if line == end_line => {
            Some((*line as usize, *start as usize, *end as usize))
        }
        _ if occ.has_single_line_range() => {
            let r = occ.single_line_range();
            Some((
                r.line as usize,
                r.start_character as usize,
                r.end_character as usize,
            ))
        }
        _ => None,
    }
}

fn encoding_of(doc: &scip::types::Document) -> Encoding {
    use scip::types::PositionEncoding;
    match doc.position_encoding.enum_value() {
        Ok(PositionEncoding::UTF16CodeUnitOffsetFromLineStart) => Encoding::Utf16,
        Ok(PositionEncoding::UTF32CodeUnitOffsetFromLineStart) => Encoding::Utf32,
        // rust-analyzer writes UTF-8 offsets; an index that does not say is
        // read the same way, and every site is checked against the text anyway.
        _ => Encoding::Utf8,
    }
}

/// Every symbol named `name` that this workspace declares, with every place it
/// is written (`LSP-3`). Locals are never returned: they belong to one body, and
/// a rename by name should not reach into it. More than one entry means the name
/// is ambiguous — a function and a field can share it — and the caller has to
/// choose; this never chooses.
pub fn symbols_named(workspace: &Path, name: &str) -> Result<Vec<NamedSymbol>, ScipError> {
    freshness(workspace)?;
    let index = read_index(&index_path(workspace))?;
    let mut defined: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for doc in &index.documents {
        for occ in &doc.occurrences {
            if occ.symbol_roles & DEFINITION != 0
                && display_name(&occ.symbol).as_deref() == Some(name)
            {
                defined
                    .entry(occ.symbol.clone())
                    .or_default()
                    .insert(normalise(&doc.relative_path));
            }
        }
    }
    let mut out: Vec<NamedSymbol> = defined
        .into_iter()
        .map(|(symbol, files)| NamedSymbol {
            kind: kind_of(&symbol),
            symbol,
            defined_in: files.into_iter().collect(),
            sites: Vec::new(),
        })
        .collect();
    for doc in &index.documents {
        let file = normalise(&doc.relative_path);
        let encoding = encoding_of(doc);
        for occ in &doc.occurrences {
            let Some(entry) = out.iter_mut().find(|n| n.symbol == occ.symbol) else {
                continue;
            };
            let Some((line, start, end)) = single_line(occ) else {
                continue;
            };
            let site = SymbolSite {
                file: file.clone(),
                line,
                start,
                end,
                encoding,
                definition: occ.symbol_roles & DEFINITION != 0,
            };
            if !entry.sites.contains(&site) {
                entry.sites.push(site);
            }
        }
    }
    Ok(out)
}

/// The byte range `site` covers in `text`, or `None` when the line or the
/// columns do not exist there — which is what a file edited since the index was
/// built looks like.
pub fn byte_range(text: &str, site: &SymbolSite) -> Option<(usize, usize)> {
    let mut line_start = 0usize;
    for _ in 0..site.line {
        line_start += text[line_start..].find('\n')? + 1;
    }
    let line_end = text[line_start..]
        .find('\n')
        .map(|i| line_start + i)
        .unwrap_or(text.len());
    let line = &text[line_start..line_end];
    let to_byte = |units: usize| -> Option<usize> {
        if site.encoding == Encoding::Utf8 {
            return (units <= line.len() && line.is_char_boundary(units)).then_some(units);
        }
        let mut counted = 0usize;
        for (byte, ch) in line.char_indices() {
            if counted == units {
                return Some(byte);
            }
            counted += match site.encoding {
                Encoding::Utf16 => ch.len_utf16(),
                _ => 1,
            };
        }
        (counted == units).then_some(line.len())
    };
    Some((
        line_start + to_byte(site.start)?,
        line_start + to_byte(site.end)?,
    ))
}

/// Which of the items with one name was meant.
#[derive(Debug)]
pub enum Pick {
    One(NamedSymbol),
    /// No item matches; the message says what does exist.
    Nothing(String),
    /// More than one does, and the caller has to say which.
    Several(Vec<NamedSymbol>),
}

/// Narrow `found` to the one item `in_file` declares, by path or by tail, or to
/// the only item there is. Never chooses between two.
pub fn pick(found: Vec<NamedSymbol>, name: &str, in_file: Option<&str>) -> Pick {
    if found.is_empty() {
        return Pick::Nothing(format!(
            "rust-analyzer's index has no declaration of `{name}` in this workspace; a \
             local variable is not looked up by name, and an item from another crate \
             is not indexed here"
        ));
    }
    let candidates: Vec<NamedSymbol> = match in_file {
        Some(asked) => {
            let asked = normalise(asked.trim().trim_start_matches("./"));
            let all: Vec<String> = found.iter().flat_map(|n| n.defined_in.clone()).collect();
            let kept: Vec<NamedSymbol> = found
                .into_iter()
                .filter(|n| {
                    n.defined_in.iter().any(|f| {
                        *f == asked
                            || f.ends_with(&format!("/{asked}"))
                            || asked.ends_with(&format!("/{f}"))
                    })
                })
                .collect();
            if kept.is_empty() {
                return Pick::Nothing(format!(
                    "`{name}` is declared in {}, not in {asked}",
                    all.join(", ")
                ));
            }
            kept
        }
        None => found,
    };
    if candidates.len() == 1 {
        Pick::One(candidates.into_iter().next().expect("one"))
    } else {
        Pick::Several(candidates)
    }
}

/// One place an item is written, as a reader would want it listed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// Workspace-relative, `/`-separated.
    pub file: String,
    /// One-based.
    pub line: usize,
    /// The source line, trimmed.
    pub text: String,
    /// The function the reference is written in, `Type::method` inside an
    /// `impl`; `None` outside every function.
    pub caller: Option<String>,
    pub definition: bool,
}

/// Every place `item` is written, read from the files as they are now (`LSP-1`).
/// A site whose text no longer holds the name is left out rather than reported
/// at a position that is not true any more.
pub fn references(workspace: &Path, item: &NamedSymbol, name: &str) -> Vec<Reference> {
    let mut out = Vec::new();
    let mut texts: BTreeMap<&str, Option<String>> = BTreeMap::new();
    for site in &item.sites {
        let text = texts
            .entry(site.file.as_str())
            .or_insert_with(|| std::fs::read_to_string(workspace.join(&site.file)).ok());
        let Some(text) = text.as_deref() else {
            continue;
        };
        let Some((start, end)) = byte_range(text, site) else {
            continue;
        };
        if text.get(start..end) != Some(name) {
            continue;
        }
        out.push(Reference {
            file: site.file.clone(),
            line: site.line + 1,
            text: text.lines().nth(site.line).unwrap_or("").trim().to_string(),
            caller: if site.definition {
                None
            } else {
                crate::parse::enclosing_function(text, start)
            },
            definition: site.definition,
        });
    }
    out.sort_by(|a, b| {
        b.definition
            .cmp(&a.definition)
            .then_with(|| a.file.cmp(&b.file))
            .then_with(|| a.line.cmp(&b.line))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A three-file crate: `lib.rs` defines `helper` and `Unused`, `user.rs`
    /// calls `helper` through a re-export the name tier cannot follow, and
    /// `bystander.rs` names `Unused` only.
    fn sample_crate(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xencode-scip-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"scipsample\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("src/lib.rs"),
            "mod core_impl;\nmod user;\nmod bystander;\npub use core_impl::*;\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("src/core_impl.rs"),
            "pub fn helper() -> u32 { 7 }\npub struct Unused;\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("src/user.rs"),
            "pub fn twice() -> u32 { crate::helper() * 2 }\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("src/bystander.rs"),
            "pub fn make() -> crate::Unused { crate::Unused }\n",
        )
        .unwrap();
        dir
    }

    fn rust_analyzer_works() -> bool {
        match rust_analyzer_version() {
            Ok(_) => true,
            Err(e) => {
                eprintln!("skipping: {e}");
                false
            }
        }
    }

    #[test]
    fn a_real_index_links_a_caller_through_a_reexport_and_narrows_by_symbol() {
        if !rust_analyzer_works() {
            return;
        }
        let dir = sample_crate("reexport");
        let meta = generate(&dir, SCIP_TIMEOUT).expect("rust-analyzer indexes the sample");
        assert!(
            meta.files.contains(&"src/user.rs".to_string()),
            "{:?}",
            meta.files
        );
        assert!(meta.rust_analyzer.contains("rust-analyzer"));

        let (graph, _) = load(&dir).expect("a fresh index loads");
        let all = semantic_impact(&graph, "src/core_impl.rs", None, 3);
        let direct: Vec<&str> = all
            .files
            .iter()
            .filter(|f| f.hops == 1)
            .map(|f| f.file.as_str())
            .collect();
        // `user.rs` reaches `helper` only through `pub use core_impl::*` in
        // lib.rs, which no `use` path in user.rs names.
        assert!(direct.contains(&"src/user.rs"), "{direct:?}");
        assert!(direct.contains(&"src/bystander.rs"), "{direct:?}");
        let user = all.files.iter().find(|f| f.file == "src/user.rs").unwrap();
        assert_eq!(user.via, vec!["helper".to_string()]);
        assert!(all.declared.contains(&"helper".to_string()));
        assert_eq!(all.tier, ImpactTier::Semantic);
        // The name tier, asked the same question about the same files, does not
        // see that caller: `crate::helper()` is a path in an expression, not a
        // `use`, and it resolves only through the glob re-export.
        let by_name = crate::impact_from_filesystem(&dir, "src/core_impl.rs", None)
            .expect("the name tier answers");
        assert!(
            !by_name.files.iter().any(|f| f.file == "src/user.rs"),
            "the name tier was expected to miss user.rs: {:?}",
            by_name.files
        );

        // Writing `crate::helper` is not a use of anything lib.rs defines: the
        // crate root's module symbol does not make lib.rs a dependency.
        let root = semantic_impact(&graph, "src/lib.rs", None, 3);
        assert!(
            !root.files.iter().any(|f| f.file == "src/user.rs"),
            "a module path linked user.rs to lib.rs: {:?}",
            root.files
        );

        let narrowed = semantic_impact(&graph, "src/core_impl.rs", Some("helper"), 3);
        let direct: Vec<&str> = narrowed
            .files
            .iter()
            .filter(|f| f.hops == 1)
            .map(|f| f.file.as_str())
            .collect();
        assert!(direct.contains(&"src/user.rs"), "{direct:?}");
        assert!(
            !direct.contains(&"src/bystander.rs"),
            "bystander never names helper: {direct:?}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_edit_after_the_build_makes_the_index_stale_and_it_is_not_used() {
        if !rust_analyzer_works() {
            return;
        }
        let dir = sample_crate("stale");
        generate(&dir, SCIP_TIMEOUT).expect("rust-analyzer indexes the sample");
        assert!(freshness(&dir).is_ok());
        // Modification times are whole seconds here; wait past the run's start.
        std::thread::sleep(Duration::from_millis(1100));
        std::fs::write(dir.join("src/user.rs"), "pub fn twice() -> u32 { 4 }\n").unwrap();
        match load(&dir) {
            Err(ScipError::Stale(why)) => assert!(why.contains("src/user.rs"), "{why}"),
            other => panic!("expected a stale index, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_site_maps_to_bytes_in_each_encoding_and_a_vanished_line_maps_to_nothing() {
        let text = "fn a() {}\nlet é = helper();\n";
        let site = |start, end, encoding| SymbolSite {
            file: "x.rs".into(),
            line: 1,
            start,
            end,
            encoding,
            definition: false,
        };
        // `é` is two bytes and one UTF-16 unit, so the two encodings disagree on
        // where `helper` starts.
        let (s, e) = byte_range(text, &site(9, 15, Encoding::Utf8)).unwrap();
        assert_eq!(&text[s..e], "helper");
        let (s, e) = byte_range(text, &site(8, 14, Encoding::Utf16)).unwrap();
        assert_eq!(&text[s..e], "helper");
        let gone = SymbolSite {
            line: 7,
            ..site(0, 1, Encoding::Utf8)
        };
        assert_eq!(byte_range(text, &gone), None);
    }

    #[test]
    fn a_name_shared_by_a_function_and_a_field_is_two_symbols_and_a_local_is_none() {
        if !rust_analyzer_works() {
            return;
        }
        let dir = sample_crate("named");
        std::fs::write(
            dir.join("src/bystander.rs"),
            "pub struct Holder { pub helper: u32 }\n\
             pub fn make() -> u32 { let helper = 2; Holder { helper }.helper + helper }\n",
        )
        .unwrap();
        generate(&dir, SCIP_TIMEOUT).expect("rust-analyzer indexes the sample");
        let found = symbols_named(&dir, "helper").expect("a fresh index answers");
        let kinds: Vec<&str> = found.iter().map(|n| n.kind).collect();
        assert_eq!(
            found.len(),
            2,
            "the function and the field, not the local: {found:?}"
        );
        assert!(
            kinds.contains(&"function") && kinds.contains(&"value"),
            "{kinds:?}"
        );
        let function = found.iter().find(|n| n.kind == "function").unwrap();
        assert_eq!(function.defined_in, vec!["src/core_impl.rs".to_string()]);
        // Declared once, used in user.rs — through the glob re-export.
        assert!(function
            .sites
            .iter()
            .any(|s| s.definition && s.file == "src/core_impl.rs"));
        assert!(function
            .sites
            .iter()
            .any(|s| !s.definition && s.file == "src/user.rs"));
        for site in &function.sites {
            let text = std::fs::read_to_string(dir.join(&site.file)).unwrap();
            let (s, e) = byte_range(&text, site).expect("every site is on disk");
            assert_eq!(&text[s..e], "helper", "{site:?}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn references_name_the_function_each_use_is_written_in() {
        if !rust_analyzer_works() {
            return;
        }
        let dir = sample_crate("refs");
        std::fs::write(
            dir.join("src/user.rs"),
            "pub struct W;\n\
             impl W {\n    pub fn go(&self) -> u32 {\n        crate::helper()\n    }\n}\n\
             pub fn twice() -> u32 {\n    crate::helper() * 2\n}\n",
        )
        .unwrap();
        generate(&dir, SCIP_TIMEOUT).expect("rust-analyzer indexes the sample");
        let found = symbols_named(&dir, "helper").unwrap();
        let Pick::One(item) = pick(found, "helper", None) else {
            panic!("one item is called helper here");
        };
        let refs = references(&dir, &item, "helper");
        assert!(refs[0].definition, "the declaration comes first: {refs:?}");
        assert_eq!(refs[0].file, "src/core_impl.rs");
        let callers: BTreeSet<String> = refs.iter().filter_map(|r| r.caller.clone()).collect();
        assert_eq!(
            callers,
            BTreeSet::from(["W::go".to_string(), "twice".to_string()]),
            "{refs:?}"
        );
        let in_go = refs
            .iter()
            .find(|r| r.caller.as_deref() == Some("W::go"))
            .unwrap();
        assert_eq!((in_go.file.as_str(), in_go.line), ("src/user.rs", 4));
        assert_eq!(in_go.text, "crate::helper()");
        assert!(matches!(
            pick(
                symbols_named(&dir, "helper").unwrap(),
                "helper",
                Some("src/user.rs")
            ),
            Pick::Nothing(_)
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A repository can ship its own `.xencode/scip/`, so an index that names a
    /// file outside the workspace is refused whole — in its record or in the
    /// index itself — rather than read from.
    #[test]
    fn an_index_that_points_outside_the_workspace_is_not_used() {
        let dir = std::env::temp_dir().join(format!("xencode-scip-outside-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/lib.rs"), "pub fn helper() {}\n").unwrap();
        std::fs::create_dir_all(scip_dir(&dir)).unwrap();
        let write_meta = |files: Vec<&str>| {
            let meta = ScipMeta {
                started_unix: now_unix() + 100_000,
                head: None,
                rust_analyzer: "crafted".into(),
                seconds: 0.0,
                files: files.into_iter().map(str::to_string).collect(),
            };
            std::fs::write(meta_path(&dir), serde_json::to_string(&meta).unwrap()).unwrap();
        };
        let write_index = |path: &str| {
            let mut doc = scip::types::Document::new();
            doc.relative_path = path.to_string();
            let mut index = scip::types::Index::new();
            index.documents.push(doc);
            std::fs::write(index_path(&dir), index.write_to_bytes().unwrap()).unwrap();
        };

        write_index("src/lib.rs");
        write_meta(vec!["../../outside.rs"]);
        assert!(
            matches!(freshness(&dir), Err(ScipError::Unreadable(_))),
            "a record pointing out"
        );

        write_meta(vec!["src/lib.rs"]);
        write_index("../../outside.rs");
        assert!(
            matches!(symbols_named(&dir, "helper"), Err(ScipError::Unreadable(_))),
            "an index pointing out"
        );
        assert!(!stays_inside("C:/Windows/win.ini") && !stays_inside("/etc/passwd"));
        assert!(stays_inside("crates/x/src/lib.rs"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_index_is_missing_rather_than_empty() {
        let dir = std::env::temp_dir().join(format!("xencode-scip-none-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        assert!(matches!(load(&dir), Err(ScipError::Missing(_))));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_local_symbol_has_no_name_and_a_global_one_has_its_last_descriptor() {
        assert_eq!(display_name("local 12"), None);
        assert_eq!(display_name(""), None);
        assert_eq!(
            display_name("rust-analyzer cargo scipsample 0.1.0 crate/"),
            None
        );
        assert_eq!(
            display_name("rust-analyzer cargo scipsample 0.1.0 core_impl/"),
            None
        );
        assert_eq!(
            display_name("rust-analyzer cargo scipsample 0.1.0 core_impl/helper()."),
            Some("helper".to_string())
        );
    }
}
