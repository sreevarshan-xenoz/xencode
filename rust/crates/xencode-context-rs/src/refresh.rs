//! Incremental snapshot refresh (Milestone F, F1-01).
//!
//! `/init` writes the full symbol/dep snapshot; this keeps it current when a
//! single already-indexed Rust file changes on disk — the watcher's normal
//! diet. The graph is rebuilt wholesale from the updated symbol map:
//! `build_graph` is cheap and deterministic, and piecemeal edge surgery could
//! silently drift from what `/init` would produce.
//!
//! A path missing from the snapshot is a deliberate no-op: brand-new files
//! (and everything non-Rust) still need `/init` to enter the index.

use std::collections::BTreeMap;
use std::path::Path;

use crate::index::{
    deps_json_path, file_index_path, manifest_path, read_json, symbols_json_path, write_atomic,
    FilesIndex, Manifest,
};
use crate::init::{ContextError, XENCODE_DIR};
use crate::scanner::count_loc;
use crate::symbols::{build_graph, extract_rust_symbols, DepEdge, PerFileSymbols};

/// Outcome of [`refresh_rust_file`]: either the snapshot changed and carries
/// the new graph, or there was nothing to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefreshOutcome {
    /// Symbols/deps/index/manifest were rewritten; `graph` is the new edge set.
    Updated(Vec<DepEdge>),
    /// No snapshot yet, unknown path, non-Rust or not an indexed file.
    NoOp,
}

/// Refresh the `.xencode` snapshot of `root` after one Rust file changed on
/// disk. `rel_path` may be repo-relative (the watcher's format) or absolute
/// inside `root`. Removal (file no longer readable) drops the index entry,
/// symbol record and mtime alongside the edges.
pub fn refresh_rust_file(
    root: &Path,
    rel_path: &str,
) -> Result<RefreshOutcome, ContextError> {
    let root = root.canonicalize().map_err(|source| ContextError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    let xencode = root.join(XENCODE_DIR);
    let rel = match Path::new(rel_path).strip_prefix(&root) {
        Ok(r) => r.to_string_lossy().into_owned(),
        Err(_) => rel_path.trim_start_matches("./").to_string(),
    };

    let Some(mut index): Option<FilesIndex> = read_json(&file_index_path(&xencode)) else {
        return Ok(RefreshOutcome::NoOp);
    };
    let indexed = index
        .files
        .iter()
        .any(|e| e.path == rel && e.language == "rust");
    if !indexed {
        return Ok(RefreshOutcome::NoOp);
    }

    let mut symbols: BTreeMap<String, PerFileSymbols> =
        read_json(&symbols_json_path(&xencode)).unwrap_or_default();
    // Only touch the manifest if one exists — never invent a snapshot /init
    // did not write.
    let mut manifest: Option<Manifest> = read_json(&manifest_path(&xencode));

    let full = root.join(&rel);
    match std::fs::read_to_string(&full) {
        Ok(content) => {
            symbols.insert(rel.clone(), extract_rust_symbols(&content));
            if let Some(entry) = index.files.iter_mut().find(|e| e.path == rel) {
                entry.size = content.len() as u64;
                entry.loc = count_loc(&full);
            }
            let mtime = crate::init::file_mtime_millis(&full).unwrap_or(0);
            if let Some(m) = manifest.as_mut() {
                m.mtime_map.insert(rel.clone(), mtime);
            }
        }
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            symbols.remove(&rel);
            index.files.retain(|e| e.path != rel);
            if let Some(m) = manifest.as_mut() {
                m.mtime_map.remove(&rel);
            }
        }
        Err(source) => return Err(ContextError::Io { path: full, source }),
    }

    let rust_files: Vec<String> = index
        .files
        .iter()
        .filter(|e| e.language == "rust")
        .map(|e| e.path.clone())
        .collect();
    let graph = build_graph(&rust_files, &symbols);

    write_atomic(&file_index_path(&xencode), &index)?;
    write_atomic(&symbols_json_path(&xencode), &symbols)?;
    write_atomic(&deps_json_path(&xencode), &graph)?;
    if let Some(m) = manifest {
        write_atomic(&manifest_path(&xencode), &m)?;
    }
    Ok(RefreshOutcome::Updated(graph))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::init_project;
    use std::fs::{self, File};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
    use std::sync::Arc;

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_workspace() -> PathBuf {
        let unique = (std::process::id() as u64) * 1_000_000
            + NEXT.fetch_add(1, AtomicOrdering::Relaxed);
        let dir = std::env::temp_dir().join(format!("xencode-refresh-test-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn init(root: &Path) {
        init_project(root, Arc::new(AtomicBool::new(false)), |_| {}).expect("init");
    }

    fn deps(root: &Path) -> Vec<DepEdge> {
        read_json(&deps_json_path(&root.join(XENCODE_DIR))).expect("deps.json")
    }

    fn snap_file(root: &Path, name: &str) -> PathBuf {
        root.join(XENCODE_DIR).join("index").join(name)
    }

    fn setup() -> PathBuf {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "mod a;\nmod b;\n").unwrap();
        fs::write(
            root.join("src/a.rs"),
            "use crate::b::bee;\npub fn ay() { bee(); }\n",
        )
        .unwrap();
        fs::write(root.join("src/b.rs"), "pub fn bee() {}\n").unwrap();
        File::create(root.join("Cargo.toml")).unwrap();
        init(&root);
        assert!(deps(&root)
            .iter()
            .any(|e| e.from == "src/a.rs" && e.to == "src/b.rs"));
        root
    }

    #[test]
    fn edit_updates_symbols_and_edges_on_disk() {
        let root = setup();
        // Drop the import: the a→b edge must vanish from the returned graph
        // and from deps.json.
        fs::write(
            root.join("src/a.rs"),
            "pub fn ay() { crate::other(); }\n",
        )
        .unwrap();
        let out = refresh_rust_file(&root, "src/a.rs").unwrap();
        let RefreshOutcome::Updated(graph) = out else {
            panic!("expected Updated, got {out:?}");
        };
        assert!(!graph.iter().any(|e| e.from == "src/a.rs"));
        assert_eq!(graph, deps(&root));
        let index: FilesIndex = read_json(&snap_file(&root, "files.json")).unwrap();
        let entry = index.files.iter().find(|e| e.path == "src/a.rs").unwrap();
        assert!(entry.loc > 0);
    }

    #[test]
    fn removal_drops_record_entry_and_edges() {
        let root = setup();
        fs::remove_file(root.join("src/b.rs")).unwrap();
        let out = refresh_rust_file(&root, "src/b.rs").unwrap();
        assert!(matches!(out, RefreshOutcome::Updated(_)));
        let symbols: BTreeMap<String, PerFileSymbols> =
            read_json(&snap_file(&root, "symbols.json")).unwrap();
        assert!(!symbols.contains_key("src/b.rs"));
        let index: FilesIndex = read_json(&snap_file(&root, "files.json")).unwrap();
        assert!(!index.files.iter().any(|e| e.path == "src/b.rs"));
        assert!(!deps(&root).iter().any(|e| e.to == "src/b.rs"));
        // Manifest stayed consistent with the index (freshness check compares
        // their lengths).
        let manifest: Manifest = read_json(&snap_file(&root, "manifest.json")).unwrap();
        assert_eq!(manifest.mtime_map.len(), index.files.len());
        assert!(!manifest.mtime_map.contains_key("src/b.rs"));
    }

    #[test]
    fn unknown_or_non_rust_paths_are_no_ops() {
        let root = setup();
        let before = fs::read(snap_file(&root, "deps.json")).unwrap();

        // A path that exists but was never indexed (created after /init).
        fs::write(root.join("src/new.rs"), "pub fn n() {}\n").unwrap();
        assert_eq!(
            refresh_rust_file(&root, "src/new.rs").unwrap(),
            RefreshOutcome::NoOp
        );
        // Non-Rust and truly unknown paths, too.
        assert_eq!(
            refresh_rust_file(&root, "Cargo.toml").unwrap(),
            RefreshOutcome::NoOp
        );
        assert_eq!(
            refresh_rust_file(&root, "src/gone.rs").unwrap(),
            RefreshOutcome::NoOp
        );
        assert_eq!(before, fs::read(snap_file(&root, "deps.json")).unwrap());
    }

    #[test]
    fn absolute_paths_and_refresh_keep_init_fresh() {
        let root = setup();
        fs::write(root.join("src/b.rs"), "pub fn bee() {}\nfn extra() {}\n").unwrap();
        let abs = format!("{}/src/b.rs", root.display());
        let out = refresh_rust_file(&root, &abs).unwrap();
        assert!(matches!(out, RefreshOutcome::Updated(_)));
        // The manifest mtime was bumped to the edited file's, so the next
        // /init considers the snapshot fresh — refresh is init-consistent.
        let summary = init_project(&root, Arc::new(AtomicBool::new(false)), |_| {})
            .expect("second init");
        assert!(summary.fresh, "init should see the snapshot as fresh");
    }
}
