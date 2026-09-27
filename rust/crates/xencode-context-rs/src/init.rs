//! `/init` orchestration — the deterministic structural passes (M0 + M1).
//!
//! [`init_project`] runs the structural passes of the context engine:
//!   1. scaffold `.xencode/` (+ auto-gitignore)
//!   2. git snapshot (branch/HEAD/dirty + ignored-file set)
//!   3. full repository scan
//!   4. language/size analytics
//!   5. symbol extraction + dependency graph (`symbols.json`, `deps.json`)
//!   6. atomic writes of `files.json` and `manifest.json`
//!
//! Re-running against an unchanged workspace returns a `fresh` summary
//! without rewriting anything (mtime + git-HEAD based resume).

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::gitinfo::{current_git_info, git_file_set, is_git_repo, GitInfo};
use crate::index::{write_atomic, FileEntry, FilesIndex, Manifest};
use crate::scanner::{language_for_extension, scan_tree, ScanOptions};
use crate::symbols::{build_graph, extract_rust_symbols, DepEdge, PerFileSymbols};

/// Directory name of the context engine inside a project.
pub const XENCODE_DIR: &str = ".xencode";

#[derive(Debug)]
pub enum ContextError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A snapshot read found no usable `.xencode` index at `path`.
    NoIndex(PathBuf),
    /// The index has no file matching what was asked for. `near` are paths it
    /// does hold that share the name, because a wrong directory is the common
    /// mistake and `not found` alone would not help.
    NotIndexed {
        asked: String,
        near: Vec<String>,
        indexed: usize,
    },
    /// A path given by its tail matches more than one indexed file, so the
    /// caller has to say which one.
    AmbiguousTarget {
        asked: String,
        matches: Vec<String>,
    },
    Scan(String),
    Cancelled,
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextError::Io { path, source } => {
                write!(f, "{}: {}", path.display(), source)
            }
            ContextError::NoIndex(path) => write!(
                f,
                "no project index in {} — start the TUI and run /init first",
                path.display()
            ),
            ContextError::NotIndexed {
                asked,
                near,
                indexed,
            } => {
                write!(f, "nothing in the project index is `{asked}`")?;
                if near.is_empty() {
                    write!(f, " (the index covers {indexed} files)")
                } else {
                    write!(f, " — it does hold: {}", near.join(", "))
                }
            }
            ContextError::AmbiguousTarget { asked, matches } => write!(
                f,
                "`{asked}` matches more than one indexed file: {} — name the one you \
                 mean by its full path",
                matches.join(", ")
            ),
            ContextError::Scan(msg) => f.write_str(msg),
            ContextError::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for ContextError {}

impl From<crate::scanner::StopScan> for ContextError {
    fn from(value: crate::scanner::StopScan) -> Self {
        ContextError::Scan(value.0)
    }
}

impl From<std::io::Error> for ContextError {
    fn from(source: std::io::Error) -> Self {
        ContextError::Scan(source.to_string())
    }
}

#[derive(Debug, Default, Clone)]
pub struct InitSummary {
    /// `true` when the existing index was already fresh (nothing rewritten).
    pub fresh: bool,
    pub files_scanned: u64,
    pub total_loc: u64,
    /// (language, file count), sorted by count descending.
    pub languages: Vec<(String, u64)>,
    pub secret_files: Vec<String>,
    pub binary_files: Vec<String>,
    /// Files outside the git filter set (ignored by .gitignore or excluded).
    pub skipped: u64,
    /// Combined byte size of the four index files on disk.
    pub index_bytes: u64,
    /// Rust files that yielded at least one extracted symbol.
    pub symbol_files: u64,
    /// Resolved file→file dependency edges written to `deps.json`.
    pub dep_edges: u64,
    /// Present only when the workspace is a git repository.
    pub git: Option<GitInfo>,
}

/// Run the structural `/init` pass.
///
/// `progress` receives tagged lines: `phase_start:<n>`, `phase_done:<n>`,
/// `log:<msg>`. `cancel` is checked cooperatively between phases and during
/// the scan; setting it aborts with [`ContextError::Cancelled`].
pub fn init_project(
    root: &Path,
    cancel: std::sync::Arc<AtomicBool>,
    mut progress: impl FnMut(&str) + Send,
) -> Result<InitSummary, ContextError> {
    check_abort(&cancel)?;

    let root = root.canonicalize().map_err(|source| ContextError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    if !root.is_dir() {
        return Err(ContextError::Scan(format!(
            "workspace root is not a directory: {}",
            root.display()
        )));
    }

    let xencode = root.join(XENCODE_DIR);

    // ── Phase 1: scaffold ──────────────────────────────────────────────────
    emit(&mut progress, "phase_start:Create .xencode directory");
    for sub in ["", "index", "summaries", "cache/cmd", "cache/transcript"] {
        fs_create_dir_all(&xencode.join(sub))?;
    }
    auto_gitignore_index_dir(&root)?;
    emit(&mut progress, "phase_done:Create .xencode directory");
    check_abort(&cancel)?;

    // ── Phase 2: resume check ──────────────────────────────────────────────
    emit(&mut progress, "phase_start:Resume check");
    let prior_manifest =
        crate::index::read_json::<Manifest>(&crate::index::manifest_path(&xencode));
    let current_head = current_git_info(&root).and_then(|g| g.revision().map(str::to_string));

    let prior_files =
        crate::index::read_json::<FilesIndex>(&crate::index::file_index_path(&xencode));
    let prior_symbols = crate::index::read_json::<BTreeMap<String, PerFileSymbols>>(
        &crate::index::symbols_json_path(&xencode),
    );
    let prior_deps =
        crate::index::read_json::<Vec<DepEdge>>(&crate::index::deps_json_path(&xencode));
    // Kept aside because the resume check below consumes the manifest: phase 6b
    // needs to know which commit the history on disk was mined at.
    let indexed_head = prior_manifest.as_ref().and_then(|m| m.git_head.clone());
    if let (Some(m), Some(f), Some(s), Some(d)) =
        (prior_manifest, prior_files, prior_symbols, prior_deps)
    {
        if m.version == crate::index::VERSION
            && m.git_head == current_head
            && manifest_mtimes_fresh(&root, &m, &f)
        {
            emit(
                &mut progress,
                "log:✓ Index is up to date — no changes since last run.",
            );
            emit(&mut progress, "phase_done:Resume check");
            return Ok(summary_from_existing(
                &f,
                m.skipped,
                s.len() as u64,
                d.len() as u64,
            ));
        }
    }
    emit(&mut progress, "phase_done:Resume check");
    check_abort(&cancel)?;

    // ── Phase 3: git snapshot ──────────────────────────────────────────────
    emit(&mut progress, "phase_start:Git snapshot");
    let git = current_git_info(&root);
    let git_filter = git_file_set(&root);
    if let Some(info) = &git {
        emit(
            &mut progress,
            &format!(
                "log:🎋 {} @ {} — {} file(s) dirty",
                info.branch,
                info.revision_label(),
                info.dirty
            ),
        );
    } else {
        emit(
            &mut progress,
            "log:ℹ️ Not a git repository — using exclusion-based scan.",
        );
    }
    emit(&mut progress, "phase_done:Git snapshot");
    check_abort(&cancel)?;

    // ── Phase 4: scan ──────────────────────────────────────────────────────
    emit(&mut progress, "phase_start:Scan repository");
    let scan = scan_tree(
        &root,
        &ScanOptions {
            cancel: cancel.clone(),
            git_filter,
        },
    )?;
    let files: Vec<FileEntry> = scan
        .files
        .iter()
        .map(|e| FileEntry {
            path: e.path.clone(),
            language: e.language.as_str().to_string(),
            size: e.size,
            loc: e.loc,
            ext: e.ext.clone(),
            important: e.important,
            secret: e.is_secret,
            binary: e.is_binary,
        })
        .collect();
    emit(
        &mut progress,
        &format!("log:📁 Scanned {} files", files.len()),
    );
    if scan.skipped > 0 {
        emit(
            &mut progress,
            &format!("log:🚫 Skipped {} ignored/excluded files", scan.skipped),
        );
    }
    emit(&mut progress, "phase_done:Scan repository");
    check_abort(&cancel)?;

    // ── Phase 5: analytics ─────────────────────────────────────────────────
    emit(&mut progress, "phase_start:Analyze languages & sizes");
    let mut languages: Vec<(String, u64)> = scan.languages.into_iter().collect();
    languages.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    for (lang, count) in &languages {
        emit(&mut progress, &format!("log:  ▸ {lang}: {count} file(s)"));
    }
    if !scan.secret_files.is_empty() {
        emit(
            &mut progress,
            &format!(
                "log:🔒 {} secret-detected file(s) listed (not read)",
                scan.secret_files.len()
            ),
        );
    }
    if !scan.binary_files.is_empty() {
        emit(
            &mut progress,
            &format!(
                "log:🧊 {} binary file(s) listed (not read)",
                scan.binary_files.len()
            ),
        );
    }
    emit(&mut progress, "phase_done:Analyze languages & sizes");
    check_abort(&cancel)?;

    // ── Phase 6: symbols & dependency graph ───────────────────────────────
    emit(&mut progress, "phase_start:Extract symbols & dependencies");
    let rust_files: Vec<String> = scan
        .files
        .iter()
        .filter(|e| {
            language_for_extension(&e.ext).has_semantic_tier() && !e.is_secret && !e.is_binary
        })
        .map(|e| e.path.clone())
        .collect();
    let symbols = extract_repo_symbols(&root, &rust_files)?;
    let graph = build_graph(&rust_files, &symbols);
    if !rust_files.is_empty() {
        let symbol_count: usize = symbols
            .values()
            .map(|s| {
                s.structs.len()
                    + s.enums.len()
                    + s.traits.len()
                    + s.impls.len()
                    + s.types.len()
                    + s.functions.len()
                    + s.imports.len()
                    + s.exports.len()
                    + s.mods.len()
            })
            .sum();
        emit(
            &mut progress,
            &format!(
                "log:🧩 {} Rust file(s) → {} symbol(s), {} dependency edge(s)",
                rust_files.len(),
                symbol_count,
                graph.len()
            ),
        );
    } else {
        emit(
            &mut progress,
            "log:🧩 No Rust files — symbols/deps skipped.",
        );
    }
    write_atomic(&crate::index::symbols_json_path(&xencode), &symbols)?;
    write_atomic(&crate::index::deps_json_path(&xencode), &graph)?;
    emit(&mut progress, "phase_done:Extract symbols & dependencies");
    check_abort(&cancel)?;

    // ── Phase 6b: commit history ─────────────────────────────────────────
    // One `git log` over the whole history, which is the expensive part of
    // co-change retrieval and therefore happens here rather than per turn. A
    // rebuild that has not moved HEAD reuses what is already on disk.
    emit(&mut progress, "phase_start:Mine commit history");
    let prior_history = crate::cochange::load_history(&xencode);
    let history = match prior_history {
        ref prior if !prior.is_empty() && indexed_head == current_head => {
            emit(
                &mut progress,
                "log:🕯 commit history reused — HEAD has not moved.",
            );
            prior.clone()
        }
        _ => {
            let started = std::time::Instant::now();
            let mined = crate::cochange::mine_commit_history(&root).unwrap_or_default();
            let hubs = mined.values().filter(|h| h.hub).count();
            emit(
                &mut progress,
                &format!(
                    "log:🕯 {} file(s) with history, {} too often edited to pair, read in {} ms",
                    mined.len(),
                    hubs,
                    started.elapsed().as_millis()
                ),
            );
            mined
        }
    };
    crate::cochange::save_history(&xencode, &history)?;
    emit(&mut progress, "phase_done:Mine commit history");
    check_abort(&cancel)?;

    // ── Phase 7: write files index + manifest ─────────────────────────────
    emit(&mut progress, "phase_start:Write index files");
    let indexed_at = now_millis();
    let files_index = FilesIndex {
        version: crate::index::VERSION,
        generated_at: indexed_at,
        files: files.clone(),
    };
    let mut mtime_map = std::collections::BTreeMap::new();
    for entry in &scan.files {
        let mtime = file_mtime_millis(&root.join(&entry.path)).unwrap_or(0);
        mtime_map.insert(entry.path.clone(), mtime);
    }
    let manifest = Manifest {
        version: crate::index::VERSION,
        git_head: git.as_ref().and_then(|g| g.revision().map(str::to_string)),
        branch: git.as_ref().map(|g| g.branch.clone()),
        dirty: git.as_ref().map(|g| g.dirty).unwrap_or(0),
        skipped: scan.skipped,
        mtime_map,
        indexed_at,
    };

    write_atomic(&crate::index::file_index_path(&xencode), &files_index)?;
    write_atomic(&crate::index::manifest_path(&xencode), &manifest)?;
    let index_bytes = index_on_disk_bytes(&xencode);
    emit(
        &mut progress,
        &format!(
            "log:🗂  files.json + manifest.json + symbols.json + deps.json written ({index_bytes} bytes)"
        ),
    );
    emit(&mut progress, "phase_done:Write index files");

    Ok(InitSummary {
        fresh: false,
        files_scanned: files.len() as u64,
        total_loc: scan.total_loc,
        languages,
        secret_files: scan.secret_files.clone(),
        binary_files: scan.binary_files.clone(),
        skipped: scan.skipped,
        index_bytes,
        symbol_files: symbols.len() as u64,
        dep_edges: graph.len() as u64,
        git,
    })
}

/// Read every Rust file and extract its symbol inventory. Secret and binary
/// files were already filtered out upstream; unreadable files are skipped.
fn extract_repo_symbols(
    root: &Path,
    rust_files: &[String],
) -> Result<BTreeMap<String, PerFileSymbols>, ContextError> {
    let mut symbols = BTreeMap::new();
    for path in rust_files {
        let full = root.join(path);
        let content = std::fs::read_to_string(&full).map_err(|source| ContextError::Io {
            path: full.clone(),
            source,
        })?;
        symbols.insert(path.clone(), extract_rust_symbols(&content));
    }
    Ok(symbols)
}

/// Rebuild a `fresh` summary straight from an existing untouched index.
fn summary_from_existing(
    files: &FilesIndex,
    skipped: u64,
    symbol_files: u64,
    dep_edges: u64,
) -> InitSummary {
    let mut languages: std::collections::BTreeMap<String, u64> = Default::default();
    let mut total_loc = 0u64;
    for f in &files.files {
        *languages.entry(f.language.clone()).or_insert(0) += 1;
        total_loc += f.loc;
    }
    let mut languages: Vec<(String, u64)> = languages.into_iter().collect();
    languages.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    InitSummary {
        fresh: true,
        files_scanned: files.files.len() as u64,
        total_loc,
        languages,
        secret_files: Vec::new(),
        binary_files: Vec::new(),
        skipped,
        index_bytes: 0,
        symbol_files,
        dep_edges,
        git: None,
    }
}

fn emit(progress: &mut impl FnMut(&str), line: &str) {
    progress(line);
}

fn check_abort(cancel: &AtomicBool) -> Result<(), ContextError> {
    if cancel.load(Ordering::Relaxed) {
        Err(ContextError::Cancelled)
    } else {
        Ok(())
    }
}

/// Fresh when every file's current mtime matches the manifest snapshot.
fn manifest_mtimes_fresh(root: &Path, manifest: &Manifest, files: &FilesIndex) -> bool {
    if manifest.mtime_map.is_empty() || files.files.is_empty() {
        return false;
    }
    if manifest.mtime_map.len() != files.files.len() {
        return false;
    }
    for entry in &files.files {
        match manifest.mtime_map.get(&entry.path) {
            None => return false,
            Some(expected) => {
                let mtime = file_mtime_millis(&root.join(&entry.path)).unwrap_or(0);
                if mtime != *expected {
                    return false;
                }
            }
        }
    }
    true
}

pub(crate) fn file_mtime_millis(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn fs_create_dir_all(path: &Path) -> Result<(), ContextError> {
    std::fs::create_dir_all(path).map_err(|source| ContextError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Ensure `.xencode/` is gitignored so generated artifacts never get committed.
fn auto_gitignore_index_dir(root: &Path) -> Result<(), ContextError> {
    if !is_git_repo(root) {
        return Ok(());
    }
    let ignore_file = root.join(".gitignore");
    let existing = std::fs::read_to_string(&ignore_file).unwrap_or_default();
    let wanted = format!("{}/", XENCODE_DIR);
    let mut lines: Vec<&str> = existing.lines().collect();
    if lines
        .iter()
        .any(|l| l.trim().trim_end_matches('/') == wanted.trim().trim_end_matches('/'))
    {
        return Ok(());
    }
    lines.push(wanted.trim_end());
    let merged = format!("{}\n", lines.join("\n"));
    std::fs::write(&ignore_file, merged).map_err(|source| ContextError::Io {
        path: ignore_file.clone(),
        source,
    })
}

fn index_on_disk_bytes(xencode: &Path) -> u64 {
    let mut total = 0;
    for path in [
        crate::index::file_index_path(xencode),
        crate::index::manifest_path(xencode),
        crate::index::symbols_json_path(xencode),
        crate::index::deps_json_path(xencode),
    ] {
        total += std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Arc;

    fn temp_workspace() -> PathBuf {
        // A process-wide counter, not a timestamp. Tests in one binary run in
        // parallel threads and can read the same nanosecond, which had them share
        // a directory and clobber each other's assertions.
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, AtomicOrdering::Relaxed)
        );
        std::env::temp_dir().join(format!("xencode-init-test-{unique}"))
    }

    fn run(root: &Path) -> (Result<InitSummary, ContextError>, Vec<String>) {
        let mut lines = Vec::new();
        let result = init_project(root, Arc::new(AtomicBool::new(false)), |line| {
            lines.push(line.to_string())
        });
        (result, lines)
    }

    #[test]
    fn scaffolds_index_and_returns_summary() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
        File::create(root.join("Cargo.toml")).unwrap();

        let (result, lines) = run(&root);
        let summary = result.expect("init should succeed");

        assert!(root.join(XENCODE_DIR).is_dir());
        assert!(root.join(XENCODE_DIR).join("index/files.json").is_file());
        assert!(root.join(XENCODE_DIR).join("index/manifest.json").is_file());
        assert!(root.join(XENCODE_DIR).join("index/symbols.json").is_file());
        assert!(root.join(XENCODE_DIR).join("index/deps.json").is_file());
        assert!(root.join(XENCODE_DIR).join("index/history.json").is_file());
        assert!(root.join(XENCODE_DIR).join("summaries").is_dir());
        assert!(root.join(XENCODE_DIR).join("cache/cmd").is_dir());

        assert!(!summary.fresh);
        assert_eq!(summary.files_scanned, 2);
        assert!(summary.languages.contains(&("rust".to_string(), 1)));
        assert_eq!(summary.symbol_files, 1);
        assert_eq!(summary.dep_edges, 0);

        assert_eq!(lines[0], "phase_start:Create .xencode directory");
        assert!(lines
            .iter()
            .any(|l| l == "phase_done:Extract symbols & dependencies"));
        assert!(lines.iter().any(|l| l == "phase_done:Write index files"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn history_is_mined_once_and_reused_until_the_head_moves() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/a.rs"), "fn a() {}\n").unwrap();
        fs::write(root.join("src/b.rs"), "fn b() {}\n").unwrap();
        let git = |args: &[&str]| {
            let ran = std::process::Command::new("git")
                .current_dir(&root)
                .args(args)
                .output()
                .unwrap();
            assert!(
                ran.status.success(),
                "{args:?}: {}",
                String::from_utf8_lossy(&ran.stderr)
            );
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "test@example.invalid"]);
        git(&["config", "user.name", "Test"]);
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "both files in one commit"]);

        let xencode = root.join(XENCODE_DIR);
        let (result, lines) = run(&root);
        assert!(!result.expect("init should succeed").fresh);
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("log:🕯 ") && l.contains("file(s) with history")),
            "the first run reports what it mined: {lines:?}"
        );
        let history = crate::cochange::load_history(&xencode);
        assert_eq!(
            history["src/a.rs"].partners[0].0, "src/b.rs",
            "two files committed together are partners on disk"
        );

        // An edit forces a rebuild, but HEAD has not moved, so the log is not
        // re-read.
        fs::write(root.join("src/a.rs"), "fn a() {}\nfn c() {}\n").unwrap();
        let (result, lines) = run(&root);
        assert!(!result.expect("init should succeed").fresh);
        assert!(
            lines.iter().any(|l| l.contains("commit history reused")),
            "a rebuild at the same commit keeps the history it already has: {lines:?}"
        );

        // A new commit does change what the log holds, so it is re-read.
        fs::write(root.join("src/b.rs"), "fn b() {}\nfn d() {}\n").unwrap();
        git(&["commit", "-qam", "second commit"]);
        fs::write(root.join("src/a.rs"), "fn a() {}\nfn c() {}\nfn e() {}\n").unwrap();
        let (result, lines) = run(&root);
        assert!(!result.expect("init should succeed").fresh);
        assert!(
            !lines.iter().any(|l| l.contains("commit history reused")),
            "a rebuild past the commit it was mined at must not reuse it: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("log:🕯 ") && l.contains("file(s) with history")),
            "and it must re-mine: {lines:?}"
        );
        let history = crate::cochange::load_history(&xencode);
        assert_eq!(
            history["src/b.rs"].commits, 2,
            "the newer commit is in the history now on disk"
        );

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn second_run_is_fresh_and_rewrites_nothing() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "fn f() {}\n").unwrap();

        let (first, _) = run(&root);
        assert!(!first.expect("ok").fresh);

        let before = fs::metadata(root.join(XENCODE_DIR).join("index/files.json"))
            .unwrap()
            .modified()
            .unwrap();

        let (second, lines) = run(&root);
        let summary = second.expect("ok");
        assert!(summary.fresh);
        assert!(lines.iter().any(|l| l.contains("up to date")));

        let after = fs::metadata(root.join(XENCODE_DIR).join("index/files.json"))
            .unwrap()
            .modified()
            .unwrap();
        assert!(after <= before, "fresh run must not rewrite the index");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn edit_invalidates_resume() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn a() {}\n").unwrap();

        let (first, _) = run(&root);
        assert!(!first.expect("ok").fresh);

        // Fresh until we touch a file, then not fresh.
        fs::write(root.join("src/main.rs"), "fn b() {}\n").unwrap();
        let (second, _) = run(&root);
        let summary = second.expect("ok");
        assert!(!summary.fresh);
        assert_eq!(summary.files_scanned, 1);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fresh_resume_reports_persisted_skipped_count() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join(".gitignore"), "ignored.tmp\n").unwrap();
        fs::write(root.join("src/main.rs"), "fn a() {}\n").unwrap();
        fs::write(root.join("new.py"), "print(1)\n").unwrap();
        fs::write(root.join("ignored.tmp"), "x\n").unwrap();

        let git_ok = std::process::Command::new("git")
            .args(["--version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !git_ok {
            fs::remove_dir_all(root).unwrap();
            return;
        }
        std::process::Command::new("git")
            .args(["-C", root.to_string_lossy().as_ref(), "init", "-q"])
            .status()
            .unwrap();
        std::process::Command::new("git")
            .args([
                "-C",
                root.to_string_lossy().as_ref(),
                "config",
                "user.email",
                "test@xencode.local",
            ])
            .status()
            .unwrap();
        std::process::Command::new("git")
            .args([
                "-C",
                root.to_string_lossy().as_ref(),
                "config",
                "user.name",
                "Xencode Test",
            ])
            .status()
            .unwrap();

        let (first, _) = run(&root);
        let first = first.expect("ok");
        assert!(!first.fresh);
        // .gitignore + src/main.rs + untracked-not-ignored new.py are all indexed;
        // the gitignored ignored.tmp is filtered out and counted as skipped.
        assert_eq!(first.files_scanned, 3);
        assert_eq!(
            first.skipped, 1,
            "gitignored file must be counted as skipped"
        );

        let (second, _) = run(&root);
        let second = second.expect("ok");
        assert!(second.fresh);
        assert_eq!(
            second.skipped, first.skipped,
            "fresh resume must report the skipped count from the manifest, not the file count"
        );
        assert_ne!(second.skipped, second.files_scanned);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn abort_before_scan_returns_cancelled() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        File::create(root.join("src/main.rs")).unwrap();

        let cancel = Arc::new(AtomicBool::new(true));
        let mut lines = Vec::new();
        let result = init_project(&root, cancel, |l| lines.push(l.to_string()));

        assert!(matches!(result, Err(ContextError::Cancelled)));
        assert!(!root.join(XENCODE_DIR).exists(), "abort before any phase");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn auto_gitignores_xencode_dir() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        File::create(root.join("src/main.rs")).unwrap();
        File::create(root.join(".gitignore")).unwrap();

        // Simulate a git repo by writing a `.git` directory marker only; the
        // helper checks `git rev-parse` so we instead force the write path via
        // a real git init when available. Skip when git is unavailable.
        let git_ok = std::process::Command::new("git")
            .args(["--version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !git_ok {
            fs::remove_dir_all(root).unwrap();
            return;
        }

        let status = std::process::Command::new("git")
            .args(["-C", root.to_string_lossy().as_ref(), "init", "-q"])
            .status()
            .unwrap();
        assert!(status.success());

        let _ = run(&root);

        let ignore = fs::read_to_string(root.join(".gitignore")).unwrap();
        assert!(
            ignore.contains(".xencode/"),
            "expected .xencode gitignored, got: {ignore}"
        );

        // Second run must not duplicate the entry.
        let _ = run(&root);
        let ignore = fs::read_to_string(root.join(".gitignore")).unwrap();
        assert_eq!(
            ignore.matches(".xencode/").count(),
            1,
            "auto-gitignore must be idempotent"
        );

        fs::remove_dir_all(root).unwrap();
    }
}
