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
use crate::scanner::{scan_tree, ScanOptions};
use crate::symbols::{build_graph, extract_rust_symbols, DepEdge, PerFileSymbols};

/// Directory name of the context engine inside a project.
pub const XENCODE_DIR: &str = ".xencode";

#[derive(Debug)]
pub enum ContextError {
    Io {
        path: PathBuf,
        source: std::io::Error,
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
    let current_head = current_git_info(&root).map(|g| g.head);

    let prior_files =
        crate::index::read_json::<FilesIndex>(&crate::index::file_index_path(&xencode));
    let prior_symbols = crate::index::read_json::<BTreeMap<String, PerFileSymbols>>(
        &crate::index::symbols_json_path(&xencode),
    );
    let prior_deps =
        crate::index::read_json::<Vec<DepEdge>>(&crate::index::deps_json_path(&xencode));
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
        let head_short = truncate(&info.head, 8);
        emit(
            &mut progress,
            &format!(
                "log:🎋 {} @ {} — {} file(s) dirty",
                info.branch,
                if head_short.is_empty() {
                    "(unborn HEAD)".to_string()
                } else {
                    head_short
                },
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
        .filter(|e| e.ext == "rs" && !e.is_secret && !e.is_binary)
        .map(|e| e.path.clone())
        .collect();
    let symbols = extract_repo_symbols(&root, &rust_files)?;
    let graph = build_graph(&rust_files, &symbols);
    if !rust_files.is_empty() {
        let symbol_count: usize = symbols
            .values()
            .map(|s| s.structs.len() + s.functions.len() + s.imports.len() + s.exports.len())
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
        git_head: git.as_ref().and_then(|g| {
            if g.head.is_empty() {
                None
            } else {
                Some(g.head.clone())
            }
        }),
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

fn file_mtime_millis(path: &Path) -> Option<u64> {
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

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        s[..n].to_string()
    }
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
