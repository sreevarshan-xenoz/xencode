//! Deterministic repository scanner (Pass 1 of the context engine).
//!
//! Walks a workspace without any LLM involvement:
//!   - respects `.gitignore` when a git filter set is supplied (`git ls-files`)
//!   - never reads secret files (listed only)
//!   - never LOC-counts or reads binary files (listed only)
//!   - detects language per file and flags "important" files
//!
//! Cancellation is cooperative: the walk checks the cancel flag every 128
//! files so `/init abort` terminates promptly.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Relative path to the context engine's own directory (always excluded).
pub const XENCODE_DIR_NAME: &str = ".xencode";

/// Source/important directories that are hidden but must still be indexed.
const HIDDEN_DIR_ALLOWLIST: &[&str] = &[".github"];

/// Directories that are always skipped on top of `.gitignore`.
const EXCLUDED_DIRS: &[&str] = &[
    ".git",
    ".xencode",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".next",
    "out",
    ".gradle",
    "htmlcov",
];

/// Filenames/patterns that are treated as secrets: recorded, never read.
fn is_secret_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if let Some(_example) = lower.strip_suffix(".example") {
        return false; // .env.example and friends carry no real secrets
    }
    if lower == ".env" || lower.starts_with(".env.") {
        return true;
    }
    let mut parts = name.rsplitn(2, '.');
    let ext = parts.next().unwrap_or("");
    matches!(
        ext,
        "key" | "pem" | "p12" | "pfx" | "jks" | "cer" | "crt" | "kdbx"
    ) || lower.contains("secret")
        || lower.contains("credential")
        || lower == "id_rsa"
        || lower == "id_ed25519"
}

/// Files that are worth surfacing in the project map.
fn is_important_file(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper.starts_with("README")
        || upper == "AGENTS.MD"
        || upper == "LICENSE"
        || upper.starts_with("LICENSE.")
        || upper == "MAKEFILE"
        || upper.starts_with("DOCKERFILE")
        || upper == "DOCKER-COMPOSE.YML"
        || upper == "DOCKER-COMPOSE.YAML"
        || upper == "CARGO.TOML"
        || upper == "PACKAGE.JSON"
        || upper == "PYPROJECT.TOML"
        || upper == "SETUP.PY"
        || upper == "REQUIREMENTS.TXT"
        || upper == "GO.MOD"
        || upper == "POM.XML"
        || upper == "GRADLE.BUILD"
        || upper == "BUILD.GRADLE"
        || upper == "PUBSPEC.YAML"
        || upper == "TSCONFIG.JSON"
        || upper == ".GITIGNORE"
        || upper == ".GITHUB"
}

/// Programming/markup language identifiers stored in `files.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    C,
    Cpp,
    Csharp,
    Ruby,
    Php,
    Shell,
    Html,
    Css,
    Json,
    Yaml,
    Markdown,
    Toml,
    Sql,
    Proto,
    Docker,
    Make,
    Config,
    Text,
    Other,
}

impl Language {
    /// Stable lowercase identifier used in the index and summaries.
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
            Language::TypeScript => "typescript",
            Language::JavaScript => "javascript",
            Language::Go => "go",
            Language::Java => "java",
            Language::C => "c",
            Language::Cpp => "cpp",
            Language::Csharp => "csharp",
            Language::Ruby => "ruby",
            Language::Php => "php",
            Language::Shell => "shell",
            Language::Html => "html",
            Language::Css => "css",
            Language::Json => "json",
            Language::Yaml => "yaml",
            Language::Markdown => "markdown",
            Language::Toml => "toml",
            Language::Sql => "sql",
            Language::Proto => "proto",
            Language::Docker => "dockerfile",
            Language::Make => "makefile",
            Language::Config => "config",
            Language::Text => "text",
            Language::Other => "other",
        }
    }
}

/// Map a file extension (without the leading dot, lowercased) to a language.
pub fn language_for_extension(ext: &str) -> Language {
    match ext {
        "rs" => Language::Rust,
        "py" => Language::Python,
        "ts" | "tsx" | "mts" | "cts" => Language::TypeScript,
        "js" | "mjs" | "cjs" | "jsx" => Language::JavaScript,
        "go" => Language::Go,
        "java" | "kt" | "kts" => Language::Java,
        "c" | "h" => Language::C,
        "cc" | "cpp" | "cxx" | "hpp" | "hh" => Language::Cpp,
        "cs" => Language::Csharp,
        "rb" => Language::Ruby,
        "php" => Language::Php,
        "sh" | "bash" | "zsh" | "fish" | "ps1" => Language::Shell,
        "html" | "htm" | "vue" => Language::Html,
        "css" | "scss" | "sass" | "less" => Language::Css,
        "json" | "jsonc" | "webmanifest" => Language::Json,
        "yml" | "yaml" => Language::Yaml,
        "md" | "mdx" => Language::Markdown,
        "toml" => Language::Toml,
        "sql" => Language::Sql,
        "proto" => Language::Proto,
        "dockerfile" => Language::Docker,
        "mk" | "make" => Language::Make,
        "ini" | "cfg" | "conf" | "editorconfig" => Language::Config,
        "txt" | "log" => Language::Text,
        _ => Language::Other,
    }
}

/// Filename-based language detection for extension-less important files.
pub fn language_for_filename(name: &str) -> Option<Language> {
    let upper = name.to_ascii_uppercase();
    if upper.starts_with("DOCKERFILE") {
        return Some(Language::Docker);
    }
    if upper == "MAKEFILE" || upper == "GNUMakefile" || upper == "GOFILE" {
        return Some(Language::Make);
    }
    if upper.starts_with("README") || upper == "AGENTS.md" {
        return Some(Language::Markdown);
    }
    if upper == ".GITHUB" {
        return Some(Language::Config);
    }
    None
}

/// Detect the language for a path (filename first, then extension).
pub fn detect_language(path: &Path) -> Language {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if let Some(lang) = language_for_filename(&name) {
        return lang;
    }
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    language_for_extension(&ext)
}

/// Count lines, skipping blanks and the cheap common comment prefixes.
/// Best-effort; never reads more of the file than needed for line counting.
pub fn count_loc(path: &Path) -> u64 {
    let Ok(bytes) = fs::read(path) else {
        return 0;
    };
    let mut loc: u64 = 0;
    for line in bytes.split(|&b| b == b'\n') {
        let trimmed = trim_ascii(line);
        if trimmed.is_empty() {
            continue;
        }
        let first = trimmed[0] as char;
        if first == '#' || first == ';' {
            continue;
        }
        if trimmed.starts_with(b"//") || trimmed.starts_with(b"/*") || trimmed.starts_with(b"*") {
            continue;
        }
        if trimmed.starts_with(b"--") {
            continue;
        }
        loc += 1;
    }
    loc
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|&b| b != b' ' && b != b'\t' && b != b'\r')
        .unwrap_or(bytes.len());
    &bytes[start.min(bytes.len())..]
}

/// Detect binary content by sniffing the first block for a NUL byte.
pub fn is_binary(path: &Path) -> bool {
    let Ok(mut f) = fs::File::open(path) else {
        return true;
    };
    let mut buf = [0u8; 1024];
    let n = f.read(&mut buf).unwrap_or(0);
    buf[..n].contains(&0)
}

/// A single scanned file, ready to serialize into `files.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanEntry {
    /// Repo-relative path using `/` separators.
    pub path: String,
    pub language: Language,
    pub size: u64,
    pub loc: u64,
    pub ext: String,
    pub important: bool,
    /// Secret-detected: listed but never read (loc stays 0).
    pub is_secret: bool,
    /// Binary-detected: listed but loc stays 0.
    pub is_binary: bool,
}

#[derive(Debug, Clone)]
pub struct ScanOutcome {
    pub files: Vec<ScanEntry>,
    pub languages: BTreeMap<String, u64>,
    pub total_loc: u64,
    pub secret_files: Vec<String>,
    pub binary_files: Vec<String>,
    pub skipped: u64,
}

#[derive(Debug)]
pub struct ScanOptions {
    /// Optional set of repo-relative paths (with `/` separators) that git
    /// considers part of the project (tracked + untracked non-ignored).
    /// When present, any file outside the set is treated as ignored.
    pub git_filter: Option<std::collections::HashSet<String>>,
    /// Cooperative cancellation flag checked during the walk.
    pub cancel: std::sync::Arc<AtomicBool>,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            git_filter: None,
            cancel: std::sync::Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Recursively scan `root` and return enriched entries for every file.
pub fn scan_tree(
    root: &Path,
    options: &ScanOptions,
) -> Result<ScanOutcome, StopScan> {
    if !root.exists() {
        return Err(StopScan("workspace root does not exist".to_string()));
    }
    if !root.is_dir() {
        return Err(StopScan(
            "workspace root is not a directory".to_string(),
        ));
    }

    let mut outcome = ScanOutcome {
        files: Vec::new(),
        languages: BTreeMap::new(),
        total_loc: 0,
        secret_files: Vec::new(),
        binary_files: Vec::new(),
        skipped: 0,
    };
    let mut walked = 0u64;

    walk_dir(root, root, options, &mut outcome, &mut walked)?;
    outcome.files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(outcome)
}

/// Error carrying a human-readable reason; a cancelled scan uses this type.
#[derive(Debug)]
pub struct StopScan(pub String);

impl std::fmt::Display for StopScan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn walk_dir(
    root: &Path,
    current: &Path,
    options: &ScanOptions,
    outcome: &mut ScanOutcome,
    walked: &mut u64,
) -> Result<(), StopScan> {
    let read_dir = fs::read_dir(current).map_err(|e| {
        StopScan(format!("cannot list {}: {}", current.display(), e))
    })?;

    for child in read_dir {
        *walked += 1;
        if (*walked).is_multiple_of(128) && options.cancel.load(Ordering::Relaxed) {
            return Err(StopScan("scan cancelled".to_string()));
        }

        let child = child.map_err(|e| {
            StopScan(format!(
                "cannot read entry in {}: {}",
                current.display(),
                e
            ))
        })?;
        let path = child.path();
        let name = child.file_name().to_string_lossy().to_string();

        let metadata = match fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(_) => {
                outcome.skipped += 1;
                continue;
            }
        };

        if metadata.file_type().is_symlink() {
            continue; // avoid cycles
        }

        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");

        if metadata.is_dir() {
            if should_skip_dir(&name) {
                continue;
            }
            walk_dir(root, &path, options, outcome, walked)?;
            continue;
        }

        if !metadata.is_file() {
            continue;
        }

        // Git ignore filtering: skip files git would not include.
        if let Some(filter) = &options.git_filter {
            if !filter.contains(&relative) {
                outcome.skipped += 1;
                continue;
            }
        }

        let is_secret = is_secret_file(&name);
        let binary = !is_secret && is_binary(&path);

        let entry = ScanEntry {
            ext: path
                .extension()
                .map(|e| e.to_string_lossy().to_string())
                .unwrap_or_default(),
            path: relative.clone(),
            language: detect_language(&path),
            size: metadata.len(),
            loc: if is_secret || binary { 0 } else { count_loc(&path) },
            important: is_important_file(&name),
            is_secret,
            is_binary: binary,
        };

        *outcome.languages
            .entry(entry.language.as_str().to_string())
            .or_insert(0) += 1;
        outcome.total_loc += entry.loc;
        if is_secret {
            outcome.secret_files.push(relative.clone());
        }
        if binary {
            outcome.binary_files.push(relative.clone());
        }
        outcome.files.push(entry);
    }

    Ok(())
}

fn should_skip_dir(name: &str) -> bool {
    if EXCLUDED_DIRS.contains(&name) {
        return true;
    }
    (name.starts_with('.') && !HIDDEN_DIR_ALLOWLIST.contains(&name))
        || name == "node_modules"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_workspace() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("xencode-scan-test-{stamp}"))
    }

    #[test]
    fn detects_language_by_extension_and_filename() {
        assert_eq!(detect_language(Path::new("m.rs")), Language::Rust);
        assert_eq!(detect_language(Path::new("m.py")), Language::Python);
        assert_eq!(detect_language(Path::new("m.tsx")), Language::TypeScript);
        assert_eq!(detect_language(Path::new("m.yaml")), Language::Yaml);
        assert_eq!(detect_language(Path::new("Dockerfile")), Language::Docker);
        assert_eq!(detect_language(Path::new("Makefile")), Language::Make);
        assert_eq!(detect_language(Path::new("README.md")), Language::Markdown);
        assert_eq!(detect_language(Path::new("a.bin")), Language::Other);
    }

    #[test]
    fn counts_loc_skipping_blanks_and_comment_prefixes() {
        let root = temp_workspace();
        fs::create_dir_all(&root).unwrap();
        let file = root.join("sample.rs");
        fs::write(
            &file,
            "// header\n\nfn main() {\n    // inner\n    let x = 1;\n    let y = 2;\n    /* block */\n}\n",
        )
        .unwrap();
        // Best-effort comment skipping: // # ; /* and leading * lines excluded.
        assert_eq!(count_loc(&file), 4);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn flags_secrets_and_never_presumes_read() {
        assert!(is_secret_file(".env"));
        assert!(is_secret_file(".env.local"));
        assert!(is_secret_file("credentials.json"));
        assert!(is_secret_file("id_rsa"));
        assert!(is_secret_file("server.key"));
        assert!(is_secret_file("api.pem"));
        assert!(!is_secret_file(".env.example"));
        assert!(!is_secret_file("main.rs"));
    }

    #[test]
    fn tracks_important_files() {
        assert!(is_important_file("README.md"));
        assert!(is_important_file("AGENTS.md"));
        assert!(is_important_file("Cargo.toml"));
        assert!(is_important_file("Dockerfile"));
        assert!(!is_important_file("main.rs"));
    }

    #[test]
    fn skips_excluded_dirs_and_flags_binary() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("target")).unwrap();
        fs::create_dir_all(root.join("node_modules")).unwrap();
        fs::create_dir_all(root.join(".github/workflows")).unwrap();
        File::create(root.join("src/main.rs")).unwrap();
        File::create(root.join("src/lib.rs")).unwrap();
        File::create(root.join("target/out")).unwrap();
        fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        File::create(root.join("node_modules/pkg/index.js")).unwrap();
        fs::write(root.join("blob.bin"), vec![0u8, 1, 2, 3, 0]).unwrap();
        fs::write(root.join(".env"), "TOKEN=abc").unwrap();
        File::create(root.join(".github/workflows/ci.yml")).unwrap();

        let outcome = scan_tree(&root, &ScanOptions::default()).unwrap();
        let paths: Vec<String> = outcome.files.iter().map(|e| e.path.clone()).collect();

        assert!(paths.contains(&"src/main.rs".to_string()));
        assert!(paths.contains(&"src/lib.rs".to_string()));
        assert!(paths.contains(&".github/workflows/ci.yml".to_string()));
        assert!(!paths.iter().any(|p| p.contains("target")));
        assert!(!paths.iter().any(|p| p.contains("node_modules")));

        let env = outcome.files.iter().find(|e| e.path == ".env").unwrap();
        assert!(env.is_secret);
        assert_eq!(env.loc, 0);

        let bin = outcome.files.iter().find(|e| e.path == "blob.bin").unwrap();
        assert!(bin.is_binary);
        assert_eq!(bin.loc, 0);

        assert_eq!(outcome.secret_files, vec![".env".to_string()]);
        assert_eq!(outcome.binary_files, vec!["blob.bin".to_string()]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn honours_git_filter_set() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        File::create(root.join("src/main.rs")).unwrap();
        File::create(root.join("ignored.tmp")).unwrap();

        let mut allow = std::collections::HashSet::new();
        allow.insert("src/main.rs".to_string());
        let opts = ScanOptions {
            git_filter: Some(allow),
            cancel: std::sync::Arc::new(AtomicBool::new(false)),
        };

        let outcome = scan_tree(&root, &opts).unwrap();
        assert_eq!(outcome.files.len(), 1);
        assert_eq!(outcome.files[0].path, "src/main.rs");
        assert_eq!(outcome.skipped, 1);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cooperative_cancellation_stops_walk() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        for i in 0..300usize {
            File::create(root.join("src").join(format!("f{i}.rs"))).unwrap();
        }
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let opts = ScanOptions {
            cancel: cancel.clone(),
            ..Default::default()
        };
        cancel.store(true, Ordering::Relaxed);
        let err = scan_tree(&root, &opts).unwrap_err();
        assert_eq!(err.0, "scan cancelled");
        fs::remove_dir_all(root).unwrap();
    }
}