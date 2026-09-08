use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryKind {
    Directory,
    File,
    Symlink,
    Other,
}

impl fmt::Display for EntryKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntryKind::Directory => formatter.write_str("directory"),
            EntryKind::File => formatter.write_str("file"),
            EntryKind::Symlink => formatter.write_str("symlink"),
            EntryKind::Other => formatter.write_str("other"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceEntry {
    pub path: PathBuf,
    pub kind: EntryKind,
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanOptions {
    pub max_depth: Option<usize>,
    pub include_hidden: bool,
    pub excluded_dirs: Vec<String>,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            max_depth: None,
            include_hidden: false,
            excluded_dirs: vec![
                ".git".to_string(),
                ".mypy_cache".to_string(),
                ".pytest_cache".to_string(),
                ".ruff_cache".to_string(),
                ".venv".to_string(),
                "__pycache__".to_string(),
                "htmlcov".to_string(),
                "node_modules".to_string(),
                "target".to_string(),
                "venv".to_string(),
            ],
        }
    }
}

#[derive(Debug)]
pub enum WorkspaceScanError {
    RootNotFound(PathBuf),
    RootIsNotDirectory(PathBuf),
    Io { path: PathBuf, source: io::Error },
}

impl fmt::Display for WorkspaceScanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkspaceScanError::RootNotFound(path) => {
                write!(
                    formatter,
                    "workspace root does not exist: {}",
                    path.display()
                )
            }
            WorkspaceScanError::RootIsNotDirectory(path) => {
                write!(
                    formatter,
                    "workspace root is not a directory: {}",
                    path.display()
                )
            }
            WorkspaceScanError::Io { path, source } => {
                write!(formatter, "failed to scan {}: {}", path.display(), source)
            }
        }
    }
}

impl std::error::Error for WorkspaceScanError {}

pub fn scan_workspace(
    root: impl AsRef<Path>,
    options: &ScanOptions,
) -> Result<Vec<WorkspaceEntry>, WorkspaceScanError> {
    let root = root.as_ref();
    if !root.exists() {
        return Err(WorkspaceScanError::RootNotFound(root.to_path_buf()));
    }
    if !root.is_dir() {
        return Err(WorkspaceScanError::RootIsNotDirectory(root.to_path_buf()));
    }

    let mut entries = Vec::new();
    scan_dir(root, root, 0, options, &mut entries)?;
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(entries)
}

fn scan_dir(
    root: &Path,
    current: &Path,
    depth: usize,
    options: &ScanOptions,
    entries: &mut Vec<WorkspaceEntry>,
) -> Result<(), WorkspaceScanError> {
    if let Some(max_depth) = options.max_depth {
        if depth > max_depth {
            return Ok(());
        }
    }

    let read_dir = fs::read_dir(current).map_err(|source| WorkspaceScanError::Io {
        path: current.to_path_buf(),
        source,
    })?;

    for child in read_dir {
        let child = child.map_err(|source| WorkspaceScanError::Io {
            path: current.to_path_buf(),
            source,
        })?;
        let path = child.path();
        let name = child.file_name();
        let name = name.to_string_lossy();

        if should_skip(&name, options) {
            continue;
        }

        let metadata = fs::symlink_metadata(&path).map_err(|source| WorkspaceScanError::Io {
            path: path.clone(),
            source,
        })?;
        let file_type = metadata.file_type();
        let kind = if file_type.is_symlink() {
            EntryKind::Symlink
        } else if file_type.is_dir() {
            EntryKind::Directory
        } else if file_type.is_file() {
            EntryKind::File
        } else {
            EntryKind::Other
        };

        let relative_path = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
        entries.push(WorkspaceEntry {
            path: relative_path,
            kind: kind.clone(),
            bytes: if kind == EntryKind::File {
                Some(metadata.len())
            } else {
                None
            },
        });

        if kind == EntryKind::Directory {
            scan_dir(root, &path, depth + 1, options, entries)?;
        }
    }

    Ok(())
}

fn should_skip(name: &str, options: &ScanOptions) -> bool {
    if !options.include_hidden && name.starts_with('.') {
        return true;
    }

    options
        .excluded_dirs
        .iter()
        .any(|excluded| excluded == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_workspace() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("xencode-core-rs-test-{stamp}"))
    }

    #[test]
    fn scans_workspace_in_stable_order() {
        let root = temp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        File::create(root.join("src").join("main.rs")).unwrap();
        File::create(root.join("README.md")).unwrap();

        let entries = scan_workspace(&root, &ScanOptions::default()).unwrap();
        let paths: Vec<String> = entries
            .iter()
            .map(|entry| entry.path.to_string_lossy().replace('\\', "/"))
            .collect();

        assert_eq!(paths, vec!["README.md", "src", "src/main.rs"]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn skips_hidden_and_excluded_directories_by_default() {
        let root = temp_workspace();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("target")).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        File::create(root.join(".env")).unwrap();
        File::create(root.join("src").join("lib.rs")).unwrap();
        File::create(root.join("target").join("artifact")).unwrap();

        let entries = scan_workspace(&root, &ScanOptions::default()).unwrap();
        let paths: Vec<String> = entries
            .iter()
            .map(|entry| entry.path.to_string_lossy().replace('\\', "/"))
            .collect();

        assert_eq!(paths, vec!["src", "src/lib.rs"]);

        fs::remove_dir_all(root).unwrap();
    }
}
