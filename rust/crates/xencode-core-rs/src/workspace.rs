//! Workspace scanning and metadata types.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Options for workspace scanning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanOptions {
    pub max_depth: Option<usize>,
    pub include_hidden: bool,
    pub respect_gitignore: bool,
    pub follow_symlinks: bool,
    pub max_file_size: u64,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            max_depth: Some(10),
            include_hidden: false,
            respect_gitignore: true,
            follow_symlinks: false,
            max_file_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// Metadata about a scanned file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: PathBuf,
    pub relative_path: PathBuf,
    pub file_name: String,
    pub extension: String,
    pub size: u64,
    pub is_binary: bool,
    pub language: String,
}

/// Metadata about a scanned workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceInfo {
    pub root: PathBuf,
    pub files: Vec<FileEntry>,
    pub total_size: u64,
    pub file_count: usize,
    pub dir_count: usize,
    pub languages: Vec<String>,
}

impl WorkspaceInfo {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            files: Vec::new(),
            total_size: 0,
            file_count: 0,
            dir_count: 0,
            languages: Vec::new(),
        }
    }
}

/// Detect language from file extension.
pub fn detect_language(extension: &str) -> String {
    match extension {
        "rs" => "rust".to_string(),
        "py" => "python".to_string(),
        "js" => "javascript".to_string(),
        "ts" | "tsx" => "typescript".to_string(),
        "jsx" => "react".to_string(),
        "go" => "go".to_string(),
        "java" => "java".to_string(),
        "rb" => "ruby".to_string(),
        "cpp" | "cc" | "cxx" => "cpp".to_string(),
        "c" => "c".to_string(),
        "h" | "hpp" => "header".to_string(),
        "rs" => "rust".to_string(),
        "toml" => "toml".to_string(),
        "yaml" | "yml" => "yaml".to_string(),
        "json" => "json".to_string(),
        "md" | "markdown" => "markdown".to_string(),
        "html" => "html".to_string(),
        "css" => "css".to_string(),
        "scss" | "sass" => "scss".to_string(),
        "sql" => "sql".to_string(),
        "sh" | "bash" => "shell".to_string(),
        "ps1" => "powershell".to_string(),
        "dockerfile" => "dockerfile".to_string(),
        "txt" => "text".to_string(),
        _ => "unknown".to_string(),
    }
}

/// Check if a file is binary by scanning null bytes.
pub fn is_binary(content: &[u8]) -> bool {
    content[..content.len().min(1024)].contains(&0x00)
}

/// Scan a workspace directory recursively.
pub fn scan_workspace(root: &Path, options: &ScanOptions) -> Result<WorkspaceInfo, std::io::Error> {
    let mut info = WorkspaceInfo::new(root.to_path_buf());
    scan_directory(root, root, options, 0, &mut info)?;
    info.languages.sort();
    info.languages.dedup();
    Ok(info)
}

fn scan_directory(
    root: &Path,
    dir: &Path,
    options: &ScanOptions,
    depth: usize,
    info: &mut WorkspaceInfo,
) -> Result<(), std::io::Error> {
    if let Some(max_depth) = options.max_depth {
        if depth > max_depth {
            return Ok(());
        }
    }

    if !dir.is_dir() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip hidden files unless explicitly included
        if !options.include_hidden {
            if let Some(name) = path.file_name() {
                if name.to_string_lossy().starts_with('.') {
                    continue;
                }
            }
        }

        if path.is_dir() {
            info.dir_count += 1;
            scan_directory(root, &path, options, depth + 1, info)?;
        } else if path.is_file() {
            let metadata = std::fs::metadata(&path)?;
            let file_size = metadata.len();

            if file_size > options.max_file_size {
                continue;
            }

            let relative = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            let extension = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();

            let file_info = FileEntry {
                path: path.clone(),
                relative_path: relative.clone(),
                file_name: path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
                extension: extension.clone(),
                size: file_size,
                is_binary: false, // full scan would check actual content
                language: detect_language(&extension),
            };

            if !info.languages.contains(&file_info.language) {
                info.languages.push(file_info.language.clone());
            }

            info.total_size += file_size;
            info.file_count += 1;
            info.files.push(file_info);
        }
    }

    Ok(())
}
