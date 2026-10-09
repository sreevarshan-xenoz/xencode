//! Where a project's engine listens (EN-2): one named pipe or socket file per
//! project folder, named from a fingerprint of the folder's path.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Where an engine listens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Address {
    /// A Windows named pipe, `\\.\pipe\…`.
    Pipe(String),
    /// A Unix socket file.
    Socket(PathBuf),
}

/// The project folder as one string, the same however it was written:
/// resolved, with forward slashes, without a trailing slash, and on Windows
/// without the `\\?\` prefix and in lower case (its file names ignore case).
fn canonical(project: &Path) -> String {
    let resolved = std::fs::canonicalize(project).unwrap_or_else(|_| project.to_path_buf());
    let text = resolved
        .to_string_lossy()
        .trim_start_matches(r"\\?\")
        .replace('\\', "/");
    let text = text.trim_end_matches('/').to_string();
    if cfg!(windows) {
        text.to_lowercase()
    } else {
        text
    }
}

/// The first 16 hex characters of the SHA-256 of the canonical folder path.
pub fn fingerprint(project: &Path) -> String {
    let digest = Sha256::digest(canonical(project).as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// The user name, reduced to characters a pipe name can hold.
fn user() -> String {
    let name = std::env::var("USERNAME")
        .or_else(|_| std::env::var("USER"))
        .unwrap_or_else(|_| "user".to_string());
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect()
}

/// `<state dir>/engine`, where the lock and (on Unix) the socket live.
fn engine_dir() -> Result<PathBuf, String> {
    xencode_config_rs::paths::state_dir()
        .map(|dir| dir.join("engine"))
        .map_err(|e| e.to_string())
}

impl Address {
    /// Where the engine for `project` listens.
    pub fn for_project(project: &Path) -> Result<Address, String> {
        let fp = fingerprint(project);
        if cfg!(windows) {
            Ok(Address::Pipe(format!(
                r"\\.\pipe\xencode-engine-{}-{fp}",
                user()
            )))
        } else {
            Ok(Address::Socket(engine_dir()?.join(format!("{fp}.sock"))))
        }
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Address::Pipe(name) => f.write_str(name),
            Address::Socket(path) => write!(f, "{}", path.display()),
        }
    }
}

/// The lock file only the running engine for `project` holds.
pub fn lock_path(project: &Path) -> Result<PathBuf, String> {
    Ok(engine_dir()?.join(format!("{}.lock", fingerprint(project))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_folder_written_two_ways_has_one_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let plain = dir.path().to_path_buf();
        let messy = std::path::PathBuf::from(format!("{}/./", plain.display()));
        assert_eq!(fingerprint(&plain), fingerprint(&messy));
        assert_eq!(fingerprint(&plain).len(), 16);
        let other = tempfile::tempdir().unwrap();
        assert_ne!(fingerprint(&plain), fingerprint(other.path()));
    }

    #[test]
    fn the_address_names_the_user_and_the_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let fp = fingerprint(dir.path());
        match Address::for_project(dir.path()).unwrap() {
            Address::Pipe(name) => {
                assert!(name.starts_with(r"\\.\pipe\xencode-engine-"), "{name}");
                assert!(name.ends_with(&fp), "{name}");
            }
            Address::Socket(path) => {
                assert!(path.ends_with(format!("{fp}.sock")), "{}", path.display());
                assert!(path.parent().unwrap().ends_with("engine"));
            }
        }
        let lock = lock_path(dir.path()).unwrap();
        assert!(lock.ends_with(format!("{fp}.lock")), "{}", lock.display());
    }
}
