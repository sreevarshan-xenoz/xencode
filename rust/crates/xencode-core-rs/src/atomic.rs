//! Atomic, owner-only writes for state files.
//!
//! Every file this helper writes is private state (config with API keys, cache
//! entries, transcripts, session and task state), so the temp file is created
//! with mode `0600` *before* it holds any data. Setting the mode after the
//! rename would leave a window where the new contents are world-readable.

use std::io;
use std::io::Write;
use std::path::Path;

/// Write `bytes` to `path` so readers never see a half-written file.
///
/// The data goes to a fresh temp file in the same directory (mode `0600` on
/// creation), is flushed to disk, and is then renamed over `path`. The parent
/// directory of `path` is created if missing, and its own entry is synced so
/// the rename survives a crash. An existing file's mode is replaced by `0600`.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;

    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    tmp.as_file_mut().write_all(bytes)?;
    tmp.as_file_mut().sync_all()?;
    tmp.persist(path).map_err(|e| e.error)?;
    sync_dir(parent);
    Ok(())
}

/// The rename is only durable once the directory entry itself is on disk.
/// Not possible on Windows, where a directory cannot be opened as a file; the
/// rename still happened, only a crash may lose it.
#[cfg(unix)]
fn sync_dir(dir: &Path) {
    if let Ok(file) = std::fs::File::open(dir) {
        let _ = file.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) {}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn mode_of(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn writes_file_and_creates_missing_parent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("state.json");
        write_atomic(&path, b"{\"a\":1}").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"a\":1}");
    }

    #[test]
    fn result_is_owner_readable_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        write_atomic(&path, b"secret").unwrap();
        assert_eq!(mode_of(&path), 0o600);
    }

    #[test]
    fn tightens_a_world_readable_file_on_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, b"old").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(mode_of(&path), 0o644);

        write_atomic(&path, b"new").unwrap();

        assert_eq!(mode_of(&path), 0o600);
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    #[test]
    fn leaves_no_temp_files_behind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json");
        write_atomic(&path, b"one").unwrap();
        write_atomic(&path, b"two").unwrap();
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path() != path)
            .collect();
        assert!(leftovers.is_empty(), "temp files left: {leftovers:?}");
    }

    #[test]
    fn failed_write_leaves_the_existing_file_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json");
        write_atomic(&path, b"good").unwrap();
        // A path whose parent is a file cannot be written through; the error
        // must arrive with the original contents still in place.
        let a_file = dir.path().join("notadir");
        std::fs::write(&a_file, b"i am a file").unwrap();
        let bad = a_file.join("child.json");
        assert!(write_atomic(&bad, b"nope").is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"good".as_slice());
    }
}
