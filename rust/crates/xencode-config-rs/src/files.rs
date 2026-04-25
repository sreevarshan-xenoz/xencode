use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

/// Errors that can occur during file operations.
#[derive(Debug)]
pub enum FileError {
    Io(io::Error),
    NotFound(String),
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileError::Io(source) => write!(f, "file I/O error: {source}"),
            FileError::NotFound(path) => write!(f, "file not found: {path}"),
        }
    }
}

impl std::error::Error for FileError {}

impl From<io::Error> for FileError {
    fn from(err: io::Error) -> Self {
        FileError::Io(err)
    }
}

/// Create a file at `path` with the given `content`.
///
/// Parent directories are created automatically if they don't exist.
pub fn create_file(path: impl AsRef<Path>, content: &str) -> Result<(), FileError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)?;
    Ok(())
}

/// Read the entire contents of a file as a UTF-8 string.
pub fn read_file(path: impl AsRef<Path>) -> Result<String, FileError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(FileError::NotFound(path.display().to_string()));
    }
    Ok(fs::read_to_string(path)?)
}

/// Write `content` to a file, overwriting any existing content.
///
/// Parent directories are created automatically if they don't exist.
pub fn write_file(path: impl AsRef<Path>, content: &str) -> Result<(), FileError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)?;
    Ok(())
}

/// Delete a file if it exists. Returns `Ok(())` even if the file doesn't exist.
pub fn delete_file(path: impl AsRef<Path>) -> Result<(), FileError> {
    let path = path.as_ref();
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("xencode-files-test-{stamp}"))
    }

    #[test]
    fn create_and_read_file() {
        let dir = temp_dir();
        let path = dir.join("subdir").join("test.txt");

        create_file(&path, "hello world").unwrap();
        let content = read_file(&path).unwrap();
        assert_eq!(content, "hello world");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn write_overwrites_existing() {
        let dir = temp_dir();
        let path = dir.join("overwrite.txt");

        create_file(&path, "first").unwrap();
        write_file(&path, "second").unwrap();
        let content = read_file(&path).unwrap();
        assert_eq!(content, "second");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn delete_existing_file() {
        let dir = temp_dir();
        let path = dir.join("delete_me.txt");

        create_file(&path, "temp").unwrap();
        assert!(path.exists());
        delete_file(&path).unwrap();
        assert!(!path.exists());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn delete_nonexistent_file_succeeds() {
        let dir = temp_dir();
        let path = dir.join("does_not_exist.txt");
        // Should not error
        delete_file(&path).unwrap();
    }

    #[test]
    fn read_nonexistent_file_returns_error() {
        let path = temp_dir().join("nope.txt");
        let result = read_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, FileError::NotFound(_)));
    }
}
