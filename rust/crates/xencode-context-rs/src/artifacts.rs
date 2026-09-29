//! Per-session artifact directories (`EVd-4`).
//!
//! `.xencode/artifacts/<session>/` holds the evidence ledger rows point at:
//! compact run logs, not raw transcripts. Two rules keep the directory honest:
//!
//! - **Tails, not heads.** Every write keeps the last bytes on a character
//!   boundary, because the end of a log is where the failure is and the start
//!   is where the environment dump is. The cap matches the tool loop's command
//!   output cap, so one number governs both.
//! - **Prune by policy, not by hope.** `prune_artifacts` keeps the newest N
//!   session dirs plus every session with a failing ledger row, and removes
//!   the rest. A `cargo test` loop that never prunes fills a disk; the trap is
//!   handled by running the prune after every write rather than by asking.
//!
//! The directory lives under `.xencode/`, which is git-ignored, so artifacts
//! never reach a commit by accident.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Where session evidence lives.
pub const ARTIFACTS_DIR: &str = "artifacts";

/// Maximum bytes kept per artifact. Same value as the tool loop's command
/// output cap: one number for both, so a log written here and a log shown
/// there cannot disagree about what "the tail" means.
pub const ARTIFACT_BYTES_CAP: usize = 8 * 1024;

/// How many passing session dirs survive a prune, besides every failing one.
pub const KEEP_LAST_PASSING: usize = 5;

/// The directory for one session's evidence, created on demand.
pub fn artifact_dir(xencode_dir: &Path, session: &str) -> std::io::Result<PathBuf> {
    let dir = xencode_dir.join(ARTIFACTS_DIR).join(session);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Write an artifact, keeping the tail when the content exceeds the cap.
///
/// Returns the path written. The cap is on bytes at a character boundary, so a
/// multi-byte character is never split — a split character is a corrupt file
/// wearing a complete one's name.
pub fn write_artifact(
    xencode_dir: &Path,
    session: &str,
    name: &str,
    content: &str,
) -> std::io::Result<PathBuf> {
    let dir = artifact_dir(xencode_dir, session)?;
    let kept = tail_bytes(content, ARTIFACT_BYTES_CAP);
    let path = dir.join(safe_name(name));
    std::fs::write(&path, kept)?;
    Ok(path)
}

/// The last `cap` bytes of `content` on a character boundary.
fn tail_bytes(content: &str, cap: usize) -> &str {
    if content.len() <= cap {
        return content;
    }
    let mut start = content.len() - cap;
    while start < content.len() && !content.is_char_boundary(start) {
        start += 1;
    }
    &content[start..]
}

/// A file name that cannot escape the session directory.
fn safe_name(name: &str) -> String {
    let kept: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = kept.trim_matches('.').trim();
    if trimmed.is_empty() {
        "artifact.log".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Sessions with at least one failing ledger row. These are never pruned: a
/// failure whose evidence was deleted is a verdict without a witness.
pub fn failing_sessions(xencode_dir: &Path) -> BTreeSet<String> {
    crate::ledger::read_ledger(xencode_dir)
        .into_iter()
        .filter(|row| !row.passed())
        .filter_map(|row| row.session)
        .collect()
}

/// Remove old passing session dirs, keeping the newest few and every failure.
///
/// Returns the number of directories removed. A session the ledger calls
/// failing survives regardless of age; everything else older than the newest
/// `keep_last` goes.
pub fn prune_artifacts(xencode_dir: &Path, keep_last: usize) -> usize {
    let root = xencode_dir.join(ARTIFACTS_DIR);
    let Ok(entries) = std::fs::read_dir(&root) else {
        return 0;
    };
    let mut dirs: Vec<(PathBuf, std::time::SystemTime)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .map(|p| {
            let mtime = std::fs::metadata(&p)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (p, mtime)
        })
        .collect();
    // Newest first; a tie breaks by name so the order is stable.
    dirs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let failing = failing_sessions(xencode_dir);
    let mut removed = 0;
    for (index, (dir, _)) in dirs.iter().enumerate() {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if failing.contains(&name) || index < keep_last {
            continue;
        }
        if std::fs::remove_dir_all(dir).is_ok() {
            removed += 1;
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_xencode(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xe-art-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let xencode = dir.join(".xencode");
        std::fs::create_dir_all(&xencode).unwrap();
        xencode
    }

    #[test]
    fn a_long_log_keeps_its_tail_not_its_head() {
        let xencode = temp_xencode("tail");
        let content = format!("HEAD-{}END", "x".repeat(ARTIFACT_BYTES_CAP + 100));
        let path = write_artifact(&xencode, "s1", "test.log", &content).unwrap();
        let kept = std::fs::read_to_string(&path).unwrap();
        assert!(kept.ends_with("END"), "the failure is at the end");
        assert!(kept.len() <= ARTIFACT_BYTES_CAP);
        assert!(!kept.contains("HEAD-"), "the head was cut, not kept");
    }

    #[test]
    fn a_name_cannot_escape_the_session_dir() {
        let xencode = temp_xencode("safe");
        let path = write_artifact(&xencode, "s1", "../../evil.log", "x").unwrap();
        assert!(path.starts_with(xencode.join(ARTIFACTS_DIR).join("s1")));
        assert_eq!(safe_name("..."), "artifact.log");
        assert_eq!(safe_name("test-1.log"), "test-1.log");
    }

    #[test]
    fn prune_keeps_failures_and_the_newest_and_removes_the_rest() {
        let xencode = temp_xencode("prune");
        // Aged oldest-first with explicit mtimes: sleeps are timing-dependent
        // and this ordering is the thing under test, so it is pinned, not hoped for.
        let sessions = ["old-pass", "mid-pass", "new-pass", "old-fail"];
        for (index, session) in sessions.iter().enumerate() {
            write_artifact(&xencode, session, "t.log", "x").unwrap();
            set_mtime(
                &xencode.join(ARTIFACTS_DIR).join(session),
                1_700_000_000 + index as u64,
            );
        }
        // Ledger rows live in this xencode dir, not the throwaway above.
        for (session, code) in [
            ("old-pass", 0),
            ("mid-pass", 0),
            ("new-pass", 0),
            ("old-fail", 1),
        ] {
            crate::ledger::append_ledger(
                &xencode,
                &crate::ledger::LedgerEntry {
                    ts_unix_ms: 1,
                    session: Some(session.to_string()),
                    run_class: crate::ledger::RunClass::Test,
                    exit_code: code,
                    subjects: vec![],
                    log_ref: String::new(),
                    note: String::new(),
                },
            )
            .unwrap();
        }
        let removed = prune_artifacts(&xencode, 3);
        assert_eq!(
            removed, 1,
            "only old-pass goes: the rest are newest or failed"
        );
        assert!(xencode.join(ARTIFACTS_DIR).join("old-fail").is_dir());
        assert!(!xencode.join(ARTIFACTS_DIR).join("old-pass").exists());
    }

    #[cfg(unix)]
    fn set_mtime(path: &Path, secs: u64) {
        use std::os::unix::ffi::OsStrExt;
        let cpath = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        let times = [
            libc::timespec {
                tv_sec: secs as libc::time_t,
                tv_nsec: 0,
            },
            libc::timespec {
                tv_sec: secs as libc::time_t,
                tv_nsec: 0,
            },
        ];
        let ret = unsafe { libc::utimensat(libc::AT_FDCWD, cpath.as_ptr(), times.as_ptr(), 0) };
        assert_eq!(ret, 0, "could not set the mtime under test");
    }

    #[cfg(not(unix))]
    fn set_mtime(_path: &Path, _secs: u64) {}

    #[test]
    fn pruning_nothing_is_not_an_error() {
        let xencode = temp_xencode("empty");
        assert_eq!(prune_artifacts(&xencode, 5), 0);
    }
}
