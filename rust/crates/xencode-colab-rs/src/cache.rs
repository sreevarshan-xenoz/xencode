//! Pure logic for reusing GGUF weights from a Google Drive cache inside a
//! Colab VM. Nothing here mounts Drive or talks to Google: that plumbing is not
//! wired up yet and stays blocked. The module only answers one question from a
//! real directory on disk — which of the parts a model needs are already there
//! — so the bootstrap can skip a download when a mount happens to be present
//! and a person has saved weights into it.
//!
//! `unsafe` is forbidden here: this is path arithmetic and file metadata.

use std::path::{Path, PathBuf};

/// Where xencode looks for reusable weights on a mounted Google Drive,
/// scoped by model, quantisation and the pinned llama.cpp build so a different
/// build cannot silently reuse binaries fetched for another one.
///
/// Colab mounts personal Drive at `/content/drive/MyDrive`. The product does
/// **not** mount it yet — this is the layout the generated bootstrap inspects,
/// and it stays dormant unless a directory with these names really exists.
pub fn drive_cache_dir(model: &str, quant: &str, build: &str) -> PathBuf {
    PathBuf::from("/content/drive/MyDrive/xencode-models")
        .join(model)
        .join(quant)
        .join(build)
}

/// Result of checking a cache directory for the parts a model needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheDecision {
    /// Every expected part is already a non-empty file in the cache: the
    /// bootstrap can reuse it without fetching anything.
    UseCache,
    /// These parts are absent, empty, or not regular files, and must be
    /// fetched. Order and exact names are the caller's, unchanged.
    DownloadMissing(Vec<String>),
}

/// The path a part occupies in the cache. The generated bootstrap keeps each
/// GGUF under its basename (llama.cpp finds the other parts of a split model
/// by name beside the first one), so an HF sub-folder prefix is dropped here
/// exactly as the shell's `basename "$PART"` drops it.
fn cache_file(cache_dir: &Path, part: &str) -> PathBuf {
    let name = Path::new(part)
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_else(|| part.into());
    cache_dir.join(name)
}

/// Decide, from a real directory, which of `expected_parts` are usable from
/// the cache. A part counts as cached only when its file is a non-empty
/// regular file — the same `test -s` the generated shell performs. A missing
/// directory, a missing file, a directory in place of a file, or a zero-byte
/// file all mean "download this part".
///
/// This is deliberately plain I/O over a path the caller supplies. Production
/// passes the Drive mount path; tests pass a real temporary directory as a
/// stand-in. Nothing here pretends a Drive mount exists.
pub fn cache_decision(cache_dir: &Path, expected_parts: &[String]) -> CacheDecision {
    let missing: Vec<String> = expected_parts
        .iter()
        .filter(|part| {
            !cache_file(cache_dir, part)
                .metadata()
                .map(|m| m.is_file() && m.len() > 0)
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    if missing.is_empty() {
        CacheDecision::UseCache
    } else {
        CacheDecision::DownloadMissing(missing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Removes the temporary stand-in mount when a test ends, including on a
    /// failed assertion, so no `xencode-*` directory is left in the temp dir.
    struct MountGuard(PathBuf);
    impl Drop for MountGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn mount(tag: &str) -> (MountGuard, PathBuf) {
        let dir = crate::testutil::temp_dir(tag);
        (MountGuard(dir.clone()), dir)
    }

    fn parts(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn write_part(dir: &Path, name: &str, bytes: &[u8]) {
        fs::write(dir.join(name), bytes).expect("write cached part");
    }

    #[test]
    fn drive_cache_dir_is_scoped_by_model_quant_and_build() {
        assert_eq!(
            drive_cache_dir("Qwen/Qwen2.5-7B-Instruct-GGUF", "Q4_K_M", "b11120"),
            PathBuf::from(
                "/content/drive/MyDrive/xencode-models/Qwen/Qwen2.5-7B-Instruct-GGUF/Q4_K_M/b11120"
            )
        );
    }

    #[test]
    fn a_missing_cache_dir_downloads_every_part_in_order() {
        let (_guard, dir) = mount("cache-missing-dir");
        let absent = dir.join("no-mount");
        let expected = parts(&["q4-00001-of-00002.gguf", "q4-00002-of-00002.gguf"]);
        assert_eq!(
            cache_decision(&absent, &expected),
            CacheDecision::DownloadMissing(expected)
        );
    }

    #[test]
    fn an_empty_cache_dir_downloads_every_part() {
        let (_guard, dir) = mount("cache-empty-dir");
        let expected = parts(&["model-q4_k_m.gguf"]);
        assert_eq!(
            cache_decision(&dir, &expected),
            CacheDecision::DownloadMissing(expected)
        );
    }

    #[test]
    fn all_parts_present_uses_the_cache() {
        let (_guard, dir) = mount("cache-all-present");
        let expected = parts(&["q4-00001-of-00002.gguf", "q4-00002-of-00002.gguf"]);
        for part in &expected {
            write_part(&dir, part, b"non-empty gguf bytes");
        }
        assert_eq!(cache_decision(&dir, &expected), CacheDecision::UseCache);
    }

    #[test]
    fn some_parts_present_downloads_only_the_missing_ones_in_order() {
        let (_guard, dir) = mount("cache-some-present");
        let expected = parts(&[
            "q4-00001-of-00002.gguf",
            "q4-00002-of-00002.gguf",
            "q4-00003-of-00003.gguf",
        ]);
        write_part(&dir, &expected[1], b"part two");
        assert_eq!(
            cache_decision(&dir, &expected),
            CacheDecision::DownloadMissing(vec![expected[0].clone(), expected[2].clone(),])
        );
    }

    #[test]
    fn empty_files_and_directories_do_not_count_as_cached() {
        let (_guard, dir) = mount("cache-empty-file");
        let good = "model-q4_k_m.gguf".to_string();
        let empty = "model-q8_0.gguf".to_string();
        let as_dir = "model-q5_k_m.gguf".to_string();
        write_part(&dir, &good, b"real weights");
        write_part(&dir, &empty, b"");
        fs::create_dir(dir.join(&as_dir)).expect("directory named like a part");
        assert_eq!(
            cache_decision(&dir, &[good, empty.clone(), as_dir.clone()]),
            CacheDecision::DownloadMissing(vec![empty, as_dir])
        );
    }

    #[test]
    fn an_hf_subfolder_part_is_looked_up_by_its_basename() {
        let (_guard, dir) = mount("cache-subfolder");
        // The decision reports the caller's exact string back when it is
        // missing, so the bootstrap can still build the HF URL from it.
        let part = "subfolder/model-q4_k_m.gguf".to_string();
        assert_eq!(
            cache_decision(&dir, std::slice::from_ref(&part)),
            CacheDecision::DownloadMissing(vec![part.clone()])
        );
        write_part(&dir, "model-q4_k_m.gguf", b"real weights");
        assert_eq!(cache_decision(&dir, &[part]), CacheDecision::UseCache);
    }
}
