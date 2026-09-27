//! The upstream source of the crates this project locked, as a read-only
//! reference the agent can consult.
//!
//! `Cargo.lock` names the exact version of every dependency the build uses, and
//! cargo has already unpacked that version's source somewhere under
//! `$CARGO_HOME/registry/src/`. Reading it answers "what does this API actually
//! do" with no network call — but only if the version is the locked one. That
//! directory holds several versions of the same crate side by side, so picking
//! "whatever is on disk" answers a question about a build this project does not
//! have. Everything here therefore starts from the lock file and refuses to
//! guess when the lock file leaves more than one candidate.

use std::path::{Component, Path, PathBuf};

/// One locked dependency whose unpacked source is on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateSource {
    pub name: String,
    pub version: String,
    /// `…/registry/src/<registry>/<name>-<version>`.
    pub dir: PathBuf,
}

impl CrateSource {
    fn dir_name(&self) -> String {
        format!("{}-{}", self.name, self.version)
    }

    /// How a read that came from here is labelled, so the version a model is
    /// quoting is never implicit.
    pub fn label(&self) -> String {
        format!(
            "{} {} — the version this project's Cargo.lock pins",
            self.name, self.version
        )
    }

    /// The same location expressed as the `crate:` shorthand.
    pub fn spec_for(&self, inside: &str) -> String {
        if inside.is_empty() {
            format!("crate:{}", self.name)
        } else {
            format!("crate:{}/{}", self.name, inside)
        }
    }

    /// The crate-relative part of a path inside this source, or "" when the
    /// path is not inside it.
    pub fn inside(&self, path: &Path) -> Option<String> {
        let rel = path.strip_prefix(&self.dir).ok()?;
        Some(rel.to_string_lossy().replace('\\', "/"))
    }
}

/// The `name`/`version` of every `[[package]]` entry in Cargo.lock text, in
/// file order. Deliberately a line reader rather than a TOML parser: the file is
/// machine-generated with one shape, and a dependency in the agent's path
/// checker is not worth adding for it.
pub fn locked_packages(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut name: Option<String> = None;
    let mut version: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            if let (Some(n), Some(v)) = (name.take(), version.take()) {
                out.push((n, v));
            }
            continue;
        }
        if let Some(value) = lock_key(line, "name") {
            name = Some(value);
        } else if let Some(value) = lock_key(line, "version") {
            version = Some(value);
        }
    }
    if let (Some(n), Some(v)) = (name, version) {
        out.push((n, v));
    }
    out
}

fn lock_key(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim();
    let rest = rest.strip_prefix('=')?.trim();
    let value = rest.strip_prefix('"')?.strip_suffix('"')?;
    Some(value.to_string())
}

/// The lock file that governs `root`: `root/Cargo.lock`, then its parents up to
/// the project boundary (a directory holding `.git`), because the agent is often
/// pointed at one crate inside a larger workspace.
pub fn lock_file(root: &Path) -> Option<PathBuf> {
    let start = std::path::absolute(root).unwrap_or_else(|_| root.to_path_buf());
    let mut current = Some(start.as_path());
    while let Some(dir) = current {
        let candidate = dir.join("Cargo.lock");
        if candidate.is_file() {
            return Some(candidate);
        }
        if dir.join(".git").exists() {
            return None;
        }
        current = dir.parent();
    }
    None
}

/// Every locked dependency of the workspace containing `root`; empty when there
/// is no readable lock file.
pub fn locked_packages_for(root: &Path) -> Vec<(String, String)> {
    lock_file(root)
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| locked_packages(&text))
        .unwrap_or_default()
}

/// The directories cargo unpacks registry sources into: one per registry, each
/// holding `<name>-<version>` directories.
pub fn registry_src_dirs(cargo_home: &Path) -> Vec<PathBuf> {
    let src = cargo_home.join("registry").join("src");
    let Ok(entries) = std::fs::read_dir(&src) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// Where cargo keeps its registry: `$CARGO_HOME`, else `$HOME/.cargo`.
pub fn cargo_home() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("CARGO_HOME") {
        if !dir.trim().is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    std::env::var("HOME")
        .ok()
        .filter(|home| !home.is_empty())
        .map(|home| PathBuf::from(home).join(".cargo"))
}

/// Whether a model-supplied relative path stays inside the crate it names.
pub(crate) fn path_is_contained(path: &str) -> bool {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        return false;
    }
    !candidate.components().any(|c| {
        !matches!(c, Component::Normal(_)) || matches!(c, Component::Normal(name) if name == ".git")
    })
}

/// The locked crate whose unpacked source contains `path`, if `path` is inside
/// one. A directory that is on disk but not locked is not a source of truth
/// here, so it comes back as None and the caller's normal workspace rule
/// applies to it.
pub fn crate_source_for_path(root: &Path, path: &Path) -> Option<CrateSource> {
    let home = cargo_home()?;
    let locked = locked_packages_for(root);
    if locked.is_empty() {
        return None;
    }
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    crate_source_in_dirs(&registry_src_dirs(&home), &locked, &normalize(&path))
}

/// Which locked crate a path sits in, given the registry layout and the lock.
/// Separate from [`crate_source_for_path`] so the version rules can be tested
/// against a directory that only a test has built.
pub fn crate_source_in_dirs(
    dirs: &[PathBuf],
    locked: &[(String, String)],
    path: &Path,
) -> Option<CrateSource> {
    for src in dirs {
        let Ok(relative) = path.strip_prefix(src) else {
            continue;
        };
        let mut components = relative.components();
        let Component::Normal(dir_name) = components.next()? else {
            continue;
        };
        let rest: PathBuf = components.collect();
        if !rest.components().all(|c| matches!(c, Component::Normal(_))) {
            return None;
        }
        let dir_name = dir_name.to_string_lossy().to_string();
        let (name, version) = locked
            .iter()
            .find(|(name, version)| format!("{name}-{version}") == dir_name)?;
        return Some(CrateSource {
            name: name.clone(),
            version: version.clone(),
            dir: src.join(&dir_name),
        });
    }
    None
}

fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Whether `raw` is addressed as `crate:<name>[/<path inside>]`.
pub fn is_crate_spec(raw: &str) -> bool {
    raw.trim().starts_with("crate:")
}

/// Resolve a `crate:` address against the lock file of `root`.
///
/// The error strings are written for the model that caused them: they name what
/// is missing and what to do instead, because "permission denied" would send it
/// back to searching the internet for a text file already on this machine.
pub fn resolve_crate_spec(root: &Path, raw: &str) -> Result<(PathBuf, CrateSource), String> {
    let Some(address) = raw.trim().strip_prefix("crate:") else {
        return Err(format!("{raw} is not a crate: address"));
    };
    let (name, inside) = match address.split_once('/') {
        Some((name, inside)) => (name.trim(), inside.trim()),
        None => (address.trim(), ""),
    };
    if name.is_empty() {
        return Err("crate: needs a package name, as in crate:serde/src/lib.rs".to_string());
    }
    if !inside.is_empty() && !path_is_contained(inside) {
        return Err(format!(
            "crate:{name}/{inside} reaches outside the crate directory; keep the path after \
             the name inside it"
        ));
    }
    let Some(home) = cargo_home() else {
        return Err(
            "cannot find cargo's home directory, so no unpacked crate source can be located; \
             set CARGO_HOME or pass a path inside the workspace"
                .to_string(),
        );
    };
    resolve_crate_in(
        &registry_src_dirs(&home),
        &locked_packages_for(root),
        name,
        inside,
    )
}

/// Where a crate source directory is, given the lock and the registry layout;
/// separate so the version rules are testable without touching `$CARGO_HOME`.
pub fn resolve_crate_in(
    dirs: &[PathBuf],
    locked: &[(String, String)],
    name: &str,
    inside: &str,
) -> Result<(PathBuf, CrateSource), String> {
    if locked.is_empty() {
        return Err(
            "no Cargo.lock is readable for this project, so there is no locked version to \
             read: pass a path inside the workspace instead"
                .to_string(),
        );
    }
    let pinned: Vec<&(String, String)> = locked.iter().filter(|(n, _)| n == name).collect();
    if pinned.is_empty() {
        return Err(format!(
            "{name} is not a dependency of this project — nothing in Cargo.lock names it"
        ));
    }
    let mut available: Vec<CrateSource> = Vec::new();
    for (n, v) in pinned.iter() {
        let source = CrateSource {
            name: (*n).clone(),
            version: (*v).clone(),
            dir: PathBuf::new(),
        };
        if let Some(found) = dirs
            .iter()
            .map(|d| d.join(source.dir_name()))
            .find(|d| d.is_dir())
        {
            available.push(CrateSource {
                dir: found,
                ..source
            });
        }
    }
    let source = match available.len() {
        0 => {
            let versions = pinned
                .iter()
                .map(|(_, v)| v.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "Cargo.lock pins {name} {versions} but its source is not unpacked on this \
                 machine; run `cargo fetch` in the workspace, then ask again"
            ));
        }
        1 => available.pop().expect("one entry"),
        _ => {
            let listing = available
                .iter()
                .map(|s| format!("{} = {}", s.version, s.dir.display()))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(format!(
                "Cargo.lock pins {name} in more than one version, so the name alone is \
                 ambiguous: {listing}. Pass one of those directories as a path."
            ));
        }
    };
    let path = if inside.is_empty() {
        source.dir.clone()
    } else {
        source.dir.join(inside)
    };
    Ok((path, source))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("xencode-crate-src-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A cargo home holding one registry with `<name>-<version>` unpacked for
    /// each entry, each carrying a Cargo.toml that repeats its own version.
    fn unpacked(crates: &[(&str, &str)]) -> PathBuf {
        let home = scratch("home");
        let src = home
            .join("registry")
            .join("src")
            .join("index.crates.io-test");
        for (name, version) in crates {
            let package = src.join(format!("{name}-{version}"));
            std::fs::create_dir_all(package.join("src")).unwrap();
            std::fs::write(
                package.join("Cargo.toml"),
                format!("[package]\nname = \"{name}\"\nversion = \"{version}\"\n"),
            )
            .unwrap();
            std::fs::write(package.join("src").join("lib.rs"), "// upstream source\n").unwrap();
        }
        home
    }

    const LOCK: &str = "\
# This file is automatically @generated by Cargo.
version = 4

[[package]]
name = \"serde\"
version = \"1.0.229\"
source = \"registry+https://github.com/rust-lang/crates.io-index\"
checksum = \"abcd\"

[[package]]
name = \"adobe-cmap-parser\"
version = \"0.4.1\"
source = \"registry+https://github.com/rust-lang/crates.io-index\"

[[package]]
name = \"regex\"
version = \"1.11.1\"
source = \"registry+https://github.com/rust-lang/crates.io-index\"
";

    fn locked() -> Vec<(String, String)> {
        locked_packages(LOCK)
    }

    #[test]
    fn the_lock_text_yields_every_package_it_pins_in_order() {
        assert_eq!(
            locked(),
            vec![
                ("serde".to_string(), "1.0.229".to_string()),
                ("adobe-cmap-parser".to_string(), "0.4.1".to_string()),
                ("regex".to_string(), "1.11.1".to_string()),
            ]
        );
        // A key that merely starts like the ones read must not be read as one.
        assert_eq!(
            locked_packages("[[package]]\nnames = \"not-this\"\n"),
            Vec::new()
        );
    }

    #[test]
    fn the_pinned_version_is_the_one_reached_when_two_are_on_disk() {
        let home = unpacked(&[("serde", "1.0.219"), ("serde", "1.0.229")]);
        let dirs = registry_src_dirs(&home);
        assert_eq!(dirs.len(), 1, "{dirs:?}");
        let (path, source) = resolve_crate_in(&dirs, &locked(), "serde", "src/lib.rs").unwrap();
        assert_eq!(source.version, "1.0.229");
        assert!(
            path.ends_with("serde-1.0.229/src/lib.rs"),
            "{}",
            path.display()
        );
        assert_eq!(
            source.label(),
            "serde 1.0.229 — the version this project's Cargo.lock pins"
        );
        assert_eq!(source.spec_for("src/lib.rs"), "crate:serde/src/lib.rs");
        // The other directory on disk is not a source of truth for this project.
        let other = dirs[0].join("serde-1.0.219").join("src").join("lib.rs");
        assert_eq!(crate_source_in_dirs(&dirs, &locked(), &other), None);
        // … and the pinned one is recognised from an absolute path back to a name.
        let found = crate_source_in_dirs(&dirs, &locked(), &path).unwrap();
        assert_eq!(
            (found.name.as_str(), found.version.as_str()),
            ("serde", "1.0.229")
        );
        assert_eq!(found.inside(&path).as_deref(), Some("src/lib.rs"));
    }

    #[test]
    fn a_crate_the_lock_does_not_name_is_refused_by_name() {
        let home = unpacked(&[("serde", "1.0.229")]);
        let error = resolve_crate_in(&registry_src_dirs(&home), &locked(), "rand", "").unwrap_err();
        assert!(error.contains("rand is not a dependency"), "{error}");
        assert!(error.contains("Cargo.lock"), "{error}");
    }

    #[test]
    fn a_pinned_crate_that_is_not_unpacked_points_at_cargo_fetch() {
        let home = scratch("empty-home"); // no registry/src at all
        let error =
            resolve_crate_in(&registry_src_dirs(&home), &locked(), "serde", "").unwrap_err();
        assert!(error.contains("not unpacked on this machine"), "{error}");
        assert!(error.contains("cargo fetch"), "{error}");
    }

    #[test]
    fn two_pinned_versions_of_one_name_are_listed_rather_than_guessed() {
        let home = unpacked(&[("serde", "1.0.229"), ("serde", "1.0.219")]);
        let both = vec![
            ("serde".to_string(), "1.0.229".to_string()),
            ("serde".to_string(), "1.0.219".to_string()),
        ];
        let error = resolve_crate_in(&registry_src_dirs(&home), &both, "serde", "").unwrap_err();
        assert!(error.contains("more than one version"), "{error}");
        assert!(
            error.contains("1.0.229") && error.contains("1.0.219"),
            "{error}"
        );
    }

    #[test]
    fn a_project_with_no_lock_has_no_crate_addresses() {
        let home = unpacked(&[("serde", "1.0.229")]);
        let error = resolve_crate_in(&registry_src_dirs(&home), &[], "serde", "").unwrap_err();
        assert!(error.contains("no Cargo.lock"), "{error}");
    }

    #[test]
    fn an_address_that_walks_out_of_the_crate_is_refused_before_any_read() {
        for spec in [
            "crate:serde/../../etc/passwd",
            "crate:serde/src/../../../Cargo.toml",
            "crate:serde//etc/passwd",
            "crate:serde/.git/config",
        ] {
            let error = resolve_crate_spec(Path::new("."), spec).unwrap_err();
            assert!(
                error.contains("reaches outside the crate directory"),
                "{error}"
            );
        }
        assert!(is_crate_spec(" crate:serde"));
        assert!(!is_crate_spec("./src/lib.rs"));
    }

    #[test]
    fn the_lock_is_found_above_the_crate_but_not_past_the_project() {
        let project = scratch("project");
        std::fs::create_dir_all(project.join(".git")).unwrap();
        let member = project.join("crates").join("one");
        std::fs::create_dir_all(&member).unwrap();
        // Nothing above the member inside this project: the walk stops at the
        // directory holding `.git` rather than escaping into the home folder.
        assert_eq!(lock_file(&member), None);
        std::fs::write(project.join("Cargo.lock"), LOCK).unwrap();
        assert_eq!(lock_file(&member), Some(project.join("Cargo.lock")));
        assert_eq!(locked_packages_for(&member).len(), 3);
    }
}
