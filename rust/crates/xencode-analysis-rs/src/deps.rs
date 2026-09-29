//! Dependency health for `xcode doctor --deps` (`QO-1`).
//!
//! Composed, not built: `cargo update --dry-run` for the outdated-but-
//! constrained list, the existing advisory corpus (`advisories.rs`, RS-5) for
//! per-crate lookups, and `Cargo.lock` for the locked versions. No fourth
//! thing, and no network of its own — the corpus sync is an explicit command
//! elsewhere, and everything here reads local files or runs local commands.
//!
//! # The honesty rule this is built around
//!
//! Offline — no usable corpus — the answer is **"advisory state unknown"**,
//! never "clean". A missing database and a clean bill of health look
//! identical only to a tool that wants them to.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::advisories::{advisories_for, Outcome};

/// A pending update `cargo update --dry-run` reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Update {
    /// Crate name.
    pub krate: String,
    /// Locked version.
    pub from: String,
    /// Version the resolver would take.
    pub to: String,
}

/// Parse `cargo update --dry-run` output. Only `Updating <name> <from> ->
/// <to>` lines count; everything else (locks created, notes, warnings) is not
/// an update and is ignored rather than misread.
pub fn parse_update_dry_run(text: &str) -> Vec<Update> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("Updating ") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let (Some(krate), Some(from), Some(arrow), Some(to)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if arrow != "->" {
            continue;
        }
        out.push(Update {
            krate: krate.to_string(),
            from: from.trim_start_matches('v').to_string(),
            to: to.trim_start_matches('v').to_string(),
        });
    }
    out
}

/// Locked versions from a `Cargo.lock` file.
pub fn locked_versions(lock_text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut name: Option<&str> = None;
    for line in lock_text.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            name = None;
            continue;
        }
        if let Some(rest) = line.strip_prefix("name = ") {
            name = Some(rest.trim().trim_matches('"'));
        } else if let Some(rest) = line.strip_prefix("version = ") {
            if let Some(name) = name.take() {
                // First pin wins: a lock lists one version per package except
                // when two majors coexist, and the direct dependency is the
                // first the resolver wrote.
                out.entry(name.to_string())
                    .or_insert_with(|| rest.trim().trim_matches('"').to_string());
            }
        }
    }
    out
}

/// Direct dependencies across `crates/*/Cargo.toml`, by name.
pub fn direct_deps(workspace: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Ok(crates) = std::fs::read_dir(workspace.join("crates")) else {
        return out;
    };
    for entry in crates.flatten() {
        let manifest = entry.path().join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&manifest) else {
            continue;
        };
        let mut in_deps = false;
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with('[') {
                in_deps = line == "[dependencies]";
                continue;
            }
            if in_deps {
                if let Some((key, _)) = line.split_once('=') {
                    let key = key.trim();
                    if !key.is_empty() && !key.starts_with('#') {
                        out.insert(key.to_string());
                    }
                }
            }
        }
    }
    out
}

/// What is known about one dependency's advisory state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepState {
    /// No advisory in the corpus touches the locked version.
    Clean,
    /// Advisory ids affecting the locked version.
    Vulnerable(Vec<String>),
    /// No usable corpus (or an unreadable one). Not clean — unknown.
    Unknown(String),
}

/// Assess one crate at one version against a corpus both must share.
pub fn assess(corpus: Option<&Path>, krate: &str, version: &str) -> DepState {
    let Some(corpus) = corpus else {
        return DepState::Unknown("no advisory corpus; run `xencode advisories sync`".to_string());
    };
    let lookup = match advisories_for(corpus, krate) {
        Ok(lookup) => lookup,
        Err(e) => return DepState::Unknown(e.to_string()),
    };
    let mut ids = Vec::new();
    for advisory in &lookup.advisories {
        if let Ok(Outcome::Vulnerable { .. }) = advisory.assess(version) {
            ids.push(advisory.id.clone());
        }
    }
    ids.sort();
    ids.dedup();
    if ids.is_empty() {
        DepState::Clean
    } else {
        DepState::Vulnerable(ids)
    }
}

/// One row of the dependency health report.
#[derive(Debug, Clone)]
pub struct DepRow {
    /// Crate name.
    pub krate: String,
    /// Locked version, when the lock names it.
    pub locked: Option<String>,
    /// Version an update would take, when one is pending.
    pub update_to: Option<String>,
    /// Advisory state.
    pub state: DepState,
}

/// The `Cargo.lock` committed at HEAD, when this is a git repository with
/// one. `None` anywhere the past is unavailable — a report without history
/// still reports the present rather than failing.
pub fn head_lock_text(manifest: &Path) -> Option<(String, String)> {
    let toplevel = std::process::Command::new("git")
        .current_dir(manifest)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())?;
    let rel = Path::new(&manifest)
        .strip_prefix(&toplevel)
        .ok()?
        .join("Cargo.lock")
        .to_string_lossy()
        .into_owned();
    let output = std::process::Command::new("git")
        .current_dir(&toplevel)
        .args(["show", &format!("HEAD:{rel}")])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    Some((
        toplevel,
        String::from_utf8_lossy(&output.stdout).into_owned(),
    ))
}

/// The whole report: rows plus whether the update check ran.
///
/// An empty update list means two different things — fully updated, or the
/// check never ran — and only one of them is good news. The flag keeps them
/// apart instead of letting absence pose as currency.
pub struct DepsReport {
    /// One row per direct dependency.
    pub rows: Vec<DepRow>,
    /// Whether `cargo update --dry-run` ran. `false` means the update column
    /// is unknown, not clean.
    pub updates_checked: bool,
}

/// The whole report: one row per direct dependency.
pub fn doctor_deps(workspace: &Path, corpus: Option<&Path>) -> Result<DepsReport, String> {
    let lock_text = std::fs::read_to_string(workspace.join("Cargo.lock"))
        .map_err(|e| format!("could not read Cargo.lock: {e}"))?;
    let locked = locked_versions(&lock_text);
    let (updates, updates_checked) = match run_update_dry_run(workspace) {
        Some(text) => (
            parse_update_dry_run(&text)
                .into_iter()
                .map(|u| (u.krate, u.to))
                .collect(),
            true,
        ),
        None => (BTreeMap::new(), false),
    };
    let mut rows = Vec::new();
    for krate in direct_deps(workspace) {
        let version = locked.get(&krate).cloned();
        let state = match &version {
            Some(version) => assess(corpus, &krate, version),
            None => DepState::Unknown("not pinned in Cargo.lock".to_string()),
        };
        rows.push(DepRow {
            krate: krate.clone(),
            locked: version,
            update_to: updates.get(&krate).cloned(),
            state,
        });
    }
    Ok(DepsReport {
        rows,
        updates_checked,
    })
}

/// `cargo update --dry-run`: what the resolver *would* take, changing nothing.
/// `None` when cargo cannot run or the network is absent — an outdated list
/// that cannot be computed is absent, not empty.
fn run_update_dry_run(workspace: &Path) -> Option<String> {
    let output = std::process::Command::new("cargo")
        .current_dir(workspace)
        .args(["update", "--dry-run"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// One crate pinned at two or more versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Duplicate {
    /// Crate name.
    pub krate: String,
    /// Distinct locked versions, ascending.
    pub versions: Vec<String>,
}

/// Every `(name, version)` in the lock with more than one distinct version.
pub fn duplicate_versions(lock_text: &str) -> Vec<Duplicate> {
    let mut by_name: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (name, version) in lock_packages(lock_text) {
        by_name.entry(name).or_default().insert(version);
    }
    by_name
        .into_iter()
        .filter(|(_, versions)| versions.len() > 1)
        .map(|(krate, versions)| Duplicate {
            krate,
            versions: versions.into_iter().collect(),
        })
        .collect()
}

/// All `(name, version)` pairs in a lock file, in order. Unlike
/// [`locked_versions`], which keeps the first pin per name, every pin counts
/// here — duplicates are the subject, not a parsing edge.
fn lock_packages(lock_text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut name: Option<&str> = None;
    for line in lock_text.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            name = None;
            continue;
        }
        if let Some(rest) = line.strip_prefix("name = ") {
            name = Some(rest.trim().trim_matches('"'));
        } else if let Some(rest) = line.strip_prefix("version = ") {
            if let Some(name) = name.take() {
                out.push((name.to_string(), rest.trim().trim_matches('"').to_string()));
            }
        }
    }
    out
}

/// Who depends on each `(name, version)`: `"depname depversion"` entries in
/// `dependencies` lists, bare names resolving to the lock's only pin of that
/// name. A dependent the lock cannot resolve is still named, with its version
/// left blank, rather than dropped.
pub fn reverse_deps(lock_text: &str) -> BTreeMap<(String, String), Vec<String>> {
    let mut pins: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, version) in lock_packages(lock_text) {
        pins.entry(name).or_default().push(version);
    }
    let mut out: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    let mut current: Option<(String, String)> = None;
    let mut in_deps = false;
    for line in lock_text.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            current = None;
            in_deps = false;
            continue;
        }
        if let Some(rest) = line.strip_prefix("name = ") {
            let name = rest.trim().trim_matches('"').to_string();
            current = Some((name, String::new()));
        } else if let Some(rest) = line.strip_prefix("version = ") {
            if let Some((name, _)) = current.take() {
                current = Some((name, rest.trim().trim_matches('"').to_string()));
            }
        } else if line == "dependencies = [" {
            in_deps = true;
        } else if line == "]" {
            in_deps = false;
        } else if in_deps {
            // Comma before quotes: `"syn 2.0.0",` ends in a comma, not a
            // quote, and stripping in the other order keeps a stray `"` that
            // then matches nothing downstream.
            let entry = line.trim().trim_end_matches(',').trim_matches('"');
            let mut parts = entry.split_whitespace();
            let (Some(dep), version) = (parts.next(), parts.next()) else {
                continue;
            };
            let version = version
                .map(str::to_string)
                .or_else(|| {
                    pins.get(dep).and_then(|pins| {
                        if pins.len() == 1 {
                            Some(pins[0].clone())
                        } else {
                            None
                        }
                    })
                })
                .unwrap_or_default();
            if let Some(owner) = &current {
                let owner_label = format!("{} {}", owner.0, owner.1);
                out.entry((dep.to_string(), version))
                    .or_default()
                    .push(owner_label);
            }
        }
    }
    for dependents in out.values_mut() {
        dependents.sort();
        dependents.dedup();
    }
    out
}

/// What changed between two lock files: added and removed pins, and upgrades
/// of the same crate. "New" versus "already present" is a diff of machine
/// facts, never a judgement about whether the addition was justified.
#[derive(Debug, Clone, Default)]
pub struct LockDelta {
    /// `(crate, version)` pins only in the new lock.
    pub added: Vec<(String, String)>,
    /// `(crate, version)` pins only in the old lock.
    pub removed: Vec<(String, String)>,
    /// `(crate, from, to)` version moves.
    pub upgraded: Vec<(String, String, String)>,
}

pub fn lock_delta(old_text: &str, new_text: &str) -> LockDelta {
    let old_pins: BTreeSet<(String, String)> = lock_packages(old_text).into_iter().collect();
    let new_pins: BTreeSet<(String, String)> = lock_packages(new_text).into_iter().collect();
    let mut delta = LockDelta::default();
    for pin in new_pins.difference(&old_pins) {
        delta.added.push(pin.clone());
    }
    for pin in old_pins.difference(&new_pins) {
        delta.removed.push(pin.clone());
    }
    let old_versions: BTreeMap<&str, &str> = old_pins
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_str()))
        .collect();
    let new_versions: BTreeMap<&str, &str> = new_pins
        .iter()
        .map(|(n, v)| (n.as_str(), v.as_str()))
        .collect();
    for (name, from) in &old_versions {
        if let Some(to) = new_versions.get(name) {
            if from != to {
                delta
                    .upgraded
                    .push((name.to_string(), from.to_string(), to.to_string()));
            }
        }
    }
    // A crate pinned twice complicates "upgraded": report the simple case only
    // when each side pins the name once, and move those pins out of added and
    // removed so one version move is not reported three times.
    delta.upgraded.retain(|(name, _, _)| {
        old_pins.iter().filter(|(n, _)| n == name).count() == 1
            && new_pins.iter().filter(|(n, _)| n == name).count() == 1
    });
    for (name, from, to) in &delta.upgraded {
        delta.added.retain(|pin| pin != &(name.clone(), to.clone()));
        delta
            .removed
            .retain(|pin| pin != &(name.clone(), from.clone()));
    }
    delta.added.sort();
    delta.removed.sort();
    delta.upgraded.sort();
    delta
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRY_RUN: &str = "    Updating serde v1.0.229 -> v1.0.230\n    Updating index\nwarning: unused manifest key\n    Updating tokio v1.44 -> v1.45.0\n";

    #[test]
    fn only_updating_lines_count() {
        assert_eq!(
            parse_update_dry_run(DRY_RUN),
            vec![
                Update {
                    krate: "serde".to_string(),
                    from: "1.0.229".to_string(),
                    to: "1.0.230".to_string()
                },
                Update {
                    krate: "tokio".to_string(),
                    from: "1.44".to_string(),
                    to: "1.45.0".to_string()
                },
            ]
        );
        assert!(parse_update_dry_run("Locking 3 packages\n").is_empty());
    }

    #[test]
    fn the_first_pin_wins_when_two_majors_coexist() {
        let lock = "[[package]]\nname = \"serde\"\nversion = \"1.0.229\"\n\n[[package]]\nname = \"serde\"\nversion = \"0.9.9\"\n";
        assert_eq!(
            locked_versions(lock).get("serde").map(String::as_str),
            Some("1.0.229")
        );
    }

    const DUP_LOCK: &str = "[[package]]\nname = \"app\"\nversion = \"0.1.0\"\ndependencies = [\n \"syn 2.0.0\",\n \"helper\",\n]\n\n[[package]]\nname = \"syn\"\nversion = \"2.0.0\"\n\n[[package]]\nname = \"syn\"\nversion = \"3.0.0\"\n\n[[package]]\nname = \"helper\"\nversion = \"0.2.0\"\ndependencies = [\n \"syn\",\n]\n";

    #[test]
    fn a_second_major_version_is_named_with_both_versions() {
        let dups = duplicate_versions(DUP_LOCK);
        assert_eq!(dups.len(), 1);
        assert_eq!(dups[0].krate, "syn");
        assert_eq!(dups[0].versions, vec!["2.0.0", "3.0.0"]);
    }

    #[test]
    fn reverse_deps_name_both_paths() {
        let rev = reverse_deps(DUP_LOCK);
        // Pinned entries resolve exactly; the bare `syn` in helper resolves
        // only when the lock pins it once — here it pins twice, so the version
        // stays blank rather than guessed.
        let two = rev
            .get(&("syn".to_string(), "2.0.0".to_string()))
            .cloned()
            .unwrap_or_default();
        assert!(two.iter().any(|d| d.starts_with("app ")), "{rev:?}");
        assert!(
            rev.contains_key(&("syn".to_string(), String::new())),
            "{rev:?}"
        );
    }

    #[test]
    fn a_delta_distinguishes_new_from_moved() {
        let old = "[[package]]\nname = \"a\"\nversion = \"1.0.0\"\n\n[[package]]\nname = \"b\"\nversion = \"1.0.0\"\n";
        let new = "[[package]]\nname = \"a\"\nversion = \"1.1.0\"\n\n[[package]]\nname = \"c\"\nversion = \"2.0.0\"\n";
        let delta = lock_delta(old, new);
        assert_eq!(
            delta.upgraded,
            vec![("a".to_string(), "1.0.0".to_string(), "1.1.0".to_string())]
        );
        assert_eq!(delta.added, vec![("c".to_string(), "2.0.0".to_string())]);
        assert_eq!(delta.removed, vec![("b".to_string(), "1.0.0".to_string())]);
    }

    #[test]
    fn no_corpus_is_unknown_never_clean() {
        // The honesty rule, pinned: absence of data must not read as health.
        assert!(matches!(
            assess(None, "serde", "1.0.229"),
            DepState::Unknown(_)
        ));
    }

    #[test]
    fn a_fixture_corpus_distinguishes_vulnerable_from_clean() {
        let dir = std::env::temp_dir().join(format!("xe-deps-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("sync.json"),
            r#"{"synced_at_unix":1,"rustsec_revision":"x","rustsec_advisories":1,"osv_records":0,"index_lines":1}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("index.tsv"),
            "democrate	rustsec	rustsec/RUSTSEC-2026-0001.md",
        )
        .unwrap();
        std::fs::create_dir_all(dir.join("rustsec")).unwrap();
        std::fs::write(
            dir.join("rustsec").join("RUSTSEC-2026-0001.md"),
            "```toml\n[advisory]\nid = \"RUSTSEC-2026-0001\"\npackage = \"democrate\"\ndate = \"2026-01-01\"\nurl = \"https://example.invalid\"\n\n[versions]\npatched = [\">=2.0.0\"]\n```\n",
        )
        .unwrap();
        match assess(Some(&dir), "democrate", "1.0.0") {
            DepState::Vulnerable(ids) => assert_eq!(ids, vec!["RUSTSEC-2026-0001"]),
            other => panic!("1.0.0 is affected: {other:?}"),
        }
        assert!(matches!(
            assess(Some(&dir), "democrate", "2.1.0"),
            DepState::Clean
        ));
        assert!(matches!(
            assess(Some(&dir), "othercrate", "1.0.0"),
            DepState::Clean
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
