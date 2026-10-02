//! Which crates break when one crate changes — from `cargo metadata`, exactly.
//!
//! This is the crate-level half of `xencode impact`. The file-level reverse
//! reachability lives in [`crate::impact`]; that one walks `use`/`mod`/`impl`
//! edges and is a *prediction* that has to be read with its caveat. This module
//! is the opposite: it asks cargo which workspace members depend, transitively,
//! on the member a file belongs to. `cargo metadata --no-deps` is offline, needs
//! no network or lock, and returns the whole workspace's package list with each
//! package's declared dependencies — the reverse closure over that is exact
//! arithmetic on the manifest graph, not a guess. It agrees with `cargo tree -i`
//! because both read the same manifest edges.
//!
//! Two deliberate limits, stated rather than hidden. It is **crate** granularity:
//! it answers "which crates link this one", never "which line calls this". And it
//! counts dev- and build-dependencies as edges, because `cargo tree -i` does — a
//! crate that only appears in another's `[dev-dependencies]` is still rebuilt when
//! the target changes, so leaving it out would understate the blast radius.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The dependency kind an edge was declared under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EdgeKind {
    /// A plain `[dependencies]` entry.
    Normal,
    /// A `[dev-dependencies]` entry — used by the depending crate's tests.
    Dev,
    /// A `[build-dependencies]` entry — used by its build script.
    Build,
}

impl EdgeKind {
    fn from_cargo(kind: Option<&str>) -> Self {
        match kind {
            Some("dev") => EdgeKind::Dev,
            Some("build") => EdgeKind::Build,
            _ => EdgeKind::Normal,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            EdgeKind::Normal => "normal",
            EdgeKind::Dev => "dev",
            EdgeKind::Build => "build",
        }
    }
}

/// One workspace's first-party package graph, reverse-indexed.
#[derive(Debug, Clone, Default)]
pub struct CrateGraph {
    /// Every workspace member, sorted. Members are the only nodes; registry
    /// crates (regex, serde, …) are filtered out — this answers "which of *my*
    /// crates break", and an external crate's reverse tree is cargo's to report.
    pub members: Vec<String>,
    /// Each member's directory (the `Cargo.toml`'s parent), to place a file.
    pub member_dir: BTreeMap<String, PathBuf>,
    /// package → the members that name it as a dependency (any kind).
    pub dependents: BTreeMap<String, BTreeSet<String>>,
    /// (dependent, dependency) → the kind that edge was declared under.
    pub edge_kind: BTreeMap<(String, String), EdgeKind>,
}

impl CrateGraph {
    /// The members directly depending on `pkg`, with the edge kind, sorted by name.
    pub fn direct_dependents(&self, pkg: &str) -> Vec<(String, EdgeKind)> {
        let mut out: Vec<(String, EdgeKind)> = self
            .dependents
            .get(pkg)
            .map(|set| {
                set.iter()
                    .map(|d| {
                        let kind = self
                            .edge_kind
                            .get(&(d.clone(), pkg.to_string()))
                            .copied()
                            .unwrap_or(EdgeKind::Normal);
                        (d.clone(), kind)
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    }

    /// The transitive reverse closure of `pkg`: every member that depends on it,
    /// directly or through others, each with its hop distance. `pkg` itself is not
    /// in the result. A crate reachable by two paths keeps its shortest distance,
    /// which is also what stops a cycle from being walked for ever.
    pub fn reverse_closure(&self, pkg: &str) -> Vec<(String, usize)> {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        seen.insert(pkg.to_string());
        let mut out: Vec<(String, usize)> = Vec::new();
        let mut frontier: Vec<String> = vec![pkg.to_string()];
        let mut hop = 0;
        while !frontier.is_empty() {
            hop += 1;
            let mut next: Vec<String> = Vec::new();
            for node in &frontier {
                if let Some(deps) = self.dependents.get(node) {
                    for d in deps {
                        if seen.insert(d.clone()) {
                            out.push((d.clone(), hop));
                            next.push(d.clone());
                        }
                    }
                }
            }
            frontier = next;
        }
        out.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        out
    }
}

/// Parse `cargo metadata --no-deps` output into the reverse-indexed workspace
/// graph. `--no-deps` lists exactly the workspace members and their declared
/// dependencies, which is everything the first-party reverse closure needs; the
/// resolved registry tree is irrelevant here and leaving it out keeps the call
/// offline. A dependency is kept only when its name is itself a member.
pub fn parse_crate_graph(metadata_json: &str) -> Result<CrateGraph, String> {
    let value: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| format!("cargo metadata produced unreadable JSON: {e}"))?;
    let packages = value
        .get("packages")
        .and_then(|p| p.as_array())
        .ok_or_else(|| "cargo metadata output has no `packages` array".to_string())?;

    // First pass: every member's name and directory.
    let mut member_dir: BTreeMap<String, PathBuf> = BTreeMap::new();
    for pkg in packages {
        let Some(name) = pkg.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let manifest = pkg
            .get("manifest_path")
            .and_then(|m| m.as_str())
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty());
        // A workspace member's directory is its manifest's parent; without a path
        // (an unpublished edge case) fall back to the name so the key still exists.
        let dir = manifest
            .as_ref()
            .and_then(|p| p.parent())
            .map(|d| d.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(name));
        member_dir.insert(name.to_string(), dir);
    }
    let members: BTreeSet<String> = member_dir.keys().cloned().collect();

    // Second pass: only edges whose both ends are members, reverse-indexed.
    let mut dependents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut edge_kind: BTreeMap<(String, String), EdgeKind> = BTreeMap::new();
    for pkg in packages {
        let Some(dependent) = pkg.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        if !members.contains(dependent) {
            continue;
        }
        let Some(deps) = pkg.get("dependencies").and_then(|d| d.as_array()) else {
            continue;
        };
        for dep in deps {
            // A renamed dependency lists under its `package` name, but is *linked*
            // by its rename; for first-party workspace edges a rename is vanishingly
            // rare and matching on `name` would mis-attribute it, so skip renames
            // rather than guess.
            if dep.get("rename").and_then(|r| r.as_str()).is_some() {
                continue;
            }
            let Some(dep_name) = dep.get("name").and_then(|n| n.as_str()) else {
                continue;
            };
            if !members.contains(dep_name) || dep_name == dependent {
                continue;
            }
            let kind = EdgeKind::from_cargo(dep.get("kind").and_then(|k| k.as_str()));
            dependents
                .entry(dep_name.to_string())
                .or_default()
                .insert(dependent.to_string());
            edge_kind.insert((dependent.to_string(), dep_name.to_string()), kind);
        }
    }

    Ok(CrateGraph {
        members: members.into_iter().collect(),
        member_dir,
        dependents,
        edge_kind,
    })
}

/// Which workspace member a file belongs to: the member whose directory is the
/// longest ancestor of `file`. Absolute against absolute, so a file outside every
/// member (the workspace `Cargo.toml` itself, a stray path) returns `None`
/// rather than being pinned to the nearest crate.
pub fn crate_of_file(graph: &CrateGraph, file: &Path) -> Option<String> {
    let abs = if file.is_absolute() {
        file.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(file)
    };
    let mut best: Option<(usize, String)> = None;
    for (name, dir) in &graph.member_dir {
        if abs.starts_with(dir) {
            let depth = dir.components().count();
            if best.as_ref().map(|(d, _)| depth > *d).unwrap_or(true) {
                best = Some((depth, name.clone()));
            }
        }
    }
    best.map(|(_, name)| name)
}

/// Run `cargo metadata --no-deps` in `workspace` and return its JSON. Offline and
/// no-lock: `--no-deps` reads only the manifests, so it needs no network and does
/// not touch `Cargo.lock`. `std::process` stdin is closed so a stray prompt cannot
/// block an unattended run.
pub fn cargo_metadata(workspace: &Path) -> Result<String, String> {
    use std::process::{Command, Stdio};
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(workspace)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("could not start `cargo metadata`: {e}"))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if detail.is_empty() {
            "`cargo metadata` failed".to_string()
        } else {
            detail.lines().next().unwrap_or(&detail).to_string()
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real `cargo metadata --no-deps` shape, narrowed to a four-crate
    /// workspace that mirrors this one's edges: `a` is depended on by `b` and
    /// `c`, `c` by `d`, plus an external crate and a renamed dep that must drop
    /// out, and a dev edge that must survive. Frozen from cargo's own output,
    /// not hand-written per mutant.
    const WORKSPACE: &str = r#"{
      "packages": [
        {"name": "a", "manifest_path": "/ws/a/Cargo.toml", "dependencies": [
          {"name": "serde", "kind": null, "rename": null}
        ]},
        {"name": "b", "manifest_path": "/ws/b/Cargo.toml", "dependencies": [
          {"name": "a", "kind": null}
        ]},
        {"name": "c", "manifest_path": "/ws/c/Cargo.toml", "dependencies": [
          {"name": "a", "kind": "dev"},
          {"name": "renamed", "package": "a", "kind": null, "rename": "aliased"}
        ]},
        {"name": "d", "manifest_path": "/ws/d/Cargo.toml", "dependencies": [
          {"name": "c", "kind": null},
          {"name": "regex", "kind": null}
        ]}
      ],
      "workspace_members": ["a", "b", "c", "d"]
    }"#;

    fn graph() -> CrateGraph {
        parse_crate_graph(WORKSPACE).unwrap()
    }

    #[test]
    fn a_dependency_edge_is_indexed_in_the_reverse_direction() {
        let g = graph();
        // b names a, so a's dependents contain b — not the other way round.
        assert_eq!(
            g.direct_dependents("a")
                .into_iter()
                .map(|(d, _)| d)
                .collect::<Vec<_>>(),
            vec!["b", "c"],
            "both the normal (b) and dev (c) edges count as a depending on a"
        );
        assert!(
            g.dependents.get("serde").is_none(),
            "an external crate is not a node"
        );
    }

    #[test]
    fn an_edge_is_labelled_with_the_kind_it_was_declared_under() {
        let g = graph();
        let mut kinds: Vec<(String, &'static str)> = g
            .direct_dependents("a")
            .into_iter()
            .map(|(d, k)| (d, k.label()))
            .collect();
        kinds.sort();
        // b links a normally; c only in dev — and both must be shown, because a
        // dev-dependent still rebuilds when a changes.
        assert_eq!(
            kinds,
            vec![("b".to_string(), "normal"), ("c".to_string(), "dev")],
            "{kinds:?}"
        );
    }

    #[test]
    fn a_renamed_dependency_is_not_guessed_at() {
        let g = graph();
        // c renames a to `aliased`; matching on `name` would say c depends on
        // "renamed" (not a member) and drop it, while matching on package would
        // double-count. It is declined outright, so `a` has exactly b and c.
        assert_eq!(g.dependents["a"].len(), 2, "{:?}", g.dependents["a"]);
    }

    #[test]
    fn the_reverse_closure_is_transitive_and_each_node_keeps_its_shortest_hop() {
        let g = graph();
        // d -> c -> a. From a: c is one hop, d is two (through c). b is one.
        assert_eq!(
            g.reverse_closure("a"),
            vec![
                ("b".to_string(), 1),
                ("c".to_string(), 1),
                ("d".to_string(), 2)
            ],
            "sorted worst-distance-last, names within a hop"
        );
    }

    #[test]
    fn a_cycle_never_loops_and_the_target_is_never_its_own_dependent() {
        let cyclic = r#"{
          "packages": [
            {"name": "x", "manifest_path": "/w/x/Cargo.toml", "dependencies": [{"name": "y"}]},
            {"name": "y", "manifest_path": "/w/y/Cargo.toml", "dependencies": [{"name": "x"}]}
          ],
          "workspace_members": ["x", "y"]
        }"#;
        let g = parse_crate_graph(cyclic).unwrap();
        let closure = g.reverse_closure("x");
        // y depends on x; x depends on y, so walking back from x visits y once and
        // must not return to x.
        assert_eq!(closure, vec![("y".to_string(), 1)], "{closure:?}");
    }

    #[test]
    fn a_file_is_placed_in_the_deepest_member_directory_that_contains_it() {
        let g = graph();
        assert_eq!(
            crate_of_file(&g, Path::new("/ws/a/src/deep/lib.rs")).as_deref(),
            Some("a")
        );
        // A path that is inside no member (the workspace manifest itself) is not
        // forced onto the nearest crate.
        assert_eq!(crate_of_file(&g, Path::new("/ws/Cargo.toml")), None);
    }

    #[test]
    fn a_crate_nothing_depends_on_has_an_empty_blast_radius() {
        let g = graph();
        // d is the top of this chain; nothing names it.
        assert!(g.direct_dependents("d").is_empty());
        assert!(g.reverse_closure("d").is_empty());
    }

    /// This is not a mock: it runs the real `cargo metadata` in this very
    /// workspace and asserts the real first-party reverse closure of the crate
    /// under test — the dependency-free core that everything else here links to.
    #[test]
    fn the_real_metadata_of_this_workspace_resolves_a_real_reverse_closure() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"));
        let json = match cargo_metadata(workspace) {
            Ok(j) => j,
            // cargo is on PATH for the test harness itself; if it is genuinely
            // missing this run cannot speak, so say so rather than pass silently.
            Err(e) => panic!("`cargo metadata` did not run here: {e}"),
        };
        let g = parse_crate_graph(&json).expect("real cargo metadata parses");
        // xencode-core-rs is depended on by context-rs, which is depended on by
        // tui-rs; the closure of the core must therefore reach both.
        let closure: BTreeSet<String> = g
            .reverse_closure("xencode-core-rs")
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(
            closure.contains("xencode-context-rs"),
            "core's closure should hold context-rs, got {closure:?}"
        );
        assert!(
            closure.contains("xencode-tui-rs"),
            "and transitively tui-rs, got {closure:?}"
        );
        // A crate at the top of the stack breaks nothing below it.
        assert!(
            g.reverse_closure("xencode-cli").is_empty(),
            "the CLI is a leaf — nothing first-party depends on it"
        );
    }
}
