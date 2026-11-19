//! Build-graph assertions read from the real manifests at test time.
//!
//! A rule that only lives in a document drifts. These tests open the workspace
//! manifests and refuse a dependency edge that would turn the coordination
//! crate into a domain crate or pull the CLI and TUI into it.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

const OUR_CRATE: &str = "xencode-intelligence-rs";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the workspace root must exist")
}

fn manifest(workspace: &Path, crate_name: &str) -> PathBuf {
    workspace.join("crates").join(crate_name).join("Cargo.toml")
}

fn direct_dependencies(body: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_section = false;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_section = trimmed == "[dependencies]";
            continue;
        }
        if !in_section || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((name, _)) = trimmed.split_once('=') {
            names.insert(name.trim().trim_matches('"').to_string());
        }
    }
    names
}

#[test]
fn no_existing_crate_depends_on_the_coordinator() {
    let workspace = workspace_root();
    let crates = workspace.join("crates");
    let mut checked = 0;
    for entry in fs::read_dir(&crates).expect("the crates directory must be readable") {
        let path = entry.expect("a directory entry").path();
        if !path.is_dir() || path.file_name().is_some_and(|name| name == OUR_CRATE) {
            continue;
        }
        let file = path.join("Cargo.toml");
        if !file.is_file() {
            continue;
        }
        let body = fs::read_to_string(&file).expect("a crate manifest must be readable");
        assert!(
            !body.contains(OUR_CRATE),
            "{} must not depend on the coordination crate",
            file.display()
        );
        checked += 1;
    }
    assert!(
        checked >= 20,
        "expected to scan the workspace's crates, scanned {checked}"
    );
}

#[test]
fn the_coordinator_does_not_depend_on_cli_tui_or_other_domain_crates() {
    let workspace = workspace_root();
    let body = fs::read_to_string(manifest(&workspace, OUR_CRATE))
        .expect("the coordinator manifest must be readable");
    assert!(
        !body.contains("xencode-cli"),
        "the coordinator must not depend on the CLI"
    );
    assert!(
        !body.contains("xencode-tui-rs"),
        "the coordinator must not depend on the TUI"
    );

    let allowed: BTreeSet<String> = [
        "xencode-context-rs",
        "xencode-core-rs",
        "xencode-config-rs",
        "serde",
        "serde_json",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    let actual = direct_dependencies(&body);
    assert_eq!(actual, allowed, "the coordinator's dependency set changed");

    // Every listed path must point at a crate that exists, so no dependency
    // edge is a stale string that happens not to form a cycle.
    for line in body.lines().filter(|line| line.contains("path = \"../")) {
        let relative = line
            .split("path = \"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .expect("a path dependency must have a path");
        assert!(
            workspace
                .join("crates")
                .join(relative.trim_start_matches("../"))
                .is_dir(),
            "dependency path {relative} must exist"
        );
    }
}

#[test]
fn the_workspace_lists_the_coordinator() {
    let workspace = workspace_root();
    let body = fs::read_to_string(workspace.join("Cargo.toml"))
        .expect("the workspace manifest must be readable");
    assert!(
        body.contains(&format!("crates/{OUR_CRATE}")),
        "the workspace members must include the coordination crate"
    );
    // The three crates this one reuses must not depend on it, which would make
    // a cycle. Their manifests are checked again here to pin the direction.
    for name in ["xencode-context-rs", "xencode-core-rs", "xencode-config-rs"] {
        let body = fs::read_to_string(manifest(&workspace, name))
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", name));
        assert!(
            !body.contains(OUR_CRATE),
            "{name} must not depend on the coordinator"
        );
    }
}
