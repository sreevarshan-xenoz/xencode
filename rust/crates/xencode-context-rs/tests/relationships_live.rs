//! Relationship queries against this repository's own index.
//!
//! Ignored by default because it builds the real `.xencode/` snapshot (the
//! same way `repo_map_live` does). It checks that the derived relations point
//! at real files in this workspace and prints what the evidence looks like;
//! it is a smoke check on real data, not a precision claim.
//!
//! ```text
//! cargo test -p xencode-context-rs --test relationships_live -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use xencode_context_rs::{
    impact_evidence, init_project, query_relations, RelationKind, RetrievalIndex, XENCODE_DIR,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("crate sits at <root>/rust/crates/<name>")
        .to_path_buf()
}

#[test]
#[ignore]
fn relationships_on_this_repository_resolve_to_real_files() {
    let root = workspace_root();
    init_project(&root, Arc::new(AtomicBool::new(false)), |_| {})
        .expect("init on the real workspace");
    let index = RetrievalIndex::load(&root.join(XENCODE_DIR)).expect("index on disk");
    assert!(!index.is_empty());

    for (symbol, file) in [
        (
            "extract_rust_symbols",
            "rust/crates/xencode-context-rs/src/symbols.rs",
        ),
        (
            "rank_repo_map",
            "rust/crates/xencode-context-rs/src/repo_map.rs",
        ),
        (
            "query_relations",
            "rust/crates/xencode-context-rs/src/relationships.rs",
        ),
    ] {
        let derived = query_relations(&index, RelationKind::Defines, symbol);
        assert!(
            derived.iter().any(|relation| relation.object == file),
            "{symbol} should be declared in {file}, got {derived:?}"
        );
    }

    // Every non-unknown relation in the aggregator still carries provenance.
    let evidence = impact_evidence(
        &index,
        "rust/crates/xencode-context-rs/src/relationships.rs",
    );
    let known = evidence
        .all()
        .into_iter()
        .filter(|relation| !relation.is_unknown())
        .count();
    assert!(evidence
        .all()
        .into_iter()
        .all(|relation| !relation.evidence.is_empty()));
    assert!(evidence
        .unknown
        .iter()
        .all(|relation| relation.is_unknown() && relation.object == "<unknown>"));
    println!(
        "real-index impact for relationships.rs: {known} known relations, \
         {} explicit unknowns; {} index files, {} dependency edges",
        evidence.unknown.len(),
        index.files.len(),
        index.deps.len()
    );
}
