//! Relationship queries over real fixture files, with declared ground truth.
//!
//! The fixtures under `fixtures/relationships/` are ordinary Rust source. Their
//! inventories and dependency edges come from the same extractor and graph
//! builder the product uses; the history comes from the same `git log` parser.
//! Nothing here is a canned answer keyed on a query string.
//!
//! Run with:
//! ```text
//! cargo test -p xencode-context-rs --test relationships -- --nocapture
//! ```

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tempfile::tempdir;
use xencode_context_rs::cochange::LOG_MARKER;
use xencode_context_rs::{
    build_graph, extract_rust_symbols, impact_evidence, init_project, parse_commit_log,
    query_relations, read_json, refresh_rust_file, symbols_json_path, DepEdge, FileEntry,
    PerFileSymbols, RefreshOutcome, Relation, RelationKind, RelationSource, RetrievalIndex,
    MASS_COMMIT_FILES, TOP_PARTNERS, XENCODE_DIR,
};

/// The logical path each checked-in fixture file is given in the test index.
/// `build_graph` resolves `crate::…` imports against this real file set.
const FIXTURE_FILES: [(&str, &str); 5] = [
    ("src/lib.rs", include_str!("fixtures/relationships/lib.rs")),
    (
        "src/traits.rs",
        include_str!("fixtures/relationships/traits.rs"),
    ),
    (
        "src/store.rs",
        include_str!("fixtures/relationships/store.rs"),
    ),
    (
        "src/service.rs",
        include_str!("fixtures/relationships/service.rs"),
    ),
    (
        "src/view.rs",
        include_str!("fixtures/relationships/view.rs"),
    ),
];

fn fixture_symbols() -> BTreeMap<String, PerFileSymbols> {
    FIXTURE_FILES
        .iter()
        .map(|(path, text)| ((*path).to_string(), extract_rust_symbols(text)))
        .collect()
}

fn fixture_index() -> RetrievalIndex {
    let symbols = fixture_symbols();
    let rust_files: Vec<String> = FIXTURE_FILES.iter().map(|(p, _)| p.to_string()).collect();
    let deps = build_graph(&rust_files, &symbols);
    // A real `git log` shape, parsed by the same function `mine_commit_history`
    // uses. Three tiny commits: store+service, store+lib, view+traits.
    let log = format!(
        "{m}1000\n\nsrc/store.rs\nsrc/service.rs\n\
         {m}2000\n\nsrc/store.rs\nsrc/lib.rs\n\
         {m}3000\n\nsrc/view.rs\nsrc/traits.rs\n",
        m = LOG_MARKER
    );
    let history = parse_commit_log(&log, LOG_MARKER, MASS_COMMIT_FILES, TOP_PARTNERS);
    RetrievalIndex {
        files: FIXTURE_FILES
            .iter()
            .map(|(path, _)| FileEntry {
                path: path.to_string(),
                language: "rust".to_string(),
                size: 0,
                loc: 0,
                ext: "rs".to_string(),
                important: false,
                secret: false,
                binary: false,
            })
            .collect(),
        symbols,
        deps,
        mtimes: BTreeMap::new(),
        history,
    }
}

fn objects(relations: &[Relation]) -> Vec<String> {
    relations
        .iter()
        .filter(|relation| !relation.is_unknown())
        .map(|relation| relation.object.clone())
        .collect()
}

/// Curated ground truth. Counts are declared here, independently of the
/// derivation: the fixture source is small enough to read the answers off it.
const EXPECTED_RELATIONS: usize = 23;
const EXPECTED_UNKNOWN_ANSWERS: usize = 5;

struct CuratedCase {
    kind: RelationKind,
    subject: &'static str,
    expected: &'static [&'static str],
}

fn curated_cases() -> Vec<CuratedCase> {
    vec![
        CuratedCase {
            kind: RelationKind::TestFor,
            subject: "save_persists_value",
            expected: &["save"],
        },
        CuratedCase {
            kind: RelationKind::TestFor,
            subject: "load_missing_returns_none",
            expected: &["load"],
        },
        CuratedCase {
            kind: RelationKind::TestFor,
            subject: "no_such_test",
            expected: &[],
        },
        CuratedCase {
            kind: RelationKind::Implements,
            subject: "src/store.rs",
            expected: &["Store"],
        },
        CuratedCase {
            kind: RelationKind::Implements,
            subject: "src/view.rs",
            expected: &["Render"],
        },
        CuratedCase {
            kind: RelationKind::Implements,
            subject: "src/lib.rs",
            expected: &[],
        },
        CuratedCase {
            kind: RelationKind::DependedOnBy,
            subject: "src/store.rs",
            expected: &["src/lib.rs", "src/service.rs"],
        },
        CuratedCase {
            kind: RelationKind::DependedOnBy,
            subject: "src/traits.rs",
            expected: &[
                "src/lib.rs",
                "src/service.rs",
                "src/store.rs",
                "src/view.rs",
            ],
        },
        CuratedCase {
            kind: RelationKind::DependedOnBy,
            subject: "src/lib.rs",
            expected: &[],
        },
        CuratedCase {
            kind: RelationKind::Defines,
            subject: "save",
            expected: &["src/store.rs"],
        },
        CuratedCase {
            kind: RelationKind::Defines,
            subject: "load",
            expected: &["src/store.rs"],
        },
        CuratedCase {
            kind: RelationKind::Defines,
            subject: "MemoryStore",
            expected: &["src/store.rs"],
        },
        CuratedCase {
            kind: RelationKind::Defines,
            subject: "Store",
            expected: &["src/traits.rs"],
        },
        CuratedCase {
            kind: RelationKind::Defines,
            subject: "Render",
            expected: &["src/traits.rs"],
        },
        CuratedCase {
            kind: RelationKind::Defines,
            subject: "sync",
            expected: &["src/service.rs"],
        },
        CuratedCase {
            kind: RelationKind::Defines,
            subject: "TextView",
            expected: &["src/view.rs"],
        },
        CuratedCase {
            kind: RelationKind::Defines,
            subject: "persist",
            expected: &["src/store.rs", "src/traits.rs"],
        },
        CuratedCase {
            kind: RelationKind::Defines,
            subject: "not_declared_anywhere",
            expected: &[],
        },
        CuratedCase {
            kind: RelationKind::CoChangesWith,
            subject: "src/store.rs",
            expected: &["src/lib.rs", "src/service.rs"],
        },
        CuratedCase {
            kind: RelationKind::CoChangesWith,
            subject: "src/service.rs",
            expected: &["src/store.rs"],
        },
        CuratedCase {
            kind: RelationKind::CoChangesWith,
            subject: "src/view.rs",
            expected: &["src/traits.rs"],
        },
        CuratedCase {
            kind: RelationKind::CoChangesWith,
            subject: "src/never_touched.rs",
            expected: &[],
        },
    ]
}

#[test]
fn fixture_inventories_pin_known_symbols_tests_and_edges() {
    let symbols = fixture_symbols();
    let store = &symbols["src/store.rs"];
    assert_eq!(store.structs, vec!["MemoryStore"]);
    assert_eq!(store.traits, Vec::<String>::new());
    assert_eq!(store.impls, vec!["Store"]);
    assert_eq!(
        store.tests,
        vec!["load_missing_returns_none", "save_persists_value"]
    );
    assert_eq!(
        store.functions,
        vec![
            "load",
            "load_missing_returns_none",
            "persist",
            "save",
            "save_persists_value",
        ]
    );
    assert_eq!(symbols["src/traits.rs"].traits, vec!["Render", "Store"]);
    assert_eq!(symbols["src/view.rs"].structs, vec!["TextView"]);
    assert_eq!(symbols["src/view.rs"].impls, vec!["Render"]);
    assert_eq!(symbols["src/service.rs"].functions, vec!["sync"]);

    let index = fixture_index();
    let has = |from: &str, to: &str, via: &str| {
        index
            .deps
            .iter()
            .any(|edge| edge.from == from && edge.to == to && edge.via == via)
    };
    assert!(has("src/lib.rs", "src/store.rs", "mod store"));
    assert!(has("src/store.rs", "src/traits.rs", "crate::traits"));
    assert!(has("src/store.rs", "src/traits.rs", "impl Store"));
    assert!(has("src/view.rs", "src/traits.rs", "impl Render"));
    assert!(index
        .deps
        .iter()
        .any(|edge: &DepEdge| edge.from == "src/service.rs" && edge.to == "src/store.rs"));
    assert_eq!(index.history["src/store.rs"].partners.len(), 2);
}

#[test]
fn known_relations_are_found_and_missing_ones_are_unknown() {
    let index = fixture_index();
    for case in curated_cases() {
        let relations = query_relations(&index, case.kind, case.subject);
        let mut expected: Vec<&str> = case.expected.to_vec();
        expected.sort_unstable();
        if expected.is_empty() {
            assert_eq!(relations.len(), 1, "{:?} {}", case.kind, case.subject);
            assert!(
                relations[0].is_unknown(),
                "{:?} {}",
                case.kind,
                case.subject
            );
            assert_eq!(relations[0].source, RelationSource::Unknown);
            continue;
        }
        assert!(relations.iter().all(|r| !r.is_unknown()), "{relations:?}");
        assert!(relations
            .iter()
            .all(|r| r.source != RelationSource::Unknown));
        let mut found = objects(&relations);
        found.sort();
        assert_eq!(
            found, expected,
            "{:?} {}: {relations:?}",
            case.kind, case.subject
        );
        // Every returned row carries a real source and a reason.
        for relation in &relations {
            assert!(!relation.evidence.is_empty());
            assert!(!relation.subject.is_empty());
        }
    }
}

#[test]
fn impact_evidence_combines_test_dependent_and_cochange_relations() {
    let index = fixture_index();
    let evidence = impact_evidence(&index, "src/store.rs");
    assert_eq!(evidence.changed_file, "src/store.rs");

    let tests: Vec<&str> = evidence
        .tests
        .iter()
        .filter(|r| !r.is_unknown())
        .map(|r| r.subject.as_str())
        .collect();
    assert_eq!(
        tests,
        vec!["load_missing_returns_none", "save_persists_value"]
    );

    let dependents: HashSet<&str> = evidence
        .dependents
        .iter()
        .filter(|r| !r.is_unknown())
        .map(|r| r.object.as_str())
        .collect();
    assert_eq!(dependents, HashSet::from(["src/lib.rs", "src/service.rs"]));

    let partners: HashSet<&str> = evidence
        .co_changes
        .iter()
        .filter(|r| !r.is_unknown())
        .map(|r| r.object.as_str())
        .collect();
    assert_eq!(partners, HashSet::from(["src/lib.rs", "src/service.rs"]));

    // Every constituent is still a provenance-bearing relation.
    assert!(evidence.all().iter().all(|r| !r.evidence.is_empty()));
    assert!(evidence
        .all()
        .iter()
        .all(|r| r.source == RelationSource::Unknown || !r.object.starts_with('<')));

    // A file with no tests, no dependents and no history reports three unknowns
    // rather than an empty success.
    let bare = impact_evidence(&index, "src/never_touched.rs");
    assert_eq!(bare.unknown.len(), 3, "{bare:?}");
    assert!(bare.is_empty());
}

#[test]
fn curated_relations_precision_and_recall() {
    let index = fixture_index();
    let mut true_positive = 0usize;
    let mut false_positive = 0usize;
    let mut false_negative = 0usize;
    let mut correct_rejections = 0usize;

    for case in curated_cases() {
        let relations = query_relations(&index, case.kind, case.subject);
        let found: HashSet<&str> = relations
            .iter()
            .filter(|r| !r.is_unknown())
            .map(|r| r.object.as_str())
            .collect();
        let expected: HashSet<&str> = case.expected.iter().copied().collect();
        true_positive += found.intersection(&expected).count();
        false_positive += found.difference(&expected).count();
        false_negative += expected.difference(&found).count();
        if expected.is_empty()
            && relations.len() == 1
            && relations[0].is_unknown()
            && relations[0].source == RelationSource::Unknown
        {
            correct_rejections += 1;
        }
    }

    let declared_expected: usize = curated_cases().iter().map(|case| case.expected.len()).sum();
    let declared_rejections = curated_cases()
        .iter()
        .filter(|case| case.expected.is_empty())
        .count();
    assert_eq!(declared_expected, EXPECTED_RELATIONS);
    assert_eq!(declared_rejections, EXPECTED_UNKNOWN_ANSWERS);

    let precision = true_positive as f64 / (true_positive + false_positive) as f64;
    let recall = true_positive as f64 / (true_positive + false_negative) as f64;
    println!(
        "curated relationship set: {} positive relations declared, {} must-be-unknown queries",
        EXPECTED_RELATIONS, EXPECTED_UNKNOWN_ANSWERS
    );
    println!(
        "derived: TP={true_positive} FP={false_positive} FN={false_negative}; \
         correct Unknown answers={correct_rejections}/{EXPECTED_UNKNOWN_ANSWERS}"
    );
    println!(
        "measured precision={precision:.3} recall={recall:.3} \
         (95% precision is a release gate on this fixture set, not a claim about other repositories)"
    );

    assert_eq!(
        true_positive, EXPECTED_RELATIONS,
        "missed a declared relation"
    );
    assert_eq!(false_positive, 0, "invented a relation");
    assert_eq!(false_negative, 0, "missed a declared relation");
    assert_eq!(correct_rejections, EXPECTED_UNKNOWN_ANSWERS);
    assert!(
        precision >= 0.95,
        "95% precision release gate not met: {precision:.3}"
    );
}

#[test]
fn refresh_invalidates_affected_relations() {
    let dir = tempdir().expect("temp workspace");
    let root = dir.path();
    fs::create_dir_all(root.join("src")).expect("src dir");
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("Cargo.toml");
    for (path, text) in FIXTURE_FILES {
        fs::write(root.join(path), text).expect("fixture file");
    }
    init_project(root, Arc::new(AtomicBool::new(false)), |_| {}).expect("init");
    let load = || RetrievalIndex::load(&root.join(XENCODE_DIR)).expect("index");
    let index = load();

    let defined = query_relations(&index, RelationKind::Defines, "save");
    assert_eq!(objects(&defined), vec!["src/store.rs"]);
    let tested = query_relations(&index, RelationKind::TestFor, "save_persists_value");
    assert_eq!(objects(&tested), vec!["save"]);
    let service_dependents = objects(&query_relations(
        &index,
        RelationKind::DependedOnBy,
        "src/store.rs",
    ));
    assert!(service_dependents.contains(&"src/service.rs".to_string()));

    // Rename the function and its test on disk, then refresh the one file.
    let edited = include_str!("fixtures/relationships/store.rs")
        .replace("pub fn save(", "pub fn store_value(")
        .replace("fn save_persists_value()", "fn store_value_writes()")
        .replace("save(&store);", "store_value(&store);");
    fs::write(root.join("src/store.rs"), edited).expect("edited store.rs");
    assert!(matches!(
        refresh_rust_file(root, "src/store.rs").expect("refresh"),
        RefreshOutcome::Updated(_)
    ));

    let index = load();
    let old_symbol = query_relations(&index, RelationKind::Defines, "save");
    assert!(
        old_symbol[0].is_unknown(),
        "stale symbol survived: {old_symbol:?}"
    );
    let new_symbol = query_relations(&index, RelationKind::Defines, "store_value");
    assert_eq!(objects(&new_symbol), vec!["src/store.rs"]);
    let old_test = query_relations(&index, RelationKind::TestFor, "save_persists_value");
    assert!(
        old_test[0].is_unknown(),
        "stale test survived: {old_test:?}"
    );
    let new_test = query_relations(&index, RelationKind::TestFor, "store_value_writes");
    assert_eq!(objects(&new_test), vec!["store_value"]);

    // Dropping the consumer's import removes its dependent edge too.
    let edited_service = include_str!("fixtures/relationships/service.rs")
        .replace("use crate::store::{load, save};\n\n", "");
    fs::write(root.join("src/service.rs"), edited_service).expect("edited service.rs");
    assert!(matches!(
        refresh_rust_file(root, "src/service.rs").expect("refresh"),
        RefreshOutcome::Updated(_)
    ));
    // The symbol map on disk must reflect the first refresh as well.
    let on_disk: BTreeMap<String, PerFileSymbols> =
        read_json(&symbols_json_path(&root.join(XENCODE_DIR))).expect("symbols.json");
    assert!(on_disk["src/store.rs"]
        .functions
        .contains(&"store_value".to_string()));

    let index = load();
    let dependents = objects(&query_relations(
        &index,
        RelationKind::DependedOnBy,
        "src/store.rs",
    ));
    assert!(
        !dependents.contains(&"src/service.rs".to_string()),
        "stale dependent survived: {dependents:?}"
    );
    assert!(dependents.contains(&"src/lib.rs".to_string()));
}
