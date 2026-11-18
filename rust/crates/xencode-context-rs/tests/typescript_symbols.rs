//! The TypeScript lexical symbol tier: exact inventories from real source files
//! and the names that reach `symbols.json` through `/init`.

use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tempfile::tempdir;
use xencode_context_rs::{
    deps_json_path, extract_typescript_symbols, init_project, read_json, refresh_rust_file,
    symbols_json_path, DepEdge, PerFileSymbols, RefreshOutcome,
};

#[test]
fn tssymbols_extracts_exact_sorted_sets_from_ts_fixture() {
    let symbols = extract_typescript_symbols(include_str!("fixtures/sample.ts"));
    assert_eq!(symbols.structs, vec!["BaseWidget", "Widget"]);
    assert_eq!(symbols.traits, vec!["WidgetOptions"]);
    assert_eq!(symbols.enums, vec!["WidgetState"]);
    assert_eq!(symbols.types, vec!["WidgetId"]);
    assert_eq!(
        symbols.functions,
        vec![
            "bindWidget",
            "caption",
            "constructor",
            "create",
            "legacyWidget",
            "makeWidget",
            "onClick",
            "refresh",
            "render",
        ]
    );
    assert_eq!(
        symbols.imports,
        vec!["./helper", "config", "side-effect-polyfill"]
    );
    assert_eq!(
        symbols.exports,
        vec![
            "BaseWidget",
            "Widget",
            "WidgetId",
            "WidgetOptions",
            "WidgetState",
            "bindWidget",
            "buildWidget",
            "default",
            "legacyWidget",
            "makeWidget",
        ]
    );
    assert_eq!(
        symbols.docs,
        "A widget that renders a label. Called when the widget is clicked."
    );
    // Rust-only fields stay empty: this tier reports names, never module edges
    // or symbol-level edits.
    assert!(symbols.mods.is_empty());
    assert!(symbols.impls.is_empty());
    assert!(symbols.tests.is_empty());

    // The same text must always give the same inventory, byte for byte.
    assert_eq!(
        symbols,
        extract_typescript_symbols(include_str!("fixtures/sample.ts"))
    );
}

#[test]
fn tssymbols_ignores_keywords_in_comments_strings_and_templates() {
    let symbols = extract_typescript_symbols(include_str!("fixtures/decoys.ts"));
    assert_eq!(symbols.functions, vec!["realFn"]);
    assert_eq!(symbols.exports, vec!["realFn"]);
    assert_eq!(symbols.docs, "A real declaration.");
    assert!(symbols.structs.is_empty(), "{:?}", symbols.structs);
    assert!(symbols.traits.is_empty(), "{:?}", symbols.traits);
    assert!(symbols.enums.is_empty(), "{:?}", symbols.enums);
    assert!(symbols.types.is_empty(), "{:?}", symbols.types);
    assert!(symbols.imports.is_empty(), "{:?}", symbols.imports);
}

#[test]
fn tssymbols_extracts_tsx_component_symbols() {
    let symbols = extract_typescript_symbols(include_str!("fixtures/component.tsx"));
    assert_eq!(symbols.structs, vec!["ButtonGroup"]);
    assert_eq!(symbols.traits, vec!["ButtonProps"]);
    assert_eq!(symbols.functions, vec!["Button", "onClick", "render"]);
    assert_eq!(symbols.imports, vec!["react"]);
    assert_eq!(
        symbols.exports,
        vec!["Button", "ButtonGroup", "ButtonProps"]
    );
    assert_eq!(symbols.docs, "A button with a label.");
}

#[test]
fn index_contains_typescript_symbol_names_and_no_rust_edges_from_them() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    fs::write(root.join("widget.ts"), include_str!("fixtures/sample.ts")).unwrap();
    fs::write(
        root.join("component.tsx"),
        include_str!("fixtures/component.tsx"),
    )
    .unwrap();

    let summary = init_project(root, Arc::new(AtomicBool::new(false)), |_| {}).unwrap();
    assert_eq!(summary.symbol_files, 2);

    let xencode = root.join(".xencode");
    let index: BTreeMap<String, PerFileSymbols> = read_json(&symbols_json_path(&xencode)).unwrap();
    assert_eq!(index.len(), 2);
    let widget = &index["widget.ts"];
    assert!(widget.structs.iter().any(|name| name == "BaseWidget"));
    assert!(widget.traits.iter().any(|name| name == "WidgetOptions"));
    assert!(widget.functions.iter().any(|name| name == "render"));
    let component = &index["component.tsx"];
    assert!(component.functions.iter().any(|name| name == "Button"));

    let deps: Vec<DepEdge> = read_json(&deps_json_path(&xencode)).unwrap();
    assert!(
        deps.is_empty(),
        "TypeScript imports must not enter the Rust resolver: {deps:?}"
    );
}

#[test]
fn refresh_updates_typescript_symbols_without_changing_rust_edges() {
    let dir = tempdir().unwrap();
    let root = dir.path();
    fs::write(root.join("app.ts"), "export function first() {}\n").unwrap();
    fs::write(root.join("lib.rs"), "pub fn rusty() {}\n").unwrap();
    init_project(root, Arc::new(AtomicBool::new(false)), |_| {}).unwrap();

    fs::write(
        root.join("app.ts"),
        "export class Replaced {}\nexport function second() {}\n",
    )
    .unwrap();
    assert!(matches!(
        refresh_rust_file(root, "app.ts").unwrap(),
        RefreshOutcome::Updated(_)
    ));

    let xencode = root.join(".xencode");
    let index: BTreeMap<String, PerFileSymbols> = read_json(&symbols_json_path(&xencode)).unwrap();
    assert_eq!(index["app.ts"].structs, vec!["Replaced"]);
    assert_eq!(index["app.ts"].functions, vec!["second"]);
    let deps: Vec<DepEdge> = read_json(&deps_json_path(&xencode)).unwrap();
    assert!(deps.is_empty(), "{deps:?}");
}
