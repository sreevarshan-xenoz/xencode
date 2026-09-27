//! The impact report for this workspace, measured against the index the real
//! `/init` builds rather than a fixture. Ignored by default because it rebuilds
//! the project index (about a second) and writes it into `.xencode/`, which is
//! git-ignored but shared with the TUI — the same reason the retrieval baseline
//! in `gold_baseline.rs` is ignored.
//!
//! ```text
//! cargo test -p xencode-context-rs --test impact_baseline -- --ignored --nocapture
//! ```
//!
//! The plan item this measures asks whether naming one file surfaces the files
//! that link to it. The numbers are reported, never asserted: this repo's file
//! set changes, and a fixed threshold would turn a measurement into a fixture.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

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
fn what_links_to_a_real_file_in_this_workspace() {
    let root = workspace_root();
    xencode_context_rs::init_project(&root, std::sync::Arc::new(AtomicBool::new(false)), |_| {})
        .expect("init_project on the real workspace");

    let asked = "symbols.rs";
    let report = xencode_context_rs::impact_from_snapshot(&root, asked, None)
        .unwrap_or_else(|e| panic!("impact report for {asked}: {e}"));
    println!(
        "{} links to {}  (index: {} Rust files, {} edges)",
        report.files.len(),
        report.target,
        report.indexed_files,
        report.edges,
    );
    println!(
        "it declares {} name(s){}: {}",
        report.declared.len(),
        if report.declared_more > 0 {
            format!(" (+{} more)", report.declared_more)
        } else {
            String::new()
        },
        report.declared.join(", ")
    );
    for (hop, label) in [(1, "direct"), (2, "2 hops"), (3, "3 hops")] {
        let rows: Vec<&_> = report.files.iter().filter(|f| f.hops == hop).collect();
        println!("{label}: {} file(s)", rows.len());
        for row in rows {
            println!("  {}  via {}", row.file, row.via.join(", "));
        }
    }

    // The same question narrowed to one name from that file's own surface, which
    // is what a model asks when it is about to edit a function rather than a
    // module. Picked from what the index says the file declares, so the probe
    // cannot pass by guessing a name that is not there.
    let symbol = report
        .declared
        .iter()
        .find(|name| *name == "build_graph")
        .cloned()
        .or_else(|| report.declared.first().cloned())
        .expect("the index names at least one thing symbols.rs declares");
    let narrowed = xencode_context_rs::impact_from_snapshot(&root, asked, Some(&symbol))
        .expect("impact report narrowed to one symbol");
    let naming: Vec<&str> = narrowed
        .files
        .iter()
        .filter(|f| f.uses_symbol)
        .map(|f| f.file.as_str())
        .collect();
    println!(
        "narrowed to `{symbol}`: {} file(s) link to {}, {} of them write \
         `{symbol}` in their own `use`:",
        narrowed.files.len(),
        narrowed.target,
        naming.len()
    );
    for file in naming {
        println!("  {file}");
    }
}
