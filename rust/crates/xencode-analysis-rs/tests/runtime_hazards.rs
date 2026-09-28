//! U-2's completion condition, against the real `ast-grep` binary.
//!
//! The unit tests in the module cover the shapes and the honesty rules without
//! needing anything installed. This file covers the part that cannot be faked:
//! that the queries as written actually match the code they are meant to, and
//! that they match *only* it.
//!
//! Skipped in full when `ast-grep` is absent, which is the state this feature has
//! to survive anyway — the unavailable-engine path is covered by a unit test.

use std::path::{Path, PathBuf};
use std::time::Duration;
use xencode_analysis_rs::runtime_hazards::{
    analyze_runtime, EngineStatus, HazardClass, RuntimeHazard,
};

fn ast_grep_installed() -> bool {
    std::env::var_os("PATH")
        .map(|path| {
            std::env::split_paths(&path)
                .any(|dir| dir.join("ast-grep").is_file() || dir.join("sg").is_file())
        })
        .unwrap_or(false)
}

fn scratch(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "xencode-hazards-{label}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write(root: &Path, name: &str, body: &str) {
    let path = root.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, body).unwrap();
}

fn of_class(
    scan: &xencode_analysis_rs::runtime_hazards::RuntimeScan,
    class: HazardClass,
) -> Vec<&RuntimeHazard> {
    scan.findings.iter().filter(|f| f.class == class).collect()
}

/// The seeded file: three hazards, and three near-misses that must stay quiet.
const SEEDED: &str = r#"use std::sync::Mutex;
use std::time::Duration;

async fn reads_blocking() -> String {
    std::fs::read_to_string("a.txt").unwrap()
}

async fn sleeps_blocking() {
    std::thread::sleep(Duration::from_millis(5));
}

fn makes_a_channel() {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<u8>();
    let _ = tx;
}

fn spawns_tasks() {
    tokio::spawn(async {});
    let _tracked = tokio::spawn(async {});
    tokio::spawn(async {}).await.unwrap();
}

// A blocking call in a *sync* function is not a reactor hazard.
fn sync_reader() -> String {
    std::fs::read_to_string("b.txt").unwrap()
}

// A bounded channel is the fix, not the finding.
fn makes_a_bounded_channel() {
    let (tx, _rx) = tokio::sync::mpsc::channel::<u8>(8);
    let _ = tx;
}

async fn uses_the_async_form() -> String {
    tokio::fs::read_to_string("c.txt").await.unwrap()
}
"#;

#[test]
fn the_queries_match_the_seeded_hazards_and_quietly_ignore_the_near_misses() {
    if !ast_grep_installed() {
        eprintln!("skipping: ast-grep is not installed");
        return;
    }
    let root = scratch("seeded");
    write(&root, "code.rs", SEEDED);

    let scan = analyze_runtime(&root, Duration::from_secs(30));
    assert!(
        matches!(scan.engine, EngineStatus::Ran { .. }),
        "the engine should have run: {:?}",
        scan.engine
    );

    // The three seeded hazards.
    let blocking = of_class(&scan, HazardClass::BlockingCall);
    assert_eq!(
        blocking.len(),
        2,
        "expected read_to_string and thread::sleep, got {:?}",
        blocking
            .iter()
            .map(|f| format!("{}:{} {}", f.path, f.line, f.snippet))
            .collect::<Vec<_>>()
    );
    for finding in &blocking {
        assert_eq!(
            finding.severity,
            xencode_analysis_rs::runtime_hazards::Severity::High
        );
        assert!(!finding.test_only, "a shipped file is not test-only");
    }
    // The report names the replacement, which is the done-when's requirement.
    let report = blocking[0].to_report();
    assert!(report.contains("tokio::fs"), "{report}");
    assert!(report.contains("tokio::time::sleep"), "{report}");

    assert_eq!(of_class(&scan, HazardClass::UnboundedChannel).len(), 1);

    // The done-when's second and third cases: only the *bare* spawn is a
    // hazard. The tracked one and the awaited one must not be reported.
    let detached = of_class(&scan, HazardClass::DetachedTask);
    assert_eq!(
        detached.len(),
        1,
        "only the discarded handle is a hazard, got {:?}",
        detached
            .iter()
            .map(|f| format!("{}:{} {}", f.path, f.line, f.snippet))
            .collect::<Vec<_>>()
    );
    // And it is the bare one, on the line seeded for it.
    let spawn_line = SEEDED
        .lines()
        .position(|l| l.trim() == "tokio::spawn(async {});")
        .expect("seeded line")
        + 1;
    assert_eq!(detached[0].line, spawn_line);
    let report = detached[0].to_report();
    assert!(
        report.contains("deliberate"),
        "no way to say it is on purpose: {report}"
    );
    assert!(report.contains("await"), "no way to fix it: {report}");

    // The near-misses, stated explicitly so a regression names itself. The line
    // numbers are computed from SEEDED rather than written down, so this file
    // and the assertion cannot drift apart.
    let line_of = |needle: &str| -> usize {
        SEEDED
            .lines()
            .position(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("{needle} is not in the seeded file"))
            + 1
    };
    assert!(
        !blocking
            .iter()
            .any(|f| f.line == line_of("read_to_string(\"b.txt\")")),
        "a blocking call in a sync fn is not a reactor hazard"
    );
    assert!(
        blocking
            .iter()
            .any(|f| f.line == line_of("Duration::from_millis(5)")),
        "a thread::sleep inside an async fn is a hazard and must be reported"
    );
    assert!(
        !of_class(&scan, HazardClass::UnboundedChannel)
            .iter()
            .any(|f| f.snippet.contains("channel::<u8>(8)")),
        "a bounded channel is the fix, not the finding"
    );
    assert!(
        !blocking.iter().any(|f| f.snippet.contains("tokio::fs")),
        "the async form is not a hazard"
    );

    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_clean_file_proves_clean_rather_than_reporting_nothing() {
    if !ast_grep_installed() {
        eprintln!("skipping: ast-grep is not installed");
        return;
    }
    let root = scratch("clean");
    write(
        &root,
        "fine.rs",
        "async fn fine(p: &str) -> String {\n    tokio::fs::read_to_string(p).await.unwrap()\n}\n",
    );
    let scan = analyze_runtime(&root, Duration::from_secs(30));
    assert!(scan.proved_clean(), "{:?}", scan.findings);
    assert!(scan.summary().contains("none found"), "{}", scan.summary());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn a_test_only_hazard_is_reported_and_labelled() {
    if !ast_grep_installed() {
        eprintln!("skipping: ast-grep is not installed");
        return;
    }
    let root = scratch("cfgtest");
    write(
        &root,
        "code.rs",
        "#[cfg(test)]\nmod tests {\n    async fn t() -> String {\n        \
         std::fs::read_to_string(\"a.txt\").unwrap()\n    }\n}\n",
    );
    let scan = analyze_runtime(&root, Duration::from_secs(30));
    assert_eq!(scan.findings.len(), 1, "{:?}", scan.findings);
    assert!(
        scan.findings[0].test_only,
        "a finding inside #[cfg(test)] must say so"
    );
    assert!(
        scan.findings[0].to_report().contains("#[cfg(test)] module"),
        "{}",
        scan.findings[0].to_report()
    );
    std::fs::remove_dir_all(&root).unwrap();
}

/// Two false positives this tool produced on this repository's own source before
/// the rule set was corrected. Both are correct code, and a check that flags
/// correct code gets switched off.
#[test]
fn correct_code_is_not_reported() {
    if !ast_grep_installed() {
        eprintln!("skipping: ast-grep is not installed");
        return;
    }
    let root = scratch("correct");
    write(
        &root,
        "code.rs",
        // A blocking call inside spawn_blocking is the right place for it. This
        // is `run_profiler`'s actual shape.
        "async fn sampled() -> String {\n    let v = tokio::task::spawn_blocking(move || {\n        \
         std::thread::sleep(std::time::Duration::from_millis(5));\n        \"x\".to_string()\n    });\n    \
         v.await\n}\n\
         // std::thread::spawn starts a new OS thread and returns, so it never
         // blocks the reactor whatever it is called from.\n\
         async fn fans_out() {\n    std::thread::spawn(|| ());\n}\n\
         // A spawn bound to a name is tracked, not detached.\n\
         async fn tracked() {\n    let _h = tokio::spawn(async {});\n}\n",
    );

    let scan = analyze_runtime(&root, Duration::from_secs(30));
    assert!(
        scan.proved_clean(),
        "correct code was reported: {:?}",
        scan.findings
            .iter()
            .map(|f| format!("{}:{} {}", f.path, f.line, f.snippet))
            .collect::<Vec<_>>()
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn findings_are_reported_in_a_stable_order() {
    if !ast_grep_installed() {
        eprintln!("skipping: ast-grep is not installed");
        return;
    }
    let root = scratch("order");
    for name in ["b.rs", "a.rs"] {
        write(
            &root,
            name,
            "async fn r() -> String {\n    std::fs::read_to_string(\"a.txt\").unwrap()\n}\n",
        );
    }
    let first = analyze_runtime(&root, Duration::from_secs(30));
    let second = analyze_runtime(&root, Duration::from_secs(30));
    let order = |s: &xencode_analysis_rs::runtime_hazards::RuntimeScan| {
        s.findings
            .iter()
            .map(|f| format!("{}:{}", f.path, f.line))
            .collect::<Vec<_>>()
    };
    assert_eq!(order(&first), order(&second), "two scans disagreed");
    // a.rs before b.rs: the order is by path, not by which rule ran first.
    assert_eq!(
        order(&first),
        vec!["a.rs:2", "b.rs:2"],
        "{:?}",
        order(&first)
    );
    std::fs::remove_dir_all(&root).unwrap();
}
