//! VF-4-B evaluation: does the VF-4-A observation survive a change in shape?
//!
//! Target: `covdiff::intersect`, which crosses added lines with executed lines.
//! Same discipline as VF-4-A: four seeded defects, each breaking exactly one
//! rule, scored by seeded defects caught over seeded defects introduced.
//!
//! The four defects:
//!
//! - unknown lines reported as uncovered (collapses the tool's blind spot into
//!   a verdict about the code);
//! - unknown lines reported as covered (counts what was never measured as ran);
//! - files with no coverage data dropped silently (no entry, no listing, no
//!   note);
//! - a line counted as covered when it ran in *any* file, not its own (file
//!   misattribution — the shape of the macro/derive trap the plan names).

use crate::covdiff::{intersect, DiffCoverage};
use proptest::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

type IntersectFn =
    fn(&BTreeMap<String, Vec<u32>>, &BTreeMap<String, Vec<u32>>, &mut Vec<String>) -> DiffCoverage;

fn correct(
    added: &BTreeMap<String, Vec<u32>>,
    hits: &BTreeMap<String, Vec<u32>>,
    notes: &mut Vec<String>,
) -> DiffCoverage {
    intersect(added, hits, notes)
}

fn finish(out: &mut DiffCoverage, unknown_files: Vec<String>, notes: &mut Vec<String>) {
    if !unknown_files.is_empty() {
        notes.push(format!(
            "{} changed file(s) have no coverage data",
            unknown_files.len()
        ));
    }
    out.unmeasured = unknown_files;
    out.unmeasured_lines = out
        .files
        .iter()
        .filter(|f| out.unmeasured.contains(&f.file))
        .map(|f| f.unknown.len() as u64)
        .sum();
}

fn collapse_unknown_to_uncovered(
    added: &BTreeMap<String, Vec<u32>>,
    hits: &BTreeMap<String, Vec<u32>>,
    notes: &mut Vec<String>,
) -> DiffCoverage {
    let mut out = DiffCoverage::default();
    let mut unknown_files = Vec::new();
    for (file, lines) in added {
        let mut result = crate::covdiff::FileCoverage {
            file: file.clone(),
            ..Default::default()
        };
        match hits.get(file) {
            None => {
                result.uncovered = lines.clone();
                unknown_files.push(file.clone());
            }
            Some(covered) => {
                for line in lines {
                    if covered.contains(line) {
                        result.covered.push(*line);
                    } else {
                        result.uncovered.push(*line);
                    }
                }
            }
        }
        out.files.push(result);
    }
    finish(&mut out, unknown_files, notes);
    out
}

fn count_unknown_as_covered(
    added: &BTreeMap<String, Vec<u32>>,
    hits: &BTreeMap<String, Vec<u32>>,
    notes: &mut Vec<String>,
) -> DiffCoverage {
    let mut out = DiffCoverage::default();
    let mut unknown_files = Vec::new();
    for (file, lines) in added {
        let mut result = crate::covdiff::FileCoverage {
            file: file.clone(),
            ..Default::default()
        };
        match hits.get(file) {
            None => {
                result.covered = lines.clone();
                unknown_files.push(file.clone());
            }
            Some(covered) => {
                for line in lines {
                    if covered.contains(line) {
                        result.covered.push(*line);
                    } else {
                        result.uncovered.push(*line);
                    }
                }
            }
        }
        out.files.push(result);
    }
    finish(&mut out, unknown_files, notes);
    out
}

fn drop_unmeasured(
    added: &BTreeMap<String, Vec<u32>>,
    hits: &BTreeMap<String, Vec<u32>>,
    notes: &mut Vec<String>,
) -> DiffCoverage {
    let _ = notes;
    let mut out = DiffCoverage::default();
    for (file, lines) in added {
        let Some(covered) = hits.get(file) else {
            continue;
        };
        let mut result = crate::covdiff::FileCoverage {
            file: file.clone(),
            ..Default::default()
        };
        for line in lines {
            if covered.contains(line) {
                result.covered.push(*line);
            } else {
                result.uncovered.push(*line);
            }
        }
        out.files.push(result);
    }
    out
}

fn cover_from_any_file(
    added: &BTreeMap<String, Vec<u32>>,
    hits: &BTreeMap<String, Vec<u32>>,
    notes: &mut Vec<String>,
) -> DiffCoverage {
    let mut out = DiffCoverage::default();
    let mut unknown_files = Vec::new();
    for (file, lines) in added {
        let mut result = crate::covdiff::FileCoverage {
            file: file.clone(),
            ..Default::default()
        };
        match hits.get(file) {
            None => {
                result.unknown = lines.clone();
                unknown_files.push(file.clone());
            }
            Some(own) => {
                for line in lines {
                    let elsewhere = hits.values().any(|other| other.contains(line));
                    if own.contains(line) || elsewhere {
                        result.covered.push(*line);
                    } else {
                        result.uncovered.push(*line);
                    }
                }
            }
        }
        out.files.push(result);
    }
    finish(&mut out, unknown_files, notes);
    out
}

fn rich_fixture() -> (BTreeMap<String, Vec<u32>>, BTreeMap<String, Vec<u32>>) {
    let added: BTreeMap<String, Vec<u32>> = [
        ("src/a.rs".to_string(), vec![10, 11]),
        ("src/b.rs".to_string(), vec![10]),
        ("notes.md".to_string(), vec![1]),
    ]
    .into_iter()
    .collect();
    let hits: BTreeMap<String, Vec<u32>> = [
        ("src/a.rs".to_string(), vec![10, 12]),
        ("src/b.rs".to_string(), vec![]),
    ]
    .into_iter()
    .collect();
    (added, hits)
}

const VARIANTS: [(&str, IntersectFn); 4] = [
    ("unknown as uncovered", collapse_unknown_to_uncovered),
    ("unknown as covered", count_unknown_as_covered),
    ("drop unmeasured", drop_unmeasured),
    ("cover from any file", cover_from_any_file),
];

fn p0_vacuous(output: &DiffCoverage, _added: &BTreeMap<String, Vec<u32>>) -> bool {
    !output.summary().is_empty()
}

fn p1_weak(output: &DiffCoverage, added: &BTreeMap<String, Vec<u32>>) -> bool {
    added
        .keys()
        .all(|f| output.files.iter().any(|r| &r.file == f) || output.unmeasured.contains(f))
}

fn p2_candidate(
    output: &DiffCoverage,
    added: &BTreeMap<String, Vec<u32>>,
    hits: &BTreeMap<String, Vec<u32>>,
) -> bool {
    for (file, lines) in added {
        let Some(result) = output.files.iter().find(|r| &r.file == file) else {
            return false;
        };
        let covered: BTreeSet<u32> = result.covered.iter().copied().collect();
        let uncovered: BTreeSet<u32> = result.uncovered.iter().copied().collect();
        let unknown: BTreeSet<u32> = result.unknown.iter().copied().collect();
        let added_set: BTreeSet<u32> = lines.iter().copied().collect();
        if !covered.is_disjoint(&uncovered) || !covered.is_disjoint(&unknown) {
            return false;
        }
        if !covered.is_subset(&added_set) || !uncovered.is_subset(&added_set) {
            return false;
        }
        // Unknown is exactly the added lines no data exists for: everything
        // when the file is absent from the report, nothing when present.
        let expected_unknown: BTreeSet<u32> = if hits.contains_key(file) {
            BTreeSet::new()
        } else {
            added_set
        };
        if unknown != expected_unknown {
            return false;
        }
    }
    true
}

fn expected_exact(
    added: &BTreeMap<String, Vec<u32>>,
    hits: &BTreeMap<String, Vec<u32>>,
) -> DiffCoverage {
    intersect(added, hits, &mut Vec::new())
}

fn p3_exact(
    output: &DiffCoverage,
    added: &BTreeMap<String, Vec<u32>>,
    hits: &BTreeMap<String, Vec<u32>>,
) -> bool {
    let mut expected = expected_exact(added, hits);
    let mut actual = output.clone();
    expected.files.sort_by(|a, b| a.file.cmp(&b.file));
    actual.files.sort_by(|a, b| a.file.cmp(&b.file));
    actual.files == expected.files
        && actual.unmeasured == expected.unmeasured
        && actual.unmeasured_lines == expected.unmeasured_lines
}

fn catches_p01(
    predicate: fn(&DiffCoverage, &BTreeMap<String, Vec<u32>>) -> bool,
    variant: IntersectFn,
) -> bool {
    let (added, hits) = rich_fixture();
    !predicate(&variant(&added, &hits, &mut Vec::new()), &added)
}

fn catches_p2(variant: IntersectFn) -> bool {
    let (added, hits) = rich_fixture();
    !p2_candidate(&variant(&added, &hits, &mut Vec::new()), &added, &hits)
}

fn catches_p3(variant: IntersectFn) -> bool {
    let (added, hits) = rich_fixture();
    !p3_exact(&variant(&added, &hits, &mut Vec::new()), &added, &hits)
}

fn file_names() -> impl Strategy<Value = String> {
    prop::sample::select(vec![
        "src/a.rs".to_string(),
        "src/b.rs".to_string(),
        "notes.md".to_string(),
    ])
}

fn line_numbers() -> impl Strategy<Value = Vec<u32>> {
    prop::collection::vec(1u32..5, 0..4).prop_map(|mut lines| {
        lines.sort();
        lines.dedup();
        lines
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn generated_inputs_keep_the_unknown_distinction(
        added in prop::collection::btree_map(file_names(), line_numbers(), 1..3),
        hits in prop::collection::btree_map(file_names(), line_numbers(), 0..3),
    ) {
        let output = correct(&added, &hits, &mut Vec::new());
        prop_assert!(p1_weak(&output, &added), "every added file is reported");
        prop_assert!(
            p2_candidate(&output, &added, &hits),
            "unknown is exactly the unmeasured lines: {output:?}"
        );
    }
}

#[test]
fn effectiveness_is_measured_not_claimed() {
    assert_eq!(
        score_p01(p0_vacuous),
        0,
        "a property asserting nothing catches nothing"
    );
    assert_eq!(
        score_p01(p1_weak),
        1,
        "presence alone misses almost everything"
    );
    assert_eq!(score_p2(), 3, "the candidate misses file misattribution");
    assert_eq!(
        score_p3(),
        4,
        "the exact property catches every seeded defect"
    );
}

fn score_p01(predicate: fn(&DiffCoverage, &BTreeMap<String, Vec<u32>>) -> bool) -> usize {
    VARIANTS
        .iter()
        .filter(|(_, v)| catches_p01(predicate, *v))
        .count()
}

fn score_p2() -> usize {
    VARIANTS.iter().filter(|(_, v)| catches_p2(*v)).count()
}

fn score_p3() -> usize {
    VARIANTS.iter().filter(|(_, v)| catches_p3(*v)).count()
}

#[test]
fn each_seeded_defect_is_real() {
    let (added, hits) = rich_fixture();
    let baseline = correct(&added, &hits, &mut Vec::new());
    for (name, variant) in VARIANTS {
        let other = variant(&added, &hits, &mut Vec::new());
        assert_ne!(
            (other.files.clone(), other.unmeasured.clone()),
            (baseline.files.clone(), baseline.unmeasured.clone()),
            "{name} must change the result or it is not a defect"
        );
    }
}

#[test]
fn the_existing_example_tests_catch_three_of_four() {
    // S1 mirrors `a_line_with_no_data_is_unknown_not_uncovered`; S2 mirrors
    // `a_file_the_tool_never_saw_is_named`. Neither fixture puts the same line
    // number in two files, so cross-file misattribution is invisible to both.
    let s1_added: BTreeMap<String, Vec<u32>> =
        [("f.rs".to_string(), vec![1, 2])].into_iter().collect();
    let s1_hits: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let s2_added: BTreeMap<String, Vec<u32>> = [
        ("src/a.rs".to_string(), vec![1]),
        ("notes.md".to_string(), vec![1]),
    ]
    .into_iter()
    .collect();
    let s2_hits: BTreeMap<String, Vec<u32>> =
        [("src/a.rs".to_string(), vec![])].into_iter().collect();

    let s1_holds = |v: IntersectFn| {
        let out = v(&s1_added, &s1_hits, &mut Vec::new());
        out.total_uncovered() == 0 && out.total_unknown() == 2
    };
    let s2_holds = |v: IntersectFn| {
        let out = v(&s2_added, &s2_hits, &mut Vec::new());
        out.unmeasured == vec!["notes.md".to_string()] && out.unmeasured_lines == 1
    };

    let caught = VARIANTS
        .iter()
        .filter(|(_, v)| !s1_holds(*v) || !s2_holds(*v))
        .count();
    assert_eq!(caught, 3, "only cross-file misattribution escapes both");
}
