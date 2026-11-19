//! Semantic relationships derived from the indexes that already exist.
//!
//! This module never builds a second index and never keeps a universal graph.
//! Every [`Relation`] is computed on demand from one of the representations the
//! context engine already writes:
//!
//! - `symbols.json` (`PerFileSymbols`) — declarations, tests, trait impls;
//! - `deps.json` and the reverse-reachability walk in [`crate::advise`] —
//!   which files a change can break;
//! - `history.json` — which files people change together.
//!
//! Two rules matter to callers. First, a relation always says where it came
//! from: [`Relation::source`] names the exact index/field/traversal that
//! produced it, and [`Relation::scope`] says which language tier supports that
//! derivation. Second, a relation that cannot be derived is returned as a
//! single [`RelationSource::Unknown`] row with a reason in
//! [`Relation::evidence`] — never as a guessed edge. The Rust dependency graph
//! and the test/impl fields belong to Rust only; the TypeScript lexical tier
//! contributes declarations, but no resolver edges, so asking it for an impl
//! or dependent relation is answered `Unknown`, not extrapolated.
//!
//! The coordinator in `xencode-intelligence-rs` can call these queries when it
//! needs evidence about a change; this crate never depends on that one.

use crate::cochange::CommitHistory;
use crate::retrieve::{word_tokens, RetrievalIndex};
use crate::symbols::PerFileSymbols;
use serde::{Deserialize, Serialize};

/// How far the dependent-component walk reaches. This is the same order of
/// magnitude as the watcher's advice query; it is deliberately bounded so a
/// relationship answer stays a local explanation, not a whole-repo claim.
pub const RELATIONSHIP_MAX_HOPS: usize = 3;

/// The relationship a query asks for. The direction is part of the name:
/// `Defines(subject = symbol)` returns the declaring file, while
/// `DependedOnBy(subject = changed file)` returns the files that would be
/// affected by changing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationKind {
    /// A `#[test]` function and a symbol whose name it carries.
    TestFor,
    /// A file implementing a trait and the trait's declaring file.
    Implements,
    /// A changed file and the files that (transitively) depend on it.
    DependedOnBy,
    /// A symbol and the file that declares it.
    Defines,
    /// A file and the files it was changed alongside in real git history.
    CoChangesWith,
}

/// Which existing representation produced a relation. `Unknown` is the explicit
/// answer when none did; it is not a relation and must not be drawn as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationSource {
    /// `symbols.json`, `tests` field.
    SymbolsTests,
    /// `symbols.json`, `functions` field.
    SymbolsFunctions,
    /// `symbols.json`, `structs` field.
    SymbolsStructs,
    /// `symbols.json`, `enums` field.
    SymbolsEnums,
    /// `symbols.json`, `traits` field.
    SymbolsTraits,
    /// `symbols.json`, `types` field.
    SymbolsTypes,
    /// `symbols.json`, `impls` field (carried into `deps.json` by `build_graph`).
    SymbolsImpls,
    /// `deps.json` edge whose `via` starts with `impl `.
    DepsImplVia,
    /// `advise::affected_dependents` over `deps.json`.
    AffectedDependents,
    /// `history.json`, `FileCommits::partners` mined from `git log`.
    HistoryPartners,
    /// Nothing in the existing indexes supports the relation.
    Unknown,
}

/// The language tier whose rules make a derivation valid. Explicit so a
/// TypeScript declaration is never presented as a Rust dependency and a
/// language-agnostic history pair is never presented as a symbol-level fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationScope {
    /// Rust parse-tree symbol tier and the Rust `use`/`mod`/`impl` resolver.
    Rust,
    /// TypeScript lexical symbol tier (names only, no edges).
    TypeScript,
    /// Mined git history; paths only, independent of language.
    History,
    /// No tier supports the answer.
    Unknown,
}

impl RelationScope {
    fn of(index: &RetrievalIndex, path: &str) -> Self {
        match index
            .files
            .iter()
            .find(|entry| entry.path == path)
            .map(|entry| entry.language.as_str())
        {
            Some("rust") => Self::Rust,
            Some("typescript") => Self::TypeScript,
            _ => Self::Unknown,
        }
    }

    fn tier_name(self) -> &'static str {
        match self {
            Self::Rust => "Rust parse-tree symbol tier",
            Self::TypeScript => "TypeScript lexical symbol tier",
            Self::History => "mined git history",
            Self::Unknown => "no indexed language tier",
        }
    }
}

/// One derived relationship with its provenance and a human-readable reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relation {
    pub kind: RelationKind,
    /// The entity the caller asked about (a path or a symbol, per `kind`).
    pub subject: String,
    /// The entity it relates to (the other path or symbol).
    pub object: String,
    /// The existing index/field/traversal this was derived from.
    pub source: RelationSource,
    /// The language tier whose rules make the derivation valid.
    pub scope: RelationScope,
    /// Why this relation holds, or why none could be established.
    pub evidence: String,
}

impl Relation {
    /// True when the answer is an explicit "not derivable" row. Callers that
    /// draw edges must skip these; callers that render unknown must show them.
    pub fn is_unknown(&self) -> bool {
        self.source == RelationSource::Unknown
    }
}

fn unknown(kind: RelationKind, subject: &str, reason: impl Into<String>) -> Relation {
    Relation {
        kind,
        subject: subject.to_string(),
        object: "<unknown>".to_string(),
        source: RelationSource::Unknown,
        scope: RelationScope::Unknown,
        evidence: reason.into(),
    }
}

/// A symbol name is carried by a test when its lowercased word tokens appear
/// as one contiguous run in the test's tokens. `save` matches
/// `save_persists_value`; an unrelated `persists` does not match because the
/// tokens are compared whole. Both sides come from the symbol tier, so this is
/// a derivation over indexed names, not a claim about what the test asserts.
fn test_names_symbol(test: &str, symbol: &str) -> bool {
    if test == symbol {
        return false;
    }
    let test_words = word_tokens(test);
    let symbol_words = word_tokens(symbol);
    !symbol_words.is_empty()
        && test_words.len() >= symbol_words.len()
        && test_words
            .windows(symbol_words.len())
            .any(|window| window == symbol_words.as_slice())
}

/// Every name a file declares, paired with the `symbols.json` field it came
/// from. Test functions are listed under `tests`, not also repeated under
/// `functions`, so a caller sees each declaration exactly once.
fn declared_names_in_file(symbols: &PerFileSymbols) -> Vec<(String, RelationSource)> {
    let mut out = Vec::new();
    let mut push = |names: &[String], source| {
        out.extend(names.iter().map(|name| (name.clone(), source)));
    };
    push(&symbols.structs, RelationSource::SymbolsStructs);
    push(&symbols.enums, RelationSource::SymbolsEnums);
    push(&symbols.traits, RelationSource::SymbolsTraits);
    push(&symbols.types, RelationSource::SymbolsTypes);
    let functions: Vec<String> = symbols
        .functions
        .iter()
        .filter(|name| !symbols.tests.contains(name))
        .cloned()
        .collect();
    push(&functions, RelationSource::SymbolsFunctions);
    push(&symbols.tests, RelationSource::SymbolsTests);
    out
}

fn field_label(source: RelationSource) -> &'static str {
    match source {
        RelationSource::SymbolsTests => "tests",
        RelationSource::SymbolsFunctions => "functions",
        RelationSource::SymbolsStructs => "structs",
        RelationSource::SymbolsEnums => "enums",
        RelationSource::SymbolsTraits => "traits",
        RelationSource::SymbolsTypes => "types",
        _ => "unknown",
    }
}

fn query_test_for(index: &RetrievalIndex, test: &str) -> Vec<Relation> {
    let mut out = Vec::new();
    for (file, symbols) in &index.symbols {
        if !symbols.tests.iter().any(|name| name == test) {
            continue;
        }
        let scope = RelationScope::of(index, file);
        for function in &symbols.functions {
            if function == test || !test_names_symbol(test, function) {
                continue;
            }
            out.push(Relation {
                kind: RelationKind::TestFor,
                subject: test.to_string(),
                object: function.clone(),
                source: RelationSource::SymbolsTests,
                scope,
                evidence: format!(
                    "`{test}` is in `tests` of symbols.json for `{file}`; `{function}` \
                     is in `functions` of the same file ({}) and its name appears in the test name",
                    scope.tier_name()
                ),
            });
        }
    }
    if out.is_empty() {
        return vec![unknown(
            RelationKind::TestFor,
            test,
            format!(
                "no symbol in the symbol tier has a name carried by test `{test}`; \
                 cross-file test association is not inferred"
            ),
        )];
    }
    out
}

fn query_implements(index: &RetrievalIndex, file: &str) -> Vec<Relation> {
    let scope = RelationScope::of(index, file);
    if scope != RelationScope::Rust {
        return vec![unknown(
            RelationKind::Implements,
            file,
            format!(
                "trait-implementation edges come from the Rust dependency graph; `{file}` \
                 is in {} and has no resolver edges",
                scope.tier_name()
            ),
        )];
    }
    let mut out: Vec<Relation> = index
        .deps
        .iter()
        .filter(|edge| edge.from == file && edge.via.starts_with("impl "))
        .map(|edge| {
            let trait_name = edge.via.trim_start_matches("impl ");
            Relation {
                kind: RelationKind::Implements,
                subject: file.to_string(),
                object: trait_name.to_string(),
                source: RelationSource::DepsImplVia,
                scope: RelationScope::Rust,
                evidence: format!(
                    "deps.json edge `{file}` -> `{}` via `{}`, built from the `impls` and \
                     `traits` fields of symbols.json; the trait `{trait_name}` is declared in `{}`",
                    edge.to, edge.via, edge.to
                ),
            }
        })
        .collect();
    out.sort_by(|a, b| a.object.cmp(&b.object));
    if out.is_empty() {
        return vec![unknown(
            RelationKind::Implements,
            file,
            format!("no `impl Trait for …` edge starts at `{file}` in deps.json"),
        )];
    }
    out
}

fn query_depended_on_by(index: &RetrievalIndex, file: &str) -> Vec<Relation> {
    let scope = RelationScope::of(index, file);
    if scope != RelationScope::Rust {
        return vec![unknown(
            RelationKind::DependedOnBy,
            file,
            format!(
                "dependent-component traversal walks the Rust dependency graph; `{file}` \
                 is in {} and has no resolver edges",
                scope.tier_name()
            ),
        )];
    }
    // Reuse the watcher's reverse-reachability query rather than reimplementing
    // a second traversal over the same edges.
    let affected = crate::advise::affected_dependents(&index.deps, &[file], RELATIONSHIP_MAX_HOPS);
    let mut out: Vec<Relation> = affected
        .get(file)
        .into_iter()
        .flatten()
        .map(|dependent| Relation {
            kind: RelationKind::DependedOnBy,
            subject: file.to_string(),
            object: dependent.clone(),
            source: RelationSource::AffectedDependents,
            scope: RelationScope::Rust,
            evidence: format!(
                "advise::affected_dependents over deps.json (reverse dependency walk, \
                 max {RELATIONSHIP_MAX_HOPS} hops) reached `{dependent}` from changed file `{file}`"
            ),
        })
        .collect();
    out.sort_by(|a, b| a.object.cmp(&b.object));
    if out.is_empty() {
        return vec![unknown(
            RelationKind::DependedOnBy,
            file,
            format!("no indexed Rust file depends on `{file}` within {RELATIONSHIP_MAX_HOPS} hops"),
        )];
    }
    out
}

fn query_defines(index: &RetrievalIndex, subject: &str) -> Vec<Relation> {
    if index.files.iter().any(|entry| entry.path == subject) {
        let Some(symbols) = index.symbols.get(subject) else {
            return vec![unknown(
                RelationKind::Defines,
                subject,
                format!("no symbols.json record for `{subject}`"),
            )];
        };
        let scope = RelationScope::of(index, subject);
        let mut out: Vec<Relation> = declared_names_in_file(symbols)
            .into_iter()
            .map(|(name, source)| Relation {
                kind: RelationKind::Defines,
                subject: subject.to_string(),
                object: name.clone(),
                source,
                scope,
                evidence: format!(
                    "`{name}` is in the `{}` field of symbols.json for `{subject}` ({})",
                    field_label(source),
                    scope.tier_name()
                ),
            })
            .collect();
        out.sort_by(|a, b| (&a.object, a.source as u8).cmp(&(&b.object, b.source as u8)));
        if out.is_empty() {
            return vec![unknown(
                RelationKind::Defines,
                subject,
                format!("no declarations are indexed for `{subject}`"),
            )];
        }
        return out;
    }

    let mut out = Vec::new();
    for (file, symbols) in &index.symbols {
        for (name, source) in declared_names_in_file(symbols) {
            if name != subject {
                continue;
            }
            let scope = RelationScope::of(index, file);
            out.push(Relation {
                kind: RelationKind::Defines,
                subject: subject.to_string(),
                object: file.clone(),
                source,
                scope,
                evidence: format!(
                    "`{subject}` is in the `{}` field of symbols.json for `{file}` ({})",
                    field_label(source),
                    scope.tier_name()
                ),
            });
        }
    }
    out.sort_by(|a, b| (&a.object, a.source as u8).cmp(&(&b.object, b.source as u8)));
    if out.is_empty() {
        return vec![unknown(
            RelationKind::Defines,
            subject,
            format!("no indexed file declares a symbol named `{subject}`"),
        )];
    }
    out
}

fn query_co_changes_with(history: &CommitHistory, file: &str) -> Vec<Relation> {
    let Some(entry) = history.get(file) else {
        return vec![unknown(
            RelationKind::CoChangesWith,
            file,
            format!("no mined git history records `{file}`"),
        )];
    };
    if entry.hub || entry.partners.is_empty() {
        return vec![unknown(
            RelationKind::CoChangesWith,
            file,
            format!(
                "history.json has no informative co-change partner for `{file}` \
                 (hub: {}, partners: {})",
                entry.hub,
                entry.partners.len()
            ),
        )];
    }
    let mut out: Vec<Relation> = entry
        .partners
        .iter()
        .map(|(partner, commits)| Relation {
            kind: RelationKind::CoChangesWith,
            subject: file.to_string(),
            object: partner.clone(),
            source: RelationSource::HistoryPartners,
            scope: RelationScope::History,
            evidence: format!(
                "history.json records `{partner}` as a partner of `{file}` in {commits} \
                 commits, mined from a real `git log` pass (paths only, any language)"
            ),
        })
        .collect();
    out.sort_by(|a, b| a.object.cmp(&b.object));
    out
}

/// Answer one relationship question from the existing indexes.
///
/// `subject` is a path for `Implements`, `DependedOnBy` and `CoChangesWith`; a
/// test name for `TestFor`; and either a symbol name or a known file path for
/// `Defines`. An empty or unanswerable question returns exactly one
/// [`RelationSource::Unknown`] row — no edge is ever invented.
pub fn query_relations(index: &RetrievalIndex, kind: RelationKind, subject: &str) -> Vec<Relation> {
    match kind {
        RelationKind::TestFor => query_test_for(index, subject),
        RelationKind::Implements => query_implements(index, subject),
        RelationKind::DependedOnBy => query_depended_on_by(index, subject),
        RelationKind::Defines => query_defines(index, subject),
        RelationKind::CoChangesWith => query_co_changes_with(&index.history, subject),
    }
}

/// The three kinds of evidence the coordinator can show for one changed file:
/// tests that name its symbols, components that depend on it, and files it was
/// changed alongside. Unknown answers are kept alongside the known relations
/// so a caller never mistakes "nothing was indexed" for "nothing relates".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImpactEvidence {
    pub changed_file: String,
    pub dependents: Vec<Relation>,
    pub tests: Vec<Relation>,
    pub co_changes: Vec<Relation>,
    /// Explicit `Unknown` answers, one per kind that could not be established.
    pub unknown: Vec<Relation>,
}

impl ImpactEvidence {
    /// All relations, known and unknown, in a stable order.
    pub fn all(&self) -> Vec<&Relation> {
        self.dependents
            .iter()
            .chain(self.tests.iter())
            .chain(self.co_changes.iter())
            .chain(self.unknown.iter())
            .collect()
    }

    /// True when no known relation was established. Unknown rows are not
    /// evidence, so a file with three explicit `Unknown` answers is empty.
    pub fn is_empty(&self) -> bool {
        self.dependents.iter().all(|r| r.is_unknown())
            && self.tests.iter().all(|r| r.is_unknown())
            && self.co_changes.iter().all(|r| r.is_unknown())
    }
}

fn collect_unknowns(into: &mut Vec<Relation>, relations: &[Relation]) {
    into.extend(
        relations
            .iter()
            .filter(|relation| relation.is_unknown())
            .cloned(),
    );
}

/// Combine test, dependent and co-change evidence for a changed file. Each
/// constituent answer is produced by [`query_relations`], so every relation in
/// the result still carries its own source and scope.
pub fn impact_evidence(index: &RetrievalIndex, changed_file: &str) -> ImpactEvidence {
    let dependents = query_relations(index, RelationKind::DependedOnBy, changed_file);
    let co_changes = query_relations(index, RelationKind::CoChangesWith, changed_file);

    let mut tests = Vec::new();
    let declared: Vec<(String, RelationSource)> = index
        .symbols
        .get(changed_file)
        .map(declared_names_in_file)
        .unwrap_or_default();
    for (test_file, symbols) in &index.symbols {
        for test in &symbols.tests {
            for (name, source) in &declared {
                if test_names_symbol(test, name) {
                    let scope = RelationScope::of(index, test_file);
                    tests.push(Relation {
                        kind: RelationKind::TestFor,
                        subject: test.clone(),
                        object: name.clone(),
                        source: RelationSource::SymbolsTests,
                        scope,
                        evidence: format!(
                            "`{test}` is in `tests` of symbols.json for `{test_file}`; it names \
                             `{name}`, which is in the `{}` field of `{changed_file}` ({})",
                            field_label(*source),
                            scope.tier_name()
                        ),
                    });
                }
            }
        }
    }
    tests.sort_by(|a, b| (&a.subject, &a.object).cmp(&(&b.subject, &b.object)));

    let mut unknown_answers = Vec::new();
    collect_unknowns(&mut unknown_answers, &dependents);
    collect_unknowns(&mut unknown_answers, &co_changes);
    if tests.is_empty() {
        unknown_answers.push(unknown(
            RelationKind::TestFor,
            changed_file,
            format!(
                "no test in the symbol tier names a symbol declared in `{changed_file}`; \
                 tests in other languages are not inferred"
            ),
        ));
    }

    ImpactEvidence {
        changed_file: changed_file.to_string(),
        dependents,
        tests,
        co_changes,
        unknown: unknown_answers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::FileEntry;
    use crate::symbols::DepEdge;
    use std::collections::BTreeMap;

    fn entry(path: &str, language: &str) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            language: language.to_string(),
            size: 0,
            loc: 0,
            ext: "rs".to_string(),
            important: false,
            secret: false,
            binary: false,
        }
    }

    fn index() -> RetrievalIndex {
        let store = PerFileSymbols {
            functions: vec!["load".into(), "save".into(), "save_persists_value".into()],
            tests: vec!["save_persists_value".into()],
            structs: vec!["MemoryStore".into()],
            ..Default::default()
        };
        let ts = PerFileSymbols {
            functions: vec!["render".into()],
            structs: vec!["Widget".into()],
            ..Default::default()
        };
        RetrievalIndex {
            files: vec![
                entry("src/store.rs", "rust"),
                entry("src/widget.ts", "typescript"),
            ],
            symbols: BTreeMap::from([("src/store.rs".into(), store), ("src/widget.ts".into(), ts)]),
            deps: vec![DepEdge {
                from: "src/service.rs".into(),
                to: "src/store.rs".into(),
                via: "crate::store".into(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn an_unanswerable_query_is_an_explicit_unknown() {
        let index = index();
        let answers = query_relations(&index, RelationKind::Defines, "never_declared");
        assert_eq!(answers.len(), 1);
        assert!(answers[0].is_unknown());
        assert_eq!(answers[0].source, RelationSource::Unknown);
        assert_eq!(answers[0].object, "<unknown>");
        assert!(answers[0].evidence.contains("never_declared"));
    }

    #[test]
    fn a_typescript_file_has_no_rust_dependency_edges() {
        let index = index();
        for kind in [RelationKind::Implements, RelationKind::DependedOnBy] {
            let answers = query_relations(&index, kind, "src/widget.ts");
            assert_eq!(answers.len(), 1, "{kind:?}");
            assert!(answers[0].is_unknown(), "{kind:?}: {:?}", answers[0]);
            assert!(answers[0]
                .evidence
                .contains("TypeScript lexical symbol tier"));
        }
        // Declarations are supported in that tier, and say so.
        let defined = query_relations(&index, RelationKind::Defines, "Widget");
        assert_eq!(defined.len(), 1);
        assert_eq!(defined[0].scope, RelationScope::TypeScript);
        assert_eq!(defined[0].object, "src/widget.ts");
    }

    #[test]
    fn a_test_names_a_symbol_from_the_same_file() {
        let index = index();
        let answers = query_relations(&index, RelationKind::TestFor, "save_persists_value");
        assert_eq!(answers.len(), 1, "{answers:?}");
        assert_eq!(answers[0].object, "save");
        assert_eq!(answers[0].source, RelationSource::SymbolsTests);
        assert_eq!(answers[0].scope, RelationScope::Rust);
    }
}
