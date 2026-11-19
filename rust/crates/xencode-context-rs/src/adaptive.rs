//! Adaptive context decisions (XI-3), computed without a model call.
//!
//! This module decides *how much* of the context a turn should carry and *in
//! what order*, from facts the rest of the engine already has: the task shape
//! ([`crate::shape`]), the turn's [`ContextCaps`], the candidate files from
//! [`crate::retrieve`] and, for a bugfix, the relationship evidence derived by
//! [`crate::relationships`] from the indexes already on disk. It never calls a
//! model, never reads the network and never mutates state; the same inputs give
//! the same [`ContextPlan`] every time.
//!
//! **Nothing here changes production context by itself.** [`assemble_chat`] is
//! untouched. A caller opts in by calling [`assemble_chat_adaptive`] with a
//! plan, which reorders and trims the retrieved blocks and then hands the
//! result to the existing assembler. The stable system prefix and the redactor
//! are inside that assembler and are never bypassed: the byte-stable head is
//! still turn 0 verbatim, and every dynamic tier is still redacted on the way
//! out with one shared [`crate::Redactor`].
//!
//! What stays authoritative is unchanged: BM25/hybrid selection in
//! [`crate::embed`], reciprocal-rank fusion, [`crate::compact`] for history,
//! the tier budget in [`crate::budget`], and the redactor in [`crate::redact`].
//! This module only sets limits and an order for the files those systems
//! already produced. There are no embeddings here and no learned weights;
//! corpus term association ([`crate::cooccurrence`]) remains a default-off
//! experiment, and real task-completion gains need a GPU-backed model run this
//! environment cannot do.

use crate::budget::{est_tokens, truncate_to_tokens, ContextCaps};
use crate::context::{assemble_chat, ChatAssembly, ChatInput, RetrievedBlock};
use crate::relationships::{ImpactEvidence, Relation, RelationSource};
use crate::retrieve::RetrievedFile;
use crate::shape::TaskShape;

/// The order retrieved bodies are handed to the assembler. It trims from the
/// bottom, so what leads the list is what survives a tight budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalOrder {
    /// Keep the score order retrieval returned. This is what ships today.
    ByScore,
    /// Move files named by known impact relations to the front, keeping the
    /// score order inside each group. Only chosen for a bugfix with evidence.
    EvidenceFirst,
}

/// One reason for a decision, naming the fact it came from. A decision without
/// a reason is a guess, so every field of [`ContextPlan`] is backed by one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionEvidence {
    /// The plan field this explains, e.g. `retrieval_depth`.
    pub decision: &'static str,
    /// The existing system the fact was read from, e.g. `ContextCaps.top_k`.
    pub source: &'static str,
    /// Plain-English reason, including the numbers used.
    pub why: String,
}

impl DecisionEvidence {
    fn new(decision: &'static str, source: &'static str, why: impl Into<String>) -> Self {
        Self {
            decision,
            source,
            why: why.into(),
        }
    }
}

/// The whole adaptive decision for one turn. Pure data: no function pointers,
/// no I/O handles, nothing that could make two identical inputs disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPlan {
    pub shape: TaskShape,
    /// The caps this plan was decided against, kept so applying it cannot
    /// silently use a different budget than the decision did.
    pub caps: ContextCaps,
    /// How many of the retrieved candidates to keep, never above `caps.top_k`.
    pub retrieval_depth: usize,
    /// Test-to-symbol facts to surface in the compact evidence block. Zero
    /// unless the task is a bugfix with known test relations.
    pub symbol_facts: usize,
    /// Dependent files to surface in the compact evidence block. Zero unless
    /// the task is a bugfix with known dependent relations.
    pub dependents: usize,
    /// Whether test relations may be shown at all.
    pub include_tests: bool,
    /// Structured history stays on by default: the assembler already keeps the
    /// most recent turns within budget, and dropping them here would be a
    /// behaviour change with no measurement behind it.
    pub include_history: bool,
    pub order: RetrievalOrder,
    /// True when there were more candidates than the plan keeps and bodies
    /// still have room to shrink: [`assemble_chat_adaptive`] then trims each
    /// body with the existing token truncator instead of dropping whole files.
    pub compress: bool,
    /// Why each field has this value. Never empty.
    pub evidence: Vec<DecisionEvidence>,
}

fn known<'a>(relations: impl IntoIterator<Item = &'a Relation>) -> Vec<&'a Relation> {
    relations
        .into_iter()
        .filter(|relation| !relation.is_unknown())
        .collect()
}

/// Decide the context plan for one turn.
///
/// `evidence` is `None` when no changed file was supplied or no index was
/// present. That is not zero evidence — it is the absence of a reading, so the
/// plan records an explicit unknown instead of inventing a relation.
///
/// The default is deliberately today's behaviour: score order, every retrieved
/// candidate the caps allow, no injected evidence, history on, no compression.
/// Only a bugfix with *known* XI-2 relations moves any of those, and even then
/// the number of files is never raised above the caps.
pub fn decide(
    shape: TaskShape,
    task_text: &str,
    caps: ContextCaps,
    retrieved: &[RetrievedFile],
    evidence: Option<&ImpactEvidence>,
) -> ContextPlan {
    let mut reasons = Vec::new();

    // Depth is bounded by the measured caps and by what retrieval actually
    // returned. A small-context task therefore cannot over-fetch: one slot is
    // one slot, however many candidates scored.
    let depth = caps.top_k.min(retrieved.len());
    reasons.push(DecisionEvidence::new(
        "retrieval_depth",
        "ContextCaps.top_k and retrieve() results",
        format!(
            "caps allow {} file(s), retrieval returned {} candidate(s), so keep {}; \
             '{}' is read as {}",
            caps.top_k,
            retrieved.len(),
            depth,
            task_text.chars().take(48).collect::<String>(),
            shape.as_str()
        ),
    ));

    let compress =
        retrieved.len() > depth && caps.content_cap_chars > ContextCaps::MIN_CONTENT_CAP_CHARS;
    reasons.push(DecisionEvidence::new(
        "compress",
        "candidate count and ContextCaps.content_cap_chars",
        if compress {
            format!(
                "{} candidate(s) exceed the {} slot(s) and bodies have room to shrink \
                 ({} chars), so trim bodies instead of dropping whole files",
                retrieved.len(),
                depth,
                caps.content_cap_chars
            )
        } else {
            "no over-subscription or no room to shrink bodies; keep bodies whole".to_string()
        },
    ));

    reasons.push(DecisionEvidence::new(
        "include_history",
        "context.rs structured history budget",
        "history stays on: the assembler already keeps the most recent turns inside the budget",
    ));

    let mut plan = ContextPlan {
        shape,
        caps,
        retrieval_depth: depth,
        symbol_facts: 0,
        dependents: 0,
        include_tests: false,
        include_history: true,
        order: RetrievalOrder::ByScore,
        compress,
        evidence: reasons,
    };

    let Some(evidence) = evidence else {
        plan.evidence.push(DecisionEvidence::new(
            "order",
            "impact evidence",
            "no index or changed file was supplied, so no relationship is claimed and score order stands",
        ));
        return plan;
    };

    let dependents = known(evidence.dependents.iter());
    let tests = known(evidence.tests.iter());

    if shape != TaskShape::Bugfix {
        plan.evidence.push(DecisionEvidence::new(
            "order",
            "task shape",
            format!(
                "a {} task keeps score order; impact evidence is not spent on turns \
                 that did not ask about broken code",
                shape.as_str()
            ),
        ));
        return plan;
    }

    if dependents.is_empty() && tests.is_empty() {
        // The index answered, but every answer was Unknown. Keep that fact
        // visible rather than turning it into a phantom edge.
        let unknowns = evidence.unknown.len();
        plan.evidence.push(DecisionEvidence::new(
            "order",
            "impact evidence unknowns",
            format!(
                "bugfix turn, but all {} impact answer(s) are unknown; nothing is \
                 reordered and no edge is invented",
                unknowns
            ),
        ));
        return plan;
    }

    // A bugfix with real relations moves evidence to the front of the same
    // number of slots. The compact evidence block is bounded by the same depth,
    // so the whole decision cannot ask for more room than the caps allow.
    let dependent_count = dependents.len().min(depth.max(1));
    let symbol_count = tests.len().min(depth.max(1));
    plan.order = RetrievalOrder::EvidenceFirst;
    plan.dependents = dependent_count;
    plan.symbol_facts = symbol_count;
    plan.include_tests = symbol_count > 0;

    let sources = {
        let mut seen: Vec<&str> = Vec::new();
        for relation in dependents.iter().chain(tests.iter()) {
            let label = relation_source_label(relation.source);
            if !seen.contains(&label) {
                seen.push(label);
            }
        }
        seen.join(", ")
    };
    plan.evidence.push(DecisionEvidence::new(
        "order",
        "impact_evidence (XI-2 relations)",
        format!(
            "bugfix turn with {} dependent(s) and {} test-to-symbol relation(s) from {}; \
             move the named files to the front of the same {} slot(s)",
            dependents.len(),
            tests.len(),
            sources,
            depth
        ),
    ));
    plan.evidence.push(DecisionEvidence::new(
        "dependents",
        "impact_evidence.dependents",
        format!("show up to {dependent_count} dependent file(s), each with its own source"),
    ));
    plan.evidence.push(DecisionEvidence::new(
        "symbol_facts",
        "impact_evidence.tests",
        format!("show up to {symbol_count} test-to-symbol fact(s), each with its own source"),
    ));
    plan
}

/// The plain name of the existing index field or traversal behind a relation.
/// Kept here so a decision reason never says `Unknown` for a known edge.
pub fn relation_source_label(source: RelationSource) -> &'static str {
    match source {
        RelationSource::SymbolsTests => "symbols.json tests",
        RelationSource::SymbolsFunctions => "symbols.json functions",
        RelationSource::SymbolsStructs => "symbols.json structs",
        RelationSource::SymbolsEnums => "symbols.json enums",
        RelationSource::SymbolsTraits => "symbols.json traits",
        RelationSource::SymbolsTypes => "symbols.json types",
        RelationSource::SymbolsImpls => "symbols.json impls",
        RelationSource::DepsImplVia => "deps.json impl edge",
        RelationSource::AffectedDependents => "affected_dependents over deps.json",
        RelationSource::HistoryPartners => "history.json partners",
        RelationSource::Unknown => "no indexed source",
    }
}

/// The path of every file a known relation names. Used to put evidence files
/// at the head of the retrieval list before the assembler trims from the bottom.
fn evidence_paths(evidence: Option<&ImpactEvidence>) -> Vec<String> {
    let Some(evidence) = evidence else {
        return Vec::new();
    };
    let mut paths = vec![evidence.changed_file.clone()];
    for relation in evidence
        .dependents
        .iter()
        .chain(evidence.co_changes.iter())
        .filter(|relation| !relation.is_unknown())
    {
        paths.push(relation.object.clone());
    }
    paths.sort();
    paths.dedup();
    paths
}

/// Trim the code inside an already-fenced retrieved body to `max_tokens`,
/// keeping the `File: …` header and closing fence intact. The content itself
/// goes through the same [`truncate_to_tokens`] the tier budget uses, so the
/// compression decision reuses the authoritative truncator rather than a
/// second cut with different line-boundary rules.
fn compress_body(body: &str, max_tokens: u64) -> String {
    let Some(first_newline) = body.find('\n') else {
        return body.to_string();
    };
    let header = &body[..=first_newline];
    let rest = &body[first_newline + 1..];
    let Some(fence_end) = rest.find('\n') else {
        return body.to_string();
    };
    let fence = &rest[..=fence_end];
    let tail = &rest[fence_end + 1..];
    let content = tail.strip_suffix("```").unwrap_or(tail);
    let content = content.trim_end_matches('\n');
    let (kept, _) = truncate_to_tokens(content, max_tokens, true);
    format!("{header}{fence}{kept}\n```")
}

/// Build one compact, fenced evidence block from the plan's limits. Every line
/// quotes a real relation and names where it came from; no line is invented to
/// fill a requested slot. Returns `None` when the plan asked for no evidence.
fn evidence_block(plan: &ContextPlan, evidence: Option<&ImpactEvidence>) -> Option<RetrievedBlock> {
    let evidence = evidence?;
    let mut lines: Vec<String> = Vec::new();
    for relation in evidence
        .tests
        .iter()
        .filter(|r| !r.is_unknown())
        .take(plan.symbol_facts)
    {
        if plan.include_tests {
            lines.push(format!(
                "test: {} names {} — {}",
                relation.subject,
                relation.object,
                relation_source_label(relation.source)
            ));
        }
    }
    for relation in evidence
        .dependents
        .iter()
        .filter(|r| !r.is_unknown())
        .take(plan.dependents)
    {
        lines.push(format!(
            "dependent: {} — {}",
            relation.object,
            relation_source_label(relation.source)
        ));
    }
    if lines.is_empty() {
        return None;
    }
    Some(RetrievedBlock {
        path: "impact-evidence".to_string(),
        score: 0,
        body: format!("Impact evidence\n```text\n{}\n```", lines.join("\n")),
    })
}

/// What the integration actually did, so a caller can explain the difference
/// between the two assemblies instead of trusting a plan that was never used.
#[derive(Debug, Clone)]
pub struct AppliedPlan {
    pub assembly: ChatAssembly,
    /// Files from retrieval that survived the plan's depth cap.
    pub retrieval_kept: usize,
    /// Whether a compact impact-evidence block was added.
    pub evidence_added: bool,
    /// Whether history was cleared because the plan said so.
    pub history_cleared: bool,
    /// Bodies trimmed by the compression decision.
    pub bodies_compressed: usize,
}

/// Consume a plan inside the existing chat assembly pipeline.
///
/// This is the opt-in integration point: production callers that use
/// [`assemble_chat`] directly are unchanged. Here the retrieved blocks are
/// ordered and capped by the plan, oversized bodies are trimmed with the
/// existing token truncator when `plan.compress` is set, and a compact
/// impact-evidence block is prepended when the plan asked for one. The
/// resulting [`ChatInput`] is passed to [`assemble_chat`] itself, so the stable
/// prefix, tier budget and redactor all behave exactly as they do today.
pub fn assemble_chat_adaptive(
    mut input: ChatInput,
    plan: &ContextPlan,
    evidence: Option<&ImpactEvidence>,
) -> AppliedPlan {
    let paths = evidence_paths(evidence);
    let mut blocks = std::mem::take(&mut input.retrieved);
    if plan.order == RetrievalOrder::EvidenceFirst {
        // Stable partition: evidence files lead, score order inside each group.
        // The comparator only compares membership, never scores, so equal paths
        // keep the order retrieval gave them.
        blocks.sort_by_key(|block| !paths.contains(&block.path));
    }
    blocks.truncate(plan.retrieval_depth);

    let mut bodies_compressed = 0usize;
    if plan.compress {
        let half_chars = (plan.caps.content_cap_chars / 2).max(ContextCaps::MIN_CONTENT_CAP_CHARS);
        let max_tokens = est_tokens(half_chars, true);
        for block in &mut blocks {
            let trimmed = compress_body(&block.body, max_tokens);
            if trimmed.len() < block.body.len() {
                bodies_compressed += 1;
                block.body = trimmed;
            }
        }
    }

    let retrieval_kept = blocks.len();
    let mut evidence_added = false;
    if let Some(block) = evidence_block(plan, evidence) {
        evidence_added = true;
        // First, so the assembler's trim-from-the-bottom rule drops the least
        // important file before it drops critical evidence.
        blocks.insert(0, block);
    }

    let history_cleared = !plan.include_history && !input.history.is_empty();
    if history_cleared {
        input.history = &[];
    }
    input.retrieved = blocks;
    AppliedPlan {
        assembly: assemble_chat(input),
        retrieval_kept,
        evidence_added,
        history_cleared,
        bodies_compressed,
    }
}

/// Estimate the tokens a body of `chars` code characters costs. Exposed so a
/// caller can compare a plan's budget with what the assembler will count.
pub fn body_tokens(chars: usize) -> u64 {
    est_tokens(chars, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::HardwareProfile;
    use crate::relationships::{RelationKind, RelationScope, RelationSource};
    use crate::retrieve::{RetrievalIndex, RetrieveOptions};
    use std::collections::HashSet;

    fn candidate(path: &str, score: u64) -> RetrievedFile {
        RetrievedFile {
            path: path.to_string(),
            score,
            reasons: vec!["symbol: save".to_string()],
        }
    }

    fn caps() -> ContextCaps {
        ContextCaps::from_profile(HardwareProfile::Low) // top_k = 3
    }

    #[test]
    fn default_plan_matches_todays_assembly_choices() {
        let retrieved = vec![candidate("a.rs", 9), candidate("b.rs", 4)];
        let plan = decide(
            TaskShape::General,
            "what does this do",
            caps(),
            &retrieved,
            None,
        );
        assert_eq!(plan.shape, TaskShape::General);
        assert_eq!(plan.retrieval_depth, 2);
        assert_eq!(plan.order, RetrievalOrder::ByScore);
        assert_eq!(plan.symbol_facts, 0);
        assert_eq!(plan.dependents, 0);
        assert!(!plan.include_tests);
        assert!(plan.include_history);
        assert!(!plan.compress);
        assert!(!plan.evidence.is_empty());
        assert!(plan
            .evidence
            .iter()
            .all(|e| !e.why.is_empty() && !e.source.is_empty()));
    }

    #[test]
    fn small_context_never_over_fetches() {
        let small = ContextCaps {
            top_k: 1,
            content_cap_chars: 1_536,
        };
        let many: Vec<RetrievedFile> = (0..12)
            .map(|i| candidate(&format!("f{i}.rs"), 20 - i))
            .collect();
        let plan = decide(
            TaskShape::General,
            "fix the tiny budget",
            small,
            &many,
            None,
        );
        assert_eq!(plan.retrieval_depth, 1, "one slot stays one slot");
    }

    #[test]
    fn identical_inputs_give_an_identical_plan() {
        let retrieved = vec![
            candidate("a.rs", 9),
            candidate("b.rs", 4),
            candidate("c.rs", 1),
        ];
        let a = decide(
            TaskShape::Bugfix,
            "the save is broken",
            caps(),
            &retrieved,
            None,
        );
        let b = decide(
            TaskShape::Bugfix,
            "the save is broken",
            caps(),
            &retrieved,
            None,
        );
        assert_eq!(a, b);
    }

    fn evidence_with_relations() -> ImpactEvidence {
        let dependent = Relation {
            kind: RelationKind::DependedOnBy,
            subject: "src/store.rs".into(),
            object: "src/service.rs".into(),
            source: RelationSource::AffectedDependents,
            scope: RelationScope::Rust,
            evidence: "reverse dependency walk reached service.rs".into(),
        };
        let test = Relation {
            kind: RelationKind::TestFor,
            subject: "save_persists_value".into(),
            object: "save".into(),
            source: RelationSource::SymbolsTests,
            scope: RelationScope::Rust,
            evidence: "test names save".into(),
        };
        ImpactEvidence {
            changed_file: "src/store.rs".into(),
            dependents: vec![dependent],
            tests: vec![test],
            co_changes: vec![],
            unknown: vec![],
        }
    }

    #[test]
    fn bugfix_uses_xi2_relations_and_names_their_source() {
        let retrieved = vec![
            candidate("src/store.rs", 9),
            candidate("src/unrelated.rs", 4),
        ];
        let plan = decide(
            TaskShape::Bugfix,
            "fix the broken save",
            caps(),
            &retrieved,
            Some(&evidence_with_relations()),
        );
        assert_eq!(plan.order, RetrievalOrder::EvidenceFirst);
        assert_eq!(plan.dependents, 1);
        assert_eq!(plan.symbol_facts, 1);
        assert!(plan.include_tests);
        let order = plan
            .evidence
            .iter()
            .find(|e| e.decision == "order")
            .unwrap();
        assert!(order.why.contains("affected_dependents over deps.json"));
        assert!(order.why.contains("symbols.json tests"));
    }

    #[test]
    fn missing_measurements_stay_unknown_and_no_edge_is_invented() {
        let retrieved = vec![candidate("src/store.rs", 9)];
        let unknown_only = ImpactEvidence {
            changed_file: "src/store.rs".into(),
            dependents: vec![],
            tests: vec![],
            co_changes: vec![],
            unknown: vec![crate::relationships::Relation {
                kind: RelationKind::DependedOnBy,
                subject: "src/store.rs".into(),
                object: "<unknown>".into(),
                source: RelationSource::Unknown,
                scope: RelationScope::Unknown,
                evidence: "no indexed Rust file depends on it".into(),
            }],
        };
        let plan = decide(
            TaskShape::Bugfix,
            "fix it",
            caps(),
            &retrieved,
            Some(&unknown_only),
        );
        assert_eq!(plan.order, RetrievalOrder::ByScore);
        assert_eq!(plan.dependents, 0);
        assert_eq!(plan.symbol_facts, 0);
        // The absence is stated, not silently turned into a zero-count success.
        let reasons = plan
            .evidence
            .iter()
            .find(|e| e.decision == "order")
            .unwrap();
        assert!(reasons.why.contains("unknown"));
    }

    #[test]
    fn an_empty_index_returns_no_retrieval_and_a_zero_depth_plan() {
        let index = RetrievalIndex::default();
        let opts = RetrieveOptions::default();
        let found = crate::retrieve("anything", &index, &HashSet::new(), &opts);
        assert!(found.is_empty());
        let plan = decide(TaskShape::General, "anything", caps(), &found, None);
        assert_eq!(plan.retrieval_depth, 0);
    }

    #[test]
    fn compression_trims_the_body_without_losing_its_fence() {
        let body = format!("File: a.rs\n```rust\n{}\n```", "x\n".repeat(400));
        let trimmed = compress_body(&body, est_tokens(64, true));
        assert!(trimmed.starts_with("File: a.rs\n```rust\n"));
        assert!(trimmed.ends_with("```"));
        assert!(trimmed.len() < body.len());
    }
}
