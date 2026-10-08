//! `OR-1` — one task split into a dependency tree, and the number that says
//! whether that tree may be scheduled.
//!
//! The scheduler (`OR-2`) runs any graph it is handed: it checks the shape,
//! computes readiness, launches real children. It has no opinion about whether
//! the graph is a correct reading of the task, and a wrong edge does not crash —
//! two nodes with no edge between them simply start together, and the second
//! fails against a file the first has not written yet. So a split that came from
//! anywhere other than a person who checked it has to be scored before
//! [`crate::Scheduler`] sees it. That is this module's whole job.
//!
//! Two choices carry the design, and both exist because the obvious alternative
//! would hide a claim:
//!
//! - **Nodes are matched by the paths they own, never by their names.** A
//!   planner invents ids, and two splits that describe the same work in
//!   different words must still be comparable. Paths are facts on disk, so a
//!   [`Reference`] is the file set a change really touches plus the order the
//!   build really enforces, and every figure below comes from comparing two sets
//!   of paths — which a reader can re-check by hand.
//! - **Silence is not agreement.** A split that says nothing about the order of
//!   two changes the reference says must be ordered is not neutral: the
//!   scheduler reads a missing edge as *start together*. Scoring that as a match
//!   would give a flat, information-free split a perfect number, so an unstated
//!   order is counted against the split and named, exactly like a wrong one.
//!
//! The quality number is therefore two counts rather than one blended figure —
//! how much of the file set the split owns, and how many required orders it
//! itself states correctly — each quoted beside the same two counts for
//! [`Split::flat_baseline`]: the split that names every file as its own node and
//! claims nothing about order. That baseline is what "we gained nothing by
//! scheduling" looks like as a number, and [`decide`] requires a split to beat it
//! instead of matching it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::scheduler::{NodeId, TaskGraph, TaskNode};
use crate::task_contract::TaskContract;

/// One unit of work: what it is for, which files it is *the one that changes*,
/// what must finish before it starts, and the command that proves it did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subtask {
    pub id: NodeId,
    /// One line, in the planner's own words. Never matched on — see the module
    /// docs — but it is what a person reads before approving a launch.
    pub goal: String,
    /// The files this node is responsible for. Ownership is longest-path: a node
    /// listing a directory owns everything under it unless another node lists
    /// something deeper.
    pub paths: Vec<String>,
    /// Ids that must finish first. Empty means the scheduler may start it at once.
    pub needs: Vec<NodeId>,
    /// A command whose exit code decides the node. This becomes the graph node's
    /// command, so a split with nothing here is a node that cannot be graded.
    pub verify: String,
}

/// A task read as a dependency tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    pub task: String,
    pub subtasks: Vec<Subtask>,
}

/// Why a split is not even a shape the scheduler could be shown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitError {
    /// A node with no id, so nothing else could name it.
    MissingId,
    /// Two nodes share an id, so an edge would not name one of them.
    DuplicateId(NodeId),
    /// A node owns no file, so no change can ever be attributed to it.
    NoPaths(NodeId),
    /// A node has no command, so finishing it would be a claim rather than a fact.
    NoVerification(NodeId),
    /// The dependency edges themselves: reused from the graph layer rather than
    /// re-derived, so one rule decides shape for both.
    Graph(crate::scheduler::GraphError),
    /// The answer is not shaped like a split: a field is missing, or it is not the
    /// type it has to be. Reported by name, because a planner that left `needs`
    /// out is claiming every node may start at once, and that is not a thing to
    /// read as an empty list.
    Shape(String),
}

impl std::fmt::Display for SplitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SplitError::MissingId => write!(f, "a node has no id"),
            SplitError::DuplicateId(id) => write!(f, "two nodes are both named `{id}`"),
            SplitError::NoPaths(id) => write!(f, "`{id}` claims no file, so nothing proves it"),
            SplitError::NoVerification(id) => {
                write!(
                    f,
                    "`{id}` states no command, so its completion is unverifiable"
                )
            }
            SplitError::Graph(error) => write!(f, "{error}"),
            SplitError::Shape(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for SplitError {}

impl Split {
    pub fn new(task: impl Into<String>) -> Self {
        Split {
            task: task.into(),
            subtasks: Vec::new(),
        }
    }

    pub fn push(&mut self, subtask: Subtask) {
        self.subtasks.push(subtask);
    }

    /// Read a planner's answer as a split, with nothing repaired on the way.
    ///
    /// Every field a node needs has to be there, and no field the shape does not
    /// declare is allowed: a planner that wrote `dependencies` where `needs` was
    /// asked for would otherwise be read as a node with no dependencies at all,
    /// which the scheduler would start immediately alongside everything else. A
    /// mis-shapen answer is refused by name, not guessed at.
    pub fn from_value(value: &serde_json::Value) -> Result<Split, SplitError> {
        let root = object(value, "the answer")?;
        let task = string_field(root, "task", "the answer")?;
        let subtasks = root
            .get("subtasks")
            .ok_or_else(|| SplitError::Shape("the answer has no `subtasks`".to_string()))?;
        let items = subtasks
            .as_array()
            .ok_or_else(|| SplitError::Shape("`subtasks` is not a list".to_string()))?;
        if items.is_empty() {
            return Err(SplitError::Shape(
                "`subtasks` is empty, which is a task that was not split".to_string(),
            ));
        }
        let mut subtasks = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let label = format!("subtask {}", index + 1);
            let node = object(item, &label)?;
            let id = string_field(node, "id", &label)?;
            // The unknown-key scan runs before the remaining fields are read, so a
            // node that wrote `dependencies` is told that `dependencies` is not a
            // field of this shape — which is the diagnosis — instead of being told
            // it has no `needs`, which is only the symptom.
            let mut unexpected: Vec<&str> = node
                .keys()
                .map(|k| k.as_str())
                .filter(|k| !matches!(*k, "id" | "goal" | "paths" | "needs" | "verify"))
                .collect();
            if !unexpected.is_empty() {
                unexpected.sort_unstable();
                return Err(SplitError::Shape(format!(
                    "`{id}` carries {}, which this shape does not declare",
                    unexpected
                        .iter()
                        .map(|k| format!("`{k}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            let goal = string_field(node, "goal", &label)?;
            let paths = string_list(node, "paths", &label)?;
            let needs = string_list(node, "needs", &label)?;
            let verify = string_field(node, "verify", &label)?;
            subtasks.push(Subtask {
                id,
                goal,
                paths,
                needs,
                verify,
            });
        }
        Ok(Split { task, subtasks })
    }

    /// Everything that must be true before this split means anything: named and
    /// unique nodes, each owning at least one file and carrying one command, and
    /// edges that form a graph. The edge rules are delegated to
    /// [`TaskGraph::validate`] so a split and a hand-written recipe cannot drift
    /// into disagreeing about what a valid dependency is.
    pub fn validate(&self) -> Result<(), SplitError> {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for node in &self.subtasks {
            if node.id.trim().is_empty() {
                return Err(SplitError::MissingId);
            }
            if !seen.insert(node.id.as_str()) {
                return Err(SplitError::DuplicateId(node.id.clone()));
            }
            if node.paths.iter().all(|p| p.trim().is_empty()) {
                return Err(SplitError::NoPaths(node.id.clone()));
            }
            if node.verify.trim().is_empty() {
                return Err(SplitError::NoVerification(node.id.clone()));
            }
        }
        TaskGraph::validate(&self.to_task_graph()).map_err(SplitError::Graph)
    }

    /// The graph the scheduler would actually run: each node's command is the
    /// command that verifies it, so an unschedulable node is never quietly
    /// substituted with something that always succeeds.
    pub fn to_task_graph(&self) -> TaskGraph {
        let mut graph = TaskGraph::new();
        for node in &self.subtasks {
            graph.add(TaskNode {
                id: node.id.clone(),
                command: node.verify.clone(),
                needs: node.needs.clone(),
            });
        }
        graph
    }

    /// The task contract each node would be launched under (`OR-15`), so a split
    /// is directly consumable by the launch path instead of needing a second
    /// description of the same work. A node may write exactly the files it claims,
    /// must leave them behind, and is graded by the command it named.
    pub fn contracts(&self, workspace: &Path) -> Vec<TaskContract> {
        self.subtasks
            .iter()
            .map(|node| TaskContract {
                task: node.id.clone(),
                lease: format!("split:{}:{}", self.task, node.id),
                workspace: workspace.to_path_buf(),
                allowed_files: node.paths.clone(),
                forbidden_paths: Vec::new(),
                deliverables: node.paths.clone(),
                verification_commands: vec![node.verify.clone()],
            })
            .collect()
    }

    /// Every node that claims `path`, by the most specific claim first. A node
    /// claims a path when it names it exactly, or names a directory it sits in —
    /// and only if no *other* claim of that node is deeper. Two nodes claiming at
    /// the same depth is a collision, and it is reported as one rather than
    /// resolved by picking a winner.
    pub fn owners_of(&self, path: &str) -> Vec<(NodeId, usize)> {
        let mut claims: Vec<(NodeId, usize)> = Vec::new();
        for node in &self.subtasks {
            let depth = node
                .paths
                .iter()
                .filter_map(|owned| claim_depth(owned, path))
                .max();
            if let Some(depth) = depth {
                claims.push((node.id.clone(), depth));
            }
        }
        let deepest = claims.iter().map(|(_, d)| *d).max().unwrap_or(0);
        claims.retain(|(_, d)| *d == deepest);
        claims.sort();
        claims
    }

    /// The nodes that must finish before `id`, transitively.
    fn ancestors(&self, id: &str) -> BTreeSet<NodeId> {
        let by_id: BTreeMap<&str, &Subtask> =
            self.subtasks.iter().map(|n| (n.id.as_str(), n)).collect();
        let mut found = BTreeSet::new();
        let mut queue: Vec<&Subtask> = match by_id.get(id) {
            Some(node) => node
                .needs
                .iter()
                .filter_map(|key| by_id.get(key.as_str()).copied())
                .collect(),
            None => Vec::new(),
        };
        while let Some(node) = queue.pop() {
            if found.insert(node.id.clone()) {
                queue.extend(
                    node.needs
                        .iter()
                        .filter_map(|key| by_id.get(key.as_str()).copied()),
                );
            }
        }
        found
    }

    /// Whether the split itself says `a` before `b` — that is, `a` is among the
    /// nodes `b` waits on, however far back the chain runs.
    pub fn precedes(&self, a: &str, b: &str) -> bool {
        self.ancestors(b).contains(a)
    }

    /// The baseline: one node per file the reference names, no edges at all. This
    /// is what a planner that split *nothing* would produce, and it owns the whole
    /// file set — so the only number it can lose on is ordering, where it orders
    /// zero pairs and states zero pairs. Every other split is compared to this
    /// one, and a split that does not beat it has bought no information.
    pub fn flat_baseline(reference: &Reference) -> Split {
        let mut split = Split::new(format!("baseline for {}", reference.source));
        for path in &reference.paths {
            split.push(Subtask {
                id: reference
                    .id_for(path)
                    .unwrap_or_else(|| path.replace('/', "-")),
                goal: format!("change {path}"),
                paths: vec![path.clone()],
                needs: Vec::new(),
                verify: reference.verify_for(path),
            });
        }
        split
    }

    /// Score this split against what the repository says the change is.
    pub fn score_against(&self, reference: &Reference) -> SplitScore {
        let mut score = SplitScore {
            paths_expected: reference.paths.len(),
            ..SplitScore::default()
        };
        let mut by_path: BTreeMap<String, Vec<(NodeId, usize)>> = BTreeMap::new();
        for path in &reference.paths {
            let owners = self.owners_of(path);
            by_path.insert(path.clone(), owners.clone());
            match owners.len() {
                0 => score.unclaimed.push(path.clone()),
                1 => score.paths_owned += 1,
                _ => score
                    .contested
                    .push((path.clone(), owners.into_iter().map(|(id, _)| id).collect())),
            }
        }
        score.pairs_expected = reference.before.len();
        for [first, second] in &reference.before {
            let owner_of = |path: &str| {
                by_path
                    .get(path)
                    .and_then(|owners| owners.first())
                    .map(|(id, _)| id.clone())
            };
            let (Some(a), Some(b)) = (owner_of(first), owner_of(second)) else {
                // One of the two files nobody owns: the order cannot be judged on
                // a split that left the work out, and that is the `unclaimed`
                // figure's business, not this one.
                continue;
            };
            let pair = [first.clone(), second.clone()];
            if a == b {
                score.same_node.push(pair);
            } else if self.precedes(&a, &b) {
                score.agreed += 1;
            } else if self.precedes(&b, &a) {
                score.contradicted.push(pair);
            } else {
                score.unstated.push(pair);
            }
        }
        let mut invented: Vec<String> = self
            .subtasks
            .iter()
            .flat_map(|node| node.paths.iter())
            .filter(|path| {
                !path.trim().is_empty()
                    && !reference
                        .paths
                        .iter()
                        .any(|claimed| overlaps(path, claimed))
            })
            .cloned()
            .collect();
        invented.sort();
        invented.dedup();
        score.invented = invented;
        score
    }
}

/// Whether either path reaches into the other: one names the other exactly, or
/// one names a directory the other sits in. Used to tell a broad claim from an
/// invented file — a node that owns `rust/crates/x/src/` is not planning work
/// outside the change when the change is in that directory.
fn overlaps(a: &str, b: &str) -> bool {
    claim_depth(a, b).is_some() || claim_depth(b, a).is_some()
}

fn object<'a>(
    value: &'a serde_json::Value,
    label: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, SplitError> {
    value
        .as_object()
        .ok_or_else(|| SplitError::Shape(format!("{label} is not an object")))
}

fn string_field(
    node: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    label: &str,
) -> Result<String, SplitError> {
    node.get(key)
        .and_then(|v| v.as_str())
        .map(|v| v.to_string())
        .ok_or_else(|| SplitError::Shape(format!("{label} has no string `{key}`")))
}

fn string_list(
    node: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    label: &str,
) -> Result<Vec<String>, SplitError> {
    let items = node
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| SplitError::Shape(format!("{label} has no list `{key}`")))?;
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| SplitError::Shape(format!("{label}'s `{key}` has a non-string")))
        })
        .collect()
}

/// How deep `owned` reaches into `path`: exact match is the longest, a directory
/// prefix is measured in segments, and anything else is not a claim at all.
fn claim_depth(owned: &str, path: &str) -> Option<usize> {
    let owned = owned.trim_end_matches('/');
    if owned.is_empty() {
        return None;
    }
    if owned == path {
        return Some(path.split('/').count());
    }
    let prefix = format!("{owned}/");
    if let Some(rest) = path.strip_prefix(&prefix) {
        // The claim covers the path; deeper paths claim it harder, so the depth
        // scored is the claimed path's own length.
        let _ = rest;
        return Some(path.split('/').count() - rest.split('/').count());
    }
    None
}

/// What the change really consists of, read off the repository. This is the only
/// thing a split is judged against, so it is built from facts a reader can look
/// up — the files the change touches, and which of them the build forces to land
/// first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// The files the change touches, as the repository names them.
    pub paths: Vec<String>,
    /// `pair[0]` must land before `pair[1]`, both sides drawn from `paths`.
    pub before: Vec<[String; 2]>,
    /// Where these facts came from, in words — a file list from a real diff, a
    /// build order from the manifests — so a score is never quoted without its
    /// source.
    pub source: String,
    /// The command that decides one path's change is in. Kept here so a baseline
    /// and a candidate are graded by the same rule.
    pub verification: String,
}

impl Reference {
    pub fn new(source: impl Into<String>, verification: impl Into<String>) -> Self {
        Reference {
            paths: Vec::new(),
            before: Vec::new(),
            source: source.into(),
            verification: verification.into(),
        }
    }

    pub fn verify_for(&self, _path: &str) -> String {
        self.verification.clone()
    }

    /// A stable node name for a path, used only by the baseline, which has no
    /// planner to disagree with.
    pub fn id_for(&self, path: &str) -> Option<NodeId> {
        if path.trim().is_empty() {
            return None;
        }
        Some(path.replace('/', "-"))
    }
}

/// The two numbers a split is judged on, with every disagreement kept by name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SplitScore {
    pub paths_expected: usize,
    pub paths_owned: usize,
    pub unclaimed: Vec<String>,
    pub contested: Vec<(String, Vec<NodeId>)>,
    pub pairs_expected: usize,
    /// Orders the split states itself, between two different nodes. This is the
    /// figure the baseline is beaten on.
    pub agreed: usize,
    /// Orders the split states backwards — the scheduler would launch the
    /// consumer first and it would fail.
    pub contradicted: Vec<[String; 2]>,
    /// Orders the reference states and the split leaves silent, which the
    /// scheduler reads as "start together".
    pub unstated: Vec<[String; 2]>,
    /// Orders satisfied only because both files sit in one node: safe, and worth
    /// nothing to a scheduler, so it is never folded into `agreed`.
    pub same_node: Vec<[String; 2]>,
    /// Files this split names that the change does not consist of. The two blind
    /// local runs measured on this repository produced two of these and four: one
    /// read a symptom out of the task's own prose and gave it a plausible file, the
    /// other named a crate that is not in this workspace. A score that only counts
    /// what was missed cannot show that a planner invented work.
    pub invented: Vec<String>,
}

impl SplitScore {
    /// The fraction of the file set this split owns.
    pub fn coverage(&self) -> f64 {
        if self.paths_expected == 0 {
            0.0
        } else {
            self.paths_owned as f64 / self.paths_expected as f64
        }
    }

    /// The fraction of the reference's required orders this split states correctly,
    /// counting only orders that cross a node boundary.
    pub fn ordering(&self) -> f64 {
        if self.pairs_expected == 0 {
            0.0
        } else {
            self.agreed as f64 / self.pairs_expected as f64
        }
    }

    /// The two figures, as words. A score is never quoted without both: how much
    /// of the change the split owns, and how much of the order it states.
    pub fn counts(&self) -> Vec<String> {
        vec![
            format!(
                "files owned {}/{} ( {:.0}% of the change)",
                self.paths_owned,
                self.paths_expected,
                self.coverage() * 100.0
            ),
            format!(
                "orders stated {}/{} — {} backwards, {} left silent, {} inside one node",
                self.agreed,
                self.pairs_expected,
                self.contradicted.len(),
                self.unstated.len(),
                self.same_node.len()
            ),
        ]
    }

    /// Every disagreement by name, so a reader can check the figure rather than
    /// take it. These are the lines a person acts on: the file nobody owns, the
    /// two nodes reaching for one file, the pair in the wrong order, the file the
    /// split plans that the change does not consist of.
    pub fn disagreements(&self) -> Vec<String> {
        let mut out = Vec::new();
        for path in &self.unclaimed {
            out.push(format!("  nobody owns `{path}`"));
        }
        for (path, owners) in &self.contested {
            out.push(format!(
                "  {} claim `{path}`: {}",
                owners.len(),
                owners
                    .iter()
                    .map(|id| format!("`{id}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        for pair in self
            .contradicted
            .iter()
            .chain(self.unstated.iter())
            .chain(self.same_node.iter())
        {
            let kind = if self.contradicted.contains(pair) {
                "backwards"
            } else if self.unstated.contains(pair) {
                "silent"
            } else {
                "one node"
            };
            out.push(format!("  `{}` before `{}` — {kind}", pair[0], pair[1]));
        }
        if !self.invented.is_empty() {
            for path in &self.invented {
                out.push(format!("  `{path}` is not part of the change"));
            }
        }
        out
    }

    /// The report lines: the two figures, then every disagreement named.
    pub fn lines(&self) -> Vec<String> {
        let mut out = self.counts();
        out.extend(self.disagreements());
        out
    }
}

/// Whether a split may be handed to the scheduler, and the reason it may not.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitDecision {
    pub schedulable: bool,
    /// Each refusal names the nodes or paths at fault and the number behind it,
    /// so "we did not schedule it" is never the whole answer.
    pub reasons: Vec<String>,
    pub score: SplitScore,
    pub baseline: SplitScore,
}

impl SplitDecision {
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!(
            "quality against the baseline: files owned {}/{} vs {}/{}, orders stated {} vs {}",
            self.score.paths_owned,
            self.score.paths_expected,
            self.baseline.paths_owned,
            self.baseline.paths_expected,
            self.score.agreed,
            self.baseline.agreed,
        ));
        if self.schedulable {
            out.push("  the scheduler may run this split".to_string());
        } else {
            out.push("  the scheduler will not run this split:".to_string());
            out.extend(self.reasons.iter().map(|r| format!("    - {r}")));
        }
        out
    }
}

/// Score a split, and decide whether it beats doing the work in one piece.
///
/// `flat` is the baseline this split is compared against — [`Split::flat_baseline`]
/// of the same reference — so the caller can see the comparison rather than trust
/// it. A split is refused when it leaves part of the task unowned, puts two
/// workers on one file, states a required order backwards, leaves a required order
/// unstated, or orders no more pairs than the baseline does.
pub fn decide(candidate: &Split, reference: &Reference, flat: &Split) -> SplitDecision {
    let score = candidate.score_against(reference);
    let baseline = flat.score_against(reference);
    let mut reasons = Vec::new();

    if let Err(error) = candidate.validate() {
        reasons.push(format!("the split is not a shape that can run: {error}"));
    }
    if !score.unclaimed.is_empty() {
        reasons.push(format!(
            "{} of {} files in the change are owned by no node, so nothing would write them: {}",
            score.unclaimed.len(),
            score.paths_expected,
            score
                .unclaimed
                .iter()
                .map(|p| format!("`{p}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !score.contested.is_empty() {
        reasons.push(format!(
            "{} files are claimed by more than one node, which is a lease collision before it \
             is a schedule: {}",
            score.contested.len(),
            score
                .contested
                .iter()
                .map(|(p, ids)| format!("`{p}` by {}", ids.join(", ")))
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    if !score.invented.is_empty() {
        reasons.push(format!(
            "a worker would be launched on {} file(s) this split names that the change does \
             not consist of: {}",
            score.invented.len(),
            score
                .invented
                .iter()
                .map(|p| format!("`{p}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !score.contradicted.is_empty() {
        reasons.push(format!(
            "{} required orders are stated backwards — the scheduler would start the consumer \
             before the file it reads exists: {}",
            score.contradicted.len(),
            score
                .contradicted
                .iter()
                .map(|p| format!("`{}` before `{}`", p[0], p[1]))
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    if !score.unstated.is_empty() {
        reasons.push(format!(
            "{} required orders are left silent, and a missing edge is not caution — the \
             scheduler starts those two nodes together: {}",
            score.unstated.len(),
            score
                .unstated
                .iter()
                .map(|p| format!("`{}` before `{}`", p[0], p[1]))
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    if score.agreed <= baseline.agreed {
        reasons.push(format!(
            "it orders {} pair(s) and the baseline orders {} — a split that states no more of \
             the real order than naming every file on its own is not worth scheduling; the work \
             should go as one task to one worker",
            score.agreed, baseline.agreed
        ));
    }

    SplitDecision {
        schedulable: reasons.is_empty(),
        reasons,
        score,
        baseline,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, paths: &[&str], needs: &[&str], verify: &str) -> Subtask {
        Subtask {
            id: id.to_string(),
            goal: format!("{id} does the thing"),
            paths: paths.iter().map(|p| p.to_string()).collect(),
            needs: needs.iter().map(|n| n.to_string()).collect(),
            verify: verify.to_string(),
        }
    }

    /// A change across three files where the build forces a real order:
    /// `types.rs` first, then the store that uses the type, then the main that
    /// calls the store.
    fn reference() -> Reference {
        let mut r = Reference::new("three-file change", "cargo build");
        r.paths = vec![
            "src/types.rs".to_string(),
            "src/store.rs".to_string(),
            "src/main.rs".to_string(),
        ];
        r.before = vec![
            ["src/types.rs".to_string(), "src/store.rs".to_string()],
            ["src/store.rs".to_string(), "src/main.rs".to_string()],
        ];
        r
    }

    /// The split a planner got right: three nodes, in the order the files need.
    fn correct_split() -> Split {
        let mut s = Split::new("add a store");
        s.push(node("types", &["src/types.rs"], &[], "cargo build"));
        s.push(node("store", &["src/store.rs"], &["types"], "cargo build"));
        s.push(node("main", &["src/main.rs"], &["store"], "cargo build"));
        s
    }

    #[test]
    fn a_split_that_states_the_real_order_beats_the_baseline() {
        let reference = reference();
        let decision = decide(
            &correct_split(),
            &reference,
            &Split::flat_baseline(&reference),
        );
        assert!(
            decision.schedulable,
            "a correct split was refused: {:?}",
            decision.reasons
        );
        assert_eq!(decision.score.agreed, 2);
        assert_eq!(decision.baseline.agreed, 0);
        assert_eq!(decision.score.paths_owned, 3);
    }

    #[test]
    fn the_two_figures_are_always_printed_and_only_real_disagreements_are_named() {
        // The report a reader trusts: the two counts are there even when nothing is
        // wrong, and every extra line names a file or a pair. A score that printed a
        // bare number for a refused split could not be checked against the change.
        let reference = reference();
        let good = correct_split().score_against(&reference);
        assert_eq!(good.counts().len(), 2);
        assert!(
            good.disagreements().is_empty(),
            "{:?}",
            good.disagreements()
        );

        let flat = Split::flat_baseline(&reference).score_against(&reference);
        assert_eq!(flat.counts().len(), 2);
        assert_eq!(flat.disagreements().len(), 2, "{:?}", flat.disagreements());
        assert!(flat
            .disagreements()
            .iter()
            .all(|line| line.contains("before") && line.contains("silent")));
        assert_eq!(flat.lines().len(), 4);
    }

    #[test]
    fn a_file_the_change_never_names_is_counted_and_named() {
        // Both local planner runs measured on this repository did this: one read a
        // symptom out of the task's own prose and gave it a file, the other named a
        // module that does not exist. Owning every reference file is not enough to
        // be a reading of *this* change, so the extra path is a disagreement.
        let reference = reference();
        let mut split = correct_split();
        split.push(node(
            "extra",
            &["src/hallucinated.rs"],
            &["types"],
            "cargo build",
        ));
        let score = split.score_against(&reference);
        assert_eq!(score.paths_owned, 3);
        assert_eq!(score.invented, vec!["src/hallucinated.rs".to_string()]);
        let line = &score.disagreements()[score.disagreements().len() - 1];
        assert!(line.contains("not part of the change"), "{line}");
        assert!(line.contains("hallucinated"), "{line}");
        let decision = decide(&split, &reference, &Split::flat_baseline(&reference));
        assert!(!decision.schedulable);
        assert!(
            decision
                .reasons
                .iter()
                .any(|r| r.contains("does not consist of") && r.contains("hallucinated")),
            "{:?}",
            decision.reasons
        );
    }

    #[test]
    fn a_directory_claim_is_not_a_file_outside_the_change() {
        // The honest direction of the same rule, in both shapes: a node that names
        // the directory the change lives in is describing it broadly, not inventing
        // work, and so is a node that names one file inside a directory the change
        // was stated as. Without this, a split would be refused for the shape of how
        // it wrote down files it already covers.
        let reference = reference();
        let mut split = Split::new("one wide unit");
        split.push(node("all", &["src/"], &[], "cargo build"));
        let score = split.score_against(&reference);
        assert!(score.invented.is_empty(), "{:?}", score.invented);
        assert_eq!(score.paths_owned, 3);

        let mut wide_reference = Reference::new("the whole directory", "cargo build");
        wide_reference.paths = vec!["src/".to_string()];
        let mut narrow = Split::new("one narrow unit");
        narrow.push(node("one", &["src/store.rs"], &[], "cargo build"));
        let score = narrow.score_against(&wide_reference);
        assert!(score.invented.is_empty(), "{:?}", score.invented);
    }

    #[test]
    fn the_baseline_owns_every_file_and_orders_none_of_them() {
        // The whole point of the comparison: the flat split is *safe* and useless,
        // so a candidate is not credited for owning the file set — that is the
        // part it has to match — it has to state an order the baseline cannot.
        let reference = reference();
        let flat = Split::flat_baseline(&reference).score_against(&reference);
        assert_eq!(flat.paths_owned, 3);
        assert_eq!(flat.agreed, 0);
        assert_eq!(flat.unstated.len(), 2);
        assert_eq!(flat.contradicted.len(), 0);
        assert!(flat.same_node.is_empty());
    }

    #[test]
    fn an_order_left_silent_is_a_refusal_not_a_neither_of_these() {
        // The defect this rule exists to catch: `store` and `main` are both nodes
        // and neither waits on the other, so the scheduler starts them together
        // and `main` fails against a store that is not written yet. A score that
        // read "no edge, no complaint" would hand this split a clean sheet.
        let reference = reference();
        let mut split = Split::new("add a store");
        split.push(node("types", &["src/types.rs"], &[], "cargo build"));
        split.push(node("store", &["src/store.rs"], &["types"], "cargo build"));
        split.push(node("main", &["src/main.rs"], &["types"], "cargo build"));
        let decision = decide(&split, &reference, &Split::flat_baseline(&reference));
        assert!(!decision.schedulable);
        assert_eq!(decision.score.unstated.len(), 1);
        assert_eq!(decision.score.agreed, 1);
        assert!(
            decision
                .reasons
                .iter()
                .any(|r| r.contains("left silent") && r.contains("src/main.rs")),
            "{:?}",
            decision.reasons
        );
    }

    #[test]
    fn an_order_stated_backwards_is_named_and_refused() {
        let reference = reference();
        let mut split = Split::new("add a store");
        split.push(node("main", &["src/main.rs"], &[], "cargo build"));
        split.push(node("store", &["src/store.rs"], &["main"], "cargo build"));
        split.push(node("types", &["src/types.rs"], &["store"], "cargo build"));
        let decision = decide(&split, &reference, &Split::flat_baseline(&reference));
        assert!(!decision.schedulable);
        assert_eq!(decision.score.contradicted.len(), 2);
        assert!(
            decision
                .reasons
                .iter()
                .any(|r| r.contains("stated backwards")),
            "{:?}",
            decision.reasons
        );
    }

    #[test]
    fn a_file_nobody_owns_and_a_file_two_nodes_claim_are_both_named() {
        let reference = reference();
        let mut split = Split::new("add a store");
        // `src/main.rs` is named by nobody, and two nodes both reach for the store.
        split.push(node(
            "a",
            &["src/types.rs", "src/store.rs"],
            &[],
            "cargo build",
        ));
        split.push(node("b", &["src/store.rs"], &["a"], "cargo build"));
        split.push(node("c", &["README.md"], &[], "cargo build"));
        let decision = decide(&split, &reference, &Split::flat_baseline(&reference));
        assert!(!decision.schedulable);
        assert_eq!(decision.score.unclaimed, vec!["src/main.rs".to_string()]);
        assert_eq!(
            decision.score.contested,
            vec![(
                "src/store.rs".to_string(),
                vec!["a".to_string(), "b".to_string()]
            )]
        );
        assert!(decision
            .reasons
            .iter()
            .any(|r| r.contains("owned by no node")));
        assert!(decision
            .reasons
            .iter()
            .any(|r| r.contains("lease collision")));
    }

    #[test]
    fn a_directory_claim_covers_a_file_the_split_never_named_exactly() {
        // The other half of ownership: a node that says `src/` is on the hook for
        // everything under it, so those files are owned rather than "unclaimed
        // because the planner spelled the path more coarsely than the diff did".
        // A split is judged on the work it covers, not on how it wrote the names.
        let reference = reference();
        let mut split = Split::new("add a store");
        split.push(node("wide", &["src/"], &[], "cargo build"));
        let score = split.score_against(&reference);
        assert_eq!(score.paths_owned, 3);
        assert!(score.unclaimed.is_empty());
    }

    #[test]
    fn a_split_that_orders_nothing_more_than_the_baseline_is_not_scheduled() {
        // Equal is not better. Three nodes inside one directory, all independent:
        // the file set is covered and nothing is contradicted, and the scheduler
        // would still run all three at once for no reason a person agreed to.
        let reference = reference();
        let mut split = Split::new("add a store");
        split.push(node("all", &["src/"], &[], "cargo build"));
        let decision = decide(&split, &reference, &Split::flat_baseline(&reference));
        assert!(!decision.schedulable);
        // Both files of every pair sit in the same node, which is safe and worth
        // nothing — and it is counted separately so the number cannot hide it.
        assert_eq!(decision.score.agreed, 0);
        assert_eq!(decision.score.same_node.len(), 2);
        assert!(decision
            .reasons
            .iter()
            .any(|r| r.contains("not worth scheduling") && r.contains("one task to one worker")),
            "{:?}",
            decision.reasons
        );
    }

    #[test]
    fn the_baseline_split_is_itself_refused_having_stated_no_order() {
        // Not a trick: the reason the baseline is the comparison and not a plan.
        let reference = reference();
        let flat = Split::flat_baseline(&reference);
        let decision = decide(&flat, &reference, &flat);
        assert!(!decision.schedulable);
        assert_eq!(decision.score.paths_owned, 3);
        assert_eq!(decision.baseline.paths_owned, 3);
    }

    #[test]
    fn a_node_with_no_command_is_refused_because_it_cannot_be_graded() {
        let reference = reference();
        let mut split = Split::new("add a store");
        split.push(node("types", &["src/types.rs"], &[], ""));
        split.push(node(
            "rest",
            &["src/store.rs", "src/main.rs"],
            &["types"],
            "",
        ));
        let decision = decide(&split, &reference, &Split::flat_baseline(&reference));
        assert!(!decision.schedulable);
        assert!(decision
            .reasons
            .iter()
            .any(|r| r.contains("its completion is unverifiable")),);
    }

    #[test]
    fn a_cycle_is_refused_by_the_graph_rule_not_a_second_one() {
        let reference = reference();
        let mut split = Split::new("add a store");
        split.push(node("a", &["src/types.rs"], &["b"], "cargo build"));
        split.push(node("b", &["src/store.rs"], &["a"], "cargo build"));
        split.push(node("c", &["src/main.rs"], &["b"], "cargo build"));
        let error = split.validate().unwrap_err();
        assert!(
            matches!(
                error,
                SplitError::Graph(crate::scheduler::GraphError::Cycle(_))
            ),
            "{error:?}"
        );
        let decision = decide(&split, &reference, &Split::flat_baseline(&reference));
        assert!(decision
            .reasons
            .iter()
            .any(|r| r.starts_with("the split is not a shape that can run")));
    }

    #[test]
    fn ownership_is_deepest_claim_wins_across_a_directory_boundary() {
        let mut split = Split::new("dir");
        split.push(node("wide", &["src/"], &[], "cargo build"));
        split.push(node("narrow", &["src/store.rs"], &[], "cargo build"));
        let owners = split.owners_of("src/store.rs");
        assert_eq!(owners, vec![("narrow".to_string(), 2)]);
        // A file only the directory covers is still owned, by the shallow claim.
        assert_eq!(
            split.owners_of("src/types.rs"),
            vec![("wide".to_string(), 1)]
        );
    }

    #[test]
    fn a_split_becomes_the_graph_the_scheduler_runs_and_the_contract_it_launches_under() {
        let split = correct_split();
        let graph = split.to_task_graph();
        assert_eq!(graph.nodes().len(), 3);
        // The command a node runs is the command that verifies it — nothing is
        // substituted with a placeholder that always exits 0.
        assert_eq!(graph.nodes()[0].command, "cargo build");
        let contracts = split.contracts(Path::new("/work"));
        assert_eq!(contracts.len(), 3);
        assert_eq!(contracts[1].allowed_files, vec!["src/store.rs".to_string()]);
        assert_eq!(
            contracts[1].verification_commands,
            vec!["cargo build".to_string()]
        );
    }

    #[test]
    fn a_well_shaped_answer_reads_straight_through() {
        let value = serde_json::json!({
            "task": "add a store",
            "subtasks": [
                {"id": "types", "goal": "define the record",
                 "paths": ["src/types.rs"], "needs": [], "verify": "cargo build"},
                {"id": "store", "goal": "store the record",
                 "paths": ["src/store.rs"], "needs": ["types"], "verify": "cargo build"},
            ]
        });
        let split = Split::from_value(&value).expect("the answer should read");
        assert_eq!(split.task, "add a store");
        assert_eq!(split.subtasks.len(), 2);
        assert_eq!(split.subtasks[1].needs, vec!["types".to_string()]);
    }

    #[test]
    fn a_dependency_field_spelled_otherwise_is_refused_not_read_as_no_dependency() {
        // The dangerous one. A planner that wrote `dependencies` instead of `needs`
        // would otherwise come back as a node with nothing to wait for, and the
        // scheduler would start it beside the file it reads. Silence about an order
        // has to be an error, never an empty list.
        let value = serde_json::json!({
            "task": "t",
            "subtasks": [
                {"id": "a", "goal": "g", "paths": ["a.rs"], "needs": [],
                 "verify": "cargo build"},
                {"id": "b", "goal": "g", "paths": ["b.rs"], "dependencies": ["a"],
                 "verify": "cargo build"},
            ]
        });
        let error = Split::from_value(&value).unwrap_err();
        assert!(
            matches!(&error, SplitError::Shape(why) if why.contains("dependencies")
                && why.contains("`b`")),
            "{error:?}"
        );
    }

    #[test]
    fn an_answer_missing_any_slot_says_which_one() {
        for (label, value) in [
            ("no subtasks", serde_json::json!({"task": "t"})),
            (
                "no verify",
                serde_json::json!({"task": "t", "subtasks": [
                    {"id": "a", "goal": "g", "paths": ["a.rs"], "needs": []}]}),
            ),
            (
                "no needs",
                serde_json::json!({"task": "t", "subtasks": [
                    {"id": "a", "goal": "g", "paths": ["a.rs"], "verify": "x"}]}),
            ),
            (
                "an empty split",
                serde_json::json!({"task": "t", "subtasks": []}),
            ),
        ] {
            match Split::from_value(&value) {
                Ok(split) => panic!("{label} read as a split: {:?}", split.subtasks),
                Err(error) => assert!(
                    matches!(&error, SplitError::Shape(_)),
                    "{label} came back as {error:?}"
                ),
            }
        }
    }
}
