//! `OR-2` — the task graph and scheduler.
//!
//! Two layers, the same split `tasks.rs` uses: [`TaskGraph`] is pure state —
//! nodes, their dependencies, and which are ready — and is testable without
//! spawning anything; [`Scheduler`] sits on top and drives real commands through
//! the [`TaskManager`], so a scheduled node is an actual `sh -c` child, not a
//! stand-in for one. Nothing here pretends to run work: readiness is computed
//! from state, but the running itself is the OS's.
//!
//! The two rules the item is built to prove live in the pure layer, where they
//! can be checked on any shape rather than only the one the test happens to
//! run:
//!
//! - A node is **ready** once every node it needs has finished. Two nodes with
//!   no unsatisfied need and room in the queue start together — that is the
//!   parallel branch — and a third that needs one of them waits until it is
//!   actually done, not until the pass it was discovered in.
//! - The queue's capacity is `min(workers, verification throughput)`, not the
//!   worker count. A machine that can launch eight agents but verify two at a
//!   time has a real concurrency of two: the surplus workers would only pile
//!   finished work up against a verification step that cannot consume it. The
//!   scheduler honours the smaller number and [`Scheduler::binding`] names which
//!   of the two it was, so the limit that bites is reported rather than hidden
//!   behind a worker count that looks larger.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use crate::tasks::{TaskManager, TaskStatus};

/// A task, named by a human-readable id the graph keys its edges on.
pub type NodeId = String;

/// One schedulable unit: a command and the set of nodes that must finish first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskNode {
    pub id: NodeId,
    pub command: String,
    /// Nodes whose completion this one waits on. Empty means it can start at once.
    pub needs: Vec<NodeId>,
}

/// Why a graph is not something that can be run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// Two nodes share an id, so an edge would not name one node unambiguously.
    DuplicateNode(NodeId),
    /// A node needs a name that is not a node in the graph.
    UnknownDependency { node: NodeId, needs: NodeId },
    /// A node depends on itself, directly.
    SelfDependency(NodeId),
    /// A cycle, named by the nodes on it. No node in a cycle can ever be ready,
    /// so this is caught before running rather than surfacing as a hang.
    Cycle(Vec<NodeId>),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::DuplicateNode(id) => write!(f, "two nodes are both named `{id}`"),
            GraphError::UnknownDependency { node, needs } => {
                write!(f, "`{node}` needs `{needs}`, which is not a node")
            }
            GraphError::SelfDependency(id) => write!(f, "`{id}` needs itself"),
            GraphError::Cycle(ids) => write!(
                f,
                "these tasks form a dependency cycle: {}",
                ids.join(" → ")
            ),
        }
    }
}

impl std::error::Error for GraphError {}

/// The dependency graph, held in node insertion order so readiness is
/// deterministic — the same run always picks the same node first when two are
/// ready at once.
#[derive(Debug, Clone, Default)]
pub struct TaskGraph {
    nodes: Vec<TaskNode>,
}

impl TaskGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, node: TaskNode) {
        self.nodes.push(node);
    }

    pub fn nodes(&self) -> &[TaskNode] {
        &self.nodes
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Check the graph is runnable before anything is launched: unique ids, every
    /// edge pointing at a real node, no self-loop, no cycle. A schedule built on a
    /// cycle would wait forever on a node that can never finish, so this fails the
    /// run at the door rather than hanging it.
    pub fn validate(&self) -> Result<(), GraphError> {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for node in &self.nodes {
            if !seen.insert(node.id.as_str()) {
                return Err(GraphError::DuplicateNode(node.id.clone()));
            }
        }
        let ids: BTreeSet<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();
        for node in &self.nodes {
            for need in &node.needs {
                if need == &node.id {
                    return Err(GraphError::SelfDependency(node.id.clone()));
                }
                if !ids.contains(need.as_str()) {
                    return Err(GraphError::UnknownDependency {
                        node: node.id.clone(),
                        needs: need.clone(),
                    });
                }
            }
        }
        // A cycle is the only structural failure left; find it with a DFS that
        // keeps the recursion stack so the back edge names the actual loop.
        if let Some(cycle) = self.find_cycle() {
            return Err(GraphError::Cycle(cycle));
        }
        Ok(())
    }

    fn find_cycle(&self) -> Option<Vec<NodeId>> {
        let by_id: BTreeMap<&str, &TaskNode> =
            self.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
        // 0 = unvisited, 1 = on the current path, 2 = fully explored.
        let mut state: BTreeMap<String, u8> = BTreeMap::new();
        let mut path: Vec<NodeId> = Vec::new();
        for node in &self.nodes {
            if Self::dfs_cycle(node.id.as_str(), &by_id, &mut state, &mut path) {
                // The path ends with a repeat of a node still on it; trim to the loop.
                let last = path.last()?.clone();
                if let Some(start) = path.iter().position(|p| *p == last) {
                    let mut cycle = path[start..].to_vec();
                    cycle.push(last);
                    return Some(cycle);
                }
                return Some(path);
            }
        }
        None
    }

    fn dfs_cycle(
        id: &str,
        by_id: &BTreeMap<&str, &TaskNode>,
        state: &mut BTreeMap<String, u8>,
        path: &mut Vec<NodeId>,
    ) -> bool {
        match state.get(id).copied().unwrap_or(0) {
            1 => return true,  // back edge to a node still on the path
            2 => return false, // already proven acyclic from here
            _ => {}
        }
        state.insert(id.to_string(), 1);
        path.push(id.to_string());
        if let Some(node) = by_id.get(id) {
            for need in &node.needs {
                if Self::dfs_cycle(need, by_id, state, path) {
                    return true;
                }
            }
        }
        path.pop();
        state.insert(id.to_string(), 2);
        false
    }

    /// The nodes that can start now: not already finished, every node they need
    /// has finished. Kept in insertion order, so with two ready nodes the one
    /// added first is the one offered first — the schedule is reproducible.
    pub fn ready(&self, done: &BTreeSet<NodeId>) -> Vec<&TaskNode> {
        self.nodes
            .iter()
            .filter(|n| !done.contains(&n.id))
            .filter(|n| n.needs.iter().all(|need| done.contains(need)))
            .collect()
    }

    /// The serial bottleneck: the node with more than one dependency that sits on
    /// the longest dependency chain. It is where independent branches are forced
    /// back into one line of work, so it is the honest answer to "what caps this
    /// graph's parallelism" — named rather than left for a reader to infer from a
    /// queue depth. `None` when the graph is already linear or has no join.
    pub fn bottleneck(&self) -> Option<&TaskNode> {
        let longest = self.longest_chain()?;
        // The first node on the critical path that is a join (more than one
        // dependency) — where two branches were forced to wait on each other.
        longest
            .iter()
            .find_map(|id| self.nodes.iter().find(|n| &n.id == id && n.needs.len() > 1))
    }

    /// The longest chain of dependencies (by node count), as ids ordered from the
    /// deepest ancestor down to the node that ends it. This is the minimum wall
    /// clock no amount of workers can beat, so it is the other half of "what is
    /// the real limit."
    pub fn longest_chain(&self) -> Option<Vec<NodeId>> {
        let by_id: BTreeMap<&str, &TaskNode> =
            self.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
        let mut memo: BTreeMap<String, Vec<NodeId>> = BTreeMap::new();
        let mut best: Option<Vec<NodeId>> = None;
        for node in &self.nodes {
            let chain = Self::chain_through(node.id.as_str(), &by_id, &mut memo);
            if best.as_ref().is_none_or(|b| chain.len() > b.len()) {
                best = Some(chain);
            }
        }
        best
    }

    fn chain_through(
        id: &str,
        by_id: &BTreeMap<&str, &TaskNode>,
        memo: &mut BTreeMap<String, Vec<NodeId>>,
    ) -> Vec<NodeId> {
        if let Some(hit) = memo.get(id) {
            return hit.clone();
        }
        let mut longest_from_needs: Vec<NodeId> = Vec::new();
        if let Some(node) = by_id.get(id) {
            for need in &node.needs {
                let up = Self::chain_through(need, by_id, memo);
                if up.len() > longest_from_needs.len() {
                    longest_from_needs = up;
                }
            }
        }
        let mut chain = longest_from_needs;
        chain.push(id.to_string());
        memo.insert(id.to_string(), chain.clone());
        chain
    }
}

/// Which end of `min(workers, verification throughput)` was the smaller one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    Workers,
    Verification,
    /// Both are the same number, so neither is more binding than the other.
    Both,
}

impl Binding {
    pub fn label(&self) -> &'static str {
        match self {
            Binding::Workers => "workers",
            Binding::Verification => "verification throughput",
            Binding::Both => "workers and verification (equal)",
        }
    }
}

/// How one node ended, and when, in milliseconds since the schedule started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeOutcome {
    pub id: NodeId,
    pub status: TaskStatus,
    /// Milliseconds from the start of the run to this node being launched.
    pub started_ms: u64,
    /// Milliseconds from the start to this node finishing.
    pub finished_ms: u64,
}

/// What actually ran, so the done-when can be checked against recorded times
/// rather than a claim. The scheduler keeps no ledger of its own: this is the
/// report, and the command output lives in the [`TaskManager`] the caller passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleReport {
    /// The nodes in the order they were launched.
    pub launch_order: Vec<NodeId>,
    /// One entry per launched node, in launch order.
    pub outcomes: BTreeMap<NodeId, NodeOutcome>,
    /// The highest number of nodes observed running at the same instant. This is
    /// the proof two independent branches did run together, not merely could.
    pub peak_concurrency: usize,
    /// The capacity the run used: `min(workers, verification throughput)`.
    pub capacity: usize,
    /// Which end of that minimum set the capacity.
    pub binding: Binding,
    /// The named serial bottleneck, if the graph has one.
    pub bottleneck: Option<NodeId>,
    /// The critical path — the longest chain no added worker shortens.
    pub critical_path: Vec<NodeId>,
    /// Total wall clock of the run.
    pub elapsed_ms: u64,
}

impl ScheduleReport {
    pub fn outcome(&self, id: &str) -> Option<&NodeOutcome> {
        self.outcomes.get(id)
    }
}

/// Why a run could not be scheduled at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    Graph(GraphError),
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduleError::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// The scheduler. It owns no task records — the [`TaskManager`] the caller hands
/// each [`Scheduler::run`] does — so switching who runs what never copies the
/// work state; both read one manager.
#[derive(Debug, Clone)]
pub struct Scheduler {
    workers: usize,
    verification_throughput: usize,
}

impl Scheduler {
    /// `workers` is how many agents could run at once; `verification_throughput`
    /// is how many finished results can be checked at once. The effective
    /// capacity is the smaller, floored at one so a zero on either side still
    /// makes progress instead of a deadlock on an empty queue.
    pub fn new(workers: usize, verification_throughput: usize) -> Self {
        Self {
            workers,
            verification_throughput,
        }
    }

    /// `min(workers, verification throughput)`, at least one.
    pub fn capacity(&self) -> usize {
        self.workers.min(self.verification_throughput).max(1)
    }

    /// Which end of the minimum was binding for this configuration.
    pub fn binding(&self) -> Binding {
        match self.workers.cmp(&self.verification_throughput) {
            std::cmp::Ordering::Less => Binding::Workers,
            std::cmp::Ordering::Greater => Binding::Verification,
            std::cmp::Ordering::Equal => Binding::Both,
        }
    }

    /// Run the graph to completion on real subprocesses through `manager`.
    ///
    /// Each pass reaps whatever has finished, then launches as many ready nodes
    /// as there is room for, then waits a short interval if it launched nothing
    /// and is still waiting on work. It is a `&mut` borrow of the manager, so one
    /// schedule cannot race another over the same children.
    pub async fn run(
        &self,
        graph: &TaskGraph,
        manager: &mut TaskManager,
    ) -> Result<ScheduleReport, ScheduleError> {
        graph.validate().map_err(ScheduleError::Graph)?;
        let capacity = self.capacity();
        let mut done: BTreeSet<NodeId> = BTreeSet::new();
        let mut outcomes: BTreeMap<NodeId, NodeOutcome> = BTreeMap::new();
        let mut launch_order: Vec<NodeId> = Vec::new();
        let mut running: BTreeMap<NodeId, u64> = BTreeMap::new(); // id -> manager task id
        let mut launched: BTreeSet<NodeId> = BTreeSet::new();
        let start = Instant::now();
        let mut peak_concurrency = 0usize;

        loop {
            // Reap first: a node only becomes "done" when its child has actually
            // exited, so readiness downstream is driven by real completion.
            let mut finished_now = false;
            for id in running.clone().into_keys() {
                let task_id = running[&id];
                let Ok(record) = manager.poll(task_id).await else {
                    continue;
                };
                if !matches!(record.status, TaskStatus::Running) {
                    running.remove(&id);
                    let started_ms = outcomes
                        .get(&id)
                        .map(|o| o.started_ms)
                        .unwrap_or_else(|| millis(start));
                    outcomes.insert(
                        id.clone(),
                        NodeOutcome {
                            id: id.clone(),
                            status: record.status.clone(),
                            started_ms,
                            finished_ms: millis(start),
                        },
                    );
                    done.insert(id);
                    finished_now = true;
                }
            }
            peak_concurrency = peak_concurrency.max(running.len());

            // Launch everything ready that fits.
            let room_before = capacity.saturating_sub(running.len());
            let mut launched_now = 0usize;
            if room_before > 0 {
                for node in graph.ready(&done) {
                    if launched.contains(&node.id) || launched_now >= room_before {
                        continue;
                    }
                    let task_id = manager.start(&node.id, &node.command).await.ok();
                    if let Some(task_id) = task_id {
                        running.insert(node.id.clone(), task_id);
                        launched.insert(node.id.clone());
                        launch_order.push(node.id.clone());
                        outcomes.insert(
                            node.id.clone(),
                            NodeOutcome {
                                id: node.id.clone(),
                                status: TaskStatus::Running,
                                started_ms: millis(start),
                                finished_ms: 0,
                            },
                        );
                        launched_now += 1;
                    }
                }
                peak_concurrency = peak_concurrency.max(running.len());
            }

            if done.len() == graph.nodes().len() {
                break;
            }
            // Made no progress this pass (nothing reaped, nothing launched) and
            // nothing in flight to wait for means the queue is stuck: either an
            // empty graph or a child that would not spawn. Break rather than
            // spin — the report will show the unfinished nodes.
            if running.is_empty() && !finished_now && launched_now == 0 {
                break;
            }
            // Nothing to do but wait for a child to exit. Sleep one poll interval
            // so a full queue does not busy-spin `try_wait` at 100% CPU; when work
            // did finish or start this pass, loop again at once.
            if !finished_now && launched_now == 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }

        Ok(ScheduleReport {
            launch_order,
            outcomes,
            peak_concurrency,
            capacity,
            binding: self.binding(),
            bottleneck: graph.bottleneck().map(|n| n.id.clone()),
            critical_path: graph.longest_chain().unwrap_or_default(),
            elapsed_ms: millis(start),
        })
    }
}

fn millis(start: Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, command: &str, needs: &[&str]) -> TaskNode {
        TaskNode {
            id: id.to_string(),
            command: command.to_string(),
            needs: needs.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// A → C, B → (both) → D: two independent branches, a third node that waits
    /// on one of them, and a join that is the only serial point.
    fn four_node() -> TaskGraph {
        let mut g = TaskGraph::new();
        g.add(node("A", "true", &[]));
        g.add(node("B", "true", &[]));
        g.add(node("C", "true", &["A"]));
        g.add(node("D", "true", &["B", "C"]));
        g
    }

    #[test]
    fn a_valid_graph_starts_with_only_the_roots_ready() {
        let g = four_node();
        g.validate().expect("the four-node shape is a DAG");
        let none_done = BTreeSet::new();
        let ready: Vec<&str> = g.ready(&none_done).iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ready, vec!["A", "B"], "only the two branch heads are ready");

        let mut a_done = BTreeSet::new();
        a_done.insert("A".to_string());
        let ready: Vec<&str> = g.ready(&a_done).iter().map(|n| n.id.as_str()).collect();
        assert_eq!(
            ready,
            vec!["B", "C"],
            "C becomes ready once A is done; D still waits on B and C"
        );
    }

    #[test]
    fn the_bottleneck_is_the_join_and_the_critical_path_names_the_chain() {
        let g = four_node();
        assert_eq!(
            g.bottleneck().map(|n| n.id.as_str()),
            Some("D"),
            "D is where the two branches converge into one serial step"
        );
        assert_eq!(
            g.longest_chain(),
            Some(vec!["A".into(), "C".into(), "D".into()]),
            "the path no added worker shortens"
        );
    }

    #[test]
    fn a_linear_graph_has_no_bottleneck_but_the_queue_is_still_one_deep() {
        let mut g = TaskGraph::new();
        g.add(node("A", "true", &[]));
        g.add(node("B", "true", &["A"]));
        g.add(node("C", "true", &["B"]));
        assert!(
            g.bottleneck().is_none(),
            "nothing joins, so nothing serialises"
        );
        assert_eq!(g.longest_chain().map(|c| c.len()), Some(3));
    }

    #[test]
    fn validate_rejects_the_three_structural_faults() {
        let mut dup = TaskGraph::new();
        dup.add(node("A", "true", &[]));
        dup.add(node("A", "true", &[]));
        assert_eq!(dup.validate(), Err(GraphError::DuplicateNode("A".into())));

        let mut dangling = TaskGraph::new();
        dangling.add(node("A", "true", &["ghost"]));
        assert_eq!(
            dangling.validate(),
            Err(GraphError::UnknownDependency {
                node: "A".into(),
                needs: "ghost".into()
            })
        );

        let mut cycle = TaskGraph::new();
        cycle.add(node("A", "true", &["B"]));
        cycle.add(node("B", "true", &["A"]));
        assert!(
            matches!(cycle.validate(), Err(GraphError::Cycle(_))),
            "a two-node cycle must be named, not run"
        );
    }

    #[test]
    fn capacity_is_the_smaller_of_workers_and_verification() {
        // Eight agents, two verifications: real concurrency is two, and the
        // binding is the verification side that actually caps it.
        let s = Scheduler::new(8, 2);
        assert_eq!(s.capacity(), 2);
        assert_eq!(s.binding(), Binding::Verification);
        // A machine with fewer workers than verifiers is bound by the workers.
        let s = Scheduler::new(2, 8);
        assert_eq!(s.capacity(), 2);
        assert_eq!(s.binding(), Binding::Workers);
        // Zero on either side still makes progress rather than deadlocking empty.
        assert_eq!(Scheduler::new(0, 5).capacity(), 1);
    }

    #[tokio::test]
    async fn four_real_subprocesses_run_two_at_a_time_and_the_third_waits() {
        // The done-when, against actual children: two branch heads sleep long
        // enough to overlap, the dependent node does not start until its need has
        // exited, the join is named, and every node finishes for real.
        let mut g = TaskGraph::new();
        g.add(node("A", "sleep 0.2; echo a", &[]));
        g.add(node("B", "sleep 0.2; echo b", &[]));
        g.add(node("C", "echo c", &["A"]));
        g.add(node("D", "echo d", &["B", "C"]));
        let scheduler = Scheduler::new(2, 2);
        let mut manager = TaskManager::new();
        let report = scheduler.run(&g, &mut manager).await.expect("runs");

        // All four reached a clean exit — the commands actually ran.
        for id in ["A", "B", "C", "D"] {
            let outcome = report
                .outcome(id)
                .unwrap_or_else(|| panic!("{id} never ran"));
            assert_eq!(outcome.status, TaskStatus::Exited(0), "{id} did not exit 0");
        }

        // "runs both": the two independent branch heads were both launched before
        // anything finished, so they overlap in wall clock.
        assert_eq!(report.launch_order[0], "A");
        assert_eq!(report.launch_order[1], "B");
        let a = report.outcome("A").unwrap();
        let b = report.outcome("B").unwrap();
        assert!(
            a.started_ms < b.finished_ms && b.started_ms < a.finished_ms,
            "the two branches genuinely overlapped: A {a:?} B {b:?}"
        );
        assert!(report.peak_concurrency >= 2, "two ran at once");

        // "the third waits": C started only after A had actually exited.
        let c = report.outcome("C").unwrap();
        assert!(
            c.started_ms >= a.finished_ms,
            "C waited on A, so it cannot start before A finished: A {a:?} C {c:?}"
        );

        // "the serial bottleneck is named in the output rather than hidden".
        assert_eq!(report.bottleneck.as_deref(), Some("D"));
        assert_eq!(report.critical_path, vec!["A", "C", "D"]);
        assert_eq!(report.capacity, 2);
    }

    #[tokio::test]
    async fn a_verification_bound_queue_still_finishes_every_node() {
        // Five independent nodes, two verifications: capacity is two, so all five
        // run but never more than two at once, and none is lost.
        let mut g = TaskGraph::new();
        for i in 0..5 {
            g.add(node(&format!("n{i}"), "true", &[]));
        }
        let scheduler = Scheduler::new(9, 2);
        let mut manager = TaskManager::new();
        let report = scheduler.run(&g, &mut manager).await.expect("runs");
        assert_eq!(report.capacity, 2);
        assert_eq!(report.launch_order.len(), 5);
        assert!(report.peak_concurrency <= 2, "never exceeds capacity");
        assert_eq!(
            report
                .outcomes
                .values()
                .filter(|o| o.status == TaskStatus::Exited(0))
                .count(),
            5
        );
    }
}
