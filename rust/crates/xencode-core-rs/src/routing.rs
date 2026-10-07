//! OR-6 — capability-gated routing.
//!
//! Choose a worker only from probed capabilities (`AR-3`) plus current load
//! and cost ceiling — never a vendor name.
//!
//! A task requiring a capability that only one agent has cannot be routed to
//! the others even when they are completely idle.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A candidate worker evaluated purely by probed capabilities, load, and cost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerCandidate {
    /// Worker identifier (e.g., worker handle, token, or session key).
    pub id: String,
    /// Capabilities confirmed by probe (never inferred from name or unverified docs).
    pub probed_capabilities: BTreeSet<String>,
    /// Currently active tasks assigned to this worker.
    pub current_load: usize,
    /// Maximum concurrent tasks allowed for this worker before it is saturated.
    pub max_load: usize,
    /// Estimated cost for executing a task on this worker (e.g. in dollars).
    pub estimated_cost: f64,
}

/// Requirements for a task dispatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskRequirement {
    /// Task identifier.
    pub task_id: String,
    /// Set of capabilities required by this task.
    pub required_capabilities: BTreeSet<String>,
    /// Optional maximum cost ceiling allowed for this task.
    pub cost_ceiling: Option<f64>,
}

/// Reason why a worker candidate was rejected during routing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RejectionReason {
    /// The worker lacks one or more required capabilities confirmed by probe.
    MissingCapabilities(Vec<String>),
    /// The worker is currently at or over capacity.
    LoadExceeded { current: usize, max: usize },
    /// The worker's estimated cost exceeds the specified cost ceiling.
    CostCeilingExceeded { cost: f64, ceiling: f64 },
}

/// Evaluation record for a single worker candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub worker_id: String,
    pub eligible: bool,
    pub rejection: Option<RejectionReason>,
    pub probed_capabilities: Vec<String>,
    pub current_load: usize,
    pub estimated_cost: f64,
}

/// The result of routing a task across worker candidates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub task_id: String,
    pub selected_worker: Option<String>,
    pub explanation: String,
    pub candidate_evaluations: Vec<CandidateEvaluation>,
}

/// Capability-gated router choosing workers based strictly on probed capabilities,
/// load, and cost ceiling.
pub struct CapabilityRouter;

impl CapabilityRouter {
    /// Evaluates candidate workers and chooses the best eligible worker for `task`.
    ///
    /// Rules:
    /// 1. Must satisfy all `required_capabilities` from `probed_capabilities`.
    ///    An idle worker lacking any required capability is strictly rejected.
    /// 2. Must not exceed `max_load`.
    /// 3. Must not exceed `cost_ceiling` if specified.
    /// 4. Eligible candidates are ranked by:
    ///    a. Lowest `current_load` (distribute load)
    ///    b. Lowest `estimated_cost` (minimize cost)
    ///    c. Deterministic tie-break by `worker_id`
    pub fn route(
        task: &TaskRequirement,
        candidates: &[WorkerCandidate],
    ) -> Result<RoutingDecision, String> {
        if candidates.is_empty() {
            return Err("no worker candidates provided for routing".to_string());
        }

        let mut evaluations = Vec::new();
        let mut eligible_candidates: Vec<&WorkerCandidate> = Vec::new();

        for candidate in candidates {
            // 1. Probed capabilities check: must have all required capabilities
            let missing: Vec<String> = task
                .required_capabilities
                .iter()
                .filter(|req| !candidate.probed_capabilities.contains(*req))
                .cloned()
                .collect();

            if !missing.is_empty() {
                evaluations.push(CandidateEvaluation {
                    worker_id: candidate.id.clone(),
                    eligible: false,
                    rejection: Some(RejectionReason::MissingCapabilities(missing)),
                    probed_capabilities: candidate.probed_capabilities.iter().cloned().collect(),
                    current_load: candidate.current_load,
                    estimated_cost: candidate.estimated_cost,
                });
                continue;
            }

            // 2. Load capacity check
            if candidate.current_load >= candidate.max_load {
                evaluations.push(CandidateEvaluation {
                    worker_id: candidate.id.clone(),
                    eligible: false,
                    rejection: Some(RejectionReason::LoadExceeded {
                        current: candidate.current_load,
                        max: candidate.max_load,
                    }),
                    probed_capabilities: candidate.probed_capabilities.iter().cloned().collect(),
                    current_load: candidate.current_load,
                    estimated_cost: candidate.estimated_cost,
                });
                continue;
            }

            // 3. Cost ceiling check
            if let Some(ceiling) = task.cost_ceiling {
                if candidate.estimated_cost > ceiling {
                    evaluations.push(CandidateEvaluation {
                        worker_id: candidate.id.clone(),
                        eligible: false,
                        rejection: Some(RejectionReason::CostCeilingExceeded {
                            cost: candidate.estimated_cost,
                            ceiling,
                        }),
                        probed_capabilities: candidate
                            .probed_capabilities
                            .iter()
                            .cloned()
                            .collect(),
                        current_load: candidate.current_load,
                        estimated_cost: candidate.estimated_cost,
                    });
                    continue;
                }
            }

            // Passed all checks
            evaluations.push(CandidateEvaluation {
                worker_id: candidate.id.clone(),
                eligible: true,
                rejection: None,
                probed_capabilities: candidate.probed_capabilities.iter().cloned().collect(),
                current_load: candidate.current_load,
                estimated_cost: candidate.estimated_cost,
            });
            eligible_candidates.push(candidate);
        }

        if eligible_candidates.is_empty() {
            let explanation = format!(
                "Routing refused for task '{}': no worker candidates satisfy all requirements \
                 (required capabilities: {:?}, cost ceiling: {:?})",
                task.task_id, task.required_capabilities, task.cost_ceiling
            );
            return Ok(RoutingDecision {
                task_id: task.task_id.clone(),
                selected_worker: None,
                explanation,
                candidate_evaluations: evaluations,
            });
        }

        // Sort eligible candidates: lowest load, then lowest cost, then worker_id
        eligible_candidates.sort_by(|a, b| {
            a.current_load
                .cmp(&b.current_load)
                .then_with(|| {
                    a.estimated_cost
                        .partial_cmp(&b.estimated_cost)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.id.cmp(&b.id))
        });

        let chosen = eligible_candidates[0];
        let explanation = format!(
            "Routed task '{}' to worker '{}' (capabilities: {:?}, load: {}/{}, cost: ${:.2})",
            task.task_id,
            chosen.id,
            chosen.probed_capabilities,
            chosen.current_load,
            chosen.max_load,
            chosen.estimated_cost
        );

        Ok(RoutingDecision {
            task_id: task.task_id.clone(),
            selected_worker: Some(chosen.id.clone()),
            explanation,
            candidate_evaluations: evaluations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_requiring_exclusive_capability_cannot_route_to_idle_workers() {
        // Worker 1 has capability "acp"
        let mut caps1 = BTreeSet::new();
        caps1.insert("stream".to_string());
        caps1.insert("acp".to_string());
        let worker1 = WorkerCandidate {
            id: "worker-specialist".to_string(),
            probed_capabilities: caps1,
            current_load: 2,
            max_load: 5,
            estimated_cost: 0.10,
        };

        // Worker 2 and 3 are completely idle (load 0) and cheap, but lack "acp"
        let mut caps2 = BTreeSet::new();
        caps2.insert("stream".to_string());
        caps2.insert("mcp".to_string());
        let worker2 = WorkerCandidate {
            id: "worker-idle-1".to_string(),
            probed_capabilities: caps2,
            current_load: 0,
            max_load: 5,
            estimated_cost: 0.01,
        };

        let mut caps3 = BTreeSet::new();
        caps3.insert("stream".to_string());
        caps3.insert("resume".to_string());
        let worker3 = WorkerCandidate {
            id: "worker-idle-2".to_string(),
            probed_capabilities: caps3,
            current_load: 0,
            max_load: 5,
            estimated_cost: 0.00,
        };

        let candidates = vec![worker1, worker2, worker3];

        // Task requires "acp"
        let mut required = BTreeSet::new();
        required.insert("acp".to_string());
        let task = TaskRequirement {
            task_id: "acp-task-1".to_string(),
            required_capabilities: required,
            cost_ceiling: None,
        };

        let decision = CapabilityRouter::route(&task, &candidates).unwrap();

        // Must select worker-specialist despite worker-idle-1 and worker-idle-2 being 100% idle
        assert_eq!(
            decision.selected_worker.as_deref(),
            Some("worker-specialist")
        );

        // Both idle workers must be rejected with MissingCapabilities(["acp"])
        let eval_idle1 = decision
            .candidate_evaluations
            .iter()
            .find(|e| e.worker_id == "worker-idle-1")
            .unwrap();
        assert!(!eval_idle1.eligible);
        assert_eq!(
            eval_idle1.rejection,
            Some(RejectionReason::MissingCapabilities(
                vec!["acp".to_string()]
            ))
        );

        let eval_idle2 = decision
            .candidate_evaluations
            .iter()
            .find(|e| e.worker_id == "worker-idle-2")
            .unwrap();
        assert!(!eval_idle2.eligible);
        assert_eq!(
            eval_idle2.rejection,
            Some(RejectionReason::MissingCapabilities(
                vec!["acp".to_string()]
            ))
        );
    }

    #[test]
    fn saturated_specialist_refuses_routing_rather_than_routing_to_incapable_workers() {
        // Specialist is at max capacity
        let mut caps1 = BTreeSet::new();
        caps1.insert("acp".to_string());
        let worker1 = WorkerCandidate {
            id: "worker-specialist".to_string(),
            probed_capabilities: caps1,
            current_load: 5,
            max_load: 5,
            estimated_cost: 0.10,
        };

        // Idle worker lacks "acp"
        let mut caps2 = BTreeSet::new();
        caps2.insert("stream".to_string());
        let worker2 = WorkerCandidate {
            id: "worker-idle".to_string(),
            probed_capabilities: caps2,
            current_load: 0,
            max_load: 5,
            estimated_cost: 0.00,
        };

        let candidates = vec![worker1, worker2];

        let mut required = BTreeSet::new();
        required.insert("acp".to_string());
        let task = TaskRequirement {
            task_id: "acp-task-2".to_string(),
            required_capabilities: required,
            cost_ceiling: None,
        };

        let decision = CapabilityRouter::route(&task, &candidates).unwrap();

        // Must NOT route to worker-idle!
        assert_eq!(decision.selected_worker, None);
        assert!(decision.explanation.contains("Routing refused"));
    }

    #[test]
    fn cost_ceiling_filters_expensive_worker() {
        let mut caps = BTreeSet::new();
        caps.insert("stream".to_string());

        let expensive = WorkerCandidate {
            id: "expensive-worker".to_string(),
            probed_capabilities: caps.clone(),
            current_load: 0,
            max_load: 5,
            estimated_cost: 1.50,
        };

        let cheap = WorkerCandidate {
            id: "cheap-worker".to_string(),
            probed_capabilities: caps.clone(),
            current_load: 1,
            max_load: 5,
            estimated_cost: 0.25,
        };

        let candidates = vec![expensive, cheap];

        let task = TaskRequirement {
            task_id: "budget-task".to_string(),
            required_capabilities: caps,
            cost_ceiling: Some(0.50),
        };

        let decision = CapabilityRouter::route(&task, &candidates).unwrap();
        assert_eq!(decision.selected_worker.as_deref(), Some("cheap-worker"));

        let expensive_eval = decision
            .candidate_evaluations
            .iter()
            .find(|e| e.worker_id == "expensive-worker")
            .unwrap();
        assert!(!expensive_eval.eligible);
        assert_eq!(
            expensive_eval.rejection,
            Some(RejectionReason::CostCeilingExceeded {
                cost: 1.50,
                ceiling: 0.50,
            })
        );
    }
}
