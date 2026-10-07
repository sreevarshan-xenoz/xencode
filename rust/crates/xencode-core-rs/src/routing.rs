//! OR-6 — routing by what was probed. OR-11 — and saying, in words, how each
//! number used for the choice was known.
//!
//! Choose a worker only from probed capabilities (`AR-3`) plus current load and
//! cost ceiling — never a vendor name. A task requiring a capability that only one
//! agent has cannot be routed to the others even when they are completely idle.
//!
//! The second half is what makes the first half worth reading. A routing decision
//! printed as `load: 0/5, cost: $0.05` looks like reasoning and can be pure
//! invention: nothing on this machine measures what somebody else's agent is
//! doing, and xencode's price documents name models, not vendor CLIs. So every
//! number a comparison could use arrives as a [`Fact`], which carries the way it
//! was known beside it, and the decision carries the ordered [`StepNote`]s of what
//! ran, what decided, and what did not run because nothing measures it.
//!
//! Three rules follow from that, and they are the whole of this module's honesty:
//!
//! - **Only a measured number may rule a worker out.** An estimated figure can
//!   break a tie; it cannot reject anybody. A ceiling the caller set is reported
//!   as *not applied* to a worker xencode cannot price, rather than quietly
//!   passing it.
//! - **A capability nobody probed is not a capability the worker lacks.** The
//!   router still refuses to use what it cannot see — that is `AR-3`'s rule — but
//!   the reason says the measurement is missing, not that the agent is incapable.
//! - **A ranking that could not run is said out loud.** When load and cost are
//!   both unknown for every candidate, the choice was settled by name, and saying
//!   "settled by name" is the truth the printed decision owes the reader.
//!
//! A fourth check runs ahead of all three, and it is not a measurement: the
//! [`Profile`] in force may refuse to hand work to another vendor's agent at all
//! (`OR-13`). A worker ruled out that way is ruled out by the posture, in the
//! posture's own words, and it is printed as the first thing the router asked.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::profile::Profile;

/// How a number used by routing came to be known.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Watched happening on this machine. `source` names what was watched, so the
    /// reader can check it.
    Measured { source: String },
    /// Worked out from something else. `basis` says what it was worked out from
    /// and what it therefore does not cover. It may break a tie and may never
    /// rule a worker out.
    Estimated { basis: String },
    /// Nothing here measures this. `reason` says why, and the check that would
    /// have used it reports that it did not run.
    Unknown { reason: String },
}

impl Provenance {
    /// The one word a reader scans for: is this a measurement, an estimate, or
    /// nothing at all.
    pub fn label(&self) -> &'static str {
        match self {
            Provenance::Measured { .. } => "measured",
            Provenance::Estimated { .. } => "estimated",
            Provenance::Unknown { .. } => "not measured",
        }
    }

    /// The clause that follows the word, naming where the number came from.
    pub fn detail(&self) -> Option<&str> {
        match self {
            Provenance::Measured { source } => Some(source),
            Provenance::Estimated { basis } => Some(basis),
            Provenance::Unknown { reason } => Some(reason),
        }
    }
}

/// One number, with the way it was known attached to it. A fact that was never
/// measured holds no number at all, so nothing can print it as though it did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fact<T> {
    Measured { value: T, source: String },
    Estimated { value: T, basis: String },
    Unknown { reason: String },
}

impl<T: Copy + PartialOrd> Fact<T> {
    pub fn measured(value: T, source: impl Into<String>) -> Self {
        Fact::Measured {
            value,
            source: source.into(),
        }
    }

    pub fn estimated(value: T, basis: impl Into<String>) -> Self {
        Fact::Estimated {
            value,
            basis: basis.into(),
        }
    }

    pub fn unknown(reason: impl Into<String>) -> Self {
        Fact::Unknown {
            reason: reason.into(),
        }
    }

    /// The number, when there is one. `None` means nothing measures this.
    pub fn value(&self) -> Option<T> {
        match self {
            Fact::Measured { value, .. } => Some(*value),
            Fact::Estimated { value, .. } => Some(*value),
            Fact::Unknown { .. } => None,
        }
    }

    /// Whether this fact may take part in a comparison that rejects a worker.
    /// Only an actual observation qualifies; an estimate that threw a worker out
    /// would be a decision made by a guess wearing a number.
    pub fn may_reject(&self) -> bool {
        matches!(self, Fact::Measured { .. })
    }

    pub fn provenance(&self) -> Provenance {
        match self {
            Fact::Measured { source, .. } => Provenance::Measured {
                source: source.clone(),
            },
            Fact::Estimated { basis, .. } => Provenance::Estimated {
                basis: basis.clone(),
            },
            Fact::Unknown { reason } => Provenance::Unknown {
                reason: reason.clone(),
            },
        }
    }

    /// The number and its provenance in one line, in the order a reader wants
    /// them: what it is, then how it was known.
    pub fn words(&self, render: impl Fn(&T) -> String) -> String {
        match self {
            Fact::Measured { value, source } => {
                format!("{} — measured ({})", render(value), source)
            }
            Fact::Estimated { value, basis } => {
                format!("{} — estimated ({})", render(value), basis)
            }
            Fact::Unknown { reason } => format!("not measured ({reason})"),
        }
    }
}

/// A candidate worker evaluated by probed capabilities, load and cost — each of
/// the last three carrying how it was known.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerCandidate {
    /// Worker identifier (worker handle, token, or session key).
    pub id: String,
    /// Capabilities confirmed by probe (never inferred from name or unverified docs).
    pub probed_capabilities: BTreeSet<String>,
    /// Whether this worker could be probed here at all. `false` means the set
    /// above is empty because nothing was observed — which is a different answer
    /// from observing that the worker cannot do something, and is printed as one.
    pub probed: bool,
    /// Whether the agent roster says this name belongs to another vendor's
    /// program (`OR-13`). Supplied by the caller because this crate has no view
    /// of which binaries are installed here; `false` is the roster having no row
    /// to read, which is not a claim that the worker is xencode's own.
    pub external: bool,
    /// One line per capability the probe spoke to, saying what was read and what
    /// was found in it. Kept beside the candidate so a reason can be checked
    /// rather than taken on trust.
    pub capability_evidence: BTreeMap<String, String>,
    /// Tasks running on this worker right now.
    pub load: Fact<usize>,
    /// How many tasks this worker takes at once before it is saturated.
    pub capacity: Fact<usize>,
    /// What one task on this worker costs, in dollars.
    pub cost: Fact<f64>,
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
    /// The posture the task is being routed under (`OR-13`). A worker the profile
    /// refuses is refused before its capabilities, load or price are looked at, so
    /// a policy decision is never printed as though it were a measurement.
    pub profile: Profile,
}

/// Why a worker candidate was rejected during routing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RejectionReason {
    /// The probe of this worker did not record one or more required
    /// capabilities. `probed` separates the two very different ways that happens:
    /// looked and not found, versus nothing to look at because this machine has
    /// no such worker.
    MissingCapabilities {
        required: Vec<String>,
        probed: bool,
        evidence: Vec<String>,
    },
    /// The worker is at or over capacity. Only ever produced from measured
    /// numbers, so `source` says where both counts came from.
    LoadExceeded {
        current: usize,
        max: usize,
        source: String,
    },
    /// The worker's measured cost exceeds the ceiling. Only ever produced from a
    /// measured cost.
    CostCeilingExceeded {
        cost: f64,
        ceiling: f64,
        source: String,
    },
    /// The posture in force refuses to hand work to another vendor's agent
    /// (`OR-13`). This is not a measurement and nothing about the worker was
    /// checked to produce it: the refusal quotes the profile it was refused by,
    /// and the rule line the surface prints above the refusals names the setting
    /// that opens it — a reader who disagrees with a policy should be arguing with
    /// the policy, not hunting for a failed check.
    ExternalWorkerRefused { refusal: String },
}

impl RejectionReason {
    /// The reason in one plain sentence, with the evidence that supports it.
    pub fn words(&self) -> String {
        match self {
            RejectionReason::MissingCapabilities {
                required,
                probed,
                evidence,
            } => {
                let needs = required.join(", ");
                if !probed {
                    return format!(
                        "needs {needs}, which xencode never probed on this machine — the word \
                         for what an uninstalled worker can do is unknown, not absent"
                    );
                }
                let mut words = format!("needs {needs}, which no probe of this worker confirmed");
                if !evidence.is_empty() {
                    words.push_str(&format!(" — {}", evidence.join("; ")));
                }
                words
            }
            RejectionReason::LoadExceeded {
                current,
                max,
                source,
            } => format!(
                "is carrying {current} tasks at its limit of {max} ({source}), so launching \
                 another here would queue behind its own work"
            ),
            RejectionReason::CostCeilingExceeded {
                cost,
                ceiling,
                source,
            } => format!(
                "costs {cost:.2}$ against a ceiling of {ceiling:.2}$ ({source}), so it is out \
                 on price"
            ),
            RejectionReason::ExternalWorkerRefused { refusal } => refusal.clone(),
        }
    }
}

/// Evaluation record for a single worker candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub worker_id: String,
    pub eligible: bool,
    pub rejection: Option<RejectionReason>,
    pub probed_capabilities: Vec<String>,
    /// The facts this candidate was judged on, each with the way it was known:
    /// the evidence behind every required capability, then load, capacity and
    /// cost as they stood. Printed so a reader can check the reason instead of
    /// trusting it.
    pub facts: Vec<String>,
    /// Checks that could not be run against this candidate, and why. A ceiling
    /// that could not be applied belongs here rather than in a pass.
    pub not_checked: Vec<String>,
}

/// One step of the decision: a check or a comparison, whether it ran, whether it
/// separated anything, and what to tell a person about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepNote {
    pub check: String,
    pub ran: bool,
    pub decided: bool,
    pub words: String,
}

/// The result of routing a task across worker candidates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub task_id: String,
    pub selected_worker: Option<String>,
    /// The whole decision in one sentence, naming what actually settled it —
    /// including "settled by name" when nothing else could.
    pub explanation: String,
    /// Every check and comparison in the order the router applied them.
    pub steps: Vec<StepNote>,
    pub candidate_evaluations: Vec<CandidateEvaluation>,
}

/// `1 worker`, `3 workers`. The counts in these sentences are small, and
/// `worker(s)` reads like a form rather than a reason.
fn count(n: usize, singular: &str, plural: &str) -> String {
    if n == 1 {
        format!("1 {singular}")
    } else {
        format!("{n} {plural}")
    }
}

fn was_were(n: usize) -> &'static str {
    if n == 1 {
        "was"
    } else {
        "were"
    }
}

/// Capability-gated router choosing workers from probed capabilities, load and
/// cost ceiling, and reporting how each of those was known.
pub struct CapabilityRouter;

impl CapabilityRouter {
    /// Evaluates candidate workers and chooses the best eligible worker for `task`.
    ///
    /// Rules:
    /// 0. The posture comes first: a worker the [`Profile`] refuses to hand work
    ///    to is out, by name, with the profile's own sentence as the reason.
    /// 1. Must satisfy all `required_capabilities` from `probed_capabilities`. An
    ///    idle worker lacking any required capability is strictly rejected; a
    ///    worker that was never probed is rejected too, and told apart in words.
    /// 2. Must not exceed its capacity — only when both counts are measured.
    /// 3. Must not exceed `cost_ceiling` — only when the cost is measured. A
    ///    ceiling that cannot be applied to a candidate is reported as such.
    /// 4. Ranking: lowest load, then lowest cost, then name. A comparison only
    ///    runs when every remaining candidate has a number for it, and the first
    ///    one that separates the field is named as what decided the choice.
    pub fn route(
        task: &TaskRequirement,
        candidates: &[WorkerCandidate],
    ) -> Result<RoutingDecision, String> {
        if candidates.is_empty() {
            return Err("no worker candidates provided for routing".to_string());
        }

        let required: Vec<String> = task.required_capabilities.iter().cloned().collect();
        let mut evaluations = Vec::new();
        let mut eligible: Vec<&WorkerCandidate> = Vec::new();
        let mut load_rejections = 0usize;
        let mut cost_rejections = 0usize;
        let mut profile_rejections = 0usize;
        let mut ceiling_skipped: Vec<String> = Vec::new();
        let mut load_checks_run = 0usize;

        for candidate in candidates {
            let mut facts = Vec::new();
            let mut not_checked = Vec::new();

            // 0. The posture, before anything is measured. A worker the roster
            //    says belongs to another vendor is not eligible while the worker
            //    rule is closed, and saying so first keeps the capability, load
            //    and cost lines from reading as the reason.
            if let Err(refusal) = task.profile.check_worker(&candidate.id, candidate.external) {
                facts.push(format!("profile: {}", task.profile.name()));
                evaluations.push(CandidateEvaluation {
                    worker_id: candidate.id.clone(),
                    eligible: false,
                    rejection: Some(RejectionReason::ExternalWorkerRefused {
                        refusal: refusal.why,
                    }),
                    probed_capabilities: candidate.probed_capabilities.iter().cloned().collect(),
                    facts,
                    not_checked,
                });
                profile_rejections += 1;
                continue;
            }

            let missing: Vec<String> = required
                .iter()
                .filter(|need| !candidate.probed_capabilities.contains(*need))
                .cloned()
                .collect();
            if !missing.is_empty() {
                let evidence: Vec<String> = missing
                    .iter()
                    .filter_map(|need| {
                        candidate
                            .capability_evidence
                            .get(need)
                            .map(|line| format!("{need}: {line}"))
                    })
                    .collect();
                for need in &missing {
                    if let Some(line) = candidate.capability_evidence.get(need) {
                        facts.push(format!("{need}: {line}"));
                    }
                }
                evaluations.push(CandidateEvaluation {
                    worker_id: candidate.id.clone(),
                    eligible: false,
                    rejection: Some(RejectionReason::MissingCapabilities {
                        required: missing,
                        probed: candidate.probed,
                        evidence,
                    }),
                    probed_capabilities: candidate.probed_capabilities.iter().cloned().collect(),
                    facts,
                    not_checked,
                });
                continue;
            }
            for need in &required {
                if let Some(line) = candidate.capability_evidence.get(need) {
                    facts.push(format!("{need}: {line}"));
                }
            }
            // With nothing required, what a worker was probed as able to do is
            // still a fact the choice rests on — it is what the name was chosen
            // *from*. The evidence line travels with it, so the line is checkable.
            if required.is_empty() {
                for (cap, line) in &candidate.capability_evidence {
                    facts.push(format!("{cap}: {line}"));
                }
            }

            // 2. Load against capacity. A worker is only ever ruled out on
            // numbers that were actually counted.
            match (candidate.load.may_reject(), candidate.capacity.value()) {
                (true, Some(max)) => {
                    let current = candidate.load.value().unwrap_or_default();
                    facts.push(format!(
                        "load: {}",
                        candidate.load.words(|v| format!("{v} task(s) running"))
                    ));
                    facts.push(format!(
                        "capacity: {}",
                        candidate.capacity.words(|v| format!("{v} at once"))
                    ));
                    if current >= max {
                        evaluations.push(CandidateEvaluation {
                            worker_id: candidate.id.clone(),
                            eligible: false,
                            rejection: Some(RejectionReason::LoadExceeded {
                                current,
                                max,
                                source: candidate
                                    .load
                                    .provenance()
                                    .detail()
                                    .unwrap_or_default()
                                    .to_string(),
                            }),
                            probed_capabilities: candidate
                                .probed_capabilities
                                .iter()
                                .cloned()
                                .collect(),
                            facts,
                            not_checked,
                        });
                        load_rejections += 1;
                        continue;
                    }
                    load_checks_run += 1;
                }
                _ => {
                    facts.push(format!(
                        "load: {}",
                        candidate.load.words(|v| format!("{v} task(s) running"))
                    ));
                    facts.push(format!(
                        "capacity: {}",
                        candidate.capacity.words(|v| format!("{v} at once"))
                    ));
                    not_checked.push(
                        "whether it is already full — xencode has no counted task load or \
                         documented capacity for this worker, so nobody was ruled out on it"
                            .to_string(),
                    );
                }
            }

            // 3. Cost ceiling, applied only to a cost that was measured.
            if let Some(ceiling) = task.cost_ceiling {
                if candidate.cost.may_reject() {
                    let cost = candidate.cost.value().unwrap_or_default();
                    facts.push(format!(
                        "cost: {}",
                        candidate.cost.words(|v| format!("{v:.2}$ per task"))
                    ));
                    if cost > ceiling {
                        evaluations.push(CandidateEvaluation {
                            worker_id: candidate.id.clone(),
                            eligible: false,
                            rejection: Some(RejectionReason::CostCeilingExceeded {
                                cost,
                                ceiling,
                                source: candidate
                                    .cost
                                    .provenance()
                                    .detail()
                                    .unwrap_or_default()
                                    .to_string(),
                            }),
                            probed_capabilities: candidate
                                .probed_capabilities
                                .iter()
                                .cloned()
                                .collect(),
                            facts,
                            not_checked,
                        });
                        cost_rejections += 1;
                        continue;
                    }
                } else {
                    facts.push(format!(
                        "cost: {}",
                        candidate.cost.words(|v| format!("{v:.2}$ per task"))
                    ));
                    not_checked.push(format!(
                        "your ceiling of {:.2}$ — the figure xencode has for this worker is not \
                         a measurement, so it was not used to throw the worker out",
                        ceiling
                    ));
                    ceiling_skipped.push(candidate.id.clone());
                }
            } else {
                facts.push(format!(
                    "cost: {}",
                    candidate.cost.words(|v| format!("{v:.2}$ per task"))
                ));
            }

            evaluations.push(CandidateEvaluation {
                worker_id: candidate.id.clone(),
                eligible: true,
                rejection: None,
                probed_capabilities: candidate.probed_capabilities.iter().cloned().collect(),
                facts,
                not_checked,
            });
            eligible.push(candidate);
        }

        let mut steps = Vec::new();
        steps.push(StepNote {
            check: "profile".to_string(),
            ran: true,
            decided: profile_rejections > 0,
            words: if profile_rejections > 0 {
                let refused: Vec<String> = evaluations
                    .iter()
                    .filter(|e| {
                        matches!(
                            e.rejection,
                            Some(RejectionReason::ExternalWorkerRefused { .. })
                        )
                    })
                    .map(|e| e.worker_id.clone())
                    .collect();
                format!(
                    "{} of {} {} refused by the {} profile before the capability, load and cost \
                     checks were consulted: {}",
                    count(refused.len(), "worker", "workers"),
                    candidates.len(),
                    was_were(refused.len()),
                    task.profile.name(),
                    refused.join(", ")
                )
            } else {
                format!(
                    "the {} profile let every candidate through to the measurements",
                    task.profile.name()
                )
            },
        });
        // A worker the posture refused was never measured for this task, so it
        // must not be counted as a capability refusal below.
        let capability_refusals: Vec<&CandidateEvaluation> = evaluations
            .iter()
            .filter(|e| {
                !e.eligible
                    && !matches!(
                        e.rejection,
                        Some(RejectionReason::ExternalWorkerRefused { .. })
                    )
            })
            .collect();
        steps.push(StepNote {
            check: "capabilities".to_string(),
            ran: true,
            decided: !required.is_empty() && !capability_refusals.is_empty(),
            words: if required.is_empty() {
                "no capability was required, so nothing was ruled out on it".to_string()
            } else {
                let needs = required.join(", ");
                let out: Vec<String> = capability_refusals
                    .iter()
                    .map(|e| e.worker_id.clone())
                    .collect();
                let mut words = format!(
                    "{needs} was required of every worker that reached this check, and {} of {} \
                     {} probed here as able to do it",
                    eligible.len(),
                    candidates.len() - profile_rejections,
                    was_were(eligible.len())
                );
                if !out.is_empty() {
                    words.push_str(&format!(
                        "; {} {} ruled out: {}",
                        count(out.len(), "worker", "workers"),
                        was_were(out.len()),
                        out.join(", ")
                    ));
                }
                words
            },
        });
        steps.push(StepNote {
            check: "load".to_string(),
            ran: load_checks_run > 0 || load_rejections > 0,
            decided: load_rejections > 0,
            words: if load_rejections > 0 {
                if load_rejections == 1 {
                    "one worker was at the capacity measured for it".to_string()
                } else {
                    format!("{load_rejections} workers were at the capacity measured for them")
                }
            } else {
                "no worker was ruled out on load, because nothing on this machine counts the \
                 tasks a vendor's own agent has been given"
                    .to_string()
            },
        });
        steps.push(StepNote {
            check: "cost ceiling".to_string(),
            ran: task.cost_ceiling.is_some(),
            decided: cost_rejections > 0,
            words: match task.cost_ceiling {
                None => "no ceiling was set, so price was never a question".to_string(),
                Some(ceiling) if cost_rejections > 0 => format!(
                    "{} cost more than the ceiling of {ceiling:.2}$ and {} removed on that \
                     measurement",
                    count(cost_rejections, "worker", "workers"),
                    was_were(cost_rejections)
                ),
                Some(ceiling) if !ceiling_skipped.is_empty() => format!(
                    "the ceiling of {ceiling:.2}$ was not applied to {} ({}) — xencode has no \
                     measured price for a vendor's agent, and an unmeasured number may not \
                     reject a worker",
                    count(ceiling_skipped.len(), "worker", "workers"),
                    ceiling_skipped.join(", ")
                ),
                Some(_) => {
                    "no candidate could be priced, so the ceiling ruled nobody out".to_string()
                }
            },
        });

        if eligible.is_empty() {
            // One sentence per reason, with every name that drew it. Listing the
            // candidates separately is right when the reasons differ and wrong
            // when they do not: the shipped posture refuses every agent on the
            // roster for one sentence's worth of reason, and printing it once per
            // name buries the answer under it (`OR-13`). A posture refusal is
            // grouped on its own, in the rule's words, because the sentence it
            // carries names the worker it refused.
            let mut posture_refused: Vec<String> = Vec::new();
            let mut groups: Vec<(String, Vec<String>)> = Vec::new();
            for evaluation in &evaluations {
                let Some(reason) = evaluation.rejection.as_ref() else {
                    continue;
                };
                if matches!(reason, RejectionReason::ExternalWorkerRefused { .. }) {
                    posture_refused.push(evaluation.worker_id.clone());
                    continue;
                }
                let words = reason.words();
                match groups.iter_mut().find(|(w, _)| *w == words) {
                    Some((_, names)) => names.push(evaluation.worker_id.clone()),
                    None => groups.push((words, vec![evaluation.worker_id.clone()])),
                }
            }
            let mut refused: Vec<String> = Vec::new();
            if !posture_refused.is_empty() {
                refused.push(format!(
                    "{}: refused by the {} posture, which hands work only to xencode's own \
                     loop — xencode has a roster row for {}, so the work would leave this \
                     machine through a program xencode does not control",
                    posture_refused.join(", "),
                    task.profile.name(),
                    if posture_refused.len() == 1 {
                        "that name"
                    } else {
                        "those names"
                    },
                ));
            }
            refused.extend(
                groups
                    .iter()
                    .map(|(words, names)| format!("{}: {}", names.join(", "), words)),
            );
            let explanation = format!(
                "Nothing was routed for task '{}': none of the {} candidates could take it. \
                 {}.",
                task.task_id,
                candidates.len(),
                refused.join(" ")
            );
            steps.push(StepNote {
                check: "choice".to_string(),
                ran: false,
                decided: false,
                words: "there was nothing left to choose between".to_string(),
            });
            return Ok(RoutingDecision {
                task_id: task.task_id.clone(),
                selected_worker: None,
                explanation,
                steps,
                candidate_evaluations: evaluations,
            });
        }

        // 4. Ranking. A comparison runs only when every remaining candidate has a
        // number for it — ordering the ones that were counted against the ones
        // that were not is ordering against nothing. The name is always the last
        // key, so even an unrankable field comes out in an order that is stated.
        let mut ordered: Vec<&WorkerCandidate> = eligible.clone();
        let load_comparable = ordered
            .iter()
            .all(|candidate| candidate.load.value().is_some());
        let cost_comparable = ordered
            .iter()
            .all(|candidate| candidate.cost.value().is_some());
        ordered.sort_by(|a, b| {
            let by_load = if load_comparable {
                a.load
                    .value()
                    .partial_cmp(&b.load.value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                std::cmp::Ordering::Equal
            };
            let by_cost = if cost_comparable {
                a.cost
                    .value()
                    .partial_cmp(&b.cost.value())
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                std::cmp::Ordering::Equal
            };
            by_load.then(by_cost).then_with(|| a.id.cmp(&b.id))
        });
        let chosen = ordered[0];
        let load_decided = load_comparable
            && ordered.len() > 1
            && ordered[1].load.value() != ordered[0].load.value();
        let cost_decided = !load_decided
            && cost_comparable
            && ordered.len() > 1
            && ordered[1].cost.value() != ordered[0].cost.value();
        let name_decided = !load_decided && !cost_decided && ordered.len() > 1;

        let load_unknown_count = ordered.iter().filter(|c| c.load.value().is_none()).count();
        let estimate_count = ordered
            .iter()
            .filter(|c| matches!(c.cost, Fact::Estimated { .. }))
            .count();

        steps.push(StepNote {
            check: "load ranking".to_string(),
            ran: load_comparable,
            decided: load_decided,
            words: if load_comparable {
                format!(
                    "the {} {} ordered by how many tasks each was carrying; the winner's count \
                     is {}",
                    count(ordered.len(), "candidate", "candidates"),
                    was_were(ordered.len()),
                    chosen.load.provenance().label()
                )
            } else if load_unknown_count == ordered.len() {
                "no candidate left has a counted load, so there was nothing to order here"
                    .to_string()
            } else {
                format!(
                    "{} of the {} candidates {} no counted load, and ranking the ones that were \
                     counted against the ones that were not would rank the uncounted ones out of \
                     existence",
                    load_unknown_count,
                    ordered.len(),
                    if load_unknown_count == 1 {
                        "has"
                    } else {
                        "have"
                    }
                )
            },
        });
        steps.push(StepNote {
            check: "cost ranking".to_string(),
            ran: !load_decided && cost_comparable,
            decided: cost_decided,
            words: if load_decided {
                "the load counts had already separated the field, so price never came up"
                    .to_string()
            } else if cost_comparable {
                let ordered_word = count(ordered.len(), "candidate", "candidates");
                let were = was_were(ordered.len());
                if estimate_count > 0 {
                    format!(
                        "the {ordered_word} {were} ordered by price, with {} of {} figures being \
                         estimates rather than measurements — enough to move the order, never \
                         enough to remove anybody",
                        estimate_count,
                        ordered.len()
                    )
                } else {
                    format!(
                        "the {ordered_word} {were} ordered by price, every figure measured ({})",
                        chosen
                            .cost
                            .provenance()
                            .detail()
                            .unwrap_or("source unnamed")
                    )
                }
            } else {
                "xencode has no price for every candidate left, and comparing some of them on \
                 cost would leave the rest unranked rather than cheap"
                    .to_string()
            },
        });
        steps.push(StepNote {
            check: "name".to_string(),
            ran: ordered.len() > 1,
            decided: name_decided,
            words: if name_decided {
                format!(
                    "{} of {} candidates were eligible and nothing measurable separated them, \
                     so the choice fell on the first name in order — a convention, not a \
                     finding about {}",
                    ordered.len(),
                    candidates.len(),
                    chosen.id
                )
            } else if ordered.len() == 1 {
                "only one worker was eligible, so there was nothing to order".to_string()
            } else {
                "a counted difference decided the order first, so the names never mattered"
                    .to_string()
            },
        });

        let settled_by = if load_decided {
            "how many tasks each worker was carrying"
        } else if cost_decided {
            "the price of a task on each"
        } else if name_decided {
            "the order of their names, because nothing measurable separated them"
        } else {
            "there being exactly one worker left that met the requirements"
        };
        let explanation = format!(
            "Routed task '{}' to '{}'. It was settled by {}. Required: {}. Probed here as able \
             to do all of it: {} of {} candidates.",
            task.task_id,
            chosen.id,
            settled_by,
            if required.is_empty() {
                "nothing".to_string()
            } else {
                required.join(", ")
            },
            eligible.len(),
            candidates.len(),
        );

        Ok(RoutingDecision {
            task_id: task.task_id.clone(),
            selected_worker: Some(chosen.id.clone()),
            explanation,
            steps,
            candidate_evaluations: evaluations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(list: &[&str]) -> BTreeSet<String> {
        list.iter().map(|c| c.to_string()).collect()
    }

    /// A candidate whose load, capacity and cost are all things nobody measured —
    /// which is every vendor worker on this machine today. `external` is `false`
    /// because these fixtures name invented workers; a test that wants the
    /// roster's answer calls [`external`], so the posture is never what a
    /// capability test is quietly watching.
    fn unmeasured(id: &str, capabilities: &[&str]) -> WorkerCandidate {
        WorkerCandidate {
            id: id.to_string(),
            probed_capabilities: caps(capabilities),
            probed: true,
            external: false,
            capability_evidence: capabilities
                .iter()
                .map(|c| {
                    (
                        c.to_string(),
                        format!("confirmed by `{id} --help` read just now"),
                    )
                })
                .collect(),
            load: Fact::unknown("xencode cannot see work another process gave this worker"),
            capacity: Fact::unknown("nothing in the roster says how many tasks it tolerates"),
            cost: Fact::unknown("xencode's price documents name models, not agents"),
        }
    }

    /// The same candidate with the roster's answer on it: somebody else's agent.
    fn external(mut candidate: WorkerCandidate) -> WorkerCandidate {
        candidate.external = true;
        candidate
    }

    fn task(needs: &[&str], ceiling: Option<f64>) -> TaskRequirement {
        TaskRequirement {
            task_id: "t-1".to_string(),
            required_capabilities: caps(needs),
            cost_ceiling: ceiling,
            profile: Profile::LOCAL_ONLY,
        }
    }

    /// The same task under a posture that lets work reach other vendors.
    fn open_task(needs: &[&str], ceiling: Option<f64>) -> TaskRequirement {
        TaskRequirement {
            profile: Profile::new(true, true),
            ..task(needs, ceiling)
        }
    }

    #[test]
    fn a_capability_no_probe_recorded_rules_a_worker_out_and_says_which_probe() {
        let decision = CapabilityRouter::route(
            &task(&["stream"], None),
            &[
                unmeasured("can-stream", &["stream"]),
                unmeasured("cannot-stream", &["acp"]),
            ],
        )
        .unwrap();

        assert_eq!(decision.selected_worker.as_deref(), Some("can-stream"));
        let out = decision
            .candidate_evaluations
            .iter()
            .find(|e| e.worker_id == "cannot-stream")
            .unwrap();
        assert!(!out.eligible);
        assert_eq!(
            out.rejection,
            Some(RejectionReason::MissingCapabilities {
                required: vec!["stream".to_string()],
                probed: true,
                evidence: vec![],
            })
        );
        // The eligible worker's line names the probe behind the capability, so
        // the reader can check the claim instead of trusting it.
        assert!(decision
            .candidate_evaluations
            .iter()
            .find(|e| e.worker_id == "can-stream")
            .unwrap()
            .facts
            .iter()
            .any(|f| f.contains("stream: confirmed by `can-stream --help` read just now")));
    }

    #[test]
    fn an_unprobed_worker_is_refused_for_a_missing_measurement_not_a_missing_ability() {
        let mut never = unmeasured("not-here", &[]);
        never.probed = false;
        let decision = CapabilityRouter::route(
            &task(&["stream"], None),
            &[unmeasured("here", &["stream"]), never],
        )
        .unwrap();

        let rejected = decision
            .candidate_evaluations
            .iter()
            .find(|e| e.worker_id == "not-here")
            .unwrap();
        let words = rejected.rejection.as_ref().unwrap().words();
        assert!(
            words.contains("never probed on this machine"),
            "the reason must say the measurement is missing: {words}"
        );
        assert!(!words.contains("cannot"), "{words}");
        // And it is still refused: `AR-3`'s rule is that an unprobed capability
        // cannot be used, whichever reason stands behind it.
        assert!(!rejected.eligible);
    }

    #[test]
    fn a_ceiling_xencode_cannot_apply_is_reported_as_not_applied() {
        let decision = CapabilityRouter::route(
            &task(&["stream"], Some(0.50)),
            &[unmeasured("cheap-unknown", &["stream"])],
        )
        .unwrap();

        assert_eq!(decision.selected_worker.as_deref(), Some("cheap-unknown"));
        let evaluated = &decision.candidate_evaluations[0];
        assert!(evaluated
            .not_checked
            .iter()
            .any(|n| n.contains("your ceiling of 0.50$")));
        let ceiling_step = decision
            .steps
            .iter()
            .find(|s| s.check == "cost ceiling")
            .unwrap();
        assert!(!ceiling_step.decided);
        assert!(
            ceiling_step.words.contains("was not applied"),
            "{}",
            ceiling_step.words
        );
    }

    #[test]
    fn a_measured_price_can_reject_a_worker_but_an_estimated_one_cannot() {
        let mut priced = unmeasured("priced", &["stream"]);
        priced.cost = Fact::measured(2.50, "the vendor's published rate for this account");
        let mut guess = unmeasured("guess", &["stream"]);
        guess.cost = Fact::estimated(
            3.00,
            "what a minute of this machine's electricity costs at four times the load",
        );

        let with_ceiling = CapabilityRouter::route(
            &task(&["stream"], Some(2.00)),
            &[priced.clone(), guess.clone()],
        )
        .unwrap();
        assert_eq!(
            with_ceiling.selected_worker.as_deref(),
            Some("guess"),
            "the measured price is thrown out; the estimate is not"
        );
        let rejected = with_ceiling
            .candidate_evaluations
            .iter()
            .find(|e| e.worker_id == "priced")
            .unwrap();
        assert!(matches!(
            rejected.rejection,
            Some(RejectionReason::CostCeilingExceeded { .. })
        ));
        let kept = with_ceiling
            .candidate_evaluations
            .iter()
            .find(|e| e.worker_id == "guess")
            .unwrap();
        assert!(
            kept.not_checked
                .iter()
                .any(|n| n.contains("your ceiling of 2.00$")),
            "a ceiling that could not be applied has to be said: {:?}",
            kept.not_checked
        );

        // With no ceiling both remain, and price orders them — with the estimate
        // labelled as one rather than presented as a measurement.
        let ranked = CapabilityRouter::route(&task(&["stream"], None), &[guess, priced]).unwrap();
        assert_eq!(ranked.selected_worker.as_deref(), Some("priced"));
        let cost_step = ranked
            .steps
            .iter()
            .find(|s| s.check == "cost ranking")
            .unwrap();
        assert!(cost_step.decided);
        assert!(cost_step.words.contains("estimates"), "{}", cost_step.words);
    }

    #[test]
    fn a_saturated_worker_is_only_ever_ruled_out_on_counted_numbers() {
        let mut full = unmeasured("full", &["stream"]);
        full.load = Fact::measured(5, "xencode's own lease registry for this project");
        full.capacity = Fact::measured(5, "the same registry");
        let mut idle = unmeasured("idle", &["stream"]);
        idle.load = Fact::measured(1, "xencode's own lease registry for this project");
        idle.capacity = Fact::measured(5, "the same registry");

        let decision = CapabilityRouter::route(&task(&["stream"], None), &[full, idle]).unwrap();
        assert_eq!(decision.selected_worker.as_deref(), Some("idle"));
        let load_step = decision.steps.iter().find(|s| s.check == "load").unwrap();
        assert!(load_step.decided);

        // The same pair with nobody counted: nothing is ruled out on load.
        let uncounted = CapabilityRouter::route(
            &task(&["stream"], None),
            &[
                unmeasured("full", &["stream"]),
                unmeasured("idle", &["stream"]),
            ],
        )
        .unwrap();
        let load_step = uncounted.steps.iter().find(|s| s.check == "load").unwrap();
        assert!(!load_step.decided);
        assert!(load_step.words.contains("nothing on this machine counts"));
    }

    #[test]
    fn a_load_counted_for_every_worker_does_decide_the_order() {
        let mut busy = unmeasured("aaa-busy", &["stream"]);
        busy.load = Fact::measured(4, "this session's task list");
        let mut quiet = unmeasured("zzz-quiet", &["stream"]);
        quiet.load = Fact::measured(1, "this session's task list");

        let decision = CapabilityRouter::route(&task(&["stream"], None), &[busy, quiet]).unwrap();
        assert_eq!(decision.selected_worker.as_deref(), Some("zzz-quiet"));
        let load_rank = decision
            .steps
            .iter()
            .find(|s| s.check == "load ranking")
            .unwrap();
        assert!(load_rank.ran && load_rank.decided);
        assert!(decision
            .explanation
            .contains("settled by how many tasks each worker was carrying"));
        let name_step = decision.steps.iter().find(|s| s.check == "name").unwrap();
        assert!(!name_step.decided);
    }

    #[test]
    fn when_nothing_separates_the_candidates_the_decision_says_the_names_decided() {
        let decision = CapabilityRouter::route(
            &task(&["stream"], None),
            &[
                unmeasured("zeta", &["stream"]),
                unmeasured("alpha", &["stream"]),
            ],
        )
        .unwrap();

        assert_eq!(decision.selected_worker.as_deref(), Some("alpha"));
        assert!(
            decision
                .explanation
                .contains("settled by the order of their names"),
            "{}",
            decision.explanation
        );
        let name_step = decision.steps.iter().find(|s| s.check == "name").unwrap();
        assert!(name_step.decided);
        assert!(name_step.words.contains("a convention, not a finding"));
        // The load line says plainly that nobody was counted, rather than the
        // mixed-case sentence about comparing counted workers to uncounted ones.
        let load_rank = decision
            .steps
            .iter()
            .find(|s| s.check == "load ranking")
            .unwrap();
        assert!(
            load_rank
                .words
                .contains("no candidate left has a counted load"),
            "{}",
            load_rank.words
        );
        // No debug-shaped output anywhere in what a reader sees.
        for step in &decision.steps {
            assert!(!step.words.contains('{'), "{}", step.words);
        }
    }

    #[test]
    fn a_load_counted_for_one_worker_alone_does_not_rank_anybody() {
        let mut busy = unmeasured("aaa-busy", &["stream"]);
        busy.load = Fact::measured(4, "this session's task list");
        let quiet = unmeasured("zzz-quiet", &["stream"]);

        let decision = CapabilityRouter::route(&task(&["stream"], None), &[busy, quiet]).unwrap();
        // The counted worker must not win for being counted, and the uncounted one
        // must not lose for being uncounted: the comparison cannot run at all, so
        // the order falls to the name and says that out loud.
        assert_eq!(decision.selected_worker.as_deref(), Some("aaa-busy"));
        let load_rank = decision
            .steps
            .iter()
            .find(|s| s.check == "load ranking")
            .unwrap();
        assert!(!load_rank.ran);
        assert!(
            load_rank
                .words
                .contains("1 of the 2 candidates has no counted load"),
            "{}",
            load_rank.words
        );
        assert!(decision.explanation.contains("order of their names"));
    }

    #[test]
    fn one_eligible_worker_is_not_described_as_a_ranking() {
        let decision = CapabilityRouter::route(
            &task(&["acp"], None),
            &[
                unmeasured("only-one", &["acp"]),
                unmeasured("other", &["stream"]),
            ],
        )
        .unwrap();
        assert_eq!(decision.selected_worker.as_deref(), Some("only-one"));
        let name_step = decision.steps.iter().find(|s| s.check == "name").unwrap();
        assert!(!name_step.decided);
        assert!(name_step.words.contains("only one worker was eligible"));
    }

    #[test]
    fn a_task_nothing_can_serve_names_every_refusal_instead_of_one_generic_no() {
        let decision = CapabilityRouter::route(
            &task(&["telepathy"], None),
            &[unmeasured("a", &["stream"]), unmeasured("b", &["acp"])],
        )
        .unwrap();
        assert_eq!(decision.selected_worker, None);
        assert!(decision.explanation.contains("Nothing was routed"));
        for worker in ["a", "b"] {
            assert!(
                decision.explanation.contains(worker),
                "every refusal should be quotable: {}",
                decision.explanation
            );
        }
        assert!(decision.explanation.contains("telepathy"));
        // The two drew one reason, so the reason is stated once and both names
        // stand under it. Repeating a sentence per candidate is not a longer
        // explanation, and under the shipped posture every candidate shares one.
        assert!(
            decision.explanation.contains("a, b: needs telepathy"),
            "{}",
            decision.explanation
        );
        assert_eq!(
            decision.explanation.matches("telepathy").count(),
            1,
            "{}",
            decision.explanation
        );
    }

    /// Names that drew different reasons never merge into one line: the grouping
    /// is by reason, so a reader is never told two workers were refused for the
    /// same thing when they were not.
    #[test]
    fn candidates_refused_for_different_reasons_are_not_merged_into_one_line() {
        let mut priced = unmeasured("refused-on-price", &["telepathy"]);
        priced.cost = Fact::measured(0.02, "the rate card this project keeps");
        let decision = CapabilityRouter::route(
            &task(&["telepathy"], Some(0.01)),
            &[unmeasured("refused-on-capability", &["stream"]), priced],
        )
        .unwrap();
        assert_eq!(decision.selected_worker, None);
        assert!(
            decision
                .explanation
                .contains("refused-on-capability: needs telepathy"),
            "{}",
            decision.explanation
        );
        assert!(
            decision
                .explanation
                .contains("refused-on-price: costs 0.02$ against a ceiling of 0.01$"),
            "{}",
            decision.explanation
        );
    }

    /// What the shipped posture actually produces on a machine with the roster
    /// installed: every candidate refused by one rule. The answer is one sentence
    /// with the names under it, not that sentence five times.
    #[test]
    fn one_posture_refusing_every_candidate_is_answered_with_one_sentence() {
        let ids = ["opencode", "cline", "codex", "claude", "gemini"];
        let candidates: Vec<WorkerCandidate> = ids
            .iter()
            .map(|id| external(unmeasured(id, &["stream"])))
            .collect();
        let decision = CapabilityRouter::route(&task(&[], None), &candidates).unwrap();
        assert!(decision.selected_worker.is_none());
        assert_eq!(
            decision
                .explanation
                .matches("refused by the Local Only posture")
                .count(),
            1,
            "the reason is said once, with every name under it: {}",
            decision.explanation
        );
        for id in ids {
            assert!(
                decision.explanation.contains(id),
                "{id} is named: {}",
                decision.explanation
            );
        }
    }

    /// An estimate is labelled as a guess and a measurement is credited to where
    /// it was seen, in the same line, because a reader comparing two numbers has
    /// to be able to tell which kind each one is.
    #[test]
    fn an_estimate_is_labelled_and_a_measurement_is_credited_in_the_same_line() {
        let guess: Fact<f64> = Fact::estimated(0.02, "one minute of this machine at 12 c/kWh");
        let seen: Fact<usize> = Fact::measured(2, "the task list this session started");
        assert!(guess
            .words(|v| format!("{v:.2}$"))
            .contains("0.02$ — estimated (one minute"));
        assert!(seen
            .words(|v| v.to_string())
            .contains("2 — measured (the task list"));
        assert!(!guess.may_reject());
        assert!(seen.may_reject());
        assert_eq!(guess.provenance().label(), "estimated");
        assert_eq!(seen.provenance().label(), "measured");
    }

    #[test]
    fn a_fact_that_was_never_measured_holds_no_number_to_print() {
        let unknown: Fact<f64> = Fact::unknown("no rate exists for this worker");
        assert_eq!(unknown.value(), None);
        assert!(!unknown.may_reject());
        assert!(unknown
            .words(|v| format!("{v:.2}$"))
            .starts_with("not measured (no rate exists"));
    }

    #[test]
    fn no_candidates_is_still_a_hard_error() {
        assert!(CapabilityRouter::route(&task(&["stream"], None), &[]).is_err());
    }

    /// `OR-13`: the posture is asked before the measurements, and the refusal is
    /// the profile's own sentence rather than a capability, load or price reason.
    #[test]
    fn a_worker_the_profile_refuses_is_refused_by_that_name_before_any_check() {
        let decision = CapabilityRouter::route(
            &task(&[], None),
            &[
                external(unmeasured("codex", &["stream"])),
                unmeasured("xencode-own-loop", &[]),
            ],
        )
        .unwrap();
        let refused = decision
            .candidate_evaluations
            .iter()
            .find(|e| e.worker_id == "codex")
            .unwrap();
        assert!(!refused.eligible);
        match refused.rejection.as_ref().unwrap() {
            RejectionReason::ExternalWorkerRefused { refusal } => {
                assert!(refusal.contains("Local Only profile"), "{refusal}");
                assert!(
                    refusal.contains("another vendor's agent"),
                    "the reason must say what the worker is: {refusal}"
                );
                assert!(
                    !refusal.contains("allow_external_workers"),
                    "the setting is stated once, by the rule line a surface prints above its \
                     refusals, rather than pasted onto every one of them: {refusal}"
                );
                assert!(
                    Profile::LOCAL_ONLY
                        .worker_rule()
                        .contains("allow_external_workers=false"),
                    "and that rule line is where it is stated: {}",
                    Profile::LOCAL_ONLY.worker_rule()
                );
            }
            other => panic!("refused for {other:?}, not by the profile"),
        }
        assert_eq!(
            decision.selected_worker.as_deref(),
            Some("xencode-own-loop")
        );
        // The first thing the router asked is the posture, and it names who was
        // refused rather than leaving a reader to notice an absent worker.
        assert_eq!(decision.steps[0].check, "profile");
        assert!(decision.steps[0].decided);
        assert!(
            decision.steps[0].words.contains("codex"),
            "{}",
            decision.steps[0].words
        );
        assert!(
            !decision.steps[0].words.contains("xencode-own-loop"),
            "only the refused are named: {}",
            decision.steps[0].words
        );
    }

    /// A worker the posture refused was never asked about its capabilities, so
    /// the capability line must not report it as a worker that failed them.
    #[test]
    fn a_profile_refusal_is_not_reported_as_a_capability_refusal() {
        let decision = CapabilityRouter::route(
            &task(&["stream"], None),
            &[external(unmeasured("claude", &["acp"]))],
        )
        .unwrap();
        assert!(decision.selected_worker.is_none());
        let capability = decision
            .steps
            .iter()
            .find(|s| s.check == "capabilities")
            .unwrap();
        assert!(!capability.decided);
        assert!(
            capability.words.contains("0 of 0"),
            "nobody reached the capability check: {}",
            capability.words
        );
        assert!(
            decision
                .explanation
                .contains("claude: refused by the Local Only posture"),
            "the summary says which rule refused it, in the rule's own words: {}",
            decision.explanation
        );
        assert!(
            !decision
                .explanation
                .contains("xencode has a roster row for `claude`"),
            "the per-worker sentence is the refusal's own and stays on the candidate, not \
             repeated in the summary: {}",
            decision.explanation
        );
        assert!(
            !decision.explanation.contains("needs stream"),
            "a worker never asked about `stream` cannot be refused for it: {}",
            decision.explanation
        );
    }

    #[test]
    fn opening_the_worker_rule_puts_the_same_candidates_back_in_the_running() {
        let decision = CapabilityRouter::route(
            &open_task(&[], None),
            &[
                external(unmeasured("codex", &["stream"])),
                external(unmeasured("agy", &["stream"])),
            ],
        )
        .unwrap();
        let profile = &decision.steps[0];
        assert_eq!(profile.check, "profile");
        assert!(!profile.decided);
        assert!(profile.words.contains("let every candidate through"));
        assert_eq!(decision.selected_worker.as_deref(), Some("agy"));
    }

    /// The roster row is the whole of what decides this, so a worker name xencode
    /// cannot place is not quietly treated as xencode's own.
    #[test]
    fn a_name_the_roster_cannot_place_reaches_the_measurements() {
        let decision = CapabilityRouter::route(
            &task(&[], None),
            &[unmeasured("my-home-grown-runner", &["stream"])],
        )
        .unwrap();
        assert!(decision.candidate_evaluations[0].eligible);
        assert!(!decision.steps[0].decided);
    }
}
