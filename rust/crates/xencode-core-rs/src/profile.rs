//! The Local-Only profile (`OR-13`) — one named answer to two questions.
//!
//! xencode already refused the two things separately, in two places that had
//! never heard of each other: a model route that reaches an internet service is
//! refused by `xencode-providers-rs`'s egress policy, and it is refused by
//! default. What was never said anywhere is that those two refusals are one
//! posture with a name, and that the posture is the one the product ships in.
//! This module gives it the name and puts both rules beside each other, so a
//! screen, a report and a refusal can quote one thing instead of two settings
//! the reader has to combine.
//!
//! The pair is descriptive on one side and enforcing on the other, and that
//! split is deliberate:
//!
//! - `allow_cloud_models` is *reported* here and *enforced* by
//!   `xencode_providers_rs::EgressPolicy`, which is where the prefix rules live.
//!   Nothing in this module dials or refuses a model request; a second copy of
//!   that decision would be a second chance to disagree with the router.
//! - `allow_external_workers` is enforced *here*, through [`Profile::check_worker`],
//!   because refusing work is a routing and planning decision and belongs with
//!   the other routing decisions (`routing.rs`).
//!
//! What "external" means is not guessed from a name. The roster in
//! `xencode-agents-rs` is the list of coding-agent CLIs on this machine that
//! belong to somebody else, and a caller reads the answer off it
//! (`is_external_worker`) and hands it in. This crate does not know which
//! binaries are installed here, and it should not have to: a worker whose owner
//! the roster cannot name is *not* claimed as xencode's own, and every caller
//! that reports one says so as the unanswered question it is.

use serde::{Deserialize, Serialize};

/// The two rules a posture is made of, as the settings that hold them name them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    /// Whether a prompt may reach an internet service. Enforced by the egress
    /// policy at the router; carried here so the posture can be named whole.
    pub allow_cloud_models: bool,
    /// Whether work may be handed to an agent that is not xencode's own.
    /// Enforced by [`Profile::check_worker`].
    pub allow_external_workers: bool,
}

/// A worker the posture refused, with the sentence that says why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerRefusal {
    pub worker: String,
    /// Why this worker is refused, in one sentence, naming the posture that
    /// refused it. It deliberately does not repeat the setting that opens the
    /// rule: `worker_rule` states it once above the list, and a surface refusing
    /// ten agents has said it ten times by the end of pasting it onto each.
    pub why: String,
}

impl WorkerRefusal {
    pub fn line(&self) -> String {
        format!("{}: {}", self.worker, self.why)
    }
}

impl Profile {
    /// The posture xencode ships in: a prompt stays on this machine unless the
    /// owner opens the egress rule, and no work is handed to another vendor's
    /// agent unless the owner opens the worker rule. Both are off in a new
    /// config and off in a config written before either key existed, because a
    /// missing switch must not read as permission.
    pub const LOCAL_ONLY: Profile = Profile {
        allow_cloud_models: false,
        allow_external_workers: false,
    };

    pub const fn new(allow_cloud_models: bool, allow_external_workers: bool) -> Self {
        Self {
            allow_cloud_models,
            allow_external_workers,
        }
    }

    /// Both rules still closed.
    pub const fn is_local_only(&self) -> bool {
        !self.allow_cloud_models && !self.allow_external_workers
    }

    /// What to call this posture. A rule that has been opened is said in the
    /// name, because a screen that read `Local Only` while a cloud route was
    /// permitted would be describing a setting nobody made. Once both are open
    /// there is nothing left that is local-only about it, and the name says so.
    pub fn name(&self) -> String {
        match (self.allow_cloud_models, self.allow_external_workers) {
            (false, false) => "Local Only".to_string(),
            (true, false) => "Local Only (cloud models allowed)".to_string(),
            (false, true) => "Local Only (external workers allowed)".to_string(),
            (true, true) => "Not Local Only — both rules opened".to_string(),
        }
    }

    /// The model half of the posture, in one line. A surface that does not
    /// enforce it says so rather than quoting it as if it did.
    pub fn model_rule(&self) -> String {
        if self.allow_cloud_models {
            "model routes: an internet service may be dialled — the egress policy \
             allows an off-machine route (`allow_cloud_models=true`)"
                .to_string()
        } else {
            "model routes: confined to this machine — a `qwen:…`, `google_gemini:…`, \
             `vendor/model` or off-machine `remote:…` route is refused before a \
             connection is opened (`allow_cloud_models=false`; open it with \
             `xencode config set allow_cloud_models true`)"
                .to_string()
        }
    }

    /// The worker half of the posture — the one this module enforces.
    pub fn worker_rule(&self) -> String {
        if self.allow_external_workers {
            "worker routes: work may be handed to another vendor's agent \
             (`allow_external_workers=true`)"
                .to_string()
        } else {
            "worker routes: work is handed only to xencode's own loop — a name on \
             the agent roster is refused by that name (`allow_external_workers=false`; \
             open it with `xencode config set allow_external_workers true`)"
                .to_string()
        }
    }

    /// The two rules as the reader needs them: what is in force, what that
    /// means in practice, and which setting changes it. Reports and panels print
    /// these lines rather than writing their own, so the words a refusal quotes
    /// and the words the status line quotes are one pair.
    pub fn rules(&self) -> Vec<String> {
        vec![self.model_rule(), self.worker_rule()]
    }

    /// Refuse to hand work to an agent xencode has a roster row for, while the
    /// worker rule is closed.
    ///
    /// `external` is the roster's own answer, supplied by the caller: `true` for
    /// a name the roster records as another vendor's agent. A `false` here means
    /// only that the roster has no row to read, and a caller printing that must
    /// say it as an unanswered question — nothing here is claiming a stranger is
    /// local.
    pub fn check_worker(&self, worker: &str, external: bool) -> Result<(), WorkerRefusal> {
        if self.allow_external_workers || !external {
            return Ok(());
        }
        Err(WorkerRefusal {
            worker: worker.to_string(),
            why: format!(
                "the {} profile does not hand work to it: xencode has a roster row for \
                 `{worker}`, so it is another vendor's agent, and what that program sends \
                 off this machine is not xencode's to police",
                self.name()
            ),
        })
    }

    /// Every worker a caller considered, refused or not, as one list of the
    /// refusals. A caller that has already built the refusals does not need it;
    /// it is here so a surface that has a list of names does not write the loop
    /// — and the order of the refusals — itself.
    pub fn refused_workers(&self, considered: &[(String, bool)]) -> Vec<WorkerRefusal> {
        let mut refused = Vec::new();
        for (name, external) in considered {
            if let Err(refusal) = self.check_worker(name, *external) {
                refused.push(refusal);
            }
        }
        refused
    }
}

impl Default for Profile {
    /// The shipped posture, not a permissive one. A caller with no config to
    /// read gets the same answer as a fresh install.
    fn default() -> Self {
        Self::LOCAL_ONLY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_profile_is_local_only_and_refuses_an_agent_on_the_roster() {
        let profile = Profile::default();
        assert_eq!(profile, Profile::LOCAL_ONLY);
        assert!(profile.is_local_only());
        assert_eq!(profile.name(), "Local Only");
        let refusal = profile
            .check_worker("codex", true)
            .expect_err("a roster agent is refused while the rule is closed");
        assert!(refusal.worker == "codex");
        assert!(
            refusal.why.contains("another vendor's agent"),
            "{}",
            refusal.why
        );
        assert!(
            refusal.why.contains("roster row for `codex`"),
            "names the row the refusal was read from: {}",
            refusal.why
        );
        assert!(
            !refusal.why.contains("xencode config set"),
            "the setting is stated once, by the rule line a surface prints above its \
             refusals, rather than once per worker: {}",
            refusal.why
        );
        assert!(
            profile
                .worker_rule()
                .contains("xencode config set allow_external_workers true"),
            "and that rule line is where the way out is named: {}",
            profile.worker_rule()
        );
    }

    /// Opening the rule is the only way work is allowed, and it does not open
    /// the model rule with it — the two are separate consents, so a posture
    /// that says otherwise would be describing a setting nobody changed.
    #[test]
    fn opening_the_worker_rule_leaves_the_model_rule_closed_and_says_so() {
        let profile = Profile::new(false, true);
        assert!(!profile.is_local_only());
        assert_eq!(profile.name(), "Local Only (external workers allowed)");
        assert!(profile.check_worker("codex", true).is_ok());
        let rules = profile.rules();
        assert_eq!(rules.len(), 2);
        assert!(
            rules[0].contains("allow_cloud_models=false"),
            "the model rule stays quoted as closed: {}",
            rules[0]
        );
        assert!(rules[1].contains("allow_external_workers=true"));
    }

    #[test]
    fn a_name_the_roster_cannot_place_is_not_claimed_as_an_agent() {
        let profile = Profile::LOCAL_ONLY;
        assert!(
            profile.check_worker("my-home-grown-runner", false).is_ok(),
            "no roster row means no refusal, which is not the same as a check that passed"
        );
    }

    /// The posture's own name travels into the refusal, so a screen quoting a
    /// refusal cannot contradict the screen quoting the posture.
    #[test]
    fn a_refusal_quotes_the_posture_it_was_refused_by() {
        let profile = Profile::new(false, false);
        let refusal = profile.check_worker("crush", true).unwrap_err();
        assert!(
            refusal.why.contains("Local Only profile"),
            "{}",
            refusal.why
        );
        assert_eq!(refusal.line(), format!("crush: {}", refusal.why));
    }

    /// Both rules open leaves nothing local-only about it, and the name is not
    /// allowed to keep the old label anyway.
    #[test]
    fn opening_both_rules_stops_calling_itself_local_only() {
        let profile = Profile::new(true, true);
        assert_eq!(profile.name(), "Not Local Only — both rules opened");
        assert!(profile.rules()[0].contains("may be dialled"));
        assert!(profile.rules()[1].contains("another vendor's agent ("));
    }

    /// Opening the *other* rule must still show up in the refusal, so a reader
    /// is never told a stricter posture than the one they are actually in.
    #[test]
    fn a_refusal_from_a_partly_opened_posture_names_itself_as_partly_opened() {
        let refusal = Profile::new(true, false)
            .check_worker("kilo", true)
            .unwrap_err();
        assert!(
            refusal
                .why
                .contains("Local Only (cloud models allowed) profile"),
            "{}",
            refusal.why
        );
    }

    /// A surface that enforces only one half quotes that half on its own, so it
    /// never has to reach into the pair by position.
    #[test]
    fn each_rule_can_be_quoted_on_its_own_and_the_pair_is_in_order() {
        let profile = Profile::LOCAL_ONLY;
        assert!(profile.model_rule().starts_with("model routes:"));
        assert!(profile.worker_rule().starts_with("worker routes:"));
        assert_eq!(
            profile.rules(),
            vec![profile.model_rule(), profile.worker_rule()]
        );
    }

    #[test]
    fn the_refusal_list_keeps_the_order_it_was_handed() {
        let profile = Profile::LOCAL_ONLY;
        let considered = vec![
            ("opencode".to_string(), true),
            ("custom-runner".to_string(), false),
            ("agy".to_string(), true),
        ];
        let refused = profile.refused_workers(&considered);
        let names: Vec<&str> = refused.iter().map(|r| r.worker.as_str()).collect();
        assert_eq!(names, vec!["opencode", "agy"]);
        assert_eq!(refused[1].line(), format!("agy: {}", refused[1].why));
    }
}
