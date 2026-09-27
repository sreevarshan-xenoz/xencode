//! Which saved profile takes a turn (MI-7).
//!
//! A profile has always been a model plus two sampling knobs, and it has always
//! been applied by hand: the Custom Models panel writes it into the session with
//! `Enter`. This module adds the one thing that was missing — a turn asking for
//! the kind of work a profile is marked for, choosing it without anyone
//! remembering to. It is a mapping, not a decision-maker: the reading is
//! [`xencode_context_rs::shape_of`], the same whole-word rule the retrieval
//! consults, and what that rule can tell apart is what a profile can be marked
//! for. Nothing here classifies a prompt a second time, and nothing here guesses.
//!
//! The shape the reading reports is not the shape of the *work*; it is the shape
//! of the words. `bugfix` means the prompt said something about broken code, and
//! `general` means it did not — which is a very wide net, and the reason the whole
//! mechanism is off until `model_routing` is turned on.
//!
//! ## What a turn may not do
//!
//! Move a llama.cpp server. A self-started `llama-server` holds one model, and
//! changing it is a control call that stops the current load and starts another —
//! seconds, and a different conversation than the one the user is in. A hand
//! applied profile does that on purpose, because a key was pressed; a profile
//! chosen by a rule does not get to. So a turn that matches a llama.cpp profile for
//! a model other than the one already chosen is reported and left alone.
//!
//! Every other route is a request away: Ollama loads the model named in the
//! request, so a switch there costs a reload rather than a refusal — which is what
//! `ollama_keep_alive` exists to budget.

use xencode_config_rs::XencodeConfig;
use xencode_context_rs::{shape_of, TaskShape};

/// The profile a turn was given, in the shape a caller can put into a request.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnProfile {
    pub name: String,
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    /// One line saying which profile took the turn and the words that decided it,
    /// so a wrong choice can be argued with instead of guessed at.
    pub reason: String,
}

/// What a turn's model should be, once the profiles have been consulted.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum TurnChoice {
    /// No profile was configured, or none matched. The session's own model stands.
    #[default]
    Unclaimed,
    /// This profile takes the turn.
    Taken(TurnProfile),
    /// A profile matched the reading and was not used, with the reason to say so.
    /// A rule that silently does nothing is indistinguishable from a rule that
    /// never fired, so this is reported in the same place the choice would be.
    Refused { name: String, why: String },
}

impl TurnChoice {
    /// The profile to run the turn with, if there is one.
    pub fn profile(&self) -> Option<&TurnProfile> {
        match self {
            TurnChoice::Taken(profile) => Some(profile),
            _ => None,
        }
    }

    /// The line for the transcript: what was chosen, or what was declined and
    /// why. `None` when no profile claimed the turn, because there is nothing to
    /// report about a turn that ran as configured.
    pub fn note(&self) -> Option<String> {
        match self {
            TurnChoice::Unclaimed => None,
            TurnChoice::Taken(profile) => Some(profile.reason.clone()),
            TurnChoice::Refused { name, why } => {
                Some(format!("profile {name} was not used: {why}"))
            }
        }
    }
}

/// Which profile, if any, this prompt's reading claims.
///
/// `current_model` is the model this turn would have gone to anyway, which is what
/// the llama.cpp rule below compares against; a caller that resolved a model for
/// itself — `xencode query` does, when the configured default turns out not to be
/// installed — passes that answer rather than the config's.
///
/// The first profile in the list wins a tie, because the list is the order it was
/// written in and a rule with two answers needs a stated tie-break rather than a
/// HashMap. A profile with no `for_task` is applicable by hand only, which is what
/// every profile written before this field existed means.
pub fn choose_profile(config: &XencodeConfig, current_model: &str, prompt: &str) -> TurnChoice {
    if !config.model_routing {
        return TurnChoice::Unclaimed;
    }
    let read = shape_of(prompt);
    let Some(profile) = config.model_profiles.iter().find(|profile| {
        profile
            .for_task
            .as_deref()
            .and_then(TaskShape::parse)
            .is_some_and(|shape| shape == read.shape)
    }) else {
        return TurnChoice::Unclaimed;
    };

    let same_model = profile.model == current_model;
    if !same_model
        && (xencode_providers_rs::routes_to_llamacpp(&profile.model)
            || xencode_providers_rs::routes_to_llamacpp(current_model))
    {
        return TurnChoice::Refused {
            name: profile.name.clone(),
            why: format!(
                "it names {model}, and a running llama.cpp server holds one model at a time — \
                 apply a profile like this by hand in the model panel if you meant to swap it",
                model = profile.model
            ),
        };
    }

    let mut knobs = Vec::new();
    if let Some(temperature) = profile.temperature {
        knobs.push(format!("temperature {temperature}"));
    }
    if let Some(max_tokens) = profile.max_tokens {
        knobs.push(format!("{max_tokens} tokens at most"));
    }
    // The two numbers are llama.cpp's request fields; an Ollama or cloud model has
    // nowhere to put either, which is what the Custom Models panel says when it
    // offers them. A profile marked with a number and pointed at Ollama therefore
    // changes nothing about sampling, and the note has to say that rather than
    // announce a number the request never carried.
    let knob_words = if knobs.is_empty() {
        String::new()
    } else if xencode_providers_rs::routes_to_llamacpp(&profile.model) {
        format!(", with {}", knobs.join(" and "))
    } else {
        format!(
            ", which sets {} — numbers that route has no place for, so the model's own \
             defaults answer",
            knobs.join(" and ")
        )
    };
    // Two sayings for the model, because two different things can be true of it:
    // the profile names the model already chosen, or it names another one. The
    // first reads as no change at all unless it is said plainly, and a note that
    // reads as no change is a note a user cannot act on.
    let subject = if same_model {
        format!(
            "{} took this turn on the model already in use",
            profile.name
        )
    } else {
        format!("{} took this turn on {}", profile.name, profile.model)
    };
    let reason = format!("{subject}{knob_words} — {}", read.reasons[0]);
    TurnChoice::Taken(TurnProfile {
        name: profile.name.clone(),
        model: profile.model.clone(),
        temperature: profile.temperature,
        max_tokens: profile.max_tokens,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xencode_config_rs::ModelProfile;

    fn profile(name: &str, model: &str, for_task: Option<&str>) -> ModelProfile {
        ModelProfile {
            name: name.to_string(),
            model: model.to_string(),
            temperature: None,
            max_tokens: None,
            for_task: for_task.map(str::to_string),
        }
    }

    fn config(routing: bool, profiles: Vec<ModelProfile>) -> XencodeConfig {
        XencodeConfig {
            default_model: "bigmodel:latest".to_string(),
            model_routing: routing,
            model_profiles: profiles,
            ..XencodeConfig::default()
        }
    }

    const BUGFIX: &str = "the parser fails on a trailing comma";
    const PLAIN: &str = "explain what this file does";

    /// The usual case: the turn goes to whatever the config names as default.
    /// `choose_profile` takes the model from the caller instead of reading it
    /// here, because a caller may already have resolved a different one; the
    /// llama.cpp tests below pass it by hand to show that difference matters.
    fn choose(config: &XencodeConfig, prompt: &str) -> TurnChoice {
        choose_profile(config, &config.default_model, prompt)
    }

    #[test]
    fn nothing_moves_a_turn_until_routing_is_turned_on() {
        let config = config(
            false,
            vec![profile("quick", "smallmodel:latest", Some("bugfix"))],
        );
        assert_eq!(
            choose(&config, BUGFIX),
            TurnChoice::Unclaimed,
            "a profile that was never enabled must not take a turn"
        );
    }

    #[test]
    fn a_profile_marked_for_bugfix_work_takes_a_turn_that_says_something_is_broken() {
        let config = config(
            true,
            vec![profile("quick", "smallmodel:latest", Some("bugfix"))],
        );
        let choice = choose(&config, BUGFIX);
        let taken = choice.profile().expect("the bugfix profile took the turn");
        assert_eq!(taken.model, "smallmodel:latest");
        assert!(
            taken.reason.contains("the prompt says fail"),
            "{}",
            taken.reason
        );
        assert_eq!(
            choose(&config, PLAIN),
            TurnChoice::Unclaimed,
            "a turn that says nothing about broken code belongs to no bugfix profile"
        );
    }

    #[test]
    fn a_profile_marked_general_claims_every_turn_that_is_not_read_as_a_bugfix() {
        // This is the wide one, and the test exists to keep it honest: `general`
        // is what the rule falls back to, so marking a profile with it hands that
        // profile the majority of turns rather than a narrow kind of work.
        let config = config(
            true,
            vec![profile("reading", "smallmodel:latest", Some("general"))],
        );
        assert!(choose(&config, PLAIN).profile().is_some());
        assert_eq!(choose(&config, BUGFIX), TurnChoice::Unclaimed);
    }

    #[test]
    fn the_first_profile_written_for_a_shape_wins_it() {
        let config = config(
            true,
            vec![
                profile("first", "one:latest", Some("bugfix")),
                profile("second", "two:latest", Some("bugfix")),
            ],
        );
        assert_eq!(choose(&config, BUGFIX).profile().unwrap().name, "first");
    }

    #[test]
    fn a_word_the_rule_has_no_reading_for_claims_nothing() {
        // The reading has two words for two shapes, so a profile wearing another
        // one — `refactor`, `feature`, a misspelling — can only ever sit unused.
        // That is not an error, and it is not allowed to look like one either: the
        // turn runs on the model the user chose.
        let config = config(
            true,
            vec![
                profile("rework", "smallmodel:latest", Some("refactor")),
                profile("broken", "fixer:latest", Some("bugfix")),
            ],
        );
        let choice = choose(&config, BUGFIX);
        let taken = choice.profile().unwrap();
        assert_eq!(taken.name, "broken");
        assert_eq!(choose(&config, PLAIN), TurnChoice::Unclaimed);
    }

    #[test]
    fn a_profile_that_only_sets_sampling_keeps_the_sessions_own_model() {
        // A llama.cpp profile naming the model already chosen: the turn's model is
        // unchanged, and both numbers really do go into the request.
        let config = XencodeConfig {
            default_model: "llamacpp:gemma".to_string(),
            model_routing: true,
            model_profiles: vec![ModelProfile {
                temperature: Some(0.2),
                max_tokens: Some(256),
                ..profile("careful", "llamacpp:gemma", Some("bugfix"))
            }],
            ..XencodeConfig::default()
        };
        let choice = choose_profile(&config, "llamacpp:gemma", BUGFIX);
        let taken = choice.profile().unwrap();
        assert_eq!(taken.model, "llamacpp:gemma");
        assert_eq!(
            (taken.temperature, taken.max_tokens),
            (Some(0.2), Some(256))
        );
        assert!(
            taken.reason.contains("the model already in use"),
            "a turn that changed nothing but sampling should say so: {}",
            taken.reason
        );
        assert!(
            taken
                .reason
                .contains("with temperature 0.2 and 256 tokens at most"),
            "{}",
            taken.reason
        );
    }

    #[test]
    fn a_number_the_route_cannot_carry_is_named_as_one_that_changes_nothing() {
        // The same two numbers on a profile pointed at Ollama, where neither has
        // anywhere to go. Saying them as if the request carried them would be the
        // one thing in this module a user could not check.
        let config = XencodeConfig {
            default_model: "bigmodel:latest".to_string(),
            model_routing: true,
            model_profiles: vec![ModelProfile {
                temperature: Some(0.2),
                max_tokens: Some(256),
                ..profile("careful", "smallmodel:latest", Some("bugfix"))
            }],
            ..XencodeConfig::default()
        };
        let choice = choose(&config, BUGFIX);
        let reason = &choice.profile().unwrap().reason;
        assert!(reason.contains("smallmodel:latest"), "{reason}");
        assert!(reason.contains("no place for"), "{reason}");
        assert!(
            reason.contains("the model's own defaults answer"),
            "{reason}"
        );
    }

    #[test]
    fn a_llama_cpp_model_is_never_swapped_by_a_rule() {
        // The trap this item names: the server holds one model, and the cost of
        // changing it is a reload the user did not ask for.
        let to_llama = config(
            true,
            vec![profile("small", "llamacpp:gemma", Some("bugfix"))],
        );
        let choice = choose_profile(&to_llama, "bigmodel:latest", BUGFIX);
        let TurnChoice::Refused { name, why } = &choice else {
            panic!("a llama.cpp swap must be refused, not run: {choice:?}");
        };
        assert_eq!(name, "small");
        assert!(why.contains("one model at a time"), "{why}");
        assert!(choice.note().unwrap().contains("was not used"));

        // The same rule from the other side: the turn is already going to a
        // llama.cpp model and the profile is not. The config still says
        // `bigmodel:latest`, which is why the caller passes the model it settled
        // on rather than this rule reading it back out of the config.
        let from_llama = config(true, vec![profile("other", "qwen:latest", Some("bugfix"))]);
        assert!(matches!(
            choose_profile(&from_llama, "llamacpp:gemma", BUGFIX),
            TurnChoice::Refused { .. }
        ));

        // A llama.cpp profile that names the model already in use is not a swap,
        // so the rule leaves it alone: it takes the turn for its sampling.
        let same = config(
            true,
            vec![profile("small", "llamacpp:gemma", Some("bugfix"))],
        );
        let choice = choose_profile(&same, "llamacpp:gemma", BUGFIX);
        assert!(
            choice.profile().is_some(),
            "a profile naming the model already in use only sets sampling: {choice:?}"
        );
    }

    #[test]
    fn an_ollama_profile_may_move_a_turn_because_the_server_loads_what_is_named() {
        let config = config(
            true,
            vec![profile("quick", "smallmodel:latest", Some("bugfix"))],
        );
        assert!(choose(&config, BUGFIX).profile().is_some());
    }
}
