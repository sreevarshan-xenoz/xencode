//! What kind of turn this is, worked out from what is already in the tree.
//!
//! No model call and no classifier: the verbs the prompt itself uses are enough
//! to pick between the shapes the retrieval weight table can act on. There is
//! one shape that acts — [`TaskShape::Bugfix`] — and everything else is
//! `General`, because that is what the measurement said. Two others were built,
//! priced on this repository's gold set, and taken back out again:
//!
//! - **refactor** widened the per-file symbol bonus from 16 to 24. On the three
//!   probes written for it, mean reciprocal rank moved by 0.000 on both retrieval
//!   arms — raising the cap raises every candidate's ceiling at once, so the file
//!   that was outranked is still outranked.
//! - **feature** lifted the files the project writes its rules and manifests in.
//!   On its two probes: 0.000 on both arms as well, and +6 is nowhere near the
//!   11–27 a file with a matching name and symbols earns, so the lift changes no
//!   ordering.
//!
//! A shape that changes no weight is a label, so the reading is deliberately
//! narrow, and the plan's `secure` verb is absent for the same reason. So is "did
//! the last command fail": the exit status of a command is not kept anywhere this
//! layer can read, for the reason given on [`shape_of`].
//!
//! The derivation is separate from its use on purpose. [`shape_of`] reads one
//! prompt and returns a [`ShapeRead`], so a rule can be read and tested without a
//! repository, and [`crate::retrieve`] consults only the resulting [`TaskShape`].

use serde::{Deserialize, Serialize};
use std::fmt;

/// The kind of work a prompt asked for, as far as retrieval can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskShape {
    /// Nothing retrieval is priced to respond to. The weight table is used
    /// exactly as it stands.
    #[default]
    General,
    /// Something is broken and the turn is about making it not broken.
    Bugfix,
}

impl TaskShape {
    /// The word the interface and the evaluation files use for this shape.
    pub fn as_str(self) -> &'static str {
        match self {
            TaskShape::General => "general",
            TaskShape::Bugfix => "bugfix",
        }
    }

    /// Read [`TaskShape::as_str`] back. `None` for anything else, including an
    /// empty word, so a gold set that leaves the shape out stays out rather than
    /// silently becoming a `general` row that was never labelled. A word for a
    /// shape this version does not price — `refactor`, `feature` — reads as no
    /// label, which is what a gold set written for an earlier build needs.
    pub fn parse(word: &str) -> Option<TaskShape> {
        match word.trim().to_ascii_lowercase().as_str() {
            "general" => Some(TaskShape::General),
            "bugfix" => Some(TaskShape::Bugfix),
            _ => None,
        }
    }
}

impl fmt::Display for TaskShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A shape and, beside it, the words that chose it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeRead {
    pub shape: TaskShape,
    /// The words that spoke, in the order they were consulted. An interface
    /// shows this beside the shape — `read as bugfix work — the prompt says fix`
    /// — including for `General`, so a wrong reading can be traced back to the
    /// rule that made it rather than to the shape's author.
    pub reasons: Vec<String>,
}

/// Words that mean "this is broken", matched as whole words.
const BUGFIX_WORDS: &[&str] = &[
    "fix",
    "fixes",
    "fixed",
    "fixing",
    "bug",
    "bugs",
    "broken",
    "broke",
    "break",
    "breaks",
    "breaking",
    "crash",
    "crashes",
    "fail",
    "fails",
    "failing",
    "failure",
    "wrong",
    "error",
    "errors",
    "panic",
    "panics",
    "regression",
    "regressions",
    "repair",
];

/// Split a prompt into the lowercase words the verb list is matched against.
///
/// A word is a maximal run of letters, digits and underscores, so `fixed_width`
/// is one word rather than three: an identifier that merely contains a verb is
/// not a request to do that verb, and prompts quote identifiers constantly.
fn words(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// Which shape a turn has, and why — read from the prompt's own verbs.
///
/// Two signals the plan asked for are deliberately not consulted, and the
/// reasons are different:
///
/// - **Whether the last `run_command` exited non-zero** cannot be answered after
///   the fact. A shell command's result reaches the model as text whose *first*
///   line is `exit <n>`, and the per-turn trace keeps only a redacted *tail* of
///   that result, so the one part that says whether a build passed is exactly
///   what is not stored. Recording it would be a change to the trace schema on
///   its own, and a gold query has no previous turn for it to describe, so no
///   partition of the evaluation could show it earning a weight.
/// - **The set git reports as changed** needs no shape to carry it: [`crate::retrieve`]
///   already scores every file in that set above the files it merely resembles.
///   Repeating the fact here would change no weight and make the shape look
///   better-informed than it is.
///
/// A prompt that says both `fix` and `rename` is read as bugfix, because the
/// words for broken code are consulted first. That order is a judgement rather
/// than a measurement, and it is written down as one: retrieving as though a test
/// exists to be found costs little when the turn was really a rename, while a
/// rename's own weights measured 0.000, so it would be giving up the one bias
/// with a number beside it for one without.
pub fn shape_of(prompt: &str) -> ShapeRead {
    let prompt_words = words(prompt);
    let hits: Vec<String> = BUGFIX_WORDS
        .iter()
        .filter(|verb| prompt_words.iter().any(|w| w == *verb))
        .map(|verb| verb.to_string())
        .collect();

    if !hits.is_empty() {
        return ShapeRead {
            shape: TaskShape::Bugfix,
            reasons: vec![format!("the prompt says {}", join_words(&hits))],
        };
    }

    ShapeRead {
        shape: TaskShape::General,
        reasons: vec![
            "no word for broken code in the prompt, so the weights are used as they stand"
                .to_string(),
        ],
    }
}

/// Up to three words, comma-separated, so a reason line stays one line.
fn join_words(words: &[String]) -> String {
    let shown: Vec<&str> = words.iter().take(3).map(String::as_str).collect();
    if words.len() > shown.len() {
        format!("{} (+{} more)", shown.join(", "), words.len() - shown.len())
    } else {
        shown.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(prompt: &str) -> ShapeRead {
        shape_of(prompt)
    }

    #[test]
    fn a_prompt_about_broken_code_is_a_bugfix() {
        assert_eq!(
            read("why does the cache break on a cold start").shape,
            TaskShape::Bugfix
        );
        assert_eq!(read("the count is wrong").shape, TaskShape::Bugfix);
        assert_eq!(
            read("add a flag that asks the server for a count").shape,
            TaskShape::General
        );
        assert_eq!(
            read("rename the caps struct and its callers").shape,
            TaskShape::General
        );
    }

    #[test]
    fn an_ordinary_prompt_stays_general() {
        let read = read("what does this file do");
        assert_eq!(read.shape, TaskShape::General);
        assert!(
            !read.reasons.is_empty(),
            "a shape with no reason is a guess"
        );
    }

    #[test]
    fn a_verb_inside_an_identifier_does_not_count() {
        // "addend" and "fixed_width" are not a request to add or fix anything:
        // whole-word matching is the difference between a rule and a substring
        // hunt that fires on most English prompts.
        assert_eq!(
            read("explain the addend in this table").shape,
            TaskShape::General
        );
        assert_eq!(read("use a fixed_width layout").shape, TaskShape::General);
    }

    #[test]
    fn a_verb_wearing_punctuation_still_counts() {
        // Prompts are typed, not parsed: a question mark or a line break next to
        // the verb must not cost the shape.
        assert_eq!(read("does this fail?").shape, TaskShape::Bugfix);
        assert_eq!(read("please,\nit broke").shape, TaskShape::Bugfix);
    }

    #[test]
    fn every_shape_says_which_words_chose_it() {
        // A shape that arrives without its reason cannot be argued with, and
        // `General` needs the explanation most of all: it is the one that
        // changed nothing, so it is the one a reader will ask about.
        for prompt in [
            "why does it fail",
            "rename the reader and its callers",
            "what does this file do",
        ] {
            let read = read(prompt);
            assert_eq!(read.reasons.len(), 1, "{prompt:?} gave {:?}", read.reasons);
            assert!(!read.reasons[0].is_empty(), "{prompt:?} explained nothing");
        }
        assert!(read("the build fails, then refactor the module").reasons[0]
            .starts_with("the prompt says fail"));
    }

    #[test]
    fn a_word_for_an_unpriced_shape_reads_as_no_shape() {
        for shape in [TaskShape::General, TaskShape::Bugfix] {
            assert_eq!(TaskShape::parse(shape.as_str()), Some(shape));
        }
        // Words a previous build did price nothing on, plus the plan's unpicked
        // `secure` and an empty word: all read as no label rather than as a
        // `general` row somebody never wrote.
        for word in ["secure", "refactor", "feature", ""] {
            assert_eq!(TaskShape::parse(word), None, "{word:?} is not a shape");
        }
    }
}
