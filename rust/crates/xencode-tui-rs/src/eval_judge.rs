//! Ranking the attempts that came close, and only those (EV-10).
//!
//! [`crate::task_eval`] already decides pass or fail from an exit code and a
//! diff, and nothing here is allowed to change its mind. What this module adds is
//! an answer to the question a pass rate cannot ask: of the attempts that did not
//! fix it, which one came closest? That is a ranking, so the model is asked to
//! order things and never to grade them — its reply has no field to say "this one
//! was actually fine" in.
//!
//! Only near-miss outcomes are shown to it, which is the whole point of the item:
//! a case that never ran has nothing to rank, a case that passed needs no judge,
//! and a case that edited its own grader is explained by that fact alone. What
//! remains is work someone wrote down that the exit code rejected.
//!
//! Three biases are the reason this is not just an extra column of text, and each
//! is handled in code rather than by a warning in the prompt:
//!
//! - **Position.** The candidates are listed in an order derived from a digest of
//!   their identities rather than the order they ran, and then the same question
//!   is asked a second time with the list in the opposite order. If the two
//!   answers disagree, the ordering is thrown away and the report says it
//!   disagreed — which is a measurement of the bias, not a correction of it.
//! - **Verbosity.** A candidate is shown as the change it left and what the tests
//!   said afterwards. The agent's own sentences are never sent, and the change is
//!   capped, so a wall of edits cannot outshout a small one.
//! - **Self-preference.** Nothing in the request names a model, and the caller may
//!   point the judge at a different one (`--judge-model`). It is still the case
//!   that a judge written by the same model it is reading may recognise its own
//!   habits, and no amount of relabeling proves otherwise; that limit is reported
//!   rather than asserted away.
//!
//! Adding this prompt to the registry moves the digest the eval records, so a run
//! taken before it is not offered as a comparison — even though nothing the agent
//! was told has changed. The rule is deliberately blunt: the version names the
//! whole set of instructions in the tree, and a comparison across a change to that
//! set is one a person should make on purpose.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use xencode_context_rs::prompts::{eval_judge_prompt, registry};

use crate::task_eval::{CaseResult, TaskEvalOptions, TaskEvalReport};

/// How many attempts one ranking request is allowed to hold. Beyond this the
/// judge is shown the ones that ran first and the report says how many were left
/// out, because silently truncating a comparison is worse than an incomplete one.
pub const MAX_SHOWN: usize = 26;
/// A ranking is a list of letters. This is not a place where an answer can go
/// long, and a judge that writes an essay has not ranked anything.
pub const ANSWER_CAP: u32 = 256;
/// How much of the test output one candidate is allowed. The verdict is already
/// known; this is only the evidence a person would want to read.
const GRADER_TAIL_CAP: usize = 400;

/// One judge-assisted ranking, and what it was asked under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeRun {
    pub model: String,
    /// The version of the ranking instruction itself, so a ordering can be told
    /// apart from one asked under different words.
    pub prompt_version: String,
    /// Near-miss cases in the run, including any left out for want of room.
    pub near_misses: usize,
    /// The ones actually shown, in the order they were shown. Ids, not letters.
    pub shown: Vec<String>,
    /// The first answer, as ids in the order the judge named them.
    pub first: Vec<String>,
    /// The second answer, from the reversed listing.
    pub second: Vec<String>,
    /// The ordering, when both arrangements produced one. `None` when the judge
    /// named nothing, or named the same things in a different order.
    pub order: Option<Vec<String>>,
    /// Whether asking twice, with the list the other way round, said the same
    /// thing. This is the only evidence about position bias this harness collects,
    /// and one agreement is one data point.
    pub stable: bool,
    /// Set when no question could be asked at all, which is not the same as a
    /// judge that declined to rank.
    pub error: Option<String>,
}

impl JudgeRun {
    /// Where the run stands, in as many lines as it needs and not one more.
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.error.is_some() {
            out.push(format!(
                "judge: {}",
                self.error.as_deref().unwrap_or_default()
            ));
            return out;
        }
        if self.near_misses == 0 {
            out.push(
                "judge: no case was a near miss, so nothing was ranked — a case that never ran, \
                 that passed, that changed no files, or that edited its own grader is not a \
                 close call"
                    .to_string(),
            );
            return out;
        }
        let head = format!(
            "judge {} at prompts {}: {} of {} near-miss attempt(s) shown{}",
            self.model,
            self.prompt_version,
            self.shown.len(),
            self.near_misses,
            if self.shown.len() == self.near_misses {
                String::new()
            } else {
                format!(
                    ", {} left out because one ranking holds {}",
                    self.near_misses - self.shown.len(),
                    MAX_SHOWN
                )
            }
        );
        out.push(head);
        match &self.order {
            Some(order) if !order.is_empty() => {
                out.push(format!(
                    "closest to a fix, as ranked: {} · the same attempts listed in the opposite \
                     order produced the same ranking, which is the only hold this ordering has on \
                     position bias",
                    order.join(", ")
                ));
            }
            _ if self.first.is_empty() && self.second.is_empty() => out.push(
                "judge: it named nothing it had been shown, so nothing is ranked".to_string(),
            ),
            _ => out.push(format!(
                "judge: the ranking moved when the same attempts were listed in the opposite \
                 order ({} against {}) — it is not reported, because a list that shuffles when \
                 the questions are shuffled is measuring where things sit, not how good they are",
                if self.first.is_empty() {
                    "nothing".to_string()
                } else {
                    self.first.join(", ")
                },
                if self.second.is_empty() {
                    "nothing".to_string()
                } else {
                    self.second.join(", ")
                }
            )),
        }
        out
    }
}

/// The cases worth a judge's time: they ran, they were graded as not-a-fix by an
/// exit code that has already spoken, and they left something behind to read.
/// Returns indices into `cases`, in the order the cases ran.
pub fn near_misses(cases: &[CaseResult]) -> Vec<usize> {
    cases
        .iter()
        .enumerate()
        .filter(|(_, case)| {
            case.error.is_none()
                && !case.passed
                && !case.edited_grader
                && !case.changed.is_empty()
                && !case.diff.trim().is_empty()
        })
        .map(|(index, _)| index)
        .collect()
}

/// The id one case is known by inside a run.
pub(crate) fn case_id(case: &CaseResult) -> String {
    format!("r{}/{}", case.attempt, case.shape)
}

/// The labels handed out, `A` first. Kept to single capitals so a reply can be
/// read back by word rather than by guesswork.
fn labels(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| char::from(b'A' + i as u8).to_string())
        .collect()
}

/// The order the attempts are listed in: derived from their ids, so it is stable
/// across runs of the same cases and is not the order they happened to run in.
/// `salt` is the prompt digest, so editing the instruction reshuffles the listing
/// rather than letting one arrangement settle into the numbers.
pub(crate) fn arrangement(ids: &[String], salt: &str) -> Vec<usize> {
    let mut keyed: Vec<(u64, usize)> = ids
        .iter()
        .enumerate()
        .map(|(index, id)| (stable_key(salt, id), index))
        .collect();
    keyed.sort_by_key(|(key, index)| (*key, *index));
    keyed.into_iter().map(|(_, index)| index).collect()
}

fn stable_key(salt: &str, id: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    salt.hash(&mut hasher);
    id.hash(&mut hasher);
    hasher.finish()
}

/// The attempts block: the letters, the defect each was aimed at, the change it
/// left, what the tests said afterwards. Nothing else — not who wrote it, not how
/// long it took, not how many rounds it used, not a word of what it said.
/// `labels` is indexed the same way `cases` is, so a letter stays with the
/// attempt it was given no matter what order they are listed in.
pub(crate) fn render_attempts(
    cases: &[&CaseResult],
    listing: &[usize],
    labels: &[String],
) -> String {
    let mut out = String::from("The attempts:\n");
    for index in listing {
        let case = cases[*index];
        let tail = one_line_tail(&case.grader_tail, GRADER_TAIL_CAP);
        out.push_str(&format!(
            "\n{} — defect: {}\nthe tests afterwards said: {}\nthe change it left:\n{}\n",
            labels[*index], case.title, tail, case.diff
        ));
    }
    out
}

/// The last thing the grader printed, on as few lines as it takes.
fn one_line_tail(tail: &str, cap: usize) -> String {
    let flat = tail.trim().replace('\n', " ");
    let flat = match flat.split_once("test result") {
        Some((_, rest)) => format!("test result{rest}"),
        None => flat,
    };
    if flat.len() <= cap {
        return flat;
    }
    let mut cut = cap;
    while !flat.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &flat[..cut])
}

/// Which of the letters it was shown the judge named, in the order it named them,
/// as ids. Anything it made up is dropped, and so is everything it did not say:
/// an attempt the judge forgot is simply not ranked rather than ranked last.
pub(crate) fn parse_order(answer: &str, shown: &[(String, String)]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for token in answer.split(|c: char| !c.is_ascii_alphabetic()) {
        if let Some((id, _)) = shown.iter().find(|(_, label)| label == token) {
            if !out.iter().any(|known| known == id) {
                out.push(id.clone());
            }
        }
    }
    out
}

/// What the two arrangements add up to: an ordering, or the finding that the
/// judge's ordering did not survive being asked twice.
pub(crate) fn combine(
    first: &[String],
    second: &[String],
    shown: &[String],
) -> (Option<Vec<String>>, bool) {
    // Both silent is a judge that declined, not a disagreement.
    if first.is_empty() && second.is_empty() {
        return (None, false);
    }
    // A listing is only worth anything if it covers every attempt it was shown;
    // a partial one says the judge stopped, not that the rest are worse.
    let covers =
        |named: &[String]| named.len() == shown.len() && named.iter().all(|id| shown.contains(id));
    if covers(first) && covers(second) && first == second {
        (Some(first.to_vec()), true)
    } else {
        (None, false)
    }
}

/// Ask, twice, over the same route the eval itself used.
pub async fn judge(options: &TaskEvalOptions, report: &TaskEvalReport) -> JudgeRun {
    let model = options
        .judge_model
        .clone()
        .filter(|m| !m.trim().is_empty())
        .unwrap_or_else(|| options.model.clone());
    let prompt_version = registry()
        .into_iter()
        .find(|prompt| prompt.name == "eval-judge")
        .map(|prompt| prompt.version())
        .unwrap_or_else(|| "unversioned".to_string());
    let indexes = near_misses(&report.cases);
    let empty = JudgeRun {
        model: model.clone(),
        prompt_version: prompt_version.clone(),
        near_misses: indexes.len(),
        shown: Vec::new(),
        first: Vec::new(),
        second: Vec::new(),
        order: None,
        stable: false,
        error: None,
    };
    if indexes.is_empty() {
        return empty;
    }

    let shown_cases: Vec<&CaseResult> = indexes.iter().map(|i| &report.cases[*i]).collect();
    let ids: Vec<String> = shown_cases.iter().map(|case| case_id(case)).collect();
    let mut listing = arrangement(&ids, &prompt_version);
    listing.truncate(MAX_SHOWN);
    // A letter belongs to an attempt for the whole run, so the second asking can
    // list the same attempts backwards without moving any of them.
    let letters = labels(listing.len());
    let mut by_case = vec![String::new(); shown_cases.len()];
    for (position, index) in listing.iter().enumerate() {
        by_case[*index] = letters[position].clone();
    }
    let shown_ids: Vec<String> = listing.iter().map(|index| ids[*index].clone()).collect();
    let pairs: Vec<(String, String)> = shown_ids
        .iter()
        .cloned()
        .zip(letters.iter().cloned())
        .collect();

    let request = eval_judge_prompt(&render_attempts(&shown_cases, &listing, &by_case));
    let reversed: Vec<usize> = listing.iter().rev().copied().collect();
    let second_request = eval_judge_prompt(&render_attempts(&shown_cases, &reversed, &by_case));

    let manager = manager_for(options);
    let answer_a = ask(&manager, &model, options, &request).await;
    let answer_b = ask(&manager, &model, options, &second_request).await;
    let (answer_a, answer_b) = match (answer_a, answer_b) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(error), _) | (_, Err(error)) => {
            return JudgeRun {
                error: Some(error),
                ..empty
            };
        }
    };

    let first = parse_order(&answer_a, &pairs);
    // The second listing hands out the same letters to the same attempts — a
    // label belongs to an attempt, not to a position — so its answer is read
    // against the same pairs.
    let second = parse_order(&answer_b, &pairs);
    let (order, stable) = combine(&first, &second, &shown_ids);
    JudgeRun {
        shown: shown_ids,
        first,
        second,
        order,
        stable,
        ..empty
    }
}

/// The manager the eval's own addresses imply, so a judge dials the same server
/// as the agent unless `--judge-model` says otherwise.
fn manager_for(options: &TaskEvalOptions) -> xencode_providers_rs::ProviderManager {
    use xencode_models_rs::{LlamaCppClient, OllamaClient};
    use xencode_providers_rs::ProviderManager;
    let timeout = options.timeout_secs;
    let ollama = options
        .ollama_url
        .clone()
        .unwrap_or_else(|| "http://localhost:11434".to_string());
    let llama = options
        .llama_cpp_url
        .clone()
        .unwrap_or_else(|| "http://localhost:8080".to_string());
    let manager = ProviderManager::new(OllamaClient::new(&ollama, timeout), None, None, None, None)
        .with_llama_cpp(LlamaCppClient::new(&llama, timeout))
        .with_request_timeout(timeout);
    match &options.remote_base_url {
        Some(url) => manager.with_remote(url, None),
        None => manager,
    }
}

async fn ask(
    manager: &xencode_providers_rs::ProviderManager,
    model: &str,
    options: &TaskEvalOptions,
    request: &str,
) -> Result<String, String> {
    use xencode_models_rs::LlamaCppOptions;
    let sampling = LlamaCppOptions {
        temperature: options.temperature,
        seed: options.seed,
        max_tokens: Some(match options.max_tokens {
            None => ANSWER_CAP,
            Some(0) => 0,
            Some(cap) => cap.min(ANSWER_CAP),
        }),
        ..Default::default()
    };
    let messages = vec![xencode_providers_rs::ChatMessage::text(
        "user",
        request.to_string(),
    )];
    manager
        .generate_with_options(model, &messages, Some(&sampling))
        .await
        .map_err(|error| format!("the judge could not be asked: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task_eval::test_case as case;

    #[test]
    fn only_an_attempt_that_left_a_rejected_change_behind_is_a_near_miss() {
        let cases = vec![
            // Ran, failed, and left work: the definition.
            case("off-by-one", 1, &["src/lib.rs"], "-sum += 1\n+sum += 2"),
            // Never ran at all — nothing was asked, nothing to rank.
            {
                let mut c = case("swallowed-error", 1, &[], "");
                c.error = Some("the model request failed".to_string());
                c
            },
            // Passed: the exit code already settled it.
            {
                let mut c = case("lost-update", 1, &["src/lib.rs"], "+fixed");
                c.passed = true;
                c
            },
            // Failed having rewritten its own test: explained by that alone.
            {
                let mut c = case(
                    "dead-code",
                    1,
                    &["tests/behaviour.rs"],
                    "+#[test] fn ok(){}",
                );
                c.edited_grader = true;
                c
            },
            // Failed without touching anything: there is no attempt to read.
            case("inverted-check", 1, &[], ""),
        ];
        assert_eq!(near_misses(&cases), vec![0]);
    }

    #[test]
    fn a_ranking_names_only_what_the_judge_was_shown_and_in_the_order_it_said() {
        let shown = vec![
            ("r1/off-by-one".to_string(), "A".to_string()),
            ("r1/lost-update".to_string(), "B".to_string()),
            ("r1/dead-code".to_string(), "C".to_string()),
        ];
        assert_eq!(
            parse_order("C, A, B", &shown),
            vec!["r1/dead-code", "r1/off-by-one", "r1/lost-update"]
        );
        // A letter that was never handed out is not an attempt.
        assert_eq!(
            parse_order("Z beats everything", &shown),
            Vec::<String>::new()
        );
        // Prose is not a ranking, and neither is a repeat.
        assert_eq!(parse_order("unsure", &shown), Vec::<String>::new());
        assert_eq!(
            parse_order("A then A again", &shown),
            vec!["r1/off-by-one".to_string()]
        );
    }

    #[test]
    fn an_ordering_that_moves_when_the_list_is_reversed_is_not_reported() {
        let ids = vec!["a".to_string(), "b".to_string()];
        let (order, stable) = combine(
            &["a".to_string(), "b".to_string()],
            &["a".to_string(), "b".to_string()],
            &ids,
        );
        assert_eq!(
            order.as_deref(),
            Some(&["a".to_string(), "b".to_string()][..])
        );
        assert!(stable);
        let (moved, stable) = combine(
            &["a".to_string(), "b".to_string()],
            &["b".to_string(), "a".to_string()],
            &ids,
        );
        assert_eq!(moved, None, "the judge's order followed the list");
        assert!(!stable);
        // Naming two of three is a judge that stopped, not a ranking.
        let (partial, _) = combine(&["a".to_string()], &["a".to_string()], &ids);
        assert_eq!(partial, None);
        // Saying nothing twice is not a disagreement.
        let (silent, _) = combine(&[], &[], &ids);
        assert_eq!(silent, None);
    }

    #[test]
    fn a_candidate_is_shown_as_its_change_and_nothing_else() {
        let mut c = case("off-by-one", 1, &["src/lib.rs"], "-sum += 1\n+sum += 2");
        c.title = "a loop that stops one reading early".to_string();
        c.grader_tail = "test result: FAILED. 0 passed; 1 failed".to_string();
        c.completion_tokens = Some(900);
        c.elapsed_ms = 61_000;
        let request = render_attempts(&[&c], &[0], &["A".to_string()]);
        assert!(request.contains("A — defect: a loop that stops one reading early"));
        assert!(request.contains("+sum += 2"));
        assert!(request.contains("test result: FAILED"));
        // The things that let a longer or luckier answer win, and the name of
        // whoever wrote it, are not in what the judge reads.
        for absent in ["dolphin", "900", "61", "round", "tool"] {
            assert!(
                !request.contains(absent),
                "the judge was told about {absent}: {request}"
            );
        }
    }

    #[test]
    fn the_listing_is_not_the_order_the_cases_ran_in() {
        let ids: Vec<String> = (0..8).map(|i| format!("r1/shape-{i}")).collect();
        let first = arrangement(&ids, "prompt-v1");
        assert_ne!(
            first,
            (0..8).collect::<Vec<usize>>(),
            "a listing that matches the running order tests nothing"
        );
        // Same ids, same instruction: same listing. A new instruction: a new one.
        assert_eq!(first, arrangement(&ids, "prompt-v1"));
        assert_ne!(first, arrangement(&ids, "prompt-v2"));
    }

    /// The one rule the whole item exists for: a judge may order, never decide.
    #[test]
    fn a_ranking_cannot_change_what_the_exit_code_graded() {
        let mut report = TaskEvalReport {
            model: "m".to_string(),
            server: "s".to_string(),
            prompt_version: "p".to_string(),
            approval: "edit-allow".to_string(),
            temperature: Some(0.0),
            seed: Some(42),
            max_tokens: Some(1024),
            out_dir: std::path::PathBuf::new(),
            results: std::path::PathBuf::new(),
            cases: vec![case("off-by-one", 1, &["src/lib.rs"], "+changed")],
            judge: None,
        };
        report.judge = Some(JudgeRun {
            model: "m".to_string(),
            prompt_version: "v".to_string(),
            near_misses: 1,
            shown: vec!["r1/off-by-one".to_string()],
            first: vec!["r1/off-by-one".to_string()],
            second: vec!["r1/off-by-one".to_string()],
            order: Some(vec!["r1/off-by-one".to_string()]),
            stable: true,
            error: None,
        });
        assert_eq!(report.graded(), 1);
        assert_eq!(report.passed(), 0);
        assert_eq!(report.pass_rate(), Some(0.0));
        let lines = report.lines().join("\n");
        assert!(lines.contains("pass rate: 0/1"), "{lines}");
        assert!(lines.contains("closest to a fix"), "{lines}");
    }

    #[tokio::test]
    async fn nothing_to_rank_asks_no_question() {
        let report = TaskEvalReport {
            model: "m".to_string(),
            server: "http://127.0.0.1:1".to_string(),
            prompt_version: "p".to_string(),
            approval: "edit-allow".to_string(),
            temperature: Some(0.0),
            seed: Some(42),
            max_tokens: Some(1024),
            out_dir: std::path::PathBuf::new(),
            results: std::path::PathBuf::new(),
            cases: Vec::new(),
            judge: None,
        };
        // Nothing is listening at that address and nothing needs to be: with no
        // near miss in the run, the judge is not asked at all.
        let run = judge(&TaskEvalOptions::default(), &report).await;
        assert_eq!(run.near_misses, 0);
        assert!(run.error.is_none());
        assert!(run.lines().join("\n").contains("nothing was ranked"));
    }

    /// The same two questions, asked of a model that is really running. Ignored
    /// by default because it needs a server: point it at one with
    /// `XENCODE_JUDGE_LIVE_URL` and name the id with `XENCODE_JUDGE_LIVE_MODEL`.
    /// The two attempts are written out here rather than produced by an agent —
    /// what this checks is the route, the reversal and the reading back of an
    /// answer, not any model's work on a defect.
    #[tokio::test]
    #[ignore]
    async fn a_model_on_a_real_server_can_be_asked_twice_backwards() {
        let Ok(url) = std::env::var("XENCODE_JUDGE_LIVE_URL") else {
            eprintln!("XENCODE_JUDGE_LIVE_URL is not set, so no question was asked");
            return;
        };
        let model = std::env::var("XENCODE_JUDGE_LIVE_MODEL")
            .unwrap_or_else(|_| "llamacpp:dolphin".to_string());
        let options = TaskEvalOptions {
            model: model.clone(),
            llama_cpp_url: Some(url),
            ..TaskEvalOptions::default()
        };
        let report = TaskEvalReport {
            model,
            server: options.llama_cpp_url.clone().unwrap_or_default(),
            prompt_version: "live".to_string(),
            approval: "edit-allow".to_string(),
            temperature: options.temperature,
            seed: options.seed,
            max_tokens: options.max_tokens,
            out_dir: std::path::PathBuf::new(),
            results: std::path::PathBuf::new(),
            cases: vec![
                case(
                    "off-by-one",
                    1,
                    &["src/lib.rs"],
                    "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,3 @@\n-    for i in 0..items.len() - 1 {\n+    for i in 0..=items.len() - 1 {\n",
                ),
                case(
                    "swallowed-error",
                    1,
                    &["src/lib.rs"],
                    "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,3 +1,3 @@\n-    let _ = parse(line);\n+    if let Ok(v) = parse(line) { total += v; }\n",
                ),
            ],
            judge: None,
        };
        let run = judge(&options, &report).await;
        for line in run.lines() {
            println!("{line}");
        }
        assert!(
            run.error.is_none(),
            "the judge could not be asked: {:?}",
            run.error
        );
        assert_eq!(run.near_misses, 2);
        assert_eq!(run.shown.len(), 2, "both attempts should have been shown");
    }
}
