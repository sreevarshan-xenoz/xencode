//! EV-4 — which durable facts a turn is actually about.
//!
//! `state.md` is a store, not a message: every fact a person promotes stays in it
//! until the code contradicts one, and a project keeps promoting for months. A
//! prompt budget is a property of one turn, not of the store, so the two are
//! different numbers and something has to decide which facts a turn pays for.
//!
//! Before this module that decision was made twice by writing order and never by
//! relevance. Promotion trimmed the *file* down to the *turn's* budget
//! ([`crate::compact::enforce_state_caps`]), so a project kept only the first
//! handful of facts it ever approved; the tier then took the front of whatever was
//! left. A fact about the login flow promoted in March reached every turn
//! afterwards, and a fact about the file being edited now reached none until the
//! old ones were contradicted.
//!
//! Facts are ranked with the signals [`crate::retrieve`] already uses for files, at
//! the same weights, so a fact and a file are judged by one vocabulary rather than
//! two. The difference is what a fact has to offer: it quotes paths and names things
//! in prose, and it has no symbol table of its own, so retrieval's symbol arm
//! becomes word overlap with the sentence — capped below one path hit, because
//! prose echoes almost anything a query says.
//!
//! What this does not decide is whether a fact is still *true*. A fact the code
//! contradicts never reaches this function: [`crate::compact::drop_stale_facts`]
//! removes it upstream and [`crate::factgc`] keeps the date it stopped being true.
//! Relevance chooses among facts that all still hold.

use std::collections::HashSet;

use crate::budget::est_tokens;
use crate::compact::fact_prose;
use crate::retrieve::word_tokens;

/// A fact that names the file this turn is asking about. `retrieve()` gives a file
/// +10 when its whole name is the query and +6 when a query word appears in it; a
/// fact quotes a path rather than being one, so it carries the second of those.
const PATH_NAME: u64 = 6;

/// … and +5 when a query word matches a directory along that path, as it does for a
/// file.
const PATH_SEGMENT: u64 = 5;

/// A fact about a file the working tree has already changed: +4, the same signal
/// retrieval pays a file for being dirty.
const GIT_CHANGED: u64 = 4;

/// Each distinct query word the fact's own sentence repeats, up to a ceiling worth
/// less than one path hit. A fact that merely echoes the query's nouns is not
/// evidence that it is about the thing being asked.
const WORD_MATCH: u64 = 2;
const WORD_MATCH_CAP: u64 = 2;

/// The task sentence rather than a claim about the code, so the one section whose
/// lines are never ranked and never dropped: a turn is about what it says it is.
const WORKING_ON: &str = "## working-on";

/// One fact line: its bytes as the file writes them, how much it has to do with this
/// turn, and whether this turn is sending it.
struct Fact<'a> {
    text: &'a str,
    score: u64,
    sent: bool,
    /// The `## working-on` line. `score` is never consulted for these.
    task: bool,
}

/// A `##` heading and the fact lines under it.
///
/// The heading rides with its facts rather than counting as one: a section that
/// loses every line it held loses its heading too, so a model is never shown an
/// empty `## completed` and left to report on it.
struct Section<'a> {
    header: &'a str,
    facts: Vec<Fact<'a>>,
}

/// What one turn sends of `state.md`, and what it left in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSelection {
    /// The text for tier 4 — the file's own bytes whenever the whole store fits.
    pub text: String,
    /// Fact lines that reached the turn, the task line aside.
    pub sent: usize,
    /// Fact lines the budget could not carry this time. They stay in the file and
    /// are ranked again next turn against whatever the next turn asks.
    pub left_out: usize,
}

/// The words retrieval would score this turn with: tokenised, lowercased, two
/// characters or more — the same cut [`crate::retrieve::retrieve`] makes, so the
/// words that pick files and the words that pick facts cannot disagree about what
/// was asked.
pub fn query_words(query: &str) -> Vec<String> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    word_tokens(trimmed)
        .into_iter()
        .filter(|word| word.len() >= 2)
        .collect()
}

/// How much this fact line has to do with this query, in retrieval's weights.
///
/// `changed` is the working tree's dirty paths, so a fact about a file being edited
/// now outscores the same fact on a turn that is editing something else.
pub fn fact_score(line: &str, words: &[String], changed: &HashSet<String>) -> u64 {
    if words.is_empty() {
        return 0;
    }
    let paths = fact_paths(line);
    // Three questions about the same set of paths, answered once each: a fact that
    // names five files is about five files, not about the query five times over.
    let names_file = paths.iter().any(|path| {
        let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
        words.iter().any(|word| name.contains(word.as_str()))
    });
    let in_directory = paths.iter().any(|path| {
        words.iter().any(|word| {
            path.split('/')
                .any(|segment| segment.to_ascii_lowercase().contains(word.as_str()))
        })
    });
    let dirty = paths.iter().any(|path| changed.contains(path));
    let prose_words = word_tokens(fact_prose(line));
    let tokens: HashSet<&str> = prose_words.iter().map(String::as_str).collect();
    let unique: HashSet<&str> = words.iter().map(String::as_str).collect();
    let echoed = unique.iter().filter(|word| tokens.contains(*word)).count();
    u64::from(names_file) * PATH_NAME
        + u64::from(in_directory) * PATH_SEGMENT
        + u64::from(dirty) * GIT_CHANGED
        + WORD_MATCH * u64::try_from(echoed.min(WORD_MATCH_CAP as usize)).unwrap_or(0)
}

/// The paths a fact line names, in its prose and in its `[src:path@commit]` marker.
///
/// Detection is lexical on purpose. Whether a path is real is
/// [`crate::compact`]'s question, answered once per turn with a stat, and a fact
/// about a file that has gone never reaches ranking at all. The marker is read as
/// well because a folded fact often repeats its file in prose and a hand-written one
/// often does not, while every promoted fact carries one naming what it was checked
/// against.
fn fact_paths(line: &str) -> Vec<String> {
    let mut paths = Vec::new();
    if let Some((_, tail)) = line.rsplit_once(crate::compact::SRC_OPEN) {
        let cited = tail.split_once(']').map_or(tail, |(p, _)| p);
        let cited = cited.rsplit_once('@').map_or(cited, |(p, _)| p);
        if !cited.is_empty() {
            paths.push(cited.replace('\\', "/"));
        }
    }
    for word in fact_prose(line).split_whitespace() {
        let word = word.trim_end_matches(|c: char| ".;:!?)]}\"'`".contains(c));
        if word.is_empty() || word.starts_with('-') || word.contains("://") {
            continue;
        }
        if !word.contains('/') && !word.contains('.') {
            continue;
        }
        let candidate = word.replace('\\', "/");
        if !paths.contains(&candidate) {
            paths.push(candidate);
        }
    }
    paths
}

/// Choose the part of `state.md` this turn is about, within `budget_tokens`.
///
/// A store that already fits comes back as its own bytes, unchanged: ranking costs a
/// small project nothing and only begins to matter once the file outgrows a turn.
/// The result is what tier 4 sends, and the caller's own cap still bounds it — this
/// chooses, the budgeter still has the last word.
pub fn select_state(
    state_md: &str,
    query: &str,
    changed: &HashSet<String>,
    budget_tokens: u64,
) -> StateSelection {
    let words = query_words(query);
    let mut preamble = String::new();
    let mut sections: Vec<Section> = Vec::new();
    let mut under_working_on = false;
    for line in state_md.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with("## ") {
            under_working_on = line.starts_with(WORKING_ON);
            sections.push(Section {
                header: line,
                facts: Vec::new(),
            });
            continue;
        }
        // `# State`, and anything written before the first section: the store's own
        // title, not a claim to rank.
        if line.starts_with('#') || sections.is_empty() {
            preamble.push_str(line);
            preamble.push('\n');
            continue;
        }
        let score = if under_working_on {
            0
        } else {
            fact_score(line, &words, changed)
        };
        sections
            .last_mut()
            .expect("a section is open")
            .facts
            .push(Fact {
                text: line,
                score,
                sent: false,
                task: under_working_on,
            });
    }
    let facts = sections
        .iter()
        .flat_map(|section| section.facts.iter())
        .filter(|fact| !fact.task)
        .count();
    if facts == 0 || est_tokens(state_md.len(), false) <= budget_tokens {
        return StateSelection {
            text: state_md.to_string(),
            sent: facts,
            left_out: 0,
        };
    }

    // The task line goes wherever the store goes, and pays its own way.
    for section in &mut sections {
        for fact in &mut section.facts {
            fact.sent = fact.task;
        }
    }
    // Best score first, ties to whoever was written first. Each candidate is added
    // and measured against the text the renderer would actually produce, so the
    // cost of a section's heading is counted exactly once and nothing here has to
    // guess at the shape of the file. A store of this size is a few dozen lines:
    // measuring is cheaper by a wide margin than being clever about it.
    let mut ranked: Vec<(usize, usize)> = Vec::new();
    for (s, section) in sections.iter().enumerate() {
        for (f, fact) in section.facts.iter().enumerate() {
            if !fact.task {
                ranked.push((s, f));
            }
        }
    }
    ranked.sort_by(|a, b| {
        let left = sections[a.0].facts[a.1].score;
        let right = sections[b.0].facts[b.1].score;
        right.cmp(&left).then_with(|| a.cmp(b))
    });
    let mut sent = 0usize;
    for (s, f) in ranked {
        sections[s].facts[f].sent = true;
        if est_tokens(render(&preamble, &sections).len(), false) > budget_tokens {
            sections[s].facts[f].sent = false;
        } else {
            sent += 1;
        }
    }
    let text = render(&preamble, &sections);
    StateSelection {
        left_out: facts - sent,
        sent,
        text,
    }
}

/// The chosen facts, in the order the file states them, each under its own heading.
fn render(preamble: &str, sections: &[Section]) -> String {
    let mut out = String::new();
    if !preamble.is_empty() {
        out.push_str(preamble.trim_end_matches('\n'));
        out.push('\n');
    }
    for section in sections {
        if !section.facts.iter().any(|fact| fact.sent) {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(section.header.trim_end_matches('\n'));
        out.push('\n');
        for fact in section.facts.iter().filter(|fact| fact.sent) {
            out.push_str(fact.text.trim_end_matches('\n'));
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::truncate_to_tokens;

    fn tree(paths: &[&str]) -> HashSet<String> {
        paths.iter().map(|p| p.to_string()).collect()
    }

    /// A store whose last line is the one the question is about, and whose forty
    /// earlier lines are about something else — the shape a project reaches by
    /// promoting for months, and the shape a head cut cannot answer.
    fn forty_facts_then(target: &str) -> String {
        let mut out = String::from(
            "# State\n\n## working-on\nanswer why the export has no header\n\n## decisions\n",
        );
        for i in 0..40 {
            out.push_str(&format!(
                "- filler {i}: the retry ladder waits a second between attempts and gives up after four\n"
            ));
        }
        out.push_str(&format!("- {target}\n"));
        out
    }

    #[test]
    fn the_fact_a_question_is_about_arrives_from_the_end_of_the_store() {
        let store = forty_facts_then(
            "the header row is written by src/csv_export.rs before any data row is asked for",
        );
        let ask = "why does the export omit a header row?";
        let pick = select_state(&store, ask, &tree(&[]), 40);
        assert!(
            pick.text.contains("src/csv_export.rs"),
            "the fact the question was about never reached the turn:\n{}",
            pick.text
        );
        assert!(
            pick.left_out > 30,
            "the turn sent most of a store it cannot hold: {} sent, {} left out",
            pick.sent,
            pick.left_out
        );

        // What the rule before this module would have sent: the front of the file,
        // which is the part that has nothing to do with the question.
        let (head, _) = truncate_to_tokens(&store, 40, false);
        assert!(
            head.contains("filler 0"),
            "this fixture no longer reproduces a head cut, so it proves nothing: {head}"
        );
        assert!(
            !head.contains("src/csv_export.rs"),
            "a head cut would have found the fact anyway, so the ranking is untested"
        );
    }

    #[test]
    fn a_store_that_fits_one_turn_is_returned_as_its_own_bytes() {
        // Built by the writer a promotion uses, so the fixture has the shape a real
        // `state.md` has — which ends without a newline, and so is one byte away
        // from anything this module could re-render.
        let store = crate::ContextState {
            working_on: "answer the question".to_string(),
            completed: vec![],
            decisions: vec!["one fact".to_string(), "two fact".to_string()],
            unresolved: vec![],
        }
        .to_markdown();
        let pick = select_state(&store, "anything", &tree(&[]), 800);
        assert_eq!(
            pick.text, store,
            "a store with nothing to choose between was rewritten by the chooser"
        );
        assert_eq!(pick.sent, 2);
        assert_eq!(pick.left_out, 0);
    }

    #[test]
    fn the_task_line_is_sent_when_no_fact_can_be() {
        let store = forty_facts_then("a fact about src/csv_export.rs and its header row");
        let pick = select_state(&store, "header", &tree(&[]), 1);
        assert!(
            pick.text.contains("answer why the export has no header"),
            "the turn lost the one line that says what it is working on:\n{}",
            pick.text
        );
        assert_eq!(pick.sent, 0, "a fact was sent past a budget of one token");
        assert_eq!(pick.left_out, 41);
    }

    #[test]
    fn a_section_that_keeps_no_fact_keeps_no_heading() {
        // A model told "## unresolved" with nothing under it reports an open
        // question that was actually answered three weeks ago.
        let store = "# State\n\n## working-on\nanswer the question\n\n## decisions\n\
                     - the export in src/csv_export.rs writes the header first\n\n## unresolved\n\
                     - does the ladder retry after four\n- does it log the fourth\n";
        let pick = select_state(store, "csv header", &tree(&[]), 35);
        assert!(
            pick.text.contains("## decisions"),
            "the section this turn is about was dropped:\n{}",
            pick.text
        );
        assert!(
            !pick.text.contains("## unresolved"),
            "an empty section was sent for the model to fill in:\n{}",
            pick.text
        );
    }

    #[test]
    fn a_fact_about_a_dirty_file_beats_an_equal_fact_about_a_clean_one() {
        // Both lines name a file the question names, so only the working tree
        // separates them — and the file being edited now is what this turn is doing.
        let store = "# State\n\n## working-on\nanswer it\n\n## decisions\n\
                     - the report is built in src/export_header.rs before rows\n\
                     - the export is built in src/export_footer.rs before rows\n";
        let clean = select_state(store, "export rows", &tree(&[]), 30);
        assert!(
            clean.text.contains("export_header.rs") && !clean.text.contains("export_footer.rs"),
            "written order, not relevance, chose between two unlike facts:\n{}",
            clean.text
        );
        let dirty = select_state(store, "export rows", &tree(&["src/export_footer.rs"]), 30);
        assert!(
            dirty.text.contains("export_footer.rs"),
            "the file being edited now lost to one that is not:\n{}",
            dirty.text
        );
    }

    #[test]
    fn echoing_the_question_never_outbuys_naming_the_file_it_asks_about() {
        let words = query_words("the export header row");
        let echo = "- the header row repeats the export and the row order again";
        // Four of the question's words come back, and the ceiling pays for two.
        assert_eq!(
            fact_score(echo, &words, &tree(&[])),
            WORD_MATCH * WORD_MATCH_CAP
        );
        let path = "- a ladder waits [src:src/header.rs@abc1234]";
        assert_eq!(
            fact_score(path, &words, &tree(&[])),
            PATH_NAME + PATH_SEGMENT
        );
        assert!(
            fact_score(path, &words, &tree(&[])) > fact_score(echo, &words, &tree(&[])),
            "a fact that merely echoes the question's nouns can win the budget"
        );
        assert_eq!(
            fact_score(echo, &[], &tree(&[])),
            0,
            "a turn with no question ranks by nothing"
        );
    }

    #[test]
    fn a_marker_names_the_file_even_when_the_prose_never_repeats_it() {
        // A promoted fact carries its provenance marker; a hand-written one often
        // does not. Both have to rank, or the store's own format decides relevance.
        let words = query_words("which file holds ledger data");
        assert_eq!(
            fact_score(
                "- this keeps the write durable [src:src/ledger.rs@abc1234]",
                &words,
                &tree(&[])
            ),
            PATH_NAME + PATH_SEGMENT
        );
        assert_eq!(
            fact_score("- the ledger lives in src/ledger.rs", &words, &tree(&[])),
            PATH_NAME + PATH_SEGMENT + WORD_MATCH,
            "a fact that names its file in prose ranks apart from one that marks it"
        );
    }

    #[test]
    fn facts_of_equal_relevance_keep_the_order_the_file_writes_them_in() {
        // Nothing about the question separates them, so the tie is not a coin toss:
        // a store read the same way twice must send the same bytes.
        let store = "# State\n\n## working-on\nanswer it\n\n## decisions\n\
                     - alpha filler about the ladder\n\
                     - beta filler about the ladder\n\
                     - gamma filler about the ladder\n\
                     - delta filler about the ladder\n";
        let pick = select_state(store, "nothing matches this at all", &tree(&[]), 35);
        assert!(
            pick.text.contains("alpha") && pick.text.contains("beta"),
            "ties came from the end of the file:\n{}",
            pick.text
        );
        assert!(
            !pick.text.contains("gamma"),
            "more than the budget was sent:\n{}",
            pick.text
        );
    }

    #[test]
    fn sent_and_left_out_together_are_the_facts_in_the_file() {
        let store = forty_facts_then("the header row in src/csv_export.rs");
        for budget in [1, 12, 40, 800] {
            let pick = select_state(&store, "csv header row", &tree(&[]), budget);
            assert_eq!(
                pick.sent + pick.left_out,
                41,
                "the task line was counted as a fact, or a fact went uncounted, at budget {budget}"
            );
        }
    }
}
