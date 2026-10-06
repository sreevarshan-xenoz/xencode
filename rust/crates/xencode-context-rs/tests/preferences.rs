//! QK-2 — the part of `AGENTS.md` a person wrote by hand has to reach the model
//! whatever else is in that file.
//!
//! `AGENTS.md` is capped at [`AGENTS_CAP_TOKENS`] and the cap is a *head* cut, so
//! the tail of a long file never arrives. The tail is exactly where the product
//! itself puts the lines a human approved: `/lesson approve` appends under
//! `## Lessons` at the end of the file. So a project whose instructions grew past
//! the cap held an approval queue that filled a file its own prompts had stopped
//! reading, and said nothing about it.

use xencode_context_rs::context::{
    stable_system_text, AGENTS_CAP_TOKENS, PREFERENCES_CAP_TOKENS, STABLE_END_MARKER,
};
use xencode_context_rs::{assemble_chat, truncate_to_tokens, ChatInput, HardwareProfile};

/// One line of ordinary project prose. A real `AGENTS.md` is wrapped, and the cap
/// cuts at a line boundary, so a fixture of one unwrapped paragraph would be
/// truncated by that rule rather than by the budget being tested here.
const RULE: &str = "Rules that matter, written out in full.\n";

/// `## Lessons` is what `/lesson approve` writes; `## Preferences` is the block a
/// person types for themselves. Both are human-owned and both live at the tail.
fn a_long_file_ending_in_a_lesson() -> String {
    let mut text = String::from("# Project instructions\n\n");
    for n in 0..220 {
        text.push_str(&format!("{n}. {RULE}"));
    }
    text.push_str("\n## Lessons\n- Never edit src/auth.rs without running the auth tests first.\n");
    text
}

fn head_of(agents: &str) -> String {
    stable_system_text("SYSTEM", Some(agents), None)
}

fn tiers_of(agents: &str) -> Vec<xencode_context_rs::TierDoc> {
    chat_of(agents, None).tiers
}

fn chat_of(agents: &str, anchor: Option<&str>) -> xencode_context_rs::ChatAssembly {
    assemble_chat(ChatInput {
        profile: HardwareProfile::Low,
        context_window: None,
        system: "SYSTEM",
        agents_md: Some(agents),
        anchor_md: anchor,
        scoped_md: None,
        state_md: None,
        notes_md: None,
        git_summary: "",
        repo_map: "",
        retrieved: vec![],
        attached_block: "",
        history: &[],
        prompt: "the question",
    })
}

#[test]
fn the_fixture_is_actually_longer_than_the_cap_it_has_to_survive() {
    let agents = a_long_file_ending_in_a_lesson();
    assert!(
        agents.len() > AGENTS_CAP_TOKENS as usize * 4,
        "a fixture inside the cap proves nothing: {} chars against a {} char budget",
        agents.len(),
        AGENTS_CAP_TOKENS * 4
    );
}

#[test]
fn an_approved_lesson_at_the_end_of_a_long_file_still_reaches_the_prompt() {
    let head = head_of(&a_long_file_ending_in_a_lesson());
    assert!(
        head.contains("Never edit src/auth.rs"),
        "a person approved this line and the prompt dropped it"
    );
}

#[test]
fn the_lessons_are_cut_out_of_the_bulk_so_no_byte_is_sent_twice() {
    let agents = a_long_file_ending_in_a_lesson();
    let head = head_of(&agents);
    assert_eq!(
        head.matches("Never edit src/auth.rs").count(),
        1,
        "the pinned block and the bulk it was removed from cannot both ride"
    );
}

#[test]
fn the_pinned_block_is_a_tier_a_budget_report_can_see() {
    let tiers = tiers_of(&a_long_file_ending_in_a_lesson());
    let pinned = tiers
        .iter()
        .find(|tier| tier.name == "preferences")
        .expect("a tier that is not named in the ledger is a tier no report can explain");
    assert!(
        pinned.tokens > 0,
        "the block reached the head but counted nothing"
    );
    let bulk = tiers
        .iter()
        .find(|tier| tier.name == "agents.md")
        .expect("the bulk keeps its own tier");
    assert!(
        pinned.tokens + bulk.tokens <= AGENTS_CAP_TOKENS + PREFERENCES_CAP_TOKENS,
        "two caps govern two tiers: bulk {}, pinned {}",
        bulk.tokens,
        pinned.tokens
    );
}

#[test]
fn a_block_past_its_own_ceiling_is_cut_rather_than_eating_the_file_budget() {
    let mut agents = String::from("# Project instructions\n\n## Lessons\n");
    for n in 0..400 {
        agents.push_str(&format!(
            "- Lesson {n} runs past the block's own ceiling.\n"
        ));
    }
    let head = head_of(&agents);
    assert!(
        head.contains("- Lesson 0 "),
        "the earliest lessons still belong"
    );
    assert!(
        !head.contains("- Lesson 399 "),
        "the block has a ceiling of its own and must not spend the whole file's"
    );
    let tiers = tiers_of(&agents);
    let pinned = tiers
        .iter()
        .find(|tier| tier.name == "preferences")
        .unwrap();
    assert!(pinned.tokens <= PREFERENCES_CAP_TOKENS, "{}", pinned.tokens);
    assert!(head.ends_with(STABLE_END_MARKER), "the head still closes");
}

#[test]
fn a_file_holding_nothing_human_owned_sends_the_bytes_it_always_sent() {
    let agents = "# Project instructions\n\nJust rules, no lessons.\n";
    assert_eq!(
        head_of(agents),
        format!("SYSTEM\n\n{agents}\n\n{STABLE_END_MARKER}"),
        "every project without the block keeps the KV-cache prefix it already had"
    );
    assert!(tiers_of(agents)
        .iter()
        .all(|tier| tier.name != "preferences"));
}

#[test]
fn a_preference_block_is_pinned_the_way_a_lesson_is() {
    let mut agents = String::from("# Project instructions\n\n");
    for n in 0..220 {
        agents.push_str(&format!("{n}. {RULE}"));
    }
    agents.push_str(
        "\n## Preferences\n- Answer in under five sentences before any code.\n## Other things\n",
    );
    agents.push_str(RULE);
    let head = head_of(&agents);
    assert!(
        head.contains("Answer in under five sentences"),
        "the heading a person types for themselves is the other half of this row"
    );
}

#[test]
fn a_section_ends_where_the_next_heading_opens_and_no_byte_is_lost() {
    let mut agents = String::from("# Project instructions\n\n## Lessons\n- Pin me.\n\n## Rules\n");
    for n in 0..220 {
        agents.push_str(&format!("{n}. {RULE}"));
    }
    let (bulk, pinned) = xencode_context_rs::context::human_owned_sections(&agents);
    assert_eq!(
        pinned, "## Lessons\n- Pin me.\n\n",
        "the section is its own bytes and stops where the next heading opens"
    );
    assert!(
        bulk.contains("## Rules"),
        "the next heading opens the bulk again"
    );
    assert_eq!(
        bulk.len() + pinned.len(),
        agents.len(),
        "every byte of the file lands in exactly one of the two"
    );
    let head = head_of(&agents);
    assert!(
        head.contains("- Pin me."),
        "the lesson arrives on its own cap"
    );
    assert!(
        head.contains("0. Rules that matter"),
        "and the bulk keeps the text after the section, on the file's own cap"
    );
}

/// `## Lessons` opened and never closed. Markdown reads that section as
/// "everything below it", so a file that simply ends puts far more inside it than
/// the block's own ceiling holds.
fn a_file_whose_lessons_never_close(rules: usize) -> String {
    let mut agents = String::from("# Project instructions\n\n## Lessons\n");
    for n in 0..rules {
        agents.push_str(&format!("{n}. {RULE}"));
    }
    agents
}

/// Which rule numbers actually arrived, in the order they arrived. Counting lines
/// rather than guessing indices is the only way to assert where the two budgets
/// meet without hard-coding what they add up to today.
fn rules_in(text: &str) -> Vec<usize> {
    text.lines()
        .filter_map(|line| line.split('.').next()?.trim().parse().ok())
        .collect()
}

#[test]
fn an_unclosed_section_sends_a_contiguous_run_of_rules_with_no_gap_and_no_repeat() {
    let head = head_of(&a_file_whose_lessons_never_close(400));
    let mut arrived = rules_in(&head);
    assert!(
        !arrived.is_empty() && arrived.len() < 400,
        "the file has to be cut, but not emptied: {} rules arrived",
        arrived.len()
    );
    arrived.sort_unstable();
    assert_eq!(
        arrived,
        (0..arrived.len()).collect::<Vec<usize>>(),
        "the rules that arrive must be 0, 1, 2 … a gap is a byte the block swallowed \
         and never handed back, a repeat is the same rule billed twice"
    );
    assert!(head.ends_with(STABLE_END_MARKER), "the head still closes");
}

#[test]
fn the_pinned_block_rides_last_so_editing_a_lesson_disturbs_the_prefix_as_little_as_it_can() {
    // The stable head is what a provider caches its key-value prefix on. The bulk
    // of the file comes first and the human-owned block comes after it, so the
    // edit a person actually makes — rewording a lesson — invalidates the anchor
    // and nothing before it. Reorder these two and every lesson approval would
    // blow away the cached instructions of every project that has one.
    let agreed = a_long_file_ending_in_a_lesson();
    let reworded = agreed.replace("src/auth.rs", "src/payments.rs");
    let head = head_of(&agreed);
    let other = head_of(&reworded);
    let bulk_end = head
        .find("## Lessons")
        .expect("the pinned block is in the head, after the bulk");
    assert!(
        head.find("0. Rules that matter").unwrap() < bulk_end,
        "the file's bulk has to come first"
    );
    let shared = head
        .bytes()
        .zip(other.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        shared >= bulk_end,
        "rewording one lesson invalidated the instructions ahead of it: the two \
         prompts part ways at byte {shared}, before the block opens at {bulk_end}"
    );
}

#[test]
fn an_unclosed_section_never_un_sends_a_rule_the_head_cut_used_to_keep() {
    // Before the pin existed, the whole file rode one head cut. Opening a section
    // at its tail must not make the prompt *smaller*: whatever that cut was
    // already sending has to still be there, and the remainder past the block's
    // ceiling falls back into the file's own budget rather than the bin.
    let agents = a_file_whose_lessons_never_close(400);
    let (sent_before_change, _) = truncate_to_tokens(&agents, AGENTS_CAP_TOKENS, false);
    let kept_before = rules_in(&sent_before_change);
    let head = head_of(&agents);
    for n in &kept_before {
        assert!(
            head.contains(&format!("{n}. Rules")),
            "rule {n} was in every prompt before this change and is not now"
        );
    }
    assert!(
        rules_in(&head).len() > kept_before.len(),
        "and the two budgets together send more than one did: {} against {}",
        rules_in(&head).len(),
        kept_before.len()
    );
    assert!(
        agents.len() > head.len(),
        "the file still does not fit, which is the point of a cap"
    );
}

#[test]
fn a_top_level_heading_ends_the_block_so_a_new_chapter_is_not_pinned_with_it() {
    // `## Lessons` is a section of a document, not a trapdoor to the end of it.
    // Without the level-1 reset, a file that opens a new chapter after its
    // lessons would have that whole chapter billed against the block's ceiling.
    let agents = "# Project instructions\n\n## Lessons\n- Pin me.\n\n\
                  # A later chapter\n\nProse that belongs to the file, not to the block.\n";
    let (bulk, pinned) = xencode_context_rs::context::human_owned_sections(agents);
    assert_eq!(
        pinned, "## Lessons\n- Pin me.\n\n",
        "the block closes where the chapter opens"
    );
    assert!(bulk.contains("# A later chapter"), "the chapter is bulk");
    assert_eq!(bulk.len() + pinned.len(), agents.len());
}

#[test]
fn the_stable_prefix_size_a_report_prints_covers_every_leading_tier() {
    // `/ctx` prints one number for the part of the prompt a provider can cache.
    // It used to add up the first three tiers by position, which was right only
    // while three was all there were: a fourth leading tier would have made it
    // under-report the person's own block. The count now comes from the
    // assembler, so it cannot go stale when a tier is added.
    let agents = a_long_file_ending_in_a_lesson();
    let chat = chat_of(
        &agents,
        Some("# Anchor\n\nProject facts, unchanged between turns.\n"),
    );
    let pinned = chat
        .tiers
        .iter()
        .find(|tier| tier.name == "preferences")
        .expect("this file has a lessons block")
        .tokens;
    let first_three_by_position: u64 = chat.tiers.iter().take(3).map(|tier| tier.tokens).sum();
    let every_leading_tier: u64 = chat
        .tiers
        .iter()
        .take_while(|tier| {
            matches!(
                tier.name,
                "system" | "agents.md" | "preferences" | "anchor.md"
            )
        })
        .map(|tier| tier.tokens)
        .sum();
    assert!(
        pinned > 0 && chat.stable_tokens == every_leading_tier,
        "the reported head is the whole head: {} against {}",
        chat.stable_tokens,
        every_leading_tier
    );
    assert!(
        chat.stable_tokens > first_three_by_position,
        "counting three leading tiers is only correct while there are three; here it \
         would print {} instead of {}",
        first_three_by_position,
        chat.stable_tokens
    );
}

#[test]
fn two_human_owned_sections_share_one_budget_and_neither_is_lost_whole() {
    // The case a real project reaches for: a person's own `## Preferences` at the
    // end of the file, and the lessons `/lesson approve` appended after it. One
    // budget governs both, in the order the bytes already sit in, so the block
    // that comes second is the one that runs past the ceiling. Past the ceiling it
    // falls back into the file's budget, so a long lesson history still reaches the
    // prompt — the file's cap, not the block's, is what finally decides how much of
    // it arrives.
    let mut agents = String::from("# Project instructions\n\n## Preferences\n- Answer in under five sentences.\n\n## Lessons\n");
    for n in 0..400 {
        agents.push_str(&format!("- Lesson {n} runs past the shared ceiling.\n"));
    }
    let head = head_of(&agents);
    assert!(
        head.contains("Answer in under five sentences."),
        "the block that opens the section is inside the ceiling and arrives"
    );
    let (_, pinned) = xencode_context_rs::context::human_owned_sections(&agents);
    let (pin_head, _) = truncate_to_tokens(&pinned, PREFERENCES_CAP_TOKENS, false);
    let in_the_block = pinned_lines(&pin_head).len();
    let in_the_prompt = pinned_lines(&head).len();
    assert!(
        in_the_prompt > in_the_block,
        "the lessons past the block's ceiling ride the file's cap, not the bin: \
         {in_the_block} lines fit the block, {in_the_prompt} arrived"
    );
    assert!(head.ends_with(STABLE_END_MARKER), "the head still closes");
}

/// Count the lesson lines in a block of text — the fixture's own shape, not a
/// guessed index into it.
fn pinned_lines(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| line.starts_with("- Lesson "))
        .map(str::to_string)
        .collect()
}
