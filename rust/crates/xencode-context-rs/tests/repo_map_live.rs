//! What the symbol-only repo map (AC-6) is worth on this repository, measured
//! rather than asserted. Ignored by default because it builds the real project
//! index into `.xencode/`, the same way `gold_baseline` does.
//!
//! ```text
//! cargo test -p xencode-context-rs --test repo_map_live -- --ignored --nocapture
//! ```
//!
//! The row's own trap is "a map only helps if the model then asks for the right
//! file", so this reports the part of that claim which is checkable without a
//! model in the loop: at the budget a `Low` machine actually has, retrieval
//! hands over three file bodies, and the map names files around them. When the
//! gold answer is not one of those bodies, is it at least *named* on the map —
//! which is the only way the model can ask for it by name?

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use xencode_context_rs::{
    repo_map_text, retrieve, HardwareProfile, RetrievalIndex, RetrieveOptions,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("crate sits at <root>/rust/crates/<name>")
        .to_path_buf()
}

#[test]
#[ignore]
fn the_map_on_this_repository_names_what_the_bodies_miss() {
    let root = workspace_root();
    xencode_context_rs::init_project(&root, Arc::new(AtomicBool::new(false)), |_| {})
        .expect("init_project on the real workspace");
    let xencode = root.join(xencode_context_rs::XENCODE_DIR);
    let index = RetrievalIndex::load(&xencode).expect("index on disk");

    // What a Low machine sends, and what the tier ceiling allows.
    let opts = RetrieveOptions {
        top_k: HardwareProfile::Low.top_k(),
        ..Default::default()
    };
    let no_changes = HashSet::new();
    let gold = xencode_context_rs::default_gold();
    let mut bodies_only = 0usize;
    let mut bodies_plus_map = 0usize;
    let mut costs = Vec::new();
    for item in &gold {
        let results = retrieve(&item.query, &index, &no_changes, &opts);
        let named: HashSet<&str> = results.iter().map(|r| r.path.as_str()).collect();
        let seeds: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
        let text = repo_map_text(&index, &seeds);
        costs.push(xencode_context_rs::budget::est_tokens(text.len(), false));
        let on_map: HashSet<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("  • "))
            .map(|l| l.split_once(':').map(|(p, _)| p.trim_end()).unwrap_or(l))
            .map(|p| p.split(" [").next().unwrap_or(p))
            .collect();
        let hit_bodies = item.expected.iter().any(|e| named.contains(e.as_str()));
        let hit_map = item
            .expected
            .iter()
            .any(|e| on_map.contains(e.as_str()) || on_map.contains(format!("{e} [").as_str()));
        bodies_only += hit_bodies as usize;
        bodies_plus_map += (hit_bodies || hit_map) as usize;
    }
    costs.sort_unstable();
    println!(
        "over {} gold queries at Low's top_k={}: the {} bodies named the answer in {}, \
         adding the repo map named it in {}",
        gold.len(),
        opts.top_k,
        opts.top_k,
        bodies_only,
        bodies_plus_map
    );
    println!(
        "map cost: median {} tokens, max {} tokens, ceiling {}",
        costs[costs.len() / 2],
        costs[costs.len() - 1],
        xencode_context_rs::REPO_MAP_CAP_TOKENS
    );
    // The consumer, on the same real data: a Low-budget chat turn is asked
    // whether it carries the tier, and what that cost against its target.
    let map = repo_map_text(
        &index,
        &retrieve("where is the login handler?", &index, &no_changes, &opts)
            .iter()
            .map(|r| r.path.clone())
            .collect::<Vec<_>>(),
    );
    let assembly = xencode_context_rs::assemble_chat(xencode_context_rs::ChatInput {
        profile: HardwareProfile::Low,
        context_window: Some(HardwareProfile::Low.ctx_tokens() as u32),
        system: "You are an assistant.",
        agents_md: None,
        anchor_md: None,
        scoped_md: None,
        state_md: None,
        notes_md: None,
        git_summary: "",
        repo_map: &map,
        retrieved: Vec::new(),
        attached_block: "",
        history: &[],
        prompt: "where is the login handler?",
    });
    let last = assembly.turns.last().unwrap().content.clone();
    let map_tokens = assembly
        .tiers
        .iter()
        .filter(|t| t.name == "repo map")
        .map(|t| t.tokens)
        .sum::<u64>();
    println!(
        "assembled Low turn: {} / {} tokens, repo map tier {} tokens, carries the tier: {}",
        assembly.total_tokens,
        assembly.target_tokens,
        map_tokens,
        last.contains("## Repo Map")
    );
    assert!(
        last.contains("## Repo Map"),
        "a Low turn must carry the tier"
    );
    assert!(assembly.total_tokens <= assembly.target_tokens);

    println!("--- the map for \"where is the login handler?\" ---");
    let results = retrieve("where is the login handler?", &index, &no_changes, &opts);
    let seeds: Vec<String> = results.iter().map(|r| r.path.clone()).collect();
    print!("{}", repo_map_text(&index, &seeds));
    assert!(
        costs[costs.len() - 1] <= xencode_context_rs::REPO_MAP_CAP_TOKENS,
        "the tier must never exceed its own ceiling"
    );
}
