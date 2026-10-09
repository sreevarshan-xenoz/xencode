//! QM-2 and MEM-3, the half a person sees: when a durable fact's cited file has
//! moved on, or the code it names has stopped existing, `/ctx kv` says which fact
//! left the prompt instead of silently shrinking it.
//!
//! The staleness itself is decided in `xencode-context-rs/tests/state_staleness.rs`,
//! at the level the plan's done-when names — assembly. This file drives the real
//! command through the real `App`, because a fact that is correctly dropped and
//! incorrectly hidden is still a person losing a note with no way to find out
//! which one or why. Everything here runs on the shipped path: the candidate is
//! folded by `/ctx promote`, which stamps the `[src:…]` marker from a real
//! `git` commit in a scratch repository, and the edit that makes the fact stale
//! is an edit to that same real file.
//!
//! This is a separate test binary from `state_fold.rs` on purpose, and the
//! scenarios below share one run for the same reason: `.xencode/` is written
//! under the process working directory, `set_current_dir` is process-wide, and
//! cargo runs the tests inside one binary on parallel threads.

use std::path::Path;
use std::time::Duration;

use tokio::sync::mpsc;
use xencode_context_rs::{STATE_CANDIDATE_FILE, XENCODE_DIR};
use xencode_tui_rs::app::App;

fn git(root: &Path, args: &[&str]) {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .expect("git must be on PATH for this test");
    assert!(
        out.status.success(),
        "`git {}` failed in {}: {}",
        args.join(" "),
        root.display(),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Type what a person types and wait for the panel to answer.
async fn submit_and_wait(app: &mut App<'_>, prompt: &str, expect: &str) -> Vec<String> {
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    app.chat_input.insert_str(prompt);
    app.submit_message(tx);
    let mut lines = Vec::new();
    for _ in 0..400 {
        while let Ok(line) = rx.try_recv() {
            lines.push(line);
        }
        if lines.iter().any(|line| line.contains(expect)) {
            return lines;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("nothing said {expect:?} within 10s of {prompt}: {lines:#?}");
}

/// `/ctx kv` answers in one pass, so a command that should say *nothing* about
/// staleness still needs its tier-4 line as proof the panel actually ran.
async fn context_panel(app: &mut App<'_>) -> Vec<String> {
    submit_and_wait(app, "/ctx kv", "Tier 4 state.md").await
}

/// The fact, in the shape a fold writes it: a sentence plus the file it is
/// about. The path is what the promotion stamps, so it has to name a file that
/// really exists in the scratch repository.
const FACT: &str = "the token check runs before the handler in src/auth.rs";

#[tokio::test]
async fn the_context_panel_names_a_durable_fact_it_had_to_drop() {
    let root = std::env::temp_dir().join(format!(
        "xencode-stale-notice-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(root.join("src")).unwrap();
    git(&root, &["init", "-q"]);
    git(&root, &["config", "user.name", "xencode test"]);
    git(&root, &["config", "user.email", "test@xencode.invalid"]);
    std::fs::write(
        root.join("src/auth.rs"),
        "pub fn check(token: &str) -> bool {\n    !token.is_empty()\n}\n",
    )
    .unwrap();
    git(&root, &["add", "-A"]);
    git(&root, &["commit", "-q", "-m", "auth module"]);

    let previous = std::env::current_dir().unwrap();
    std::env::set_current_dir(&root).unwrap();

    let xencode = root.join(XENCODE_DIR);
    std::fs::create_dir_all(&xencode).unwrap();
    std::fs::write(
        xencode.join(STATE_CANDIDATE_FILE),
        format!("## working-on\n- finish the durable-tier reader\n\n## decisions\n- {FACT}\n"),
    )
    .unwrap();

    let mut app = App::for_tests();

    // ── promote: the marker comes from this repository's own commit ────────
    submit_and_wait(&mut app, "/ctx promote", "state.md written").await;
    let state_path = xencode.join("state.md");
    let durable = std::fs::read_to_string(&state_path).expect("state.md was not written");
    let marked = durable
        .lines()
        .find(|line| line.contains(FACT))
        .unwrap_or_else(|| panic!("the promoted fact is not in state.md:\n{durable}"));
    assert!(
        marked.contains("[src:src/auth.rs@"),
        "the fact carries no provenance marker, so nothing could ever notice it went stale:\n{marked}"
    );

    // ── while the file matches the marker, the panel stays quiet about it ──
    let clean = context_panel(&mut app).await;
    assert!(
        !clean
            .iter()
            .any(|line| line.contains("dropped as stale") || line.contains("stale: ")),
        "a fact whose file has not moved was reported as stale: {clean:#?}"
    );

    // ── the done-when: edit the cited file ─────────────────────────────────
    std::fs::write(
        root.join("src/auth.rs"),
        "pub fn check(token: &str) -> bool {\n    !token.is_empty() && !token.starts_with(\"dev\")\n}\n",
    )
    .unwrap();

    let lines = context_panel(&mut app).await;
    let count = lines
        .iter()
        .find(|line| line.contains("dropped as stale"))
        .unwrap_or_else(|| panic!("the tier-4 line never said a fact was dropped: {lines:#?}"));
    assert!(
        count.contains("1 dropped as stale"),
        "the panel says a whole set of facts went stale when only one did: {count}"
    );
    assert!(
        count.contains("· 1 fact line(s) on disk"),
        "the fact count on that line no longer matches the file it describes — {count}\nstate.md:\n{durable}"
    );
    let named = lines
        .iter()
        .find(|line| line.contains("stale: ") && line.contains(FACT))
        .unwrap_or_else(|| panic!("the dropped fact was counted but never named: {lines:#?}"));
    assert!(
        named.contains("/ctx fold"),
        "the person is told the note is gone with no way to get it back: {named}"
    );
    assert!(
        named.contains("[src:src/auth.rs@"),
        "the notice drops the marker, which is the one piece of evidence for why this fact is stale: {named}"
    );

    // The durable file itself is untouched — dropping is per-turn, not destructive.
    assert_eq!(
        std::fs::read_to_string(&state_path).unwrap(),
        durable,
        "a stale fact was deleted from state.md instead of kept out of one prompt"
    );

    // ── a marker this repository cannot resolve ────────────────────────────
    // The fact is kept: history that is unreadable here is not history that
    // disproved the line. But the person has to be told the check did not run,
    // or a tier going in unverified looks exactly like a tier that passed.
    git(&root, &["add", "-A"]);
    git(&root, &["commit", "-q", "-m", "reject dev tokens"]);
    std::fs::write(
        &state_path,
        "# State\n\n## decisions\n- the token check runs before the handler [src:src/auth.rs@deadbeef]\n",
    )
    .unwrap();
    let uncheckable = context_panel(&mut app).await;
    let row = uncheckable
        .iter()
        .find(|line| line.contains("not checkable here"))
        .unwrap_or_else(|| {
            panic!("a fact no local commit can be found for was not reported: {uncheckable:#?}")
        });
    assert!(
        row.contains("1 not checkable here (no such commit locally)"),
        "the panel says a set of facts could not be checked when only one could not: {row}"
    );
    assert!(
        !uncheckable
            .iter()
            .any(|line| line.contains("dropped as stale") || line.contains("stale: ")),
        "an unverifiable fact was treated as a disproven one: {uncheckable:#?}"
    );

    // ── the other direction the trap points in ─────────────────────────────
    // A `state.md` this feature never stamped — someone wrote it by hand before
    // QM-2, or on another machine — is not a pile of stale facts. Absence of a
    // marker is not evidence of a change.
    std::fs::write(
        &state_path,
        "# State\n\n## decisions\n- Rust-first for new code\n",
    )
    .unwrap();
    let quiet = context_panel(&mut app).await;
    assert!(
        !quiet.iter().any(|line| {
            line.contains("dropped as stale")
                || line.contains("stale: ")
                || line.contains("not checkable here")
        }),
        "unmarked facts were reported as stale or unchecked for having no marker: {quiet:#?}"
    );
    assert!(
        quiet
            .iter()
            .any(|line| line.contains("· 1 fact line(s) on disk")),
        "the hand-written fact did not reach the panel's count at all: {quiet:#?}"
    );

    // ── MEM-3: a fact about the code, re-checked against the code ───────────
    // The symbol case, through the real command. This fact cites no file, so the
    // provenance marker above cannot see it at all — the name in the tree is the
    // only evidence, and the panel has to say that much when the name goes.
    std::fs::write(
        root.join("src/token_check.rs"),
        "pub fn validate_token(token: &str) -> bool {\n    !token.is_empty()\n}\n",
    )
    .unwrap();
    git(&root, &["add", "src/token_check.rs"]);
    git(
        &root,
        &["commit", "-q", "-m", "the token validator, by name"],
    );
    std::fs::write(
        xencode.join(STATE_CANDIDATE_FILE),
        "## working-on\n- finish the durable-tier reader\n\n## decisions\n- validate_token rejects an empty token\n",
    )
    .unwrap();
    let promoted = submit_and_wait(&mut app, "/ctx promote", "state.md written").await;
    let durable = std::fs::read_to_string(&state_path).unwrap();
    let marked = durable
        .lines()
        .find(|line| line.contains("validate_token rejects"))
        .unwrap_or_else(|| panic!("the promoted fact is not in state.md:\n{durable}"));
    assert!(
        marked.contains("[chk:validate_token"),
        "a fact naming this repository's own code was made durable with no claim \
         a later turn could re-run:\n{marked}"
    );
    assert!(
        promoted
            .iter()
            .any(|line| line.contains("given a claim this code can be re-checked against")),
        "the promotion wrote a check and never told the person it had: {promoted:#?}"
    );

    // While the name is declared, the panel stays quiet about it.
    let clean = context_panel(&mut app).await;
    assert!(
        !clean
            .iter()
            .any(|line| line.contains("dropped as stale") || line.contains("stale: ")),
        "a fact naming code that is still there was reported as stale: {clean:#?}"
    );

    // ── the done-when: rename the function, commit it ──────────────────────
    std::fs::write(
        root.join("src/token_check.rs"),
        "pub fn check_the_token(token: &str) -> bool {\n    !token.is_empty()\n}\n",
    )
    .unwrap();
    git(&root, &["add", "-A"]);
    git(&root, &["commit", "-q", "-m", "rename the token validator"]);

    let lines = context_panel(&mut app).await;
    let count = lines
        .iter()
        .find(|line| line.contains("dropped as stale"))
        .unwrap_or_else(|| {
            panic!("a fact whose symbol stopped existing was not reported: {lines:#?}")
        });
    assert!(
        count.contains("1 dropped as stale"),
        "the panel says a set of facts went stale when one symbol moved: {count}"
    );
    let named = lines
        .iter()
        .find(|line| line.contains("stale: ") && line.contains("validate_token rejects"))
        .unwrap_or_else(|| panic!("the dropped fact was counted but never named: {lines:#?}"));
    assert!(
        named.contains("the code it names is no longer declared here"),
        "the panel blames a symbol rename on the wrong kind of change: {named}"
    );
    assert!(
        std::fs::read_to_string(&state_path)
            .unwrap()
            .contains("validate_token rejects an empty token"),
        "a stale fact was deleted from state.md instead of kept out of one prompt"
    );

    // ── AB-2: a symbol check in a non-git tree reports unsearchable tree ───────
    let outside = std::env::temp_dir().join(format!(
        "xencode-unsearchable-notice-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let outside_xencode = outside.join(XENCODE_DIR);
    std::fs::create_dir_all(&outside_xencode).unwrap();
    std::fs::write(
        outside_xencode.join("state.md"),
        "# State\n\n## decisions\n- validate_token rejects an empty token [chk:validate_token]\n",
    )
    .unwrap();
    std::env::set_current_dir(&outside).unwrap();
    let unsearchable = context_panel(&mut app).await;
    let unsearchable_row = unsearchable
        .iter()
        .find(|line| line.contains("not checkable here"))
        .unwrap_or_else(|| {
            panic!("a fact in an unsearchable tree was not reported: {unsearchable:#?}")
        });
    assert!(
        unsearchable_row.contains("1 not checkable here (not a searchable tree)"),
        "the panel blamed a missing commit on an unsearchable tree: {unsearchable_row}"
    );
    // Leave the directory before deleting it: Windows will not remove the
    // process's current directory.
    std::env::set_current_dir(&root).unwrap();
    std::fs::remove_dir_all(&outside).unwrap();

    // ── AB-1: aged anchor proof reports warning in /ctx kv ────────────────────
    let aged_anchor_meta = xencode_context_rs::AnchorMeta {
        proved_at_unix_s: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - (21 * 86400),
        candidates: 2,
        verified: 2,
    };
    xencode_context_rs::write_anchor_meta(&root, &aged_anchor_meta).unwrap();
    let aged_panel = context_panel(&mut app).await;
    let anchor_notice = aged_panel
        .iter()
        .find(|line| line.contains("anchor proved"))
        .unwrap_or_else(|| panic!("aged anchor was not reported in /ctx kv: {aged_panel:#?}"));
    assert!(
        anchor_notice.contains("21 days ago")
            && anchor_notice.contains("run `xencode anchor` to re-check"),
        "anchor notice wording did not match: {anchor_notice}"
    );

    std::env::set_current_dir(previous).unwrap();
    std::fs::remove_dir_all(&root).unwrap();
}
