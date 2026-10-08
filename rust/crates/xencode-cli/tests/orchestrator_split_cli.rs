//! `OR-1` — `xencode orchestrator split`, end to end.
//!
//! The row's done-when is that a decomposition is *measured* before it is trusted,
//! and refused when it does not beat doing the work in one piece. So this test
//! builds a repository for real — a three-crate Cargo workspace whose crates
//! depend on each other, committed by git — and runs the command inside it. The
//! file set a split is scored against comes out of `git show`, and the orders come
//! out of `cargo metadata`, so nothing here is an opinion about which unit should
//! land first: the compiler is the one being asked, and a reader can run both
//! commands on the same directory.
//!
//! Every case uses `--answer`, which reads a split off disk. That is the half that
//! can be checked without spending anything: no model is asked, no provider is
//! reached, and no vendor's program starts. The refusal cases matter most, because
//! a split that is not scheduled is the only thing standing between a small
//! planner and two workers editing one file at once.

use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

fn xencode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xencode")
}

/// `repo/rust` holds three crates that really depend on one another — `middle` on
/// `leaf`, `top` on `middle` — and `repo` is a git repository with two commits: a
/// base one that writes the README, and a change that touches all three
/// `lib.rs` files. The workspace sits below the repository on purpose, because
/// that is how this repository is shaped and it is where git's names and cargo's
/// names part company.
fn repo() -> TempDir {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let ws = root.join("rust");
    fs::create_dir_all(&ws).unwrap();
    fs::write(
        ws.join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = [\"crates/leaf\", \"crates/middle\", \
         \"crates/top\"]\n",
    )
    .unwrap();
    for (name, dep) in [
        ("leaf", None),
        ("middle", Some("xleaf")),
        ("top", Some("xmiddle")),
    ] {
        let crate_dir = ws.join(format!("crates/{name}"));
        fs::create_dir_all(crate_dir.join("src")).unwrap();
        let manifest = match dep {
            None => {
                format!("[package]\nname = \"x{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n")
            }
            Some(dep) => format!(
                "[package]\nname = \"x{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
                 [dependencies]\n{dep} = {{ path = \"../{}\" }}\n",
                dep.trim_start_matches('x')
            ),
        };
        fs::write(crate_dir.join("Cargo.toml"), manifest).unwrap();
        fs::write(crate_dir.join("src/lib.rs"), "// empty\n").unwrap();
    }
    fs::write(root.join("README.md"), "# a repository\n").unwrap();
    git(root, &["init", "-q", "."]);
    git(root, &["config", "user.name", "tester"]);
    git(root, &["config", "user.email", "tester@example.invalid"]);
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "base"]);
    fs::write(ws.join("crates/leaf/src/lib.rs"), "// the record type\n").unwrap();
    fs::write(ws.join("crates/middle/src/lib.rs"), "// the store\n").unwrap();
    fs::write(ws.join("crates/top/src/lib.rs"), "// the command\n").unwrap();
    git(root, &["add", "-A"]);
    git(
        root,
        &[
            "commit",
            "-qm",
            "add a record type, its store and the command",
        ],
    );
    dir
}

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("git {args:?} failed to start: {e}"));
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// One unit of a split, as the planner is asked to write it.
fn unit(id: &str, path: &str, needs: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "goal": format!("{id} does its part"),
        "paths": [path],
        "needs": needs,
        "verify": "cargo check --offline --workspace",
    })
}

fn split(units: Vec<serde_json::Value>) -> String {
    serde_json::to_string_pretty(&serde_json::json!({
        "task": "add a record type, its store and the command",
        "subtasks": units,
    }))
    .unwrap()
}

/// Run the command with the temp repository's workspace as the working directory,
/// which is where it looks for the Cargo workspace to read crates from.
fn run(dir: &Path, args: &[&str]) -> Output {
    let out = Command::new(xencode_bin())
        .args(["orchestrator", "split"])
        .args(args)
        .current_dir(dir.join("rust"))
        .output()
        .expect("must run xencode orchestrator split");
    assert!(
        String::from_utf8_lossy(&out.stderr).lines().count() < 40,
        "a refusal should be readable, not a dump: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn text(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn error(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn a_split_that_states_the_orders_the_build_insists_on_is_accepted() {
    let dir = repo();
    let answer = dir.path().join("split.json");
    fs::write(
        &answer,
        split(vec![
            unit("leaf", "crates/leaf/src/lib.rs", &[]),
            unit("middle", "crates/middle/src/lib.rs", &["leaf"]),
            unit("top", "crates/top/src/lib.rs", &["middle"]),
        ]),
    )
    .unwrap();

    let out = run(
        dir.path(),
        &["--commit", "HEAD", "--answer", answer.to_str().unwrap()],
    );
    assert!(
        out.status.success(),
        "a correct split was refused: {}",
        error(&out)
    );
    let report = text(&out);
    // The reference is the commit, in words a reader can go and check.
    assert!(
        report.contains("commit ")
            && report.contains("add a record type, its store and the command"),
        "{report}"
    );
    // Three files, three orders, and the comparison the whole item is about: the
    // baseline owns the same files and states none of the orders.
    assert!(
        report.contains("files owned 3/3 vs 3/3, orders stated 3 vs 0"),
        "{report}"
    );
    assert!(
        report.contains("the scheduler may run this split"),
        "{report}"
    );
    assert!(
        report.contains("Nothing was launched or scheduled"),
        "{report}"
    );
}

#[test]
fn a_split_that_leaves_a_touched_file_unowned_is_refused_by_its_name() {
    // `top` is in the commit and in no unit, so nothing would write it. This is the
    // case a plan of work gets wrong most often and notices least.
    let dir = repo();
    let answer = dir.path().join("short.json");
    fs::write(
        &answer,
        split(vec![
            unit("leaf", "crates/leaf/src/lib.rs", &[]),
            unit("middle", "crates/middle/src/lib.rs", &["leaf"]),
        ]),
    )
    .unwrap();

    let out = run(
        dir.path(),
        &["--commit", "HEAD", "--answer", answer.to_str().unwrap()],
    );
    assert!(
        !out.status.success(),
        "an unowned file was scheduled: {}",
        text(&out)
    );
    let why = error(&out);
    assert!(why.contains("this split is not scheduled"), "{why}");
    assert!(
        why.contains("owned by no node") && why.contains("crates/top/src/lib.rs"),
        "{why}"
    );
}

#[test]
fn a_split_that_states_an_order_backwards_is_refused_and_named() {
    // The dangerous reading: `leaf` waits on `top`, which is the opposite of what
    // the dependency graph says. Run as scheduled, the store would be built before
    // the type it reads exists, and the failure would look like a worker's mistake.
    let dir = repo();
    let answer = dir.path().join("backwards.json");
    fs::write(
        &answer,
        split(vec![
            unit("top", "crates/top/src/lib.rs", &[]),
            unit("middle", "crates/middle/src/lib.rs", &["top"]),
            unit("leaf", "crates/leaf/src/lib.rs", &["middle"]),
        ]),
    )
    .unwrap();

    let out = run(
        dir.path(),
        &["--commit", "HEAD", "--answer", answer.to_str().unwrap()],
    );
    assert!(
        !out.status.success(),
        "a backwards graph was scheduled: {}",
        text(&out)
    );
    let why = error(&out);
    assert!(why.contains("stated backwards"), "{why}");
    assert!(
        text(&out).contains("3 backwards, 0 left silent"),
        "{}",
        text(&out)
    );
}

#[test]
fn an_order_left_unstated_is_a_refusal_rather_than_a_neither_of_these() {
    // `top` waits on `leaf` only, so the scheduler would start `middle` and `top`
    // together and `top` would read a store that is not written yet. Counting that
    // as neutral is the mistake this rule exists to prevent: the split states 2 of
    // 3 orders and the baseline states 0, so the numbers alone would let it through.
    let dir = repo();
    let answer = dir.path().join("silent.json");
    fs::write(
        &answer,
        split(vec![
            unit("leaf", "crates/leaf/src/lib.rs", &[]),
            unit("middle", "crates/middle/src/lib.rs", &["leaf"]),
            unit("top", "crates/top/src/lib.rs", &["leaf"]),
        ]),
    )
    .unwrap();

    let out = run(
        dir.path(),
        &["--commit", "HEAD", "--answer", answer.to_str().unwrap()],
    );
    assert!(
        !out.status.success(),
        "a missing edge was scheduled: {}",
        text(&out)
    );
    let why = error(&out);
    assert!(why.contains("left silent"), "{why}");
    assert!(why.contains("a missing edge is not caution"), "{why}");
    let report = text(&out);
    assert!(report.contains("orders stated 2/3"), "{report}");
    assert!(
        report.contains("`crates/middle/src/lib.rs` before `crates/top/src/lib.rs`"),
        "{report}"
    );
}

#[test]
fn the_two_formats_refuse_the_same_split_for_the_same_reason() {
    let dir = repo();
    let answer = dir.path().join("short.json");
    fs::write(
        &answer,
        split(vec![unit("leaf", "crates/leaf/src/lib.rs", &[])]),
    )
    .unwrap();
    let json_out = run(
        dir.path(),
        &[
            "--commit",
            "HEAD",
            "--answer",
            answer.to_str().unwrap(),
            "--format",
            "json",
        ],
    );
    let text_out = run(
        dir.path(),
        &["--commit", "HEAD", "--answer", answer.to_str().unwrap()],
    );
    assert!(!json_out.status.success() && !text_out.status.success());
    let value: serde_json::Value =
        serde_json::from_str(&text(&json_out)).expect("the json report parses");
    assert_eq!(value["schedulable"], serde_json::json!(false));
    assert_eq!(value["reference"]["paths"].as_array().unwrap().len(), 3);
    assert_eq!(value["score"]["files_expected"].as_u64().unwrap(), 3);
    assert_eq!(value["score"]["files_owned"].as_u64().unwrap(), 1);
    assert_eq!(value["baseline"]["orders_stated"].as_u64().unwrap(), 0);
    assert_eq!(value["score"]["orders_stated"].as_u64().unwrap(), 0);
    // Whatever the format, the figures a reader is given are the same figures.
    let report = text(&text_out);
    assert!(
        report.contains("files owned 1/3 vs 3/3, orders stated 0 vs 0"),
        "{report}"
    );
    let reasons = value["reasons"].as_array().unwrap();
    assert!(!reasons.is_empty());
    for reason in reasons {
        assert!(
            report.contains(reason.as_str().unwrap()),
            "a reason in the json report is missing from the text one: {reason}"
        );
    }
    // The prompt and the model's words are in the json only when a model was asked;
    // reading a file spent nothing and must not pretend otherwise.
    assert!(value["planner"].is_null(), "{}", value["planner"]);
}

#[test]
fn a_split_compared_only_with_itself_says_which_half_was_measured() {
    // No commit and no file list: the reference's paths are the split's own, so
    // coverage can only ever come out full and must not be quoted as a result. The
    // orders are still cargo's, which is why the split can be accepted at all.
    let dir = repo();
    let answer = dir.path().join("chain.json");
    fs::write(
        &answer,
        split(vec![
            unit("leaf", "crates/leaf/src/lib.rs", &[]),
            unit("middle", "crates/middle/src/lib.rs", &["leaf"]),
        ]),
    )
    .unwrap();

    let out = run(
        dir.path(),
        &[
            "--task",
            "add a record type, its store and the command",
            "--answer",
            answer.to_str().unwrap(),
        ],
    );
    assert!(out.status.success(), "{}", error(&out));
    let report = text(&out);
    assert!(
        report.contains("files owned 2/2 vs 2/2, orders stated 1 vs 0"),
        "{report}"
    );
    assert!(
        report.contains("Only the ordering above is a measurement"),
        "{report}"
    );
    assert!(report.contains("--commit <sha> or --path"), "{report}");
}

#[test]
fn a_task_that_names_no_change_is_refused_before_anything_is_asked() {
    // The command must not ask a model, and so must not spend, to discover it has
    // nothing to measure against.
    let dir = repo();
    let out = run(dir.path(), &[]);
    assert!(!out.status.success());
    assert!(error(&out).contains("nothing to split"), "{}", error(&out));
    assert!(text(&out).is_empty(), "{}", text(&out));
}

#[test]
fn a_split_written_in_a_field_the_shape_does_not_declare_is_refused_not_guessed_at() {
    // The failure this guards: a unit that wrote `depends_on` instead of `needs`
    // would otherwise read as a unit that waits for nothing, and the scheduler
    // would start it beside the file it reads. The file is refused by name.
    let dir = repo();
    let answer = dir.path().join("misshapen.json");
    fs::write(
        &answer,
        serde_json::to_string_pretty(&serde_json::json!({
            "task": "add a record type, its store and the command",
            "subtasks": [
                {"id": "leaf", "goal": "the record", "paths": ["crates/leaf/src/lib.rs"],
                 "needs": [], "verify": "cargo check --offline --workspace"},
                {"id": "middle", "goal": "the store", "paths": ["crates/middle/src/lib.rs"],
                 "depends_on": ["leaf"], "verify": "cargo check --offline --workspace"},
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let out = run(
        dir.path(),
        &["--commit", "HEAD", "--answer", answer.to_str().unwrap()],
    );
    assert!(
        !out.status.success(),
        "a mis-shapen answer was scored: {}",
        text(&out)
    );
    let why = error(&out);
    assert!(why.contains("depends_on"), "{why}");
    assert!(why.contains("does not declare"), "{why}");
}

#[test]
fn a_named_path_outside_the_workspace_still_has_to_be_owned() {
    // The change under test is the one commit that touched the README as well, so
    // the reference has four files and three orders — and a split that says nothing
    // about the README is short of the change even though every order it states is
    // right. The build cannot say when a README lands; a person can.
    let dir = repo();
    let root = dir.path();
    fs::write(root.join("README.md"), "# a repository, updated\n").unwrap();
    fs::write(
        root.join("rust/crates/leaf/src/lib.rs"),
        "// the record type, with one more field\n",
    )
    .unwrap();
    git(root, &["add", "-A"]);
    git(root, &["commit", "-qm", "the same change, with the manual"]);
    let answer = root.join("rust-only.json");
    fs::write(
        &answer,
        split(vec![unit("leaf", "crates/leaf/src/lib.rs", &[])]),
    )
    .unwrap();

    let out = run(
        root,
        &["--commit", "HEAD", "--answer", answer.to_str().unwrap()],
    );
    assert!(!out.status.success(), "{}", text(&out));
    let report = text(&out);
    assert!(report.contains("files owned 1/2 vs 2/2"), "{report}");
    assert!(error(&out).contains("README.md"), "{}", error(&out));
}

#[test]
fn a_split_that_also_plans_a_file_the_change_never_touched_is_refused() {
    // Both local planner runs measured on this repository did exactly this: one
    // read a symptom out of the task's prose and wrote a module for it, the other
    // named a crate that is not here. Covering every file of the change and stating
    // every real order is not enough when a worker is also handed a file nobody
    // decided should change — here `README.md`, which exists and is untouched by
    // the commit being scored.
    let dir = repo();
    let answer = dir.path().join("split.json");
    fs::write(
        &answer,
        split(vec![
            unit("leaf", "crates/leaf/src/lib.rs", &[]),
            unit("middle", "crates/middle/src/lib.rs", &["leaf"]),
            unit("top", "crates/top/src/lib.rs", &["middle"]),
            unit("loose-end", "README.md", &[]),
        ]),
    )
    .unwrap();

    let out = run(
        dir.path(),
        &["--commit", "HEAD", "--answer", answer.to_str().unwrap()],
    );
    assert!(!out.status.success(), "{}", text(&out));
    let report = text(&out);
    // The file set is covered and every real order is stated — the baseline is the
    // one that orders nothing — so this split would be accepted if not for the one
    // file it adds.
    assert!(report.contains("files owned 3/3"), "{report}");
    assert!(report.contains("orders stated 3 vs 0"), "{report}");
    let why = error(&out);
    assert!(
        why.contains("does not consist of") && why.contains("`README.md`"),
        "{why}"
    );
}
