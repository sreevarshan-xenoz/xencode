//! End-to-end tests for competing candidate arms (`xencode compete`, AF-5).
//!
//! These run against a real git repository and the real `cargo fmt` toolchain —
//! nothing is stubbed:
//!
//! 1. Two arms on two branches, each verified, printed as an objective
//!    `{ran, skipped, failed, evidence-ref}` table with no composite score.
//! 2. A genuinely misformatted arm is reported as failed by the machine, while
//!    its well-formatted rival is reported as passed.
//! 3. The evidence log each row points at exists on disk.
//! 4. `list` and `show` read a recorded run back, in text and as JSON.
//! 5. `pick` checks the chosen branch out and leaves the rival branch plus both
//!    arms' evidence intact.
//! 6. Refusals: one arm, four arms, an edit aimed outside the worktree, an edit
//!    for an arm that was never declared, and a check name that isn't on the
//!    checklist.

use std::path::{Path, PathBuf};
use std::process::Output;

fn xencode(config_dir: &Path, repo: &Path, args: &[&str]) -> Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xencode"))
        .args(args)
        .current_dir(repo)
        .env("XCODE_CONFIG_DIR", config_dir)
        .output()
        .expect("xencode binary should execute")
}

fn git(repo: &Path, args: &[&str]) -> String {
    let out = std::process::Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .expect("git should execute");
    assert!(
        out.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-compete-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A real, committed Rust crate with a well-formatted `answer()` stub.
fn make_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = temp_dir(tag);
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(
        repo.join("Cargo.toml"),
        "[package]\nname = \"compete-e2e\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn answer() -> u32 {\n    0\n}\n",
    )
    .unwrap();
    git(&repo, &["init", "-q", "-b", "main", "."]);
    git(&repo, &["config", "user.name", "Compete E2E"]);
    git(
        &repo,
        &["config", "user.email", "compete-e2e@example.invalid"],
    );
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-qm", "base"]);
    (root, repo)
}

const GOOD_LIB: &str = "pub fn answer() -> u32 {\n    42\n}\n";
const MESSY_LIB: &str = "pub fn answer()->u32{42}\n";

/// `test` and `lint` are the two slots a workspace this size cannot answer in
/// reasonable time; `fmt` is left running because it is the check that proves
/// the arm was really verified rather than assumed.
const SKIP_SLOW: &[&str] = &["--skip", "test", "--skip", "lint"];

fn run_args<'a>(prompt: &'a str, arm_and_edits: &[&'a str]) -> Vec<&'a str> {
    let mut args: Vec<&str> = vec!["compete", "run", prompt];
    args.extend(arm_and_edits.iter().copied());
    args.extend(SKIP_SLOW.iter().copied());
    args
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

fn run_id_of(table: &str) -> String {
    table
        .lines()
        .find_map(|l| l.strip_prefix("Competing Arms Run: "))
        .unwrap_or_else(|| panic!("no run id in table:\n{table}"))
        .trim()
        .to_string()
}

/// The block of the printed table describing one arm.
fn arm_section(table: &str, arm_id: &str) -> String {
    let marker = format!("Arm: {arm_id} ");
    let start = table
        .find(&marker)
        .unwrap_or_else(|| panic!("no arm `{arm_id}` in table:\n{table}"));
    let rest = &table[start..];
    match rest[marker.len()..].find("\nArm: ") {
        Some(next) => rest[..marker.len() + next].to_string(),
        None => rest.split("\nWorktrees kept").next().unwrap().to_string(),
    }
}

/// One printed check row, split into its fields.
fn row(section: &str, check: &str) -> Vec<String> {
    section
        .lines()
        .find(|l| l.split_whitespace().next() == Some(check))
        .unwrap_or_else(|| panic!("no `{check}` row in section:\n{section}"))
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// The `ran,skipped,failed` triple of one printed row.
fn flags(section: &str, check: &str) -> String {
    let parts = row(section, check);
    assert_eq!(parts[0], check, "{parts:?}");
    parts[1..4].join(",")
}

fn branch_of(section: &str) -> String {
    section
        .lines()
        .find(|l| l.starts_with("Arm: "))
        .unwrap()
        .split('[')
        .nth(1)
        .unwrap()
        .split(':')
        .nth(1)
        .unwrap()
        .trim_end_matches(']')
        .trim()
        .to_string()
}

#[test]
fn compete_run_tables_both_arms_and_the_machine_decides_pass_from_fail() {
    let (root, repo) = make_repo("run");
    let cfg = root.join("config");
    std::fs::create_dir_all(&cfg).unwrap();

    let args = run_args(
        "Should answer() return 42 eagerly or lazily?",
        &[
            "--arm",
            "eager=Eager return",
            "--arm",
            "lazy=Lazy return",
            "--edit",
            "eager",
            "src/lib.rs",
            GOOD_LIB,
            "--edit",
            "lazy",
            "src/lib.rs",
            MESSY_LIB,
        ],
    );
    let out = xencode(&cfg, &repo, &args);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let table = stdout_of(&out);

    // The table's own header, one block per arm, both strategies named.
    assert!(
        table.contains("check   ran     skipped failed  evidence-ref"),
        "{table}"
    );
    let eager = arm_section(&table, "eager");
    let lazy = arm_section(&table, "lazy");
    assert!(eager.contains("Strategy: Eager return"), "{eager}");
    assert!(lazy.contains("Strategy: Lazy return"), "{lazy}");

    // fmt ran for both arms, and only the well-formatted arm got it.
    assert_eq!(flags(&eager, "fmt"), "true,false,false");
    assert_eq!(flags(&lazy, "fmt"), "true,false,true");
    let fmt_row = row(&eager, "fmt");
    assert_eq!(
        fmt_row.len(),
        5,
        "a ran check must point at its evidence: {fmt_row:?}"
    );
    assert!(fmt_row[4].ends_with("verify-fmt.log"), "{fmt_row:?}");

    // Skipped checks are named as skipped — never folded into a pass, and they
    // point at no evidence because they produced none.
    assert_eq!(flags(&eager, "lint"), "false,true,false");
    assert_eq!(flags(&eager, "test"), "false,true,false");
    assert_eq!(
        row(&eager, "lint").len(),
        4,
        "a skipped check has no evidence ref"
    );

    // No composite score, no grade, no winner, no ranking, no percentage.
    let lower = table.to_lowercase();
    for banned in ["score", "grade", "winner", "rank", "%"] {
        assert!(
            !lower.contains(banned),
            "table must not contain {banned:?}:\n{table}"
        );
    }

    // Both branches exist, and the evidence each row points at is real.
    let branches = git(&repo, &["branch", "--list"]);
    assert!(
        branches.contains("eager") && branches.contains("lazy"),
        "{branches}"
    );
    let run_id = run_id_of(&table);
    for arm in ["eager", "lazy"] {
        let log = repo
            .join(".xencode/artifacts")
            .join(format!("{run_id}-{arm}"))
            .join("verify-fmt.log");
        assert!(log.is_file(), "missing evidence log {}", log.display());
    }
    assert!(
        repo.join(".xencode/compete")
            .join(format!("{run_id}.json"))
            .is_file(),
        "the run must be recorded on disk"
    );

    // The failed arm's log holds rustfmt's own report, not just a verdict word.
    let messy_log = repo
        .join(".xencode/artifacts")
        .join(format!("{run_id}-lazy"))
        .join("verify-fmt.log");
    let messy_text = std::fs::read_to_string(&messy_log).unwrap();
    assert!(
        messy_text.contains("src/lib.rs"),
        "the fmt log must name the file: {messy_text}"
    );
}

#[test]
fn compete_run_json_stays_parseable_while_a_check_prints_a_diff() {
    // `cargo fmt --check` writes its diff to stdout. If the checklist let that
    // through, `--format json` would emit rustfmt's diff followed by JSON.
    let (root, repo) = make_repo("json-stdout");
    let cfg = root.join("config");
    std::fs::create_dir_all(&cfg).unwrap();

    let mut args = run_args(
        "Should answer() return 42 eagerly or lazily?",
        &[
            "--arm",
            "messy=Messy return",
            "--edit",
            "messy",
            "src/lib.rs",
            MESSY_LIB,
        ],
    );
    args.extend(
        ["--arm", "tidy=Tidy return", "--format", "json"]
            .iter()
            .copied(),
    );
    let out = xencode(&cfg, &repo, &args);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let doc: serde_json::Value =
        serde_json::from_str(&stdout_of(&out)).expect("run --format json is valid JSON");
    assert_eq!(doc["arms"].as_array().unwrap().len(), 2);
    assert_eq!(doc["picked_arm"], serde_json::Value::Null);
}

#[test]
fn compete_list_and_show_read_the_recorded_run_back() {
    let (root, repo) = make_repo("list-show");
    let cfg = root.join("config");
    std::fs::create_dir_all(&cfg).unwrap();

    let args = run_args(
        "Which return style?",
        &[
            "--arm",
            "a=First",
            "--arm",
            "b=Second",
            "--edit",
            "a",
            "src/lib.rs",
            GOOD_LIB,
            "--edit",
            "b",
            "src/lib.rs",
            GOOD_LIB,
        ],
    );
    let out = xencode(&cfg, &repo, &args);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let run_id = run_id_of(&stdout_of(&out));

    let list = xencode(&cfg, &repo, &["compete", "list"]);
    assert!(list.status.success());
    let listed = stdout_of(&list);
    assert!(
        listed.contains(&run_id) && listed.contains("Which return style?"),
        "{listed}"
    );

    let show = xencode(&cfg, &repo, &["compete", "show", &run_id]);
    assert!(show.status.success());
    let shown = stdout_of(&show);
    assert!(
        shown.contains("check   ran     skipped failed  evidence-ref"),
        "{shown}"
    );
    assert!(
        arm_section(&shown, "a").contains("Strategy: First"),
        "{shown}"
    );

    let json = xencode(
        &cfg,
        &repo,
        &["compete", "show", &run_id, "--format", "json"],
    );
    assert!(json.status.success());
    let doc: serde_json::Value =
        serde_json::from_str(&stdout_of(&json)).expect("show --format json is valid JSON");
    assert_eq!(doc["run_id"].as_str(), Some(run_id.as_str()));
    assert!(doc["picked_arm"].is_null(), "nothing is picked yet");
    let arms = doc["arms"].as_array().unwrap();
    assert_eq!(arms.len(), 2);
    for arm in arms {
        let checks = arm["checks"].as_array().unwrap();
        assert_eq!(checks.len(), 3, "fmt, lint and test each appear as a row");
        for entry in checks {
            assert!(entry["ran"].is_boolean());
            assert!(entry["skipped"].is_boolean());
            assert!(entry["failed"].is_boolean());
            assert!(entry["evidence_ref"].is_string());
            assert!(entry.get("score").is_none(), "no score field may exist");
        }
    }

    let missing = xencode(&cfg, &repo, &["compete", "show", "compete-does-not-exist"]);
    assert!(!missing.status.success());
    assert!(
        stderr_of(&missing).contains("not found"),
        "an unknown run must be refused by name: {}",
        stderr_of(&missing)
    );
}

#[test]
fn compete_pick_checks_out_the_winning_branch_and_keeps_the_rival() {
    let (root, repo) = make_repo("pick");
    let cfg = root.join("config");
    std::fs::create_dir_all(&cfg).unwrap();

    let args = run_args(
        "Loop or tail-call?",
        &[
            "--arm",
            "loop=Iterative loop",
            "--arm",
            "tail=Tail recursion",
            "--edit",
            "loop",
            "src/lib.rs",
            GOOD_LIB,
            "--edit",
            "tail",
            "src/lib.rs",
            "pub fn answer() -> u32 {\n    7\n}\n",
        ],
    );
    let out = xencode(&cfg, &repo, &args);
    assert!(out.status.success(), "stderr: {}", stderr_of(&out));
    let table = stdout_of(&out);
    let run_id = run_id_of(&table);
    let loop_branch = branch_of(&arm_section(&table, "loop"));
    let tail_branch = branch_of(&arm_section(&table, "tail"));

    let pick = xencode(&cfg, &repo, &["compete", "pick", &run_id, "loop"]);
    assert!(pick.status.success(), "stderr: {}", stderr_of(&pick));
    let picked = stdout_of(&pick);
    assert!(picked.contains("Checked out arm `loop`"), "{picked}");
    assert!(
        picked.contains(&tail_branch),
        "the rival branch must be named as preserved:\n{picked}"
    );

    // The repository now sits on the picked branch, holding the picked code.
    let head = git(&repo, &["rev-parse", "--abbrev-ref", "HEAD"]);
    assert_eq!(head.trim(), loop_branch);
    let lib = std::fs::read_to_string(repo.join("src/lib.rs")).unwrap();
    assert!(
        lib.contains("42"),
        "the picked arm's code must be in the tree: {lib}"
    );

    // The rival branch and both arms' evidence stay on disk.
    git(&repo, &["rev-parse", "--verify", &tail_branch]);
    for arm in ["loop", "tail"] {
        assert!(
            repo.join(".xencode/artifacts")
                .join(format!("{run_id}-{arm}"))
                .is_dir(),
            "the {arm} arm's evidence must survive the pick"
        );
    }

    // Picking an arm that is not in the run names the arms that are.
    let bad = xencode(&cfg, &repo, &["compete", "pick", &run_id, "third"]);
    assert!(!bad.status.success());
    assert!(
        stderr_of(&bad).contains("Available arms: loop, tail"),
        "{}",
        stderr_of(&bad)
    );
}

#[test]
fn compete_refuses_an_unbounded_field_or_an_edit_outside_the_worktree() {
    let (root, repo) = make_repo("refusals");
    let cfg = root.join("config");
    std::fs::create_dir_all(&cfg).unwrap();

    let one = xencode(
        &cfg,
        &repo,
        &run_args("Only one idea?", &["--arm", "solo=Lonely arm"]),
    );
    assert!(!one.status.success());
    assert!(
        stderr_of(&one).contains("2 or 3 candidate implementations"),
        "one arm must be refused: {}",
        stderr_of(&one)
    );

    let four = xencode(
        &cfg,
        &repo,
        &run_args(
            "Four is too many?",
            &["--arm", "a", "--arm", "b", "--arm", "c", "--arm", "d"],
        ),
    );
    assert!(!four.status.success());
    assert!(
        stderr_of(&four).contains("2 or 3 candidate implementations"),
        "four arms must be refused: {}",
        stderr_of(&four)
    );

    let dupe = xencode(
        &cfg,
        &repo,
        &run_args("Same name twice?", &["--arm", "a", "--arm", "a"]),
    );
    assert!(!dupe.status.success());
    assert!(
        stderr_of(&dupe).contains("arm id `a` given twice"),
        "a repeated arm id must be refused: {}",
        stderr_of(&dupe)
    );

    let escape = xencode(
        &cfg,
        &repo,
        &run_args(
            "Where does this file go?",
            &[
                "--arm",
                "a",
                "--arm",
                "b",
                "--edit",
                "a",
                "../outside.txt",
                "pwned",
            ],
        ),
    );
    assert!(!escape.status.success());
    assert!(
        stderr_of(&escape).contains("escapes the worktree"),
        "an edit above the worktree root must be refused: {}",
        stderr_of(&escape)
    );
    assert!(
        !root.join("outside.txt").exists(),
        "the refused write must not have landed anywhere"
    );

    let ghost = xencode(
        &cfg,
        &repo,
        &run_args(
            "Which arm is this for?",
            &[
                "--arm",
                "a",
                "--arm",
                "b",
                "--edit",
                "c",
                "src/lib.rs",
                "fn ghost() {}",
            ],
        ),
    );
    assert!(!ghost.status.success());
    assert!(
        stderr_of(&ghost).contains("known arms: a, b"),
        "an edit for an unknown arm must name the arms that exist: {}",
        stderr_of(&ghost)
    );

    let bad_skip = xencode(
        &cfg,
        &repo,
        &["compete", "run", "Skip what?", "--skip", "coverage"],
    );
    assert!(!bad_skip.status.success());
    assert!(
        stderr_of(&bad_skip).contains("the checklist is test, lint, fmt"),
        "only a real check name may be skipped: {}",
        stderr_of(&bad_skip)
    );

    // Not one of the refusals left a run recorded.
    let list = xencode(&cfg, &repo, &["compete", "list"]);
    assert!(list.status.success());
    assert!(
        stdout_of(&list).contains("No competing runs recorded"),
        "a refused run must not be recorded: {}",
        stdout_of(&list)
    );
}
