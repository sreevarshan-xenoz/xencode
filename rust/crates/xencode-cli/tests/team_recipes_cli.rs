//! `OR-9` — a team recipe is a file a person can read and diff, and removing it
//! removes nothing else. Every case here runs the real `xencode` binary in a real
//! directory and reads real files; nothing is simulated.

use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn xencode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xencode")
}

/// The three-role shape the plan item describes: two branch heads and a join.
fn recipe(name: &str) -> String {
    format!(
        r#"name = "{name}"

[[roles]]
name = "survey"
worker = "opencode"
gate = []
command = "true"

[[roles]]
name = "harden"
worker = "claude"
gate = ["test"]
command = "true"

[[roles]]
name = "integrate"
worker = "opencode"
gate = ["lint", "test"]
command = "true"
needs = ["survey", "harden"]

[capacity]
workers = 2
verification_throughput = 2
"#
    )
}

/// A project directory whose `.xencode/teams` holds exactly these files. The
/// caller keeps the `TempDir` alive, so nothing here leaks or is cleaned early.
fn project(files: &[(&str, String)]) -> TempDir {
    let dir = TempDir::new().unwrap();
    let teams = dir.path().join(".xencode").join("teams");
    fs::create_dir_all(&teams).unwrap();
    for (name, text) in files {
        fs::write(teams.join(name), text).unwrap();
    }
    dir
}

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(xencode_bin())
        .args(args)
        .current_dir(dir)
        .output()
        .expect("xencode ran")
}

fn succeeds(dir: &Path, args: &[&str]) -> String {
    let out = run(dir, args);
    assert!(
        out.status.success(),
        "`xencode {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

#[test]
fn no_recipes_directory_at_all_is_a_normal_answer_and_not_a_failure() {
    let dir = TempDir::new().unwrap();
    let out = run(dir.path(), &["team", "list"]);
    assert!(
        out.status.success(),
        "a project with no team recipes is not an error: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let said = String::from_utf8_lossy(&out.stdout);
    assert!(said.contains("No team recipes in"), "{said}");
    assert!(
        said.contains(".xencode/teams"),
        "says where one goes: {said}"
    );
    assert!(
        !dir.path().join(".xencode").exists(),
        "listing must not create the directory it reads"
    );
}

#[test]
fn removing_one_recipe_leaves_the_other_reading_exactly_as_it_did() {
    let dir = project(&[
        ("rust-fix.toml", recipe("rust-fix")),
        ("docs.toml", recipe("docs-pass")),
    ]);
    let before = succeeds(dir.path(), &["team", "show", "docs-pass"]);
    assert!(before.contains("docs-pass  —  3 roles"), "{before}");

    fs::remove_file(
        dir.path()
            .join(".xencode")
            .join("teams")
            .join("rust-fix.toml"),
    )
    .unwrap();

    let after = succeeds(dir.path(), &["team", "show", "docs-pass"]);
    assert_eq!(
        before, after,
        "deleting an unrelated recipe changed how this one reads"
    );
    let gone = run(dir.path(), &["team", "show", "rust-fix"]);
    assert!(!gone.status.success(), "the deleted recipe still resolves");
    let listed = succeeds(dir.path(), &["team", "list"]);
    assert!(listed.contains("docs-pass"), "{listed}");
    assert!(!listed.contains("rust-fix"), "{listed}");
}

#[test]
fn one_unreadable_recipe_is_reported_beside_the_good_one_instead_of_replacing_it() {
    let dir = project(&[
        ("rust-fix.toml", recipe("rust-fix")),
        ("half-written.toml", "name = = 3\n".to_string()),
    ]);
    let listed = succeeds(dir.path(), &["team", "list"]);
    assert!(listed.contains("rust-fix"), "{listed}");
    assert!(
        listed.contains("half-written.toml is not a readable team recipe"),
        "the bad file says which one it is: {listed}"
    );
}

#[test]
fn a_gate_naming_a_check_that_does_not_exist_is_refused_with_the_three_real_checks() {
    let bad = recipe("bench-team").replace("gate = [\"lint\", \"test\"]", "gate = [\"bench\"]");
    let dir = project(&[("bench.toml", bad)]);
    let out = run(dir.path(), &["team", "plan", "bench-team"]);
    assert!(
        !out.status.success(),
        "an invented gate would be a gate nobody runs"
    );
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("bench"), "names what was asked for: {said}");
    assert!(
        said.contains("fmt, lint, test"),
        "names the real set: {said}"
    );
    assert!(said.contains("integrate"), "names the role: {said}");
    // Listing says the same thing rather than pretending the file is a team.
    let listed = succeeds(dir.path(), &["team", "list"]);
    assert!(listed.contains("gate `bench` is not a check"), "{listed}");
}

/// `xencode team` never runs a role's own command. The gates a recipe names are
/// printed, not executed — `xencode verify` is the command that runs the three
/// checks — and this covers every subcommand that reads a recipe.
#[test]
fn planning_a_recipe_never_runs_a_role_command() {
    // The join's command would leave a file behind if anything ever executed it.
    let planned = recipe("touch-team").replace(
        "command = \"true\"\nneeds = [\"survey\", \"harden\"]",
        "command = \"touch SHOULD-NOT-EXIST\"\nneeds = [\"survey\", \"harden\"]",
    );
    assert!(
        planned.contains("touch "),
        "the role really does carry a command"
    );
    let dir = project(&[("touch.toml", planned)]);
    let out = succeeds(dir.path(), &["team", "plan", "touch-team"]);
    assert!(out.contains("touch "), "the plan shows the command: {out}");
    assert!(out.contains("Nothing was launched"), "and says so: {out}");
    for args in [
        &["team", "list"][..],
        &["team", "show", "touch-team"],
        &["team", "plan", "touch-team", "--format", "json"],
    ] {
        succeeds(dir.path(), args);
    }
    assert!(
        !dir.path().join("SHOULD-NOT-EXIST").exists(),
        "`xencode team` executed a role's command"
    );
}

#[test]
fn the_plan_reports_the_wave_order_the_critical_path_and_what_limits_them() {
    let dir = project(&[("rust-fix.toml", recipe("rust-fix"))]);
    let out = succeeds(dir.path(), &["team", "plan", "rust-fix"]);
    let survey = out.find("wave 1  survey").expect("survey is in wave 1");
    let harden = out.find("wave 1  harden").expect("harden is in wave 1");
    let integrate = out.find("wave 2  integrate").expect("the join waits");
    assert!(survey < harden && harden < integrate, "{out}");
    assert!(out.contains("survey → integrate"), "critical path: {out}");
    assert!(out.contains("serial bottleneck: integrate"), "{out}");
    assert!(out.contains("at most 2 at once"), "{out}");
    assert!(
        out.contains("gate:     none — no check gates this role"),
        "an ungated role is reported as ungated, not as passing: {out}"
    );

    // The same numbers as data, for a caller that reads them rather than looks.
    let json: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        &["team", "plan", "rust-fix", "--format", "json"],
    ))
    .unwrap();
    assert_eq!(json["capacity"], 2);
    assert_eq!(json["launches"], false);
    assert_eq!(json["waves"][0], serde_json::json!(["survey", "harden"]));
    assert_eq!(json["bottleneck"], "integrate");
}

#[test]
fn two_files_claiming_one_recipe_name_are_refused_by_path() {
    let dir = project(&[("a.toml", recipe("same")), ("b.toml", recipe("same"))]);
    let out = run(dir.path(), &["team", "plan", "same"]);
    assert!(!out.status.success(), "one name must not be two teams");
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("a.toml") && said.contains("b.toml"), "{said}");
}

#[test]
fn only_dot_toml_files_are_read_as_recipes() {
    let dir = project(&[("rust-fix.toml", recipe("rust-fix"))]);
    let teams = dir.path().join(".xencode").join("teams");
    fs::write(teams.join("notes.md"), "# not a recipe at all\n").unwrap();
    fs::write(teams.join("rust-fix.toml.bak"), "garbage = = =\n").unwrap();
    let listed = succeeds(dir.path(), &["team", "list"]);
    assert!(
        !listed.contains("notes.md") && !listed.contains(".bak"),
        "only *.toml is a recipe: {listed}"
    );
    assert!(listed.contains("rust-fix"), "{listed}");
}

#[test]
fn a_capacity_of_zero_is_refused_rather_than_reported_as_a_team_of_one() {
    let zero = recipe("idle-team").replace("workers = 2", "workers = 0");
    let dir = project(&[("idle.toml", zero)]);
    let listed = succeeds(dir.path(), &["team", "list"]);
    assert!(
        listed.contains("workers is 0, so no role could ever start"),
        "{listed}"
    );
    let out = run(dir.path(), &["team", "plan", "idle-team"]);
    assert!(
        !out.status.success(),
        "a team that can start nothing must not be plannable"
    );
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("idle-team"), "names the recipe: {said}");
    assert!(said.contains("workers is 0"), "{said}");
}
