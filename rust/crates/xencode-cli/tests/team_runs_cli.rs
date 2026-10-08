//! `OR-10` — a team launches only under a name, and the number a plan quotes
//! comes from a run that actually happened. Every case here runs the real
//! `xencode` binary in a real directory, waits on real `sh -c` children, and reads
//! the record file back off disk.

use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

/// Whether this machine has the CPU energy counter a run's power figure comes
/// from. GitHub's virtual machines expose none, so a figure is not expected there.
fn power_counter_readable() -> bool {
    xencode_context_rs::power::package_energy_uj(xencode_context_rs::power::POWERCAP_ROOT).is_some()
}

fn xencode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xencode")
}

/// Two branch heads and a join. The heads are slow enough that a queue which ran
/// them one after the other would be visible in the recorded timings.
fn recipe(name: &str) -> String {
    format!(
        r#"name = "{name}"

[[roles]]
name = "survey"
worker = "opencode"
gate = []
command = "sleep 0.3"

[[roles]]
name = "harden"
worker = "claude"
gate = ["test"]
command = "sleep 0.3"

[[roles]]
name = "integrate"
worker = "opencode"
gate = ["lint", "test"]
command = "sleep 0.1"
needs = ["survey", "harden"]

[capacity]
workers = 2
verification_throughput = 2
"#
    )
}

/// A project whose `.xencode/teams` holds exactly these files. The caller keeps
/// the `TempDir` alive, so nothing here is cleaned up early or leaks.
fn project(files: &[(&str, String)]) -> TempDir {
    let dir = TempDir::new().unwrap();
    let teams = dir.path().join(".xencode").join("teams");
    fs::create_dir_all(&teams).unwrap();
    for (name, text) in files {
        fs::write(teams.join(name), text).unwrap();
    }
    dir
}

/// A home directory of its own, so the tariff a run is priced at is whatever this
/// test set rather than whatever the person running it has in their settings. The
/// tariff is written with `xencode config set`, the same way a person sets it.
///
/// This file is about a team that really launches, and the shipped posture refuses
/// a roster agent by name before anything is launched (`OR-13`), so the home it
/// builds opens that one rule with the same command a person would. The refusal
/// itself is what `the_shipped_posture_refuses_a_run_and_records_nothing` checks,
/// in a home that leaves the rule as the product installed it.
fn sandbox_home(tariff: Option<&str>) -> TempDir {
    let home = TempDir::new().unwrap();
    for (key, value) in [
        ("allow_external_workers", "true"),
        ("power_cents_per_kwh", tariff.unwrap_or("")),
    ] {
        if value.is_empty() {
            continue;
        }
        let out = Command::new(xencode_bin())
            .args(["config", "set", key, value])
            .current_dir(home.path())
            .env("HOME", home.path())
            .env("XDG_CONFIG_HOME", home.path().join(".config"))
            .env(
                "XCODE_CONFIG_DIR",
                home.path().join(".config").join("xencode"),
            )
            .output()
            .expect("xencode ran");
        assert!(
            out.status.success(),
            "setting {key} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    home
}

fn run_in(dir: &Path, home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(xencode_bin())
        .args(args)
        .current_dir(dir)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XCODE_CONFIG_DIR", home.join(".config").join("xencode"))
        .output()
        .expect("xencode ran")
}

/// A command that has to succeed; its stdout, so a failing case prints what the
/// binary said instead of only its exit code.
fn succeeds(dir: &Path, home: &Path, args: &[&str]) -> String {
    let out = run_in(dir, home, args);
    assert!(
        out.status.success(),
        "`xencode {}` failed: {}{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

/// Every record the project's `.xencode/team-runs` holds, read back off disk.
fn records(dir: &Path) -> Vec<serde_json::Value> {
    let runs = dir.join(".xencode").join("team-runs");
    let mut found: Vec<serde_json::Value> = Vec::new();
    for entry in
        fs::read_dir(&runs).unwrap_or_else(|e| panic!("{} is not there: {e}", runs.display()))
    {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        found.push(
            serde_json::from_str(&fs::read_to_string(&path).unwrap())
                .unwrap_or_else(|e| panic!("{} is not a readable record: {e}", path.display())),
        );
    }
    found.sort_by_key(|r| r["started_at_unix_ms"].as_u64().unwrap_or(0));
    found
}

fn assert_no_history(dir: &Path, what_ran: &str) {
    assert!(
        !dir.join(".xencode").join("team-runs").exists(),
        "`xencode {what_ran}` created a history directory"
    );
}

#[test]
fn asking_to_run_a_recipe_launches_nothing_and_writes_no_history() {
    // The plan view is the default, so "no changes will be made" has to be true
    // of it: no role child, no record file, no runs directory.
    let text = recipe("touch-team").replace("sleep 0.3", "touch SHOULD-NOT-EXIST");
    assert!(
        text.contains("touch "),
        "the role really does carry a command"
    );
    let dir = project(&[("quick.toml", text)]);
    let home = sandbox_home(None);

    let out = succeeds(dir.path(), home.path(), &["team", "run", "touch-team"]);
    assert!(out.contains("Nothing was launched"), "says so: {out}");
    assert!(
        out.contains("Nothing above has been launched, and nothing will be"),
        "and says what would change that: {out}"
    );
    assert!(
        out.contains("command:  touch SHOULD-NOT-EXIST"),
        "shows the command it did not run: {out}"
    );
    assert!(
        !dir.path().join("SHOULD-NOT-EXIST").exists(),
        "an unapproved run executed a role's command"
    );
    assert_no_history(dir.path(), "team run touch-team");

    let json = succeeds(
        dir.path(),
        home.path(),
        &["team", "run", "touch-team", "--format", "json"],
    );
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["launches"], serde_json::json!(false));
    assert_eq!(parsed["approval_required"], serde_json::json!(true));
    assert!(parsed["estimate"].is_null(), "no run yet, so no estimate");
    assert!(
        !dir.path().join("SHOULD-NOT-EXIST").exists(),
        "the JSON form executed a role either"
    );
    assert_no_history(dir.path(), "team run --format json");
}

#[test]
fn an_empty_approval_is_refused_because_a_run_needs_a_name_on_it() {
    let dir = project(&[("quick.toml", recipe("quick"))]);
    let home = sandbox_home(None);
    let out = run_in(
        dir.path(),
        home.path(),
        &["team", "run", "quick", "--approved-by", ""],
    );
    assert!(!out.status.success(), "a blank name approved a run");
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("--approved-by"), "{said}");
    assert!(said.contains("nobody"), "says why: {said}");
    assert_no_history(dir.path(), "team run --approved-by \"\"");
}

#[test]
fn an_approved_run_launches_the_roles_and_records_what_it_took() {
    let text = recipe("quick").replace("sleep 0.1", "touch DID-RUN");
    let dir = project(&[("quick.toml", text)]);
    let home = sandbox_home(None);
    let out = run_in(
        dir.path(),
        home.path(),
        &["team", "run", "quick", "--approved-by", "Sree"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        dir.path().join("DID-RUN").exists(),
        "the join's command never ran as a child"
    );
    assert!(
        stdout.contains("Launching now, approved by Sree"),
        "{stdout}"
    );
    assert!(stdout.contains("survey         exited(0)"), "{stdout}");
    if power_counter_readable() {
        assert!(
            stdout.contains("no $/kWh set"),
            "an unpriced run must not read as a free one: {stdout}"
        );
    }

    let found = records(dir.path());
    assert_eq!(found.len(), 1, "one run, one record");
    let record = &found[0];
    assert_eq!(record["approved_by"], serde_json::json!("Sree"));
    assert_eq!(record["recipe"], serde_json::json!("quick"));
    assert_eq!(
        record["launch_order"],
        serde_json::json!(["survey", "harden", "integrate"])
    );
    let elapsed = record["elapsed_ms"].as_u64().unwrap();
    assert!(
        (300..60_000).contains(&elapsed),
        "the recorded wall clock was {elapsed} ms, which is not two sleeps and a join"
    );
    assert_eq!(record["peak_concurrency"], serde_json::json!(2));
    assert!(
        record["fingerprint"]
            .as_str()
            .is_some_and(|f| f.len() == 16),
        "the record carries no fingerprint to estimate from: {record}"
    );
    assert!(
        record["cents_per_kwh"].is_null(),
        "no tariff was set, so the record must not claim one"
    );
}

#[test]
fn two_roles_that_do_not_need_each_other_really_ran_at_the_same_time() {
    let dir = project(&[("quick.toml", recipe("quick"))]);
    let home = sandbox_home(None);
    succeeds(
        dir.path(),
        home.path(),
        &["team", "run", "quick", "--approved-by", "Sree"],
    );
    let record = &records(dir.path())[0];
    let roles = record["roles"].as_array().unwrap();
    let survey = roles.iter().find(|r| r["name"] == "survey").unwrap();
    let harden = roles.iter().find(|r| r["name"] == "harden").unwrap();
    // Overlap, read off the times the queue itself recorded: one head was still
    // running when the other was launched. A serial run would put harden's start
    // at or after survey's finish.
    assert!(
        harden["started_ms"].as_u64().unwrap() < survey["finished_ms"].as_u64().unwrap(),
        "the two branch heads did not overlap: {survey} {harden}"
    );
    assert_eq!(
        record["peak_concurrency"],
        serde_json::json!(2),
        "the queue never observed two roles in flight"
    );
}

#[test]
fn a_role_that_fails_fails_the_run_and_is_recorded_as_the_code_the_os_reported() {
    let text = recipe("quick").replace("command = \"sleep 0.1\"", "command = \"exit 3\"");
    let dir = project(&[("quick.toml", text)]);
    let home = sandbox_home(None);
    let out = run_in(
        dir.path(),
        home.path(),
        &["team", "run", "quick", "--approved-by", "Sree"],
    );
    assert!(
        !out.status.success(),
        "a run whose role exited 3 reported success"
    );
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("integrate"), "names the role: {said}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("integrate      exited(3)"), "{stdout}");
    // The failure is what the next plan measures against, so it is recorded
    // rather than thrown away.
    let record = &records(dir.path())[0];
    let integrate = record["roles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "integrate")
        .unwrap();
    assert_eq!(integrate["status"], serde_json::json!("exited(3)"));
}

#[test]
fn the_first_plan_has_no_estimate_and_a_run_gives_the_next_one_one() {
    let dir = project(&[("quick.toml", recipe("quick"))]);
    let home = sandbox_home(None);
    let first = succeeds(dir.path(), home.path(), &["team", "plan", "quick"]);
    assert!(
        first.contains("estimated wall clock: unknown"),
        "a recipe that never ran was given a number anyway: {first}"
    );
    assert!(first.contains("never run here"), "{first}");
    assert_no_history(dir.path(), "team plan quick");

    succeeds(
        dir.path(),
        home.path(),
        &["team", "run", "quick", "--approved-by", "Sree"],
    );
    let second = succeeds(dir.path(), home.path(), &["team", "plan", "quick"]);
    assert!(
        second.contains("estimated wall clock:") && !second.contains("wall clock: unknown"),
        "the plan still quotes nothing after a run: {second}"
    );
    assert!(second.contains("measured from run quick-"), "{second}");
    assert!(second.contains("approved by Sree"), "{second}");
    assert!(
        second.contains("not a promise"),
        "an estimate has to say what it is: {second}"
    );
}

#[test]
fn an_approved_run_checks_its_estimate_after_it_ran() {
    let dir = project(&[("quick.toml", recipe("quick"))]);
    let home = sandbox_home(None);
    let mut last = Vec::new();
    for _ in 0..2 {
        let out = run_in(
            dir.path(),
            home.path(),
            &["team", "run", "quick", "--approved-by", "Sree"],
        );
        assert!(
            out.status.success(),
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        last = out.stdout;
    }
    let said = String::from_utf8_lossy(&last);
    // The second run is the one that had something to check itself against.
    assert!(said.contains("the estimate was"), "{said}");
    assert!(said.contains("checked:"), "{said}");
    assert!(said.contains("ms against an estimated"), "{said}");
    assert_eq!(
        records(dir.path()).len(),
        2,
        "a run that was checked left no record of itself"
    );
}

#[test]
fn editing_a_role_command_retires_the_estimate_because_it_measured_a_different_team() {
    let dir = project(&[("quick.toml", recipe("quick"))]);
    let home = sandbox_home(None);
    succeeds(
        dir.path(),
        home.path(),
        &["team", "run", "quick", "--approved-by", "Sree"],
    );
    let recorded = records(dir.path())[0].clone();
    let planned = succeeds(
        dir.path(),
        home.path(),
        &["team", "plan", "quick", "--format", "json"],
    );
    let before: serde_json::Value = serde_json::from_str(&planned).unwrap();
    assert_eq!(before["fingerprint"], recorded["fingerprint"]);
    assert!(
        before["estimate"].is_object(),
        "the plan of an unedited recipe quotes no run: {before}"
    );

    // One command changed: the recorded run is now a measurement of a team that
    // no longer exists.
    let path = dir.path().join(".xencode").join("teams").join("quick.toml");
    fs::write(&path, recipe("quick").replace("sleep 0.3", "sleep 0.35")).unwrap();
    let after: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["team", "plan", "quick", "--format", "json"],
    ))
    .unwrap();
    assert_ne!(
        after["fingerprint"], recorded["fingerprint"],
        "an edited recipe kept the identity of the one that ran"
    );
    assert!(
        after["estimate"].is_null(),
        "an estimate survived an edited command: {after}"
    );
    let said = succeeds(dir.path(), home.path(), &["team", "plan", "quick"]);
    assert!(
        said.contains("estimated wall clock: unknown"),
        "the words a person reads disagree with the JSON: {said}"
    );
}

#[test]
fn a_recipe_copied_into_another_project_brings_no_estimate_with_it() {
    let text = recipe("quick");
    let first = project(&[("quick.toml", text.clone())]);
    let home = sandbox_home(None);
    succeeds(
        first.path(),
        home.path(),
        &["team", "run", "quick", "--approved-by", "Sree"],
    );

    let second = project(&[("quick.toml", text)]);
    let said = succeeds(second.path(), home.path(), &["team", "plan", "quick"]);
    assert!(
        said.contains("estimated wall clock: unknown"),
        "a measured run followed the file into a project that never had it: {said}"
    );
    assert_no_history(second.path(), "team plan quick");
}

#[test]
fn a_recipe_that_cannot_schedule_is_refused_before_anything_launches() {
    let broken = recipe("quick").replace(
        "needs = [\"survey\", \"harden\"]",
        "needs = [\"survey\", \"ghost\"]",
    );
    let dir = project(&[("quick.toml", broken)]);
    let home = sandbox_home(None);
    let out = run_in(
        dir.path(),
        home.path(),
        &["team", "run", "quick", "--approved-by", "Sree"],
    );
    assert!(
        !out.status.success(),
        "a recipe naming a role that does not exist ran"
    );
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("ghost"), "names what is missing: {said}");
    assert!(
        said.contains("nothing was launched"),
        "says what did not happen: {said}"
    );
    assert_no_history(dir.path(), "team run on a broken recipe");
}

#[test]
fn a_set_tariff_prices_the_run_and_the_estimate_at_the_rate_the_person_wrote() {
    // Without a tariff the same watt-hours stay unpriced rather than becoming
    // free, which the approved-run case above asserts. Here the number a person
    // set has to reach both the record and the estimate built from it.
    let dir = project(&[("quick.toml", recipe("quick"))]);
    let home = sandbox_home(Some("40"));
    let out = run_in(
        dir.path(),
        home.path(),
        &["team", "run", "quick", "--approved-by", "Sree"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!stdout.contains("no $/kWh set"), "{stdout}");

    let record = &records(dir.path())[0];
    assert_eq!(
        record["cents_per_kwh"],
        serde_json::json!(40.0),
        "the record did not keep the tariff it was priced at"
    );

    let planned: serde_json::Value = serde_json::from_str(&succeeds(
        dir.path(),
        home.path(),
        &["team", "plan", "quick", "--format", "json"],
    ))
    .unwrap();
    let estimate = &planned["estimate"];
    assert!(estimate.is_object(), "the plan quotes no run: {planned}");
    assert_eq!(estimate["cents_per_kwh"], serde_json::json!(40.0));
    // The arithmetic only has an answer on a machine that reports a package
    // energy counter; where it does not, an unpriced estimate is the honest one.
    match record["watt_hours"].as_f64() {
        Some(watt_hours) => {
            assert!(
                watt_hours > 0.0,
                "a run measured no energy at all: {record}"
            );
            assert_eq!(estimate["watt_hours"], record["watt_hours"]);
            let expected = (watt_hours * 40.0 * 10.0).round() as u64;
            assert_eq!(
                estimate["cost_micros"],
                serde_json::json!(expected),
                "the estimate is not priced at the tariff that is set"
            );
        }
        None => assert!(
            estimate["cost_micros"].is_null(),
            "a machine that reported no energy priced something anyway: {estimate}"
        ),
    }
    let said = succeeds(dir.path(), home.path(), &["team", "plan", "quick"]);
    assert!(said.contains("estimated cost:"), "{said}");
    assert!(
        !said.contains("no $/kWh set"),
        "the estimate stayed unpriced: {said}"
    );
}

#[test]
fn the_run_report_in_json_carries_the_plan_the_approval_and_the_actuals() {
    let dir = project(&[("quick.toml", recipe("quick"))]);
    let home = sandbox_home(None);
    let out = run_in(
        dir.path(),
        home.path(),
        &[
            "team",
            "run",
            "quick",
            "--approved-by",
            "Sree",
            "--format",
            "json",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("the JSON form is one document");
    assert_eq!(parsed["launches"], serde_json::json!(true));
    assert_eq!(parsed["approved_by"], serde_json::json!("Sree"));
    assert_eq!(
        parsed["waves"],
        serde_json::json!([["survey", "harden"], ["integrate"]])
    );
    assert_eq!(parsed["capacity"], serde_json::json!(2));
    assert!(parsed["estimate"].is_null(), "nothing had run before this");
    let actual = &parsed["actual"];
    assert!(actual["wall_clock_ms"].as_u64().unwrap() >= 300);
    assert_eq!(actual["failed_roles"], serde_json::json!([]));
    assert_eq!(
        parsed["record"]
            .as_str()
            .unwrap()
            .rsplit('/')
            .next()
            .unwrap(),
        format!("{}.json", actual["run_id"].as_str().unwrap())
    );
    // The estimate the next plan offers is tied to this run by its fingerprint.
    assert_eq!(
        parsed["fingerprint"],
        records(dir.path())[0]["fingerprint"],
        "the plan and the record disagree about which recipe ran"
    );
}

/// The posture a new install is in, driven against the real binary (`OR-13`): a
/// recipe whose roles name another vendor's agents is refused as a team — no role
/// child, no record, no runs directory — and the refusal quotes the setting that
/// opens the rule. Opening it with that same command is what lets it run.
#[test]
fn the_shipped_posture_refuses_a_run_and_launches_nothing_at_all() {
    let text = recipe("touch-team").replace("sleep 0.3", "touch SHOULD-NOT-EXIST");
    let dir = project(&[("quick.toml", text)]);
    // A home with nothing written to it, so the settings are the ones the product
    // installs with rather than ones this file arranged.
    let home = TempDir::new().unwrap();

    // A plan is still a read: it shows the whole team and marks what it refuses.
    let plan = succeeds(dir.path(), home.path(), &["team", "plan", "touch-team"]);
    assert!(plan.contains("Posture: Local Only"), "{plan}");
    assert!(
        plan.contains("worker routes: work is handed only to xencode's own loop"),
        "the rule in force is stated, not implied: {plan}"
    );
    assert!(
        plan.contains("refused:  the Local Only profile does not hand work to it"),
        "{plan}"
    );
    assert!(
        plan.contains("would launch nothing at all while it stands"),
        "and the plan says what that means for a run: {plan}"
    );
    assert_eq!(
        plan.matches("xencode config set allow_external_workers true")
            .count(),
        1,
        "the plan states the setting once, on the posture line, rather than under every \
         refused role: {plan}"
    );

    let out = run_in(
        dir.path(),
        home.path(),
        &["team", "run", "touch-team", "--approved-by", "Sree"],
    );
    assert!(
        !out.status.success(),
        "a team the posture refuses must not report a run"
    );
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        said.contains("nothing was launched and nothing was recorded"),
        "{said}"
    );
    assert!(
        said.contains("xencode config set allow_external_workers true"),
        "a refusal that refuses to say how to stop being refused is an obstacle, not a \
         rule: {said}"
    );
    assert!(
        !dir.path().join("SHOULD-NOT-EXIST").exists(),
        "the posture refused the team, yet a role's command ran"
    );
    assert_no_history(
        dir.path(),
        "team run --approved-by Sree under the shipped posture",
    );

    // The JSON plan carries the same refusal against each role, so a script
    // reading it does not have to parse the prose to see why nothing would run.
    let json = succeeds(
        dir.path(),
        home.path(),
        &["team", "plan", "touch-team", "--format", "json"],
    );
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["posture"], serde_json::json!("Local Only"));
    assert_eq!(parsed["launches"], serde_json::json!(false));
    let refused: Vec<&serde_json::Value> = parsed["roles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|role| !role["refused_by_posture"].is_null())
        .collect();
    assert_eq!(
        refused.len(),
        3,
        "every role in this recipe names a roster agent: {json}"
    );

    // Open the one rule the way a person would, and the same project runs.
    let opened = run_in(
        dir.path(),
        home.path(),
        &["config", "set", "allow_external_workers", "true"],
    );
    assert!(
        opened.status.success(),
        "{}",
        String::from_utf8_lossy(&opened.stderr)
    );
    let ran = succeeds(
        dir.path(),
        home.path(),
        &["team", "run", "touch-team", "--approved-by", "Sree"],
    );
    assert!(ran.contains("Recorded in"), "{ran}");
    assert!(
        dir.path().join("SHOULD-NOT-EXIST").exists(),
        "the roles' commands really ran once the rule was open"
    );
    let recorded = records(dir.path());
    assert_eq!(recorded.len(), 1, "one run, one record");
    assert_eq!(recorded[0]["approved_by"], serde_json::json!("Sree"));
    assert!(
        !ran.contains("refused by the Local Only"),
        "the opened posture refuses nothing: {ran}"
    );
}
