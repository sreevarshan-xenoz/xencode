//! `xencode agents --route` as the person runs it (`OR-6` gating, `OR-11`
//! explainability). Every one of these drives the real binary, which probes the
//! real `--help` of every roster agent installed on this machine, so the facts
//! printed are facts a reader could check.
//!
//! Two invariants hold wherever the command runs, and the tests below are built
//! out of them rather than out of a list of agents that happens to be installed
//! here: a worker is only ever offered for a capability a probe confirmed on it,
//! and nothing is printed as a number unless something measured it.
//!
//! Every roster agent is refused outright by the posture a new install is in
//! (`OR-13`), so the gating cases below run in a home that has opened that one
//! rule — they are about what a probe confirmed, and a refusal that happens
//! before any probe would test nothing. The shipped posture is the two cases at
//! the end of the file.

use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;
use tempfile::TempDir;

fn xencode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xencode")
}

/// A home with the worker rule opened by the command a person would use, so a
/// candidate reaches the measurements these cases are about.
fn open_home() -> &'static Path {
    static HOME: OnceLock<TempDir> = OnceLock::new();
    HOME.get_or_init(|| {
        let home = TempDir::new().unwrap();
        let out = Command::new(xencode_bin())
            .args(["config", "set", "allow_external_workers", "true"])
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
            "opening the worker rule failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        home
    })
    .path()
}

/// A home with nothing written to it: the posture the product installs with, both
/// rules closed.
fn default_home() -> &'static Path {
    static HOME: OnceLock<TempDir> = OnceLock::new();
    HOME.get_or_init(|| TempDir::new().unwrap()).path()
}

fn run_in(home: &Path, args: &[&str]) -> String {
    let out = Command::new(xencode_bin())
        .args(args)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("XCODE_CONFIG_DIR", home.join(".config").join("xencode"))
        .output()
        .expect("must run xencode agents --route");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn run(args: &[&str]) -> String {
    run_in(open_home(), args)
}

fn route_json(task: &str, extra: &[&str]) -> serde_json::Value {
    let mut args = vec!["agents", "--route", task, "--format", "json"];
    args.extend(extra);
    serde_json::from_str(&run(&args)).expect("the json format must print one parseable decision")
}

#[test]
fn a_worker_is_offered_only_for_a_capability_a_probe_confirmed_on_it() {
    let decision = route_json("acp-task-dispatch", &["--require-cap", "acp"]);
    let evaluations = decision["candidate_evaluations"]
        .as_array()
        .expect("evaluations array");
    assert!(!evaluations.is_empty(), "the roster must offer candidates");

    for ev in evaluations {
        let caps: Vec<&str> = ev["probed_capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let eligible = ev["eligible"].as_bool().unwrap();
        if caps.contains(&"acp") {
            assert!(eligible, "{ev:?} was probed as able to do acp");
            assert!(ev["rejection"].is_null());
        } else {
            assert!(!eligible, "{ev:?} has no confirmed acp");
            let words = ev["rejection"]["MissingCapabilities"]["required"]
                .as_array()
                .expect("the refusal names what was missing");
            assert!(words.iter().any(|w| w == "acp"), "{words:?}");
        }
    }
}

#[test]
fn a_confirmed_absence_is_not_counted_as_an_ability() {
    // The probe confirms two opposite things: that a flag is there, and that it is
    // not. Only the first may open a door, so a worker admitted to an `acp` task
    // must show evidence that *found* something. On a machine with none of these
    // binaries installed nothing is admitted and there is nothing to check — the
    // binaries installed nothing is admitted and there is nothing to check — the
    // refusal path is its own test below.
    let decision = route_json("acp-absence-task", &["--require-cap", "acp"]);
    for ev in decision["candidate_evaluations"].as_array().unwrap() {
        if ev["eligible"].as_bool().unwrap_or(false) {
            let line = ev["facts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| f.as_str().unwrap())
                .find(|f| f.starts_with("acp: "))
                .unwrap_or_else(|| panic!("an eligible worker must show its acp evidence: {ev:?}"));
            assert!(
                !line.contains("not advertised") && !line.contains("contradicted"),
                "a confirmed absence was used as an ability: {line}"
            );
        }
    }
}

#[test]
fn a_cost_ceiling_that_cannot_be_applied_is_said_as_not_applied() {
    // The claim this replaces was a bug wearing a test: an impossible ceiling was
    // reported as refusing every worker, on the strength of a price nobody had
    // ever measured. A ceiling xencode cannot check must be printed as unchecked.
    let decision = route_json(
        "budget-constrained-task",
        &["--require-cap", "stream", "--max-cost", "0.0001"],
    );
    let ceiling_step = decision["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["check"] == "cost ceiling")
        .expect("the decision lists the ceiling step");
    // Two honest answers, depending on the machine: a ceiling over candidates
    // nobody could price "was not applied"; with no candidate at all (no agent
    // installed, as on CI) there was nothing to price and it "ruled nobody out".
    let words = ceiling_step["words"].as_str().unwrap();
    assert!(
        words.contains("was not applied") || words.contains("ruled nobody out"),
        "{}",
        ceiling_step["words"]
    );
    assert!(
        !ceiling_step["decided"].as_bool().unwrap(),
        "nothing was measured, so nothing was decided"
    );
    for ev in decision["candidate_evaluations"].as_array().unwrap() {
        if ev["eligible"].as_bool().unwrap_or(false) {
            assert!(
                ev["not_checked"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|n| n.as_str().unwrap().contains("your ceiling of 0.00$")),
                "a ceiling that could not be checked has to be said per worker: {ev:?}"
            );
        }
    }
}

#[test]
fn no_number_appears_in_the_printed_decision_that_nothing_measured() {
    // The whole of OR-11: the old output read `load: 0/5, cost: $0.05` for every
    // worker on the strength of two literals in the source. Those shapes must not
    // come back, and the honest version of the same line says `not measured`.
    let stdout = run(&["agents", "--route", "honesty-check"]);
    for fabricated in ["load: 0/5", "$0.05", "0/5"] {
        assert!(
            !stdout.contains(fabricated),
            "`{fabricated}` is a number nothing measured:\n{stdout}"
        );
    }
    assert!(stdout.contains("not measured"), "{stdout}");
    assert!(stdout.contains("What the router asked"), "{stdout}");
    assert!(stdout.contains("Every worker considered"), "{stdout}");
}

#[test]
fn the_choice_prints_where_its_own_capability_claims_came_from() {
    let stdout = run(&[
        "agents",
        "--route",
        "evidence-check",
        "--require-cap",
        "stream",
    ]);
    // A decision that chose a worker explains its facts; one that chose nothing
    // (no agent installed, as on CI) says so instead.
    if !stdout.contains("chosen: nothing") {
        assert!(stdout.contains("The facts behind the choice"), "{stdout}");
    }
    // Either the probe read this machine's help output — and the screen it read is
    // named — or this machine has none of these binaries, and the decision says so.
    let evidence_or_absence = stdout.contains("read from `")
        || stdout.contains("never probed")
        || stdout.contains("nothing this machine could see");
    assert!(evidence_or_absence, "{stdout}");
}

#[test]
fn the_checks_are_reported_in_the_order_the_router_applied_them() {
    let decision = route_json("step-order-check", &[]);
    let checks: Vec<&str> = decision["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["check"].as_str().unwrap())
        .collect();
    assert_eq!(
        checks,
        vec![
            "profile",
            "capabilities",
            "load",
            "cost ceiling",
            "load ranking",
            "cost ranking",
            "name"
        ],
        "a step that did not run still has to appear, or the reader cannot see its absence"
    );
    for step in decision["steps"].as_array().unwrap() {
        assert!(step["ran"].is_boolean());
        assert!(step["decided"].is_boolean());
        let words = step["words"].as_str().unwrap();
        assert!(!words.is_empty());
        // Debug-shaped text is the tell of a placeholder.
        assert!(!words.contains('{'), "{words}");
    }
}

#[test]
fn a_task_nothing_can_serve_refuses_by_name_and_cites_the_probe() {
    let decision = route_json("impossible-task", &["--require-cap", "telepathy"]);
    assert!(decision["selected_worker"].is_null());
    let explanation = decision["explanation"].as_str().unwrap();
    assert!(
        explanation.starts_with("Nothing was routed"),
        "{explanation}"
    );
    for ev in decision["candidate_evaluations"].as_array().unwrap() {
        assert!(
            explanation.contains(ev["worker_id"].as_str().unwrap()),
            "every refusal belongs in the sentence: {explanation}"
        );
    }
    assert!(explanation.contains("telepathy"), "{explanation}");
}

#[test]
fn an_unrankable_field_is_reported_as_the_convention_that_it_is() {
    // Nothing on this machine measures another process's queue, so the choice
    // between equally capable workers falls to the order of their names — and the
    // printed reason has to own that rather than imply a measurement was made.
    let stdout = run(&["agents", "--route", "convention-check"]);
    let chosen = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("chosen: "))
        .expect("the text output names the choice")
        .split("chosen: ")
        .nth(1)
        .unwrap()
        .trim()
        .to_string();
    assert!(
        stdout.contains("a convention, not a finding about")
            || stdout.contains("only one worker was eligible")
            || stdout.contains("chosen: nothing"),
        "{stdout}"
    );
    if stdout.contains("a convention, not a finding about") {
        assert!(
            stdout.contains(&format!("about {chosen}")),
            "the convention line must name the worker it settled on: {stdout}"
        );
    }
}

/// The posture a new install is in, run for real (`OR-13`): every candidate the
/// roster knows is refused by that name before a single probe is read, the
/// refusal is printed as its own step and against every worker, and the rule
/// line above them names the setting that opens it — once, not per worker.
#[test]
fn the_shipped_posture_refuses_every_roster_agent_before_any_probe_is_read() {
    let text = run_in(default_home(), &["agents", "--route", "posture-check"]);
    assert!(text.contains("Posture: Local Only"), "{text}");
    assert!(
        text.contains("work is handed only to xencode's own loop"),
        "the rule in force is stated, not implied: {text}"
    );
    assert!(
        text.contains("xencode config set allow_external_workers true"),
        "the printed rule names the way out: {text}"
    );

    let decision: serde_json::Value = serde_json::from_str(&run_in(
        default_home(),
        &["agents", "--route", "posture-check", "--format", "json"],
    ))
    .expect("the json form still prints one decision");
    assert!(
        decision["selected_worker"].is_null(),
        "nothing on the roster may be chosen under it: {:?}",
        decision["selected_worker"]
    );
    let steps = decision["steps"].as_array().unwrap();
    assert_eq!(steps[0]["check"].as_str().unwrap(), "profile");
    assert!(
        steps[0]["decided"].as_bool().unwrap(),
        "the posture decided this one: {}",
        steps[0]["words"]
    );
    assert!(
        steps[0]["words"]
            .as_str()
            .unwrap()
            .contains("before the capability, load and cost checks were consulted"),
        "{}",
        steps[0]["words"]
    );

    let evaluations = decision["candidate_evaluations"].as_array().unwrap();
    assert!(!evaluations.is_empty(), "the roster names candidates");
    for ev in evaluations {
        let refusal = &ev["rejection"]["ExternalWorkerRefused"];
        assert!(
            refusal.is_object(),
            "a roster agent is refused by name, not left unexplained: {ev:?}"
        );
        let words = refusal["refusal"].as_str().unwrap();
        assert!(
            words.contains("another vendor's agent"),
            "the reason is said: {words}"
        );
        assert!(
            words.contains("Local Only profile"),
            "and says which posture said it: {words}"
        );
        assert!(
            ev["rejection"]["MissingCapabilities"].is_null(),
            "a posture refusal is not reported as a capability refusal: {ev:?}"
        );
    }
    // The summary is the one line a reader actually parses, so it groups every
    // name under the reason instead of repeating that reason once per name.
    let explanation = decision["explanation"].as_str().unwrap();
    let names = evaluations
        .iter()
        .map(|ev| ev["worker_id"].as_str().unwrap())
        .collect::<Vec<_>>();
    for name in &names {
        assert!(explanation.contains(name), "{name} is named: {explanation}");
    }
    assert_eq!(
        explanation
            .matches("refused by the Local Only posture")
            .count(),
        1,
        "one sentence for one rule, shared by all {}: {explanation}",
        names.len()
    );
    assert!(
        explanation.contains("a program xencode does not control"),
        "and it says what the rule is protecting: {explanation}"
    );
}

/// Opening the one rule puts the same machine's agents back in the running, and
/// nothing else changes: the capability, load and cost steps are the ones that
/// decide, exactly as the cases above describe them.
#[test]
fn opening_the_worker_rule_puts_the_same_agents_back_in_the_running() {
    let decision = route_json("posture-check", &[]);
    let steps = decision["steps"].as_array().unwrap();
    assert_eq!(steps[0]["check"].as_str().unwrap(), "profile");
    assert!(
        !steps[0]["decided"].as_bool().unwrap(),
        "with the rule open the posture settles nothing: {}",
        steps[0]["words"]
    );
    assert!(
        steps[0]["words"]
            .as_str()
            .unwrap()
            .contains("external workers allowed"),
        "the posture is named as what it is once opened: {}",
        steps[0]["words"]
    );
    for ev in decision["candidate_evaluations"].as_array().unwrap() {
        assert!(
            ev["rejection"]["ExternalWorkerRefused"].is_null(),
            "nothing is refused on the posture now: {ev:?}"
        );
    }
    let text = run(&["agents", "--route", "posture-check"]);
    assert!(
        !text.contains("The one thing decided before the capability, load and cost checks"),
        "that sentence belongs to a decision the posture actually made: {text}"
    );
}
