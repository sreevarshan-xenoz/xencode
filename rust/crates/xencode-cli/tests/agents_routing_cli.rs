//! `xencode agents --route` as the person runs it (`OR-6` gating, `OR-11`
//! explainability). Every one of these drives the real binary, which probes the
//! real `--help` of every roster agent installed on this machine, so the facts
//! printed are facts a reader could check.
//!
//! Two invariants hold wherever the command runs, and the tests below are built
//! out of them rather than out of a list of agents that happens to be installed
//! here: a worker is only ever offered for a capability a probe confirmed on it,
//! and nothing is printed as a number unless something measured it.

use std::process::Command;

fn xencode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xencode")
}

fn run(args: &[&str]) -> String {
    let out = Command::new(xencode_bin())
        .args(args)
        .output()
        .expect("must run xencode agents --route");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
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
    assert!(
        ceiling_step["words"]
            .as_str()
            .unwrap()
            .contains("was not applied"),
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
    assert!(stdout.contains("The facts behind the choice"), "{stdout}");
    // Either the probe read this machine's help output — and the screen it read is
    // named — or this machine has none of these binaries, and the decision says so.
    let evidence_or_absence = stdout.contains("read from `") || stdout.contains("never probed");
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
