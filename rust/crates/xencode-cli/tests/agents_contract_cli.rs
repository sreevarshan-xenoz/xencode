//! `xencode agents --contract`, end to end (`AR-3`).
//!
//! The probe runs the real `--help` of every roster agent installed on this
//! machine, so what it can say depends on the machine. These tests do not pin an
//! answer that only holds here; they pin the rules the command must follow with
//! whatever it does find: the three outcomes are counted apart, the two formats
//! agree, and a capability is never reported as absent unless a word was actually
//! searched for it. On 2026-10-08 the summary line printed
//! `54 claims confirmed, 0 contradicted` while one of those fifty-four was an
//! agent whose help could not be read at all, and five of the eight denials named
//! no searched token — both of which these checks would have refused.
//!
//! One probe of the real binary is shared by both tests: reading eight CLIs' help
//! takes about nine seconds, and repeating it per test would triple that.

use std::process::Command;
use std::sync::OnceLock;

fn contract(extra: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_xencode"))
        .arg("agents")
        .arg("--contract")
        .args(extra)
        .output()
        .expect("must run xencode agents --contract");
    assert!(
        output.status.success(),
        "the contract probe exited {:?} with: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The same run's two formats: the human summary, and the parsed report.
fn report() -> &'static (String, serde_json::Value) {
    static REPORT: OnceLock<(String, serde_json::Value)> = OnceLock::new();
    REPORT.get_or_init(|| {
        let json = contract(&["--format", "json"]);
        let text = contract(&[]);
        let parsed = serde_json::from_str(&json).expect("the json report parses");
        (text, parsed)
    })
}

#[test]
fn the_summary_counts_what_was_measured_apart_from_what_was_not() {
    let (_, json) = report();
    let summary = &json["summary"];
    let claims = summary["claims"].as_u64().unwrap() as usize;
    let confirmed = summary["confirmed"].as_u64().unwrap() as usize;
    let contradicted = summary["contradicted"].as_u64().unwrap() as usize;
    let untested = summary["untested"].as_u64().unwrap() as usize;
    let absences = summary["confirmed_absences"].as_u64().unwrap() as usize;

    // The arithmetic `results.len() - contradicted` got wrong: an untested claim
    // is not a confirmed one, and the old line counted both as confirmed.
    assert_eq!(
        claims,
        confirmed + contradicted + untested,
        "the summary adds up to something other than the claims probed: {summary}"
    );
    assert!(
        absences <= confirmed,
        "{absences} absences cannot be a subset of {confirmed} confirmations"
    );
    // The arithmetic above holds on any machine. Whether anything was probed
    // depends on which agents are installed: CI has none, so zero is a fact
    // about the runner there, not a failure of the summary.
    if claims == 0 {
        eprintln!("no roster agent is installed here, so nothing was probed: {summary}");
    }

    // Every denied claim is either a searched absence or an untested claim, and
    // the two are distinguishable in the line a reader is shown.
    for claim in json["claims"].as_array().unwrap() {
        if claim["expected"].as_bool().unwrap() {
            continue;
        }
        let evidence = claim["evidence"].as_str().unwrap();
        if !evidence.contains("not advertised") {
            continue;
        }
        let searched = claim["missing"]
            .as_array()
            .unwrap()
            .iter()
            .any(|token| !token.as_str().unwrap().is_empty());
        assert!(
            searched,
            "{} {} printed as an absence with no token searched: {evidence}",
            claim["agent"].as_str().unwrap(),
            claim["claim"].as_str().unwrap()
        );
    }
}

#[test]
fn the_text_report_names_every_claim_it_could_not_test() {
    // The rows a reader must not have to hunt for. An agent can be present on
    // `PATH` and still yield no testable claim — on 2026-10-08 that was `cline`,
    // a mise shim left behind by an uninstalled tool, which answered `--help`
    // with an error and a non-zero exit. Whether this machine has such an agent
    // today is its own business, so the test checks that the two reports agree
    // rather than pinning one answer: every untestable row in the JSON appears in
    // the text, the text lists exactly as many as the JSON counts, and the summary
    // line carries that number instead of folding it into "confirmed" — which is
    // what it did on 2026-10-08, printing `54 claims confirmed` with one of them
    // never tested.
    let (text, json) = report();
    let untested = json["summary"]["untested"].as_u64().unwrap();

    let listed = text
        .lines()
        .filter(|line| line.trim_start().starts_with("untestable "))
        .count() as u64;
    assert_eq!(
        listed, untested,
        "the report lists {listed} untestable rows but counted {untested}: {text}"
    );
    for claim in json["claims"].as_array().unwrap() {
        if !claim["verdict"].as_str().unwrap().starts_with("Untestable") {
            continue;
        }
        let row = format!(
            "{} {}",
            claim["agent"].as_str().unwrap(),
            claim["claim"].as_str().unwrap()
        );
        assert!(
            text.contains(&row),
            "an untested claim is missing from the text report: {row}"
        );
    }
    let summary_line = text
        .lines()
        .find(|line| line.contains("claims confirmed"))
        .unwrap_or_else(|| panic!("no summary line in the text report: {text}"));
    assert!(
        summary_line.contains("untested") && summary_line.contains(&untested.to_string()),
        "the summary line hides the claims that were never tested: {summary_line}"
    );
}
