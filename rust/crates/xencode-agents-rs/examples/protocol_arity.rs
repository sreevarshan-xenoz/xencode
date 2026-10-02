use std::collections::BTreeMap;

use xencode_agents_rs::protocol::{normalise_line, run_completed, Origin};

/// Reads a captured stream off disk and prints how it lands in the common
/// model. Run against real captures, because the point is to see what the model
/// actually captures rather than what a hand-picked line suggested.
fn main() {
    let mut args = std::env::args().skip(1);
    let agent = args
        .next()
        .expect("usage: protocol_arity <agent> <captured stdout>");
    let stream = std::fs::read_to_string(args.next().expect("a captured stream")).unwrap();

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut synthesised = 0usize;
    let mut events = Vec::new();
    for line in stream.lines() {
        for event in normalise_line(&agent, line) {
            if event.origin() == Origin::Synthesised {
                synthesised += 1;
            }
            *counts.entry(event.name().to_string()).or_default() += 1;
            events.push(event);
        }
    }
    println!(
        "{agent}: {} | completed={} synthesised={synthesised}",
        counts
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" "),
        run_completed(&events),
    );
}
