use std::process::Command;

fn xencode_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xencode")
}

#[test]
fn cli_agents_route_gated_by_capabilities_never_by_vendor_name() {
    // 1. Route requiring a specific capability (e.g. acp)
    let output = Command::new(xencode_bin())
        .arg("agents")
        .arg("--route")
        .arg("acp-task-dispatch")
        .arg("--require-cap")
        .arg("acp")
        .arg("--format")
        .arg("json")
        .output()
        .expect("cli execution failed");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let decision: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON expected");

    assert_eq!(decision["task_id"], "acp-task-dispatch");
    assert!(decision["explanation"].is_string());

    // Check candidate evaluations
    let evaluations = decision["candidate_evaluations"]
        .as_array()
        .expect("evaluations array");
    assert!(!evaluations.is_empty());

    for ev in evaluations {
        let caps: Vec<String> = ev["probed_capabilities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();

        if caps.contains(&"acp".to_string()) {
            // Must be eligible if within load/cost
            if ev["eligible"].as_bool().unwrap() {
                assert!(ev["rejection"].is_null());
            }
        } else {
            // Must be rejected with MissingCapabilities
            assert_eq!(ev["eligible"], false);
            let rej = &ev["rejection"];
            assert!(
                rej.to_string().contains("MissingCapabilities")
                    || rej["MissingCapabilities"].is_array()
            );
        }
    }
}

#[test]
fn cli_agents_route_enforces_cost_ceiling() {
    // Route requiring stream capability but with an impossible $0.0001 cost ceiling
    let output = Command::new(xencode_bin())
        .arg("agents")
        .arg("--route")
        .arg("budget-constrained-task")
        .arg("--require-cap")
        .arg("stream")
        .arg("--max-cost")
        .arg("0.0001")
        .arg("--format")
        .arg("json")
        .output()
        .expect("cli execution failed");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let decision: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON expected");

    assert_eq!(decision["task_id"], "budget-constrained-task");
    // Should select None because candidates cost > 0.0001
    assert!(decision["selected_worker"].is_null());
    assert!(decision["explanation"]
        .as_str()
        .unwrap()
        .contains("Routing refused"));
}
