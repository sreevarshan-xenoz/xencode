use std::path::PathBuf;
use xencode_tui_rs::{
    agent_tools::{ApprovalMode, HeadlessPolicy},
    run_agent, serve_scripted_answers, AgentRunError, AgentRunOptions,
};

fn unique_dir(label: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("xencode-{label}-{nonce}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn headless_agent_refuses_when_neither_approval_mode_nor_policy_supplied() {
    let dir = unique_dir("refuse");
    let options = AgentRunOptions {
        tool_root: dir.clone(),
        prompt: "do something".to_string(),
        model: None,
        approval_mode: None,
        headless_policy: None,
        max_rounds: 1,
        xencode_dir: None,
        run_id: None,
        session_id: None,
        ollama_url: None,
        llama_cpp_url: None,
    };

    let result = run_agent(options).await;
    assert_eq!(result.err(), Some(AgentRunError::MissingApprovalPolicy));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn headless_agent_drives_one_round_asserts_diff_and_ledger_rows() {
    let dir = unique_dir("headless-round");

    // 1. Initialize a real git working tree
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(&dir)
            .output()
            .expect("git must run");
        assert!(out.status.success(), "git {:?} failed: {:?}", args, out);
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    git(&["init", "-q", "."]);
    git(&["config", "user.name", "tester"]);
    git(&["config", "user.email", "tester@example.invalid"]);
    std::fs::write(dir.join("file.txt"), "hello world\n").unwrap();
    git(&["add", "-A"]);
    git(&["commit", "-qm", "initial"]);

    // 2. Start real local loopback HTTP server serving scripted model response
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(serve_scripted_answers(
        listener,
        vec![
            serde_json::json!({
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "function": {
                            "name": "write_file",
                            "arguments": {
                                "path": "file.txt",
                                "content": "updated by agent\n"
                            }
                        }
                    }]
                },
                "done": true
            }),
            serde_json::json!({
                "message": {
                    "role": "assistant",
                    "content": "All edits applied successfully."
                },
                "done": true
            }),
        ],
    ));

    // 3. Drive one round as a library call
    let prompt = "update file.txt with new content";
    let run_id = "agent-round-test-1".to_string();
    let session_id = "agent-session-test-1".to_string();
    let xencode_dir = dir.join(".xencode");

    let options = AgentRunOptions {
        tool_root: dir.clone(),
        prompt: prompt.to_string(),
        model: Some("local-model".to_string()),
        approval_mode: Some(ApprovalMode::AllAllow),
        headless_policy: Some(HeadlessPolicy::new(vec!["write_file".to_string()])),
        max_rounds: 2,
        xencode_dir: Some(xencode_dir.clone()),
        run_id: Some(run_id.clone()),
        session_id: Some(session_id.clone()),
        ollama_url: Some(format!("http://{addr}")),
        llama_cpp_url: None,
    };

    let output = run_agent(options).await.expect("run_agent succeeds");
    let _ = server.await;

    // 4. Assert the diff it produced
    assert!(
        output.diff.contains("+updated by agent"),
        "diff must show the added line: {}",
        output.diff
    );
    assert!(
        output.diff.contains("-hello world"),
        "diff must show the removed line: {}",
        output.diff
    );
    assert_eq!(output.edited_files, vec!["file.txt"]);

    // 5. Assert the ledger rows it wrote
    assert!(
        output.ledger_file.exists(),
        "ledger file must exist at {}",
        output.ledger_file.display()
    );
    let ledger_entries = xencode_context_rs::ledger::read_ledger(&xencode_dir);
    assert!(
        !ledger_entries.is_empty(),
        "ledger must contain rows for the completed run"
    );
    let row = ledger_entries
        .iter()
        .find(|r| r.session.as_deref() == Some(&session_id))
        .expect("must find ledger row for session");
    assert_eq!(row.exit_code, 0, "ledger row exit code must be 0");
    assert_eq!(
        row.subjects,
        vec![xencode_context_rs::ledger::digest_hex(prompt)],
        "ledger row subject must be digest of the prompt"
    );

    // Assert cache runs record exists
    let runs_cache = xencode_dir.join("cache/runs.jsonl");
    assert!(runs_cache.exists(), "cache runs file must exist");
    let runs_content = std::fs::read_to_string(&runs_cache).unwrap();
    assert!(
        runs_content.contains(&run_id),
        "runs cache must record run_id"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// RA-1: against a real llama.cpp server named by XENCODE_LIVE_LLAMACPP_URL,
/// the headless agent returns the model's final answer and how many rounds
/// it actually took.
#[tokio::test]
#[ignore = "needs a running llama.cpp server: XENCODE_LIVE_LLAMACPP_URL"]
async fn headless_agent_returns_its_answer_and_round_count() {
    let url = std::env::var("XENCODE_LIVE_LLAMACPP_URL")
        .expect("set XENCODE_LIVE_LLAMACPP_URL to a running llama.cpp server");
    let model =
        std::env::var("XENCODE_LIVE_MODEL").unwrap_or_else(|_| "llamacpp:qwen3-4b".to_string());
    let dir = unique_dir("headless-live");
    let options = AgentRunOptions {
        tool_root: dir.clone(),
        prompt: "Create a file named hello.txt containing the word hi, then say you are done."
            .to_string(),
        model: Some(model),
        approval_mode: Some(ApprovalMode::AllAllow),
        headless_policy: Some(HeadlessPolicy::new(vec!["write_file".to_string()])),
        max_rounds: 4,
        xencode_dir: Some(dir.join(".xencode")),
        run_id: None,
        session_id: None,
        ollama_url: None,
        llama_cpp_url: Some(url),
    };
    let output = run_agent(options).await.expect("run_agent succeeds");
    let written = std::fs::read_to_string(dir.join("hello.txt")).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!output.final_answer.trim().is_empty(), "no final answer");
    assert!(
        (1..=4).contains(&output.rounds),
        "rounds counted from the loop: {}",
        output.rounds
    );
    assert_eq!(written.trim(), "hi");
}
