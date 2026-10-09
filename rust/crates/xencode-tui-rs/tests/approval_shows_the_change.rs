//! What a person is asked to approve is the change itself (AF-6).
//!
//! This drives the real gating path — the same `execute_tool_call_approved` the
//! chat loop calls — against a real file on disk, takes the request it actually
//! sent, and paints it with the real overlay renderer. Nothing here is a stub
//! preview: the diff on the screen is the diff the write would produce.

use ratatui::{backend::TestBackend, Terminal};
use std::path::PathBuf;
use std::sync::{atomic::AtomicBool, Arc};
use tokio::sync::{mpsc, oneshot};
use xencode_tui_rs::{
    agent_tools::{
        new_plan_handle, new_task_runtime, ApprovalAnswer, ApprovalCtx, ApprovalMode,
        ApprovalRequest, TaskRuntime, DEFAULT_COMMAND_TIMEOUT,
    },
    app::{App, UiMessage},
    ui::draw,
};

fn temp_repo(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "xencode-approval-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A context on the asking path, built from the same public fields the chat
/// loop fills — no listening UI beyond the channel this test drains.
fn asking() -> (
    ApprovalCtx,
    mpsc::UnboundedReceiver<(ApprovalRequest, oneshot::Sender<ApprovalAnswer>)>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let checkpoints = Arc::new(xencode_tui_rs::agent_tools::CheckpointStore::new());
    let turn = checkpoints.begin_turn();
    (
        ApprovalCtx {
            mode: ApprovalMode::Ask,
            headless_policy: None,
            grants: Arc::new(std::sync::Mutex::new(Vec::new())),
            prompts: tx,
            checkpoints,
            turn,
            command_timeout: DEFAULT_COMMAND_TIMEOUT,
            plan: new_plan_handle(),
            mcp: Arc::new(xencode_tui_rs::mcp::McpHub::new()),
            skills: Arc::new(xencode_plugin_rs::SkillRuntime::empty(
                PathBuf::new(),
                PathBuf::new(),
            )),
            hooks: xencode_config_rs::AgentHooks::default(),
            schemas: std::collections::HashMap::new(),
            online_docs: false,
            web_fetch: false,
            search: Ok(xencode_analysis_rs::SearchProvider::None),
            session_id: None,
            approvals: Arc::new(std::sync::Mutex::new(Vec::new())),
            taint: Arc::new(AtomicBool::new(false)),
            sandbox: xencode_tui_rs::sandbox::Sandbox::disabled(),
            redaction: Arc::new(xencode_context_rs::Vault::default()),
            ask: None,
            repro: Arc::new(xencode_tui_rs::reprogate::ReproGate::new()),
        },
        rx,
    )
}

/// The screen text the overlay paints for one request, at a real terminal size.
fn overlay_text(request: &ApprovalRequest) -> String {
    let mut app = App::for_tests();
    app.messages.push(UiMessage {
        role: "assistant".into(),
        content: "proposing a change".into(),
    });
    let (dead_responder, _dropped) = oneshot::channel();
    app.approval_queue
        .push_back((request.clone(), dead_responder));
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal
        .draw(|f| draw(f, &mut app))
        .expect("the approval overlay must render");
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect::<Vec<_>>()
        .join("")
}

#[tokio::test]
async fn the_overlay_paints_the_diff_the_write_would_produce_and_reviews_a_moved_file() {
    let root = temp_repo("shown");
    std::fs::write(root.join("a.txt"), "one\n").unwrap();
    let (ctx, mut prompts) = asking();
    let rt: TaskRuntime = new_task_runtime();

    let call = xencode_providers_rs::ToolCall {
        id: "call-1".into(),
        name: "write_file".into(),
        arguments: serde_json::json!({"path": "a.txt", "content": "two\n"}),
    };
    let running = {
        let (root, ctx, rt, call) = (root.clone(), ctx.clone(), rt.clone(), call.clone());
        tokio::spawn(async move {
            xencode_tui_rs::agent_tools::execute_tool_call_approved(&rt, &root, &call, &ctx, None)
                .await
        })
    };

    // The first review: what the screen actually says about the change.
    let (request, responder) = prompts.recv().await.expect("a prompt arrives");
    let screen = overlay_text(&request);
    for expected in ["Allow file change?", "write_file a.txt", "-one", "+two"] {
        assert!(
            screen.contains(expected),
            "the overlay must show {expected:?}:\n{screen}\n(preview was {:?})",
            request.preview
        );
    }

    // The file moves while the person is reading it, and they answer anyway.
    std::fs::write(root.join("a.txt"), "ZERO\n").unwrap();
    responder.send(ApprovalAnswer::Approved).unwrap();
    let (second, responder) = prompts.recv().await.expect("the moved file is re-shown");
    let screen = overlay_text(&second);
    assert!(
        screen.contains("-ZERO"),
        "the second review shows the bytes on disk now, not the ones first shown:\n{screen}"
    );
    responder.send(ApprovalAnswer::Approved).unwrap();

    let result = running.await.unwrap();
    assert!(result.starts_with("updated a.txt"), "{result}");
    assert_eq!(
        std::fs::read_to_string(root.join("a.txt")).unwrap(),
        "two\n"
    );
    std::fs::remove_dir_all(&root).unwrap();
}

/// Every regular file under `root`, keyed by its path and valued by its SHA-256,
/// skipping `.xencode` — the denial there writes a lesson draft on purpose, and
/// that is not the person's tree.
fn checksums(root: &PathBuf) -> std::collections::BTreeMap<String, String> {
    use sha2::{Digest, Sha256};
    let mut out = std::collections::BTreeMap::new();
    let mut walk = vec![root.clone()];
    while let Some(dir) = walk.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name() == Some(std::ffi::OsStr::new(".xencode")) {
                    continue;
                }
                walk.push(path);
                continue;
            }
            let mut hasher = Sha256::new();
            hasher.update(std::fs::read(&path).unwrap());
            out.insert(
                path.strip_prefix(root).unwrap().display().to_string(),
                format!("{:x}", hasher.finalize()),
            );
        }
    }
    out
}

/// AF-6's done-when, watched end to end: the overlay paints the real diff, and
/// declining it leaves the checksum of every file in the tree exactly where it
/// was — including the file the change would have replaced and the new file it
/// would have created.
#[tokio::test]
async fn declining_the_shown_change_leaves_every_checksum_untouched() {
    use xencode_tui_rs::agent_tools::DENIED_RESULT;

    let root = temp_repo("declined");
    std::fs::write(root.join("a.txt"), "one\n").unwrap();
    std::fs::write(root.join("keep.md"), "untouched\n").unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/mod.rs"), "pub fn kept() {}\n").unwrap();
    let before = checksums(&root);
    assert_eq!(before.len(), 3, "{before:?}");

    let (ctx, mut prompts) = asking();
    let rt: TaskRuntime = new_task_runtime();
    let call = xencode_providers_rs::ToolCall {
        id: "call-1".into(),
        name: "write_file".into(),
        arguments: serde_json::json!({"path": "a.txt", "content": "two\n"}),
    };
    let running = {
        let (root, ctx, rt, call) = (root.clone(), ctx.clone(), rt.clone(), call.clone());
        tokio::spawn(async move {
            xencode_tui_rs::agent_tools::execute_tool_call_approved(&rt, &root, &call, &ctx, None)
                .await
        })
    };

    let (request, responder) = prompts.recv().await.expect("a prompt arrives");
    let screen = overlay_text(&request);
    assert!(
        screen.contains("-one") && screen.contains("+two"),
        "the person is deciding on the change itself:\n{screen}"
    );
    responder.send(ApprovalAnswer::Denied).unwrap();

    let result = running.await.unwrap();
    assert_eq!(result, DENIED_RESULT, "{result}");
    assert_eq!(
        checksums(&root),
        before,
        "a refusal must not move a single byte in the tree"
    );
    assert_eq!(
        ctx.checkpoints.turns(),
        0,
        "nothing was written, so there is nothing to undo"
    );

    // A refusal must not create a file either, only overwrite one.
    let creator = xencode_providers_rs::ToolCall {
        id: "call-2".into(),
        name: "write_file".into(),
        arguments: serde_json::json!({"path": "new.rs", "content": "fn never()\n"}),
    };
    let running = {
        let (root, ctx, rt) = (root.clone(), ctx.clone(), rt.clone());
        let creator = creator.clone();
        tokio::spawn(async move {
            xencode_tui_rs::agent_tools::execute_tool_call_approved(
                &rt, &root, &creator, &ctx, None,
            )
            .await
        })
    };
    let (_, responder) = prompts.recv().await.expect("a prompt arrives");
    responder.send(ApprovalAnswer::Denied).unwrap();
    assert_eq!(running.await.unwrap(), DENIED_RESULT);
    assert_eq!(
        checksums(&root),
        before,
        "declining a new file must leave the tree byte-identical"
    );
    assert!(!root.join("new.rs").exists(), "new.rs must not exist");

    std::fs::remove_dir_all(&root).unwrap();
}
