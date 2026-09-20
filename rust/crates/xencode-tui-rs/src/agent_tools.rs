//! Executes the tool calls the model requests in the chat loop:
//! background tasks (Milestone D) and repo insights (Milestone F, F3-02).
//!
//! The [`TaskManager`](xencode_core_rs::TaskManager) lives in an
//! `Arc<tokio::sync::Mutex>` on `App` so the chat loop, later CLI commands
//! and the D2 panel share one registry. Results come back as plain text —
//! readable to the model and cheap to echo into chat.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use xencode_core_rs::{TaskError, TaskManager, TaskRecord};
use xencode_providers_rs::ToolCall;

/// Safety valve: how many assistant→tool→assistant rounds one user turn may
/// take before we stop offering tools and let the model answer.
pub const MAX_TOOL_ROUNDS: usize = 8;

// ── Permission policy (I1-01) ───────────────────────────────────────────
// One source of truth for "may the agent run this call?". The chat loop
// (and later the approval overlay) asks `classify`; nothing else decides.

/// Named values for the `agent_approval` config key (Settings row + CLI).
pub const APPROVAL_MODE_NAMES: &[&str] = &["ask", "edit-allow", "all-allow"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalMode {
    /// Mutating and shell tools prompt; read-only tools run freely.
    Ask,
    /// File edits are auto-approved; shell still prompts.
    EditAllow,
    /// Everything except hard-denied paths is auto-approved.
    AllAllow,
}

impl ApprovalMode {
    /// Unknown values fall back to the strictest mode (theme/layout precedent).
    pub fn parse(name: &str) -> Self {
        match name {
            "edit-allow" => Self::EditAllow,
            "all-allow" => Self::AllAllow,
            _ => Self::Ask,
        }
    }
}

/// What a tool touches; drives both the mode decision and session grants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolClass {
    ReadOnly,
    Edit,
    Shell,
}

/// The outcome of the policy for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// Run it.
    Allow,
    /// Show the user an approval prompt (I1-03).
    Ask,
    /// Never run it, no prompt (outside the workspace, `.git`, config dir).
    Deny,
}

/// What a tool call touches. Unknown tools count as Shell — the executor
/// errors on them anyway, but they are never silently treated as read-only.
pub fn tool_class(tool: &str) -> ToolClass {
    match tool {
        "background_poll" | "repo_advise" | "read_file" | "list_dir" | "search_files" => {
            ToolClass::ReadOnly
        }
        "write_file" | "edit_file" => ToolClass::Edit,
        _ => ToolClass::Shell,
    }
}

/// Lexical path tidy: resolves `.` and `..` without touching the filesystem,
/// so paths that do not exist yet (a file a write tool is about to create)
/// can still be judged. A `..` that pops above the root leaves a path that no
/// longer starts with it, and the descendant check rejects that.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Lexical absolute path (cwd-prefixed if relative); never errors for paths
/// that merely do not exist yet.
fn absolutize(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Whether `raw` (relative paths resolve against `root`) lands inside the
/// workspace and outside the forbidden zones (`.git/`, the config dir).
/// Best-effort lexical check — symlinks are not resolved — which is why
/// in-workspace writes still prompt in `ask` mode instead of running blindly.
pub fn path_allowed(root: &Path, raw: &str) -> bool {
    let root = normalize(&absolutize(root));
    let candidate = Path::new(raw);
    let joined_path = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        root.join(candidate)
    };
    let joined = normalize(&absolutize(&joined_path));
    if !joined.starts_with(&root) {
        return false;
    }
    let relative = joined.strip_prefix(&root).unwrap_or(&joined);
    if relative
        .components()
        .any(|c| matches!(c, std::path::Component::Normal(name) if name == ".git"))
    {
        return false;
    }
    if let Ok(config_dir) = xencode_config_rs::XencodeConfig::config_dir() {
        if joined.starts_with(normalize(&config_dir)) {
            return false;
        }
    }
    true
}

/// The policy decision for one call. `granted` lists classes the user
/// approved for the whole session at an earlier prompt.
pub fn classify(
    root: &Path,
    tool: &str,
    args: &serde_json::Map<String, serde_json::Value>,
    mode: ApprovalMode,
    granted: &[ToolClass],
) -> Permission {
    // Path arguments are hard-denied outside the workspace in every mode:
    // "all-allow" never means "anywhere on disk".
    for key in ["path", "cwd"] {
        if let Some(serde_json::Value::String(raw)) = args.get(key) {
            if !raw.is_empty() && !path_allowed(root, raw) {
                return Permission::Deny;
            }
        }
    }
    let class = tool_class(tool);
    let base = match class {
        ToolClass::ReadOnly => Permission::Allow,
        ToolClass::Edit if mode == ApprovalMode::Ask => Permission::Ask,
        ToolClass::Shell if mode != ApprovalMode::AllAllow => Permission::Ask,
        _ => Permission::Allow,
    };
    if base == Permission::Ask && granted.contains(&class) {
        Permission::Allow
    } else {
        base
    }
}

/// Output lines handed back to the model per poll (the store keeps 500).
const MODEL_OUTPUT_TAIL: usize = 50;

/// Findings handed back to the model per `repo_advise` call.
const MODEL_ADVISE_CAP: usize = 40;

pub type TaskRuntime = Arc<tokio::sync::Mutex<TaskManager>>;

pub fn new_task_runtime() -> TaskRuntime {
    Arc::new(tokio::sync::Mutex::new(TaskManager::new()))
}

/// One-line description of a call for the chat log, args truncated.
pub fn summarize_call(call: &ToolCall) -> String {
    let args = call.arguments_json();
    let args = truncate_one_line(&args, 100);
    format!("{}({args})", call.name)
}

pub async fn execute_tool_call(rt: &TaskRuntime, root: &Path, call: &ToolCall) -> String {
    let args = call.arguments_object();
    match call.name.as_str() {
        "background_start" => {
            let Some(command) = args.get("command").and_then(|v| v.as_str()) else {
                return "error: background_start needs a string \"command\"".to_string();
            };
            let name = args
                .get("name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(command);
            let cwd = args
                .get("cwd")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(std::path::PathBuf::from);
            let mut m = rt.lock().await;
            match m.start_with_cwd(name, command, cwd.as_deref()).await {
                Ok(id) => {
                    let pid = m
                        .snapshot(id)
                        .ok()
                        .and_then(|r| r.pid)
                        .map(|p| p.to_string())
                        .unwrap_or_else(|| "?".into());
                    match &cwd {
                        Some(dir) => format!(
                            "started task {id} (pid {pid}) in {}: {command}",
                            dir.display()
                        ),
                        None => format!("started task {id} (pid {pid}): {command}"),
                    }
                }
                Err(e) => format!("error: {e}"),
            }
        }
        "background_poll" => {
            let Some(id) = arg_id(&args) else {
                return "error: background_poll needs an integer \"id\"".to_string();
            };
            let mut m = rt.lock().await;
            match m.poll(id).await {
                Ok(rec) => render_record(&rec),
                Err(e) => format!("error: {e}"),
            }
        }
        "background_stop" => {
            let Some(id) = arg_id(&args) else {
                return "error: background_stop needs an integer \"id\"".to_string();
            };
            let mut m = rt.lock().await;
            match m.stop(id).await {
                Ok(()) => format!("stopped task {id}"),
                // Killing something already finished is what the caller wanted.
                Err(TaskError::AlreadyFinished(id)) => {
                    format!("task {id} had already finished")
                }
                Err(e) => format!("error: {e}"),
            }
        }
        "repo_advise" => {
            let filter = args
                .get("filter")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            match xencode_context_rs::advise_from_snapshot(root) {
                Ok(mut items) => {
                    if let Some(needle) = filter {
                        items.retain(|a| a.file.contains(needle));
                    }
                    render_advise(&items)
                }
                Err(e) => format!("error: {e}"),
            }
        }
        other => format!("error: unknown tool {other}"),
    }
}

/// Compact, line-capped advice report for the model. The panel/CLI show the
/// same findings with full formatting; this is the token-cheap view.
fn render_advise(items: &[xencode_context_rs::Advice]) -> String {
    if items.is_empty() {
        return "no findings — the symbol graph is clean.".to_string();
    }
    let mut out = format!("{} finding(s):\n", items.len());
    for a in items.iter().take(MODEL_ADVISE_CAP) {
        out.push_str(&format!("{:?}: {}\n", a.kind, a.message));
    }
    if items.len() > MODEL_ADVISE_CAP {
        out.push_str(&format!(
            "… +{} more (call again with a path filter)\n",
            items.len() - MODEL_ADVISE_CAP
        ));
    }
    out.truncate(out.len() - 1);
    out
}

fn arg_id(args: &serde_json::Map<String, serde_json::Value>) -> Option<u64> {
    args.get("id").and_then(|v| match v {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    })
}

fn render_record(rec: &TaskRecord) -> String {
    let mut out = format!(
        "task {} [{}] {}",
        rec.id,
        rec.status.label(),
        truncate_one_line(&rec.command, 100)
    );
    if !rec.output().is_empty() {
        let tail = rec.output().len().saturating_sub(MODEL_OUTPUT_TAIL);
        out.push_str("\noutput:\n");
        for line in &rec.output()[tail..] {
            out.push_str(line);
            out.push('\n');
        }
        out.truncate(out.len() - 1);
    }
    out
}

pub fn truncate_one_line(s: &str, max: usize) -> String {
    let flat = s.replace('\n', " ");
    if flat.chars().count() <= max {
        flat
    } else {
        let cut: String = flat.chars().take(max).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "c1".to_string(),
            name: name.to_string(),
            arguments: args,
        }
    }

    async fn wait_exit(rt: &TaskRuntime, id: u64) -> TaskRecord {
        for _ in 0..100 {
            let rec = rt.lock().await.poll(id).await.unwrap();
            if rec.status != xencode_core_rs::TaskStatus::Running {
                return rec;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("task never exited");
    }

    #[tokio::test]
    async fn start_poll_stop_round_trip() {
        let rt = new_task_runtime();
        let started = execute_tool_call(
            &rt,
            Path::new("."),
            &call(
                "background_start",
                serde_json::json!({"command": "echo hi"}),
            ),
        )
        .await;
        assert!(started.starts_with("started task 1 (pid "), "{started}");
        let mut polled = String::new();
        for _ in 0..100 {
            polled = execute_tool_call(
                &rt,
                Path::new("."),
                &call("background_poll", serde_json::json!({"id": 1})),
            )
            .await;
            if !polled.contains("[running]") && polled.contains("output:") {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(polled.starts_with("task 1 [exited(0)] echo hi"), "{polled}");
        assert!(polled.ends_with("output:\nhi"), "{polled}");
        // Already finished: stop reports it, and killing a live task works too.
        assert_eq!(
            execute_tool_call(
                &rt,
                Path::new("."),
                &call("background_stop", serde_json::json!({"id": 1}))
            )
            .await,
            "task 1 had already finished"
        );
        execute_tool_call(
            &rt,
            Path::new("."),
            &call(
                "background_start",
                serde_json::json!({"name": "sleeper", "command": "sleep 30"}),
            ),
        )
        .await;
        assert_eq!(
            execute_tool_call(
                &rt,
                Path::new("."),
                &call("background_stop", serde_json::json!({"id": "2"}))
            )
            .await,
            "stopped task 2"
        );
        let rec = rt.lock().await.snapshot(2).unwrap();
        assert_eq!(rec.status, xencode_core_rs::TaskStatus::Killed);
    }

    #[tokio::test]
    async fn argument_errors_do_not_panic() {
        let rt = new_task_runtime();
        assert!(execute_tool_call(
            &rt,
            Path::new("."),
            &call("background_start", serde_json::json!({}))
        )
        .await
        .starts_with("error:"));
        assert!(execute_tool_call(
            &rt,
            Path::new("."),
            &call("background_poll", serde_json::json!({"id": "x"}))
        )
        .await
        .starts_with("error:"));
        assert!(
            execute_tool_call(
                &rt,
                Path::new("."),
                &call("background_stop", serde_json::json!({"id": 99}))
            )
            .await
                == "error: no such task: 99"
        );
        assert!(
            execute_tool_call(
                &rt,
                Path::new("."),
                &call("read_file", serde_json::json!({}))
            )
            .await
                == "error: unknown tool read_file"
        );
    }

    #[tokio::test]
    async fn poll_output_tail_is_bounded() {
        let rt = new_task_runtime();
        let cmd = "for i in $(seq 1 60); do echo line$i; done";
        let id = rt.lock().await.start("bulk", cmd).await.unwrap();
        let rec = wait_exit(&rt, id).await;
        // Record itself keeps everything (under the store cap)…
        assert_eq!(rec.output().len(), 60);
        // …but the model-facing render only gets the tail window.
        let text = render_record(&rec);
        assert!(text.contains("line60"));
        assert!(text.contains("line11"), "tail starts 50 lines back");
        assert!(!text.contains("line5\n"));
    }

    #[test]
    fn summarize_truncates_and_flattens() {
        // String-shaped args (OpenAI wire) pass through verbatim, newlines
        // and all — the summary must stay on one line.
        let c = ToolCall {
            id: "c1".to_string(),
            name: "background_start".to_string(),
            arguments: serde_json::Value::String("echo\nhi".to_string()),
        };
        let s = summarize_call(&c);
        assert!(
            s.starts_with("background_start(") && s.contains("echo hi"),
            "{s}"
        );
        let long = call(
            "background_start",
            serde_json::json!({"command": "x".repeat(200)}),
        );
        assert!(summarize_call(&long).chars().count() < 130);
    }

    /// D3-03: an optional `cwd` is passed through and echoed back so the
    /// model knows where the command actually ran.
    #[tokio::test]
    async fn start_with_cwd_reports_the_directory() {
        let rt = new_task_runtime();
        let dir = std::env::temp_dir();
        let started = execute_tool_call(
            &rt,
            Path::new("."),
            &call(
                "background_start",
                serde_json::json!({"command": "pwd", "cwd": dir.display().to_string()}),
            ),
        )
        .await;
        assert!(
            started.contains(&format!("in {}", dir.display())),
            "{started}"
        );
        wait_exit(&rt, 1).await;
        // A nonexistent cwd is an error string, never a panic.
        let bad = execute_tool_call(
            &rt,
            Path::new("."),
            &call(
                "background_start",
                serde_json::json!({"command": "pwd", "cwd": "/definitely/not/here-xyz"}),
            ),
        )
        .await;
        assert!(bad.starts_with("error:"), "{bad}");
    }

    /// F3-02: the model-facing insight tool reads the on-disk snapshot.
    #[tokio::test]
    async fn repo_advise_reads_snapshot_and_honours_the_filter() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "xencode-repo-advise-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "mod a;\nmod b;\n").unwrap();
        std::fs::write(
            root.join("src/a.rs"),
            "use crate::b::bee;\npub fn ay() {}\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/b.rs"),
            "use crate::a::ay;\npub fn bee() {}\n",
        )
        .unwrap();
        std::fs::File::create(root.join("Cargo.toml")).unwrap();
        xencode_context_rs::init_project(
            &root,
            std::sync::Arc::new(AtomicBool::new(false)),
            |_| {},
        )
        .expect("init");

        let rt = new_task_runtime();
        let out = execute_tool_call(&rt, &root, &call("repo_advise", serde_json::json!({}))).await;
        assert!(out.starts_with("1 finding(s):"), "{out}");
        assert!(out.contains("Cycle"), "{out}");
        assert!(out.contains("import cycle"), "{out}");
        // A filter that matches nothing is a clean report, not an error.
        let clean = execute_tool_call(
            &rt,
            &root,
            &call("repo_advise", serde_json::json!({"filter": "nope"})),
        )
        .await;
        assert_eq!(clean, "no findings — the symbol graph is clean.");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn repo_advise_without_index_reports_an_error_string() {
        let rt = new_task_runtime();
        let out = execute_tool_call(
            &rt,
            Path::new("/definitely-not-a-repo-xencode"),
            &call("repo_advise", serde_json::json!({})),
        )
        .await;
        assert!(
            out.starts_with("error:") && out.contains("no project index"),
            "{out}"
        );
    }

    #[test]
    fn render_advise_is_capped_and_tells_the_model_so() {
        use xencode_context_rs::{Advice, AdviceKind};
        let out = render_advise(&[]);
        assert_eq!(out, "no findings — the symbol graph is clean.");

        let items: Vec<Advice> = (0..MODEL_ADVISE_CAP + 5)
            .map(|i| Advice {
                file: format!("src/f{i}.rs"),
                kind: AdviceKind::Orphan,
                message: format!("msg {i}"),
            })
            .collect();
        let out = render_advise(&items);
        assert!(out.starts_with("45 finding(s):\n"), "{out}");
        assert!(out.contains("msg 0"), "{out}");
        assert!(out.contains("msg 39"), "cap boundary is shown");
        assert!(!out.contains("msg 40"), "overflow findings are dropped");
        assert!(
            out.ends_with("… +5 more (call again with a path filter)"),
            "{out}"
        );
    }

    fn args_of(v: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn approval_mode_parses_named_values_and_falls_back_to_ask() {
        assert_eq!(ApprovalMode::parse("ask"), ApprovalMode::Ask);
        assert_eq!(ApprovalMode::parse("edit-allow"), ApprovalMode::EditAllow);
        assert_eq!(ApprovalMode::parse("all-allow"), ApprovalMode::AllAllow);
        assert_eq!(ApprovalMode::parse("yolo"), ApprovalMode::Ask);
        assert_eq!(ApprovalMode::parse(""), ApprovalMode::Ask);
    }

    #[test]
    fn tool_classes_cover_the_registry_and_default_to_shell() {
        assert_eq!(tool_class("repo_advise"), ToolClass::ReadOnly);
        assert_eq!(tool_class("background_poll"), ToolClass::ReadOnly);
        assert_eq!(tool_class("write_file"), ToolClass::Edit);
        assert_eq!(tool_class("edit_file"), ToolClass::Edit);
        assert_eq!(tool_class("background_start"), ToolClass::Shell);
        // Unknown tools are never silently read-only.
        assert_eq!(tool_class("mystery"), ToolClass::Shell);
    }

    #[test]
    fn path_allowed_follows_the_workspace_and_blocks_traversal() {
        let root = Path::new(".");
        assert!(path_allowed(root, "src/app.rs"));
        assert!(path_allowed(root, "./src/../src/lib.rs"));
        assert!(path_allowed(
            root,
            &std::env::current_dir().unwrap().display().to_string()
        ));
        assert!(!path_allowed(root, "../escape"));
        assert!(!path_allowed(root, "/etc/passwd"));
        // Dot-git anywhere below the root is off-limits, even for reads.
        assert!(!path_allowed(root, "x/.git/config"));
        assert!(!path_allowed(root, ".git/HEAD"));
    }

    #[test]
    fn classify_asks_per_mode_and_grants_shortcut_the_prompt() {
        let root = Path::new(".");
        let none = args_of(serde_json::json!({}));
        assert_eq!(
            classify(root, "repo_advise", &none, ApprovalMode::Ask, &[]),
            Permission::Allow
        );
        assert_eq!(
            classify(root, "write_file", &none, ApprovalMode::Ask, &[]),
            Permission::Ask
        );
        assert_eq!(
            classify(root, "write_file", &none, ApprovalMode::EditAllow, &[]),
            Permission::Allow,
            "edit-allow auto-approves file edits"
        );
        assert_eq!(
            classify(
                root,
                "background_start",
                &none,
                ApprovalMode::EditAllow,
                &[]
            ),
            Permission::Ask,
            "edit-allow still prompts for shell"
        );
        assert_eq!(
            classify(root, "background_start", &none, ApprovalMode::AllAllow, &[]),
            Permission::Allow
        );
        assert_eq!(
            classify(
                root,
                "background_start",
                &none,
                ApprovalMode::Ask,
                &[ToolClass::Shell]
            ),
            Permission::Allow,
            "a session grant replaces the prompt for that class"
        );
        assert_eq!(
            classify(
                root,
                "write_file",
                &none,
                ApprovalMode::Ask,
                &[ToolClass::Shell]
            ),
            Permission::Ask,
            "a shell grant must not unlock edits"
        );
    }

    #[test]
    fn classify_denies_out_of_workspace_paths_in_every_mode() {
        let root = std::env::temp_dir().join(format!("xencode-perm-{}", std::process::id()));
        let outside = args_of(serde_json::json!({"path": "../escape.txt"}));
        for mode in [
            ApprovalMode::Ask,
            ApprovalMode::EditAllow,
            ApprovalMode::AllAllow,
        ] {
            assert_eq!(
                classify(&root, "write_file", &outside, mode, &[]),
                Permission::Deny,
                "all-allow never means anywhere on disk ({mode:?})"
            );
        }
        let git = args_of(serde_json::json!({"path": "repo/.git/config"}));
        assert_eq!(
            classify(&root, "edit_file", &git, ApprovalMode::AllAllow, &[]),
            Permission::Deny
        );
        // background_start's cwd gets the same treatment; an in-root cwd does not.
        let bad_cwd = args_of(serde_json::json!({"command": "ls", "cwd": "/etc"}));
        assert_eq!(
            classify(
                &root,
                "background_start",
                &bad_cwd,
                ApprovalMode::AllAllow,
                &[]
            ),
            Permission::Deny
        );
        let good_cwd = args_of(serde_json::json!({"command": "ls", "cwd": "sub/dir"}));
        assert_eq!(
            classify(&root, "background_start", &good_cwd, ApprovalMode::Ask, &[]),
            Permission::Ask
        );
    }
}
