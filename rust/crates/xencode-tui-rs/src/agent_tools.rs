//! Executes the tool calls the model requests in the chat loop:
//! background tasks (Milestone D) and repo insights (Milestone F, F3-02).
//!
//! The [`TaskManager`](xencode_core_rs::TaskManager) lives in an
//! `Arc<tokio::sync::Mutex>` on `App` so the chat loop, later CLI commands
//! and the D2 panel share one registry. Results come back as plain text —
//! readable to the model and cheap to echo into chat.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot};
use xencode_core_rs::{TaskError, TaskManager, TaskRecord};
use xencode_providers_rs::ToolCall;

/// How many assistant→tool→assistant rounds one user turn may take before
/// tools stop being offered and the model must answer in prose. The budget
/// itself is the `agent_max_rounds` config key (default 16, clamped to
/// 1..=64 by the loop); this is the wording the model is taught.
pub const TOOL_HINT: &str = "\n\n## Tools\n\
Available: read_file(path, offset?, limit?), list_dir(path?), search_files(pattern, path?), \
write_file(path, content), edit_file(path, old, new, all?), background_start(command, cwd?, name?), \
background_poll(id), background_stop(id), repo_advise(filter?).\n\
Paths are relative to the project root and must stay inside it; anything outside is refused without asking. \
File writes, edits and shell commands need the user's approval, which they may grant once, allow for the \
session, or deny. If a result begins with `error:`, do not retry that call unchanged - say what failed and \
try a different approach. Prefer edit_file over write_file, and read_file before touching code you have not seen.";

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

/// The user's answer at an approval prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalAnswer {
    Approved,
    ApprovedForSession,
    Denied,
}

impl ApprovalAnswer {
    /// Word for the chat transcript's tool-call record.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Approved => "approved",
            Self::ApprovedForSession => "always allowed",
            Self::Denied => "denied",
        }
    }
}

/// What the approval overlay shows for one pending call. `preview` carries
/// the proposed unified diff (file edits) or the exact command line (shell),
/// already size-capped by [`approval_preview`].
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub tool: String,
    pub class: ToolClass,
    pub summary: String,
    pub preview: String,
}

impl ApprovalRequest {
    pub fn class_label(&self) -> &'static str {
        match self.class {
            ToolClass::ReadOnly => "read-only",
            ToolClass::Edit => "file change",
            ToolClass::Shell => "shell command",
        }
    }
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

/// The workspace root as a normalized absolute path — the anchor every
/// containment check compares against.
fn workspace_root(root: &Path) -> PathBuf {
    normalize(&absolutize(root))
}

/// Resolve a model-supplied path against `root` without touching the disk:
/// absolute paths pass through, relative ones join the normalized root, and
/// `.`/`..` are collapsed lexically so a `..` that escapes leaves a path
/// that no longer starts with the root.
fn resolve_path(root: &Path, raw: &str) -> PathBuf {
    let candidate = Path::new(raw.trim());
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        workspace_root(root).join(candidate)
    };
    normalize(&absolutize(&joined))
}

/// Whether `raw` (relative paths resolve against `root`) lands inside the
/// workspace and outside the forbidden zones (`.git/`, the config dir).
/// Best-effort lexical check — symlinks are not resolved — which is why
/// in-workspace writes still prompt in `ask` mode instead of running blindly.
pub fn path_allowed(root: &Path, raw: &str) -> bool {
    let root = workspace_root(root);
    let joined = resolve_path(root.as_path(), raw);
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
        if joined.starts_with(normalize(&absolutize(&config_dir))) {
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

// ── File tools (I1-02) ──────────────────────────────────────────────────

const READ_DEFAULT_LINES: usize = 200;
const READ_MAX_LINES: usize = 2000;
const LIST_MAX_ENTRIES: usize = 300;
const SEARCH_MAX_HITS: usize = 100;
const SEARCH_MAX_WALK_DEPTH: usize = 24;
const DIFF_MAX_LINES: usize = 80;
/// Directory names the workspace walk always skips (dot-directories are
/// skipped wholesale, these are the common non-dot offenders).
const SEARCH_SKIP_DIRS: &[&str] = &["target", "node_modules", "dist", "build", "venv"];

fn err(msg: impl std::fmt::Display) -> String {
    format!("error: {msg}")
}

fn arg_str<'a>(args: &'a serde_json::Map<String, serde_json::Value>, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

fn arg_usize(args: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<usize> {
    args.get(key).and_then(|v| match v {
        serde_json::Value::Number(n) => n.as_u64().map(|x| x as usize),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    })
}

fn arg_bool(args: &serde_json::Map<String, serde_json::Value>, key: &str) -> bool {
    match args.get(key) {
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::String(s)) => s == "true",
        _ => false,
    }
}

/// Resolve a model-supplied path to an absolute in-workspace path, returning
/// the hard-deny error string when it escapes (outside the root, `.git/`,
/// config dir). This is the executor-side mirror of `classify`'s path rule:
/// until the approval prompt (I1-03) is wired, the file tools still refuse
/// every out-of-workspace byte.
fn workspace_path(root: &Path, raw: &str) -> Result<(PathBuf, String), String> {
    if raw.trim().is_empty() {
        return Err(err("\"path\" must not be empty"));
    }
    if !path_allowed(root, raw) {
        return Err(err(format!(
            "path outside the workspace (or a forbidden directory): {raw}"
        )));
    }
    let full = resolve_path(root, raw);
    Ok((full, raw.trim().to_string()))
}

/// Read a workspace file as text; unreadable, binary and non-UTF-8 files
/// come back as model-actionable error strings, never panics.
fn read_text(path: &Path, display: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| err(format!("cannot read {display}: {e}")))?;
    if bytes.contains(&0) {
        return Err(err(format!("{display} looks like a binary file")));
    }
    String::from_utf8(bytes).map_err(|_| err(format!("{display} is not valid UTF-8")))
}

/// Capped unified diff between old and new content ("" for a new file).
fn unified_diff(old: &str, new: &str) -> String {
    use similar::TextDiff;
    let mut out = String::new();
    let mut lines = 0usize;
    for hunk in TextDiff::from_lines(old, new).unified_diff().iter_hunks() {
        for line in hunk.to_string().lines() {
            if lines >= DIFF_MAX_LINES {
                out.push_str("… diff truncated\n");
                return out;
            }
            out.push_str(line);
            out.push('\n');
            lines += 1;
        }
    }
    out
}

fn tool_read_file(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let Some(raw) = arg_str(args, "path") else {
        return err("read_file needs a string \"path\"");
    };
    let (full, display) = match workspace_path(root, raw) {
        Ok(ok) => ok,
        Err(e) => return e,
    };
    let text = match read_text(&full, &display) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return format!("{display}: empty file");
    }
    let offset = arg_usize(args, "offset").unwrap_or(1).max(1);
    let limit = arg_usize(args, "limit")
        .unwrap_or(READ_DEFAULT_LINES)
        .clamp(1, READ_MAX_LINES);
    if offset > lines.len() {
        return err(format!(
            "offset {offset} is past the end of {display} ({} lines)",
            lines.len()
        ));
    }
    let end = lines.len().min(offset - 1 + limit);
    let mut out = String::new();
    for (i, line) in lines[offset - 1..end].iter().enumerate() {
        out.push_str(&format!("{}\t{line}\n", offset + i));
    }
    if end < lines.len() {
        out.push_str(&format!(
            "… lines {offset}–{end} of {}, pass offset={} for the next page\n",
            lines.len(),
            end + 1
        ));
    }
    out.truncate(out.len() - 1);
    out
}

fn tool_list_dir(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let raw = arg_str(args, "path").filter(|s| !s.trim().is_empty());
    let (full, display) = match raw {
        Some(raw) => match workspace_path(root, raw) {
            Ok(ok) => ok,
            Err(e) => return e,
        },
        None => (workspace_root(root), ".".to_string()),
    };
    let Ok(entries) = std::fs::read_dir(&full) else {
        return err(format!("cannot list {display}: is it a directory?"));
    };
    let mut names: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let name = entry.file_name().to_string_lossy().into_owned();
        names.push(if is_dir { format!("{name}/") } else { name });
    }
    names.sort();
    let total = names.len();
    if total > LIST_MAX_ENTRIES {
        names.truncate(LIST_MAX_ENTRIES);
        names.push(format!("… +{} more entries", total - LIST_MAX_ENTRIES));
    }
    if names.is_empty() {
        return format!("{display}: empty");
    }
    format!("{display}:\n{}", names.join("\n"))
}

fn is_skipped_dir(name: &str) -> bool {
    name.starts_with('.') || SEARCH_SKIP_DIRS.contains(&name)
}

/// Collect the files a search walks, bounded in depth; symlinks are never
/// followed (DirEntry types report the link itself, so cycles cannot occur).
fn walk_files(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > SEARCH_MAX_WALK_DEPTH || out.len() >= 20_000 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            if !is_skipped_dir(&entry.file_name().to_string_lossy()) {
                walk_files(&entry.path(), depth + 1, out);
            }
        } else if file_type.is_file() {
            out.push(entry.path());
        }
    }
}

fn relative_display(root: &Path, path: &Path) -> String {
    normalize(path)
        .strip_prefix(workspace_root(root))
        .map(|rel| rel.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
}

fn search_one_file(
    full: &Path,
    display: &str,
    re: &regex::Regex,
    hits: &mut Vec<String>,
) -> Result<(), String> {
    let Ok(bytes) = std::fs::read(full) else {
        return Ok(()); // unreadable files are silently out of scope
    };
    if bytes.contains(&0) {
        return Ok(());
    }
    let Ok(text) = String::from_utf8(bytes) else {
        return Ok(());
    };
    for (i, line) in text.lines().enumerate() {
        if re.is_match(line) {
            let excerpt: String = line.trim_start().chars().take(200).collect();
            hits.push(format!("{display}:{}:{excerpt}", i + 1));
            if hits.len() >= SEARCH_MAX_HITS {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn tool_search_files(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let Some(pattern) = arg_str(args, "pattern") else {
        return err("search_files needs a string \"pattern\"");
    };
    let Ok(re) = regex::Regex::new(pattern) else {
        return err(format!("invalid regular expression: {pattern}"));
    };
    let raw = arg_str(args, "path").filter(|s| !s.trim().is_empty());
    let scope = match raw {
        Some(raw) => match workspace_path(root, raw) {
            Ok((full, _)) => full,
            Err(e) => return e,
        },
        None => workspace_root(root),
    };
    let mut hits: Vec<String> = Vec::new();
    if scope.is_file() {
        let display = relative_display(root, &scope);
        let _ = search_one_file(&scope, &display, &re, &mut hits);
    } else {
        let mut files = Vec::new();
        walk_files(&scope, 0, &mut files);
        files.sort();
        for file in files {
            let display = relative_display(root, &file);
            let _ = search_one_file(&file, &display, &re, &mut hits);
            if hits.len() >= SEARCH_MAX_HITS {
                break;
            }
        }
    }
    if hits.is_empty() {
        return format!("no matches for /{pattern}/");
    }
    let mut out = format!("{} match(es):\n", hits.len());
    out.push_str(&hits.join("\n"));
    if hits.len() >= SEARCH_MAX_HITS {
        out.push_str(&format!(
            "\n… {SEARCH_MAX_HITS}-hit cap reached — narrow the pattern or path"
        ));
    }
    out
}

fn tool_write_file(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let Some(raw) = arg_str(args, "path") else {
        return err("write_file needs a string \"path\"");
    };
    let Some(content) = arg_str(args, "content") else {
        return err("write_file needs a string \"content\"");
    };
    let (full, display) = match workspace_path(root, raw) {
        Ok(ok) => ok,
        Err(e) => return e,
    };
    if full.is_dir() {
        return err(format!("{display} is a directory"));
    }
    let existed = full.exists();
    let old = if existed {
        match read_text(&full, &display) {
            Ok(t) => t,
            Err(e) => return e,
        }
    } else {
        String::new()
    };
    if let Some(parent) = full.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return err(format!("cannot create {}: {e}", parent.display()));
        }
    }
    if let Err(e) = std::fs::write(&full, content) {
        return err(format!("cannot write {display}: {e}"));
    }
    let diff = unified_diff(&old, content);
    let body = if diff.is_empty() {
        "no textual change".to_string()
    } else {
        diff.trim_end().to_string()
    };
    format!(
        "{} {display} ({} line(s))\n{body}",
        if existed { "updated" } else { "created" },
        content.lines().count()
    )
}

/// Result string fed back to the model when the user denies at the prompt.
/// Wording matters: weak models otherwise retry the identical call forever.
pub const DENIED_RESULT: &str = "error: the user denied this action. Do not retry it unchanged — explain, adjust, or ask the user.";

/// Result string fed back to the model for a policy-denied call.
pub const FORBIDDEN_RESULT: &str =
    "error: refused by the permission policy (path outside the allowed workspace).";

/// One-line label for the approval overlay: tool + its focus argument.
pub fn approval_summary(call: &ToolCall) -> String {
    let args = call.arguments_object();
    let focus = arg_str(&args, "path")
        .or_else(|| arg_str(&args, "command"))
        .or_else(|| arg_str(&args, "pattern"))
        .unwrap_or("");
    if focus.is_empty() {
        call.name.clone()
    } else {
        truncate_one_line(&format!("{} {}", call.name, focus), 90)
    }
}

/// The overlay's body: the concrete bytes at stake. File writes/edits get
/// the exact unified diff of the proposed change (computed on a snapshot of
/// the current file — the call re-reads at execution, so the real change
/// could differ if the file is edited mid-prompt); shell tools get the
/// literal command line; everything else gets the argument summary.
pub fn approval_preview(root: &Path, call: &ToolCall) -> String {
    let args = call.arguments_object();
    let diff_for = |path: &str, new_text: &str| -> String {
        let (full, display) = match workspace_path(root, path) {
            Ok(ok) => ok,
            Err(e) => return e,
        };
        let old = if full.exists() {
            match read_text(&full, &display) {
                Ok(t) => t,
                Err(e) => return e,
            }
        } else {
            String::new()
        };
        let header = if full.exists() {
            format!("target: {display}")
        } else {
            format!("target: {display} (new file)")
        };
        format!("{header}\n{}", unified_diff(&old, new_text).trim_end())
    };
    match call.name.as_str() {
        "write_file" => match (arg_str(&args, "path"), arg_str(&args, "content")) {
            (Some(p), Some(c)) => diff_for(p, c),
            _ => summarize_call(call),
        },
        "edit_file" => {
            let (Some(p), Some(old), Some(new)) = (
                arg_str(&args, "path"),
                arg_str(&args, "old"),
                arg_str(&args, "new"),
            ) else {
                return summarize_call(call);
            };
            if old.is_empty() {
                return summarize_call(call);
            }
            let (full, _) = match workspace_path(root, p) {
                Ok(ok) => ok,
                Err(e) => return e,
            };
            let Ok(current) = read_text(&full, p) else {
                return summarize_call(call);
            };
            let count = current.matches(old).count();
            let updated = if arg_bool(&args, "all") {
                current.replace(old, new)
            } else {
                current.replacen(old, new, 1)
            };
            let mut preview = diff_for(p, &updated);
            if count != 1 && !arg_bool(&args, "all") {
                preview.push_str(&format!(
                    "\n(note: \"old\" currently matches {count} times — the edit \
                     would fail unless all=true)"
                ));
            }
            preview
        }
        "background_start" => match arg_str(&args, "command") {
            Some(command) => format!("command: sh -c {command:?}"),
            None => summarize_call(call),
        },
        _ => summarize_call(call),
    }
}

fn tool_edit_file(root: &Path, args: &serde_json::Map<String, serde_json::Value>) -> String {
    let Some(raw) = arg_str(args, "path") else {
        return err("edit_file needs a string \"path\"");
    };
    let Some(old) = arg_str(args, "old") else {
        return err("edit_file needs a string \"old\"");
    };
    let Some(new) = arg_str(args, "new") else {
        return err("edit_file needs a string \"new\"");
    };
    if old.is_empty() {
        return err("\"old\" must not be empty (use write_file to create content)");
    }
    let (full, display) = match workspace_path(root, raw) {
        Ok(ok) => ok,
        Err(e) => return e,
    };
    let text = match read_text(&full, &display) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let count = text.matches(old).count();
    let replace_all = arg_bool(args, "all");
    if count == 0 {
        return err(format!(
            "\"old\" not found in {display} — read_file it first and copy the \
             exact text (including indentation)"
        ));
    }
    if count > 1 && !replace_all {
        return err(format!(
            "\"old\" appears {count} times in {display} — pass more context to \
             make it unique, or all=true to replace every occurrence"
        ));
    }
    let updated = if replace_all {
        text.replace(old, new)
    } else {
        text.replacen(old, new, 1)
    };
    if let Err(e) = std::fs::write(&full, &updated) {
        return err(format!("cannot write {display}: {e}"));
    }
    let n = if replace_all { count } else { 1 };
    let diff = unified_diff(&text, &updated);
    format!("edited {display}: replaced {n} occurrence(s)\n{diff}")
        .trim_end()
        .to_string()
}

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
                // Resolve against the workspace so a relative cwd means "in
                // the project", not "in the directory xencode was started
                // from" — the registry spawns relative to the process dir.
                .map(|raw| resolve_path(root, raw));
            if let Some(dir) = &cwd {
                if !path_allowed(root, &dir.to_string_lossy()) {
                    return err(format!(
                        "cwd outside the workspace (or a forbidden directory): {}",
                        dir.display()
                    ));
                }
            }
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
        "read_file" => tool_read_file(root, &args),
        "list_dir" => tool_list_dir(root, &args),
        "search_files" => tool_search_files(root, &args),
        "write_file" => tool_write_file(root, &args),
        "edit_file" => tool_edit_file(root, &args),
        other => format!("error: unknown tool {other}"),
    }
}

/// What the loop needs to gate a call before running it: the configured
/// mode, the session grants — shared with `App` so an "always allow" answer
/// is still in force for the next message — and the channel the approval
/// overlay drains each frame.
#[derive(Clone)]
pub struct ApprovalCtx {
    pub mode: ApprovalMode,
    pub grants: Arc<std::sync::Mutex<Vec<ToolClass>>>,
    pub prompts: mpsc::UnboundedSender<(ApprovalRequest, oneshot::Sender<ApprovalAnswer>)>,
}

impl ApprovalCtx {
    fn granted(&self) -> Vec<ToolClass> {
        self.grants
            .lock()
            .map(|grants| grants.clone())
            .unwrap_or_default()
    }

    fn grant(&self, class: ToolClass) {
        if let Ok(mut grants) = self.grants.lock() {
            if !grants.contains(&class) {
                grants.push(class);
            }
        }
    }
}

/// The loop's entry point (I1-04): policy first, prompt if the policy says
/// `Ask`, execute only on a yes. The result string is always something the
/// model can act on — a denial is stated as a denial.
pub async fn execute_tool_call_approved(
    rt: &TaskRuntime,
    root: &Path,
    call: &ToolCall,
    ctx: &ApprovalCtx,
) -> String {
    let args = call.arguments_object();
    match classify(root, &call.name, &args, ctx.mode, &ctx.granted()) {
        // Refused without asking: the path is outside what the agent may
        // touch in any mode, so a prompt would only invite a mistake.
        Permission::Deny => FORBIDDEN_RESULT.to_string(),
        Permission::Allow => execute_tool_call(rt, root, call).await,
        Permission::Ask => {
            let class = tool_class(&call.name);
            let request = ApprovalRequest {
                tool: call.name.clone(),
                class,
                summary: approval_summary(call),
                preview: approval_preview(root, call),
            };
            let (responder, answer) = oneshot::channel();
            if ctx.prompts.send((request, responder)).is_err() {
                // Nothing is listening — no TUI attached. The strictest
                // possible answer is the only honest one.
                return DENIED_RESULT.to_string();
            }
            match answer.await {
                Ok(ApprovalAnswer::Approved) => execute_tool_call(rt, root, call).await,
                Ok(ApprovalAnswer::ApprovedForSession) => {
                    ctx.grant(class);
                    execute_tool_call(rt, root, call).await
                }
                // A dropped responder means the prompt vanished with the app.
                Ok(ApprovalAnswer::Denied) | Err(_) => DENIED_RESULT.to_string(),
            }
        }
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
                &call("mystery_tool", serde_json::json!({}))
            )
            .await
                == "error: unknown tool mystery_tool"
        );
        assert!(execute_tool_call(
            &rt,
            Path::new("."),
            &call("read_file", serde_json::json!({}))
        )
        .await
        .starts_with("error: read_file needs"));
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
    /// model knows where the command actually ran. I1-02: the cwd must sit
    /// inside the workspace — an out-of-root cwd is refused by the executor
    /// itself, not just by `classify`.
    #[tokio::test]
    async fn start_with_cwd_reports_the_directory() {
        let root = std::env::temp_dir().join(format!("xencode-cwd-test-{}", std::process::id()));
        std::fs::create_dir_all(root.join("sub")).unwrap();
        let rt = new_task_runtime();
        let started = execute_tool_call(
            &rt,
            &root,
            &call(
                "background_start",
                serde_json::json!({"command": "pwd", "cwd": "sub"}),
            ),
        )
        .await;
        assert!(
            started.contains(&format!("in {}", root.join("sub").display())),
            "{started}"
        );
        wait_exit(&rt, 1).await;
        // A nonexistent (but in-root) cwd is an error string, never a panic.
        let bad = execute_tool_call(
            &rt,
            &root,
            &call(
                "background_start",
                serde_json::json!({"command": "pwd", "cwd": "definitely/not/here-xyz"}),
            ),
        )
        .await;
        assert!(bad.starts_with("error:"), "{bad}");
        // An out-of-workspace cwd is refused without ever spawning.
        let outside = execute_tool_call(
            &rt,
            &root,
            &call(
                "background_start",
                serde_json::json!({"command": "pwd", "cwd": "/etc"}),
            ),
        )
        .await;
        assert!(outside.contains("outside the workspace"), "{outside}");
        let _ = std::fs::remove_dir_all(&root);
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

    fn temp_root(label: &str) -> PathBuf {
        use std::sync::atomic::AtomicUsize;
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "xencode-filetools-{label}-{}-{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn read_file_numbers_pages_and_rejects_bad_input() {
        let root = temp_root("read");
        let body: String = (1..=300).map(|i| format!("line{i}\n")).collect::<String>();
        std::fs::write(root.join("big.txt"), &body).unwrap();

        let out = tool_read_file(&root, &args_of(serde_json::json!({"path": "big.txt"})));
        assert!(
            out.starts_with("1\tline1\n"),
            "{}",
            &out[..40.min(out.len())]
        );
        assert!(out.contains("200\tline200"));
        assert!(!out.contains("201\tline201"));
        assert!(out.contains("pass offset=201 for the next page"), "{out}");

        let paged = tool_read_file(
            &root,
            &args_of(serde_json::json!({"path": "big.txt", "offset": 250})),
        );
        assert!(paged.starts_with("250\tline250"));
        assert!(paged.contains("300\tline300"));
        assert!(!paged.contains("next page"), "last page has no pointer");

        // Out-of-workspace and traversal attempts are refused before any I/O.
        for path in ["../escape", "/etc/passwd", ".git/config"] {
            let denied = tool_read_file(&root, &args_of(serde_json::json!({"path": path})));
            assert!(
                denied.starts_with("error:") && denied.contains("outside"),
                "{denied}"
            );
        }
        // Binary and missing files are error strings, not panics.
        std::fs::write(root.join("bin.dat"), [b'a', 0, b'b']).unwrap();
        let bin = tool_read_file(&root, &args_of(serde_json::json!({"path": "bin.dat"})));
        assert!(bin.contains("binary"), "{bin}");
        let missing = tool_read_file(&root, &args_of(serde_json::json!({"path": "nope.txt"})));
        assert!(missing.starts_with("error: cannot read"), "{missing}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn list_dir_sorts_marks_dirs_and_caps() {
        let root = temp_root("list");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("a.txt"), "x").unwrap();
        let out = tool_list_dir(&root, &args_of(serde_json::json!({})));
        assert!(out.starts_with(".:\n"), "{out}");
        assert!(out.contains("a.txt\n") && out.contains("src/"), "{out}");
        assert!(
            out.find("a.txt").unwrap() < out.find("src/").unwrap(),
            "sorted: {out}"
        );
        let file_as_dir = tool_list_dir(&root, &args_of(serde_json::json!({"path": "a.txt"})));
        assert!(file_as_dir.starts_with("error:"), "{file_as_dir}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn search_files_matches_scopes_and_caps() {
        let root = temp_root("search");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("src/a.rs"), "fn main() {}\n// fn main again\n").unwrap();
        std::fs::write(root.join("target/leak.rs"), "fn main() {}\n").unwrap();
        let out = tool_search_files(
            &root,
            &args_of(serde_json::json!({"pattern": r"fn\s+main"})),
        );
        assert!(out.contains("src/a.rs:1:fn main() {}"), "{out}");
        assert!(out.contains("src/a.rs:2:// fn main again"), "{out}");
        assert!(!out.contains("target"), "ignored dir leaked: {out}");
        assert!(out.starts_with("2 match(es)"), "{out}");

        let scoped = tool_search_files(
            &root,
            &args_of(serde_json::json!({"pattern": "again", "path": "src"})),
        );
        assert!(
            scoped.contains("1 match(es)") && scoped.contains("src/a.rs:2"),
            "{scoped}"
        );
        assert!(
            tool_search_files(&root, &args_of(serde_json::json!({"pattern": "zzz-nope"})))
                .starts_with("no matches"),
        );
        let bad_re = tool_search_files(&root, &args_of(serde_json::json!({"pattern": "("})));
        assert!(bad_re.contains("invalid regular expression"), "{bad_re}");

        // Hit cap: 150 matches report the cap and stop.
        std::fs::write(root.join("src/many.txt"), "hit\n".repeat(150)).unwrap();
        let capped = tool_search_files(&root, &args_of(serde_json::json!({"pattern": "^hit$"})));
        assert!(capped.contains("100 match(es)"), "{capped}");
        assert!(capped.ends_with("narrow the pattern or path"), "{capped}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn write_file_creates_pages_and_returns_diffs() {
        let root = temp_root("write");
        let out = tool_write_file(
            &root,
            &args_of(serde_json::json!({"path": "deep/dir/new.txt", "content": "hello\nworld\n"})),
        );
        assert!(
            out.starts_with("created deep/dir/new.txt (2 line(s))"),
            "{out}"
        );
        assert!(out.contains("+hello") && out.contains("+world"), "{out}");
        assert_eq!(
            std::fs::read_to_string(root.join("deep/dir/new.txt")).unwrap(),
            "hello\nworld\n"
        );

        let again = tool_write_file(
            &root,
            &args_of(serde_json::json!({"path": "deep/dir/new.txt", "content": "hello\nmars\n"})),
        );
        assert!(again.starts_with("updated"), "{again}");
        assert!(
            again.contains("-world") && again.contains("+mars"),
            "{again}"
        );

        let denied = tool_write_file(
            &root,
            &args_of(serde_json::json!({"path": "../../oops", "content": "x"})),
        );
        assert!(denied.contains("outside the workspace"), "{denied}");
        assert!(!Path::new("/oops").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn edit_file_requires_unique_matches_or_all() {
        let root = temp_root("edit");
        std::fs::write(root.join("code.rs"), "alpha\nbeta\nalpha\n").unwrap();

        let dup = tool_edit_file(
            &root,
            &args_of(serde_json::json!({"path": "code.rs", "old": "alpha", "new": "gamma"})),
        );
        assert!(
            dup.contains("appears 2 times") && dup.contains("all=true"),
            "{dup}"
        );
        // The failed edit must not touch the file.
        assert_eq!(
            std::fs::read_to_string(root.join("code.rs")).unwrap(),
            "alpha\nbeta\nalpha\n"
        );

        let missing = tool_edit_file(
            &root,
            &args_of(serde_json::json!({"path": "code.rs", "old": "zzz", "new": "q"})),
        );
        assert!(
            missing.contains("not found") && missing.contains("read_file"),
            "{missing}"
        );

        let every = tool_edit_file(
            &root,
            &args_of(
                serde_json::json!({"path": "code.rs", "old": "alpha", "new": "gamma", "all": true}),
            ),
        );
        assert!(
            every.starts_with("edited code.rs: replaced 2 occurrence(s)"),
            "{every}"
        );
        assert!(
            every.contains("-alpha") && every.contains("+gamma"),
            "{every}"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("code.rs")).unwrap(),
            "gamma\nbeta\ngamma\n"
        );

        std::fs::write(root.join("code.rs"), "gamma\nbeta\ngamma\n").unwrap();
        let unique = tool_edit_file(
            &root,
            &args_of(serde_json::json!({"path": "code.rs", "old": "beta", "new": "delta"})),
        );
        assert!(
            unique.starts_with("edited code.rs: replaced 1 occurrence(s)"),
            "{unique}"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn file_tools_route_through_execute_tool_call() {
        let root = temp_root("route");
        let written = execute_tool_call(
            &new_task_runtime(),
            &root,
            &call(
                "write_file",
                serde_json::json!({"path": "r.txt", "content": "ok\n"}),
            ),
        )
        .await;
        assert!(written.starts_with("created r.txt"), "{written}");
        let read = execute_tool_call(
            &new_task_runtime(),
            &root,
            &call("read_file", serde_json::json!({"path": "r.txt"})),
        )
        .await;
        assert_eq!(read, "1\tok");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn approval_summary_focuses_the_argument_that_matters() {
        assert_eq!(
            approval_summary(&call(
                "write_file",
                serde_json::json!({"path": "src/lib.rs", "content": "x"}),
            )),
            "write_file src/lib.rs"
        );
        assert_eq!(
            approval_summary(&call(
                "background_start",
                serde_json::json!({"command": "cargo test"})
            )),
            "background_start cargo test"
        );
        // No recognisable focus argument → bare tool name, and long lines
        // stay on one screen row.
        assert_eq!(
            approval_summary(&call("repo_advise", serde_json::json!({"filter": "all"}))),
            "repo_advise"
        );
        let long = approval_summary(&call(
            "search_files",
            serde_json::json!({"pattern": "x".repeat(200)}),
        ));
        // truncate_one_line keeps `max` chars and appends the ellipsis.
        assert!(long.chars().count() <= 91, "{long}");
    }

    #[test]
    fn approval_preview_shows_the_exact_diff_for_write_and_edit() {
        let root = temp_root("preview");
        std::fs::write(root.join("code.rs"), "fn a() {}\nfn b() {}\n").unwrap();
        let write = approval_preview(
            &root,
            &call(
                "write_file",
                serde_json::json!({"path": "code.rs", "content": "fn a() {}\nfn c() {}\n"}),
            ),
        );
        assert!(write.contains("target: code.rs"), "{write}");
        assert!(write.contains("-fn b() {}"), "{write}");
        assert!(write.contains("+fn c() {}"), "{write}");

        let fresh = approval_preview(
            &root,
            &call(
                "write_file",
                serde_json::json!({"path": "new/mod.rs", "content": "pub fn n() {}\n"}),
            ),
        );
        assert!(fresh.contains("(new file)"), "{fresh}");

        let edit = approval_preview(
            &root,
            &call(
                "edit_file",
                serde_json::json!({"path": "code.rs", "old": "fn a", "new": "fn z", "all": true}),
            ),
        );
        assert!(edit.contains("-fn a"), "{edit}");
        assert!(edit.contains("+fn z"), "{edit}");

        // An ambiguous edit is the one case the diff alone cannot show: the
        // preview must say the old text matches twice.
        std::fs::write(root.join("dup.rs"), "x\nx\n").unwrap();
        let dup = approval_preview(
            &root,
            &call(
                "edit_file",
                serde_json::json!({"path": "dup.rs", "old": "x", "new": "y"}),
            ),
        );
        assert!(dup.contains("2 times"), "{dup}");

        // Out-of-workspace paths are refused in the preview, not displayed
        // as if they were editable.
        let outside = approval_preview(
            &root,
            &call(
                "write_file",
                serde_json::json!({"path": "../../etc/pwned", "content": "x"}),
            ),
        );
        assert!(outside.contains("outside the workspace"), "{outside}");

        // Shell tools show the command line, no diff.
        let shell = approval_preview(
            &root,
            &call(
                "background_start",
                serde_json::json!({"command": "rm -rf /"}),
            ),
        );
        assert!(shell.contains("rm -rf /"), "{shell}");
        assert!(!shell.contains("@@"), "{shell}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The loop's gating path (I1-04), driven without a provider: these
    /// tests play the user at the overlay by answering the oneshot the
    /// prompt carried.
    struct Harness {
        ctx: ApprovalCtx,
        prompts: mpsc::UnboundedReceiver<(ApprovalRequest, oneshot::Sender<ApprovalAnswer>)>,
    }

    fn harness(mode: ApprovalMode) -> Harness {
        let (tx, rx) = mpsc::unbounded_channel();
        Harness {
            ctx: ApprovalCtx {
                mode,
                grants: Arc::new(std::sync::Mutex::new(Vec::new())),
                prompts: tx,
            },
            prompts: rx,
        }
    }

    fn write_call(path: &str, content: &str) -> ToolCall {
        call(
            "write_file",
            serde_json::json!({"path": path, "content": content}),
        )
    }

    fn bg_call(command: &str) -> ToolCall {
        call("background_start", serde_json::json!({"command": command}))
    }

    /// Run one gated call to completion, answering its prompt (if it raises
    /// one) with `answer`. The call runs on its own task because on this
    /// single-threaded test runtime an unpolled future would never get as
    /// far as sending the prompt we are waiting to receive.
    async fn gated(
        rt: TaskRuntime,
        root: PathBuf,
        tool: ToolCall,
        ctx: ApprovalCtx,
        prompts: &mut mpsc::UnboundedReceiver<(ApprovalRequest, oneshot::Sender<ApprovalAnswer>)>,
        answer: ApprovalAnswer,
    ) -> String {
        let running =
            tokio::spawn(async move { execute_tool_call_approved(&rt, &root, &tool, &ctx).await });
        tokio::pin!(running);
        tokio::select! {
            maybe_prompt = prompts.recv() => {
                // None here means the receiver was dropped, i.e. no UI is
                // listening: the loop must treat that as a denial, which is
                // what it does, so there is nothing to answer.
                if let Some((_request, responder)) = maybe_prompt {
                    responder.send(answer).unwrap();
                }
            }
            finished = &mut running => return finished.unwrap(),
        }
        running.await.unwrap()
    }

    #[tokio::test]
    async fn ask_mode_prompts_before_a_write_and_runs_after_a_yes() {
        let root = temp_root("gate-ask");
        let h = harness(ApprovalMode::Ask);
        let mut prompts = h.prompts;
        let result = gated(
            new_task_runtime(),
            root.clone(),
            write_call("hello.txt", "hi\n"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Approved,
        )
        .await;
        assert!(result.starts_with("created hello.txt"), "{result}");
        assert_eq!(
            std::fs::read_to_string(root.join("hello.txt")).unwrap(),
            "hi\n"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn a_denied_write_touches_nothing_and_says_so_to_the_model() {
        let root = temp_root("gate-deny");
        std::fs::write(root.join("keep.txt"), "original\n").unwrap();
        let h = harness(ApprovalMode::Ask);
        let mut prompts = h.prompts;
        let result = gated(
            new_task_runtime(),
            root.clone(),
            write_call("keep.txt", "overwritten\n"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
        )
        .await;
        assert_eq!(result, DENIED_RESULT);
        assert_eq!(
            std::fs::read_to_string(root.join("keep.txt")).unwrap(),
            "original\n"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn allow_for_session_stops_prompting_that_class_but_not_other_classes() {
        let root = temp_root("gate-session");
        let rt = new_task_runtime();
        let h = harness(ApprovalMode::Ask);
        let mut prompts = h.prompts;

        let first = gated(
            rt.clone(),
            root.clone(),
            write_call("a.txt", "a\n"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::ApprovedForSession,
        )
        .await;
        assert!(first.starts_with("created a.txt"), "{first}");

        // Same class now auto-allows: no prompt, so `gated` just runs.
        let second = gated(
            rt.clone(),
            root.clone(),
            write_call("b.txt", "b\n"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
        )
        .await;
        assert!(second.starts_with("created b.txt"), "{second}");
        assert!(
            prompts.try_recv().is_err(),
            "an allowed class must not keep asking"
        );
        assert_eq!(
            h.ctx.grants.lock().unwrap().as_slice(),
            &[ToolClass::Edit],
            "the grant lives in the shared list, so the next turn inherits it"
        );

        // Shell is a separate class: granting edits says nothing about it.
        let shell = gated(
            rt.clone(),
            root.clone(),
            bg_call("true"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
        )
        .await;
        assert_eq!(shell, DENIED_RESULT);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn read_only_tools_run_without_a_prompt_and_policy_denies_never_ask() {
        let root = temp_root("gate-readonly");
        std::fs::write(root.join("r.txt"), "readable\n").unwrap();
        let rt = new_task_runtime();
        let h = harness(ApprovalMode::Ask);
        let mut prompts = h.prompts;

        let read = gated(
            rt.clone(),
            root.clone(),
            call("read_file", serde_json::json!({"path": "r.txt"})),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
        )
        .await;
        assert_eq!(read, "1\treadable");
        assert!(prompts.try_recv().is_err(), "reads must not prompt");

        // A path outside the workspace is refused outright in every mode:
        // prompting would offer the user something policy already forbids.
        let outside = gated(
            rt.clone(),
            root.clone(),
            write_call("../outside-of-root.txt", "x\n"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Approved,
        )
        .await;
        assert_eq!(outside, FORBIDDEN_RESULT);
        assert!(prompts.try_recv().is_err(), "a hard deny must not prompt");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn edit_allow_mode_writes_silently_but_shell_still_asks() {
        let root = temp_root("gate-editallow");
        let rt = new_task_runtime();
        let h = harness(ApprovalMode::EditAllow);
        let mut prompts = h.prompts;

        let written = gated(
            rt.clone(),
            root.clone(),
            write_call("q.txt", "q\n"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
        )
        .await;
        assert!(written.starts_with("created q.txt"), "{written}");
        assert!(prompts.try_recv().is_err());

        // Answering "nothing" (dropping the responder) is a denial too, never
        // an implicit yes.
        let shell = gated(
            rt.clone(),
            root.clone(),
            bg_call("true"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
        )
        .await;
        assert_eq!(shell, DENIED_RESULT);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn with_no_ui_listening_a_gated_call_is_denied_not_run() {
        let root = temp_root("gate-noui");
        let h = harness(ApprovalMode::Ask);
        drop(h.prompts);
        let result = execute_tool_call_approved(
            &new_task_runtime(),
            &root,
            &write_call("z.txt", "z\n"),
            &h.ctx,
        )
        .await;
        assert_eq!(result, DENIED_RESULT);
        assert!(
            !root.join("z.txt").exists(),
            "nothing may be written unasked"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
