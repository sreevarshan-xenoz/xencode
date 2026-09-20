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
write_file(path, content), edit_file(path, old, new, all?), run_command(command), \
background_start(command, cwd?, name?), \
background_poll(id), background_stop(id), update_plan(items), repo_advise(filter?).\n\
Paths are relative to the project root and must stay inside it; anything outside is refused without asking. \
File writes, edits and shell commands need the user's approval, which they may grant once, allow for the \
session, or deny. If a result begins with `error:`, do not retry that call unchanged - say what failed and \
try a different approach. Prefer edit_file over write_file, and read_file before touching code you have not seen.\n\
For anything that takes several steps, post a short plan with update_plan(items=[{text,status}]) before the \
first edit and update the statuses as you go; the user watches that list.";

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
        "background_poll" | "repo_advise" | "read_file" | "list_dir" | "search_files"
        | "update_plan" => ToolClass::ReadOnly,
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

/// How a call ended, for the surfaces that show progress (I2-04). The result
/// string is the only record the loop keeps, so this reads it back rather
/// than re-deriving the policy: an `error:` from an executor is a failure, the
/// two constants above are refusals.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CallOutcome {
    Finished,
    Denied,
    Refused,
    Failed,
}

impl CallOutcome {
    /// The word a step row shows.
    pub fn label(self) -> &'static str {
        match self {
            CallOutcome::Finished => "done",
            CallOutcome::Denied => "denied",
            CallOutcome::Refused => "refused",
            CallOutcome::Failed => "failed",
        }
    }
}

pub fn call_outcome(result: &str) -> CallOutcome {
    if result == DENIED_RESULT {
        CallOutcome::Denied
    } else if result == FORBIDDEN_RESULT {
        CallOutcome::Refused
    } else if result.starts_with("error:") {
        CallOutcome::Failed
    } else {
        CallOutcome::Finished
    }
}

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
        "background_start" | "run_command" => match arg_str(&args, "command") {
            Some(command) if !command.trim().is_empty() => format!("command: sh -c {command:?}"),
            _ => summarize_call(call),
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

/// Foreground-command budget when nothing is configured (I2-02). The real
/// value comes from `agent_command_timeout`; this is the fallback.
pub const DEFAULT_COMMAND_TIMEOUT: u64 = 30;

/// Bytes of combined output kept for the model — the *tail*, because when a
/// build fails the reason is at the end.
pub const COMMAND_OUTPUT_CAP: usize = 8 * 1024;

/// Keep the last `cap` bytes of `text` on a char boundary, reporting whether
/// anything was dropped.
fn cap_tail(text: &str, cap: usize) -> (bool, &str) {
    if text.len() <= cap {
        return (false, text);
    }
    let mut start = text.len() - cap;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    (true, &text[start..])
}

/// `sh -c` in the workspace root, waiting up to `timeout_secs` for it. The
/// result always states the exit status first, so a capped or empty body can
/// never be mistaken for success.
async fn run_foreground(root: &Path, command: &str, timeout_secs: u64) -> String {
    let child = match tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(child) => child,
        Err(e) => return err(format!("cannot run {command:?}: {e}")),
    };
    let secs = timeout_secs.max(1);
    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(secs),
        child.wait_with_output(),
    )
    .await
    {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => return err(format!("{command:?} failed to run: {e}")),
        Err(_) => {
            // The cancelled future dropped the child, and `kill_on_drop` did
            // the killing; the pipes went with it, so nothing was captured.
            // Saying so beats showing a truncated body with no explanation.
            return err(format!(
                "timed out after {secs}s and was killed, so no output was captured \
                 — use background_start for anything this slow"
            ));
        }
    };
    let body = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let body = body.trim();
    let (dropped, tail) = cap_tail(body, COMMAND_OUTPUT_CAP);
    let status = match output.status.code() {
        Some(code) => format!("exit {code}"),
        None => "killed by signal".to_string(),
    };
    let mut result = format!("$ {command}\n{status}");
    if dropped {
        result.push_str(&format!(
            "\n(output capped to the last {} bytes of {})",
            COMMAND_OUTPUT_CAP,
            body.len()
        ));
    }
    if !tail.is_empty() {
        result.push('\n');
        result.push_str(tail);
    }
    result
}

pub async fn execute_tool_call(rt: &TaskRuntime, root: &Path, call: &ToolCall) -> String {
    execute_tool_call_timed(rt, root, call, DEFAULT_COMMAND_TIMEOUT).await
}

/// [`execute_tool_call`] with the caller's configured foreground timeout.
pub async fn execute_tool_call_timed(
    rt: &TaskRuntime,
    root: &Path,
    call: &ToolCall,
    command_timeout: u64,
) -> String {
    execute_tool_call_plan(rt, root, call, command_timeout, None).await
}

/// The dispatcher. `plan` is the chat's visible todo list: only the loop has
/// one, so `update_plan` outside it is an error rather than a silent no-op.
async fn execute_tool_call_plan(
    rt: &TaskRuntime,
    root: &Path,
    call: &ToolCall,
    command_timeout: u64,
    plan: Option<&PlanHandle>,
) -> String {
    let args = call.arguments_object();
    match call.name.as_str() {
        "update_plan" => match plan {
            Some(handle) => {
                let items = args
                    .get("items")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                apply_plan(handle, &items)
            }
            None => err("update_plan is only available in the chat loop"),
        },
        "run_command" => match arg_str(&args, "command") {
            Some(command) if !command.trim().is_empty() => {
                run_foreground(root, command, command_timeout).await
            }
            _ => "error: run_command needs a non-empty string \"command\"".to_string(),
        },
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
    /// Where writes get snapshotted before they land (I2-01 checkpoints).
    pub checkpoints: Arc<CheckpointStore>,
    /// Which turn's group this loop records into.
    pub turn: usize,
    /// Wall-clock budget for `run_command` (`agent_command_timeout`).
    pub command_timeout: u64,
    /// The chat pane's todo list, written by `update_plan`.
    pub plan: PlanHandle,
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

    /// Save what a file looks like right before an edit-class call changes
    /// it. Returns a note when the snapshot was impossible, so the model and
    /// the transcript are never left believing a change is undoable.
    fn snapshot_before(&self, root: &Path, call: &ToolCall) -> Option<String> {
        if tool_class(&call.name) != ToolClass::Edit {
            return None;
        }
        let args = call.arguments_object();
        let raw = arg_str(&args, "path")?;
        let Ok((full, display)) = workspace_path(root, raw) else {
            return None;
        };
        let prior = match std::fs::read(&full) {
            Ok(bytes) if bytes.len() > CHECKPOINT_MAX_BYTES => {
                return Some(format!(
                    "note: {display} is larger than {} MiB, so /rewind cannot undo this change.",
                    CHECKPOINT_MAX_BYTES / (1024 * 1024)
                ));
            }
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            // A directory or an unreadable file: there is no byte state to
            // put back, and guessing would risk deleting user data.
            Err(_) => return None,
        };
        self.checkpoints.record(
            self.turn,
            Checkpoint {
                full,
                display,
                prior,
            },
        );
        None
    }
}

/// Run an approved call, checkpointing the target first.
async fn run_and_checkpoint(
    rt: &TaskRuntime,
    root: &Path,
    call: &ToolCall,
    ctx: &ApprovalCtx,
) -> String {
    let note = ctx.snapshot_before(root, call);
    let mut result =
        execute_tool_call_plan(rt, root, call, ctx.command_timeout, Some(&ctx.plan)).await;
    if let Some(note) = note {
        if !result.starts_with("error:") {
            result.push('\n');
            result.push_str(&note);
        }
    }
    result
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
        Permission::Allow => run_and_checkpoint(rt, root, call, ctx).await,
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
                Ok(ApprovalAnswer::Approved) => run_and_checkpoint(rt, root, call, ctx).await,
                Ok(ApprovalAnswer::ApprovedForSession) => {
                    ctx.grant(class);
                    run_and_checkpoint(rt, root, call, ctx).await
                }
                // A dropped responder means the prompt vanished with the app.
                Ok(ApprovalAnswer::Denied) | Err(_) => DENIED_RESULT.to_string(),
            }
        }
    }
}

// ── Checkpoints (I2-01) ───────────────────────────────────────────────
// Session-scoped undo for what the agent wrote: the bytes a file had before
// the change, grouped per chat turn. Deliberately not git plumbing — no
// commits, no stash, no branch touched; quitting forgets everything and the
// user's own `git` workflow stays the durable history.

/// Files above this size are not snapshotted (the change still happens, but
/// `/rewind` says it cannot undo that one).
pub const CHECKPOINT_MAX_BYTES: usize = 4 * 1024 * 1024;

/// One file's state before the agent touched it. `prior == None` means the
/// file did not exist, so undoing deletes it.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub full: PathBuf,
    pub display: String,
    pub prior: Option<Vec<u8>>,
}

/// What a rewind actually did, for the transcript line and the toast.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RewindReport {
    /// Workspace-relative paths put back or deleted.
    pub files: Vec<String>,
    /// Of those, the ones the agent had created (deleted by the rewind).
    pub removed: usize,
    /// Snapshots that could not be restored (unreadable, vanished parent).
    pub failed: Vec<String>,
    /// Turns actually undone (groups holding at least one change).
    pub turns: usize,
    /// Absolute paths put back or deleted, for callers that must refresh
    /// their own view of a file (the editor buffer).
    pub paths: Vec<PathBuf>,
}

#[derive(Default)]
pub struct CheckpointStore {
    groups: std::sync::Mutex<Vec<Vec<Checkpoint>>>,
}

impl CheckpointStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Vec<Checkpoint>>> {
        self.groups
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Open a group for one chat turn; the index goes into `ApprovalCtx`.
    pub fn begin_turn(&self) -> usize {
        let mut groups = self.lock();
        groups.push(Vec::new());
        groups.len() - 1
    }

    fn record(&self, turn: usize, checkpoint: Checkpoint) {
        if let Some(group) = self.lock().get_mut(turn) {
            group.push(checkpoint);
        }
    }

    /// How many turns actually changed files (turns that wrote nothing are
    /// not steps to walk back).
    pub fn turns(&self) -> usize {
        self.lock().iter().filter(|group| !group.is_empty()).count()
    }

    /// Undo the last `back` turns that changed files, newest first. Turns
    /// with no writes are skipped rather than counted.
    pub fn rewind(&self, back: usize) -> RewindReport {
        let mut report = RewindReport::default();
        let mut undone = 0;
        while undone < back {
            let group = {
                let mut groups = self.lock();
                match groups.last_mut() {
                    Some(group) => {
                        let taken = std::mem::take(group);
                        groups.pop();
                        taken
                    }
                    None => break,
                }
            };
            if group.is_empty() {
                continue;
            }
            // Reverse order so an earlier snapshot of the same file wins.
            for checkpoint in group.iter().rev() {
                restore_one(checkpoint, &mut report);
            }
            undone += 1;
            report.turns += 1;
        }
        report
    }
}

fn restore_one(checkpoint: &Checkpoint, report: &mut RewindReport) {
    match &checkpoint.prior {
        Some(bytes) => {
            if let Some(parent) = checkpoint.full.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::write(&checkpoint.full, bytes) {
                Ok(()) => {
                    report.files.push(checkpoint.display.clone());
                    report.paths.push(checkpoint.full.clone());
                }
                Err(_) => report.failed.push(checkpoint.display.clone()),
            }
        }
        // The agent created it: undo means deleting — and only ever a file,
        // never whatever else may occupy that path now.
        None => match std::fs::remove_file(&checkpoint.full) {
            Ok(()) => {
                report.files.push(checkpoint.display.clone());
                report.paths.push(checkpoint.full.clone());
                report.removed += 1;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => report.failed.push(checkpoint.display.clone()),
        },
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

// ── Plan / TODO visibility (I2-03) ────────────────────────────────────
// The model's todo list. It is the only tool here that changes no files and
// runs nothing: its whole purpose is to show the user what the agent thinks
// it is doing. Weak models may ignore it entirely — that costs the strip,
// never the work.

/// How long the list is allowed to be. Longer plans are truncated rather than
/// refused: the user is better off with the first steps than with nothing.
pub const PLAN_MAX_ITEMS: usize = 12;

/// Rows shown before `/plan` is needed to see the rest.
pub const PLAN_COMPACT_ITEMS: usize = 6;

/// Longest plan line worth rendering (the rest is an essay, not a step).
const PLAN_TEXT_WIDTH: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanStatus {
    Pending,
    InProgress,
    Done,
}

impl PlanStatus {
    /// Transcript/strip glyph.
    pub fn glyph(self) -> &'static str {
        match self {
            Self::Pending => "·",
            Self::InProgress => "▶",
            Self::Done => "✓",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanItem {
    pub text: String,
    pub status: PlanStatus,
}

/// Shared between `App` (renders it) and the tool loop (writes it), so an
/// update lands on the next frame rather than at the end of the turn.
pub type PlanHandle = Arc<std::sync::Mutex<Vec<PlanItem>>>;

pub fn new_plan_handle() -> PlanHandle {
    Arc::new(std::sync::Mutex::new(Vec::new()))
}

/// Read the current plan (the renderer and `/plan` both need a snapshot).
pub fn plan_items(plan: &PlanHandle) -> Vec<PlanItem> {
    plan.lock().map(|plan| plan.clone()).unwrap_or_default()
}

/// Statuses arrive in many spellings from models (`in-progress`, `doing`,
/// `complete`); anything unrecognized stays pending, which is the safest thing
/// to show — nothing claims to be finished that is not.
fn parse_status(raw: &str) -> PlanStatus {
    let key = raw.trim().to_lowercase().replace([' ', '-', '_'], "");
    match key.as_str() {
        "done" | "complete" | "completed" | "finished" | "closed" => PlanStatus::Done,
        "inprogress" | "doing" | "active" | "current" | "wip" | "started" => PlanStatus::InProgress,
        _ => PlanStatus::Pending,
    }
}

/// Read one entry of the `items` array: an object with a text-ish key, or a
/// bare string carrying an optional markdown checkbox.
fn parse_plan_entry(entry: &serde_json::Value) -> Option<PlanItem> {
    let (raw, status) = match entry {
        serde_json::Value::String(s) => {
            let s = s.trim();
            let (marker, rest) = if let Some(tail) = s
                .strip_prefix("[ ]")
                .or_else(|| s.strip_prefix("[x]"))
                .or_else(|| s.strip_prefix("[X]"))
            {
                let done = !s.starts_with("[ ]");
                (Some(done), tail.trim())
            } else {
                (None, s)
            };
            let status = match marker {
                Some(true) => PlanStatus::Done,
                _ => PlanStatus::Pending,
            };
            (rest, status)
        }
        serde_json::Value::Object(map) => {
            let raw = ["text", "content", "title", "step", "description", "item"]
                .iter()
                .find_map(|key| map.get(*key).and_then(|v| v.as_str()))
                .unwrap_or("")
                .trim();
            let status = ["status", "state"]
                .iter()
                .find_map(|key| map.get(*key).and_then(|v| v.as_str()))
                .map(parse_status)
                .unwrap_or(PlanStatus::Pending);
            (raw, status)
        }
        _ => return None,
    };
    if raw.is_empty() {
        return None;
    }
    Some(PlanItem {
        text: truncate_one_line(raw, PLAN_TEXT_WIDTH),
        status,
    })
}

/// Parse an `items` argument into a plan. The Err is a bare reason (the caller
/// words it for the model). An empty list is a success — it clears the strip —
/// but a list whose entries are all unusable is an error, because otherwise
/// the model would be told "plan updated" about a plan nobody can read.
pub fn parse_plan(value: &serde_json::Value) -> Result<Vec<PlanItem>, String> {
    let decoded;
    let value = match value {
        // Models routinely pass the array as a JSON-encoded string.
        serde_json::Value::String(s) => match serde_json::from_str::<serde_json::Value>(s) {
            Ok(parsed @ serde_json::Value::Array(_)) => {
                decoded = parsed;
                &decoded
            }
            _ => return Err("needs an \"items\" array of {text, status} objects".into()),
        },
        other => other,
    };
    let Some(entries) = value.as_array() else {
        return Err("needs an \"items\" array of {text, status} objects".into());
    };
    let items: Vec<PlanItem> = entries.iter().filter_map(parse_plan_entry).collect();
    if items.is_empty() && !entries.is_empty() {
        return Err("no usable items — each one needs a string \"text\"".into());
    }
    Ok(items)
}

/// Post a plan, answering with what the model should believe about it.
pub fn apply_plan(plan: &PlanHandle, items: &serde_json::Value) -> String {
    let parsed = match parse_plan(items) {
        Ok(parsed) => parsed,
        Err(why) => return err(format!("update_plan {why}")),
    };
    if parsed.is_empty() {
        if let Ok(mut guard) = plan.lock() {
            guard.clear();
        }
        return "plan cleared".to_string();
    }
    let dropped = parsed.len().saturating_sub(PLAN_MAX_ITEMS);
    let kept: Vec<PlanItem> = parsed.into_iter().take(PLAN_MAX_ITEMS).collect();
    let done = kept.iter().filter(|i| i.status == PlanStatus::Done).count();
    let running = kept
        .iter()
        .filter(|i| i.status == PlanStatus::InProgress)
        .count();
    if let Ok(mut guard) = plan.lock() {
        *guard = kept.clone();
    }
    let mut answer = format!("plan updated: {} step(s), {done} done", kept.len());
    if running > 0 {
        answer.push_str(&format!(", {running} in progress"));
    }
    if dropped > 0 {
        answer.push_str(&format!(
            " — {dropped} more were dropped; keep a plan under {PLAN_MAX_ITEMS} items"
        ));
    }
    answer
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
        assert_eq!(tool_class("run_command"), ToolClass::Shell);
        // The todo list touches no files, so it must never cost an approval.
        assert_eq!(tool_class("update_plan"), ToolClass::ReadOnly);
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
        let checkpoints = Arc::new(CheckpointStore::new());
        let turn = checkpoints.begin_turn();
        Harness {
            ctx: ApprovalCtx {
                mode,
                grants: Arc::new(std::sync::Mutex::new(Vec::new())),
                prompts: tx,
                checkpoints,
                turn,
                command_timeout: DEFAULT_COMMAND_TIMEOUT,
                plan: new_plan_handle(),
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

    fn cmd_call(command: &str) -> ToolCall {
        call("run_command", serde_json::json!({"command": command}))
    }

    async fn timed(root: &Path, tool: ToolCall, seconds: u64) -> String {
        execute_tool_call_timed(&new_task_runtime(), root, &tool, seconds).await
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
    // ── Checkpoints (I2-01) ──────────────────────────────────────────────

    #[tokio::test]
    async fn an_approved_write_is_snapshotted_and_rewinds_to_absent() {
        let root = temp_root("cp-created");
        let h = harness(ApprovalMode::Ask);
        let mut prompts = h.prompts;
        let result = gated(
            new_task_runtime(),
            root.clone(),
            write_call("new.rs", "fn n() {}\n"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Approved,
        )
        .await;
        assert!(result.starts_with("created new.rs"), "{result}");
        assert!(root.join("new.rs").exists());
        assert_eq!(h.ctx.checkpoints.turns(), 1);

        let report = h.ctx.checkpoints.rewind(1);
        assert_eq!(report.turns, 1);
        assert_eq!(report.files, vec!["new.rs".to_string()]);
        assert_eq!(report.removed, 1);
        assert!(
            !root.join("new.rs").exists(),
            "a rewind deletes what the agent created"
        );
        assert_eq!(h.ctx.checkpoints.turns(), 0, "a rewound group is gone");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn a_rewritten_file_comes_back_byte_for_byte() {
        let root = temp_root("cp-modified");
        std::fs::write(root.join("keep.txt"), "original\né\n").unwrap();
        let h = harness(ApprovalMode::AllAllow);
        let mut prompts = h.prompts;
        let result = gated(
            new_task_runtime(),
            root.clone(),
            write_call("keep.txt", "overwritten\n"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
            // AllAllow: no prompt is raised, so `answer` is never used.
        )
        .await;
        assert!(result.starts_with("updated keep.txt"), "{result}");
        assert_eq!(
            std::fs::read_to_string(root.join("keep.txt")).unwrap(),
            "overwritten\n"
        );
        h.ctx.checkpoints.rewind(1);
        assert_eq!(
            std::fs::read_to_string(root.join("keep.txt")).unwrap(),
            "original\né\n"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn rewind_steps_back_one_turn_at_a_time_and_keeps_the_oldest_state() {
        let root = temp_root("cp-turns");
        let store = Arc::new(CheckpointStore::new());
        let rt = new_task_runtime();
        for turn in 0..3 {
            let turn_group = {
                let _ = &store;
                store.begin_turn()
            };
            let ctx = ApprovalCtx {
                mode: ApprovalMode::AllAllow,
                grants: Arc::new(std::sync::Mutex::new(Vec::new())),
                prompts: mpsc::unbounded_channel().0,
                checkpoints: store.clone(),
                turn: turn_group,
                command_timeout: DEFAULT_COMMAND_TIMEOUT,
                plan: new_plan_handle(),
            };
            let content = format!("written in turn {turn}\n");
            let result =
                execute_tool_call_approved(&rt, &root, &write_call("loop.txt", &content), &ctx)
                    .await;
            assert!(
                result.starts_with("created loop.txt") || result.starts_with("updated loop.txt"),
                "{result}"
            );
        }
        assert_eq!(
            std::fs::read_to_string(root.join("loop.txt")).unwrap(),
            "written in turn 2\n"
        );
        assert_eq!(store.turns(), 3);

        // One step back: the state at the end of turn 1, not of turn 0 —
        // the snapshot taken *before* turn 2 was turn 1's output.
        store.rewind(1);
        assert_eq!(
            std::fs::read_to_string(root.join("loop.txt")).unwrap(),
            "written in turn 1\n"
        );
        assert_eq!(store.turns(), 2);
        store.rewind(1);
        assert_eq!(
            std::fs::read_to_string(root.join("loop.txt")).unwrap(),
            "written in turn 0\n"
        );
        store.rewind(1);
        assert!(
            !root.join("loop.txt").exists(),
            "the last step undoes the creation itself"
        );
        assert_eq!(store.turns(), 0);
        // Rewinding with nothing left is a no-op, not a panic.
        let empty = store.rewind(5);
        assert_eq!(empty.turns, 0);
        assert!(empty.files.is_empty());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn reads_never_checkpoint_and_denied_calls_leave_nothing_behind() {
        let root = temp_root("cp-noise");
        std::fs::write(root.join("r.txt"), "x\n").unwrap();
        let h = harness(ApprovalMode::Ask);
        let mut prompts = h.prompts;
        let read = gated(
            new_task_runtime(),
            root.clone(),
            call("read_file", serde_json::json!({"path": "r.txt"})),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
        )
        .await;
        assert_eq!(read, "1\tx");
        assert_eq!(h.ctx.checkpoints.turns(), 0, "a read has nothing to undo");

        gated(
            new_task_runtime(),
            root.clone(),
            write_call("never.txt", "y\n"),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
        )
        .await;
        assert_eq!(
            h.ctx.checkpoints.turns(),
            0,
            "a denied call must not record a snapshot either"
        );
        assert!(!root.join("never.txt").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn an_oversized_target_says_rewind_cannot_undo_it() {
        let root = temp_root("cp-big");
        // One byte over the cap, written by hand so the test does not need a
        // 4 MiB write_file argument through JSON.
        let big = root.join("big.bin");
        std::fs::write(&big, vec![b'a'; CHECKPOINT_MAX_BYTES + 1]).unwrap();
        let h = harness(ApprovalMode::AllAllow);
        let mut prompts = h.prompts;
        let result = gated(
            new_task_runtime(),
            root.clone(),
            call(
                "edit_file",
                serde_json::json!({"path": "big.bin", "old": "aaa", "new": "bbb", "all": true}),
            ),
            h.ctx.clone(),
            &mut prompts,
            ApprovalAnswer::Denied,
        )
        .await;
        assert!(result.starts_with("edited big.bin"), "{result}");
        assert!(result.contains("/rewind cannot undo"), "{result}");
        assert_eq!(h.ctx.checkpoints.turns(), 0);
        std::fs::remove_dir_all(&root).unwrap();
    }

    // ── run_command (I2-02) ────────────────────────────────────────────

    #[tokio::test]
    async fn run_command_reports_status_and_output_for_success_and_failure() {
        let root = temp_root("cmd-status");
        let ok = timed(&root, cmd_call("echo hello"), 10).await;
        assert_eq!(ok, "$ echo hello\nexit 0\nhello");

        // A failing command is not an `error:` result — the shell ran it and
        // the model needs the real exit code, not our verdict on it.
        let bad = timed(&root, cmd_call("echo oops >&2; exit 3"), 10).await;
        assert!(bad.starts_with("$ echo oops >&2; exit 3\nexit 3"), "{bad}");
        assert!(bad.ends_with("oops"), "{bad}");
        assert!(!bad.starts_with("error:"));

        let empty = timed(&root, cmd_call("true"), 10).await;
        assert_eq!(empty, "$ true\nexit 0");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn run_command_runs_in_the_workspace_root() {
        let root = temp_root("cmd-cwd");
        let result = timed(&root, cmd_call("printf line > made.txt"), 10).await;
        assert!(result.ends_with("exit 0"), "{result}");
        // The write landed in the workspace, not wherever xencode was started.
        assert_eq!(
            std::fs::read_to_string(root.join("made.txt")).unwrap(),
            "line"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn run_command_keeps_the_tail_of_oversized_output() {
        let root = temp_root("cmd-cap");
        let result = timed(&root, cmd_call("seq 1 20000"), 20).await;
        assert!(result.contains("output capped"), "{result}");
        assert!(result.ends_with("\n20000"), "{}", tail(&result, 40));
        assert!(
            !result.contains("\n1\n2\n"),
            "the head must be dropped, not the tail"
        );
        assert!(result.len() < COMMAND_OUTPUT_CAP + 256, "{}", result.len());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn run_command_kills_a_command_that_overruns_its_budget() {
        let root = temp_root("cmd-slow");
        let result = timed(&root, cmd_call("sleep 5; echo never"), 1).await;
        assert!(result.starts_with("error: timed out after 1s"), "{result}");
        assert!(result.contains("background_start"), "{result}");
        assert!(!result.contains("never"), "{result}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn run_command_needs_a_command_and_prompts_as_shell_with_the_line() {
        let root = temp_root("cmd-gate");
        let rt = new_task_runtime();
        let missing = execute_tool_call(&rt, &root, &cmd_call("   ")).await;
        assert_eq!(
            missing,
            "error: run_command needs a non-empty string \"command\""
        );

        // Ask mode: the prompt shows the literal command line and nothing
        // else — no diff, because there is no proposed file change to show.
        let mut h = harness(ApprovalMode::Ask);
        assert_eq!(tool_class("run_command"), ToolClass::Shell);
        h.ctx.command_timeout = 10;
        let pending = cmd_call("echo gated");
        let running = {
            let (rt, root, ctx) = (rt.clone(), root.clone(), h.ctx.clone());
            tokio::spawn(
                async move { execute_tool_call_approved(&rt, &root, &pending, &ctx).await },
            )
        };
        let (request, responder) = h.prompts.recv().await.expect("shell prompts in ask mode");
        assert_eq!(request.class, ToolClass::Shell);
        assert_eq!(request.class_label(), "shell command");
        assert_eq!(request.summary, "run_command echo gated");
        assert_eq!(request.preview, "command: sh -c \"echo gated\"");
        responder.send(ApprovalAnswer::Approved).unwrap();
        assert_eq!(running.await.unwrap(), "$ echo gated\nexit 0\ngated");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Last `n` chars, for assertion messages that must not dump 8 KiB.
    fn tail(text: &str, n: usize) -> String {
        let mut start = text.len().saturating_sub(n);
        while !text.is_char_boundary(start) {
            start += 1;
        }
        text[start..].to_string()
    }

    // ── update_plan (I2-03) ──────────────────────────────────────────

    fn plan_call(items: serde_json::Value) -> ToolCall {
        call("update_plan", serde_json::json!({ "items": items }))
    }

    #[tokio::test]
    async fn update_plan_posts_the_list_the_strip_renders() {
        let root = temp_root("plan-post");
        let h = harness(ApprovalMode::Ask);
        let answer = execute_tool_call_approved(
            &new_task_runtime(),
            &root,
            &plan_call(serde_json::json!([
                {"text": "read the failing test", "status": "done"},
                {"text": "fix the off-by-one", "status": "in_progress"},
                {"text": "run cargo test"},
            ])),
            &h.ctx,
        )
        .await;
        assert_eq!(
            answer, "plan updated: 3 step(s), 1 done, 1 in progress",
            "the model must hear what the user will see"
        );
        let items = plan_items(&h.ctx.plan);
        assert_eq!(
            items
                .iter()
                .map(|i| (i.text.as_str(), i.status))
                .collect::<Vec<_>>(),
            vec![
                ("read the failing test", PlanStatus::Done),
                ("fix the off-by-one", PlanStatus::InProgress),
                ("run cargo test", PlanStatus::Pending),
            ]
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn update_plan_accepts_what_models_actually_write() {
        let plan = new_plan_handle();
        // Bare strings, markdown checkboxes…
        let first = apply_plan(&plan, &serde_json::json!(["[x] recon", "[ ] fix"]));
        assert!(
            first.starts_with("plan updated: 2 step(s), 1 done"),
            "{first}"
        );
        assert_eq!(plan_items(&plan)[1].text, "fix");
        // …invented key names and statuses…
        apply_plan(
            &plan,
            &serde_json::json!([{"content": "step one", "state": "doing"}]),
        );
        assert_eq!(plan_items(&plan)[0].status, PlanStatus::InProgress);
        // …and a JSON array smuggled through a string.
        let encoded = serde_json::Value::String(
            serde_json::to_string(&serde_json::json!([{"text": "escaped"}])).unwrap(),
        );
        apply_plan(&plan, &encoded);
        assert_eq!(plan_items(&plan)[0].text, "escaped");
    }

    #[test]
    fn a_rejected_plan_update_leaves_the_visible_plan_alone() {
        let plan = new_plan_handle();
        apply_plan(&plan, &serde_json::json!([{"text": "still here"}]));

        let junk = apply_plan(&plan, &serde_json::json!([1, 2, true]));
        assert!(junk.starts_with("error: update_plan"), "{junk}");
        assert!(junk.contains("no usable items"), "{junk}");
        assert_eq!(
            plan_items(&plan)[0].text,
            "still here",
            "an unreadable call must not wipe the list"
        );

        let missing = apply_plan(&plan, &serde_json::Value::Null);
        assert!(missing.contains("needs an \"items\" array"), "{missing}");

        // An explicit empty list is the model's way of finishing up.
        assert_eq!(apply_plan(&plan, &serde_json::json!([])), "plan cleared");
        assert!(plan_items(&plan).is_empty());
    }

    #[test]
    fn an_overlong_plan_keeps_the_first_steps_and_admits_trimming() {
        let plan = new_plan_handle();
        let items: Vec<serde_json::Value> = (0..15)
            .map(|i| serde_json::json!({"text": format!("step {i}")}))
            .collect();
        let answer = apply_plan(&plan, &serde_json::Value::Array(items));
        assert_eq!(plan_items(&plan).len(), PLAN_MAX_ITEMS);
        assert!(answer.contains("12 step(s)"), "{answer}");
        assert!(answer.contains("3 more were dropped"), "{answer}");
    }

    #[tokio::test]
    async fn a_plan_never_costs_an_approval_even_in_ask_mode() {
        let root = temp_root("plan-free");
        let h = harness(ApprovalMode::Ask);
        let mut prompts = h.prompts;
        let answer = execute_tool_call_approved(
            &new_task_runtime(),
            &root,
            &plan_call(serde_json::json!([{"text": "x"}])),
            &h.ctx,
        )
        .await;
        assert_eq!(answer, "plan updated: 1 step(s), 0 done");
        assert!(
            prompts.try_recv().is_err(),
            "a checklist is not an action to approve"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[tokio::test]
    async fn update_plan_outside_a_chat_loop_says_so() {
        let root = temp_root("plan-alone");
        let answer = execute_tool_call(
            &new_task_runtime(),
            &root,
            &plan_call(serde_json::json!([])),
        )
        .await;
        assert_eq!(
            answer,
            "error: update_plan is only available in the chat loop"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// I2-04: the panel's step rows say how a call ended, and "the user said
    /// no", "the policy said no" and "it ran and broke" are three different
    /// answers even though all three start with `error:`.
    #[test]
    fn call_outcomes_separate_a_refusal_from_a_failure() {
        assert_eq!(call_outcome("edited src/lib.rs"), CallOutcome::Finished);
        assert_eq!(call_outcome(DENIED_RESULT), CallOutcome::Denied);
        assert_eq!(call_outcome(FORBIDDEN_RESULT), CallOutcome::Refused);
        assert_eq!(call_outcome("error: file not found"), CallOutcome::Failed);
        assert_eq!(CallOutcome::Denied.label(), "denied");
        assert_eq!(CallOutcome::Refused.label(), "refused");
        assert_eq!(CallOutcome::Failed.label(), "failed");
    }
}
